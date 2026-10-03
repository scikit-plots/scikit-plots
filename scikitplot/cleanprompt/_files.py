"""
Two file primitives a vault needs: an atomic write, and a lock between processes.

A vault is the only place the removed values live. Two ways of losing or
corrupting it were measured in round seventeen, and each primitive here
closes one:

* **Writing in place** (open with truncation, then write) leaves a window in
  which the file is empty. Another process reading it then — ``decode``, or a
  second ``encode`` — failed with "not valid JSON", and a crash or Ctrl-C in
  the window would have destroyed the vault and every placeholder it could
  restore (``CP-069``). :func:`atomic_write` writes a sibling file, flushes it
  to disk and renames it over the target: readers see the old document or the
  new one, never a part.
* **Reading, encoding and writing without a lock** let two processes seed
  from the same vault, issue the same label to two values and each write its
  own version; one set of values was lost and a reply decoded to the wrong
  value (``CP-068``). :func:`locked` holds an exclusive lock on a sibling
  ``.lock`` file across the whole read-encode-write.

Notes
-----
**User notes.** Nothing to configure. Two runs that use the same vault now
take turns; the second prints that it is waiting, and gives up with an
explanation after :data:`LOCK_TIMEOUT` seconds. The ``.lock`` file holds no
data and is left in place on purpose (see below).

**Developer notes.**

*The lock is advisory and kernel-held* — ``fcntl.flock`` on POSIX,
``msvcrt.locking`` on Windows. The kernel releases it when the process exits
for any reason, so a crashed run can never leave a stale lock behind, which
is the failure a "lock file that exists" scheme has.

*The lock file is never deleted.* Deleting it while another process waits on
it would let a third process create a new file and lock that one, and two
holders would run at once. It is empty and ``0600``.

*Waiting is polling with a deadline.* Neither platform offers a blocking lock
with a timeout, so a non-blocking attempt is retried every
:data:`_POLL_SECONDS` until :data:`LOCK_TIMEOUT`. That bound is what turns "a
run hung behind another one" into an error a person can act on.

*Symbolic links are resolved first.* A vault path that is a link to a file
elsewhere (an encrypted volume, say) keeps pointing there: the target is
replaced, not the link.

See Also
--------
scikitplot.cleanprompt._cli : Every command that reads and writes a vault.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import time
from collections.abc import Callable, Iterator

from ._exceptions import CleanPromptError

__all__ = [
    "LOCK_TIMEOUT",
    "atomic_write",
    "locked",
]

#: Seconds a run waits for another run to finish with the same vault. Long
#: enough for a folder ``batch``, which holds the lock while it encodes.
LOCK_TIMEOUT = 600.0

#: Seconds between attempts while another run holds the lock.
_POLL_SECONDS = 0.05


def _try_lock(descriptor: int) -> bool:
    """Take an exclusive lock without waiting; return whether it was taken."""
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import msvcrt  # ruff: ignore[import-outside-top-level]

        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl  # ruff: ignore[import-outside-top-level]

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):
        return False
    return True


def _unlock(descriptor: int) -> None:
    """Release a lock taken by :func:`_try_lock`."""
    if os.name == "nt":  # pragma: no cover - exercised on Windows only
        import msvcrt  # ruff: ignore[import-outside-top-level]

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    import fcntl  # ruff: ignore[import-outside-top-level]

    fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextlib.contextmanager
def locked(
    path: str,
    timeout: float = LOCK_TIMEOUT,
    waiting: Callable[[], None] | None = None,
) -> Iterator[None]:
    """
    Hold an exclusive lock, shared by every process, for the file at ``path``.

    Parameters
    ----------
    path : str
        The file being protected (a vault). The lock is taken on
        ``<path>.lock`` beside it, after resolving symbolic links.
    timeout : float, default=LOCK_TIMEOUT
        Seconds to wait for another holder before giving up. Must be > 0.
    waiting : callable, optional
        Called once, with no arguments, if the lock is not free at once — so
        a command can say why it is pausing.

    Yields
    ------
    None
        While the lock is held.

    Raises
    ------
    ValueError
        If ``timeout`` is not a positive number.
    CleanPromptError
        If the lock is still held by another process after ``timeout``.

    Examples
    --------
    >>> import os, tempfile
    >>> target = os.path.join(tempfile.mkdtemp(), "vault.json")
    >>> with locked(target):
    ...     atomic_write(target, "{}")
    >>> with open(target) as handle:
    ...     handle.read()
    '{}'
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not timeout > 0
    ):
        raise ValueError(
            f"timeout must be a positive number of seconds, got {timeout!r}"
        )
    lock_path = os.path.realpath(path) + ".lock"
    _ensure_directory(lock_path)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + timeout
        told = False
        while not _try_lock(descriptor):
            if not told and waiting is not None:
                waiting()
                told = True
            if time.monotonic() >= deadline:
                msg = (
                    f"another cleanprompt run has held the vault lock for over "
                    f"{timeout:g}s ({lock_path}); wait for it to finish, or stop it"
                )
                raise CleanPromptError(msg)
            time.sleep(_POLL_SECONDS)
        try:
            yield
        finally:
            _unlock(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path: str, text: str, mode: int = 0o600) -> None:
    """
    Replace the file at ``path`` with ``text`` in one step.

    Parameters
    ----------
    path : str
        The file to write; created if absent. A symbolic link is followed and
        its target replaced.
    text : str
        The whole new content, written as UTF-8.
    mode : int, default=0o600
        Permission bits of the new file, set when it is created so there is no
        moment at which it is readable by others.

    Raises
    ------
    OSError
        If the directory cannot be written. The target is then unchanged.

    Notes
    -----
    **Developer notes.** The sibling file is created with ``O_EXCL`` under a
    random name, so two writers never share one; it is flushed and
    ``fsync``-ed before :func:`os.replace`, which is atomic on POSIX and on
    Windows for a target on the same volume — hence a sibling, never the
    system temporary directory. On POSIX the directory is synced too, so the
    rename itself survives a power cut.
    """
    target = os.path.realpath(path)
    _ensure_directory(target)
    directory, name = os.path.split(target)
    scratch = os.path.join(
        directory, f".{name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    descriptor = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(scratch, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(scratch)
        raise
    if hasattr(os, "O_DIRECTORY"):
        with contextlib.suppress(OSError):
            folder = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(folder)
            finally:
                os.close(folder)


def _ensure_directory(path: str) -> None:
    """Create the directory ``path`` lives in, owner-only when new."""
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, mode=0o700, exist_ok=True)
