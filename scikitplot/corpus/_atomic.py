# scikitplot/corpus/_atomic.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Atomic file publication primitive.

A single, well-tested way to publish a file so that concurrent writers never
observe or clobber each other's staging files, and readers never see a partial
file. Every cache/export/storage writer in :mod:`scikitplot.corpus` funnels
through here instead of rolling its own ``path + ".tmp"`` scheme (the
CORPUS-TMP-001 predictable-temp race).

Guarantees:

* **Unique staging file.** Each publish stages to a unique same-directory temp
  created with :func:`tempfile.mkstemp`, so two processes publishing the same
  target do not share (and cannot delete) each other's temp.
* **Durable.** The staging file is ``fsync``-ed before it is published, and the
  containing directory is ``fsync``-ed after (best-effort; a no-op where the
  platform cannot sync a directory).
* **Atomic.** Publication is :func:`os.replace`, which is atomic on POSIX and
  Windows for same-filesystem paths: a reader sees the old file or the new
  one, never a mix. On Windows the call itself can be *refused* while another
  process is replacing or opening the same target; it is then repeated for a
  bounded time (see :func:`_replace`), and each attempt is still atomic.
* **No orphans on failure.** If the writer or sync fails, the staging file is
  removed and the original error propagates.

Notes
-----
**Developer note:** Same-directory staging is required for atomic replace —
``os.replace`` across filesystems is not atomic and raises ``OSError``. Callers
therefore never pass a temp dir; the temp always lives beside the target.
"""

from __future__ import annotations

import errno
import logging
import os
import pathlib
import shutil
import tempfile
import time
from typing import Callable, Union

logger = logging.getLogger(__name__)

__all__ = [
    "atomic_write_bytes",
    "atomic_write_path",
]

StrPath = Union[str, "os.PathLike[str]"]


#: ``errno`` values meaning "this platform cannot sync this object", as opposed
#: to "the sync was attempted and failed". Windows cannot sync a directory
#: handle at all, and some filesystems reject ``fsync`` on a directory with
#: ``EINVAL``. Anything outside this set is a real I/O failure and is reported.
_UNSUPPORTED_SYNC_ERRNOS: frozenset[int] = frozenset(
    code
    for code in (
        getattr(errno, name, None)
        for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "ENOSYS", "EBADF")
    )
    if code is not None
)

#: Additional ``errno`` values accepted when syncing a *directory*: opening a
#: directory for reading is refused outright on Windows.
_UNSUPPORTED_DIR_SYNC_ERRNOS: frozenset[int] = _UNSUPPORTED_SYNC_ERRNOS | frozenset(
    code
    for code in (getattr(errno, name, None) for name in ("EACCES", "EPERM", "EISDIR"))
    if code is not None
)


def _describe(exc: OSError) -> str:
    """Return the symbolic ``errno`` name for ``exc``, or its numeric value."""
    return errno.errorcode.get(exc.errno, str(exc.errno))


def _sync(path: pathlib.Path, *, tolerated: frozenset[int], kind: str) -> None:
    """``fsync`` *path*, propagating a real failure and reporting a downgrade.

    Parameters
    ----------
    path : pathlib.Path
        File or directory to sync.
    tolerated : frozenset of int
        ``errno`` values that mean the platform cannot perform this sync.
    kind : str
        ``"file"`` or ``"directory"``, used in the downgrade message.

    Raises
    ------
    OSError
        If the sync was attempted and failed for any reason outside
        ``tolerated`` — the payload may not be on stable storage, and a caller
        that was told the write succeeded would be wrong.

    Notes
    -----
    **Developer.** The previous form caught every :class:`OSError` and returned,
    so ``EIO`` from a failing disk and ``EINVAL`` from a platform that cannot
    sync a directory produced identical behaviour and publication still reported
    success. Only the second is a platform limitation; it is downgraded and
    logged with its ``errno`` named, because a weaker durability guarantee that
    nobody is told about is indistinguishable from the strong one.

    Carrying the downgrade in the return value rather than the log belongs to
    the publication-sequence slice, which changes this function's contract.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError as exc:
        if exc.errno in tolerated:
            logger.warning(
                "durability downgraded: cannot open %s %s to sync (%s); "
                "the write is published but not known to be on stable storage.",
                kind,
                path,
                _describe(exc),
            )
            return
        raise
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno in tolerated:
            logger.warning(
                "durability downgraded: this platform cannot sync %s %s (%s); "
                "the write is published but not known to be on stable storage.",
                kind,
                path,
                _describe(exc),
            )
            return
        raise
    finally:
        os.close(fd)


def _fsync_file(path: pathlib.Path) -> None:
    """``fsync`` a file by path; a real I/O failure is raised, not swallowed."""
    _sync(path, tolerated=_UNSUPPORTED_SYNC_ERRNOS, kind="file")


def _fsync_dir(path: pathlib.Path) -> None:
    """``fsync`` a directory so the rename is durable, where the platform can."""
    _sync(path, tolerated=_UNSUPPORTED_DIR_SYNC_ERRNOS, kind="directory")


#: Whether a refused replace is repeated. Windows refuses ``os.replace`` with
#: ``PermissionError`` while another process holds the target (replacing it,
#: or reading it without sharing deletion); POSIX never does. A module
#: constant, so that a test takes either branch on any platform.
_RETRY_REFUSED_REPLACE: bool = os.name == "nt"

#: Pauses, in seconds, between attempts of a refused replace: 15 attempts in
#: about 1.2 s. The holder is another publisher in the middle of the same
#: call, which lasts microseconds; the last pauses are long only so that a
#: reader that opened the file briefly has let go.
_REPLACE_PAUSES: tuple[float, ...] = (
    0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064,
    0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.2,
)  # fmt: skip


def _replace(source: pathlib.Path, target: pathlib.Path) -> None:
    """
    Replace ``target`` with ``source``, repeating a refusal where one is transient.

    Parameters
    ----------
    source : pathlib.Path
        The staged file or directory.
    target : pathlib.Path
        The path to publish at.

    Raises
    ------
    PermissionError
        If the replace is refused and repeating does not apply
        (``_RETRY_REFUSED_REPLACE`` is false) or did not help within
        ``_REPLACE_PAUSES``. The last refusal is raised unchanged.
    OSError
        Any other failure, at once.

    Notes
    -----
    **User.** Several processes may publish the same target at the same time;
    one of them wins and every one of them returns normally. If a reader keeps
    the target open for longer than about a second on Windows, the publisher
    gets ``PermissionError`` and the previous file stays as it was.

    **Developer.** Measured in the Windows job of CI run 37637526602: eight
    processes publishing one target, ``PermissionError: [WinError 5] Access is
    denied`` from ``os.replace`` in ``test_contended_target_stays_consistent``.
    ``MoveFileEx`` cannot replace a file that another handle holds without
    ``FILE_SHARE_DELETE``, and a concurrent replace holds one for an instant.
    There is no handle to wait on from here, so the call is repeated; the
    pauses are a fixed table, not a guess per call, and only this one error is
    repeated. Every repetition is logged at DEBUG.
    """
    attempts = len(_REPLACE_PAUSES) + 1 if _RETRY_REFUSED_REPLACE else 1
    for attempt in range(attempts):
        try:
            os.replace(source, target)
        except PermissionError:  # noqa: PERF203 - a retry is a try in a loop
            if attempt == attempts - 1:
                raise
            logger.debug(
                "replace of %s refused (attempt %d of %d); repeating",
                target,
                attempt + 1,
                attempts,
            )
            time.sleep(_REPLACE_PAUSES[attempt])
        else:
            return


def atomic_write_path(
    target: StrPath,
    writer: Callable[[pathlib.Path], None],
    *,
    suffix: str = ".tmp",
) -> pathlib.Path:
    """Atomically publish a file produced by ``writer``.

    Creates a unique staging file beside *target*, invokes ``writer(tmp_path)``
    to populate it, ``fsync``s it, then atomically replaces *target*. On any
    error the staging file is removed and the error re-raised.

    Parameters
    ----------
    target : str or os.PathLike
        Final path to publish. Parent directories are created if needed.
    writer : callable
        ``writer(tmp_path)`` must write the complete file contents to the given
        staging path (e.g. ``lambda p: numpy.save(str(p), arr)``).
    suffix : str, optional
        Suffix for the staging file. Use one the writer expects — e.g.
        ``".npy"`` for :func:`numpy.save`, which otherwise appends it.

    Returns
    -------
    pathlib.Path
        The published *target* path.

    Raises
    ------
    Exception
        Whatever ``writer`` raises (after the staging file is cleaned up).
    """
    target = pathlib.Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=suffix
    )
    os.close(fd)  # the writer opens the path itself
    tmp_path = pathlib.Path(tmp_name)
    try:
        writer(tmp_path)
        # A writer may replace the empty staging file with a populated
        # *directory* (the artifact writer does). A directory is synced as a
        # directory: opening one the way a file is opened is refused on
        # Windows (EACCES), which ``_fsync_file`` rightly treats as a real
        # failure, so every artifact publication failed there with
        # "Permission denied: ...candidate-<id>.<random>.tmp". ``_fsync_dir``
        # knows that refusal is a platform limit and reports the downgrade.
        if tmp_path.is_dir():
            _fsync_dir(tmp_path)
        else:
            _fsync_file(tmp_path)
        _replace(tmp_path, target)
    except BaseException:
        # The staging path may be a directory: a writer is free to replace the
        # empty staging file with a populated directory, which the artifact
        # writer does. unlink() cannot remove one, so the failure was swallowed
        # here and the staging directory was left beside the target after every
        # failed publication.
        try:  # ruff: ignore[suppressible-exception]
            if tmp_path.is_dir():
                shutil.rmtree(tmp_path, ignore_errors=True)
            else:
                tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    _fsync_dir(target.parent)
    return target


def atomic_write_bytes(
    target: StrPath,
    data: bytes,
    *,
    suffix: str = ".tmp",
) -> pathlib.Path:
    """Atomically publish raw *data* bytes to *target*.

    Convenience wrapper over :func:`atomic_write_path` that writes and
    ``fsync``s the bytes itself.

    Parameters
    ----------
    target : str or os.PathLike
        Final path to publish.
    data : bytes
        Payload to write.
    suffix : str, optional
        Staging-file suffix.

    Returns
    -------
    pathlib.Path
        The published *target* path.
    """

    def _write(tmp_path: pathlib.Path) -> None:
        with open(tmp_path, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    return atomic_write_path(target, _write, suffix=suffix)
