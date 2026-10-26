# scikitplot/_env/_core/_lock.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
One writer per environment: an inter-process lock file next to the prefix.

Notes
-----
**User notes.** Two ``create`` runs for the same prefix (two CI jobs, a
notebook and a terminal) would corrupt it; the second waits up to
``timeout`` seconds and then fails with :py:class:`~.LockBusyError`. A lock left by
a process that died on the same machine is detected and replaced.

**Developer notes.** The lock is a file created with ``O_CREAT | O_EXCL``
(atomic on local filesystems and on NFSv3+), holding the owner's host name
and pid. Liveness is checked only for the same host; a lock from another
host is never broken automatically.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from ._errors import LockBusyError

__all__ = ["PrefixLock"]


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - Windows
        import ctypes  # noqa: PLC0415

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PrefixLock:
    """
    Context manager holding the lock of one environment prefix.

    Parameters
    ----------
    prefix : str or os.PathLike
        Environment directory; the lock is ``<prefix>.scikitplot-env.lock``.
    timeout : float, default=0.0
        Seconds to wait for a busy lock.
    poll : float, default=0.2
        Seconds between attempts.

    Raises
    ------
    LockBusyError
        If another live process keeps the lock past ``timeout``.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from scikitplot._env._core._lock import PrefixLock
    >>> prefix = pathlib.Path(tempfile.mkdtemp()) / "env"
    >>> with PrefixLock(prefix) as lock:
    ...     lock.path.exists()
    True
    >>> lock.path.exists()
    False
    """

    def __init__(
        self,
        prefix: Any,
        *,
        timeout: float = 0.0,
        poll: float = 0.2,
    ) -> None:  # noqa: D107
        prefix = Path(prefix)
        self.path = prefix.with_name(prefix.name + ".scikitplot-env.lock")
        self.timeout = float(timeout)
        self.poll = float(poll)
        self._held = False
        self._token: str | None = None

    def _owner(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def acquire(self) -> None:
        """Take the lock, waiting up to ``timeout`` seconds."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        me = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "time": time.time(),
            "token": uuid.uuid4().hex,
        }
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                owner = self._owner()
                same_host = owner.get("host") == me["host"]
                if same_host and not _alive(int(owner.get("pid", 0) or 0)):
                    self._break_stale(owner, me["token"])
                    continue
                if time.monotonic() >= deadline:
                    raise LockBusyError(
                        f"environment is locked by {owner.get('host', '?')}"
                        f" pid {owner.get('pid', '?')} ({self.path})",
                        hint="wait for the other run, or remove the lock file if"
                        " that process is gone",
                    ) from None
                time.sleep(self.poll)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(me, handle)
            self._token = me["token"]
            self._held = True
            return

    def _break_stale(self, owner: dict[str, Any], mine: str) -> None:
        """
        Remove a dead owner's lock without ever removing a live one.

        Parameters
        ----------
        owner : dict
            The lock content that was judged stale.
        mine : str
            This process's token (names the side file).

        Notes
        -----
        The lock is renamed first (atomic: of several waiters only one
        succeeds) and the renamed file is checked to be the one judged
        stale; a fresh lock taken in between is linked back in place.
        """
        aside = self.path.with_name(self.path.name + f".stale-{mine}")
        try:
            os.replace(self.path, aside)
        except FileNotFoundError:
            return  # another waiter broke it first
        try:
            moved = json.loads(aside.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            moved = {}
        if moved.get("token") != owner.get("token"):
            with contextlib.suppress(FileExistsError, OSError):
                os.link(aside, self.path)  # put a live lock back (fails if retaken)
        with contextlib.suppress(FileNotFoundError):
            aside.unlink()

    def release(self) -> None:
        """Release the lock if this object holds it (and nobody broke it)."""
        if self._held:
            if self._owner().get("token") == self._token:
                with contextlib.suppress(FileNotFoundError):
                    self.path.unlink()
            self._held = False

    def __enter__(self) -> PrefixLock:  # noqa: D105, PYI034
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:  # noqa: D105
        self.release()
