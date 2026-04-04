# scikitplot/cython/_lock.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Cross-platform build lock.

This uses an atomic directory creation pattern (<path>.lock) to avoid concurrent
compilation of the same cache key (parallel tests, multi-process builds).

Stale lock recovery
-------------------
If the owning process is killed hard (SIGKILL, OOM, power loss), the lock
directory is never removed by the ``finally`` block.  To prevent permanent
deadlock, this implementation treats a lock directory whose *mtime* is older
than ``timeout_s`` seconds as stale and removes it before retrying.  This is
safe in practice because a lock that has been held longer than the timeout
deadline would have already caused a ``TimeoutError`` in all other waiters;
the owning process is assumed to be dead.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = [
    "build_lock",
]


@contextmanager
def build_lock(
    lock_dir: Path, *, timeout_s: float = 60.0, poll_s: float = 0.05
) -> Iterator[None]:
    """
    Acquire an exclusive build lock via atomic directory creation.

    Parameters
    ----------
    lock_dir : pathlib.Path
        Lock directory path to create atomically.
    timeout_s : float, default=60.0
        Maximum seconds to wait for the lock.  A value of ``0`` makes a single
        acquisition attempt and raises ``TimeoutError`` immediately if the lock
        is already held.
    poll_s : float, default=0.05
        Sleep interval in seconds between acquisition retries.

    Returns
    -------
    Iterator[None]
        Context manager that yields once the lock is held.

    Raises
    ------
    TimeoutError
        If the lock cannot be acquired within ``timeout_s`` seconds.
    ValueError
        If ``timeout_s < 0`` or ``poll_s <= 0``.

    Notes
    -----
    **Stale lock recovery**: if a lock directory exists but its ``mtime`` is
    older than ``timeout_s`` seconds, it is treated as stale (left by a killed
    process) and removed before the next acquisition attempt.  This prevents
    permanent deadlock after hard crashes.

    **Clean release**: the lock directory is always removed in the ``finally``
    block, so normal exceptions inside the ``with`` body release the lock
    correctly.
    """
    if timeout_s < 0:
        raise ValueError(f"timeout_s must be >= 0, got {timeout_s!r}")
    if poll_s <= 0:
        raise ValueError(f"poll_s must be > 0, got {poll_s!r}")

    deadline = time.monotonic() + timeout_s
    lock_dir = lock_dir.resolve()

    while True:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError as e:
            now = time.monotonic()

            # Stale-lock detection runs BEFORE the deadline check so that a
            # zero-timeout caller can still recover from a crashed process.
            # Use time.time() (wall clock) for mtime comparison because
            # st_mtime is a Unix epoch timestamp; time.monotonic() is relative
            # to an arbitrary epoch and must NEVER be compared against st_mtime.
            try:
                lock_age = time.time() - lock_dir.stat().st_mtime
                if lock_age > timeout_s:
                    lock_dir.rmdir()
                    # Retry immediately — do not sleep, do not check deadline.
                    continue
            except (FileNotFoundError, OSError):
                # Lock was removed by another waiter between stat() and
                # rmdir(); retry the mkdir on the next loop iteration.
                pass

            if now >= deadline:
                raise TimeoutError(f"Timed out acquiring build lock: {lock_dir}") from e

            time.sleep(poll_s)

    try:
        yield
    finally:
        try:  # noqa: SIM105
            lock_dir.rmdir()
        except FileNotFoundError:
            # Already removed (e.g., by a concurrent stale-lock cleaner).
            pass
