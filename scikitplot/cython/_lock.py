# scikitplot/cython/_lock.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Cross-platform build lock.

This uses an atomic *exclusive* directory-creation pattern (<path>.lock) to
avoid concurrent compilation of the same cache key (parallel tests,
multi-process builds).  Acquisition uses ``mkdir(exist_ok=False)`` so that a
directory that already exists raises ``FileExistsError`` and the caller waits;
an existing lock is never accepted as a successful acquisition.  Each acquired
lock records owner metadata (``owner.json``) and is released only by the owner
whose token matches, so a stale-lock takeover cannot delete a live owner's lock.

Stale lock recovery
-------------------
If the owning process is killed hard (SIGKILL, OOM, power loss), the lock
directory is never removed by the ``finally`` block. To prevent permanent
deadlock, a lock directory whose *mtime* exceeds the effective stale threshold
is reclaimed before retrying. The threshold is ``stale_after_s`` when supplied;
otherwise it is the greater of ``timeout_s`` and
``_DEFAULT_STALE_AFTER_S``. This keeps crash recovery independent from a
caller's willingness to wait and prevents a non-blocking probe from deleting a
fresh, live lock.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = [
    "build_lock",
]

# Name of the ownership-metadata file written inside the lock directory.
# It records which process/host currently holds the lock so that release is
# token-matched (a waiter that recovers a stale lock never deletes a lock that
# a live owner has since re-created).
_OWNER_FILE = "owner.json"

#: Default grace period (seconds) after which a lock directory is considered
#: stale (left by a crashed process).  Decoupled from ``timeout_s`` so that a
#: non-blocking probe (``timeout_s=0``) never reclaims a live lock
#: (CYTHON-TEST-001).  Chosen well beyond any reasonable single build.
_DEFAULT_STALE_AFTER_S = 900.0  # 15 minutes


def _write_owner(lock_dir: Path, token: str) -> None:
    """Best-effort write of ownership metadata into an acquired lock directory.

    Failure to write the owner file must never turn a successfully acquired
    lock into a failure, so write errors are swallowed; token-matched release
    simply falls back to unconditional removal when the file is unreadable.
    """
    payload = {
        "token": token,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "start_monotonic": time.monotonic(),
        "start_wall": time.time(),
    }
    try:  # ruff:ignore[suppressible-exception]
        (lock_dir / _OWNER_FILE).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def _owned_by(lock_dir: Path, token: str) -> bool:
    """Return True when the lock's owner file matches ``token``.

    Returns True when the owner file is absent or unreadable so that a normal
    clean release still removes the directory; only a *different* recorded
    token blocks removal.
    """
    try:
        data = json.loads((lock_dir / _OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return data.get("token") == token


@contextmanager
def build_lock(  # ruff:ignore[too-many-branches]
    lock_dir: Path,
    *,
    timeout_s: float = 60.0,
    poll_s: float = 0.05,
    stale_after_s: float | None = None,
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
    stale_after_s : float | None, default=None
        Age in seconds after which an existing lock may be reclaimed as stale.
        When ``None``, the effective threshold is
        ``max(timeout_s, _DEFAULT_STALE_AFTER_S)``.

    Returns
    -------
    Iterator[None]
        Context manager that yields once the lock is held.

    Raises
    ------
    TimeoutError
        If the lock cannot be acquired within ``timeout_s`` seconds.
    ValueError
        If ``timeout_s < 0``, ``poll_s <= 0``, or an explicit
        ``stale_after_s <= 0``.

    Notes
    -----
    **Stale lock recovery**: if a lock directory exists but its ``mtime`` is
    older than the effective stale threshold, it is treated as stale (left by
    a killed process) and removed before the next acquisition attempt. The
    threshold is ``stale_after_s`` when supplied; otherwise it is
    ``max(timeout_s, _DEFAULT_STALE_AFTER_S)``.

    **Clean release**: the lock directory is always removed in the ``finally``
    block, so normal exceptions inside the ``with`` body release the lock
    correctly.
    """
    if timeout_s < 0:
        raise ValueError(f"timeout_s must be >= 0, got {timeout_s!r}")
    if poll_s <= 0:
        raise ValueError(f"poll_s must be > 0, got {poll_s!r}")
    if stale_after_s is not None and stale_after_s <= 0:
        raise ValueError(f"stale_after_s must be > 0 or None, got {stale_after_s!r}")

    # Staleness threshold is DECOUPLED from timeout_s (CYTHON-TEST-001).  Using
    # timeout_s as the staleness threshold was a correctness bug: with
    # timeout_s=0 (a non-blocking probe) EVERY live lock (age > 0) was treated
    # as stale and destroyed, breaking interprocess exclusivity.  A lock is
    # stale only after a fixed grace period well beyond any reasonable build.
    if stale_after_s is None:
        # Default: max(timeout, floor) so a long timeout still recovers crashes,
        # but a zero/short timeout never falsely reclaims a live lock.
        effective_stale_after = max(timeout_s, _DEFAULT_STALE_AFTER_S)
    else:
        effective_stale_after = stale_after_s

    deadline = time.monotonic() + timeout_s
    lock_dir = lock_dir.resolve()
    token = uuid.uuid4().hex

    while True:
        try:
            # Exclusive acquisition: exist_ok=False means an already-held lock
            # raises FileExistsError instead of being silently accepted.  This
            # is the correctness fix for CYTHON-CON-001 — mkdir(exist_ok=True)
            # previously let concurrent callers "acquire" the same lock and
            # rendered the stale/timeout branch below unreachable dead code.
            lock_dir.parent.mkdir(parents=True, exist_ok=True)
            lock_dir.mkdir(exist_ok=False)
            _write_owner(lock_dir, token)
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
                if lock_age > effective_stale_after:
                    # A stale lock may now contain an owner file; remove it
                    # before rmdir so recovery still succeeds on a non-empty
                    # crashed-owner directory.
                    try:  # ruff:ignore[suppressible-exception]
                        (lock_dir / _OWNER_FILE).unlink()
                    except FileNotFoundError:
                        pass
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
        # Token-matched release: only remove the directory this call created.
        # If another owner has since recreated the lock (e.g. after a stale
        # takeover), the mismatching token prevents us from deleting theirs.
        try:
            if _owned_by(lock_dir, token):
                try:  # ruff:ignore[suppressible-exception]
                    (lock_dir / _OWNER_FILE).unlink()
                except FileNotFoundError:
                    pass
                lock_dir.rmdir()
        except FileNotFoundError:
            # Already removed (e.g., by a concurrent stale-lock cleaner) or the
            # in-body code deleted it deliberately.
            pass
        except OSError:
            # Non-empty (foreign owner file present) or transient FS error:
            # leave the foreign lock intact rather than corrupting another
            # owner's state.
            pass
