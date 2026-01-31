# scikitplot/cython/_lock.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Cross-platform build lock.

This uses an atomic directory creation pattern (<path>.lock) to avoid concurrent
compilation of the same cache key (parallel tests, multi-process builds).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def build_lock(
    lock_dir: Path, *, timeout_s: float = 60.0, poll_s: float = 0.05
) -> Iterator[None]:
    """
    Acquire an exclusive build lock.

    Parameters
    ----------
    lock_dir : pathlib.Path
        Lock directory path to create (atomic).
    timeout_s : float, default=60.0
        Maximum time to wait for the lock.
    poll_s : float, default=0.05
        Poll interval while waiting.

    Returns
    -------
    Iterator[None]
        Context manager.

    Raises
    ------
    TimeoutError
        If the lock cannot be acquired within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    lock_dir = lock_dir.resolve()

    while True:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError as e:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring build lock: {lock_dir}") from e
            time.sleep(poll_s)

    try:
        yield
    finally:
        lock_dir.rmdir()
