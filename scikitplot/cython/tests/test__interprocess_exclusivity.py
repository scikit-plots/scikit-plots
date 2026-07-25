# scikitplot/cython/tests/test__interprocess_exclusivity.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-TEST-001.

The suite lacked a true interprocess exclusivity test, and writing one exposed a
real bug: the stale-lock threshold was ``timeout_s`` itself, so a non-blocking
probe (``timeout_s=0``) treated **every** live lock as stale and destroyed it —
breaking exclusivity.  Staleness is now decoupled (``stale_after_s``).  These
tests run actual child processes to prove one holder excludes another.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

# Child-process entry points must be module-level (picklable for spawn).


def _hold_lock(lock_dir_str: str, ready, hold_secs: float) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scikitplot.cython._lock import build_lock

    with build_lock(Path(lock_dir_str), timeout_s=10.0):
        ready.set()
        time.sleep(hold_secs)


def _try_acquire(lock_dir_str: str, result_q, timeout_s: float) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scikitplot.cython._lock import build_lock

    try:
        with build_lock(Path(lock_dir_str), timeout_s=timeout_s):
            result_q.put("acquired")
    except TimeoutError:
        result_q.put("timeout")
    except Exception as e:  # noqa: BLE001
        result_q.put(f"error:{type(e).__name__}")


class TestInterprocessExclusivity:
    def test_second_process_cannot_acquire_while_held(self, tmp_path: Path) -> None:
        ctx = mp.get_context("spawn")
        lock_dir = str(tmp_path / "build.lock")
        ready = ctx.Event()
        q = ctx.Queue()

        holder = ctx.Process(target=_hold_lock, args=(lock_dir, ready, 1.5))
        holder.start()
        try:
            assert ready.wait(timeout=15), "holder never acquired the lock"

            # While the holder has it, a non-blocking probe MUST time out
            # (the exclusivity invariant — previously VIOLATED, CYTHON-TEST-001).
            probe = ctx.Process(target=_try_acquire, args=(lock_dir, q, 0.0))
            probe.start()
            probe.join(15)
            assert q.get(timeout=5) == "timeout"
        finally:
            holder.join(15)

        # After the holder releases, acquisition succeeds.
        after = ctx.Process(target=_try_acquire, args=(lock_dir, q, 5.0))
        after.start()
        after.join(15)
        assert q.get(timeout=5) == "acquired"

    def test_waiter_acquires_after_holder_releases(self, tmp_path: Path) -> None:
        ctx = mp.get_context("spawn")
        lock_dir = str(tmp_path / "build2.lock")
        ready = ctx.Event()
        q = ctx.Queue()

        holder = ctx.Process(target=_hold_lock, args=(lock_dir, ready, 0.8))
        holder.start()
        assert ready.wait(timeout=15), "holder never acquired"

        # A waiter with a generous timeout should block until release, then win.
        waiter = ctx.Process(target=_try_acquire, args=(lock_dir, q, 10.0))
        waiter.start()
        waiter.join(20)
        holder.join(15)
        assert q.get(timeout=5) == "acquired"


class TestStalenessDecoupledFromTimeout:
    """The core bug fix: timeout_s=0 must not reclaim a live lock."""

    def test_zero_timeout_does_not_reclaim_fresh_lock(self, tmp_path: Path) -> None:
        from .._lock import build_lock

        lock_dir = tmp_path / "fresh.lock"
        lock_dir.mkdir()  # a brand-new (live) lock, age ~0

        # timeout_s=0 must TIME OUT, not treat the fresh lock as stale.
        with pytest.raises(TimeoutError):
            with build_lock(lock_dir, timeout_s=0.0):
                pass

    def test_explicit_stale_after_still_recovers(self, tmp_path: Path) -> None:
        import os

        from .._lock import build_lock

        lock_dir = tmp_path / "crashed.lock"
        lock_dir.mkdir()
        old = time.time() - 3600
        os.utime(lock_dir, (old, old))

        # A genuinely old lock is reclaimable even with timeout_s=0 when the
        # caller sets an appropriate staleness grace period.
        with build_lock(lock_dir, timeout_s=0.0, stale_after_s=60.0):
            assert lock_dir.exists()

    def test_default_staleness_is_large(self, tmp_path: Path) -> None:
        from .._lock import _DEFAULT_STALE_AFTER_S

        # Default grace period is well beyond any reasonable build.
        assert _DEFAULT_STALE_AFTER_S >= 300.0
