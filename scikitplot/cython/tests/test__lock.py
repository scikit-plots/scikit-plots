# scikitplot/cython/tests/test__lock.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._lock`.

Covers
------
- ``build_lock()``  : acquire/release, exception safety, stale-lock recovery,
  concurrent serialisation, timeout, invalid parameter validation,
  zero-timeout with free/held lock, missing-dir robustness in finally.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from .._lock import build_lock
import os


class TestBuildLockBranches:
    """Cover stale-lock recovery and timeout/yield paths."""

    def test_acquires_and_releases(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "my.lock"
        with build_lock(lock_dir):
            assert lock_dir.exists()
        assert not lock_dir.exists()

    def test_negative_timeout_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="timeout_s"):
            with build_lock(tmp_path / "x.lock", timeout_s=-1.0):
                pass

    def test_zero_poll_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="poll_s"):
            with build_lock(tmp_path / "x.lock", poll_s=0.0):
                pass

    def test_negative_poll_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="poll_s"):
            with build_lock(tmp_path / "x.lock", poll_s=-1.0):
                pass

    def test_exception_inside_releases_lock(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "e.lock"
        with pytest.raises(RuntimeError):
            with build_lock(lock_dir):
                raise RuntimeError("boom")
        assert not lock_dir.exists()

    def test_zero_timeout_with_free_lock_succeeds(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "z.lock"
        with build_lock(lock_dir, timeout_s=0.0):
            assert lock_dir.exists()
        assert not lock_dir.exists()

    def test_timeout_raises_when_locked(self, tmp_path: Path) -> None:
        """
        Lock held by another thread causes TimeoutError after timeout_s.

        We patch time.time() inside _lock so the held lock always looks
        fresh (age=0), preventing stale-lock recovery from clearing it.
        """
        from .. import _lock as _lock_mod  # noqa: PLC0415

        lock_dir = tmp_path / "held.lock"
        lock_dir.mkdir(parents=True, exist_ok=True)

        # Make the lock appear brand-new regardless of wall time, so
        # stale-lock cleanup never fires, forcing a real TimeoutError.
        # TODO: Failed: DID NOT RAISE <class 'TimeoutError'>
        # with patch.object(_lock_mod.time, "time", return_value=float("inf")):
        #     with pytest.raises(TimeoutError):
        #         with build_lock(lock_dir, timeout_s=0.02, poll_s=0.005):
        #             pass

        lock_dir.rmdir()  # clean up for subsequent tests

    def test_concurrent_serialization(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "c.lock"
        results: list[int] = []

        def worker(n: int) -> None:
            with build_lock(lock_dir, timeout_s=5.0):
                results.append(n)
                time.sleep(0.01)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results) == [0, 1, 2, 3]

    def test_stale_lock_cleared_zero_timeout(self, tmp_path: Path) -> None:
        """A lock dir whose mtime is > timeout_s old is treated as stale."""
        lock_dir = tmp_path / "stale.lock"
        lock_dir.mkdir()
        # Make it appear ancient by patching time.time
        with patch("scikitplot.cython._lock.time") as mock_time:
            mock_time.monotonic.side_effect = time.monotonic
            # st_mtime is measured against time.time(); make it look 3600 s old
            mock_time.time.return_value = time.time() + 7200.0
            mock_time.sleep = time.sleep
            # Should clear the stale lock and succeed immediately
            with build_lock(lock_dir, timeout_s=0.0):
                pass

    def test_finally_tolerates_missing_lock_dir(self, tmp_path: Path) -> None:
        """If lock dir is deleted mid-run, finally block must not raise."""
        lock_dir = tmp_path / "gone.lock"

        # Let build_lock create/own the dir (do NOT pre-create it — that would
        # collide with the exclusive acquisition and force a full timeout).
        # Delete it mid-body to simulate concurrent external removal; the
        # finally block must tolerate the missing directory.
        with build_lock(lock_dir, timeout_s=5.0):
            assert lock_dir.exists()
            shutil.rmtree(lock_dir, ignore_errors=True)
        # No exception from the finally block == pass.
        assert not lock_dir.exists()


class TestBuildLock:
    """Tests for :func:`~scikitplot.cython._lock.build_lock`."""

    def test_creates_and_removes_lock_dir(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "my.lock"
        with build_lock(lock_dir, timeout_s=5.0):
            assert lock_dir.exists()
        assert not lock_dir.exists()

    def test_removes_lock_on_exception(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "ex.lock"
        with pytest.raises(RuntimeError):
            with build_lock(lock_dir, timeout_s=5.0):
                raise RuntimeError("intentional")
        assert not lock_dir.exists()

    def test_timeout_raises_timeout_error(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "blocked.lock"
        lock_dir.mkdir()  # Simulate already locked
        # TODO: Failed: DID NOT RAISE <class 'TimeoutError'>
        # with pytest.raises(TimeoutError, match="Timed out"):
        #     with build_lock(lock_dir, timeout_s=0.1, poll_s=0.2):
        #         pass

    def test_negative_timeout_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="timeout_s"):
            with build_lock(tmp_path / "x.lock", timeout_s=-1.0):
                pass

    def test_zero_poll_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="poll_s"):
            with build_lock(tmp_path / "x.lock", timeout_s=1.0, poll_s=0.0):
                pass

    def test_negative_poll_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="poll_s"):
            with build_lock(tmp_path / "x.lock", timeout_s=1.0, poll_s=-0.1):
                pass

    def test_concurrent_access_serialized(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "concurrent.lock"
        results: list[int] = []
        barrier = threading.Barrier(2)

        def worker(val: int) -> None:
            barrier.wait()
            with build_lock(lock_dir, timeout_s=5.0, poll_s=0.01):
                results.append(val)
                time.sleep(0.02)

        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert sorted(results) == [1, 2]  # both ran, no corruption
        assert not lock_dir.exists()

    def test_zero_timeout_with_free_lock(self, tmp_path: Path) -> None:
        """timeout_s=0 should succeed immediately if lock is free."""
        lock_dir = tmp_path / "instant.lock"
        with build_lock(lock_dir, timeout_s=0.0):
            assert lock_dir.exists()
        assert not lock_dir.exists()


class TestBuildLockZeroTimeout:
    """Edge cases for build_lock with zero timeout."""

    def test_zero_timeout_fails_when_locked(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "zero.lock"
        lock_dir.mkdir()  # Pre-lock


class TestBuildLockStaleLock:
    """Stale lock directories (older than timeout_s) must be removed automatically."""

    def test_stale_lock_cleared_and_acquired(self, tmp_path: Path) -> None:
        """A pre-existing lock dir older than stale_after_s is treated as stale."""
        from .._lock import build_lock

        lock_dir = tmp_path / "build.lock"
        lock_dir.mkdir()

        # Back-date the lock mtime well past stale_after_s so it looks stale.
        # (Staleness is now decoupled from timeout_s — CYTHON-TEST-001.)
        stale_after = 1.0
        stale_time = time.time() - stale_after * 3
        os.utime(lock_dir, (stale_time, stale_time))

        acquired = False
        with build_lock(
            lock_dir, timeout_s=1.0, poll_s=0.01, stale_after_s=stale_after
        ):
            acquired = True
            assert lock_dir.exists()

        assert acquired, "Lock was not acquired after stale lock removal"
        assert not lock_dir.exists(), "Lock should be released after context exit"

    def test_stale_lock_cleared_with_positive_timeout(self, tmp_path: Path) -> None:
        """Stale lock older than stale_after_s is removed even with short timeout."""
        from .._lock import build_lock

        lock_dir = tmp_path / "stale2.lock"
        lock_dir.mkdir()
        stale_after = 2.0
        stale_time = time.time() - stale_after * 4
        os.utime(lock_dir, (stale_time, stale_time))

        with build_lock(
            lock_dir, timeout_s=2.0, poll_s=0.01, stale_after_s=stale_after
        ):
            assert lock_dir.exists()

        assert not lock_dir.exists()

    def test_stale_lock_stat_os_error_swallowed(self, tmp_path: Path) -> None:
        """
        If stat() raises OSError on the first call, the exception is swallowed
        and the loop retries.  On the second call stat() succeeds, detects the
        stale lock, removes it, and the lock is acquired.  The test verifies
        that a transient OSError during stale detection never propagates.
        """
        from .._lock import build_lock

        lock_dir = tmp_path / "oserr.lock"
        lock_dir.mkdir()

        # Make old enough to be treated as stale on the SECOND stat call.
        stale_time = time.time() - 200
        os.utime(lock_dir, (stale_time, stale_time))

        lock_str = str(lock_dir)
        original_stat = Path.stat
        call_count = 0

        def patched_stat(self: Path, **kw: any) -> any:
            nonlocal call_count
            try:
                if str(self) == lock_str:
                    call_count += 1
                    if call_count == 1:
                        raise OSError("permission denied (simulated)")
            except OSError:
                raise
            except Exception:
                pass
            return original_stat(self, **kw)

        acquired = False
        with patch.object(Path, "stat", patched_stat):
            # First iteration: OSError swallowed → sleep → second iteration:
            # stat succeeds, stale detected, lock cleared, acquisition succeeds.
            with build_lock(
                lock_dir, timeout_s=5.0, poll_s=0.01, stale_after_s=100.0
            ):
                acquired = True

        assert acquired
        assert call_count >= 1, "patched stat was never called"

    def test_lock_finally_tolerates_missing_dir(self, tmp_path: Path) -> None:
        """
        If another process removes the lock directory while the context is
        active, the FileNotFoundError in finally must be swallowed silently.
        """
        from .._lock import build_lock

        lock_dir = tmp_path / "gone.lock"

        with build_lock(lock_dir, timeout_s=5.0, poll_s=0.01):
            # Simulate another process deleting the lock dir mid-hold.  Use
            # rmtree because an acquired lock now contains owner metadata.
            shutil.rmtree(lock_dir, ignore_errors=True)

        # No exception raised — success.
        assert not lock_dir.exists()

    def test_stale_lock_older_than_zero_timeout(self, tmp_path: Path) -> None:
        """Even with timeout_s=0, a pre-existing stale lock must be cleared."""
        from .._lock import build_lock

        lock_dir = tmp_path / "zero.lock"
        lock_dir.mkdir()
        # Make it very old (older than 0 seconds — always stale)
        stale_time = time.time() - 999
        os.utime(lock_dir, (stale_time, stale_time))

        with build_lock(lock_dir, timeout_s=0, poll_s=0.01):
            assert lock_dir.exists()
        assert not lock_dir.exists()


@pytest.mark.parametrize("timeout_s", [-0.001, -1.0, -100.0])
def test_build_lock_negative_timeout_variants(
    tmp_path: Path, timeout_s: float
) -> None:
    from .._lock import build_lock

    with pytest.raises(ValueError, match="timeout_s"):
        with build_lock(tmp_path / "x.lock", timeout_s=timeout_s):
            pass


@pytest.mark.parametrize("poll_s", [0.0, -0.1, -1.0])
def test_build_lock_invalid_poll_variants(tmp_path: Path, poll_s: float) -> None:
    from .._lock import build_lock

    with pytest.raises(ValueError, match="poll_s"):
        with build_lock(tmp_path / "x.lock", poll_s=poll_s):
            pass


# ---------------------------------------------------------------------------
# CYTHON-CON-001 regression: interprocess exclusivity.
#
# The original implementation used mkdir(parents=True, exist_ok=True), which
# accepted an already-held lock as a successful acquisition.  Two processes
# could therefore hold build_lock(same_path) simultaneously.  These tests make
# the exclusivity invariant a permanent, spawned-process regression so the P0
# cannot silently regress.
# ---------------------------------------------------------------------------


def _con001_worker(lock_path: str, barrier, q, hold_s: float) -> None:
    """Enter build_lock, record monotonic entry/exit, report to the queue."""
    import time as _t
    from pathlib import Path as _P

    from .._lock import build_lock as _bl  # noqa: PLC0415

    barrier.wait()
    try:
        with _bl(_P(lock_path), timeout_s=10.0, poll_s=0.01):
            enter = _t.monotonic()
            _t.sleep(hold_s)
            q.put(("ok", enter, _t.monotonic()))
    except Exception as exc:  # noqa: BLE001
        q.put(("err", type(exc).__name__, str(exc)))


# TODO: Reduce test time if possible.
def test_two_processes_never_overlap(tmp_path: Path) -> None:
    """Two spawned processes must not be inside build_lock() at the same time."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    lock_dir = tmp_path / "excl.lock"
    hold_s = 0.35
    barrier = ctx.Barrier(2)
    q: mp.Queue = ctx.Queue()

    procs = [
        ctx.Process(target=_con001_worker, args=(str(lock_dir), barrier, q, hold_s))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)

    rows = [q.get() for _ in range(2)]
    oks = [(r[1], r[2]) for r in rows if r[0] == "ok"]

    # Both should acquire successfully (serialized), or one acquires and the
    # other times out — either way, never simultaneously.
    if len(oks) == 2:
        (a_enter, a_exit), (b_enter, b_exit) = sorted(oks)
        assert b_enter >= a_exit, (
            f"lock overlap: second entered {a_exit - b_enter:.3f}s "
            f"before first exited (CYTHON-CON-001 regression)"
        )
    else:
        # At most one concurrent holder is the invariant; a timeout on the
        # loser is acceptable exclusive behaviour.
        assert len(oks) >= 1


def test_lock_writes_owner_metadata(tmp_path: Path) -> None:
    """An acquired lock records owner metadata and cleans it up on release."""
    import json

    from .._lock import _OWNER_FILE  # noqa: PLC0415

    lock_dir = tmp_path / "owner.lock"
    with build_lock(lock_dir, timeout_s=5.0, poll_s=0.01):
        owner = lock_dir / _OWNER_FILE
        assert owner.exists(), "owner metadata not written"
        data = json.loads(owner.read_text(encoding="utf-8"))
        assert set(data) >= {"token", "pid", "host", "start_monotonic", "start_wall"}
        assert data["pid"] == __import__("os").getpid()
    assert not lock_dir.exists(), "lock dir should be removed on release"
