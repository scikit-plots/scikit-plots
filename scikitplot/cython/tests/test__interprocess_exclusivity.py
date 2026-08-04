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
breaking exclusivity. Staleness is now decoupled through ``stale_after_s``.

The interprocess tests use isolated Python subprocesses and load ``_lock.py``
directly. This avoids ``multiprocessing`` ``spawn`` importing the full
``scikitplot.cython`` package hierarchy before the worker starts, which can hang
under a large test suite and produce a false lock failure.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

_LOCK_MODULE = Path(__file__).resolve().parents[1] / "_lock.py"
_CHILD_START_TIMEOUT_S = 30.0
_CHILD_EXIT_TIMEOUT_S = 30.0

_HOLDER_CODE = r"""
# import multiprocessing as mp
import importlib.util
import pathlib
import sys
import time
import traceback

module_path = pathlib.Path(sys.argv[1])
lock_dir = pathlib.Path(sys.argv[2])
ready_path = pathlib.Path(sys.argv[3])
release_path = pathlib.Path(sys.argv[4])

try:
    spec = importlib.util.spec_from_file_location(
        "_isolated_scikitplot_build_lock",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load lock module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with module.build_lock(lock_dir, timeout_s=30.0):
        ready_path.write_text("acquired", encoding="utf-8")

        deadline = time.monotonic() + 60.0
        while not release_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "parent did not release holder within 60 seconds"
                )
            time.sleep(0.02)
except BaseException:
    traceback.print_exc()
    raise
"""

_PROBE_CODE = r"""
import importlib.util
import pathlib
import sys
import traceback

module_path = pathlib.Path(sys.argv[1])
lock_dir = pathlib.Path(sys.argv[2])
timeout_s = float(sys.argv[3])

try:
    spec = importlib.util.spec_from_file_location(
        "_isolated_scikitplot_build_lock",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load lock module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        with module.build_lock(lock_dir, timeout_s=timeout_s):
            print("acquired", flush=True)
    except TimeoutError:
        print("timeout", flush=True)
except BaseException:
    traceback.print_exc()
    raise
"""


def _python_command(code: str, *args: object) -> list[str]:
    """Return a clean child-interpreter command for the lock worker."""
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        code,
        *(str(arg) for arg in args),
    ]


def _start_holder(
    tmp_path: Path,
    lock_dir: Path,
) -> tuple[subprocess.Popen[str], Path, Path]:
    """Start an isolated process that owns the lock until released."""
    ready_path = tmp_path / "holder.ready"
    release_path = tmp_path / "holder.release"

    process = subprocess.Popen(
        _python_command(
            _HOLDER_CODE,
            _LOCK_MODULE,
            lock_dir,
            ready_path,
            release_path,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process, ready_path, release_path


def _process_diagnostics(
    label: str,
    process: subprocess.Popen[str],
    stdout: str = "",
    stderr: str = "",
) -> str:
    """Format useful child-process state for assertion messages."""
    return (
        f"{label}\n"
        f"returncode: {process.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def _cleanup_process(process: subprocess.Popen[str]) -> None:
    """Best-effort, non-raising cleanup for a child process."""
    if process.poll() is None:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
    else:
        # Reap the process and drain its pipes when not already communicated.
        try:
            process.communicate(timeout=0)
        except (subprocess.TimeoutExpired, ValueError):
            pass


def _wait_until_holder_ready(
    holder: subprocess.Popen[str],
    ready_path: Path,
    *,
    timeout: float = _CHILD_START_TIMEOUT_S,
) -> None:
    """Wait until the holder confirms ownership or report its early failure."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if ready_path.exists():
            assert ready_path.read_text(encoding="utf-8") == "acquired"
            return

        if holder.poll() is not None:
            stdout, stderr = holder.communicate()
            pytest.fail(
                _process_diagnostics(
                    "holder exited before acquiring the lock",
                    holder,
                    stdout,
                    stderr,
                )
            )

        time.sleep(0.02)

    pytest.fail(
        "holder produced no readiness signal within "
        f"{timeout} seconds; alive={holder.poll() is None}, "
        f"returncode={holder.returncode}"
    )


def _release_holder(
    holder: subprocess.Popen[str],
    release_path: Path,
    *,
    timeout: float = _CHILD_EXIT_TIMEOUT_S,
) -> None:
    """Release a healthy holder and require a clean child exit."""
    release_path.touch()

    try:
        stdout, stderr = holder.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"holder did not exit within {timeout} seconds after release")

    assert holder.returncode == 0, _process_diagnostics(
        "holder exited abnormally",
        holder,
        stdout,
        stderr,
    )


def _run_probe(
    lock_dir: Path,
    *,
    timeout_s: float,
    process_timeout: float = _CHILD_EXIT_TIMEOUT_S,
) -> str:
    """Run one isolated acquisition attempt and return its reported result."""
    try:
        completed = subprocess.run(
            _python_command(
                _PROBE_CODE,
                _LOCK_MODULE,
                lock_dir,
                timeout_s,
            ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=process_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"lock probe did not exit within {process_timeout} seconds; "
            f"stdout={exc.stdout!r}, stderr={exc.stderr!r}"
        )

    assert completed.returncode == 0, (
        "lock probe exited abnormally\n"
        f"returncode: {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return completed.stdout.strip()


class TestInterprocessExclusivity:
    def test_second_process_cannot_acquire_while_held(
        self,
        tmp_path: Path,
    ) -> None:
        lock_dir = tmp_path / "build.lock"
        holder, ready_path, release_path = _start_holder(tmp_path, lock_dir)

        try:
            _wait_until_holder_ready(holder, ready_path)

            # A non-blocking second process must not reclaim or acquire the
            # holder's fresh, live lock.
            assert _run_probe(lock_dir, timeout_s=0.0) == "timeout"

            _release_holder(holder, release_path)
        finally:
            release_path.touch(exist_ok=True)
            _cleanup_process(holder)

        # Once the real owner releases, acquisition must succeed.
        assert _run_probe(lock_dir, timeout_s=5.0) == "acquired"

    def test_waiter_acquires_after_holder_releases(
        self,
        tmp_path: Path,
    ) -> None:
        lock_dir = tmp_path / "build2.lock"
        holder, ready_path, release_path = _start_holder(tmp_path, lock_dir)
        waiter: subprocess.Popen[str] | None = None

        try:
            _wait_until_holder_ready(holder, ready_path)

            waiter = subprocess.Popen(
                _python_command(
                    _PROBE_CODE,
                    _LOCK_MODULE,
                    lock_dir,
                    10.0,
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Confirm that the waiter remains blocked while ownership is live.
            time.sleep(0.25)
            assert waiter.poll() is None, (
                "waiter exited before the holder released the lock"
            )

            _release_holder(holder, release_path)

            try:
                stdout, stderr = waiter.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                pytest.fail("waiter did not acquire after the holder released")

            assert waiter.returncode == 0, _process_diagnostics(
                "waiter exited abnormally",
                waiter,
                stdout,
                stderr,
            )
            assert stdout.strip() == "acquired"
        finally:
            release_path.touch(exist_ok=True)
            _cleanup_process(holder)
            if waiter is not None:
                _cleanup_process(waiter)


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
