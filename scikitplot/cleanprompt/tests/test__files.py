"""
Tests for :mod:`scikitplot.cleanprompt._files`.

Notes
-----
**Developer notes.** The lock is tested across real processes, because a lock
that only works within one process is exactly what ``CP-068`` needed more
than. The holder is a child Python that takes the lock, prints ``held`` and
waits; the parent then tries to take it.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time

import pytest

from .._exceptions import CleanPromptError
from .._files import atomic_write, locked

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _holder(path, seconds):
    """Start a process that holds the lock on ``path`` for ``seconds``."""
    code = (
        "import sys, time\n"
        "from scikitplot.cleanprompt._files import locked\n"
        f"with locked({str(path)!r}):\n"
        "    print('held', flush=True)\n"
        f"    time.sleep({seconds})\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        cwd=_REPO,
    )
    assert child.stdout.readline().strip() == "held"
    return child


def _stop(child):
    """
    Kill a holder, reap it, and close its pipe.

    A pipe left open is closed by the garbage collector at some later point,
    and the project runs pytest with warnings as errors: the resulting
    ``ResourceWarning`` then fails whichever test is running (``CP-087``).
    """
    child.kill()
    child.wait()
    child.stdout.close()


class TestLocked:
    def test_a_second_process_waits_then_gives_up_with_the_reason(self, tmp_path):
        target = tmp_path / "vault.json"
        child = _holder(target, 30)
        told = []
        try:
            started = time.monotonic()
            with pytest.raises(CleanPromptError, match="held the vault lock"):
                with locked(str(target), timeout=0.3, waiting=lambda: told.append(1)):
                    pass
            assert time.monotonic() - started < 5
            assert told == [1]
        finally:
            _stop(child)

    def test_a_killed_holder_leaves_no_stale_lock(self, tmp_path):
        target = tmp_path / "vault.json"
        child = _holder(target, 30)
        _stop(child)
        with locked(str(target), timeout=5):
            pass

    def test_the_lock_is_taken_again_after_release(self, tmp_path):
        target = str(tmp_path / "vault.json")
        for _ in range(3):
            with locked(target, timeout=1):
                pass
        assert os.path.exists(target + ".lock")

    @pytest.mark.parametrize("bad", [0, -1, True, "5", float("nan")])
    def test_timeout_is_validated(self, tmp_path, bad):
        with pytest.raises(ValueError, match="timeout"):
            with locked(str(tmp_path / "v"), timeout=bad):
                pass


class TestAtomicWrite:
    def test_creates_owner_only_with_a_parent(self, tmp_path):
        target = tmp_path / "state" / "vault.json"
        atomic_write(str(target), "{}\n")
        assert target.read_text(encoding="utf-8") == "{}\n"
        if os.name == "posix":
            assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_a_failure_leaves_the_old_file_and_no_scratch(self, tmp_path, monkeypatch):
        target = tmp_path / "vault.json"
        target.write_text("old", encoding="utf-8")

        def refuse(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", refuse)
        with pytest.raises(OSError, match="disk full"):
            atomic_write(str(target), "new")
        assert target.read_text(encoding="utf-8") == "old"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["vault.json"]

    def test_a_symlink_keeps_pointing_at_its_target(self, tmp_path):
        real = tmp_path / "real.json"
        real.write_text("old", encoding="utf-8")
        link = tmp_path / "link.json"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("symlinks unavailable")
        atomic_write(str(link), "new")
        assert link.is_symlink() and real.read_text(encoding="utf-8") == "new"

    def test_readers_never_see_a_partial_file(self, tmp_path):
        import threading

        target = tmp_path / "vault.json"
        atomic_write(str(target), "a" * 100_000)
        seen = set()
        stop = threading.Event()

        def read():
            while not stop.is_set():
                seen.add(len(target.read_text(encoding="utf-8")))

        reader = threading.Thread(target=read)
        reader.start()
        try:
            for size in (100_000, 200_000) * 20:
                atomic_write(str(target), "a" * size)
        finally:
            stop.set()
            reader.join()
        assert seen <= {100_000, 200_000}
