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

    def test_readers_never_see_a_partial_file(self, tmp_path, monkeypatch):
        """
        A reader gets a whole document or, on Windows only, a refusal.

        Notes
        -----
        **Developer notes.** Windows does not replace a file that a reader
        has open, nor open one that is being replaced; either side can be
        refused with ``PermissionError`` (measured: ``[WinError 5]`` from
        ``os.replace`` in the Windows job of CI run 37668804532). A refusal
        leaves the previous document whole, so it is counted, not failed.
        What must hold on every platform is that no reader ever sees a part
        and that no scratch file is left behind. On POSIX nothing is refused.

        The pauses between repeated attempts are shortened: a reader in a
        tight loop is the worst case for a writer, and this test is about
        what a reader sees, not about how long a writer waits. With open
        files made unreplaceable on Linux, 14 or 15 of the 40 writes were
        still refused after the full pauses, and the test took 20 s.
        """
        import threading

        from .. import _files

        monkeypatch.setattr(_files, "_REPLACE_PAUSES", (0.001, 0.001))

        target = tmp_path / "vault.json"
        atomic_write(str(target), "a" * 100_000)
        seen = set()
        refused = []
        stop = threading.Event()

        def read():
            while not stop.is_set():
                try:
                    seen.add(len(target.read_text(encoding="utf-8")))
                except PermissionError as exc:
                    refused.append(exc)

        reader = threading.Thread(target=read)
        reader.start()
        try:
            for size in (100_000, 200_000) * 20:
                try:
                    atomic_write(str(target), "a" * size)
                except PermissionError as exc:
                    refused.append(exc)
        finally:
            stop.set()
            reader.join()
        assert seen <= {100_000, 200_000}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["vault.json"]
        assert len(target.read_text(encoding="utf-8")) in {100_000, 200_000}
        if os.name != "nt":
            assert refused == []


class TestRefusedReplace:
    """A replace that the platform refuses for an instant is repeated, bounded."""

    @pytest.fixture()
    def refusals(self, monkeypatch):
        """Make ``os.replace`` refuse a set number of times; record the pauses."""
        from .. import _files

        state = {"refuse": 0, "calls": 0, "pauses": []}
        real = os.replace

        def replace(source, target):
            state["calls"] += 1
            if state["calls"] <= state["refuse"]:
                raise PermissionError(13, "Access is denied", str(source))
            return real(source, target)

        monkeypatch.setattr(_files.os, "replace", replace)
        monkeypatch.setattr(_files.time, "sleep", state["pauses"].append)
        return state

    def test_a_transient_refusal_is_repeated_where_the_platform_has_them(
        self, tmp_path, monkeypatch, refusals
    ):
        from .. import _files

        monkeypatch.setattr(_files, "_RETRY_REFUSED_REPLACE", True)
        refusals["refuse"] = 3
        target = tmp_path / "vault.json"
        atomic_write(str(target), "new")
        assert target.read_text(encoding="utf-8") == "new"
        assert refusals["calls"] == 4
        assert refusals["pauses"] == list(_files._REPLACE_PAUSES[:3])
        assert [p.name for p in tmp_path.iterdir()] == ["vault.json"]

    def test_a_lasting_refusal_is_raised_and_leaves_the_old_file(
        self, tmp_path, monkeypatch, refusals
    ):
        from .. import _files

        monkeypatch.setattr(_files, "_RETRY_REFUSED_REPLACE", True)
        target = tmp_path / "vault.json"
        target.write_bytes(b"old")
        refusals["refuse"] = 10**6
        with pytest.raises(PermissionError):
            atomic_write(str(target), "new")
        assert refusals["calls"] == len(_files._REPLACE_PAUSES) + 1
        assert target.read_bytes() == b"old"
        assert [p.name for p in tmp_path.iterdir()] == ["vault.json"]

    def test_no_repetition_where_the_platform_has_no_such_refusal(
        self, tmp_path, monkeypatch, refusals
    ):
        from .. import _files

        monkeypatch.setattr(_files, "_RETRY_REFUSED_REPLACE", False)
        refusals["refuse"] = 1
        with pytest.raises(PermissionError):
            atomic_write(str(tmp_path / "vault.json"), "new")
        assert refusals["calls"] == 1 and refusals["pauses"] == []

    def test_another_error_is_never_repeated(self, tmp_path, monkeypatch):
        from .. import _files

        calls = []

        def replace(source, target):
            calls.append(source)
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(_files, "_RETRY_REFUSED_REPLACE", True)
        monkeypatch.setattr(_files.os, "replace", replace)
        with pytest.raises(OSError, match="No space"):
            atomic_write(str(tmp_path / "vault.json"), "new")
        assert len(calls) == 1

    def test_the_pauses_are_bounded(self):
        from .. import _files

        assert all(pause > 0 for pause in _files._REPLACE_PAUSES)
        assert sum(_files._REPLACE_PAUSES) <= 2.0
