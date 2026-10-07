# corpus/tests/test__atomic.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Atomic publication primitive gate (CORPUS-TMP-001)
==================================================

``scikitplot.corpus._atomic`` is the single publish primitive that every
cache/export/storage writer funnels through instead of rolling its own
predictable ``path + ".tmp"`` scheme. These are the permanent regressions:
unique staging temps, durable + atomic publish, failure cleanup, and — the
finding's exit criterion — a concurrent-publication stress test that must leave
exactly one complete payload and no orphan temporary files.

Run with::

    pytest scikitplot/corpus/tests/test__atomic.py -v
"""

from __future__ import annotations

import multiprocessing as mp
import pathlib

import pytest

from scikitplot.corpus._atomic import atomic_write_bytes, atomic_write_path


# Top-level so multiprocessing can pickle it under spawn or fork.
def _publish_worker(args) -> bool:
    target, payload = args
    atomic_write_bytes(target, payload)
    return True


class TestAtomicWriteBytes:
    def test_publishes_content(self, tmp_path: pathlib.Path) -> None:
        t = tmp_path / "out.bin"
        atomic_write_bytes(t, b"hello")
        assert t.read_bytes() == b"hello"

    def test_no_predictable_temp_left(self, tmp_path: pathlib.Path) -> None:
        t = tmp_path / "out.bin"
        atomic_write_bytes(t, b"data")
        assert list(tmp_path.glob("*.tmp")) == []
        assert not (tmp_path / "out.bin.tmp").exists()

    def test_atomic_overwrite(self, tmp_path: pathlib.Path) -> None:
        t = tmp_path / "out.bin"
        atomic_write_bytes(t, b"first")
        atomic_write_bytes(t, b"second-longer")
        assert t.read_bytes() == b"second-longer"

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        t = tmp_path / "a" / "b" / "c.bin"
        atomic_write_bytes(t, b"x")
        assert t.read_bytes() == b"x"


class TestAtomicWritePath:
    def test_writer_based_publish(self, tmp_path: pathlib.Path) -> None:
        t = tmp_path / "file.dat"
        atomic_write_path(t, lambda p: p.write_text("payload", encoding="utf-8"), suffix=".dat")
        assert t.read_text(encoding="utf-8") == "payload"

    def test_staging_suffix_is_used(self, tmp_path: pathlib.Path) -> None:
        seen: list[str] = []
        t = tmp_path / "f"
        atomic_write_path(t, lambda p: seen.append(p.suffix) or p.write_text("ok", encoding="utf-8"), suffix=".npy")
        assert seen == [".npy"]
        assert t.read_text(encoding="utf-8") == "ok"

    def test_failure_cleanup_and_reraise(self, tmp_path: pathlib.Path) -> None:
        t = tmp_path / "target.bin"

        def _boom(p: pathlib.Path) -> None:
            p.write_bytes(b"partial")
            raise RuntimeError("writer failed")

        with pytest.raises(RuntimeError):
            atomic_write_path(t, _boom)
        assert not t.exists()
        assert list(tmp_path.iterdir()) == []  # no orphan temp

    def test_a_writer_may_publish_a_directory(self, tmp_path: pathlib.Path) -> None:
        """The staging path can become a directory, which is then the target."""
        t = tmp_path / "generation"

        def _directory(p: pathlib.Path) -> None:
            p.unlink()
            p.mkdir()
            (p / "payload").write_text("ok", encoding="utf-8")

        atomic_write_path(t, _directory)
        assert (t / "payload").read_text(encoding="utf-8") == "ok"

    def test_a_directory_is_published_where_one_cannot_be_opened(
        self, tmp_path: pathlib.Path, monkeypatch, caplog
    ) -> None:
        """
        Windows refuses to open a directory; that must not fail the publication.

        The refusal is a platform limit on *syncing* a directory, so it is
        reported as a durability downgrade. Treating the staged directory as a
        file made every artifact publication fail on Windows with
        "Permission denied".
        """
        import errno
        import logging
        import os

        import scikitplot.corpus._atomic as mod

        real_open = os.open

        def _open(path, flags, *args, **kwargs):
            if os.path.isdir(path):
                raise PermissionError(errno.EACCES, "Permission denied", os.fspath(path))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(mod.os, "open", _open)
        t = tmp_path / "generation"

        def _directory(p: pathlib.Path) -> None:
            p.unlink()
            p.mkdir()
            (p / "payload").write_text("ok", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            atomic_write_path(t, _directory)
        assert (t / "payload").read_text(encoding="utf-8") == "ok"
        assert any("durability downgraded" in r.getMessage() for r in caplog.records)

    def test_a_file_that_cannot_be_opened_to_sync_still_fails(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        """The tolerance is for directories only: a file's EACCES is a real failure."""
        import errno
        import os

        import scikitplot.corpus._atomic as mod

        real_open = os.open

        def _open(path, flags, *args, **kwargs):
            if str(path).endswith(".tmp") and flags == os.O_RDONLY:
                raise PermissionError(errno.EACCES, "Permission denied", os.fspath(path))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(mod.os, "open", _open)
        t = tmp_path / "file.bin"
        with pytest.raises(PermissionError):
            atomic_write_path(t, lambda p: p.write_bytes(b"x"))
        assert not t.exists()

    def test_unique_staging_names(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        import scikitplot.corpus._atomic as mod

        seen: list[str] = []
        orig = mod.tempfile.mkstemp

        def _spy(*a, **k):
            fd, name = orig(*a, **k)
            seen.append(name)
            return fd, name

        monkeypatch.setattr(mod.tempfile, "mkstemp", _spy)
        t = tmp_path / "same.bin"
        atomic_write_bytes(t, b"a")
        atomic_write_bytes(t, b"b")
        assert len(seen) == 2 and seen[0] != seen[1]
        assert t.read_bytes() == b"b"


class TestConcurrentPublication:
    def test_contended_target_stays_consistent(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "contended.bin"
        n = 20
        payloads = [bytes([i]) * 4096 for i in range(n)]
        args = [(str(target), p) for p in payloads]

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(processes=8) as pool:
            results = pool.map(_publish_worker, args)

        assert all(results)
        final = target.read_bytes()
        # Exactly one worker's full payload — never a mix or a partial write.
        assert final in payloads
        assert len(final) == 4096
        # No orphan temporary files, only the published target remains.
        assert list(tmp_path.glob("*.tmp")) == []
        assert [p.name for p in tmp_path.iterdir()] == ["contended.bin"]
