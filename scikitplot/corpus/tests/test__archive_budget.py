# corpus/tests/test__archive_budget.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Archive streaming + actual-byte budget gate (CORPUS-ARC-001)
============================================================

Untrusted compressed input must be extracted under an **actual**-decompressed-byte
budget with bounded peak memory — never ``dst.write(src.read())`` (which
materialises the whole member) and never trusting the declared ``file_size``.

Run with::

    pytest scikitplot/corpus/tests/test__archive_budget.py -v
"""

from __future__ import annotations

import io
import pathlib
import zipfile

import pytest

from scikitplot.corpus._archive_handler import extract_archive, stream_copy_bounded

MB = 1024 * 1024


class _FakeSrc:
    """Stream that would yield ``total`` bytes, recording how much was read."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.produced = 0

    def read(self, n: int) -> bytes:
        if self.produced >= self.total:
            return b""
        give = min(n, self.total - self.produced)
        self.produced += give
        return b"\0" * give


class TestStreamCopyBounded:
    def test_over_budget_raises_and_stops_early(self):
        src = _FakeSrc(100 * MB)
        with pytest.raises(ValueError):
            stream_copy_bounded(
                src, io.BytesIO(), max_bytes=5 * MB,
                member_name="m", archive_name="a.zip",
            )
        # Never materialised the full 100 MB — peak memory stays bounded.
        assert src.produced <= 7 * MB

    def test_under_budget_copies_all(self):
        src = _FakeSrc(3 * MB)
        dst = io.BytesIO()
        n = stream_copy_bounded(
            src, dst, max_bytes=10 * MB, member_name="m", archive_name="a",
        )
        assert n == 3 * MB
        assert len(dst.getvalue()) == 3 * MB


class TestExtractArchiveBudget:
    def test_compression_bomb_blocked(self, tmp_path: pathlib.Path):
        bomb = tmp_path / "bomb.zip"
        with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("big.txt", b"\0" * (30 * MB))  # compresses to a few KB
        out = tmp_path / "out"
        with pytest.raises(ValueError):
            extract_archive(bomb, out, max_total_bytes=10 * MB)
        written = (
            sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            if out.exists() else 0
        )
        assert written <= 11 * MB  # did not write the full 30 MB

    def test_cumulative_budget_across_members(self, tmp_path: pathlib.Path):
        multi = tmp_path / "multi.zip"
        with zipfile.ZipFile(multi, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(5):
                zf.writestr(f"m{i}.txt", b"\0" * (3 * MB))  # 15 MB total
        with pytest.raises(ValueError):
            extract_archive(multi, tmp_path / "out", max_total_bytes=10 * MB)

    def test_legit_archive_extracts_correctly(self, tmp_path: pathlib.Path):
        good = tmp_path / "good.zip"
        files = {"a.txt": b"hello alpha", "sub/b.txt": b"bravo content"}
        with zipfile.ZipFile(good, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        out = tmp_path / "out"
        extracted = extract_archive(good, out, max_total_bytes=10 * MB)
        assert len(extracted) == 2
        for name, data in files.items():
            assert (out / name).read_bytes() == data


class TestTransactionalPublish:
    """CORPUS-ARC-003: private staging + atomic publish; no partial state."""

    @staticmethod
    def _zip(path, members):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in members.items():
                zf.writestr(name, data)

    @staticmethod
    def _leftovers(parent, out):
        return [
            p for p in parent.iterdir()
            if p.name.startswith("." + out.name + ".extract-")
        ]

    def test_failure_publishes_nothing(self, tmp_path):
        bomb = tmp_path / "bomb.zip"
        self._zip(bomb, {"ok.txt": b"fine", "big.txt": b"\0" * (30 * MB)})
        out = tmp_path / "dest"
        with pytest.raises(ValueError):
            extract_archive(bomb, out, max_total_bytes=10 * MB)
        assert not out.exists()                       # no partial publish
        assert self._leftovers(tmp_path, out) == []   # no staging leftover

    def test_success_publishes_atomically(self, tmp_path):
        good = tmp_path / "good.zip"
        files = {"a.txt": b"alpha", "sub/b.txt": b"bravo"}
        self._zip(good, files)
        out = tmp_path / "dest"
        published = extract_archive(good, out, max_total_bytes=10 * MB)
        assert len(published) == 2
        assert all((out / n).read_bytes() == data for n, data in files.items())
        assert self._leftovers(tmp_path, out) == []
        assert all(str(p).startswith(str(out)) for p in published)

    def test_reextraction_preserves_existing(self, tmp_path):
        out = tmp_path / "dest"
        out.mkdir()
        (out / "prior.txt").write_bytes(b"keep me")
        z = tmp_path / "more.zip"
        self._zip(z, {"new.txt": b"new content"})
        extract_archive(z, out, max_total_bytes=10 * MB)
        assert (out / "prior.txt").read_bytes() == b"keep me"
        assert (out / "new.txt").read_bytes() == b"new content"

    def test_zipslip_blocked(self, tmp_path):
        evil = tmp_path / "evil.zip"
        self._zip(evil, {"../escape.txt": b"pwned", "safe.txt": b"ok"})
        out = tmp_path / "dest"
        extract_archive(evil, out, max_total_bytes=10 * MB)
        assert not (tmp_path / "escape.txt").exists()  # no escape
        assert (out / "safe.txt").exists()

    def test_publish_failure_does_not_corrupt_existing(self, tmp_path, monkeypatch):
        import scikitplot.corpus._archive_handler as ah

        out = tmp_path / "dest"
        out.mkdir()
        (out / "prior.txt").write_bytes(b"original")
        good = tmp_path / "g.zip"
        self._zip(good, {"x.txt": b"data"})

        def boom(*_a, **_k):
            raise OSError("injected publish failure")

        monkeypatch.setattr(ah.os, "replace", boom)
        with pytest.raises(OSError):
            extract_archive(good, out, max_total_bytes=10 * MB)
        assert (out / "prior.txt").read_bytes() == b"original"
        assert self._leftovers(tmp_path, out) == []
