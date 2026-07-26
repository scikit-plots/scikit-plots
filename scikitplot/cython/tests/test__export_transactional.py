# scikitplot/cython/tests/test__export_transactional.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-CACHE-004.

``export_cached`` did ``rmtree(dst)`` then ``copytree(src, dst)`` non-atomically,
so a failure mid-copy destroyed any prior export and left a half-written
destination.  The export now stages into a sibling and atomically swaps it in,
preserving the prior export on failure and leaving no staging/backup residue on
success.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest

from .. import _public
from .._public import export_cached


def _make_entry(root: Path, key: str, artifact: bytes) -> Path:
    entry = root / key
    entry.mkdir(parents=True)
    (entry / "artifact.so").write_bytes(artifact)
    (entry / "meta.json").write_text('{"kind":"module"}', encoding="utf-8")
    return entry


class TestSuccessfulExport:
    def test_fresh_export(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        _make_entry(root, "k", b"ARTIFACT")
        dest = tmp_path / "dest"
        out = export_cached("k", dest_dir=dest, cache_dir=root)
        assert out == (dest / "k")
        assert (out / "artifact.so").read_bytes() == b"ARTIFACT"
        assert (out / "meta.json").exists()

    def test_replaces_existing_export(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        _make_entry(root, "k", b"NEW")
        dest = tmp_path / "dest"
        (dest / "k").mkdir(parents=True)
        (dest / "k" / "stale.txt").write_text("stale")
        out = export_cached("k", dest_dir=dest, cache_dir=root)
        names = {p.name for p in out.iterdir()}
        assert "stale.txt" not in names
        assert (out / "artifact.so").read_bytes() == b"NEW"

    def test_no_staging_or_backup_leftovers(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        _make_entry(root, "k", b"A")
        dest = tmp_path / "dest"
        export_cached("k", dest_dir=dest, cache_dir=root)
        residue = [
            p.name
            for p in dest.iterdir()
            if p.name.startswith(".staging") or p.name.startswith(".backup")
        ]
        assert residue == []


class TestFailedExportRollsBack:
    def test_prior_export_preserved_on_failure(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        _make_entry(root, "k", b"NEW-GOOD")
        dest = tmp_path / "dest"
        # Pre-existing good export.
        (dest / "k").mkdir(parents=True)
        (dest / "k" / "artifact.so").write_bytes(b"OLD-GOOD")
        (dest / "k" / "meta.json").write_text("{}", encoding="utf-8")

        real_copytree = shutil.copytree

        def failing_copytree(s, d, *a, **k):
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / "partial").write_bytes(b"X")
            raise OSError("disk full mid-copy")

        with mock.patch.object(_public.shutil, "copytree", failing_copytree):
            with pytest.raises(OSError):
                export_cached("k", dest_dir=dest, cache_dir=root)

        # Prior export intact; no partial/staging residue.
        assert (dest / "k" / "artifact.so").read_bytes() == b"OLD-GOOD"
        assert (dest / "k" / "meta.json").exists()
        assert "partial" not in {p.name for p in (dest / "k").iterdir()}
        residue = [
            p.name
            for p in dest.iterdir()
            if p.name.startswith(".staging") or p.name.startswith(".backup")
        ]
        assert residue == []

    def test_no_prior_export_no_stray_dir_on_failure(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        _make_entry(root, "k", b"NEW")
        dest = tmp_path / "dest"  # no prior export of k

        def failing_copytree(s, d, *a, **k):
            Path(d).mkdir(parents=True, exist_ok=True)
            raise OSError("fail")

        with mock.patch.object(_public.shutil, "copytree", failing_copytree):
            with pytest.raises(OSError):
                export_cached("k", dest_dir=dest, cache_dir=root)

        # No half-written dest/k, no staging residue.
        assert not (dest / "k").exists()
        residue = [p.name for p in dest.iterdir() if p.name.startswith(".staging")]
        assert residue == []


class TestMissingKey:
    def test_missing_key_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        with pytest.raises(FileNotFoundError):
            export_cached("nope", dest_dir=tmp_path / "dest", cache_dir=root)
