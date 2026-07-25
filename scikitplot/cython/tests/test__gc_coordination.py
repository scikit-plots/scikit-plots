# scikitplot/cython/tests/test__gc_coordination.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-GC-001.

Original behaviour: ``gc_cache`` snapshotted pins and entries *before* taking
the root GC lock and never coordinated with per-entry build locks, so an active
build or a newly pinned entry could be deleted.  ``purge_cache`` took no lock
and recursively removed the entire root, destroying active builds and pins.

These tests pin the corrected coordination contract:

- an entry whose per-key build lock is held is reported in
  ``skipped_active_keys`` and never deleted;
- a pin created during the GC transaction is honoured;
- ``purge_cache`` refuses to run while any build lock is held.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from .._gc import gc_cache
from .._public import purge_cache
from .._pins import pin


KEY_A = "a" * 64
KEY_B = "b" * 64


def _make_entry(root: Path, key: str, *, age_days: float = 10.0) -> Path:
    """Create a minimal, old cache entry directory under ``root``."""
    d = root / key
    d.mkdir(parents=True)
    (d / "artifact.txt").write_text("x", encoding="utf-8")
    old = os.stat(d).st_mtime - age_days * 86400.0
    os.utime(d, (old, old))
    return d


class TestGcCoordination:
    def test_active_build_lock_is_not_deleted(self, tmp_path: Path) -> None:
        """An entry with a held build lock is skipped as active, not deleted."""
        root = tmp_path / "cache"
        root.mkdir()
        _make_entry(root, KEY_A)
        _make_entry(root, KEY_B)

        # Simulate an active build on KEY_A by holding its per-key lock dir.
        (root / f"{KEY_A}.lock").mkdir()

        res = gc_cache(cache_dir=root, max_age_days=0)

        assert KEY_A in res.skipped_active_keys, "active build was not protected"
        assert KEY_A not in res.deleted_keys
        assert (root / KEY_A).exists()
        # KEY_B has no active lock and is old → deleted.
        assert KEY_B in res.deleted_keys
        assert not (root / KEY_B).exists()

    def test_pin_created_before_gc_is_honoured(self, tmp_path: Path) -> None:
        """A pinned entry is preserved and reported as skipped_pinned."""
        root = tmp_path / "cache"
        root.mkdir()
        _make_entry(root, KEY_A)
        pin(KEY_A, alias="keep_me", cache_dir=root)

        res = gc_cache(cache_dir=root, max_age_days=0)

        assert KEY_A in res.skipped_pinned_keys
        assert KEY_A not in res.deleted_keys
        assert (root / KEY_A).exists()

    def test_dry_run_reports_without_deleting(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        _make_entry(root, KEY_A)
        res = gc_cache(cache_dir=root, max_age_days=0, dry_run=True)
        assert KEY_A in res.deleted_keys  # reported
        assert (root / KEY_A).exists()    # but not actually removed


class TestPurgeCoordination:
    def test_purge_refuses_while_build_active(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        _make_entry(root, KEY_A)
        (root / f"{KEY_A}.lock").mkdir()  # active build

        with pytest.raises(RuntimeError, match="builds are active"):
            purge_cache(root)

        # Nothing was deleted.
        assert (root / KEY_A).exists()

    def test_purge_removes_root_when_idle(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        _make_entry(root, KEY_A)
        purge_cache(root)
        assert not root.exists()

    def test_purge_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            purge_cache(tmp_path / "nope")
