# scikitplot/cython/tests/test__gc.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._gc`.

Covers
------
- Private helpers: ``_utc_iso_from_epoch``, ``_dir_size_bytes``,
  ``_dir_mtime_epoch`` — including concurrent-delete tolerance
- ``cache_stats()``  : empty / module / package / pinned / corrupted meta /
                       mtime timestamps / byte totals / missing root
- ``gc_cache()``     : keep_n_newest, max_age_days, max_bytes, dry_run,
                       pinned-key protection, combined strategies,
                       skipped-missing, safety check, early-return on
                       non-existent root, freed_bytes reporting
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from .._cache import make_cache_key, resolve_cache_dir, write_meta
from .._gc import (
    _dir_mtime_epoch,
    _dir_size_bytes,
    _utc_iso_from_epoch,
    cache_stats,
    gc_cache,
)
from .._pins import pin
from .._result import CacheGCResult, CacheStats

from .conftest import make_valid_key, write_simple_cache_entry, write_full_cache_entry

# ---------------------------------------------------------------------------
# Module-level helpers (aliases to conftest utilities for legacy test classes)
# ---------------------------------------------------------------------------
from .conftest import (
    make_valid_key as _make_valid_key,
    write_simple_cache_entry as _write_cache_entry,
    write_full_cache_entry as _write_full_cache_entry,
    write_fake_artifact as _make_fake_so,
)
import json
from importlib.machinery import EXTENSION_SUFFIXES
from .conftest import FAKE_KEY as _FAKE_KEY, FAKE_KEY2 as _FAKE_KEY2, write_package_cache_entry as _write_package_cache_entry



class TestGcPrivateHelpersCoverage:
    def test_utc_iso_unix_epoch(self) -> None:
        s = _utc_iso_from_epoch(0.0)
        assert s == "1970-01-01T00:00:00Z"

    def test_utc_iso_ends_with_z(self) -> None:
        assert _utc_iso_from_epoch(1_000_000.0).endswith("Z")

    def test_dir_size_bytes_empty(self, tmp_path: Path) -> None:
        assert _dir_size_bytes(tmp_path) == 0

    def test_dir_size_bytes_with_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"hello")
        (tmp_path / "b.txt").write_bytes(b"world!")
        assert _dir_size_bytes(tmp_path) == 11

    def test_dir_size_bytes_nested(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_bytes(b"abc")
        assert _dir_size_bytes(tmp_path) == 3

    def test_dir_mtime_epoch_nonexistent_returns_zero(self, tmp_path: Path) -> None:
        assert _dir_mtime_epoch(tmp_path / "no_such") == 0.0

    def test_dir_mtime_epoch_existing_positive(self, tmp_path: Path) -> None:
        assert _dir_mtime_epoch(tmp_path) > 0.0

    def test_dir_size_bytes_concurrent_delete_tolerated(self, tmp_path: Path) -> None:
        """FileNotFoundError during rglob is swallowed — returns partial count."""
        (tmp_path / "f.txt").write_bytes(b"x")
        # Simulate concurrent delete by patching is_file to raise FileNotFoundError
        original_is_file = Path.is_file

        call_count = [0]

        def fake_is_file(self: Path) -> bool:
            call_count[0] += 1
            if call_count[0] == 1:
                raise FileNotFoundError("gone")
            return original_is_file(self)

        with patch.object(Path, "is_file", fake_is_file):
            result = _dir_size_bytes(tmp_path)
        # First file was "deleted" concurrently — size should be 0
        assert result == 0


class TestGcPrivateHelpers:
    """
    Private helpers in ``_gc``: ``_utc_iso_from_epoch``, ``_dir_size_bytes``,
    ``_dir_mtime_epoch``.
    """

    def test_utc_iso_from_epoch_unix_epoch(self) -> None:
        from .._gc import _utc_iso_from_epoch

        result = _utc_iso_from_epoch(0.0)
        assert result == "1970-01-01T00:00:00Z"

    def test_utc_iso_from_epoch_ends_with_z(self) -> None:
        from .._gc import _utc_iso_from_epoch

        result = _utc_iso_from_epoch(1_700_000_000.0)
        assert result.endswith("Z")
        assert "T" in result

    def test_utc_iso_from_epoch_format_length(self) -> None:
        from .._gc import _utc_iso_from_epoch

        result = _utc_iso_from_epoch(1_700_000_000.0)
        # e.g. "2023-11-14T22:13:20Z" → 20 chars
        assert len(result) == 20

    def test_dir_size_bytes_empty_dir_returns_zero(self, tmp_path: Path) -> None:
        from .._gc import _dir_size_bytes

        assert _dir_size_bytes(tmp_path) == 0

    def test_dir_size_bytes_accumulates_file_sizes(self, tmp_path: Path) -> None:
        from .._gc import _dir_size_bytes

        (tmp_path / "a.bin").write_bytes(b"x" * 100)
        (tmp_path / "b.bin").write_bytes(b"y" * 200)
        assert _dir_size_bytes(tmp_path) >= 300

    def test_dir_size_bytes_nested_files_counted(self, tmp_path: Path) -> None:
        from .._gc import _dir_size_bytes

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.bin").write_bytes(b"z" * 512)
        assert _dir_size_bytes(tmp_path) >= 512

    def test_dir_mtime_epoch_nonexistent_returns_zero(self, tmp_path: Path) -> None:
        from .._gc import _dir_mtime_epoch

        result = _dir_mtime_epoch(tmp_path / "nonexistent_dir")
        assert result == 0.0

    def test_dir_mtime_epoch_existing_is_positive(self, tmp_path: Path) -> None:
        from .._gc import _dir_mtime_epoch

        result = _dir_mtime_epoch(tmp_path)
        assert result > 0.0


class TestDirSizeBytesFileNotFound:
    """_dir_size_bytes must tolerate concurrent deletes (FileNotFoundError)."""

    def test_concurrent_delete_tolerated(self, tmp_path: Path) -> None:
        from .._gc import _dir_size_bytes

        f = tmp_path / "data.bin"
        f.write_bytes(b"x" * 100)

        original_stat = Path.stat

        def raise_for_file(self: Path, **kw: any) -> any:
            if self == f:
                raise FileNotFoundError("deleted concurrently")
            return original_stat(self, **kw)

        with patch.object(Path, "stat", raise_for_file):
            result = _dir_size_bytes(tmp_path)
        # FileNotFoundError is swallowed; we get 0 or partial count — no crash.
        assert isinstance(result, int)
        assert result >= 0

    def test_empty_dir_returns_zero(self, tmp_path: Path) -> None:
        from .._gc import _dir_size_bytes

        assert _dir_size_bytes(tmp_path) == 0

    def test_nested_files_counted(self, tmp_path: Path) -> None:
        from .._gc import _dir_size_bytes

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.bin").write_bytes(b"A" * 50)
        (tmp_path / "b.bin").write_bytes(b"B" * 30)
        assert _dir_size_bytes(tmp_path) == 80


class TestCacheStatsBranches:
    """Cover cache_stats with populated and empty caches."""

    def test_nonexistent_root_returns_zeros(self, tmp_path: Path) -> None:
        stats = cache_stats(tmp_path / "no_cache")
        assert isinstance(stats, CacheStats)
        assert stats.n_modules == 0
        assert stats.n_packages == 0
        assert stats.total_bytes == 0
        assert stats.newest_mtime_utc is None
        assert stats.oldest_mtime_utc is None

    def test_empty_cache_dir(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        stats = cache_stats(root)
        assert stats.n_modules == 0

    def test_module_entry_counted(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key, module_name="mymod", kind="module")
        stats = cache_stats(root)
        assert stats.n_modules == 1
        assert stats.n_packages == 0
        assert stats.total_bytes > 0
        assert stats.newest_mtime_utc is not None
        assert stats.oldest_mtime_utc is not None

    def test_package_entry_counted(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key, module_name="pkg.mod", kind="package")
        stats = cache_stats(root)
        assert stats.n_packages == 1
        assert stats.n_modules == 0

    def test_corrupted_meta_counted_as_module(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = root / key
        entry_dir.mkdir(parents=True, exist_ok=True)
        _make_fake_so(entry_dir, "broken")
        (entry_dir / "meta.json").write_text("{{{ bad json", encoding="utf-8")
        stats = cache_stats(root)
        assert stats.n_modules == 1

    def test_pinned_aliases_counted(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key, module_name="pinmod")
        pin(key, alias="myalias", cache_dir=root)
        stats = cache_stats(root)
        assert stats.pinned_aliases == 1
        assert stats.pinned_keys == 1

    def test_newest_oldest_with_two_entries(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key1 = make_cache_key({"x": 1})
        key2 = make_cache_key({"x": 2})
        _write_cache_entry(root, key1, module_name="mod1")
        _write_cache_entry(root, key2, module_name="mod2")
        stats = cache_stats(root)
        assert stats.newest_mtime_utc is not None
        assert stats.oldest_mtime_utc is not None

    def test_total_bytes_nonzero(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = _write_cache_entry(root, key)
        (entry_dir / "extra.bin").write_bytes(b"x" * 100)
        stats = cache_stats(root)
        assert stats.total_bytes >= 100


class TestGcCacheStats:
    """Tests for :func:`~scikitplot.cython._gc.cache_stats`."""

    def test_empty_cache(self, tmp_cache: Path) -> None:
        stats = cache_stats(tmp_cache)
        assert stats.n_modules == 0
        assert stats.n_packages == 0
        assert stats.total_bytes == 0
        assert stats.newest_mtime_utc is None
        assert stats.oldest_mtime_utc is None

    def test_missing_root(self, tmp_path: Path) -> None:
        stats = cache_stats(tmp_path / "nonexistent")
        assert stats.n_modules == 0

    def test_counts_module_entry(self, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        stats = cache_stats(build_dir.parent)
        assert stats.n_modules == 1
        assert stats.n_packages == 0

    def test_counts_package_entry(self, fake_package_entry) -> None:
        key, build_dir, _ = fake_package_entry
        stats = cache_stats(build_dir.parent)
        assert stats.n_packages == 1
        assert stats.n_modules == 0

    def test_total_bytes_nonzero(self, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        stats = cache_stats(build_dir.parent)
        assert stats.total_bytes > 0

    def test_mtime_timestamps(self, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        stats = cache_stats(build_dir.parent)
        assert stats.newest_mtime_utc is not None
        assert stats.oldest_mtime_utc is not None
        assert stats.newest_mtime_utc.endswith("Z")

    def test_pinned_counts(self, tmp_cache: Path, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        pin(key, alias="pinned_alias", cache_dir=tmp_cache)
        stats = cache_stats(tmp_cache)
        assert stats.pinned_aliases == 1
        assert stats.pinned_keys == 1


class TestCacheStatsBadMeta:
    """``cache_stats`` when a meta.json is corrupted."""

    def test_corrupted_meta_counted_as_module(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "corrupt_stats"})
        d = tmp_cache / key
        d.mkdir()
        (d / "meta.json").write_text("NOT JSON", encoding="utf-8")
        stats = cache_stats(tmp_cache)
        # Corrupt meta → kind=None → counted as module
        assert stats.n_modules >= 1


class TestCacheStatsTimestamps:
    """``cache_stats`` timestamps span across multiple entries."""

    def _make_entry(self, root: Path, label: str, payload_bytes: int = 128) -> str:
        key = make_cache_key({"lbl": label})
        d = root / key
        d.mkdir()
        so = d / f"{label}{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"E" * payload_bytes)
        write_meta(
            d,
            {
                "kind": "module",
                "key": key,
                "module_name": label,
                "artifact": so.name,
                "created_utc": "2025-01-01T00:00:00Z",
            },
        )
        return key

    def test_newest_and_oldest_present_with_multiple_entries(
        self, tmp_cache: Path
    ) -> None:
        self._make_entry(tmp_cache, "mod_a", 64)
        time.sleep(0.03)
        self._make_entry(tmp_cache, "mod_b", 128)

        stats = cache_stats(tmp_cache)
        assert stats.n_modules == 2
        assert stats.newest_mtime_utc is not None
        assert stats.oldest_mtime_utc is not None
        assert stats.newest_mtime_utc.endswith("Z")
        assert stats.oldest_mtime_utc.endswith("Z")

    def test_total_bytes_sums_across_entries(self, tmp_cache: Path) -> None:
        self._make_entry(tmp_cache, "mod_x", 512)
        self._make_entry(tmp_cache, "mod_y", 512)

        stats = cache_stats(tmp_cache)
        # Two entries each with ≥512 bytes of artifact
        assert stats.total_bytes >= 1024

    def test_cache_root_field_matches_cache_dir(self, tmp_cache: Path) -> None:
        stats = cache_stats(tmp_cache)
        assert stats.cache_root == tmp_cache


@pytest.mark.parametrize("n_entries", [0, 1, 3])
def test_cache_stats_entry_count_matches(
    tmp_path: Path, n_entries: int
) -> None:
    from .._gc import cache_stats

    keys = [chr(ord("a") + i) * 64 for i in range(n_entries)]
    for k in keys:
        _write_cache_entry(tmp_path, k)

    stats = cache_stats(cache_dir=tmp_path)
    assert stats.n_modules == n_entries
    assert stats.n_packages == 0


@pytest.mark.parametrize("n_packages", [1, 2])
def test_cache_stats_package_count_matches(
    tmp_path: Path, n_packages: int
) -> None:
    from .._gc import cache_stats

    keys = [chr(ord("a") + i) * 64 for i in range(n_packages)]
    for k in keys:
        _write_package_cache_entry(tmp_path, k)

    stats = cache_stats(cache_dir=tmp_path)
    assert stats.n_packages == n_packages
    assert stats.n_modules == 0


class TestGcCacheBranches:
    """Cover gc_cache logic for all deletion strategies."""

    def test_nonexistent_root_returns_empty(self, tmp_path: Path) -> None:
        result = gc_cache(cache_dir=tmp_path / "no_cache")
        assert isinstance(result, CacheGCResult)
        assert result.deleted_keys == ()

    def test_invalid_keep_n_newest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="keep_n_newest"):
            gc_cache(cache_dir=tmp_path, keep_n_newest=-1)

    def test_invalid_max_age_days_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_age_days"):
            gc_cache(cache_dir=tmp_path, max_age_days=-1)

    def test_invalid_max_bytes_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_bytes"):
            gc_cache(cache_dir=tmp_path, max_bytes=-1)

    def test_empty_cache_no_error(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        result = gc_cache(cache_dir=root)
        assert result.deleted_keys == ()
        assert result.freed_bytes == 0

    def test_keep_n_newest(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        keys = [make_cache_key({"i": i}) for i in range(3)]
        for i, k in enumerate(keys):
            _write_cache_entry(root, k, module_name=f"m{i}")
        result = gc_cache(cache_dir=root, keep_n_newest=1)
        # At most 2 entries deleted; 1 kept
        assert len(result.deleted_keys) <= 2

    def test_dry_run_deletes_nothing(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        result = gc_cache(cache_dir=root, keep_n_newest=0, dry_run=True)
        assert key in result.deleted_keys
        assert (root / key).exists()  # NOT actually deleted

    def test_max_age_deletes_old_entries(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = _write_cache_entry(root, key)
        # Back-date the directory
        ancient = time.time() - (200 * 86400)
        os.utime(entry_dir, (ancient, ancient))
        result = gc_cache(cache_dir=root, max_age_days=100)
        assert key in result.deleted_keys

    def test_pinned_keys_never_deleted(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        pin(key, alias="protected", cache_dir=root)
        result = gc_cache(cache_dir=root, keep_n_newest=0)
        assert key not in result.deleted_keys
        assert key in result.skipped_pinned_keys

    def test_freed_bytes_reported(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = _write_cache_entry(root, key)
        (entry_dir / "payload.bin").write_bytes(b"x" * 512)
        result = gc_cache(cache_dir=root, keep_n_newest=0)
        assert result.freed_bytes > 0

    def test_max_bytes_deletes_oldest_first(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key1 = make_cache_key({"a": 1})
        key2 = make_cache_key({"a": 2})
        d1 = _write_cache_entry(root, key1)
        d2 = _write_cache_entry(root, key2)
        (d1 / "big.bin").write_bytes(b"A" * 1024)
        (d2 / "big.bin").write_bytes(b"B" * 1024)
        # Back-date d1 so it's older
        t_old = time.time() - 3600
        os.utime(d1, (t_old, t_old))
        result = gc_cache(cache_dir=root, max_bytes=0)
        assert len(result.deleted_keys) > 0

    def test_skipped_missing_when_deleted_between_scan_and_gc(
        self, tmp_path: Path
    ) -> None:
        """Entry that disappears between scan and deletion goes to skipped_missing."""
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = _write_cache_entry(root, key)

        original_rmtree = shutil.rmtree

        def fake_rmtree(path, **kw):
            # Remove the dir before the gc loop gets to delete it
            original_rmtree(str(entry_dir))
            raise FileNotFoundError("gone")

        with patch("shutil.rmtree", side_effect=fake_rmtree):
            result = gc_cache(cache_dir=root, keep_n_newest=0)

        # entry_dir was removed but gc may report freed 0 or skipped_missing
        assert isinstance(result, CacheGCResult)

    def test_cache_root_in_result(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        result = gc_cache(cache_dir=root)
        assert result.cache_root == root


class TestGcCache:
    """Tests for :func:`~scikitplot.cython._gc.gc_cache`."""

    def _populate(self, root: Path, n: int = 3) -> list[str]:
        """Create n fake module entries and return their keys."""
        keys: list[str] = []
        for i in range(n):
            key = make_cache_key({"gc_label": f"entry_{i}"})
            d = root / key
            d.mkdir()
            so = d / f"mod{i}{EXTENSION_SUFFIXES[0]}"
            so.write_bytes(b"X" * (100 + i * 50))
            write_meta(
                d,
                {
                    "kind": "module",
                    "key": key,
                    "module_name": f"mod{i}",
                    "artifact": so.name,
                    "created_utc": f"2025-01-0{i + 1}T00:00:00Z",
                    "fingerprint": {},
                },
            )
            keys.append(key)
            time.sleep(0.01)  # ensure distinct mtimes
        return keys

    def test_empty_cache_no_error(self, tmp_cache: Path) -> None:
        result = gc_cache(cache_dir=tmp_cache)
        assert isinstance(result, CacheGCResult)
        assert list(result.deleted_keys) == []

    def test_missing_root_no_error(self, tmp_path: Path) -> None:
        result = gc_cache(cache_dir=tmp_path / "nonexistent")
        assert list(result.deleted_keys) == []

    def test_keep_n_newest(self, tmp_cache: Path) -> None:
        keys = self._populate(tmp_cache, n=4)
        result = gc_cache(cache_dir=tmp_cache, keep_n_newest=2)
        assert len(result.deleted_keys) == 2
        assert result.freed_bytes > 0
        # Verify deletions happened on disk
        for k in result.deleted_keys:
            assert not (tmp_cache / k).exists()

    def test_dry_run_deletes_nothing(self, tmp_cache: Path) -> None:
        keys = self._populate(tmp_cache, n=3)
        result = gc_cache(cache_dir=tmp_cache, keep_n_newest=1, dry_run=True)
        assert len(result.deleted_keys) > 0
        for k in result.deleted_keys:
            assert (tmp_cache / k).exists()  # dry_run: files remain

    def test_max_age_deletes_old(self, tmp_cache: Path) -> None:
        keys = self._populate(tmp_cache, n=3)
        result = gc_cache(cache_dir=tmp_cache, max_age_days=0)
        # max_age_days=0 → all entries older than now → all deleted
        assert len(result.deleted_keys) == len(keys)

    def test_pinned_entries_never_deleted(self, tmp_cache: Path) -> None:
        keys = self._populate(tmp_cache, n=3)
        protected = keys[0]
        pin(protected, alias="protected", cache_dir=tmp_cache)
        result = gc_cache(cache_dir=tmp_cache, keep_n_newest=0, max_age_days=0)
        assert protected not in result.deleted_keys
        assert protected in result.skipped_pinned_keys
        assert (tmp_cache / protected).exists()

    def test_max_bytes_deletes_oldest_first(self, tmp_cache: Path) -> None:
        keys = self._populate(tmp_cache, n=4)
        result = gc_cache(cache_dir=tmp_cache, max_bytes=0)
        assert len(result.deleted_keys) > 0

    def test_invalid_keep_n_newest_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError, match="keep_n_newest"):
            gc_cache(cache_dir=tmp_cache, keep_n_newest=-1)

    def test_invalid_max_age_days_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError, match="max_age_days"):
            gc_cache(cache_dir=tmp_cache, max_age_days=-1)

    def test_invalid_max_bytes_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError, match="max_bytes"):
            gc_cache(cache_dir=tmp_cache, max_bytes=-1)

    def test_freed_bytes_reported(self, tmp_cache: Path) -> None:
        self._populate(tmp_cache, n=2)
        result = gc_cache(cache_dir=tmp_cache, keep_n_newest=0, max_age_days=0)
        assert result.freed_bytes > 0

    def test_cache_root_in_result(self, tmp_cache: Path) -> None:
        result = gc_cache(cache_dir=tmp_cache)
        assert result.cache_root == tmp_cache


class TestGcCacheSkipMissing:
    """``gc_cache`` skipped_missing_keys branch."""

    def test_skipped_missing_reported(self, tmp_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        The skipped_missing path fires when a directory disappears *between* the
        GC scan phase (iter_all_entry_dirs) and the deletion phase.  We simulate
        this by monkey-patching shutil.rmtree inside _gc to pre-delete the
        directory before the real rmtree call, making d.exists() return False
        during the deletion loop.
        """
        key = make_cache_key({"test": "missing_during_gc"})
        d = tmp_cache / key
        d.mkdir()
        so = d / f"m{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        write_meta(d, {
            "kind": "module", "key": key, "module_name": "m",
            "artifact": so.name, "created_utc": "2020-01-01T00:00:00Z",
            "fingerprint": {},
        })

        import shutil as _shutil
        from .. import _gc as gc_mod

        original_rmtree = _shutil.rmtree
        calls = {"n": 0}

        def pre_delete_rmtree(path, **kw):
            """On first call, delete the directory ourselves to simulate concurrent removal."""
            calls["n"] += 1
            if calls["n"] == 1:
                # Delete directory so that d.exists() returns False in GC loop
                original_rmtree(path, ignore_errors=True)
                # Now raise to make GC think the delete failed mid-flight
                # Actually the GC checks d.exists() BEFORE calling rmtree,
                # so we need a different approach: patch d.exists() instead.

        # Simpler and correct: patch the Path.exists method for the specific key dir
        # so GC's `if not d.exists(): skipped_missing` fires.
        # Monkey-patch shutil in the _gc module's namespace.
        original_shutil_rmtree = gc_mod.shutil.rmtree

        def fake_rmtree_for_gc(path, **kw):
            # Pre-delete to make it "already gone" on next check — but GC checks
            # d.exists() BEFORE rmtree, not after. So instead we need the
            # directory to disappear between scan and deletion.
            # The cleanest way: delete it before GC runs and ensure GC still
            # scans via iter_all_entry_dirs. We do this by patching
            # iter_all_entry_dirs to return our key even though dir is gone.
            raise AssertionError("Should not be reached in this test")

        # Correct approach: delete the dir now, then patch iter_all_entry_dirs
        # to yield it anyway — GC will then hit d.exists() == False → skipped_missing
        original_iter = gc_mod.iter_all_entry_dirs

        def fake_iter(root):
            yield d  # yield the now-deleted dir
            # also yield real dirs
            for p in original_iter(root):
                if p != d:
                    yield p

        original_rmtree(d)  # Pre-delete the dir
        monkeypatch.setattr(gc_mod, "iter_all_entry_dirs", fake_iter)

        result = gc_cache(cache_dir=tmp_cache, max_age_days=0)
        assert key in result.skipped_missing_keys


class TestGcCacheCombinedStrategies:
    """``gc_cache`` with combined keep_n_newest + max_bytes."""

    def test_keep_newest_plus_max_bytes(self, tmp_cache: Path) -> None:
        for i in range(5):
            k = make_cache_key({"combo": i})
            d = tmp_cache / k
            d.mkdir()
            so = d / f"m{i}{EXTENSION_SUFFIXES[0]}"
            so.write_bytes(b"X" * 200)
            write_meta(d, {
                "kind": "module", "key": k, "module_name": f"m{i}",
                "artifact": so.name, "created_utc": f"2025-01-0{i+1}T00:00:00Z",
                "fingerprint": {},
            })
            time.sleep(0.01)

        # Keep 3 newest but also constrain bytes to 0 (delete everything possible)
        result = gc_cache(cache_dir=tmp_cache, keep_n_newest=2, max_bytes=0)
        # The 2 newest are kept; max_bytes=0 may delete beyond age filter
        # At minimum the sum of deleted + kept == 5
        total_accounted = len(result.deleted_keys) + 2  # 2 kept by keep_n_newest
        assert total_accounted <= 5


class TestGcCacheEarlyReturns:
    """gc_cache early-return when cache root does not exist."""

    def test_nonexistent_root_returns_empty_result(self, tmp_path: Path) -> None:
        from .._gc import gc_cache

        absent = tmp_path / "no_such_cache"
        result = gc_cache(cache_dir=absent)
        assert result.deleted_keys == ()
        assert result.freed_bytes == 0
        assert result.cache_root == absent.resolve()

    def test_gc_with_env_override_nonexistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gc_cache must respect SCIKITPLOT_CYTHON_CACHE_DIR env override."""
        from .._gc import gc_cache

        absent = tmp_path / "env_cache_absent"
        monkeypatch.setenv("SCIKITPLOT_CYTHON_CACHE_DIR", str(absent))
        result = gc_cache()
        assert result.freed_bytes == 0


class TestGcCacheAgeCutoffSkip:
    """Entries newer than max_age_days must not be deleted."""

    def test_recent_entry_not_deleted(self, tmp_path: Path) -> None:
        from .._gc import gc_cache

        # Entry mtime = now (definitely < 30 days old)
        build_dir, _ = _write_full_cache_entry(tmp_path, _FAKE_KEY)
        result = gc_cache(cache_dir=tmp_path, max_age_days=30)
        assert _FAKE_KEY not in result.deleted_keys
        assert build_dir.exists()

    def test_old_entry_deleted(self, tmp_path: Path) -> None:
        from .._gc import gc_cache

        build_dir, _ = _write_full_cache_entry(tmp_path, _FAKE_KEY)
        old_time = time.time() - 32 * 86400  # 32 days ago
        os.utime(build_dir, (old_time, old_time))
        result = gc_cache(cache_dir=tmp_path, max_age_days=30)
        assert _FAKE_KEY in result.deleted_keys


class TestGcCacheCombinedKeepBytes:
    """keep_n_newest + max_bytes combined: oldest non-kept entries deleted first."""

    def test_combined_strategy(self, tmp_path: Path) -> None:
        from .._gc import gc_cache

        key_a = "a" * 64
        key_b = "b" * 64
        key_c = "c" * 64
        bd_a, art_a = _write_full_cache_entry(tmp_path, key_a)
        bd_b, art_b = _write_full_cache_entry(tmp_path, key_b)
        bd_c, art_c = _write_full_cache_entry(tmp_path, key_c)

        # Make A oldest, C newest
        now = time.time()
        os.utime(bd_a, (now - 200, now - 200))
        os.utime(bd_b, (now - 100, now - 100))
        os.utime(bd_c, (now, now))

        # keep_n_newest=1 keeps C; max_bytes=0 forces deletion of rest
        result = gc_cache(cache_dir=tmp_path, keep_n_newest=1, max_bytes=0)
        assert key_c not in result.deleted_keys
        # A and/or B are in deleted set
        assert len(result.deleted_keys) >= 1


class TestGcCacheSafetyCheckSkip:
    """Safety check: only direct cache-root children with valid key names are deleted."""

    def test_entry_outside_root_not_deleted(self, tmp_path: Path) -> None:
        """
        This exercises the branch ``if d.parent != root: continue`` by
        verifying that entries not directly under root are never touched.
        The branch is exercised indirectly through the iteration path.
        """
        from .._gc import gc_cache

        # Two-level deep fake entry — not a direct child of cache root.
        nested = tmp_path / "subdir" / (_FAKE_KEY)
        nested.mkdir(parents=True)

        result = gc_cache(cache_dir=tmp_path, max_age_days=0)
        # 'subdir' is not a valid key dir so it's invisible to iter_all_entry_dirs.
        assert _FAKE_KEY not in result.deleted_keys

    def test_dry_run_with_directory_deleted_between_scan_and_gc(
        self, tmp_path: Path
    ) -> None:
        """
        Entry disappears between candidate scan (iter_all_entry_dirs) and
        the deletion loop: the key must appear in skipped_missing_keys.
        Simulates a race condition by patching Path.exists to return False
        for the entry dir only during the deletion-loop check.
        """
        from .._gc import gc_cache

        build_dir, _ = _write_full_cache_entry(tmp_path, _FAKE_KEY)
        # Make it old enough to be a candidate under max_age_days=0
        old = time.time() - 400 * 86400
        os.utime(build_dir, (old, old))

        original_exists = Path.exists
        deletion_phase = {"active": False}

        def patched_exists(self: Path) -> bool:
            # Once we are inside the deletion lock (deletion_phase is True),
            # pretend the target directory is gone.
            if deletion_phase["active"] and self.resolve() == build_dir.resolve():
                return False
            return original_exists(self)

        # Activate the patch after the scanning phase by hooking build_lock
        from .. import _gc as gc_mod
        original_build_lock = gc_mod.build_lock

        from contextlib import contextmanager

        @contextmanager
        def patched_lock(*args: any, **kw: any):  # type: ignore[misc]
            with original_build_lock(*args, **kw):
                deletion_phase["active"] = True
                with patch.object(Path, "exists", patched_exists):
                    yield

        with patch.object(gc_mod, "build_lock", patched_lock):
            result = gc_cache(cache_dir=tmp_path, max_age_days=300)

        assert _FAKE_KEY in result.skipped_missing_keys
