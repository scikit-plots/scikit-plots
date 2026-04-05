# scikitplot/cython/tests/test__result.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`scikitplot.cython._result`.

Covers
------
- ``BuildResult``        : construction, slots, frozen, repr, source_sha256
- ``PackageBuildResult`` : modules property, frozen, repr stability
- ``CacheStats``         : defaults, all fields, frozen, repr
- ``CacheGCResult``      : defaults, all fields, frozen, repr
- ``CacheEntry``         : default construction, frozen, repr
- ``PackageCacheEntry``  : default construction, frozen, repr
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from .._result import (
    BuildResult,
    CacheGCResult,
    CacheStats,
    PackageBuildResult,
)
from .._cache import CacheEntry, PackageCacheEntry
import os
from .._gc import cache_stats
from .conftest import FAKE_KEY as _FAKE_KEY, FAKE_KEY2 as _FAKE_KEY2


class TestBuildResult:
    """Unit tests for :class:`scikitplot.cython._result.BuildResult`."""

    def test_default_construction(self) -> None:
        r = BuildResult()
        assert r.key == ""
        assert r.module_name == ""
        assert r.used_cache is False
        assert r.created_utc is None
        assert r.fingerprint is None
        assert r.source_sha256 is None
        assert isinstance(r.meta, dict)
        assert isinstance(r.module, ModuleType)

    def test_frozen(self) -> None:
        r = BuildResult()
        with pytest.raises((TypeError, AttributeError)):
            r.key = "new"  # type: ignore[misc]

    def test_slots(self) -> None:
        r = BuildResult()
        with pytest.raises(AttributeError):
            r.__dict__  # slots=True means no __dict__

    def test_repr_contains_fields(self) -> None:
        r = BuildResult(key="abc", module_name="mymod")
        rep = repr(r)
        assert "abc" in rep
        assert "mymod" in rep

    def test_full_construction(self) -> None:
        mod = ModuleType("testmod")
        path = Path("/tmp/test.so")
        r = BuildResult(
            module=mod,
            key="a" * 64,
            module_name="testmod",
            build_dir=path.parent,
            artifact_path=path,
            used_cache=True,
            created_utc="2025-01-01T00:00:00Z",
            fingerprint={"python": "3.12"},
            source_sha256="deadbeef",
            meta={"kind": "module"},
        )
        assert r.used_cache is True
        assert r.source_sha256 == "deadbeef"
        assert r.fingerprint == {"python": "3.12"}


class TestBuildResultSourceSha256:
    """``BuildResult.source_sha256`` is correctly stored and defaults to None."""

    def test_source_sha256_stored(self) -> None:
        digest = "a" * 64
        r = BuildResult(source_sha256=digest)
        assert r.source_sha256 == digest

    def test_source_sha256_none_by_default(self) -> None:
        assert BuildResult().source_sha256 is None

    def test_source_sha256_is_str(self) -> None:
        r = BuildResult(source_sha256="deadbeef" * 8)
        assert isinstance(r.source_sha256, str)


class TestPackageBuildResult:
    """Unit tests for :class:`scikitplot.cython._result.PackageBuildResult`."""

    def test_default_construction(self) -> None:
        r = PackageBuildResult()
        assert r.package_name == ""
        assert r.key == ""
        assert list(r.results) == []

    def test_modules_property(self) -> None:
        m1 = ModuleType("m1")
        m2 = ModuleType("m2")
        r1 = BuildResult(module=m1)
        r2 = BuildResult(module=m2)
        pkg = PackageBuildResult(results=(r1, r2))
        mods = pkg.modules
        assert list(mods) == [m1, m2]

    def test_frozen(self) -> None:
        pkg = PackageBuildResult()
        with pytest.raises((TypeError, AttributeError)):
            pkg.package_name = "new"  # type: ignore[misc]

    def test_repr_is_stable(self) -> None:
        pkg = PackageBuildResult(package_name="mypkg", key="b" * 64)
        assert "mypkg" in repr(pkg)


class TestCacheStatsDataclass:
    """Unit tests for :class:`scikitplot.cython._result.CacheStats`."""

    def test_default_zeros(self) -> None:
        s = CacheStats()
        assert s.n_modules == 0
        assert s.n_packages == 0
        assert s.total_bytes == 0
        assert s.pinned_aliases == 0
        assert s.pinned_keys == 0
        assert s.newest_mtime_utc is None
        assert s.oldest_mtime_utc is None

    def test_repr_all_fields(self) -> None:
        s = CacheStats(n_modules=3, n_packages=1, total_bytes=4096)
        rep = repr(s)
        assert "3" in rep
        assert "4096" in rep


class TestCacheStatsAllFields:
    """``CacheStats`` dataclass: every field is correctly stored and readable."""

    def test_cache_root_field_set_by_cache_stats(self, tmp_cache: Path) -> None:
        stats = cache_stats(tmp_cache)
        assert stats.cache_root == tmp_cache

    def test_cache_root_for_nonexistent_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "nowhere"
        stats = cache_stats(missing)
        assert isinstance(stats.cache_root, Path)

    def test_all_fields_manually_constructed(self, tmp_cache: Path) -> None:
        s = CacheStats(
            cache_root=tmp_cache,
            n_modules=5,
            n_packages=3,
            total_bytes=8192,
            pinned_aliases=4,
            pinned_keys=3,
            newest_mtime_utc="2025-06-01T00:00:00Z",
            oldest_mtime_utc="2024-01-01T00:00:00Z",
        )
        assert s.cache_root == tmp_cache
        assert s.n_modules == 5
        assert s.n_packages == 3
        assert s.total_bytes == 8192
        assert s.pinned_aliases == 4
        assert s.pinned_keys == 3
        assert s.newest_mtime_utc == "2025-06-01T00:00:00Z"
        assert s.oldest_mtime_utc == "2024-01-01T00:00:00Z"

    def test_frozen(self) -> None:
        s = CacheStats()
        with pytest.raises((TypeError, AttributeError)):
            s.n_modules = 99  # type: ignore[misc]


class TestCacheGCResult:
    """Unit tests for :class:`scikitplot.cython._result.CacheGCResult`."""

    def test_default_empty(self) -> None:
        r = CacheGCResult()
        assert list(r.deleted_keys) == []
        assert r.freed_bytes == 0

    def test_repr_stable(self) -> None:
        r = CacheGCResult(
            deleted_keys=("abc",),
            skipped_pinned_keys=("def",),
            freed_bytes=1024,
        )
        rep = repr(r)
        assert "abc" in rep
        assert "1024" in rep


class TestCacheGCResultAllFields:
    """``CacheGCResult`` dataclass: ``skipped_missing_keys`` and ``cache_root``."""

    def test_skipped_missing_keys_populated(self) -> None:
        r = CacheGCResult(skipped_missing_keys=("aaa", "bbb"), freed_bytes=0)
        assert "aaa" in r.skipped_missing_keys
        assert "bbb" in r.skipped_missing_keys

    def test_skipped_missing_keys_empty_by_default(self) -> None:
        r = CacheGCResult()
        assert list(r.skipped_missing_keys) == []

    def test_cache_root_set_on_gc_result(self, tmp_cache: Path) -> None:
        from .._gc import gc_cache as _gc

        result = _gc(cache_dir=tmp_cache)
        assert isinstance(result.cache_root, Path)

    def test_all_fields_manually_constructed(self, tmp_cache: Path) -> None:
        r = CacheGCResult(
            cache_root=tmp_cache,
            deleted_keys=("k1", "k2"),
            skipped_pinned_keys=("k3",),
            skipped_missing_keys=("k4",),
            freed_bytes=2048,
        )
        assert r.cache_root == tmp_cache
        assert "k1" in r.deleted_keys
        assert "k3" in r.skipped_pinned_keys
        assert "k4" in r.skipped_missing_keys
        assert r.freed_bytes == 2048

    def test_frozen(self) -> None:
        r = CacheGCResult()
        with pytest.raises((TypeError, AttributeError)):
            r.freed_bytes = 99  # type: ignore[misc]


class TestCacheEntryDataclass:
    """Tests for :class:`scikitplot.cython._cache.CacheEntry`."""

    def test_default_construction(self) -> None:
        e = CacheEntry()
        assert e.key == ""
        assert e.module_name == ""

    def test_frozen(self) -> None:
        e = CacheEntry()
        with pytest.raises((TypeError, AttributeError)):
            e.key = "new"  # type: ignore[misc]

    def test_repr_contains_key(self) -> None:
        e = CacheEntry(key="abc123")
        assert "abc123" in repr(e)


class TestPackageCacheEntryDataclass:
    """Tests for :class:`scikitplot.cython._cache.PackageCacheEntry`."""

    def test_default_construction(self) -> None:
        e = PackageCacheEntry()
        assert e.package_name == ""
        assert e.modules == ()
        assert e.artifacts == ()

    def test_frozen(self) -> None:
        e = PackageCacheEntry()
        with pytest.raises((TypeError, AttributeError)):
            e.package_name = "new"  # type: ignore[misc]

    def test_repr_stable(self) -> None:
        e = PackageCacheEntry(package_name="mypkg")
        assert "mypkg" in repr(e)


class TestResultReprBranches:
    def test_cache_stats_repr_contains_fields(self) -> None:
        stats = CacheStats(
            cache_root=Path("/tmp/cache"),
            n_modules=2,
            n_packages=1,
            total_bytes=1024,
            pinned_aliases=0,
            pinned_keys=0,
            newest_mtime_utc="2025-01-02T00:00:00Z",
            oldest_mtime_utc="2025-01-01T00:00:00Z",
        )
        r = repr(stats)
        assert "n_modules=2" in r
        assert "n_packages=1" in r

    def test_cache_gc_result_repr_contains_fields(self) -> None:
        result = CacheGCResult(
            cache_root=Path("/tmp/cache"),
            deleted_keys=("abc",),
            skipped_pinned_keys=(),
            skipped_missing_keys=(),
            freed_bytes=512,
        )
        r = repr(result)
        assert "freed_bytes=512" in r

    def test_cache_stats_frozen(self) -> None:
        stats = CacheStats(
            cache_root=Path("/tmp"),
            n_modules=0,
            n_packages=0,
            total_bytes=0,
            pinned_aliases=0,
            pinned_keys=0,
            newest_mtime_utc=None,
            oldest_mtime_utc=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            stats.n_modules = 99  # type: ignore[misc]

    def test_cache_gc_result_frozen(self) -> None:
        result = CacheGCResult(
            cache_root=Path("/tmp"),
            deleted_keys=(),
            skipped_pinned_keys=(),
            skipped_missing_keys=(),
            freed_bytes=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.freed_bytes = 99  # type: ignore[misc]


class TestReprBuildResult:
    """BuildResult.__repr__ must serialise every declared field."""

    def test_repr_default(self) -> None:
        from .._result import BuildResult

        r = BuildResult()
        s = repr(r)
        assert "BuildResult(" in s
        assert "key=" in s
        assert "module_name=" in s
        assert "used_cache=" in s

    def test_repr_with_fingerprint(self) -> None:
        from .._result import BuildResult

        r = BuildResult(key="abc", fingerprint={"python": "3.12"}, source_sha256="d1")
        s = repr(r)
        assert "fingerprint=" in s
        assert "source_sha256=" in s

    def test_repr_none_fingerprint(self) -> None:
        from .._result import BuildResult

        r = BuildResult(fingerprint=None)
        s = repr(r)
        assert "fingerprint=None" in s

    def test_repr_with_meta(self) -> None:
        from .._result import BuildResult

        r = BuildResult(meta={"k": "v"})
        s = repr(r)
        assert "meta=" in s


class TestReprPackageBuildResult:
    """PackageBuildResult.__repr__ must serialise every declared field."""

    def test_repr_default(self) -> None:
        from .._result import PackageBuildResult

        r = PackageBuildResult()
        s = repr(r)
        assert "PackageBuildResult(" in s
        assert "package_name=" in s
        assert "results=" in s

    def test_repr_with_fingerprint(self) -> None:
        from .._result import PackageBuildResult

        r = PackageBuildResult(fingerprint={"cython": "3.x"})
        s = repr(r)
        assert "fingerprint=" in s

    def test_repr_none_fingerprint(self) -> None:
        from .._result import PackageBuildResult

        r = PackageBuildResult(fingerprint=None)
        s = repr(r)
        assert "fingerprint=None" in s


class TestReprCacheStats:
    """CacheStats.__repr__ must list all fields."""

    def test_repr_default(self) -> None:
        from .._result import CacheStats

        s = repr(CacheStats())
        assert "CacheStats(" in s
        assert "n_modules=" in s
        assert "n_packages=" in s
        assert "pinned_aliases=" in s

    def test_repr_populated(self) -> None:
        from .._result import CacheStats

        s = repr(
            CacheStats(
                n_modules=3,
                n_packages=1,
                total_bytes=4096,
                pinned_aliases=2,
                pinned_keys=2,
                newest_mtime_utc="2024-01-01T00:00:00Z",
                oldest_mtime_utc="2023-01-01T00:00:00Z",
            )
        )
        assert "n_modules=3" in s
        assert "newest_mtime_utc=" in s


class TestReprCacheGCResult:
    """CacheGCResult.__repr__ must list all fields."""

    def test_repr_default(self) -> None:
        from .._result import CacheGCResult

        s = repr(CacheGCResult())
        assert "CacheGCResult(" in s
        assert "deleted_keys=" in s
        assert "freed_bytes=" in s

    def test_repr_populated(self) -> None:
        from .._result import CacheGCResult

        s = repr(
            CacheGCResult(
                deleted_keys=(_FAKE_KEY,),
                skipped_pinned_keys=(_FAKE_KEY2,),
                skipped_missing_keys=(),
                freed_bytes=512,
            )
        )
        assert "freed_bytes=512" in s
        assert "skipped_pinned_keys=" in s
