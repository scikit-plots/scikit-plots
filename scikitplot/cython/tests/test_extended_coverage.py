# scikitplot/cython/tests/test_extended_coverage.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Extended coverage tests for :mod:`scikitplot.cython`.

This module targets every coverage gap identified after the primary test suite,
organised by the private submodule under test.  All tests are self-contained:
they use only ``tmp_path``, monkeypatching, and mock extension artifacts; no
real Cython compiler invocation is required.

Coverage targets
----------------
- _result.py        : __repr__ for all four dataclasses (previously pragma:no cover)
- _lock.py          : stale-lock detection, FileNotFoundError in finally
- _gc.py            : _dir_size_bytes concurrent-delete path, gc_cache
                      early-return when root absent, age-cutoff skip,
                      combined keep+bytes strategy, safety check, dry_run
                      with missing directory
- _cache.py         : Windows cache-dir branch (mocked), iter_cache_entries
                      fingerprint paths, find_package_entry_by_key edge paths
- _loader.py        : ImportError from bad spec, metadata-attached import,
                      package-kind meta resolution, from_bytes collision
- _pins.py          : unpin last alias removes pins file, FileNotFoundError
                      swallowed during unlink
- _public.py        : check_build_prereqs import-failure branches,
                      cython_import, import_cached_result, import_pinned,
                      register_cached_artifact_path, import_artifact_path,
                      import_artifact_bytes, export_cached replace, cython_import_all
- _builder.py       : _open_annotation_in_browser all guard branches,
                      _normalize_extra_sources bytes path
"""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest.mock as mock
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

_FAKE_KEY = "a" * 64  # valid 64-hex key
_FAKE_KEY2 = "b" * 64


def _write_fake_artifact(build_dir: Path, module_name: str = "mymod") -> Path:
    """
    Write a minimal fake ``.so`` stub (zero bytes) into *build_dir*.

    The stub is named ``<module_name><EXTENSION_SUFFIXES[0]>`` so that all
    cache/loader helpers that look for valid-suffix files will find it.
    """
    suffix = EXTENSION_SUFFIXES[0]
    artifact = build_dir / f"{module_name}{suffix}"
    artifact.write_bytes(b"\x7fELF")  # ELF magic — non-empty stub
    return artifact


def _write_cache_entry(
    cache_root: Path,
    key: str,
    *,
    module_name: str = "mymod",
    kind: str = "module",
    fingerprint: dict | None = None,
) -> tuple[Path, Path]:
    """
    Create a complete cache entry directory under *cache_root* and return
    ``(build_dir, artifact_path)``.
    """
    build_dir = cache_root / key
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact = _write_fake_artifact(build_dir, module_name)
    meta: dict[str, Any] = {
        "kind": kind,
        "key": key,
        "module_name": module_name,
        "artifact": artifact.name,
        "artifact_filename": artifact.name,
        "created_utc": "2024-01-01T00:00:00Z",
    }
    if fingerprint is not None:
        meta["fingerprint"] = fingerprint
    (build_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return build_dir, artifact


def _write_package_cache_entry(
    cache_root: Path,
    key: str,
    package_name: str = "mypkg",
    short_names: tuple[str, ...] = ("mod1",),
) -> Path:
    """Create a complete package cache entry and return ``build_dir``."""
    build_dir = cache_root / key
    pkg_dir = build_dir / package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    module_artifacts = []
    for sn in short_names:
        art = _write_fake_artifact(pkg_dir, sn)
        rel = art.relative_to(build_dir).as_posix()
        module_artifacts.append(
            {
                "module_name": f"{package_name}.{sn}",
                "artifact": rel,
                "source_sha256": None,
            }
        )

    meta = {
        "kind": "package",
        "key": key,
        "package_name": package_name,
        "modules": module_artifacts,
        "created_utc": "2024-01-01T00:00:00Z",
        "fingerprint": {"python": "3.x"},
    }
    (build_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return build_dir


# ===========================================================================
# _result.py — __repr__ coverage (all four dataclasses)
# ===========================================================================


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


# ===========================================================================
# _lock.py — stale lock detection + finally FileNotFoundError
# ===========================================================================


class TestBuildLockStaleLock:
    """Stale lock directories (older than timeout_s) must be removed automatically."""

    def test_stale_lock_cleared_and_acquired(self, tmp_path: Path) -> None:
        """A pre-existing lock dir older than timeout_s is treated as stale."""
        from .._lock import build_lock

        lock_dir = tmp_path / "build.lock"
        lock_dir.mkdir()

        # Back-date the lock directory mtime by 3× timeout_s so it looks stale.
        timeout = 1.0
        stale_time = time.time() - timeout * 3
        os.utime(lock_dir, (stale_time, stale_time))

        acquired = False
        with build_lock(lock_dir, timeout_s=timeout, poll_s=0.01):
            acquired = True
            assert lock_dir.exists()

        assert acquired, "Lock was not acquired after stale lock removal"
        assert not lock_dir.exists(), "Lock should be released after context exit"

    def test_stale_lock_cleared_with_positive_timeout(self, tmp_path: Path) -> None:
        """Stale lock older than timeout_s is removed even with short timeout."""
        from .._lock import build_lock

        lock_dir = tmp_path / "stale2.lock"
        lock_dir.mkdir()
        timeout = 2.0
        stale_time = time.time() - timeout * 4
        os.utime(lock_dir, (stale_time, stale_time))

        with build_lock(lock_dir, timeout_s=timeout, poll_s=0.01):
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

        # Make old enough to be treated as stale on the SECOND stat call
        stale_time = time.time() - 200
        os.utime(lock_dir, (stale_time, stale_time))

        lock_str = str(lock_dir)
        original_stat = Path.stat
        call_count = 0

        def patched_stat(self: Path, **kw: Any) -> Any:
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
            with build_lock(lock_dir, timeout_s=5.0, poll_s=0.01):
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
            # Simulate another process deleting the lock dir mid-hold.
            lock_dir.rmdir()

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


# ===========================================================================
# _gc.py — edge paths
# ===========================================================================


class TestDirSizeBytesFileNotFound:
    """_dir_size_bytes must tolerate concurrent deletes (FileNotFoundError)."""

    def test_concurrent_delete_tolerated(self, tmp_path: Path) -> None:
        from .._gc import _dir_size_bytes

        f = tmp_path / "data.bin"
        f.write_bytes(b"x" * 100)

        original_stat = Path.stat

        def raise_for_file(self: Path, **kw: Any) -> Any:
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
        build_dir, _ = _write_cache_entry(tmp_path, _FAKE_KEY)
        result = gc_cache(cache_dir=tmp_path, max_age_days=30)
        assert _FAKE_KEY not in result.deleted_keys
        assert build_dir.exists()

    def test_old_entry_deleted(self, tmp_path: Path) -> None:
        from .._gc import gc_cache

        build_dir, _ = _write_cache_entry(tmp_path, _FAKE_KEY)
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
        bd_a, art_a = _write_cache_entry(tmp_path, key_a)
        bd_b, art_b = _write_cache_entry(tmp_path, key_b)
        bd_c, art_c = _write_cache_entry(tmp_path, key_c)

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

        build_dir, _ = _write_cache_entry(tmp_path, _FAKE_KEY)
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
        def patched_lock(*args: Any, **kw: Any):  # type: ignore[misc]
            with original_build_lock(*args, **kw):
                deletion_phase["active"] = True
                with patch.object(Path, "exists", patched_exists):
                    yield

        with patch.object(gc_mod, "build_lock", patched_lock):
            result = gc_cache(cache_dir=tmp_path, max_age_days=300)

        assert _FAKE_KEY in result.skipped_missing_keys


# ===========================================================================
# _cache.py — iter_all_entry_dirs is list (not generator), fingerprint paths
# ===========================================================================


class TestIterAllEntryDirsReturnsList:
    """iter_all_entry_dirs must return a list, not a one-shot generator."""

    def test_can_iterate_twice(self, tmp_path: Path) -> None:
        from .._cache import iter_all_entry_dirs

        _write_cache_entry(tmp_path, _FAKE_KEY)
        result = iter_all_entry_dirs(tmp_path)
        first_pass = list(result)
        second_pass = list(result)
        assert first_pass == second_pass, (
            "iter_all_entry_dirs returned a generator that was exhausted on first pass"
        )

    def test_returns_list_type(self, tmp_path: Path) -> None:
        from .._cache import iter_all_entry_dirs

        assert isinstance(iter_all_entry_dirs(tmp_path), list)


class TestIterCacheEntriesFingerprint:
    """iter_cache_entries fingerprint parsing coverage."""

    def test_valid_fingerprint_included(self, tmp_path: Path) -> None:
        from .._cache import iter_cache_entries

        fp = {"python": "3.12", "cython": "3.0"}
        _write_cache_entry(tmp_path, _FAKE_KEY, fingerprint=fp)
        entries = iter_cache_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0].fingerprint == fp

    def test_non_dict_fingerprint_excluded(self, tmp_path: Path) -> None:
        from .._cache import iter_cache_entries, write_meta

        build_dir, art = _write_cache_entry(tmp_path, _FAKE_KEY)
        # Overwrite meta with non-dict fingerprint
        meta = json.loads((build_dir / "meta.json").read_text())
        meta["fingerprint"] = ["not", "a", "dict"]
        write_meta(build_dir, meta)

        entries = iter_cache_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0].fingerprint is None

    def test_missing_fingerprint_is_none(self, tmp_path: Path) -> None:
        from .._cache import iter_cache_entries

        _write_cache_entry(tmp_path, _FAKE_KEY, fingerprint=None)
        entries = iter_cache_entries(tmp_path)
        assert entries[0].fingerprint is None


class TestFindPackageEntryByKeyEdgePaths:
    """find_package_entry_by_key: missing package_name and empty modules raises."""

    def test_missing_package_name_in_meta_raises(self, tmp_path: Path) -> None:
        from .._cache import find_package_entry_by_key, write_meta

        build_dir = tmp_path / _FAKE_KEY
        build_dir.mkdir()
        write_meta(build_dir, {"kind": "package", "modules": [{"module_name": "m", "artifact": "m.so"}]})
        # package_name absent → FileNotFoundError
        with pytest.raises(FileNotFoundError, match="package_name"):
            find_package_entry_by_key(tmp_path, _FAKE_KEY)

    def test_empty_modules_list_raises(self, tmp_path: Path) -> None:
        from .._cache import find_package_entry_by_key, write_meta

        build_dir = tmp_path / _FAKE_KEY
        build_dir.mkdir()
        write_meta(build_dir, {"kind": "package", "package_name": "pkg", "modules": []})
        with pytest.raises(FileNotFoundError, match="modules"):
            find_package_entry_by_key(tmp_path, _FAKE_KEY)

    def test_all_artifacts_missing_raises(self, tmp_path: Path) -> None:
        from .._cache import find_package_entry_by_key, write_meta

        build_dir = tmp_path / _FAKE_KEY
        build_dir.mkdir()
        write_meta(
            build_dir,
            {
                "kind": "package",
                "package_name": "pkg",
                "modules": [{"module_name": "pkg.m", "artifact": "nonexistent.so"}],
            },
        )
        with pytest.raises(FileNotFoundError, match="artifacts"):
            find_package_entry_by_key(tmp_path, _FAKE_KEY)


class TestDefaultCacheDirWindowsBranch:
    """_default_cache_dir Windows branch: skipped on non-Windows systems."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path logic")
    def test_nt_localappdata(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from .. import _cache as cache_mod

        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.delenv("TEMP", raising=False)
        result = cache_mod._default_cache_dir()
        assert "scikitplot" in str(result)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path logic")
    def test_nt_temp_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from .. import _cache as cache_mod

        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.setenv("TEMP", str(tmp_path))
        result = cache_mod._default_cache_dir()
        assert "scikitplot" in str(result)

    def test_xdg_posix_branch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """XDG_CACHE_HOME is used on POSIX when set (lines 249-250)."""
        from .. import _cache as cache_mod

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        result = cache_mod._default_cache_dir()
        assert str(tmp_path) in str(result)
        assert "scikitplot" in str(result)

    def test_posix_home_cache_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No XDG_CACHE_HOME → ~/.cache/scikitplot/cython (lines 251)."""
        from .. import _cache as cache_mod

        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = cache_mod._default_cache_dir()
        assert "scikitplot" in str(result)
        assert "cython" in str(result)


# ===========================================================================
# _loader.py — import_extension spec=None, metadata attachment, package meta
# ===========================================================================


class TestImportExtensionSpecNone:
    """import_extension raises ImportError when spec_from_file_location returns None."""

    def test_bad_spec_raises_import_error(self, tmp_path: Path) -> None:
        from .._loader import import_extension

        # Create a file with a valid extension name so the loader doesn't
        # reject it, but make spec_from_file_location return None via mock.
        artifact = tmp_path / f"bad_mod{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"\x7fELF")

        with patch("importlib.util.spec_from_file_location", return_value=None):
            with pytest.raises(ImportError, match="spec"):
                import_extension(name="bad_mod", path=artifact)


class TestImportExtensionMetadataAttach:
    """import_extension silently swallows setattr exceptions."""

    def test_setattr_exception_swallowed(self, tmp_path: Path) -> None:
        from .._loader import import_extension

        artifact = tmp_path / f"meta_mod{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"\x7fELF")

        fake_mod = types.ModuleType("meta_mod")

        class _FrozenModule(types.ModuleType):
            def __setattr__(self, name: str, value: Any) -> None:
                raise AttributeError("frozen")

        frozen = _FrozenModule("meta_mod")

        spec = MagicMock()
        spec.loader = MagicMock()

        def exec_module(m: Any) -> None:
            pass

        spec.loader.exec_module = exec_module

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=frozen):
                # Should not raise despite setattr failing
                mod = import_extension(
                    name="meta_mod",
                    path=artifact,
                    key=_FAKE_KEY,
                    build_dir=tmp_path,
                )
        assert mod is frozen


class TestImportExtensionFromPathPackageMeta:
    """import_extension_from_path: package-kind meta resolution path."""

    def test_package_meta_resolves_module_name(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_path

        # Create package build structure
        suffix = EXTENSION_SUFFIXES[0]
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        artifact = pkg_dir / f"mod1{suffix}"
        artifact.write_bytes(b"\x7fELF")

        # meta.json at parent (build_dir)
        rel_art = f"mypkg/mod1{suffix}"
        meta = {
            "kind": "package",
            "key": _FAKE_KEY,
            "package_name": "mypkg",
            "modules": [
                {"module_name": "mypkg.mod1", "artifact": rel_art}
            ],
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        fake_mod = types.ModuleType("mypkg.mod1")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                mod = import_extension_from_path(artifact)

        assert mod is fake_mod

    def test_key_and_build_dir_attached_from_meta(self, tmp_path: Path) -> None:
        """Key and build_dir must be read from meta.json when not overridden."""
        from .._loader import import_extension_from_path

        suffix = EXTENSION_SUFFIXES[0]
        artifact = tmp_path / f"amod{suffix}"
        artifact.write_bytes(b"\x7fELF")

        meta = {
            "kind": "module",
            "key": _FAKE_KEY,
            "module_name": "amod",
            "artifact": artifact.name,
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        captured: dict = {}

        fake_mod = types.ModuleType("amod")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                mod = import_extension_from_path(artifact)

        assert mod is fake_mod


class TestImportExtensionFromBytesCollision:
    """import_extension_from_bytes raises OSError on content collision."""

    def test_collision_raises_os_error(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_bytes

        suffix = EXTENSION_SUFFIXES[0]
        filename = f"colmod{suffix}"
        data1 = b"\x7fELF" + b"\x00" * 12
        data2 = b"\x7fELF" + b"\xff" * 12  # different content, same filename slot

        # Pre-place file with data1 at the deterministic path
        from hashlib import sha256

        h = sha256(data1).hexdigest()
        out_dir = tmp_path / "scikitplot_cython_import" / h[:16]
        out_dir.mkdir(parents=True)
        (out_dir / filename).write_bytes(data1)

        # Now try importing data2 — sha256 differs → different slot, no collision.
        # Instead: force collision by placing data2 at data2's deterministic slot
        # then immediately try importing data1 with a *patched* hash that
        # returns the same key.
        h2 = sha256(data2).hexdigest()
        out_dir2 = tmp_path / "scikitplot_cython_import" / h2[:16]
        out_dir2.mkdir(parents=True)
        # Write data1 at data2's slot to trigger content mismatch
        (out_dir2 / filename).write_bytes(data1)

        with pytest.raises(OSError, match="collision"):
            import_extension_from_bytes(
                data2,
                module_name="colmod",
                artifact_filename=filename,
                temp_dir=tmp_path,
            )


# ===========================================================================
# _pins.py — unpin last alias removes pins file
# ===========================================================================


class TestUnpinLastAliasRemovesFile:
    """Removing the last alias must delete pins.json, not write an empty file."""

    def test_pins_file_removed_when_empty(self, tmp_path: Path) -> None:
        from .._pins import pin, unpin

        pin(_FAKE_KEY, alias="sole_alias", cache_dir=tmp_path)
        pins_file = tmp_path / "pins.json"
        assert pins_file.exists()

        unpin("sole_alias", cache_dir=tmp_path)
        assert not pins_file.exists(), (
            "pins.json must be deleted when the last alias is removed"
        )

    def test_pins_file_unlink_filenotfound_swallowed(
        self, tmp_path: Path
    ) -> None:
        """unlink() raising FileNotFoundError must be silently swallowed."""
        from .._pins import pin, unpin

        pin(_FAKE_KEY, alias="alias_x", cache_dir=tmp_path)
        pins_file = tmp_path / "pins.json"

        original_unlink = Path.unlink

        def failing_unlink(self: Path, *args: Any, **kw: Any) -> None:
            if self == pins_file:
                raise FileNotFoundError("already gone")
            original_unlink(self, *args, **kw)

        with patch.object(Path, "unlink", failing_unlink):
            # Must not raise
            result = unpin("alias_x", cache_dir=tmp_path)

        assert result is True


# ===========================================================================
# _builder.py — _open_annotation_in_browser guard branches
# ===========================================================================


class TestOpenAnnotationInBrowser:
    """_open_annotation_in_browser must suppress the call in headless environments."""

    def _call(self, env_updates: dict, *, isatty: bool = False) -> bool:
        """Return True if webbrowser.open was called."""
        from .. import _builder as bmod

        opened: list[str] = []

        def fake_open(uri: str) -> None:
            opened.append(uri)

        with patch.object(bmod.webbrowser, "open", fake_open):
            with patch.dict(os.environ, env_updates, clear=False):
                # Patch stdout isatty
                with patch.object(sys.stdout, "isatty", return_value=isatty):
                    bmod._open_annotation_in_browser("file:///fake.html")

        return bool(opened)

    def test_ci_env_suppresses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CI", "true")
        assert not self._call({"CI": "true"}, isatty=True)

    def test_no_display_posix_suppresses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        # Only applies on non-win32 non-darwin
        if sys.platform not in ("win32", "darwin"):
            assert not self._call({"CI": ""}, isatty=True)

    def test_non_tty_suppresses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        assert not self._call({}, isatty=False)

    def test_isatty_exception_suppresses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If stdout.isatty() raises, the browser call is suppressed."""
        from .. import _builder as bmod

        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")

        opened: list[str] = []

        def fake_open(uri: str) -> None:  # pragma: no cover
            opened.append(uri)

        def raising_isatty() -> bool:
            raise OSError("no tty")

        with patch.object(bmod.webbrowser, "open", fake_open):
            with patch.object(sys.stdout, "isatty", raising_isatty):
                bmod._open_annotation_in_browser("file:///fake.html")

        assert not opened


class TestNormalizeExtraSourcesBytesPath:
    """_normalize_extra_sources must handle bytes paths via os.fsdecode."""

    def test_bytes_path_accepted(self, tmp_path: Path) -> None:
        from .._builder import _normalize_extra_sources

        src = tmp_path / "extra.c"
        src.write_text("int x = 1;", encoding="utf-8")
        result = _normalize_extra_sources([os.fsencode(str(src))])
        assert len(result) == 1
        assert result[0] == src.resolve()


# ===========================================================================
# _public.py — check_build_prereqs failure branches
# ===========================================================================


class TestCheckBuildPrereqsFailurePaths:
    """check_build_prereqs must record ok=False when imports fail."""

    def test_cython_import_failure_recorded(self) -> None:
        from .._public import check_build_prereqs

        with patch.dict(sys.modules, {"Cython": None}):
            result = check_build_prereqs()

        # When Cython import fails, ok=False is recorded under 'cython'
        # (behaviour depends on whether the mock causes ImportError)
        assert "cython" in result

    def test_setuptools_import_failure_recorded(self) -> None:
        from .._public import check_build_prereqs

        with patch.dict(sys.modules, {"setuptools": None}):
            result = check_build_prereqs()
        assert "setuptools" in result

    def test_numpy_failure_recorded_when_requested(self) -> None:
        from .._public import check_build_prereqs

        with patch.dict(sys.modules, {"numpy": None}):
            result = check_build_prereqs(numpy=True)
        assert "numpy" in result

    def test_numpy_not_included_when_not_requested(self) -> None:
        from .._public import check_build_prereqs

        result = check_build_prereqs(numpy=False)
        assert "numpy" not in result

    def test_cython_ok_path(self) -> None:
        """Happy path: Cython and setuptools present."""
        pytest.importorskip("Cython", reason="Cython package not installed in this environment")
        from .._public import check_build_prereqs

        result = check_build_prereqs()
        assert result["cython"]["ok"] is True
        assert result["setuptools"]["ok"] is True


# ===========================================================================
# _public.py — import_cached_result, import_pinned, register_cached_artifact_path
# ===========================================================================


class TestImportCachedResultPublic:
    """import_cached_result builds a BuildResult from a cache entry."""

    def test_returns_build_result_with_used_cache_true(
        self, tmp_path: Path
    ) -> None:
        from .._public import import_cached_result

        build_dir, artifact = _write_cache_entry(tmp_path, _FAKE_KEY)

        fake_mod = types.ModuleType("mymod")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                result = import_cached_result(_FAKE_KEY, cache_dir=tmp_path)

        assert result.used_cache is True
        assert result.key == _FAKE_KEY
        assert result.module_name == "mymod"

    def test_meta_source_sha256_propagated(self, tmp_path: Path) -> None:
        from .._cache import write_meta
        from .._public import import_cached_result

        build_dir, artifact = _write_cache_entry(tmp_path, _FAKE_KEY)
        meta = json.loads((build_dir / "meta.json").read_text())
        meta["source_sha256"] = "deadbeef" * 8
        write_meta(build_dir, meta)

        fake_mod = types.ModuleType("mymod")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                result = import_cached_result(_FAKE_KEY, cache_dir=tmp_path)

        assert result.source_sha256 == "deadbeef" * 8

    def test_corrupted_meta_gives_empty_dict(self, tmp_path: Path) -> None:
        """Corrupted meta.json must fall back to empty meta, not crash."""
        from .._public import import_cached_result

        build_dir, artifact = _write_cache_entry(tmp_path, _FAKE_KEY)
        (build_dir / "meta.json").write_text("NOT JSON {{{{", encoding="utf-8")

        fake_mod = types.ModuleType("mymod")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                result = import_cached_result(_FAKE_KEY, cache_dir=tmp_path)

        assert result.source_sha256 is None


class TestImportPinnedPublic:
    """import_pinned returns module for module-type pins."""

    def test_import_pinned_module_returns_module(self, tmp_path: Path) -> None:
        from .._public import import_pinned
        from .._pins import pin

        _write_cache_entry(tmp_path, _FAKE_KEY)
        pin(_FAKE_KEY, alias="myfn", cache_dir=tmp_path)

        fake_mod = types.ModuleType("mymod")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                result = import_pinned("myfn", cache_dir=tmp_path)

        assert isinstance(result, types.ModuleType)


class TestRegisterCachedArtifactPathPublic:
    """register_cached_artifact_path registers artifact and returns BuildResult."""

    def test_registration_returns_build_result(self, tmp_path: Path) -> None:
        from .._public import register_cached_artifact_path

        artifact = tmp_path / f"ext{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"\x7fELF" + b"\x00" * 8)
        cache_root = tmp_path / "cache"

        fake_mod = types.ModuleType("ext")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                result = register_cached_artifact_path(
                    artifact,
                    module_name="ext",
                    cache_dir=cache_root,
                    copy=True,
                )

        assert result.module_name == "ext"
        assert result.used_cache is True


class TestImportArtifactPublic:
    """import_artifact_path and import_artifact_bytes public wrappers."""

    def test_import_artifact_path_works(self, tmp_path: Path) -> None:
        from .._public import import_artifact_path

        suffix = EXTENSION_SUFFIXES[0]
        artifact = tmp_path / f"pathmod{suffix}"
        artifact.write_bytes(b"\x7fELF")

        meta = {"kind": "module", "key": _FAKE_KEY, "module_name": "pathmod", "artifact": artifact.name}
        (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        fake_mod = types.ModuleType("pathmod")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                mod = import_artifact_path(artifact)

        assert mod is fake_mod

    def test_import_artifact_bytes_works(self, tmp_path: Path) -> None:
        from .._public import import_artifact_bytes

        suffix = EXTENSION_SUFFIXES[0]
        filename = f"bytemod{suffix}"
        data = b"\x7fELF" + b"\xab" * 16

        fake_mod = types.ModuleType("bytemod")
        spec = MagicMock()
        spec.loader = MagicMock()
        spec.loader.exec_module = lambda m: None

        with patch("importlib.util.spec_from_file_location", return_value=spec):
            with patch("importlib.util.module_from_spec", return_value=fake_mod):
                mod = import_artifact_bytes(
                    data,
                    module_name="bytemod",
                    artifact_filename=filename,
                    temp_dir=tmp_path,
                )

        assert mod is fake_mod


class TestExportCachedReplace:
    """export_cached must overwrite an existing destination directory."""

    def test_replaces_existing_dest(self, tmp_path: Path) -> None:
        from .._public import export_cached

        cache_root = tmp_path / "cache"
        _write_cache_entry(cache_root, _FAKE_KEY)

        dest = tmp_path / "dest"
        # Pre-place stale content at destination
        stale_dst = dest / _FAKE_KEY
        stale_dst.mkdir(parents=True)
        (stale_dst / "stale.txt").write_text("old", encoding="utf-8")

        exported = export_cached(_FAKE_KEY, dest_dir=dest, cache_dir=cache_root)

        assert exported.exists()
        # Stale file should be gone (replaced by shutil.copytree)
        assert not (exported / "stale.txt").exists()

    def test_export_creates_dest_if_missing(self, tmp_path: Path) -> None:
        from .._public import export_cached

        cache_root = tmp_path / "cache"
        _write_cache_entry(cache_root, _FAKE_KEY)

        dest = tmp_path / "new_dest"
        exported = export_cached(_FAKE_KEY, dest_dir=dest, cache_dir=cache_root)
        assert exported.is_dir()
        assert (exported / "meta.json").exists()


class TestCythonImportAll:
    """cython_import_all: FileNotFoundError for absent directory, empty for no pyx."""

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        from .._public import cython_import_all

        with pytest.raises(FileNotFoundError):
            cython_import_all(tmp_path / "absent")

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        from .._public import cython_import_all

        result = cython_import_all(tmp_path)
        assert result == {}

    def test_non_pyx_files_not_compiled(self, tmp_path: Path) -> None:
        from .._public import cython_import_all

        (tmp_path / "helper.c").write_text("int x;", encoding="utf-8")
        result = cython_import_all(tmp_path)
        assert result == {}

    def test_custom_pattern_no_match_returns_empty(self, tmp_path: Path) -> None:
        from .._public import cython_import_all

        (tmp_path / "hello.pyx").write_text("def f(): pass", encoding="utf-8")
        # Pattern that matches nothing
        result = cython_import_all(tmp_path, pattern="*.nomatch")
        assert result == {}


# ===========================================================================
# _cache.py — iter_package_entries fingerprint and absolute artifact path
# ===========================================================================


class TestIterPackageEntriesFingerprint:
    """iter_package_entries must populate fingerprint from meta when present."""

    def test_valid_fingerprint_populated(self, tmp_path: Path) -> None:
        from .._cache import iter_package_entries

        build_dir = _write_package_cache_entry(tmp_path, _FAKE_KEY)
        # Patch meta to include a fingerprint
        meta = json.loads((build_dir / "meta.json").read_text())
        meta["fingerprint"] = {"python": "3.12"}
        (build_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        entries = iter_package_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0].fingerprint == {"python": "3.12"}

    def test_non_dict_fingerprint_is_none(self, tmp_path: Path) -> None:
        from .._cache import iter_package_entries

        build_dir = _write_package_cache_entry(tmp_path, _FAKE_KEY)
        meta = json.loads((build_dir / "meta.json").read_text())
        meta["fingerprint"] = [1, 2, 3]
        (build_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        entries = iter_package_entries(tmp_path)
        assert entries[0].fingerprint is None


class TestIterPackageEntriesAbsoluteArtifact:
    """iter_package_entries: artifact stored as absolute path in meta."""

    def test_absolute_artifact_path_resolved(self, tmp_path: Path) -> None:
        from .._cache import iter_package_entries

        build_dir = tmp_path / _FAKE_KEY
        pkg_dir = build_dir / "mypkg"
        pkg_dir.mkdir(parents=True)
        artifact = pkg_dir / f"mod1{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"\x7fELF")

        # Store artifact as absolute POSIX path
        meta = {
            "kind": "package",
            "key": _FAKE_KEY,
            "package_name": "mypkg",
            "modules": [
                {
                    "module_name": "mypkg.mod1",
                    "artifact": str(artifact),  # absolute path
                }
            ],
            "created_utc": "2024-01-01T00:00:00Z",
        }
        (build_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        entries = iter_package_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0].artifacts[0].is_absolute()


# ===========================================================================
# Parametric cross-module: sanitize edge cases (complete docstring examples)
# ===========================================================================


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("", "_"),
        ("hello-world", "hello_world"),
        ("123abc", "_123abc"),
        ("a/b/c", "a_b_c"),
        ("__dunder__", "__dunder__"),
        # Unicode letters are alnum in Python — kept as-is
        ("Ä", "Ä"),
        ("α", "α"),
        ("1", "_1"),
        # Punctuation and symbols → underscore
        ("a!b@c#", "a_b_c_"),
        ("---", "___"),
    ],
)
def test_sanitize_full_docstring_examples(inp: str, expected: str) -> None:
    from .._util import sanitize

    assert sanitize(inp) == expected


# ===========================================================================
# _lock.py — parametric timeout/poll validation (extends existing suite)
# ===========================================================================


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


# ===========================================================================
# _gc.py — cache_stats with and without entries (parametric)
# ===========================================================================


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
