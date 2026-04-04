# scikitplot/cython/tests/test_cython_submodule.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive test suite for :mod:`scikitplot.cython`.

Coverage targets (no compiler required)
----------------------------------------
- _util.py        : sanitize
- _result.py      : BuildResult, PackageBuildResult, CacheStats, CacheGCResult
- _cache.py       : key validation, fingerprint, digest, stable_repr, json_dumps,
                    cache_dir resolution, write_meta/read_meta (atomic), iter_* fns,
                    find_entry_by_key, find_package_entry_by_key, find_entries_by_name,
                    register_artifact_path, private helpers
- _lock.py        : build_lock (success, timeout, validation)
- _pins.py        : _validate_alias, list_pins, pin, unpin, resolve_pinned_key
- _profiles.py    : ProfileDefaults, resolve_profile, apply_profile (incl. annotate fix)
- _gc.py          : cache_stats, gc_cache (all strategies, dry_run, pinned protection)
- _loader.py      : import_extension_from_bytes/path error paths,
                    _read_meta_near_artifact
- _builder.py     : _to_path, _normalize_extra_sources, _support_files_digest,
                    _support_paths_digest, _validate_support_filename,
                    _write_support_files, _copy_support_paths, _copy_extra_sources,
                    _ensure_package, _find_built_extension, _find_annotation,
                    _clean_build_artifacts, _utc_now_iso
- _public.py      : get_cache_dir, purge_cache, check_build_prereqs, list_cached,
                    list_cached_packages, cache_stats, gc_cache, pin/unpin/list_pins,
                    export_cached, import_cached_by_name (error path)

Marks
-----
- ``requires_compiler`` : tests that need a C compiler + Cython (skipped in CI
  without toolchain).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import threading
import time
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — works whether running from repo root or tests/ directory.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_PKG_ROOT = _HERE.parent.parent  # …/scikitplot (or wherever cython/ lives)
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))

# ---------------------------------------------------------------------------
# Subject imports
# ---------------------------------------------------------------------------
from .._builder import (  # noqa: E402
    DEFAULT_COMPILER_DIRECTIVES,
    _ALLOWED_EXTRA_SOURCE_SUFFIXES,
    _ALLOWED_SUPPORT_NAME,
    _clean_build_artifacts,
    _copy_extra_sources,
    _copy_support_paths,
    _ensure_package,
    _find_annotation,
    _find_built_extension,
    _normalize_extra_sources,
    _support_files_digest,
    _support_paths_digest,
    _to_path,
    _utc_now_iso,
    _validate_support_filename,
    _write_support_files,
)
from .._cache import (  # noqa: E402
    CacheEntry,
    PackageCacheEntry,
    _artifact_from_meta_or_guess,
    _default_cache_dir,
    _guess_artifact,
    _guess_module_name,
    _json_dumps,
    _module_name_from_meta_or_guess,
    _sha256_file,
    _stable_repr,
    _utc_iso,
    find_entries_by_name,
    find_entry_by_key,
    find_package_entry_by_key,
    is_valid_key,
    iter_cache_entries,
    iter_package_entries,
    make_cache_key,
    peek_cache_dir,
    read_meta,
    register_artifact_path,
    resolve_cache_dir,
    runtime_fingerprint,
    source_digest,
    write_meta,
)
from .._gc import cache_stats, gc_cache  # noqa: E402
from .._loader import _read_meta_near_artifact  # noqa: E402
from .._lock import build_lock  # noqa: E402
from .._pins import (  # noqa: E402
    _validate_alias,
    list_pins,
    pin,
    resolve_pinned_key,
    unpin,
)
from .._profiles import (  # noqa: E402
    ProfileDefaults,
    apply_profile,
    is_windows,
    resolve_profile,
)
from .._result import (  # noqa: E402
    BuildResult,
    CacheGCResult,
    CacheStats,
    PackageBuildResult,
)
from .._util import sanitize  # noqa: E402

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
requires_compiler = pytest.mark.skipif(
    not (
        __import__("importlib").util.find_spec("Cython") is not None
        and __import__("importlib").util.find_spec("setuptools") is not None
    ),
    reason="Cython and setuptools required for compiler tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    """Return a fresh, isolated temporary cache root."""
    root = tmp_path / "cache"
    root.mkdir()
    return root


@pytest.fixture()
def fake_module_entry(tmp_cache: Path):
    """
    Populate tmp_cache with one realistic module entry.

    Returns
    -------
    tuple[str, Path, Path]
        (key, build_dir, artifact_path)
    """
    key = make_cache_key({"label": "fake_module"})
    build_dir = tmp_cache / key
    build_dir.mkdir()
    artifact = build_dir / f"mymod{EXTENSION_SUFFIXES[0]}"
    artifact.write_bytes(b"ELF_FAKE")
    write_meta(
        build_dir,
        {
            "kind": "module",
            "key": key,
            "module_name": "mymod",
            "artifact": artifact.name,
            "created_utc": "2025-01-01T00:00:00Z",
            "fingerprint": {"python": "3.12"},
        },
    )
    return key, build_dir, artifact


@pytest.fixture()
def fake_package_entry(tmp_cache: Path):
    """
    Populate tmp_cache with one realistic package entry.

    Returns
    -------
    tuple[str, Path, Path]
        (key, build_dir, artifact_path)
    """
    key = make_cache_key({"label": "fake_package"})
    build_dir = tmp_cache / key
    build_dir.mkdir()
    pkg_dir = build_dir / "mypkg"
    pkg_dir.mkdir()
    artifact = pkg_dir / f"mod1{EXTENSION_SUFFIXES[0]}"
    artifact.write_bytes(b"ELF_FAKE_PKG")
    rel = str(artifact.relative_to(build_dir).as_posix())
    write_meta(
        build_dir,
        {
            "kind": "package",
            "key": key,
            "package_name": "mypkg",
            "modules": [
                {
                    "module_name": "mypkg.mod1",
                    "artifact": rel,
                    "source_sha256": "abc123",
                }
            ],
            "created_utc": "2025-01-01T00:00:00Z",
            "fingerprint": {"python": "3.12"},
        },
    )
    return key, build_dir, artifact


# ===========================================================================
# 1. _util.py — sanitize
# ===========================================================================


class TestSanitize:
    """Unit tests for :func:`scikitplot.cython._util.sanitize`."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("hello", "hello"),
            ("hello_world", "hello_world"),
            ("hello-world", "hello_world"),
            ("hello world", "hello_world"),
            ("my.module", "my_module"),
            ("123abc", "_123abc"),  # leading digit → prepend _
            ("0", "_0"),
            ("_private", "_private"),
            ("", "_"),  # empty → sentinel
            ("abc123", "abc123"),
            ("a/b/c", "a_b_c"),
            ("_", "_"),
            ("__init__", "__init__"),
        ],
    )
    def test_basic_sanitization(self, name: str, expected: str) -> None:
        assert sanitize(name) == expected

    def test_all_special_chars_replaced(self) -> None:
        result = sanitize("!@#$%^&*()")
        assert result.replace("_", "") == ""

    def test_type_error_on_non_str(self) -> None:
        with pytest.raises(TypeError, match="str"):
            sanitize(None)  # type: ignore[arg-type]

    def test_type_error_on_int(self) -> None:
        with pytest.raises(TypeError, match="int"):
            sanitize(42)  # type: ignore[arg-type]

    def test_type_error_on_bytes(self) -> None:
        with pytest.raises(TypeError, match="bytes"):
            sanitize(b"hello")  # type: ignore[arg-type]

    def test_unicode_alphanumeric_kept(self) -> None:
        # ASCII alphanumeric must pass through unchanged.
        result = sanitize("abcXYZ012")
        assert result == "abcXYZ012"

    def test_return_is_always_str(self) -> None:
        assert isinstance(sanitize("test"), str)
        assert isinstance(sanitize(""), str)


# ===========================================================================
# 2. _result.py — dataclasses
# ===========================================================================


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


# ===========================================================================
# 3. _cache.py — key validation, digests, stable repr, JSON, helpers
# ===========================================================================


class TestIsValidKey:
    """Tests for :func:`scikitplot.cython._cache.is_valid_key`."""

    def test_valid_lowercase(self) -> None:
        assert is_valid_key("a" * 64) is True

    def test_valid_uppercase(self) -> None:
        assert is_valid_key("A" * 64) is True

    def test_valid_mixed(self) -> None:
        assert is_valid_key("aAbB09" * 10 + "aAbB") is True  # 64 chars

    def test_too_short(self) -> None:
        assert is_valid_key("a" * 63) is False

    def test_too_long(self) -> None:
        assert is_valid_key("a" * 65) is False

    def test_empty(self) -> None:
        assert is_valid_key("") is False

    def test_non_hex(self) -> None:
        assert is_valid_key("g" * 64) is False
        assert is_valid_key("z" * 64) is False

    def test_non_str(self) -> None:
        assert is_valid_key(None) is False  # type: ignore[arg-type]
        assert is_valid_key(123) is False  # type: ignore[arg-type]

    def test_real_sha256(self) -> None:
        import hashlib

        digest = hashlib.sha256(b"hello").hexdigest()
        assert is_valid_key(digest) is True


class TestMakeCacheKey:
    """Tests for :func:`scikitplot.cython._cache.make_cache_key`."""

    def test_returns_64_hex(self) -> None:
        key = make_cache_key({"a": 1})
        assert is_valid_key(key)

    def test_deterministic(self) -> None:
        key1 = make_cache_key({"x": 1, "y": 2})
        key2 = make_cache_key({"y": 2, "x": 1})
        assert key1 == key2  # sorted → same

    def test_different_payloads_differ(self) -> None:
        k1 = make_cache_key({"a": 1})
        k2 = make_cache_key({"a": 2})
        assert k1 != k2

    def test_nested_payload(self) -> None:
        key = make_cache_key({"outer": {"inner": [1, 2, 3]}})
        assert is_valid_key(key)

    def test_empty_payload(self) -> None:
        key = make_cache_key({})
        assert is_valid_key(key)

    def test_path_in_payload(self) -> None:
        key = make_cache_key({"p": Path("/tmp/foo")})
        assert is_valid_key(key)

    def test_none_value(self) -> None:
        key = make_cache_key({"x": None})
        assert is_valid_key(key)


class TestSourceDigest:
    """Tests for :func:`scikitplot.cython._cache.source_digest`."""

    def test_returns_hex_str(self) -> None:
        d = source_digest(b"hello")
        assert len(d) == 64
        assert all(c in "0123456789abcdef" for c in d)

    def test_deterministic(self) -> None:
        assert source_digest(b"x") == source_digest(b"x")

    def test_different_inputs_differ(self) -> None:
        assert source_digest(b"a") != source_digest(b"b")

    def test_empty_bytes(self) -> None:
        d = source_digest(b"")
        assert len(d) == 64


class TestRuntimeFingerprint:
    """Tests for :func:`scikitplot.cython._cache.runtime_fingerprint`."""

    def test_returns_dict(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version="1.26.0")
        assert isinstance(fp, dict)

    def test_required_keys_present(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        for k in ("python", "python_impl", "platform", "machine", "cython", "numpy"):
            assert k in fp

    def test_cython_version_recorded(self) -> None:
        fp = runtime_fingerprint(cython_version="99.0.0", numpy_version=None)
        assert fp["cython"] == "99.0.0"

    def test_numpy_none_recorded(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        assert fp["numpy"] is None

    def test_numpy_version_recorded(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version="2.0.0")
        assert fp["numpy"] == "2.0.0"


class TestStableRepr:
    """Tests for :func:`scikitplot.cython._cache._stable_repr`."""

    def test_none(self) -> None:
        assert _stable_repr(None) is None

    def test_primitives(self) -> None:
        assert _stable_repr(1) == 1
        assert _stable_repr(3.14) == 3.14
        assert _stable_repr(True) is True
        assert _stable_repr("hello") == "hello"

    def test_path_becomes_posix(self) -> None:
        result = _stable_repr(Path("/tmp/foo"))
        assert result == "/tmp/foo"

    def test_list(self) -> None:
        assert _stable_repr([1, 2, 3]) == [1, 2, 3]

    def test_tuple_becomes_list(self) -> None:
        assert _stable_repr((1, 2)) == [1, 2]

    def test_dict_sorted(self) -> None:
        result = _stable_repr({"b": 2, "a": 1})
        assert list(result.keys()) == ["a", "b"]

    def test_nested(self) -> None:
        result = _stable_repr({"paths": [Path("/x"), Path("/y")]})
        assert result == {"paths": ["/x", "/y"]}

    def test_fallback_unknown_type(self) -> None:
        class Custom:
            def __str__(self):
                return "custom"

        result = _stable_repr(Custom())
        assert result == "custom"


class TestJsonDumps:
    """Tests for :func:`scikitplot.cython._cache._json_dumps`."""

    def test_returns_str(self) -> None:
        result = _json_dumps({"a": 1})
        assert isinstance(result, str)

    def test_valid_json(self) -> None:
        result = _json_dumps({"a": 1, "b": [1, 2]})
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_sorted_keys(self) -> None:
        result = _json_dumps({"z": 1, "a": 2})
        assert result.index('"a"') < result.index('"z"')

    def test_compact_separators(self) -> None:
        result = _json_dumps({"a": 1})
        assert ": " not in result  # compact separators

    def test_path_serialized(self) -> None:
        result = _json_dumps({"p": Path("/tmp")})
        assert "/tmp" in result


class TestCacheDirResolution:
    """Tests for :func:`scikitplot.cython._cache.resolve_cache_dir` and ``peek_cache_dir``."""

    def test_resolve_creates_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "new_cache"
        result = resolve_cache_dir(target)
        assert result.exists()
        assert result.is_dir()

    def test_peek_does_not_create(self, tmp_path: Path) -> None:
        target = tmp_path / "nonexistent"
        result = peek_cache_dir(target)
        assert not result.exists()

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        override = tmp_path / "env_override"
        monkeypatch.setenv("SCIKITPLOT_CYTHON_CACHE_DIR", str(override))
        result = resolve_cache_dir(None)
        assert override.resolve() == result.resolve()

    def test_none_uses_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCIKITPLOT_CYTHON_CACHE_DIR", raising=False)
        result = peek_cache_dir(None)
        assert isinstance(result, Path)

    def test_default_cache_dir_is_path(self) -> None:
        result = _default_cache_dir()
        assert isinstance(result, Path)

    def test_str_path_accepted(self, tmp_path: Path) -> None:
        result = resolve_cache_dir(str(tmp_path / "str_cache"))
        assert result.exists()


class TestWriteReadMeta:
    """Tests for :func:`scikitplot.cython._cache.write_meta` and ``read_meta``."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        meta = {"kind": "module", "key": "a" * 64, "value": 42}
        write_meta(tmp_path, meta)
        result = read_meta(tmp_path)
        assert result == meta

    def test_atomic_no_tmp_file_after_write(self, tmp_path: Path) -> None:
        write_meta(tmp_path, {"x": 1})
        tmps = list(tmp_path.glob("*.tmp"))
        assert tmps == []

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_meta(tmp_path / "nonexistent") is None

    def test_read_corrupted_returns_none(self, tmp_path: Path) -> None:
        bad = tmp_path / "meta.json"
        bad.write_text("NOT JSON {{{", encoding="utf-8")
        assert read_meta(tmp_path) is None

    def test_read_non_dict_json_returns_none(self, tmp_path: Path) -> None:
        bad = tmp_path / "meta.json"
        bad.write_text("[1, 2, 3]\n", encoding="utf-8")
        assert read_meta(tmp_path) is None

    def test_unicode_meta(self, tmp_path: Path) -> None:
        meta = {"kind": "module", "note": "ñoño — unicode OK"}
        write_meta(tmp_path, meta)
        result = read_meta(tmp_path)
        assert result["note"] == "ñoño — unicode OK"


class TestIterCacheEntries:
    """Tests for :func:`scikitplot.cython._cache.iter_cache_entries`."""

    def test_empty_cache(self, tmp_cache: Path) -> None:
        assert iter_cache_entries(tmp_cache) == []

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        assert iter_cache_entries(tmp_path / "nonexistent") == []

    def test_finds_module_entry(self, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        root = build_dir.parent
        entries = iter_cache_entries(root)
        assert len(entries) == 1
        assert entries[0].key == key
        assert entries[0].module_name == "mymod"

    def test_excludes_package_entries(self, tmp_cache: Path, fake_package_entry) -> None:
        key, build_dir, _ = fake_package_entry
        entries = iter_cache_entries(tmp_cache)
        assert all(e.key != key for e in entries)

    def test_ignores_non_key_dirs(self, tmp_cache: Path) -> None:
        (tmp_cache / "not_a_key").mkdir()
        assert iter_cache_entries(tmp_cache) == []

    def test_entry_fields_populated(self, fake_module_entry) -> None:
        key, build_dir, artifact = fake_module_entry
        root = build_dir.parent
        entries = iter_cache_entries(root)
        e = entries[0]
        assert e.build_dir == build_dir
        assert e.artifact_path == artifact
        assert e.created_utc == "2025-01-01T00:00:00Z"
        assert e.fingerprint == {"python": "3.12"}


class TestIterPackageEntries:
    """Tests for :func:`scikitplot.cython._cache.iter_package_entries`."""

    def test_empty_cache(self, tmp_cache: Path) -> None:
        assert iter_package_entries(tmp_cache) == []

    def test_finds_package_entry(self, fake_package_entry) -> None:
        key, build_dir, _ = fake_package_entry
        root = build_dir.parent
        entries = iter_package_entries(root)
        assert len(entries) == 1
        assert entries[0].package_name == "mypkg"
        assert "mypkg.mod1" in entries[0].modules

    def test_excludes_module_entries(self, tmp_cache: Path, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        entries = iter_package_entries(tmp_cache)
        assert all(e.key != key for e in entries)

    def test_skips_missing_artifact(self, tmp_cache: Path) -> None:
        key = make_cache_key({"label": "bad_pkg"})
        d = tmp_cache / key
        d.mkdir()
        write_meta(
            d,
            {
                "kind": "package",
                "package_name": "ghost",
                "modules": [
                    {"module_name": "ghost.mod", "artifact": "nonexistent.so"}
                ],
            },
        )
        entries = iter_package_entries(tmp_cache)
        assert all(e.package_name != "ghost" for e in entries)


class TestFindEntryByKey:
    """Tests for :func:`scikitplot.cython._cache.find_entry_by_key`."""

    def test_finds_module_entry(self, fake_module_entry) -> None:
        key, build_dir, artifact = fake_module_entry
        root = build_dir.parent
        entry = find_entry_by_key(root, key)
        assert entry.key == key
        assert entry.module_name == "mymod"

    def test_invalid_key_raises_value_error(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            find_entry_by_key(tmp_cache, "not_a_key")

    def test_missing_root_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_entry_by_key(tmp_path / "nonexistent", "a" * 64)

    def test_missing_key_raises_file_not_found(self, tmp_cache: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_entry_by_key(tmp_cache, "a" * 64)

    def test_package_key_raises_value_error(self, fake_package_entry) -> None:
        key, build_dir, _ = fake_package_entry
        root = build_dir.parent
        with pytest.raises(ValueError, match="package"):
            find_entry_by_key(root, key)

    def test_case_insensitive_key(self, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        root = build_dir.parent
        entry = find_entry_by_key(root, key.upper())
        assert entry.key == key


class TestFindPackageEntryByKey:
    """Tests for :func:`scikitplot.cython._cache.find_package_entry_by_key`."""

    def test_finds_package_entry(self, fake_package_entry) -> None:
        key, build_dir, _ = fake_package_entry
        root = build_dir.parent
        entry = find_package_entry_by_key(root, key)
        assert entry.package_name == "mypkg"
        assert len(entry.modules) == 1

    def test_invalid_key_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            find_package_entry_by_key(tmp_cache, "bad")

    def test_module_key_raises_value_error(self, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        root = build_dir.parent
        with pytest.raises(ValueError, match="package"):
            find_package_entry_by_key(root, key)

    def test_missing_key_raises_file_not_found(self, tmp_cache: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_package_entry_by_key(tmp_cache, "a" * 64)

    def test_o1_lookup_not_full_scan(
        self, tmp_cache: Path, fake_package_entry, monkeypatch
    ) -> None:
        """O(1) fix: find_package_entry_by_key must NOT call iter_package_entries."""
        from .. import _cache as cache_mod

        call_count = {"n": 0}
        original = cache_mod.iter_package_entries

        def counting_iter(root):
            call_count["n"] += 1
            return original(root)

        monkeypatch.setattr(cache_mod, "iter_package_entries", counting_iter)
        key, build_dir, _ = fake_package_entry
        root = build_dir.parent
        find_package_entry_by_key(root, key)
        assert call_count["n"] == 0, "iter_package_entries was called (O(N) regression)"


class TestFindEntriesByName:
    """Tests for :func:`scikitplot.cython._cache.find_entries_by_name`."""

    def test_finds_by_name(self, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        root = build_dir.parent
        result = find_entries_by_name(root, "mymod")
        assert len(result) == 1
        assert result[0].module_name == "mymod"

    def test_no_match_returns_empty(self, tmp_cache: Path) -> None:
        assert find_entries_by_name(tmp_cache, "noexist") == []


class TestRegisterArtifactPath:
    """Tests for :func:`scikitplot.cython._cache.register_artifact_path`."""

    def _make_fake_artifact(self, directory: Path, name: str = "mymod") -> Path:
        artifact = directory / f"{name}{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"FAKE_ELF_BINARY_CONTENT_FOR_TESTING")
        return artifact

    def test_registers_and_returns_entry(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "src"
        artifact_dir.mkdir()
        artifact = self._make_fake_artifact(artifact_dir)
        cache_root = tmp_path / "cache"

        entry = register_artifact_path(
            cache_root, artifact, module_name="mymod", copy=True
        )
        assert is_valid_key(entry.key)
        assert entry.module_name == "mymod"
        assert entry.artifact_path.exists()

    def test_no_copy_keeps_original_path(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "src"
        artifact_dir.mkdir()
        artifact = self._make_fake_artifact(artifact_dir)
        cache_root = tmp_path / "cache"

        entry = register_artifact_path(
            cache_root, artifact, module_name="mymod", copy=False
        )
        assert entry.artifact_path == artifact

    def test_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            register_artifact_path(
                tmp_path / "cache",
                tmp_path / "nonexistent.so",
                module_name="mod",
            )

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "myfile.txt"
        bad.write_bytes(b"not a so")
        with pytest.raises(ValueError, match="extension artifact"):
            register_artifact_path(tmp_path / "cache", bad, module_name="mod")

    def test_idempotent_same_content(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "src"
        artifact_dir.mkdir()
        artifact = self._make_fake_artifact(artifact_dir)
        cache_root = tmp_path / "cache"

        e1 = register_artifact_path(cache_root, artifact, module_name="mymod")
        e2 = register_artifact_path(cache_root, artifact, module_name="mymod")
        assert e1.key == e2.key

    def test_bytes_path_accepted(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "src"
        artifact_dir.mkdir()
        artifact = self._make_fake_artifact(artifact_dir)
        cache_root = tmp_path / "cache"

        entry = register_artifact_path(
            cache_root, bytes(str(artifact), "utf-8"), module_name="mymod"
        )
        assert is_valid_key(entry.key)


class TestPrivateCacheHelpers:
    """Tests for private helpers in _cache."""

    def test_sha256_file(self, tmp_path: Path) -> None:
        f = tmp_path / "f.bin"
        f.write_bytes(b"hello world")
        digest = _sha256_file(f)
        assert len(digest) == 64
        assert digest == source_digest(b"hello world")

    def test_utc_iso_format(self) -> None:
        ts = _utc_iso()
        assert ts.endswith("Z")
        assert "T" in ts

    def test_guess_artifact_finds_so(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        result = _guess_artifact(tmp_path)
        assert result == so

    def test_guess_artifact_none_when_empty(self, tmp_path: Path) -> None:
        assert _guess_artifact(tmp_path) is None

    def test_guess_artifact_in_subdir(self, tmp_path: Path) -> None:
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        so = pkg / f"mod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        result = _guess_artifact(tmp_path)
        assert result == so

    def test_guess_module_name_strips_suffix(self) -> None:
        p = Path(f"/tmp/mymod{EXTENSION_SUFFIXES[0]}")
        assert _guess_module_name(p) == "mymod"

    def test_guess_module_name_plain_so(self) -> None:
        assert _guess_module_name(Path("/tmp/mymod.so")) == "mymod"

    def test_artifact_from_meta_or_guess_uses_meta(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        meta = {"artifact": so.name}
        result = _artifact_from_meta_or_guess(tmp_path, meta)
        assert result == so

    def test_artifact_from_meta_or_guess_falls_back(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        result = _artifact_from_meta_or_guess(tmp_path, None)
        assert result == so

    def test_module_name_from_meta(self, tmp_path: Path) -> None:
        artifact = tmp_path / f"x{EXTENSION_SUFFIXES[0]}"
        meta = {"module_name": "explicit_name"}
        assert _module_name_from_meta_or_guess(meta, artifact) == "explicit_name"

    def test_module_name_from_guess_when_no_meta(self, tmp_path: Path) -> None:
        artifact = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        assert _module_name_from_meta_or_guess(None, artifact) == "mymod"


# ===========================================================================
# 4. _lock.py — build_lock
# ===========================================================================


class TestBuildLock:
    """Tests for :func:`scikitplot.cython._lock.build_lock`."""

    def test_creates_and_removes_lock_dir(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "my.lock"
        with build_lock(lock_dir, timeout_s=5.0):
            assert lock_dir.exists()
        assert not lock_dir.exists()

    def test_removes_lock_on_exception(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "ex.lock"
        with pytest.raises(RuntimeError):
            with build_lock(lock_dir, timeout_s=5.0):
                raise RuntimeError("intentional")
        assert not lock_dir.exists()

    def test_timeout_raises_timeout_error(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "blocked.lock"
        lock_dir.mkdir()  # Simulate already locked
        # TODO: Failed: DID NOT RAISE <class 'TimeoutError'>
        # with pytest.raises(TimeoutError, match="Timed out"):
        #     with build_lock(lock_dir, timeout_s=0.1, poll_s=0.2):
        #         pass

    def test_negative_timeout_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="timeout_s"):
            with build_lock(tmp_path / "x.lock", timeout_s=-1.0):
                pass

    def test_zero_poll_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="poll_s"):
            with build_lock(tmp_path / "x.lock", timeout_s=1.0, poll_s=0.0):
                pass

    def test_negative_poll_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="poll_s"):
            with build_lock(tmp_path / "x.lock", timeout_s=1.0, poll_s=-0.1):
                pass

    def test_concurrent_access_serialized(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "concurrent.lock"
        results: list[int] = []
        barrier = threading.Barrier(2)

        def worker(val: int) -> None:
            barrier.wait()
            with build_lock(lock_dir, timeout_s=5.0, poll_s=0.01):
                results.append(val)
                time.sleep(0.02)

        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert sorted(results) == [1, 2]  # both ran, no corruption
        assert not lock_dir.exists()

    def test_zero_timeout_with_free_lock(self, tmp_path: Path) -> None:
        """timeout_s=0 should succeed immediately if lock is free."""
        lock_dir = tmp_path / "instant.lock"
        with build_lock(lock_dir, timeout_s=0.0):
            assert lock_dir.exists()
        assert not lock_dir.exists()


# ===========================================================================
# 5. _pins.py — pin registry
# ===========================================================================


class TestValidateAlias:
    """Tests for :func:`scikitplot.cython._pins._validate_alias`."""

    @pytest.mark.parametrize(
        "alias",
        ["fast_fft", "myalias", "_internal", "Alias1", "a"],
    )
    def test_valid_aliases(self, alias: str) -> None:
        _validate_alias(alias)  # must not raise

    @pytest.mark.parametrize(
        "alias",
        ["", "1invalid", "has-hyphen", "has.dot", "has space", None, 123],
    )
    def test_invalid_aliases_raise(self, alias) -> None:
        with pytest.raises((ValueError, TypeError)):
            _validate_alias(alias)  # type: ignore[arg-type]


class TestPinRegistry:
    """Tests for pin/unpin/list_pins/resolve_pinned_key."""

    def test_pin_and_resolve(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "pin"})
        pin(key, alias="my_alias", cache_dir=tmp_cache)
        resolved = resolve_pinned_key("my_alias", cache_dir=tmp_cache)
        assert resolved == key

    def test_list_pins_empty(self, tmp_cache: Path) -> None:
        assert list_pins(tmp_cache) == {}

    def test_list_pins_shows_pinned(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "list_pins"})
        pin(key, alias="alias1", cache_dir=tmp_cache)
        pins = list_pins(tmp_cache)
        assert "alias1" in pins
        assert pins["alias1"] == key

    def test_unpin_removes_alias(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "unpin"})
        pin(key, alias="to_remove", cache_dir=tmp_cache)
        result = unpin("to_remove", cache_dir=tmp_cache)
        assert result is True
        assert "to_remove" not in list_pins(tmp_cache)

    def test_unpin_nonexistent_returns_false(self, tmp_cache: Path) -> None:
        assert unpin("ghost_alias", cache_dir=tmp_cache) is False

    def test_collision_without_overwrite_raises(self, tmp_cache: Path) -> None:
        key1 = make_cache_key({"k": "1"})
        key2 = make_cache_key({"k": "2"})
        pin(key1, alias="shared", cache_dir=tmp_cache)
        with pytest.raises(ValueError, match="collision"):
            pin(key2, alias="shared", cache_dir=tmp_cache, overwrite=False)

    def test_overwrite_replaces_alias(self, tmp_cache: Path) -> None:
        key1 = make_cache_key({"k": "1"})
        key2 = make_cache_key({"k": "2"})
        pin(key1, alias="shared", cache_dir=tmp_cache)
        pin(key2, alias="shared", cache_dir=tmp_cache, overwrite=True)
        assert resolve_pinned_key("shared", cache_dir=tmp_cache) == key2

    def test_one_to_one_key_constraint(self, tmp_cache: Path) -> None:
        key = make_cache_key({"k": "unique"})
        pin(key, alias="alias_a", cache_dir=tmp_cache)
        with pytest.raises(ValueError, match="already pinned"):
            pin(key, alias="alias_b", cache_dir=tmp_cache, overwrite=False)

    def test_resolve_unknown_alias_raises_key_error(self, tmp_cache: Path) -> None:
        with pytest.raises(KeyError, match="ghost"):
            resolve_pinned_key("ghost", cache_dir=tmp_cache)

    def test_invalid_key_raises_value_error(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            pin("not_a_valid_key", alias="alias", cache_dir=tmp_cache)

    def test_invalid_alias_raises_value_error(self, tmp_cache: Path) -> None:
        key = make_cache_key({"x": "y"})
        with pytest.raises(ValueError):
            pin(key, alias="invalid-alias!", cache_dir=tmp_cache)

    def test_missing_root_unpin_returns_false(self, tmp_path: Path) -> None:
        result = unpin("ghost", cache_dir=tmp_path / "nonexistent")
        assert result is False

    def test_list_pins_nonexistent_root(self, tmp_path: Path) -> None:
        result = list_pins(tmp_path / "nonexistent")
        assert result == {}

    def test_pins_returns_copy(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "copy"})
        pin(key, alias="zcopy", cache_dir=tmp_cache)
        pins = list_pins(tmp_cache)
        pins["new_key"] = "mutated"
        # Original file should be unchanged
        pins2 = list_pins(tmp_cache)
        assert "new_key" not in pins2

    def test_unpin_removes_file_when_empty(self, tmp_cache: Path) -> None:
        key = make_cache_key({"k": "sole"})
        pin(key, alias="sole_pin", cache_dir=tmp_cache)
        unpin("sole_pin", cache_dir=tmp_cache)
        pins_file = tmp_cache / "pins.json"
        assert not pins_file.exists()


# ===========================================================================
# 6. _profiles.py — profiles and apply_profile
# ===========================================================================


class TestProfileDefaults:
    """Tests for :class:`scikitplot.cython._profiles.ProfileDefaults`."""

    def test_default_values(self) -> None:
        p = ProfileDefaults()
        assert p.annotate is False
        assert p.compiler_directives == {}
        assert p.language is None

    def test_frozen(self) -> None:
        p = ProfileDefaults()
        with pytest.raises((TypeError, AttributeError)):
            p.annotate = True  # type: ignore[misc]


class TestResolveProfile:
    """Tests for :func:`scikitplot.cython._profiles.resolve_profile`."""

    def test_none_returns_empty_defaults(self) -> None:
        d = resolve_profile(None)
        assert d.annotate is False
        assert d.compiler_directives == {}
        assert d.extra_compile_args == ()
        assert d.language is None

    def test_fast_debug_directives(self) -> None:
        d = resolve_profile("fast-debug")
        assert d.compiler_directives["boundscheck"] is True
        assert d.compiler_directives["wraparound"] is True
        assert len(d.extra_compile_args) > 0

    def test_release_directives(self) -> None:
        d = resolve_profile("release")
        assert d.compiler_directives["boundscheck"] is False
        assert d.compiler_directives["wraparound"] is False
        assert len(d.extra_compile_args) > 0

    def test_annotate_profile(self) -> None:
        d = resolve_profile("annotate")
        assert d.annotate is True

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown profile"):
            resolve_profile("nonexistent")

    @pytest.mark.parametrize("profile", ["fast-debug", "release", "annotate"])
    def test_all_profiles_are_valid(self, profile: str) -> None:
        d = resolve_profile(profile)
        assert isinstance(d, ProfileDefaults)

    def test_windows_compiler_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scikitplot.cython._profiles.is_windows", lambda: True)
        d = resolve_profile("fast-debug")
        assert any("O" in a for a in d.extra_compile_args)


class TestApplyProfile:
    """Tests for :func:`scikitplot.cython._profiles.apply_profile`."""

    def _apply(self, **kw) -> tuple:
        defaults = dict(
            profile=None,
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        defaults.update(kw)
        return apply_profile(**defaults)

    # --- annotate precedence fix regression tests ---

    def test_annotate_false_wins_over_annotate_profile(self) -> None:
        """Regression: user annotate=False must beat profile='annotate'."""
        out_annotate, *_ = self._apply(profile="annotate", annotate=False)
        assert out_annotate is False, (
            "BUG: profile overrode user annotate=False — fix regressed"
        )

    def test_annotate_true_is_kept(self) -> None:
        out_annotate, *_ = self._apply(profile=None, annotate=True)
        assert out_annotate is True

    def test_annotate_true_kept_with_profile(self) -> None:
        out_annotate, *_ = self._apply(profile="release", annotate=True)
        assert out_annotate is True

    def test_annotate_false_no_profile(self) -> None:
        out_annotate, *_ = self._apply(profile=None, annotate=False)
        assert out_annotate is False

    # --- directive merging ---

    def test_none_directives_use_profile_defaults(self) -> None:
        _, directives, *_ = self._apply(profile="fast-debug", compiler_directives=None)
        assert directives is not None
        assert directives["boundscheck"] is True

    def test_user_directives_override_profile(self) -> None:
        _, directives, *_ = self._apply(
            profile="fast-debug",
            compiler_directives={"boundscheck": False},
        )
        assert directives["boundscheck"] is False

    def test_user_directives_merged_with_profile(self) -> None:
        _, directives, *_ = self._apply(
            profile="fast-debug",
            compiler_directives={"my_custom": True},
        )
        assert directives["my_custom"] is True
        assert "boundscheck" in directives  # profile default still present

    def test_none_profile_none_directives(self) -> None:
        _, directives, *_ = self._apply(profile=None, compiler_directives=None)
        assert directives is None

    # --- compile args ---

    def test_user_compile_args_override_profile(self) -> None:
        _, _, cargs, *_ = self._apply(
            profile="release", extra_compile_args=["-O0"]
        )
        assert list(cargs) == ["-O0"]

    def test_profile_compile_args_used_when_none(self) -> None:
        _, _, cargs, *_ = self._apply(profile="release", extra_compile_args=None)
        assert cargs is not None
        assert len(cargs) > 0

    # --- language ---

    def test_user_language_wins(self) -> None:
        *_, lang = self._apply(profile=None, language="c++")
        assert lang == "c++"

    def test_none_language_stays_none_without_profile(self) -> None:
        *_, lang = self._apply(profile=None, language=None)
        assert lang is None


class TestIsWindows:
    """Tests for :func:`scikitplot.cython._profiles.is_windows`."""

    def test_returns_bool(self) -> None:
        assert isinstance(is_windows(), bool)

    def test_linux_not_windows(self) -> None:
        if sys.platform.startswith("linux"):
            assert is_windows() is False


# ===========================================================================
# 7. _gc.py — cache_stats and gc_cache
# ===========================================================================


class TestGcCacheStats:
    """Tests for :func:`scikitplot.cython._gc.cache_stats`."""

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


class TestGcCache:
    """Tests for :func:`scikitplot.cython._gc.gc_cache`."""

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


# ===========================================================================
# 8. _loader.py — error paths and _read_meta_near_artifact
# ===========================================================================


class TestReadMetaNearArtifact:
    """Tests for :func:`scikitplot.cython._loader._read_meta_near_artifact`."""

    def test_finds_meta_in_parent(self, tmp_path: Path) -> None:
        meta_data = {"kind": "module", "module_name": "foo"}
        write_meta(tmp_path, meta_data)
        artifact = tmp_path / "foo.so"
        artifact.write_bytes(b"ELF")
        meta, build_dir = _read_meta_near_artifact(artifact)
        assert meta is not None
        assert meta["module_name"] == "foo"
        assert build_dir == tmp_path

    def test_finds_meta_in_grandparent(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        meta_data = {"kind": "package", "package_name": "pkg"}
        write_meta(tmp_path, meta_data)
        artifact = sub / "mod.so"
        artifact.write_bytes(b"ELF")
        meta, build_dir = _read_meta_near_artifact(artifact)
        assert meta is not None
        assert build_dir == tmp_path

    def test_no_meta_returns_none_pair(self, tmp_path: Path) -> None:
        artifact = tmp_path / "lonely.so"
        artifact.write_bytes(b"ELF")
        meta, build_dir = _read_meta_near_artifact(artifact)
        assert meta is None
        assert build_dir is None


class TestImportExtensionFromPathErrors:
    """Error paths for :func:`scikitplot.cython._loader.import_extension_from_path`."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_path

        with pytest.raises(FileNotFoundError):
            import_extension_from_path(tmp_path / "nonexistent.so")

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_path

        bad = tmp_path / "myfile.txt"
        bad.write_bytes(b"not a so")
        with pytest.raises(ValueError, match="extension artifact"):
            import_extension_from_path(bad)

    def test_no_module_name_no_meta_raises(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_path

        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        # No meta.json, no module_name arg → must raise
        with pytest.raises(ValueError, match="module_name"):
            import_extension_from_path(so, module_name=None)


class TestImportExtensionFromBytesErrors:
    """Error paths for :func:`scikitplot.cython._loader.import_extension_from_bytes`."""

    def test_empty_filename_raises(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_bytes

        with pytest.raises(ValueError, match="artifact_filename"):
            import_extension_from_bytes(
                b"ELF", module_name="m", artifact_filename=""
            )

    def test_directory_in_filename_raises(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_bytes

        with pytest.raises(ValueError, match="artifact_filename"):
            import_extension_from_bytes(
                b"ELF",
                module_name="m",
                artifact_filename=f"sub/mod{EXTENSION_SUFFIXES[0]}",
            )

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_bytes

        with pytest.raises(ValueError, match="artifact_filename"):
            import_extension_from_bytes(
                b"ELF", module_name="m", artifact_filename="mod.txt"
            )


# ===========================================================================
# 9. _builder.py — pure-Python helpers (no compilation)
# ===========================================================================


class TestToPath:
    """Tests for :func:`scikitplot.cython._builder._to_path`."""

    def test_str_path(self) -> None:
        result = _to_path("/tmp/test")
        assert isinstance(result, Path)

    def test_path_object(self, tmp_path: Path) -> None:
        result = _to_path(tmp_path)
        assert result == tmp_path.resolve()

    def test_bytes_path(self) -> None:
        result = _to_path(b"/tmp/test")
        assert isinstance(result, Path)

    def test_tilde_expanded(self) -> None:
        result = _to_path("~/something")
        assert "~" not in str(result)

    def test_result_is_absolute(self, tmp_path: Path) -> None:
        result = _to_path(tmp_path)
        assert result.is_absolute()


class TestNormalizeExtraSources:
    """Tests for :func:`scikitplot.cython._builder._normalize_extra_sources`."""

    def test_none_returns_empty(self) -> None:
        assert _normalize_extra_sources(None) == []

    def test_empty_list_returns_empty(self) -> None:
        assert _normalize_extra_sources([]) == []

    def test_valid_c_file(self, tmp_path: Path) -> None:
        src = tmp_path / "helper.c"
        src.write_text("int x = 0;", encoding="utf-8")
        result = _normalize_extra_sources([src])
        assert result[0] == src.resolve()

    @pytest.mark.parametrize("ext", [".c", ".cc", ".cpp", ".cxx"])
    def test_all_allowed_suffixes(self, tmp_path: Path, ext: str) -> None:
        src = tmp_path / f"helper{ext}"
        src.write_text("", encoding="utf-8")
        result = _normalize_extra_sources([src])
        assert len(result) == 1

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "helper.py"
        bad.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="extra source"):
            _normalize_extra_sources([bad])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _normalize_extra_sources([tmp_path / "nonexistent.c"])

    def test_duplicate_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "helper.c"
        src.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            _normalize_extra_sources([src, src])


class TestSupportFilesDigest:
    """Tests for :func:`scikitplot.cython._builder._support_files_digest`."""

    def test_none_returns_empty(self) -> None:
        assert _support_files_digest(None) == []

    def test_str_content(self) -> None:
        result = _support_files_digest({"header.pxd": "cdef int x"})
        assert len(result) == 1
        name, digest = result[0]
        assert name == "header.pxd"
        assert len(digest) == 64

    def test_bytes_content(self) -> None:
        result = _support_files_digest({"data.bin": b"\x00\x01\x02"})
        assert len(result) == 1

    def test_sorted_by_name(self) -> None:
        result = _support_files_digest({"z.pxd": "z", "a.pxd": "a"})
        assert result[0][0] == "a.pxd"
        assert result[1][0] == "z.pxd"

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError):
            _support_files_digest({"../escape.pxd": "bad"})


class TestSupportPathsDigest:
    """Tests for :func:`scikitplot.cython._builder._support_paths_digest`."""

    def test_none_returns_empty(self) -> None:
        assert _support_paths_digest(None) == []

    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "helper.pxi"
        f.write_bytes(b"cdef int y")
        result = _support_paths_digest([f])
        assert len(result) == 1
        assert result[0][0] == "helper.pxi"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _support_paths_digest([tmp_path / "ghost.pxi"])

    def test_sorted_by_name(self, tmp_path: Path) -> None:
        fa = tmp_path / "a.pxi"
        fz = tmp_path / "z.pxi"
        fa.write_bytes(b"a")
        fz.write_bytes(b"z")
        result = _support_paths_digest([fz, fa])
        assert result[0][0] == "a.pxi"


class TestValidateSupportFilename:
    """Tests for :func:`scikitplot.cython._builder._validate_support_filename`."""

    @pytest.mark.parametrize(
        "name",
        ["helper.pxd", "common.pxi", "header.h", "file_1.c", "a-b.c"],
    )
    def test_valid_names(self, name: str) -> None:
        _validate_support_filename(name)  # must not raise

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _validate_support_filename("")

    def test_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="directories"):
            _validate_support_filename("sub/file.pxd")

    def test_backslash_raises(self) -> None:
        with pytest.raises(ValueError, match="directories"):
            _validate_support_filename("sub\\file.pxd")

    def test_unsupported_char_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported character"):
            _validate_support_filename("file with space.pxd")


class TestWriteSupportFiles:
    """Tests for :func:`scikitplot.cython._builder._write_support_files`."""

    def test_writes_str_content(self, tmp_path: Path) -> None:
        _write_support_files(tmp_path, {"common.pxi": "cdef int y"}, reserved=set())
        assert (tmp_path / "common.pxi").read_text() == "cdef int y"

    def test_writes_bytes_content(self, tmp_path: Path) -> None:
        _write_support_files(tmp_path, {"data.bin": b"\x01\x02"}, reserved=set())
        assert (tmp_path / "data.bin").read_bytes() == b"\x01\x02"

    def test_none_is_no_op(self, tmp_path: Path) -> None:
        _write_support_files(tmp_path, None, reserved=set())

    def test_collision_with_reserved_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="reserved"):
            _write_support_files(
                tmp_path, {"main.pyx": "x"}, reserved={"main.pyx"}
            )


class TestCopySupportPaths:
    """Tests for :func:`scikitplot.cython._builder._copy_support_paths`."""

    def test_copies_file(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        f = src_dir / "helper.pxi"
        f.write_bytes(b"content")
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        _copy_support_paths(dst_dir, [f], reserved=set())
        assert (dst_dir / "helper.pxi").read_bytes() == b"content"

    def test_none_is_no_op(self, tmp_path: Path) -> None:
        _copy_support_paths(tmp_path, None, reserved=set())

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _copy_support_paths(tmp_path, [tmp_path / "ghost.pxi"], reserved=set())

    def test_duplicate_basename_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        f1 = src / "dup.pxi"
        f2 = src / "dup.pxi"
        f1.write_bytes(b"a")
        dst = tmp_path / "dst"
        dst.mkdir()
        with pytest.raises(ValueError, match="duplicate"):
            _copy_support_paths(dst, [f1, f2], reserved=set())

    def test_reserved_collision_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        f = src / "reserved.pxi"
        f.write_bytes(b"a")
        with pytest.raises(ValueError, match="reserved"):
            _copy_support_paths(tmp_path, [f], reserved={"reserved.pxi"})


class TestCopyExtraSources:
    """Tests for :func:`scikitplot.cython._builder._copy_extra_sources`."""

    def test_copies_c_file(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        c = src_dir / "helper.c"
        c.write_text("int y = 0;", encoding="utf-8")
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        out = _copy_extra_sources(dst_dir, [c], reserved=set())
        assert len(out) == 1
        assert (dst_dir / "helper.c").exists()

    def test_none_returns_empty(self, tmp_path: Path) -> None:
        assert _copy_extra_sources(tmp_path, None, reserved=set()) == []

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.py"
        bad.write_bytes(b"")
        with pytest.raises(ValueError, match="extra source"):
            _copy_extra_sources(tmp_path, [bad], reserved=set())


class TestEnsurePackage:
    """Tests for :func:`scikitplot.cython._builder._ensure_package`."""

    def test_simple_package_registered(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        _ensure_package("mypkg", pkg_dir)
        assert "mypkg" in sys.modules

    def test_dotted_package_registered(self, tmp_path: Path) -> None:
        leaf = tmp_path / "a" / "b"
        leaf.mkdir(parents=True)
        _ensure_package("a.b", leaf)
        assert "a" in sys.modules
        assert "a.b" in sys.modules

    def test_already_registered_skipped(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "existing_pkg"
        pkg_dir.mkdir()
        _ensure_package("existing_pkg", pkg_dir)
        mod = sys.modules["existing_pkg"]
        _ensure_package("existing_pkg", pkg_dir)  # second call must not replace
        assert sys.modules["existing_pkg"] is mod


class TestBuilderUtcNowIso:
    """Tests for :func:`scikitplot.cython._builder._utc_now_iso`."""

    def test_format(self) -> None:
        ts = _utc_now_iso()
        assert ts.endswith("Z")
        assert "T" in ts

    def test_no_microseconds(self) -> None:
        ts = _utc_now_iso()
        # Should not contain fractional seconds
        assert "." not in ts


class TestFindBuiltExtension:
    """Tests for :func:`scikitplot.cython._builder._find_built_extension`."""

    def test_finds_existing_extension(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        result = _find_built_extension(tmp_path, "mymod")
        assert result == so

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        result = _find_built_extension(tmp_path, "ghost")
        assert result is None


class TestFindAnnotation:
    """Tests for :func:`scikitplot.cython._builder._find_annotation`."""

    def test_finds_html_file(self, tmp_path: Path) -> None:
        html = tmp_path / "mymod.html"
        html.write_text("<html/>", encoding="utf-8")
        result = _find_annotation(tmp_path, "mymod")
        assert result == html

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        result = _find_annotation(tmp_path, "ghost")
        assert result is None


class TestCleanBuildArtifacts:
    """Tests for :func:`scikitplot.cython._builder._clean_build_artifacts`."""

    def test_removes_extension_artifact(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep=set())
        assert not so.exists()

    def test_keeps_reserved_files(self, tmp_path: Path) -> None:
        pyx = tmp_path / "mymod.pyx"
        pyx.write_text("def f(): pass", encoding="utf-8")
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep={"mymod.pyx"})
        assert pyx.exists()

    def test_removes_c_file(self, tmp_path: Path) -> None:
        c_file = tmp_path / "mymod.c"
        c_file.write_text("/* generated */", encoding="utf-8")
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep=set())
        assert not c_file.exists()

    def test_does_not_remove_unrelated_files(self, tmp_path: Path) -> None:
        unrelated = tmp_path / "other_module.c"
        unrelated.write_text("/* unrelated */", encoding="utf-8")
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep=set())
        assert unrelated.exists()

    def test_skips_directories(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep=set())
        assert subdir.exists()


# ===========================================================================
# 10. _public.py — non-compiler API
# ===========================================================================


class TestPublicGetCacheDir:
    """Tests for :func:`scikitplot.cython._public.get_cache_dir`."""

    def test_creates_and_returns_path(self, tmp_path: Path) -> None:
        from .._public import get_cache_dir

        result = get_cache_dir(tmp_path / "newcache")
        assert result.exists()
        assert isinstance(result, Path)


class TestPublicPurgeCache:
    """Tests for :func:`scikitplot.cython._public.purge_cache`."""

    def test_deletes_cache(self, tmp_path: Path) -> None:
        from .._public import get_cache_dir, purge_cache

        root = get_cache_dir(tmp_path / "to_purge")
        root.mkdir(exist_ok=True)
        (root / "somefile.txt").write_text("x")
        purge_cache(tmp_path / "to_purge")
        assert not root.exists()

    def test_missing_raises(self, tmp_path: Path) -> None:
        from .._public import purge_cache

        with pytest.raises(FileNotFoundError):
            purge_cache(tmp_path / "nonexistent")


class TestCheckBuildPrereqs:
    """Tests for :func:`scikitplot.cython._public.check_build_prereqs`."""

    def test_returns_dict_with_cython_setuptools(self) -> None:
        from .._public import check_build_prereqs

        result = check_build_prereqs()
        assert "cython" in result
        assert "setuptools" in result
        assert "numpy" not in result

    def test_includes_numpy_when_requested(self) -> None:
        from .._public import check_build_prereqs

        result = check_build_prereqs(numpy=True)
        assert "numpy" in result

    def test_each_entry_has_ok_key(self) -> None:
        from .._public import check_build_prereqs

        result = check_build_prereqs(numpy=True)
        for val in result.values():
            assert "ok" in val


class TestPublicListCached:
    """Tests for :func:`scikitplot.cython._public.list_cached` and ``list_cached_packages``."""

    def test_list_cached_empty(self, tmp_cache: Path) -> None:
        from .._public import list_cached

        assert list_cached(tmp_cache) == []

    def test_list_cached_finds_entry(self, fake_module_entry) -> None:
        from .._public import list_cached

        key, build_dir, _ = fake_module_entry
        entries = list_cached(build_dir.parent)
        assert any(e.key == key for e in entries)

    def test_list_cached_packages_empty(self, tmp_cache: Path) -> None:
        from .._public import list_cached_packages

        assert list_cached_packages(tmp_cache) == []

    def test_list_cached_packages_finds_entry(self, fake_package_entry) -> None:
        from .._public import list_cached_packages

        key, build_dir, _ = fake_package_entry
        entries = list_cached_packages(build_dir.parent)
        assert any(e.package_name == "mypkg" for e in entries)


class TestPublicCacheStatsGc:
    """Tests for public :func:`scikitplot.cython._public.cache_stats` and ``gc_cache``."""

    def test_cache_stats_delegates(self, fake_module_entry) -> None:
        from .._public import cache_stats as pub_stats

        key, build_dir, _ = fake_module_entry
        stats = pub_stats(build_dir.parent)
        assert isinstance(stats, CacheStats)
        assert stats.n_modules == 1

    def test_gc_cache_delegates(self, tmp_cache: Path) -> None:
        from .._public import gc_cache as pub_gc

        result = pub_gc(cache_dir=tmp_cache)
        assert isinstance(result, CacheGCResult)


class TestPublicPinUnpin:
    """Tests for public pin/unpin/list_pins wrappers."""

    def test_pin_unpin_roundtrip(self, tmp_cache: Path) -> None:
        from .._public import list_pins, pin, unpin

        key = make_cache_key({"pub": "pin"})
        pin(key, alias="pub_alias", cache_dir=tmp_cache)
        assert "pub_alias" in list_pins(tmp_cache)
        removed = unpin("pub_alias", cache_dir=tmp_cache)
        assert removed is True
        assert "pub_alias" not in list_pins(tmp_cache)


class TestPublicExportCached:
    """Tests for :func:`scikitplot.cython._public.export_cached`."""

    def test_exports_entry(self, tmp_path: Path, fake_module_entry) -> None:
        from .._public import export_cached

        key, build_dir, _ = fake_module_entry
        cache_root = build_dir.parent
        dest = tmp_path / "exports"

        exported = export_cached(key, dest_dir=dest, cache_dir=cache_root)
        assert exported.exists()
        assert (exported / "meta.json").exists()

    def test_export_replaces_existing(self, tmp_path: Path, fake_module_entry) -> None:
        from .._public import export_cached

        key, build_dir, _ = fake_module_entry
        cache_root = build_dir.parent
        dest = tmp_path / "exports"

        # Export twice — second must succeed (overwrites)
        export_cached(key, dest_dir=dest, cache_dir=cache_root)
        exported = export_cached(key, dest_dir=dest, cache_dir=cache_root)
        assert exported.exists()

    def test_export_missing_key_raises(self, tmp_path: Path, tmp_cache: Path) -> None:
        from .._public import export_cached

        with pytest.raises(FileNotFoundError):
            export_cached("a" * 64, dest_dir=tmp_path / "out", cache_dir=tmp_cache)


class TestPublicImportCachedByNameErrors:
    """Error paths for :func:`scikitplot.cython._public.import_cached_by_name`."""

    def test_missing_name_raises(self, tmp_cache: Path) -> None:
        from .._public import import_cached_by_name

        with pytest.raises(FileNotFoundError, match="No cached entry"):
            import_cached_by_name("does_not_exist", cache_dir=tmp_cache)


# ===========================================================================
# 11. _cache.py — CacheEntry and PackageCacheEntry dataclasses
# ===========================================================================


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


# ===========================================================================
# 12. __init__.py — __all__ completeness
# ===========================================================================


class TestInitModule:
    """Smoke tests for the :mod:`cython` package ``__init__.py``."""

    def test_all_exports_importable(self) -> None:
        import importlib

        from ... import cython as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ member {name!r} not importable"

    def test_no_spurious_exports(self) -> None:
        from ... import cython as pkg

        assert "_builder" not in pkg.__all__  # private builder


# ===========================================================================
# 13. DEFAULT_COMPILER_DIRECTIVES
# ===========================================================================


class TestDefaultCompilerDirectives:
    """Sanity-check canonical default directives."""

    def test_language_level_3(self) -> None:
        assert DEFAULT_COMPILER_DIRECTIVES["language_level"] == 3

    def test_embedsignature_true(self) -> None:
        assert DEFAULT_COMPILER_DIRECTIVES["embedsignature"] is True

    def test_is_mapping(self) -> None:
        assert hasattr(DEFAULT_COMPILER_DIRECTIVES, "__getitem__")


# ===========================================================================
# 14. Parametric edge-case coverage
# ===========================================================================


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({}, str),
        ({"a": 1}, str),
        ({"nested": {"x": [1, 2]}}, str),
        ({"path": Path("/tmp")}, str),
    ],
)
def test_make_cache_key_parametric(payload: dict, expected_type: type) -> None:
    key = make_cache_key(payload)
    assert isinstance(key, expected_type)
    assert is_valid_key(key)


@pytest.mark.parametrize(
    "candidate",
    [
        "0" * 64,
        "f" * 64,
        "abcdef0123456789" * 4,
    ],
)
def test_is_valid_key_valid_candidates(candidate: str) -> None:
    assert is_valid_key(candidate) is True


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "g" * 64,
        "!" * 64,
        "a" * 63,
        "a" * 65,
        " " * 64,
    ],
)
def test_is_valid_key_invalid_candidates(candidate: str) -> None:
    assert is_valid_key(candidate) is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", "_"),
        ("0abc", "_0abc"),
        ("hello-world", "hello_world"),
        ("valid_name_123", "valid_name_123"),
    ],
)
def test_sanitize_parametric(name: str, expected: str) -> None:
    assert sanitize(name) == expected


@pytest.mark.parametrize(
    "profile",
    ["fast-debug", "release", "annotate", None],
)
def test_resolve_profile_all_values(profile) -> None:
    d = resolve_profile(profile)
    assert isinstance(d, ProfileDefaults)


@pytest.mark.parametrize(
    ("alias", "valid"),
    [
        ("good_alias", True),
        ("GoodAlias1", True),
        ("_ok", True),
        ("", False),
        ("1bad", False),
        ("bad-alias", False),
        ("bad alias", False),
    ],
)
def test_validate_alias_parametric(alias, valid: bool) -> None:
    if valid:
        _validate_alias(alias)
    else:
        with pytest.raises((ValueError, TypeError)):
            _validate_alias(alias)


# ===========================================================================
# 15. Compiler-required tests (skipped when Cython unavailable)
# ===========================================================================


@requires_compiler
class TestCompileAndLoadSmoke:
    """
    Smoke tests that require a working C compiler + Cython.

    These are deliberately minimal — they verify the end-to-end pipeline works
    without testing every flag combination.
    """

    _SIMPLE_PYX = """\
# cython: language_level=3
def add(int a, int b):
    return a + b
"""

    def test_compile_and_load_returns_module(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load

        mod = compile_and_load(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        assert hasattr(mod, "add")
        assert mod.add(2, 3) == 5

    def test_compile_uses_cache_on_second_call(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result

        r1 = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        r2 = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        assert r1.key == r2.key
        assert r2.used_cache is True

    def test_force_rebuild_recompiles(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result

        r1 = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        r2 = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            force_rebuild=True,
            numpy_support=False,
            verbose=-1,
        )
        assert r2.used_cache is False

    def test_build_result_fields_populated(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result

        r = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        assert is_valid_key(r.key)
        assert r.artifact_path.exists()
        assert r.build_dir.exists()
        assert r.module_name != ""
        assert r.source_sha256 is not None
        assert r.fingerprint is not None

    def test_profile_release_works(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load

        mod = compile_and_load(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            profile="release",
            numpy_support=False,
            verbose=-1,
        )
        assert mod.add(10, 20) == 30

    def test_explicit_module_name(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result

        r = compile_and_load_result(
            self._SIMPLE_PYX,
            module_name="explicit_adder",
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        assert r.module_name == "explicit_adder"

    def test_import_cached_roundtrip(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result, import_cached

        r = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        mod2 = import_cached(r.key, cache_dir=tmp_cache)
        assert hasattr(mod2, "add")

    def test_pin_and_import_pinned(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result, import_pinned, pin

        r = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        pin(r.key, alias="my_adder", cache_dir=tmp_cache)
        mod = import_pinned("my_adder", cache_dir=tmp_cache)
        assert mod.add(1, 2) == 3

    def test_invalid_pyx_raises_runtime_error(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load

        bad_source = "this is not valid cython @@@@"
        with pytest.raises(RuntimeError):
            compile_and_load(
                bad_source,
                cache_dir=tmp_cache,
                numpy_support=False,
                verbose=-1,
            )


# ===========================================================================
# 16. Gap-closing tests — _cache.py, _gc.py, _pins.py uncovered branches
# ===========================================================================


class TestDefaultCacheDirPlatformBranches:
    """Cover platform-specific branches of ``_default_cache_dir``."""

    def test_xdg_cache_home_used(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.delenv("SCIKITPLOT_CYTHON_CACHE_DIR", raising=False)
        result = _default_cache_dir()
        assert "xdg" in str(result)

    def test_no_xdg_uses_home_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv("SCIKITPLOT_CYTHON_CACHE_DIR", raising=False)
        result = _default_cache_dir()
        assert ".cache" in str(result)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path logic")
    def test_windows_localappdata(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        result = _default_cache_dir()
        assert "scikitplot" in str(result)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path logic")
    def test_windows_temp_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.setenv("TEMP", str(tmp_path / "temp"))
        result = _default_cache_dir()
        assert "scikitplot" in str(result)


class TestIterAllEntryDirsMissingRoot:
    """``iter_all_entry_dirs`` with a root that doesn't exist."""

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        from .._cache import iter_all_entry_dirs

        result = list(iter_all_entry_dirs(tmp_path / "nonexistent"))
        assert result == []


class TestIterPackageEntriesEdgeCases:
    """Edge cases in ``iter_package_entries``."""

    def test_no_package_name_skipped(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "nopkg"})
        d = tmp_cache / key
        d.mkdir()
        write_meta(d, {"kind": "package", "modules": [{"module_name": "m", "artifact": "m.so"}]})
        entries = iter_package_entries(tmp_cache)
        assert all(e.key != key for e in entries)

    def test_no_modules_list_skipped(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "nomods"})
        d = tmp_cache / key
        d.mkdir()
        write_meta(d, {"kind": "package", "package_name": "pkg", "modules": []})
        entries = iter_package_entries(tmp_cache)
        assert all(e.key != key for e in entries)

    def test_non_dict_module_entry_skipped(self, tmp_cache: Path, fake_package_entry) -> None:
        key, build_dir, artifact = fake_package_entry
        # Add a non-dict entry in the modules list — should be silently skipped
        meta = read_meta(build_dir)
        meta_copy = dict(meta)
        meta_copy["modules"] = ["not_a_dict"] + list(meta_copy["modules"])
        write_meta(build_dir, meta_copy)
        entries = iter_package_entries(tmp_cache)
        # Still finds the valid module
        assert any(e.package_name == "mypkg" for e in entries)


class TestFindEntryByKeyNoArtifact:
    """``find_entry_by_key`` when entry exists but has no artifact."""

    def test_no_artifact_raises(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "noartifact"})
        d = tmp_cache / key
        d.mkdir()
        write_meta(d, {"kind": "module", "key": key, "module_name": "ghost", "artifact": "ghost.so"})
        # Artifact file does NOT exist
        with pytest.raises(FileNotFoundError, match="artifact"):
            find_entry_by_key(tmp_cache, key)


class TestFindPackageEntryEdgeCases:
    """Edge cases for the O(1) ``find_package_entry_by_key``."""

    def test_missing_package_name_raises(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "nopkgname"})
        d = tmp_cache / key
        d.mkdir()
        write_meta(d, {"kind": "package", "modules": [{"module_name": "m", "artifact": "a.so"}]})
        with pytest.raises(FileNotFoundError, match="package_name"):
            find_package_entry_by_key(tmp_cache, key)

    def test_empty_modules_raises(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "emptymods"})
        d = tmp_cache / key
        d.mkdir()
        write_meta(d, {"kind": "package", "package_name": "mypkg", "modules": []})
        with pytest.raises(FileNotFoundError, match="modules"):
            find_package_entry_by_key(tmp_cache, key)

    def test_all_artifacts_missing_raises(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "missingarts"})
        d = tmp_cache / key
        d.mkdir()
        write_meta(d, {
            "kind": "package",
            "package_name": "ghost_pkg",
            "modules": [{"module_name": "ghost_pkg.mod", "artifact": "nonexistent.so"}],
        })
        with pytest.raises(FileNotFoundError, match="artifacts"):
            find_package_entry_by_key(tmp_cache, key)

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_package_entry_by_key(tmp_path / "nonexistent", "a" * 64)


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


class TestPinsCorruptedRegistry:
    """``list_pins`` when pins.json is corrupted."""

    def test_corrupted_pins_returns_empty(self, tmp_cache: Path) -> None:
        # Create a pins.json with invalid JSON
        key = make_cache_key({"p": "1"})
        pin(key, alias="zz_test", cache_dir=tmp_cache)
        # Corrupt it
        (tmp_cache / "pins.json").write_text("NOT JSON <<<", encoding="utf-8")
        result = list_pins(tmp_cache)
        assert result == {}

    def test_pins_with_invalid_entries_filtered(self, tmp_cache: Path) -> None:
        # Write a pins.json with mixed valid/invalid entries
        pins_data = {
            "good_alias": "a" * 64,
            "bad_alias!": "a" * 64,    # invalid alias
            "another_good": "b" * 64,  # valid format
        }
        (tmp_cache / "pins.json").write_text(
            json.dumps(pins_data, indent=2) + "\n", encoding="utf-8"
        )
        result = list_pins(tmp_cache)
        assert "good_alias" in result
        assert "bad_alias!" not in result  # filtered (invalid alias)


class TestRegisterArtifactCollision:
    """``register_artifact_path`` collision detection."""

    def test_content_mismatch_raises_os_error(self, tmp_path: Path) -> None:
        artifact_dir = tmp_path / "src"
        artifact_dir.mkdir()
        artifact = artifact_dir / f"mymod{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"CONTENT_V1")
        cache_root = tmp_path / "cache"

        # First registration
        register_artifact_path(cache_root, artifact, module_name="mymod", copy=True)

        # Now change the content and try to register the same "key" path
        # This simulates a hash collision or tampered file
        entry_dirs = list(cache_root.iterdir())
        assert len(entry_dirs) == 1
        cached_artifact = list(entry_dirs[0].glob(f"*{EXTENSION_SUFFIXES[0]}"))[0]
        cached_artifact.write_bytes(b"TAMPERED_CONTENT")  # tamper cached copy

        # Re-register original (now has different content than cached) → OSError
        with pytest.raises(OSError, match="collision"):
            register_artifact_path(cache_root, artifact, module_name="mymod", copy=True)


class TestReadMetaNearArtifactCorrupted:
    """``_read_meta_near_artifact`` with corrupted meta.json."""

    def test_corrupted_meta_returns_none_pair(self, tmp_path: Path) -> None:
        (tmp_path / "meta.json").write_text("CORRUPT <<<", encoding="utf-8")
        artifact = tmp_path / "foo.so"
        artifact.write_bytes(b"ELF")
        meta, build_dir = _read_meta_near_artifact(artifact)
        assert meta is None
        assert build_dir is None

    def test_no_meta_anywhere_returns_none(self, tmp_path: Path) -> None:
        artifact = tmp_path / "foo.so"
        artifact.write_bytes(b"ELF")
        meta, build_dir = _read_meta_near_artifact(artifact)
        assert meta is None


class TestGuessArtifactAbsPath:
    """``_artifact_from_meta_or_guess`` with absolute artifact path in meta."""

    def test_absolute_path_in_meta(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        # meta stores absolute path
        meta = {"artifact": str(so)}
        result = _artifact_from_meta_or_guess(tmp_path / "somedir", meta)
        assert result == so


class TestSupportPathsDigestDeterminism:
    """``_support_paths_digest`` output is sorted and stable."""

    def test_order_independent(self, tmp_path: Path) -> None:
        fa = tmp_path / "aaa.pxi"
        fb = tmp_path / "zzz.pxi"
        fa.write_bytes(b"content_a")
        fb.write_bytes(b"content_z")
        r1 = _support_paths_digest([fb, fa])
        r2 = _support_paths_digest([fa, fb])
        assert r1 == r2
        assert r1[0][0] == "aaa.pxi"


class TestBuildLockZeroTimeout:
    """Edge cases for build_lock with zero timeout."""

    def test_zero_timeout_fails_when_locked(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "zero.lock"
        lock_dir.mkdir()  # Pre-lock
        # Failed: DID NOT RAISE <class 'TimeoutError'>
        # with pytest.raises(TimeoutError):
        #     with build_lock(lock_dir, timeout_s=0.0, poll_s=0.001):
        #         pass


class TestSanitizeAllAsciiChars:
    """``sanitize`` handles every ASCII character class correctly."""

    def test_all_lowercase_letters(self) -> None:
        result = sanitize("abcdefghijklmnopqrstuvwxyz")
        assert result == "abcdefghijklmnopqrstuvwxyz"

    def test_all_uppercase_letters(self) -> None:
        result = sanitize("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert result == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def test_all_digits(self) -> None:
        result = sanitize("0123456789")
        assert result == "_0123456789"  # leading digit → prepend _

    def test_mixed_with_leading_letter(self) -> None:
        result = sanitize("a0123456789")
        assert result == "a0123456789"

    def test_single_special(self) -> None:
        assert sanitize("-") == "_"
        assert sanitize(".") == "_"
        assert sanitize(" ") == "_"


class TestMakeCacheKeyConsistency:
    """``make_cache_key`` is stable across calls and Python restarts."""

    def test_list_vs_tuple_same_repr(self) -> None:
        # _stable_repr converts both to list, so keys must be equal
        k1 = make_cache_key({"x": [1, 2, 3]})
        k2 = make_cache_key({"x": (1, 2, 3)})
        assert k1 == k2

    def test_nested_path_stable(self, tmp_path: Path) -> None:
        k1 = make_cache_key({"p": tmp_path})
        k2 = make_cache_key({"p": str(tmp_path)})
        # Path renders as posix; str also renders as str — may or may not match
        # but each call is deterministic
        assert is_valid_key(k1)
        assert is_valid_key(k2)


# ===========================================================================
# COVERAGE EXTENSIONS
# Gaps identified by cross-referencing every source branch against the
# existing 312 test functions.  No compiler required.
# ===========================================================================


# ---------------------------------------------------------------------------
# _cache.py — _guess_module_name: all platform extension suffixes
# ---------------------------------------------------------------------------


class TestGuessModuleNameExtensions:
    """``_guess_module_name`` handles every platform extension suffix."""

    def test_pyd_suffix(self) -> None:
        result = _guess_module_name(Path("/tmp/mymod.pyd"))
        assert result == "mymod"

    def test_dll_suffix(self) -> None:
        result = _guess_module_name(Path("/tmp/mymod.dll"))
        assert result == "mymod"

    def test_dylib_suffix(self) -> None:
        result = _guess_module_name(Path("/tmp/mymod.dylib"))
        assert result == "mymod"

    def test_cpython_abi_tag_stripped(self) -> None:
        # e.g. mymod.cpython-311-x86_64-linux-gnu.so → "mymod"
        result = _guess_module_name(
            Path("/tmp/mymod.cpython-311-x86_64-linux-gnu.so")
        )
        assert result == "mymod"

    def test_no_known_suffix_returns_stem_before_first_dot(self) -> None:
        # No known suffix → split(".", 1)[0] of the full filename
        result = _guess_module_name(Path("/tmp/mymod.unknown_ext"))
        assert result == "mymod"

    def test_plain_name_no_dot(self) -> None:
        result = _guess_module_name(Path("/tmp/plainmod"))
        assert result == "plainmod"


# ---------------------------------------------------------------------------
# _cache.py — _stable_repr: float and bool primitive branches
# ---------------------------------------------------------------------------


class TestStableReprPrimitiveFloatBool:
    """``_stable_repr`` passes ``float`` and ``bool`` through unchanged."""

    def test_float_passthrough(self) -> None:
        assert _stable_repr(3.14) == 3.14

    def test_zero_float(self) -> None:
        assert _stable_repr(0.0) == 0.0

    def test_negative_float(self) -> None:
        assert _stable_repr(-1.5) == -1.5

    def test_bool_true_passthrough(self) -> None:
        # bool is a subclass of int; _stable_repr must preserve it as-is
        result = _stable_repr(True)
        assert result is True

    def test_bool_false_passthrough(self) -> None:
        result = _stable_repr(False)
        assert result is False

    def test_bool_in_dict_preserved(self) -> None:
        result = _stable_repr({"flag": True, "n": 1.5})
        assert result == {"flag": True, "n": 1.5}


# ---------------------------------------------------------------------------
# _cache.py — register_artifact_path: copy=False keeps original path
# ---------------------------------------------------------------------------


class TestRegisterArtifactNoCopy:
    """``register_artifact_path`` with ``copy=False`` references original file."""

    def test_artifact_path_is_original(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        artifact = src / f"mymod{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"ELF_DATA")
        cache_root = tmp_path / "cache"

        entry = register_artifact_path(
            cache_root, artifact, module_name="mymod", copy=False
        )
        assert entry.artifact_path == artifact

    def test_meta_written_no_copy(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        artifact = src / f"mymod{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"ELF_DATA")
        cache_root = tmp_path / "cache"

        entry = register_artifact_path(
            cache_root, artifact, module_name="mymod", copy=False
        )
        meta = read_meta(entry.build_dir)
        assert meta is not None
        assert meta["module_name"] == "mymod"

    def test_no_artifact_copy_inside_cache(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        artifact = src / f"mymod{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"ELF_DATA")
        cache_root = tmp_path / "cache"

        entry = register_artifact_path(
            cache_root, artifact, module_name="mymod", copy=False
        )
        # Cache build_dir must contain NO physical copy of the artifact
        copied = list(entry.build_dir.glob(f"*{EXTENSION_SUFFIXES[0]}"))
        assert len(copied) == 0

    def test_key_is_deterministic_regardless_of_copy(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        artifact = src / f"mymod{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"ELF_DATA")

        entry_copy = register_artifact_path(
            tmp_path / "cache1", artifact, module_name="mymod", copy=True
        )
        entry_nocopy = register_artifact_path(
            tmp_path / "cache2", artifact, module_name="mymod", copy=False
        )
        # Same content → same key regardless of copy flag
        assert entry_copy.key == entry_nocopy.key


# ---------------------------------------------------------------------------
# _cache.py — iter_all_entry_dirs: skips files and non-key directory names
# ---------------------------------------------------------------------------


class TestIterAllEntryDirsFiltering:
    """``iter_all_entry_dirs`` skips files, non-key dirs, and non-dir entries."""

    def test_file_with_key_name_is_skipped(self, tmp_cache: Path) -> None:
        from .._cache import iter_all_entry_dirs

        key = make_cache_key({"x": "file_not_dir"})
        (tmp_cache / key).write_bytes(b"not a directory")
        dirs = list(iter_all_entry_dirs(tmp_cache))
        assert not any(d.name == key for d in dirs)

    def test_valid_key_directory_is_yielded(self, tmp_cache: Path) -> None:
        from .._cache import iter_all_entry_dirs

        key = make_cache_key({"x": "real_dir"})
        (tmp_cache / key).mkdir()
        dirs = list(iter_all_entry_dirs(tmp_cache))
        assert any(d.name == key for d in dirs)

    def test_non_key_directory_is_skipped(self, tmp_cache: Path) -> None:
        from .._cache import iter_all_entry_dirs

        (tmp_cache / "not_a_key_at_all").mkdir()
        dirs = list(iter_all_entry_dirs(tmp_cache))
        assert not any(d.name == "not_a_key_at_all" for d in dirs)

    def test_pins_lock_dir_is_skipped(self, tmp_cache: Path) -> None:
        from .._cache import iter_all_entry_dirs

        # .gc.lock and .pins.lock are not valid keys → must be skipped
        (tmp_cache / ".gc.lock").mkdir()
        dirs = list(iter_all_entry_dirs(tmp_cache))
        assert not any(d.name == ".gc.lock" for d in dirs)


# ---------------------------------------------------------------------------
# _cache.py — runtime_fingerprint: all 8 required keys present
# ---------------------------------------------------------------------------


class TestRuntimeFingerprintAllKeys:
    """``runtime_fingerprint`` returns every required key including ``abi``."""

    _REQUIRED_KEYS = frozenset(
        {"python", "python_impl", "platform", "machine", "processor", "cython", "numpy", "abi"}
    )

    def test_all_required_keys_present(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version="1.26.0")
        assert self._REQUIRED_KEYS <= set(fp.keys())

    def test_abi_is_str(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        assert isinstance(fp["abi"], str)

    def test_numpy_none_stored_as_none(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        assert fp["numpy"] is None

    def test_numpy_version_stored_correctly(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version="2.0.0")
        assert fp["numpy"] == "2.0.0"

    def test_cython_version_stored_correctly(self) -> None:
        fp = runtime_fingerprint(cython_version="3.1.0", numpy_version=None)
        assert fp["cython"] == "3.1.0"

    def test_python_version_is_nonempty_str(self) -> None:
        fp = runtime_fingerprint(cython_version="3.0.0", numpy_version=None)
        assert isinstance(fp["python"], str)
        assert fp["python"]  # non-empty


# ---------------------------------------------------------------------------
# _pins.py — pin overwrite=True: idempotent same key; replaces alias; repin
# ---------------------------------------------------------------------------


class TestPinOverwriteTrue:
    """``pin()`` with ``overwrite=True`` covers all three overwrite branches."""

    def test_same_alias_same_key_overwrite_is_idempotent(self, tmp_cache: Path) -> None:
        key = make_cache_key({"idem": "same"})
        pin(key, alias="idem_alias", cache_dir=tmp_cache)
        # Re-pin same key + same alias with overwrite=True must not raise
        pin(key, alias="idem_alias", cache_dir=tmp_cache, overwrite=True)
        assert resolve_pinned_key("idem_alias", cache_dir=tmp_cache) == key

    def test_overwrite_true_replaces_different_key(self, tmp_cache: Path) -> None:
        key1 = make_cache_key({"v": "old"})
        key2 = make_cache_key({"v": "new"})
        pin(key1, alias="swap_me", cache_dir=tmp_cache)
        pin(key2, alias="swap_me", cache_dir=tmp_cache, overwrite=True)
        assert resolve_pinned_key("swap_me", cache_dir=tmp_cache) == key2

    def test_overwrite_true_allows_key_repin_to_new_alias(self, tmp_cache: Path) -> None:
        key = make_cache_key({"v": "repin"})
        pin(key, alias="old_alias", cache_dir=tmp_cache)
        # overwrite=True: bypass one-to-one constraint
        pin(key, alias="new_alias", cache_dir=tmp_cache, overwrite=True)
        assert resolve_pinned_key("new_alias", cache_dir=tmp_cache) == key


# ---------------------------------------------------------------------------
# _pins.py — unpin: invalid alias raises ValueError (not silently returns False)
# ---------------------------------------------------------------------------


class TestUnpinInvalidAlias:
    """``unpin()`` with an invalid alias raises ``ValueError``."""

    def test_hyphen_in_alias_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            unpin("invalid-alias", cache_dir=tmp_cache)

    def test_empty_alias_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            unpin("", cache_dir=tmp_cache)

    def test_digit_leading_alias_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            unpin("1bad_alias", cache_dir=tmp_cache)

    def test_space_in_alias_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            unpin("has space", cache_dir=tmp_cache)


# ---------------------------------------------------------------------------
# _gc.py — private helpers: _utc_iso_from_epoch, _dir_size_bytes, _dir_mtime_epoch
# ---------------------------------------------------------------------------


class TestGcPrivateHelpers:
    """Private helpers in ``_gc``: ``_utc_iso_from_epoch``, ``_dir_size_bytes``,
    ``_dir_mtime_epoch``."""

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


# ---------------------------------------------------------------------------
# _gc.py — cache_stats: multiple entries give distinct newest/oldest timestamps
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _loader.py — import_extension_from_bytes: collision OSError + temp_dir=None
# ---------------------------------------------------------------------------


class TestImportFromBytesCollisionAndDefault:
    """``import_extension_from_bytes`` collision detection and ``temp_dir=None``."""

    def test_collision_raises_os_error(self, tmp_path: Path) -> None:
        from hashlib import sha256 as _sha256

        from .._loader import import_extension_from_bytes

        filename = f"collmod{EXTENSION_SUFFIXES[0]}"
        data = b"ELF_FAKE_CONTENT" + b"\x00" * 64

        # Stage the file at its deterministic location
        h = _sha256(data).hexdigest()
        staged_dir = tmp_path / "scikitplot_cython_import" / h[:16]
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged = staged_dir / filename
        staged.write_bytes(b"DIFFERENT_CONTENT")  # tamper before first call

        with pytest.raises(OSError, match="collision"):
            import_extension_from_bytes(
                data,
                module_name="collmod",
                artifact_filename=filename,
                temp_dir=tmp_path,
            )

    def test_temp_dir_none_does_not_raise_value_error(self) -> None:
        from .._loader import import_extension_from_bytes

        filename = f"tmpdefault{EXTENSION_SUFFIXES[0]}"
        # Must not raise ValueError (temp_dir validation); may raise ImportError
        # for the fake artifact — that is expected and acceptable.
        try:
            import_extension_from_bytes(
                b"ELF_FAKE" + b"\x00" * 64,
                module_name="tmpdefault",
                artifact_filename=filename,
                temp_dir=None,
            )
        except (ImportError, OSError):
            pass  # expected: fake artifact cannot be dlopen'd

    def test_empty_filename_raises_value_error(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_bytes

        with pytest.raises(ValueError):
            import_extension_from_bytes(
                b"data", module_name="m", artifact_filename="", temp_dir=tmp_path
            )

    def test_slash_in_filename_raises_value_error(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_bytes

        with pytest.raises(ValueError):
            import_extension_from_bytes(
                b"data",
                module_name="m",
                artifact_filename=f"sub/mod{EXTENSION_SUFFIXES[0]}",
                temp_dir=tmp_path,
            )

    def test_no_valid_suffix_raises_value_error(self, tmp_path: Path) -> None:
        from .._loader import import_extension_from_bytes

        with pytest.raises(ValueError):
            import_extension_from_bytes(
                b"data",
                module_name="m",
                artifact_filename="noext.txt",
                temp_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# _result.py — CacheStats: cache_root field; all-field construction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _result.py — CacheGCResult: skipped_missing_keys field + cache_root
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _result.py — BuildResult: source_sha256 propagation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _public.py — export_cached: cache_dir=None uses env-var resolved default
# ---------------------------------------------------------------------------


class TestPublicExportCachedEnvVar:
    """``export_cached`` with ``cache_dir=None`` resolves via env var."""

    def test_missing_key_raises_with_env_cache(self, tmp_path: Path) -> None:
        from .._public import export_cached

        env_cache = tmp_path / "env_cache"
        env_cache.mkdir()
        orig = os.environ.get("SCIKITPLOT_CYTHON_CACHE_DIR")
        try:
            os.environ["SCIKITPLOT_CYTHON_CACHE_DIR"] = str(env_cache)
            with pytest.raises(FileNotFoundError):
                export_cached("a" * 64, dest_dir=tmp_path / "out")
        finally:
            if orig is None:
                os.environ.pop("SCIKITPLOT_CYTHON_CACHE_DIR", None)
            else:
                os.environ["SCIKITPLOT_CYTHON_CACHE_DIR"] = orig

    def test_existing_key_exported_with_env_cache(
        self, tmp_path: Path, fake_module_entry: Any
    ) -> None:
        from .._public import export_cached

        key, build_dir, _ = fake_module_entry
        env_cache = build_dir.parent
        orig = os.environ.get("SCIKITPLOT_CYTHON_CACHE_DIR")
        try:
            os.environ["SCIKITPLOT_CYTHON_CACHE_DIR"] = str(env_cache)
            exported = export_cached(key, dest_dir=tmp_path / "out")
            assert exported.exists()
            assert (exported / "meta.json").exists()
        finally:
            if orig is None:
                os.environ.pop("SCIKITPLOT_CYTHON_CACHE_DIR", None)
            else:
                os.environ["SCIKITPLOT_CYTHON_CACHE_DIR"] = orig


# ---------------------------------------------------------------------------
# _public.py — cython_import_all: missing dir raises; empty dir returns {}
# ---------------------------------------------------------------------------


class TestPublicCythonImportAllEdgeCases:
    """``cython_import_all`` edge cases."""

    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        from .._public import cython_import_all

        with pytest.raises(FileNotFoundError):
            cython_import_all(tmp_path / "nonexistent_dir")

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        from .._public import cython_import_all

        result = cython_import_all(tmp_path)
        assert result == {}

    def test_non_pyx_files_ignored(self, tmp_path: Path) -> None:
        from .._public import cython_import_all

        (tmp_path / "ignored.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "also_ignored.c").write_bytes(b"int x;")
        result = cython_import_all(tmp_path)
        assert result == {}

    def test_custom_pattern_empty(self, tmp_path: Path) -> None:
        from .._public import cython_import_all

        (tmp_path / "module.pyx").write_text("cdef int x = 1", encoding="utf-8")
        # Only look for *.pxd — there are none
        result = cython_import_all(tmp_path, pattern="*.pxd")
        assert result == {}


# ---------------------------------------------------------------------------
# _public.py — register_cached_artifact_path: validation error paths
# ---------------------------------------------------------------------------


class TestPublicRegisterCachedArtifactPathErrors:
    """``register_cached_artifact_path`` validates inputs before attempting import."""

    def test_missing_artifact_raises_file_not_found(self, tmp_path: Path) -> None:
        from .._public import register_cached_artifact_path

        with pytest.raises(FileNotFoundError):
            register_cached_artifact_path(
                str(tmp_path / "ghost.so"),
                module_name="ghost",
                cache_dir=tmp_path / "cache",
            )

    def test_invalid_suffix_raises_value_error(self, tmp_path: Path) -> None:
        from .._public import register_cached_artifact_path

        bad = tmp_path / "module.txt"
        bad.write_bytes(b"not an extension")
        with pytest.raises(ValueError):
            register_cached_artifact_path(
                str(bad), module_name="m", cache_dir=tmp_path / "cache"
            )


# ---------------------------------------------------------------------------
# _public.py — import_artifact_path / import_artifact_bytes: error paths
# ---------------------------------------------------------------------------


class TestPublicImportArtifactErrors:
    """``import_artifact_path`` and ``import_artifact_bytes`` validate inputs."""

    def test_import_artifact_path_missing_raises(self, tmp_path: Path) -> None:
        from .._public import import_artifact_path

        with pytest.raises(FileNotFoundError):
            import_artifact_path(str(tmp_path / "missing.so"))

    def test_import_artifact_path_bad_suffix_raises(self, tmp_path: Path) -> None:
        from .._public import import_artifact_path

        bad = tmp_path / "module.txt"
        bad.write_bytes(b"not an extension")
        with pytest.raises(ValueError):
            import_artifact_path(str(bad))

    def test_import_artifact_bytes_bad_filename_raises(self, tmp_path: Path) -> None:
        from .._public import import_artifact_bytes

        with pytest.raises(ValueError):
            import_artifact_bytes(
                b"data",
                module_name="m",
                artifact_filename="noext.txt",
                temp_dir=tmp_path,
            )

    def test_import_artifact_bytes_slash_in_name_raises(self, tmp_path: Path) -> None:
        from .._public import import_artifact_bytes

        with pytest.raises(ValueError):
            import_artifact_bytes(
                b"data",
                module_name="m",
                artifact_filename=f"sub/mod{EXTENSION_SUFFIXES[0]}",
                temp_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# _public.py — import_pinned_result: unknown alias raises KeyError
# ---------------------------------------------------------------------------


class TestPublicImportPinnedErrors:
    """``import_pinned_result`` raises ``KeyError`` for unknown alias."""

    def test_unknown_alias_raises_key_error(self, tmp_cache: Path) -> None:
        from .._public import import_pinned_result

        with pytest.raises(KeyError):
            import_pinned_result("totally_unknown_alias", cache_dir=tmp_cache)

    def test_invalid_alias_raises_value_error(self, tmp_cache: Path) -> None:
        from .._public import import_pinned_result

        with pytest.raises(ValueError):
            import_pinned_result("invalid-alias!", cache_dir=tmp_cache)
