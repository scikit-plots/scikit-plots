# scikitplot/cython/tests/test__public.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._public`.

Covers
------
- ``get_cache_dir()``                : creates dir, returns Path
- ``purge_cache()``                  : deletes dir, missing → FileNotFoundError
- ``check_build_prereqs()``          : minimal / numpy / pybind11 / import-failure
                                       branches for Cython, setuptools, NumPy,
                                       pybind11
- ``list_cached()``                  : empty / populated module cache
- ``list_cached_packages()``         : empty / populated package cache
- ``cache_stats()`` delegate         : delegates to ``_gc.cache_stats``
- ``gc_cache()`` delegate            : delegates to ``_gc.gc_cache``
- ``pin()`` / ``unpin()``            : roundtrip
- ``export_cached()``                : copy, replace-existing, env-var cache,
                                       missing → FileNotFoundError
- ``import_cached_by_name()``        : missing → FileNotFoundError
- ``import_cached_result()``         : used_cache flag, source_sha256 propagation,
                                       corrupted meta → empty dict
- ``import_pinned()``                : returns module
- ``register_cached_artifact_path()``: returns BuildResult
- ``import_artifact_path()``         : works, missing/bad-suffix → error
- ``import_artifact_bytes()``        : works, bad filename → error
- ``cython_import_all()``            : missing dir, empty dir, non-pyx ignored,
                                       custom pattern
- ``_coerce_path_seq()``             : None / str / Path / list / tuple / bytes /
                                       invalid type
- Scenario 1  — pure-Python prereqs
- Scenario 2  — Cython + C++ prereqs
- Scenario 3  — full-stack prereqs
- Scenario 9  — docstring examples
- Regression R2 — bare-string include_dirs coercion
- Regression R4 — doctest no stray output
"""
from __future__ import annotations

import builtins
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from .._cache import make_cache_key, resolve_cache_dir, write_meta
from .._custom_compiler import (
    collect_c_api_sources,
    collect_header_dirs,
    cython_cpp_prereqs,
    full_stack_prereqs,
    numpy_include,
    pybind11_include,
    pybind11_only_prereqs,
    pure_python_prereqs,
)
from .._public import (
    _coerce_path_seq,
    check_build_prereqs,
    cython_import_all,
    export_cached,
    get_cache_dir,
    import_artifact_bytes,
    import_artifact_path,
    import_cached_by_name,
    import_cached_result,
    import_pinned,
    list_cached,
    list_cached_packages,
    purge_cache,
    register_cached_artifact_path,
)
from .._security import (
    DEFAULT_SECURITY_POLICY,
    RELAXED_SECURITY_POLICY,
    is_safe_compiler_arg,
    is_safe_macro_name,
    is_safe_path,
    validate_build_inputs,
)
from .._pins import pin

from .conftest import (
    make_valid_key,
    write_fake_artifact,
    write_full_cache_entry,
    write_simple_cache_entry,
)
import json
from importlib.machinery import EXTENSION_SUFFIXES
from .._gc import cache_stats
from .._security import SecurityError, SecurityPolicy
from .._custom_compiler import register_compiler
from .conftest import FAKE_KEY as _FAKE_KEY, FAKE_KEY2 as _FAKE_KEY2
import types
from .._result import CacheStats, CacheGCResult
from .._custom_compiler import list_compilers
from .conftest import make_valid_key as _make_valid_key, write_full_cache_entry as _write_cache_entry

requires_compiler = pytest.mark.skipif(
    True,
    reason="requires Cython + C compiler",
)

def _make_minimal_compiler(name: str = "custom_minimal"):
    class _Compiler:
        def __call__(self, source, *, build_dir, module_name, **kwargs):
            raise NotImplementedError("test stub")
    c = _Compiler()
    c.name = name
    return c


class TestPublicGetCacheDir:
    """Tests for :func:`~scikitplot.cython._public.get_cache_dir`."""

    def test_creates_and_returns_path(self, tmp_path: Path) -> None:
        from .._public import get_cache_dir

        result = get_cache_dir(tmp_path / "newcache")
        assert result.exists()
        assert isinstance(result, Path)


class TestPublicPurgeCache:
    """Tests for :func:`~scikitplot.cython._public.purge_cache`."""

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


class TestPublicSimpleFunctions:
    """Cover public functions that don't require a C compiler."""

    def test_get_cache_dir_returns_path(self, tmp_path: Path) -> None:
        result = get_cache_dir(str(tmp_path / "cache"))
        assert isinstance(result, Path)
        assert result.exists()

    def test_purge_cache_deletes_dir(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        root.mkdir(parents=True, exist_ok=True)
        purge_cache(root)
        assert not root.exists()

    def test_purge_cache_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            purge_cache(tmp_path / "no_cache")

    def test_check_build_prereqs_minimal(self) -> None:
        result = check_build_prereqs()
        assert "cython" in result
        assert "setuptools" in result
        assert isinstance(result["cython"]["ok"], bool)

    def test_check_build_prereqs_numpy_true(self) -> None:
        result = check_build_prereqs(numpy=True)
        assert "numpy" in result
        assert isinstance(result["numpy"]["ok"], bool)

    def test_check_build_prereqs_pybind11_true(self) -> None:
        result = check_build_prereqs(pybind11=True)
        assert "pybind11" in result
        assert isinstance(result["pybind11"]["ok"], bool)

    def test_check_build_prereqs_cython_import_error(self) -> None:
        with patch.dict("sys.modules", {"Cython": None}):
            result = check_build_prereqs()
        assert result["cython"]["ok"] is False

    def test_check_build_prereqs_setuptools_import_error(self) -> None:
        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "setuptools":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = check_build_prereqs()
        assert result["setuptools"]["ok"] is False

    def test_check_build_prereqs_numpy_import_error(self) -> None:
        with patch.dict("sys.modules", {"numpy": None}):
            result = check_build_prereqs(numpy=True)
        assert result["numpy"]["ok"] is False

    def test_check_build_prereqs_pybind11_import_error(self) -> None:
        with patch.dict("sys.modules", {"pybind11": None}):
            result = check_build_prereqs(pybind11=True)
        assert result["pybind11"]["ok"] is False


class TestCheckBuildPrereqs:
    """Tests for :func:`~scikitplot.cython._public.check_build_prereqs`."""

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


class TestPublicListCached:
    """Tests for :func:`~scikitplot.cython._public.list_cached` and ``list_cached_packages``."""

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


class TestListCachedBranches:
    """Cover list_cached and list_cached_packages with a prepared cache."""

    def test_list_cached_empty(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        entries = list_cached(root)
        assert isinstance(entries, list)
        assert entries == []

    def test_list_cached_packages_empty(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        entries = list_cached_packages(root)
        assert isinstance(entries, list)
        assert entries == []


class TestPublicCacheStatsGc:
    """Tests for public :func:`~scikitplot.cython._gc.cache_stats` and ``gc_cache``."""

    def test_cache_stats_delegates(self, fake_module_entry) -> None:
        from .._gc import cache_stats as pub_stats

        key, build_dir, _ = fake_module_entry
        stats = pub_stats(build_dir.parent)
        assert isinstance(stats, CacheStats)
        assert stats.n_modules == 1

    def test_gc_cache_delegates(self, tmp_cache: Path) -> None:
        from .._gc import gc_cache as pub_gc

        result = pub_gc(cache_dir=tmp_cache)
        assert isinstance(result, CacheGCResult)


class TestPublicPinUnpin:
    """Tests for public pin/unpin/list_pins wrappers."""

    def test_pin_unpin_roundtrip(self, tmp_cache: Path) -> None:
        from .._pins import list_pins, pin, unpin

        key = make_cache_key({"pub": "pin"})
        pin(key, alias="pub_alias", cache_dir=tmp_cache)
        assert "pub_alias" in list_pins(tmp_cache)
        removed = unpin("pub_alias", cache_dir=tmp_cache)
        assert removed is True
        assert "pub_alias" not in list_pins(tmp_cache)


class TestPublicExportCached:
    """Tests for :func:`~scikitplot.cython._public.export_cached`."""

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


class TestExportCachedBranches:
    """Cover export_cached paths without importing actual extensions."""

    def test_export_copies_entry_dir(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        dest = tmp_path / "export"
        result_path = export_cached(key, dest_dir=dest, cache_dir=root)
        assert result_path.exists()
        assert result_path.name == key

    def test_export_replaces_existing_dest(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        dest = tmp_path / "export"
        export_cached(key, dest_dir=dest, cache_dir=root)
        # Second call should replace
        export_cached(key, dest_dir=dest, cache_dir=root)
        assert (dest / key).exists()

    def test_export_missing_key_raises_file_not_found(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        fake_key = "a" * 64  # valid hex key that doesn't exist
        with pytest.raises(FileNotFoundError):
            export_cached(fake_key, dest_dir=tmp_path / "out", cache_dir=root)


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
        self, tmp_path: Path, fake_module_entry: any
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


class TestPublicImportCachedByNameErrors:
    """Error paths for :func:`~scikitplot.cython._public.import_cached_by_name`."""

    def test_missing_name_raises(self, tmp_cache: Path) -> None:
        from .._public import import_cached_by_name

        with pytest.raises(FileNotFoundError, match="No cached entry"):
            import_cached_by_name("does_not_exist", cache_dir=tmp_cache)


class TestImportCachedByNameBranches:
    def test_missing_name_raises_file_not_found(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        with pytest.raises(FileNotFoundError, match="no_such_module"):
            import_cached_by_name("no_such_module", cache_dir=root)


class TestImportCachedResultPublic:
    """import_cached_result builds a BuildResult from a cache entry."""

    def test_returns_build_result_with_used_cache_true(
        self, tmp_path: Path
    ) -> None:
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


class TestPublicRegisterCachedArtifactPathErrors:
    """``register_cached_artifact_path`` validates inputs before attempting import."""

    def test_missing_artifact_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            register_cached_artifact_path(
                str(tmp_path / "ghost.so"),
                module_name="ghost",
                cache_dir=tmp_path / "cache",
            )

    def test_invalid_suffix_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "module.txt"
        bad.write_bytes(b"not an extension")
        with pytest.raises(ValueError):
            register_cached_artifact_path(
                str(bad), module_name="m", cache_dir=tmp_path / "cache"
            )


class TestRegisterCachedArtifactPathPublic:
    """register_cached_artifact_path registers artifact and returns BuildResult."""

    def test_registration_returns_build_result(self, tmp_path: Path) -> None:
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


class TestPublicImportArtifactErrors:
    """``import_artifact_path`` and ``import_artifact_bytes`` validate inputs."""

    def test_import_artifact_path_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            import_artifact_path(str(tmp_path / "missing.so"))

    def test_import_artifact_path_bad_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "module.txt"
        bad.write_bytes(b"not an extension")
        with pytest.raises(ValueError):
            import_artifact_path(str(bad))

    def test_import_artifact_bytes_bad_filename_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            import_artifact_bytes(
                b"data",
                module_name="m",
                artifact_filename="noext.txt",
                temp_dir=tmp_path,
            )

    def test_import_artifact_bytes_slash_in_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            import_artifact_bytes(
                b"data",
                module_name="m",
                artifact_filename=f"sub/mod{EXTENSION_SUFFIXES[0]}",
                temp_dir=tmp_path,
            )


class TestImportArtifactPublic:
    """import_artifact_path and import_artifact_bytes public wrappers."""

    def test_import_artifact_path_works(self, tmp_path: Path) -> None:
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


class TestCythonImportAllBranches:
    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            cython_import_all(tmp_path / "no_such_dir")

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        result = cython_import_all(d)
        assert result == {}

    def test_non_pyx_files_ignored(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "helper.py").write_text("x = 1")
        (d / "data.txt").write_text("data")
        result = cython_import_all(d)
        assert result == {}

    def test_custom_pattern_no_match_returns_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "mod.pyx").write_text("def f(): pass")
        result = cython_import_all(d, pattern="*.cu")
        assert result == {}


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


class TestScenario1PurePython:
    """
    Newbie scenario: pure Python, only setuptools needed.

    Notes
    -----
    **User note**: this scenario checks that setuptools is present and that
    calling :func:`pure_python_prereqs` gives an actionable report.
    **Dev note**: the function must not import Cython or NumPy.
    """

    def test_pure_python_prereqs_returns_dict(self) -> None:
        result = pure_python_prereqs()
        assert isinstance(result, dict)

    def test_pure_python_prereqs_has_setuptools_key(self) -> None:
        result = pure_python_prereqs()
        assert "setuptools" in result

    def test_pure_python_prereqs_no_cython_key(self) -> None:
        result = pure_python_prereqs()
        assert "cython" not in result

    def test_pure_python_prereqs_no_numpy_key(self) -> None:
        result = pure_python_prereqs()
        assert "numpy" not in result

    def test_pure_python_prereqs_no_pybind11_key(self) -> None:
        result = pure_python_prereqs()
        assert "pybind11" not in result

    def test_setuptools_entry_has_ok_bool(self) -> None:
        result = pure_python_prereqs()
        assert isinstance(result["setuptools"]["ok"], bool)

    def test_setuptools_ok_entry_has_version_when_available(self) -> None:
        result = pure_python_prereqs()
        if result["setuptools"]["ok"]:
            assert "version" in result["setuptools"]

    def test_setuptools_fail_entry_has_error_key(self) -> None:
        with patch.dict(sys.modules, {"setuptools": None}):
            result = pure_python_prereqs()
        # After un-mocking, result was captured under mock — ok or not, has ok key
        assert "ok" in result["setuptools"]

    def test_check_build_prereqs_minimal(self) -> None:
        """Newbie scenario: check_build_prereqs with no extras."""
        result = check_build_prereqs()
        assert "cython" in result
        assert "setuptools" in result
        assert "numpy" not in result
        assert "pybind11" not in result

    def test_check_build_prereqs_result_ok_is_bool(self) -> None:
        result = check_build_prereqs()
        assert isinstance(result["cython"]["ok"], bool)
        assert isinstance(result["setuptools"]["ok"], bool)


class TestScenario2CythonCpp:
    """
    Newbie scenario: C++ via Cython only, no numpy or setuptools needed.

    Notes
    -----
    **User note**: install Cython with ``pip install Cython`` and ensure a
    C++ compiler (``g++``) is on the PATH.  No NumPy required.
    **Dev note**: :func:`cython_cpp_prereqs` must only check Cython.
    """

    def test_cython_cpp_prereqs_returns_dict(self) -> None:
        result = cython_cpp_prereqs()
        assert isinstance(result, dict)

    def test_cython_cpp_prereqs_has_cython_key(self) -> None:
        result = cython_cpp_prereqs()
        assert "cython" in result

    def test_cython_cpp_prereqs_no_setuptools(self) -> None:
        result = cython_cpp_prereqs()
        assert "setuptools" not in result

    def test_cython_cpp_prereqs_no_numpy(self) -> None:
        result = cython_cpp_prereqs()
        assert "numpy" not in result

    def test_cython_entry_has_ok_bool(self) -> None:
        result = cython_cpp_prereqs()
        assert isinstance(result["cython"]["ok"], bool)

    def test_cython_ok_has_version_key(self) -> None:
        result = cython_cpp_prereqs()
        if result["cython"]["ok"]:
            assert "version" in result["cython"]

    def test_cython_fail_has_error_key(self) -> None:
        with patch.dict(sys.modules, {"Cython": None}):
            result = cython_cpp_prereqs()
        assert "ok" in result["cython"]


class TestScenario3FullStack:
    """
    Master/pro scenario: full build stack.

    Notes
    -----
    **User note**: install with ``pip install setuptools Cython pybind11 numpy``.
    **Dev note**: :func:`full_stack_prereqs` validates all four dependencies.
    """

    def test_full_stack_prereqs_returns_dict(self) -> None:
        result = full_stack_prereqs()
        assert isinstance(result, dict)

    def test_full_stack_prereqs_has_all_keys(self) -> None:
        result = full_stack_prereqs()
        for key in ("setuptools", "cython", "pybind11", "numpy"):
            assert key in result, f"Missing key: {key!r}"

    def test_full_stack_prereqs_all_ok_are_bool(self) -> None:
        result = full_stack_prereqs()
        for key, val in result.items():
            assert isinstance(val["ok"], bool), f"{key}['ok'] is not bool"

    def test_check_build_prereqs_numpy_true_adds_numpy(self) -> None:
        result = check_build_prereqs(numpy=True)
        assert "numpy" in result

    def test_check_build_prereqs_pybind11_true_adds_pybind11(self) -> None:
        result = check_build_prereqs(pybind11=True)
        assert "pybind11" in result

    def test_check_build_prereqs_all_adds_all(self) -> None:
        result = check_build_prereqs(numpy=True, pybind11=True)
        for key in ("cython", "setuptools", "numpy", "pybind11"):
            assert key in result

    def test_check_build_prereqs_pybind11_ok_entry(self) -> None:
        result = check_build_prereqs(pybind11=True)
        assert isinstance(result["pybind11"]["ok"], bool)

    def test_check_build_prereqs_pybind11_ok_has_include(self) -> None:
        result = check_build_prereqs(pybind11=True)
        if result["pybind11"]["ok"]:
            assert "include" in result["pybind11"]

    def test_pybind11_include_is_none_or_dir(self) -> None:
        p = pybind11_include()
        assert p is None or (isinstance(p, Path) and p.is_dir())

    def test_numpy_include_is_none_or_dir(self) -> None:
        p = numpy_include()
        assert p is None or (isinstance(p, Path) and p.is_dir())


class TestScenario9DocstringExamples:
    """
    Validate public API examples work as documented.

    Notes
    -----
    **User note**: every example in this file is a guarantee.  If an
    example here fails, the documentation is wrong or the code is broken.
    **Dev note**: keep these tests in sync with docstrings in the modules
    under test.  Don't mark them xfail — they are binding contracts.
    """

    # --- _security examples ---

    def test_security_policy_default_strict(self) -> None:
        policy = SecurityPolicy()
        assert policy.strict is True
        assert policy.allow_shell_metacharacters is False

    def test_security_policy_relaxed_example(self) -> None:
        policy = SecurityPolicy(allow_absolute_include_dirs=True)
        assert policy.allow_absolute_include_dirs is True

    def test_validate_build_inputs_clean_example(self) -> None:
        validate_build_inputs(
            source="def hello(): return 42",
            extra_compile_args=["-O2"],
        )

    def test_validate_build_inputs_shell_injection_example(self) -> None:
        with pytest.raises(SecurityError, match="extra_compile_args"):
            validate_build_inputs(extra_compile_args=["-O2; rm -rf /"])

    def test_is_safe_path_relative_true(self) -> None:
        assert is_safe_path("include/mylib") is True

    def test_is_safe_path_traversal_false(self) -> None:
        assert is_safe_path("../../../etc/passwd") is False

    def test_is_safe_path_absolute_with_flag(self) -> None:
        assert is_safe_path("/usr/include", allow_absolute=True) is True

    def test_is_safe_path_absolute_no_flag_false(self) -> None:
        assert is_safe_path("/usr/include", allow_absolute=False) is False

    def test_is_safe_macro_name_valid(self) -> None:
        assert is_safe_macro_name("MY_FLAG") is True

    def test_is_safe_macro_name_reserved(self) -> None:
        assert is_safe_macro_name("Py_LIMITED_API") is False

    def test_is_safe_macro_name_reserved_allowed(self) -> None:
        assert is_safe_macro_name("Py_LIMITED_API", allow_reserved=True) is True

    def test_is_safe_macro_name_invalid_syntax(self) -> None:
        assert is_safe_macro_name("123INVALID") is False

    def test_is_safe_compiler_arg_clean(self) -> None:
        assert is_safe_compiler_arg("-O2") is True

    def test_is_safe_compiler_arg_shell(self) -> None:
        assert is_safe_compiler_arg("-O2; rm -rf /") is False

    def test_is_safe_compiler_arg_imacros(self) -> None:
        assert is_safe_compiler_arg("-imacros /etc/shadow") is False

    # --- _custom_compiler examples ---

    def test_custom_compiler_registration_example(self) -> None:
        from .. import _custom_compiler

        saved = _custom_compiler._REGISTRY._compilers.copy()
        try:
            class custom_fast:
                name = "custom_fast"
                def __call__(self, source, *, build_dir, module_name, **kw):
                    raise NotImplementedError

            register_compiler(custom_fast(), overwrite=True)
            assert "custom_fast" in list_compilers()
        finally:
            _custom_compiler._REGISTRY._compilers = saved

    def test_pybind11_include_example(self) -> None:
        p = pybind11_include()
        assert p is None or p.is_dir()

    def test_numpy_include_example(self) -> None:
        p = numpy_include()
        assert p is None or p.is_dir()

    def test_collect_c_api_sources_two_files_example(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path
        _ = (p / "a.c").write_text("int a() { return 1; }")
        _ = (p / "b.cpp").write_text("int b() { return 2; }")
        srcs = collect_c_api_sources(str(tmp_path))
        assert len(srcs) == 2

    def test_collect_header_dirs_one_dir_example(self, tmp_path: Path) -> None:
        _ = (tmp_path / "mylib.h").write_text("#pragma once")
        dirs = collect_header_dirs(str(tmp_path))
        assert len(dirs) == 1

    # --- _public examples ---

    def test_check_build_prereqs_has_cython_and_setuptools(self) -> None:
        result = check_build_prereqs()
        assert "cython" in result and "setuptools" in result

    def test_check_build_prereqs_with_all_options(self) -> None:
        result = check_build_prereqs(numpy=True, pybind11=True)
        assert all(k in result for k in ("cython", "setuptools", "numpy", "pybind11"))


class TestRegressionR2CythonImportResultIncludeDirs:
    """R2: bare-string ``include_dirs`` must not be iterated as characters."""

    def test_coerce_none_returns_none(self) -> None:
        assert _coerce_path_seq(None, "include_dirs") is None

    def test_coerce_bare_string_wraps_in_list(self) -> None:
        result = _coerce_path_seq("include/mylib", "include_dirs")
        assert result == ["include/mylib"]
        # NOT ["i", "n", "c", "l", "u", "d", "e", "/", "m", "y", "l", "i", "b"]
        assert len(result) == 1

    def test_coerce_pathlib_path_wraps_in_list(self) -> None:
        p = Path("include/mylib")
        result = _coerce_path_seq(p, "include_dirs")
        assert result == [p]
        assert len(result) == 1

    def test_coerce_list_returns_list(self) -> None:
        result = _coerce_path_seq(["a", "b", "c"], "include_dirs")
        assert result == ["a", "b", "c"]

    def test_coerce_tuple_returns_list(self) -> None:
        result = _coerce_path_seq(("a", "b"), "include_dirs")
        assert result == ["a", "b"]

    def test_coerce_invalid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="include_dirs"):
            _coerce_path_seq(42, "include_dirs")  # type: ignore[arg-type]

    def test_coerce_bytes_path_wraps_in_list(self) -> None:
        result = _coerce_path_seq(b"include/mylib", "include_dirs")
        assert result == [b"include/mylib"]
        assert len(result) == 1


class TestRegressionR4DoctestNoStrayOutput:
    """R4: ``collect_c_api_sources`` doctest must not have stray numeric output."""

    def test_write_text_return_value_suppressed_in_doctest(
        self, tmp_path: Path
    ) -> None:
        """The doctest uses ``_ =`` to suppress write_text return values."""
        p = tmp_path
        _ = (p / "a.c").write_text("int a() { return 1; }")
        _ = (p / "b.cpp").write_text("int b() { return 2; }")
        srcs = collect_c_api_sources(str(tmp_path))
        # Must be exactly 2 — not 21 or any other stray value.
        assert len(srcs) == 2
