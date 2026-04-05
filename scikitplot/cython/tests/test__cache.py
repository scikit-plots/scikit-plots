# scikitplot/cython/tests/test__cache.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._cache`.

Covers
------
- Cache directory resolution: ``peek_cache_dir``, ``resolve_cache_dir``,
  ``_default_cache_dir`` (XDG / HOME / Windows branches), env-var override
- Meta I/O: ``write_meta`` / ``read_meta`` (atomic, roundtrip, corrupted)
- Entry iteration: ``iter_cache_entries``, ``iter_package_entries``,
  ``iter_all_entry_dirs`` (filtering, list semantics, fingerprint paths)
- Lookup: ``find_entry_by_key``, ``find_package_entry_by_key``,
  ``find_entries_by_name``
- Registration: ``register_artifact_path`` (copy/no-copy, collision, idempotent)
- Private helpers: ``_sha256_file``, ``_utc_iso``, ``_guess_artifact``,
  ``_guess_module_name``, ``_artifact_from_meta_or_guess``,
  ``_module_name_from_meta_or_guess``
"""
from __future__ import annotations

import json
import os
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from unittest.mock import patch

import pytest

from .._cache import (
    CacheEntry,
    PackageCacheEntry,
    _artifact_from_meta_or_guess,
    _default_cache_dir,
    _guess_artifact,
    _guess_module_name,
    _module_name_from_meta_or_guess,
    _sha256_file,
    _stable_repr,
    _utc_iso,
    find_entries_by_name,
    find_entry_by_key,
    find_package_entry_by_key,
    is_valid_key,
    iter_all_entry_dirs,
    iter_cache_entries,
    iter_package_entries,
    make_cache_key,
    peek_cache_dir,
    read_meta,
    register_artifact_path,
    resolve_cache_dir,
    write_meta,
)

from .conftest import (
    make_valid_key,
    write_fake_artifact,
    write_full_cache_entry,
    write_package_cache_entry,
    write_simple_cache_entry,
)

# ---------------------------------------------------------------------------
# Module-level helpers (aliases to conftest utilities for legacy test classes)
# ---------------------------------------------------------------------------
from .conftest import make_valid_key as _make_valid_key, FAKE_KEY as _FAKE_KEY, FAKE_KEY2 as _FAKE_KEY2
from .._builder import _support_paths_digest
from .._cache import source_digest
from .conftest import write_package_cache_entry as _write_package_cache_entry
# _write_cache_entry in this file comes from ext-style (write_full_cache_entry)
_write_cache_entry = write_full_cache_entry



class TestCacheDirResolution:
    """Tests for :func:`~scikitplot.cython._cache.resolve_cache_dir` and ``peek_cache_dir``."""

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


class TestCacheDirEnvVarBranch:
    def test_env_var_overrides_default(self, tmp_path: Path) -> None:
        custom = str(tmp_path / "custom_cache")
        with patch.dict(os.environ, {"SKPLT_CYTHON_CACHE_DIR": custom}):
            result = resolve_cache_dir(None)
        assert str(result) == custom

    def test_peek_does_not_create_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "peek_target"
        result = peek_cache_dir(str(target))
        assert not result.exists()

    def test_resolve_creates_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "resolve_target"
        result = resolve_cache_dir(str(target))
        assert result.exists()

    def test_none_uses_platform_default(self) -> None:
        result = peek_cache_dir(None)
        assert isinstance(result, Path)


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


class TestWriteReadMeta:
    """Tests for :func:`~scikitplot.cython._cache.write_meta` and ``read_meta``."""

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


class TestGuessArtifactAbsPath:
    """``_artifact_from_meta_or_guess`` with absolute artifact path in meta."""

    def test_absolute_path_in_meta(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        # meta stores absolute path
        meta = {"artifact": str(so)}
        result = _artifact_from_meta_or_guess(tmp_path / "somedir", meta)
        assert result == so


class TestIterCacheEntries:
    """Tests for :func:`~scikitplot.cython._cache.iter_cache_entries`."""

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


class TestIterPackageEntries:
    """Tests for :func:`~scikitplot.cython._cache.iter_package_entries`."""

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


class TestIterAllEntryDirsCoverage:
    def test_non_key_dir_skipped(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        (root / "not_a_key").mkdir()
        dirs = iter_all_entry_dirs(root)
        assert all(is_valid_key(d.name) for d in dirs)

    def test_valid_key_dir_yielded(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        (root / key).mkdir()
        dirs = iter_all_entry_dirs(root)
        assert any(d.name == key for d in dirs)

    def test_file_with_key_name_skipped(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        (root / key).write_bytes(b"file_not_dir")  # file, not dir
        dirs = iter_all_entry_dirs(root)
        assert not any(d.name == key for d in dirs)

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        dirs = iter_all_entry_dirs(tmp_path / "no_cache")
        assert dirs == []

    def test_returns_list_iterable_twice(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        (root / key).mkdir()
        dirs = iter_all_entry_dirs(root)
        assert isinstance(dirs, list)
        assert len(dirs) == len(dirs)  # list is reusable


class TestIterAllEntryDirsMissingRoot:
    """``iter_all_entry_dirs`` with a root that doesn't exist."""

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        from .._cache import iter_all_entry_dirs

        result = list(iter_all_entry_dirs(tmp_path / "nonexistent"))
        assert result == []


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


class TestFindEntryByKey:
    """Tests for :func:`~scikitplot.cython._cache.find_entry_by_key`."""

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


class TestFindPackageEntryByKey:
    """Tests for :func:`~scikitplot.cython._cache.find_package_entry_by_key`."""

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


class TestFindEntriesByName:
    """Tests for :func:`~scikitplot.cython._cache.find_entries_by_name`."""

    def test_finds_by_name(self, fake_module_entry) -> None:
        key, build_dir, _ = fake_module_entry
        root = build_dir.parent
        result = find_entries_by_name(root, "mymod")
        assert len(result) == 1
        assert result[0].module_name == "mymod"

    def test_no_match_returns_empty(self, tmp_cache: Path) -> None:
        assert find_entries_by_name(tmp_cache, "noexist") == []


class TestRegisterArtifactPath:
    """Tests for :func:`~scikitplot.cython._cache.register_artifact_path`."""

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
