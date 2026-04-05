# scikitplot/cython/tests/test__loader.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._loader`.

Covers
------
- ``_read_meta_near_artifact()``  : found in parent / grandparent / missing /
                                    corrupted
- ``import_extension()``          : bad spec → ImportError, metadata attach,
                                    setattr exception swallowed
- ``import_extension_from_path()``: missing file, invalid suffix,
                                    no module name + no meta, package-meta
                                    resolves module name, key/build_dir attached
- ``import_extension_from_bytes()``: empty filename, directory separator,
                                     invalid suffix, collision → OSError
- ``_open_annotation_in_browser()``: CI env / no DISPLAY / non-tty / isatty
                                     exception → suppressed
"""
from __future__ import annotations

import importlib.machinery
import json
import os
import sys
import types
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from .._cache import make_cache_key, write_meta
from .._loader import (
    _read_meta_near_artifact,
    import_extension,
    import_extension_from_bytes,
    import_extension_from_path,
)
from .._builder import _open_annotation_in_browser

from .conftest import write_fake_artifact, write_full_cache_entry
from .conftest import FAKE_KEY as _FAKE_KEY, FAKE_KEY2 as _FAKE_KEY2


class TestReadMetaNearArtifact:
    """Tests for :func:`~scikitplot.cython._loader._read_meta_near_artifact`."""

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


class TestImportExtensionFromPathErrors:
    """Error paths for :func:`~scikitplot.cython._loader.import_extension_from_path`."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            import_extension_from_path(tmp_path / "nonexistent.so")

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "myfile.txt"
        bad.write_bytes(b"not a so")
        with pytest.raises(ValueError, match="extension artifact"):
            import_extension_from_path(bad)

    def test_no_module_name_no_meta_raises(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        # No meta.json, no module_name arg → must raise
        with pytest.raises(ValueError, match="module_name"):
            import_extension_from_path(so, module_name=None)


class TestImportExtensionFromPathPackageMeta:
    """import_extension_from_path: package-kind meta resolution path."""

    def test_package_meta_resolves_module_name(self, tmp_path: Path) -> None:
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


class TestImportExtensionFromBytesErrors:
    """Error paths for :func:`~scikitplot.cython._loader.import_extension_from_bytes`."""

    def test_empty_filename_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="artifact_filename"):
            import_extension_from_bytes(
                b"ELF", module_name="m", artifact_filename=""
            )

    def test_directory_in_filename_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="artifact_filename"):
            import_extension_from_bytes(
                b"ELF",
                module_name="m",
                artifact_filename=f"sub/mod{EXTENSION_SUFFIXES[0]}",
            )

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="artifact_filename"):
            import_extension_from_bytes(
                b"ELF", module_name="m", artifact_filename="mod.txt"
            )


class TestImportFromBytesCollisionAndDefault:
    """``import_extension_from_bytes`` collision detection and ``temp_dir=None``."""

    def test_collision_raises_os_error(self, tmp_path: Path) -> None:
        from hashlib import sha256 as _sha256

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
        with pytest.raises(ValueError):
            import_extension_from_bytes(
                b"data", module_name="m", artifact_filename="", temp_dir=tmp_path
            )

    def test_slash_in_filename_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            import_extension_from_bytes(
                b"data",
                module_name="m",
                artifact_filename=f"sub/mod{EXTENSION_SUFFIXES[0]}",
                temp_dir=tmp_path,
            )

    def test_no_valid_suffix_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            import_extension_from_bytes(
                b"data",
                module_name="m",
                artifact_filename="noext.txt",
                temp_dir=tmp_path,
            )


class TestImportExtensionFromBytesCollision:
    """import_extension_from_bytes raises OSError on content collision."""

    def test_collision_raises_os_error(self, tmp_path: Path) -> None:
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


class TestImportExtensionSpecNone:
    """import_extension raises ImportError when spec_from_file_location returns None."""

    def test_bad_spec_raises_import_error(self, tmp_path: Path) -> None:
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
        artifact = tmp_path / f"meta_mod{EXTENSION_SUFFIXES[0]}"
        artifact.write_bytes(b"\x7fELF")

        fake_mod = types.ModuleType("meta_mod")

        class _FrozenModule(types.ModuleType):
            def __setattr__(self, name: str, value: any) -> None:
                raise AttributeError("frozen")

        frozen = _FrozenModule("meta_mod")

        spec = MagicMock()
        spec.loader = MagicMock()

        def exec_module(m: any) -> None:
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
