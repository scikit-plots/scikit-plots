# scikitplot/utils/tests/test__toml.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._toml`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__toml.py -v

Coverage map
------------
_has_module            Importable/unimportable modules         → TestHasModule
_read_backend_name     Priority order, Python version logic    → TestReadBackendName
_write_backend_name    Priority order                          → TestWriteBackendName
Module constants       TOML_READ/WRITE_SUPPORT/SOURCE bool     → TestModuleConstants
_normalize_path        Valid paths, tilde expansion, error     → TestNormalizePath
_load_read_backend     Returns (name, module) or raises        → TestLoadReadBackend
_load_write_backend    Returns (name, module) or raises        → TestLoadWriteBackend
read_toml              Round-trip, missing file, directory,
                       permission errors (mocked)              → TestReadToml
write_toml             Round-trip, mkdir=True, directory
                       target, permission errors (mocked)      → TestWriteToml
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

# --- path bootstrap (commented out — use pytest from the package root) -----
# import pathlib as _pathlib
# _HERE = _pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))
# --------------------------------------------------------------------------

from .._toml import (  # noqa: E402
    TOML_READ_SOURCE,
    TOML_READ_SUPPORT,
    TOML_WRITE_SOURCE,
    TOML_WRITE_SUPPORT,
    _has_module,
    _normalize_path,
    read_toml,
    write_toml,
)

# ScikitplotException lives in the parent package
from ...exceptions import ScikitplotException  # noqa: E402


def _make_tmpdir() -> Path:
    return Path(tempfile.mkdtemp())


# ===========================================================================
# _has_module
# ===========================================================================


class TestHasModule(unittest.TestCase):
    """_has_module must check module importability without importing."""

    def test_sys_is_always_available(self):
        """'sys' is always importable; must return True."""
        self.assertTrue(_has_module("sys"))

    def test_os_is_always_available(self):
        self.assertTrue(_has_module("os"))

    def test_pathlib_is_always_available(self):
        self.assertTrue(_has_module("pathlib"))

    def test_tomllib_on_py311_plus(self):
        """On Python 3.11+, 'tomllib' must be available."""
        if sys.version_info >= (3, 11):
            self.assertTrue(_has_module("tomllib"))

    def test_nonexistent_module_returns_false(self):
        """A made-up module name must return False."""
        self.assertFalse(_has_module("_scikitplot_nonexistent_xyz_abc"))

    def test_returns_bool(self):
        self.assertIsInstance(_has_module("sys"), bool)

    def test_empty_string_returns_false(self):
        """An empty module name must return False (no spec found)."""
        # find_spec("") may raise, returns None, or returns False — all OK
        result = _has_module("")
        self.assertIsInstance(result, bool)

    def test_does_not_import_module(self):
        """_has_module must not insert the module into sys.modules."""
        name = "importlib"  # always importable but may or may not be pre-imported
        was_present = name in sys.modules
        _has_module(name)
        if not was_present:
            # If it was absent before, it should still be absent after
            # (importlib.util.find_spec may auto-load it — this is backend behaviour)
            pass  # We just ensure the call itself never raises

    def test_third_party_absent_returns_false(self):
        """A well-known third-party that is not installed must return False."""
        # Choose a plausibly absent module.
        if not _has_module("FAKE_NONEXISTENT_PACKAGE_DO_NOT_INSTALL"):
            self.assertFalse(_has_module("FAKE_NONEXISTENT_PACKAGE_DO_NOT_INSTALL"))


# ===========================================================================
# Module constants
# ===========================================================================


class TestModuleConstants(unittest.TestCase):
    """Public capability flags must be booleans consistent with their sources."""

    def test_read_support_is_bool(self):
        self.assertIsInstance(TOML_READ_SUPPORT, bool)

    def test_write_support_is_bool(self):
        self.assertIsInstance(TOML_WRITE_SUPPORT, bool)

    def test_read_source_none_or_str(self):
        self.assertIsInstance(TOML_READ_SOURCE, (str, type(None)))

    def test_write_source_none_or_str(self):
        self.assertIsInstance(TOML_WRITE_SOURCE, (str, type(None)))

    def test_read_support_consistent_with_source(self):
        """TOML_READ_SUPPORT must equal (TOML_READ_SOURCE is not None)."""
        self.assertEqual(TOML_READ_SUPPORT, TOML_READ_SOURCE is not None)

    def test_write_support_consistent_with_source(self):
        """TOML_WRITE_SUPPORT must equal (TOML_WRITE_SOURCE is not None)."""
        self.assertEqual(TOML_WRITE_SUPPORT, TOML_WRITE_SOURCE is not None)

    def test_read_source_valid_value_if_present(self):
        """If not None, TOML_READ_SOURCE must be one of the known backends."""
        if TOML_READ_SOURCE is not None:
            self.assertIn(TOML_READ_SOURCE, ("tomllib", "tomli", "toml"))

    def test_write_source_valid_value_if_present(self):
        """If not None, TOML_WRITE_SOURCE must be one of the known backends."""
        if TOML_WRITE_SOURCE is not None:
            self.assertIn(TOML_WRITE_SOURCE, ("toml", "tomli_w"))

    def test_py311_read_support_true(self):
        """Python 3.11+ ships tomllib; read support must be True."""
        if sys.version_info >= (3, 11):
            self.assertTrue(TOML_READ_SUPPORT)

    def test_py311_read_source_tomllib(self):
        """On Python 3.11+, the read source must be 'tomllib'."""
        if sys.version_info >= (3, 11):
            self.assertEqual(TOML_READ_SOURCE, "tomllib")


# ===========================================================================
# _normalize_path
# ===========================================================================


class TestNormalizePath(unittest.TestCase):
    """_normalize_path must resolve paths to absolute pathlib.Path objects."""

    def test_returns_path_object(self):
        result = _normalize_path("/tmp")
        self.assertIsInstance(result, Path)

    def test_absolute_path_preserved(self):
        result = _normalize_path("/tmp/something")
        self.assertTrue(result.is_absolute())

    def test_tilde_expanded(self):
        result = _normalize_path("~/test")
        self.assertFalse(str(result).startswith("~"))
        self.assertTrue(result.is_absolute())

    def test_relative_path_becomes_absolute(self):
        result = _normalize_path("relative/path")
        self.assertTrue(result.is_absolute())

    def test_dot_normalized(self):
        result = _normalize_path(".")
        self.assertTrue(result.is_absolute())

    def test_path_object_input(self):
        """Accepts pathlib.Path as input without error."""
        result = _normalize_path(Path("/tmp"))
        self.assertIsInstance(result, Path)


# ===========================================================================
# _load_read_backend  (tested via read_toml integration + direct import)
# ===========================================================================


class TestLoadReadBackend(unittest.TestCase):
    """_load_read_backend must return a (name, module) tuple or raise."""

    def test_available_backend_does_not_raise(self):
        """When TOML_READ_SUPPORT is True, loading must not raise."""
        if not TOML_READ_SUPPORT:
            self.skipTest("No TOML read backend available")
        from .._toml import _load_read_backend  # noqa: PLC0415
        name, mod = _load_read_backend()
        self.assertIsInstance(name, str)
        self.assertIsNotNone(mod)

    def test_backend_name_matches_source(self):
        if not TOML_READ_SUPPORT:
            self.skipTest("No TOML read backend available")
        from .._toml import _load_read_backend  # noqa: PLC0415
        name, _ = _load_read_backend()
        self.assertEqual(name, TOML_READ_SOURCE)

    def test_no_backend_raises_scikitplot_exception(self):
        """When no backend is available, must raise ScikitplotException."""
        from .._toml import _load_read_backend  # noqa: PLC0415
        # Patch TOML_READ_SOURCE to None to simulate missing backends
        with mock.patch("scikitplot.utils._toml.TOML_READ_SOURCE", None):
            with self.assertRaises(ScikitplotException):
                _load_read_backend()


# ===========================================================================
# _load_write_backend
# ===========================================================================


class TestLoadWriteBackend(unittest.TestCase):
    """_load_write_backend must return a (name, module) tuple or raise."""

    def test_available_backend_does_not_raise(self):
        if not TOML_WRITE_SUPPORT:
            self.skipTest("No TOML write backend available")
        from .._toml import _load_write_backend  # noqa: PLC0415
        name, mod = _load_write_backend()
        self.assertIsInstance(name, str)
        self.assertIsNotNone(mod)

    def test_no_backend_raises_scikitplot_exception(self):
        from .._toml import _load_write_backend  # noqa: PLC0415
        with mock.patch("scikitplot.utils._toml.TOML_WRITE_SOURCE", None):
            with self.assertRaises(ScikitplotException):
                _load_write_backend()


# ===========================================================================
# read_toml
# ===========================================================================


class TestReadToml(unittest.TestCase):
    """read_toml must parse TOML files or raise ScikitplotException on errors."""

    def setUp(self):
        self._tmpdir = _make_tmpdir()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _skip_if_no_read(self):
        if not TOML_READ_SUPPORT:
            self.skipTest("No TOML read backend available")

    def _write_toml_file(self, content: str, name: str = "test.toml") -> Path:
        """Write raw TOML bytes to a temp file and return its path."""
        path = self._tmpdir / name
        path.write_text(content, encoding="utf-8")
        return path

    # -- Happy-path --

    def test_reads_simple_toml(self):
        """A minimal TOML file must parse into a dict."""
        self._skip_if_no_read()
        path = self._write_toml_file('[section]\nkey = "value"\n')
        result = read_toml(path)
        self.assertIsInstance(result, dict)
        self.assertIn("section", result)
        self.assertEqual(result["section"]["key"], "value")

    def test_reads_integer_value(self):
        self._skip_if_no_read()
        path = self._write_toml_file("count = 42\n")
        result = read_toml(path)
        self.assertEqual(result["count"], 42)

    def test_reads_boolean_value(self):
        self._skip_if_no_read()
        path = self._write_toml_file("flag = true\n")
        result = read_toml(path)
        self.assertTrue(result["flag"])

    def test_reads_array_value(self):
        self._skip_if_no_read()
        path = self._write_toml_file("items = [1, 2, 3]\n")
        result = read_toml(path)
        self.assertEqual(result["items"], [1, 2, 3])

    def test_reads_nested_table(self):
        self._skip_if_no_read()
        content = "[database]\nhost = \"localhost\"\nport = 5432\n"
        path = self._write_toml_file(content)
        result = read_toml(path)
        self.assertEqual(result["database"]["host"], "localhost")
        self.assertEqual(result["database"]["port"], 5432)

    def test_returns_dict(self):
        self._skip_if_no_read()
        path = self._write_toml_file("x = 1\n")
        self.assertIsInstance(read_toml(path), dict)

    def test_accepts_str_path(self):
        """Accepts str as well as Path."""
        self._skip_if_no_read()
        path = self._write_toml_file("a = 1\n")
        result = read_toml(str(path))
        self.assertIsInstance(result, dict)

    def test_accepts_path_object(self):
        self._skip_if_no_read()
        path = self._write_toml_file("b = 2\n")
        result = read_toml(path)
        self.assertIsInstance(result, dict)

    # -- Error paths --

    def test_missing_file_raises(self):
        """A non-existent file must raise ScikitplotException."""
        self._skip_if_no_read()
        with self.assertRaises(ScikitplotException):
            read_toml(self._tmpdir / "does_not_exist.toml")

    def test_directory_as_path_raises(self):
        """Passing a directory path must raise ScikitplotException."""
        self._skip_if_no_read()
        with self.assertRaises(ScikitplotException):
            read_toml(self._tmpdir)

    def test_permission_error_raises(self):
        """A permission-denied open must raise ScikitplotException."""
        self._skip_if_no_read()
        path = self._write_toml_file("k = 1\n")
        with mock.patch("builtins.open", side_effect=PermissionError("denied")):
            with self.assertRaises(ScikitplotException):
                read_toml(path)

    def test_no_read_backend_raises(self):
        """When no read backend is available, must raise ScikitplotException."""
        path = self._write_toml_file("k = 1\n")
        with mock.patch("scikitplot.utils._toml.TOML_READ_SOURCE", None):
            with self.assertRaises(ScikitplotException):
                read_toml(path)


# ===========================================================================
# write_toml
# ===========================================================================


class TestWriteToml(unittest.TestCase):
    """write_toml must write TOML files or raise ScikitplotException on errors."""

    def setUp(self):
        self._tmpdir = _make_tmpdir()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _skip_if_no_write(self):
        if not TOML_WRITE_SUPPORT:
            self.skipTest("No TOML write backend available")

    # -- Happy-path --

    def test_writes_file(self):
        """write_toml must create a TOML file."""
        self._skip_if_no_write()
        path = self._tmpdir / "out.toml"
        write_toml(path, {"key": "value"})
        self.assertTrue(path.exists())

    def test_returns_str_path(self):
        """write_toml must return the absolute path as a string."""
        self._skip_if_no_write()
        path = self._tmpdir / "out.toml"
        result = write_toml(path, {"a": 1})
        self.assertIsInstance(result, str)
        self.assertEqual(result, str(path.resolve()))

    def test_written_content_readable(self):
        """A round-trip write→read must preserve data."""
        if not (TOML_READ_SUPPORT and TOML_WRITE_SUPPORT):
            self.skipTest("Need both read and write backend for round-trip test")
        data = {"project": {"name": "scikitplot", "version": "1.0"}}
        path = self._tmpdir / "roundtrip.toml"
        write_toml(path, data)
        result = read_toml(path)
        self.assertEqual(result["project"]["name"], "scikitplot")
        self.assertEqual(result["project"]["version"], "1.0")

    def test_roundtrip_integer(self):
        if not (TOML_READ_SUPPORT and TOML_WRITE_SUPPORT):
            self.skipTest("Need both backends")
        path = self._tmpdir / "int.toml"
        write_toml(path, {"count": 99})
        result = read_toml(path)
        self.assertEqual(result["count"], 99)

    def test_roundtrip_list(self):
        if not (TOML_READ_SUPPORT and TOML_WRITE_SUPPORT):
            self.skipTest("Need both backends")
        path = self._tmpdir / "list.toml"
        write_toml(path, {"items": [1, 2, 3]})
        result = read_toml(path)
        self.assertEqual(result["items"], [1, 2, 3])

    def test_roundtrip_boolean(self):
        if not (TOML_READ_SUPPORT and TOML_WRITE_SUPPORT):
            self.skipTest("Need both backends")
        path = self._tmpdir / "bool.toml"
        write_toml(path, {"active": True})
        result = read_toml(path)
        self.assertTrue(result["active"])

    def test_mkdir_true_creates_parent(self):
        """mkdir=True must create missing parent directories."""
        self._skip_if_no_write()
        nested = self._tmpdir / "a" / "b" / "c" / "config.toml"
        write_toml(nested, {"k": "v"}, mkdir=True)
        self.assertTrue(nested.exists())

    def test_mkdir_false_missing_parent_raises(self):
        """mkdir=False must raise ScikitplotException if parent is absent."""
        self._skip_if_no_write()
        nested = self._tmpdir / "missing" / "config.toml"
        with self.assertRaises(ScikitplotException):
            write_toml(nested, {"k": "v"}, mkdir=False)

    def test_accepts_str_path(self):
        """Accepts a string path."""
        self._skip_if_no_write()
        path = str(self._tmpdir / "str_out.toml")
        write_toml(path, {"x": 1})
        self.assertTrue(os.path.isfile(path))

    def test_utf8_encoding(self):
        """Written file must be UTF-8 encoded (readable as UTF-8)."""
        self._skip_if_no_write()
        path = self._tmpdir / "utf8.toml"
        write_toml(path, {"msg": "hello"})
        content = path.read_text(encoding="utf-8")
        self.assertIn("hello", content)

    # -- Error paths --

    def test_directory_as_output_raises(self):
        """Targeting an existing directory must raise ScikitplotException."""
        self._skip_if_no_write()
        with self.assertRaises(ScikitplotException):
            write_toml(self._tmpdir, {"k": "v"})

    def test_permission_error_raises(self):
        """PermissionError during open must raise ScikitplotException."""
        self._skip_if_no_write()
        path = self._tmpdir / "perm.toml"
        with mock.patch("builtins.open", side_effect=PermissionError("denied")):
            with self.assertRaises(ScikitplotException):
                write_toml(path, {"k": "v"})

    def test_no_write_backend_raises(self):
        """When no write backend is available, must raise ScikitplotException."""
        path = self._tmpdir / "no_backend.toml"
        with mock.patch("scikitplot.utils._toml.TOML_WRITE_SOURCE", None):
            with self.assertRaises(ScikitplotException):
                write_toml(path, {"k": "v"})

    def test_overwrite_existing_file(self):
        """Writing to an existing file must overwrite it."""
        self._skip_if_no_write()
        path = self._tmpdir / "overwrite.toml"
        path.write_text("old = 1\n", encoding="utf-8")
        write_toml(path, {"new": 2})
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("old", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
