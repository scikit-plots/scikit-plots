# scikitplot/nc/tests/test___init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot.nc` (the ``nc`` package ``__init__``).

Coverage targets
----------------
* Module-level metadata: ``__author__``, ``__author_email__``, ``__git_hash__``
* :func:`~scikitplot.nc.get_include`:

  - Branch 1 (source tree): ``scikitplot.show_config is None``  -> path
    derived from ``scikitplot.__file__``.
  - Branch 2 (installed):   ``scikitplot.show_config`` is callable -> path
    derived from ``nc.__file__`` via ``../cexternals/_numcpp/include``.
  - Error path: directory absent -> :exc:`FileNotFoundError`.

* Public API surface: symbol presence, ``__all__``, sub-package import.

Notes
-----
* Tests requiring a compiled C++ extension guard themselves with
  ``pytest.skip`` inside the test body so the pure-Python layer stays
  runnable in build-free environments (e.g. CI without Meson).
* The ``get_include()`` tests patch **both** ``os.path.isdir`` *and*
  ``scikitplot.show_config`` to exercise the two conditional branches
  independently, without touching the file system.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Always-available import (pure-Python layer)
# ---------------------------------------------------------------------------
import scikitplot.nc as nc  # noqa: PLC0415


# ===========================================================================
# Module-level metadata
# ===========================================================================

class TestMetadata:
    """Tests for constant metadata attributes of the nc package."""

    def test_author_is_string(self):
        """__author__ is a non-empty str."""
        assert isinstance(nc.__author__, str)
        assert nc.__author__

    def test_author_is_david_pilger(self):
        """__author__ is 'David Pilger' (upstream NumCpp author)."""
        assert nc.__author__ == "David Pilger"

    def test_author_email_is_string(self):
        """__author_email__ is a non-empty str."""
        assert isinstance(nc.__author_email__, str)
        assert nc.__author_email__

    def test_author_email_contains_at_sign(self):
        """__author_email__ contains exactly one '@' (basic e-mail sanity check)."""
        assert nc.__author_email__.count("@") == 1

    def test_git_hash_is_string(self):
        """__git_hash__ is a non-empty str."""
        assert isinstance(nc.__git_hash__, str)
        assert nc.__git_hash__

    def test_git_hash_is_lowercase_hex(self):
        """__git_hash__ contains only hex characters (git SHA-1 / SHA-256)."""
        assert all(c in "0123456789abcdef" for c in nc.__git_hash__.lower()), (
            f"Non-hex character found in __git_hash__: {nc.__git_hash__!r}"
        )

    def test_git_hash_length_at_least_7(self):
        """__git_hash__ is at least 7 characters (shortest valid git short SHA)."""
        assert len(nc.__git_hash__) >= 7

    def test_git_hash_length_at_most_64(self):
        """__git_hash__ is at most 64 characters (SHA-256 full length)."""
        assert len(nc.__git_hash__) <= 64


# ===========================================================================
# get_include()
# ===========================================================================

class TestGetInclude:
    """Tests for :func:`scikitplot.nc.get_include`."""

    # -------------------------------------------------------------------
    # Basic contract (directory present)
    # -------------------------------------------------------------------

    def test_is_callable(self):
        """get_include is a callable attribute of the nc package."""
        assert callable(nc.get_include)

    def test_returns_string_when_dir_present(self):
        """get_include() returns a str when the include directory exists."""
        with patch("os.path.isdir", return_value=True):
            result = nc.get_include()
        assert isinstance(result, str)

    def test_returns_non_empty_string(self):
        """get_include() returns a non-empty path string."""
        with patch("os.path.isdir", return_value=True):
            result = nc.get_include()
        assert result

    def test_path_ends_with_include_component(self):
        """The returned path's final component is 'include'."""
        with patch("os.path.isdir", return_value=True):
            result = nc.get_include()
        assert Path(result).name == "include"

    def test_returns_absolute_path(self):
        """get_include() always returns an absolute (not relative) path."""
        with patch("os.path.isdir", return_value=True):
            result = nc.get_include()
        assert os.path.isabs(result), f"Expected absolute path, got: {result!r}"

    # -------------------------------------------------------------------
    # Error path (directory absent)
    # -------------------------------------------------------------------

    def test_raises_file_not_found_when_dir_missing(self):
        """get_include() raises FileNotFoundError when the include dir is absent."""
        with patch("os.path.isdir", return_value=False):
            with pytest.raises(FileNotFoundError):
                nc.get_include()

    def test_file_not_found_message_mentions_include(self):
        """FileNotFoundError message references the 'include' directory."""
        with patch("os.path.isdir", return_value=False):
            with pytest.raises(FileNotFoundError, match="include"):
                nc.get_include()

    def test_file_not_found_message_mentions_numcpp(self):
        """FileNotFoundError message references NumCpp / _numcpp."""
        with patch("os.path.isdir", return_value=False):
            with pytest.raises(FileNotFoundError, match="_numcpp"):
                nc.get_include()

    # -------------------------------------------------------------------
    # Branch 1: source tree (show_config is None — default in test env)
    # -------------------------------------------------------------------

    def test_branch1_returns_str_without_show_config(self):
        """Branch 1: when scikitplot has no show_config, get_include() returns str."""
        scikitplot_mod = sys.modules["scikitplot"]
        # Ensure show_config attribute does not exist (branch 1 condition).
        show_config_orig = scikitplot_mod.__dict__.pop("show_config", _SENTINEL)
        try:
            with patch("os.path.isdir", return_value=True):
                result = nc.get_include()
            assert isinstance(result, str)
            assert result
        finally:
            if show_config_orig is not _SENTINEL:
                scikitplot_mod.show_config = show_config_orig

    # -------------------------------------------------------------------
    # Branch 2: installed package (show_config is not None)
    # -------------------------------------------------------------------

    def test_branch2_installed_path_is_str(self):
        """Branch 2: when show_config is callable, get_include() still returns str."""
        scikitplot_mod = sys.modules["scikitplot"]
        with patch.object(scikitplot_mod, "show_config", lambda: {}, create=True):
            with patch("os.path.isdir", return_value=True):
                result = nc.get_include()
        assert isinstance(result, str)
        assert result

    def test_branch2_installed_path_ends_with_include(self):
        """Branch 2: returned path's final component is 'include'."""
        scikitplot_mod = sys.modules["scikitplot"]
        with patch.object(scikitplot_mod, "show_config", lambda: {}, create=True):
            with patch("os.path.isdir", return_value=True):
                result = nc.get_include()
        assert Path(result).name == "include"

    def test_branch2_installed_path_is_absolute(self):
        """Branch 2: returned path is absolute."""
        scikitplot_mod = sys.modules["scikitplot"]
        with patch.object(scikitplot_mod, "show_config", lambda: {}, create=True):
            with patch("os.path.isdir", return_value=True):
                result = nc.get_include()
        assert os.path.isabs(result)

    def test_branch2_raises_file_not_found_when_dir_missing(self):
        """Branch 2: FileNotFoundError when include directory is absent."""
        scikitplot_mod = sys.modules["scikitplot"]
        with patch.object(scikitplot_mod, "show_config", lambda: {}, create=True):
            with patch("os.path.isdir", return_value=False):
                with pytest.raises(FileNotFoundError):
                    nc.get_include()


# ===========================================================================
# Public API surface
# ===========================================================================

class TestPublicAPI:
    """Tests for the public symbols exposed by nc.__init__."""

    def test_get_include_is_present(self):
        """get_include is a top-level attribute of the nc package."""
        assert hasattr(nc, "get_include")

    def test_dot_attribute_callable_if_ext_compiled(self):
        """nc.dot is callable when the C++ extension has been compiled."""
        if not hasattr(nc, "dot"):
            pytest.skip("C++ extension not compiled; nc.dot not available")
        assert callable(nc.dot)

    def test_version_attribute_str_if_ext_compiled(self):
        """nc.__version__ is a non-empty str when the C++ extension is compiled."""
        if not hasattr(nc, "__version__"):
            pytest.skip("C++ extension not compiled; nc.__version__ not available")
        assert isinstance(nc.__version__, str)
        assert nc.__version__

    def test_linalg_subpackage_importable(self):
        """nc._linalg can always be imported (pure Python __init__)."""
        try:
            linalg = importlib.import_module("scikitplot.nc._linalg")
        except ModuleNotFoundError as exc:
            pytest.skip(f"Could not import scikitplot.nc._linalg: {exc}")
        assert linalg is not None

    def test_linalg_all_contains_dot(self):
        """nc._linalg.__all__ lists 'dot'."""
        try:
            linalg = importlib.import_module("scikitplot.nc._linalg")
        except ModuleNotFoundError as exc:
            pytest.skip(f"Could not import scikitplot.nc._linalg: {exc}")
        assert "dot" in linalg.__all__

    def test_linalg_dot_is_callable(self):
        """nc._linalg.dot is callable (pure Python wrapper always present)."""
        try:
            linalg = importlib.import_module("scikitplot.nc._linalg")
        except ModuleNotFoundError as exc:
            pytest.skip(f"Could not import scikitplot.nc._linalg: {exc}")
        assert callable(linalg.dot)

    def test_version_subpackage_importable(self):
        """nc._version can be imported (pure Python __init__ layer)."""
        try:
            ver_pkg = importlib.import_module("scikitplot.nc._version")
        except ModuleNotFoundError:
            pytest.skip("scikitplot.nc._version not importable in this environment")
        assert ver_pkg is not None

    def test_nc_has_all_metadata_attributes(self):
        """nc.__init__ exports the expected metadata attributes."""
        for attr in ("__author__", "__author_email__", "__git_hash__"):
            assert hasattr(nc, attr), f"Missing expected metadata attribute: {attr!r}"


# ---------------------------------------------------------------------------
# Module-level sentinel for optional attribute removal
# ---------------------------------------------------------------------------
_SENTINEL = object()
