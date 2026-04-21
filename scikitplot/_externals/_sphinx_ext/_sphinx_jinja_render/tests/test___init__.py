# scikitplot/_externals/_sphinx_ext/_url_helper/tests/test___init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for :mod:`._url_helper` (the package ``__init__``).

Verifies that:

* The package is importable.
* Every symbol declared in ``__all__`` is accessible as a package-level
  attribute.
* Re-exported callables are the correct objects from their origin modules.
* The ``__all__`` list contains no duplicates.
* No private (``_``-prefixed) internal names leak into the public API.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import pytest

from ... import _sphinx_jinja_render as pkg
from .._bootstrap import load_bootstrap_code
from .._constants import (
    JUPYTERLITE_BASE_URL,
    EXTENSION_VERSION,
)
from .._extension import setup
from .._rst_renderer import render_rst_templates
from .._url_builder import build_repl_url

# ---------------------------------------------------------------------------
# Basic importability
# ---------------------------------------------------------------------------


class TestPackageImport:
    def test_package_is_a_module(self):
        assert isinstance(pkg, ModuleType)

    def test_all_is_defined(self):
        assert hasattr(pkg, "__all__")
        assert isinstance(pkg.__all__, list)

    def test_all_is_non_empty(self):
        assert len(pkg.__all__) > 0

    def test_all_contains_no_duplicates(self):
        assert len(pkg.__all__) == len(set(pkg.__all__))

    def test_all_entries_are_strings(self):
        for name in pkg.__all__:
            assert isinstance(name, str), f"__all__ entry {name!r} is not a str"


# ---------------------------------------------------------------------------
# __all__ contract: every listed symbol is accessible
# ---------------------------------------------------------------------------


class TestAllSymbolsAccessible:
    @pytest.mark.parametrize("name", pkg.__all__)
    def test_symbol_accessible_as_attribute(self, name: str):
        assert hasattr(pkg, name), (
            f"'{name}' is listed in __all__ but not accessible as pkg.{name}"
        )

    @pytest.mark.parametrize("name", pkg.__all__)
    def test_symbol_is_not_none(self, name: str):
        # None is a legitimate value only for sentinel constants; none of ours are None.
        assert getattr(pkg, name) is not None, (
            f"pkg.{name} is None — was the re-export assignment missed?"
        )


# ---------------------------------------------------------------------------
# Specific symbol type checks
# ---------------------------------------------------------------------------


class TestSymbolTypes:
    # Callables
    def test_setup_is_callable(self):
        assert callable(pkg.setup)

    def test_build_repl_url_is_callable(self):
        assert callable(pkg.build_repl_url)

    def test_load_bootstrap_code_is_callable(self):
        assert callable(pkg.load_bootstrap_code)

    def test_render_rst_templates_is_callable(self):
        assert callable(pkg.render_rst_templates)

    # String constants
    def test_default_base_repl_url_is_str(self):
        assert isinstance(pkg.JUPYTERLITE_BASE_URL, str)

    def test_default_repl_kernel_is_str(self):
        assert isinstance(pkg.DEFAULT_KERNEL_NAME, str)

    def test_extension_version_is_str(self):
        assert isinstance(pkg.EXTENSION_VERSION, str)

    def test_wasm_bootstrap_code_is_str(self):
        assert isinstance(pkg.WASM_BOOTSTRAP_CODE, str)

    def test_wasm_fallback_code_is_str(self):
        assert isinstance(pkg.WASM_FALLBACK_CODE, str)


# ---------------------------------------------------------------------------
# Re-export identity: pkg.X is the same object as origin_module.X
# ---------------------------------------------------------------------------


class TestReExportIdentity:
    def test_build_repl_url_origin(self):
        assert pkg.build_repl_url is build_repl_url

    def test_load_bootstrap_code_origin(self):
        assert pkg.load_bootstrap_code is load_bootstrap_code

    def test_render_rst_templates_origin(self):
        assert pkg.render_rst_templates is render_rst_templates

    def test_setup_origin(self):
        assert pkg.setup is setup

    def test_constants_origin(self):
        assert pkg.JUPYTERLITE_BASE_URL is JUPYTERLITE_BASE_URL
        assert pkg.EXTENSION_VERSION is EXTENSION_VERSION


# ---------------------------------------------------------------------------
# No private-name leakage into the public API
# ---------------------------------------------------------------------------


class TestNoPrivateLeakage:
    def test_all_contains_no_private_names(self):
        private_in_all = [n for n in pkg.__all__ if n.startswith("_")]
        assert private_in_all == [], (
            f"Private names found in __all__: {private_in_all}"
        )
