# scikitplot/cython/tests/test__init.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`scikitplot.cython.__init__`.

Covers
------
- ``__all__``  : all exports importable, no duplicates, key public names present
- Module-level re-exports work without side-effects at import time
"""
from __future__ import annotations

import importlib
import pytest

from .. import __all__ as _PKG_ALL
from .._security import SecurityError, SecurityPolicy
from .._custom_compiler import register_compiler
from ... import cython as pkg  # scikitplot.cython


class TestInitModule:
    """Smoke tests for the :mod:`scikitplot.cython` package ``__init__.py``."""

    def test_all_exports_importable(self) -> None:
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"__all__ member {name!r} not importable"

    def test_no_spurious_exports(self) -> None:
        assert "_builder" not in pkg.__all__  # private builder


class TestRegressionR3NoDuplicatesInAll:
    """R3: ``scikitplot.cython.__all__`` must contain no duplicate names."""

    def test_no_duplicate_names(self) -> None:
        duplicates = [x for x in _PKG_ALL if _PKG_ALL.count(x) > 1]
        assert duplicates == [], f"Duplicate __all__ entries: {sorted(set(duplicates))}"

    def test_all_is_list_of_strings(self) -> None:
        assert isinstance(_PKG_ALL, list)
        for name in _PKG_ALL:
            assert isinstance(name, str), f"Non-string in __all__: {name!r}"

    def test_key_public_names_present(self) -> None:
        for name in (
            "compile_and_load",
            "compile_and_load_result",
            "check_build_prereqs",
            "SecurityPolicy",
            "SecurityError",
            "validate_build_inputs",
            "CustomCompilerProtocol",
            "CompilerRegistry",
            "register_compiler",
            "collect_c_api_sources",
            "collect_header_dirs",
            "PybindCompiler",
            "CApiCompiler",
        ):
            assert name in _PKG_ALL, f"Expected name missing from __all__: {name!r}"
