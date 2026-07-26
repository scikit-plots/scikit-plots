# scikitplot/cython/tests/test__loader_transaction.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-LOAD-001.

``import_extension`` used to ``sys.modules.pop(name)`` *before* loading the new
artifact, so a failed (re)load destroyed a previously-working module and never
restored it.  The fix makes loading a transaction: the prior entry is preserved
and restored on any failure, and the new module is registered before
``exec_module`` per the import protocol.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from .._loader import import_extension


class TestLoadTransaction:
    def test_failed_reload_preserves_prior_module(self) -> None:
        name = "load001_prior"
        prior = types.ModuleType(name)
        prior.MARKER = "prior"
        sys.modules[name] = prior
        try:
            with pytest.raises(ImportError):
                import_extension(name=name, path=Path("/nonexistent/x.so"))
            assert sys.modules.get(name) is prior
            assert sys.modules[name].MARKER == "prior"
        finally:
            sys.modules.pop(name, None)

    def test_failed_load_without_prior_leaves_no_entry(self) -> None:
        name = "load001_absent"
        sys.modules.pop(name, None)
        with pytest.raises(ImportError):
            import_extension(name=name, path=Path("/nonexistent/y.so"))
        assert name not in sys.modules

    def test_spec_failure_does_not_touch_prior(self, tmp_path: Path) -> None:
        # A file that exists but is not a loadable extension → spec/exec failure.
        name = "load001_specfail"
        prior = types.ModuleType(name)
        prior.MARKER = "keep"
        sys.modules[name] = prior
        bogus = tmp_path / "not_a_real.so"
        bogus.write_bytes(b"not an elf")
        try:
            with pytest.raises(Exception):  # noqa: B017
                import_extension(name=name, path=bogus)
            assert sys.modules.get(name) is prior
        finally:
            sys.modules.pop(name, None)
