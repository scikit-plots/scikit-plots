# scikitplot/mlflow/tests/test__utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._utils.

Naming convention: test__<module_name>.py

Covers
------
- is_mlflow_installed       : with/without mlflow importable
- _parse_version            : standard, pre-release, garbage, empty, complex PEP 440
- mlflow_version            : module attribute path, metadata fallback (lines 111-120),
                              not-installed returns None, module import error path
- MlflowVersion.triple      : property correctness
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from scikitplot.mlflow._utils import (
    MlflowVersion,
    _parse_version,
    is_mlflow_installed,
    mlflow_version,
)


# ===========================================================================
# MlflowVersion
# ===========================================================================


class TestMlflowVersion:
    """Tests for the MlflowVersion dataclass."""

    def test_defaults_are_zeros(self) -> None:
        v = MlflowVersion()
        assert v.raw == ""
        assert v.major == v.minor == v.patch == 0

    def test_triple_property(self) -> None:
        v = MlflowVersion(raw="2.14.3", major=2, minor=14, patch=3)
        assert v.triple == (2, 14, 3)

    def test_triple_comparison(self) -> None:
        v1 = MlflowVersion(raw="2.14.0", major=2, minor=14, patch=0)
        v2 = MlflowVersion(raw="2.14.1", major=2, minor=14, patch=1)
        assert v1.triple < v2.triple

    def test_immutability(self) -> None:
        v = MlflowVersion(raw="1.0.0", major=1, minor=0, patch=0)
        with pytest.raises((AttributeError, TypeError)):
            v.major = 99  # type: ignore[misc]


# ===========================================================================
# _parse_version
# ===========================================================================


class TestParseVersion:
    """Tests for the _parse_version internal function."""

    def test_standard_version(self) -> None:
        v = _parse_version("2.14.3")
        assert v.major == 2
        assert v.minor == 14
        assert v.patch == 3
        assert v.raw == "2.14.3"

    def test_pre_release_version(self) -> None:
        v = _parse_version("2.14.0rc1")
        assert v.major == 2
        assert v.minor == 14
        assert v.patch == 0

    def test_garbage_returns_zeros(self) -> None:
        v = _parse_version("not-a-version")
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 0
        assert v.raw == "not-a-version"

    def test_empty_string_returns_zeros(self) -> None:
        v = _parse_version("")
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 0

    def test_complex_pep440(self) -> None:
        v = _parse_version("1.2.3.post4+g1234abc")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3


# ===========================================================================
# is_mlflow_installed
# ===========================================================================


class TestIsMlflowInstalled:
    """Tests for is_mlflow_installed."""

    def test_returns_bool(self) -> None:
        result = is_mlflow_installed()
        assert isinstance(result, bool)

    def test_true_when_mlflow_in_sys_modules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If mlflow is in sys.modules (as a mock), find_spec sees it."""
        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.__version__ = "2.0.0"
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        assert is_mlflow_installed() is True

    def test_false_when_mlflow_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When find_spec returns None, mlflow is not installed."""
        monkeypatch.setitem(sys.modules, "mlflow", None)  # type: ignore[assignment]
        # Patch find_spec to return None
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
        assert is_mlflow_installed() is False


# ===========================================================================
# mlflow_version
# ===========================================================================


class TestMlflowVersionFunction:
    """Tests for the ``mlflow_version`` module-level function."""

    def test_returns_none_when_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
        result = mlflow_version()
        assert result is None

    def test_returns_version_from_module_attribute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Primary path: reads __version__ from the imported module."""
        fake_mlflow = types.ModuleType("mlflow")
        fake_mlflow.__version__ = "2.14.3"
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        # Ensure find_spec returns non-None
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())
        result = mlflow_version()
        assert result is not None
        assert result.major == 2
        assert result.minor == 14
        assert result.patch == 3

    def test_falls_back_to_metadata_when_no_version_attr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lines 111-120: fallback to importlib.metadata when __version__ is absent."""
        fake_mlflow = types.ModuleType("mlflow")
        # No __version__ attribute
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

        # Patch import_module to return fake_mlflow (no __version__)
        import scikitplot.mlflow._utils as m

        monkeypatch.setattr(m.importlib, "import_module", lambda _: fake_mlflow)
        monkeypatch.setattr(
            m.importlib.metadata,
            "version",
            lambda _: "3.1.2",
        )
        result = mlflow_version()
        assert result is not None
        assert result.major == 3
        assert result.minor == 1
        assert result.patch == 2

    def test_returns_none_when_metadata_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lines 111-120: metadata fallback raises PackageNotFoundError → return None."""
        fake_mlflow = types.ModuleType("mlflow")
        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

        import scikitplot.mlflow._utils as m

        monkeypatch.setattr(m.importlib, "import_module", lambda _: fake_mlflow)
        monkeypatch.setattr(
            m.importlib.metadata,
            "version",
            MagicMock(side_effect=importlib.metadata.PackageNotFoundError("mlflow")),
        )
        result = mlflow_version()
        assert result is None

    def test_module_import_error_falls_back_to_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If import_module raises, fall through to metadata path."""
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec", lambda _: object())

        import scikitplot.mlflow._utils as m

        monkeypatch.setattr(
            m.importlib, "import_module", MagicMock(side_effect=ImportError("boom"))
        )
        monkeypatch.setattr(m.importlib.metadata, "version", lambda _: "1.0.0")
        result = mlflow_version()
        assert result is not None
        assert result.major == 1
