# scikitplot/nc/_version/tests/test___version.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot.nc._version._version` (Cython module).

This module exposes ``nc::VERSION`` via a Cython binding (``_version.pyx``).
The compiled ``.so`` is required; tests are automatically skipped per-class
if it has not been built, so the rest of the suite remains green.

What is tested
--------------
* ``_version.__version__`` is a non-empty string.
* The version string matches a recognizable semver-like pattern
  (e.g. ``"2.3.0"``).
* The public ``__all__`` contract of the parent ``_version`` package
  is satisfied without triggering :exc:`NameError` when the package
  cannot be imported.

Notes
-----
* The Cython module exposes ``__version__`` as the decoded ``const char*``
  from ``nc::VERSION``.  The test does *not* hard-code the exact version
  string so it keeps passing as NumCpp is updated.
* All tests that require the compiled extension are decorated with
  ``pytestmark`` (class-level skip) so they are clearly isolated from
  the package-level contract tests that are always runnable.
"""

from __future__ import annotations

import importlib
import re

import pytest

# ---------------------------------------------------------------------------
# Extension availability — do NOT use importorskip at module level.
# Module-level importorskip would skip the package-contract tests too.
# ---------------------------------------------------------------------------
try:
    import scikitplot.nc._version._version as _version_mod  # noqa: PLC0415
    _CYTHON_AVAILABLE = True
except ImportError:
    _version_mod = None  # type: ignore[assignment]
    _CYTHON_AVAILABLE = False

_CYTHON_REASON = "Cython extension _version._version not compiled"

# ---------------------------------------------------------------------------
# Package-level availability — separate from the Cython extension.
# ---------------------------------------------------------------------------
try:
    from scikitplot.nc._version import __all__ as _ver_pkg_all  # noqa: PLC0415
    import scikitplot.nc._version as _ver_pkg                   # noqa: PLC0415
    _PKG_AVAILABLE = True
except ImportError:
    _ver_pkg_all = []   # safe default — no NameError risk
    _ver_pkg = None     # type: ignore[assignment]
    _PKG_AVAILABLE = False


# ===========================================================================
# Cython _version module tests (skipped when extension not compiled)
# ===========================================================================

@pytest.mark.skipif(not _CYTHON_AVAILABLE, reason=_CYTHON_REASON)
class TestCythonVersionModule:
    """Tests for the Cython ``_version`` extension module."""

    def test_has_version_attribute(self):
        """Module exposes a ``__version__`` attribute."""
        assert hasattr(_version_mod, "__version__")

    def test_version_is_string(self):
        """``__version__`` is a str."""
        assert isinstance(_version_mod.__version__, str)

    def test_version_is_non_empty(self):
        """``__version__`` is non-empty."""
        assert _version_mod.__version__

    def test_version_looks_like_semver(self):
        """``__version__`` matches a semver-like pattern (MAJOR.MINOR[.PATCH]…)."""
        pattern = r"^\d+\.\d+(\.\d+)?.*$"
        assert re.match(pattern, _version_mod.__version__), (
            f"Unexpected version string: {_version_mod.__version__!r}"
        )

    def test_version_has_no_leading_trailing_whitespace(self):
        """Version string has no leading or trailing whitespace."""
        v = _version_mod.__version__
        assert v == v.strip(), (
            f"Version string has surrounding whitespace: {v!r}"
        )

    def test_version_major_is_positive_int(self):
        """The MAJOR component of the version is a non-negative integer."""
        major = _version_mod.__version__.split(".")[0]
        assert major.isdigit(), f"Non-numeric major version: {major!r}"
        assert int(major) >= 0

    def test_version_minor_is_non_negative_int(self):
        """The MINOR component of the version is a non-negative integer."""
        parts = _version_mod.__version__.split(".")
        assert len(parts) >= 2, "Version string has fewer than 2 components"
        minor = parts[1]
        # Strip any non-digit suffix (e.g. '3rc1' -> '3')
        numeric = "".join(c for c in minor if c.isdigit())
        assert numeric, f"Minor version has no numeric part: {minor!r}"
        assert int(numeric) >= 0


# ===========================================================================
# Package-level contract tests (always runnable — no compiled extension needed)
# ===========================================================================

class TestVersionPackageContract:
    """Tests for the ``nc._version`` package ``__init__`` contract.

    These tests verify the ``__all__`` list and package-level symbol exports
    without touching any compiled extension.  They are skipped gracefully
    (not failed) when the package itself cannot be imported.
    """

    def test_package_importable_or_skip(self):
        """nc._version can be imported in this environment."""
        if not _PKG_AVAILABLE:
            pytest.skip("scikitplot.nc._version not importable")
        assert _ver_pkg is not None

    def test_package_all_contains_version_string(self):
        """nc._version.__all__ includes '__version__' (pybind11 string export)."""
        if not _PKG_AVAILABLE:
            pytest.skip("scikitplot.nc._version not importable")
        assert "__version__" in _ver_pkg_all, (
            f"'__version__' missing from nc._version.__all__: {_ver_pkg_all!r}"
        )

    def test_package_all_contains_version_module(self):
        """nc._version.__all__ includes '_version' (the Cython module name)."""
        if not _PKG_AVAILABLE:
            pytest.skip("scikitplot.nc._version not importable")
        assert "_version" in _ver_pkg_all, (
            f"'_version' missing from nc._version.__all__: {_ver_pkg_all!r}"
        )

    def test_package_all_is_list_or_tuple(self):
        """nc._version.__all__ is a list or tuple (standard convention)."""
        if not _PKG_AVAILABLE:
            pytest.skip("scikitplot.nc._version not importable")
        assert isinstance(_ver_pkg_all, (list, tuple))

    def test_package_exposes_version_string_equal_to_cython_module(self):
        """nc._version.__version__ matches the Cython module's __version__."""
        if not _PKG_AVAILABLE:
            pytest.skip("scikitplot.nc._version not importable")
        if not _CYTHON_AVAILABLE:
            pytest.skip("Cython extension not compiled; cannot compare versions")

        assert hasattr(_ver_pkg, "__version__"), (
            "nc._version package does not expose __version__"
        )
        assert _ver_pkg.__version__ == _version_mod.__version__, (
            f"Package __version__ {_ver_pkg.__version__!r} != "
            f"Cython module __version__ {_version_mod.__version__!r}"
        )
