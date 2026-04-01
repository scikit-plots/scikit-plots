# scikitplot/nc/_version/tests/test__version.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot.nc._version.__version__` (pybind11 module).

The ``__version__`` sub-module is compiled from ``src/module_version.cpp``
using pybind11.  It exposes the ``nc::VERSION`` constant as the Python
string attribute ``__version__``.

Tests are guarded per-class with ``pytest.mark.skipif`` so the suite stays
fully green in build-free environments that have no compiled extension.

What is tested
--------------
Attribute contract
    ``__version__`` exists, is a non-empty ``str``, and follows semver.

Consistency
    The version exposed by the pybind11 module agrees with the Cython
    module (``_version._version.__version__``) when both are compiled,
    since both wrap the same ``nc::VERSION`` constant.

Package integration
    ``nc._version.__version__`` is re-exported at the package level and
    equals the pybind11 module's value.

Notes
-----
* The pybind11 module name *is* ``__version__``, which is imported by the
  ``_version`` package ``__init__`` as::

      from .__version import __version__

  Therefore ``nc._version.__version__`` (the string attribute) must equal
  ``nc._version.__version__.__version__`` (the pybind11 module attribute)
  — a deliberate but slightly unusual naming choice.
* No arithmetic is performed on the version string; only structural /
  lexicographic properties are verified.
"""

from __future__ import annotations

import re

import pytest

# ---------------------------------------------------------------------------
# pybind11 __version__ module availability guard
# ---------------------------------------------------------------------------
try:
    import scikitplot.nc._version.__version__ as _pyb11_ver_mod  # noqa: PLC0415
    _PYBIND_AVAILABLE = True
except ImportError:
    _pyb11_ver_mod = None  # type: ignore[assignment]
    _PYBIND_AVAILABLE = False

_PYBIND_REASON = "pybind11 extension _version.__version__ not compiled"

# ---------------------------------------------------------------------------
# Cython module availability (for cross-comparison tests)
# ---------------------------------------------------------------------------
try:
    import scikitplot.nc._version._version as _cython_ver_mod   # noqa: PLC0415
    _CYTHON_AVAILABLE = True
except ImportError:
    _cython_ver_mod = None  # type: ignore[assignment]
    _CYTHON_AVAILABLE = False

# ---------------------------------------------------------------------------
# Package availability (for integration tests)
# ---------------------------------------------------------------------------
try:
    import scikitplot.nc._version as _ver_pkg   # noqa: PLC0415
    _PKG_AVAILABLE = True
except ImportError:
    _ver_pkg = None  # type: ignore[assignment]
    _PKG_AVAILABLE = False


# ===========================================================================
# pybind11 __version__ module — attribute contract
# ===========================================================================

@pytest.mark.skipif(not _PYBIND_AVAILABLE, reason=_PYBIND_REASON)
class TestPybind11VersionModule:
    """Tests for the pybind11 ``__version__`` extension module."""

    def test_has_version_attribute(self):
        """Module exposes a ``__version__`` attribute."""
        assert hasattr(_pyb11_ver_mod, "__version__")

    def test_version_is_str(self):
        """``__version__`` is a Python ``str``."""
        assert isinstance(_pyb11_ver_mod.__version__, str)

    def test_version_is_non_empty(self):
        """``__version__`` is not the empty string."""
        assert _pyb11_ver_mod.__version__, (
            "pybind11 __version__ module returned an empty version string"
        )

    def test_version_has_no_leading_trailing_whitespace(self):
        """Version string has no surrounding whitespace."""
        v = _pyb11_ver_mod.__version__
        assert v == v.strip(), (
            f"Whitespace detected around version string: {v!r}"
        )

    def test_version_matches_semver_pattern(self):
        """``__version__`` matches a semver-like pattern (MAJOR.MINOR[.PATCH]…)."""
        pattern = r"^\d+\.\d+(\.\d+)?.*$"
        assert re.match(pattern, _pyb11_ver_mod.__version__), (
            f"Unexpected version string format: {_pyb11_ver_mod.__version__!r}"
        )

    def test_version_major_is_digit(self):
        """The MAJOR component is a digit string."""
        major = _pyb11_ver_mod.__version__.split(".")[0]
        assert major.isdigit(), f"Non-numeric MAJOR component: {major!r}"

    def test_version_minor_component_present(self):
        """The version string has at least a MAJOR.MINOR structure."""
        parts = _pyb11_ver_mod.__version__.split(".")
        assert len(parts) >= 2, (
            f"Version string lacks MINOR component: {_pyb11_ver_mod.__version__!r}"
        )

    def test_version_minor_starts_with_digit(self):
        """The MINOR component starts with a digit."""
        parts = _pyb11_ver_mod.__version__.split(".")
        assert len(parts) >= 2
        assert parts[1][0].isdigit(), (
            f"MINOR component does not start with a digit: {parts[1]!r}"
        )

    def test_version_does_not_start_with_v(self):
        """Version string does not begin with a 'v' prefix (PEP 440 style)."""
        assert not _pyb11_ver_mod.__version__.startswith("v"), (
            f"Version should not start with 'v': {_pyb11_ver_mod.__version__!r}"
        )

    def test_version_is_ascii(self):
        """Version string contains only ASCII characters."""
        assert _pyb11_ver_mod.__version__.isascii(), (
            f"Non-ASCII character in version: {_pyb11_ver_mod.__version__!r}"
        )


# ===========================================================================
# Cross-version consistency: pybind11 vs Cython
# ===========================================================================

@pytest.mark.skipif(
    not (_PYBIND_AVAILABLE and _CYTHON_AVAILABLE),
    reason="Both pybind11 and Cython extensions must be compiled for comparison",
)
class TestPybind11CythonVersionConsistency:
    """Verify pybind11 and Cython modules expose the same nc::VERSION string."""

    def test_pybind11_version_equals_cython_version(self):
        """pybind11 __version__ == Cython _version.__version__ (same C++ constant)."""
        assert _pyb11_ver_mod.__version__ == _cython_ver_mod.__version__, (
            f"Version mismatch: pybind11 reports {_pyb11_ver_mod.__version__!r}, "
            f"Cython reports {_cython_ver_mod.__version__!r}"
        )


# ===========================================================================
# Package integration: nc._version re-exports the pybind11 __version__ string
# ===========================================================================

@pytest.mark.skipif(
    not (_PYBIND_AVAILABLE and _PKG_AVAILABLE),
    reason="pybind11 extension and nc._version package must both be available",
)
class TestVersionPackageIntegration:
    """Verify nc._version package correctly re-exports the pybind11 __version__."""

    def test_package_version_str_equals_pybind11_module_version(self):
        """nc._version.__version__ (str) equals the pybind11 module's __version__."""
        assert hasattr(_ver_pkg, "__version__"), (
            "nc._version package does not expose a top-level __version__ attribute"
        )
        assert _ver_pkg.__version__ == _pyb11_ver_mod.__version__, (
            f"Package re-export mismatch: "
            f"nc._version.__version__={_ver_pkg.__version__!r} "
            f"but pybind11 module.__version__={_pyb11_ver_mod.__version__!r}"
        )

    def test_package_version_is_str(self):
        """nc._version.__version__ is a str after package import."""
        assert isinstance(_ver_pkg.__version__, str)

    def test_package_version_is_non_empty(self):
        """nc._version.__version__ is non-empty after package import."""
        assert _ver_pkg.__version__


# ===========================================================================
# Standalone: ensure tests are meaningful even without any compiled extension
# ===========================================================================

class TestNoBuildEnvironment:
    """Smoke tests for build-free environments (no compiled extension).

    These always run and verify that the skip guards themselves are correct
    (i.e. we don't accidentally run extension-dependent assertions when the
    extension is absent).
    """

    def test_pybind_available_flag_is_bool(self):
        """_PYBIND_AVAILABLE is a bool (guards are well-formed)."""
        assert isinstance(_PYBIND_AVAILABLE, bool)

    def test_cython_available_flag_is_bool(self):
        """_CYTHON_AVAILABLE is a bool."""
        assert isinstance(_CYTHON_AVAILABLE, bool)

    def test_pkg_available_flag_is_bool(self):
        """_PKG_AVAILABLE is a bool."""
        assert isinstance(_PKG_AVAILABLE, bool)

    def test_semver_regex_matches_typical_numcpp_version(self):
        """The semver regex used in tests correctly matches typical NumCpp versions."""
        pattern = r"^\d+\.\d+(\.\d+)?.*$"
        valid_examples = ["2.3.0", "1.0", "2.3.0rc1", "10.2.1"]
        for version in valid_examples:
            assert re.match(pattern, version), (
                f"Semver regex does not match expected version: {version!r}"
            )

    def test_semver_regex_rejects_invalid_strings(self):
        """The semver regex correctly rejects clearly invalid version strings."""
        pattern = r"^\d+\.\d+(\.\d+)?.*$"
        invalid_examples = ["v2.3.0", "abc", ""]
        for bad in invalid_examples:
            assert not re.match(pattern, bad), (
                f"Semver regex unexpectedly matched invalid string: {bad!r}"
            )
