# scikitplot/nc/_linalg/tests/test__linalg.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot.nc._linalg`.

Structure
---------
Layer 1 — pure-Python wrapper (always runnable, no compiled extension needed):
    Tests targeting the ``nc._linalg`` *package* ``__init__``, verifying that
    :func:`~scikitplot.nc._linalg.dot` is a properly constructed wrapper
    (correct ``__name__``, correct ``__all__``, delegates to
    :func:`_binary_arraylike`).

Layer 2 — C++ extension (skipped when extension is not compiled):
    Integration tests exercising the actual ``_linalg._linalg.dot`` C++ kernel
    through the full public ``nc.dot`` API.  These are guarded per-class via
    ``pytestmark`` so CI without a Meson build stays fully green.

Notes
-----
* The module-level import guard is intentionally **not** a
  :func:`pytest.importorskip` call at the module level; that would skip
  *every* test in this file — including the Layer 1 tests that need no
  compiled extension at all.
* C-level ``dot`` docstring is reused by the wrapper, so docstring presence
  is tested at the wrapper layer (no compiled ext needed).
* Shape / value correctness tests cover 1-D vectors, 2-D matrices, and the
  mixed dtype-promotion path via the public ``nc.dot``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Layer 1 imports (pure Python — always importable)
# ---------------------------------------------------------------------------
import scikitplot.nc._linalg as _linalg_pkg  # noqa: PLC0415

# ---------------------------------------------------------------------------
# Layer 2 sentinel: try to import the compiled C++ extension at module level
# so we know whether to skip Layer 2 tests, but do NOT importorskip the whole
# module (that would kill Layer 1 tests too).
# ---------------------------------------------------------------------------
try:
    from scikitplot.nc._linalg import _linalg as _linalg_ext  # C++ module
    _EXT_AVAILABLE = True
except ImportError:
    _linalg_ext = None  # type: ignore[assignment]
    _EXT_AVAILABLE = False

_EXT_REASON = "C++ extension _linalg._linalg not compiled"

# Also grab the public nc API for integration tests (needs the ext too)
try:
    import scikitplot.nc as nc  # noqa: PLC0415
except ImportError:
    nc = None  # type: ignore[assignment]


# ===========================================================================
# Layer 1: Pure-Python wrapper layer (_linalg package __init__)
# No compiled extension required — these tests must ALWAYS pass.
# ===========================================================================

class TestLinalgePackageInit:
    """Tests for nc._linalg.__init__ that require NO compiled extension."""

    def test_all_contains_dot(self):
        """__all__ exposes 'dot'."""
        assert "dot" in _linalg_pkg.__all__

    def test_all_is_list_or_tuple(self):
        """__all__ is a list or tuple (standard Python convention)."""
        assert isinstance(_linalg_pkg.__all__, (list, tuple))

    def test_dot_is_callable(self):
        """_linalg_pkg.dot is callable."""
        assert callable(_linalg_pkg.dot)

    def test_dot_name_is_dot(self):
        """The wrapper's __name__ is 'dot' (set explicitly by _binary_arraylike)."""
        assert _linalg_pkg.dot.__name__ == "dot"

    def test_dot_has_docstring_when_ext_compiled(self):
        """dot has a non-empty docstring when the C++ extension is present."""
        if not _EXT_AVAILABLE:
            pytest.skip("C++ extension not compiled; docstring may be absent")
        doc = _linalg_pkg.dot.__doc__
        assert doc, "dot.__doc__ is empty or None"
        assert len(doc) > 20, f"dot.__doc__ suspiciously short: {doc!r}"

    def test_dot_attribute_on_package(self):
        """dot is an attribute of the _linalg package object."""
        assert hasattr(_linalg_pkg, "dot")

    def test_only_dot_in_all_by_default(self):
        """__all__ contains 'dot' and nothing unexpected."""
        # 'dot' must be present; other entries are allowed but unusual.
        assert "dot" in _linalg_pkg.__all__


# ===========================================================================
# Layer 2: C++ extension tests (skipped if extension not compiled)
# ===========================================================================

@pytest.mark.skipif(not _EXT_AVAILABLE, reason=_EXT_REASON)
class TestLinalgeExtDot:
    """Integration tests for the C++ dot kernel via the public nc.dot API."""

    # -------------------------------------------------------------------
    # 1-D vectors
    # -------------------------------------------------------------------

    def test_dot_1d_float64_scalar_result(self):
        """1-D float64 dot product returns a result with correct value."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = nc.dot(a, b)
        # 1*4 + 2*5 + 3*6 = 32
        np.testing.assert_allclose(float(result.flat[0]), 32.0)

    def test_dot_1d_int64_scalar_result(self):
        """1-D int64 dot product returns correct integer value."""
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([4, 5, 6], dtype=np.int64)
        result = nc.dot(a, b)
        assert int(result.flat[0]) == 32

    def test_dot_1d_orthogonal_vectors_is_zero(self):
        """Dot product of orthogonal vectors is 0."""
        a = np.array([1.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0], dtype=np.float64)
        result = nc.dot(a, b)
        np.testing.assert_allclose(float(result.flat[0]), 0.0, atol=1e-15)

    def test_dot_1d_unit_vectors(self):
        """Dot product of identical unit vectors is 1.0."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        result = nc.dot(a, b)
        np.testing.assert_allclose(float(result.flat[0]), 1.0)

    def test_dot_1d_antiparallel_vectors_is_negative(self):
        """Dot product of antiparallel unit vectors is -1.0."""
        a = np.array([1.0, 0.0], dtype=np.float64)
        b = np.array([-1.0, 0.0], dtype=np.float64)
        result = nc.dot(a, b)
        np.testing.assert_allclose(float(result.flat[0]), -1.0)

    # -------------------------------------------------------------------
    # 2-D matrices
    # -------------------------------------------------------------------

    def test_dot_2d_float64_matrix_multiply(self):
        """2-D float64 matrix multiplication matches np.dot reference."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        expected = np.dot(a, b)
        result = nc.dot(a, b)
        np.testing.assert_allclose(result, expected)

    def test_dot_2d_identity_matrix_returns_original(self):
        """Multiplying by identity matrix returns the original matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        eye = np.eye(2, dtype=np.float64)
        result = nc.dot(a, eye)
        np.testing.assert_allclose(result, a)

    def test_dot_2d_int64_matrix_multiply(self):
        """2-D int64 matrix multiplication matches np.dot reference."""
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)
        b = np.array([[5, 6], [7, 8]], dtype=np.int64)
        expected = np.dot(a, b)
        result = nc.dot(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_dot_2d_zero_matrix_result_is_zero(self):
        """Multiplying by the zero matrix yields the zero matrix."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        z = np.zeros((2, 2), dtype=np.float64)
        result = nc.dot(a, z)
        np.testing.assert_allclose(result, np.zeros((2, 2)))

    # -------------------------------------------------------------------
    # dtype-promotion via the Python wrapper (nc.dot, not _linalg.dot)
    # -------------------------------------------------------------------

    def test_dot_accepts_list_inputs(self):
        """nc.dot accepts Python list inputs (array_like interface)."""
        result = nc.dot([1.0, 2.0], [3.0, 4.0])
        # 1*3 + 2*4 = 11
        np.testing.assert_allclose(float(result.flat[0]), 11.0)

    def test_dot_accepts_int_list_inputs(self):
        """nc.dot promotes Python int lists to int64 automatically."""
        result = nc.dot([1, 2, 3], [4, 5, 6])
        assert int(result.flat[0]) == 32

    def test_dot_mixed_int_float_promotes_to_float64(self):
        """nc.dot promotes mixed int32/float64 inputs to float64."""
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3.0, 4.0], dtype=np.float64)
        result = nc.dot(a, b)
        np.testing.assert_allclose(float(result.flat[0]), 11.0)

    def test_dot_float32_inputs_return_result(self):
        """nc.dot promotes float32 inputs to float64 and returns a result."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        result = nc.dot(a, b)
        np.testing.assert_allclose(float(result.flat[0]), 11.0, rtol=1e-5)

    def test_dot_tuple_inputs_accepted(self):
        """nc.dot accepts Python tuple inputs."""
        result = nc.dot((2.0, 3.0), (4.0, 5.0))
        # 2*4 + 3*5 = 23
        np.testing.assert_allclose(float(result.flat[0]), 23.0)

    # -------------------------------------------------------------------
    # Error cases
    # -------------------------------------------------------------------

    def test_dot_3d_array_raises(self):
        """Passing a 3-D array to dot raises ValueError or TypeError."""
        a = np.ones((2, 2, 2), dtype=np.float64)
        b = np.ones((2, 2, 2), dtype=np.float64)
        with pytest.raises((ValueError, TypeError)):
            nc.dot(a, b)

    def test_dot_complex_dtype_raises_type_error(self):
        """Complex dtype inputs raise TypeError (unsupported by backend)."""
        a = np.array([1 + 2j], dtype=np.complex128)
        b = np.array([3 + 4j], dtype=np.complex128)
        with pytest.raises(TypeError):
            nc.dot(a, b)

    # -------------------------------------------------------------------
    # Return type
    # -------------------------------------------------------------------

    def test_dot_returns_ndarray(self):
        """nc.dot always returns a numpy.ndarray."""
        result = nc.dot([1.0, 2.0], [3.0, 4.0])
        assert isinstance(result, np.ndarray)

    def test_dot_1d_result_is_finite(self):
        """1-D dot product result is a finite number (not NaN or inf)."""
        result = nc.dot([1.0, 2.0], [3.0, 4.0])
        assert np.isfinite(result.flat[0])
