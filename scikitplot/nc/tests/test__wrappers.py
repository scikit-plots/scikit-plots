# scikitplot/nc/tests/test__wrappers.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot.nc._wrappers`.

Coverage targets
----------------
* :func:`~scikitplot.nc._wrappers._promote_to_supported_dtype`

  - All supported dtype branches: ``float*``, ``int*``, ``uint*``, ``bool_``,
    ``object`` (numeric & non-numeric), complex (unsupported).
  - Cross-kind promotions (int + float → float64).
  - Memory layout: F-contiguous inputs must become C-contiguous.
  - Value preservation after dtype cast.
  - ``func_name`` propagation into error messages.

* :func:`~scikitplot.nc._wrappers._binary_arraylike`

  - Callable creation and ``__name__`` / ``__doc__`` assignment.
  - ``promote_dtypes=True``: dtype promotion, C-contiguity, single call.
  - ``promote_dtypes=False``: no promotion, ``require_same_dtype`` guard.
  - Return value forwarding.
  - Default parameter values.

Design principles
-----------------
* Every dtype-promotion path is exercised explicitly.
* Edge cases (empty arrays, 0-D scalars, F-order, object dtype, scalars)
  each have a dedicated test so failures pinpoint the exact code path.
* Mocked ``core`` callables capture their arguments so assertions can
  verify exactly what reaches the C++ layer.
* Tests are independent: no shared mutable state, no order dependency.

Notes
-----
These are pure-Python helpers — no compiled C++ extension is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Portable import — works both inside the full scikitplot tree and standalone.
# ---------------------------------------------------------------------------
from scikitplot.nc._wrappers import (
    _binary_arraylike,
    _promote_to_supported_dtype,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_core(return_value: Any = None):
    """Return a simple callable that records its positional arguments.

    Parameters
    ----------
    return_value : any, optional
        Value returned by the mock core. Defaults to ``np.array(0)``.

    Returns
    -------
    callable
        A mock core with a ``_calls`` attribute recording every invocation.
    """
    _calls: list[tuple] = []

    def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        _calls.append((a, b))
        return return_value if return_value is not None else np.array(0)

    core.__name__ = "mock_core"
    core.__doc__ = "Mock core docstring."
    core._calls = _calls  # type: ignore[attr-defined]
    return core


# ===========================================================================
# _promote_to_supported_dtype — dtype-promotion logic
# ===========================================================================

class TestPromoteDtype:
    """Tests for :func:`_promote_to_supported_dtype`."""

    # -------------------------------------------------------------------
    # Float inputs -> float64
    # -------------------------------------------------------------------

    def test_float64_float64_stays_float64(self):
        """Both float64 -> target is float64 (no copy required)."""
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    def test_float32_float32_promoted_to_float64(self):
        """Two float32 arrays -> both promoted to float64."""
        a = np.array([1.0], dtype=np.float32)
        b = np.array([2.0], dtype=np.float32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    def test_float16_float16_promoted_to_float64(self):
        """Two float16 arrays -> both promoted to float64 (float branch)."""
        a = np.array([1.0], dtype=np.float16)
        b = np.array([2.0], dtype=np.float16)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    def test_float32_float64_promoted_to_float64(self):
        """Mixed float32 / float64 -> both float64."""
        a = np.array([1.0], dtype=np.float32)
        b = np.array([2.0], dtype=np.float64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    # -------------------------------------------------------------------
    # Signed integer inputs -> int64
    # -------------------------------------------------------------------

    def test_int64_int64_stays_int64(self):
        """Both int64 -> target is int64 (no copy required)."""
        a = np.array([1, 2], dtype=np.int64)
        b = np.array([3, 4], dtype=np.int64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_int32_int32_promoted_to_int64(self):
        """Two int32 arrays -> both promoted to int64."""
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_int16_int16_promoted_to_int64(self):
        """Two int16 arrays -> both promoted to int64 (integer branch)."""
        a = np.array([1, 2], dtype=np.int16)
        b = np.array([3, 4], dtype=np.int16)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_int8_int8_promoted_to_int64(self):
        """Two int8 arrays -> both promoted to int64."""
        a = np.array([1, 2], dtype=np.int8)
        b = np.array([3, 4], dtype=np.int8)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_int32_int64_promoted_to_int64(self):
        """Mixed int32 / int64 -> both int64."""
        a = np.array([1], dtype=np.int32)
        b = np.array([2], dtype=np.int64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    # -------------------------------------------------------------------
    # Unsigned integer inputs -> int64
    # -------------------------------------------------------------------

    def test_uint8_uint8_promoted_to_int64(self):
        """Two uint8 arrays -> both promoted to int64 (integer branch)."""
        a = np.array([1, 2], dtype=np.uint8)
        b = np.array([3, 4], dtype=np.uint8)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_uint16_uint16_promoted_to_int64(self):
        """Two uint16 arrays -> both promoted to int64."""
        a = np.array([1, 2], dtype=np.uint16)
        b = np.array([3, 4], dtype=np.uint16)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_uint32_uint32_promoted_to_int64(self):
        """Two uint32 arrays -> both promoted to int64."""
        a = np.array([1, 2], dtype=np.uint32)
        b = np.array([3, 4], dtype=np.uint32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_uint32_int64_cross_promoted_to_int64(self):
        """uint32 + int64 -> both int64 (all-integer branch)."""
        a = np.array([5], dtype=np.uint32)
        b = np.array([10], dtype=np.int64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    # -------------------------------------------------------------------
    # Boolean inputs -> int64
    # -------------------------------------------------------------------

    def test_bool_bool_promoted_to_int64(self):
        """Two bool arrays -> both promoted to int64."""
        a = np.array([True, False], dtype=np.bool_)
        b = np.array([False, True], dtype=np.bool_)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_bool_int32_promoted_to_int64(self):
        """bool + int32 -> both int64."""
        a = np.array([True], dtype=np.bool_)
        b = np.array([5], dtype=np.int32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_bool_uint8_promoted_to_int64(self):
        """bool + uint8 -> both int64."""
        a = np.array([True], dtype=np.bool_)
        b = np.array([200], dtype=np.uint8)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    # -------------------------------------------------------------------
    # Cross-kind: integer + float -> float64
    # -------------------------------------------------------------------

    def test_int32_float64_cross_type_to_float64(self):
        """int32 + float64 -> NumPy result_type gives float64."""
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3.0, 4.0], dtype=np.float64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    def test_int64_float32_cross_type_to_float64(self):
        """int64 + float32 -> float64 (floating branch takes precedence)."""
        a = np.array([1], dtype=np.int64)
        b = np.array([2.5], dtype=np.float32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    def test_bool_float64_cross_type_to_float64(self):
        """bool + float64 -> float64 (floating branch takes precedence)."""
        a = np.array([True, False], dtype=np.bool_)
        b = np.array([1.0, 2.0], dtype=np.float64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    def test_uint8_float32_cross_type_to_float64(self):
        """uint8 + float32 -> float64."""
        a = np.array([1], dtype=np.uint8)
        b = np.array([1.5], dtype=np.float32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    # -------------------------------------------------------------------
    # Object dtype handling
    # -------------------------------------------------------------------

    def test_object_with_none_becomes_float64_nan(self):
        """Object array containing None -> float64 with NaN for None entries."""
        a = np.array([1, None, 2], dtype=object)
        b = np.array([3, 4, None], dtype=object)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64
        assert np.isnan(a_out[1])
        assert np.isnan(b_out[2])

    def test_object_all_numeric_converts_to_float64(self):
        """Object array of Python floats -> float64 via numeric coercion."""
        a = np.array([1.5, 2.5], dtype=object)
        b = np.array([3.5, 4.5], dtype=object)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    def test_object_with_string_raises_typeerror(self):
        """Object array containing a non-numeric string -> TypeError."""
        a = np.array(["hello", "world"], dtype=object)
        b = np.array([1.0, 2.0], dtype=object)
        with pytest.raises(TypeError):
            _promote_to_supported_dtype(a, b)

    def test_object_nonnumeric_error_mentions_func_name(self):
        """TypeError message for object dtype contains the func_name."""
        a = np.array(["bad"], dtype=object)
        b = np.array([1.0], dtype=object)
        with pytest.raises(TypeError, match="my_func"):
            _promote_to_supported_dtype(a, b, func_name="my_func")

    # -------------------------------------------------------------------
    # Unsupported / complex dtypes
    # -------------------------------------------------------------------

    def test_complex128_raises_typeerror(self):
        """complex128 is not a supported backend dtype -> TypeError."""
        a = np.array([1 + 2j], dtype=np.complex128)
        b = np.array([3 + 4j], dtype=np.complex128)
        with pytest.raises(TypeError):
            _promote_to_supported_dtype(a, b)

    def test_complex64_raises_typeerror(self):
        """complex64 is not a supported backend dtype -> TypeError."""
        a = np.array([1 + 2j], dtype=np.complex64)
        b = np.array([3 + 4j], dtype=np.complex64)
        with pytest.raises(TypeError):
            _promote_to_supported_dtype(a, b)

    def test_unsupported_dtype_error_mentions_complex_kind(self):
        """TypeError for unsupported dtype mentions something identifiable."""
        a = np.array([1 + 2j], dtype=np.complex128)
        b = np.array([3 + 4j], dtype=np.complex128)
        with pytest.raises(TypeError, match="complex"):
            _promote_to_supported_dtype(a, b)

    # -------------------------------------------------------------------
    # func_name propagation
    # -------------------------------------------------------------------

    def test_default_func_name_is_dot(self):
        """Default func_name 'dot' appears in object-dtype error messages."""
        a = np.array(["bad"], dtype=object)
        b = np.array([1.0], dtype=object)
        with pytest.raises(TypeError, match="dot"):
            _promote_to_supported_dtype(a, b)

    def test_custom_func_name_in_complex_error(self):
        """Custom func_name propagates into TypeError for complex dtype."""
        a = np.array([1 + 2j])
        b = np.array([3 + 4j])
        with pytest.raises(TypeError, match="custom_func"):
            _promote_to_supported_dtype(a, b, func_name="custom_func")

    def test_custom_func_name_in_object_error(self):
        """Custom func_name propagates into TypeError for non-numeric object dtype."""
        a = np.array(["x"], dtype=object)
        b = np.array([1.0], dtype=object)
        with pytest.raises(TypeError, match="special_name"):
            _promote_to_supported_dtype(a, b, func_name="special_name")

    # -------------------------------------------------------------------
    # List / tuple / scalar inputs
    # -------------------------------------------------------------------

    def test_list_inputs_converted_to_ndarray(self):
        """Plain Python lists are converted to np.ndarray."""
        a_out, b_out = _promote_to_supported_dtype([1.0, 2.0], [3.0, 4.0])
        assert isinstance(a_out, np.ndarray)
        assert isinstance(b_out, np.ndarray)

    def test_tuple_inputs_converted_to_ndarray(self):
        """Plain Python tuples are converted to np.ndarray."""
        a_out, b_out = _promote_to_supported_dtype((1, 2), (3, 4))
        assert isinstance(a_out, np.ndarray)
        assert isinstance(b_out, np.ndarray)

    def test_python_int_list_promoted_to_int64(self):
        """List of Python ints -> int64 (integer branch)."""
        a_out, b_out = _promote_to_supported_dtype([1, 2], [3, 4])
        assert a_out.dtype == np.int64
        assert b_out.dtype == np.int64

    def test_python_float_list_promoted_to_float64(self):
        """List of Python floats -> float64 (floating branch)."""
        a_out, b_out = _promote_to_supported_dtype([1.0, 2.0], [3.0, 4.0])
        assert a_out.dtype == np.float64
        assert b_out.dtype == np.float64

    def test_scalar_0d_array_accepted(self):
        """0-D array-scalars are accepted without error."""
        a = np.float64(3.0)
        b = np.float64(5.0)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.float64

    def test_scalar_int_0d_accepted(self):
        """0-D int64 scalars are accepted and stay int64."""
        a = np.int64(3)
        b = np.int64(7)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.dtype == np.int64

    # -------------------------------------------------------------------
    # Output memory layout
    # -------------------------------------------------------------------

    def test_output_is_c_contiguous_for_c_input(self):
        """C-contiguous inputs produce C-contiguous outputs."""
        a = np.ones((2, 2), dtype=np.float64, order="C")
        b = np.ones((2, 2), dtype=np.float64, order="C")
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.flags["C_CONTIGUOUS"]
        assert b_out.flags["C_CONTIGUOUS"]

    def test_f_contiguous_input_becomes_c_contiguous(self):
        """Fortran-order arrays are re-ordered to C-contiguous."""
        a = np.ones((3, 3), dtype=np.float64, order="F")
        b = np.ones((3, 3), dtype=np.float64, order="F")
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.flags["C_CONTIGUOUS"]
        assert b_out.flags["C_CONTIGUOUS"]

    # -------------------------------------------------------------------
    # Return type
    # -------------------------------------------------------------------

    def test_returns_tuple_of_two_ndarrays(self):
        """Return value is a 2-tuple of np.ndarray."""
        result = _promote_to_supported_dtype([1.0], [2.0])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, np.ndarray) for x in result)

    # -------------------------------------------------------------------
    # Dimensionality
    # -------------------------------------------------------------------

    def test_2d_arrays_shape_preserved(self):
        """2-D arrays are handled correctly (shape is preserved)."""
        a = np.ones((2, 3), dtype=np.float64)
        b = np.ones((3, 2), dtype=np.float64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.shape == (2, 3)
        assert b_out.shape == (3, 2)

    def test_empty_array_handled(self):
        """Zero-length arrays are accepted without error."""
        a = np.array([], dtype=np.float64)
        b = np.array([], dtype=np.float64)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.shape == (0,)
        assert b_out.shape == (0,)
        assert a_out.dtype == np.float64

    def test_empty_int_array_handled(self):
        """Zero-length integer arrays are accepted and promote to int64."""
        a = np.array([], dtype=np.int32)
        b = np.array([], dtype=np.int32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        assert a_out.shape == (0,)
        assert a_out.dtype == np.int64

    # -------------------------------------------------------------------
    # Values preserved through promotion
    # -------------------------------------------------------------------

    def test_int32_values_preserved_after_cast_to_int64(self):
        """Integer values are exactly preserved through int32->int64 cast."""
        a = np.array([10, 20, 30], dtype=np.int32)
        b = np.array([1, 2, 3], dtype=np.int32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        np.testing.assert_array_equal(a_out, [10, 20, 30])
        np.testing.assert_array_equal(b_out, [1, 2, 3])

    def test_float32_values_preserved_after_cast_to_float64(self):
        """Float values are preserved within float64 precision after cast."""
        a = np.array([1.5, 2.5], dtype=np.float32)
        b = np.array([3.5, 4.5], dtype=np.float32)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        np.testing.assert_allclose(a_out, [1.5, 2.5], rtol=1e-6)
        np.testing.assert_allclose(b_out, [3.5, 4.5], rtol=1e-6)

    def test_uint8_values_preserved_after_cast_to_int64(self):
        """uint8 values in range [0,255] are exactly preserved through -> int64."""
        a = np.array([0, 127, 255], dtype=np.uint8)
        b = np.array([10, 20, 30], dtype=np.uint8)
        a_out, b_out = _promote_to_supported_dtype(a, b)
        np.testing.assert_array_equal(a_out, [0, 127, 255])
        np.testing.assert_array_equal(b_out, [10, 20, 30])


# ===========================================================================
# _binary_arraylike — decorator / factory for C++ kernel wrappers
# ===========================================================================

class TestBinaryArraylike:
    """Tests for :func:`_binary_arraylike`."""

    # -------------------------------------------------------------------
    # Return type and basic attributes
    # -------------------------------------------------------------------

    def test_returns_callable(self):
        """Factory returns a callable (the wrapped function)."""
        core = _make_core()
        wrapped = _binary_arraylike(core)
        assert callable(wrapped)

    def test_name_set_from_name_param(self):
        """__name__ of wrapped function equals the ``name`` argument."""
        core = _make_core()
        wrapped = _binary_arraylike(core, name="my_dot")
        assert wrapped.__name__ == "my_dot"

    def test_name_defaults_to_core_name(self):
        """When ``name`` is omitted, __name__ falls back to core.__name__."""
        core = _make_core()
        core.__name__ = "core_dot"
        wrapped = _binary_arraylike(core)
        assert wrapped.__name__ == "core_dot"

    def test_name_none_uses_core_name(self):
        """Passing name=None explicitly still uses core.__name__."""
        core = _make_core()
        core.__name__ = "fallback_name"
        wrapped = _binary_arraylike(core, name=None)
        assert wrapped.__name__ == "fallback_name"

    def test_doc_inherited_from_core(self):
        """__doc__ of the wrapper is the core's docstring (avoids duplication)."""
        core = _make_core()
        core.__doc__ = "Core implementation docstring."
        wrapped = _binary_arraylike(core)
        assert wrapped.__doc__ == "Core implementation docstring."

    def test_doc_is_none_when_core_doc_is_none(self):
        """When core.__doc__ is None, the wrapper's __doc__ is also None."""
        core = _make_core()
        core.__doc__ = None
        wrapped = _binary_arraylike(core)
        assert wrapped.__doc__ is None

    # -------------------------------------------------------------------
    # promote_dtypes=True (default) — dtype promotion path
    # -------------------------------------------------------------------

    def test_promote_dtypes_true_promotes_mixed_types(self):
        """With promote_dtypes=True, mixed int+float inputs are cast to float64."""
        received: dict = {}

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            received["a"] = a
            received["b"] = b
            return np.array(0.0)

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=True)
        wrapped(np.array([1], dtype=np.int32), np.array([2.0], dtype=np.float64))

        assert received["a"].dtype == np.float64
        assert received["b"].dtype == np.float64

    def test_promote_dtypes_true_list_inputs_converted(self):
        """List inputs are converted to ndarray before reaching core."""
        received: dict = {}

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            received["a"] = a
            received["b"] = b
            return np.array(0)

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=True)
        wrapped([1, 2], [3, 4])

        assert isinstance(received["a"], np.ndarray)
        assert isinstance(received["b"], np.ndarray)

    def test_promote_dtypes_true_core_receives_c_contiguous(self):
        """Core receives C-contiguous arrays even if F-order input is given."""
        received: dict = {}

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            received["a"] = a
            received["b"] = b
            return np.array(0.0)

        core.__name__ = "core"
        core.__doc__ = ""
        a = np.asfortranarray(np.ones((2, 2), dtype=np.float64))
        b = np.asfortranarray(np.ones((2, 2), dtype=np.float64))
        wrapped = _binary_arraylike(core, promote_dtypes=True)
        wrapped(a, b)

        assert received["a"].flags["C_CONTIGUOUS"]
        assert received["b"].flags["C_CONTIGUOUS"]

    def test_promote_dtypes_true_result_returned(self):
        """Return value from core is forwarded to the caller unchanged."""
        expected = np.array([42.0])

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return expected

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=True)
        result = wrapped([1.0], [2.0])
        assert result is expected

    def test_promote_dtypes_true_core_called_exactly_once(self):
        """Core is called exactly once per wrapped call."""
        call_count = [0]

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            call_count[0] += 1
            return np.array(0.0)

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=True)
        wrapped([1.0], [2.0])
        assert call_count[0] == 1

    def test_promote_dtypes_true_complex_raises_typeerror(self):
        """promote_dtypes=True with complex input raises TypeError (unsupported)."""
        core = _make_core()
        wrapped = _binary_arraylike(core, promote_dtypes=True)
        with pytest.raises(TypeError):
            wrapped(np.array([1 + 2j]), np.array([3 + 4j]))

    # -------------------------------------------------------------------
    # promote_dtypes=False — no dtype promotion
    # -------------------------------------------------------------------

    def test_promote_dtypes_false_does_not_promote(self):
        """With promote_dtypes=False, dtypes are passed through as-is."""
        received: dict = {}

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            received["a"] = a
            received["b"] = b
            return np.array(0)

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=False)
        wrapped(np.array([1], dtype=np.int32), np.array([2], dtype=np.int32))

        assert received["a"].dtype == np.int32
        assert received["b"].dtype == np.int32

    def test_promote_dtypes_false_list_inputs_still_converted(self):
        """Even without promotion, list inputs become ndarray."""
        received: dict = {}

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            received["a"] = a
            received["b"] = b
            return np.array(0)

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=False)
        wrapped([1, 2], [3, 4])

        assert isinstance(received["a"], np.ndarray)
        assert isinstance(received["b"], np.ndarray)

    def test_promote_dtypes_false_result_forwarded(self):
        """Return value from core is forwarded when promote_dtypes=False."""
        expected = np.array([99])

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return expected

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=False)
        result = wrapped(np.array([1]), np.array([2]))
        assert result is expected

    # -------------------------------------------------------------------
    # require_same_dtype (only active when promote_dtypes=False)
    # -------------------------------------------------------------------

    def test_require_same_dtype_raises_on_different_dtypes(self):
        """require_same_dtype=True, promote_dtypes=False -> TypeError for mismatch."""
        core = _make_core()
        wrapped = _binary_arraylike(core, promote_dtypes=False, require_same_dtype=True)
        with pytest.raises(TypeError):
            wrapped(
                np.array([1.0], dtype=np.float64),
                np.array([2], dtype=np.int32),
            )

    def test_require_same_dtype_error_mentions_both_dtypes(self):
        """TypeError message for dtype mismatch names at least the first dtype."""
        core = _make_core()
        wrapped = _binary_arraylike(core, promote_dtypes=False, require_same_dtype=True)
        with pytest.raises(TypeError, match="float64"):
            wrapped(
                np.array([1.0], dtype=np.float64),
                np.array([2], dtype=np.int32),
            )

    def test_require_same_dtype_error_mentions_func_name(self):
        """TypeError message for dtype mismatch includes the function name."""
        core = _make_core()
        wrapped = _binary_arraylike(
            core,
            name="special_dot",
            promote_dtypes=False,
            require_same_dtype=True,
        )
        with pytest.raises(TypeError, match="special_dot"):
            wrapped(
                np.array([1.0], dtype=np.float64),
                np.array([2], dtype=np.int32),
            )

    def test_require_same_dtype_false_no_error_on_mismatch(self):
        """require_same_dtype=False does not raise even for mismatched dtypes."""
        received: dict = {}

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            received["a_dtype"] = a.dtype
            return np.array(0)

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=False, require_same_dtype=False)
        wrapped(np.array([1.0], dtype=np.float64), np.array([2], dtype=np.int32))
        assert received["a_dtype"] == np.float64

    def test_require_same_dtype_true_same_types_does_not_raise(self):
        """require_same_dtype=True does NOT raise when dtypes are identical."""
        core = _make_core()
        wrapped = _binary_arraylike(core, promote_dtypes=False, require_same_dtype=True)
        # Must not raise:
        wrapped(np.array([1.0], dtype=np.float64), np.array([2.0], dtype=np.float64))

    # -------------------------------------------------------------------
    # Core callable always receives ndarray
    # -------------------------------------------------------------------

    def test_core_receives_ndarrays_promote_true(self):
        """promote_dtypes=True: core is called with np.ndarray, not raw list."""
        received_types: list[str] = []

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            received_types.append(type(a).__name__)
            received_types.append(type(b).__name__)
            return np.array(0.0)

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=True)
        wrapped([1.0, 2.0], [3.0, 4.0])
        assert received_types == ["ndarray", "ndarray"]

    def test_core_receives_ndarrays_promote_false(self):
        """promote_dtypes=False: core is also called with np.ndarray."""
        received_types: list[str] = []

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            received_types.append(type(a).__name__)
            received_types.append(type(b).__name__)
            return np.array(0)

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core, promote_dtypes=False)
        wrapped([1, 2], [3, 4])
        assert received_types == ["ndarray", "ndarray"]

    # -------------------------------------------------------------------
    # Interaction: promote_dtypes=True ignores require_same_dtype check
    # -------------------------------------------------------------------

    def test_require_same_dtype_ignored_when_promote_dtypes_true(self):
        """When promote_dtypes=True the require_same_dtype branch is not reached."""
        core = _make_core()
        wrapped = _binary_arraylike(core, promote_dtypes=True, require_same_dtype=True)
        # int + float -> both become float64; no require_same_dtype error.
        wrapped(np.array([1], dtype=np.int32), np.array([2.0], dtype=np.float64))

    # -------------------------------------------------------------------
    # Default parameter values
    # -------------------------------------------------------------------

    def test_defaults_promote_true_no_same_dtype_check(self):
        """Default: promote_dtypes=True, require_same_dtype=False.
        Mixed dtypes should be promoted without raising."""
        core = _make_core()
        wrapped = _binary_arraylike(core)
        # Should not raise:
        wrapped(np.array([1], dtype=np.int32), np.array([2.0], dtype=np.float64))

    def test_default_returns_ndarray(self):
        """A default-configured wrapper forwards the core return value."""
        sentinel = np.array([77.0])

        def core(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return sentinel

        core.__name__ = "core"
        core.__doc__ = ""
        wrapped = _binary_arraylike(core)
        assert wrapped([1.0], [2.0]) is sentinel
