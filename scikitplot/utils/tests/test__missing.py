# scikitplot/utils/tests/test__missing.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
# codespell:ignore coo
#
# This module was copied from the scikit-learn project.
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/fixes.py
#
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._missing`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__missing.py -v

Coverage map
------------
is_scalar_nan     NaN detection; ints, None, str, list, pd.NA  → TestIsScalarNan
is_pandas_na      pandas.NA detection; non-NA types; no pandas  → TestIsPandasNa
"""

from __future__ import annotations

import math
import sys
import unittest
import pytest

import numpy as np

from .._missing import is_pandas_na, is_scalar_nan  # noqa: E402


# ===========================================================================
# is_scalar_nan
# ===========================================================================


class TestIsScalarNan(unittest.TestCase):
    """is_scalar_nan must detect NaN correctly for all numeric/non-numeric types."""

    @pytest.mark.parametrize(
        "value, result",
        [
            (float("nan"), True),
            (np.nan, True),
            (float(np.nan), True),
            (np.float32(np.nan), True),
            (np.float64(np.nan), True),
            (0, False),
            (0.0, False),
            (None, False),
            ("", False),
            ("nan", False),
            ([np.nan], False),
            (9867966753463435747313673, False),  # Python int that overflows with C type
        ],
    )
    def test_is_scalar_nan(value, result):
        assert is_scalar_nan(value) is result
        # make sure that we are returning a Python bool
        assert isinstance(is_scalar_nan(value), bool)

    # -- True cases: should be NaN --

    def test_float_nan_is_true(self):
        """float('nan') must return True."""
        self.assertTrue(is_scalar_nan(float("nan")))

    def test_numpy_nan_is_true(self):
        """np.nan must return True."""
        self.assertTrue(is_scalar_nan(np.nan))

    def test_math_nan_is_true(self):
        """math.nan must return True."""
        self.assertTrue(is_scalar_nan(math.nan))

    def test_numpy_float64_nan_is_true(self):
        """np.float64('nan') must return True."""
        self.assertTrue(is_scalar_nan(np.float64("nan")))

    def test_numpy_float32_nan_is_true(self):
        """np.float32('nan') must return True."""
        self.assertTrue(is_scalar_nan(np.float32("nan")))

    # -- False cases: should NOT be NaN --

    def test_none_is_false(self):
        """None must return False (not a number)."""
        self.assertFalse(is_scalar_nan(None))

    def test_empty_string_is_false(self):
        """Empty string must return False."""
        self.assertFalse(is_scalar_nan(""))

    def test_string_nan_is_false(self):
        """The string 'nan' must return False (not a float)."""
        self.assertFalse(is_scalar_nan("nan"))

    def test_zero_float_is_false(self):
        """0.0 is a valid float but not NaN."""
        self.assertFalse(is_scalar_nan(0.0))

    def test_positive_float_is_false(self):
        """1.5 must return False."""
        self.assertFalse(is_scalar_nan(1.5))

    def test_negative_float_is_false(self):
        """-3.14 must return False."""
        self.assertFalse(is_scalar_nan(-3.14))

    def test_integer_zero_is_false(self):
        """int 0 must return False (integers excluded by design)."""
        self.assertFalse(is_scalar_nan(0))

    def test_positive_integer_is_false(self):
        """Positive int must return False."""
        self.assertFalse(is_scalar_nan(42))

    def test_negative_integer_is_false(self):
        """Negative int must return False."""
        self.assertFalse(is_scalar_nan(-1))

    def test_numpy_int64_is_false(self):
        """np.int64 is Integral; must return False."""
        self.assertFalse(is_scalar_nan(np.int64(5)))

    def test_numpy_float64_finite_is_false(self):
        """np.float64(1.0) must return False."""
        self.assertFalse(is_scalar_nan(np.float64(1.0)))

    def test_list_containing_nan_is_false(self):
        """A list [nan] must return False (not a scalar)."""
        self.assertFalse(is_scalar_nan([float("nan")]))

    def test_numpy_array_is_false(self):
        """A numpy array must return False (not a scalar)."""
        self.assertFalse(is_scalar_nan(np.array([float("nan")])))

    def test_bool_true_is_false(self):
        """bool is Integral; True must return False."""
        self.assertFalse(is_scalar_nan(True))

    def test_bool_false_is_false(self):
        """bool is Integral; False must return False."""
        self.assertFalse(is_scalar_nan(False))

    def test_infinity_is_false(self):
        """float('inf') is real but not NaN; must return False."""
        self.assertFalse(is_scalar_nan(float("inf")))

    def test_negative_infinity_is_false(self):
        """float('-inf') is real but not NaN; must return False."""
        self.assertFalse(is_scalar_nan(float("-inf")))

    def test_returns_bool(self):
        """Return type must be bool."""
        result = is_scalar_nan(float("nan"))
        self.assertIsInstance(result, bool)

    def test_complex_nan_real_is_false(self):
        """A complex number with nan real part — complex is not numbers.Real."""
        # complex is not a subclass of numbers.Real, so returns False.
        self.assertFalse(is_scalar_nan(complex(float("nan"), 0)))

    def test_pandas_na_is_false(self):
        """pandas.NA must return False (not a float/Real)."""
        try:
            import pandas as pd  # noqa: PLC0415

            self.assertFalse(is_scalar_nan(pd.NA))
        except ImportError:
            self.skipTest("pandas not installed")


# ===========================================================================
# is_pandas_na
# ===========================================================================


class TestIsPandasNa(unittest.TestCase):
    """is_pandas_na must return True only for pandas.NA, False for everything else."""

    def _pandas_available(self):
        """Return True if pandas is importable, skip if not."""
        try:
            import pandas  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False

    def test_pandas_na_returns_true(self):
        """pandas.NA must return True."""
        if not self._pandas_available():
            self.skipTest("pandas not installed")
        import pandas as pd  # noqa: PLC0415

        self.assertTrue(is_pandas_na(pd.NA))

    def test_float_nan_returns_false(self):
        """float('nan') must return False — it is NOT pandas.NA."""
        self.assertFalse(is_pandas_na(float("nan")))

    def test_numpy_nan_returns_false(self):
        """np.nan must return False."""
        self.assertFalse(is_pandas_na(np.nan))

    def test_none_returns_false(self):
        """None must return False."""
        self.assertFalse(is_pandas_na(None))

    def test_integer_returns_false(self):
        """An integer must return False."""
        self.assertFalse(is_pandas_na(0))

    def test_string_returns_false(self):
        """A string must return False."""
        self.assertFalse(is_pandas_na("NA"))

    def test_pandas_nat_returns_false(self):
        """pandas.NaT is NOT pandas.NA; must return False."""
        if not self._pandas_available():
            self.skipTest("pandas not installed")
        import pandas as pd  # noqa: PLC0415

        self.assertFalse(is_pandas_na(pd.NaT))

    def test_numpy_nan_is_not_pandas_na(self):
        """np.nan and pd.NA are distinct objects; must return False."""
        if not self._pandas_available():
            self.skipTest("pandas not installed")
        self.assertFalse(is_pandas_na(np.nan))

    def test_returns_bool(self):
        """Return value must always be bool."""
        result = is_pandas_na(None)
        self.assertIsInstance(result, bool)

    def test_without_pandas_returns_false(self):
        """If pandas is not importable, the function must return False gracefully.

        We simulate this by temporarily hiding pandas from sys.modules.
        """
        original = sys.modules.pop("pandas", None)
        try:
            # After removal, is_pandas_na should catch ImportError and return False.
            result = is_pandas_na(None)
            self.assertFalse(result)
        finally:
            if original is not None:
                sys.modules["pandas"] = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
