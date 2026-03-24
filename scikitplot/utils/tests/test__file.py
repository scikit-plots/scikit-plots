# scikitplot/utils/tests/test__file.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._file`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__file.py -v

Coverage map
------------
humansize        Zero, positive, negative, float, non-numeric,
                 custom suffixes, all magnitude levels       → TestHumansize
humansize_vector Scalar, list, numpy array, non-iterable,
                 string/bytes passthrough, pandas Series     → TestHumansizeVector
SUFFIXES         Module constant is correct default list     → TestConstants
"""

from __future__ import annotations

import unittest

import numpy as np

# --- path bootstrap (commented out — use pytest from the package root) -----
# import pathlib, sys
# _HERE = pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))
# --------------------------------------------------------------------------

from .._file import SUFFIXES, humansize, humansize_vector  # noqa: E402


# ===========================================================================
# Module constants
# ===========================================================================


class TestConstants(unittest.TestCase):
    """SUFFIXES must be the canonical byte-unit list."""

    def test_suffixes_is_list(self):
        self.assertIsInstance(SUFFIXES, list)

    def test_suffixes_starts_with_bytes(self):
        self.assertEqual(SUFFIXES[0], "B")

    def test_suffixes_contains_standard_units(self):
        for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
            self.assertIn(unit, SUFFIXES)

    def test_suffixes_length(self):
        self.assertEqual(len(SUFFIXES), 6)


# ===========================================================================
# humansize
# ===========================================================================


class TestHumansize(unittest.TestCase):
    """humansize must convert byte counts to human-readable strings."""

    # -- Exact powers of 1024 --

    def test_zero_bytes(self):
        """0 bytes must display as '0 B'."""
        self.assertEqual(humansize(0), "0 B")

    def test_one_byte(self):
        self.assertEqual(humansize(1), "1 B")

    def test_1023_bytes(self):
        """Just below 1 KB must remain in bytes."""
        self.assertEqual(humansize(1023), "1023 B")

    def test_exactly_1_kb(self):
        """1024 bytes must display as '1 KB'."""
        self.assertEqual(humansize(1024), "1 KB")

    def test_exactly_1_mb(self):
        """1024 ** 2 bytes must display as '1 MB'."""
        self.assertEqual(humansize(1024**2), "1 MB")

    def test_exactly_1_gb(self):
        """1024 ** 3 bytes must display as '1 GB'."""
        self.assertEqual(humansize(1024**3), "1 GB")

    def test_exactly_1_tb(self):
        """1024 ** 4 bytes must display as '1 TB'."""
        self.assertEqual(humansize(1024**4), "1 TB")

    def test_exactly_1_pb(self):
        """1024 ** 5 bytes must display as '1 PB'."""
        self.assertEqual(humansize(1024**5), "1 PB")

    # -- Fractional values --

    def test_1536_bytes_is_1_5_kb(self):
        """1536 = 1.5 * 1024 → '1.5 KB'."""
        self.assertEqual(humansize(1536), "1.5 KB")

    def test_2048_bytes_is_2_kb(self):
        self.assertEqual(humansize(2048), "2 KB")

    def test_fractional_mb(self):
        """10_000_000 bytes ≈ 9.54 MB."""
        result = humansize(10_000_000)
        self.assertIn("MB", result)
        self.assertTrue(result.startswith("9.5"))

    # -- Negative values --

    def test_negative_1024(self):
        """Negative values must be prefixed with '-'."""
        self.assertEqual(humansize(-1024), "-1 KB")

    def test_negative_2048(self):
        self.assertEqual(humansize(-2048), "-2 KB")

    def test_negative_bytes(self):
        """Small negative value must stay in bytes."""
        self.assertIn("B", humansize(-100))
        self.assertTrue(humansize(-100).startswith("-"))

    # -- Float input --

    def test_float_input_1024(self):
        """Float 1024.0 must behave the same as int 1024."""
        self.assertEqual(humansize(1024.0), "1 KB")

    def test_float_input_fractional(self):
        """Float 1536.0 must give '1.5 KB'."""
        self.assertEqual(humansize(1536.0), "1.5 KB")

    # -- Non-numeric input (graceful fallback) --

    def test_string_returns_str(self):
        """A non-numeric string must be returned as-is via str()."""
        result = humansize("not_a_number")
        self.assertIsInstance(result, str)

    def test_none_returns_str(self):
        """None cannot be cast to float; must return 'None'."""
        result = humansize(None)
        self.assertIsInstance(result, str)

    # -- Return type --

    def test_returns_string(self):
        self.assertIsInstance(humansize(1024), str)

    def test_no_trailing_zero(self):
        """Trailing zeros must be stripped (e.g. '1 KB', not '1.00 KB')."""
        result = humansize(1024)
        self.assertNotIn(".00", result)
        self.assertNotIn(".0 ", result)

    # -- Custom suffixes --

    def test_custom_suffixes_two_levels(self):
        """Custom two-level suffix list must be used."""
        result = humansize(1024, suffixes=["Byte", "Kilobyte"])
        self.assertEqual(result, "1 Kilobyte")

    def test_custom_suffixes_single_level(self):
        """Single-element suffix list: all values remain at index 0."""
        result = humansize(1024**3, suffixes=["X"])
        self.assertEqual(result, "1073741824 X")

    # -- Overflow stays at PB --

    def test_huge_value_stays_at_pb(self):
        """Values beyond PB must clamp at the last suffix."""
        result = humansize(1024**6)
        self.assertIn("PB", result)

    # -- numpy integer / float types --

    def test_numpy_int64(self):
        """np.int64(2048) must give '2 KB'."""
        self.assertEqual(humansize(np.int64(2048)), "2 KB")

    def test_numpy_float32(self):
        """np.float32(1024) must give '1 KB'."""
        self.assertEqual(humansize(np.float32(1024)), "1 KB")


# ===========================================================================
# humansize_vector
# ===========================================================================


class TestHumansizeVector(unittest.TestCase):
    """humansize_vector must apply humansize element-wise."""

    # -- List input --

    def test_list_two_elements(self):
        out = humansize_vector([1024, 2048])
        self.assertEqual(list(out), ["1 KB", "2 KB"])

    def test_list_mixed_magnitudes(self):
        out = humansize_vector([1024, 1024**2])
        self.assertEqual(list(out), ["1 KB", "1 MB"])

    # -- NumPy array input --

    def test_numpy_array_integers(self):
        arr = np.array([1024, 2048])
        out = humansize_vector(arr)
        self.assertEqual(list(out), ["1 KB", "2 KB"])

    def test_numpy_array_large(self):
        arr = np.array([1024, 1024**2])
        out = humansize_vector(arr)
        self.assertEqual(out[1], "1 MB")

    def test_numpy_dtype_object(self):
        """Object-dtype array must still vectorize correctly."""
        arr = np.array([1024, 2048], dtype=object)
        out = humansize_vector(arr)
        self.assertEqual(list(out), ["1 KB", "2 KB"])

    def test_returns_numpy_array(self):
        out = humansize_vector([1024, 2048])
        self.assertIsInstance(out, np.ndarray)

    # -- Scalar (non-iterable) input --

    def test_scalar_int(self):
        """A plain int is not iterable; scalar path must return a string."""
        result = humansize_vector(1024)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "1 KB")

    # -- String / bytes (iterable but excluded by isinstance check) --

    def test_string_input_treated_as_scalar(self):
        """str is Iterable but excluded by the isinstance guard."""
        result = humansize_vector("not_a_number")
        self.assertIsInstance(result, str)

    def test_bytes_input_treated_as_scalar(self):
        """bytes is Iterable but excluded by the isinstance guard."""
        result = humansize_vector(b"data")
        self.assertIsInstance(result, str)

    # -- Empty list --

    def test_empty_list(self):
        out = humansize_vector([])
        self.assertEqual(len(out), 0)

    # -- Custom suffixes --

    def test_custom_suffixes_vector(self):
        """Custom suffixes must be passed through to each element."""
        out = humansize_vector([1024], suffixes=["Byte", "Kilo"])
        self.assertIn("Kilo", out[0])

    # -- Negative values in list --

    def test_negative_values_in_list(self):
        out = humansize_vector([-1024, -2048])
        for item in out:
            self.assertTrue(item.startswith("-"))

    # -- pandas Series (if available) --

    def test_pandas_series(self):
        """pandas Series must be accepted and return an ndarray."""
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError:
            self.skipTest("pandas not installed")
        s = pd.Series([1024, 2048])
        out = humansize_vector(s)
        self.assertEqual(list(out), ["1 KB", "2 KB"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
