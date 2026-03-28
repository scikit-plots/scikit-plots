# scikitplot/_utils/tests/test_string_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.string_utils`.

Coverage map
------------
strip_prefix                    prefix present, absent, empty         → TestStripPrefix
strip_suffix                    suffix present, absent, empty         → TestStripSuffix
is_string_type                  str, bytes, int                       → TestIsStringType
generate_feature_name           str passthrough, int→feature_N        → TestGenerateFeatureName
truncate_str_from_middle        short, exact, long, assert guard      → TestTruncateStrFromMiddle
_create_table                   single row, multi-row, headers        → TestCreateTable
_backtick_quote                 plain, already quoted                 → TestBacktickQuote
quote                           delegates to shlex.quote on POSIX     → TestQuote

Run standalone::

    python -m unittest scikitplot._utils.tests.test_string_utils -v
"""

from __future__ import annotations

import shlex
import sys
import unittest

from ..string_utils import (
    _backtick_quote,
    _create_table,
    generate_feature_name_if_not_string,
    is_string_type,
    quote,
    strip_prefix,
    strip_suffix,
    truncate_str_from_middle,
)


# ===========================================================================
# strip_prefix
# ===========================================================================


class TestStripPrefix(unittest.TestCase):
    """strip_prefix must remove the prefix when present and leave string alone otherwise."""

    def test_prefix_present(self):
        """When the prefix is present it must be stripped."""
        self.assertEqual(strip_prefix("hello world", "hello "), "world")

    def test_prefix_absent(self):
        """When the prefix is absent the original string must be returned."""
        self.assertEqual(strip_prefix("world", "hello "), "world")

    def test_full_string_is_prefix(self):
        """When the entire string is the prefix, an empty string is returned."""
        self.assertEqual(strip_prefix("abc", "abc"), "")

    def test_empty_prefix(self):
        """An empty prefix must return the original string unchanged."""
        self.assertEqual(strip_prefix("hello", ""), "hello")

    def test_empty_string_empty_prefix(self):
        """Empty string with empty prefix must return empty string."""
        self.assertEqual(strip_prefix("", ""), "")

    def test_prefix_longer_than_string(self):
        """A prefix longer than the string must leave the string unchanged."""
        self.assertEqual(strip_prefix("hi", "hello"), "hi")

    def test_case_sensitive(self):
        """strip_prefix must be case-sensitive."""
        self.assertEqual(strip_prefix("Hello", "hello"), "Hello")

    def test_partial_prefix_not_stripped(self):
        """Only a full prefix match is stripped; partial match is ignored."""
        self.assertEqual(strip_prefix("foobar", "foob"), "ar")


# ===========================================================================
# strip_suffix
# ===========================================================================


class TestStripSuffix(unittest.TestCase):
    """strip_suffix must remove the suffix when present and leave string alone otherwise."""

    def test_suffix_present(self):
        """When the suffix is present it must be stripped."""
        self.assertEqual(strip_suffix("hello.py", ".py"), "hello")

    def test_suffix_absent(self):
        """When the suffix is absent the original string must be returned."""
        self.assertEqual(strip_suffix("hello.py", ".txt"), "hello.py")

    def test_full_string_is_suffix(self):
        """When the entire string is the suffix, empty string is returned."""
        self.assertEqual(strip_suffix("abc", "abc"), "")

    def test_empty_suffix_returns_original(self):
        """An empty suffix must return the original string unchanged."""
        self.assertEqual(strip_suffix("hello", ""), "hello")

    def test_empty_string_empty_suffix(self):
        """Empty string with empty suffix must return empty string unchanged."""
        self.assertEqual(strip_suffix("", ""), "")

    def test_suffix_longer_than_string(self):
        """A suffix longer than the string must leave the string unchanged."""
        self.assertEqual(strip_suffix("hi", "hello"), "hi")

    def test_case_sensitive(self):
        """strip_suffix must be case-sensitive."""
        self.assertEqual(strip_suffix("Hello.PY", ".py"), "Hello.PY")


# ===========================================================================
# is_string_type
# ===========================================================================


class TestIsStringType(unittest.TestCase):
    """is_string_type must return True only for str instances."""

    def test_str_is_true(self):
        """A plain str must return True."""
        self.assertTrue(is_string_type("hello"))

    def test_empty_str_is_true(self):
        """An empty str must return True."""
        self.assertTrue(is_string_type(""))

    def test_bytes_is_false(self):
        """bytes must return False."""
        self.assertFalse(is_string_type(b"hello"))

    def test_int_is_false(self):
        """int must return False."""
        self.assertFalse(is_string_type(42))

    def test_none_is_false(self):
        """None must return False."""
        self.assertFalse(is_string_type(None))

    def test_list_is_false(self):
        """list must return False."""
        self.assertFalse(is_string_type([]))

    def test_unicode_str_is_true(self):
        """A unicode str must return True."""
        self.assertTrue(is_string_type("café"))


# ===========================================================================
# generate_feature_name_if_not_string
# ===========================================================================


class TestGenerateFeatureName(unittest.TestCase):
    """generate_feature_name_if_not_string must pass str through and format others."""

    def test_str_returned_unchanged(self):
        """A str value must be returned as-is."""
        s = "my_feature"
        result = generate_feature_name_if_not_string(s)
        self.assertIs(result, s)

    def test_int_generates_feature_name(self):
        """An integer N must produce 'feature_N'."""
        self.assertEqual(generate_feature_name_if_not_string(0), "feature_0")
        self.assertEqual(generate_feature_name_if_not_string(7), "feature_7")

    def test_float_generates_feature_name(self):
        """A float must produce a feature_<float> string."""
        result = generate_feature_name_if_not_string(1.5)
        self.assertTrue(result.startswith("feature_"))

    def test_return_type_is_str(self):
        """Return type must always be str."""
        for v in ["x", 0, 3.14, None]:
            with self.subTest(v=v):
                self.assertIsInstance(generate_feature_name_if_not_string(v), str)


# ===========================================================================
# truncate_str_from_middle
# ===========================================================================


class TestTruncateStrFromMiddle(unittest.TestCase):
    """truncate_str_from_middle must trim long strings from the middle."""

    def test_short_string_returned_unchanged(self):
        """Strings shorter than max_length must not be modified."""
        s = "hello"
        self.assertEqual(truncate_str_from_middle(s, 20), s)

    def test_exact_length_returned_unchanged(self):
        """A string exactly at max_length must not be modified.

        Note: the implementation asserts max_length > 5 (strictly), so
        the minimum testable exact-match length is 6 characters.
        """
        s = "helloo"  # length 6 == min valid max_length
        self.assertEqual(truncate_str_from_middle(s, 6), s)

    def test_long_string_truncated_with_ellipsis(self):
        """A string longer than max_length must be truncated with '...'."""
        s = "a" * 20
        result = truncate_str_from_middle(s, 10)
        self.assertEqual(len(result), 10)
        self.assertIn("...", result)

    def test_result_length_equals_max_length(self):
        """Truncated result must be exactly max_length characters."""
        s = "abcdefghij" * 5
        max_len = 15
        result = truncate_str_from_middle(s, max_len)
        self.assertEqual(len(result), max_len)

    def test_prefix_and_suffix_preserved(self):
        """Characters from the start and end of the string must be kept."""
        s = "START" + "x" * 50 + "END"
        result = truncate_str_from_middle(s, 15)
        self.assertTrue(result.startswith("S"))
        self.assertTrue(result.endswith("D"))

    def test_max_length_too_small_raises(self):
        """max_length <= 5 must raise (assert guard in implementation)."""
        with self.assertRaises((AssertionError, ValueError)):
            truncate_str_from_middle("hello world", 5)

    def test_min_valid_max_length(self):
        """max_length=6 (just above the assert threshold) must work."""
        result = truncate_str_from_middle("hello world this is long", 6)
        self.assertEqual(len(result), 6)
        self.assertIn("...", result)


# ===========================================================================
# _create_table
# ===========================================================================


class TestCreateTable(unittest.TestCase):
    """_create_table must produce a plain-text table with headers and separator."""

    def test_single_row_three_columns(self):
        """A single row with three columns must produce valid table text."""
        result = _create_table([["a", "b", "c"]], ["x", "y", "z"])
        lines = result.splitlines()
        # header + separator + data
        self.assertGreaterEqual(len(lines), 3)

    def test_headers_in_output(self):
        """All header names must appear in the output."""
        result = _create_table([["val1", "val2"]], ["header1", "header2"])
        self.assertIn("header1", result)
        self.assertIn("header2", result)

    def test_data_values_in_output(self):
        """All cell values must appear in the output."""
        result = _create_table([["alpha", "beta"]], ["A", "B"])
        self.assertIn("alpha", result)
        self.assertIn("beta", result)

    def test_separator_line_present(self):
        """A separator line made of dashes must be present."""
        result = _create_table([["a", "b"]], ["X", "Y"])
        self.assertIn("----", result)

    def test_multi_row_output(self):
        """Multiple rows must all appear in the output."""
        rows = [["r1c1", "r1c2"], ["r2c1", "r2c2"]]
        result = _create_table(rows, ["H1", "H2"])
        for row in rows:
            for cell in row:
                self.assertIn(cell, result)

    def test_returns_str(self):
        """Return type must be str."""
        result = _create_table([["x"]], ["H"])
        self.assertIsInstance(result, str)

    def test_min_column_width_respected(self):
        """All columns must be at least min_column_width wide (default 4)."""
        result = _create_table([["a"]], ["H"])
        lines = result.splitlines()
        # Separator line must be at least 4 chars wide
        sep_line = lines[1]
        self.assertGreaterEqual(len(sep_line.strip()), 4)


# ===========================================================================
# _backtick_quote
# ===========================================================================


class TestBacktickQuote(unittest.TestCase):
    """_backtick_quote must wrap plain strings in backticks."""

    def test_plain_string_gets_backticks(self):
        """A plain string must get leading and trailing backticks."""
        result = _backtick_quote("hello")
        self.assertEqual(result, "`hello`")

    def test_already_quoted_unchanged(self):
        """A string already in backticks must not get double-quoted."""
        already = "`hello`"
        result = _backtick_quote(already)
        self.assertEqual(result, already)

    def test_empty_string_gets_backticks(self):
        """An empty string must get backtick wrappers."""
        result = _backtick_quote("")
        self.assertEqual(result, "``")

    def test_backtick_in_middle_not_treated_as_quoted(self):
        """A string with backtick only in middle must still be wrapped."""
        result = _backtick_quote("he`llo")
        self.assertEqual(result, "`he`llo`")


# ===========================================================================
# quote (POSIX / Windows dispatch)
# ===========================================================================


class TestQuote(unittest.TestCase):
    """quote must produce shell-safe strings on the current platform."""

    def test_simple_string_is_safe(self):
        """A simple alphanumeric string must be returned safely quotable."""
        result = quote("hello")
        self.assertIn("hello", result)

    def test_string_with_spaces_is_quoted(self):
        """A string containing spaces must be wrapped to prevent word-splitting."""
        result = quote("hello world")
        # On POSIX: shlex.quote gives "'hello world'"
        self.assertIn("hello world", result)
        # The result must not split on the space when re-parsed by shlex
        if sys.platform != "win32":
            parsed = shlex.split(result)
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0], "hello world")

    def test_returns_str(self):
        """Return type must be str."""
        result = quote("test")
        self.assertIsInstance(result, str)

    def test_safe_string_unchanged_on_posix(self):
        """On POSIX, a shell-safe token must be returned as-is."""
        if sys.platform != "win32":
            result = quote("safe_token")
            self.assertEqual(result, "safe_token")

    def test_special_chars_quoted_on_posix(self):
        """On POSIX, a string with shell-special characters must be quoted."""
        if sys.platform != "win32":
            result = quote("hello; rm -rf /")
            # Should not contain unquoted semicolon
            self.assertNotEqual(result, "hello; rm -rf /")


if __name__ == "__main__":
    unittest.main(verbosity=2)
