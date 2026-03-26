# scikitplot/_utils/tests/test__convertions.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils._convertions`.

Coverage map
------------
asunicode     bytes→str latin-1, str passthrough, non-str coercion  → TestAsunicode
asbytes       bytes passthrough, str→bytes latin-1, non-str coerce   → TestAsbytes
module        __all__ completeness, callable check                    → TestConvertionsModule

Run standalone::

    python -m unittest scikitplot._utils.tests.test__convertions -v
"""

from __future__ import annotations

import unittest

from .._convertions import asbytes, asunicode


# ===========================================================================
# asunicode
# ===========================================================================


class TestAsunicode(unittest.TestCase):
    """asunicode must decode bytes via latin-1 and pass str through unchanged."""

    # -- bytes -> str --

    def test_ascii_bytes_decoded(self):
        """ASCII bytes must be decoded to the equivalent str."""
        self.assertEqual(asunicode(b"hello"), "hello")

    def test_empty_bytes_gives_empty_str(self):
        """Empty bytes must yield an empty string."""
        self.assertEqual(asunicode(b""), "")

    def test_latin1_bytes_decoded(self):
        """Latin-1 byte 0xe9 must decode to the matching unicode codepoint."""
        self.assertEqual(asunicode(b"\xe9"), "\xe9")

    def test_high_byte_roundtrip(self):
        """Byte 0xff must survive a latin-1 decode without data loss."""
        raw = b"\xff"
        result = asunicode(raw)
        self.assertIsInstance(result, str)
        self.assertEqual(result.encode("latin-1"), raw)

    def test_multi_byte_string_decoded(self):
        """A multi-character byte string must be fully decoded."""
        self.assertEqual(asunicode(b"abc"), "abc")

    # -- str -> str (passthrough) --

    def test_str_input_returned_unchanged(self):
        """A plain str must be returned as-is (no re-encoding)."""
        s = "hello"
        result = asunicode(s)
        self.assertIs(result, s)

    def test_empty_str_passthrough(self):
        """Empty str must return empty str."""
        self.assertEqual(asunicode(""), "")

    def test_unicode_str_passthrough(self):
        """A str with non-ASCII characters must pass through unchanged."""
        s = "cafe\u0301"
        self.assertEqual(asunicode(s), s)

    # -- non-str, non-bytes coercion via str() --

    def test_int_coerced_to_str(self):
        """An integer must be coerced to str via str()."""
        self.assertEqual(asunicode(42), "42")

    def test_float_coerced_to_str(self):
        """A float must be coerced to str via str()."""
        result = asunicode(3.14)
        self.assertIsInstance(result, str)
        self.assertIn("3.14", result)

    def test_none_coerced_to_none_string(self):
        """None must be coerced to the string 'None'."""
        self.assertEqual(asunicode(None), "None")

    def test_return_type_always_str(self):
        """Return type must always be str for any input type."""
        for v in [b"x", "x", 0, 1.0, None, True]:
            with self.subTest(v=v):
                self.assertIsInstance(asunicode(v), str)


# ===========================================================================
# asbytes
# ===========================================================================


class TestAsbytes(unittest.TestCase):
    """asbytes must pass bytes through unchanged and encode str via latin-1."""

    # -- bytes -> bytes (passthrough) --

    def test_bytes_passthrough_identity(self):
        """bytes input must be returned as the same object (identity)."""
        b = b"hello"
        result = asbytes(b)
        self.assertIs(result, b)

    def test_empty_bytes_passthrough(self):
        """Empty bytes must return empty bytes."""
        self.assertEqual(asbytes(b""), b"")

    def test_high_byte_passthrough(self):
        """Bytes containing 0xff must pass through without modification."""
        b = b"\xff\xfe"
        self.assertEqual(asbytes(b), b)

    # -- str -> bytes encoding --

    def test_ascii_str_encoded(self):
        """ASCII str must be encoded to equivalent bytes via latin-1."""
        self.assertEqual(asbytes("hello"), b"hello")

    def test_empty_str_encoded_to_empty_bytes(self):
        """Empty str must produce empty bytes."""
        self.assertEqual(asbytes(""), b"")

    def test_latin1_str_encoded(self):
        """A str with latin-1 character must be encoded correctly."""
        result = asbytes("\xe9")
        self.assertEqual(result, b"\xe9")

    def test_str_roundtrip_via_latin1(self):
        """str -> bytes -> str must be an identity transform over latin-1."""
        original = "abc\xe9"
        self.assertEqual(asunicode(asbytes(original)), original)

    # -- non-str, non-bytes coercion --

    def test_int_coerced_then_encoded(self):
        """An integer must be converted to str first, then encoded to bytes."""
        result = asbytes(42)
        self.assertEqual(result, b"42")

    def test_float_coerced_then_encoded(self):
        """A float must be converted to str first, then encoded to bytes."""
        result = asbytes(3.14)
        self.assertIsInstance(result, bytes)
        self.assertIn(b"3.14", result)

    def test_none_coerced_then_encoded(self):
        """None must be coerced to 'None' and then encoded to bytes."""
        self.assertEqual(asbytes(None), b"None")

    def test_return_type_always_bytes(self):
        """Return type must always be bytes for any input type."""
        for v in [b"x", "x", 0, 1.0, None, True]:
            with self.subTest(v=v):
                self.assertIsInstance(asbytes(v), bytes)


# ===========================================================================
# Module-level invariants
# ===========================================================================


class TestConvertionsModule(unittest.TestCase):
    """Module-level invariants for _convertions."""

    def test_all_contains_asunicode(self):
        """__all__ must include 'asunicode'."""
        from .. import _convertions
        self.assertIn("asunicode", _convertions.__all__)

    def test_all_contains_asbytes(self):
        """__all__ must include 'asbytes'."""
        from .. import _convertions
        self.assertIn("asbytes", _convertions.__all__)

    def test_all_has_exactly_two_entries(self):
        """__all__ must declare exactly two public names."""
        from .. import _convertions
        self.assertEqual(len(_convertions.__all__), 2)

    def test_both_symbols_are_callable(self):
        """Both public symbols must be callable."""
        self.assertTrue(callable(asbytes))
        self.assertTrue(callable(asunicode))


if __name__ == "__main__":
    unittest.main(verbosity=2)
