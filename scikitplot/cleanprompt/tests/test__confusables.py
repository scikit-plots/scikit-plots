"""Tests for :mod:`scikitplot.cleanprompt._confusables`, the look-alike table (CP-114)."""

from __future__ import annotations

import re
import unicodedata

import pytest

from .._confusables import CONFUSABLES, UNICODE_VERSION, UNICODEDATA_VERSION

#: A dotted three-part version, as Unicode writes it.
_VERSION = re.compile(r"\d+\.\d+\.\d+")

#: BENGALI DIGIT FOUR, shaped like 8.
_BENGALI_FOUR = 0x09EA

_ALNUM = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


class TestTable:
    def test_records_its_unicode_version(self):
        assert _VERSION.fullmatch(UNICODE_VERSION)

    def test_keys_are_non_ascii_letters_or_digits(self):
        # Categories come from this Python's unicodedata; a code point it does
        # not know yet (Python 3.8 has Unicode 12.1) is checked where it is.
        for point in CONFUSABLES:
            char = chr(point)
            assert not char.isascii(), hex(point)
            category = unicodedata.category(char)
            if category == "Cn":
                continue
            assert category.startswith("L") or category == "Nd", hex(point)

    def test_values_are_one_ascii_letter_or_digit(self):
        assert set(CONFUSABLES.values()) <= _ALNUM

    def test_case_is_kept(self):
        for point, char in CONFUSABLES.items():
            category = unicodedata.category(chr(point))
            if category == "Lu":
                assert char.isupper(), hex(point)
            elif category == "Ll":
                assert char.islower(), hex(point)

    def test_records_the_unicodedata_it_was_generated_with(self):
        assert _VERSION.fullmatch(UNICODEDATA_VERSION)

    def test_a_digit_keeps_its_value(self):
        for point, char in CONFUSABLES.items():
            if unicodedata.category(chr(point)) == "Nd":
                assert str(unicodedata.digit(chr(point))) == char, hex(point)

    @pytest.mark.parametrize(
        ("point", "char"),
        [
            (0x0410, "A"),
            (0x0430, "a"),
            (0x0406, "I"),
            (0x0456, "i"),
            (0x039F, "O"),
            (0x03BF, "o"),
        ],
    )
    def test_known_look_alikes(self, point, char):
        assert CONFUSABLES[point] == char

    def test_a_digit_that_only_looks_like_another_is_left_out(self):
        assert _BENGALI_FOUR not in CONFUSABLES
