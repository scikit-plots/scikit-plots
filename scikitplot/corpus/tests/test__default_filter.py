# scikitplot/corpus/tests/test__default_filter.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DefaultFilter's word counting.

Chinese, Japanese, Thai, Lao, Khmer, Myanmar and Tibetan are written without
spaces between words. Counted as whitespace tokens, a whole document in them
is one word, and the default ``min_words=3`` dropped it.
"""

from __future__ import annotations

import pytest

from .. import _base
from .._base import DefaultFilter
from .._schema import CorpusDocument

# ===========================================================================
# DefaultFilter: word counting in scripts with and without word spaces
# ===========================================================================

_ZH = "\u8a18\u61b6\u8207\u56de\u61b6\u662f\u5fc3\u667a\u7684\u6838\u5fc3"  # 11 Han
_JA = "\u8a18\u61b6\u3068\u60f3\u8d77\u306f\u5fc3\u306e\u4e2d\u5fc3\u3067\u3059"
_TH = "\u0e04\u0e27\u0e32\u0e21\u0e17\u0e23\u0e07\u0e08\u0e33\u0e41\u0e25\u0e30"


def _doc(text: str) -> CorpusDocument:
    return CorpusDocument.create("f.txt", 0, text)


class TestCountWords:
    @pytest.mark.parametrize(
        "text",
        [
            "Memory and recollection.",
            "  a  b\tc\nd  ",
            "caf\u00e9 na\u00efve r\u00e9sum\u00e9 -- 42 ... !",
            "\u041f\u0430\u043c\u044f\u0442\u044c \u0438 \u0432\u043e\u0441\u043f\u043e\u043c\u0438\u043d\u0430\u043d\u0438\u0435",
            "\ud55c\uad6d\uc5b4 \ubb38\uc7a5",  # Korean separates words with spaces
            "",
        ],
    )
    def test_spaced_scripts_count_as_str_split(self, text: str) -> None:
        assert _base._count_words(text) == len(text.split())

    @pytest.mark.parametrize(("text", "expected"), [(_ZH, 11), (_JA, 12), (_TH, 12)])
    def test_each_character_of_an_unspaced_script_counts(
        self, text: str, expected: int
    ) -> None:
        assert _base._count_words(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("GPT\u6a21\u578b v2", 4),  # GPT, two Han, v2
            ("\u6a21\u578bGPT", 3),
            ("\u6a21GPT\u578b", 3),
            ("\u8a18\u61b6\u3002", 2),  # the ideographic full stop is not a word
            ("\u300c\u8a18\u61b6\u300d", 2),  # nor are corner brackets
            ("\u8a18-\u61b6", 2),  # a run of punctuation alone is not a word
            ("2026\u5e74", 2),
        ],
    )
    def test_mixed_tokens_count_runs_with_letters_or_digits(
        self, text: str, expected: int
    ) -> None:
        assert _base._count_words(text) == expected

    @pytest.mark.parametrize(
        "char",
        ["\u3005", "\u30fc", "\uff76", "\U00020000", "\u0e01", "\u17a0", "\u1000"],
    )
    def test_iteration_marks_extensions_and_southeast_asian_scripts(
        self, char: str
    ) -> None:
        assert _base._is_unspaced(char)

    @pytest.mark.parametrize("char", ["a", "\u00e9", "\u3001", "\u3002", "\uac00", " "])
    def test_other_characters_are_not_unspaced(self, char: str) -> None:
        assert not _base._is_unspaced(char)


class TestDefaultFilterWordCount:
    def test_script_is_the_default_and_keeps_chinese_japanese_thai(self) -> None:
        f = DefaultFilter()
        assert f.word_count == "script"
        for text in (_ZH, _JA, _TH):
            assert f.include(_doc(text)), text

    def test_whitespace_drops_text_without_word_spaces(self) -> None:
        f = DefaultFilter(word_count="whitespace")
        assert not f.include(_doc(_ZH))
        assert f.include(_doc("Memory and recollection matter."))

    @pytest.mark.parametrize(
        "text",
        [
            "Hello world test.",
            "p. 56, 57.",
            "one two",
            "one two three",
            "a b c",
            "12 34 56 78",
            "\u2014 \u2014 \u2014 \u2014",
        ],
    )
    def test_spaced_text_gets_the_same_verdict_as_whitespace(self, text: str) -> None:
        assert DefaultFilter().include(_doc(text)) == DefaultFilter(
            word_count="whitespace"
        ).include(_doc(text))

    def test_min_chars_still_applies_to_short_unspaced_text(self) -> None:
        assert not DefaultFilter().include(_doc("\u8a18\u61b6\u8207"))  # 3 chars < 10
        assert DefaultFilter(min_chars=3).include(_doc("\u8a18\u61b6\u8207"))

    def test_a_callable_counts_words(self) -> None:
        seen = []

        def count(text: str) -> int:
            seen.append(text)
            return 5

        f = DefaultFilter(word_count=count, min_words=5)
        assert f.include(_doc("  just two words here  "))
        assert seen == ["just two words here"]
        assert not DefaultFilter(word_count=lambda text: 4, min_words=5).include(
            _doc("several words in this text")
        )

    @pytest.mark.parametrize("bad", [-1, 2.0, True, "3", None])
    def test_a_callable_must_return_a_non_negative_int(self, bad) -> None:
        f = DefaultFilter(word_count=lambda text: bad)
        with pytest.raises(TypeError, match="must return a non-negative int"):
            f.include(_doc("several words in this text"))

    def test_min_words_zero_does_not_count(self) -> None:
        def count(text: str) -> int:
            raise AssertionError("not called")

        assert DefaultFilter(min_words=0, word_count=count).include(
            _doc("ok words enough here")
        )

    @pytest.mark.parametrize("bad", ["spaces", "", None, 3])
    def test_an_unknown_word_count_is_a_value_error(self, bad) -> None:
        with pytest.raises(ValueError, match="word_count must be one of"):
            DefaultFilter(word_count=bad)

    def test_repr_names_the_policy(self) -> None:
        assert repr(DefaultFilter()) == (
            "DefaultFilter(min_words=3, min_chars=10, word_count='script')"
        )
