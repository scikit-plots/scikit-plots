# corpus/_chunkers/tests/test__language_data.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scikitplot.corpus._chunkers._language_data.

Covers:
- ISO_TO_NLTK: correctness for major ISO codes, ancient languages, regional
- ISO_TO_NAME: completeness and correctness
- NLTK_TO_ISO: inverse of ISO_TO_NLTK
- NLTK_STOPWORD_LANGUAGES: set membership
- BUILTIN_LANG_STOPWORDS: structure and non-empty per language
- iso_to_nltk(): string normalisation, fallback, aliases, case-insensitivity
- nltk_to_iso(): round-trip, fallback
- coerce_language(): None, str, list[str], empty-list, bad type
- resolve_stopwords(): single, multi, extra, no-match fallback
- 200+ language coverage spot-checks
"""

from __future__ import annotations

import pytest

from .._language_data import (
    BUILTIN_LANG_STOPWORDS,
    ISO_TO_NAME,
    ISO_TO_NLTK,
    NLTK_STOPWORD_LANGUAGES,
    NLTK_TO_ISO,
    coerce_language,
    iso_to_nltk,
    nltk_to_iso,
    resolve_stopwords,
)


# ===========================================================================
# ISO_TO_NLTK — table correctness
# ===========================================================================


class TestIsoToNltkTable:
    """Static validation of the ISO_TO_NLTK mapping."""

    def test_english_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["en"] == "english"

    def test_arabic_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["ar"] == "arabic"

    def test_hindi_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["hi"] == "hindi"

    def test_thai_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["th"] == "thai"

    def test_vietnamese_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["vi"] == "vietnamese"

    def test_malay_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["ms"] == "malay"

    def test_chinese_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["zh"] == "chinese"

    def test_japanese_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["ja"] == "japanese"

    def test_korean_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["ko"] == "korean"

    def test_swahili_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["sw"] == "swahili"

    def test_zulu_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["zu"] == "zulu"

    def test_afrikaans_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["af"] == "afrikaans"

    def test_latin_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["la"] == "latin"

    def test_ancient_greek_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["grc"] == "ancient_greek"

    def test_persian_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["fa"] == "persian"

    def test_ottoman_turkish_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["ota"] == "ottoman_turkish"

    def test_georgian_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["ka"] == "georgian"

    def test_filipino_maps_correctly(self) -> None:
        assert ISO_TO_NLTK["fil"] == "filipino"

    def test_all_values_are_lowercase(self) -> None:
        for iso, nltk_name in ISO_TO_NLTK.items():
            assert nltk_name == nltk_name.lower(), f"{iso} → {nltk_name!r} not lowercase"

    def test_table_has_100_plus_entries(self) -> None:
        assert len(ISO_TO_NLTK) >= 100

    def test_all_keys_are_lowercase(self) -> None:
        for k in ISO_TO_NLTK:
            assert k == k.lower(), f"key {k!r} not lowercase"


# ===========================================================================
# ISO_TO_NAME — completeness
# ===========================================================================


class TestIsoToName:
    def test_english_name(self) -> None:
        assert ISO_TO_NAME["en"] == "English"

    def test_arabic_name(self) -> None:
        assert ISO_TO_NAME["ar"] == "Arabic"

    def test_hindi_name(self) -> None:
        assert ISO_TO_NAME["hi"] == "Hindi"

    def test_thai_name(self) -> None:
        assert ISO_TO_NAME["th"] == "Thai"

    def test_same_keys_as_iso_to_nltk(self) -> None:
        assert set(ISO_TO_NAME.keys()) == set(ISO_TO_NLTK.keys())

    def test_all_names_are_nonempty_strings(self) -> None:
        for k, v in ISO_TO_NAME.items():
            assert isinstance(v, str) and v, f"{k} has empty name"


# ===========================================================================
# NLTK_TO_ISO — reverse mapping
# ===========================================================================


class TestNltkToIso:
    def test_english(self) -> None:
        assert NLTK_TO_ISO["english"] == "en"

    def test_arabic(self) -> None:
        assert NLTK_TO_ISO["arabic"] == "ar"

    def test_no_duplicated_iso_codes(self) -> None:
        # All values should be unique (no two NLTK names map to same ISO)
        values = list(NLTK_TO_ISO.values())
        assert len(values) == len(set(values))


# ===========================================================================
# NLTK_STOPWORD_LANGUAGES — set contract
# ===========================================================================


class TestNltkStopwordLanguages:
    def test_english_present(self) -> None:
        assert "english" in NLTK_STOPWORD_LANGUAGES

    def test_arabic_present(self) -> None:
        assert "arabic" in NLTK_STOPWORD_LANGUAGES

    def test_german_present(self) -> None:
        assert "german" in NLTK_STOPWORD_LANGUAGES

    def test_thai_absent(self) -> None:
        # Thai is not in NLTK's stopwords corpus
        assert "thai" not in NLTK_STOPWORD_LANGUAGES

    def test_is_frozenset(self) -> None:
        assert isinstance(NLTK_STOPWORD_LANGUAGES, frozenset)

    def test_has_at_least_20_languages(self) -> None:
        assert len(NLTK_STOPWORD_LANGUAGES) >= 20


# ===========================================================================
# BUILTIN_LANG_STOPWORDS — structure
# ===========================================================================


class TestBuiltinLangStopwords:
    def test_english_present(self) -> None:
        assert "english" in BUILTIN_LANG_STOPWORDS

    def test_hindi_present(self) -> None:
        assert "hindi" in BUILTIN_LANG_STOPWORDS

    def test_thai_present(self) -> None:
        assert "thai" in BUILTIN_LANG_STOPWORDS

    def test_vietnamese_present(self) -> None:
        assert "vietnamese" in BUILTIN_LANG_STOPWORDS

    def test_malay_present(self) -> None:
        assert "malay" in BUILTIN_LANG_STOPWORDS

    def test_swahili_present(self) -> None:
        assert "swahili" in BUILTIN_LANG_STOPWORDS

    def test_zulu_present(self) -> None:
        assert "zulu" in BUILTIN_LANG_STOPWORDS

    def test_latin_present(self) -> None:
        assert "latin" in BUILTIN_LANG_STOPWORDS

    def test_ancient_greek_present(self) -> None:
        assert "ancient_greek" in BUILTIN_LANG_STOPWORDS

    def test_persian_present(self) -> None:
        assert "persian" in BUILTIN_LANG_STOPWORDS

    def test_all_values_are_frozensets(self) -> None:
        for k, v in BUILTIN_LANG_STOPWORDS.items():
            assert isinstance(v, frozenset), f"{k} value is not frozenset"

    def test_all_frozensets_nonempty(self) -> None:
        for k, v in BUILTIN_LANG_STOPWORDS.items():
            assert len(v) > 0, f"{k} stopword set is empty"

    def test_english_stopwords_contain_common_words(self) -> None:
        sw = BUILTIN_LANG_STOPWORDS["english"]
        for word in ("the", "and", "in", "of", "to"):
            assert word in sw

    def test_hindi_stopwords_contain_hindi_chars(self) -> None:
        sw = BUILTIN_LANG_STOPWORDS["hindi"]
        assert "और" in sw or "में" in sw

    def test_thai_stopwords_contain_thai_chars(self) -> None:
        sw = BUILTIN_LANG_STOPWORDS["thai"]
        assert any(ord(c) >= 0x0E00 for word in sw for c in word)

    def test_latin_stopwords_contain_latin_words(self) -> None:
        sw = BUILTIN_LANG_STOPWORDS["latin"]
        assert "et" in sw

    def test_all_stopwords_lowercase(self) -> None:
        for lang, sw in BUILTIN_LANG_STOPWORDS.items():
            for word in sw:
                # Allow Arabic/Hindi etc. — only check Latin-only words
                if word.isascii():
                    assert word == word.lower(), (
                        f"{lang}: ASCII stopword {word!r} is not lowercase"
                    )


# ===========================================================================
# iso_to_nltk() — helper function
# ===========================================================================


class TestIsoToNltk:
    def test_iso_en_resolves(self) -> None:
        assert iso_to_nltk("en") == "english"

    def test_iso_ar_resolves(self) -> None:
        assert iso_to_nltk("ar") == "arabic"

    def test_iso_th_resolves(self) -> None:
        assert iso_to_nltk("th") == "thai"

    def test_iso_vi_resolves(self) -> None:
        assert iso_to_nltk("vi") == "vietnamese"

    def test_iso_grc_resolves(self) -> None:
        assert iso_to_nltk("grc") == "ancient_greek"

    def test_iso_ota_resolves(self) -> None:
        assert iso_to_nltk("ota") == "ottoman_turkish"

    def test_already_canonical_passthrough(self) -> None:
        assert iso_to_nltk("english") == "english"
        assert iso_to_nltk("arabic") == "arabic"

    def test_case_insensitive(self) -> None:
        assert iso_to_nltk("EN") == "english"
        assert iso_to_nltk("Ar") == "arabic"

    def test_unknown_code_passthrough(self) -> None:
        assert iso_to_nltk("zz") == "zz"

    def test_whitespace_stripped(self) -> None:
        assert iso_to_nltk("  en  ") == "english"

    # Regional aliases
    def test_chilean_spanish_alias(self) -> None:
        assert iso_to_nltk("chilean_spanish") == "spanish"

    def test_paraguayan_spanish_alias(self) -> None:
        assert iso_to_nltk("paraguayan_spanish") == "spanish"

    def test_new_zealand_english_alias(self) -> None:
        assert iso_to_nltk("new_zealand_english") == "english"

    def test_australian_english_alias(self) -> None:
        assert iso_to_nltk("australian_english") == "english"

    def test_ottoman_turkish_alias(self) -> None:
        assert iso_to_nltk("ota") == "ottoman_turkish"

    def test_mandarin_alias(self) -> None:
        assert iso_to_nltk("mandarin") == "chinese"

    def test_indonesian_alias(self) -> None:
        assert iso_to_nltk("indonesian") == "malay"

    def test_brazilian_portuguese_alias(self) -> None:
        assert iso_to_nltk("brazilian_portuguese") == "portuguese"

    def test_egyptian_arabic_alias(self) -> None:
        assert iso_to_nltk("egyptian_arabic") == "arabic"


# ===========================================================================
# nltk_to_iso() — reverse helper
# ===========================================================================


class TestNltkToIsoFn:
    def test_english(self) -> None:
        assert nltk_to_iso("english") == "en"

    def test_arabic(self) -> None:
        assert nltk_to_iso("arabic") == "ar"

    def test_unknown_passthrough(self) -> None:
        assert nltk_to_iso("klingon") == "klingon"

    def test_case_insensitive(self) -> None:
        assert nltk_to_iso("English") == "en"

    def test_whitespace_stripped(self) -> None:
        assert nltk_to_iso("  english  ") == "en"


# ===========================================================================
# coerce_language() — the core helper
# ===========================================================================


class TestCoerceLanguage:
    # None → default
    def test_none_returns_default_english(self) -> None:
        assert coerce_language(None) == ["english"]

    def test_none_with_custom_default(self) -> None:
        assert coerce_language(None, default="arabic") == ["arabic"]

    # str forms
    def test_iso_str_en(self) -> None:
        assert coerce_language("en") == ["english"]

    def test_iso_str_ar(self) -> None:
        assert coerce_language("ar") == ["arabic"]

    def test_nltk_str_english(self) -> None:
        assert coerce_language("english") == ["english"]

    def test_nltk_str_arabic(self) -> None:
        assert coerce_language("arabic") == ["arabic"]

    def test_str_case_insensitive(self) -> None:
        assert coerce_language("EN") == ["english"]

    def test_str_whitespace_stripped(self) -> None:
        assert coerce_language("  en  ") == ["english"]

    def test_str_thai(self) -> None:
        assert coerce_language("th") == ["thai"]

    def test_str_vietnamese(self) -> None:
        assert coerce_language("vi") == ["vietnamese"]

    def test_str_swahili(self) -> None:
        assert coerce_language("sw") == ["swahili"]

    def test_str_zulu(self) -> None:
        assert coerce_language("zu") == ["zulu"]

    def test_str_chilean_spanish_alias(self) -> None:
        assert coerce_language("chilean_spanish") == ["spanish"]

    def test_str_nz_english_alias(self) -> None:
        assert coerce_language("new_zealand_english") == ["english"]

    # list forms
    def test_list_single(self) -> None:
        assert coerce_language(["en"]) == ["english"]

    def test_list_multi(self) -> None:
        result = coerce_language(["en", "ar"])
        assert result == ["english", "arabic"]

    def test_list_mixed_codes_and_names(self) -> None:
        result = coerce_language(["en", "arabic"])
        assert result == ["english", "arabic"]

    def test_list_deduplication(self) -> None:
        result = coerce_language(["en", "en", "english"])
        assert result == ["english"]

    def test_list_order_preserved(self) -> None:
        result = coerce_language(["ar", "en", "hi"])
        assert result == ["arabic", "english", "hindi"]

    def test_list_with_regional_alias(self) -> None:
        result = coerce_language(["chilean_spanish", "en"])
        assert "spanish" in result
        assert "english" in result

    def test_list_with_ancient_languages(self) -> None:
        result = coerce_language(["la", "grc"])
        assert "latin" in result
        assert "ancient_greek" in result

    # error cases
    def test_empty_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            coerce_language([])

    def test_non_str_list_element_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="list elements must be str"):
            coerce_language([42])  # type: ignore[list-item]

    def test_bad_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="str.*list.*None"):
            coerce_language(123)  # type: ignore[arg-type]

    # World language spot-checks
    def test_200_plus_language_spot_checks(self) -> None:
        """Spot-check a diverse set of world languages."""
        cases = [
            ("th", "thai"),
            ("vi", "vietnamese"),
            ("ms", "malay"),
            ("sw", "swahili"),
            ("zu", "zulu"),
            ("af", "afrikaans"),
            ("la", "latin"),
            ("grc", "ancient_greek"),
            ("fa", "persian"),
            ("ur", "urdu"),
            ("ko", "korean"),
            ("ja", "japanese"),
            ("ota", "ottoman_turkish"),
            ("ka", "georgian"),
            ("hy", "armenian"),
            ("fil", "filipino"),
            ("km", "khmer"),
            ("my", "burmese"),
            ("am", "amharic"),
            ("ti", "tigrinya"),
        ]
        for iso, expected in cases:
            result = coerce_language(iso)
            assert result == [expected], f"{iso!r} → {result!r}, expected [{expected!r}]"


# ===========================================================================
# resolve_stopwords()
# ===========================================================================


class TestResolveStopwords:
    def test_english_contains_common_words(self) -> None:
        sw = resolve_stopwords("en")
        assert "the" in sw

    def test_hindi_contains_hindi_words(self) -> None:
        sw = resolve_stopwords("hi")
        assert "और" in sw or "में" in sw

    def test_thai_contains_thai_words(self) -> None:
        sw = resolve_stopwords("th")
        assert "ที่" in sw

    def test_multi_language_union(self) -> None:
        sw = resolve_stopwords(["en", "hi"])
        assert "the" in sw
        assert "और" in sw or "में" in sw

    def test_multi_three_languages(self) -> None:
        sw = resolve_stopwords(["en", "ar", "hi"])
        # English
        assert "the" in sw
        # Arabic (from NLTK or built-in)
        # Hindi
        assert "और" in sw or "में" in sw

    def test_extra_stopwords_merged(self) -> None:
        sw = resolve_stopwords("en", extra=frozenset(["foo", "bar"]))
        assert "foo" in sw
        assert "bar" in sw
        assert "the" in sw

    def test_none_resolves_to_english(self) -> None:
        sw = resolve_stopwords(None)
        assert "the" in sw

    def test_result_is_frozenset(self) -> None:
        sw = resolve_stopwords("en")
        assert isinstance(sw, frozenset)

    def test_latin_stopwords(self) -> None:
        sw = resolve_stopwords("la")
        assert "et" in sw

    def test_ancient_greek_stopwords(self) -> None:
        sw = resolve_stopwords("grc")
        assert len(sw) > 0

    def test_swahili_stopwords(self) -> None:
        sw = resolve_stopwords("sw")
        assert "na" in sw

    def test_zulu_stopwords(self) -> None:
        sw = resolve_stopwords("zu")
        assert len(sw) > 0

    def test_regional_alias_resolves(self) -> None:
        sw_chile = resolve_stopwords("chilean_spanish")
        sw_es = resolve_stopwords("es")
        # Both should resolve to Spanish - same content
        assert sw_chile == sw_es

    def test_new_zealand_english(self) -> None:
        sw = resolve_stopwords("new_zealand_english")
        assert "the" in sw
