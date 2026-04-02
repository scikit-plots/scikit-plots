"""
tests/test__normalizers.py
============================
Tests for scikitplot.corpus._normalizers.
All langdetect calls are mocked — zero optional dependencies required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from .._normalizer import (
    DedupLinesNormalizer,
    HTMLStripNormalizer,
    LanguageDetectionNormalizer,
    LowercaseNormalizer,
    NormalizationPipeline,
    NormalizerBase,
    UnicodeNormalizer,
    WhitespaceNormalizer,
)
from ..._schema import CorpusDocument


def _doc(text: str, normalized_text: str | None = None) -> CorpusDocument:
    doc = CorpusDocument.create("f.txt", 0, text)
    if normalized_text is not None:
        doc = doc.replace(normalized_text=normalized_text)
    return doc


# ===========================================================================
# UnicodeNormalizer
# ===========================================================================


class TestUnicodeNormalizer:
    def test_nfc_default(self) -> None:
        norm = UnicodeNormalizer()
        assert norm.form == "NFC"

    def test_invalid_form_raises(self) -> None:
        with pytest.raises(ValueError, match="form"):
            UnicodeNormalizer(form="XYZ")

    def test_nfkc_decomposes_ligature(self) -> None:
        norm = UnicodeNormalizer(form="NFKC")
        # fi ligature (U+FB01)
        doc = _doc("\ufb01le")
        result = norm.normalize_doc(doc)
        assert result.normalized_text == "file"
        assert result.text == "\ufb01le"  # original preserved

    def test_no_change_returns_same_object(self) -> None:
        norm = UnicodeNormalizer(form="NFC")
        doc = _doc("hello world")
        result = norm.normalize_doc(doc)
        # When already NFC, text is unchanged and no replace() call made
        assert result.normalized_text is None or result.normalized_text == "hello world"

    def test_uses_normalized_text_as_source(self) -> None:
        norm = UnicodeNormalizer(form="NFKC")
        doc = _doc("base", normalized_text="\ufb01le")
        result = norm.normalize_doc(doc)
        assert result.normalized_text == "file"
        assert result.text == "base"

    def test_repr(self) -> None:
        assert "NFKC" in repr(UnicodeNormalizer(form="NFKC"))


# ===========================================================================
# WhitespaceNormalizer
# ===========================================================================


class TestWhitespaceNormalizer:
    def test_collapses_spaces(self) -> None:
        norm = WhitespaceNormalizer()
        doc = _doc("hello   world")
        result = norm.normalize_doc(doc)
        assert result.normalized_text == "hello world"

    def test_strips_leading_trailing(self) -> None:
        norm = WhitespaceNormalizer(strip=True)
        doc = _doc("  hello  ")
        result = norm.normalize_doc(doc)
        assert result.normalized_text == "hello"

    def test_preserves_newlines_by_default(self) -> None:
        norm = WhitespaceNormalizer(collapse_newlines=False)
        doc = _doc("line1\nline2")
        result = norm.normalize_doc(doc)
        assert "\n" in (result.normalized_text or result.text)

    def test_collapse_newlines(self) -> None:
        norm = WhitespaceNormalizer(collapse_newlines=True)
        doc = _doc("line1\nline2")
        result = norm.normalize_doc(doc)
        assert "\n" not in (result.normalized_text or "")

    def test_no_change_preserves_doc(self) -> None:
        norm = WhitespaceNormalizer()
        doc = _doc("clean text")
        result = norm.normalize_doc(doc)
        assert result.text == doc.text

    def test_repr(self) -> None:
        assert "WhitespaceNormalizer" in repr(WhitespaceNormalizer())


# ===========================================================================
# HTMLStripNormalizer
# ===========================================================================


class TestHTMLStripNormalizer:
    def test_strips_tags_regex(self) -> None:
        norm = HTMLStripNormalizer()
        doc = _doc("<p>Hello <b>world</b>.</p>")
        result = norm.normalize_doc(doc)
        assert result.normalized_text is not None
        assert "<p>" not in result.normalized_text
        assert "Hello" in result.normalized_text

    def test_decode_entities(self) -> None:
        norm = HTMLStripNormalizer(decode_entities=True)
        doc = _doc("&amp; &lt; &gt;")
        result = norm.normalize_doc(doc)
        assert "&amp;" not in (result.normalized_text or "")

    def test_no_tags_returns_unchanged(self) -> None:
        norm = HTMLStripNormalizer()
        doc = _doc("plain text without tags")
        result = norm.normalize_doc(doc)
        # No change means normalized_text may still be None
        text_out = result.normalized_text or result.text
        assert "plain text" in text_out

    def test_bs4_import_error_raises(self) -> None:
        norm = HTMLStripNormalizer(use_beautifulsoup=True)
        doc = _doc("<p>hello</p>")
        with patch.dict("sys.modules", {"bs4": None}):
            with pytest.raises(ImportError, match="beautifulsoup4"):
                norm.normalize_doc(doc)

    def test_repr(self) -> None:
        assert "HTMLStripNormalizer" in repr(HTMLStripNormalizer())


# ===========================================================================
# LowercaseNormalizer
# ===========================================================================


class TestLowercaseNormalizer:
    def test_lowercases_text(self) -> None:
        norm = LowercaseNormalizer()
        doc = _doc("HELLO World")
        result = norm.normalize_doc(doc)
        assert result.normalized_text == "hello world"

    def test_casefold(self) -> None:
        norm = LowercaseNormalizer(locale_aware=True)
        doc = _doc("Straße")  # German sharp s
        result = norm.normalize_doc(doc)
        assert result.normalized_text == "strasse"

    def test_already_lowercase_no_change(self) -> None:
        norm = LowercaseNormalizer()
        doc = _doc("already lowercase")
        result = norm.normalize_doc(doc)
        # normalized_text should be None (no change) or same
        text_out = result.normalized_text or result.text
        assert text_out == "already lowercase"

    def test_original_text_preserved(self) -> None:
        norm = LowercaseNormalizer()
        doc = _doc("CAPS")
        result = norm.normalize_doc(doc)
        assert result.text == "CAPS"


# ===========================================================================
# DedupLinesNormalizer
# ===========================================================================


class TestDedupLinesNormalizer:
    def test_removes_exact_duplicates(self) -> None:
        norm = DedupLinesNormalizer()
        doc = _doc("Hello.\nHello.\nWorld.")
        result = norm.normalize_doc(doc)
        assert result.normalized_text == "Hello.\nWorld."

    def test_preserves_order(self) -> None:
        norm = DedupLinesNormalizer()
        doc = _doc("a\nb\na\nc")
        result = norm.normalize_doc(doc)
        result_text = result.normalized_text
        assert result_text == "a\nb\nc"

    def test_short_lines_always_kept(self) -> None:
        norm = DedupLinesNormalizer(min_line_length=10)
        doc = _doc("ab\nab\nlong enough line")
        result = norm.normalize_doc(doc)
        result_text = result.normalized_text
        # Short "ab" lines kept even if duplicate
        assert result_text is not None
        assert result_text.count("ab") == 2

    def test_ignore_whitespace_default(self) -> None:
        norm = DedupLinesNormalizer(ignore_whitespace=True)
        doc = _doc("  Hello.  \nHello.\nWorld.")
        result = norm.normalize_doc(doc)
        result_text = result.normalized_text
        # "  Hello.  " and "Hello." are considered duplicates
        assert result_text is not None
        lines = result_text.splitlines()
        assert len(lines) == 2

    def test_negative_min_length_raises(self) -> None:
        with pytest.raises(ValueError, match="min_line_length"):
            DedupLinesNormalizer(min_line_length=-1)

    def test_no_duplicates_unchanged(self) -> None:
        norm = DedupLinesNormalizer()
        doc = _doc("Line one.\nLine two.\nLine three.")
        result = norm.normalize_doc(doc)
        result_text = result.normalized_text
        text_out = result_text or result.text
        assert text_out.count("\n") == 2


# ===========================================================================
# LanguageDetectionNormalizer
# ===========================================================================


class TestLanguageDetectionNormalizer:
    def _make_mock_langs(self, lang: str, prob: float) -> list:
        m = MagicMock()
        m.lang = lang
        m.prob = prob
        return [m]

    def _make_langdetect_module(self, mock_langs: list) -> MagicMock:
        """Return a fake ``langdetect`` module with ``detect_langs`` pre-set."""
        mod = MagicMock()
        mod.detect_langs.return_value = mock_langs
        return mod

    def test_detects_language(self) -> None:
        norm = LanguageDetectionNormalizer()
        doc = _doc("The quick brown fox jumps over the lazy dog.")
        mock_langs = self._make_mock_langs("en", 0.99)
        fake_ld = self._make_langdetect_module(mock_langs)
        with patch.dict("sys.modules", {"langdetect": fake_ld}):
            with patch(
                "scikitplot.corpus._normalizers._normalizer."
                "LanguageDetectionNormalizer._get_source_text",
                return_value="The quick brown fox jumps over the lazy dog.",
            ):
                result = norm.normalize_doc(doc)
                assert result.language == "en"

    def test_low_confidence_uses_fallback(self) -> None:
        norm = LanguageDetectionNormalizer(fallback_language="en", min_confidence=0.9)
        doc = _doc("The quick brown fox jumps over the lazy dog.")
        mock_langs = self._make_mock_langs("de", 0.5)
        fake_ld = self._make_langdetect_module(mock_langs)
        with patch.dict("sys.modules", {"langdetect": fake_ld}):
            result = norm.normalize_doc(doc)
            assert result.language == "en"

    def test_skip_if_language_already_set(self) -> None:
        norm = LanguageDetectionNormalizer(overwrite=False)
        doc = CorpusDocument.create("f.txt", 0, "Hello.", language="fr")
        result = norm.normalize_doc(doc)
        assert result.language == "fr"

    def test_overwrite_existing_language(self) -> None:
        norm = LanguageDetectionNormalizer(overwrite=True)
        doc = CorpusDocument.create("f.txt", 0, "Hello world today.", language="fr")
        mock_langs = self._make_mock_langs("en", 0.99)
        fake_ld = self._make_langdetect_module(mock_langs)
        with patch.dict("sys.modules", {"langdetect": fake_ld}):
            result = norm.normalize_doc(doc)
            assert result.language == "en"

    def test_import_error_raises(self) -> None:
        norm = LanguageDetectionNormalizer()
        doc = _doc("Hello world today tomorrow yesterday.")
        with patch.dict("sys.modules", {"langdetect": None}):
            with pytest.raises(ImportError, match="langdetect"):
                norm.normalize_doc(doc)

    def test_invalid_min_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            LanguageDetectionNormalizer(min_confidence=1.5)


# ===========================================================================
# NormalizationPipeline
# ===========================================================================


class TestNormalizationPipeline:
    def test_applies_in_order(self) -> None:
        pipeline = NormalizationPipeline([
            LowercaseNormalizer(),
            WhitespaceNormalizer(),
        ])
        doc = _doc("  HELLO   WORLD  ")
        result = pipeline.normalize_doc(doc)
        nt = result.normalized_text or result.text
        assert nt == "hello world"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            NormalizationPipeline([])

    def test_non_normalizer_raises(self) -> None:
        with pytest.raises(TypeError, match="NormalizerBase"):
            NormalizationPipeline(["not a normalizer"])  # type: ignore[list-item]

    def test_normalize_batch(self) -> None:
        pipeline = NormalizationPipeline([LowercaseNormalizer()])
        docs = [_doc("Hello"), _doc("World")]
        results = pipeline.normalize_batch(docs)
        assert len(results) == 2
        nts = [r.normalized_text or r.text for r in results]
        assert nts == ["hello", "world"]

    def test_original_text_never_modified(self) -> None:
        pipeline = NormalizationPipeline([
            HTMLStripNormalizer(),
            LowercaseNormalizer(),
        ])
        doc = _doc("<P>HELLO</P>")
        result = pipeline.normalize_doc(doc)
        assert result.text == "<P>HELLO</P>"

    def test_repr_includes_stage_names(self) -> None:
        pipeline = NormalizationPipeline([LowercaseNormalizer()])
        assert "LowercaseNormalizer" in repr(pipeline)
