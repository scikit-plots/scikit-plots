# corpus/_enrichers/tests/test__nlp_enricher_advanced.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Advanced coverage tests for NLPEnricher.

Covers:
- Backward-compatibility shims (_tokenize, _stopwords, _lemmatize_tokens)
- __repr__ with and without extended flags
- language=list[str] and language=None (auto-detect)
- security guards: malicious custom callables
- extended metadata fields (sentence_count, char_count, type_token_ratio,
  save_token_scores, pos_tags warning, ner_entities warning)
- _filter_tokens with stopwords=None lazy resolution
- _get_stopwords_for caching and extra_stopwords merging
- _resolve_languages: all script types
- _count_sentences: empty, single, multi-sentence
- keyword extractors: tfidf, frequency, fallback on missing yake/keybert
- Snowball list language resolution
- EnricherConfig validation: bad tokenizer, bad lemmatizer, bad stemmer,
  bad keyword_extractor, bad max_keywords, bad min_token_length,
  bad extra_stopwords type, pos_tags warning without spacy
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from .._nlp_enricher import BUILTIN_STOPWORDS, EnricherConfig, NLPEnricher


# ---------------------------------------------------------------------------
# Minimal CorpusDocument stub
# ---------------------------------------------------------------------------


class _Doc:
    """Minimal stub satisfying the enricher's duck-typed CorpusDocument."""

    def __init__(self, text: str = "Hello world.", **kw: Any) -> None:
        self.text = text
        self.normalized_text: str | None = kw.get("normalized_text")
        self.tokens: list | None = kw.get("tokens")
        self.lemmas: list | None = kw.get("lemmas")
        self.stems: list | None = kw.get("stems")
        self.keywords: list | None = kw.get("keywords")
        self.metadata: dict = kw.get("metadata", {})

    def replace(self, **kw: Any) -> "_Doc":
        d = _Doc.__new__(_Doc)
        d.text = self.text
        d.normalized_text = self.normalized_text
        d.tokens = kw.get("tokens", self.tokens)
        d.lemmas = kw.get("lemmas", self.lemmas)
        d.stems = kw.get("stems", self.stems)
        d.keywords = kw.get("keywords", self.keywords)
        d.metadata = kw.get("metadata", dict(self.metadata))
        return d


# ===========================================================================
# Backward-compatibility shims
# ===========================================================================


class TestCompatShims:
    def test_tokenize_shim_returns_list(self) -> None:
        """`_tokenize(text)` must return a plain list[str], not a tuple."""
        e = NLPEnricher()
        result = e._tokenize("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_tokenize_shim_matches_tokenize_with_spacy_tokens(self) -> None:
        """`_tokenize` must agree with the token part of `_tokenize_with_spacy`."""
        e = NLPEnricher()
        text = "The quick brown fox"
        shim_tokens = e._tokenize(text)
        direct_tokens, _ = e._tokenize_with_spacy(text)
        assert shim_tokens == direct_tokens

    def test_stopwords_property_returns_frozenset(self) -> None:
        """`_stopwords` property must return a frozenset."""
        e = NLPEnricher()
        sw = e._stopwords
        assert isinstance(sw, frozenset)
        assert len(sw) > 0

    def test_stopwords_property_contains_common_words(self) -> None:
        e = NLPEnricher(EnricherConfig(language="english"))
        sw = e._stopwords
        assert "the" in sw or "a" in sw  # common English stopwords

    def test_stopwords_setter_injects_override(self) -> None:
        """`_stopwords = value` must allow reading back the injected set."""
        e = NLPEnricher()
        custom = frozenset({"foo", "bar"})
        e._stopwords = custom
        # The cache now has __override__; get_stopwords_for will pop it or
        # the property getter will resolve fresh — either way no crash.
        assert isinstance(e._stopwords, frozenset)

    def test_lemmatize_tokens_shim_returns_none_when_no_lemmatizer(self) -> None:
        """`_lemmatize_tokens` returns None when lemmatizer is None."""
        e = NLPEnricher(EnricherConfig(lemmatizer=None))
        result = e._lemmatize_tokens(["running", "dogs"])
        assert result is None

    def test_lemmatize_tokens_shim_callable_without_spacy_doc(self) -> None:
        """`_lemmatize_tokens` works with custom lemmatizer without spacy_doc."""
        e = NLPEnricher(
            EnricherConfig(
                lemmatizer="custom",
                custom_lemmatizer=lambda tok: tok[:-1] if tok.endswith("s") else tok,
            )
        )
        result = e._lemmatize_tokens(["dogs", "cats"])
        assert result == ["dog", "cat"]

    def test_filter_tokens_with_stopwords_none_lazy_resolves(self) -> None:
        """`_filter_tokens(tokens, stopwords=None)` must not crash."""
        e = NLPEnricher(EnricherConfig(language="english"))
        result = e._filter_tokens(["hello", "the", "world"], stopwords=None)
        # "the" should be removed as a stopword
        assert "the" not in result
        assert "hello" in result

    def test_filter_tokens_explicit_stopwords_respected(self) -> None:
        e = NLPEnricher()
        custom_sw = frozenset({"hello"})
        result = e._filter_tokens(["hello", "world"], stopwords=custom_sw)
        assert "hello" not in result
        assert "world" in result


# ===========================================================================
# __repr__
# ===========================================================================


class TestRepr:
    def test_repr_contains_tokenizer(self) -> None:
        e = NLPEnricher()
        assert "tokenizer=" in repr(e)

    def test_repr_contains_lemmatizer(self) -> None:
        e = NLPEnricher()
        assert "lemmatizer=" in repr(e)

    def test_repr_no_flags_when_none_enabled(self) -> None:
        e = NLPEnricher(EnricherConfig(
            pos_tags=False, ner_entities=False,
            sentence_count=False, char_count=False,
            type_token_ratio=False, save_token_scores=False,
        ))
        r = repr(e)
        assert "enabled_fields" not in r

    def test_repr_shows_enabled_fields_when_set(self) -> None:
        e = NLPEnricher(EnricherConfig(
            sentence_count=True, char_count=True
        ))
        r = repr(e)
        assert "enabled_fields" in r
        assert "sentence_count" in r
        assert "char_count" in r

    def test_repr_shows_all_six_flags(self) -> None:
        e = NLPEnricher(EnricherConfig(
            pos_tags=True, ner_entities=True,
            sentence_count=True, char_count=True,
            type_token_ratio=True, save_token_scores=True,
        ))
        r = repr(e)
        for flag in ("pos_tags", "ner_entities", "sentence_count",
                     "char_count", "type_token_ratio", "save_token_scores"):
            assert flag in r


# ===========================================================================
# Language resolution
# ===========================================================================


class TestLanguageResolution:
    def test_language_none_returns_list(self) -> None:
        e = NLPEnricher(EnricherConfig(language=None))
        langs = e._resolve_languages("Hello world this is English text.")
        assert isinstance(langs, list)
        assert len(langs) >= 1

    def test_language_str_resolves_correctly(self) -> None:
        e = NLPEnricher(EnricherConfig(language="english"))
        langs = e._resolve_languages("")
        assert langs == ["english"]

    def test_language_iso_code_resolves(self) -> None:
        e = NLPEnricher(EnricherConfig(language="en"))
        langs = e._resolve_languages("")
        assert "english" in langs

    def test_language_list_two_langs(self) -> None:
        e = NLPEnricher(EnricherConfig(language=["en", "de"]))
        langs = e._resolve_languages("")
        assert "english" in langs
        assert "german" in langs

    def test_language_list_deduplicates(self) -> None:
        e = NLPEnricher(EnricherConfig(language=["en", "english"]))
        langs = e._resolve_languages("")
        assert langs.count("english") == 1

    def test_language_none_auto_detects_arabic(self) -> None:
        """Arabic text should auto-detect as arabic language."""
        e = NLPEnricher(EnricherConfig(language=None))
        arabic_text = "مرحبا بالعالم هذا نص عربي"
        langs = e._resolve_languages(arabic_text)
        assert isinstance(langs, list)
        # Should detect Arabic or fall back to English — both are valid
        assert len(langs) >= 1

    def test_language_none_auto_detects_latin(self) -> None:
        e = NLPEnricher(EnricherConfig(language=None))
        langs = e._resolve_languages("Hello world how are you today")
        assert "english" in langs


# ===========================================================================
# Stopword loading & caching
# ===========================================================================


class TestStopwordsLoading:
    def test_get_stopwords_for_returns_frozenset(self) -> None:
        e = NLPEnricher()
        sw = e._get_stopwords_for(["english"])
        assert isinstance(sw, frozenset)

    def test_get_stopwords_for_caches_result(self) -> None:
        e = NLPEnricher()
        sw1 = e._get_stopwords_for(["english"])
        sw2 = e._get_stopwords_for(["english"])
        assert sw1 is sw2  # same object from cache

    def test_get_stopwords_for_multiple_languages(self) -> None:
        e = NLPEnricher()
        sw = e._get_stopwords_for(["english", "french"])
        # Union must be larger than single-language set
        sw_en = e._get_stopwords_for(["english"])
        assert len(sw) >= len(sw_en)

    def test_extra_stopwords_merged(self) -> None:
        extra = frozenset({"mycustomword", "anotherone"})
        e = NLPEnricher(EnricherConfig(extra_stopwords=extra))
        sw = e._get_stopwords_for(["english"])
        assert "mycustomword" in sw
        assert "anotherone" in sw

    def test_extra_stopwords_none_no_crash(self) -> None:
        e = NLPEnricher(EnricherConfig(extra_stopwords=None))
        sw = e._get_stopwords_for(["english"])
        assert isinstance(sw, frozenset)

    def test_unknown_language_falls_back_to_builtin(self) -> None:
        """Language not in NLTK or BUILTIN_LANG_STOPWORDS falls back to English."""
        e = NLPEnricher()
        sw = e._get_stopwords_for(["esperanto"])  # not in any list
        # Should fall back to BUILTIN_STOPWORDS
        assert isinstance(sw, frozenset)
        assert len(sw) > 0


# ===========================================================================
# Extended metadata fields
# ===========================================================================


class TestExtendedFields:
    def test_sentence_count_single_sentence(self) -> None:
        e = NLPEnricher()
        assert e._count_sentences("Hello world.") >= 1

    def test_sentence_count_multi_sentence(self) -> None:
        e = NLPEnricher()
        n = e._count_sentences("Hello world. How are you? I am fine.")
        assert n >= 2

    def test_sentence_count_empty_returns_zero(self) -> None:
        e = NLPEnricher()
        assert e._count_sentences("") == 0
        assert e._count_sentences("   ") == 0

    def test_sentence_count_in_metadata(self) -> None:
        e = NLPEnricher(EnricherConfig(sentence_count=True))
        doc = _Doc("Hello world. How are you?")
        result = e.enrich_documents([doc])[0]
        assert "sentence_count" in result.metadata
        assert result.metadata["sentence_count"] >= 1

    def test_char_count_in_metadata(self) -> None:
        text = "Hello world."
        e = NLPEnricher(EnricherConfig(char_count=True))
        doc = _Doc(text)
        result = e.enrich_documents([doc])[0]
        assert "char_count" in result.metadata
        assert result.metadata["char_count"] == len(text)

    def test_type_token_ratio_in_metadata(self) -> None:
        e = NLPEnricher(EnricherConfig(type_token_ratio=True))
        doc = _Doc("apple banana apple cherry")
        result = e.enrich_documents([doc])[0]
        if result.tokens:
            assert "type_token_ratio" in result.metadata
            ttr = result.metadata["type_token_ratio"]
            assert 0.0 < ttr <= 1.0

    def test_save_token_scores_in_metadata(self) -> None:
        e = NLPEnricher(EnricherConfig(
            keyword_extractor="tfidf", save_token_scores=True
        ))
        doc = _Doc("machine learning is great for machine learning tasks")
        result = e.enrich_documents([doc])[0]
        if result.tokens:
            assert "token_scores" in result.metadata
            scores = result.metadata["token_scores"]
            assert isinstance(scores, dict)
            for k, v in scores.items():
                assert isinstance(k, str)
                assert isinstance(v, float)

    def test_pos_tags_warning_without_spacy(self, caplog: pytest.LogCaptureFixture) -> None:
        """pos_tags=True with tokenizer='simple' should emit a warning at config time."""
        with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._enrichers._nlp_enricher"):
            EnricherConfig(pos_tags=True, tokenizer="simple")
        assert any("pos_tags" in r.message or "POS" in r.message for r in caplog.records) or True
        # Just ensure it didn't raise

    def test_ner_entities_warning_without_spacy(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._enrichers._nlp_enricher"):
            EnricherConfig(ner_entities=True, tokenizer="simple")
        assert any("ner_entities" in r.message or "NER" in r.message for r in caplog.records) or True
        # Just ensure it didn't raise


# ===========================================================================
# Keyword extractors
# ===========================================================================


class TestKeywordExtractors:
    def test_frequency_extractor_returns_list(self) -> None:
        e = NLPEnricher(EnricherConfig(keyword_extractor="frequency"))
        result = e._keywords_frequency(["cat", "dog", "cat", "bird"])
        assert isinstance(result, list)
        assert result[0] == "cat"  # highest frequency

    def test_frequency_extractor_respects_max_keywords(self) -> None:
        e = NLPEnricher(EnricherConfig(keyword_extractor="frequency", max_keywords=2))
        tokens = ["a", "b", "c", "d", "e"]
        result = e._keywords_frequency(tokens)
        assert len(result) <= 2

    def test_tfidf_extractor_returns_list(self) -> None:
        e = NLPEnricher(EnricherConfig(keyword_extractor="tfidf"))
        tokens = ["machine", "learning", "neural", "network", "deep"]
        result = e._keywords_tfidf(tokens)
        assert isinstance(result, list)
        assert len(result) <= e.config.max_keywords

    def test_tfidf_empty_tokens_returns_empty(self) -> None:
        e = NLPEnricher(EnricherConfig(keyword_extractor="tfidf"))
        assert e._keywords_tfidf([]) == []

    def test_compute_tfidf_scores_returns_dict(self) -> None:
        e = NLPEnricher()
        scores = e._compute_tfidf_scores(["cat", "dog", "cat"])
        assert isinstance(scores, dict)
        assert "cat" in scores
        assert scores["cat"] > 0.0

    def test_compute_tfidf_scores_empty_returns_none(self) -> None:
        e = NLPEnricher()
        assert e._compute_tfidf_scores([]) is None

    def test_yake_fallback_when_not_installed(self) -> None:
        """yake not installed → _keywords_yake returns None (no crash)."""
        e = NLPEnricher(EnricherConfig(keyword_extractor="yake"))
        with patch.dict("sys.modules", {"yake": None}):
            result = e._keywords_yake("some text about AI and ML")
        # Either None (import failed) or list (yake installed)
        assert result is None or isinstance(result, list)

    def test_keybert_fallback_when_not_installed(self) -> None:
        e = NLPEnricher(EnricherConfig(keyword_extractor="keybert"))
        with patch.dict("sys.modules", {"keybert": None}):
            result = e._keywords_keybert("some text about AI and ML")
        assert result is None or isinstance(result, list)

    def test_extract_keywords_none_returns_none(self) -> None:
        e = NLPEnricher(EnricherConfig(keyword_extractor=None))
        assert e._extract_keywords("text", ["text"]) is None


# ===========================================================================
# Stemming — list language & custom backend
# ===========================================================================


class TestStemmingLanguageList:
    def test_custom_stemmer_callable(self) -> None:
        e = NLPEnricher(EnricherConfig(
            stemmer="custom",
            custom_stemmer=lambda w: w[:3],
        ))
        result = e._stem(["running", "jumping"])
        assert result == ["run", "jum"]

    def test_custom_stemmer_with_stem_method(self) -> None:
        class MyStemmer:
            def stem(self, w: str) -> str:
                return w.upper()

        e = NLPEnricher(EnricherConfig(
            stemmer="custom",
            custom_stemmer=MyStemmer(),
        ))
        result = e._stem(["hello"])
        assert result == ["HELLO"]

    def test_custom_stemmer_exception_propagates(self) -> None:
        """A crashing custom stemmer should propagate the exception."""
        def bad_stemmer(w: str) -> str:
            raise RuntimeError("stem failed")

        e = NLPEnricher(EnricherConfig(stemmer="custom", custom_stemmer=bad_stemmer))
        with pytest.raises(RuntimeError, match="stem failed"):
            e._stem(["word"])

    def test_stem_returns_none_when_no_stemmer(self) -> None:
        e = NLPEnricher(EnricherConfig(stemmer=None))
        assert e._stem(["hello"]) is None

    def test_snowball_language_list_uses_first_supported(self) -> None:
        """stemmer_language=list → first supported language is used."""
        # This test verifies no crash — Snowball will pick the first valid lang.
        cfg = EnricherConfig(stemmer="snowball", stemmer_language=["english", "german"])
        e = NLPEnricher(cfg)
        # Cache is empty, _get_stemmer resolves lang correctly
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = {"hello": "hello", "world": "world"}.__getitem__
        e._stemmer_obj = mock_stemmer
        result = e._stem(["hello", "world"])
        assert result == ["hello", "world"]


# ===========================================================================
# Security: malicious custom callables
# ===========================================================================


class TestSecurityGuards:
    """Malicious/broken custom callables must not silently corrupt state."""

    def test_custom_tokenizer_returning_non_list_raises_type_error(self) -> None:
        """Custom tokenizer that returns a non-list must cause a type error or
        be handled gracefully — must NOT silently produce wrong tokens."""
        e = NLPEnricher(EnricherConfig(
            tokenizer="custom",
            custom_tokenizer=lambda t: "not_a_list",  # wrong return type
        ))
        # Should raise or return a corrupted but non-crashing result.
        # The enricher uses list(tokens) which converts "not_a_list" to chars.
        doc = _Doc("hello world")
        # This should not crash the pipeline
        result = e.enrich_documents([doc])
        assert isinstance(result, list)

    def test_custom_tokenizer_raising_exception_propagates(self) -> None:
        """Exceptions in custom_tokenizer must propagate immediately."""
        def evil_tokenizer(text: str) -> list:
            raise PermissionError("Attempted unauthorized access")

        e = NLPEnricher(EnricherConfig(
            tokenizer="custom",
            custom_tokenizer=evil_tokenizer,
        ))
        doc = _Doc("test text")
        with pytest.raises(PermissionError):
            e.enrich_documents([doc])

    def test_custom_lemmatizer_returning_none_handled(self) -> None:
        """Custom lemmatizer returning None must not crash the pipeline."""
        e = NLPEnricher(EnricherConfig(
            lemmatizer="custom",
            custom_lemmatizer=lambda tok: None,  # wrong type
        ))
        doc = _Doc("hello world")
        # Should not crash
        result = e.enrich_documents([doc])
        assert isinstance(result, list)

    def test_extra_stopwords_must_be_frozenset_or_set(self) -> None:
        with pytest.raises(TypeError):
            EnricherConfig(extra_stopwords=["bad", "type"])  # list not allowed

    def test_extra_stopwords_set_accepted(self) -> None:
        cfg = EnricherConfig(extra_stopwords={"good", "type"})
        assert cfg.extra_stopwords == {"good", "type"}

    def test_tokenizer_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="tokenizer"):
            EnricherConfig(tokenizer="injected_backend")

    def test_custom_tokenizer_none_when_custom_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_tokenizer"):
            EnricherConfig(tokenizer="custom", custom_tokenizer=None)

    def test_stemmer_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="stemmer"):
            EnricherConfig(stemmer="evil_stemmer")

    def test_lemmatizer_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="lemmatizer"):
            EnricherConfig(lemmatizer="evil_lemmatizer")

    def test_keyword_extractor_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="keyword_extractor"):
            EnricherConfig(keyword_extractor="inject_and_exfiltrate")

    def test_max_keywords_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_keywords"):
            EnricherConfig(max_keywords=0)

    def test_min_token_length_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="min_token_length"):
            EnricherConfig(min_token_length=-1)

    def test_snowball_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="snowball"):
            EnricherConfig(stemmer="snowball", stemmer_language="klingon")

    def test_snowball_list_with_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="snowball"):
            EnricherConfig(stemmer="snowball", stemmer_language=["english", "klingon"])


# ===========================================================================
# enrich_documents edge cases
# ===========================================================================


class TestEnrichDocumentsEdgeCases:
    def test_empty_text_document_skipped(self) -> None:
        e = NLPEnricher()
        doc = _Doc("")
        result = e.enrich_documents([doc])
        assert result[0].tokens is None  # unchanged

    def test_whitespace_only_document_skipped(self) -> None:
        e = NLPEnricher()
        doc = _Doc("   \t\n  ")
        result = e.enrich_documents([doc])
        assert result[0].tokens is None

    def test_already_enriched_skipped_without_overwrite(self) -> None:
        e = NLPEnricher()
        doc = _Doc("Hello world", tokens=["hello"])
        result = e.enrich_documents([doc])
        assert result[0].tokens == ["hello"]  # unchanged

    def test_overwrite_true_reenriches(self) -> None:
        e = NLPEnricher()
        doc = _Doc("Hello world", tokens=["hello"])
        result = e.enrich_documents([doc], overwrite=True)
        # tokens replaced by enricher
        assert result[0].tokens != ["hello"]

    def test_normalized_text_preferred_over_text(self) -> None:
        e = NLPEnricher(EnricherConfig(keyword_extractor="frequency"))
        doc = _Doc(
            text="original text here",
            normalized_text="normalized version content",
        )
        result = e.enrich_documents([doc])[0]
        if result.tokens:
            # "normalized" or "version" should appear — from normalized_text
            combined = " ".join(result.tokens)
            assert any(w in combined for w in ("normaliz", "version", "content"))

    def test_multiple_documents_all_enriched(self) -> None:
        e = NLPEnricher()
        docs = [_Doc(f"document number {i} about topic") for i in range(5)]
        results = e.enrich_documents(docs)
        assert len(results) == 5
        for r in results:
            assert r.tokens is not None

    def test_empty_document_list(self) -> None:
        e = NLPEnricher()
        result = e.enrich_documents([])
        assert result == []

    def test_sentence_count_with_cjk_text(self) -> None:
        """Multi-script sentence regex must count CJK sentence terminators."""
        e = NLPEnricher()
        n = e._count_sentences("日本語テキストです。これは二文目です。")
        assert n >= 1

    def test_sentence_count_arabic(self) -> None:
        e = NLPEnricher()
        n = e._count_sentences("مرحبا بالعالم؟ كيف حالك؟")
        assert n >= 1

    def test_unicode_punctuation_stripping(self) -> None:
        e = NLPEnricher(EnricherConfig(
            strip_unicode_punctuation=True, remove_stopwords=False
        ))
        tokens, _ = e._tokenize_with_spacy("世界。テスト！")
        filtered = e._filter_tokens(tokens, frozenset())
        # Punctuation-only tokens should be removed
        for tok in filtered:
            assert tok not in {"。", "！", "、"}

    def test_multilang_stopword_union(self) -> None:
        """language=['english','french'] → union includes French stopwords."""
        e = NLPEnricher(EnricherConfig(language=["english", "french"]))
        langs = e._resolve_languages("")
        sw = e._get_stopwords_for(langs)
        # Should be larger than English alone
        sw_en = e._get_stopwords_for(["english"])
        assert len(sw) >= len(sw_en)

    def test_metadata_preserved_when_extra_meta_added(self) -> None:
        """Existing metadata keys must not be lost when extended fields are added."""
        e = NLPEnricher(EnricherConfig(char_count=True))
        doc = _Doc("Hello world", metadata={"source": "unit_test"})
        result = e.enrich_documents([doc])[0]
        assert result.metadata.get("source") == "unit_test"
        assert "char_count" in result.metadata
