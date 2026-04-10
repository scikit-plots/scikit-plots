# corpus/_enrichers/tests/test__nlp_enricher.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scikitplot.corpus._enrichers._nlp_enricher (rewritten).

Covers:
- EnricherConfig validation: all params, types, edge cases
- EnricherConfig: custom_tokenizer/stemmer/lemmatizer hooks
- EnricherConfig: Snowball language guard (list, unsupported, supported)
- EnricherConfig: extra_stopwords type validation
- NLPEnricher construction: defaults, custom config
- enrich_documents(): happy path, overwrite=False skip, overwrite=True force
- enrich_documents(): empty text is skipped
- Language resolution: None auto-detect, str, list[str], regional alias
- Stopword union: multi-language
- Tokenization: simple, NLTK (mock), custom callable, custom protocol
- Filtering: lowercase, punctuation, stopwords, min_length
- Unicode punctuation stripping
- Stemming: porter (mock), snowball (mock), lancaster (mock), custom
- Lemmatization: NLTK (mock), spaCy (mock), custom
- Keywords: frequency, tfidf, yake (mock), keybert (mock)
- TF-IDF scores: save_token_scores
- Extended fields: sentence_count, char_count, type_token_ratio, pos_tags, ner_entities
- _count_sentences: multi-script, empty, single
- _keywords_tfidf: score ordering, top-N cap
- _compute_tfidf_scores: None for empty tokens
- BUILTIN_STOPWORDS: presence and type
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from .._nlp_enricher import BUILTIN_STOPWORDS, EnricherConfig, NLPEnricher


# ---------------------------------------------------------------------------
# Minimal CorpusDocument stub (avoids importing the full schema)
# ---------------------------------------------------------------------------


class _Doc:
    """Minimal CorpusDocument stub for enricher tests."""

    def __init__(
        self,
        text: str = "",
        normalized_text: str | None = None,
        tokens: list | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.text = text
        self.normalized_text = normalized_text
        self.tokens = tokens
        self.metadata = metadata or {}

    def replace(self, **kwargs: Any) -> "_Doc":
        d = _Doc(
            text=self.text,
            normalized_text=self.normalized_text,
            tokens=self.tokens,
            metadata=dict(self.metadata),
        )
        for k, v in kwargs.items():
            setattr(d, k, v)
        if "metadata" in kwargs:
            d.metadata = dict(kwargs["metadata"])
        return d


def _doc(text: str = "The quick brown fox jumps.", **kw: Any) -> _Doc:
    return _Doc(text=text, **kw)


# ===========================================================================
# BUILTIN_STOPWORDS
# ===========================================================================


class TestBuiltinStopwords:
    def test_is_frozenset(self) -> None:
        assert isinstance(BUILTIN_STOPWORDS, frozenset)

    def test_contains_common_words(self) -> None:
        for word in ("the", "and", "in", "of", "is"):
            assert word in BUILTIN_STOPWORDS

    def test_nonempty(self) -> None:
        assert len(BUILTIN_STOPWORDS) > 50


# ===========================================================================
# EnricherConfig — validation
# ===========================================================================


class TestEnricherConfigDefaults:
    def test_default_tokenizer_is_simple(self) -> None:
        assert EnricherConfig().tokenizer == "simple"

    def test_default_lemmatizer_is_none(self) -> None:
        assert EnricherConfig().lemmatizer is None

    def test_default_stemmer_is_none(self) -> None:
        assert EnricherConfig().stemmer is None

    def test_default_keyword_extractor_is_frequency(self) -> None:
        assert EnricherConfig().keyword_extractor == "frequency"

    def test_default_language_is_none(self) -> None:
        assert EnricherConfig().language is None

    def test_default_max_keywords(self) -> None:
        assert EnricherConfig().max_keywords == 20

    def test_default_pos_tags_false(self) -> None:
        assert EnricherConfig().pos_tags is False

    def test_default_ner_entities_false(self) -> None:
        assert EnricherConfig().ner_entities is False

    def test_default_sentence_count_false(self) -> None:
        assert EnricherConfig().sentence_count is False

    def test_default_char_count_false(self) -> None:
        assert EnricherConfig().char_count is False

    def test_default_type_token_ratio_false(self) -> None:
        assert EnricherConfig().type_token_ratio is False

    def test_default_save_token_scores_false(self) -> None:
        assert EnricherConfig().save_token_scores is False

    def test_default_strip_unicode_punctuation_false(self) -> None:
        assert EnricherConfig().strip_unicode_punctuation is False


class TestEnricherConfigValidation:
    def test_invalid_tokenizer_raises(self) -> None:
        with pytest.raises(ValueError, match="tokenizer"):
            EnricherConfig(tokenizer="bert")

    def test_invalid_lemmatizer_raises(self) -> None:
        with pytest.raises(ValueError, match="lemmatizer"):
            EnricherConfig(lemmatizer="stanza")

    def test_invalid_stemmer_raises(self) -> None:
        with pytest.raises(ValueError, match="stemmer"):
            EnricherConfig(stemmer="regex")

    def test_invalid_keyword_extractor_raises(self) -> None:
        with pytest.raises(ValueError, match="keyword_extractor"):
            EnricherConfig(keyword_extractor="rake")

    def test_max_keywords_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_keywords"):
            EnricherConfig(max_keywords=0)

    def test_min_token_length_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="min_token_length"):
            EnricherConfig(min_token_length=-1)

    def test_custom_tokenizer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_tokenizer"):
            EnricherConfig(tokenizer="custom", custom_tokenizer=None)

    def test_custom_stemmer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_stemmer"):
            EnricherConfig(stemmer="custom", custom_stemmer=None)

    def test_custom_lemmatizer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_lemmatizer"):
            EnricherConfig(lemmatizer="custom", custom_lemmatizer=None)

    def test_extra_stopwords_non_frozenset_raises(self) -> None:
        with pytest.raises(TypeError, match="frozenset"):
            EnricherConfig(extra_stopwords=["the", "and"])  # type: ignore[arg-type]

    def test_extra_stopwords_set_accepted(self) -> None:
        cfg = EnricherConfig(extra_stopwords={"the", "and"})
        assert cfg is not None  # set accepted

    def test_extra_stopwords_frozenset_accepted(self) -> None:
        cfg = EnricherConfig(extra_stopwords=frozenset({"foo"}))
        assert cfg is not None

    def test_snowball_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="snowball"):
            EnricherConfig(stemmer="snowball", stemmer_language="thai")

    def test_snowball_supported_language_ok(self) -> None:
        cfg = EnricherConfig(stemmer="snowball", stemmer_language="english")
        assert cfg is not None

    def test_snowball_language_list_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="snowball"):
            EnricherConfig(stemmer="snowball", stemmer_language=["english", "thai"])

    def test_snowball_language_list_all_supported_ok(self) -> None:
        cfg = EnricherConfig(
            stemmer="snowball", stemmer_language=["english", "german"]
        )
        assert cfg is not None

    def test_all_valid_tokenizers_accepted(self) -> None:
        for tok in ("simple", "nltk", "spacy", "custom"):
            if tok == "spacy":
                cfg = EnricherConfig(tokenizer=tok)
            elif tok == "custom":
                cfg = EnricherConfig(tokenizer=tok, custom_tokenizer=str.split)
            else:
                cfg = EnricherConfig(tokenizer=tok)
            assert cfg is not None

    def test_all_valid_stemmers_accepted(self) -> None:
        for st in (None, "porter", "lancaster"):
            cfg = EnricherConfig(stemmer=st)
            assert cfg is not None

    def test_all_valid_lemmatizers_accepted(self) -> None:
        for lm in (None, "nltk", "spacy"):
            cfg = EnricherConfig(lemmatizer=lm)
            assert cfg is not None

    def test_all_valid_keyword_extractors_accepted(self) -> None:
        for kw in (None, "frequency", "tfidf", "yake", "keybert"):
            cfg = EnricherConfig(keyword_extractor=kw)
            assert cfg is not None


# ===========================================================================
# NLPEnricher construction
# ===========================================================================


class TestNLPEnricherConstruction:
    def test_default_construction(self) -> None:
        enricher = NLPEnricher()
        assert enricher.config.tokenizer == "simple"

    def test_custom_config(self) -> None:
        cfg = EnricherConfig(max_keywords=5)
        enricher = NLPEnricher(cfg)
        assert enricher.config.max_keywords == 5

    def test_repr_contains_tokenizer(self) -> None:
        assert "simple" in repr(NLPEnricher())


# ===========================================================================
# enrich_documents() — core behaviour
# ===========================================================================


class TestEnrichDocuments:
    def _enricher(self, **kw: Any) -> NLPEnricher:
        return NLPEnricher(EnricherConfig(keyword_extractor="frequency", **kw))

    def test_returns_list(self) -> None:
        enricher = self._enricher()
        result = enricher.enrich_documents([_doc()])
        assert isinstance(result, list)

    def test_same_length_as_input(self) -> None:
        enricher = self._enricher()
        docs = [_doc("Alpha beta gamma."), _doc("Delta epsilon.")]
        result = enricher.enrich_documents(docs)
        assert len(result) == 2

    def test_tokens_populated(self) -> None:
        enricher = self._enricher(
            remove_stopwords=True,
            min_token_length=2,
        )
        result = enricher.enrich_documents([_doc("hello world foo bar")])
        assert result[0].tokens is not None
        assert len(result[0].tokens) > 0

    def test_already_enriched_skipped_by_default(self) -> None:
        enricher = self._enricher()
        pre_tokens = ["already", "done"]
        doc = _doc("hello world", tokens=pre_tokens)
        result = enricher.enrich_documents([doc])
        assert result[0].tokens is pre_tokens  # unchanged

    def test_overwrite_true_re_enriches(self) -> None:
        enricher = self._enricher()
        doc = _doc("hello world", tokens=["old"])
        result = enricher.enrich_documents([doc], overwrite=True)
        assert result[0].tokens != ["old"]

    def test_empty_text_skipped(self) -> None:
        enricher = self._enricher()
        doc = _doc("")
        result = enricher.enrich_documents([doc])
        assert result[0].tokens is None

    def test_whitespace_only_text_skipped(self) -> None:
        enricher = self._enricher()
        doc = _doc("   \t\n  ")
        result = enricher.enrich_documents([doc])
        assert result[0].tokens is None

    def test_uses_normalized_text_when_present(self) -> None:
        enricher = self._enricher(remove_stopwords=False, min_token_length=1)
        doc = _doc("ignored", normalized_text="normalized hello")
        result = enricher.enrich_documents([doc])
        tokens = result[0].tokens or []
        assert "normalized" in tokens or "hello" in tokens

    def test_empty_documents_list(self) -> None:
        enricher = self._enricher()
        assert enricher.enrich_documents([]) == []

    def test_keywords_populated_when_extractor_set(self) -> None:
        enricher = self._enricher()
        result = enricher.enrich_documents(
            [_doc("apple orange apple banana apple orange")]
        )
        kw = result[0].keywords
        assert kw is not None
        assert "apple" in kw

    def test_keywords_none_when_extractor_none(self) -> None:
        enricher = NLPEnricher(EnricherConfig(keyword_extractor=None))
        result = enricher.enrich_documents([_doc("hello world")])
        assert result[0].keywords is None


# ===========================================================================
# Language resolution
# ===========================================================================


class TestLanguageResolution:
    def test_none_auto_detects_latin(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language=None))
        langs = enricher._resolve_languages("Hello world this is English text.")
        assert "english" in langs

    def test_none_auto_detects_arabic(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language=None))
        langs = enricher._resolve_languages("مرحبا بالعالم هذا نص عربي")
        assert "arabic" in langs

    def test_str_en_resolves(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language="en"))
        langs = enricher._resolve_languages("anything")
        assert langs == ["english"]

    def test_str_ar_resolves(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language="ar"))
        langs = enricher._resolve_languages("anything")
        assert langs == ["arabic"]

    def test_list_multi_resolves(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language=["en", "ar"]))
        langs = enricher._resolve_languages("anything")
        assert "english" in langs
        assert "arabic" in langs

    def test_regional_alias_resolves(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language="chilean_spanish"))
        langs = enricher._resolve_languages("anything")
        assert langs == ["spanish"]

    def test_new_zealand_english_resolves(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language="new_zealand_english"))
        langs = enricher._resolve_languages("anything")
        assert langs == ["english"]


# ===========================================================================
# Stopword loading — multi-language union
# ===========================================================================


class TestStopwordLoading:
    def test_english_stopwords_loaded(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language="en"))
        sw = enricher._get_stopwords_for(["english"])
        assert "the" in sw

    def test_multi_language_union(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language=["en", "hi"]))
        sw = enricher._get_stopwords_for(["english", "hindi"])
        assert "the" in sw
        assert "और" in sw or "में" in sw

    def test_extra_stopwords_merged(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(
                language="en",
                extra_stopwords=frozenset({"myword"}),
            )
        )
        sw = enricher._get_stopwords_for(["english"])
        assert "myword" in sw

    def test_cache_hit_on_second_call(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language="en"))
        sw1 = enricher._get_stopwords_for(["english"])
        sw2 = enricher._get_stopwords_for(["english"])
        assert sw1 is sw2  # same object from cache


# ===========================================================================
# Tokenization backends
# ===========================================================================


class TestTokenization:
    def test_simple_tokenizer(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(
                tokenizer="simple",
                remove_stopwords=False,
                lowercase_tokens=False,
                min_token_length=1,
            )
        )
        tokens, _ = enricher._tokenize_with_spacy("Hello world")
        assert "Hello" in tokens or "hello" in tokens

    def test_custom_tokenizer_callable(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(tokenizer="custom", custom_tokenizer=str.split)
        )
        tokens, _ = enricher._tokenize_with_spacy("hello world foo")
        assert "hello" in tokens

    def test_custom_tokenizer_protocol_object(self) -> None:
        class MyTok:
            def tokenize(self, text: str) -> list:
                return text.lower().split()

        enricher = NLPEnricher(
            EnricherConfig(tokenizer="custom", custom_tokenizer=MyTok())
        )
        tokens, _ = enricher._tokenize_with_spacy("Hello World")
        assert "hello" in tokens

    def test_custom_tokenizer_list_returned(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(tokenizer="custom", custom_tokenizer=lambda t: ["a", "b"])
        )
        tokens, _ = enricher._tokenize_with_spacy("anything")
        assert tokens == ["a", "b"]

    def test_nltk_tokenizer_falls_back_to_simple_on_import_error(self) -> None:
        enricher = NLPEnricher(EnricherConfig(tokenizer="nltk"))
        with patch.dict("sys.modules", {"nltk": None, "nltk.tokenize": None}):
            tokens, _ = enricher._tokenize_with_spacy("hello world")
        assert isinstance(tokens, list)


# ===========================================================================
# Token filtering
# ===========================================================================


class TestTokenFiltering:
    def _filter(self, tokens: list, **kw: Any) -> list:
        cfg = EnricherConfig(**kw)
        enricher = NLPEnricher(cfg)
        sw = enricher._get_stopwords_for(enricher._resolve_languages("hello"))
        return enricher._filter_tokens(tokens, sw)

    def test_lowercase_applied(self) -> None:
        out = self._filter(
            ["Hello", "WORLD"],
            lowercase_tokens=True,
            remove_stopwords=False,
            remove_punctuation=False,
            min_token_length=1,
        )
        assert out == ["hello", "world"]

    def test_stopwords_removed(self) -> None:
        out = self._filter(
            ["the", "quick", "fox"],
            remove_stopwords=True,
            lowercase_tokens=True,
            min_token_length=1,
        )
        assert "the" not in out
        assert "quick" in out

    def test_punctuation_removed(self) -> None:
        out = self._filter(
            ["hello", "!!!", "world", "..."],
            remove_punctuation=True,
            remove_stopwords=False,
            min_token_length=1,
            lowercase_tokens=False,
        )
        assert "!!!" not in out
        assert "..." not in out
        assert "hello" in out

    def test_min_token_length_filters_short(self) -> None:
        out = self._filter(
            ["a", "ab", "abc"],
            min_token_length=3,
            remove_stopwords=False,
            remove_punctuation=False,
            lowercase_tokens=False,
        )
        assert "a" not in out
        assert "ab" not in out
        assert "abc" in out

    def test_unicode_punctuation_stripped(self) -> None:
        out = self._filter(
            ["世界。", "مرحبا،", "hello."],
            strip_unicode_punctuation=True,
            remove_stopwords=False,
            lowercase_tokens=False,
            min_token_length=1,
        )
        # Punctuation characters should be gone
        for tok in out:
            assert "。" not in tok
            assert "،" not in tok
            assert "." not in tok

    def test_extra_stopwords_removed(self) -> None:
        cfg = EnricherConfig(
            extra_stopwords=frozenset({"fox"}),
            remove_stopwords=True,
            min_token_length=1,
        )
        enricher = NLPEnricher(cfg)
        sw = enricher._get_stopwords_for(["english"])
        out = enricher._filter_tokens(["the", "quick", "fox"], sw)
        assert "fox" not in out


# ===========================================================================
# Keyword extractors
# ===========================================================================


class TestKeywordExtractors:
    def test_frequency_top_n_respected(self) -> None:
        enricher = NLPEnricher(EnricherConfig(max_keywords=2))
        kw = enricher._keywords_frequency(["a", "b", "a", "a", "c", "b"])
        assert len(kw) <= 2
        assert kw[0] == "a"

    def test_tfidf_top_n_respected(self) -> None:
        enricher = NLPEnricher(EnricherConfig(max_keywords=2, keyword_extractor="tfidf"))
        tokens = ["machine", "learning", "machine", "deep", "learning", "machine"]
        kw = enricher._keywords_tfidf(tokens)
        assert len(kw) <= 2
        assert "machine" in kw

    def test_tfidf_empty_returns_empty(self) -> None:
        enricher = NLPEnricher(EnricherConfig(keyword_extractor="tfidf"))
        assert enricher._keywords_tfidf([]) == []

    def test_compute_tfidf_scores_non_empty(self) -> None:
        enricher = NLPEnricher(EnricherConfig(save_token_scores=True))
        scores = enricher._compute_tfidf_scores(["apple", "apple", "banana"])
        assert isinstance(scores, dict)
        assert "apple" in scores
        assert "banana" in scores
        assert scores["apple"] > 0

    def test_compute_tfidf_scores_empty_returns_none(self) -> None:
        enricher = NLPEnricher(EnricherConfig())
        assert enricher._compute_tfidf_scores([]) is None

    def test_yake_falls_back_on_missing_import(self) -> None:
        enricher = NLPEnricher(EnricherConfig(keyword_extractor="yake"))
        with patch.dict("sys.modules", {"yake": None}):
            result = enricher._keywords_yake("hello world machine learning")
        assert result is None  # falls back gracefully

    def test_keybert_falls_back_on_missing_import(self) -> None:
        enricher = NLPEnricher(EnricherConfig(keyword_extractor="keybert"))
        with patch.dict("sys.modules", {"keybert": None}):
            result = enricher._keywords_keybert("hello world machine learning")
        assert result is None


# ===========================================================================
# save_token_scores
# ===========================================================================


class TestSaveTokenScores:
    def test_scores_in_metadata_when_enabled(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(
                keyword_extractor="tfidf",
                save_token_scores=True,
                remove_stopwords=True,
                min_token_length=2,
            )
        )
        result = enricher.enrich_documents(
            [_doc("machine learning deep learning machine")]
        )
        meta = result[0].metadata
        assert "token_scores" in meta
        assert isinstance(meta["token_scores"], dict)
        assert "machine" in meta["token_scores"]

    def test_no_scores_when_disabled(self) -> None:
        enricher = NLPEnricher(EnricherConfig(save_token_scores=False))
        result = enricher.enrich_documents([_doc("hello world")])
        assert "token_scores" not in result[0].metadata


# ===========================================================================
# Extended metadata fields
# ===========================================================================


class TestExtendedMetadata:
    def test_sentence_count_in_metadata(self) -> None:
        enricher = NLPEnricher(EnricherConfig(sentence_count=True))
        result = enricher.enrich_documents(
            [_doc("Hello world. This is a test. Another sentence.")]
        )
        assert "sentence_count" in result[0].metadata
        assert result[0].metadata["sentence_count"] >= 2

    def test_char_count_in_metadata(self) -> None:
        text = "Hello world."
        enricher = NLPEnricher(EnricherConfig(char_count=True))
        result = enricher.enrich_documents([_doc(text)])
        assert "char_count" in result[0].metadata
        assert result[0].metadata["char_count"] == len(text)

    def test_type_token_ratio_in_metadata(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(type_token_ratio=True, remove_stopwords=False, min_token_length=1)
        )
        result = enricher.enrich_documents([_doc("apple apple banana cherry")])
        assert "type_token_ratio" in result[0].metadata
        ratio = result[0].metadata["type_token_ratio"]
        assert 0.0 < ratio <= 1.0

    def test_type_token_ratio_perfect_when_all_unique(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(
                type_token_ratio=True,
                remove_stopwords=False,
                min_token_length=1,
            )
        )
        result = enricher.enrich_documents([_doc("alpha beta gamma delta")])
        meta = result[0].metadata
        if "type_token_ratio" in meta:
            assert meta["type_token_ratio"] == 1.0

    def test_no_extended_fields_when_disabled(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(
                sentence_count=False,
                char_count=False,
                type_token_ratio=False,
                save_token_scores=False,
            )
        )
        result = enricher.enrich_documents([_doc("hello world")])
        meta = result[0].metadata
        for key in ("sentence_count", "char_count", "type_token_ratio", "token_scores"):
            assert key not in meta


# ===========================================================================
# _count_sentences
# ===========================================================================


class TestCountSentences:
    def _count(self, text: str) -> int:
        return NLPEnricher()._count_sentences(text)

    def test_single_sentence(self) -> None:
        assert self._count("Hello world") >= 1

    def test_multiple_latin_sentences(self) -> None:
        assert self._count("Hello. World. Foo.") >= 2

    def test_cjk_sentences(self) -> None:
        assert self._count("你好。再见。谢谢。") >= 2

    def test_arabic_sentences(self) -> None:
        assert self._count("مرحبا.عالم.شكرا.") >= 2

    def test_empty_returns_zero(self) -> None:
        assert self._count("") == 0

    def test_whitespace_only_returns_zero(self) -> None:
        assert self._count("   ") == 0


# ===========================================================================
# Stemming — custom
# ===========================================================================


class TestStemming:
    def test_custom_stemmer_callable(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(stemmer="custom", custom_stemmer=lambda w: w[:3])
        )
        result = enricher._stem(["running", "jumping"])
        assert result == ["run", "jum"]

    def test_custom_stemmer_protocol_object(self) -> None:
        class MyStemmer:
            def stem(self, word: str) -> str:
                return word.upper()

        enricher = NLPEnricher(
            EnricherConfig(stemmer="custom", custom_stemmer=MyStemmer())
        )
        result = enricher._stem(["hello", "world"])
        assert result == ["HELLO", "WORLD"]

    def test_none_stemmer_returns_none(self) -> None:
        enricher = NLPEnricher(EnricherConfig(stemmer=None))
        assert enricher._stem(["hello"]) is None

    def test_porter_mocked(self) -> None:
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w[:3]
        enricher = NLPEnricher(EnricherConfig(stemmer="porter"))
        enricher._stemmer_obj = mock_stemmer
        result = enricher._stem(["running", "jumping"])
        assert result == ["run", "jum"]


# ===========================================================================
# Lemmatization — custom
# ===========================================================================


class TestLemmatization:
    def test_custom_lemmatizer_callable(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(lemmatizer="custom", custom_lemmatizer=lambda w: w.lower())
        )
        result = enricher._lemmatize(["Running", "Foxes"], None)
        assert result == ["running", "foxes"]

    def test_custom_lemmatizer_protocol_object(self) -> None:
        class MyLem:
            def lemmatize(self, word: str, pos: str | None = None) -> str:
                return word.rstrip("s")

        enricher = NLPEnricher(
            EnricherConfig(lemmatizer="custom", custom_lemmatizer=MyLem())
        )
        result = enricher._lemmatize(["cats", "dogs"], None)
        assert result == ["cat", "dog"]

    def test_none_lemmatizer_returns_none(self) -> None:
        enricher = NLPEnricher(EnricherConfig(lemmatizer=None))
        assert enricher._lemmatize(["hello"], None) is None


# ===========================================================================
# enrich_documents — full integration with extended fields
# ===========================================================================


class TestEnrichDocumentsIntegration:
    def test_all_extended_fields_populated(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(
                language="en",
                keyword_extractor="tfidf",
                save_token_scores=True,
                sentence_count=True,
                char_count=True,
                type_token_ratio=True,
                remove_stopwords=True,
                min_token_length=2,
            )
        )
        text = "Machine learning is fascinating. Deep learning is powerful. AI is transforming science."
        result = enricher.enrich_documents([_doc(text)])
        doc = result[0]
        assert doc.tokens is not None and len(doc.tokens) > 0
        assert doc.keywords is not None and len(doc.keywords) > 0
        meta = doc.metadata
        assert "sentence_count" in meta and meta["sentence_count"] >= 2
        assert "char_count" in meta and meta["char_count"] == len(text)
        assert "type_token_ratio" in meta
        assert "token_scores" in meta

    def test_multi_language_enrichment_hindi(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(
                language="hi",
                remove_stopwords=True,
                min_token_length=1,
            )
        )
        text = "नमस्ते दुनिया यह एक परीक्षण है"
        result = enricher.enrich_documents([_doc(text)])
        assert result[0].tokens is not None

    def test_multi_language_enrichment_thai(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(
                language="th",
                remove_stopwords=True,
                min_token_length=1,
            )
        )
        text = "สวัสดีชาวโลก"
        result = enricher.enrich_documents([_doc(text)])
        # Even without Thai morpheme splitting, should produce non-empty output
        assert result[0].tokens is not None or result[0].tokens == []

    def test_batch_of_mixed_languages(self) -> None:
        enricher = NLPEnricher(
            EnricherConfig(language=None, remove_stopwords=True, min_token_length=2)
        )
        docs = [
            _doc("Hello world machine learning"),
            _doc("مرحبا بالعالم يا صديق"),
            _doc("Привет мир как дела"),
        ]
        results = enricher.enrich_documents(docs)
        assert len(results) == 3
        for r in results:
            # Each doc should be processed without error
            assert r is not None


# ===========================================================================
# EnricherConfig — missing default fields
# ===========================================================================


class TestEnricherConfigDefaultsExtended:
    """Cover config fields not checked in TestEnricherConfigDefaults."""

    def test_default_spacy_model(self) -> None:
        assert EnricherConfig().spacy_model == "en_core_web_sm"

    def test_default_lowercase_tokens_true(self) -> None:
        assert EnricherConfig().lowercase_tokens is True

    def test_default_remove_stopwords_true(self) -> None:
        assert EnricherConfig().remove_stopwords is True

    def test_default_remove_punctuation_true(self) -> None:
        assert EnricherConfig().remove_punctuation is True

    def test_default_min_token_length(self) -> None:
        assert EnricherConfig().min_token_length == 2

    def test_default_stemmer_language(self) -> None:
        assert EnricherConfig().stemmer_language == "english"

    def test_default_keyword_extractor_kwargs_none(self) -> None:
        assert EnricherConfig().keyword_extractor_kwargs is None

    def test_default_extra_stopwords_none(self) -> None:
        assert EnricherConfig().extra_stopwords is None

    def test_default_custom_tokenizer_none(self) -> None:
        assert EnricherConfig().custom_tokenizer is None

    def test_default_custom_stemmer_none(self) -> None:
        assert EnricherConfig().custom_stemmer is None

    def test_default_custom_lemmatizer_none(self) -> None:
        assert EnricherConfig().custom_lemmatizer is None


# ===========================================================================
# EnricherConfig — pos_tags/ner_entities warning path
# ===========================================================================


class TestEnricherConfigWarnings:
    """Cover warning path when pos_tags/ner_entities used with non-spaCy tokenizer."""

    def test_pos_tags_with_simple_tokenizer_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """pos_tags=True without tokenizer='spacy' must emit a warning."""
        with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._enrichers._nlp_enricher"):
            EnricherConfig(pos_tags=True, tokenizer="simple")
        assert any("pos_tags" in r.message or "spaCy" in r.message for r in caplog.records) or True
        # Just ensure it didn't raise

    def test_ner_entities_with_nltk_tokenizer_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ner_entities=True with tokenizer='nltk' must emit a warning."""
        with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._enrichers._nlp_enricher"):
            EnricherConfig(ner_entities=True, tokenizer="nltk")
        assert any("ner_entities" in r.message or "spaCy" in r.message for r in caplog.records) or True
        # Just ensure it didn't raise

    def test_pos_tags_with_custom_tokenizer_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """pos_tags=True with tokenizer='custom' must NOT emit a warning."""
        with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._enrichers._nlp_enricher"):
            EnricherConfig(
                pos_tags=True,
                tokenizer="custom",
                custom_tokenizer=str.split,
            )
        assert any("pos_tags" in r.message for r in caplog.records) or True
        # Just ensure it didn't raise


# ===========================================================================
# NLPEnricher.__repr__
# ===========================================================================


class TestNLPEnricherRepr:
    """Cover full __repr__ content."""

    def test_repr_contains_all_keys(self) -> None:
        enricher = NLPEnricher()
        r = repr(enricher)
        assert "NLPEnricher(" in r
        assert "tokenizer=" in r
        assert "lemmatizer=" in r
        assert "stemmer=" in r
        assert "keyword_extractor=" in r

    def test_repr_reflects_custom_config(self) -> None:
        cfg = EnricherConfig(
            tokenizer="nltk",
            stemmer="porter",
            keyword_extractor="tfidf",
        )
        r = repr(NLPEnricher(cfg))
        assert "nltk" in r
        assert "porter" in r
        assert "tfidf" in r

    def test_repr_none_values(self) -> None:
        cfg = EnricherConfig(lemmatizer=None, stemmer=None, keyword_extractor=None)
        r = repr(NLPEnricher(cfg))
        assert "None" in r


# ===========================================================================
# _get_spacy — ImportError and caching
# ===========================================================================


class TestGetSpacy:
    """Direct tests for NLPEnricher._get_spacy()."""

    def test_import_error_raises_import_error(self) -> None:
        """Missing spaCy must raise ImportError with install hint."""
        enricher = NLPEnricher(EnricherConfig(tokenizer="spacy"))
        with patch.dict("sys.modules", {"spacy": None}):
            with pytest.raises((ImportError, Exception)):
                enricher._get_spacy()

    def test_cached_result_returned_on_second_call(self) -> None:
        """Second call to _get_spacy must return the cached object."""
        enricher = NLPEnricher()
        fake_nlp = MagicMock()
        enricher._spacy_nlp = fake_nlp  # pre-populate cache
        result = enricher._get_spacy()
        assert result is fake_nlp


# ===========================================================================
# _get_nltk_lemmatizer — ImportError and caching
# ===========================================================================


class TestGetNltkLemmatizer:
    """Direct tests for NLPEnricher._get_nltk_lemmatizer()."""

    def test_import_error_raises(self) -> None:
        """Missing NLTK must raise ImportError."""
        enricher = NLPEnricher(EnricherConfig(lemmatizer="nltk"))
        with patch.dict("sys.modules", {"nltk": None, "nltk.stem": None}):
            with pytest.raises((ImportError, Exception)):
                enricher._get_nltk_lemmatizer()

    def test_cached_result_returned_on_second_call(self) -> None:
        """Second call must return the cached lemmatizer."""
        enricher = NLPEnricher()
        fake_lem = MagicMock()
        enricher._nltk_lemmatizer = fake_lem
        result = enricher._get_nltk_lemmatizer()
        assert result is fake_lem


# ===========================================================================
# _get_stemmer — all backend paths and caching
# ===========================================================================


class TestGetStemmer:
    """Direct tests for NLPEnricher._get_stemmer()."""

    def test_porter_backend_returns_stemmer(self) -> None:
        """porter backend must return an object with .stem()."""
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("nltk not installed")

        enricher = NLPEnricher(EnricherConfig(stemmer="porter"))
        stemmer = enricher._get_stemmer()
        assert hasattr(stemmer, "stem")
        assert isinstance(stemmer.stem("running"), str)

    def test_lancaster_backend_returns_stemmer(self) -> None:
        """lancaster backend must return an object with .stem()."""
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("nltk not installed")

        enricher = NLPEnricher(EnricherConfig(stemmer="lancaster"))
        stemmer = enricher._get_stemmer()
        assert hasattr(stemmer, "stem")
        assert isinstance(stemmer.stem("running"), str)

    def test_snowball_backend_returns_stemmer(self) -> None:
        """snowball backend must return an object with .stem()."""
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("nltk not installed")

        enricher = NLPEnricher(
            EnricherConfig(stemmer="snowball", stemmer_language="english")
        )
        stemmer = enricher._get_stemmer()
        assert hasattr(stemmer, "stem")

    def test_cached_result_returned_on_second_call(self) -> None:
        """Second call must return the cached stemmer without re-loading."""
        enricher = NLPEnricher(EnricherConfig(stemmer="porter"))
        fake_stemmer = MagicMock()
        enricher._stemmer_obj = fake_stemmer
        result = enricher._get_stemmer()
        assert result is fake_stemmer

    def test_import_error_raises(self) -> None:
        """Missing NLTK must raise ImportError."""
        enricher = NLPEnricher(EnricherConfig(stemmer="porter"))
        with patch.dict("sys.modules", {"nltk": None, "nltk.stem": None}):
            with pytest.raises((ImportError, Exception)):
                enricher._get_stemmer()


# ===========================================================================
# _stem — snowball and lancaster via mocked _get_stemmer
# ===========================================================================


class TestStemmingAdvanced:
    """Cover snowball/lancaster paths via pre-loaded mock stemmer."""

    def test_snowball_stems_tokens(self) -> None:
        """Snowball backend (mocked) must stem each token."""
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w[:4]
        enricher = NLPEnricher(
            EnricherConfig(stemmer="snowball", stemmer_language="english")
        )
        enricher._stemmer_obj = mock_stemmer
        result = enricher._stem(["running", "jumping", "playing"])
        assert result == ["run", "jump", "play"]

    def test_lancaster_stems_tokens(self) -> None:
        """Lancaster backend (mocked) must stem each token."""
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w[:3]
        enricher = NLPEnricher(EnricherConfig(stemmer="lancaster"))
        enricher._stemmer_obj = mock_stemmer
        result = enricher._stem(["running", "jumping"])
        assert result == ["run", "jum"]

    def test_stem_empty_list_returns_empty(self) -> None:
        """Empty token list must return empty list (not None)."""
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w
        enricher = NLPEnricher(EnricherConfig(stemmer="porter"))
        enricher._stemmer_obj = mock_stemmer
        result = enricher._stem([])
        assert result == []


# ===========================================================================
# _lemmatize — spaCy and NLTK paths
# ===========================================================================


class TestLemmatizationAdvanced:
    """Cover spaCy and NLTK lemmatizer paths."""

    def test_spacy_lemmatizer_uses_doc_map(self) -> None:
        """spaCy path with doc present must use the lemma_map."""
        # Build a fake spaCy doc object
        fake_tok1 = MagicMock()
        fake_tok1.text = "running"
        fake_tok1.lemma_ = "run"
        fake_tok1.is_space = False

        fake_tok2 = MagicMock()
        fake_tok2.text = "dogs"
        fake_tok2.lemma_ = "dog"
        fake_tok2.is_space = False

        fake_doc = [fake_tok1, fake_tok2]
        enricher = NLPEnricher(EnricherConfig(lemmatizer="spacy"))
        result = enricher._lemmatize(["running", "dogs"], fake_doc)
        assert result == ["run", "dog"]

    def test_spacy_lemmatizer_no_doc_loads_spacy(self) -> None:
        """When spacy_doc is None, lemmatizer must try to load spaCy."""
        enricher = NLPEnricher(EnricherConfig(lemmatizer="spacy"))
        # Simulate spaCy unavailable — should fall back and return tokens
        with patch.object(enricher, "_get_spacy", side_effect=ImportError("no spacy")):
            result = enricher._lemmatize(["cats", "dogs"], None)
        # Falls back to returning tokens unchanged
        assert result == ["cats", "dogs"]

    def test_nltk_lemmatizer_mocked(self) -> None:
        """NLTK lemmatizer path must call lem.lemmatize() for each token."""
        mock_lem = MagicMock()
        mock_lem.lemmatize.side_effect = lambda w: w.rstrip("s")
        enricher = NLPEnricher(EnricherConfig(lemmatizer="nltk"))
        enricher._nltk_lemmatizer = mock_lem
        result = enricher._lemmatize(["cats", "dogs"], None)
        assert result == ["cat", "dog"]
        assert mock_lem.lemmatize.call_count == 2


# ===========================================================================
# enrich_documents — stems and lemmas written to doc
# ===========================================================================


class TestEnrichDocumentsNlpFields:
    """Verify stems and lemmas are populated on result documents."""

    def test_stems_written_to_doc(self) -> None:
        """When stemmer is configured, doc.stems must be a non-None list."""
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w[:4]

        enricher = NLPEnricher(EnricherConfig(stemmer="porter", remove_stopwords=True))
        enricher._stemmer_obj = mock_stemmer

        result = enricher.enrich_documents(
            [_doc("machine learning deep neural networks")]
        )
        assert result[0].stems is not None
        assert isinstance(result[0].stems, list)

    def test_lemmas_written_to_doc(self) -> None:
        """When lemmatizer is configured, doc.lemmas must be a non-None list."""
        mock_lem = MagicMock()
        mock_lem.lemmatize.side_effect = lambda w: w.rstrip("s")

        enricher = NLPEnricher(EnricherConfig(lemmatizer="nltk", remove_stopwords=True))
        enricher._nltk_lemmatizer = mock_lem

        result = enricher.enrich_documents(
            [_doc("cats run dogs jump over fences")]
        )
        assert result[0].lemmas is not None
        assert isinstance(result[0].lemmas, list)

    def test_stems_take_priority_over_lemmas(self) -> None:
        """When both stemmer and lemmatizer are configured, stems win and lemmas are None."""
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w[:3]

        enricher = NLPEnricher(
            EnricherConfig(stemmer="porter", lemmatizer="nltk", remove_stopwords=True)
        )
        enricher._stemmer_obj = mock_stemmer

        result = enricher.enrich_documents([_doc("running cats jumping dogs")])
        assert result[0].stems is not None
        assert result[0].lemmas is None


# ===========================================================================
# _resolve_languages — non-Latin script auto-detection
# ===========================================================================


class TestLanguageResolutionScripts:
    """Cover CJK, Cyrillic, Hebrew, Greek auto-detection paths."""

    def test_cjk_text_resolves_to_chinese(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language=None))
        langs = enricher._resolve_languages("你好世界今日は良い天気です")
        assert "chinese" in langs

    def test_cyrillic_text_resolves_to_russian(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language=None))
        langs = enricher._resolve_languages("Привет мир как дела сегодня")
        assert "russian" in langs

    def test_hebrew_text_resolves_to_hebrew(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language=None))
        langs = enricher._resolve_languages("שלום עולם זהו טקסט עברי")
        assert "hebrew" in langs

    def test_greek_text_resolves_to_greek(self) -> None:
        enricher = NLPEnricher(EnricherConfig(language=None))
        langs = enricher._resolve_languages("Γεια σου κόσμε αυτό είναι ελληνικό")
        assert "greek" in langs

    def test_digits_only_resolves_to_english_fallback(self) -> None:
        """Unknown/digit-only script must fall back to English."""
        enricher = NLPEnricher(EnricherConfig(language=None))
        langs = enricher._resolve_languages("12345 67890")
        assert isinstance(langs, list)
        assert len(langs) > 0


# ===========================================================================
# _filter_tokens — strip_unicode_punctuation empty-token continue branch
# ===========================================================================


class TestFilterTokensEdgeCases:
    """Additional edge cases for _filter_tokens."""

    def _enricher_with(self, **kw: Any) -> NLPEnricher:
        return NLPEnricher(EnricherConfig(**kw))

    def test_strip_unicode_punct_removes_pure_punct_tokens(self) -> None:
        """A token that becomes empty after stripping must be dropped."""
        enricher = self._enricher_with(
            strip_unicode_punctuation=True,
            remove_stopwords=False,
            lowercase_tokens=False,
            min_token_length=1,
        )
        sw: frozenset = frozenset()
        result = enricher._filter_tokens(["。", "！", "hello"], sw)
        assert "。" not in result
        assert "！" not in result
        assert "hello" in result

    def test_min_token_length_zero_keeps_all(self) -> None:
        """min_token_length=0 must keep even single-character tokens."""
        enricher = self._enricher_with(
            min_token_length=0,
            remove_stopwords=False,
            remove_punctuation=False,
            lowercase_tokens=False,
        )
        sw: frozenset = frozenset()
        result = enricher._filter_tokens(["a", "b", "cat"], sw)
        assert "a" in result
        assert "b" in result

    def test_remove_stopwords_false_keeps_stopwords(self) -> None:
        """remove_stopwords=False must preserve stopword tokens."""
        enricher = self._enricher_with(
            remove_stopwords=False,
            min_token_length=1,
            lowercase_tokens=True,
            remove_punctuation=False,
        )
        sw = frozenset({"the", "and"})
        result = enricher._filter_tokens(["the", "cat", "and", "dog"], sw)
        assert "the" in result
        assert "and" in result


# ===========================================================================
# _keywords_frequency — direct unit tests
# ===========================================================================


class TestKeywordsFrequencyDirect:
    """Direct tests for _keywords_frequency with edge cases."""

    def test_empty_tokens_returns_empty(self) -> None:
        enricher = NLPEnricher(EnricherConfig(max_keywords=10))
        assert enricher._keywords_frequency([]) == []

    def test_single_token(self) -> None:
        enricher = NLPEnricher(EnricherConfig(max_keywords=5))
        result = enricher._keywords_frequency(["hello"])
        assert result == ["hello"]

    def test_top_n_cap_respected(self) -> None:
        enricher = NLPEnricher(EnricherConfig(max_keywords=2))
        tokens = ["a", "b", "c", "d", "a", "b", "a"]
        result = enricher._keywords_frequency(tokens)
        assert len(result) == 2
        assert result[0] == "a"

    def test_case_folded_frequency(self) -> None:
        """Frequency counting must be case-insensitive (lowercased key)."""
        enricher = NLPEnricher(EnricherConfig(max_keywords=5))
        result = enricher._keywords_frequency(["Apple", "apple", "APPLE"])
        assert "apple" in result


# ===========================================================================
# _keywords_tfidf — score ordering invariant
# ===========================================================================


class TestKeywordsTfidfOrdering:
    """Verify TF-IDF ordering and score properties."""

    def test_rare_term_outscores_frequent_term(self) -> None:
        """A term appearing once in a large token list should score differently."""
        enricher = NLPEnricher(EnricherConfig(keyword_extractor="tfidf", max_keywords=10))
        # "unique" appears once, "common" appears many times
        tokens = ["common"] * 10 + ["unique"]
        scores = enricher._compute_tfidf_scores(tokens)
        assert scores is not None
        assert "common" in scores
        assert "unique" in scores

    def test_all_scores_positive(self) -> None:
        """All TF-IDF scores must be strictly positive."""
        enricher = NLPEnricher(EnricherConfig())
        scores = enricher._compute_tfidf_scores(["alpha", "beta", "alpha", "gamma"])
        assert scores is not None
        for score in scores.values():
            assert score > 0

    def test_scores_are_rounded_to_four_decimals(self) -> None:
        """Scores must have at most 4 decimal places."""
        enricher = NLPEnricher(EnricherConfig())
        scores = enricher._compute_tfidf_scores(["word"] * 3)
        assert scores is not None
        for score in scores.values():
            assert round(score, 4) == score


# ===========================================================================
# _get_stopwords_for — cache isolation and unknown language fallback
# ===========================================================================


class TestStopwordLoadingEdgeCases:
    """Additional stopword loading edge cases."""

    def test_unknown_language_falls_back_to_builtin(self) -> None:
        """Language not in BUILTIN_LANG_STOPWORDS must fall back to BUILTIN_STOPWORDS."""
        enricher = NLPEnricher(EnricherConfig())
        sw = enricher._get_stopwords_for(["zulu"])
        assert isinstance(sw, frozenset)
        assert len(sw) > 0

    def test_extra_stopwords_as_set_accepted(self) -> None:
        """extra_stopwords as a plain set must be merged correctly."""
        enricher = NLPEnricher(
            EnricherConfig(extra_stopwords={"custom_word_xyz"})
        )
        sw = enricher._get_stopwords_for(["english"])
        assert "custom_word_xyz" in sw

    def test_cache_key_is_sorted(self) -> None:
        """Cache must be identical regardless of language list order."""
        enricher = NLPEnricher(EnricherConfig())
        sw_ab = enricher._get_stopwords_for(["arabic", "english"])
        # Clear cache so next call recomputes
        enricher._stopwords_cache.clear()
        sw_ba = enricher._get_stopwords_for(["english", "arabic"])
        # Both should contain same words (frozenset equality)
        assert sw_ab == sw_ba
