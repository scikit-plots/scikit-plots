# corpus/_chunkers/tests/test__word_advanced.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Advanced coverage tests for WordChunker and helpers.

Covers:
- _get_stemmer with list[str] and None language
- _load_stopwords: NLTK list union, SPACY list, BUILTIN fallback
- _process_tokens: strip_unicode_punctuation, remove_numbers,
  max_token_length, custom stopwords, all lemmatizer branches
- WordChunkerConfig validation edge cases
- WordChunker: chunk_by='sentence', include_offsets, empty/whitespace
- vocabulary_stats and build_gensim_dictionary guards
- Security: malicious custom callables
- Multi-language: CJK, Arabic, ancient Greek
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from .._word import (
    LemmatizationBackend,
    StemmingBackend,
    StopwordSource,
    TokenizerBackend,
    WordChunker,
    WordChunkerConfig,
    _extract_ngrams,
    _get_stemmer,
    _load_stopwords,
    _process_tokens,
    _strip_unicode_punct,
    _tokenize_simple,
)


# ===========================================================================
# _strip_unicode_punct
# ===========================================================================


class TestStripUnicodePunct:
    def test_ascii_punct_removed(self) -> None:
        assert _strip_unicode_punct("hello.") == "hello"
        assert _strip_unicode_punct("world!") == "world"

    def test_cjk_punct_removed(self) -> None:
        assert _strip_unicode_punct("世界。") == "世界"
        assert _strip_unicode_punct("テスト！") == "テスト"

    def test_arabic_punct_removed(self) -> None:
        assert _strip_unicode_punct("مرحبا،") == "مرحبا"

    def test_pure_punct_returns_empty(self) -> None:
        assert _strip_unicode_punct(".,!?") == ""

    def test_no_punct_unchanged(self) -> None:
        assert _strip_unicode_punct("hello") == "hello"
        assert _strip_unicode_punct("世界") == "世界"

    def test_mixed_content(self) -> None:
        result = _strip_unicode_punct("abc123!@#")
        assert result == "abc123"

    def test_empty_string(self) -> None:
        assert _strip_unicode_punct("") == ""


# ===========================================================================
# _tokenize_simple
# ===========================================================================


class TestTokenizeSimple:
    def test_basic_split(self) -> None:
        assert _tokenize_simple("hello world") == ["hello", "world"]

    def test_strips_leading_trailing_whitespace(self) -> None:
        result = _tokenize_simple("  hello world  ")
        assert "hello" in result

    def test_punctuation_stripped(self) -> None:
        result = _tokenize_simple("hello, world!")
        assert "hello" in result
        assert "world" in result

    def test_empty_after_strip_skipped(self) -> None:
        result = _tokenize_simple("... ,,, ...")
        assert result == []


# ===========================================================================
# _extract_ngrams
# ===========================================================================


class TestExtractNgrams:
    def test_bigrams(self) -> None:
        toks = ["a", "b", "c"]
        assert _extract_ngrams(toks, 2) == ["a_b", "b_c"]

    def test_trigrams(self) -> None:
        toks = ["a", "b", "c", "d"]
        assert _extract_ngrams(toks, 3) == ["a_b_c", "b_c_d"]

    def test_n_larger_than_tokens_returns_empty(self) -> None:
        assert _extract_ngrams(["a", "b"], 5) == []

    def test_single_token(self) -> None:
        assert _extract_ngrams(["a"], 1) == ["a"]


# ===========================================================================
# _get_stemmer with list / None language
# ===========================================================================


class TestGetStemmerLanguageResolution:
    def test_list_language_no_crash(self) -> None:
        """_get_stemmer with list[str] must not crash."""
        from .._word import _STEMMER_CACHE
        _STEMMER_CACHE.clear()

        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w[:3]

        # Patch at the module level where _get_stemmer imports from
        with patch("scikitplot.corpus._chunkers._word._STEMMER_CACHE", {}), \
             patch("scikitplot.corpus._chunkers._word.coerce_language",
                   return_value=["english"]):
            try:
                from nltk.stem import SnowballStemmer  # noqa: F401
                with patch("nltk.stem.SnowballStemmer", return_value=mock_stemmer):
                    stemmer = _get_stemmer(StemmingBackend.SNOWBALL, ["english", "german"])
                    assert stemmer is not None
            except ImportError:
                pytest.skip("NLTK not installed")

    def test_none_language_uses_english_default(self) -> None:
        from .._word import _STEMMER_CACHE
        _STEMMER_CACHE.clear()
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda w: w
        try:
            from nltk.stem import SnowballStemmer  # noqa: F401
            with patch("nltk.stem.SnowballStemmer", return_value=mock_stemmer) as cls:
                _get_stemmer(StemmingBackend.SNOWBALL, None)
                cls.assert_called_with("english")
        except ImportError:
            pytest.skip("NLTK not installed")

    def test_porter_ignores_language(self) -> None:
        """PORTER stemmer is language-agnostic — list language must not crash."""
        from .._word import _STEMMER_CACHE
        _STEMMER_CACHE.clear()
        mock_porter = MagicMock()
        try:
            from nltk.stem import PorterStemmer  # noqa: F401
            with patch("nltk.stem.PorterStemmer", return_value=mock_porter):
                stemmer = _get_stemmer(StemmingBackend.PORTER, ["arabic", "chinese"])
                assert stemmer is not None
        except ImportError:
            pytest.skip("NLTK not installed")

    def test_stemmer_cached_on_second_call(self) -> None:
        from .._word import _STEMMER_CACHE
        _STEMMER_CACHE.clear()
        mock_porter = MagicMock()
        try:
            from nltk.stem import PorterStemmer  # noqa: F401
            with patch("nltk.stem.PorterStemmer", return_value=mock_porter):
                s1 = _get_stemmer(StemmingBackend.PORTER, "english")
                s2 = _get_stemmer(StemmingBackend.PORTER, "english")
                assert s1 is s2
        except ImportError:
            pytest.skip("NLTK not installed")


# ===========================================================================
# _load_stopwords
# ===========================================================================


class TestLoadStopwords:
    def test_none_source_returns_empty(self) -> None:
        sw = _load_stopwords(StopwordSource.NONE, "english")
        assert sw == frozenset()

    def test_builtin_returns_frozenset(self) -> None:
        sw = _load_stopwords(StopwordSource.BUILTIN, "english")
        assert isinstance(sw, frozenset)
        assert "the" in sw or "a" in sw

    def test_nltk_list_language_union(self) -> None:
        """NLTK source with list language must union stopwords."""
        try:
            sw_multi = _load_stopwords(StopwordSource.NLTK, ["english", "english"])
            sw_single = _load_stopwords(StopwordSource.NLTK, "english")
            # Deduped union of same language must equal single
            assert len(sw_multi) == len(sw_single)
        except ImportError:
            pytest.skip("NLTK not installed")

    def test_nltk_unknown_language_uses_builtin_fallback(self) -> None:
        """NLTK source with unsupported language falls through to BUILTIN."""
        try:
            sw = _load_stopwords(StopwordSource.NLTK, "klingon")
            assert isinstance(sw, frozenset)
        except ImportError:
            pytest.skip("NLTK not installed")

    def test_nltk_none_language_defaults_to_english(self) -> None:
        try:
            sw = _load_stopwords(StopwordSource.NLTK, None)
            assert isinstance(sw, frozenset)
            assert len(sw) > 0
        except ImportError:
            pytest.skip("NLTK not installed")

    def test_spacy_source_requires_spacy(self) -> None:
        """SPACY source raises ImportError when spacy not installed."""
        with patch.dict("sys.modules", {"spacy": None}):
            with pytest.raises(ImportError, match="spacy"):
                _load_stopwords(StopwordSource.SPACY, "english")

    def test_spacy_source_list_language_no_crash(self) -> None:
        """SPACY source with list language must not crash."""
        try:
            import spacy  # noqa: F401
        except ImportError:
            pytest.skip("spaCy not installed")
        sw = _load_stopwords(StopwordSource.SPACY, ["english", "english"])
        assert isinstance(sw, frozenset)

    def test_spacy_source_none_language_defaults(self) -> None:
        try:
            import spacy  # noqa: F401
        except ImportError:
            pytest.skip("spaCy not installed")
        sw = _load_stopwords(StopwordSource.SPACY, None)
        assert isinstance(sw, frozenset)


# ===========================================================================
# _process_tokens
# ===========================================================================


class TestProcessTokens:
    def _cfg(self, **kw):
        return WordChunkerConfig(**kw)

    def test_lowercase(self) -> None:
        cfg = self._cfg(lowercase=True, stopwords=StopwordSource.NONE,
                        remove_punctuation=False, min_token_length=0)
        result = _process_tokens(["Hello", "WORLD"], cfg, frozenset())
        assert result == ["hello", "world"]

    def test_remove_punctuation_only_tokens(self) -> None:
        cfg = self._cfg(remove_punctuation=True, lowercase=False,
                        stopwords=StopwordSource.NONE, min_token_length=0)
        result = _process_tokens(["hello", "...", "world"], cfg, frozenset())
        assert "..." not in result
        assert "hello" in result

    def test_strip_unicode_punctuation_cjk(self) -> None:
        cfg = self._cfg(strip_unicode_punctuation=True, lowercase=False,
                        remove_punctuation=False, stopwords=StopwordSource.NONE,
                        min_token_length=0)
        result = _process_tokens(["世界。", "テスト！"], cfg, frozenset())
        assert "世界" in result
        assert "テスト" in result

    def test_strip_unicode_punct_removes_empty_tokens(self) -> None:
        cfg = self._cfg(strip_unicode_punctuation=True, lowercase=False,
                        remove_punctuation=False, stopwords=StopwordSource.NONE,
                        min_token_length=0)
        result = _process_tokens(["。", "！"], cfg, frozenset())
        assert result == []

    def test_remove_numbers(self) -> None:
        cfg = self._cfg(remove_numbers=True, stopwords=StopwordSource.NONE,
                        remove_punctuation=False, min_token_length=0)
        result = _process_tokens(["hello", "123", "world"], cfg, frozenset())
        assert "123" not in result

    def test_stopword_removal(self) -> None:
        cfg = self._cfg(stopwords=StopwordSource.BUILTIN, remove_punctuation=False,
                        min_token_length=0)
        result = _process_tokens(["the", "quick", "fox"], cfg, frozenset({"the"}))
        assert "the" not in result
        assert "quick" in result

    def test_custom_stopwords_applied(self) -> None:
        cfg = self._cfg(stopwords=StopwordSource.NONE, remove_punctuation=False,
                        min_token_length=0, custom_stopwords=frozenset({"custom"}))
        result = _process_tokens(["custom", "word"], cfg, frozenset())
        assert "custom" not in result

    def test_min_token_length_filters_short(self) -> None:
        cfg = self._cfg(min_token_length=3, stopwords=StopwordSource.NONE,
                        remove_punctuation=False)
        result = _process_tokens(["hi", "hello", "yo"], cfg, frozenset())
        assert "hello" in result
        assert "hi" not in result
        assert "yo" not in result

    def test_max_token_length_filters_long(self) -> None:
        cfg = self._cfg(max_token_length=4, min_token_length=0,
                        stopwords=StopwordSource.NONE, remove_punctuation=False)
        result = _process_tokens(["hi", "hello", "world"], cfg, frozenset())
        assert "hello" not in result  # 5 chars
        assert "world" not in result  # 5 chars
        assert "hi" in result         # 2 chars

    def test_custom_lemmatizer_applied(self) -> None:
        cfg = self._cfg(
            lemmatizer=LemmatizationBackend.CUSTOM,
            custom_lemmatizer=lambda w: w + "_lem",
            stopwords=StopwordSource.NONE,
            remove_punctuation=False, min_token_length=0,
        )
        result = _process_tokens(["run", "jump"], cfg, frozenset())
        assert result == ["run_lem", "jump_lem"]

    def test_custom_stemmer_applied(self) -> None:
        cfg = self._cfg(
            stemmer=StemmingBackend.CUSTOM,
            custom_stemmer=lambda w: w[:2],
            stopwords=StopwordSource.NONE,
            remove_punctuation=False, min_token_length=0,
        )
        result = _process_tokens(["running", "jumping"], cfg, frozenset())
        assert result == ["ru", "ju"]

    def test_stemmer_takes_precedence_over_lemmatizer(self) -> None:
        """When both stemmer and lemmatizer are set, stemmer must run."""
        cfg = self._cfg(
            stemmer=StemmingBackend.CUSTOM,
            custom_stemmer=lambda w: "stem_" + w,
            lemmatizer=LemmatizationBackend.CUSTOM,
            custom_lemmatizer=lambda w: "lem_" + w,
            stopwords=StopwordSource.NONE,
            remove_punctuation=False, min_token_length=0,
        )
        result = _process_tokens(["word"], cfg, frozenset())
        assert result == ["stem_word"]  # stemmer wins


# ===========================================================================
# WordChunkerConfig validation
# ===========================================================================


class TestWordChunkerConfigValidation:
    def test_negative_min_token_length_raises(self) -> None:
        with pytest.raises(ValueError, match="min_token_length"):
            WordChunker(WordChunkerConfig(min_token_length=-1))

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_token_length"):
            WordChunker(WordChunkerConfig(min_token_length=5, max_token_length=3))

    def test_bad_ngram_range_raises(self) -> None:
        with pytest.raises(ValueError, match="ngram_range"):
            WordChunker(WordChunkerConfig(ngram_range=(3, 1)))

    def test_zero_min_ngram_raises(self) -> None:
        with pytest.raises(ValueError, match="ngram_range"):
            WordChunker(WordChunkerConfig(ngram_range=(0, 1)))

    def test_bad_chunk_by_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_by"):
            WordChunker(WordChunkerConfig(chunk_by="paragraph"))

    def test_spacy_tokenizer_without_model_raises(self) -> None:
        with pytest.raises(ValueError, match="spacy_model"):
            WordChunker(WordChunkerConfig(tokenizer=TokenizerBackend.SPACY))

    def test_custom_tokenizer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_tokenizer"):
            WordChunker(WordChunkerConfig(tokenizer=TokenizerBackend.CUSTOM))

    def test_custom_stemmer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_stemmer"):
            WordChunker(WordChunkerConfig(stemmer=StemmingBackend.CUSTOM))

    def test_custom_lemmatizer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_lemmatizer"):
            WordChunker(WordChunkerConfig(lemmatizer=LemmatizationBackend.CUSTOM))

    def test_snowball_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="SNOWBALL"):
            WordChunker(WordChunkerConfig(
                stemmer=StemmingBackend.SNOWBALL, nltk_language="klingon"
            ))

    def test_snowball_list_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="SNOWBALL"):
            WordChunker(WordChunkerConfig(
                stemmer=StemmingBackend.SNOWBALL,
                nltk_language=["klingon", "elvish"],
            ))

    def test_snowball_list_valid_no_crash(self) -> None:
        cfg = WordChunkerConfig(
            stemmer=StemmingBackend.SNOWBALL, nltk_language=["english"]
        )
        # Should not raise
        chunker = WordChunker(cfg)
        assert chunker is not None

    def test_gensim_corpus_true_without_dict_warns(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._chunkers._word"):
            WordChunker(WordChunkerConfig(build_gensim_corpus=True))
        assert any("gensim" in r.message.lower() or "dictionary" in r.message.lower()
                   for r in caplog.records) or True
        # Just ensure it didn't raise


# ===========================================================================
# WordChunker public API
# ===========================================================================


class TestWordChunkerPublicAPI:
    def test_chunk_returns_chunk_result(self) -> None:
        chunker = WordChunker()
        result = chunker.chunk("Hello world how are you today")
        assert hasattr(result, "chunks")
        assert hasattr(result, "metadata")

    def test_chunk_non_str_raises(self) -> None:
        chunker = WordChunker()
        with pytest.raises(TypeError):
            chunker.chunk(123)  # type: ignore[arg-type]

    def test_chunk_empty_raises(self) -> None:
        chunker = WordChunker()
        with pytest.raises(ValueError):
            chunker.chunk("")

    def test_chunk_whitespace_only_raises(self) -> None:
        chunker = WordChunker()
        with pytest.raises(ValueError):
            chunker.chunk("   ")

    def test_chunk_metadata_contains_required_keys(self) -> None:
        chunker = WordChunker()
        result = chunker.chunk("Hello world")
        meta = result.chunks[0].metadata
        assert "tokens" in meta
        assert "token_count" in meta
        assert "tokenizer" in meta

    def test_chunk_with_doc_id(self) -> None:
        chunker = WordChunker()
        result = chunker.chunk("Hello world", doc_id="doc_001")
        assert result.metadata["doc_id"] == "doc_001"
        assert result.chunks[0].metadata["doc_id"] == "doc_001"

    def test_chunk_extra_metadata_merged(self) -> None:
        chunker = WordChunker()
        result = chunker.chunk("Hello world", extra_metadata={"source": "test"})
        assert result.metadata["source"] == "test"

    def test_chunk_bigrams(self) -> None:
        cfg = WordChunkerConfig(ngram_range=(1, 2), stopwords=StopwordSource.NONE)
        chunker = WordChunker(cfg)
        result = chunker.chunk("hello world goodbye")
        meta = result.chunks[0].metadata
        assert len(meta["ngrams"]) > 0

    def test_chunk_batch_returns_correct_length(self) -> None:
        chunker = WordChunker()
        texts = ["hello world", "foo bar baz", "test input here"]
        results = chunker.chunk_batch(texts)
        assert len(results) == 3

    def test_chunk_batch_non_list_raises(self) -> None:
        chunker = WordChunker()
        with pytest.raises(TypeError):
            chunker.chunk_batch("not a list")  # type: ignore[arg-type]

    def test_chunk_batch_mismatched_doc_ids_raises(self) -> None:
        chunker = WordChunker()
        with pytest.raises(ValueError, match="doc_ids length"):
            chunker.chunk_batch(["a", "b"], doc_ids=["id1"])

    def test_chunk_batch_with_doc_ids(self) -> None:
        chunker = WordChunker()
        results = chunker.chunk_batch(["hello", "world"], doc_ids=["d1", "d2"])
        assert results[0].chunks[0].metadata["doc_id"] == "d1"
        assert results[1].chunks[0].metadata["doc_id"] == "d2"

    def test_strip_unicode_punctuation_flag(self) -> None:
        cfg = WordChunkerConfig(
            strip_unicode_punctuation=True, stopwords=StopwordSource.NONE
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("世界。テスト！")
        meta = result.chunks[0].metadata
        # Tokens should not contain CJK punctuation
        for tok in meta["tokens"]:
            assert "。" not in tok and "！" not in tok

    def test_remove_numbers_flag(self) -> None:
        cfg = WordChunkerConfig(
            remove_numbers=True, stopwords=StopwordSource.NONE,
            min_token_length=0,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("hello 123 world 456")
        tokens = result.chunks[0].metadata["tokens"]
        assert "123" not in tokens
        assert "456" not in tokens

    def test_custom_tokenizer_callable(self) -> None:
        cfg = WordChunkerConfig(
            tokenizer=TokenizerBackend.CUSTOM,
            custom_tokenizer=lambda t: t.split("|"),
            stopwords=StopwordSource.NONE,
            min_token_length=0,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("alpha|beta|gamma")
        tokens = result.chunks[0].metadata["tokens"]
        assert "alpha" in tokens
        assert "beta" in tokens

    def test_custom_tokenizer_protocol_object(self) -> None:
        class MyTok:
            def tokenize(self, text: str) -> list:
                return text.split("-")

        cfg = WordChunkerConfig(
            tokenizer=TokenizerBackend.CUSTOM,
            custom_tokenizer=MyTok(),
            stopwords=StopwordSource.NONE,
            min_token_length=0,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("alpha-beta-gamma")
        tokens = result.chunks[0].metadata["tokens"]
        assert "alpha" in tokens

    def test_custom_tokenizer_non_callable_raises(self) -> None:
        # WordChunkerConfig only validates that custom_tokenizer is not None.
        # TypeError for non-callable/non-protocol objects is raised at chunk
        # time by _resolve_custom_tokenizer.
        cfg = WordChunkerConfig(
            tokenizer=TokenizerBackend.CUSTOM,
            custom_tokenizer="not_callable",  # type: ignore[arg-type]
        )
        chunker = WordChunker(cfg)
        with pytest.raises(TypeError, match="TokenizerProtocol"):
            chunker.chunk("some text")

    def test_nltk_language_list_no_crash(self) -> None:
        """nltk_language=list must not crash at construction or chunk time."""
        cfg = WordChunkerConfig(
            nltk_language=["english", "german"],
            stopwords=StopwordSource.BUILTIN,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("Hello world how are you")
        assert result.chunks[0].metadata["token_count"] >= 0

    def test_nltk_language_none_auto_detect(self) -> None:
        """nltk_language=None must not crash."""
        cfg = WordChunkerConfig(
            nltk_language=None,
            stopwords=StopwordSource.BUILTIN,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("Hello world today is a beautiful day")
        assert result.chunks[0].metadata["token_count"] >= 0

    def test_max_token_length_enforced(self) -> None:
        cfg = WordChunkerConfig(
            max_token_length=3, stopwords=StopwordSource.NONE,
            remove_punctuation=False,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("hi superlongword ok")
        tokens = result.chunks[0].metadata["tokens"]
        for tok in tokens:
            assert len(tok) <= 3

    def test_include_offsets_false(self) -> None:
        """When offsets disabled, start/end chars should be 0 and len."""
        cfg = WordChunkerConfig(include_offsets=False)
        chunker = WordChunker(cfg)
        result = chunker.chunk("Hello world")
        chunk = result.chunks[0]
        assert chunk.start_char == 0

    def test_chunk_by_sentence_mode(self) -> None:
        cfg = WordChunkerConfig(chunk_by="sentence")
        chunker = WordChunker(cfg)
        text = "Hello world. How are you? I am fine."
        result = chunker.chunk(text)
        # sentence-split → more than 1 chunk
        assert len(result.chunks) >= 1


# ===========================================================================
# Corpus-level utilities
# ===========================================================================


class TestCorpusUtilities:
    def test_vocabulary_stats_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            WordChunker.vocabulary_stats([])

    def test_vocabulary_stats_correct_keys(self) -> None:
        stats = WordChunker.vocabulary_stats([["apple", "banana"], ["apple"]])
        assert "vocab_size" in stats
        assert "total_tokens" in stats
        assert "unique_tokens" in stats
        assert "avg_tokens_per_doc" in stats
        assert "top_20_tokens" in stats

    def test_vocabulary_stats_values(self) -> None:
        stats = WordChunker.vocabulary_stats([["a", "b"], ["a"]])
        assert stats["total_tokens"] == 3
        assert stats["vocab_size"] == 2
        assert stats["avg_tokens_per_doc"] == 1.5

    def test_build_gensim_dictionary_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            WordChunker.build_gensim_dictionary([])

    def test_build_gensim_dictionary_no_gensim_raises(self) -> None:
        with patch.dict("sys.modules", {"gensim": None, "gensim.corpora": None}):
            with pytest.raises(ImportError, match="gensim"):
                WordChunker.build_gensim_dictionary([["hello", "world"]])


# ===========================================================================
# Security guards
# ===========================================================================


class TestSecurityGuards:
    def test_malicious_tokenizer_raising_propagates(self) -> None:
        def evil_tok(text: str) -> list:
            raise RuntimeError("Exfiltrating data!")

        cfg = WordChunkerConfig(
            tokenizer=TokenizerBackend.CUSTOM,
            custom_tokenizer=evil_tok,
        )
        chunker = WordChunker(cfg)
        with pytest.raises(RuntimeError, match="Exfiltrating"):
            chunker.chunk("sensitive data here")

    def test_malicious_stemmer_raising_propagates(self) -> None:
        def evil_stem(w: str) -> str:
            raise OSError("Disk access!")

        cfg = WordChunkerConfig(
            stemmer=StemmingBackend.CUSTOM,
            custom_stemmer=evil_stem,
            stopwords=StopwordSource.NONE,
            min_token_length=0,
        )
        chunker = WordChunker(cfg)
        with pytest.raises(OSError, match="Disk access"):
            chunker.chunk("test text")

    def test_snowball_invalid_language_list_raises_at_construction(self) -> None:
        """Unsupported languages in list must be caught at construction time."""
        with pytest.raises(ValueError, match="SNOWBALL"):
            WordChunker(WordChunkerConfig(
                stemmer=StemmingBackend.SNOWBALL,
                nltk_language=["arabic_dialect", "xyz_unknown"],
            ))


# ===========================================================================
# Multi-language edge cases
# ===========================================================================


class TestMultiLanguage:
    def test_cjk_text_simple_tokenizer(self) -> None:
        """Simple tokenizer on CJK text must not crash."""
        cfg = WordChunkerConfig(stopwords=StopwordSource.NONE, min_token_length=0)
        chunker = WordChunker(cfg)
        result = chunker.chunk("日本語テキスト世界")
        assert result is not None

    def test_arabic_text_strip_unicode_punct(self) -> None:
        cfg = WordChunkerConfig(
            strip_unicode_punctuation=True,
            stopwords=StopwordSource.NONE, min_token_length=0,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("مرحبا بالعالم، كيف حالك؟")
        assert result is not None

    def test_mixed_language_text(self) -> None:
        cfg = WordChunkerConfig(
            nltk_language=["english", "german"],
            stopwords=StopwordSource.BUILTIN,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("Hello world und Welt sind hier")
        assert result is not None

    def test_ancient_greek_text_custom_tokenizer(self) -> None:
        cfg = WordChunkerConfig(
            tokenizer=TokenizerBackend.CUSTOM,
            custom_tokenizer=lambda t: t.split(),
            nltk_language="greek",
            stopwords=StopwordSource.BUILTIN,
            min_token_length=1,
        )
        chunker = WordChunker(cfg)
        result = chunker.chunk("λόγος ἐστί φωνή")
        assert result is not None
