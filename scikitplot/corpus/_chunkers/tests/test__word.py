"""Tests for scikitplot.corpus._chunkers._word."""

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
    _load_stopwords,
    _process_tokens,
    _tokenize_simple,
)
from ..._types import Chunk, ChunkResult

TEXT = "The quick brown foxes jumped over the lazy dogs near the riverbank."
SHORT_TEXT = "Hello beautiful world."


@pytest.fixture()
def default_chunker() -> WordChunker:
    return WordChunker()


@pytest.fixture()
def porter_chunker() -> WordChunker:
    return WordChunker(
        WordChunkerConfig(stemmer=StemmingBackend.PORTER)
    )


@pytest.fixture()
def bigram_chunker() -> WordChunker:
    return WordChunker(
        WordChunkerConfig(ngram_range=(1, 2))
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestWordChunkerConfig:
    def test_negative_min_token_length_raises(self) -> None:
        with pytest.raises(ValueError, match="min_token_length"):
            WordChunker(WordChunkerConfig(min_token_length=-1))

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_token_length"):
            WordChunker(
                WordChunkerConfig(min_token_length=5, max_token_length=3)
            )

    def test_invalid_ngram_range_raises(self) -> None:
        with pytest.raises(ValueError, match="ngram_range"):
            WordChunker(WordChunkerConfig(ngram_range=(0, 2)))

    def test_invalid_chunk_by_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_by"):
            WordChunker(WordChunkerConfig(chunk_by="word"))

    def test_spacy_without_model_raises(self) -> None:
        with pytest.raises(ValueError, match="spacy_model"):
            WordChunker(
                WordChunkerConfig(tokenizer=TokenizerBackend.SPACY, spacy_model=None)
            )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_string_raises(
        self, default_chunker: WordChunker
    ) -> None:
        with pytest.raises(TypeError, match="str"):
            default_chunker.chunk(99)  # type: ignore[arg-type]

    def test_empty_string_raises(
        self, default_chunker: WordChunker
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            default_chunker.chunk("")

    def test_whitespace_only_raises(
        self, default_chunker: WordChunker
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            default_chunker.chunk("   ")


# ---------------------------------------------------------------------------
# Default (SIMPLE) tokenizer
# ---------------------------------------------------------------------------


class TestWordChunkerSimple:
    def test_returns_chunk_result(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        assert isinstance(result, ChunkResult)

    def test_single_chunk_per_document(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        assert len(result.chunks) == 1

    def test_chunk_is_chunk_instance(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        assert isinstance(result.chunks[0], Chunk)

    def test_metadata_has_tokens(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        meta = result.chunks[0].metadata
        assert "tokens" in meta
        assert isinstance(meta["tokens"], list)

    def test_metadata_token_count(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        meta = result.chunks[0].metadata
        assert meta["token_count"] == len(meta["tokens"])

    def test_stopwords_removed(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        tokens = result.chunks[0].metadata["tokens"]
        assert "The" not in tokens
        assert "the" not in tokens

    def test_punctuation_removed(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(SHORT_TEXT)
        tokens = result.chunks[0].metadata["tokens"]
        for tok in tokens:
            assert "." not in tok

    def test_lowercase_applied(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk("Hello World")
        tokens = result.chunks[0].metadata["tokens"]
        for tok in tokens:
            assert tok == tok.lower()

    def test_doc_id_in_metadata(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT, doc_id="doc_99")
        assert result.chunks[0].metadata["doc_id"] == "doc_99"

    def test_extra_metadata_merged(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(
            TEXT, extra_metadata={"version": "1.0"}
        )
        assert result.metadata["version"] == "1.0"

    def test_result_metadata_chunker_key(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        assert result.metadata["chunker"] == "word"


# ---------------------------------------------------------------------------
# Stemming
# ---------------------------------------------------------------------------


class TestStemming:
    def test_porter_stems_tokens(
        self, porter_chunker: WordChunker
    ) -> None:
        pytest.importorskip("nltk", reason="nltk not installed; stemming tests skipped")
        result = porter_chunker.chunk("The jumping runners ran quickly.")
        tokens = result.chunks[0].metadata["tokens"]
        # 'jumping' → 'jump', 'runners' → 'runner' or 'run'
        assert any("jump" in t for t in tokens)

    def test_stemmer_key_in_metadata(
        self, porter_chunker: WordChunker
    ) -> None:
        pytest.importorskip("nltk", reason="nltk not installed; stemming tests skipped")
        result = porter_chunker.chunk(TEXT)
        assert result.chunks[0].metadata["stemmer"] == "porter"

    @patch("scikitplot.corpus._chunkers._word._get_stemmer")
    def test_stemmer_called(self, mock_get: MagicMock) -> None:
        mock_stemmer = MagicMock()
        mock_stemmer.stem.side_effect = lambda x: x + "_stem"
        mock_get.return_value = mock_stemmer

        chunker = WordChunker(
            WordChunkerConfig(stemmer=StemmingBackend.PORTER)
        )
        result = chunker.chunk("running foxes")
        # Tokens should have been passed through mock stemmer.
        assert mock_stemmer.stem.called


# ---------------------------------------------------------------------------
# Lemmatization — mocked
# ---------------------------------------------------------------------------


class TestLemmatization:
    @patch("scikitplot.corpus._chunkers._word._get_nltk_lemmatizer")
    def test_nltk_lemmatizer_called(self, mock_get: MagicMock) -> None:
        mock_lemmatizer = MagicMock()
        mock_lemmatizer.lemmatize.side_effect = lambda x: x
        mock_get.return_value = mock_lemmatizer

        chunker = WordChunker(
            WordChunkerConfig(lemmatizer=LemmatizationBackend.NLTK_WORDNET)
        )
        result = chunker.chunk("The foxes jumped quickly over rivers.")
        assert mock_lemmatizer.lemmatize.called


# ---------------------------------------------------------------------------
# N-gram extraction
# ---------------------------------------------------------------------------


class TestNgrams:
    def test_bigrams_extracted(
        self, bigram_chunker: WordChunker
    ) -> None:
        result = bigram_chunker.chunk("quick brown fox")
        ngrams = result.chunks[0].metadata["ngrams"]
        assert any("_" in ng for ng in ngrams)

    def test_unigrams_only(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk("quick brown fox")
        ngrams = result.chunks[0].metadata["ngrams"]
        assert ngrams == []

    def test_ngram_range_in_metadata(
        self, bigram_chunker: WordChunker
    ) -> None:
        result = bigram_chunker.chunk(TEXT)
        assert result.chunks[0].metadata["ngram_range"] == [1, 2]


# ---------------------------------------------------------------------------
# Chunk-by-sentence mode
# ---------------------------------------------------------------------------


class TestChunkBySentence:
    def test_sentence_mode_multiple_chunks(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(chunk_by="sentence", min_token_length=1)
        )
        result = chunker.chunk("Hello world. How are you? Fine thanks.")
        assert len(result.chunks) >= 2

    def test_document_mode_single_chunk(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        assert len(result.chunks) == 1


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_basic(self, default_chunker: WordChunker) -> None:
        results = default_chunker.chunk_batch([TEXT, SHORT_TEXT])
        assert len(results) == 2

    def test_batch_type_error(
        self, default_chunker: WordChunker
    ) -> None:
        with pytest.raises(TypeError, match="list"):
            default_chunker.chunk_batch("oops")  # type: ignore[arg-type]

    def test_batch_doc_ids_mismatch(
        self, default_chunker: WordChunker
    ) -> None:
        with pytest.raises(ValueError, match="length"):
            default_chunker.chunk_batch([TEXT], doc_ids=["a", "b"])


# ---------------------------------------------------------------------------
# NLTK backend — mocked
# ---------------------------------------------------------------------------


class TestNLTKTokenizer:
    @patch("scikitplot.corpus._chunkers._word._tokenize_nltk")
    def test_nltk_tokenizer_called(self, mock_tok: MagicMock) -> None:
        mock_tok.return_value = ["quick", "brown", "fox"]
        chunker = WordChunker(
            WordChunkerConfig(tokenizer=TokenizerBackend.NLTK)
        )
        result = chunker.chunk(TEXT)
        mock_tok.assert_called_once()


# ---------------------------------------------------------------------------
# Gensim BoW — mocked
# ---------------------------------------------------------------------------


class TestGensimBoW:
    def test_bow_attached_when_dict_provided(self) -> None:
        mock_dict = MagicMock()
        mock_dict.doc2bow.return_value = [(0, 1), (1, 2)]
        chunker = WordChunker(
            WordChunkerConfig(build_gensim_corpus=True),
            gensim_dictionary=mock_dict,
        )
        result = chunker.chunk(TEXT)
        assert "bow" in result.chunks[0].metadata
        assert result.chunks[0].metadata["bow"] == [(0, 1), (1, 2)]

    def test_no_bow_without_dict(
        self, default_chunker: WordChunker
    ) -> None:
        result = default_chunker.chunk(TEXT)
        assert "bow" not in result.chunks[0].metadata


# ---------------------------------------------------------------------------
# Corpus utilities
# ---------------------------------------------------------------------------


class TestCorpusUtilities:
    def test_vocabulary_stats_basic(self) -> None:
        token_lists = [["fox", "jump"], ["fox", "run"], ["cat"]]
        stats = WordChunker.vocabulary_stats(token_lists)
        assert stats["vocab_size"] == 4
        assert stats["total_tokens"] == 5
        assert stats["avg_tokens_per_doc"] == pytest.approx(5 / 3)

    def test_vocabulary_stats_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            WordChunker.vocabulary_stats([])

    def test_build_gensim_dictionary(self) -> None:
        import sys
        from unittest.mock import MagicMock

        mock_instance = MagicMock()
        mock_dict_cls = MagicMock(return_value=mock_instance)
        mock_corpora = MagicMock()
        mock_corpora.Dictionary = mock_dict_cls
        mock_gensim = MagicMock()
        mock_gensim.corpora = mock_corpora

        token_lists = [["fox", "jump"], ["fox", "run"]]
        with patch.dict(
            "sys.modules",
            {"gensim": mock_gensim, "gensim.corpora": mock_corpora},
        ):
            from .._word import WordChunker as WC  # noqa: PLC0415

            result = WC.build_gensim_dictionary(token_lists)
        mock_dict_cls.assert_called_once_with(token_lists)
        mock_instance.filter_extremes.assert_called_once()

    def test_build_gensim_dictionary_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            WordChunker.build_gensim_dictionary([])


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_tokenize_simple_basic(self) -> None:
        tokens = _tokenize_simple("Hello, world!")
        assert "Hello" in tokens or "hello" in tokens.copy()
        assert len(tokens) >= 2

    def test_tokenize_simple_strips_punctuation(self) -> None:
        tokens = _tokenize_simple("fox, jumps. over!")
        for tok in tokens:
            assert "," not in tok
            assert "." not in tok

    def test_extract_ngrams_bigrams(self) -> None:
        tokens = ["a", "b", "c", "d"]
        bigrams = _extract_ngrams(tokens, 2)
        assert bigrams == ["a_b", "b_c", "c_d"]

    def test_extract_ngrams_trigrams(self) -> None:
        tokens = ["a", "b", "c", "d"]
        trigrams = _extract_ngrams(tokens, 3)
        assert trigrams == ["a_b_c", "b_c_d"]

    def test_extract_ngrams_insufficient_tokens(self) -> None:
        assert _extract_ngrams(["a"], 3) == []

    def test_load_stopwords_builtin(self) -> None:
        sw = _load_stopwords(StopwordSource.BUILTIN, "english")
        assert "the" in sw
        assert isinstance(sw, frozenset)

    def test_load_stopwords_none(self) -> None:
        sw = _load_stopwords(StopwordSource.NONE, "english")
        assert sw == frozenset()

    def test_process_tokens_lowercase(self) -> None:
        cfg = WordChunkerConfig(
            lowercase=True,
            stopwords=StopwordSource.NONE,
            remove_punctuation=False,
            min_token_length=1,
        )
        out = _process_tokens(
            ["Hello", "World"], cfg, frozenset()
        )
        assert out == ["hello", "world"]

    def test_process_tokens_removes_stopwords(self) -> None:
        cfg = WordChunkerConfig(
            lowercase=True,
            stopwords=StopwordSource.BUILTIN,
            min_token_length=1,
        )
        from .._word import _BUILTIN_STOPWORDS
        out = _process_tokens(
            ["the", "fox", "jumps"], cfg, _BUILTIN_STOPWORDS
        )
        assert "the" not in out
        assert "fox" in out


# ---------------------------------------------------------------------------
# _to_gensim_bow — direct unit test
# ---------------------------------------------------------------------------


class TestToGensimBow:
    """Direct tests for ``_to_gensim_bow``."""

    def test_to_gensim_bow_produces_bow_vector(self) -> None:
        """BoW vector must be a list of (int, int) tuples."""
        try:
            from gensim.corpora import Dictionary  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("gensim not installed")

        from .._word import _to_gensim_bow  # noqa: PLC0415

        tokens = ["cat", "dog", "cat"]
        dictionary = Dictionary([tokens])
        bow = _to_gensim_bow(tokens, dictionary)
        assert isinstance(bow, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in bow)
        counts = dict(bow)
        # "cat" appears twice — count must be 2
        cat_id = dictionary.token2id["cat"]
        assert counts[cat_id] == 2

    def test_to_gensim_bow_empty_tokens(self) -> None:
        """Empty token list must produce an empty BoW vector."""
        try:
            from gensim.corpora import Dictionary  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("gensim not installed")

        from .._word import _to_gensim_bow  # noqa: PLC0415

        dictionary = Dictionary([["hello"]])
        bow = _to_gensim_bow([], dictionary)
        assert bow == []


# ---------------------------------------------------------------------------
# _resolve_custom_tokenizer / _resolve_custom_stemmer / _resolve_custom_lemmatizer
# ---------------------------------------------------------------------------


class TestResolveCustomHelpers:
    """Direct tests for the three _resolve_custom_* helpers."""

    def test_resolve_custom_tokenizer_none_raises_value_error(self) -> None:
        """``custom_tokenizer=None`` must raise ``ValueError``."""
        from .._word import TokenizerBackend, WordChunkerConfig, _resolve_custom_tokenizer  # noqa: PLC0415

        cfg = WordChunkerConfig(
            tokenizer=TokenizerBackend.CUSTOM,
            custom_tokenizer=None,
        )
        with pytest.raises(ValueError, match="custom_tokenizer"):
            _resolve_custom_tokenizer(cfg)

    def test_resolve_custom_tokenizer_raw_callable_wraps_in_function_tokenizer(
        self,
    ) -> None:
        """A plain callable must be auto-wrapped in FunctionTokenizer."""
        from .._custom_tokenizer import FunctionTokenizer  # noqa: PLC0415
        from .._word import TokenizerBackend, WordChunkerConfig, _resolve_custom_tokenizer  # noqa: PLC0415

        fn = lambda text: text.split()  # noqa: E731
        cfg = WordChunkerConfig(
            tokenizer=TokenizerBackend.CUSTOM,
            custom_tokenizer=fn,
        )
        result = _resolve_custom_tokenizer(cfg)
        assert isinstance(result, FunctionTokenizer)

    def test_resolve_custom_tokenizer_non_protocol_raises_type_error(self) -> None:
        """Non-callable, non-protocol object must raise TypeError."""
        from .._word import TokenizerBackend, WordChunkerConfig, _resolve_custom_tokenizer  # noqa: PLC0415

        cfg = WordChunkerConfig(
            tokenizer=TokenizerBackend.CUSTOM,
            custom_tokenizer=42,  # type: ignore[arg-type]
        )
        with pytest.raises(TypeError, match="TokenizerProtocol"):
            _resolve_custom_tokenizer(cfg)

    def test_resolve_custom_stemmer_none_raises_value_error(self) -> None:
        """``custom_stemmer=None`` must raise ``ValueError``."""
        from .._word import StemmingBackend, WordChunkerConfig, _resolve_custom_stemmer  # noqa: PLC0415

        cfg = WordChunkerConfig(
            stemmer=StemmingBackend.CUSTOM,
            custom_stemmer=None,
        )
        with pytest.raises(ValueError, match="custom_stemmer"):
            _resolve_custom_stemmer(cfg)

    def test_resolve_custom_stemmer_raw_callable_wraps(self) -> None:
        """A plain callable stemmer must be auto-wrapped in FunctionStemmer."""
        from .._custom_tokenizer import FunctionStemmer  # noqa: PLC0415
        from .._word import StemmingBackend, WordChunkerConfig, _resolve_custom_stemmer  # noqa: PLC0415

        fn = lambda w: w[:4]  # noqa: E731
        cfg = WordChunkerConfig(
            stemmer=StemmingBackend.CUSTOM,
            custom_stemmer=fn,
        )
        result = _resolve_custom_stemmer(cfg)
        assert isinstance(result, FunctionStemmer)

    def test_resolve_custom_stemmer_non_protocol_raises_type_error(self) -> None:
        """Non-callable, non-protocol object must raise TypeError."""
        from .._word import StemmingBackend, WordChunkerConfig, _resolve_custom_stemmer  # noqa: PLC0415

        cfg = WordChunkerConfig(
            stemmer=StemmingBackend.CUSTOM,
            custom_stemmer="not_a_stemmer",  # type: ignore[arg-type]
        )
        with pytest.raises(TypeError, match="StemmerProtocol"):
            _resolve_custom_stemmer(cfg)

    def test_resolve_custom_lemmatizer_none_raises_value_error(self) -> None:
        """``custom_lemmatizer=None`` must raise ``ValueError``."""
        from .._word import LemmatizationBackend, WordChunkerConfig, _resolve_custom_lemmatizer  # noqa: PLC0415

        cfg = WordChunkerConfig(
            lemmatizer=LemmatizationBackend.CUSTOM,
            custom_lemmatizer=None,
        )
        with pytest.raises(ValueError, match="custom_lemmatizer"):
            _resolve_custom_lemmatizer(cfg)

    def test_resolve_custom_lemmatizer_raw_callable_wraps(self) -> None:
        """A plain callable lemmatizer must be auto-wrapped in FunctionLemmatizer."""
        from .._custom_tokenizer import FunctionLemmatizer  # noqa: PLC0415
        from .._word import LemmatizationBackend, WordChunkerConfig, _resolve_custom_lemmatizer  # noqa: PLC0415

        fn = lambda w: w.lower()  # noqa: E731
        cfg = WordChunkerConfig(
            lemmatizer=LemmatizationBackend.CUSTOM,
            custom_lemmatizer=fn,
        )
        result = _resolve_custom_lemmatizer(cfg)
        assert isinstance(result, FunctionLemmatizer)

    def test_resolve_custom_lemmatizer_non_protocol_raises_type_error(self) -> None:
        """Non-callable, non-protocol object must raise TypeError."""
        from .._word import LemmatizationBackend, WordChunkerConfig, _resolve_custom_lemmatizer  # noqa: PLC0415

        cfg = WordChunkerConfig(
            lemmatizer=LemmatizationBackend.CUSTOM,
            custom_lemmatizer=99,  # type: ignore[arg-type]
        )
        with pytest.raises(TypeError, match="LemmatizerProtocol"):
            _resolve_custom_lemmatizer(cfg)


# ---------------------------------------------------------------------------
# WordChunker constructor — edge cases
# ---------------------------------------------------------------------------


class TestWordChunkerConstructor:
    """WordChunker constructor and config-property coverage."""

    def test_default_config_is_word_chunker_config(self) -> None:
        """Default constructor must produce a ``WordChunkerConfig`` instance."""
        from .._word import WordChunker, WordChunkerConfig  # noqa: PLC0415

        chunker = WordChunker()
        assert isinstance(chunker._cfg, WordChunkerConfig)

    def test_custom_config_stored(self) -> None:
        """Provided config must be stored and accessible."""
        from .._word import TokenizerBackend, WordChunker, WordChunkerConfig  # noqa: PLC0415

        cfg = WordChunkerConfig(tokenizer=TokenizerBackend.SIMPLE, min_token_length=3)
        chunker = WordChunker(cfg)
        assert chunker._cfg is cfg

    def test_remove_numbers_filters_numeric_tokens(self) -> None:
        """Setting ``remove_numbers=True`` must drop purely numeric tokens."""
        from .._word import WordChunker, WordChunkerConfig  # noqa: PLC0415

        chunker = WordChunker(WordChunkerConfig(remove_numbers=True))
        result = chunker.chunk("There are 42 cats and 7 dogs.")
        tokens = result.chunks[0].metadata["tokens"]
        assert "42" not in tokens
        assert "7" not in tokens

    def test_max_token_length_filters_long_tokens(self) -> None:
        """``max_token_length`` must discard tokens exceeding the limit."""
        from .._word import WordChunker, WordChunkerConfig  # noqa: PLC0415

        chunker = WordChunker(WordChunkerConfig(max_token_length=4))
        result = chunker.chunk("cat hippopotamus dog")
        tokens = result.chunks[0].metadata["tokens"]
        assert "hippopotamus" not in tokens
        assert "cat" in tokens

    def test_include_offsets_true_sets_nonzero_start(self) -> None:
        """When offsets are enabled, at least one chunk should have start=0."""
        from .._word import WordChunker, WordChunkerConfig  # noqa: PLC0415

        chunker = WordChunker(WordChunkerConfig(include_offsets=True))
        result = chunker.chunk("Hello world test.")
        # Offsets enabled — start_char of first chunk must be 0
        assert result.chunks[0].start_char == 0


# ---------------------------------------------------------------------------
# Lancaster stemmer path in _get_stemmer
# ---------------------------------------------------------------------------


class TestLancasterStemmer:
    """Cover the LANCASTER branch of ``_get_stemmer``."""

    def test_lancaster_stems_tokens(self) -> None:
        """LANCASTER backend must stem tokens without error."""
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("nltk not installed")

        from .._word import StemmingBackend, WordChunker, WordChunkerConfig  # noqa: PLC0415

        chunker = WordChunker(
            WordChunkerConfig(stemmer=StemmingBackend.LANCASTER)
        )
        result = chunker.chunk("The cats are running quickly.")
        tokens = result.chunks[0].metadata["tokens"]
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_lancaster_stemmer_key_in_metadata(self) -> None:
        """Metadata must report the stemmer backend used."""
        try:
            import nltk  # noqa: F401
        except ImportError:
            pytest.skip("nltk not installed")

        from .._word import StemmingBackend, WordChunker, WordChunkerConfig  # noqa: PLC0415

        chunker = WordChunker(
            WordChunkerConfig(stemmer=StemmingBackend.LANCASTER)
        )
        result = chunker.chunk("Running quickly across fields.")
        assert result.chunks[0].metadata["stemmer"] == "lancaster"


# ---------------------------------------------------------------------------
# chunk_batch with extra_metadata
# ---------------------------------------------------------------------------


class TestChunkBatchExtraMetadata:
    """Verify extra_metadata propagation in WordChunker.chunk_batch."""

    def test_extra_metadata_present_in_all_results(
        self, default_chunker: WordChunker
    ) -> None:
        """extra_metadata keys must appear in every batch result's metadata."""
        texts = [TEXT, SHORT_TEXT]
        extra = {"pipeline": "batch_test", "run_id": 7}
        results = default_chunker.chunk_batch(texts, extra_metadata=extra)
        for r in results:
            assert r.metadata.get("pipeline") == "batch_test"
            assert r.metadata.get("run_id") == 7

    def test_batch_preserves_doc_ids_with_extra_metadata(
        self, default_chunker: WordChunker
    ) -> None:
        """doc_ids and extra_metadata must co-exist in batch output."""
        texts = [TEXT, SHORT_TEXT]
        results = default_chunker.chunk_batch(
            texts,
            doc_ids=["d1", "d2"],
            extra_metadata={"version": 1},
        )
        assert results[0].metadata["doc_id"] == "d1"
        assert results[1].metadata["doc_id"] == "d2"
        for r in results:
            assert r.metadata["version"] == 1


# ---------------------------------------------------------------------------
# _process_tokens — advanced branches
# ---------------------------------------------------------------------------


class TestProcessTokensAdvanced:
    """Cover unchecked branches in ``_process_tokens``."""

    def test_no_lowercase_preserves_case(self) -> None:
        """``lowercase=False`` must preserve token casing."""
        from .._word import WordChunker, WordChunkerConfig  # noqa: PLC0415

        chunker = WordChunker(WordChunkerConfig(lowercase=False, remove_punctuation=False))
        result = chunker.chunk("Hello World")
        tokens = result.chunks[0].metadata["tokens"]
        # At least one capitalised token must survive
        assert any(t[0].isupper() for t in tokens if t)

    def test_stopwords_none_disables_filtering(self) -> None:
        """``StopwordSource.NONE`` must disable all stopword filtering."""
        from .._word import StopwordSource, WordChunker, WordChunkerConfig  # noqa: PLC0415

        chunker = WordChunker(
            WordChunkerConfig(stopwords=StopwordSource.NONE, min_token_length=1)
        )
        result = chunker.chunk("the cat sat on the mat")
        tokens = result.chunks[0].metadata["tokens"]
        # "the" should survive because stopwords are disabled
        assert "the" in tokens
