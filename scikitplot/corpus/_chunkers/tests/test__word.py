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
