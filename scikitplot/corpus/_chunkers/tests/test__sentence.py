"""Tests for scikitplot.corpus._chunkers._sentence."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from .._sentence import (
    SentenceBackend,
    SentenceChunker,
    SentenceChunkerConfig,
    _compute_char_offsets,
    _protect_abbreviations,
    _restore_abbreviations,
    _split_regex,
)
from ..._types import Chunk, ChunkResult


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

SIMPLE_TEXT = "Hello world. How are you? Fine thanks."
THREE_SENTENCES = "The cat sat. The dog ran. The bird flew."


@pytest.fixture()
def default_chunker() -> SentenceChunker:
    return SentenceChunker()


@pytest.fixture()
def overlap_chunker() -> SentenceChunker:
    return SentenceChunker(
        SentenceChunkerConfig(overlap=1, min_length=5)
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestSentenceChunkerConfig:
    def test_invalid_min_length_raises(self) -> None:
        with pytest.raises(ValueError, match="min_length"):
            SentenceChunker(SentenceChunkerConfig(min_length=-1))

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            SentenceChunker(SentenceChunkerConfig(overlap=-1))

    def test_spacy_without_model_raises(self) -> None:
        with pytest.raises(ValueError, match="spacy_model"):
            SentenceChunker(
                SentenceChunkerConfig(
                    backend=SentenceBackend.SPACY, spacy_model=None
                )
            )

    def test_default_config_valid(self, default_chunker: SentenceChunker) -> None:
        assert default_chunker._cfg.backend == SentenceBackend.REGEX
        assert default_chunker._cfg.min_length == 10
        assert default_chunker._cfg.overlap == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestChunkInputValidation:
    def test_non_string_raises_type_error(
        self, default_chunker: SentenceChunker
    ) -> None:
        with pytest.raises(TypeError, match="str"):
            default_chunker.chunk(123)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(
        self, default_chunker: SentenceChunker
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            default_chunker.chunk("")

    def test_whitespace_only_raises_value_error(
        self, default_chunker: SentenceChunker
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            default_chunker.chunk("   \n\t  ")


# ---------------------------------------------------------------------------
# Regex backend — functional
# ---------------------------------------------------------------------------


class TestSentenceChunkerRegex:
    def test_splits_simple_text(
        self, default_chunker: SentenceChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE_TEXT)
        assert isinstance(result, ChunkResult)
        assert len(result.chunks) == 3

    def test_chunk_text_content(
        self, default_chunker: SentenceChunker
    ) -> None:
        result = default_chunker.chunk(THREE_SENTENCES)
        texts = [c.text for c in result.chunks]
        assert "The cat sat." in texts
        assert "The dog ran." in texts

    def test_chunk_type(self, default_chunker: SentenceChunker) -> None:
        result = default_chunker.chunk(SIMPLE_TEXT)
        for chunk in result.chunks:
            assert isinstance(chunk, Chunk)

    def test_chunk_metadata_keys(
        self, default_chunker: SentenceChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE_TEXT)
        for chunk in result.chunks:
            assert "chunk_index" in chunk.metadata
            assert "sentence_index" in chunk.metadata
            assert "backend" in chunk.metadata

    def test_doc_id_in_metadata(
        self, default_chunker: SentenceChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE_TEXT, doc_id="doc_1")
        for chunk in result.chunks:
            assert chunk.metadata["doc_id"] == "doc_1"
        assert result.metadata["doc_id"] == "doc_1"

    def test_extra_metadata_merged(
        self, default_chunker: SentenceChunker
    ) -> None:
        result = default_chunker.chunk(
            SIMPLE_TEXT, extra_metadata={"source": "test"}
        )
        assert result.metadata["source"] == "test"

    def test_result_metadata_total_chunks(
        self, default_chunker: SentenceChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE_TEXT)
        assert result.metadata["total_chunks"] == len(result.chunks)

    def test_result_metadata_chunker_key(
        self, default_chunker: SentenceChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE_TEXT)
        assert result.metadata["chunker"] == "sentence"

    def test_min_length_filter(self) -> None:
        chunker = SentenceChunker(SentenceChunkerConfig(min_length=50))
        result = chunker.chunk(SIMPLE_TEXT)
        # All three sentences are < 50 chars, so nothing should remain.
        assert len(result.chunks) == 0

    def test_offsets_are_integers(
        self, default_chunker: SentenceChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE_TEXT)
        for chunk in result.chunks:
            assert isinstance(chunk.start_char, int)
            assert isinstance(chunk.end_char, int)

    def test_offsets_disabled(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(include_offsets=False)
        )
        result = chunker.chunk(SIMPLE_TEXT)
        for chunk in result.chunks:
            assert chunk.start_char == 0
            assert chunk.end_char == 0


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


class TestSentenceChunkerOverlap:
    def test_overlap_prepends_context(
        self, overlap_chunker: SentenceChunker
    ) -> None:
        result = overlap_chunker.chunk(THREE_SENTENCES)
        # Second chunk should contain first sentence as context.
        second = result.chunks[1].text
        assert "The cat sat." in second

    def test_overlap_count_in_metadata(
        self, overlap_chunker: SentenceChunker
    ) -> None:
        result = overlap_chunker.chunk(THREE_SENTENCES)
        # First chunk has no overlap.
        assert result.chunks[0].metadata["overlap_count"] == 0
        # Second chunk has 1 context sentence.
        assert result.chunks[1].metadata["overlap_count"] == 1


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


class TestChunkBatch:
    def test_batch_returns_list(
        self, default_chunker: SentenceChunker
    ) -> None:
        results = default_chunker.chunk_batch([SIMPLE_TEXT, THREE_SENTENCES])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_non_list_raises(
        self, default_chunker: SentenceChunker
    ) -> None:
        with pytest.raises(TypeError, match="list"):
            default_chunker.chunk_batch("not a list")  # type: ignore[arg-type]

    def test_batch_doc_ids_mismatch_raises(
        self, default_chunker: SentenceChunker
    ) -> None:
        with pytest.raises(ValueError, match="length"):
            default_chunker.chunk_batch(
                [SIMPLE_TEXT], doc_ids=["a", "b"]
            )

    def test_batch_doc_ids_assigned(
        self, default_chunker: SentenceChunker
    ) -> None:
        results = default_chunker.chunk_batch(
            [SIMPLE_TEXT, THREE_SENTENCES], doc_ids=["d1", "d2"]
        )
        assert results[0].metadata["doc_id"] == "d1"
        assert results[1].metadata["doc_id"] == "d2"


# ---------------------------------------------------------------------------
# NLTK backend — mocked
# ---------------------------------------------------------------------------


class TestSentenceChunkerNLTK:
    @patch("scikitplot.corpus._chunkers._sentence._split_nltk")
    def test_nltk_backend_called(self, mock_split: MagicMock) -> None:
        mock_split.return_value = ["Hello world.", "How are you?"]
        chunker = SentenceChunker(
            SentenceChunkerConfig(backend=SentenceBackend.NLTK)
        )
        result = chunker.chunk(SIMPLE_TEXT)
        mock_split.assert_called_once()
        assert len(result.chunks) == 2

    def test_nltk_import_error_propagates(self) -> None:
        with patch.dict("sys.modules", {"nltk": None, "nltk.tokenize": None}):
            with pytest.raises(ImportError, match="nltk"):
                from .._sentence import _split_nltk
                _split_nltk("some text", "english")


# ---------------------------------------------------------------------------
# spaCy backend — mocked
# ---------------------------------------------------------------------------


class TestSentenceChunkerSpacy:
    @patch("scikitplot.corpus._chunkers._sentence._split_spacy")
    def test_spacy_backend_called(self, mock_split: MagicMock) -> None:
        mock_split.return_value = ["Hello world.", "How are you?"]
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.SPACY,
                spacy_model="en_core_web_sm",
            )
        )
        result = chunker.chunk(SIMPLE_TEXT)
        mock_split.assert_called_once_with(SIMPLE_TEXT, "en_core_web_sm", chunker._nlp_cache)
        assert len(result.chunks) == 2

    def test_spacy_import_error_propagates(self) -> None:
        with patch.dict("sys.modules", {"spacy": None}):
            with pytest.raises(ImportError, match="spacy"):
                from .._sentence import _split_spacy
                _split_spacy("text", "en_core_web_sm")


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_protect_restore_abbreviations_roundtrip(self) -> None:
        text = "Dr. Smith is here. Mr. Jones will follow."
        protected = _protect_abbreviations(text)
        restored = _restore_abbreviations(protected)
        assert restored == text

    def test_protect_prevents_split(self) -> None:
        text = "Dr. Smith arrived."
        protected = _protect_abbreviations(text)
        parts = _split_regex(protected)
        # Should not split inside 'Dr. Smith'.
        assert len(parts) == 1

    def test_compute_offsets_simple(self) -> None:
        source = "Hello world. How are you?"
        segs = ["Hello world.", "How are you?"]
        offsets = _compute_char_offsets(source, segs)
        assert offsets[0] == (0, 12)
        assert offsets[1][0] == 13

    def test_compute_offsets_empty_list(self) -> None:
        offsets = _compute_char_offsets("text", [])
        assert offsets == []
