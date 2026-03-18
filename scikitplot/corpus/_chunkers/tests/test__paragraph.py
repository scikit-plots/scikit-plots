"""Tests for corpus._chunkers._paragraph."""

from __future__ import annotations

import pytest

from .._paragraph import (
    ParagraphChunker,
    ParagraphChunkerConfig,
    _compute_char_offsets,
    _merge_short_paragraphs,
    _split_long_paragraph,
    _split_paragraphs,
)
from ..._types import Chunk, ChunkResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
SHORT = "Hi.\n\nHello there.\n\nThis is the third paragraph which is longer."


@pytest.fixture()
def default_chunker() -> ParagraphChunker:
    return ParagraphChunker()


@pytest.fixture()
def overlap_chunker() -> ParagraphChunker:
    return ParagraphChunker(ParagraphChunkerConfig(overlap=1, min_length=3))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestParagraphChunkerConfig:
    def test_negative_min_length_raises(self) -> None:
        with pytest.raises(ValueError, match="min_length"):
            ParagraphChunker(ParagraphChunkerConfig(min_length=-1))

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            ParagraphChunker(
                ParagraphChunkerConfig(min_length=100, max_length=50)
            )

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            ParagraphChunker(ParagraphChunkerConfig(overlap=-1))

    def test_zero_max_length_raises(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            ParagraphChunker(ParagraphChunkerConfig(max_length=0))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestChunkInputValidation:
    def test_non_string_raises_type_error(
        self, default_chunker: ParagraphChunker
    ) -> None:
        with pytest.raises(TypeError, match="str"):
            default_chunker.chunk(42)  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(
        self, default_chunker: ParagraphChunker
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            default_chunker.chunk("")

    def test_whitespace_only_raises_value_error(
        self, default_chunker: ParagraphChunker
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            default_chunker.chunk("  \n  ")


# ---------------------------------------------------------------------------
# Core splitting
# ---------------------------------------------------------------------------


class TestParagraphChunker:
    def test_splits_three_paragraphs(
        self, default_chunker: ParagraphChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE)
        assert len(result.chunks) == 3

    def test_chunk_type(self, default_chunker: ParagraphChunker) -> None:
        result = default_chunker.chunk(SIMPLE)
        for c in result.chunks:
            assert isinstance(c, Chunk)

    def test_result_type(self, default_chunker: ParagraphChunker) -> None:
        result = default_chunker.chunk(SIMPLE)
        assert isinstance(result, ChunkResult)

    def test_metadata_chunker_key(
        self, default_chunker: ParagraphChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE)
        assert result.metadata["chunker"] == "paragraph"

    def test_total_chunks_in_metadata(
        self, default_chunker: ParagraphChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE)
        assert result.metadata["total_chunks"] == 3

    def test_doc_id_propagated(
        self, default_chunker: ParagraphChunker
    ) -> None:
        result = default_chunker.chunk(SIMPLE, doc_id="doc_42")
        for c in result.chunks:
            assert c.metadata["doc_id"] == "doc_42"

    def test_extra_metadata_merged(
        self, default_chunker: ParagraphChunker
    ) -> None:
        result = default_chunker.chunk(
            SIMPLE, extra_metadata={"pipeline": "test"}
        )
        assert result.metadata["pipeline"] == "test"

    def test_min_length_filters(self) -> None:
        chunker = ParagraphChunker(
            ParagraphChunkerConfig(min_length=200)
        )
        result = chunker.chunk(SIMPLE)
        assert len(result.chunks) == 0

    def test_offsets_disabled(self) -> None:
        chunker = ParagraphChunker(
            ParagraphChunkerConfig(include_offsets=False)
        )
        result = chunker.chunk(SIMPLE)
        for c in result.chunks:
            assert c.start_char == 0
            assert c.end_char == 0

    def test_max_length_splits_long_paragraph(self) -> None:
        long_para = (
            "This is a sentence. " * 20
        ).strip()
        text = long_para
        chunker = ParagraphChunker(
            ParagraphChunkerConfig(max_length=100, min_length=5)
        )
        result = chunker.chunk(text)
        assert len(result.chunks) > 1
        for c in result.chunks:
            assert len(c.text) <= 120   # allow small tolerance for joining


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


class TestParagraphOverlap:
    def test_overlap_prepends_previous(
        self, overlap_chunker: ParagraphChunker
    ) -> None:
        result = overlap_chunker.chunk(SIMPLE)
        second_text = result.chunks[1].text
        assert "First paragraph." in second_text

    def test_first_chunk_no_overlap(
        self, overlap_chunker: ParagraphChunker
    ) -> None:
        result = overlap_chunker.chunk(SIMPLE)
        assert result.chunks[0].metadata["overlap_count"] == 0


# ---------------------------------------------------------------------------
# merge_short
# ---------------------------------------------------------------------------


class TestMergeShort:
    def test_merge_enabled_reduces_chunks(self) -> None:
        chunker = ParagraphChunker(
            ParagraphChunkerConfig(merge_short=True, min_length=5)
        )
        result = chunker.chunk(SHORT)
        # "Hi." and "Hello there." are short and should be merged.
        assert len(result.chunks) < 3


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_returns_list(
        self, default_chunker: ParagraphChunker
    ) -> None:
        results = default_chunker.chunk_batch([SIMPLE, SIMPLE])
        assert len(results) == 2

    def test_batch_mismatch_raises(
        self, default_chunker: ParagraphChunker
    ) -> None:
        with pytest.raises(ValueError, match="length"):
            default_chunker.chunk_batch(
                [SIMPLE], doc_ids=["a", "b"]
            )

    def test_batch_non_list_raises(
        self, default_chunker: ParagraphChunker
    ) -> None:
        with pytest.raises(TypeError, match="list"):
            default_chunker.chunk_batch(SIMPLE)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_split_paragraphs_basic(self) -> None:
        parts = _split_paragraphs(SIMPLE, strip=True)
        assert len(parts) == 3

    def test_split_paragraphs_windows_linebreak(self) -> None:
        text = "Para 1.\r\n\r\nPara 2."
        parts = _split_paragraphs(text, strip=True)
        assert len(parts) == 2

    def test_split_long_paragraph(self) -> None:
        para = ("Word sentence. " * 15).strip()
        parts = _split_long_paragraph(para, max_len=50)
        for p in parts:
            assert len(p) <= 80   # realistic sentence boundary tolerance

    def test_merge_short_paragraphs(self) -> None:
        paras = ["Hi.", "Hello.", "This is a much longer paragraph here."]
        merged = _merge_short_paragraphs(paras, min_length=10)
        assert len(merged) < len(paras)

    def test_compute_offsets_basic(self) -> None:
        source = "First.\n\nSecond."
        segs = ["First.", "Second."]
        offsets = _compute_char_offsets(source, segs)
        assert offsets[0] == (0, 6)
        start, end = offsets[1]
        assert source[start:end] == "Second."
