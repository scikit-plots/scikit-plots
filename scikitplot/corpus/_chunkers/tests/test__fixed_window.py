"""Tests for scikitplot.corpus._chunkers._fixed_window."""

from __future__ import annotations

import pytest

from .._fixed_window import (
    FixedWindowChunker,
    FixedWindowChunkerConfig,
    WindowUnit,
    _windows_chars,
    _windows_tokens,
)
from ..._types import Chunk, ChunkResult

TEXT = "The quick brown fox jumps over the lazy dog. " * 5


@pytest.fixture()
def char_chunker() -> FixedWindowChunker:
    return FixedWindowChunker(
        FixedWindowChunkerConfig(
            window_size=50, step_size=25, unit=WindowUnit.CHARS
        )
    )


@pytest.fixture()
def token_chunker() -> FixedWindowChunker:
    return FixedWindowChunker(
        FixedWindowChunkerConfig(
            window_size=5, step_size=3, unit=WindowUnit.TOKENS
        )
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestFixedWindowConfig:
    def test_window_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            FixedWindowChunker(FixedWindowChunkerConfig(window_size=0))

    def test_step_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="step_size"):
            FixedWindowChunker(
                FixedWindowChunkerConfig(window_size=10, step_size=0)
            )

    def test_step_greater_than_window_raises(self) -> None:
        with pytest.raises(ValueError, match="step_size"):
            FixedWindowChunker(
                FixedWindowChunkerConfig(window_size=10, step_size=20)
            )

    def test_negative_min_length_raises(self) -> None:
        with pytest.raises(ValueError, match="min_length"):
            FixedWindowChunker(FixedWindowChunkerConfig(min_length=-1))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_string_raises(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        with pytest.raises(TypeError, match="str"):
            char_chunker.chunk(None)  # type: ignore[arg-type]

    def test_empty_string_raises(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            char_chunker.chunk("")


# ---------------------------------------------------------------------------
# Character windows
# ---------------------------------------------------------------------------


class TestCharWindows:
    def test_returns_chunk_result(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        result = char_chunker.chunk(TEXT)
        assert isinstance(result, ChunkResult)

    def test_chunks_are_chunk_instances(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        result = char_chunker.chunk(TEXT)
        for c in result.chunks:
            assert isinstance(c, Chunk)

    def test_chunk_length_at_most_window_size(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        result = char_chunker.chunk(TEXT)
        for c in result.chunks:
            assert len(c.text) <= 50 + 5   # strip tolerance

    def test_overlapping_windows_more_chunks(self) -> None:
        non_overlap = FixedWindowChunker(
            FixedWindowChunkerConfig(window_size=50, step_size=50)
        )
        overlap = FixedWindowChunker(
            FixedWindowChunkerConfig(window_size=50, step_size=25)
        )
        r1 = non_overlap.chunk(TEXT)
        r2 = overlap.chunk(TEXT)
        assert len(r2.chunks) >= len(r1.chunks)

    def test_metadata_has_window_size(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        result = char_chunker.chunk(TEXT)
        for c in result.chunks:
            assert c.metadata["window_size"] == 50
            assert c.metadata["step_size"] == 25

    def test_doc_id_in_metadata(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        result = char_chunker.chunk(TEXT, doc_id="doc_x")
        for c in result.chunks:
            assert c.metadata["doc_id"] == "doc_x"

    def test_offsets_disabled(self) -> None:
        chunker = FixedWindowChunker(
            FixedWindowChunkerConfig(
                window_size=50, step_size=25, include_offsets=False
            )
        )
        result = chunker.chunk(TEXT)
        for c in result.chunks:
            assert c.start_char == 0
            assert c.end_char == 0

    def test_result_metadata_chunker_key(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        result = char_chunker.chunk(TEXT)
        assert result.metadata["chunker"] == "fixed_window"


# ---------------------------------------------------------------------------
# Token windows
# ---------------------------------------------------------------------------


class TestTokenWindows:
    def test_token_mode_works(
        self, token_chunker: FixedWindowChunker
    ) -> None:
        result = token_chunker.chunk(TEXT)
        assert len(result.chunks) > 0

    def test_token_unit_in_metadata(
        self, token_chunker: FixedWindowChunker
    ) -> None:
        result = token_chunker.chunk(TEXT)
        assert result.metadata["unit"] == "tokens"


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_basic(self, char_chunker: FixedWindowChunker) -> None:
        results = char_chunker.chunk_batch([TEXT, TEXT[:50]])
        assert len(results) == 2

    def test_batch_type_error(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        with pytest.raises(TypeError, match="list"):
            char_chunker.chunk_batch("not a list")  # type: ignore[arg-type]

    def test_batch_doc_ids_mismatch(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        with pytest.raises(ValueError, match="length"):
            char_chunker.chunk_batch([TEXT], doc_ids=["a", "b"])


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_windows_chars_non_overlapping(self) -> None:
        text = "abcdefghij"
        windows = _windows_chars(text, window_size=5, step_size=5)
        assert len(windows) == 2
        assert windows[0][0] == "abcde"
        assert windows[1][0] == "fghij"

    def test_windows_chars_overlapping(self) -> None:
        text = "abcdefghij"
        windows = _windows_chars(text, window_size=6, step_size=3)
        assert len(windows) == 3

    def test_windows_tokens_basic(self) -> None:
        text = "one two three four five"
        windows = _windows_tokens(text, window_size=3, step_size=2)
        assert windows[0][0] == "one two three"
        assert "three four five" in windows[1][0]

    def test_windows_chars_offsets_correct(self) -> None:
        text = "abcdefghij"
        windows = _windows_chars(text, window_size=5, step_size=5)
        for win_text, start, end in windows:
            assert text[start:end] == win_text
