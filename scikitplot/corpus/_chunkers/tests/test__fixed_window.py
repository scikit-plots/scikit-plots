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


# ---------------------------------------------------------------------------
# _tokenize_whitespace — CJK auto-detect and edge cases
# ---------------------------------------------------------------------------


class TestTokenizeWhitespace:
    """Tests for the private ``_tokenize_whitespace`` helper."""

    from .._fixed_window import _tokenize_whitespace  # noqa: PLC0415

    def _tok(self, text: str) -> list:
        from .._fixed_window import _tokenize_whitespace  # noqa: PLC0415

        return _tokenize_whitespace(text)

    def test_latin_whitespace_split(self) -> None:
        """Latin text must be split on whitespace."""
        result = self._tok("the quick brown fox")
        assert result == ["the", "quick", "brown", "fox"]

    def test_empty_string_returns_empty(self) -> None:
        """Empty string must return an empty list."""
        assert self._tok("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only string must return an empty list."""
        assert self._tok("   ") == []

    def test_cjk_chinese_character_level(self) -> None:
        """CJK text must be tokenised at the character level."""
        result = self._tok("你好世界")
        # Each ideograph becomes its own token
        assert "你" in result
        assert "世" in result
        assert len(result) >= 3

    def test_cjk_path_returns_nonempty(self) -> None:
        """CJK text must produce a non-empty token list."""
        result = self._tok("今日は良い天気です")
        assert len(result) > 0

    def test_mixed_cjk_latin_handled(self) -> None:
        """Mixed CJK+Latin text must produce at least one token per word."""
        result = self._tok("hello 世界 world")
        assert len(result) >= 3

    def test_single_latin_word(self) -> None:
        """Single word must yield exactly one token."""
        assert self._tok("hello") == ["hello"]


# ---------------------------------------------------------------------------
# chunk_batch with extra_metadata
# ---------------------------------------------------------------------------


class TestBatchExtraMetadata:
    """Verify ``extra_metadata`` flows into batch results."""

    def test_batch_extra_metadata_merged(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        extra = {"pipeline": "test", "version": 42}
        results = char_chunker.chunk_batch([TEXT, TEXT[:60]], extra_metadata=extra)
        for r in results:
            assert r.metadata["pipeline"] == "test"
            assert r.metadata["version"] == 42

    def test_batch_with_doc_ids_and_extra_metadata(
        self, char_chunker: FixedWindowChunker
    ) -> None:
        results = char_chunker.chunk_batch(
            [TEXT, TEXT[:60]],
            doc_ids=["doc_a", "doc_b"],
            extra_metadata={"source": "test"},
        )
        assert results[0].metadata["doc_id"] == "doc_a"
        assert results[1].metadata["doc_id"] == "doc_b"
        for r in results:
            assert r.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# _windows_tokens — offset search fallback
# ---------------------------------------------------------------------------


class TestWindowsTokensFallback:
    """Edge cases for ``_windows_tokens`` char-offset computation."""

    def test_single_token_window(self) -> None:
        """Single-token text must produce exactly one window."""
        windows = _windows_tokens("hello", window_size=1, step_size=1)
        assert len(windows) == 1
        assert windows[0][0] == "hello"

    def test_non_overlapping_tokens(self) -> None:
        """Non-overlapping step must produce the expected number of windows."""
        text = "one two three four"
        windows = _windows_tokens(text, window_size=2, step_size=2)
        assert len(windows) == 2

    def test_window_char_start_non_negative(self) -> None:
        """char_start for all windows must be >= 0."""
        text = "alpha beta gamma delta"
        windows = _windows_tokens(text, window_size=2, step_size=1)
        for _, start, _ in windows:
            assert start >= 0
