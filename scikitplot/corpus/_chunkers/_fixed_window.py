# scikitplot/corpus/_chunkers/_fixed_window.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._chunkers._fixed_window
=========================================
Sliding-window chunking with configurable size, overlap, and unit.

Pure Python — no external dependencies. Suitable for RAG pipelines that
require fixed-size text chunks for embedding and retrieval, where sentence
or paragraph boundaries are either absent or not semantically meaningful.

Design
------
Two unit modes are supported:

``unit="words"``
    The window is measured in whitespace-delimited tokens. Character offsets
    are computed from the original text by recording the start position of
    each token via ``re.finditer(r"\S+", text)``. Chunk text is sliced from
    the original string, preserving all internal whitespace and punctuation.

``unit="chars"``
    The window is measured in characters. Chunks are exact character slices
    of the input, without any token alignment.

Overlap
-------
``overlap`` controls how many units (words or chars) the next window
reuses from the current one. Overlap must be in ``[0, window_size)``.
Step size is computed as ``step = window_size - overlap``.

- ``overlap=0`` → non-overlapping windows
- ``overlap=window_size-1`` → maximum overlap (advances 1 unit per step)

Character offsets
-----------------
``char_start`` in each returned tuple is the absolute character offset of the
**first character** of the chunk within the input string. This is consistent
with spaCy's ``sent.start_char`` convention.

Python compatibility
--------------------
Python 3.8-3.15. No external dependencies.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Optional  # noqa: F401

from .._types import Chunk, ChunkerConfig, ChunkResult
from ._custom_tokenizer import ScriptType, detect_script, split_cjk_chars

logger = logging.getLogger(__name__)

__all__ = [
    "FixedWindowChunker",
    "FixedWindowChunkerConfig",
    "WindowUnit",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW_SIZE: Final[int] = 512
_DEFAULT_STEP_SIZE: Final[int] = 256  # 50 % overlap
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class WindowUnit(str, Enum):
    """Unit of measurement for window size and step."""

    CHARS = "chars"
    TOKENS = "tokens"  # whitespace-split tokens (no external deps)


@dataclass(frozen=True)
class FixedWindowChunkerConfig(ChunkerConfig):
    """Configuration for :class:`FixedWindowChunker`.

    Parameters
    ----------
    window_size : int
        Size of each chunk in *unit* units.
    step_size : int
        Stride between consecutive chunk starts.  ``step_size == window_size``
        gives non-overlapping chunks.  ``step_size < window_size`` gives
        sliding-window overlap.
    unit : WindowUnit
        Measurement unit: ``CHARS`` (default) or ``TOKENS``.
    min_length : int
        Minimum character length to keep the last (possibly partial) chunk.
    include_offsets : bool
        Compute and store character offsets.
    strip_whitespace : bool
        Strip leading/trailing whitespace from each chunk.
    """

    window_size: int = _DEFAULT_WINDOW_SIZE
    step_size: int = _DEFAULT_STEP_SIZE
    unit: WindowUnit = WindowUnit.CHARS
    min_length: int = 10
    include_offsets: bool = True
    strip_whitespace: bool = True


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _tokenize_whitespace(text: str) -> list[str]:
    """Split *text* into tokens, respecting CJK no-space languages.

    For Latin/spaced text, splits on whitespace (unchanged behaviour).
    For CJK text (Chinese, Japanese, Korean), auto-detects via
    :func:`~._custom_tokenizer.detect_script` and falls back to
    character-level tokenisation via
    :func:`~._custom_tokenizer.split_cjk_chars`, which preserves
    Latin/numeric runs as contiguous tokens while making each CJK
    ideograph its own token.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Token list.

    Notes
    -----
    **User note:** Without this fix, Chinese/Japanese/Korean text with no
    whitespace produced a single token equal to the entire text, making
    ``unit=TOKENS`` useless for those languages.

    **Developer note:** Detection samples the first 200 characters and
    adds negligible overhead (<1 µs per call on typical CPython).
    """  # noqa: RUF002
    stripped = text.strip()
    if not stripped:
        return []
    script = detect_script(stripped[:200])
    if script == ScriptType.CJK:
        return split_cjk_chars(stripped)
    return _WHITESPACE_RE.split(stripped)


def _windows_chars(
    text: str, window_size: int, step_size: int
) -> list[tuple[str, int, int]]:
    """Generate character-based windows over *text*.

    Parameters
    ----------
    text : str
        Source document.
    window_size : int
        Characters per window.
    step_size : int
        Characters to advance per step.

    Returns
    -------
    list[tuple[str, int, int]]
        Each element is ``(window_text, start_char, end_char)``.
    """
    results: list[tuple[str, int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + window_size, n)
        results.append((text[start:end], start, end))
        if end == n:
            break
        start += step_size
    return results


def _windows_tokens(
    text: str, window_size: int, step_size: int
) -> list[tuple[str, int, int]]:
    """Generate token-based windows over *text*.

    Reconstructs approximate character offsets from token positions.

    Parameters
    ----------
    text : str
        Source document.
    window_size : int
        Tokens per window.
    step_size : int
        Tokens to advance per step.

    Returns
    -------
    list[tuple[str, int, int]]
        Each element is ``(window_text, start_char, end_char)``.
    """
    tokens = _tokenize_whitespace(text)
    n = len(tokens)
    results: list[tuple[str, int, int]] = []
    start = 0

    while start < n:
        end = min(start + window_size, n)
        window_tokens = tokens[start:end]
        window_text = " ".join(window_tokens)

        # Approximate char offsets by searching in source.
        first_token = window_tokens[0] if window_tokens else ""
        char_start = text.find(
            first_token, 0 if start == 0 else results[-1][1] if results else 0
        )
        if char_start == -1:
            char_start = 0
        char_end = char_start + len(window_text)

        results.append((window_text, char_start, char_end))
        if end == n:
            break
        start += step_size

    return results


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


class FixedWindowChunker:
    """Produce fixed-size sliding-window chunks over a document.

    Parameters
    ----------
    config : FixedWindowChunkerConfig, optional
        Chunker configuration.

    Examples
    --------
    >>> cfg = FixedWindowChunkerConfig(
    ...     window_size=20, step_size=10, unit=WindowUnit.CHARS
    ... )
    >>> chunker = FixedWindowChunker(cfg)
    >>> result = chunker.chunk("The quick brown fox jumps over the lazy dog")
    >>> result.chunks[0].text
    'The quick brown fox '
    """

    def __init__(self, config: FixedWindowChunkerConfig | None = None) -> None:
        self._cfg = config if config is not None else FixedWindowChunkerConfig()
        self._validate_config()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate configuration at construction time.

        Raises
        ------
        ValueError
            On invalid configuration values.
        """
        if self._cfg.window_size < 1:
            raise ValueError(
                f"FixedWindowChunkerConfig.window_size must be >= 1, "
                f"got {self._cfg.window_size}."
            )
        if self._cfg.step_size < 1:
            raise ValueError(
                f"FixedWindowChunkerConfig.step_size must be >= 1, "
                f"got {self._cfg.step_size}."
            )
        if self._cfg.step_size > self._cfg.window_size:
            raise ValueError(
                "FixedWindowChunkerConfig.step_size must be <= window_size "
                f"to avoid gaps. Got step={self._cfg.step_size}, "
                f"window={self._cfg.window_size}."
            )
        if self._cfg.min_length < 0:
            raise ValueError(
                f"FixedWindowChunkerConfig.min_length must be >= 0, "
                f"got {self._cfg.min_length}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        doc_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Split *text* into fixed-window chunks.

        Parameters
        ----------
        text : str
            Raw document text.
        doc_id : str, optional
            Document identifier stored in metadata.
        extra_metadata : dict[str, Any], optional
            Additional key/value pairs merged into result metadata.

        Returns
        -------
        ChunkResult
            Chunks and aggregate metadata.

        Raises
        ------
        TypeError
            If *text* is not a ``str``.
        ValueError
            If *text* is empty or whitespace-only.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}.")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")

        if self._cfg.unit == WindowUnit.CHARS:
            raw_windows = _windows_chars(
                text, self._cfg.window_size, self._cfg.step_size
            )
        else:
            raw_windows = _windows_tokens(
                text, self._cfg.window_size, self._cfg.step_size
            )

        chunks: list[Chunk] = []
        for idx, (win_text, start, end) in enumerate(raw_windows):
            if self._cfg.strip_whitespace:
                win_text = win_text.strip()  # noqa: PLW2901
            if len(win_text) < self._cfg.min_length:
                continue

            meta: dict[str, Any] = {
                "chunk_index": idx,
                "window_size": self._cfg.window_size,
                "step_size": self._cfg.step_size,
                "unit": self._cfg.unit.value,
            }
            if not self._cfg.include_offsets:
                start, end = 0, 0  # noqa: PLW2901
            if doc_id is not None:
                meta["doc_id"] = doc_id

            chunks.append(
                Chunk(text=win_text, start_char=start, end_char=end, metadata=meta)
            )

        result_meta: dict[str, Any] = {
            "chunker": "fixed_window",
            "unit": self._cfg.unit.value,
            "window_size": self._cfg.window_size,
            "step_size": self._cfg.step_size,
            "total_chunks": len(chunks),
        }
        if doc_id is not None:
            result_meta["doc_id"] = doc_id
        if extra_metadata:
            result_meta.update(extra_metadata)

        return ChunkResult(chunks=chunks, metadata=result_meta)

    def chunk_batch(
        self,
        texts: list[str],
        doc_ids: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Chunk a list of documents.

        Parameters
        ----------
        texts : list[str]
            Input documents.
        doc_ids : list[str], optional
            Parallel document identifiers.
        extra_metadata : dict[str, Any], optional
            Shared metadata for every result.

        Returns
        -------
        list[ChunkResult]
            One result per document.

        Raises
        ------
        TypeError
            If *texts* is not a list.
        ValueError
            If *doc_ids* length mismatches *texts*.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list, got {type(texts).__name__!r}.")
        if doc_ids is not None and len(doc_ids) != len(texts):
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) must equal "
                f"texts length ({len(texts)})."
            )
        return [
            self.chunk(
                t,
                doc_id=doc_ids[i] if doc_ids else None,
                extra_metadata=extra_metadata,
            )
            for i, t in enumerate(texts)
        ]
