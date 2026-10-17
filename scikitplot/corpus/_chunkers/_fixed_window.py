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

Design:

Two unit modes are supported:

``unit="words"``
    The window is measured in whitespace-delimited tokens. Character offsets
    are computed from the original text by recording the start position of
    each token via ``re.finditer(r"\S+", text)``. Chunk text is sliced from
    the original string, preserving all internal whitespace and punctuation.

``unit="chars"``
    The window is measured in characters. Chunks are exact character slices
    of the input, without any token alignment.

Overlap:

``overlap`` controls how many units (words or chars) the next window
reuses from the current one. Overlap must be in ``[0, window_size)``.
Step size is computed as ``step = window_size - overlap``.

- ``overlap=0`` → non-overlapping windows
- ``overlap=window_size-1`` → maximum overlap (advances 1 unit per step)

Character offsets:

``char_start`` in each returned tuple is the absolute character offset of the
**first character** of the chunk within the input string. This is consistent
with spaCy's ``sent.start_char`` convention.

Python compatibility:

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
from ._multilang_mixin import MultilangConfig, MultilangMixin
from ._sentence import _validate_text_input
from ._spans import cover, locate, strip_span

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
    # Multilang settings
    multilang_config: MultilangConfig | None = None


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
    if script in (
        ScriptType.CJK,  # deprecated alias — backward compat
        ScriptType.HAN,  # Chinese logographs
        ScriptType.HIRAGANA,  # Japanese Hiragana
        ScriptType.KATAKANA,  # Japanese Katakana
    ):
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

    Each window is the source text from its first token to its last.

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
    # R3-3: each token is located in the source and a window is the source
    # from its first token to its last, separators included (re-joining with
    # " " put spaces between CJK characters and collapsed runs of
    # whitespace, and the window's end was estimated from that join).
    spans = locate(text, tokens, where="FixedWindowChunker (tokens)")
    n = len(spans)
    results: list[tuple[str, int, int]] = []
    start = 0
    while start < n:
        end = min(start + window_size, n)
        char_start, char_end = cover(spans[start:end])
        results.append((text[char_start:char_end], char_start, char_end))
        if end == n:
            break
        start += step_size
    return results


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


class FixedWindowChunker(MultilangMixin):
    """Produce fixed-size sliding-window chunks over a document.

    Handles all scripts via :func:`~._custom_tokenizer.detect_script` and
    :func:`~._custom_tokenizer.split_cjk_chars` for no-space East Asian scripts.
    Multilang metadata is attached to each chunk when enabled.

    Parameters
    ----------
    config : FixedWindowChunkerConfig, optional
        Chunker configuration.
    multilang_config : MultilangConfig, optional
        Multilang feature flags.

    Notes
    -----
    **User note (multilang):** Fixed-window chunking is script-aware for
    token-unit mode (``unit=TOKENS``): CJK / Hiragana / Katakana text is
    split at character level rather than whitespace.  Char-unit mode
    (``unit=CHARS``) is strictly grapheme-cluster agnostic (raw codepoint
    slices), which is safe for RAG pipelines that only need byte-aligned
    embedding windows.

    **Developer note:** Inherits :class:`MultilangMixin`.  Every chunk
    produced by :meth:`chunk` carries ``metadata["multilang"]`` when
    ``multilang_config.enabled=True``.

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

    #: The configuration dataclass this chunker takes as ``config=``; the
    #: component registry pairs them by it (R3-1).
    config_class = FixedWindowChunkerConfig

    #: Whether each chunk's text is ``source[context_span]`` (R3-3).
    text_is_source_slice = True

    def __init__(
        self,
        config: FixedWindowChunkerConfig | None = None,
        multilang_config: MultilangConfig | None = None,
    ) -> None:
        if config is not None and not isinstance(config, FixedWindowChunkerConfig):
            raise TypeError(
                f"FixedWindowChunker: config must be a FixedWindowChunkerConfig or None, "
                f"got {type(config).__name__}."
            )
        self._cfg = config if config is not None else FixedWindowChunkerConfig()
        self._validate_config()
        ml_cfg = multilang_config or (
            self._cfg.multilang_config
            if isinstance(getattr(self._cfg, "multilang_config", None), MultilangConfig)
            else None
        )
        self._ml_init(ml_cfg)

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> FixedWindowChunkerConfig:
        """The resolved :class:`FixedWindowChunkerConfig` for this instance (read-only)."""
        return self._cfg

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

    def chunk(  # noqa: PLR0912
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
        _validate_text_input(text, "FixedWindowChunker.chunk")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")

        # Multilang: preprocessing trace
        raw_text = text
        preprocessing_trace = None
        if self._ml_cfg.enabled and (
            self._ml_cfg.include_preprocessing_trace or self._ml_cfg.include_raw_text
        ):
            text, preprocessing_trace = self._ml_build_preprocessing_trace(raw_text)
        # R3-3: spans index the text that was split. When preprocessing
        # (NFC, BOM/control stripping) changed it, they index that text, not
        # the input: say so, and slice no "raw" text with them (review F2).
        span_basis = "input" if text == raw_text else "preprocessed"

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
                # R3-3: narrow the span with the text, so text == source[span].
                start, end = strip_span(text, start, end)  # noqa: PLW2901
                win_text = text[start:end]  # noqa: PLW2901
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

            chunk = Chunk(text=win_text, start_char=start, end_char=end, metadata=meta)

            # Multilang enrichment
            if self._ml_cfg.enabled:
                win_raw = (
                    raw_text[start:end]
                    if self._ml_cfg.include_raw_text
                    and start < end
                    and span_basis == "input"
                    else None
                )
                ml_meta = self._ml_build_meta(
                    win_text,
                    chunking_unit="fixed_window",
                    tokens=win_text.split() if win_text else [],
                    raw_text=win_raw,
                    preprocessing_trace=preprocessing_trace,
                )
                chunk = self._ml_enrich_chunk(chunk, ml_meta)
            chunks.append(chunk)

        result_meta: dict[str, Any] = {
            "span_basis": span_basis,
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
