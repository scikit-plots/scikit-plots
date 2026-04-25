# scikitplot/corpus/_chunkers/_paragraph.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._chunkers._paragraph
======================================
Paragraph-boundary segmentation via blank-line splitting.

Pure Python — no external dependencies. Suitable for plain text, pre-processed
XML body text, and any format where paragraph boundaries are expressed by one
or more consecutive blank lines (``\n\n``).

Design:

A "blank line" is any line that contains only whitespace characters.
The split regex ``\n\s*\n+`` matches one newline, any amount of
whitespace (including none), followed by one or more additional newlines.
This correctly handles:

- Standard double-newline (``\n\n``)
- Windows-style line endings after normalisation (``\n`` after ``\r\n`` strip)
- Extra blank lines (``\n\n\n``) collapsed to one boundary
- Lines containing only spaces or tabs between newlines

Character offsets:

``char_start`` in each returned tuple is the absolute offset of the **first
non-whitespace character** of the paragraph within the input string. This
matches spaCy's ``sent.start_char`` convention used by
:class:`~scikitplot.corpus._chunkers.SentenceChunker`.

Python compatibility:

Python 3.8-3.15. No external dependencies.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Final

from .._types import Chunk, ChunkerConfig, ChunkResult

logger = logging.getLogger(__name__)

__all__ = [
    "ParagraphChunker",
    "ParagraphChunkerConfig",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MIN_LEN: Final[int] = 0  # no lower limit by default
_DEFAULT_MAX_LEN: Final[int | None] = None  # no upper limit by default
_DEFAULT_OVERLAP: Final[int] = 0  # no by default

# One or more blank lines (optional CR) separate paragraphs.
_PARA_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"\r?\n\s*\r?\n+")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParagraphChunkerConfig(ChunkerConfig):
    """Configuration for :class:`ParagraphChunker`.

    Parameters
    ----------
    min_length : int
        Minimum character length to retain a paragraph.
    max_length : int or None
        Maximum character length.  Paragraphs exceeding this are split
        at sentence boundaries (``[.!?]``).  ``None`` disables the limit.
    overlap : int
        Number of preceding paragraphs prepended as context.
    strip_whitespace : bool
        Strip leading/trailing whitespace from each paragraph.
    include_offsets : bool
        Compute and store character offsets.
    merge_short : bool
        Merge consecutive short paragraphs (below *min_length*) into
        one block instead of discarding them.
    """

    min_length: int = _DEFAULT_MIN_LEN
    max_length: int | None = _DEFAULT_MAX_LEN
    overlap: int = _DEFAULT_OVERLAP
    strip_whitespace: bool = True
    include_offsets: bool = True
    merge_short: bool = False


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str, strip: bool) -> list[str]:
    """Split *text* on blank lines.

    Parameters
    ----------
    text : str
        Input document.
    strip : bool
        Whether to strip each paragraph.

    Returns
    -------
    list[str]
        Non-empty paragraph strings.
    """
    parts = _PARA_SPLIT_RE.split(text)
    if strip:
        parts = [p.strip() for p in parts]
    return [p for p in parts if p]


def _split_long_paragraph(para: str, max_len: int) -> list[str]:
    """Divide an oversized paragraph at sentence boundaries.

    Falls back to hard character splitting when no sentence boundary
    is found within *max_len* characters.

    Parameters
    ----------
    para : str
        Single paragraph text.
    max_len : int
        Target maximum length for sub-paragraphs.

    Returns
    -------
    list[str]
        Sub-paragraph strings, each ``<= max_len`` characters when possible.
    """
    sentence_boundary = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_boundary.split(para)
    chunks: list[str] = []
    current: list[str] = []
    current_len: int = 0

    for sent in sentences:
        candidate_len = current_len + len(sent) + (1 if current else 0)
        if current and candidate_len > max_len:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c]


def _merge_short_paragraphs(paragraphs: list[str], min_length: int) -> list[str]:
    """Merge consecutive short paragraphs into one.

    Parameters
    ----------
    paragraphs : list[str]
        Input paragraph list.
    min_length : int
        Paragraphs shorter than this are merged with their successor.

    Returns
    -------
    list[str]
        Merged paragraph list.
    """
    merged: list[str] = []
    buffer: list[str] = []

    for para in paragraphs:
        if len(para) < min_length:
            buffer.append(para)
        else:  # noqa: PLR5501
            if buffer:
                buffer.append(para)
                merged.append("\n".join(buffer))
                buffer = []
            else:
                merged.append(para)

    if buffer:
        merged.append("\n".join(buffer))

    return merged


def _compute_char_offsets(source: str, segments: list[str]) -> list[tuple[int, int]]:
    """Compute ``(start_char, end_char)`` for each segment in *source*.

    Parameters
    ----------
    source : str
        Original document string.
    segments : list[str]
        Ordered paragraph strings.

    Returns
    -------
    list[tuple[int, int]]
        Character index pairs.
    """
    offsets: list[tuple[int, int]] = []
    cursor: int = 0
    for seg in segments:
        idx = source.find(seg, cursor)
        if idx == -1:
            idx = cursor
        offsets.append((idx, idx + len(seg)))
        cursor = idx + len(seg)
    return offsets


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


class ParagraphChunker:
    r"""
    Split a document into paragraph-level :class:`~.._types.Chunk` objects.

    Parameters
    ----------
    config : ParagraphChunkerConfig, optional
        Chunker configuration.

    Examples
    --------
    >>> chunker = ParagraphChunker()
    >>> text = "First paragraph.\n\nSecond paragraph."
    >>> result = chunker.chunk(text)
    >>> len(result.chunks)
    2
    """

    def __init__(self, config: ParagraphChunkerConfig | None = None) -> None:
        self._cfg = config if config is not None else ParagraphChunkerConfig()
        self._validate_config()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate config values at construction time.

        Raises
        ------
        ValueError
            On invalid config values.
        """
        if self._cfg.min_length < 0:
            raise ValueError(
                f"ParagraphChunkerConfig.min_length must be >= 0, "
                f"got {self._cfg.min_length}."
            )
        if self._cfg.max_length is not None and self._cfg.max_length < 1:
            raise ValueError(
                f"ParagraphChunkerConfig.max_length must be >= 1 or None, "
                f"got {self._cfg.max_length}."
            )
        if (
            self._cfg.max_length is not None
            and self._cfg.max_length < self._cfg.min_length
        ):
            raise ValueError(
                "ParagraphChunkerConfig.max_length must be >= min_length. "
                f"Got max_length={self._cfg.max_length}, "
                f"min_length={self._cfg.min_length}."
            )
        if self._cfg.overlap < 0:
            raise ValueError(
                f"ParagraphChunkerConfig.overlap must be >= 0, got {self._cfg.overlap}."
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
        """Split *text* into paragraph-level chunks.

        Parameters
        ----------
        text : str
            Raw document text.
        doc_id : str, optional
            Document identifier stored in each chunk's metadata.
        extra_metadata : dict[str, Any], optional
            Additional key/value pairs merged into the result metadata.

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

        paragraphs = _split_paragraphs(text, strip=self._cfg.strip_whitespace)

        # Optional: merge short paragraphs before filtering.
        if self._cfg.merge_short:
            paragraphs = _merge_short_paragraphs(paragraphs, self._cfg.min_length)

        # Filter by minimum length.
        paragraphs = [p for p in paragraphs if len(p) >= self._cfg.min_length]

        # Split overly long paragraphs.
        if self._cfg.max_length is not None:
            expanded: list[str] = []
            for para in paragraphs:
                if len(para) > self._cfg.max_length:
                    expanded.extend(_split_long_paragraph(para, self._cfg.max_length))
                else:
                    expanded.append(para)
            paragraphs = expanded

        offsets: list[tuple[int, int]] = (
            _compute_char_offsets(text, paragraphs)
            if self._cfg.include_offsets
            else [(0, 0)] * len(paragraphs)
        )

        chunks: list[Chunk] = []
        for idx, para in enumerate(paragraphs):
            overlap_start = max(0, idx - self._cfg.overlap)
            context = paragraphs[overlap_start:idx]
            full_text = "\n\n".join([*context, para]) if context else para

            meta: dict[str, Any] = {
                "chunk_index": idx,
                "paragraph_index": idx,
                "overlap_count": len(context),
            }
            if doc_id is not None:
                meta["doc_id"] = doc_id

            start, end = offsets[idx]
            chunks.append(
                Chunk(text=full_text, start_char=start, end_char=end, metadata=meta)
            )

        result_meta: dict[str, Any] = {
            "chunker": "paragraph",
            "total_chunks": len(chunks),
            "min_length": self._cfg.min_length,
            "max_length": self._cfg.max_length,
            "overlap": self._cfg.overlap,
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
