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
from ._multilang_mixin import MultilangConfig, MultilangMixin
from ._sentence import _validate_text_input
from ._spans import Span, cover, strip_span

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
_SENTENCE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")


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
    # Multilang settings (all 5 chunkers)
    multilang_config: MultilangConfig | None = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _paragraph_spans(text: str, strip: bool) -> list[Span]:
    """Return the span of each paragraph (text between blank lines).

    Parameters
    ----------
    text : str
        Input document.
    strip : bool
        Narrow each span to exclude surrounding whitespace.

    Returns
    -------
    list of (int, int)
        Non-empty paragraph spans in source order (R3-3: spans, not strings,
        so every later step can slice the source instead of re-joining).
    """
    spans: list[Span] = []
    position = 0
    for separator in _PARA_SPLIT_RE.finditer(text):
        spans.append((position, separator.start()))
        position = separator.end()
    spans.append((position, len(text)))
    if strip:
        spans = [strip_span(text, start, end) for start, end in spans]
    return [(start, end) for start, end in spans if end > start]


def _split_long_span(text: str, span: Span, max_len: int) -> list[Span]:
    """Divide an oversized paragraph span at sentence boundaries.

    Sentences (ending in ``.``, ``!`` or ``?``) are grouped greedily while the
    group's source length stays within *max_len*. A single sentence longer
    than *max_len* stays whole: there is no character-level split.

    Parameters
    ----------
    text : str
        Source document.
    span : (int, int)
        The paragraph's span.
    max_len : int
        Target maximum length, in source characters.

    Returns
    -------
    list of (int, int)
        Sub-spans covering the sentences of *span*, in order.
    """
    start, end = span
    sentences: list[Span] = []
    position = start
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text, start, end):
        sentences.append((position, boundary.start()))
        position = boundary.end()
    sentences.append((position, end))
    sentences = [(s, e) for s, e in sentences if e > s]

    groups: list[Span] = []
    group: list[Span] = []
    for sentence in sentences:
        if group and sentence[1] - group[0][0] > max_len:
            groups.append(cover(group))
            group = [sentence]
        else:
            group.append(sentence)
    if group:
        groups.append(cover(group))
    return groups


def _merge_short_spans(text: str, spans: list[Span], min_length: int) -> list[Span]:
    """Merge each run of short paragraphs into the next one.

    Parameters
    ----------
    text : str
        Source document (a merged span keeps its real separators).
    spans : list of (int, int)
        Paragraph spans.
    min_length : int
        Paragraphs shorter than this join the following paragraph; a
        trailing run joins into one span.

    Returns
    -------
    list of (int, int)
        Merged spans.
    """
    merged: list[Span] = []
    buffer: list[Span] = []
    for span in spans:
        if span[1] - span[0] < min_length:
            buffer.append(span)
        elif buffer:
            merged.append(cover([*buffer, span]))
            buffer = []
        else:
            merged.append(span)
    if buffer:
        merged.append(cover(buffer))
    return merged


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


class ParagraphChunker(MultilangMixin):
    r"""Split a document into paragraph-level :class:`~.._types.Chunk` objects.

    Paragraph boundaries are blank lines (``\n\n``) — script-universal
    and dependency-free.  Within each paragraph the dominant Unicode script
    is detected and reported in ``chunk.metadata["multilang"]["script"]``.

    Parameters
    ----------
    config : ParagraphChunkerConfig, optional
        Chunker configuration.
    multilang_config : MultilangConfig, optional
        Multilang feature flags.  Overrides ``config.multilang_config``
        when provided explicitly.  Default: ``MultilangConfig()``
        (script detection + semanteme count only, no raw text / trace).

    Notes
    -----
    **User note (multilang):** Blank-line paragraph splitting works for
    every script.  Set ``multilang_config=MultilangConfig(
    include_raw_text=True, include_preprocessing_trace=True)`` to attach
    full preprocessing audit to each chunk.  To add embeddings after
    chunking, call :meth:`attach_embedding_batch`.

    **Developer note:** Inherits :class:`MultilangMixin`.  Initialised via
    ``self._ml_init()``.  The ``chunk()`` method enriches each paragraph
    chunk with a :class:`~.._types.MultilangChunkMeta` dict stored under
    ``chunk.metadata["multilang"]``.

    Examples
    --------
    >>> chunker = ParagraphChunker()
    >>> text = "First paragraph.\n\nSecond paragraph."
    >>> result = chunker.chunk(text)
    >>> len(result.chunks)
    2
    >>> chunker_ml = ParagraphChunker(
    ...     multilang_config=MultilangConfig(include_raw_text=True)
    ... )
    >>> result_ml = chunker_ml.chunk("مرحبا.\\n\\nHello.")
    >>> result_ml.chunks[0].metadata["multilang"]["script"]
    'arabic'
    """

    #: The configuration dataclass this chunker takes as ``config=``; the
    #: component registry pairs them by it (R3-1).
    config_class = ParagraphChunkerConfig

    #: Whether each chunk's text is ``source[context_span]`` (R3-3).
    text_is_source_slice = True

    def __init__(
        self,
        config: ParagraphChunkerConfig | None = None,
        multilang_config: MultilangConfig | None = None,
    ) -> None:
        if config is not None and not isinstance(config, ParagraphChunkerConfig):
            raise TypeError(
                f"ParagraphChunker: config must be a ParagraphChunkerConfig or None, "
                f"got {type(config).__name__}."
            )
        self._cfg = config if config is not None else ParagraphChunkerConfig()
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
    def config(self) -> ParagraphChunkerConfig:
        """The resolved :class:`ParagraphChunkerConfig` for this instance (read-only)."""
        return self._cfg

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

    def chunk(  # noqa: PLR0912
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
        # Bug 4 fix: NUL byte + lone surrogate rejection (shared guard)
        _validate_text_input(text, "ParagraphChunker.chunk")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")

        # Multilang: build preprocessing trace from raw → NFC
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

        # R3-3: paragraphs are spans of the source; merging and splitting work
        # on spans, and every chunk's text is a slice (re-joining with "\n" or
        # " " produced text the source never had, then a failed search fell
        # back to a wrong position).
        spans = _paragraph_spans(text, strip=self._cfg.strip_whitespace)

        # Optional: merge short paragraphs before filtering.
        if self._cfg.merge_short:
            spans = _merge_short_spans(text, spans, self._cfg.min_length)

        # Filter by minimum length.
        spans = [(s, e) for s, e in spans if e - s >= self._cfg.min_length]

        # Split overly long paragraphs.
        if self._cfg.max_length is not None:
            spans = [
                piece
                for span in spans
                for piece in (
                    _split_long_span(text, span, self._cfg.max_length)
                    if span[1] - span[0] > self._cfg.max_length
                    else [span]
                )
            ]

        paragraphs = [text[s:e] for s, e in spans]
        offsets: list[tuple[int, int]] = (
            list(spans) if self._cfg.include_offsets else [(0, 0)] * len(spans)
        )

        chunks: list[Chunk] = []
        for idx, para in enumerate(paragraphs):
            overlap_start = max(0, idx - self._cfg.overlap)
            context = paragraphs[overlap_start:idx]
            # The context is the source slice from the first overlapped
            # paragraph to this one, so a paragraph min_length dropped between
            # them is part of it (review F5; the text is the source's own).
            full_text = (
                text[spans[overlap_start][0] : spans[idx][1]] if context else para
            )

            meta: dict[str, Any] = {
                "chunk_index": idx,
                "paragraph_index": idx,
                "overlap_count": len(context),
            }
            if doc_id is not None:
                meta["doc_id"] = doc_id

            start, end = offsets[idx]
            chunk = Chunk(
                text=full_text,
                start_char=start,
                end_char=end,
                metadata=meta,
                context_start=(
                    spans[overlap_start][0]
                    if context and self._cfg.include_offsets
                    else None
                ),
            )
            # Multilang enrichment
            if self._ml_cfg.enabled:
                # The context's raw text, as the document spans the context;
                # script detection below still uses the paragraph alone.
                para_raw = (
                    raw_text[(spans[overlap_start][0] if context else start) : end]
                    if self._ml_cfg.include_raw_text
                    and start < end
                    and span_basis == "input"
                    else None
                )
                ml_meta = self._ml_build_meta(
                    para,
                    chunking_unit="paragraph",
                    tokens=para.split() if para else [],
                    raw_text=para_raw,
                    preprocessing_trace=preprocessing_trace,
                )
                chunk = self._ml_enrich_chunk(chunk, ml_meta)
            chunks.append(chunk)

        result_meta: dict[str, Any] = {
            "span_basis": span_basis,
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
