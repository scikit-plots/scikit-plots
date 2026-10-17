# scikitplot/corpus/_chunkers/_recursive.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Recursive and parent-child chunking (Run 4, R4-2).

:class:`RecursiveChunker` cuts a text by a hierarchy of separators (blank
line, line, sentence end, space, character) until every piece fits a
character budget. :class:`ParentChildChunker` applies the same rule twice:
large parents that give context, and small children inside each parent that
are retrieved; a child names its parent, and :class:`~.._base.DocumentReader`
turns that into ``parent_doc_id``.

Notes
-----
**User.** Neither is a default: each is measured against the fixed-unit
chunkers in the evaluation harness first.

**Developer.** Both work on spans (R3-3): the pieces of a span partition it
(each separator stays with the piece before it), merged pieces are covers of
adjacent spans, and every chunk's text is ``source[context_span]``. No
string is ever re-joined, and there is no search for a piece's position.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Iterator

from .._schema import ChunkingStrategy
from .._types import Chunk, ChunkerConfig, ChunkResult
from ._sentence import _validate_text_input
from ._spans import Span, strip_span

logger = logging.getLogger(__name__)

__all__ = [
    "ParentChildChunker",
    "ParentChildChunkerConfig",
    "RecursiveChunker",
    "RecursiveChunkerConfig",
]

#: Blank line, line, sentence end, space, then single characters.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def _validate_separators(separators: Any, owner: str) -> None:
    """Refuse a separator list that cannot always reach the budget."""
    if not isinstance(separators, tuple):
        raise TypeError(f"{owner}.separators must be a tuple of str.")
    if not all(isinstance(sep, str) for sep in separators):
        raise TypeError(f"{owner}.separators must hold only str.")
    if not separators:
        raise ValueError(f"{owner}.separators must not be empty.")
    if separators[-1] != "":
        raise ValueError(
            f"{owner}.separators must end with '' (cut characters), so every"
            " piece can be made to fit the budget."
        )
    if len(set(separators)) != len(separators):
        raise ValueError(f"{owner}.separators must not repeat a separator.")


def _coerce_separators(config: Any) -> None:
    """Accept a list of separators (a JSON or YAML configuration) as a tuple."""
    if isinstance(config.separators, list):
        object.__setattr__(config, "separators", tuple(config.separators))


def _positive(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int; got {value!r}.")


_NON_SPACE = re.compile(r"\S")


def _bounds(source: str, start: int, end: int) -> tuple[int, int] | None:
    """Return the first and one-past-last non-whitespace positions, or ``None``.

    The scan covers only the whitespace at either end of the span.
    """
    first = _NON_SPACE.search(source, start, end)
    if first is None:
        return None
    last = end
    while source[last - 1].isspace():
        last -= 1
    return first.start(), last


def _cut(source: str, span: Span, separator: str) -> Iterator[Span]:
    """Yield the pieces of ``span`` after each occurrence of ``separator``.

    The separator stays with the piece before it, so the pieces partition
    ``span``. Must not be called with ``""`` (see :func:`_character_windows`).
    A generator, so a span of many tiny pieces (a long run of spaces) never
    holds one tuple per piece.
    """
    start, end = span
    position = start
    while True:
        found = source.find(separator, position, end)
        if found == -1:
            break
        cut = found + len(separator)
        yield (position, cut)
        position = cut
    if position < end:
        yield (position, end)


def _character_windows(source: str, span: Span, size: int) -> list[Span]:
    """Partition ``span`` into windows of at most ``size`` non-space-bounded characters.

    The same result as cutting after every character and merging greedily,
    computed without one piece per character: a window runs from its start
    to just before the first non-whitespace character that would make its
    stripped length exceed ``size``.
    """
    start, end = span
    windows: list[Span] = []
    position = start
    while position < end:
        first = _NON_SPACE.search(source, position, end)
        if first is None:
            windows.append((position, end))
            break
        limit = first.start() + size
        if limit >= end:
            windows.append((position, end))
            break
        following = _NON_SPACE.search(source, limit, end)
        cut = end if following is None else following.start()
        windows.append((position, cut))
        position = cut
    return windows


def recursive_spans(
    source: str,
    span: Span,
    size: int,
    separators: tuple[str, ...],
) -> list[Span]:
    """Return adjacent spans that partition ``span``, each at most ``size``.

    Parameters
    ----------
    source : str
        The text the span indexes.
    span : (int, int)
        The region to split.
    size : int
        The largest span length, in characters, not counting the whitespace
        around it.
    separators : tuple of str
        Separators, coarsest first, ending with ``""``.

    Returns
    -------
    list of (int, int)
        Spans in source order; together they cover ``span`` exactly, and
        each holds at most ``size`` characters once its surrounding
        whitespace is stripped.

    Notes
    -----
    **Developer.** The first separator the span contains is used. Pieces
    are merged left to right while the merge fits ``size``; a piece longer
    than ``size`` is split again with the remaining separators; the first
    part of such a split does not merge back into the piece before it (as
    in LangChain's recursive splitter), the next piece may merge into its
    last part. Each merge test is O(1): the
    stripped bounds of every piece are computed once, so a level costs
    O(len(span)), and ``""`` emits windows directly. The result is
    deterministic, and ``""`` guarantees termination.
    """
    start, end = span
    whole = _bounds(source, start, end)
    if whole is None or whole[1] - whole[0] <= size:
        return [span]
    for level, separator in enumerate(separators):
        if separator == "":
            return _character_windows(source, span, size)
        if source.find(separator, start, end) == -1:
            continue
        rest = separators[level + 1 :] or ("",)
        merged: list[Span] = []
        # Stripped bounds of each merged span (None: whitespace only).
        merged_bounds: list[tuple[int, int] | None] = []
        for piece in _cut(source, span, separator):
            bounds = _bounds(source, *piece)
            if bounds is not None and bounds[1] - bounds[0] > size:
                for part in recursive_spans(source, piece, size, rest):
                    merged.append(part)
                    merged_bounds.append(_bounds(source, *part))
                continue
            if merged:
                previous = merged_bounds[-1]
                if previous is None or bounds is None:
                    combined = previous or bounds
                else:
                    combined = (previous[0], bounds[1])
                if combined is None or combined[1] - combined[0] <= size:
                    merged[-1] = (merged[-1][0], piece[1])
                    merged_bounds[-1] = combined
                    continue
            merged.append(piece)
            merged_bounds.append(bounds)
        return merged
    return [span]  # unreachable: separators end with ""


def _word_start_within(source: str, start: int, overlap: int, floor: int) -> int:
    """Return the earliest word start in ``[start - overlap, start)``, else ``start``.

    The context never begins before ``floor`` (the previous chunk's start),
    nor in the middle of a word.
    """
    low = max(floor, start - overlap)
    for position in range(low, start):
        if not source[position].isspace() and (
            position == 0 or source[position - 1].isspace()
        ):
            return position
    return start


def _strip_or_keep(source: str, span: Span, strip: bool) -> Span:
    return strip_span(source, *span) if strip else span


class _BatchMixin:
    """``chunk_batch`` shared by both chunkers (``ChunkerProtocol``)."""

    def chunk_batch(
        self,
        texts: list[str],
        doc_ids: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Chunk a list of documents.

        Parameters
        ----------
        texts : list of str
            Input documents.
        doc_ids : list of str or None, default=None
            Parallel document identifiers.
        extra_metadata : dict or None, default=None
            Shared metadata for every result.

        Returns
        -------
        list of ChunkResult
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
            self.chunk(  # type: ignore[attr-defined]
                text,
                doc_id=doc_ids[i] if doc_ids else None,
                extra_metadata=extra_metadata,
            )
            for i, text in enumerate(texts)
        ]


@dataclass(frozen=True)
class RecursiveChunkerConfig(ChunkerConfig):
    r"""Configuration for :class:`RecursiveChunker`.

    Parameters
    ----------
    chunk_size : int, default=1000
        Largest chunk anchor, in characters, not counting the whitespace
        around it.
    chunk_overlap : int, default=0
        Characters of the preceding text added as context (from a word
        start, never before the previous chunk's start); ``0`` adds none.
        Must be smaller than ``chunk_size``.
    separators : tuple of str, default=("\n\n", "\n", ". ", " ", "")
        Separators, coarsest first; must end with ``""`` (characters). A
        list is accepted and stored as a tuple. The defaults assume ``"\n"``
        line ends (readers normalise ``"\r\n"``); add ``"\r\n\r\n"`` for
        text that keeps them. A hard-wrapped text is cut at line ends before
        sentence ends.
    strip_whitespace : bool, default=True
        Narrow each chunk to exclude surrounding whitespace.
    min_length : int, default=1
        Drop a chunk with fewer characters than this, not counting the
        whitespace around it (so a whitespace-only chunk is always dropped).
    """

    chunk_size: int = 1000
    chunk_overlap: int = 0
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    strip_whitespace: bool = True
    min_length: int = 1

    def __post_init__(self) -> None:
        """Validate the fields."""
        super().__post_init__()
        _coerce_separators(self)
        _positive(self.chunk_size, "RecursiveChunkerConfig.chunk_size")
        if (
            isinstance(self.chunk_overlap, bool)
            or not isinstance(self.chunk_overlap, int)
            or not 0 <= self.chunk_overlap < self.chunk_size
        ):
            raise ValueError(
                "RecursiveChunkerConfig.chunk_overlap must be an int in"
                f" [0, chunk_size); got {self.chunk_overlap!r}."
            )
        _validate_separators(self.separators, "RecursiveChunkerConfig")
        _positive(self.min_length, "RecursiveChunkerConfig.min_length")


def _text_or_raise(text: Any, owner: str) -> None:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__!r}.")
    _validate_text_input(text, f"{owner}.chunk")
    if not text.strip():
        raise ValueError("text must not be empty or whitespace-only.")


class RecursiveChunker(_BatchMixin):
    r"""Split text by a separator hierarchy until each piece fits a budget.

    Parameters
    ----------
    config : RecursiveChunkerConfig or None, default=None
        ``None`` is ``RecursiveChunkerConfig()``.

    See Also
    --------
    ParentChildChunker : The same rule at two sizes, linked.

    Notes
    -----
    **User.** A paragraph that fits stays whole; a longer one is cut at line
    ends, then sentence ends, then spaces, then characters, only as far as
    needed. Each chunk's ``start_char``/``end_char`` is the piece it stands
    for; with ``chunk_overlap`` its text also holds the end of the previous
    chunk (``context_span``).

    Examples
    --------
    >>> text = "Alpha one. Beta two.\n\nGamma three."
    >>> chunks = (
    ...     RecursiveChunker(RecursiveChunkerConfig(chunk_size=12)).chunk(text).chunks
    ... )
    >>> [c.text for c in chunks]
    ['Alpha one.', 'Beta two.', 'Gamma three.']
    >>> all(c.text == text[slice(*c.context_span)] for c in chunks)
    True
    """

    config_class = RecursiveChunkerConfig
    text_is_source_slice = True
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.RECURSIVE

    def __init__(self, config: RecursiveChunkerConfig | None = None) -> None:
        if config is not None and not isinstance(config, RecursiveChunkerConfig):
            raise TypeError(
                "RecursiveChunker: config must be a RecursiveChunkerConfig or"
                f" None, got {type(config).__name__}."
            )
        self._cfg = config if config is not None else RecursiveChunkerConfig()

    @property
    def config(self) -> RecursiveChunkerConfig:
        """The configuration (read-only)."""
        return self._cfg

    def spans(self, text: str) -> list[Span]:
        """Return the anchor span of every chunk ``chunk`` would produce.

        Parameters
        ----------
        text : str
            The source.

        Returns
        -------
        list of (int, int)
            Anchor spans in source order.

        Raises
        ------
        TypeError, ValueError
            As :meth:`chunk`.
        """
        _text_or_raise(text, "RecursiveChunker")
        cfg = self._cfg
        spans = recursive_spans(text, (0, len(text)), cfg.chunk_size, cfg.separators)
        out = []
        for span in spans:
            start, end = _strip_or_keep(text, span, cfg.strip_whitespace)
            bounds = _bounds(text, start, end)
            # min_length counts characters without surrounding whitespace, so
            # a whitespace-only span is never a chunk.
            if bounds is not None and bounds[1] - bounds[0] >= cfg.min_length:
                out.append((start, end))
        return out

    def chunk(
        self,
        text: str,
        doc_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Split *text* into chunks of at most ``chunk_size`` characters.

        Parameters
        ----------
        text : str
            Raw document text.
        doc_id : str or None, default=None
            Stored in each chunk's metadata.
        extra_metadata : dict or None, default=None
            Merged into the result metadata.

        Returns
        -------
        ChunkResult
            The chunks; ``metadata["span_basis"]`` is ``"input"``.

        Raises
        ------
        TypeError
            If *text* is not a ``str``.
        ValueError
            If *text* is empty or whitespace-only, or holds a NUL character
            or a lone surrogate.
        """
        _text_or_raise(text, "RecursiveChunker")
        cfg = self._cfg
        chunks: list[Chunk] = []
        previous_start = 0
        for index, (start, end) in enumerate(self.spans(text)):
            context = (
                _word_start_within(text, start, cfg.chunk_overlap, previous_start)
                if cfg.chunk_overlap and index
                else start
            )
            meta: dict[str, Any] = {"chunk_index": index}
            if doc_id is not None:
                meta["doc_id"] = doc_id
            chunks.append(
                Chunk(
                    text=text[context:end],
                    start_char=start,
                    end_char=end,
                    metadata=meta,
                    context_start=context if context != start else None,
                )
            )
            previous_start = start
        result_meta: dict[str, Any] = {
            "span_basis": "input",
            "chunker": "recursive",
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "total_chunks": len(chunks),
        }
        if extra_metadata:
            result_meta.update(extra_metadata)
        return ChunkResult(chunks=chunks, metadata=result_meta)


@dataclass(frozen=True)
class ParentChildChunkerConfig(ChunkerConfig):
    r"""Configuration for :class:`ParentChildChunker`.

    Parameters
    ----------
    parent_size : int, default=2000
        Largest parent, in characters.
    child_size : int, default=400
        Largest child, in characters; smaller than ``parent_size``.
    child_overlap : int, default=0
        Context characters of a child, as ``chunk_overlap`` of
        :class:`RecursiveChunkerConfig`, within its parent.
    separators : tuple of str, default=("\n\n", "\n", ". ", " ", "")
        Separators for both levels, coarsest first, ending with ``""``.
    emit_parents : bool, default=True
        Emit each parent before its children. ``False`` emits children only
        (each still records its parent's span in ``parent_span``).
    """

    parent_size: int = 2000
    child_size: int = 400
    child_overlap: int = 0
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    emit_parents: bool = True

    def __post_init__(self) -> None:
        """Validate the fields."""
        super().__post_init__()
        _coerce_separators(self)
        _positive(self.parent_size, "ParentChildChunkerConfig.parent_size")
        _positive(self.child_size, "ParentChildChunkerConfig.child_size")
        if self.child_size >= self.parent_size:
            raise ValueError(
                "ParentChildChunkerConfig.child_size must be smaller than"
                f" parent_size; got {self.child_size} >= {self.parent_size}."
            )
        if (
            isinstance(self.child_overlap, bool)
            or not isinstance(self.child_overlap, int)
            or not 0 <= self.child_overlap < self.child_size
        ):
            raise ValueError(
                "ParentChildChunkerConfig.child_overlap must be an int in"
                f" [0, child_size); got {self.child_overlap!r}."
            )
        _validate_separators(self.separators, "ParentChildChunkerConfig")
        if not isinstance(self.emit_parents, bool):
            raise TypeError("ParentChildChunkerConfig.emit_parents must be a bool.")


class ParentChildChunker(_BatchMixin):
    r"""Large parents for context, small children for retrieval, linked.

    Parameters
    ----------
    config : ParentChildChunkerConfig or None, default=None
        ``None`` is ``ParentChildChunkerConfig()``.

    See Also
    --------
    RecursiveChunker : The splitting rule used at both levels.
    scikitplot.corpus.children : Traverse the documents it produces.

    Notes
    -----
    **User.** Each parent is followed by its children. Every chunk's
    metadata holds ``chunk_role`` (``"parent"`` or ``"child"``) and
    ``parent_span``; a child holds ``parent_chunk``, the position of its
    parent in the result. A document built from a child gets
    ``parent_doc_id`` (its parent's ``doc_id``): retrieve children, then
    give the model the parent. Documents keep ``chunk_role`` and
    ``parent_span`` in their metadata, so with ``emit_parents=False`` a child
    still locates its parent's region. Both levels report the ``RECURSIVE``
    strategy; ``chunk_role`` tells them apart. A parent no longer than
    ``child_size`` has one child with the same text: index children only.

    Examples
    --------
    >>> text = "Alpha one. Beta two.\n\nGamma three. Delta four."
    >>> config = ParentChildChunkerConfig(parent_size=25, child_size=12)
    >>> result = ParentChildChunker(config).chunk(text)
    >>> [
    ...     (c.metadata["chunk_role"], c.text) for c in result.chunks
    ... ]  # doctest: +NORMALIZE_WHITESPACE
    [('parent', 'Alpha one. Beta two.'), ('child', 'Alpha one.'), ('child', 'Beta two.'),
     ('parent', 'Gamma three. Delta four.'), ('child', 'Gamma three.'), ('child', 'Delta four.')]
    """

    config_class = ParentChildChunkerConfig
    text_is_source_slice = True
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.RECURSIVE

    def __init__(self, config: ParentChildChunkerConfig | None = None) -> None:
        if config is not None and not isinstance(config, ParentChildChunkerConfig):
            raise TypeError(
                "ParentChildChunker: config must be a ParentChildChunkerConfig or"
                f" None, got {type(config).__name__}."
            )
        self._cfg = config if config is not None else ParentChildChunkerConfig()

    @property
    def config(self) -> ParentChildChunkerConfig:
        """The configuration (read-only)."""
        return self._cfg

    def chunk(
        self,
        text: str,
        doc_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Split *text* into parents and the children inside each.

        Parameters
        ----------
        text : str
            Raw document text.
        doc_id : str or None, default=None
            Stored in each chunk's metadata.
        extra_metadata : dict or None, default=None
            Merged into the result metadata.

        Returns
        -------
        ChunkResult
            Parents and children in source order (each parent first).

        Raises
        ------
        TypeError
            If *text* is not a ``str``.
        ValueError
            If *text* is empty or whitespace-only, or holds a NUL character
            or a lone surrogate.
        """
        _text_or_raise(text, "ParentChildChunker")
        cfg = self._cfg
        chunks: list[Chunk] = []
        n_parents = n_children = 0
        parents = recursive_spans(text, (0, len(text)), cfg.parent_size, cfg.separators)
        for parent in parents:
            p_start, p_end = strip_span(text, *parent)
            if p_end == p_start:
                continue
            base: dict[str, Any] = {"parent_span": [p_start, p_end]}
            if doc_id is not None:
                base["doc_id"] = doc_id
            parent_position: int | None = None
            if cfg.emit_parents:
                parent_position = len(chunks)
                chunks.append(
                    Chunk(
                        text=text[p_start:p_end],
                        start_char=p_start,
                        end_char=p_end,
                        metadata={
                            **base,
                            "chunk_role": "parent",
                            "chunk_index": parent_position,
                        },
                    )
                )
                n_parents += 1
            previous_start = p_start
            children = recursive_spans(
                text, (p_start, p_end), cfg.child_size, cfg.separators
            )
            first = True
            for child in children:
                c_start, c_end = strip_span(text, *child)
                if c_end == c_start:
                    continue
                context = (
                    _word_start_within(text, c_start, cfg.child_overlap, previous_start)
                    if cfg.child_overlap and not first
                    else c_start
                )
                meta = {**base, "chunk_role": "child", "chunk_index": len(chunks)}
                if parent_position is not None:
                    meta["parent_chunk"] = parent_position
                chunks.append(
                    Chunk(
                        text=text[context:c_end],
                        start_char=c_start,
                        end_char=c_end,
                        metadata=meta,
                        context_start=context if context != c_start else None,
                    )
                )
                n_children += 1
                previous_start = c_start
                first = False
        result_meta: dict[str, Any] = {
            "span_basis": "input",
            "chunker": "parent_child",
            "parent_size": cfg.parent_size,
            "child_size": cfg.child_size,
            "parents": n_parents,
            "children": n_children,
            "total_chunks": len(chunks),
        }
        if extra_metadata:
            result_meta.update(extra_metadata)
        return ChunkResult(chunks=chunks, metadata=result_meta)
