# scikitplot/corpus/_chunkers/_chunker_bridge.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Adapter bridge: new standalone chunkers → ``ChunkerBase`` contract.

The new chunkers (``SentenceChunker``, ``ParagraphChunker``,
``FixedWindowChunker``, ``WordChunker``) are standalone classes that
return ``ChunkResult`` objects and do **not** extend ``ChunkerBase``.
The existing pipeline (``_base.py:get_documents``, ``_pipeline.py``)
calls ``chunker.chunk(text, metadata=dict) → list[tuple[int, str]]``
and reads ``chunker.strategy``.

This module provides thin adapter wrappers that bridge the two
interfaces without modifying either the new chunkers or the
pipeline internals.

.. admonition:: Why an adapter instead of modifying the chunkers?

   The new chunkers have a richer API (``chunk_batch``, per-chunk
   metadata, ``ChunkResult`` with offsets) that would be lost by
   collapsing them into ``ChunkerBase``.  The adapter preserves
   both interfaces and lets the pipeline work immediately.

Supports Python 3.8 through 3.15.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, ClassVar

from .._schema import ChunkingStrategy
from .._types import ChunkResult  # CRITICAL-02: bridge.chunk() returns ChunkResult

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkedTextList",
    "ChunkerBridge",
    "FixedWindowChunkerBridge",
    "ParagraphChunkerBridge",
    "SemanticChunkerBridge",
    "SentenceChunkerBridge",
    "WordChunkerBridge",
    "bridge_chunker",
    "register_bridge",
    "unregister_bridge",
]


# ---------------------------------------------------------------------------
# ChunkedTextList — backward-compatible list that carries chunk metadata
# ---------------------------------------------------------------------------


class ChunkedTextList(list):
    """A ``list`` subclass that carries per-chunk metadata alongside ``(int, str)`` pairs.

    Returned by every :class:`ChunkerBridge` subclass in place of a plain
    ``list``.  Behaves identically to ``list`` for all existing callers
    that unpack ``(char_start, chunk_text)`` pairs.

    The extra ``chunk_metadata_list`` attribute lets :meth:`_base.DocumentReader.
    get_documents` read multilang metadata and populate :class:`~._schema.
    CorpusDocument` fields without breaking the existing ``(int, str)``
    contract.

    Parameters
    ----------
    pairs : list[tuple[int, str]]
        The ``(char_start, chunk_text)`` tuples (standard contract).
    chunk_metadata_list : list[dict]
        One metadata dict per pair.  Must have the same length as *pairs*.
        Each dict is ``chunk.metadata`` from the inner :class:`~._types.Chunk`.

    Notes
    -----
    **Developer note:** Callers that only do ``for start, text in sub_chunks``
    are unaffected — they never see ``chunk_metadata_list``.  Only
    :meth:`_base.DocumentReader.get_documents` inspects this attribute.

    Idempotency: ``ChunkedTextList`` is constructed once per ``chunk()``
    call and never mutated after construction.

    Examples
    --------
    >>> cl = ChunkedTextList([(0, "Hello")], [{"multilang": {"script": "latin"}}])
    >>> for start, text in cl:
    ...     print(start, text)
    0 Hello
    >>> cl.chunk_metadata_list[0]["multilang"]["script"]
    'latin'
    """

    def __init__(
        self,
        pairs: list[tuple[int, str]],
        chunk_metadata_list: list[dict[str, Any]],
    ) -> None:
        super().__init__(pairs)
        if len(chunk_metadata_list) != len(pairs):
            raise ValueError(
                f"ChunkedTextList: pairs ({len(pairs)}) and "
                f"chunk_metadata_list ({len(chunk_metadata_list)}) must have "
                f"the same length."
            )
        self.chunk_metadata_list: list[dict[str, Any]] = chunk_metadata_list


# ---------------------------------------------------------------------------
# Abstract bridge
# ---------------------------------------------------------------------------


class ChunkerBridge(abc.ABC):
    """
    Adapter that wraps a new-style chunker as a ``ChunkerBase``-
    compatible object.

    Parameters
    ----------
    inner : object
        The new-style chunker instance (``SentenceChunker``,
        ``ParagraphChunker``, ``FixedWindowChunker``, or
        ``WordChunker``).

    Attributes
    ----------
    strategy : ChunkingStrategy
        Required by ``_base.py:get_documents()`` line 739.
    inner : object
        The wrapped chunker — retained for direct access to the
        richer ``ChunkResult`` API when needed.

    Notes
    -----
    **Developer note:** ``_base.py`` calls exactly two things on a
    chunker:

    1. ``self.chunker.strategy`` — a ``ChunkingStrategy`` enum value.
    2. ``self.chunker.chunk(text, metadata=raw_chunk)``
       → ``list[tuple[int, str]]`` where ``int`` is ``char_start``
       and ``str`` is the chunk text.

    This bridge satisfies both without touching ``ChunkerBase`` or
    the new chunkers.
    """  # noqa: D205

    strategy: ClassVar[ChunkingStrategy]

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Chunk *text* and return a :class:`~.._types.ChunkResult`.

        **CRITICAL-02 (Phase 2):** Returns ``ChunkResult`` directly.
        :meth:`DocumentReader.get_documents` now iterates
        ``chunk_result.chunks`` instead of ``(char_start, chunk_text)``
        tuples.

        Parameters
        ----------
        text : str
            Raw text to chunk.
        metadata : dict[str, Any] or None, optional
            Raw-chunk metadata dict passed by ``get_documents()``.
            Forwarded as ``extra_metadata`` to the inner chunker.

        Returns
        -------
        ChunkResult
            Ordered list of :class:`~.._types.Chunk` objects with
            ``text``, ``start_char``, ``end_char``, and ``metadata``.

        Notes
        -----
        Use :meth:`_to_tuples` to convert to the legacy
        ``list[tuple[int, str]]`` format if needed for backward compat.
        """
        return self._call_inner(text, metadata)

    @abc.abstractmethod
    def _call_inner(
        self,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        """Call the inner chunker and return its native result.

        Subclasses override this to match the inner chunker's
        actual call signature.
        """

    # ------------------------------------------------------------------
    # Shared conversion logic
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tuples(
        source_text: str,
        chunk_result: Any,
    ) -> ChunkedTextList:
        """Convert a ``ChunkResult`` to :class:`ChunkedTextList`.

        .. deprecated:: 0.5.0
            Internal pipeline use has been removed (CRITICAL-02).
            :meth:`ChunkerBridge.chunk` now returns ``ChunkResult`` directly.
            This method is retained as a **backward-compat utility** for user
            code that calls it directly.  It will be removed in 0.7.0.

        Uses chunk offsets if available, otherwise falls back to a
        forward-cursor ``str.find`` scan (O(n) total, not O(n²)).

        Parameters
        ----------
        source_text : str
            Original raw text passed to the chunker.
        chunk_result : ChunkResult
            Output from the inner chunker.

        Returns
        -------
        ChunkedTextList
            Pairs of ``(char_start, chunk_text)`` with metadata attached.
        """
        pairs: list[tuple[int, str]] = []
        metas: list[dict[str, Any]] = []
        cursor = 0

        chunks = getattr(chunk_result, "chunks", None)
        if chunks is None:
            # Fallback: treat as single chunk — no metadata
            return ChunkedTextList([(0, source_text)], [{}])

        for ch in chunks:
            ch_text: str = ch.text
            ch_start = getattr(ch, "start_char", None)
            if ch_start is None:
                ch_start = getattr(ch, "char_start", None)

            if ch_start is not None:
                pairs.append((ch_start, ch_text))
                cursor = ch_start + len(ch_text)
            else:
                idx = source_text.find(ch_text, cursor)
                if idx == -1:
                    idx = cursor
                pairs.append((idx, ch_text))
                cursor = idx + len(ch_text)

            # Carry chunk-level metadata (includes "multilang" when enabled)
            ch_meta = getattr(ch, "metadata", None)
            metas.append(dict(ch_meta) if ch_meta else {})

        return ChunkedTextList(pairs, metas)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"strategy={self.strategy!r}, "
            f"inner={type(self.inner).__name__})"
        )


# ---------------------------------------------------------------------------
# Concrete bridges
# ---------------------------------------------------------------------------


class SentenceChunkerBridge(ChunkerBridge):
    """Bridge for ``SentenceChunker`` → ``ChunkerBase`` contract."""

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.SENTENCE

    def _call_inner(
        self,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        return self.inner.chunk(text, extra_metadata=metadata)


class ParagraphChunkerBridge(ChunkerBridge):
    """Bridge for ``ParagraphChunker`` → ``ChunkerBase`` contract."""

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.PARAGRAPH

    def _call_inner(
        self,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        return self.inner.chunk(text, extra_metadata=metadata)


class FixedWindowChunkerBridge(ChunkerBridge):
    """Bridge for ``FixedWindowChunker`` → ``ChunkerBase`` contract."""

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.FIXED_WINDOW

    def _call_inner(
        self,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        return self.inner.chunk(text, extra_metadata=metadata)


class WordChunkerBridge(ChunkerBridge):
    """Bridge for ``WordChunker`` → ``ChunkerBase`` contract.

    Notes
    -----
    ``WordChunker`` splits text at the word-token level, which does not
    correspond to any named :class:`~scikitplot.corpus._schema.ChunkingStrategy`
    value.  ``CUSTOM`` is used as the closest approximation — it signals
    that user-supplied or non-standard logic was applied, and downstream
    consumers should not assume standard segment boundaries.
    """

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.CUSTOM

    def _call_inner(
        self,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        return self.inner.chunk(text, extra_metadata=metadata)


class SemanticChunkerBridge(ChunkerBridge):
    """Bridge for ``SemanticChunker`` → ``ChunkerBase`` contract.

    The :class:`~._semantic.SemanticChunker` (Layer 3) is a new-style
    chunker that returns :class:`~._types.ChunkResult` objects.  This
    bridge adapts it to the ``ChunkerBase`` interface so it can be
    passed directly to :class:`~._pipeline.CorpusPipeline` and
    :class:`~._base.DocumentReader` without modification.

    ``SEMANTIC`` is used as the strategy value.  If
    :class:`~._schema.ChunkingStrategy` does not yet define a
    ``SEMANTIC`` member, ``CUSTOM`` is used as a fallback so the
    bridge never raises an :class:`AttributeError`.

    Notes
    -----
    **User note:** Pass ``SemanticChunker(...)`` directly to
    :class:`~._pipeline.CorpusPipeline` — the bridge is applied
    automatically via :func:`bridge_chunker`.

    **Developer note:** Multilang metadata in ``chunk.metadata["multilang"]``
    flows through :class:`ChunkedTextList` to
    :meth:`_base.DocumentReader.get_documents` which maps it to
    :class:`~._schema.CorpusDocument` fields.
    """

    strategy: ClassVar[ChunkingStrategy] = getattr(
        ChunkingStrategy, "SEMANTIC", ChunkingStrategy.CUSTOM
    )

    def _call_inner(
        self,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        return self.inner.chunk(text, extra_metadata=metadata)


# ---------------------------------------------------------------------------
# Auto-detect factory
# ---------------------------------------------------------------------------

_BRIDGE_MAP: dict[str, type[ChunkerBridge]] = {
    "SentenceChunker": SentenceChunkerBridge,
    "ParagraphChunker": ParagraphChunkerBridge,
    "FixedWindowChunker": FixedWindowChunkerBridge,
    "WordChunker": WordChunkerBridge,
    "SemanticChunker": SemanticChunkerBridge,
}


def register_bridge(chunker_class: type, bridge_class: type[ChunkerBridge]) -> None:
    """Register a custom bridge for a user-defined chunker class.

    After registration, :func:`bridge_chunker` will automatically wrap
    instances of *chunker_class* in *bridge_class*.

    Parameters
    ----------
    chunker_class : type
        The user-defined chunker class to register.  Matched by exact
        ``type(chunker).__name__`` string so subclasses must be registered
        separately if needed.
    bridge_class : type[ChunkerBridge]
        A :class:`ChunkerBridge` subclass that wraps *chunker_class*.

    Raises
    ------
    TypeError
        If *bridge_class* is not a subclass of :class:`ChunkerBridge`.

    Examples
    --------
    >>> class MyChunker:
    ...     def chunk(self, text, extra_metadata=None):
    ...         from .._types import Chunk, ChunkResult
    ...
    ...         return ChunkResult(
    ...             chunks=[
    ...                 Chunk(text=text, start_char=0, end_char=len(text), metadata={})
    ...             ],
    ...             metadata={},
    ...         )
    >>> class MyChunkerBridge(ChunkerBridge):
    ...     from .._schema import ChunkingStrategy
    ...
    ...     strategy = ChunkingStrategy.CUSTOM
    ...
    ...     def _call_inner(self, text, metadata):
    ...         return self.inner.chunk(text, extra_metadata=metadata)
    >>> register_bridge(MyChunker, MyChunkerBridge)
    >>> bridged = bridge_chunker(MyChunker())
    >>> hasattr(bridged, "strategy")
    True
    """
    if not (isinstance(bridge_class, type) and issubclass(bridge_class, ChunkerBridge)):
        raise TypeError(
            f"register_bridge: bridge_class must be a ChunkerBridge subclass, "
            f"got {bridge_class!r}."
        )
    key = chunker_class.__name__
    if key in _BRIDGE_MAP:
        logger.debug("register_bridge: overwriting existing bridge for %r.", key)
    _BRIDGE_MAP[key] = bridge_class
    logger.debug("register_bridge: registered %r → %s.", key, bridge_class.__name__)


def unregister_bridge(chunker_class: type) -> None:
    """Remove a previously registered bridge for *chunker_class*.

    Parameters
    ----------
    chunker_class : type
        The chunker class whose bridge should be removed.

    Raises
    ------
    KeyError
        If no bridge is registered for *chunker_class*.
    """
    key = chunker_class.__name__
    if key not in _BRIDGE_MAP:
        raise KeyError(f"unregister_bridge: no bridge registered for {key!r}.")
    del _BRIDGE_MAP[key]
    logger.debug("unregister_bridge: removed bridge for %r.", key)


def bridge_chunker(chunker: Any) -> ChunkerBridge | Any:
    """Wrap *chunker* in a bridge if it is a new-style chunker.

    Parameters
    ----------
    chunker : object
        Either a ``ChunkerBase`` subclass (returned as-is) or a
        new-style chunker (wrapped in the appropriate bridge).

    Returns
    -------
    ChunkerBridge or object
        The bridged chunker, or the original if no bridge is needed.

    Examples
    --------
    >>> from scikitplot.corpus._chunkers import SentenceChunker
    >>> bridged = bridge_chunker(SentenceChunker())
    >>> hasattr(bridged, "strategy")
    True
    >>> isinstance(bridged.chunk("Hello world. Goodbye.", metadata={}), list)
    True
    """
    # Already has the ChunkerBase interface
    if hasattr(chunker, "strategy") and callable(getattr(chunker, "chunk", None)):
        sig_ok = True
        try:
            import inspect  # noqa: PLC0415

            sig = inspect.signature(chunker.chunk)
            sig_ok = "metadata" in sig.parameters
        except (ValueError, TypeError):
            pass
        if sig_ok:
            return chunker

    cls_name = type(chunker).__name__
    bridge_cls = _BRIDGE_MAP.get(cls_name)
    if bridge_cls is not None:
        logger.debug("Auto-bridging %s → %s", cls_name, bridge_cls.__name__)
        return bridge_cls(chunker)

    # Unknown chunker — return as-is with a warning
    logger.warning(
        "Chunker %r does not match ChunkerBase contract and has no "
        "registered bridge. Pipeline may fail.",
        cls_name,
    )
    return chunker
