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

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkerBridge",
    "FixedWindowChunkerBridge",
    "ParagraphChunkerBridge",
    "SentenceChunkerBridge",
    "WordChunkerBridge",
    "bridge_chunker",
    "register_bridge",
    "unregister_bridge",
]


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
    ) -> list[tuple[int, str]]:
        """Chunk *text* and return ``(char_start, chunk_text)`` pairs.

        Parameters
        ----------
        text : str
            Raw text to chunk.
        metadata : dict[str, Any] or None, optional
            Raw-chunk metadata dict passed by ``get_documents()``.
            Forwarded as ``extra_metadata`` to the inner chunker
            where supported.

        Returns
        -------
        list[tuple[int, str]]
            Each element is ``(char_offset, chunk_text)``.
            If the inner chunker does not provide offsets, a
            forward-cursor scan computes them.
        """
        result = self._call_inner(text, metadata)
        return self._to_tuples(text, result)

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
    ) -> list[tuple[int, str]]:
        """Convert a ``ChunkResult`` to ``list[tuple[int, str]]``.

        Uses chunk offsets if available, otherwise falls back to a
        forward-cursor ``str.find`` scan (O(n) total, not O(n²)).
        """
        pairs: list[tuple[int, str]] = []
        cursor = 0

        # ChunkResult has .chunks → list[Chunk]
        # Chunk has .text, .char_start (int | None), .char_end
        chunks = getattr(chunk_result, "chunks", None)
        if chunks is None:
            # Fallback: treat as single chunk
            return [(0, source_text)]

        for ch in chunks:
            ch_text: str = ch.text
            # Chunk (from _types.py) exposes start_char / end_char.
            # A legacy guard also checks the old alias char_start so that
            # third-party or schema-derived objects still work.
            ch_start = getattr(ch, "start_char", None)
            if ch_start is None:
                ch_start = getattr(ch, "char_start", None)

            if ch_start is not None:
                pairs.append((ch_start, ch_text))
                cursor = ch_start + len(ch_text)
            else:
                # Forward-cursor scan — avoids O(n²)
                idx = source_text.find(ch_text, cursor)
                if idx == -1:
                    # Chunker may have normalised whitespace; use cursor
                    idx = cursor
                pairs.append((idx, ch_text))
                cursor = idx + len(ch_text)

        return pairs

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


# ---------------------------------------------------------------------------
# Auto-detect factory
# ---------------------------------------------------------------------------

_BRIDGE_MAP: dict[str, type[ChunkerBridge]] = {
    "SentenceChunker": SentenceChunkerBridge,
    "ParagraphChunker": ParagraphChunkerBridge,
    "FixedWindowChunker": FixedWindowChunkerBridge,
    "WordChunker": WordChunkerBridge,
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
