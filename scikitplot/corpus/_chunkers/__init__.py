r"""
scikitplot.corpus._chunkers
==========================
Text segmentation strategies for the corpus pipeline.

All chunkers implement :class:`~scikitplot.corpus._base.ChunkerBase` and
return ``list[tuple[int, str]]`` — a list of ``(char_start, chunk_text)``
pairs where ``char_start`` is the character offset of the chunk within the
input text block.

Available chunkers
------------------
:class:`SentenceChunker`
    Sentence-boundary segmentation via spaCy. Language-agnostic through
    configurable model names. Caches loaded models as instance attributes.
    Optional auto-download gate for CI/Docker safety.

:class:`ParagraphChunker`
    Splits on one or more consecutive blank lines (``\\n\\n``). Pure Python,
    no external dependencies. Configurable minimum paragraph length.

:class:`FixedWindowChunker`
    Sliding window with configurable size and overlap. Operates on either
    whitespace-delimited word tokens or raw characters. No external
    dependencies.

Quick usage
-----------
>>> from scikitplot.corpus._chunkers import SentenceChunker, ParagraphChunker
>>> chunker = ParagraphChunker()
>>> chunks = chunker.chunk("Para one.\\n\\nPara two.")
>>> [(start, text) for start, text in chunks]
[(0, 'Para one.'), (11, 'Para two.')]
"""  # noqa: D205, D400

from __future__ import annotations

from scikitplot.corpus._chunkers._chunker_bridge import (
    ChunkerBridge,
    bridge_chunker,
)
from scikitplot.corpus._chunkers._fixed_window import FixedWindowChunker
from scikitplot.corpus._chunkers._paragraph import ParagraphChunker
from scikitplot.corpus._chunkers._sentence import SentenceChunker
from scikitplot.corpus._chunkers._word import WordChunker

__all__ = [  # noqa: RUF022
    "FixedWindowChunker",
    "ParagraphChunker",
    "SentenceChunker",
    "WordChunker",
    # Chunker bridge
    "ChunkerBridge",
    "bridge_chunker",
]
