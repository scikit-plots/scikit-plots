# scikitplot/corpus/_chunkers/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

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
    Splits on one or more consecutive blank lines (``\n\n``). Pure Python,
    no external dependencies. Configurable minimum paragraph length.

:class:`FixedWindowChunker`
    Sliding window with configurable size and overlap. Operates on either
    whitespace-delimited word tokens or raw characters. No external
    dependencies.

Quick usage
-----------
>>> from scikitplot.corpus._chunkers import SentenceChunker, ParagraphChunker
>>> chunker = ParagraphChunker()
>>> chunks = chunker.chunk("Para one.\n\nPara two.")
>>> [(start, text) for start, text in chunks]
[(0, 'Para one.'), (11, 'Para two.')]
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _chunker_bridge,
    _custom_tokenizer,
    _fixed_window,
    _language_data,
    _paragraph,
    _sentence,
    _word,
)
from ._chunker_bridge import *  # noqa: F403
from ._custom_tokenizer import *  # noqa: F403
from ._fixed_window import *  # noqa: F403
from ._language_data import *  # noqa: F403
from ._paragraph import *  # noqa: F403
from ._sentence import *  # noqa: F403
from ._word import *  # noqa: F403

__all__ = []
__all__ += _chunker_bridge.__all__
__all__ += _custom_tokenizer.__all__
__all__ += _fixed_window.__all__
__all__ += _language_data.__all__
__all__ += _paragraph.__all__
__all__ += _sentence.__all__
__all__ += _word.__all__
