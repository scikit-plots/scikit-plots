# scikitplot/corpus/_chunkers/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._chunkers
===========================
Text segmentation strategies for the corpus pipeline.

All five chunkers implement :class:`~scikitplot.corpus._base.ChunkerBase`
and return :class:`~._types.ChunkResult` objects containing
:class:`~._types.Chunk` instances with full multilang metadata.

**Chunkers (all multilang-aware via MultilangMixin):**

:class:`WordChunker`
    Word-level tokenisation with stemming, lemmatisation, stopword removal.
    Supports 200+ languages via NLTK, spaCy, and custom tokenisers.

:class:`SentenceChunker`
    Sentence-boundary segmentation via REGEX, NLTK, spaCy, or custom
    backends.  Multi-script mode auto-detects script and uses the correct
    terminal character set.

:class:`ParagraphChunker`
    Blank-line paragraph splitting.  Pure Python.  Multilang-enriched per
    paragraph.

:class:`FixedWindowChunker`
    Sliding-window chunking by chars or tokens.  CJK-aware token splitting.

:class:`SemanticChunker`
    Layer 3 semantic chunker.  MORPHOLOGICAL / EMBEDDING / HYBRID backends.
    Full multilang pipeline: Layer 0 → Layer 1 → Layer 2 → Layer 3.

**Layer infrastructure:**

:class:`GraphemeClusterNormalizer`   — Layer 0 (in _normalizers/)
:class:`ScriptSegmenter`             — Layer 1 (in _custom_tokenizer.py)
:class:`WritingSystemAdapter`        — Layer 2 (in _writing_system.py)
:class:`SemanticChunker`             — Layer 3

**Multilang config and types:**

:class:`MultilangConfig`      — Feature flags (all 5 chunkers)
:class:`MultilangMixin`       — Shared mixin (all 5 chunkers)
:class:`SemantemeInfo`        — Per-semanteme analysis record
:class:`PreprocessingStep`    — Single preprocessing transformation record
:class:`PreprocessingTrace`   — Full preprocessing audit trail
:class:`MultilangChunkMeta`   — Per-chunk multilang bundle

Examples
--------
>>> from scikitplot.corpus._chunkers import (
...     SemanticChunker,
...     SemanticChunkerConfig,
...     SemanticBackend,
...     MultilangConfig,
... )
>>> cfg = SemanticChunkerConfig(backend=SemanticBackend.MORPHOLOGICAL)
>>> ml = MultilangConfig(include_raw_text=True, include_semantemes=True)
>>> chunker = SemanticChunker(cfg)
>>> result = chunker.chunk("Hello world. مرحبا بالعالم。")
>>> result.chunks[0].metadata["multilang"]["script"]
'latin'
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _chunker_bridge,
    _custom_tokenizer,
    _fixed_window,
    _language_data,
    _multilang_mixin,
    _paragraph,
    _semantic,
    _sentence,
    _word,
    _writing_system,
)
from ._chunker_bridge import *  # noqa: F403
from ._custom_tokenizer import *  # noqa: F403
from ._fixed_window import *  # noqa: F403
from ._language_data import *  # noqa: F403
from ._multilang_mixin import *  # noqa: F403
from ._paragraph import *  # noqa: F403
from ._semantic import *  # noqa: F403
from ._sentence import *  # noqa: F403
from ._word import *  # noqa: F403
from ._writing_system import *  # noqa: F403

__all__ = []
__all__ += _chunker_bridge.__all__
__all__ += _custom_tokenizer.__all__
__all__ += _fixed_window.__all__
__all__ += _language_data.__all__
__all__ += _multilang_mixin.__all__
__all__ += _paragraph.__all__
__all__ += _semantic.__all__
__all__ += _sentence.__all__
__all__ += _word.__all__
__all__ += _writing_system.__all__
