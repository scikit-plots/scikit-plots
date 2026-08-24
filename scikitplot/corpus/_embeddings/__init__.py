"""
scikitplot.corpus._embeddings
==============================
Embedding engines and deterministic local hashing baselines.

``EmbeddingEngine`` provides model/API/custom backends with file-based caching.
``HashEmbedder`` provides a dependency-light deterministic lexical baseline for
offline tests, examples, and local retrieval paths without model downloads.

Python compatibility
--------------------
Python 3.8-3.15. ``numpy`` is required. ``sentence_transformers``,
``openai``, and ``tiktoken`` are optional; graceful ``ImportError`` at
call time when not installed.
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _embedding,
    _hashing,
    _multimodal_embedding,
)
from ._embedding import *  # noqa: F403
from ._hashing import *  # noqa: F403
from ._multimodal_embedding import *  # noqa: F403

__all__ = []
__all__ += _embedding.__all__
__all__ += _hashing.__all__
__all__ += _multimodal_embedding.__all__
