"""
scikitplot.corpus._embeddings
==============================
Multi-backend sentence embedding engine with file-based caching.

Produces dense vector representations of text chunks. The embedding step
is optional in the pipeline (``embed=False`` skips it entirely) and is
triggered per-document or in batches. Results are cached to ``.npy``
files keyed by a SHA-256 hash of ``(model_name, source_path, mtime, n_texts)``
so that re-running the pipeline on an unchanged corpus is O(1).

Python compatibility
--------------------
Python 3.8-3.15. ``numpy`` is required. ``sentence_transformers``,
``openai``, and ``tiktoken`` are optional; graceful ``ImportError`` at
call time when not installed.
"""  # noqa: D205, D400

from __future__ import annotations

from ._embedding import *  # noqa: F403
from ._multimodal_embedding import (  # noqa: F401
    LLMTrainingExporter,
    MultimodalEmbeddingEngine,
)
