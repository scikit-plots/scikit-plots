# scikitplot/mcp/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
:mod:`scikitplot.mcp` — an MCP server exposing scikit-plots capabilities.

Flagship toolset: **documentation retrieval** ("search_docs"), which composes
:mod:`scikitplot.corpus` (ingest / chunk / embed / store) with
:mod:`scikitplot.annoy` (vector index) to answer queries with **source-cited**
passages — served to MCP clients (Claude, Cursor, Copilot) and to the
scikit-plots AI documentation panel's proxy from one retrieval core.

This ``__init__`` re-exports only the SDK-agnostic core so importing the package
never requires an MCP SDK, corpus, or annoy to be installed. The transport /
server layer (stdio + HTTP-SSE) is wired separately — see
``_maintenance/MCP_MODULE_DESIGN.md`` (gated).

Notes
-----
This module intentionally avoids side effects at import time (no implicit
network, model, or vector-index construction), consistent with
:mod:`scikitplot.annoy`.
"""

from __future__ import annotations

from ._core import (
    MAX_CHUNK_CHARS,
    MAX_RESULTS,
    DocsRetriever,
    RetrievedChunk,
    build_search_docs_result,
)
from ._corpus_annoy import CorpusAnnoyRetriever
from ._hybrid import (
    DEFAULT_RRF_K,
    Bm25Retriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)

__all__ = [
    "DEFAULT_RRF_K",
    "MAX_CHUNK_CHARS",
    "MAX_RESULTS",
    "Bm25Retriever",
    "CorpusAnnoyRetriever",
    "DocsRetriever",
    "HybridRetriever",
    "RetrievedChunk",
    "build_search_docs_result",
    "reciprocal_rank_fusion",
]

__version__ = "0.1.0.dev0"
