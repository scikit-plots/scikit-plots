# scikitplot/mcp/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Composable documentation retrieval for Model Context Protocol servers.

The default import surface remains independent of the MCP SDK and optional
corpus/vector dependencies. Call :func:`create_server` only when the stable ``mcp>=2,<3``
server dependency is installed. No models, indexes, files, or network connections are
opened at import time.
"""

from __future__ import annotations

from ._core import (
    MAX_CHUNK_CHARS,
    MAX_QUERY_CHARS,
    MAX_RESULTS,
    DocsRetriever,
    RetrievedChunk,
    build_search_docs_result,
)
from ._corpus_annoy import CorpusAnnoyRetriever
from ._demo import DemoDocument, InMemoryBm25Retriever, builtin_demo_retriever
from ._hybrid import (
    DEFAULT_RRF_K,
    Bm25Retriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from ._version import __version__  # ruff: ignore[unused-import]

__all__ = [
    "DEFAULT_RRF_K",
    "MAX_CHUNK_CHARS",
    "MAX_QUERY_CHARS",
    "MAX_RESULTS",
    "Bm25Retriever",
    "CorpusAnnoyRetriever",
    "DemoDocument",
    "DocsRetriever",
    "HybridRetriever",
    "InMemoryBm25Retriever",
    "RetrievedChunk",
    "build_search_docs_result",
    "builtin_demo_retriever",
    "create_server",
    "reciprocal_rank_fusion",
]


# https://github.com/modelcontextprotocol/modelcontextprotocol
def create_server(*args, **kwargs):
    """Lazily import and create the MCP SDK v2 server."""
    from ._server import (  # ruff: ignore[import-outside-top-level]
        create_server as _create_server,
    )

    return _create_server(*args, **kwargs)
