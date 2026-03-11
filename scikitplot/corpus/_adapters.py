# scikitplot/corpus/_adapters.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Adapter layer: ``CorpusDocument`` → downstream consumer formats.

Converts :class:`~scikitplot.corpus._schema.CorpusDocument` and
:class:`~scikitplot.corpus._similarity.SearchResult` objects into
standardised formats consumed by:

- **LangChain** — ``langchain_core.documents.Document``
- **LangGraph** — state-compatible dicts for graph nodes
- **MCP** (Model Context Protocol) — ``resources/read`` response
- **HuggingFace** — ``datasets.Dataset`` rows
- **Generic RAG** — ``(text, metadata, embedding)`` tuples
- **JSONL streaming** — newline-delimited JSON for any consumer

.. admonition:: Design philosophy

   Each adapter is a pure function with zero required dependencies.
   LangChain/LangGraph/MCP libraries are **lazy-imported** only
   when the adapter is called.  If the library is not installed,
   the adapter returns the equivalent plain-Python structure and
   logs a warning.

Supports Python 3.8 through 3.15.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

__all__ = [  # noqa: RUF022
    # Core converters
    "to_langchain_documents",
    "to_langgraph_state",
    "to_mcp_resources",
    "to_mcp_tool_result",
    "to_huggingface_dataset",
    "to_rag_tuples",
    "to_jsonl",
    # Retriever adapters
    "LangChainCorpusRetriever",
    "MCPCorpusServer",
]


# =====================================================================
# Helper: extract metadata dict from CorpusDocument
# =====================================================================


def _doc_metadata(doc: Any) -> dict[str, Any]:
    """Build a flat metadata dict from a ``CorpusDocument``.

    Includes all non-None first-class fields plus the ``metadata``
    dict.  Embedding is excluded (binary, not serialisable).
    """
    meta: dict[str, Any] = {}

    # Provenance
    for key in (
        "doc_id",
        "source_file",
        "chunk_index",
        "source_type",
        "source_title",
        "source_author",
        "source_date",
        "collection_id",
        "url",
        "doi",
        "isbn",
    ):
        val = getattr(doc, key, None)
        if val is not None:
            # Convert enums to string
            meta[key] = str(val) if hasattr(val, "value") else val

    # Position
    for key in (
        "page_number",
        "paragraph_index",
        "line_number",
        "char_start",
        "char_end",
        "parent_doc_id",
        "act",
        "scene_number",
    ):
        val = getattr(doc, key, None)
        if val is not None:
            meta[key] = val

    # Media
    for key in (
        "timecode_start",
        "timecode_end",
        "confidence",
        "ocr_engine",
    ):
        val = getattr(doc, key, None)
        if val is not None:
            meta[key] = val

    bbox = getattr(doc, "bbox", None)
    if bbox is not None:
        meta["bbox"] = list(bbox)

    # Section / chunking
    for key in ("section_type", "chunking_strategy", "language"):
        val = getattr(doc, key, None)
        if val is not None:
            meta[key] = str(val) if hasattr(val, "value") else val

    # Merge the open-ended metadata dict (lower priority)
    extra = getattr(doc, "metadata", None)
    if extra and isinstance(extra, dict):
        for k, v in extra.items():
            if k not in meta:
                meta[k] = v

    return meta


def _get_text(doc: Any) -> str:
    """Prefer ``normalized_text`` over ``text``."""
    nt = getattr(doc, "normalized_text", None)
    return nt or getattr(doc, "text", "")


# =====================================================================
# LangChain adapter
# =====================================================================


def to_langchain_documents(
    documents: Sequence[Any],
) -> list[Any]:
    """Convert ``CorpusDocument`` instances to LangChain ``Document``.

    Parameters
    ----------
    documents : Sequence[CorpusDocument]
        Source documents.

    Returns
    -------
    list[langchain_core.documents.Document] or list[dict]
        LangChain Document objects.  Falls back to equivalent dicts
        if ``langchain_core`` is not installed.

    Notes
    -----
    **User note:** These documents are directly usable with any
    LangChain retriever, chain, or agent::

        from langchain.chains import RetrievalQA

        lc_docs = to_langchain_documents(corpus_docs)
        # Feed to vector store, retriever, etc.

    Examples
    --------
    >>> lc_docs = to_langchain_documents(corpus_docs)
    >>> type(lc_docs[0]).__name__
    'Document'
    """
    try:
        from langchain_core.documents import (  # type: ignore[] # noqa: PLC0415
            Document as LCDoc,
        )
    except ImportError:
        logger.info(
            "langchain_core not installed; returning plain dicts "
            "with 'page_content' and 'metadata' keys."
        )
        LCDoc = None  # type: ignore[assignment]  # noqa: N806

    results = []
    for doc in documents:
        text = _get_text(doc)
        meta = _doc_metadata(doc)

        if LCDoc is not None:
            results.append(LCDoc(page_content=text, metadata=meta))
        else:
            results.append({"page_content": text, "metadata": meta})

    return results


# =====================================================================
# LangGraph adapter
# =====================================================================


def to_langgraph_state(
    documents: Sequence[Any],
    *,
    query: str = "",
    match_mode: str = "",
) -> dict[str, Any]:
    """Convert documents to a LangGraph-compatible state dict.

    Parameters
    ----------
    documents : Sequence[CorpusDocument]
        Source documents.
    query : str, optional
        The original query text (for context in graph state).
    match_mode : str, optional
        The match mode used (e.g., ``"hybrid"``).

    Returns
    -------
    dict[str, Any]
        State dict with keys ``"documents"``, ``"query"``,
        ``"match_mode"``, ``"n_results"``.  ``"documents"`` is a
        list of LangChain-compatible dicts.

    Notes
    -----
    **User note:** Use as input to a LangGraph node::

        state = to_langgraph_state(results, query="...")
        graph.invoke(state)
    """
    lc_docs = to_langchain_documents(documents)
    return {
        "documents": lc_docs,
        "query": query,
        "match_mode": match_mode,
        "n_results": len(lc_docs),
    }


# =====================================================================
# MCP (Model Context Protocol) adapter
# =====================================================================


def to_mcp_resources(
    documents: Sequence[Any],
    *,
    uri_prefix: str = "corpus://",
) -> list[dict[str, Any]]:
    """Convert documents to MCP ``resources/read`` response format.

    Parameters
    ----------
    documents : Sequence[CorpusDocument]
        Source documents.
    uri_prefix : str, optional
        URI prefix for resource identifiers.

    Returns
    -------
    list[dict[str, Any]]
        MCP-compatible resource objects with ``uri``, ``name``,
        ``mimeType``, ``text``, and ``metadata`` keys.

    Notes
    -----
    **User note:** These resources can be served by any MCP server
    implementation.  The URI scheme ``corpus://{doc_id}`` provides
    unique addressability for each chunk.

    References
    ----------
    .. [1] Model Context Protocol specification,
       https://modelcontextprotocol.io/
    """
    resources = []
    for doc in documents:
        doc_id = getattr(doc, "doc_id", "unknown")
        text = _get_text(doc)
        meta = _doc_metadata(doc)

        resources.append(
            {
                "uri": f"{uri_prefix}{doc_id}",
                "name": meta.get(
                    "source_title",
                    meta.get("source_file", doc_id),
                ),
                "mimeType": "text/plain",
                "text": text,
                "metadata": meta,
            }
        )

    return resources


def to_mcp_tool_result(
    documents: Sequence[Any],
    *,
    tool_name: str = "corpus_search",
    is_error: bool = False,
) -> dict[str, Any]:
    """Format documents as an MCP ``tools/call`` response.

    Parameters
    ----------
    documents : Sequence[CorpusDocument]
        Search results.
    tool_name : str, optional
        Name of the MCP tool that produced these results.
    is_error : bool, optional
        Whether this response represents an error.

    Returns
    -------
    dict[str, Any]
        MCP tool result with ``content`` array and ``isError`` flag.

    Notes
    -----
    **User note:** Return this from your MCP server's
    ``tools/call`` handler::

        @server.tool("corpus_search")
        async def search(query: str) -> dict:
            results = builder.search(query)
            return to_mcp_tool_result([r.doc for r in results])
    """
    content = []
    for doc in documents:
        text = _get_text(doc)
        meta = _doc_metadata(doc)
        content.append(
            {
                "type": "text",
                "text": text,
                "annotations": {
                    "doc_id": meta.get("doc_id"),
                    "source_file": meta.get("source_file"),
                    "source_title": meta.get("source_title"),
                    "chunk_index": meta.get("chunk_index"),
                    "score": meta.get("score"),
                },
            }
        )

    return {
        "content": content,
        "isError": is_error,
    }


# =====================================================================
# HuggingFace adapter
# =====================================================================


def to_huggingface_dataset(
    documents: Sequence[Any],
) -> Any:
    """Convert documents to a HuggingFace ``Dataset``.

    Parameters
    ----------
    documents : Sequence[CorpusDocument]
        Source documents.

    Returns
    -------
    datasets.Dataset or dict[str, list]
        HuggingFace Dataset.  Falls back to a column-dict if
        ``datasets`` is not installed.

    Notes
    -----
    **User note:** Directly usable for fine-tuning or upload::

        ds = to_huggingface_dataset(docs)
        ds.push_to_hub("my-org/my-corpus")
    """
    columns: dict[str, list] = {
        "doc_id": [],
        "text": [],
        "source_file": [],
        "source_type": [],
        "source_title": [],
        "chunk_index": [],
        "language": [],
        "metadata_json": [],
    }

    for doc in documents:
        columns["doc_id"].append(getattr(doc, "doc_id", ""))
        columns["text"].append(_get_text(doc))
        columns["source_file"].append(getattr(doc, "source_file", ""))
        st = getattr(doc, "source_type", None)
        columns["source_type"].append(str(st) if st is not None else "unknown")
        columns["source_title"].append(getattr(doc, "source_title", None) or "")
        columns["chunk_index"].append(getattr(doc, "chunk_index", 0))
        columns["language"].append(getattr(doc, "language", None) or "")
        columns["metadata_json"].append(json.dumps(_doc_metadata(doc), default=str))

    try:
        from datasets import Dataset  # type: ignore[import]  # noqa: PLC0415

        return Dataset.from_dict(columns)
    except ImportError:
        logger.info("HuggingFace datasets not installed; returning column dict.")
        return columns


# =====================================================================
# Generic RAG adapter
# =====================================================================


def to_rag_tuples(
    documents: Sequence[Any],
) -> list[tuple[str, dict[str, Any], Any]]:
    """Convert documents to ``(text, metadata, embedding)`` tuples.

    Parameters
    ----------
    documents : Sequence[CorpusDocument]
        Source documents.

    Returns
    -------
    list[tuple[str, dict, Any]]
        Each tuple is ``(text, metadata_dict, embedding_or_None)``.
        Compatible with most vector store ``upsert`` interfaces.

    Notes
    -----
    **User note:** Feed directly to any vector store::

        for text, meta, emb in to_rag_tuples(docs):
            vector_store.upsert(id=meta["doc_id"], vector=emb, metadata=meta, text=text)
    """
    return [
        (
            _get_text(doc),
            _doc_metadata(doc),
            getattr(doc, "embedding", None),
        )
        for doc in documents
    ]


# =====================================================================
# JSONL streaming adapter
# =====================================================================


def to_jsonl(
    documents: Sequence[Any],
) -> Iterator[str]:
    r"""
    Yield documents as newline-delimited JSON strings.

    Parameters
    ----------
    documents : Sequence[CorpusDocument]
        Source documents.

    Yields
    ------
    str
        One JSON object per line (no trailing newline).

    Notes
    -----
    **User note:** Write to a file for streaming ingestion::

        with open("corpus.jsonl", "w") as f:
            for line in to_jsonl(docs):
                f.write(line + "\\n")
    """
    for doc in documents:
        record = {
            "text": _get_text(doc),
            **_doc_metadata(doc),
        }
        yield json.dumps(record, default=str, ensure_ascii=False)


# =====================================================================
# LangChain Retriever (class-based adapter)
# =====================================================================


class LangChainCorpusRetriever:
    """LangChain-compatible retriever backed by ``SimilarityIndex``.

    Parameters
    ----------
    index : SimilarityIndex
        A built similarity index.
    embedding_fn : Callable[[str], list[float]] or None, optional
        Function to embed query text.  Required for SEMANTIC mode.
    config : SearchConfig or None, optional
        Default search configuration.

    Notes
    -----
    **User note:** Plug into any LangChain chain::

        retriever = LangChainCorpusRetriever(index, embedding_fn)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
        )

    This class implements the ``BaseRetriever`` interface if
    ``langchain_core`` is installed, otherwise it provides a
    compatible ``get_relevant_documents`` method.

    See Also
    --------
    scikitplot.corpus._similarity.SimilarityIndex : The underlying
        search engine.
    """

    def __init__(
        self,
        index: Any,
        embedding_fn: Any = None,
        config: Any = None,
    ) -> None:
        self.index = index
        self.embedding_fn = embedding_fn
        self.config = config

    def get_relevant_documents(
        self,
        query: str,
    ) -> list[Any]:
        """Retrieve documents relevant to *query*.

        Parameters
        ----------
        query : str
            Natural language query.

        Returns
        -------
        list[langchain_core.documents.Document] or list[dict]
            LangChain-compatible documents.
        """
        qe = None
        if self.embedding_fn is not None:
            qe = self.embedding_fn(query)

        results = self.index.search(
            query,
            config=self.config,
            query_embedding=qe,
        )
        return to_langchain_documents([r.doc for r in results])

    # Alias for newer LangChain versions
    invoke = get_relevant_documents

    def __repr__(self) -> str:
        return (
            f"LangChainCorpusRetriever("
            f"index={self.index!r}, "
            f"has_embedding_fn={self.embedding_fn is not None})"
        )


# =====================================================================
# MCP Server adapter (class-based)
# =====================================================================


class MCPCorpusServer:
    """MCP server adapter for corpus search.

    Provides a structured interface for building MCP servers that
    expose corpus search as tools and resources.

    Parameters
    ----------
    index : SimilarityIndex
        A built similarity index.
    embedding_fn : Callable[[str], list[float]] or None, optional
        Function to embed query text.
    server_name : str, optional
        Name of the MCP server.

    Notes
    -----
    **User note:** Use with any MCP server framework::

        from mcp.server import Server

        mcp_adapter = MCPCorpusServer(index, embedding_fn)

        server = Server("corpus-search")


        @server.tool("search")
        async def search(query: str, top_k: int = 10):
            return mcp_adapter.handle_search(query, top_k=top_k)


        @server.resource("corpus://{doc_id}")
        async def get_doc(doc_id: str):
            return mcp_adapter.handle_resource(doc_id)

    References
    ----------
    .. [1] Model Context Protocol,
       https://modelcontextprotocol.io/
    """

    def __init__(
        self,
        index: Any,
        embedding_fn: Any = None,
        server_name: str = "corpus-search",
    ) -> None:
        self.index = index
        self.embedding_fn = embedding_fn
        self.server_name = server_name

    def handle_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        match_mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Handle an MCP ``tools/call`` request.

        Returns
        -------
        dict[str, Any]
            MCP tool result.
        """
        from scikitplot.corpus._similarity import SearchConfig  # noqa: PLC0415

        qe = None
        if self.embedding_fn is not None:
            qe = self.embedding_fn(query)

        cfg = SearchConfig(top_k=top_k, match_mode=match_mode)
        results = self.index.search(
            query,
            config=cfg,
            query_embedding=qe,
        )
        return to_mcp_tool_result([r.doc for r in results])

    def handle_resource(self, doc_id: str) -> dict[str, Any] | None:
        """Handle an MCP ``resources/read`` request.

        Returns
        -------
        dict[str, Any] or None
            MCP resource, or ``None`` if not found.
        """
        for doc in self.index._documents:
            if getattr(doc, "doc_id", None) == doc_id:
                resources = to_mcp_resources([doc])
                return resources[0] if resources else None
        return None

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions for this server.

        Returns
        -------
        list[dict]
            Tool schemas compatible with MCP ``tools/list``.
        """
        return [
            {
                "name": "corpus_search",
                "description": (
                    "Search the corpus for relevant documents. "
                    "Supports strict, keyword, semantic, and "
                    "hybrid match modes."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results",
                            "default": 10,
                        },
                        "match_mode": {
                            "type": "string",
                            "enum": [
                                "strict",
                                "keyword",
                                "semantic",
                                "hybrid",
                            ],
                            "default": "hybrid",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def __repr__(self) -> str:
        return (
            f"MCPCorpusServer("
            f"name={self.server_name!r}, "
            f"n_docs={self.index.n_documents})"
        )
