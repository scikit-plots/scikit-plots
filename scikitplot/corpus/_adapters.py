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
- **NumPy arrays** — ``(N, D)`` matrices for batch ML workflows
- **TensorFlow** — ``tf.data.Dataset`` for Keras / TF-Hub models
- **PyTorch** — ``DataLoader`` / ``CorpusDataset`` for PyTorch/Lightning
- **MLflow** — artifact logging via :class:`LLMTrainingExporter`

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
    # LLM consumer adapters
    "to_langchain_documents",
    "to_langgraph_state",
    "to_mcp_resources",
    "to_mcp_tool_result",
    "to_huggingface_dataset",
    "to_rag_tuples",
    "to_jsonl",
    # ML / tensor adapters (new)
    "to_numpy_arrays",
    "to_tensorflow_dataset",
    "to_torch_dataloader",
    # Retriever / server adapters
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
    """Return the best available text from a :class:`~scikitplot.corpus.CorpusDocument`.

    Parameters
    ----------
    doc : CorpusDocument
        Document to extract text from.

    Returns
    -------
    str
        ``doc.normalized_text`` when non-empty, otherwise ``doc.text``,
        otherwise empty string.  Never raises.
    """
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
        """Initialise the retriever.

        Parameters
        ----------
        index : SimilarityIndex
            Pre-built corpus index.  Must have a ``search(query, config)``
            method returning a list of :class:`~scikitplot.corpus._similarity.SearchResult`.
        embedding_fn : callable or None, optional
            Function ``str → ndarray`` for semantic search.  When ``None``,
            keyword search is used.  Default: ``None``.
        config : SearchConfig or None, optional
            Default search configuration.  When ``None``, uses
            ``SearchConfig(match_mode="hybrid", top_k=4)``.
            Default: ``None``.
        """
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
        """Return ``LangChainCorpusRetriever(n_docs=N, mode=...)``."""
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
        """Initialise the MCP server adapter.

        Parameters
        ----------
        index : SimilarityIndex
            Pre-built corpus index used to handle ``corpus_search`` tool calls.
        embedding_fn : callable or None, optional
            Function ``str → ndarray`` for semantic query embedding.
            ``None`` uses keyword (BM25) search.  Default: ``None``.
        server_name : str, optional
            MCP server name returned in ``list_tools`` responses.
            Default: ``"corpus-search"``.
        """
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
        """Return ``MCPCorpusServer(name=..., n_docs=N)``."""
        return (
            f"MCPCorpusServer("
            f"name={self.server_name!r}, "
            f"n_docs={self.index.n_documents})"
        )


# ===========================================================================
# ML / tensor consumer adapters — TensorFlow · PyTorch · NumPy
# ===========================================================================


def to_numpy_arrays(
    documents: list[Any],
    *,
    include_text: bool = True,
    include_raw_tensor: bool = True,
    include_embedding: bool = True,
    include_metadata: bool = True,
    dtype_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert documents to a dict of NumPy arrays suitable for batch ML.

    Parameters
    ----------
    documents : list[CorpusDocument]
        Documents to convert.
    include_text : bool, optional
        Include ``texts`` column (list of str, empty str for None).
        Default: ``True``.
    include_raw_tensor : bool, optional
        Stack ``raw_tensor`` fields when all documents share the same
        shape.  Skipped when shapes differ.  Default: ``True``.
    include_embedding : bool, optional
        Stack ``embedding`` fields when all documents have embeddings.
        Default: ``True``.
    include_metadata : bool, optional
        Include ``doc_ids``, ``source_files``, ``source_types`` columns.
        Default: ``True``.
    dtype_map : dict[str, Any] or None, optional
        Override dtypes, e.g. ``{"raw_tensor": "float32"}``.
        Default: ``None``.

    Returns
    -------
    dict[str, Any]
        Column dict. Keys depend on *include_* flags:

        - ``"texts"`` — list[str]
        - ``"raw_tensors"`` — ndarray shape ``(N, H, W, C)`` or ``(N, S)``
          (only when all shapes match)
        - ``"embeddings"`` — ndarray shape ``(N, D)``
        - ``"doc_ids"`` — list[str]
        - ``"source_files"`` — list[str]
        - ``"source_types"`` — list[str]
        - ``"modalities"`` — list[str]
        - ``"content_hashes"`` — list[str | None]

    Notes
    -----
    Requires ``numpy``.  Raises ``ImportError`` when not installed.

    Examples
    --------
    >>> arrays = to_numpy_arrays(docs, include_raw_tensor=True)
    >>> arrays["raw_tensors"].shape  # (N, H, W, C) for image batch
    (32, 224, 224, 3)
    >>> arrays["embeddings"].shape  # (N, D)
    (32, 384)
    """
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("to_numpy_arrays requires numpy: pip install numpy") from exc

    result: dict[str, Any] = {}

    if include_text:
        result["texts"] = [
            getattr(d, "normalized_text", None) or getattr(d, "text", None) or ""
            for d in documents
        ]

    if include_metadata:
        result["doc_ids"] = [getattr(d, "doc_id", "") for d in documents]
        result["source_files"] = [getattr(d, "source_file", "") for d in documents]
        result["source_types"] = [
            str(getattr(d, "source_type", "unknown")) for d in documents
        ]
        result["modalities"] = [str(getattr(d, "modality", "text")) for d in documents]
        result["content_hashes"] = [getattr(d, "content_hash", None) for d in documents]

    if include_raw_tensor:
        tensors = [getattr(d, "raw_tensor", None) for d in documents]
        tensors = [t for t in tensors if t is not None]
        if tensors:
            try:
                stacked = np.stack(tensors, axis=0)
                if dtype_map and "raw_tensor" in dtype_map:
                    stacked = stacked.astype(dtype_map["raw_tensor"])
                result["raw_tensors"] = stacked
            except ValueError:
                # Shapes differ — store as object array
                result["raw_tensors"] = np.array(tensors, dtype=object)

    if include_embedding:
        embeddings = [getattr(d, "embedding", None) for d in documents]
        embeddings = [e for e in embeddings if e is not None]
        if embeddings:
            try:
                emb_arr = np.stack(embeddings, axis=0)
                if dtype_map and "embedding" in dtype_map:
                    emb_arr = emb_arr.astype(dtype_map["embedding"])
                result["embeddings"] = emb_arr
            except (ValueError, TypeError):
                result["embeddings"] = np.array(embeddings, dtype=object)

    return result


def to_tensorflow_dataset(  # noqa: PLR0912
    documents: list[Any],
    *,
    text_feature: bool = True,
    raw_tensor_feature: bool = False,
    embedding_feature: bool = False,
    label_field: str | None = None,
    label_map: dict[str, int] | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
    dtype_map: dict[str, Any] | None = None,
) -> Any:
    """
    Convert documents to a ``tf.data.Dataset``.

    Parameters
    ----------
    documents : list[CorpusDocument]
        Documents to convert.
    text_feature : bool, optional
        Include ``"text"`` feature (tf.string). Default: ``True``.
    raw_tensor_feature : bool, optional
        Include ``"raw_tensor"`` feature (tf.uint8) when documents carry
        pixel arrays.  Requires all tensors to share the same shape.
        Default: ``False``.
    embedding_feature : bool, optional
        Include ``"embedding"`` feature (tf.float32). Default: ``False``.
    label_field : str or None, optional
        ``CorpusDocument`` attribute to use as label (e.g.
        ``"source_type"``).  Default: ``None`` (no label).
    label_map : dict[str, int] or None, optional
        Map string label values to integer class ids.  Required when
        *label_field* is set and the field contains strings.
        Default: ``None``.
    batch_size : int, optional
        Batch size.  ``None`` disables batching.  Default: 32.
    shuffle : bool, optional
        Shuffle the dataset before batching.  Default: ``False``.
    shuffle_seed : int or None, optional
        Seed for deterministic shuffling.  Default: ``None``.
    dtype_map : dict or None, optional
        Cast feature dtypes, e.g. ``{"raw_tensor": tf.float32}``.

    Returns
    -------
    tf.data.Dataset
        Batched dataset of feature dicts (and optionally labels).

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    ValueError
        If *raw_tensor_feature* is True but raw tensors have different
        shapes across documents.

    Notes
    -----
    **Fallback:** When TensorFlow is not available, returns a dict of
    NumPy arrays (via :func:`to_numpy_arrays`) so pipelines can test
    the shape of the output without requiring a GPU environment.

    Examples
    --------
    Text-only dataset for a Keras text classifier:

    >>> ds = to_tensorflow_dataset(docs, text_feature=True, batch_size=16)
    >>> for batch in ds.take(1):
    ...     print(batch["text"].shape)  # (16,)

    Image dataset for a CNN:

    >>> ds = to_tensorflow_dataset(
    ...     docs,
    ...     text_feature=False,
    ...     raw_tensor_feature=True,
    ...     label_field="source_type",
    ...     label_map={"image": 0, "research": 1},
    ... )
    """
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError:
        import logging as _log  # noqa: PLC0415

        _log.getLogger(__name__).warning(
            "tensorflow not installed; returning numpy arrays fallback. "
            "Install with: pip install tensorflow"
        )
        return to_numpy_arrays(
            documents,
            include_text=text_feature,
            include_raw_tensor=raw_tensor_feature,
            include_embedding=embedding_feature,
        )

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("to_tensorflow_dataset requires numpy") from exc

    # Build feature dict
    features: dict[str, Any] = {}

    if text_feature:
        texts = [
            getattr(d, "normalized_text", None) or getattr(d, "text", None) or ""
            for d in documents
        ]
        features["text"] = tf.constant(texts, dtype=tf.string)

    if raw_tensor_feature:
        tensors = [getattr(d, "raw_tensor", None) for d in documents]
        if any(t is None for t in tensors):
            raise ValueError(
                "to_tensorflow_dataset: raw_tensor_feature=True but "
                "some documents have raw_tensor=None. Set yield_raw=True "
                "on the reader."
            )
        try:
            stacked = np.stack(tensors, axis=0)
        except ValueError as exc:
            raise ValueError(
                "to_tensorflow_dataset: raw tensors have different shapes. "
                "Resize images to a common size before calling this function."
            ) from exc
        raw_dtype = (dtype_map or {}).get("raw_tensor", tf.uint8)
        features["raw_tensor"] = tf.cast(tf.constant(stacked), dtype=raw_dtype)

    if embedding_feature:
        embeddings = [getattr(d, "embedding", None) for d in documents]
        if any(e is None for e in embeddings):
            raise ValueError(
                "to_tensorflow_dataset: embedding_feature=True but some "
                "documents have embedding=None. Run EmbeddingEngine first."
            )
        emb_arr = np.stack(embeddings, axis=0).astype(np.float32)
        features["embedding"] = tf.constant(emb_arr, dtype=tf.float32)

    # Build label tensor
    if label_field is not None:
        raw_labels = [getattr(d, label_field, None) for d in documents]
        if label_map is not None:
            int_labels = [label_map.get(str(lbl), -1) for lbl in raw_labels]
            labels_tensor = tf.constant(int_labels, dtype=tf.int32)
        else:
            labels_tensor = tf.constant([str(l) for l in raw_labels], dtype=tf.string)
        ds = tf.data.Dataset.from_tensor_slices((features, labels_tensor))
    else:
        ds = tf.data.Dataset.from_tensor_slices(features)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(documents),
            seed=shuffle_seed,
            reshuffle_each_iteration=False,
        )

    if batch_size:
        ds = ds.batch(batch_size, drop_remainder=False)

    return ds


def to_torch_dataloader(
    documents: list[Any],
    *,
    text_feature: bool = True,
    raw_tensor_feature: bool = False,
    embedding_feature: bool = False,
    label_field: str | None = None,
    label_map: dict[str, int] | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    dtype_map: dict[str, Any] | None = None,
) -> Any:
    """
    Convert documents to a ``torch.utils.data.DataLoader``.

    Parameters
    ----------
    documents : list[CorpusDocument]
        Documents to convert.
    text_feature : bool, optional
        Include ``"text"`` key (list of str per batch). Default: ``True``.
    raw_tensor_feature : bool, optional
        Include ``"raw_tensor"`` key (torch.Tensor, NCHW float32).
        Requires all tensors to have the same shape.  Default: ``False``.
    embedding_feature : bool, optional
        Include ``"embedding"`` key (torch.Tensor, shape ``(N, D)``).
        Default: ``False``.
    label_field : str or None, optional
        Attribute to use as label. Default: ``None``.
    label_map : dict[str, int] or None, optional
        Map string labels to class indices. Default: ``None``.
    batch_size : int, optional
        Batch size. Default: 32.
    shuffle : bool, optional
        Shuffle data each epoch. Default: ``False``.
    num_workers : int, optional
        DataLoader worker processes. Default: 0 (main process only).
    dtype_map : dict or None, optional
        Cast tensors, e.g. ``{"raw_tensor": torch.float32}``.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader over a ``CorpusDataset``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Notes
    -----
    **Fallback:** When PyTorch is not available, returns a dict of
    NumPy arrays so pipelines can test without GPU hardware.

    **Channel order:** Raw tensors from :class:`ImageReader` are
    ``(H, W, C)`` uint8 (channels-last).  This function converts them to
    ``(C, H, W)`` float32 in ``[0, 1]`` (channels-first, PyTorch
    convention) when ``dtype_map`` is not set.

    Examples
    --------
    Image classification loader:

    >>> loader = to_torch_dataloader(
    ...     docs,
    ...     raw_tensor_feature=True,
    ...     label_field="source_type",
    ...     label_map={"image": 0, "research": 1},
    ...     batch_size=16,
    ... )
    >>> for batch in loader:
    ...     imgs = batch["raw_tensor"]  # (16, C, H, W) float32
    ...     labels = batch["label"]  # (16,) int64
    """
    try:
        import torch  # noqa: PLC0415
        from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415
    except ImportError:
        import logging as _log  # noqa: PLC0415

        _log.getLogger(__name__).warning(
            "torch not installed; returning numpy arrays fallback. "
            "Install with: pip install torch"
        )
        return to_numpy_arrays(
            documents,
            include_text=text_feature,
            include_raw_tensor=raw_tensor_feature,
            include_embedding=embedding_feature,
        )

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("to_torch_dataloader requires numpy") from exc

    class CorpusDataset(Dataset):  # type: ignore[type-arg]
        """In-memory PyTorch Dataset wrapping CorpusDocuments."""

        def __init__(self, docs: list) -> None:
            """Store the document list.

            Parameters
            ----------
            docs : list[CorpusDocument]
                Documents to wrap.
            """
            self.docs = docs

        def __len__(self) -> int:
            return len(self.docs)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            doc = self.docs[idx]
            item: dict[str, Any] = {}

            if text_feature:
                item["text"] = (
                    getattr(doc, "normalized_text", None)
                    or getattr(doc, "text", None)
                    or ""
                )

            if raw_tensor_feature:
                arr = getattr(doc, "raw_tensor", None)
                if arr is None:
                    raise ValueError(
                        f"raw_tensor_feature=True but doc {idx} has raw_tensor=None."
                    )
                arr_np = np.array(arr)
                # Convert HWC uint8 → CHW float32 [0,1] (PyTorch convention)
                target_dtype = (dtype_map or {}).get("raw_tensor", torch.float32)
                if arr_np.ndim == 3:  # (H, W, C)  # noqa: PLR2004
                    t = torch.from_numpy(arr_np).permute(2, 0, 1)
                else:
                    t = torch.from_numpy(arr_np)
                if target_dtype == torch.float32:
                    t = t.float() / 255.0
                else:
                    t = t.to(target_dtype)
                item["raw_tensor"] = t

            if embedding_feature:
                emb = getattr(doc, "embedding", None)
                if emb is None:
                    raise ValueError(
                        f"embedding_feature=True but doc {idx} has embedding=None."
                    )
                item["embedding"] = torch.tensor(np.array(emb), dtype=torch.float32)

            if label_field is not None:
                raw_lbl = getattr(doc, label_field, None)
                if label_map is not None:
                    item["label"] = torch.tensor(
                        label_map.get(str(raw_lbl), -1), dtype=torch.long
                    )
                else:
                    item["label"] = str(raw_lbl)

            return item

    dataset = CorpusDataset(documents)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_torch_collate_fn,
    )


def _torch_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate that handles mixed text/tensor batches.

    String fields are collated as lists; tensor fields use the default
    torch collation (stack along batch dimension).
    """
    result: dict[str, Any] = {}
    if not batch:
        return result
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], str):
            result[key] = values  # keep as list of str
        else:
            try:
                import torch  # noqa: PLC0415

                result[key] = torch.stack(values)
            except Exception:  # noqa: BLE001
                result[key] = values
    return result
