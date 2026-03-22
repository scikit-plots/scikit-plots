# scikitplot/corpus/_corpus_builder.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Unified corpus builder — single entry point for end-to-end pipelines.

:class:`CorpusBuilder` orchestrates every submodule in
``scikitplot.corpus`` through a fluent, user-friendly API.  It wraps
the existing ``CorpusPipeline``, ``DocumentReader``, normaliser,
enricher, embedding, similarity, and adapter layers into one object.

.. code-block:: text

   Source (file / URL / dir / manifest)
       │
       ▼  ① Auto-detect format, route to reader
   DocumentReader
       │
       ▼  ② Split text into chunks
   Chunker (sentence / paragraph / fixed-window / word)
       │
       ▼  ③ Filter short / empty / low-confidence chunks
   Filter
       │
       ▼  ④ Clean text for embedding quality
   TextNormalizer                           ← optional
       │
       ▼  ⑤ Populate tokens / lemmas / stems / keywords
   NLPEnricher                              ← optional
       │
       ▼  ⑥ Embed with sentence-transformers / OpenAI / custom
   EmbeddingEngine                          ← optional
       │
       ▼  ⑦ Build similarity index
   SimilarityIndex                          ← optional
       │
       ▼  ⑧ Export / adapt for downstream consumers
   Export / Adapters
       ├── LangChain Documents
       ├── LangGraph state
       ├── MCP resources / tools
       ├── HuggingFace Dataset
       ├── RAG vector-store tuples
       ├── JSONL / CSV / Parquet
       └── MLflow logging

.. admonition:: Design principles

   1. **One object, full pipeline** — no manual wiring.
   2. **Sane defaults, full control** — every component is optional
      and configurable.
   3. **Lazy evaluation** — heavy components (embedder, spaCy, FAISS)
      load only when first needed.
   4. **Immutable documents** — the builder never mutates input
      documents; each stage yields new instances via ``replace()``.
   5. **Format-agnostic** — text, PDF, image, video, audio, web,
      XML, ALTO all enter through the same API.

.. code-block:: python

    from scikitplot.corpus import CorpusBuilder, BuilderConfig

    # Simple: process a directory of PDFs
    builder = CorpusBuilder()
    result = builder.build("./papers/")

    # Full pipeline
    config = BuilderConfig(
        chunker="paragraph",
        normalize=True,
        enrich=True,
        embed=True,
        build_index=True,
        collection_id="shakespeare-corpus",
    )
    builder = CorpusBuilder(config)
    result = builder.build(["hamlet.txt", "othello.txt"])

    # Search
    results = builder.search("To be or not to be")

    # Export to LangChain / MCP / HuggingFace / RAG
    lc_docs = builder.to_langchain()
    mcp_response = builder.to_mcp_tool_result("death soliloquy")

Supports Python 3.8 through 3.15.
"""

from __future__ import annotations

import glob as _glob
import logging
import os  # noqa: F401
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence  # noqa: F401

from typing_extensions import Self

logger = logging.getLogger(__name__)

__all__ = [
    "BuildResult",
    "BuilderConfig",
    "CorpusBuilder",
]


# =====================================================================
# Configuration
# =====================================================================


@dataclass
class BuilderConfig:
    """Configuration for :class:`CorpusBuilder`.

    Parameters
    ----------
    chunker : str or object
        Chunker to use.  One of ``"sentence"``, ``"paragraph"``,
        ``"fixed_window"``, ``"word"``; or a pre-configured chunker
        instance (either ``ChunkerBase`` subclass or new-style
        chunker — auto-bridged).
    chunker_kwargs : dict[str, Any]
        Keyword arguments passed to the chunker constructor
        (ignored if *chunker* is already an instance).
    normalize : bool
        Run normalisation pipeline after filtering.
    normalizer_steps : list[str]
        Normaliser names: ``"unicode"``, ``"whitespace"``,
        ``"html_strip"``, ``"lowercase"``, ``"dedup_lines"``.
        Default: ``["unicode", "whitespace"]``.
    normalizer_kwargs : dict[str, Any]
        Run ``TextNormalizer`` after filtering.
        Kwargs for ``NormalizerConfig``.
    enrich : bool
        Run ``NLPEnricher`` after normalisation.
    enricher_kwargs : dict[str, Any]
        Kwargs for ``EnricherConfig``.
    embed : bool
        Run ``EmbeddingEngine`` after enrichment.
    embedding_model : str
        Model name for ``EmbeddingEngine``.
    embedding_kwargs : dict[str, Any]
        Kwargs for ``EmbeddingEngine`` constructor.
    build_index : bool
        Build a ``SimilarityIndex`` after embedding.
    index_kwargs : dict[str, Any]
        Kwargs for ``SearchConfig``.
    source_title : str or None
        Default ``source_title`` for all documents.
    source_author : str or None
        Default ``source_author`` for all documents.
    source_type : str or None
        Default ``source_type`` (e.g., ``"book"``, ``"movie"``).
    collection_id : str or None
        Group identifier for this corpus build.
    default_language : str or None
        ISO 639-1 language code.
    filter_kwargs : dict[str, Any]
        Kwargs for ``DefaultFilter``.
    max_workers : int
        Parallelism for multi-file ingestion.
    probe_url_content_type : bool
        When ``True`` (default), extensionless URLs are probed with an
        HTTP HEAD request to infer the correct reader before downloading.
        Disable to save a round-trip when all URLs have file extensions.
    probe_url_timeout : int
        HTTP timeout in seconds for :func:`~scikitplot.corpus._url_handler.probe_url_kind`
        calls.  Default: 15.

    Notes
    -----
    **User note:** Most users need only::

        config = BuilderConfig(chunker="sentence", embed=True)

    Everything else has sensible defaults.
    """

    # Chunking
    chunker: str | Any = "sentence"
    chunker_kwargs: dict[str, Any] = field(default_factory=dict)

    # Normalisation
    normalize: bool = True
    normalizer_steps: list[str] = field(
        default_factory=lambda: ["unicode", "whitespace"]
    )
    normalizer_kwargs: dict[str, Any] = field(default_factory=dict)

    # NLP enrichment
    enrich: bool = False
    enricher_kwargs: dict[str, Any] = field(default_factory=dict)

    # Embedding
    embed: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_kwargs: dict[str, Any] = field(default_factory=dict)

    # Similarity index
    build_index: bool = False
    index_kwargs: dict[str, Any] = field(default_factory=dict)

    # Provenance defaults
    source_title: str | None = None
    source_author: str | None = None
    source_type: str | None = None
    collection_id: str | None = None
    default_language: str | None = None

    # Filter
    filter_kwargs: dict[str, Any] = field(default_factory=dict)

    # Download / URL handling
    max_download_bytes: int = 500 * 1024 * 1024
    """Maximum download size per URL in bytes. Default: 500 MB."""

    download_timeout: int = 120
    """HTTP timeout for URL downloads in seconds. Default: 120."""

    download_max_retries: int = 3
    """Maximum retry attempts for transient HTTP errors (429, 500, 502,
    503, 504) during URL downloads.  Set to ``0`` to disable retries.
    Default: 3."""

    download_retry_backoff: float = 1.0
    """Base delay in seconds for exponential back-off between download
    retries.  Actual wait = ``download_retry_backoff * 2 ** attempt``.
    Default: 1.0."""

    max_archive_files: int = 10_000
    """Maximum file count inside a single archive. Default: 10,000."""

    max_archive_bytes: int = 2 * 1024 * 1024 * 1024
    """Maximum cumulative extracted size per archive. Default: 2 GB."""

    # URL content-type probing
    probe_url_content_type: bool = True
    """Probe extensionless URLs with a HEAD request to determine the correct
    reader.  When ``True`` (default), any URL that :func:`classify_url`
    classifies as ``WEB_PAGE`` *and* has no file extension in its path
    is probed via :func:`probe_url_kind` before routing.  Set to
    ``False`` to skip the extra network round-trip (e.g. when all your
    URLs already carry file extensions or you want pure-offline
    operation).  Default: ``True``."""

    probe_url_timeout: int = 15
    """HTTP timeout in seconds for the URL-probing HEAD request.
    Default: 15."""

    # Parallelism
    max_workers: int = 1


# =====================================================================
# Build result
# =====================================================================


@dataclass
class BuildResult:
    """Result of a corpus build operation.

    Parameters
    ----------
    documents : list[CorpusDocument]
        The processed documents.
    n_sources : int
        Number of source files/URLs processed.
    n_raw : int
        Total raw chunks before filtering.
    n_filtered : int
        Chunks removed by filtering.
    n_normalised : int
        Chunks that were text-normalised.
    n_enriched : int
        Chunks that were NLP-enriched.
    n_embedded : int
        Chunks that were embedded.
    index : SimilarityIndex or None
        Built similarity index (if ``build_index=True``).
    errors : list[tuple[str, Exception]]
        ``(source_path, exception)`` pairs for failed sources.

    Notes
    -----
    **User note:** Access documents directly::

        result = builder.build("./data/")
        for doc in result.documents:
            print(doc.text[:80])
    """

    documents: list[Any] = field(default_factory=list)
    n_sources: int = 0
    n_raw: int = 0
    n_filtered: int = 0
    n_normalised: int = 0
    n_enriched: int = 0
    n_embedded: int = 0
    index: Any = None
    errors: list[tuple[str, Exception]] = field(default_factory=list)

    @property
    def n_documents(self) -> int:
        """Number of :class:`~scikitplot.corpus.CorpusDocument` instances in :attr:`documents`.

        Returns
        -------
        int
        """
        return len(self.documents)

    @property
    def success_rate(self) -> float:
        """Fraction of ingested sources that completed without error.

        Returns
        -------
        float
            ``(n_sources - len(errors)) / n_sources`` in ``[0.0, 1.0]``.
            Returns ``1.0`` when no sources were processed.
        """
        total = self.n_sources
        if total == 0:
            return 1.0
        return (total - len(self.errors)) / total

    def summary(self) -> str:
        """Return a multi-line human-readable build summary.

        Returns
        -------
        str
            Multi-line string reporting sources, documents, normalisation,
            enrichment, embedding counts, and any errors.
        """
        lines = [
            "CorpusBuilder result:",
            f"  Sources:    {self.n_sources}",
            f"  Raw chunks: {self.n_raw}",
            f"  Filtered:   {self.n_filtered}",
            f"  Documents:  {self.n_documents}",
            f"  Normalised: {self.n_normalised}",
            f"  Enriched:   {self.n_enriched}",
            f"  Embedded:   {self.n_embedded}",
            f"  Index:      {'built' if self.index else 'none'}",
            f"  Errors:     {len(self.errors)}",
        ]
        return "\n".join(lines)


# =====================================================================
# CorpusBuilder
# =====================================================================


class CorpusBuilder:
    """Unified corpus builder — end-to-end pipeline orchestrator.

    Parameters
    ----------
    config : BuilderConfig or None, optional
        Pipeline configuration.  ``None`` uses defaults.

    Notes
    -----
    **User note:** Typical usage::

        from scikitplot.corpus import CorpusBuilder, BuilderConfig

        # Simple: process a directory of PDFs
        builder = CorpusBuilder()
        result = builder.build("./papers/")

        # Full pipeline: chunk → normalise → enrich → embed → index
        config = BuilderConfig(
            chunker="paragraph",
            normalize=True,
            enrich=True,
            embed=True,
            build_index=True,
            collection_id="shakespeare-corpus",
        )
        builder = CorpusBuilder(config)
        result = builder.build(["hamlet.txt", "othello.txt"])

        # Search
        results = builder.search("To be or not to be")

        # Export to LangChain
        lc_docs = builder.to_langchain()

        # Export to MCP
        mcp_result = builder.to_mcp_tool_result("death soliloquy")

    **Developer note:** The builder is the single orchestration
    point.  It lazily creates component instances on first use
    and caches them.  Each ``build()`` call produces an independent
    ``BuildResult``.

    See Also
    --------
    scikitplot.corpus._pipeline.CorpusPipeline : Lower-level
        pipeline (used internally by the builder).
    scikitplot.corpus._adapters : Conversion functions for
        downstream consumers.
    scikitplot.corpus._similarity.SimilarityIndex : Search engine.

    Examples
    --------
    >>> builder = CorpusBuilder(BuilderConfig(embed=True))
    >>> result = builder.build("./data/books/")
    >>> print(result.summary())
    >>> lc_docs = builder.to_langchain()
    """

    def __init__(
        self,
        config: BuilderConfig | None = None,
    ) -> None:
        """Initialise the builder with a :class:`BuilderConfig`.

        Parameters
        ----------
        config : BuilderConfig or None, optional
            Configuration object controlling every stage of the pipeline.
            When ``None``, defaults are used:
            ``BuilderConfig(chunker="sentence", normalize=True)``.

        Notes
        -----
        Heavy components (embedder, enricher, similarity index) are
        **lazily initialised** — the first call to :meth:`build` loads
        them.  Constructing a :class:`CorpusBuilder` is therefore fast
        and does not import optional dependencies.
        """
        self.config = config or BuilderConfig()
        self._result: BuildResult | None = None

        # Lazy-initialised components
        self._chunker: Any = None
        self._normalizer_pipeline: Any = None
        self._enricher: Any = None
        self._embedding_engine: Any = None
        self._index: Any = None

        # Accumulates omitted-chunk count during a single build() call.
        # Reset to 0 at the start of each build() and read after ingestion.
        self._n_filtered_current_build: int = 0

        # Temporary directory for downloaded/extracted files.
        # Created lazily on first use; cleaned up by close() or __exit__.
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    # ------------------------------------------------------------------
    # Context manager — auto-cleanup of temp files
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Enter the context manager.

        Enables ``with CorpusBuilder(config) as builder:`` pattern.
        Resources are lazy-loaded on the first :meth:`build` call.

        Returns
        -------
        CorpusBuilder
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager.

        Notes
        -----
        Cleans up any temporary directories created during URL ingestion.
        The internal ``_result`` and lazy-loaded components remain usable
        after the ``with`` block.
        """
        self.close()

    def close(self) -> None:
        """Clean up temporary files created during downloads/extraction.

        Notes
        -----
        Safe to call multiple times. After calling, the builder can
        still be used — a new temp directory will be created on next
        download.
        """
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Temp dir cleanup error: %s", exc)
            self._temp_dir = None

    def _get_temp_dir(self) -> Path:
        """Get or create the temporary directory for downloads/extraction.

        Returns
        -------
        Path
            Absolute path to the temporary directory.
        """
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="skplt_corpus_")
        return Path(self._temp_dir.name)

    # ==================================================================
    # Primary API: build
    # ==================================================================

    def build(  # noqa: PLR0912
        self,
        sources: str | Path | Sequence[str | Path],
        *,
        source_title: str | None = None,
        source_author: str | None = None,
        collection_id: str | None = None,
    ) -> BuildResult:
        """Build a corpus from one or more sources.

        Parameters
        ----------
        sources : str, Path, or Sequence[str | Path]
            File path(s), directory path(s), or URL(s).  Accepts:

            - A single file path: ``"hamlet.txt"``
            - A directory: ``"./papers/"`` (recursive)
            - A URL: ``"https://example.com/article"``
            - A list of any mix: ``["a.pdf", "b.mp4", "https://..."]``
        source_title : str or None, optional
            Override ``config.source_title`` for this build.
        source_author : str or None, optional
            Override ``config.source_author``.
        collection_id : str or None, optional
            Override ``config.collection_id``.

        Returns
        -------
        BuildResult
            The build result with documents, counts, and index.

        Raises
        ------
        ValueError
            If no valid sources are found.
        """
        cfg = self.config
        title = source_title or cfg.source_title
        author = source_author or cfg.source_author
        coll_id = collection_id or cfg.collection_id

        # Normalise sources to a list
        if isinstance(sources, (str, Path)):  # noqa: SIM108
            source_list = [sources]
        else:
            source_list = list(sources)

        # Expand directories
        expanded = self._expand_sources(source_list)
        if not expanded:
            raise ValueError(f"No valid sources found in: {source_list!r}")

        result = BuildResult(n_sources=len(expanded))

        # ① Ingest: source → reader → raw chunks → documents
        # Reset per-build omitted counter before ingestion starts.
        self._n_filtered_current_build = 0
        all_docs: list[Any] = []

        if cfg.max_workers > 1 and len(expanded) > 1:
            # Parallel ingestion via ThreadPoolExecutor.
            # IO-bound work (downloads, OCR, ASR) benefits from threads.
            # self._n_filtered_current_build is protected by the GIL:
            # reads/writes to a Python int are atomic at the bytecode level.
            import concurrent.futures  # noqa: PLC0415

            n_workers = min(cfg.max_workers, len(expanded))
            logger.info(
                "build(): parallel ingestion of %d sources with %d workers.",
                len(expanded),
                n_workers,
            )

            def _ingest_one(src: Any) -> tuple[Any, list[Any], Exception | None]:
                try:
                    docs = self._ingest_source(
                        src,
                        source_title=title,
                        source_author=author,
                        collection_id=coll_id,
                    )
                    return src, docs, None
                except Exception as exc:  # noqa: BLE001
                    return src, [], exc

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers,
                thread_name_prefix="skplt_corpus_",
            ) as executor:
                futures = {executor.submit(_ingest_one, src): src for src in expanded}
                for future in concurrent.futures.as_completed(futures):
                    src, docs, exc = future.result()
                    if exc is not None:
                        logger.error("Failed to ingest %s: %s", src, exc)
                        result.errors.append((str(src), exc))
                    else:
                        all_docs.extend(docs)
        else:
            # Serial ingestion (default, max_workers=1).
            for src in expanded:
                try:
                    docs = self._ingest_source(
                        src,
                        source_title=title,
                        source_author=author,
                        collection_id=coll_id,
                    )
                    all_docs.extend(docs)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to ingest %s: %s", src, exc)
                    result.errors.append((str(src), exc))

        result.n_raw = len(all_docs)

        # ② Filter count — the reader's get_documents() already filtered
        # chunks and stored the omitted count in _last_n_omitted.  Collect
        # it from every reader that was created during _ingest_source.
        # _last_n_omitted is set on each DocumentReader after get_documents()
        # completes.  We sum it here so BuildResult.n_filtered is accurate.
        result.n_filtered = self._n_filtered_current_build

        documents = all_docs

        # ④ Normalise (using existing _normalizers.NormalizationPipeline)
        if cfg.normalize and documents:
            # normalizer = self._get_normalizer()
            pipeline = self._get_normalizer_pipeline()
            documents = pipeline.normalize_batch(documents)
            result.n_normalised = sum(
                1 for d in documents if getattr(d, "normalized_text", None) is not None
            )

        # ⑤ Enrich (NLPEnricher — new component)
        if cfg.enrich and documents:
            enricher = self._get_enricher()
            documents = enricher.enrich_documents(documents)
            result.n_enriched = sum(
                1 for d in documents if getattr(d, "tokens", None) is not None
            )

        # ⑥ Embed
        if cfg.embed and documents:
            documents, n_embedded = self._embed_documents(documents)
            result.n_embedded = n_embedded

        result.documents = documents

        # ⑦ Build index
        if cfg.build_index and documents:
            from ._similarity import (  # noqa: PLC0415
                SearchConfig,
                SimilarityIndex,
            )

            idx_cfg = SearchConfig(**cfg.index_kwargs)
            self._index = SimilarityIndex(config=idx_cfg)
            self._index.build(documents)
            result.index = self._index

        self._result = result
        logger.info(result.summary())
        return result

    def add(
        self,
        sources: str | Path | Sequence[str | Path],
        *,
        source_title: str | None = None,
        source_author: str | None = None,
        source_type: str | None = None,
        collection_id: str | None = None,
        rebuild_index: bool = True,
    ) -> BuildResult:
        """Add sources to an existing corpus without re-processing.

        Incrementally ingests new sources and appends their documents to
        the existing :attr:`BuildResult`. Optionally rebuilds the
        similarity index to include the new documents.

        Parameters
        ----------
        sources : str, Path, or Sequence[str | Path]
            File path(s), directory path(s), or URL(s) to add.
        source_title : str or None, optional
            Override title for new sources.
        source_author : str or None, optional
            Override author for new sources.
        source_type : str or None, optional
            Override ``source_type`` for new sources (e.g. ``"audio"``).
            When ``None`` the type is inferred from each file extension.
            Default: ``None``.
        collection_id : str or None, optional
            Override collection id for new sources.
        rebuild_index : bool, optional
            When ``True`` and ``config.build_index`` is enabled,
            rebuild the similarity index with all documents (existing +
            new). Default: ``True``.

        Returns
        -------
        BuildResult
            The updated result containing all documents.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called yet.
        ValueError
            If no valid sources are found.

        Notes
        -----
        **User note:** Use this to extend a corpus after the initial
        ``build()``::

            builder = CorpusBuilder(config)
            result = builder.build("./initial_data/")
            result = builder.add("./new_data/")
            result = builder.add("https://example.com/article")

        **Developer note:** Normalisation, enrichment, and embedding
        are applied to the new documents only. The index is rebuilt
        from scratch over all documents because incremental index
        updates are not supported by all backends.
        """
        if self._result is None:
            raise RuntimeError(
                "No corpus built yet. Call build() first, then add() to extend."
            )

        # Save existing documents before build() replaces _result
        existing_docs = list(self._result.documents)
        existing_counts = (
            self._result.n_sources,
            self._result.n_raw,
            self._result.n_filtered,
            self._result.n_normalised,
            self._result.n_enriched,
            self._result.n_embedded,
            list(self._result.errors),
        )

        # Temporarily override source_type in config when the caller
        # supplies an explicit value.  Restore it after build() returns
        # (or raises) so the builder's config remains consistent.
        _prev_source_type = self.config.source_type
        if source_type is not None:
            self.config.source_type = source_type
        try:
            # Build new sources (this replaces self._result)
            new_result = self.build(
                sources,
                source_title=source_title,
                source_author=source_author,
                collection_id=collection_id,
            )
        finally:
            self.config.source_type = _prev_source_type

        # Merge: prepend existing docs, accumulate counts
        merged_docs = existing_docs + new_result.documents
        new_result.documents = merged_docs
        new_result.n_sources += existing_counts[0]
        new_result.n_raw += existing_counts[1]
        new_result.n_filtered += existing_counts[2]
        new_result.n_normalised += existing_counts[3]
        new_result.n_enriched += existing_counts[4]
        new_result.n_embedded += existing_counts[5]
        new_result.errors = existing_counts[6] + new_result.errors

        # Rebuild index over all documents if configured
        if rebuild_index and self.config.build_index and merged_docs:
            from ._similarity import (  # noqa: PLC0415
                SearchConfig,
                SimilarityIndex,
            )

            idx_cfg = SearchConfig(**self.config.index_kwargs)
            self._index = SimilarityIndex(config=idx_cfg)
            self._index.build(merged_docs)
            new_result.index = self._index

        self._result = new_result
        logger.info(
            "add(): merged %d existing + %d new = %d total documents.",
            len(existing_docs),
            len(new_result.documents) - len(existing_docs),
            len(merged_docs),
        )
        return self._result

    # ==================================================================
    # Search API
    # ==================================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        match_mode: str = "hybrid",
        **kwargs: Any,
    ) -> list[Any]:
        """Search the built corpus.

        Parameters
        ----------
        query : str
            Natural language query.
        top_k : int, optional
            Maximum results.
        match_mode : str, optional
            ``"strict"``, ``"keyword"``, ``"semantic"``, or
            ``"hybrid"``.
        **kwargs
            Additional ``SearchConfig`` parameters.

        Returns
        -------
        list[SearchResult]
            Ranked results.

        Raises
        ------
        RuntimeError
            If no index has been built.
        RuntimeError
            If ``match_mode`` is ``"semantic"`` or ``"hybrid"`` and
            no embedding engine is configured.
        """
        if self._index is None:
            raise RuntimeError(
                "No index built. Call build() with config.build_index=True first."
            )

        if match_mode in ("semantic", "hybrid") and not self.config.embed:
            raise RuntimeError(
                f"match_mode={match_mode!r} requires embeddings. "
                f"Set embed=True in BuilderConfig and rebuild."
            )

        from ._similarity import SearchConfig  # noqa: PLC0415

        cfg = SearchConfig(
            top_k=top_k,
            match_mode=match_mode,
            **kwargs,
        )

        # Auto-embed query for SEMANTIC/HYBRID
        qe = None
        if match_mode in ("semantic", "hybrid"):
            engine = self._get_embedding_engine()
            if engine is not None:
                embs = engine.embed([query])
                if embs and len(embs) > 0:
                    qe = embs[0]

        return self._index.search(
            query,
            config=cfg,
            query_embedding=qe,
        )

    # ==================================================================
    # Export / adapter API
    # ==================================================================

    def to_langchain(self) -> list[Any]:
        """Export documents as LangChain ``Document`` objects.

        Returns
        -------
        list[langchain_core.documents.Document] or list[dict]
        """
        from scikitplot.corpus._adapters import to_langchain_documents  # noqa: PLC0415

        return to_langchain_documents(self._get_documents())

    def to_langgraph_state(
        self,
        query: str = "",
        match_mode: str = "",
    ) -> dict[str, Any]:
        """Export as LangGraph-compatible state dict.

        Returns
        -------
        dict[str, Any]
        """
        from scikitplot.corpus._adapters import to_langgraph_state  # noqa: PLC0415

        return to_langgraph_state(
            self._get_documents(),
            query=query,
            match_mode=match_mode,
        )

    def to_mcp_resources(
        self,
        uri_prefix: str = "corpus://",
    ) -> list[dict[str, Any]]:
        """Export as MCP resources.

        Returns
        -------
        list[dict[str, Any]]
        """
        from scikitplot.corpus._adapters import to_mcp_resources  # noqa: PLC0415

        return to_mcp_resources(
            self._get_documents(),
            uri_prefix=uri_prefix,
        )

    def to_mcp_tool_result(
        self,
        query: str,
        *,
        top_k: int = 10,
        match_mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Search and format result as MCP tool response.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int, optional
            Maximum results.
        match_mode : str, optional
            Match mode.

        Returns
        -------
        dict[str, Any]
            MCP ``tools/call`` response.
        """
        from scikitplot.corpus._adapters import to_mcp_tool_result  # noqa: PLC0415

        results = self.search(
            query,
            top_k=top_k,
            match_mode=match_mode,
        )
        return to_mcp_tool_result([r.doc for r in results])

    def to_mcp_server(
        self,
        server_name: str = "corpus-search",
    ) -> Any:
        """Create an MCP server adapter.

        Parameters
        ----------
        server_name : str, optional
            MCP server name.

        Returns
        -------
        MCPCorpusServer
        """
        from scikitplot.corpus._adapters import MCPCorpusServer  # noqa: PLC0415

        if self._index is None:
            raise RuntimeError(
                "No index built. Call build() with config.build_index=True first."
            )
        return MCPCorpusServer(
            index=self._index,
            embedding_fn=self._make_embed_fn(),
            server_name=server_name,
        )

    def to_langchain_retriever(self) -> Any:
        """Create a LangChain-compatible retriever.

        Returns
        -------
        LangChainCorpusRetriever
        """
        from scikitplot.corpus._adapters import (  # noqa: PLC0415
            LangChainCorpusRetriever,
        )

        if self._index is None:
            raise RuntimeError(
                "No index built. Call build() with config.build_index=True first."
            )
        return LangChainCorpusRetriever(
            index=self._index,
            embedding_fn=self._make_embed_fn(),
            config=self._index.config,
        )

    def to_huggingface(self) -> Any:
        """Export as HuggingFace Dataset.

        Returns
        -------
        datasets.Dataset or dict[str, list]
        """
        from scikitplot.corpus._adapters import to_huggingface_dataset  # noqa: PLC0415

        return to_huggingface_dataset(self._get_documents())

    def to_rag_tuples(self) -> list[tuple[str, dict[str, Any], Any]]:
        """Export as ``(text, metadata, embedding)`` tuples.

        Returns
        -------
        list[tuple[str, dict, Any]]
        """
        from scikitplot.corpus._adapters import to_rag_tuples  # noqa: PLC0415

        return to_rag_tuples(self._get_documents())

    def to_jsonl(self) -> Iterator[str]:
        """Export as JSONL lines.

        Yields
        ------
        str
        """
        from scikitplot.corpus._adapters import to_jsonl  # noqa: PLC0415

        return to_jsonl(self._get_documents())

    def export(
        self,
        path: str | Path,
        *,
        format: str = "parquet",
        **kwargs: Any,
    ) -> Path:
        """Export documents to a file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        format : str, optional
            ``"csv"``, ``"parquet"``, ``"jsonl"``, ``"json"``,
            ``"pickle"``.
        **kwargs
            Additional kwargs for the export function.

        Returns
        -------
        Path
            The output file path.
        """
        from scikitplot.corpus._export._export import export_documents  # noqa: PLC0415

        docs = self._get_documents()
        out_path = Path(path)
        export_documents(docs, out_path, format=format, **kwargs)
        logger.info("Exported %d documents to %s", len(docs), out_path)
        return out_path

    # ==================================================================
    # Internal: source expansion
    # ==================================================================

    @staticmethod
    def _expand_sources(  # noqa: PLR0912
        sources: list[str | Path],
    ) -> list[str | Path]:
        """Expand directories, glob patterns, and validate paths.

        Handles four kinds of source entries:

        1. **URLs** — passed through unchanged (``http://`` or ``https://``).
        2. **Glob patterns** — expanded via ``glob.glob`` with recursive
           support (``**`` is honoured). A string is treated as a glob if
           it contains ``*``, ``?``, or ``[`` and is *not* a URL.
        3. **Directories** — recursively expanded. Only files with
           extensions registered in the ``DocumentReader`` registry are
           included.
        4. **Files** — included directly (even if their extension is not
           registered — the reader will fail later with a clear error).

        Hidden files/directories (starting with ``"."``) and
        ``__pycache__`` are always excluded from directory and glob
        expansion. Symlinks are skipped to prevent traversal loops.
        """
        from scikitplot.corpus._base import DocumentReader  # noqa: PLC0415

        supported = set(DocumentReader.supported_types())
        expanded: list[str | Path] = []

        for src in sources:
            src_str = str(src)

            # 1. URL — pass through
            if src_str.startswith(("http://", "https://")):
                expanded.append(src_str)
                continue

            # 2. Glob pattern — expand and filter
            if any(c in src_str for c in ("*", "?", "[")):
                matches = sorted(_glob.glob(src_str, recursive=True))
                if not matches:
                    logger.warning("Glob pattern matched no files: %s", src_str)
                    continue
                for m in matches:
                    mp = Path(m)
                    if not mp.is_file():
                        continue
                    if mp.is_symlink():
                        logger.debug("Skipping symlink: %s", mp)
                        continue
                    if any(
                        part.startswith(".") or part == "__pycache__"
                        for part in mp.parts
                    ):
                        continue
                    expanded.append(mp)
                continue

            # 3 / 4. Path (file or directory)
            # Guard: empty or whitespace-only strings silently resolve to CWD
            # via Path("").is_dir() == True.  Treat them as invalid sources.
            if not src_str.strip():
                logger.warning(
                    "_expand_sources: ignoring empty or whitespace-only source %r.",
                    src_str,
                )
                continue

            p = Path(src_str)
            if p.is_file():
                expanded.append(p)
            elif p.is_dir():
                for fp in sorted(p.rglob("*")):
                    if not fp.is_file():
                        continue
                    if fp.is_symlink():
                        logger.debug("Skipping symlink: %s", fp)
                        continue
                    if any(
                        part.startswith(".") or part == "__pycache__"
                        for part in fp.relative_to(p).parts
                    ):
                        continue
                    if supported and fp.suffix.lower() not in supported:
                        logger.debug("Skipping unsupported extension: %s", fp)
                        continue
                    expanded.append(fp)
            else:
                logger.warning("Source not found: %s", src)

        return expanded

    # ==================================================================
    # Internal: source ingestion
    # ==================================================================

    def _ingest_source(
        self,
        source: str | Path,
        *,
        source_title: str | None = None,
        source_author: str | None = None,
        collection_id: str | None = None,
    ) -> list[Any]:
        """Ingest a single source file or URL into documents.

        Handles three categories of source:

        1. **URLs** — classified via :func:`~._url_handler.classify_url`.
           ``WEB_PAGE`` and ``YOUTUBE`` URLs are routed to
           ``DocumentReader.from_url`` as before. All other URL kinds
           (downloadable file, Google Drive, GitHub) are downloaded to
           a temp file and then processed via ``DocumentReader.create``.
        2. **Archive files** (zip, tar, tar.gz, etc.) — extracted to a
           temp directory, and each supported file inside is processed
           individually via ``DocumentReader.create``.
        3. **Regular files** — processed via ``DocumentReader.create``
           using the extension-based reader registry.

        Parameters
        ----------
        source : str or Path
            File path or URL.
        source_title : str or None, optional
            Override title for provenance.
        source_author : str or None, optional
            Override author for provenance.
        collection_id : str or None, optional
            Override collection id for provenance.

        Returns
        -------
        list[CorpusDocument]
            Documents ingested from the source.
        """
        from scikitplot.corpus._base import DocumentReader  # noqa: PLC0415

        cfg = self.config
        source_str = str(source)
        is_url = source_str.startswith(("http://", "https://"))

        reader_kwargs: dict[str, Any] = {}
        if source_title:
            reader_kwargs["source_title"] = source_title
        if source_author:
            reader_kwargs["source_author"] = source_author
        if collection_id:
            reader_kwargs["collection_id"] = collection_id
        if cfg.source_type:
            reader_kwargs["source_type"] = cfg.source_type
        if cfg.default_language:
            reader_kwargs["default_language"] = cfg.default_language

        chunker = self._get_chunker()

        # ── URL handling ─────────────────────────────────────────────
        if is_url:
            return self._ingest_url(
                source_str,
                chunker=chunker,
                reader_kwargs=reader_kwargs,
            )

        # ── Archive handling ─────────────────────────────────────────
        # Archives (zip, tar, etc.) are extracted and each file inside
        # is processed individually.  However, some archive extensions
        # have a *dedicated* reader registered (e.g. ALTOReader for
        # .zip).  In that case, try the dedicated reader first — only
        # fall back to generic extraction when the reader yields zero
        # documents (meaning the archive contents didn't match the
        # reader's expected format).
        from scikitplot.corpus._archive_handler import is_archive  # noqa: PLC0415

        local_path = Path(source_str)
        if local_path.is_file() and is_archive(local_path):
            ext = local_path.suffix.lower()
            has_dedicated_reader = ext in DocumentReader._registry

            if has_dedicated_reader:
                try:
                    reader = DocumentReader.create(
                        source_str,
                        chunker=chunker,
                        **reader_kwargs,
                    )
                    documents = list(reader.get_documents())
                    if documents:
                        logger.debug(
                            "Ingested %d documents from %s via dedicated reader (%s)",
                            len(documents),
                            source_str,
                            type(reader).__name__,
                        )
                        return documents
                    logger.debug(
                        "Dedicated reader for %s yielded 0 documents; "
                        "falling back to archive extraction.",
                        source_str,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Dedicated reader for %s failed (%s); "
                        "falling back to archive extraction.",
                        source_str,
                        exc,
                    )

            return self._ingest_archive(
                local_path,
                chunker=chunker,
                reader_kwargs=reader_kwargs,
            )

        # ── Regular file ─────────────────────────────────────────────
        reader = DocumentReader.create(
            source_str,
            chunker=chunker,
            **reader_kwargs,
        )
        documents = list(reader.get_documents())
        self._n_filtered_current_build += getattr(reader, "_last_n_omitted", 0)
        logger.debug(
            "Ingested %d documents from %s",
            len(documents),
            source_str,
        )
        return documents

    def _ingest_url(
        self,
        url: str,
        *,
        chunker: Any,
        reader_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Ingest a URL source with automatic classification and download.

        Parameters
        ----------
        url : str
            HTTP/HTTPS URL.
        chunker : ChunkerBase or None
            Chunker to inject into the reader.
        reader_kwargs : dict
            Provenance keyword arguments for the reader.

        Returns
        -------
        list[CorpusDocument]
            Ingested documents.

        Notes
        -----
        **URL classification logic (two-stage):**

        Stage 1 — :func:`classify_url` (extension-based, no network):

        - ``YOUTUBE`` → :class:`YouTubeReader` (always)
        - ``GOOGLE_DRIVE`` → resolve → download → :class:`DocumentReader.create`
        - ``GITHUB_BLOB`` / ``GITHUB_RAW`` → resolve/download → :class:`DocumentReader.create`
        - ``DOWNLOADABLE`` (path has known extension) → download → :class:`DocumentReader.create`
        - ``WEB_PAGE`` (no extension in path) → Stage 2

        Stage 2 — :func:`probe_url_kind` (HEAD request, only for extensionless
        ``WEB_PAGE`` results when :attr:`BuilderConfig.probe_url_content_type`
        is ``True``):

        - ``DOWNLOADABLE`` (Content-Type != text/html) → download → :class:`DocumentReader.create`
        - ``WEB_PAGE`` (Content-Type is text/html or probe failed) → :class:`WebReader`

        This two-stage design means a single extra HEAD request correctly routes
        API endpoints like ``https://iris.who.int/.../content`` (returns
        ``application/pdf``) to :class:`PDFReader` instead of :class:`WebReader`.

        Downloaded files that are archives (ZIP, TAR) are further expanded
        via :meth:`_ingest_archive`.

        Developer note
        --------------
        Stage 2 is only triggered when ALL of:

        1. ``kind == WEB_PAGE`` after stage 1.
        2. ``config.probe_url_content_type is True``.
        3. The URL path has no recognisable file extension — checked via
           ``os.path.splitext(urllib.parse.urlparse(url).path)[1]``.

        Condition 3 avoids redundant HEAD requests for URLs that already carry
        a known extension (e.g. ``https://example.com/article.html``).
        """
        import os as _os  # noqa: PLC0415
        import urllib.parse as _up  # noqa: PLC0415

        from scikitplot.corpus._base import DocumentReader  # noqa: PLC0415
        from scikitplot.corpus._url_handler import (  # noqa: PLC0415
            URLKind,
            classify_url,
            download_url,
            probe_url_kind,
            resolve_url,
        )

        cfg = self.config
        kind = classify_url(url)
        logger.debug("URL classified as %s: %s", kind.value, url)

        # ── Stage 2: probe extensionless WEB_PAGE URLs ──────────────────────
        # Only activate when the initial classification is WEB_PAGE, the
        # config enables probing, and the URL path truly has no extension.
        # This correctly reclassifies API endpoints that serve binary files
        # (PDFs, images, audio) without a file-extension suffix in the path.
        if (
            kind == URLKind.WEB_PAGE
            and cfg.probe_url_content_type
            and not _os.path.splitext(_up.urlparse(url).path)[1]
        ):
            probed = probe_url_kind(url, timeout=cfg.probe_url_timeout)
            if probed != kind:
                logger.info(
                    "probe_url_kind reclassified %s: %s → %s",
                    url,
                    kind.value,
                    probed.value,
                )
                kind = probed

        # ── YouTube: direct transcript fetch ────────────────────────────────
        if kind == URLKind.YOUTUBE:
            from_url_kwargs = {
                k: v
                for k, v in reader_kwargs.items()
                if k
                in (
                    "source_title",
                    "source_author",
                    "collection_id",
                    "source_type",
                    "default_language",
                )
            }
            reader = DocumentReader.from_url(
                url,
                chunker=chunker,
                **from_url_kwargs,
            )
            documents = list(reader.get_documents())
            self._n_filtered_current_build += getattr(reader, "_last_n_omitted", 0)
            logger.debug(
                "Ingested %d documents from YouTube URL %s", len(documents), url
            )
            return documents

        # ── WEB_PAGE: HTML scraping ──────────────────────────────────────────
        if kind == URLKind.WEB_PAGE:
            from_url_kwargs = {
                k: v
                for k, v in reader_kwargs.items()
                if k
                in (
                    "source_title",
                    "source_author",
                    "collection_id",
                    "source_type",
                    "default_language",
                )
            }
            reader = DocumentReader.from_url(
                url,
                chunker=chunker,
                **from_url_kwargs,
            )
            documents = list(reader.get_documents())
            self._n_filtered_current_build += getattr(reader, "_last_n_omitted", 0)
            logger.debug("Ingested %d documents from web URL %s", len(documents), url)
            return documents

        # ── Downloadable file: resolve → download → create reader ────────────
        resolved = resolve_url(url, kind=kind)
        if resolved != url:
            logger.info("Resolved URL: %s → %s", url, resolved)

        temp_dir = self._get_temp_dir()
        local_path = download_url(
            resolved,
            dest_dir=temp_dir,
            max_bytes=cfg.max_download_bytes,
            timeout=cfg.download_timeout,
            max_retries=cfg.download_max_retries,
            retry_backoff=cfg.download_retry_backoff,
        )

        # If the downloaded file is an archive, apply the same
        # reader-first logic as _ingest_source: try a dedicated reader
        # (e.g. ALTOReader for .zip) before generic extraction.
        from scikitplot.corpus._archive_handler import is_archive  # noqa: PLC0415

        if is_archive(local_path):
            ext = local_path.suffix.lower()
            has_dedicated_reader = ext in DocumentReader._registry

            if has_dedicated_reader:
                try:
                    reader = DocumentReader.create(
                        local_path,
                        chunker=chunker,
                        filename_override=url,
                        **reader_kwargs,
                    )
                    documents = list(reader.get_documents())
                    if documents:
                        logger.debug(
                            "Ingested %d documents from downloaded "
                            "archive %s via dedicated reader (%s)",
                            len(documents),
                            url,
                            type(reader).__name__,
                        )
                        return documents
                    logger.debug(
                        "Dedicated reader for downloaded %s yielded "
                        "0 documents; falling back to extraction.",
                        url,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Dedicated reader for downloaded %s failed "
                        "(%s); falling back to extraction.",
                        url,
                        exc,
                    )

            return self._ingest_archive(
                local_path,
                chunker=chunker,
                reader_kwargs=reader_kwargs,
            )

        # Regular downloaded file → create reader by extension.
        # download_url already inferred the correct extension from Content-Type
        # headers (via _infer_extension_from_headers), so the file on disk has
        # the right suffix even when the URL path had none.
        reader = DocumentReader.create(
            local_path,
            chunker=chunker,
            filename_override=url,
            **reader_kwargs,
        )
        documents = list(reader.get_documents())
        self._n_filtered_current_build += getattr(reader, "_last_n_omitted", 0)
        logger.debug(
            "Ingested %d documents from downloaded %s → %s",
            len(documents),
            url,
            local_path,
        )
        return documents

    def _ingest_archive(
        self,
        archive_path: Path,
        *,
        chunker: Any,
        reader_kwargs: dict[str, Any],
    ) -> list[Any]:
        """Extract an archive and ingest each file inside.

        Parameters
        ----------
        archive_path : Path
            Path to the archive file.
        chunker : ChunkerBase or None
            Chunker to inject.
        reader_kwargs : dict
            Provenance keyword arguments.

        Returns
        -------
        list[CorpusDocument]
            Documents ingested from all files in the archive.
        """
        from scikitplot.corpus._archive_handler import (  # noqa: PLC0415
            extract_archive,
        )
        from scikitplot.corpus._base import DocumentReader  # noqa: PLC0415

        cfg = self.config
        supported = set(DocumentReader.supported_types()) or None
        temp_dir = self._get_temp_dir()
        extract_dir = temp_dir / f"archive_{archive_path.stem}"

        extracted_files = extract_archive(
            archive_path,
            extract_dir,
            supported_extensions=(frozenset(supported) if supported else None),
            max_files=cfg.max_archive_files,
            max_total_bytes=cfg.max_archive_bytes,
        )

        all_docs: list[Any] = []
        for fp in extracted_files:
            try:
                reader = DocumentReader.create(
                    fp,
                    chunker=chunker,
                    filename_override=f"{archive_path.name}/{fp.relative_to(extract_dir)}",
                    **reader_kwargs,
                )
                docs = list(reader.get_documents())
                all_docs.extend(docs)
                logger.debug(
                    "Ingested %d documents from archive member %s",
                    len(docs),
                    fp.name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to ingest archive member %s: %s",
                    fp,
                    exc,
                )

        logger.info(
            "Ingested %d documents from %d files in archive %s",
            len(all_docs),
            len(extracted_files),
            archive_path.name,
        )
        return all_docs

    # ==================================================================
    # Internal: lazy component creation
    # ==================================================================

    def _get_chunker(self) -> Any:
        """Get or create the chunker, auto-bridging if needed."""
        if self._chunker is not None:
            return self._chunker

        cfg = self.config
        chunker_spec = cfg.chunker

        # Already an instance
        if not isinstance(chunker_spec, str):
            from scikitplot.corpus._chunkers._chunker_bridge import (  # noqa: PLC0415
                bridge_chunker,
            )

            self._chunker = bridge_chunker(chunker_spec)
            return self._chunker

        # Create by name
        kwargs = cfg.chunker_kwargs

        if chunker_spec == "sentence":
            try:
                from scikitplot.corpus._chunkers import SentenceChunker  # noqa: PLC0415
                from scikitplot.corpus._chunkers._chunker_bridge import (  # noqa: PLC0415
                    SentenceChunkerBridge,
                )

                inner = SentenceChunker(**kwargs)
                self._chunker = SentenceChunkerBridge(inner)
            except ImportError:
                # Fall back to old chunker
                logger.info(
                    "New SentenceChunker not available; "
                    "using ChunkerBase sentence chunker."
                )
                self._chunker = None  # Pipeline will use default

        elif chunker_spec == "paragraph":
            try:
                from scikitplot.corpus._chunkers import (  # noqa: PLC0415
                    ParagraphChunker,
                )
                from scikitplot.corpus._chunkers._chunker_bridge import (  # noqa: PLC0415
                    ParagraphChunkerBridge,
                )

                inner = ParagraphChunker(**kwargs)
                self._chunker = ParagraphChunkerBridge(inner)
            except ImportError:
                self._chunker = None

        elif chunker_spec == "fixed_window":
            try:
                from scikitplot.corpus._chunkers import (  # noqa: PLC0415
                    FixedWindowChunker,
                )
                from scikitplot.corpus._chunkers._chunker_bridge import (  # noqa: PLC0415
                    FixedWindowChunkerBridge,
                )

                inner = FixedWindowChunker(**kwargs)
                self._chunker = FixedWindowChunkerBridge(inner)
            except ImportError:
                self._chunker = None

        elif chunker_spec == "word":
            try:
                from scikitplot.corpus._chunkers._chunker_bridge import (  # noqa: PLC0415
                    WordChunkerBridge,
                )
                from scikitplot.corpus._chunkers._word import (  # noqa: PLC0415
                    WordChunker,
                )

                inner = WordChunker(**kwargs)
                self._chunker = WordChunkerBridge(inner)
            except ImportError:
                self._chunker = None

        else:
            raise ValueError(
                f"Unknown chunker: {chunker_spec!r}. "
                f"Use 'sentence', 'paragraph', 'fixed_window', "
                f"'word', or pass an instance."
            )

        return self._chunker

    def _get_normalizer_pipeline(self) -> Any:
        """Build normalisation pipeline from existing _normalizers module."""
        if self._normalizer_pipeline is not None:
            return self._normalizer_pipeline

        from scikitplot.corpus._normalizers import (  # noqa: PLC0415
            DedupLinesNormalizer,
            HTMLStripNormalizer,
            LowercaseNormalizer,
            NormalizationPipeline,
            UnicodeNormalizer,
            WhitespaceNormalizer,
        )

        step_map = {
            "unicode": UnicodeNormalizer,
            "whitespace": WhitespaceNormalizer,
            "html_strip": HTMLStripNormalizer,
            "lowercase": LowercaseNormalizer,
            "dedup_lines": DedupLinesNormalizer,
        }

        steps = []
        for name in self.config.normalizer_steps:
            cls = step_map.get(name)
            if cls is None:
                logger.warning("Unknown normalizer step: %r", name)
                continue
            steps.append(cls())

        self._normalizer_pipeline = NormalizationPipeline(steps=steps)
        return self._normalizer_pipeline

    def _get_enricher(self) -> Any:
        """Get or create the NLP enricher."""
        if self._enricher is not None:
            return self._enricher

        from scikitplot.corpus._enrichers._nlp_enricher import (  # noqa: PLC0415
            EnricherConfig,
            NLPEnricher,
        )

        ecfg = EnricherConfig(**self.config.enricher_kwargs)
        self._enricher = NLPEnricher(config=ecfg)
        return self._enricher

    def _get_embedding_engine(self) -> Any:
        """Get or create the embedding engine."""
        if self._embedding_engine is not None:
            return self._embedding_engine

        if not self.config.embed:
            return None

        try:
            from scikitplot.corpus._embeddings import (  # noqa: PLC0415
                EmbeddingEngine,
            )

            self._embedding_engine = EmbeddingEngine(
                model_name=self.config.embedding_model,
                **self.config.embedding_kwargs,
            )
        except ImportError:
            logger.warning(
                "EmbeddingEngine not available. Embeddings will not be computed."
            )
            self._embedding_engine = None

        return self._embedding_engine

    def _embed_documents(
        self,
        documents: list[Any],
    ) -> tuple[list[Any], int]:
        """Embed documents, returning (updated_docs, n_embedded)."""
        engine = self._get_embedding_engine()
        if engine is None:
            return documents, 0

        texts = []
        for doc in documents:
            nt = getattr(doc, "normalized_text", None)
            texts.append(nt or getattr(doc, "text", ""))

        try:
            embeddings = engine.embed(texts)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding failed: %s", exc)
            return documents, 0

        out = []
        n_embedded = 0
        for doc, emb in zip(documents, embeddings):
            if emb is not None:
                out.append(doc.replace(embedding=emb))
                n_embedded += 1
            else:
                out.append(doc)

        return out, n_embedded

    def _make_embed_fn(self):
        """Create a query embedding function for adapters."""
        engine = self._get_embedding_engine()
        if engine is None:
            return None

        def fn(text: str) -> Any:
            """Embed a single query string using the configured engine."""
            embs = engine.embed([text])
            return embs[0] if embs else None

        return fn

    # ==================================================================
    # Internal: result access
    # ==================================================================

    def _get_documents(self) -> list[Any]:
        """Get documents from the last build result."""
        if self._result is None:
            raise RuntimeError("No corpus built yet. Call build() first.")
        return self._result.documents

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        """Return a concise summary of the builder state."""
        cfg = self.config
        parts = [
            f"chunker={cfg.chunker!r}",
            f"normalize={cfg.normalize}",
            f"enrich={cfg.enrich}",
            f"embed={cfg.embed}",
            f"build_index={cfg.build_index}",
        ]
        n_docs = self._result.n_documents if self._result else 0
        return f"CorpusBuilder({', '.join(parts)}, n_docs={n_docs})"
