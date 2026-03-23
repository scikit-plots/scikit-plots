"""
scikitplot.corpus._pipeline
============================
High-level orchestration of the full corpus ingestion pipeline:
**source → read → chunk → filter → embed → export**.

This module provides :class:`CorpusPipeline`, which replaces the
``create.py`` entry point from remarx and extends it with:

- Multi-file batch processing (``run_batch``)
- URL-based ingestion (``run_url``)
- Optional embedding (disabled by default for speed)
- Pluggable export formats (CSV, Parquet, JSON, JSONL, pickle, etc.)
- Structured result objects with timing, counts, and output paths
- Progress hooks for long-running jobs

Original issues fixed (from remarx ``create.py``)
--------------------------------------------------
1. **Single-file only** — ``run_batch`` processes any number of files.
2. **CSV only** — all :attr:`~scikitplot.corpus._schema.ExportFormat`
   values are supported via :func:`~scikitplot.corpus._export.export_documents`.
3. **No embedding integration** — ``embedding_engine`` is an optional
   constructor parameter.
4. **CLI hard-coded to spaCy** — chunker, filter, and model are all
   parameterised.
5. **No URL support** — ``run_url`` handles any ``http(s)://`` or
   YouTube URL.
6. **No result object** — :class:`PipelineResult` carries counts,
   timings, output path, and the document list for downstream use.
7. **f-string logging** — all log calls use ``%`` formatting.

Python compatibility
--------------------
Python 3.8-3.15. ``numpy`` is required for embedding. All other
dependencies are optional lazy imports.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field  # noqa: F401
from timeit import default_timer as timer
from typing import Any, Callable, Dict, Iterator, List, Optional, Union  # noqa: F401

from ._base import ChunkerBase, DocumentReader, FilterBase
from ._schema import CorpusDocument, ExportFormat

logger = logging.getLogger(__name__)

__all__ = [
    "CorpusPipeline",
    "PipelineResult",
    "create_corpus",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """
    Immutable summary of a single pipeline run.

    Parameters
    ----------
    source : str
        Input source identifier (file path, URL, or batch label).
    documents : list of CorpusDocument
        All documents produced (after chunking, filtering, and optional
        embedding). Empty list if the source yielded no usable text.
    output_path : pathlib.Path or None
        Path to the exported file, or ``None`` when no export was
        requested (``output_path=None`` in the pipeline call).
    n_read : int
        Total raw chunks yielded by the reader before filtering.
    n_omitted : int
        Chunks dropped by the filter.
    n_embedded : int
        Documents that received an embedding vector (0 when embedding
        is disabled).
    elapsed_seconds : float
        Wall-clock time for the entire run, in seconds.
    export_format : ExportFormat or None
        Format used for export, or ``None`` when no export was done.

    Notes
    -----
    ``n_read - n_omitted == len(documents)`` is an invariant maintained
    by the pipeline.

    Examples
    --------
    >>> result.n_read
    512
    >>> result.elapsed_seconds
    3.14
    >>> len(result.documents)
    487
    """

    source: str
    documents: list[CorpusDocument]
    output_path: pathlib.Path | None
    n_read: int
    n_omitted: int
    n_embedded: int
    elapsed_seconds: float
    export_format: ExportFormat | None

    @property
    def n_documents(self) -> int:
        """Number of documents in the result."""
        return len(self.documents)

    def __repr__(self) -> str:
        return (
            f"PipelineResult("
            f"source={self.source!r},"
            f" n_documents={self.n_documents},"
            f" n_omitted={self.n_omitted},"
            f" n_embedded={self.n_embedded},"
            f" elapsed={self.elapsed_seconds:.1f}s,"
            f" output={self.output_path})"
        )


# ---------------------------------------------------------------------------
# CorpusPipeline
# ---------------------------------------------------------------------------


class CorpusPipeline:
    """
    Orchestrates the full corpus ingestion pipeline.

    Instantiate once, then call :meth:`run` (single file),
    :meth:`run_batch` (multiple files), or :meth:`run_url` (URL source)
    any number of times. The pipeline is stateless between calls; all
    configuration is set at construction time.

    Parameters
    ----------
    chunker : ChunkerBase or None, optional
        Chunker to inject into every reader. ``None`` yields one
        :class:`~scikitplot.corpus._schema.CorpusDocument` per raw chunk.
        Default: ``None``.
    filter_ : FilterBase or None, optional
        Filter applied after chunking. ``None`` uses
        :class:`~scikitplot.corpus._base.DefaultFilter`.
        Default: ``None``.
    embedding_engine : EmbeddingEngine or None, optional
        When provided, documents are embedded in batches after
        chunking/filtering. Embeddings are stored in
        :attr:`~scikitplot.corpus._schema.CorpusDocument.embedding`.
        Default: ``None`` (no embedding).
    output_dir : pathlib.Path or None, optional
        Directory where exported files are written. When ``None``,
        export is skipped unless ``output_path`` is supplied explicitly
        in a :meth:`run` call. Default: ``None``.
    export_format : ExportFormat or None, optional
        Default export format. Individual :meth:`run` calls can override.
        Default: :attr:`~scikitplot.corpus._schema.ExportFormat.CSV`.
    default_language : str or None, optional
        ISO 639-1 language code applied to all documents when the reader
        cannot detect language. Default: ``None``.
    progress_callback : callable or None, optional
        Called after each batch of documents is processed.
        Signature: ``(source: str, n_done: int, n_total_estimate: int) → None``.
        ``n_total_estimate`` is ``-1`` when the total is unknown.
        Default: ``None``.
    reader_kwargs : dict or None, optional
        Extra keyword arguments forwarded to
        :meth:`~scikitplot.corpus._base.DocumentReader.create`.
        Default: ``None``.

    Attributes
    ----------
    chunker : ChunkerBase or None
    filter_ : FilterBase or None
    embedding_engine : EmbeddingEngine or None
    output_dir : pathlib.Path or None
    export_format : ExportFormat or None
    default_language : str or None

    See Also
    --------
    scikitplot.corpus._export.export_documents : Low-level export function.
    scikitplot.corpus._embeddings.EmbeddingEngine : Embedding backend.

    Notes
    -----
    **Thread safety:** :class:`CorpusPipeline` is not thread-safe.
    Run one instance per thread, or use :meth:`run_batch` (which
    processes files sequentially, not in parallel).

    **Embedding and caching:** When ``embedding_engine`` is provided,
    embeddings are cached to disk using the source file path and mtime
    as the cache key. URL sources disable caching (no stable mtime).

    Examples
    --------
    Basic single-file run:

    >>> from pathlib import Path
    >>> from scikitplot.corpus._pipeline import CorpusPipeline
    >>> from scikitplot.corpus._chunkers import SentenceChunker
    >>> pipeline = CorpusPipeline(
    ...     chunker=SentenceChunker("en_core_web_sm"),
    ...     output_dir=Path("output/"),
    ... )
    >>> result = pipeline.run(Path("corpus.txt"))
    >>> print(result)

    Batch processing with embeddings:

    >>> from scikitplot.corpus._embeddings import EmbeddingEngine
    >>> engine = EmbeddingEngine(backend="sentence_transformers")
    >>> pipeline = CorpusPipeline(
    ...     chunker=SentenceChunker("en_core_web_sm"),
    ...     embedding_engine=engine,
    ...     output_dir=Path("output/"),
    ...     export_format=ExportFormat.PARQUET,
    ... )
    >>> results = pipeline.run_batch(list(Path("corpus/").glob("*.txt")))

    URL ingestion:

    >>> result = pipeline.run_url("https://en.wikipedia.org/wiki/Python")
    """

    def __init__(
        self,
        chunker: ChunkerBase | None = None,
        filter_: FilterBase | None = None,
        embedding_engine: Any | None = None,
        output_dir: pathlib.Path | None = None,
        export_format: ExportFormat | None = ExportFormat.CSV,
        default_language: str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        reader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Bridge new-style chunkers (SentenceChunker, ParagraphChunker,
        # FixedWindowChunker, WordChunker) to the ChunkerBase contract
        # (_base.py needs .strategy and chunk(text, metadata) → list[tuple]).
        # Already-compliant ChunkerBase subclasses are returned as-is.
        from ._chunkers._chunker_bridge import (  # noqa: PLC0415
            bridge_chunker,
        )

        self.chunker = bridge_chunker(chunker) if chunker is not None else None
        self.filter_ = filter_
        self.embedding_engine = embedding_engine
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        self.export_format = export_format
        self.default_language = default_language
        self.progress_callback = progress_callback
        self.reader_kwargs = reader_kwargs or {}

    # ------------------------------------------------------------------
    # Public API — single source
    # ------------------------------------------------------------------

    def run(
        self,
        input_file: pathlib.Path | str,
        *,
        output_path: pathlib.Path | None = None,
        export_format: ExportFormat | None = None,
        filename_override: str | None = None,
    ) -> PipelineResult:
        """
        Process a single file and return a :class:`PipelineResult`.

        Parameters
        ----------
        input_file : pathlib.Path or str
            Path to the input file.
        output_path : pathlib.Path or None, optional
            Explicit output file path. When ``None``, the path is
            derived from ``output_dir`` and the input stem. If both
            are ``None``, export is skipped.
        export_format : ExportFormat or None, optional
            Override the pipeline-level ``export_format`` for this call.
        filename_override : str or None, optional
            Override the ``source_file`` label in generated documents.

        Returns
        -------
        PipelineResult
            Result summary including the document list.

        Raises
        ------
        ValueError
            If the input file does not exist.
        ValueError
            If no reader is registered for the file extension.

        Examples
        --------
        >>> result = pipeline.run(Path("chapter01.txt"))
        >>> len(result.documents)
        312
        """
        input_path = pathlib.Path(input_file)
        start = timer()

        logger.info("CorpusPipeline.run: processing %s.", input_path.name)

        # Build reader via factory
        reader = DocumentReader.create(
            input_path,
            chunker=self.chunker,
            filter_=self.filter_,
            filename_override=filename_override,
            default_language=self.default_language,
            **self.reader_kwargs,
        )

        documents, n_read, n_omitted = self._collect_documents(reader, str(input_path))

        # Embed
        n_embedded = 0
        if self.embedding_engine is not None and documents:
            documents, n_embedded = self._embed_documents(documents, input_path)

        # Export
        fmt = export_format if export_format is not None else self.export_format
        resolved_output = self._resolve_output_path(input_path.stem, output_path, fmt)
        if resolved_output is not None and fmt is not None:
            self._export(documents, resolved_output, fmt)

        elapsed = timer() - start
        logger.info(
            "CorpusPipeline.run: %s — %d docs, %d omitted, %.1fs.",
            input_path.name,
            len(documents),
            n_omitted,
            elapsed,
        )

        return PipelineResult(
            source=str(input_path),
            documents=documents,
            output_path=resolved_output,
            n_read=n_read,
            n_omitted=n_omitted,
            n_embedded=n_embedded,
            elapsed_seconds=round(elapsed, 3),
            export_format=fmt,
        )

    def run_url(
        self,
        url: str | list[str],
        *,
        output_path: pathlib.Path | None = None,
        export_format: ExportFormat | None = None,
        stop_on_error: bool = False,
    ) -> PipelineResult | list[PipelineResult]:
        """
        Process one URL or a list of URLs.

        Accepts a single URL string or a list of URL strings.  When a list
        is passed each URL is processed independently and a parallel list of
        :class:`PipelineResult` objects is returned.  The single-URL form
        returns a single :class:`PipelineResult` (backwards compatible).

        Supported URL shapes
        --------------------
        * Single video — ``watch?v=``, ``youtu.be/``, ``/shorts/``,
          ``/embed/``, ``/live/``
        * Video + playlist context — ``watch?v=…&list=…``
          (treated as single video; ``list=`` is ignored)
        * Channel / handle page — ``@Handle``, ``@Handle/videos``,
          ``@Handle/shorts``, ``@Handle/podcasts``,
          ``/channel/UCxxx``, ``/c/Name``, ``/user/Name``
        * Pure playlist — ``/playlist?list=…``
        * Any ``http(s)://`` URL — routed to :class:`WebReader`

        Parameters
        ----------
        url : str or list of str
            One URL string or a list of URL strings.  Every string must
            start with ``http://`` or ``https://``.
        output_path : pathlib.Path or None, optional
            Explicit output file path.  Ignored when *url* is a list
            (each result derives its own path from the URL).
        export_format : ExportFormat or None, optional
            Override the pipeline-level ``export_format`` for this call.
        stop_on_error : bool, optional
            When ``True`` and *url* is a list, re-raise the first
            exception encountered instead of continuing.  Has no effect
            for single-URL calls (exceptions always propagate).

        Returns
        -------
        PipelineResult
            When *url* is a ``str``.
        list of PipelineResult
            When *url* is a ``list``.  Results are in the same order as
            *url*.  Failed URLs (when ``stop_on_error=False``) are
            omitted from the list and logged at ERROR level.

        Raises
        ------
        TypeError
            If *url* is not a ``str`` or ``list``.
        ValueError
            If any URL string does not start with ``http://`` or
            ``https://``.
        ImportError
            If ``scikitplot.corpus._readers`` has not been imported yet.

        Examples
        --------
        Single video:

        >>> result = pipeline.run_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        >>> isinstance(result, PipelineResult)
        True

        List of URLs (returns list):

        >>> results = pipeline.run_url(
        ...     [
        ...         "https://www.youtube.com/@WHO/shorts",
        ...         "https://www.youtube.com/@WHO/videos",
        ...     ]
        ... )
        >>> isinstance(results, list)
        True
        """
        # ── List form: recurse per URL, collect results ──────────────
        if isinstance(url, list):
            if not url:
                raise ValueError("run_url: url list must not be empty.")
            results: list[PipelineResult] = []
            for u in url:
                try:
                    results.append(
                        self.run_url(  # type: ignore[arg-type]
                            u,
                            output_path=None,
                            export_format=export_format,
                            stop_on_error=stop_on_error,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    if stop_on_error:
                        raise
                    logger.error(
                        "CorpusPipeline.run_url: skipping %s — %s: %s",
                        u,
                        type(exc).__name__,
                        exc,
                    )
            return results

        if not isinstance(url, str):
            raise TypeError(
                f"run_url: url must be str or list[str], got {type(url).__name__!r}."
            )

        import re  # noqa: PLC0415

        from . import _readers  # noqa: F401, PLC0415

        start = timer()
        logger.info("CorpusPipeline.run_url: processing %s.", url)

        reader = DocumentReader.from_url(
            url,
            chunker=self.chunker,
            filter_=self.filter_,
            default_language=self.default_language,
        )

        documents, n_read, n_omitted = self._collect_documents(reader, url)

        # URL sources: no file-based cache (no stable mtime)
        n_embedded = 0
        if self.embedding_engine is not None and documents:
            documents, n_embedded = self._embed_documents(documents, source_path=None)

        fmt = export_format if export_format is not None else self.export_format
        # Derive a filename from URL if no explicit path
        stem = re.sub(r"[^\w.-]", "_", url)[:60]
        resolved_output = self._resolve_output_path(stem, output_path, fmt)
        if resolved_output is not None and fmt is not None:
            self._export(documents, resolved_output, fmt)

        elapsed = timer() - start
        logger.info(
            "CorpusPipeline.run_url: %s — %d docs in %.1fs.",
            url,
            len(documents),
            elapsed,
        )

        return PipelineResult(
            source=url,
            documents=documents,
            output_path=resolved_output,
            n_read=n_read,
            n_omitted=n_omitted,
            n_embedded=n_embedded,
            elapsed_seconds=round(elapsed, 3),
            export_format=fmt,
        )

    # ------------------------------------------------------------------
    # Public API — batch
    # ------------------------------------------------------------------

    def run_batch(
        self,
        input_files: list[pathlib.Path | str],
        *,
        stop_on_error: bool = False,
        export_format: ExportFormat | None = None,
    ) -> list[PipelineResult]:
        """
        Process multiple files sequentially.

        Parameters
        ----------
        input_files : list of pathlib.Path or str
            Paths to process in order.
        stop_on_error : bool, optional
            When ``False`` (default), errors on individual files are
            logged as warnings and processing continues. When ``True``,
            the first error is re-raised immediately.
        export_format : ExportFormat or None, optional
            Override the pipeline-level ``export_format`` for all files
            in this batch.

        Returns
        -------
        list of PipelineResult
            One result per successfully processed file. Failed files
            (when ``stop_on_error=False``) are omitted from the list.

        Examples
        --------
        >>> paths = list(Path("corpus/").glob("*.txt"))
        >>> results = pipeline.run_batch(paths)
        >>> total_docs = sum(r.n_documents for r in results)
        """
        results: list[PipelineResult] = []
        total = len(input_files)

        for idx, input_file in enumerate(input_files):
            input_path = pathlib.Path(input_file)
            logger.info(
                "CorpusPipeline.run_batch: [%d/%d] %s.",
                idx + 1,
                total,
                input_path.name,
            )
            try:
                result = self.run(input_path, export_format=export_format)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                if stop_on_error:
                    raise
                logger.warning(
                    "CorpusPipeline.run_batch: skipping %s due to error: %s",
                    input_path.name,
                    exc,
                )

            if self.progress_callback is not None:
                done = sum(r.n_documents for r in results)
                self.progress_callback(str(input_path), done, -1)

        logger.info(
            "CorpusPipeline.run_batch: processed %d/%d files, %d total documents.",
            len(results),
            total,
            sum(r.n_documents for r in results),
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_documents(
        self,
        reader: DocumentReader,
        source_label: str,
    ) -> tuple:
        """
        Collect all documents from a reader, tracking n_read / n_omitted.

        Parameters
        ----------
        reader : DocumentReader
            Initialised reader instance.
        source_label : str
            Human-readable source label for logging.

        Returns
        -------
        tuple of (list[CorpusDocument], int, int)
            ``(documents, n_read, n_omitted)``
        """
        from ._base import DefaultFilter  # noqa: F401, PLC0415

        documents: list[CorpusDocument] = []
        documents = list(reader.get_documents())

        # get_documents() now stores counters as instance attributes.
        # n_included + n_omitted = total raw chunks processed.
        n_included = getattr(reader, "_last_n_included", len(documents))
        n_omitted = getattr(reader, "_last_n_omitted", 0)
        n_read = n_included + n_omitted
        return documents, n_read, n_omitted

    def _embed_documents(  # noqa: D417
        self,
        documents: list[CorpusDocument],
        source_path: pathlib.Path | None,
    ) -> tuple:
        """
        Embed documents using the configured embedding engine.

        Parameters
        ----------
        documents : list of CorpusDocument
        source_path : pathlib.Path or None

        Returns
        -------
        tuple of (list[CorpusDocument], int)
            ``(embedded_documents, n_embedded)``
        """
        logger.info("CorpusPipeline: embedding %d documents.", len(documents))
        embedded = self.embedding_engine.embed_documents(
            documents,
            source_path=source_path,
        )
        n_embedded = sum(1 for d in embedded if d.has_embedding)
        return embedded, n_embedded

    def _resolve_output_path(  # noqa: D417
        self,
        stem: str,
        explicit_path: pathlib.Path | None,
        fmt: ExportFormat | None,
    ) -> pathlib.Path | None:
        """
        Determine the output file path.

        Priority: explicit ``output_path`` > derived from ``output_dir``
        > ``None`` (no export).

        Parameters
        ----------
        stem : str
            Base name (no extension) derived from the input file.
        explicit_path : pathlib.Path or None
        fmt : ExportFormat or None

        Returns
        -------
        pathlib.Path or None
        """
        if explicit_path is not None:
            return explicit_path
        if self.output_dir is not None and fmt is not None:
            suffix = _FORMAT_SUFFIX.get(fmt, ".csv")
            return self.output_dir / f"{stem}{suffix}"
        return None

    def _export(  # noqa: D417
        self,
        documents: list[CorpusDocument],
        output_path: pathlib.Path,
        fmt: ExportFormat,
    ) -> None:
        """
        Export documents to ``output_path`` in the given format.

        Parameters
        ----------
        documents : list of CorpusDocument
        output_path : pathlib.Path
        fmt : ExportFormat
        """
        from ._export import export_documents  # noqa: PLC0415

        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_documents(documents, output_path=output_path, fmt=fmt)
        logger.info(
            "CorpusPipeline: exported %d docs to %s (%s).",
            len(documents),
            output_path,
            fmt.value,
        )


# ---------------------------------------------------------------------------
# Format → file extension mapping (used in _resolve_output_path)
# ---------------------------------------------------------------------------

_FORMAT_SUFFIX: dict[ExportFormat, str] = {
    ExportFormat.CSV: ".csv",
    ExportFormat.PARQUET: ".parquet",
    ExportFormat.JSON: ".json",
    ExportFormat.JSONL: ".jsonl",
    ExportFormat.PICKLE: ".pkl",
    ExportFormat.JOBLIB: ".joblib",
    ExportFormat.NUMPY: ".npy",
    ExportFormat.PANDAS: ".csv",
    ExportFormat.POLARS: ".parquet",
    ExportFormat.HUGGINGFACE: ".hf",
    ExportFormat.MLFLOW: ".mlflow",
}


# ---------------------------------------------------------------------------
# Convenience functions (replace remarx create.py)
# ---------------------------------------------------------------------------


def create_corpus(
    input_file: pathlib.Path | str,
    output_path: pathlib.Path | str,
    *,
    chunker: ChunkerBase | None = None,
    filter_: FilterBase | None = None,
    filename_override: str | None = None,
    export_format: ExportFormat = ExportFormat.CSV,
    default_language: str | None = None,
) -> PipelineResult:
    """
    Create and export a corpus from a single source file.

    Convenience wrapper around :class:`CorpusPipeline` for the common
    single-file, single-output use case. Directly replaces remarx's
    ``create_corpus()`` function.

    Parameters
    ----------
    input_file : pathlib.Path or str
        Path to the input file.
    output_path : pathlib.Path or str
        Path for the exported corpus file.
    chunker : ChunkerBase or None, optional
        Text chunker. Default: ``None`` (one doc per raw chunk).
    filter_ : FilterBase or None, optional
        Document filter. Default: ``None`` (:class:`DefaultFilter`).
    filename_override : str or None, optional
        Override the ``source_file`` label.
    export_format : ExportFormat, optional
        Output format. Default: :attr:`ExportFormat.CSV`.
    default_language : str or None, optional
        ISO 639-1 language code. Default: ``None``.

    Returns
    -------
    PipelineResult

    Examples
    --------
    >>> from pathlib import Path
    >>> from scikitplot.corpus._pipeline import create_corpus
    >>> result = create_corpus(
    ...     input_file=Path("chapter01.txt"),
    ...     output_path=Path("output/chapter01.csv"),
    ... )
    >>> len(result.documents)
    312
    """
    pipeline = CorpusPipeline(
        chunker=chunker,
        filter_=filter_,
        default_language=default_language,
    )
    return pipeline.run(
        input_file=pathlib.Path(input_file),
        output_path=pathlib.Path(output_path),
        export_format=export_format,
        filename_override=filename_override,
    )
