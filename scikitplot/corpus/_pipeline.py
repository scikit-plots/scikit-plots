# scikitplot/corpus/_pipeline.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
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

from ._base import ChunkerBase, DocumentReader, FilterBase, _is_url, _MultiSourceReader
from ._enrichers._nlp_enricher import NLPEnricher
from ._normalizers._text_normalizer import TextNormalizer
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
    input_path : str
        Input source identifier (file path, URL, or batch label).
    output_path : pathlib.Path or None
        Path to the exported file, or ``None`` when no export was
        requested (``output_path=None`` in the pipeline call).
    documents : list of CorpusDocument
        All documents produced (after chunking, filtering, and optional
        embedding). Empty list if the source yielded no usable text.
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

    input_path: str
    output_path: pathlib.Path | None
    export_format: ExportFormat | None
    documents: list[CorpusDocument]
    n_read: int
    n_omitted: int
    n_embedded: int
    elapsed_seconds: float

    @property
    def n_documents(self) -> int:
        """Number of documents in the result."""
        return len(self.documents)

    def __repr__(self) -> str:
        return (
            f"PipelineResult("
            f"input_path={self.input_path!r},"
            f" output_path={self.output_path},"
            f" export_format={self.export_format},"
            f" n_documents={self.n_documents},"
            f" n_read={self.n_read},"
            f" n_omitted={self.n_omitted},"
            f" n_embedded={self.n_embedded},"
            f" elapsed_seconds={self.elapsed_seconds:.1f}s)"
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
    output_path : pathlib.Path or None, optional
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
        Signature: ``(input_path: str, n_done: int, n_total_estimate: int) → None``.
        ``n_total_estimate`` is ``-1`` when the total is unknown.
        Default: ``None``.
    normalizer : TextNormalizer or None, optional
        When provided, ``normalized_text`` is populated on every document
        after chunking/filtering and before embedding.  Insert between the
        filter and embedding stages to clean OCR noise, collapsed whitespace,
        ligatures, and other artefacts.  Default: ``None`` (skip).
    enricher : NLPEnricher or None, optional
        When provided, NLP enrichment fields (``tokens``, ``lemmas``,
        ``stems``, ``keywords``, and optional metadata such as ``pos_tags``,
        ``ner_entities``, ``sentence_count``, ``char_count``,
        ``type_token_ratio``, ``token_scores``) are populated on every
        document after normalisation and before embedding.  Supports
        200+ world languages via the ``language`` parameter of
        :class:`~._enrichers._nlp_enricher.EnricherConfig`.
        Default: ``None`` (skip).
    default_language : str or list[str] or None, optional
        ISO 639-1 language code (or list of codes, or ``None``) applied to
        all documents when the reader cannot detect language.  Accepts ISO
        639-1 two-letter codes (``"en"``, ``"ar"``), NLTK names
        (``"english"``, ``"arabic"``), lists (``["en", "ar"]``), or ``None``
        (auto-detect per document via :func:`~._custom_tokenizer.detect_script`).
        Forwarded to the reader; the enricher uses its own ``language``
        config when set.  Default: ``None``.
    reader_kwargs : dict or None, optional
        Extra keyword arguments forwarded to every reader constructed by
        this pipeline — both :meth:`~scikitplot.corpus._base.DocumentReader.create`
        (used by :meth:`run` and :meth:`run_batch`) and
        :meth:`~scikitplot.corpus._base.DocumentReader.from_url`
        (used by :meth:`run_url`).  Default: ``None``.

        **Audio / video URL transcription** — forward Whisper kwargs
        directly so ``run_url`` on an ``.mp3`` URL transcribes it::

            pipeline = CorpusPipeline(
                reader_kwargs={
                    "transcribe": True,
                    "whisper_model": "small",  # "tiny" / "base" / "medium" / "large"
                },
            )
            result = pipeline.run_url("https://archive.org/details/.../episode.mp3")

        **ZIP archive with per-extension overrides** — when the source is
        a ``.zip`` file, ``reader_kwargs`` is forwarded to
        :class:`~scikitplot.corpus._readers.ZipReader`.  Pass a nested
        ``"reader_kwargs"`` key to control individual member types::

            pipeline = CorpusPipeline(
                reader_kwargs={
                    "reader_kwargs": {
                        ".mp3": {"transcribe": True, "whisper_model": "small"},
                        ".jpg": {"backend": "easyocr"},
                    },
                },
            )
            result = pipeline.run(Path("WHO-EURO-2025.zip"))

        **Single-type files** — for a pipeline that only processes audio
        files (no ZIP), pass the kwargs flat::

            pipeline = CorpusPipeline(
                reader_kwargs={"transcribe": True, "whisper_model": "base"},
            )
            result = pipeline.run(Path("podcast.mp3"))

    Attributes
    ----------
    chunker : ChunkerBase or None
    filter_ : FilterBase or None
    embedding_engine : EmbeddingEngine or None
    output_path : pathlib.Path or None
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
    ...     output_path=Path("output/"),
    ... )
    >>> result = pipeline.run(Path("corpus.txt"))
    >>> print(result)

    Batch processing with embeddings:

    >>> from scikitplot.corpus._embeddings import EmbeddingEngine
    >>> engine = EmbeddingEngine(backend="sentence_transformers")
    >>> pipeline = CorpusPipeline(
    ...     chunker=SentenceChunker("en_core_web_sm"),
    ...     embedding_engine=engine,
    ...     output_path=Path("output/"),
    ...     export_format=ExportFormat.PARQUET,
    ... )
    >>> results = pipeline.run_batch(list(Path("corpus/").glob("*.txt")))

    URL ingestion:

    >>> result = pipeline.run_url("https://en.wikipedia.org/wiki/Python")

    Audio URL transcription via ``reader_kwargs``:

    >>> pipeline = CorpusPipeline(
    ...     reader_kwargs={"transcribe": True, "whisper_model": "small"},
    ...     output_path=Path("output/"),
    ... )
    >>> result = pipeline.run_url(
    ...     "https://archive.org/details/tale_two_cities_librivox/"
    ...     "tale_of_two_cities_01_dickens.mp3"
    ... )

    ZIP archive with per-extension kwargs:

    >>> pipeline = CorpusPipeline(
    ...     reader_kwargs={
    ...         "reader_kwargs": {
    ...             ".mp3": {"transcribe": True, "whisper_model": "small"},
    ...             ".jpg": {"backend": "easyocr"},
    ...         },
    ...     },
    ...     output_path=Path("output/"),
    ... )
    >>> result = pipeline.run(Path("WHO-EURO-2025.zip"))
    """

    def __init__(
        self,
        chunker: ChunkerBase | None = None,
        filter_: FilterBase | None = None,
        embedding_engine: Any | None = None,
        output_path: pathlib.Path | None = None,
        export_format: ExportFormat | None = ExportFormat.CSV,
        normalizer: TextNormalizer | None = None,
        enricher: NLPEnricher | None = None,
        default_language: str | list | None = None,
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
        self.output_path = (
            pathlib.Path(output_path) if output_path is not None else None
        )
        self.export_format = export_format
        self.normalizer = normalizer
        self.enricher = enricher
        self.default_language = default_language
        self.progress_callback = progress_callback
        self.reader_kwargs = reader_kwargs or {}

    # ------------------------------------------------------------------
    # Public API — single source
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str | pathlib.Path,
        *,
        output_path: pathlib.Path | None = None,
        export_format: ExportFormat | None = None,
        filename_override: str | None = None,
    ) -> PipelineResult:
        """
        Process a single source and return a :class:`PipelineResult`.

        Accepts a local file path **or** an ``http(s)://`` URL string.
        URL detection is performed before any ``pathlib.Path`` conversion,
        so passing a URL string routes correctly to the web/YouTube/audio
        reader rather than crashing with a "file not found" error.

        Parameters
        ----------
        input_path : str or pathlib.Path
            Path to a local file **or** an ``http(s)://`` URL string.
            A ``str`` that starts with ``http://`` or ``https://``
            (case-insensitive) is treated as a URL and routed through
            :meth:`~scikitplot.corpus._base.DocumentReader.from_url`;
            all other values are treated as local file paths and
            dispatched by extension via the reader registry.
        output_path : pathlib.Path or None, optional
            Explicit output file path.  When ``None``, the path is
            derived from ``output_path`` and the input stem.  If both
            are ``None``, export is skipped.
        export_format : ExportFormat or None, optional
            Override the pipeline-level ``export_format`` for this call.
        filename_override : str or None, optional
            Override the ``input_path`` label in generated documents.
            Ignored for URL sources.

        Returns
        -------
        PipelineResult
            Result summary including the document list.

        Raises
        ------
        TypeError
            If *input_path* is not a ``str`` or :class:`pathlib.Path`.
        ValueError
            If a local file path does not exist, or no reader is
            registered for the file extension.
        ValueError
            If *input_path* is a URL string and the URL is invalid or
            cannot be resolved.

        See Also
        --------
        run_batch : Process multiple sources (files and/or URLs).
        run_url : Process one or more URLs directly (legacy entry point).

        Examples
        --------
        Local file:

        >>> result = pipeline.run(Path("chapter01.txt"))
        >>> len(result.documents)
        312

        URL string — no separate ``run_url`` call needed:

        >>> result = pipeline.run("https://en.wikipedia.org/wiki/Python")
        >>> result.input_path
        'https://en.wikipedia.org/wiki/Python'
        """
        if not isinstance(input_path, (str, pathlib.Path)):
            raise TypeError(
                f"CorpusPipeline.run: input_path must be str or pathlib.Path;"
                f" got {type(input_path).__name__!r}."
            )
        return self._run_source(
            input_path,
            output_path=output_path,
            export_format=export_format,
            filename_override=filename_override,
        )

    # ------------------------------------------------------------------
    # Unified private dispatcher — used by run() and run_batch()
    # ------------------------------------------------------------------

    def _run_source(
        self,
        input_path: pathlib.Path | str,
        *,
        output_path: pathlib.Path | None = None,
        export_format: ExportFormat | None = None,
        filename_override: str | None = None,
    ) -> PipelineResult:
        """
        Dispatch a single input_path — URL or local file — to the correct reader.

        This is the single implementation backing both :meth:`run` and
        each item of :meth:`run_batch`.  It tests the raw ``input_path`` value
        with :func:`~scikitplot.corpus._base._is_url` **before** any
        ``pathlib.Path`` conversion, preventing silent URL mangling.
        ``pathlib.Path`` collapses ``https://`` → ``https:/`` on POSIX
        systems, which breaks ``_is_url`` and silently routes the URL to
        the wrong reader.

        Parameters
        ----------
        input_path : pathlib.Path or str
            A local file path or an ``http(s)://`` URL string.
        output_path : pathlib.Path or None, optional
            Explicit output file path override.
        export_format : ExportFormat or None, optional
            Override the pipeline-level ``export_format``.
        filename_override : str or None, optional
            Override the ``input_path`` label.  Silently ignored for
            URL sources (URLs have no meaningful stable local filename).

        Returns
        -------
        PipelineResult

        Notes
        -----
        **Routing rule (applied in this order):**

        1. ``_is_url(input_path)`` is ``True`` (str matching ``^https?://``)
           → URL path: pass as a raw string to
           :meth:`~scikitplot.corpus._base.DocumentReader.create`, which
           delegates to ``from_url()``.
        2. Otherwise → local-file path: convert to :class:`pathlib.Path`
           and dispatch by extension through the reader registry.

        Both routes call the same ``DocumentReader.create()`` factory so
        any future factory changes apply automatically here.
        """
        import re as _re  # noqa: PLC0415

        is_url = _is_url(input_path)
        start = timer()

        if is_url:
            source_str = str(input_path)
            source_label = source_str
            log_name = source_str[:80]
        else:
            input_path = pathlib.Path(input_path)
            source_label = str(input_path)
            log_name = input_path.name

        logger.info("CorpusPipeline._run_source: processing %s.", log_name)

        # Pass URL strings raw (not wrapped in Path) so DocumentReader.create()
        # can route them via _is_url().  Local paths are explicitly converted.
        reader = DocumentReader.create(
            source_str if is_url else pathlib.Path(input_path),
            chunker=self.chunker,
            filter_=self.filter_,
            filename_override=None if is_url else filename_override,
            default_language=self.default_language,
            **self.reader_kwargs,
        )

        documents, n_read, n_omitted = self._collect_documents(reader, source_label)

        # Normalisation stage (optional, before enrichment and embedding)
        if self.normalizer is not None and documents:
            try:
                documents = self.normalizer.normalize_documents(documents)
                logger.debug(
                    "CorpusPipeline: normalizer ran on %d documents.", len(documents)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CorpusPipeline: normalizer raised %s: %s — documents unchanged.",
                    type(exc).__name__,
                    exc,
                )

        # NLP enrichment stage (optional, before embedding)
        if self.enricher is not None and documents:
            try:
                documents = self.enricher.enrich_documents(documents)
                logger.debug(
                    "CorpusPipeline: enricher ran on %d documents.", len(documents)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CorpusPipeline: enricher raised %s: %s — documents unchanged.",
                    type(exc).__name__,
                    exc,
                )

        # URL sources pass embed input_path=None (no stable mtime for cache key).
        n_embedded = 0
        if self.embedding_engine is not None and documents:
            embed_path = None if is_url else pathlib.Path(input_path)
            documents, n_embedded = self._embed_documents(documents, embed_path)

        # Output stem: sanitised URL slug or file stem.
        if is_url:
            stem = _re.sub(r"[^\w.-]", "_", source_str)[:60]
        else:
            stem = pathlib.Path(input_path).stem

        fmt = export_format if export_format is not None else self.export_format
        resolved_output = self._resolve_output_path(stem, output_path, fmt)
        if resolved_output is not None and fmt is not None:
            self._export(documents, resolved_output, fmt)

        elapsed = timer() - start
        logger.info(
            "CorpusPipeline._run_source: %s — %d docs, %d omitted, %.1fs.",
            log_name,
            len(documents),
            n_omitted,
            elapsed,
        )

        return PipelineResult(
            input_path=source_label,
            output_path=resolved_output,
            export_format=fmt,
            documents=documents,
            n_read=n_read,
            n_omitted=n_omitted,
            n_embedded=n_embedded,
            elapsed_seconds=round(elapsed, 3),
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

        Supported URL shapes:

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
            **self.reader_kwargs,
        )

        documents, n_read, n_omitted = self._collect_documents(reader, url)

        # URL sources: no file-based cache (no stable mtime)
        n_embedded = 0
        if self.embedding_engine is not None and documents:
            documents, n_embedded = self._embed_documents(documents, input_path=None)

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
            input_path=url,
            output_path=resolved_output,
            export_format=fmt,
            documents=documents,
            n_read=n_read,
            n_omitted=n_omitted,
            n_embedded=n_embedded,
            elapsed_seconds=round(elapsed, 3),
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
        Process multiple sources sequentially.

        Each item may be a local file path **or** an ``http(s)://`` URL
        string.  Mixed lists (some paths, some URLs) are fully supported.
        Each item is dispatched through :meth:`_run_source`, which tests
        for URL strings **before** any ``pathlib.Path`` conversion so that
        URL strings are never silently mangled.

        Parameters
        ----------
        input_files : list of pathlib.Path or str
            Sources to process in order.  Each element may be:

            * a :class:`pathlib.Path` or ``str`` pointing to a local file, or
            * a ``str`` starting with ``http://`` or ``https://`` (a URL).

            Mixed lists are allowed:
            ``[Path("paper.pdf"), "https://en.wikipedia.org/wiki/Python"]``.
        stop_on_error : bool, optional
            When ``False`` (default), errors on individual sources are
            logged as warnings and processing continues.  When ``True``,
            the first error is re-raised immediately.
        export_format : ExportFormat or None, optional
            Override the pipeline-level ``export_format`` for all sources
            in this batch.

        Returns
        -------
        list of PipelineResult
            One result per successfully processed source, in input order.
            Failed sources (when ``stop_on_error=False``) are omitted
            from the list and logged at WARNING level.

        Raises
        ------
        TypeError
            If any element of *input_files* is not a ``str`` or
            :class:`pathlib.Path`.
        ValueError
            Re-raised from :meth:`_run_source` when ``stop_on_error=True``
            and a source fails.

        See Also
        --------
        run : Process a single source (file or URL).
        run_url : Process one or more URLs directly (legacy entry point).

        Examples
        --------
        Local files only (original behaviour, unchanged):

        >>> paths = list(Path("corpus/").glob("*.txt"))
        >>> results = pipeline.run_batch(paths)
        >>> total_docs = sum(r.n_documents for r in results)

        Mixed files and URLs:

        >>> results = pipeline.run_batch(
        ...     [
        ...         Path("local_report.pdf"),
        ...         "https://en.wikipedia.org/wiki/Python",
        ...         "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ...     ]
        ... )
        >>> [r.input_path for r in results]
        ['local_report.pdf', 'https://...', 'https://...']
        """
        results: list[PipelineResult] = []
        total = len(input_files)

        for idx, i_path in enumerate(input_files):
            if not isinstance(i_path, (str, pathlib.Path)):
                raise TypeError(
                    f"CorpusPipeline.run_batch: input_files[{idx}] must be"
                    f" str or pathlib.Path; got {type(i_path).__name__!r}."
                )
            # Build a human-readable label for logging without Path-wrapping URLs.
            log_label = str(i_path) if _is_url(i_path) else pathlib.Path(i_path).name
            logger.info(
                "CorpusPipeline.run_batch: [%d/%d] %s.",
                idx + 1,
                total,
                log_label,
            )
            try:
                result = self._run_source(i_path, export_format=export_format)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                if stop_on_error:
                    raise
                logger.warning(
                    "CorpusPipeline.run_batch: skipping %s — %s: %s",
                    log_label,
                    type(exc).__name__,
                    exc,
                )

            if self.progress_callback is not None:
                done = sum(r.n_documents for r in results)
                self.progress_callback(str(i_path), done, -1)

        logger.info(
            "CorpusPipeline.run_batch: processed %d/%d sources, %d total documents.",
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

        Exhausts the reader's ``get_documents()`` generator, then reads
        the ``_last_n_included`` and ``_last_n_omitted`` counters that
        each :class:`~scikitplot.corpus._base.DocumentReader` sets on
        itself after the generator finishes.

        For a :class:`~scikitplot.corpus._base._MultiSourceReader` the
        counters are aggregated by summing over all constituent sub-readers.
        Each sub-reader's counters are set during the ``yield from`` inside
        ``_MultiSourceReader.get_documents()``, so they are all populated
        by the time the top-level ``list()`` call returns.

        Parameters
        ----------
        reader : DocumentReader or _MultiSourceReader
            Initialised reader instance.
        source_label : str
            Human-readable source label for logging.

        Returns
        -------
        tuple of (list[CorpusDocument], int, int)
            ``(documents, n_read, n_omitted)`` where
            ``n_read == len(documents) + n_omitted``.

        Notes
        -----
        The ``getattr(..., fallback)`` guards are retained for forward
        compatibility: any third-party :class:`DocumentReader` subclass
        that does not set the counter attributes will default to
        ``n_included = len(documents)`` and ``n_omitted = 0``, preserving
        the pre-existing behaviour rather than crashing.
        """
        # Exhaust the generator.  Sub-reader counters are set lazily inside
        # each DocumentReader.get_documents() after all yields complete, so
        # they are fully populated once list() returns.
        documents: list[CorpusDocument] = list(reader.get_documents())

        if isinstance(reader, _MultiSourceReader):
            # Aggregate counters from every sub-reader.  Fallback to
            # len(sub-reader-docs) / 0 for any reader missing the attrs.
            n_included = sum(
                (
                    getattr(r, "_last_n_included", None)
                    # If a sub-reader never set the attribute, fall back to
                    # counting only its included docs — we can't recover omits.
                    if getattr(r, "_last_n_included", None) is not None
                    else 0
                )
                for r in reader.readers
            )
            n_omitted = sum(getattr(r, "_last_n_omitted", 0) for r in reader.readers)
            # If all sub-readers lacked counters, fall back safely.
            if n_included == 0 and n_omitted == 0:
                n_included = len(documents)
        else:
            n_included = getattr(reader, "_last_n_included", len(documents))
            n_omitted = getattr(reader, "_last_n_omitted", 0)

        n_read = n_included + n_omitted
        return documents, n_read, n_omitted

    def _embed_documents(  # noqa: D417
        self,
        documents: list[CorpusDocument],
        input_path: pathlib.Path | None,
    ) -> tuple:
        """
        Embed documents using the configured embedding engine.

        Parameters
        ----------
        documents : list of CorpusDocument
        input_path : pathlib.Path or None

        Returns
        -------
        tuple of (list[CorpusDocument], int)
            ``(embedded_documents, n_embedded)``
        """
        logger.info("CorpusPipeline: embedding %d documents.", len(documents))
        embedded = self.embedding_engine.embed_documents(
            documents,
            input_path=input_path,
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

        Priority: explicit ``output_path`` > derived from ``output_path``
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
        if self.output_path is not None and fmt is not None:
            suffix = _FORMAT_SUFFIX.get(fmt, ".csv")
            return self.output_path / f"{stem}{suffix}"
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
    input_path: pathlib.Path | str,
    output_path: pathlib.Path | str,
    *,
    chunker: ChunkerBase | None = None,
    filter_: FilterBase | None = None,
    normalizer: TextNormalizer | None = None,
    enricher: NLPEnricher | None = None,
    filename_override: str | None = None,
    export_format: ExportFormat = ExportFormat.CSV,
    default_language: str | list | None = None,
) -> PipelineResult:
    """Create and export a corpus from a single source file.

    Convenience wrapper around :class:`CorpusPipeline` for the common
    single-file, single-output use case.  Directly replaces remarx's
    ``create_corpus()`` function.

    Parameters
    ----------
    input_path : pathlib.Path or str
        Path to the input file (local) or an ``http(s)://`` URL string.
    output_path : pathlib.Path or str
        Path for the exported corpus file.
    chunker : ChunkerBase or None, optional
        Text chunker.  Default: ``None`` (one doc per raw chunk).
    filter_ : FilterBase or None, optional
        Document filter.  Default: ``None`` (:class:`DefaultFilter`).
    normalizer : TextNormalizer or None, optional
        When provided, ``normalized_text`` is populated on every document
        after chunking/filtering.  Cleans OCR noise, ligatures, and
        whitespace artefacts before embedding.  Default: ``None`` (skip).
    enricher : NLPEnricher or None, optional
        When provided, NLP fields (``tokens``, ``lemmas``, ``stems``,
        ``keywords``, and optional extended metadata) are populated on every
        document after normalisation.  Supports 200+ world languages via
        :class:`~._enrichers._nlp_enricher.EnricherConfig.language`.
        Default: ``None`` (skip).
    filename_override : str or None, optional
        Override the ``input_path`` label in generated documents.
    export_format : ExportFormat, optional
        Output format.  Default: :attr:`~._schema.ExportFormat.CSV`.
    default_language : str or list[str] or None, optional
        ISO 639-1 code(s) or NLTK language name(s) applied when the reader
        cannot detect language.  Accepts ``"en"``, ``"english"``,
        ``["en", "ar"]``, or ``None`` (auto-detect).  Default: ``None``.

    Returns
    -------
    PipelineResult
        Immutable summary including the document list, counts, timing, and
        output path.

    Examples
    --------
    Basic single-file corpus:

    >>> from pathlib import Path
    >>> from scikitplot.corpus._pipeline import create_corpus
    >>> result = create_corpus(
    ...     input_path=Path("chapter01.txt"),
    ...     output_path=Path("output/chapter01.csv"),
    ... )
    >>> len(result.documents)
    312

    With normalisation and NLP enrichment:

    >>> from scikitplot.corpus import TextNormalizer, NLPEnricher, EnricherConfig
    >>> result = create_corpus(
    ...     input_path=Path("scan.png"),
    ...     output_path=Path("output/scan.csv"),
    ...     normalizer=TextNormalizer(),
    ...     enricher=NLPEnricher(
    ...         EnricherConfig(
    ...             language="en",
    ...             keyword_extractor="tfidf",
    ...             sentence_count=True,
    ...             char_count=True,
    ...         )
    ...     ),
    ... )

    Multi-language corpus:

    >>> result = create_corpus(
    ...     input_path=Path("multilang.txt"),
    ...     output_path=Path("output/multilang.csv"),
    ...     enricher=NLPEnricher(EnricherConfig(language=["en", "ar", "hi"])),
    ... )
    """
    pipeline = CorpusPipeline(
        chunker=chunker,
        filter_=filter_,
        normalizer=normalizer,
        enricher=enricher,
        default_language=default_language,
    )
    return pipeline.run(
        input_path=pathlib.Path(input_path),
        output_path=pathlib.Path(output_path),
        export_format=export_format,
        filename_override=filename_override,
    )
