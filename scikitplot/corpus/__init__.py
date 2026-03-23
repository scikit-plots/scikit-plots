"""
scikitplot.corpus
==================
A production-grade document corpus ingestion, chunking, filtering,
embedding, and export pipeline for NLP and ML workflows.

This package is a ground-up rewrite of the ``remarx.sentence.corpus``
module, preserving all proven design patterns while resolving every known
correctness, robustness, and maintainability issue identified during the
migration audit.

Quick start
-----------
Single file, no embedding:

>>> from pathlib import Path
>>> from scikitplot.corpus import CorpusPipeline, ParagraphChunker
>>> pipeline = CorpusPipeline(chunker=ParagraphChunker())
>>> result = pipeline.run(Path("article.txt"))
>>> print(f"{result.n_documents} chunks from {result.source}")

Batch processing with sentence chunking:

>>> from scikitplot.corpus import CorpusPipeline, SentenceChunker, ExportFormat
>>> pipeline = CorpusPipeline(
...     chunker=SentenceChunker("en_core_web_sm"),
...     output_dir=Path("output/"),
...     export_format=ExportFormat.PARQUET,
... )
>>> results = pipeline.run_batch(list(Path("corpus/").glob("*.txt")))

URL ingestion:

>>> # https://archive.org/download/WHO-documents
>>> # https://www.who.int/europe/news/item/...
>>> result = pipeline.run_url("https://en.wikipedia.org/wiki/Python")

YouTube transcript:

>>> result = pipeline.run_url("https://www.youtube.com/watch?v=rwPISgZcYIk")

Image OCR:

>>> reader = DocumentReader.create(Path("scan.png"))
>>> docs = list(reader.get_documents())

Video transcription (subtitle-first):

>>> reader = DocumentReader.create(Path("lecture.mp4"))
>>> docs = list(reader.get_documents())

With embeddings:

>>> from scikitplot.corpus import EmbeddingEngine
>>> engine = EmbeddingEngine(backend="sentence_transformers")
>>> pipeline = CorpusPipeline(
...     chunker=ParagraphChunker(),
...     embedding_engine=engine,
... )
>>> result = pipeline.run(Path("article.txt"))
>>> result.documents[0].has_embedding
True

Convenience function (direct replacement for remarx ``create_corpus``):

>>> from scikitplot.corpus import create_corpus
>>> result = create_corpus(
...     input_file=Path("chapter01.txt"),
...     output_path=Path("output/chapter01.csv"),
... )

>>> from scikitplot.corpus import CorpusBuilder, BuilderConfig
>>> builder = CorpusBuilder(
...     BuilderConfig(
...         chunker="paragraph",
...         normalize=True,
...         enrich=True,
...         embed=True,
...         build_index=True,
...     )
... )
>>> result = builder.build("./data/")
>>> results = builder.search("quantum computing")
>>> lc_docs = builder.to_langchain()
>>> mcp_response = builder.to_mcp_tool_result("quantum computing")

Package structure
-----------------
``scikitplot.corpus._schema``
    Core data types: CorpusDocument, SectionType, ChunkingStrategy,
    ExportFormat, SourceType, MatchMode.

``scikitplot.corpus._base``
    Abstract bases: DocumentReader, ChunkerBase, FilterBase, DefaultFilter.

``scikitplot.corpus._chunkers``
    WordChunker, WordChunkerConfig, SentenceChunker, SentenceChunkerConfig,
    ParagraphChunker, ParagraphChunkerConfig, FixedWindowChunker, FixedWindowChunkerConfig.

``scikitplot.corpus._readers``
    TextReader, MarkdownReader, ReSTReader, XMLReader, TEIReader, AudioReader,
    ALTOReader, PDFReader, ImageReader, VideoReader, WebReader, YouTubeReader.

``scikitplot.corpus._embeddings``
    EmbeddingEngine -- multi-backend embedding with disk cache.

``scikitplot.corpus._pipeline``
    CorpusPipeline, PipelineResult, create_corpus.

``scikitplot.corpus._export``
    export_documents, load_documents.
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _adapters,
    _archive_handler,
    _base,
    _chunkers,
    _corpus_builder,
    _embeddings,
    _enrichers,
    _export,
    _metadata,
    _normalizers,
    _pipeline,
    _readers,  # Readers -- import triggers registry population for all 24 extensions
    _registry,
    _schema,
    _similarity,
    _sources,
    _storage,
    _url_handler,
)
from ._adapters import *  # noqa: F403  # --- Adapters (LangChain / LangGraph / MCP / HuggingFace / RAG) ---
from ._archive_handler import *  # noqa: F403  # --- Archive handling (zip / tar extraction) ---
from ._base import *  # noqa: F403  # Base classes
from ._chunkers import *  # noqa: F403
from ._corpus_builder import *  # noqa: F403  # --- Unified builder (the user-friendly orchestration API) ---
from ._embeddings import *  # noqa: F403  # Embeddings by sentence-transformer
from ._enrichers import *  # noqa: F403  # --- NLP enricher ---
from ._export import *  # noqa: F403
from ._metadata import *  # noqa: F403
from ._normalizers import *  # noqa: F403  # --- Text normaliser ---
from ._pipeline import *  # noqa: F403
from ._readers import *  # noqa: F403
from ._registry import *  # noqa: F403  # Registry
from ._schema import *  # noqa: F403  # Schema -- always first, zero optional dependencies
from ._similarity import *  # noqa: F403  # --- Similarity index ---
from ._sources import *  # noqa: F403  # Sources
from ._storage import *  # noqa: F403  # Storage
from ._url_handler import *  # noqa: F403  # --- URL handling (classification, resolution, secure download) ---

__all__ = []
__all__ += _adapters.__all__
__all__ += _archive_handler.__all__
__all__ += _base.__all__
__all__ += _chunkers.__all__
__all__ += _corpus_builder.__all__
__all__ += _embeddings.__all__
__all__ += _enrichers.__all__
__all__ += _export.__all__
__all__ += _metadata.__all__
__all__ += _normalizers.__all__
__all__ += _pipeline.__all__
__all__ += _readers.__all__
__all__ += _registry.__all__
__all__ += _schema.__all__
__all__ += _similarity.__all__
__all__ += _sources.__all__
__all__ += _storage.__all__
__all__ += _url_handler.__all__
