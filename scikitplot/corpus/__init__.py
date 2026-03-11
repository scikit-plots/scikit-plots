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

>>> result = pipeline.run_url("https://en.wikipedia.org/wiki/Python")

YouTube transcript:

>>> result = pipeline.run_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

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
    SentenceChunker, ParagraphChunker, FixedWindowChunker.

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

# Readers -- import triggers registry population for all 24 extensions
import scikitplot.corpus._readers  # noqa: F401

# --- Adapters (LangChain / LangGraph / MCP / HuggingFace / RAG) ---
from scikitplot.corpus._adapters import (
    LangChainCorpusRetriever,
    MCPCorpusServer,
    to_huggingface_dataset,
    to_jsonl,
    to_langchain_documents,
    to_langgraph_state,
    to_mcp_resources,
    to_mcp_tool_result,
    to_rag_tuples,
)

# Base classes
from scikitplot.corpus._base import (
    ChunkerBase,
    DefaultFilter,
    DocumentReader,
    FilterBase,
)

# Chunkers
from scikitplot.corpus._chunkers import (
    ChunkerBridge,
    FixedWindowChunker,
    ParagraphChunker,
    SentenceChunker,
    WordChunker,
    bridge_chunker,
)

# --- Unified builder (the user-friendly orchestration API) ---
from scikitplot.corpus._corpus_builder import (
    BuilderConfig,
    BuildResult,
    CorpusBuilder,
)

# Embeddings by sentence-transformer
from scikitplot.corpus._embeddings import (
    DEFAULT_MODEL,
    EmbeddingEngine,
    EmbedFn,
)

# --- NLP enricher ---
from scikitplot.corpus._enrichers import (
    EnricherConfig,
    NLPEnricher,
)

# Export
from scikitplot.corpus._export import (
    export_documents,
    load_documents,
)

# Metadata
from scikitplot.corpus._metadata import (
    CollectionManifest,
    CorpusStats,
)

# --- Text normaliser ---
from scikitplot.corpus._normalizers import (
    DedupLinesNormalizer,
    HTMLStripNormalizer,
    LanguageDetectionNormalizer,
    LowercaseNormalizer,
    NormalizationPipeline,
    NormalizerBase,
    NormalizerConfig,
    TextNormalizer,
    UnicodeNormalizer,
    WhitespaceNormalizer,
    normalize_text,
)

# Pipeline
from scikitplot.corpus._pipeline import (
    CorpusPipeline,
    PipelineResult,
    create_corpus,
)
from scikitplot.corpus._readers import (
    ALTOReader,
    AudioReader,
    ImageReader,
    MarkdownReader,
    PDFReader,
    ReSTReader,
    TEIReader,
    TextReader,
    VideoReader,
    WebReader,
    XMLReader,
    YouTubeReader,
)

# Registry
from scikitplot.corpus._registry import (
    ComponentRegistry,
    registry,
)

# Schema -- always first, zero optional dependencies
from scikitplot.corpus._schema import (
    ChunkingStrategy,
    CorpusDocument,
    ExportFormat,
    MatchMode,
    SectionType,
    SourceType,
)

# --- Similarity index ---
from scikitplot.corpus._similarity import (
    SearchConfig,
    SearchResult,
    SimilarityIndex,
)

# Sources
from scikitplot.corpus._sources import (
    CorpusSource,
    SourceEntry,
    SourceKind,
)

# Storage
from scikitplot.corpus._storage import (
    InMemoryStorage,
    JSONLStorage,
    SQLiteStorage,
    StorageBase,
    StorageQuery,
)

__all__ = [  # noqa: RUF022
    # Adapters
    "LangChainCorpusRetriever",
    "MCPCorpusServer",
    "to_huggingface_dataset",
    "to_jsonl",
    "to_langchain_documents",
    "to_langgraph_state",
    "to_mcp_resources",
    "to_mcp_tool_result",
    "to_rag_tuples",
    # Base
    "DocumentReader",
    "ChunkerBase",
    "FilterBase",
    "DefaultFilter",
    # Chunkers
    "WordChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "FixedWindowChunker",
    # Chunker bridge
    "ChunkerBridge",
    "bridge_chunker",
    # Builder
    "BuilderConfig",
    "BuildResult",
    "CorpusBuilder",
    # Embeddings
    "EmbeddingEngine",
    "EmbedFn",
    "DEFAULT_MODEL",
    # Enricher
    "EnricherConfig",
    "NLPEnricher",
    # Export
    "export_documents",
    "load_documents",
    # Metadata
    "CollectionManifest",
    "CorpusStats",
    # Normalizers
    "NormalizerBase",
    "UnicodeNormalizer",
    "WhitespaceNormalizer",
    "HTMLStripNormalizer",
    "LowercaseNormalizer",
    "DedupLinesNormalizer",
    "LanguageDetectionNormalizer",
    "NormalizationPipeline",
    "NormalizerConfig",
    "TextNormalizer",
    "normalize_text",
    # Pipeline
    "CorpusPipeline",
    "PipelineResult",
    "create_corpus",
    # Readers
    "TextReader",
    "MarkdownReader",
    "ReSTReader",
    "XMLReader",
    "TEIReader",
    "ALTOReader",
    "PDFReader",
    "ImageReader",
    "VideoReader",
    "AudioReader",
    "WebReader",
    "YouTubeReader",
    # Registry
    "ComponentRegistry",
    "registry",
    # Schema
    "CorpusDocument",
    "SectionType",
    "ChunkingStrategy",
    "ExportFormat",
    "SourceType",
    "MatchMode",
    # Similarity
    "SearchConfig",
    "SearchResult",
    "SimilarityIndex",
    # Sources
    "CorpusSource",
    "SourceEntry",
    "SourceKind",
    # Storage
    "StorageBase",
    "InMemoryStorage",
    "JSONLStorage",
    "SQLiteStorage",
    "StorageQuery",
]
