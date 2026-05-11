# scikitplot/corpus/_types.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Core type contracts for the corpus submodule.

This module is the single source of truth for every data structure,
protocol, enumeration, and type alias used across:

  - Document ingestion   (``_sources``)
  - Chunking             (``_chunkers``)
  - Normalisation        (``_normalizers``)
  - Storage / IO         (``_storage``)
  - Metadata             (``_metadata``)
  - Pipeline             (``_pipeline``)
  - Registry             (``_registry``)

Design principles:

* All public types are ``dataclass(frozen=True)`` or ``Protocol`` — no
  mutable shared state.
* Every field has an explicit type annotation and default where safe.
* ``ChunkerConfig``, ``SourceConfig``, ``NormalizerConfig``, and
  ``StorageConfig`` are abstract base dataclasses — concrete subclasses
  in their respective submodules inherit from them.
* No circular imports: this module imports only from the standard library.

Python compatibility:

Python 3.8-3.15. No external dependencies.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import hashlib
import types
import uuid
from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path  # noqa: F401
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    Final,
    Iterator,
    Mapping,
    Protocol,
    runtime_checkable,
)

# if TYPE_CHECKING:
#     pass  # reserved for forward-reference stubs

__all__: Final[list[str]] = [  # noqa: RUF022
    # Aliases
    "MetadataDict",
    "CharOffset",
    "TokenId",
    "BowVector",
    "EmbeddingVector",
    # Enumerations
    "DocumentStatus",
    "ContentType",
    "ChunkStrategy",
    "StorageBackend",
    "NormalizerType",
    # Core containers
    "Chunk",
    "ChunkResult",
    "Document",
    # Abstract configs
    "ChunkerConfig",
    "SourceConfig",
    "NormalizerConfig",
    "StorageConfig",
    # Pipeline
    "PipelineStep",
    "PipelineConfig",
    "PipelineResult",
    # Embedding / retrieval
    "EmbeddedChunk",
    "RetrievalQuery",
    "RetrievalResult",
    # Storage
    "CorpusRecord",
    # Protocols
    "ChunkerProtocol",
    "NormalizerProtocol",
    "SourceProtocol",
    "StorageProtocol",
    # Registry
    "ChunkerRegistration",
    # Validation
    "ValidationError",
    "ValidationResult",
    # LLM training
    "TrainingExample",
    "TrainingDataset",
    # MCP
    "MCPToolInput",
    "MCPToolResult",
    # Multilang / preprocessing (Layer 0–3)  # noqa: RUF003
    "SemantemeInfo",
    "PreprocessingStep",
    "PreprocessingTrace",
    "MultilangChunkMeta",
]


# ===========================================================================
# Section 1 — Primitive type aliases
# ===========================================================================

#: Arbitrary JSON-serialisable metadata dictionary.
MetadataDict = dict[str, Any]

#: Zero-based character offset within a source document.
CharOffset = int

#: Integer identifier for a token inside a vocabulary.
TokenId = int

#: Sparse Bag-of-Words vector: list of (token_id, count) pairs.
BowVector = list[tuple[TokenId, int]]

#: Dense embedding vector produced by an encoder.
EmbeddingVector = list[float]


# ===========================================================================
# Section 2 — Enumerations
# ===========================================================================


@unique
class DocumentStatus(str, Enum):
    """Lifecycle state of a :class:`Document` in the corpus pipeline.

    Values
    ------
    PENDING
        Ingested but not yet processed.
    PROCESSING
        Currently being chunked / normalised.
    READY
        All pipeline stages completed successfully.
    FAILED
        A pipeline stage raised an unrecoverable error.
    ARCHIVED
        Document retained for audit but excluded from active retrieval.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


@unique
class ContentType(str, Enum):
    """MIME-style content type for raw document payloads.

    Values
    ------
    PLAIN_TEXT
        UTF-8 encoded plain text.
    MARKDOWN
        Markdown-formatted text.
    HTML
        HTML document (tags may or may not be stripped upstream).
    PDF
        PDF binary — must be converted before chunking.
    DOCX
        Microsoft Word document.
    JSON
        Structured JSON payload.
    JSONL
        Newline-delimited JSON (one record per line).
    CSV
        Comma-separated values.
    CODE
        Source code (language unspecified).
    UNKNOWN
        Content type could not be determined.
    """

    PLAIN_TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    JSON = "application/json"
    JSONL = "application/jsonl"
    CSV = "text/csv"
    CODE = "text/x-code"
    UNKNOWN = "application/octet-stream"


@unique
class ChunkStrategy(str, Enum):
    """Registered chunking strategy identifiers.

    These strings are the canonical keys used by :class:`ChunkerRegistry`
    to look up :class:`ChunkerProtocol` implementations.

    Values
    ------
    SENTENCE
        Sentence-boundary splitting.
    PARAGRAPH
        Blank-line paragraph splitting.
    FIXED_WINDOW
        Fixed-size sliding window.
    WORD
        Word-level tokenisation / normalisation.
    SEMANTIC
        Embedding-based semantic segmentation (future).
    RECURSIVE
        Recursive character splitter (LangChain-style, future).
    CUSTOM
        User-registered strategy not in the standard set.
    """

    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    FIXED_WINDOW = "fixed_window"
    WORD = "word"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    CUSTOM = "custom"


@unique
class StorageBackend(str, Enum):
    """Storage backend identifiers for the corpus store.

    Values
    ------
    MEMORY
        In-process dict store (testing / prototyping only).
    SQLITE
        SQLite file-backed store.
    POSTGRES
        PostgreSQL database.
    CHROMA
        ChromaDB vector store.
    QDRANT
        Qdrant vector store.
    WEAVIATE
        Weaviate vector store.
    PINECONE
        Pinecone managed vector store.
    FILESYSTEM
        Flat-file JSON/JSONL dump on disk.
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    CHROMA = "chroma"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    FILESYSTEM = "filesystem"


@unique
class NormalizerType(str, Enum):
    """Normalisation step identifiers for pipeline ordering.

    Values
    ------
    UNICODE
        Unicode NFC/NFD/NFKC/NFKD normalisation.
    WHITESPACE
        Collapse and strip whitespace.
    HTML_STRIP
        Remove HTML tags.
    LOWERCASE
        Convert to lowercase.
    DEDUP_LINES
        Remove duplicate lines.
    LANGUAGE_DETECT
        Detect and tag language (ISO 639-1).
    PII_REDACT
        Redact personally identifiable information.
    CUSTOM
        User-defined normaliser.
    """

    UNICODE = "unicode"
    WHITESPACE = "whitespace"
    HTML_STRIP = "html_strip"
    LOWERCASE = "lowercase"
    DEDUP_LINES = "dedup_lines"
    LANGUAGE_DETECT = "language_detect"
    PII_REDACT = "pii_redact"
    CUSTOM = "custom"


# ===========================================================================
# Section 3 — Core immutable data containers
# ===========================================================================


@dataclass(frozen=True)  # frozen=True → makes dataclass hashable
class Chunk:
    """A single unit of text produced by a chunker.

    Parameters
    ----------
    text : str
        The chunk's text content (possibly including overlap context).
    start_char : int
        Zero-based character offset of this chunk's start in the source
        document.  Set to ``0`` when offsets are disabled.
    end_char : int
        Zero-based exclusive character offset of this chunk's end.
        Set to ``0`` when offsets are disabled.
    metadata : MetadataDict
        Arbitrary per-chunk key/value pairs (e.g. ``chunk_index``,
        ``doc_id``, ``backend``, ``tokens``).  Always a dict — never
        ``None``.

    Notes
    -----
    * ``text`` must be a non-empty string.  Validation is the chunker's
      responsibility; :class:`Chunk` itself does not re-validate to keep
      it a pure value object.
    * ``metadata`` is stored as-is.  Callers must not mutate the dict
      after construction (frozen dataclass contract).
    """

    text: str
    start_char: CharOffset
    end_char: CharOffset
    metadata: MetadataDict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        # MappingProxyType → immutable dictionary wrapper
        object.__setattr__(
            self, "metadata", types.MappingProxyType(dict(self.metadata))
        )

    def __hash__(self) -> int:
        # Explicit, deterministic, stable
        return hash((self.text, self.start_char, self.end_char))

    def char_length(self) -> int:
        """Return the character span length of this chunk.

        Returns
        -------
        int
            ``end_char - start_char``.  Returns ``0`` when offsets are
            disabled (both fields are ``0``).
        """
        return self.end_char - self.start_char

    def content_hash(self) -> str:
        """Return a stable SHA-256 hex digest of ``text``.

        Useful as a deduplication key in storage backends.

        Returns
        -------
        str
            64-character lowercase hex string.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def with_metadata(self, **kwargs: Any) -> Chunk:
        """Return a new :class:`Chunk` with additional metadata keys.

        Parameters
        ----------
        **kwargs
            Key/value pairs to merge into a copy of ``metadata``.

        Returns
        -------
        Chunk
            New frozen instance with merged metadata.
        """
        merged = {**self.metadata, **kwargs}
        return Chunk(
            text=self.text,
            start_char=self.start_char,
            end_char=self.end_char,
            metadata=merged,
        )


@dataclass(frozen=True)
class ChunkResult:
    """Container returned by every :class:`ChunkerProtocol` implementation.

    Parameters
    ----------
    chunks : list[Chunk]
        Ordered list of chunks produced from a single document.
        May be empty if no content passes filtering.
    metadata : MetadataDict
        Aggregate metadata for the entire chunking run (e.g.
        ``chunker``, ``backend``, ``total_chunks``, ``doc_id``).

    Notes
    -----
    ``chunks`` is a plain list stored by value.  Because
    ``frozen=True`` prevents re-assignment of the ``chunks`` attribute,
    the list itself must not be mutated after construction.
    """

    chunks: list[Chunk]
    metadata: MetadataDict = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return ``True`` if no chunks were produced.

        Returns
        -------
        bool
        """
        return len(self.chunks) == 0

    def total_chars(self) -> int:
        """Return the total character count across all chunks.

        Returns
        -------
        int
            Sum of ``len(c.text)`` for all chunks.
        """
        return sum(len(c.text) for c in self.chunks)

    def texts(self) -> list[str]:
        """Return the text of every chunk as a plain list.

        Returns
        -------
        list[str]
        """
        return [c.text for c in self.chunks]

    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield chunks one at a time.

        Yields
        ------
        Chunk
        """
        yield from self.chunks


# ===========================================================================
# Section 4 — Document type
# ===========================================================================


@dataclass(frozen=True)
class Document:
    """A raw source document entering the corpus pipeline.

    Parameters
    ----------
    doc_id : str
        Globally unique document identifier.  Auto-generated (UUID4) if
        not supplied explicitly — callers should prefer explicit IDs for
        reproducible pipelines.
    text : str
        Full raw document text.  Must not be empty.
    content_type : ContentType
        MIME-style type of the document payload.
    input_path : str or None
        Human-readable origin descriptor (file path, URL, database key).
    status : DocumentStatus
        Current lifecycle state.  Defaults to ``PENDING``.
    metadata : MetadataDict
        Arbitrary document-level key/value pairs (language, author,
        timestamp, tags, …).
    checksum : str or None
        SHA-256 hex digest of ``text``.  Set automatically via
        :meth:`with_checksum` — not computed at construction to keep the
        constructor pure.

    Examples
    --------
    >>> doc = Document(
    ...     doc_id="d1", text="Hello world.", content_type=ContentType.PLAIN_TEXT
    ... )
    >>> doc.status
    <DocumentStatus.PENDING: 'pending'>
    """

    doc_id: str
    text: str
    content_type: ContentType = ContentType.PLAIN_TEXT
    input_path: str | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: MetadataDict = field(default_factory=dict)
    checksum: str | None = None

    @staticmethod
    def new(
        text: str,
        content_type: ContentType = ContentType.PLAIN_TEXT,
        input_path: str | None = None,
        metadata: MetadataDict | None = None,
    ) -> Document:
        """Construct a :class:`Document` with an auto-generated UUID4 ID.

        Parameters
        ----------
        text : str
            Raw document text.
        content_type : ContentType
            MIME type of the content.
        input_path : str, optional
            Origin descriptor.
        metadata : MetadataDict, optional
            Extra metadata.

        Returns
        -------
        Document
            New instance with ``doc_id=str(uuid.uuid4())``.

        Raises
        ------
        ValueError
            If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Document text must not be empty.")
        return Document(
            doc_id=str(uuid.uuid4()),
            text=text,
            content_type=content_type,
            input_path=input_path,
            metadata=metadata or {},
        )

    def with_checksum(self) -> Document:
        """Return a copy of this document with ``checksum`` populated.

        Returns
        -------
        Document
            New frozen instance with SHA-256 checksum set.
        """
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return Document(
            doc_id=self.doc_id,
            text=self.text,
            content_type=self.content_type,
            input_path=self.input_path,
            status=self.status,
            metadata=self.metadata,
            checksum=digest,
        )

    def with_status(self, status: DocumentStatus) -> Document:
        """Return a copy of this document with an updated status.

        Parameters
        ----------
        status : DocumentStatus
            New lifecycle state.

        Returns
        -------
        Document
            New frozen instance with updated status.
        """
        return Document(
            doc_id=self.doc_id,
            text=self.text,
            content_type=self.content_type,
            input_path=self.input_path,
            status=status,
            metadata=self.metadata,
            checksum=self.checksum,
        )

    def char_count(self) -> int:
        """Return the character length of the document text.

        Returns
        -------
        int
        """
        return len(self.text)


# ===========================================================================
# Section 5 — Abstract config base classes
# ===========================================================================


@dataclass(frozen=True)
class ChunkerConfig:
    """Abstract base configuration for all chunker implementations.

    Concrete subclasses (e.g. ``SentenceChunkerConfig``) must inherit
    from this class.  No fields are defined here; this exists purely
    as a type-safe base for ``isinstance`` checks and registry typing.

    Notes
    -----
    All concrete configs must remain ``frozen=True`` dataclasses.
    Mutable config objects create non-deterministic pipeline behaviour.
    """


@dataclass(frozen=True)
class SourceConfig:
    """Abstract base configuration for document source implementations.

    Concrete subclasses (e.g. ``FileSourceConfig``, ``S3SourceConfig``)
    inherit from this class.
    """


@dataclass(frozen=True)
class NormalizerConfig:
    """Abstract base configuration for text normaliser implementations.

    Parameters
    ----------
    normalizer_type : NormalizerType
        The type of normalisation this config applies to.
    enabled : bool
        When ``False``, the normaliser is skipped in the pipeline.
    """

    normalizer_type: NormalizerType = NormalizerType.CUSTOM
    enabled: bool = True


@dataclass(frozen=True)
class StorageConfig:
    """Abstract base configuration for storage backend implementations.

    Parameters
    ----------
    backend : StorageBackend
        Target storage system.
    collection_name : str
        Name of the collection / table / index to write to.
    """

    backend: StorageBackend = StorageBackend.MEMORY
    collection_name: str = "corpus"


# ===========================================================================
# Section 6 — Pipeline types
# ===========================================================================


@dataclass(frozen=True)
class PipelineStep:
    """Descriptor for a single step in a :class:`PipelineConfig`.

    Parameters
    ----------
    name : str
        Human-readable step name used in logs and error messages.
    step_type : str
        One of ``"source"``, ``"normalizer"``, ``"chunker"``,
        ``"embedder"``, ``"storage"``.
    config : ChunkerConfig | NormalizerConfig | SourceConfig | StorageConfig
        Configuration object for this step.  Must be an instance of the
        appropriate abstract config base.
    enabled : bool
        When ``False``, this step is skipped during pipeline execution.
    """

    name: str
    step_type: str
    config: ChunkerConfig | NormalizerConfig | SourceConfig | StorageConfig
    enabled: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration composed of ordered steps.

    Parameters
    ----------
    pipeline_id : str
        Unique identifier for this pipeline configuration.
    steps : list[PipelineStep]
        Ordered sequence of pipeline steps.  Steps are executed in
        list order.
    description : str
        Human-readable description of the pipeline's purpose.
    metadata : MetadataDict
        Arbitrary pipeline-level metadata (version, author, …).

    Notes
    -----
    A valid pipeline must contain at least one step.  Validation is
    performed by the pipeline executor, not here.
    """

    pipeline_id: str
    steps: list[PipelineStep]
    description: str = ""
    metadata: MetadataDict = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """Result returned after a complete pipeline run.

    Parameters
    ----------
    pipeline_id : str
        Identifier of the pipeline that produced this result.
    doc_id : str
        Identifier of the document that was processed.
    chunk_results : list[ChunkResult]
        One :class:`ChunkResult` per chunker step in the pipeline.
    status : DocumentStatus
        Final document status after the run.
    error : str or None
        Error message if ``status == FAILED``.  ``None`` on success.
    metadata : MetadataDict
        Aggregate run metadata (duration_ms, step_timings, …).
    """

    pipeline_id: str
    doc_id: str
    chunk_results: list[ChunkResult]
    status: DocumentStatus
    error: str | None = None
    metadata: MetadataDict = field(default_factory=dict)

    def succeeded(self) -> bool:
        """Return ``True`` if the pipeline completed without error.

        Returns
        -------
        bool
        """
        return self.status == DocumentStatus.READY

    def all_chunks(self) -> list[Chunk]:
        """Flatten all :class:`ChunkResult` objects into a single list.

        Returns
        -------
        list[Chunk]
            All chunks from all chunker steps, in order.
        """
        return [chunk for result in self.chunk_results for chunk in result.chunks]


# ===========================================================================
# Section 7 — Embedding and retrieval types
# ===========================================================================


@dataclass(frozen=True)
class EmbeddedChunk:
    """A :class:`Chunk` decorated with a dense embedding vector.

    Parameters
    ----------
    chunk : Chunk
        The original text chunk.
    embedding : EmbeddingVector
        Dense float vector produced by an encoder model.
    model_name : str
        Name / identifier of the encoder model used.
    embedding_dim : int
        Dimensionality of the embedding vector.

    Notes
    -----
    ``embedding_dim`` is stored explicitly so consumers can validate
    without computing ``len(embedding)`` repeatedly.
    """

    chunk: Chunk
    embedding: EmbeddingVector
    model_name: str
    embedding_dim: int

    def validate_dimension(self) -> None:
        """Raise if ``len(embedding) != embedding_dim``.

        Raises
        ------
        ValueError
            If the embedding length does not match ``embedding_dim``.
        """
        if len(self.embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(self.embedding)}."
            )


@dataclass(frozen=True)
class RetrievalQuery:
    """A query submitted to the corpus retrieval layer.

    Parameters
    ----------
    query_id : str
        Unique query identifier.
    text : str
        Raw query text.
    top_k : int
        Maximum number of results to return.
    filters : MetadataDict
        Metadata filter predicates (e.g. ``{"doc_id": "d1"}``).
    embedding : EmbeddingVector or None
        Pre-computed query embedding.  When ``None``, the retrieval
        backend is responsible for encoding the query text.
    """

    query_id: str
    text: str
    top_k: int = 10
    filters: MetadataDict = field(default_factory=dict)
    embedding: EmbeddingVector | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """A single result returned from the corpus retrieval layer.

    Parameters
    ----------
    chunk : Chunk
        The matched text chunk.
    score : float
        Similarity / relevance score (higher is better for cosine).
    rank : int
        Zero-based rank within the result set.
    retrieval_metadata : MetadataDict
        Backend-specific metadata (distance, recall@k, …).
    """

    chunk: Chunk
    score: float
    rank: int
    retrieval_metadata: MetadataDict = field(default_factory=dict)


# ===========================================================================
# Section 8 — Storage record types
# ===========================================================================


@dataclass(frozen=True)
class CorpusRecord:
    """A persisted record combining a :class:`Chunk` with storage metadata.

    Parameters
    ----------
    record_id : str
        Unique storage record identifier (e.g. UUID or hash).
    chunk : Chunk
        The stored text chunk.
    doc_id : str
        Parent document identifier.
    collection : str
        Storage collection / table / index name.
    created_at : str
        ISO-8601 creation timestamp (``YYYY-MM-DDTHH:MM:SSZ``).
    embedding : EmbeddingVector or None
        Optional pre-computed embedding for this record.
    storage_metadata : MetadataDict
        Backend-specific storage metadata.

    Notes
    -----
    ``created_at`` is stored as a string to avoid importing ``datetime``
    and to remain fully JSON-serialisable without extra encoders.
    """

    record_id: str
    chunk: Chunk
    doc_id: str
    collection: str
    created_at: str
    embedding: EmbeddingVector | None = None
    storage_metadata: MetadataDict = field(default_factory=dict)


# ===========================================================================
# Section 9 — Protocol interfaces
# ===========================================================================


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Structural interface every chunker must satisfy.

    Any class that implements ``chunk`` and ``chunk_batch`` with the
    correct signatures is considered a valid chunker, regardless of
    inheritance.

    Notes
    -----
    ``runtime_checkable`` enables ``isinstance(obj, ChunkerProtocol)``
    in pipeline validation and registry lookups.
    """

    def chunk(
        self,
        text: str,
        doc_id: str | None = None,
        extra_metadata: MetadataDict | None = None,
    ) -> ChunkResult:
        """Split a single document into chunks.

        Parameters
        ----------
        text : str
            Raw document text.
        doc_id : str, optional
            Document identifier propagated into chunk metadata.
        extra_metadata : MetadataDict, optional
            Extra key/value pairs merged into the result metadata.

        Returns
        -------
        ChunkResult
        """
        ...

    def chunk_batch(
        self,
        texts: list[str],
        doc_ids: list[str] | None = None,
        extra_metadata: MetadataDict | None = None,
    ) -> list[ChunkResult]:
        """Split a list of documents into chunks.

        Parameters
        ----------
        texts : list[str]
            Input documents.
        doc_ids : list[str], optional
            Parallel document identifiers.
        extra_metadata : MetadataDict, optional
            Shared extra metadata for every result.

        Returns
        -------
        list[ChunkResult]
        """
        ...


@runtime_checkable
class NormalizerProtocol(Protocol):
    """Structural interface every text normaliser must satisfy."""

    def normalize(self, text: str) -> str:
        """Apply normalisation to *text*.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Normalised text.
        """
        ...

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Apply normalisation to a list of texts.

        Parameters
        ----------
        texts : list[str]
            Input texts.

        Returns
        -------
        list[str]
            Normalised texts.
        """
        ...


@runtime_checkable
class SourceProtocol(Protocol):
    """Structural interface every document source must satisfy."""

    def load(self) -> list[Document]:
        """Load and return all available documents.

        Returns
        -------
        list[Document]
        """
        ...

    def stream(self) -> Iterator[Document]:
        """Yield documents one at a time without loading all into memory.

        Yields
        ------
        Document
        """
        ...


@runtime_checkable
class StorageProtocol(Protocol):
    """Structural interface every storage backend must satisfy."""

    def save(self, record: CorpusRecord) -> None:
        """Persist a single :class:`CorpusRecord`.

        Parameters
        ----------
        record : CorpusRecord
            Record to persist.
        """
        ...

    def save_batch(self, records: list[CorpusRecord]) -> None:
        """Persist a batch of :class:`CorpusRecord` objects.

        Parameters
        ----------
        records : list[CorpusRecord]
            Records to persist.
        """
        ...

    def get(self, record_id: str) -> CorpusRecord | None:
        """Retrieve a record by its identifier.

        Parameters
        ----------
        record_id : str
            Storage record ID.

        Returns
        -------
        CorpusRecord or None
            ``None`` if no record with this ID exists.
        """
        ...

    def delete(self, record_id: str) -> bool:
        """Delete a record by its identifier.

        Parameters
        ----------
        record_id : str
            Storage record ID.

        Returns
        -------
        bool
            ``True`` if the record existed and was deleted.
        """
        ...

    def search(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """Execute a retrieval query against this backend.

        Parameters
        ----------
        query : RetrievalQuery
            Query specification.

        Returns
        -------
        list[RetrievalResult]
            Ranked results.
        """
        ...


# ===========================================================================
# Section 10 — Registry types
# ===========================================================================


@dataclass(frozen=True)
class ChunkerRegistration:
    """An entry in the :class:`ChunkerRegistry`.

    Parameters
    ----------
    strategy : ChunkStrategy
        The strategy key this registration maps to.
    chunker_class : type
        The concrete chunker class (must satisfy :class:`ChunkerProtocol`).
    default_config : ChunkerConfig
        Default configuration instance used when none is supplied.
    description : str
        Human-readable description of this chunker.
    """

    strategy: ChunkStrategy
    chunker_class: type
    default_config: ChunkerConfig
    description: str = ""


# ===========================================================================
# Section 11 — Validation / error types
# ===========================================================================


@dataclass(frozen=True)
class ValidationError:
    """A single validation failure raised during corpus processing.

    Parameters
    ----------
    field : str
        The name of the field or attribute that failed validation.
    message : str
        Human-readable description of the failure.
    value : Any
        The offending value (stored as-is for diagnostics).
    """

    field: str
    message: str
    value: Any = None


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate result of a validation pass.

    Parameters
    ----------
    valid : bool
        ``True`` if no errors were found.
    errors : list[ValidationError]
        All validation errors.  Empty when *valid* is ``True``.

    Examples
    --------
    >>> result = ValidationResult(valid=True, errors=[])
    >>> result.valid
    True
    """

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    def raise_if_invalid(self, context: str = "") -> None:
        """Raise a ``ValueError`` if this result is invalid.

        Parameters
        ----------
        context : str, optional
            Extra context prepended to the error message.

        Raises
        ------
        ValueError
            If ``valid`` is ``False``, with all error messages joined.
        """
        if not self.valid:
            prefix = f"{context}: " if context else ""
            msgs = "; ".join(
                f"{e.field}={e.value!r} — {e.message}" for e in self.errors
            )
            raise ValueError(f"{prefix}Validation failed: {msgs}")


# ===========================================================================
# Section 12 — LLM training export types
# ===========================================================================


@dataclass(frozen=True)
class TrainingExample:
    """A single example in an LLM fine-tuning dataset.

    Parameters
    ----------
    example_id : str
        Unique identifier for this training example.
    prompt : str
        Input prompt / instruction text.
    completion : str
        Expected model completion / response.
    chunk : Chunk or None
        Source chunk this example was derived from (for provenance).
    metadata : MetadataDict
        Task type, dataset split, difficulty, …

    Notes
    -----
    This schema is intentionally format-agnostic.  Serialisation to
    OpenAI JSONL, Anthropic HH, or Alpaca format is handled by the
    ``_export`` submodule, not here.
    """

    example_id: str
    prompt: str
    completion: str
    chunk: Chunk | None = None
    metadata: MetadataDict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingDataset:
    """A named collection of :class:`TrainingExample` objects.

    Parameters
    ----------
    dataset_id : str
        Unique dataset identifier.
    examples : list[TrainingExample]
        Training examples.
    split : str
        Dataset split label: ``"train"``, ``"validation"``, ``"test"``.
    metadata : MetadataDict
        Dataset-level metadata (version, task, language, …).
    """

    dataset_id: str
    examples: list[TrainingExample]
    split: str = "train"
    metadata: MetadataDict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.examples)


# ===========================================================================
# Section 13 — MCP / tool-call integration types  (future-proof)
# ===========================================================================


@dataclass(frozen=True)
class MCPToolInput:
    """Input payload for a Model Context Protocol tool call.

    Parameters
    ----------
    tool_name : str
        Registered tool identifier.
    arguments : MetadataDict
        Tool-specific argument dictionary.
    call_id : str
        Unique call identifier for result correlation.
    """

    tool_name: str
    arguments: MetadataDict
    call_id: str


@dataclass(frozen=True)
class MCPToolResult:
    """Result returned from a Model Context Protocol tool call.

    Parameters
    ----------
    call_id : str
        Identifier matching the originating :class:`MCPToolInput`.
    content : Any
        Tool result payload (str, list, dict, …).
    is_error : bool
        ``True`` if the tool raised an error.
    error_message : str or None
        Error description when ``is_error`` is ``True``.
    """

    call_id: str
    content: Any
    is_error: bool = False
    error_message: str | None = None


# ===========================================================================
# Section N — Multilang / Preprocessing types (Layer 0-3)
# ===========================================================================


@dataclass(frozen=True)
class SemantemeInfo:
    """Rich semantic unit descriptor attached to each multilang chunk.

    A *semanteme* is the smallest unit of meaning in a language.  Depending
    on the script and chunking backend this may be a morpheme, a syllabic
    aksara, a Han ideograph, or a grapheme cluster.

    All fields are optional so that callers that do not have the full
    analysis available can still populate a partial record.

    Parameters
    ----------
    surface : str
        The surface form as it appears in the normalised text.
    script : str or None
        :class:`~._chunkers._custom_tokenizer.ScriptType` value string of
        this semanteme (e.g. ``"latin"``, ``"han"``).
    direction : str or None
        Writing direction: ``"ltr"``, ``"rtl"``, or ``"ttb"``.
    morphemes : list[str] or None
        Morpheme decomposition (e.g. MeCab or camel-tools output).
        ``None`` when morphological analysis was not run.
    lemma : str or None
        Dictionary lemma of this semanteme.  ``None`` if unavailable.
    pos_tag : str or None
        Part-of-speech tag (language-specific tagset string).
    stem : str or None
        Stemmed form (Porter, Snowball, etc.).
    grapheme_count : int or None
        Number of UAX #29 grapheme clusters in ``surface``.
    codepoint_count : int or None
        Number of Unicode codepoints in ``surface``.
    is_stopword : bool or None
        ``True`` if this semanteme is a stopword in its detected language.
    language_hint : str or None
        ISO 639-1 / BCP-47 language code hint (e.g. ``"ja"``, ``"ar"``).
    determinative : str or None
        For Egyptian hieroglyphs: the determinative sign glyph.
    raw_surface : str or None
        Pre-normalisation surface form.  ``None`` when no normalisation
        changed the text (idempotent normalisation).
    embedding : list[float] or None
        Per-semanteme dense embedding vector (if computed).  Allows
        fine-grained semantic search at morpheme/aksara level.

    Notes
    -----
    **Developer note:** ``SemantemeInfo`` is stored as a list in
    ``MultilangChunkMeta.semantemes`` and serialised to
    ``CorpusDocument.morphemes`` (list of surface strings) and
    ``chunk.metadata["semantemes"]`` (list of dicts).

    Examples
    --------
    >>> info = SemantemeInfo(
    ...     surface="run", script="latin", lemma="run", stem="run", morphemes=["run"]
    ... )
    >>> info.surface
    'run'
    """

    surface: str
    script: str | None = None
    direction: str | None = None
    morphemes: list[str] | None = None
    lemma: str | None = None
    pos_tag: str | None = None
    stem: str | None = None
    grapheme_count: int | None = None
    codepoint_count: int | None = None
    is_stopword: bool | None = None
    language_hint: str | None = None
    determinative: str | None = None
    raw_surface: str | None = None
    embedding: list[float] | None = field(default=None, compare=False, hash=False)

    def to_dict(self) -> MetadataDict:
        """Serialise to a JSON-safe dict for ``chunk.metadata`` storage.

        Returns
        -------
        MetadataDict
            All fields; fields that are ``None`` are omitted to minimise
            payload size.
        """
        out: MetadataDict = {"surface": self.surface}
        for fname in (
            "script",
            "direction",
            "morphemes",
            "lemma",
            "pos_tag",
            "stem",
            "grapheme_count",
            "codepoint_count",
            "is_stopword",
            "language_hint",
            "determinative",
            "raw_surface",
        ):
            val = getattr(self, fname)
            if val is not None:
                out[fname] = val
        # Embedding is included only when explicitly requested to avoid
        # inflating small payloads.
        return out

    def with_embedding(self, vector: list[float]) -> SemantemeInfo:
        """Return a copy of this record with an embedding vector attached.

        Parameters
        ----------
        vector : list[float]
            Dense embedding from any encoder (sentence-transformers, etc.).

        Returns
        -------
        SemantemeInfo
            New frozen instance.
        """
        return SemantemeInfo(
            surface=self.surface,
            script=self.script,
            direction=self.direction,
            morphemes=self.morphemes,
            lemma=self.lemma,
            pos_tag=self.pos_tag,
            stem=self.stem,
            grapheme_count=self.grapheme_count,
            codepoint_count=self.codepoint_count,
            is_stopword=self.is_stopword,
            language_hint=self.language_hint,
            determinative=self.determinative,
            raw_surface=self.raw_surface,
            embedding=vector,
        )


@dataclass(frozen=True)
class PreprocessingStep:
    """Record of a single preprocessing transformation applied to raw text.

    Used to build a :class:`PreprocessingTrace` that makes the
    preprocessing pipeline fully retrospective — every change is
    documented with enough information to verify, replay, or invert it.

    Parameters
    ----------
    name : str
        Short identifier of the transformation (e.g.
        ``"nfc_normalise"``, ``"strip_control"``, ``"bom_strip"``,
        ``"whitespace_collapse"``, ``"ocr_correction"``).
    description : str
        Human-readable explanation of what this step does.
    changed : bool
        ``True`` if this step actually modified the text.
        ``False`` when the text was already in the expected form
        (idempotent execution).
    char_delta : int
        Change in codepoint count: ``len(after) - len(before)``.
        Negative for stripping steps; zero for reordering steps.
    grapheme_delta : int or None
        Change in grapheme cluster count.  ``None`` if grapheme
        counting was not available at this step (``regex`` not installed).
    params : MetadataDict or None
        Configuration parameters used for this step.  Must be
        JSON-serialisable.  ``None`` if the step has no parameters.
    input_hash : str or None
        MD5 hex of the text *before* this step.  Allows exact change
        detection without storing the full text.  ``None`` if not tracked.
    output_hash : str or None
        MD5 hex of the text *after* this step.

    Notes
    -----
    **Developer note:** Hash all PreprocessingStep objects into a stable
    fingerprint to build a preprocessing fingerprint for idempotency
    verification on pipeline re-runs.  The fingerprint is:
    ``hashlib.md5(",".join(s.name for s in trace.steps).encode()).hexdigest()``
    """

    name: str
    description: str
    changed: bool
    char_delta: int
    grapheme_delta: int | None = None
    params: MetadataDict | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    # Timing (wall-clock seconds via time.perf_counter())
    duration_s: float | None = None
    """Wall-clock seconds this step took.  ``None`` when not timed."""

    def to_dict(self) -> MetadataDict:
        """Serialise to a JSON-safe dict.

        Returns
        -------
        MetadataDict
        """
        out: MetadataDict = {
            "name": self.name,
            "description": self.description,
            "changed": self.changed,
            "char_delta": self.char_delta,
        }
        if self.grapheme_delta is not None:
            out["grapheme_delta"] = self.grapheme_delta
        if self.params:
            out["params"] = self.params
        if self.input_hash:
            out["input_hash"] = self.input_hash
        if self.output_hash:
            out["output_hash"] = self.output_hash
        if self.duration_s is not None:
            out["duration_ms"] = round(self.duration_s * 1000, 3)
        return out


@dataclass(frozen=True)
class PreprocessingTrace:
    """Ordered audit trail of all preprocessing transformations.

    Provides full retrospective visibility into how raw text was
    transformed into the normalised text stored in
    :attr:`~._schema.CorpusDocument.text`.

    Parameters
    ----------
    raw_text : str or None
        The original raw text *before any preprocessing*.
        ``None`` when ``include_raw_text=False`` (the default, to save
        memory on large corpora).
    steps : list[PreprocessingStep]
        Ordered list of transformations, from first applied to last.
    final_text : str
        The normalised text after all steps.
    pipeline_fingerprint : str
        MD5 hex digest of step names (``",".join(s.name for s in steps)``).
        Use this to detect preprocessing pipeline changes across corpus
        versions or pipeline runs.
    total_char_delta : int
        Sum of all ``PreprocessingStep.char_delta`` values.

    Notes
    -----
    **User note:** Use ``PreprocessingTrace`` to:

    * Verify that preprocessing was applied consistently across corpus
      versions (compare ``pipeline_fingerprint``).
    * Debug unexpected chunk boundaries by comparing
      ``raw_text`` vs ``final_text``.
    * Feed the raw text into a separate embedding index for raw-vs-normalised
      retrieval comparison.

    **Developer note:** ``PreprocessingTrace`` is stored in
    ``chunk.metadata["preprocessing_trace"]`` as a dict (via
    :meth:`to_dict`).  It is NOT stored in ``CorpusDocument`` fields
    directly (too large); consumers must pull it from the chunk metadata.

    Examples
    --------
    >>> step = PreprocessingStep(
    ...     name="nfc_normalise", description="Apply NFC", changed=False, char_delta=0
    ... )
    >>> trace = PreprocessingTrace.build(
    ...     raw_text="café", steps=[step], final_text="café"
    ... )
    >>> trace.pipeline_fingerprint
    '...'
    """

    raw_text: str | None
    steps: list[PreprocessingStep]
    final_text: str
    pipeline_fingerprint: str
    total_char_delta: int

    @staticmethod
    def build(
        raw_text: str | None,
        steps: list[PreprocessingStep],
        final_text: str,
    ) -> PreprocessingTrace:
        """Construct a :class:`PreprocessingTrace` from parts.

        Parameters
        ----------
        raw_text : str or None
            Original text before preprocessing.
        steps : list[PreprocessingStep]
            Ordered list of applied steps.
        final_text : str
            Text after all steps.

        Returns
        -------
        PreprocessingTrace
        """
        import hashlib  # noqa: PLC0415

        fp = hashlib.md5(  # noqa: S324
            ",".join(s.name for s in steps).encode("utf-8")
        ).hexdigest()
        delta = sum(s.char_delta for s in steps)
        return PreprocessingTrace(
            raw_text=raw_text,
            steps=steps,
            final_text=final_text,
            pipeline_fingerprint=fp,
            total_char_delta=delta,
        )

    def to_dict(self, *, include_raw_text: bool = False) -> MetadataDict:
        """Serialise to a JSON-safe dict for chunk metadata storage.

        Parameters
        ----------
        include_raw_text : bool, optional
            When ``True``, include ``raw_text`` in the output.
            Default ``False`` to keep chunk metadata small.

        Returns
        -------
        MetadataDict
        """
        out: MetadataDict = {
            "steps": [s.to_dict() for s in self.steps],
            "final_text": self.final_text,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "total_char_delta": self.total_char_delta,
        }
        if include_raw_text and self.raw_text is not None:
            out["raw_text"] = self.raw_text
        return out


@dataclass(frozen=True)
class MultilangChunkMeta:
    """Per-chunk multilang analysis bundle — stored in ``chunk.metadata``.

    Collects all Layer 0-3 artefacts for a single chunk in one place
    so downstream consumers have a uniform access pattern regardless of
    which chunker produced the chunk.

    Parameters
    ----------
    script : str or None
        Dominant :class:`~._chunkers._custom_tokenizer.ScriptType` value
        string for this chunk.
    script_direction : str or None
        Writing direction: ``"ltr"``, ``"rtl"``, ``"ttb"``.
    is_mixed_script : bool or None
        ``True`` if the chunk contains codepoints from more than one script.
    chunking_unit : str or None
        Granularity: ``"word"``, ``"sentence"``, ``"paragraph"``,
        ``"fixed_window"``, ``"grapheme_cluster"``, ``"semanteme"``.
    grapheme_count : int or None
        UAX #29 grapheme cluster count for ``chunk.text``.
    codepoint_count : int or None
        ``len(chunk.text)`` — stored explicitly for comparison.
    semantemes : list[SemantemeInfo] or None
        Per-semanteme analysis list.  ``None`` when semantic analysis
        was not applied.
    semanteme_count : int or None
        ``len(semantemes)`` pre-computed for fast access.
    morphemes : list[str] or None
        Flat morpheme surface list (deduplicated from ``semantemes``).
    script_spans : list[dict] or None
        :class:`~._chunkers._custom_tokenizer.ScriptSpan` dicts for
        mixed-script chunks.
    script_model_version : str or None
        Embedding model version used, format ``"name@version"``.
    preprocessing_trace : PreprocessingTrace or None
        Full audit trail of preprocessing steps.  ``None`` when
        ``include_preprocessing_trace=False``.
    raw_text : str or None
        Original pre-processing text for this chunk's span.
        ``None`` when ``include_raw_text=False``.
    embedding : list[float] or None
        Dense chunk-level embedding vector.  ``None`` until an embedder
        is applied.  Use :meth:`with_embedding` to attach one.
    language_hint : str or None
        ISO 639-1 / BCP-47 detected language code.
    model_name : str or None
        Name of the embedding model (e.g.
        ``"paraphrase-multilingual-mpnet-base-v2"``).

    Notes
    -----
    **User note:** Access via ``chunk.metadata.get("multilang")`` — it is
    stored as a dict (via :meth:`to_dict`).  If you need to attach an
    embedding later, call :meth:`with_embedding` which returns a new
    frozen instance without mutating the original.

    **Developer note:** All chunkers populate this via
    :class:`~._chunkers._multilang_mixin.MultilangMixin._build_multilang_meta`.
    The mixin's ``_build_multilang_meta`` is the single authority for
    constructing these objects — do not construct them inline in chunker code.
    """

    script: str | None = None
    script_direction: str | None = None
    is_mixed_script: bool | None = None
    chunking_unit: str | None = None
    grapheme_count: int | None = None
    codepoint_count: int | None = None
    semantemes: list[SemantemeInfo] | None = field(
        default=None, compare=False, hash=False
    )
    semanteme_count: int | None = None
    morphemes: list[str] | None = field(default=None, compare=False, hash=False)
    script_spans: list[MetadataDict] | None = field(
        default=None, compare=False, hash=False
    )
    script_model_version: str | None = None
    preprocessing_trace: PreprocessingTrace | None = field(
        default=None, compare=False, hash=False
    )
    raw_text: str | None = field(default=None, compare=False, hash=False)
    embedding: list[float] | None = field(default=None, compare=False, hash=False)
    language_hint: str | None = None
    model_name: str | None = None
    # ------------------------------------------------------------------
    # Timing and pipeline provenance tracking
    # ------------------------------------------------------------------
    chunking_duration_ms: float | None = None
    """Wall-clock time in milliseconds to produce this chunk.  Populated by
    the chunker's :meth:`MultilangMixin._ml_build_meta`."""

    preprocessing_duration_ms: float | None = None
    """Wall-clock time in milliseconds for all preprocessing steps combined
    (BOM strip + control strip + NFC normalisation)."""

    layer2_strategy: str | None = None
    """Name of the Layer 2 :class:`~._writing_system.SegmentationStrategy`
    class that produced this chunk (e.g. ``"JapaneseStrategy"``,
    ``"ArabicMorphologicalStrategy"``, ``"GraphemeClusterStrategy"``)."""

    pipeline_id: str | None = None
    """Optional pipeline run identifier injected by the caller for
    correlation across large batch runs."""

    created_at_utc: str | None = None
    """ISO-8601 UTC timestamp of when this chunk was produced,
    format ``YYYY-MM-DDTHH:MM:SS.ffffffZ``.  ``None`` when not tracked."""

    char_offset_start: int | None = None
    """Start character offset of this chunk in the *raw* (pre-NFC) text.
    Allows exact re-location of the original source span for comparison."""

    char_offset_end: int | None = None
    """End character offset of this chunk in the *raw* (pre-NFC) text."""

    token_count: int | None = None
    """Number of whitespace tokens in ``chunk.text`` after normalisation."""

    stopword_count: int | None = None
    """Number of stopwords among the tokens (if stopword analysis was run)."""

    unique_token_count: int | None = None
    """Number of unique tokens (type count, not token count)."""

    char_count: int | None = None
    """Character count of ``chunk.text`` (``len(chunk.text)``)."""

    avg_token_length: float | None = None
    """Average token length in grapheme clusters."""

    is_rtl: bool | None = None
    """``True`` when ``script_direction == "rtl"``; precomputed for fast
    dataframe filtering."""

    def with_embedding(
        self,
        vector: list[float],
        *,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> MultilangChunkMeta:
        """Return a copy with an embedding vector and optional model metadata.

        Parameters
        ----------
        vector : list[float]
            Dense embedding produced by any encoder.
        model_name : str, optional
            Name of the model used (e.g.
            ``"paraphrase-multilingual-mpnet-base-v2"``).
        model_version : str, optional
            Model version string (e.g. ``"1.2.0"``).

        Returns
        -------
        MultilangChunkMeta
            New frozen instance.
        """
        ver = (
            f"{model_name}@{model_version}"
            if model_name and model_version
            else (model_name or self.script_model_version)
        )
        return MultilangChunkMeta(
            script=self.script,
            script_direction=self.script_direction,
            is_mixed_script=self.is_mixed_script,
            chunking_unit=self.chunking_unit,
            grapheme_count=self.grapheme_count,
            codepoint_count=self.codepoint_count,
            semantemes=self.semantemes,
            semanteme_count=self.semanteme_count,
            morphemes=self.morphemes,
            script_spans=self.script_spans,
            script_model_version=ver,
            preprocessing_trace=self.preprocessing_trace,
            raw_text=self.raw_text,
            embedding=vector,
            language_hint=self.language_hint,
            model_name=model_name or self.model_name,
            # Preserve all tracking fields unchanged
            chunking_duration_ms=self.chunking_duration_ms,
            preprocessing_duration_ms=self.preprocessing_duration_ms,
            layer2_strategy=self.layer2_strategy,
            pipeline_id=self.pipeline_id,
            created_at_utc=self.created_at_utc,
            char_offset_start=self.char_offset_start,
            char_offset_end=self.char_offset_end,
            token_count=self.token_count,
            stopword_count=self.stopword_count,
            unique_token_count=self.unique_token_count,
            char_count=self.char_count,
            avg_token_length=self.avg_token_length,
            is_rtl=self.is_rtl,
        )

    def to_dict(
        self,
        *,
        include_raw_text: bool = False,
        include_preprocessing_trace: bool = False,
        include_embedding: bool = False,
        include_semanteme_detail: bool = True,
    ) -> MetadataDict:
        """Serialise to a JSON-safe dict for ``chunk.metadata["multilang"]``.

        Parameters
        ----------
        include_raw_text : bool
            Include ``raw_text`` in output.  Default ``False``.
        include_preprocessing_trace : bool
            Include the full preprocessing trace.  Default ``False``.
        include_embedding : bool
            Include the dense embedding vector.  Default ``False``.
        include_semanteme_detail : bool
            Include full ``semantemes`` list.  Default ``True``.
            Set to ``False`` to include only ``semanteme_count``.

        Returns
        -------
        MetadataDict
            All non-None fields populated per the flag settings.
        """
        out: MetadataDict = {}
        for scalar_field in (
            "script",
            "script_direction",
            "is_mixed_script",
            "chunking_unit",
            "grapheme_count",
            "codepoint_count",
            "semanteme_count",
            "script_model_version",
            "language_hint",
            "model_name",
            # Timing + tracking
            "chunking_duration_ms",
            "preprocessing_duration_ms",
            "layer2_strategy",
            "pipeline_id",
            "created_at_utc",
            "char_offset_start",
            "char_offset_end",
            "token_count",
            "stopword_count",
            "unique_token_count",
            "char_count",
            "avg_token_length",
            "is_rtl",
        ):
            val = getattr(self, scalar_field)
            if val is not None:
                out[scalar_field] = val

        if self.morphemes is not None:
            out["morphemes"] = list(self.morphemes)

        if self.script_spans is not None:
            out["script_spans"] = list(self.script_spans)

        if include_semanteme_detail and self.semantemes is not None:
            out["semantemes"] = [s.to_dict() for s in self.semantemes]

        if include_raw_text and self.raw_text is not None:
            out["raw_text"] = self.raw_text

        if include_preprocessing_trace and self.preprocessing_trace is not None:
            out["preprocessing_trace"] = self.preprocessing_trace.to_dict(
                include_raw_text=include_raw_text
            )

        if include_embedding and self.embedding is not None:
            out["embedding"] = list(self.embedding)

        return out
