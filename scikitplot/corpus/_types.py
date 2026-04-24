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
