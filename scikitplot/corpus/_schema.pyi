# scikitplot/corpus/_schema.pyi
# ============================================================
# Type stubs for scikitplot.corpus._schema
#
# Provides full numpy.typing annotations for static type checkers
# (mypy, pyright, pylance) without imposing a numpy import at runtime.
#
# Mirrors _schema.py exactly.  Must be updated in the same commit as
# any schema change.
#
# Requires: numpy >= 1.20 (NDArray introduced in 1.20, stable in 1.26+)
#
# Connection map
# --------------
# _schema.py      ← this stub mirrors
# _base.py        ← imports CorpusDocument, SourceType, _PROMOTED_RAW_KEYS
# _readers/       ← every reader yields CorpusDocument via get_documents()
# _embeddings/    ← EmbeddingEngine.embed_documents() sets doc.embedding
#                 ← MultimodalEmbeddingEngine routes by doc.modality
#                 ← raw_tensor, raw_bytes, raw_shape, raw_dtype set by readers
# _adapters.py    ← all to_*() functions accept list[CorpusDocument]
# _similarity/    ← SimilarityIndex.build() indexes doc.embedding
# _pipeline.py    ← CorpusPipeline.run() returns list[CorpusDocument]
# _corpus_builder ← CorpusBuilder.build() → BuildResult.documents
# _storage/       ← StorageBase.save(doc) / .get(doc_id) → CorpusDocument
# PipelineGuard   ← uses content_hash for dedup; reads doc_id for checkpoint
# ============================================================

import sys
from dataclasses import dataclass
from enum import Enum
from typing import (  # noqa: F401
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from typing_extensions import Self, TypeAlias  # noqa: F401

if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnumBase
else:
    class _StrEnumBase(str, Enum): ...

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EmbeddingArray: TypeAlias = npt.NDArray[np.float32]
"""1-D ``float32`` embedding vector — the type of :attr:`CorpusDocument.embedding`.

Shape: ``(D,)`` where D is the model dimension (e.g. 384, 768, 1536).
Stored as ``Any`` at runtime; typed here for static analysers.

Used by
-------
- :class:`~scikitplot.corpus._embeddings._embedding.EmbeddingEngine.embed_documents`
- :class:`~scikitplot.corpus._embeddings._multimodal_embedding.MultimodalEmbeddingEngine.embed_documents`
- :class:`~scikitplot.corpus._similarity.SimilarityIndex`
- :func:`~scikitplot.corpus._adapters.to_rag_tuples`
"""

RawTensorArray: TypeAlias = npt.NDArray[Any]
"""Raw media tensor array — the type of :attr:`CorpusDocument.raw_tensor`.

Shape conventions (channels-last throughout):
  - **Image** : ``(H, W, C)`` uint8 — RGB, as produced by ``PIL.Image``
  - **Audio** : ``(samples,)`` float32 — normalised ``[-1, 1]``, 16 kHz
  - **Video** : ``(T, H, W, C)`` uint8 — T frames, channels-last

Set by readers when ``yield_raw=True`` (ImageReader) or
``yield_waveform=True`` (AudioReader) or ``yield_frames=True`` (VideoReader).

Consumed by
-----------
- :class:`~scikitplot.corpus._embeddings._multimodal_embedding.MultimodalEmbeddingEngine`
- :func:`~scikitplot.corpus._adapters.to_tensorflow_dataset`
- :func:`~scikitplot.corpus._adapters.to_torch_dataloader`
- :func:`~scikitplot.corpus._adapters.to_numpy_arrays`
"""

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SectionType(_StrEnumBase):
    TEXT: str
    FOOTNOTE: str
    TITLE: str
    TABLE: str
    HEADER: str
    FIGURE: str
    CODE: str
    CAPTION: str
    METADATA: str
    UNKNOWN: str
    ABSTRACT: str
    REFERENCES: str
    STAGE_DIRECTION: str
    DIALOGUE: str
    VERSE: str
    ACKNOWLEDGEMENTS: str
    LIST_ITEM: str
    SIDEBAR: str
    LYRICS: str
    TRANSCRIPT: str

class ChunkingStrategy(_StrEnumBase):
    SENTENCE: str
    PARAGRAPH: str
    FIXED_WINDOW: str
    SEMANTIC: str
    PAGE: str
    BLOCK: str
    CUSTOM: str
    NONE: str

class ExportFormat(_StrEnumBase):
    CSV: str
    PARQUET: str
    JSON: str
    JSONL: str
    HUGGINGFACE: str
    MLFLOW: str
    PICKLE: str
    JOBLIB: str
    NUMPY: str
    POLARS: str
    PANDAS: str

class SourceType(_StrEnumBase):
    BOOK: str
    ARTICLE: str
    RESEARCH: str
    BIOGRAPHY: str
    PLAY: str
    POEM: str
    NEWS: str
    BLOG: str
    NEWSLETTER: str
    PRESS_RELEASE: str
    MOVIE: str
    SUBTITLE: str
    VIDEO: str
    AUDIO: str
    PODCAST: str
    LECTURE: str
    INTERVIEW: str
    WEB: str
    WIKI: str
    SOCIAL_MEDIA: str
    FORUM: str
    FAQ: str
    DOCUMENTATION: str
    TUTORIAL: str
    MANUAL: str
    REPORT: str
    LEGAL: str
    MEDICAL: str
    PATENT: str
    SPREADSHEET: str
    DATASET: str
    CODE: str
    EMAIL: str
    CHAT: str
    IMAGE: str
    UNKNOWN: str

    @classmethod
    def infer(
        cls,
        input_path: Any | None = ...,
        *,
        mime_type: str | None = ...,
    ) -> Self: ...

class MatchMode(_StrEnumBase):
    STRICT: str
    KEYWORD: str
    SEMANTIC: str
    HYBRID: str

class Modality(_StrEnumBase):
    TEXT: str
    IMAGE: str
    AUDIO: str
    VIDEO: str
    MULTIMODAL: str

class ErrorPolicy(_StrEnumBase):
    RAISE: str
    SKIP: str
    LOG: str
    RETRY: str

# ---------------------------------------------------------------------------
# Promoted-key registry
# ---------------------------------------------------------------------------

_PROMOTED_RAW_KEYS: frozenset[str]
"""
Keys in ``get_raw_chunks()`` dicts promoted to first-class :class:`CorpusDocument` fields.

Any key **not** in this set and not ``"text"`` or ``"section_type"`` flows
into :attr:`CorpusDocument.metadata`.

Includes (in addition to all provenance and position keys):
``"modality"``, ``"raw_bytes"``, ``"raw_tensor"``, ``"raw_shape"``,
``"raw_dtype"``, ``"frame_index"``, ``"content_hash"``.

See Also
--------
scikitplot.corpus._base.DocumentReader.get_documents : Uses this registry.
"""

# ---------------------------------------------------------------------------
# CorpusDocument
# ---------------------------------------------------------------------------

@dataclass
class CorpusDocument:
    REQUIRED_FIELDS: ClassVar[tuple[str, ...]]
    """Field names checked by :meth:`validate`."""

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    doc_id: str
    input_path: str
    chunk_index: int
    text: str | None
    section_type: SectionType
    chunking_strategy: ChunkingStrategy
    language: str | None
    char_start: int | None
    char_end: int | None
    embedding: EmbeddingArray | None
    metadata: dict[str, Any]

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    source_type: SourceType
    source_title: str | None
    source_author: str | None
    source_date: str | None
    collection_id: str | None
    url: str | None
    doi: str | None
    isbn: str | None

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------
    page_number: int | None
    paragraph_index: int | None
    line_number: int | None
    parent_doc_id: str | None

    # ------------------------------------------------------------------
    # Dramatic position
    # ------------------------------------------------------------------
    act: int | None
    scene_number: int | None

    # ------------------------------------------------------------------
    # Media-specific
    # ------------------------------------------------------------------
    timecode_start: float | None
    timecode_end: float | None
    confidence: float | None
    ocr_engine: str | None
    bbox: tuple[float, float, float, float] | None

    # ------------------------------------------------------------------
    # NLP enrichment
    # ------------------------------------------------------------------
    normalized_text: str | None
    tokens: list[str] | None
    lemmas: list[str] | None
    stems: list[str] | None
    keywords: list[str] | None

    # ------------------------------------------------------------------
    # Raw media / multimodal
    # ------------------------------------------------------------------
    modality: Modality
    raw_bytes: bytes | None
    raw_tensor: RawTensorArray | None
    raw_shape: tuple[int, ...] | None
    raw_dtype: str | None
    frame_index: int | None
    content_hash: str | None

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def has_embedding(self) -> bool: ...
    @property
    def has_raw(self) -> bool: ...
    @property
    def word_count(self) -> int: ...
    @property
    def char_count(self) -> int: ...

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None: ...

    # ------------------------------------------------------------------
    # Factory / ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_content_hash(
        text: str | None = ...,
        raw_bytes: bytes | None = ...,
    ) -> str: ...
    @classmethod
    def make_doc_id(
        cls,
        input_path: str,
        chunk_index: int,
        text: str | None,
        source_type: SourceType = ...,
    ) -> str: ...
    @classmethod
    def create(
        cls,
        input_path: str,
        chunk_index: int,
        text: str | None,
        section_type: SectionType = ...,
        chunking_strategy: ChunkingStrategy = ...,
        language: str | None = ...,
        char_start: int | None = ...,
        char_end: int | None = ...,
        embedding: EmbeddingArray | None = ...,
        metadata: dict[str, Any] | None = ...,
        doc_id: str | None = ...,
        # Provenance
        source_type: SourceType = ...,
        source_title: str | None = ...,
        source_author: str | None = ...,
        source_date: str | None = ...,
        collection_id: str | None = ...,
        url: str | None = ...,
        doi: str | None = ...,
        isbn: str | None = ...,
        # Position
        page_number: int | None = ...,
        paragraph_index: int | None = ...,
        line_number: int | None = ...,
        parent_doc_id: str | None = ...,
        # Dramatic position
        act: int | None = ...,
        scene_number: int | None = ...,
        # Media-specific
        timecode_start: float | None = ...,
        timecode_end: float | None = ...,
        confidence: float | None = ...,
        ocr_engine: str | None = ...,
        bbox: tuple[float, float, float, float] | None = ...,
        # NLP enrichment
        normalized_text: str | None = ...,
        tokens: list[str] | None = ...,
        lemmas: list[str] | None = ...,
        stems: list[str] | None = ...,
        keywords: list[str] | None = ...,
        # Raw media
        modality: Modality | None = ...,
        raw_bytes: bytes | None = ...,
        raw_tensor: RawTensorArray | None = ...,
        raw_shape: tuple[int, ...] | None = ...,
        raw_dtype: str | None = ...,
        frame_index: int | None = ...,
        content_hash: str | None = ...,
    ) -> Self: ...

    # ------------------------------------------------------------------
    # Copy-on-write mutation
    # ------------------------------------------------------------------

    def replace(self, **changes: Any) -> Self: ...

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, *, include_embedding: bool = ...) -> dict[str, Any]: ...
    def to_flat_dict(self, *, include_embedding: bool = ...) -> dict[str, Any]: ...
    def to_pandas_row(self, *, include_embedding: bool = ...) -> dict[str, Any]: ...
    def to_polars_row(self, *, include_embedding: bool = ...) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...

_T = TypeVar("_T", bound=CorpusDocument)  # noqa: PYI018

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def documents_to_pandas(
    docs: list[CorpusDocument],
    *,
    include_embedding: bool = ...,
) -> pd.DataFrame: ...
def documents_to_polars(
    docs: list[CorpusDocument],
    *,
    include_embedding: bool = ...,
) -> pl.DataFrame: ...
