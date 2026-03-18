# scikitplot/corpus/_schema.pyi
# ============================================================
# Type stubs for scikitplot.corpus._schema
#
# Provides full numpy.typing annotations for static type checkers
# (mypy, pyright, pylance) without imposing a numpy import at runtime.
#
# Mirrors _schema.py exactly. Must be updated in the same commit as
# any schema change (Issue S-8).
#
# Requires: numpy >= 1.20 (NDArray introduced in 1.20, stable in 1.26+)
# ============================================================
# from __future__ import annotations   # NOT used in .pyi — stubs are
#                                       # evaluated by type checkers directly.

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
    Tuple,
    Type,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from typing_extensions import Self, TypeAlias

if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnumBase
else:
    class _StrEnumBase(str, Enum): ...

# Canonical embedding type: 1-D float32 array
EmbeddingArray: TypeAlias = npt.NDArray[np.float32]
"""Type alias for a 1-D ``numpy.float32`` embedding vector."""

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
    # Literary / dramatic / research-paper values (Issue S-1)
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
    """Semantic label for the kind of source. (Issue S-2)."""  # noqa: PYI021

    BOOK: str
    ARTICLE: str
    RESEARCH: str
    MOVIE: str
    SUBTITLE: str
    PLAY: str
    POEM: str
    BIOGRAPHY: str
    WEB: str
    WIKI: str
    IMAGE: str
    VIDEO: str
    AUDIO: str
    SPREADSHEET: str
    CODE: str
    UNKNOWN: str

class MatchMode(_StrEnumBase):
    """Search mode for intertextual matching queries. (Issue S-3)."""  # noqa: PYI021

    STRICT: str
    KEYWORD: str
    SEMANTIC: str
    HYBRID: str

# ---------------------------------------------------------------------------
# Promoted-key registry
# ---------------------------------------------------------------------------

_PROMOTED_RAW_KEYS: frozenset[str]

# ---------------------------------------------------------------------------
# CorpusDocument
# ---------------------------------------------------------------------------

@dataclass
class CorpusDocument:
    REQUIRED_FIELDS: ClassVar[tuple[str, ...]]

    # Core fields
    doc_id: str
    source_file: str
    chunk_index: int
    text: str
    section_type: SectionType
    chunking_strategy: ChunkingStrategy
    language: str | None
    char_start: int | None
    char_end: int | None
    # Fully typed embedding in stubs — NDArray at static analysis time
    embedding: EmbeddingArray | None
    metadata: dict[str, Any]

    # Provenance fields (Issue S-4)
    source_type: SourceType
    source_title: str | None
    source_author: str | None
    source_date: str | None
    collection_id: str | None
    url: str | None
    doi: str | None
    isbn: str | None

    # Position fields (Issue S-4)
    page_number: int | None
    paragraph_index: int | None
    line_number: int | None
    parent_doc_id: str | None

    # Dramatic position fields (Issue S-4)
    act: int | None
    scene_number: int | None

    # Media-specific fields (Issue S-4)
    timecode_start: float | None
    timecode_end: float | None
    confidence: float | None
    ocr_engine: str | None
    bbox: tuple[float, ...] | None

    # NLP enrichment fields (Issue S-4)
    normalized_text: str | None
    tokens: list[str] | None
    lemmas: list[str] | None
    stems: list[str] | None
    keywords: list[str] | None

    @property
    def has_embedding(self) -> bool: ...
    @property
    def word_count(self) -> int: ...
    @property
    def char_count(self) -> int: ...
    def validate(self) -> None: ...
    @classmethod
    def make_doc_id(
        cls,
        source_file: str,
        chunk_index: int,
        text: str,
        source_type: SourceType = ...,
    ) -> str: ...
    @classmethod
    def create(
        cls,
        source_file: str,
        chunk_index: int,
        text: str,
        # Core classification
        section_type: SectionType = ...,
        chunking_strategy: ChunkingStrategy = ...,
        language: str | None = ...,
        # Character offsets
        char_start: int | None = ...,
        char_end: int | None = ...,
        # Embedding
        embedding: EmbeddingArray | None = ...,
        # Ad-hoc metadata
        metadata: dict[str, Any] | None = ...,
        # Explicit doc_id override
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
        bbox: tuple[float, ...] | None = ...,
        # NLP enrichment
        normalized_text: str | None = ...,
        tokens: list[str] | None = ...,
        lemmas: list[str] | None = ...,
        stems: list[str] | None = ...,
        keywords: list[str] | None = ...,
    ) -> Self: ...
    def replace(self, **changes: Any) -> Self: ...
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
