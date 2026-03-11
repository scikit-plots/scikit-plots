# scikitplot/corpus/_base.pyi
# ============================================================
# Type stubs for scikitplot.corpus._base
#
# Mirrors _base.py exactly. Must be updated in the same commit as any
# base-class change.
# ============================================================
# from __future__ import annotations   # NOT used in .pyi files.

import pathlib
from dataclasses import dataclass
from typing import (  # noqa: F401
    Any,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import Self  # noqa: F401

from scikitplot.corpus._schema import (  # noqa: F401
    ChunkingStrategy,
    CorpusDocument,
    SectionType,
    SourceType,
)

# ---------------------------------------------------------------------------
# ChunkerBase
# ---------------------------------------------------------------------------

class ChunkerBase:
    strategy: ClassVar[ChunkingStrategy]

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = ...,
    ) -> list[tuple[int, str]]: ...

# ---------------------------------------------------------------------------
# FilterBase / DefaultFilter
# ---------------------------------------------------------------------------

class FilterBase:
    def include(self, doc: CorpusDocument) -> bool: ...

class DefaultFilter(FilterBase):
    min_words: int
    min_chars: int

    def __init__(
        self,
        min_words: int = ...,
        min_chars: int = ...,
    ) -> None: ...
    def include(self, doc: CorpusDocument) -> bool: ...

# ---------------------------------------------------------------------------
# DocumentReader
# ---------------------------------------------------------------------------

@dataclass
class DocumentReader:
    _registry: ClassVar[dict[str, type[Self]]]
    file_type: ClassVar[str]
    file_types: ClassVar[list[str] | None]

    # Instance fields
    input_file: pathlib.Path
    chunker: ChunkerBase | None
    filter_: FilterBase | None
    filename_override: str | None
    default_language: str | None
    source_uri: str | None
    source_provenance: dict[str, Any]

    @property
    def file_name(self) -> str: ...
    def validate_input(self) -> None: ...
    def get_raw_chunks(
        self,
    ) -> Generator[dict[str, Any], None, None]: ...  # noqa: UP043
    def get_documents(
        self,
    ) -> Generator[CorpusDocument, None, None]: ...  # noqa: UP043
    @classmethod
    def supported_types(cls) -> list[str]: ...
    @classmethod
    def subclass_by_type(cls) -> dict[str, type[Self]]: ...
    @classmethod
    def create(
        cls,
        input_file: pathlib.Path | str,
        *,
        chunker: ChunkerBase | None = ...,
        filter_: FilterBase | None = ...,
        filename_override: str | None = ...,
        default_language: str | None = ...,
        source_type: SourceType | None = ...,
        source_title: str | None = ...,
        source_author: str | None = ...,
        source_date: str | None = ...,
        collection_id: str | None = ...,
        doi: str | None = ...,
        isbn: str | None = ...,
    ) -> Self: ...
    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        chunker: ChunkerBase | None = ...,
        filter_: FilterBase | None = ...,
        default_language: str | None = ...,
    ) -> Self: ...

_R = TypeVar("_R", bound=DocumentReader)  # noqa: PYI018
_C = TypeVar("_C", bound=ChunkerBase)  # noqa: PYI018
_F = TypeVar("_F", bound=FilterBase)  # noqa: PYI018
