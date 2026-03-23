# scikitplot/corpus/_base.pyi
#
# flake8: noqa: E301
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# ============================================================
# Type stubs for scikitplot.corpus._base
#
# Mirrors _base.py exactly.  Must be updated in the same commit as any
# base-class change.
#
# Connection map
# --------------
# _base.py             ← this stub mirrors
# _schema.py           ← CorpusDocument, SourceType, Modality, ErrorPolicy
# _readers/            ← every concrete reader subclasses DocumentReader
# _embeddings/         ← EmbeddingEngine / MultimodalEmbeddingEngine consume
#                         the output of get_documents()
# _corpus_builder.py   ← CorpusBuilder calls DocumentReader.create() /
#                         from_url() / from_manifest()
# _url_handler.py      ← from_url() calls classify_url(), probe_url_kind(),
#                         download_url(), resolve_url()
# _pipeline.py         ← CorpusPipeline wraps a DocumentReader
# PipelineGuard        ← wraps get_documents() for resilience + dedup
# ============================================================

import pathlib
import types
from dataclasses import dataclass
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generator,
    Iterator,
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
    ErrorPolicy,
    Modality,
    SourceType,
)

_MultiSourceReader = object  # future definition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_url(s: object) -> bool: ...

# ---------------------------------------------------------------------------
# ChunkerBase — text segmentation contract
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
# DocumentReader — format-dispatch base
# ---------------------------------------------------------------------------

@dataclass
class DocumentReader:
    _registry: ClassVar[dict[str, type[Self]]]
    _DOWNLOADABLE_EXTENSIONS: ClassVar[frozenset[str]]

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

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def file_name(self) -> str: ...

    # ------------------------------------------------------------------
    # Abstract protocol
    # ------------------------------------------------------------------

    def validate_input(self) -> None: ...
    def get_raw_chunks(
        self,
    ) -> Generator[dict[str, Any], None, None]: ...  # noqa: UP043
    def get_documents(
        self,
    ) -> Generator[CorpusDocument, None, None]: ...  # noqa: UP043

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    @classmethod
    def supported_types(cls) -> list[str]: ...
    @classmethod
    def subclass_by_type(cls) -> dict[str, type[Self]]: ...

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    @classmethod
    def _build_prov(
        cls,
        *,
        source_type: SourceType | None = ...,
        source_title: str | None = ...,
        source_author: str | None = ...,
        source_date: str | None = ...,
        collection_id: str | None = ...,
        doi: str | None = ...,
        isbn: str | None = ...,
    ) -> dict[str, Any]: ...

    # ------------------------------------------------------------------
    # Factory entry points
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *inputs: pathlib.Path | str,
        chunker: ChunkerBase | None = ...,
        filter_: FilterBase | None = ...,
        filename_override: str | None = ...,
        default_language: str | None = ...,
        source_type: SourceType | list[SourceType | None] | None = ...,
        source_title: str | None = ...,
        source_author: str | None = ...,
        source_date: str | None = ...,
        collection_id: str | None = ...,
        doi: str | None = ...,
        isbn: str | None = ...,
        **kwargs: Any,
    ) -> Self | _MultiSourceReader: ...
    @classmethod
    def _create_one(
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
        **kwargs: Any,
    ) -> Self: ...
    @classmethod
    def from_manifest(
        cls,
        manifest_path: pathlib.Path | str,
        *,
        chunker: ChunkerBase | None = ...,
        filter_: FilterBase | None = ...,
        default_language: str | None = ...,
        source_type: SourceType | None = ...,
        source_title: str | None = ...,
        source_author: str | None = ...,
        source_date: str | None = ...,
        collection_id: str | None = ...,
        doi: str | None = ...,
        isbn: str | None = ...,
        encoding: str = ...,
        **kwargs: Any,
    ) -> _MultiSourceReader: ...
    @classmethod
    def from_url(
        cls,
        url: str,
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
        **kwargs: Any,
    ) -> Self: ...

_R = TypeVar("_R", bound=DocumentReader)  # noqa: PYI018
_C = TypeVar("_C", bound=ChunkerBase)  # noqa: PYI018
_F = TypeVar("_F", bound=FilterBase)  # noqa: PYI018

# ---------------------------------------------------------------------------
# _MultiSourceReader
# ---------------------------------------------------------------------------

class _MultiSourceReader:
    readers: list[DocumentReader]

    def __init__(self, readers: list[DocumentReader]) -> None: ...
    @property
    def n_readers(self) -> int: ...
    def get_documents(
        self,
    ) -> Generator[CorpusDocument, None, None]: ...  # noqa: UP043
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...
    def close(self) -> None: ...

# ---------------------------------------------------------------------------
# DummyReader
# ---------------------------------------------------------------------------

class DummyReader(DocumentReader):
    file_type: ClassVar[str]
    file_types: ClassVar[list[str] | None]

    @classmethod
    def check(
        cls,
        *sources: pathlib.Path | str,
        timeout: int = ...,
        raise_on_first: bool = ...,
    ) -> tuple[
        list[pathlib.Path | str], list[tuple[pathlib.Path | str, Exception]]
    ]: ...
    def get_raw_chunks(
        self,
    ) -> Generator[dict[str, Any], None, None]: ...  # noqa: UP043
    def validate_input(self) -> None: ...

# ---------------------------------------------------------------------------
# PipelineGuard
# ---------------------------------------------------------------------------

class PipelineGuard:
    policy: ErrorPolicy
    dedup: bool
    checkpoint_path: pathlib.Path | None
    checkpoint_every: int
    max_retries: int
    retry_delay: float

    def __init__(
        self,
        policy: Any | None = ...,
        *,
        dedup: bool = ...,
        checkpoint_path: pathlib.Path | None = ...,
        checkpoint_every: int = ...,
        max_retries: int = ...,
        retry_delay: float = ...,
    ) -> None: ...
    def iter(
        self,
        source: Any,
    ) -> Generator[CorpusDocument, None, None]: ...  # noqa: UP043
    def close(self) -> None: ...
    @property
    def stats(self) -> dict[str, int]: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...
