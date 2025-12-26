# scikitplot/annoy/_mixins/_pickle.pyi
"""Typing stubs for :mod:`scikitplot.annoy._mixins._pickle`."""  # noqa: PYI021

# from __future__ import annotations

from collections.abc import Mapping
from os import PathLike  # noqa: F401
from typing import Any, ClassVar

from typing_extensions import Literal, Self, TypeAlias

CompressMode: TypeAlias = Literal["zlib", "gzip"] | None
PickleMode: TypeAlias = Literal["auto", "disk", "byte"]

__all__: list[str]  # noqa: PYI035

class PickleMixin:
    _PICKLE_STATE_VERSION: ClassVar[int]
    _compress_mode: CompressMode
    _pickle_mode: PickleMode

    @property
    def compress_mode(self) -> CompressMode: ...
    @compress_mode.setter
    def compress_mode(self, value: CompressMode) -> None: ...
    @property
    def pickle_mode(self) -> PickleMode: ...
    @pickle_mode.setter
    def pickle_mode(self, value: PickleMode) -> None: ...
    def __reduce__(self) -> object: ...
    def __reduce_ex__(self, protocol: int) -> object: ...
    @classmethod
    def _rebuild(cls: type[Self], state: Mapping[str, Any]) -> Self: ...
