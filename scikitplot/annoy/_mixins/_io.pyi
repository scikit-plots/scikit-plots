# scikitplot/annoy/_mixins/_io.pyi
"""Typing stubs for :mod:`~scikitplot.annoy._mixins._io`."""  # noqa: PYI021

# from __future__ import annotations

from os import PathLike

from typing_extensions import Self

__all__: list[str]  # noqa: PYI035

class IndexIOMixin:
    def save_index(
        self,
        path: str | PathLike[str],
        *,
        prefault: bool | None = ...,
    ) -> str | PathLike: ...
    def load_index(
        self,
        f: int,
        metric: str,
        path: str | PathLike[str],
        *,
        prefault: bool | None = ...,
    ) -> None: ...
    def save_bundle(
        self,
        manifest_filename: str = ...,
        index_filename: str = ...,
        *,
        prefault: bool | None = ...,
    ) -> list[str]: ...
    @classmethod
    def load_bundle(
        cls: type[Self],
        manifest_filename: str = ...,
        index_filename: str = ...,
        *,
        prefault: bool | None = ...,
    ) -> Self: ...
    def to_bytes(
        self,
        format: str | None = ...,
    ) -> bytes: ...
    @classmethod
    def from_bytes(
        cls: type[Self],
        data: bytes | bytearray | memoryview,
        *,
        f: int | None = ...,
        metric: str | None = ...,
        prefault: bool | None = ...,
    ) -> Self: ...
