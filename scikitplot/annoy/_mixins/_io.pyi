# scikitplot/annoy/_mixins/_io.pyi
"""
Typing stubs for :mod:`~scikitplot.annoy._mixins._io`.
"""  # noqa: PYI021

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
        atomic: bool = ...,
    ) -> None: ...
    def load_index(
        self,
        path: str | PathLike[str],
        *,
        prefault: bool | None = ...,
    ) -> None: ...
    def to_bytes(self) -> bytes: ...
    @classmethod
    def from_bytes(
        cls: type[Self],
        data: bytes,
        *,
        f: int,
        metric: str,
        prefault: bool | None = ...,
    ) -> Self: ...
    def save_bundle(
        self,
        directory: str | PathLike[str],
        *,
        index_filename: str = ...,
        manifest_filename: str = ...,
        prefault: bool | None = ...,
    ) -> None: ...
    @classmethod
    def load_bundle(
        cls: type[Self],
        directory: str | PathLike[str],
        *,
        index_filename: str = ...,
        manifest_filename: str = ...,
        prefault: bool | None = ...,
    ) -> Self: ...
