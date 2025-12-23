# scikitplot/annoy/_mixins/_ndarray.pyi

"""
NDArray export and interoperability mixin.

This module defines :class:`~scikitplot.annoy._mixins._ndarray.NDArrayExportMixin`,
a high-level, user-facing mixin that adds deterministic export utilities to an
Annoy-like index wrapper (e.g., :class:`scikitplot.annoy.Index`).

The runtime implementation avoids importing heavy optional dependencies at
import time. Type hints below use forward references where appropriate.
"""  # noqa: PYI021

# from __future__ import annotations

import os
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)

import numpy as np
from typing_extensions import Literal, TypeAlias

PathLikeStr: TypeAlias = str | os.PathLike[str]
IdsInput: TypeAlias = Sequence[int] | Iterable[int] | None

__all__ = ("NDArrayExportMixin",)

class NDArrayExportMixin:
    # Minimal Annoy-like surface assumed by this mixin.
    f: int
    def get_n_items(self) -> int: ...
    def get_item_vector(self, i: int) -> Sequence[float]: ...
    def iter_item_vectors(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        with_ids: bool = True,
    ) -> Iterator[Sequence[float] | tuple[int, Sequence[float]]]: ...
    def to_numpy(
        self,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        validate_vector_len: bool = True,
    ) -> np.ndarray: ...
    def save_vectors_npy(
        self,
        path: PathLikeStr,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",
        start: int = 0,
        stop: int | None = None,
        overwrite: bool = True,
        n_rows: int | None = None,
        flush_every: int | None = None,
        validate_vector_len: bool = True,
    ) -> str: ...
    def save_vectors_npz(
        self,
        path: PathLikeStr,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        compressed: bool = True,
        include_ids: bool = True,
        array_name: str = "vectors",
        ids_name: str = "ids",
        validate_vector_len: bool = True,
    ) -> str: ...
    def to_dataframe(
        self,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        include_id: bool = True,
        columns: Sequence[str] | None = None,
        validate_vector_len: bool = True,
    ) -> Any: ...
    def to_csv(
        self,
        path: PathLikeStr,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",
        start: int = 0,
        stop: int | None = None,
        include_id: bool = True,
        header: bool = True,
        delimiter: str = ",",
        float_format: str | None = None,
        columns: Sequence[str] | None = None,
        encoding: str = "utf-8",
        validate_vector_len: bool = True,
    ) -> str: ...
    def to_parquet(
        self,
        path: PathLikeStr,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",
        start: int = 0,
        stop: int | None = None,
        include_id: bool = True,
        columns: Sequence[str] | None = None,
        batch_size: int = ...,
        compression: str | None = ...,
        validate_vector_len: bool = True,
    ) -> str: ...
    def to_sqlite(
        self,
        path: PathLikeStr,
        *,
        table: str = ...,
        dtype: str | np.dtype = "float32",
        ids: IdsInput = None,
        start: int = 0,
        stop: int | None = None,
        if_exists: Literal["fail", "replace", "append"] = ...,
        vector_format: Literal["blob", "json"] = ...,
        batch_size: int = ...,
        validate_vector_len: bool = True,
    ) -> str: ...
    def to_scipy_csr(
        self,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
    ) -> Any: ...
    def log_export_to_mlflow(
        self,
        artifact_path: PathLikeStr,
        *,
        mlflow_artifact_path: str = ...,
        log_params: bool = ...,
        extra_params: Mapping[str, Any] | None = None,
    ) -> str: ...
    def partition_existing_ids(
        self,
        ids: Sequence[int],
        *,
        missing_exceptions: tuple[type[BaseException], ...] = ...,
    ) -> tuple[list[int], list[int]]: ...
