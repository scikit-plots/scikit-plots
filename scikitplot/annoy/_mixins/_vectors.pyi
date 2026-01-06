# scikitplot/annoy/_mixins/_vectors.pyi

"""Typing stubs for vector neighbor utilities."""  # noqa: PYI021

# from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Protocol, TypeAlias, overload, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__: tuple[str, ...] = ("VectorOpsMixin",)

IndexArray: TypeAlias = NDArray[np.intp]
DistanceArray: TypeAlias = NDArray[np.float32]
VectorArray: TypeAlias = NDArray[np.float32]
VectorMatrix: TypeAlias = NDArray[np.float32]
VectorBatch: TypeAlias = NDArray[np.float32]

@runtime_checkable
class _AnnoyVectorBackend(Protocol):  # noqa: PYI046
    def get_item_vector(self, item: int) -> list[float]: ...
    def get_n_items(self) -> int: ...
    def get_n_trees(self) -> int: ...
    def get_nns_by_item(
        self,
        item: int,
        n: int,
        search_k: int = ...,
        include_distances: bool = ...,
    ) -> list[int] | tuple[list[int], list[float]]: ...
    def get_nns_by_vector(
        self,
        vector: list[float],
        n: int,
        search_k: int = ...,
        include_distances: bool = ...,
    ) -> list[int] | tuple[list[int], list[float]]: ...

class VectorOpsMixin:
    """User-facing neighbor queries for Annoy-like backends."""  # noqa: PYI021

    @overload
    def query_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
    ) -> IndexArray: ...
    @overload
    def query_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[True],
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
    ) -> tuple[IndexArray, DistanceArray]: ...
    @overload
    def query_vectors_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        dtype: Any = ...,
    ) -> VectorMatrix: ...
    @overload
    def query_vectors_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[True],
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        dtype: Any = ...,
    ) -> tuple[VectorMatrix, DistanceArray]: ...
    @overload
    def query_by_vector(
        self,
        vector: ArrayLike,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
    ) -> IndexArray: ...
    @overload
    def query_by_vector(
        self,
        vector: ArrayLike,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[True],
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
    ) -> tuple[IndexArray, DistanceArray]: ...
    @overload
    def query_vectors_by_vector(
        self,
        vector: ArrayLike,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        dtype: Any = ...,
        output_type: Literal["item"],
    ) -> IndexArray: ...
    @overload
    def query_vectors_by_vector(
        self,
        vector: ArrayLike,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[True],
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        dtype: Any = ...,
        output_type: Literal["item"],
    ) -> tuple[IndexArray, DistanceArray]: ...
    @overload
    def query_vectors_by_vector(
        self,
        vector: ArrayLike,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        dtype: Any = ...,
        output_type: Literal["vector"] = ...,
    ) -> VectorMatrix: ...
    @overload
    def query_vectors_by_vector(
        self,
        vector: ArrayLike,
        n_neighbors: int,
        *,
        search_k: int = ...,
        include_distances: Literal[True],
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        dtype: Any = ...,
        output_type: Literal["vector"] = ...,
    ) -> tuple[VectorMatrix, DistanceArray]: ...
    @overload
    def kneighbors(
        self,
        X: ArrayLike,
        n_neighbors: int = ...,
        *,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        output_type: Literal["item"],
    ) -> NDArray[np.intp]: ...
    @overload
    def kneighbors(
        self,
        X: ArrayLike,
        n_neighbors: int = ...,
        *,
        search_k: int = ...,
        include_distances: Literal[True],
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        output_type: Literal["item"],
    ) -> tuple[NDArray[np.intp], NDArray[np.float32]]: ...
    @overload
    def kneighbors(
        self,
        X: ArrayLike,
        n_neighbors: int = ...,
        *,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        output_type: Literal["vector"] = ...,
    ) -> NDArray[np.float32]: ...
    @overload
    def kneighbors(
        self,
        X: ArrayLike,
        n_neighbors: int = ...,
        *,
        search_k: int = ...,
        include_distances: Literal[True],
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        output_type: Literal["vector"] = ...,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
    def kneighbors_graph(
        self,
        X: ArrayLike,
        n_neighbors: int = ...,
        *,
        search_k: int = ...,
        mode: Literal["connectivity", "distance"] = ...,
        exclude_self: bool = ...,
        exclude_item_ids: Iterable[int] | None = ...,
        ensure_all_finite: bool | Literal["allow-nan"] = ...,
        copy: bool = ...,
        output_type: Literal["item"] = ...,
    ) -> Any: ...
