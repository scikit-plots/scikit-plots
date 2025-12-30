# scikitplot/annoy/_mixins/_vectors.pyi

"""Typing stubs for vector neighbor utilities."""  # noqa: PYI021

# from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__: tuple[str, ...] = ("VectorOpsMixin",)

IndexArray: TypeAlias = NDArray[np.intp]
DistanceArray: TypeAlias = NDArray[np.floating]

@runtime_checkable
class _AnnoyVectorBackend(Protocol):  # noqa: PYI046
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
    def get_item_vector(self, item: int) -> list[float]: ...
    def get_n_trees(self) -> int: ...
    def get_n_items(self) -> int: ...

class VectorOpsMixin:
    def query_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
    ) -> IndexArray | tuple[IndexArray, DistanceArray]: ...
    def query_by_vector(
        self,
        vector: ArrayLike,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
    ) -> IndexArray | tuple[IndexArray, DistanceArray]: ...
    def query_vectors_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        dtype: Any = ...,
    ) -> NDArray[Any] | tuple[NDArray[Any], DistanceArray]: ...
    def query_vectors_by_vector(
        self,
        vector: ArrayLike,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
        dtype: Any = ...,
    ) -> NDArray[Any] | tuple[NDArray[Any], DistanceArray]: ...
    def kneighbors(
        self,
        X: ArrayLike,
        n_neighbors: int = 5,
        *,
        search_k: int = -1,
        include_distances: bool = True,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
    ) -> IndexArray | tuple[IndexArray, DistanceArray]: ...
    def kneighbors_graph(
        self,
        X: ArrayLike,
        n_neighbors: int = 5,
        *,
        search_k: int = -1,
        mode: Literal["connectivity", "distance"] = "connectivity",
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
    ) -> Any: ...
