# scikitplot/annoy/_mixins/_vectors.pyi
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore[]

"""Vector utilities for Annoy-style indexes (typing stubs)."""

# from __future__ import annotations

from typing import Any, Iterable, Iterator, Protocol, Sequence, TypeAlias, overload, runtime_checkable
from typing_extensions import Literal

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = Any  # type: ignore[assignment]

Ids: TypeAlias = list[int]
Dists: TypeAlias = list[float]
Vector: TypeAlias = Sequence[float]

NeighborIdsReturn: TypeAlias = Ids | tuple[Ids, Dists]
NeighborVectorsMatrix: TypeAlias = list[Sequence[float]] | "np.ndarray"
NeighborVectorsReturn: TypeAlias = NeighborVectorsMatrix | tuple[NeighborVectorsMatrix, Dists]

__all__: list[str]


@runtime_checkable
class AnnoyVectorOps(Protocol):
    def get_nns_by_item(
        self,
        item: int,
        n: int,
        search_k: int = ...,
        include_distances: bool = ...,
    ) -> Ids | tuple[Ids, Dists]: ...

    def get_nns_by_vector(
        self,
        vector: Vector,
        n: int,
        search_k: int = ...,
        include_distances: bool = ...,
    ) -> Ids | tuple[Ids, Dists]: ...

    def get_item_vector(self, item: int) -> Sequence[float]: ...


class VectorOpsMixin:

    def _filter_ids(
        self,
        ids: Ids,
        dists: Dists | None,
        *,
        exclude_ids: set[int] | None = ...,
    ) -> tuple[Ids, Dists | None]: ...

    def _require_numpy(self) -> Any: ...
    def _vectors_equal_strict(self, a: Vector, b: Vector) -> bool: ...
    def _find_first_exact_match_id(self, vector: Vector, ids: Ids) -> int | None: ...

    def _neighbor_ids_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: bool = False,
    ) -> NeighborIdsReturn: ...

    def _neighbor_ids_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
    ) -> NeighborIdsReturn: ...

    @overload
    def get_neighbor_ids_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: Literal[False] = False,
    ) -> Ids: ...
    @overload
    def get_neighbor_ids_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: Literal[True] = True,
    ) -> tuple[Ids, Dists]: ...

    @overload
    def get_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: Literal[False] = False,
        as_numpy: Literal[False] = False,
        dtype: str = "float32",
    ) -> list[Sequence[float]]: ...
    @overload
    def get_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: Literal[False] = False,
        as_numpy: Literal[True],
        dtype: str = "float32",
    ) -> "np.ndarray": ...
    @overload
    def get_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: Literal[True] = True,
        as_numpy: Literal[False] = False,
        dtype: str = "float32",
    ) -> tuple[list[Sequence[float]], Dists]: ...
    @overload
    def get_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: Literal[True] = True,
        as_numpy: Literal[True] = True,
        dtype: str = "float32",
    ) -> tuple["np.ndarray", Dists]: ...

    def iter_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
    ) -> Iterator[Sequence[float]]: ...

    @overload
    def get_neighbor_ids_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: Literal[False] = False,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
    ) -> Ids: ...
    @overload
    def get_neighbor_ids_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: Literal[True] = True,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
    ) -> tuple[Ids, Dists]: ...

    @overload
    def get_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: Literal[False] = False,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
        as_numpy: Literal[False] = False,
        dtype: str = "float32",
    ) -> list[Sequence[float]]: ...
    @overload
    def get_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: Literal[False] = False,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
        as_numpy: Literal[True] = True,
        dtype: str = "float32",
    ) -> "np.ndarray": ...
    @overload
    def get_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: Literal[True] = True,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
        as_numpy: Literal[False] = False,
        dtype: str = "float32",
    ) -> tuple[list[Sequence[float]], Dists]: ...
    @overload
    def get_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: Literal[True] = True,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
        as_numpy: Literal[True] = True,
        dtype: str = "float32",
    ) -> tuple["np.ndarray", Dists]: ...

    def iter_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
    ) -> Iterator[Sequence[float]]: ...
