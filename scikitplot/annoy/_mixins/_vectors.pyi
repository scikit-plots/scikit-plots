# _vectors.pyi

from typing import Iterable, Iterator, Sequence, Union

from typing_extensions import TypeAlias

try:
    import numpy as np
except Exception:  # pragma: no cover
    np: TypeAlias = None  # type: ignore[]  # noqa: PYI042

Ids: TypeAlias = list[int]
Dists: TypeAlias = list[float]
Vector: TypeAlias = Sequence[float]

NeighborIdsReturn: TypeAlias = Ids | tuple[Ids, Dists]
NeighborVectorsMatrix: TypeAlias = Union[list[Sequence[float]], "np.ndarray"]  # type: ignore[] # noqa: PYI020
NeighborVectorsReturn: TypeAlias = (
    NeighborVectorsMatrix | tuple[NeighborVectorsMatrix, Dists]
)

class VectorOpsMixin:
    """
    High-level vector operations for Annoy-like objects.
    """  # noqa: PYI021

    # internal
    def _filter_ids(
        self,
        ids: Ids,
        dists: Dists | None = None,
        *,
        exclude_ids: set[int] | None = None,
    ) -> tuple[Ids, Dists | None]: ...
    def _vectors_equal_strict(self, a: Vector, b: Vector) -> bool: ...
    def _detect_self_candidate_id(
        self,
        vector: Vector,
        ids: Ids,
        dists: Dists | None = None,
    ) -> int | None: ...
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

    # public by-item
    def get_neighbor_ids_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: bool = False,
    ) -> NeighborIdsReturn: ...
    def get_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: bool = False,
        as_numpy: bool = False,
        dtype: str = "float32",
    ) -> NeighborVectorsReturn: ...
    def iter_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
    ) -> Iterator[Sequence[float]]: ...

    # public by-vector
    def get_neighbor_ids_by_vector(
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
    def get_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
        as_numpy: bool = False,
        dtype: str = "float32",
    ) -> NeighborVectorsReturn: ...
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
