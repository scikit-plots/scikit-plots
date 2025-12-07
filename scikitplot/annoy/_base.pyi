from typing import Any, Sequence, overload

from ..cexternals._annoy import Annoy

class Index(Annoy):
    """
    High-level Annoy index for approximate nearest-neighbour search.

    This class wraps the low-level C++ core
    :class:`~scikitplot.cexternals._annoy.Annoy` and exposes a
    typed, user-friendly pickable Python interface.
    """  # noqa: PYI021

    f: int
    metric: str

    def __init__(self, f: int = 0, metric: str = "angular") -> None: ...

    # Annoy core operations
    def build(self, n_trees: int = -1, n_jobs: int = -1) -> bool: ...
    def unbuild(self) -> bool: ...
    def on_disk_build(self, fn: str) -> bool: ...
    def set_seed(self, seed: int = 0) -> None: ...
    def verbose(self, v: int = 0) -> None: ...

    # persistence
    def save(self, fn: str, prefault: bool = False) -> bool: ...
    def load(self, fn: str, prefault: bool = False) -> bool: ...
    def serialize(self, prefault: bool = False) -> bytes: ...
    @classmethod
    def deserialize(
        cls,
        bytes: bytes,
        prefault: bool = False,
    ) -> "Index": ...  # noqa: PYI020, UP037

    # modern pickling (e.g., __reduce_ex__, __reduce__, _rebuild)
    def __reduce__(self) -> Any: ...
    @classmethod
    def _rebuild(cls, state: dict) -> "Index": ...  # noqa: PYI020, UP037

    # utilities
    def save_to_file(self, path: str) -> None: ...
    @classmethod
    def load_from_file(cls, path: str) -> "Index": ...  # noqa: PYI020, UP037

    # items
    def add_item(self, i: int, v: Sequence[float]) -> bool: ...
    def get_item_vector(self, i: int) -> list[float]: ...
    def get_distance(self, i: int, j: int) -> float: ...
    def get_n_items(self) -> int: ...
    def get_n_trees(self) -> int: ...
    def memory_usage(self) -> int: ...
    def __len__(self) -> int: ...
    def info(self) -> dict: ...

    # -------------------------------------------
    # Nearest neighbor queries (clean signatures)
    # -------------------------------------------

    @overload
    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: int | None = ...,
        *,
        include_distances: bool = False,
    ) -> list[int]: ...
    @overload
    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: int | None = ...,
        *,
        include_distances: bool = False,
    ) -> tuple[list[int], list[float]]: ...

    # Vector version
    @overload
    def get_nns_by_vector(
        self,
        v: Sequence[float],
        n: int,
        search_k: int | None = ...,
        *,
        include_distances: bool = False,
    ) -> list[int]: ...
    @overload
    def get_nns_by_vector(
        self,
        v: Sequence[float],
        n: int,
        search_k: int | None = ...,
        *,
        include_distances: bool = False,
    ) -> tuple[list[int], list[float]]: ...
