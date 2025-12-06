from typing import Iterable, Sequence, Tuple, List, Dict, Any, Optional, overload
import pandas as pd
from . import annoylib


class Index(annoylib.Annoy):
    """
    High-level Annoy index for approximate nearest-neighbour search.

    This class wraps the low-level C++ core
    :class:`~scikitplot.cexternals.annoy.annoylib.Annoy` and exposes a
    typed, user-friendly pickable Python interface.
    """
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
    def deserialize(cls, bytes: bytes, prefault: bool = False) -> "Index": ...

    # modern pickling (e.g., __reduce_ex__, __reduce__, _rebuild)
    def __reduce__(self) -> Any: ...
    @classmethod
    def _rebuild(cls, state: dict) -> "Index": ...

    # utilities
    def save_to_file(self, path: str) -> None: ...
    @classmethod
    def load_from_file(cls, path: str) -> "Index": ...

    # items
    def add_item(self, i: int, v: Sequence[float]) -> bool: ...
    def get_item_vector(self, i: int) -> List[float]: ...
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
        search_k: Optional[int] = ...,
        *,
        include_distances: bool = False,
    ) -> List[int]: ...

    @overload
    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: Optional[int] = ...,
        *,
        include_distances: bool = False,
    ) -> Tuple[List[int], List[float]]: ...

    @overload
    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: Optional[int] = ...,
        *,
        include_distances: bool = False,
    ) -> NNSResult: ...

    # Vector version
    @overload
    def get_nns_by_vector(
        self,
        v: Sequence[float],
        n: int,
        search_k: Optional[int] = ...,
        *,
        include_distances: bool = False,
    ) -> List[int]: ...

    @overload
    def get_nns_by_vector(
        self,
        v: Sequence[float],
        n: int,
        search_k: Optional[int] = ...,
        *,
        include_distances: bool = False,
    ) -> Tuple[List[int], List[float]]: ...

    @overload
    def get_nns_by_vector(
        self,
        v: Sequence[float],
        n: int,
        search_k: Optional[int] = ...,
        *,
        include_distances: bool = False,
    ) -> NNSResult: ...
