# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the annoy project.
# https://github.com/spotify/annoy/blob/main/annoy/__init__.pyi

from typing import TYPE_CHECKING, Sized, TypeAlias, TypeVar, overload, runtime_checkable
from typing_extensions import Literal, LiteralString, Protocol

# --- Allowed metric literals (simple type hints) ---
AnnoyMetric: TypeAlias = Literal[
    "angular", "cosine",
    "euclidean", "l2",
    "manhattan", "l1", "taxicab", "cityblock",
    "hamming",
    "dot", ".",
]

# --- Generic type variable that preserves literal type ---
# AnnoyMetricT = TypeVar("AnnoyMetricT", AnnoyMetric, bound=LiteralString)
AnnoyMetricT = TypeVar("AnnoyMetricT", bound=AnnoyMetric)

class _Vector(Protocol, Sized):
    def __getitem__(self, __index: int) -> float: ...
    def __len__(self) -> int: ...

class AnnoyIndex:
    """
    Annoy index for approximate nearest neighbor search.

    Parameters
    ----------
    f : int
        Dimensionality of the vectors.
    metric : {'angular', 'cosine', 'euclidean', 'l2', \
        'manhattan', 'l1', 'taxicab', 'cityblock', 'hamming', 'dot', '.'}
        Distance metric to use for the index.

    Notes
    -----
    Supports adding items, building trees, querying nearest neighbors,
    serializing/deserializing, and memory-mapped disk operations.
    """

    f: int
    metric: AnnoyMetric

    def __init__(self, f: int, metric: AnnoyMetric) -> None: ...

    def add_item(self, i: int, vector: _Vector) -> None:
        """
        Add a single item to the index.

        Parameters
        ----------
        i : int
            Unique non-negative integer ID for the item.
        vector : Sequence[float]
            A vector of length `f` representing the item in feature space.

        Notes
        -----
        Memory is allocated for `max(i) + 1` items.
        This function must be called before `build()`.
        """
        ...

    def build(self, n_trees: int, n_jobs: int = ...) -> Literal[True]:
        """
        Construct the forest of trees for nearest neighbor search.

        Parameters
        ----------
        n_trees : int
            Number of trees to build. More trees increase accuracy but take longer.
        n_jobs : int, optional
            Number of parallel jobs. Default uses all cores.

        Returns
        -------
        True
        """
        ...

    def deserialize(self, data: bytes, prefault: bool = ...) -> Literal[True]:
        """
        Deserialize an Annoy index from a bytes object.

        Parameters
        ----------
        data : bytes
            Byte string containing a serialized Annoy index.
        prefault : bool, default=False
            If True, memory pages are prefaulted (preloaded into RAM) to reduce page faults
            during queries, which may improve performance for large indices.

        Returns
        -------
        True

        Notes
        -----
        After deserialization, the index is ready for querying without needing to call `build()`.
        """
        ...

    def get_distance(self, __i: int, __j: int) -> float:
        """
        Compute the distance between two items in the index.

        Parameters
        ----------
        __i : int
            ID of the first item.
        __j : int
            ID of the second item.

        Returns
        -------
        float
            Distance between the two items according to the index's metric.

        Notes
        -----
        - Valid IDs must be in the range [0, get_n_items() - 1].
        - The distance metric is defined by the index's `metric` attribute.
        - For 'angular', 'euclidean', 'manhattan', 'hamming', or 'dot', the behavior
          follows standard definitions.
        """
        ...

    def get_item_vector(self, __i: int) -> list[float]:
        """
        Retrieve the vector associated with a given item ID.

        Parameters
        ----------
        __i : int
            ID of the item.

        Returns
        -------
        list[float]
            Vector representing the item in the index's feature space.

        Notes
        -----
        - The returned vector has dimensionality equal to `f`.
        - Useful for inspecting, copying, or computing custom metrics.
        """
        ...

    def get_n_items(self) -> int:
        """
        Get the total number of items currently added to the index.

        Returns
        -------
        int
            Number of items in the index.

        Notes
        -----
        - This value increases with each `add_item()` call.
        - Does not require the index to be built.
        """
        ...

    def get_n_trees(self) -> int:
        """
        Get the number of trees in the built Annoy index.

        Returns
        -------
        int
            Number of trees currently built in the index.

        Notes
        -----
        - Returns 0 if `build()` has not yet been called.
        - More trees generally increase search accuracy at the cost of memory and build time.
        """
        ...

    @overload
    def get_nns_by_item(
        self, i: int, n: int, search_k: int = ..., include_distances: Literal[False] = ...
    ) -> list[int]: ...

    @overload
    def get_nns_by_item(
        self, i: int, n: int, search_k: int, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...

    @overload
    def get_nns_by_item(
        self, i: int, n: int, search_k: int = ..., *, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]:
        """
        Retrieve the `n` nearest neighbors of a given item in the index.

        Parameters
        ----------
        i : int
            Index of the query item.
        n : int
            Number of nearest neighbors to return.
        search_k : int, optional
            Number of nodes to inspect during the search.
            Higher values increase accuracy at the cost of speed.
            Defaults to `n_trees * n` if not specified.
        include_distances : bool, default=False
            If True, returns a tuple of (neighbors, distances), where distances are floats
            representing the metric distance to each neighbor.

        Returns
        -------
        list[int] or tuple[list[int], list[float]]
            List of nearest neighbor IDs, optionally with corresponding distances.

        Notes
        -----
        - `search_k` controls the trade-off between speed and accuracy.
        - Using `include_distances=True` is helpful when you need the distance metrics
          for further computation.
        - The query item `i` itself is usually included as the closest neighbor.

        Example
        -------
        >>> index.get_nns_by_item(42, 5)
        [42, 10, 7, 88, 3]
        >>> index.get_nns_by_item(42, 5, include_distances=True)
        ([42, 10, 7, 88, 3], [0.0, 0.12, 0.34, 0.56, 0.78])
        """
        ...

    @overload
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...
    ) -> list[int]: ...

    @overload
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...

    @overload
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int = ..., *, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]:
        """
        Retrieve the `n` nearest neighbors of a given vector in the index.

        Parameters
        ----------
        vector : Sequence[float]
            Query vector of dimensionality `f`.
        n : int
            Number of nearest neighbors to return.
        search_k : int, optional
            Number of nodes to inspect during the search.
            Higher values increase accuracy at the cost of speed.
            Defaults to `n_trees * n` if not specified.
        include_distances : bool, default=False
            If True, returns a tuple of (neighbors, distances), where distances are floats
            representing the metric distance to each neighbor.

        Returns
        -------
        list[int] or tuple[list[int], list[float]]
            List of nearest neighbor IDs, optionally with corresponding distances.

        Notes
        -----
        - Use `search_k` to control the speed-accuracy trade-off.
        - This method is suitable when querying with an arbitrary vector not already in the index.
        - `include_distances=True` is useful for ranking or thresholding neighbors.

        Example
        -------
        >>> index.get_nns_by_vector([0.1, 0.2, 0.3], 3)
        [5, 12, 7]
        >>> index.get_nns_by_vector([0.1, 0.2, 0.3], 3, include_distances=True)
        ([5, 12, 7], [0.05, 0.15, 0.20])
        """
        ...

    def load(self, fn: str, prefault: bool = ...) -> Literal[True]:
        """
        Load an index from a file on disk (memory-mapped).

        Parameters
        ----------
        fn : str
            Path to the Annoy index file.
        prefault : bool, default=False
            Whether to prefault memory pages when loading the file.

        Returns
        -------
        True

        Notes
        -----
        Uses memory-mapped files for efficient sharing across multiple processes.
        The index can be queried immediately after loading.
        """
        ...

    def on_disk_build(self, fn: str) -> Literal[True]:
        """
        Build the index directly on disk instead of in memory.

        Parameters
        ----------
        fn : str
            Path to the target file where the on-disk index will be stored.

        Returns
        -------
        True

        Notes
        -----
        Useful for very large datasets that cannot fit entirely in RAM.
        The resulting file can be memory-mapped using `load()`.
        """
        ...

    def save(self, fn: str, prefault: bool = ...) -> Literal[True]:
        """
        Save the current Annoy index to disk.

        Parameters
        ----------
        fn : str
            File path to save the index.
        prefault : bool, default=False
            Whether to prefault memory pages when saving the file.

        Returns
        -------
        True

        Notes
        -----
        The saved index can later be loaded using `load()` or `deserialize()`.
        """
        ...

    def serialize(self) -> bytes:
        """
        Serialize the current Annoy index to bytes.

        Returns
        -------
        bytes
            Byte representation of the index.

        Notes
        -----
        This allows storing or transmitting the index without writing to disk.
        Can be deserialized later using `deserialize()`.
        """
        ...

    def set_seed(self, __s: int) -> None:
        """
        Set the random seed for the Annoy index.

        Parameters
        ----------
        __s : int
            Seed value for reproducibility of tree construction and search.

        Notes
        -----
        Setting the seed ensures consistent results across builds on the same dataset.
        """
        ...

    def unbuild(self) -> Literal[True]:
        """
        Remove the existing tree structure to allow adding new items.

        Returns
        -------
        True

        Notes
        -----
        After calling `unbuild()`, you must call `build()` again before running queries.
        Useful when dynamically updating the index with new items.
        """
        ...

    def unload(self) -> Literal[True]:
        """
        Unload a memory-mapped index from RAM.

        Returns
        -------
        True

        Notes
        -----
        Frees memory used by a loaded index without deleting the file on disk.
        The index can be reloaded later with `load()`.
        """
        ...

    def verbose(self, __v: bool) -> Literal[True]:
        """
        Enable or disable verbose output.

        Parameters
        ----------
        __v : bool
            Set to True to enable verbose logging of internal operations, useful for debugging.

        Returns
        -------
        True
        """
        ...
