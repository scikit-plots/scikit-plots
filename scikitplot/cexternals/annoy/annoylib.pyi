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
from typing import Iterable, Iterator, List, Tuple
from typing_extensions import Literal, LiteralString, Protocol
from typing import TypedDict

from . import annoylib

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


class AnnoyInfo(TypedDict):
    dimension: int
    metric: str
    n_items: int
    n_trees: int
    memory_usage_byte: int
    memory_usage_mib: float
    on_disk_path: str | None


class _Vector(Protocol, Sized):
    def __getitem__(self, __index: int) -> float: ...
    def __len__(self) -> int: ...


class Annoy(annoylib.Annoy):
    """
    Annoy index for approximate nearest neighbor search.

    Parameters
    ----------
    f : int
        Dimensionality of the input vectors.
    metric : {"angular", "cosine", "euclidean", "l2",
              "manhattan", "l1", "taxicab", "cityblock",
              "hamming", "dot", "."}, optional
        Distance function.  If omitted, the runtime default is
        ``"angular"`` (matching the original ``annoy`` package).
        Passing the metric explicitly is recommended to avoid future
        deprecation warnings.

    Attributes
    ----------
    f : int
        Stored vector dimensionality.
    metric : AnnoyMetric
        Metric used for all distances in this index.

    Notes
    -----
    * Items must be added *before* calling :meth:`build`.
    * After :meth:`build` the index becomes read-only; to add more items,
      call :meth:`unbuild`, :meth:`add_item` again, then :meth:`build`.
    * Large indices can be built directly on disk with
      :meth:`on_disk_build`, then memory-mapped with :meth:`load`.
    * :meth:`info` returns a multi-line summary including dimension,
      metric, number of items, number of trees and memory usage - the
      same values printed in ``test.ipynb``.
    * If ``f == 0`` you may add the first item with any non-empty vector and the
      dimensionality will be inferred from that vector and fixed for the lifetime
      of the index.
    * The default metric may change in future versions; to avoid warnings and
      behaviour changes, always pass ``metric=...`` explicitly.

    Examples
    --------
    The lifecycle follows the examples in ``test.ipynb``:

    1. **Construct the index**

    >>> from annoy import Annoy
    >>> idx = Annoy(f=3, metric="angular")
    >>> idx.f, idx.metric
    (3, 'angular')

    If you pass ``f=0`` the dimension can be inferred on the first
    call to :meth:`add_item`.

    2. **Add items**

    >>> idx.add_item(0, [1.0, 0.0, 0.0])
    >>> idx.add_item(1, [0.0, 1.0, 0.0])
    >>> idx.add_item(2, [0.0, 0.0, 1.0])
    >>> idx.get_n_items()
    3

    3. **Build the forest**

    >>> idx.build(n_trees=10)
    >>> idx.get_n_trees()
    10
    >>> idx.memory_usage()  # byte
    543076

    After :meth:`build` the index becomes read-only.  You can still
    query, save, load and serialize it.

    4. **Query neighbours**

    By stored item id:

    >>> idx.get_nns_by_item(0, 5)
    [0, 1, 2, ...]

    With distances:

    >>> idx.get_nns_by_item(0, 5, include_distances=True)
    ([0, 1, 2, ...], [0.0, 1.22, 1.26, ...])

    Or by an explicit query vector:

    >>> idx.get_nns_by_vector([0.1, 0.2, 0.3], 5, include_distances=True)
    ([103, 71, 160, 573, 672], [...])

    5. **Persistence**

    To work with memory-mapped indices on disk:

    >>> idx.save("annoy_test.annoy")
    >>> idx2 = Annoy(f=100, metric="angular")
    >>> idx2.load("annoy_test.annoy")
    >>> idx2.get_n_items()
    1000

    Or via raw byte:

    >>> buf = idx.serialize()
    >>> new_idx = Annoy(f=100, metric="angular")
    >>> new_idx.deserialize(buf)
    >>> new_idx.get_n_items()
    1000

    You can release OS resources with :meth:`unload` and drop the
    current forest with :meth:`unbuild`.
    """

    f: int
    metric: AnnoyMetric

    def __init__(self, f: int, metric: AnnoyMetric = "angular") -> None: ...

    def add_item(self, i: int, vector: _Vector) -> None:
        """
        Add a single embedding vector to the index.

        Parameters
        ----------
        i : int
            Non-negative integer identifier for this row.  Annoy
            internally allocates storage up to ``max(i) + 1``.
        vector : Sequence[float]
            1D embedding of length ``f``.  If ``f == 0`` on the first
            call, the dimensionality is inferred from ``vector`` and
            fixed for the lifetime of the index.

        Notes
        -----
        Items must be added *before* calling :meth:`build`.  After
        building the forest, further calls to :meth:`add_item` are not
        supported.

        Examples
        --------
        >>> idx = Annoy(f=3, metric="angular")
        >>> idx.add_item(0, [1.0, 0.0, 0.0])
        >>> idx.add_item(1, [0.0, 1.0, 0.0])
        >>> idx.get_n_items()
        2
        """
        ...

    def build(self, n_trees: int, n_jobs: int = ...) -> Literal[True]:
        """
        Build a forest of random projection trees.

        Parameters
        ----------
        n_trees : int
            Number of trees in the forest.  Larger values typically
            improve recall at the cost of slower build and query time.
        n_jobs : int, optional
            Number of threads to use while building.  ``-1`` means
            “use all available CPU cores”.

        Returns
        -------
        True
            Indicates that the build completed successfully.

        Notes
        -----
        After :meth:`build` completes, the index becomes read-only for
        queries.  To add more items, call :meth:`unbuild`, add items
        again, and then rebuild.

        Examples
        --------
        >>> idx = Annoy(f=100, metric="angular")
        >>> # add 1000 random Gaussian vectors
        >>> for i in range(1000):
        ...     v = [random.gauss(0, 1) for _ in range(100)]
        ...     idx.add_item(i, v)
        >>> idx.build(10)
        >>> idx.get_n_trees()
        10
        >>> idx.memory_usage()  # byte, as printed in test.ipynb
        543076
        """
        ...

    def deserialize(self, byte: bytes, prefault: bool = ...) -> Literal[True]:
        """
        Restore the index from a serialized byte string.

        Parameters
        ----------
        byte : bytes
            Byte string produced by :meth:`serialize`.
        prefault : bool, default=False
            If True, fault pages into memory while restoring.

        Returns
        -------
        success : bool
            ``True`` on success. On failure an exception is raised.

        See Also
        --------
        serialize : full in-memory index into a byte string.
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
        Return the ``n`` nearest neighbours for a stored item.

        Parameters
        ----------
        i : int
            Row identifier previously passed to :meth:`add_item`.
        n : int
            Number of nearest neighbours to return.
        search_k : int, optional
            Maximum number of nodes to inspect.  Larger values usually
            improve recall at the cost of slower queries.  If ``-1``,
            defaults to approximately ``n_trees * n``.
        include_distances : bool, default=False
            If False, return only the list of neighbour indices.
            If True, return a 2-tuple ``(indices, distances)``.

        Returns
        -------
        indices : list[int]
            Nearest neighbour indices (item ids).
        distances : list[float]
            Corresponding distances (only if ``include_distances=True``).

        See Also
        --------
        get_nns_by_vector : Query with an explicit query embedding.

        Examples
        --------
        High-level “result” mode with distances (as in ``test.ipynb``):

        >>> res = idx.get_nns_by_item(0, 5, include_distances=True)
        >>> res
        ([0, 165, 67, 908, 745],
         [0.0003169..., 1.2278156..., 1.2642685..., 1.2750054..., 1.2758896...])

        Low-level mode without distances:

        >>> items = idx.get_nns_by_item(0, 2, include_distances=False)
        >>> items
        [0, 165]

        Explicit tuple unpacking:

        >>> items, dists = idx.get_nns_by_item(0, 2, include_distances=True)
        >>> items
        [0, 165]
        >>> dists
        [0.0003169..., 1.2278156...]
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
        Return the ``n`` nearest neighbours for a query embedding.

        Parameters
        ----------
        vector : Sequence[float]
            Query embedding of length ``f``.
        n : int
            Number of nearest neighbours to return.
        search_k : int, optional
            Maximum number of nodes to inspect.  Larger values typically
            improve recall at the cost of slower queries.  If ``-1``,
            defaults to approximately ``n_trees * n``.
        include_distances : bool, default=False
            If False, return only the list of neighbour indices.
            If True, return a 2-tuple ``(indices, distances)``.

        Returns
        -------
        list[int] or (list[int], list[float])
            Nearest neighbour indices, optionally with distances.

        See Also
        --------
        get_nns_by_item : Main API for item-id based queries.

        Examples
        --------
        From ``test.ipynb``:

        >>> res = idx.get_nns_by_vector(
        ...     [random.gauss(0, 1) for _ in range(100)],
        ...     5,
        ...     include_distances=True,
        ... )
        >>> res
        ([103, 71, 160, 573, 672],
         [1.2087514..., 1.2708953..., 1.2750196..., 1.2815759..., 1.2957668...])
        """
        ...

    def info(self) -> AnnoyInfo:
        """
        Return a structured summary of the index.

        This method returns a JSON-like dictionary with the most useful
        state fields for debugging, monitoring and tooling.

        Keys
        ----
        dimension : int
            Dimensionality of the index.
        metric : str
            Distance metric name.
        n_items : int
            Number of items currently stored.
        n_trees : int
            Number of trees built.
        memory_usage_byte : int
            Approximate memory usage in byte.
        memory_usage_mib : float
            Approximate memory usage in MiB.
        on_disk_path : str | None
            Path used for on-disk build, if configured.

        Returns
        -------
        info : AnnoyInfo
            Dictionary describing the current index state.

        Examples
        --------
        >>> info = idx.info()
        >>> info["dimension"]
        100
        >>> info["metric"]
        "angular"
        >>> info["n_items"] >= 0
        True

        JSON export:

        >>> import json
        >>> print(json.dumps(idx.info(), indent=2))
        """
        ...

    def load(self, fn: str, prefault: bool = ...) -> Literal[True]:
        """
        Load (memory-map) an index from disk into the current object.

        Parameters
        ----------
        fn : str
            Path to the Annoy index file.
        prefault : bool, default=False
            If True, fault pages into memory when the file is mapped.

        Returns
        -------
        True

        Notes
        -----
        Annoy index files are not guaranteed to be binary compatible
        across builds that change the internal index type or node
        layout (for example, changing the template parameter ``S`` from
        32-bit to 64-bit). If you upgrade to such a version and see the
        error "Index size is not a multiple of vector size", you will
        need to rebuild your index files with the new version.

        Notes
        -----
        Uses memory-mapped files for efficient sharing across multiple processes.
        The index can be queried immediately after loading.
        """
        ...

    def memory_usage(self) -> int:
        """
        Approximate memory usage of the index in byte.

        Returns
        -------
        n_byte : int
            Approximate number of byte used by the index.  When native
            support is unavailable, this is estimated via
            :meth:`serialize`.

        Examples
        --------
        >>> idx.build(10)
        >>> idx.memory_usage()
        611056
        """
        ...

    def on_disk_build(self, fn: str) -> Literal[True]:
        """
        Configure the index to build using an on-disk backing file instead of in memory.

        Parameters
        ----------
        fn : str
            Path to a file that will hold the index during build.
            The file is created or overwritten as needed.

        Returns
        -------
        success : bool
            ``True`` on success. On failure an exception is raised.

        Notes
        -----
        Useful for very large datasets that cannot fit entirely in RAM.
        The resulting file can be memory-mapped using `load()`.
        """
        ...

    def save(self, fn: str, prefault: bool = ...) -> Literal[True]:
        """
        Persist the index to a binary file on disk.

        Parameters
        ----------
        fn : str
            Path to the output file.  Existing files will be overwritten.
        prefault : bool, default=False
            If True, aggressively fault pages into memory during save.
            Mainly useful on some platforms for very large indexes.

        Returns
        -------
        success : bool
            ``True`` on success. On failure an exception is raised.

        Notes
        -----
        Annoy index files are not guaranteed to be binary compatible
        across builds that change the internal index type or node
        layout (for example, changing the template parameter ``S`` from
        32-bit to 64-bit). If you upgrade to such a version and see the
        error "Index size is not a multiple of vector size", you will
        need to rebuild your index files with the new version.

        Examples
        --------
        >>> idx.save("annoy_test.annoy")
        >>> idx2 = Annoy(100, metric="angular")
        >>> idx2.load("annoy_test.annoy")
        >>> idx2.get_n_items()
        1000
        """
        ...

    def serialize(self) -> bytes:
        """
        Serialize the full in-memory index into a byte string.

        Returns
        -------
        byte : bytes
            Opaque binary blob containing the entire Annoy index.

        Notes
        -----
        The serialized form is a snapshot of the internal C++ data
        structures. It can be stored, transmitted or used with joblib
        without rebuilding trees.

        See Also
        --------
        deserialize : restore an index from byte.

        Examples
        --------
        >>> buf = idx.serialize()
        >>> new_idx = Annoy(100, metric="angular")
        >>> new_idx.deserialize(buf)
        >>> new_idx.get_n_items()
        1000
        """
        ...

    def set_seed(self, seed: int | None = None) -> None:
        """
        Set the random seed used for tree construction.

        Parameters
        ----------
        seed : int, optional
            Non-negative integer seed. If omitted, a library-specific
            default is used. For strict reproducibility, always call
            this method explicitly before :meth:`build`.

        Notes
        -----
        Using the same seed, data and ``n_trees`` usually produces
        bitwise-identical forests (subject to CPU / threading details).
        """
        ...

    def unbuild(self) -> Literal[True]:
        """
        Discard the current forest, allowing new items to be added.

        Returns
        -------
        True

        Notes
        -----
        After calling :meth:`unbuild`, you must call :meth:`build`
        again before running nearest-neighbour queries.

        Examples
        --------
        >>> idx.unbuild()
        >>> idx.get_n_trees()
        0
        """
        ...

    def unload(self) -> Literal[True]:
        """
        Unmap any memory-mapped file backing this index.

        Returns
        -------
        True

        Examples
        --------
        >>> idx.unload()
        >>> idx
        Annoy(f=100, metric='angular', n_items=0, n_trees=0, on_disk_path=None)
        """
        ...

    def verbose(self, __v: bool) -> Literal[True]:
        """
        Control verbosity of the underlying C++ index.

        Parameters
        ----------
        level : int, optional (default=1)
            Logging level inspired by gradient-boosting libraries:

            * ``<= 0`` : quiet (warnings only)
            * ``1``    : info (Annoy's ``verbose=True``)
            * ``>= 2`` : debug (currently same as info, reserved
              for future use)

        Returns
        -------
        True
        """
        ...
