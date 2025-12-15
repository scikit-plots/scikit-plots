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
from typing import Iterable, Iterator, List, Tuple, TypedDict
from typing_extensions import Literal, LiteralString, Protocol, Self

from . import annoylib

# --- Allowed metric literals (simple type hints) ---
AnnoyMetric: TypeAlias = Literal[
    "angular", "cosine",
    "euclidean", "l2", "lstsq",
    "manhattan", "l1", "cityblock", "taxicab",
    "dot", "@", ".", "dotproduct", "inner", "innerproduct",
    "hamming",
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


# // scipy.spatial.distance.cosine
# {"angular",   "angular"},
# {"cosine",    "angular"},
# // scipy.spatial.distance.euclidean
# {"euclidean", "euclidean"},
# {"l2",        "euclidean"},
# {"lstsq",        "euclidean"},
# // scipy.spatial.distance.cityblock
# {"manhattan", "manhattan"},
# {"l1",        "manhattan"},
# {"cityblock", "manhattan"},
# {"taxicab",   "manhattan"},
# // scipy.sparse.coo_array.dot
# {"dot",          "dot"},
# {"@",            "dot"},
# {".",            "dot"},
# {"dotproduct",   "dot"},
# {"inner",        "dot"},
# {"innerproduct", "dot"},
# // scipy.spatial.distance.hamming
# {"hamming", "hamming"},
class Annoy(annoylib.Annoy):
    """
    Annoy index for approximate nearest neighbor search.

    Parameters
    ----------
    f : int
        Dimensionality of the input vectors.
    metric : {"angular", "cosine", \
              "euclidean", "l2", "lstsq", \
              "manhattan", "l1", "cityblock", "taxicab", \
              "dot", "@", ".", "dotproduct", "inner", "innerproduct", \
              "hamming"}, optional, default='angular'
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

    >>> import random; random.seed(0)
    >>> # from annoy import AnnoyIndex
    >>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
    >>> from scikitplot.annoy import Annoy, AnnoyIndex, Index

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
    metric: AnnoyMetric | None

    @property
    def metric(self) -> AnnoyMetric | None: ...

    @metric.setter
    def metric(self, metric: AnnoyMetric) -> None: ...

    def __init__(self, f: int, metric: AnnoyMetric = "angular") -> None: ...

    def add_item(self, i: int, vector: _Vector) -> Self:
        """
        Add a single embedding vector to the index.

        Parameters
        ----------
        i : int
            Item id (index) must be non-negative.
            Ids may be non-contiguous; the index allocates up to ``max(i) + 1``.
        vector : sequence of float
            1D embedding of length ``f``. Values are converted to ``float``.
            If ``f == 0`` and this is the first item, ``f`` is inferred from
            ``vector`` and then fixed for the lifetime of this index.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Notes
        -----
        Items must be added *before* calling :meth:`build`. After building
        the forest, further calls to :meth:`add_item` are not supported.

        Examples
        --------
        >>> index.add_item(0, [1.0, 0.0, 0.0])
        >>> index.add_item(1, [0.0, 1.0, 0.0])
        >>> idx.add_item(0, [1.0, 0.0, 0.0]).add_item(1, [0.0, 1.0, 0.0])
        """
        ...

    def build(self, n_trees: int, n_jobs: int = -1 -> Self:
        """
        Build a forest of random projection trees.

        Parameters
        ----------
        n_trees : int
            Number of trees in the forest. Larger values typically improve recall
            at the cost of slower build time and higher memory usage.
        n_jobs : int, optional, default=-1
            Number of threads to use while building. ``-1`` means "auto" (use
            the implementation's default, typically all available CPU cores).

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Notes
        -----
        After :meth:`build` completes, the index becomes read-only for queries.
        To add more items, call :meth:`unbuild`, add items, and then rebuild.

        References
        ----------
        .. [1] Erik Bernhardsson, "Annoy: Approximate Nearest Neighbours in C++/Python".

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

    def deserialize(self, byte: bytes, prefault: bool = False) -> Self:
        """
        Restore the index from a serialized byte string.

        Parameters
        ----------
        byte : bytes
            Byte string produced by :meth:`serialize`.
        prefault : bool, optional, default=False
            If True, fault pages into memory while restoring.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Raises
        ------
        IOError
            If deserialization fails due to invalid or incompatible data.
        RuntimeError
            If the index is not initialized.

        See Also
        --------
        serialize : full in-memory index into a byte string.
        """
        ...

    def get_distance(self, i: int, j: int) -> float:
        """
        Return the distance between two stored items.

        Parameters
        ----------
        i, j : int
            Item ids (index) of two stored samples.

        Returns
        -------
        d : float
            Distance between items ``i`` and ``j`` under the current metric.

        Raises
        ------
        RuntimeError
            If the index is not initialized.
        IndexError
            If either index is out of range.

        Notes
        -----
        - Valid IDs must be in the range [0, get_n_items() - 1].
        - The distance metric is defined by the index's `metric` attribute.
        - For 'angular', 'euclidean', 'manhattan', 'hamming', or 'dot', the behavior
          follows standard definitions.
        """
        ...

    def get_item_vector(self, i: int) -> list[float]:
        """
        Return the stored embedding vector for a given item id.

        Parameters
        ----------
        i : int
            Item id (index) previously passed to :meth:`add_item`.

        Returns
        -------
        vector : list[float]
            Stored embedding of length ``f``.

        Raises
        ------
        RuntimeError
            If the index is not initialized.
        IndexError
            If ``i`` is out of range.

        Notes
        -----
        - The returned vector has dimensionality equal to `f`.
        - Useful for inspecting, copying, or computing custom metrics.
        """
        ...

    def get_n_items(self) -> int:
        """
        Return the number of stored items in the index.

        Returns
        -------
        n_items : int
            Number of items that have been added and are currently addressable.

        Raises
        ------
        RuntimeError
            If the index is not initialized.

        Notes
        -----
        - This value increases with each `add_item()` call.
        - Does not require the index to be built.
        """
        ...

    def get_n_trees(self) -> int:
        """
        Return the number of trees in the current forest.

        Returns
        -------
        n_trees : int
            Number of trees that have been built.

        Raises
        ------
        RuntimeError
            If the index is not initialized.

        Notes
        -----
        - Returns 0 if `build()` has not yet been called.
        - More trees generally increase search accuracy at the cost of memory and build time.
        """
        ...

    @overload
    def get_nns_by_item(
        self, i: int, n: int, search_k: int = -1, include_distances: Literal[False] = False
    ) -> list[int]: ...
    @overload
    def get_nns_by_item(
        self, i: int, n: int, search_k: int, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    @overload
    def get_nns_by_item(
        self, i: int, n: int, search_k: int = -1, *, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    def get_nns_by_item(
        self, i: int, n: int, search_k: int = -1, include_distances: bool = False
    ) -> list[int] | tuple[list[int], list[float]]:
        """
        get_nns_by_item(i, n, search_k=-1, include_distances=False)

        Return the `n` nearest neighbours for a stored item id.

        Parameters
        ----------
        i : int
            Item id (index) previously passed to :meth:`add_item(i, embedding)`.
        n : int
            Number of nearest neighbours to return.
        search_k : int, optional, default=-1
            Maximum number of nodes to inspect. Larger values usually improve recall
            at the cost of slower queries. If ``-1``, defaults to approximately
            ``n_trees * n``.
        include_distances : bool, optional, default=False
            If True, return a ``(indexs, distances)`` tuple. Otherwise return only
            the list of indexs.

        Returns
        -------
        indexs : list[int] | tuple[list[int], list[float]]
            If ``include_distances=False``: list of neighbour item ids.
            If ``include_distances=True``: ``(indexs, distances)``.

        Raises
        ------
        RuntimeError
            If the index is not initialized or has not been built.
        IndexError
            If ``i`` is out of range.

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
        self, vector: _Vector, n: int, search_k: int = -1, include_distances: Literal[False] = False
    ) -> list[int]: ...
    @overload
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    @overload
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int = -1, *, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int = -1, include_distances: bool = False
    ) -> list[int] | tuple[list[int], list[float]]:
        """
        get_nns_by_vector(vector, n, search_k=-1, include_distances=False)

        Return the `n` nearest neighbours for a query embedding.

        Parameters
        ----------
        vector : sequence of float
            Query embedding of length ``f``.
        n : int
            Number of nearest neighbours to return.
        search_k : int, optional, default=-1
            Maximum number of nodes to inspect. Larger values typically improve recall
            at the cost of slower queries. If ``-1``, defaults to approximately
            ``n_trees * n``.
        include_distances : bool, optional, default=False
            If True, return a ``(indexs, distances)`` tuple. Otherwise return only
            the list of indexs.

        Returns
        -------
        indexs : list[int] | tuple[list[int], list[float]]
            If ``include_distances=False``: list of neighbour item ids.
            If ``include_distances=True``: ``(indexs, distances)``.

        Raises
        ------
        RuntimeError
            If the index is not initialized or has not been built.
        ValueError
            If ``len(vector) != f``.

        See Also
        --------
        get_nns_by_item : Query by stored item id.

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

        This method returns a JSON-like Python dictionary that is easier to
        inspect programmatically than the legacy multi-line string format.

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
            Approximate memory usage in bytes.
        memory_usage_mib : float
            Approximate memory usage in MiB.
        on_disk_path : str | None
            Path used for on-disk build, if configured.

        Returns
        -------
        info : dict or None
            Dictionary describing the current index state.

        Examples
        --------
        >>> info = idx.info()
        >>> info['dimension']
        100
        >>> info['n_items']
        1000
        >>> info["metric"]
        "angular"
        >>> info["n_items"] >= 0
        True

        JSON export:

        >>> import json
        >>> print(json.dumps(idx.info(), indent=2))
        """
        ...

    def load(self, fn: str, prefault: bool = False) -> Self:
        """
        Load (mmap) an index from disk into the current object.

        Parameters
        ----------
        fn : str
            Path to a file previously created by :meth:`save` or
            :meth:`on_disk_build`.
        prefault : bool, optional, default=False
            If True, fault pages into memory when the file is mapped.

        Raises
        ------
        IOError
            If the file cannot be opened or mapped.
        RuntimeError
            If the index is not initialized or the file is incompatible.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Notes
        -----
        The in-memory index must have been constructed with the same dimension
        and metric as the on-disk file.
        """
        ...

    def memory_usage(self) -> int | None:
        """
        Approximate memory usage of the index in bytess.

        Returns
        -------
        n_bytess : int or None
            Approximate number of bytes used by the index. Returns ``None`` if the
            index is not initialized.

        Raises
        ------
        RuntimeError
            If memory usage cannot be computed.

        Examples
        --------
        >>> idx.build(10)
        >>> idx.memory_usage()
        611056
        """
        ...

    def on_disk_build(self, fn: str) -> Self:
        """
        Configure the index to build using an on-disk backing file.

        Parameters
        ----------
        fn : str
            Path to a file that will hold the index during build.
            The file is created or overwritten as needed.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Notes
        -----
        This mode is useful for very large datasets that do not fit
        comfortably in RAM during construction.
        """
        ...

    def save(self, fn: str, prefault: bool = False) -> Self:
        """
        Persist the index to a binary file on disk.

        Parameters
        ----------
        fn : str
            Path to the output file. Existing files will be overwritten.
        prefault : bool, optional, default=False
            If True, aggressively fault pages into memory during save.
            Primarily useful on some platforms for very large indexes.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Raises
        ------
        IOError
            If the file cannot be written.
        RuntimeError
            If the index is not initialized or save fails.

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
        Serialize the built in-memory index into a byte string.

        Returns
        -------
        data : bytes
            Opaque binary blob containing the Annoy index.

        Raises
        ------
        RuntimeError
            If the index is not initialized or serialization fails.

        Notes
        -----
        The serialized form is a snapshot of the internal C++ data structures.
        It can be stored, transmitted, or used with joblib without rebuilding trees.

        See Also
        --------
        deserialize : Restore an index from a serialized byte string.

        Examples
        --------
        >>> buf = idx.serialize()
        >>> new_idx = Annoy(100, metric="angular")
        >>> new_idx.deserialize(buf)
        >>> new_idx.get_n_items()
        1000
        """
        ...

    def set_seed(self, seed: int = 0) -> Self:
        """
        Set the random seed used for tree construction.

        Parameters
        ----------
        seed : int, optional
            Non-negative integer seed. If called before the index is constructed,
            the seed is stored and applied when the C++ index is created.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Notes
        -----
        Annoy is deterministic by default. Setting an explicit seed is useful for
        reproducible experiments and debugging.
        """
        ...

    def unbuild(self) -> Self:
        """
        Discard the current forest, allowing new items to be added.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

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

    def unload(self) -> Self:
        """
        Unmap any memory-mapped file backing this index.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Notes
        -----
        This releases OS-level resources associated with the mmap,
        but keeps the Python object alive.

        Examples
        --------
        >>> idx.unload()
        >>> idx
        Annoy(f=100, metric='angular', n_items=0, n_trees=0, on_disk_path=None)
        """
        ...

    def verbose(self, level: int = 1) -> Self:
        """
        Control verbosity of the underlying C++ index.

        Parameters
        ----------
        level : int, optional, default=1
            Verbosity level. Values are clamped to the range ``[-2, 2]``.
            ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.
            Logging level inspired by gradient-boosting libraries:

            * ``<= 0`` : quiet (warnings only)
            * ``1``    : info (Annoy's ``verbose=True``)
            * ``>= 2`` : debug (currently same as info, reserved for future use)

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.
        """
        ...

    def __getstate__(self) -> dict[str, object]:
        """
        Return a versioned state dictionary for pickle/joblib.

        This method is primarily used by :mod:`pickle` (and joblib) and is not
        intended to be called directly in normal workflows.

        Returns
        -------
        state : dict
            A JSON-like dictionary with the following keys:

            * ``_pickle_version`` : int
            * ``f`` : int | None
            * ``metric`` : str | None
            * ``on_disk_path`` : str | None
            * ``has_pending_seed`` : bool
            * ``pending_seed`` : int
            * ``has_pending_verbose`` : bool
            * ``pending_verbose`` : int
            * ``data`` : bytes | None

        Notes
        -----
        If the underlying C++ index is initialized, ``data`` contains a serialized
        snapshot (see :meth:`serialize`). Otherwise, ``data`` is ``None``.
        """
        ...

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Restore object state from a dictionary produced by :meth:`__getstate__`.

        Parameters
        ----------
        state : dict
            State dictionary returned by :meth:`__getstate__`.

        Raises
        ------
        TypeError
            If ``state`` is not a dictionary.
        ValueError
            If required fields are missing or invalid (e.g., negative ``f``).
        FileNotFoundError
            If disk fallback is required and ``on_disk_path`` does not exist.
        IOError
            If restoring from the serialized snapshot or disk file fails.

        Notes
        -----
        Restoration first attempts to deserialize the in-memory snapshot from
        ``state["data"]``. If that fails and ``on_disk_path`` is present,
        the index is loaded from disk as a compatibility fallback.
        """
        ...

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...], dict[str, object]]:
        """
        Pickle protocol support.

        This returns the standard 3-tuple ``(cls, args, state)`` used by pickle.
        Users typically do not need to call this directly.
        """
        ...

    def __reduce__(self) -> tuple[object, tuple[object, ...], dict[str, object]]:
        """
        Pickle support.

        Equivalent to :meth:`__reduce_ex__` with the default protocol.
        """
        ...
