# scikitplot/cexternals/_annoy/annoylib.pyi
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the annoy project.
# https://github.com/spotify/annoy/blob/main/annoy/__init__.pyi

# from __future__ import annotations

from typing import TYPE_CHECKING, Sized, TypeAlias, TypeVar, overload, runtime_checkable
from typing import Iterable, Iterator, List, Tuple, TypedDict
from typing_extensions import Literal, LiteralString, NotRequired, Protocol, Required, Self

# from . import annoylib

# --- Allowed metric literals (simple type hints) ---
# --- Metric typing -----------------------------------------------------------
# Annoy accepts many aliases on input, but always normalizes to a canonical
# metric name on output (see :attr:`Annoy.metric`).
# // scipy.spatial.distance.cosine
# // scipy.spatial.distance.euclidean
# // scipy.spatial.distance.cityblock
# // scipy.sparse.coo_array.dot
# // scipy.spatial.distance.hamming
# {"angular",   "angular"},
# {"cosine",    "angular"},
# {"euclidean", "euclidean"},
# {"l2",        "euclidean"},
# {"lstsq",        "euclidean"},
# {"manhattan", "manhattan"},
# {"l1",        "manhattan"},
# {"cityblock", "manhattan"},
# {"taxicab",   "manhattan"},
# {"dot",          "dot"},
# {"@",            "dot"},
# {".",            "dot"},
# {"dotproduct",   "dot"},
# {"inner",        "dot"},
# {"innerproduct", "dot"},
# {"hamming", "hamming"},
AnnoyMetric: TypeAlias = Literal[
    "angular", "cosine",
    "euclidean", "l2", "lstsq",
    "manhattan", "l1", "cityblock", "taxicab",
    "dot", "@", ".", "dotproduct", "inner", "innerproduct",
    "hamming",
]
AnnoyMetricCanonical: TypeAlias = Literal['angular', 'euclidean', 'manhattan', 'dot', 'hamming']

ItemIndex: TypeAlias = int
TreeCount: TypeAlias = int
SearchK: TypeAlias = int
Neighbors: TypeAlias = list[ItemIndex]
Distances: TypeAlias = list[float]
NeighborsWithDistances: TypeAlias = tuple[Neighbors, Distances]

# --- Generic type variable that preserves literal type ---
# AnnoyMetricT = TypeVar("AnnoyMetricT", AnnoyMetric, bound=LiteralString)
# AnnoyMetricT = TypeVar("AnnoyMetricT", bound=AnnoyMetric)


class Vector(Protocol, Sized):
    def __getitem__(self, index: int) -> float: ...
    def __len__(self) -> int: ...


class AnnoyInfo(TypedDict):
    """
    JSON-like summary returned by :meth:`~.Annoy.info`.

    Notes
    -----
    The following keys are always present:

    - ``f``, ``metric``, ``on_disk_path``, ``prefault``, ``schema_version``,
      ``seed``, ``verbose``

    The following keys are included only when requested via ``include_*`` flags:

    - ``n_items``, ``n_trees``, ``memory_usage_byte``, ``memory_usage_mib``
    """

    # Always-present keys (stable)
    f: Required[int]
    metric: Required[AnnoyMetricCanonical | None]
    on_disk_path: Required[str | None]
    prefault: Required[bool]
    schema_version: Required[int]
    seed: Required[int | None]
    verbose: Required[int | None]

    # Optional keys (controlled by include_* flags)
    n_items: NotRequired[int]
    n_trees: NotRequired[int]
    memory_usage_byte: NotRequired[int]
    memory_usage_mib: NotRequired[float]


class Annoy:
    """
    Approximate Nearest Neighbors index (Annoy) with a small, lazy C-extension wrapper.

    ::

    >>> Annoy(
    >>>     f=None,
    >>>     metric=None,
    >>>     *,
    >>>     on_disk_path=None,
    >>>     prefault=None,
    >>>     schema_version=None,
    >>>     seed=None,
    >>>     verbose=None,
    >>> )

    Parameters
    ----------
    f : int or None, optional, default=None
        Vector dimension. If ``0`` or ``None``, dimension may be inferred from the
        first vector passed to ``add_item`` (lazy mode).
        If None, treated as ``0`` (reset to default).
    metric : {"angular", "cosine", \
            "euclidean", "l2", "lstsq", \
            "manhattan", "l1", "cityblock", "taxicab", \
            "dot", "@", ".", "dotproduct", "inner", "innerproduct", \
            "hamming"} or None, optional, default=None
        Distance metric (one of 'angular', 'euclidean', 'manhattan', 'dot', 'hamming').
        If omitted and ``f > 0``, defaults to ``'angular'`` (cosine-like).
        If omitted and ``f == 0``, metric may be set later before construction.
        If None, treated as ``'angular'`` (reset to default).
    on_disk_path : str or None, optional, default=None
        Path for on-disk build/load. None if not configured.
    prefault : bool or None, optional, default=None
        If True, request page-faulting index pages into memory when loading
        (when supported by the underlying platform/backing).
        If None, treated as ``False`` (reset to default).
    schema_version : int, optional, default=None
        Reserved for future schema/version tracking. Currently stored on the
        object and reported by :meth:`~.Annoy.info`, but does not change the
        on-disk format.
        If None, treated as ``0`` (reset to default).
    seed : int or None, optional, default=None
        Non-negative integer seed. If set before the index is constructed,
        the seed is stored and applied when the C++ index is created.
    verbose : int or None, optional, default=None
        Verbosity level. Values are clamped to the range ``[-2, 2]``.
        ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.
        Logging level inspired by gradient-boosting libraries:

        * ``<= 0`` : quiet (warnings only)
        * ``1``    : info (Annoy's ``verbose=True``)
        * ``>= 2`` : debug (currently same as info, reserved for future use)

    Attributes
    ----------
    f : int, default=0
        Vector dimension. ``0`` means "unknown / lazy".
    metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'}, default="angular"
        Canonical metric name, or None if not configured yet (lazy).
    on_disk_path : str, default=""
        Path for on-disk build/load. None if not configured.
    prefault : bool, default=False
        Stored prefault flag (see :meth:`load`/`:meth:`save` prefault parameters).
    schema_version : int, default=0
        Reserved schema/version marker (stored; does not affect on-disk format).

    Notes
    -----
    * Once the underlying C++ index is created, ``f`` and ``metric`` are immutable.
      This keeps the object consistent and avoids undefined behavior.
    * The C++ index is created lazily when sufficient information is available:
      when both ``f > 0`` and ``metric`` are known, or when an operation that
      requires the index is first executed.
    * If ``f == 0``, the dimensionality is inferred from the first non-empty vector
      passed to :meth:`add_item` and is then fixed for the lifetime of the index.
    * If ``metric`` is omitted while ``f > 0``, the current behavior defaults to
      ``'angular'`` and may emit a :class:`FutureWarning`. To avoid warnings and
      future behavior changes, always pass ``metric=...`` explicitly.
    * Items must be added *before* calling :meth:`build`. After :meth:`build`, the
      index becomes read-only; to add more items, call :meth:`unbuild`, add items
      again with :meth:`add_item`, then call :meth:`build` again.
    * Very large indexes can be built directly on disk with :meth:`on_disk_build`
      and then memory-mapped with :meth:`load`.
    * :meth:`info` returns a structured summary (dimension, metric, counts, and
      optional memory usage) suitable for programmatic inspection.
    * This wrapper stores user configuration (e.g., seed/verbosity) even before the
      C++ index exists and applies it deterministically upon construction.

    Developer Notes:

    - Source of truth:
    * ``f`` (int) and ``metric_id`` (enum) describe configuration.
    * ``ptr`` is NULL when index is not constructed.
    - Invariant:
    * ``ptr != NULL`` implies ``f > 0`` and ``metric_id != METRIC_UNKNOWN``.

    See Also
    --------
    add_item : Add a vector to the index.
    build : Build the forest after adding items.
    unbuild : Remove trees to allow adding more items.
    get_nns_by_item, get_nns_by_vector : Query nearest neighbours.
    save, load : Persist the index to/from disk.
    serialize, deserialize : Persist the index to/from bytes.
    set_seed : Set the random seed deterministically.
    verbose : Set verbosity level.
    info : Return a structured summary of the current index.

    Examples
    --------
    >>> from annoy import Annoy, AnnoyIndex

    High-level API:

    >>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
    >>> from scikitplot.annoy import Annoy, AnnoyIndex, Index

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

    _schema_version: int
    _f: int
    _metric_id: int
    _prefault: bool

    # --- Core configuration (lazy-safe properties) ---------------------------
    @property
    def f(self) -> int: ...

    @f.setter
    def f(self, f: int) -> None: ...

    @property
    def metric(self) -> AnnoyMetricCanonical | None: ...

    @metric.setter
    def metric(self, metric: AnnoyMetric | None) -> None: ...

    @property
    def _on_disk_path(self) -> str | None: ...

    @property
    def on_disk_path(self) -> str | None: ...

    @on_disk_path.setter
    def on_disk_path(self, path: str | None) -> None: ...

    @property
    def prefault(self) -> bool: ...

    @prefault.setter
    def prefault(self, prefault: bool | None) -> None: ...

    @property
    def schema_version(self) -> int: ...

    @schema_version.setter
    def schema_version(self, schema_version: int | None) -> None: ...

    def __init__(
        self,
        f: int | None = None,
        metric: AnnoyMetric | None = None,
        *,
        on_disk_path: str | None = None,
        prefault: bool | None = None,
        schema_version: int | None = None,
        seed: int | None = None,
        verbose: int | None = None,
    ) -> None: ...

    def add_item(self, i: int, vector: Vector) -> Self:
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

    def build(self, n_trees: int, n_jobs: int = -1) -> Self:
        """
        Build a forest of random projection trees.

        Parameters
        ----------
        n_trees : int
            Number of trees in the forest. Larger values typically improve recall
            at the cost of slower build time and higher memory usage.

            If set to ``n_trees=-1``, trees are built dynamically until the index
            reaches approximately twice the number of items
            ``_n_nodes >= 2 * n_items``.

            Guidelines:

            * Small datasets (<10k samples): 10-20 trees.
            * Medium datasets (10k-1M samples): 20-50 trees.
            * Large datasets (>1M samples): 50-100+ trees.
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

    def deserialize(self, byte: bytes, prefault: bool | None = None) -> Self:
        """
        Restore the index from a serialized byte string.

        Parameters
        ----------
        byte : bytes
            Byte string produced by :meth:`serialize`.
        prefault : bool or None, optional, default=None
            If None, use the stored :attr:`prefault` value.
            Primarily useful on some platforms for very large indexes.

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
            If True, return a ``(indices, distances)`` tuple. Otherwise return only
            the list of indices.

        Returns
        -------
        indices : list[int] | tuple[list[int], list[float]]
            If ``include_distances=False``: list of neighbour item ids.
            If ``include_distances=True``: ``(indices, distances)``.

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
        self, vector: Vector, n: int, search_k: int = -1, include_distances: Literal[False] = False
    ) -> list[int]: ...
    @overload
    def get_nns_by_vector(
        self, vector: Vector, n: int, search_k: int, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    @overload
    def get_nns_by_vector(
        self, vector: Vector, n: int, search_k: int = -1, *, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    def get_nns_by_vector(
        self, vector: Vector, n: int, search_k: int = -1, include_distances: bool = False
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
            If True, return a ``(indices, distances)`` tuple. Otherwise return only
            the list of indices.

        Returns
        -------
        indices : list[int] | tuple[list[int], list[float]]
            If ``include_distances=False``: list of neighbour item ids.
            If ``include_distances=True``: ``(indices, distances)``.

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

    def info(self, *, include_n_items: bool = True, include_n_trees: bool = True, include_memory: bool | None = None) -> AnnoyInfo:
        """
        Return a structured summary of the index.

        This method returns a JSON-like Python dictionary that is easier to
        inspect programmatically than the legacy multi-line string format.

        Parameters
        ----------
        include_n_items : bool, optional, default=True
            If True, include ``n_items``.
        include_n_trees : bool, optional, default=True
            If True, include ``n_trees``.
        include_memory : bool or None, optional, default=None
            Controls whether memory usage fields are included.

            * ``None``: include memory usage only if the index is built.
            * ``True``: include memory usage if available (built).
            * ``False``: omit memory usage fields.

            Memory usage is computed after :meth:`build` and may be expensive for
            very large indexes.

        Returns
        -------
        info : dict
            Dictionary describing the current index state.

        Notes
        ----
        - Some keys are optional depending on include_* flags.

        Keys:

        * f : int, default=0
            Dimensionality of the index.
        * metric : str, default="angular"
            Distance metric name.
        * on_disk_path : str, default=""
            Path used for on-disk build, if configured.
        * prefault : bool, default=False
            If True, aggressively fault pages into memory during save.
            Primarily useful on some platforms for very large indexes.
        * schema_version : int, default=0
            Stored schema/version marker on this object (reserved for future use).
        * seed : int or None, optional, default=None
            Non-negative integer seed. If called before the index is constructed,
            the seed is stored and applied when the C++ index is created.
        * verbose : int or None, optional, default=None
            Verbosity level. Values are clamped to the range ``[-2, 2]``.
            ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.
            Logging level inspired by gradient-boosting libraries:

            * ``<= 0`` : quiet (warnings only)
            * ``1``    : info (Annoy's ``verbose=True``)
            * ``>= 2`` : debug (currently same as info, reserved for future use)

        Optional Keys:

        * n_items : int
            Number of items currently stored.
        * n_trees : int
            Number of built trees in the forest.
        * memory_usage_byte : int
            Approximate memory usage in bytes. Present only when requested and available.
        * memory_usage_mib : float
            Approximate memory usage in MiB. Present only when requested and available.

        See Also
        --------
        serialize : Create a binary snapshot of the index.
        deserialize : Restore from a binary snapshot.
        save : Persist the index to disk.
        load : Load the index from disk.

        Examples
        --------
        >>> info = idx.info()
        >>> info['f']
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

    def repr_info(self, *, include_n_items: bool = True, include_n_trees: bool = True, include_memory: bool | None = None) -> str:
        """
        Return a dict-like string representation with optional extra fields.

        Unlike ``__repr__``, this method can include additional fields on demand.
        Note that ``include_memory=True`` may be expensive for large indexes.
        Memory is calculated after :meth:`build`.
        """
        ...

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the Annoy index for Jupyter notebooks.

        Returns
        -------
        html : str
            HTML string (safe to embed) describing the current configuration.

        Notes
        -----
        This representation is deterministic and side-effect free. It intentionally
        avoids expensive operations such as serialization or memory-usage estimation.

        See Also
        --------
        info : Return a Python dict with configuration and metadata.
        __repr__ : Text representation.
        """
        ...

    def load(self, fn: str, prefault: bool | None = None) -> Self:
        """
        Load (mmap) an index from disk into the current object.

        Parameters
        ----------
        fn : str
            Path to a file previously created by :meth:`save` or
            :meth:`on_disk_build`.
        prefault : bool or None, optional, default=None
            If None, use the stored :attr:`prefault` value.
            Primarily useful on some platforms for very large indexes.

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
        Approximate memory usage of the index in bytes.

        Returns
        -------
        n_bytes : int or None
            Approximate number of bytes used by the index. Returns ``None`` if the
            index is not initialized or the forest has not been built yet.

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

    def save(self, fn: str, prefault: bool | None = None) -> Self:
        """
        Persist the index to a binary file on disk.

        Parameters
        ----------
        fn : str
            Path to the output file. Existing files will be overwritten.
        prefault : bool or None, optional, default=None
            If None, use the stored :attr:`prefault` value.
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
            * ``metric_id`` : int
            * ``on_disk_path`` : str | None
            * ``prefault`` : bool
            * ``schema_version`` : int
            * ``has_pending_seed`` : bool
            * ``pending_seed`` : int
            * ``has_pending_verbose`` : bool
            * ``pending_verbose`` : int
            * ``seed`` : int | None
            * ``verbose`` : int | None
            * ``data`` : bytes | None

        Notes
        -----
        If the underlying C++ index is initialized, ``data`` contains a serialized
        snapshot (see :meth:`serialize`). Otherwise, ``data`` is ``None``.

        Configuration keys like ``prefault`` and ``schema_version`` are stored on the
        Python wrapper and restored deterministically. Unknown keys are ignored.
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


@runtime_checkable
class AnnoyLike(Protocol):
    """
    Structural (duck-typed) interface for Annoy-style indexes.

    This protocol is intended for *typing* and interoperability. Any object that
    implements this interface can be accepted where an :class:`~.Annoy` instance
    is expected.

    See Also
    --------
    Annoy
        Concrete Annoy index implementation provided by this module.

    Notes
    -----
    - This protocol models the public API exposed by :class:`~.Annoy`.
    - It does **not** imply anything about internal storage, performance, or
      implementation details.
    """

    # won’t expose those internals
    # _f: int
    # _metric_id: int
    # _on_disk_path: str or None

    # --- Core configuration ---
    # Configuration surface (mirrors :class:`~.Annoy` properties)
    @property
    def f(self) -> int: ...
    @f.setter
    def f(self, f: int) -> None: ...

    @property
    def metric(self) -> AnnoyMetricCanonical | None: ...
    @metric.setter
    def metric(self, metric: AnnoyMetric | None) -> None: ...

    @property
    def _on_disk_path(self) -> str | None: ...
    @property
    def on_disk_path(self) -> str | None: ...
    @on_disk_path.setter
    def on_disk_path(self, path: str | None) -> None: ...

    @property
    def prefault(self) -> bool: ...

    @prefault.setter
    def prefault(self, prefault: bool | None) -> None: ...

    @property
    def schema_version(self) -> int: ...

    @schema_version.setter
    def schema_version(self, schema_version: int | None) -> None: ...

    # --- Build / lifecycle ---
    def set_seed(self, seed: int = 0) -> Self: ...
    def verbose(self, level: int = 1) -> Self: ...

    def add_item(self, i: ItemIndex, vector: Vector) -> Self: ...
    def build(self, n_trees: TreeCount, n_jobs: int = -1) -> Self: ...
    def unbuild(self) -> Self: ...
    def unload(self) -> Self: ...

    # --- Persistence ---
    def on_disk_build(self, fn: str) -> Self: ...
    def save(self, fn: str, prefault: bool | None = None) -> Self: ...
    def load(self, fn: str, prefault: bool | None = None) -> Self: ...
    def serialize(self) -> bytes: ...
    def deserialize(self, byte: bytes, prefault: bool | None = None) -> Self: ...

    # --- Introspection ---
    def get_n_items(self) -> int: ...
    def get_n_trees(self) -> int: ...
    def memory_usage(self) -> int | None: ...
    def info(self, *, include_n_items: bool = True, include_n_trees: bool = True, include_memory: bool | None = None) -> AnnoyInfo: ...
    def repr_info(self, *, include_n_items: bool = True, include_n_trees: bool = True, include_memory: bool | None = None) -> str: ...
    def _repr_html_(self) -> str: ...

    # --- Queries ---
    @overload
    def get_nns_by_item(
        self,
        i: ItemIndex,
        n: int,
        search_k: SearchK = -1,
        include_distances: Literal[False] = False,
    ) -> Neighbors: ...
    @overload
    def get_nns_by_item(
        self,
        i: ItemIndex,
        n: int,
        search_k: SearchK,
        include_distances: Literal[True],
    ) -> NeighborsWithDistances: ...
    def get_nns_by_item(
        self,
        i: ItemIndex,
        n: int,
        search_k: SearchK = -1,
        include_distances: bool = False,
    ) -> Neighbors | NeighborsWithDistances: ...

    @overload
    def get_nns_by_vector(
        self,
        vector: Vector,
        n: int,
        search_k: SearchK = -1,
        include_distances: Literal[False] = False,
    ) -> Neighbors: ...
    @overload
    def get_nns_by_vector(
        self,
        vector: Vector,
        n: int,
        search_k: SearchK,
        include_distances: Literal[True],
    ) -> NeighborsWithDistances: ...
    def get_nns_by_vector(
        self,
        vector: Vector,
        n: int,
        search_k: SearchK = -1,
        include_distances: bool = False,
    ) -> Neighbors | NeighborsWithDistances: ...

    def get_item_vector(self, i: ItemIndex) -> list[float]: ...
    def get_distance(self, i: ItemIndex, j: ItemIndex) -> float: ...
