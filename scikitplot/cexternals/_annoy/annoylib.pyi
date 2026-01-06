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

"""
Typing stubs for the Annoy C-extension wrapper.

This module is derived from the upstream Annoy project stubs and extended to
match the API implemented by this repository's C-extension (see the corresponding
``annoymodule.cc`` file exposing ``PyMethodDef`` entries).

Notes
-----
- This is a ``annoylib.pyi`` stub: it contains signatures and docstrings only.
- Keep signatures and docstrings in sync with the C-extension's public API.

Examples
--------
The public import path is stable and should be used as-is:

>>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex

Build an index from vectors (``fit``) and query neighbors (``transform``):

>>> idx = Annoy(f=128, metric='angular', n_neighbors=10)
>>> idx.fit(X, n_trees=10)
>>> ids = idx.transform(X_query, input_type='vector', output_type='item', n_neighbors=10)

Query by stored item ids (and exclude the query id itself):

>>> ids = idx.transform([0, 1], input_type='item', output_type='item', n_neighbors=10, exclude_self=True)

Return neighbor vectors instead of ids:

>>> vecs = idx.transform([0], input_type='item', output_type='vector', n_neighbors=5, exclude_self=True)

Exclude explicit ids for vector queries (recommended; do not infer "self" from distance):

>>> ids = idx.transform(X_query, input_type='vector', output_type='item', n_neighbors=10, exclude_items=[0, 1])

Tip: set ``include_distances=True`` and/or ``return_labels=True`` to return distances and labels alongside neighbors.
"""

# from __future__ import annotations
# from . import annoylib

from typing import Any, Mapping, Sequence, Sized, TypedDict, overload, runtime_checkable
from typing_extensions import Literal, NotRequired, Protocol, Required, Self, TypeAlias

__backend__: Literal["cpp"]

# Annoy accepts many aliases on input, but normalizes to a canonical metric name
# on output (see :attr:`Annoy.metric`).
AnnoyMetric: TypeAlias = Literal[
    # Canonical + common aliases
    "angular", "cosine",
    "euclidean", "l2", "lstsq",
    "manhattan", "l1", "cityblock", "taxicab",
    "dot", "@", ".", "dotproduct", "inner", "innerproduct",
    "hamming",
]
AnnoyMetricCanonical: TypeAlias = Literal[
    "angular", "euclidean", "manhattan", "dot", "hamming"
]

SerializeFormat: TypeAlias = Literal["native", "portable", "canonical"]
TransformInputType: TypeAlias = Literal["item", "vector"]
TransformOutputType: TypeAlias = Literal["item", "vector"]

ItemIndex: TypeAlias = int
TreeCount: TypeAlias = int
SearchK: TypeAlias = int

Neighbors: TypeAlias = list[ItemIndex]
Distances: TypeAlias = list[float]
NeighborsWithDistances: TypeAlias = tuple[Neighbors, Distances]

NeighborLabels: TypeAlias = list[Any]
Indices2D: TypeAlias = list[list[int]]
Distances2D: TypeAlias = list[list[float]]
Labels2D: TypeAlias = list[list[Any]]
NeighborVectors3D: TypeAlias = list[list[list[float]]]

TransformItemOutput: TypeAlias = (
    Indices2D
    | tuple[Indices2D, Distances2D]
    | tuple[Indices2D, Labels2D]
    | tuple[Indices2D, Distances2D, Labels2D]
)

TransformVectorOutput: TypeAlias = (
    NeighborVectors3D
    | tuple[NeighborVectors3D, Distances2D]
    | tuple[NeighborVectors3D, Labels2D]
    | tuple[NeighborVectors3D, Distances2D, Labels2D]
)

TransformOutput: TypeAlias = TransformItemOutput | TransformVectorOutput

# --- Generic type variable that preserves literal type ---
# AnnoyMetricT = TypeVar("AnnoyMetricT", AnnoyMetric, bound=LiteralString)
# AnnoyMetricT = TypeVar("AnnoyMetricT", bound=AnnoyMetric)


class Vector(Protocol, Sized):
    """1D, indexable float vector."""

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
    n_neighbors: Required[int]
    on_disk_path: Required[str | None]
    prefault: Required[bool]
    seed: Required[int | None]
    verbose: Required[int | None]
    schema_version: Required[int]

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
    >>>     n_neighbors=5,
    >>>     on_disk_path=None,
    >>>     prefault=None,
    >>>     seed=None,
    >>>     verbose=None,
    >>>     schema_version=None,
    >>> )

    Parameters
    ----------
    f : int or None, optional, default=None
        Vector dimension. If ``0`` or ``None``, dimension may be inferred from the
        first vector passed to ``add_item`` (lazy mode).
        If None, treated as ``0`` (reset to default).
    metric : {"angular", "cosine", "euclidean", "l2", "lstsq", "manhattan", "l1", "cityblock", "taxicab", \
            "dot", "@", ".", "dotproduct", "inner", "innerproduct", "hamming"} or None, optional, default=None
        Distance metric (one of 'angular', 'euclidean', 'manhattan', 'dot', 'hamming').
        If omitted and ``f > 0``, defaults to ``'angular'`` (cosine-like).
        If omitted and ``f == 0``, metric may be set later before construction.
        If None, behavior depends on ``f``:

        * If ``f > 0``: defaults to ``'angular'`` (legacy behavior; may emit a
        :class:`FutureWarning`).
        * If ``f == 0``: leaves the metric unset (lazy). You may set
        :attr:`metric` later before construction, or it will default to
        ``'angular'`` on first :meth:`add_item`.
    n_neighbors : int, default=5
        Non-negative integer Number of neighbors to retrieve for each query.
    on_disk_path : str or None, optional, default=None
        If provided, configures the path for on-disk building. When the underlying
        index exists, this enables on-disk build mode (equivalent to calling
        :meth:`on_disk_build` with the same filename).

        Note: Annoy core truncates the target file when enabling on-disk build.
        This wrapper treats ``on_disk_path`` as strictly equivalent to calling
        :meth:`on_disk_build` with the same filename (truncate allowed).

        In lazy mode (``f==0`` and/or ``metric is None``), activation occurs once
        the underlying C++ index is created.
    prefault : bool or None, optional, default=None
        If True, request page-faulting index pages into memory when loading
        (when supported by the underlying platform/backing).
        If None, treated as ``False`` (reset to default).
    seed : int or None, optional, default=None
        Non-negative integer seed. If set before the index is constructed,
        the seed is stored and applied when the C++ index is created.
        Seed value ``0`` is treated as \"use Annoy's deterministic default seed\"
        (a :class:`UserWarning` is emitted when ``0`` is explicitly provided).
    verbose : int or None, optional, default=None
        Verbosity level. Values are clamped to the range ``[-2, 2]``.
        ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.
        Logging level inspired by gradient-boosting libraries:

        * ``<= 0`` : quiet (warnings only)
        * ``1``    : info (Annoy's ``verbose=True``)
        * ``>= 2`` : debug (currently same as info, reserved for future use)
    schema_version : int, optional, default=None
        Serialization/compatibility strategy marker.

        This does not change the Annoy on-disk format, but it *does* control
        how the index is snapshotted in pickles.

        * ``0`` or ``1``: pickle stores a ``portable-v1`` snapshot (fast restore,
        ABI-checked).
        * ``2``: pickle stores ``canonical-v1`` (portable across ABIs; restores by
        rebuilding deterministically).
        * ``>=3``: pickle stores both portable and canonical (canonical is used as
        a fallback if the ABI check fails).

        If None, treated as ``0`` (reset to default).

    Attributes
    ----------
    f : int, default=0
        Vector dimension. ``0`` means "unknown / lazy".
    metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'}, default="angular"
        Canonical metric name, or None if not configured yet (lazy).
    n_neighbors : int, default=5
        Non-negative integer Number of neighbors to retrieve for each query.
    on_disk_path : str or None, optional, default=None
        Configured on-disk build path. Setting this attribute enables on-disk
        build mode (equivalent to :meth:`on_disk_build`), with safety checks
        to avoid implicit truncation of existing files.
    seed, random_state : int or None, optional, default=None
        Non-negative integer seed.
    verbose : int or None, optional, default=None
        Verbosity level.
    prefault : bool, default=False
        Stored prefault flag (see :meth:`load`/`:meth:`save` prefault parameters).
    schema_version : int, default=0
        Reserved schema/version marker (stored; does not affect on-disk format).
    n_features, n_features_, n_features_in_ : int
        Alias of `f` (dimension), provided for scikit-learn naming parity.
    n_features_out_ : int
        Number of output features produced by transform.
    feature_names_in_ : list-like
        Input feature names seen during fit.
        Set only when explicitly provided via fit(..., feature_names=...).
    y : dict | None, optional, default=None
        If provided to fit(X, y), labels are stored here after a successful build.
        You may also set this property manually. When possible, the setter enforces
        that len(y) matches the current number of items (n_items).

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

    Notes
    -----
    * Once the underlying C++ index is created, ``f`` and ``metric`` are immutable.
    This keeps the object consistent and avoids undefined behavior.
    * The C++ index is created lazily when sufficient information is available:
    when both ``f > 0`` and ``metric`` are known, or when an operation that
    requires the index is first executed.
    * If ``f == 0``, the dimensionality is inferred from the first non-empty vector
    passed to :meth:`add_item` and is then fixed for the lifetime of the index.
    * Assigning ``None`` to :attr:`f` is not supported. Use ``0`` for lazy
    inference (this matches ``Annoy(f=None, ...)`` at construction time).
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

    # ---- Internal fields (implementation detail; present for type-checkers) ---
    _f: int
    _metric_id: int
    _prefault: bool
    _schema_version: int

    # ---- Core configuration (lazy-safe properties) ---------------------------
    @property
    def f(self) -> int: ...
    @f.setter
    def f(self, f: int) -> None: ...

    @property
    def metric(self) -> AnnoyMetricCanonical | None: ...
    @metric.setter
    def metric(self, metric: AnnoyMetric | None) -> None: ...

    @property
    def n_neighbors(self) -> int: ...
    @n_neighbors.setter
    def n_neighbors(self, n_neighbors: int) -> None: ...

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
    def seed(self) -> int | None: ...
    @seed.setter
    def seed(self, seed: int | None) -> None: ...

    @property
    def random_state(self) -> int | None: ...
    @random_state.setter
    def random_state(self, seed: int | None) -> None: ...

    @property
    def verbose(self) -> int | None: ...
    @verbose.setter
    def verbose(self, level: int | None) -> None: ...

    @property
    def schema_version(self) -> int: ...
    @schema_version.setter
    def schema_version(self, schema_version: int | None) -> None: ...

    @property
    def n_features_(self) -> int: ...
    @property
    def n_features_in_(self) -> int: ...
    @property
    def n_features(self) -> int: ...
    @n_features.setter
    def n_features(self, f: int) -> None: ...

    @property
    def n_features_out_(self) -> int: ...

    @property
    def feature_names_in_(self) -> list[str]: ...

    # Stored labels y (optional, scikit-learn compatible).
    @property
    def _y(self) -> dict | None: ...
    @property
    def y(self) -> dict | None: ...
    @y.setter
    def y(self, value: dict | None) -> None: ...

    def __init__(
        self,
        f: int | None = None,
        metric: AnnoyMetric | None = None,
        *,
        n_neighbors: int = 5,
        on_disk_path: str | None = None,
        prefault: bool | None = None,
        seed: int | None = None,
        verbose: int | None = None,
        schema_version: int | None = None,
    ) -> None: ...

    def __len__(self) -> int: ...

    # ------------------------------------------------------------------
    # Core index construction / mutation
    # ------------------------------------------------------------------

    def add_item(self, i: int, vector: Vector) -> Self:
        """
        Add a single embedding vector to the index ``add_item(i, vector)``.

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

        See Also
        --------
        build : Build the forest after adding items.
        unbuild : Remove trees to allow adding more items.
        get_nns_by_item, get_nns_by_vector : Query nearest neighbours.

        Notes
        -----
        Items must be added *before* calling :meth:`build`. After building
        the forest, further calls to :meth:`add_item` are not supported.

        Examples
        --------
        >>> import random
        >>> from scikitplot.cexternals._annoy import AnnoyIndex
        >>> f = 100
        >>> n = 1000
        >>> idx = AnnoyIndex(f, metric="l2")
        >>> for i in range(n):
        ...     v = [random.gauss(0, 1) for _ in range(f)]
        ...     idx.add_item(i, v)
        """
        ...

    def build(self, n_trees: int, n_jobs: int = -1) -> Self:
        """
        Build a forest of random projection trees ``build(n_trees, n_jobs=-1)``.

        Parameters
        ----------
        n_trees : int
            Number of trees in the forest. Larger values typically improve recall
            at the cost of slower build time and higher memory usage.

            If set to ``n_trees=-1``, trees are built dynamically until
            the index reaches approximately twice the number of items
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

        See Also
        --------
        add_item : Add vectors before building.
        unbuild : Drop trees to add more items.
        get_nns_by_item, get_nns_by_vector : Query nearest neighbours.
        save, load : Persist the index to/from disk.

        Notes
        -----
        After :meth:`build` completes, the index becomes read-only for queries.
        To add more items, call :meth:`unbuild`, add items, and then rebuild.

        References
        ----------
        .. [1] Erik Bernhardsson, "Annoy: Approximate Nearest Neighbours in C++/Python".

        Examples
        --------
        >>> import random
        >>> from scikitplot.cexternals._annoy import AnnoyIndex
        >>> f = 100
        >>> n = 1000
        >>> idx = AnnoyIndex(f, metric="l2")
        >>> for i in range(n):
        ...     v = [random.gauss(0, 1) for _ in range(f)]
        ...     idx.add_item(i, v)
        >>> idx.build(10)
        """
        ...

    def deserialize(self, byte: bytes, prefault: bool | None = None) -> Self:
        """
        Restore the index from a serialized byte string ``deserialize(byte, prefault=None)``.

        Parameters
        ----------
        byte : bytes
            Byte string produced by :meth:`serialize`. Both native (legacy)
            blobs and portable blobs (created with ``serialize(format='portable')``)
            are accepted; portable and canonical blobs are auto-detected.
            Canonical blobs restore by rebuilding the index deterministically.
        prefault : bool or None, optional, default=None
            Accepted for API symmetry with :meth:`load`.
            If None, the stored :attr:`prefault` value is used.
            Ignored for canonical blobs.

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

        Notes
        -----
        Portable blobs add a small header (version, ABI sizes, endianness, metric, f)
        to ensure incompatible binaries fail loudly and safely. They are not a
        cross-architecture wire format; the payload remains Annoy's native snapshot.
        """
        ...

    def fit(
        self,
        X: Sequence[Sequence[Any]] | None = None,
        y: Sequence[Any] | None = None,
        *,
        n_trees: int = -1,
        n_jobs: int = -1,
        reset: bool = True,
        start_index: int | None = None,
        missing_value: float | None = None,
        feature_names: list[str] | None = None,
    ) -> Self:
        """
        Fit the Annoy index (scikit-learn compatible).

        fit(X=None, y=None, *, n_trees=-1, n_jobs=-1, reset=True, start_index=None, missing_value=None, feature_names=None)

        This method supports two deterministic workflows:

        1) Manual add/build:
           If X is None and y is None, fit() builds the forest using items
           previously added via add_item().

        2) Array-like X:
           If X is provided (2D array-like), fit() optionally resets or appends,
           adds all rows as items, then builds the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Vectors to add to the index. If None (and y is None), fit() only builds.
        y : array-like of shape (n_samples,), default=None
            Optional labels associated with X. Stored as y after successful build.
        n_trees : int, default=-1
            Number of trees to build. Use -1 for Annoy's internal default.
        n_jobs : int, default=-1
            Number of threads to use during build (-1 means "auto").
        reset : bool, default=True
            If True, clear existing items before adding X. If False, append.
        start_index : int or None, default=None
            Item id for the first row of X. If None, uses 0 when reset=True,
            otherwise uses current n_items when reset=False.
        missing_value : float or None, default=None
            If not None, imputes missing entries in X.

            - Dense rows: replaces None elements with missing_value.
            - Dict rows: fills missing keys (and None values) with missing_value.

            If None, missing entries raise an error (strict mode).
        feature_names : sequence of str or None, optional, default=None
            Provide feature names :attr:`feature_names_in_`
            and the expected input dimensionality.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        See Also
        --------
        add_item : Add one item at a time.
        build : Build the forest after manual calls to add_item.
        unbuild : Remove trees so items can be appended.
        get_params, set_params : Estimator parameter API.
        y : Stored labels (if provided).
        """
        ...

    def fit_transform(
        self,
        X: Sequence[Sequence[Any]],
        y: Sequence[Any] | None = None,
        *,
        n_trees: int = -1,
        n_jobs: int = -1,
        reset: bool = True,
        start_index: int | None = None,
        missing_value: float | None = None,
        feature_names: list[str] | None = None,
        n_neighbors: int = 5,
        search_k: int = -1,
        include_distances: bool = False,
        return_labels: bool = False,
        y_fill_value: Any = None,
    ) -> TransformOutput:
        """fit_transform(X, y=None, *, n_trees=-1, n_jobs=-1, reset=True, start_index=None, \
            missing_value=None, feature_names=None, n_neighbors=None, search_k=-1, \
            include_distances=False, return_labels=False, y_fill_value=None)

        Fit the index and transform X in a single deterministic call.

        This is equivalent to:
            self.fit(X, y=y, n_trees=..., n_jobs=..., reset=..., start_index=..., missing_value=...) \
            self.transform(X, n_neighbors=..., search_k=..., include_distances=..., return_labels=..., \
            y_fill_value=..., input_type='vector', output_type='item', exclude_items=..., missing_value=...)

        See Also
        --------
        fit : Build the index.
        transform : Query the built index.
        """
        ...

    def get_distance(self, i: int, j: int) -> float:
        """
        Return the distance between two stored items ``get_distance(i, j) -> float``.

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
        Return the stored embedding vector for a given item id ``get_item_vector(i) -> list[float]``.

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
        Return the number of stored items in the index ``get_n_items() -> int``.

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
        Return the number of trees in the current forest ``get_n_trees() -> int``.

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

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    @overload
    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: int = -1,
        include_distances: Literal[False] = False,
    ) -> list[int]: ...
    @overload
    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: int = -1,
        include_distances: Literal[True] = True,
    ) -> tuple[list[int], list[float]]: ...
    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: int = -1,
        include_distances: bool = False
    ) -> list[int] | tuple[list[int], list[float]]:
        """
        get_nns_by_item(i, n, search_k=-1, include_distances=False).

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
        self,
        vector: Vector,
        n: int,
        search_k: int = -1,
        include_distances: Literal[False] = False,
    ) -> list[int]: ...
    @overload
    def get_nns_by_vector(
        self,
        vector: Vector,
        n: int,
        search_k: int,
        include_distances: Literal[True] = True,
    ) -> tuple[list[int], list[float]]: ...
    def get_nns_by_vector(
        self,
        vector: Vector,
        n: int,
        search_k: int = -1,
        include_distances: bool = False,
    ) -> list[int] | tuple[list[int], list[float]]:
        """
        get_nns_by_vector(vector, n, search_k=-1, include_distances=False).

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

    # ------------------------------------------------------------------
    # scikit-learn compatible API
    # ------------------------------------------------------------------

    def get_params(self, deep=True) -> dict:
        """
        Return estimator-style parameters (scikit-learn compatibility) ``get_params(deep=True) -> dict``.

        Parameters
        ----------
        deep : bool, optional, default=True
            Included for scikit-learn API compatibility. Ignored because Annoy
            does not contain nested estimators.

        Returns
        -------
        params : dict
            Dictionary of stable, user-facing parameters.

        See Also
        --------
        set_params : Set estimator-style parameters.
        schema_version : Controls pickle / snapshot strategy.

        Notes
        -----
        This is intended to make Annoy behave like a scikit-learn estimator for
        tools such as :func:`sklearn.base.clone` and parameter grids.
        """
        ...

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """
        Get output feature names for the transformer-style API.

        Parameters
        ----------
        input_features : sequence of str or None, optional, default=None
            If provided, validated deterministically against the fitted input
            feature names (if available) and the expected input dimensionality.

        Returns
        -------
        tuple of str
            Output feature names: ``('neighbor_0', ..., 'neighbor_{k-1}')`` where
            ``k == n_neighbors``.

        Raises
        ------
        AttributeError
            If called before :meth:`fit`/`build`.
        ValueError
            If ``input_features`` is provided but does not match
            :attr:`feature_names_in_`.
        """
        ...

    def info(self, *, include_n_items: bool = True, include_n_trees: bool = True, include_memory: bool | None = None) -> AnnoyInfo:
        """
        Return a structured summary of the index.

        This method returns a JSON-like Python dictionary that is easier to
        inspect programmatically than the legacy multi-line string format.

        Parameters
        ----------
        include_n_items : bool or None, optional, default=True
            If True, include ``n_items``. If None, behaves like the default (True).
        include_n_trees : bool or None, optional, default=True
            If True, include ``n_trees``. If None, behaves like the default (True).
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

        See Also
        --------
        serialize : Create a binary snapshot of the index.
        deserialize : Restore from a binary snapshot.
        save : Persist the index to disk.
        load : Load the index from disk.

        Notes
        -----
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

    def load(self, fn: str, prefault: bool | None = None) -> Self:
        """
        Load (mmap) an index from disk into the current object ``load(fn, prefault=None)``.

        Parameters
        ----------
        fn : str
            Path to a file previously created by :meth:`save` or
            :meth:`on_disk_build`.
        prefault : bool or None, optional, default=None
            If True, fault pages into memory when the file is mapped.
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

        See Also
        --------
        save : Save the current index to disk.
        on_disk_build : Build directly using an on-disk backing file.
        unload : Release mmap resources.

        Notes
        -----
        The in-memory index must have been constructed with the same dimension
        and metric as the on-disk file.
        """
        ...

    def memory_usage(self) -> int | None:
        """
        Approximate memory usage of the index in bytes ``memory_usage() -> int``.

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
        Configure the index to build using an on-disk backing file ``on_disk_build(fn)``.

        Parameters
        ----------
        fn : str
            Path to a file that will hold the index during build.
            The file is created or overwritten as needed.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        See Also
        --------
        build : Build trees after adding items (on-disk backed).
        load : Memory-map the built index.
        save : Persist the built index to disk.

        Notes
        -----
        This mode is useful for very large datasets that do not fit
        comfortably in RAM during construction.
        """
        ...

    def rebuild(
        self,
        metric: AnnoyMetricCanonical | None = None,
        *,
        on_disk_path: str | None = None,
        n_trees: int | None = None,
        n_jobs: int = -1,
    ) -> Self:
        """rebuild(metric=None, *, on_disk_path=None, n_trees=None, n_jobs=-1) -> Annoy

        Return a new Annoy index rebuilt from the current index contents.

        This helper is intended for deterministic, explicit rebuilds when changing
        structural constraints such as the metric (Annoy uses metric-specific C++
        index types). The source index is not mutated.

        Parameters
        ----------
        metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'} or None, optional
            Metric for the new index. If None, reuse the current metric.
        on_disk_path : path-like or None, optional
            Optional on-disk build path for the new index.

            Safety: the source object's on_disk_path is never carried over implicitly.
            If on_disk_path is provided and is string-equal to the source's configured
            path, it is ignored to avoid accidental overwrite/truncation hazards.
        n_trees : int or None, optional
            If provided, build the new index with this number of trees (or -1 for
            Annoy's internal auto mode). If None, reuse the source's tree count only
            when the source index is already built; otherwise do not build.
        n_jobs : int, optional, default=-1
            Number of threads to use while building (-1 means "auto").

        Returns
        -------
        :class:`~.Annoy`
            A new Annoy instance containing the same items (and y metadata if present).

        See Also
        --------
        get_params : Read constructor parameters.
        set_params : Update estimator parameters (use with `fit(X)` when refitting from data).
        fit : Build the index from `X` (preferred if you already have `X` available).
        serialize, deserialize : Persist / restore indexes; canonical restores rebuild deterministically.
        __sklearn_clone__ : Unfitted clone hook (no fitted state).

        Notes
        -----
        `rebuild(metric=...)` is deterministic and preserves item ids (0..n_items-1)
        by copying item vectors from the current fitted index into a new instance
        and rebuilding trees.

        Use `rebuild()` when you want to change `metric` while *reusing the already-stored
        vectors* (e.g., you do not want to re-read or re-materialize `X`, or you loaded an
        index from disk and only have access to its stored vectors).
        """
        ...

    def repr_info(self, *, include_n_items: bool | None = True, include_n_trees: bool | None = True, include_memory: bool | None = None) -> str:
        """
        Return a dict-like string representation with optional extra fields.

        repr_info(include_n_items=True, include_n_trees=True, include_memory=None) -> str

        Unlike ``__repr__``, this method can include additional fields on demand.
        Note that ``include_memory=True`` may be expensive for large indexes.
        Memory is calculated after :meth:`build`.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence: disk + byte + memory usage
    # ------------------------------------------------------------------

    def save(self, fn: str, prefault: bool | None = None) -> Self:
        """
        Persist the index to a binary file on disk ``save(fn, prefault=None)``.

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

        See Also
        --------
        load : Load an index from disk.
        serialize : Snapshot to bytes for in-memory persistence.

        Notes
        -----
        The output file will be overwritten if it already exists.
        Use prefault=None to fall back to the stored :attr:`prefault` setting.

        Examples
        --------
        >>> idx.save("annoy_test.annoy")
        >>> idx2 = Annoy(100, metric="angular")
        >>> idx2.load("annoy_test.annoy")
        >>> idx2.get_n_items()
        1000
        """
        ...

    def serialize(self, format=None) -> bytes:
        """
        Serialize the built in-memory index into a byte string ``serialize(format=None) -> bytes``.

        Parameters
        ----------
        format : {"native", "portable", "canonical"} or None, optional, default=None
            Serialization format.

            * "native" (legacy): raw Annoy memory snapshot. Fastest, but
              only compatible when the ABI matches exactly.
            * "portable": prepend a small compatibility header (version,
              endianness, sizeof checks, metric, f) so deserialization fails
              loudly on mismatches.
            * "canonical": rebuildable wire format storing item vectors + build
              parameters. Portable across ABIs (within IEEE-754 float32) and
              restores by rebuilding trees deterministically.

        Returns
        -------
        data : bytes
            Opaque binary blob containing the Annoy index.

        Raises
        ------
        RuntimeError
            If the index is not initialized or serialization fails.
        OverflowError
            If the serialized payload is too large to fit in a Python bytes object.

        See Also
        --------
        deserialize : Restore an index from a serialized byte string.

        Notes
        -----
        "Portable" blobs are the native snapshot with additional compatibility guards.
        They are not a cross-architecture wire format.

        "Canonical" blobs trade load time for portability: deserialization rebuilds
        the index with ``n_jobs=1`` for deterministic reconstruction.

        Examples
        --------
        >>> buf = idx.serialize()
        >>> new_idx = Annoy(100, metric="angular")
        >>> new_idx.deserialize(buf)
        >>> new_idx.get_n_items()
        1000
        """
        ...

    def set_params(self, **params) -> Self:
        """
        Set estimator-style parameters (scikit-learn compatibility) ``set_params(**params) -> Annoy``.

        Parameters
        ----------
        **params
            Keyword parameters to set. Unknown keys raise ``ValueError``.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        Raises
        ------
        ValueError
            If an unknown parameter name is provided.
        TypeError
            If parameter names are not strings or types are invalid.

        See Also
        --------
        get_params : Return estimator-style parameters.

        Notes
        -----
        Changing structural parameters (notably ``metric``) on an already
        initialized index resets the index deterministically (drops all items,
        trees, and :attr:`y`). Refit/rebuild is required before querying.

        This behavior matches scikit-learn expectations: ``set_params`` may be
        called at any time, but parameter changes that affect learned state
        invalidate the fitted model.
        """
        ...

    # ------------------------------------------------------------------
    # RNG / logging controls
    # ------------------------------------------------------------------

    def set_seed(self, seed: int | None = None) -> Self:
        """
        Set the random seed used for tree construction ``set_seed(seed=None)``.

        Parameters
        ----------
        seed : int or None, optional, default=None
            Non-negative integer seed. If called before the index is constructed,
            the seed is stored and applied when the C++ index is created.
            Seed value ``0`` resets to Annoy's core default seed (with a :class:`UserWarning`).

            * If omitted (or None, NULL), the seed is set to Annoy's default seed.
            * If 0, clear any pending override and reset to Annoy's default seed
              (a :class:`UserWarning` is emitted).

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

    @overload
    def transform(
        self,
        X: Sequence[Sequence[Any]],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        return_labels: Literal[False] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["vector"] = ...,
        output_type: Literal["item"] = ...,
        exclude_self: Literal[False] = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> Indices2D: ...

    @overload
    def transform(
        self,
        X: Sequence[Sequence[Any]],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[True] = ...,
        return_labels: Literal[False] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["vector"] = ...,
        output_type: Literal["item"] = ...,
        exclude_self: Literal[False] = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[Indices2D, Distances2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[Sequence[Any]],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        return_labels: Literal[True] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["vector"] = ...,
        output_type: Literal["item"] = ...,
        exclude_self: Literal[False] = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[Indices2D, Labels2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[Sequence[Any]],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[True] = ...,
        return_labels: Literal[True] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["vector"] = ...,
        output_type: Literal["item"] = ...,
        exclude_self: Literal[False] = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[Indices2D, Distances2D, Labels2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[Sequence[Any]],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        return_labels: Literal[False] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["vector"] = ...,
        output_type: Literal["vector"] = ...,
        exclude_self: Literal[False] = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> NeighborVectors3D: ...

    @overload
    def transform(
        self,
        X: Sequence[Sequence[Any]],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[True] = ...,
        return_labels: Literal[False] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["vector"] = ...,
        output_type: Literal["vector"] = ...,
        exclude_self: Literal[False] = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[NeighborVectors3D, Distances2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[Sequence[Any]],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        return_labels: Literal[True] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["vector"] = ...,
        output_type: Literal["vector"] = ...,
        exclude_self: Literal[False] = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[NeighborVectors3D, Labels2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[Sequence[Any]],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[True] = ...,
        return_labels: Literal[True] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["vector"] = ...,
        output_type: Literal["vector"] = ...,
        exclude_self: Literal[False] = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[NeighborVectors3D, Distances2D, Labels2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[int],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        return_labels: Literal[False] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["item"],
        output_type: Literal["item"] = ...,
        exclude_self: bool = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> Indices2D: ...

    @overload
    def transform(
        self,
        X: Sequence[int],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[True] = ...,
        return_labels: Literal[False] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["item"],
        output_type: Literal["item"] = ...,
        exclude_self: bool = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[Indices2D, Distances2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[int],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        return_labels: Literal[True] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["item"],
        output_type: Literal["item"] = ...,
        exclude_self: bool = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[Indices2D, Labels2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[int],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[True] = ...,
        return_labels: Literal[True] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["item"],
        output_type: Literal["item"] = ...,
        exclude_self: bool = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[Indices2D, Distances2D, Labels2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[int],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        return_labels: Literal[False] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["item"],
        output_type: Literal["vector"],
        exclude_self: bool = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> NeighborVectors3D: ...

    @overload
    def transform(
        self,
        X: Sequence[int],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[True] = ...,
        return_labels: Literal[False] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["item"],
        output_type: Literal["vector"],
        exclude_self: bool = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[NeighborVectors3D, Distances2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[int],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[False] = ...,
        return_labels: Literal[True] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["item"],
        output_type: Literal["vector"],
        exclude_self: bool = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[NeighborVectors3D, Labels2D]: ...

    @overload
    def transform(
        self,
        X: Sequence[int],
        *,
        n_neighbors: int = ...,
        search_k: int = ...,
        include_distances: Literal[True] = ...,
        return_labels: Literal[True] = ...,
        y_fill_value: Any = ...,
        input_type: Literal["item"],
        output_type: Literal["vector"],
        exclude_self: bool = ...,
        exclude_items: Sequence[int] | None = ...,
        missing_value: float | None = ...,
    ) -> tuple[NeighborVectors3D, Distances2D, Labels2D]: ...

    def transform(
        self,
        X: Any,
        *,
        n_neighbors: int = 5,
        search_k: int = -1,
        include_distances: bool = False,
        return_labels: bool = False,
        y_fill_value: Any = None,
        input_type: TransformInputType = "vector",
        output_type: TransformOutputType = "vector",
        exclude_self: bool = False,
        exclude_items: Sequence[int] | None = None,
        missing_value: float | None = None,
    ) -> TransformOutput:
        """transform(X, *, n_neighbors=5, search_k=-1, include_distances=False, return_labels=False, \
            y_fill_value=None, input_type='vector', output_type='vector', exclude_self=False, \
            exclude_items=None, missing_value=None)

        Transform queries into nearest-neighbor outputs (ids or vectors), with optional distances and labels.

        Parameters
        ----------
        X : array-like
            Query inputs. The expected shape/type depends on ``input_type``:

            - ``input_type='item'``  : X must be a 1D sequence of item ids.
            - ``input_type='vector'``: X must be a 2D array-like of shape (n_queries, f).
        n_neighbors : int, default=5
            Number of neighbors to retrieve for each query.
        search_k : int, default=-1
            Search parameter passed to Annoy (-1 uses Annoy's default).
        include_distances : bool, default=False
            If True, also return per-neighbor distances.
        return_labels : bool, default=False
            If True, also return per-neighbor labels resolved from :attr:`y` (as set via :meth:`fit`).
        y_fill_value : object, default=None
            Value used when ``y`` is unset or missing an entry for a neighbor id.
        input_type : {'vector', 'item'}, default='vector'
            Controls how ``X`` is interpreted.
        output_type : {'item', 'vector'}, default='item'
            Controls what is returned for each neighbor:

            - ``'item'``  : return neighbor ids (integers).
            - ``'vector'``: return neighbor vectors (lists of floats).
        exclude_self : bool, default=False
            If True, exclude the query item id from results. Only valid when ``input_type='item'``.
            For vector queries, "self" is undefined; use ``exclude_items``.
        exclude_items : sequence of int or None, default=None
            Explicit neighbor ids to exclude from results (deterministic filtering by id equality).
        missing_value : float or None, default=None
            If not None, imputes missing entries in X (None values in dense rows; missing keys / None values in dict rows).
            If None, missing entries raise.

        Returns
        -------
        neighbors
            Neighbor ids or vectors for each query. The returned shape depends on `output_type`:

            - ``output_type='item'``  : list[list[int]] with shape (n_queries, n_neighbors)
            - ``output_type='vector'``: list[list[list[float]]] with shape (n_queries, n_neighbors, f)
        (neighbors, distances) : tuple
            Returned when include_distances=True.
        (neighbors, labels) : tuple
            Returned when return_labels=True.
        (neighbors, distances, labels) : tuple
            Returned when include_distances=True and return_labels=True.

        See Also
        --------
        get_nns_by_item : Neighbor search by item id.
        get_nns_by_vector : Neighbor search by query vector.
        fit, fit_transform : Estimator-style APIs.

        Notes
        -----
        - Excluding self is performed by matching neighbor ids to the query id (not by checking distance values).
        - For input_type='vector', exclude_self=True is an error; use exclude_items for explicit, deterministic filtering.
        - If exclusions prevent returning exactly `n_neighbors` results, this method raises ValueError.

        Examples
        --------
        Item queries (exclude the query id itself):

        >>> idx.transform([10, 20], input_type='item', output_type='item', n_neighbors=5, exclude_self=True)

        Vector queries (exclude explicit ids):

        >>> idx.transform(X_query, input_type='vector', output_type='item', n_neighbors=5, exclude_items=[10, 20])

        Return neighbor vectors:

        >>> idx.transform([10], input_type='item', output_type='vector', n_neighbors=5, exclude_self=True)
        """
        ...

    def unbuild(self) -> Self:
        """
        Discard the current forest, allowing new items to be added ``unbuild()``.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        See Also
        --------
        build : Rebuild the forest after adding new items.
        add_item : Add items (only valid when no trees are built).

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
        Unmap any memory-mapped file backing this index ``unload()``.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.

        See Also
        --------
        load : Memory-map an on-disk index into this object.
        on_disk_build : Configure on-disk build mode.

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

    def set_verbose(self, level: int = 1) -> Self:
        """
        Control verbosity of the underlying C++ index ``set_verbose(level=1)``.

        Set the verbosity level (callable setter).

        This method exists to preserve a callable interface while keeping the
        parameter name ``verbose`` available as an attribute for scikit-learn
        compatibility.

        Parameters
        ----------
        level : int, optional, default=1
            Verbosity level. Values are clamped to the range ``[-2, 2]``.
            ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.
            Logging level inspired by gradient-boosting libraries:

            * ``<= 0`` : quiet (warnings only)
            * ``1``    : info (Annoy's ``verbose=True``)
            * ``>= 2`` : debug (currently same as info, reserved for future use)

        See Also
        --------
        set_verbosity : Alias of :meth:`set_verbose`.
        verbose : Parameter attribute (int | None).
        get_params, set_params : Estimator parameter API.

        Returns
        -------
        :class:`~.Annoy`
            This instance (self), enabling method chaining.
        """
        ...

    def set_verbosity(self, level: int = 1) -> Self:
        """
        set_verbosity(level=1)

        Alias of :meth:`set_verbose`.
        """
        ...

    # ------------------------------------------------------------------
    # Pickle / joblib hooks
    # ------------------------------------------------------------------

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
        Pickle protocol support ``__reduce_ex__(protocol)``.

        This returns the standard 3-tuple ``(cls, args, state)`` used by pickle.
        Users do not need to call this directly.
        """
        ...

    def __reduce__(self) -> tuple[object, tuple[object, ...], dict[str, object]]:
        """
        Pickle support ``__reduce__()``.

        Equivalent to :meth:`__reduce_ex__` with the default protocol.
        """
        ...

    # ------------------------------------------------------------------
    # Rich display (Jupyter)
    # ------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the Annoy index for Jupyter notebooks.

        Returns
        -------
        html : str
            HTML string (safe to embed) describing the current configuration.

        See Also
        --------
        info : Return a Python dict with configuration and metadata.
        __repr__ : Text representation.

        Notes
        -----
        This representation is deterministic and side-effect free. It intentionally
        avoids expensive operations such as serialization or memory-usage estimation.
        """
        ...

    def __len__(self) -> int:
        """
        Return the number of items currently stored in the index.

        Notes
        -----
        This is equivalent to :meth:`get_n_items`. If the underlying index is not
        initialized yet, ``0`` is returned.
        """
        ...

    def __sklearn_is_fitted__(self) -> bool:
        """__sklearn_is_fitted__() -> bool

        Return whether this estimator is fitted (scikit-learn protocol hook).

        Returns
        -------
        is_fitted : bool
            True iff the index has been built (n_trees > 0).
        """
        ...

    def __sklearn_tags__(self) -> Any:
        """__sklearn_tags__() -> sklearn.utils.Tags

        Return estimator tags (scikit-learn protocol hook).

        Returns
        -------
        tags : sklearn.utils.Tags
            Conservative tags for this estimator.

        See Also
        --------
        sklearn.utils.get_tags : Read estimator tags.

        Notes
        -----
        This method is consulted by scikit-learn utilities such as
        ``sklearn.utils.get_tags``.
        """
        ...

    def __sklearn_clone__(self) -> Self:
        """__sklearn_clone__() -> Annoy

        Return an unfitted clone (scikit-learn protocol hook).

        Returns
        -------
        clone : :class:`~.Annoy`
            New unfitted instance with identical parameters.

        See Also
        --------
        get_params : Parameters used for cloning.
        sklearn.base.clone : Delegates to this hook when available.
        """
        ...

# Backwards-compatible alias exposed by the C-extension.
AnnoyIndex: TypeAlias = Annoy


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

    # ---- Core configuration (lazy-safe properties) ---------------------------
    @property
    def f(self) -> int: ...
    @f.setter
    def f(self, f: int) -> None: ...

    @property
    def metric(self) -> AnnoyMetricCanonical | None: ...
    @metric.setter
    def metric(self, metric: AnnoyMetric | None) -> None: ...

    @property
    def n_neighbors(self) -> int: ...
    @n_neighbors.setter
    def n_neighbors(self, n_neighbors: int) -> None: ...

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
    def seed(self) -> int | None: ...
    @seed.setter
    def seed(self, seed: int | None) -> None: ...

    @property
    def random_state(self) -> int | None: ...
    @random_state.setter
    def random_state(self, seed: int | None) -> None: ...

    @property
    def verbose(self) -> int | None: ...
    @verbose.setter
    def verbose(self, level: int | None) -> None: ...

    @property
    def schema_version(self) -> int: ...
    @schema_version.setter
    def schema_version(self, schema_version: int | None) -> None: ...

    @property
    def n_features_(self) -> int: ...
    @property
    def n_features_in_(self) -> int: ...
    @property
    def n_features(self) -> int: ...
    @n_features.setter
    def n_features(self, f: int) -> None: ...

    @property
    def n_features_out_(self) -> int: ...

    @property
    def feature_names_in_(self) -> list[str]: ...

    # Stored labels y (optional, scikit-learn compatible).
    @property
    def _y(self) -> dict | None: ...
    @property
    def y(self) -> dict | None: ...
    @y.setter
    def y(self, value: dict | None) -> None: ...

    # --- Build / lifecycle ---
    def add_item(self, i: ItemIndex, vector: Vector) -> Self: ...
    def build(self, n_trees: TreeCount, n_jobs: int = -1) -> Self: ...
    def unbuild(self) -> Self: ...
    def unload(self) -> Self: ...

    # --- Persistence ---
    def on_disk_build(self, fn: str) -> Self: ...
    def save(self, fn: str, prefault: bool | None = None) -> Self: ...
    def load(self, fn: str, prefault: bool | None = None) -> Self: ...
    def serialize(self, format: SerializeFormat | None = None) -> bytes: ...
    def deserialize(self, byte: bytes, prefault: bool | None = None) -> Self: ...

    # --- Introspection ---
    def __len__(self) -> int: ...

    def get_n_items(self) -> int: ...
    def get_n_trees(self) -> int: ...
    def memory_usage(self) -> int | None: ...
    def info(self, include_n_items: bool = True, include_n_trees: bool = True, include_memory: bool | None = None) -> AnnoyInfo: ...
    def repr_info(self, include_n_items: bool = True, include_n_trees: bool = True, include_memory: bool | None = None) -> str: ...

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

    # --- Estimator API / sklearn hooks ---
    def get_params(self, deep: bool = True) -> dict: ...
    def set_params(self, **params: Any) -> Self: ...
    def fit(
        self,
        X: Sequence[Sequence[Any]] | None = None,
        y: Sequence[Any] | None = None,
        *,
        n_trees: int = -1,
        n_jobs: int = -1,
        reset: bool = True,
        start_index: int | None = None,
        missing_value: float | None = None,
    ) -> Self: ...
    def transform(
        self,
        X: Any,
        *,
        n_neighbors: int = 5,
        search_k: int = -1,
        include_distances: bool = False,
        return_labels: bool = False,
        y_fill_value: Any = None,
        input_type: TransformInputType = "vector",
        missing_value: float | None = None,
    ) -> TransformOutput: ...
    def fit_transform(
        self,
        X: Sequence[Sequence[Any]],
        y: Sequence[Any] | None = None,
        *,
        n_trees: int = -1,
        n_jobs: int = -1,
        reset: bool = True,
        start_index: int | None = None,
        missing_value: float | None = None,
        n_neighbors: int = 5,
        search_k: int = -1,
        include_distances: bool = False,
        return_labels: bool = False,
        y_fill_value: Any = None,
    ) -> TransformOutput: ...
    def rebuild(
        self,
        metric: AnnoyMetricCanonical | None = None,
        *,
        on_disk_path: str | None = None,
        n_trees: int | None = None,
        n_jobs: int = -1,
    ) -> Self: ...

    def __sklearn_is_fitted__(self) -> bool: ...
    def __sklearn_tags__(self) -> Any: ...
    def __sklearn_clone__(self) -> Self: ...

    # --- RNG / logging controls ---
    def set_seed(self, seed: int | None = None) -> Self: ...
    def set_verbose(self, level: int = 1) -> Self: ...
    def set_verbosity(self, level: int = 1) -> Self: ...

    # --- Pickle / display ---
    def __getstate__(self) -> dict[str, object]: ...
    def __setstate__(self, state: dict[str, object]) -> None: ...
    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[object, ...], dict[str, object]]: ...
    def __reduce__(self) -> tuple[object, tuple[object, ...], dict[str, object]]: ...
    def _repr_html_(self) -> str: ...
