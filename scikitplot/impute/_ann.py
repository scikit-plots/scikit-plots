# ruff: noqa: F401
# pylint: disable=unused-import

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Vector-based Approximate K-Nearest Neighbors (KNN) [1]_ imputation :py:class:`~.ANNImputer`.

Annoy [2]_ (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings
to search for points in space that are close to a given query point.

Voyager [3]_ is an HNSW-based approximate nearest-neighbor index with a Python API.

Both libraries create large read-only file-based data structures that can be
memory-mapped so that many processes may share the same data.

References
----------
.. [1] http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor
.. [2] https://github.com/spotify/annoy
.. [3] https://github.com/spotify/voyager
"""  # noqa: D205

from __future__ import annotations

# from numbers import Integral
import atexit
import contextlib
import os
import tempfile
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# sklearn
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.impute._base import _BaseImputer
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.pairwise import _NAN_METRICS
from sklearn.neighbors._base import _get_weights
from sklearn.utils._mask import _get_mask
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._param_validation import (
    Hidden,
    Integral,
    Interval,
    Options,
    StrOptions,
)
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_feature_names_in,
    check_array,
    check_is_fitted,
    validate_data,
)

from ..utils._path import PathNamer, make_path, make_temp_path, normalize_directory_path
from ..utils._time import Timer
from ._privacy import OutsourcedIndexMixin

try:
    from ..annoy import Index as AnnoyIndex
except Exception:  # pragma: no cover - fallback to external annoy
    from annoy import AnnoyIndex

try:
    import voyager
except ImportError as e:  # noqa: N816
    voyager = None
    _VOYAGER_IMPORT_ERROR = e
else:
    _VOYAGER_IMPORT_ERROR = None

__all__ = [
    "ANNImputer",  # Unified Approximate KNN imputer
    "AnnoyKNNImputer",  # Annoy backend alias
    "VoyagerKNNImputer",  # Voyager backend shim
]


# --------------------------------------------------------------------------- #
# Approximate KNN-based imputer using Annoy (inherits sklearn _BaseImputer)
# --------------------------------------------------------------------------- #
# class _BaseImputer(TransformerMixin, BaseEstimator):
class ANNImputer(OutsourcedIndexMixin, _BaseImputer):
    """
    Approximate K-nearest-neighbours (KNN) imputer with pluggable ANN backends.

    :class:`~.ANNImputer` performs vector-based imputation by querying an
    approximate nearest-neighbours (ANN) index instead of using exact
    brute-force distances as in :class:`~sklearn.impute.KNNImputer`.

    Two backends are currently supported:

    * ``backend='annoy'`` (default):
      uses the Spotify Annoy library and the in-tree
      :class:`~scikitplot.annoy.Index` wrapper.
    * ``backend='voyager'``:
      uses the optional :mod:`voyager` package (HNSW-based index).

    All high-level imputation parameters (:attr:`n_neighbors`,
    :attr:`weights`, :attr:`metric`, :attr:`index_access`,
    :attr:`index_store_path`. Backend-specific details
    (such as the Annoy forest size) are handled internally.

    This imputer identifies approximate nearest neighbors for samples
    containing missing values and imputes those values using statistics computed
    from the retrieved neighbor vectors.

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to np.nan, since `pd.NA` will be converted to np.nan.

    backend : {'annoy', 'voyager'}, default='annoy'
        Name of the approximate nearest-neighbour backend to use.

        - ``'annoy'``: use the modified Spotify Annoy library (in-tree wrapper).
        - ``'voyager'``: use the optional :mod:`voyager` package.

        When ``backend='voyager'`` the :mod:`voyager` package must be
        installed. An :class:`ImportError` is raised otherwise.

        Parameters :attr:`index_access`, :attr:`index_store_path` and
        behave identically for both backends.

    index_access : {'public', 'private', 'external'}, default='external'
        Controls whether and how the fitted ANN index is exposed or stored.

        - ``'public'``: :attr:`train_index_` returns the underlying
          ANN index instance (backwards compatible behaviour).
        - ``'private'``: Any attempt to access :attr:`train_index_`
          raises :class:`AttributeError`. The index is still used
          internally during :meth:`transform`, but is not directly
          exposed to user code.
        - ``'external'``: The fitted ANN index is persisted to disk using
          the backend index's :py:meth:`save` method
          (e.g. :meth:`AnnoyIndex.save` or :meth:`voyager.Index.save`) and
          only the file name (:attr:`index_path_`) and metadata
          (:attr:`index_created_at_`) are stored on the estimator.
          At runtime the index is reloaded from that file as needed.
          In this mode :attr:`train_index_` is not available.

        For production and privacy-sensitive workloads, it is strongly
        recommended to keep the default ``'external'`` mode so that the
        underlying ANN index is not part of the public API by default.

    index_store_path : str or path-like, PathNamer, default=None
        Target file path used when ``index_access='external'``. The
        fitted ANN index is saved to this location via the backend index
        :py:meth:`save` method, and only the file name and metadata
        are stored in the estimator.

        If ``index_access='external'`` and this is ``None``,
        :meth:`fit` will automatically generate an OS-friendly unique
        file name by :class:`~.PathNamer` (or the current working directory) and
        save the ANN index there.

    on_disk_build : bool, default=False
        Only used when ``backend='annoy'``. Ignored for other backends.

        If ``True``, the underlying Annoy index is built using
        :meth:`AnnoyIndex.on_disk_build`, which streams the index to a backing
        file during construction. This can significantly reduce peak RAM
        usage for very large datasets.

        This only affects how the index is **built**. How the index is
        stored and accessed at runtime is still controlled by
        ``index_access`` (``'public'``, ``'private'``, ``'external'``) and
        ``index_store_path``.

    n_trees : int, default=-1
        Number of trees in the Annoy forest. Increasing the number of trees
        generally improves nearest-neighbor accuracy but increases build time
        and memory usage.

        If set to ``-1``, the value is passed as-is to the backend index
        implementation, which may interpret it built dynamically
        until the index reaches approximately twice the number of items
        If -1, defaults to ``_n_nodes >= 2 * n_items``.
        This situation can lead to a stochastic result.

        Guidelines:

        - Small datasets (<10k samples): 10-20 trees.
        - Medium datasets (10k-1M samples): 20-50 trees.
        - Large datasets (>1M samples): 50-100+ trees.

    search_k : int, default=-1
        Backend-specific search-depth parameter.

        For Annoy, this is passed as ``search_k`` to
        :meth:`AnnoyIndex.get_nns_by_vector`. Larger values inspect more
        nodes during search and are therefore slower but more accurate.
        If -1, defaults to `n_trees * n_neighbors`.

        In Voyager Index.query() passed to as query_ef - The depth of search
        to perform for this query. Up to query_ef candidates will be searched
        through to try to find up the k nearest neighbors per query vector.

    n_neighbors : int, default=5
        Number of neighboring samples used for imputation.
        Higher values produce smoother imputations but may reduce locality.

    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weighting strategy for neighbor contributions:

        - `'uniform'` : all neighbors have equal weight.
        - `'distance'` : inverse-distance weighting,
          where closer neighbors contribute more
          (`w_ik = 1 / (1 + d(x_i, x_k))`).
        - callable : custom function taking an array of distances
          and returning an array of weights.

    metric : {"angular", "cosine", "euclidean", "l2", "lstsq", "manhattan", "l1", "cityblock", "taxicab", \
              "dot", "@", ".", "dotproduct", "inner", "innerproduct", "hamming"}, optional, default='angular'
        Distance metric used for nearest-neighbor search:

        - `'angular'` : Cosine similarity (angle only, ignores magnitude).
          Best for normalized embeddings (e.g., text embeddings, image features).
        - `'euclidean'` : L2 distance, defined as √Σ(xᵢ - yᵢ)².
          Standard geometric distance, sensitive to scale.
        - `'manhattan'` : L1 (City-block) distance, defined as Σ|xᵢ - yᵢ|.
          More robust to outliers than L2, still scale-sensitive.
        - `'hamming'` : Fraction or count of of differing elements.
          Suitable for binary or categorical features (e.g., 0/1).
        - `'dot'` : Negative inner product (-x·y).
          Sensitive to both direction and magnitude of vectors.

        Aliases:

        - cosine <-> angular
        - euclidean <- l2, lstsq
        - manhattan <- l1, cityblock, taxicab
        - dot <-> innerproduct <- @, ., dotproduct, inner

        Note that when ``backend='voyager'`` not support all metrics
        (such as ``"manhattan"`` or ``"hamming"``)
        with the voyager backend will raise :class:`ValueError`.

        .. seealso::
            * :py:func:`~scipy.spatial.distance.cosine`
            * :py:func:`~scipy.spatial.distance.euclidean`
            * :py:func:`~scipy.spatial.distance.cityblock`
            * :py:meth:`~scipy.sparse.coo_array.dot`
            * :py:func:`~scipy.spatial.distance.hamming`

    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, default='mean'
        Which strategy to use to initialize the missing values when building
        the ANN index. This is analogous to the `strategy` parameter in
        :class:`~sklearn.impute.SimpleImputer`:

        - ``'mean'``: use the column-wise mean (ignoring NaNs).
        - ``'median'``: use the column-wise median (ignoring NaNs).
        - ``'most_frequent'``: use the column-wise mode (most frequent value,
          ignoring NaNs; if a column has no observed values, it falls back
          to 0.0).
        - ``'constant'``: use :attr:`fill_value` for all features. If
          :attr:`fill_value` is ``None``, a default of ``0.0`` is used for
          numeric data.

        This strategy affects only the temporary fill vector used to build the
        ANN index and the global fallback used when no valid neighbor values
        are available. The main imputation logic is still k-nearest-neighbours
        based on the ANN index.

    fill_value : str or numerical value, default=None
        When `strategy="constant"`, `fill_value` is used to replace all
        occurrences of missing_values. For string or object data types,
        `fill_value` must be a string.
        If `None`, `fill_value` will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto the
        output of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on the
        missing indicator even if there are missing values at transform/test
        time.

    keep_empty_features : bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0`.

    n_jobs : int or None, default=-1
        Parallelism level used in two places:

        - during Annoy index construction, passed to
          :meth:`AnnoyIndex.build`,
        - during Voyager query construction, passed to
          :meth:`Voyager.query`,
        - during imputation, used as the number of worker threads in a
          :class:`joblib.Parallel` loop.

        A value of ``-1`` uses all available CPU cores. Using threads
        for the imputation step avoids spawning new Python processes and
        keeps this estimator compatible with editable installs and other
        environments where the package cannot be safely re-imported in
        child processes.

    random_state : int or None, default=None
        Seed for the backend index construction (e.g. Annoy hyperplanes,
        Voyager graph initialization).

        .. caution::
            ⚠️ Reproducibility for Annoy required both ``random_state``
            with ``n_trees``.

    Attributes
    ----------
    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        ``None`` if add_indicator is False.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    temp_fill_vector_ : ndarray of shape (n_features_in_,)
        Per-feature statistics (e.g. mean or median) used to temporarily
        fill missing values when building the ANN index and as a
        fallback when neighbor information is not available.

    index_path_ : str
        File path of the persisted ANN index when
        ``index_access='external'``. Only set after :meth:`fit`.

    index_created_at_ : str
        UTC ISO 8601 timestamp recording when the ANN index was
        persisted to :attr:`index_path_` in ``index_access='external'``
        mode.

    train_index_ : object or property
        In ``index_access='public'`` mode, this property returns the
        fitted ANN index instance (for example an :class:`~.AnnoyIndex` or
        :class:`voyager.Index`). In ``index_access='private'`` or
        ``index_access='external'`` mode, accessing this attribute
        raises :class:`AttributeError` to prevent direct access to the
        underlying index object.

    .. warning::

       ``index_access='private'`` or ``index_access='external'``
       prevents access to the underlying ANN index
       through the public API (for example :attr:`train_index_`). This protects
       against accidental leaks and misuse, but it is **not** a hard security
       boundary: any Python code running in the same process can still inspect
       the estimator using introspection facilities.

       If you need strong confidentiality for the training data or the ANN index,
       do **not** share the :class:`~.ANNImputer` instance with untrusted code.
       Instead, run it inside a separate process or service and expose only a
       high-level API (e.g. an ``/impute`` endpoint) rather than the Python
       object itself (model-as-a-service pattern).

    Notes
    -----
    For each sample :math:`x_i` and feature :math:`j`, the imputed value is:

    .. math::
        \\hat{x}_{ij} = \\frac{\\sum_{k \\in N_i} w_{ik} x_{kj}}{\\sum_{k \\in N_i} w_{ik}}

    where :math:`N_i` is the set of K nearest neighbors of :math:`x_i`,
    and :math:`w_{ik}` is the neighbor weight:

    :math:`w_{ik} = \\frac{1}{1 + d(x_i, x_k)}`

    - ANN provides approximate neighbor search, so imputations are not exact.
    - Annoy uses random projections to split the vector space at each node in
      the tree, selecting a random hyperplane defined by two sampled points.
    - In ``index_access='public'`` or ``'private'`` mode the Annoy index
      is kept in memory after :meth:`fit` for efficient queries.
      In ``index_access='external'`` mode the index is stored on disk
      and loaded on demand at transform-time.
    - Index creation is separate from lookup. After calling ``build()``,
      no additional vectors may be added.
    - Index files created by Annoy are memory-mapped, allowing multiple processes
      to share the same data without additional memory overhead.
    - Annoy is optimized for scenarios with many items in moderate to high
      dimensional spaces where fast approximate neighbor retrieval is more
      important than exact results.
    - Annoy supports specific metrics; `'euclidean'` (p=2) and `'manhattan'` (p=1)
      are special cases of the Minkowski distance.

    See Also
    --------
    sklearn.impute.KNNImputer : Multivariate imputer that estimates missing features using
        nearest samples. Exact KNN-based imputer using brute-force search.
    sklearn_ann.kneighbors.annoy.AnnoyTransformer : Wrapper for using annoy.AnnoyIndex as
        sklearn's KNeighborsTransformer `AnnoyTransformer
        <https://sklearn-ann.readthedocs.io/en/latest/kneighbors.html#annoy>`_
    PathNamer : Naming helper for external index file.

    References
    ----------
    .. [1] `Bernhardsson, E. (2013). "ANNoy (Approximate Nearest Neighbors Oh Yeah)."
       Spotify AB. https://github.com/spotify/annoy
       <https://github.com/spotify/annoy>`_

    Examples
    --------
    >>> import numpy as np
    >>> from scikitplot.experimental import enable_aknn_imputer
    >>> from scikitplot.impute import ANNImputer
    >>> X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
    >>> # imputer = ANNImputer(backend="voyager", n_neighbors=5, metric="euclidean")
    >>> imputer = ANNImputer(n_trees=5, n_neighbors=5)
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 5. ],
           [3. , 4. , 3. ],
           [4. , 6. , 5. ],
           [8. , 8. , 7. ]])
    """  # noqa: D301

    # ------------------------------------------------------------------ #
    # sklearn parameter constraints
    # ------------------------------------------------------------------ #
    _parameter_constraints = {  # noqa: RUF012
        **_BaseImputer._parameter_constraints,
        "backend": [
            StrOptions({"annoy", "voyager"}),
        ],
        "index_access": [
            StrOptions({"public", "private", "external"}),
        ],
        "index_store_path": [str, os.PathLike, PathNamer, None],
        "on_disk_build": ["boolean"],
        "n_trees": [
            Interval(Integral, 1, None, closed="left"),  # integers ≥ 1
            Options(int, {-1}),  # allow -1 handle internally
        ],
        "search_k": [
            Interval(Integral, 1, None, closed="left"),  # integers ≥ 1
            Options(int, {-1}),  # allow -1 handle internally
        ],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "weights": [
            StrOptions({"uniform", "distance"}),
            callable,
            # None,
            Hidden(None),
            # Hidden(StrOptions({"deprecated"})),
        ],
        "metric": [
            StrOptions(
                {
                    "angular",
                    "cosine",
                    "euclidean",
                    "l2",
                    "lstsq",
                    "manhattan",
                    "l1",
                    "cityblock",
                    "taxicab",
                    "dot",
                    "@",
                    ".",
                    "dotproduct",
                    "inner",
                    "innerproduct",
                    "hamming",
                },
            ),
        ],
        "initial_strategy": [
            StrOptions({"mean", "median", "most_frequent", "constant"})
        ],
        "fill_value": "no_validation",  # any object is valid
        "copy": ["boolean"],
        "n_jobs": [None, Integral],
        "random_state": ["random_state"],
        # "verbose": ["verbose"],
        # "include_distances": "no_validation",  # any object is valid
    }

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        missing_values=np.nan,
        backend="annoy",
        index_access="external",
        index_store_path=None,  # PathNamer .annoy/.voy
        on_disk_build=False,
        n_trees=-1,  # annoy default -1 (dynamic heuristic)
        search_k=-1,  # backend search depth (-1: backend default)
        n_neighbors=5,
        weights="uniform",
        metric="angular",  # annoy default
        initial_strategy="mean",
        fill_value=None,
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
        n_jobs=-1,  # annoy default
        random_state=None,
    ):
        # Base imputer handles missing_values / indicators
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        # Backend / index configuration
        self.backend = backend
        self.index_access = index_access
        self.index_store_path = index_store_path
        self.on_disk_build = on_disk_build
        # ANN / imputation hyperparameters
        self.n_trees = n_trees
        self.search_k = search_k  # or (self.n_trees * self.n_neighbors)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        # Temporary fill strategy when building the ANN index
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.copy = copy
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Optional public view of the index (encapsulated)
    # ------------------------------------------------------------------ #
    @property
    def train_index_(self):
        """
        Optionally expose the fitted ANN index (Annoy or Voyager).

        This attribute is only available when ``index_access='public'``.
        For other values, :class:`AttributeError` is raised by
        :meth:`OutsourcedIndexMixin._get_index`.
        """
        return self._get_index("train_index_")

    @train_index_.setter
    def train_index_(self, value):
        """
        Prevent external code from overwriting the underlying index.

        The index should only be created via :meth:`fit`. This setter exists
        mainly to avoid accidental ``imputer.train_index_ = ...`` assignments.
        """
        self._set_index("train_index_", value)

    # ------------------------------------------------------------------ #
    # Backend helpers
    # ------------------------------------------------------------------ #
    def _resolve_index_store_path(self) -> str | None:
        """
        Resolve the on-disk path used to persist the ANN index.

        In ``index_access='external'`` mode, a path is required. If none is provided,
        a new unique path is generated once and cached.

        In non-external modes, a path is only used if explicitly provided.

        Returns
        -------
        path : str or None
            Absolute file path for the index (e.g., ``*.annoy`` or ``*.voy``),
            or ``None`` when no on-disk persistence is used.

        Notes
        -----
        - The resolved path is cached into ``self.index_store_path`` as a string to
        ensure stability across repeated calls.
        - If ``self.index_store_path`` is a :class:`~scikitplot.utils._path.PathNamer`,
        a single path is generated once and then cached.
        """
        index_store_path = getattr(self, "index_store_path", None)
        index_access = getattr(self, "index_access", "external")

        # Cosmetic extension only; actual backend governs file content/format.
        backend = getattr(self, "backend", "annoy")
        ext = ".voy" if backend == "voyager" else ".annoy"

        if index_access == "external":
            # ------------------------------------------------------------------ #
            # External mode: a path is required; generate one if not provided.
            # ------------------------------------------------------------------ #
            if index_store_path is None:
                # Generate once and cache. Keep in current working directory by default.
                self.index_store_path = str(
                    make_path(
                        prefix="ANNImputer",
                        ext=ext,
                    )
                )
                return self.index_store_path

            if isinstance(index_store_path, (str, os.PathLike)):
                self.index_store_path = str(normalize_directory_path(index_store_path))
                return self.index_store_path

            if isinstance(index_store_path, PathNamer):
                self.index_store_path = str(index_store_path.make_path())
                return self.index_store_path

            raise TypeError(
                "index_store_path must be None, a path-like (str/PathLike), or PathNamer."
            )

        # ------------------------------------------------------------------ #
        # Non-external modes: never auto-generate a path.
        # ------------------------------------------------------------------ #
        if index_store_path is None:
            return None

        # If already cached as a string/pathlike, normalize + re-cache.
        if isinstance(index_store_path, (str, os.PathLike)):
            self.index_store_path = str(normalize_directory_path(index_store_path))
            return self.index_store_path

        # If user provided a generator, generate once then cache.
        if isinstance(index_store_path, PathNamer):
            self.index_store_path = str(index_store_path.make_path())
            return self.index_store_path

        raise TypeError(
            "index_store_path must be None, a path-like (str/PathLike), or PathNamer."
        )

    def _ensure_voyager_available(self) -> None:
        """Raise a clear error if backend='voyager' but voyager is not installed."""
        if voyager is None:
            msg = (
                f"{self.__class__.__name__}(backend='voyager') requires the "
                "`voyager` package to be installed. Install with:\n\n"
                "    pip install voyager\n"
            )
            raise ImportError(msg) from _VOYAGER_IMPORT_ERROR

    def _resolve_voyager_space(self) -> "voyager.Space":  # type: ignore[] # noqa: UP037
        """
        Map :attr:`metric` to a :class:`voyager.Space` value.

        Returns
        -------
        space : voyager.Space
            Distance space used to construct :class:`voyager.Index`.
        """
        self._ensure_voyager_available()

        # metric = (self.metric or "euclidean").lower()
        mapping = {
            # cosine-like
            "angular": voyager.Space.Cosine,
            "cosine": voyager.Space.Cosine,
            # L2
            "euclidean": voyager.Space.Euclidean,
            "l2": voyager.Space.Euclidean,
            "lstsq": voyager.Space.Euclidean,
            # inner-product-like
            "dot": voyager.Space.InnerProduct,
            "@": voyager.Space.InnerProduct,
            ".": voyager.Space.InnerProduct,
            "dotproduct": voyager.Space.InnerProduct,
            "inner": voyager.Space.InnerProduct,
            "innerproduct": voyager.Space.InnerProduct,
        }
        try:
            return mapping[self.metric]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported metric {self.metric!r} for backend='voyager'. "
                f"Supported values are: {list(mapping.keys())}."
            ) from exc

    def _resolve_metric(self) -> str:
        """
        Map a configured metric alias to one of Annoy's native metrics.

        Returns
        -------
        metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'}
            The metric name passed to :class:`~.AnnoyIndex`.
        """
        alias_map = {
            "angular": "angular",
            "cosine": "angular",
            "euclidean": "euclidean",
            "l2": "euclidean",
            "lstsq": "euclidean",
            "manhattan": "manhattan",
            "l1": "manhattan",
            "cityblock": "manhattan",
            "taxicab": "manhattan",
            "dot": "dot",
            "@": "dot",
            ".": "dot",
            "dotproduct": "dot",
            "inner": "dot",
            "innerproduct": "dot",
            "hamming": "hamming",
        }
        try:
            return alias_map[self.metric]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported metric {self.metric!r} for backend='annoy'. "
                f"Supported values are: {list(alias_map.keys())}."
            ) from exc

    def _compute_initial_fill_vector(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-feature fill statistics according to `initial_strategy`.

        This is used only for:

        * temporarily filling missing values when building the ANN index, and
        * as a last-resort fallback if neighbor information is missing.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data with possible NaNs.

        Returns
        -------
        fill_vec : ndarray of shape (n_features,)
            Per-feature fill values.
        """
        strategy = self.initial_strategy

        if strategy == "mean":
            stats = np.nanmean(X, axis=0)
            # For columns that are entirely missing, nanmean returns np.nan.
            # Use a neutral numeric fallback (0.0) there.
            return np.where(np.isnan(stats), 0.0, stats)

        if strategy == "median":
            stats = np.nanmedian(X, axis=0)
            return np.where(np.isnan(stats), 0.0, stats)

        if strategy == "most_frequent":
            # Column-wise mode, ignoring NaNs.
            fill_vec = np.empty(X.shape[1], dtype=float)
            for j in range(X.shape[1]):
                col = X[:, j]
                col = col[~np.isnan(col)]
                if col.size == 0:
                    # No observed values: fall back to 0.0
                    fill_vec[j] = 0.0
                else:
                    vals, counts = np.unique(col, return_counts=True)
                    fill_vec[j] = vals[np.argmax(counts)]
            return fill_vec

        if strategy == "constant":
            if self.fill_value is None:
                # SimpleImputer default for numeric data
                fill = 0.0
            else:
                try:
                    fill = float(self.fill_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{self.__class__.__name__}: `initial_strategy='constant'` "
                        "requires a numeric `fill_value` for numeric input data."
                    ) from exc
            return np.full(X.shape[1], fill, dtype=float)

        # This should not happen thanks to _parameter_constraints,
        # but we fail explicitly rather than silently falling back.
        raise ValueError(
            f"{self.__class__.__name__}: unsupported `initial_strategy={strategy!r}`. "
            "Expected one of {'mean', 'median', 'most_frequent', 'constant'}."
        )

    def _apply_fill_vector(self, X: np.ndarray) -> np.ndarray:
        """
        Temporarily fill missing values before index build.

        Missing entries (encoded as ``np.nan``) are replaced column-wise by the
        statistics returned from :meth:`_compute_initial_fill_vector`.
        """
        # Compute per-feature fill statistics
        # self.temp_fill_vector_ = np.full(X.shape[1], np.nan)
        # self.temp_fill_vector_ = np.nanmedian(X, axis=0)
        # self.temp_fill_vector_ = np.nanmean(X, axis=0)
        # # Temporary fill missing for index building
        # if self.initial_strategy == "median":
        #     self.temp_fill_vector_ = np.nanmedian(X, axis=0)
        # elif self.initial_strategy == "mean":
        #     self.temp_fill_vector_ = np.nanmean(X, axis=0)
        # else:
        #     self.temp_fill_vector_ = np.nanmean(X, axis=0)
        # Temporarily fill NaNs (mean/median) before index build.
        # Compute per-feature fill statistics according to `initial_strategy`
        self.temp_fill_vector_ = self._compute_initial_fill_vector(X)
        # be conservative: treat non-finite (inf, -inf) as missing
        try:
            X = np.where(
                np.isnan(
                    np.nan_to_num(
                        X,
                        nan=np.nan,
                        posinf=np.nan,
                        neginf=np.nan,
                    ),
                ),
                self.temp_fill_vector_,
                X,
            )
        except Exception:  # pragma: no cover - fallback
            X = np.where(np.isnan(X), self.temp_fill_vector_, X)
        return X

    # ------------------------------------------------------------------ #
    # Per-row utilities (shared by Annoy + Voyager)
    # ------------------------------------------------------------------ #
    def _prepare_query_vector(
        self,
        row: np.ndarray,
        fill_vec: np.ndarray,
    ) -> np.ndarray:
        """Replace missing / non-finite entries in ``row`` using ``fill_vec``."""
        try:
            return np.where(
                np.isnan(
                    np.nan_to_num(
                        row,
                        nan=np.nan,
                        posinf=np.nan,
                        neginf=np.nan,
                    ),
                ),
                fill_vec,
                row,
            )
        except Exception:  # pragma: no cover - ultra-conservative fallback
            return np.where(np.isnan(row), fill_vec, row)

    def _iter_non_nan_rows(self, X: np.ndarray):
        """Yield ``(index, row)`` pairs where ``row`` contains no NaN values."""
        for i, row in enumerate(X):
            # Annoy requires full vectors — for simplicity we skip rows with NaNs
            # Adds item `i` (any nonnegative integer) with vector `v`.
            # Note that it will allocate memory for `max(i)+1` items.
            if not np.isnan(row).any():
                yield i, row  # v.tolist()

    # ------------------------------------------------------------------ #
    # Build Annoy / Voyager index indices
    # ------------------------------------------------------------------ #
    def _fit_annoy_index(self, X: np.ndarray) -> None:
        """
        Build an :class:`~.AnnoyIndex` from rows without missing values.

        Defines:

        * :attr:`temp_fill_vector_` (per-feature statistics)
        * index storage via :meth:`_store_index` according to
          :attr:`index_access` (in-memory or external file).
        """
        with Timer("Building `AnnoyIndex` for nearest neighbors approximate search..."):
            # Path used to *persist* the index when index_access='external'
            external_path = self._resolve_index_store_path()

            # Path used only for on-disk build (may be temporary)
            build_path = None
            temp_build_path = None
            # if getattr(self, "on_disk_build", None):
            if self.on_disk_build:
                if external_path is not None:
                    # external mode: build directly into the final file
                    build_path = external_path
                else:
                    # public/private mode: create a temporary backing file
                    build_path = make_temp_path(prefix="ANNImputer-", ext=".annoy")

            # Temporarily fill NaNs (mean/median/etc.) before index build
            X = self._apply_fill_vector(X)

            # ----- Initialise annoy.AnnoyIndex -----
            # Annoy requires full vectors — for simplicity we skip rows with NaNs
            # determine valid any row not nan
            train_index = AnnoyIndex(
                f=X.shape[1],
                metric=self._resolve_metric(),  # or 'angular'
            )

            # Set random seed if provided
            if self.random_state is not None:
                train_index.set_seed(self.random_state)

            # Configure on-disk build if requested
            if build_path is not None:
                train_index.on_disk_build(build_path)

            # Add all valid (non-NaN) rows
            for i, row in self._iter_non_nan_rows(X):
                train_index.add_item(i, row)

            if train_index.get_n_items() == 0:
                # No valid rows to index → do not attempt KNN imputation
                raise ValueError(
                    f"{self.__class__.__name__}: no valid rows to build an Annoy index. "
                    "All rows contained missing values after preprocessing. "
                    "KNN-based imputation is not available in this configuration."
                )

            # Build forest of trees; -1 delegates to Annoy default
            train_index.build(self.n_trees or -1, self.n_jobs or -1)  # n_trees

            # Delegate storage behaviour (in-memory vs external file)
            self._store_index(
                train_index,
                public_name="train_index_",
                index_path=external_path,
            )

            # Clean up temporary build file (public/private + on_disk_build)
            if temp_build_path is not None:
                # try: os.remove(build_path)
                # except OSError: pass
                with contextlib.suppress(OSError):
                    os.remove(temp_build_path)

    def _fit_voyager_index(self, X):
        """
        Build a :class:`voyager.Index` using temporarily filled vectors.

        Defines:

        * :attr:`temp_fill_vector_` (per-feature statistics)
        * index storage via :meth:`_store_index` according to
          :attr:`index_access` (in-memory or external file).
        """
        self._ensure_voyager_available()

        with Timer(
            "Building `voyager.Index` for nearest neighbors approximate search..."
        ):
            # Path used to *persist* the index when index_access='external'
            external_path = self._resolve_index_store_path()

            # Temporarily fill NaNs (mean/median/etc.) before index build
            X = self._apply_fill_vector(X)
            # Add all samples to index, voyager expects
            X_32 = np.asarray(X, dtype=np.float32)

            # ----- Initialise voyager.AnnoyIndex -----
            train_index = voyager.Index(
                space=self._resolve_voyager_space(),
                num_dimensions=X_32.shape[1],
                random_seed=self.random_state,
            )

            for i, row in self._iter_non_nan_rows(X_32):
                train_index.add_item(row, i)

            if len(train_index) == 0:
                raise ValueError(
                    f"{self.__class__.__name__}: no valid rows to build a voyager "
                    "index. KNN-based imputation is not available in this "
                    "configuration."
                )

            # 4) Delegate storage (in-memory vs external file)
            self._store_index(
                train_index,
                public_name="train_index_",
                index_path=external_path,
            )

    def _build_index_for_backend(self, X_for_index: np.ndarray) -> None:
        """Backend dispatcher for index building (single extension point)."""
        backend = getattr(self, "backend", "annoy")
        if backend == "annoy":
            self._fit_annoy_index(X_for_index)
        elif backend == "voyager":
            self._fit_voyager_index(X_for_index)
        else:  # pragma: no cover - guarded by _parameter_constraints
            raise ValueError(
                f"{self.__class__.__name__}: unsupported backend={backend!r}. "
                "Expected one of {'annoy', 'voyager'}."
            )

    # ------------------------------------------------------------------ #
    # Fit: compute fill statistics and build ANN index
    # ------------------------------------------------------------------ #
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the imputer on X and build the underlying ANN index.

        This step:

        * validates the input data,
        * records which features are completely empty,
        * builds the backend-specific ANN index, and
        * fits the missing-value indicator (if enabled).
        """
        # Determine how to handle NaNs
        # X = check_array(X, force_all_finite="allow-nan")
        if is_scalar_nan(self.missing_values):  # noqa: SIM108
            ensure_all_finite = "allow-nan"
        else:
            ensure_all_finite = True
        # Validate input (numeric, no sparse)
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            ensure_all_finite=ensure_all_finite,
            copy=self.copy,
            # force_writeable=True,
            # reset=False,
        )

        # Track sample and feature counts (sklearn convention)
        # n_features_in_ is handled by _BaseImputer; we add n_samples_fit_
        self.n_samples_fit_ = X.shape[0]

        # Boolean mask for missing entries
        X_missing_mask = _get_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()

        # Keep track of non-valid (all-missing) features
        # determine non-valid all col is nan
        self._is_empty_feature = np.all(mask_missing_values, axis=0)

        # For index construction we always encode missingness as np.nan
        X = np.asarray(X, dtype=float, copy=True)
        if not is_scalar_nan(self.missing_values):
            X[X_missing_mask] = np.nan

        # Build backend-specific index
        self._build_index_for_backend(X)

        # Fit missing value indicator transformer (if used)
        super()._fit_indicator(X_missing_mask)
        return self

    # ------------------------------------------------------------------ #
    # Transform Annoy index helper
    # ------------------------------------------------------------------ #
    # def _transform_annoy_index(self, X, missing_mask):  # noqa: PLR0912
    #     # Iterate over rows with missing values
    #     for i, row in enumerate(X):
    #         if not np.any(missing_mask[i]):
    #             continue  # skip complete rows has no missing values
    #         try:
    #             vec = np.where(
    #                 np.isnan(
    #                     np.nan_to_num(row, nan=np.nan, posinf=np.nan, neginf=np.nan),
    #                 ),
    #                 self.temp_fill_vector_,
    #                 row,
    #             )
    #         except:
    #             # vec = row
    #             vec = np.where(np.isnan(row), self.temp_fill_vector_, row)
    #         # Query Annoy neighbors (fill NaNs with precomputed values), returns the n closest items
    #         # find neighbours using the observed parts: simplest approach uses only complete rows
    #         potential_donors_idx, annoy_dists_1d = self._annoy_index.get_nns_by_vector(
    #             vec,  # row, X[i]
    #             self.n_neighbors,
    #             search_k=self.search_k,
    #             include_distances=True,
    #         )
    #         # Remove self-neighbors (distance == 0)
    #         # But Annoy can return distance 0 even for non-identical vectors (if vectors coincide).
    #         # while 0 in dists:
    #         # That's safer and avoids dropping legitimate zero-distance donors (rare but possible in normalized data).
    #         # if i in potential_donors_idx:
    #         #     idx = dists.index(0)
    #         #     potential_donors_idx.pop(idx)
    #         #     dists.pop(idx)
    #         if not potential_donors_idx:
    #             continue

    #         # Convert to 2D so _get_weights expects shape (n_queries, n_neighbors)
    #         # np.asarray(annoy_dists_1d, dtype=float).reshape(1, -1)
    #         dists_2d = np.atleast_2d(np.asarray(annoy_dists_1d, dtype=float))
    #         weights_2d = _get_weights(dists_2d, self.weights)

    #         if weights_2d is not None:
    #             # Replace inf values (e.g. due to zero distances) with 1.0
    #             if np.isinf(weights_2d).any():
    #                 weights_2d[np.isinf(weights_2d)] = 1.0  # np.isfinite

    #             # Normalize weights row-wise (avoid division by zero)
    #             # if np.any(weights_2d):
    #             row_sums = weights_2d.sum(axis=1, keepdims=True)
    #             weights_2d /= np.where(row_sums == 0, 1, row_sums)

    #         # Fall back to uniform if None
    #         weights = (
    #             np.ones_like(annoy_dists_1d)
    #             if weights_2d is None
    #             else weights_2d.ravel()
    #         )

    #         # Retrieve donor samples
    #         # Compute mean for missing columns from neighbor values
    #         # neighbors = self._fit_X[potential_donors_idx]
    #         # Get vector dimension from Annoy index
    #         _dim = self._annoy_index.f
    #         # Preallocate array
    #         neighbors = np.empty((len(potential_donors_idx), _dim), dtype=np.float32)
    #         # Fill the array using indices from the list
    #         for _i, idx in enumerate(potential_donors_idx):
    #             neighbors[_i] = self._annoy_index.get_item_vector(idx)
    #         # if not neighbors:
    #         #     continue

    #         # Impute missing features
    #         for j, is_nan in enumerate(missing_mask[i]):
    #             if is_nan and ~self._is_empty_feature[j]:
    #                 valid_idx = ~np.isnan(neighbors[:, j])
    #                 col_vals = neighbors[valid_idx, j]
    #                 w = weights[valid_idx][: len(col_vals)]
    #                 if col_vals.size:
    #                     # Weighted average or mean
    #                     if w.sum() == 0:
    #                         # fallback to unweighted mean or fill value
    #                         X[i, j] = np.mean(col_vals)  # or self.temp_fill_vector_[j]
    #                     else:
    #                         X[i, j] = np.average(col_vals, weights=w)
    #                 else:
    #                     # Fallback to fill value if no neighbor values
    #                     X[i, j] = self.temp_fill_vector_[j]
    #                     # pass
    #     return X

    def _compute_neighbor_weights(
        self,
        dists: np.ndarray,
        weights_method,
    ) -> np.ndarray:
        """
        Compute 1D weights from neighbor distances with safe handling of inf.

        Parameters
        ----------
        dists : ndarray of shape (n_neighbors,)
            Distances to neighbor points.
        weights_method : {'uniform', 'distance'} or callable or None
            Weighting strategy as accepted by :func:`_get_weights`.

        Returns
        -------
        weights : ndarray of shape (n_neighbors,)
            Normalized weights (sum up to 1 when possible). Falls back to
            uniform when `weights_method` is ``None`` or if `_get_weights`
            returns ``None``.
        """
        dists = np.asarray(dists, dtype=float).reshape(-1)
        # if dists.size == 0: return dists

        if weights_method is None or weights_method == "uniform":
            return np.ones_like(dists, dtype=float)

        # Use sklearn's helper; it expects shape (n_queries, n_neighbors)
        # Replace inf values (e.g. due to zero distances) with 1.0
        dists_2d = dists.reshape(1, -1)
        weights_2d = _get_weights(dists_2d, weights_method)

        if weights_2d is None:
            return np.ones_like(dists, dtype=float)

        # Ensure final weights is 1D
        weights = np.asarray(weights_2d, dtype=float).reshape(-1)

        # If there are infinite weights (e.g. zero distance), make them
        # dominate but keep things finite: give them equal weight and
        # zero out the others.
        if np.isinf(weights).any():
            inf_mask = np.isinf(weights)
            n_inf = inf_mask.sum()
            if n_inf > 0:
                weights = np.where(inf_mask, 1.0 / n_inf, 0.0)
                return weights  # noqa: RET504

        s = weights.sum()
        if s > 0:
            weights /= s

        return weights

    def _impute_from_neighbors(
        self,
        row_idx: int,
        row: np.ndarray,
        row_missing_mask: np.ndarray,
        neighbors: np.ndarray,
        weights: np.ndarray,
        fill_vec: np.ndarray,
        is_empty_feature: np.ndarray,
    ) -> np.ndarray:
        """
        Impute missing entries in `row` using neighbor vectors + weights.

        Parameters
        ----------
        row_idx : int
            Index of the row in the batch.
        row : (n_features,)
            Original row (with NaNs for missing values).
        row_missing_mask : ndarray of shape (n_features,)
            Boolean mask for missing values in this row.
        neighbors : ndarray of shape (n_neighbors, n_features)
            Neighbor feature values.
        weights : ndarray of shape (n_neighbors,)
            Neighbor weights (typically normalized to sum to 1).
        fill_vec : ndarray of shape (n_features,)
            Global per-feature statistics used as fallback.
        is_empty_feature : ndarray of shape (n_features,)
            Boolean mask marking features that were entirely missing at fit time.

        Returns
        -------
        new_row : ndarray of shape (n_features,)
            The imputed row.
        """
        new_row = row.copy()

        for j, is_nan in enumerate(row_missing_mask):
            # if is_nan and ~self._is_empty_feature[j]:
            if not is_nan or is_empty_feature[j]:
                continue

            col_vals = neighbors[:, j]
            valid = ~np.isnan(col_vals)
            vals = neighbors[valid, j]
            if vals.size == 0:
                # No neighbor information → fallback to global statistic
                new_row[j] = fill_vec[j]
                continue

            w = weights[valid]
            w_sum = w.sum()
            if w_sum > 0:
                new_row[j] = np.average(vals, weights=w)
            else:
                new_row[j] = vals.mean()

        return new_row

    # ------------------------------------------------------------------ #
    # Per-row backends (Annoy / Voyager)
    # ------------------------------------------------------------------ #
    def _process_single_row_annoy(  # noqa: PLR0912, PLR0913
        self,
        i: int,
        row: np.ndarray,
        row_missing_mask: np.ndarray,
        train_index,
        fill_vec: np.ndarray,
        n_neighbors: int,
        search_k: int,
        weights_method,
        is_empty_feature: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """
        Impute missing values for a single row using an Annoy-based index.

        Parameters
        ----------
        i : int
            Row index in ``X``.
        row : ndarray of shape (n_features,)
            The input row (possibly containing NaNs).
        row_missing_mask : ndarray of shape (n_features,)
            Boolean mask for missing values in this row.
        train_index : AnnoyIndex
            Fitted Annoy index built on (temporarily) complete vectors.
        fill_vec : ndarray of shape (n_features,)
            Global per-feature fill statistics used as a fallback.
        n_neighbors : int
            Number of neighbors to retrieve from the Annoy index.
        search_k : int
            Annoy search parameter controlling the accuracy/speed trade-off.
        weights_method : {'uniform', 'distance'} or callable
            Weighting strategy passed to :func:`sklearn.neighbors._base._get_weights`.
        is_empty_feature : ndarray of shape (n_features,)
            Boolean mask marking features that were entirely missing at fit time.

        Returns
        -------
        idx : int
            The original row index ``i``.
        new_row : ndarray of shape (n_features,)
            The imputed row.
        """
        # skip complete rows has no missing values
        if not np.any(row_missing_mask):
            return i, row  # nothing to impute

        # 1. Prepare query vector
        vec = self._prepare_query_vector(row, fill_vec)

        # 2. Query neighbors (backend may be our wrapper or raw spotify Annoy)
        query_kwargs = {
            "n": n_neighbors,
            "search_k": search_k,  # If -1, defaults to approximately n_trees * n
            "include_distances": True,  # (weights == "distance")
        }
        if hasattr(train_index, "query_vectors_by_vector"):
            # scikit-plot wrapper API
            query_kwargs.pop("n", None)
            query_kwargs["n_neighbors"] = n_neighbors
            query_kwargs["exclude_self"] = True
            query_kwargs["output_type"] = "item"
            neighbor_ids, dists = train_index.query_vectors_by_vector(
                vec,  # row, X[i]
                **query_kwargs,
            )
        else:
            # raw spotify/annoy API
            neighbor_ids, dists = train_index.get_nns_by_vector(
                vec,  # row, X[i]
                **query_kwargs,
            )
        # Annoy returns the query point itself as the first element
        # Remove self-neighbors (distance == 0)
        if not list(neighbor_ids):  # python list or not neighbor_ids
            return i, row  # nothing to impute

        # Convert to 2D so _get_weights expects shape (n_queries, n_neighbors)
        # np.asarray(dists, dtype=float).reshape(1, -1)
        # dists_2d = np.atleast_2d(np.asarray(dists, dtype=float))
        # weights_2d = _get_weights(dists_2d, self.weights)

        # Ensure dists is always 1D (even when a scalar sneaks in)
        neighbor_ids = np.asarray(neighbor_ids, dtype=int).ravel()
        dists = np.asarray(dists, dtype=float).reshape(-1)

        if neighbor_ids.size == 0:
            return i, row
        if dists.size != len(neighbor_ids):
            raise ValueError(
                f"Annoy returned mismatched ids/distances lengths: "
                f"{len(neighbor_ids)} vs {dists.size}"
            )

        # 4. Retrieve neighbor rows
        neighbors = np.asarray(
            [train_index.get_item_vector(int(idx)) for idx in neighbor_ids],
            dtype=float,
        )
        if dists.size != neighbors.shape[0]:
            raise ValueError(
                f"Annoy returned mismatched ids/distances lengths: "
                f"{neighbors.shape[0]} vs {dists.size}"
            )

        # 4. Compute weights (strict shape safe), If no distances, fallback to uniform
        weights = self._compute_neighbor_weights(dists, weights_method)

        # 5. Impute row
        new_row = self._impute_from_neighbors(
            row=row,
            row_idx=i,
            row_missing_mask=row_missing_mask,
            neighbors=neighbors,
            weights=weights,
            fill_vec=fill_vec,
            is_empty_feature=is_empty_feature,
        )
        return i, new_row

    def _process_single_row_voyager(  # noqa: PLR0912
        self,
        i: int,
        row: np.ndarray,
        row_missing_mask: np.ndarray,
        train_index,
        fill_vec: np.ndarray,
        n_neighbors: int,
        search_k: int,
        weights_method,
        is_empty_feature: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """
        Impute missing values for a single row using a voyager.Index.
        """
        if not np.any(row_missing_mask):
            return i, row  # nothing to impute

        self._ensure_voyager_available()

        # 1. Prepare query vector
        vec = self._prepare_query_vector(row, fill_vec).astype(np.float32, copy=False)

        # 2. Query neighbors
        query_kwargs = {
            "k": n_neighbors,
            "num_threads": self.n_jobs or -1,  # n_jobs controls outer-level parallelism
            "query_ef": search_k,  # try to find up the k nearest neighbors per query.
        }
        # query(vectors: ndarray[Any, dtype[float32]], k: int = 1, num_threads: int = -1,
        # query_ef: int = -1)
        neighbor_ids, dists = train_index.query(
            vec,  # row, X[i],
            **query_kwargs,
        )

        # ensure 1D
        neighbor_ids = np.asarray(neighbor_ids, dtype=int).reshape(-1)
        dists = np.asarray(dists, dtype=float).reshape(-1)

        if neighbor_ids.size == 0:
            return i, row
        # if dists.size != len(neighbor_ids):
        #     raise ValueError(
        #         f"voyager returned mismatched ids/distances lengths: "
        #         f"{len(neighbor_ids)} vs {dists.size}"
        #     )

        # 3. Retrieve neighbor vectors
        neighbors = train_index.get_vectors(neighbor_ids.tolist()).astype(float)
        if neighbors.ndim == 1:
            neighbors = neighbors.reshape(1, -1)

        if dists.size != neighbors.shape[0]:
            raise ValueError(
                f"voyager returned mismatched ids/distances lengths: "
                f"{neighbors.shape[0]} vs {dists.size}"
            )

        # 4. Compute weights
        weights = self._compute_neighbor_weights(dists, weights_method)

        # 5. Impute row
        new_row = self._impute_from_neighbors(
            row=row,
            row_idx=i,
            row_missing_mask=row_missing_mask,
            neighbors=neighbors,
            weights=weights,
            fill_vec=fill_vec,
            is_empty_feature=is_empty_feature,
        )
        return i, new_row

    # ------------------------------------------------------------------ #
    # Transform backends (Annoy / Voyager)
    # ------------------------------------------------------------------ #
    def _transform_annoy_index(self, X: np.ndarray, missing_mask: np.ndarray):
        """
        Impute missing values in ``X`` using a fitted Annoy index.

        This helper:

        - obtains an index for runtime use via :meth:`_get_index_for_runtime`
          (in-memory or external depending on :attr:`index_access`),
        - parallelises neighbour lookups over rows using :mod:`joblib`,
        - writes the imputed rows back into ``X`` in-place.
        """
        with Timer("Transforming with Annoy..."):

            def _loader(path: str):
                # Reconstruct an Annoy index from a saved file.
                # We use the current feature dimension and metric.
                dim = X.shape[1]
                idx = AnnoyIndex(dim, self._resolve_metric())
                idx.load(path)
                return idx

            train_index = self._get_index_for_runtime(
                public_name="train_index_",
                loader=_loader,
            )

            # Iterate over rows with missing values
            # for i, row in enumerate(X):
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_single_row_annoy)(
                    i,
                    X[i],
                    missing_mask[i],
                    train_index,
                    self.temp_fill_vector_,
                    self.n_neighbors,
                    self.search_k,
                    self.weights,
                    self._is_empty_feature,
                )
                for i in range(X.shape[0])
            )
            # Write updated rows back
            for i, new_row in results:
                X[i] = new_row
            return X

    def _transform_voyager_index(self, X, missing_mask):
        """Impute missing values in ``X`` using a fitted voyager.Index."""
        self._ensure_voyager_available()

        with Timer("Transforming with voyager..."):

            def _loader(path: str):
                # Load voyager index from disk for `index_access='external'`.
                return voyager.Index.load(path)

            train_index = self._get_index_for_runtime(
                public_name="train_index_",
                loader=_loader,
            )

            # results = Parallel(n_jobs=self.n_jobs)(
            #     delayed(self._process_single_row_voyager)(
            #         i,
            #         X[i],
            #         missing_mask[i],
            #         train_index,
            #         self.temp_fill_vector_,
            #         self.n_neighbors,
            #         self.search_k,
            #         self.weights,
            #         self._is_empty_feature,
            #     )
            #     for i in range(X.shape[0])
            # )
            # for i, new_row in results:
            #     X[i] = new_row
            for i in range(X.shape[0]):
                X[i] = self._process_single_row_voyager(
                    i,
                    X[i],
                    missing_mask[i],
                    train_index,
                    self.temp_fill_vector_,
                    self.n_neighbors,
                    self.search_k,
                    self.weights,
                    self._is_empty_feature,
                )[1]
            return X

    # ------------------------------------------------------------------ #
    # Transform: approximate KNN imputation
    # ------------------------------------------------------------------ #
    # def _impute_with_global_stats(self, X, mask):
    #     # mask True olan yerlere temp_fill_vector_ değerlerini yaz
    #     X = X.copy()  # veya self.copy kontrolüyle
    #     X[mask] = np.where(
    #         mask[mask],
    #         self.temp_fill_vector_[np.newaxis, :].repeat(X.shape[0], axis=0)[mask],
    #         X[mask],
    #     )
    #     return X
    def transform(self, X):  # noqa: PLR0912
        """Impute missing values in X using approximate nearest neighbors."""
        # check_is_fitted(self)
        # Ensure we've been fitted (at least temp_fill_vector_)
        check_is_fitted(self, "temp_fill_vector_")
        # if getattr(self, "_annoy_index", None) is None:
        #     raise ValueError("Annoy Index not Fitted.")
        # check_is_fitted(self, "_annoy_index")
        # Determine how to handle NaNs
        # X = check_array(X, force_all_finite="allow-nan")
        if is_scalar_nan(self.missing_values):  # noqa: SIM108
            ensure_all_finite = "allow-nan"
        else:
            ensure_all_finite = True
        # Validate input (numeric, no sparse)
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            force_writeable=True,
            ensure_all_finite=ensure_all_finite,
            copy=self.copy,
            reset=False,
        )
        # Boolean mask for missing entries
        missing_mask = _get_mask(X, self.missing_values)
        # Compute indicator matrix
        X_indicator = super()._transform_indicator(missing_mask)

        # If no missing values in current valid features → skip imputation
        # Removes columns where the training data is all nan
        if not np.any(missing_mask[:, ~self._is_empty_feature]):
            # No missing values in X
            if self.keep_empty_features:
                X_cur = X
                X_cur[:, self._is_empty_feature] = 0
            else:
                X_cur = X[:, ~self._is_empty_feature]

            # Even if there are no missing values in X, we still concatenate Xc
            # with the missing value indicator matrix, X_indicator.
            # This is to ensure that the output maintains consistency in terms
            # of columns, regardless of whether missing values exist in X or not.
            return super()._concatenate_indicator(X_cur, X_indicator)

        # Backend-specific transform
        if self.backend == "annoy":
            self._transform_annoy_index(X, missing_mask)
        elif self.backend == "voyager":
            self._transform_voyager_index(X, missing_mask)
        else:  # pragma: no cover - guarded by _parameter_constraints
            raise ValueError(
                f"{self.__class__.__name__}: unsupported backend={self.backend!r}. "
                "Expected one of {'annoy', 'voyager'}."
            )

        # Restore any empty features if requested
        # Handle empty features if requested
        if self.keep_empty_features:
            X_cur = X
            X_cur[:, self._is_empty_feature] = 0
        else:
            X_cur = X[:, ~self._is_empty_feature]
        # Add indicator columns if enabled
        # Concatenate indicator if used
        return super()._concatenate_indicator(X_cur, X_indicator)

    # ------------------------------------------------------------------ #
    # Feature name support
    # ------------------------------------------------------------------ #
    def get_feature_names_out(self, input_features=None):
        """Return output feature names, including indicator features if used."""
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(self, input_features)
        names = input_features[~self._is_empty_feature]
        return self._concatenate_indicator_feature_names_out(names, input_features)


AnnoyKNNImputer = ANNImputer  # backward compat for Annoy-based imputer


class VoyagerKNNImputer(ANNImputer):
    """
    Shorthand for :class:`~.ANNImputer` with ``backend='voyager'``.

    This class simply fixes the :attr:`backend` parameter to ``'voyager'``
    while exposing the same public API as :class:`~.ANNImputer`.
    """

    def __init__(self, *args, **kwargs):
        backend = kwargs.pop("backend", "voyager")
        if backend != "voyager":
            raise ValueError(
                "VoyagerKNNImputer always uses backend='voyager'. "
                "Do not override the `backend` parameter."
            )
        super().__init__(*args, backend=backend, **kwargs)
