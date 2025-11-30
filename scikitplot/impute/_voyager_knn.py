# ruff: noqa: F401
# pylint: disable=unused-import

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings
to search for points in space that are close to a given query point.

The originates from Spotify. It uses a forest of random projection trees.

It also creates large read-only file-based data structures that are mmapped into memory
so that many processes may share the same data.

References
----------
.. [1] http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor
.. [2] https://github.com/spotify/annoy
"""  # noqa: D205

# from numbers import Integral

import numpy as np
import pandas as pd

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

try:
    from ..cexternals.annoy import AnnoyIndex
except Exception:
    from annoy import AnnoyIndex

__all__ = [
    "AnnoyKNNImputer",
]


# --------------------------------------------------------------------------- #
# Approximate KNN-based imputer using Voyager (inherits sklearn _BaseImputer)
# --------------------------------------------------------------------------- #
# class _BaseImputer(TransformerMixin, BaseEstimator):
class AnnoyKNNImputer(_BaseImputer):
    r"""
    Fast approximate vector nearest-neighbors-based imputation using the Spotify Voyager library.

    This imputer replaces the exact neighbor search of :class:`
    ~sklearn.impute.KNNImputer` with a approximate nearest neighbor index (Voyager),
    providing significant scalability improvements on large datasets.

    Voyager is a library for performing fast approximate nearest-neighbor searches
    on an in-memory collection of vectors.

    Voyager features bindings to both Python and Java, with feature parity and
    index compatibility between both languages. It uses the HNSW algorithm,
    based on the open-source hnswlib package, with numerous features added
    for convenience and speed. Voyager is used extensively in production at Spotify,
    and is queried hundreds of millions of times per day to power numerous
    user-facing features.

    Think of Voyager like Sparkey, but for vector/embedding data; or like Annoy,
    but with much higher recall. It got its name because it searches through
    (embedding) space(s), much like the Voyager interstellar probes launched by
    NASA in 1977.

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to np.nan, since `pd.NA` will be converted to np.nan.

    n_trees : int, default=10
        Number of trees in the Annoy forest. Increasing the number of trees
        generally improves nearest-neighbor accuracy but increases build time
        and memory usage.

        If set to ``n_trees=-1``, annoy trees are built dynamically
        until the index reaches approximately twice the number of items
        (heuristic: ``_n_nodes >= 2 * n_items``).

        Guidelines:

        - Small datasets (<10k samples): 10-20 trees.
        - Medium datasets (10k-1M samples): 20-50 trees.
        - Large datasets (>1M samples): 50-100+ trees.

    search_k : int or None, default=-1
        Number of nodes inspected during neighbor search.
        Larger values yield more accurate but slower queries.
        If -1, defaults to `n_trees * n_neighbors`.

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

    metric : {'angular', 'euclidean', 'manhattan', 'hamming', 'dot'}, default='angular'
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

    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, default='mean'
        Which strategy to use to initialize the missing values. Same as the
        `strategy` parameter in :class:`~sklearn.impute.SimpleImputer`.

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
        Specifies the number of threads used to build the trees.
        `n_jobs=-1` uses all available CPU cores.

    random_state : int or None, default=None
        Seed for Annoy's random hyperplane generation.

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

    annoy_index_ : Fitted Annoy Index.

    fill_annoy_vector_ : 1d nanmean array for fill annoy vector.

    Notes
    -----
    For each sample :math:`x_i` and feature :math:`j`, the imputed value is:

    .. math::
        \hat{x}_{ij} = \frac{\sum_{k \in N_i} w_{ik} x_{kj}}{\sum_{k \in N_i} w_{ik}}

    where :math:`N_i` is the set of K nearest neighbors of :math:`x_i`,
    and :math:`w_{ik}` is the neighbor weight:

    :math:`w_{ik} = \\frac{1}{1 + d(x_i, x_k)}`

    - Annoy provides approximate neighbor search, so imputations are not exact.
    - Annoy uses random projections to split the vector space at each node in
      the tree, selecting a random hyperplane defined by two sampled points.
    - The index remains in memory after fitting for efficient queries.
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

    References
    ----------
    .. [1] `Bernhardsson, E. (2013). "Annoy: Approximate Nearest Neighbors Oh Yeah."
       Spotify AB. https://github.com/spotify/annoy
       <https://github.com/spotify/annoy>`_

    Examples
    --------
    >>> import numpy as np
    >>> from scikitplot.experimental import enable_annoyknn_imputer
    >>> from scikitplot.impute import AnnoyKNNImputer
    >>> X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
    >>> imputer = AnnoyKNNImputer(n_trees=5, n_neighbors=5)
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 5. ],
           [3. , 4. , 3. ],
           [4. , 6. , 5. ],
           [8. , 8. , 7. ]])
    """

    # Define parameter constraints for sklearn validation
    _parameter_constraints = {  # noqa: RUF012
        **_BaseImputer._parameter_constraints,
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
                    "euclidean",
                    "manhattan",
                    "hamming",
                    "dot",
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
        n_trees=10,  # annoy -1
        search_k=-1,  # annoy default
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
        # Initialize superclass (handles indicator, missing value support)
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        # Annoy / imputation hyperparameters
        self.n_trees = n_trees
        # Default search_k (controls accuracy/speed tradeoff)
        self.search_k = search_k  # or (self.n_trees * self.n_neighbors)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        # Temporary fill strategy for NaNs when building Annoy index
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.copy = copy
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Build Annoy index helper
    # ------------------------------------------------------------------ #
    def _fit_annoy_index(self, X):
        """Build Annoy index using rows without missing values."""
        # Compute per-feature fill statistics
        # self.fill_annoy_vector_ = np.full(X.shape[1], np.nan)
        # self.fill_annoy_vector_ = np.nanmedian(X, axis=0)
        self.fill_annoy_vector_ = np.nanmean(X, axis=0)
        # Temporary fill missing for index building
        if self.initial_strategy == "median":
            self.fill_annoy_vector_ = np.nanmedian(X, axis=0)
        elif self.initial_strategy == "mean":
            self.fill_annoy_vector_ = np.nanmean(X, axis=0)
        else:
            self.fill_annoy_vector_ = np.nanmean(X, axis=0)
        # Temporarily fill NaNs (mean/median) before index build.
        try:
            X = np.where(
                np.isnan(
                    np.nan_to_num(X, nan=np.nan, posinf=np.nan, neginf=np.nan),
                ),
                self.fill_annoy_vector_,
                X,
            )
        except:
            X = np.where(np.isnan(X), self.fill_annoy_vector_, X)

        f = X.shape[1]
        # Annoy requires full vectors — for simplicity we skip rows with NaNs
        # determine valid any row not nan
        annoy_index = AnnoyIndex(f, self.metric)  # or 'euclidean'
        # Set random seed if provided
        if self.random_state is not None:
            annoy_index.set_seed(self.random_state)
        # Add all samples to index
        for i, row in enumerate(X):
            # Annoy requires full vectors — for simplicity we skip rows with NaNs
            # Adds item `i` (any nonnegative integer) with vector `v`.
            # Note that it will allocate memory for `max(i)+1` items.
            if not np.isnan(row).any():
                annoy_index.add_item(i, row)  # v.tolist()
        # Build forest of trees or -1, n_jobs annoy not handle None
        annoy_index.build(self.n_trees or -1, self.n_jobs or -1)  # n_trees
        self.annoy_index_ = annoy_index

    # ------------------------------------------------------------------ #
    # Transform Annoy index helper
    # ------------------------------------------------------------------ #
    def _transform_annoy_index(self, X, initial_mask):  # noqa: PLR0912
        # Iterate over rows with missing values
        for i, row in enumerate(X):
            if not np.any(initial_mask[i]):
                continue  # skip complete rows has no missing values
            try:
                vec = np.where(
                    np.isnan(
                        np.nan_to_num(row, nan=np.nan, posinf=np.nan, neginf=np.nan),
                    ),
                    self.fill_annoy_vector_,
                    row,
                )
            except:
                # vec = row
                vec = np.where(np.isnan(row), self.fill_annoy_vector_, row)
            # Query Annoy neighbors (fill NaNs with precomputed values), returns the n closest items
            # find neighbours using the observed parts: simplest approach uses only complete rows
            potential_donors_idx, annoy_dists_1d = self.annoy_index_.get_nns_by_vector(
                vec,  # row, X[i]
                self.n_neighbors,
                search_k=self.search_k,
                include_distances=True,
            )
            # Remove self-neighbors (distance == 0)
            # But Annoy can return distance 0 even for non-identical vectors (if vectors coincide).
            # while 0 in dists:
            # That's safer and avoids dropping legitimate zero-distance donors (rare but possible in normalized data).
            # if i in potential_donors_idx:
            #     idx = dists.index(0)
            #     potential_donors_idx.pop(idx)
            #     dists.pop(idx)
            if not potential_donors_idx:
                continue

            # Convert to 2D so _get_weights expects shape (n_queries, n_neighbors)
            # np.asarray(annoy_dists_1d, dtype=float).reshape(1, -1)
            dists_2d = np.atleast_2d(np.asarray(annoy_dists_1d, dtype=float))
            weights_2d = _get_weights(dists_2d, self.weights)

            if weights_2d is not None:
                # Replace inf values (e.g. due to zero distances) with 1.0
                if np.isinf(weights_2d).any():
                    weights_2d[np.isinf(weights_2d)] = 1.0  # np.isfinite

                # Normalize weights row-wise (avoid division by zero)
                # if np.any(weights_2d):
                row_sums = weights_2d.sum(axis=1, keepdims=True)
                weights_2d /= np.where(row_sums == 0, 1, row_sums)

            # Fall back to uniform if None
            weights = (
                np.ones_like(annoy_dists_1d)
                if weights_2d is None
                else weights_2d.ravel()
            )

            # Retrieve donor samples
            # Compute mean for missing columns from neighbor values
            # neighbors = self._fit_X[potential_donors_idx]
            # Get vector dimension from Annoy index
            _dim = self.annoy_index_.f
            # Preallocate array
            neighbors = np.empty((len(potential_donors_idx), _dim), dtype=np.float32)
            # Fill the array using indices from the list
            for _i, idx in enumerate(potential_donors_idx):
                neighbors[_i] = self.annoy_index_.get_item_vector(idx)
            # if not neighbors:
            #     continue

            # Impute missing features
            for j, is_nan in enumerate(initial_mask[i]):
                if is_nan and ~self._is_empty_feature[j]:
                    valid_idx = ~np.isnan(neighbors[:, j])
                    col_vals = neighbors[valid_idx, j]
                    w = weights[valid_idx][: len(col_vals)]
                    if col_vals.size:
                        # Weighted average or mean
                        if w.sum() == 0:
                            # fallback to unweighted mean or fill value
                            X[i, j] = np.mean(col_vals)  # or self.fill_annoy_vector_[j]
                        else:
                            X[i, j] = np.average(col_vals, weights=w)
                    else:
                        # Fallback to fill value if no neighbor values
                        X[i, j] = self.fill_annoy_vector_[j]
                        # pass
        return X

    # ------------------------------------------------------------------ #
    # Fit: compute fill statistics and build Annoy index
    # ------------------------------------------------------------------ #
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the imputer on X and build Annoy index."""
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

        # Boolean mask for missing entries
        X_missing_mask = _get_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()
        # Keep track of non-valid (all-missing) features
        # determine non-valid all col is nan
        self._is_empty_feature = np.all(mask_missing_values, axis=0)

        # Annoy requires full vectors — for simplicity we skip rows with NaNs
        # determine valid any row not nan
        # self._valid_mask_any_row = ~np.any(X_missing_mask, axis=1)
        # self._valid_rows_ = np.where(self._valid_mask_any_row)[0]
        # Build Annoy index on filled data
        self._fit_annoy_index(X)
        # Fit missing value indicator transformer (if used)
        super()._fit_indicator(X_missing_mask)

        return self

    # ------------------------------------------------------------------ #
    # Transform: approximate KNN imputation
    # ------------------------------------------------------------------ #
    def transform(self, X):  # noqa: PLR0912
        """Impute missing values in X using approximate nearest neighbors."""
        # check_is_fitted(self)
        check_is_fitted(self, "annoy_index_")
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
        complete_mask = _get_mask(X, self.missing_values)
        # Compute indicator matrix
        X_indicator = super()._transform_indicator(complete_mask)

        # If no missing values in current valid features → skip imputation
        # Removes columns where the training data is all nan
        if not np.any(complete_mask[:, ~self._is_empty_feature]):
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

        self._transform_annoy_index(X, complete_mask)

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
