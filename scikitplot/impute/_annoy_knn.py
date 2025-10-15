# ruff: noqa: F401
# pylint: disable=unused-import

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Approximate KNN-based [1]_ imputation using Annoy (Approximate Nearest Neighbors Oh Yeah in C++/Python).

References
----------
.. [1] http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor
.. [2] https://github.com/spotify/annoy
"""

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

# annoy
try:
    from annoy2 import AnnoyIndex
except Exception:
    from ..cexternals.annoy import AnnoyIndex

__all__ = [
    # 'AnnoyIndex',
    "AnnoyKNNImputer",
]


# def _get_weights(dist, weights):
#     """Get the weights from an array of distances and a parameter ``weights``.

#     Assume weights have already been validated.

#     Parameters
#     ----------
#     dist : ndarray
#         The input distances.

#     weights : {'uniform', 'distance'}, callable or None
#         The kind of weighting used.

#     Returns
#     -------
#     weights_arr : array of the same shape as ``dist``
#         If ``weights == 'uniform'``, then returns None.
#     """
#     if weights in (None, "uniform"):
#         return None

#     if weights == "distance":
#         # if user attempts to classify a point that was zero distance from one
#         # or more training points, those training points are weighted as 1.0
#         # and the other points as 0.0
#         if dist.dtype is np.dtype(object):
#             for point_dist_i, point_dist in enumerate(dist):
#                 # check if point_dist is iterable
#                 # (ex: RadiusNeighborClassifier.predict may set an element of
#                 # dist to 1e-6 to represent an 'outlier')
#                 if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
#                     dist[point_dist_i] = point_dist == 0.0
#                 else:
#                     dist[point_dist_i] = 1.0 / point_dist
#         else:
#             with np.errstate(divide="ignore"):
#                 dist = 1.0 / dist
#             inf_mask = np.isinf(dist)
#             inf_row = np.any(inf_mask, axis=1)
#             dist[inf_row] = inf_mask[inf_row]
#         return dist

#     if callable(weights):
#         return weights(dist)


# --------------------------------------------------------------------------- #
# Approximate KNN-based imputer using Annoy (inherits sklearn _BaseImputer)
# --------------------------------------------------------------------------- #
# class _BaseImputer(TransformerMixin, BaseEstimator):
class AnnoyKNNImputer(_BaseImputer):
    r"""
    Fast approximate KNN-based imputation using Spotify's Annoy library.

    This imputer replaces the exact neighbor search of
    :class:`~sklearn.impute.KNNImputer` with a tree-based
    approximate nearest neighbor index (Annoy), providing
    significant scalability improvements on large datasets.

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to np.nan, since `pd.NA` will be converted to np.nan.

    n_trees : int, default=-1
        Number of trees in the Annoy forest. More trees improve neighbor
        accuracy at the cost of build time and memory.
        If -1, trees are built dynamically until the index reaches roughly
        twice the number of items (heuristic: `_n_nodes >= 2 * n_items`).
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

    metric : {'angular', 'euclidean', 'manhattan', 'hamming', 'dot'}, default='euclidean'
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

    index_nan_strategy : {'mean', 'median', 'skip'} or None, default='skip'
        Strategy to handle NaNs when building the Annoy index.
        Rows containing NaNs cannot be indexed directly. The temporary fill
        affects only index construction, not the final imputed values.

        - `'mean'` : fill NaNs with the column mean.
        - `'median'` : fill NaNs with the column median.
        - `'skip'` or `None` : skip rows with NaNs during index build.

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

    n_jobs : int or None, default=None
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

    Notes
    -----
    For each sample :math:`x_i` and feature :math:`j`, the imputed value is:

    .. math::
        \hat{x}_{ij} = \frac{\sum_{k \in N_i} w_{ik} x_{kj}}{\sum_{k \in N_i} w_{ik}}

    where :math:`N_i` is the set of K nearest neighbors of :math:`x_i`,
    and :math:`w_{ik}` is the neighbor weight:

    .. math::
        w_{ik} = \frac{1}{1 + d(x_i, x_k)}

    - Annoy provides *approximate* neighbor search, so imputations are not exact.
    - The index remains in memory after fitting for efficient queries.
    - Annoy supports specific metrics; `'euclidean'` (p=2) and `'manhattan'` (p=1)
      are special cases of the Minkowski distance.

    See Also
    --------
    sklearn.impute.KNNImputer : Multivariate imputer that estimates missing features using
        nearest samples. Exact KNN-based imputer using brute-force search.

    References
    ----------
    .. [1] `Bernhardsson, E. (2013). "Annoy: Approximate Nearest Neighbors Oh Yeah."
       Spotify Engineering. https://github.com/spotify/annoy
       <https://github.com/spotify/annoy>`_

    Examples
    --------
    >>> import numpy as np
    >>> from scikitplot.impute import AnnoyKNNImputer
    >>> X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
    >>> imputer = AnnoyKNNImputer(n_trees=5, n_neighbors=5)
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
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
        "index_nan_strategy": [
            StrOptions({"mean", "median", "skip"}),
            # Hidden(None),  # allow None
            None,
        ],
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
        n_trees=-1,
        search_k=-1,
        n_neighbors=5,
        weights="uniform",
        metric="euclidean",
        index_nan_strategy="skip",
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
        n_jobs=None,
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
        self.index_nan_strategy = index_nan_strategy
        self.copy = copy
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Core imputation helper (per-column)
    # ------------------------------------------------------------------ #
    # def _calc_impute(
    #     self,
    #     dist_pot_donors,
    #     n_neighbors,
    #     fit_X_col,  # noqa: N803
    #     mask_fit_X_col,  # noqa: N803
    # ):
    #     """
    #     Help function to impute a single column.

    #     Parameters
    #     ----------
    #     dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
    #         Distance matrix between the receivers and potential donors from
    #         training set. There must be at least one non-nan distance between
    #         a receiver and a potential donor.

    #     n_neighbors : int
    #         Number of neighbors to consider.

    #     fit_X_col : ndarray of shape (n_potential_donors,)
    #         Column of potential donors from training set.

    #     mask_fit_X_col : ndarray of shape (n_potential_donors,)
    #         Missing mask for fit_X_col.

    #     Returns
    #     -------
    #     imputed_values: ndarray of shape (n_receivers,)
    #         Imputed values for receiver.
    #     """
    #     # Get donors
    #     # Select n closest donors per receiver
    #     donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
    #         :, :n_neighbors
    #     ]

    #     # Get weight matrix from distance matrix
    #     # Gather distances of selected donors
    #     donors_dist = dist_pot_donors[
    #         np.arange(donors_idx.shape[0])[:, None], donors_idx
    #     ]

    #     # Compute weights (uniform or distance)
    #     # _get_weights() assumes that dist is 2D, shape (n_queries, n_neighbors).
    #     weight_matrix = _get_weights(donors_dist, self.weights)

    #     # fill nans with zeros
    #     # Handle NaNs in weight matrix (replace with zeros)
    #     if weight_matrix is not None:
    #         weight_matrix[np.isnan(weight_matrix)] = 0.0
    #     else:
    #         # If uniform, use ones but mask NaN distances
    #         weight_matrix = np.ones_like(donors_dist)
    #         weight_matrix[np.isnan(donors_dist)] = 0.0

    #     # Retrieve donor values and calculate kNN average
    #     donors = fit_X_col.take(donors_idx)
    #     donors_mask = mask_fit_X_col.take(donors_idx)
    #     # Mask missing donors
    #     donors = np.ma.array(donors, mask=donors_mask)
    #     # Weighted average over donors for each receiver
    #     return np.ma.average(donors, axis=1, weights=weight_matrix).data

    # ------------------------------------------------------------------ #
    # Temporary fill for NaNs (for Annoy index build)
    # ------------------------------------------------------------------ #
    def _fill_missing_temp(self, X):
        """Temporarily fill NaNs (mean/median) before index build."""
        # Compute per-feature fill statistics
        if self.index_nan_strategy == "median":
            self.fill_values_ = np.nanmedian(X, axis=0)
        elif self.index_nan_strategy == "mean":
            self.fill_values_ = np.nanmean(X, axis=0)
        else:
            # Default: leave as NaN
            # self.fill_values_ = np.where(np.nanmedian(X, axis=0), np.nan, np.nan)
            self.fill_values_ = np.full(X.shape[1], np.nan)
            return X
        # Temporary fill missing for index building
        X_filled = np.where(np.isnan(X), self.fill_values_, X)
        return X_filled  # noqa: RET504

    # ------------------------------------------------------------------ #
    # Build Annoy index helper
    # ------------------------------------------------------------------ #
    def _fit_annoy_index(self, X):
        """Build Annoy index using rows without missing values."""
        # Fill missing temporarily
        # Compute per-feature fill statistics
        X = self._fill_missing_temp(X)

        # Annoy requires full vectors — for simplicity we skip rows with NaNs
        # determine valid any row not nan
        # self._valid_mask_any_row = ~np.any(self._mask_fit_X, axis=1)
        # self._valid_rows_ = np.where(self._valid_mask_any_row)[0]

        f = X.shape[1]
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
        annoy_index.build(self.n_trees, self.n_jobs or -1)  # n_trees
        self._annoy_index_ = annoy_index

    # ------------------------------------------------------------------ #
    # Fit: compute fill statistics and build Annoy index
    # ------------------------------------------------------------------ #
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the imputer on X and build Annoy index."""
        # Determine how to handle NaNs
        # X = check_array(X, force_all_finite="allow-nan")
        # ensure_all_finite = "allow-nan" if is_scalar_nan(self.missing_values) else True
        if not is_scalar_nan(self.missing_values):
            ensure_all_finite = True
        else:
            ensure_all_finite = "allow-nan"
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
        self._fit_X = X
        # Boolean mask for missing entries
        self._mask_fit_X = _get_mask(self._fit_X, self.missing_values)
        # Keep track of valid (non-all-missing) features
        # determine valid all col not nan
        self._valid_mask = ~np.all(self._mask_fit_X, axis=0)

        # Build Annoy index on filled data
        self._fit_annoy_index(self._fit_X)
        # Fit missing value indicator transformer (if used)
        super()._fit_indicator(self._mask_fit_X)

        return self

    # ------------------------------------------------------------------ #
    # Transform: approximate KNN imputation
    # ------------------------------------------------------------------ #
    def transform(self, X):  # noqa: PLR0912
        """Impute missing values in X using approximate nearest neighbors."""
        check_is_fitted(self, "_annoy_index_")
        # Determine how to handle NaNs
        # X = check_array(X, force_all_finite="allow-nan")
        # ensure_all_finite = "allow-nan" if is_scalar_nan(self.missing_values) else True
        if not is_scalar_nan(self.missing_values):
            ensure_all_finite = True
        else:
            ensure_all_finite = "allow-nan"
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
        mask = _get_mask(X, self.missing_values)
        mask_fit_X = self._mask_fit_X  # noqa: N806
        valid_mask = self._valid_mask
        # Compute indicator matrix
        X_indicator = super()._transform_indicator(mask)

        # If no missing values in current valid features → skip imputation
        # Removes columns where the training data is all nan
        if not np.any(mask[:, valid_mask]):
            # No missing values in X
            if self.keep_empty_features:
                X_cur = X
                X_cur[:, ~valid_mask] = 0
            else:
                X_cur = X[:, valid_mask]

            # Even if there are no missing values in X, we still concatenate Xc
            # with the missing value indicator matrix, X_indicator.
            # This is to ensure that the output maintains consistency in terms
            # of columns, regardless of whether missing values exist in X or not.
            return super()._concatenate_indicator(X_cur, X_indicator)

        # row_missing_idx = np.flatnonzero(mask[:, valid_mask].any(axis=1))
        # non_missing_fix_X = np.logical_not(mask_fit_X)
        # (potential_donors_idx,) = np.nonzero(non_missing_fix_X[:, col])

        # Maps from indices from X to indices in dist matrix
        # dist_idx_map = np.zeros(X.shape[0], dtype=int)
        # dist_idx_map[row_missing_idx] = np.arange(row_missing_idx.shape[0])

        # Iterate over rows with missing values
        for i, row in enumerate(X):
            # if not valid_mask[col]:
            #     # column was all missing during training
            #     continue

            # col_mask = mask[row_missing_chunk, col]
            # if not np.any(col_mask):
            #     # column has no missing values
            #     continue

            # nan_mask
            if not np.any(mask[i]):
                # row has no missing values
                continue  # skip complete rows
            try:
                vec = np.nan_to_num(
                    np.where(np.isnan(row), self.fill_values_, row),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            except:
                vec = np.where(np.isnan(row), self.fill_values_, row)
            # Query Annoy neighbors (fill NaNs with precomputed values), returns the n closest items
            # find neighbours using the observed parts: simplest approach uses only complete rows
            potential_donors_idx, annoy_dists_1d = self._annoy_index_.get_nns_by_vector(
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
            neighbors = self._fit_X[potential_donors_idx]

            # Impute missing features
            for j, is_nan in enumerate(mask[i]):
                if is_nan and valid_mask[j]:
                    valid_idx = ~np.isnan(neighbors[:, j])
                    col_vals = neighbors[valid_idx, j]
                    w = weights[valid_idx][: len(col_vals)]
                    if col_vals.size:
                        # Weighted average or mean
                        if w.sum() == 0:
                            # fallback to unweighted mean or fill value
                            X[i, j] = np.mean(col_vals)  # or self.fill_values_[j]
                        else:
                            X[i, j] = np.average(col_vals, weights=w)
                    else:
                        # Fallback to fill value if no neighbor values
                        X[i, j] = self.fill_values_[j]

        # Restore any empty features if requested
        # Handle empty features if requested
        if self.keep_empty_features:
            X_cur = X
            X_cur[:, ~valid_mask] = 0
        else:
            X_cur = X[:, valid_mask]
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
        names = input_features[self._valid_mask]
        return self._concatenate_indicator_feature_names_out(names, input_features)
