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


# --------------------------------------------------------------------------- #
# Approximate KNN-based imputer using Annoy (inherits sklearn _BaseImputer)
# --------------------------------------------------------------------------- #
# class _BaseImputer(TransformerMixin, BaseEstimator):
class AnnoyKNNImputer(_BaseImputer):
    r"""
    Fast approximate KNN-based imputation using Spotify's Annoy library.

    This imputer replaces the exact neighbor search of
    :class:`~sklearn.impute.KNNImputer` with a tree-based
    approximate nearest neighbor index found via Annoy,
    dramatically improving scalability on large datasets.

    Supports both uniform and distance-based weighting schemes.

    **Mathematical Formulation**

    For each sample :math:`x_i` and feature :math:`j`:

    .. math::

        \hat{x}_{ij} =
        \\frac{\\sum_{k \\in N_i} w_{ik} x_{kj}}
              {\\sum_{k \\in N_i} w_{ik}}

    where:

    - :math:`N_i` are the *K* nearest neighbors of :math:`x_i`
      (found via Annoy's approximate search)
    - :math:`w_{ik} = 1 / (1 + d(x_i, x_k))` are inverse-distance weights

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to np.nan, since `pd.NA` will be converted to np.nan.

    n_trees : int, default=-1
        Number of Annoy trees. -1 and positive integers.

    n_neighbors : int, default=100
        Number of neighboring samples to use for imputation.

    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction. Possible values:

        - `'uniform'` : uniform weights. All points in each neighborhood are
          weighted equally.
        - `'distance'` : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - `callable` : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    metric : {'angular', 'euclidean', 'manhattan', 'hamming', 'dot'}, default='euclidean'
        Distance metric used for nearest neighbor search. Metric choices:

        - `'angular'` → Cosine similarity (angle between vectors).
        Measures *directional similarity* only — magnitude is ignored.
        Useful for normalized embeddings (e.g., text, image features).

        - `'euclidean'` → L2 distance, defined as √Σ(xᵢ - yᵢ)².
        Standard geometric distance.
        Penalizes large feature differences more strongly.
        Recommended for continuous numeric data.
        Sensitive to feature scale — normalize or standardize inputs before use.

        - `'manhattan'` → L1 distance, defined as Σ|xᵢ - yᵢ|.
        “City-block” distance; more robust to outliers than L2.
        Still sensitive to feature scaling.
        Good alternative when data contains noise or heavy-tailed distributions.

        - `'hamming'` → Fraction or count of differing elements.
        Suitable for binary or categorical integer-encoded data (e.g., 0/1).
        Not meaningful for continuous numeric features.

        - `'dot'` → Negative inner product (-x·y).
        Considers both direction and magnitude of vectors.
        Often used for dense, normalized embeddings where higher dot product
        indicates greater similarity. Sensitive to scaling.

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

    search_k : int or None, default=None
        Number of nodes to inspect during query.
        Defaults to n_trees * n_neighbors.

    include_distances : bool, default=True
        If you set include_distances to True, it will return a 2 element tuple
        with two lists in it: the second one containing all corresponding distances.

    fill_value : {'mean', 'median'} or None, default=None
        When building Annoy index ignores including None rows.
        Strategy to fill temporary NaNs before building Annoy index building,
        especially useful for small dataset.
        If all neighbor distances are NaN for a feature, to support
        a global mean or median can be use.

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
    - Annoy builds an approximate index, so imputations are not exact.
    - After fitting, the Annoy index is kept in memory for fast queries.
    - Annoy itself does not natively support a general `'minkowski'` metric,
      but `'euclidean'` (p=2) and `'manhattan'` (p=1) are special cases of
      the Minkowski distance.

    See Also
    --------
    KNNImputer : Multivariate imputer that estimates missing features using
        nearest samples.

    References
    ----------
    .. [1] Bernhardsson Erik, "Annoy: Approximate Nearest Neighbors Oh Yeah"
      (Spotify Engineering, 2013)
      https://github.com/spotify/annoy

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
            Options(int, {-1}),  # allow -1
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
        "copy": ["boolean"],
        "fill_value": "no_validation",  # any object is valid
        "include_distances": ["boolean"],
        "random_state": ["random_state"],
        # "verbose": ["verbose"],
        # "n_jobs": [None, Integral],
    }

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        missing_values=np.nan,
        n_trees=-1,
        n_neighbors=100,
        weights="uniform",
        metric="euclidean",
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
        search_k=None,
        include_distances=True,
        fill_value=None,
        random_state=None,
        # n_jobs=None,
    ):
        # Initialize superclass (handles indicator, missing value support)
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        # Annoy / imputation hyperparameters
        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.copy = copy
        # Default search_k (controls accuracy/speed tradeoff)
        self.search_k = search_k or (self.n_trees * self.n_neighbors)
        self.include_distances = include_distances
        # Temporary fill strategy for NaNs when building Annoy index
        self.fill_value = fill_value
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Core imputation helper (per-column)
    # ------------------------------------------------------------------ #
    def _calc_impute(
        self,
        dist_pot_donors,
        n_neighbors,
        fit_X_col,  # noqa: N803
        mask_fit_X_col,  # noqa: N803
    ):
        """
        Help function to impute a single column.

        Parameters
        ----------
        dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
            Distance matrix between the receivers and potential donors from
            training set. There must be at least one non-nan distance between
            a receiver and a potential donor.

        n_neighbors : int
            Number of neighbors to consider.

        fit_X_col : ndarray of shape (n_potential_donors,)
            Column of potential donors from training set.

        mask_fit_X_col : ndarray of shape (n_potential_donors,)
            Missing mask for fit_X_col.

        Returns
        -------
        imputed_values: ndarray of shape (n_receivers,)
            Imputed values for receiver.
        """
        # Get donors
        # Select n closest donors per receiver
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]

        # Get weight matrix from distance matrix
        # Gather distances of selected donors
        donors_dist = dist_pot_donors[
            np.arange(donors_idx.shape[0])[:, None], donors_idx
        ]

        # Compute weights (uniform or distance)
        weight_matrix = _get_weights(donors_dist, self.weights)

        # fill nans with zeros
        # Handle NaNs in weight matrix (replace with zeros)
        if weight_matrix is not None:
            weight_matrix[np.isnan(weight_matrix)] = 0.0
        else:
            # If uniform, use ones but mask NaN distances
            weight_matrix = np.ones_like(donors_dist)
            weight_matrix[np.isnan(donors_dist)] = 0.0

        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        # Mask missing donors
        donors = np.ma.array(donors, mask=donors_mask)
        # Weighted average over donors for each receiver
        return np.ma.average(donors, axis=1, weights=weight_matrix).data

    # ------------------------------------------------------------------ #
    # Temporary fill for NaNs (for Annoy index build)
    # ------------------------------------------------------------------ #
    def _fill_missing_temp(self, X):
        """Temporarily fill NaNs (mean/median) before index build."""
        # Compute per-feature fill statistics
        if self.fill_value == "median":
            self.fill_values_ = np.nanmedian(X, axis=0)
        elif self.fill_value == "mean":
            self.fill_values_ = np.nanmean(X, axis=0)
        else:
            # Default: leave as NaN
            self.fill_values_ = np.where(np.nanmedian(X, axis=0), np.nan, np.nan)
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
        f = X.shape[1]
        annoy_index = AnnoyIndex(f, self.metric)
        # Set random seed if provided
        if self.random_state is not None:
            annoy_index.set_seed(self.random_state)
        # Add all samples to index
        for i, row in enumerate(X):
            annoy_index.add_item(i, row)
        # Build forest of trees
        annoy_index.build(self.n_trees)
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

        # determine valid any row not nan
        # self._valid_mask_any_row = ~np.any(self._mask_fit_X, axis=1)
        # self._valid_rows_ = np.where(self._valid_mask_any_row)[0]

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

            # Query Annoy neighbors (fill NaNs with precomputed values)
            # Find neighbors via Annoy, returns the n closest items
            potential_donors_idx, dists = self._annoy_index_.get_nns_by_vector(
                # row,  # X[i]
                np.where(np.isnan(row), self.fill_values_, row),
                self.n_neighbors,
                search_k=self.search_k,
                include_distances=self.include_distances,
            )
            # Remove self-neighbors (distance == 0)
            while 0 in dists:
                idx = dists.index(0)
                potential_donors_idx.pop(idx)
                dists.pop(idx)
            if not potential_donors_idx:
                continue

            # Retrieve donor samples
            # Compute mean for missing columns from neighbor values
            neighbors = self._fit_X[potential_donors_idx]

            # Impute missing features
            for j, is_nan in enumerate(mask[i]):
                if is_nan and valid_mask[j]:
                    col_vals = neighbors[:, j]
                    col_vals = col_vals[~np.isnan(col_vals)]
                    if col_vals.size:
                        # Compute weights
                        weight_matrix = _get_weights(
                            np.asarray(dists)[: len(col_vals)].reshape(1, -1),
                            self.weights,
                        )
                        if weight_matrix is not None:
                            weight_matrix = np.ravel(weight_matrix)
                        # Weighted average or mean
                        # X[i, j] = np.mean(col_vals)
                        X[i, j] = np.average(col_vals, weights=weight_matrix)
                    else:
                        # Fallback to fill value if no neighbor values
                        X[i, j] = self.fill_values_[j]
                        # pass

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
