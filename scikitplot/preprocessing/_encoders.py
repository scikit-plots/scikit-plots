# scikitplot/preprocessing/_encoders.py
#
# flake8: noqa: D213
#
# ruff: noqa
# ruff: noqa: PGH004
# ruff: noqa: D205, D401, D404
# ruff: noqa: N802, N806
# ruff: noqa: UP030, UP032
# ruff: noqa: PLR0912
# ruff: noqa: RET506
# ruff: noqa: SIM102, SIM108
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-column multi-label string column one-hot encoder [1]_.
"""

from __future__ import annotations

import array
import itertools
import numbers
import re
import warnings
from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas

    # from collections import defaultdict
    from collections.abc import (
        # Hashable,
        Iterable,
    )

import numpy as np
import scipy.sparse as sp

# from scipy import sparse
from sklearn.base import (
    BaseEstimator,
    # OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import _safe_indexing, check_array
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils._set_output import _get_output_config
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_is_fitted,
    validate_data,
)

from scikitplot import __version__
from scikitplot.externals._packaging.version import parse as parse_version

scikitplot_version = parse_version(__version__)
if scikitplot_version.dev is None:
    version_url = f"{scikitplot_version.major}.{scikitplot_version.minor}"
else:
    version_url = "dev"

__all__ = [
    # "OneHotEncoder",
    # "OrdinalEncoder",
    "DummyCodeEncoder",
    "GetDummies",
]


# order matters in Python's Method Resolution Order (MRO) then the first one in the list takes precedence.
# print(GetDummies.mro()): [GetDummies, TransformerMixin, BaseEstimator, object]
class GetDummies(TransformerMixin, BaseEstimator):
    """
    Multi-column multi-label string column one-hot encoder [1]_.

    Custom transformer to expand string columns that contain multiple labels
    separated by `sep` into one-hot encoded columns by :func:`pandas.get_dummies`.

    Compatible with sklearn pipelines, `set_output` API, and supports both
    dense and sparse output.

    Parameters
    ----------
    columns : str, list of str or None, default=None
        Column(s) to encode. If str, single column. If list, multiple columns.
        If None, automatically detect object (string) columns containing `sep`.
    sep : str, default "|"
        String to split on (e.g., "a,b,c").
    col_name_sep : str, default="_"
        Separator for new dummy column names, e.g., "tags_a".
    sparse_output : bool, default=False
        If True, return a SciPy sparse CSR matrix.
        If False, return pandas DataFrame (or numpy array if set_output="default").
    handle_unknown : {"ignore", "error"}, default="ignore"
        Strategy for unknown categories at transform time.
    drop : {"first", True, None}, default=None
        Drop the first dummy in each feature (sorted order) to avoid collinearity.
    dtype : number type, default=np.float64
        Data type for the output values. (sklearn default is float)

    See Also
    --------
    DummyCodeEncoder : Same but more extended and support convert to dummy codes to
        :py:class:`scipy.sparse._csr.csr_matrix` compressed Sparse Row matrix.
    pandas.Series.str.get_dummies : Convert Series of strings to dummy codes.
    pandas.from_dummies : Convert dummy codes back to categorical DataFrame.
    sklearn.preprocessing.OneHotEncoder : General-purpose one-hot encoder.
    sklearn.preprocessing.MultiLabelBinarizer : Multi-label binarizer for
        iterable of iterables.

    References
    ----------
    .. [1] `Çelik, M. (2023, December 9).
       "How to converting pandas column of comma-separated strings into dummy variables?."
       Medium. https://medium.com/@celik-muhammed/how-to-converting-pandas-column-of-comma-separated-strings-into-dummy-variables-762c02282a6c
       <https://medium.com/@celik-muhammed/how-to-converting-pandas-column-of-comma-separated-strings-into-dummy-variables-762c02282a6c>`_

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "tags": ["a,b,", " A , b", "a,B,C", None],
    ...         "color": ["red", "blue", "green", "Red"],
    ...         "value": [1, 2, 3, 4],
    ...     }
    ... )
    >>> from sklearn.pipeline import Pipeline
    >>> from scikitplot.preprocessing import GetDummies
    >>> pipe = Pipeline(
    ...     [
    ...         (
    ...             "encoder",
    ...             GetDummies(
    ...                 columns=["tags", "color"], drop=None, sparse_output=False
    ...             ),
    ...         )
    ...     ]
    ... )
    >>> X_trans = pipe.fit_transform(df)
    >>> print(X_trans)
       value  ta_a  ta_b  ta_c  co_blue  co_green  co_red
    0      1   1.0   1.0   0.0      0.0       0.0     1.0
    1      2   1.0   1.0   0.0      1.0       0.0     0.0
    2      3   1.0   1.0   1.0      0.0       1.0     0.0
    3      4   0.0   0.0   0.0      0.0       0.0     1.0
    >>> type(X_trans)
    <class 'pandas.core.frame.DataFrame'>
    """

    def __init__(
        self,
        *,
        columns=None,
        sep=",",  # pandas default '|'
        col_name_sep="_",
        drop=None,
        sparse_output=False,
        dtype=np.float64,
        handle_unknown="error",
    ):
        # NOTE: sklearn contract — __init__ must store ALL params exactly as received.
        # get_params() / clone() / set_params() rely on __init__ param names matching
        # instance attribute names with no mutation.  Normalization (str → list) is
        # done lazily inside fit(), not here.
        self.columns = columns  # str, list, or None — preserved as-is
        # Separator for multiple values in a cell
        self.sep = sep
        # Separator for dummy column names
        self.col_name_sep = col_name_sep
        # Output data type
        self.dtype = dtype
        # Whether to output SciPy sparse matrix
        self.sparse_output = sparse_output
        # Strategy for unknown categories at transform
        self.handle_unknown = handle_unknown
        # Drop first dummy column to avoid multicollinearity
        self.drop = drop

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _to_dataframe(X):
        """
        Ensure convert input into a pandas DataFrame (supports DataFrame, ndarray, sparse).

        Handles dense, NumPy, or sparse inputs.
        - If already DataFrame, copy it.
        - If ndarray, convert to DataFrame.
        """
        import pandas as pd  # noqa: PLC0415
        import scipy.sparse as sp  # SciPy sparse CSR/CSC matrices  # noqa: PLC0415

        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        if sp.issparse(X):
            # Convert sparse input to dense DataFrame (warning: may be memory-heavy)
            return pd.DataFrame(X.toarray())
        raise TypeError(
            "Input must be a pandas DataFrame, NumPy array, or SciPy sparse matrix"
        )

    def _split_and_clean(self, series: pandas.Series) -> pandas.Series:
        """
        Split strings by separator and normalize, preserving NaN positions.

        Each non-NaN entry is lowercased, whitespace-stripped, deduplicated,
        sorted, and rejoined by ``self.sep`` so that ``str.get_dummies`` can
        parse it in a deterministic, order-independent way.

        NaN values are kept as NaN throughout — never coerced to empty string.
        The downstream invariant (NaN row → all-zero dummy row) is enforced
        explicitly in :meth:`_make_dummies`, not here.

        Parameters
        ----------
        series : pd.Series
            Raw string series from a DataFrame column; may contain ``NaN`` /
            ``None`` values.

        Returns
        -------
        pd.Series
            Normalised, separator-rejoined strings for valid entries.
            NaN entries remain NaN, preserving the original index and length.

        Notes
        -----
        Developer note
            **Do NOT replace** ``fillna("")`` here.  In pandas ≥ 3 the empty
            string ``""`` is a valid, non-null value.  ``str.get_dummies``
            treats it as a real token and produces a spurious ``""`` dummy
            column, giving the NaN row a ``1`` instead of all zeros.

            The NaN → zero-row contract is owned entirely by ``_make_dummies``,
            which captures the NaN mask before calling this method and zeroes
            those rows explicitly after dummification.
        """
        nan_mask = series.isna()
        result = series.copy()

        valid = ~nan_mask
        if valid.any():
            result[valid] = (
                series[valid]
                .str.split(self.sep)
                .apply(
                    lambda lst: self.sep.join(
                        sorted({s.strip().lower() for s in lst if s.strip()})
                    )
                )
            )
        # NaN rows remain NaN — _make_dummies enforces the zero-row invariant.
        return result

    def _make_dummies(self, series: pandas.Series, colname: str) -> pandas.DataFrame:
        """
        Convert a raw string column into prefixed one-hot dummy variables.

        Enforces the invariant: a row whose input value was NaN produces a
        row of all zeros in the output, deterministically across all pandas
        versions.

        Parameters
        ----------
        series : pd.Series
            Raw string column from the input DataFrame; may contain NaN.
        colname : str
            Name of the source column, used to look up the abbreviation
            prefix stored in ``self.dummy_prefix_`` and to build output
            column names (e.g. ``"tags"`` → prefix ``"ta"`` → ``"ta_a"``).

        Returns
        -------
        dummies : pd.DataFrame
            One-hot encoded DataFrame.  Column names are prefixed with
            ``<abbrev><col_name_sep><token>`` (e.g. ``"ta_a"``).
            Shape: ``(len(series), n_unique_tokens)``.  Rows where the
            original value was NaN are guaranteed to be all-zero.

        Notes
        -----
        Developer note
            This method is the **invariant owner** for NaN handling.  The
            two-step design is intentional:

            1. ``_split_and_clean`` is a pure normaliser — it preserves NaN
               as NaN and never decides what missing means semantically.

            2. This method captures ``nan_mask`` *before* normalisation,
               calls ``str.get_dummies`` on the cleaned series, then
               unconditionally zeros the NaN rows.  That explicit zero-out
               is the load-bearing line: it makes the contract hold
               regardless of how any pandas version internally handles NaN
               or empty strings in ``str.get_dummies``.

            A defensive ``drop(columns=[""])`` follows dummification.  In
            some pandas builds a non-NaN row that normalises to ``""`` (e.g.
            input was ``","`` — only separator, no tokens) can produce a
            spurious ``""`` column.  Dropping it keeps the column set clean.

        Raises
        ------
        KeyError
            If ``colname`` is not present in ``self.dummy_prefix_``.  This
            indicates ``fit`` was not called before ``transform``.
        """
        # ── Step 1: capture NaN positions before any mutation ────────────
        # This is the authoritative source of truth for missing rows.
        nan_mask = series.isna()

        # ── Step 2: normalise tokens; NaN rows remain NaN (no fillna) ────
        series = self._split_and_clean(series)

        # ── Step 3: expand into one-hot dummy columns ─────────────────────
        # str.get_dummies(NaN) → all-zeros in pandas 2.x; behaviour is
        # version-sensitive in pandas 3.x.  We enforce the invariant below
        # so behaviour is deterministic across all supported versions.
        dummies = series.str.get_dummies(sep=self.sep)

        # ── Step 4: drop spurious empty-token column ──────────────────────
        # Appears in certain pandas builds when a valid (non-NaN) row
        # normalises to "" (e.g. input "," → all tokens stripped → "").
        # Dropping it prevents an unwanted column from polluting the schema.
        if "" in dummies.columns:
            dummies = dummies.drop(columns=[""])

        # ── Step 5: enforce NaN-row → zero-row invariant ─────────────────
        # Unconditional: does not depend on any pandas NA behaviour.
        if nan_mask.any():
            dummies.loc[nan_mask] = 0

        # ── Step 6: prefix column names to avoid cross-column collisions ──
        dummies = dummies.add_prefix(self.dummy_prefix_[colname] + self.col_name_sep)

        # ── Step 7: optionally drop first dummy (collinearity reduction) ──
        if self.drop in ["first", True] and len(dummies.columns) > 1:
            dummies = dummies.iloc[:, 1:]

        # ── Step 8: cast to requested output dtype ────────────────────────
        if self.dtype is not None:
            dummies = dummies.astype(self.dtype)

        return dummies

    # -----------------------
    # Core API
    # -----------------------
    def fit(self, X, y=None):
        """
        Learn dummy categories from training data.

        Stores column order, prefixes, and categories for later alignment.
        """
        # Ensure DataFrame
        X = self._to_dataframe(X)
        # Store number of input features (sklearn convention)
        self.n_features_in_ = X.shape[1]

        # Normalize columns at fit-time only — NOT in __init__ — to honour sklearn's
        # clone contract: __init__ must store params verbatim so that get_params()
        # returns the original value and clone() can reconstruct the estimator.
        # A str shorthand is expanded to a single-element list here only.
        if isinstance(self.columns, str):
            cols_requested = [self.columns]
        else:
            cols_requested = self.columns  # list or None

        # Determine columns to encode
        if cols_requested is not None:
            # User-specified columns, Only keep those that exist in DataFrame
            self.dummy_cols_ = [col for col in cols_requested if col in X.columns]
        else:
            # Automatically detect object/string columns containing the separator, e.g., "a,b,c".
            # Pandas 3 changed select_dtypes: "O"/"object" no longer implicitly
            # includes the new StringDtype. Use both to support pandas 2 and 3.
            object_cols = X.select_dtypes(
                include=["object", "string"]
            ).columns  # Identify object/string columns
            # Keep only those that contain multiple values (separator present)
            self.dummy_cols_ = [
                col
                for col in object_cols
                if X[col].str.contains(self.sep, regex=False).any()
            ]

        # Define prefixes for dummy column names
        self.dummy_prefix_ = {
            col: (
                col[:2]
                if self.col_name_sep not in col  # default prefix first 2 letters
                else "".join(
                    part[0] for part in col.split(self.col_name_sep)
                )  # e.g., "tag_name" -> "tn"
            )
            for col in self.dummy_cols_
        }
        # Detect prefix abbreviation collisions (e.g. "ta" and "tags" both → "ta").
        # The original check compared len(set(keys)) vs len(set(values)) which is
        # always equal (dict keys are inherently unique).  The correct check is:
        # if any two columns share the same abbreviated prefix, fall back to full names.
        prefix_values = list(self.dummy_prefix_.values())
        if len(set(prefix_values)) != len(prefix_values):
            # Collision detected — use full column names as prefixes for safety
            self.dummy_prefix_ = {col: col for col in self.dummy_cols_}

        # Learn/Store categories seen during fit for each dummy column
        self.categories_ = {}
        for col in self.dummy_cols_:
            dummies = self._make_dummies(X[col], col)
            self.categories_[col] = sorted(dummies.columns.tolist())

        # Final build global column order: (non-dummy + all dummy) columns
        self.columns_ = X.drop(columns=self.dummy_cols_).columns.tolist() + [
            c for cats in self.categories_.values() for c in cats
        ]
        return self

    def transform(self, X, y=None):
        """
        Transform new data into dummy-expanded format.

        Steps:
        - Align columns with fit.
        - Drop unknown categories or raise error.
        - Return dense/pandas or sparse output.
        """
        import pandas as pd  # noqa: PLC0415
        import scipy.sparse as sp  # SciPy sparse CSR/CSC matrices  # noqa: PLC0415

        # Ensure fit has been called
        check_is_fitted(self, "columns_")
        # Convert to DataFrame if needed
        X = self._to_dataframe(X)

        dummies_list = []
        for col in self.dummy_cols_:
            # Create dummy columns for this feature
            dummies = self._make_dummies(X[col], col)

            # Detect/Handle unseen categories
            unseen = set(dummies.columns) - set(self.categories_[col])
            if unseen:
                if self.handle_unknown == "error":
                    raise ValueError(
                        f"Found unknown categories {unseen} in column '{col}' not seen during fit."
                    )
                # Drop unseen if ignoring
                dummies = dummies.drop(columns=list(unseen), errors="ignore")

            # Align with fitted columns: ensure same order and missing columns filled with 0
            dummies = dummies.reindex(columns=self.categories_[col], fill_value=0)
            dummies_list.append(dummies)

        # Combine non-dummy + all dummy columns
        X_out = pd.concat([X.drop(columns=self.dummy_cols_), *dummies_list], axis=1)
        # Reindex to preserve global column order
        X_out = X_out.reindex(columns=self.columns_, fill_value=0)

        # Return SciPy sparse CSR matrix if requested.
        # Only dummy columns are numeric; non-dummy passthrough columns may contain
        # strings or objects.  Cast only the dummy sub-block so we never attempt
        # float-casting string columns (which raises ValueError on mixed DataFrames).
        if self.sparse_output:
            dummy_cols = [c for cats in self.categories_.values() for c in cats]
            numeric_block = X_out[dummy_cols].to_numpy(dtype=self.dtype)
            return sp.csr_matrix(numeric_block)

        # Default: return pandas DataFrame.
        # NOTE: The `set_output` API (pandas / numpy wrapping) is handled automatically
        # by TransformerMixin's decorated `transform` wrapper — no manual dispatch needed
        # here.  The previously present `hasattr(self, "_get_output_config")` branch was
        # dead code: `_get_output_config` is a module-level function, not an instance
        # method, so `hasattr(self, ...)` was always False.
        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        check_is_fitted(self, "columns_")
        return np.array(self.columns_)


class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.
    """

    def _check_X(self, X, ensure_all_finite=True):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, e.g. for the `categories_` attribute.

        """
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            # if not a dataframe, do normal check_array validation
            X_temp = check_array(X, dtype=None, ensure_all_finite=ensure_all_finite)
            if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, ensure_all_finite=ensure_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            # pandas dataframe, do validation later column by column, in order
            # to keep the dtype information to be used in the encoder.
            needs_validation = ensure_all_finite

        n_samples, n_features = X.shape
        X_columns = []

        for i in range(n_features):
            Xi = _safe_indexing(X, indices=i, axis=1)
            Xi = check_array(
                Xi, ensure_2d=False, dtype=None, ensure_all_finite=needs_validation
            )
            X_columns.append(Xi)

        return X_columns, n_samples, n_features

    def _fit(
        self,
        X,
        handle_unknown="error",
        ensure_all_finite=True,
        return_counts=False,
        return_and_ignore_missing_for_infrequent=False,
    ):
        self._check_infrequent_enabled()
        validate_data(self, X=X, reset=True, skip_check_array=True)
        X_list, n_samples, n_features = self._check_X(
            X, ensure_all_finite=ensure_all_finite
        )
        self.n_features_in_ = n_features

        if self.categories != "auto":
            if len(self.categories) != n_features:
                raise ValueError(
                    "Shape mismatch: if categories is an array,"
                    " it has to be of shape (n_features,)."
                )

        self.categories_ = []
        category_counts = []
        compute_counts = return_counts or self._infrequent_enabled

        for i in range(n_features):
            Xi = X_list[i]

            if self.categories == "auto":
                result = _unique(Xi, return_counts=compute_counts)
                if compute_counts:
                    cats, counts = result
                    category_counts.append(counts)
                else:
                    cats = result
            else:
                if np.issubdtype(Xi.dtype, np.str_):
                    # Always convert string categories to objects to avoid
                    # unexpected string truncation for longer category labels
                    # passed in the constructor.
                    Xi_dtype = object
                else:
                    Xi_dtype = Xi.dtype

                cats = np.array(self.categories[i], dtype=Xi_dtype)
                if (
                    cats.dtype == object
                    and isinstance(cats[0], bytes)
                    and Xi.dtype.kind != "S"
                ):
                    msg = (
                        f"In column {i}, the predefined categories have type 'bytes'"
                        " which is incompatible with values of type"
                        f" '{type(Xi[0]).__name__}'."
                    )
                    raise ValueError(msg)

                # `nan` must be the last stated category
                for category in cats[:-1]:
                    if is_scalar_nan(category):
                        raise ValueError(
                            "Nan should be the last element in user"
                            f" provided categories, see categories {cats}"
                            f" in column #{i}"
                        )

                if cats.size != len(_unique(cats)):
                    msg = (
                        f"In column {i}, the predefined categories"
                        " contain duplicate elements."
                    )
                    raise ValueError(msg)

                if Xi.dtype.kind not in "OUS":
                    sorted_cats = np.sort(cats)
                    error_msg = (
                        "Unsorted categories are not supported for numerical categories"
                    )
                    # if there are nans, nan should be the last element
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]):
                        raise ValueError(error_msg)

                if handle_unknown == "error":
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)
                if compute_counts:
                    category_counts.append(_get_counts(Xi, cats))

            self.categories_.append(cats)

        output = {"n_samples": n_samples}
        if return_counts:
            output["category_counts"] = category_counts

        missing_indices = {}
        if return_and_ignore_missing_for_infrequent:
            for feature_idx, categories_for_idx in enumerate(self.categories_):
                if is_scalar_nan(categories_for_idx[-1]):
                    # `nan` values can only be placed in the latest position
                    missing_indices[feature_idx] = categories_for_idx.size - 1
            output["missing_indices"] = missing_indices

        if self._infrequent_enabled:
            self._fit_infrequent_category_mapping(
                n_samples,
                category_counts,
                missing_indices,
            )
        return output

    def _transform(
        self,
        X,
        handle_unknown="error",
        ensure_all_finite=True,
        warn_on_unknown=False,
        ignore_category_indices=None,
    ):
        X_list, n_samples, n_features = self._check_X(
            X, ensure_all_finite=ensure_all_finite
        )
        validate_data(self, X=X, reset=False, skip_check_array=True)

        X_int = np.zeros((n_samples, n_features), dtype=int)
        X_mask = np.ones((n_samples, n_features), dtype=bool)

        columns_with_unknown = []
        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)

            if not np.all(valid_mask):
                if handle_unknown == "error":
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    if warn_on_unknown:
                        columns_with_unknown.append(i)
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    # cast Xi into the largest string type necessary
                    # to handle different lengths of numpy strings
                    if (
                        self.categories_[i].dtype.kind in ("U", "S")
                        and self.categories_[i].itemsize > Xi.itemsize
                    ):
                        Xi = Xi.astype(self.categories_[i].dtype)
                    elif self.categories_[i].dtype.kind == "O" and Xi.dtype.kind == "U":
                        # categories are objects and Xi are numpy strings.
                        # Cast Xi to an object dtype to prevent truncation
                        # when setting invalid values.
                        Xi = Xi.astype("O")
                    else:
                        Xi = Xi.copy()

                    Xi[~valid_mask] = self.categories_[i][0]
            # We use check_unknown=False, since _check_unknown was
            # already called above.
            X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)
        if columns_with_unknown:
            warnings.warn(
                (
                    "Found unknown categories in columns "
                    f"{columns_with_unknown} during transform. These "
                    "unknown categories will be encoded as all zeros"
                ),
                UserWarning,
                stacklevel=2,
            )

        self._map_infrequent_categories(X_int, X_mask, ignore_category_indices)
        return X_int, X_mask

    @property
    def infrequent_categories_(self):
        """Infrequent categories for each feature."""
        # raises an AttributeError if `_infrequent_indices` is not defined
        infrequent_indices = self._infrequent_indices
        # categories_ may be a dict (DummyCodeEncoder) or list (_BaseEncoder subclasses).
        # Always iterate values so both forms work correctly.
        cats_iter = (
            self.categories_.values()
            if isinstance(self.categories_, dict)
            else self.categories_
        )
        return [
            None if indices is None else category[indices]
            for category, indices in zip(cats_iter, infrequent_indices)
        ]

    def _check_infrequent_enabled(self):
        """
        This functions checks whether _infrequent_enabled is True or False.
        This has to be called after parameter validation in the fit function.
        """
        max_categories = getattr(self, "max_categories", None)
        min_frequency = getattr(self, "min_frequency", None)
        self._infrequent_enabled = (
            max_categories is not None and max_categories >= 1
        ) or min_frequency is not None

    def _identify_infrequent(self, category_count, n_samples, col_idx):
        """Compute the infrequent indices.

        Parameters
        ----------
        category_count : ndarray of shape (n_cardinality,)
            Category counts.
        n_samples : int
            Number of samples.
        col_idx : int
            Index of the current category. Only used for the error message.

        Returns
        -------
        output : ndarray of shape (n_infrequent_categories,) or None
            If there are infrequent categories, indices of infrequent
            categories. Otherwise None.
        """
        if isinstance(self.min_frequency, numbers.Integral):
            infrequent_mask = category_count < self.min_frequency
        elif isinstance(self.min_frequency, numbers.Real):
            min_frequency_abs = n_samples * self.min_frequency
            infrequent_mask = category_count < min_frequency_abs
        else:
            infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)

        n_current_features = category_count.size - infrequent_mask.sum() + 1
        if self.max_categories is not None and self.max_categories < n_current_features:
            # max_categories includes the one infrequent category
            frequent_category_count = self.max_categories - 1
            if frequent_category_count == 0:
                # All categories are infrequent
                infrequent_mask[:] = True
            else:
                # stable sort to preserve original count order
                smallest_levels = np.argsort(category_count, kind="mergesort")[
                    :-frequent_category_count
                ]
                infrequent_mask[smallest_levels] = True

        output = np.flatnonzero(infrequent_mask)
        return output if output.size > 0 else None

    def _fit_infrequent_category_mapping(
        self, n_samples, category_counts, missing_indices
    ):
        """Fit infrequent categories.

        Defines the private attribute: `_default_to_infrequent_mappings`. For
        feature `i`, `_default_to_infrequent_mappings[i]` defines the mapping
        from the integer encoding returned by `super().transform()` into
        infrequent categories. If `_default_to_infrequent_mappings[i]` is None,
        there were no infrequent categories in the training set.

        For example if categories 0, 2 and 4 were frequent, while categories
        1, 3, 5 were infrequent for feature 7, then these categories are mapped
        to a single output:
        `_default_to_infrequent_mappings[7] = array([0, 3, 1, 3, 2, 3])`

        Defines private attribute: `_infrequent_indices`. `_infrequent_indices[i]`
        is an array of indices such that
        `categories_[i][_infrequent_indices[i]]` are all the infrequent category
        labels. If the feature `i` has no infrequent categories
        `_infrequent_indices[i]` is None.

        .. versionadded:: 1.1

        Parameters
        ----------
        n_samples : int
            Number of samples in training set.
        category_counts: list of ndarray
            `category_counts[i]` is the category counts corresponding to
            `self.categories_[i]`.
        missing_indices : dict
            Dict mapping from feature_idx to category index with a missing value.
        """
        # Remove missing value from counts, so it is not considered as infrequent
        if missing_indices:
            category_counts_ = []
            for feature_idx, count in enumerate(category_counts):
                if feature_idx in missing_indices:
                    category_counts_.append(
                        np.delete(count, missing_indices[feature_idx])
                    )
                else:
                    category_counts_.append(count)
        else:
            category_counts_ = category_counts

        self._infrequent_indices = [
            self._identify_infrequent(category_count, n_samples, col_idx)
            for col_idx, category_count in enumerate(category_counts_)
        ]

        # compute mapping from default mapping to infrequent mapping
        self._default_to_infrequent_mappings = []

        for feature_idx, infreq_idx in enumerate(self._infrequent_indices):
            cats = self.categories_[feature_idx]
            # no infrequent categories
            if infreq_idx is None:
                self._default_to_infrequent_mappings.append(None)
                continue

            n_cats = len(cats)
            if feature_idx in missing_indices:
                # Missing index was removed from this category when computing
                # infrequent indices, thus we need to decrease the number of
                # total categories when considering the infrequent mapping.
                n_cats -= 1

            # infrequent indices exist
            mapping = np.empty(n_cats, dtype=np.int64)
            n_infrequent_cats = infreq_idx.size

            # infrequent categories are mapped to the last element.
            n_frequent_cats = n_cats - n_infrequent_cats
            mapping[infreq_idx] = n_frequent_cats

            frequent_indices = np.setdiff1d(np.arange(n_cats), infreq_idx)
            mapping[frequent_indices] = np.arange(n_frequent_cats)

            self._default_to_infrequent_mappings.append(mapping)

    def _map_infrequent_categories(self, X_int, X_mask, ignore_category_indices):
        """Map infrequent categories to integer representing the infrequent category.

        This modifies X_int in-place. Values that were invalid based on `X_mask`
        are mapped to the infrequent category if there was an infrequent
        category for that feature.

        Parameters
        ----------
        X_int: ndarray of shape (n_samples, n_features)
            Integer encoded categories.
        X_mask: ndarray of shape (n_samples, n_features)
            Bool mask for valid values in `X_int`.
        ignore_category_indices : dict
            Dictionary mapping from feature_idx to category index to ignore.
            Ignored indexes will not be grouped and the original ordinal encoding
            will remain.
        """
        if not self._infrequent_enabled:
            return

        ignore_category_indices = ignore_category_indices or {}

        for col_idx in range(X_int.shape[1]):
            infrequent_idx = self._infrequent_indices[col_idx]
            if infrequent_idx is None:
                continue

            X_int[~X_mask[:, col_idx], col_idx] = infrequent_idx[0]
            if self.handle_unknown == "infrequent_if_exist":
                # All the unknown values are now mapped to the
                # infrequent_idx[0], which makes the unknown values valid
                # This is needed in `transform` when the encoding is formed
                # using `X_mask`.
                X_mask[:, col_idx] = True

        # Remaps encoding in `X_int` where the infrequent categories are
        # grouped together.
        for i, mapping in enumerate(self._default_to_infrequent_mappings):
            if mapping is None:
                continue

            if i in ignore_category_indices:
                # Update rows that are **not** ignored
                rows_to_update = X_int[:, i] != ignore_category_indices[i]
            else:
                rows_to_update = slice(None)

            X_int[rows_to_update, i] = np.take(mapping, X_int[rows_to_update, i])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = True
        return tags


class DummyCodeEncoder(_BaseEncoder):
    """
    Encode categorical features into dummy/indicator 0/1 variables.

    Each string in Series is split by ``sep`` and returned as a DataFrame
    of dummy/indicator 0/1 variables.

    Each variable is converted in as many 0/1 variables as there are different
    values. Columns in the output are each named after a value; if the input is
    a DataFrame, the name of the original variable is prepended to the value.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse_output``
    parameter).

    By default, the encoder derives the categories based on the unique values
    in each feature that contain multiple string labels separated by ``sep``
    into one-hot encoded columns by :func:`pandas.get_dummies`.
    Alternatively, you can also specify the `categories` manually.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Compatible with sklearn pipelines, `set_output` API, and supports both
    dense and sparse output.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    For a comparison of different encoders, refer to:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder.py`.

    .. caution::
        ⚠️ These parameters are reserved for future use;
        some have no impact on the current implementation,
        and their behavior or presence may change in future versions
        without notice.

    Parameters
    ----------
    columns : list-like, default=None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object`, `string`, or `category` dtype will be converted.
    sep : callable or str, default='|'
        String regex or literal separator to split on (e.g., "a,b,c").

        - sep=',',
        - sep=r'\\s*[,;|]\\s*',
        - sep=lambda s: re.split(r'\\s*[,;|]\\s*', s.lower()),
    regex : bool, default=True
        Use regex to split on (e.g., "a,b|C;") by ``sep`` like:

        - ``pattern=r'\\s*[,;|]\\s*'``
    prefix : str, list of str, or dict of str, default=None
        String to append DataFrame column names.
        Pass a list with length equal to the number of columns
        when calling get_dummies on a DataFrame. Alternatively, `prefix`
        can be a dictionary mapping column names to prefixes.
    prefix_sep : str, default='_'
        If appending prefix, separator/delimiter to use. Or pass a
        list or dictionary as with `prefix` (e.g., "tags_a").
    dummy_na : bool, default=False
        Add a column to indicate NaNs, if False NaNs are ignored.

        .. caution::
            If enabled to encode multi-feature supports only one contains ``None``.
            Due to total categories need to unique so suggested dummy
            fill instead of keeping one of (e.g., None, np.nan, pd.Na, pd.NAT).
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.
    drop : {'first', 'if_binary'} or an array-like of shape (n_features,), \
            default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into an unregularized linear regression model.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - 'if_binary' : drop the first category in each feature with two
          categories. Features with 1 or more than 2 categories are
          left intact.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped.

        When `max_categories` or `min_frequency` is configured to group
        infrequent categories, the dropping behavior is handled after the
        grouping.
    sparse_output : bool, default=True
        When ``True``, it returns a :class:`scipy.sparse.csr_matrix`,
        i.e. a sparse matrix in "Compressed Sparse Row" (CSR) format.
    dtype : number type, default=np.float64
        Desired dtype of output.
    handle_unknown : {'error', 'ignore', 'infrequent_if_exist', 'warn'}, \
                     default='error'
        Specifies the way unknown categories are handled during :meth:`transform`.

        - 'error' : Raise an error if an unknown category is present during transform.
        - 'ignore' : When an unknown category is encountered during
          transform, the resulting one-hot encoded columns for this feature
          will be all zeros. In the inverse transform, an unknown category
          will be denoted as None.
        - 'infrequent_if_exist' : When an unknown category is encountered
          during transform, the resulting one-hot encoded columns for this
          feature will map to the infrequent category if it exists. The
          infrequent category will be mapped to the last position in the
          encoding. During inverse transform, an unknown category will be
          mapped to the category denoted `'infrequent'` if it exists. If the
          `'infrequent'` category does not exist, then :meth:`transform` and
          :meth:`inverse_transform` will handle an unknown category as with
          `handle_unknown='ignore'`. Infrequent categories exist based on
          `min_frequency` and `max_categories`. Read more in the
          :ref:`User Guide <encoder_infrequent_categories>`.
        - 'warn' : When an unknown category is encountered during transform
          a warning is issued, and the encoding then proceeds as described for
          `handle_unknown="infrequent_if_exist"`.
    min_frequency : int or float, default=None
        Specifies the minimum frequency below which a category will be
        considered infrequent.

        - If `int`, categories with a smaller cardinality will be considered
          infrequent.

        - If `float`, categories with a smaller cardinality than
          `min_frequency * n_samples`  will be considered infrequent.

        .. versionadded:: 1.1
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.
    max_categories : int, default=None
        Specifies an upper limit to the number of output features for each input
        feature when considering infrequent categories. If there are infrequent
        categories, `max_categories` includes the category representing the
        infrequent categories along with the frequent categories. If `None`,
        there is no limit to the number of output features.

        .. versionadded:: 1.1
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.
    feature_name_combiner : "concat" or callable, default="concat"
        Callable with signature `def callable(input_feature, category)` that returns a
        string. This is used to create feature names to be returned by
        :meth:`get_feature_names_out`.

        `"concat"` concatenates encoded feature name and category with
        `feature + "_" + str(category)`.E.g. feature X with values 1, 6, 7 create
        feature names `X_1, X_6, X_7`.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).
    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
          to be dropped for each feature.
        - ``drop_idx_[i] = None`` if no category is to be dropped from the
          feature with index ``i``, e.g. when `drop='if_binary'` and the
          feature isn't binary.
        - ``drop_idx_ = None`` if all the transformed features will be
          retained.

        If infrequent categories are enabled by setting `min_frequency` or
        `max_categories` to a non-default value and `drop_idx[i]` corresponds
        to an infrequent category, then the entire infrequent category is
        dropped.

        .. versionchanged:: 0.23
           Added the possibility to contain `None` values.
    infrequent_categories_ : list of ndarray
        Defined only if infrequent categories are enabled by setting
        `min_frequency` or `max_categories` to a non-default value.
        `infrequent_categories_[i]` are the infrequent categories for feature
        `i`. If the feature `i` has no infrequent categories
        `infrequent_categories_[i]` is None.

        .. versionadded:: 1.1
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0
    feature_name_combiner : callable or None
        Callable with signature `def callable(input_feature, category)` that returns a
        string. This is used to create feature names to be returned by
        :meth:`get_feature_names_out`.

        .. versionadded:: 1.3

    See Also
    --------
    GetDummies : Same but more limited and pandas based convert to dummy codes.
    pandas.Series.str.get_dummies : Convert Series of strings to dummy codes.
    pandas.from_dummies : Convert dummy codes back to categorical DataFrame.
    sklearn.preprocessing.OrdinalEncoder : Performs an ordinal (integer)
      encoding of the categorical features.
    sklearn.preprocessing.TargetEncoder : Encodes categorical features using the target.
    sklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : Performs an approximate one-hot
      encoding of dictionary items or strings.
    sklearn.preprocessing.LabelBinarizer : Binarizes labels in a one-vs-all
      fashion.
    sklearn.preprocessing.MultiLabelBinarizer : Transforms between iterable of
      iterables and a multilabel format, e.g. a (samples x classes) binary
      matrix indicating the presence of a class label.

    References
    ----------
    .. [1] `Çelik, M. (2023, December 9).
       "How to converting pandas column of comma-separated strings into dummy variables?."
       Medium. https://medium.com/@celik-muhammed/how-to-converting-pandas-column-of-comma-separated-strings-into-dummy-variables-762c02282a6c
       <https://medium.com/@celik-muhammed/how-to-converting-pandas-column-of-comma-separated-strings-into-dummy-variables-762c02282a6c>`_

    Examples
    --------
    Given a dataset with three features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot dummy encoding.

    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "tags": ["a,b,", " A , b", "a,B,C", None],
    ...         "color": ["red", "blue", "green", "Red"],
    ...         "value": [1, 2, 3, 4],
    ...     }
    ... )

    >>> from sklearn.pipeline import Pipeline
    >>> from scikitplot.preprocessing import DummyCodeEncoder
    >>> pipe = Pipeline(
    ...     [
    ...         (
    ...             "encoder",
    ...             DummyCodeEncoder(
    ...                 # sep=',',
    ...                 # sep=r'\\s*[,;|]\\s*',
    ...                 sep=lambda s: re.split(r'\\s*[,;|]\\s*', s.lower()),
    ...                 regex=True,
    ...                 sparse_output=True,
    ...             ),
    ...         )
    ...     ]
    ... )
    >>> X_trans = pipe.fit_transform(df)
    >>> print(X_trans)
    <Compressed Sparse Row sparse matrix of dtype 'float64'
        with 15 stored elements and shape (4, 10)>
    >>> type(X_trans)
    scipy.sparse._csr.csr_matrix

    >>> from sklearn.pipeline import Pipeline
    >>> from scikitplot.preprocessing import DummyCodeEncoder
    >>> pipe = Pipeline(
    ...     [
    ...         (
    ...             "encoder",
    ...             DummyCodeEncoder(
    ...                 # sep=',',
    ...                 # sep=r'\\s*[,;|]\\s*',
    ...                 sep=lambda s: re.split(r'\\s*[,;|]\\s*', s.lower()),
    ...                 regex=True,
    ...                 sparse_output=False,
    ...             ),
    ...         )
    ...     ]
    ... ).set_output(transform='pandas')
    >>> X_trans = pipe.fit_transform(df)
    >>> print(X_trans)
       value  tags_a  tags_b  tags_c  color_blue  color_green  color_red  value_1  value_2  value_3  value_4
    0      1   1.0     1.0     0.0      0.0         0.0          1.0        1.0	     0.0	  0.0	   0.0
    1      2   1.0     1.0     0.0      1.0         0.0          0.0        0.0	     1.0	  0.0	   0.0
    2      3   1.0     1.0     1.0      0.0         1.0          0.0        0.0	     0.0	  1.0	   0.0
    3      4   0.0     0.0     0.0      0.0         0.0          1.0        0.0	     0.0	  0.0	   1.0
    >>> type(X_trans)
    pandas.core.frame.DataFrame
    """

    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_repr_html/base.py#L11
    _doc_link_module = "scikitplot"
    _doc_link_template = (
        f"https://scikit-plots.github.io/{version_url}/modules/generated/"
        "{estimator_module}.{estimator_name}.html"
    )

    _parameter_constraints: dict = {  # noqa: RUF012
        "columns": "no_validation",  # any object is valid
        "sep": [callable, str],
        "regex": ["boolean"],
        "prefix": "no_validation",  # any object is valid
        "prefix_sep": "no_validation",  # any object is valid
        "dummy_na": ["boolean"],
        "categories": [StrOptions({"auto"}), list],
        "drop": [StrOptions({"first", "if_binary"}), "array-like", None],
        "dtype": "no_validation",  # validation delegated to numpy
        "handle_unknown": [
            StrOptions({"error", "ignore", "infrequent_if_exist", "warn"})
        ],
        "max_categories": [Interval(Integral, 1, None, closed="left"), None],
        "min_frequency": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="neither"),
            None,
        ],
        "sparse_output": ["boolean"],
        "feature_name_combiner": [StrOptions({"concat"}), callable],
    }

    def __init__(
        self,
        *,
        # get_dummies, str.get_dummies
        columns=None,
        sep: str = "|",
        regex=False,
        prefix=None,
        prefix_sep: "str | Iterable[str] | dict[str, str]" = "_",
        dummy_na: bool = False,
        # OneHotEncoder
        categories="auto",
        drop=None,
        sparse_output=True,
        dtype=np.float64,
        handle_unknown="error",
        min_frequency=None,
        max_categories=None,
        feature_name_combiner="concat",
    ):
        self.columns = columns
        self.sep = sep
        self.regex = regex
        self.prefix = prefix
        self.prefix_sep = prefix_sep
        self.dummy_na = dummy_na
        self.categories = categories
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner

    def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
        """Convert `drop_idx` into the index for infrequent categories.

        If there are no infrequent categories, then `drop_idx` is
        returned. This method is called in `_set_drop_idx` when the `drop`
        parameter is an array-like.
        """
        if not self._infrequent_enabled:
            return drop_idx

        default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
        if default_to_infrequent is None:
            return drop_idx

        # Raise error when explicitly dropping a category that is infrequent
        infrequent_indices = self._infrequent_indices[feature_idx]
        if infrequent_indices is not None and drop_idx in infrequent_indices:
            categories = self.categories_[feature_idx]
            raise ValueError(
                f"Unable to drop category {categories[drop_idx].item()!r} from"
                f" feature {feature_idx} because it is infrequent"
            )
        return default_to_infrequent[drop_idx]

    def _set_drop_idx(self):
        """Compute the drop indices associated with `self.categories_`.

        If `self.drop` is:
        - `None`, No categories have been dropped.
        - `'first'`, All zeros to drop the first category.
        - `'if_binary'`, All zeros if the category is binary and `None`
          otherwise.
        - array-like, The indices of the categories that match the
          categories in `self.drop`. If the dropped category is an infrequent
          category, then the index for the infrequent category is used. This
          means that the entire infrequent category is dropped.

        This methods defines a public `drop_idx_` and a private
        `_drop_idx_after_grouping`.

        - `drop_idx_`: Public facing API that references the drop category in
          `self.categories_`.
        - `_drop_idx_after_grouping`: Used internally to drop categories *after* the
          infrequent categories are grouped together.

        If there are no infrequent categories or drop is `None`, then
        `drop_idx_=_drop_idx_after_grouping`.
        """
        if self.drop is None:
            drop_idx_after_grouping = None
        elif isinstance(self.drop, str):
            if self.drop == "first":
                drop_idx_after_grouping = np.zeros(len(self.categories_), dtype=object)
            elif self.drop == "if_binary":
                n_features_out_no_drop = [len(cat) for cat in self.categories_.values()]
                if self._infrequent_enabled:
                    for i, infreq_idx in enumerate(self._infrequent_indices):
                        if infreq_idx is None:
                            continue
                        n_features_out_no_drop[i] -= infreq_idx.size - 1

                drop_idx_after_grouping = np.array(
                    [
                        0 if n_features_out == 2 else None
                        for n_features_out in n_features_out_no_drop
                    ],
                    dtype=object,
                )

        else:
            drop_array = np.asarray(self.drop, dtype=object)
            droplen = len(drop_array)

            if droplen != len(self.categories_):
                msg = (
                    "`drop` should have length equal to the number "
                    "of features ({}), got {}"
                )
                raise ValueError(msg.format(len(self.categories_), droplen))
            missing_drops = []
            drop_indices = []
            for feature_idx, (drop_val, cat_list) in enumerate(
                zip(drop_array, self.categories_.values())
            ):
                if not is_scalar_nan(drop_val):
                    drop_idx = np.where(np.asarray(cat_list) == drop_val)[0]
                    if drop_idx.size:  # found drop idx
                        drop_indices.append(
                            self._map_drop_idx_to_infrequent(feature_idx, drop_idx[0])
                        )
                    else:
                        missing_drops.append((feature_idx, drop_val))
                    continue

                # drop_val is nan, find nan in categories manually
                if is_scalar_nan(cat_list[-1]):
                    drop_indices.append(
                        self._map_drop_idx_to_infrequent(feature_idx, cat_list.size - 1)
                    )
                else:  # nan is missing
                    missing_drops.append((feature_idx, drop_val))

            if any(missing_drops):
                msg = (
                    "The following categories were supposed to be "
                    "dropped, but were not found in the training "
                    "data.\n{}".format(
                        "\n".join(
                            [
                                "Category: {}, Feature: {}".format(c, v)
                                for c, v in missing_drops
                            ]
                        )
                    )
                )
                raise ValueError(msg)
            drop_idx_after_grouping = np.array(drop_indices, dtype=object)

        # `_drop_idx_after_grouping` are the categories to drop *after* the infrequent
        # categories are grouped together. If needed, we remap `drop_idx` back
        # to the categories seen in `self.categories_`.
        self._drop_idx_after_grouping = drop_idx_after_grouping

        if not self._infrequent_enabled or drop_idx_after_grouping is None:
            self.drop_idx_ = self._drop_idx_after_grouping
        else:
            drop_idx_ = []
            for feature_idx, drop_idx in enumerate(drop_idx_after_grouping):
                default_to_infrequent = self._default_to_infrequent_mappings[
                    feature_idx
                ]
                if drop_idx is None or default_to_infrequent is None:
                    orig_drop_idx = drop_idx
                else:
                    orig_drop_idx = np.flatnonzero(default_to_infrequent == drop_idx)[0]

                drop_idx_.append(orig_drop_idx)

            self.drop_idx_ = np.asarray(drop_idx_, dtype=object)

    def _compute_transformed_categories(self, i, remove_dropped=True):
        """Compute the transformed categories used for column `i`.

        1. If there are infrequent categories, the category is named
        'infrequent_sklearn'.
        2. Dropped columns are removed when remove_dropped=True.
        """
        cats = self.categories_[i]

        if self._infrequent_enabled:
            infreq_map = self._default_to_infrequent_mappings[i]
            if infreq_map is not None:
                frequent_mask = infreq_map < infreq_map.max()
                infrequent_cat = "infrequent_sklearn"
                # infrequent category is always at the end
                cats = np.concatenate(
                    (cats[frequent_mask], np.array([infrequent_cat], dtype=object))
                )

        if remove_dropped:
            cats = self._remove_dropped_categories(cats, i)
        return cats

    def _remove_dropped_categories(self, categories, i):
        """Remove dropped categories."""
        if (
            self._drop_idx_after_grouping is not None
            and self._drop_idx_after_grouping[i] is not None
        ):
            return np.delete(categories, self._drop_idx_after_grouping[i])
        return categories

    def _compute_n_features_outs(self):
        """Compute the n_features_out for each input feature."""
        output = [len(cats) for cats in self.categories_.values()]

        if self._drop_idx_after_grouping is not None:
            for i, drop_idx in enumerate(self._drop_idx_after_grouping):
                if drop_idx is not None:
                    output[i] -= 1

        if not self._infrequent_enabled:
            return output

        # infrequent is enabled, the number of features out are reduced
        # because the infrequent categories are grouped together
        for i, infreq_idx in enumerate(self._infrequent_indices):
            if infreq_idx is None:
                continue
            output[i] -= infreq_idx.size - 1

        return output

    def _expand_by_separators(self, data, sep=None, regex=None):
        """
        Expand each string element by given separators.
        Keeps None, np.nan, np.inf, -np.inf intact.
        Returns a NumPy array with same dtype if compatible, else dtype=object.

        Parameters
        ----------
        sep: str or callable
            Expand a list of parts:
            - If callable: must return list of substrings
            - If str and regex=True: treated as regex pattern
            - If str and regex=False: literal split
        regex : bool
            If True, treat `sep` as a regex pattern.
        """
        # Determine input dtype
        dtype = getattr(data, "dtype", object)
        sep = sep or getattr(self, "sep", "|")
        regex = regex if regex is not None else getattr(self, "regex", False)
        # Prepare regex pattern if needed
        # if regex:
        #     pattern = f"[{re.escape(sep)}]"  # e.g., sep='|,;' -> pattern='[\\|,;]'
        # else:
        #     pattern = sep  # treat as literal string

        # Ensure iterable
        if isinstance(data, np.ndarray):
            iterable = data.tolist()
        elif not isinstance(data, list):
            iterable = np.asarray(data).tolist()
        else:
            iterable = data

        result = []
        for idx, item in enumerate(iterable):
            # --- Handle missing or special float values safely ---
            if item is None or (
                isinstance(item, float) and (np.isnan(item) or not np.isfinite(item))
            ):
                result.append(item)
                continue
            # --- Handle non-string types (numbers, etc.) ---
            if not isinstance(item, str):
                # May 'NAType', 'NaTType'
                result.append(item)
                continue

            # --- Split string using user-defined logic ---
            if callable(sep):
                parts = sep(item)
            # --- Split using regex or literal separator ---
            elif regex:
                # treat each char in sep as regex alternative, e.g., "|,;" -> splits on any
                # pattern = f"[{re.escape(sep)}]"
                # pattern=r'\s*[,;|]\s*'
                parts = re.split(sep, item)
            else:
                # treat as literal string sep='|'
                parts = item.split(sep)

            # --- Clean & append 2d or extend 1d ---
            clean_parts = [p.strip() for p in parts if p.strip()]
            result.extend(clean_parts)
            # (optional) Example of using index
            # print(f"[{idx}] expanded '{item}' -> {clean_parts}")

        # --- Preserve dtype if possible ---
        # list(dict.fromkeys(result))
        try:
            return np.array(result, dtype=dtype)
        except (TypeError, ValueError):
            return np.array(result, dtype=object)

    def _sort_with_none_nan_last(self, Xi):
        # Flatten and get unique values
        uniq = set(itertools.chain.from_iterable([Xi]))

        # Function to check None/NaN
        def is_none_or_nan(x):
            return x is None or (isinstance(x, float) and np.isnan(x))

        # Separate normal values and None/NaN
        normal_vals = [x for x in uniq if not is_none_or_nan(x)]
        none_nan_vals = [x for x in uniq if is_none_or_nan(x)]

        # Sort normal values safely
        try:
            normal_vals_sorted = sorted(normal_vals)
        except TypeError:
            # Mixed types → fallback to string comparison
            normal_vals_sorted = sorted(normal_vals, key=lambda x: str(x))

        # Put None and NaN at end
        return (
            normal_vals_sorted + none_nan_vals if self.dummy_na else normal_vals_sorted
        )

    def _fit(
        self,
        X,
        handle_unknown="error",
        ensure_all_finite=True,
        return_counts=False,
        return_and_ignore_missing_for_infrequent=False,
    ):
        self._check_infrequent_enabled()
        validate_data(self, X=X, reset=True, skip_check_array=True)
        X_list, n_samples, n_features = self._check_X(
            X, ensure_all_finite=ensure_all_finite
        )
        self.n_features_in_ = n_features

        self._cached_dict = None
        # input_features = _check_feature_names_in(self)
        input_features = range(n_features)
        self.categories_ = dict.fromkeys(input_features)

        for i in range(n_features):
            Xi = X_list[i]
            Xi = self._expand_by_separators(Xi)

            if self.categories == "auto":
                # Sort: normal values first, None/NaN at the end
                cats = self._sort_with_none_nan_last(Xi)
            elif not isinstance(self.categories, list):
                raise ValueError(
                    "categories must be 'auto' or a list of per-feature arrays."
                )
            else:
                if np.issubdtype(Xi.dtype, np.str_):
                    # Always convert string categories to objects to avoid
                    # unexpected string truncation for longer category labels
                    # passed in the constructor.
                    Xi_dtype = object
                else:
                    Xi_dtype = Xi.dtype

                cats = np.array(self.categories[i], dtype=Xi_dtype)
                if (
                    cats.dtype == object
                    and isinstance(cats[0], bytes)
                    and Xi.dtype.kind != "S"
                ):
                    msg = (
                        f"In column {i}, the predefined categories have type 'bytes'"
                        " which is incompatible with values of type"
                        f" '{type(Xi[0]).__name__}'."
                    )
                    raise ValueError(msg)

                # `nan` must be the last stated category
                for category in cats[:-1]:
                    if is_scalar_nan(category):
                        raise ValueError(
                            "Nan should be the last element in user"
                            f" provided categories, see categories {cats}"
                            f" in column #{i}"
                        )

                if cats.size != len(_unique(cats)):
                    msg = (
                        f"In column {i}, the predefined categories"
                        " contain duplicate elements."
                    )
                    raise ValueError(msg)

                if Xi.dtype.kind not in "OUS":
                    sorted_cats = np.sort(cats)
                    error_msg = (
                        "Unsorted categories are not supported for numerical categories"
                    )
                    # if there are nans, nan should be the last element
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]):
                        raise ValueError(error_msg)

                if handle_unknown == "error":
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)

            categories = np.empty(len(cats), dtype=Xi.dtype)
            categories[:] = cats
            self.categories_[input_features[i]] = categories

        output = {"n_samples": n_samples}

        # When infrequent-category grouping is requested (min_frequency /
        # max_categories), we must compute per-category counts and run the
        # infrequent mapping.  The base-class _fit() does this, but
        # DummyCodeEncoder._fit() is fully overridden and cannot reuse it
        # directly (different categories_ structure).  We compute the counts
        # here and delegate to the shared helper so _infrequent_indices and
        # _default_to_infrequent_mappings are always defined before fit()
        # calls _set_drop_idx() and _compute_n_features_outs().
        if self._infrequent_enabled:
            category_counts = []
            for i in range(n_features):
                Xi = X_list[i]
                Xi_expanded = self._expand_by_separators(Xi)
                counts = []
                for cat in self.categories_[i]:
                    counts.append(np.sum(Xi_expanded == cat))
                category_counts.append(np.array(counts, dtype=np.int64))
            self._fit_infrequent_category_mapping(
                n_samples, category_counts, missing_indices={}
            )
        else:
            # No infrequent grouping: initialise attributes to safe defaults so
            # _set_drop_idx() and _compute_n_features_outs() can always rely on them.
            self._infrequent_indices = [None] * n_features
            self._default_to_infrequent_mappings = [None] * n_features

        return output

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
            Fitted encoder.
        """
        self._fit(
            X,
            handle_unknown=self.handle_unknown,
            ensure_all_finite="allow-nan",
        )
        self._set_drop_idx()
        self._n_features_outs = self._compute_n_features_outs()
        return self

    def _build_cache(self):
        if self._cached_dict is None:
            categories_flat_ = [
                item for arr in self.categories_.values() for item in arr
            ]
            self._cached_dict = dict(
                zip(categories_flat_, range(len(categories_flat_)))
            )
        return self._cached_dict

    def _transform(
        self,
        X,
        handle_unknown="error",
        ensure_all_finite=True,
        warn_on_unknown=False,
        ignore_category_indices=None,
        class_to_index=None,
    ):
        X_list, n_samples, n_features = self._check_X(
            X, ensure_all_finite=ensure_all_finite
        )
        validate_data(self, X=X, reset=False, skip_check_array=True)

        class_mapping = (
            class_to_index if class_to_index is not None else self._build_cache()
        )
        X_int = np.zeros((n_samples, len(class_mapping)), dtype=int)
        X_mask = np.ones((n_samples, len(class_mapping)), dtype=bool)

        # FIX: convert col-based X_list (list-cols) to row-based (list-rows)
        # X_list = list(zip(*X_list))   # row-major view like X_list.T
        # Instead of building each row via zip(*X_list),
        # process column-by-column and push values into row sets.
        per_row_buckets = [set() for _ in range(n_samples)]

        # Build CSR matrix
        indices = array.array("i")
        indptr = array.array("i", [0])
        columns_with_unknown = set()
        for i in range(n_features):
            # index = set()
            X_col = X_list[i]  # length = n_samples
            # vals = self._expand_by_separators([X_col])
            # diff, valid_mask = _check_unknown(vals, self.categories_[i], return_mask=True)
            # if not np.all(valid_mask):
            #     if handle_unknown == "error":
            #         msg = (
            #             "Found unknown categories {0} in column {1}"
            #             " during transform".format(diff, i)
            #         )
            #         raise ValueError(msg)
            #     else:
            #         if warn_on_unknown:
            #             columns_with_unknown.add(i)
            #         # Set the problematic rows to an acceptable value and
            #         # continue `The rows are marked `X_mask` and will be
            #         # removed later.
            #         X_mask[:, i] = valid_mask
            #         # cast X_col into the largest string type necessary
            #         # to handle different lengths of numpy strings
            #         if (
            #             self.categories_[i].dtype.kind in ("U", "S")
            #             and self.categories_[i].itemsize > X_col.itemsize
            #         ):
            #             X_col = X_col.astype(self.categories_[i].dtype)
            #         elif self.categories_[i].dtype.kind == "O" and X_col.dtype.kind == "U":
            #             # categories are objects and X_col are numpy strings.
            #             # Cast X_col to an object dtype to prevent truncation
            #             # when setting invalid values.
            #             X_col = X_col.astype("O")
            #         else:
            #             X_col = X_col.copy()
            #         X_col[~valid_mask] = self.categories_[i][0]
            # X_int[:, i] = _encode(X_col, uniques=self.categories_[i], check_unknown=False)
            for j in range(n_samples):
                vals = self._expand_by_separators([X_col[j]])
                for d in vals:
                    try:
                        # index.add(class_mapping[d])
                        per_row_buckets[j].add(class_mapping[d])
                    except KeyError:
                        columns_with_unknown.add(d)

        if columns_with_unknown:
            sorted_unknown = sorted(columns_with_unknown, key=str)
            if handle_unknown == "error":
                # Raise immediately — do not silently encode unknowns as zeros.
                # This enforces the contract documented in the class docstring.
                raise ValueError(
                    f"Found unknown categories {sorted_unknown} in data during "
                    "transform. Set handle_unknown='ignore' or handle_unknown='warn' "
                    "to silently encode unknowns as all zeros."
                )
            # Only warn when explicitly requested.
            # handle_unknown='ignore'  → warn_on_unknown=False → silent (no warning)
            # handle_unknown='warn'    → warn_on_unknown=True  → issue UserWarning
            # NaN/None rows are treated as unknowns and silently zeroed under 'ignore'.
            if warn_on_unknown:
                warnings.warn(
                    (
                        "Found unknown categories in columns "
                        f"{sorted_unknown} during transform. These "
                        "unknown categories will be encoded as all zeros"
                    ),
                    UserWarning,
                )

        # Convert row buckets → CSR lists
        for index in per_row_buckets:
            indices.extend(index)
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=int)

        # self._map_infrequent_categories(X_int, X_mask, ignore_category_indices)
        return (
            sp.csr_matrix(
                (data, indices, indptr),
                shape=(len(indptr) - 1, len(class_mapping)),
                dtype=self.dtype,
            ),
            X_mask,
        )

    def transform(self, X):
        """
        Transform X using one-hot encoding.

        If `sparse_output=True` (default), it returns an instance of
        :class:`scipy.sparse._csr.csr_matrix` (CSR format).

        If there are infrequent categories for a feature, set by specifying
        `max_categories` or `min_frequency`, the infrequent categories are
        grouped into a single category.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : {ndarray, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            Transformed input. If `sparse_output=True`, a sparse matrix will be
            returned.
        """
        check_is_fitted(self)
        transform_output = _get_output_config("transform", estimator=self)["dense"]
        if transform_output != "default" and self.sparse_output:
            capitalize_transform_output = transform_output.capitalize()
            raise ValueError(
                f"{capitalize_transform_output} output does not support sparse data."
                f" Set sparse_output=False to output {transform_output} dataframes or"
                f" disable {capitalize_transform_output} output via"
                '` ohe.set_output(transform="default").'
            )

        # validation of X happens in _check_X called by _transform
        if self.handle_unknown == "warn":
            warn_on_unknown, handle_unknown = True, "infrequent_if_exist"
        else:
            warn_on_unknown = self.drop is not None and self.handle_unknown in {
                "ignore",
                "infrequent_if_exist",
            }
            handle_unknown = self.handle_unknown

        # NOTE: _transform() returns (csr_matrix, X_mask).  X_mask is a boolean
        # array marking valid (known) entries; it is produced by the parent-class
        # _transform() but DummyCodeEncoder overrides _transform() with its own CSR
        # builder that already encodes unknowns as all-zeros.  The mask is therefore
        # not needed here and is intentionally discarded (_mask_unused).
        X_int, _mask_unused = self._transform(
            X,
            handle_unknown=handle_unknown,
            ensure_all_finite="allow-nan",
            warn_on_unknown=warn_on_unknown,
        )

        # Apply drop filtering: remove globally-indexed columns that correspond to
        # the `drop` parameter (e.g. drop='first' removes index 0 within each feature).
        # The CSR was built from all categories; the drop bookkeeping (_set_drop_idx)
        # computed which local-within-feature indices to remove — translate those to
        # global column indices and slice them out now.
        if self._drop_idx_after_grouping is not None:
            # Compute per-feature start offsets in the flat column space
            offsets = np.cumsum([0] + [len(v) for v in self.categories_.values()])
            cols_to_drop = []
            for feat_i, drop_idx in enumerate(self._drop_idx_after_grouping):
                if drop_idx is not None:
                    cols_to_drop.append(int(offsets[feat_i]) + int(drop_idx))
            if cols_to_drop:
                all_cols = list(range(X_int.shape[1]))
                keep_cols = [c for c in all_cols if c not in cols_to_drop]
                X_int = X_int[:, keep_cols]

        if not self.sparse_output:
            return X_int.toarray()
        return X_int

    # @_fit_context(prefer_skip_nested_validation=True)
    # def fit_transform(self, X):
    #     """Fit the label sets binarizer and transform the given label sets.

    #     Parameters
    #     ----------
    #     X : iterable of iterables
    #         A set of labels (any orderable and hashable object) for each
    #         sample. If the `classes` parameter is set, `X` will not be
    #         iterated.

    #     Returns
    #     -------
    #     X_indicator : {ndarray, sparse matrix} of shape (n_samples, n_classes)
    #         A matrix such that `X_indicator[i, j] = 1` iff `classes_[j]`
    #         is in `X[i]`, and 0 otherwise. Sparse matrix will be of CSR
    #         format.
    #     """
    #     if self.categories == "auto":
    #         return self.fit(X).transform(X)

    #     self._cached_dict = None

    #     # Automatically increment on new class
    #     class_mapping = defaultdict(int)
    #     class_mapping.default_factory = class_mapping.__len__
    #     yt = self._transform(y, class_mapping)

    #     # sort classes and reorder columns
    #     tmp = sorted(class_mapping, key=class_mapping.get)

    #     # (make safe for tuples)
    #     dtype = int if all(isinstance(c, int) for c in tmp) else object
    #     class_mapping = np.empty(len(tmp), dtype=dtype)
    #     class_mapping[:] = tmp
    #     self.classes_, inverse = np.unique(class_mapping, return_inverse=True)
    #     # ensure yt.indices keeps its current dtype
    #     yt.indices = np.asarray(inverse[yt.indices], dtype=yt.indices.dtype)

    #     if not self.sparse_output:
    #         yt = yt.toarray()

    #     return yt

    def _decode_rows_from_csr(self, X, categories=None, sep=","):
        categories = categories or self.categories_
        # ---------- Build column offsets ----------
        lens = [len(v) for v in categories.values()]
        offsets = np.cumsum([0] + lens)

        def decode_group(indices):
            groups = [[] for _ in range(len(categories))]

            # fill groups
            for idx in indices:
                col = np.searchsorted(offsets, idx, side="right") - 1
                local = idx - offsets[col]
                groups[col].append(categories[col][local])

            # -------- formatting section --------
            result = []
            for g in groups:
                if len(g) == 0:
                    result.append(None)
                elif len(g) == 1:
                    result.append(g[0])
                else:
                    # join multi-values into a comma-separated string
                    result.append(sep.join(map(str, g)))
            return tuple(result)

        # build all rows
        return [
            decode_group(X.indices[start:end])
            for start, end in zip(X.indptr[:-1], X.indptr[1:])
        ]

    def _decode_rows_from_dense(self, X, categories=None, sep=","):
        """
        Decode a dense label-indicator array (2D) into original categorical values.
        Works for NumPy arrays or pandas DataFrames.
        """
        categories = categories or self.categories_
        X = np.asarray(X)  # convert DataFrame or array to np.array

        n_samples, n_classes = X.shape

        # Build reverse mapping from class index → (col, value)
        offsets = np.cumsum([0] + [len(v) for v in categories.values()])
        n_cols = len(categories)

        # create a flat list of tuples: (column_index, value)
        flat_classes = []
        for col, vals in categories.items():
            for v in vals:
                flat_classes.append((col, v))
        flat_classes = np.array(flat_classes, dtype=object)

        rows = []
        for i in range(n_samples):
            # indices of active classes
            active_idx = np.where(X[i] != 0)[0]

            # build per-column groups
            groups = [[] for _ in range(n_cols)]
            for idx in active_idx:
                col = np.searchsorted(offsets, idx, side="right") - 1
                local_idx = idx - offsets[col]
                groups[col].append(categories[col][local_idx])

            # format: multi → comma string, single → scalar, 0 → None
            decoded_row = []
            for g in groups:
                if len(g) == 0:
                    decoded_row.append(None)
                elif len(g) == 1:
                    decoded_row.append(g[0])
                else:
                    decoded_row.append(sep.join(map(str, g)))
            rows.append(tuple(decoded_row))

        return rows

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        When unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category. If the
        feature with the unknown category has a dropped category, the dropped
        category will be its inverse.

        For a given input feature, if there is an infrequent category,
        'infrequent_sklearn' will be used to represent the infrequent category.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape \
                (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        n_features_out = np.sum(self._n_features_outs)

        # validate shape of passed X
        msg = (
            "Shape of the passed X data is not correct. Expected {0} columns, got {1}."
        )
        if X.shape[1] != n_features_out:
            raise ValueError(msg.format(n_features_out, X.shape[1]))

        # Decode sparse or dense label-indicator matrix back to original categories.
        # NOTE: The commented-out sklearn-style X_tr reconstruction block was removed —
        # it was unreachable dead code that followed two explicit `return` statements.
        # The decode helpers below are the only active paths.
        if sp.issparse(X):
            X = X.tocsr()
            if len(X.data) != 0 and len(np.setdiff1d(X.data, [0, 1])) > 0:
                raise ValueError("Expected only 0s and 1s in label indicator.")
            return self._decode_rows_from_csr(X)
        # dense array / DataFrame
        X_arr = np.asarray(X)
        unexpected = np.setdiff1d(X_arr, [0, 1])
        if len(unexpected) > 0:
            raise ValueError(
                "Expected only 0s and 1s in label indicator. Also got {0}".format(
                    unexpected
                )
            )
        return self._decode_rows_from_dense(X_arr)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        # cats = self.categories_
        cats = [
            self._compute_transformed_categories(i)
            for i, _ in enumerate(self.categories_)
        ]

        name_combiner = self._check_get_feature_name_combiner()
        feature_names = []
        for i in range(len(cats)):
            names = [name_combiner(input_features[i], t) for t in cats[i]]
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)

    def _check_get_feature_name_combiner(self):
        if self.feature_name_combiner == "concat":
            return lambda feature, category: feature + "_" + str(category)
        # callable
        dry_run_combiner = self.feature_name_combiner("feature", "category")
        if not isinstance(dry_run_combiner, str):
            raise TypeError(
                "When `feature_name_combiner` is a callable, it should return a "
                f"Python string. Got {type(dry_run_combiner)} instead."
            )
        return self.feature_name_combiner
