# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-column multi-label string column one-hot encoder [1]_.

References
----------
.. [1] `Çelik, M. (2023, December 9).
   "How to converting pandas column of comma-separated strings into dummy variables?."
   Medium. https://medium.com/@celik-muhammed/how-to-converting-pandas-column-of-comma-separated-strings-into-dummy-variables-762c02282a6c
   <https://medium.com/@celik-muhammed/how-to-converting-pandas-column-of-comma-separated-strings-into-dummy-variables-762c02282a6c>`_
"""

# get_dummies.py
import numpy as np
import pandas as pd
import scipy.sparse as sp  # SciPy sparse CSR/CSC matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# from sklearn.preprocessing import MultiLabelBinarizer


class GetDummies(BaseEstimator, TransformerMixin):
    """
    Multi-column multi-label string column one-hot encoder [1]_.

    Custom transformer to expand string columns that contain multiple labels
    separated by `data_sep` into one-hot encoded columns by :func:`pandas.get_dummies`.

    Compatible with sklearn pipelines, `set_output` API, and supports both
    dense and sparse output.

    Parameters
    ----------
    columns : str, list of str or None, default=None
        Column(s) to encode. If str, single column. If list, multiple columns.
        If None, automatically detect object (string) columns containing `data_sep`.
    data_sep : str, default=","
        Separator used inside string cells, e.g., "a,b,c".
    col_name_sep : str, default="_"
        Separator for new dummy column names, e.g., "tags_a".
    dtype : data-type, default=float
        Data type for the output values. (sklearn default is float)
    sparse_output : bool, default=False
        If True, return a SciPy sparse CSR matrix.
        If False, return pandas DataFrame (or numpy array if set_output="default").
    handle_unknown : {"ignore", "error"}, default="ignore"
        Strategy for unknown categories at transform time.
    drop_first : {"first", True, None}, default=None
        Drop the first dummy in each feature (sorted order) to avoid collinearity.

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> df = pd.DataFrame(
    ...     {
    ...         "tags": ["a,b,", " A , b", "a,B,C", None],
    ...         "color": ["red", "blue", "green", "Red"],
    ...         "value": [1, 2, 3, 4],
    ...     }
    ... )
    >>> pipe = Pipeline(
    ...     [
    ...         (
    ...             "encoder",
    ...             GetDummies(
    ...                 columns=["tags", "color"], drop_first=None, sparse_output=False
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

    See Also
    --------
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
    """

    def __init__(
        self,
        columns=None,
        data_sep=",",
        col_name_sep="_",
        dtype=float,  # np.uint8,
        sparse_output=False,
        handle_unknown="ignore",
        drop_first=None,
    ):
        if isinstance(columns, str):
            self.columns = [columns]  # Normalize single column str → list
        else:
            # Columns to encode. If None, scan all object columns
            self.columns = columns  # list or None
        # Separator for multiple values in a cell
        self.data_sep = data_sep
        # Separator for dummy column names
        self.col_name_sep = col_name_sep
        # Output data type
        self.dtype = dtype
        # Whether to output SciPy sparse matrix
        self.sparse_output = sparse_output
        # Strategy for unknown categories at transform
        self.handle_unknown = handle_unknown
        # Drop first dummy column to avoid multicollinearity
        self.drop_first = drop_first

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

    def _split_and_clean(self, series):
        """
        Split strings by separator and normalize.

        Steps:
        - lowercase
        - strip whitespace
        - drop empty entries
        - rejoin into normalized string so pd.get_dummies can handle
        """
        return (
            series.fillna("")  # Replace NaN with empty string
            .str.split(self.data_sep)  # split into list of tokens
            .apply(
                lambda lst: self.data_sep.join(  # rejoin back to single string
                    sorted(  # optional: sort tokens for consistency
                        {
                            s.strip().lower() for s in lst if s.strip()
                        }  # deduplicate + normalize
                    )
                )
            )
        )

    def _make_dummies(self, series, colname):
        """
        Convert cleaned string column into dummy variables.

        Parameters
        ----------
        series : pd.Series
            The column to convert.
        colname : str
            Column name (used to create prefixed dummy names).
        """
        # Split + normalize string into sets of labels
        series = self._split_and_clean(series)

        # TODO: Use MultiLabelBinarizer if sparse_output=True (efficient)
        # if self.sparse_output:
        #     mlb = MultiLabelBinarizer(
        #         sparse_output=self.sparse_output,
        #         dtype=self.dtype,
        #     )
        #     mlb.fit(series)
        #     cats = mlb.classes_

        # Expand multi-label strings into one-hot dummy columns
        # dummies = pd.get_dummies(  # Expand to dummy variables
        #     data=pd.Series(series),
        #     dtype=self.dtype,
        #     # drop_first=self.drop_first,
        # )
        dummies = series.str.get_dummies(  # Expand to dummy variables
            sep=self.data_sep  # default "|"
        )
        # Prefix to column names to avoid collisions
        dummies = dummies.add_prefix(self.dummy_prefix_[colname] + self.col_name_sep)
        # Drop first dummy if requested (to avoid multicollinearity)
        if self.drop_first in ["first", True] and len(dummies.columns) > 1:
            dummies = dummies.iloc[:, 1:]
        # Ensure dtype is correct
        if self.dtype is not None:
            dummies = dummies.astype(self.dtype)  # Set dtype
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

        # Determine columns to encode
        if self.columns is not None:
            # User-specified columns, Only keep those that exist in DataFrame
            self.dummy_cols_ = [col for col in self.columns if col in X.columns]
        else:
            # Automatically detect object/string columns containing the separator, e.g., "a,b,c".
            object_cols = X.select_dtypes(
                include="O"
            ).columns  # Identify object/string columns
            # Keep only those that contain multiple values (separator present)
            self.dummy_cols_ = [
                col
                for col in object_cols
                if X[col].str.contains(self.data_sep, regex=False).any()
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
        if len(set(self.dummy_prefix_)) != len(set(self.dummy_prefix_.values())):
            # Build mapping for prefixes (use full col name for safety, not abbreviations)
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

        # Return SciPy sparse CSR matrix if requested
        if self.sparse_output:
            return sp.csr_matrix(X_out.to_numpy(dtype=self.dtype))

        # Return numpy array if set_output(transform="default") and pipeline expects dense array
        if hasattr(self, "_get_output_config"):
            cfg = self._get_output_config()
            if cfg.get("dense", False) is False:
                return X_out.to_numpy(dtype=self.dtype)  # Dense numpy output

        # Default: return pandas DataFrame
        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        check_is_fitted(self, "columns_")
        return np.array(self.columns_)
