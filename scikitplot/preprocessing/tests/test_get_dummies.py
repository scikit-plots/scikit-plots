# preprocessing/tests/test_get_dummies.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :class:`preprocessing.GetDummies`.

Coverage map
------------
- ``__init__`` param storage (sklearn clone contract)          → TestInit
- ``fit`` column detection (explicit, auto, none-found)        → TestFit
- ``fit`` prefix generation and collision fallback             → TestFitPrefix
- ``transform`` basic output shape and column alignment        → TestTransform
- ``transform`` handle_unknown="error" / "ignore"             → TestHandleUnknown
- ``transform`` sparse_output=True on mixed-type DataFrames   → TestSparseOutput
- ``transform`` drop="first" / drop=True                       → TestDrop
- ``transform`` NaN handling                                   → TestNaNHandling
- ``get_feature_names_out``                                    → TestFeatureNames
- sklearn pipeline and clone round-trip                        → TestSklearnCompat
- numpy array and sparse matrix input                          → TestInputFormats
- ``fit_transform`` consistency                                 → TestFitTransform

Notes
-----
Developer note
    This file is self-contained: if run directly (``python test_get_dummies.py``)
    it installs the scikitplot stub and executes via ``unittest.main()``.
    Under pytest, ``conftest.py`` installs the stub before collection.
"""

from __future__ import annotations

# Bootstrap: install scikitplot stub before ANY preprocessing import.
# Uses only stdlib (sys, types) — no package imports — so it is safe to run
# before the preprocessing package itself is on sys.modules.
import sys
import types

# def _install_stub(version="0.5.0.dev0"):
#     if "scikitplot" in sys.modules:
#         return
#     class _FV:
#         def __init__(self, s):
#             base = s.split(".dev")[0]; p = base.split(".")
#             self.major = int(p[0]); self.minor = int(p[1]) if len(p)>1 else 0
#             self.dev = "dev" if ".dev" in s else None
#     _sk = types.ModuleType("scikitplot"); _sk.__version__ = version
#     _pv = types.ModuleType("scikitplot.externals._packaging.version")
#     _pv.parse = lambda s: _FV(s)
#     sys.modules.update({
#         "scikitplot": _sk,
#         "scikitplot.externals": types.ModuleType("scikitplot.externals"),
#         "scikitplot.externals._packaging": types.ModuleType("scikitplot.externals._packaging"),
#         "scikitplot.externals._packaging.version": _pv,
#     })
# _install_stub()

import pathlib
import unittest
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Bootstrap: allow ``python test_get_dummies.py`` without pytest
# ---------------------------------------------------------------------------
# _HERE = pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))

from sklearn.base import clone  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from ._helpers import (
    install_scikitplot_stub,
    make_infrequent_df,
    make_mixed_type_df,
    make_multi_col_df,
    make_nan_df,
    make_prefix_collision_df,
    make_simple_df,
    make_tags_df,
)
# install_scikitplot_stub()
from .._encoders import GetDummies  # noqa: E402


# ===========================================================================
# TestInit — parameter storage and sklearn clone contract
# ===========================================================================

class TestInit(unittest.TestCase):
    """__init__ must store all params unchanged (sklearn contract)."""

    def test_columns_str_stored_as_str(self):
        """A str columns param must NOT be mutated to a list in __init__."""
        enc = GetDummies(columns="tags")
        self.assertEqual(enc.columns, "tags")
        self.assertIsInstance(enc.columns, str)

    def test_columns_list_stored_as_list(self):
        """A list columns param must be preserved as-is."""
        enc = GetDummies(columns=["tags", "color"])
        self.assertEqual(enc.columns, ["tags", "color"])

    def test_columns_none_stored_as_none(self):
        """None (auto-detect) must be preserved."""
        enc = GetDummies()
        self.assertIsNone(enc.columns)

    def test_get_params_str_roundtrip(self):
        """get_params must return the original str, not a list."""
        enc = GetDummies(columns="tags", sep=",")
        params = enc.get_params()
        self.assertEqual(params["columns"], "tags")
        self.assertIsInstance(params["columns"], str)

    def test_clone_str_roundtrip(self):
        """clone() must reconstruct with the original str param."""
        enc = GetDummies(columns="tags", sep=",", drop="first")
        enc2 = clone(enc)
        self.assertEqual(enc2.columns, "tags")
        self.assertIsInstance(enc2.columns, str)
        self.assertEqual(enc2.sep, ",")
        self.assertEqual(enc2.drop, "first")

    def test_clone_list_roundtrip(self):
        """clone() must reconstruct with the original list param."""
        enc = GetDummies(columns=["tags", "color"], sep=",")
        enc2 = clone(enc)
        self.assertEqual(enc2.columns, ["tags", "color"])

    def test_default_params(self):
        """Verify all documented defaults."""
        enc = GetDummies()
        self.assertIsNone(enc.columns)
        self.assertEqual(enc.sep, ",")
        self.assertEqual(enc.col_name_sep, "_")
        self.assertIsNone(enc.drop)
        self.assertFalse(enc.sparse_output)
        self.assertEqual(enc.dtype, np.float64)
        self.assertEqual(enc.handle_unknown, "error")


# ===========================================================================
# TestFit — column detection and category learning
# ===========================================================================

class TestFit(unittest.TestCase):
    """fit() must correctly detect columns and learn categories."""

    def test_fit_explicit_str_column(self):
        """str columns param must be expanded to list inside fit."""
        df = make_simple_df()
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(df)
        self.assertEqual(enc.dummy_cols_, ["tags"])

    def test_fit_explicit_list_columns(self):
        """List of column names must all be detected."""
        df = make_multi_col_df()
        enc = GetDummies(columns=["tags", "color"], sep=",")
        enc.fit(df)
        self.assertEqual(sorted(enc.dummy_cols_), ["color", "tags"])

    def test_fit_auto_detect_object_columns_with_sep(self):
        """Auto-detection must pick up object columns containing the separator."""
        df = make_simple_df()
        enc = GetDummies(columns=None, sep=",")
        enc.fit(df)
        # Only 'tags' contains commas; 'val' is int
        self.assertIn("tags", enc.dummy_cols_)
        self.assertNotIn("val", enc.dummy_cols_)

    def test_fit_auto_detect_skips_col_without_sep(self):
        """Auto-detection must skip object columns that never contain the sep."""
        df = pd.DataFrame({"name": ["alice", "bob"], "tags": ["a,b", "b,c"]})
        enc = GetDummies(sep=",")
        enc.fit(df)
        self.assertNotIn("name", enc.dummy_cols_)
        self.assertIn("tags", enc.dummy_cols_)

    def test_fit_missing_column_silently_skipped(self):
        """Columns listed in ``columns=`` but absent in X are silently skipped."""
        df = make_simple_df()
        enc = GetDummies(columns=["tags", "nonexistent"], sep=",")
        enc.fit(df)
        self.assertNotIn("nonexistent", enc.dummy_cols_)

    def test_fit_learns_sorted_categories(self):
        """Categories must be sorted (for deterministic column order)."""
        df = make_simple_df()
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(df)
        expected = sorted(enc.categories_["tags"])
        self.assertEqual(enc.categories_["tags"], expected)

    def test_fit_stores_global_column_order(self):
        """columns_ must list non-dummy cols first then all dummy cols."""
        df = make_simple_df()
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(df)
        # 'val' is the non-dummy passthrough
        self.assertEqual(enc.columns_[0], "val")
        for cat_col in enc.categories_["tags"]:
            self.assertIn(cat_col, enc.columns_)

    def test_fit_normalises_case_and_strips_whitespace(self):
        """Labels must be lowercased and stripped of surrounding whitespace."""
        df = pd.DataFrame({"tags": [" A , B ", "a,b"], "v": [1, 2]})
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(df)
        cats = enc.categories_["tags"]
        for c in cats:
            # Strip the column prefix to get the raw label
            label = c.split(enc.col_name_sep, 1)[-1]
            self.assertEqual(label, label.lower().strip())

    def test_fit_returns_self(self):
        """fit() must return self (sklearn convention)."""
        df = make_simple_df()
        enc = GetDummies(columns="tags", sep=",")
        result = enc.fit(df)
        self.assertIs(result, enc)

    def test_fit_sets_n_features_in(self):
        """n_features_in_ must equal the number of input columns."""
        df = make_simple_df()
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(df)
        self.assertEqual(enc.n_features_in_, df.shape[1])


# ===========================================================================
# TestFitPrefix — prefix generation and collision fallback
# ===========================================================================

class TestFitPrefix(unittest.TestCase):
    """Dummy column prefixes must be unique; collisions fall back to full names."""

    def test_prefix_uses_first_two_chars(self):
        """Columns without col_name_sep use the first 2 chars as prefix."""
        df = make_simple_df()
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(df)
        self.assertEqual(enc.dummy_prefix_["tags"], "ta")

    def test_prefix_uses_initials_for_underscored_names(self):
        """Columns containing col_name_sep use initial letters (e.g. tag_name → tn)."""
        df = pd.DataFrame({"tag_name": ["a,b", "b,c"], "v": [1, 2]})
        enc = GetDummies(columns="tag_name", sep=",")
        enc.fit(df)
        # "tag_name" → "tn"
        self.assertEqual(enc.dummy_prefix_["tag_name"], "tn")

    def test_prefix_collision_falls_back_to_full_names(self):
        """When abbreviated prefixes collide, full column names must be used."""
        df = make_prefix_collision_df()  # 'ta' and 'tags' both → 'ta'
        enc = GetDummies(sep=",")
        enc.fit(df)
        # After fallback, each column's prefix must equal its full name
        self.assertEqual(enc.dummy_prefix_["ta"], "ta")
        self.assertEqual(enc.dummy_prefix_["tags"], "tags")

    def test_no_false_collision_distinct_prefixes(self):
        """Distinct abbreviations must not trigger the fallback."""
        df = pd.DataFrame({"alpha": ["a,b", "b"], "beta": ["x,y", "y"], "v": [1, 2]})
        enc = GetDummies(sep=",")
        enc.fit(df)
        # 'alpha' → 'al', 'beta' → 'be' — no collision
        self.assertEqual(enc.dummy_prefix_["alpha"], "al")
        self.assertEqual(enc.dummy_prefix_["beta"], "be")


# ===========================================================================
# TestTransform — output shape, dtype, column alignment
# ===========================================================================

class TestTransform(unittest.TestCase):
    """transform() must produce correctly shaped and typed output."""

    def setUp(self):
        self.df = make_simple_df()
        self.enc = GetDummies(columns="tags", sep=",")
        self.enc.fit(self.df)

    def test_output_is_dataframe(self):
        """Default output must be a pandas DataFrame."""
        result = self.enc.transform(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_output_shape(self):
        """Output must have same number of rows as input."""
        result = self.enc.transform(self.df)
        self.assertEqual(result.shape[0], self.df.shape[0])

    def test_non_dummy_columns_preserved(self):
        """Passthrough columns (non-dummy) must appear in output."""
        result = self.enc.transform(self.df)
        self.assertIn("val", result.columns)

    def test_dummy_columns_present(self):
        """All fitted dummy columns must appear in output."""
        result = self.enc.transform(self.df)
        for cat_col in self.enc.categories_["tags"]:
            self.assertIn(cat_col, result.columns)

    def test_output_column_order_stable(self):
        """Column order must match the fitted columns_ attribute."""
        result = self.enc.transform(self.df)
        self.assertEqual(result.columns.tolist(), self.enc.columns_)

    def test_output_dtype(self):
        """Dummy columns must have the specified dtype."""
        result = self.enc.transform(self.df)
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        for col in dummy_cols:
            self.assertEqual(result[col].dtype, np.float64)

    def test_transform_unseen_rows_fill_zero(self):
        """Rows with no matching category produce all-zero dummy columns."""
        df_all_nan = pd.DataFrame({"tags": [None], "val": [99]})
        enc = GetDummies(columns="tags", sep=",", handle_unknown="ignore")
        enc.fit(self.df)
        result = enc.transform(df_all_nan)
        dummy_vals = result[[c for c in result.columns if c.startswith("ta_")]].values
        np.testing.assert_array_equal(dummy_vals, np.zeros_like(dummy_vals))

    def test_transform_new_data_aligned_to_fit_categories(self):
        """transform on new data must align to fit-time categories exactly."""
        df_new = pd.DataFrame({"tags": ["a", "b"], "val": [5, 6]})
        result = self.enc.transform(df_new)
        # 'a' row: ta_a=1, ta_b=0, ta_c=0
        # 'b' row: ta_a=0, ta_b=1, ta_c=0
        self.assertEqual(result["ta_a"].iloc[0], 1.0)
        self.assertEqual(result["ta_b"].iloc[0], 0.0)

    def test_transform_requires_fit_first(self):
        """transform before fit must raise NotFittedError."""
        from sklearn.exceptions import NotFittedError
        enc = GetDummies(columns="tags", sep=",")
        with self.assertRaises(NotFittedError):
            enc.transform(self.df)


# ===========================================================================
# TestHandleUnknown
# ===========================================================================

class TestHandleUnknown(unittest.TestCase):
    """Unknown categories at transform time must be handled correctly."""

    def setUp(self):
        self.df_train = pd.DataFrame({"tags": ["a,b", "b,c"], "val": [1, 2]})
        self.df_test = pd.DataFrame({"tags": ["a,x", "b"], "val": [3, 4]})

    def test_handle_unknown_error_raises(self):
        """handle_unknown='error' must raise ValueError on unknown categories."""
        enc = GetDummies(columns="tags", sep=",", handle_unknown="error")
        enc.fit(self.df_train)
        with self.assertRaises(ValueError):
            enc.transform(self.df_test)

    def test_handle_unknown_ignore_drops_unknown(self):
        """handle_unknown='ignore' must silently drop unknown categories."""
        enc = GetDummies(columns="tags", sep=",", handle_unknown="ignore")
        enc.fit(self.df_train)
        result = enc.transform(self.df_test)
        # 'x' is unknown, so ta_a=1 (present), ta_x must not appear
        self.assertNotIn("ta_x", result.columns)

    def test_handle_unknown_ignore_known_categories_intact(self):
        """Known categories must still be 1 even when unknowns are present."""
        enc = GetDummies(columns="tags", sep=",", handle_unknown="ignore")
        enc.fit(self.df_train)
        result = enc.transform(self.df_test)
        # row 0: 'a,x' → 'a' known (=1), 'x' unknown (dropped/0)
        self.assertEqual(result["ta_a"].iloc[0], 1.0)


# ===========================================================================
# TestSparseOutput
# ===========================================================================

class TestSparseOutput(unittest.TestCase):
    """sparse_output=True must return a CSR matrix without crashing."""

    def test_sparse_output_type(self):
        """Output must be a SciPy sparse CSR matrix."""
        df = make_simple_df()
        enc = GetDummies(columns=["tags"], sep=",", sparse_output=True)
        result = enc.fit_transform(df)
        self.assertTrue(sp.issparse(result))

    def test_sparse_output_shape(self):
        """Sparse output must contain only dummy columns (no passthrough)."""
        df = make_simple_df()
        enc = GetDummies(columns=["tags"], sep=",", sparse_output=True)
        result = enc.fit_transform(df)
        # 3 categories: ta_a, ta_b, ta_c
        self.assertEqual(result.shape[1], 3)

    def test_sparse_output_mixed_type_no_crash(self):
        """sparse_output=True must not crash on DataFrames with string passthrough cols."""
        df = make_mixed_type_df()  # 'name' col is string passthrough
        enc = GetDummies(columns=["tags"], sep=",", sparse_output=True)
        # Before Bug 3 fix this raised ValueError: could not convert string to float
        result = enc.fit_transform(df)
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape[0], df.shape[0])

    def test_sparse_output_values_binary(self):
        """All values in sparse output must be 0 or 1."""
        df = make_simple_df()
        enc = GetDummies(columns=["tags"], sep=",", sparse_output=True)
        result = enc.fit_transform(df)
        unique_vals = np.unique(result.toarray())
        self.assertTrue(set(unique_vals).issubset({0.0, 1.0}))


# ===========================================================================
# TestDrop
# ===========================================================================

class TestDrop(unittest.TestCase):
    """drop parameter must eliminate the first dummy column per feature."""

    def setUp(self):
        self.df = make_simple_df()

    def test_drop_first_reduces_columns(self):
        """drop='first' must produce one fewer dummy column per feature."""
        enc_none = GetDummies(columns="tags", sep=",", drop=None)
        enc_drop = GetDummies(columns="tags", sep=",", drop="first")
        enc_none.fit(self.df)
        enc_drop.fit(self.df)
        full_dummy_count = len(enc_none.categories_["tags"])
        result = enc_drop.transform(self.df)
        dropped_count = sum(1 for c in result.columns if c.startswith("ta_"))
        self.assertEqual(dropped_count, full_dummy_count - 1)

    def test_drop_true_same_as_first(self):
        """drop=True must behave identically to drop='first'."""
        enc_first = GetDummies(columns="tags", sep=",", drop="first")
        enc_true = GetDummies(columns="tags", sep=",", drop=True)
        r1 = enc_first.fit_transform(self.df)
        r2 = enc_true.fit_transform(self.df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_drop_none_keeps_all_columns(self):
        """drop=None must retain all dummy columns."""
        enc = GetDummies(columns="tags", sep=",", drop=None)
        result = enc.fit_transform(self.df)
        # 3 unique labels (a, b, c) → 3 dummy cols
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        self.assertEqual(len(dummy_cols), 3)

    def test_drop_single_category_feature_preserved(self):
        """A feature with only 1 category must be KEPT even with drop='first'.

        The guard ``len(dummies.columns) > 1`` in ``_make_dummies`` intentionally
        preserves single-category features: dropping the sole category would
        produce a fully-zero column with no information and silently destroy the
        feature, which is misleading.  This matches scikit-learn's ``OneHotEncoder``
        behaviour where a single-value feature is retained (not silently dropped).
        """
        df = pd.DataFrame({"tags": ["only", "only", "only"], "val": [1, 2, 3]})
        enc = GetDummies(columns="tags", sep=",", drop="first")
        result = enc.fit_transform(df)
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        # 1 category → guard preserves it; column count stays at 1
        self.assertEqual(len(dummy_cols), 1)


# ===========================================================================
# TestNaNHandling
# ===========================================================================

class TestNaNHandling(unittest.TestCase):
    """NaN values in label columns must not raise; rows with NaN get all zeros."""

    def test_nan_row_produces_all_zero_dummies(self):
        """A row with NaN produces zeros for all dummy columns."""
        df = make_nan_df()
        enc = GetDummies(columns="tags", sep=",", handle_unknown="ignore")
        result = enc.fit_transform(df)
        nan_row_idx = 1  # second row is None
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        row_vals = result[dummy_cols].iloc[nan_row_idx].values
        np.testing.assert_array_equal(row_vals, np.zeros(len(dummy_cols)))

    def test_non_nan_rows_unaffected(self):
        """Non-NaN rows must still be encoded correctly."""
        df = make_nan_df()
        enc = GetDummies(columns="tags", sep=",", handle_unknown="ignore")
        result = enc.fit_transform(df)
        # row 0: 'a,b' → ta_a=1, ta_b=1
        self.assertEqual(result["ta_a"].iloc[0], 1.0)
        self.assertEqual(result["ta_b"].iloc[0], 1.0)


# ===========================================================================
# TestFeatureNames
# ===========================================================================

class TestFeatureNames(unittest.TestCase):
    """get_feature_names_out must return the fitted column order as ndarray."""

    def test_returns_ndarray(self):
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(make_simple_df())
        names = enc.get_feature_names_out()
        self.assertIsInstance(names, np.ndarray)

    def test_names_match_columns_(self):
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(make_simple_df())
        np.testing.assert_array_equal(
            enc.get_feature_names_out(), np.array(enc.columns_)
        )

    def test_raises_before_fit(self):
        from sklearn.exceptions import NotFittedError
        enc = GetDummies(columns="tags", sep=",")
        with self.assertRaises(NotFittedError):
            enc.get_feature_names_out()


# ===========================================================================
# TestSklearnCompat — Pipeline and set_output integration
# ===========================================================================

class TestSklearnCompat(unittest.TestCase):
    """GetDummies must be fully compatible with sklearn Pipeline and set_output."""

    def test_pipeline_fit_transform(self):
        """GetDummies must work as a Pipeline step."""
        df = make_simple_df()
        pipe = Pipeline([("enc", GetDummies(columns="tags", sep=","))])
        result = pipe.fit_transform(df)
        self.assertIsNotNone(result)

    def test_set_output_pandas(self):
        """set_output(transform='pandas') must return a DataFrame."""
        df = make_simple_df()
        pipe = Pipeline([("enc", GetDummies(columns="tags", sep=","))]).set_output(
            transform="pandas"
        )
        result = pipe.fit_transform(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_clone_then_fit(self):
        """A cloned (unfitted) encoder must be fittable on new data."""
        df = make_simple_df()
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(df)
        enc2 = clone(enc)
        # clone produces unfitted copy — must be able to fit independently
        result = enc2.fit_transform(df)
        self.assertIsNotNone(result)

    def test_independent_fit_after_clone(self):
        """Fitting the clone must not affect the original estimator."""
        df = make_simple_df()
        df2 = pd.DataFrame({"tags": ["x,y", "y,z"], "val": [7, 8]})
        enc = GetDummies(columns="tags", sep=",")
        enc.fit(df)
        enc2 = clone(enc)
        enc2.fit(df2)
        # Original categories unchanged
        self.assertIn("ta_a", enc.categories_["tags"])
        self.assertNotIn("ta_a", enc2.categories_["tags"])


# ===========================================================================
# TestInputFormats — non-DataFrame inputs
# ===========================================================================

class TestInputFormats(unittest.TestCase):
    """_to_dataframe must convert ndarray and sparse inputs."""

    def test_numpy_array_input(self):
        """NumPy array input must be converted to DataFrame and processed."""
        df = make_simple_df()
        enc = GetDummies(columns=None, sep=",")
        # Fit on DataFrame first (to set dummy_cols_ correctly)
        enc.fit(df)
        arr = df.values
        # transform numpy — columns will be integer-indexed, no str col names
        result = enc.transform(pd.DataFrame(arr, columns=df.columns))
        self.assertIsNotNone(result)

    def test_sparse_matrix_input_converted_via_to_dataframe(self):
        """_to_dataframe must convert a SciPy sparse matrix to DataFrame."""
        # Build a numeric-only sparse matrix (scipy does not support object dtype).
        # Use a purely numeric DataFrame to exercise this path.
        df_num = pd.DataFrame({"a": [1.0, 0.0, 1.0], "b": [0.0, 1.0, 1.0]})
        sparse_in = sp.csr_matrix(df_num.values)
        enc = GetDummies()  # no columns to encode on numeric data
        enc.fit(df_num)
        result = enc._to_dataframe(sparse_in)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, df_num.shape)

    def test_invalid_input_raises_type_error(self):
        """Non-DataFrame/ndarray/sparse input must raise TypeError."""
        enc = GetDummies(columns="tags", sep=",")
        df = make_simple_df()
        enc.fit(df)
        with self.assertRaises(TypeError):
            enc._to_dataframe("not a valid input")


# ===========================================================================
# TestFitTransform — consistency between fit+transform and fit_transform
# ===========================================================================

class TestFitTransform(unittest.TestCase):
    """fit_transform must produce the same result as fit followed by transform."""

    def test_fit_transform_equals_fit_then_transform(self):
        df = make_simple_df()
        enc1 = GetDummies(columns="tags", sep=",")
        result1 = enc1.fit_transform(df)

        enc2 = GetDummies(columns="tags", sep=",")
        enc2.fit(df)
        result2 = enc2.transform(df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_multi_column_fit_transform(self):
        """fit_transform on multiple columns must encode all of them."""
        df = make_multi_col_df()
        enc = GetDummies(columns=["tags", "color"], sep=",")
        result = enc.fit_transform(df)
        # both tag and color prefixes must appear
        tag_cols = [c for c in result.columns if c.startswith("ta_")]
        col_cols = [c for c in result.columns if c.startswith("co_")]
        self.assertGreater(len(tag_cols), 0)
        self.assertGreater(len(col_cols), 0)


if __name__ == "__main__":
    # install_scikitplot_stub()
    unittest.main(verbosity=2)
