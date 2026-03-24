# preprocessing/tests/test_encoders_advanced.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Advanced and edge-case tests for :mod:`preprocessing._encoders`.

This module extends the base test suites with scenarios that require
cross-encoder comparisons, unusual parameter combinations, and
all boundary paths not covered in the primary test files.

Coverage map (gaps addressed here)
-----------------------------------
GetDummies
  - ``dtype`` parameter forwarded to dummy columns                → TestGetDummiesDtype
  - ``columns=None`` on all-numeric DataFrame (passthrough)       → TestGetDummiesPassthrough
  - Two multi-label columns encoded together                       → TestGetDummiesMultiCol
  - ``handle_unknown='ignore'`` all-unknown rows produce zeros    → TestGetDummiesAllUnknown
  - ``set_output(transform='pandas')`` in Pipeline                → TestGetDummiesSetOutput

DummyCodeEncoder
  - Manual ``categories=[...]`` parameter                         → TestDCEManualCategories
  - Bug 10: manual categories unhashable-list TypeError fixed     → TestDCEManualCategories
  - ``dummy_na=True`` NaN column added to output                  → TestDCEDummyNa
  - ``handle_unknown='warn'`` issues warning, still encodes       → TestDCEWarnUnknown
  - ``drop='first'`` + ``sparse_output=True`` combined           → TestDCEDropSparse
  - ``inverse_transform`` on drop-reduced output                  → TestDCEInverseWithDrop
  - ``fit_transform`` == ``fit`` + ``transform``                  → TestDCEFitTransformConsistency
  - ``set_output(transform='pandas')`` dense mode                 → TestDCESetOutput
  - ``feature_names_in_`` set after fit on named DataFrame        → TestDCEFeatureNamesIn
  - Multi-label inverse: row with 3 active labels                 → TestDCEInverseMultiLabel

Notes
-----
Developer note
    Run standalone::

        python test_encoders_advanced.py

    Run under pytest::

        pytest preprocessing/tests/ -v --tb=short
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
#             self.major = int(p[0]); self.minor = int(p[1]) if len(p) > 1 else 0
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
import re
import unittest
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

# _HERE = pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))
from sklearn.base import clone  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .._encoders import DummyCodeEncoder, GetDummies  # noqa: E402


# ===========================================================================
# GetDummies — dtype
# ===========================================================================

class TestGetDummiesDtype(unittest.TestCase):
    """The ``dtype`` parameter must propagate to all dummy columns."""

    def test_dtype_float32(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c"], "v": [1, 2]})
        enc = GetDummies(columns="tags", sep=",", dtype=np.float32)
        result = enc.fit_transform(df)
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        for col in dummy_cols:
            self.assertEqual(result[col].dtype, np.float32)

    def test_dtype_uint8(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c"], "v": [1, 2]})
        enc = GetDummies(columns="tags", sep=",", dtype=np.uint8)
        result = enc.fit_transform(df)
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        for col in dummy_cols:
            self.assertEqual(result[col].dtype, np.uint8)

    def test_dtype_int32(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c"], "v": [1, 2]})
        enc = GetDummies(columns="tags", sep=",", dtype=np.int32)
        result = enc.fit_transform(df)
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        for col in dummy_cols:
            self.assertEqual(result[col].dtype, np.int32)

    def test_dtype_preserved_clone(self):
        """dtype must survive clone() and produce the same column types."""
        df = pd.DataFrame({"tags": ["a,b", "b,c"], "v": [1, 2]})
        enc = GetDummies(columns="tags", sep=",", dtype=np.uint8)
        enc2 = clone(enc)
        result = enc2.fit_transform(df)
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        for col in dummy_cols:
            self.assertEqual(result[col].dtype, np.uint8)


# ===========================================================================
# GetDummies — all-numeric passthrough (no columns to encode)
# ===========================================================================

class TestGetDummiesPassthrough(unittest.TestCase):
    """When no encodable columns are found, input is returned unchanged."""

    def test_all_numeric_no_sep(self):
        """All-numeric DataFrame with no string columns → pure passthrough."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        enc = GetDummies(sep=",")
        result = enc.fit_transform(df)
        self.assertEqual(result.shape, df.shape)
        self.assertEqual(enc.dummy_cols_, [])

    def test_string_cols_without_sep_not_encoded(self):
        """String columns that never contain the separator are not encoded."""
        df = pd.DataFrame({"name": ["alice", "bob", "carol"], "v": [1, 2, 3]})
        enc = GetDummies(sep=",")
        result = enc.fit_transform(df)
        # 'name' has no commas — should remain untouched
        self.assertIn("name", result.columns)
        self.assertEqual(enc.dummy_cols_, [])

    def test_passthrough_output_matches_input(self):
        """Passthrough output must equal the original DataFrame."""
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        enc = GetDummies(sep=",")
        result = enc.fit_transform(df)
        pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                      df.reset_index(drop=True))


# ===========================================================================
# GetDummies — multiple encode columns
# ===========================================================================

class TestGetDummiesMultiCol(unittest.TestCase):
    """Two multi-label columns must both be expanded and interleaved correctly."""

    def setUp(self):
        self.df = pd.DataFrame({
            "a": ["x,y", "y,z"],
            "b": ["p,q", "q,r"],
            "v": [1, 2],
        })

    def test_both_cols_expanded(self):
        enc = GetDummies(columns=["a", "b"], sep=",")
        result = enc.fit_transform(self.df)
        a_cols = [c for c in result.columns if c.startswith("a_")]
        b_cols = [c for c in result.columns if c.startswith("b_")]
        self.assertGreater(len(a_cols), 0)
        self.assertGreater(len(b_cols), 0)

    def test_passthrough_col_present(self):
        enc = GetDummies(columns=["a", "b"], sep=",")
        result = enc.fit_transform(self.df)
        self.assertIn("v", result.columns)

    def test_output_rows_match(self):
        enc = GetDummies(columns=["a", "b"], sep=",")
        result = enc.fit_transform(self.df)
        self.assertEqual(result.shape[0], len(self.df))

    def test_column_order_non_dummy_first(self):
        """Non-dummy columns must appear before dummy columns."""
        enc = GetDummies(columns=["a", "b"], sep=",")
        result = enc.fit_transform(self.df)
        # 'v' is the only non-dummy col; it must be first
        self.assertEqual(result.columns[0], "v")


# ===========================================================================
# GetDummies — all-unknown rows with handle_unknown='ignore'
# ===========================================================================

class TestGetDummiesAllUnknown(unittest.TestCase):
    """Rows where all labels are unknown must produce all-zero dummy blocks."""

    def setUp(self):
        self.df_train = pd.DataFrame({"tags": ["a,b", "b,c"], "val": [1, 2]})
        self.df_test_all_unknown = pd.DataFrame({"tags": ["x,y", "z"], "val": [3, 4]})

    def test_all_unknown_rows_are_zero(self):
        enc = GetDummies(columns="tags", sep=",", handle_unknown="ignore")
        enc.fit(self.df_train)
        result = enc.transform(self.df_test_all_unknown)
        dummy_cols = [c for c in result.columns if c.startswith("ta_")]
        dummy_block = result[dummy_cols].values
        np.testing.assert_array_equal(dummy_block, np.zeros_like(dummy_block))

    def test_passthrough_cols_still_correct(self):
        """Passthrough values must not be affected by unknown category handling."""
        enc = GetDummies(columns="tags", sep=",", handle_unknown="ignore")
        enc.fit(self.df_train)
        result = enc.transform(self.df_test_all_unknown)
        np.testing.assert_array_equal(result["val"].values, [3, 4])


# ===========================================================================
# GetDummies — set_output API
# ===========================================================================

class TestGetDummiesSetOutput(unittest.TestCase):
    """set_output integration via sklearn Pipeline."""

    def setUp(self):
        self.df = pd.DataFrame({"tags": ["a,b", "b,c", "a"], "val": [1, 2, 3]})

    def test_set_output_pandas_returns_dataframe(self):
        pipe = Pipeline([("enc", GetDummies(columns="tags", sep=","))]).set_output(
            transform="pandas"
        )
        result = pipe.fit_transform(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_set_output_pandas_has_feature_names(self):
        """DataFrame columns must match get_feature_names_out."""
        pipe = Pipeline([("enc", GetDummies(columns="tags", sep=","))]).set_output(
            transform="pandas"
        )
        result = pipe.fit_transform(self.df)
        enc = pipe.named_steps["enc"]
        np.testing.assert_array_equal(
            result.columns.tolist(), enc.get_feature_names_out().tolist()
        )

    def test_set_output_default_returns_dataframe(self):
        """set_output='default' returns the native GetDummies output (DataFrame)."""
        pipe = Pipeline([("enc", GetDummies(columns="tags", sep=","))]).set_output(
            transform="default"
        )
        result = pipe.fit_transform(self.df)
        # GetDummies.transform returns a DataFrame by default
        self.assertIsInstance(result, pd.DataFrame)


# ===========================================================================
# DummyCodeEncoder — manual categories
# ===========================================================================

class TestDCEManualCategories(unittest.TestCase):
    """Manual ``categories=[...]`` must be used verbatim; no TypeError."""

    def setUp(self):
        self.df = pd.DataFrame({"tags": ["a,b", "b,c"]})

    def test_manual_categories_shape(self):
        """Specifying 4 categories must produce 4 output columns per feature."""
        enc = DummyCodeEncoder(sep=",", categories=[["a", "b", "c", "d"]],
                               sparse_output=False)
        result = enc.fit_transform(self.df)
        # 4 categories for 1 feature
        self.assertEqual(result.shape[1], 4)

    def test_manual_categories_no_unhashable_error(self):
        """Bug 10: passing categories=[list] must not raise TypeError."""
        # Before fix this raised: TypeError: unhashable type: 'list'
        enc = DummyCodeEncoder(sep=",", categories=[["a", "b", "c"]],
                               sparse_output=False)
        try:
            enc.fit(self.df)
        except TypeError as e:
            self.fail(f"Raised TypeError unexpectedly: {e}")

    def test_manual_categories_feature_names(self):
        """Feature names must reflect the manually supplied categories."""
        enc = DummyCodeEncoder(sep=",", categories=[["a", "b", "c", "d"]],
                               sparse_output=False)
        enc.fit(self.df)
        names = enc.get_feature_names_out()
        self.assertIn("tags_a", names)
        self.assertIn("tags_d", names)

    def test_manual_categories_preserves_unseen_in_fit(self):
        """Categories not present in training data must appear as zero columns."""
        enc = DummyCodeEncoder(sep=",", categories=[["a", "b", "c", "d"]],
                               sparse_output=False)
        result = enc.fit_transform(self.df)
        # 'd' never appears → its column should be all zeros
        d_col_idx = list(enc.get_feature_names_out()).index("tags_d")
        self.assertTrue(np.all(result[:, d_col_idx] == 0))

    def test_manual_categories_wrong_type_raises(self):
        """Passing non-list non-'auto' categories must raise ValueError."""
        enc = DummyCodeEncoder(sep=",", categories="invalid_string",
                               sparse_output=False)
        with self.assertRaises((ValueError, AttributeError)):
            enc.fit(self.df)


# ===========================================================================
# DummyCodeEncoder — dummy_na
# ===========================================================================

class TestDCEDummyNa(unittest.TestCase):
    """``dummy_na=True`` must add a NaN-indicator column to each feature."""

    def setUp(self):
        self.df_with_nan = pd.DataFrame({"tags": ["a,b", None, "b,c"]})
        self.df_no_nan = pd.DataFrame({"tags": ["a,b", "b,c", "a"]})

    def test_dummy_na_adds_nan_column(self):
        """dummy_na=True must produce an extra column when NaN is present in fit data.

        ``dummy_na=True`` adds a NaN-indicator column only when the fit data
        actually contains missing values (None / np.nan).  On clean data the
        two encoders produce the same width; on NaN-containing data the
        ``dummy_na=True`` encoder is one column wider.
        """
        # Fit on data that contains NaN → dummy_na=True includes a nan category
        enc_na = DummyCodeEncoder(sep=",", dummy_na=True, sparse_output=False)
        # dummy_na=False + handle_unknown='ignore' → NaN row treated as all-zeros
        enc_no = DummyCodeEncoder(sep=",", dummy_na=False,
                                  handle_unknown="ignore", sparse_output=False)
        enc_na.fit(self.df_with_nan)
        enc_no.fit(self.df_with_nan)
        r_na = enc_na.transform(self.df_with_nan)
        r_no = enc_no.transform(self.df_with_nan)
        # dummy_na=True learned NaN as a category → one extra output column
        self.assertGreater(r_na.shape[1], r_no.shape[1])

    def test_dummy_na_column_named_nan(self):
        """The NaN-indicator column must appear in feature names."""
        enc = DummyCodeEncoder(sep=",", dummy_na=True, sparse_output=False)
        enc.fit(self.df_with_nan)
        names = enc.get_feature_names_out()
        nan_names = [n for n in names if "nan" in n.lower() or "none" in n.lower()]
        self.assertGreater(len(nan_names), 0)

    def test_dummy_na_false_no_nan_column(self):
        """dummy_na=False must not add a NaN column."""
        enc = DummyCodeEncoder(sep=",", dummy_na=False, sparse_output=False)
        enc.fit(self.df_with_nan)
        names = enc.get_feature_names_out()
        nan_names = [n for n in names if "nan" in n.lower()]
        self.assertEqual(len(nan_names), 0)

    def test_dummy_na_on_clean_data(self):
        """dummy_na=True on data with no NaNs must not crash."""
        enc = DummyCodeEncoder(sep=",", dummy_na=True, sparse_output=False)
        try:
            enc.fit_transform(self.df_no_nan)
        except Exception as e:
            self.fail(f"dummy_na=True on clean data raised: {e}")


# ===========================================================================
# DummyCodeEncoder — handle_unknown='warn'
# ===========================================================================

class TestDCEWarnUnknown(unittest.TestCase):
    """handle_unknown='warn' must issue a UserWarning and still encode."""

    def setUp(self):
        self.df_train = pd.DataFrame({"tags": ["a,b", "b,c"]})
        self.df_test = pd.DataFrame({"tags": ["a,x", "b"]})

    def test_warn_issues_user_warning(self):
        enc = DummyCodeEncoder(sep=",", handle_unknown="warn", sparse_output=False)
        enc.fit(self.df_train)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enc.transform(self.df_test)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0)

    def test_warn_unknown_still_encodes_known(self):
        """Known categories must still be encoded even when a warning fires."""
        enc = DummyCodeEncoder(sep=",", handle_unknown="warn", sparse_output=False)
        enc.fit(self.df_train)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = enc.transform(self.df_test)
        # row 0: 'a' is known (active), 'x' unknown (zero)
        # 'a' maps to column index 0 in categories_[0]
        a_idx = list(enc.categories_[0]).index("a")
        self.assertEqual(result[0, a_idx], 1.0)

    def test_warn_unknown_encoded_as_zeros(self):
        """Unknown tokens must be encoded as all zeros, not as errors."""
        enc = DummyCodeEncoder(sep=",", handle_unknown="warn", sparse_output=False)
        enc.fit(self.df_train)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = enc.transform(self.df_test)
        # Row sum for 'a,x': only 'a' matches → sum should be 1
        self.assertEqual(result[0].sum(), 1.0)

    def test_ignore_does_not_warn(self):
        """handle_unknown='ignore' must produce NO warnings."""
        enc = DummyCodeEncoder(sep=",", handle_unknown="ignore", sparse_output=False)
        enc.fit(self.df_train)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enc.transform(self.df_test)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 0,
                         f"handle_unknown='ignore' must not warn; got: {user_warnings}")

    def test_nan_ignored_silently(self):
        """NaN values in transform data with handle_unknown='ignore' must not warn."""
        df_test_nan = pd.DataFrame({"tags": [None]})
        enc = DummyCodeEncoder(sep=",", handle_unknown="ignore", sparse_output=False)
        enc.fit(self.df_train)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enc.transform(df_test_nan)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(user_warnings), 0,
                         "NaN with handle_unknown='ignore' must not issue a UserWarning")


# ===========================================================================
# DummyCodeEncoder — drop + sparse combined
# ===========================================================================

class TestDCEDropSparse(unittest.TestCase):
    """drop and sparse_output=True must work together without shape mismatch."""

    def setUp(self):
        self.df = pd.DataFrame({"tags": ["a,b", "b,c", "a,c"]})

    def test_drop_first_sparse_shape(self):
        enc = DummyCodeEncoder(sep=",", drop="first", sparse_output=True)
        result = enc.fit_transform(self.df)
        self.assertTrue(sp.issparse(result))
        # 3 categories - 1 dropped = 2 columns
        self.assertEqual(result.shape[1], 2)

    def test_drop_if_binary_sparse(self):
        df = pd.DataFrame({"flag": ["yes", "no", "yes"]})
        enc = DummyCodeEncoder(sep=",", drop="if_binary", sparse_output=True)
        result = enc.fit_transform(df)
        self.assertTrue(sp.issparse(result))
        self.assertEqual(result.shape[1], 1)

    def test_drop_none_sparse_shape(self):
        enc = DummyCodeEncoder(sep=",", drop=None, sparse_output=True)
        result = enc.fit_transform(self.df)
        self.assertEqual(result.shape[1], 3)

    def test_drop_first_sparse_feature_names_consistent(self):
        """Feature names must match actual sparse column count."""
        enc = DummyCodeEncoder(sep=",", drop="first", sparse_output=True)
        result = enc.fit_transform(self.df)
        names = enc.get_feature_names_out()
        self.assertEqual(len(names), result.shape[1])

    def test_drop_first_sparse_values_binary(self):
        enc = DummyCodeEncoder(sep=",", drop="first", sparse_output=True)
        result = enc.fit_transform(self.df)
        unique_vals = np.unique(result.toarray())
        self.assertTrue(set(unique_vals.tolist()).issubset({0.0, 1.0}))


# ===========================================================================
# DummyCodeEncoder — inverse_transform with drop
# ===========================================================================

class TestDCEInverseWithDrop(unittest.TestCase):
    """inverse_transform on drop-reduced output must not crash."""

    def setUp(self):
        self.df = pd.DataFrame({"tags": ["a,b", "b,c", "a"]})

    def test_inverse_drop_first_no_crash(self):
        """inverse_transform on drop=first output must not raise."""
        enc = DummyCodeEncoder(sep=",", drop="first", sparse_output=False)
        X = enc.fit_transform(self.df)
        try:
            inv = enc.inverse_transform(X)
            self.assertIsInstance(inv, list)
        except Exception as e:
            self.fail(f"inverse_transform with drop='first' raised: {e}")

    def test_inverse_drop_first_returns_list_of_tuples(self):
        enc = DummyCodeEncoder(sep=",", drop="first", sparse_output=False)
        X = enc.fit_transform(self.df)
        inv = enc.inverse_transform(X)
        for row in inv:
            self.assertIsInstance(row, tuple)

    def test_inverse_drop_first_sparse_no_crash(self):
        enc = DummyCodeEncoder(sep=",", drop="first", sparse_output=True)
        X = enc.fit_transform(self.df)
        try:
            inv = enc.inverse_transform(X)
            self.assertIsInstance(inv, list)
        except Exception as e:
            self.fail(f"Sparse inverse with drop='first' raised: {e}")


# ===========================================================================
# DummyCodeEncoder — fit_transform == fit + transform
# ===========================================================================

class TestDCEFitTransformConsistency(unittest.TestCase):
    """fit_transform output must always equal fit().transform()."""

    def _assert_equal(self, df, **kwargs):
        enc_ft = DummyCodeEncoder(**kwargs)
        enc_fb = DummyCodeEncoder(**kwargs)
        r_ft = enc_ft.fit_transform(df)
        enc_fb.fit(df)
        r_fb = enc_fb.transform(df)
        if sp.issparse(r_ft):
            np.testing.assert_array_almost_equal(r_ft.toarray(), r_fb.toarray())
        else:
            np.testing.assert_array_almost_equal(r_ft, r_fb)

    def test_dense_single_col(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c", "a"]})
        self._assert_equal(df, sep=",", sparse_output=False)

    def test_sparse_single_col(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c", "a"]})
        self._assert_equal(df, sep=",", sparse_output=True)

    def test_multi_col(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c"], "color": ["red", "blue"]})
        self._assert_equal(df, sep=",", sparse_output=False)

    def test_with_drop_first(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c", "a,c"]})
        self._assert_equal(df, sep=",", drop="first", sparse_output=False)

    def test_with_regex_sep(self):
        df = pd.DataFrame({"tags": ["a,b", "b;c", "a|c"]})
        self._assert_equal(df, sep=r"\s*[,;|]\s*", regex=True, sparse_output=False)


# ===========================================================================
# DummyCodeEncoder — set_output pandas
# ===========================================================================

class TestDCESetOutput(unittest.TestCase):
    """set_output(transform='pandas') must return a named DataFrame."""

    def test_set_output_pandas_returns_dataframe(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c"]})
        pipe = Pipeline([
            ("enc", DummyCodeEncoder(sep=",", sparse_output=False))
        ]).set_output(transform="pandas")
        result = pipe.fit_transform(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_set_output_pandas_columns_are_strings(self):
        df = pd.DataFrame({"tags": ["a,b", "b,c"]})
        pipe = Pipeline([
            ("enc", DummyCodeEncoder(sep=",", sparse_output=False))
        ]).set_output(transform="pandas")
        result = pipe.fit_transform(df)
        for col in result.columns:
            self.assertIsInstance(col, str)

    def test_set_output_sparse_with_pandas_raises(self):
        """sparse_output=True + set_output(pandas) must raise ValueError."""
        df = pd.DataFrame({"tags": ["a,b", "b,c"]})
        pipe = Pipeline([
            ("enc", DummyCodeEncoder(sep=",", sparse_output=True))
        ]).set_output(transform="pandas")
        with self.assertRaises(ValueError):
            pipe.fit_transform(df)

    def test_set_output_default_returns_ndarray(self):
        """set_output='default' with sparse_output=False must return ndarray."""
        df = pd.DataFrame({"tags": ["a,b", "b,c"]})
        pipe = Pipeline([
            ("enc", DummyCodeEncoder(sep=",", sparse_output=False))
        ]).set_output(transform="default")
        result = pipe.fit_transform(df)
        self.assertIsInstance(result, np.ndarray)


# ===========================================================================
# DummyCodeEncoder — feature_names_in_
# ===========================================================================

class TestDCEFeatureNamesIn(unittest.TestCase):
    """feature_names_in_ must be set after fit on a named DataFrame."""

    def test_feature_names_in_set(self):
        df = pd.DataFrame({"tags": ["a,b", "b"], "color": ["red", "blue"]})
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        enc.fit(df)
        self.assertTrue(hasattr(enc, "feature_names_in_"))
        np.testing.assert_array_equal(enc.feature_names_in_, ["tags", "color"])

    def test_feature_names_in_count_matches_n_features_in(self):
        df = pd.DataFrame({"tags": ["a,b", "b"], "color": ["red", "blue"],
                           "size": ["S", "M"]})
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        enc.fit(df)
        self.assertEqual(len(enc.feature_names_in_), enc.n_features_in_)

    def test_feature_names_in_not_set_for_array_input(self):
        """Fitting on a plain ndarray must NOT set feature_names_in_."""
        arr = np.array([["a,b", "red"], ["b,c", "blue"]], dtype=object)
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        enc.fit(arr)
        self.assertFalse(hasattr(enc, "feature_names_in_"))


# ===========================================================================
# DummyCodeEncoder — inverse_transform multi-label rows
# ===========================================================================

class TestDCEInverseMultiLabel(unittest.TestCase):
    """Rows with multiple active labels must be decoded to comma-joined strings."""

    def setUp(self):
        self.df = pd.DataFrame({
            "tags": ["a,b,c", "a"],
            "color": ["red", "blue"],
        })

    def test_multi_label_row_decoded_as_joined_string(self):
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        X = enc.fit_transform(self.df)
        inv = enc.inverse_transform(X)
        # Row 0: 'a,b,c' → tuple entry for tags must contain all three labels
        tags_val = inv[0][0]
        self.assertIn("a", tags_val)
        self.assertIn("b", tags_val)
        self.assertIn("c", tags_val)

    def test_single_label_row_decoded_as_scalar(self):
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        X = enc.fit_transform(self.df)
        inv = enc.inverse_transform(X)
        # Row 1: 'a' only → tuple entry for tags should be just 'a' (no comma)
        tags_val = inv[1][0]
        self.assertNotIn(",", str(tags_val))

    def test_sparse_multi_label_roundtrip(self):
        enc = DummyCodeEncoder(sep=",", sparse_output=True)
        X = enc.fit_transform(self.df)
        inv = enc.inverse_transform(X)
        self.assertEqual(len(inv), 2)
        tags_val_0 = inv[0][0]
        self.assertIn("a", tags_val_0)

    def test_inverse_length_equals_n_samples(self):
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        X = enc.fit_transform(self.df)
        inv = enc.inverse_transform(X)
        self.assertEqual(len(inv), len(self.df))

    def test_inverse_tuple_length_equals_n_features(self):
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        X = enc.fit_transform(self.df)
        inv = enc.inverse_transform(X)
        for row in inv:
            self.assertEqual(len(row), self.df.shape[1])


# ===========================================================================
# Cross-encoder consistency — GetDummies vs DummyCodeEncoder
# ===========================================================================

class TestCrossEncoderConsistency(unittest.TestCase):
    """GetDummies and DummyCodeEncoder must agree on basic single-column encoding."""

    def test_same_categories_discovered(self):
        """Both encoders must discover the same set of unique labels."""
        df = pd.DataFrame({"tags": ["a,b", "b,c", "a,c"]})

        gd = GetDummies(columns="tags", sep=",")
        gd.fit(df)
        gd_labels = sorted({
            c.split("_", 1)[1]
            for c in gd.categories_["tags"]
        })

        dce = DummyCodeEncoder(sep=",", sparse_output=False)
        dce.fit(df)
        dce_labels = sorted(dce.categories_[0].tolist())

        self.assertEqual(gd_labels, dce_labels)

    def test_same_row_sums(self):
        """Both encoders must produce identical row sums for single-label data."""
        df = pd.DataFrame({"tags": ["a", "b", "c"]})

        gd = GetDummies(columns="tags", sep=",")
        gd_result = gd.fit_transform(df)
        gd_sums = gd_result[[c for c in gd_result.columns if c.startswith("ta_")]].sum(axis=1).values

        dce = DummyCodeEncoder(sep=",", sparse_output=False)
        dce_result = dce.fit_transform(df)
        dce_sums = dce_result.sum(axis=1)

        np.testing.assert_array_equal(gd_sums, dce_sums)


if __name__ == "__main__":
    # _install_stub()
    unittest.main(verbosity=2)
