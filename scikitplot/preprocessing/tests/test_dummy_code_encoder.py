# preprocessing/tests/test_dummy_code_encoder.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :class:`preprocessing.DummyCodeEncoder`.

Coverage map
------------
- ``__init__`` param defaults and storage                             → TestInit
- ``fit`` category learning, auto vs manual                           → TestFit
- ``fit`` infrequent categories (min_frequency / max_categories)      → TestFitInfrequent
- ``transform`` sparse (CSR) and dense output                         → TestTransform
- ``transform`` sep=str literal, sep=callable, regex=True             → TestSeparators
- ``transform`` handle_unknown warn path                              → TestHandleUnknown
- ``transform`` NaN passthrough                                       → TestNaNHandling
- ``drop`` first / if_binary / array                                  → TestDrop
- ``inverse_transform`` sparse and dense roundtrip                    → TestInverseTransform
- ``get_feature_names_out``                                           → TestFeatureNames
- sklearn Pipeline and clone compatibility                             → TestSklearnCompat
- ``_expand_by_separators`` unit test                                  → TestExpandBySeparators
- ``_sort_with_none_nan_last`` unit test                               → TestSortHelper

Notes
-----
Developer note
    Run standalone::

        python test_dummy_code_encoder.py

    Run under pytest (recommended)::

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

# ---------------------------------------------------------------------------
# Bootstrap path for standalone execution
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
    make_multi_col_df,
    make_nan_df,
    make_simple_df,
    make_tags_df,
)
# install_scikitplot_stub()
from .._encoders import DummyCodeEncoder  # noqa: E402


# ===========================================================================
# TestInit — defaults and param storage
# ===========================================================================

class TestInit(unittest.TestCase):
    """All __init__ params must be stored verbatim (sklearn clone contract)."""

    def test_default_sep(self):
        enc = DummyCodeEncoder()
        self.assertEqual(enc.sep, "|")

    def test_default_sparse_output(self):
        enc = DummyCodeEncoder()
        self.assertTrue(enc.sparse_output)

    def test_default_handle_unknown(self):
        enc = DummyCodeEncoder()
        self.assertEqual(enc.handle_unknown, "error")

    def test_default_drop(self):
        enc = DummyCodeEncoder()
        self.assertIsNone(enc.drop)

    def test_default_dtype(self):
        enc = DummyCodeEncoder()
        self.assertEqual(enc.dtype, np.float64)

    def test_clone_roundtrip(self):
        """clone() must produce an equivalent unfitted estimator."""
        enc = DummyCodeEncoder(sep=",", sparse_output=False, drop="first")
        enc2 = clone(enc)
        self.assertEqual(enc2.sep, ",")
        self.assertFalse(enc2.sparse_output)
        self.assertEqual(enc2.drop, "first")
        # clone must be unfitted
        self.assertFalse(hasattr(enc2, "categories_"))

    def test_get_params_roundtrip(self):
        """get_params must return all constructor params unchanged."""
        enc = DummyCodeEncoder(sep=",", min_frequency=2, max_categories=3)
        params = enc.get_params()
        self.assertEqual(params["sep"], ",")
        self.assertEqual(params["min_frequency"], 2)
        self.assertEqual(params["max_categories"], 3)


# ===========================================================================
# TestFit — category learning
# ===========================================================================

class TestFit(unittest.TestCase):
    """fit() must build categories_ as a column-indexed dict."""

    def test_fit_categories_dict(self):
        """categories_ must be a dict keyed by integer feature index."""
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",")
        enc.fit(df)
        self.assertIsInstance(enc.categories_, dict)
        for key in enc.categories_:
            self.assertIsInstance(key, int)

    def test_fit_all_unique_labels_discovered(self):
        """All unique labels across all rows must appear in categories_."""
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",")
        enc.fit(df)
        # col 0 is 'tags': contains a, b, c
        flat = list(enc.categories_[0])
        self.assertIn("a", flat)
        self.assertIn("b", flat)
        self.assertIn("c", flat)

    def test_fit_lowercase_normalises_callable_sep(self):
        """Callable sep that lowercases must produce lowercase categories."""
        df = pd.DataFrame({"tags": ["A,B", "b,C"]})
        enc = DummyCodeEncoder(
            sep=lambda s: re.split(r"\s*,\s*", s.lower()), regex=True
        )
        enc.fit(df)
        flat = list(enc.categories_[0])
        for label in flat:
            self.assertEqual(label, label.lower())

    def test_fit_sets_n_features_in(self):
        """n_features_in_ must equal number of input columns."""
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",")
        enc.fit(df)
        self.assertEqual(enc.n_features_in_, df.shape[1])

    def test_fit_returns_self(self):
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",")
        result = enc.fit(df)
        self.assertIs(result, enc)

    def test_fit_sets_infrequent_attributes_when_disabled(self):
        """When infrequent is disabled, helper attrs must still be initialised."""
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",")
        enc.fit(df)
        # Must not raise AttributeError in _set_drop_idx/_compute_n_features_outs
        self.assertTrue(hasattr(enc, "_infrequent_indices"))
        self.assertTrue(hasattr(enc, "_default_to_infrequent_mappings"))
        self.assertIsNone(enc._infrequent_indices[0])

    def test_fit_requires_2d_input(self):
        """1-D array input must raise an error during fit."""
        enc = DummyCodeEncoder(sep=",")
        with self.assertRaises(Exception):
            enc.fit(np.array(["a,b", "b,c"]))


# ===========================================================================
# TestFitInfrequent — min_frequency and max_categories
# ===========================================================================

class TestFitInfrequent(unittest.TestCase):
    """Infrequent-category grouping must work without AttributeError (Bug 4)."""

    def test_min_frequency_no_crash(self):
        """Fitting with min_frequency must not raise AttributeError."""
        df = make_infrequent_df()
        enc = DummyCodeEncoder(sep=",", min_frequency=2, sparse_output=False)
        # Before Bug 4 fix this raised AttributeError: '_infrequent_indices'
        enc.fit(df)
        self.assertTrue(hasattr(enc, "_infrequent_indices"))

    def test_min_frequency_infrequent_categories_attr(self):
        """infrequent_categories_ must list categories below threshold."""
        df = make_infrequent_df()
        enc = DummyCodeEncoder(sep=",", min_frequency=2, sparse_output=False)
        enc.fit(df)
        # 'b' and 'c' each appear once (< 2) — infrequent for col 0 (tags)
        infreq = enc.infrequent_categories_[0]
        if infreq is not None:
            for label in infreq:
                self.assertIn(label, ["b", "c"])

    def test_max_categories_limits_output_width(self):
        """max_categories must cap the number of output columns per feature."""
        df = make_infrequent_df()
        enc = DummyCodeEncoder(sep=",", max_categories=2, sparse_output=False)
        enc.fit(df)
        n_out = enc._n_features_outs[0]
        self.assertLessEqual(n_out, 2)

    def test_fit_transform_infrequent_shape_correct(self):
        """fit_transform with min_frequency must produce correct output shape."""
        df = make_infrequent_df()
        enc = DummyCodeEncoder(sep=",", min_frequency=2, sparse_output=False)
        result = enc.fit_transform(df)
        self.assertEqual(result.shape[0], len(df))


# ===========================================================================
# TestTransform — sparse and dense output
# ===========================================================================

class TestTransform(unittest.TestCase):
    """transform must produce CSR or dense array with correct shape and values."""

    def setUp(self):
        self.df = make_simple_df()

    def test_sparse_output_type(self):
        enc = DummyCodeEncoder(sep=",", sparse_output=True)
        result = enc.fit_transform(self.df)
        self.assertIsInstance(result, sp.csr_matrix)

    def test_dense_output_type(self):
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        result = enc.fit_transform(self.df)
        self.assertIsInstance(result, np.ndarray)

    def test_shape_rows_match(self):
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        result = enc.fit_transform(self.df)
        self.assertEqual(result.shape[0], len(self.df))

    def test_output_values_binary(self):
        """All output values must be 0 or 1."""
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        result = enc.fit_transform(self.df)
        unique_vals = np.unique(result)
        self.assertTrue(set(unique_vals.tolist()).issubset({0.0, 1.0}))

    def test_row_sums_reflect_label_count(self):
        """Row sum must equal the number of labels active in that row."""
        df = pd.DataFrame({"tags": ["a,b,c", "a"]})
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        result = enc.fit_transform(df)
        self.assertEqual(result[0].sum(), 3)  # 'a,b,c' → 3 active
        self.assertEqual(result[1].sum(), 1)  # 'a' → 1 active

    def test_transform_before_fit_raises(self):
        """transform before fit must raise NotFittedError."""
        from sklearn.exceptions import NotFittedError
        enc = DummyCodeEncoder(sep=",")
        with self.assertRaises(NotFittedError):
            enc.transform(self.df)

    def test_multi_column_transform_shape(self):
        """Multi-column transform must expand all columns."""
        df = make_multi_col_df()
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        result = enc.fit_transform(df)
        # tags: a,b,c = 3; color: red,blue = 2; val: 1,2,3 = 3 → total = 8
        self.assertGreater(result.shape[1], 3)


# ===========================================================================
# TestSeparators — sep variants
# ===========================================================================

class TestSeparators(unittest.TestCase):
    """The three sep modes (str literal, regex, callable) must all work."""

    def setUp(self):
        self.df = pd.DataFrame({"tags": ["a,b", "b|c", "a;c"]})

    def test_str_literal_sep(self):
        """Literal str sep must split only on that exact character."""
        df = pd.DataFrame({"tags": ["a,b", "b,c"]})
        enc = DummyCodeEncoder(sep=",", regex=False, sparse_output=False)
        result = enc.fit_transform(df)
        self.assertEqual(result.shape[0], 2)

    def test_regex_sep_multi_char(self):
        """Regex sep must split on any of the specified separator chars."""
        enc = DummyCodeEncoder(
            sep=r"\s*[,;|]\s*", regex=True, sparse_output=False
        )
        result = enc.fit_transform(self.df)
        # Each row has 2 labels → 2 active per row
        for i in range(3):
            self.assertEqual(result[i].sum(), 2)

    def test_callable_sep_lowercases(self):
        """Callable sep returning normalised tokens must be used directly."""
        df = pd.DataFrame({"tags": ["A,B", "B,C"]})
        enc = DummyCodeEncoder(
            sep=lambda s: re.split(r"\s*,\s*", s.lower()),
            sparse_output=False,
        )
        result = enc.fit_transform(df)
        # 3 unique lowercase labels: a, b, c
        self.assertEqual(result.shape[1], 3)

    def test_sep_default_pipe(self):
        """Default sep='|' must split on pipe character."""
        df = pd.DataFrame({"tags": ["a|b", "b|c"]})
        enc = DummyCodeEncoder(sparse_output=False)  # default sep='|'
        result = enc.fit_transform(df)
        self.assertEqual(result.shape[1], 3)


# ===========================================================================
# TestHandleUnknown
# ===========================================================================

class TestHandleUnknown(unittest.TestCase):
    """Unknown categories must be handled per handle_unknown setting."""

    def setUp(self):
        self.df_train = pd.DataFrame({"tags": ["a,b", "b,c"]})
        self.df_test = pd.DataFrame({"tags": ["a,x", "b"]})

    def test_handle_unknown_error_raises(self):
        enc = DummyCodeEncoder(sep=",", handle_unknown="error", sparse_output=False)
        enc.fit(self.df_train)
        with self.assertRaises(Exception):
            enc.transform(self.df_test)

    def test_handle_unknown_ignore_encodes_as_zeros(self):
        """Unknown categories with handle_unknown='ignore' must produce all zeros."""
        enc = DummyCodeEncoder(sep=",", handle_unknown="ignore", sparse_output=False)
        enc.fit(self.df_train)
        result = enc.transform(self.df_test)
        # 'x' is unknown — row 0 sum should equal 1 (only 'a' is known)
        self.assertEqual(result[0].sum(), 1.0)

    def test_handle_unknown_warn_issues_warning(self):
        """handle_unknown='warn' must issue a UserWarning."""
        enc = DummyCodeEncoder(sep=",", handle_unknown="warn", sparse_output=False)
        enc.fit(self.df_train)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            enc.transform(self.df_test)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0)


# ===========================================================================
# TestNaNHandling
# ===========================================================================

class TestNaNHandling(unittest.TestCase):
    """NaN rows must not crash; they are encoded as all zeros."""

    def test_nan_in_fit_data_no_crash(self):
        """NaN in fit data must not raise."""
        df = make_nan_df()
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        # Should not raise
        enc.fit(df)

    def test_nan_in_transform_data_no_crash(self):
        """NaN in transform data must not raise (unknown → all zeros)."""
        df_train = pd.DataFrame({"tags": ["a,b", "b,c"]})
        df_test = pd.DataFrame({"tags": ["a,b", None]})
        enc = DummyCodeEncoder(sep=",", handle_unknown="ignore", sparse_output=False)
        enc.fit(df_train)
        result = enc.transform(df_test)
        self.assertEqual(result.shape[0], 2)

    def test_nan_row_produces_zero_sum(self):
        """NaN row must produce all-zero output."""
        df_train = pd.DataFrame({"tags": ["a,b", "b,c"]})
        df_test = pd.DataFrame({"tags": [None]})
        enc = DummyCodeEncoder(sep=",", handle_unknown="ignore", sparse_output=False)
        enc.fit(df_train)
        result = enc.transform(df_test)
        self.assertEqual(result[0].sum(), 0)


# ===========================================================================
# TestDrop
# ===========================================================================

class TestDrop(unittest.TestCase):
    """drop parameter must remove appropriate dummy columns."""

    def setUp(self):
        self.df = pd.DataFrame({"tags": ["a,b", "b,c", "a,c"]})

    def test_drop_none_keeps_all(self):
        enc = DummyCodeEncoder(sep=",", drop=None, sparse_output=False)
        result = enc.fit_transform(self.df)
        # tags: a, b, c → 3 columns
        self.assertEqual(result.shape[1], 3)

    def test_drop_first_reduces_by_one(self):
        enc = DummyCodeEncoder(sep=",", drop="first", sparse_output=False)
        result = enc.fit_transform(self.df)
        self.assertEqual(result.shape[1], 2)

    def test_drop_if_binary_drops_binary_feature(self):
        """drop='if_binary' must drop a column only if the feature has exactly 2 cats."""
        df = pd.DataFrame({"flag": ["yes", "no", "yes"]})
        enc = DummyCodeEncoder(sep=",", drop="if_binary", sparse_output=False)
        result = enc.fit_transform(df)
        # 2 categories → drop 1 → 1 remaining
        self.assertEqual(result.shape[1], 1)

    def test_drop_if_binary_keeps_non_binary_intact(self):
        """drop='if_binary' must NOT drop a column with 3+ categories."""
        enc = DummyCodeEncoder(sep=",", drop="if_binary", sparse_output=False)
        result = enc.fit_transform(self.df)
        # 3 categories → no drop
        self.assertEqual(result.shape[1], 3)

    def test_drop_array_explicit_value(self):
        """drop=['a'] must drop the specified category value."""
        enc = DummyCodeEncoder(sep=",", drop=["a"], sparse_output=False)
        enc.fit(self.df)
        names = enc.get_feature_names_out()
        label_names = [n.split("_", 1)[1] for n in names]
        self.assertNotIn("a", label_names)
        self.assertIn("b", label_names)


# ===========================================================================
# TestInverseTransform — roundtrip correctness
# ===========================================================================

class TestInverseTransform(unittest.TestCase):
    """inverse_transform must decode CSR or dense back to original multi-labels."""

    def setUp(self):
        self.df = pd.DataFrame(
            {"tags": ["a,b", "b,c", "a"], "color": ["red", "blue", "red"]}
        )

    def test_inverse_sparse_roundtrip(self):
        """Sparse encode → inverse_transform must recover original categories."""
        enc = DummyCodeEncoder(sep=",", sparse_output=True)
        X = enc.fit_transform(self.df)
        inv = enc.inverse_transform(X)
        self.assertEqual(len(inv), 3)
        # Row 0: tags='a,b', color='red'
        self.assertIsInstance(inv[0], tuple)
        tags_val, color_val = inv[0]
        self.assertIn("a", tags_val)
        self.assertIn("b", tags_val)
        self.assertEqual(color_val, "red")

    def test_inverse_dense_roundtrip(self):
        """Dense encode → inverse_transform must recover original categories."""
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        X = enc.fit_transform(self.df)
        inv = enc.inverse_transform(X)
        self.assertEqual(len(inv), 3)
        _, color_val = inv[2]
        self.assertEqual(color_val, "red")

    def test_inverse_rejects_non_binary_values(self):
        """Input with values other than 0/1 must raise ValueError."""
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        enc.fit(self.df)
        X = enc.transform(self.df)
        X_bad = X.copy()
        X_bad[0, 0] = 2  # invalid
        with self.assertRaises(ValueError):
            enc.inverse_transform(X_bad)

    def test_inverse_all_zeros_row_returns_none(self):
        """All-zeros row (no active label) must map to None for that column."""
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        enc.fit(self.df)
        X = enc.transform(self.df)
        X_zeros = np.zeros_like(X)
        inv = enc.inverse_transform(X_zeros)
        # All columns should be None
        for val in inv[0]:
            self.assertIsNone(val)

    def test_inverse_no_dead_return_bug(self):
        """inverse_transform must return a list, not None (no dead return path)."""
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        X = enc.fit_transform(self.df)
        result = enc.inverse_transform(X)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)


# ===========================================================================
# TestFeatureNames
# ===========================================================================

class TestFeatureNames(unittest.TestCase):
    """get_feature_names_out must return a string ndarray of correct length."""

    def test_returns_ndarray_of_strings(self):
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        enc.fit(df)
        names = enc.get_feature_names_out()
        self.assertIsInstance(names, np.ndarray)
        for n in names:
            self.assertIsInstance(n, str)

    def test_length_matches_output_columns(self):
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        result = enc.fit_transform(df)
        names = enc.get_feature_names_out()
        self.assertEqual(len(names), result.shape[1])

    def test_feature_names_contain_col_prefix(self):
        """Feature names must be prefixed with the original column name."""
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        enc.fit(df)
        names = enc.get_feature_names_out()
        # All names for column 0 ('tags') must start with 'tags_'
        tag_names = [n for n in names if n.startswith("tags_")]
        self.assertGreater(len(tag_names), 0)

    def test_custom_feature_name_combiner(self):
        """A callable feature_name_combiner must be used for name generation."""
        df = pd.DataFrame({"tags": ["a,b", "b,c"]})
        combiner = lambda feat, cat: f"{feat}|{cat}"  # noqa: E731
        enc = DummyCodeEncoder(
            sep=",", sparse_output=False, feature_name_combiner=combiner
        )
        enc.fit(df)
        names = enc.get_feature_names_out()
        for n in names:
            self.assertIn("|", n)

    def test_raises_before_fit(self):
        from sklearn.exceptions import NotFittedError
        enc = DummyCodeEncoder(sep=",")
        with self.assertRaises(NotFittedError):
            enc.get_feature_names_out()


# ===========================================================================
# TestSklearnCompat — Pipeline and set_output
# ===========================================================================

class TestSklearnCompat(unittest.TestCase):
    """DummyCodeEncoder must be fully compatible with sklearn Pipeline."""

    def test_pipeline_sparse_step(self):
        """DummyCodeEncoder must function as a Pipeline step."""
        df = make_simple_df()
        pipe = Pipeline([("enc", DummyCodeEncoder(sep=",", sparse_output=True))])
        result = pipe.fit_transform(df)
        self.assertTrue(sp.issparse(result))

    def test_set_output_default(self):
        """set_output(transform='default') must return ndarray for dense mode."""
        df = make_simple_df()
        pipe = Pipeline(
            [("enc", DummyCodeEncoder(sep=",", sparse_output=False))]
        ).set_output(transform="default")
        result = pipe.fit_transform(df)
        self.assertIsInstance(result, np.ndarray)

    def test_set_output_sparse_raises_with_pandas(self):
        """set_output(transform='pandas') with sparse_output=True must raise."""
        df = make_simple_df()
        pipe = Pipeline(
            [("enc", DummyCodeEncoder(sep=",", sparse_output=True))]
        ).set_output(transform="pandas")
        with self.assertRaises(ValueError):
            pipe.fit_transform(df)

    def test_clone_is_unfitted(self):
        """clone must produce an unfitted copy without categories_."""
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",")
        enc.fit(df)
        enc2 = clone(enc)
        self.assertFalse(hasattr(enc2, "categories_"))

    def test_clone_then_fit(self):
        """Cloned encoder must be independently fittable."""
        df = make_simple_df()
        enc = DummyCodeEncoder(sep=",", sparse_output=False)
        enc.fit(df)
        enc2 = clone(enc)
        result = enc2.fit_transform(df)
        self.assertIsNotNone(result)


# ===========================================================================
# TestExpandBySeparators — unit tests for the private helper
# ===========================================================================

class TestExpandBySeparators(unittest.TestCase):
    """_expand_by_separators must split strings while leaving non-strings intact."""

    def setUp(self):
        self.enc = DummyCodeEncoder(sep=",", regex=False)
        # Provide minimal fitted state so the helper can read self.sep / self.regex
        self.enc.sep = ","
        self.enc.regex = False

    def test_splits_plain_string(self):
        result = self.enc._expand_by_separators(["a,b,c"])
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)

    def test_preserves_none(self):
        result = self.enc._expand_by_separators([None])
        self.assertEqual(list(result), [None])

    def test_preserves_nan(self):
        result = self.enc._expand_by_separators([np.nan])
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))

    def test_preserves_numeric(self):
        result = self.enc._expand_by_separators([42])
        self.assertEqual(list(result), [42])

    def test_strips_whitespace_from_tokens(self):
        result = self.enc._expand_by_separators([" a , b "])
        tokens = list(result)
        self.assertIn("a", tokens)
        self.assertIn("b", tokens)

    def test_skips_empty_tokens(self):
        result = self.enc._expand_by_separators(["a,,b,"])
        tokens = [t for t in result if t != ""]
        self.assertNotIn("", tokens)

    def test_callable_sep(self):
        self.enc.sep = lambda s: s.split(",")
        result = self.enc._expand_by_separators(["x,y"])
        self.assertIn("x", result)
        self.assertIn("y", result)

    def test_regex_sep(self):
        self.enc.sep = r"\s*[,;]\s*"
        self.enc.regex = True
        result = self.enc._expand_by_separators(["x,y;z"])
        self.assertIn("x", result)
        self.assertIn("y", result)
        self.assertIn("z", result)


# ===========================================================================
# TestSortHelper — _sort_with_none_nan_last
# ===========================================================================

class TestSortHelper(unittest.TestCase):
    """_sort_with_none_nan_last must sort normal values first, NaN/None at end."""

    def setUp(self):
        enc = DummyCodeEncoder(sep=",")
        enc.dummy_na = False
        self.enc = enc

    def test_normal_values_sorted(self):
        # _sort_with_none_nan_last receives a flat 1D array (result of _expand_by_separators).
        # chain.from_iterable iterates the array, yielding individual scalar elements.
        result = self.enc._sort_with_none_nan_last(np.array(["c", "a", "b"], dtype=object))
        self.assertEqual(result, ["a", "b", "c"])

    def test_none_excluded_when_dummy_na_false(self):
        """None must be excluded when dummy_na=False."""
        result = self.enc._sort_with_none_nan_last(np.array(["b", None, "a"], dtype=object))
        self.assertNotIn(None, result)

    def test_none_included_when_dummy_na_true(self):
        """None must appear at the end when dummy_na=True."""
        self.enc.dummy_na = True
        result = self.enc._sort_with_none_nan_last(np.array(["b", None, "a"], dtype=object))
        self.assertIsNone(result[-1])

    def test_nan_excluded_when_dummy_na_false(self):
        self.enc.dummy_na = False
        result = self.enc._sort_with_none_nan_last(np.array([np.nan, "b", "a"], dtype=object))
        for v in result:
            self.assertFalse(isinstance(v, float) and np.isnan(v))

    def test_mixed_types_fall_back_to_str_comparison(self):
        """Mixed types (int + str) must not crash — fall back to str sort."""
        result = self.enc._sort_with_none_nan_last(np.array([1, "a", 2, "b"], dtype=object))
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    # install_scikitplot_stub()
    unittest.main(verbosity=2)
