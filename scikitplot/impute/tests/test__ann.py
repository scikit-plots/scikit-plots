# scikitplot/impute/tests/test__ann.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit tests for :mod:`scikitplot.impute._ann`.

Covers
------
ANNImputer.__init__
  * default parameter values
  * custom parameters stored

ANNImputer.fit
  * returns self, sets n_features_in_, n_samples_fit_
  * temp_fill_vector_ shape and finiteness
  * _is_empty_feature flag for all-NaN columns
  * complete data (no missing)
  * add_indicator fitted / not fitted
  * all-NaN input → ValueError (regression fix)
  * partial all-NaN rows (some rows NaN, some valid) → succeeds
  * single all-NaN row with other valid rows → succeeds

ANNImputer — initial_strategy variants
  * mean, median, most_frequent, constant (0 and custom)
  * constant with non-numeric fill_value → ValueError
  * all-NaN column fallback to 0.0 for each strategy
  * no RuntimeWarning emitted for all-NaN columns (errstate fix)

ANNImputer — metric aliases
  * all annoy aliases fit without error
  * unknown metric → ValueError

ANNImputer.transform
  * output shape matches input
  * no NaNs in output
  * non-missing values unchanged
  * complete input unchanged
  * fit_transform consistent with fit+transform
  * transform before fit → exception
  * feature count mismatch → ValueError

ANNImputer — empty features (keep_empty_features)
  * dropped by default
  * kept and zeroed when keep_empty_features=True
  * fast path when no missing in valid cols

ANNImputer — add_indicator
  * columns appended, no NaNs

ANNImputer — weights
  * uniform, distance, callable

ANNImputer — index_access modes
  * public, private, external (file), auto-generated path, PathNamer, pathlib

ANNImputer — on_disk_build

ANNImputer._resolve_index_store_path
  * external/None auto-generates, str normalized, PathNamer cached
  * non-external None returns None, non-external str normalized
  * invalid type → TypeError

ANNImputer._compute_neighbor_weights
  * uniform (string and None), distance normalized, zero-distance inf,
    callable, zero-sum fallback

ANNImputer._impute_from_neighbors
  * single missing imputed, complete row unchanged
  * all-neighbor-NaN falls back to fill_vec
  * empty feature skipped
  * zero-weight-sum falls back to mean

ANNImputer._process_single_row_annoy
  * complete row returned unchanged
  * missing row imputed, returns correct index

ANNImputer.get_feature_names_out
  * length matches n_features_in_, custom names, empty col excluded,
    indicator names appended

VoyagerKNNImputer / AnnoyKNNImputer
  * backend forced, bad backend raises, alias identity

Integration
  * larger dataset, single feature, idempotent on complete data,
    clone, get/set_params, delete external index, n_jobs
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from scikitplot.impute._ann import ANNImputer, AnnoyKNNImputer, VoyagerKNNImputer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_X4 = np.array(
    [[1.0, 2.0, np.nan],
     [3.0, 4.0, 3.0],
     [np.nan, 6.0, 5.0],
     [8.0, 8.0, 7.0]],
    dtype=float,
)

_X_COMPLETE = np.array(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]],
    dtype=float,
)


def _make_imp(**kwargs):
    defaults = dict(
        backend="annoy",
        index_access="public",
        n_trees=5,
        n_neighbors=2,
        metric="euclidean",
        random_state=42,
    )
    defaults.update(kwargs)
    return ANNImputer(**defaults)


# ===========================================================================
# __init__ / parameter defaults
# ===========================================================================

class TestANNImputerInit(unittest.TestCase):

    def test_default_params(self):
        imp = ANNImputer()
        self.assertEqual(imp.backend, "annoy")
        self.assertEqual(imp.index_access, "external")
        self.assertEqual(imp.n_trees, -1)
        self.assertEqual(imp.search_k, -1)
        self.assertEqual(imp.n_neighbors, 5)
        self.assertEqual(imp.weights, "uniform")
        self.assertEqual(imp.metric, "angular")
        self.assertEqual(imp.initial_strategy, "mean")
        self.assertIsNone(imp.fill_value)
        self.assertTrue(imp.copy)
        self.assertFalse(imp.add_indicator)
        self.assertFalse(imp.keep_empty_features)
        self.assertIsNone(imp.n_jobs)
        self.assertIsNone(imp.random_state)

    def test_custom_params_stored(self):
        imp = ANNImputer(n_neighbors=10, weights="distance", metric="manhattan",
                         index_access="private", n_trees=20, random_state=42)
        self.assertEqual(imp.n_neighbors, 10)
        self.assertEqual(imp.weights, "distance")
        self.assertEqual(imp.metric, "manhattan")
        self.assertEqual(imp.index_access, "private")
        self.assertEqual(imp.n_trees, 20)
        self.assertEqual(imp.random_state, 42)


# ===========================================================================
# fit — basic
# ===========================================================================

class TestANNImputerFit(unittest.TestCase):

    def test_fit_returns_self(self):
        imp = _make_imp()
        self.assertIs(imp.fit(_X4), imp)

    def test_n_features_in_set(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertEqual(imp.n_features_in_, 3)

    def test_n_samples_fit_set(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertEqual(imp.n_samples_fit_, 4)

    def test_temp_fill_vector_shape(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertEqual(imp.temp_fill_vector_.shape, (3,))

    def test_temp_fill_vector_finite(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertTrue(np.all(np.isfinite(imp.temp_fill_vector_)))

    def test_is_empty_feature_no_empty_cols(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertFalse(np.any(imp._is_empty_feature))

    def test_is_empty_feature_flags_all_nan_col(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0],
                      [np.nan, 3.0], [np.nan, 4.0], [np.nan, 5.0]])
        imp = _make_imp()
        imp.fit(X)
        self.assertTrue(imp._is_empty_feature[0])
        self.assertFalse(imp._is_empty_feature[1])

    def test_fit_complete_data(self):
        imp = _make_imp()
        imp.fit(_X_COMPLETE)
        self.assertEqual(imp.n_features_in_, 3)

    def test_all_nan_rows_raises_value_error(self):
        """Regression: all-NaN input must raise ValueError, not RuntimeWarning."""
        X = np.full((3, 2), np.nan)
        imp = _make_imp()
        with self.assertRaises(ValueError):
            imp.fit(X)

    def test_all_nan_rows_raises_not_runtime_warning(self):
        """
        Ensure the RuntimeWarning from np.nanmean is suppressed before the
        early guard fires, so pytest warning-as-error mode passes.
        """
        X = np.full((4, 3), np.nan)
        imp = _make_imp()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(ValueError):
                imp.fit(X)

    def test_all_nan_value_error_message_informative(self):
        X = np.full((2, 2), np.nan)
        imp = _make_imp()
        with self.assertRaises(ValueError) as ctx:
            imp.fit(X)
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "missing" in msg or "nan" in msg or "observed" in msg,
            f"Error message not informative: {ctx.exception}",
        )

    def test_partial_all_nan_rows_succeeds(self):
        """Some rows are all-NaN but at least one row has observed values → OK."""
        X = np.array([[np.nan, np.nan], [1.0, 2.0], [3.0, 4.0]])
        imp = _make_imp()
        imp.fit(X)          # must not raise
        self.assertEqual(imp.n_samples_fit_, 3)

    def test_single_valid_row_among_all_nan_rows_succeeds(self):
        X = np.array([[np.nan, np.nan], [np.nan, np.nan], [1.0, 2.0]])
        imp = _make_imp()
        imp.fit(X)
        self.assertFalse(np.any(np.isnan(imp.temp_fill_vector_)))

    def test_indicator_fitted_when_add_indicator_true(self):
        X = np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0]])
        imp = _make_imp(add_indicator=True)
        imp.fit(X)
        self.assertIsNotNone(imp.indicator_)

    def test_indicator_none_when_add_indicator_false(self):
        imp = _make_imp(add_indicator=False)
        imp.fit(_X4)
        self.assertIsNone(imp.indicator_)


# ===========================================================================
# fit — initial_strategy variants
# ===========================================================================

class TestANNImputerInitialStrategy(unittest.TestCase):

    def _check_finite(self, strategy, fill_value=None):
        kw = dict(initial_strategy=strategy)
        if fill_value is not None:
            kw["fill_value"] = fill_value
        imp = _make_imp(**kw)
        imp.fit(_X4)
        self.assertTrue(np.all(np.isfinite(imp.temp_fill_vector_)))

    def test_mean(self):
        self._check_finite("mean")

    def test_median(self):
        self._check_finite("median")

    def test_most_frequent(self):
        self._check_finite("most_frequent")

    def test_constant_zero(self):
        self._check_finite("constant", fill_value=0.0)

    def test_constant_custom(self):
        self._check_finite("constant", fill_value=99.0)

    def test_constant_non_numeric_fill_raises(self):
        imp = _make_imp(initial_strategy="constant", fill_value="bad")
        with self.assertRaises(ValueError):
            imp.fit(_X4)

    def test_mean_all_nan_col_fills_zero(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0],
                      [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(initial_strategy="mean")
        imp.fit(X)
        self.assertAlmostEqual(imp.temp_fill_vector_[0], 0.0)

    def test_median_all_nan_col_fills_zero(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0],
                      [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(initial_strategy="median")
        imp.fit(X)
        self.assertAlmostEqual(imp.temp_fill_vector_[0], 0.0)

    def test_most_frequent_all_nan_col_fills_zero(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0],
                      [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(initial_strategy="most_frequent")
        imp.fit(X)
        self.assertAlmostEqual(imp.temp_fill_vector_[0], 0.0)

    def test_no_runtime_warning_emitted_for_all_nan_column_mean(self):
        """errstate fix: np.nanmean on all-NaN column must not emit RuntimeWarning."""
        X = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(initial_strategy="mean")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            imp.fit(X)  # must not raise

    def test_no_runtime_warning_emitted_for_all_nan_column_median(self):
        """errstate fix covers nanmedian as well."""
        X = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(initial_strategy="median")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            imp.fit(X)  # must not raise


# ===========================================================================
# fit — metric aliases
# ===========================================================================

class TestANNImputerMetricAliases(unittest.TestCase):

    _ALIASES = [
        "angular", "cosine",
        "euclidean", "l2", "lstsq",
        "manhattan", "l1", "cityblock", "taxicab",
        "dot", "@", ".", "dotproduct", "inner", "innerproduct",
        "hamming",
    ]

    def test_all_metric_aliases_fit_without_error(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        for alias in self._ALIASES:
            with self.subTest(metric=alias):
                imp = _make_imp(metric=alias)
                imp.fit(X)

    def test_unknown_metric_raises_value_error(self):
        imp = ANNImputer(metric="angular")
        imp.metric = "unknown_metric"
        with self.assertRaises(ValueError):
            imp._resolve_metric()


# ===========================================================================
# transform
# ===========================================================================

class TestANNImputerTransform(unittest.TestCase):

    def test_output_shape_matches_input(self):
        imp = _make_imp()
        out = imp.fit(_X4).transform(_X4)
        self.assertEqual(out.shape, (4, 3))

    def test_no_nans_in_output(self):
        imp = _make_imp()
        out = imp.fit(_X4).transform(_X4)
        self.assertFalse(np.any(np.isnan(out)))

    def test_non_missing_values_unchanged(self):
        imp = _make_imp()
        out = imp.fit(_X4).transform(_X4)
        assert_allclose(out[1, :], _X4[1, :])
        assert_allclose(out[3, :], _X4[3, :])

    def test_complete_input_returned_unchanged(self):
        imp = _make_imp()
        out = imp.fit(_X_COMPLETE).transform(_X_COMPLETE)
        assert_allclose(out, _X_COMPLETE)

    def test_fit_transform_consistent(self):
        out1 = _make_imp().fit(_X4).transform(_X4)
        out2 = _make_imp().fit_transform(_X4)
        assert_allclose(out1, out2, rtol=1e-5)

    def test_transform_before_fit_raises(self):
        imp = _make_imp()
        with self.assertRaises(Exception):
            imp.transform(_X4)

    def test_feature_count_mismatch_raises(self):
        imp = _make_imp()
        imp.fit(_X4)
        with self.assertRaises(ValueError):
            imp.transform(np.array([[1.0, 2.0]]))


# ===========================================================================
# Empty features
# ===========================================================================

class TestANNImputerEmptyFeatures(unittest.TestCase):

    def _X_with_empty_col(self):
        return np.array([[np.nan, 1.0], [np.nan, 2.0],
                         [np.nan, 3.0], [np.nan, 4.0]])

    def test_empty_col_dropped_by_default(self):
        imp = _make_imp(keep_empty_features=False)
        out = imp.fit_transform(self._X_with_empty_col())
        self.assertEqual(out.shape[1], 1)

    def test_empty_col_kept_and_zeroed(self):
        imp = _make_imp(keep_empty_features=True)
        out = imp.fit_transform(self._X_with_empty_col())
        self.assertEqual(out.shape[1], 2)
        assert_array_equal(out[:, 0], [0.0, 0.0, 0.0, 0.0])

    def test_fast_path_no_missing_in_valid_cols(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0],
                      [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(keep_empty_features=False)
        imp.fit(X)
        X_new = np.array([[np.nan, 5.0], [np.nan, 6.0]])
        out = imp.transform(X_new)
        self.assertEqual(out.shape[1], 1)
        assert_allclose(out[:, 0], [5.0, 6.0])


# ===========================================================================
# add_indicator
# ===========================================================================

class TestANNImputerAddIndicator(unittest.TestCase):

    def test_indicator_appended(self):
        X = np.array([[np.nan, 1.0, 2.0], [3.0, np.nan, 4.0],
                      [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        imp = _make_imp(add_indicator=True)
        out = imp.fit_transform(X)
        self.assertGreater(out.shape[1], 3)

    def test_output_no_nans(self):
        X = np.array([[np.nan, 1.0], [2.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        imp = _make_imp(add_indicator=True)
        out = imp.fit_transform(X)
        self.assertFalse(np.any(np.isnan(out)))


# ===========================================================================
# Weights
# ===========================================================================

class TestANNImputerWeights(unittest.TestCase):

    def test_uniform(self):
        out = _make_imp(weights="uniform").fit_transform(_X4)
        self.assertFalse(np.any(np.isnan(out)))

    def test_distance(self):
        out = _make_imp(weights="distance").fit_transform(_X4)
        self.assertFalse(np.any(np.isnan(out)))

    def test_callable(self):
        out = _make_imp(weights=lambda d: 1.0 / (1.0 + d)).fit_transform(_X4)
        self.assertFalse(np.any(np.isnan(out)))


# ===========================================================================
# index_access modes
# ===========================================================================

class TestANNImputerIndexAccess(unittest.TestCase):

    def test_public_train_index_accessible(self):
        imp = _make_imp(index_access="public")
        imp.fit(_X4)
        self.assertIsNotNone(imp.train_index_)

    def test_private_train_index_raises(self):
        imp = _make_imp(index_access="private")
        imp.fit(_X4)
        with self.assertRaises(AttributeError):
            _ = imp.train_index_

    def test_train_index_setter_always_raises(self):
        imp = _make_imp(index_access="public")
        imp.fit(_X4)
        with self.assertRaises(AttributeError):
            imp.train_index_ = None

    def test_external_index_saved_to_disk(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.annoy")
            imp = _make_imp(index_access="external", index_store_path=path)
            imp.fit(_X4)
            self.assertTrue(os.path.exists(imp.index_path_))
            out = imp.transform(_X4)
            self.assertFalse(np.any(np.isnan(out)))

    def test_external_auto_path_generated(self):
        imp = _make_imp(index_access="external", index_store_path=None)
        imp.fit(_X4)
        self.assertIsNotNone(imp.index_path_)
        self.assertTrue(os.path.exists(imp.index_path_))
        imp.delete_external_index()

    def test_external_pathnamer(self):
        from scikitplot.utils._path import PathNamer
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(prefix="test_", ext=".annoy", directory=td)
            imp = _make_imp(index_access="external", index_store_path=namer)
            imp.fit(_X4)
            self.assertTrue(os.path.exists(imp.index_path_))
            self.assertFalse(np.any(np.isnan(imp.transform(_X4))))

    def test_external_pathlib(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "idx.annoy"
            imp = _make_imp(index_access="external", index_store_path=path)
            imp.fit(_X4)
            self.assertFalse(np.any(np.isnan(imp.transform(_X4))))


# ===========================================================================
# on_disk_build
# ===========================================================================

class TestANNImputerOnDiskBuild(unittest.TestCase):

    def test_on_disk_build_external(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "disk.annoy")
            imp = _make_imp(index_access="external", index_store_path=path,
                            on_disk_build=True)
            out = imp.fit(_X4).transform(_X4)
            self.assertFalse(np.any(np.isnan(out)))

    def test_on_disk_build_public(self):
        imp = _make_imp(index_access="public", on_disk_build=True)
        out = imp.fit(_X4).transform(_X4)
        self.assertFalse(np.any(np.isnan(out)))


# ===========================================================================
# _resolve_index_store_path
# ===========================================================================

class TestResolveIndexStorePath(unittest.TestCase):

    def test_external_none_auto_generates(self):
        imp = _make_imp(index_access="external", index_store_path=None)
        path = imp._resolve_index_store_path()
        self.assertIsNotNone(path)
        try:
            os.remove(path)
        except OSError:
            pass

    def test_external_str_path_normalised(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "my.annoy")
            imp = _make_imp(index_access="external", index_store_path=p)
            result = imp._resolve_index_store_path()
            self.assertEqual(result, str(Path(p).resolve()))

    def test_external_pathnamer_cached(self):
        from scikitplot.utils._path import PathNamer
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(prefix="p_", ext=".annoy", directory=td)
            imp = _make_imp(index_access="external", index_store_path=namer)
            p1 = imp._resolve_index_store_path()
            p2 = imp._resolve_index_store_path()
            self.assertEqual(p1, p2)

    def test_non_external_none_returns_none(self):
        imp = _make_imp(index_access="public", index_store_path=None)
        self.assertIsNone(imp._resolve_index_store_path())

    def test_non_external_str_path_normalised(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "public.annoy")
            imp = _make_imp(index_access="public", index_store_path=p)
            result = imp._resolve_index_store_path()
            self.assertEqual(result, str(Path(p).resolve()))

    def test_invalid_type_raises_type_error(self):
        imp = _make_imp(index_access="external", index_store_path=12345)
        with self.assertRaises(TypeError):
            imp._resolve_index_store_path()


# ===========================================================================
# _compute_neighbor_weights
# ===========================================================================

class TestComputeNeighborWeights(unittest.TestCase):

    def _imp(self):
        return _make_imp()

    def test_uniform_string(self):
        w = self._imp()._compute_neighbor_weights(np.array([1.0, 2.0, 3.0]), "uniform")
        assert_allclose(w, [1.0, 1.0, 1.0])

    def test_uniform_none(self):
        w = self._imp()._compute_neighbor_weights(np.array([1.0, 2.0]), None)
        assert_allclose(w, [1.0, 1.0])

    def test_distance_normalized(self):
        w = self._imp()._compute_neighbor_weights(np.array([1.0, 2.0]), "distance")
        self.assertAlmostEqual(w.sum(), 1.0, places=10)
        self.assertGreater(w[0], w[1])

    def test_zero_distance_inf_weight_dominates(self):
        w = self._imp()._compute_neighbor_weights(np.array([0.0, 1.0]), "distance")
        self.assertAlmostEqual(w[0], 1.0, places=5)
        self.assertAlmostEqual(w[1], 0.0, places=5)

    def test_multiple_zero_distances_shared_equally(self):
        w = self._imp()._compute_neighbor_weights(np.array([0.0, 0.0, 1.0]), "distance")
        self.assertAlmostEqual(w[0], 0.5, places=5)
        self.assertAlmostEqual(w[1], 0.5, places=5)
        self.assertAlmostEqual(w[2], 0.0, places=5)

    def test_callable_weights_normalized(self):
        w = self._imp()._compute_neighbor_weights(np.array([1.0, 3.0]),
                                                   lambda d: 1.0 / (1.0 + d))
        self.assertEqual(len(w), 2)
        self.assertAlmostEqual(w.sum(), 1.0, places=10)

    def test_single_neighbor(self):
        w = self._imp()._compute_neighbor_weights(np.array([0.5]), "distance")
        self.assertAlmostEqual(w[0], 1.0, places=8)


# ===========================================================================
# _impute_from_neighbors
# ===========================================================================

class TestImputeFromNeighbors(unittest.TestCase):

    def _call(self, row, mask, neighbors, weights, fill_vec=None, is_empty=None):
        imp = _make_imp()
        n = len(row)
        fill_vec = fill_vec if fill_vec is not None else np.zeros(n)
        is_empty = is_empty if is_empty is not None else np.zeros(n, bool)
        return imp._impute_from_neighbors(
            row_idx=0, row=row.copy(),
            row_missing_mask=mask, neighbors=neighbors,
            weights=weights, fill_vec=fill_vec,
            is_empty_feature=is_empty,
        )

    def test_imputes_single_missing(self):
        row       = np.array([np.nan, 2.0, 3.0])
        mask      = np.array([True, False, False])
        neighbors = np.array([[1.0, 1.0, 1.0], [4.0, 2.0, 3.0]])
        weights   = np.array([0.5, 0.5])
        result = self._call(row, mask, neighbors, weights)
        assert_allclose(result[0], 2.5)

    def test_no_missing_row_unchanged(self):
        row       = np.array([1.0, 2.0, 3.0])
        mask      = np.array([False, False, False])
        neighbors = np.array([[0.0, 0.0, 0.0]])
        result = self._call(row, mask, neighbors, np.array([1.0]))
        assert_allclose(result, row)

    def test_all_neighbor_nan_falls_back_to_fill(self):
        row       = np.array([np.nan, 2.0])
        mask      = np.array([True, False])
        neighbors = np.array([[np.nan, 1.0], [np.nan, 3.0]])
        fill_vec  = np.array([99.0, 0.0])
        result = self._call(row, mask, neighbors, np.array([0.5, 0.5]),
                            fill_vec=fill_vec)
        assert_allclose(result[0], 99.0)

    def test_empty_feature_skipped(self):
        row       = np.array([np.nan, 2.0])
        mask      = np.array([True, False])
        neighbors = np.array([[5.0, 1.0]])
        is_empty  = np.array([True, False])
        result = self._call(row, mask, neighbors, np.array([1.0]),
                            is_empty=is_empty)
        self.assertTrue(np.isnan(result[0]))

    def test_zero_weight_sum_falls_back_to_plain_mean(self):
        row       = np.array([np.nan])
        mask      = np.array([True])
        neighbors = np.array([[2.0], [4.0]])
        weights   = np.array([0.0, 0.0])
        result = self._call(row, mask, neighbors, weights)
        assert_allclose(result[0], 3.0)

    def test_two_missing_features_both_imputed(self):
        row       = np.array([np.nan, 2.0, np.nan])
        mask      = np.array([True, False, True])
        neighbors = np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 7.0]])
        weights   = np.array([0.5, 0.5])
        result = self._call(row, mask, neighbors, weights)
        assert_allclose(result[0], 3.0)
        assert_allclose(result[2], 5.0)
        assert_allclose(result[1], 2.0)


# ===========================================================================
# _process_single_row_annoy
# ===========================================================================

class TestProcessSingleRowAnnoy(unittest.TestCase):

    def setUp(self):
        self.imp = _make_imp(index_access="public")
        self.imp.fit(_X4)
        self.train_index = self.imp.train_index_
        self.fill_vec    = self.imp.temp_fill_vector_
        self.is_empty    = self.imp._is_empty_feature

    def _call(self, i, row, mask):
        return self.imp._process_single_row_annoy(
            i, row, mask, self.train_index, self.fill_vec,
            self.imp.n_neighbors, self.imp.search_k,
            self.imp.weights, self.is_empty,
        )

    def test_complete_row_returned_unchanged(self):
        row  = np.array([3.0, 4.0, 3.0])
        mask = np.array([False, False, False])
        _, new_row = self._call(0, row, mask)
        assert_allclose(new_row, row)

    def test_missing_row_imputed(self):
        row  = _X4[0].copy()
        mask = np.array([False, False, True])
        _, new_row = self._call(0, row, mask)
        self.assertFalse(np.isnan(new_row[2]))

    def test_returns_original_row_index(self):
        row  = np.array([3.0, 4.0, 3.0])
        mask = np.array([False, False, False])
        idx, _ = self._call(7, row, mask)
        self.assertEqual(idx, 7)


# ===========================================================================
# get_feature_names_out
# ===========================================================================

class TestANNImputerFeatureNames(unittest.TestCase):

    def test_names_match_n_features(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertEqual(len(imp.get_feature_names_out()), imp.n_features_in_)

    def test_custom_names(self):
        imp = _make_imp()
        imp.fit(_X4)
        assert_array_equal(imp.get_feature_names_out(["a", "b", "c"]),
                           ["a", "b", "c"])

    def test_empty_col_excluded_from_names(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0],
                      [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(keep_empty_features=False)
        imp.fit(X)
        assert_array_equal(
            imp.get_feature_names_out(["empty_col", "valid_col"]),
            ["valid_col"],
        )

    def test_indicator_names_appended(self):
        X = np.array([[np.nan, 1.0], [2.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        imp = _make_imp(add_indicator=True)
        imp.fit(X)
        names = imp.get_feature_names_out(["x", "y"])
        self.assertIn("missingindicator_x", list(names))


# ===========================================================================
# VoyagerKNNImputer
# ===========================================================================

class TestVoyagerKNNImputer(unittest.TestCase):

    def test_backend_forced_to_voyager(self):
        self.assertEqual(VoyagerKNNImputer().backend, "voyager")

    def test_bad_backend_override_raises(self):
        with self.assertRaises(ValueError):
            VoyagerKNNImputer(backend="annoy")


# ===========================================================================
# AnnoyKNNImputer backward-compat alias
# ===========================================================================

class TestAnnoyKNNImputer(unittest.TestCase):

    def test_is_same_class(self):
        self.assertIs(AnnoyKNNImputer, ANNImputer)

    def test_fit_transform_works(self):
        imp = AnnoyKNNImputer(backend="annoy", index_access="public",
                               n_trees=3, n_neighbors=2,
                               metric="euclidean", random_state=42)
        out = imp.fit_transform(_X4)
        self.assertFalse(np.any(np.isnan(out)))


# ===========================================================================
# Voyager backend — skipped when voyager not installed
# ===========================================================================

class TestANNImputerVoyagerBackend(unittest.TestCase):

    def setUp(self):
        try:
            import voyager  # noqa: F401
            self._has_voyager = True
        except ImportError:
            self._has_voyager = False

    def test_voyager_unavailable_raises_import_error(self):
        if self._has_voyager:
            self.skipTest("voyager installed")
        imp = ANNImputer(backend="voyager", index_access="public",
                         n_trees=3, n_neighbors=2)
        with self.assertRaises(ImportError):
            imp.fit(_X4)

    def test_voyager_metric_raises_when_unavailable(self):
        if self._has_voyager:
            self.skipTest("voyager installed")
        with self.assertRaises(ImportError):
            ANNImputer(backend="voyager")._resolve_voyager_space()


# ===========================================================================
# Integration tests
# ===========================================================================

class TestANNImputerIntegration(unittest.TestCase):

    def test_larger_dataset(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        X[rng.rand(50, 5) < 0.2] = np.nan
        X[0, :] = 1.0  # guarantee at least one valid row
        out = _make_imp(n_neighbors=3, index_access="public").fit_transform(X)
        self.assertEqual(out.shape, (50, 5))
        self.assertFalse(np.any(np.isnan(out)))

    def test_single_feature(self):
        X = np.array([[np.nan], [1.0], [2.0], [3.0]])
        out = _make_imp(n_neighbors=2).fit_transform(X)
        self.assertEqual(out.shape, (4, 1))
        self.assertFalse(np.isnan(out[0, 0]))

    def test_fit_transform_idempotent_on_complete(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert_allclose(_make_imp().fit_transform(X), X)

    def test_sklearn_clone_preserves_params(self):
        from sklearn.base import clone
        imp = _make_imp(n_neighbors=7, weights="distance")
        imp2 = clone(imp)
        self.assertEqual(imp2.n_neighbors, 7)
        self.assertEqual(imp2.weights, "distance")

    def test_get_set_params_roundtrip(self):
        imp = _make_imp()
        self.assertIn("n_neighbors", imp.get_params())
        imp.set_params(n_neighbors=10)
        self.assertEqual(imp.n_neighbors, 10)

    def test_delete_external_index_removes_file(self):
        imp = _make_imp(index_access="external")
        imp.fit(_X4)
        path = imp.index_path_
        self.assertTrue(os.path.exists(path))
        imp.delete_external_index()
        self.assertFalse(os.path.exists(path))

    def test_n_jobs_set(self):
        out = _make_imp(n_jobs=2).fit_transform(_X4)
        self.assertFalse(np.any(np.isnan(out)))

    def test_all_nan_does_not_succeed_silently(self):
        """Regression: all-NaN must not produce a silently imputed result."""
        X = np.full((5, 3), np.nan)
        imp = _make_imp()
        raised = False
        try:
            imp.fit_transform(X)
        except (ValueError, RuntimeWarning, RuntimeError):
            raised = True
        self.assertTrue(raised, "Expected an error for all-NaN input")



# ===========================================================================
# _prepare_query_vector
# ===========================================================================

class TestPrepareQueryVector(unittest.TestCase):
    """Tests for ``ANNImputer._prepare_query_vector``.

    Covers: NaN substitution, +inf/-inf treated as missing, complete rows,
    and mixed NaN + inf inputs.
    """

    def _imp(self):
        return _make_imp()

    def test_nan_replaced_by_fill_vec(self):
        imp = self._imp()
        row      = np.array([np.nan, 2.0, np.nan])
        fill_vec = np.array([10.0, 99.0, 20.0])
        result   = imp._prepare_query_vector(row, fill_vec)
        assert_allclose(result, [10.0, 2.0, 20.0])

    def test_complete_row_unchanged(self):
        imp = self._imp()
        row      = np.array([1.0, 2.0, 3.0])
        fill_vec = np.array([99.0, 99.0, 99.0])
        result   = imp._prepare_query_vector(row, fill_vec)
        assert_allclose(result, row)

    def test_positive_inf_treated_as_missing(self):
        imp = self._imp()
        row      = np.array([np.inf, 2.0])
        fill_vec = np.array([7.0, 0.0])
        result   = imp._prepare_query_vector(row, fill_vec)
        assert_allclose(result[0], 7.0)
        assert_allclose(result[1], 2.0)

    def test_negative_inf_treated_as_missing(self):
        imp = self._imp()
        row      = np.array([-np.inf, 3.0])
        fill_vec = np.array([5.0, 0.0])
        result   = imp._prepare_query_vector(row, fill_vec)
        assert_allclose(result[0], 5.0)

    def test_mixed_nan_and_inf_both_replaced(self):
        imp = self._imp()
        row      = np.array([np.nan, np.inf, -np.inf, 4.0])
        fill_vec = np.array([1.0, 2.0, 3.0, 0.0])
        result   = imp._prepare_query_vector(row, fill_vec)
        assert_allclose(result[:3], [1.0, 2.0, 3.0])
        assert_allclose(result[3], 4.0)

    def test_single_element_nan(self):
        imp = self._imp()
        row      = np.array([np.nan])
        fill_vec = np.array([42.0])
        result   = imp._prepare_query_vector(row, fill_vec)
        assert_allclose(result, [42.0])

    def test_all_missing_replaced_by_fill_vec(self):
        imp = self._imp()
        row      = np.array([np.nan, np.nan, np.nan])
        fill_vec = np.array([1.0, 2.0, 3.0])
        result   = imp._prepare_query_vector(row, fill_vec)
        assert_allclose(result, fill_vec)


# ===========================================================================
# _iter_non_nan_rows
# ===========================================================================

class TestIterNonNanRows(unittest.TestCase):
    """Tests for ``ANNImputer._iter_non_nan_rows``.

    Covers: complete rows yielded, rows with NaN skipped, mixed input,
    all-NaN yields nothing, single-row matrix.
    """

    def _imp(self):
        return _make_imp()

    def test_complete_rows_yielded(self):
        imp = self._imp()
        X   = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = list(imp._iter_non_nan_rows(X))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[1][0], 1)

    def test_nan_rows_skipped(self):
        imp = self._imp()
        X   = np.array([[np.nan, 2.0], [3.0, 4.0], [5.0, np.nan]])
        result = list(imp._iter_non_nan_rows(X))
        # Only row index 1 is complete
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 1)
        assert_allclose(result[0][1], [3.0, 4.0])

    def test_all_nan_yields_nothing(self):
        imp = self._imp()
        X   = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        result = list(imp._iter_non_nan_rows(X))
        self.assertEqual(len(result), 0)

    def test_single_complete_row(self):
        imp = self._imp()
        X   = np.array([[1.0, 2.0, 3.0]])
        result = list(imp._iter_non_nan_rows(X))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 0)

    def test_index_values_are_correct(self):
        imp = self._imp()
        X   = np.array([[np.nan], [1.0], [np.nan], [2.0]])
        idxs = [idx for idx, _ in imp._iter_non_nan_rows(X)]
        self.assertEqual(idxs, [1, 3])


# ===========================================================================
# copy parameter behaviour
# ===========================================================================

class TestANNImputerCopyParam(unittest.TestCase):
    """Tests for the ``copy`` init parameter during transform.

    When ``copy=True`` (default) the original array must not be mutated.
    When ``copy=False`` the imputer is allowed to work in-place.
    """

    def test_copy_true_does_not_mutate_input(self):
        X = np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                     dtype=float)
        X_orig = X.copy()
        imp = _make_imp(copy=True)
        imp.fit(X)
        imp.transform(X)
        assert_allclose(X, X_orig)

    def test_copy_false_runs_without_error(self):
        X = np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                     dtype=float)
        imp = _make_imp(copy=False)
        out = imp.fit_transform(X)
        self.assertFalse(np.any(np.isnan(out)))


# ===========================================================================
# transform — keep_empty_features=True with missing values in valid cols
# ===========================================================================

class TestTransformKeepEmptyFeaturesWithMissing(unittest.TestCase):
    """Tests for transform when ``keep_empty_features=True`` and the valid
    columns still contain missing values that must be imputed.
    """

    def test_output_shape_preserved_with_empty_col(self):
        X_train = np.array([[np.nan, 1.0], [np.nan, 2.0],
                             [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(keep_empty_features=True)
        out = imp.fit_transform(X_train)
        self.assertEqual(out.shape, (4, 2))

    def test_empty_col_zeroed_in_output(self):
        X_train = np.array([[np.nan, 1.0], [np.nan, 2.0],
                             [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(keep_empty_features=True)
        out = imp.fit_transform(X_train)
        assert_array_equal(out[:, 0], [0.0, 0.0, 0.0, 0.0])

    def test_valid_col_imputed(self):
        X_train = np.array([[np.nan, 1.0], [np.nan, 2.0],
                             [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(keep_empty_features=True)
        out = imp.fit_transform(X_train)
        # Valid column (index 1) must be present and finite
        self.assertTrue(np.all(np.isfinite(out[:, 1])))

    def test_transform_with_empty_col_and_missing_in_valid(self):
        X_train = np.array([[np.nan, 1.0], [np.nan, 2.0],
                             [np.nan, 3.0], [np.nan, 4.0]])
        X_test  = np.array([[np.nan, np.nan], [np.nan, 3.0]])
        imp = _make_imp(keep_empty_features=True)
        imp.fit(X_train)
        out = imp.transform(X_test)
        self.assertEqual(out.shape, (2, 2))
        self.assertFalse(np.any(np.isnan(out)))


# ===========================================================================
# transform — fast-path (no missing in valid cols) with keep_empty_features
# ===========================================================================

class TestTransformFastPathKeepEmptyFeatures(unittest.TestCase):
    """Fast-path branch: no missing values in valid columns during transform.

    The code skips imputation entirely and only zeroes empty features.
    """

    def test_fast_path_keep_empty_true_shape(self):
        """keep_empty_features=True → all columns retained."""
        X_train = np.array([[np.nan, 1.0], [np.nan, 2.0],
                             [np.nan, 3.0], [np.nan, 4.0]])
        X_test  = np.array([[np.nan, 5.0], [np.nan, 6.0]])
        imp = _make_imp(keep_empty_features=True)
        imp.fit(X_train)
        out = imp.transform(X_test)
        self.assertEqual(out.shape, (2, 2))

    def test_fast_path_keep_empty_true_zeroes_empty_col(self):
        X_train = np.array([[np.nan, 1.0], [np.nan, 2.0],
                             [np.nan, 3.0], [np.nan, 4.0]])
        X_test  = np.array([[np.nan, 5.0], [np.nan, 6.0]])
        imp = _make_imp(keep_empty_features=True)
        imp.fit(X_train)
        out = imp.transform(X_test)
        assert_array_equal(out[:, 0], [0.0, 0.0])

    def test_fast_path_keep_empty_false_drops_empty_col(self):
        X_train = np.array([[np.nan, 1.0], [np.nan, 2.0],
                             [np.nan, 3.0], [np.nan, 4.0]])
        X_test  = np.array([[np.nan, 5.0], [np.nan, 6.0]])
        imp = _make_imp(keep_empty_features=False)
        imp.fit(X_train)
        out = imp.transform(X_test)
        self.assertEqual(out.shape[1], 1)
        assert_allclose(out[:, 0], [5.0, 6.0])


# ===========================================================================
# DataFrame input
# ===========================================================================

class TestANNImputerDataFrameInput(unittest.TestCase):
    """Tests for fit/transform with pandas DataFrame inputs."""

    def setUp(self):
        try:
            import pandas as pd
            self._pd = pd
        except ImportError:
            self.skipTest("pandas not installed")

    def test_fit_with_dataframe(self):
        pd = self._pd
        X  = pd.DataFrame({"a": [np.nan, 2.0, 3.0, 4.0],
                            "b": [1.0, np.nan, 3.0, 4.0],
                            "c": [1.0, 2.0, np.nan, 4.0]})
        imp = _make_imp()
        imp.fit(X)
        self.assertEqual(imp.n_features_in_, 3)

    def test_transform_with_dataframe(self):
        pd = self._pd
        X  = pd.DataFrame({"a": [np.nan, 2.0, 3.0, 4.0],
                            "b": [1.0, np.nan, 3.0, 4.0],
                            "c": [1.0, 2.0, np.nan, 4.0]})
        imp = _make_imp()
        out = imp.fit_transform(X)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertEqual(out.shape, (4, 3))

    def test_feature_names_from_dataframe(self):
        pd = self._pd
        X  = pd.DataFrame({"alpha": [np.nan, 2.0, 3.0, 4.0],
                            "beta":  [1.0, 2.0, 3.0, 4.0],
                            "gamma": [1.0, 2.0, 3.0, 4.0]})
        imp = _make_imp()
        imp.fit(X)
        names = imp.get_feature_names_out()
        self.assertEqual(len(names), 3)


# ===========================================================================
# _compute_initial_fill_vector — warning suppression verified
# ===========================================================================

class TestComputeInitialFillVectorWarningSuppression(unittest.TestCase):
    """Verify that warnings.catch_warnings correctly suppresses
    RuntimeWarning from nanmean/nanmedian on all-NaN columns in all
    warning-filter modes, including ``simplefilter("error")``.
    """

    def _X_all_nan_col(self):
        return np.array([[np.nan, 1.0], [np.nan, 2.0],
                         [np.nan, 3.0], [np.nan, 4.0]])

    def test_mean_no_warning_in_error_filter_mode(self):
        X   = self._X_all_nan_col()
        imp = _make_imp(initial_strategy="mean")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            imp.fit(X)   # must not raise

    def test_median_no_warning_in_error_filter_mode(self):
        X   = self._X_all_nan_col()
        imp = _make_imp(initial_strategy="median")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            imp.fit(X)   # must not raise

    def test_mean_fallback_value_is_zero(self):
        imp = _make_imp(initial_strategy="mean")
        X   = self._X_all_nan_col()
        # Access internal method directly
        fv  = imp._compute_initial_fill_vector(X)
        self.assertAlmostEqual(fv[0], 0.0)

    def test_median_fallback_value_is_zero(self):
        imp = _make_imp(initial_strategy="median")
        X   = self._X_all_nan_col()
        fv  = imp._compute_initial_fill_vector(X)
        self.assertAlmostEqual(fv[0], 0.0)

    def test_most_frequent_fallback_value_is_zero(self):
        imp = _make_imp(initial_strategy="most_frequent")
        X   = self._X_all_nan_col()
        fv  = imp._compute_initial_fill_vector(X)
        self.assertAlmostEqual(fv[0], 0.0)

    def test_constant_fill_value_used(self):
        imp = _make_imp(initial_strategy="constant", fill_value=7.0)
        X   = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        fv  = imp._compute_initial_fill_vector(X)
        assert_allclose(fv, [7.0, 7.0])

    def test_mean_returns_correct_value_for_non_nan_col(self):
        imp = _make_imp(initial_strategy="mean")
        X   = self._X_all_nan_col()
        fv  = imp._compute_initial_fill_vector(X)
        # col 1: mean of [1,2,3,4] = 2.5
        self.assertAlmostEqual(fv[1], 2.5)

    def test_median_returns_correct_value_for_non_nan_col(self):
        imp = _make_imp(initial_strategy="median")
        X   = self._X_all_nan_col()
        fv  = imp._compute_initial_fill_vector(X)
        # col 1: median of [1,2,3,4] = 2.5
        self.assertAlmostEqual(fv[1], 2.5)


# ===========================================================================
# n_samples_fit_ and n_features_in_ attributes
# ===========================================================================

class TestANNImputerFitAttributes(unittest.TestCase):
    """Tests for post-fit attributes beyond basic checks."""

    def test_n_samples_fit_set_correctly(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertEqual(imp.n_samples_fit_, 4)

    def test_n_samples_fit_changes_on_refit(self):
        imp = _make_imp()
        imp.fit(_X4)
        X_big = np.vstack([_X4, _X4])  # 8 rows
        imp.fit(X_big)
        self.assertEqual(imp.n_samples_fit_, 8)

    def test_n_features_in_set(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertEqual(imp.n_features_in_, 3)

    def test_is_empty_feature_shape(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertEqual(imp._is_empty_feature.shape, (3,))

    def test_temp_fill_vector_no_nan_after_fit(self):
        imp = _make_imp()
        imp.fit(_X4)
        self.assertFalse(np.any(np.isnan(imp.temp_fill_vector_)))

    def test_index_created_at_set_after_fit(self):
        imp = _make_imp(index_access="external")
        imp.fit(_X4)
        self.assertTrue(hasattr(imp, "index_created_at_"))
        imp.delete_external_index()


# ===========================================================================
# _compute_neighbor_weights — additional edge cases
# ===========================================================================

class TestComputeNeighborWeightsExtra(unittest.TestCase):
    """Additional edge cases for ``_compute_neighbor_weights``."""

    def _imp(self):
        return _make_imp()

    def test_empty_distance_array_uniform(self):
        """Uniform weights on a zero-length array returns empty array."""
        w = self._imp()._compute_neighbor_weights(np.array([]), "uniform")
        self.assertEqual(w.shape, (0,))

    def test_callable_with_all_zeros(self):
        """Callable called with zero distances; result normalized."""
        def fn(d):
            return np.ones_like(d)
        w = self._imp()._compute_neighbor_weights(np.array([0.0, 0.0]), fn)
        self.assertAlmostEqual(w.sum(), 1.0, places=10)

    def test_distance_weights_sum_to_one(self):
        w = self._imp()._compute_neighbor_weights(
            np.array([0.5, 1.0, 2.0]), "distance"
        )
        self.assertAlmostEqual(w.sum(), 1.0, places=10)

    def test_distance_weights_closest_has_highest_weight(self):
        w = self._imp()._compute_neighbor_weights(
            np.array([0.1, 0.5, 2.0]), "distance"
        )
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])

    def test_uniform_none_returns_all_ones(self):
        w = self._imp()._compute_neighbor_weights(np.array([3.0, 1.0, 2.0]), None)
        assert_allclose(w, [1.0, 1.0, 1.0])


# ===========================================================================
# _impute_from_neighbors — additional edge cases
# ===========================================================================

class TestImputeFromNeighborsExtra(unittest.TestCase):
    """Additional edge cases for ``_impute_from_neighbors``."""

    def _call(self, row, mask, neighbors, weights,
              fill_vec=None, is_empty=None):
        imp = _make_imp()
        n = len(row)
        fill_vec = fill_vec if fill_vec is not None else np.zeros(n)
        is_empty = is_empty if is_empty is not None else np.zeros(n, bool)
        return imp._impute_from_neighbors(
            row_idx=0, row=row.copy(),
            row_missing_mask=mask, neighbors=neighbors,
            weights=weights, fill_vec=fill_vec,
            is_empty_feature=is_empty,
        )

    def test_partial_nan_in_neighbors_uses_valid_subset(self):
        """If some neighbors are NaN in a feature, only valid ones are used."""
        row       = np.array([np.nan, 2.0])
        mask      = np.array([True, False])
        neighbors = np.array([[np.nan, 1.0], [4.0, 2.0]])
        weights   = np.array([0.5, 0.5])
        result    = self._call(row, mask, neighbors, weights)
        # Only neighbor[1] has valid data for feature 0 → result is 4.0
        assert_allclose(result[0], 4.0)

    def test_observed_features_not_modified(self):
        """Non-missing entries must be carried through unchanged."""
        row       = np.array([1.0, np.nan, 3.0])
        mask      = np.array([False, True, False])
        neighbors = np.array([[9.0, 5.0, 9.0], [9.0, 7.0, 9.0]])
        weights   = np.array([0.5, 0.5])
        result    = self._call(row, mask, neighbors, weights)
        assert_allclose(result[0], 1.0)
        assert_allclose(result[2], 3.0)

    def test_empty_feature_left_as_nan(self):
        """is_empty_feature=True → slot must remain NaN (zeroed by transform)."""
        row       = np.array([np.nan, 2.0])
        mask      = np.array([True, False])
        neighbors = np.array([[5.0, 1.0]])
        is_empty  = np.array([True, False])
        result    = self._call(row, mask, neighbors, np.array([1.0]),
                               is_empty=is_empty)
        self.assertTrue(np.isnan(result[0]))

    def test_weighted_average_correctness(self):
        """Verify the weighted imputed value is arithmetically correct."""
        row       = np.array([np.nan])
        mask      = np.array([True])
        neighbors = np.array([[2.0], [8.0]])
        weights   = np.array([0.75, 0.25])
        result    = self._call(row, mask, neighbors, weights)
        expected  = 2.0 * 0.75 + 8.0 * 0.25
        assert_allclose(result[0], expected, rtol=1e-12)


# ===========================================================================
# _resolve_metric — additional aliases
# ===========================================================================

class TestResolveMetricExtra(unittest.TestCase):
    """Direct unit tests for ``_resolve_metric`` canonical mapping."""

    _CANONICAL = {
        "angular":   "angular",
        "cosine":    "angular",
        "euclidean": "euclidean",
        "l2":        "euclidean",
        "lstsq":     "euclidean",
        "manhattan": "manhattan",
        "l1":        "manhattan",
        "cityblock": "manhattan",
        "taxicab":   "manhattan",
        "dot":       "dot",
        "@":         "dot",
        ".":         "dot",
        "dotproduct": "dot",
        "inner":      "dot",
        "innerproduct": "dot",
        "hamming":   "hamming",
    }

    def test_all_aliases_resolve_to_canonical(self):
        for alias, canonical in self._CANONICAL.items():
            with self.subTest(alias=alias):
                imp = _make_imp(metric=alias)
                self.assertEqual(imp._resolve_metric(), canonical)

    def test_unknown_metric_raises_value_error(self):
        imp = _make_imp(metric="not_a_metric")
        with self.assertRaises(ValueError):
            imp._resolve_metric()


# ===========================================================================
# transform — add_indicator with empty features
# ===========================================================================

class TestTransformIndicatorWithEmptyFeature(unittest.TestCase):
    """Tests for the indicator matrix path combined with empty features."""

    def test_indicator_shape_with_empty_col_keep_false(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0],
                      [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(keep_empty_features=False, add_indicator=True)
        out = imp.fit_transform(X)
        # 1 valid col + 1 indicator col for the originally missing col
        self.assertEqual(out.shape[0], 4)
        self.assertGreaterEqual(out.shape[1], 1)

    def test_indicator_shape_with_empty_col_keep_true(self):
        X = np.array([[np.nan, 1.0], [np.nan, 2.0],
                      [np.nan, 3.0], [np.nan, 4.0]])
        imp = _make_imp(keep_empty_features=True, add_indicator=True)
        out = imp.fit_transform(X)
        self.assertEqual(out.shape[0], 4)
        self.assertGreaterEqual(out.shape[1], 2)


# ===========================================================================
# Integration — sklearn compatibility
# ===========================================================================

class TestANNImputerSklearnCompat(unittest.TestCase):
    """sklearn estimator API contract tests."""

    def test_get_params_keys(self):
        imp    = _make_imp()
        params = imp.get_params()
        for key in ("n_neighbors", "weights", "metric", "backend",
                    "n_trees", "index_access", "random_state"):
            self.assertIn(key, params)

    def test_set_params_round_trip(self):
        imp = _make_imp()
        imp.set_params(n_neighbors=7, weights="distance")
        self.assertEqual(imp.n_neighbors, 7)
        self.assertEqual(imp.weights, "distance")

    def test_clone_preserves_all_params(self):
        from sklearn.base import clone
        imp  = _make_imp(n_neighbors=9, metric="manhattan",
                         weights="distance", n_trees=10)
        imp2 = clone(imp)
        self.assertEqual(imp2.n_neighbors, 9)
        self.assertEqual(imp2.metric, "manhattan")
        self.assertEqual(imp2.weights, "distance")
        self.assertEqual(imp2.n_trees, 10)

    def test_repr_contains_class_name(self):
        imp = _make_imp()
        self.assertIn("ANNImputer", repr(imp))

    def test_fit_transform_output_is_ndarray(self):
        out = _make_imp().fit_transform(_X4)
        self.assertIsInstance(out, np.ndarray)

    def test_transform_output_is_float(self):
        out = _make_imp().fit(_X4).transform(_X4)
        self.assertTrue(np.issubdtype(out.dtype, np.floating))


# ===========================================================================
# Integration — robustness with varied input sizes / patterns
# ===========================================================================

class TestANNImputerRobustness(unittest.TestCase):
    """Stress / robustness tests for varied missing patterns."""

    def test_single_row_single_col_complete(self):
        X   = np.array([[5.0]])
        out = _make_imp(n_neighbors=1).fit_transform(X)
        assert_allclose(out, [[5.0]])

    def test_high_missing_rate(self):
        """50 % missing rate must not raise or leave NaNs."""
        rng = np.random.RandomState(0)
        X   = rng.randn(20, 4)
        X[rng.rand(20, 4) < 0.5] = np.nan
        # Ensure at least one fully-observed row
        X[0, :] = 1.0
        out = _make_imp(n_neighbors=3, index_access="public").fit_transform(X)
        self.assertFalse(np.any(np.isnan(out)))

    def test_two_by_two_minimal(self):
        """Minimal 2×2 dataset with one missing."""
        X = np.array([[np.nan, 1.0], [2.0, 3.0]])
        out = _make_imp(n_neighbors=1).fit_transform(X)
        self.assertFalse(np.any(np.isnan(out)))

    def test_output_dtype_float64(self):
        X   = np.array([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        out = _make_imp().fit_transform(X)
        self.assertTrue(np.issubdtype(out.dtype, np.floating))

    def test_all_columns_missing_in_test_raises_or_fills(self):
        """A row that is all-NaN at transform time must not crash."""
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_test  = np.array([[np.nan, np.nan]])
        imp     = _make_imp(n_neighbors=2, index_access="public")
        imp.fit(X_train)
        # Must either succeed (fill from neighbors) or raise — not crash
        try:
            out = imp.transform(X_test)
            self.assertFalse(np.any(np.isnan(out)))
        except (ValueError, RuntimeError):
            pass

    def test_refit_clears_old_state(self):
        """Re-fitting on different-shape data must update n_features_in_."""
        imp = _make_imp()
        imp.fit(_X4)
        self.assertEqual(imp.n_features_in_, 3)
        X2 = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
        imp.fit(X2)
        self.assertEqual(imp.n_features_in_, 2)

    def test_fit_transform_consistent_with_separate_calls(self):
        out1 = _make_imp(random_state=42).fit(_X4).transform(_X4)
        out2 = _make_imp(random_state=42).fit_transform(_X4)
        assert_allclose(out1, out2, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
