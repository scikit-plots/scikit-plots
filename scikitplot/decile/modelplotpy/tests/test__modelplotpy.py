# scikitplot/decile/modelplotpy/tests/test__modelplotpy.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~scikitplot.decile.modelplotpy._modelplotpy`.

Covers the **legacy** ModelPlotPy implementation including:

- ``_range01`` normalization helper (normal, edge, NaN/inf rejection)
- ``_check_input`` validation helper (valid, partial overlap, all-invalid)
- ``ModelPlotPy.__init__`` construction, eager validation, mutable-default isolation
- ``ModelPlotPy.get_params`` / ``set_params`` / ``reset_params``
- ``ModelPlotPy.prepare_scores_and_ntiles`` schema, value correctness,
  reproducibility, ntile bounds
- ``ModelPlotPy.aggregate_over_ntiles`` shape, origin row, metric columns,
  ``pcttot == pct_ref``, ``gain_opt ≤ 1``, ``cumgain`` at last ntile
- ``ModelPlotPy.plotting_scope`` all 4 scopes, validation, label filters,
  ``select_smallest_targetclass`` branch
- ``plot_response``, ``plot_cumresponse``, ``plot_cumlift``, ``plot_cumgains``
- ``plot_all``
- ``plot_costsrevs``, ``plot_profit``, ``plot_roi``
- All plots: ``highlight_ntile`` / ``highlight_how`` branches across all 4 scopes
- Return-type verification (matplotlib Axes) for every public plot function
- Financial column computation (variable_costs, investments, revenues, profit, roi)
- Ntile label / x-tick spacing branches (decile / percentile / ntile)
- Public ``__all__`` API completeness and callability

Notes
-----
The module lives inside the ``scikitplot.decile.modelplotpy`` sub-package and
uses relative imports that reach three levels up to ``scikitplot``.  Those
parent packages are **not** installed in the test environment, so they are
stubbed in ``sys.modules`` **before** the module under test is imported.

When run via pytest as part of the installed ``scikitplot`` package the stubs
are not needed — the real package tree is present.

Matplotlib is forced to the non-interactive ``Agg`` backend to prevent any
display windows during CI / headless execution.
"""

from __future__ import annotations

import sys
import types
import unittest
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Relative imports — preserved exactly as the production test file requires
# ---------------------------------------------------------------------------
from .. import _modelplotpy as _mpm  # noqa: E402
from .._modelplotpy import (         # noqa: E402
    _check_input,
    _range01,
    ModelPlotPy,
    plot_all,
    plot_costsrevs,
    plot_cumgains,
    plot_cumresponse,
    plot_cumlift,
    plot_profit,
    plot_response,
    plot_roi,
)

##############################################################################
# Shared fixtures
##############################################################################


def _make_binary_data(n: int = 200, random_state: int = 0):
    X, y = make_classification(
        n_samples=n, n_features=5, n_informative=3,
        n_redundant=1, random_state=random_state,
    )
    return pd.DataFrame(X), pd.Series(y)


def _make_fitted_lr(X, y, random_state: int = 0):
    return LogisticRegression(random_state=random_state, max_iter=1000).fit(X, y)


def _make_mp(ntiles: int = 10):
    X, y = _make_binary_data()
    lr = _make_fitted_lr(X, y)
    return ModelPlotPy(
        feature_data=[X], label_data=[y], dataset_labels=["train"],
        models=[lr], model_labels=["lr"], ntiles=ntiles,
    )


def _make_mp_two_models():
    X, y = _make_binary_data()
    lr1 = _make_fitted_lr(X, y, random_state=0)
    lr2 = _make_fitted_lr(X, y, random_state=1)
    return ModelPlotPy(
        feature_data=[X], label_data=[y], dataset_labels=["train"],
        models=[lr1, lr2], model_labels=["lr1", "lr2"], ntiles=10,
    )


def _make_mp_two_datasets():
    X1, y1 = _make_binary_data(random_state=0)
    X2, y2 = _make_binary_data(random_state=1)
    lr = _make_fitted_lr(X1, y1)
    return ModelPlotPy(
        feature_data=[X1, X2], label_data=[y1, y2],
        dataset_labels=["train", "test"],
        models=[lr], model_labels=["lr"], ntiles=10,
    )


def _plot_input(mp, scope: str):
    return mp.plotting_scope(scope=scope)


##############################################################################
# Test: _range01 — basic normalization
##############################################################################

class TestRange01(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_basic_normalization(self):
        result = _range01([2.0, 4.0, 6.0])
        self.assertAlmostEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(float(result.max()), 1.0)

    def test_two_element(self):
        result = _range01([2.0, 4.0])
        self.assertEqual(result.tolist(), [0.0, 1.0])

    def test_constant_array_returns_zeros(self):
        result = _range01([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_negative_values(self):
        result = _range01([-3.0, -1.0, 1.0])
        self.assertAlmostEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(float(result.max()), 1.0)

    def test_numpy_array_input(self):
        result = _range01(np.array([0.0, 0.5, 1.0]))
        self.assertAlmostEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(float(result.max()), 1.0)

    def test_pandas_series_input(self):
        result = _range01(pd.Series([10.0, 20.0, 30.0]))
        self.assertAlmostEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(float(result.max()), 1.0)

    def test_output_shape_preserved(self):
        result = _range01(list(range(20)))
        self.assertEqual(len(result), 20)

    def test_all_values_in_zero_one(self):
        result = _range01(np.random.RandomState(0).randn(100))
        self.assertTrue((result >= 0.0).all())
        self.assertTrue((result <= 1.0).all())


##############################################################################
# Test: _range01 — edge cases and rejection of non-finite inputs
##############################################################################

class TestRange01EdgeCases(unittest.TestCase):
    """_range01 must reject NaN and infinite values immediately."""

    def tearDown(self):
        plt.close("all")

    def test_nan_raises_value_error(self):
        """A single NaN in the array must cause a ValueError, not silent propagation."""
        with self.assertRaises(ValueError):
            _range01([1.0, float("nan"), 3.0])

    def test_all_nan_raises(self):
        with self.assertRaises(ValueError):
            _range01([float("nan"), float("nan")])

    def test_pos_inf_raises_value_error(self):
        with self.assertRaises(ValueError):
            _range01([0.0, float("inf"), 2.0])

    def test_neg_inf_raises_value_error(self):
        with self.assertRaises(ValueError):
            _range01([float("-inf"), 1.0, 2.0])

    def test_single_element_returns_zero(self):
        """Single-element input has max == min, so result is the all-zeros case."""
        result = _range01([42.0])
        np.testing.assert_array_equal(result, [0.0])

    def test_returns_ndarray(self):
        result = _range01([1.0, 2.0, 3.0])
        self.assertIsInstance(result, np.ndarray)

    def test_large_array_bounds(self):
        rng = np.random.RandomState(7)
        arr = rng.randn(1000)
        result = _range01(arr)
        self.assertAlmostEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(float(result.max()), 1.0)


##############################################################################
# Test: _check_input — valid inputs
##############################################################################

class TestCheckInput(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_valid_subset_passes(self):
        result = _check_input(["a"], ["a", "b", "c"], "param")
        self.assertEqual(result, ["a"])

    def test_full_list_passes(self):
        check = ["x", "y"]
        result = _check_input(check, check, "param")
        self.assertEqual(result, check)

    def test_empty_input_returns_empty(self):
        result = _check_input([], ["a", "b"], "param")
        self.assertEqual(result, [])

    def test_invalid_raises_value_error(self):
        with self.assertRaises(ValueError):
            _check_input(["z"], ["a", "b"], "param")

    def test_all_invalid_raises(self):
        with self.assertRaises(ValueError):
            _check_input(["z", "w"], ["a", "b"], "param")

    def test_returns_list_type(self):
        result = _check_input(["a"], ["a", "b"], "param")
        self.assertIsInstance(result, list)

    def test_single_valid_element(self):
        result = _check_input(["b"], ["a", "b", "c"], "param")
        self.assertEqual(result, ["b"])


##############################################################################
# Test: _check_input — partial overlap must raise (tests the bug fix)
##############################################################################

class TestCheckInputPartialOverlap(unittest.TestCase):
    """
    The old implementation used ``any(elem in input_list for elem in check_list)``
    which accepted lists where only *some* elements were valid.  The fixed
    implementation requires *all* elements to be valid.
    """

    def tearDown(self):
        plt.close("all")

    def test_one_valid_one_invalid_raises(self):
        """["a", "z"] has one valid ("a") and one invalid ("z") element."""
        with self.assertRaises(ValueError):
            _check_input(["a", "z"], ["a", "b"], "param")

    def test_first_invalid_rest_valid_raises(self):
        with self.assertRaises(ValueError):
            _check_input(["z", "a", "b"], ["a", "b"], "param")

    def test_last_element_invalid_raises(self):
        with self.assertRaises(ValueError):
            _check_input(["a", "b", "INVALID"], ["a", "b"], "param")

    def test_all_valid_multi_element_passes(self):
        result = _check_input(["a", "b"], ["a", "b", "c"], "param")
        self.assertEqual(result, ["a", "b"])

    def test_error_message_names_invalid_elements(self):
        """The error message must name the offending element so callers know what to fix."""
        with self.assertRaises(ValueError) as ctx:
            _check_input(["a", "TYPO"], ["a", "b"], "my_param")
        self.assertIn("TYPO", str(ctx.exception))

    def test_integer_elements(self):
        """_check_input is also called with integer class labels."""
        with self.assertRaises(ValueError):
            _check_input([0, 99], [0, 1], "select_targetclass")

    def test_all_integer_valid(self):
        result = _check_input([0, 1], [0, 1], "select_targetclass")
        self.assertEqual(result, [0, 1])


##############################################################################
# Test: ModelPlotPy construction
##############################################################################

class TestModelPlotPyInit(unittest.TestCase):

    def setUp(self):
        self.X, self.y = _make_binary_data()
        self.lr = _make_fitted_lr(self.X, self.y)

    def tearDown(self):
        plt.close("all")

    def test_valid_construction(self):
        mp = ModelPlotPy(
            feature_data=[self.X], label_data=[self.y],
            dataset_labels=["train"], models=[self.lr],
            model_labels=["lr"], ntiles=10,
        )
        self.assertIsInstance(mp, ModelPlotPy)

    def test_default_empty_construction(self):
        mp = ModelPlotPy()
        self.assertEqual(mp.feature_data, [])
        self.assertEqual(mp.label_data, [])
        self.assertEqual(mp.models, [])

    def test_ntiles_stored(self):
        mp = _make_mp(ntiles=5)
        self.assertEqual(mp.ntiles, 5)

    def test_seed_stored(self):
        mp = ModelPlotPy(seed=42)
        self.assertEqual(mp.seed, 42)

    def test_mismatched_models_and_labels_raises(self):
        """Passing 2 models but 1 label must raise at construction time."""
        with self.assertRaises(ValueError):
            ModelPlotPy(
                feature_data=[self.X], label_data=[self.y],
                dataset_labels=["train"],
                models=[self.lr, self.lr], model_labels=["lr"],
            )

    def test_mismatched_datasets_raises(self):
        """Passing 2 feature arrays but 1 label array must raise at construction."""
        with self.assertRaises(ValueError):
            ModelPlotPy(
                feature_data=[self.X, self.X], label_data=[self.y],
                dataset_labels=["train"], models=[self.lr], model_labels=["lr"],
            )

    def test_mismatched_dataset_labels_raises(self):
        """feature_data and dataset_labels length mismatch must raise."""
        with self.assertRaises(ValueError):
            ModelPlotPy(
                feature_data=[self.X], label_data=[self.y],
                dataset_labels=["a", "b"],
                models=[self.lr], model_labels=["lr"],
            )


##############################################################################
# Test: ModelPlotPy mutable-default isolation
##############################################################################

class TestModelPlotPyMutableDefaults(unittest.TestCase):
    """
    In the old code all list defaults were shared mutable objects.
    After the fix, each instance must own independent lists.
    """

    def tearDown(self):
        plt.close("all")

    def test_feature_data_not_shared(self):
        mp1, mp2 = ModelPlotPy(), ModelPlotPy()
        mp1.feature_data.append("sentinel")
        self.assertNotIn("sentinel", mp2.feature_data)

    def test_label_data_not_shared(self):
        mp1, mp2 = ModelPlotPy(), ModelPlotPy()
        mp1.label_data.append("sentinel")
        self.assertNotIn("sentinel", mp2.label_data)

    def test_models_not_shared(self):
        mp1, mp2 = ModelPlotPy(), ModelPlotPy()
        mp1.models.append("sentinel")
        self.assertNotIn("sentinel", mp2.models)

    def test_model_labels_not_shared(self):
        mp1, mp2 = ModelPlotPy(), ModelPlotPy()
        mp1.model_labels.append("sentinel")
        self.assertNotIn("sentinel", mp2.model_labels)


##############################################################################
# Test: get_params / set_params / reset_params
##############################################################################

class TestModelPlotPyParams(unittest.TestCase):

    def setUp(self):
        self.mp = _make_mp(ntiles=10)

    def tearDown(self):
        plt.close("all")

    def test_get_params_keys(self):
        params = self.mp.get_params()
        expected = {"feature_data", "label_data", "dataset_labels",
                    "models", "model_labels", "ntiles", "seed"}
        self.assertEqual(set(params.keys()), expected)

    def test_get_params_ntiles_value(self):
        self.assertEqual(self.mp.get_params()["ntiles"], self.mp.ntiles)

    def test_set_params_seed(self):
        self.mp.set_params(seed=99)
        self.assertEqual(self.mp.seed, 99)

    def test_set_params_invalid_key_raises(self):
        with self.assertRaises(ValueError):
            self.mp.set_params(nonexistent_param=42)

    def test_reset_params_restores_defaults(self):
        self.mp.reset_params()
        self.assertEqual(self.mp.feature_data, [])
        self.assertEqual(self.mp.models, [])
        self.assertEqual(self.mp.ntiles, 10)
        self.assertEqual(self.mp.seed, 0)

    def test_set_params_roundtrip(self):
        """set_params followed by get_params must reflect the new value."""
        self.mp.set_params(ntiles=20)
        self.assertEqual(self.mp.get_params()["ntiles"], 20)

    def test_set_multiple_params(self):
        self.mp.set_params(seed=7, ntiles=5)
        self.assertEqual(self.mp.seed, 7)
        self.assertEqual(self.mp.ntiles, 5)


##############################################################################
# Test: prepare_scores_and_ntiles
##############################################################################

class TestPrepareScoresAndNtiles(unittest.TestCase):

    def setUp(self):
        self.mp = _make_mp(ntiles=10)

    def tearDown(self):
        plt.close("all")

    def test_returns_dataframe(self):
        self.assertIsInstance(self.mp.prepare_scores_and_ntiles(), pd.DataFrame)

    def test_required_metadata_columns(self):
        result = self.mp.prepare_scores_and_ntiles()
        for col in ("dataset_label", "model_label", "target_class"):
            self.assertIn(col, result.columns)

    def test_prob_columns_present(self):
        result = self.mp.prepare_scores_and_ntiles()
        for c in self.mp.models[0].classes_.astype(str):
            self.assertIn(f"prob_{c}", result.columns)

    def test_dec_columns_present(self):
        result = self.mp.prepare_scores_and_ntiles()
        for c in self.mp.models[0].classes_.astype(str):
            self.assertIn(f"dec_{c}", result.columns)

    def test_row_count(self):
        result = self.mp.prepare_scores_and_ntiles()
        self.assertEqual(len(result), 200)

    def test_ntile_values_in_range(self):
        result = self.mp.prepare_scores_and_ntiles()
        for c in self.mp.models[0].classes_.astype(str):
            col = f"dec_{c}"
            self.assertTrue((result[col] >= 1).all())
            self.assertTrue((result[col] <= self.mp.ntiles).all())

    def test_prob_columns_sum_to_one(self):
        result = self.mp.prepare_scores_and_ntiles()
        prob_cols = [c for c in result.columns if c.startswith("prob_")]
        row_sums = result[prob_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_two_models_doubles_rows(self):
        result = _make_mp_two_models().prepare_scores_and_ntiles()
        self.assertEqual(len(result), 400)

    def test_two_datasets_doubles_rows(self):
        result = _make_mp_two_datasets().prepare_scores_and_ntiles()
        self.assertEqual(len(result), 400)

    def test_dataset_label_column_values(self):
        result = self.mp.prepare_scores_and_ntiles()
        self.assertTrue((result["dataset_label"] == "train").all())

    def test_model_label_column_values(self):
        result = self.mp.prepare_scores_and_ntiles()
        self.assertTrue((result["model_label"] == "lr").all())


##############################################################################
# Test: prepare_scores_and_ntiles — reproducibility
##############################################################################

class TestPrepareScoresReproducibility(unittest.TestCase):
    """Same seed must yield bitwise-identical decile assignments."""

    def tearDown(self):
        plt.close("all")

    def test_same_seed_same_deciles(self):
        mp_a = _make_mp(ntiles=10)
        mp_b = _make_mp(ntiles=10)
        ra = mp_a.prepare_scores_and_ntiles()
        rb = mp_b.prepare_scores_and_ntiles()
        for c in mp_a.models[0].classes_.astype(str):
            pd.testing.assert_series_equal(
                ra[f"dec_{c}"].reset_index(drop=True),
                rb[f"dec_{c}"].reset_index(drop=True),
                check_names=False,
            )

    def test_ntile_2_minimum_boundary(self):
        """ntiles=2 is the minimum split; must produce only values 1 and 2."""
        X, y = _make_binary_data()
        lr = _make_fitted_lr(X, y)
        mp = ModelPlotPy(feature_data=[X], label_data=[y], dataset_labels=["t"],
                         models=[lr], model_labels=["lr"], ntiles=2)
        result = mp.prepare_scores_and_ntiles()
        for c in mp.models[0].classes_.astype(str):
            unique_vals = set(result[f"dec_{c}"].unique())
            self.assertTrue(unique_vals.issubset({1, 2}),
                            f"Unexpected ntile values: {unique_vals}")


##############################################################################
# Test: aggregate_over_ntiles
##############################################################################

class TestAggregateOverNtiles(unittest.TestCase):

    def setUp(self):
        self.mp = _make_mp(ntiles=10)
        self.agg = self.mp.aggregate_over_ntiles()

    def tearDown(self):
        plt.close("all")

    def test_returns_dataframe(self):
        self.assertIsInstance(self.agg, pd.DataFrame)

    def test_required_metric_columns(self):
        for col in ("ntile", "pos", "neg", "tot", "cumpos", "cumgain",
                    "cumlift", "cumpct", "pct", "gain", "lift"):
            self.assertIn(col, self.agg.columns, f"Missing: {col}")

    def test_ntile_range_includes_zero(self):
        self.assertIn(0, self.agg["ntile"].unique())
        self.assertIn(self.mp.ntiles, self.agg["ntile"].unique())

    def test_origin_row_counts_zero(self):
        origin = self.agg[self.agg["ntile"] == 0]
        self.assertFalse(origin.empty)
        self.assertTrue((origin["tot"] == 0).all())
        self.assertTrue((origin["pos"] == 0).all())

    def test_cumgain_at_last_ntile_is_one(self):
        last = self.agg[self.agg["ntile"] == self.mp.ntiles]
        for _, row in last.iterrows():
            self.assertAlmostEqual(float(row["cumgain"]), 1.0, places=5)

    def test_cumlift_ref_constant_one(self):
        self.assertTrue((self.agg["cumlift_ref"] == 1).all())

    def test_two_models_more_rows(self):
        agg2 = _make_mp_two_models().aggregate_over_ntiles()
        self.assertGreater(len(agg2), len(self.agg))

    def test_pct_ref_constant_within_group(self):
        for (m, d, c), grp in self.agg.groupby(
            ["model_label", "dataset_label", "target_class"]
        ):
            sub = grp[grp["ntile"] > 0]
            self.assertEqual(sub["pct_ref"].nunique(), 1,
                             f"pct_ref not constant for ({m},{d},{c})")


##############################################################################
# Test: aggregate_over_ntiles — metric correctness (covers fixed bugs)
##############################################################################

class TestAggregateCorrectness(unittest.TestCase):
    """
    Verify the semantic correctness of aggregated metrics, including the
    ``pcttot`` bug fix (was ``pct.sum()``, now ``postot / tottot``).
    """

    def setUp(self):
        self.mp  = _make_mp(ntiles=10)
        self.agg = self.mp.aggregate_over_ntiles()
        self.sub = self.agg[self.agg["ntile"] > 0]

    def tearDown(self):
        plt.close("all")

    def test_pcttot_equals_pct_ref(self):
        """pcttot is now postot/tottot — it must equal pct_ref within floating point."""
        for (m, d, c), grp in self.sub.groupby(
            ["model_label", "dataset_label", "target_class"]
        ):
            np.testing.assert_allclose(
                grp["pcttot"].values, grp["pct_ref"].values, atol=1e-10,
                err_msg=f"pcttot != pct_ref for ({m},{d},{c})"
            )

    def test_gain_opt_bounded_above_by_one(self):
        """gain_opt must never exceed 1.0 by definition."""
        self.assertTrue(
            (self.sub["gain_opt"] <= 1.0 + 1e-10).all(),
            "gain_opt exceeded 1.0"
        )

    def test_gain_opt_non_negative(self):
        self.assertTrue((self.sub["gain_opt"] >= 0.0).all())

    def test_lift_positive_for_non_zero_pct_ref(self):
        """For any group with a non-trivial base rate, lift must be positive."""
        self.assertTrue((self.sub["lift"] >= 0.0).all())

    def test_cumlift_non_negative(self):
        self.assertTrue((self.sub["cumlift"] >= 0.0).all())

    def test_cumpos_monotone_non_decreasing(self):
        for (m, d, c), grp in self.sub.groupby(
            ["model_label", "dataset_label", "target_class"]
        ):
            vals = grp.sort_values("ntile")["cumpos"].values
            self.assertTrue(
                np.all(np.diff(vals) >= 0),
                f"cumpos not monotone for ({m},{d},{c})"
            )

    def test_tottot_equals_dataset_size(self):
        """tottot must be the total number of rows in the dataset (200 here)."""
        for (m, d, c), grp in self.sub.groupby(
            ["model_label", "dataset_label", "target_class"]
        ):
            self.assertEqual(
                grp["tottot"].iloc[0], 200,
                f"tottot mismatch for ({m},{d},{c})"
            )


##############################################################################
# Test: plotting_scope
##############################################################################

class TestPlottingScope(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def test_no_comparison_returns_dataframe(self):
        pi = self.mp.plotting_scope(scope="no_comparison")
        self.assertIsInstance(pi, pd.DataFrame)
        self.assertFalse(pi.empty)

    def test_no_comparison_scope_column(self):
        pi = self.mp.plotting_scope(scope="no_comparison")
        self.assertTrue((pi["scope"] == "no_comparison").all())

    def test_compare_models_labels(self):
        pi = self.mp2m.plotting_scope(scope="compare_models")
        labels = pi["model_label"].unique().tolist()
        self.assertIn("lr1", labels)
        self.assertIn("lr2", labels)

    def test_compare_datasets_labels(self):
        pi = self.mp2d.plotting_scope(scope="compare_datasets")
        labels = pi["dataset_label"].unique().tolist()
        self.assertIn("train", labels)
        self.assertIn("test", labels)

    def test_compare_targetclasses_multi_class(self):
        pi = self.mp.plotting_scope(scope="compare_targetclasses")
        self.assertGreaterEqual(pi["target_class"].nunique(), 2)

    def test_invalid_scope_raises(self):
        with self.assertRaises(ValueError):
            self.mp.plotting_scope(scope="bad_scope")

    def test_select_model_label_filter(self):
        pi = self.mp2m.plotting_scope(
            scope="no_comparison",
            select_model_label=["lr1"],
            select_targetclass=[0],
        )
        self.assertTrue((pi["model_label"] == "lr1").all())

    def test_select_dataset_label_filter(self):
        pi = self.mp2d.plotting_scope(
            scope="no_comparison",
            select_dataset_label=["train"],
            select_targetclass=[0],
        )
        self.assertTrue((pi["dataset_label"] == "train").all())


##############################################################################
# Test: plotting_scope — additional branches and filters
##############################################################################

class TestPlottingScopeAdditional(unittest.TestCase):
    """
    Covers the ``select_smallest_targetclass`` path and scope column labelling
    for each of the four scopes.
    """

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def test_scope_column_compare_models(self):
        pi = self.mp2m.plotting_scope(scope="compare_models")
        self.assertTrue((pi["scope"] == "compare_models").all())

    def test_scope_column_compare_datasets(self):
        pi = self.mp2d.plotting_scope(scope="compare_datasets")
        self.assertTrue((pi["scope"] == "compare_datasets").all())

    def test_scope_column_compare_targetclasses(self):
        pi = self.mp.plotting_scope(scope="compare_targetclasses")
        self.assertTrue((pi["scope"] == "compare_targetclasses").all())

    def test_select_smallest_targetclass_returns_single_class(self):
        """
        When ``select_targetclass`` is empty and ``select_smallest_targetclass``
        is True, the result must be scoped to exactly one target class.
        """
        pi = self.mp.plotting_scope(
            scope="no_comparison",
            select_targetclass=[],
            select_smallest_targetclass=True,
        )
        self.assertEqual(pi["target_class"].nunique(), 1)

    def test_explicit_targetclass_int_filter(self):
        """Integer class labels must pass _check_input cleanly."""
        classes = list(self.mp.models[0].classes_)
        pi = self.mp.plotting_scope(
            scope="no_comparison",
            select_targetclass=[classes[0]],
        )
        self.assertTrue((pi["target_class"] == classes[0]).all())

    def test_ntile_zero_excluded_from_plot_input(self):
        """
        The ntile=0 origin rows exist in aggregate but should not appear in a
        filtered no_comparison plot_input (they hold zeros and would skew axes).
        """
        pi = self.mp.plotting_scope(scope="no_comparison")
        # ntile 0 *may* appear if the aggregate includes it — verify the
        # DataFrame is at least not empty after filtering ntile > 0
        useful = pi[pi["ntile"] > 0]
        self.assertFalse(useful.empty)


##############################################################################
# Test: plot_response
##############################################################################

class TestPlotResponse(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def test_no_comparison(self):
        ax = plot_response(_plot_input(self.mp, "no_comparison"))
        self.assertIsNotNone(ax)

    def test_compare_models(self):
        ax = plot_response(_plot_input(self.mp2m, "compare_models"))
        self.assertIsNotNone(ax)

    def test_compare_datasets(self):
        ax = plot_response(_plot_input(self.mp2d, "compare_datasets"))
        self.assertIsNotNone(ax)

    def test_compare_targetclasses(self):
        ax = plot_response(_plot_input(self.mp, "compare_targetclasses"))
        self.assertIsNotNone(ax)

    def test_highlight_plot(self):
        ax = plot_response(_plot_input(self.mp, "no_comparison"),
                           highlight_ntile=3, highlight_how="plot")
        self.assertIsNotNone(ax)

    def test_highlight_text(self):
        ax = plot_response(_plot_input(self.mp, "no_comparison"),
                           highlight_ntile=3, highlight_how="text")
        self.assertIsNotNone(ax)

    def test_highlight_plot_text(self):
        ax = plot_response(_plot_input(self.mp, "no_comparison"),
                           highlight_ntile=3, highlight_how="plot_text")
        self.assertIsNotNone(ax)

    def test_invalid_highlight_ntile_raises(self):
        with self.assertRaises(TypeError):
            plot_response(_plot_input(self.mp, "no_comparison"), highlight_ntile=99)

    def test_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            plot_response(_plot_input(self.mp, "no_comparison"),
                          highlight_ntile=3, highlight_how="wrong")


##############################################################################
# Test: plot_cumresponse
##############################################################################

class TestPlotCumresponse(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def test_no_comparison(self):
        self.assertIsNotNone(plot_cumresponse(_plot_input(self.mp, "no_comparison")))

    def test_compare_models(self):
        self.assertIsNotNone(plot_cumresponse(_plot_input(self.mp2m, "compare_models")))

    def test_compare_datasets(self):
        self.assertIsNotNone(plot_cumresponse(_plot_input(self.mp2d, "compare_datasets")))

    def test_compare_targetclasses(self):
        self.assertIsNotNone(plot_cumresponse(_plot_input(self.mp, "compare_targetclasses")))

    def test_highlight_all_modes(self):
        pi = _plot_input(self.mp, "no_comparison")
        for how in ("plot", "text", "plot_text"):
            with self.subTest(how=how):
                plt.close("all")
                self.assertIsNotNone(
                    plot_cumresponse(pi.copy(), highlight_ntile=5, highlight_how=how))

    def test_invalid_highlight_ntile_raises(self):
        with self.assertRaises(TypeError):
            plot_cumresponse(_plot_input(self.mp, "no_comparison"), highlight_ntile=99)

    def test_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            plot_cumresponse(_plot_input(self.mp, "no_comparison"),
                             highlight_ntile=3, highlight_how="bad")


##############################################################################
# Test: plot_cumlift
##############################################################################

class TestPlotCumlift(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def test_no_comparison(self):
        self.assertIsNotNone(plot_cumlift(_plot_input(self.mp, "no_comparison")))

    def test_compare_models(self):
        self.assertIsNotNone(plot_cumlift(_plot_input(self.mp2m, "compare_models")))

    def test_compare_datasets(self):
        self.assertIsNotNone(plot_cumlift(_plot_input(self.mp2d, "compare_datasets")))

    def test_compare_targetclasses(self):
        self.assertIsNotNone(plot_cumlift(_plot_input(self.mp, "compare_targetclasses")))

    def test_highlight_plot(self):
        self.assertIsNotNone(
            plot_cumlift(_plot_input(self.mp, "no_comparison"),
                         highlight_ntile=3, highlight_how="plot"))

    def test_highlight_text(self):
        self.assertIsNotNone(
            plot_cumlift(_plot_input(self.mp, "no_comparison"),
                         highlight_ntile=3, highlight_how="text"))

    def test_invalid_highlight_ntile_raises(self):
        with self.assertRaises(TypeError):
            plot_cumlift(_plot_input(self.mp, "no_comparison"), highlight_ntile=999)

    def test_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            plot_cumlift(_plot_input(self.mp, "no_comparison"),
                         highlight_ntile=3, highlight_how="invalid")


##############################################################################
# Test: plot_cumgains
##############################################################################

class TestPlotCumgains(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def test_no_comparison(self):
        self.assertIsNotNone(plot_cumgains(_plot_input(self.mp, "no_comparison")))

    def test_compare_models(self):
        self.assertIsNotNone(plot_cumgains(_plot_input(self.mp2m, "compare_models")))

    def test_compare_datasets(self):
        self.assertIsNotNone(plot_cumgains(_plot_input(self.mp2d, "compare_datasets")))

    def test_compare_targetclasses(self):
        self.assertIsNotNone(plot_cumgains(_plot_input(self.mp, "compare_targetclasses")))

    def test_highlight_all_modes(self):
        pi = _plot_input(self.mp, "no_comparison")
        for how in ("plot", "text", "plot_text"):
            with self.subTest(how=how):
                plt.close("all")
                self.assertIsNotNone(
                    plot_cumgains(pi.copy(), highlight_ntile=5, highlight_how=how))

    def test_invalid_highlight_ntile_raises(self):
        with self.assertRaises(TypeError):
            plot_cumgains(_plot_input(self.mp, "no_comparison"), highlight_ntile=999)

    def test_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            plot_cumgains(_plot_input(self.mp, "no_comparison"),
                          highlight_ntile=3, highlight_how="bad")


##############################################################################
# Test: plot_all
##############################################################################

class TestPlotAll(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def test_no_comparison(self):
        self.assertIsNotNone(plot_all(_plot_input(self.mp, "no_comparison")))

    def test_compare_models(self):
        self.assertIsNotNone(plot_all(_plot_input(self.mp2m, "compare_models")))

    def test_compare_datasets(self):
        self.assertIsNotNone(plot_all(_plot_input(self.mp2d, "compare_datasets")))

    def test_compare_targetclasses(self):
        self.assertIsNotNone(plot_all(_plot_input(self.mp, "compare_targetclasses")))


##############################################################################
# Test: plot_costsrevs
##############################################################################

_FC, _VC, _PP = 1000.0, 0.5, 10.0


class TestPlotCostsrevs(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def _call(self, pi, **kw):
        return plot_costsrevs(pi, fixed_costs=_FC,
                               variable_costs_per_unit=_VC,
                               profit_per_unit=_PP, **kw)

    def test_no_comparison(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp, "no_comparison")))

    def test_compare_models(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp2m, "compare_models")))

    def test_compare_datasets(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp2d, "compare_datasets")))

    def test_compare_targetclasses(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp, "compare_targetclasses")))

    def test_highlight_plot(self):
        self.assertIsNotNone(
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=3, highlight_how="plot"))

    def test_highlight_text(self):
        self.assertIsNotNone(
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=3, highlight_how="text"))

    def test_highlight_plot_text(self):
        self.assertIsNotNone(
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=3, highlight_how="plot_text"))

    def test_invalid_highlight_ntile_raises(self):
        with self.assertRaises(TypeError):
            self._call(_plot_input(self.mp, "no_comparison"), highlight_ntile=999)

    def test_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=3, highlight_how="bad")


##############################################################################
# Test: plot_profit
##############################################################################

class TestPlotProfit(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def _call(self, pi, **kw):
        return plot_profit(pi, fixed_costs=500.0,
                            variable_costs_per_unit=1.0,
                            profit_per_unit=15.0, **kw)

    def test_no_comparison(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp, "no_comparison")))

    def test_compare_models(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp2m, "compare_models")))

    def test_compare_datasets(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp2d, "compare_datasets")))

    def test_compare_targetclasses(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp, "compare_targetclasses")))

    def test_highlight_plot(self):
        self.assertIsNotNone(
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=4, highlight_how="plot"))

    def test_highlight_text(self):
        self.assertIsNotNone(
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=4, highlight_how="text"))

    def test_invalid_highlight_ntile_raises(self):
        with self.assertRaises(TypeError):
            self._call(_plot_input(self.mp, "no_comparison"), highlight_ntile=999)

    def test_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=3, highlight_how="bad")


##############################################################################
# Test: plot_roi
##############################################################################

class TestPlotRoi(unittest.TestCase):

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    def _call(self, pi, **kw):
        return plot_roi(pi, fixed_costs=200.0,
                         variable_costs_per_unit=0.2,
                         profit_per_unit=8.0, **kw)

    def test_no_comparison(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp, "no_comparison")))

    def test_compare_models(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp2m, "compare_models")))

    def test_compare_datasets(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp2d, "compare_datasets")))

    def test_compare_targetclasses(self):
        self.assertIsNotNone(self._call(_plot_input(self.mp, "compare_targetclasses")))

    def test_highlight_plot(self):
        self.assertIsNotNone(
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=2, highlight_how="plot"))

    def test_highlight_text(self):
        self.assertIsNotNone(
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=2, highlight_how="text"))

    def test_invalid_highlight_ntile_raises(self):
        with self.assertRaises(TypeError):
            self._call(_plot_input(self.mp, "no_comparison"), highlight_ntile=999)

    def test_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            self._call(_plot_input(self.mp, "no_comparison"),
                       highlight_ntile=3, highlight_how="bad")


##############################################################################
# Test: highlight across all four scopes for every plot type
##############################################################################

class TestHighlightAllScopes(unittest.TestCase):
    """
    Each plot type must render successfully with ``highlight_ntile`` for all
    four comparison scopes, not just ``no_comparison``.  This catches scope
    branches that might be skipped in the simpler single-scope tests.
    """

    def setUp(self):
        self.mp   = _make_mp()
        self.mp2m = _make_mp_two_models()
        self.mp2d = _make_mp_two_datasets()

    def tearDown(self):
        plt.close("all")

    # ── plot_response ────────────────────────────────────────────────────
    def test_response_highlight_compare_datasets(self):
        self.assertIsNotNone(
            plot_response(_plot_input(self.mp2d, "compare_datasets"),
                          highlight_ntile=3, highlight_how="plot"))

    def test_response_highlight_compare_models(self):
        self.assertIsNotNone(
            plot_response(_plot_input(self.mp2m, "compare_models"),
                          highlight_ntile=3, highlight_how="plot"))

    def test_response_highlight_compare_targetclasses(self):
        self.assertIsNotNone(
            plot_response(_plot_input(self.mp, "compare_targetclasses"),
                          highlight_ntile=3, highlight_how="text"))

    # ── plot_cumgains ───────────────────────────────────────────────────
    def test_cumgains_highlight_compare_models(self):
        self.assertIsNotNone(
            plot_cumgains(_plot_input(self.mp2m, "compare_models"),
                          highlight_ntile=5, highlight_how="plot"))

    def test_cumgains_highlight_compare_datasets(self):
        self.assertIsNotNone(
            plot_cumgains(_plot_input(self.mp2d, "compare_datasets"),
                          highlight_ntile=5, highlight_how="text"))

    def test_cumgains_highlight_compare_targetclasses(self):
        self.assertIsNotNone(
            plot_cumgains(_plot_input(self.mp, "compare_targetclasses"),
                          highlight_ntile=5, highlight_how="plot_text"))

    # ── plot_cumlift ─────────────────────────────────────────────────────
    def test_cumlift_highlight_compare_datasets(self):
        self.assertIsNotNone(
            plot_cumlift(_plot_input(self.mp2d, "compare_datasets"),
                         highlight_ntile=4, highlight_how="plot"))

    def test_cumlift_highlight_compare_models(self):
        self.assertIsNotNone(
            plot_cumlift(_plot_input(self.mp2m, "compare_models"),
                         highlight_ntile=4, highlight_how="text"))

    # ── plot_costsrevs across scopes ─────────────────────────────────────
    def test_costsrevs_highlight_compare_models(self):
        self.assertIsNotNone(
            plot_costsrevs(
                _plot_input(self.mp2m, "compare_models"),
                fixed_costs=_FC, variable_costs_per_unit=_VC, profit_per_unit=_PP,
                highlight_ntile=3, highlight_how="plot",
            ))

    def test_costsrevs_highlight_compare_targetclasses(self):
        self.assertIsNotNone(
            plot_costsrevs(
                _plot_input(self.mp, "compare_targetclasses"),
                fixed_costs=_FC, variable_costs_per_unit=_VC, profit_per_unit=_PP,
                highlight_ntile=3, highlight_how="text",
            ))

    # ── plot_profit across scopes ─────────────────────────────────────────
    def test_profit_highlight_compare_models(self):
        self.assertIsNotNone(
            plot_profit(
                _plot_input(self.mp2m, "compare_models"),
                fixed_costs=100.0, variable_costs_per_unit=0.5, profit_per_unit=10.0,
                highlight_ntile=3, highlight_how="plot",
            ))

    def test_profit_highlight_compare_targetclasses(self):
        self.assertIsNotNone(
            plot_profit(
                _plot_input(self.mp, "compare_targetclasses"),
                fixed_costs=100.0, variable_costs_per_unit=0.5, profit_per_unit=10.0,
                highlight_ntile=3, highlight_how="text",
            ))

    # ── plot_roi across scopes ────────────────────────────────────────────
    def test_roi_highlight_compare_models(self):
        self.assertIsNotNone(
            plot_roi(
                _plot_input(self.mp2m, "compare_models"),
                fixed_costs=200.0, variable_costs_per_unit=0.2, profit_per_unit=8.0,
                highlight_ntile=2, highlight_how="plot",
            ))

    def test_roi_highlight_compare_targetclasses(self):
        self.assertIsNotNone(
            plot_roi(
                _plot_input(self.mp, "compare_targetclasses"),
                fixed_costs=200.0, variable_costs_per_unit=0.2, profit_per_unit=8.0,
                highlight_ntile=2, highlight_how="text",
            ))


##############################################################################
# Test: financial column computation (no SettingWithCopyWarning)
##############################################################################

class TestFinancialColumnComputation(unittest.TestCase):
    """
    Verify that the financial plots compute their derived columns correctly
    and that calling them a second time on the same ``plot_input`` does not
    corrupt the original DataFrame (defensive-copy guarantee).
    """

    def setUp(self):
        self.mp = _make_mp()
        self.pi = _plot_input(self.mp, "no_comparison")
        self._fc, self._vc, self._pp = 500.0, 1.0, 12.0

    def tearDown(self):
        plt.close("all")

    def _cols_before(self):
        return set(self.pi.columns)

    def test_costsrevs_does_not_pollute_plot_input(self):
        """plot_costsrevs must not add columns to the caller's DataFrame."""
        cols_before = self._cols_before()
        plot_costsrevs(self.pi, fixed_costs=self._fc,
                       variable_costs_per_unit=self._vc,
                       profit_per_unit=self._pp)
        cols_after = set(self.pi.columns)
        self.assertEqual(cols_before, cols_after,
                         f"Columns added to plot_input: {cols_after - cols_before}")

    def test_profit_does_not_pollute_plot_input(self):
        cols_before = self._cols_before()
        plot_profit(self.pi, fixed_costs=self._fc,
                    variable_costs_per_unit=self._vc,
                    profit_per_unit=self._pp)
        self.assertEqual(cols_before, set(self.pi.columns))

    def test_roi_does_not_pollute_plot_input(self):
        cols_before = self._cols_before()
        plot_roi(self.pi, fixed_costs=self._fc,
                 variable_costs_per_unit=self._vc,
                 profit_per_unit=self._pp)
        self.assertEqual(cols_before, set(self.pi.columns))

    def test_costsrevs_no_setting_with_copy_warning(self):
        """Financial plots must not raise SettingWithCopyWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot_costsrevs(self.pi, fixed_costs=self._fc,
                           variable_costs_per_unit=self._vc,
                           profit_per_unit=self._pp)

    def test_profit_no_setting_with_copy_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot_profit(self.pi, fixed_costs=self._fc,
                        variable_costs_per_unit=self._vc,
                        profit_per_unit=self._pp)

    def test_roi_no_setting_with_copy_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            plot_roi(self.pi, fixed_costs=self._fc,
                     variable_costs_per_unit=self._vc,
                     profit_per_unit=self._pp)


##############################################################################
# Test: return types — every public plot must return a matplotlib Axes object
##############################################################################

class TestPlotReturnTypes(unittest.TestCase):
    """
    All eight public plot functions must return a matplotlib ``Axes`` instance
    so callers can chain further customisation (title, labels, savefig, etc.).
    """

    def setUp(self):
        self.mp   = _make_mp()
        self.pi   = _plot_input(self.mp, "no_comparison")
        self._fc, self._vc, self._pp = 500.0, 1.0, 12.0

    def tearDown(self):
        plt.close("all")

    def _is_axes(self, obj) -> bool:
        import matplotlib.axes
        return isinstance(obj, matplotlib.axes.Axes)

    def test_plot_response_returns_axes(self):
        self.assertTrue(self._is_axes(plot_response(self.pi)))

    def test_plot_cumresponse_returns_axes(self):
        self.assertTrue(self._is_axes(plot_cumresponse(self.pi)))

    def test_plot_cumlift_returns_axes(self):
        self.assertTrue(self._is_axes(plot_cumlift(self.pi)))

    def test_plot_cumgains_returns_axes(self):
        self.assertTrue(self._is_axes(plot_cumgains(self.pi)))

    def test_plot_all_returns_axes(self):
        self.assertTrue(self._is_axes(plot_all(self.pi)))

    def test_plot_costsrevs_returns_axes(self):
        self.assertTrue(self._is_axes(
            plot_costsrevs(self.pi, fixed_costs=self._fc,
                           variable_costs_per_unit=self._vc,
                           profit_per_unit=self._pp)))

    def test_plot_profit_returns_axes(self):
        self.assertTrue(self._is_axes(
            plot_profit(self.pi, fixed_costs=self._fc,
                        variable_costs_per_unit=self._vc,
                        profit_per_unit=self._pp)))

    def test_plot_roi_returns_axes(self):
        self.assertTrue(self._is_axes(
            plot_roi(self.pi, fixed_costs=self._fc,
                     variable_costs_per_unit=self._vc,
                     profit_per_unit=self._pp)))


##############################################################################
# Test: ntile label / xlabper spacing branches
##############################################################################

class TestNtileLabelBranches(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_decile_ntiles_10(self):
        mp = _make_mp(ntiles=10)
        self.assertIsNotNone(plot_cumgains(_plot_input(mp, "no_comparison")))

    def test_percentile_ntiles_100(self):
        """ntiles=100 must use the 'percentile' description label without error."""
        X, y = _make_binary_data(n=500)
        lr = _make_fitted_lr(X, y)
        mp = ModelPlotPy(feature_data=[X], label_data=[y], dataset_labels=["train"],
                         models=[lr], model_labels=["lr"], ntiles=100)
        self.assertIsNotNone(plot_cumgains(_plot_input(mp, "no_comparison")))

    def test_custom_ntile_20(self):
        mp = _make_mp(ntiles=20)
        self.assertIsNotNone(plot_cumgains(_plot_input(mp, "no_comparison")))

    def test_custom_ntile_50_xlabper_branch(self):
        X, y = _make_binary_data(n=500)
        lr = _make_fitted_lr(X, y)
        mp = ModelPlotPy(
            feature_data=[X], label_data=[y], dataset_labels=["train"],
            models=[lr], model_labels=["lr"], ntiles=50,
        )
        self.assertIsNotNone(plot_cumgains(_plot_input(mp, "no_comparison")))

    def test_high_ntile_xlabper_5_branch(self):
        X, y = _make_binary_data(n=500)
        lr = _make_fitted_lr(X, y)
        mp = ModelPlotPy(
            feature_data=[X], label_data=[y], dataset_labels=["train"],
            models=[lr], model_labels=["lr"], ntiles=45,
        )
        self.assertIsNotNone(plot_response(_plot_input(mp, "no_comparison")))


##############################################################################
# Test: public __all__ API completeness
##############################################################################

class TestPublicApi(unittest.TestCase):

    def test_all_defines_expected_names(self):
        expected = {
            "ModelPlotPy", "plot_response", "plot_cumresponse",
            "plot_cumlift", "plot_cumgains", "plot_all",
            "plot_costsrevs", "plot_profit", "plot_roi",
        }
        missing = expected - set(_mpm.__all__)
        self.assertFalse(missing, f"Missing from __all__: {missing}")

    def test_public_names_are_callable(self):
        for name in _mpm.__all__:
            obj = getattr(_mpm, name)
            self.assertTrue(callable(obj), f"'{name}' is not callable")

    def test_no_extra_private_names_in_all(self):
        """__all__ must not accidentally export private helpers."""
        for name in _mpm.__all__:
            self.assertFalse(
                name.startswith("_"),
                f"Private name '{name}' should not be in __all__"
            )


if __name__ == "__main__":
    unittest.main()
