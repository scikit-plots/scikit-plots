# scikitplot/seaborn/tests/test__decile.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for scikitplot.seaborn._decile (decileplot / _DecilePlotter).

Coverage
--------
- print_labels / get_feature_infos: structure, key presence, JSON output.
- _prepare_subset: all validation rules added by Bug #6 fix (binary, range,
  finite, weight guards).
- compute_decile_table: column completeness (rate_resp_pct Bug #9), monotonic
  cumulative columns, count identities, division-by-zero guards (Bug #7).
- decileplot: all kind values, returns correct type, early-exit, n_deciles param.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._decile import (
    _DecilePlotter,
    decileplot,
    get_feature_infos,
    print_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=200, seed=42, prevalence=0.3):
    rng = np.random.default_rng(seed)
    y_true = (rng.uniform(size=n) < prevalence).astype(int)
    y_score = np.clip(y_true * 0.5 + rng.uniform(0, 0.5, n), 0.0, 1.0)
    return y_true, y_score.astype(float)


def _make_df(n=200, seed=42, prevalence=0.3):
    y_true, y_score = _make_data(n, seed, prevalence)
    return pd.DataFrame({"x": y_true, "y": y_score})


class _Base(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def _p(self):
        return _DecilePlotter()

    def _sub(self, x, y, weights=None):
        d = {"x": x, "y": y}
        if weights is not None:
            d["weights"] = weights
        return pd.DataFrame(d)


# ===========================================================================
# 1. get_feature_infos / print_labels
# ===========================================================================

class TestGetFeatureInfos(_Base):

    EXPECTED_KEYS = [
        "decile", "prob_min", "prob_max", "prob_avg",
        "cnt_resp_true", "cnt_resp_false", "cnt_resp_total",
        "cnt_resp_rndm_true", "cnt_resp_wiz_true",
        "rate_resp", "overall_rate",
        "cum_resp_true", "cum_resp_true_pct",
        "cum_resp_false", "cum_resp_false_pct",
        "cum_resp_total", "cum_resp_total_pct",
        "cumulative_lift", "decile_wise_lift", "KS",
    ]

    def test_returns_dict(self):
        self.assertIsInstance(get_feature_infos(), dict)

    def test_expected_keys_present(self):
        info = get_feature_infos()
        for key in self.EXPECTED_KEYS:
            self.assertIn(key, info, f"Missing key: {key}")

    def test_all_values_non_empty_strings(self):
        for k, v in get_feature_infos().items():
            self.assertIsInstance(v, str)
            self.assertGreater(len(v), 0, f"Empty value for key: {k}")

    def test_print_labels_json_valid(self):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_labels(as_json=True, indent=2)
        finally:
            sys.stdout = old_stdout
        parsed = json.loads(buf.getvalue())
        self.assertIsInstance(parsed, dict)

    def test_print_labels_pprint_nonempty(self):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            print_labels(as_json=False)
        finally:
            sys.stdout = old_stdout
        self.assertGreater(len(buf.getvalue()), 0)


# ===========================================================================
# 2. _prepare_subset — Bug #6 validations
# ===========================================================================

class TestPrepareSubset(_Base):

    def test_correct_types(self):
        y_true, y_score, w = self._p()._prepare_subset(
            self._sub([0, 1, 0, 1], [0.1, 0.9, 0.3, 0.7])
        )
        self.assertIsInstance(y_true, np.ndarray)
        self.assertIsInstance(y_score, np.ndarray)
        self.assertIsNone(w)

    def test_y_true_dtype_int(self):
        y_true, _, _ = self._p()._prepare_subset(self._sub([0.0, 1.0], [0.2, 0.8]))
        self.assertTrue(np.issubdtype(y_true.dtype, np.integer))

    def test_empty_returns_none_triple(self):
        sub = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
        self.assertEqual(self._p()._prepare_subset(sub), (None, None, None))

    def test_nan_rows_dropped(self):
        sub = self._sub([0, np.nan, 1], [0.1, 0.5, 0.9])
        y_true, y_score, _ = self._p()._prepare_subset(sub)
        self.assertEqual(len(y_true), 2)

    def test_weights_extracted(self):
        sub = self._sub([0, 1, 0], [0.1, 0.9, 0.2], [1.0, 2.0, 3.0])
        _, _, w = self._p()._prepare_subset(sub)
        self.assertIsNotNone(w)
        np.testing.assert_array_equal(w, [1.0, 2.0, 3.0])

    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(self._sub([0, 1], [0.2, 0.8], [-1.0, 1.0]))
        self.assertIn("non-negative", str(ctx.exception))

    def test_inf_weight_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(self._sub([0, 1], [0.2, 0.8], [np.inf, 1.0]))
        self.assertIn("NaN or infinite", str(ctx.exception))

    def test_non_binary_y_true_raises(self):
        """Bug #6: multiclass labels must now raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(self._sub([0, 1, 2], [0.1, 0.5, 0.9]))
        self.assertIn("binary", str(ctx.exception).lower())

    def test_y_score_above_1_raises(self):
        """Bug #6: scores > 1 must raise."""
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(self._sub([0, 1], [0.5, 1.5]))
        self.assertIn("[0, 1]", str(ctx.exception))

    def test_y_score_below_0_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(self._sub([0, 1], [-0.1, 0.8]))
        self.assertIn("[0, 1]", str(ctx.exception))

    def test_inf_in_y_true_raises(self):
        with self.assertRaises(ValueError):
            self._p()._prepare_subset(self._sub([0, np.inf], [0.2, 0.8]))

    def test_inf_in_y_score_raises(self):
        with self.assertRaises(ValueError):
            self._p()._prepare_subset(self._sub([0, 1], [0.2, np.inf]))


# ===========================================================================
# 3. compute_decile_table
# ===========================================================================

class TestComputeDecileTable(_Base):

    # All columns the table must contain (rate_resp_pct is Bug #9)
    REQUIRED_COLS = [
        "decile", "prob_min", "prob_max", "prob_avg",
        "cnt_resp_total", "cnt_resp_true", "cnt_resp_false",
        "cnt_resp_rndm_true", "cnt_resp_wiz_true",
        "rate_resp", "rate_resp_pct",
        "overall_rate",
        "cum_resp_total", "cum_resp_total_pct",
        "cum_resp_true", "cum_resp_true_pct",
        "cum_resp_false", "cum_resp_false_pct",
        "cumulative_lift", "decile_wise_lift", "KS",
    ]

    def _dt(self, n=200, seed=42, n_deciles=10, prevalence=0.3):
        y_true, y_score = _make_data(n, seed, prevalence)
        return self._p().compute_decile_table(y_true, y_score, n_deciles=n_deciles)

    def test_returns_dataframe(self):
        self.assertIsInstance(self._dt(), pd.DataFrame)

    def test_row_count_equals_n_deciles(self):
        self.assertEqual(len(self._dt(n_deciles=10)), 10)

    def test_all_required_columns_present(self):
        dt = self._dt()
        for col in self.REQUIRED_COLS:
            self.assertIn(col, dt.columns, f"Missing column: {col}")

    def test_rate_resp_pct_equals_rate_resp(self):
        """Bug #9: rate_resp_pct must equal rate_resp (alias)."""
        dt = self._dt()
        pd.testing.assert_series_equal(
            dt["rate_resp"], dt["rate_resp_pct"], check_names=False
        )

    def test_cumulative_true_monotonically_nondecreasing(self):
        dt = self._dt()
        diffs = np.diff(dt["cum_resp_true"].values)
        self.assertTrue((diffs >= 0).all(),
                        f"cum_resp_true is not monotone: {diffs}")

    def test_cum_resp_total_pct_final_value_100(self):
        dt = self._dt(n=500)
        self.assertAlmostEqual(float(dt["cum_resp_total_pct"].iloc[-1]), 100.0, places=5)

    def test_cnt_true_plus_false_equals_total(self):
        dt = self._dt()
        np.testing.assert_array_equal(
            dt["cnt_resp_true"] + dt["cnt_resp_false"],
            dt["cnt_resp_total"],
        )

    def test_total_true_sum_matches_input(self):
        y_true, y_score = _make_data()
        dt = self._p().compute_decile_table(y_true, y_score, n_deciles=10)
        self.assertEqual(dt["cnt_resp_true"].sum(), y_true.sum())

    def test_ks_bounded_0_to_100(self):
        dt = self._dt()
        self.assertTrue((dt["KS"].abs() <= 100.0 + 1e-9).all())

    def test_custom_n_deciles_5(self):
        self.assertEqual(len(self._dt(n=500, n_deciles=5)), 5)

    def test_all_positive_warns_not_raises(self):
        """Bug #7: all-positive dataset must warn, not ZeroDivisionError."""
        y_true = np.ones(100, dtype=int)
        y_score = np.linspace(0.1, 0.9, 100)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt = self._p().compute_decile_table(y_true, y_score, n_deciles=10)
        self.assertIsInstance(dt, pd.DataFrame)
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any("negative" in m.lower() or "NaN" in m or "positive" in m.lower()
                for m in messages),
            f"Expected a warning for all-positive dataset; got: {messages}"
        )

    def test_all_negative_warns_not_raises(self):
        """Bug #7: all-negative dataset must warn, not ZeroDivisionError."""
        y_true = np.zeros(100, dtype=int)
        y_score = np.linspace(0.1, 0.9, 100)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt = self._p().compute_decile_table(y_true, y_score, n_deciles=10)
        self.assertIsInstance(dt, pd.DataFrame)
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any("positive" in m.lower() or "NaN" in m for m in messages),
            f"Expected a warning for all-negative dataset; got: {messages}"
        )

    def test_prob_avg_within_0_1(self):
        dt = self._dt()
        self.assertTrue((dt["prob_avg"] >= 0.0).all())
        self.assertTrue((dt["prob_avg"] <= 1.0).all())

    def test_overall_rate_constant_across_rows(self):
        dt = self._dt()
        # overall_rate is a scalar broadcast; all values should be equal
        vals = dt["overall_rate"].values
        self.assertAlmostEqual(float(vals.max() - vals.min()), 0.0, places=10)


# ===========================================================================
# 4. decileplot public API
# ===========================================================================

class TestDecileplot(_Base):

    def test_kind_df_returns_dataframe(self):
        result = decileplot(_make_df(), x="x", y="y", kind="df")
        self.assertIsInstance(result, pd.DataFrame)

    def test_kind_df_has_decile_col(self):
        dt = decileplot(_make_df(), x="x", y="y", kind="df")
        self.assertIn("decile", dt.columns)

    def test_kind_df_has_cumulative_lift(self):
        dt = decileplot(_make_df(), x="x", y="y", kind="df")
        self.assertIn("cumulative_lift", dt.columns)

    def test_kind_report_returns_axes(self):
        ax = decileplot(_make_df(), x="x", y="y", kind="report")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_cumulative_lift_returns_axes(self):
        ax = decileplot(_make_df(), x="x", y="y", kind="cumulative_lift")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_decile_wise_lift_returns_axes(self):
        ax = decileplot(_make_df(), x="x", y="y", kind="decile_wise_lift")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_cumulative_gain_returns_axes(self):
        ax = decileplot(_make_df(), x="x", y="y", kind="cumulative_gain")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_decile_wise_gain_returns_axes(self):
        ax = decileplot(_make_df(), x="x", y="y", kind="decile_wise_gain")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_cumulative_response_returns_axes(self):
        ax = decileplot(_make_df(), x="x", y="y", kind="cumulative_response")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_ks_statistic_returns_axes(self):
        ax = decileplot(_make_df(), x="x", y="y", kind="ks_statistic")
        self.assertIsInstance(ax, plt.Axes)

    def test_invalid_kind_raises(self):
        with self.assertRaises(Exception):
            decileplot(_make_df(), x="x", y="y", kind="bad_kind")

    def test_no_xy_returns_ax_early(self):
        fig, ax = plt.subplots()
        result = decileplot(ax=ax)
        self.assertIs(result, ax)

    def test_ax_param_respected(self):
        fig, ax = plt.subplots()
        result = decileplot(_make_df(), x="x", y="y", kind="cumulative_lift", ax=ax)
        self.assertIs(result, ax)

    def test_accepts_numpy_arrays(self):
        y_true, y_score = _make_data()
        ax = decileplot(x=y_true, y=y_score, kind="cumulative_lift")
        self.assertIsInstance(ax, plt.Axes)

    def test_n_deciles_5(self):
        dt = decileplot(_make_df(500), x="x", y="y", kind="df", n_deciles=5)
        self.assertEqual(len(dt), 5)

    def test_kind_case_insensitive(self):
        ax = decileplot(_make_df(), x="x", y="y", kind="Cumulative_Lift")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_none_defaults_to_report(self):
        ax = decileplot(_make_df(), x="x", y="y")
        self.assertIsInstance(ax, plt.Axes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
