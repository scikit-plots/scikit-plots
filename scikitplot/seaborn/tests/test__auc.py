# scikitplot/seaborn/tests/test__auc.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for scikitplot.seaborn._auc (aucplot / _AucPlotter).

Coverage
--------
- _prepare_subset : type coercion, NaN drops, weight validation,
  probability range, binary label enforcement.
- _validate_curve_inputs : static validator invariants.
- _compute_pr : AP in [0,1], recall starts at 0, sample_weight path.
- _compute_roc : AUC in [0,1], fpr starts at 0, error message says "ROC" (Bug #2).
- _plot_pr : drawstyle default and override (Bug #3).
- aucplot public API : kind normalisation, returns Axes, hue, legend, labels.
"""

from __future__ import annotations

import io
import sys
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._auc import _AucPlotter, aucplot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary(n=200, seed=42):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(int)
    y_score = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, size=n), 0.0, 1.0)
    return y_true, y_score.astype(float)


def _make_df(n=200, seed=42):
    y_true, y_score = _make_binary(n, seed)
    return pd.DataFrame({"x": y_true, "y": y_score})


# ---------------------------------------------------------------------------
# Base: auto close figures
# ---------------------------------------------------------------------------

class _Base(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def _p(self):
        return _AucPlotter()

    def _sub(self, x, y, weights=None):
        d = {"x": x, "y": y}
        if weights is not None:
            d["weights"] = weights
        return pd.DataFrame(d)


# ===========================================================================
# 1. _prepare_subset
# ===========================================================================

class TestPrepareSubset(_Base):

    def test_returns_ndarray_and_none_weights(self):
        y_true, y_score, w = self._p()._prepare_subset(
            self._sub([0, 1, 0, 1], [0.1, 0.9, 0.3, 0.7])
        )
        self.assertIsInstance(y_true, np.ndarray)
        self.assertIsInstance(y_score, np.ndarray)
        self.assertIsNone(w)

    def test_y_true_dtype_int(self):
        y_true, _, _ = self._p()._prepare_subset(self._sub([0.0, 1.0], [0.2, 0.8]))
        self.assertTrue(np.issubdtype(y_true.dtype, np.integer))

    def test_y_score_dtype_float(self):
        _, y_score, _ = self._p()._prepare_subset(self._sub([0, 1], [0, 1]))
        self.assertTrue(np.issubdtype(y_score.dtype, np.floating))

    def test_empty_after_dropna_returns_none_triple(self):
        sub = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
        self.assertEqual(self._p()._prepare_subset(sub), (None, None, None))

    def test_nan_rows_dropped(self):
        sub = self._sub([0, np.nan, 1], [0.1, 0.5, 0.9])
        y_true, y_score, _ = self._p()._prepare_subset(sub)
        self.assertEqual(len(y_true), 2)
        self.assertEqual(len(y_score), 2)

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

    def test_y_score_above_1_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(self._sub([0, 1], [0.5, 1.5]))
        self.assertIn("[0, 1]", str(ctx.exception))

    def test_y_score_below_0_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(self._sub([0, 1], [-0.1, 0.8]))
        self.assertIn("[0, 1]", str(ctx.exception))


# ===========================================================================
# 2. _validate_curve_inputs (static)
# ===========================================================================

class TestValidateCurveInputs(_Base):

    def test_valid_inputs_pass(self):
        yt, ys, sw = _AucPlotter._validate_curve_inputs(
            np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.3, 0.7]), None
        )
        self.assertEqual(yt.shape, (4,))
        self.assertEqual(ys.shape, (4,))
        self.assertIsNone(sw)

    def test_non_binary_y_true_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _AucPlotter._validate_curve_inputs(
                np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]), None
            )
        self.assertIn("binary", str(ctx.exception))

    def test_negative_sample_weight_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _AucPlotter._validate_curve_inputs(
                np.array([0, 1]), np.array([0.2, 0.8]), np.array([-1.0, 1.0])
            )
        self.assertIn("non-negative", str(ctx.exception))

    def test_length_mismatch_raises(self):
        with self.assertRaises(Exception):
            _AucPlotter._validate_curve_inputs(
                np.array([0, 1, 0]), np.array([0.2, 0.8]), None
            )


# ===========================================================================
# 3. _compute_pr
# ===========================================================================

class TestComputePR(_Base):

    def test_returns_three_items(self):
        y_true, y_score = _make_binary()
        recall, precision, ap = _AucPlotter._compute_pr(y_true, y_score, None)
        self.assertIsNotNone(recall)
        self.assertIsNotNone(precision)
        self.assertIsInstance(ap, float)

    def test_ap_in_unit_interval(self):
        y_true, y_score = _make_binary()
        _, _, ap = _AucPlotter._compute_pr(y_true, y_score, None)
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)

    def test_recall_anchor_at_zero(self):
        # sklearn.precision_recall_curve appends the (recall=0, precision=1)
        # anchor as the LAST element (index -1), not the first.
        # See sklearn docs: "The last precision and recall values are 1. and 0."
        y_true, y_score = _make_binary()
        recall, precision, _ = _AucPlotter._compute_pr(y_true, y_score, None)
        self.assertAlmostEqual(float(recall[-1]), 0.0, places=9)
        self.assertAlmostEqual(float(precision[-1]), 1.0, places=9)

    def test_with_sample_weight(self):
        y_true, y_score = _make_binary(100)
        recall, precision, ap = _AucPlotter._compute_pr(y_true, y_score, np.ones(100))
        self.assertGreater(ap, 0.0)

    def test_degenerate_score_returns_three_items(self):
        result = _AucPlotter._compute_pr(
            np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5]), None
        )
        self.assertEqual(len(result), 3)


# ===========================================================================
# 4. _compute_roc  — Bug #2: warning must say "ROC" not "PR"
# ===========================================================================

class TestComputeROC(_Base):

    def test_returns_three_items(self):
        y_true, y_score = _make_binary()
        fpr, tpr, roc_auc = _AucPlotter._compute_roc(y_true, y_score, None)
        self.assertIsNotNone(fpr)
        self.assertIsNotNone(tpr)
        self.assertIsInstance(roc_auc, float)

    def test_auc_in_unit_interval(self):
        y_true, y_score = _make_binary()
        _, _, roc_auc = _AucPlotter._compute_roc(y_true, y_score, None)
        self.assertGreaterEqual(roc_auc, 0.0)
        self.assertLessEqual(roc_auc, 1.0)

    def test_fpr_starts_at_zero(self):
        y_true, y_score = _make_binary()
        fpr, _, _ = _AucPlotter._compute_roc(y_true, y_score, None)
        self.assertAlmostEqual(float(fpr[0]), 0.0, places=9)

    def test_with_sample_weight(self):
        y_true, y_score = _make_binary(100)
        fpr, tpr, roc_auc = _AucPlotter._compute_roc(y_true, y_score, np.ones(100))
        self.assertGreater(roc_auc, 0.0)

    def test_warning_message_says_roc_not_pr_curve(self):
        """Bug #2: _compute_roc warning must say 'ROC', never 'PR curve'."""
        # Force a failure: all-NaN scores cause sklearn to raise.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _AucPlotter._compute_roc(
                np.array([0, 1]), np.array([np.nan, np.nan]), None
            )
        if caught:
            for w in caught:
                msg = str(w.message)
                self.assertNotIn(
                    "PR curve", msg,
                    f"_compute_roc warning incorrectly mentions 'PR curve': {msg!r}"
                )
            self.assertTrue(
                any("ROC" in str(w.message) for w in caught),
                f"Expected 'ROC' in warning; got: {[str(w.message) for w in caught]}"
            )


# ===========================================================================
# 5. drawstyle — Bug #3
# ===========================================================================

class TestDrawstyle(_Base):

    def test_pr_default_drawstyle_is_steps_post(self):
        """Default drawstyle on PR curve must be 'steps-post'."""
        fig, ax = plt.subplots()
        y_true, y_score = _make_binary(100)
        p = self._p()
        p._plot_pr(
            y_true=y_true, y_score=y_score, sample_weight=None,
            classes=np.array([0, 1]), label_base="M", legend=False,
            ax=ax, fmt=".4g", digits=4, baseline=False,
        )
        lines = ax.get_lines()
        self.assertGreater(len(lines), 0)
        self.assertEqual(
            lines[0].get_drawstyle(), "steps-post",
            f"Expected 'steps-post', got '{lines[0].get_drawstyle()}'"
        )

    def test_pr_drawstyle_overridable(self):
        """Bug #3: caller must be able to override drawstyle to 'default'."""
        fig, ax = plt.subplots()
        y_true, y_score = _make_binary(100)
        p = self._p()
        p._plot_pr(
            y_true=y_true, y_score=y_score, sample_weight=None,
            classes=np.array([0, 1]), label_base="M", legend=False,
            ax=ax, fmt=".4g", digits=4, baseline=False,
            drawstyle="default",
        )
        lines = ax.get_lines()
        self.assertGreater(len(lines), 0)
        self.assertEqual(
            lines[0].get_drawstyle(), "default",
            f"drawstyle override failed; got '{lines[0].get_drawstyle()}'"
        )


# ===========================================================================
# 6. aucplot public API
# ===========================================================================

class TestAucplot(_Base):

    def test_returns_axes_roc(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="roc")
        self.assertIsInstance(ax, plt.Axes)

    def test_returns_axes_pr(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="pr")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_none_defaults_to_roc(self):
        fig, ax = plt.subplots()
        result = aucplot(_make_df(), x="x", y="y", ax=ax)
        self.assertIs(result, ax)

    def test_kind_uppercase_accepted(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="ROC")
        self.assertIsInstance(ax, plt.Axes)

    def test_invalid_kind_raises(self):
        with self.assertRaises(Exception):
            aucplot(_make_df(), x="x", y="y", kind="bad_kind")

    def test_no_xy_returns_ax_immediately_no_lines(self):
        fig, ax = plt.subplots()
        result = aucplot(ax=ax)
        self.assertIs(result, ax)
        self.assertEqual(len(ax.get_lines()), 0)

    def test_accepts_numpy_arrays(self):
        y_true, y_score = _make_binary()
        ax = aucplot(x=y_true, y=y_score, kind="roc")
        self.assertIsInstance(ax, plt.Axes)

    def test_accepts_pandas_series(self):
        y_true, y_score = _make_binary(100)
        ax = aucplot(x=pd.Series(y_true), y=pd.Series(y_score), kind="pr")
        self.assertIsInstance(ax, plt.Axes)

    def test_roc_xlabel_contains_fpr(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="roc")
        self.assertIn("False Positive Rate", ax.get_xlabel())

    def test_roc_ylabel_contains_tpr(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="roc")
        self.assertIn("True Positive Rate", ax.get_ylabel())

    def test_pr_xlabel_contains_recall(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="pr")
        self.assertIn("Recall", ax.get_xlabel())

    def test_pr_ylabel_contains_precision(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="pr")
        self.assertIn("Precision", ax.get_ylabel())

    def test_hue_produces_multiple_lines(self):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "x": rng.integers(0, 2, n),
            "y": np.clip(rng.uniform(0, 1, n), 0, 1),
            "model": ["A"] * (n // 2) + ["B"] * (n // 2),
        })
        ax = aucplot(df, x="x", y="y", hue="model", kind="roc")
        self.assertGreaterEqual(len(ax.get_lines()), 2)

    def test_legend_false_no_legend(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="roc", legend=False)
        self.assertIsNone(ax.get_legend())

    def test_ax_param_respected(self):
        fig, ax = plt.subplots()
        result = aucplot(_make_df(), x="x", y="y", kind="roc", ax=ax)
        self.assertIs(result, ax)

    def test_fill_true_does_not_crash(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="pr", fill=True)
        self.assertIsInstance(ax, plt.Axes)

    def test_auc_in_legend_text_roc(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="roc", legend=True)
        legend = ax.get_legend()
        if legend is not None:
            text = " ".join(t.get_text() for t in legend.get_texts())
            self.assertIn("AUC", text)

    def test_auc_in_legend_text_pr(self):
        ax = aucplot(_make_df(), x="x", y="y", kind="pr", legend=True)
        legend = ax.get_legend()
        if legend is not None:
            text = " ".join(t.get_text() for t in legend.get_texts())
            self.assertIn("AUC", text)

    def test_all_positive_no_crash(self):
        df = pd.DataFrame({"x": [1] * 50, "y": np.linspace(0.1, 0.9, 50)})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ax = aucplot(df, x="x", y="y", kind="roc")
        self.assertIsInstance(ax, plt.Axes)

    def test_minimum_two_samples(self):
        df = pd.DataFrame({"x": [0, 1], "y": [0.2, 0.8]})
        ax = aucplot(df, x="x", y="y", kind="roc")
        self.assertIsInstance(ax, plt.Axes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
