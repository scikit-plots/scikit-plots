# scikitplot/seaborn/tests/test__confusion_matrix.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for scikitplot.seaborn._confusion_matrix (evalplot).

Coverage
--------
- _prepare_subset: type coercion, NaN drops, weight validation,
  allow_probs binary guard (Bug #4), probability range.
- _compute_classification_report: returns string, handles bad inputs.
- _compute_confusion_matrix: shape, values, normalise modes.
- _plot_confusion_matrix: text color correctness (Bug #5).
- evalplot: all kind values, allow_probs, multiclass, axes param.
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

from .._confusion_matrix import _ConfusionMatrixPlotter, evalplot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=100, seed=0, multiclass=False):
    rng = np.random.default_rng(seed)
    if multiclass:
        y_true = rng.integers(0, 3, size=n)
        y_pred = rng.integers(0, 3, size=n)
    else:
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.integers(0, 2, size=n)
    return pd.DataFrame({"x": y_true, "y": y_pred})


class _Base(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def _p(self):
        return _ConfusionMatrixPlotter()

    def _sub(self, x, y, weights=None):
        d = {"x": x, "y": y}
        if weights is not None:
            d["weights"] = weights
        return pd.DataFrame(d)


# ===========================================================================
# 1. _prepare_subset
# ===========================================================================

class TestPrepareSubset(_Base):

    def test_correct_types(self):
        y_true, y_pred, w = self._p()._prepare_subset(self._sub([0, 1, 0], [0, 1, 0]))
        self.assertIsInstance(y_true, np.ndarray)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsNone(w)

    def test_empty_returns_none_triple(self):
        sub = pd.DataFrame({"x": [np.nan], "y": [np.nan]})
        self.assertEqual(self._p()._prepare_subset(sub), (None, None, None))

    def test_nan_rows_dropped(self):
        sub = self._sub([0, np.nan, 1], [0, 0, 1])
        y_true, y_pred, _ = self._p()._prepare_subset(sub)
        self.assertEqual(len(y_true), 2)

    def test_weights_extracted(self):
        sub = self._sub([0, 1], [0, 1], [2.0, 3.0])
        _, _, w = self._p()._prepare_subset(sub)
        self.assertIsNotNone(w)
        np.testing.assert_array_equal(w, [2.0, 3.0])

    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(self._sub([0, 1], [0, 1], [-1.0, 1.0]))
        self.assertIn("non-negative", str(ctx.exception))

    def test_allow_probs_binary_ok(self):
        sub = self._sub([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        y_true, y_pred, _ = self._p()._prepare_subset(sub, allow_probs=True)
        self.assertIsNotNone(y_true)
        self.assertIsNotNone(y_pred)

    def test_allow_probs_multiclass_raises_valueerror_not_indexerror(self):
        """Bug #4: 3 classes + allow_probs=True must raise ValueError, not IndexError."""
        sub = self._sub([0, 1, 2, 0, 1, 2], [0.3, 0.6, 0.5, 0.1, 0.8, 0.4])
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(sub, allow_probs=True)
        self.assertIn("binary", str(ctx.exception).lower())

    def test_allow_probs_score_above_1_raises(self):
        sub = self._sub([0, 1], [0.5, 1.5])
        with self.assertRaises(ValueError) as ctx:
            self._p()._prepare_subset(sub, allow_probs=True)
        self.assertIn("[0, 1]", str(ctx.exception))


# ===========================================================================
# 2. _compute_classification_report
# ===========================================================================

class TestComputeClassificationReport(_Base):

    def test_returns_string(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        n, pos, s = self._p()._compute_classification_report(y_true, y_pred)
        self.assertIsInstance(s, str)
        self.assertTrue("precision" in s.lower() or len(s) > 0)

    def test_bad_input_warns_and_returns_none(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self._p()._compute_classification_report(y_true, y_pred)
        # Either n or s is None
        self.assertTrue(result[0] is None or result[2] is None)


# ===========================================================================
# 3. _compute_confusion_matrix
# ===========================================================================

class TestComputeConfusionMatrix(_Base):

    def test_shape_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        cm, _, _ = self._p()._compute_confusion_matrix(y_true, y_pred, None)
        self.assertEqual(cm.shape, (2, 2))

    def test_diagonal_perfect_classifier(self):
        y = np.array([0, 0, 1, 1])
        cm, _, _ = self._p()._compute_confusion_matrix(y, y, None)
        self.assertEqual(cm[0, 0], 2)
        self.assertEqual(cm[1, 1], 2)
        self.assertEqual(cm[0, 1], 0)
        self.assertEqual(cm[1, 0], 0)

    def test_multiclass_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 2, 0, 1])
        cm, _, _ = self._p()._compute_confusion_matrix(y_true, y_pred, None)
        self.assertEqual(cm.shape, (3, 3))

    def test_normalize_true_diagonal_is_1(self):
        y = np.array([0, 0, 1, 1])
        cm, _, _ = self._p()._compute_confusion_matrix(y, y, None, normalize="true")
        np.testing.assert_allclose(cm.diagonal(), [1.0, 1.0], atol=1e-6)

    def test_bad_input_returns_none_first_element(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self._p()._compute_confusion_matrix(
                np.array([0, 1]), np.array([0]), None
            )
        self.assertIsNone(result[0])


# ===========================================================================
# 4. CM text color — Bug #5
# ===========================================================================

class TestCMTextColor(_Base):

    def _draw_cm(self, ax):
        p = self._p()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        classes = np.array([0, 1])
        p._plot_confusion_matrix(
            y_true=y_true, y_pred=y_pred, sample_weight=None,
            classes=classes, labels=None, legend=False, ax=ax,
            cbar_kws={}, cbar=False, cbar_ax=None,
            text_kws={}, image_kws={}, annot_kws={}, annot=True,
            fmt="", digits=2,
        )

    def test_four_annotations_for_2x2_matrix(self):
        fig, ax = plt.subplots()
        self._draw_cm(ax)
        self.assertEqual(len(ax.texts), 4)

    def test_annotation_colors_are_black_or_white(self):
        """Bug #5: text color must be 'k' or 'w', derived from actual cell color."""
        fig, ax = plt.subplots()
        self._draw_cm(ax)
        for txt in ax.texts:
            c = txt.get_color()
            self.assertIn(c, ("k", "w"),
                          f"Unexpected annotation color: {c!r}")

    def test_xlabel_predicted(self):
        fig, ax = plt.subplots()
        self._draw_cm(ax)
        self.assertIn("Predicted", ax.get_xlabel())

    def test_ylabel_true(self):
        fig, ax = plt.subplots()
        self._draw_cm(ax)
        self.assertIn("True", ax.get_ylabel())


# ===========================================================================
# 5. evalplot public API
# ===========================================================================

class TestEvalplot(_Base):

    def test_returns_axes_confusion_matrix(self):
        ax = evalplot(_make_df(), x="x", y="y", kind="confusion_matrix")
        self.assertIsInstance(ax, plt.Axes)

    def test_returns_axes_classification_report(self):
        ax = evalplot(_make_df(), x="x", y="y", kind="classification_report")
        self.assertIsInstance(ax, plt.Axes)

    def test_returns_axes_all(self):
        ax = evalplot(_make_df(), x="x", y="y", kind="all")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_none_defaults_to_all(self):
        ax = evalplot(_make_df(), x="x", y="y")
        self.assertIsInstance(ax, plt.Axes)

    def test_invalid_kind_raises(self):
        with self.assertRaises(Exception):
            evalplot(_make_df(), x="x", y="y", kind="garbage")

    def test_no_xy_returns_ax_early(self):
        fig, ax = plt.subplots()
        result = evalplot(ax=ax)
        self.assertIs(result, ax)

    def test_ax_param_respected(self):
        fig, ax = plt.subplots()
        result = evalplot(_make_df(), x="x", y="y", kind="confusion_matrix", ax=ax)
        self.assertIs(result, ax)

    def test_multiclass_no_crash(self):
        ax = evalplot(_make_df(multiclass=True), x="x", y="y", kind="confusion_matrix")
        self.assertIsInstance(ax, plt.Axes)

    def test_allow_probs_binary(self):
        rng = np.random.default_rng(1)
        n = 100
        df = pd.DataFrame({
            "x": rng.integers(0, 2, n),
            "y": np.clip(rng.uniform(0, 1, n), 0, 1),
        })
        ax = evalplot(df, x="x", y="y", kind="confusion_matrix", allow_probs=True)
        self.assertIsInstance(ax, plt.Axes)

    def test_allow_probs_multiclass_emits_warning(self):
        """
        Bug #4: evalplot wraps _prepare_subset ValueError as a UserWarning.

        The underlying validation error is confirmed in TestPrepareSubset;
        here we confirm the public API does not silently succeed.
        """
        rng = np.random.default_rng(2)
        n = 60
        df = pd.DataFrame({
            "x": rng.integers(0, 3, n),
            "y": np.clip(rng.uniform(0, 1, n), 0, 1),
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            evalplot(df, x="x", y="y", kind="confusion_matrix", allow_probs=True)
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any("binary" in m.lower() or "allow_probs" in m.lower()
                or "2 unique" in m for m in messages),
            f"Expected a warning about binary/allow_probs; got: {messages}"
        )

    def test_kind_case_insensitive(self):
        ax = evalplot(_make_df(), x="x", y="y", kind="Confusion_Matrix")
        self.assertIsInstance(ax, plt.Axes)

    def test_minimum_dataset(self):
        df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        ax = evalplot(df, x="x", y="y", kind="confusion_matrix")
        self.assertIsInstance(ax, plt.Axes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
