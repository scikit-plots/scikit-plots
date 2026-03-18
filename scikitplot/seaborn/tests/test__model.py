# scikitplot/seaborn/tests/test__model.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for scikitplot.seaborn._model (modelplot / _ModelPlotter).

Coverage
--------
- Source-level AST checks: no bare except (Bug #1), no assert statsmodels (Bug #8).
- _prepare_subset: type coercion, NaN drops, weight extraction.
- _compute_confusion_matrix: shape, values, normalize modes.
- _plot_confusion_matrix: annotation count, text color black/white (Bug #5).
- modelplot: all kind values, returns Axes, early-exit, ax param, multiclass.
"""

from __future__ import annotations

import ast
import inspect
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._model import _ModelPlotter, modelplot
from .. import _model as _model_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n).astype(int)
    y_pred = rng.integers(0, 2, n).astype(float)
    return pd.DataFrame({"x": y_true, "y": y_pred})


class _Base(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def _p(self):
        return _ModelPlotter()

    def _sub(self, x, y, weights=None):
        d = {"x": x, "y": y}
        if weights is not None:
            d["weights"] = weights
        return pd.DataFrame(d)


# ===========================================================================
# 1. Source-level AST checks (Bug #1 and Bug #8)
# ===========================================================================

class TestSourceLevel(_Base):

    def setUp(self):
        self._src = inspect.getsource(_model_module)
        self._tree = ast.parse(self._src)

    def test_no_bare_except(self):
        """Bug #1: no bare except: anywhere in the module."""
        for node in ast.walk(self._tree):
            if isinstance(node, ast.ExceptHandler):
                self.assertIsNotNone(
                    node.type,
                    "Bare `except:` found in _model.py — "
                    "must be `except ImportError:` or similar."
                )

    def test_no_assert_statsmodels(self):
        """Bug #8: assert statsmodels must not appear (disabled under python -O)."""
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Assert):
                if isinstance(node.test, ast.Name) and node.test.id == "statsmodels":
                    self.fail(
                        "`assert statsmodels` found in _model.py — "
                        "use `_has_statsmodels = statsmodels is not None` instead."
                    )

    def test_has_statsmodels_flag_defined(self):
        """_has_statsmodels must be defined regardless of statsmodels availability."""
        self.assertTrue(
            hasattr(_model_module, "_has_statsmodels"),
            "_has_statsmodels flag not defined in _model.py"
        )
        self.assertIsInstance(_model_module._has_statsmodels, bool)


# ===========================================================================
# 2. _prepare_subset
# ===========================================================================

class TestPrepareSubset(_Base):

    def test_correct_types(self):
        y_true, y_pred, w = self._p()._prepare_subset(self._sub([0, 1, 0], [0, 1, 0]))
        self.assertIsInstance(y_true, np.ndarray)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsNone(w)

    def test_y_true_dtype_int(self):
        y_true, _, _ = self._p()._prepare_subset(self._sub([0.0, 1.0], [0.0, 1.0]))
        self.assertTrue(np.issubdtype(y_true.dtype, np.integer))

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


# ===========================================================================
# 3. _compute_confusion_matrix
# ===========================================================================

class TestComputeConfusionMatrix(_Base):

    def test_shape_binary(self):
        cm, _, _ = self._p()._compute_confusion_matrix(
            np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]), None
        )
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

    def test_bad_input_returns_none(self):
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
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        classes = np.array([0, 1])
        p._plot_confusion_matrix(
            y_true=y_true, y_pred=y_pred, sample_weight=None,
            classes=classes, labels=None, legend=False, ax=ax,
            cbar=False, cbar_ax=None, cbar_kws={},
            text_kws={}, image_kws={}, annot_kws={}, digits=2,
        )

    def test_four_annotations_for_2x2_matrix(self):
        fig, ax = plt.subplots()
        self._draw_cm(ax)
        self.assertEqual(len(ax.texts), 4)

    def test_annotation_colors_are_black_or_white(self):
        """Bug #5: each cell annotation must be 'k' or 'w'."""
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
# 5. modelplot public API
# ===========================================================================

class TestModelplot(_Base):

    def test_returns_axes(self):
        ax = modelplot(_make_df(), x="x", y="y", kind="feature_importances")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_none_does_not_raise(self):
        ax = modelplot(_make_df(), x="x", y="y")
        self.assertIsInstance(ax, plt.Axes)

    def test_invalid_kind_raises(self):
        with self.assertRaises(Exception):
            modelplot(_make_df(), x="x", y="y", kind="bad_kind")

    def test_no_xy_returns_ax_early(self):
        fig, ax = plt.subplots()
        result = modelplot(ax=ax)
        self.assertIs(result, ax)

    def test_ax_param_respected(self):
        fig, ax = plt.subplots()
        result = modelplot(_make_df(), x="x", y="y", ax=ax)
        self.assertIs(result, ax)

    def test_multiclass_no_crash(self):
        rng = np.random.default_rng(5)
        n = 60
        df = pd.DataFrame({
            "x": rng.integers(0, 3, n).astype(int),
            "y": rng.integers(0, 3, n).astype(float),
        })
        ax = modelplot(df, x="x", y="y")
        self.assertIsInstance(ax, plt.Axes)

    def test_kind_case_insensitive(self):
        ax = modelplot(_make_df(), x="x", y="y", kind="Feature_Importances")
        self.assertIsInstance(ax, plt.Axes)

    def test_minimum_dataset(self):
        df = pd.DataFrame({"x": [0, 1], "y": [0.0, 1.0]})
        ax = modelplot(df, x="x", y="y")
        self.assertIsInstance(ax, plt.Axes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
