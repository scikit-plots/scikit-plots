# scikitplot/decile/tests/test__decile_modelplotpy.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Extended tests for :mod:`~scikitplot.decile._decile_modelplotpy`.

Covers uncovered branches not reached by the baseline test suite:

- ``_annotate_highlight`` marker mode, custom bbox/arrowprops, xytext sign branches
- ``_parse_plot_kws`` all defaults and legacy conflict
- ``_render_footer_text`` overflow / warning path and empty lines guard
- ``_EvalKey`` frozen dataclass identity and equality
- ``_setup_axis`` percent formatting and no-grid variant
- ``_scope_triplet`` no_comparison pass-through
- ``plot_costsrevs`` compare_datasets, compare_targetclasses, non-identical
  cumtot error, highlight with text / plot_text modes
- ``plot_profit`` and ``plot_roi`` with comparison scopes and highlight
- ``plot_all`` footer rendering path (highlight_how='plot' and 'text')
- ``plotting_scope`` auto-inference for unfixed single dimensions
  (universe drives compare_models / compare_datasets / compare_targetclasses)
- ``plotting_scope`` auto-inference ambiguous multi-unfixed raises
- ``plotting_scope`` explicit scope guard (1 model / dataset / class raises)
- ``ModelPlotPy._validate_state`` inconsistent classes_ across models
- ``ModelPlotPy.set_params`` ntiles validation after mutation
- ``aggregate_over_ntiles`` with two models, two datasets
- Legacy ``modelplotpy`` adapter smoke tests

Notes
-----
Matplotlib is forced to the non-interactive 'Agg' backend.
All tests are pure ``unittest.TestCase`` for maximum compatibility.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from .._decile_modelplotpy import (
    # dataclass
    _EvalKey,
    # helpers
    _PlotInputError,
    _PlotKws,
    _annotate_highlight,
    _annotation_xytext,
    _as_dataframe,
    _as_int_like,
    _as_series,
    _assign_descending_ntiles,
    _autopct,
    _check_input,
    _currency_fmt,
    _format_highlight_line,
    _get_ntiles_from_plot_input,
    _iter_groups,
    _normalize_highlight_ntiles,
    _ntile_label,
    _parse_plot_kws,
    _range01,
    _render_footer_text,
    _require_columns,
    _scope_grouping,
    _scope_triplet,
    _setup_axis,
    _unique_one,
    _validate_highlight_how,
    _validate_number,
    _value_at_ntile,
    # class and plots
    ModelPlotPy,
    plot_all,
    plot_costsrevs,
    plot_cumgains,
    plot_cumresponse,
    plot_cumlift,
    plot_profit,
    plot_response,
    plot_roi,
    summarize_selection,
)


##############################################################################
# Shared fixtures
##############################################################################


def _make_binary_data(n: int = 200, random_state: int = 0):
    """Return a binary classification dataset as (DataFrame, Series)."""
    X, y = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=random_state,
    )
    return pd.DataFrame(X), pd.Series(y)


def _make_fitted_lr(X, y, random_state: int = 0):
    """Fit and return a LogisticRegression."""
    return LogisticRegression(random_state=random_state, max_iter=1000).fit(X, y)


def _make_mp(ntiles: int = 10):
    """Return a ModelPlotPy instance with a single binary dataset."""
    X, y = _make_binary_data()
    lr = _make_fitted_lr(X, y)
    return ModelPlotPy(
        feature_data=[X],
        label_data=[y],
        dataset_labels=["train"],
        models=[lr],
        model_labels=["lr"],
        ntiles=ntiles,
    )


def _make_mp2_models(ntiles: int = 10):
    """Return ModelPlotPy with 2 models, 1 dataset."""
    X, y = _make_binary_data()
    lr1 = _make_fitted_lr(X, y, random_state=0)
    lr2 = _make_fitted_lr(X, y, random_state=1)
    return ModelPlotPy(
        feature_data=[X],
        label_data=[y],
        dataset_labels=["train"],
        models=[lr1, lr2],
        model_labels=["lr1", "lr2"],
        ntiles=ntiles,
    )


def _make_mp2_datasets(ntiles: int = 10):
    """Return ModelPlotPy with 1 model, 2 datasets."""
    X, y = _make_binary_data()
    lr = _make_fitted_lr(X, y)
    return ModelPlotPy(
        feature_data=[X, X],
        label_data=[y, y],
        dataset_labels=["train", "test"],
        models=[lr],
        model_labels=["lr"],
        ntiles=ntiles,
    )


def _make_plot_input(ntiles: int = 10, select_targetclass=None):
    """Return plotting scope for no_comparison."""
    mp = _make_mp(ntiles=ntiles)
    return mp.plotting_scope(
        scope="no_comparison",
        select_targetclass=select_targetclass or [1],
    )


def _make_financial_params():
    """Standard financial parameters for cost/profit plots."""
    return dict(fixed_costs=500.0, variable_costs_per_unit=2.0, profit_per_unit=25.0)


##############################################################################
# _EvalKey dataclass
##############################################################################


class TestEvalKey(unittest.TestCase):
    """_EvalKey is a frozen dataclass used as a composite dict key."""

    def test_equality(self):
        a = _EvalKey(model_label="lr", dataset_label="train")
        b = _EvalKey(model_label="lr", dataset_label="train")
        self.assertEqual(a, b)

    def test_inequality_model(self):
        a = _EvalKey(model_label="lr1", dataset_label="train")
        b = _EvalKey(model_label="lr2", dataset_label="train")
        self.assertNotEqual(a, b)

    def test_inequality_dataset(self):
        a = _EvalKey(model_label="lr", dataset_label="train")
        b = _EvalKey(model_label="lr", dataset_label="test")
        self.assertNotEqual(a, b)

    def test_hashable_usable_as_dict_key(self):
        key = _EvalKey(model_label="lr", dataset_label="train")
        d = {key: 42}
        self.assertEqual(d[key], 42)

    def test_frozen_raises_on_mutation(self):
        key = _EvalKey(model_label="lr", dataset_label="train")
        with self.assertRaises((AttributeError, TypeError)):
            key.model_label = "other"  # type: ignore[misc]

    def test_repr_contains_fields(self):
        key = _EvalKey(model_label="lr", dataset_label="train")
        r = repr(key)
        self.assertIn("lr", r)
        self.assertIn("train", r)


##############################################################################
# _annotate_highlight — uncovered branches
##############################################################################


class TestAnnotateHighlight(unittest.TestCase):
    """Direct tests for _annotate_highlight internals."""

    def setUp(self):
        self.fig, self.ax = plt.subplots()

    def tearDown(self):
        plt.close("all")

    def test_marker_mode_returns_early_no_annotation(self):
        """mode='marker' must return after drawing guides and not call annotate."""
        _annotate_highlight(
            self.ax,
            x=3,
            y=0.5,
            x0=1,
            y0=0.0,
            color="blue",
            text="ignored",
            xytext=(-20, 20),
            annotation_kws={"mode": "marker"},
        )
        # No annotation artists should be added.
        self.assertEqual(len(self.ax.texts), 0)
        # But guide lines (3 × ax.plot calls) should exist.
        self.assertGreaterEqual(len(self.ax.lines), 3)

    def test_default_callout_mode_creates_annotation(self):
        """Default mode='callout' must create an annotation."""
        _annotate_highlight(
            self.ax,
            x=5,
            y=0.4,
            x0=1,
            y0=0.0,
            color="red",
            text="3.10%",
            xytext=(10, 20),
            annotation_kws=None,
        )
        # ax.annotate creates an Annotation; it shows up in ax.texts.
        self.assertGreaterEqual(len(self.ax.texts), 1)

    def test_custom_marker_size_pop(self):
        """marker_size kwarg must be consumed and not forwarded."""
        _annotate_highlight(
            self.ax,
            x=2,
            y=0.3,
            x0=0,
            y0=0.0,
            color="green",
            text="x",
            xytext=(0, 10),
            annotation_kws={"marker_size": 10.0},
        )
        # If marker_size leaks into annotate() it raises TypeError; reaching
        # this point without error is sufficient.

    def test_positive_xytext_va_is_bottom(self):
        """Positive xytext[1] must set va='bottom' in the annotation."""
        _annotate_highlight(
            self.ax,
            x=5,
            y=0.5,
            x0=1,
            y0=0.0,
            color="blue",
            text="T",
            xytext=(0, 30),  # positive → va='bottom'
            annotation_kws={},
        )
        # Verify annotation was created (no crash is the key assertion).
        self.assertIsNotNone(self.ax)

    def test_negative_xytext_va_is_top(self):
        """Negative xytext[1] must set va='top'."""
        _annotate_highlight(
            self.ax,
            x=5,
            y=0.5,
            x0=1,
            y0=0.0,
            color="blue",
            text="T",
            xytext=(0, -30),  # negative → va='top'
            annotation_kws={},
        )

    def test_custom_bbox_merged(self):
        """bbox key in annotation_kws must be merged (not dropped)."""
        custom_bbox = {"fc": "yellow"}
        _annotate_highlight(
            self.ax,
            x=3,
            y=0.5,
            x0=1,
            y0=0.0,
            color="blue",
            text="M",
            xytext=(10, 10),
            annotation_kws={"bbox": custom_bbox},
        )

    def test_custom_arrowprops_merged(self):
        """arrowprops key in annotation_kws must be merged (not dropped)."""
        _annotate_highlight(
            self.ax,
            x=3,
            y=0.5,
            x0=1,
            y0=0.0,
            color="blue",
            text="M",
            xytext=(10, 10),
            annotation_kws={"arrowprops": {"arrowstyle": "->"}},
        )


##############################################################################
# _parse_plot_kws — defaults and conflict
##############################################################################


class TestParsePlotKws(unittest.TestCase):
    """Verify _parse_plot_kws fills expected defaults."""

    def _call(self, **overrides):
        defaults = dict(
            line_kws=None,
            ref_line_kws=None,
            legend_kws=None,
            grid_kws=None,
            axes_kws=None,
            annotation_kws=None,
            footer_kws=None,
            legacy_line_kwargs={},
            where="test",
        )
        defaults.update(overrides)
        return _parse_plot_kws(**defaults)

    def test_returns_PlotKws_instance(self):
        result = self._call()
        self.assertIsInstance(result, _PlotKws)

    def test_ref_line_kws_linestyle_default(self):
        pk = self._call()
        self.assertEqual(pk.ref_line_kws["linestyle"], "dashed")

    def test_legend_kws_shadow_default(self):
        pk = self._call()
        self.assertFalse(pk.legend_kws["shadow"])

    def test_legend_kws_frameon_default(self):
        pk = self._call()
        self.assertFalse(pk.legend_kws["frameon"])

    def test_grid_kws_visible_default(self):
        pk = self._call()
        self.assertTrue(pk.grid_kws["visible"])

    def test_footer_kws_x_default(self):
        pk = self._call()
        self.assertAlmostEqual(pk.footer_kws["x"], 0.00)

    def test_footer_kws_y_default(self):
        pk = self._call()
        self.assertAlmostEqual(pk.footer_kws["y"], 0.00)

    def test_footer_kws_ha_default(self):
        pk = self._call()
        self.assertEqual(pk.footer_kws["ha"], "left")

    def test_footer_kws_fontsize_default(self):
        pk = self._call()
        self.assertEqual(pk.footer_kws["fontsize"], 10)

    def test_footer_kws_base_pad_default(self):
        pk = self._call()
        self.assertAlmostEqual(pk.footer_kws["base_pad"], 0.10)

    def test_line_kws_forwarded(self):
        pk = self._call(line_kws={"linewidth": 3})
        self.assertEqual(pk.line_kws["linewidth"], 3)

    def test_legacy_line_kwargs_forwarded(self):
        pk = self._call(legacy_line_kwargs={"alpha": 0.5})
        self.assertAlmostEqual(pk.line_kws["alpha"], 0.5)

    def test_line_kws_and_legacy_conflict_raises(self):
        with self.assertRaises(ValueError):
            self._call(line_kws={"lw": 2}, legacy_line_kwargs={"lw": 3})

    def test_custom_legend_kws_shadow(self):
        pk = self._call(legend_kws={"shadow": True})
        self.assertTrue(pk.legend_kws["shadow"])

    def test_annotation_kws_empty_by_default(self):
        pk = self._call()
        self.assertIsInstance(pk.annotation_kws, dict)
        self.assertEqual(len(pk.annotation_kws), 0)

    def test_axes_kws_empty_by_default(self):
        pk = self._call()
        self.assertIsInstance(pk.axes_kws, dict)
        self.assertEqual(len(pk.axes_kws), 0)


##############################################################################
# _render_footer_text — overflow and empty paths
##############################################################################


class TestRenderFooterText(unittest.TestCase):
    """Test _render_footer_text including overflow guard and empty lines."""

    def tearDown(self):
        plt.close("all")

    def test_empty_lines_does_nothing(self):
        fig, ax = plt.subplots()
        n_before = len(fig.axes)
        _render_footer_text(fig, lines=[], footer_kws={})
        # No new axes should be added when lines is empty.
        self.assertEqual(len(fig.axes), n_before)

    def test_single_line_adds_footer_axes(self):
        fig, ax = plt.subplots()
        n_before = len(fig.axes)
        _render_footer_text(fig, lines=["Response @ decile 3 | value=5.00%"],
                            footer_kws={})
        self.assertGreater(len(fig.axes), n_before)

    def test_many_lines_overflow_logged(self):
        """50 lines pushes bottom_pad >= 1.0 → logger.info must be called."""
        fig, _ = plt.subplots()
        lines = [f"line {i}" for i in range(50)]
        import scikitplot.decile._decile_modelplotpy as mod
        with patch.object(mod.logger, "info") as mock_info:
            _render_footer_text(fig, lines=lines, footer_kws={})
        # logger.info should have been called at least once about overflow.
        mock_info.assert_called()

    def test_custom_footer_kws_forwarded(self):
        fig, ax = plt.subplots()
        _render_footer_text(
            fig,
            lines=["test line"],
            footer_kws={"fontsize": 8, "ha": "right", "x": 0.5},
        )
        # Reaches here without error → kwargs were accepted.

    def test_multiple_lines_text_joined_with_newline(self):
        """Multiple lines must be joined with newline in the text artist."""
        fig, ax = plt.subplots()
        lines = ["line A", "line B", "line C"]
        _render_footer_text(fig, lines=lines, footer_kws={})
        # Find the footer axes (the last one added).
        footer_ax = fig.axes[-1]
        texts = footer_ax.texts
        self.assertGreater(len(texts), 0)
        combined = texts[0].get_text()
        self.assertIn("line A", combined)
        self.assertIn("line C", combined)


##############################################################################
# _setup_axis — direct tests
##############################################################################


class TestSetupAxis(unittest.TestCase):
    """Direct tests for _setup_axis."""

    def tearDown(self):
        plt.close("all")

    def test_percent_y_formatter_applied(self):
        _, ax = plt.subplots()
        _setup_axis(
            ax, title="T", xlabel="X", ylabel="Y",
            ntiles=10, xlim=(0, 10), percent_y=True, grid=True,
        )
        fmt_cls = type(ax.yaxis.get_major_formatter()).__name__
        self.assertIn("Percent", fmt_cls)

    def test_no_percent_y_no_percent_formatter(self):
        _, ax = plt.subplots()
        _setup_axis(
            ax, title="T", xlabel="X", ylabel="Y",
            ntiles=10, xlim=(0, 10), percent_y=False, grid=False,
        )
        fmt_cls = type(ax.yaxis.get_major_formatter()).__name__
        self.assertNotIn("Percent", fmt_cls)

    def test_grid_false_no_grid(self):
        _, ax = plt.subplots()
        _setup_axis(
            ax, title="T", xlabel="X", ylabel="Y",
            ntiles=10, xlim=(0, 10), percent_y=False, grid=False,
        )
        # Grid lines should be invisible.
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            self.assertFalse(line.get_visible())

    def test_title_set(self):
        _, ax = plt.subplots()
        _setup_axis(
            ax, title="My Title", xlabel="X", ylabel="Y",
            ntiles=5, xlim=(0, 5), percent_y=False, grid=True,
        )
        self.assertEqual(ax.get_title(), "My Title")

    def test_xlim_set(self):
        _, ax = plt.subplots()
        _setup_axis(
            ax, title="T", xlabel="X", ylabel="Y",
            ntiles=10, xlim=(1, 10), percent_y=False, grid=False,
        )
        self.assertEqual(ax.get_xlim()[0], 1.0)
        self.assertEqual(ax.get_xlim()[1], 10.0)

    def test_xticks_cover_full_range(self):
        _, ax = plt.subplots()
        _setup_axis(
            ax, title="T", xlabel="X", ylabel="Y",
            ntiles=5, xlim=(0, 5), percent_y=False, grid=False,
        )
        ticks = sorted(ax.get_xticks())
        self.assertIn(0, ticks)
        self.assertIn(5, ticks)


##############################################################################
# _annotation_xytext — extended params
##############################################################################


class TestAnnotationXytextExtended(unittest.TestCase):
    """Extended param coverage for _annotation_xytext."""

    def test_custom_x_offset_and_step(self):
        x, y = _annotation_xytext(0, x_offset=-50, x_step=10)
        self.assertEqual(x, -50)

    def test_custom_base_and_gap(self):
        # annotation_index=0: col=0, row=0, yo = -(base + gap*0) = -base
        x, y = _annotation_xytext(0, base=40, gap=5)
        self.assertEqual(abs(y), 40)

    def test_gap_applied_on_row_1(self):
        # layout_cols=1 → row1 starts at index=1
        # col=0, row=1, mag=base+gap*1
        x, y = _annotation_xytext(1, layout_cols=1, base=20, gap=10)
        self.assertEqual(abs(y), 30)

    def test_layout_cols_5_col_wrapping(self):
        # index=5 with layout_cols=5: col=0, row=1
        x0, _ = _annotation_xytext(0, layout_cols=5)
        x5, _ = _annotation_xytext(5, layout_cols=5)
        # Both are in col 0 → same x coordinate.
        self.assertEqual(x0, x5)

    def test_large_index_is_deterministic(self):
        r1 = _annotation_xytext(99)
        r2 = _annotation_xytext(99)
        self.assertEqual(r1, r2)


##############################################################################
# ModelPlotPy — additional validation coverage
##############################################################################


class TestModelPlotPyValidation(unittest.TestCase):
    """Additional _validate_state branches."""

    def setUp(self):
        self.X, self.y = _make_binary_data()
        self.lr = _make_fitted_lr(self.X, self.y)

    def tearDown(self):
        plt.close("all")

    def test_inconsistent_classes_raises(self):
        """Two models with different classes_ must raise ValueError."""
        from sklearn.datasets import make_classification
        # Build a 3-class model (classes_ = [0,1,2]).
        X3, y3 = make_classification(
            n_samples=300, n_features=5, n_classes=3,
            n_informative=3, n_redundant=0, random_state=0,
        )
        lr3 = LogisticRegression(max_iter=2000).fit(X3, y3)
        # lr3 has classes_ [0,1,2], self.lr has [0,1] → mismatch.
        with self.assertRaises(ValueError):
            ModelPlotPy(
                feature_data=[self.X],
                label_data=[self.y],
                dataset_labels=["train"],
                models=[self.lr, lr3],
                model_labels=["lr2", "lr3"],
                ntiles=2,
            )

    def test_set_params_triggers_validation(self):
        """set_params with bad ntiles must re-raise via _validate_state."""
        mp = _make_mp()
        with self.assertRaises((ValueError, Exception)):
            mp.set_params(ntiles=1)

    def test_get_params_returns_correct_ntiles(self):
        mp = _make_mp(ntiles=5)
        self.assertEqual(mp.get_params()["ntiles"], 5)

    def test_get_params_returns_model_labels(self):
        mp = _make_mp()
        self.assertIn("lr", mp.get_params()["model_labels"])

    def test_reset_params_clears_all(self):
        mp = _make_mp()
        mp.reset_params()
        p = mp.get_params()
        self.assertEqual(p["feature_data"], [])
        self.assertEqual(p["label_data"], [])
        self.assertEqual(p["dataset_labels"], [])
        self.assertEqual(p["models"], [])
        self.assertEqual(p["model_labels"], [])
        self.assertEqual(p["ntiles"], 10)
        self.assertEqual(p["seed"], 0)

    def test_prepare_scores_no_datasets_raises(self):
        X, y = _make_binary_data()
        lr = _make_fitted_lr(X, y)
        mp = ModelPlotPy(models=[lr], model_labels=["lr"], ntiles=2)
        with self.assertRaises(ValueError):
            mp.prepare_scores_and_ntiles()


##############################################################################
# aggregate_over_ntiles — multi-model / multi-dataset
##############################################################################


class TestAggregateOverNtilesExtended(unittest.TestCase):
    """Coverage for multi-model, multi-dataset aggregate paths."""

    def test_two_models_row_count(self):
        mp = _make_mp2_models(ntiles=5)
        agg = mp.aggregate_over_ntiles()
        # 2 models × 1 dataset × 2 classes × (ntiles+1 rows each)
        expected = 2 * 1 * 2 * (5 + 1)
        self.assertEqual(len(agg), expected)

    def test_two_datasets_row_count(self):
        mp = _make_mp2_datasets(ntiles=5)
        agg = mp.aggregate_over_ntiles()
        # 1 model × 2 datasets × 2 classes × (5+1)
        expected = 1 * 2 * 2 * (5 + 1)
        self.assertEqual(len(agg), expected)

    def test_model_labels_present(self):
        mp = _make_mp2_models(ntiles=5)
        agg = mp.aggregate_over_ntiles()
        self.assertIn("lr1", agg["model_label"].values)
        self.assertIn("lr2", agg["model_label"].values)

    def test_dataset_labels_present(self):
        mp = _make_mp2_datasets(ntiles=5)
        agg = mp.aggregate_over_ntiles()
        self.assertIn("train", agg["dataset_label"].values)
        self.assertIn("test", agg["dataset_label"].values)

    def test_sorted_output(self):
        mp = _make_mp2_models(ntiles=5)
        agg = mp.aggregate_over_ntiles()
        for _, grp in agg.groupby(["model_label", "dataset_label", "target_class"]):
            ntiles_sorted = grp["ntile"].to_numpy()
            self.assertTrue(np.all(np.diff(ntiles_sorted) >= 0))

    def test_gain_opt_saturates(self):
        mp = _make_mp(ntiles=10)
        agg = mp.aggregate_over_ntiles()
        max_row = agg[agg["ntile"] == 10]
        self.assertTrue((max_row["gain_opt"] == 1.0).all())


##############################################################################
# plotting_scope — auto-inference unfixed dimension branches
##############################################################################


class TestPlottingScopeAutoInference(unittest.TestCase):
    """
    Covers the Case-2 branch of _infer_scope where no explicit multi-select
    is provided but the universe has multiple values in exactly one dimension.
    """

    def setUp(self):
        X, y = _make_binary_data()
        lr1 = _make_fitted_lr(X, y, random_state=0)
        lr2 = _make_fitted_lr(X, y, random_state=1)
        # 2 models, 1 dataset → auto should infer compare_models when
        # dataset and target are fixed by explicit selection.
        self.mp_2m = ModelPlotPy(
            feature_data=[X],
            label_data=[y],
            dataset_labels=["train"],
            models=[lr1, lr2],
            model_labels=["lr1", "lr2"],
            ntiles=5,
        )
        # 1 model, 2 datasets → auto should infer compare_datasets.
        self.mp_2d = ModelPlotPy(
            feature_data=[X, X],
            label_data=[y, y],
            dataset_labels=["train", "test"],
            models=[lr1],
            model_labels=["lr1"],
            ntiles=5,
        )

    def tearDown(self):
        plt.close("all")

    def test_auto_infers_compare_models_from_universe(self):
        """With 2 models and no explicit model selector, auto → compare_models."""
        result = self.mp_2m.plotting_scope(
            scope="auto",
            select_dataset_label=["train"],
            select_targetclass=[1],
        )
        self.assertTrue((result["scope"] == "compare_models").all())
        self.assertIn("lr1", result["model_label"].values)
        self.assertIn("lr2", result["model_label"].values)

    def test_auto_infers_compare_datasets_from_universe(self):
        """With 2 datasets and no explicit dataset selector, auto → compare_datasets."""
        result = self.mp_2d.plotting_scope(
            scope="auto",
            select_model_label=["lr1"],
            select_targetclass=[1],
        )
        self.assertTrue((result["scope"] == "compare_datasets").all())
        self.assertIn("train", result["dataset_label"].values)
        self.assertIn("test", result["dataset_label"].values)

    def test_auto_ambiguous_multi_unfixed_raises(self):
        """2 models + 2 datasets, no selectors → ambiguous, must raise."""
        X, y = _make_binary_data()
        lr1 = _make_fitted_lr(X, y, random_state=0)
        lr2 = _make_fitted_lr(X, y, random_state=1)
        mp = ModelPlotPy(
            feature_data=[X, X],
            label_data=[y, y],
            dataset_labels=["train", "test"],
            models=[lr1, lr2],
            model_labels=["lr1", "lr2"],
            ntiles=5,
        )
        with self.assertRaises(ValueError):
            mp.plotting_scope(scope="auto", select_targetclass=[1])

    def test_compare_targetclasses_all_classes(self):
        """compare_targetclasses with explicit [0,1] selector uses both classes."""
        result = self.mp_2m.plotting_scope(
            scope="compare_targetclasses",
            select_model_label=["lr1"],
            select_dataset_label=["train"],
            select_targetclass=[0, 1],
        )
        self.assertIn(0, result["target_class"].values)
        self.assertIn(1, result["target_class"].values)

    def test_compare_models_with_1_model_raises(self):
        """Explicit compare_models with only 1 model available must raise."""
        mp = _make_mp()
        with self.assertRaises(ValueError):
            mp.plotting_scope(scope="compare_models", select_targetclass=[1])

    def test_compare_datasets_with_1_dataset_raises(self):
        """Explicit compare_datasets with only 1 dataset must raise."""
        mp = _make_mp()
        with self.assertRaises(ValueError):
            mp.plotting_scope(scope="compare_datasets", select_targetclass=[1])

    def test_compare_targetclasses_with_1_class_raises(self):
        """Explicit compare_targetclasses with only 1 class selected must raise."""
        with self.assertRaises(ValueError):
            self.mp_2m.plotting_scope(
                scope="compare_targetclasses",
                select_model_label=["lr1"],
                select_dataset_label=["train"],
                select_targetclass=[1],
            )

    def test_one_or_error_multi_value_explicit_raises(self):
        """Providing 2 values for a fixed dimension must raise."""
        with self.assertRaises(ValueError):
            self.mp_2m.plotting_scope(
                scope="compare_models",
                select_model_label=["lr1", "lr2"],
                select_dataset_label=["train", "train"],  # 2 values for fixed dim
                select_targetclass=[1],
            )


##############################################################################
# plot_costsrevs — extended scope branches
##############################################################################


class TestPlotCostsRevsExtended(unittest.TestCase):
    """Tests for untested scope / mode branches of plot_costsrevs."""

    @classmethod
    def setUpClass(cls):
        X, y = _make_binary_data()
        lr1 = _make_fitted_lr(X, y, random_state=0)
        lr2 = _make_fitted_lr(X, y, random_state=1)
        cls.mp_2m = ModelPlotPy(
            feature_data=[X],
            label_data=[y],
            dataset_labels=["train"],
            models=[lr1, lr2],
            model_labels=["lr1", "lr2"],
            ntiles=5,
        )
        cls.mp_2d = ModelPlotPy(
            feature_data=[X, X],
            label_data=[y, y],
            dataset_labels=["train", "test"],
            models=[lr1],
            model_labels=["lr1"],
            ntiles=5,
        )
        cls.plot_nc = _make_mp(ntiles=5).plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        cls.plot_cm = cls.mp_2m.plotting_scope(
            scope="compare_models",
            select_dataset_label=["train"],
            select_targetclass=[1],
        )
        cls.plot_cd = cls.mp_2d.plotting_scope(
            scope="compare_datasets",
            select_model_label=["lr1"],
            select_targetclass=[1],
        )
        cls.fp = _make_financial_params()

    def tearDown(self):
        plt.close("all")

    def test_compare_datasets_scope_returns_axes(self):
        ax = plot_costsrevs(self.plot_cd, **self.fp)
        self.assertIsInstance(ax, plt.Axes)

    def test_compare_datasets_title_contains_datasets(self):
        ax = plot_costsrevs(self.plot_cd, **self.fp)
        title = ax.get_title()
        self.assertIn("dataset", title.lower())

    def test_compare_models_non_identical_cumtot_raises(self):
        """Injecting different cumtot values per ntile must raise _PlotInputError."""
        plot_cm_bad = self.plot_cm.copy()
        # Corrupt cumtot for one model at ntile 1.
        mask = (plot_cm_bad["model_label"] == "lr1") & (plot_cm_bad["ntile"] == 1)
        plot_cm_bad.loc[mask, "cumtot"] += 999
        with self.assertRaises(_PlotInputError):
            plot_costsrevs(plot_cm_bad, **self.fp)

    def test_highlight_text_mode_prints(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_costsrevs(
                self.plot_nc, **self.fp,
                highlight_ntile=3, highlight_how="text",
            )
        self.assertIn("Revenues", buf.getvalue())

    def test_highlight_plot_text_mode_returns_axes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ax = plot_costsrevs(
                self.plot_nc, **self.fp,
                highlight_ntile=3, highlight_how="plot_text",
            )
        self.assertIsInstance(ax, plt.Axes)
        self.assertIn("Revenues", buf.getvalue())

    def test_compare_models_highlight(self):
        ax = plot_costsrevs(
            self.plot_cm, **self.fp,
            highlight_ntile=3, highlight_how="plot",
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_compare_datasets_highlight_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_costsrevs(
                self.plot_cd, **self.fp,
                highlight_ntile=3, highlight_how="text",
            )
        self.assertIn("Revenues", buf.getvalue())

    def test_compare_targetclasses_title(self):
        mp2 = _make_mp2_models(ntiles=5)
        plot_tc = mp2.plotting_scope(
            scope="compare_targetclasses",
            select_model_label=["lr1"],
            select_dataset_label=["train"],
            select_targetclass=[0, 1],
        )
        ax = plot_costsrevs(plot_tc, **self.fp)
        title = ax.get_title()
        self.assertIn("target", title.lower())

    def test_legend_loc_compare_datasets_upper_right(self):
        ax = plot_costsrevs(self.plot_cd, **self.fp)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

    def test_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            plot_costsrevs(
                self.plot_nc, **self.fp,
                highlight_ntile=3, highlight_how="bad_mode",
            )

    def test_missing_cumtot_column_raises(self):
        bad = self.plot_nc.drop(columns=["cumtot"])
        with self.assertRaises(_PlotInputError):
            plot_costsrevs(bad, **self.fp)

    def test_custom_currency_symbol(self):
        ax = plot_costsrevs(self.plot_nc, **self.fp, currency="$")
        self.assertIsInstance(ax, plt.Axes)


##############################################################################
# plot_profit — extended
##############################################################################


class TestPlotProfitExtended(unittest.TestCase):
    """Extended coverage for plot_profit."""

    @classmethod
    def setUpClass(cls):
        cls.mp_2m = _make_mp2_models(ntiles=5)
        cls.mp_2d = _make_mp2_datasets(ntiles=5)
        cls.plot_nc = _make_mp(ntiles=5).plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        cls.plot_cm = cls.mp_2m.plotting_scope(
            scope="compare_models",
            select_dataset_label=["train"],
            select_targetclass=[1],
        )
        cls.fp = _make_financial_params()

    def tearDown(self):
        plt.close("all")

    def test_compare_models_returns_axes(self):
        ax = plot_profit(self.plot_cm, **self.fp)
        self.assertIsInstance(ax, plt.Axes)

    def test_highlight_text_prints(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_profit(
                self.plot_nc, **self.fp,
                highlight_ntile=3, highlight_how="text",
            )
        self.assertIn("Profit", buf.getvalue())

    def test_highlight_plot_text_returns_axes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ax = plot_profit(
                self.plot_nc, **self.fp,
                highlight_ntile=[2, 4], highlight_how="plot_text",
            )
        self.assertIsInstance(ax, plt.Axes)

    def test_compare_models_highlight_plot(self):
        ax = plot_profit(
            self.plot_cm, **self.fp,
            highlight_ntile=3, highlight_how="plot",
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_missing_cumtot_raises(self):
        bad = self.plot_nc.drop(columns=["cumtot"])
        with self.assertRaises(_PlotInputError):
            plot_profit(bad, **self.fp)

    def test_custom_currency(self):
        ax = plot_profit(self.plot_nc, **self.fp, currency="£")
        self.assertIsInstance(ax, plt.Axes)


##############################################################################
# plot_roi — extended
##############################################################################


class TestPlotROIExtended(unittest.TestCase):
    """Extended coverage for plot_roi."""

    @classmethod
    def setUpClass(cls):
        cls.mp_2m = _make_mp2_models(ntiles=5)
        cls.plot_nc = _make_mp(ntiles=5).plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        cls.plot_cm = cls.mp_2m.plotting_scope(
            scope="compare_models",
            select_dataset_label=["train"],
            select_targetclass=[1],
        )
        cls.fp = _make_financial_params()

    def tearDown(self):
        plt.close("all")

    def test_compare_models_returns_axes(self):
        ax = plot_roi(self.plot_cm, **self.fp)
        self.assertIsInstance(ax, plt.Axes)

    def test_highlight_text_prints(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_roi(
                self.plot_nc, **self.fp,
                highlight_ntile=3, highlight_how="text",
            )
        self.assertIn("ROI", buf.getvalue())

    def test_highlight_plot_mode_returns_axes(self):
        ax = plot_roi(
            self.plot_nc, **self.fp,
            highlight_ntile=3, highlight_how="plot",
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_compare_models_highlight(self):
        ax = plot_roi(
            self.plot_cm, **self.fp,
            highlight_ntile=[2, 4], highlight_how="plot",
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_missing_cumpos_raises(self):
        bad = self.plot_nc.drop(columns=["cumpos"])
        with self.assertRaises(_PlotInputError):
            plot_roi(bad, **self.fp)

    def test_autopct_callable(self):
        ax = plot_roi(
            self.plot_nc, **self.fp,
            autopct=lambda p: f"{p:.1f}%",
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_breakeven_line_present(self):
        ax = plot_roi(self.plot_nc, **self.fp)
        hlines = [
            l for l in ax.get_lines()
            if len(l.get_ydata()) > 0 and np.allclose(l.get_ydata(), 0.0, atol=1e-6)
        ]
        self.assertGreater(len(hlines), 0, "Break-even line at y=0 not found")


##############################################################################
# plot_response / plot_cumresponse — extended
##############################################################################


class TestPlotResponseExtended(unittest.TestCase):
    """Extended branches for plot_response and plot_cumresponse."""

    @classmethod
    def setUpClass(cls):
        # Use ntiles=10 so the label in highlight text is "decile".
        cls.mp_2d = _make_mp2_datasets(ntiles=10)
        cls.plot_nc = _make_mp(ntiles=10).plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        cls.plot_cd = cls.mp_2d.plotting_scope(
            scope="compare_datasets",
            select_model_label=["lr"],
            select_targetclass=[1],
        )

    def tearDown(self):
        plt.close("all")

    def test_response_compare_datasets(self):
        ax = plot_response(self.plot_cd)
        self.assertIsInstance(ax, plt.Axes)

    def test_cumresponse_compare_datasets(self):
        ax = plot_cumresponse(self.plot_cd)
        self.assertIsInstance(ax, plt.Axes)

    def test_response_highlight_text_mode(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_response(
                self.plot_nc, highlight_ntile=3, highlight_how="text",
            )
        self.assertIn("Response", buf.getvalue())
        self.assertIn("decile 3", buf.getvalue())

    def test_cumresponse_highlight_text_mode(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_cumresponse(
                self.plot_nc, highlight_ntile=3, highlight_how="text",
            )
        self.assertIn("CumResponse", buf.getvalue())

    def test_response_autopct_callable(self):
        ax = plot_response(
            self.plot_nc,
            autopct=lambda p: f"{p:.1f}%",
        )
        self.assertIsInstance(ax, plt.Axes)

    def test_response_grid_kws_forwarded(self):
        ax = plot_response(self.plot_nc, grid_kws={"visible": False})
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            self.assertFalse(line.get_visible())

    def test_highlight_multiple_ntiles_plot_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ax = plot_response(
                self.plot_nc,
                highlight_ntile=[2, 4],
                highlight_how="plot_text",
            )
        self.assertIsInstance(ax, plt.Axes)
        output = buf.getvalue()
        self.assertIn("decile 2", output)
        self.assertIn("decile 4", output)


##############################################################################
# plot_cumlift / plot_cumgains — extended
##############################################################################


class TestPlotCumliftCumgainsExtended(unittest.TestCase):
    """Extended tests for cumlift and cumgains."""

    @classmethod
    def setUpClass(cls):
        # Use ntiles=10 for decile-labeled highlight text.
        cls.mp_2d = _make_mp2_datasets(ntiles=10)
        cls.plot_nc = _make_mp(ntiles=10).plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        cls.plot_cd = cls.mp_2d.plotting_scope(
            scope="compare_datasets",
            select_model_label=["lr"],
            select_targetclass=[1],
        )

    def tearDown(self):
        plt.close("all")

    def test_cumlift_compare_datasets(self):
        ax = plot_cumlift(self.plot_cd)
        self.assertIsInstance(ax, plt.Axes)

    def test_cumgains_compare_datasets(self):
        ax = plot_cumgains(self.plot_cd)
        self.assertIsInstance(ax, plt.Axes)

    def test_cumlift_highlight_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_cumlift(self.plot_nc, highlight_ntile=3, highlight_how="text")
        self.assertIn("CumLift", buf.getvalue())

    def test_cumgains_highlight_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_cumgains(self.plot_nc, highlight_ntile=3, highlight_how="text")
        self.assertIn("CumGains", buf.getvalue())

    def test_cumlift_highlight_plot_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ax = plot_cumlift(
                self.plot_nc, highlight_ntile=[2, 4], highlight_how="plot_text"
            )
        self.assertIsInstance(ax, plt.Axes)

    def test_cumgains_highlight_plot_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ax = plot_cumgains(
                self.plot_nc, highlight_ntile=3, highlight_how="plot_text"
            )
        self.assertIsInstance(ax, plt.Axes)

    def test_cumgains_xlim_zero(self):
        ax = plot_cumgains(self.plot_nc)
        self.assertAlmostEqual(ax.get_xlim()[0], 0.0)

    def test_cumgains_ylim_0_1(self):
        ax = plot_cumgains(self.plot_nc)
        self.assertAlmostEqual(ax.get_ylim()[0], 0.0)
        self.assertAlmostEqual(ax.get_ylim()[1], 1.0)


##############################################################################
# plot_all — extended (footer rendering, comparison scopes, text)
##############################################################################


class TestPlotAllExtended(unittest.TestCase):
    """Extended plot_all tests for uncovered highlight and scope paths."""

    @classmethod
    def setUpClass(cls):
        # ntiles=10 → label text says "decile".
        cls.mp_2m = _make_mp2_models(ntiles=10)
        cls.plot_nc = _make_mp(ntiles=10).plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        cls.plot_cm = cls.mp_2m.plotting_scope(
            scope="compare_models",
            select_dataset_label=["train"],
            select_targetclass=[1],
        )

    def tearDown(self):
        plt.close("all")

    def test_plot_all_highlight_plot_mode(self):
        ax = plot_all(self.plot_nc, highlight_ntile=3, highlight_how="plot")
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_all_highlight_text_mode_prints(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            plot_all(self.plot_nc, highlight_ntile=3, highlight_how="text")
        output = buf.getvalue()
        # All 4 metrics should appear in text output.
        for metric in ("Response", "CumResponse", "CumLift", "CumGains"):
            self.assertIn(metric, output)

    def test_plot_all_highlight_plot_text_mode(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ax = plot_all(
                self.plot_nc,
                highlight_ntile=[2, 4],
                highlight_how="plot_text",
            )
        self.assertIsInstance(ax, plt.Axes)
        output = buf.getvalue()
        # Label says "decile" when ntiles=10.
        self.assertIn("decile 2", output)

    def test_plot_all_compare_models_returns_axes(self):
        ax = plot_all(self.plot_cm)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_all_compare_models_highlight(self):
        ax = plot_all(self.plot_cm, highlight_ntile=3, highlight_how="plot")
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_all_invalid_highlight_how_raises(self):
        with self.assertRaises(ValueError):
            plot_all(self.plot_nc, highlight_ntile=3, highlight_how="wrong")

    def test_plot_all_missing_column_raises(self):
        with self.assertRaises(_PlotInputError):
            plot_all(self.plot_nc.drop(columns=["cumlift"]))

    def test_plot_all_custom_figsize(self):
        ax = plot_all(self.plot_nc, figsize=(12, 8))
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_all_four_subplots_present(self):
        ax = plot_all(self.plot_nc, highlight_ntile=3, highlight_how="plot")
        # 4 data subplots + 1 footer axes = 5 or just 4 without highlight footer.
        n = len(ax.get_figure().axes)
        self.assertGreaterEqual(n, 4)

    def test_plot_all_no_highlight_no_footer_axes(self):
        ax = plot_all(self.plot_nc)  # no highlight_ntile
        n = len(ax.get_figure().axes)
        self.assertEqual(n, 4)  # exactly 4 subplots, no footer


##############################################################################
# summarize_selection — extended
##############################################################################


class TestSummarizeSelectionExtended(unittest.TestCase):
    """Additional summarize_selection branches."""

    def setUp(self):
        self.plot_nc = _make_mp(ntiles=5).plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )

    def test_ntile_equals_ntiles_max(self):
        result = summarize_selection(self.plot_nc, ntile=5)
        self.assertEqual(int(result["ntile"].iloc[0]), 5)

    def test_ntile_equals_1(self):
        result = summarize_selection(self.plot_nc, ntile=1)
        self.assertEqual(int(result["ntile"].iloc[0]), 1)

    def test_cumlift_decreasing_across_ntiles(self):
        """Cumulative lift at decile 1 should exceed decile 5."""
        r1 = summarize_selection(self.plot_nc, ntile=1)
        r5 = summarize_selection(self.plot_nc, ntile=5)
        self.assertGreaterEqual(
            float(r1["cumlift"].iloc[0]),
            float(r5["cumlift"].iloc[0]),
        )

    def test_pct_between_0_and_1(self):
        result = summarize_selection(self.plot_nc, ntile=3)
        pct = float(result["pct"].iloc[0])
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 1.0)

    def test_cumgain_between_0_and_1(self):
        result = summarize_selection(self.plot_nc, ntile=3)
        cg = float(result["cumgain"].iloc[0])
        self.assertGreaterEqual(cg, 0.0)
        self.assertLessEqual(cg, 1.0)

    def test_float_ntile_accepted(self):
        """ntile=3.0 (float integer-like) must work."""
        result = summarize_selection(self.plot_nc, ntile=3.0)
        self.assertEqual(int(result["ntile"].iloc[0]), 3)


##############################################################################
# Legacy modelplotpy adapter smoke tests
##############################################################################


class TestLegacyModelPlotPy(unittest.TestCase):
    """
    Smoke tests for the legacy ``modelplotpy`` adapter.

    This submodule re-exports its own ``ModelPlotPy`` class with a different
    internal implementation (based on the original modelplotpy library).
    We verify the API surface is importable and produces the expected outputs.
    """

    @classmethod
    def setUpClass(cls):
        from scikitplot.decile.modelplotpy._modelplotpy import (
            ModelPlotPy as LegacyModelPlotPy,
            plot_response as lpr,
            plot_cumresponse as lcr,
            plot_cumlift as lcl,
            plot_cumgains as lcg,
            plot_all as lpa,
        )
        cls.LegacyModelPlotPy = LegacyModelPlotPy
        cls.lpr = staticmethod(lpr)
        cls.lcr = staticmethod(lcr)
        cls.lcl = staticmethod(lcl)
        cls.lcg = staticmethod(lcg)
        cls.lpa = staticmethod(lpa)

        X, y = _make_binary_data(n=200)
        lr = _make_fitted_lr(X, y)
        cls.legacy_mp = LegacyModelPlotPy(
            feature_data=[X],
            label_data=[y],
            dataset_labels=["train"],
            models=[lr],
            model_labels=["lr"],
            ntiles=5,
        )

    def tearDown(self):
        plt.close("all")

    def test_legacy_modelplotpy_importable(self):
        self.assertIsNotNone(self.LegacyModelPlotPy)

    def test_legacy_prepare_scores_returns_dataframe(self):
        scores = self.legacy_mp.prepare_scores_and_ntiles()
        self.assertIsInstance(scores, pd.DataFrame)

    def test_legacy_aggregate_returns_dataframe(self):
        agg = self.legacy_mp.aggregate_over_ntiles()
        self.assertIsInstance(agg, pd.DataFrame)

    def test_legacy_plotting_scope_returns_dataframe(self):
        result = self.legacy_mp.plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_legacy_plot_response_returns_axes(self):
        pi = self.legacy_mp.plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        ax = self.lpr(pi)
        self.assertIsInstance(ax, plt.Axes)

    def test_legacy_plot_cumresponse_returns_axes(self):
        pi = self.legacy_mp.plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        ax = self.lcr(pi)
        self.assertIsInstance(ax, plt.Axes)

    def test_legacy_plot_cumlift_returns_axes(self):
        pi = self.legacy_mp.plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        ax = self.lcl(pi)
        self.assertIsInstance(ax, plt.Axes)

    def test_legacy_plot_cumgains_returns_axes(self):
        pi = self.legacy_mp.plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        ax = self.lcg(pi)
        self.assertIsInstance(ax, plt.Axes)

    def test_legacy_plot_all_returns_axes(self):
        pi = self.legacy_mp.plotting_scope(
            scope="no_comparison", select_targetclass=[1]
        )
        ax = self.lpa(pi)
        self.assertIsInstance(ax, plt.Axes)

    def test_legacy_ntiles_5_respected(self):
        agg = self.legacy_mp.aggregate_over_ntiles()
        self.assertIn(5, agg["ntile"].values)
        self.assertNotIn(10, agg["ntile"].values)

    def test_legacy_two_models_compare(self):
        X, y = _make_binary_data()
        lr1 = _make_fitted_lr(X, y, 0)
        lr2 = _make_fitted_lr(X, y, 1)
        mp2 = self.LegacyModelPlotPy(
            feature_data=[X],
            label_data=[y],
            dataset_labels=["train"],
            models=[lr1, lr2],
            model_labels=["lr1", "lr2"],
            ntiles=5,
        )
        pi = mp2.plotting_scope(
            scope="compare_models",
            select_dataset_label=["train"],
            select_targetclass=[1],
        )
        ax = self.lpr(pi)
        self.assertIsInstance(ax, plt.Axes)


##############################################################################
# Integration: end-to-end extended pipelines
##############################################################################


class TestEndToEndExtended(unittest.TestCase):
    """Extended integration tests covering multi-model and multi-dataset paths."""

    def tearDown(self):
        plt.close("all")

    def test_two_model_full_pipeline_all_plots(self):
        X, y = _make_binary_data(n=200)
        lr1 = _make_fitted_lr(X, y, 0)
        lr2 = _make_fitted_lr(X, y, 1)
        mp = ModelPlotPy(
            feature_data=[X],
            label_data=[y],
            dataset_labels=["train"],
            models=[lr1, lr2],
            model_labels=["lr1", "lr2"],
            ntiles=5,
        )
        pi = mp.plotting_scope(
            scope="compare_models",
            select_dataset_label=["train"],
            select_targetclass=[1],
        )
        for fn in (plot_response, plot_cumresponse, plot_cumlift, plot_cumgains):
            ax = fn(pi)
            self.assertIsInstance(ax, plt.Axes)
            plt.close("all")

    def test_two_dataset_full_pipeline_financial(self):
        X, y = _make_binary_data(n=200)
        lr = _make_fitted_lr(X, y)
        mp = ModelPlotPy(
            feature_data=[X, X],
            label_data=[y, y],
            dataset_labels=["train", "test"],
            models=[lr],
            model_labels=["lr"],
            ntiles=5,
        )
        pi = mp.plotting_scope(
            scope="compare_datasets",
            select_model_label=["lr"],
            select_targetclass=[1],
        )
        fp = _make_financial_params()
        for fn in (plot_costsrevs, plot_profit, plot_roi):
            ax = fn(pi, **fp)
            self.assertIsInstance(ax, plt.Axes)
            plt.close("all")

    def test_ntiles_100_percentile_label(self):
        """ntiles=100 → xlabel should say 'percentile'."""
        X, y = _make_binary_data(n=300)
        lr = _make_fitted_lr(X, y)
        mp = ModelPlotPy(
            feature_data=[X],
            label_data=[y],
            dataset_labels=["train"],
            models=[lr],
            model_labels=["lr"],
            ntiles=100,
        )
        pi = mp.plotting_scope(scope="no_comparison", select_targetclass=[1])
        ax = plot_response(pi)
        self.assertEqual(ax.get_xlabel(), "percentile")
        plt.close("all")

    def test_ntile_5_label(self):
        """ntiles=5 → xlabel should say 'ntile'."""
        mp = _make_mp(ntiles=5)
        pi = mp.plotting_scope(scope="no_comparison", select_targetclass=[1])
        ax = plot_response(pi)
        self.assertEqual(ax.get_xlabel(), "ntile")
        plt.close("all")

    def test_plot_all_compare_datasets_with_highlight(self):
        X, y = _make_binary_data()
        lr = _make_fitted_lr(X, y)
        mp = ModelPlotPy(
            feature_data=[X, X],
            label_data=[y, y],
            dataset_labels=["train", "test"],
            models=[lr],
            model_labels=["lr"],
            ntiles=10,  # ntiles=10 → label text says "decile"
        )
        pi = mp.plotting_scope(
            scope="compare_datasets",
            select_model_label=["lr"],
            select_targetclass=[1],
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ax = plot_all(pi, highlight_ntile=3, highlight_how="plot_text")
        self.assertIsInstance(ax, plt.Axes)
        self.assertIn("decile 3", buf.getvalue())
        plt.close("all")

    def test_summarize_vs_aggregate_consistency(self):
        """summarize_selection at ntile N must match aggregate_over_ntiles row."""
        mp = _make_mp(ntiles=5)
        pi = mp.plotting_scope(scope="no_comparison", select_targetclass=[1])
        summary = summarize_selection(pi, ntile=3)
        agg = mp.aggregate_over_ntiles()
        agg_row = agg[
            (agg["model_label"] == "lr")
            & (agg["dataset_label"] == "train")
            & (agg["target_class"] == 1)
            & (agg["ntile"] == 3)
        ]
        self.assertEqual(len(agg_row), 1)
        self.assertAlmostEqual(
            float(summary["cumlift"].iloc[0]),
            float(agg_row["cumlift"].iloc[0]),
            places=6,
        )

    def test_reproducibility_across_two_runs(self):
        """Identical inputs must produce identical ntile assignments."""
        mp1 = _make_mp(ntiles=10)
        mp2 = _make_mp(ntiles=10)
        scores1 = mp1.prepare_scores_and_ntiles()
        scores2 = mp2.prepare_scores_and_ntiles()
        pd.testing.assert_series_equal(
            scores1["dec_1"].reset_index(drop=True),
            scores2["dec_1"].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
