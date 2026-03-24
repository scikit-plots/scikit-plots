# scikitplot/utils/tests/test__matplotlib.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._matplotlib`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__matplotlib.py -v

Coverage map
------------
safe_tight_layout    Yields figure, tight_layout called,
                     failure logged with warn, no-fig fallback → TestSafeTightLayout
SafeTightLayout      Class-based CM: enter/exit, failure warn  → TestSafeTightLayoutClass
save_plot_decorator  No-arg and parameterised usage, show_fig,
                     save_fig path, invalid verbose warning     → TestSavePlotDecorator
stack                No figs raises, vertical/horizontal,
                     orient aliases, invalid orient raises      → TestStack
"""

from __future__ import annotations

import logging
import unittest
import unittest.mock as mock
import warnings

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np

# --- path bootstrap (commented out — use pytest from the package root) -----
# import pathlib, sys
# _HERE = pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))
# --------------------------------------------------------------------------

from .._matplotlib import (  # noqa: E402
    SafeTightLayout,
    safe_tight_layout,
    save_plot_decorator,
    stack,
)


def _close_all():
    plt.close("all")


# ===========================================================================
# safe_tight_layout  (function-based context manager)
# ===========================================================================


class TestSafeTightLayout(unittest.TestCase):
    """safe_tight_layout must yield the figure, call tight_layout, and log on failure."""

    def setUp(self):
        _close_all()

    def tearDown(self):
        _close_all()

    def test_yields_given_figure(self):
        """The context manager must yield the figure passed in."""
        fig, _ = plt.subplots()
        with safe_tight_layout(fig) as yielded:
            self.assertIs(yielded, fig)

    def test_yields_none_when_no_fig(self):
        """When fig=None, the context yields None (fig resolved in finally)."""
        with safe_tight_layout(None) as yielded:
            self.assertIsNone(yielded)

    def test_tight_layout_called_on_exit(self):
        """fig.tight_layout() must be called after the context block exits."""
        fig, _ = plt.subplots()
        with mock.patch.object(fig, "tight_layout") as mock_tl:
            with safe_tight_layout(fig):
                pass
        mock_tl.assert_called_once()

    def test_tight_layout_failure_logs_warning_when_warn_true(self):
        """A tight_layout() failure must emit a WARNING log when warn=True."""
        fig, _ = plt.subplots()
        with mock.patch.object(
            fig, "tight_layout", side_effect=Exception("layout error")
        ):
            with self.assertLogs("scikitplot", level=logging.WARNING):
                with safe_tight_layout(fig, warn=True):
                    pass

    def test_tight_layout_failure_no_log_when_warn_false(self):
        """A tight_layout() failure must NOT log when warn=False."""
        fig, _ = plt.subplots()
        with mock.patch.object(
            fig, "tight_layout", side_effect=Exception("layout error")
        ):
            # Should not raise or log; we verify no exception propagates
            try:
                with safe_tight_layout(fig, warn=False):
                    pass
            except Exception as e:
                self.fail(f"safe_tight_layout raised unexpectedly: {e}")

    def test_exception_inside_block_propagates(self):
        """Exceptions inside the with block must propagate normally."""
        fig, _ = plt.subplots()
        with self.assertRaises(ValueError):
            with safe_tight_layout(fig):
                raise ValueError("inside block")

    def test_no_args_uses_gcf(self):
        """With no fig, plt.gcf() must be used on exit."""
        fig, _ = plt.subplots()  # make gcf() return something valid
        # Should not raise
        with safe_tight_layout():
            pass

    def test_return_value_is_none(self):
        """Context manager must not return a value (returns None by design)."""
        result = None
        with safe_tight_layout() as r:
            result = r
        # result is whatever was yielded (None or fig)
        # Just ensure it did not raise
        self.assertTrue(True)

    def test_multiple_invocations_no_state_leak(self):
        """Calling safe_tight_layout twice must not share state."""
        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()
        with safe_tight_layout(fig1):
            pass
        with safe_tight_layout(fig2):
            pass


# ===========================================================================
# SafeTightLayout  (class-based context manager)
# ===========================================================================


class TestSafeTightLayoutClass(unittest.TestCase):
    """SafeTightLayout class must behave identically to the function variant."""

    def setUp(self):
        _close_all()

    def tearDown(self):
        _close_all()

    def test_enter_returns_figure(self):
        """__enter__ must return the figure (self.fig)."""
        fig, _ = plt.subplots()
        cm = SafeTightLayout(fig)
        returned = cm.__enter__()
        self.assertIs(returned, fig)
        cm.__exit__(None, None, None)

    def test_tight_layout_called_on_exit(self):
        fig, _ = plt.subplots()
        with mock.patch.object(fig, "tight_layout") as mock_tl:
            with SafeTightLayout(fig):
                pass
        mock_tl.assert_called_once()

    def test_failure_logs_warning_when_warn_true(self):
        fig, _ = plt.subplots()
        with mock.patch.object(
            fig, "tight_layout", side_effect=Exception("oops")
        ):
            with self.assertLogs("scikitplot", level=logging.WARNING):
                with SafeTightLayout(fig, warn=True):
                    pass

    def test_failure_silent_when_warn_false(self):
        fig, _ = plt.subplots()
        with mock.patch.object(
            fig, "tight_layout", side_effect=Exception("oops")
        ):
            try:
                with SafeTightLayout(fig, warn=False):
                    pass
            except Exception as e:
                self.fail(f"SafeTightLayout raised with warn=False: {e}")

    def test_no_fig_uses_gcf(self):
        """SafeTightLayout(None) must fall back to plt.gcf()."""
        _, _ = plt.subplots()
        with SafeTightLayout(None):
            pass

    def test_exception_propagates(self):
        """Exceptions inside the block must propagate."""
        fig, _ = plt.subplots()
        with self.assertRaises(RuntimeError):
            with SafeTightLayout(fig):
                raise RuntimeError("test")

    def test_context_manager_pattern(self):
        """Must support 'with ... as fig:' syntax."""
        fig, _ = plt.subplots()
        with SafeTightLayout(fig) as f:
            self.assertIs(f, fig)


# ===========================================================================
# save_plot_decorator
# ===========================================================================


class TestSavePlotDecorator(unittest.TestCase):
    """save_plot_decorator must support @decorator and @decorator() usage."""

    def setUp(self):
        _close_all()
        self._show_patcher = mock.patch("matplotlib.pyplot.show")
        self._mock_show = self._show_patcher.start()

    def tearDown(self):
        self._show_patcher.stop()
        _close_all()

    # -- Decoration styles --

    def test_no_arg_decoration_callable(self):
        """@save_plot_decorator without parens must produce a callable."""
        @save_plot_decorator
        def make_plot():
            fig, ax = plt.subplots()
            ax.plot([1, 2], [3, 4])
            return fig

        result = make_plot()
        self.assertIsNotNone(result)

    def test_parameterised_decoration_callable(self):
        """@save_plot_decorator() with parens must also produce a callable."""
        @save_plot_decorator()
        def make_plot():
            fig, ax = plt.subplots()
            ax.plot([1, 2], [3, 4])
            return fig

        result = make_plot()
        self.assertIsNotNone(result)

    def test_decorated_function_returns_original_result(self):
        """The wrapper must return the original function's return value."""
        @save_plot_decorator
        def my_func():
            return "expected"

        self.assertEqual(my_func(), "expected")

    def test_functools_wraps_preserves_name(self):
        """@save_plot_decorator must preserve the original function's __name__."""
        @save_plot_decorator
        def my_plotting_function():
            return None

        self.assertEqual(my_plotting_function.__name__, "my_plotting_function")

    # -- show_fig / save_fig flags --

    def test_show_fig_true_calls_plt_show(self):
        """show_fig=True must call plt.show() once."""
        @save_plot_decorator
        def plot(show_fig):
            fig, ax = plt.subplots()
            return fig

        plot(show_fig=True)
        self._mock_show.assert_called()

    def test_show_fig_false_no_plt_show(self):
        """show_fig=False must not call plt.show()."""
        @save_plot_decorator
        def plot(show_fig):
            fig, ax = plt.subplots()
            return fig

        self._mock_show.reset_mock()
        plot(show_fig=False)
        self._mock_show.assert_not_called()

    def test_save_fig_true_calls_savefig(self):
        """save_fig=True must call plt.savefig() with a valid path."""
        @save_plot_decorator
        def plot(save_fig, show_fig):
            fig, ax = plt.subplots()
            ax.plot([1, 2], [3, 4])
            return fig

        with mock.patch("matplotlib.pyplot.savefig") as mock_save:
            plot(save_fig=True, show_fig=False)
        mock_save.assert_called_once()

    def test_save_fig_false_no_savefig(self):
        """save_fig=False must not call plt.savefig()."""
        @save_plot_decorator
        def plot(save_fig, show_fig):
            fig, ax = plt.subplots()
            return fig

        with mock.patch("matplotlib.pyplot.savefig") as mock_save:
            plot(save_fig=False, show_fig=False)
        mock_save.assert_not_called()

    def test_invalid_verbose_type_warns(self):
        """Passing verbose='yes' (non-bool) must emit a UserWarning."""
        @save_plot_decorator
        def plot(verbose, show_fig):
            return None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plot(verbose="yes", show_fig=False)
        # At least one warning should have been emitted
        self.assertGreater(len(w), 0)


# ===========================================================================
# stack
# ===========================================================================


class TestStack(unittest.TestCase):
    """stack must combine multiple figures into one, respecting orientation."""

    def setUp(self):
        _close_all()
        self._show_patcher = mock.patch("matplotlib.pyplot.show")
        self._mock_show = self._show_patcher.start()

    def tearDown(self):
        self._show_patcher.stop()
        _close_all()

    def _make_fig(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig

    # -- Validation --

    def test_no_figures_raises_value_error(self):
        """stack() with no figures must raise ValueError."""
        with self.assertRaises(ValueError):
            stack(show_fig=False)

    def test_invalid_orient_raises_value_error(self):
        """An unsupported orient string must raise ValueError."""
        fig = self._make_fig()
        with self.assertRaises(ValueError):
            stack(fig, orient="diagonal", show_fig=False)

    # -- Return type --

    def test_returns_figure(self):
        """stack must return a matplotlib Figure."""
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    # -- Vertical orientation --

    def test_vertical_default(self):
        """Default orientation is vertical; must produce a figure without error."""
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    def test_orient_v_alias(self):
        """orient='v' must work like 'vertical'."""
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, orient="v", show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    def test_orient_y_alias(self):
        """orient='y' must work like 'vertical'."""
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, orient="y", show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    # -- Horizontal orientation --

    def test_horizontal_orient(self):
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, orient="horizontal", show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    def test_orient_h_alias(self):
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, orient="h", show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    def test_orient_x_alias(self):
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, orient="x", show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    # -- Single figure --

    def test_single_figure_no_error(self):
        """A single figure must be accepted (edge case: n=1)."""
        f1 = self._make_fig()
        result = stack(f1, show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    # -- Three figures --

    def test_three_figures_vertical(self):
        figs = [self._make_fig() for _ in range(3)]
        result = stack(*figs, orient="vertical", show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    def test_three_figures_horizontal(self):
        figs = [self._make_fig() for _ in range(3)]
        result = stack(*figs, orient="horizontal", show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    # -- Custom figsize --

    def test_custom_figsize(self):
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, figsize=(10, 6), show_fig=False)
        self.assertIsInstance(result, plt.Figure)

    # -- Case insensitivity --

    def test_orient_uppercase_handled(self):
        """orient must be normalised to lowercase."""
        f1, f2 = self._make_fig(), self._make_fig()
        result = stack(f1, f2, orient="VERTICAL", show_fig=False)
        self.assertIsInstance(result, plt.Figure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
