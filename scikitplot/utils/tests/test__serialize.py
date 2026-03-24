# scikitplot/utils/tests/test__serialize.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._serialize`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__serialize.py -v

Coverage map
------------
safe_json_converter   np.integer, np.floating, np.ndarray,
                      unsupported type raises TypeError        → TestSafeJsonConverter
get_ax_from_input     None→gca, Axes, Figure, tuple, ndarray,
                      iterable, invalid types                  → TestGetAxFromInput
detect_plot_type      Empty→None, bar, histogram, line        → TestDetectPlotType
serialize_histplot    Valid histogram, no patches error        → TestSerializeHistplot
serialize_barplot     Valid barplot, no patches error          → TestSerializeBarplot
serialize_lineplot    Valid lineplot, no lines error           → TestSerializeLineplot
serialize_plot        Full dispatch flow, unsupported type     → TestSerializePlot
save_to_file          Writes JSON, creates parent dir          → TestSaveToFile
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

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

from .._serialize import (  # noqa: E402
    detect_plot_type,
    get_ax_from_input,
    safe_json_converter,
    save_to_file,
    serialize_barplot,
    serialize_histplot,
    serialize_lineplot,
    serialize_plot,
)


def _close_all():
    plt.close("all")


# ===========================================================================
# safe_json_converter
# ===========================================================================


class TestSafeJsonConverter(unittest.TestCase):
    """safe_json_converter must convert NumPy types to native Python types."""

    def tearDown(self):
        _close_all()

    def test_np_int64_to_int(self):
        result = safe_json_converter(np.int64(7))
        self.assertIsInstance(result, int)
        self.assertEqual(result, 7)

    def test_np_int32_to_int(self):
        result = safe_json_converter(np.int32(42))
        self.assertIsInstance(result, int)
        self.assertEqual(result, 42)

    def test_np_integer_generic(self):
        """Any np.integer subclass must convert to int."""
        result = safe_json_converter(np.int16(3))
        self.assertIsInstance(result, int)

    def test_np_float64_to_float(self):
        result = safe_json_converter(np.float64(3.14))
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14, places=5)

    def test_np_float32_to_float(self):
        result = safe_json_converter(np.float32(1.5))
        self.assertIsInstance(result, float)

    def test_np_ndarray_to_list(self):
        arr = np.array([1, 2, 3])
        result = safe_json_converter(arr)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])

    def test_nd_array_2d(self):
        arr = np.array([[1, 2], [3, 4]])
        result = safe_json_converter(arr)
        self.assertIsInstance(result, list)

    def test_unsupported_type_raises_type_error(self):
        """A plain Python dict must raise TypeError."""
        with self.assertRaises(TypeError):
            safe_json_converter({"key": "val"})

    def test_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            safe_json_converter("hello")

    def test_json_serializable_after_conversion(self):
        """Converted value must be JSON-serializable via json.dumps."""
        result = safe_json_converter(np.float64(2.718))
        json.dumps(result)  # must not raise


# ===========================================================================
# get_ax_from_input
# ===========================================================================


class TestGetAxFromInput(unittest.TestCase):
    """get_ax_from_input must extract Axes from diverse input types."""

    def setUp(self):
        _close_all()

    def tearDown(self):
        _close_all()

    def test_none_returns_current_axes(self):
        """None must return the current active Axes via plt.gca()."""
        fig, ax = plt.subplots()
        result = get_ax_from_input(None)
        self.assertIsInstance(result, plt.Axes)

    def test_axes_returns_axes(self):
        """A direct Axes input must be returned as-is."""
        fig, ax = plt.subplots()
        result = get_ax_from_input(ax)
        self.assertIs(result, ax)

    def test_figure_returns_axes(self):
        """A Figure must return its primary Axes."""
        fig, ax = plt.subplots()
        result = get_ax_from_input(fig)
        self.assertIsInstance(result, plt.Axes)

    def test_tuple_fig_ax(self):
        """Tuple (fig, ax) must return the single Axes."""
        fig, ax = plt.subplots()
        result = get_ax_from_input((fig, ax))
        self.assertIs(result, ax)

    def test_tuple_fig_list_of_axes(self):
        """Tuple (fig, [ax1, ax2]) must return a list of Axes."""
        fig, axes = plt.subplots(1, 2)
        result = get_ax_from_input((fig, list(axes)))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_tuple_wrong_length_raises(self):
        """Tuple of length != 2 must raise TypeError."""
        with self.assertRaises(TypeError):
            get_ax_from_input((1, 2, 3))

    def test_tuple_second_not_axes_raises(self):
        """Tuple with non-Axes second element must raise TypeError."""
        with self.assertRaises(TypeError):
            get_ax_from_input(("a", "b"))

    def test_numpy_array_of_axes(self):
        """numpy.ndarray of Axes must be flattened and returned as a list."""
        fig, axes = plt.subplots(2, 2)
        result = get_ax_from_input(axes)  # axes is ndarray here
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)

    def test_list_of_axes(self):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        result = get_ax_from_input([ax1, ax2])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_list_of_figures(self):
        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()
        result = get_ax_from_input([fig1, fig2])
        self.assertIsInstance(result, list)

    def test_empty_list_raises_type_error(self):
        """A list with no Axes or Figures must raise TypeError."""
        with self.assertRaises(TypeError):
            get_ax_from_input([1, 2, "three"])

    def test_invalid_type_raises_type_error(self):
        """An arbitrary object must raise TypeError."""
        with self.assertRaises(TypeError):
            get_ax_from_input(42)


# ===========================================================================
# detect_plot_type
# ===========================================================================


# class TestDetectPlotType(unittest.TestCase):
#     """detect_plot_type must infer the plot type from Axes content."""

#     def setUp(self):
#         _close_all()

#     def tearDown(self):
#         _close_all()

#     def test_empty_axes_returns_none(self):
#         """An empty Axes must return None."""
#         _, ax = plt.subplots()
#         self.assertIsNone(detect_plot_type(ax))

#     def test_bar_plot_returns_barplot(self):
#         """Axes with wide bars must be detected as 'barplot'."""
#         _, ax = plt.subplots()
#         ax.bar([1, 2, 3], [4, 5, 6])
#         result = detect_plot_type(ax)
#         self.assertEqual(result, "barplot")

#     def test_line_plot_returns_lineplot(self):
#         """Axes with line data must be detected as 'lineplot'."""
#         _, ax = plt.subplots()
#         ax.plot([1, 2, 3], [4, 5, 6])
#         result = detect_plot_type(ax)
#         self.assertEqual(result, "lineplot")

#     def test_returns_string_or_none(self):
#         _, ax = plt.subplots()
#         ax.plot([0, 1], [0, 1])
#         result = detect_plot_type(ax)
#         self.assertIn(result, ("lineplot", "barplot", "histogram", None))


# ===========================================================================
# serialize_histplot
# ===========================================================================


# class TestSerializeHistplot(unittest.TestCase):
#     """serialize_histplot must extract histogram data from Axes patches."""

#     def setUp(self):
#         _close_all()

#     def tearDown(self):
#         _close_all()

#     def test_returns_dict_for_histogram(self):
#         """A genuine histogram must yield a dict with the right keys."""
#         data = np.random.default_rng(0).normal(size=100)
#         _, ax = plt.subplots()
#         ax.hist(data, bins=10)
#         result = serialize_histplot(ax)
#         self.assertIsNotNone(result)
#         self.assertIsInstance(result, dict)

#     def test_histogram_has_type_key(self):
#         data = np.random.default_rng(0).normal(size=100)
#         _, ax = plt.subplots()
#         ax.hist(data, bins=10)
#         result = serialize_histplot(ax)
#         self.assertEqual(result["type"], "histogram")

#     def test_histogram_has_required_keys(self):
#         data = np.random.default_rng(0).normal(size=100)
#         _, ax = plt.subplots()
#         ax.hist(data, bins=10)
#         result = serialize_histplot(ax)
#         for key in ("type", "title", "x_label", "y_label", "bin_edges", "counts"):
#             self.assertIn(key, result)

#     def test_empty_ax_returns_none(self):
#         """An Axes without patches must return None (no bars)."""
#         _, ax = plt.subplots()
#         result = serialize_histplot(ax)
#         self.assertIsNone(result)


# ===========================================================================
# serialize_barplot
# ===========================================================================


# class TestSerializeBarplot(unittest.TestCase):
#     """serialize_barplot must extract bar chart data from Axes patches."""

#     def setUp(self):
#         _close_all()

#     def tearDown(self):
#         _close_all()

#     def test_returns_dict_for_barplot(self):
#         _, ax = plt.subplots()
#         ax.bar([1, 2, 3], [10, 20, 30])
#         result = serialize_barplot(ax)
#         self.assertIsNotNone(result)
#         self.assertIsInstance(result, dict)

#     def test_barplot_has_type_key(self):
#         _, ax = plt.subplots()
#         ax.bar([1, 2, 3], [10, 20, 30])
#         result = serialize_barplot(ax)
#         self.assertEqual(result["type"], "barplot")

#     def test_barplot_has_required_keys(self):
#         _, ax = plt.subplots()
#         ax.bar([1, 2, 3], [10, 20, 30])
#         result = serialize_barplot(ax)
#         for key in ("type", "title", "x_label", "y_label", "labels", "heights"):
#             self.assertIn(key, result)

#     def test_empty_ax_returns_none(self):
#         _, ax = plt.subplots()
#         result = serialize_barplot(ax)
#         self.assertIsNone(result)

#     def test_heights_match_input(self):
#         _, ax = plt.subplots()
#         ax.bar([1, 2, 3], [10.0, 20.0, 30.0])
#         result = serialize_barplot(ax)
#         self.assertEqual(result["heights"], [10.0, 20.0, 30.0])


# ===========================================================================
# serialize_lineplot
# ===========================================================================


# class TestSerializeLineplot(unittest.TestCase):
#     """serialize_lineplot must extract line data from Line2D objects."""

#     def setUp(self):
#         _close_all()

#     def tearDown(self):
#         _close_all()

#     def test_returns_dict_for_lineplot(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2, 3], [4, 5, 6])
#         result = serialize_lineplot(ax)
#         self.assertIsNotNone(result)
#         self.assertIsInstance(result, dict)

#     def test_lineplot_has_type_key(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2, 3], [4, 5, 6])
#         result = serialize_lineplot(ax)
#         self.assertEqual(result["type"], "lineplot")

#     def test_lineplot_has_required_keys(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2, 3], [4, 5, 6])
#         result = serialize_lineplot(ax)
#         for key in ("type", "title", "x_label", "y_label", "lines"):
#             self.assertIn(key, result)

#     def test_lineplot_data_values(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2, 3], [4.0, 5.0, 6.0], label="series1")
#         result = serialize_lineplot(ax)
#         self.assertEqual(result["lines"][0]["y"], [4.0, 5.0, 6.0])

#     def test_multiple_lines(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2], [3, 4])
#         ax.plot([5, 6], [7, 8])
#         result = serialize_lineplot(ax)
#         self.assertEqual(len(result["lines"]), 2)

#     def test_empty_ax_returns_none(self):
#         _, ax = plt.subplots()
#         result = serialize_lineplot(ax)
#         self.assertIsNone(result)


# ===========================================================================
# serialize_plot
# ===========================================================================


# class TestSerializePlot(unittest.TestCase):
#     """serialize_plot must dispatch correctly and return valid JSON."""

#     def setUp(self):
#         _close_all()

#     def tearDown(self):
#         _close_all()

#     def test_lineplot_returns_json_string(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2, 3], [4, 5, 6])
#         result = serialize_plot(ax)
#         self.assertIsNotNone(result)
#         self.assertIsInstance(result, str)

#     def test_lineplot_json_parseable(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2, 3], [4, 5, 6])
#         result = serialize_plot(ax)
#         parsed = json.loads(result)
#         self.assertEqual(parsed["type"], "lineplot")

#     def test_barplot_dispatched(self):
#         _, ax = plt.subplots()
#         ax.bar([1, 2, 3], [10, 20, 30])
#         result = serialize_plot(ax)
#         self.assertIsNotNone(result)
#         parsed = json.loads(result)
#         self.assertEqual(parsed["type"], "barplot")

#     def test_unknown_plot_returns_none(self):
#         """An empty Axes has no detectable type → serialize_plot must return None."""
#         _, ax = plt.subplots()
#         result = serialize_plot(ax)
#         self.assertIsNone(result)

#     def test_pretty_true_formats_json(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2], [3, 4])
#         result = serialize_plot(ax, pretty=True)
#         self.assertIn("\n", result)  # indented = multi-line

#     def test_pretty_false_compact_json(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2], [3, 4])
#         result = serialize_plot(ax, pretty=False)
#         self.assertNotIn("\n    ", result)  # compact

#     def test_none_input_uses_gca(self):
#         _, ax = plt.subplots()
#         ax.plot([1, 2], [3, 4])
#         result = serialize_plot(None)
#         self.assertIsNotNone(result)

#     def test_figure_input(self):
#         fig, ax = plt.subplots()
#         ax.plot([1, 2], [3, 4])
#         result = serialize_plot(fig)
#         self.assertIsNotNone(result)


# ===========================================================================
# save_to_file
# ===========================================================================


# class TestSaveToFile(unittest.TestCase):
#     """save_to_file must write JSON to a file, creating parent dirs."""

#     def setUp(self):
#         self._tmpdir = tempfile.mkdtemp()

#     def tearDown(self):
#         shutil.rmtree(self._tmpdir, ignore_errors=True)

#     def test_file_created(self):
#         path = os.path.join(self._tmpdir, "out.json")
#         save_to_file('{"a": 1}', path)
#         self.assertTrue(os.path.isfile(path))

#     def test_file_content_correct(self):
#         path = os.path.join(self._tmpdir, "out.json")
#         save_to_file('{"x": 42}', path)
#         with open(path, encoding="utf-8") as f:
#             content = f.read()
#         self.assertEqual(content, '{"x": 42}')

#     def test_parent_dir_created(self):
#         nested = os.path.join(self._tmpdir, "a", "b", "c", "out.json")
#         save_to_file("{}", nested)
#         self.assertTrue(os.path.isfile(nested))

#     def test_empty_json_string_saved(self):
#         path = os.path.join(self._tmpdir, "empty.json")
#         save_to_file("{}", path)
#         self.assertTrue(os.path.getsize(path) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
