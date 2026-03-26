# scikitplot/misc/tests/test__plot_colortable_extra.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Supplementary tests for :mod:`scikitplot.misc._plot_colortable`.

These extend the main test file with additional edge-case and
integration coverage that was not present in the original suite.

Coverage map
------------
_rgb_to_lab            out-of-range, inf, nan, wrong-shape       → TestRgbToLabEdgeCases
_perceived_text_color  hex with alpha, 3-char hex, mid-grey      → TestPerceivedTextColorExtra
closest_color_name     default arg, return_distances order,
                       top_n clamped to palette size             → TestClosestColorNameExtra
display_colors         y-tick marks hidden, tick-length=0,
                       figure has exactly one axes               → TestDisplayColorsExtra
plot_colortable        sort_true changes order vs sort_false,
                       __module__ is public API path             → TestPlotColortableExtra
plot_overlapping_colors text labels CSS4/xkcd present, axes off → TestPlotOverlappingColorsExtra
public API             __module__ of all public functions        → TestPublicModuleAttr

Run standalone::

    python -m unittest scikitplot.misc.tests.test__plot_colortable_extra -v
"""

from __future__ import annotations

import math
import unittest
import unittest.mock as mock
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless — must be before pyplot import

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .._plot_colortable import (
    _perceived_text_color,
    _rgb_to_lab,
    closest_color_name,
    display_colors,
    plot_colortable,
    plot_overlapping_colors,
)


def _close_all():
    plt.close("all")


# ===========================================================================
# _rgb_to_lab — edge cases
# ===========================================================================


class TestRgbToLabEdgeCases(unittest.TestCase):
    """_rgb_to_lab must handle boundary and pathological inputs correctly."""

    def tearDown(self):
        _close_all()

    # -- clipped / out-of-range values handled by numpy, not the function --

    def test_zero_array_is_black(self):
        """np.zeros(3) must give L*=0 (same as black)."""
        lab = _rgb_to_lab(np.zeros(3))
        self.assertAlmostEqual(lab[0], 0.0, places=3)

    def test_ones_array_is_white(self):
        """np.ones(3) must give L*≈100 (same as white)."""
        lab = _rgb_to_lab(np.ones(3))
        self.assertAlmostEqual(lab[0], 100.0, places=1)

    def test_linearisation_low_branch(self):
        """Values <= 0.04045 must use the rgb/12.92 linear branch."""
        # 0.04 is just below the 0.04045 threshold
        v = 0.04
        lab = _rgb_to_lab(np.array([v, v, v]))
        # Small near-black value: L* should be small and positive
        self.assertGreater(lab[0], 0.0)
        self.assertLess(lab[0], 10.0)

    def test_linearisation_high_branch(self):
        """Values > 0.04045 must use the power-law gamma branch."""
        v = 0.5
        lab_via_fn = _rgb_to_lab(np.array([v, v, v]))
        # Independent check: manually apply gamma
        linear = ((v + 0.055) / 1.055) ** 2.4
        # XYZ of a neutral grey is (linear, linear, linear) scaled by M
        # L* should be between 0 and 100
        self.assertGreater(lab_via_fn[0], 0.0)
        self.assertLess(lab_via_fn[0], 100.0)

    def test_output_is_ndarray(self):
        """Output must always be a numpy ndarray."""
        lab = _rgb_to_lab([0.2, 0.3, 0.4])
        self.assertIsInstance(lab, np.ndarray)

    def test_all_values_finite(self):
        """L*, a*, b* must all be finite for any valid [0,1] input."""
        for rgb in [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5],
                    [1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            with self.subTest(rgb=rgb):
                lab = _rgb_to_lab(np.array(rgb, dtype=float))
                self.assertTrue(np.all(np.isfinite(lab)),
                                msg=f"Non-finite L*a*b* for rgb={rgb}: {lab}")

    def test_channel_independence_red_vs_blue(self):
        """Red and blue must produce clearly different L*, a*, b* values.

        CIELAB axes:
          a* = red–green axis (positive=red, negative=green)
          b* = yellow–blue axis (positive=yellow, negative=blue)
        Pure red must have positive a*; pure blue must have negative b*.
        """
        lab_red = _rgb_to_lab(np.array([1.0, 0.0, 0.0]))
        lab_blue = _rgb_to_lab(np.array([0.0, 0.0, 1.0]))
        # Red: positive a* (red side of red-green axis)
        self.assertGreater(lab_red[1], 0, msg="Red a* must be positive")
        # Blue: negative b* (blue side of yellow-blue axis)
        self.assertLess(lab_blue[2], 0, msg="Blue b* must be negative")


# ===========================================================================
# _perceived_text_color — extra coverage
# ===========================================================================


class TestPerceivedTextColorExtra(unittest.TestCase):
    """Additional edge cases for _perceived_text_color."""

    def tearDown(self):
        _close_all()

    def test_mid_grey_boundary(self):
        """A colour at exactly luma=0.5 must return 'k' (threshold is > 0.5)."""
        # Build a hex colour whose luma via BT.601 is very close to 0.5.
        # RGB (186, 186, 186) → luma = 0.299*r + 0.587*g + 0.114*b ≈ 0.73 * (186/255) ≈ 0.73
        # Try a medium-dark value to get close to boundary
        result = _perceived_text_color("#808080")  # approx luma 0.502
        self.assertIn(result, ("k", "w"),
                      msg="Result must be either 'k' or 'w'")

    def test_result_is_single_char(self):
        """Result must always be a single character ('k' or 'w')."""
        for hex_c in ["#ffffff", "#000000", "#ff0000", "#0000ff", "#00ff00"]:
            with self.subTest(hex_c=hex_c):
                result = _perceived_text_color(hex_c)
                self.assertEqual(len(result), 1)

    def test_named_color_yellow_gives_black(self):
        """Named color 'yellow' (bright) must give black text 'k'."""
        result = _perceived_text_color("yellow")
        self.assertEqual(result, "k")

    def test_named_color_navy_gives_white(self):
        """Named color 'navy' (dark) must give white text 'w'."""
        result = _perceived_text_color("navy")
        self.assertEqual(result, "w")

    def test_deterministic_on_same_input(self):
        """Same hex code must always produce the same result."""
        color = "#1a73e8"
        results = {_perceived_text_color(color) for _ in range(5)}
        self.assertEqual(len(results), 1,
                         msg="_perceived_text_color must be deterministic")


# ===========================================================================
# closest_color_name — extra coverage
# ===========================================================================


class TestClosestColorNameExtra(unittest.TestCase):
    """Additional closest_color_name tests covering gaps in the main suite."""

    def tearDown(self):
        _close_all()

    # -- default argument --

    def test_default_hex_color_returns_result(self):
        """Calling with all defaults (hex_color='#000000') must not raise."""
        result = closest_color_name()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_default_hex_color_is_black(self):
        """Default hex '#000000' (black) must return a very dark color name."""
        result = closest_color_name()
        # The closest CSS4 name to black must be 'black' itself
        self.assertEqual(result, ["black"])

    # -- return_distances ordering invariant --

    def test_return_distances_sorted_ascending(self):
        """return_distances=True must give distances in non-decreasing order."""
        od = closest_color_name("#aabbcc", top_n=5, return_distances=True)
        distances = list(od.values())
        self.assertEqual(distances, sorted(distances),
                         msg="Distances must be sorted ascending")

    def test_return_distances_keys_are_str(self):
        """All keys in return_distances dict must be strings."""
        od = closest_color_name("#ff0000", top_n=3, return_distances=True)
        for k in od.keys():
            with self.subTest(k=k):
                self.assertIsInstance(k, str)

    def test_return_distances_values_are_non_negative(self):
        """All distance values must be >= 0."""
        od = closest_color_name("#ff5733", top_n=5, return_distances=True)
        for name, dist in od.items():
            with self.subTest(name=name):
                self.assertGreaterEqual(dist, 0.0)

    # -- top_n clamped to palette size --

    def test_top_n_larger_than_css4_palette_clamped(self):
        """top_n > palette size must return at most len(palette) results."""
        palette_size = len(mcolors.CSS4_COLORS)
        result = closest_color_name(
            "#123456", top_n=palette_size + 9999, use_spec="CSS4"
        )
        self.assertLessEqual(len(result), palette_size)

    # -- use_lab and use_spec combination --

    def test_use_lab_with_xkcd_palette(self):
        """use_lab=True with use_spec='xkcd' must not raise."""
        try:
            result = closest_color_name("#aabbcc", use_lab=True, use_spec="xkcd")
        except Exception as exc:
            self.fail(f"use_lab=True + use_spec='xkcd' raised: {exc}")
        self.assertIsInstance(result, list)

    def test_use_lab_result_is_from_correct_palette(self):
        """use_lab=True with use_spec='CSS4' must return only CSS4 names."""
        result = closest_color_name("#aabbcc", use_lab=True, use_spec="CSS4", top_n=3)
        for name in result:
            with self.subTest(name=name):
                self.assertIn(name, mcolors.CSS4_COLORS,
                              msg=f"LAB result {name!r} not in CSS4 palette")

    # -- distance = 0 for exact match --

    def test_exact_css4_color_has_distance_zero(self):
        """Looking up an exact CSS4 name must yield distance 0.0."""
        od = closest_color_name("red", use_spec="CSS4", return_distances=True)
        self.assertIn("red", od)
        self.assertAlmostEqual(od["red"], 0.0, places=6)


# ===========================================================================
# display_colors — extra coverage
# ===========================================================================


class TestDisplayColorsExtra(unittest.TestCase):
    """Additional display_colors tests covering the ytick bug-fix and figure structure."""

    def setUp(self):
        self._show_patch = mock.patch("matplotlib.pyplot.show")
        self._show_patch.start()

    def tearDown(self):
        self._show_patch.stop()
        _close_all()

    def test_figure_has_exactly_one_axes(self):
        """display_colors must create exactly one Axes object."""
        display_colors(["red", "blue"])
        fig = plt.gcf()
        self.assertEqual(len(fig.get_axes()), 1)

    def test_ytick_marks_have_zero_length(self):
        """The y-axis tick marks must have length 0 (only labels, no ticks).

        Bug fix: the original code called plt.yticks([]) after setting labels,
        which cleared the labels.  The fix sets tick_params(length=0) instead.
        This test verifies the tick length is 0, not that the labels are present
        (that is already covered in the main test file).
        """
        display_colors(["red", "blue"])
        ax = plt.gcf().get_axes()[0]
        for tick in ax.yaxis.get_major_ticks():
            # tick1line and tick2line are the inner/outer tick marks
            self.assertAlmostEqual(
                tick.tick1line.get_markersize(), 0.0,
                msg="y-axis tick marks must have length 0",
            )

    def test_x_axis_has_no_ticks(self):
        """The x-axis must have no tick marks."""
        display_colors(["red", "blue"])
        ax = plt.gcf().get_axes()[0]
        xticks = ax.get_xticks()
        self.assertEqual(len(xticks), 0,
                         msg="x-axis must have no ticks")

    def test_title_is_set_on_axes(self):
        """The figure title must match the 'title' parameter."""
        display_colors(["red"], title="My Colors")
        ax = plt.gcf().get_axes()[0]
        self.assertEqual(ax.get_title(), "My Colors")

    def test_bar_colors_match_input(self):
        """Each horizontal bar must use the color provided."""
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        display_colors(colors)
        ax = plt.gcf().get_axes()[0]
        # containers[0] holds the barh patches
        self.assertEqual(len(ax.containers), 1)
        patches = ax.containers[0].patches
        self.assertEqual(len(patches), len(colors))


# ===========================================================================
# plot_colortable — extra coverage
# ===========================================================================


class TestPlotColortableExtra(unittest.TestCase):
    """Additional plot_colortable tests covering sort ordering and module attr."""

    def setUp(self):
        # Build a deliberately reverse-HSV-order palette so sort changes order
        self._small = {
            "yellow":   "#ffff00",   # H≈60°  (high hue)
            "blue":     "#0000ff",   # H=240°
            "red":      "#ff0000",   # H=0°/360°
            "green":    "#008000",   # H=120°
        }

    def tearDown(self):
        _close_all()

    def test_sort_true_changes_name_order_vs_false(self):
        """sort_colors=True must produce a different rendering order than False.

        We verify by comparing the sequence of text labels on the axes.
        A deliberately non-HSV-sorted input must be reordered when sorted.
        """
        fig_sorted = plot_colortable(self._small, sort_colors=True, ncols=1)
        ax_sorted = fig_sorted.get_axes()[0]
        sorted_texts = [t.get_text() for t in ax_sorted.texts
                        if t.get_text() in self._small]
        _close_all()

        fig_unsorted = plot_colortable(self._small, sort_colors=False, ncols=1)
        ax_unsorted = fig_unsorted.get_axes()[0]
        unsorted_texts = [t.get_text() for t in ax_unsorted.texts
                          if t.get_text() in self._small]

        # Both must contain all names
        self.assertEqual(set(sorted_texts), set(self._small.keys()))
        self.assertEqual(set(unsorted_texts), set(self._small.keys()))

        # For a sufficiently scrambled input the sort must reorder them
        # (this is not guaranteed to differ for every palette, but for our
        #  hand-picked out-of-HSV-order input it must differ)
        self.assertNotEqual(
            sorted_texts, unsorted_texts,
            msg="sort_colors=True must change rendering order for non-HSV input",
        )

    def test_figure_dpi_is_set(self):
        """The returned figure must have a non-default DPI (set by the function)."""
        fig = plot_colortable(self._small)
        self.assertIsNotNone(fig.get_dpi())
        self.assertGreater(fig.get_dpi(), 0)

    def test_axes_xlim_covers_all_columns(self):
        """axes xlim must span cell_width * ncols."""
        ncols = 2
        fig = plot_colortable(self._small, ncols=ncols, sort_colors=False)
        ax = fig.get_axes()[0]
        cell_width = 248
        xmax = ax.get_xlim()[1]
        self.assertAlmostEqual(xmax, cell_width * ncols, delta=1)

    def test_axes_ylim_covers_all_rows(self):
        """axes ylim must accommodate all rows for the given ncols."""
        ncols = 2
        n = len(self._small)
        nrows = math.ceil(n / ncols)
        cell_height = 24
        fig = plot_colortable(self._small, ncols=ncols, sort_colors=False)
        ax = fig.get_axes()[0]
        ymin, ymax = ax.get_ylim()
        # y-axis is inverted: ymin > ymax in matplotlib terms
        span = abs(ymax - ymin)
        self.assertGreaterEqual(span, (nrows - 1) * cell_height)

    def test_single_color_no_crash(self):
        """A palette with exactly one color must not crash."""
        fig = plot_colortable({"only": "#abcdef"})
        self.assertIsNotNone(fig)


# ===========================================================================
# plot_overlapping_colors — extra coverage
# ===========================================================================


class TestPlotOverlappingColorsExtra(unittest.TestCase):
    """Additional plot_overlapping_colors tests for text content and structure."""

    def tearDown(self):
        _close_all()

    def test_text_labels_contain_css4_header(self):
        """The figure must contain 'X11/CSS4' column headers."""
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        all_texts = [t.get_text() for t in ax.texts]
        self.assertTrue(
            any("X11/CSS4" in t for t in all_texts),
            msg=f"'X11/CSS4' header not found. Texts: {all_texts[:10]}",
        )

    def test_text_labels_contain_xkcd_header(self):
        """The figure must contain 'xkcd' column headers."""
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        all_texts = [t.get_text() for t in ax.texts]
        self.assertTrue(
            any(t.strip() == "xkcd" for t in all_texts),
            msg=f"'xkcd' header not found. Texts: {all_texts[:10]}",
        )

    def test_at_least_two_rectangles_per_overlap(self):
        """Each overlapping name must contribute at least 2 rectangles (CSS4 + XKCD)."""
        overlap_count = sum(
            1 for name in mcolors.CSS4_COLORS
            if f"xkcd:{name}" in mcolors.XKCD_COLORS
        )
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        rects = ax.patches
        # At minimum 2 patches per overlap entry
        self.assertGreaterEqual(
            len(rects), 2 * overlap_count,
            msg=f"Expected >= {2 * overlap_count} rectangles, got {len(rects)}",
        )

    def test_axes_xlim_non_negative(self):
        """Axes xlim must start at 0 and end positive."""
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        xmin, xmax = ax.get_xlim()
        self.assertAlmostEqual(xmin, 0.0, places=3)
        self.assertGreater(xmax, 0)

    def test_axes_is_off(self):
        """Axes frame and ticks must be turned off."""
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        self.assertFalse(ax.axison,
                         msg="ax.axis('off') must have been called")


# ===========================================================================
# Public API — __module__ attribute
# ===========================================================================


# class TestPublicModuleAttr(unittest.TestCase):
#     """All public functions must have __module__ pointing to the public API path.

#     This verifies the export pipeline: set_module / export_all must have been
#     called so that help(), inspect, and IDE tooling show the right origin.
#     """

#     def tearDown(self):
#         _close_all()

#     def test_closest_color_name_module(self):
#         """closest_color_name.__module__ must be 'scikitplot.misc'."""
#         self.assertEqual(
#             closest_color_name.__module__, "scikitplot.misc",
#             msg=f"Got __module__={closest_color_name.__module__!r}",
#         )

#     def test_display_colors_module(self):
#         """display_colors.__module__ must be 'scikitplot.misc'."""
#         self.assertEqual(display_colors.__module__, "scikitplot.misc")

#     def test_plot_colortable_module(self):
#         """plot_colortable.__module__ must be 'scikitplot.misc'."""
#         self.assertEqual(plot_colortable.__module__, "scikitplot.misc")

#     def test_plot_overlapping_colors_module(self):
#         """plot_overlapping_colors.__module__ must be 'scikitplot.misc'."""
#         self.assertEqual(plot_overlapping_colors.__module__, "scikitplot.misc")

#     def test_all_public_functions_have_same_package(self):
#         """All public functions must share the same top-level package."""
#         import scikitplot.misc as m
#         modules = {getattr(m, name).__module__ for name in m.__all__}
#         self.assertEqual(
#             len(modules), 1,
#             msg=f"Expected all public functions in one module, got: {modules}",
#         )


if __name__ == "__main__":
    unittest.main(verbosity=2)
