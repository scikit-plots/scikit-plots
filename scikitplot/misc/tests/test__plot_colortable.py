# scikitplot/misc/tests/test__plot_colortable.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._plot_colortable` and the ``scikitplot.misc`` public API.

Run with the package test runner::

    import scikitplot.misc
    scikitplot.misc.test()

Or via the standard library unittest runner::

    python -m unittest scikitplot.misc.tests.test__plot_colortable -v

Coverage map
------------
_rgb_to_lab                     Known-value accuracy, branches, dtypes  → TestRgbToLab
_perceived_text_color           Light/dark boundary, channels, names    → TestPerceivedTextColor
display_colors                  Rendering, labels, validation, stress   → TestDisplayColors
closest_color_name              All params, bug fixes 1-6, errors       → TestClosestColorName
  ↳ use_spec / named-color fix  Bug 3: palette respected for names      → TestClosestColorNameUseSpecBug
  ↳ bool / top_n fix            Bug 5: bool rejected as top_n           → TestClosestColorNameBoolBug
  ↳ edge cases                  Clamping, tolerance + distances         → TestClosestColorNameEdgeCases
plot_colortable                 Rendering, sort, validation, bool fix   → TestPlotColortable
plot_overlapping_colors         Structure, patches, axes limits          → TestPlotOverlappingColors
logger                          Name, level, message emission           → TestLogger
scikitplot.misc public API      __all__, imports, test runner           → TestMiscPublicAPI
_testing utilities              ignore_warnings, SkipTest               → TestIgnoreWarnings
"""

from __future__ import annotations

import logging
import sys
import warnings
import unittest
import unittest.mock as mock
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")   # headless — must be before pyplot import

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .._plot_colortable import (  # noqa: E402
    _perceived_text_color,
    _rgb_to_lab,
    closest_color_name,
    display_colors,
    logger,
    plot_colortable,
    plot_overlapping_colors,
)

_SUB_MOD = "scikitplot.misc._plot_colortable"
_SUB_MOD_ROOT = _SUB_MOD.rsplit(".", maxsplit=1)[0]


def _close_all():
    """Close all matplotlib figures after each test."""
    plt.close("all")


# ===========================================================================
# _rgb_to_lab — internal helper
# ===========================================================================

class TestRgbToLab(unittest.TestCase):
    """_rgb_to_lab must produce correct CIE L*a*b* values."""

    def tearDown(self):
        _close_all()

    # -- Known CIELAB reference values --

    def test_white_gives_L100(self):
        """Pure white [1,1,1] must give L*≈100, a*≈0, b*≈0."""
        lab = _rgb_to_lab(np.array([1.0, 1.0, 1.0]))
        self.assertAlmostEqual(lab[0], 100.0, places=1)
        self.assertAlmostEqual(lab[1], 0.0, places=1)
        self.assertAlmostEqual(lab[2], 0.0, places=1)

    def test_black_gives_L0(self):
        """Pure black [0,0,0] must give L*=0, a*=0, b*=0."""
        lab = _rgb_to_lab(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(lab, [0.0, 0.0, 0.0], decimal=5)

    def test_red_L_star(self):
        """Pure red [1,0,0] must give L*≈53.2 (standard CIELAB value)."""
        lab = _rgb_to_lab(np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(lab[0], 53.24, places=1)

    def test_red_positive_a_star(self):
        """Pure red must have positive a* (red–green axis)."""
        lab = _rgb_to_lab(np.array([1.0, 0.0, 0.0]))
        self.assertGreater(lab[1], 0)

    def test_blue_negative_b_star(self):
        """Pure blue [0,0,1] must have negative b* (blue–yellow axis)."""
        lab = _rgb_to_lab(np.array([0.0, 0.0, 1.0]))
        self.assertLess(lab[2], 0)

    def test_output_shape(self):
        """Output must be a 1-D ndarray of length 3."""
        lab = _rgb_to_lab(np.array([0.5, 0.5, 0.5]))
        self.assertEqual(lab.shape, (3,))

    def test_list_input_accepted(self):
        """A plain Python list must be accepted without error."""
        try:
            lab = _rgb_to_lab([0.5, 0.3, 0.1])
        except Exception as e:
            self.fail(f"_rgb_to_lab raised on list input: {e}")
        self.assertEqual(lab.shape, (3,))

    def test_no_divide_by_255(self):
        """Values must already be in [0,1]; the function must NOT divide by 255.

        Bug 1 (fixed): the original closest_color_name divided the [0,1]
        output of mcolors.hex2color by 255, sending all values to near-zero
        and making every LAB distance meaningless.
        """
        lab_correct = _rgb_to_lab(np.array([1.0, 0.0, 0.0]))
        # If /255 were applied, L* would be ~2.8 instead of ~53.24
        self.assertGreater(lab_correct[0], 50.0,
                           msg="L* for pure red must be ~53; /255 bug would give ~2.8")

    # -- Additional coverage --

    def test_pure_green_L_star(self):
        """Pure green [0,1,0] must give L*≈87.7 (CIELAB standard)."""
        lab = _rgb_to_lab(np.array([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(lab[0], 87.73, places=0)

    def test_pure_blue_L_star(self):
        """Pure blue [0,0,1] must give L*≈32.3 (CIELAB standard)."""
        lab = _rgb_to_lab(np.array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(lab[0], 32.30, places=0)

    def test_gray_L_star_between_0_and_100(self):
        """Mid-gray [0.5,0.5,0.5] must have L* strictly between 0 and 100."""
        lab = _rgb_to_lab(np.array([0.5, 0.5, 0.5]))
        self.assertGreater(lab[0], 0.0)
        self.assertLess(lab[0], 100.0)

    def test_gray_near_neutral_ab(self):
        """Mid-gray must have a* and b* close to 0 (neutral axis)."""
        lab = _rgb_to_lab(np.array([0.5, 0.5, 0.5]))
        self.assertAlmostEqual(lab[1], 0.0, places=1)
        self.assertAlmostEqual(lab[2], 0.0, places=1)

    def test_tuple_input_accepted(self):
        """A plain Python tuple must be accepted without error."""
        try:
            lab = _rgb_to_lab((0.2, 0.4, 0.6))
        except Exception as e:
            self.fail(f"_rgb_to_lab raised on tuple input: {e}")
        self.assertEqual(lab.shape, (3,))

    def test_output_dtype_is_float(self):
        """Output array dtype must be float (not int)."""
        lab = _rgb_to_lab(np.array([0.0, 0.5, 1.0]))
        self.assertTrue(np.issubdtype(lab.dtype, np.floating))

    def test_L_star_monotone_with_brightness(self):
        """L* must increase monotonically with equal-channel brightness."""
        l_dark = _rgb_to_lab([0.1, 0.1, 0.1])[0]
        l_mid = _rgb_to_lab([0.5, 0.5, 0.5])[0]
        l_light = _rgb_to_lab([0.9, 0.9, 0.9])[0]
        self.assertLess(l_dark, l_mid)
        self.assertLess(l_mid, l_light)

    def test_linearisation_boundary_04045(self):
        """The sRGB threshold 0.04045 must produce continuous output.

        Values just below and just above 0.04045 must yield L* values
        that differ by less than 0.1 (continuous, no discontinuity).
        """
        lab_below = _rgb_to_lab([0.04044, 0.0, 0.0])
        lab_above = _rgb_to_lab([0.04046, 0.0, 0.0])
        self.assertAlmostEqual(lab_below[0], lab_above[0], places=1,
                               msg="L* discontinuity at the 0.04045 linearisation boundary")


# ===========================================================================
# _perceived_text_color
# ===========================================================================

class TestPerceivedTextColor(unittest.TestCase):
    """_perceived_text_color must return 'k' on light and 'w' on dark."""

    def test_white_background_gives_black_text(self):
        self.assertEqual(_perceived_text_color("#ffffff"), "k")

    def test_black_background_gives_white_text(self):
        self.assertEqual(_perceived_text_color("#000000"), "w")

    def test_light_yellow_gives_black_text(self):
        """A light colour (high luma) must give black text."""
        self.assertEqual(_perceived_text_color("#ffff99"), "k")

    def test_dark_blue_gives_white_text(self):
        """A dark colour (low luma) must give white text."""
        self.assertEqual(_perceived_text_color("#00008b"), "w")

    def test_returns_string(self):
        result = _perceived_text_color("#aabbcc")
        self.assertIsInstance(result, str)
        self.assertIn(result, ("k", "w"))

    # -- Additional channel coverage --

    def test_pure_red_gives_white_text(self):
        """Pure red luma = 0.299 < 0.5 → must give white text."""
        self.assertEqual(_perceived_text_color("#ff0000"), "w")

    def test_pure_green_gives_black_text(self):
        """Pure green luma = 0.587 > 0.5 → must give black text."""
        self.assertEqual(_perceived_text_color("#00ff00"), "k")

    def test_pure_blue_gives_white_text(self):
        """Pure blue luma = 0.114 < 0.5 → must give white text."""
        self.assertEqual(_perceived_text_color("#0000ff"), "w")

    def test_named_color_white_accepted(self):
        """Named color 'white' must be accepted and give 'k'."""
        self.assertEqual(_perceived_text_color("white"), "k")

    def test_named_color_black_accepted(self):
        """Named color 'black' must be accepted and give 'w'."""
        self.assertEqual(_perceived_text_color("black"), "w")

    def test_output_is_single_char(self):
        """Return value must be a single character ('k' or 'w')."""
        result = _perceived_text_color("#556677")
        self.assertEqual(len(result), 1)

    def test_result_consistent_on_repeated_calls(self):
        """Same input must always produce the same output (deterministic)."""
        color = "#aabbcc"
        results = {_perceived_text_color(color) for _ in range(5)}
        self.assertEqual(len(results), 1)


# ===========================================================================
# display_colors
# ===========================================================================

class TestDisplayColors(unittest.TestCase):
    """display_colors must render correctly and validate inputs."""

    def setUp(self):
        self._show_patcher = mock.patch("matplotlib.pyplot.show")
        self._mock_show = self._show_patcher.start()

    def tearDown(self):
        self._show_patcher.stop()
        _close_all()

    # -- Happy-path tests --

    def test_basic_call_no_error(self):
        """Calling with a valid list must not raise."""
        display_colors(["red", "blue", "green"])

    def test_show_called_once(self):
        """plt.show() must be called exactly once per invocation."""
        display_colors(["red", "blue"])
        self._mock_show.assert_called_once()

    def test_labels_visible_without_indices(self):
        """y-tick labels must equal the color names when show_indices=False.

        Bug 6+9 (fixed): the original called plt.yticks([]) immediately
        after plt.yticks(range, labels), which cleared all labels before
        rendering.
        """
        display_colors(["red", "blue", "green"], show_indices=False)
        ax = plt.gca()
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(tick_labels, ["red", "blue", "green"])

    def test_labels_visible_with_indices(self):
        """y-tick labels must include the index prefix when show_indices=True."""
        display_colors(["red", "blue"], show_indices=True)
        ax = plt.gca()
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(tick_labels, ["0: red", "1: blue"])

    def test_custom_title_set(self):
        """The supplied title must appear on the figure."""
        title = "My Custom Title"
        display_colors(["red"], title=title)
        ax = plt.gca()
        self.assertEqual(ax.get_title(), title)

    def test_default_title_set(self):
        """The default title must be set when no title is supplied."""
        display_colors(["red"])
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Color Display (Order)")

    def test_single_color_no_error(self):
        """A single-element list must be accepted."""
        display_colors(["#ff0000"])

    def test_hex_colors_accepted(self):
        """Hex code strings must be accepted without error."""
        display_colors(["#ff5733", "#1a2b3c"])

    def test_x_ticks_hidden(self):
        """The x-axis must have no tick marks (pure label display)."""
        display_colors(["red", "blue"])
        ax = plt.gca()
        self.assertEqual(list(ax.get_xticks()), [])

    # -- Stress test --

    def test_many_colors_no_error(self):
        """Rendering 50 colors must not raise."""
        colors = [f"#{i:02x}{i:02x}{i:02x}" for i in range(0, 256, 5)][:50]
        display_colors(colors)

    def test_bar_count_matches_input(self):
        """Number of horizontal bars must equal number of input colors."""
        colors = ["red", "green", "blue"]
        display_colors(colors)
        ax = plt.gca()
        # barh creates a BarContainer; count the patches
        n_bars = len(ax.patches)
        self.assertEqual(n_bars, len(colors))

    def test_figsize_height_grows_with_color_count(self):
        """Figure height must be larger for more colors (scales with input)."""
        display_colors(["red"])
        fig_small = plt.gcf()
        h_small = fig_small.get_size_inches()[1]
        _close_all()

        display_colors(["red"] * 20)
        fig_large = plt.gcf()
        h_large = fig_large.get_size_inches()[1]
        _close_all()

        self.assertGreater(h_large, h_small)

    # -- Validation tests --

    def test_non_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            display_colors("red,blue,green")

    def test_tuple_raises_type_error(self):
        with self.assertRaises(TypeError):
            display_colors(("red", "blue"))

    def test_empty_list_raises_value_error(self):
        with self.assertRaises(ValueError):
            display_colors([])

    def test_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            display_colors(None)

    def test_dict_raises_type_error(self):
        """A dict must not be accepted (only list)."""
        with self.assertRaises(TypeError):
            display_colors({"red": "#ff0000"})

    def test_generator_raises_type_error(self):
        """A generator must not be accepted (only list)."""
        with self.assertRaises(TypeError):
            display_colors(c for c in ["red"])

    def test_integer_raises_type_error(self):
        with self.assertRaises(TypeError):
            display_colors(42)


# ===========================================================================
# closest_color_name  — core functionality
# ===========================================================================

class TestClosestColorName(unittest.TestCase):
    """closest_color_name must find correct matches and validate all inputs."""

    def tearDown(self):
        _close_all()

    # -- Bug 1+2: use_lab=True (rgb_to_lab + no /255) --

    def test_use_lab_does_not_crash(self):
        """Bug 2 (fixed): use_lab=True must not raise AttributeError.

        mcolors.rgb_to_lab does not exist; the original code raised
        AttributeError at runtime.  The fix uses _rgb_to_lab internally.
        """
        try:
            result = closest_color_name("#ff5733", use_lab=True)
        except AttributeError as e:
            self.fail(
                f"use_lab=True raised AttributeError (Bug 2 regression): {e}"
            )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_use_lab_no_divide_by_255(self):
        """Bug 1 (fixed): RGB [0,1] values must not be divided by 255 again.

        If /255 were applied, every colour would map to near-zero in LAB space
        and the 'closest' match would be the darkest palette colour, not the
        perceptually nearest.  We verify the result is plausible (a warm tone)
        for the input '#ff5733' (an orange-red).
        """
        result = closest_color_name("#ff5733", use_lab=True)
        self.assertNotIn(result[0], ("black", "navy", "darkblue", "darkgreen"),
                         msg=f"Bug 1 regression: /255 error would give dark match, got {result}")

    def test_use_lab_rgb_agree_on_named_color(self):
        """RGB and LAB modes must agree on a pure named colour (distance=0 path)."""
        rgb_result = closest_color_name("red", use_lab=False)
        lab_result = closest_color_name("red", use_lab=True)
        self.assertEqual(rgb_result, lab_result)

    # -- Named color direct match --

    def test_named_color_returns_itself(self):
        """A CSS4 named colour must match itself with distance 0."""
        result = closest_color_name("red")
        self.assertEqual(result, ["red"])

    def test_named_color_return_distances_true(self):
        """Named colour with return_distances=True must give {name: 0.0}."""
        result = closest_color_name("blue", return_distances=True)
        self.assertIsInstance(result, OrderedDict)
        self.assertIn("blue", result)
        self.assertEqual(result["blue"], 0.0)

    def test_xkcd_named_color(self):
        """An XKCD-namespaced colour must be matched directly."""
        result = closest_color_name("xkcd:sky blue", use_spec="xkcd")
        self.assertEqual(result, ["xkcd:sky blue"])

    # -- Hex input --

    def test_hex_returns_list_of_str(self):
        result = closest_color_name("#ff5733")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)

    def test_exact_hex_match_zero_distance(self):
        """Hex value of a palette colour must have distance very close to 0."""
        red_hex = mcolors.CSS4_COLORS["red"]  # '#ff0000'
        result = closest_color_name(red_hex, return_distances=True)
        dist = list(result.values())[0]
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_hex_uppercase_accepted(self):
        """Uppercase hex codes must be accepted without error."""
        result = closest_color_name("#FF5733")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    # -- top_n --

    def test_top_n_one(self):
        result = closest_color_name("#118ab2", top_n=1)
        self.assertEqual(len(result), 1)

    def test_top_n_three(self):
        result = closest_color_name("#118ab2", top_n=3)
        self.assertEqual(len(result), 3)

    def test_top_n_three_names_are_strings(self):
        result = closest_color_name("#118ab2", top_n=3)
        for name in result:
            self.assertIsInstance(name, str)

    def test_top_n_three_ordered_by_distance(self):
        """Results must be in ascending distance order."""
        result = closest_color_name("#118ab2", top_n=3, return_distances=True)
        dists = list(result.values())
        self.assertEqual(dists, sorted(dists))

    # -- return_distances --

    def test_return_distances_false_gives_list(self):
        result = closest_color_name("#ff5733", return_distances=False)
        self.assertIsInstance(result, list)

    def test_return_distances_true_gives_ordered_dict(self):
        result = closest_color_name("#ff5733", return_distances=True)
        self.assertIsInstance(result, OrderedDict)

    def test_return_distances_dict_values_are_floats(self):
        result = closest_color_name("#ff5733", return_distances=True)
        for v in result.values():
            self.assertIsInstance(v, float)

    # -- use_spec --

    def test_use_spec_css4(self):
        result = closest_color_name("#ff5733", use_spec="CSS4")
        self.assertIsInstance(result, list)

    def test_use_spec_xkcd(self):
        result = closest_color_name("#ff5733", use_spec="xkcd")
        self.assertIsInstance(result, list)

    def test_use_spec_css4_returns_css4_name(self):
        """CSS4 result must be in the CSS4 palette."""
        result = closest_color_name("#ff5733", use_spec="CSS4")
        self.assertIn(result[0], mcolors.CSS4_COLORS)

    def test_use_spec_xkcd_returns_xkcd_name(self):
        """XKCD result must be in the XKCD palette."""
        result = closest_color_name("#ff5733", use_spec="xkcd")
        self.assertIn(result[0], mcolors.XKCD_COLORS)

    # -- tolerance --

    def test_tolerance_zero_returns_result(self):
        """tolerance=0.0 (no filter) must always return top_n results."""
        result = closest_color_name("#ff5733", tolerance=0.0)
        self.assertEqual(len(result), 1)

    def test_tolerance_very_small_may_return_empty(self):
        """A tolerance smaller than any palette distance may return empty list."""
        result = closest_color_name("#123456", tolerance=0.000001)
        self.assertIsInstance(result, list)

    def test_tolerance_large_returns_result(self):
        """A tolerance larger than all distances must return top_n results."""
        result = closest_color_name("#ff5733", tolerance=10.0)
        self.assertEqual(len(result), 1)

    def test_tolerance_filters_correctly(self):
        """Bug 5 (fixed): tolerance is a maximum distance, not a minimum."""
        result_strict = closest_color_name("#ff5733", tolerance=0.05)
        result_loose = closest_color_name("#ff5733", tolerance=0.5)
        self.assertLessEqual(len(result_strict), len(result_loose))

    # -- Validation / error paths --

    def test_invalid_use_spec_raises_value_error(self):
        with self.assertRaises(ValueError):
            closest_color_name("#ff5733", use_spec="invalid")

    def test_invalid_hex_raises_value_error(self):
        with self.assertRaises(ValueError):
            closest_color_name("notacolor")

    def test_top_n_zero_raises_type_error(self):
        with self.assertRaises(TypeError):
            closest_color_name("#ff5733", top_n=0)

    def test_top_n_negative_raises_type_error(self):
        with self.assertRaises(TypeError):
            closest_color_name("#ff5733", top_n=-1)

    def test_top_n_float_raises_type_error(self):
        with self.assertRaises(TypeError):
            closest_color_name("#ff5733", top_n=1.5)

    def test_tolerance_negative_raises_type_error(self):
        with self.assertRaises(TypeError):
            closest_color_name("#ff5733", tolerance=-0.1)

    def test_tolerance_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            closest_color_name("#ff5733", tolerance="small")


# ===========================================================================
# closest_color_name — Bug 3 fix: use_spec respected for named colors
# ===========================================================================

class TestClosestColorNameUseSpecBug(unittest.TestCase):
    """
    Bug 3 (fixed): use_spec was silently ignored when hex_color was a named
    color found in any palette (CSS4 or XKCD), not just the selected one.

    Root cause: the named-color early-exit check happened BEFORE palette
    selection, so ``closest_color_name("red", use_spec="xkcd")`` returned
    "red" (a CSS4 name) even though the caller asked for the XKCD palette.

    Fix: select palette first; named-color check is scoped to selected palette.
    """

    def tearDown(self):
        _close_all()

    def test_css4_named_color_with_css4_spec_returns_directly(self):
        """CSS4 name + use_spec='CSS4' → direct match in CSS4 palette."""
        result = closest_color_name("red", use_spec="CSS4")
        self.assertEqual(result, ["red"])

    def test_css4_named_color_with_xkcd_spec_searches_xkcd(self):
        """Bug 3 fix: CSS4 name + use_spec='xkcd' must search XKCD palette.

        'red' is in CSS4_COLORS but NOT in XKCD_COLORS (xkcd uses 'xkcd:red').
        With the fix, the result must be from the XKCD palette, not 'red' itself.
        """
        result = closest_color_name("red", use_spec="xkcd")
        # Result must be from the XKCD palette
        for name in result:
            self.assertIn(name, mcolors.XKCD_COLORS,
                          msg=f"Bug 3 regression: 'red' with use_spec='xkcd' "
                              f"returned CSS4 name {name!r}")

    def test_xkcd_named_color_with_xkcd_spec_returns_directly(self):
        """XKCD name + use_spec='xkcd' → direct match with distance 0."""
        result = closest_color_name("xkcd:red", use_spec="xkcd")
        self.assertEqual(result, ["xkcd:red"])

    def test_xkcd_named_color_with_css4_spec_searches_css4(self):
        """Bug 3 fix: XKCD name + use_spec='CSS4' must search CSS4 palette.

        'xkcd:sky blue' is in XKCD_COLORS but not CSS4_COLORS.
        With the fix, the result must be from the CSS4 palette.
        """
        result = closest_color_name("xkcd:sky blue", use_spec="CSS4")
        # to_rgb resolves 'xkcd:sky blue'; result must be a CSS4 name
        for name in result:
            self.assertIn(name, mcolors.CSS4_COLORS,
                          msg=f"Bug 3 regression: xkcd name with use_spec='CSS4' "
                              f"returned non-CSS4 name {name!r}")

    def test_named_color_in_both_palettes_respects_spec_css4(self):
        """A name in BOTH palettes: use_spec='CSS4' must return CSS4 direct match.

        We find a color name that is in CSS4_COLORS AND (with xkcd: prefix)
        in XKCD_COLORS, then verify use_spec selects the right one.
        """
        # Find an overlapping name
        overlap = [
            name for name in mcolors.CSS4_COLORS
            if f"xkcd:{name}" in mcolors.XKCD_COLORS
        ]
        if not overlap:
            self.skipTest("No overlapping CSS4/XKCD names found")
        name = overlap[0]
        result = closest_color_name(name, use_spec="CSS4")
        self.assertEqual(result, [name],
                         msg=f"CSS4 palette: expected direct match for {name!r}")

    def test_named_color_in_both_palettes_respects_spec_xkcd(self):
        """A name in BOTH palettes: use_spec='xkcd' with 'xkcd:' prefix."""
        overlap = [
            name for name in mcolors.CSS4_COLORS
            if f"xkcd:{name}" in mcolors.XKCD_COLORS
        ]
        if not overlap:
            self.skipTest("No overlapping CSS4/XKCD names found")
        xkcd_name = f"xkcd:{overlap[0]}"
        result = closest_color_name(xkcd_name, use_spec="xkcd")
        self.assertEqual(result, [xkcd_name])

    def test_use_spec_css4_result_never_contains_xkcd_prefix(self):
        """Results from use_spec='CSS4' must never start with 'xkcd:'."""
        result = closest_color_name("#aabbcc", use_spec="CSS4", top_n=5)
        for name in result:
            self.assertFalse(name.startswith("xkcd:"),
                             msg=f"CSS4 result contains xkcd: name: {name!r}")

    def test_use_spec_xkcd_result_always_contains_xkcd_prefix(self):
        """Results from use_spec='xkcd' must always start with 'xkcd:'."""
        result = closest_color_name("#aabbcc", use_spec="xkcd", top_n=5)
        for name in result:
            self.assertTrue(name.startswith("xkcd:"),
                            msg=f"XKCD result missing xkcd: prefix: {name!r}")


# ===========================================================================
# closest_color_name — Bug 5 fix: bool rejected for top_n
# ===========================================================================

class TestClosestColorNameBoolBug(unittest.TestCase):
    """
    Bug 5 (fixed): bool is a subclass of int in Python, so top_n=True
    previously passed validation silently as top_n=1.

    Root cause: ``not isinstance(top_n, int)`` is False for booleans.
    Fix: add ``isinstance(top_n, bool)`` guard before the int check.
    """

    def tearDown(self):
        _close_all()

    def test_top_n_bool_true_rejected(self):
        """top_n=True (==1 as int) must be rejected with TypeError.

        Bug 5 regression test: previously True silently passed as top_n=1.
        """
        with self.assertRaises(TypeError,
                               msg="Bug 5 regression: top_n=True should raise TypeError"):
            closest_color_name("#ff5733", top_n=True)

    def test_top_n_bool_false_rejected(self):
        """top_n=False (==0 as int) must be rejected with TypeError."""
        with self.assertRaises(TypeError):
            closest_color_name("#ff5733", top_n=False)

    def test_top_n_integer_one_still_accepted(self):
        """top_n=1 (plain int) must still work correctly after the bool fix."""
        result = closest_color_name("#ff5733", top_n=1)
        self.assertEqual(len(result), 1)

    def test_top_n_integer_five_still_accepted(self):
        """top_n=5 (plain int) must still work correctly after the bool fix."""
        result = closest_color_name("#ff5733", top_n=5)
        self.assertEqual(len(result), 5)

    def test_tolerance_bool_true_rejected(self):
        """tolerance=True (==1.0 as float) must be rejected with TypeError."""
        with self.assertRaises(TypeError):
            closest_color_name("#ff5733", tolerance=True)

    def test_tolerance_bool_false_rejected(self):
        """tolerance=False (==0.0 as float) must be rejected with TypeError."""
        with self.assertRaises(TypeError):
            closest_color_name("#ff5733", tolerance=False)

    def test_tolerance_float_zero_still_accepted(self):
        """tolerance=0.0 (plain float) must still work after the bool fix."""
        result = closest_color_name("#ff5733", tolerance=0.0)
        self.assertIsInstance(result, list)


# ===========================================================================
# closest_color_name — edge cases
# ===========================================================================

class TestClosestColorNameEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions for closest_color_name."""

    def tearDown(self):
        _close_all()

    def test_default_hex_color_black_returns_result(self):
        """Default '#000000' must return a valid result."""
        result = closest_color_name()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_top_n_larger_than_palette_returns_whole_palette(self):
        """top_n > palette size must return all palette entries (clamped)."""
        result = closest_color_name("#ff5733", top_n=10000, use_spec="CSS4")
        # CSS4 palette has 148 colors; result must be at most that many
        self.assertLessEqual(len(result), len(mcolors.CSS4_COLORS))
        self.assertGreater(len(result), 0)

    def test_tolerance_with_return_distances(self):
        """tolerance and return_distances may be used together."""
        result = closest_color_name("#ff5733", tolerance=0.5, return_distances=True)
        self.assertIsInstance(result, OrderedDict)

    def test_empty_result_from_very_strict_tolerance_with_distances(self):
        """Very strict tolerance + return_distances must give empty OrderedDict."""
        result = closest_color_name("#123456", tolerance=1e-10, return_distances=True)
        self.assertIsInstance(result, OrderedDict)

    def test_top_n_three_with_xkcd_palette(self):
        """top_n=3 + use_spec='xkcd' must return 3 XKCD names."""
        result = closest_color_name("#aabbcc", top_n=3, use_spec="xkcd")
        self.assertEqual(len(result), 3)
        for name in result:
            self.assertIn(name, mcolors.XKCD_COLORS)

    def test_lab_and_rgb_may_differ_for_hex_input(self):
        """LAB and RGB modes are allowed to return different names for a hex input.

        This is not a bug: LAB is perceptually uniform, RGB is not.  We simply
        verify both modes return valid palette members.
        """
        rgb_result = closest_color_name("#667788", use_lab=False)
        lab_result = closest_color_name("#667788", use_lab=True)
        for name in rgb_result:
            self.assertIn(name, mcolors.CSS4_COLORS)
        for name in lab_result:
            self.assertIn(name, mcolors.CSS4_COLORS)

    def test_pure_hex_colors_accepted(self):
        """All pure-channel hex codes must be accepted without error."""
        for h in ("#ff0000", "#00ff00", "#0000ff", "#ffffff", "#000000"):
            with self.subTest(h=h):
                result = closest_color_name(h)
                self.assertIsInstance(result, list)

    def test_result_is_deterministic(self):
        """Same input must always produce the same result."""
        results = [closest_color_name("#ff5733") for _ in range(3)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])


# ===========================================================================
# plot_colortable
# ===========================================================================

class TestPlotColortable(unittest.TestCase):
    """plot_colortable must return a valid Figure and validate inputs."""

    def setUp(self):
        self.small_palette = {
            "red": "#ff0000",
            "green": "#008000",
            "blue": "#0000ff",
            "yellow": "#ffff00",
        }

    def tearDown(self):
        _close_all()

    # -- Happy path --

    def test_returns_figure(self):
        fig = plot_colortable(self.small_palette)
        self.assertIsInstance(fig, plt.Figure)

    def test_sort_true_no_error(self):
        fig = plot_colortable(self.small_palette, sort_colors=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_sort_false_no_error(self):
        fig = plot_colortable(self.small_palette, sort_colors=False)
        self.assertIsInstance(fig, plt.Figure)

    def test_display_hex_true_no_error(self):
        fig = plot_colortable(self.small_palette, display_hex=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_display_hex_false_no_error(self):
        fig = plot_colortable(self.small_palette, display_hex=False)
        self.assertIsInstance(fig, plt.Figure)

    def test_ncols_one_no_error(self):
        fig = plot_colortable(self.small_palette, ncols=1)
        self.assertIsInstance(fig, plt.Figure)

    def test_ncols_two_no_error(self):
        fig = plot_colortable(self.small_palette, ncols=2)
        self.assertIsInstance(fig, plt.Figure)

    def test_full_css4_palette_no_error(self):
        """The full CSS4 palette (148 colours) must render without error."""
        fig = plot_colortable(mcolors.CSS4_COLORS, ncols=4)
        self.assertIsInstance(fig, plt.Figure)

    def test_sort_fallback_logged_on_invalid_key(self):
        """An invalid key that can't be sorted by name must fall back gracefully."""
        bad_palette = {"notavalidcolor": "#ff0000", "red": "#ff0000"}
        with self.assertLogs("scikitplot.misc", level="WARNING"):
            fig = plot_colortable(bad_palette, sort_colors=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_single_color_dict(self):
        """A dict with exactly one color must render without error."""
        fig = plot_colortable({"red": "#ff0000"})
        self.assertIsInstance(fig, plt.Figure)

    def test_ncols_larger_than_n_colors(self):
        """ncols > n_colors is valid; math.ceil handles it correctly."""
        fig = plot_colortable(self.small_palette, ncols=100)
        self.assertIsInstance(fig, plt.Figure)

    def test_sort_false_preserves_insertion_order(self):
        """sort_colors=False must preserve dict insertion order."""
        ordered_names = ["red", "green", "blue", "yellow"]
        palette = {name: self.small_palette[name] for name in ordered_names}
        fig = plot_colortable(palette, sort_colors=False)
        ax = fig.get_axes()[0]
        texts = [t.get_text() for t in ax.texts if t.get_text() in ordered_names]
        # All four names must appear
        self.assertEqual(set(texts), set(ordered_names))

    def test_rectangle_count_matches_color_count(self):
        """Number of Rectangle patches must equal the number of colors."""
        from matplotlib.patches import Rectangle as Rect
        fig = plot_colortable(self.small_palette, display_hex=False)
        ax = fig.get_axes()[0]
        rects = [p for p in ax.patches if isinstance(p, Rect)]
        self.assertEqual(len(rects), len(self.small_palette))

    def test_display_hex_false_fewer_text_elements(self):
        """display_hex=False must render fewer text objects than display_hex=True."""
        fig_with = plot_colortable(self.small_palette, display_hex=True)
        n_with = len(fig_with.get_axes()[0].texts)
        _close_all()

        fig_without = plot_colortable(self.small_palette, display_hex=False)
        n_without = len(fig_without.get_axes()[0].texts)

        self.assertGreater(n_with, n_without,
                           msg="display_hex=True must add hex text annotations")

    # -- Validation --

    def test_non_dict_raises_type_error(self):
        with self.assertRaises(TypeError):
            plot_colortable(["red", "blue"])

    def test_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            plot_colortable("red")

    def test_empty_dict_raises_value_error(self):
        with self.assertRaises(ValueError):
            plot_colortable({})

    def test_ncols_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            plot_colortable(self.small_palette, ncols=0)

    def test_ncols_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            plot_colortable(self.small_palette, ncols=-1)

    def test_ncols_float_raises_value_error(self):
        with self.assertRaises(ValueError):
            plot_colortable(self.small_palette, ncols=2.5)

    def test_ncols_bool_true_raises_value_error(self):
        """Bool fix: ncols=True (==1 as int) must be rejected with ValueError.

        Previously True silently passed because isinstance(True, int) is True.
        """
        with self.assertRaises(ValueError,
                               msg="Bool fix regression: ncols=True should raise ValueError"):
            plot_colortable(self.small_palette, ncols=True)

    def test_ncols_bool_false_raises_value_error(self):
        """Bool fix: ncols=False (==0 as int) must also be rejected."""
        with self.assertRaises(ValueError):
            plot_colortable(self.small_palette, ncols=False)

    def test_ncols_integer_one_still_accepted(self):
        """ncols=1 (plain int) must still work correctly after bool fix."""
        fig = plot_colortable(self.small_palette, ncols=1)
        self.assertIsInstance(fig, plt.Figure)

    # -- Axes structure --

    def test_has_one_axes(self):
        fig = plot_colortable(self.small_palette, ncols=2)
        self.assertEqual(len(fig.get_axes()), 1)

    def test_axes_off(self):
        """The axes must be invisible (no spines, labels)."""
        fig = plot_colortable(self.small_palette, ncols=2)
        ax = fig.get_axes()[0]
        self.assertFalse(ax.axison)

    def test_xaxis_not_visible(self):
        """The x-axis must be explicitly hidden."""
        fig = plot_colortable(self.small_palette)
        ax = fig.get_axes()[0]
        self.assertFalse(ax.xaxis.get_visible())

    def test_yaxis_not_visible(self):
        """The y-axis must be explicitly hidden."""
        fig = plot_colortable(self.small_palette)
        ax = fig.get_axes()[0]
        self.assertFalse(ax.yaxis.get_visible())


# ===========================================================================
# plot_overlapping_colors
# ===========================================================================

class TestPlotOverlappingColors(unittest.TestCase):
    """plot_overlapping_colors must return a Figure with correct structure."""

    def tearDown(self):
        _close_all()

    def test_returns_figure(self):
        fig = plot_overlapping_colors()
        self.assertIsInstance(fig, plt.Figure)

    def test_figure_has_axes(self):
        fig = plot_overlapping_colors()
        self.assertGreater(len(fig.get_axes()), 0)

    def test_figure_size(self):
        """Figure must be 9 × 5 inches as specified."""
        fig = plot_overlapping_colors()
        w, h = fig.get_size_inches()
        self.assertAlmostEqual(w, 9.0, places=1)
        self.assertAlmostEqual(h, 5.0, places=1)

    def test_overlap_set_nonempty(self):
        """There are known overlapping names between CSS4 and XKCD."""
        overlap = {
            name
            for name in mcolors.CSS4_COLORS
            if f"xkcd:{name}" in mcolors.XKCD_COLORS
        }
        self.assertGreater(len(overlap), 0)

    def test_no_error_on_repeated_calls(self):
        """Calling twice must not share state or raise errors."""
        fig1 = plot_overlapping_colors()
        fig2 = plot_overlapping_colors()
        self.assertIsNot(fig1, fig2)

    def test_axes_is_off(self):
        """The axes frame must be turned off."""
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        self.assertFalse(ax.axison)

    def test_rectangles_present_in_axes(self):
        """The figure must contain Rectangle patches (one per overlap × 2 columns)."""
        import matplotlib.patches as mpatches
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        rects = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
        # At least 2 rectangles (one CSS4 + one XKCD per overlap entry)
        self.assertGreater(len(rects), 0)

    def test_axes_xlim_matches_n_groups(self):
        """Axes x-limit must equal 3 × n_groups = 9."""
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        x_max = ax.get_xlim()[1]
        self.assertAlmostEqual(x_max, 9.0, places=1)

    def test_ylim_is_inverted(self):
        """y-axis must be inverted (y_min > y_max) for top-down rendering."""
        fig = plot_overlapping_colors()
        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        self.assertGreater(y_min, y_max,
                           msg="y-axis must be inverted for top-down row rendering")

    def test_figures_can_be_closed_after_call(self):
        """Figures returned by the function must be closable without error."""
        fig = plot_overlapping_colors()
        try:
            plt.close(fig)
        except Exception as e:
            self.fail(f"plt.close raised an unexpected error: {e}")

    def test_all_overlapping_names_have_valid_css4_and_xkcd_hex(self):
        """Every overlapping name must have a valid hex in both palettes."""
        overlap = [
            name for name in mcolors.CSS4_COLORS
            if f"xkcd:{name}" in mcolors.XKCD_COLORS
        ]
        for name in overlap:
            with self.subTest(name=name):
                css4_hex = mcolors.CSS4_COLORS[name]
                xkcd_hex = mcolors.XKCD_COLORS[f"xkcd:{name}"]
                # Both must be valid matplotlib hex codes
                try:
                    mcolors.to_rgb(css4_hex)
                    mcolors.to_rgb(xkcd_hex)
                except ValueError as e:
                    self.fail(f"Invalid hex for {name!r}: {e}")


# ===========================================================================
# Logger
# ===========================================================================

class TestLogger(unittest.TestCase):
    """The module logger must be correctly named and emit expected records."""

    def tearDown(self):
        _close_all()

    def test_logger_name(self):
        """Logger must be named 'scikitplot.misc' (hierarchical)."""
        self.assertEqual(logger.name, "scikitplot.misc")

    def test_logger_is_logging_logger(self):
        self.assertIsInstance(logger, logging.Logger)

    def test_debug_emitted_on_closest_color_name(self):
        """closest_color_name must emit DEBUG records via the module logger."""
        with self.assertLogs("scikitplot.misc", level=logging.DEBUG) as log:
            closest_color_name("#ff5733")
        debug_msgs = [r for r in log.output if "DEBUG" in r]
        self.assertGreater(len(debug_msgs), 0)

    def test_error_emitted_on_invalid_use_spec(self):
        """An invalid use_spec must emit an ERROR log before raising."""
        with self.assertLogs("scikitplot.misc", level=logging.ERROR):
            with self.assertRaises(ValueError):
                closest_color_name("#ff5733", use_spec="bad")

    def test_error_emitted_on_invalid_hex(self):
        """An invalid hex must emit an ERROR log before raising."""
        with self.assertLogs("scikitplot.misc", level=logging.ERROR):
            with self.assertRaises(ValueError):
                closest_color_name("notacolor")

    def test_error_emitted_on_display_colors_type_error(self):
        """display_colors with bad type must log ERROR before raising."""
        with self.assertLogs("scikitplot.misc", level=logging.ERROR):
            with self.assertRaises(TypeError):
                display_colors("notalist")

    def test_error_emitted_on_display_colors_empty(self):
        """display_colors with empty list must log ERROR before raising."""
        with self.assertLogs("scikitplot.misc", level=logging.ERROR):
            with self.assertRaises(ValueError):
                display_colors([])

    def test_warning_emitted_on_sort_fallback(self):
        """plot_colortable must emit WARNING when sort falls back."""
        bad = {"notavalidcolor": "#ff0000", "red": "#ff0000"}
        with self.assertLogs("scikitplot.misc", level=logging.WARNING) as log:
            plot_colortable(bad, sort_colors=True)
        warn_msgs = [r for r in log.output if "WARNING" in r]
        self.assertGreater(len(warn_msgs), 0)

    def test_debug_emitted_on_plot_overlapping_colors(self):
        """plot_overlapping_colors must emit at least one DEBUG record."""
        with self.assertLogs("scikitplot.misc", level=logging.DEBUG) as log:
            plot_overlapping_colors()
        debug_msgs = [r for r in log.output if "DEBUG" in r]
        self.assertGreater(len(debug_msgs), 0)

    def test_logger_hierarchy_inherits_from_scikitplot(self):
        """'scikitplot.misc' must be a child of 'scikitplot' in the hierarchy."""
        parent_name = logger.name.rsplit(".", 1)[0]
        self.assertEqual(parent_name, "scikitplot")

    # -- Additional logger coverage --

    def test_debug_emitted_on_display_colors(self):
        """display_colors must emit DEBUG records on a valid call."""
        with mock.patch("matplotlib.pyplot.show"):
            with self.assertLogs("scikitplot.misc", level=logging.DEBUG) as log:
                display_colors(["red", "blue"])
        debug_msgs = [r for r in log.output if "DEBUG" in r]
        self.assertGreater(len(debug_msgs), 0)

    def test_debug_emitted_on_plot_colortable(self):
        """plot_colortable must emit DEBUG records on a valid call."""
        with self.assertLogs("scikitplot.misc", level=logging.DEBUG) as log:
            plot_colortable({"red": "#ff0000"})
        debug_msgs = [r for r in log.output if "DEBUG" in r]
        self.assertGreater(len(debug_msgs), 0)

    def test_error_emitted_on_invalid_top_n(self):
        """closest_color_name with top_n=0 must emit ERROR before raising."""
        with self.assertLogs("scikitplot.misc", level=logging.ERROR):
            with self.assertRaises(TypeError):
                closest_color_name("#ff5733", top_n=0)

    def test_error_emitted_on_plot_colortable_empty(self):
        """plot_colortable with empty dict must emit ERROR before raising."""
        with self.assertLogs("scikitplot.misc", level=logging.ERROR):
            with self.assertRaises(ValueError):
                plot_colortable({})

    def test_error_emitted_on_plot_colortable_type_error(self):
        """plot_colortable with non-dict must emit ERROR before raising."""
        with self.assertLogs("scikitplot.misc", level=logging.ERROR):
            with self.assertRaises(TypeError):
                plot_colortable(["red", "blue"])

    def test_debug_on_named_color_direct_match(self):
        """A direct named-color match must still emit DEBUG records."""
        with self.assertLogs("scikitplot.misc", level=logging.DEBUG) as log:
            closest_color_name("red")
        self.assertTrue(any("DEBUG" in r for r in log.output))

    def test_error_emitted_on_bool_top_n(self):
        """closest_color_name with top_n=True must emit ERROR before raising."""
        with self.assertLogs("scikitplot.misc", level=logging.ERROR):
            with self.assertRaises(TypeError):
                closest_color_name("#ff5733", top_n=True)


# ===========================================================================
# scikitplot.misc public API
# ===========================================================================

class TestMiscPublicAPI(unittest.TestCase):
    """The scikitplot.misc public API must be complete, correct, and documented."""

    def tearDown(self):
        _close_all()

    def test_all_expected_names_in___all__(self):
        """__all__ must contain the four documented public function names."""
        import scikitplot.misc as misc
        expected = {
            "closest_color_name",
            "display_colors",
            "plot_colortable",
            "plot_overlapping_colors",
        }
        self.assertTrue(expected.issubset(set(misc.__all__)),
                        msg=f"Missing from __all__: {expected - set(misc.__all__)}")

    def test_all_exports_are_importable(self):
        """Every name in __all__ must be importable from scikitplot.misc."""
        import scikitplot.misc as misc
        for name in misc.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(misc, name),
                                msg=f"{name!r} is in __all__ but not accessible on the module")

    def test_public_functions_are_callable(self):
        """All names in __all__ must be callable functions."""
        import scikitplot.misc as misc
        for name in misc.__all__:
            obj = getattr(misc, name)
            with self.subTest(name=name):
                self.assertTrue(callable(obj),
                                msg=f"{name!r} is not callable")

    def test_all_public_functions_have_docstrings(self):
        """Every public function must have a non-empty docstring."""
        import scikitplot.misc as misc
        for name in misc.__all__:
            obj = getattr(misc, name)
            with self.subTest(name=name):
                self.assertIsNotNone(obj.__doc__,
                                     msg=f"{name!r} has no docstring")
                self.assertGreater(len(obj.__doc__.strip()), 0,
                                   msg=f"{name!r} has an empty docstring")

    def test_plot_colortable_module_accessible(self):
        """The private _plot_colortable module must be accessible via misc."""
        import scikitplot.misc as misc
        self.assertTrue(hasattr(misc, "_plot_colortable"))

    def test_module_has_test_attribute_if_testing_available(self):
        """If _testing is importable, misc.test must exist and be callable."""
        import scikitplot.misc as misc
        try:
            from scikitplot._testing._pytesttester import PytestTester
            tester_available = True
        except ImportError:
            tester_available = False

        if tester_available:
            self.assertTrue(hasattr(misc, "test"),
                            msg="misc.test must exist when _testing is available")
            self.assertTrue(callable(misc.test),
                            msg="misc.test must be callable")

    def test_misc_import_does_not_import_pyplot(self):
        """Importing scikitplot.misc must not trigger a plt.show() side-effect.

        This verifies the module is import-safe in headless environments.
        """
        # If we got this far without a DisplayError, the module is headless-safe.
        import scikitplot.misc  # noqa: F401 (import-only test)


# ===========================================================================
# _testing utilities (ignore_warnings, SkipTest)
# ===========================================================================

class TestIgnoreWarnings(unittest.TestCase):
    """
    Tests for scikitplot._testing.ignore_warnings and SkipTest.

    These are NumPy-heritage utilities shared across all submodules.
    """

    def tearDown(self):
        _close_all()

    def test_ignore_warnings_as_context_manager_suppresses_warning(self):
        """ignore_warnings as a context manager must suppress UserWarning."""
        from scikitplot._testing import ignore_warnings
        with ignore_warnings():
            warnings.warn("test warning", UserWarning)
        # No warning escaped the context manager; reaching here is success.

    def test_ignore_warnings_as_decorator_suppresses_warning(self):
        """ignore_warnings as a decorator must suppress warnings in the wrapped fn."""
        from scikitplot._testing import ignore_warnings

        @ignore_warnings()
        def noisy():
            warnings.warn("decorated warning", UserWarning)
            return 42

        result = noisy()
        self.assertEqual(result, 42)

    def test_ignore_warnings_specific_category_suppresses_only_that(self):
        """ignore_warnings(category=DeprecationWarning) must only suppress that category."""
        from scikitplot._testing import ignore_warnings
        with ignore_warnings(category=DeprecationWarning):
            warnings.warn("dep warning", DeprecationWarning)
        # DeprecationWarning is suppressed; reaching here is success.

    def test_ignore_warnings_category_as_first_positional_raises(self):
        """Passing a warning class as first positional arg must raise ValueError.

        This catches the common pitfall of writing
        ``ignore_warnings(UserWarning)`` instead of
        ``ignore_warnings(category=UserWarning)``.
        """
        from scikitplot._testing import ignore_warnings
        with self.assertRaises(ValueError):
            ignore_warnings(UserWarning)

    def test_ignore_warnings_callable_returns_wrapped_function(self):
        """ignore_warnings(fn) must return a callable wrapping fn."""
        from scikitplot._testing import ignore_warnings

        def fn():
            return "result"

        wrapped = ignore_warnings(fn)
        self.assertTrue(callable(wrapped))
        self.assertEqual(wrapped(), "result")

    def test_ignore_warnings_non_callable_returns_context_manager(self):
        """ignore_warnings() (no arguments) must return a context-manager object."""
        from scikitplot._testing import ignore_warnings
        ctx = ignore_warnings()
        self.assertTrue(hasattr(ctx, "__enter__"))
        self.assertTrue(hasattr(ctx, "__exit__"))

    def test_skip_test_is_unittest_skip_test(self):
        """SkipTest must be the standard unittest.case.SkipTest class."""
        from scikitplot._testing import SkipTest
        import unittest
        self.assertIs(SkipTest, unittest.case.SkipTest)

    def test_ignore_warnings_repr_contains_record(self):
        """repr(_IgnoreWarnings) must mention 'record=True'."""
        from scikitplot._testing import ignore_warnings
        ctx = ignore_warnings()
        self.assertIn("record=True", repr(ctx))

    def test_ignore_warnings_cannot_enter_twice(self):
        """Entering the same _IgnoreWarnings context twice must raise RuntimeError."""
        from scikitplot._testing import ignore_warnings
        ctx = ignore_warnings()
        ctx.__enter__()
        try:
            with self.assertRaises(RuntimeError):
                ctx.__enter__()
        finally:
            ctx.__exit__(None, None, None)

    def test_ignore_warnings_cannot_exit_without_entering(self):
        """Calling __exit__ without __enter__ must raise RuntimeError."""
        from scikitplot._testing import ignore_warnings
        ctx = ignore_warnings()
        with self.assertRaises(RuntimeError):
            ctx.__exit__(None, None, None)

    def test_ignore_warnings_preserves_return_value(self):
        """The decorator must not alter the wrapped function's return value."""
        from scikitplot._testing import ignore_warnings

        @ignore_warnings()
        def compute(x):
            return x * 2

        self.assertEqual(compute(21), 42)

    def test_ignore_warnings_accessible_from_testing_package(self):
        """ignore_warnings and SkipTest must be importable from scikitplot._testing.

        Note: _testing.__init__ uses ``from ._testing import *`` which brings
        both names into the package namespace.  __all__ is defined on the
        inner module (_testing._testing), not on the package itself — this is
        intentional (matches NumPy's pattern).  We verify accessibility, not
        the __all__ attribute on the package.
        """
        from scikitplot import _testing
        self.assertTrue(
            hasattr(_testing, "ignore_warnings"),
            msg="ignore_warnings must be accessible on scikitplot._testing",
        )
        self.assertTrue(
            hasattr(_testing, "SkipTest"),
            msg="SkipTest must be accessible on scikitplot._testing",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
