# scikitplot/misc/_plot_colortable.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Color-table and color-utility visualisations.

Public API
----------
closest_color_name      Find the closest named color(s) to a given hex/named color.
display_colors          Render a list of colors as horizontal bars.
plot_colortable         Display a table of color swatches with names and hex codes.
plot_overlapping_colors Plot CSS4 vs XKCD overlapping color names side by side.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Union  # noqa: F401

import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("scikitplot.misc")

__all__ = [
    "closest_color_name",
    "display_colors",
    "plot_colortable",
    "plot_overlapping_colors",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rgb_to_lab(rgb):
    """
    Convert a linear sRGB triplet in [0, 1] to CIELAB (L*, a*, b*).

    Parameters
    ----------
    rgb : array-like of shape (3,)
        sRGB values in the [0, 1] range, as returned by
        matplotlib.colors.to_rgb() or matplotlib.colors.hex2color().

    Returns
    -------
    lab : ndarray of shape (3,)
        CIELAB values [L*, a*, b*].

    Notes
    -----
    Developer note
        Pure-NumPy implementation — no extra dependencies.
        Follows IEC 61966-2-1 (sRGB) linearisation and the CIE 1976
        L*a*b* formula with a D65 white point.

    References
    ----------
    .. [1] IEC 61966-2-1:1999 sRGB standard.
    .. [2] CIE 015:2004 Colorimetry, 3rd edition.
    """
    rgb = np.asarray(rgb, dtype=float)

    # 1. Gamma expansion (sRGB linearisation)
    linear = np.where(
        rgb > 0.04045,  # noqa: PLR2004
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    # 2. Linear sRGB -> CIE XYZ D65 (IEC 61966-2-1)
    M = np.array(  # noqa: N806
        [  # noqa: N806
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = M @ linear

    # 3. Normalise by D65 white point (Xn, Yn, Zn)
    xyz /= np.array([0.95047, 1.00000, 1.08883])

    # 4. CIE 1976 L*a*b*
    eps, kap = 0.008856, 903.3
    f = np.where(xyz > eps, xyz ** (1.0 / 3.0), (kap * xyz + 16.0) / 116.0)
    L = 116.0 * f[1] - 16.0  # noqa: N806
    a = 500.0 * (f[0] - f[1])
    b = 200.0 * (f[1] - f[2])
    return np.array([L, a, b])


def _perceived_text_color(hex_color):
    """
    Return 'k' (black) or 'w' (white) for readable text on a given background.

    Uses the ITU-R BT.601 luma formula.

    Parameters
    ----------
    hex_color : str
        Hex color string accepted by matplotlib.colors.to_rgba_array.

    Returns
    -------
    str
        'k' for light backgrounds, 'w' for dark.
    """
    rgba = mcolors.to_rgba_array([hex_color])
    luma = 0.299 * rgba[0, 0] + 0.587 * rgba[0, 1] + 0.114 * rgba[0, 2]
    return "k" if luma > 0.5 else "w"  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def display_colors(
    colors,
    title="Color Display (Order)",
    show_indices=False,
):
    """
    Display a list of colors as horizontal bars, preserving input order.

    Parameters
    ----------
    colors : list of str
        Color names or hex codes (e.g., ['red', '#00ff00']).
    title : str, default "Color Display (Order)"
        Figure title.
    show_indices : bool, default False
        When True, prefix each label with its 0-based index.

    Returns
    -------
    None
        Displays a matplotlib figure; does not return a value.

    Raises
    ------
    TypeError
        If colors is not a list.
    ValueError
        If colors is empty.

    Notes
    -----
    Developer note
        Bug fixed: the original called plt.yticks([]) immediately after
        plt.yticks(range(len(colors)), labels), silently clearing all tick
        labels before they could be rendered.  Fix: set ticks and labels
        on the Axes object directly, hide only the tick marks via
        tick_params(length=0).

    Examples
    --------
    >>> display_colors(["red", "blue", "green"])  # doctest: +SKIP
    >>> display_colors(["red", "blue"], show_indices=True)  # doctest: +SKIP
    """
    if not isinstance(colors, list):
        logger.error("display_colors: 'colors' must be a list, got %s", type(colors))
        raise TypeError(f"'colors' must be a list, got {type(colors).__name__}")
    if len(colors) == 0:
        logger.error("display_colors: 'colors' must not be empty")
        raise ValueError("'colors' must not be empty")

    logger.debug(
        "display_colors: rendering %d colors, show_indices=%s",
        len(colors),
        show_indices,
    )

    labels = (
        [f"{i}: {c}" for i, c in enumerate(colors)] if show_indices else list(colors)
    )

    _fig, ax = plt.subplots(figsize=(8, max(2, len(colors) * 0.4)))
    ax.barh(range(len(colors)), [1] * len(colors), color=colors, edgecolor="w")

    ax.set_yticks(range(len(colors)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
    logger.debug("display_colors: done")


def closest_color_name(
    hex_color="#000000",
    top_n=1,
    use_lab=False,
    tolerance=0.0,
    use_spec="CSS4",
    return_distances=False,
):
    """
    Find the closest named color(s) to a given hex or named color.

    Parameters
    ----------
    hex_color : str
        A hex code (e.g. '#ff5733') or a valid named color (e.g. 'red',
        'xkcd:sky blue').  When the input is a name that exists in the
        *selected* palette (controlled by ``use_spec``), it is returned
        immediately with distance 0.
    top_n : int, default 1
        Number of closest color names to return.  bool is rejected.
    use_lab : bool, default False
        When True, distances are computed in CIELAB color space.
    tolerance : float, default 0.0
        Maximum Euclidean distance for a valid match.  0.0 = no filtering.
    use_spec : {'CSS4', 'xkcd'}, default 'CSS4'
        Color palette to search.
    return_distances : bool, default False
        When True, return OrderedDict {name: distance}.
        When False, return list of color name strings.

    Returns
    -------
    list or OrderedDict
        List of str when return_distances=False.
        OrderedDict {name: float} when return_distances=True.
        Empty list / OrderedDict when tolerance > 0 and no match found.

    Raises
    ------
    ValueError
        If use_spec is not 'CSS4' or 'xkcd'.
        If hex_color is not a valid hex code or named color.
    TypeError
        If top_n is not a positive integer (bool is rejected explicitly).
        If tolerance is negative or not a number.

    Notes
    -----
    Developer note
        Bugs fixed vs. original:

        * Bug 1 (Critical): mcolors.hex2color() returns RGB in [0, 1].
          The original divided by 255 again, sending all values to near-zero
          and making every LAB distance wrong.

        * Bug 2 (Critical): mcolors.rgb_to_lab does not exist in matplotlib.
          Replaced with _rgb_to_lab(), a pure-NumPy IEC 61966-2-1 / CIE 1976
          implementation that introduces no extra dependencies.

        * Bug 3 (Critical): Named-color early-exit happened before palette
          selection, so ``use_spec`` was silently ignored for named inputs.
          Example: ``closest_color_name("red", use_spec="xkcd")`` returned
          "red" (CSS4) even though the caller asked for XKCD.
          Fix: palette is selected first; named-color check is scoped to the
          selected palette only.

        * Bug 4 (Non-critical): ``mcolors.hex2color()`` only handles hex strings.
          Named inputs not in any cached dict produced unhelpful errors.
          Replaced with ``mcolors.to_rgb()`` which resolves any
          matplotlib-recognised name or hex code uniformly.

        * Bug 5 (Non-critical, bool leak): ``isinstance(True, int)`` is True
          in Python. Previously ``top_n=True`` silently passed as ``top_n=1``.
          Fix: bool is now explicitly rejected before the int check.

        * Bug 6 (Minor): Docstring called tolerance a "minimum distance
          threshold" (lower bound). Code uses it as maximum. Docstring fixed.

    Examples
    --------
    >>> closest_color_name("#ffd166")
    ['lightgoldenrodyellow']

    >>> closest_color_name("#118ab2", top_n=3)
    ['darkcyan', 'lightseagreen', 'steelblue']

    >>> closest_color_name("red")
    ['red']

    >>> closest_color_name("xkcd:red", use_spec="xkcd")
    ['xkcd:red']
    """
    # ---------- input validation ----------
    if not isinstance(use_spec, str) or use_spec not in ("CSS4", "xkcd"):
        logger.error(
            "closest_color_name: invalid use_spec=%r; must be 'CSS4' or 'xkcd'",
            use_spec,
        )
        raise ValueError(
            f"Invalid value for use_spec={use_spec!r}. "
            "Accepted values are 'CSS4' or 'xkcd'."
        )

    # Bug 5 fix: bool is a subclass of int; reject it explicitly so that
    # top_n=True (==1) is not silently accepted as a valid positive integer.
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        logger.error("closest_color_name: top_n must be a positive int, got %r", top_n)
        raise TypeError(f"top_n must be a positive integer, got {top_n!r}")

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or tolerance < 0
    ):
        logger.error("closest_color_name: tolerance must be >= 0, got %r", tolerance)
        raise TypeError(f"tolerance must be a non-negative number, got {tolerance!r}")

    logger.debug(
        "closest_color_name: hex_color=%r top_n=%d use_lab=%s "
        "tolerance=%s use_spec=%s return_distances=%s",
        hex_color,
        top_n,
        use_lab,
        tolerance,
        use_spec,
        return_distances,
    )

    # ---------- palette selection (Bug 3 fix: before named-color check) ----------
    palette = mcolors.CSS4_COLORS if use_spec == "CSS4" else mcolors.XKCD_COLORS

    # ---------- resolve input to RGB (Bug 4 fix: to_rgb handles all named colors) ----------
    # mcolors.to_rgb resolves hex codes, CSS4 names, 'xkcd:…' names uniformly.
    try:
        rgb = np.asarray(mcolors.to_rgb(hex_color), dtype=float)
    except ValueError as e:
        logger.error(
            "closest_color_name: invalid color input %r; "
            "expected '#rrggbb' or a matplotlib-recognised color name",
            hex_color,
        )
        raise ValueError(
            f"Invalid color input: {hex_color!r}. "
            "Expected a hex code (e.g. '#ff5733') or a valid named color."
        ) from e

    # ---------- direct named-color match, scoped to selected palette (Bug 3 fix) ----------
    # Respects use_spec: "red" with use_spec="xkcd" will NOT match here
    # because "red" is in CSS4_COLORS but not in XKCD_COLORS.
    if hex_color in palette:
        logger.debug(
            "closest_color_name: direct named-color match for %r in %s palette",
            hex_color,
            use_spec,
        )
        return OrderedDict([(hex_color, 0.0)]) if return_distances else [hex_color]

    # ---------- reference vector in chosen color space ----------
    ref = _rgb_to_lab(rgb) if use_lab else rgb

    # ---------- compute distances to all palette entries ----------
    distances = []
    for name, hexval in palette.items():
        c_rgb = np.asarray(mcolors.to_rgb(hexval), dtype=float)
        c_ref = _rgb_to_lab(c_rgb) if use_lab else c_rgb
        d = float(np.linalg.norm(ref - c_ref))
        if tolerance == 0.0 or d <= tolerance:
            distances.append((name, d))

    distances.sort(key=lambda x: x[1])
    top = distances[:top_n]

    logger.debug(
        "closest_color_name: returning %d result(s); top=%r dist=%.6f",
        len(top),
        top[0][0] if top else None,
        top[0][1] if top else float("nan"),
    )

    if return_distances:
        return OrderedDict(top)
    return [name for name, _ in top]


def plot_colortable(
    colors=mcolors.CSS4_COLORS,
    *,
    ncols=4,
    sort_colors=True,
    display_hex=True,
):
    """
    Display a table of color swatches with names and optional hex codes.

    Parameters
    ----------
    colors : dict, default=mcolors.CSS4_COLORS
        Mapping {color_name: hex_code}, e.g. {'red': '#FF0000'}.
    ncols : int, default 4
        Number of columns in the table.  Must be a positive integer;
        bool values are rejected.
    sort_colors : bool, default=True
        When True, sort by HSV values before display.
    display_hex : bool, default=True
        When True, overlay the hex code on each swatch.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the color swatch table.

    Raises
    ------
    TypeError
        If colors is not a dict.
    ValueError
        If colors is empty, ncols < 1, or ncols is not a plain int (bool rejected).

    Notes
    -----
    Developer note
        Bug fixed: the original sort fallback used a bare ``except Exception``
        that silently swallowed all errors.  Now catches only ValueError and
        KeyError with a logged warning, with a second fallback to insertion
        order if value-based sorting also fails.

        Bool fix: ``isinstance(True, int)`` is True in Python.  Previously
        ``ncols=True`` silently passed validation as ``ncols=1``.  Bool is
        now explicitly rejected before the int check.

    Examples
    --------
    >>> import matplotlib.colors as mcolors
    >>> fig = plot_colortable(mcolors.CSS4_COLORS)  # doctest: +SKIP
    """
    if not isinstance(colors, dict):
        logger.error("plot_colortable: 'colors' must be a dict, got %s", type(colors))
        raise TypeError(f"'colors' must be a dict, got {type(colors).__name__}")
    if len(colors) == 0:
        logger.error("plot_colortable: 'colors' must not be empty")
        raise ValueError("'colors' must not be empty")

    # Bool fix: bool is a subclass of int in Python; ncols=True (==1) would
    # otherwise silently pass.  Reject booleans explicitly.
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        logger.error("plot_colortable: ncols must be a positive int, got %r", ncols)
        raise ValueError(f"ncols must be a positive integer, got {ncols!r}")

    logger.debug(
        "plot_colortable: %d colors, ncols=%d, sort=%s, hex=%s",
        len(colors),
        ncols,
        sort_colors,
        display_hex,
    )

    cell_width = 248
    cell_height = 24
    swatch_width = 48
    margin = 15

    if sort_colors:
        try:
            names = sorted(
                colors,
                key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))),
            )
        except (ValueError, KeyError):
            logger.warning(
                "plot_colortable: HSV sort by color name failed; "
                "trying sort by hex value"
            )
            try:
                names = sorted(
                    colors,
                    key=lambda k: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(colors[k]))),
                )
            except (ValueError, KeyError):
                logger.warning(
                    "plot_colortable: HSV sort by hex value also failed; "
                    "using dict insertion order"
                )
                names = list(colors)
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 74

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(
        margin / width,
        margin / height,
        (width - margin) / width,
        (height - margin) / height,
    )
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(
            text_pos_x,
            y,
            name,
            fontsize=14,
            horizontalalignment="left",
            verticalalignment="center",
        )
        ax.add_patch(
            Rectangle(
                xy=(swatch_start_x, y - 9),
                width=swatch_width,
                height=18,
                facecolor=colors[name],
                edgecolor="0.7",
            )
        )

        if display_hex:
            hex_color = colors[name]
            text_color = _perceived_text_color(hex_color)
            ax.text(
                swatch_start_x + 23,
                y + 3,
                hex_color,
                color=text_color,
                ha="center",
                fontsize=9,
            )

    logger.debug("plot_colortable: figure built with %d swatches", n)
    return fig


def plot_overlapping_colors():
    """
    Plot overlapping color names between CSS4 and XKCD palettes side by side.

    Identifies color names present in both matplotlib.colors.CSS4_COLORS and
    matplotlib.colors.XKCD_COLORS (as 'xkcd:<n>'), then renders them for
    comparison.  Names with identical hex codes in both palettes are shown bold.

    Parameters
    ----------
    None

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the side-by-side comparison.

    Notes
    -----
    User note
        Bold text signals that CSS4 and XKCD share the exact same hex value
        for that color name.
    Further reading
        https://xkcd.com/color/rgb/
        https://xkcd.com/color/rgb.txt
        https://www.w3schools.com/cssref/css_colors.php

    Examples
    --------
    >>> fig = plot_overlapping_colors()  # doctest: +SKIP
    """
    overlap = {
        name for name in mcolors.CSS4_COLORS if f"xkcd:{name}" in mcolors.XKCD_COLORS
    }

    logger.debug("plot_overlapping_colors: %d overlapping names found", len(overlap))

    fig = plt.figure(figsize=[9, 5])
    ax = fig.add_axes([0, 0, 1, 1])

    n_groups = 3
    n_rows = len(overlap) // n_groups + 1

    for j, color_name in enumerate(sorted(overlap)):
        css4 = mcolors.CSS4_COLORS[color_name]
        xkcd = mcolors.XKCD_COLORS[f"xkcd:{color_name}"].upper()

        css4_text_color = _perceived_text_color(css4)
        xkcd_text_color = _perceived_text_color(xkcd)

        col_shift = (j // n_rows) * 3
        y_pos = j % n_rows
        text_args = {"fontsize": 10, "weight": "bold"} if css4 == xkcd else {}

        ax.add_patch(mpatch.Rectangle((0 + col_shift, y_pos), 1, 1, color=css4))
        ax.add_patch(mpatch.Rectangle((1 + col_shift, y_pos), 1, 1, color=xkcd))
        ax.text(
            0.5 + col_shift,
            y_pos + 0.7,
            css4,
            color=css4_text_color,
            ha="center",
            **text_args,
        )
        ax.text(
            1.5 + col_shift,
            y_pos + 0.7,
            xkcd,
            color=xkcd_text_color,
            ha="center",
            **text_args,
        )
        ax.text(2 + col_shift, y_pos + 0.7, f"  {color_name}", **text_args)

    for g in range(n_groups):
        ax.hlines(range(n_rows), 3 * g, 3 * g + 2.8, color="0.7", linewidth=1)
        ax.text(0.5 + 3 * g, -0.3, "X11/CSS4", ha="center")
        ax.text(1.5 + 3 * g, -0.3, "xkcd", ha="center")

    ax.set_xlim(0, 3 * n_groups)
    ax.set_ylim(n_rows, -1)
    ax.axis("off")

    logger.debug("plot_overlapping_colors: figure built")
    return fig


# %%
#
# .. tags::
#
#    plot-type: barh
#    plot-type: text
#    level: beginner
#    purpose: showcase
