"""
misc
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import math
from collections import OrderedDict

import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

# from typing import List
import numpy as np
from matplotlib.patches import Rectangle

__all__ = [
    "closest_color_name",
    "display_colors",
    "plot_colortable",
    "plot_overlapping_colors",
]


def display_colors(
    colors: list[str], title: str = "Color Display (Order)", show_indices: bool = False
) -> None:
    """
    Display a list of colors as horizontal bars, preserving the order of the colors.

    Parameters
    ----------
    colors : list[str]
        A list of color names or color codes (e.g., ['red', 'blue', 'green']).

    title : str, optional, default="Color Display (Order)"
        The title to display on the plot.

    show_indices : bool, optional, default=False
        Whether to display the color indices beside each color. If True, the index of each color in the list will
        be shown next to the color bar.

    Returns
    -------
    None
        This function displays a plot and does not return any value.

    Example
    -------
    display_colors(['gray', 'orange', 'red', 'pink', 'salmon', 'olive', 'limegreen', 'green', 'dodgerblue', 'cyan', 'blue', 'purple', 'brown'])
    display_colors(['gray', 'orange', 'red', 'pink'], title="My Custom Color Display", show_indices=True)

    """
    # Create a plot to display the colors, ensuring that order is preserved
    plt.figure(figsize=(8, 2))  # Adjust the figure size as needed

    # Display the colors as horizontal bars
    plt.barh(range(len(colors)), [1] * len(colors), color=colors, edgecolor="w")

    # Set the labels to be the color names, preserving the order
    if show_indices:
        # If show_indices is True, display the index alongside the color name
        labels = [f"{i}: {color}" for i, color in enumerate(colors)]
    else:
        labels = colors

    plt.yticks(range(len(colors)), labels)

    # Remove the axis labels and ticks
    plt.xticks([])
    plt.yticks([])

    # Add a title
    plt.title(title)

    # Show the plot
    plt.show()


def closest_color_name(
    hex_color: str,
    top_n: int = 1,
    use_lab: bool = False,
    tolerance: float = 0.0,
    use_spec: str = "CSS4",
    return_distances: bool = False,
) -> OrderedDict:
    """
    Find the closest color name(s) to a given hex color code using either CSS4 or xkcd color palettes.

    Parameters
    ----------
    hex_color : str
        A string representing a hex color code (e.g., '#ff5733') or a named color (e.g., 'red').

    top_n : int, optional, default=1
        The number of closest color names to return. If set to 1, only the closest match is returned.
        If set to 3, the top 3 closest colors will be returned, and so on.

    use_lab : bool, optional, default=False
        If True, the comparison will be done in the CIELAB color space instead of RGB.
        This may improve color perception matching.

    tolerance : float, optional, default=0.0
        Minimum distance threshold for a valid match. If set to a positive value (e.g., 0.1),
        only colors within this distance are returned. A tolerance of 0.0 means no filtering.

    use_spec : str, optional, default="CSS4"
        Specify the color palette to use for matching. Accepted values are:
        - "CSS4" (default): Uses the CSS4 color palette.
        - "xkcd": Uses the xkcd color palette.

    return_distances : bool, optional, default=False
        If True, the function will return a list of tuples containing the color name and its distance from the input color.
        If False, only the names of the closest colors are returned.

    Returns
    -------
    OrderedDict or list
        - If `return_distances=True`, returns an ordered dictionary with color names as keys and their distances as values.
        - If `return_distances=False`, returns a list of the closest color names.

    Notes
    -----
    This function computes the color distance using either RGB or CIELAB color space. The color with the smallest distance
    to the input color is returned. If `top_n` is greater than 1, the top N closest matches are returned.

    Example
    -------
    >>> closest_color_name("#ffd166", return_distances=False)
    ['lightgoldenrodyellow']

    >>> closest_color_name("#ffd166", return_distances=True)
    OrderedDict([('lightgoldenrodyellow', 0.00234)])

    >>> closest_color_name("#118ab2", top_n=3, return_distances=False)
    ['deepskyblue', 'dodgerblue', 'royalblue']

    >>> closest_color_name("#ff5733", use_spec="xkcd")
    OrderedDict([('red', 0.00456), ('orange', 0.00567), ('yellow', 0.00678)])

    >>> closest_color_name("#ff5733", use_spec="xkcd", tolerance=0.05)
    ['red', 'orange']

    >>> closest_color_name("red", return_distances=False)
    ['red']

    >>> closest_color_name("notacolor", return_distances=False)
    ValueError: Invalid color input

    """
    # Validate the use_spec parameter
    if use_spec not in ["CSS4", "xkcd"]:
        raise ValueError(
            "Invalid value for use_spec. Accepted values are 'CSS4' or 'xkcd'."
        )

    # Check if the input is a named color or a hex code
    if hex_color in mcolors.CSS4_COLORS or hex_color in mcolors.XKCD_COLORS:
        # Direct match if color name is found
        return OrderedDict([(hex_color, 0.0)]) if return_distances else [hex_color]

    # Convert hex color to RGB
    rgb = mcolors.hex2color(hex_color)

    # Convert RGB to LAB if needed
    if use_lab:
        rgb = np.array(rgb)  # Convert to numpy array for easier manipulation
        rgb = rgb / 255.0  # Normalize to [0, 1] range
        lab = mcolors.rgb_to_lab(rgb)
    else:
        lab = None

    # Function to calculate Euclidean distance
    def calculate_distance(color1, color2, use_lab=False):
        if use_lab:
            return np.linalg.norm(color1 - color2)
        return np.linalg.norm(np.array(color1) - np.array(color2))

    # Select the correct color palette (CSS4 or xkcd)
    color_palette = mcolors.CSS4_COLORS if use_spec == "CSS4" else mcolors.XKCD_COLORS

    # Create a list of tuples containing (color_name, distance)
    color_distances = []

    for color_name, color_hex in color_palette.items():
        color_rgb = mcolors.hex2color(color_hex)
        if use_lab:
            color_rgb = np.array(color_rgb) / 255.0
            color_lab = mcolors.rgb_to_lab(color_rgb)
            distance = calculate_distance(lab, color_lab, use_lab=True)
        else:
            distance = calculate_distance(rgb, color_rgb, use_lab=False)

        # Only add colors within the tolerance distance
        if tolerance == 0.0 or distance <= tolerance:
            color_distances.append((color_name, distance))

    # Sort by distance and get the top N closest colors
    color_distances.sort(key=lambda x: x[1])

    # Return the color names in the desired format
    if return_distances:
        # Return an ordered dictionary with names and distances
        ordered_result = OrderedDict(color_distances[:top_n])
        return ordered_result  # noqa: RET504
    # Return a list of color names only
    color_names = [color_name for color_name, _ in color_distances[:top_n]]
    return color_names  # noqa: RET504


def plot_colortable(
    colors: dict, *, ncols: int = 4, sort_colors: bool = True, display_hex: bool = True
) -> plt.Figure:
    """
    Display a table of color swatches with corresponding names and optional hex values.

    Parameters
    ----------
    colors : dict
        A dictionary where keys are color names and values are their corresponding
        hex color codes (e.g., {'red': '#FF0000', 'green': '#00FF00'}).

    ncols : int, optional
        The number of columns in the color table. Default is 4.

    sort_colors : bool, optional
        If True, colors will be sorted by hue, saturation, value, and name before
        being displayed. Default is True.

    display_hex : bool, optional
        If True, the hex color code will be displayed next to the swatch. Default is True.

    Returns
    -------
    plt.Figure
        A matplotlib figure object containing the color swatch table.

    Notes
    -----
    This function visualizes a list of colors with their corresponding names
    and optional hex codes in a tabular format. The colors can be sorted
    based on their HSV values if the `sort_colors` parameter is set to True.
    The swatch for each color is displayed alongside its name and hex code (if enabled).

    """
    cell_width = 248
    cell_height = 24
    swatch_width = 48
    margin = 15

    # Sort colors by hue, saturation, value and name.
    if sort_colors:
        try:
            names = sorted(
                colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c)))
            )
        except Exception:
            names = sorted(
                colors,
                key=lambda k: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(colors.get(k)))),
            )
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
            # Pick text colour based on perceived luminance.
            rgba = mcolors.to_rgba_array([hex_color])
            luma = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
            thr = 0.5
            text_color = "k" if luma[0] > thr else "w"
            text_args = dict(fontsize=9)  # noqa: C408
            ax.text(
                swatch_start_x + 23,
                y + 3,
                hex_color,
                color=text_color,
                ha="center",
                **text_args,
            )

    return fig


def plot_overlapping_colors():
    """
    Plot overlapping color names from CSS4 and XKCD color dictionaries using matplotlib.

    This function identifies overlapping color names between the CSS4 and XKCD color collections
    provided by `matplotlib`. It plots these overlapping colors side by side, with:

    - The CSS4 color on the left.
    - The XKCD color on the right.

    If the CSS4 and XKCD colors have identical hex values, the name is displayed in bold.
    A horizontal line separates each row for better readability, and column headers indicate
    the source of each color.

    Parameters
    ----------
    None

    Returns
    -------
    plt.Figure
        A matplotlib figure object containing the plotted overlapping colors.

    Notes
    -----
    This function visualizes the overlapping colors between CSS4 and XKCD, making it easier to
    compare the color representations from both collections. Each color swatch is labeled with
    its respective name, and the layout adapts to the number of overlapping colors.

    """
    overlap = {
        name for name in mcolors.CSS4_COLORS if f"xkcd:{name}" in mcolors.XKCD_COLORS
    }

    fig = plt.figure(figsize=[9, 5])
    ax = fig.add_axes([0, 0, 1, 1])

    n_groups = 3
    n_rows = len(overlap) // n_groups + 1

    for j, color_name in enumerate(sorted(overlap)):
        css4 = mcolors.CSS4_COLORS[color_name]
        xkcd = mcolors.XKCD_COLORS[f"xkcd:{color_name}"].upper()

        # Pick text colour based on perceived luminance.
        rgba = mcolors.to_rgba_array([css4, xkcd])
        luma = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
        thr = 0.5
        css4_text_color = "k" if luma[0] > thr else "w"
        xkcd_text_color = "k" if luma[1] > thr else "w"

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

    return fig
