"""plot_serializer.py"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught

import collections.abc as cab
import json
import os

import matplotlib as mpl  # type: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
import numpy as np  # type: ignore[reportMissingModuleSource]

# ========== UTILITIES ==========


def safe_json_converter(o):
    """Convert NumPy types to native Python types for JSON."""
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def get_ax_from_input(input_plot=None):  # noqa: PLR0912
    """
    Safely extract one or more matplotlib Axes from a wide range of inputs.

    This utility is designed for flexibility and robustness in data visualization workflows.

    Parameters
    ----------
    input_plot: Can be any of the following:
        - None: returns the current active Axes (plt.gca())
        - mpl.axes.Axes: returned as-is
        - mpl.figure.Figure: returns the default Axes using fig.gca()
        - tuple: expected form is (Figure, Axes) or (Figure, list/array of Axes)
        - Iterable: list, tuple, or numpy.ndarray of Axes or Figures

    Returns
    -------
    matplotlib.axes.Axes
        A single mpl.axes.Axes object or a list of such objects.

    Raises
    ------
    TypeError
        If the input is unsupported or invalid.
    """

    # Case 1: No input → return current active Axes
    if input_plot is None:
        return plt.gca()

    # Case 2: Direct Axes instance
    if isinstance(input_plot, mpl.axes.Axes):
        return input_plot

    # Case 3: A single Figure instance → return its main Axes
    if isinstance(input_plot, mpl.figure.Figure):
        return input_plot.gca()

    # Case 4: Tuple input, typically from plt.subplots()
    if isinstance(input_plot, tuple):
        if len(input_plot) != 2:  # noqa: PLR2004
            raise TypeError(
                "Tuple input must be of form (Figure, Axes or list of Axes)"
            )

        fig, ax = input_plot

        # Subcase: second element is single Axes
        if isinstance(ax, mpl.axes.Axes):
            return ax

        # Subcase: second element is a sequence of Axes (e.g., from subplots)
        elif isinstance(ax, (list, tuple, np.ndarray)) and all(  # noqa: RET505
            isinstance(a, mpl.axes.Axes) for a in ax
        ):
            return list(ax)

        raise TypeError("Second element of tuple must be an Axes or iterable of Axes")

    # Case 5: Handle numpy arrays of Axes by flattening
    if isinstance(input_plot, np.ndarray):
        input_plot = input_plot.flatten().tolist()

    # Case 6: Handle lists/tuples of Figures or Axes
    if isinstance(input_plot, cab.Iterable):
        axes_list = []

        for item in input_plot:
            if isinstance(item, mpl.axes.Axes):
                axes_list.append(item)
            elif isinstance(item, mpl.figure.Figure):
                # Extract all Axes from the figure
                axes_list.extend(item.get_axes())

        if axes_list:
            return axes_list

        raise TypeError("Iterable did not contain any valid Axes or Figures")

    # Fallback: input not recognized
    raise TypeError(
        "Input must be a matplotlib Figure, Axes, tuple, or iterable of them."
    )


def detect_plot_type(ax, thr=1.5):
    """
    Attempt to infer plot type from Axes content.

    Returns
    -------
    str
        'histogram', 'barplot', 'lineplot', or None.
    """
    if ax.patches:
        # Bar or histogram: distinguish by spacing
        widths = [patch.get_width() for patch in ax.patches]
        unique_widths = set(round(w, 5) for w in widths)  # noqa: C401
        if len(unique_widths) == 1 and next(iter(unique_widths)) < thr:
            return "histogram"
        return "barplot"

    if ax.lines:
        return "lineplot"

    return None


# ========== SERIALIZER FUNCTIONS ==========


def serialize_histplot(ax):
    """Serialize histogram-style Axes to JSON."""
    try:
        bars = ax.patches
        if not bars:
            raise ValueError("No histogram bars found.")

        bin_edges = []
        counts = []

        for bar in bars:
            x_start = float(bar.get_x())
            width = float(bar.get_width())
            height = float(bar.get_height())

            bin_edges.append(x_start)
            counts.append(height)

        bin_edges.append(float(bars[-1].get_x() + bars[-1].get_width()))

        return {
            "type": "histogram",
            "title": ax.get_title(),
            "x_label": ax.get_xlabel(),
            "y_label": ax.get_ylabel(),
            "bin_edges": bin_edges,
            "counts": counts,
        }

    except Exception as e:
        print(f"[serialize_histplot ERROR] {e}")  # noqa: T201
        return None


def serialize_barplot(ax):
    """Serialize barplot-style Axes to JSON."""
    try:
        bars = ax.patches
        if not bars:
            raise ValueError("No bar patches found.")

        labels = []
        heights = []

        for bar in bars:
            labels.append(str(bar.get_x() + bar.get_width() / 2))
            heights.append(float(bar.get_height()))

        return {
            "type": "barplot",
            "title": ax.get_title(),
            "x_label": ax.get_xlabel(),
            "y_label": ax.get_ylabel(),
            "labels": labels,
            "heights": heights,
        }

    except Exception as e:
        print(f"[serialize_barplot ERROR] {e}")  # noqa: T201
        return None


def serialize_lineplot(ax):
    """Serialize line plot from Line2D objects."""
    try:
        lines = ax.lines
        if not lines:
            raise ValueError("No line data found.")

        all_lines = []
        for line in lines:
            x = line.get_xdata()
            y = line.get_ydata()
            all_lines.append(
                {
                    "label": line.get_label(),
                    "x": list(map(float, x)),
                    "y": list(map(float, y)),
                }
            )

        return {
            "type": "lineplot",
            "title": ax.get_title(),
            "x_label": ax.get_xlabel(),
            "y_label": ax.get_ylabel(),
            "lines": all_lines,
        }

    except Exception as e:
        print(f"[serialize_lineplot ERROR] {e}")  # noqa: T201
        return None


# ========== UNIFIED ENTRY POINT ==========


def serialize_plot(input_plot=None, pretty=True):
    """
    Main entry: serialize any supported plot to JSON.

    Args:
        input_plot: matplotlib Figure, Axes, or None
        pretty: pretty-print JSON

    Returns
    -------
    str or None
        JSON string or None
    """
    try:
        ax = get_ax_from_input(input_plot)
        plot_type = detect_plot_type(ax)

        if plot_type == "histogram":
            data = serialize_histplot(ax)
        elif plot_type == "barplot":
            data = serialize_barplot(ax)
        elif plot_type == "lineplot":
            data = serialize_lineplot(ax)
        else:
            raise ValueError("Unsupported or unknown plot type.")

        if data is None:
            raise ValueError("Serialization returned no data.")

        return json.dumps(
            data, indent=4 if pretty else None, default=safe_json_converter
        )

    except Exception as e:
        print(f"[serialize_plot ERROR] {e}")  # noqa: T201
        return None


# ========== FILE WRITER ==========


def save_to_file(json_str, path):
    """
    Save JSON string to a file, ensuring directory exists.

    Parameters
    ----------
    json_str : str
        JSON content
    path : str
        File path to save to
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # noqa: PTH103, PTH120
        with open(path, "w", encoding="utf-8") as f:  # noqa: PTH123
            f.write(json_str)
        print(f"[INFO] Saved plot JSON to: {path}")  # noqa: T201
    except Exception as e:
        print(f"[save_to_file ERROR] {e}")  # noqa: T201
