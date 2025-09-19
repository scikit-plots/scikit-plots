# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

"""
Decile-based model evaluation module (Lift, Gains, KS statistics).

The :mod:`~scikitplot.kds` module includes plots for machine learning
evaluation decile analysis e.g. Gain, Lift and Decile charts, etc.

Seaborn-style decile analysis (Lift / Gain / KS) rewritten to use VectorPlotter.

Design principles:
- Accept `data=..., x=..., y=..., hue=...` like seaborn.
- Compute decile-level statistics per hue/facet subset using VectorPlotter.iter_data.
- Return pandas DataFrame for decile_table and Axes for plotting functions.

This module provides:
- decile_table(...) : compute decile-level statistics
- plot_cumulative_lift(...) : cumulative lift curve
- plot_decile_wise_lift(...) : lift per decile
- plot_cumulative_gain(...) : cumulative gain curve
- plot_ks_statistic(...) : KS statistic and helper markers
- report(...) : convenience function producing decile table + subplots

References
----------
[1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py
"""

# code that needs to be compatible with both Python 2 and Python 3
from __future__ import annotations

import json
import warnings
from pprint import pprint
from typing import ClassVar, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba

try:
    from seaborn._base import VectorPlotter
    from seaborn._compat import groupby_apply_include_groups
    from seaborn.external import husl
    from seaborn.utils import _check_argument, _default_color
except:
    from ..externals._seaborn._base import VectorPlotter
    from ..externals._seaborn._compat import groupby_apply_include_groups
    from ..externals._seaborn.external import husl
    from ..externals._seaborn.utils import (
        _check_argument,
        _default_color,
    )

## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
    "kdsplot",
    "print_labels",
]


def get_label_info() -> "dict[str, str]":  # noqa: UP037
    """
    Return a comprehensive legend for decile table columns.

    Each key represents a decile table column.
    The value contains:
    - Meaning: What the column represents
    - Critical check / logic: How to interpret, why it matters, and potential pitfalls
    """
    return {
        "decile": (
            "Meaning: Ranked group (1 = highest predicted probability). "
            "Critical: Ensure sorted descending by model score. Fatal if top deciles don't capture positives."
            "Formula: rank by model score into k quantiles (e.g., 10 deciles). "
        ),
        "prob_min": (
            "Meaning: Lowest predicted probability in the decile. "
            "Critical: Signals model calibration. Fatal if too close to prob_max (poor ranking)."
            "Formula: min(score in decile). "
        ),
        "prob_max": (
            "Meaning: Highest predicted probability in the decile. "
            "Critical: Checks separation. Fatal if overlaps lower deciles (poor discrimination)."
            "Formula: max(score in decile). "
        ),
        "prob_avg": (
            "Meaning: Average predicted probability in the decile. "
            "Critical: Useful for calibration curves; should decrease monotonically across deciles."
            "Formula: mean(score in decile). "
        ),
        "cnt_resp_total": (
            "Meaning: Total samples in the decile. "
            "Critical: Denominator for rate_resp and cumulative % calculations. Fatal if deciles uneven."
            "Formula: count(samples in decile). "
        ),
        "cnt_resp": (
            "Meaning: Actual responders in the decile (how many responders we captured). "
            "Critical: Should never exceed cnt_resp_wiz. Flat counts across deciles indicate useless model."
            "Formula: sum(y_true=1 in decile). "
        ),
        "cnt_resp_non": (
            "Meaning: Non-responders in the decile. "
            "Critical: Used for KS/statistics. Too high in top deciles is a warning."
            "Formula: cnt_resp_total - cnt_resp. "
        ),
        "cnt_resp_rndm": (
            "Meaning: Expected responders if randomly assigned. "
            "Critical: Baseline for comparison. Fatal if model only slightly above random."
            "Formula: cnt_resp_total * (total_responders / total_samples). "
        ),
        "cnt_resp_wiz": (
            "Meaning: Ideal responders if model were perfect. "
            "Critical: Must be ≥ cnt_resp. Fatal if NaN or actual far below."
            "Formula: allocate top responders directly into highest deciles. "
        ),
        "rate_resp": (
            "Meaning: Per-decile response rate (alias to decile_wise_response, decile_wise_gain). "
            "Critical: Measures decile quality. Early deciles should outperform later ones."
            "Formula: rate_resp = decile_wise_response = cnt_resp / cnt_resp_total. "
        ),
        "cum_resp_total": (
            "Meaning: Cumulative total samples. Critical: Tracks population coverage."
            "Formula: Σ cnt_resp_total(≤ current decile). "
        ),
        "cum_resp_total_pct": (
            "Meaning: % cumulative population. "
            "Critical: X-axis for lift/gain curves; check decile balance."
            "Formula: cum_resp_total / total_samples * 100. "
        ),
        "cum_resp": (
            "Meaning: Cumulative responders (alias to cumulative_gain) up to this decile so ML evaluation (how much `gain` vs random baseline). "
            "Critical: Should increase; max = total responders. Flat curve = weak model."
            "Formula: cumulative_gain = cumulative_response = Σ cnt_resp(≤ current decile) = cum_resp_pct vs cum_resp_total_pct. "
        ),
        "cum_resp_pct": (
            "Meaning: % cumulative responders = cum_resp / total_responders * 100. "
            "Critical: Wizard curve should be ≥ model; used in lift/gain charts."
            "Formula: cum_resp / total_responders * 100. "
        ),
        "cum_resp_non": (
            "Meaning: Cumulative non-responders. "
            "Critical: Used in KS statistic; early dominance is bad."
            "Formula: Σ cnt_resp_non(≤ current decile). "
        ),
        "cum_resp_non_pct": (
            "Meaning: % cumulative non-responders. "
            "Critical: Should differ from cum_resp_pct; almost equal = model fails."
            "Formula: cum_resp_non / total_nonresponders * 100. "
        ),
        "cum_resp_rndm": (
            "Meaning: Cumulative expected responders if randomly assigned. "
            "Critical: Baseline for cumulative lift. Fatal if model ≈ random curve."
            "Formula: Σ cnt_resp_rndm(≤ current decile). "
        ),
        "cum_resp_rndm_pct": (
            "Meaning: % cumulative random responders = cum_resp_rndm / total_responders * 100. "
            "Critical: Random baseline curve (diagonal). Always linear from (0,0) to (100,100). "
            "Fatal if model curve is near or below it."
            "Formula: cum_resp_rndm / total_responders * 100. "
        ),
        "cum_resp_wiz": (
            "Meaning: Cumulative ideal responders. "
            "Critical: Should always ≥ model; never NaN."
            "Formula: Σ cnt_resp_wiz(≤ current decile). "
        ),
        "cum_resp_wiz_pct": (
            "Meaning: % cumulative ideal responders. "
            "Critical: Wizard benchmark for lift/gain curves; gaps indicate model weakness."
            "Formula: cum_resp_wiz / total_responders * 100. "
        ),
        "KS": (
            "Meaning: KS Kolmogorov-Smirnov statistic. "
            "Range: 0-100 (percent scale) or 0-1 (fractional scale). "
            "Interpretation: "
            "- <20 → Poor discrimination (model barely better than random). "
            "- 20-40 → Fair. "
            "- 40-60 → Good. "
            "- ≥60 → Excellent. "
            "- ≥70 → Suspiciously high; likely overfitting or data leakage unless justified by very strong signal. "
            "Critical: Report max KS and check across train/validation/test. "
            "Fatal if KS is too low (<0.2) or unrealistically high (≥0.7 without strong justification)."
            "Formula: KS = max(cum_resp_pct - cum_resp_non_pct). "
        ),
        "cumulative_lift": (
            "Meaning: Cumulative lift = cum_resp_pct / cum_resp_total_pct. "
            "Critical: Shows model gain over random. Always cumulative. Fatal if <1 or <2 in top decile."
            "Formula: Lift@k = cum_resp_pct / cum_resp_total_pct. "
        ),
        "decile_wise_lift": (
            "Meaning: Decile-wise lift = cnt_resp / cnt_resp_rndm. "
            "Critical: Measures decile-level improvement vs random. Fatal if <1."
            "Formula: cnt_resp / cnt_resp_rndm. "
        ),
    }


def print_labels(
    as_json: bool = True,
    indent: int = 2,
) -> None:
    """
    Pretty-print the legend of decile table column names.

    ::

        [
            "decile",
            "prob_min",
            "prob_max",
            "prob_avg",
            "cnt_resp_total",
            "cnt_resp",
            "cnt_resp_non",
            "cnt_resp_rndm",
            "cnt_resp_wiz",
            "rate_resp",  # (alias to decile_wise_response, decile_wise_gain)
            "cum_resp_total",
            "cum_resp_total_pct",
            "cum_resp",  #  (alias to cumulative_gain)
            "cum_resp_pct",
            "cum_resp_non",
            "cum_resp_non_pct",
            "cum_resp_rndm",
            "cum_resp_rndm_pct",
            "cum_resp_wiz",
            "cum_resp_wiz_pct",
            "KS",
            "cumulative_lift",
            "decile_wise_lift",
        ]

    Parameters
    ----------
    as_json : bool, default=True
        If True, pretty-print as JSON. If False, use Python's pprint.
    indent : int, default=2
        Indentation for JSON formatting.

    See Also
    --------
    kdsplot : Given binary labels y_true (0/1) and probabilities y_score 1d array,
        compute/plot a decile table.

    References
    ----------
    .. [1] `https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L382
       <https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L382>`_

    .. [2] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L382

    Examples
    --------
    .. jupyter-execute::

        >>> import scikitplot.snsx as sp
        >>> sp.print_labels()
    """
    labels = get_label_info()

    try:
        if as_json:
            print(json.dumps(labels, indent=indent, ensure_ascii=False))  # noqa: T201
        else:
            pprint(labels, sort_dicts=False, width=100)  # noqa: T203
    except Exception as e:
        print("⚠️ Failed to print labels:", e)  # noqa: T201
        pprint(labels, sort_dicts=False, width=100)  # noqa: T203


# --------------------------------------------------------------------
# Core Plotter Class
# --------------------------------------------------------------------
class _DecilePlotter(VectorPlotter):
    """
    Internal seaborn-style plotter for decile-based metrics.

    This class uses VectorPlotter's data mapping and iteration utilities to
    compute decile tables and to draw lift/gain/ks plots per hue/facet subset.

    Key responsibilities:
    - Provide `_compute_decile_table_for_subset(sub_data)` for per-subset decile stats.
    - Provide high-level plot methods that iterate subsets and draw onto axes.
    """

    # minimal structural hints for wide vs flat data (keeps consistency with VectorPlotter)
    wide_structure: "ClassVar[dict[str, str]]" = {  # noqa: RUF012, UP037
        "x": "@index",
        "y": "@values",
        "hue": "@columns",
    }
    flat_structure: "ClassVar[dict[str, str]]" = {  # noqa: RUF012, UP037
        "x": "@index",
        "y": "@values",
    }

    def __init__(
        self,
        data=None,
        variables=None,
    ):
        variables = {} if variables is None else variables
        super().__init__(data=data, variables=variables)

    @property
    def univariate(self):
        """Return True if only x or y are used."""
        # TODO this could go down to core, but putting it here now.
        # We'd want to be conceptually clear that univariate only applies
        # to x/y and not to other semantics, which can exist.
        # We haven't settled on a good conceptual name for x/y.
        return bool({"x", "y"} - set(self.variables))

    @property
    def data_variable(self):
        """Return the variable with data for univariate plots."""
        # TODO This could also be in core, but it should have a better name.
        if not self.univariate:
            raise AttributeError("This is not a univariate plot")
        return {"x", "y"}.intersection(self.variables).pop()

    @property
    def has_xy_data(self):
        """Return True at least one of x or y is defined."""
        # TODO see above points about where this should go
        return bool({"x", "y"} & set(self.variables))

    def _add_legend(
        self,
        ax_obj,
        artist,
        fill,
        element,
        multiple,
        alpha,
        artist_kws,
        legend_kws,
    ):
        """Add artists that reflect semantic mappings and put then in a legend."""
        # TODO note that this doesn't handle numeric mappings like the relational plots
        handles = []
        labels = []
        for level in self._hue_map.levels:
            color = self._hue_map(level)

            kws = self._artist_kws(artist_kws, fill, element, multiple, color, alpha)

            # color gets added to the kws to workaround an issue with barplot's color
            # cycle integration but it causes problems in this context where we are
            # setting artist properties directly, so pop it off here
            if "facecolor" in kws:
                kws.pop("color", None)

            handles.append(artist(**kws))
            labels.append(level)

        if isinstance(ax_obj, mpl.axes.Axes):
            ax_obj.legend(handles, labels, title=self.variables["hue"], **legend_kws)
        else:  # i.e. a FacetGrid. TODO make this better
            legend_data = dict(zip(labels, handles))
            ax_obj.add_legend(
                legend_data,
                title=self.variables["hue"],
                label_order=self.var_levels["hue"],
                **legend_kws,
            )

    def _artist_kws(self, kws, fill, element, multiple, color, alpha):
        """Handle differences between artists in filled/unfilled plots."""
        kws = kws.copy()
        if fill:
            kws = normalize_kwargs(kws, mpl.collections.PolyCollection)
            kws.setdefault("facecolor", to_rgba(color, alpha))

            if element == "bars":
                # Make bar() interface with property cycle correctly
                # https://github.com/matplotlib/matplotlib/issues/19385
                kws["color"] = "none"

            if multiple in ["stack", "fill"] or element == "bars":
                kws.setdefault("edgecolor", mpl.rcParams["patch.edgecolor"])
            else:
                kws.setdefault("edgecolor", to_rgba(color, 1))
        elif element == "bars":
            kws["facecolor"] = "none"
            kws["edgecolor"] = to_rgba(color, alpha)
        else:
            kws["color"] = to_rgba(color, alpha)
        return kws

    def _cmap_from_color(self, color):
        """Return a sequential colormap given a color seed."""
        # Like so much else here, this is broadly useful, but keeping it
        # in this class to signify that I haven't thought overly hard about it...
        r, g, b, _ = to_rgba(color)
        h, s, _ = husl.rgb_to_husl(r, g, b)
        xx = np.linspace(-1, 1, int(1.15 * 256))[:256]
        ramp = np.zeros((256, 3))
        ramp[:, 0] = h
        ramp[:, 1] = s * np.cos(xx)
        ramp[:, 2] = np.linspace(35, 80, 256)
        colors = np.clip([husl.husl_to_rgb(*hsl) for hsl in ramp], 0, 1)
        return mpl.colors.ListedColormap(colors[::-1])

    def _make_legend_proxies(self, artists, labels):
        """
        Convert PolyCollections from fill_between into Line2D proxies so legends remain clean.
        """
        # If artists are PolyCollections (from fill_between), create Line2D proxies
        handles = []
        for art in artists:
            if isinstance(art, mpl.collections.PolyCollection):
                # proxy line for legend using first artist's color and linestyle
                # color_proxy = art.get_facecolor()[0] if len(art.get_facecolor()) else None
                face = art.get_facecolor()
                c = face[0] if len(face) > 0 else None  # (0, 0, 0, 1)
                proxy = mpl.lines.Line2D([], [], color=c, linewidth=1.5)
                handles.append(proxy)
            else:
                handles.append(art)
        # filter out None labels
        labels = [lbl if lbl is not None else "" for lbl in labels]
        return handles, labels

    # ------------------------
    # Helpers - data extraction
    # ------------------------
    def _prepare_subset(
        self,
        sub_data: pd.DataFrame,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":  # noqa: UP037
        """
        Extract y_true (binary labels) and y_score (probabilities) arrays from a subset DataFrame.

        - Expects that VectorPlotter has already standardized columns to "x" and "y"
          in `sub_data` when `from_comp_data=True` is used in iter_data (this matches
          the approach used in seaborn's VectorPlotter pattern).
        - Enforces types: y_true -> int (0/1), y_score -> float.

        Returns
        -------
        y_true : ndarray of shape (n_samples,)
        y_score : ndarray of shape (n_samples,)
        weights : ndarray or None
        """
        # Extract the data points from this sub set
        # compute PR curve: y_true (x_col), y_score (y_col)
        # Map seaborn-style x/y/hue to our data
        # x_col, y_col = "x", "y"
        # y_true = sub_data[x_col].astype(int)
        # y_score = sub_data[y_col]

        # Drop rows missing either true labels or predicted scores
        # df.dropna(axis=1, how="all")  # Drop completely empty columns

        # Drop rows missing either true labels or predicted scores
        sub = sub_data.dropna(subset=["x", "y"])
        if sub.empty:
            return None, None, None

        # Coerce true labels to integers (0/1 for binary classification)
        try:
            # extract array
            y_true = np.asarray(sub["x"]).astype(int)
        except Exception as e:
            raise ValueError(f"Cannot convert x to integer labels: {e}") from e

        # Scores must be float
        try:
            y_score = np.asarray(sub["y"], dtype=float)
        except Exception as e:
            raise ValueError(f"Cannot convert y to float scores: {e}") from e

        # Extract weights if present
        weights = sub["weights"].to_numpy() if "weights" in sub.columns else None

        return y_true, y_score, weights

    # ---------------------------
    # Plot helpers
    # ---------------------------
    def _plot_decile_wise_lift(self, dt: pd.DataFrame, legend, ax, **kws):
        n_deciles = dt["decile"].max()
        # Save artist and label (include statement if legend requested)
        artists, labels = [], []
        (artist,) = ax.plot(
            dt["decile"],
            dt["decile_wise_lift"],
            marker="o",
            label="Model (Discriminative Power)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Discriminative Power)")
        (artist,) = ax.plot(
            [1, n_deciles],
            [1, 1],
            "k--",
            label="Random Baseline (Lift=1)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Random Baseline (Lift=1)")
        ax.set(
            # "Population %\n(sorted by predicted score, descending; decile/percentile)"
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Decile-wise Lift Ratio\n(Lift = decile response rate / overall response rate)"
            ylabel="Decile-wise Lift Ratio",
            title="Decile-wise Lift Curve",
        )
        ax.grid(True)

        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

    def _plot_cumulative_lift(self, dt: pd.DataFrame, legend, ax, **kws):
        n_deciles = dt["decile"].max()
        artists, labels = [], []
        # Plot curve: fill area under curve if requested, otherwise plot line
        # if kws.pop("fill", None):
        #     artist = ax.fill_between(y_true, 0.0, y_score, **kws)
        #     # fill_between returns PolyCollection; put a proxy Line2D for legend if needed
        #     # but store the PolyCollection as artist for potential manipulation
        # else:
        #     # draw standard PR line plot
        #     (artist,) = ax.plot(y_true, y_score, **kws)
        # Model curve
        (artist,) = ax.plot(
            dt["decile"],
            dt["cumulative_lift"],
            marker="o",
            label="Model (Discriminative Power)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Discriminative Power)")
        # Random baseline: always flat at 1
        # plt.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
        (artist,) = ax.plot(
            [1, n_deciles],
            [1, 1],
            "k--",
            label="Random Baseline (Lift=1)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Random Baseline (Lift=1)")
        # Axis and title (parallels cumulative response / decile response plots)
        ax.set(
            # "Population %\n(sorted by predicted score, descending; decile/percentile)"
            # "Population %\n(sorted by score (desc), decile/percentile)"
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Cumulative Lift Ratio\n(Lift@k = responders% up to k / population% up to k)"
            ylabel="Cumulative Lift Ratio\n(Lift@k, top k%)",
            # "Cumulative Lift Curve\n(Model vs Random Baseline = 1)"
            title="Cumulative Lift Curve",
        )
        ax.grid(True)
        # plt.xlabel("Decile")  # Population Percentage
        # plt.ylabel("Lift")
        # plt.title("Lift Curve")

        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

    def _plot_decile_wise_gain(self, dt: pd.DataFrame, legend, ax, **kws):
        n_deciles = dt["decile"].max()
        artists, labels = [], []

        # Model decile response rate
        (artist,) = ax.plot(
            dt["decile"],
            dt["rate_resp"],  # Ensure convert to %
            marker="o",
            label="Model (Discriminative Power)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Discriminative Power)")

        # Random baseline: flat at overall response rate
        overall_rate = (dt["cnt_resp"].sum() / dt["cnt_resp_total"].sum()) * 100.0
        (artist,) = ax.plot(
            [1, n_deciles],
            [overall_rate, overall_rate],
            "k--",
            # "Random Baseline (Response = flat line (overall average))"
            label=f"Random Baseline (Response={overall_rate:.2f}% (overall average response rate))",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist)
        labels.append(
            f"Random Baseline (Response={overall_rate:.2f}% (overall average response rate))"
        )

        ax.set(
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Response Rate per Decile"
            ylabel="Decile Response Rate (%)\n(Per-decile performance)",
            title="Decile-wise Gain/Response Curve",
        )
        ax.grid(True)

        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

    def _plot_cumulative_gain(self, dt: pd.DataFrame, legend, ax, **kws):
        n_deciles = dt["decile"].max()
        artists, labels = [], []
        # Model curve = your actual model → discriminative power lies between random and wizard.
        # Wizard baseline = perfect model, responders ranked first → theoretical maximum performance.
        # Random baseline = no model, responders distributed evenly → Lift = 1 / diagonal line.
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cum_resp_pct"]),
            dt["cum_resp_pct"],
            marker="o",
            label="Model (Discriminative Power)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Discriminative Power)")
        # Wizard model = a hypothetically perfect classifier.
        # It always ranks all responders at the very top of the population
        # It gives the theoretical upper bound of performance.
        # If your model curve is close to wizard → very strong.
        # If your model is close to random → useless.
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cum_resp_wiz_pct"]),
            dt["cum_resp_wiz_pct"],
            "c--",
            # https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html#marker-reference
            # ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
            marker="2",
            # alpha=1,
            label="Wizard Baseline (perfect model)",
            # zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Wizard Baseline (perfect model)")
        (artist,) = ax.plot(
            [1, n_deciles],
            [1, 100],
            "k--",
            label="Random Baseline (Responders=diagonal)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Random Baseline (Responders=diagonal)")
        ax.set(
            # "Population %\n(sorted by predicted score, descending; decile/percentile)"
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Cumulative Responders %\n(captured vs total responders)"
            ylabel="Cumulative Responders %",
            # "Cumulative Gain Curve\n(Model vs Random Baseline = diagonal)"
            title="Cumulative Gain Curve",
        )
        ax.grid(True)

        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

    def _plot_cumulative_response(self, dt: pd.DataFrame, legend, ax, **kws):
        n_deciles = dt["decile"].max()
        artists, labels = [], []

        # Model cumulative response %
        (artist,) = ax.plot(
            dt["cum_resp_total_pct"],
            dt["cum_resp_pct"],
            marker="o",
            label="Model (Discriminative Power)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Discriminative Power)")

        # Wizard (perfect model): straight line, captures all responders as fast as possible
        (artist,) = ax.plot(
            dt["cum_resp_total_pct"],
            dt["cum_resp_wiz_pct"],
            "c--",
            marker="2",
            label="Wizard Baseline (Perfect Model)",
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Wizard Baseline (Perfect Model)")

        # Random baseline: diagonal line (responders captured ~ proportional to population)
        (artist,) = ax.plot(
            # [1, n_deciles],
            # [1, 100],
            dt["cum_resp_total_pct"],
            dt["cum_resp_rndm_pct"],
            "k--",
            label="Random Baseline (Responders=diagonal)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Random Baseline (Responders=diagonal)")

        ax.set(
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            ylabel="Cumulative Responders %\n(Capture rate up to decile)",
            title="Cumulative Response Curve",
        )
        ax.grid(True)

        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

    def _plot_ks_statistic(self, dt: pd.DataFrame, legend, ax, **kws):
        n_deciles = dt["decile"].max()
        # Save artist and label (include AUC if legend requested)
        artists, labels = [], []
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cum_resp_pct"]),
            dt["cum_resp_pct"],
            marker="o",
            label="Responders",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Responders")
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cum_resp_non_pct"]),
            dt["cum_resp_non_pct"],
            marker="o",
            c="darkorange",
            label="Non-Responders",
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Non-Responders")
        ks_max = dt["KS"].max()
        ks_decile = dt.loc[
            dt["KS"].idxmax(), "decile"
        ]  # ksdcl = dt[ksmx == dt.KS].decile.values
        # ax.axvline(ks_decile, linestyle="--", color="g",
        #            label=f"KS={ks_max:.2f} @ decile {ks_decile}")
        (artist,) = ax.plot(
            [ks_decile, ks_decile],
            [
                dt[ks_max == dt.KS].cum_resp_pct.to_numpy(),
                dt[ks_max == dt.KS].cum_resp_non_pct.to_numpy(),
            ],
            "g--",
            marker="o",
            # label="KS Statistic: " + str(ks_max) + " at decile " + str(list(ks_decile)[0]),
            label=f"KS={ks_max:.2f} @ decile {ks_decile}",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append(f"KS={ks_max:.4f} @ decile {ks_decile}")
        (artist,) = ax.plot(
            [1, n_deciles],
            [1, 100],
            "k--",
            label="Random Baseline (Responders=diagonal)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Random Baseline (Responders=diagonal)")
        ax.set(
            # "Population %\n(sorted by predicted score, descending; decile/percentile)"
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Cumulative Distribution %"
            ylabel="Cumulative Responders %",
            # "KS Statistic Curve\n(Difference between responders and non-responders)"
            title="KS Statistic Curve",
        )
        ax.grid(True)

        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

    def _plot_report(self, dt: pd.DataFrame, legend, ax, **kws):
        """2x2 dashboard with all metrics."""
        # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        # Get the parent figure from the given Axes
        fig = ax.figure
        # Replace single Axes with a 2x2 grid inside the same figure
        fig.clf()  # clear any existing axes
        axes = fig.subplots(2, 2)

        # Draw plots
        self._plot_cumulative_lift(dt, legend, axes[0, 0], **kws)
        self._plot_decile_wise_lift(dt, legend, axes[0, 1], **kws)
        self._plot_cumulative_gain(dt, legend, axes[1, 0], **kws)
        self._plot_ks_statistic(dt, legend, axes[1, 1], **kws)
        fig.tight_layout()

    # ------------------------
    # User-facing Core computation Decile assignment & table
    # ------------------------
    def compute_decile_table(  # noqa: PLR0912
        self,
        kind: "Literal['df', 'cumulative_lift', 'decile_wise_lift', 'cumulative_gain', 'decile_wise_gain', 'cumulative_response', 'ks_statistic', 'report'] | None" = None,  # noqa: UP037
        fill=None,
        color=None,
        legend=None,
        n_deciles: int = 10,
        round_digits: "int | None" = None,  # noqa: UP037
        verbose=True,
        **plot_kws,
    ) -> "pd.DataFrame | None":  # noqa: UP037
        """
        Given labels y_true (0/1) and probabilities y_score arrays, compute a decile table.

        The function sorts observations by descending score, assigns decile index
        (1..n_deciles) using pandas qcut on the rank/index to ensure near-equal bins,
        and computes standard decile-level (features)::

            [
                "decile",
                "prob_min",
                "prob_max",
                "prob_avg",
                "cnt_resp_total",
                "cnt_resp",
                "cnt_resp_non",
                "cnt_resp_rndm",
                "cnt_resp_wiz",
                "rate_resp",  # (alias to decile_wise_response, decile_wise_gain)
                "cum_resp_total",
                "cum_resp_total_pct",
                "cum_resp",  #  (alias to cumulative_gain)
                "cum_resp_pct",
                "cum_resp_non",
                "cum_resp_non_pct",
                "cum_resp_rndm",
                "cum_resp_rndm_pct",
                "cum_resp_wiz",
                "cum_resp_wiz_pct",
                "KS",
                "cumulative_lift",
                "decile_wise_lift",
            ]

        Returns DataFrame indexed by decile (sorted ascending).
        """
        # Interpret plotting options
        label = plot_kws.get("label", "")
        alpha = plot_kws.get("alpha", 1.0)
        linestyle = plot_kws.get("linestyle", plot_kws.get("ls"))
        marker = plot_kws.get("marker", plot_kws.get("m"))
        chance = bool(plot_kws.pop("chance", None))

        # Save artist and label (include AUC if legend requested)
        # artists, labels = [], []
        dfs = []

        # Iterate through subsets (handles hue + facets if used)
        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):
            if sub_data.empty:
                continue

            # get axis for this subset
            ax = self._get_axes(sub_vars)

            # Label and color handling/determine
            key = tuple(sub_vars.items())
            if "hue" in self.variables:
                level = sub_vars["hue"]
                sub_color = self._hue_map(level)
                label_base = str(level)
            else:
                # # Default color
                # sub_color = color or self._hue_map(sub_vars["hue"], "color")
                sub_color = color or _default_color(ax.plot, None, None, {})
                label_base = ""

            # prepare arrays
            try:
                y_true, y_score, _sw = self._prepare_subset(sub_data)  # noqa: RUF059
            except ValueError as e:
                warnings.warn(
                    f"Skipping subset {sub_vars} due to data error: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if y_true is None:
                continue

            # binary vs multiclass detection
            classes = np.unique(y_true)
            multiclass = len(classes) > 2  # noqa: PLR2004
            # Compute PR curve
            if multiclass:
                # TODO:x or y → Seaborn expects 1D vectors (arrays, Series, lists), not a 2D DataFrame
                # so multiclass not supported by DataFrame or Seaborn
                warnings.warn(
                    f"Cannot label_binarize multiclass labels: {''}",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # binary case (most common)
                # Defensive copy via DataFrame to use pandas utilities
                dt = pd.DataFrame(
                    {"y_true": y_true, "y_score": y_score},
                )  # .dropna()
                # Ranked group (1=highest probability)
                # Ensure sorted descending by model score. Fatal if not.
                # Sort probabilities by descending score so highest scores are in decile 1
                dt = dt.sort_values(by="y_score", ascending=False).reset_index(
                    drop=True
                )
                # Assign deciles (pop_pct) by index using qcut behavior. Use ranks (index) to avoid duplicate-edge errors.
                # qcut on the index will create near-equal sized bins; duplicates="drop" avoids failures with many ties.
                try:
                    # dt["decile"] = pd.qcut(dt.index, q=n_deciles, labels=False, duplicates="drop") + 1
                    # dt['decile'] = pd.qcut(dt['y_score'], n_deciles, labels=list(np.arange(n_deciles, 0, -1)))  # ValueError: Bin edges must be unique
                    dt["decile"] = np.linspace(
                        1, n_deciles + 1, len(dt), False, dtype=int
                    )
                except ValueError:
                    # Fallback: if qcut fails (e.g., too few unique values), use linspace index-based bins
                    n = len(dt)
                    bins = np.linspace(0, n, n_deciles + 1)
                    dt["decile"] = (
                        pd.cut(dt.index, bins=bins, labels=False, include_lowest=True)
                        + 1
                    )
                # Calculate additional columns Wizard by "y_true"
                tmp = dt[["y_true"]].sort_values("y_true", ascending=False)
                tmp["decile"] = np.linspace(
                    1, n_deciles + 1, len(tmp), False, dtype=int
                )
                tmp = tmp.groupby(
                    "decile",
                    as_index=True,
                    group_keys=False,  # Changed in version 2.0.0: group_keys now defaults to True.
                )
                wiz = tmp["y_true"].sum().reset_index(drop=True)
                # Calculate additional columns rndm
                rndm = np.sum(dt["y_true"]) / n_deciles  # scaler constant

                # Aggregate per decile as decile table
                agg = (
                    # Changed in version 2.0.0: group_keys now defaults to True.
                    dt.groupby("decile", group_keys=False)
                    .apply(
                        lambda x: pd.Series(
                            [
                                np.min(x["y_score"]),
                                np.max(x["y_score"]),
                                np.mean(x["y_score"]),
                                np.size(x["y_score"]),
                                np.sum(x["y_true"]),
                                np.size(x["y_true"][x["y_true"] == 0]),
                                # rndm,  # noqa: B023
                            ],
                            index=(
                                [
                                    "prob_min",
                                    "prob_max",
                                    "prob_avg",
                                    "cnt_resp_total",
                                    "cnt_resp",
                                    "cnt_resp_non",
                                    # "cnt_resp_rndm",
                                ]
                            ),
                        ),
                        **groupby_apply_include_groups(
                            False
                        ),  # Deprecated since version 2.2.0: Setting include_groups to True is deprecated.
                    )
                    .reset_index()
                )
                # agg = agg.sort_values(by='decile', ascending=False).reset_index(drop=True)
                # agg = agg.sort_index()  # decile ascending 1..k

                # Response compute non-resp
                # agg["cnt_nonresp"] = agg["cnt_total"] - agg["cnt_resp"]

                # Add random/wizard responders for benchmarking
                agg["cnt_resp_rndm"] = rndm  # scaler constant
                agg["cnt_resp_wiz"] = wiz  # by y_true

                # Direct decile quality metric
                # decile-wise response (per decile rate)
                agg["rate_resp"] = (agg["cnt_resp"] / agg["cnt_resp_total"]) * 100.0

                # cumulative columns and percentages: avoid division by zero
                agg["cum_resp_total"] = np.cumsum(agg["cnt_resp_total"])
                agg["cum_resp_total_pct"] = (
                    agg["cum_resp_total"] / np.sum(agg["cnt_resp_total"])
                ) * 100.0
                # cumulative gain/response as % of total responders in population
                agg["cum_resp"] = np.cumsum(agg["cnt_resp"])
                agg["cum_resp_pct"] = (
                    agg["cum_resp"] / np.sum(agg["cnt_resp"])
                ) * 100.0
                agg["cum_resp_non"] = np.cumsum(agg["cnt_resp_non"])
                agg["cum_resp_non_pct"] = (
                    agg["cum_resp_non"] / np.sum(agg["cnt_resp_non"])
                ) * 100.0
                agg["cum_resp_rndm"] = np.cumsum(agg["cnt_resp_rndm"])
                agg["cum_resp_rndm_pct"] = (
                    agg["cum_resp_rndm"] / np.sum(agg["cnt_resp_rndm"])
                ) * 100.0
                agg["cum_resp_wiz"] = np.cumsum(agg["cnt_resp_wiz"])
                agg["cum_resp_wiz_pct"] = (
                    agg["cum_resp_wiz"] / np.sum(agg["cnt_resp_wiz"])
                ) * 100.0

                # KS and cumulative_lift
                agg["KS"] = (
                    agg["cum_resp_pct"] - agg["cum_resp_non_pct"]
                )  # .round(round_digits)
                # avoid division by zero in lift
                agg["cumulative_lift"] = agg["cum_resp_pct"] / agg["cum_resp_total_pct"]
                # This is mathematically equivalent to the formula below
                # (just expressed using expected random responders instead of overall rate).
                # overall_rate = agg["cnt_resp"].sum() / agg["cnt_resp_total"].sum()
                # agg["decile_wise_lift"] = (agg["cnt_resp"] / agg["cnt_resp_total"]) / overall_rate
                agg["decile_wise_lift"] = agg["cnt_resp"] / agg["cnt_resp_rndm"]
                # agg = agg.fillna(0)  # replace NaNs from division by zeros

                if "hue" in self.variables:
                    level = sub_vars["hue"]
                    agg = agg.assign(hue=level)
                dfs.append(agg)

                # plot line and optional fill, Use element "line" for curve plotting
                artist_kws = self._artist_kws(
                    plot_kws, fill, "line", "layer", sub_color, alpha
                )
                # Merge user requested line style/marker if present
                if linestyle is not None:
                    artist_kws["linestyle"] = linestyle
                if marker is not None:
                    artist_kws["marker"] = marker
                if fill is not None:
                    # fill_between expects x sorted ascending; recall is monotonic increasing
                    artist_kws.setdefault(
                        "alpha",
                        artist_kws.get("alpha", alpha),
                    )

                # plot line and optional fill, Use element "line" for curve plotting
                _plot = getattr(self, f"_plot_{kind}", None)
                if _plot:
                    _plot(agg, legend, ax, **artist_kws)

        agg = pd.concat(dfs, ignore_index=True)
        return agg.round(round_digits) if round_digits else agg


# -----------------------------------------------------------------------------
# Public API functions (wrappers)
# -----------------------------------------------------------------------------
def kdsplot(  # noqa: D417
    data: "pd.DataFrame | None" = None,  # noqa: UP037
    *,
    x: "str | pd.Series | np.ndarray | None" = None,  # noqa: UP037
    y: "str | pd.Series | np.ndarray | None" = None,  # noqa: UP037
    kind: "Literal['df', 'cumulative_lift', 'decile_wise_lift', 'cumulative_gain', 'decile_wise_gain', 'cumulative_response', 'ks_statistic', 'report'] | None" = None,  # noqa: UP037
    n_deciles: int = 10,
    round_digits: "int | None" = None,  # noqa: UP037
    hue: "str | pd.Series | np.ndarray | None" = None,  # noqa: UP037
    weights=None,
    verbose: bool = False,
    # appearance parameters
    fill=False,
    # smoothing
    line_kws=None,
    color=None,
    legend=True,
    ax=None,
    **kwargs,
) -> "pd.DataFrame | mpl.axes.Axes":  # mpl.figure.Figure  # noqa: UP037
    """
    Given binary labels y_true (0/1) and probabilities y_score 1d array, compute/plot a decile table.

    The function sorts observations by descending score, assigns decile index
    (1..n_deciles) using pandas qcut on the rank/index to ensure near-equal bins,
    and computes standard decile-level stats (features)::

        [
            "decile",
            "prob_min",
            "prob_max",
            "prob_avg",
            "cnt_resp_total",
            "cnt_resp",
            "cnt_resp_non",
            "cnt_resp_rndm",
            "cnt_resp_wiz",
            "rate_resp",  # (alias to decile_wise_response, decile_wise_gain)
            "cum_resp_total",
            "cum_resp_total_pct",
            "cum_resp",  #  (alias to cumulative_gain)
            "cum_resp_pct",
            "cum_resp_non",
            "cum_resp_non_pct",
            "cum_resp_rndm",
            "cum_resp_rndm_pct",
            "cum_resp_wiz",
            "cum_resp_wiz_pct",
            "KS",
            "cumulative_lift",
            "decile_wise_lift",
        ]

    Parameters
    ----------
    data : pandas.DataFrame | None
    x : str | pd.Series | np.ndarray | None
        Ground truth (correct/actual) target values.
    y : str | pd.Series | np.ndarray | None
        Prediction probabilities for target class returned by a classifier/algorithm.
    n_deciles : int, optional, default=10
        The number of partitions for creating the table. Defaults to 10 for deciles.
    round_digits : int, optional, default=None
        The decimal precision for the result.
        return df.round(round_digits)
    verbose : bool, optional, default=False
        If True, prints a legend for the abbreviations of decile table column names.

    Returns
    -------
    pandas.DataFrame | matplotlib.axes.Axes
        The dataframe (decile-table) with the indexed by deciles (sorted ascending)
        and related information (decile-level metrics).
        If hue/facet semantics were used, the returned table will include
        extra columns for those keys (e.g., 'hue').
    """
    _check_argument(
        "kind",
        [
            "df",
            "cumulative_lift",
            "decile_wise_lift",
            "cumulative_gain",
            "decile_wise_gain",
            "cumulative_response",
            "ks_statistic",
            "report",
        ],
        kind,
    )

    # Build the VectorPlotter and attach data/variables in seaborn style
    p = _DecilePlotter(
        data=data,
        variables={"x": x, "y": y, "hue": hue, "weights": weights},
    )
    # map hue palette/order if needed - we only need internal mapping for iter_data to work
    p.map_hue()  # default mapping; callers can later set palette via more advanced API if needed

    # ax = plt.gca() if ax is None else ax
    if ax is None:
        ax = plt.gca()

    # Now attach the axes object to the plotter object
    p._attach(ax)

    method = ax.fill_between if fill else ax.plot
    color = _default_color(method, hue, color, kwargs)
    # color = kwargs.pop("color", kwargs.pop("c", None))
    # if color is None and hue is None:
    #     color = "C0"
    # XXX else warn if hue is not None?

    # kwargs["color"] = color
    # fill = bool(kwargs.pop("fill", False))
    # kwargs.setdefault("color", color)
    # kwargs.setdefault("fill", fill)
    # kwargs.setdefault("legend", legend)

    # Check for a specification that lacks x/y data and return early
    if not p.has_xy_data:
        return ax

    # compute decile table and/or Draw the plots
    dt_kws = kwargs.copy()
    # pr_kws["color"] = color
    # _assign_default_kwargs(pr_kws, p.plot_prauc, prplot)
    dt = p.compute_decile_table(
        kind=kind,
        n_deciles=n_deciles,
        round_digits=round_digits,
        verbose=verbose,
        color=color,
        legend=legend,
        fill=fill,
        **dt_kws,
    )
    if verbose is True:
        print_labels()
    if kind == "df":
        plt.gca().remove()
        plt.close()
        return dt
    if kind == "report":
        try:
            from IPython.display import display  # noqa: PLC0415

            display(dt)
        except Exception:
            print(dt)  # noqa: T201
    return ax
