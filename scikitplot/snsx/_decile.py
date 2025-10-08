# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

"""
Decile-based [2]_ model evaluation module (Lift, Gains, KS statistics).

The :mod:`~scikitplot.snsx._decile` module includes plots for machine learning
evaluation decile analysis e.g. Gain, Lift and Decile charts, etc.

In descriptive statistics, a decile is any of the nine values that divide the sorted data
into ten equal parts, so that each part represents 1/10 of the sample or population.
A decile is one possible form of a quantile; others include the quartile and percentile.
A decile rank arranges the data in order from lowest to highest and
is done on a scale of one to ten where each successive number corresponds to
an increase of 10 percentage points. See [2]_ for model details.

References
----------
.. [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py

.. [2] `Wikipedia contributors. (2024).
   "Decile"
   Wikipedia. https://en.wikipedia.org/wiki/Decile
   <https://en.wikipedia.org/wiki/Decile>`_
"""

# Seaborn-style decile analysis (Lift / Gain / KS) rewritten to use VectorPlotter.
#
# Design principles:
# - Accept `data=..., x=..., y=..., hue=...` like seaborn.
# - Compute decile-level statistics per hue/facet subset using VectorPlotter.iter_data.
# - Return pandas DataFrame for decile_table and Axes for plotting functions.
#
# This module provides:
# - decile_table(...) : compute decile-level statistics
# - plot_cumulative_lift(...) : cumulative lift curve
# - plot_decile_wise_lift(...) : lift per decile
# - plot_cumulative_gain(...) : cumulative gain curve
# - plot_ks_statistic(...) : KS statistic and helper markers
# - report(...) : convenience function producing decile table + subplots

# code that needs to be compatible with both Python 2 and Python 3
# from __future__ import (
#     absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
#     division,  # Changes the division operator `/` to always perform true division.
#     print_function,  # Treats `print` as a function, consistent with Python 3 syntax.
#     unicode_literals,  # Makes all string literals Unicode by default, similar to Python 3.
# )
from __future__ import annotations  # Allows string type hints

import json
import warnings
from pprint import pprint
from typing import ClassVar, Literal

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.typing as mpt  # Typing support  # noqa: F401
import numpy as np
import numpy.typing as npt  # Typing support  # noqa: F401
import pandas as pd
import pandas._typing as pdt  # Typing support  # noqa: F401
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from sklearn.utils.multiclass import unique_labels  # noqa: F401
from sklearn.utils.validation import check_array, check_consistent_length

try:
    from seaborn._base import VectorPlotter
    from seaborn._compat import groupby_apply_include_groups  # noqa: F401
    from seaborn._docstrings import (
        DocstringComponents,
        _core_docs,
    )
    from seaborn._statistics import ECDF, KDE, Histogram
    from seaborn._stats.counting import Hist  # noqa: F401
    from seaborn.axisgrid import _facet_docs
    from seaborn.external import husl
    from seaborn.utils import _check_argument, _default_color
except:
    from ..externals._seaborn._base import VectorPlotter
    from ..externals._seaborn._compat import groupby_apply_include_groups  # noqa: F401
    from ..externals._seaborn._docstrings import (
        DocstringComponents,
        _core_docs,
    )
    from ..externals._seaborn._statistics import ECDF, KDE, Histogram
    from ..externals._seaborn._stats.counting import Hist  # noqa: F401
    from ..externals._seaborn.axisgrid import _facet_docs
    from ..externals._seaborn.external import husl
    from ..externals._seaborn.utils import (
        _check_argument,
        _default_color,
    )


# Define __all__ to specify the public interface of the module,
# not required all module instances (default)
__all__ = [
    "decileplot",
    "print_labels",
]


# ==================================================================================== #
# Module documentation
# ==================================================================================== #

_dist_params = dict(  # noqa: C408
    multiple="""
multiple : {{"layer", "stack", "fill"}}
    Method for drawing multiple elements when semantic mapping creates subsets.
    Only relevant with univariate data.
    """,
    log_scale="""
log_scale : bool or number, or pair of bools or numbers
    Set axis scale(s) to log. A single value sets the data axis for any numeric
    axes in the plot. A pair of values sets each axis independently.
    Numeric values are interpreted as the desired base (default 10).
    When `None` or `False`, seaborn defers to the existing Axes scale.
    """,
    legend="""
legend : bool
    If False, suppress the legend for semantic variables.
    """,
    cbar="""
cbar : bool
    If True, add a colorbar to annotate the color mapping in a bivariate plot.
    Note: Does not currently support plots with a ``hue`` variable well.
    """,
    cbar_ax="""
cbar_ax : :class:`matplotlib.axes.Axes`
    Pre-existing axes for the colorbar.
    """,
    cbar_kws="""
cbar_kws : dict
    Additional parameters passed to :meth:`matplotlib.figure.Figure.colorbar`.
    """,
)

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    facets=DocstringComponents(_facet_docs),
    dist=DocstringComponents(_dist_params),
    kde=DocstringComponents.from_function_params(KDE.__init__),
    hist=DocstringComponents.from_function_params(Histogram.__init__),
    ecdf=DocstringComponents.from_function_params(ECDF.__init__),
)


# ==================================================================================== #
# Internal API
# ==================================================================================== #


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
            "Meaning: Ranked group based on predicted probabilities (1 = highest probability). "
            "Critical: Ensure data is sorted descending by model score; top deciles should capture the majority of positives. "
            "Formula: Assign samples to k quantiles (e.g., 10 deciles) based on model score."
        ),
        "prob_min": (
            "Meaning: Minimum predicted probability within the decile. "
            "Critical: Indicates model calibration; values too close to prob_max suggest poor separation. "
            "Formula: min(score in decile)."
        ),
        "prob_max": (
            "Meaning: Maximum predicted probability within the decile. "
            "Critical: Checks separation; overlap with lower deciles indicates poor discrimination. "
            "Formula: max(score in decile)."
        ),
        "prob_avg": (
            "Meaning: Average predicted probability within the decile. "
            "Critical: Useful for calibration checks; should decrease monotonically across deciles. "
            "Formula: mean(score in decile)."
        ),
        "cnt_resp_true": (
            "Meaning: Actual positives/responders in the decile. "
            "Critical: Should never exceed cnt_resp_wiz_true; flat counts across deciles indicate a weak or non-discriminative model. "
            "Formula: sum(y_true = 1 in decile)."
        ),
        "cnt_resp_false": (
            "Meaning: Actual negatives/non-responders in the decile. "
            "Critical: Used in KS/statistical calculations; too many negatives in top deciles is a warning. "
            "Formula: cnt_resp_total - cnt_resp_true."
        ),
        "cnt_resp_total": (
            "Meaning: Total samples in the decile (positives + negatives). "
            "Critical: Denominator for rate_resp and cumulative % calculations; decile imbalance can distort lift/gain. "
            "Formula: count(samples in decile)."
        ),
        "cnt_resp_rndm_true": (
            "Meaning: Expected positives in the decile under a random model. "
            "Critical: Baseline for lift/gain comparison; fatal if model barely exceeds random. "
            "Formula: total_positives / n_deciles."
        ),
        "cnt_resp_wiz_true": (
            "Meaning: Ideal/maximum possible positives if the model were perfect. "
            "Critical: Must always be ≥ cnt_resp_true; NaN or extremely low values indicate data issues. "
            "Formula: allocate top positives directly to highest scoring deciles."
        ),
        "rate_resp": (
            "Meaning: Decile-level response rate (alias: decile_wise_response, decile_wise_gain). "
            "Critical: Measures decile quality; early deciles should outperform later ones. "
            "Formula: rate_resp = cnt_resp_true / cnt_resp_total."
        ),
        "overall_rate": (
            "Meaning: Overall response rate across the dataset; serves as the baseline probability of a positive. "
            "Critical: Used as the denominator in decile-wise lift; essential to assess improvement vs random. "
            "Formula: overall_rate = sum(cnt_resp_true) / sum(cnt_resp_total) (fraction or %)."
        ),
        "cum_resp_true": (
            "Meaning: Cumulative number of positives captured up to the current decile (alias: cumulative_gain). "
            "Critical: Should increase monotonically; maximum = total responders. Flat curve indicates weak model. "
            "Formula: Σ cnt_resp_true (≤ current decile)."
        ),
        "cum_resp_true_pct": (
            "Meaning: Cumulative % of positives captured = cum_resp_true / total_responders * 100. "
            "Critical: Used for lift/gain curves; should always be ≥ model baseline. "
            "Formula: cum_resp_true / total_responders * 100."
        ),
        "cum_resp_false": (
            "Meaning: Cumulative number of negatives captured up to the current decile. "
            "Critical: Used for KS/statistical calculations; dominance in early deciles is undesirable. "
            "Formula: Σ cnt_resp_false (≤ current decile)."
        ),
        "cum_resp_false_pct": (
            "Meaning: Cumulative % of negatives captured = cum_resp_false / total_nonresponders * 100. "
            "Critical: Should differ from cum_resp_true_pct; nearly equal curves indicate model failure. "
            "Formula: cum_resp_false / total_nonresponders * 100."
        ),
        "cum_resp_total": (
            "Meaning: Cumulative total samples up to the current decile. "
            "Critical: Tracks population coverage for lift/gain charts. "
            "Formula: Σ cnt_resp_total (≤ current decile)."
        ),
        "cum_resp_total_pct": (
            "Meaning: Cumulative % of total population covered. "
            "Critical: X-axis for lift/gain curves; check decile balance. "
            "Formula: cum_resp_total / total_samples * 100."
        ),
        "cum_resp_rndm_true": (
            "Meaning: Cumulative expected positives if randomly assigned. "
            "Critical: Baseline for cumulative lift; fatal if model ≈ random curve. "
            "Formula: Σ cnt_resp_rndm_true (≤ current decile)."
        ),
        "cum_resp_rndm_true_pct": (
            "Meaning: Cumulative % of expected positives under random = cum_resp_rndm_true / total_responders * 100. "
            "Critical: Baseline curve is linear from (0,0) to (100,100); model curve must exceed this. "
            "Formula: cum_resp_rndm_true / total_responders * 100."
        ),
        "cum_resp_wiz_true": (
            "Meaning: Cumulative ideal/maximum possible positives. "
            "Critical: Must always be ≥ model values; never NaN. "
            "Formula: Σ cnt_resp_wiz_true (≤ current decile)."
        ),
        "cum_resp_wiz_true_pct": (
            "Meaning: % cumulative ideal positives = cum_resp_wiz_true / total_responders * 100. "
            "Critical: Wizard benchmark for lift/gain curves; gaps indicate model weakness. "
            "Formula: cum_resp_wiz_true / total_responders * 100."
        ),
        "cumulative_lift": (
            "Meaning: Empirical discriminative power; shows cumulative improvement vs random. "
            "Critical: Always cumulative; should exceed 1 (or ≥2 in top decile). "
            "Formula: cumulative_lift = cum_resp_true_pct / cum_resp_total_pct."
        ),
        "decile_wise_lift": (
            "Meaning: Improvement factor for individual deciles; shows how much better each decile performs vs random. "
            "Critical: Fatal if <1. Early deciles should show highest lift. "
            "Formula: decile_wise_lift = cnt_resp_true / cnt_resp_rndm_true."
        ),
        "KS": (
            "Meaning: Peak discriminative power (scalar) extracted from cumulative gain curves; maximum distance between cumulative distributions of positives and negatives. "
            "Range: 0-1 (fraction) or 0-100 (percent). "
            "Interpretation: "
            "- <0.2 → Poor discrimination "
            "- 0.2-0.4 → Fair "
            "- 0.4-0.6 → Good "
            "- ≥0.6 → Excellent "
            "- ≥0.7 → Suspiciously high (possible overfitting or leakage). "
            "Critical: Report across train/validation/test; ensure top deciles dominate appropriately. "
            "Formula: KS = max(cum_resp_true_pct - cum_resp_false_pct) (sorted descending by model score)."
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
            "cnt_resp_true",
            "cnt_resp_false",
            "cnt_resp_rndm_true",
            "cnt_resp_wiz_true",
            "rate_resp",  # (alias to decile_wise_response, decile_wise_gain)
            "rate_resp_pct",  # (alias to decile_wise_response, decile_wise_gain %)
            "overall_rate",
            "cum_resp_total",
            "cum_resp_total_pct",
            "cum_resp_true",  #  (alias to cumulative_gain)
            "cum_resp_true_pct",  #  (alias to cumulative_gain %)
            "cum_resp_false",
            "cum_resp_false_pct",
            "cum_resp_rndm_true",
            "cum_resp_rndm_true_pct",
            "cum_resp_wiz_true",
            "cum_resp_wiz_true_pct",
            "cumulative_lift",
            "decile_wise_lift",
            "KS",
        ]

    Parameters
    ----------
    as_json : bool, default=True
        If True, pretty-print as JSON. If False, use Python's pprint.
    indent : int, default=2
        Indentation for JSON formatting.

    See Also
    --------
    decileplot : Given binary labels y_true (0/1) and probabilities y_score 1d array,
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

    def compute_decile_table(  # noqa: PLR0912
        self,
        y_true,
        y_score,
        n_deciles,
        **kws,
    ) -> "pd.DataFrame | None":  # noqa: UP037
        # binary case (most common)
        # Defensive copy via DataFrame to use pandas utilities
        dt = pd.DataFrame(
            {"y_true": y_true, "y_score": y_score},
        )  # .dropna()
        # Ranked group (1=highest probability)
        # Ensure sorted descending by model score. Fatal if not.
        # Sort probabilities by descending score so highest scores are in decile 1
        dt = dt.sort_values(by="y_score", ascending=False).reset_index(drop=True)
        # Assign deciles (pop_pct) by index using qcut behavior. Use ranks (index) to avoid duplicate-edge errors.
        # qcut on the index will create near-equal sized bins; duplicates="drop" avoids failures with many ties.
        try:
            # dt["decile"] = pd.qcut(dt.index, q=n_deciles, labels=False, duplicates="drop") + 1
            # dt['decile'] = pd.qcut(dt['y_score'], n_deciles, labels=list(np.arange(n_deciles, 0, -1)))  # ValueError: Bin edges must be unique
            dt["decile"] = np.linspace(1, n_deciles + 1, len(dt), False, dtype=int)
        except ValueError:
            # Fallback: if qcut fails (e.g., too few unique values), use linspace index-based bins
            n = len(dt)
            bins = np.linspace(0, n, n_deciles + 1)
            dt["decile"] = (
                pd.cut(dt.index, bins=bins, labels=False, include_lowest=True) + 1
            )
        # Calculate additional columns rndm scaler constant
        # total_positives / n_deciles
        rndm = dt["y_true"].sum() / n_deciles  # sorted y_score, scaler constant
        # Calculate additional columns Wizard by "y_true"
        # tmp['y_true'] = np.sort(dt['y_true'])[::-1]  # all positives first
        tmp = dt[["y_true"]].sort_values("y_true", ascending=False)
        tmp["decile"] = np.linspace(1, n_deciles + 1, len(tmp), False, dtype=int)
        tmp = tmp.groupby(
            "decile",
            group_keys=False,  # Changed in version 2.0.0: group_keys now defaults to True.
            as_index=True,  # for as df
        )
        wiz = (
            tmp["y_true"].sum().reset_index(drop=True)
        )  # sorted y_true, total positives

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
                        np.sum(x["y_true"]),
                        np.size(x["y_true"][x["y_true"] == 0]),
                        np.size(x["y_score"]),
                    ],
                    index=(
                        [
                            "prob_min",
                            "prob_max",
                            "prob_avg",
                            "cnt_resp_true",
                            "cnt_resp_false",
                            "cnt_resp_total",
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
        # agg["cnt_nonresp"] = agg["cnt_total"] - agg["cnt_resp_true"]

        # Add random/wizard responders for benchmarking
        agg["cnt_resp_rndm_true"] = rndm  # Baseline scaler constant
        agg["cnt_resp_wiz_true"] = wiz  # Perfect by y_true

        # Direct decile quality metric
        # decile-wise response (per decile rate)
        agg["rate_resp"] = (agg["cnt_resp_true"] / agg["cnt_resp_total"]) * 100.0
        agg["rate_resp_wiz"] = (
            agg["cnt_resp_wiz_true"] / agg["cnt_resp_total"]
        ) * 100.0
        agg["overall_rate"] = (
            agg["cnt_resp_true"].sum() / agg["cnt_resp_total"].sum()
        ) * 100.0

        # cumulative columns and percentages: avoid division by zero
        # cumulative gain/response as % of total responders in population
        agg["cum_resp_true"] = np.cumsum(agg["cnt_resp_true"])
        agg["cum_resp_true_pct"] = (
            agg["cum_resp_true"] / np.sum(agg["cnt_resp_true"])
        ) * 100.0
        agg["cum_resp_false"] = np.cumsum(agg["cnt_resp_false"])
        agg["cum_resp_false_pct"] = (
            agg["cum_resp_false"] / np.sum(agg["cnt_resp_false"])
        ) * 100.0
        agg["cum_resp_total"] = np.cumsum(agg["cnt_resp_total"])
        agg["cum_resp_total_pct"] = (
            agg["cum_resp_total"] / np.sum(agg["cnt_resp_total"])
        ) * 100.0
        agg["cum_resp_rndm_true"] = np.cumsum(agg["cnt_resp_rndm_true"])
        agg["cum_resp_rndm_true_pct"] = (
            agg["cum_resp_rndm_true"] / np.sum(agg["cnt_resp_rndm_true"])
        ) * 100.0
        agg["cum_resp_wiz_true"] = np.cumsum(agg["cnt_resp_wiz_true"])
        agg["cum_resp_wiz_true_pct"] = (
            agg["cum_resp_wiz_true"] / np.sum(agg["cnt_resp_wiz_true"])
        ) * 100.0

        # avoid division by zero in lift
        agg["cumulative_lift"] = agg["cum_resp_true_pct"] / agg["cum_resp_total_pct"]
        # perfect lift, random lift = 1
        agg["cumulative_lift_wiz"] = (
            agg["cum_resp_wiz_true_pct"] / agg["cum_resp_total_pct"]
        )
        # if cnt_resp_rndm_true = overall_rate * cnt_resp_total
        # This is mathematically equivalent to the formula below
        # (just expressed using expected random responders instead of overall rate).
        # This computes the local improvement factor in that decile:
        # overall_rate = agg["cnt_resp_true"].sum() / agg["cnt_resp_total"].sum()
        # agg["decile_wise_lift1"] = (agg["cnt_resp_true"] / agg["cnt_resp_total"]) / overall_rate
        agg["decile_wise_lift"] = agg["cnt_resp_true"] / agg["cnt_resp_rndm_true"]
        # perfect lift, random lift = 1
        agg["decile_wise_lift_wiz"] = (
            agg["cnt_resp_wiz_true"] / agg["cnt_resp_rndm_true"]
        )

        # KS and cumulative_lift
        agg["KS"] = (
            agg["cum_resp_true_pct"] - agg["cum_resp_false_pct"]
        )  # .round(digits)

        # agg = agg.fillna(0)  # replace NaNs from division by zeros
        return agg

    # ---------------------------
    # Plot helpers
    # ---------------------------
    def _plot_decile_wise_lift(
        self, dt: pd.DataFrame, digits, legend, ax, annot, fmt, annot_kws, **kws
    ):
        n_deciles = dt["decile"].max()
        # Save artist and label (include statement if legend requested)
        artists, labels = [], []
        for idx, lift in zip(dt["decile"], dt["decile_wise_lift"]):
            if annot:
                # va: Vertical alignment ('top', 'center', 'bottom').
                annot_kws["va"] = "top" if idx % 2 else "bottom"
                ax.annotate(
                    # np.around(lift, decimals=digits),  # or np.round
                    # np.format_float_scientific(lift, precision=digits),
                    np.format_float_positional(lift, precision=digits or 4),
                    (idx, lift),
                    # size=9,
                    # color='k',
                    # weight='heavy',
                    **annot_kws,
                )
            if legend and idx == 1:
                # proxy lines for legend
                # color='gray', linestyle='--', label='Baseline'
                # baseline_proxy = mlines.Line2D([], [])
                # color='green', linestyle=':', label='Perfect model'
                perfect_proxy = mlines.Line2D([], [], marker="o")
                # Save artist and label (include statement if legend requested)
                # ("{:" + self.fmt + "}").format(val)
                # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
                # Double braces {{ or }} → escape to literal { or }.
                (
                    artists.append(perfect_proxy),
                    labels.append(f"lift k-th = {idx}, score = {lift:{fmt}}"),
                )
        (artist,) = ax.plot(
            dt["decile"],
            dt["decile_wise_lift"],
            marker="o",
            label="Model (Improvement Factor)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Improvement Factor)")
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
        # Wizard model = a hypothetically perfect classifier.
        # It always ranks all responders at the very top of the population
        # It gives the theoretical upper bound of performance.
        # If your model curve is close to wizard → very strong.
        # If your model is close to random → useless.
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cumulative_lift_wiz"]),
            dt["decile_wise_lift_wiz"],
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
        ax.set(
            # "Population %\n(sorted by predicted score, descending; decile/percentile)"
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Decile-wise Lift Ratio\n(Lift = decile response rate / overall response rate)"
            ylabel="Decile-wise Lift Ratio\n(Lift k-th, Per-decile k%)",
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

    def _plot_cumulative_lift(
        self, dt: pd.DataFrame, digits, legend, ax, annot, fmt, annot_kws, **kws
    ):
        n_deciles = dt["decile"].max()
        artists, labels = [], []
        for idx, lift in zip(dt["decile"], dt["cumulative_lift"]):
            if annot:
                # va: Vertical alignment ('top', 'center', 'bottom').
                annot_kws["va"] = "top" if idx % 2 else "bottom"
                ax.annotate(
                    # np.around(lift, decimals=digits),  # or np.round
                    # np.format_float_scientific(lift, precision=digits),
                    np.format_float_positional(lift, precision=digits or 4),
                    (idx, lift),
                    # size=9,
                    # color='k',
                    # weight='heavy',
                    **annot_kws,
                )
            if legend and idx == 1:
                # proxy lines for legend
                # color='gray', linestyle='--', label='Baseline'
                # baseline_proxy = mlines.Line2D([], [])
                # color='green', linestyle=':', label='Perfect model'
                perfect_proxy = mlines.Line2D([], [], marker="o")
                # Save artist and label (include statement if legend requested)
                # ("{:" + self.fmt + "}").format(val)
                # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
                # Double braces {{ or }} → escape to literal { or }.
                (
                    artists.append(perfect_proxy),
                    labels.append(f"lift @ k = {idx}, score = {lift:{fmt}}"),
                )
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
            label="Model (Empirical Discriminative Power)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Empirical Discriminative Power)")
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
        # Wizard model = a hypothetically perfect classifier.
        # It always ranks all responders at the very top of the population
        # It gives the theoretical upper bound of performance.
        # If your model curve is close to wizard → very strong.
        # If your model is close to random → useless.
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cumulative_lift_wiz"]),
            dt["cumulative_lift_wiz"],
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
        # Axis and title (parallels cumulative response / decile response plots)
        ax.set(
            # "Population %\n(sorted by predicted score, descending; decile/percentile)"
            # "Population %\n(sorted by score (desc), decile/percentile)"
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Cumulative Lift Ratio\n(Lift@k = responders% up to k / population% up to k)"
            ylabel="Cumulative Lift Ratio\n(Lift @ k, top k%)",
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

    def _plot_decile_wise_gain(
        self, dt: pd.DataFrame, digits, legend, ax, annot, fmt, annot_kws, **kws
    ):
        n_deciles = dt["decile"].max()
        artists, labels = [], []
        for idx, gain in zip(dt["decile"], dt["rate_resp"]):
            if annot:
                # va: Vertical alignment ('top', 'center', 'bottom').
                annot_kws["va"] = "top" if idx % 2 else "bottom"
                ax.annotate(
                    # np.around(gain, decimals=digits),  # or np.round
                    # np.format_float_scientific(gain, precision=digits),
                    np.format_float_positional(gain, precision=digits or 4),
                    (idx, gain),
                    # size=9,
                    # color='k',
                    # weight='heavy',
                    **annot_kws,
                )
            if legend and idx == 1:
                # proxy lines for legend
                # color='gray', linestyle='--', label='Baseline'
                # baseline_proxy = mlines.Line2D([], [])
                # color='green', linestyle=':', label='Perfect model'
                perfect_proxy = mlines.Line2D([], [], marker="o")
                # Save artist and label (include statement if legend requested)
                # ("{:" + self.fmt + "}").format(val)
                # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
                # Double braces {{ or }} → escape to literal { or }.
                (
                    artists.append(perfect_proxy),
                    labels.append(f"gain k-th = {idx}, score = {gain:{fmt}}"),
                )
        # Model decile response rate
        (artist,) = ax.plot(
            dt["decile"],
            dt["rate_resp"],  # Ensure convert to %
            marker="o",
            label="Model (Improvement Factor)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Improvement Factor)")
        # Random baseline: flat at overall response rate
        # (dt["cnt_resp_true"].sum() / dt["cnt_resp_total"].sum()) * 100.0
        overall_rate = dt["overall_rate"].mean()  # .iloc[0]
        (artist,) = ax.plot(
            [1, n_deciles],
            [overall_rate, overall_rate],
            "k--",
            # "Random Baseline (Response = flat line (overall average))"
            label=f"Random Baseline (Response={overall_rate:{fmt}}% (overall avg))",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        # ("{:" + self.fmt + "}").format(val)
        # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
        # Double braces {{ or }} → escape to literal { or }.
        artists.append(artist)
        labels.append(f"Random Baseline (Response={overall_rate:{fmt}}% (overall avg))")
        # Wizard model = a hypothetically perfect classifier.
        # It always ranks all responders at the very top of the population
        # It gives the theoretical upper bound of performance.
        # If your model curve is close to wizard → very strong.
        # If your model is close to random → useless.
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cumulative_lift_wiz"]),
            dt["rate_resp_wiz"],
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
        ax.set(
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Response Rate per Decile"
            ylabel="Decile Response Rate (%)\n(Gain k-th, Per-decile k%)",
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

    def _plot_cumulative_gain(
        self, dt: pd.DataFrame, digits, legend, ax, annot, fmt, annot_kws, **kws
    ):
        n_deciles = dt["decile"].max()
        artists, labels = [], []
        for idx, gain in zip(dt["decile"], dt["cum_resp_true_pct"]):
            if annot:
                # va: Vertical alignment ('top', 'center', 'bottom').
                annot_kws["va"] = "top" if idx % 2 else "bottom"
                ax.annotate(
                    # np.around(gain, decimals=digits),  # or np.round
                    # np.format_float_scientific(gain, precision=digits),
                    np.format_float_positional(gain, precision=digits or 4),
                    (idx, gain),
                    # size=9,
                    # color='k',
                    # weight='heavy',
                    **annot_kws,
                )
            if legend and idx == 1:
                # proxy lines for legend
                # color='gray', linestyle='--', label='Baseline'
                # baseline_proxy = mlines.Line2D([], [])
                # color='green', linestyle=':', label='Perfect model'
                perfect_proxy = mlines.Line2D([], [], marker="o")
                # Save artist and label (include statement if legend requested)
                # ("{:" + self.fmt + "}").format(val)
                # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
                # Double braces {{ or }} → escape to literal { or }.
                artists.append(perfect_proxy)
                labels.append(f"gain @ k = {idx}, score = {gain:{fmt}}")
        # Model curve = your actual model → discriminative power lies between random and wizard.
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cum_resp_true_pct"]),
            dt["cum_resp_true_pct"],
            marker="o",
            label="Model (Empirical Discriminative Power)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Empirical Discriminative Power)")
        # Random baseline = no model, responders distributed evenly → Lift = 1 / diagonal line.
        (artist,) = ax.plot(
            [1, n_deciles],
            [1, 100],
            "k--",
            label="Random Baseline (Responders=diagonal)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Random Baseline (Responders=diagonal)")
        # Wizard model = a hypothetically perfect classifier.
        # It always ranks all responders at the very top of the population
        # It gives the theoretical upper bound of performance.
        # If your model curve is close to wizard → very strong.
        # If your model is close to random → useless.
        # Wizard baseline = perfect model, responders ranked first → theoretical maximum performance.
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cum_resp_wiz_true_pct"]),
            dt["cum_resp_wiz_true_pct"],
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
        ax.set(
            # "Population %\n(sorted by predicted score, descending; decile/percentile)"
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Cumulative Responders %\n(captured vs total responders)"
            ylabel="Cumulative Responders %\n(Gain @ k, top k%)",
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

    def _plot_cumulative_response(
        self, dt: pd.DataFrame, digits, legend, ax, annot, fmt, annot_kws, **kws
    ):
        n_deciles = dt["decile"].max()
        artists, labels = [], []
        for idx, gain in zip(dt["cum_resp_total_pct"], dt["cum_resp_true_pct"]):
            if annot:
                # va: Vertical alignment ('top', 'center', 'bottom').
                annot_kws["va"] = "top" if idx % 2 else "bottom"
                ax.annotate(
                    # np.around(gain, decimals=digits),  # or np.round
                    # np.format_float_scientific(gain, precision=digits),
                    np.format_float_positional(gain, precision=digits or 4),
                    (idx, gain),
                    # size=9,
                    # color='k',
                    # weight='heavy',
                    **annot_kws,
                )
            if legend and idx == dt["cum_resp_total_pct"].min():
                # proxy lines for legend
                # color='gray', linestyle='--', label='Baseline'
                # baseline_proxy = mlines.Line2D([], [])
                # color='green', linestyle=':', label='Perfect model'
                perfect_proxy = mlines.Line2D([], [], marker="o")
                # Save artist and label (include statement if legend requested)
                # ("{:" + self.fmt + "}").format(val)
                # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
                # Double braces {{ or }} → escape to literal { or }.
                artists.append(perfect_proxy)
                labels.append(f"responders @ k = {idx}, score = {gain:{fmt}}")
        # Model cumulative response %
        (artist,) = ax.plot(
            dt["cum_resp_total_pct"],
            dt["cum_resp_true_pct"],
            marker="o",
            label="Model (Empirical Discriminative Power)",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Model (Empirical Discriminative Power)")
        # Random baseline: diagonal line (responders captured ~ proportional to population)
        (artist,) = ax.plot(
            # [1, n_deciles],
            # [1, 100],
            dt["cum_resp_total_pct"],
            dt["cum_resp_rndm_true_pct"],
            "k--",
            label="Random Baseline (Responders=diagonal)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Random Baseline (Responders=diagonal)")
        # Wizard (perfect model): straight line, captures all responders as fast as possible
        (artist,) = ax.plot(
            dt["cum_resp_total_pct"],
            dt["cum_resp_wiz_true_pct"],
            "c--",
            marker="2",
            label="Wizard Baseline (Perfect Model)",
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Wizard Baseline (Perfect Model)")
        ax.set(
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # (Capture rate up to decile)
            ylabel="Cumulative Responders %\n(Responders @ k, top k%)",
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

    def _plot_ks_statistic(
        self, dt: pd.DataFrame, digits, legend, ax, annot, fmt, annot_kws, **kws
    ):
        n_deciles = dt["decile"].max()
        # Save artist and label (include KS if legend requested)
        artists, labels = [], []
        ks_max = dt["KS"].max()
        ks_decile = dt.loc[
            dt["KS"].idxmax(), "decile"
        ]  # ksdcl = dt[ksmx == dt.KS].decile.values
        # ax.axvline(ks_decile, linestyle="--", color="g",
        #            label=f"KS={ks_max:.2f} @ decile {ks_decile}")
        (artist,) = ax.plot(
            [ks_decile, ks_decile],
            [
                dt[ks_max == dt.KS].cum_resp_true_pct.to_numpy(),
                dt[ks_max == dt.KS].cum_resp_false_pct.to_numpy(),
            ],
            "g--",
            marker="o",
            # label="KS Statistic: " + str(np.format_float_positional(ks_max, precision=digits or 4)) + " at decile " + str(list(ks_decile)[0]),
            label=f"KS k-th = {ks_decile}, score = {ks_max} (Discriminative Power)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist)
        # ("{:" + self.fmt + "}").format(val)
        # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
        # Double braces {{ or }} → escape to literal { or }.
        labels.append(
            f"KS k-th = {ks_decile}, score = {ks_max:{fmt}} (Discriminative Power)"
        )
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cum_resp_true_pct"]),
            dt["cum_resp_true_pct"],
            marker="o",
            label="Responders",
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Responders")
        (artist,) = ax.plot(
            # np.append(0, dt["decile"]),
            dt["decile"],
            # np.append(0, dt["cum_resp_false_pct"]),
            dt["cum_resp_false_pct"],
            marker="o",
            c="darkorange",
            label="Non-Responders",
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append("Non-Responders")
        # (artist,) = ax.plot(
        #     [1, n_deciles],
        #     [1, 100],
        #     "k--",
        #     label="Random Baseline (Responders=diagonal)",
        #     zorder=-1,
        # )
        # # Save artist and label (include statement if legend requested)
        # artists.append(artist), labels.append("Random Baseline (Responders=diagonal)")
        ax.set(
            # "Population %\n(sorted by predicted score, descending; decile/percentile)"
            xlabel="Population %\n(sorted by score, desc; decile/percentile)",
            # "Cumulative Distribution %"
            ylabel="Cumulative Responders %\n(positive & negative rates, max diff %)",
            # "KS Statistic Curve\n(Difference between responders and non-responders)"
            title="Kolmogorov-Smirnov (KS) Statistic Curve",
        )
        ax.grid(True)

        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

    def _plot_report(
        self, dt: pd.DataFrame, digits, legend, ax, annot, fmt, annot_kws, **kws
    ):
        """2x2 dashboard with all metrics."""
        # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        # Get the parent figure from the given Axes
        fig = ax.figure
        # fig override subplot define (nrows, ncols, index)
        # int, (int, int, index), or SubplotSpec, default: (1, 1, 1)
        # Each subplot is placed in a grid defined by nrows x ncols at position idx
        # plt.figure(figsize=kws.pop('figsize', (7.5, 1)))
        width, height = fig.get_size_inches()
        width, height = width * 2, height * 2  # np.log1p(3.5)
        fig.set_size_inches(kws.pop("figsize", (width, height)), forward=True)

        # Replace single Axes with a 2x2 grid inside the same figure
        # plt.gcf().clf()  # clear any existing axes, clar figure
        # plt.clf()  # clear any existing axes, clar figure
        fig.clf()  # clear any existing axes
        axes = fig.subplots(2, 2)

        # Draw plots
        self._plot_cumulative_lift(
            dt, digits, legend, axes[0, 0], annot, fmt, annot_kws, **kws
        )
        self._plot_decile_wise_lift(
            dt, digits, legend, axes[0, 1], annot, fmt, annot_kws, **kws
        )
        self._plot_cumulative_gain(
            dt, digits, legend, axes[1, 0], annot, fmt, annot_kws, **kws
        )
        self._plot_ks_statistic(
            dt, digits, legend, axes[1, 1], annot, fmt, annot_kws, **kws
        )
        fig.tight_layout()

    def assert_binary_compat(  # noqa: PLR0912
        self,
        y_true,
        y_score=None,
        y_pred=None,
        threshold=0.5,
        allow_probs=True,
    ):
        """
        Validate compatibility of y_true, y_score, and y_pred for binary classification input consistency.

        Supports:
            - (y_true, y_pred)
            - (y_true, y_score) → auto-derives y_pred from y_score if needed.

        Ensures:
            • same length and shape
            • no NaN / inf values
            • binary {0,1} targets
            • valid [0,1] probability ranges when allow_probs=True
            • consistent return of (y_true, y_score, y_pred)

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary target values {0, 1}.

        y_score : array-like of shape (n_samples,), optional
            Predicted scores or probabilities. Used if `y_pred` is not given.

        y_pred : array-like of shape (n_samples,), optional
            Predicted binary labels {0, 1}. If not provided, will be derived from
            `y_score` using `threshold`.

        threshold : float, default=0.5
            Threshold used to convert `y_score` to binary labels when `y_pred` is None.

        allow_probs : bool, default=False
            Whether `y_score` can contain probabilities in [0, 1].
            When True, `y_pred` is derived as (y_score > threshold) like :py:func:`numpy.argmax`.

        Returns
        -------
        tuple of ndarray of shape (n_samples,)
            (y_true, y_score, y_pred)

            • `y_score` may be None if not provided.
            • `y_pred` is always returned and guaranteed strictly binary {0, 1}.

        Raises
        ------
        ValueError
            If input arrays are incompatible, contain NaN/inf, or non-binary values.

        Examples
        --------
        >>> assert_binary_compat([0, 1, 0], [0.1, 0.9, 0.3])  # auto thresholds
        >>> assert_binary_compat([0, 1, 0], y_pred=[0, 1, 0])  # direct
        >>> assert_binary_compat([0, 1, 0], [0.1, 0.9, 0.3], allow_probs=True)
        """
        # --- Input normalization ---
        # Convert if provided (keeps None otherwise)
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel() if y_score is not None else None
        y_pred = np.asarray(y_pred).ravel() if y_pred is not None else None

        # --- Normalize and validate inputs ---
        y_true = check_array(y_true, ensure_2d=False, dtype=int)
        y_score = (
            check_array(y_score, ensure_2d=False, dtype=float)
            if y_score is not None
            else None
        )
        y_pred = (
            check_array(y_pred, ensure_2d=False, dtype=int)
            if y_pred is not None
            else None
        )

        # --- Validate presence/combination ---
        if y_score is None and y_pred is None:
            raise ValueError(
                "Either y_score or y_pred must be provided (not both None)."
            )
        if y_score is not None and y_pred is not None:
            raise ValueError("Provide only one of y_score or y_pred, not both.")

        # --- Auto-Derive y_pred if y_score is given ---
        if y_score is not None:
            if allow_probs:
                if not ((y_score >= 0) & (y_score <= 1)).all():
                    raise ValueError(
                        "y_score must be within [0, 1] when allow_probs=True"
                    )
                y_pred = np.asarray(y_score > threshold, dtype=int)  # .astype(int)
            else:
                unique_score = np.unique(y_score)
                if not np.isin(unique_score, [0, 1]).all():
                    raise ValueError(
                        f"y_score must be binary when allow_probs=False; got unique values {unique_score}"
                    )
                y_pred = np.asarray(y_score, dtype=int)  # .astype(int)

        # --- Validate binary y_true ---
        unique_true = np.unique(y_true)
        if not np.isin(unique_true, [0, 1]).all():
            raise ValueError(f"y_true must be binary; got unique values: {unique_true}")

        # --- Validate binary y_pred ---
        unique_pred = np.unique(y_pred)
        if not np.isin(unique_pred, [0, 1]).all():
            raise ValueError(f"y_pred must be binary; got unique values: {unique_pred}")

        # --- Length consistency ---
        check_consistent_length(y_true, y_pred)

        # --- NaN / Inf checks (faster in one call) ---
        # assert np.isfinite(y_true).all(), "y_true contains NaN or inf"  # noqa: S101
        if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
            raise ValueError("y_true or y_pred contains NaN or inf values")

        # --- Sanity & consistency checks ---
        n_true, n_pred = len(y_true), len(y_pred)
        if n_true != n_pred:
            raise ValueError(f"Length mismatch: y_true={n_true}, y_pred={n_pred}")

        # --- Return consistent tuple ---
        return y_true, y_score, y_pred

    # ------------------------
    # User-facing Core computation Decile assignment & table
    # ------------------------
    def plot_decileplot(  # noqa: PLR0912
        self,
        kind: "Literal['df', 'cumulative_lift', 'decile_wise_lift', 'cumulative_gain', 'decile_wise_gain', 'cumulative_response', 'ks_statistic', 'report'] | None" = None,  # noqa: UP037
        n_deciles: int = 10,
        fill=None,
        color=None,
        legend=None,
        digits: "int | None" = None,  # noqa: UP037
        annot=None,
        fmt=".4g",
        annot_kws=None,
        verbose=True,
        **plot_kws,
    ) -> "pd.DataFrame | None":  # noqa: UP037
        # -- Default keyword dicts
        # annot_kws = plot_kws.pop('annot_kws', {})
        annot_kws = {} if annot_kws is None else annot_kws.copy()
        # ha: Horizontal alignment ('left', 'center', 'right').
        # annot_kws["ha"] = annot_kws.pop("horizontalalignment", annot_kws.pop("ha", 'left'))
        annot_kws.setdefault(
            "ha", annot_kws.pop("horizontalalignment", annot_kws.pop("ha", "center"))
        )
        # va: Vertical alignment ('top', 'center', 'bottom').
        annot_kws.setdefault(
            "va", annot_kws.pop("verticalalignment", annot_kws.pop("va", "center"))
        )
        annot_kws.setdefault(
            "fontfamily", annot_kws.pop("fontfamily", "monospace")
        )  # 'serif'
        # annot_kws.setdefault("fontname", annot_kws.pop("fontname", mpl.rcParams["font.monospace"]))
        annot_kws.setdefault(
            "fontsize", annot_kws.pop("fontsize", annot_kws.pop("size", 9))
        )
        annot_kws.setdefault(
            "weight", annot_kws.pop("weight", "heavy")
        )  # "bold", "normal"
        # annot_kws.setdefault("color", annot_kws.pop("color", annot_kws.pop("c", sub_color)))
        # annot_kws.setdefault("textcoords", annot_kws.pop("textcoords", "offset points"))
        annot_kws.setdefault(
            "bbox",
            annot_kws.pop(
                "bbox",
                dict(boxstyle="square", pad=0, lw=0, fc=(1, 1, 1, 0.7)),  # noqa: C408
            ),
        )

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

            annot_kws.setdefault(
                "color", annot_kws.pop("color", annot_kws.pop("c", sub_color))
            )

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

            # prepare arrays
            try:
                # y_true, y_score, sample_weight
                y_true, y_score, _sw = self._prepare_subset(sub_data)  # noqa: RUF059
            except ValueError as e:
                warnings.warn(
                    f"Skipping subset {sub_vars} due to data error: {e}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            y_true, y_score, _ = self.assert_binary_compat(
                y_true,
                y_score,
                # threshold=threshold,
                # allow_probs=allow_probs,
            )
            if y_true is None:
                continue

            # binary vs multiclass detection
            # classes = unique_labels(y_true, y_pred)  # both sorted
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
                agg = self.compute_decile_table(y_true, y_score, n_deciles)
                if "hue" in self.variables:
                    level = sub_vars["hue"]
                    agg = agg.assign(hue=level)
                dfs.append(agg)
                # plot decile plotting
                _plot = getattr(self, f"_plot_{kind}", None)
                if _plot:
                    _plot(
                        agg,
                        digits,
                        legend,
                        ax,
                        annot,
                        fmt,
                        annot_kws,
                        **artist_kws,
                    )
        agg = pd.concat(dfs, ignore_index=True)
        return agg.round(digits) if digits else agg


# -----------------------------------------------------------------------------
# Public API functions (wrappers)
# -----------------------------------------------------------------------------
def decileplot(  # noqa: D417
    data: "pd.DataFrame | None" = None,  # noqa: UP037
    *,
    # Vector variables
    x: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    y: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    hue: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    kind: "Literal['df', 'cumulative_lift', 'decile_wise_lift', 'cumulative_gain', 'decile_wise_gain', 'cumulative_response', 'ks_statistic', 'report'] | None" = None,  # noqa: UP037
    weights=None,
    n_deciles: int = 10,
    # Hue mapping parameters
    hue_order=None,
    hue_norm=None,
    palette=None,
    color=None,
    # appearance parameters
    fill=False,
    # Other appearance keywords
    baseline=False,
    # smoothing
    line_kws=None,
    # Axes information
    log_scale=None,
    legend=True,
    ax=None,
    # smoothing
    annot=None,
    fmt="",
    annot_kws=None,
    # computation parameters
    digits: "int | None" = None,  # noqa: UP037
    common_norm=None,
    verbose: bool = False,
    **kwargs,
) -> "pd.DataFrame | mpl.axes.Axes":  # mpl.figure.Figure  # noqa: UP037
    # https://rsted.info.ucl.ac.be/
    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
    # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
    # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist
    kind = (kind and kind.lower().strip()) or "report"
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
    dt = p.plot_decileplot(
        kind=kind,
        n_deciles=n_deciles,
        digits=digits,
        color=color,
        legend=legend,
        fill=fill,
        annot=annot,
        fmt=fmt,
        annot_kws=annot_kws,
        **dt_kws,
    )
    if verbose is True:
        print_labels()
    if kind == "df":
        plt.gca().remove()  # clear the specific axes
        plt.gcf().clf()  # clear any existing axes, clar figure
        plt.clf()  # clear any existing axes, clar figure
        plt.close()  # move trash figure
        return dt
    if kind == "report":
        try:
            from IPython.display import display  # noqa: PLC0415

            display(dt)
        except Exception:
            print(dt)  # noqa: T201
    return ax


decileplot.__doc__ = """\
Given binary labels y_true (0/1) and probabilities y_score 1d array, compute/plot a decile [2]_ table.

The function sorts observations by descending score, assigns decile index
(1..n_deciles) using pandas qcut on the rank/index to ensure near-equal bins,
and computes standard decile-level stats (features)::

    [
        "decile",
        "prob_min",
        "prob_max",
        "prob_avg",
        "cnt_resp_total",
        "cnt_resp_true",
        "cnt_resp_false",
        "cnt_resp_rndm_true",
        "cnt_resp_wiz_true",
        "rate_resp",  # (alias to decile_wise_response, decile_wise_gain)
        "rate_resp_pct",  # (alias to decile_wise_response, decile_wise_gain %)
        "overall_rate",
        "cum_resp_total",
        "cum_resp_total_pct",
        "cum_resp_true",  #  (alias to cumulative_gain)
        "cum_resp_true_pct",  #  (alias to cumulative_gain %)
        "cum_resp_false",
        "cum_resp_false_pct",
        "cum_resp_rndm_true",
        "cum_resp_rndm_true_pct",
        "cum_resp_wiz_true",
        "cum_resp_wiz_true_pct",
        "cumulative_lift",
        "decile_wise_lift",
        "KS",
    ]

Parameters
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
kind : {{'df', 'cumulative_lift', 'decile_wise_lift', 'cumulative_gain', 'decile_wise_gain', 'cumulative_response', 'ks_statistic', 'report'}} or None, default=None
    Kind of plot to make.

    - if `'df'`, not plot return as pandas.DataFrame;
    - if ``None``, the plot is roc curve.
weights : vector or key in ``data``
    If provided, observation weights used for computing the distribution function.
n_deciles : int, optional, default=10
    The number of partitions for creating the table. Defaults to 10 for deciles.
{params.core.hue_order}
{params.core.hue_norm}
{params.core.palette}
{params.core.color}
fill : bool or None
    If True, fill in the area under univariate density curves or between
    bivariate contours. If None, the default depends on ``multiple``.
{{line}}_kws : dictionaries
    Additional keyword arguments to pass to ``plt.plot``.
{params.dist.log_scale}
{params.dist.legend}
{params.core.ax}
digits : int, optional, default=4
    Number of digits for formatting output floating point values.
    When ``output_dict`` is ``True``, this will be ignored and the
    returned values will not be rounded.
output_dict : bool, default=False
    If True, return output as dict.
zero_division : {{'warn', 0.0, 1.0, np.nan}}, default='warn'
    Sets the value to return when there is a zero division. If set to
    'warn', this acts as 0, but warnings are also raised.
common_norm : bool
    If True, scale each conditional density by the number of observations
    such that the total area under all densities sums to 1. Otherwise,
    normalize each density independently.
verbose : bool, optional, default=False
    Whether to be verbose.
annot : bool or rectangular dataset, optional
    If True, write the data value in each cell. If an array-like with the
    same shape as ``data``, then use this to annotate the heatmap instead
    of the data. Note that DataFrames will match on position, not index.
fmt : str, optional, default=''
    String formatting code to use when adding annotations
    (e.g., '.2g', '.4g').
annot_kws : dict of key, value mappings, optional
    Keyword arguments for :meth:`matplotlib.axes.Axes.text` when ``annot``
    is True.
kwargs
    Other keyword arguments are passed to one of the following matplotlib
    functions:

    - :meth:`matplotlib.axes.Axes.plot`

Returns
-------
pandas.DataFrame | matplotlib.axes.Axes | dict
    The dataframe (decile-table) with the indexed by deciles (sorted ascending)
    and related information (decile-level metrics).
    If hue/facet semantics were used, the returned table will include
    extra columns for those keys (e.g., 'hue').

.. warning::

    Some function parameters are experimental prototypes.
    These may be modified, renamed, or removed in future library versions.
    Use with caution and check documentation for the latest updates.

References
----------
.. [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py

.. [2] `Wikipedia contributors. (2024).
    "Decile"
    Wikipedia. https://en.wikipedia.org/wiki/Decile
    <https://en.wikipedia.org/wiki/Decile>`_
""".format(  # noqa: UP032
    params=_param_docs,
    # returns=_core_docs["returns"],
    # seealso=_core_docs["seealso"],
)
