# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Seaborn-style PR and ROC curve plotting.

This module provides:
- A core `_AucPlotter` class (subclass of VectorPlotter) for handling
  data preparation and plotting PR/ROC curves.
- User-facing functions `prplot()` and `rocplot()` that work just like
  seaborn's high-level API (e.g., scatterplot, lineplot).

Features:
- Multiple groups with `hue`
- Automatic legends with AUC scores
- Optional filled area under curve
- Baselines (random chance lines or prevalence lines)

References
----------
.. [1] `scikit-learn contributors. (2025).
   "sklearn.metrics"
   scikit-learn. https://scikit-learn.org/stable/api/sklearn.metrics.html
   <https://scikit-learn.org/stable/api/sklearn.metrics.html>`_
"""

# code that needs to be compatible with both Python 2 and Python 3
# from __future__ import (
#     absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
#     division,  # Changes the division operator `/` to always perform true division.
#     print_function,  # Treats `print` as a function, consistent with Python 3 syntax.
#     unicode_literals,  # Makes all string literals Unicode by default, similar to Python 3.
# )
from __future__ import annotations  # Allows string type hints

# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import warnings
from typing import ClassVar, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
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
    "aucplot",
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


# --------------------------------------------------------------------
# Core Plotter Class
# --------------------------------------------------------------------
class _AucPlotter(VectorPlotter):
    """
    Seaborn-style Auc plotter internal class for PR / ROC curves plotting.

    Expects the VectorPlotter pipeline to have mapped incoming data into
    standardized columns "x", "y", optionally grouping (hue), legend creation,
    "weights", and subset iteration.
    """

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

    # def _quantile_to_level(self, data, quantile):
    #     """Return data levels corresponding to quantile cuts of mass."""
    #     isoprop = np.asarray(quantile)
    #     values = np.ravel(data)
    #     sorted_values = np.sort(values)[::-1]
    #     normalized_values = np.cumsum(sorted_values) / values.sum()
    #     idx = np.searchsorted(normalized_values, 1 - isoprop)
    #     levels = np.take(sorted_values, idx, mode="clip")
    #     return levels

    # def _default_discrete(self):
    #     """Find default values for discrete hist estimation based on variable type."""
    #     if self.univariate:
    #         discrete = self.var_types[self.data_variable] == "categorical"
    #     else:
    #         discrete_x = self.var_types["x"] == "categorical"
    #         discrete_y = self.var_types["y"] == "categorical"
    #         discrete = discrete_x, discrete_y
    #     return discrete

    # def _resolve_multiple(self, curves, multiple):
    #     """Modify the density data structure to handle multiple densities."""

    #     # Default baselines have all densities starting at 0
    #     baselines = {k: np.zeros_like(v) for k, v in curves.items()}

    #     # TODO we should have some central clearinghouse for checking if any
    #     # "grouping" (terminnology?) semantics have been assigned
    #     if "hue" not in self.variables:
    #         return curves, baselines

    #     if multiple in ("stack", "fill"):

    #         # Setting stack or fill means that the curves share a
    #         # support grid / set of bin edges, so we can make a dataframe
    #         # Reverse the column order to plot from top to bottom
    #         curves = pd.DataFrame(curves).iloc[:, ::-1]

    #         # Find column groups that are nested within col/row variables
    #         column_groups = {}
    #         for i, keyd in enumerate(map(dict, curves.columns)):
    #             facet_key = keyd.get("col", None), keyd.get("row", None)
    #             column_groups.setdefault(facet_key, [])
    #             column_groups[facet_key].append(i)

    #         baselines = curves.copy()

    #         for col_idxs in column_groups.values():
    #             cols = curves.columns[col_idxs]

    #             norm_constant = curves[cols].sum(axis="columns")

    #             # Take the cumulative sum to stack
    #             curves[cols] = curves[cols].cumsum(axis="columns")

    #             # Normalize by row sum to fill
    #             if multiple == "fill":
    #                 curves[cols] = curves[cols].div(norm_constant, axis="index")

    #             # Define where each segment starts
    #             baselines[cols] = curves[cols].shift(1, axis=1).fillna(0)

    #     if multiple == "dodge":

    #         # Account for the unique semantic (non-faceting) levels
    #         # This will require rethiniking if we add other semantics!
    #         hue_levels = self.var_levels["hue"]
    #         n = len(hue_levels)
    #         f_fwd, f_inv = self._get_scale_transforms(self.data_variable)
    #         for key in curves:

    #             level = dict(key)["hue"]
    #             hist = curves[key].reset_index(name="heights")
    #             level_idx = hue_levels.index(level)

    #             a = f_fwd(hist["edges"])
    #             b = f_fwd(hist["edges"] + hist["widths"])
    #             w = (b - a) / n
    #             new_min = f_inv(a + level_idx * w)
    #             new_max = f_inv(a + (level_idx + 1) * w)
    #             hist["widths"] = new_max - new_min
    #             hist["edges"] = new_min

    #             curves[key] = hist.set_index(["edges", "widths"])["heights"]

    #     return curves, baselines

    # -------------------------------------------------------------------------------- #
    # Computation
    # -------------------------------------------------------------------------------- #
    # def _compute_univariate_density(
    #     self,
    #     data_variable,
    #     common_norm,
    #     common_grid,
    #     estimate_kws,
    #     warn_singular=True,
    # ):

    #     # Initialize the estimator object
    #     estimator = KDE(**estimate_kws)

    #     if set(self.variables) - {"x", "y"}:
    #         if common_grid:
    #             all_observations = self.comp_data.dropna()
    #             estimator.define_support(all_observations[data_variable])
    #     else:
    #         common_norm = False

    #     all_data = self.plot_data.dropna()
    #     if common_norm and "weights" in all_data:
    #         whole_weight = all_data["weights"].sum()
    #     else:
    #         whole_weight = len(all_data)

    #     densities = {}

    #     for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

    #         # Extract the data points from this sub set and remove nulls
    #         observations = sub_data[data_variable]

    #         # Extract the weights for this subset of observations
    #         if "weights" in self.variables:
    #             weights = sub_data["weights"]
    #             part_weight = weights.sum()
    #         else:
    #             weights = None
    #             part_weight = len(sub_data)

    #         # Estimate the density of observations at this level
    #         variance = np.nan_to_num(observations.var())
    #         singular = len(observations) < 2 or math.isclose(variance, 0)
    #         try:
    #             if not singular:
    #                 # Convoluted approach needed because numerical failures
    #                 # can manifest in a few different ways.
    #                 density, support = estimator(observations, weights=weights)
    #         except np.linalg.LinAlgError:
    #             singular = True

    #         if singular:
    #             msg = (
    #                 "Dataset has 0 variance; skipping density estimate. "
    #                 "Pass `warn_singular=False` to disable this warning."
    #             )
    #             if warn_singular:
    #                 warnings.warn(msg, UserWarning, stacklevel=4)
    #             continue

    #         # Invert the scaling of the support points
    #         _, f_inv = self._get_scale_transforms(self.data_variable)
    #         support = f_inv(support)

    #         # Apply a scaling factor so that the integral over all subsets is 1
    #         if common_norm:
    #             density *= part_weight / whole_weight

    #         # Store the density for this level
    #         key = tuple(sub_vars.items())
    #         densities[key] = pd.Series(density, index=support)

    #     return densities

    # -------------------------
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

    def _compute_pr(
        self,
        y_true,
        y_score,
        sample_weight,
        # sub_vars,
    ) -> "tuple[np.ndarray, np.ndarray, float] | None":  # noqa: UP037
        # pick one
        # auc(recall, precision) vs average_precision_score
        # The area under PR curve (trapezoidal rule with auc) is not the same as average precision (AP).
        # average_precision_score(y_true, y_score) → step-wise interpolation (preferred in ML evaluation,
        # auc(recall, precision) → trapezoidal integration
        # precision_recall_curve always prepends a point (recall=0, precision=1.0) → this can artificially inflate the trapezoidal auc.
        try:
            # prepare coordinates (recall on x, precision on y)
            # recall is sorted ascending, precision follows.
            # recall: monotonically increasing from 0 → 1
            # precision: non-monotonic, often zig-zagging, starting at 1.0
            precision, recall, _ = precision_recall_curve(
                y_true,
                y_score,
                sample_weight=sample_weight,  # pos_label=None
            )
            # If recall is not strictly monotonic, enforce monotonicity:
            # order = np.argsort(recall)
            # recall, precision = recall[order], precision[order]
            # Option 1: trapezoidal PR-AUC (not standard)
            # pr_auc_trapz = auc(recall, precision)
            # Option 2: average precision (preferred sklearn)
            pr_auc_avg = average_precision_score(
                y_true,
                y_score,
                sample_weight=sample_weight,
            )
            return recall, precision, pr_auc_avg
        except Exception as e:
            warnings.warn(
                f"Unable to compute PR curve for subset {''}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return None, None, None

    def _compute_roc(
        self,
        y_true,
        y_score,
        sample_weight,
        # sub_vars,
    ) -> "tuple[np.ndarray, np.ndarray, float] | None":  # noqa: UP037
        try:
            # prepare coordinates (fpr on x, tpr on y)
            # fpr is sorted ascending, tpr follows.
            fpr, tpr, _ = roc_curve(
                y_true,
                y_score,
                sample_weight=sample_weight,  # pos_label=None
            )
            return fpr, tpr, auc(fpr, tpr)  # roc_auc
        except Exception as e:
            warnings.warn(
                f"Unable to compute PR curve for subset {''}: {e}",
                UserWarning,
                stacklevel=2,
            )
            return None, None, None

    def _plot_pr(
        self,
        y_true,
        y_score,
        sample_weight,
        classes,
        label_base,
        legend,
        ax,
        fmt,
        digits,
        baseline=False,
        **kws,
    ):
        # binary case (most common)
        x, y, auc_score = self._compute_pr(
            y_true,
            y_score,
            sample_weight,
        )
        # Save artist and label (include statement if legend requested)
        artists, labels = [], []
        # # Plot curve: fill area under curve if requested, otherwise plot line
        # if kws.pop("fill", None):
        #     artist = ax.fill_between(x, 0.0, y, **kws)
        #     # fill_between returns PolyCollection; put a proxy Line2D for legend if needed
        #     # but store the PolyCollection as artist for potential manipulation
        # else:
        #     # draw standard AUC line plot
        #     (artist,) = ax.plot(x, y, **kws)
        # Model curve
        # ("{:" + self.fmt + "}").format(val)
        # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
        # Double braces {{ or }} → escape to literal { or }.
        label = (
            f"{label_base} {kws.pop('label', '')}"
            + f" {'pr'}".upper()
            + f" (AUC={auc_score:{fmt}})"
        ).strip()
        label = label if legend else None
        kws["label"] = label
        # Plot
        # use step plot to match sklearn docs
        # scientifically, step is the correct representation for PR.
        # where="post",  # for ax.step
        kws["drawstyle"] = kws.pop("steps-post", "steps-post")
        (artist,) = ax.plot(
            x,
            y,
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append(label)
        ax.grid(True)
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title="PR (Precision-Recall) Curve",
        )
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.title("PR (Precision-Recall) Curve")
        # Finalize subset axes, set axis properties, ensure axes limits and labels look right
        # ax.set_xlim(-0.01, 1.01)
        # ax.set_ylim(-0.012, 1.012)
        # Plot the Random baseline, label='Baseline', Chance level (AUC = 0.5)
        (artist,) = ax.plot(
            [0, 1],
            [1, 0],  # pr
            # "k--",
            c="gray",
            ls="--",
            # lw=1,
            # label="Baseline (diagonal)",
            zorder=-1,
        )
        # optional: add baseline
        # Baseline: horizontal line at positive prevalence (x==1)
        # positive_rate = np.mean(y_true)
        # ax.hlines(positive_rate, 0, 1, colors="gray", linestyles="--", label="baseline")
        # if baseline:
        #     # pos_rate = np.average(y_true == 1, weights=sample_weight)
        #     if sample_weight is None:
        #         pos_rate = float(np.mean(x == 1))
        #     else:
        #         pos_rate = float(
        #             np.sum((x == 1) * sample_weight) / np.sum(sample_weight)
        #         )
        #     # draw baseline
        #     ax.axhline(
        #         pos_rate, color="gray", linestyle="--", linewidth=0.9, zorder=-1
        #     )
        # Save artist and label (include AUC if legend requested)
        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

    def _plot_roc(
        self,
        y_true,
        y_score,
        sample_weight,
        classes,
        label_base,
        legend,
        ax,
        fmt,
        digits,
        baseline=False,
        **kws,
    ):
        # binary case (most common)
        x, y, auc_score = self._compute_roc(
            y_true,
            y_score,
            sample_weight,
        )
        # Save artist and label (include statement if legend requested)
        artists, labels = [], []
        # Model curve
        # ("{:" + self.fmt + "}").format(val)
        # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
        # Double braces {{ or }} → escape to literal { or }.
        label = (
            f"{label_base} {kws.pop('label', '')}"
            + f" {'roc'}".upper()
            + f" (AUC={auc_score:{fmt}})"
        ).strip()
        label = label if legend else None
        kws["label"] = label
        # Plot
        (artist,) = ax.plot(
            x,
            y,
            **kws,
        )
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append(label)
        ax.grid(True)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC (Receiver Operating Characteristic) Curve")
        # Plot the Random baseline, label='Baseline', Chance level (AUC = 0.5)
        (artist,) = ax.plot(
            [0, 1],
            [0, 1],  # roc
            # "k--",
            c="gray",
            ls="--",
            # lw=1,
            # label="Baseline (diagonal)",
            zorder=-1,
        )
        # Save artist and label (include AUC if legend requested)
        # Add legend (use axes of the first plotted subset)
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            # ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax.legend(handles, labels, title=self.variables.get("hue", None))

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

    # -------------------------
    # AUC plotting
    # -------------------------
    def plot_aucplot(  # noqa: PLR0912
        self,
        kind: "Literal['pr', 'roc'] | None" = None,  # noqa: UP037
        fill=None,
        color=None,
        legend=None,
        baseline=False,
        # multiple,
        # element,
        # common_norm,
        # common_grid,
        # common_bins,
        # shrink,
        # kde,
        # kde_kws,
        # line_kws,
        # cbar_ax,
        # cbar_kws,
        # estimate_kws,
        annot=None,
        fmt=".4g",
        annot_kws=None,
        digits: "int | None" = None,  # noqa: UP037
        verbose=False,
        **plot_kws,
    ):
        # x_col = self.variables.get("x")
        # y_col = self.variables.get("y")
        # # If no x/y data return early
        # if x_col is None or y_col is None:
        #     return

        # -- Default keyword dicts
        # kde_kws = {} if kde_kws is None else kde_kws.copy()
        # line_kws = {} if line_kws is None else line_kws.copy()
        # estimate_kws = {} if estimate_kws is None else estimate_kws.copy()
        # orient = self.data_variable

        # Interpret plotting options
        # label = plot_kws.get("label", "")
        alpha = plot_kws.get("alpha", 1.0)
        linestyle = plot_kws.get("linestyle", plot_kws.get("ls"))
        marker = plot_kws.get("marker", plot_kws.get("m"))
        chance = bool(plot_kws.pop("chance", None))

        # Save artist and label (include AUC if legend requested)
        # artists, labels = [], []

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

            # plot line and optional fill, Use element "line" for curve plotting
            artist_kws = self._artist_kws(
                plot_kws, fill, "line", "layer", sub_color, alpha
            )
            # Merge user requested line style/marker if present
            if linestyle is not None:
                artist_kws["linestyle"] = linestyle
            if marker is not None:
                artist_kws["marker"] = marker

            # Prepare arrays
            try:
                y_true, y_score, _sw = self._prepare_subset(sub_data)
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
            # Compute ROC curve
            if multiclass:
                # TODO:x or y → Seaborn expects 1D vectors (arrays, Series, lists), not a 2D DataFrame
                # so multiclass not supported by DataFrame or Seaborn
                warnings.warn(
                    f"Cannot label_binarize multiclass labels: {''}",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                # plt.matshow(a) # Or `plt.imshow(a)`
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.matshow.html#matplotlib.axes.Axes.matshow
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
                _plot = getattr(self, f"_plot_{kind}", None)
                if _plot:
                    _plot(
                        y_true,
                        y_score,
                        _sw,
                        classes,
                        label_base,  # labels,
                        legend,
                        ax,
                        fmt,
                        digits,
                        baseline,
                        **artist_kws,
                    )


# --------------------------------------------------------------------
# Public API functions (wrappers)
# --------------------------------------------------------------------
def aucplot(  # noqa: D417
    data: "pd.DataFrame | None" = None,  # noqa: UP037
    *,
    # Vector variables
    x: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    y: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    hue: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    kind: "Literal['pr', 'roc'] | None" = None,  # noqa: UP037
    weights=None,
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
    fmt=".4g",  # ''
    annot_kws=None,
    # computation parameters
    digits: "int | None" = None,  # noqa: UP037
    common_norm=None,
    verbose: bool = False,
    **kwargs,
) -> mpl.axes.Axes:
    # https://rsted.info.ucl.ac.be/
    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
    # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
    # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist
    kind = (kind and kind.lower().strip()) or "roc"
    _check_argument(
        "kind",
        [
            "pr",
            "roc",
        ],
        kind,
    )
    # Build the VectorPlotter and attach data/variables in seaborn style
    p = _AucPlotter(
        data=data,
        variables={"x": x, "y": y, "hue": hue, "weights": weights},
    )
    # map hue palette/order if needed - we only need internal mapping for iter_data to work
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    # ax = plt.gca() if ax is None else ax
    if ax is None:
        ax = plt.gca()

    # Now attach the axes object to the plotter object
    p._attach(ax, log_scale=log_scale)

    # method = ax.fill_between if fill else ax.plot
    # color = _default_color(method, hue, color, kwargs)
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

    # --- Draw the plots
    auc_kws = kwargs.copy()
    # auc_kws["color"] = color
    # _assign_default_kwargs(pr_kws, p.plot_rocauc, prplot)
    p.plot_aucplot(
        kind=kind,
        digits=digits,
        color=color,
        legend=legend,
        fill=fill,
        annot=annot,
        fmt=fmt,
        annot_kws=annot_kws,
        baseline=baseline,
        **auc_kws,
    )
    return ax


aucplot.__doc__ = """\
AUC curve plot, Seaborn-style.

Parameters
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
kind : {{'pr', 'roc'}} or None, default=None
    Kind of plot to make.

    - if `'pr'`, the plot is pr curve;
    - if `'roc'`, the plot is roc curve;
    - if `None`, the plot is roc curve.
weights : vector or key in ``data``
    If provided, observation weights used for computing the distribution function.
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
zero_division : {{"warn", 0.0, 1.0, np.nan}}, default="warn"
    Sets the value to return when there is a zero division. If set to
    "warn", this acts as 0, but warnings are also raised.
common_norm : bool
    If True, scale each conditional density by the number of observations
    such that the total area under all densities sums to 1. Otherwise,
    normalize each density independently.
verbose : bool, optional, default=False
    Whether to be verbose.
kwargs
    Other keyword arguments are passed to one of the following matplotlib
    functions:

    - :meth:`matplotlib.axes.Axes.plot`

Returns
-------
{returns.ax}

.. warning::

    Some function parameters are experimental prototypes.
    These may be modified, renamed, or removed in future library versions.
    Use with caution and check documentation for the latest updates.
""".format(  # noqa: UP032
    params=_param_docs,
    returns=_core_docs["returns"],
    # seealso=_core_docs["seealso"],
)
