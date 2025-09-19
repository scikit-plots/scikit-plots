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
"""

# code that needs to be compatible with both Python 2 and Python 3
from __future__ import annotations

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

try:
    from seaborn._base import VectorPlotter
    from seaborn.external import husl
    from seaborn.utils import _check_argument, _default_color
except:
    from ..externals._seaborn._base import VectorPlotter
    from ..externals._seaborn.external import husl
    from ..externals._seaborn.utils import (
        _check_argument,
        _default_color,
    )


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

    def _compute_auc(
        self,
        y_true,
        y_score,
        sample_weight,
        kind,
        sub_vars,
    ) -> "float | None":  # noqa: UP037
        # return
        x, y, auc_score = None, None, None

        if kind and kind.lower() in ["roc"]:
            try:
                # prepare coordinates (fpr on x, tpr on y)
                # fpr is sorted ascending, tpr follows.
                fpr, tpr, _ = roc_curve(
                    y_true,
                    y_score,
                    sample_weight=sample_weight,  # pos_label=None
                )
                x, y, auc_score = fpr, tpr, auc(fpr, tpr)  # roc_auc
            except Exception as e:
                warnings.warn(
                    f"Unable to compute PR curve for subset {sub_vars}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
        elif kind and kind.lower() in ["pr"]:
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
                x, y, auc_score = recall, precision, pr_auc_avg  # pick one
            except Exception as e:
                warnings.warn(
                    f"Unable to compute PR curve for subset {sub_vars}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
        return x, y, auc_score

    def _plot_auc(
        self,
        x,
        y,
        sample_weight,
        auc_score,
        kind,
        label_base,
        legend,
        ax,
        baseline=False,
        **kws,
    ):
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
        label = (
            f"{label_base} {kws.get('label', '')}"
            + f" {kind}".upper()
            + f" (AUC={auc_score:.4f})"
        ).strip()
        label = label if legend else None
        kws["label"] = label
        if kind and kind.lower() in ["roc"]:
            (artist,) = ax.plot(
                x,
                y,
                **kws,
            )
        elif kind and kind.lower() in ["pr"]:
            # use step plot to match sklearn docs
            # scientifically, step is the correct representation for PR.
            # where="post",  # for ax.step
            kws["drawstyle"] = kws.get("steps-post", "steps-post")
            (artist,) = ax.plot(
                x,
                y,
                **kws,
            )
            # optional: add baseline
            # positive_rate = np.mean(y_true)
            # ax.hlines(positive_rate, 0, 1, colors="gray", linestyles="--", label="baseline")
        # Save artist and label (include statement if legend requested)
        artists.append(artist), labels.append(label)
        # Plot the Random baseline, label='Baseline', Chance level (AUC = 0.5)
        (artist,) = ax.plot(
            [0, 1],
            [0, 1] if kind == "roc" else [1, 0],  # else pr
            # "k--",
            c="gray",
            ls="--",
            # lw=1,
            # label="Baseline (diagonal)",
            zorder=-1,
        )
        # Save artist and label (include statement if legend requested)
        # artists.append(artist), labels.append(label)
        # Finalize subset axes, set axis properties, ensure axes limits and labels look right
        # ax.set_xlim(-0.01, 1.01)
        # ax.set_ylim(-0.012, 1.012)
        if kind and kind.lower() in ["roc"]:
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC (Receiver Operating Characteristic) Curve")
        elif kind and kind.lower() in ["pr"]:
            ax.set(
                xlabel="Recall",
                ylabel="Precision",
                title="PR (Precision-Recall) Curve",
            )
        ax.grid(True)
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.title("PR (Precision-Recall) Curve")

        # Baseline: horizontal line at positive prevalence (x==1)
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
        verbose=False,
        **plot_kws,
    ):
        """
        Plot AUC curves for each hue/facet subset (one per hue level if hue assigned).

        Parameters
        ----------
        kind : {'pr', 'roc', None}, default=None
            Draw pr or roc plot.
        color : color or None
            Fallback color when no hue mapping is used.
        fill : bool
            Whether to fill a axis.
        color : str
            Line color.
        legend : bool
            Whether to draw a legend.
        baseline : bool
            Whether to draw a baseline.
        verbose : bool, optional, default=False
            If True, prints.
        **plot_kws : dict
            Extra keyword args forwarded to plotting functions and _artist_kws.
            Recognized keys:
              - fill (bool): if True, fill the area under the AUC curve.
              - alpha (float)
              - linestyle / ls
              - marker
        """
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

            # Prepare arrays
            try:
                y_true, y_score, sample_weight = self._prepare_subset(sub_data)
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
                # binary case (most common)
                x, y, auc_score = self._compute_auc(
                    y_true,
                    y_score,
                    sample_weight,
                    kind,
                    sub_vars,
                )
                if x is None:
                    continue

                # plot line and optional fill, Use element "line" for curve plotting
                artist_kws = self._artist_kws(
                    plot_kws, fill, "line", "layer", sub_color, alpha
                )
                # Merge user requested line style/marker if present
                if linestyle is not None:
                    artist_kws["linestyle"] = linestyle
                if marker is not None:
                    artist_kws["marker"] = marker

                self._plot_auc(
                    x,
                    y,
                    sample_weight,
                    auc_score,
                    kind,
                    label_base,
                    legend,
                    ax,
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
    x: "str | pd.Series | np.ndarray | None" = None,  # noqa: UP037
    y: "str | pd.Series | np.ndarray | None" = None,  # noqa: UP037
    kind: "Literal['pr', 'roc'] | None" = None,  # noqa: UP037
    hue: "str | pd.Series | np.ndarray | None" = None,  # noqa: UP037
    weights=None,
    # computation parameters
    common_norm=None,
    # appearance parameters
    fill=False,
    # smoothing
    line_kws=None,
    # Hue mapping parameters
    palette=None,
    hue_order=None,
    hue_norm=None,
    color=None,
    # Axes information
    log_scale=None,
    legend=True,
    ax=None,
    # Other appearance keywords
    baseline=False,
    **kwargs,
) -> mpl.axes.Axes:
    """
    AUC curve plot, Seaborn-style.

    Parameters
    ----------
    data : pandas.DataFrame | None
    x : str | pd.Series | np.ndarray | None
        Ground truth (correct/actual) target values.
    y : str | pd.Series | np.ndarray | None
        Prediction probabilities for target class returned by a classifier/algorithm.
    kind : {'pr', 'roc'}, default='roc'
    hue : str | pd.Series | np.ndarray | None
    fill : bool, default=False
        Fill area under curve
    baseline : bool, default=False
        Show diagonal baseline
    legend : bool, default=True
    """
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
        fill=fill,
        color=color,
        legend=legend,
        baseline=baseline,
        **auc_kws,
    )
    return ax
