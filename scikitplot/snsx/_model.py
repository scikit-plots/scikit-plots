# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Model Instance and/or Attributes Visualization.

Supported attributes:

- feature_names_in_
- feature_importances_
- coef_
- eigenvalues_
- eigenvectors_

See Also
--------
FastICA : A fast algorithm for Independent Component Analysis.
IncrementalPCA : Incremental Principal Component Analysis.
NMF : Non-Negative Matrix Factorization.
PCA : Principal component analysis (PCA).
KernelPCA : Kernel Principal component analysis (KPCA).
SparsePCA : Sparse Principal Components Analysis (SparsePCA).
TruncatedSVD : Dimensionality reduction using truncated SVD.
permutation_importance : Permutation importance for feature evaluation [BRE].
partial_dependence : Compute Partial Dependence values.
PartialDependenceDisplay : Partial dependence visualization.
PartialDependenceDisplay.from_estimator : Plot Partial Dependence.
DecisionBoundaryDisplay : Decision boundary visualization.
DecisionBoundaryDisplay.from_estimator : Plot decision boundary given an estimator.

References
----------
.. [BRE] :doi:`L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
   2001. <10.1023/A:1010933404324>`
.. [1] `scikit-learn contributors. (2025).
   "sklearn.decomposition"
   scikit-learn. https://scikit-learn.org/stable/api/sklearn.decomposition.html
   <https://scikit-learn.org/stable/api/sklearn.decomposition.html>`_
.. [2] `scikit-learn contributors. (2025).
   "sklearn.inspection"
   scikit-learn. https://scikit-learn.org/stable/api/sklearn.inspection.html
   <https://scikit-learn.org/stable/api/sklearn.inspection.html>`_
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
# import copy
import warnings
from textwrap import dedent
from typing import ClassVar, Literal

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.utils.multiclass import unique_labels

try:
    import statsmodels

    assert statsmodels  # noqa: S101
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

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
    "modelplot",
]


# ==================================================================================== #
# Module documentation
# ==================================================================================== #

_model_docs = dict(  # noqa: C408
    model_api=dedent(
        """\
        There are a number of mutually exclusive options for estimating the
        regression model. See the :ref:`tutorial <regression_tutorial>` for more
        information.\
        """
    ),
    x_estimator=dedent(
        """\
        x_estimator : model
            An instance so access the fitted model attribute(s).\
        """
    ),
    scatter_line_kws=dedent(
        """\
        {scatter,line}_kws : dictionaries
            Additional keyword arguments to pass to ``plt.scatter`` and
            ``plt.plot``.\
        """
    ),
)
# _model_docs.update(_facet_docs)

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
    model=DocstringComponents(_model_docs),
)


# ==================================================================================== #
# Internal API
# ==================================================================================== #


# --------------------------------------------------------------------
# Core Plotter Class
# --------------------------------------------------------------------
class _ConfusionMatrixPlotter(VectorPlotter):
    """
    Seaborn-style ConfusionMatrix visualization.

    Expects the VectorPlotter pipeline to have mapped incoming data into
    standardized columns "x", "y", optionally grouping (hue), legend creation,
    "weights", and subset iteration.
    """

    # self.variables
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

    # -------------------------
    # Helpers - data extraction
    # ------------------------
    def _prepare_subset(
        self,
        sub_data: pd.DataFrame,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":  # noqa: UP037
        """
        Extract y_true (binary labels) and y_pred (prediction) arrays from a subset DataFrame.

        - Expects that VectorPlotter has already standardized columns to "x" and "y"
          in `sub_data` when `from_comp_data=True` is used in iter_data (this matches
          the approach used in seaborn's VectorPlotter pattern).
        - Enforces types: y_true -> int (0/1), y_pred -> int.

        Returns
        -------
        y_true : ndarray of shape (n_samples,)
        y_pred : ndarray of shape (n_samples,)
        weights : ndarray or None
        """
        # Extract the data points from this sub set
        # compute PR curve: y_true (x_col), y_pred (y_col)
        # Map seaborn-style x/y/hue to our data
        # x_col, y_col = "x", "y"
        # y_true = sub_data[x_col].astype(int)
        # y_pred = sub_data[y_col]

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
            y_pred = np.asarray(sub["y"], dtype=int)
        except Exception as e:
            raise ValueError(f"Cannot convert y to float scores: {e}") from e

        # Extract weights if present
        weights = sub["weights"].to_numpy() if "weights" in sub.columns else None

        return y_true, y_pred, weights

    def _compute_eval(
        self,
        y_true,
        y_pred,
        sample_weight,
        kind,
        sub_vars,
        digits=4,
        labels=None,
        normalize=None,
    ) -> "float | None":  # noqa: UP037
        # return
        xs, ys, ss = {}, {}, {}
        classes = unique_labels(y_true, y_pred)
        labels = classes if labels is None else np.array(labels)
        if kind and kind.lower() in ["all", "classification_report"]:
            try:
                # Generate the classification report
                s = classification_report(
                    y_true,
                    y_pred,
                    labels=labels,
                    digits=digits,
                    zero_division=np.nan,
                )
                xs["classification_report"] = 0
                ys["classification_report"] = 0.5
                ss["classification_report"] = str(s)
            except Exception as e:
                warnings.warn(
                    f"Unable to compute classification_report for subset {sub_vars}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
        if kind and kind.lower() in ["all", "confusion_matrix"]:
            try:
                # Generate the confusion matrix
                cm = np.around(
                    confusion_matrix(
                        y_true,
                        y_pred,
                        labels=labels,
                        normalize=normalize,
                    ),
                    decimals=2,
                )
                xs["confusion_matrix"] = cm
                ys["confusion_matrix"] = None
                ss["confusion_matrix"] = None
            except Exception as e:
                warnings.warn(
                    f"Unable to compute confusion_matrix for subset {sub_vars}: {e}",
                    UserWarning,
                    stacklevel=2,
                )
        return xs, ys, ss

    def _plot_evalplot(
        self,
        xs,
        ys,
        sample_weight,
        classes,
        ss: str,
        kind,
        label_base,
        legend,
        ax,
        cbar,
        cbar_ax,
        cbar_kws,
        text_kws,
        image_kws,
        **kws,
    ):
        label = label_base if legend else None
        kws.setdefault("label", kws.pop("label", label))

        # text_kws = kws.pop('text_kws', {})
        # ha: Horizontal alignment ('left', 'center', 'right').
        # text_kws["ha"] = text_kws.pop("horizontalalignment", text_kws.pop("ha", 'left'))
        text_kws.setdefault(
            "ha", text_kws.pop("horizontalalignment", text_kws.pop("ha", "center"))
        )
        # va: Vertical alignment ('top', 'center', 'bottom').
        text_kws.setdefault(
            "va", text_kws.pop("verticalalignment", text_kws.pop("va", "center"))
        )
        text_kws.setdefault(
            "fontfamily", text_kws.pop("fontfamily", "monospace")
        )  # 'serif'
        # text_kws.setdefault("fontname", text_kws.pop("fontname", mpl.rcParams["font.monospace"]))
        text_kws.setdefault("fontsize", text_kws.pop("fontsize", 8))

        # image_kws = kws.pop('image_kws', {})
        # Choose a colormap
        # image_kws["cmap"] = plt.get_cmap(image_kws.pop("cmap", None))
        image_kws.setdefault("aspect", image_kws.pop("aspect", "auto"))  # "equal"
        image_kws.setdefault("cmap", plt.get_cmap(image_kws.pop("cmap", None)))

        fig = ax.figure
        ax.axis("off")
        # Save artist and label (include statement if legend requested)
        artists, labels = [], []
        axs = {}
        if kind and kind.lower() in ["all"]:
            # fig override subplot define (nrows, ncols, index)
            # int, (int, int, index), or SubplotSpec, default: (1, 1, 1)
            # Each subplot is placed in a grid defined by nrows x ncols at position idx
            # plt.figure(figsize=kws.pop('figsize', (7.5, 1)))
            width, height = fig.get_size_inches()
            width, height = width * 0.85, height / 2.1  # np.log1p(3.5)
            fig.set_size_inches(kws.pop("figsize", (width, height)), forward=True)
            axs["classification_report"] = fig.add_subplot(1, 2, 1)
            axs["confusion_matrix"] = fig.add_subplot(1, 2, 2)
        if "classification_report" in xs:
            # x, y
            x = xs.pop("classification_report", None)
            y = ys.pop("classification_report", None)
            s = ss.pop("classification_report", None)
            # Plot the classification report on the first subplot
            ax = axs.pop("classification_report", ax)
            ax.axis("off")
            artist = ax.text(
                x,
                y,
                s,
                **text_kws,
            )
            # ax.set_aspect('auto')  # not work expected
            # ax.set_xlabel("")
            # ax.set_ylabel("")
            ax.set_title("Classification Report", loc="center")  # x=-1e-1 pad=1
        if "confusion_matrix" in xs:
            # x, y
            x = xs.pop("confusion_matrix", None)
            y = ys.pop("confusion_matrix", None)
            s = ss.pop("confusion_matrix", None)
            # Plot the confusion matrix on the second subplot
            ax = axs.pop("confusion_matrix", ax)
            # artist = ax.matshow(cm, cmap=cmap_, aspect="auto")
            artist = ax.imshow(
                x,
                **image_kws,
            )
            # ax.set_aspect('auto')
            ax.set(
                xlabel="Predicted Labels",
                ylabel="True Labels",
                title="Confusion Matrix",
            )
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(
                classes,
                fontsize=text_kws["fontsize"],
            )
            ax.set_yticklabels(
                classes,
                fontsize=text_kws["fontsize"],
                # rotation=0,
            )
            # ax.grid(True)

            # Remove the edge of the matshow
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Move x-axis labels to the bottom and y-axis labels to the right
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.tick_bottom()
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()

            # Add colorbar
            # from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
            # ax1_divider = make_axes_locatable(ax1)
            # # Add an Axes to the right of the main Axes.
            # cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
            # cb1 = fig.colorbar(im1, cax=cax1)
            cbar = fig.colorbar(mappable=artist, ax=ax)  # shrink=0.8
            # Also remove the colorbar edge
            cbar.outline.set_edgecolor("none")

            # Annotate the matrix with dynamic text color
            threshold = x.max() / 2.0
            for (i, j), val in np.ndenumerate(x):
                # val == cm[i, j]
                cmap_method = (
                    image_kws["cmap"].get_over
                    if val > threshold
                    else image_kws["cmap"].get_under
                )
                # Get the color at the top end of the colormap
                rgba = cmap_method()  # Get the RGB values

                # Calculate the luminance of this color
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]

                # If luminance is low (dark color), use white text; otherwise, use black text
                text_color = {True: "w", False: "k"}[(luminance < 0.5)]  # noqa: PLR2004
                ax.text(
                    j,
                    i,
                    f"{val}",
                    color=text_color,
                    # transform=ax.transAxes,
                    **text_kws,
                )
        # Save artist and label (include statement if legend requested)
        # Get the color cycle from Matplotlib
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        artist = mpatches.Patch(color=colors[0], label=label, zorder=3)
        artists.append(artist), labels.append(label)

        # Add legend (use axes of the first plotted subset)
        # https://matplotlib.org/stable/users/explain/axes/legend_guide.html#controlling-the-legend-entries
        if legend and artists:
            # Build simple legend entries: prefer Line2D proxies if fill used
            ax0 = self.ax if self.ax is not None else ax
            # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
            handles, labels = self._make_legend_proxies(artists, labels)
            ax0.legend(
                handles,
                labels,
                title=self.variables.get("hue", None),
                loc="lower left",
                bbox_to_anchor=(-0.3, -0.1),
            )
        # ax.figure.tight_layout()
        # plt.tight_layout()

    # -------------------------
    # AUC plotting
    # -------------------------
    def plot_evalplot(  # noqa: PLR0912
        self,
        kind: "Literal['all', 'classification_report', 'confusion_matrix'] | None",  # noqa: UP037
        color,
        legend,
        # multiple,
        # element,
        # common_norm,
        # common_grid,
        # common_bins,
        # shrink,
        # kde,
        # kde_kws,
        # line_kws,
        cbar,
        cbar_ax,
        cbar_kws,
        text_kws,
        image_kws,
        verbose=False,
        **plot_kws,
    ):
        # x_col = self.variables.get("x")
        # y_col = self.variables.get("y")
        # # If no x/y data return early
        # if x_col is None or y_col is None:
        #     return
        # orient = self.data_variable

        # -- Default keyword dicts
        cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
        text_kws = {} if text_kws is None else text_kws.copy()
        image_kws = {} if image_kws is None else image_kws.copy()

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
                label_base = "Eval Report"

            # Prepare arrays
            try:
                y_true, y_pred, sample_weight = self._prepare_subset(sub_data)
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
            # classes = unique_labels(y_true, y_pred)  # both sorted
            classes = np.unique(y_true)
            # multiclass = len(classes) > 2  # noqa: PLR2004

            # plt.matshow(a) # Or `plt.imshow(a)`
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.matshow.html#matplotlib.axes.Axes.matshow
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
            x, y, s = self._compute_eval(
                y_true,
                y_pred,
                sample_weight,
                kind,
                sub_vars,
            )
            if x is None:
                continue
            # plot line and optional fill, Use element "line" for curve plotting
            artist_kws = self._artist_kws(
                plot_kws, None, "line", "layer", sub_color, alpha
            )
            # Merge user requested line style/marker if present
            if linestyle is not None:
                artist_kws["linestyle"] = linestyle
            if marker is not None:
                artist_kws["marker"] = marker
            # plot
            self._plot_evalplot(
                x,
                y,
                sample_weight,
                classes,
                s,
                kind,
                label_base,
                legend,
                ax,
                cbar=cbar,
                cbar_ax=cbar_ax,
                cbar_kws=cbar_kws,
                text_kws=text_kws,
                image_kws=image_kws,
                **artist_kws,
            )


# --------------------------------------------------------------------
# Public API functions (wrappers)
# --------------------------------------------------------------------
def modelplot(  # noqa: D417  # evalmap
    data: "pd.DataFrame | None" = None,  # noqa: UP037
    *,
    # Vector variables
    x: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    y: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    hue: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    x_estimator=None,
    kind: "Literal['feature_importances', 'coef'] | None" = None,  # noqa: UP037
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
    cbar=True,
    cbar_ax=None,
    cbar_kws=None,
    # smoothing
    text_kws=None,
    image_kws=None,
    # computation parameters
    digits: "int | None" = None,  # noqa: UP037
    common_norm=None,
    verbose: bool = False,
    **kwargs,
) -> mpl.axes.Axes:
    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
    # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
    # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist
    kind = (kind and kind.lower().strip()) or "feature_importances"
    _check_argument(
        "kind",
        [
            "feature_importances",
            "coef",
        ],
        kind,
    )
    # Build the VectorPlotter and attach data/variables in seaborn style
    p = _ConfusionMatrixPlotter(
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
    eval_kws = kwargs.copy()
    # eval_kws["color"] = color
    # _assign_default_kwargs(eval_kws, p.plot_evalplot, evalplot)
    p.plot_evalplot(
        kind=kind,
        color=color,
        legend=legend,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        text_kws=text_kws,
        image_kws=image_kws,
        **eval_kws,
    )
    return ax


modelplot.__doc__ = """\
Model Instance and/or Attributes Visualization.

Supported attributes:

- feature_names_in_
- feature_importances_
- coef_
- eigenvalues_
- eigenvectors_

Parameters
----------
{params.model.x_estimator}
{params.core.data}
{params.core.xy}
{params.core.hue}
kind : {{'all', 'classification_report', 'confusion_matrix'}} or None, default=None
    Kind of plot to make.
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
{params.dist.cbar}
{params.dist.cbar_ax}
{params.dist.cbar_kws}
text_kws : dict
    Parameters that control the `'classification_report'` visualization, passed to
    :py:func:`~matplotlib.axes.Axes.text`.
image_kws : dict
    Parameters that control the `'confusion_matrix'` visualization, passed to
    :py:func:`~matplotlib.axes.Axes.imshow`.
    Recognized keys:

    cmap : None, str or matplotlib.colors.Colormap, optional, default=None
        Colormap used for plotting.
        Options include 'viridis', 'PiYG', 'plasma', 'inferno', 'nipy_spectral', etc.
        See Matplotlib Colormap documentation for available choices.

        - https://matplotlib.org/stable/users/explain/colors/index.html
        - plt.colormaps()
        - plt.get_cmap()  # None == 'viridis'
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
normalize : {{'true', 'pred', 'all', None}}, optional
    Normalizes the confusion matrix according to the specified mode. Defaults to None.

    - 'true': Normalizes by true (actual) values.
    - 'pred': Normalizes by predicted values.
    - 'all': Normalizes by total values.
    - None: No normalization.
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

See Also
--------
FastICA : A fast algorithm for Independent Component Analysis.
IncrementalPCA : Incremental Principal Component Analysis.
NMF : Non-Negative Matrix Factorization.
PCA : Principal component analysis (PCA).
KernelPCA : Kernel Principal component analysis (KPCA).
SparsePCA : Sparse Principal Components Analysis (SparsePCA).
TruncatedSVD : Dimensionality reduction using truncated SVD.
permutation_importance : Permutation importance for feature evaluation [BRE].
partial_dependence : Compute Partial Dependence values.
PartialDependenceDisplay : Partial dependence visualization.
PartialDependenceDisplay.from_estimator : Plot Partial Dependence.
DecisionBoundaryDisplay : Decision boundary visualization.
DecisionBoundaryDisplay.from_estimator : Plot decision boundary given an estimator.

References
----------
.. [BRE] :doi:`L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
   2001. <10.1023/A:1010933404324>`
.. [1] `scikit-learn contributors. (2025).
   "sklearn.decomposition"
   scikit-learn. https://scikit-learn.org/stable/api/sklearn.decomposition.html
   <https://scikit-learn.org/stable/api/sklearn.decomposition.html>`_
.. [2] `scikit-learn contributors. (2025).
   "sklearn.inspection"
   scikit-learn. https://scikit-learn.org/stable/api/sklearn.inspection.html
   <https://scikit-learn.org/stable/api/sklearn.inspection.html>`_
""".format(  # noqa: UP032
    params=_param_docs,
    returns=_core_docs["returns"],
    # seealso=_core_docs["seealso"],
)
