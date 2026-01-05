# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Confusion Matrix [1]_ and a text report showing the main classification metrics visualization.

This module provides :func:`evalplot`, a seaborn-like wrapper around scikit-learn's
classification evaluation metrics. It can render a confusion matrix heatmap and/or
a text-based classification report.

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
    classification_report,
    confusion_matrix,
)
from sklearn.utils.multiclass import unique_labels
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
except (ImportError, ModuleNotFoundError):
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
    "evalplot",
]


# ==================================================================================== #
# Docstring components (seaborn-style)
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
class _ConfusionMatrixPlotter(VectorPlotter):
    """
    Seaborn-style plotter for evaluation metrics.

    Expects the VectorPlotter pipeline to have mapped incoming data into
    standardized columns "x", "y", optionally grouping (hue), legend creation,
    "weights", and subset iteration.
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
        # self.variables
        variables = {} if variables is None else variables
        super().__init__(data=data, variables=variables)

    @property
    def univariate(self) -> bool:
        """Return True if only x or y are used."""
        # TODO this could go down to core, but putting it here now.
        # We'd want to be conceptually clear that univariate only applies
        # to x/y and not to other semantics, which can exist.
        # We haven't settled on a good conceptual name for x/y.
        return bool({"x", "y"} - set(self.variables))

    @property
    def data_variable(self) -> str:
        """Return the variable with data for univariate plots."""
        # TODO This could also be in core, but it should have a better name.
        if not self.univariate:
            raise AttributeError("This is not a univariate plot")
        return {"x", "y"}.intersection(self.variables).pop()

    @property
    def has_xy_data(self) -> bool:
        """Return True at least one of x or y is defined."""
        # TODO see above points about where this should go
        return bool({"x", "y"} & set(self.variables))

    # -----------------------------------------------------------------
    # Seaborn helpers (kept for consistency with VectorPlotter)
    # -----------------------------------------------------------------
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
        """Normalize artist kwargs for filled/unfilled plotting."""
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

    # -----------------------------------------------------------------
    # Data extraction / validation
    # -----------------------------------------------------------------
    def _prepare_subset(  # noqa: PLR0912
        self,
        sub_data: pd.DataFrame,
        *,
        threshold: float = 0.5,
        allow_probs: bool = True,
        labels=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """
        Extract y_true (binary labels) and y_pred (prediction) arrays from a subset DataFrame.

        - Expects that VectorPlotter has already standardized columns to "x" and "y"
          in `sub_data` when `from_comp_data=True` is used in iter_data (this matches
          the approach used in seaborn's VectorPlotter pattern).
        - Enforces types: y_true -> int (0/1), y_score -> float.
        - Internally y_pred -> int.

        Parameters
        ----------
        sub_data : DataFrame
            Subset data from :meth:`VectorPlotter.iter_data` with
            ``from_comp_data=True``.
        threshold : float
            Threshold used to convert probabilities into predicted labels when
            ``allow_probs=True``.
        allow_probs : bool
            If True, interpret ``y`` as probabilities in ``[0, 1]`` and threshold
            into predicted labels. Requires binary classification.
        labels : array-like, optional
            Label ordering forwarded to scikit-learn. When ``allow_probs=True``,
            this must be length 2 and is used to define the negative/positive class.

        Returns
        -------
        y_true : ndarray of shape (n_samples,)
        y_score or y_pred : ndarray of shape (n_samples,)
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
        required = ["x", "y"]
        subset_cols = required + (["weights"] if "weights" in sub_data.columns else [])
        sub = sub_data.dropna(subset=subset_cols)
        if sub.empty:
            return None, None, None

        try:
            # Coerce true labels to integers array (0/1 for binary classification)
            y_true = np.asarray(sub["x"], dtype=int)  # .astype(int)
        except Exception as e:
            raise ValueError(f"Cannot convert x to integer labels: {e}") from e

        try:
            # Scores must be float
            y_in = np.asarray(sub["y"], dtype=float)  # .astype(float)
        except Exception as e:
            raise ValueError(f"Cannot convert y to float scores: {e}") from e

        # Extract weights if present
        weights = None
        if "weights" in sub.columns:
            weights = sub["weights"].to_numpy(dtype=float)
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            if not np.isfinite(weights).all():
                raise ValueError("weights contains NaN or infinite values.")

        if np.issubdtype(y_true.dtype, np.number) and not np.isfinite(y_true).all():
            raise ValueError("x contains NaN or infinite values.")
        if np.issubdtype(y_in.dtype, np.number) and not np.isfinite(y_in).all():
            raise ValueError("y contains NaN or infinite values.")

        if allow_probs:
            y_score = np.asarray(y_in, dtype=float)
            if np.any((y_score < 0) | (y_score > 1)):
                raise ValueError("y must be within [0, 1] when allow_probs=True.")

            classes = unique_labels(y_true) if labels is None else np.asarray(labels)

            if len(classes) != 2:  # noqa: PLR2004
                raise ValueError(
                    "allow_probs=True requires binary classification (2 unique labels)."  # noqa: E501
                )

            neg_label, pos_label = classes[0], classes[1]
            y_pred = np.where(y_score > threshold, pos_label, neg_label)
        else:
            # Interpret y as predicted labels
            y_pred = y_in

        if weights is None:
            check_consistent_length(y_true, y_pred)
        else:
            check_consistent_length(y_true, y_pred, weights)

        return y_true, y_pred, weights

    # -----------------------------------------------------------------
    # Metric computation
    # -----------------------------------------------------------------
    def _compute_classification_report(
        self,
        y_true,
        y_pred,
        sample_weight=None,
        labels=None,
        digits: int = 4,
        normalize: "Literal['true', 'pred', 'all'] | None" = None,  # noqa: UP037
        # sub_vars=None,
    ) -> "tuple[int, float, str] | None":  # noqa: UP037
        """Compute a scikit-learn classification report string."""
        try:
            # Generate the classification report
            s = classification_report(
                y_true,
                y_pred,
                sample_weight=sample_weight,
                labels=labels,
                digits=digits,
                zero_division=np.nan,
            )
            return 0, 0.5, str(s)
        except Exception as e:
            warnings.warn(
                f"Unable to compute classification_report for subset : {e}",
                UserWarning,
                stacklevel=2,
            )
            return None, None, None

    def _compute_confusion_matrix(
        self,
        y_true,
        y_pred,
        sample_weight=None,
        labels=None,
        digits: int = 4,
        normalize: "Literal['true', 'pred', 'all'] | None" = None,  # noqa: UP037
        # sub_vars=None,
    ) -> "tuple[np.ndarray, None, None] | None":  # noqa: UP037
        """Compute a scikit-learn confusion matrix."""
        try:
            # Generate the confusion matrix
            cm = np.around(
                confusion_matrix(
                    y_true,
                    y_pred,
                    sample_weight=sample_weight,
                    labels=labels,
                    normalize=normalize,
                ),
                decimals=2,
            )
            return cm, None, None
        except Exception as e:
            warnings.warn(
                f"Unable to compute confusion_matrix for subset : {e}",
                UserWarning,
                stacklevel=2,
            )
            return None, None, None

    # -----------------------------------------------------------------
    # Plotting primitives
    # -----------------------------------------------------------------
    def _plot_classification_report(
        self,
        y_true,
        y_pred,
        sample_weight,
        classes,
        labels,
        legend,
        ax,
        cbar_kws,
        cbar,
        cbar_ax,
        text_kws,
        image_kws,
        annot_kws,
        annot,
        fmt,
        digits,
        **kws,
    ):
        """Render the classification report into an axes."""
        # x, y
        x, y, s = self._compute_classification_report(
            y_true,
            y_pred,
            sample_weight,
            labels,
            digits=int(digits) if digits is not None else 4,
        )
        # text_kws = kws.pop('text_kws', {})
        # ha: Horizontal alignment ('left', 'center', 'right').
        # text_kws["ha"] = text_kws.pop("horizontalalignment", text_kws.pop("ha", 'left'))
        text_kws = {} if text_kws is None else text_kws.copy()
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

        # Save artist and label (include statement if legend requested)
        # artists, labels = [], []
        # Plot the classification report on the first subplot
        ax.axis("off")
        artist = ax.text(
            x,
            y,
            s if s is not None else "Unable to compute classification report.",
            # transform=ax.transAxes,
            **text_kws,
        )
        # ax.set_aspect('auto')  # not work expected
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        ax.set_title("Classification Report", loc="center")  # x=-1e-1 pad=1

    def _plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        sample_weight,
        classes,
        labels,
        legend,
        ax,
        cbar_kws,
        cbar,
        cbar_ax,
        text_kws,
        image_kws,
        annot_kws,
        annot,
        fmt,
        digits,
        **kws,
    ):
        """Plot confusion matrix as a heatmap with optional annotations."""
        normalize = kws.pop("normalize", None)
        # x, y
        x, _, _ = self._compute_confusion_matrix(
            y_true,
            y_pred,
            sample_weight,
            labels,
            digits,
            normalize=normalize,
        )
        if x is None:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Unable to compute confusion matrix.",
                ha="center",
                va="center",
                # transform=ax.transAxes,
            )
            return
        # text_kws = kws.pop('text_kws', {})
        # ha: Horizontal alignment ('left', 'center', 'right').
        # text_kws["ha"] = text_kws.pop("horizontalalignment", text_kws.pop("ha", 'left'))
        text_kws = {} if text_kws is None else text_kws.copy()
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
        image_kws = {} if image_kws is None else image_kws.copy()
        image_kws.setdefault("aspect", image_kws.pop("aspect", "auto"))  # "equal"
        image_kws.setdefault("cmap", plt.get_cmap(image_kws.pop("cmap", None)))

        if normalize is not None:
            image_kws.setdefault("vmin", 0.0)
            image_kws.setdefault("vmax", 1.0)

        fig = ax.figure
        ax.axis("off")
        # ax.grid(True)

        # Save artist and label (include statement if legend requested)
        # artists, labels = [], []
        # Plot the confusion matrix on the second subplot
        # artist = ax.matshow(cm, cmap=cmap_, aspect="auto")
        artist = ax.imshow(
            x,
            **image_kws,
        )
        # ax.set_aspect('auto')
        ax.set(
            xlabel="Predicted Label",
            ylabel="True Label",
            title="Confusion Matrix",
        )

        classes = np.asarray(classes)
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

        # Move x-axis labels to the bottom and y-axis labels to the right
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()

        # Remove the edge of the matshow
        # ax.spines["top"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add colorbar
        # from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        # ax1_divider = make_axes_locatable(ax1)
        # # Add an Axes to the right of the main Axes.
        # cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        # cb1 = fig.colorbar(im1, cax=cax1)
        if cbar:  # bool
            cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
            cb = fig.colorbar(
                mappable=artist, ax=ax, cax=cbar_ax, **cbar_kws
            )  # shrink=0.8
            # Also remove the colorbar edge
            cb.outline.set_edgecolor("none")

        # if annot and not fmt:
        #     fmt = ".2f"

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
            # rgba = artist.cmap(artist.norm(val))
            rgba = cmap_method()  # Get the RGB values

            # Calculate the luminance of this color
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            # If luminance is low (dark color), use white text; otherwise, use black text
            text_color = {True: "w", False: "k"}[(luminance < 0.5)]  # noqa: PLR2004

            ax.text(
                j,
                i,
                f"{val}",  # format(val, fmt) if fmt else f"{val}"
                color=text_color,
                # transform=ax.transAxes,
                **text_kws,
            )

    def _plot_all(
        self,
        y_true,
        y_pred,
        _sw,
        classes,
        labels,
        legend,
        ax,
        cbar_kws,
        cbar,
        cbar_ax,
        text_kws,
        image_kws,
        annot_kws,
        annot,
        fmt,
        digits,
        **kws,
    ):
        """Render a 1x2 dashboard: report + confusion matrix."""
        # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        # Get the parent figure from the given Axes
        fig = ax.figure
        # ax.axis("off")
        # fig override subplot define (nrows, ncols, index)
        # int, (int, int, index), or SubplotSpec, default: (1, 1, 1)
        # Each subplot is placed in a grid defined by nrows x ncols at position idx
        # plt.figure(figsize=kws.pop('figsize', (7.5, 1)))
        width, height = fig.get_size_inches()
        width, height = width * 0.85, height / 2.1  # np.log1p(3.5)
        fig.set_size_inches(kws.pop("figsize", (width, height)), forward=True)

        # Replace single Axes with a 2x2 grid inside the same figure
        # plt.gcf().clf()  # clear any existing axes, clar figure
        # plt.clf()  # clear any existing axes, clar figure
        fig.clf()  # clear any existing axes
        axes = fig.subplots(1, 2)
        # Draw plots
        for _ax, kind in zip(
            (axes[0], axes[1]), ("classification_report", "confusion_matrix")
        ):
            # plt.matshow(a) # Or `plt.imshow(a)`
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.matshow.html#matplotlib.axes.Axes.matshow
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
            _plot = getattr(self, f"_plot_{kind}", None)
            if _plot:
                _plot(
                    y_true,
                    y_pred,
                    _sw,
                    classes,
                    labels,
                    legend,
                    _ax,
                    cbar_kws,
                    cbar,
                    cbar_ax,
                    text_kws,
                    image_kws,
                    annot_kws,
                    annot,
                    fmt,
                    digits,
                    **kws,
                )
        # fig.tight_layout()

        # # Save artist and label (include statement if legend requested)
        # artists, labels = [], []
        # # Save artist and label (include statement if legend requested)
        # # Get the color cycle from Matplotlib
        # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # artist = mpatches.Patch(color=colors[0], label="eval", zorder=3)
        # # ("{:" + self.fmt + "}").format(val)
        # # {val:fmt} If nested formatting in Python f-strings f"{val:{fmt}}"
        # # Double braces {{ or }} → escape to literal { or }.
        # artists.append(artist), labels.append("eval")

        # # Add legend (use axes of the first plotted subset)
        # # https://matplotlib.org/stable/users/explain/axes/legend_guide.html#controlling-the-legend-entries
        # if legend and artists:
        #     # Build simple legend entries: prefer Line2D proxies if fill used
        #     ax0 = self.ax if self.ax is not None else ax
        #     # ax0.legend(artists, [lbl for lbl in labels], title=self.variables.get("hue", None))
        #     handles, labels = self._make_legend_proxies(artists, labels)
        #     ax0.legend(
        #         handles,
        #         labels,
        #         title=self.variables.get("hue", None),
        #         loc="lower left",
        #         bbox_to_anchor=(-0.3, -0.1),
        #     )

    def assert_binary_compat(  # noqa: PLR0912
        self,
        y_true,
        y_score=None,
        y_pred=None,
        threshold=0.5,
        allow_probs=False,
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
    # Evaluation plotting
    # -------------------------
    def plot_evalplot(  # noqa: PLR0912
        self,
        kind: "Literal['all', 'classification_report', 'confusion_matrix'] | None",  # noqa: UP037
        labels,
        threshold,
        allow_probs,
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
        cbar_kws,
        cbar,
        cbar_ax,
        text_kws,
        image_kws,
        annot_kws,
        annot,
        fmt,
        digits,
        # normalize: "Literal['true', 'pred', 'all'] | None" = None,  # noqa: UP037
        verbose=False,
        **plot_kws,
    ) -> None:
        """Draw evaluation plots for each semantic subset."""
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
        # label = plot_kws.pop("label", "")
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

            annot_kws.setdefault(
                "color", annot_kws.pop("color", annot_kws.pop("c", sub_color))
            )

            # plot line and optional fill, Use element "line" for curve plotting
            artist_kws = self._artist_kws(
                plot_kws, None, "line", "layer", sub_color, alpha
            )
            label = label_base if legend else None
            artist_kws.setdefault("label", artist_kws.pop("label", label))
            # Merge user requested line style/marker if present
            if linestyle is not None:
                artist_kws["linestyle"] = linestyle
            if marker is not None:
                artist_kws["marker"] = marker

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
            y_true, _, y_pred = self.assert_binary_compat(
                y_true,
                y_score,
                threshold=threshold,
                allow_probs=allow_probs,
            )
            if y_true is None:
                continue

            # binary vs multiclass detection
            # classes = np.unique(y_true)
            classes = unique_labels(y_true, y_pred)  # both sorted
            labels = classes if labels is None else np.array(labels)
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
                # plt.matshow(a) # Or `plt.imshow(a)`
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.matshow.html#matplotlib.axes.Axes.matshow
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html
                _plot = getattr(self, f"_plot_{kind}", None)
                if _plot:
                    _plot(
                        y_true,
                        y_pred,
                        _sw,
                        classes,
                        labels,
                        legend,
                        ax,
                        cbar_kws,
                        cbar,
                        cbar_ax,
                        text_kws,
                        image_kws,
                        annot_kws,
                        annot,
                        fmt,
                        digits,
                        **artist_kws,
                    )


# --------------------------------------------------------------------
# Public API functions (wrappers)
# --------------------------------------------------------------------
def evalplot(  # noqa: D417  # evalmap
    data: "pd.DataFrame | None" = None,  # noqa: UP037
    *,
    # Vector variables
    x: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    y: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    hue: "str | np.ndarray[np.generic] | pd.Series | None" = None,  # noqa: UP037
    kind: "Literal['all', 'classification_report', 'confusion_matrix'] | None" = None,  # noqa: UP037
    weights=None,
    labels=None,
    threshold: float = 0.5,
    allow_probs: bool = False,
    # normalize: "Literal['true', 'pred', 'all'] | None" = None,  # noqa: UP037
    # Hue mapping parameters
    hue_order=None,
    hue_norm=None,
    palette=None,
    color=None,
    # Appearance keywords (reserved for API compatibility)
    fill=False,
    # Other appearance keywords
    baseline=False,
    # smoothing
    line_kws=None,
    # Axes information
    log_scale=None,
    legend=False,
    ax=None,
    cbar_kws=None,
    cbar=True,
    cbar_ax=None,
    # smoothing
    text_kws=None,
    image_kws=None,
    annot_kws=None,
    annot=True,
    fmt="",
    # computation parameters
    digits: "int | None" = 4,  # noqa: UP037
    common_norm=None,
    verbose: bool = False,
    **kwargs,
) -> mpl.axes.Axes:
    """Plot confusion matrix / classification report with a seaborn-like API."""
    # https://rsted.info.ucl.ac.be/
    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
    # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
    # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist
    kind = (kind and kind.lower().strip()) or "all"
    _check_argument(
        "kind",
        [
            "all",
            "classification_report",
            "confusion_matrix",
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
        labels=labels,
        threshold=threshold,
        allow_probs=allow_probs,
        color=color,
        legend=legend,
        cbar_kws=cbar_kws,
        cbar=cbar,
        cbar_ax=cbar_ax,
        text_kws=text_kws,
        image_kws=image_kws,
        annot_kws=annot_kws,
        annot=annot,
        fmt=fmt,
        digits=digits or 2,
        **eval_kws,
    )
    return ax


evalplot.__doc__ = """\
Visualization of the Confusion Matrix [1]_ alongside a text report showing key classification metrics.

For guidance on interpreting these plots, refer to the
`Model Evaluation Guide <https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix>`_.

Parameters
----------
{params.core.data}
{params.core.xy}
{params.core.hue}
kind : {{'all', 'classification_report', 'confusion_matrix'}} or None, default=None
    Which visualization to draw.

    - ``'classification_report'``: text report from
      :func:`sklearn.metrics.classification_report`.
    - ``'confusion_matrix'``: heatmap from :func:`sklearn.metrics.confusion_matrix`.
    - ``'all'``: a 1x2 dashboard (classification report + confusion matrix).
weights : vector or key in ``data``
    Sample weights passed to the underlying scikit-learn metric functions.
labels : array-like, optional
    Class label ordering. When provided, it is forwarded to
    :func:`sklearn.metrics.confusion_matrix` and :func:`sklearn.metrics.classification_report`.
threshold : float, default=0.5
    Threshold used to convert probabilities into predicted class labels when
    ``allow_probs=True``.
allow_probs : bool, default=False
    If True, interpret ``y`` as probabilities in ``[0, 1]`` and derive predicted
    labels via ``y > threshold``. This requires binary classification.
    Behavior like 'y > thr' see :py:func:`numpy.argmax`.
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
{params.dist.cbar_kws}
{params.dist.cbar}
{params.dist.cbar_ax}
text_kws : dict, optional
    Keyword arguments passed to :meth:`matplotlib.axes.Axes.text` when rendering
    the classification report (and for confusion-matrix annotations).
image_kws : dict, optional
    Keyword arguments passed to :meth:`matplotlib.axes.Axes.imshow` when drawing
    the confusion matrix.
    Recognized keys:

    cmap : None, str or matplotlib.colors.Colormap, optional, default=None
        Colormap used for plotting.
        Options include 'viridis', 'PiYG', 'plasma', 'inferno', 'nipy_spectral', etc.
        See Matplotlib Colormap documentation for available choices.

        - https://matplotlib.org/stable/users/explain/colors/index.html
        - plt.colormaps()
        - plt.get_cmap()  # None == 'viridis'
annot_kws : dict of key, value mappings, optional
    Keyword arguments for :meth:`matplotlib.axes.Axes.text` when ``annot``
    is True.
annot : bool or rectangular dataset, optional
    If True, write the data value in each cell. If an array-like with the
    same shape as ``data``, then use this to annotate the heatmap instead
    of the data. Note that DataFrames will match on position, not index.
fmt : str, optional, default=''
    Formatting spec for confusion matrix annotations (e.g., ``'.2f'``). When
    ``normalize`` is not None and ``fmt`` is empty, it defaults to ``'.2f'``.
digits : int, optional, default=4
    Number of digits for formatting output floating point values.
    When ``output_dict`` is ``True``, this will be ignored and the
    returned values will not be rounded.
normalize : {{'true', 'pred', 'all', None}}, optional, default=None
    Normalization mode passed to :func:`sklearn.metrics.confusion_matrix` when
    ``kind`` includes ``'confusion_matrix'``.

    - 'true': Normalizes by true (actual) values.
    - 'pred': Normalizes by predicted values.
    - 'all': Normalizes by total values.
    - None: No normalization.
common_norm : bool
    If True, scale each conditional density by the number of observations
    such that the total area under all densities sums to 1. Otherwise,
    normalize each density independently.
zero_division : {{"warn", 0.0, 1.0, np.nan}}, default="warn"
    Sets the value to return when there is a zero division. If set to
    "warn", this acts as 0, but warnings are also raised.
output_dict : bool, default=False
    If True, return output as dict.
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

See Also
--------
sklearn.metrics.confusion_matrix
sklearn.metrics.classification_report

Notes
-----
- When ``kind='all'``, a 1x2 dashboard is created. If the provided ``ax`` is not
  the only axes in its figure, a new figure is created.
- If multiple semantic subsets are present (e.g., multiple ``hue`` levels),
  confusion-matrix-style plots do not overlay cleanly; only the first subset is
  plotted and a warning is emitted.

References
----------
.. [1] `scikit-learn contributors. (2025). "sklearn.metrics"
    scikit-learn documentation. https://scikit-learn.org/stable/api/sklearn.metrics.html
    <https://scikit-learn.org/stable/api/sklearn.metrics.html>`_
""".format(  # noqa: UP032
    params=_param_docs,
    returns=_core_docs["returns"],
    # seealso=_core_docs["seealso"],
)
