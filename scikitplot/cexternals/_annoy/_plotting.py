# scikitplot/cexternals/_annoy/_plotting.py
"""
Plotting and visualization utilities.

This module is based on helper code from *openTSNE* and is vendored into
scikit-plots to provide small, dependency-minimal helpers for common embedding
visualization tasks. It also includes utilities for visualizing Annoy
(approximate nearest-neighbor) indices.

Matplotlib is imported lazily inside plotting functions so importing this module
does not require a graphical backend.

See Also
--------
openTSNE
    Upstream project from which parts of this module originated.
    https://github.com/pavlin-policar/openTSNE/blob/master/examples/utils.py
"""

from os.path import abspath, dirname, join

import numpy as np
import scipy.sparse as sp

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy.typing as npt
except Exception:  # pragma: no cover
    npt = Any  # type: ignore[]

FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}
ZEISEL_COLORS = {
    "Astroependymal cells": "#d7abd4",
    "Cerebellum neurons": "#2d74bf",
    "Cholinergic, monoaminergic and peptidergic neurons": "#9e3d1b",
    "Di- and mesencephalon neurons": "#3b1b59",
    "Enteric neurons": "#1b5d2f",
    "Hindbrain neurons": "#51bc4c",
    "Immature neural": "#ffcb9a",
    "Immune cells": "#768281",
    "Neural crest-like glia": "#a0daaa",
    "Oligodendrocytes": "#8c7d2b",
    "Peripheral sensory neurons": "#98cc41",
    "Spinal cord neurons": "#c52d94",
    "Sympathetic neurons": "#11337d",
    "Telencephalon interneurons": "#ff9f2b",
    "Telencephalon projecting neurons": "#fea7c1",
    "Vascular cells": "#3d672d",
}
MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",
}

__all__ = [
    "MACOSKO_COLORS",
    "ZEISEL_COLORS",
    "MOUSE_10X_COLORS",
    "plot",
    "evaluate_embedding",
    "annoy_index_to_array",
    "project_to_2d",
    "plot_annoy_index",
    "plot_annoy_knn_edges",
    "calculate_cpm",
    "log_normalize",
    "pca",
    "select_genes",
    "maxabs_scale_dense",
    "l2_normalize_rows",
    "pca_project_2d",
]


def calculate_cpm(x: Union[np.ndarray, sp.spmatrix], axis: int = 1) -> Union[np.ndarray, sp.spmatrix]:
    """
    Calculate counts-per-million (CPM).

    CPM rescales values along the specified axis so that each row/column sums to
    ``1_000_000``. This helper supports both dense NumPy arrays and SciPy sparse
    matrices without forcing sparse inputs to densify.

    Parameters
    ----------
    x : array_like or scipy.sparse.spmatrix
        Count matrix.
    axis : int, default=1
        Axis along which totals are computed before scaling.

        * ``axis=0`` scales columns
        * ``axis=1`` scales rows

    Returns
    -------
    cpm : numpy.ndarray or scipy.sparse.spmatrix
        CPM-normalized matrix with the same shape as ``x``. Dense inputs return
        a dense ``numpy.ndarray``; sparse inputs return a sparse matrix.

    Notes
    -----
    This function is deterministic and defines CPM for zero-total rows/columns as
    all zeros (i.e., scale factor ``0``). This avoids divisions by zero and keeps
    sparse inputs sparse.

    For sparse inputs, direct elementwise division would densify the matrix.
    Instead, CPM is expressed as multiplication by a diagonal matrix of per-row
    (or per-column) scale factors.

    See Also
    --------
    log_normalize : Log2-transform values with ``log2(x + 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> counts = np.random.poisson(1.0, size=(100, 500))   # (cells, genes)
    >>> cpm = utils.calculate_cpm(counts)  # dense
    >>> cpm.shape
    (100, 500)
    ...
    >>> counts_sp = sp.csr_matrix(counts)
    ...
    >>> cpm_sp = utils.calculate_cpm(counts_sp)  # sparse
    >>> cpm_sp.shape
    (100, 500)
    """
    normalization = np.sum(x, axis=axis)
    if axis not in (0, 1):
        raise ValueError("`axis` must be 0 or 1. Got {!r}.".format(axis))

    # Compute totals along the chosen axis.
    totals = np.sum(x, axis=axis)

    # On sparse matrices, the sum is 2D; squeeze to a 1D array of totals.
    totals = np.squeeze(np.asarray(totals)).astype(np.float64, copy=False)

    # CPM scale factor: 1e6 / totals, with a defined value for totals==0.
    scale = np.zeros_like(totals, dtype=np.float64)
    nonzero = totals != 0
    scale[nonzero] = 1e6 / totals[nonzero]

    if sp.issparse(x):
        # Keep sparse inputs sparse: apply scaling via diagonal multiplication.
        D = sp.diags(scale, offsets=0, format="csr")
        return (x @ D) if axis == 0 else (D @ x)

    # Dense inputs: apply scaling via broadcasting.
    x_arr = np.asarray(x, dtype=np.float64)
    return (x_arr * scale) if axis == 0 else (x_arr * scale[:, None])


def log_normalize(data: Union[np.ndarray, sp.spmatrix]) -> Union[np.ndarray, sp.spmatrix]:
    """
    Apply a log2 transform ``log2(x + 1)``.

    Parameters
    ----------
    data : array_like or scipy.sparse.spmatrix
        Input values. Sparse inputs are handled without densifying.

    Returns
    -------
    out : numpy.ndarray or scipy.sparse.spmatrix
        Log2-transformed values with the same shape as ``data``.

    Notes
    -----
    For sparse matrices, only the stored non-zero data are transformed (the
    implicit zeros remain zeros).

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> counts = np.random.poisson(1.0, size=(100, 500))   # (cells, genes)
    >>> counts_sp = sp.csr_matrix(counts)
    ...
    >>> log_cpm_sp = utils.log_normalize(counts_sp)     # works for dense or sparse
    >>> log_cpm_sp
    """
    if sp.issparse(data):
        data = data.copy()
        # Only transform stored values; implicit zeros remain zeros.
        data.data = np.log2(data.data + 1)
        return data

    return np.log2(data.astype(np.float64) + 1)

def pca(x: Union[np.ndarray, sp.spmatrix], n_components: int = 50) -> np.ndarray:
    """
    Compute a deterministic PCA embedding via SVD.

    This helper performs a thin SVD on ``x`` and returns the first
    ``n_components`` principal component scores. Sparse inputs are converted to
    dense arrays before SVD.

    Parameters
    ----------
    x : array_like or scipy.sparse.spmatrix, shape (n_samples, n_features)
        Input data.
    n_components : int, default=50
        Number of components to return.

    Returns
    -------
    x_reduced : numpy.ndarray, shape (n_samples, n_components)
        Principal component scores.

    Notes
    -----
    SVD is unique up to a sign flip. To make the output deterministic across
    platforms, the sign of each component is fixed based on the sum of the
    corresponding right-singular vector.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> counts = np.random.poisson(1.0, size=(100, 500))   # (cells, genes)
    >>> cpm = utils.calculate_cpm(counts)  # (dense or sparse)
    ...
    >>> log_cpm = utils.log_normalize(cpm)
    >>> z = utils.pca(log_cpm, n_components=50)  # shape (n_samples, 50)
    >>> z
    """
    if sp.issparse(x):
        x = x.toarray()

    # Thin SVD: x = U * diag(S) * V
    U, S, V = np.linalg.svd(x, full_matrices=False)

    # Make component signs deterministic: flip columns where V's row-sum is negative.
    U[:, np.sum(V, axis=1) < 0] *= -1

    x_reduced = np.dot(U, np.diag(S))
    # Keep behaviour consistent with upstream code: sort by singular values and slice.
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced

def select_genes(
    data,
    threshold=0,
    atleast=10,
    yoffset=0.02,
    xoffset=5,
    decay=1,
    n=None,
    plot=True,
    markers=None,
    genes=None,
    figsize=(6, 3.5),
    markeroffsets=None,
    labelsize=10,
    alpha=1,
) -> np.ndarray:
    """
    Select genes using a mean-expression vs zero-rate rule.

    This helper implements a deterministic selection rule commonly used for
    single-cell gene-expression matrices. For each gene (column), it computes:

    * ``zeroRate``: fraction of cells with expression ``<= threshold``
    * ``meanExpr``: mean log2-expression among detected cells

    Genes with fewer than ``atleast`` detected cells are excluded. A gene is
    selected when its ``zeroRate`` lies above an exponential curve parameterized by
    ``xoffset``, ``yoffset`` and ``decay``.

    If ``n`` is provided, ``xoffset`` is adjusted by a bounded binary search until
    exactly ``n`` genes are selected (when possible).

    Parameters
    ----------
    data : array_like or scipy.sparse.spmatrix, shape (n_cells, n_genes)
        Expression matrix with genes in columns.
    threshold : float, default=0
        Detection threshold. Values ``> threshold`` are treated as detected.
    atleast : int, default=10
        Minimum number of detected cells required for a gene to be considered.
    yoffset : float, default=0.02
        Vertical offset of the selection curve.
    xoffset : float, default=5
        Horizontal offset of the selection curve.
    decay : float, default=1
        Exponential decay parameter of the selection curve.
    n : int or None, default=None
        Target number of genes to select. When provided, ``xoffset`` is adjusted to
        reach the target.
    plot : bool, default=True
        If True, draw a diagnostic scatter plot and the selection curve using
        Matplotlib.
    markers : sequence of int or None, default=None
        Indices of specific genes to annotate on the plot.
    genes : sequence of str or None, default=None
        Gene names used when annotating markers.
    figsize : tuple, default=(6, 3.5)
        Figure size used when ``plot=True``.
    markeroffsets : mapping or None, default=None
        Optional mapping ``marker_index -> (dx, dy)`` controlling text offsets for
        marker labels.
    labelsize : int, default=10
        Font size for annotations.
    alpha : float, default=1
        Alpha (transparency) used for plotting points.

    Returns
    -------
    selected : numpy.ndarray, dtype=bool, shape (n_genes,)
        Boolean mask indicating which genes are selected.

    Notes
    -----
    Matplotlib is imported lazily when ``plot=True``.

    See Also
    --------
    calculate_cpm : Counts-per-million (CPM) normalization.
    log_normalize : Log2-transform values with ``log2(x + 1)``.
    pca : Deterministic PCA embedding via SVD.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> counts = np.random.poisson(1.0, size=(100, 500))   # (cells, genes)
    >>> selected = utils.select_genes(counts, plot=False)  #  given inputs/params
    >>> # selected is a 1D boolean mask over genes (columns)
    >>> selected.sum()
    """
    if sp.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (
            1 - zeroRate[detected]
        )
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.nanmean(
            np.where(data[:, detected] > threshold, np.log2(data[:, detected]), np.nan),
            axis=0,
        )

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    # lowDetection = (1 - zeroRate) * data.shape[0] < atleast - .00001
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = (
                zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            )
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        print("Chosen offset: {:.2f}".format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = (
            zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
        )

    if plot:
        import matplotlib.pyplot as plt

        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold > 0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1] + 0.1, 0.1)
        y = np.exp(-decay * (x - xoffset)) + yoffset
        if decay == 1:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-x+{:.2f})+{:.2f}".format(
                    np.sum(selected), xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )
        else:
            plt.text(
                0.4,
                0.2,
                "{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}".format(
                    np.sum(selected), decay, xoffset, yoffset
                ),
                color="k",
                fontsize=labelsize,
                transform=plt.gca().transAxes,
            )

        plt.plot(x, y, linewidth=2)
        xy = np.concatenate(
            (
                np.concatenate((x[:, None], y[:, None]), axis=1),
                np.array([[plt.xlim()[1], 1]]),
            )
        )
        t = plt.matplotlib.patches.Polygon(xy, color="r", alpha=0.2)
        plt.gca().add_patch(t)

        plt.scatter(meanExpr, zeroRate, s=3, alpha=alpha, rasterized=True)
        if threshold == 0:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of zero expression")
        else:
            plt.xlabel("Mean log2 nonzero expression")
            plt.ylabel("Frequency of near-zero expression")
        plt.tight_layout()

        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num, g in enumerate(markers):
                i = np.where(genes == g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color="k")
                dx, dy = markeroffsets[num]
                plt.text(
                    meanExpr[i] + dx + 0.1,
                    zeroRate[i] + dy,
                    g,
                    color="k",
                    fontsize=labelsize,
                )

    return selected


def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    *,
    figsize=(8, 8),
    axis_off=True,
    scatter_kwargs=None,
    center_kwargs=None,
    text_kwargs=None,
    legend_handle_kwargs=None,
    **kwargs,
):
    """
    Plot a 2D embedding with categorical labels.

    This function is intentionally lightweight and dependency-minimal. It is used
    as a building block by higher-level plotting helpers (e.g., Annoy index
    plotting) while remaining backwards-compatible with the original openTSNE
    example helper.

    Parameters
    ----------
    x : array_like, shape (n_samples, 2)
        2D coordinates to plot.
    y : array_like, shape (n_samples,)
        Labels used for coloring points.
    ax : matplotlib.axes.Axes, optional
        Target axes. If ``None``, a new figure and axes are created.
    title : str, optional
        Axes title.
    draw_legend : bool, default=True
        Whether to draw a legend.
    draw_centers : bool, default=False
        Whether to draw per-class medoid centers (median in 2D).
    draw_cluster_labels : bool, default=False
        Whether to draw class labels above medoid centers. Only used if
        ``draw_centers=True``.
    colors : dict, optional
        Mapping ``label -> color``. If ``None``, a Matplotlib color cycle is used.
    legend_kwargs : dict, optional
        Keyword arguments forwarded to ``ax.legend``.
    label_order : sequence, optional
        Explicit ordering of class labels in the legend. If provided, every label
        in ``y`` must appear in ``label_order``.
    figsize : tuple, default=(8, 8)
        Figure size (inches) when ``ax`` is ``None``.
    axis_off : bool, default=True
        If True, hide ticks and turn the axis frame off.
    scatter_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.scatter`` for the main points.
        This is the preferred customization mechanism instead of passing
        arbitrary ``**kwargs``.
    center_kwargs : dict, optional
        Extra keyword arguments forwarded to the medoid center ``ax.scatter``.
    text_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.text`` when cluster labels are
        drawn.
    legend_handle_kwargs : dict, optional
        Extra keyword arguments forwarded to the legend handle constructor
        (``matplotlib.lines.Line2D``).
    **kwargs
        Backwards-compatible legacy keyword arguments. The following are
        interpreted:
        - ``alpha``: point transparency (default 0.6)
        - ``s``: marker size (default 1)
        - ``fontsize``: cluster label font size (default 6)

    Notes
    -----
    - This function expects *2D* coordinates. Use a deterministic projection
      (e.g., PCA) upstream if you have higher-dimensional vectors.
    - ``rasterized`` defaults to True to keep PDF/SVG exports small for large
      scatter plots; override via ``scatter_kwargs={'rasterized': False}``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that were drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> labels = np.random.uniform(0, 1, idx.get_n_items()).round()
    >>> labels = np.zeros(idx.get_n_items(), dtype=int)
    ...
    >>> X, ids = utils.annoy_index_to_array(idx)  # X: (n_items, f), ids: (n_items,)
    >>> y2 = utils.project_to_2d(X, method="pca")   # y2: (n_items, 2)
    ...
    >>> ax = utils.plot(
    ...     x=y2,
    ...     y=labels[ids],
    ...     title="My 2D embedding",
    ...     figsize=(7, 7),
    ...     draw_legend=True,
    ...     scatter_kwargs={"s": 6, "alpha": 0.7},
    ... )
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if title is not None:
        ax.set_title(title)

    # Main scatter customization (backwards compatible defaults).
    scatter_params = {
        "alpha": kwargs.get("alpha", 0.6),
        "s": kwargs.get("s", 1),
        "rasterized": kwargs.get("rasterized", True),
    }
    if scatter_kwargs is not None:
        scatter_params.update(scatter_kwargs)

    # Create main plot
    if label_order is not None:
        # assert all(np.isin(np.unique(y), label_order))
        if not all(np.isin(np.unique(y), label_order)):
            raise ValueError("`label_order` must contain all unique values present in `y`.")
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)

    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
    else:
        missing = set(classes) - set(colors.keys())
        if missing:
            raise ValueError(
                "`colors` is missing keys for the following labels present in `y`: "
                f"{sorted(missing)!r}."
            )

    point_colors = list(map(colors.get, y))
    ax.scatter(x[:, 0], x[:, 1], c=point_colors, **scatter_params)

    # Plot medoids (class-wise median)
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_params = {"s": 48, "alpha": 1, "edgecolor": "k"}
        if center_kwargs is not None:
            center_params.update(center_kwargs)

        center_colors = list(map(colors.get, classes))
        ax.scatter(centers[:, 0], centers[:, 1], c=center_colors, **center_params)

        # Draw medoid labels
        if draw_cluster_labels:
            text_params = {
                "fontsize": kwargs.get("fontsize", 6),
                "horizontalalignment": "center",
            }
            if text_kwargs is not None:
                text_params.update(text_kwargs)

            for idx, label in enumerate(classes):
                ax.text(centers[idx, 0], centers[idx, 1] + 2.2, label, **text_params)

    # Hide ticks and axis
    if axis_off:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    if draw_legend:
        handle_params = {
            "marker": "s",
            "color": "w",
            "ms": 10,
            "alpha": 1,
            "linewidth": 0,
            "markeredgecolor": "k",
        }
        if legend_handle_kwargs is not None:
            handle_params.update(legend_handle_kwargs)

        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                markerfacecolor=colors[yi],
                label=yi,
                **handle_params,
            )
            for yi in classes
        ]

        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    return ax


def evaluate_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    projection_embedding: Optional[np.ndarray] = None,
    projection_labels: Optional[np.ndarray] = None,
    sample: Optional[int] = None,
    random_state: Optional[object] = None,
) -> float:
    """
    Evaluate an embedding using Moran's I.

    This utility computes Moran's I index using a label-derived adjacency
    relation. Optionally, it can evaluate how well a projected embedding aligns
    with a reference embedding.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_samples, n_dims)
        Embedding to evaluate.
    labels : np.ndarray, shape (n_samples,)
        Label for each point in ``embedding``.
    projection_embedding : np.ndarray or None, default=None
        If provided, compute the score between ``projection_embedding`` and
        ``embedding``. If None, ``projection_embedding`` defaults to
        ``embedding`` (self-score).
    projection_labels : np.ndarray or None, default=None
        Labels for ``projection_embedding``. If ``projection_embedding`` is None,
        this must also be None.
    sample : int or None, default=None
        If provided, compute the score on a subsample of points from both
        ``embedding`` and ``projection_embedding`` (without replacement).
    random_state : object or None, default=None
        Controls subsampling when ``sample`` is not None.

        * If ``None`` (default), uses NumPy's global RNG (backwards compatible).
        * If an ``int``, a new RNG is created via ``np.random.default_rng``.
        * If an object has a ``choice`` method compatible with NumPy's RNG API
          (e.g. a ``numpy.random.Generator``), it is used directly.

    Returns
    -------
    score : float
        Moran's I index.

    Notes
    -----
    When ``sample`` is specified and ``random_state`` is None, results depend on
    NumPy's global random state.

    Examples
    --------
    >>> import numpy as np
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> labels = np.random.uniform(0, 1, idx.get_n_items()).round()
    >>> labels = np.zeros(idx.get_n_items(), dtype=int)
    ...
    >>> X, ids = utils.annoy_index_to_array(idx)  # X: (n_items, f), ids: (n_items,)
    ...
    >>> score = utils.evaluate_embedding(
    ...     embedding=X,
    ...     labels=labels,  # labels[ids]
    ...     sample=200,
    ...     random_state=0,   # int seed, or numpy.random.Generator
    ... )
    >>> print(score)
    """
    has_projection = projection_embedding is not None
    if projection_embedding is None:
        projection_embedding = embedding
        if projection_labels is not None:
            raise ValueError(
                "If `projection_embedding` is None, `projection_labels` must also be None."
            )
        projection_labels = labels

    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("The shape of the embedding and labels don't match")

    if projection_embedding.shape[0] != projection_labels.shape[0]:
        raise ValueError("The shape of the reference embedding and labels don't match")

    if sample is not None:
        # Preserve historical behaviour by defaulting to NumPy's global RNG.
        chooser = None
        if random_state is None:
            chooser = np.random.choice
        else:
            # Accept either an object with a `choice` method or an integer seed.
            if hasattr(random_state, "choice"):
                chooser = random_state.choice  # type: ignore[assignment]
            else:
                rng = np.random.default_rng(random_state)
                chooser = rng.choice

        n_samples = embedding.shape[0]
        sample_indices = chooser(n_samples, size=min(sample, n_samples), replace=False)
        embedding = embedding[sample_indices]
        labels = labels[sample_indices]

        n_samples = projection_embedding.shape[0]
        sample_indices = chooser(n_samples, size=min(sample, n_samples), replace=False)
        projection_embedding = projection_embedding[sample_indices]
        projection_labels = projection_labels[sample_indices]

    weights = projection_labels[:, None] == labels
    if not has_projection:
        np.fill_diagonal(weights, 0)

    mu = np.asarray(embedding.mean(axis=0)).ravel()

    numerator = np.sum(weights * ((projection_embedding - mu) @ (embedding - mu).T))
    denominator = np.sum((projection_embedding - mu) ** 2)

    return projection_embedding.shape[0] / np.sum(weights) * numerator / denominator

def annoy_index_to_array(
    index: Any,
    ids: Optional[Sequence[int]] = None,
    *,
    dtype: Any = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Materialize vectors from an Annoy-like index into a dense NumPy array.

    Parameters
    ----------
    index : object
        An Annoy-like index instance. This helper expects:
        - ``index.get_n_items() -> int``
        - ``index.get_item_vector(i) -> Sequence[float]``
    ids : sequence of int, optional
        Item ids to extract. If ``None``, uses ``range(index.get_n_items())``.
        Provide this explicitly if your ids are not contiguous.
    dtype : numpy dtype, default=np.float32
        Output dtype for the returned matrix.

    Returns
    -------
    X : np.ndarray, shape (n_items, f)
        Dense matrix of stored vectors.
    ids_out : np.ndarray, shape (n_items,)
        The item ids corresponding to the rows of ``X``.

    Notes
    -----
    - This helper is deterministic: row order is defined entirely by ``ids``.
    - Annoy typically assumes item ids are contiguous ``0..n_items-1``. If your
      application uses non-contiguous ids, always pass ``ids``.

    Examples
    --------
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> X, ids = utils.annoy_index_to_array(idx)  # X: (n_items, f), ids: (n_items,)
    >>> X2, ids2 = utils.annoy_index_to_array(idx, ids=[0, 10, 20])
    >>> X2, ids2
    """
    if not hasattr(index, "get_item_vector") or not hasattr(index, "get_n_items"):
        raise TypeError(
            "`index` must provide get_n_items() and get_item_vector(i). "
            "Got: {!r}".format(type(index))
        )

    n = int(index.get_n_items())
    if ids is None:
        ids_out = np.arange(n, dtype=np.int64)
    else:
        ids_out = np.asarray(list(ids), dtype=np.int64)
        if ids_out.ndim != 1:
            raise ValueError("`ids` must be a 1D sequence of integers.")

    if ids_out.size == 0:
        raise ValueError("`ids` must contain at least one item id.")

    v0 = np.asarray(index.get_item_vector(int(ids_out[0])), dtype=dtype)
    if v0.ndim != 1:
        raise ValueError("Annoy vectors must be 1D; got shape {}.".format(v0.shape))
    f = int(v0.shape[0])

    X = np.empty((ids_out.size, f), dtype=dtype)
    X[0] = v0
    for r in range(1, ids_out.size):
        X[r] = np.asarray(index.get_item_vector(int(ids_out[r])), dtype=dtype)

    return X, ids_out


def maxabs_scale_dense(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Scale each feature by its maximum absolute value (deterministic).

    This matches the core behavior of scikit-learn's ``MaxAbsScaler`` for dense
    arrays.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Input data.
    eps : float, default=1e-12
        Small value to avoid division by zero for all-zero columns.

    Returns
    -------
    x_scaled : np.ndarray
        Scaled array of the same shape as ``x``.

    Notes
    -----
    - Columns that are entirely zero remain unchanged.
    - This operation changes geometry. Only apply it here if it matches the
      preprocessing used when building the Annoy index.

    Examples
    --------
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> X, ids = utils.annoy_index_to_array(idx)  # X: (n_items, f), ids: (n_items,)
    >>> X_scaled = utils.maxabs_scale_dense(X)
    >>> X_scaled
    """
    x = np.asarray(x)
    denom = np.max(np.abs(x), axis=0)
    denom = np.where(denom < eps, 1.0, denom)
    return x / denom


def l2_normalize_rows(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize rows of a dense matrix (deterministic).

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Input data.
    eps : float, default=1e-12
        Small value to avoid division by zero for all-zero rows.

    Returns
    -------
    x_norm : np.ndarray
        Row-normalized array of the same shape as ``x``.

    Notes
    -----
    For angular/cosine similarity workflows, normalizing vectors is often part
    of the modeling pipeline. Apply it here only if that is what your index and
    downstream expectations assume.

    Examples
    --------
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> X, ids = utils.annoy_index_to_array(idx)  # X: (n_items, f), ids: (n_items,)
    >>> X_unit = utils.l2_normalize_rows(X)
    >>> X_unit
    """
    x = np.asarray(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return x / norms


def pca_project_2d(x: np.ndarray, *, center: bool = True) -> np.ndarray:
    """
    Deterministically project data to 2D using PCA (via SVD).

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Input data.
    center : bool, default=True
        If True, subtract the per-feature mean before projection.

    Returns
    -------
    y : np.ndarray, shape (n_samples, 2)
        2D PCA scores.

    Notes
    -----
    PCA has a sign ambiguity (principal directions are defined up to sign). To
    ensure deterministic output, this function fixes the sign of each component
    by enforcing that the largest-magnitude element of each loading vector is
    non-negative.

    Examples
    --------
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> X, ids = utils.annoy_index_to_array(idx)  # X: (n_items, f), ids: (n_items,)
    >>> y2 = utils.pca_project_2d(X, center=True)
    >>> y2
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("`x` must be 2D, got shape {}.".format(x.shape))
    if x.shape[1] < 2:
        raise ValueError("Need at least 2 features for a 2D PCA projection.")

    x0 = x - x.mean(axis=0, keepdims=True) if center else x
    # SVD: x0 = U S Vt, principal directions are rows of Vt.
    _, _, Vt = np.linalg.svd(x0, full_matrices=False)
    Vt2 = Vt[:2].copy()

    # Deterministic sign disambiguation.
    for k in range(2):
        j = int(np.argmax(np.abs(Vt2[k])))
        if Vt2[k, j] < 0:
            Vt2[k] *= -1.0

    return x0 @ Vt2.T


def project_to_2d(
    x: np.ndarray,
    *,
    method: str = "pca",
    dims: Tuple[int, int] = (0, 1),
    center: bool = True,
) -> np.ndarray:
    """
    Project vectors to 2D using a deterministic method.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Input vectors.
    method : {'pca', 'dims'}, default='pca'
        - ``'pca'``: deterministic PCA projection via :func:`pca_project_2d`.
        - ``'dims'``: take two original dimensions given by ``dims``.
    dims : tuple of int, default=(0, 1)
        Dimensions used when ``method='dims'``.
    center : bool, default=True
        Whether to mean-center before PCA (only used for ``method='pca'``).

    Returns
    -------
    y : np.ndarray, shape (n_samples, 2)
        2D projection.

    Raises
    ------
    ValueError
        If ``method`` is unknown or the input is invalid.

    Examples
    --------
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> y2 = utils.project_to_2d(X, method="pca", center=True)
    >>> y2 = utils.project_to_2d(X, method="dims", dims=(0, 1))  # just take original dims 0 and 1
    >>> y2
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("`x` must be 2D, got shape {}.".format(x.shape))

    method_l = method.lower()
    if method_l == "pca":
        return pca_project_2d(x, center=center)
    if method_l == "dims":
        d0, d1 = dims
        if not (0 <= d0 < x.shape[1] and 0 <= d1 < x.shape[1]):
            raise ValueError("`dims` must be valid feature indices for x.shape[1]={}.".format(x.shape[1]))
        return x[:, [d0, d1]].astype(np.float64, copy=False)

    raise ValueError("Unknown projection method: {!r}. Use 'pca' or 'dims'.".format(method))


def plot_annoy_index(
    index: Any,
    labels: Optional[Sequence[Any]] = None,
    *,
    ids: Optional[Sequence[int]] = None,
    projection: str = "pca",
    dims: Tuple[int, int] = (0, 1),
    center: bool = True,
    maxabs: bool = False,
    l2_normalize: bool = False,
    dtype: Any = np.float32,
    ax=None,
    title: Optional[str] = None,
    plot_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Plot an Annoy index as a 2D scatter plot (deterministic).

    This is the main entry point for visualizing an Annoy index inside scikit-plots.

    Parameters
    ----------
    index : object
        Annoy-like index providing ``get_n_items`` and ``get_item_vector``.
    labels : sequence, optional
        Labels for coloring points.

        - If ``None``, all points get label ``0``.
        - If ``len(labels) == len(ids)`` (or ``len(ids_out)``), labels are assumed to already
          be aligned to the plotted subset.
        - If ``len(labels) == index.get_n_items()``, labels are subset by ``ids`` so you can
          pass a full-length label array even when plotting a subset.
    ids : sequence of int, optional
        Item ids to extract. If ``None``, uses ``range(index.get_n_items())``.
    projection : {'pca', 'dims'}, default='pca'
        Deterministic 2D projection method.
    dims : tuple of int, default=(0, 1)
        Used when ``projection='dims'``.
    center : bool, default=True
        Mean-center data before PCA (only applies to PCA).
    maxabs : bool, default=False
        Apply MaxAbs scaling before projection.
    l2_normalize : bool, default=False
        Apply row L2 normalization before projection.
    dtype : numpy dtype, default=np.float32
        dtype used when reading vectors from the index.
    ax : matplotlib.axes.Axes, optional
        Target axes. If ``None``, a new figure and axes are created.
    title : str, optional
        Plot title.
    plot_kwargs : mapping, optional
        Keyword arguments forwarded to :func:`plot`. This is preferred over
        passing many ad-hoc kwargs from call sites.

    Returns
    -------
    y2 : np.ndarray, shape (n_items, 2)
        2D projected coordinates (what is plotted).
    ids_out : np.ndarray, shape (n_items,)
        Item ids corresponding to the rows of ``y2``.
    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Notes
    -----
    - This function is *strictly deterministic* given the index contents and
      input parameters.
    - If you want to overlay Annoy neighbor edges, use
      :func:`plot_annoy_knn_edges` with the returned ``ids_out`` and ``y2``.

    Examples
    --------
    >>> import numpy as np
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> labels = np.random.uniform(0, 1, idx.get_n_items()).round()
    >>> labels = np.zeros(idx.get_n_items(), dtype=int)
    ...
    >>> nth = 1
    >>> ids_subset = np.arange(0, idx.get_n_items(), nth)  # every 1th item
    ...
    >>> y2, ids, ax = utils.plot_annoy_index(
    ...     idx,
    ...     labels=labels[ids_subset],    # must match ids length
    ...     ids=ids_subset,
    ...     projection="pca",
    ...     plot_kwargs={"draw_legend": False},
    ... )
    ...
    >>> utils.plot_annoy_knn_edges(idx, y2, ids=ids, k=nth)
    """
    X, ids_out = annoy_index_to_array(index, ids=ids, dtype=dtype)

    if maxabs:
        X = maxabs_scale_dense(X)
    if l2_normalize:
        X = l2_normalize_rows(X)

    y2 = project_to_2d(X, method=projection, dims=dims, center=center)

    if labels is None:
        labels_arr = np.zeros(y2.shape[0], dtype=int)
    else:
        labels_arr = np.asarray(labels)
        if labels_arr.ndim != 1:
            raise ValueError("`labels` must be a 1D sequence.")
        if labels_arr.shape[0] == y2.shape[0]:
            # Labels already aligned to the extracted ids/vectors.
            pass
        else:
            # User-friendly path: if labels cover the full index, subset by ids_out.
            n_items = int(index.get_n_items())
            if labels_arr.shape[0] == n_items:
                # ids_out are the item ids corresponding to y2 rows.
                if np.any(ids_out < 0) or np.any(ids_out >= n_items):
                    raise ValueError(
                        "`ids` contains values outside [0, index.get_n_items()). "
                        "When providing full-length `labels`, ids must be valid indices."
                    )
                labels_arr = labels_arr[ids_out]
            else:
                raise ValueError(
                    "`labels` must have length equal to len(ids) (plotted subset) or "
                    "index.get_n_items() (full index). "
                    f"Got labels length {labels_arr.shape[0]} and plotted length {y2.shape[0]}."
                )

    kwargs_local: Dict[str, Any] = {}
    if plot_kwargs is not None:
        kwargs_local.update(dict(plot_kwargs))

    ax = plot(y2, labels_arr, ax=ax, title=title, **kwargs_local)
    return y2, ids_out, ax


def plot_annoy_knn_edges(
    index: Any,
    y2: np.ndarray,
    *,
    ids: Optional[Sequence[int]] = None,
    k: int = 5,
    search_k: int = -1,
    ax=None,
    line_kwargs: Optional[Mapping[str, Any]] = None,
    undirected: bool = True,
    axis_off=True,
) -> Any:
    """
    Overlay kNN edges from an Annoy index on an existing 2D plot.

    Parameters
    ----------
    index : object
        Annoy-like index providing ``get_nns_by_item``.
    y2 : np.ndarray, shape (n_items, 2)
        2D coordinates (typically returned by :func:`plot_annoy_index`).
    ids : sequence of int, optional
        Item ids corresponding to rows of ``y2``. If ``None``, assumes ids are
        ``0..n_items-1``.
    k : int, default=10
        Number of neighbors (per point) to connect.
    search_k : int, default=-1
        Annoy search_k parameter forwarded to ``get_nns_by_item``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If ``None``, uses the current axes.
    line_kwargs : mapping, optional
        Keyword arguments forwarded to ``ax.plot`` for edges.
    undirected : bool, default=True
        If True, de-duplicate edges (i,j) and (j,i) into a single undirected edge.
    axis_off : bool, default=True
        If True, hide ticks and turn the axis frame off.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that were drawn on.

    Notes
    -----
    - This helper draws potentially many line segments. For large ``n_items``,
      prefer plotting on raster backends or reduce ``k``.
    - Determinism depends on the determinism of Annoy's neighbor query order for
      equal-distance ties. For typical float embeddings, results are stable.

    Examples
    --------
    >>> import numpy as np
    >>> import scikitplot.cexternals._annoy._plotting as utils
    ...
    >>> labels = np.random.uniform(0, 1, idx.get_n_items()).round()
    >>> labels = np.zeros(idx.get_n_items(), dtype=int)
    ...
    >>> y2, ids, ax = utils.plot_annoy_index(
    ...     idx,
    ...     labels=labels,
    ...     projection="pca",          # deterministic PCA -> 2D
    ...     center=True,               # center before PCA projection
    ...     maxabs=False,              # optional: per-feature maxabs scaling
    ...     l2_normalize=False,        # optional: row L2 normalization
    ...     plot_kwargs=dict(
    ...         scatter_kwargs=dict(s=6, alpha=0.7),
    ...         figsize=(7, 7),
    ...         draw_legend=False,
    ...     ),
    ... )
    ...
    >>> # optional: overlay neighbor edges among the shown points
    >>> utils.plot_annoy_knn_edges(
    ...     idx, y2, ids=ids, k=5, search_k=-1, line_kwargs={"alpha": 0.15}
    ... )
    """
    import matplotlib.pyplot as plt

    if not hasattr(index, "get_nns_by_item"):
        raise TypeError("`index` must provide get_nns_by_item(i, n, ...).")

    y2 = np.asarray(y2)
    if y2.ndim != 2 or y2.shape[1] != 2:
        raise ValueError("`y2` must be of shape (n_items, 2).")

    if ids is None:
        ids_arr = np.arange(y2.shape[0], dtype=np.int64)
    else:
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        if ids_arr.shape[0] != y2.shape[0]:
            raise ValueError("`ids` length must match y2.shape[0].")

    if k <= 0:
        raise ValueError("`k` must be positive.")

    id_to_row = {int(i): r for r, i in enumerate(ids_arr)}

    if ax is None:
        ax = plt.gca()

    # Hide ticks and axis
    if axis_off:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    params = {"linewidth": 0.4, "alpha": 0.25}
    if line_kwargs is not None:
        params.update(dict(line_kwargs))

    seen = set()  # type: ignore[var-annotated]
    for r, item_id in enumerate(ids_arr):
        nbrs = index.get_nns_by_item(int(item_id), int(k) + 1, search_k=int(search_k))
        for nb in nbrs:
            nb = int(nb)
            if nb == int(item_id):
                continue
            rr = id_to_row.get(nb)
            if rr is None:
                continue  # neighbor not in plotted subset
            if undirected:
                a, b = (r, rr) if r <= rr else (rr, r)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                r0, r1 = a, b
            else:
                r0, r1 = r, rr

            ax.plot([y2[r0, 0], y2[r1, 0]], [y2[r0, 1], y2[r1, 1]], **params)

    return ax
