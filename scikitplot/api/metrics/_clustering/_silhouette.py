# code that needs to be compatible with both Python 2 and Python 3

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught

"""
The :mod:`~scikitplot.metrics` module includes plots for machine learning
evaluation metrics e.g. confusion matrix, silhouette scores, etc.

The Scikit-plots Functional API

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""

import numpy as np

# import matplotlib as mpl
import matplotlib.pyplot as plt

# Sigmoid and Softmax functions
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder

from ...._docstrings import _docstring
from ..._utils.validation import validate_plotting_kwargs_decorator
from ....utils.utils_plot_mpl import save_plot_decorator

## Define __all__ to specify the public interface of the module,
# not required default all above func
__all__ = ["plot_silhouette"]


@validate_plotting_kwargs_decorator
@save_plot_decorator
@_docstring.interpd
def plot_silhouette(
    ## default params
    X,
    cluster_labels,
    *,
    metric="euclidean",
    # copy=True,
    ## plotting params
    title="Silhouette Analysis",
    title_fontsize="large",
    text_fontsize="medium",
    cmap=None,
    digits=4,
    ## additional params
    **kwargs,
):
    """
    Plots silhouette analysis of clusters provided.

    Silhouette analysis is a method of interpreting and validating the consistency
    within clusters of data. It measures how similar an object is to its own
    cluster compared to other clusters.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    cluster_labels : array-like, shape (n_samples,)
        Cluster label for each sample.

    metric : str or callable, optional, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.
        If metric is a string, it must be one of the options allowed by
        `sklearn.metrics.pairwise.pairwise_distances`. If `X` is the distance array itself,
        use "precomputed" as the metric.

    copy : bool, optional, default=True
        Determines whether `fit` is used on `clf` or on a copy of `clf`.

    title : str, optional, default='Silhouette Analysis'
        Title of the generated plot.

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.

    cmap : None, str or matplotlib.colors.Colormap, optional, default=None
        Colormap used for plotting.
        Options include 'viridis', 'PiYG', 'plasma', 'inferno', 'nipy_spectral', etc.
        See Matplotlib Colormap documentation for available choices.

        - https://matplotlib.org/stable/users/explain/colors/index.html
        - plt.colormaps()
        - plt.get_cmap()  # None == 'viridis'

    digits : int, optional, default=4
        Number of digits for formatting output floating point values.

        .. versionadded:: 0.3.9

    **kwargs: dict
        Generic keyword arguments.

    Other Parameters
    ----------------
    %(_validate_plotting_kwargs_doc)s

    %(_save_plot_decorator_kwargs_doc)s

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.


    .. dropdown:: References

      * `"scikit-learn silhouette_score"
        <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#>`_.

    Examples
    --------

    .. plot::
       :context: close-figs
       :align: center
       :alt: Silhouette Plot

        >>> from sklearn.datasets import make_blobs
        >>> from sklearn.cluster import KMeans
        >>> from sklearn.datasets import load_iris as data_3_classes
        >>> import scikitplot as skplt
        >>> X, y = data_3_classes(return_X_y=True, as_frame=False)
        >>> kmeans = KMeans(n_clusters=3, random_state=0)
        >>> cluster_labels = kmeans.fit_predict(X)
        >>> skplt.metrics.plot_silhouette(
        >>>     X,
        >>>     cluster_labels,
        >>> );

    """
    ##################################################################
    ## Preprocessing
    ##################################################################
    # Proceed with your preprocess logic here
    # np.asarray(...) Not copy, validate shape, or add axes. It's a lightweight,
    # shallow wrapper — useful, but limited.
    cluster_labels = np.asanyarray(cluster_labels)

    le = LabelEncoder()
    cluster_labels_encoded = le.fit_transform(cluster_labels)

    n_clusters = len(np.unique(cluster_labels))

    silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)

    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)

    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    # Proceed with your plotting logic here
    fig, ax = kwargs.get("fig"), kwargs.get("ax")
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[
            cluster_labels_encoded == i
        ]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.get_cmap(cmap)(float(i) / n_clusters)

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(
            -0.05,
            y_lower + 0.5 * size_cluster_i,
            str(le.classes_[i]),
            fontsize=text_fontsize,
        )

        y_lower = y_upper + 10

    ax.axvline(
        x=silhouette_avg,
        color="red",
        linestyle="--",
        label="Silhouette score: {0:.{digits}f}".format(silhouette_avg, digits=digits),
    )

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Silhouette coefficient values", fontsize=text_fontsize)
    ax.set_ylabel("Cluster label", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)

    ax.set_xticks(np.arange(-0.1, 1.0, 0.2))
    ax.set_yticks([])  # Clear the y-axis labels / ticks

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10 + 10])

    # Display legend
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=text_fontsize)

    plt.tight_layout()
    fig.tight_layout()
    return ax
