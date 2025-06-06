"""
The :py:mod:`~scikitplot.estimators` module includes plots built specifically for
scikit-learn estimator (classifier/regressor) instances e.g. Random Forest.
You can use your own estimators, but these plots assume specific properties
shared by scikit-learn estimators. The specific requirements are documented per
function.

The Scikit-plots Functional API

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""

# code that needs to be compatible with both Python 2 and Python 3

# pylint: disable=import-error
# pylint: disable=broad-exception-caught

import time

import numpy as np  # type: ignore[reportMissingImports]
from sklearn.base import clone  # type: ignore[reportMissingModuleSource]
from joblib import Parallel, delayed  # type: ignore[reportMissingModuleSource]

from ..._utils.validation import (
    validate_plotting_kwargs_decorator,
    # validate_shapes_decorator,
    # validate_y_true_decorator,
    # validate_y_probas_decorator,
    # validate_y_probas_bounds_decorator,
)
from ....utils.utils_plot_mpl import save_plot_decorator
from ...._docstrings import _docstring

## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
    # '_clone_and_score_clusterer',
    "plot_elbow",
]


def _clone_and_score_clusterer(clf, X, n_clusters):
    """
    Clones and scores a clusterer instance.

    This function fits the given clusterer instance on the provided data and scores the clustering performance.

    Parameters
    ----------
    clf : object
        Clusterer instance that implements `fit`, `fit_predict`,
        and `score` methods, and an `n_clusters` hyperparameter.
        Example: :class:`sklearn.cluster.KMeans` instance.

    X : array-like, shape (n_samples, n_features)
        Data to cluster, where `n_samples` is the number of samples
        and `n_features` is the number of features.

    n_clusters : int
        Number of clusters.

    Returns
    -------
    score : float
        Score of the clusters.

    time : float
        Number of seconds it took to fit the clusterer.

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(return_X_y=True)
    >>> kmeans = KMeans(n_clusters=3, random_state=0)
    >>> clone_and_score_clusterer(kmeans, X, n_clusters=3)
    (score, time)

    Notes
    -----
    The `score` value is based on the clusterer's scoring method,
    and the `time` represents the fitting duration.

    """
    start = time.time()
    clf = clone(clf)
    clf.n_clusters = n_clusters
    return clf.fit(X).score(X), time.time() - start


@validate_plotting_kwargs_decorator
@save_plot_decorator
@_docstring.interpd
def plot_elbow(
    clf,
    X,
    *,
    cluster_ranges=None,
    show_cluster_time=True,
    n_jobs=1,
    title="Elbow Curves",
    title_fontsize="large",
    text_fontsize="medium",
    **kwargs,
):
    """
    Plot the elbow curve for different values of K in KMeans clustering.

    Parameters
    ----------
    clf : object
        A clusterer instance with ``fit``, ``fit_predict``, and ``score`` methods,
        and an ``n_clusters`` hyperparameter. Typically an instance of
        :class:`sklearn.cluster.KMeans`.

    X : array-like of shape (n_samples, n_features)
        The data to cluster, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    cluster_ranges : list of int or None, optional, default=range(1, 12, 2)
        List of values for `n_clusters` over which to plot the explained variances.

    show_cluster_time : bool, optional
        Whether to include a plot of the time taken to cluster for each value of K.

    n_jobs : int, optional, default=1
        The number of jobs to run in parallel.

    title : str, optional, default="Elbow Plot"
        The title of the generated plot.

    title_fontsize : str or int, optional, default="large"
        Font size of the title. Accepts Matplotlib font sizes,
        such as "small", "medium", "large", or an integer value.

    text_fontsize : str or int, optional, default="medium"
        Font size of the text labels. Accepts Matplotlib font sizes,
        such as "small", "medium", "large", or an integer value.

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

    Examples
    --------
    .. plot::
       :context: close-figs
       :align: center
       :alt: Elbow Curve

        >>> from sklearn.cluster import KMeans
        >>> from sklearn.datasets import load_iris as data_3_classes
        >>> import scikitplot as skplt
        >>> X, y = data_3_classes(return_X_y=True, as_frame=False)
        >>> kmeans = KMeans(random_state=0)
        >>> skplt.estimators.plot_elbow(
        >>>     kmeans,
        >>>     X,
        >>>     cluster_ranges=range(1, 10),
        >>> );

    """
    if cluster_ranges is None:
        cluster_ranges = range(1, 12, 2)
    else:
        cluster_ranges = sorted(cluster_ranges)

    if not hasattr(clf, "n_clusters"):
        raise TypeError(
            '"n_clusters" attribute not in classifier. Cannot plot elbow method.'
        )

    tuples = Parallel(n_jobs=n_jobs)(
        delayed(_clone_and_score_clusterer)(clf, X, i) for i in cluster_ranges
    )
    clfs, times = zip(*tuples)

    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    # Proceed with your plotting logic here
    fig, ax = kwargs.get("fig"), kwargs.get("ax")
    ax.set_title(title, fontsize=title_fontsize)
    ax.plot(cluster_ranges, np.absolute(clfs), "b*-")
    ax.grid(True)
    ax.set_xlabel("Number of clusters", fontsize=text_fontsize)
    ax.set_ylabel("Sum of Squared Errors", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)

    if show_cluster_time:
        ax2_color = "green"
        ax2 = ax.twinx()
        ax2.plot(cluster_ranges, times, ":", alpha=0.75, color=ax2_color)
        ax2.set_ylabel(
            "Clustering duration (seconds)",
            color=ax2_color,
            alpha=0.75,
            fontsize=text_fontsize,
        )
        ax2.tick_params(colors=ax2_color, labelsize=text_fontsize)

    return ax
