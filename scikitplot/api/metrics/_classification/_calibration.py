# code that needs to be compatible with both Python 2 and Python 3

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught

"""
The :mod:`~scikitplot.metrics` module includes plots for machine learning
evaluation metrics e.g. confusion matrix, silhouette scores, etc.

The Scikit-plots Functional API

This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.
"""

from collections.abc import KeysView, ValuesView

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

from .... import logger
from ...._docstrings import _docstring
from ..._utils.validation import validate_inputs, validate_plotting_kwargs_decorator
from ....utils.utils_plot_mpl import save_plot_decorator

## Define __all__ to specify the public interface of the module,
# not required default all above func
__all__ = [
    "plot_calibration",
]

ListLike = (list, tuple, set, KeysView, ValuesView, np.ndarray)


@validate_plotting_kwargs_decorator
@save_plot_decorator
@_docstring.interpd
def plot_calibration(
    ## default params
    y_true,
    y_probas_list,
    *,
    pos_label=None,  # for binary y_true
    class_index=None,  # for multi-class y_probas
    class_names=None,
    # multi_class=None,
    to_plot_class_index=1,
    estimator_names=None,
    n_bins=10,
    strategy="uniform",
    ## plotting params
    title="Calibration Curves (Reliability Diagrams)",
    title_fontsize="large",
    text_fontsize="medium",
    cmap="inferno",
    ## additional params
    **kwargs,
):
    """
    Plot calibration curves for a set of classifier probability estimates.

    This function plots calibration curves, also known as reliability curves,
    which are useful to assess the calibration of probabilistic models.
    For a well-calibrated model, the predicted probability should match the
    observed frequency of the positive class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_probas_list : list of array-like, shape (n_samples, 2) or (n_samples,)
        A list containing the outputs of classifiers' `predict_proba` or
        `decision_function` methods.

    n_bins : int, optional, default=10
        Number of bins to use in the calibration curve. A higher number requires
        more data to produce reliable results.

    strategy : str, optional, default='uniform'
        Strategy used to define the widths of the bins:

        - 'uniform': Bins have identical widths.
        - 'quantile': Bins have the same number of samples and depend on `y_probas_list`.

        .. versionadded:: 0.3.9

    estimator_names : list of str or None, optional, default=None
        A list of classifier names corresponding to the probability estimates in
        `y_probas_list`. If None, the names will be generated automatically as
        "Classifier 1", "Classifier 2", etc.

    class_names : list of str or None, optional, default=None
        List of class names for the legend. The order should match the classes in
        `y_probas_list`. If None, class indices will be used.

    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot for the entire probability distribution.

    class_index : int, optional, default=1
        Index of the class of interest for multiclass classification. Ignored for
        binary classification. Related to `multi_class` parameter. Not Implemented.

    to_plot_class_index : int, list-like, optional, default=1
        Specific classes to plot. If a given class does not exist, it will be ignored.
        If None, all classes are plotted.

    title : str, optional, default='Calibration plots (Reliability Curves)'
        Title of the generated plot.

    title_fontsize : str or int, optional, default='large'
        Font size of the plot title. Accepts Matplotlib-style sizes like "small",
        "medium", "large", or an integer.

    text_fontsize : str or int, optional, default='medium'
        Font size of the plot text (axis labels). Accepts Matplotlib-style sizes
        like "small", "medium", "large", or an integer.

    cmap : None, str or matplotlib.colors.Colormap, optional, default=None
        Colormap used for plotting.
        Options include 'viridis', 'PiYG', 'plasma', 'inferno', 'nipy_spectral', etc.
        See Matplotlib Colormap documentation for available choices.

        - https://matplotlib.org/stable/users/explain/colors/index.html
        - plt.colormaps()
        - plt.get_cmap()  # None == 'viridis'

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

    Notes
    -----
    - The calibration curve is plotted for the class specified by `to_plot_class_index`.
    - This function currently supports binary and multiclass classification.


    .. dropdown:: References

      * `"scikit-learn calibration_curve"
        <https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#>`_.

    Examples
    --------

    .. plot::
       :context: close-figs
       :align: center
       :alt: Calibration Curves

        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.naive_bayes import GaussianNB
        >>> from sklearn.svm import LinearSVC
        >>> from sklearn.calibration import CalibratedClassifierCV
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import cross_val_predict
        >>> import numpy as np
        ...
        ... np.random.seed(0)
        >>> # importing pylab or pyplot
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Import scikit-plot
        >>> import scikitplot as skplt
        >>>
        >>> # Load the data
        >>> X, y = make_classification(
        >>>     n_samples=100000,
        >>>     n_features=20,
        >>>     n_informative=4,
        >>>     n_redundant=2,
        >>>     n_repeated=0,
        >>>     n_classes=3,
        >>>     n_clusters_per_class=2,
        >>>     random_state=0
        >>> )
        >>> X_train, y_train, X_val, y_val = (
        ...     X[:1000],
        ...     y[:1000],
        ...     X[1000:],
        ...     y[1000:],
        ... )
        >>>
        >>> # Create an instance of the LogisticRegression
        >>> lr_probas = (
        ...     LogisticRegression(max_iter=int(1e5), random_state=0)
        ...     .fit(X_train, y_train)
        ...     .predict_proba(X_val)
        ... )
        >>> nb_probas = GaussianNB().fit(X_train, y_train).predict_proba(X_val)
        >>> svc_scores = LinearSVC().fit(X_train, y_train).decision_function(X_val)
        >>> svc_isotonic = (
        ...     CalibratedClassifierCV(LinearSVC(), cv=2, method='isotonic')
        ...     .fit(X_train, y_train)
        ...     .predict_proba(X_val)
        ... )
        >>> svc_sigmoid = (
        ...     CalibratedClassifierCV(LinearSVC(), cv=2, method='sigmoid')
        ...     .fit(X_train, y_train)
        ...     .predict_proba(X_val)
        ... )
        >>> rf_probas = (
        ...     RandomForestClassifier(random_state=0)
        ...     .fit(X_train, y_train)
        ...     .predict_proba(X_val)
        ... )
        >>>
        >>> probas_dict = {
        >>>     LogisticRegression(): lr_probas,
        >>> # GaussianNB(): nb_probas,
        >>>     "LinearSVC() + MinMax": svc_scores,
        >>>     "LinearSVC() + Isotonic": svc_isotonic,
        >>>     "LinearSVC() + Sigmoid": svc_sigmoid,
        >>> # RandomForestClassifier(): rf_probas,
        >>> }
        >>> # Plot!
        >>> fig, ax = plt.subplots(figsize=(12, 6))
        >>> ax = skplt.metrics.plot_calibration(
        >>>     y_val,
        >>>     y_probas_list=probas_dict.values(),
        >>>     estimator_names=probas_dict.keys(),
        >>>     ax=ax,
        >>> );

    """
    ##################################################################
    ## Preprocessing
    ##################################################################
    # Proceed with your preprocess logic here
    # Handle the case where estimator_names are not provided
    if estimator_names is None:
        estimator_names = [f"Clf_{i + 1}" for i, _model in enumerate(y_probas_list)]

    if isinstance(estimator_names, ListLike) and isinstance(y_probas_list, ListLike):
        estimator_names = np.asanyarray(list(estimator_names), dtype=str)
        # Safely stack values from a list-like
        # y_probas_list = np.stack([np.asanyarray(v) for v in y_probas_list])
        # y_probas_list = np.asanyarray(list(map(np.asanyarray, y_probas_list)))
        y_probas_list = list(map(np.asanyarray, y_probas_list))

        # Check if the length of estimator_names matches y_probas_list
        if len(estimator_names) != len(y_probas_list):
            raise ValueError(
                f"Length of `estimator_names` ({len(estimator_names)}) does not match "
                f"length of `y_probas_list` ({len(y_probas_list)})."
            )
    else:
        raise ValueError(
            f"`estimator_names` type {type(estimator_names)} must be a None "
            f"or list of (str, model), `y_probas_list` type {type(y_probas_list)} "
            f"must be a list of probability arrays."
        )

    for idx, y_probas in enumerate(y_probas_list):
        # np.asarray(...) Not copy, validate shape, or add axes. It's a lightweight,
        # shallow wrapper — useful, but limited.
        y_probas = np.asanyarray(y_probas)
        # Convert input to numpy arrays for efficient processing
        y_true_cur, y_probas = validate_inputs(
            y_true,
            y_probas,
            pos_label=pos_label,
            class_index=class_index,
        )
        # equalize ndim for y_true and y_probas 2D
        if y_true_cur.ndim == 1:
            # Binarize the true labels only
            # y_true = y_true[:, None]
            # y_true = np.column_stack([1 - y_true, y_true])
            # Force as proper 2D NumPy array
            classes = np.unique(y_true_cur)
            y_true_cur = np.asanyarray(label_binarize(y_true_cur, classes=classes))

        # y_probas should be 2D (n_samples, n_classes) or 1D (n_samples,)
        if y_probas.ndim == 1:
            # If binary classification with probabilities for positive class only
            # Convert to 2D by stacking prob for negative class as 1 - prob
            # y_probas = y_probas[:, None]
            y_probas = np.column_stack([1 - y_probas, y_probas])

        if y_true_cur.shape != y_probas.shape:
            logger.error(
                f"Shape mismatch: `y_true` shape {y_true_cur.shape!r}, "
                f"`y_probas` shape {y_probas.shape!r}"
            )
            # raise ValueError("Shape mismatch between `y_true` and `y_probas`.")

        y_probas_list[idx] = y_probas

    y_true = y_true_cur
    # Get unique classes and filter the ones to plot
    # if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
    #     classes = np.arange(y_true.shape[1])
    # else:
    #     raise ValueError("y_true must be a 2D NumPy array")
    classes = np.arange(y_true.shape[1])
    to_plot_class_index = (
        classes if to_plot_class_index is None else to_plot_class_index
    )
    to_plot_class_index = (
        [to_plot_class_index]
        if isinstance(to_plot_class_index, int)
        else to_plot_class_index
    )
    indices_to_plot = np.isin(element=classes, test_elements=to_plot_class_index)

    ##################################################################
    ## Plotting
    ##################################################################
    # fig, ax = validate_plotting_kwargs(
    #     ax=ax, fig=fig, figsize=figsize, subplot_position=111
    # )
    # Proceed with your plotting logic here
    # Initialize dictionaries to store results
    _fig, ax = kwargs.get("fig"), kwargs.get("ax")
    fraction_of_positives_dict, mean_predicted_value_dict = {}, {}

    # Loop through classes and classifiers
    for i, to_plot in enumerate(indices_to_plot):
        for j, y_probas in enumerate(y_probas_list):
            # Calculate the calibration curve
            fraction_of_positives_dict[i], mean_predicted_value_dict[i] = (
                calibration_curve(
                    y_true[:, i], y_probas[:, i], n_bins=n_bins, strategy=strategy
                )
            )
            # Plot if the class is to be plotted
            if to_plot:
                if class_names is None:
                    class_names = classes
                color = plt.get_cmap(cmap)(float(j) / len(y_probas_list))
                ax.plot(
                    mean_predicted_value_dict[i],
                    fraction_of_positives_dict[i],
                    marker="s",
                    ls="-",
                    color=color,
                    lw=2,
                    label=f"Class {class_names[i]}, {estimator_names[j]}",
                )

    # Plot the diagonal line for reference
    ax.plot([0, 1], [0, 1], c="gray", ls=":", lw=2, label="Perfectly calibrated")

    # Set plot title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Mean predicted probability [0, 1]", fontsize=text_fontsize)
    ax.set_ylabel("Fraction of positives", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)

    ax.set_xlim([-0.035, 1.05])
    ax.set_ylim([-0.050, 1.05])

    # Define the desired number of ticks
    num_ticks = 10

    # Set x-axis ticks and labels
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator( (ax.get_xlim()[1] / 10) ))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

    # Enable grid and display legend
    ax.grid(True)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", title="Classifier Model", alignment="left")
    return ax
