"""
This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.

The :mod:`scikitplot.metrics` module includes plots for machine learning
evaluation metrics e.g. confusion matrix, silhouette scores, etc.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    label_binarize, LabelEncoder,
    MinMaxScaler,
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score,
    silhouette_score,
    silhouette_samples,
    auc, roc_curve, precision_recall_curve,
)
from sklearn.utils.multiclass import unique_labels
from sklearn.calibration import calibration_curve
from sklearn.utils import deprecated

from .utils.helpers import (
    validate_labels,
    cumulative_gain_curve,
    binary_ks_curve,
    sigmoid,
    softmax,
)


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
    'plot_calibration_curve',
    'plot_classifier_eval',
    'plot_confusion_matrix',
    'plot_roc_curve', 'plot_roc',
    'plot_precision_recall_curve', 'plot_precision_recall',
    'plot_silhouette',
]


def plot_calibration_curve(
    y_true, 
    y_prob_list, 
    y_is_decision, 
    title='Calibration Curves (Reliability Diagrams)',
    ax=None, 
    figsize=None, 
    title_fontsize="large", 
    text_fontsize="medium",
    cmap=None,
    n_bins=10,
    clf_names=None, 
    multi_class=None,
    class_index=1, 
    class_names=None,
    classes_to_plot=[1],
    strategy="uniform",
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
    
    y_prob_list : list of array-like, shape (n_samples, 2) or (n_samples,)
        A list containing the outputs of classifiers' `predict_proba` or
        `decision_function` methods.
    
    y_is_decision : list of bool
        A list indicating whether the classifier's probability method is
        `decision_function` (True) or `predict_proba` (False).
    
    title : str, optional, default='Calibration plots (Reliability Curves)'
        Title of the generated plot.
    
    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the curve. If None, a new figure and axes
        will be created.
    
    figsize : tuple, optional
        Tuple denoting the figure size of the plot, e.g., (6, 6). Defaults to `None`.
    
    title_fontsize : str or int, optional, default='large'
        Font size of the plot title. Accepts Matplotlib-style sizes like "small",
        "medium", "large", or an integer.
    
    text_fontsize : str or int, optional, default='medium'
        Font size of the plot text (axis labels). Accepts Matplotlib-style sizes
        like "small", "medium", "large", or an integer.
    
    cmap : str or matplotlib.colors.Colormap, optional, default=None
        Colormap used for plotting. If None, the default 'viridis' colormap is used.
        See Matplotlib colormap documentation for options.
    
    n_bins : int, optional, default=10
        Number of bins to use in the calibration curve. A higher number requires
        more data to produce reliable results.
    
    clf_names : list of str or None, optional, default=None
        A list of classifier names corresponding to the probability estimates in
        `y_prob_list`. If None, the names will be generated automatically as
        "Classifier 1", "Classifier 2", etc.
    
    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot for the entire probability distribution.
        - Not Implemented: Strategy not yet available.
    
    class_index : int, optional, default=1
        Index of the class of interest for multiclass classification. Ignored for
        binary classification. Related to `multi_class` parameter. Not Implemented.
    
    class_names : list of str or None, optional, default=None
        List of class names for the legend. The order should match the classes in
        `y_prob_list`. If None, class indices will be used.
    
    classes_to_plot : list-like, optional, default=[1]
        Specific classes to plot. If a given class does not exist, it will be ignored.
        If None, all classes are plotted.
    
    strategy : str, optional, default='uniform'
        Strategy used to define the widths of the bins:
        - 'uniform': Bins have identical widths.
        - 'quantile': Bins have the same number of samples and depend on `y_prob_list`.

        .. versionadded:: 0.3.9
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.
    
    Notes
    -----
    - The calibration curve is plotted for the class specified by `classes_to_plot`.
    - This function currently supports binary and multiclass classification.
    
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
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import cross_val_predict
        >>> import numpy as np; np.random.seed(0)
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
        >>> X_train, y_train, X_val, y_val = X[:1000], y[:1000], X[1000:], y[1000:]
        >>> 
        >>> # Create an instance of the LogisticRegression
        >>> lr_probas = LogisticRegression().fit(X_train, y_train).predict_proba(X_val)
        >>> nb_probas = GaussianNB().fit(X_train, y_train).predict_proba(X_val)
        >>> svc_scores = LinearSVC().fit(X_train, y_train).decision_function(X_val)
        >>> rf_probas = RandomForestClassifier().fit(X_train, y_train).predict_proba(X_val)
        >>> 
        >>> probas_dict = {
        >>>     LogisticRegression(): lr_probas,
        >>>     GaussianNB(): nb_probas,
        >>>     LinearSVC(): svc_scores,
        >>>     RandomForestClassifier(): rf_probas,
        >>> }
        >>> # Plot!
        >>> ax = skplt.metrics.plot_calibration_curve(
        >>>     y_val,
        >>>     y_prob_list=list(probas_dict.values()),
        >>>     y_is_decision=list([False, False, True, False]),
        >>>     n_bins=10, 
        >>>     clf_names=list(probas_dict.keys()),
        >>>     multi_class=None,
        >>>     class_index=2, 
        >>>     classes_to_plot=[2],
        >>> );
    """
    title_pad = None
    # Create a new figure and axes if none are provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Check if the length of clf_list matches y_prob_list
    if len(y_is_decision) != len(y_prob_list):
        raise ValueError(
            f'Length of `y_is_decision` ({len(y_is_decision)}) does not match '
            f'length of `y_prob_list` ({len(y_prob_list)}).'
        )

    # Handle the case where clf_names are not provided
    if clf_names is None:
        clf_names = [
            f'Classifier {i+1}' for i, model in enumerate(y_prob_list)
        ]
    
    # Check if the length of clf_list matches y_prob_list
    if len(clf_names) != len(y_prob_list):
        raise ValueError(
            f'Length of `clf_names` ({len(clf_names)}) does not match '
            f'length of `y_prob_list` ({len(y_prob_list)}).'
        )

    # Ensure y_prob_list is a list of arrays
    if isinstance(y_prob_list, list):
        y_prob_list = list(map(np.asarray, y_prob_list))
    else:
        raise ValueError(
            '`y_prob_list` must be a list of arrays.'
        )

    y_true = np.asarray(y_true)
    
    for j, y_probas in enumerate(y_prob_list):
        # Handle binary classification
        if len(np.unique(y_true)) == 2:
            # 1D y_probas (single class probabilities)
            if y_probas.ndim == 1:
                if y_is_decision[j]:
                    # `y_probas` to the range [0, 1]
                    y_probas = np.asarray(sigmoid(y_probas))
                # Combine into a two-column
                y_probas = np.column_stack([1 - y_probas, y_probas])
        # Handle multi-class classification
        elif len(np.unique(y_true)) > 2:
            if multi_class == 'ovr':
                # Handle 1D y_probas (single class probabilities)
                if y_probas.ndim == 1:
                    if y_is_decision[j]:
                        # `y_probas` to the range [0, 1]
                        y_probas = np.asarray(softmax(y_probas))
                    # Combine into a two-column binary format OvR
                    y_probas = np.column_stack([1 - y_probas, y_probas])
                else:
                    if y_is_decision[j]:
                        # `y_probas` to the range [0, 1]
                        y_probas = np.asarray(sigmoid(y_probas))
                    # Combine into a two-column binary format OvR
                    y_probas = y_probas[:, class_index]
                    y_probas = np.column_stack([1 - y_probas, y_probas])
                    
                # Add a subtitle indicating the use of the One-vs-Rest strategy
                plt.suptitle(
                    t="One-vs-Rest (OVR) strategy for multi-class classification.",
                    fontsize=text_fontsize, x=0.512, y=0.902,
                    ha='center', va='center',
                    bbox=dict(facecolor='none', edgecolor='w', boxstyle='round,pad=0.2')
                )
                title_pad = 25
            elif multi_class in ['multinomial', None]:
                if y_probas.ndim == 1:
                    raise ValueError(
                        "For multinomial classification, `y_probas` must be 2D."
                        "For a 1D `y_probas` with more than 2 classes in `y_true`, "
                        "only 'ovr' multi-class strategy is supported."
                    )
                if y_is_decision[j]:
                    # `y_probas` to the range [0, 1]
                    y_probas = np.asarray(softmax(y_probas))
            else:
                raise ValueError("Unsupported `multi_class` strategy.")

        y_prob_list[j] = y_probas
        
    if len(np.unique(y_true)) > 2:
        if multi_class == 'ovr':
            # Binarize y_true for multiclass classification
            y_true = y_true_bin = label_binarize(
                y_true, classes=np.unique(y_true)
            )[:, class_index]

    # Initialize dictionaries to store results
    fraction_of_positives_dict, mean_predicted_value_dict = {}, {}

    # Get unique classes and filter the ones to plot
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError(
            'Cannot calculate calibration curve for a single class.'
        )

    classes_to_plot = classes if classes_to_plot is None else classes_to_plot
    indices_to_plot = np.isin(classes, classes_to_plot)

    # Binarize y_true for multiclass classification
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    # Loop through classes and classifiers
    for i, to_plot in enumerate(indices_to_plot):
        for j, y_probas in enumerate(y_prob_list):
            # Calculate the calibration curve
            fraction_of_positives_dict[i], mean_predicted_value_dict[i] = calibration_curve(
                y_true_bin[:, i],
                y_probas[:, i],
                n_bins=n_bins,
                strategy=strategy,            
            )
            # Plot if the class is to be plotted
            if to_plot:
                if class_names is None:
                    class_names = classes
                color = plt.get_cmap(cmap)(float(j) / len(clf_names))
                ax.plot(
                    mean_predicted_value_dict[i], fraction_of_positives_dict[i],
                    marker='s', ls='-', color=color, lw=2,
                    label=f'Class {class_names[i]}, {clf_names[j]}',
                )

    # Plot the diagonal line for reference
    ax.plot([0, 1], [0, 1], ls='--', lw=1, c='gray')

    # Set plot title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel('Mean predicted value', fontsize=text_fontsize)
    ax.set_ylabel('Fraction of positives', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_ylim([-0.05, 1.05])

    # Define the desired number of ticks
    num_ticks = 10

    # Set x-axis ticks and labels
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator( (ax.get_xlim()[1] / 10) ))
    ax.xaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    ax.yaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    
    # Enable grid and display legend
    ax.grid(True)
    ax.legend(
        loc='lower right', 
        title='Classifier',
        alignment='left'
    )
    plt.tight_layout()
    return ax


def plot_classifier_eval(
    y_true,
    y_pred,
    title=None,
    ax=None,
    figsize=None,
    title_fontsize="large",                          
    text_fontsize="medium",
    cmap='viridis', 
    x_tick_rotation=0, 
    labels=None,
    normalize=None,
    digits=3,
):
    """
    Generates various evaluation plots for a classifier, including confusion matrix, precision-recall curve, and ROC curve.

    This function provides a comprehensive view of a classifier's performance through multiple plots,
    helping in the assessment of its effectiveness and areas for improvement.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like, shape (n_samples,)
        Predicted target values from the classifier.

    title : string, optional
        Title of the generated plot. If None, no title is set.

    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the figures. If None, new axes are created.

    figsize : tuple of int, optional
        Tuple denoting figure size of the plot, e.g. (10, 10). Defaults to None.

    title_fontsize : string or int, optional, default="large"
        Font size for the plot title. Use e.g. "small", "medium", "large" or integer values.

    text_fontsize : string or int, optional, default="medium"
        Font size for the text in the plot. Use e.g. "small", "medium", "large" or integer values.

    cmap : string or matplotlib.colors.Colormap, optional, default='viridis'
        Colormap used for plotting. View Matplotlib Colormap documentation for available options.

    x_tick_rotation : int, optional, default=0
        Rotates x-axis tick labels by the specified angle.

    labels : list of string, optional
        List of labels for the classes. If None, labels are automatically generated based on the class indices.

    normalize : {'true', 'pred', 'all', None}, optional
        Normalizes the confusion matrix according to the specified mode. Defaults to None.
        - 'true': Normalizes by true (actual) values.
        - 'pred': Normalizes by predicted values.
        - 'all': Normalizes by total values.
        - None: No normalization.

    digits : int, optional, default=3
        Number of digits for formatting floating point values in the plots.

        .. versionadded:: 0.3.9

    Returns
    -------
    None

    Notes
    -----
    The function generates and displays multiple evaluation plots. Ensure that `y_true` and `y_pred`
    have the same shape and contain valid class labels. The `normalize` parameter is applicable
    to the confusion matrix plot. Adjust `cmap` and `x_tick_rotation` to customize the appearance
    of the plots.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Confusion Matrix

        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.naive_bayes import GaussianNB
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = GaussianNB()
        >>> model.fit(X_train, y_train)
        >>> y_val_pred = model.predict(X_val)
        >>> skplt.metrics.plot_classifier_eval(
        >>>     y_val, y_val_pred,
        >>>     title='val',
        >>> );
    """
    figsize = (8, 3) if figsize is None else figsize
    title = '' if title is None else title
    if ax is None:
        # Create a figure with two subplots, adjusting the width ratios
        fig, ax = plt.subplots(
            1, 2, 
            figsize=figsize, 
            gridspec_kw={'width_ratios': [5, 5]}
        )
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = unique_labels(y_true, y_pred)    
    if labels is None:
        labels = classes
    else:
        labels = np.asarray(labels)
        validate_labels(classes, labels, "labels")
        
    
    # Generate the classification report
    report = classification_report(
        y_true, 
        y_pred, 
        labels=labels,
        digits=digits,
        zero_division=np.nan,
    )
    # Generate the confusion matrix
    cm = confusion_matrix(
        y_true, 
        y_pred,
        labels=labels,
        normalize = normalize,
    )
    cm = np.around(cm, decimals=2)
    
    # Plot the classification report on the first subplot
    ax[0].axis('off')
    ax[0].set_title(f'{title.capitalize()} Classification Report\n', fontsize=11)
    ax[0].text(
        0, 0.5, '\n'*3 + report, 
        ha='left', va='center', 
        fontfamily='monospace',
        fontsize=8,
    )
    
    # Choose a colormap
    cmap_ = plt.get_cmap(cmap)
    
    # Plot the confusion matrix on the second subplot
    cax = ax[1].matshow(
        cm, 
        cmap=cmap_, 
        aspect='auto'
    )
    
    # Remove the edge of the matshow
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    
    # Also remove the colorbar edge
    cbar = fig.colorbar(mappable=cax, ax=ax[1])
    cbar.outline.set_edgecolor('none')
    
    # Annotate the matrix with dynamic text color
    threshold = cm.max() / 2.0
    for (i, j), val in np.ndenumerate(cm):
        # val == cm[i, j]
        cmap_method = (
            cmap_.get_over 
            if val > threshold else 
            cmap_.get_under
        )
        # Get the color at the top end of the colormap
        rgba = cmap_method()  # Get the RGB values
        
        # Calculate the luminance of this color
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        
        # If luminance is low (dark color), use white text; otherwise, use black text
        text_color = {True:'w', False:'k'}[(luminance < 0.5)]
        ax[1].text(
            j, i, f'{val}', 
            ha='center', va='center', 
            fontsize=text_fontsize, 
            color=text_color,
        )
    
    # Set title and axis labels
    ax[1].set_title(
        f'{title.capitalize()} Confusion Matrix\n', 
        fontsize=title_fontsize
    )
    ax[1].set_xlabel(
        'Predicted Labels', 
        fontsize=text_fontsize
    )
    ax[1].set_ylabel(
        'True Labels', 
        fontsize=text_fontsize
    )    
    # Set class labels for x and y axis
    ax[1].set_xticks(np.arange(len(labels)))
    ax[1].set_yticks(np.arange(len(labels)))
    ax[1].set_xticklabels(
        labels,
        fontsize=text_fontsize,
    )
    ax[1].set_yticklabels(
        labels, 
        fontsize=text_fontsize,
        rotation=x_tick_rotation
    )
    
    # Move x-axis labels to the bottom and y-axis labels to the right
    ax[1].xaxis.set_label_position('bottom')
    ax[1].xaxis.tick_bottom()
    ax[1].yaxis.set_label_position('left')
    ax[1].yaxis.tick_left()
    
    # Adjust layout with additional space
    plt.tight_layout()    
    # Show the plot
    # plt.show()
    return fig


def plot_confusion_matrix(
    y_true, 
    y_pred, 
    labels=None, 
    true_labels=None,
    pred_labels=None, 
    title=None, 
    normalize=False,
    hide_zeros=False, 
    hide_counts=False, 
    x_tick_rotation=0, 
    ax=None,
    figsize=None, 
    cmap='Blues', 
    title_fontsize="large",
    text_fontsize="medium", 
    show_colorbar=True,
):
    """
    Generates a confusion matrix plot from predictions and true labels.

    The confusion matrix is a summary of prediction results that shows the counts of true
    and false positives and negatives for each class. This function also provides options for
    normalizing, hiding zero values, and customizing the plot appearance.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like, shape (n_samples,)
        Estimated targets as returned by a classifier.

    labels : array-like, shape (n_classes), optional
        List of labels to index the matrix. This may be used to reorder or select a subset
        of labels. If None, labels appearing at least once in `y_true` or `y_pred` are used
        in sorted order. (new in v0.2.5)

    true_labels : array-like, optional
        The true labels to display. If None, all labels are used.

    pred_labels : array-like, optional
        The predicted labels to display. If None, all labels are used.

    title : string, optional
        Title of the generated plot. Defaults to "Confusion Matrix" if `normalize` is True.
        Otherwise, defaults to "Normalized Confusion Matrix".

    normalize : bool, optional, default=False
        If True, normalizes the confusion matrix before plotting.

    hide_zeros : bool, optional, default=False
        If True, cells containing a value of zero are not plotted.

    hide_counts : bool, optional, default=False
        If True, counts are not overlaid on the plot.

    x_tick_rotation : int, optional, default=0
        Rotates x-axis tick labels by the specified angle. Useful when labels overlap.

    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the confusion matrix. If None, a new set of axes is created.

    figsize : tuple of int, optional
        Tuple denoting figure size of the plot, e.g., (6, 6). Defaults to None.

    cmap : None, str, or matplotlib.colors.Colormap, optional, default='viridis'
        Colormap used for plotting. Options include 'viridis', 'plasma', 'inferno', etc.
        Refer to the Matplotlib Colormap documentation for available choices.

    title_fontsize : string or int, optional, default="large"
        Font size for the plot title. Use "small", "medium", "large", or integer values.

    text_fontsize : string or int, optional, default="medium"
        Font size for text in the plot. Use "small", "medium", "large", or integer values.

    show_colorbar : bool, optional, default=True
        If False, the colorbar is not displayed.

        .. versionadded:: 0.3.9

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.

    Notes
    -----
    Ensure that `y_true` and `y_pred` have the same shape and contain valid class labels.
    The `normalize` parameter applies only to the confusion matrix plot. Adjust `cmap` and
    `x_tick_rotation` to customize the appearance of the plot. The `show_colorbar` parameter
    controls whether a colorbar is displayed.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Confusion Matrix

        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.naive_bayes import GaussianNB
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = GaussianNB()
        >>> model.fit(X_train, y_train)
        >>> y_val_pred = model.predict(X_val)
        >>> skplt.metrics.plot_confusion_matrix(
        >>>     y_val, y_val_pred,
        >>> );
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        classes = unique_labels(y_true, y_pred)
    else:
        classes = np.asarray(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    if true_labels is None:
        true_classes = classes
    else:
        validate_labels(classes, true_labels, "true_labels")

        true_label_indexes = np.isin(classes, true_labels)

        true_classes = classes[true_label_indexes]
        cm = cm[true_label_indexes]

    if pred_labels is None:
        pred_classes = classes
    else:
        validate_labels(classes, pred_labels, "pred_labels")

        pred_label_indexes = np.isin(classes, pred_labels)

        pred_classes = classes[pred_label_indexes]
        cm = cm[:, pred_label_indexes]


    image = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))

    if show_colorbar == True:
        plt.colorbar(mappable=image)

    thresh = cm.max() / 2.

    if not hide_counts:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if not (hide_zeros and cm[i, j] == 0):
                ax.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=text_fontsize,
                        color="white" if cm[i, j] > thresh else "black")

    # Set title, labels, and formatting
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    elif normalize:
        ax.set_title('Normalized Confusion Matrix', fontsize=title_fontsize)
    else:
        ax.set_title('Confusion Matrix', fontsize=title_fontsize)
    ax.set_xlabel('Predicted label', fontsize=text_fontsize)
    ax.set_ylabel('True label', fontsize=text_fontsize)
        
    x_tick_marks = np.arange(len(pred_classes))
    y_tick_marks = np.arange(len(true_classes))
    ax.set_xticks(x_tick_marks)
    ax.set_xticklabels(pred_classes, fontsize=text_fontsize,
                       rotation=x_tick_rotation)
    ax.set_yticks(y_tick_marks)
    ax.set_yticklabels(true_classes, fontsize=text_fontsize)
    
    ax.grid(False)
    plt.tight_layout()
    return ax


@deprecated(
    'This will be removed in v0.5.0. Please use '
    'scikitplot.metrics.plot_roc instead.'
)
def plot_roc_curve(y_true, y_probas, title='ROC Curves',
                   curves=('micro', 'macro', 'each_class'),
                   ax=None, figsize=None, cmap='nipy_spectral',
                   title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "macro", "each_class")`
            i.e. "micro" for micro-averaged curve, "macro" for macro-averaged
            curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc_curve(y_test, y_probas)

        .. image:: /images/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    if 'micro' not in curves and 'macro' not in curves and \
            'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro", "macro", or "each_class"')

    classes = np.unique(y_true)
    probas = y_probas

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true, probas[:, i],
                                      pos_label=classes[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in fpr:
        i += 1
        micro_key += str(i)

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    fpr[micro_key], tpr[micro_key], _ = roc_curve(y_true.ravel(),
                                                  probas.ravel())
    roc_auc[micro_key] = auc(fpr[micro_key], tpr[micro_key])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[x] for x in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    macro_key = 'macro'
    i = 0
    while macro_key in fpr:
        i += 1
        macro_key += str(i)
    fpr[macro_key] = all_fpr
    tpr[macro_key] = mean_tpr
    roc_auc[macro_key] = auc(fpr[macro_key], tpr[macro_key])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr[i], tpr[i], lw=2, color=color,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(classes[i], roc_auc[i]))

    if 'micro' in curves:
        ax.plot(fpr[micro_key], tpr[micro_key],
                label='micro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc[micro_key]),
                color='deeppink', linestyle=':', linewidth=4)

    if 'macro' in curves:
        ax.plot(fpr[macro_key], tpr[macro_key],
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc[macro_key]),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax


def plot_roc(
    y_true,
    y_probas,
    title='ROC AUC Curves',
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    cmap=None,
    class_index=1,
    multi_class=None,
    class_names=None,
    classes_to_plot=None,
    plot_micro=True,
    plot_macro=True,
    show_labels=True,
    digits=3,
):
    """
    Generates the ROC AUC curves from labels and predicted scores/probabilities.

    The ROC (Receiver Operating Characteristic) curve plots the true positive rate 
    against the false positive rate at various threshold settings. The AUC (Area Under 
    the Curve) represents the degree of separability achieved by the classifier. This 
    function supports both binary and multiclass classification tasks.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like, shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the ``class_index``.

    title : str, optional, default='ROC AUC Curves'
        Title of the generated plot.

    ax : matplotlib.axes.Axes, optional, default=None
        The axes on which to plot. If None, a new figure and axes are created.

    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.

    cmap : None, str or matplotlib.colors.Colormap, optional, default='viridis'
        Colormap used for plotting. Options include 'viridis', 'plasma', 'inferno', etc.
        See Matplotlib Colormap documentation for available choices.

    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification. Ignored for binary classification.

    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot for the entire probability distribution.

    class_names : list of str, optional, default=None
        List of class names for the legend. Order should match the order of classes in `y_probas`.

    classes_to_plot : list-like, optional, default=None
        Specific classes to plot. If a given class does not exist, it will be ignored. 
        If None, all classes are plotted.

    plot_micro : bool, optional, default=False
        Whether to plot the micro-average ROC AUC curve.

    plot_macro : bool, optional, default=False
        Whether to plot the macro-average ROC AUC curve.

    show_labels : bool, optional, default=True
        Whether to display the legend labels.

    digits : int, optional, default=3
        Number of digits for formatting ROC AUC values in the plot.

        .. versionadded:: 0.3.9

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted ROC AUC curves.

    Notes
    -----
    The implementation is specific to binary classification. For multiclass problems, 
    the 'ovr' or 'multinomial' strategies can be used. When ``multi_class='ovr'``, 
    the plot focuses on the specified class (``class_index``).

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: ROC AUC Curves

        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.naive_bayes import GaussianNB
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = GaussianNB()
        >>> model.fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.metrics.plot_roc(
        >>>     y_val, y_probas,
        >>> );
    """
    title_pad = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)    

    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        # 1D y_probas (single class probabilities)
        if y_probas.ndim == 1:
            # Combine into a two-column
            y_probas = np.column_stack([1 - y_probas, y_probas])
    # Handle multi-class classification    
    elif len(np.unique(y_true)) > 2:
        if multi_class == 'ovr':
            # Binarize y_true for multiclass classification
            y_true = label_binarize(y_true, classes=np.unique(y_true))[:, class_index]
            # Handle 1D y_probas (single class probabilities)
            if y_probas.ndim == 1:
                # Combine into a two-column binary format OvR
                y_probas = np.column_stack([1 - y_probas, y_probas])
            else:
                # Combine into a two-column binary format OvR
                y_probas = y_probas[:, class_index]
                y_probas = np.column_stack([1 - y_probas, y_probas])
                
            # Add a subtitle indicating the use of the One-vs-Rest strategy
            plt.suptitle(
                t="One-vs-Rest (OVR) strategy for multi-class classification.",
                fontsize=text_fontsize, x=0.512, y=0.902,
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='w', boxstyle='round,pad=0.2')
            )
            title_pad = 23
        elif multi_class in ['multinomial', None]:
            if y_probas.ndim == 1:
                raise ValueError(
                    "For multinomial classification, `y_probas` must be 2D."
                    "For a 1D `y_probas` with more than 2 classes in `y_true`, "
                    "only 'ovr' multi-class strategy is supported."
                )
        else:
            raise ValueError("Unsupported `multi_class` strategy.")

    # Initialize dictionaries to store
    fpr_dict, tpr_dict = {}, {}

    # Get unique classes and filter those to be plotted
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError(
            'Cannot calculate Curve for classes with only one category.'
        )
    classes_to_plot = classes if classes_to_plot is None else classes_to_plot
    indices_to_plot = np.isin(classes, classes_to_plot)
    
    # Binarize y_true for multiclass classification, for micro
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    # Loop for all classes to get different class
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(
            y_true, y_probas[:, i], pos_label=classes[i]
        )
        roc_auc = auc(
            fpr_dict[i], tpr_dict[i]
        )        
        if to_plot:
            if class_names is None:
                class_names = classes
            color = plt.get_cmap(cmap)( float(i) / len(classes) )
            ax.plot(
                fpr_dict[i], tpr_dict[i],
                ls='-', lw=2, color=color,
                label=(
                    f'Class {classes[i]} '
                    f'(area = {roc_auc:0>{digits}.{digits}f})'
                ),
            )

    # Whether or to plot macro or micro
    if plot_micro:
        fpr, tpr, _ = roc_curve(
            y_true_bin.ravel(), y_probas.ravel()
        )
        roc_auc = auc(
            fpr, tpr
        )
        ax.plot(
            fpr, tpr,
            ls=':', lw=3, color='deeppink',
            label=(
                'micro-average '
                f'(area = {roc_auc:0>{digits}.{digits}f})'
            ),
        )

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate(
            [ fpr_dict[i] for i in range(len(classes)) ]
        ))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += np.interp(
                all_fpr, fpr_dict[i], tpr_dict[i]
            )

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(
            all_fpr, mean_tpr
        )
        ax.plot(
            all_fpr, mean_tpr,
            ls=':', lw=3, color='navy',
            label=(
                'macro-average '
                f'(area = {roc_auc:0>{digits}.{digits}f})'
            ),
        )

    # Plot the baseline
    ax.plot([0, 1], [0, 1], ls='--', lw=1, c='gray', )  # label='Baseline'

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    # Define the desired number of ticks
    # num_ticks = 10

    ## Set x-axis ticks and labels
    ## ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator( (ax.get_xlim()[1] / 10) ))
    # ax.xaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    # ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    # ax.yaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    # ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )

    # Enable grid and display legend
    ax.grid(True)
    if show_labels:
        ax.legend(
            loc='lower right', 
            fontsize=text_fontsize, 
            title='ROC AUC', 
            alignment='left'
        )

    plt.tight_layout()
    return ax


@deprecated(
    'This will be removed in v0.5.0. Please use '
    'scikitplot.metrics.plot_precision_recall instead.'
)
def plot_precision_recall_curve(
    y_true, y_probas,
    title='Precision-Recall Curve',
    curves=('micro', 'each_class'), ax=None,
    figsize=None, cmap='nipy_spectral',
    title_fontsize="large",
    text_fontsize="medium"
):
    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "Precision-Recall curve".

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "each_class")`
            i.e. "micro" for micro-averaged curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_precision_recall_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: /images/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if 'micro' not in curves and 'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro" or "each_class"')

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true, probas[:, i], pos_label=classes[i])

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    for i in range(len(classes)):
        average_precision[i] = average_precision_score(y_true[:, i],
                                                       probas[:, i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in precision:
        i += 1
        micro_key += str(i)

    precision[micro_key], recall[micro_key], _ = precision_recall_curve(
        y_true.ravel(), probas.ravel())
    average_precision[micro_key] = average_precision_score(y_true, probas,
                                                           average='micro')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall[i], precision[i], lw=2,
                    label='Precision-recall curve of class {0} '
                          '(area = {1:0.3f})'.format(classes[i],
                                                     average_precision[i]),
                    color=color)

    if 'micro' in curves:
        ax.plot(recall[micro_key], precision[micro_key],
                label='micro-average Precision-recall curve '
                      '(area = {0:0.3f})'.format(average_precision[micro_key]),
                color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    return ax


def plot_precision_recall(
    y_true,
    y_probas,
    title='Precision-Recall AUC Curves',
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    cmap=None,
    class_index=1,
    multi_class=None,
    class_names=None,
    classes_to_plot=None,   
    plot_micro=True,
    plot_macro=False,
    show_labels=True,
    digits=4,
    area='pr_auc',
):
    """
    Generates the Precision-Recall AUC Curves from labels and predicted scores/probabilities.
    
    The Precision-Recall curve plots the precision against the recall for different threshold values. 
    The area under the curve (AUC) represents the classifier's performance. This function supports 
    both binary and multiclass classification tasks.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like, shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the ``class_index``.

    title : str, optional, default='Precision-Recall AUC Curves'
        Title of the generated plot.

    ax : matplotlib.axes.Axes, optional, default=None
        The axes on which to plot. If None, a new figure and axes are created.

    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.

    cmap : None, str or matplotlib.colors.Colormap, optional, default='viridis'
        Colormap used for plotting. Options include 'viridis', 'plasma', 'inferno', etc.
        See Matplotlib Colormap documentation for available choices.

    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification. Ignored for binary classification.

    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot for the entire probability distribution.

    class_names : list of str, optional, default=None
        List of class names for the legend. Order should match the order of classes in `y_probas`.

    classes_to_plot : list-like, optional, default=None
        Specific classes to plot. If a given class does not exist, it will be ignored. 
        If None, all classes are plotted.

    plot_micro : bool, optional, default=False
        Whether to plot the micro-average ROC AUC curve.

    plot_macro : bool, optional, default=False
        Whether to plot the macro-average ROC AUC curve.

    show_labels : bool, optional, default=True
        Whether to display the legend labels.

    digits : int, optional, default=3
        Number of digits for formatting PR AUC values in the plot.

        .. versionadded:: 0.3.9

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted PR AUC curves.

    Notes
    -----
    The implementation is specific to binary classification. For multiclass problems, 
    the 'ovr' or 'multinomial' strategies can be used. When ``multi_class='ovr'``, 
    the plot focuses on the specified class (``class_index``).

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Precision-Recall AUC Curves

        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.naive_bayes import GaussianNB
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = GaussianNB()
        >>> model.fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.metrics.plot_precision_recall(
        >>>     y_val, y_probas,
        >>> );
    """
    title_pad = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)    

    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        # 1D y_probas (single class probabilities)
        if y_probas.ndim == 1:
            # Combine into a two-column
            y_probas = np.column_stack([1 - y_probas, y_probas])
    # Handle multi-class classification    
    elif len(np.unique(y_true)) > 2:
        if multi_class == 'ovr':
            # Binarize y_true for multiclass classification
            y_true = label_binarize(y_true, classes=np.unique(y_true))[:, class_index]
            # Handle 1D y_probas (single class probabilities)
            if y_probas.ndim == 1:
                # Combine into a two-column binary format OvR
                y_probas = np.column_stack([1 - y_probas, y_probas])
            else:
                # Combine into a two-column binary format OvR
                y_probas = y_probas[:, class_index]
                y_probas = np.column_stack([1 - y_probas, y_probas])
                
            # Add a subtitle indicating the use of the One-vs-Rest strategy
            plt.suptitle(
                t="One-vs-Rest (OVR) strategy for multi-class classification.",
                fontsize=text_fontsize, x=0.512, y=0.902,
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='w', boxstyle='round,pad=0.2')
            )
            title_pad = 23
        elif multi_class in ['multinomial', None]:
            if y_probas.ndim == 1:
                raise ValueError(
                    "For multinomial classification, `y_probas` must be 2D."
                    "For a 1D `y_probas` with more than 2 classes in `y_true`, "
                    "only 'ovr' multi-class strategy is supported."
                )
        else:
            raise ValueError("Unsupported `multi_class` strategy.")

    # Initialize dictionaries to store
    precision_dict, recall_dict = {}, {}

    # Get unique classes and filter those to be plotted
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError(
            'Cannot calculate Curve for classes with only one category.'
        )
    classes_to_plot = classes if classes_to_plot is None else classes_to_plot
    indices_to_plot = np.isin(classes, classes_to_plot)
    
    # Binarize y_true for multiclass classification, for micro
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    # Loop for all classes to get different class
    for i, to_plot in enumerate(indices_to_plot):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            y_true, y_probas[:, i], pos_label=classes[i]
        )
        if area == 'pr_auc' :
            pr_auc = auc(
                recall_dict[i], precision_dict[i]
            )
        else:
            pr_auc = average_precision_score(
                y_true_bin[:, i], y_probas[:, i], #pos_label=classes[i]
            )
        if to_plot:
            if class_names is None:
                class_names = classes            
            color = plt.get_cmap(cmap)( float(i) / len(classes) )
            ax.plot(
                precision_dict[i], recall_dict[i],
                ls='-', lw=2, color=color,
                label=(
                    f'Class {classes[i]} '
                    f'(area = {pr_auc:0>{digits}.{digits}f})'
                ),
            )

    # Whether or to plot macro or micro
    if plot_micro:
        precision, recall, _ = precision_recall_curve(
            y_true_bin.ravel(), y_probas.ravel()
        )
        if area == 'pr_auc' :
            pr_auc = auc(
                recall, precision
            )
        else:
            pr_auc = average_precision_score(
                y_true_bin.ravel(), y_probas.ravel()
            )
        ax.plot(
            precision, recall,
            ls=':', lw=3, color='deeppink',
            label=(
                'micro-average '
                f'(area = {pr_auc:0>{digits}.{digits}f})'
            ),
        )

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_precision = np.unique(np.concatenate(
            [ precision_dict[i] for i in range(len(classes)) ]
        ))
        # Then interpolate all ROC curves at this points
        mean_recall = np.zeros_like(all_precision)
        for i in range(len(classes)):
            mean_recall += np.interp(
                all_precision, precision_dict[i], recall_dict[i]
            )

        # Finally average it and compute AUC
        mean_recall /= len(classes)
        pr_auc = average_precision_score(
            y_true_bin.ravel(), y_probas.ravel(), average='macro'
        )
        ax.plot(
            all_precision, mean_recall,
            ls=':', lw=3, color='navy',
            label=(
                'macro-average '
                f'(area = {pr_auc:0>{digits}.{digits}f})'
            ),
        )

    # Plot the baseline
    ax.plot([0, 1], [1, 0], ls='--', lw=1, c='gray', )  # label='Baseline'

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel('Recall', fontsize=text_fontsize)
    ax.set_ylabel('Precision', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    # Define the desired number of ticks
    # num_ticks = 10

    ## Set x-axis ticks and labels
    ## ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator( (ax.get_xlim()[1] / 10) ))
    # ax.xaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    # ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    # ax.yaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    # ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )

    # Enable grid and display legend
    ax.grid(True)
    if show_labels:
        ax.legend(
            loc='lower left', 
            fontsize=text_fontsize, 
            title=f'PR-AUC by {area}', 
            alignment='left'
        )

    plt.tight_layout()
    return ax


def plot_silhouette(
    X, 
    cluster_labels, 
    title='Silhouette Analysis',
    metric='euclidean', 
    copy=True, 
    ax=None, 
    figsize=None,
    cmap='nipy_spectral', 
    title_fontsize="large",
    text_fontsize="medium", 
    digits=3,
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

    title : str, optional, default='Silhouette Analysis'
        Title of the generated plot.

    metric : str or callable, optional, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.
        If metric is a string, it must be one of the options allowed by 
        `sklearn.metrics.pairwise.pairwise_distances`. If `X` is the distance array itself, 
        use "precomputed" as the metric.

    copy : bool, optional, default=True
        Determines whether `fit` is used on `clf` or on a copy of `clf`.

    ax : matplotlib.axes.Axes, optional, default=None
        The axes upon which to plot the curve. If None, a new figure and axes are created.

    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.

    cmap : str or matplotlib.colors.Colormap, optional, default='viridis'
        Colormap used for plotting the projection. See Matplotlib Colormap documentation 
        for available options. 
        - https://matplotlib.org/users/colormaps.html

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot.

    digits : int, optional, default=3
        Number of digits for formatting output floating point values. 

        .. versionadded:: 0.3.9

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was drawn.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Silhouette Plot
    
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
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    cluster_labels = np.asarray(cluster_labels)

    le = LabelEncoder()
    cluster_labels_encoded = le.fit_transform(cluster_labels)

    n_clusters = len(np.unique(cluster_labels))

    silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)

    sample_silhouette_values = silhouette_samples(X, cluster_labels,
                                                  metric=metric)
    
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[
            cluster_labels_encoded == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.get_cmap(cmap)(float(i) / n_clusters)

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(le.classes_[i]),
                fontsize=text_fontsize)

        y_lower = y_upper + 10

    ax.axvline(
        x=silhouette_avg, color="red", linestyle="--",
        label='Silhouette score: {0:.{digits}f}'.format(silhouette_avg, digits=2)
    )

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Silhouette coefficient values', fontsize=text_fontsize)
    ax.set_ylabel('Cluster label', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xticks(np.arange(-0.1, 1.0, 0.2))
    ax.set_yticks([])  # Clear the y-axis labels / ticks

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10 + 10])
    
    # Display legend
    ax.legend(loc='best', fontsize=text_fontsize)

    plt.tight_layout()
    return ax