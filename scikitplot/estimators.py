"""
This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.

The :mod:`scikitplot.estimators` module includes plots built specifically for
scikit-learn estimator (classifier/regressor) instances e.g. Random Forest.
You can use your own estimators, but these plots assume specific properties
shared by scikit-learn estimators. The specific requirements are documented per
function.
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
    'plot_learning_curve',
    'plot_feature_importances',
]


def plot_learning_curve(
    clf, 
    X, 
    y, 
    title='Learning Curves',
    ax=None, 
    figsize=None, 
    title_fontsize="large",
    text_fontsize="medium",
    cv=None, 
    scoring=None,
    train_sizes=None, 
    shuffle=False, 
    random_state=None,
    n_jobs=1, 
):
    """
    Generates a plot of the train and test learning curves for a classifier.

    The learning curves plot the performance of a classifier as a function of the number of training samples.
    This helps in understanding how well the classifier performs with different amounts of training data.

    Parameters
    ----------
    clf : object
        Classifier instance that implements `fit` and `predict` methods.

    X : array-like, shape (n_samples, n_features)
        Training data, where `n_samples` is the number of samples and `n_features` is the number of features.

    y : array-like, shape (n_samples,) or (n_samples, n_features), optional
        Target relative to `X` for classification or regression; None for unsupervised learning.

    title : str, optional, default="Learning Curves"
        Title of the generated plot.

    cv : int, cross-validation generator, or iterable, optional
        Determines the cross-validation strategy to use for splitting:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds,
        - An object to be used as a cross-validation generator,
        - An iterable yielding train/test splits.

        For integer/None inputs, if `y` is binary or multiclass, `StratifiedKFold` is used.
        If the estimator is not a classifier or if `y` is neither binary nor multiclass, `KFold` is used.

    shuffle : bool, optional, default=True
        Whether to shuffle the training data before splitting using cross-validation.

    random_state : int or RandomState, optional
        Pseudo-random number generator state used for random sampling.

    train_sizes : iterable, optional
        Determines the training sizes used to plot the learning curve.
        If None, `np.linspace(.1, 1.0, 5)` is used.

    n_jobs : int, optional, default=1
        Number of jobs to run in parallel.

    scoring : str, callable, or None, optional, default=None
        A string (see scikit-learn model evaluation documentation) or a scorer callable object/function
        with signature `scorer(estimator, X, y)`.

    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the curve. If None, a new figure and axes are created.

    figsize : tuple of int, optional
        Tuple denoting figure size of the plot, e.g., (6, 6). Defaults to None.

    title_fontsize : str or int, optional, default="large"
        Font size for the plot title. Use e.g., "small", "medium", "large" or integer values.

    text_fontsize : str or int, optional, default="medium"
        Font size for the text in the plot. Use e.g., "small", "medium", "large" or integer values.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was drawn.

    Notes
    -----
    If `cv` is not specified, 3-fold cross-validation is used by default. The plot will show the learning curves
    for training and test data across different training sizes.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Learning Curves

        >>> from sklearn.datasets import load_digits as data_10_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.naive_bayes import GaussianNB
        >>> import scikitplot as skplt
        >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = GaussianNB()
        >>> model.fit(X_train, y_train)
        >>> y_val_pred = model.predict(X_val)
        >>> skplt.estimators.plot_learning_curve(
        >>>     model, X_val, y_val_pred,
        >>> );
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if train_sizes is None:
        train_sizes = np.linspace(.1, 1.0, 5)

    train_sizes, train_scores, test_scores = learning_curve(
        clf, 
        X, 
        y, 
        cv=cv, 
        scoring=scoring,
        train_sizes=train_sizes, 
        shuffle=shuffle, 
        random_state=random_state,
        n_jobs=n_jobs,
    )    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    ax.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1, color="r"
    )
    ax.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1, color="g"
    )
    ax.plot(
        train_sizes, train_scores_mean, 'o-', color="r",
        label="Training score"
    )
    ax.plot(
        train_sizes, test_scores_mean, 'o-', color="g",
        label="Cross-validation score"
    )    
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Training examples", fontsize=text_fontsize)
    ax.set_ylabel("Score", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid()
    ax.legend(loc="best", fontsize=text_fontsize)

    return ax


def plot_feature_importances(
    model,
    title='Feature Importances',
    ax=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    cmap='PiYG',
    orientation='vertical',
    x_tick_rotation=None,
    bar_padding=11,
    display_labels=True,
    class_index=None,
    threshold=None,
    order=None,
    feature_names=None,
    digits=4,
):
    """
    Generates a plot of a sklearn model's feature importances.

    This function handles different types of classifiers and their respective
    feature importances (``feature_importances_``) or coefficient (``coef_``) attributes,
    if not provide its compute sklearn permutation importances.
    It supports models wrapped in pipelines.

    Supports models like:
        - :class:`~sklearn.linear_model.LinearRegression`
        - :class:`~sklearn.linear_model.LogisticRegression`
        - :class:`~sklearn.neighbors.KNeighborsClassifier`
        - :class:`~sklearn.svm.LinearSVC`
        - :class:`~sklearn.svm.SVC`
        - :class:`~sklearn.tree.DecisionTreeClassifier`
        - :class:`~sklearn.ensemble.RandomForestClassifier`
        - :class:`~sklearn.decomposition.PCA`
        - `xgboost Python API <https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn>`_
        - `catboost Python API <https://catboost.ai/en/docs/concepts/python-quickstart>`_

    Parameters
    ----------
    model : estimator object
        A fitted sklearn model or pipeline containing a classifier.

    title : str, optional
        Title of the generated plot.
        Defaults to "Feature Importances".

    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the curve. If None, the plot is drawn on
        a new set of axes.

    figsize : tuple, optional
        Tuple denoting figure size of the plot e.g. (6, 6). Defaults to None.

    title_fontsize : str or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or
        integer-values. Defaults to "large".

    text_fontsize : str or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or
        integer-values. Defaults to "medium".

    cmap : str or matplotlib.colors.Colormap, optional, default='PiYG'
        Colormap used for plotting.
        - See Matplotlib Colormap documentation for options.

    orientation : {'vertical' | 'v' | 'y', 'horizontal' | 'h' | 'y'}, optional
        Orientation of the bar plot. Defaults to 'vertical'.

    x_tick_rotation : int, optional
        Rotates x-axis tick labels by the specified angle. Defaults to None
        (automatically set based on orientation).

    bar_padding : float, optional
        Padding between bars in the plot. Defaults to 11.

    display_labels : bool, optional
        Whether to display the bar labels. Defaults to True.

    class_index : int, optional
        Index of the class of interest for multi-class classification.
        Defaults to None.

    threshold : float, optional
        Threshold for filtering features by absolute importance. Only
        features with an absolute importance greater than this threshold will
        be plotted. Defaults to None (plot all features).

    order : {'ascending', 'descending', None}, optional
        Order of feature importance in the plot. Defaults to None
        (automatically set based on orientation).

    feature_names : list of str, optional
        List of feature names corresponding to the features. If None, feature
        indices are used.

    digits : int, optional, default=4
        Number of digits for formatting AUC values in the plot.

        .. versionadded:: 0.3.9

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Feature Importances
    
       >>> from sklearn.datasets import load_digits as data_10_classes
       >>> from sklearn.model_selection import train_test_split
       >>> from sklearn.ensemble import RandomForestClassifier
       >>> import scikitplot as skplt
       >>> X, y = data_10_classes(return_X_y=True, as_frame=False)
       >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
       >>> model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
       >>> skplt.estimators.plot_feature_importances(
       >>>     model,
       >>>     orientation='y',
       >>>     figsize=(12, 5),
       >>> );
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    # Handle pipelines
    if hasattr(model, 'named_steps'):
        model = model.named_steps[next(reversed(model.named_steps))]

    # Determine the appropriate attribute for feature importances or coefficients
    if hasattr(model, 'feature_importances_'):
        importances = np.asarray(model.feature_importances_)
    # LDA (scikit-learn < 0.24)
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim > 1:  # Multi-class case
            if class_index is not None:
                importances = np.asarray(model.coef_)[class_index]
            else:
                importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.asarray(model.coef_).ravel()
    # PCA
    elif hasattr(model, 'explained_variance_ratio_'):
        importances = np.asarray(model.explained_variance_ratio_)
    else:
        raise TypeError(
            'The estimator does not have an attribute for feature '
            'importances or coefficients.'
        )
    # Obtain feature names
    if feature_names is None:
        # sklearn models
        if hasattr(model, 'feature_names_in_'):
            feature_names = np.asarray(model.feature_names_in_)
        # catboost
        elif hasattr(model, 'feature_names_'):
            feature_names = np.asarray(model.feature_names_)
        else:
            # Ensure feature_names are strings
            feature_names = np.arange(len(importances), dtype=int)
    else:
        feature_names = np.asarray(feature_names)

    # Generate indices
    indices = np.arange(len(importances))

    # Apply filtering based on the threshold
    if threshold is not None:
        mask = np.abs(importances) > threshold
        indices = indices[mask].copy()


    # Apply ordering based on orientation
    if order is None:
        order = 'ascending' if orientation in ['horizontal', 'h', 'x'] else 'descending'
    
    if order == 'descending':
        # Sort the indices based on the importances in descending order
        sorted_indices = np.argsort(importances[indices])[::-1]
        indices = indices[sorted_indices].copy()
    elif order == 'ascending':
        # Sort the indices based on the importances in ascending order
        sorted_indices = np.argsort(importances[indices])
        indices = indices[sorted_indices].copy()

    # Reorder the importances array according to the sorted indices
    importances = importances[indices].copy()
    # Reorder the feature names according to the sorted indices
    feature_names = feature_names[indices].copy()

    # Plot bars based on orientation
    for idx, (col, imp) in enumerate(zip(feature_names, importances)):
        # Default colormap if not provided, 'viridis'
        color = plt.get_cmap(cmap)( float(idx) / len(importances) )
        if orientation in ['vertical', 'v', 'y']:
            bar = ax.bar(x=str(col), height=imp, color=color)
        elif orientation in ['horizontal', 'h', 'x']:
            bar = ax.barh(y=str(col), width=imp, color=color)
        else:
            raise ValueError(
                "Invalid value for orientation: "
                "must be 'vertical' or 'horizontal'."
            )

    # Set default x_tick_rotation based on orientation
    if x_tick_rotation is None:
        x_tick_rotation = (
            0 if orientation in ['horizontal', 'h', 'x'] else 90
        )

    if display_labels:
        for bars in ax.containers:
            ax.bar_label(
                bars,
                fmt=lambda x:'{:0>{digits}.{digits}f}'.format(x, digits=digits),
                fontsize=text_fontsize, 
                rotation=x_tick_rotation,
                padding=bar_padding,
            )
    ax.set_title(title, fontsize=title_fontsize)
    if orientation in ['vertical', 'v', 'y']:
        # Set x-ticks positions and labels
        ax.set_xticks(np.arange(len(feature_names)))
        # ax.set_xticklabels(feature_names, rotation=x_tick_rotation, fontsize=text_fontsize)
        ax.tick_params(axis='x', rotation=x_tick_rotation, labelsize=text_fontsize)
        ax.set_xlabel("Features | Index", fontsize=text_fontsize)
        ax.set_ylabel("Importance", fontsize=text_fontsize)
    elif orientation in ['horizontal', 'h', 'x']:
        # Set y-ticks positions and labels
        ax.set_yticks(np.arange(len(feature_names)))
        # ax.set_yticklabels(feature_names, rotation=x_tick_rotation, fontsize=text_fontsize)
        ax.tick_params(axis='y', rotation=x_tick_rotation, labelsize=text_fontsize)
        ax.set_xlabel("Importance", fontsize=text_fontsize)
        ax.set_ylabel("Features | Index", fontsize=text_fontsize)
    else:
        raise ValueError(
            "Invalid value for orientation: must be "
            "'vertical' or 'horizontal'."
        )
    
    # Adjust plot limits if needed
    if orientation in ['vertical', 'v', 'y']:
        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] * 1.2])  # Increase the upper limit by 15%
    elif orientation in ['horizontal', 'h', 'x']:
        ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] * 1.2])  # Increase the upper limit by 15%

    
    plt.tight_layout()
    plt.legend([f'features: {len(importances)}'])
    return ax, feature_names