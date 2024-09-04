"""
This package/module is designed to be compatible with both Python 2 and Python 3.
The imports below ensure consistent behavior across different Python versions by
enforcing Python 3-like behavior in Python 2.

The :mod:`scikitplot.deciles` module includes plots for machine learning
evaluation decile analysis e.g. Gain, Lift and Decile charts, etc.

References:
[1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L5
"""
# code that needs to be compatible with both Python 2 and Python 3
from __future__ import (
    absolute_import,  # Ensures that all imports are absolute by default, avoiding ambiguity.
    division,         # Changes the division operator `/` to always perform true division.
    print_function,   # Treats `print` as a function, consistent with Python 3 syntax.
    unicode_literals  # Makes all string literals Unicode by default, similar to Python 3.
)
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    label_binarize, LabelEncoder
)
from sklearn.utils import deprecated
from sklearn.utils.multiclass import unique_labels

from .utils.helpers import (
    validate_labels,
    cumulative_gain_curve,
    binary_ks_curve,
)


## Define __all__ to specify the public interface of the module, not required default all above func
__all__ = [
    'print_labels',
    'decile_table',
    'plot_cumulative_gain',
    'plot_lift',
    'plot_lift_decile_wise',
    'plot_ks_statistic',
    'report',
]


def print_labels():
    """
    References
    ----------
    [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L5

    Examples
    --------
    
    .. jupyter-execute::

        import scikitplot as skplt
        skplt.deciles.print_labels()
    """
    print(
        "LABELS INFO:\n\n",
        "prob_min         : Minimum probability in a particular decile\n", 
        "prob_max         : Minimum probability in a particular decile\n",
        "prob_avg         : Average probability in a particular decile\n",
        "cnt_events       : Count of events in a particular decile\n",
        "cnt_resp         : Count of responders in a particular decile\n",
        "cnt_non_resp     : Count of non-responders in a particular decile\n",
        "cnt_resp_rndm    : Count of responders if events assigned randomly in a particular decile\n",
        "cnt_resp_wiz     : Count of best possible responders in a particular decile\n",
        "resp_rate        : Response Rate in a particular decile [(cnt_resp/cnt_cust)*100]\n",
        "cum_events       : Cumulative sum of events decile-wise \n",
        "cum_resp         : Cumulative sum of responders decile-wise \n",
        "cum_resp_wiz     : Cumulative sum of best possible responders decile-wise \n",
        "cum_non_resp     : Cumulative sum of non-responders decile-wise \n",
        "cum_events_pct   : Cumulative sum of percentages of events decile-wise \n",
        "cum_resp_pct     : Cumulative sum of percentages of responders decile-wise \n",
        "cum_resp_pct_wiz : Cumulative sum of percentages of best possible responders decile-wise \n",
        "cum_non_resp_pct : Cumulative sum of percentages of non-responders decile-wise \n",
        "KS               : KS Statistic decile-wise \n",
        "lift             : Cumuative Lift Value decile-wise",
         )


def decile_table(
    y_true, 
    y_prob, 
    labels=True,
    change_deciles=10, 
    digits=3,
):
    """
    Generates the Decile Table from labels and probabilities.

    The Decile Table is created by first sorting the customers by their predicted 
    probabilities, in decreasing order from highest (closest to one) to 
    lowest (closest to zero). The customers are split into equally sized segments, 
    creating groups containing the same number of customers, for example, 10 decile 
    groups each containing 10% of the customer base.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct/actual) target values.

    y_prob : array-like, shape (n_samples, n_classes)
        Prediction probabilities for each class returned by a classifier/algorithm.

    labels : bool, optional, default=True
        If True, prints a legend for the abbreviations of decile table column names.

    change_deciles : int, optional, default=10
        The number of partitions for creating the table. Defaults to 10 for deciles.

    digits : int, optional, default=3
        The decimal precision for the result.

        .. versionadded:: 0.3.9

    Returns
    -------
    pd.DataFrame
        The dataframe `dt` (decile-table) with the deciles and related information.

    References
    ----------
    [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L32

    Examples
    --------
    
    .. jupyter-execute::

        from sklearn.datasets import load_iris as data_3_classes
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        import scikitplot as skplt
        X, y = data_3_classes(return_X_y=True, as_frame=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        skplt.deciles.decile_table(y_test, y_prob[:, 1])
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_prob'] = y_prob
    # df['decile']=pd.qcut(df['y_prob'], 10, labels=list(np.arange(10,0,-1))) 
    # ValueError: Bin edges must be unique

    df.sort_values('y_prob', ascending=False, inplace=True)
    df['decile'] = np.linspace(1, change_deciles+1, len(df), False, dtype=int)

    # dt abbreviation for decile_table
    dt = df.groupby('decile').apply(lambda x: pd.Series([
        np.min(x['y_prob']),
        np.max(x['y_prob']),
        np.mean(x['y_prob']),
        np.size(x['y_prob']),
        np.sum(x['y_true']),
        np.size(x['y_true'][x['y_true'] == 0]),
    ],
        index=(["prob_min", "prob_max", "prob_avg",
                "cnt_cust", "cnt_resp", "cnt_non_resp"])
    )).reset_index()

    dt['prob_min']=dt['prob_min'].round(digits)
    dt['prob_max']=dt['prob_max'].round(digits)
    dt['prob_avg']=round(dt['prob_avg'],digits)
    # dt=dt.sort_values(by='decile',ascending=False).reset_index(drop=True)

    tmp = df[['y_true']].sort_values('y_true', ascending=False)
    tmp['decile'] = np.linspace(1, change_deciles+1, len(tmp), False, dtype=int)

    dt['cnt_resp_rndm'] = np.sum(df['y_true']) / change_deciles
    dt['cnt_resp_wiz'] = tmp.groupby('decile', as_index=False)['y_true'].sum()['y_true']

    dt['resp_rate'] = round(dt['cnt_resp'] * 100 / dt['cnt_cust'], digits)
    dt['cum_cust'] = np.cumsum(dt['cnt_cust'])
    dt['cum_resp'] = np.cumsum(dt['cnt_resp'])
    dt['cum_resp_wiz'] = np.cumsum(dt['cnt_resp_wiz'])
    dt['cum_non_resp'] = np.cumsum(dt['cnt_non_resp'])
    dt['cum_cust_pct'] = round(dt['cum_cust'] * 100 / np.sum(dt['cnt_cust']), digits)
    dt['cum_resp_pct'] = round(dt['cum_resp'] * 100 / np.sum(dt['cnt_resp']), digits)
    dt['cum_resp_pct_wiz'] = round(dt['cum_resp_wiz'] * 100 / np.sum(dt['cnt_resp_wiz']), digits)
    dt['cum_non_resp_pct'] = round(
        dt['cum_non_resp'] * 100 / np.sum(dt['cnt_non_resp']), digits)
    dt['KS'] = round(dt['cum_resp_pct'] - dt['cum_non_resp_pct'], digits)
    dt['lift'] = round(dt['cum_resp_pct'] / dt['cum_cust_pct'], digits)

    if labels is True:
        print_labels()

    plt.tight_layout()
    return dt


def plot_cumulative_gain(
    ## default params
    y_true,
    y_probas,
    *,
    class_names=None,
    multi_class=None,
    class_index=1,
    to_plot_class_index=None,
    ## plotting params
    title='Cumulative Gain Curves',
    ax=None,
    fig=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    cmap=None,
    show_labels=True,
    plot_micro=True,
    plot_macro=False,
    ## additional params
    **kwargs,
):
    """
    Generates the Cumulative Gains Plot from labels and scores/probabilities.
    
    The cumulative gains chart is used to determine the effectiveness of a
    binary classifier. It compares the model's performance with random guessing.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    
    y_probas : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the `class_index`.
    
    class_names : list of str, optional, default=None
        List of class names for the legend. Order should match the order of classes in `y_probas`.
    
    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot for the entire probability distribution.
    
    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification. Ignored for
        binary classification.
    
    to_plot_class_index : list-like, optional, default=None
        Specific classes to plot. If a given class does not exist, it will be ignored. 
        If None, all classes are plotted. e.g. [0, 'cold']
    
    title : str, optional, default='Cumulative Gain Curves'
        Title of the generated plot.

    ax : list of matplotlib.axes.Axes, optional, default=None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).
        Need two axes like [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

    fig : matplotlib.pyplot.figure, optional, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

        .. versionadded:: 0.3.9
    
    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.
    
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
    
    show_labels : bool, optional, default=True
        Whether to display the legend labels.

        .. versionadded:: 0.3.9
    
    plot_micro : bool, optional, default=True
        Whether to plot the micro-average Cumulative Gain curve.

        .. versionadded:: 0.3.9
    
    plot_macro : bool, optional, default=True
        Whether to plot the macro-average Cumulative Gain curve.

        .. versionadded:: 0.3.9
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted cumulative gain curves.
    
    Notes
    -----
    The implementation is specific to binary classification. For multiclass 
    problems, the 'ovr' or 'multinomial' strategies can be used. When 
    `multi_class='ovr'`, the plot focuses on the specified class (`class_index`).
    
    References
    ----------
    [1] http://mlwiki.org/index.php/Cumulative_Gain_Chart
    
    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Cumulative Gain Curves
    
        >>> from sklearn.datasets import load_iris as data_3_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> import scikitplot as skplt
        >>> X, y = data_3_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = LogisticRegression(max_iter=int(1e5), random_state=0).fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.deciles.plot_cumulative_gain(
        >>>     y_val, y_probas,
        >>> );
    """
    #################################################
    ## Preprocessing
    #################################################    
    # Proceed with your preprocess logic here

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        # 1D y_probas (single class probabilities)
        if y_probas.ndim == 1:
            # Combine into a two-column
            y_probas = np.column_stack(
                [1 - y_probas, y_probas]
            )
    # Handle multi-class classification    
    elif len(np.unique(y_true)) > 2:
        if multi_class == 'ovr':
            # Binarize y_true for multiclass classification
            y_true = label_binarize(
                y_true,
                classes=np.unique(y_true)
            )[:, class_index]
            # Handle 1D y_probas (single class probabilities)
            if y_probas.ndim == 1:
                # Combine into a two-column binary format OvR
                y_probas = np.column_stack(
                    [1 - y_probas, y_probas]
                )
            else:
                # Combine into a two-column binary format OvR
                y_probas = y_probas[:, class_index]
                y_probas = np.column_stack(
                    [1 - y_probas, y_probas]
                )
        elif multi_class in ['multinomial', None]:
            if y_probas.ndim == 1:
                raise ValueError(
                    "For multinomial classification, `y_probas` must be 2D."
                    "For a 1D `y_probas` with more than 2 classes in `y_true`, "
                    "only 'ovr' multi-class strategy is supported."
                )
        else:
            raise ValueError("Unsupported `multi_class` strategy.")

    # Get unique classes and filter those to be plotted
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError(
            'Cannot calculate Curve for classes with only one category.'
        )
    to_plot_class_index = classes if to_plot_class_index is None else to_plot_class_index
    indices_to_plot = np.isin(classes, to_plot_class_index)
    
    # Binarize y_true for multiclass classification, for micro
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.column_stack(
            [1 - y_true_bin, y_true_bin]
        )

    #################################################
    ## Plotting
    #################################################    
    # Validate the types of ax and fig if they are provided
    if ax is not None and not isinstance(ax, mpl.axes.Axes):
        raise ValueError(
            "Provided ax must be an instance of matplotlib.axes.Axes"
        )    
    if fig is not None and not isinstance(fig, mpl.figure.Figure):
        raise ValueError(
            "Provided fig must be an instance of matplotlib.figure.Figure"
        )        
    # Neither ax nor fig is provided.
    # Create a new figure and a single subplot (ax) with the specified figsize.
    if ax is None and fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    # fig is provided but ax is not.
    # Add a single subplot (ax) to the provided figure (fig) with default positioning.
    elif ax is None:
        # 111 means a grid of 1 row, 1 column, and we want the first (and only) subplot.
        ax = fig.add_subplot(111)
    # ax is provided (whether fig is provided or not).
    # Use the provided ax for plotting. No new figure or subplot is created.
    else:
        pass        
    # Proceed with your plotting logic here

    # Initialize dictionaries to store cumulative percentages and gains
    percentages_dict, gains_dict = {}, {}

    # Loop for all classes to get different class gain
    for i, to_plot in enumerate(indices_to_plot):
        percentages_dict[i], gains_dict[i] = cumulative_gain_curve(
            y_true_bin[:, i], y_probas[:, i], #pos_label=classes[i],
        )    
        if to_plot:
            if class_names is None:
                class_names = classes            
            color = plt.get_cmap(cmap)( float(i) / len(classes) )
            ax.plot(
                percentages_dict[i], gains_dict[i],
                ls='-', lw=2, color=color,
                label='Class {}'.format(class_names[i]),
            )

    # Whether or to plot macro or micro
    if plot_micro:
        percentage, gain = cumulative_gain_curve(
            y_true_bin.ravel(), y_probas.ravel()
        )
        ax.plot(
            percentage, gain,
            ls=':', lw=3, color='deeppink',
            label='micro-average',
        )

    if plot_macro:
        # First aggregate all percentages
        all_perc = np.unique(np.concatenate(
            [ percentages_dict[i] for i in range(len(classes)) ]
        ))
        # Then interpolate all cumulative gain
        mean_gain = np.zeros_like(all_perc)
        for i in range(len(classes)):
            mean_gain += np.interp(
                all_perc, percentages_dict[i], gains_dict[i]
            )

        mean_gain /= len(classes)

        ax.plot(
            all_perc, mean_gain,
            ls=':', lw=3, color='navy',
            label='macro-average',
        )

    # Plot the baseline, label='Baseline'
    ax.plot([0, 1], [0, 1], ls='--', lw=1, c='gray')

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xlim([-0.01, 1.00])
    ax.set_ylim([-0.00, 1.05])

    # Define the desired number of ticks
    num_ticks = 10

    # Set x-axis ticks and labels
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator( (ax.get_xlim()[1] / 10) ))
    ax.xaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, min_n_ticks=9, integer=False) )
    ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    ax.yaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )

    # Enable grid and display legend
    ax.grid(True)
    if show_labels:
        ax.legend(
            loc='lower right', 
            fontsize=text_fontsize,
            title=f'Cumulative Gain Curves' + (' One-vs-Rest (OVR)' if multi_class == 'ovr' else ''),
            alignment='left'
        )

    plt.tight_layout()
    return ax


def plot_lift(
    ## default params
    y_true,
    y_probas,
    *,
    class_names=None,
    multi_class=None,
    class_index=1,
    to_plot_class_index=None,
    ## plotting params
    title='Lift Curves',
    ax=None,
    fig=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    cmap=None,
    show_labels=True,
    plot_micro=False,
    plot_macro=False,
    ## additional params
    **kwargs,
):
    """
    Generate a Lift Curve from true labels and predicted probabilities.

    The lift curve evaluates the performance of a classifier by comparing
    the lift (or improvement) achieved by using the model compared to random 
    guessing. The implementation supports binary classification directly and 
    multiclass classification through One-vs-Rest (OVR) or multinomial strategies.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_probas : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities for each class or only target class probabilities. 
        If 1D, it is treated as probabilities for the positive class in binary 
        or multiclass classification with the `class_index`.

    class_names : list of str, optional, default=None
        List of class names for the legend. Order should match the order of classes in `y_probas`.

    multi_class : {'ovr', 'multinomial', None}, optional, default=None
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, plotting binary problems for each class.
        - 'multinomial' or None: Multinomial plot for the entire probability distribution.

    class_index : int, optional, default=1
        Index of the class of interest for multi-class classification. Ignored for
        binary classification.

    to_plot_class_index : list-like, optional, default=None
        Specific classes to plot. If a given class does not exist, it will be ignored. 
        If None, all classes are plotted. e.g. [0, 'cold']

    title : str, default='Lift Curves'
        Title of the plot.

    ax : list of matplotlib.axes.Axes, optional, default=None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).
        Need two axes like [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

    fig : matplotlib.pyplot.figure, optional, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

        .. versionadded:: 0.3.9

    figsize : tuple of int, optional, default=None
        Size of the figure (width, height) in inches.

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
    
    show_labels : bool, optional, default=True
        Whether to display the legend labels.

        .. versionadded:: 0.3.9

    plot_micro : bool, optional, default=False
        Whether to plot the micro-average Lift curve.

    plot_macro : bool, optional, default=False
        Whether to plot the macro-average Lift curve.

    show_labels : bool, optional, default=True
        Whether to display the legend labels.

        .. versionadded:: 0.3.9

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted lift curves.

    Notes
    -----
    The implementation is specific to binary classification. For multiclass 
    problems, the 'ovr' or 'multinomial' strategies can be used. When 
    `multi_class='ovr'`, the plot focuses on the specified class (`class_index`).

    References
    ----------
    [1] http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Lift Curves
    
        >>> from sklearn.datasets import load_iris as data_3_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> import scikitplot as skplt
        >>> X, y = data_3_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = LogisticRegression(max_iter=int(1e5), random_state=0).fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.deciles.plot_lift(
        >>>     y_val, y_probas,
        >>> );
    """
    #################################################
    ## Preprocessing
    #################################################    
    # Proceed with your preprocess logic here

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)    

    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        # 1D y_probas (single class probabilities)
        if y_probas.ndim == 1:
            # Combine into a two-column
            y_probas = np.column_stack(
                [1 - y_probas, y_probas]
            )
    # Handle multi-class classification    
    elif len(np.unique(y_true)) > 2:
        if multi_class == 'ovr':
            # Binarize y_true for multiclass classification
            y_true = label_binarize(
                y_true,
                classes=np.unique(y_true)
            )[:, class_index]
            # Handle 1D y_probas (single class probabilities)
            if y_probas.ndim == 1:
                # Combine into a two-column binary format OvR
                y_probas = np.column_stack(
                    [1 - y_probas, y_probas]
                )
            else:
                # Combine into a two-column binary format OvR
                y_probas = y_probas[:, class_index]
                y_probas = np.column_stack(
                    [1 - y_probas, y_probas]
                )
        elif multi_class in ['multinomial', None]:
            if y_probas.ndim == 1:
                raise ValueError(
                    "For multinomial classification, `y_probas` must be 2D."
                    "For a 1D `y_probas` with more than 2 classes in `y_true`, "
                    "only 'ovr' multi-class strategy is supported."
                )
        else:
            raise ValueError("Unsupported `multi_class` strategy.")

    # Get unique classes and filter those to be plotted
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError(
            'Cannot calculate Curve for classes with only one category.'
        )
    to_plot_class_index = classes if to_plot_class_index is None else to_plot_class_index
    indices_to_plot = np.isin(classes, to_plot_class_index)
    
    # Binarize y_true for multiclass classification, for micro
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.column_stack(
            [1 - y_true_bin, y_true_bin]
        )

    #################################################
    ## Plotting
    #################################################    
    # Validate the types of ax and fig if they are provided
    if ax is not None and not isinstance(ax, mpl.axes.Axes):
        raise ValueError(
            "Provided ax must be an instance of matplotlib.axes.Axes"
        )    
    if fig is not None and not isinstance(fig, mpl.figure.Figure):
        raise ValueError(
            "Provided fig must be an instance of matplotlib.figure.Figure"
        )        
    # Neither ax nor fig is provided.
    # Create a new figure and a single subplot (ax) with the specified figsize.
    if ax is None and fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    # fig is provided but ax is not.
    # Add a single subplot (ax) to the provided figure (fig) with default positioning.
    elif ax is None:
        # 111 means a grid of 1 row, 1 column, and we want the first (and only) subplot.
        ax = fig.add_subplot(111)
    # ax is provided (whether fig is provided or not).
    # Use the provided ax for plotting. No new figure or subplot is created.
    else:
        pass        
    # Proceed with your plotting logic here

    # Initialize dictionaries to store cumulative percentages and gains
    percentages_dict, gains_dict = {}, {}

    # Loop for all classes to get different class gain
    for i, to_plot in enumerate(indices_to_plot):
        percentages_dict[i], gains_dict[i] = cumulative_gain_curve(
            y_true_bin[:, i], y_probas[:, i], #pos_label=classes[i],
        )
        percentages = percentages_dict[i][1:]
        gains = gains_dict[i][1:]    
        gains /= percentages
        
        if to_plot:
            if class_names is None:
                class_names = classes            
            color = plt.get_cmap(cmap)( float(i) / len(classes) )
            ax.plot(
                percentages, gains,
                ls='-', lw=2, color=color,
                label='Class {}'.format(class_names[i]),
            )

    # Whether or to plot macro or micro
    if plot_micro:
        percentage, gain = cumulative_gain_curve(
            y_true_bin.ravel(), y_probas.ravel()
        )
        ax.plot(
            percentage, gain,
            ls=':', lw=3, color='deeppink',
            label='micro-average',
        )

    if plot_macro:
        # First aggregate all percentages
        all_perc = np.unique(np.concatenate(
            [ percentages_dict[i] for i in range(len(classes)) ]
        ))
        # Then interpolate all cumulative gain
        mean_gain = np.zeros_like(all_perc)
        for i in range(len(classes)):
            mean_gain += np.interp(
                all_perc, percentages_dict[i], gains_dict[i]
            )

        mean_gain /= len(classes)

        ax.plot(
            all_perc, mean_gain,
            ls=':', lw=3, color='navy',
            label='macro-average',
        )

    # Plot the baseline
    ax.plot([0, 1], [1, 1], ls='--', lw=1, c='gray', label='Baseline (1)')

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel('Lift', fontsize=text_fontsize)
    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xlim([-0.0, 1.02])
    # ax.set_ylim([-0.0, 1.05])

    # Define the desired number of ticks
    num_ticks = 10

    # Set x-axis ticks and labels
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator( (ax.get_xlim()[1] / 10) ))
    ax.xaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, min_n_ticks=9, integer=False) )
    ax.xaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )
    ax.yaxis.set_major_locator( mpl.ticker.MaxNLocator(nbins=num_ticks, integer=False) )
    ax.yaxis.set_major_formatter( mpl.ticker.FormatStrFormatter('%.1f') )

    # Enable grid and display legend
    ax.grid(True)
    if show_labels:
        ax.legend(
            loc='upper right', 
            fontsize=text_fontsize,
            title=f'Lift Curves' + (' One-vs-Rest (OVR)' if multi_class == 'ovr' else ''),
            alignment='left'
        )

    plt.tight_layout()
    return ax


def plot_lift_decile_wise(
    y_true, 
    y_prob, 
    title='Decile-wise Lift Plot',
    ax=None, 
    figsize=None, 
    title_fontsize="large",
    text_fontsize="medium",
):
    """
    Generates the Decile-wise Lift Plot from labels and probabilities.

    The lift curve is used to determine the effectiveness of a binary classifier.
    A detailed explanation can be found at:
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
    The implementation here works only for binary classification.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_prob : array-like, shape (n_samples, n_classes)
        Prediction probabilities for each class returned by a classifier.

    title : str, optional, default='Decile-wise Lift Plot'
        Title of the generated plot.

    title_fontsize : str or int, optional, default=14
        Font size for the plot title. Use e.g., "small", "medium", "large" or integer-values 
        (8, 10, 12, etc.).

    text_fontsize : str or int, optional, default=10
        Font size for the text in the plot. Use e.g., "small", "medium", "large" or integer-values 
        (8, 10, 12, etc.).

    figsize : tuple of int, optional, default=None
        Tuple denoting figure size of the plot (e.g., (6, 6)).

        .. versionadded:: 0.3.9

    Returns
    -------
    None
        This function does not return any value.

    References
    ----------
    [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L190

    Examples
    --------
    
    .. plot::
       :context: close-figs
       :align: center
       :alt: Lift Decile Wise Curves
    
        >>> import scikitplot as skplt
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> clf = DecisionTreeClassifier(max_depth=1, random_state=0)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> skplt.deciles.plot_lift_decile_wise(y_test, y_prob[:, 1])
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=title_fontsize)

    # Decile-wise Lift Plot
    # plt.subplot(2, 2, 2)
    pldw = decile_table(y_true,y_prob,labels=False)
    plt.plot(pldw.decile.values, pldw.cnt_resp.values / pldw.cnt_resp_rndm.values, marker='o', label='Model')

    # plt.plot(list(np.arange(1,11)), np.ones(10), 'k--',marker='o')
    plt.plot([1, 10], [1, 1], 'k--', marker='o', label='Random')

    # Set title, labels, and formatting
    # plt.title(title, fontsize=title_fontsize)
    plt.xlabel('Deciles', fontsize=text_fontsize)
    plt.ylabel('Lift @ Decile', fontsize=text_fontsize)
    
    # Enable grid and display legend
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    return ax


def plot_ks_statistic(
    ## default params
    y_true,
    y_probas,
    *,
    # class_names=None,
    # multi_class=None,
    # class_index=1,
    # to_plot_class_index=None,
    ## plotting params
    title='KS Statistic Plot',
    ax=None,
    fig=None,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    digits=3,
    ## additional params
    **kwargs,
):
    """
    Generates the KS Statistic plot from labels and scores/probabilities.

    Parameters
    ----------
    y_true : array-like, shape (n_samples)
        Ground truth (correct) target values.

    y_probas : array-like, shape (n_samples, n_classes)
        Prediction probabilities for each class returned by a classifier.

    title : str, optional
        Title of the generated plot. Defaults to "KS Statistic Plot".

    ax : list of matplotlib.axes.Axes, optional, default=None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).
        Need two axes like [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

    fig : matplotlib.pyplot.figure, optional, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

        .. versionadded:: 0.3.9

    figsize : tuple of 2 ints, optional
        Tuple denoting figure size of the plot e.g. (6, 6). Defaults to None.

    title_fontsize : str or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or integer-values. Defaults to "large".

    text_fontsize : str or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large" or integer-values. Defaults to "medium".

    digits : int, optional
        Number of digits for formatting output floating point values. Use e.g. 2 or 4. Defaults to 3.

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
       :alt: KS Statistic Plot
    
        >>> from sklearn.datasets import load_breast_cancer as data_2_classes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> import scikitplot as skplt
        >>> X, y = data_2_classes(return_X_y=True, as_frame=False)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
        >>> model = LogisticRegression(max_iter=int(1e5), random_state=0).fit(X_train, y_train)
        >>> y_probas = model.predict_proba(X_val)
        >>> skplt.deciles.plot_ks_statistic(
        >>>     y_val, y_probas,
        >>> );
    """
    #################################################
    ## Preprocessing
    #################################################    
    # Proceed with your preprocess logic here
    
    y_true = np.asarray(y_true)
    y_probas = np.asarray(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate KS statistic for data with '
                         '{} category/ies'.format(len(classes)))
    probas = y_probas

    # Compute KS Statistic curves
    (thresholds, pct1, pct2, ks_statistic, max_distance_at,
     classes) = binary_ks_curve(
         y_true,
         probas[:, 1].ravel()
    )

    #################################################
    ## Plotting
    #################################################    
    # Validate the types of ax and fig if they are provided
    if ax is not None and not isinstance(ax, mpl.axes.Axes):
        raise ValueError(
            "Provided ax must be an instance of matplotlib.axes.Axes"
        )    
    if fig is not None and not isinstance(fig, mpl.figure.Figure):
        raise ValueError(
            "Provided fig must be an instance of matplotlib.figure.Figure"
        )        
    # Neither ax nor fig is provided.
    # Create a new figure and a single subplot (ax) with the specified figsize.
    if ax is None and fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    # fig is provided but ax is not.
    # Add a single subplot (ax) to the provided figure (fig) with default positioning.
    elif ax is None:
        # 111 means a grid of 1 row, 1 column, and we want the first (and only) subplot.
        ax = fig.add_subplot(111)
    # ax is provided (whether fig is provided or not).
    # Use the provided ax for plotting. No new figure or subplot is created.
    else:
        pass        
    # Proceed with your plotting logic here

    ax.plot(thresholds, pct1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(thresholds, pct2, lw=3, label='Class {}'.format(classes[1]))
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(max_distance_at, *sorted([pct1[idx], pct2[idx]]),
               label = 'KS Statistic: {:.{digits}f} at {:.{digits}f}'.format(
                   ks_statistic, max_distance_at, digits=digits
                ),
               linestyle = ':', lw=3, color='black')

    # Set title, labels, and formatting
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('Threshold', fontsize=text_fontsize)
    ax.set_ylabel('Percentage below threshold', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    # Display legend
    ax.legend(loc='lower right', fontsize=text_fontsize)

    plt.tight_layout()
    return ax


def report(
    y_true, y_prob, labels=True,
    ax=None, figsize=None, title_fontsize="large",
    text_fontsize="medium", digits=3,
    plot_style = None,
):
    """
    Generates a decile table and four plots (Lift, Lift@Decile, Gain, and KS) 
    from labels and probabilities.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.

    y_prob : array-like, shape (n_samples, n_classes)
        Prediction probabilities for each class returned by a classifier.

    labels : bool, optional, default=True
        If True, prints a legend for the abbreviations of decile table column names.

    ax : matplotlib.axes.Axes, optional, default=None
        The axes upon which to plot. If None, a new set of axes is created.

    figsize : tuple of int, optional, default=None
        Tuple denoting figure size of the plot (e.g., (6, 6)).

    title_fontsize : str or int, optional, default='large'
        Font size for the plot title. Use e.g., "small", "medium", "large" or integer-values.

    text_fontsize : str or int, optional, default='medium'
        Font size for the text in the plot. Use e.g., "small", "medium", "large" or integer-values.

    digits : int, optional, default=3
        Number of digits for formatting output floating point values. Use e.g., 2 or 4.

    plot_style : str, optional, default=None
        Check available styles with "plt.style.available". Examples include:
        ['ggplot', 'seaborn', 'bmh', 'classic', 'dark_background', 'fivethirtyeight', 
        'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 
        'seaborn-dark-palette', 'tableau-colorblind10', 'fast'].

        .. versionadded:: 0.3.9

    Returns
    -------
    pandas.DataFrame
        The dataframe containing the decile table with the deciles and related information.

    References
    ----------
    [1] https://github.com/tensorbored/kds/blob/master/kds/metrics.py#L382

    Examples
    --------
    >>> import scikitplot as skplt
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn import tree
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3)
    >>> clf = tree.DecisionTreeClassifier(max_depth=1, random_state=3)
    >>> clf = clf.fit(X_train, y_train)
    >>> y_prob = clf.predict_proba(X_test)
    >>> skplt.deciles.report(y_test, y_prob[:, 1])
    """
    if plot_style is None:
        None
    else:
        plt.style.use(plot_style)

    fig = plt.figure(figsize=figsize)

    # Cumulative Lift Plot
    plt.subplot(2, 2, 1)
    plot_lift(y_true,y_prob)

    #  Decile-wise Lift Plot
    plt.subplot(2, 2, 2)
    plot_lift_decile_wise(y_true,y_prob)

    # Cumulative Gains Plot
    plt.subplot(2, 2, 3)
    plot_cumulative_gain(y_true,y_prob)

    # KS Statistic Plot
    plt.subplot(2, 2, 4)
    plot_ks_statistic(y_true,y_prob)
    
    dc = decile_table(y_true, y_prob, labels=labels, digits=digits)
    plt.tight_layout()
    return (dc)