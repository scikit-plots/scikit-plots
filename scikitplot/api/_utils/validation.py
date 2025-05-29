"""
Utility Functions for Validation

This module provides utility functions designed to validate inputs and
parameters within the scikit-plots library. These functions assist in ensuring
that inputs conform to expected formats and constraints, and support various
data handling tasks. The utilities in this module are essential for robust
data validation and manipulation, enhancing the reliability and usability
of the library.

Functions and classes provided include (Development):

- Validation and type-checking utilities
- Functions for handling numpy arrays and NaN values
- Decorators for managing deprecated or positional arguments
- Utilities for inspecting function signatures

This module is part of the scikit-plots library and is intended for internal use
to facilitate the validation and processing of inputs.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# code that needs to be compatible with both Python 2 and Python 3

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

# import inspect
import functools
import importlib
import logging
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING

import matplotlib as mpl  # type: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
import numpy as np  # type: ignore[reportMissingModuleSource]
from sklearn.preprocessing import label_binarize  # type: ignore[reportMissingModuleSource]

from ...utils.utils_params import (
    _get_args_kwargs,
    _get_param_w_index,
    _resolve_args_and_kwargs,
)
from ..._docstrings import _docstring

if TYPE_CHECKING:
    from typing import (  # noqa: F401
        Any,
        Callable,
        List,
        Optional,
        Type,
        Union,
    )

    # F = TypeVar("F", bound=Callable[..., Any])

# Configure logging as needed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "validate_plotting_decorator",
    "validate_plotting_kwargs",
    "validate_plotting_kwargs_decorator",
    "validate_shapes",
    "validate_shapes_decorator",
    "validate_y_true",
    "validate_y_true_decorator",
    "validate_y_probas",
    "validate_y_probas_decorator",
    "validate_y_probas_bounds",
    "validate_y_probas_bounds_decorator",
    "validate_inputs",
]

######################################################################
## _get_style_context
######################################################################


def _get_style_context(plot_style=None):
    plot_style = None if isinstance(plot_style, int) else plot_style
    if plot_style is None:
        return nullcontext()
    if not isinstance(plot_style, (str, list, tuple)):
        raise TypeError(
            f"`plot_style` must be a str or list of styles, not {type(plot_style).__name__}"
        )
    return plt.style.context(plot_style)


######################################################################
## validate_plotting_decorator
######################################################################


# The decorator function
# Applying the decorator using @ syntax
def validate_plotting_decorator(func):
    """
    A decorator to validate if required plotting libraries are installed.

    This decorator checks for the presence of the `matplotlib.pyplot` module
    before executing the decorated function. If the module is not found,
    an ImportError is raised.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function that checks for library availability.

    Raises
    ------
    ImportError
        If `matplotlib` is not installed.
    """

    # The wrapper function (adds behavior around `func`)
    # Preserves the original function's metadata
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if matplotlib is installed
        if importlib.util.find_spec("matplotlib.pyplot") is None:
            raise ImportError("Matplotlib is required for plotting.")
        # Continue with the original function
        return func(*args, **kwargs)

    return wrapper


######################################################################
## validate_plotting_kwargs_decorator
######################################################################

# The docstrings here must be generic enough to apply to all relevant methods.
_docstring.interpd.register(
    _validate_plotting_kwargs_doc="""\
ax : matplotlib.axes.Axes, optional, default=None
    The axis to plot the figure on. If None is passed in the current axes
    will be used (or generated if required).

    .. versionadded:: 0.4.0
fig : matplotlib.pyplot.figure, optional, default: None
    The figure to plot the Visualizer on. If None is passed in the current
    plot will be used (or generated if required).

    .. versionadded:: 0.4.0
figsize : tuple, optional, default=None
    Width, height in inches.
    Tuple denoting figure size of the plot e.g. (12, 5)

    .. versionadded:: 0.4.0
nrows : int, optional, default=1
    Number of rows in the subplot grid.

    .. versionadded:: 0.4.0
ncols : int, optional, default=1
    Number of columns in the subplot grid.

    .. versionadded:: 0.4.0
plot_style : str, optional, default=None
    Check available styles with "plt.style.available". Examples include:
    ['ggplot', 'seaborn', 'bmh', 'classic', 'dark_background', 'fivethirtyeight',
    'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark',
    'seaborn-dark-palette', 'tableau-colorblind10', 'fast'].

    .. versionadded:: 0.4.0\
""".rstrip()
)
# index : int or tuple, optional, default=1
#     The position of the subplot on the grid. It can be:
#     - An integer specifying the position (1-based).


@validate_plotting_decorator
@_docstring.interpd
def validate_plotting_kwargs(
    *args: tuple,
    **kwargs: dict,
):
    """
    Validate the provided axes and figure or create new ones if needed.

    This function checks if valid axes and figure objects are provided. If not, it creates
    new ones based on the specified parameters.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    %(_validate_plotting_kwargs_doc)s

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure to be used for plotting.

    ax : matplotlib.axes.Axes or list of matplotlib.axes.Axes
        The axes to be used for plotting.
        Returns a single Axes object if only one subplot is created,
        or a list of Axes objects if multiple subplots are created.

    Notes
    -----
    The subplot position can be described by either:

      - Three integers (nrows, ncols, index)
      - The subplot will take the index position on a grid with nrows rows and ncols columns.
        index starts at 1 in the upper left corner and increases to the right.

    Examples
    --------
    Create a new figure and axes:

    >>> fig, ax = validate_plotting_kwargs()

    Use an existing axes:

    >>> fig, ax = plt.subplots()
    >>> fig, ax = validate_plotting_kwargs(ax=ax)
    """
    # Proceed with your plotting logic here, e.g.:
    plot_style = kwargs.get("plot_style", 1)
    ax = kwargs.get("ax")
    fig = kwargs.get("fig")
    figsize = kwargs.get("figsize")
    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", 1)
    # Validate the types of ax and fig if they are provided
    if ax is not None:
        # Flatten ax to ensure consistent shape
        _ax = np.ravel([ax]).tolist()
        # Check if ax is a sequence (list or tuple), but not a single Axes object
        if isinstance(_ax, (list, tuple, np.ndarray, Sequence)):
            if not all(isinstance(a, mpl.axes.Axes) for a in _ax):
                raise ValueError(
                    "Each item in ax must be an instance of matplotlib.axes.Axes"
                )
        # elif not isinstance(ax, mpl.axes.Axes):
        #     raise ValueError(
        #         "Provided ax must be an instance of matplotlib.axes.Axes"
        #     )
    # Validate `fig` if provided
    if fig is not None and not isinstance(fig, mpl.figure.Figure):
        raise ValueError("Provided fig must be an instance of matplotlib.figure.Figure")
    # Create a new figure and axes if neither ax nor fig is provided
    if ax is None and fig is None:
        # plt.style.use('seaborn-darkgrid')  # or 'ggplot', 'fivethirtyeight', etc.
        # _ = None if plot_style is None else plt.style.use(plot_style)
        # For temporary styling, use a context manager:
        with _get_style_context(plot_style):
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        return fig, ax  # Return immediately for new figure/axes

    # If fig is provided but ax is not, create new subplots in the existing figure
    if ax is None and fig is not None:
        ax = []  # Initialize list to hold axes
        # fig override subplot define (nrows, ncols, index)
        # int, (int, int, index), or SubplotSpec, default: (1, 1, 1)
        n_subplots = nrows * ncols
        for idx in range(1, n_subplots + 1):
            # plt.style.use('seaborn-darkgrid')  # or 'ggplot', 'fivethirtyeight', etc.
            # _ = None if plot_style is None else plt.style.use(plot_style)
            # For temporary styling, use a context manager:
            with _get_style_context(plot_style):
                # Each subplot is placed in a grid defined by nrows x ncols at position idx
                ax.append(fig.add_subplot(nrows, ncols, idx))
        # Return the figure and axes (single axis if only one)
        return fig, ax[0] if len(ax) == 1 else ax

    # Use the provided ax for plotting if it is provided (whether fig is provided or not)
    # Use the provided ax and its figure for plotting. plt.gcf()
    fig = (
        ax[0].figure
        if isinstance(ax, (list, tuple, np.ndarray, Sequence)) == 1
        else ax.figure
    )
    return fig, ax


# Define the decorator to validate plotting arguments
def validate_plotting_kwargs_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        (_args, _kwargs) = _resolve_args_and_kwargs(*args, func=func, **kwargs)
        # Call the validation function to ensure proper fig and ax are set
        fig, ax = validate_plotting_kwargs(
            **_kwargs,
        )
        # Update kwargs to pass the validated fig and ax
        kwargs["fig"] = fig
        kwargs["ax"] = ax
        # Call the actual plotting function
        return func(*args, **kwargs)

    return wrapper


######################################################################
## validate_shapes_decorator
######################################################################


def validate_shapes(y_true, y_probas):
    """
    Validate the shapes of y_true and y_probas.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    y_true : array-like
        True labels for the samples. Can be 1D (binary or multi-class) or 2D (one-hot encoded).
    y_probas : array-like
        Predicted probabilities. Can be 1D (binary) or 2D (multi-class).

    Raises
    ------
    ValueError
        If shapes of y_true and y_probas do not match or are not valid.
    """
    y_true = np.asarray(y_true)
    y_probas = np.asarray(y_probas)

    # Check for binary classification
    if y_true.ndim == 1:
        if y_probas.ndim == 1:  # Binary case (single class probability)
            if len(y_true) != len(y_probas):
                raise ValueError(
                    f"Shape mismatch `y_true` length {len(y_true)}, "
                    f"`y_probas` length {len(y_probas)}"
                )
        elif y_probas.ndim == 2:  # Binary case (two class probabilities)
            if len(y_true) != y_probas.shape[0]:
                raise ValueError(
                    f"Shape mismatch `y_true` length {len(y_true)}, "
                    f"`y_probas` shape {y_probas.shape}"
                )
        else:
            raise ValueError("Invalid shape for `y_probas`.")

    # Check for multi-class classification
    elif y_true.ndim == 2:
        if y_probas.ndim != 2:
            raise ValueError(
                "`y_probas` must be a 2D array for multi-class classification."
            )

        if len(y_true) != y_probas.shape[0]:
            raise ValueError(
                f"Shape mismatch `y_true` length {len(y_true)}, "
                f"`y_probas` shape {y_probas.shape}"
            )

        # Check number of classes for one-hot encoding
        if y_true.shape[1] != y_probas.shape[1]:
            raise ValueError(
                f"Number of classes in `y_true` ({y_true.shape[1]}) does not match "
                f"`y_probas` number of classes ({y_probas.shape[1]})"
            )

        # if np.unique(y_true).size != y_probas.shape[1]:
        #     raise ValueError(
        #         f'Number of classes in `y_true` ({np.unique(y_true).size}) does not match '
        #         f'`y_probas` number of classes ({y_probas.shape[1]})'
        #     )

    else:
        raise ValueError("Invalid shape for `y_true`.")


# Define the decorator to validate shapes
def validate_shapes_decorator(func):
    """
    Decorator to validate the shapes of `y_true` (or `y`) and `y_probas`
    before passing them to the decorated function.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    func : callable
        The function to decorate.

    Returns
    -------
    callable
        The wrapped function with validated shapes.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find and validate y_true and y_probas
        # Ensure you explicitly specify params before unpacking
        _, _, y_true = _get_param_w_index(
            func=func,
            params=["y_true", "y"],  # No need to pass 'func'
            *args,  # Unpack args here
            **kwargs,  # Unpack kwargs
        )
        _, _, y_probas = _get_param_w_index(
            func=func,
            params=["y_probas"],  # Specify params after func
            *args,  # Unpack args here
            **kwargs,  # Unpack kwargs
        )

        # Validate only if both parameters are found
        if y_true is None or y_probas is None:
            raise ValueError("Both 'y_true' and 'y_probas' must be provided.")

        # Perform shape validation
        validate_shapes(y_true, y_probas)

        # Call the original function if validation passes
        return func(*args, **kwargs)

    return wrapper


######################################################################
## validate_y_true_decorator
######################################################################


def validate_y_true(y_true, pos_label=None, class_index=None):
    """
    Validate and process y_true.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    y_true : array-like
        True labels. Can be string, numeric, or mixed types.
    pos_label : scalar, optional
        The positive label for binary classification. If None, it defaults to
        `classes[1]`.
    class_index : int, optional
        Index of the class for which to extract as positive label in multi-class case.
        If None, returns all class labels in the 2D case. Ignored if y_probas is 1D.

    Returns
    -------
    numpy.ndarray
        Processed y_true, either as boolean or binarized for multi-class.

    Raises
    ------
    ValueError
        If y_true does not meet the expected criteria.

    Examples
    --------
    >>> validate_y_true([0, 1, 1, 0], pos_label=0)
    array([ True, False, False,  True])

    >>> validate_y_true(['class_0', 'class_1', 'class_1', 'class_0'], None)
    array([False,  True,  True, False])
    """
    # Check if y_true is iterable
    if not hasattr(y_true, "__iter__") or isinstance(y_true, (str, bytes)):
        raise ValueError(
            "`y_true` must be of type bool, str, numeric, or a mix (object) type."
        )

    y_true = np.asarray(y_true)

    # Ensure y_true can handle string, numeric, or mixed types
    if not (
        isinstance(y_true, (list, np.ndarray))
        and (
            np.issubdtype(y_true.dtype, np.object_)
            or np.issubdtype(y_true.dtype, np.number)
            or np.issubdtype(y_true.dtype, np.str_)
            or np.issubdtype(y_true.dtype, np.bool_)
        )
    ):
        raise ValueError(
            "`y_true` must be of type bool, str, numeric, or a mix (object) type."
        )

    # Identify unique classes in y_true
    classes = np.unique(y_true)

    # Handle binary classification, return 2D selected classes
    if len(classes) == 2:
        if pos_label is None:
            pos_label = classes[1]  # Default to the second class if pos_label is None

        if pos_label not in classes:
            raise ValueError(
                f"`pos_label` must be one of label classes: {list(classes)}"
            )

        y_true = y_true == pos_label
        return y_true

    # Handle multi-class classification, return 2D
    if len(classes) > 2:
        y_true = label_binarize(y_true, classes=classes)
        if class_index is None:
            # Return all columns if class_index is None
            # may use None instead of np.newaxis. These are the same objects:
            return y_true[:, slice(None)]

        if (
            class_index < 0 or class_index >= y_true.shape[1]
        ):  # Make sure the index is within bounds
            raise ValueError(
                f"class_index {class_index} out of bounds for `y_true`. "
                f"It must be between 0 and {y_true.shape[1] - 1}."
            )
        return y_true[:, class_index]

    raise ValueError("`y_true` must contain more than one distinct class.")


def validate_y_true_decorator(func):
    """
    Decorator to validate the `y_true` parameter before executing the function.

    This decorator will call `validate_y_true` to ensure that the
    `y_true` input meets the required criteria.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    func : callable
        The function to decorate.

    Returns
    -------
    callable
        The wrapped function with validated `y_true`.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find and validate y_true
        y_true_name, y_true_index, y_true = _get_param_w_index(
            func=func,  # Keep this first to avoid ambiguity
            params=["y_true", "y"],  # Specify params after func
            *args,  # Unpack args here
            **kwargs,  # Unpack kwargs
        )

        # Extract pos_label and class_index from kwargs (if provided)
        _, _, pos_label = _get_param_w_index(
            func=func, params=["pos_label"], *args, **kwargs
        )
        _, _, class_index = _get_param_w_index(
            func=func, params=["class_index"], *args, **kwargs
        )

        # Call the validate_y_true function to validate y_true
        validated_y_true = validate_y_true(y_true, pos_label, class_index)

        # Ensure validated_y_true is passed correctly
        if validated_y_true is not None:
            new_args, new_kwargs = _get_args_kwargs(
                param_key=y_true_name,
                param_index=y_true_index,
                param_value=validated_y_true,
                *args,
                **kwargs,
            )
        else:
            new_args, new_kwargs = args, kwargs

        # Call the original function with updated args and kwargs
        return func(*new_args, **new_kwargs)

    return wrapper


######################################################################
## validate_y_probas_decorator
######################################################################


def validate_y_probas(y_probas, class_index=None):
    """
    Validate the `y_probas` parameter.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    y_probas : array-like
        Predicted probabilities, must be 1D or 2D.
    class_index : int, optional
        Index of the class for which to extract probabilities in multi-class case.
        If None, returns all class probabilities in the 2D case. Ignored if y_probas is 1D.

    Returns
    -------
    numpy.ndarray
        The validated `y_probas` values.
        Processed probabilities for the specified class, or all classes if class_index is None.

    Raises
    ------
    ValueError
        If `y_probas` is not an array of numerical values.
        If y_probas does not meet the expected criteria.

    Examples
    --------
    >>> validate_y_probas([[0.8, 0.2], [0.2, 0.8]], class_index=1)
    array([0.2, 0.8])

    >>> validate_y_probas([0.9, 0.1])
    array([0.9, 0.1])

    >>> validate_y_probas([[0.6, 0.4], [0.3, 0.7]], class_index=None)
    array([[0.6, 0.4],
           [0.3, 0.7]])
    """
    # Check if y_probas is iterable
    if not hasattr(y_probas, "__iter__"):
        raise ValueError("`y_probas` must be of type float type.")

    y_probas = np.asarray(y_probas)

    if not np.issubdtype(y_probas.dtype, np.number):
        raise ValueError("`y_probas` must be an array of numerical values.")

    # Check y_probas shape
    if y_probas.ndim not in [1, 2]:  # Now checks for 0D or 3D
        raise ValueError("`y_probas` must be either a 1D or 2D array.")

    # In 1D case, ignore class_index and return y_probas as is
    if y_probas.ndim == 1:
        return y_probas

    # Handle multi-class probabilities
    if y_probas.ndim == 2:
        if class_index is None:
            # Return all columns if class_index is None
            # may use None instead of np.newaxis. These are the same objects:
            return y_probas[:, slice(None)]

        if (
            class_index < 0 or class_index >= y_probas.shape[1]
        ):  # Make sure the index is within bounds
            raise ValueError(
                f"class_index {class_index} out of bounds for `y_probas`. "
                f"It must be between 0 and {y_probas.shape[1] - 1}."
            )

        return y_probas[:, class_index]  # Return specified class probabilities


def validate_y_probas_decorator(func):
    """
    Decorator to validate the `y_probas` parameter before executing the function.

    This decorator will call `validate_y_probas` to ensure that the
    `y_probas` input meets the required criteria.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    func : callable
        The function to decorate.

    Returns
    -------
    callable
        The wrapped function with validated `y_probas`.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find and validate y_probas
        y_probas_name, y_probas_index, y_probas = _get_param_w_index(
            func=func,  # Keep this first to avoid ambiguity
            params=["y_probas"],  # Specify params after func
            *args,  # Unpack args here
            **kwargs,  # Unpack kwargs
        )

        # Extract class_index from kwargs (default to 0 if not provided)
        _, _, class_index = _get_param_w_index(
            func=func, params=["class_index"], *args, **kwargs
        )

        # Call the validate_y_probas function to validate y_probas
        validated_y_probas = validate_y_probas(y_probas, class_index)

        # Ensure validated_y_true is passed correctly
        if validated_y_probas is not None:
            new_args, new_kwargs = _get_args_kwargs(
                param_key=y_probas_name,
                param_index=y_probas_index,
                param_value=validated_y_probas,
                *args,
                **kwargs,
            )
        else:
            new_args, new_kwargs = args, kwargs

        # Call the original function with updated args and kwargs
        return func(*new_args, **new_kwargs)

    return wrapper


######################################################################
## validate_y_probas_bounds_decorator
######################################################################


def _range01(x):
    """
    Normalizing input

    Parameters
    ----------
    x : list of numeric data
        List of numeric data to get normalized

    Returns
    -------
    normalized version of x
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def validate_y_probas_bounds(y_probas, method="minmax", axis=0):
    """
    Apply min-max scaling (normalization) to transform numerical values into a range,
    typically between 0 and 1. This method ensures the input values (probabilities or
    other numerical data) are valid, applying scaling if necessary.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    y_probas : array-like
        Input numerical data or predicted probabilities to be scaled. This can be a 1D
        or 2D array.
    method : {'minmax', 'sigmoid'}, default='minmax'
        The scaling method to apply. 'minmax' for min-max scaling and 'sigmoid'
        for logistic sigmoid scaling.
    axis : int, optional, default=0
        Axis along which to compute the scaling for min-max method. Default is 0,
        which applies scaling column-wise. Ignored for sigmoid method.

    Returns
    -------
    numpy.ndarray
        Scaled values where each element falls within the [0, 1] range.

    Raises
    ------
    ValueError
        If y_probas contains invalid data types or dimensions, or if an unknown method is specified.

    Examples
    --------
    >>> min_max_scaling(np.array([-0.5, 0.2, 1.5]))
    array([0.  , 0.35, 1.  ])

    >>> min_max_scaling(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    array([[0.1, 0.2],
           [0.3, 0.4],
           [0.5, 0.6]])

    >>> min_max_scaling(np.array([[-5, 0], [10, 15]]))
    array([[0.  , 0.25],
           [0.75, 1.  ]])
    """

    def is_continuous(y_probas):
        # Ensure y_probas is continuous and not binary
        if np.array_equal(np.unique(y_probas), [0, 1]):
            raise ValueError(
                "`y_probas` should contain continuous values, not binary (0/1) scores."
            )
        return y_probas

    y_probas = np.asarray(y_probas)

    # Check if input is numerical
    if not np.issubdtype(y_probas.dtype, np.number):
        raise ValueError("`y_probas` must be an array of numerical values.")

    if y_probas.ndim == 1:  # Binary case
        if np.any((y_probas < 0) | (y_probas > 1)):
            if method == "minmax":
                y_probas = np.clip(
                    (y_probas - y_probas.min()) / (y_probas.max() - y_probas.min()),
                    0,
                    1,
                )
                return is_continuous(y_probas)
            if method == "sigmoid":
                y_probas = 1 / (1 + np.exp(-y_probas))
                return is_continuous(y_probas)

    elif y_probas.ndim == 2:  # Multi-class case
        if np.any(y_probas < 0) or np.any(y_probas > 1):
            if method == "minmax":
                min_val = y_probas.min(axis=axis, keepdims=True)
                max_val = y_probas.max(axis=axis, keepdims=True)
                scaled = (y_probas - min_val) / (max_val - min_val)
                y_probas = np.clip(scaled, 0, 1)
                return is_continuous(y_probas)
            if method == "sigmoid":
                y_probas = 1 / (
                    1 + np.exp(-y_probas)
                )  # Applies sigmoid to each element
                return is_continuous(y_probas)

    return is_continuous(y_probas)  # Return unmodified if within valid range


def validate_y_probas_bounds_decorator(func):
    """
    Decorator to validate and apply bounds validation on `y_probas`
    before passing it to the decorated function.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    func : callable
        The function to decorate.

    Returns
    -------
    callable
        The wrapped function with validated and scaled `y_probas`.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find and validate y_probas
        y_probas_name, y_probas_index, y_probas = _get_param_w_index(
            func=func,  # Keep this first to avoid ambiguity
            params=["y_probas"],  # Specify params after func
            *args,  # Unpack args here
            **kwargs,  # Unpack kwargs
        )

        # Retrieve method and axis from kwargs
        _, _, method = _get_param_w_index(func=func, params=["method"], *args, **kwargs)
        _, _, axis = _get_param_w_index(func=func, params=["axis"], *args, **kwargs)

        # Validate y_probas and apply bounds scaling
        validated_y_probas = validate_y_probas_bounds(
            y_probas, method=method, axis=axis
        )

        # Create new args and kwargs with validated y_probas
        new_args, new_kwargs = _get_args_kwargs(
            param_key=y_probas_name,
            param_index=y_probas_index,
            param_value=validated_y_probas,
            *args,
            **kwargs,
        )

        # Call the original function if validation passes
        return func(*new_args, **new_kwargs)

    return wrapper


######################################################################
## validate_inputs
######################################################################


def validate_inputs(
    y_true, y_probas, pos_label=None, class_index=None, method="minmax", axis=0
):
    """
    Validate the inputs for y_true and y_probas, and apply bounds validation if necessary.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    y_true : array-like
        The ground truth labels to validate.
    y_probas : array-like
        The predicted probabilities to validate.
    pos_label : optional
        The positive label for binary classification.
    class_index : optional
        The index of the class to consider for certain validations.
    method : str, optional
        The method to use for scaling y_probas, default is 'minmax'.
    axis : int, optional
        The axis along which to perform the scaling, default is 0.

    Returns
    -------
    tuple
        A tuple containing the validated y_true and y_probas.

    Raises
    ------
    ValueError
        If any of the validation checks fail.
    """
    # Validate shapes
    validate_shapes(y_true, y_probas)

    # Validate y_true
    y_true = validate_y_true(y_true, pos_label, class_index)

    # Validate y_probas
    y_probas = validate_y_probas(y_probas, class_index)

    # Validate bounds on y_probas
    y_probas = validate_y_probas_bounds(y_probas, method=method, axis=axis)

    return y_true, y_probas


######################################################################
##
######################################################################
