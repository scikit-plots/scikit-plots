"""
This module provides utilities for saving result images (such as plots)
and includes decorators for automatically saving plots.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

# from functools import wraps
import functools  # noqa: I001
import logging
import warnings

# import inspect

import numpy as np  # type: ignore[reportMissingModuleSource]
import matplotlib as mpl  # type: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]

from .utils_path import get_file_path

from typing import TYPE_CHECKING  # pylint: disable=wrong-import-order

if TYPE_CHECKING:
    # Only imported during type checking
    from typing import (  # noqa: F401
        Any,
        Callable,
        Dict,
        List,
        Optional,
        Union,
    )

logger = logging.getLogger(__name__)

######################################################################
## save_plot_decorator
######################################################################


# 1. Standard Decorator (no arguments) both with params and without params
# 2. Decorator with Arguments (takes args like @my_decorator(x=1)), need hint
# Called with params: @_decorator(), Called without params: @_decorator
# Hint: Then you'll get TypeError, because func is passed as the first positional arg
# to _decorator, which is not expecting a function yet.
# Hint: prefix _ or pylint: disable=unused-argument  # noqa: W0613
# Hint: from functools import partial _decorator = partial(_decorator, verbose=True)
def save_plot_decorator(
    # Not needed as a placeholder, but kept for parameterized usage
    # *_dargs,  # not need placeholder
    # The target function to be decorated (passed when no parameters are used)
    func: "Optional[Callable[..., Any]]" = None,
    # *,  # indicates that all following parameters must be passed as keyword
    **dkwargs: dict,  # Keyword arguments passed to the decorator for customization (e.g., verbose)
) -> "Callable[..., Any]":
    """
    A generic decorator that supports both parameterized and non-parameterized usage.

    This decorator can be used directly (`@decorator`) or
    with parameters (`@decorator(param=value)`).
    It wraps the target function, optionally modifying its behavior based on
    decorator-specific arguments.

    This supports both:
    - @decorator
    - @decorator(verbose=True)

    Parameters
    ----------
    *_dargs : tuple
        Positional arguments passed to the decorator (ignored by default).
    func : Callable, optional
        The target function to be decorated. This is automatically set when the decorator
        is used without parentheses (e.g., `@decorator`).
    **dkwargs : dict
        Keyword arguments passed to the decorator for configuration. These can be used
        to customize the behavior of the wrapper.

    Returns
    -------
    Callable
        The decorated function with additional behavior defined by the decorator.

    Examples
    --------
    >>> @_decorator
    ... def greet():
    ...     print("Hello")

    >>> @_decorator(verbose=True)
    ... def greet():
    ...     print("Hello")

    Notes
    -----
    - This decorator can be used both with and without parameters.
    - The `func` argument must be placed after `*_dargs` to support keyword-only usage and
      to avoid `W1113` (keyword-before-vararg) linter warnings.
    - This structure enables reusability across decorators with shared patterns.
    """

    # The case where the decorator is called with parameters (returns a decorator)
    def decorator(inner_func: "Callable") -> "Callable":
        """
        The actual decorator function that wraps the target function.

        Parameters
        ----------
        inner_func : Callable
            The function to be decorated.

        **dkwargs : dict
            Keyword arguments passed to the decorator for customization.

        Returns
        -------
        Callable
            The wrapped function.
        """

        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs) -> "Any":
            result = inner_func(*args, **kwargs)
            try:
                # c = a | b  # Non-destructive merge (3.9+)
                # c = {**a, **b}  # Non-destructive merge (3.5+), Safe, non-mutating
                # a.update(b)  # In-place update (All Versions), Copy Before Update
                # Save the plot if save_fig is True
                # dkwargs = dkwargs.copy(); dkwargs.update(kwargs)
                # Get dynamic saving parameters from the function arguments
                show_fig = kwargs.get("show_fig", True)
                save_fig = kwargs.get("save_fig", False)
                # Automatically get the name of the calling script using inspect.stack()
                # caller_filename = inspect.stack()[1].filename
                save_fig_filename = (
                    kwargs.get("save_fig_filename", dkwargs.get("filename"))
                    or inner_func.__name__
                )
                dkwargs["filename"] = save_fig_filename
                # print(f"[INFO]:\n\t{kwargs}\n\t{dkwargs}\n\t{save_fig}\n\t{save_fig_filename}\n")
                # Handle verbosity if specified
                if "verbose" in kwargs and not isinstance(kwargs["verbose"], bool):
                    warnings.warn(
                        "'verbose' parameter should be of type bool.",
                        stacklevel=1,
                    )
                if save_fig and save_fig_filename:
                    save_path = get_file_path(
                        **{**dkwargs, **kwargs},  # Update by inner func
                    )
                    plt.tight_layout()
                    # plt.draw()
                    # plt.pause(0.1)
                    try:
                        plt.savefig(
                            save_path, dpi=150, bbox_inches="tight", pad_inches=0
                        )
                        if kwargs.get("verbose", False):
                            print(f"[INFO] Plot saved to: {save_path}")  # noqa: T201
                    except Exception as e:
                        print(f"[ERROR] Failed to save plot: {e}")  # noqa: T201
                if show_fig:
                    # Manage the plot window
                    plt.show()
                    # plt.gcf().clear()
                    # plt.close()
            except Exception:
                pass
            return result

        return wrapper

    # Check if `func` was passed directly (i.e., decorator without parameters)
    if func is not None and callable(func):
        return decorator(func)

    return decorator


######################################################################
## stacking matplotlib figures
######################################################################


@save_plot_decorator
def stack_mpl_figures(
    *figs: tuple,
    orient: str = "vertical",
    padding: float | None = None,
    **kwargs,
):
    """
    Stack multiple matplotlib figures into a single combined figure.
    Save it (if specified).

    Parameters
    ----------
    figs : tuple of matplotlib.figure.Figure
        Figures to be combined.
    orient : {'vertical', 'horizontal', 'v', 'h', 'x', 'y'}, default='vertical'
        Direction to stack the figures.
    padding : float, optional
        Space between stacked figures, in inches.
    **kwargs : dict
        Additional keyword arguments passed to plt.subplots() (e.g., figsize, dpi).

    Returns
    -------
    matplotlib.figure.Figure
        A new combined figure containing all input figures.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.plot([1, 2, 3], [4, 5, 6])
    >>> ax1.set_title("Figure 1")
    >>> fig2, ax2 = plt.subplots()
    >>> ax2.bar(["A", "B", "C"], [3, 7, 2])
    >>> ax2.set_title("Figure 2")
    >>> import scikitplot as sp
    >>> fig = sp.stack_mpl_figures(fig1, fig2)

    """
    if not figs:
        raise ValueError("At least one figure must be provided.")
    orient = orient.lower()
    if orient in ["horizontal", "h", "x"]:
        nrows, ncols = 1, len(figs)
    elif orient in ["vertical", "v", "y"]:
        nrows, ncols = len(figs), 1
    else:
        raise ValueError(
            f"Unsupported orient '{orient}'. Use 'vertical' or 'horizontal' (or v/h/x/y)."
        )

    figsize = kwargs.get("figsize")
    dpi = kwargs.get("dpi", 100)

    if figsize is None:
        # Set a basic figsize depending on orientation
        if orient in ["horizontal", "h", "x"]:
            figsize = (3.15 * len(figs), 12)  # (5 * len(figs), 4)
        else:
            figsize = (12, 3.15 * len(figs))  # (5, 4 * len(figs))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

    # If only one figure, ax will not be a list, so we make it a list
    # ax = [ax] if n_figs == 1 else ax

    # Make axes iterable
    axes = np.atleast_1d(axes).ravel()

    for ax, fig_ in zip(axes, figs):
        canvas = mpl.backends.backend_agg.FigureCanvasAgg(fig_)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())
        ax.imshow(image)
        ax.axis("off")

    plt.tight_layout(pad=padding if padding is not None else 0.5)
    return fig


######################################################################
##
######################################################################
