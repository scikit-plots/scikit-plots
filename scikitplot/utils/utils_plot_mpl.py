"""
This module provides utilities for saving result images (such as plots)
and includes decorators for automatically saving plots.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

import functools
import logging
import warnings

# from functools import wraps
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]

from .utils_path import get_file_path

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
    # *,  # Expected one or more keyword parameter after '*'
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
                save_fig = kwargs.get("save_fig", False)
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
##
######################################################################
