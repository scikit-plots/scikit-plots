"""
Provides utilities for saving result images.

Such as plots and includes decorators for automatically saving plots.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

# import inspect
# import tempfile
# from functools import wraps
import contextlib as _contextlib
import functools as _functools
import warnings as _warnings
from typing import TYPE_CHECKING

import matplotlib as _mpl  # noqa: ICN001
import matplotlib.pyplot as _plt  # noqa: ICN001
import numpy as _np  # noqa: ICN001

from .. import logger as _logger
from .._docstrings import _docstring
from .utils_path import get_file_path

if TYPE_CHECKING:
    # Only imported during type checking
    from typing import (
        Callable,
        Optional,
        # Union,
    )

# import logging
# Set up logging
# logging.basicConfig(level=logging.INFO)
# _logger = logging.getLogger(__name__)


######################################################################
## safe tight_layout
######################################################################


@_contextlib.contextmanager
def safe_tight_layout(fig=None, *, warn=True):
    """
    Apply `tight_layout()` safely to a Matplotlib figure after the context block.

    This context manager ensures that `tight_layout()` is called after rendering
    code inside the `with` block. If `tight_layout()` raises an exception, a warning
    is logged (unless `warn` is False).

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        The figure to apply `tight_layout()` to. If None, uses the current figure.
    warn : bool, default=True
        Whether to log a warning if `tight_layout()` fails.

    Yields
    ------
    matplotlib.figure.Figure
        The figure object being managed.
    None
        Allows code to run inside the context block.

    Notes
    -----
    A function-based version using @contextmanager

    This is useful when layout issues might occur due to dynamic content (e.g., variable
    label sizes or interactive plots). It avoids hard crashes caused by layout failures
    and logs helpful diagnostics.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from your_module import safe_tight_layout
    >>> fig, ax = plt.subplots()
    >>> with safe_tight_layout(fig) as f:
    ...     # print(f)  # prints None, "anything"
    ...     ax.plot([1, 2, 3], [4, 5, 6])
    ...     ax.set_title("Safe Layout")
    """
    try:
        # yield  # None
        # yield "anything"
        yield fig
    finally:
        try:
            fig = fig or _plt.gcf()
            # import warnings
            # warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
            fig.tight_layout()
        except Exception as e:
            if warn:
                _logger.warning("tight_layout() failed: %s", e)


class SafeTightLayout:
    """
    Context manager to safely apply `tight_layout()` to a Matplotlib figure.

    Applies `tight_layout()` when exiting the context block. If the layout
    adjustment fails, a warning is logged unless `warn=False`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        The figure on which to apply `tight_layout()`. If None, the current figure is used.
    warn : bool, default=True
        Whether to log a warning if `tight_layout()` fails.

    Yields
    ------
    matplotlib.figure.Figure
        The figure object being managed.
    None
        Allows code to run inside the context block.

    Notes
    -----
    A class-based version using SafeTightLayout

    This is useful when layout issues might occur due to dynamic content (e.g., variable
    label sizes or interactive plots). It avoids hard crashes caused by layout failures
    and logs helpful diagnostics.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from your_module import SafeTightLayout
    >>> fig, ax = plt.subplots()
    >>> with SafeTightLayout(fig) as fig:
    ...     ax.plot([0, 1], [1, 0])
    ...     ax.set_xlabel("X-axis")
    ...     ax.set_ylabel("Y-axis")
    """

    def __init__(self, fig=None, *, warn=True):
        self.fig = fig or _plt.gcf()
        self.warn = warn

    def __enter__(self):
        """__enter__."""
        # return self
        return self.fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        """__exit__."""
        try:
            self.fig.tight_layout()
        except Exception as e:
            if self.warn:
                _logger.warning("tight_layout() failed: %s", e)


######################################################################
## save_plot_decorator
######################################################################

# The docstrings here must be generic enough to apply to all relevant methods.
_docstring.interpd.register(
    _save_plot_decorator_kwargs_doc="""\
show_fig : bool, default=True
    Show the plot.

    .. versionadded:: 0.4.0
save_fig : bool, default=False
    Save the plot.

    .. versionadded:: 0.4.0
save_fig_filename : str, optional, default=''
    Specify the path and filetype to save the plot.
    If nothing specified, the plot will be saved as png
    inside ``result_images`` under to the current working directory.
    Defaults to plot image named to used ``func.__name__``.

    .. versionadded:: 0.4.0
overwrite : bool, optional, default=True
    If False and a file exists, auto-increments the filename to avoid overwriting.

    .. versionadded:: 0.4.0
add_timestamp : bool, optional, default=False
    Whether to append a timestamp to the filename.
    Default is False.

    .. versionadded:: 0.4.0
verbose : bool, optional
    If True, enables verbose output with informative messages during execution.
    Useful for debugging or understanding internal operations such as backend selection,
    font loading, and file saving status. If False, runs silently unless errors occur.

    Default is False.

    .. versionadded:: 0.4.0
        The `verbose` parameter was added to control logging and user feedback verbosity.\
""".rstrip()
)


# 1. Standard Decorator (no arguments) both with params and without params
# 2. Decorator with Arguments (takes args like @my_decorator(x=1)), need hint
# Called with params: @_decorator(), Called without params: @_decorator
# Hint: Then you'll get TypeError, because func is passed as the first positional arg
# to _decorator, which is not expecting a function yet.
# Hint: prefix _ or pylint: disable=unused-argument
# Hint: from functools import partial _decorator = partial(_decorator, verbose=True)
def save_plot_decorator(
    # Not needed as a placeholder, but kept for parameterized usage
    # *dargs,  # not need placeholder
    # The target function to be decorated (passed when no parameters are used)
    func: "Optional[Callable[..., any]]" = None,
    # *,  # indicates that all following parameters must be passed as keyword
    **dkwargs: dict,  # Keyword arguments passed to the decorator for customization (e.g., verbose)
) -> "Callable[..., any]":
    """
    Decorate that supports both parameterized and non-parameterized usage.

    This decorator can be used directly (`@decorator`) or
    with parameters (`@decorator(param=value)`).
    It wraps the target function, optionally modifying its behavior based on
    decorator-specific arguments.

    This supports both:
    - @decorator
    - @decorator(verbose=True)

    Parameters
    ----------
    *dargs : tuple
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
    - The `func` argument must be placed after `*dargs` to support keyword-only usage and
      to avoid `W1113` (keyword-before-vararg) linter warnings.
    - This structure enables reusability across decorators with shared patterns.
    """

    # The case where the decorator is called with parameters (returns a decorator)
    def decorator(inner_func: "Callable") -> "Callable":
        """
        Actual decorator function that wraps the target function.

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

        @_functools.wraps(inner_func)
        def wrapper(*args, **kwargs) -> "any":
            with safe_tight_layout():
                # Call the actual plotting function
                result = inner_func(*args, **kwargs)
            # result = inner_func(*args, **kwargs)
            # _plt.tight_layout()
            # _plt.draw()
            # _plt.pause(0.1)
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
                    _warnings.warn(
                        "'verbose' parameter should be of type bool.",
                        stacklevel=1,
                    )
                if save_fig and save_fig_filename:
                    save_path = get_file_path(
                        **{**dkwargs, **kwargs},  # Update by inner func
                    )
                    try:
                        # with tempfile.TemporaryDirectory() as tmpdirname:
                        # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:  # tmpfile.name
                        _plt.savefig(
                            save_path, dpi=150, bbox_inches="tight", pad_inches=0
                        )
                        if kwargs.get("verbose", False):
                            print(f"[INFO] Plot saved to: {save_path}")  # noqa: T201
                    except Exception as e:
                        print(f"[ERROR] Failed to save plot: {e}")  # noqa: T201
                if show_fig:
                    # Manage the plot window
                    _plt.show()
                    # _plt.gcf().clear()
                    # _plt.close()
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

# +--------------------+---------------------------------+------------------------+---------------------------------------------------+
# | Function           | Adds New Axis?                  | Common Shape Result    | Substitution / Equivalent                         |
# +====================+=================================+========================+===================================================+
# | np.stack           | ✅ Yes                          | (N, ...)               | —                                                 |
# +--------------------+---------------------------------+------------------------+---------------------------------------------------+
# | np.concatenate     | ❌ No                           | Varies (shared axes)   | np.vstack / np.hstack / np.dstack (special cases) |
# +--------------------+---------------------------------+------------------------+---------------------------------------------------+
# | np.vstack          | ❌ No (axis=0)                  | (N, ...)               | np.concatenate(axis=0), np.row_stack              |
# +--------------------+---------------------------------+------------------------+---------------------------------------------------+
# | np.hstack          | ❌ No (axis=1 for 2D)           | (..., N)               | np.concatenate(axis=1), np.column_stack (1D)      |
# +--------------------+---------------------------------+------------------------+---------------------------------------------------+
# | np.dstack          | ❌ No (axis=2)                  | (..., ..., N)          | np.stack(..., axis=2)                             |
# +--------------------+---------------------------------+------------------------+---------------------------------------------------+
# | np.column_stack    | ❌ No (1D → columns)            | (N, 2) or (N, M)       | np.hstack (2D), np.stack(...).T (1D)              |
# +--------------------+---------------------------------+------------------------+---------------------------------------------------+
# | np.row_stack       | ❌ No (1D → rows)               | (2, N) or (N, ...)     | np.vstack                                         |
# +--------------------+---------------------------------+------------------------+---------------------------------------------------+
# | np.block           | ❌ No (custom nested layout)    | Flexible               | None — general case for complex layouts           |
# +--------------------+---------------------------------+------------------------+---------------------------------------------------+


@save_plot_decorator
def stack(
    *figs: tuple,
    orient: str = "vertical",
    padding: "float | None" = None,
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

    Other Parameters
    ----------------
    %(_save_plot_decorator_kwargs_doc)s

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
    >>> fig = sp.stack(fig1, fig2)

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
            f"Unsupported orient '{orient}'. "
            "Use 'vertical' or 'horizontal' (or v/h/x/y)."
        )

    figsize = kwargs.get("figsize")
    dpi = kwargs.get("dpi", 100)

    if figsize is None:
        # Set a basic figsize depending on orientation
        if orient in ["horizontal", "h", "x"]:
            figsize = (3.15 * len(figs), 12)  # (5 * len(figs), 4)
        else:
            figsize = (12, 3.15 * len(figs))  # (5, 4 * len(figs))

    fig, axes = _plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

    # If only one figure, ax will not be a list, so we make it a list
    # ax = [ax] if n_figs == 1 else ax

    # Make axes iterable
    axes = _np.atleast_1d(axes).ravel()

    for ax, fig_ in zip(axes, figs):
        canvas = _mpl.backends.backend_agg.FigureCanvasAgg(fig_)
        canvas.draw()
        image = _np.asarray(canvas.buffer_rgba())
        ax.imshow(image)
        ax.axis("off")

    _plt.tight_layout(pad=padding if padding is not None else 0.5)
    return fig


######################################################################
##
######################################################################
