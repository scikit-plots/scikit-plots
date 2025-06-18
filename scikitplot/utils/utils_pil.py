"""utils_pil.py."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-argument
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

# import inspect
# import logging
# import warnings
import functools as _functools
import os as _os
import platform as _platform
from typing import TYPE_CHECKING

import matplotlib.pyplot as _plt  # noqa: ICN001
from PIL import (
    # Image,
    # ImageColor,
    # ImageDraw,
    ImageFont,
)

from .. import logger as _logger
from .._docstrings import _docstring
from ..exceptions import ScikitplotException
from .utils_path import get_file_path
from .utils_plot_mpl import safe_tight_layout

# Runtime-safe imports for type hints (avoids runtime overhead)
if TYPE_CHECKING:
    # Only imported during type checking
    from typing import Optional, Union

    import PIL

## Define __all__ to specify the public interface of the module
__all__ = [
    "get_font",
    "save_image_pil_decorator",
]

######################################################################
## get_font
######################################################################


@_functools.lru_cache
def _cached_truetype(
    path: str,
    size: int,
) -> "PIL.ImageFont.ImageFont":
    """
    Load a TrueType font with the specified path and size.

    Parameters
    ----------
    path : str
        Path to the TrueType (.ttf or .otf) font file.

    size : int
        Font size to be used.

    Returns
    -------
    PIL.ImageFont.ImageFont
        A loaded TrueType font object.

    Raises
    ------
    OSError
        If the font file cannot be opened or read.
    """
    return ImageFont.truetype(path, size)


def _validate_font_size(
    font_size: "Optional[int]",
) -> int:
    """
    Validate that the font size is a positive integer.

    Parameters
    ----------
    font_size : int or None
        The font size to validate.

    Returns
    -------
    int
        The validated font size.

    Raises
    ------
    ValueError
        If `font_size` is not a positive integer.
    """
    if not isinstance(font_size, int) or font_size <= 0:
        raise ValueError("font_size must be a positive integer.")
    return font_size


def load_default_font(font_size: int = 10) -> "PIL.ImageFont.ImageFont":
    """
    Load the default PIL font with version compatibility for different Pillow versions.

    Parameters
    ----------
    font_size : int, optional
        Font size to apply. Will be used if supported by the current Pillow version.
        Default is 10.

    Returns
    -------
    PIL.ImageFont.ImageFont
        A default font object suitable for drawing text.
    """
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        # For Pillow versions < 9.2.0 that do not support the `size` argument
        return ImageFont.load_default()


def load_font(
    font_path: "Optional[str]" = None,
    font_size: int = 10,
    use_default_font: bool = True,
    verbose: bool = False,
) -> "PIL.ImageFont.ImageFont":
    """
    Get a font object for text rendering, with an optional custom font path and size.

    Parameters
    ----------
    font_path : str, optional
        Path to the font file. If None, the default system font will be used.
        Default is None.

    font_size : int, optional
        Size of the font to load. Must be a positive integer. Default is 10.

    use_default_font : bool, optional
        If True, directly uses `ImageFont.load_default(size=font_size)`
        regardless of the platform or custom font path. Default is True.

    verbose : bool, optional
        If True, prints font path used. Default is False.

    Returns
    -------
    PIL.ImageFont.ImageFont
        The font object that can be used for text rendering.
    """
    font_size = _validate_font_size(font_size)

    # Use PIL's default font directly
    if use_default_font:
        if verbose:
            print("Using PIL default font.")  # noqa: T201
        load_default_font(font_size=font_size)

    # Try loading custom font
    if font_path:
        supported_exts = _os.getenv("SUPPORTED_EXTS", ".ttf .otf .ttc").split()
        if _os.path.exists(font_path) and font_path.lower().endswith(  # noqa: PTH110
            supported_exts
        ):  # noqa: PTH110
            try:
                if verbose:
                    print(f"Using custom font: {font_path}")  # noqa: T201
                return _cached_truetype(font_path, font_size)
            except OSError as e:
                _logger.error(f"Failed to load font from '{font_path}': {e}")
        else:
            _logger.warning(f"Invalid font path or unsupported font file: {font_path}")

    # Platform-specific fallback
    try:
        system_platform = _platform.system().lower()
        default_font_paths = {
            # "windows": _os.getenv("DEFAULT_WINDOWS_FONT", "C:/Windows/Fonts/segoeui.ttf"),
            "windows": _os.getenv("DEFAULT_WINDOWS_FONT", "C:/Windows/Fonts/arial.ttf"),
            # "darwin": _os.getenv("DEFAULT_MAC_FONT", "/System/Library/Fonts/Helvetica.ttc"),
            "darwin": _os.getenv("DEFAULT_MAC_FONT", "/Library/Fonts/Arial.ttf"),
            # "DEFAULT_LINUX_FONT", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
            # "DEFAULT_LINUX_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            "linux": _os.getenv(
                "DEFAULT_LINUX_FONT",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            ),
        }
        system_font_path = default_font_paths.get(system_platform)

        if system_font_path:
            if verbose:
                print(f"Using system font: {system_font_path}")  # noqa: T201
            return _cached_truetype(system_font_path, font_size)
    except (OSError, ValueError) as e:
        _logger.warning(f"Error loading system font: {e}")

    _logger.warning("Falling back to PIL default font.")
    return load_default_font(font_size=font_size)


def get_font(
    font: "Optional[Union[PIL.ImageFont.ImageFont, dict]]" = None,
) -> "PIL.ImageFont.ImageFont":
    """
    Obtain a font object, either directly or by resolving parameters via `load_font`.

    Parameters
    ----------
    font : PIL.ImageFont.ImageFont or dict, optional
        If an ImageFont object, it is returned as is.
        If a dictionary, it is passed to `load_font()` to generate the font.

    Returns
    -------
    PIL.ImageFont.ImageFont
        Font object usable with PIL's drawing functions.

    Notes
    -----
    Pass `{"use_default_font": True, "font_size": 14}` to override size using default font.
    """
    if isinstance(font, ImageFont.ImageFont):
        return font

    if isinstance(font, dict):
        return load_font(**font)

    # for None
    return load_font()


######################################################################
## save_image_with_pil
######################################################################


def save_image_with_pil(
    img: "PIL.Image.Image",
    to_file: str,
    show_os_viewer: bool = False,
    **kwargs,
) -> None:
    """
    Save a PIL image using the default method and optionally display it in the OS image viewer.

    Parameters
    ----------
    img : PIL.Image.Image
        The image to save.

    to_file : str
        Full path where the image will be saved (e.g., "output/image.png").

    show_os_viewer : bool, optional
        If True, opens the image with the system's default viewer after saving.
        Default is False.

    **kwargs : optional
        Additional keyword arguments passed to `img.save()`, such as:
        - format : str, optional
        - dpi : tuple, optional (e.g., (300, 300))
        - quality : int, optional
        - optimize : bool, optional

    Raises
    ------
    UserWarning
        If the image cannot be saved, a warning is raised instead of an exception.

    Notes
    -----
    - `img.save()` supports many optional arguments depending on the format.
    - `img.show()` may use an external tool
      (e.g., Preview on macOS, default viewer (Photos) on Windows).
    - Uses `img.save(to_file)` to save the image.
    - If the format supports it, metadata such as DPI can be embedded (not shown here).
    """
    # Ensure the directory exists before saving the image
    _os.makedirs(_os.path.dirname(to_file), exist_ok=True)  # noqa: PTH103, PTH120
    try:
        # Save the image using PIL
        # Using PIL to save the image (default method)
        # img.save(to_file, dpi=(300, 300))  # dpi is ignored in PIL unless saving to PDF
        img.save(to_file)
        if kwargs.get("verbose", False):
            print(f"[INFO] Image saved using PIL: {to_file}")  # noqa: T201

        if show_os_viewer:
            # Will open in the system's default viewer
            img.show()

    except ScikitplotException as e:
        _logger.warning(
            f"[ERROR] Could not saved image using PIL to '{to_file}': {e}",
            stacklevel=1,
        )


######################################################################
## save_image_pil_decorator
######################################################################

# The docstrings here must be generic enough to apply to all relevant methods.
_docstring.interpd.register(
    _save_image_pil_kwargs_doc="""\
backend : bool, str, optional, default=None
    Specifies the backend used to process and save the image.
    If the value is one of `'matplotlib'`, `'true'`, or `'none'` (case-insensitive),
    the Matplotlib backend will be used. This is useful for better DPI control and
    consistent rendering. Any other value will fall back to using the PIL backend.
    Default is `None`. Common values include:

    - `'matplotlib'`, `'true'`, `'none'` : Use Matplotlib
    - `'pil'`, `'fast'`, etc. : Use PIL (Python Imaging Library)

    .. versionadded:: 0.4.0
        The `backend` parameter was added to allow switching between PIL and Matplotlib.
show_os_viewer : bool, optional, default=False
    If True, displays the saved image (by PIL) in the system's default image viewer
    using PIL's `.show()` method. Default is False.

    .. versionadded:: 0.4.0
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


@_docstring.interpd
def save_image_pil_kwargs(
    *args: tuple,
    result: "PIL.Image.Image" = None,
    func_name: str = "",
    **kwargs: dict,
):
    """
    Save an image using either Matplotlib or PIL, based on dynamic keyword arguments.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    *args : tuple
        Positional arguments (currently unused).
    result : PIL.Image.Image, optional
        The image result to be saved. Required for PIL fallback.
    func_name : str, optional
        A fallback name for the output file if no filename is provided.
    **kwargs : dict
        Additional keyword arguments to control behavior. Includes:

        - `backend` (str or bool): "matplotlib", "pil", or fallback behavior.
        - `show_fig` (bool): Whether to show the Matplotlib figure. Default is True.
        - `save_fig` (bool): Whether to save the figure. Default is False.
        - `save_fig_filename` or `filename` (str): Name/path to save the file.
        - `show_os_viewer` (bool): Whether to open the image in the OS viewer (PIL).
        - `verbose` (bool): Whether to print status messages. Default is False.

    Other Parameters
    ----------------
    %(_save_image_pil_kwargs_doc)s

    Returns
    -------
    None
        This function has side effects (e.g., saving or showing a plot), and does not return values.

    Notes
    -----
    This utility allows flexible saving of plots or PIL images depending on user input or
    calling context. It prefers Matplotlib but falls back to PIL if needed.

    Examples
    --------
    >>> save_image_pil_kwargs(result=img, save_fig=True, filename="output.png")

    """
    # Extract optional parameters with defaults
    backend: Optional[Union[bool, str]] = kwargs.get(  # noqa: UP037
        "backend",
        "matplotlib",
    )
    # pil
    show_os_viewer: bool = kwargs.get("show_os_viewer", False)
    # mpl
    show_fig: bool = kwargs.get("show_fig", True)
    save_fig: bool = kwargs.get("save_fig", False)
    # filename
    # Automatically get the name of the calling script using inspect.stack()
    # caller_filename = inspect.stack()[1].filename
    # Resolve output filename: prefer explicit, fallback to func name
    save_fig_filename: str = (
        kwargs.get("save_fig_filename", kwargs.get("filename")) or func_name
    )
    # Warn if 'verbose' exists but is not a bool
    # verbose: bool = kwargs.get("verbose", False)
    if "verbose" in kwargs and not isinstance(kwargs["verbose"], bool):
        _logger.warning(
            "'verbose' parameter should be of type bool.",
            stacklevel=1,
        )
    # Proceed with your plotting logic here, e.g.:
    try:
        # Save the plot if save_image is True
        if save_fig:
            save_path = get_file_path(
                # Update for inner func
                **{**kwargs, "filename": save_fig_filename},
            )
            # Use Matplotlib backend (default or fallback)
            if str(backend).lower() in ("matplotlib", "true", "none", None):
                try:
                    if _plt.get_fignums():
                        fig = (
                            _plt.gcf()
                        )  # Get current figure (create one if none exists)
                        ax = _plt.gca()  # Get current axes (create one if none exists)
                    else:
                        fig, ax = (
                            _plt.subplots()
                        )  # Attempt to show and save using matplotlib

                    with safe_tight_layout():
                        ax = ax.imshow(result)  # Display the image on the existing axes
                        # _plt.tight_layout()
                    _plt.axis("off")
                    # _plt.draw()
                    # _plt.pause(0.1)  # Pause to allow for interactive drawing
                    # Save the image using Matplotlib after showing it
                    _plt.savefig(
                        save_path,
                        dpi=150,
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    if show_fig:
                        # Manage the plot window
                        _plt.show()
                        # _plt.gcf().clear()  # Clear the figure after saving
                        # _plt.close()
                    return ax
                except ScikitplotException as e:
                    _logger.exception(
                        "Could not saved image using Matplotlib to "
                        f"'{save_path}': {e}. Falling back to PIL.",
                        stacklevel=1,
                    )
                    # Using PIL to save the image (PIL fallback)
                    save_image_with_pil(
                        img=result,
                        to_file=save_path,
                        show_os_viewer=show_os_viewer,
                    )
            else:
                # Using PIL to save the image (Explicit PIL backend)
                save_image_with_pil(
                    img=result,
                    to_file=save_path,
                    show_os_viewer=show_os_viewer,
                )
        else:
            _logger.info("Could not saved pil image or Axes plot.")
    except ScikitplotException as e:
        _logger.exception(f"Failed to save pil image: {e}")
    return result


# 1. Standard Decorator (no arguments) both with params and without params
# 2. Decorator with Arguments (takes args like @my_decorator(x=1)), need hint
# Called with params: @_decorator(), Called without params: @_decorator
# Hint: Then you'll get TypeError, because func is passed as the first positional arg
# to _decorator, which is not expecting a function yet.
# Hint: prefix _ or pylint: disable=unused-argument
# Hint: from functools import partial _decorator = partial(_decorator, verbose=True)
def save_image_pil_decorator(
    # Not needed as a placeholder, but kept for parameterized usage
    # *dargs,  # not need placeholder
    # The target function to be decorated (passed when no parameters are used)
    func: "Optional[callable[..., any]]" = None,
    # *,  # indicates that all following parameters must be passed as keyword
    **dkwargs: dict,  # Keyword arguments passed to the decorator for customization (e.g., verbose)
) -> "callable[..., any]":
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
    func : callable, optional
        The target function to be decorated. This is automatically set when the decorator
        is used without parentheses (e.g., `@decorator`).
    **dkwargs : dict
        Keyword arguments passed to the decorator for configuration. These can be used
        to customize the behavior of the wrapper.

    Returns
    -------
    callable
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
    def decorator(inner_func: "callable") -> "callable":
        """
        Decorate function that wraps the target function.

        Parameters
        ----------
        inner_func : callable
            The function to be decorated.

        **dkwargs : dict
            Keyword arguments passed to the decorator for customization.

        Returns
        -------
        callable
            The wrapped function.
        """

        @_functools.wraps(inner_func)
        def wrapper(*args, **kwargs) -> "any":
            # Call the actual plotting function
            result = inner_func(*args, **kwargs)
            # Call the validation function to ensure proper fig and ax are set
            ax_or_im = save_image_pil_kwargs(
                result=result,
                func_name=inner_func.__name__,
                **kwargs,
            )
            if ax_or_im:
                _logger.debug(f"Returned object {type(ax_or_im)}")
            return ax_or_im or result

        return wrapper

    # Check if `func` was passed directly (i.e., decorator without parameters)
    if func is not None and callable(func):
        return decorator(func)

    return decorator


######################################################################
##
######################################################################
