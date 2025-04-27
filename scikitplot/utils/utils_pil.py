"""utils_pil.py"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation

import functools
import logging
import os
import platform
import warnings

# import inspect

## Attempt to import `cache` from `functools` (Python >= 3.9)
try:
    from functools import cache
except ImportError:
    # Fallback to `lru_cache` for Python < 3.9
    from functools import lru_cache as cache

import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]  # noqa: I001
from PIL import (  # type: ignore[reportMissingModuleSource]
    # Image,
    # ImageColor,
    # ImageDraw,
    ImageFont,
)

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

    import PIL  # type: ignore[reportMissingModuleSource]

# Set up logging
logging.basicConfig(level=logging.INFO)


## Define __all__ to specify the public interface of the module
__all__ = [
    "get_font",
    "save_image_pil_decorator",
]


######################################################################
## get_font
######################################################################


@cache
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
        supported_exts = os.getenv("SUPPORTED_EXTS", ".ttf .otf .ttc").split()
        if os.path.exists(font_path) and font_path.lower().endswith(  # noqa: PTH110
            supported_exts
        ):  # noqa: PTH110
            try:
                if verbose:
                    print(f"Using custom font: {font_path}")  # noqa: T201
                return _cached_truetype(font_path, font_size)
            except OSError as e:
                logging.error(f"Failed to load font from '{font_path}': {e}")
        else:
            logging.warning(f"Invalid font path or unsupported font file: {font_path}")

    # Platform-specific fallback
    try:
        system_platform = platform.system().lower()
        default_font_paths = {
            # "windows": os.getenv("DEFAULT_WINDOWS_FONT", "C:/Windows/Fonts/segoeui.ttf"),
            "windows": os.getenv("DEFAULT_WINDOWS_FONT", "C:/Windows/Fonts/arial.ttf"),
            # "darwin": os.getenv("DEFAULT_MAC_FONT", "/System/Library/Fonts/Helvetica.ttc"),
            "darwin": os.getenv("DEFAULT_MAC_FONT", "/Library/Fonts/Arial.ttf"),
            # "DEFAULT_LINUX_FONT", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
            # "DEFAULT_LINUX_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            "linux": os.getenv(
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
        logging.warning(f"Error loading system font: {e}")  # noqa: W1203

    logging.warning("Falling back to PIL default font.")
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
## save_image_pil
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
    os.makedirs(os.path.dirname(to_file), exist_ok=True)  # noqa: PTH103, PTH120
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

    except Exception as e:
        warnings.warn(
            f"[ERROR] Could not saved image using PIL to '{to_file}': {e}",
            stacklevel=1,
        )


# 1. Standard Decorator (no arguments) both with params and without params
# 2. Decorator with Arguments (takes args like @my_decorator(x=1)), need hint
# Called with params: @_decorator(), Called without params: @_decorator
# Hint: Then you'll get TypeError, because func is passed as the first positional arg
# to _decorator, which is not expecting a function yet.
# Hint: prefix _ or pylint: disable=unused-argument  # noqa: W0613
# Hint: from functools import partial _decorator = partial(_decorator, verbose=True)
def save_image_pil_decorator(
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
                backend = kwargs.get("backend", "matplotlib")
                show_os_viewer = kwargs.get("show_os_viewer", False)
                # mpl
                show_fig = kwargs.get("show_fig", True)
                save_fig = kwargs.get("save_fig", False)
                # Automatically get the name of the calling script using inspect.stack()
                # caller_filename = inspect.stack()[1].filename
                save_fig_filename = (
                    kwargs.get("save_fig_filename", dkwargs.get("filename"))
                    or inner_func.__name__
                )
                dkwargs["filename"] = save_fig_filename

                # Handle verbosity if specified
                if "verbose" in kwargs and not isinstance(kwargs["verbose"], bool):
                    warnings.warn(
                        "'verbose' parameter should be of type bool.",
                        stacklevel=1,
                    )
                # print(f"[INFO]:\n\t{kwargs}\n\t{dkwargs}\n\t{save_fig}\n\t{save_fig_filename}\n")
                # Save the plot if save_image is True
                if save_fig and save_fig_filename:
                    save_path = get_file_path(
                        **{**dkwargs, **kwargs},  # Update by inner func
                    )
                    if str(backend).lower() in ("matplotlib", "true", "none"):
                        try:
                            plt.imshow(result)
                            plt.axis("off")

                            plt.tight_layout()
                            # plt.draw()
                            # plt.pause(0.1)  # Pause to allow for interactive drawing
                            try:
                                # Save the image using Matplotlib after showing it
                                plt.savefig(
                                    save_path,
                                    dpi=150,
                                    bbox_inches="tight",
                                    pad_inches=0,
                                )
                                if kwargs.get("verbose", False):
                                    print(  # noqa: T201
                                        f"[INFO] Image saved using Matplotlib: {save_path}"
                                    )
                            except Exception as e:
                                print(f"[ERROR] Failed to save plot: {e}")  # noqa: T201
                            if show_fig:
                                # Manage the plot window
                                plt.show()
                                # plt.gcf().clear()  # Clear the figure after saving
                                # plt.close()
                        except Exception as e:
                            warnings.warn(
                                "[ERROR] Could not saved image using Matplotlib to "
                                f"'{save_path}': {e}. Falling back to PIL.",
                                stacklevel=1,
                            )
                            # Using PIL to save the image (fallback method)
                            save_image_with_pil(
                                result,
                                save_path,
                                show_os_viewer=show_os_viewer,
                                **kwargs,
                            )
                    else:
                        # Using PIL to save the image
                        save_image_with_pil(
                            result, save_path, show_os_viewer=show_os_viewer, **kwargs
                        )
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
