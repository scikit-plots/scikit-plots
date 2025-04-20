"""
This module provides utilities for managing file paths for saving result images (such as plots)
and includes decorators for automatically saving plots.

Functions:
-----------
1. sanitize_filename(filename: str) -> str
   - Sanitizes the filename by removing or replacing invalid characters for most filesystems.
     Ensures that the filename is safe to use across different platforms.

2. get_result_image_path(
   filename=None,
   file_path=None,
   add_timestamp=False,
   ext=".png",
   subfolder=None,
   verbose=False,
   overwrite=True,
   return_parts=False
   ) -> str
   - Generates a safe file path for saving result images, ensuring that the target directory exists.
     The function can:
       - Optionally add a timestamp to the filename (useful for versioning).
       - Handle file overwriting by setting the `overwrite` flag (default is True).
       - Optionally place the image in a subfolder (useful for organizing plots).
       - Return parts of the path (filename, subfolder, etc.) if `return_parts=True`.
     The generated file path is safe and ready to use for saving images.

3. auto_save_plot_default(filename=None, **path_kwargs)
   - A decorator that automatically saves a plot when a function is executed.
   - This version saves the plot with a default file name and allows customization of the file path
     using `path_kwargs` (such as `subfolder`, `timestamp`, `extension`, and `verbose`).

4. auto_save_plot_with_params(filename=None, **path_kwargs)
   - A more flexible decorator that automatically saves a plot when a function is executed.
   - This version checks the function's parameters (`save_fig` and `save_fig_filename`) to decide
     whether to save the plot and which filename to use.
   - Allows dynamic control over whether the plot is saved and the filename used at runtime.
"""

import os
import re
from datetime import datetime
import matplotlib.pyplot as plt


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing invalid characters for most filesystems.

    Parameters
    ----------
    filename : str
        The original filename.

    Returns
    -------
    str
        A sanitized filename safe for saving.
    """
    # Replace invalid characters with '_'
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def get_unique_filename(full_path, file_path, filename):
    """
    Check if the file already exists, and if so, modify the filename to avoid overwriting.

    Parameters
    ----------
    full_path : str
        The complete path of the file to check.
    file_path : str
        The directory where the file should be saved.
    filename : str
        The base filename to check.

    Returns
    -------
    tuple
        A tuple containing the full path, file path, and filename, ensuring uniqueness.
    """
    base, extension = os.path.splitext(filename)
    counter = 1
    while os.path.exists(full_path):
        new_filename = f"{base}_{counter}{extension}"
        full_path = os.path.join(file_path, new_filename)
        counter += 1
    filename = os.path.basename(full_path)
    return full_path, file_path, filename


def get_result_image_path(
    filename=None,
    ext=".png",
    file_path=None,
    subfolder=None,
    add_timestamp=False,
    overwrite=True,
    return_parts=False,
    verbose=False,
):
    """
    Generate a full file path for saving result images, ensuring the target directory exists.

    Parameters
    ----------
    filename : str, optional
        Base name of the image file. Defaults to 'plot_image'.
    ext : str, optional
        File extension (e.g., '.png', '.jpg'). Defaults to '.png'.
    file_path : str, optional
        Directory path to save the image. Defaults to the current working directory.
    subfolder : str, optional
        Optional subdirectory inside the main path.
    add_timestamp : bool, optional
        Whether to append a timestamp to the filename. Default is False.
    overwrite : bool, optional
        If False and a file exists, auto-increments the filename to avoid overwriting.
    return_parts : bool, optional
        If True, returns (full_path, file_path, filename) instead of just the full path.
    verbose : bool, optional
        If True, prints the final save path.

    Returns
    -------
    str or tuple
        The full file path, or a tuple (full_path, file_path, filename) if return_parts=True.

    Raises
    ------
    ValueError
        If the provided file extension is not supported.
    """
    # Validate file extension
    allowed_exts = [".png", ".jpg", ".jpeg", ".pdf"]
    if ext.lower() not in allowed_exts:
        raise ValueError(f"Extension '{ext}' not supported. Use one of: {allowed_exts}")

    # Default to 'plot_image' if no filename provided
    if filename is None:
        filename = "plot_image"
    filename = sanitize_filename(filename)

    # Add timestamp to filename if specified
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"

    # Ensure the extension is included
    if not filename.endswith(ext):
        filename += ext

    # Set the default file path if not provided
    if file_path is None:
        file_path = os.path.join(os.getcwd(), "result_images")

    # Add subfolder to path if provided
    if subfolder:
        file_path = os.path.join(file_path, sanitize_filename(subfolder))

    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Full path of the file
    full_path = os.path.join(file_path, filename)

    # Handle file overwriting if the flag is set to False
    if not overwrite:
        full_path, file_path, filename = get_unique_filename(
            full_path, file_path, filename
        )

    # Verbose output for debugging
    if verbose:
        print(f"[INFO] Saving to: {full_path}")

    # Return full path or path components based on the return_parts flag
    if return_parts:
        return full_path, file_path, filename

    return full_path


def auto_save_plot_default(
    save_fig=False,
    filename=None,
    file_formats=None,
    **path_kwargs,
):
    """
    Default version of the decorator that automatically saves a plot when a function is executed.

    Parameters
    ----------
    save_fig : bool, optional
        Whether to save the figure. Defaults to False.
    filename : str, optional
        The base name for the image file. If None, uses the function's name.
    **path_kwargs : dict
        Keyword arguments passed to `get_result_image_path` to customize the save location.

    Returns
    -------
    function
        A decorator that automatically saves the plot after the decorated function runs.
    """

    def decorator(plot_func):
        def wrapper(*args, **kwargs):
            result = plot_func(*args, **kwargs)
            # Save the figure if save_fig is True
            if save_fig:
                # Default to PNG if no formats are specified
                if file_formats is None:
                    file_formats = [path_kwargs.get("ext", ".png")]
                # Loop over the file formats to save the plot in different formats
                for fmt in file_formats:
                    filename_to_save = filename or plot_func.__name__
                    save_path = get_result_image_path(
                        filename=filename_to_save, ext=fmt, **path_kwargs
                    )
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.1)
                    # Try to save the plot and handle any exceptions
                    try:
                        plt.savefig(save_path)
                    except Exception as e:
                        print(f"[ERROR] Failed to save plot: {e}")
                    if path_kwargs.get("verbose", False):
                        print(f"[INFO] Plot saved to: {save_path}")
            # Manage the plot window
            plt.show()
            # plt.gcf().clear()
            # plt.close()
            return result

        return wrapper

    return decorator


def auto_save_plot_with_params(filename=None, **path_kwargs):
    """
    A decorator that automatically saves a plot after the function execution,
    using parameters passed into the function to determine if and how to save the plot.

    Parameters
    ----------
    filename : str, optional
        The base name of the image file. Defaults to None.
    **path_kwargs : dict
        Customization options for saving the plot, such as file format, subfolder, etc.

    Returns
    -------
    function
        A decorator that saves the plot after the function execution if `save_fig` is True.
    """

    def decorator(plot_func):
        def wrapper(*args, **kwargs):
            result = plot_func(*args, **kwargs)
            # Get dynamic saving parameters from the function arguments
            save_fig = kwargs.get("save_fig", False)
            save_fig_filename = (
                kwargs.get("save_fig_filename", filename) or plot_func.__name__
            )
            # Save the plot if save_fig is True
            if save_fig:
                filename_to_save = save_fig_filename or filename
                save_path = get_result_image_path(
                    filename=filename_to_save,
                    **path_kwargs,
                )
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)
                try:
                    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
                except Exception as e:
                    print(f"[ERROR] Failed to save plot: {e}")
                if path_kwargs.get("verbose", False):
                    print(f"[INFO] Plot saved to: {save_path}")
            # Manage the plot window
            plt.show()
            # plt.gcf().clear()
            # plt.close()
            return result

        return wrapper

    return decorator
