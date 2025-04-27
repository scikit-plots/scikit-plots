"""
This module provides utilities for managing file paths
for saving result images (such as plots).
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import os  # noqa: I001
import re
import shutil

# from pathlib import Path
from datetime import datetime

from typing import TYPE_CHECKING  # pylint: disable=wrong-import-order

if TYPE_CHECKING:
    # Only imported during type checking
    from typing import (  # noqa: F401
        Any,
        Dict,
        List,
        Optional,
        Tuple,
        Union,
    )


######################################################################
## get_result_image_path
######################################################################


def _filename_extension_normalizer(
    filename: str,
    ext: "Optional[str]" = None,
    allowed_exts: "Tuple[str, ...]" = (".png", ".jpg", ".jpeg", ".pdf"),
    default_ext: str = ".png",
) -> "Tuple[str, str]":
    """
    Normalize a filename and ensure it has a valid extension.

    Ensures that the given filename has a valid extension from a predefined list.
    If not provided or not found, it falls back to a default extension.

    Parameters
    ----------
    filename : str
        The input filename, with or without an extension.
    ext : str, optional
        An explicit extension to use. If provided, it overrides the one in `filename`.
    allowed_exts : tuple of str, optional
        A tuple of allowed extensions. Defaults to (".png", ".jpg", ".jpeg", ".pdf").
    default_ext : str, optional
        Default extension to use if none is provided or inferred.

    Returns
    -------
    tuple of (str, str)
        - `filename`: The filename without extension.
        - `ext`: The normalized extension (with leading dot).

    Examples
    --------
    >>> _normalize_filename_extension("chart.png")
    ('chart', '.png')

    >>> _normalize_filename_extension("photo", ext=".jpg")
    ('photo', '.jpg')

    >>> _normalize_filename_extension("document.PDF", allowed_exts=(".pdf",))
    ('document', '.pdf')

    >>> _normalize_filename_extension("archive", ext="zip")
    ('archive', '.png')  # fallback to default_ext

    >>> _normalize_filename_extension("output")
    ('output', '.png')  # Uses default_ext

    Notes
    -----
    - Case-insensitive matching of allowed extensions is applied.
    - The returned extension always includes the leading dot.
    - If the provided extension is not allowed, `default_ext` is used.
    - Errors in parsing are safely caught, and the default extension is used.
    """
    try:
        filename_lower = filename.lower()
        matched_ext = None

        if ext is None:
            for allowed in allowed_exts:
                if filename_lower.endswith(allowed.lower()):
                    filename, matched_ext = os.path.splitext(filename)  # noqa: PTH122
                    break
            if matched_ext is None:
                filename, _ = os.path.splitext(filename)  # noqa: PTH122
                matched_ext = default_ext
        else:
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext.lower() in map(str.lower, allowed_exts):
                matched_ext = ext
            else:
                matched_ext = default_ext

        return filename, matched_ext

    except Exception:
        # Gracefully fallback if anything goes wrong
        return filename, default_ext


def _filename_sanitizer(filename: str) -> str:
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


def _filename_uniquer(full_path, file_path, filename):
    """
    Check if the file already exists, and if so,
    modify the filename to avoid overwriting.

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
        A tuple containing the full path, file path, and filename,
        ensuring uniqueness.
    """
    base, extension = os.path.splitext(filename)  # noqa: PTH122
    counter = 1
    while os.path.exists(full_path):  # noqa: PTH110
        new_filename = f"{base}_{counter}{extension}"
        full_path = os.path.join(file_path, new_filename)  # noqa: PTH118
        counter += 1
    filename = os.path.basename(full_path)  # noqa: PTH119
    return full_path, file_path, filename


def get_file_path(
    *,
    filename=None,
    ext=None,
    file_path=None,
    subfolder=None,
    add_timestamp=False,
    return_parts=False,
    overwrite=True,
    verbose=False,
    **kwargs,
):
    """
    Generate a full file path for saving result images,
    ensuring the target directory exists.

    Parameters
    ----------
    filename : str, optional
        Base name of the image file. Defaults to 'plot_image'.
    ext : str, optional, default=None
        File extension (e.g., '.png', '.jpg').
        Defaults to try to find `filename` if not fallback to '.png'.
    file_path : str, optional
        Directory path to save the image.
        Defaults to the current working directory.
    subfolder : str, optional
        Optional subdirectory inside the main path.
    add_timestamp : bool, optional, default=False
        Whether to append a timestamp to the filename.
        Default is False.
    overwrite : bool, optional, default=True
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
    allowed_exts = (".png", ".jpg", ".jpeg", ".pdf")

    # Default to 'plot_image' if no filename provided
    if filename is None:
        filename = "plot_image"
    filename = _filename_sanitizer(filename)

    filename, ext = _filename_extension_normalizer(filename, ext, allowed_exts)

    if ext.lower() not in allowed_exts:
        raise ValueError(f"Extension '{ext}' not supported. Use one of: {allowed_exts}")

    # Add timestamp to filename if specified
    if add_timestamp:
        if filename.endswith(ext):
            filename = filename.rstrip(ext)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%SZ")
        filename = f"{filename}_{timestamp}"

    # Ensure the extension is included
    if not filename.endswith(ext):
        filename += ext

    # Set the default file path if not provided
    if file_path is None:
        file_path = os.path.join(os.getcwd(), "result_images")  # noqa: PTH109,PTH118

    # Add subfolder to path if provided
    if subfolder:
        file_path = os.path.join(  # noqa: PTH118
            file_path, _filename_sanitizer(subfolder)
        )  # noqa: PTH118

    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)  # noqa: PTH103

    # Full path of the file
    full_path = os.path.join(file_path, filename)  # noqa: PTH118

    # Handle file overwriting if the flag is set to False
    if not overwrite:
        full_path, file_path, filename = _filename_uniquer(
            full_path, file_path, filename
        )

    # Verbose output for debugging
    if verbose:
        print(f"[INFO] Saving path to: {full_path}")  # noqa: T201

    # Return full path or path components based on the return_parts flag
    if return_parts:
        return full_path, file_path, filename

    return full_path


######################################################################
## remove_paths
######################################################################


def remove_paths(
    paths: "Optional[List[str]]" = None,
    base_path: "Optional[str]" = None,
) -> None:
    """
    Removes unwanted files or directories from a specified base path.

    Parameters
    ----------
    paths : List[str], optional
        A list of directory or file names to be removed.
        If these exist in the `base_path`, they will be deleted.
        (default ['__MACOSX', 'bank-additional']).

    base_path : str, optional
        The base directory where the unwanted paths will be removed from.
        If None, it defaults to the current working directory (cwd).

    Notes
    -----
    - It checks if the path exists before trying to remove it.
    - Uses `shutil.rmtree()` for directories and `os.remove()` for files.
    - Any exceptions raised during removal will be silently ignored.
    """
    if base_path is None:
        base_path = os.getcwd()  # noqa: PTH109  # Default to current working directory

    if paths is None:
        paths = [
            "__MACOSX",
            "bank-additional",
        ]  # Default for modelplotpy bank data

    for p in paths:
        try:
            path_to_remove = os.path.join(base_path, p)  # noqa: PTH118

            # Check if it's a file and remove it
            if os.path.isfile(  # noqa: PTH113
                path_to_remove
            ) and os.path.exists(  # noqa: PTH110, PTH113
                path_to_remove
            ):  # noqa: PTH110
                os.remove(path_to_remove)  # noqa: PTH107

            # Check if it's a directory and remove it
            elif os.path.isdir(  # noqa: PTH112
                path_to_remove
            ) and os.path.exists(  # noqa: PTH110, PTH112
                path_to_remove
            ):  # noqa: PTH110
                shutil.rmtree(path_to_remove)
        except Exception:  # noqa: W0718
            # Log the error silently or add specific logging if needed
            pass


######################################################################
##
######################################################################
