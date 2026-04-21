# scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_validators.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Input validation helpers for the _url_helper submodule.

Every public function in this module is a *guard*: it either returns
``None`` on success or raises a descriptive exception on failure.  No
function returns a boolean — callers are not expected to branch on the
result.

Notes
-----
Developer
    Keep validators free of side effects.  They must be safe to call
    multiple times with the same arguments (idempotent).

    All validators raise ``TypeError`` for wrong-type arguments and
    ``ValueError`` for arguments of the right type but invalid content.
    ``FileNotFoundError`` / ``NotADirectoryError`` are raised only when
    a filesystem check is explicitly part of the contract.
"""

from __future__ import annotations

import re
from pathlib import Path

from ._constants import (
    MAX_URL_LENGTH,
    TEMPLATE_SUFFIX,
)

# ---------------------------------------------------------------------------
# String validators
# ---------------------------------------------------------------------------


def validate_non_empty_string(value: object, name: str) -> None:
    """Raise if *value* is not a non-empty :class:`str`.

    Parameters
    ----------
    value : object
        The value to test.
    name : str
        Human-readable parameter name used in the error message.

    Raises
    ------
    TypeError
        If *value* is not a :class:`str`.
    ValueError
        If *value* is an empty string or contains only whitespace.

    Examples
    --------
    >>> validate_non_empty_string("hello", "greeting")  # OK — no error
    >>> validate_non_empty_string("", "greeting")
    Traceback (most recent call last):
        ...
    ValueError: 'greeting' must be a non-empty string; got ''.
    """
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be a str; got {type(value).__name__!r}.")
    if not value.strip():
        raise ValueError(f"'{name}' must be a non-empty string; got {value!r}.")


def validate_url_length(url: str) -> None:
    """Raise if *url* exceeds :data:`~._constants.MAX_URL_LENGTH`.

    Parameters
    ----------
    url : str
        Fully assembled URL string to check.

    Raises
    ------
    TypeError
        If *url* is not a :class:`str`.
    ValueError
        If ``len(url) > MAX_URL_LENGTH``.

    Examples
    --------
    >>> validate_url_length("https://example.com")  # OK — no error
    """
    validate_non_empty_string(url, "url")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(
            f"URL length {len(url)} exceeds the maximum of {MAX_URL_LENGTH} "
            "characters.  Shorten the bootstrap code or use a URL shortener."
        )


# ---------------------------------------------------------------------------
# Path / filesystem validators
# ---------------------------------------------------------------------------


def validate_directory(path: Path, name: str) -> None:
    """Raise if *path* is not an existing directory.

    Parameters
    ----------
    path : Path
        Filesystem path to test.
    name : str
        Human-readable parameter name used in the error message.

    Raises
    ------
    TypeError
        If *path* is not a :class:`~pathlib.Path`.
    FileNotFoundError
        If *path* does not exist.
    NotADirectoryError
        If *path* exists but is not a directory.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     validate_directory(pathlib.Path(d), "src_dir")  # OK
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"'{name}' must be a pathlib.Path; got {type(path).__name__!r}."
        )
    if not path.exists():
        raise FileNotFoundError(f"'{name}' directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"'{name}' path exists but is not a directory: {path}")


def validate_template_file(path: Path) -> None:
    """Raise if *path* does not point to a readable template file.

    A valid template file must:

    * exist on the filesystem,
    * be a regular file (not a directory or device),
    * have the suffix :data:`~._constants.TEMPLATE_SUFFIX`.

    Parameters
    ----------
    path : Path
        Candidate template path.

    Raises
    ------
    TypeError
        If *path* is not a :class:`~pathlib.Path`.
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *path* exists but does not have the expected suffix.
    IsADirectoryError
        If *path* is a directory.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / "example.rst.template"
    ...     _ = p.write_text("{{ content }}", encoding="utf-8")
    ...     validate_template_file(p)  # OK
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"Template path must be a pathlib.Path; got {type(path).__name__!r}."
        )
    if not path.exists():
        raise FileNotFoundError(f"Template file does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Template path is a directory, not a file: {path}")
    if not str(path).endswith(TEMPLATE_SUFFIX):
        raise ValueError(
            f"Template file must end with '{TEMPLATE_SUFFIX}'; got: {path.name!r}"
        )


# ---------------------------------------------------------------------------
# Kernel / code validators
# ---------------------------------------------------------------------------


def validate_kernel_name(kernel: str) -> None:
    """Raise if *kernel* is not a valid Jupyter kernel identifier.

    A valid kernel name contains only ASCII letters, digits, hyphens,
    underscores, and dots.

    Parameters
    ----------
    kernel : str
        Kernel name string (e.g. ``"python"`` or ``"python3"``).

    Raises
    ------
    TypeError
        If *kernel* is not a :class:`str`.
    ValueError
        If *kernel* is empty, whitespace-only, or contains illegal
        characters.

    Examples
    --------
    >>> validate_kernel_name("python")  # OK
    >>> validate_kernel_name("python 3")
    Traceback (most recent call last):
        ...
    ValueError: Kernel name contains illegal characters: 'python 3'.
    """
    validate_non_empty_string(kernel, "kernel")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", kernel):
        raise ValueError(
            f"Kernel name contains illegal characters: {kernel!r}.  "
            "Only ASCII letters, digits, hyphens, underscores and dots "
            "are permitted."
        )
