# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
TOML configuration helpers.

This module provides strict, import-safe helpers to read and write TOML files.

Design goals
------------
- **Import-safe:** Optional TOML backends are NOT imported at module import time.
  This prevents side effects during bulk imports (e.g., Sphinx/autosummary).
- **Cross-version:** Works on Python 3.8+ (including Python 3.11+ stdlib `tomllib`).
- **Strict:** Missing TOML backends raise clear, actionable errors *only* when
  TOML I/O is requested (not at import time).

The read/write backends are selected deterministically:

Read backends (priority):
- ``tomllib`` (Python 3.11+, stdlib, read-only)
- ``tomli`` (third-party, read-only)
- ``toml`` (third-party, read/write)

Write backends (priority):
- ``toml`` (third-party, read/write)
- ``tomli_w`` (third-party, write-only)

See Also
--------
scikitplot.exceptions.ScikitplotException :
    Library-specific exception raised by this module on failures.
"""

from __future__ import annotations

import importlib
import importlib.util as _importlib_util
import os as _os
import pathlib as _pathlib
import sys as _sys
from typing import Any, Mapping, Union

from .. import logger as _logger  # noqa: F401
from ..exceptions import ScikitplotException

_PathLike = Union[str, _os.PathLike]


def _has_module(module_name: str) -> bool:
    """
    Check if a module is importable without importing it.

    Parameters
    ----------
    module_name : str
        Fully qualified module name (e.g., ``"tomli"``).

    Returns
    -------
    bool
        True if the module is importable, otherwise False.

    Raises
    ------
    None

    See Also
    --------
    importlib.util.find_spec :
        Standard-library function used to locate a module spec.

    Notes
    -----
    This function intentionally avoids importing the module to prevent import-time
    side effects (logging, optional dependency initialization, etc.).

    Examples
    --------
    >>> _has_module("tomli") in (True, False)
    True
    """
    return _importlib_util.find_spec(module_name) is not None


def _read_backend_name() -> str | None:
    """
    Determine the preferred TOML read backend name.

    Parameters
    ----------
    None

    Returns
    -------
    str or None
        One of ``{"tomllib", "tomli", "toml"}`` if available, otherwise None.

    Raises
    ------
    None

    See Also
    --------
    _write_backend_name :
        Determine the preferred TOML write backend.

    Notes
    -----
    Read backend priority is deterministic:
    ``tomllib`` (Py>=3.11) → ``tomli`` → ``toml``.

    Examples
    --------
    >>> _read_backend_name() in ("tomllib", "tomli", "toml", None)
    True
    """
    if _sys.version_info >= (3, 11):
        return "tomllib"
    if _has_module("tomli"):
        return "tomli"
    if _has_module("toml"):
        return "toml"
    return None


def _write_backend_name() -> str | None:
    """
    Determine the preferred TOML write backend name.

    Parameters
    ----------
    None

    Returns
    -------
    str or None
        One of ``{"toml", "tomli_w"}`` if available, otherwise None.

    Raises
    ------
    None

    See Also
    --------
    _read_backend_name :
        Determine the preferred TOML read backend.

    Notes
    -----
    Write backend priority is deterministic:
    ``toml`` → ``tomli_w``.

    Examples
    --------
    >>> _write_backend_name() in ("toml", "tomli_w", None)
    True
    """
    if _has_module("toml"):
        return "toml"
    if _has_module("tomli_w"):
        return "tomli_w"
    return None


# Public capability flags computed WITHOUT importing optional TOML modules.
TOML_READ_SOURCE: str | None = _read_backend_name()
TOML_WRITE_SOURCE: str | None = _write_backend_name()

TOML_READ_SUPPORT: bool = TOML_READ_SOURCE is not None
TOML_WRITE_SUPPORT: bool = TOML_WRITE_SOURCE is not None

# Backward-compatibility: keep `TOML_SOURCE` as the read backend.
TOML_SOURCE: str | None = TOML_READ_SOURCE

__all__ = [
    "TOML_READ_SOURCE",
    "TOML_READ_SUPPORT",
    "TOML_SOURCE",
    "TOML_WRITE_SOURCE",
    "TOML_WRITE_SUPPORT",
    "read_toml",
    "write_toml",
]


def _load_read_backend() -> tuple[str, Any]:
    """
    Import and return the selected TOML read backend.

    Parameters
    ----------
    None

    Returns
    -------
    (str, module)
        Backend name and the imported backend module.

    Raises
    ------
    ScikitplotException
        If no supported TOML read backend is available.

    See Also
    --------
    _load_write_backend :
        Import and return the selected TOML write backend.
    read_toml :
        Public TOML read helper that uses this backend.

    Notes
    -----
    Import happens only when TOML reading is requested (import-safe module design).

    Examples
    --------
    >>> name, mod = _load_read_backend()  # doctest: +SKIP
    >>> name in ("tomllib", "tomli", "toml")
    True
    """
    backend = TOML_READ_SOURCE
    if backend is None:
        raise ScikitplotException(
            "No TOML reader backend is available. "
            "Install `tomli` (Python < 3.11) or `toml`."
        )

    if backend == "tomllib":
        import tomllib  # type: ignore[]  # noqa: PLC0415

        return backend, tomllib

    if backend == "tomli":
        return backend, importlib.import_module("tomli")

    if backend == "toml":
        return backend, importlib.import_module("toml")

    raise ScikitplotException(f"Unsupported TOML read backend: {backend!r}")


def _load_write_backend() -> tuple[str, Any]:
    """
    Import and return the selected TOML write backend.

    Parameters
    ----------
    None

    Returns
    -------
    (str, module)
        Backend name and the imported backend module.

    Raises
    ------
    ScikitplotException
        If no supported TOML write backend is available.

    See Also
    --------
    _load_read_backend :
        Import and return the selected TOML read backend.
    write_toml :
        Public TOML write helper that uses this backend.

    Notes
    -----
    Import happens only when TOML writing is requested (import-safe module design).

    Examples
    --------
    >>> name, mod = _load_write_backend()  # doctest: +SKIP
    >>> name in ("toml", "tomli_w")
    True
    """
    backend = TOML_WRITE_SOURCE
    if backend is None:
        raise ScikitplotException(
            "No TOML writer backend is available. Install `toml` or `tomli-w`."
        )

    if backend == "toml":
        return backend, importlib.import_module("toml")

    if backend == "tomli_w":
        return backend, importlib.import_module("tomli_w")

    raise ScikitplotException(f"Unsupported TOML write backend: {backend!r}")


def _normalize_path(path: _PathLike) -> _pathlib.Path:
    """
    Normalize a path into an absolute :class:`pathlib.Path`.

    Parameters
    ----------
    path : str or os.PathLike
        Input filesystem path.

    Returns
    -------
    pathlib.Path
        Expanded and resolved path.

    Raises
    ------
    ScikitplotException
        If the input cannot be converted to a path.

    See Also
    --------
    pathlib.Path.expanduser :
        Expand ``~`` and ``~user`` constructs.
    pathlib.Path.resolve :
        Resolve to an absolute path.

    Notes
    -----
    This helper provides consistent error handling for path conversion.

    Examples
    --------
    >>> p = _normalize_path(".")
    >>> isinstance(p, _pathlib.Path)
    True
    """
    try:
        return _pathlib.Path(path).expanduser().resolve()
    except Exception as e:
        raise ScikitplotException(f"Invalid path: {path!r}") from e


def read_toml(file_path: _PathLike) -> dict[str, Any]:
    """
    Read a TOML file into a Python dictionary.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to the TOML file.

    Returns
    -------
    dict
        Parsed TOML document.

    Raises
    ------
    ScikitplotException
        If the file does not exist, cannot be read, is a directory, or no TOML
        reader backend is available.

    See Also
    --------
    write_toml :
        Write a mapping to a TOML file.

    Notes
    -----
    - ``tomllib`` and ``tomli`` expect binary mode (``rb``) and use ``.load(fileobj)``.
    - ``toml`` typically uses text mode and also supports ``.load(fileobj)``.
    - This function never logs; callers can log around it if needed.

    Examples
    --------
    >>> cfg = read_toml("config.toml")  # doctest: +SKIP
    >>> isinstance(cfg, dict)
    True
    """
    path = _normalize_path(file_path)
    backend_name, backend = _load_read_backend()

    try:
        if backend_name in ("tomllib", "tomli"):
            with open(path, "rb") as f:
                return backend.load(f)
        if backend_name == "toml":
            with open(path, "r", encoding="utf-8") as f:
                return backend.load(f)
        raise ScikitplotException(f"Unsupported TOML read backend: {backend_name!r}")
    except FileNotFoundError as e:
        raise ScikitplotException(f"TOML file not found: {path}") from e
    except IsADirectoryError as e:
        raise ScikitplotException(f"Expected a file but got a directory: {path}") from e
    except PermissionError as e:
        raise ScikitplotException(f"Permission denied while reading: {path}") from e
    except Exception as e:
        raise ScikitplotException(f"Failed to read TOML file: {path}") from e


def write_toml(
    file_path: _PathLike,
    data: Mapping[str, Any],
    *,
    mkdir: bool = False,
) -> str:
    """
    Write a mapping to a TOML file.

    Parameters
    ----------
    file_path : str or os.PathLike
        Output path for the TOML file.
    data : Mapping[str, Any]
        TOML-serializable configuration mapping.
    mkdir : bool, default=False
        If True, create the parent directory if it does not exist.

    Returns
    -------
    str
        Absolute path to the written TOML file.

    Raises
    ------
    ScikitplotException
        If writing fails, the target is a directory, permission is denied, or no TOML
        writer backend is available.

    See Also
    --------
    read_toml :
        Read a TOML file into a dictionary.

    Notes
    -----
    - ``toml`` writes via ``toml.dump(mapping, fileobj)``.
    - ``tomli_w`` writes via ``tomli_w.dumps(mapping)`` followed by writing the string.
    - This function never logs; callers can log around it if needed.

    Examples
    --------
    >>> out = write_toml(
    ...     "out.toml", {"model": {"name": "mistral-7b"}}
    ... )  # doctest: +SKIP
    >>> isinstance(out, str)
    True
    """
    path = _normalize_path(file_path)
    backend_name, backend = _load_write_backend()

    try:
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)

        if backend_name == "toml":
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                backend.dump(dict(data), f)
            return str(path)

        if backend_name == "tomli_w":
            content = backend.dumps(dict(data))
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            return str(path)

        raise ScikitplotException(f"Unsupported TOML write backend: {backend_name!r}")
    except IsADirectoryError as e:
        raise ScikitplotException(
            f"Expected a file path but got a directory: {path}"
        ) from e
    except PermissionError as e:
        raise ScikitplotException(f"Permission denied while writing: {path}") from e
    except Exception as e:
        raise ScikitplotException(f"Failed to write TOML file: {path}") from e
