"""utils_toml."""

import os as _os
import pathlib as _pathlib

from .. import logger as _logger
from ..exceptions import ScikitplotException


def read_toml(
    file_path: str | _os.PathLike,
) -> dict[str, any]:
    """
    Load a TOML configuration file into a Python dictionary.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to the TOML file to be loaded.

    Returns
    -------
    dict
        Parsed TOML content.

    Raises
    ------
    ScikitplotException
        If the file is unreadable or no TOML parser is available.

    Notes
    -----
    - Uses `tomllib` if available (Python 3.11+), otherwise falls back to `tomli` or `toml`.
    - `tomllib`/`tomli` require binary mode (`'rb'`); `toml` uses text mode (`'r'`).
    - All paths are resolved absolutely and support `~` expansion.

    Examples
    --------
    >>> config = read_toml("config.toml")
    >>> config["model"]["name"]
    'mistral-7b'
    """
    # Loaded from compatibility layer
    from .._compat.python import toml as _toml
    from .._compat.python import tomllib as _tomllib

    # For tomllib and tomli, the API expects binary mode and tomllib.load(f)
    # For toml package, tomllib.load() is toml.load() and expects text mode
    path = _pathlib.Path(file_path).expanduser().resolve()

    try:
        if _tomllib:
            with open(path, "rb") as f:
                return _tomllib.load(f)
        elif _toml:
            with open(path, "r", encoding="utf-8") as f:
                return _toml.load(f)
        else:
            raise ScikitplotException(
                "No TOML parser available. Please install `tomli` or `toml`."
            )
    except FileNotFoundError as e:
        raise ScikitplotException(f"TOML file not found: {path}") from e
    except Exception as e:
        msg = f"Failed to read TOML file at {path}: {e}"
        _logger.exception(msg)
        raise ScikitplotException(msg) from e


def write_toml(
    file_path: str | _os.PathLike,
    data: dict[str, any],
) -> str:
    """
    Write a Python dictionary as a TOML configuration file.

    Parameters
    ----------
    file_path : str or os.PathLike
        Output path for the TOML file.
    data : dict
        Configuration data to be written.

    Returns
    -------
    str
        Absolute path to the written file.

    Raises
    ------
    ScikitplotException
        If writing fails or `toml` is not installed.

    Notes
    -----
    - Only the `toml` package supports writing TOML files.
    - The target directory must be writable.

    Examples
    --------
    >>> data = {"model": {"name": "mistral-7b", "provider": "hf"}}
    >>> path = write_toml("config_out.toml", data)
    >>> print(f"Saved to {path}")
    """
    # Loaded from compatibility layer
    from .._compat.python import toml as _toml

    # For tomllib and tomli, the API expects binary mode and tomllib.load(f)
    # For toml package, tomllib.load() is toml.load() and expects text mode
    path = _pathlib.Path(file_path).expanduser().resolve()

    if not _toml:
        raise ScikitplotException(
            "Writing TOML requires the `toml` package. "
            "Install it via `pip install toml`."
        )
    try:
        with open(path, "w", encoding="utf-8") as f:
            _toml.dump(data, f)
        _logger.info(f"TOML config successfully saved to {path}")
        return str(path)
    except Exception as e:
        msg = f"Failed to write TOML to {path}: {e}"
        _logger.exception(msg)
        raise ScikitplotException(msg) from e
