"""utils_toml."""

import os as _os
import pathlib as _pathlib
import sys as _sys

from .. import logger as _logger
from ..exceptions import ScikitplotException

# `tomllib` and `tomli` require binary read mode (`'rb'`), while `toml` uses text mode.
# Track TOML support
TOML_SOURCE = None
TOML_READ_SUPPORT = False
TOML_WRITE_SUPPORT = False

# Try importing tomllib (Python 3.11+)
if _sys.version_info >= (3, 11):
    try:
        import tomllib  # Python 3.11+ builtin, read-only

        TOML_READ_SUPPORT = True
        TOML_SOURCE = "tomllib"
    except ImportError as e:
        _logger.exception("Failed to import built-in `tomllib`: %s", e)
        tomllib = None

# Fallback to `tomli` (read-only)
if not TOML_READ_SUPPORT:
    try:
        import tomli as tomllib  # External tomli, API-compatible with tomllib

        TOML_READ_SUPPORT = True
        TOML_SOURCE = "tomli"
    except ImportError:
        _logger.info(
            "TOML read support requires `tomli` (for Python < 3.11) or `tomllib`."
        )
        tomllib = None

# Fallback to `tomli-w` (write-only)
# if not TOML_WRITE_SUPPORT:
#     try:
#         import tomli_w as tomllib  # External tomli, API-compatible with tomllib

#         TOML_WRITE_SUPPORT = True
#         TOML_SOURCE = "tomli-w"
#     except ImportError:
#         _logger.info(
#             "TOML write support requires `tomli-w` package. Install via `pip install tomli-w`."
#         )
#         tomllib = None

# Fallback to `toml` (read/write)
try:
    import toml  # Supports both read & write

    TOML_WRITE_SUPPORT = True
    if not TOML_READ_SUPPORT:
        TOML_READ_SUPPORT = True
        TOML_SOURCE = "toml"
except ImportError:
    _logger.info(
        "TOML write support requires `toml` package. Install via `pip install toml`."
    )
    toml = None

__all__ = [
    "TOML_READ_SUPPORT",
    "TOML_SOURCE",
    "TOML_WRITE_SUPPORT",
    "read_toml",
    "write_toml",
]


def read_toml(
    file_path: "str | _os.PathLike",
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
    # For tomllib and tomli, the API expects binary mode and tomllib.load(f)
    # For toml package, tomllib.load() is toml.load() and expects text mode
    path = _pathlib.Path(file_path).expanduser().resolve()

    try:
        if tomllib:
            with open(path, "rb") as f:
                return tomllib.load(f)
        elif toml:
            with open(path, "r", encoding="utf-8") as f:
                return toml.load(f)
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
    file_path: "str | _os.PathLike",
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
    # For tomllib and tomli, the API expects binary mode and tomllib.load(f)
    # For toml package, tomllib.load() is toml.load() and expects text mode
    path = _pathlib.Path(file_path).expanduser().resolve()

    if not toml:
        raise ScikitplotException(
            "Writing TOML requires the `toml` package. "
            "Install it via `pip install toml`."
        )
    try:
        with open(path, "w", encoding="utf-8") as f:
            toml.dump(data, f)
        _logger.info(f"TOML config successfully saved to {path}")
        return str(path)
    except Exception as e:
        msg = f"Failed to write TOML to {path}: {e}"
        _logger.exception(msg)
        raise ScikitplotException(msg) from e
