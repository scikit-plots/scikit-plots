"""env_utils."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught

import os as _os
import pathlib as _pathlib

from .. import logger as _logger
from ..exceptions import ScikitplotException


def run_load_dotenv(
    dotenv_path: "str | _os.PathLike[str]" = "",
    override: bool = False,
    verbose: bool = False,
) -> list[str]:
    """
    Load environment variables from a `.env` file into the process environment.

    Parameters
    ----------
    dotenv_path : str or pathlib.Path, optional
        Path to the `.env` file. If not provided, searches upward from CWD.
    override : bool, default=False
        If True, overwrite existing environment variables.
    verbose : bool, default=False
        If True, logs the list of variables loaded (not values).

    Returns
    -------
    bool
        True if at least one environment variable is set else False

    Raises
    ------
    ScikitplotException
        If loading fails or `dotenv` is not installed.

    Examples
    --------
    >>> run_load_dotenv("path/to/.env", override=True, verbose=True)
    """
    try:
        from dotenv import (  # type: ignore[reportMissingImports]
            find_dotenv,
            load_dotenv,
        )
    except ImportError as e:
        msg = "Missing required package 'python-dotenv'. Install with `pip install python-dotenv`."
        _logger.exception(msg)
        raise ScikitplotException(msg) from e

    try:
        # Resolve and validate path
        if dotenv_path:
            dotenv_path = _pathlib.Path(dotenv_path).expanduser().resolve()
            if not dotenv_path.is_file():
                raise ScikitplotException(f".env file not found at: {dotenv_path}")
        else:
            dotenv_path = find_dotenv(usecwd=True)

        # Load .env
        _logger.info(f"Loading environment variables from: {dotenv_path}")
        return load_dotenv(dotenv_path=dotenv_path, override=override, verbose=verbose)

    except Exception as e:
        msg = f"Failed to load .env file: {e}"
        _logger.exception(msg)
        raise ScikitplotException(msg) from e
