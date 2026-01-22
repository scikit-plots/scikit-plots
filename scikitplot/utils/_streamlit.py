# ruff: noqa: PLC0415
# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught
# pylint: disable=no-name-in-module

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""_streamlit.py."""

import os as _os
from collections.abc import Generator
from typing import Union

from .. import logger as _logger
from .._compat.optional_deps import HAS_STREAMLIT, safe_import
from ..exceptions import ScikitplotException
from ._toml import read_toml


def streamlit_stream_or_return(
    content: Union[str, Generator[str, None, None]],
) -> Union[str, None]:
    """
    Conditionally stream content in Streamlit if streaming is available.

    The content is an iterable (but not a string). Otherwise, return content as-is.

    Checks:
    - HAS_STREAMLIT is enabled
    - Streamlit object `st` has a `write_stream` method
    - content is iterable but not a string

    Parameters
    ----------
    content : Any
        The content to stream or return.

    Returns
    -------
    str or None
        The streamed content if streaming happens, else the original content.
    """
    if HAS_STREAMLIT:
        st = safe_import("streamlit")
        if (
            hasattr(st, "write_stream")
            and hasattr(content, "__iter__")
            and not isinstance(content, str)
        ):
            return st.write_stream(content)
    return content


# Default path to Streamlit secrets file (user config dir)
# secrets_path = _os.path.join(_os.getcwd(), ".streamlit", "secrets.toml")
DEFAULT_SECRETS_PATH = _os.getenv("STREAMLIT_CONFIG_DIR") or _os.path.expanduser(
    "~/.streamlit/secrets.toml"
)


def resolve_secret_path(secret_path: str = "") -> str:
    """Resolve absolute path to secrets.toml."""
    return _os.path.abspath(_os.path.expanduser(secret_path or DEFAULT_SECRETS_PATH))


# Load existing secrets (if file exists)
def load_st_secrets(
    secret_path: str = "",
) -> dict:
    """
    Load secrets from a TOML file (e.g., Streamlit `secrets.toml`).

    Parameters
    ----------
    secret_path : str, optional
        Path to secrets TOML file. Defaults to `~/.streamlit/secrets.toml`.

    Returns
    -------
    dict
        Parsed secrets dictionary. Empty if file doesn't exist.
    """
    path = resolve_secret_path(secret_path)
    if _os.path.exists(path):
        try:
            return read_toml(path)
        except ScikitplotException:
            _logger.error("Failed to load streamlit secrets to file at.")
    return {}


# Save updated secrets back
def save_st_secrets(
    secrets_dict: dict,
    secret_path: str = "",
) -> None:
    """
    Save secrets to a TOML file (e.g., Streamlit `secrets.toml`).

    Parameters
    ----------
    secrets_dict : dict
        Secrets dictionary to persist.
    secret_path : str, optional
        Path to secrets TOML file. Defaults to `~/.streamlit/secrets.toml`.
    """
    from ._toml import write_toml

    path = resolve_secret_path(secret_path)
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    try:
        write_toml(path, secrets_dict)
    except ScikitplotException:
        # 🔒 Updated save_st_secrets (secure):
        _logger.error("Failed to save secrets to file at.")


def get_env_st_secrets(
    key: str,
    default: "any | None" = None,
) -> any:
    """
    Get a secret value from Streamlit's secrets or return a fallback.

    Parameters
    ----------
    key : str
        The key to fetch from Streamlit secrets.
    default : Any, optional
        Default value to return if key is missing or Streamlit is unavailable.

    Returns
    -------
    Any
        Retrieved secret value or default fallback.
    """
    try:
        import streamlit as st

        return (
            st.secrets.get(key, default)
            if hasattr(st, "secrets") and key in st.secrets
            else default
        )
    except Exception:
        return default
