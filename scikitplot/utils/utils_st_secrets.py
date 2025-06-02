"""secret_utils."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught

import os

from .. import logger
from ..exceptions import ScikitplotException
from .utils_toml import read_toml, write_toml

# Default path to Streamlit secrets file (user config dir)
# secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
DEFAULT_SECRETS_PATH = os.getenv("STREAMLIT_CONFIG_DIR") or os.path.expanduser(
    "~/.streamlit/secrets.toml"
)


def resolve_secret_path(secret_path: str = "") -> str:
    """Resolve absolute path to secrets.toml."""
    return os.path.abspath(os.path.expanduser(secret_path or DEFAULT_SECRETS_PATH))


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
    if os.path.exists(path):
        try:
            return read_toml(path)
        except ScikitplotException as e:
            logger.warning(
                f"Failed to load secrets to file at {os.path.basename(path)}: "
                f"{type(e).__name__}"
            )
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
    path = resolve_secret_path(secret_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        write_toml(path, secrets_dict)
    except ScikitplotException as e:
        # 🔒 Updated save_st_secrets (secure):
        logger.error(
            f"Failed to save secrets to file at {os.path.basename(path)}: "
            f"{type(e).__name__}"
        )
        raise


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
