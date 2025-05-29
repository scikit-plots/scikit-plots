"""secret_utils."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught

import os

import toml

# import yaml
# import requests

# Define expected secrets.toml path
# secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
# Path to Streamlit secrets file (user config dir)
SECRETS_FILE = os.path.expanduser("~/.streamlit/secrets.toml")


# Load existing secrets (if file exists)
def load_st_secrets(secret_path="") -> dict:
    """
    Load existing secrets from the Streamlit secrets file.

    Parameters
    ----------
    secret_path : str, optional
        Path to a `secrets.toml` file to load secrets from if `st.secrets` is empty.
        Defaults to None, "~/.streamlit/secrets.toml".

    Returns
    -------
    dict
        Dictionary of stored secrets.
    """
    secret_path = secret_path or SECRETS_FILE
    secret_path = os.path.abspath(os.path.expanduser(secret_path))
    if os.path.exists(secret_path):
        with open(secret_path, "r", encoding="utf-8") as f:
            return toml.load(f)
    return {}


# Save updated secrets back
def save_st_secrets(secrets_dict: dict, secret_path="") -> None:
    """
    Save secrets to the Streamlit secrets file.

    Parameters
    ----------
    secrets_dict : dict
        Dictionary of secrets to store.
    secret_path : str, optional
        Path to a `secrets.toml` file to load secrets from if `st.secrets` is empty.
        Defaults to None, "~/.streamlit/secrets.toml".
    """
    secret_path = secret_path or SECRETS_FILE
    secret_path = os.path.abspath(os.path.expanduser(secret_path))
    os.makedirs(os.path.dirname(secret_path), exist_ok=True)
    with open(secret_path, "w", encoding="utf-8") as f:
        toml.dump(secrets_dict, f)


def get_env_st_secrets(key, default=None):
    """
    Safely retrieves a secret value from Streamlit's secrets dictionary.

    If the secrets file is missing or the key does not exist, the function
    returns the provided default value.

    Parameters
    ----------
    key : str
        The key to look up in Streamlit's `st.secrets` dictionary.
    default : Any, optional
        The fallback value to return if the key is not found or secrets are not available.
        Defaults to None.

    Returns
    -------
    Any
        The secret value associated with `key`, or `default` if not found.

    Examples
    --------
    >>> product = get_secret("PRODUCT", "default-product")
    >>> print(product)
    'default-product'
    """
    try:
        # pylint: disable=import-outside-toplevel
        import streamlit as st

        return (
            st.secrets.get(key)
            if hasattr(st, "secrets") and key in st.secrets
            else default
        )
    except Exception:
        return default
