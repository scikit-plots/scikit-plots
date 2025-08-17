# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught

"""utils_env."""

import os as _os

from .utils_dot_env import run_load_dotenv
from .utils_st_secrets import get_env_st_secrets


def get_env_var(env_var: str = "", fallback: "str | None" = None) -> "str | None":
    """
    Retrieve token from environment variable or Streamlit secrets.

    Parameters
    ----------
    env_var : str
        The name of the environment variable.
    fallback : Optional[str]
        Optional fallback value if env var is not set.

    Returns
    -------
    Optional[str]
        Token value from environment or secrets, or fallback.
    """
    try:
        run_load_dotenv(override=False)
        return _os.getenv(env_var) or get_env_st_secrets(env_var, fallback)
    except Exception:
        return fallback
