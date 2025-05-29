"""llm_provider."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught

import os
from typing import Optional

from .env_utils import run_load_dotenv
from .secret_utils import get_env_st_secrets
from .stream_utils import streamlit_stream_or_return  # noqa: F401


def get_env_var(env_var: str = "", fallback: Optional[str] = None) -> Optional[str]:
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
        return os.getenv(env_var) or get_env_st_secrets(env_var, fallback)
    except Exception:
        return fallback
