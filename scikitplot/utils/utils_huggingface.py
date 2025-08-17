"""huggingface helper."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=no-name-in-module
# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught

import os as _os

from .. import logger as _logger
from ..exceptions import ScikitplotException

# from openai import OpenAI
# from groq import Groq
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
# from huggingface_hub import InferenceClient  # on cloud
# from transformers import TFAutoModelForCausalLM, AutoTokenizer  # on local


def hf_login(
    cli: bool = False,
    token: "str | None" = None,
) -> None:
    """
    Authenticate with Hugging Face using the `huggingface_hub` login utility.

    Supports both CLI-based interactive login and environment-token login for
    automated/headless use cases.

    Parameters
    ----------
    cli : bool, default=False
        If True, prompts user for token via terminal. If False, uses `HUGGINGFACE_TOKEN`
        environment variable or a provided token.
    token : str, optional
        Optional override token to use instead of environment variable.

    Raises
    ------
    ScikitplotException
        If login fails or token is missing when required.

    Examples
    --------
    >>> hf_login(cli=True)  # Launches interactive CLI login
    >>> hf_login()  # Uses env var HUGGINGFACE_TOKEN
    >>> hf_login(token="hf_xxx")  # Uses provided token
    """
    try:
        from huggingface_hub import login

        if cli:
            ## This prompts you interactively for your token
            _logger.info("Starting interactive Hugging Face login via CLI...")
            login()
            _logger.info("Successfully logged in via CLI.")
        else:
            # Prefer explicitly provided token over environment (non-interactive)
            token = token or _os.getenv("HUGGINGFACE_TOKEN")

            if not token:
                raise ScikitplotException(
                    "Missing token: Set `HUGGINGFACE_TOKEN` env var or "
                    "pass it directly to `hf_login(token=...)`."
                )

            _logger.info("Logging in to Hugging Face Hub via token...")
            ## Log in using the token (non-interactive)
            login(token=token)
            ## Or configure a HfApi client
            # hf_api = HfApi(
            #     token=token, # Token is not persisted on the machine.
            #     # endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
            # )
            _logger.info("Hugging Face login successful.")

        # logout()
        ## Use root method
        # models = list_models()
        # models = hf_api.list_models()

    except ImportError as e:
        msg = (
            "huggingface_hub is not installed. "
            "Run `pip install huggingface_hub` to use this feature."
        )
        _logger.exception(msg)
        raise ScikitplotException(msg) from e

    except Exception as e:
        msg = f"Failed to log in to Hugging Face Hub: {e}"
        _logger.exception(msg)
        raise ScikitplotException(msg) from e
