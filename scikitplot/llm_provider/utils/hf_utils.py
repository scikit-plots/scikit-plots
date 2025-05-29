"""huggingface helper."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=no-name-in-module
# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught

import os

# from openai import OpenAI
# from groq import Groq
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
# from huggingface_hub import InferenceClient  # on cloud
# from transformers import TFAutoModelForCausalLM, AutoTokenizer  # on local


def hf_login(cli: bool = False):
    """hf_login."""
    from huggingface_hub import HfApi, list_models, login, logout  # noqa: F401

    if cli:
        ## This prompts you interactively for your token
        login()
    else:
        ## Get token from env variable
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
        ## Log in using the token (non-interactive)
        login(token=token)
        ## Or configure a HfApi client
        # hf_api = HfApi(
        #     token=token, # Token is not persisted on the machine.
        #     # endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
        # )
    # logout()
    ## Use root method
    # models = list_models()
    # models = hf_api.list_models()
