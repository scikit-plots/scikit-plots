"""chat_provider."""

# scikitplot/llm_provider/chat_provider.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=no-name-in-module
# pylint: disable=broad-exception-caught

import argparse
import os
from dataclasses import dataclass
from typing import Generator, Optional, Union  # noqa: UP035

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# import json
# import time
# import toml
# import yaml

try:
    import streamlit as st

    STREAMLIT_MODE = True
except ImportError:
    STREAMLIT_MODE = False

# import logging
from .. import logger
from .utils.env_utils import run_load_dotenv
from .utils.secret_utils import get_env_st_secrets


@dataclass
class ChatMessage:
    """ChatMessage."""

    role: str
    content: str


# default_configs
LLM_MODEL_PROVIDER2ID = {
    "huggingface": [
        {"model_id": "HuggingFaceH4/zephyr-7b-beta", "api_key": ""},
        {"model_id": "mistral-7b-instruct", "api_key": ""},
    ],
    "anthropic": [
        {"model_id": "claude-3-sonnet", "api_key": ""},
        {"model_id": "claude-3-opus-20240229", "api_key": ""},
    ],
    "cohere": [{"model_id": "command-r", "api_key": ""}],
    "gemini": [{"model_id": "gemini-pro", "api_key": ""}],
    "groq": [{"model_id": "llama3-8b-8192", "api_key": ""}],
    "openai": [{"model_id": "gpt-3.5-turbo", "api_key": ""}],
}

# ----------------------------
# Utils
# ----------------------------


def get_api_key_env(env_var: str, fallback: Optional[str] = None) -> Optional[str]:
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
    run_load_dotenv()
    return os.getenv(env_var) or get_env_st_secrets(env_var, fallback)


def display_or_return(
    content: Union[str, Generator[str, None, None]],
) -> Union[str, None]:
    """
    Return content directly or stream it using Streamlit if applicable.

    Parameters
    ----------
    content : Union[str, Generator[str, None, None]]
        The content to display or return.

    Returns
    -------
    Union[str, None]
        Content displayed or returned based on the mode.
    """
    if (
        STREAMLIT_MODE
        and hasattr(st, "write_stream")
        and hasattr(content, "__iter__")
        and not isinstance(content, str)
    ):
        return st.write_stream(content)
    return content


# ----------------------------
# Provider clients
# ----------------------------

CLIENT_FACTORY = {
    "openai": lambda api_key: __import__("openai").OpenAI(api_key=api_key),
    "groq": lambda api_key: __import__("groq").Groq(api_key=api_key),
    "huggingface": (
        lambda token: __import__("huggingface_hub").InferenceClient(token=token)
    ),
}


def get_client(model_type: str, api_key: str):
    """
    Return the appropriate client based on model_type.

    Parameters
    ----------
    model_type : str
        The type of the model provider (e.g., 'openai', 'groq', 'huggingface').
    api_key : str
        API api_key or token for authentication.

    Returns
    -------
    Any
        Initialized client for the specified provider.

    Raises
    ------
    ValueError
        If the model_type is unknown.
    """
    if model_type not in CLIENT_FACTORY:
        raise ValueError(f"Unknown model_type '{model_type}'")
    return CLIENT_FACTORY[model_type](api_key)


# ----------------------------
# Fallback Request Logic
# ----------------------------


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def hf_fallback_request(model_id: str, token: str, payload: dict) -> str:
    """
    Fallback raw POST request to HuggingFace endpoint.

    Parameters
    ----------
    model_id : str
        The model identifier.
    token : str
        API token for authentication.
    payload : dict
        JSON payload for the request.

    Returns
    -------
    str
        Content of the response message.

    Raises
    ------
    requests.HTTPError
        If the request fails.
    """
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}/v1/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ----------------------------
# Main Handler
# ----------------------------


def get_response(
    messages: list[dict[str, str]],
    model_type: str = "huggingface",
    model_id: Optional[str] = None,
    max_history: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 256,
    # repetition_penalty: float = 1.1,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0.3,
    api_key: Optional[str] = None,
    stream: bool = False,
) -> Union[str, None]:
    """
    Get response from LLM provider.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        List of chat messages with role and content.
    model_type : str
        One of "openai", "groq", or "huggingface".
    model_id : str, optional
        Model identifier for the provider.
    max_history : int
        Number of recent messages to retain.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    repetition_penalty : float
        Penalize repeated phrases (HF only).
    frequency_penalty : float
        Penalize frequent tokens (OpenAI/Groq only).
    presence_penalty : float
        Encourage/discourage topic diversity (OpenAI/Groq only).
    api_key : str, optional
        API `api_key` or `token`.
        Loaded from env by model_type like "HUGGINGFACE_TOKEN" if not provided.
    stream : bool
        Whether to stream response in Streamlit.

    Returns
    -------
    Union[str, None]
        Assistant's reply or error message.
    """
    fallback_message = "[ERROR] Model call failed."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant specialized in machine learning topics."
            ),
        },
        *messages[-max_history:],
    ]

    try:
        model_id = model_id or LLM_MODEL_PROVIDER2ID[model_type][0]["model_id"]
        api_key_env_key = {
            "huggingface": "HUGGINGFACE_TOKEN",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
        }.get(model_type)
        if not api_key_env_key:
            return f"[ERROR] Unknown model_type '{model_type}'"
        api_key = api_key or get_api_key_env(api_key_env_key)
        client = get_client(model_type, api_key)

        create_params = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if model_type in {"openai", "groq"}:
            create_params.update(
                {
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                }
            )

        # if model_type == "huggingface":
        #     create_params.update({"repetition_penalty": repetition_penalty})

        try:
            response = client.chat.completions.create(**create_params)
            content = (
                response
                if stream
                else (
                    response.choices[0].message.get("content")
                    if model_type == "huggingface"
                    else response.choices[0].message.content
                )
            )
            return display_or_return(content)
        except Exception as e:
            if model_type == "huggingface":
                logger.exception(f"[HF InferenceClient failed] {e}")
                payload = create_params.copy()
                try:
                    return hf_fallback_request(model_id, api_key, payload)
                except Exception as e1:
                    logger.exception(f"[HF POST fallback failed] {e1}")
                    return fallback_message
            else:
                logger.exception(f"[{model_type} client failed] {e}")
                return fallback_message

    except Exception as e:
        logger.exception(f"[Global Exception] {e}")
        return fallback_message


# ----------------------------
# CLI for Independent Usage
# ----------------------------


def main():
    """CLI for Independent Usage."""
    parser = argparse.ArgumentParser(description="Test LLM provider response.")
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        default="Hello Assistant!",
        help="User Input message for the assistant.",
    )
    parser.add_argument(
        "--model_type",
        "-mtp",
        type=str,
        default="huggingface",
        help="Model type (e.g. openai/groq/huggingface).",
    )
    parser.add_argument(
        "--model_id",
        "-mid",
        type=str,
        default=None,
        help="Optional Model identifier override.",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", "-mt", type=int, default=256, help="Maximum response tokens"
    )
    args = parser.parse_args()

    messages = [{"role": "user", "content": args.message}]
    result = get_response(
        messages=messages,
        model_type=args.model_type,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=False,
    )
    print(  # noqa: T201
        f"[{args.model_type}]",
        result,
    )


if __name__ == "__main__":
    main()
