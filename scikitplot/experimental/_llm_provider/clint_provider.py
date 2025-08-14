# scikitplot/llm_provider/clint_provider.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught

"""clint_provider."""

import functools
import importlib

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ... import logger
from ...exceptions import ScikitplotException

__all__ = [
    "CLIENT_FACTORY",
    "get_client",
    "hf_fallback_request",
]

# add numpydoc notes examples types and add each line comments

######################################################################
## CLIENT_FACTORY
######################################################################


# Factory that maps model provider names to their respective client initializer lambdas.
CLIENT_FACTORY: "dict[str, callable[..., any]]" = {
    # HuggingFace Hub Inference client
    "huggingface": lambda token, **kwargs: import_client(
        "huggingface_hub",
        "InferenceClient",
        **{**kwargs, "token": token},
    ),
    # Anthropic (Claude)
    "anthropic": lambda api_key, **kwargs: import_client(
        "anthropic",
        "Anthropic",
        **{**kwargs, "api_key": api_key},
    ),
    # Cohere AI
    "cohere": lambda api_key, **kwargs: import_client(
        "cohere",
        "ClientV2",
        **{**kwargs, "api_key": api_key},
    ),
    # Deepseek AI (example placeholder)
    # https://api.deepseek.com/v1
    # https://api.deepseek.com/beta
    "deepseek": lambda api_key, **kwargs: import_client(
        "openai",
        "OpenAI",
        **{
            **kwargs,
            "api_key": api_key,
            "base_url": kwargs.get("base_url") or "https://api.deepseek.com",
        },
    ),
    # Google Gemini (example placeholder, update when official client available)
    "gemini": lambda api_key, **kwargs: import_client(
        "google.genai",
        "Client",
        **{**kwargs, "api_key": api_key},
    ),
    # Groq AI client
    "groq": lambda api_key, **kwargs: import_client(
        "groq",
        "Groq",
        **{**kwargs, "api_key": api_key},
    ),
    # OpenAI official client
    "openai": lambda api_key, **kwargs: import_client(
        "openai",
        "OpenAI",
        **{**kwargs, "api_key": api_key},
    ),
    # Facebook LLaMA (hypothetical client, adjust based on real SDK)
    "llama": lambda api_key, **kwargs: import_client(
        "llama_client",
        "LlamaClient",
        **{**kwargs, "api_key": api_key},
    ),
    # Google Vertex AI (example, adjust client class accordingly)
    "vertexai": lambda api_key, **kwargs: import_client(
        "google.cloud.aiplatform",
        "Client",
        **{**kwargs, "api_key": api_key},
    ),
}


def import_client(
    module_name: str,
    class_name: str,
    **kwargs: any,
) -> any:
    """
    Dynamically import and instantiate a client class from the given module.

    Parameters
    ----------
    module_name : str
        The name of the Python module (e.g., 'openai', 'huggingface_hub').
    class_name : str
        The name of the class to instantiate from the module.
    **kwargs : Any
        Additional keyword arguments for the class constructor.
        Commonly includes 'api_key' or 'token'.

    Returns
    -------
    Any
        Instantiated client object.

    Raises
    ------
    ScikitplotException
        If the module or class cannot be imported.

    Examples
    --------
    >>> import_client("openai", "OpenAI", api_key="sk-abc123")
    <openai.OpenAI object>
    """
    try:
        # Use importlib for cleaner imports
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ScikitplotException(
            f"Could not import module '{module_name}': {e}"
        ) from e

    try:
        # Attempt to get the client class from the module
        client_cls = getattr(module, class_name)
    except AttributeError as e:
        raise ScikitplotException(
            f"Module '{module_name}' has no attribute '{class_name}'"
        ) from e

    # Instantiate the client with api_key and additional parameters
    # huggingface use token parameters
    return client_cls(**kwargs)


@functools.lru_cache(maxsize=128)
def get_client(
    model_provider: str,
    api_key: str,
    **kwargs: any,
) -> any:
    """
    Retrieve a cached client instance for the specified model provider.

    Parameters
    ----------
    model_provider : str
        Name of the model provider, e.g., 'openai', 'groq', 'huggingface'.
    api_key : str
        API key or token required by the model provider.
    **kwargs : Any
        Optional keyword arguments passed to the client constructor.
        e.g., `base_url`, `timeout`, or `model`.

    Returns
    -------
    Any
        Instantiated and cached client object.

    Raises
    ------
    ValueError
        If an unsupported model provider is passed.

    Examples
    --------
    >>> get_client("openai", api_key="sk-xxx", base_url="https://api.example.com")
    <openai.OpenAI object>
    """
    try:
        if model_provider not in CLIENT_FACTORY:
            raise ValueError(f"Unknown model_provider '{model_provider}'")

        # Call the appropriate factory with api_key and any extra kwargs
        # client = get_client("openai", "my_openai_api_key", model="gpt-4", timeout=30)
        # Return the initialized client
        return CLIENT_FACTORY[model_provider](api_key, **kwargs)
    except Exception as e:
        logger.exception(
            f"Model provider '{model_provider}' could not be imported. Error: {e}"
        )


# ----------------------------
# Fallback Request Logic
# ----------------------------


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def hf_fallback_request(
    model_id: str,
    token: str,
    payload: dict,
) -> str:
    """
    Send a fallback POST request to HuggingFace inference API endpoint.

    Parameters
    ----------
    model_id : str
        Identifier of the model deployed on HuggingFace.
    token : str
        API token used for authenticating the request.
    payload : dict
        The JSON payload containing chat parameters (e.g., messages, temperature).

    Returns
    -------
    str
        The content from the first response message.

    Raises
    ------
    requests.HTTPError
        Raised if the HTTP request fails or returns a non-200 status code.

    Notes
    -----
    - This is typically used as a fallback if the HuggingFace client invocation fails.
    - Automatically retries up to 3 times with exponential backoff.

    Examples
    --------
    >>> hf_fallback_request(
    ...     model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    ...     token="hf_abc123...",
    ...     payload={"messages": [...], "temperature": 0.7},
    ... )
    "Sure, here's how to do that..."
    """
    try:
        # Construct headers with the authorization token
        headers = {"Authorization": f"Bearer {token}"}

        # Construct the target URL for HuggingFace inference router
        url = f"https://router.huggingface.co/hf-inference/models/{model_id}/v1/chat/completions"

        # Send the POST request to the HuggingFace endpoint
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # Raise HTTPError if the request was unsuccessful (non-200 status)
        response.raise_for_status()

        # Parse and return the chat content from the first choice
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.exception(f"Hugging Face fallback request failed: {e}")
