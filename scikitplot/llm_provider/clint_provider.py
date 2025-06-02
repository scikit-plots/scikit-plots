# scikitplot/llm_provider/clint_provider.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""clint_provider."""

import importlib

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# ----------------------------
# Provider clients
# ----------------------------


class ClientImportError(ImportError):
    """Custom exception raised when a client import or attribute lookup fails."""


def import_client(module_name: str, class_name: str, **kwargs: any) -> any:
    """
    Dynamically import a client class from a module and instantiate it with an API key.

    Parameters
    ----------
    module_name : str
        The name of the Python module to import.
    class_name : str
        The class name within the module to instantiate.
    **kwargs : dict, optional
        Additional keyword arguments passed to the client constructor.
        - api_key : str
          The API key or `token` used to authenticate the client.

    Returns
    -------
    object
        An instance of the requested client class.

    Raises
    ------
    ClientImportError
        Raised if the module cannot be imported or the class is not found in the module.
    """
    try:
        # Use importlib for cleaner imports
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ClientImportError(f"Could not import module '{module_name}': {e}") from e

    try:
        # Attempt to get the client class from the module
        client_cls = getattr(module, class_name)
    except AttributeError as e:
        raise ClientImportError(
            f"Module '{module_name}' has no attribute '{class_name}'"
        ) from e

    # Instantiate the client with api_key and additional parameters
    # huggingface use token parameters
    return client_cls(**kwargs)


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


def get_client(
    model_provider: str,
    api_key: str,
    **kwargs: any,
) -> any:
    """
    Return the appropriate client instance based on the model provider.

    Parameters
    ----------
    model_provider : str
        The name of the model provider, e.g., 'openai', 'groq', 'huggingface'.
    api_key : str
        API `api_key` or `token` used to authenticate with the "model_provider.model_id".
    **kwargs : dict, optional
        Additional keyword arguments forwarded to the client constructor.
        - base_url : str, optional
          `model_provider` API format compatible with OpenAI `base_url` params.
          Like (OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com"))

    Returns
    -------
    object
        An initialized client instance for the specified provider.

    Raises
    ------
    ValueError
        If the specified model_provider is not supported.
    """
    if model_provider not in CLIENT_FACTORY:
        raise ValueError(f"Unknown model_provider '{model_provider}'")

    # Call the appropriate factory with api_key and any extra kwargs
    # client = get_client("openai", "my_openai_api_key", model="gpt-4", timeout=30)
    return CLIENT_FACTORY[model_provider](api_key, **kwargs)


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
