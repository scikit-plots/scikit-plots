# scikitplot/llm_provider/chat_provider.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=unused-import
# pylint: disable=broad-exception-caught
# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes

"""chat_provider."""

import argparse
import json
from dataclasses import asdict, dataclass, field  # noqa: F401
from typing import Optional, Union

from ... import logger
from ...utils.utils_env import get_env_var
from ...utils.utils_stream import streamlit_stream_or_return
from .clint_provider import get_client, hf_fallback_request
from .model_registry import LLM_PROVIDER_CONFIG_MAP, LLM_PROVIDER_ENV_CONNECTOR_MAP

__all__ = [
    "LLM_PROVIDER_CONFIG_MAP",
    "LLM_PROVIDER_ENV_CONNECTOR_MAP",
    "LLMResponder",
    "get_response",
]

######################################################################
## ChatMessage
######################################################################


@dataclass
class ChatMessage:
    """
    A generalized message object for LLM-based conversations, supporting text, images,
    and structured payloads.

    This abstraction is useful for chat-based applications involving structured data
    like charts, tables, or multimodal content (e.g., plots + text), particularly
    for model explanation use cases.

    Attributes
    ----------
    role : str
        The role of the message sender. One of {"user", "assistant", "system", "tool"}.
    content : Optional[str], default=None
        Main text of the message, if any.
    image : Optional[Union[str, bytes]], default=None
        Optional image content. Can be a URL, file path, or base64 string.
    payload : Optional[Dict[str, Any]], default=None
        Optional structured payload like a chart object, feature importances, or numeric values.
    purpose : Optional[str], default=None
        Optional semantic purpose for the message (e.g., "explain_plot", "diagnose_model").
    name : Optional[str], default=None
        Optional display name or sender ID.
    metadata : Dict[str, Any], default={}
        Arbitrary metadata for tracking timestamps, task context, model IDs, etc.

    Examples
    --------
    >>> ChatMessage(
    ...     role="user",
    ...     content="Explain this confusion matrix",
    ...     payload={"TP": 50, "FN": 10},
    ... )
    >>> ChatMessage(
    ...     role="assistant",
    ...     content="Your model has high recall but low precision",
    ... )
    >>> ChatMessage(
    ...     role="user",
    ...     image="data:image/png;base64,...",
    ...     purpose="explain_plot",
    ... )
    """  # noqa: D205

    role: str  # Required sender role
    content: Optional[str] = None  # Text body of the message
    image: Optional[Union[str, bytes]] = None  # Optional image payload
    payload: Optional[dict[str, any]] = None  # Optional structured data payload
    purpose: Optional[str] = None  # Purpose or intent of the message
    name: Optional[str] = None  # Sender name or tool ID
    metadata: dict[str, any] = field(  # Arbitrary metadata for tracking context
        default_factory=dict,
    )

    def as_dict(self) -> dict[str, any]:
        """
        Convert this ChatMessage into a standard dict format consumable by LLM APIs.

        Returns
        -------
        dict
            Dictionary representation, excluding None fields.
        """
        base = {
            "role": self.role,  # Role is always included
            "content": self.content,  # Content is included if not None
        }

        # Optional fields: only add if present
        if self.image:
            base["image"] = self.image  # Add image if available
        if self.payload:
            base["payload"] = self.payload  # Add structured payload
        if self.purpose:
            base["purpose"] = self.purpose  # Add intent/semantic tag
        if self.name:
            base["name"] = self.name  # Add sender name if available
        if self.metadata:
            base["metadata"] = self.metadata  # Add metadata dict

        # Remove None values for clean API formatting
        return {k: v for k, v in base.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, any]) -> "ChatMessage":
        """
        Create a ChatMessage from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with keys matching ChatMessage attributes.

        Returns
        -------
        ChatMessage
            Parsed ChatMessage object.
        """
        return cls(
            role=d["role"],  # Required
            content=d.get("content"),  # Optional
            image=d.get("image"),  # Optional
            payload=d.get("payload"),  # Optional
            purpose=d.get("purpose"),  # Optional
            name=d.get("name"),  # Optional
            metadata=d.get("metadata", {}),  # Optional, default empty dict
        )

    def to_json(self) -> str:
        """to_json."""
        return json.dumps(self.as_dict())


######################################################################
## helper client response
######################################################################


def get_model_info(
    model_provider: str,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[str, str]:
    """
    Retrieve model ID and API key for a given provider.

    Parameters
    ----------
    model_provider : str
        Name of the model provider (e.g., 'openai', 'huggingface').
    model_id : str, optional
        Specific model ID to use. If not provided, defaults to the first model ID for the provider.
    api_key : str, optional
        API key to authenticate. If not provided, fetches from environment variable.

    Returns
    -------
    tuple[str, str]
        Tuple of resolved model ID and API key.
    """
    model_id = model_id or LLM_PROVIDER_CONFIG_MAP[model_provider][0]["model_id"]
    env_name = LLM_PROVIDER_ENV_CONNECTOR_MAP.get(
        model_provider, ""
    )  # e.g. "HUGGINGFACE_TOKEN"
    api_key = api_key if api_key else get_env_var(env_name)

    if not api_key:
        logger.warning(
            f"[ERROR] API key missing for provider '{model_provider}'. "
            f"Expected env name: {env_name}"
        )
    return model_id, api_key


def format_messages(
    messages: Union[str, list[dict[str, str]]],
    max_history: int = 10,
) -> list[dict[str, str]]:
    """
    Format input into a list of chat messages with system prompt prepended.

    Parameters
    ----------
    messages : str or List[Dict[str, str]]
        User input string or list of messages with roles and content.
    max_history : int
        Number of most recent user messages to keep.

    Returns
    -------
    List[Dict[str, str]]
        List of formatted messages with a system instruction prepended.
    """
    role_system = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant specialized in machine learning topics."
        ),
    }
    if not messages:
        messages = [
            role_system,
            {
                "role": "user",
                "content": "Hello Assistant!",
            },
        ]
    elif isinstance(messages, (str, int)):
        messages = [
            role_system,
            {
                "role": "user",
                "content": f"{messages}",
            },
        ]
    return [
        role_system,
        *messages[-max_history:],
    ]


def build_params(
    provider: str,
    model_id: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
    freq_penalty: float,
    pres_penalty: float,
) -> dict:
    """
    Build parameters dictionary for model request.

    Parameters
    ----------
    provider : str
        Name of the model provider.
    model_id : str
        Model identifier.
    messages : List[Dict[str, str]]
        Formatted list of input messages.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum number of tokens to generate.
    stream : bool
        Enable/disable streaming output.
    freq_penalty : float
        Frequency penalty (OpenAI/Groq only).
    pres_penalty : float
        Presence penalty (OpenAI/Groq only).

    Returns
    -------
    dict
        Dictionary of model request parameters.
    """
    params = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if provider in {"openai", "groq"}:
        params.update(
            {
                "presence_penalty": pres_penalty,
                "frequency_penalty": freq_penalty,
            },
        )
    # if provider in {"deepseek"} and 'beta' in base_url:
    #     create_params.update(
    #         {
    #             "stop": ["```"],
    #         }
    #     )
    return params


def parse_response(
    provider: str,
    stream: bool,
    response: any,
) -> str:
    """
    Extract the content of the response.

    Parameters
    ----------
    provider : str
        Model provider name.
    stream : bool
        Whether the response is streamed.
    response : Any
        Raw response from the model client.

    Returns
    -------
    str
        Extracted content.
    """
    try:
        if stream:
            return response
        if provider == "huggingface":
            return response.choices[0].message.get("content")
        return response.choices[0].message.content
    except Exception as e:
        logger.exception(f"Response parser error: {e}")
        return response


def client_fallback_request(
    provider: str,
    model_id: str,
    api_key: str,
    params: dict,
    fallback_msg: str,
) -> Union[str, any]:
    """
    Attempt a fallback model call in case of failure.

    Parameters
    ----------
    provider : str
        Provider name.
    model_id : str
        Model identifier.
    api_key : str
        API key.
    params : dict
        Parameters for the model call.
    fallback_msg : str
        Fallback message to return on failure.

    Returns
    -------
    Union[str, any]
        Fallback response or error message.
    """
    fallback_msg = fallback_msg or "[ERROR] Model call failed."
    if provider == "huggingface":
        hf_fallback_request(model_id, api_key, params)
    else:
        logger.error(f"[{provider}] fallback not implemented.")
    return fallback_msg


######################################################################
## func based client response
######################################################################


def get_response(
    messages: str | list[dict[str, str]] = "",
    model_provider: str = "huggingface",
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    max_history: int = 10,
    temperature: float = 0.5,
    max_tokens: int = 256,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0.5,
    stream: bool = False,
    # repetition_penalty: float = 1.1,
    **kwargs: any,
) -> Union[str, None]:
    """
    Get response from LLM provider.

    Parameters
    ----------
    messages : str | List[Dict[str, str]]
        List of chat messages with role and content.
    model_provider : str
        Model provider name (e.g. 'openai', 'groq', 'huggingface').
    model_id : str, optional
        Model identifier for the provider.
        Default selected `model_provider` list first "model_id".
    api_key : str, optional
        API `api_key` or `token` used to authenticate with the "model_provider/model_id".
        Loaded from `.env` by `model_provider` like "HUGGINGFACE_TOKEN" if not provided.
    max_history : int
        Number of recent messages to retain.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    frequency_penalty : float
        Penalize frequent tokens (OpenAI/Groq only).
    presence_penalty : float
        Encourage/discourage topic diversity (OpenAI/Groq only).
    stream : bool
        Whether to stream response in Streamlit.
    **kwargs : dict, optional
        Additional keyword arguments forwarded to the client constructor.
        - base_url : str, optional
          `model_provider` API format compatible with OpenAI `base_url` params.
          Like (OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com"))

    Returns
    -------
    str or None
        Assistant's reply or error message.
    """
    fallback_message = "[ERROR] Model call failed."
    try:
        # _initialize_client()
        model_id, api_key = get_model_info(model_provider, model_id, api_key)
        client = get_client(model_provider, api_key, **kwargs)
        try:
            # _prepare_messages
            messages = format_messages(messages, max_history)
            # _prepare_params
            params = build_params(
                model_provider,
                model_id,
                messages,
                temperature,
                max_tokens,
                stream,
                frequency_penalty,
                presence_penalty,
            )
            response = client.chat.completions.create(**params)
            content = parse_response(model_provider, stream, response)
            return streamlit_stream_or_return(content)
        except Exception as e:
            logger.exception(
                f"Failed to get response from '{model_provider}' endpoint: {e}"
            )
            # _handle_fallback
            client_fallback_request(
                model_provider,
                model_id,
                api_key,
                params,
                str(e),
            )
            return fallback_message

    except Exception as e:
        logger.exception(
            f"No client configuration found for model provider '{model_provider}': {e}"
        )
        return fallback_message


# ----------------------------
# Main Handler class based
# ----------------------------


class LLMResponder:
    """
    LLMResponder provides an object-oriented interface for querying language models
    from various providers with support for customization, fallback, and streaming.

    Parameters
    ----------
    model_provider : str, optional
        Provider name, e.g., 'openai', 'huggingface', 'groq'.
    model_id : str, optional
        Specific model identifier to use.
    api_key : str, optional
        API key or token for authentication.
    max_history : int, optional
        Number of historical messages to retain for context.
    temperature : float, optional
        Sampling temperature to control randomness.
    max_tokens : int, optional
        Maximum number of tokens to generate.
    frequency_penalty : float, optional
        Penalty for token frequency (OpenAI/Groq).
    presence_penalty : float, optional
        Penalty to encourage new topic introduction (OpenAI/Groq).
    stream : bool, optional
        Whether to enable streaming responses.
    **kwargs : dict
        Additional parameters passed to the client initialization.
    """  # noqa: D205

    def __init__(
        self,
        model_provider: str = "huggingface",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        max_history: int = 10,
        temperature: float = 0.5,
        max_tokens: int = 256,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        stream: bool = False,
        **kwargs,
    ):
        # Initialize basic configuration
        self.model_provider = model_provider
        self.model_id = model_id
        self.api_key = api_key
        self.max_history = max_history
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stream = stream
        self.kwargs = kwargs

        # Initialize model client
        self.client = self._initialize_client()

    def _initialize_client(self):
        """
        Initialize and return a client object using the selected model provider.

        Returns
        -------
        object
            An instantiated client object for the LLM provider.
        """
        # Resolve model ID and API key using environment or defaults
        self.model_id, self.api_key = get_model_info(
            self.model_provider,
            self.model_id,
            self.api_key,
        )
        # Get the provider-specific client
        return get_client(
            self.model_provider,
            self.api_key,
            **self.kwargs,
        )

    def _prepare_messages(
        self,
        messages: Union[str, list[dict[str, str]]],
    ) -> list[dict[str, str]]:
        """
        Prepare the input message list with system prompt and truncation.

        Parameters
        ----------
        messages : str or list of dict
            The input message(s).

        Returns
        -------
        list of dict
            Formatted message list ready for model input.
        """
        return format_messages(messages, self.max_history)

    def _prepare_params(
        self,
        messages: list[dict[str, str]],
    ) -> dict:
        """
        Construct the parameter dictionary for the LLM API call.

        Parameters
        ----------
        messages : list of dict
            Prepared messages.

        Returns
        -------
        dict
            Model-specific request parameters.
        """
        return build_params(
            self.model_provider,
            self.model_id,
            messages,
            self.temperature,
            self.max_tokens,
            self.stream,
            self.frequency_penalty,
            self.presence_penalty,
        )

    def _handle_fallback(self, params):
        """
        Handle fallback request logic in case the primary request fails.

        Parameters
        ----------
        params : dict
            Parameters used for the original model call.

        Returns
        -------
        str or any
            The fallback result or error message.
        """
        return client_fallback_request(
            self.model_provider,
            self.model_id,
            self.api_key,
            params,
            fallback_msg="[ERROR] Model call failed.",
        )

    def get_response(
        self,
        messages: Union[str, list[dict[str, str]]],
    ) -> Union[str, None]:
        """
        Generate a response from the model for the given messages.

        Parameters
        ----------
        messages : str or list of dict
            The input messages for the model.

        Returns
        -------
        str or None
            Model's response or fallback message.
        """
        try:
            # Format messages and prepare parameters
            messages = self._prepare_messages(messages)
            params = self._prepare_params(messages)
            # Make the API call
            response = self.client.chat.completions.create(**params)
            # Parse and return response
            content = parse_response(self.model_provider, self.stream, response)
            return streamlit_stream_or_return(content)
        except Exception as e:
            # Log and fallback if error occurs
            logger.exception(f"[Model error] {e}")
            return self._handle_fallback(params=params)

    @staticmethod
    def quick_response(**kwargs) -> Union[str, None]:
        """
        Quick functional interface for a single model call.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to LLMResponder including 'messages'.

        Returns
        -------
        str or None
            Model response or error message.

        Examples
        --------
        >>> LLMResponder.quick_response(
        ...     model_provider="openai",
        ...     model_id="gpt-4",
        ...     api_key="your-api-key",
        ...     messages=[{"role": "user", "content": "Hello"}],
        ... )
        """
        try:
            return LLMResponder(**kwargs).get_response(kwargs.get("messages", ""))
        except Exception as e:
            logger.exception(f"[Global Exception] {e}")
            return "[ERROR] Model call failed."


# ----------------------------
# CLI for Independent Usage
# ----------------------------


# pylint: disable=redefined-outer-name
def parse_args(args: list[str] | None = None):
    """CLI for Independent Usage."""
    parser = argparse.ArgumentParser(
        description="Test LLM provider response.",
    )
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        default="",
        help="User Input message for the assistant.",
    )
    parser.add_argument(
        "--model_provider",
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
    args_ = parser.parse_args(args=args)
    logger.debug(args_)
    return args_


if __name__ == "__main__":
    args = parse_args()

    result = get_response(
        messages=args.message,
        model_provider=args.model_provider,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=False,
    )
    print(  # noqa: T201
        f"[{args.model_provider}]",
        result,
    )
