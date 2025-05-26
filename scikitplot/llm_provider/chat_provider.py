"""chat_provider."""

# scikitplot/llm_provider/chat_provider.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=broad-exception-caught

import argparse
from dataclasses import dataclass
from typing import Optional, Union

from .. import logger
from .clint_provider import get_client, hf_fallback_request
from .config_provider import LLM_MODEL_PROVIDER2API_KEY, LLM_MODEL_PROVIDER2CONFIG
from .utils import get_env_var, streamlit_stream_or_return


@dataclass
class ChatMessage:
    """ChatMessage."""

    role: str
    content: str


# ----------------------------
# Main Handler
# ----------------------------


def get_response(
    messages: str | list[dict[str, str]] = "",
    model_provider: str = "huggingface",
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    max_history: int = 10,
    temperature: float = 0.5,
    max_tokens: int = 256,
    # repetition_penalty: float = 1.1,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0.5,
    stream: bool = False,
    **kwargs: any,
) -> Union[str, None]:
    """
    Get response from LLM provider.

    Parameters
    ----------
    messages : str | List[Dict[str, str]]
        List of chat messages with role and content.
    model_provider : str
        One of "openai", "groq", or "huggingface".
    model_id : str, optional
        Model identifier for the provider.
        Default selected `model_provider` list first "model_id".
    api_key : str, optional
        API `api_key` or `token` used to authenticate with the "model_provider.model_id".
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
        Loaded from env by model_provider like "HUGGINGFACE_TOKEN" if not provided.
    stream : bool
        Whether to stream response in Streamlit.
    **kwargs : dict, optional
        Additional keyword arguments forwarded to the client constructor.
        - base_url : str, optional
          `model_provider` API format compatible with OpenAI `base_url` params.
          Like (OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com"))

    Returns
    -------
    Union[str, None]
        Assistant's reply or error message.
    """
    fallback_message = "[ERROR] Model call failed."
    messages = (
        messages
        and isinstance(messages, str)
        and [{"role": "user", "content": f"{messages}"}]
    )
    messages = messages or [{"role": "user", "content": "Hello Assistant!"}]

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
        model_id = model_id or LLM_MODEL_PROVIDER2CONFIG[model_provider][0]["model_id"]
        env_api_key = LLM_MODEL_PROVIDER2API_KEY.get(model_provider, "")
        api_key = api_key or get_env_var(env_api_key)
        if not api_key:
            logger.warning(
                f"[ERROR] Unknown `model_provider` or can't find 'api_key' for '{model_provider}' "
                f"in default providers: {list(LLM_MODEL_PROVIDER2API_KEY)}"
            )
        client = get_client(model_provider, api_key, **kwargs)
        create_params = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if model_provider in {"openai", "groq"}:
            create_params.update(
                {
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                }
            )
        # if model_provider in {"deepseek"} and 'beta' in base_url:
        #     create_params.update(
        #         {
        #             "stop": ["```"],
        #         }
        #     )

        try:
            response = client.chat.completions.create(**create_params)
            content = (
                response
                if stream
                else (
                    response.choices[0].message.get("content")
                    if model_provider == "huggingface"
                    else response.choices[0].message.content
                )
            )
            return streamlit_stream_or_return(content)
        except Exception as e:
            if model_provider == "huggingface":
                logger.exception(f"[HF InferenceClient failed] {e}")
                payload = create_params.copy()
                try:
                    return hf_fallback_request(model_id, api_key, payload)
                except Exception as e1:
                    logger.exception(f"[HF POST fallback failed] {e1}")
                    return fallback_message
            else:
                logger.exception(f"[{model_provider} client failed] {e}")
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
    args = parser.parse_args()

    messages = [{"role": "user", "content": args.message}]
    result = get_response(
        messages=messages,
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


if __name__ == "__main__":
    main()
