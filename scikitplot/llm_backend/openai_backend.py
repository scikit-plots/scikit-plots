# llm_backend/openai_backend.py
import os
from collections.abc import Generator
from typing import Dict, List, Optional, Union

import httpx
import openai
from tenacity import retry, stop_after_attempt, wait_random

from .base import ChatMessage, LLMBackend
from .utils import truncate_messages

FLAVOR_NAME = "openai"


class OpenAIBackend(LLMBackend):
    functions = [
        {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term"},
                },
                "required": ["query"],
            },
        }
    ]

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Args:
            model_name (str): OpenAI model name (e.g. 'gpt-4').
            api_key (str, optional): OpenAI API key. If None, will use env var OPENAI_API_KEY.
        """
        self.model = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided via argument or OPENAI_API_KEY environment variable"
            )
        openai.api_key = self.api_key  # Set global OpenAI key for openai library

    def validate_messages(self, messages: List[Dict]) -> List[Dict]:
        try:
            return [ChatMessage(**msg).dict() for msg in messages]
        except ValueError as e:
            raise ValueError(f"Invalid message format: {e}")

    @retry(wait=wait_random(1, 2), stop=stop_after_attempt(3))
    def chat(
        self, messages: List[Dict], stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        if stream:
            raise NotImplementedError("Streaming not supported yet.")
        messages = self.validate_messages(messages)
        messages = truncate_messages(messages, self)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=self.functions,
            function_call="auto",
        )
        return response.choices[0].message.content

    async def chat_async(
        self,
        messages: List[Dict],
        functions: Optional[List[Dict]] = None,
        function_call: Union[str, Dict, None] = None,
    ) -> Union[str, Dict]:
        messages = self.validate_messages(messages)
        messages = truncate_messages(messages, self)
        if not functions:
            functions = self.functions

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "functions": functions,
            "function_call": function_call or "auto",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
                data = await response.json()
                message = data["choices"][0]["message"]
                return message if "function_call" in message else message["content"]
            except Exception as e:
                return f"OpenAI async request failed: {e!s}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = openai.Embedding.create(input=texts, model=self.model)
        return [r["embedding"] for r in response["data"]]

    def tokenize(self, text: str) -> List[int]:
        # This assumes token counting is implemented elsewhere, as `openai.Engine(id=self.model).tokens()` is not valid
        return [ord(c) for c in text]  # Placeholder

    def count_tokens(self, messages: List[Dict]) -> int:
        return sum(len(self.tokenize(msg["content"])) for msg in messages)

    def reset(self) -> None:
        pass
