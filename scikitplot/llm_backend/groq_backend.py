# llm_backend/groq_backend.py
from collections.abc import Generator
from typing import Dict, List, Optional, Union

from pydantic import ValidationError

from .base import ChatMessage, LLMBackend
from .utils import truncate_messages

FLAVOR_NAME = "groq"


class GroqBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: str):
        from groq import Groq

        self.model = model_name
        self.client = Groq(api_key=api_key)

    def validate_messages(self, messages: List[Dict]) -> List[Dict]:
        try:
            return [ChatMessage(**msg).dict() for msg in messages]
        except ValidationError as e:
            raise ValueError(f"Invalid message format: {e}")

    def chat(
        self, messages: List[Dict], stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        if stream:
            raise NotImplementedError("Streaming not supported yet.")

        messages = self.validate_messages(messages)
        messages = truncate_messages(messages, self)

        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content

    async def chat_async(
        self,
        messages: List[Dict],
        functions: Optional[List[Dict]] = None,
        function_call: Union[str, Dict, None] = None,
    ) -> Union[str, Dict]:
        """
        Placeholder async support. Groq SDK might not support true async yet.
        """
        raise NotImplementedError("Async chat not supported yet for Groq.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Embedding not implemented for Groq.")

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError("Tokenization not implemented for Groq.")

    def count_tokens(self, messages: List[Dict]) -> int:
        raise NotImplementedError("Token counting not implemented for Groq.")

    def reset(self) -> None:
        pass
