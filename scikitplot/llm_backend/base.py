# llm_backend/base.py
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Dict, List, Union

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str
    type: str = "text"


# ------------------------
# Base LLM Backend Interface
# ------------------------


class LLMBackend(ABC):
    @abstractmethod
    def chat(
        self, messages: List[Dict], stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate completion for given messages.
        Return full response or generator if streaming.
        """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Return embeddings for a list of texts.
        """

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize input text.
        """

    @abstractmethod
    def count_tokens(self, messages: List[Dict]) -> int:
        """
        Count tokens in messages.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal state or cache.
        """
