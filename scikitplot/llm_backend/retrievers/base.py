# llm_backend/retrievers/base.py
from abc import ABC, abstractmethod
from typing import List


class Retriever(ABC):
    """
    Abstract base class for document retrievers.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top_k relevant documents for the given query.

        Args:
            query (str): The input search query.
            top_k (int): Number of top documents to retrieve. Default is 5.

        Returns
        -------
            List[str]: A list of retrieved document texts or identifiers.
        """
