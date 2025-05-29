# llm_backend/retrievers/weaviate_retriever.py
from typing import List

import weaviate
from llm_backend.retrievers.base import Retriever


class WeaviateRetriever(Retriever):
    """
    WeaviateRetriever enables semantic search retrieval from a Weaviate vector database.

    Args:
        client (weaviate.Client): An initialized Weaviate client instance.
        class_name (str): The name of the Weaviate class (schema) to query.

    Notes
    -----
    Replace "content" with the actual property name of the text field in your Weaviate schema.

    You must initialize the Weaviate client before passing it to WeaviateRetriever, e.g.:

    ```python
    import weaviate

    client = weaviate.Client(url="http://localhost:8080")
    retriever = WeaviateRetriever(client, class_name="Document")
    results = retriever.retrieve("your search query", top_k=5)
    ```
    """

    def __init__(self, client: weaviate.Client, class_name: str):
        self.client = client
        self.class_name = class_name

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top_k documents from Weaviate based on the query string.

        Args:
            query (str): The search query.
            top_k (int): Number of results to return.

        Returns
        -------
            List[str]: List of retrieved document texts.
        """
        result = (
            self.client.query.get(
                self.class_name, ["content"]
            )  # Adjust 'content' to your schema field
            .with_near_text({"concepts": [query]})
            .with_limit(top_k)
            .do()
        )

        documents = []
        try:
            for item in result["data"]["Get"][self.class_name]:
                documents.append(item.get("content", ""))
        except KeyError:
            pass

        return documents
