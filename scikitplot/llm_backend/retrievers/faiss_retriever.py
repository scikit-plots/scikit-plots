# llm_backend/retrievers/faiss_retriever.py

import faiss
import numpy as np
from llm_backend.retrievers.base import Retriever


class FaissRetriever(Retriever):
    def __init__(
        self,
        embedding_dim: int,
        index: faiss.Index,
        documents: list[str],
        embeddings: np.ndarray,
    ):
        """
        Faiss retriever using a pre-built index.

        Args:
        embedding_dim (int): Dimension of embeddings.
        index (faiss.Index): Pre-built FAISS index.
        documents (List[str]): List of documents corresponding to embeddings.
        embeddings (np.ndarray): Embeddings matrix of shape (num_docs, embedding_dim).
        """
        self.embedding_dim = embedding_dim
        self.index = index
        self.documents = documents
        self.embeddings = embeddings

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> list[str]:
        """
        Retrieve top_k most similar documents for the query embedding.

        Args:
        query_embedding (np.ndarray): Embedding vector for the query (shape: embedding_dim).
        top_k (int): Number of results to return.

        Returns
        -------
            List[str]: Top_k documents most similar to query.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)  # Shape (1, embedding_dim)

        distances, indices = self.index.search(
            query_embedding.astype(np.float32), top_k
        )
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    @classmethod
    def from_embeddings(cls, documents: list[str], embeddings: np.ndarray):
        """
        Factorier method to create a FaissRetriever from docs and embeddings.

        Args:
            documents (List[str]): List of documents.
            embeddings (np.ndarray): Embeddings matrix.

        Returns
        -------
            FaissRetriever instance
        """
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings.astype(np.float32))
        return cls(embedding_dim, index, documents, embeddings)
