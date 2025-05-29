# llm_backend/rag.py
from .base import LLMBackend
from .retrievers.base import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm_backend: LLMBackend):
        self.retriever = retriever
        self.llm = llm_backend

    def generate_answer(
        self,
        query: str,
        system_prompt: str = "Use the context below to answer the question.",
    ) -> str:
        # Step 1: Retrieve relevant documents
        context_chunks = self.retriever.retrieve(query, top_k=5)
        context_text = "\n\n".join(context_chunks)

        # Step 2: Build messages
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{context_text}"},
            {"role": "user", "content": query},
        ]

        # Step 3: Generate response
        return self.llm.chat(messages)


class RAGPipelineAsync(RAGPipeline):
    async def generate_answer_async(
        self,
        query: str,
        system_prompt: str = "Use the context below to answer the question.",
    ) -> str:
        context_chunks = self.retriever.retrieve(query, top_k=5)
        context_text = "\n\n".join(context_chunks)

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{context_text}"},
            {"role": "user", "content": query},
        ]

        if not hasattr(self.llm, "chat_async"):
            raise NotImplementedError("LLM backend does not support async chat")

        return await self.llm.chat_async(messages)
