"""FastAPI Test Example (async supported)."""

# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List, Dict
# from .. import get_backend

# app = FastAPI()
# backend = get_backend("openai", "gpt-4")

# class Message(BaseModel):
#     role: str
#     content: str

# @app.post("/chat")
# async def chat(messages: List[Message]):
#     msgs = [msg.dict() for msg in messages]
#     response = await backend.chat_async(msgs) if hasattr(backend, "chat_async") else backend.chat(msgs)
#     return {"response": response}

# @app.post("/embed")
# def embed(texts: List[str]):
#     vectors = backend.embed(texts)
#     return {"embeddings": vectors}
