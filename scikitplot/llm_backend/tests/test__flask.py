"""Flask Test Example (sync)."""

# from flask import Flask, request, jsonify
# from .. import get_backend

# app = Flask(__name__)
# backend = get_backend("transformers", "mistralai/Mistral-7B-Instruct-v0.1")

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     messages = data.get("messages", [])
#     response = backend.chat(messages)
#     return jsonify({"response": response})

# @app.route("/embed", methods=["POST"])
# def embed():
#     data = request.json
#     texts = data.get("texts", [])
#     embeddings = backend.embed(texts)
#     return jsonify({"embeddings": embeddings})
