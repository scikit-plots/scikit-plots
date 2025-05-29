"""
llm = get_backend("openai", "gpt-4", api_key="your_openai_api_key").
"""

# llm_backend/__init__.py

# from .openai_backend import OpenAIBackend
# from .groq_backend import GroqBackend
# from .transformers_backend import TransformersBackend

# def get_backend(name: str, model_name: str, **kwargs):
#     """
#     Factory function to get the desired LLM backend instance.

#     Args:
#         name (str): Backend name ('openai', 'groq', 'transformers').
#         model_name (str): Model identifier string.
#         **kwargs: Additional backend-specific parameters.

#     Returns:
#         LLMBackend instance

#     Raises:
#         ValueError: If the backend name is unknown or required params missing.
#     """
#     name = name.lower()
#     if name == "openai":
#         api_key = kwargs.get("api_key")
#         return OpenAIBackend(model_name, api_key=api_key)
#     elif name == "groq":
#         api_key = kwargs.get("api_key")
#         if not api_key:
#             raise ValueError("API key is required for Groq backend")
#         return GroqBackend(model_name, api_key=api_key)
#     elif name == "transformers":
#         return TransformersBackend(model_name)
#     else:
#         raise ValueError(f"Unknown backend: {name}")
