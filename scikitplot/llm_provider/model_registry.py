# scikitplot/llm_provider/model_registry.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=line-too-long

"""
model_registry.

Text Embedding Models
Embedding models will be used for embedding tasks, specifically, Xenova/gte-small model.

Multi modal model
We currently support IDEFICS (hosted on TGI), OpenAI and Claude 3 as multimodal models.

OpenAI API compatible models
Chat UI can be used with any API server that supports OpenAI API compatibility, for example text-generation-webui, LocalAI, FastChat, llama-cpp-python, and ialacol and vllm.

Cloudflare Workers AI
You can also use Cloudflare Workers AI to run your own models with serverless inference.

Google Vertex models
Chat UI can connect to the google Vertex API endpoints (`List of supported models <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models>`_).

Reasoning Models
ChatUI supports specialized reasoning/Chain-of-Thought (CoT) models through the reasoning configuration field.

Summarizing the Chain of Thought
For models like QwQ, which return a chain of thought but do not explicitly provide a final answer, the summarize type can be used. This automatically summarizes the reasoning steps using the TASK_MODEL (or the first model in the configuration if TASK_MODEL is not specified) and displays the summary as the final answer.

Token-Based Delimitations
For models like DeepSeek R1, token-based delimitations can be used to identify reasoning steps. This is done by specifying the beginToken and endToken fields in the reasoning configuration.
"""

__all__ = [
    "LLM_PROVIDER_CONFIG_MAP",
    "LLM_PROVIDER_ENV_CONNECTOR_MAP",
    "get_config_provider",
]

# API key mapping per provider
LLM_PROVIDER_ENV_CONNECTOR_MAP = {
    "huggingface": "HUGGINGFACE_TOKEN",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "llama": "LLAMA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

# Enhanced LLM Model Registry for Chat Application
# Default model registry used in the chat application.
LLM_PROVIDER_CONFIG_MAP: dict[str, list[dict[str, str]]] = {
    "huggingface": [
        {
            "name": "Command R+ 08-2024",
            "model_id": "CohereLabs/c4ai-command-r-plus-08-2024",
            "version": "2024-08",
            "auth": False,
            "context_window": "",
            "description": (
                "Cohere Labs Command R+ 08-2024 is part of a family of open weight releases from Cohere Labs and Cohere. Our smaller companion model is Cohere Labs Command R"
            ),
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["command", "instruction-tuned", "chat"],
            "docs_url": "https://huggingface.co/CohereLabs/c4ai-command-r-plus-08-2024",
        },
        {
            "name": "Command R7B 12-2024",
            "model_id": "CohereLabs/c4ai-command-r7b-12-2024",
            "version": "2024-12",
            "auth": False,
            "context_window": "",
            "description": (
                "RAG with Command R7B is supported through chat templates in Transformers. The model takes a conversation as input (with an optional user-supplied system preamble), along with a list of document snippets."
            ),
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["command", "instruction-tuned", "chat"],
            "docs_url": "https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024",
        },
        {
            "name": "DeepSeek Coder V2 Lite Instruct",
            "model_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            "version": "2024-10",
            "auth": False,
            "context_window": "16k",
            "description": "Code-generation-focused instruction-tuned model.",
            "capabilities": ["code", "chat"],
            "license": "open",
            "tags": ["developer", "coding"],
            "docs_url": (
                "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
            ),
        },
        {
            "name": "DeepSeek Coder 33b",
            "model_id": "deepseek-ai/deepseek-coder-33b-instruct",
            "version": "2024-08",
            "auth": False,
            "context_window": "16k",
            "description": "Code-generation-focused instruction-tuned model.",
            "capabilities": ["code", "chat"],
            "license": "open",
            "tags": ["developer", "coding"],
            "docs_url": (
                "https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct"
            ),
        },
        {
            "name": "DeepSeek LLM 7B Chat",
            "model_id": "deepseek-ai/deepseek-llm-7b-chat",
            "version": "2024-08",
            "auth": False,
            "context_window": "16k",
            "description": "Chat model fine-tuned on curated instruction datasets.",
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["deepseek", "chat"],
            "docs_url": "https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat",
        },
        {
            "name": "Zephyr 7B Beta",
            "model_id": "HuggingFaceH4/zephyr-7b-beta",
            "version": "2023-10",
            "auth": False,
            "context_window": "16k",
            "description": "Fine-tuned open-source chat model.",
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["open-source", "lightweight"],
            "docs_url": "https://huggingface.co/HuggingFaceH4/zephyr-7b-beta",
        },
        {
            "name": "Mistral 7B Instruct v0.3",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "version": "2023-12",
            "auth": False,
            "context_window": "32k",
            "description": "Instruction-tuned Mistral 7B v0.3 with 32k context window.",
            "capabilities": ["chat", "code"],
            "license": "open",
            "tags": ["mistral", "updated", "32k"],
            "docs_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
        },
        {
            "name": "Phi-3 Mini",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "version": "2024-04",
            "auth": False,
            "context_window": "4k",
            "description": "Small, efficient model from Microsoft, good for edge use.",
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["edge", "tiny"],
            "docs_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
        },
        # {
        #     "name": "Falcon 7B Instruct",
        #     "model_id": "tiiuae/falcon-7b-instruct",
        #     "version": "2023-07",
        #     "auth": False,
        #     "context_window": "8k",
        #     "description": "Falcon 7B fine-tuned for instruction following, good for chat and coding.",
        #     "capabilities": ["chat", "code"],
        #     "license": "open",
        #     "tags": ["instruction-tuned", "chat", "code"],
        #     "docs_url": "https://huggingface.co/tiiuae/falcon-7b-instruct"
        # },
        # {
        #     "name": "OpenLLaMA 7B",
        #     "model_id": "openlm-research/open_llama_7b",
        #     "version": "2023-06",
        #     "auth": False,
        #     "context_window": "4k",
        #     "description": "Open-source 7B LLaMA-based model fine-tuned for general-purpose usage.",
        #     "capabilities": ["chat", "text-generation"],
        #     "license": "open",
        #     "tags": ["open-source", "llama", "general-purpose"],
        #     "docs_url": "https://huggingface.co/openlm-research/open_llama_7b"
        # },
        {
            "name": "TinyLlama Chat",
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "version": "2024-01",
            "auth": False,
            "context_window": "4k",
            "description": "Lightweight and performant small model.",
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["tiny", "fast", "low-resource"],
            "docs_url": "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        },
        {
            "name": "StableVicuna-13B",
            "model_id": "CarperAI/stable-vicuna-13b-delta",
            "version": "2023-03",
            "auth": False,
            "context_window": "4k",
            "description": (
                "StableVicuna-13B is a Vicuna-13B v0 model fine-tuned using reinforcement learning from human feedback (RLHF) via Proximal Policy Optimization (PPO) on various conversational and instructional datasets."
            ),
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["vicuna", "instruction-tuned", "chat"],
            "docs_url": "https://huggingface.co/CarperAI/stable-vicuna-13b-delta",
        },
        {
            "name": "Vicuna 7B v1.5",
            "model_id": "lmsys/vicuna-7b-v1.5",
            "version": "2023-07",
            "auth": False,
            "context_window": "4k",
            "description": (
                "Instruction-tuned version of LLaMA 7B, popular open-source chat model."
            ),
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["vicuna", "instruction-tuned", "chat"],
            "docs_url": "https://huggingface.co/lmsys/vicuna-7b-v1.5",
        },
    ],
    "anthropic": [
        {
            "name": "Claude Haiku 3.5",
            "model_id": "claude-3-5-haiku-20241022",
            "version": "2024-10",
            "auth": True,
            "context_window": "200k",
            "description": "Optimized for speed and low latency.",
            "capabilities": ["chat"],
            "license": "commercial",
            "tags": ["fast"],
            "docs_url": "https://docs.anthropic.com/claude/docs/models-overview",
        },
        {
            "name": "Claude Opus 4",
            "model_id": "claude-opus-4-20250514",
            "version": "2025-05",
            "auth": True,
            "context_window": "200k",
            "description": "Most powerful Claude model for advanced reasoning.",
            "capabilities": ["chat", "reasoning"],
            "license": "commercial",
            "tags": ["premium"],
            "docs_url": "https://docs.anthropic.com/claude/docs/models-overview",
        },
        {
            "name": "Claude Sonnet 4",
            "model_id": "claude-sonnet-4-20250514",
            "version": "2025-05",
            "auth": True,
            "context_window": "200k",
            "description": "Balanced performance and speed.",
            "capabilities": ["chat"],
            "license": "commercial",
            "tags": ["balanced"],
            "docs_url": "https://docs.anthropic.com/claude/docs/models-overview",
        },
    ],
    "cohere": [
        {
            "name": "Command R",
            "model_id": "command-r-08-2024",
            "version": "2024-08",
            "auth": True,
            "context_window": "128k",
            "description": (
                "Command R is an instruction-following conversational model that performs language tasks at a higher quality, more reliably, and with a longer context than previous models."
            ),
            "capabilities": ["chat", "rag"],
            "license": "open",
            "tags": ["rag", "enhanced"],
            "docs_url": "https://docs.cohere.com/v2/docs/command-r",
        },
        {
            "name": "Command R+",
            "model_id": "command-r-plus-08-2024",
            "version": "2024-08",
            "auth": True,
            "context_window": "128k",
            "description": (
                "Enhanced version with stronger RAG and reasoning capabilities."
            ),
            "capabilities": ["chat", "rag"],
            "license": "open",
            "tags": ["rag", "enhanced"],
            "docs_url": "https://docs.cohere.com/v2/docs/command-r-plus",
        },
        {
            "name": "Command R7B",
            "model_id": "command-r7b-12-2024",
            "version": "2024-12",
            "auth": True,
            "context_window": "",
            "description": "Open weights RAG-tuned model.",
            "capabilities": ["chat", "rag"],
            "license": "open",
            "tags": ["rag", "open-weights"],
            "docs_url": "https://docs.cohere.com/v2/docs/command-r7b",
        },
    ],
    "deepseek": [
        {
            "name": "DeepSeek-V3",
            "model_id": "deepseek-chat",
            "version": "2023-12",
            "auth": False,
            "context_window": "16k",
            "description": "Code-generation-focused instruction-tuned model.",
            "capabilities": ["code", "chat"],
            "license": "open",
            "tags": ["developer", "coding"],
            "docs_url": "https://api-docs.deepseek.com",
        },
        {
            "name": "DeepSeek-R1",
            "model_id": "deepseek-reasoner",
            "version": "2024-03",
            "auth": False,
            "context_window": "16k",
            "description": "Chat model fine-tuned on curated instruction datasets.",
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["deepseek", "chat"],
            "docs_url": "https://api-docs.deepseek.com",
        },
    ],
    # pip install google-genai
    "gemini": [
        {
            "name": "Gemini 2.5 Flash Preview 05-20",
            "model_id": "gemini-2.5-flash-preview-05-20",
            "version": "2025-05",
            "auth": True,
            "context_window": "32k",
            "description": (
                "Our best model in terms of price-performance, offering well-rounded capabilities. Gemini 2.5 Flash rate limits are more restricted since it is an experimental / preview model."
            ),
            "capabilities": ["chat", "text"],
            "license": "commercial",
            "tags": ["google"],
            "docs_url": (
                "https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview"
            ),
        },
        {
            "name": "Gemini 2.0 Flash",
            "model_id": "gemini-2.0-flash",
            "version": "2025-02",
            "auth": True,
            "context_window": "32k",
            "description": (
                "Gemini 2.0 Flash delivers next-gen features and improved capabilities, including superior speed, native tool use, and a 1M token context window."
            ),
            "capabilities": ["chat", "text"],
            "license": "commercial",
            "tags": ["google"],
            "docs_url": "https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash",
        },
    ],
    "groq": [
        {
            "name": "LLaMA 3 8B @ Groq",
            "model_id": "llama3-8b-8192",
            "version": "2024-04",
            "auth": True,
            "context_window": "8k",
            "description": "Meta's LLaMA 3 (8B) on Groq's ultra-fast hardware.",
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["llama", "fast"],
            "docs_url": "https://groq.com",
        },
        {
            "name": "Mixtral 8x7B @ Groq",
            "model_id": "mixtral-8x7b-32768",
            "version": "2024-03",
            "auth": True,
            "context_window": "32k",
            "description": "Mixture of Experts model with high performance on Groq.",
            "capabilities": ["chat", "code"],
            "license": "open",
            "tags": ["moe", "performance"],
            "docs_url": "https://groq.com",
        },
    ],
    "llama": [
        {
            "name": "LLaMA 3 8B Chat",
            "model_id": "meta-llama/Llama-3-8b-chat-hf",
            "version": "2024-04",
            "auth": False,
            "context_window": "8k",
            "description": "LLaMA 3 8B chat model, strong open-source performance.",
            "capabilities": ["chat"],
            "license": "open",
            "tags": ["llama", "8b"],
            "docs_url": "https://huggingface.co/meta-llama/Llama-3-8b-chat-hf",
        },
        {
            "name": "LLaMA 3 70B Chat",
            "model_id": "meta-llama/Llama-3-70b-chat-hf",
            "version": "2024-04",
            "auth": False,
            "context_window": "8k",
            "description": "70B SOTA open weights model for enterprise use.",
            "capabilities": ["chat", "code"],
            "license": "open",
            "tags": ["llama", "70b"],
            "docs_url": "https://huggingface.co/meta-llama/Llama-3-70b-chat-hf",
        },
    ],
    "mistral": [
        {
            "name": "Mixtral 8x7B Instruct",
            "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "version": "2023-12",
            "auth": False,
            "context_window": "32k",
            "description": (
                "Mixture of Experts model from Mistral optimized for instruction following."
            ),
            "capabilities": ["chat", "code"],
            "license": "open",
            "tags": ["moe", "open-source", "high-performance"],
            "docs_url": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
        },
    ],
    "openai": [
        {
            "name": "GPT-4o",
            "model_id": "gpt-4o",
            "version": "2024-05",
            "auth": True,
            "context_window": "128k",
            "description": (
                "Multimodal GPT-4 with strong reasoning and real-time vision/audio."
            ),
            "capabilities": ["chat", "code", "vision"],
            "license": "commercial",
            "tags": ["premium", "multimodal"],
            "docs_url": "https://platform.openai.com/docs/models/gpt-4o",
        },
        {
            "name": "GPT-3.5 Turbo",
            "model_id": "gpt-3.5-turbo",
            "version": "2023-11",
            "auth": True,
            "context_window": "16k",
            "description": "Fast and cost-effective LLM from OpenAI.",
            "capabilities": ["chat", "code"],
            "license": "commercial",
            "tags": ["budget"],
            "docs_url": "https://platform.openai.com/docs/models/gpt-3-5",
        },
    ],
}


def get_config_provider() -> dict[str, list[dict[str, str]]]:
    """
    Return a dictionary of default chat model providers and their models.

    This dictionary is used to initialize or populate model selection options
    in chat applications. The structure maps each provider name to a list of
    available models and corresponding API keys.

    Each model entry includes:
    - `model_id`: Identifier or path used by the client to reference the model.
    - `api_key`: API token required to authenticate access (can be injected at runtime).

    Returns
    -------
    Dict[str, List[Dict[str, str]]]
        Mapping of provider names to a list of model configurations.

    Examples
    --------
    >>> models = get_config_provider()
    >>> models["openai"]
    [{'model_id': 'gpt-4-turbo', 'api_key': ''}, ...]

    Notes
    -----
    This list is non-exhaustive and can be extended with custom or local models.
    Ensure API keys are securely loaded (e.g., from environment variables or secrets).
    """
    return LLM_PROVIDER_CONFIG_MAP, LLM_PROVIDER_ENV_CONNECTOR_MAP
