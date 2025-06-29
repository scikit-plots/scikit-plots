.. _llm_provider-index:

======================================================================
LLM Provider (experimental)
======================================================================

This module contains functions related to :py:mod:`~.llm_provider` (Large Language Models).

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

.. seealso::

   * https://huggingface.co/scikit-plots
   * https://huggingface.co/spaces/scikit-plots/model-sight
