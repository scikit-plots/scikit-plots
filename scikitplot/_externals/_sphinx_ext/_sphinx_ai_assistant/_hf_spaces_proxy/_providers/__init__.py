# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Provider adapters for first-class chat resources."""

from .anthropic import AnthropicAdapter, AnthropicResourceExecutor
from .artifact_output import (
    GeneratedProviderArtifact,
    OpenAIProviderArtifactOutputExecutor,
    ProviderArtifactOutputExecutor,
    ProviderArtifactOutputRegistry,
    StubProviderArtifactOutputExecutor,
)
from .base import ProviderAdapter, ProviderResourceCapabilities, ResourceRoute
from .custom import CustomAdapter
from .executor import (
    PendingProviderExecutor,
    ProviderChatExecutor,
    ProviderExecutorRegistry,
    ProviderPrivateResource,
    ProviderResourceExecutor,
    ResourceExecutionError,
    ResourceExecutionReceipt,
    ResourceExecutionSession,
)
from .gemini import GeminiAdapter, GeminiResourceExecutor
from .huggingface import HuggingFaceAdapter, HuggingFaceResourceExecutor
from .openai import OpenAIAdapter, OpenAIResourceExecutor
from .registry import ProviderRegistry, StaticProviderAdapter
from .self_hosted import SelfHostedAdapter

__all__ = [
    "AnthropicAdapter",
    "AnthropicResourceExecutor",
    "CustomAdapter",
    "GeminiAdapter",
    "GeminiResourceExecutor",
    "GeneratedProviderArtifact",
    "HuggingFaceAdapter",
    "HuggingFaceResourceExecutor",
    "OpenAIAdapter",
    "OpenAIProviderArtifactOutputExecutor",
    "OpenAIResourceExecutor",
    "PendingProviderExecutor",
    "ProviderAdapter",
    "ProviderArtifactOutputExecutor",
    "ProviderArtifactOutputRegistry",
    "ProviderChatExecutor",
    "ProviderExecutorRegistry",
    "ProviderPrivateResource",
    "ProviderRegistry",
    "ProviderResourceCapabilities",
    "ProviderResourceExecutor",
    "ResourceExecutionError",
    "ResourceExecutionReceipt",
    "ResourceExecutionSession",
    "ResourceRoute",
    "SelfHostedAdapter",
    "StaticProviderAdapter",
    "StubProviderArtifactOutputExecutor",
]
