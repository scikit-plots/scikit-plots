# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Deterministic provider resource capability policies.

The selected *upstream route* owns provider identity.  Model namespaces are not
provider authority: ``openai/gpt-oss-*`` may be served by Hugging Face, Groq,
or another router.  This module therefore accepts an explicit adapter name from
the proxy's routing layer and returns only provider-facility candidates.  A
later executor must still implement the selected route before bytes leave the
proxy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from .base import ProviderResourceCapabilities

_MODALITIES = (
    "text",
    "image",
    "animated_image",
    "vector_image",
    "audio",
    "video",
    "document",
    "archive",
    "data",
    "binary",
)

# Conservative provider-facility candidates.  These are not promises that every
# model supports every route; model-specific deployment overrides narrow or
# replace them.  Unknown provider/model combinations fail closed.
_PROVIDER_DEFAULTS: dict[str, dict[str, tuple[str, ...]]] = {
    "openai": {
        "text": ("native", "context"),
        "image": ("native",),
        "document": ("native",),
        "vector_image": ("tool", "extract"),
        "data": ("tool",),
        "archive": ("tool",),
        "binary": ("tool",),
    },
    "anthropic": {
        "text": ("native", "context"),
        "image": ("native",),
        "animated_image": ("native",),
        "document": ("native",),
        "vector_image": ("tool", "extract"),
        "data": ("tool",),
        "archive": ("tool",),
        "binary": ("tool",),
    },
    "gemini": {
        # File Search-capable Gemini models may transport a text file as a
        # provider-managed retrieval resource; context remains the cheap
        # bounded default when the user does not request raw/tool delivery.
        "text": ("tool", "context"),
        "image": ("native",),
        "animated_image": ("native",),
        "audio": ("native",),
        "video": ("native",),
        "document": ("native",),
        "vector_image": ("extract",),
        "data": ("tool", "extract"),
        "archive": ("tool", "extract"),
    },
    # HF Inference Providers are heterogeneous.  Chat can be LLM or VLM and
    # other media tasks use different task APIs, so the default stays text-only
    # until the deployment declares exact model routes.
    "huggingface": {"text": ("context",)},
    # The bundled Qwen model Space is currently a text-only causal LM.  Future
    # multimodal Spaces opt in through model overrides rather than changing this
    # safe default.
    "self_hosted": {"text": ("context",)},
    "custom": {"text": ("context",)},
}


@dataclass(frozen=True)
class ModelResourceOverride:
    adapter: str
    routes: Mapping[str, tuple[str, ...]]


_HF_REVIEWED_VLM_MODELS = frozenset(
    {
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "zai-org/GLM-4.5V",
        "zai-org/GLM-5.3-Flash",
    }
)


def huggingface_vlm_supported_model(model: str, *, extra_models=()) -> bool:
    """
    Return whether *model* is an exact reviewed/opted-in HF chat VLM.

    Provider suffixes (``:baseten``/``:together``/...) select an Inference
    Provider for the same reviewed model and therefore retain the reviewed base
    model authority.  Deployment additions are exact full model strings.
    """
    raw = str(model or "").strip()
    base = raw.rsplit(":", 1)[0] if ":" in raw else raw
    extras = {str(row).strip() for row in extra_models if str(row).strip()}
    return base in _HF_REVIEWED_VLM_MODELS or raw in extras


_HF_REVIEWED_ASR_MODELS = frozenset({"openai/whisper-large-v3"})


def huggingface_asr_supported_model(model: str, *, extra_models=()) -> bool:
    """
    Return whether *model* is an exact reviewed/opted-in HF ASR task model.

    Unlike Chat/VLM provider suffixes, this executor pins the ``hf-inference``
    ASR task endpoint directly, so ASR authority is an exact model id rather
    than an OpenAI-compatible provider-qualified chat id.
    """
    raw = str(model or "").strip()
    extras = {str(row).strip() for row in extra_models if str(row).strip()}
    return raw in _HF_REVIEWED_ASR_MODELS or raw in extras


def gemini_native_media_supported_model(model: str) -> bool:
    """
    Return whether current Gemini docs list native image/video/audio/PDF input.

    Provider-wide Files storage is not model execution authority.  Keep this
    deliberately narrow so unknown/future model ids fail closed until reviewed.
    """
    raw = str(model or "").strip().lower()
    return raw in {
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3-flash-preview",
    }


def gemini_file_search_supported_model(model: str) -> bool:
    """
    Return whether current Gemini docs list *model* for File Search.

    Files storage and native multimodal input are separate facilities.  File
    Search creates persistent indexed stores and has its own model list, so
    unknown/future models fail closed until that compatibility table is
    deliberately reviewed.
    """
    raw = str(model or "").strip().lower()
    return raw in {
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
    }


def anthropic_code_execution_supported_model(model: str) -> bool:
    """
    Return whether current Claude API docs list *model* for Code Execution.

    The provider Files API may store arbitrary data, but storage is not model
    execution authority.  Unknown/future model families therefore fail closed
    until this policy is deliberately updated.
    """
    raw = str(model or "").strip().lower()
    patterns = (
        r"^claude-(?:opus|sonnet)-5(?:-\d{8})?$",
        r"^claude-(?:fable|mythos)-5(?:-1)?(?:-\d{8})?$",
        r"^claude-opus-4-(?:5|6|7|8)(?:-\d{8})?$",
        r"^claude-sonnet-4-(?:5|6)(?:-\d{8})?$",
        r"^claude-haiku-4-5(?:-\d{8})?$",
    )
    return any(re.match(pattern, raw) for pattern in patterns)


def provider_names() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDER_DEFAULTS))


def default_routes(adapter: str) -> dict[str, tuple[str, ...]]:
    raw = _PROVIDER_DEFAULTS.get(str(adapter or "").strip().lower(), {})
    return {m: tuple(raw.get(m, ("unsupported",))) for m in _MODALITIES}


def capabilities_for(
    *,
    adapter: str,
    model: str,
    overrides: Mapping[str, ModelResourceOverride] | None = None,
    max_files: int | None = None,
    max_file_bytes: int | None = None,
    max_total_bytes: int | None = None,
) -> ProviderResourceCapabilities:
    name = str(adapter or "custom").strip().lower()
    routes = default_routes(name)
    if name == "anthropic" and not anthropic_code_execution_supported_model(model):
        # Native images/text/PDF remain direct.  Tool routes require a Claude
        # family explicitly documented for server-side Code Execution.
        for modality in ("vector_image", "data", "archive", "binary"):
            routes[modality] = tuple(
                route for route in routes.get(modality, ()) if route != "tool"
            ) or ("unsupported",)
    if name == "huggingface":
        if huggingface_asr_supported_model(model):
            # ASR is a separate raw-audio task, not a conversational chat model.
            routes["text"] = ("unsupported",)
            routes["audio"] = ("native",)
        elif huggingface_vlm_supported_model(model):
            # Chat Completion is the reviewed execution surface for conversational
            # VLMs. Other HF media tasks are separate APIs and remain fail-closed.
            routes["image"] = ("native",)
    if name == "gemini":
        if not gemini_native_media_supported_model(model):
            # Files storage and Interactions availability are not model modality
            # authority. Unknown/future Gemini models retain text-context only.
            for modality in ("image", "animated_image", "audio", "video", "document"):
                routes[modality] = ("unsupported",)
        if not gemini_file_search_supported_model(model):
            # File Search is a separate persistent retrieval facility with its
            # own compatibility table. Remove only its tool route; local
            # context/extract candidates remain available to other execution
            # planes when deliberately implemented.
            for modality in ("text", "data", "archive"):
                kept = tuple(
                    route for route in routes.get(modality, ()) if route != "tool"
                )
                routes[modality] = kept or ("unsupported",)
    override = (overrides or {}).get(model)
    if override is not None and str(override.adapter or "").strip().lower() == name:
        # Overrides are provider-scoped.  A model string may be served through
        # multiple upstreams, so one adapter's declaration must never bleed into
        # another adapter merely because the model id matches.
        routes = {
            m: tuple(override.routes.get(m, ("unsupported",))) for m in _MODALITIES
        }
    return ProviderResourceCapabilities(
        provider=name,
        model=model,
        modality_routes=routes,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
