# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers import (
    AnthropicAdapter,
    GeminiAdapter,
    HuggingFaceAdapter,
    OpenAIAdapter,
    ProviderRegistry,
    SelfHostedAdapter,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.policy import (
    capabilities_for,
    provider_names,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import (
    ResourceDescriptor,
    VerifiedResource,
)


def _upload(modality: str, intent: str = "auto"):
    desc = ResourceDescriptor(
        id="r0",
        name="x.bin",
        mime_type="application/octet-stream",
        size=10,
        modality=modality,
        intent=intent,
    )
    verified = VerifiedResource(
        descriptor=desc,
        actual_size=10,
        sha256="0" * 64,
        detected_mime="application/octet-stream",
        detected_modality=modality,
        signature="unknown",
    )
    return SimpleNamespace(verified=verified)


def test_registry_contains_all_declared_provider_families() -> None:
    assert set(provider_names()) == {
        "openai", "anthropic", "gemini", "huggingface", "self_hosted", "custom"
    }
    registry = ProviderRegistry()
    assert isinstance(registry.get("openai"), type(registry.get("openai")))
    assert set(registry.names()) == set(provider_names())
    assert OpenAIAdapter().name == "openai"
    assert AnthropicAdapter().name == "anthropic"
    assert GeminiAdapter().name == "gemini"
    assert HuggingFaceAdapter().name == "huggingface"
    assert SelfHostedAdapter().name == "self_hosted"


def test_provider_facility_defaults_are_modality_specific_and_conservative() -> None:
    openai = capabilities_for(adapter="openai", model="gpt-x")
    assert openai.routes_for("image") == ("native",)
    assert openai.routes_for("document") == ("native",)
    assert openai.routes_for("video") == ("unsupported",)

    anthropic = capabilities_for(adapter="anthropic", model="claude-x")
    assert anthropic.routes_for("animated_image") == ("native",)
    assert anthropic.routes_for("archive") == ("unsupported",)
    anthropic_code = capabilities_for(adapter="anthropic", model="claude-sonnet-4-6")
    assert anthropic_code.routes_for("archive") == ("tool",)

    # Files storage is provider-wide, but native media execution is model
    # authority. Unknown Gemini ids fail closed; currently reviewed general
    # multimodal models receive image/audio/video/PDF native routes.
    gemini_unknown = capabilities_for(adapter="gemini", model="gemini-x")
    assert gemini_unknown.routes_for("audio") == ("unsupported",)
    assert gemini_unknown.routes_for("video") == ("unsupported",)
    gemini = capabilities_for(adapter="gemini", model="gemini-3.8-flash")
    assert gemini.routes_for("audio") == ("native",)
    assert gemini.routes_for("video") == ("native",)
    assert gemini.routes_for("archive") == ("tool", "extract")

    # Heterogeneous HF and bundled text-only model routes fail closed unless a
    # deployment later supplies exact model capability overrides.
    hf = capabilities_for(adapter="huggingface", model="openai/gpt-oss-120b")
    assert hf.routes_for("image") == ("unsupported",)
    assert hf.routes_for("text") == ("context",)
    local = capabilities_for(adapter="self_hosted", model="scikit-plots/Qwen2.5-Coder-7B-Instruct")
    assert local.routes_for("video") == ("unsupported",)


@pytest.mark.asyncio
async def test_intent_planner_never_silently_changes_explicit_user_intent() -> None:
    adapter = ProviderRegistry().get("gemini")
    raw = (await adapter.route_resources("gemini-3.8-flash", [_upload("archive", "raw")]))[0]
    assert raw.route == "tool"  # raw may use provider tool, never proxy extraction.
    extract = (await adapter.route_resources("gemini-3.8-flash", [_upload("archive", "extract")]))[0]
    assert extract.route == "extract"
    context = (await adapter.route_resources("gemini-3.8-flash", [_upload("archive", "context")]))[0]
    assert context.route == "unsupported"
    auto = (await adapter.route_resources("gemini-3.8-flash", [_upload("archive", "auto")]))[0]
    assert auto.route == "tool"
    assert auto.metadata["execution"] == "plan-only"


def test_actual_upstream_path_owns_adapter_identity_not_model_namespace(monkeypatch) -> None:
    monkeypatch.setattr(app, "BACKEND_URL", "")
    monkeypatch.setattr(app, "HF_SPACES_MODEL_URL", "https://example-model.hf.space")
    monkeypatch.setattr(app, "HF_SPACES_MODEL_NAMESPACES", ("scikit-plots",))
    assert app._resource_adapter_name_for_model("scikit-plots/Qwen2.5-Coder-7B-Instruct") == "self_hosted"
    # The namespace says OpenAI, but the actual path is HF Inference Providers.
    assert app._resource_adapter_name_for_model("openai/gpt-oss-120b") == "huggingface"

    monkeypatch.setattr(app, "BACKEND_URL", "https://api.example.invalid/v1/chat/completions")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "anthropic")
    assert app._resource_adapter_name_for_model("anything/model") == "anthropic"


def test_health_capability_document_exposes_only_non_secret_route_metadata(monkeypatch) -> None:
    monkeypatch.setattr(app, "BACKEND_URL", "")
    monkeypatch.setattr(app, "HF_SPACES_MODEL_URL", "https://example-model.hf.space")
    monkeypatch.setattr(app, "HF_SPACES_MODEL_NAMESPACES", ("scikit-plots",))
    monkeypatch.setattr(app, "STUB_ENABLED", True)
    caps = app._public_capabilities()["resource_transport"]
    assert caps["version"] == 3
    assert caps["execution"] == "enabled"
    assert caps["routing"] == "server-adapter"
    assert caps["model_capability_endpoint"] == "/v1/resource-capabilities"
    assert caps["models"]
    assert caps["models"]["stub/mirror"]["execution"] == "enabled"
    assert caps["models"]["stub/mirror"]["adapter"] == "stub-mirror"
    rendered = repr(caps).lower()
    assert "token" not in rendered
    assert "authorization" not in rendered
    for model, doc in caps["models"].items():
        if model == "stub/mirror":
            continue
        assert doc["adapter"] in provider_names()
        assert doc["execution"] == "plan-only"
        assert set(doc["routes"]) >= {"text", "image", "audio", "video", "document", "archive"}


def test_model_capability_endpoint_is_on_demand_bounded_and_non_secret(monkeypatch) -> None:
    monkeypatch.setattr(app, "ALLOWED_MODELS", ("model/one",))
    monkeypatch.setattr(app, "HF_SPACES_MODEL_NAMESPACES", ("scikit-plots",))
    with TestClient(app.app) as client:
        allowed = client.get("/v1/resource-capabilities", params={"model": "model/one"})
        assert allowed.status_code == 200
        doc = allowed.json()
        assert doc["model"] == "model/one"
        assert doc["capability"]["execution"] == "plan-only"
        assert "token" not in repr(doc).lower()
        namespace = client.get(
            "/v1/resource-capabilities",
            params={"model": "scikit-plots/Future-VLM"},
        )
        assert namespace.status_code == 200
        denied = client.get("/v1/resource-capabilities", params={"model": "unknown/model"})
        assert denied.status_code == 404
        too_long = client.get("/v1/resource-capabilities", params={"model": "x" * 300})
        assert too_long.status_code == 404
