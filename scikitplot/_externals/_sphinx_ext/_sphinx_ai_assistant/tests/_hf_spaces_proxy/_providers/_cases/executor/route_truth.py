from __future__ import annotations

import hashlib
import io

import httpx
import pytest
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.base import ResourceRoute
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import ProviderPrivateResource
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._chat_contract import ChatRequest
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import ResourceDescriptor, VerifiedResource
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload


def _chat(model="gpt-5.4"):
    return ChatRequest(model=model, user_message="q", page_text="", page_descriptor="", max_tokens=100, stream=False, effort=None, thinking=False, budget_tokens=None, resources=())


def _upload(modality="vector_image"):
    data = b"x"
    d = ResourceDescriptor(id="r0", name="x.svg", mime_type="image/svg+xml", size=1, modality=modality, intent="extract")
    v = VerifiedResource(descriptor=d, actual_size=1, sha256=hashlib.sha256(data).hexdigest(), detected_mime="image/svg+xml", detected_modality=modality, signature=modality)
    return ResourceUpload(verified=v, upload=UploadFile(io.BytesIO(data), filename="x.svg"))


def test_enabled_openai_health_filters_unimplemented_extract_route(monkeypatch):
    class E:
        name = "openai"
        enabled = True

        def executable_routes(self, model):
            del model
            return {
                "vector_image": ("tool",),
                "text": ("native",),
                "image": ("native",),
                "document": ("native",),
                "archive": ("tool",),
                "data": ("tool",),
                "binary": ("tool",),
            }

    monkeypatch.setattr(app, "BACKEND_URL", "https://api.openai.com/v1/chat/completions")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "openai")
    app._RESOURCE_EXECUTOR_REGISTRY.register("openai", E())
    try:
        doc = app._resource_capability_doc("gpt-5.4")
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("openai")
    assert doc["execution"] == "enabled"
    assert doc["routes"]["vector_image"] == ["tool"]
    assert "extract" not in doc["routes"]["vector_image"]
    assert doc["routes"]["text"] == ["native", "context"]


def test_enabled_anthropic_health_keeps_model_specific_tool_truth(monkeypatch):
    class E:
        name = "anthropic"
        enabled = True

        def executable_routes(self, model):
            base = {
                "text": ("native",),
                "image": ("native",),
                "animated_image": ("native",),
                "document": ("native",),
            }
            if model == "claude-sonnet-4-6":
                base["archive"] = ("tool",)
            return base

    monkeypatch.setattr(app, "BACKEND_URL", "https://api.anthropic.com/v1/messages")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "anthropic")
    app._RESOURCE_EXECUTOR_REGISTRY.register("anthropic", E())
    try:
        unknown = app._resource_capability_doc("claude-x")
        supported = app._resource_capability_doc("claude-sonnet-4-6")
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("anthropic")
    assert unknown["routes"]["archive"] == ["unsupported"]
    assert supported["routes"]["archive"] == ["tool"]
    assert unknown["routes"]["video"] == ["unsupported"]


class _Adapter:
    async def route_resources(self, model, uploads):
        del model
        return tuple(
            ResourceRoute(
                u.verified.id,
                "extract",
                metadata={
                    "modality": u.verified.detected_modality,
                    "intent": "extract",
                },
            )
            for u in uploads
        )


class _ProviderRegistry:
    def get(self, name):
        del name
        return _Adapter()


class _Executor:
    name = "openai"
    enabled = True

    def __init__(self):
        self.prepare_calls = 0

    def executable_routes(self, model):
        del model
        return {"vector_image": ("tool",)}

    async def prepare(self, *, model, route, upload):
        self.prepare_calls += 1
        return ProviderPrivateResource(
            upload.verified.id,
            route.route,
            "openai",
            upload.verified.sha256,
            upload.verified.actual_size,
            {},
        )

    async def release(self, handle):
        handle.released = True

    async def open_chat(self, *, chat, handles):
        del chat, handles
        return httpx.Response(200, json={})

    async def buffered_chat_response(self, upstream, *, maximum):
        del upstream, maximum
        return {}

    async def chat_sse(self, upstream, *, maximum):
        del upstream, maximum
        yield b""


class _ExecutorRegistry:
    def __init__(self, e):
        self.e = e

    def get(self, name):
        del name
        return self.e


@pytest.mark.asyncio
async def test_server_rejects_unimplemented_route_before_provider_prepare(monkeypatch):
    ex = _Executor()
    monkeypatch.setattr(
        app, "_resource_adapter_name_for_model", lambda _model: "openai"
    )
    monkeypatch.setattr(app, "_RESOURCE_PROVIDER_REGISTRY", _ProviderRegistry())
    monkeypatch.setattr(app, "_RESOURCE_EXECUTOR_REGISTRY", _ExecutorRegistry(ex))
    response = await app._execute_resource_chat(
        chat_req=_chat(), uploads=(_upload(),)
    )
    assert response.status_code in {400, 422, 501, 502}
    assert ex.prepare_calls == 0
    assert b"provider" in response.body.lower()


def test_real_executors_declare_only_routes_they_implement():
    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.openai import (  # type: ignore[import-not-found]
        OpenAIResourceExecutor,
    )
    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.anthropic import (  # type: ignore[import-not-found]
        AnthropicResourceExecutor,
    )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    )
    try:
        o = OpenAIResourceExecutor(api_key="k", client=client)
        a = AnthropicResourceExecutor(api_key="k", client=client)
        assert o.executable_routes("gpt-5.4")["archive"] == ("tool",)
        assert "extract" not in sum(
            (list(v) for v in o.executable_routes("gpt-5.4").values()), []
        )
        assert "archive" not in a.executable_routes("claude-x")
        assert (
            a.executable_routes("claude-sonnet-4-6")["archive"] == ("tool",)
        )
    finally:
        import asyncio
        asyncio.run(client.aclose())
