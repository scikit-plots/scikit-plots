from __future__ import annotations

import base64
import hashlib
import io
import json

import httpx
import pytest
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.base import ResourceRoute
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.huggingface import (
    HuggingFaceResourceExecutor,
    official_huggingface_router_base,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.policy import (
    ModelResourceOverride,
    capabilities_for,
    huggingface_vlm_supported_model,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._chat_contract import ChatRequest
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import ResourceDescriptor, VerifiedResource
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload


def _upload(data: bytes, *, mime="image/png", modality="image", name="x.png", rid="img") -> ResourceUpload:
    desc = ResourceDescriptor(rid, name, mime, len(data), modality, "raw")
    verified = VerifiedResource(desc, len(data), hashlib.sha256(data).hexdigest(), mime, modality, "test")
    return ResourceUpload(verified, UploadFile(io.BytesIO(data), filename=name, headers={"content-type": mime}))


def _chat(model="zai-org/GLM-4.5V", *, stream=False) -> ChatRequest:
    return ChatRequest(
        model=model,
        user_message="describe",
        page_text="ctx",
        page_descriptor="page",
        max_tokens=256,
        stream=stream,
        effort=None,
        thinking=False,
        budget_tokens=None,
        resources=(),
    )


def test_hf_router_destination_and_reviewed_vlm_authority_are_exact() -> None:
    assert official_huggingface_router_base("https://router.huggingface.co")
    assert official_huggingface_router_base("https://router.huggingface.co/")
    for bad in (
        "http://router.huggingface.co",
        "https://router.huggingface.co.evil.example",
        "https://user@router.huggingface.co",
        "https://router.huggingface.co/v1",
        "https://router.huggingface.co?x=1",
    ):
        assert not official_huggingface_router_base(bad)
    assert huggingface_vlm_supported_model("Qwen/Qwen2.5-VL-3B-Instruct")
    assert huggingface_vlm_supported_model("zai-org/GLM-4.5V:baseten")
    assert not huggingface_vlm_supported_model("openai/gpt-oss-120b")


def test_provider_override_isolation_and_exact_hf_opt_in() -> None:
    override = ModelResourceOverride(adapter="huggingface", routes={"text": ("context",), "image": ("native",)})
    hf = capabilities_for(adapter="huggingface", model="org/future-vlm", overrides={"org/future-vlm": override})
    assert hf.routes_for("image") == ("native",)
    # Same model string through another upstream must keep that upstream's policy.
    oa = capabilities_for(adapter="openai", model="org/future-vlm", overrides={"org/future-vlm": override})
    assert oa.routes_for("document") == ("native",)
    assert oa.routes_for("image") == ("native",)


@pytest.mark.asyncio
async def test_prepare_preserves_image_bytes_and_release_revokes_private_data_url() -> None:
    raw = b"\x89PNG\r\n\x1a\nraw-image"
    upload = _upload(raw)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _req: httpx.Response(200)))
    ex = HuggingFaceResourceExecutor(api_key="hf_private", client=client)
    try:
        handle = await ex.prepare(model="zai-org/GLM-4.5V", route=ResourceRoute("img", "native"), upload=upload)
        encoded = handle.opaque["data_url"].split(",", 1)[1]
        assert base64.b64decode(encoded) == raw
        assert "hf_private" not in repr(handle.opaque)
        await ex.release(handle)
        assert handle.released and handle.opaque == {}
    finally:
        await client.aclose()
        await upload.close()


@pytest.mark.asyncio
async def test_prepare_rejects_non_vlm_and_non_raster_resources() -> None:
    upload = _upload(b"gif", mime="image/gif", modality="animated_image", name="x.gif")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _req: httpx.Response(200)))
    ex = HuggingFaceResourceExecutor(api_key="k", client=client)
    try:
        with pytest.raises(Exception, match="unsupported"):
            await ex.prepare(model="openai/gpt-oss-120b", route=ResourceRoute("img", "native"), upload=upload)
        with pytest.raises(Exception, match="unsupported"):
            await ex.prepare(model="zai-org/GLM-4.5V", route=ResourceRoute("img", "native"), upload=upload)
    finally:
        await client.aclose()
        await upload.close()


@pytest.mark.asyncio
async def test_chat_request_uses_official_router_and_private_image_url_only_inside_payload() -> None:
    raw = b"\xff\xd8\xffraw-jpeg"
    upload = _upload(raw, mime="image/jpeg", name="x.jpg")
    seen = {}
    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads((await request.aread()).decode())
        return httpx.Response(200, json={"model": "zai-org/GLM-4.5V", "choices": [{"message": {"role": "assistant", "content": "ok"}}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = HuggingFaceResourceExecutor(api_key="hf_private", client=client)
    try:
        handle = await ex.prepare(model="zai-org/GLM-4.5V", route=ResourceRoute("img", "native"), upload=upload)
        upstream = await ex.open_chat(chat=_chat(), handles=[handle])
        out = await ex.buffered_chat_response(upstream, maximum=4096)
        assert out["choices"][0]["message"]["content"] == "ok"
        assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"
        assert seen["auth"] == "Bearer hf_private"
        content = seen["body"]["messages"][1]["content"]
        assert content[0]["type"] == "text" and content[1]["type"] == "image_url"
        assert base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]) == raw
    finally:
        await client.aclose()
        await upload.close()


@pytest.mark.asyncio
async def test_text_only_hf_chat_uses_same_executor_without_raw_resource_claims() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads((await request.aread()).decode())
        assert isinstance(body["messages"][1]["content"], str)
        return httpx.Response(200, json={"choices": [{"message": {"content": "text-ok"}}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = HuggingFaceResourceExecutor(api_key="k", client=client)
    try:
        response = await ex.open_chat(chat=_chat(model="openai/gpt-oss-120b"), handles=[])
        out = await ex.buffered_chat_response(response, maximum=4096)
        assert out["choices"][0]["message"]["content"] == "text-ok"
        assert ex.executable_routes("openai/gpt-oss-120b") == {}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_sse_bridge_only_exposes_content_and_requires_done() -> None:
    frames = (
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    ex = HuggingFaceResourceExecutor(api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200))))
    try:
        response = httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})
        out = b"".join([chunk async for chunk in ex.chat_sse(response, maximum=8192)])
        assert b'"content":"Hi"' in out and out.endswith(b"data: [DONE]\n\n")
        incomplete = httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
        bad = b"".join([chunk async for chunk in ex.chat_sse(incomplete, maximum=8192)])
        assert b"UPSTREAM_PROTOCOL_ERROR" in bad and b"[DONE]" not in bad
    finally:
        await ex._client.aclose()


def test_app_hf_executor_configuration_is_pinned_to_path3_router(monkeypatch) -> None:
    class FakeExecutor:
        name = "huggingface"
        enabled = True
        def __init__(self, **kwargs): self.kwargs = kwargs
        def executable_routes(self, model): return {"image": ("native",)}
    monkeypatch.setattr(app, "HuggingFaceResourceExecutor", FakeExecutor)
    monkeypatch.setattr(app, "BACKEND_URL", "")
    monkeypatch.setattr(app, "HF_TOKEN", "hf_private")
    monkeypatch.setattr(app, "HF_BASE", "https://router.huggingface.co")
    monkeypatch.setattr(app, "HF_RESOURCE_VLM_MODELS", ("org/future-vlm",))
    try:
        app._configure_resource_executors(object())
        assert app._RESOURCE_EXECUTOR_REGISTRY.execution_state("huggingface") == "enabled"
        ex = app._RESOURCE_EXECUTOR_REGISTRY.get("huggingface")
        assert ex.kwargs["api_key"] == "hf_private"
        assert ex.kwargs["vlm_models"] == ("org/future-vlm",)
        monkeypatch.setattr(app, "HF_BASE", "https://router.huggingface.co.evil.example")
        app._configure_resource_executors(object())
        assert app._RESOURCE_EXECUTOR_REGISTRY.execution_state("huggingface") == "plan-only"
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("huggingface")


def test_route_constraints_are_small_and_do_not_claim_other_media() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    ex = HuggingFaceResourceExecutor(api_key="k", client=client, vlm_models=("org/future-vlm",))
    try:
        assert ex.executable_routes("org/future-vlm") == {"image": ("native",)}
        doc = ex.route_constraints("org/future-vlm")
        assert doc["image"]["native"]["max_file_bytes"] == 8 * 1024 * 1024
        assert set(doc["image"]["native"]["mime_types"]) == {"image/jpeg", "image/png", "image/webp"}
        assert not any(key in doc for key in ("audio", "video", "document", "archive", "data"))
    finally:
        import asyncio
        asyncio.run(client.aclose())


def test_health_intersects_hf_execution_with_exact_vlm_routes(monkeypatch) -> None:
    monkeypatch.setattr(app, "BACKEND_URL", "")
    monkeypatch.setattr(app, "HF_SPACES_MODEL_URL", "")
    monkeypatch.setattr(app, "HF_TOKEN", "hf_private")
    monkeypatch.setattr(app, "HF_BASE", "https://router.huggingface.co")
    app._configure_resource_executors(object())
    try:
        vlm = app._resource_capability_doc("zai-org/GLM-4.5V")
        assert vlm["adapter"] == "huggingface" and vlm["execution"] == "enabled"
        assert vlm["routes"]["image"] == ["native"]
        assert vlm["route_constraints"]["image"]["native"]["max_file_bytes"] == 8 * 1024 * 1024
        assert vlm["routes"]["video"] == ["unsupported"]
        text_model = app._resource_capability_doc("openai/gpt-oss-120b")
        assert text_model["execution"] == "enabled"
        assert text_model["routes"]["text"] == ["context"]
        assert text_model["routes"]["image"] == ["unsupported"]
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("huggingface")
