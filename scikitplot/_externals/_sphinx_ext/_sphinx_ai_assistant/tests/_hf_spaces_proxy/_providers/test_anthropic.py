from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import httpx
import pytest
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.anthropic import (
    AnthropicResourceExecutor,
    official_anthropic_messages_backend,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.base import ResourceRoute
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import (
    ProviderPrivateResource,
    ResourceExecutionError,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.policy import (
    anthropic_code_execution_supported_model,
    capabilities_for,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._chat_contract import ChatRequest
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import (
    ResourceDescriptor,
    VerifiedResource,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload


def _chat(*, model: str = "claude-sonnet-4-6", stream: bool = False) -> ChatRequest:
    return ChatRequest(
        model=model,
        user_message="Review the attached resources.",
        page_text="Reference docs only.",
        page_descriptor="Test page",
        max_tokens=1000,
        stream=stream,
        effort="medium",
        thinking=False,
        budget_tokens=None,
        resources=(),
    )


def _upload(rid: str, data: bytes, *, modality: str, mime: str, name: str, intent: str = "raw") -> ResourceUpload:
    desc = ResourceDescriptor(id=rid, name=name, mime_type=mime, size=len(data), modality=modality, intent=intent)
    verified = VerifiedResource(
        descriptor=desc,
        actual_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        detected_mime=mime,
        detected_modality=modality,
        signature=modality,
    )
    return ResourceUpload(verified=verified, upload=UploadFile(io.BytesIO(data), filename=name))


def _handle(upload: ResourceUpload, *, route: str, file_id: str) -> ProviderPrivateResource:
    v = upload.verified
    return ProviderPrivateResource(
        resource_id=v.id,
        route=route,
        provider="anthropic",
        source_sha256=v.sha256,
        source_size=v.actual_size,
        opaque={"file_id": file_id, "modality": v.detected_modality, "mime": v.detected_mime, "name": v.name},
    )


def test_official_anthropic_backend_match_is_exact() -> None:
    assert official_anthropic_messages_backend("https://api.anthropic.com/v1/messages")
    for bad in (
        "https://api.anthropic.com/v1/messages/extra",
        "https://api.anthropic.com.evil.example/v1/messages",
        "http://api.anthropic.com/v1/messages",
        "https://user@api.anthropic.com/v1/messages",
        "https://api.anthropic.com:444/v1/messages",
        "https://api.anthropic.com/v1/messages?x=1",
    ):
        assert not official_anthropic_messages_backend(bad)


def test_anthropic_code_execution_is_model_specific_and_fails_closed() -> None:
    for model in (
        "claude-opus-5",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ):
        assert anthropic_code_execution_supported_model(model)
    for model in ("claude-x", "claude-sonnet-4", "claude-haiku-5", "other/claude-sonnet-5"):
        assert not anthropic_code_execution_supported_model(model)
    unknown = capabilities_for(adapter="anthropic", model="claude-x")
    supported = capabilities_for(adapter="anthropic", model="claude-sonnet-4-6")
    assert unknown.routes_for("archive") == ("unsupported",)
    assert "tool" in supported.routes_for("archive")
    assert unknown.routes_for("video") == ("unsupported",)
    assert supported.routes_for("text")[:2] == ("native", "context")


@pytest.mark.asyncio
async def test_pdf_upload_preserves_bytes_expires_and_deletes() -> None:
    source = b"%PDF-1.7\nraw\x00bytes"
    upload = _upload("pdf", source, modality="document", mime="application/pdf", name="paper.pdf")
    calls: list[tuple[str, str, bytes, dict[str, str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        calls.append((request.method, request.url.path, body, dict(request.headers)))
        if request.method == "POST":
            assert source in body
            assert b'expires_in_seconds"\r\n\r\n3600' in body
            assert b"Content-Type: application/pdf" in body
            assert request.headers["x-api-key"] == "private-key"
            return httpx.Response(200, json={"id": "opaque_future_123", "size_bytes": len(source), "type": "file"})
        return httpx.Response(200, json={"id": "opaque_future_123", "type": "file_deleted"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = AnthropicResourceExecutor(api_key="private-key", client=client)
    route = ResourceRoute("pdf", "native", metadata={"modality": "document", "intent": "raw"})
    try:
        handle = await ex.prepare(model="claude-sonnet-4-6", route=route, upload=upload)
        assert handle.opaque["file_id"] == "opaque_future_123"
        await ex.release(handle)
    finally:
        await client.aclose()
    assert [row[:2] for row in calls] == [("POST", "/v1/files"), ("DELETE", "/v1/files/opaque_future_123")]
    assert "private-key" not in json.dumps(handle.opaque)


@pytest.mark.asyncio
async def test_text_upload_normalizes_mime_without_mutating_bytes() -> None:
    source = "print('✓')\n".encode()
    upload = _upload("txt", source, modality="text", mime="text/x-python", name="x.py")
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        seen["body"] = body
        return httpx.Response(200, json={"id": "id_Future-9", "size_bytes": len(source), "type": "file"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = AnthropicResourceExecutor(api_key="k", client=client)
    try:
        handle = await ex.prepare(
            model="claude-sonnet-4-6",
            route=ResourceRoute("txt", "native"),
            upload=upload,
        )
    finally:
        await client.aclose()
    assert source in seen["body"]
    assert b"Content-Type: text/plain" in seen["body"]
    assert handle.opaque["mime"] == "text/plain"


def test_messages_payload_uses_native_blocks_and_container_upload() -> None:
    pdf = _upload("pdf", b"p", modality="document", mime="application/pdf", name="p.pdf")
    gif = _upload("gif", b"g", modality="animated_image", mime="image/gif", name="g.gif")
    zipf = _upload("zip", b"z", modality="archive", mime="application/zip", name="z.zip")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    try:
        ex = AnthropicResourceExecutor(api_key="k", client=client)
        payload = ex.messages_payload(
            _chat(model="claude-sonnet-4-6"),
            [
                _handle(pdf, route="native", file_id="id-pdf"),
                _handle(gif, route="native", file_id="id-gif"),
                _handle(zipf, route="tool", file_id="id-zip"),
            ],
        )
    finally:
        import asyncio
        asyncio.run(client.aclose())
    blocks = payload["messages"][0]["content"]
    assert {"type": "document", "source": {"type": "file", "file_id": "id-pdf"}} in blocks
    assert {"type": "image", "source": {"type": "file", "file_id": "id-gif"}} in blocks
    assert {"type": "container_upload", "file_id": "id-zip"} in blocks
    assert payload["tools"] == [{"type": "code_execution_20260521", "name": "code_execution"}]


def test_unknown_model_cannot_execute_tool_route_even_with_private_handle() -> None:
    zipf = _upload("zip", b"z", modality="archive", mime="application/zip", name="z.zip")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    try:
        ex = AnthropicResourceExecutor(api_key="k", client=client)
        with pytest.raises(ResourceExecutionError, match="Code Execution"):
            ex.messages_payload(_chat(model="claude-x"), [_handle(zipf, route="tool", file_id="id-zip")])
    finally:
        import asyncio
        asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_500mb_limit_fails_before_provider_io() -> None:
    upload = _upload("r", b"x", modality="document", mime="application/pdf", name="x.pdf")
    upload = ResourceUpload(verified=replace(upload.verified, actual_size=500_000_001), upload=upload.upload)
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"id": "x"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = AnthropicResourceExecutor(api_key="k", client=client)
    try:
        with pytest.raises(ResourceExecutionError, match="500 MB"):
            await ex.prepare(model="claude-sonnet-4-6", route=ResourceRoute("r", "native"), upload=upload)
    finally:
        await client.aclose()
    assert calls == 0


@pytest.mark.asyncio
async def test_cleanup_404_is_idempotent_and_opaque_id_is_future_safe() -> None:
    upload = _upload("r", b"x", modality="document", mime="application/pdf", name="x.pdf")
    handle = _handle(upload, route="native", file_id="Future.ID:123")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(404)))
    ex = AnthropicResourceExecutor(api_key="k", client=client)
    try:
        await ex.release(handle)
    finally:
        await client.aclose()
    assert handle.released is True


@pytest.mark.asyncio
async def test_buffered_and_sse_translation_hides_provider_metadata() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    ex = AnthropicResourceExecutor(api_key="k", client=client)
    buffered = httpx.Response(
        200,
        json={
            "id": "msg-private",
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "thinking", "thinking": "private thought"},
                {"type": "text", "text": "answer"},
            ],
        },
    )
    stream = httpx.Response(
        200,
        content=(
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg-private"}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"secret"}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hell"}}\n\n'
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"o"}}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        ),
        headers={"content-type": "text/event-stream"},
    )
    try:
        result = await ex.buffered_chat_response(buffered, maximum=4096)
        out = b"".join([row async for row in ex.chat_sse(stream, maximum=8192)])
    finally:
        await client.aclose()
    assert result["choices"][0]["message"]["content"] == "answer"
    assert "msg-private" not in json.dumps(result)
    assert b'"content":"Hell"' in out and b'"content":"o"' in out
    assert out.endswith(b"data: [DONE]\n\n")
    assert b"secret" not in out and b"msg-private" not in out


@pytest.mark.asyncio
async def test_sse_requires_message_stop() -> None:
    response = httpx.Response(
        200,
        content=b'event: content_block_delta\ndata: {"delta":{"type":"text_delta","text":"partial"}}\n\n',
        headers={"content-type": "text/event-stream"},
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: response))
    ex = AnthropicResourceExecutor(api_key="k", client=client)
    try:
        out = b"".join([row async for row in ex.chat_sse(response, maximum=8192)])
    finally:
        await client.aclose()
    assert b"UPSTREAM_STREAM_INCOMPLETE" in out
    assert not out.endswith(b"data: [DONE]\n\n")


def test_executor_registration_requires_exact_anthropic_authority(monkeypatch) -> None:
    class FakeAnthropic:
        name = "anthropic"
        enabled = True
        def executable_routes(self, model):
            routes = {"text": ("native",), "image": ("native",), "animated_image": ("native",), "document": ("native",)}
            if model == "claude-sonnet-4-6":
                routes.update({"archive": ("tool",), "data": ("tool",), "vector_image": ("tool",), "binary": ("tool",)})
            return routes
        def __init__(self, **kwargs): self.kwargs = kwargs

    fake_client = object()
    monkeypatch.setattr(app, "AnthropicResourceExecutor", FakeAnthropic)
    monkeypatch.setattr(app, "BACKEND_URL", "https://api.anthropic.com/v1/messages")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "anthropic")
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "anth-private")
    app._configure_resource_executors(fake_client)
    enabled = app._RESOURCE_EXECUTOR_REGISTRY.get("anthropic")
    assert enabled.enabled is True
    assert enabled.kwargs["api_key"] == "anth-private"
    for field, value in (
        ("BACKEND_URL", "https://api.anthropic.com.evil.example/v1/messages"),
        ("BACKEND_RESOURCE_ADAPTER", "custom"),
        ("BACKEND_AUTH_TOKEN", ""),
    ):
        monkeypatch.setattr(app, "BACKEND_URL", "https://api.anthropic.com/v1/messages")
        monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "anthropic")
        monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "anth-private")
        monkeypatch.setattr(app, field, value)
        app._configure_resource_executors(fake_client)
        assert app._RESOURCE_EXECUTOR_REGISTRY.execution_state("anthropic") == "plan-only"


@pytest.mark.asyncio
async def test_text_only_anthropic_open_chat_uses_messages_without_file_ids() -> None:
    seen = {}
    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        seen["body"] = json.loads(body)
        return httpx.Response(200, json={"model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "ok"}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = AnthropicResourceExecutor(api_key="k", client=client)
    try:
        upstream = await ex.open_chat(chat=_chat(), handles=())
        result = await ex.buffered_chat_response(upstream, maximum=8192)
        await upstream.aclose()
    finally:
        await client.aclose()
    assert result["choices"][0]["message"]["content"] == "ok"
    assert seen["body"]["messages"][0]["content"][0]["type"] == "text"
    assert "file_id" not in json.dumps(seen["body"])


def test_health_capability_keeps_execution_and_model_tool_support_separate(monkeypatch) -> None:
    class FakeAnthropic:
        name = "anthropic"
        enabled = True
        def executable_routes(self, model):
            routes = {"text": ("native",), "image": ("native",), "animated_image": ("native",), "document": ("native",)}
            if model == "claude-sonnet-4-6":
                routes.update({"archive": ("tool",), "data": ("tool",), "vector_image": ("tool",), "binary": ("tool",)})
            return routes
        def __init__(self, **kwargs): self.kwargs = kwargs

    monkeypatch.setattr(app, "AnthropicResourceExecutor", FakeAnthropic)
    monkeypatch.setattr(app, "BACKEND_URL", "https://api.anthropic.com/v1/messages")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "anthropic")
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "anth-secret-token")
    app._configure_resource_executors(object())
    try:
        unknown = app._resource_capability_doc("claude-x")
        supported = app._resource_capability_doc("claude-sonnet-4-6")
        assert unknown["adapter"] == "anthropic" and unknown["execution"] == "enabled"
        assert unknown["routes"]["archive"] == ["unsupported"]
        assert supported["routes"]["archive"] == ["tool"]
        assert unknown["routes"]["video"] == ["unsupported"]
        public = json.dumps({"unknown": unknown, "supported": supported})
        assert "anth-secret-token" not in public
        assert "file_id" not in public
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("anthropic")
