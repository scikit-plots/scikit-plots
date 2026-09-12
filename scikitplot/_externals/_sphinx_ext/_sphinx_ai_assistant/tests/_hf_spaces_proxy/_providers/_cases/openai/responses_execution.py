from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import httpx
import pytest
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.base import ResourceRoute
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import (
    ProviderPrivateResource,
    ResourceExecutionError,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.openai import (
    OpenAIResourceExecutor,
    official_openai_chat_backend,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._chat_contract import ChatRequest
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import (
    ResourceDescriptor,
    VerifiedResource,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload


def _chat(*, stream: bool, resources=()) -> ChatRequest:
    return ChatRequest(
        model="gpt-5.4",
        user_message="Review the attached resources.",
        page_text="Reference docs only.",
        page_descriptor="Test page",
        max_tokens=1000,
        stream=stream,
        effort="medium",
        thinking=False,
        budget_tokens=None,
        resources=tuple(resources),
    )


def _upload(
    rid: str,
    data: bytes,
    *,
    modality: str,
    mime: str,
    name: str,
    intent: str = "raw",
) -> ResourceUpload:
    desc = ResourceDescriptor(
        id=rid,
        name=name,
        mime_type=mime,
        size=len(data),
        modality=modality,
        intent=intent,
    )
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
        provider="openai",
        source_sha256=v.sha256,
        source_size=v.actual_size,
        opaque={
            "file_id": file_id,
            "modality": v.detected_modality,
            "mime": v.detected_mime,
            "name": v.name,
        },
    )


def test_official_openai_backend_match_is_exact() -> None:
    assert official_openai_chat_backend("https://api.openai.com/v1/chat/completions")
    for bad in (
        "https://api.openai.com/v1/chat/completions/extra",
        "https://api.openai.com/v1/responses",
        "https://api.openai.com.evil.example/v1/chat/completions",
        "http://api.openai.com/v1/chat/completions",
        "https://user@api.openai.com/v1/chat/completions",
        "https://api.openai.com:444/v1/chat/completions",
        "https://api.openai.com/v1/chat/completions?x=1",
    ):
        assert not official_openai_chat_backend(bad)


def test_executor_registration_requires_all_three_authorities(monkeypatch) -> None:
    class FakeOpenAI:
        name = "openai"
        enabled = True

        def executable_routes(self, model):
            del model
            return {
                "text": ("native",), "image": ("native",), "document": ("native",),
                "archive": ("tool",), "data": ("tool",), "vector_image": ("tool",), "binary": ("tool",),
            }

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_client = object()
    monkeypatch.setattr(app, "OpenAIResourceExecutor", FakeOpenAI)
    monkeypatch.setattr(app, "BACKEND_URL", "https://api.openai.com/v1/chat/completions")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "openai")
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "sk-private")
    app._configure_resource_executors(fake_client)
    enabled = app._RESOURCE_EXECUTOR_REGISTRY.get("openai")
    assert enabled.enabled is True
    assert enabled.kwargs["api_key"] == "sk-private"
    assert enabled.kwargs["client"] is fake_client

    for field, value in (
        ("BACKEND_URL", "https://api.openai.com.evil.example/v1/chat/completions"),
        ("BACKEND_RESOURCE_ADAPTER", "custom"),
        ("BACKEND_AUTH_TOKEN", ""),
    ):
        monkeypatch.setattr(app, "BACKEND_URL", "https://api.openai.com/v1/chat/completions")
        monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "openai")
        monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "sk-private")
        monkeypatch.setattr(app, field, value)
        app._configure_resource_executors(fake_client)
        assert app._RESOURCE_EXECUTOR_REGISTRY.execution_state("openai") == "plan-only"


def test_responses_payload_separates_native_image_document_and_tool_archive() -> None:
    img = _upload("img", b"png", modality="image", mime="image/png", name="x.png")
    doc = _upload("doc", b"pdf", modality="document", mime="application/pdf", name="x.pdf")
    arc = _upload("zip", b"zip", modality="archive", mime="application/zip", name="x.zip")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    try:
        ex = OpenAIResourceExecutor(api_key="k", client=client)
        payload = ex.responses_payload(
            _chat(stream=False),
            [
                _handle(img, route="native", file_id="file-img"),
                _handle(doc, route="native", file_id="file-doc"),
                _handle(arc, route="tool", file_id="file-zip"),
            ],
        )
    finally:
        import asyncio
        asyncio.run(client.aclose())
    assert payload["store"] is False
    assert payload["stream"] is False
    content = payload["input"][0]["content"]
    assert any(row.get("type") == "input_image" and row.get("file_id") == "file-img" for row in content)
    assert any(row.get("type") == "input_file" and row.get("file_id") == "file-doc" for row in content)
    assert payload["tools"][0]["type"] == "code_interpreter"
    assert payload["tools"][0]["container"]["file_ids"] == ["file-zip"]
    assert payload["tool_choice"] == "required"
    assert payload["reasoning"] == {"effort": "medium"}


def test_direct_file_limit_is_enforced_before_openai_responses_io() -> None:
    # Avoid allocating 50 MiB: the private handle size is the verified authority
    # consumed by the payload builder after provider upload identity validation.
    tiny = _upload("doc", b"x", modality="document", mime="application/pdf", name="x.pdf")
    h = _handle(tiny, route="native", file_id="file-doc")
    h.source_size = 50 * 1024 * 1024 + 1
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    try:
        ex = OpenAIResourceExecutor(api_key="k", client=client)
        with pytest.raises(ResourceExecutionError, match="50 MiB"):
            ex.responses_payload(_chat(stream=False), [h])
    finally:
        import asyncio
        asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_buffered_responses_bridge_is_bounded_chat_compatible() -> None:
    raw = {
        "id": "resp-private",
        "model": "gpt-5.4",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "answer"}]}
        ],
    }
    response = httpx.Response(200, content=json.dumps(raw).encode())
    ex = OpenAIResourceExecutor(
        api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))
    )
    try:
        result = await ex.buffered_chat_response(response, maximum=4096)
    finally:
        await ex._client.aclose()
    assert result == {
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "model": "gpt-5.4",
    }
    assert "resp-private" not in json.dumps(result)


@pytest.mark.asyncio
async def test_sse_bridge_translates_deltas_and_hides_provider_events() -> None:
    frames = (
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hell"}\n\n'
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"o"}\n\n'
        b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp-secret"}}\n\n'
    )
    response = httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})
    ex = OpenAIResourceExecutor(
        api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))
    )
    try:
        out = b"".join([chunk async for chunk in ex.chat_sse(response, maximum=8192)])
    finally:
        await ex._client.aclose()
    assert b'"content":"Hell"' in out
    assert b'"content":"o"' in out
    assert out.endswith(b"data: [DONE]\n\n")
    assert b"resp-secret" not in out


@dataclass
class _FakeAdapter:
    route: str = "native"

    async def route_resources(self, model, uploads):
        del model
        return tuple(
            ResourceRoute(
                resource_id=u.verified.id,
                route=self.route,
                metadata={"modality": u.verified.detected_modality, "intent": u.verified.intent},
            )
            for u in uploads
        )


class _FakeRegistry:
    def __init__(self, value):
        self.value = value

    def get(self, name):
        del name
        return self.value


class _FakeChatExecutor:
    name = "openai"
    enabled = True

    def executable_routes(self, model):
        del model
        return {"document": ("native",)}

    def __init__(self, *, stream: bool, status: int = 200):
        self.stream = stream
        self.status = status
        self.events: list[str] = []

    async def prepare(self, *, model, route, upload):
        del model
        self.events.append("prepare")
        return ProviderPrivateResource(
            upload.verified.id,
            route.route,
            "openai",
            upload.verified.sha256,
            upload.verified.actual_size,
            {"file_id": "file-private"},
        )

    async def release(self, handle):
        self.events.append("release")
        handle.released = True

    async def open_chat(self, *, chat, handles):
        del chat, handles
        self.events.append("open")
        if self.status != 200:
            return httpx.Response(self.status, content=b"PRIVATE PROVIDER ERROR")
        if self.stream:
            return httpx.Response(200, content=b"unused", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=b'{"ok":true}', headers={"content-type": "application/json"})

    async def buffered_chat_response(self, upstream, *, maximum):
        del upstream, maximum
        self.events.append("buffer")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    async def chat_sse(self, upstream, *, maximum):
        del upstream, maximum
        self.events.append("stream:start")
        try:
            yield b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"two"}}]}\n\n'
        finally:
            self.events.append("stream:closed")


@pytest.mark.asyncio
async def test_common_buffered_execution_releases_provider_before_return(monkeypatch) -> None:
    upload = _upload("r0", b"abc", modality="document", mime="application/pdf", name="x.pdf")
    ex = _FakeChatExecutor(stream=False)
    monkeypatch.setattr(app, "_resource_adapter_name_for_model", lambda model: "openai")
    monkeypatch.setattr(app, "_RESOURCE_PROVIDER_REGISTRY", _FakeRegistry(_FakeAdapter()))
    monkeypatch.setattr(app, "_RESOURCE_EXECUTOR_REGISTRY", _FakeRegistry(ex))
    response = await app._execute_resource_chat(chat_req=_chat(stream=False), uploads=(upload,))
    assert response.status_code == 200
    assert ex.events == ["prepare", "open", "buffer", "release"]
    assert b"file-private" not in response.body


@pytest.mark.asyncio
async def test_common_stream_execution_defers_cleanup_until_stream_closes(monkeypatch) -> None:
    upload = _upload("r0", b"abc", modality="document", mime="application/pdf", name="x.pdf")
    ex = _FakeChatExecutor(stream=True)
    monkeypatch.setattr(app, "_resource_adapter_name_for_model", lambda model: "openai")
    monkeypatch.setattr(app, "_RESOURCE_PROVIDER_REGISTRY", _FakeRegistry(_FakeAdapter()))
    monkeypatch.setattr(app, "_RESOURCE_EXECUTOR_REGISTRY", _FakeRegistry(ex))
    response = await app._execute_resource_chat(chat_req=_chat(stream=True), uploads=(upload,))
    assert response.status_code == 200
    assert ex.events == ["prepare", "open"]
    iterator = response.body_iterator
    first = await anext(iterator)
    assert b"one" in first
    assert "release" not in ex.events
    await iterator.aclose()
    assert ex.events[-2:] == ["stream:closed", "release"]


@pytest.mark.asyncio
async def test_provider_rejection_closes_private_handle_and_hides_body(monkeypatch) -> None:
    upload = _upload("r0", b"abc", modality="document", mime="application/pdf", name="x.pdf")
    ex = _FakeChatExecutor(stream=False, status=403)
    monkeypatch.setattr(app, "_resource_adapter_name_for_model", lambda model: "openai")
    monkeypatch.setattr(app, "_RESOURCE_PROVIDER_REGISTRY", _FakeRegistry(_FakeAdapter()))
    monkeypatch.setattr(app, "_RESOURCE_EXECUTOR_REGISTRY", _FakeRegistry(ex))
    response = await app._execute_resource_chat(chat_req=_chat(stream=False), uploads=(upload,))
    assert response.status_code == 403
    assert ex.events == ["prepare", "open", "release"]
    assert b"PRIVATE PROVIDER ERROR" not in response.body
    assert b"file-private" not in response.body

@pytest.mark.asyncio
async def test_end_to_end_openai_buffered_resource_chain_uploads_executes_deletes(monkeypatch) -> None:
    upload = _upload("r0", b"%PDF-1.7\nhello", modality="document", mime="application/pdf", name="paper.pdf")
    calls: list[tuple[str, str, bytes]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        calls.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file-private-doc", "bytes": upload.verified.actual_size})
        if request.method == "POST" and request.url.path == "/v1/responses":
            doc = json.loads(body)
            assert doc["store"] is False
            assert doc["input"][0]["content"][1] == {"type": "input_file", "file_id": "file-private-doc"}
            return httpx.Response(
                200,
                json={
                    "id": "resp-private",
                    "model": "gpt-5.4",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "reviewed"}]}],
                },
                headers={"content-type": "application/json"},
            )
        if request.method == "DELETE" and request.url.path == "/v1/files/file-private-doc":
            return httpx.Response(200, json={"id": "file-private-doc", "deleted": True})
        raise AssertionError((request.method, request.url.path))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="sk-private", client=client)
        monkeypatch.setattr(app, "_resource_adapter_name_for_model", lambda model: "openai")
        monkeypatch.setattr(app, "_RESOURCE_EXECUTOR_REGISTRY", _FakeRegistry(ex))
        response = await app._execute_resource_chat(chat_req=_chat(stream=False), uploads=(upload,))

    assert response.status_code == 200
    assert json.loads(response.body)["choices"][0]["message"]["content"] == "reviewed"
    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/v1/files"),
        ("POST", "/v1/responses"),
        ("DELETE", "/v1/files/file-private-doc"),
    ]
    rendered = response.body.decode()
    assert "file-private-doc" not in rendered
    assert "resp-private" not in rendered
    assert "sk-private" not in rendered


@pytest.mark.asyncio
async def test_end_to_end_openai_stream_cleanup_happens_after_downstream_close(monkeypatch) -> None:
    upload = _upload("r0", b"%PDF-1.7\nhello", modality="document", mime="application/pdf", name="paper.pdf")
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/files":
            return httpx.Response(200, json={"id": "file-stream-private", "bytes": upload.verified.actual_size})
        if request.method == "POST" and request.url.path == "/v1/responses":
            doc = json.loads(body)
            assert doc["stream"] is True
            frames = (
                b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"one"}\n\n'
                b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"two"}\n\n'
                b'event: response.completed\ndata: {"type":"response.completed"}\n\n'
            )
            return httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})
        if request.method == "DELETE" and request.url.path == "/v1/files/file-stream-private":
            return httpx.Response(200, json={"id": "file-stream-private", "deleted": True})
        raise AssertionError((request.method, request.url.path))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="sk-private", client=client)
        monkeypatch.setattr(app, "_resource_adapter_name_for_model", lambda model: "openai")
        monkeypatch.setattr(app, "_RESOURCE_EXECUTOR_REGISTRY", _FakeRegistry(ex))
        response = await app._execute_resource_chat(chat_req=_chat(stream=True), uploads=(upload,))
        assert calls == [("POST", "/v1/files"), ("POST", "/v1/responses")]
        iterator = response.body_iterator
        first = await anext(iterator)
        assert b'"content":"one"' in first
        assert not any(method == "DELETE" for method, _ in calls)
        await iterator.aclose()
        assert calls[-1] == ("DELETE", "/v1/files/file-stream-private")

@pytest.mark.asyncio
async def test_sse_requires_explicit_completed_terminal_event() -> None:
    frames = b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"partial"}\n\n'
    response = httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        ex = OpenAIResourceExecutor(api_key="k", client=client)
        out = b"".join([chunk async for chunk in ex.chat_sse(response, maximum=8192)])
    assert b'"content":"partial"' in out
    assert b"UPSTREAM_PROTOCOL_ERROR" in out
    assert not out.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_sse_incomplete_terminal_is_error_not_success() -> None:
    frames = b'event: response.incomplete\ndata: {"type":"response.incomplete","response":{"status":"incomplete"}}\n\n'
    response = httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        ex = OpenAIResourceExecutor(api_key="k", client=client)
        out = b"".join([chunk async for chunk in ex.chat_sse(response, maximum=8192)])
    assert b"UPSTREAM_SERVICE_ERROR" in out
    assert b"[DONE]" not in out


def test_buffered_incomplete_response_is_rejected() -> None:
    payload = {"status": "incomplete", "output_text": "partial"}
    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers import openai as openai_mod
    with pytest.raises(ResourceExecutionError):
        openai_mod._response_text(payload)


def test_capability_document_flips_enabled_only_after_authorized_registration(monkeypatch) -> None:
    class FakeOpenAI:
        name = "openai"
        enabled = True

        def executable_routes(self, model):
            del model
            return {
                "text": ("native",), "image": ("native",), "document": ("native",),
                "archive": ("tool",), "data": ("tool",), "vector_image": ("tool",), "binary": ("tool",),
            }

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(app, "OpenAIResourceExecutor", FakeOpenAI)
    monkeypatch.setattr(app, "BACKEND_URL", "https://api.openai.com/v1/chat/completions")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "openai")
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "sk-private")
    app._configure_resource_executors(object())
    try:
        doc = app._resource_capability_doc("gpt-5.4")
        assert doc["adapter"] == "openai"
        assert doc["execution"] == "enabled"
        assert doc["routes"]["document"] == ["native"]
        assert doc["routes"]["archive"] == ["tool"]
        assert doc["routes"]["video"] == ["unsupported"]
        rendered = json.dumps(doc)
        assert "sk-private" not in rendered
        assert "file-" not in rendered
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("openai")
