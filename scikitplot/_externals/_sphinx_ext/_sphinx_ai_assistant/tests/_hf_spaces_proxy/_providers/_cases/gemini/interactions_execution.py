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
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import ResourceExecutionError
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.gemini import (
    GeminiResourceExecutor,
    official_gemini_interactions_backend,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.policy import (
    capabilities_for,
    gemini_native_media_supported_model,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._chat_contract import ChatRequest
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import ResourceDescriptor, VerifiedResource
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload


def _chat(*, model="gemini-3.8-flash", stream=False):
    return ChatRequest(
        model=model,
        user_message="Describe the resource.",
        page_text="Reference docs only.",
        page_descriptor="Test page",
        max_tokens=1000,
        stream=stream,
        effort="medium",
        thinking=False,
        budget_tokens=None,
        resources=(),
    )


def _upload(rid: str, data: bytes, *, modality: str, mime: str, name: str) -> ResourceUpload:
    desc = ResourceDescriptor(id=rid, name=name, mime_type=mime, size=len(data), modality=modality, intent="raw")
    verified = VerifiedResource(
        descriptor=desc,
        actual_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        detected_mime=mime,
        detected_modality=modality,
        signature=modality,
    )
    return ResourceUpload(verified=verified, upload=UploadFile(io.BytesIO(data), filename=name))


def _gemini_meta(upload: ResourceUpload, *, state="ACTIVE", name="files/gem-test") -> dict:
    v = upload.verified
    return {
        "file": {
            "name": name,
            "uri": f"https://generativelanguage.googleapis.com/v1beta/{name}",
            "mimeType": v.detected_mime,
            "sizeBytes": str(v.actual_size),
            "sha256Hash": base64.b64encode(bytes.fromhex(v.sha256)).decode(),
            "state": state,
        }
    }


def test_backend_and_model_authority_fail_closed() -> None:
    assert official_gemini_interactions_backend("https://generativelanguage.googleapis.com/v1beta/interactions")
    for bad in (
        "http://generativelanguage.googleapis.com/v1beta/interactions",
        "https://generativelanguage.googleapis.com.evil.example/v1beta/interactions",
        "https://user@generativelanguage.googleapis.com/v1beta/interactions",
        "https://generativelanguage.googleapis.com/v1beta/interactions?key=x",
        "https://generativelanguage.googleapis.com/v1beta/interactions/extra",
    ):
        assert not official_gemini_interactions_backend(bad)
    for model in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview"):
        assert gemini_native_media_supported_model(model)
    assert not gemini_native_media_supported_model("gemini-future")
    assert capabilities_for(adapter="gemini", model="gemini-future").routes_for("video") == ("unsupported",)


@pytest.mark.asyncio
async def test_resumable_upload_preserves_raw_bytes_hashes_and_deletes() -> None:
    source = b"\x00\x00\x00\x18ftypmp42raw-video"
    upload = _upload("vid", source, modality="video", mime="video/mp4", name="clip.mp4")
    calls: list[tuple[str, str, bytes]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        calls.append((request.method, request.url.path, body))
        assert request.headers.get("x-goog-api-key") == "private-key" or request.url.path.startswith("/upload/v1beta/files")
        if request.url.path == "/upload/v1beta/files" and request.headers.get("x-goog-upload-command") == "start":
            return httpx.Response(200, headers={"x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=opaque"})
        if request.url.path == "/upload/v1beta/files" and request.headers.get("x-goog-upload-command") == "upload, finalize":
            assert body == source
            return httpx.Response(200, json=_gemini_meta(upload))
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        raise AssertionError((request.method, request.url))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="private-key", client=client)
    try:
        handle = await ex.prepare(model="gemini-3.8-flash", route=ResourceRoute("vid", "native"), upload=upload)
        assert handle.opaque["uri"].endswith("/files/gem-test")
        assert "private-key" not in json.dumps(handle.opaque)
        await ex.release(handle)
    finally:
        await client.aclose()
    assert any(method == "DELETE" for method, _path, _body in calls)


@pytest.mark.asyncio
async def test_processing_poll_becomes_active() -> None:
    upload = _upload("aud", b"ID3audio", modality="audio", mime="audio/mp3", name="a.mp3")
    polls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.url.path == "/upload/v1beta/files" and request.headers.get("x-goog-upload-command") == "start":
            return httpx.Response(200, headers={"x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=x"})
        if request.url.path == "/upload/v1beta/files":
            return httpx.Response(200, json=_gemini_meta(upload, state="PROCESSING"))
        if request.method == "GET":
            polls += 1
            return httpx.Response(200, json=_gemini_meta(upload, state="ACTIVE"))
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client, poll_interval_seconds=0.01)
    try:
        handle = await ex.prepare(model="gemini-3.8-flash", route=ResourceRoute("aud", "native"), upload=upload)
    finally:
        await client.aclose()
    assert polls == 1
    assert handle.opaque["modality"] == "audio"


@pytest.mark.asyncio
async def test_hash_mismatch_rolls_back_provider_file() -> None:
    upload = _upload("pdf", b"%PDF-1.7\nhello", modality="document", mime="application/pdf", name="p.pdf")
    deleted = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal deleted
        if request.url.path == "/upload/v1beta/files" and request.headers.get("x-goog-upload-command") == "start":
            return httpx.Response(200, headers={"x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=x"})
        if request.url.path == "/upload/v1beta/files":
            bad = _gemini_meta(upload)
            bad["file"]["sha256Hash"] = base64.b64encode(b"x" * 32).decode()
            return httpx.Response(200, json=bad)
        if request.method == "DELETE":
            deleted = True
            return httpx.Response(200, json={})
        raise AssertionError(request.url)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client)
    try:
        with pytest.raises(ResourceExecutionError, match="SHA-256"):
            await ex.prepare(model="gemini-3.8-flash", route=ResourceRoute("pdf", "native"), upload=upload)
    finally:
        await client.aclose()
    assert deleted


def test_interactions_payload_maps_media_and_hides_provider_capabilities() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    ex = GeminiResourceExecutor(api_key="k", client=client)
    uploads = [
        _upload("img", b"x", modality="image", mime="image/png", name="x.png"),
        _upload("gif", b"x", modality="animated_image", mime="image/gif", name="x.gif"),
        _upload("aud", b"x", modality="audio", mime="audio/mp3", name="x.mp3"),
        _upload("vid", b"x", modality="video", mime="video/mp4", name="x.mp4"),
        _upload("pdf", b"x", modality="document", mime="application/pdf", name="x.pdf"),
    ]
    handles = []
    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import ProviderPrivateResource
    for u in uploads:
        v = u.verified
        handles.append(ProviderPrivateResource(v.id, "native", "gemini", v.sha256, v.actual_size, {
            "file_name": f"files/{v.id}",
            "uri": f"https://generativelanguage.googleapis.com/v1beta/files/{v.id}",
            "modality": v.detected_modality,
            "mime": v.detected_mime,
            "name": v.name,
        }))
    try:
        payload = ex.interactions_payload(_chat(), handles)
    finally:
        import asyncio
        asyncio.run(client.aclose())
    assert payload["store"] is False
    assert payload["generation_config"]["max_output_tokens"] == 1000
    assert payload["generation_config"]["thinking_level"] == "medium"
    types = [row["type"] for row in payload["input"]]
    assert types == ["text", "image", "image", "audio", "video", "document"]
    assert "private-key" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_buffered_and_sse_bridges_only_expose_model_text() -> None:
    ex = GeminiResourceExecutor(api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200))))
    try:
        buffered = httpx.Response(200, content=json.dumps({
            "id": "private-interaction",
            "model": "gemini-3.8-flash",
            "status": "completed",
            "steps": [{"type": "thought", "summary": "secret"}, {"type": "model_output", "content": [{"type": "text", "text": "answer"}]}],
        }).encode())
        result = await ex.buffered_chat_response(buffered, maximum=4096)
        assert result["choices"][0]["message"]["content"] == "answer"
        assert "private-interaction" not in json.dumps(result)
        frames = (
            b'event: step.delta\ndata: {"event_type":"step.delta","delta":{"type":"thought_signature","signature":"SECRET"},"index":0}\n\n'
            b'event: step.delta\ndata: {"event_type":"step.delta","delta":{"type":"text","text":"Hell"},"index":1}\n\n'
            b'event: interaction.completed\ndata: {"event_type":"interaction.completed","interaction":{"id":"private","status":"completed"}}\n\n'
        )
        response = httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})
        out = b"".join([part async for part in ex.chat_sse(response, maximum=8192)])
        assert b'"content":"Hell"' in out
        assert b"SECRET" not in out and b"private" not in out
        assert out.endswith(b"data: [DONE]\n\n")
    finally:
        await ex._client.aclose()


@pytest.mark.asyncio
async def test_incomplete_stream_fails_closed() -> None:
    frames = b'event: step.delta\ndata: {"event_type":"step.delta","delta":{"type":"text","text":"partial"},"index":1}\n\n'
    response = httpx.Response(200, content=frames, headers={"content-type": "text/event-stream"})
    ex = GeminiResourceExecutor(api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(lambda req: response)))
    try:
        out = b"".join([part async for part in ex.chat_sse(response, maximum=4096)])
    finally:
        await ex._client.aclose()
    assert b"UPSTREAM_STREAM_INCOMPLETE" in out
    assert b"[DONE]" not in out


def test_app_configuration_requires_exact_official_gemini_backend(monkeypatch) -> None:
    class FakeExecutor:
        name = "gemini"
        enabled = True
        def __init__(self, **kwargs): self.kwargs = kwargs
        def executable_routes(self, model): return {"video": ("native",)}
    fake_client = object()
    monkeypatch.setattr(app, "GeminiResourceExecutor", FakeExecutor)
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "gemini")
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "private-key")
    monkeypatch.setattr(app, "BACKEND_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
    app._configure_resource_executors(fake_client)
    assert app._RESOURCE_EXECUTOR_REGISTRY.execution_state("gemini") == "enabled"
    enabled = app._RESOURCE_EXECUTOR_REGISTRY.get("gemini")
    assert enabled.kwargs["api_key"] == "private-key"
    for bad in ("https://generativelanguage.googleapis.com.evil.example/v1beta/interactions", ""):
        monkeypatch.setattr(app, "BACKEND_URL", bad)
        app._configure_resource_executors(fake_client)
        assert app._RESOURCE_EXECUTOR_REGISTRY.execution_state("gemini") == "plan-only"

@pytest.mark.asyncio
async def test_pdf_50mb_and_unknown_model_fail_before_provider_io() -> None:
    from dataclasses import replace
    upload = _upload("pdf", b"%PDF-1.7\n", modality="document", mime="application/pdf", name="p.pdf")
    oversized = ResourceUpload(verified=replace(upload.verified, actual_size=50 * 1024 * 1024 + 1), upload=upload.upload)
    calls = 0
    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client)
    try:
        with pytest.raises(ResourceExecutionError, match="50 MB"):
            await ex.prepare(model="gemini-3.8-flash", route=ResourceRoute("pdf", "native"), upload=oversized)
        with pytest.raises(ResourceExecutionError, match="modality"):
            await ex.prepare(model="gemini-future", route=ResourceRoute("pdf", "native"), upload=upload)
    finally:
        await client.aclose()
    assert calls == 0


@pytest.mark.asyncio
async def test_resumable_upload_url_is_pinned_before_raw_bytes_leave_proxy() -> None:
    upload = _upload("img", b"\x89PNG\r\n\x1a\n", modality="image", mime="image/png", name="x.png")
    calls = 0
    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, headers={"x-goog-upload-url": "https://evil.example/upload/v1beta/files?x=1"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client)
    try:
        with pytest.raises(ResourceExecutionError, match="upload URL"):
            await ex.prepare(model="gemini-3.8-flash", route=ResourceRoute("img", "native"), upload=upload)
    finally:
        await client.aclose()
    assert calls == 1


def test_enabled_health_intersects_gemini_executor_routes(monkeypatch) -> None:
    class FakeGemini:
        name = "gemini"
        enabled = True
        def executable_routes(self, model):
            assert model == "gemini-3.8-flash"
            return {"image": ("native",), "video": ("native",), "audio": ("native",), "document": ("native",), "animated_image": ("native",)}
    monkeypatch.setattr(app, "BACKEND_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "gemini")
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "secret")
    app._RESOURCE_EXECUTOR_REGISTRY.register("gemini", FakeGemini())
    try:
        doc = app._resource_capability_doc("gemini-3.8-flash")
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("gemini")
    assert doc["execution"] == "enabled"
    assert doc["routes"]["video"] == ["native"]
    assert doc["routes"]["archive"] == ["unsupported"]
    assert doc["routes"]["data"] == ["unsupported"]
    assert "secret" not in json.dumps(doc)

@pytest.mark.asyncio
async def test_processing_cancellation_rolls_back_uploaded_provider_file() -> None:
    import asyncio
    upload = _upload("vid", b"\x00\x00\x00\x18ftypmp42x", modality="video", mime="video/mp4", name="x.mp4")
    poll_seen = asyncio.Event()
    deleted = asyncio.Event()
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/v1beta/files" and request.headers.get("x-goog-upload-command") == "start":
            return httpx.Response(200, headers={"x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=x"})
        if request.url.path == "/upload/v1beta/files":
            return httpx.Response(200, json=_gemini_meta(upload, state="PROCESSING"))
        if request.method == "GET":
            poll_seen.set()
            return httpx.Response(200, json=_gemini_meta(upload, state="PROCESSING"))
        if request.method == "DELETE":
            deleted.set()
            return httpx.Response(200, json={})
        raise AssertionError(request.url)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client, poll_interval_seconds=0.05)
    task = asyncio.create_task(ex.prepare(model="gemini-3.8-flash", route=ResourceRoute("vid", "native"), upload=upload))
    try:
        await asyncio.wait_for(poll_seen.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(deleted.wait(), timeout=2)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_common_browser_media_mime_aliases_are_normalized_without_byte_changes() -> None:
    source = b"\x00\x00\x00\x18ftypqt  raw-mov"
    upload = _upload("mov", source, modality="video", mime="video/quicktime", name="clip.mov")
    seen = {}
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-goog-upload-command") == "start":
            seen["start_mime"] = request.headers.get("x-goog-upload-header-content-type")
            return httpx.Response(200, headers={"x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=x"})
        if request.url.path == "/upload/v1beta/files":
            seen["bytes"] = await request.aread()
            meta = _gemini_meta(upload)
            meta["file"]["mimeType"] = "video/mov"
            return httpx.Response(200, json=meta)
        return httpx.Response(200, json={})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client)
    try:
        handle = await ex.prepare(model="gemini-3.8-flash", route=ResourceRoute("mov", "native"), upload=upload)
    finally:
        await client.aclose()
    assert seen == {"start_mime": "video/mov", "bytes": source}
    assert handle.opaque["mime"] == "video/mov"


def test_shutdown_source_unregisters_gemini_executor() -> None:
    import inspect
    source = inspect.getsource(app._lifespan)
    assert '_RESOURCE_EXECUTOR_REGISTRY.unregister("gemini")' in source
