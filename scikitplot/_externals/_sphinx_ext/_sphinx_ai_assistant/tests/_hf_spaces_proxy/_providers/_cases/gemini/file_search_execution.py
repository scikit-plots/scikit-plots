from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import httpx
import pytest
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.base import ResourceRoute
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import (
    ProviderPrivateResource,
    ResourceExecutionError,
    ResourceExecutionSession,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.gemini import GeminiResourceExecutor
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.policy import gemini_file_search_supported_model
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._chat_contract import ChatRequest
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import ResourceDescriptor, VerifiedResource
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload

MiB = 1024 * 1024


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


def _chat(model="gemini-3.8-flash") -> ChatRequest:
    return ChatRequest(
        model=model, user_message="Review the attached project.", page_text="", page_descriptor="",
        max_tokens=500, stream=False, effort=None, thinking=False, budget_tokens=None, resources=(),
    )


def test_file_search_model_gate_and_health_constraints(monkeypatch) -> None:
    assert gemini_file_search_supported_model("gemini-3.8-flash")
    assert gemini_file_search_supported_model("gemini-3.1-pro-preview")
    assert not gemini_file_search_supported_model("gemini-future")

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    ex = GeminiResourceExecutor(api_key="secret", client=client)
    monkeypatch.setattr(app, "BACKEND_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "gemini")
    monkeypatch.setattr(app, "BACKEND_AUTH_TOKEN", "secret")
    app._RESOURCE_EXECUTOR_REGISTRY.register("gemini", ex)
    try:
        doc = app._resource_capability_doc("gemini-3.8-flash")
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("gemini")
        import asyncio
        asyncio.run(client.aclose())
    assert doc["execution"] == "enabled"
    assert doc["routes"]["archive"] == ["tool"]
    assert doc["routes"]["data"] == ["tool"]
    assert doc["route_constraints"]["archive"]["tool"] == {
        "max_file_bytes": 100 * MiB,
        "mime_types": ["application/zip"],
    }
    assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in doc["route_constraints"]["data"]["tool"]["mime_types"]
    assert doc["route_constraints"]["document"]["native"]["max_file_bytes"] == 50 * MiB
    assert "secret" not in json.dumps(doc)


@pytest.mark.asyncio
async def test_batch_file_search_uses_one_store_for_zip_and_xlsx_and_deletes_once() -> None:
    zip_upload = _upload("zip", b"PK\x03\x04zip", modality="archive", mime="application/zip", name="project.zip")
    xlsx_upload = _upload(
        "xls", b"PK\x03\x04xlsx", modality="data",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", name="book.xlsx",
    )
    store_creates = 0
    uploaded: list[bytes] = []
    deleted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal store_creates, deleted
        body = await request.aread()
        path = request.url.path
        if request.method == "POST" and path == "/v1beta/fileSearchStores":
            store_creates += 1
            payload = json.loads(body)
            assert payload["embeddingModel"] == "models/gemini-embedding-2"
            return httpx.Response(200, json={"name": "fileSearchStores/storeabc"})
        if request.method == "POST" and path == "/upload/v1beta/fileSearchStores/storeabc:uploadToFileSearchStore" and request.headers.get("x-goog-upload-command") == "start":
            return httpx.Response(200, headers={
                "x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/fileSearchStores/storeabc:uploadToFileSearchStore?upload_id=opaque"
            })
        if request.method == "POST" and path == "/upload/v1beta/fileSearchStores/storeabc:uploadToFileSearchStore":
            uploaded.append(body)
            return httpx.Response(200, json={
                "name": f"fileSearchStores/storeabc/upload/operations/op{len(uploaded)}", "done": True
            })
        if request.method == "DELETE" and path == "/v1beta/fileSearchStores/storeabc":
            assert request.url.params.get("force") == "true"
            deleted += 1
            return httpx.Response(200, json={})
        raise AssertionError((request.method, str(request.url), body[:50]))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client)
    routes = (
        ResourceRoute("zip", "tool", metadata={"modality": "archive", "intent": "raw"}),
        ResourceRoute("xls", "tool", metadata={"modality": "data", "intent": "raw"}),
    )
    session = ResourceExecutionSession(provider="gemini", model="gemini-3.8-flash", executor=ex, routes=routes)
    try:
        receipts = await session.prepare((zip_upload, xlsx_upload))
        assert len(receipts) == 2
        handles = session.private_handles
        assert handles[0].opaque["store_name"] == handles[1].opaque["store_name"] == "fileSearchStores/storeabc"
        assert sum(1 for h in handles if h.cleanup_required) == 1
        payload = ex.interactions_payload(_chat(), handles)
        assert payload["tools"] == [{"type": "file_search", "file_search_store_names": ["fileSearchStores/storeabc"]}]
        assert "fileSearchStores/storeabc" not in json.dumps(session.public_receipts())
        await session.close()
    finally:
        await client.aclose()
    assert store_creates == 1
    assert uploaded == [b"PK\x03\x04zip", b"PK\x03\x04xlsx"]
    assert deleted == 1


@pytest.mark.asyncio
async def test_parquet_and_oversize_zip_fail_before_provider_objects_or_bytes() -> None:
    calls = 0
    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client)
    parquet = _upload("pq", b"PAR1", modality="data", mime="application/vnd.apache.parquet", name="x.parquet")
    zip_small = _upload("z", b"PK\x03\x04x", modality="archive", mime="application/zip", name="x.zip")
    zip_big = ResourceUpload(verified=replace(zip_small.verified, actual_size=100 * MiB + 1), upload=zip_small.upload)
    try:
        with pytest.raises(ResourceExecutionError, match="data MIME"):
            await ex.prepare_many(model="gemini-3.8-flash", routes=(ResourceRoute("pq", "tool"),), uploads=(parquet,))
        with pytest.raises(ResourceExecutionError, match="100 MB"):
            await ex.prepare_many(model="gemini-3.8-flash", routes=(ResourceRoute("z", "tool"),), uploads=(zip_big,))
    finally:
        await client.aclose()
    assert calls == 0


@pytest.mark.asyncio
async def test_second_file_failure_rolls_back_shared_store() -> None:
    a = _upload("a", b"PK\x03\x04a", modality="archive", mime="application/zip", name="a.zip")
    b = _upload("b", b"PK\x03\x04b", modality="archive", mime="application/zip", name="b.zip")
    finalize_count = 0
    deleted = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal finalize_count, deleted
        path = request.url.path
        if request.method == "POST" and path == "/v1beta/fileSearchStores":
            return httpx.Response(200, json={"name": "fileSearchStores/storeabc"})
        if request.method == "POST" and path == "/upload/v1beta/fileSearchStores/storeabc:uploadToFileSearchStore" and request.headers.get("x-goog-upload-command") == "start":
            return httpx.Response(200, headers={"x-goog-upload-url": "https://generativelanguage.googleapis.com/upload/v1beta/fileSearchStores/storeabc:uploadToFileSearchStore?upload_id=x"})
        if request.method == "POST" and path == "/upload/v1beta/fileSearchStores/storeabc:uploadToFileSearchStore":
            finalize_count += 1
            if finalize_count == 2:
                return httpx.Response(500, json={"error": "private details"})
            return httpx.Response(200, json={"name": "fileSearchStores/storeabc/upload/operations/op1", "done": True})
        if request.method == "DELETE" and path == "/v1beta/fileSearchStores/storeabc":
            deleted += 1
            return httpx.Response(200, json={})
        raise AssertionError((request.method, str(request.url)))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ex = GeminiResourceExecutor(api_key="k", client=client)
    try:
        with pytest.raises(ResourceExecutionError, match="rejected"):
            await ex.prepare_many(
                model="gemini-3.8-flash",
                routes=(ResourceRoute("a", "tool"), ResourceRoute("b", "tool")),
                uploads=(a, b),
            )
    finally:
        await client.aclose()
    assert finalize_count == 2
    assert deleted == 1


def test_gemini_file_search_tool_handle_stays_private() -> None:
    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import ResourceExecutionReceipt
    receipt = ResourceExecutionReceipt(
        resource_id="z", provider="gemini", model="gemini-3.8-flash", route="tool",
        source_sha256="a" * 64, source_size=7, metadata={"modality": "archive", "intent": "raw"},
    )
    public = receipt.as_public_dict()
    assert "store" not in json.dumps(public).lower()
    handle = ProviderPrivateResource("z", "tool", "gemini", "a" * 64, 7, {"store_name": "fileSearchStores/private"})
    assert "private" in json.dumps(handle.opaque)


def test_health_route_constraint_sanitizer_drops_unknown_or_secret_fields(monkeypatch) -> None:
    class FakeGemini:
        name = "gemini"
        enabled = True
        def executable_routes(self, model):
            assert model == "gemini-3.8-flash"
            return {"archive": ("tool",)}
        def route_constraints(self, model):
            assert model == "gemini-3.8-flash"
            return {
                "archive": {
                    "tool": {
                        "max_file_bytes": 100 * MiB,
                        "mime_types": ["application/zip", "application/zip", "javascript:evil"],
                        "provider_file_id": "private-file-id",
                        "token": "private-token",
                        "url": "https://secret.example/resource",
                    },
                    "native": {"max_file_bytes": 1},
                }
            }
    monkeypatch.setattr(app, "BACKEND_URL", "https://generativelanguage.googleapis.com/v1beta/interactions")
    monkeypatch.setattr(app, "BACKEND_RESOURCE_ADAPTER", "gemini")
    app._RESOURCE_EXECUTOR_REGISTRY.register("gemini", FakeGemini())
    try:
        doc = app._resource_capability_doc("gemini-3.8-flash")
    finally:
        app._RESOURCE_EXECUTOR_REGISTRY.unregister("gemini")
    assert doc["route_constraints"] == {
        "archive": {"tool": {"max_file_bytes": 100 * MiB, "mime_types": ["application/zip"]}}
    }
    rendered = json.dumps(doc)
    assert "private" not in rendered and "secret.example" not in rendered and "token" not in rendered
