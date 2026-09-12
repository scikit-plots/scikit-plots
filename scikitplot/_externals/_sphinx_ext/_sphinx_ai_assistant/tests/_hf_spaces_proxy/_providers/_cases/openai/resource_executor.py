from __future__ import annotations

import hashlib
import io
import json

import httpx
import pytest
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.base import ResourceRoute
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import ResourceExecutionError, ResourceExecutionSession
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.openai import OpenAIResourceExecutor
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import ResourceDescriptor, VerifiedResource
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload


def _upload(rid: str, data: bytes, *, modality: str, mime: str, name: str) -> ResourceUpload:
    desc = ResourceDescriptor(
        id=rid, name=name, mime_type=mime, size=len(data), modality=modality, intent="raw"
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


def _route(upload: ResourceUpload, route: str = "native") -> ResourceRoute:
    return ResourceRoute(
        resource_id=upload.verified.id,
        route=route,
        metadata={"modality": upload.verified.detected_modality, "intent": upload.verified.intent},
    )


@pytest.mark.asyncio
async def test_document_upload_streams_original_bytes_user_data_and_expiry():
    data = b"%PDF-1.7\nORIGINAL-BYTES\x00\xff"
    upload = _upload("r0", data, modality="document", mime="application/pdf", name='paper "x".pdf')
    seen = {}

    async def handler(request: httpx.Request):
        body = await request.aread()
        seen.update(method=request.method, url=str(request.url), headers=dict(request.headers), body=body)
        return httpx.Response(200, json={"id": "file-doc123", "bytes": len(data), "purpose": "user_data"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="sk-test-private", client=client)
        handle = await ex.prepare(model="gpt-x", route=_route(upload), upload=upload)
    body = seen["body"]
    assert seen["method"] == "POST"
    assert seen["url"] == "https://api.openai.com/v1/files"
    assert seen["headers"]["authorization"] == "Bearer sk-test-private"
    assert b'name="purpose"\r\n\r\nuser_data' in body
    assert b'name="expires_after[anchor]"\r\n\r\ncreated_at' in body
    assert b'name="expires_after[seconds]"\r\n\r\n3600' in body
    assert data in body
    assert b"T1JJR0lOQUw" not in body  # no base64 transform
    assert int(seen["headers"]["content-length"]) == len(body)
    assert handle.opaque["file_id"] == "file-doc123"
    assert await upload.upload.read() == data  # executor restores position


@pytest.mark.asyncio
async def test_image_upload_uses_vision_purpose():
    data = b"\x89PNG\r\n\x1a\nabc"
    upload = _upload("r0", data, modality="image", mime="image/png", name="image.png")
    bodies = []

    async def handler(request):
        body = await request.aread(); bodies.append(body)
        return httpx.Response(200, json={"id": "file-img123", "bytes": len(data)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="k", client=client)
        handle = await ex.prepare(model="gpt-x", route=_route(upload), upload=upload)
    assert b'name="purpose"\r\n\r\nvision' in bodies[0]
    assert handle.opaque["purpose"] == "vision"


@pytest.mark.asyncio
async def test_tool_archive_is_uploaded_as_user_data_without_extraction():
    data = b"PK\x03\x04RAW-ZIP-PAYLOAD"
    upload = _upload("r0", data, modality="archive", mime="application/zip", name="project.zip")
    bodies = []

    async def handler(request):
        body = await request.aread(); bodies.append(body)
        return httpx.Response(200, json={"id": "file-zip123", "bytes": len(data)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="k", client=client)
        handle = await ex.prepare(model="gpt-x", route=_route(upload, "tool"), upload=upload)
    assert data in bodies[0]
    assert b'name="purpose"\r\n\r\nuser_data' in bodies[0]
    assert handle.route == "tool"


@pytest.mark.asyncio
async def test_release_deletes_private_file_and_404_is_idempotent_success():
    calls = []

    async def handler(request):
        calls.append((request.method, str(request.url), request.headers.get("authorization")))
        if len(calls) == 1:
            return httpx.Response(200, json={"deleted": True, "id": "file-abc", "object": "file"})
        return httpx.Response(404, json={"error": {"message": "gone"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="secret", client=client)
        from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import ProviderPrivateResource
        h1 = ProviderPrivateResource("r0", "native", "openai", "a"*64, 1, {"file_id": "file-abc"})
        h2 = ProviderPrivateResource("r1", "native", "openai", "b"*64, 1, {"file_id": "file-already-gone"})
        await ex.release(h1); await ex.release(h2)
    assert calls[0][0] == "DELETE"
    assert calls[0][1].endswith("/v1/files/file-abc")
    assert h1.released and h2.released


@pytest.mark.asyncio
async def test_provider_errors_are_bounded_and_do_not_reflect_provider_body_or_key():
    async def handler(request):
        await request.aread()
        return httpx.Response(500, text="PRIVATE_PROVIDER_BODY sk-secret file-private")

    upload = _upload("r0", b"abc", modality="document", mime="text/plain", name="a.txt")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="sk-secret", client=client)
        with pytest.raises(ResourceExecutionError) as err:
            await ex.prepare(model="gpt-x", route=_route(upload), upload=upload)
    text = str(err.value)
    assert "PRIVATE_PROVIDER_BODY" not in text
    assert "sk-secret" not in text
    assert "file-private" not in text


@pytest.mark.asyncio
async def test_file_metadata_identity_checks_fail_closed():
    upload = _upload("r0", b"abcd", modality="document", mime="text/plain", name="a.txt")
    responses = [
        httpx.Response(200, json={"id": "bad", "bytes": 4}),
        httpx.Response(200, json={"id": "file-x", "bytes": 999}),
        httpx.Response(200, text="not-json"),
    ]

    async def handler(request):
        await request.aread()
        return responses.pop(0)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="k", client=client)
        for _ in range(3):
            with pytest.raises(ResourceExecutionError):
                await ex.prepare(model="gpt-x", route=_route(upload), upload=upload)


def test_official_origin_is_pinned_and_key_is_required():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    try:
        with pytest.raises(ValueError):
            OpenAIResourceExecutor(api_key="", client=client)
        for bad in (
            "http://api.openai.com",
            "https://api.openai.com.evil.example",
            "https://user@api.openai.com",
            "https://api.openai.com/v1",
            "https://api.openai.com?x=1",
        ):
            with pytest.raises(ValueError):
                OpenAIResourceExecutor(api_key="k", client=client, base_url=bad)
    finally:
        import asyncio
        asyncio.run(client.aclose())


@pytest.mark.asyncio
async def test_common_session_public_receipt_never_contains_openai_file_id():
    data = b"doc"
    upload = _upload("r0", data, modality="document", mime="text/plain", name="a.txt")

    async def handler(request):
        if request.method == "POST":
            await request.aread()
            return httpx.Response(200, json={"id": "file-SECRET123", "bytes": len(data)})
        return httpx.Response(200, json={"deleted": True, "id": "file-SECRET123"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ex = OpenAIResourceExecutor(api_key="secret", client=client)
        session = ResourceExecutionSession(provider="openai", model="gpt-x", executor=ex, routes=[_route(upload)])
        await session.prepare([upload])
        encoded = json.dumps(session.public_receipts())
        assert "file-SECRET123" not in encoded
        assert "secret" not in encoded
        await session.close()
        assert session.receipts[0].state == "released"
