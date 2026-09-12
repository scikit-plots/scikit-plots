"""Run 126 — first-class raw resource contract and bounded multipart transport."""
from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import hashlib
import json
import pathlib
import sys

import pytest
from starlette.requests import Request

_PROXY = RUNTIME_ROOT / "_hf_spaces_proxy"
if str(_PROXY) not in sys.path:
    sys.path.insert(0, str(_PROXY))

from _utils._chat_contract import parse_chat_request  # noqa: E402
from _utils._resource_contract import (  # noqa: E402
    ResourceContractError,
    classify_resource,
    parse_resource_descriptors,
    resource_summary,
)
from _utils._resource_transport import (  # noqa: E402
    ResourceTransportError,
    parse_resource_chat_request,
)
from _utils._stub_model import build_stub_reply  # noqa: E402


def _resource(**over):
    row = {
        "id": "r0",
        "name": "project.zip",
        "mime_type": "application/zip",
        "size": 8,
        "modality": "archive",
        "intent": "auto",
        "relative_path": "",
        "archive_name": "",
    }
    row.update(over)
    return row


def _chat(resources):
    return {
        "contract": "scikitplot-chat-v1",
        "model": "stub/mirror",
        "user_message": "inspect resources",
        "context": {"page_text": "", "page_descriptor": ""},
        "max_tokens": 1000,
        "stream": False,
        "resources": resources,
    }


def _multipart_request(metadata, files, *, extra_fields=(), boundary="run126-boundary"):
    chunks: list[bytes] = []

    def field(name: str, value: str):
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode(), b"\r\n",
        ])

    def file_field(name: str, filename: str, content_type: str, data: bytes):
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(), data, b"\r\n",
        ])

    field("request", json.dumps(metadata, separators=(",", ":")))
    for name, value in extra_fields:
        field(name, value)
    for rid, filename, ctype, data in files:
        file_field(f"resource:{rid}", filename, ctype, data)
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "scheme": "http",
    }
    return Request(scope, receive), body


@pytest.mark.parametrize(
    ("name", "mime", "prefix", "modality", "canonical"),
    [
        ("a.pdf", "application/octet-stream", b"%PDF-1.7\n", "document", "application/pdf"),
        ("a.zip", "application/octet-stream", b"PK\x03\x04xxxx", "archive", "application/zip"),
        ("a.png", "", b"\x89PNG\r\n\x1a\n", "image", "image/png"),
        ("a.jpg", "", b"\xff\xd8\xff\xe0", "image", "image/jpeg"),
        ("a.gif", "", b"GIF89a", "animated_image", "image/gif"),
        ("a.svg", "text/plain", b"<?xml?><svg viewBox='0 0 1 1'>", "vector_image", "image/svg+xml"),
        ("a.wav", "", b"RIFF\x00\x00\x00\x00WAVE", "audio", "audio/wav"),
        ("a.mp3", "", b"ID3\x04\x00", "audio", "audio/mpeg"),
        ("a.mp4", "video/mp4", b"\x00\x00\x00\x18ftypisom", "video", "video/mp4"),
        ("a.webm", "video/webm", b"\x1a\x45\xdf\xa3", "video", "video/webm"),
        ("a.md", "text/markdown", b"hello", "text", "text/markdown"),
        ("a.parquet", "", b"PAR1", "data", "application/octet-stream"),
    ],
)
def test_signature_and_hint_classification(name, mime, prefix, modality, canonical):
    got = classify_resource(name=name, mime_type=mime, prefix=prefix)
    assert got[0] == modality
    assert got[1] == canonical


def test_descriptor_security_and_uniqueness():
    with pytest.raises(ResourceContractError):
        parse_resource_descriptors([_resource(relative_path="../secret")])
    sanitized = parse_resource_descriptors([_resource(name="bad\u202e.exe")])[0]
    assert "\u202e" not in sanitized.name
    with pytest.raises(ResourceContractError):
        parse_resource_descriptors([_resource(), _resource()])


def test_chat_v1_has_first_class_resources():
    body = json.dumps(_chat([_resource()])).encode()
    chat = parse_chat_request(body, allowed_models=["stub/mirror"])
    assert len(chat.resources) == 1
    assert chat.resources[0].intent == "auto"
    assert chat.resources[0].modality == "archive"


@pytest.mark.asyncio
async def test_multipart_hashes_measures_and_sniffs_actual_bytes():
    data = b"%PDF-1.7"
    metadata = _chat([_resource(name="claimed.zip", mime_type="application/zip", size=len(data), modality="archive")])
    request, wire = _multipart_request(metadata, [("r0", "claimed.zip", "application/zip", data)])
    request_bytes, chat, uploads, stats = await parse_resource_chat_request(
        request, allowed_models=["stub/mirror"], max_file_bytes=1024, max_resource_bytes=2048, max_request_bytes=4096
    )
    try:
        assert json.loads(request_bytes) == metadata
        assert chat.resources[0].modality == "archive"  # declaration retained for diagnostics
        assert len(uploads) == 1
        verified = uploads[0].verified
        assert verified.actual_size == len(data)
        assert verified.sha256 == hashlib.sha256(data).hexdigest()
        assert verified.detected_modality == "document"
        assert verified.detected_mime == "application/pdf"
        assert stats["wire_body_bytes"] == len(wire)
        assert stats["wire_body_sha256"] == hashlib.sha256(wire).hexdigest()
    finally:
        for row in uploads:
            await row.close()


@pytest.mark.asyncio
async def test_multipart_rejects_size_mismatch_extra_field_and_limits():
    data = b"PK\x03\x04abcd"
    metadata = _chat([_resource(size=len(data) + 1)])
    req, _ = _multipart_request(metadata, [("r0", "project.zip", "application/zip", data)])
    with pytest.raises(ResourceTransportError, match="byte length"):
        await parse_resource_chat_request(req, allowed_models=["stub/mirror"], max_file_bytes=1024, max_resource_bytes=2048, max_request_bytes=4096)

    metadata = _chat([_resource(size=len(data))])
    req, _ = _multipart_request(metadata, [("r0", "project.zip", "application/zip", data)], extra_fields=[("surprise", "x")])
    with pytest.raises(ResourceTransportError, match="unsupported multipart field"):
        await parse_resource_chat_request(req, allowed_models=["stub/mirror"], max_file_bytes=1024, max_resource_bytes=2048, max_request_bytes=4096)

    req, _ = _multipart_request(metadata, [("r0", "project.zip", "application/zip", data)])
    with pytest.raises(ResourceTransportError):
        await parse_resource_chat_request(req, allowed_models=["stub/mirror"], max_file_bytes=4, max_resource_bytes=2048, max_request_bytes=4096)


def test_stub_mirror_can_report_resource_chain_without_bytes():
    summary = [{
        "id": "r0", "name": "movie.mp4", "intent": "raw",
        "declared_modality": "video", "detected_modality": "video",
        "declared_mime": "video/mp4", "detected_mime": "video/mp4",
        "declared_size": 123, "actual_size": 123, "sha256": "a" * 64,
        "signature": "iso-bmff", "relative_path": "", "archive_name": "",
    }]
    reply, _report = build_stub_reply(
        "mirror",
        "",
        {"model": "stub/mirror", "messages": [{"role": "user", "content": "hi"}]},
        {},
        mode_context={"resources": summary, "wire_body_bytes": 321, "wire_body_sha256": "b" * 64, "multipart": True},
    )
    assert "First-class raw resources" in reply
    assert "movie.mp4" in reply
    assert "video" in reply
    assert "multipart" in reply.lower()
    assert "<RAW" not in reply
