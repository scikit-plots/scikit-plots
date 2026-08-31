"""Regression tests for proxy chat streaming mode negotiation and stub safety."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.responses import StreamingResponse

PROXY_DIR = Path(__file__).resolve().parents[1] / "_hf_spaces_proxy"
if str(PROXY_DIR) not in sys.path:
    sys.path.insert(0, str(PROXY_DIR))

# Keep module import deterministic and offline.
os.environ.setdefault("STUB_ENABLED", "false")
os.environ.setdefault("HF_TOKEN", "")
app = importlib.import_module("app")
shared = importlib.import_module("_utils._shared_logic")


class _ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], exc: Exception | None = None) -> None:
        self.chunks = list(chunks)
        self.exc = exc
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.exc is not None:
            raise self.exc

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(
        self,
        *,
        send_plan: list[Any],
        post_plan: list[Any] | None = None,
    ) -> None:
        self.send_plan = list(send_plan)
        self.post_plan = list(post_plan or [])
        self.send_calls = 0
        self.post_calls = 0
        self.last_built_content = b""
        self.last_built_headers: dict[str, str] = {}

    def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
        self.last_built_content = kwargs.get("content") or b""
        self.last_built_headers = dict(kwargs.get("headers") or {})
        return httpx.Request(method, url, content=self.last_built_content, headers=self.last_built_headers)

    async def send(self, request: httpx.Request, *, stream: bool = False) -> httpx.Response:
        self.send_calls += 1
        item = self.send_plan.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.post_calls += 1
        item = self.post_plan.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _response(
    *,
    content_type: str,
    content: bytes | None = None,
    chunks: list[bytes] | None = None,
    exc: Exception | None = None,
    status: int = 200,
) -> httpx.Response:
    request = httpx.Request("POST", "https://upstream.test/v1/chat/completions")
    if chunks is not None or exc is not None:
        return httpx.Response(
            status,
            headers={"content-type": content_type},
            stream=_ByteStream(chunks or [], exc),
            request=request,
        )
    return httpx.Response(
        status,
        headers={"content-type": content_type},
        content=content or b"",
        request=request,
    )


async def _consume_stream(response: StreamingResponse) -> bytes:
    pieces = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, str):
            chunk = chunk.encode()
        pieces.append(chunk)
    return b"".join(pieces)


def _stream_body(model: str = "scikit-plots/test") -> bytes:
    return json.dumps(
        {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode()


def test_path3_does_not_construct_blank_bearer_header() -> None:
    _url, headers, _timeout = shared._resolve_upstream_url(
        b'{"model":"Qwen/example","messages":[]}',
        backend_url="",
        hf_token="",
    )
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_disabled_stub_is_reserved_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "STUB_ENABLED", False)
    response = await app._stub_intercept(_stream_body("stub/qa"), {})
    assert response is not None
    assert response.status_code == 503
    assert b"stub_disabled" in response.body


@pytest.mark.asyncio
async def test_stream_request_preserves_json_upstream_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = {
        "choices": [
            {"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}
        ]
    }
    fake = _FakeClient(
        send_plan=[_response(content_type="application/json", content=json.dumps(doc).encode())]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert response.status_code == 200
    assert not isinstance(response, StreamingResponse)
    assert response.media_type == "application/json"
    assert json.loads(response.body)["choices"][0]["message"]["content"] == "pong"
    assert fake.send_calls == 1


@pytest.mark.asyncio
async def test_true_sse_is_streamed_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = b'data: {"choices":[{"delta":{"content":"pong"}}]}\n\ndata: [DONE]\n\n'
    fake = _FakeClient(
        send_plan=[_response(content_type="text/event-stream", chunks=[frame])]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200
    assert await _consume_stream(response) == frame


@pytest.mark.asyncio
async def test_local_protocol_error_before_headers_is_real_502(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(send_plan=[httpx.LocalProtocolError("do-not-expose")])
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert response.status_code == 502
    assert not isinstance(response, StreamingResponse)
    assert b"upstream_local_protocol_error" in response.body
    assert b"do-not-expose" not in response.body


@pytest.mark.asyncio
async def test_remote_protocol_failure_retries_only_before_output(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = {"choices": [{"message": {"content": "recovered"}}]}
    fake = _FakeClient(
        send_plan=[
            httpx.RemoteProtocolError("stale pooled connection"),
            _response(content_type="application/json", content=json.dumps(doc).encode()),
        ]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(app, "_protocol_retries", 1)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert response.status_code == 200
    assert fake.send_calls == 2
    assert b"recovered" in response.body


@pytest.mark.asyncio
async def test_midstream_protocol_failure_emits_explicit_sse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        send_plan=[
            _response(
                content_type="text/event-stream",
                chunks=[],
                exc=httpx.RemoteProtocolError("private upstream detail"),
            )
        ]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert isinstance(response, StreamingResponse)
    wire = await _consume_stream(response)
    assert b"event: error\n" in wire
    assert b"upstream_remote_protocol_error" in wire
    assert b"private upstream detail" not in wire


@pytest.mark.asyncio
async def test_json_bridge_read_failure_retries_nonstreaming(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_doc = {"choices": [{"message": {"content": "fallback"}}]}
    fake = _FakeClient(
        send_plan=[
            _response(
                content_type="application/json",
                chunks=[],
                exc=httpx.RemoteProtocolError("truncated json body"),
            ),
            _response(content_type="application/json", content=json.dumps(fallback_doc).encode()),
        ],
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert response.status_code == 200
    assert fake.send_calls == 2
    assert fake.post_calls == 0
    assert b"fallback" in response.body

@pytest.mark.asyncio
async def test_empty_json_200_becomes_502(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        send_plan=[_response(content_type="application/json", content=b"")]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert response.status_code == 502
    assert b"upstream_empty_response" in response.body


@pytest.mark.asyncio
async def test_empty_sse_200_emits_terminal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(
        send_plan=[_response(content_type="text/event-stream", chunks=[])]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert isinstance(response, StreamingResponse)
    wire = await _consume_stream(response)
    assert b"event: error\n" in wire
    assert b"upstream_empty_stream" in wire


@pytest.mark.asyncio
async def test_buffered_mislabelled_sse_is_reclassified(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = b'data: {"choices":[{"delta":{"content":"pong"}}]}\n\ndata: [DONE]\n\n'
    fake = _FakeClient(
        send_plan=[_response(content_type="text/plain", content=frame)]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://upstream.test/v1/chat/completions", {}, 30.0),
    )

    response = await app._forward(_stream_body())
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"
    assert response.body == frame

@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Illegal header value b'Bearer SECRET'", "illegal-header"),
        ("Too much data for declared Content-Length", "content-length-overrun"),
        ("Too little data for declared Content-Length", "content-length-underrun"),
        ("Missing mandatory Host header", "missing-host"),
        ("Illegal request line", "request-line"),
        ("some other local protocol issue", "unspecified"),
    ],
)
def test_local_protocol_reason_is_fixed_and_privacy_safe(message: str, expected: str) -> None:
    reason = app._local_protocol_reason(httpx.LocalProtocolError(message))
    assert reason == expected
    assert "SECRET" not in reason


@pytest.mark.asyncio
async def test_nonstreaming_upstream_403_is_sanitized_and_coded(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_body = b'{"private":"provider-account-secret"}'
    fake = _FakeClient(
        send_plan=[_response(content_type="application/json", content=secret_body, status=403)]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://router.huggingface.co/v1/chat/completions", {}, 30.0),
    )
    body = json.dumps({"model": "Qwen/example", "stream": False, "messages": []}).encode()
    response = await app._forward(body)
    assert response.status_code == 403
    doc = json.loads(response.body)
    assert doc["code"] == "UPSTREAM_AUTH_OR_ACCESS_REJECTED"
    assert b"provider-account-secret" not in response.body


@pytest.mark.asyncio
async def test_streaming_upstream_400_is_sanitized_and_coded(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_body = b'{"error":"provider-private-routing-detail"}'
    fake = _FakeClient(
        send_plan=[_response(content_type="application/json", content=secret_body, status=400)]
    )
    monkeypatch.setattr(app, "_http_client", fake)
    monkeypatch.setattr(
        app,
        "_resolve_url",
        lambda _body: ("https://router.huggingface.co/v1/chat/completions", {}, 30.0),
    )
    response = await app._forward(_stream_body("Qwen/example"))
    assert response.status_code == 400
    doc = json.loads(response.body)
    assert doc["code"] == "UPSTREAM_REQUEST_REJECTED"
    assert b"provider-private-routing-detail" not in response.body
