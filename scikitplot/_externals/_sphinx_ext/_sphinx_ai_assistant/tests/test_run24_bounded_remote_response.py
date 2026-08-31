from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "_static"
MAIN = STATIC / "ai-assistant.js"
HOST = STATIC / "ai-assistant-isolation-host.js"
FRAME = STATIC / "ai-assistant-isolated-frame.js"
APP = ROOT / "_hf_spaces_proxy" / "app.py"
DEV = ROOT / "dev_proxy.py"
WORKER = ROOT / "_cf_worker" / "index.js"
WRANGLER = ROOT / "_cf_worker" / "wrangler.toml"
SHARE = ROOT / "_hf_spaces_proxy" / "_utils" / "_share_contract.py"


class _AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class _SyncChunks(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks


def _proxy_app():
    return importlib.import_module(
        "scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy.app"
    )


def _dev_proxy():
    key = "HF_TOKEN"
    previous = os.environ.get(key)
    os.environ[key] = "hf_test_token"
    try:
        return importlib.import_module(
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant.dev_proxy"
        )
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def test_b43_browser_remote_reads_are_stream_bounded_and_fail_closed():
    main = MAIN.read_text()
    assert "var _CONTROL_RESPONSE_MAX_BYTES = 512 * 1024;" in main
    assert "var _CANONICAL_RESPONSE_MAX_BYTES = 1024 * 1024;" in main
    assert "var _CHAT_RESPONSE_MAX_BYTES = 8 * 1024 * 1024;" in main
    assert "REMOTE_RESPONSE_STREAM_UNAVAILABLE" in main
    assert "REMOTE_RESPONSE_TOO_LARGE" in main
    assert "REMOTE_RESPONSE_INVALID_LENGTH" in main
    assert "return _readResponseTextBounded(response, _CANONICAL_RESPONSE_MAX_BYTES);" in main
    assert "resp.ok ? _readResponseJsonBounded(resp, _CONTROL_RESPONSE_MAX_BYTES)" in main

    static = main[main.index("function fetchStaticMarkdown()") : main.index("function getMarkdownUrl()")]
    discovery = main[main.index("function _fetchProxyDatasetInfo") : main.index("function _buildDatasetLinkCard")]
    assert ".text()" not in static
    assert ".json()" not in discovery


def test_b43_isolation_reads_are_stream_bounded_without_text_fallback():
    host = HOST.read_text()
    frame = FRAME.read_text()
    assert "response.body.getReader" in host
    assert "response.text()" not in host
    assert "response.body.getReader" in frame
    assert "response.text()" not in frame
    assert "CANONICAL_STREAM_UNAVAILABLE" in host
    assert "POLICY_STREAM_UNAVAILABLE" in frame


def test_b43_hf_proxy_imports_and_enforces_declared_and_streamed_limits():
    app = _proxy_app()
    assert 64 * 1024 <= app.MAX_UPSTREAM_RESPONSE_BYTES <= 32 * 1024 * 1024

    declared = httpx.Response(
        200,
        headers={"content-length": "2049"},
        stream=_AsyncChunks([b"ok"]),
        request=httpx.Request("GET", "https://upstream.example"),
    )
    with pytest.raises(app._UpstreamResponseTooLarge):
        asyncio.run(app._read_upstream_limited(declared, 2048))

    invalid = httpx.Response(
        200,
        headers={"content-length": "NaN"},
        stream=_AsyncChunks([b"ok"]),
        request=httpx.Request("GET", "https://upstream.example"),
    )
    with pytest.raises(app._UpstreamResponseTooLarge):
        asyncio.run(app._read_upstream_limited(invalid, 2048))

    chunked = httpx.Response(
        200,
        stream=_AsyncChunks([b"a" * 1024, b"b" * 1025]),
        request=httpx.Request("GET", "https://upstream.example"),
    )
    with pytest.raises(app._UpstreamResponseTooLarge):
        asyncio.run(app._read_upstream_limited(chunked, 2048))

    valid = httpx.Response(
        200,
        stream=_AsyncChunks([b"a" * 1024, b"b" * 1024]),
        request=httpx.Request("GET", "https://upstream.example"),
    )
    assert asyncio.run(app._read_upstream_limited(valid, 2048)) == b"a" * 1024 + b"b" * 1024

    health = asyncio.run(app.health())
    health_doc = json.loads(health.body)
    assert health_doc["limits"] == {"max_upstream_response_bytes": app.MAX_UPSTREAM_RESPONSE_BYTES}
    assert "HF_TOKEN" not in health.body.decode("utf-8")


def test_b43_dev_proxy_enforces_declared_and_streamed_limits():
    dev = _dev_proxy()
    declared = httpx.Response(
        200,
        headers={"content-length": "4097"},
        stream=_SyncChunks([b"ok"]),
        request=httpx.Request("GET", "https://upstream.example"),
    )
    with pytest.raises(dev._UpstreamResponseTooLarge):
        dev._read_upstream_limited(declared, 4096)

    invalid = httpx.Response(
        200,
        headers={"content-length": "+12"},
        stream=_SyncChunks([b"ok"]),
        request=httpx.Request("GET", "https://upstream.example"),
    )
    with pytest.raises(dev._UpstreamResponseTooLarge):
        dev._read_upstream_limited(invalid, 4096)

    chunked = httpx.Response(
        200,
        stream=_SyncChunks([b"a" * 2048, b"b" * 2049]),
        request=httpx.Request("GET", "https://upstream.example"),
    )
    with pytest.raises(dev._UpstreamResponseTooLarge):
        dev._read_upstream_limited(chunked, 4096)


def test_b43_hf_and_dev_proxy_use_streamed_upstream_transport():
    app = APP.read_text()
    dev = DEV.read_text()
    assert "await _http_client.send(request, stream=True)" in app
    assert "await upstream.aread()" not in app
    assert "httpx.stream(" in dev
    assert "response.read()" not in dev
    assert "response.json()" not in dev
    assert "MAX_UPSTREAM_RESPONSE_BYTES" in app
    assert "MAX_UPSTREAM_RESPONSE_BYTES" in dev


def test_b43_worker_bounds_declared_and_unknown_length_upstream_responses():
    worker = WORKER.read_text()
    wrangler = WRANGLER.read_text()
    assert "CHAT_MAX_RESPONSE_BYTES_DEFAULT = 8 * 1024 * 1024" in worker
    assert "CHAT_MAX_RESPONSE_BYTES_HARD = 32 * 1024 * 1024" in worker
    assert "_upstreamLengthAllowed" in worker
    assert "_boundedUpstreamStream" in worker
    assert "await hfResponse.text()" not in worker
    assert "await hfResponse.json()" not in worker
    assert "MAX_RESPONSE_BYTES" in wrangler


def test_b43_share_viewers_stream_bound_json_in_both_server_implementations():
    hf = SHARE.read_text()
    worker = WORKER.read_text()
    for source in (hf, worker):
        assert "const max=4*1024*1024" in source
        assert "r.body.getReader" in source
        assert "await r.json()" not in source
        assert "Share response is too large." in source
        assert "Bounded Share reader unavailable." in source


def test_b43_no_direct_browser_buffering_calls_remain_in_production_runtime():
    main = MAIN.read_text()
    host = HOST.read_text()
    frame = FRAME.read_text()
    assert "response.text()" not in main
    assert "resp.json()" not in main
    assert "response.text()" not in host
    assert "response.text()" not in frame
