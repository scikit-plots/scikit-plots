from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _storage as st

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "_static" / "ai-assistant.js"
HOST = ROOT / "_static" / "ai-assistant-isolation-host.js"


def _target(provider: str, *, api_base: str = "") -> dict:
    return {
        "id": f"{provider}-1",
        "provider": provider,
        "role": "primary",
        "repo": "org/repo",
        "token_env": "AI_RECORD_STORAGE_TOKEN_TEST",
        **({"api_base": api_base} if api_base else {}),
    }


def test_b44_control_response_default_and_hard_clamp(monkeypatch):
    monkeypatch.delenv("AI_RECORD_STORAGE_CONTROL_RESPONSE_MAX_BYTES", raising=False)
    assert st._control_response_limit() == 4 * 1024 * 1024
    monkeypatch.setenv("AI_RECORD_STORAGE_CONTROL_RESPONSE_MAX_BYTES", str(99 * 1024 * 1024))
    assert st._control_response_limit() == 16 * 1024 * 1024


def test_b44_invalid_control_response_setting_falls_back(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_CONTROL_RESPONSE_MAX_BYTES", "not-an-int")
    assert st._control_response_limit() == 4 * 1024 * 1024


def test_b44_gitlab_custom_api_base_accepts_https_host():
    t = st.load_storage_targets(json.dumps([_target("gitlab", api_base="https://git.example/api/v4")]))[0]
    assert t.api_base == "https://git.example/api/v4"


def test_b44_gitlab_custom_api_base_rejects_ambiguous_authority():
    bad_values = [
        "http://git.example/api/v4",
        "https://user:pass@git.example/api/v4",
        "https://git.example/api/v4?token=x",
        "https://git.example/api/v4#frag",
        "https://git.example/a/../api/v4",
    ]
    for bad in bad_values:
        with pytest.raises(st.StorageConfigError):
            st.load_storage_targets(json.dumps([_target("gitlab", api_base=bad)]))


def test_b44_non_gitlab_rejects_api_base():
    with pytest.raises(st.StorageConfigError, match="TARGET_API_BASE_UNSUPPORTED"):
        st.load_storage_targets(json.dumps([_target("github", api_base="https://api.github.com")]))


def test_b44_no_body_request_does_not_consume_response_stream(monkeypatch):
    consumed = {"value": False}

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            consumed["value"] = True
            yield b"x" * 100

    async def handler(request):
        return httpx.Response(201, stream=Stream(), request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        c = st.StorageCoordinator([], client=client)
        assert asyncio.run(c._request_no_body("POST", "https://example.test/x")) == 201
        assert consumed["value"] is False
    finally:
        asyncio.run(client.aclose())


def test_b44_bounded_json_rejects_declared_oversize(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_CONTROL_RESPONSE_MAX_BYTES", "1024")
    async def handler(request):
        return httpx.Response(200, headers={"content-length": "2048"}, content=b"{}", request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        c = st.StorageCoordinator([], client=client)
        with pytest.raises(st.StorageWriteError, match="PROVIDER_RESPONSE_TOO_LARGE"):
            asyncio.run(c._request_bounded_json("GET", "https://example.test/x"))
    finally:
        asyncio.run(client.aclose())


def test_b44_bounded_json_rejects_streamed_oversize(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_CONTROL_RESPONSE_MAX_BYTES", "1024")
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{" + b" " * 700
            yield b" " * 700 + b"}"
    async def handler(request):
        return httpx.Response(200, stream=Stream(), request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        c = st.StorageCoordinator([], client=client)
        with pytest.raises(st.StorageWriteError, match="PROVIDER_RESPONSE_TOO_LARGE"):
            asyncio.run(c._request_bounded_json("GET", "https://example.test/x"))
    finally:
        asyncio.run(client.aclose())


def test_b44_bounded_json_parses_only_after_bounded_read(monkeypatch):
    monkeypatch.setenv("AI_RECORD_STORAGE_CONTROL_RESPONSE_MAX_BYTES", "4096")
    async def handler(request):
        return httpx.Response(200, content=b'{"sha":"abc"}', request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        c = st.StorageCoordinator([], client=client)
        status, doc = asyncio.run(c._request_bounded_json("GET", "https://example.test/x"))
        assert status == 200 and doc == {"sha": "abc"}
    finally:
        asyncio.run(client.aclose())


def test_b44_hf_factory_is_restored_after_scoped_call():
    import huggingface_hub.utils._http as hf_http
    before = hf_http._GLOBAL_CLIENT_FACTORY
    assert st._with_bounded_hf_client(lambda: "ok") == "ok"
    assert hf_http._GLOBAL_CLIENT_FACTORY is before


def test_b44_same_origin_uses_live_rendered_visibility_authority():
    src = MAIN.read_text()
    assert "function _stripModelOnlyLiveNodes(liveRoot, cloneRoot)" in src
    assert "_stripModelOnlyLiveNodes(content, cloned);" in src
    for token in ("contentVisibility === 'hidden'", "classicallyClipped", "extremeIndent", "unreachable"):
        assert token in src


def test_b44_isolation_host_uses_live_rendered_visibility_authority():
    src = HOST.read_text()
    for token in ("contentVisibility === 'hidden'", "classicallyClipped", "extremeIndent", "unreachable"):
        assert token in src
    assert "The live rendered DOM is the visibility authority" in src
