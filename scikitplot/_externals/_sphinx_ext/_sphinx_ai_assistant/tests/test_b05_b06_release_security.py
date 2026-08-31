"""Run 11: B05/B06 browser-origin, identity, body and abuse-gate contracts."""
from __future__ import annotations

import asyncio
import importlib
import pathlib
import sys

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROXY = ROOT / "_hf_spaces_proxy"
MODEL = ROOT / "_hf_spaces_model"
WORKER = ROOT / "_cf_worker" / "index.js"
DEV_PROXY = ROOT / "dev_proxy.py"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy = importlib.import_module("app")
shared = importlib.import_module("_utils._shared_logic")


def _request(*, headers: list[tuple[bytes, bytes]] | None = None, chunks: list[bytes] | None = None, client=("203.0.113.7", 1234)):
    queue = list(chunks or [b""])
    calls = {"n": 0}

    async def receive():
        calls["n"] += 1
        body = queue.pop(0) if queue else b""
        return {"type": "http.request", "body": body, "more_body": bool(queue)}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": headers or [],
        "client": client,
        "server": ("proxy.example", 443),
    }
    return Request(scope, receive), calls


def test_hf_default_cors_is_exact_and_browser_denial_happens_before_handler():
    assert proxy._DEFAULT_ALLOWED_ORIGINS == (
        "https://scikit-plots.github.io",
        "https://scikit-plots-learn.readthedocs.io",
    )
    assert proxy._allowed_origins != ["*"]
    with TestClient(proxy.app) as client:
        no_origin = client.get("/health")
        assert no_origin.status_code == 200
        assert "access-control-allow-origin" not in no_origin.headers

        denied = client.get("/health", headers={"Origin": "https://attacker.example"})
        assert denied.status_code == 403
        assert "access-control-allow-origin" not in denied.headers

        allowed = client.get("/health", headers={"Origin": "https://scikit-plots.github.io"})
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "https://scikit-plots.github.io"

        preflight = client.options(
            "/v1/share",
            headers={
                "Origin": "https://scikit-plots.github.io",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://scikit-plots.github.io"


def test_hf_stream_reader_rejects_declared_and_chunked_oversize_before_full_buffer():
    declared, calls = _request(headers=[(b"content-length", b"999")], chunks=[b"must-not-read"])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(proxy._read_limited_body(declared, 8, "Request"))
    assert exc.value.status_code == 413
    assert calls["n"] == 0

    chunked, calls = _request(chunks=[b"abcd", b"efgh", b"never-consumed"])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(proxy._read_limited_body(chunked, 6, "Request"))
    assert exc.value.status_code == 413
    assert calls["n"] == 2

    malformed, calls = _request(headers=[(b"content-length", b"NaN")], chunks=[b"must-not-read"])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(proxy._read_limited_body(malformed, 8, "Request"))
    assert exc.value.status_code == 400
    assert calls["n"] == 0


def test_hf_rate_identity_store_has_a_hard_unique_identity_bound(monkeypatch):
    store: dict[str, tuple[int, float]] = {}
    lock = asyncio.Lock()
    monkeypatch.setattr(proxy, "_MAX_RL_ENTRIES", 2)

    async def exercise():
        assert (await proxy._consume_rate_limit(store, lock, "a", limit=10))[0]
        assert (await proxy._consume_rate_limit(store, lock, "b", limit=10))[0]
        allowed, count = await proxy._consume_rate_limit(store, lock, "c", limit=10)
        return allowed, count

    allowed, count = asyncio.run(exercise())
    assert (allowed, count) == (False, 0)
    assert set(store) == {"a", "b"}


def test_forwarded_identity_is_default_deny_and_shared_helper(monkeypatch):
    req, _ = _request(headers=[(b"x-forwarded-for", b"198.51.100.22, 10.0.0.1")])
    monkeypatch.setattr(proxy, "TRUST_X_FORWARDED_FOR", False)
    assert proxy._client_ip(req) == "203.0.113.7"
    monkeypatch.setattr(proxy, "TRUST_X_FORWARDED_FOR", True)
    assert proxy._client_ip(req) == "198.51.100.22"

    src = (PROXY / "app.py").read_text(encoding="utf-8")
    for route in ("chat.ratelimit", "contribute.ratelimit", "share.ratelimit", "feedback.ratelimit"):
        pos = src.index(route)
        assert "_client_ip(request)" in src[max(0, pos - 1000):pos]


def test_all_hf_public_body_routes_use_the_streaming_gate():
    src = (PROXY / "app.py").read_text(encoding="utf-8")
    assert "await request.body()" not in src
    assert "request.stream()" in src
    assert "return await _read_limited_body(request, MAX_BODY_BYTES" in src
    assert 'await _read_limited_body(request, CONTRIBUTION_MAX_BODY_BYTES' in src
    assert 'await _read_limited_body(request, SHARE_MAX_BODY_BYTES' in src
    assert 'await _read_limited_body(request, FEEDBACK_MAX_BODY_BYTES' in src
    assert "CHAT_RATE_LIMIT_PER_HOUR" in src


def test_direct_model_body_gate_streams_and_hard_clamps_configuration():
    src = (MODEL / "app.py").read_text(encoding="utf-8")
    helper = src[src.index("async def _read_bounded_body("):src.index("\n\n", src.index("    return bytes(body)", src.index("async def _read_bounded_body(")))]
    assert "request.stream()" in helper
    assert "await request.body()" not in helper
    assert "content-length" in helper.lower()
    assert "16 * 1024 * 1024" in src
    assert '@_app_inner.middleware("http")' in src
    assert "if not _origin_allowed(request):" in src
    assert '"https://scikit-plots-ai.hf.space"' in src


def test_worker_uses_exact_default_origin_streaming_body_and_edge_identity():
    src = WORKER.read_text(encoding="utf-8")
    assert 'const DEFAULT_ALLOWED_ORIGINS = Object.freeze([' in src
    assert '"https://scikit-plots.github.io"' in src
    assert '"https://scikit-plots-learn.readthedocs.io"' in src
    assert "if (!_originAllowed(request, env))" in src
    assert "request.body.getReader()" in src
    assert "request.text()" not in src
    assert "CHAT_MAX_BODY_BYTES_HARD = 16 * 1024 * 1024" in src
    assert "CF-Connecting-IP" in src
    assert "CHAT_RATE_LIMIT_PER_HOUR_DEFAULT = 30" in src
    assert "SHARE_RATE_LIMIT_PER_HOUR_DEFAULT = 10" in src
    assert "FEEDBACK_RATE_LIMIT_PER_HOUR_DEFAULT = 30" in src
    rate = src[src.index("async function _rateLimit("):src.index("async function _kvPut(")]
    # Run 15: Durable Objects are the bundled authoritative cross-PoP decision
    # plane; the unique-event KV limiter remains only as an explicit soft
    # compatibility fallback when authoritative mode is not required.
    assert "if (env.RATE_LIMIT_DO)" in rate
    assert "await stub.consume" in rate
    assert "authoritative: true" in rate
    assert "env.SHARE_KV.list({ prefix: eventPrefix" in rate
    assert "randomUUID" in rate
    assert "authoritative: false" in rate
    assert "kv.get(key)" not in rate
    assert "kv.put(key, String(count)" not in rate


def test_dev_proxy_is_loopback_plus_exact_origin_and_prebuffer_content_length_gate():
    src = DEV_PROXY.read_text(encoding="utf-8")
    assert '"ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"' in src
    assert '"Access-Control-Allow-Origin": "*"' not in src
    assert "if length > MAX_BODY_BYTES:" in src
    assert src.index("if length > MAX_BODY_BYTES:") < src.index("self.rfile.read(length)")


def test_discovery_default_reports_same_cors_policy_as_runtime(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    # Source-level assertion prevents a reload from disturbing the shared app module.
    src = (PROXY / "_utils" / "_shared_logic.py").read_text(encoding="utf-8")
    assert 'os.environ.get("ALLOWED_ORIGINS", "")' in src


def test_worker_bundled_wrangler_main_matches_bundled_file():
    wrangler = (ROOT / "_cf_worker" / "wrangler.toml").read_text(encoding="utf-8")
    assert 'main            = "index.js"' in wrangler
    assert (ROOT / "_cf_worker" / "index.js").is_file()
    assert 'compatibility_date = "2026-08-01"' in wrangler
