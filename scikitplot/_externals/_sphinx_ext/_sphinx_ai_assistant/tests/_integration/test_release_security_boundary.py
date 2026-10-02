"""Run 11: B05/B06 browser-origin, identity, body and abuse-gate contracts."""
from __future__ import annotations

from .._paths import MAINTENANCE_ROOT, RUNTIME_ROOT

import ast
import asyncio
import importlib
import pathlib
import sys

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

ROOT = RUNTIME_ROOT
PROXY = ROOT / "_hf_spaces_proxy"
MODEL = ROOT / "_hf_spaces_model"
WORKER = ROOT / "_cf_worker" / "index.js"
DEV_PROXY = MAINTENANCE_ROOT / "_maintenance" / "tools" / "dev_proxy.py"
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

    # Every rate-limit decision must take its identity from ``_client_ip``,
    # the one function allowed to read a forwarded header. This is checked on
    # the syntax tree, for every call there is. The earlier form searched the
    # text before four named log events; it rotted when the legacy feedback
    # route was replaced by page feedback and feedback review, and it never
    # covered the routes added since.
    tree = ast.parse((PROXY / "app.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def calls(node: ast.AST) -> set[str]:
        return {
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }

    def reads_client_ip(expression: ast.AST) -> bool:
        """``_client_ip(request)``, or an identity helper that is built on it."""
        for call in ast.walk(expression):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
                continue
            if call.func.id == "_client_ip":
                return True
            helper = functions.get(call.func.id)
            if (
                helper is not None
                and call.func.id.endswith("_rate_identity")
                and "_client_ip" in calls(helper)
            ):
                return True
        return False

    readers = sorted(
        name
        for name, node in functions.items()
        if "x-forwarded-for" in ast.unparse(node).lower() or "request.client" in ast.unparse(node)
    )
    assert readers == ["_client_ip"], readers

    seen_scopes: set[str] = set()
    for name, function in functions.items():
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_consume_rate_limit"
            ):
                continue
            identity = call.args[2]
            if isinstance(identity, ast.Name):
                sources = [
                    assign.value
                    for assign in ast.walk(function)
                    if isinstance(assign, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == identity.id
                        for target in assign.targets
                    )
                ]
                assert sources, f"{name}: rate identity {identity.id!r} is not assigned in the route"
                assert all(reads_client_ip(source) for source in sources), name
            else:
                assert reads_client_ip(identity), name
            scopes = [kw.value.value for kw in call.keywords if kw.arg == "scope"]
            assert len(scopes) == 1, f"{name}: a rate-limit call names exactly one scope"
            seen_scopes.add(scopes[0])
    assert {"chat", "contribution", "share", "page-feedback", "feedback-review"} <= seen_scopes
    # The legacy Assistant-feedback route is retired, not renamed.
    assert "feedback" not in seen_scopes


def test_all_hf_public_body_routes_use_the_streaming_gate():
    src = (PROXY / "app.py").read_text(encoding="utf-8")
    assert "await request.body()" not in src
    assert "request.stream()" in src
    assert "return await _read_limited_body(request, MAX_BODY_BYTES" in src
    assert 'await _read_limited_body(request, CONTRIBUTION_MAX_BODY_BYTES' in src
    assert 'await _read_limited_body(request, SHARE_MAX_BODY_BYTES' in src
    assert "CHAT_RATE_LIMIT_PER_HOUR" in src

    # The two feedback surfaces that replaced the legacy route each have a
    # bound of their own, and no route reads a body any other way. Checked on
    # the syntax tree so a re-wrapped call is still the same call.
    tree = ast.parse(src)
    limits = {
        ast.unparse(call.args[1])
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_read_limited_body"
    }
    assert {
        "MAX_BODY_BYTES",
        "CONTRIBUTION_MAX_BODY_BYTES",
        "SHARE_MAX_BODY_BYTES",
        "PAGE_FEEDBACK_MAX_BODY_BYTES",
        "FEEDBACK_REVIEW_MAX_BODY_BYTES",
    } <= limits
    assert "FEEDBACK_MAX_BODY_BYTES" not in limits
    unbounded = sorted(
        {
            call.func.attr
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "request"
            and call.func.attr in {"body", "json", "form"}
        }
    )
    assert unbounded == [], unbounded


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
    # The Worker's feedback route is retired; page feedback is rate-limited on
    # the proxy. A limit left behind here would be a limit on nothing.
    assert "FEEDBACK_RATE_LIMIT" not in src
    assert "url.pathname === '/v1/feedback'" not in src
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
