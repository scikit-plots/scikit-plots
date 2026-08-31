"""Run 16.2.6: HF public Share URL derivation and narrow opaque-origin opt-in."""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import subprocess
import sys

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROXY = ROOT / "_hf_spaces_proxy"
WORKER = ROOT / "_cf_worker" / "index.js"
WRANGLER = ROOT / "_cf_worker" / "wrangler.toml"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy = importlib.import_module("app")
shared = importlib.import_module("_utils._shared_logic")


def _http_internal_request() -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/share",
        "raw_path": b"/v1/share",
        "query_string": b"",
        "headers": [(b"host", b"internal:7860")],
        "client": ("127.0.0.1", 1234),
        "server": ("internal", 7860),
    }
    return Request(scope)


def test_hf_space_host_is_deployment_owned_https_share_base(monkeypatch):
    monkeypatch.setattr(proxy, "SPACE_HOST", "scikit-plots-ai.hf.space")
    monkeypatch.setattr(proxy, "SHARE_PUBLIC_BASE_URL", "")
    assert proxy._hf_space_public_base() == "https://scikit-plots-ai.hf.space"
    assert proxy._share_public_base(_http_internal_request()) == "https://scikit-plots-ai.hf.space"


def test_explicit_http_same_hf_space_is_upgraded_but_arbitrary_http_stays_blocked(monkeypatch):
    monkeypatch.setattr(proxy, "SPACE_HOST", "scikit-plots-ai.hf.space")
    monkeypatch.setattr(proxy, "SHARE_PUBLIC_BASE_URL", "http://scikit-plots-ai.hf.space")
    assert proxy._share_public_base(_http_internal_request()) == "https://scikit-plots-ai.hf.space"

    monkeypatch.setattr(proxy, "SHARE_PUBLIC_BASE_URL", "http://share.example.test")
    with pytest.raises(Exception) as exc:
        proxy._share_public_base(_http_internal_request())
    assert getattr(exc.value, "status_code", None) == 500


def test_hf_space_host_rejects_free_form_or_non_hf_values(monkeypatch):
    for value in (
        "https://scikit-plots-ai.hf.space",
        "scikit-plots-ai.hf.space/path",
        "attacker.example",
        "user@scikit-plots-ai.hf.space",
        "scikit-plots-ai.hf.space:443",
    ):
        monkeypatch.setattr(proxy, "SPACE_HOST", value)
        assert proxy._hf_space_public_base() == ""


def test_health_reports_patch_version_and_safe_share_cors_state():
    with TestClient(proxy.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == shared.PROXY_VERSION
    assert body["cors"]["share_opaque_origin_allowed"] is proxy.SHARE_ALLOW_OPAQUE_ORIGIN
    assert tuple(int(part) for part in shared.PROXY_VERSION.split(".")) >= (6, 8, 0)


def test_opaque_origin_share_opt_in_is_path_scoped_in_fresh_proxy_process():
    script = r'''
import json
from fastapi.testclient import TestClient
import app
with TestClient(app.app) as client:
    share_get = client.get('/v1/share', headers={'Origin': 'null'})
    share_preflight = client.options('/v1/share', headers={
        'Origin': 'null',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type',
    })
    health = client.get('/health', headers={'Origin': 'null'})
    print(json.dumps({
        'share_get': [share_get.status_code, share_get.headers.get('access-control-allow-origin')],
        'share_preflight': [share_preflight.status_code, share_preflight.headers.get('access-control-allow-origin')],
        'health': [health.status_code, health.headers.get('access-control-allow-origin')],
        'flag': app.SHARE_ALLOW_OPAQUE_ORIGIN,
    }))
'''
    env = os.environ.copy()
    env["SHARE_ALLOW_OPAQUE_ORIGIN"] = "true"
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=PROXY, env=env, text=True,
        capture_output=True, check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["flag"] is True
    assert result["share_get"] == [200, "null"]
    assert result["share_preflight"][0] == 403
    assert result["share_preflight"][1] is None
    # The CORS layer may add ACAO:null to the rejection so the local page can
    # read the safe error, but the non-Share handler itself must remain blocked.
    assert result["health"][0] == 403


def test_opaque_origin_remains_denied_by_default_in_fresh_proxy_process():
    script = r'''
import json
from fastapi.testclient import TestClient
import app
with TestClient(app.app) as client:
    r = client.get('/v1/share', headers={'Origin': 'null'})
    print(json.dumps([r.status_code, r.headers.get('access-control-allow-origin'), app.SHARE_ALLOW_OPAQUE_ORIGIN]))
'''
    env = os.environ.copy()
    env.pop("SHARE_ALLOW_OPAQUE_ORIGIN", None)
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=PROXY, env=env, text=True,
        capture_output=True, check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == [403, None, False]



def test_client_explains_local_file_opaque_origin_opt_in_on_network_cors_failure():
    src = (ROOT / "_static" / "ai-assistant.js").read_text(encoding="utf-8")
    assert "window.location.protocol === 'file:'" in src
    assert "SHARE_ALLOW_OPAQUE_ORIGIN=true" in src
    assert "Origin:null is not trusted by default" in src

def test_worker_has_same_share_only_opaque_origin_opt_in():
    src = WORKER.read_text(encoding="utf-8")
    assert "function _shareOpaqueOriginAllowed(request, env)" in src
    assert "env.SHARE_ALLOW_OPAQUE_ORIGIN" in src
    assert "function _opaqueShareRequestMode(request)" in src
    assert "env.SHARE_ALLOW_OPAQUE_ORIGIN_WRITE" in src
    assert "if (rawOrigin === 'null') return _shareOpaqueOriginAllowed(request, env);" in src
    assert "headers[\"Access-Control-Allow-Origin\"] = 'null';" in src
    assert "share_opaque_origin_allowed" in src
    assert 'SHARE_ALLOW_OPAQUE_ORIGIN = "false"' in WRANGLER.read_text(encoding="utf-8")
