from __future__ import annotations

import importlib
import json
import os
import pathlib
import subprocess
import sys

from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROXY = ROOT / "_hf_spaces_proxy"
WORKER = ROOT / "_cf_worker" / "index.js"
WRANGLER = ROOT / "_cf_worker" / "wrangler.toml"
JS = ROOT / "_static" / "ai-assistant.js"
INIT = ROOT / "__init__.py"

if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy = importlib.import_module("app")
shared = importlib.import_module("_utils._shared_logic")


def test_proxy_version_ratchets_for_runtime_isolation():
    assert shared.PROXY_VERSION == "7.3.0"
    assert 'PROXY_VERSION: str = "7.3.0"' in (PROXY / "_utils" / "_shared_logic.py").read_text()


def test_opaque_origin_read_and_write_authority_are_separate_in_fresh_process():
    script = r'''
import json
from fastapi.testclient import TestClient
import app
with TestClient(app.app) as client:
    viewer = client.get('/v1/share', headers={'Origin': 'null'})
    read = client.post('/v1/share/read', headers={'Origin': 'null'}, json={'shareId':'0'*32})
    create = client.post('/v1/share', headers={'Origin': 'null'}, json={})
    create_preflight = client.options('/v1/share', headers={
        'Origin':'null', 'Access-Control-Request-Method':'POST',
        'Access-Control-Request-Headers':'content-type'})
    print(json.dumps({
      'read_flag': app.SHARE_ALLOW_OPAQUE_ORIGIN,
      'write_flag': app.SHARE_ALLOW_OPAQUE_ORIGIN_WRITE,
      'viewer': viewer.status_code,
      'read': read.status_code,
      'create': create.status_code,
      'preflight': create_preflight.status_code,
      'health': client.get('/health').json()['cors'],
    }))
'''
    env = os.environ.copy()
    env["SHARE_ALLOW_OPAQUE_ORIGIN"] = "true"
    env.pop("SHARE_ALLOW_OPAQUE_ORIGIN_WRITE", None)
    proc = subprocess.run([sys.executable, "-c", script], cwd=PROXY, env=env,
                          text=True, capture_output=True, check=True)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["read_flag"] is True
    assert out["write_flag"] is False
    assert out["viewer"] == 200
    assert out["read"] != 403
    assert out["create"] == 403
    assert out["preflight"] == 403
    assert out["health"]["share_opaque_origin_allowed"] is True
    assert out["health"]["share_opaque_origin_write_allowed"] is False


def test_opaque_origin_write_requires_second_opt_in_but_strict_refuses_it(monkeypatch):
    monkeypatch.setattr(proxy, "SHARE_ALLOW_OPAQUE_ORIGIN", True)
    monkeypatch.setattr(proxy, "SHARE_ALLOW_OPAQUE_ORIGIN_WRITE", True)
    monkeypatch.setattr(proxy, "DEPLOYMENT_STRICT", True)
    assert proxy._deployment_policy_error() == "OPAQUE_ORIGIN_WRITE_FORBIDDEN"


def test_share_viewer_denies_framing_and_sensitive_browser_permissions():
    with TestClient(proxy.app) as client:
        r = client.get("/v1/share")
    assert r.status_code == 200
    assert r.headers["x-frame-options"].upper() == "DENY"
    csp = r.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp
    permissions = r.headers["permissions-policy"]
    for feature in ("camera=()", "microphone=()", "geolocation=()", "payment=()", "usb=()"):
        assert feature in permissions


def test_worker_matches_opaque_origin_split_and_viewer_headers():
    src = WORKER.read_text()
    assert "function _opaqueShareRequestMode(request)" in src
    assert "SHARE_ALLOW_OPAQUE_ORIGIN_WRITE" in src
    assert "mode === 'write'" in src
    assert "X-Frame-Options': 'DENY'" in src
    assert "Permissions-Policy" in src
    wrangler = WRANGLER.read_text()
    assert 'SHARE_ALLOW_OPAQUE_ORIGIN = "false"' in wrangler
    assert 'SHARE_ALLOW_OPAQUE_ORIGIN_WRITE = "false"' in wrangler


def test_runtime_bearer_tokens_are_site_owner_opt_in_and_default_off():
    js = JS.read_text()
    py = INIT.read_text()
    assert "allowRuntimeTokens" in js
    assert "if (!_runtimeTokensAllowed()) return '';" in js
    assert "_runtimeTokensAllowed() && typeof tok === 'string'" in js
    assert '"allowRuntimeTokens": _cfg_bool(' in py
    assert '"ai_assistant_allow_runtime_tokens", False' in py
    assert 'app.add_config_value("ai_assistant_allow_runtime_tokens", False, "html")' in py


def test_private_event_bus_is_the_internal_lifecycle_chokepoint():
    src = JS.read_text()
    assert "var _assistantEvents = (function ()" in src
    assert "function _dispatchAssistantEvent(event)" in src
    assert "The only deliberate crossing from the private bus. In B41" in src
    # Full-model edit information is still usable internally but the public
    # projector must not return it.
    assert "case 'ai-assistant-model-edit':" in src
    assert "return { isCustom: !!detail.isCustom };" in src
    assert "case 'ai-assistant:profile-changed':" in src
    profile_projection = src.split("case 'ai-assistant:profile-changed':", 1)[1].split("case 'ai-assistant-conversation-reset':", 1)[0]
    assert "activeKey" not in profile_projection
    # No active production listener should use document as the internal bus.
    assert "document.addEventListener('ai-assistant-model-change'" not in src
    assert "document.addEventListener('ai-assistant-open-contribution'" not in src


def test_page_integration_permission_is_v2_and_old_feedback_key_does_not_migrate_authority():
    src = JS.read_text()
    assert "var _FEEDBACK_DOM_CONSENT_VERSION = '2.0.0';" in src
    assert "var _FEEDBACK_DOM_PREF_KEY = 'ai-assistant-page-integration-consent';" in src
    assert "ai-assistant-feedback-page-integration-consent" not in src
    assert "assistant lifecycle events stay on the private internal event bus" in src
