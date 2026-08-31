"""Run 16.2.4: official documentation origin cannot be displaced by env overrides."""
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
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy = importlib.import_module("app")
shared = importlib.import_module("_utils._shared_logic")


def test_hf_env_adds_origins_without_displacing_official_docs_origin():
    assert proxy._build_allowed_origins("") == ["https://scikit-plots.github.io"]
    assert proxy._build_allowed_origins("https://docs.example.test") == [
        "https://scikit-plots.github.io",
        "https://docs.example.test",
    ]
    assert proxy._build_allowed_origins(
        "https://docs.example.test,https://scikit-plots.github.io"
    ) == ["https://scikit-plots.github.io", "https://docs.example.test"]
    assert proxy._build_allowed_origins("*") == ["*"]


def test_hf_rejects_malformed_configured_origins_but_keeps_official_origin():
    assert proxy._build_allowed_origins("https://example.test/path,javascript:alert(1)") == [
        "https://scikit-plots.github.io"
    ]
    assert proxy._normalise_browser_origin("https://SCIKIT-PLOTS.GITHUB.IO/") == "https://scikit-plots.github.io"


def test_hf_health_exposes_privacy_safe_cors_deploy_diagnostics():
    with TestClient(proxy.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == shared.PROXY_VERSION
    assert body["cors"] == {
        "official_docs_origin": "https://scikit-plots.github.io",
        "official_docs_origin_allowed": True,
        "wildcard": False,
        "allowed_origin_count": len(proxy._allowed_origins),
        "env_semantics": "additive",
        "share_opaque_origin_allowed": proxy.SHARE_ALLOW_OPAQUE_ORIGIN,
        "share_opaque_origin_write_allowed": bool(
            proxy.SHARE_ALLOW_OPAQUE_ORIGIN and proxy.SHARE_ALLOW_OPAQUE_ORIGIN_WRITE
        ),
    }
    # Do not publish custom origin values in the public diagnostic.
    assert "allowed_origins" not in body["cors"]


def test_official_docs_origin_passes_fresh_runtime_even_with_env_override():
    script = r"""
import json
from fastapi.testclient import TestClient
import app
with TestClient(app.app) as client:
    out = {}
    for name, origin in [
        ('official', 'https://scikit-plots.github.io'),
        ('extra', 'https://docs.example.test'),
        ('denied', 'https://attacker.example'),
    ]:
        r = client.get('/health', headers={'Origin': origin})
        out[name] = [r.status_code, r.headers.get('access-control-allow-origin')]
print(json.dumps(out))
"""
    env = os.environ.copy()
    env['ALLOWED_ORIGINS'] = 'https://docs.example.test'
    proc = subprocess.run(
        [sys.executable, '-c', script], cwd=PROXY, env=env, text=True,
        capture_output=True, check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result['official'] == [200, 'https://scikit-plots.github.io']
    assert result['extra'] == [200, 'https://docs.example.test']
    assert result['denied'] == [403, None]


def test_worker_cors_env_is_additive_and_health_reports_safe_status():
    src = WORKER.read_text(encoding="utf-8")
    assert 'const OFFICIAL_ALLOWED_ORIGINS = Object.freeze([DEFAULT_ALLOWED_ORIGINS]);' in src
    allowed_fn = src[src.index("function _allowedOrigins(env)"):src.index("function _originAllowed(request, env)")]
    assert "const merged = [...OFFICIAL_ALLOWED_ORIGINS]" in allowed_fn
    assert "env.ALLOWED_ORIGINS || ''" in allowed_fn
    assert "if (raw === '*') return ['*'];" in allowed_fn
    assert "official_docs_origin_allowed" in src
    assert "env_semantics: 'additive'" in src


def test_proxy_patch_version_is_bumped_for_deploy_verification():
    assert tuple(int(part) for part in shared.PROXY_VERSION.split(".")) >= (6, 8, 0)
