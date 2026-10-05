"""Run 16.2.4+: built-in origins are safe defaults and downstream deployments can replace them explicitly."""
from __future__ import annotations

from ...._paths import RUNTIME_ROOT

import importlib
import json
import os
import pathlib
import re
import subprocess
import sys

from fastapi.testclient import TestClient

ROOT = RUNTIME_ROOT
PROXY = ROOT / "_hf_spaces_proxy"
WORKER = ROOT / "_cf_worker" / "index.js"
README = PROXY / "README.md"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy = importlib.import_module("app")
shared = importlib.import_module("_utils._shared_logic")


def test_hf_default_origins_cover_both_current_scikit_plots_sites():
    defaults = [
        "https://scikit-plots.github.io",
        "https://scikit-plots-learn.readthedocs.io",
    ]
    assert list(proxy._DEFAULT_ALLOWED_ORIGINS) == defaults
    assert proxy._build_allowed_origins("") == defaults
    assert proxy._build_allowed_origins("https://docs.example.test") == [
        *defaults,
        "https://docs.example.test",
    ]
    assert proxy._build_allowed_origins(
        "https://docs.example.test,https://scikit-plots.github.io"
    ) == [*defaults, "https://docs.example.test"]
    assert proxy._build_allowed_origins("*") == ["*"]


def test_hf_replace_mode_supports_downstream_open_source_sites_without_builtins():
    assert proxy._build_allowed_origins(
        "https://docs.example.test,https://learn.example.test", mode="replace"
    ) == ["https://docs.example.test", "https://learn.example.test"]
    assert proxy._build_allowed_origins("", mode="replace") == []
    # Invalid mode fails safely back to additive defaults.
    assert proxy._build_allowed_origins("https://docs.example.test", mode="nonsense") == [
        *proxy._DEFAULT_ALLOWED_ORIGINS,
        "https://docs.example.test",
    ]


def test_hf_rejects_malformed_configured_origins_but_keeps_additive_defaults():
    assert proxy._build_allowed_origins(
        "https://example.test/path,javascript:alert(1)"
    ) == list(proxy._DEFAULT_ALLOWED_ORIGINS)
    assert proxy._build_allowed_origins(
        "https://example.test/path,javascript:alert(1)", mode="replace"
    ) == []
    assert proxy._normalise_browser_origin(
        "https://SCIKIT-PLOTS.GITHUB.IO/"
    ) == "https://scikit-plots.github.io"


def test_hf_health_exposes_privacy_safe_cors_deploy_diagnostics():
    with TestClient(proxy.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == shared.PROXY_VERSION
    wildcard = proxy._allowed_origins == ["*"]
    primary = proxy._DEFAULT_ALLOWED_ORIGINS[0]
    assert body["cors"] == {
        "official_docs_origin": primary,
        "official_docs_origin_allowed": (wildcard or primary in proxy._allowed_origins),
        "default_allowed_origin_count": len(proxy._DEFAULT_ALLOWED_ORIGINS),
        "default_allowed_origins_allowed": (
            wildcard
            or all(origin in proxy._allowed_origins for origin in proxy._DEFAULT_ALLOWED_ORIGINS)
        ),
        "wildcard": wildcard,
        "allowed_origin_count": None if wildcard else len(proxy._allowed_origins),
        "local_dev_origin_count": len(proxy._local_dev_origins),
        "env_semantics": proxy.ALLOWED_ORIGINS_MODE,
        "share_opaque_origin_allowed": proxy.SHARE_ALLOW_OPAQUE_ORIGIN,
        "share_opaque_origin_write_allowed": bool(
            proxy.SHARE_ALLOW_OPAQUE_ORIGIN and proxy.SHARE_ALLOW_OPAQUE_ORIGIN_WRITE
        ),
    }
    # Do not publish deployment-specific origin values in the public diagnostic.
    assert "allowed_origins" not in body["cors"]


def test_both_builtin_origins_pass_fresh_runtime_with_additive_env_override():
    script = r"""
import json
from fastapi.testclient import TestClient
import app
with TestClient(app.app) as client:
    out = {}
    for name, origin in [
        ('main_docs', 'https://scikit-plots.github.io'),
        ('learn_docs', 'https://scikit-plots-learn.readthedocs.io'),
        ('extra', 'https://docs.example.test'),
        ('denied', 'https://attacker.example'),
    ]:
        r = client.get('/health', headers={'Origin': origin})
        out[name] = [r.status_code, r.headers.get('access-control-allow-origin')]
print(json.dumps(out))
"""
    env = os.environ.copy()
    env['ALLOWED_ORIGINS'] = 'https://docs.example.test'
    env['ALLOWED_ORIGINS_MODE'] = 'additive'
    proc = subprocess.run(
        [sys.executable, '-c', script], cwd=PROXY, env=env, text=True,
        capture_output=True, check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result['main_docs'] == [200, 'https://scikit-plots.github.io']
    assert result['learn_docs'] == [200, 'https://scikit-plots-learn.readthedocs.io']
    assert result['extra'] == [200, 'https://docs.example.test']
    assert result['denied'] == [403, None]


def test_replace_mode_gives_downstream_site_complete_origin_authority():
    script = r"""
import json
from fastapi.testclient import TestClient
import app
with TestClient(app.app) as client:
    out = {}
    for name, origin in [
        ('custom', 'https://docs.example.test'),
        ('builtin', 'https://scikit-plots.github.io'),
    ]:
        r = client.get('/health', headers={'Origin': origin})
        out[name] = [r.status_code, r.headers.get('access-control-allow-origin')]
print(json.dumps(out))
"""
    env = os.environ.copy()
    env['ALLOWED_ORIGINS'] = 'https://docs.example.test'
    env['ALLOWED_ORIGINS_MODE'] = 'replace'
    proc = subprocess.run(
        [sys.executable, '-c', script], cwd=PROXY, env=env, text=True,
        capture_output=True, check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result['custom'] == [200, 'https://docs.example.test']
    assert result['builtin'] == [403, None]


def test_local_dev_origin_builder_is_loopback_only_and_exact():
    assert proxy._build_local_dev_origins(
        "http://127.0.0.1:8000,http://localhost:8000,http://[::1]:8000"
    ) == [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://[::1]:8000",
    ]
    assert proxy._build_local_dev_origins(
        "https://docs.example.test,http://127.0.0.1.evil.example:8000,*,http://localhost:8000/path"
    ) == []
    assert proxy._merge_local_dev_origins(
        proxy._build_allowed_origins(""),
        proxy._build_local_dev_origins("http://127.0.0.1:8000"),
    ) == [*proxy._DEFAULT_ALLOWED_ORIGINS, "http://127.0.0.1:8000"]
    assert proxy._merge_local_dev_origins(
        proxy._build_allowed_origins("https://docs.example.test", mode="replace"),
        proxy._build_local_dev_origins("http://localhost:8000"),
    ) == ["https://docs.example.test", "http://localhost:8000"]
    assert proxy._merge_local_dev_origins(
        ["*"], proxy._build_local_dev_origins("http://127.0.0.1:8000")
    ) == ["*"]


def test_local_dev_origin_allows_preflight_and_chat_post_without_wildcard():
    script = r"""
import json
from fastapi.testclient import TestClient
import app
origin = 'http://127.0.0.1:8000'
payload = {
    'contract': 'scikitplot-chat-v1',
    'model': 'stub/echo',
    'user_message': 'local dev CORS smoke',
    'context': {'page_text': '', 'page_descriptor': 'Local Learn'},
    'max_tokens': 128,
    'stream': False,
}
with TestClient(app.app) as client:
    preflight = client.options('/v1/chat/completions', headers={
        'Origin': origin,
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type',
    })
    posted = client.post('/v1/chat/completions', headers={'Origin': origin}, json=payload)
    denied = client.get('/health', headers={'Origin': 'http://127.0.0.1.evil.example:8000'})
    print(json.dumps({
        'preflight': [preflight.status_code, preflight.headers.get('access-control-allow-origin')],
        'posted': [posted.status_code, posted.headers.get('access-control-allow-origin'), posted.headers.get('x-stub-model')],
        'denied': [denied.status_code, denied.headers.get('access-control-allow-origin')],
        'cors': client.get('/health').json()['cors'],
    }))
"""
    env = os.environ.copy()
    env['LOCAL_DEV_ORIGINS'] = 'http://127.0.0.1:8000'
    env['ALLOWED_ORIGINS_MODE'] = 'additive'
    proc = subprocess.run(
        [sys.executable, '-c', script], cwd=PROXY, env=env, text=True,
        capture_output=True, check=True,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result['preflight'] == [200, 'http://127.0.0.1:8000']
    assert result['posted'] == [200, 'http://127.0.0.1:8000', 'true']
    assert result['denied'] == [403, None]
    assert result['cors']['wildcard'] is False
    assert result['cors']['local_dev_origin_count'] == 1


def test_strict_deployment_accepts_exact_loopback_local_dev_origins(monkeypatch):
    monkeypatch.setattr(proxy, "_DEPLOYMENT_PROFILE_VALID", True)
    monkeypatch.setattr(proxy, "REQUIRE_NON_ROOT", False)
    monkeypatch.setattr(proxy, "DEPLOYMENT_STRICT", True)
    monkeypatch.setattr(proxy, "SHARE_ALLOW_OPAQUE_ORIGIN_WRITE", False)
    monkeypatch.setattr(
        proxy,
        "_allowed_origins",
        proxy._merge_local_dev_origins(
            proxy._build_allowed_origins(""),
            proxy._build_local_dev_origins("http://127.0.0.1:8000"),
        ),
    )
    assert proxy._deployment_policy_error() == ""


def test_disallowed_origin_error_explains_narrow_local_dev_opt_in():
    with TestClient(proxy.app) as client:
        response = client.get('/health', headers={'Origin': 'http://127.0.0.1:8000'})
    # Module import has no LOCAL_DEV_ORIGINS in the normal test environment.
    assert response.status_code == 403
    detail = response.json()['detail']
    assert 'LOCAL_DEV_ORIGINS' in detail
    assert 'http://127.0.0.1:8000' in detail
    assert 'ALLOWED_ORIGINS' in detail


def test_worker_cors_matches_default_and_replace_semantics():
    src = WORKER.read_text(encoding="utf-8")
    assert 'const DEFAULT_ALLOWED_ORIGINS = Object.freeze([' in src
    assert '"https://scikit-plots.github.io"' in src
    assert '"https://scikit-plots-learn.readthedocs.io"' in src
    allowed_fn = src[src.index("function _allowedOriginsMode(env)"):src.index("function _originAllowed(request, env)")]
    assert "env.ALLOWED_ORIGINS_MODE || 'additive'" in allowed_fn
    assert "ALLOWED_ORIGIN_MODES.includes(mode) ? mode : 'additive'" in allowed_fn
    assert "_allowedOriginsMode(env) === 'additive' ? [...DEFAULT_ALLOWED_ORIGINS] : []" in allowed_fn
    assert "env.ALLOWED_ORIGINS || ''" in allowed_fn
    assert "env.LOCAL_DEV_ORIGINS || ''" in src
    assert "function _normaliseLocalDevOrigin(value)" in src
    assert "function _localDevOrigins(env)" in src
    assert "local_dev_origin_count" in src
    assert "if (raw === '*') return ['*'];" in allowed_fn
    assert "default_allowed_origin_count" in src
    assert "default_allowed_origins_allowed" in src
    assert "env_semantics: _allowedOriginsMode(env)" in src



def test_readme_documents_builtin_and_downstream_origin_workflows():
    text = README.read_text(encoding="utf-8")
    # Extract whole origins and compare exactly.  A substring test also passes
    # for `https://scikit-plots.github.io.evil.example`, so it cannot show that
    # the documented origin is the one the operator would copy.
    documented_origins = set(re.findall(r"https://[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", text))
    assert documented_origins.issuperset(
        {
            "https://scikit-plots.github.io",
            "https://scikit-plots-learn.readthedocs.io",
        }
    )
    assert "ALLOWED_ORIGINS_MODE=replace" in text
    assert "ALLOWED_ORIGINS_MODE=additive" in text
    assert "TRAINING_DATASET_REPO" in text
    assert "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY" in text
    assert "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR" in text
    assert "HF_TOKEN" in text
    assert "Never use `ALLOWED_ORIGINS=*` in production" in text
    assert "LOCAL_DEV_ORIGINS=http://127.0.0.1:8000,http://localhost:8000" in text
    assert "loopback-only" in text

def test_proxy_patch_version_is_bumped_for_deploy_verification():
    assert tuple(int(part) for part in shared.PROXY_VERSION.split(".")) >= (6, 8, 0)
