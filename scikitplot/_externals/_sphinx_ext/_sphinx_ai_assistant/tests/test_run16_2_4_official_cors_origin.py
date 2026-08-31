"""Run 16.2.4+: built-in origins are safe defaults and downstream deployments can replace them explicitly."""
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
    assert "if (raw === '*') return ['*'];" in allowed_fn
    assert "default_allowed_origin_count" in src
    assert "default_allowed_origins_allowed" in src
    assert "env_semantics: _allowedOriginsMode(env)" in src



def test_readme_documents_builtin_and_downstream_origin_workflows():
    text = README.read_text(encoding="utf-8")
    assert "https://scikit-plots.github.io" in text
    assert "https://scikit-plots-learn.readthedocs.io" in text
    assert "ALLOWED_ORIGINS_MODE=replace" in text
    assert "ALLOWED_ORIGINS_MODE=additive" in text
    assert "TRAINING_DATASET_REPO" in text
    assert "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY" in text
    assert "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR" in text
    assert "HF_TOKEN" in text
    assert "Never use `ALLOWED_ORIGINS=*` in production" in text

def test_proxy_patch_version_is_bumped_for_deploy_verification():
    assert tuple(int(part) for part in shared.PROXY_VERSION.split(".")) >= (6, 8, 0)
