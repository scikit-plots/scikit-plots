from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy_app = importlib.import_module("app")
redis_security = importlib.import_module("_utils._redis_security")
rate_limit = importlib.import_module("_utils._rate_limit")
share_store = importlib.import_module("_utils._share_store")
ledger = importlib.import_module("_utils._contribution_ledger")


def test_offline_supply_chain_verifier_and_locked_advisory_floors():
    proc = subprocess.run(
        [sys.executable, str(PROXY / "security/verify_supply_chain.py")],
        cwd=PROXY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["ok"] is True
    assert result["locked_packages"] == 30
    lock = (PROXY / "requirements.lock").read_text().lower()
    assert "click==8.3.3 " in lock
    assert "starlette==1.6.0 " in lock
    assert "click==8.1.8" not in lock
    assert "starlette==0.50.0" not in lock


def test_container_release_path_is_digest_locked_minimal_non_root_and_strict():
    docker = (PROXY / "Dockerfile").read_text()
    assert "python:3.11.16-slim-bookworm@sha256:0bee7276f83efd4a1ee05bbbf4281d95ed28e079220a9457f25a93e3f1e3c31b" in docker
    assert "--require-hashes" in docker
    assert "--only-binary=:all:" in docker
    assert "COPY --from=builder /opt/venv /opt/venv" in docker
    assert "USER 1000:1000" in docker
    assert "FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS builder" in docker
    assert "FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS runtime" in docker
    assert "DEPLOYMENT_PROFILE=strict" in docker
    assert "uvicorn[standard]" not in docker.lower()
    assert "fastapi[standard]" not in docker.lower()
    # Installer tooling belongs to the builder, not the application runtime.
    assert "/usr/local/lib/python3.11/site-packages/pip*" in docker
    direct = [
        line.strip() for line in (PROXY / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(direct) == 5
    assert all("==" in line and not any(x in line for x in (">", "<", "~=", "[", "]")) for line in direct)


def test_docker_build_context_and_read_only_reference_are_fail_closed():
    ignore = (PROXY / ".dockerignore").read_text().splitlines()
    active = [x.strip() for x in ignore if x.strip() and not x.lstrip().startswith("#")]
    assert active[0] == "*"
    assert "!requirements.lock" in active
    assert "!_utils/**" in active
    compose = (PROXY / "docker-compose.hardened.reference.yml").read_text()
    for marker in (
        'user: "1000:1000"',
        "read_only: true",
        "cap_drop:",
        "- ALL",
        "no-new-privileges:true",
        "pids_limit: 256",
        "DEPLOYMENT_PROFILE: strict",
        "/tmp:rw,nosuid,nodev",
    ):
        assert marker in compose


def test_python_sbom_is_exactly_the_lock_not_a_claim_about_the_os_image():
    sbom = json.loads((PROXY / "security/python-runtime.cdx.json").read_text())
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.6"
    assert len(sbom["components"]) == 30
    properties = {p["name"]: p["value"] for p in sbom["metadata"]["properties"]}
    assert properties["scikitplot:scope"] == "python-runtime-lock-only"
    assert "full image SBOM" in properties["scikitplot:note"]
    for component in sbom["components"]:
        assert component["scope"] == "required"
        assert component["purl"].startswith("pkg:pypi/")
        assert len(component["hashes"][0]["content"]) == 64


def test_redis_transport_policy_rejects_downgrades_and_forces_verification():
    with pytest.raises(redis_security.RedisSecurityError, match="REDIS_TLS_REQUIRED"):
        redis_security.validate_redis_url("redis://cache.internal:6379/0", require_tls=True)
    for url in (
        "rediss://cache.internal:6379/0?ssl_cert_reqs=none",
        "rediss://cache.internal:6379/0?ssl_check_hostname=false",
        "rediss://cache.internal:6379/0#fragment",
    ):
        with pytest.raises(redis_security.RedisSecurityError):
            redis_security.validate_redis_url(url, require_tls=True)
    policy, kwargs = redis_security.redis_connection_kwargs(
        "rediss://user:secret@cache.internal:6380/2",
        require_tls=True,
        socket_timeout_seconds=99,
    )
    assert policy.tls is True and policy.database == 2
    assert policy.manifest()["transport"] == "tls_verified"
    assert kwargs["ssl_cert_reqs"] == "required"
    assert kwargs["ssl_check_hostname"] is True
    assert kwargs["socket_timeout"] == 10.0
    # Public manifest contains transport facts only, never connection authority.
    assert "cache.internal" not in json.dumps(policy.manifest())
    assert "secret" not in json.dumps(policy.manifest())


def test_every_redis_control_plane_obeys_the_same_tls_requirement():
    with pytest.raises(rate_limit.RateLimitBackendError, match="REDIS_TLS_REQUIRED"):
        rate_limit.RedisRateLimiter(
            "redis://cache.internal/0", identity_secret="x" * 32,
            require_tls=True, client=object(),
        )
    with pytest.raises(share_store.ShareStoreError, match="REDIS_TLS_REQUIRED"):
        share_store.RedisShareStore(
            "redis://cache.internal/0", key_prefix="test", max_entries=5,
            max_total_bytes=1024, require_tls=True, client=object(),
        )
    with pytest.raises(ledger.ContributionLedgerError, match="REDIS_TLS_REQUIRED"):
        ledger.RedisContributionLedger(
            "redis://cache.internal/0", key_secret="x" * 32, key_prefix="test",
            max_pending_entries=5, max_pending_bytes=1024, max_receipts=5,
            require_tls=True, client=object(),
        )


def test_strict_deployment_profile_fails_closed_without_leaking_deployment_identity(monkeypatch):
    monkeypatch.setattr(proxy_app, "_DEPLOYMENT_PROFILE_VALID", True)
    monkeypatch.setattr(proxy_app, "DEPLOYMENT_PROFILE", "strict")
    monkeypatch.setattr(proxy_app, "DEPLOYMENT_STRICT", True)
    monkeypatch.setattr(proxy_app, "REQUIRE_NON_ROOT", True)
    monkeypatch.setattr(proxy_app, "REDIS_REQUIRE_TLS", True)
    monkeypatch.setattr(proxy_app, "_is_non_root_process", lambda: False)
    assert proxy_app._deployment_policy_error() == "ROOT_PROCESS_FORBIDDEN"

    monkeypatch.setattr(proxy_app, "_is_non_root_process", lambda: True)
    monkeypatch.setattr(proxy_app, "_allowed_origins", ["*"])
    assert proxy_app._deployment_policy_error() == "CORS_WILDCARD_FORBIDDEN"

    monkeypatch.setattr(proxy_app, "_allowed_origins", ["https://docs.example.test"])
    monkeypatch.setattr(proxy_app, "SHARE_ALLOW_OPAQUE_ORIGIN", True)
    monkeypatch.setattr(proxy_app, "SHARE_ALLOW_OPAQUE_ORIGIN_WRITE", True)
    assert proxy_app._deployment_policy_error() == "OPAQUE_ORIGIN_WRITE_FORBIDDEN"

    monkeypatch.setattr(proxy_app, "SHARE_ALLOW_OPAQUE_ORIGIN_WRITE", False)
    # Strict mode may still permit the separately-scoped read-only compatibility.
    assert proxy_app._deployment_policy_error() == ""
    monkeypatch.setattr(proxy_app, "SHARE_ALLOW_OPAQUE_ORIGIN", False)
    assert proxy_app._deployment_policy_error() == ""
    status = proxy_app._deployment_public_status()
    assert status == {
        "profile": "strict",
        "strict": True,
        "non_root": True,
        "non_root_required": True,
        "redis_tls_required": True,
        "policy_ready": True,
    }
    serialized = json.dumps(status).lower()
    for forbidden in ("uid", "gid", "hostname", "cache.internal", "redis_url"):
        assert forbidden not in serialized


def test_proxy_version_retains_b38_minimum_after_later_security_runs():
    shared = importlib.import_module("_utils._shared_logic")
    version = tuple(int(part) for part in shared.PROXY_VERSION.split("."))
    assert version >= (6, 8, 0)
