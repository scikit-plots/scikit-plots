from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROXY = ROOT / "_hf_spaces_proxy"
SECURITY = PROXY / "security"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


verify_mod = _load("run20_verify_release_evidence", SECURITY / "verify_release_evidence.py")
probe_mod = _load("run20_probe_redis_authority", SECURITY / "probe_redis_authority.py")
gate_mod = _load("run20_verify_release_gate", SECURITY / "verify_release_gate.py")
subjects_mod = _load("run20_release_subjects", SECURITY / "release_subjects.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(root: Path, name: str, data: object) -> dict[str, object]:
    path = root / name
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, sort_keys=True))
    else:
        path.write_text(str(data))
    return {
        "path": name,
        "sha256": _sha(path),
        "status": "pass",
        "tool": {"name": "test-tool", "version": "1.0"},
    }


def _evidence(tmp_path: Path, now: datetime) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_digest = "sha256:" + "a" * 64
    dep = _write_artifact(tmp_path, "pip-audit.json", {"vulnerabilities": []})
    scan = _write_artifact(tmp_path, "image-scan.json", {"high": 0, "critical": 0})
    sbom = _write_artifact(tmp_path, "image.cdx.json", {"bomFormat": "CycloneDX", "specVersion": "1.6", "components": []})
    provenance_doc = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": "image", "digest": {"sha256": "a" * 64}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {"buildDefinition": {"resolvedDependencies": [{"uri": "base-image", "digest": {"sha256": "b" * 64}}]}, "runDetails": {}},
    }
    provenance = _write_artifact(tmp_path, "provenance.json", provenance_doc)
    provenance["predicateType"] = "https://slsa.dev/provenance/v1"
    provenance["signatureVerified"] = True
    sig = _write_artifact(tmp_path, "signature-verification.json", {"verified": True, "subject": image_digest})
    dep["subject"] = "sha256:" + _sha(PROXY / "requirements.lock")
    scan["subject"] = image_digest
    sbom["subject"] = image_digest
    provenance["subject"] = image_digest
    sig["subject"] = image_digest

    def fmt(dt):
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    recent_backup = fmt(now - timedelta(days=2))
    doc = {
        "schemaVersion": 1,
        "release": {
            "releaseId": "run20-test",
            "generatedAt": fmt(now - timedelta(minutes=5)),
            "expiresAt": fmt(now + timedelta(hours=12)),
            "proxyVersion": "7.3.0",
            "targetPlatform": "linux/amd64",
        },
        "source": {
            "requirementsLockSha256": _sha(PROXY / "requirements.lock"),
            "pythonSbomSha256": _sha(PROXY / "security/python-runtime.cdx.json"),
            "runtimeSourceSha256": verify_mod._runtime_source_sha256(),
            "baseImageIndexDigest": verify_mod.SUPPLY["base_image"]["index_digest"],
            "baseImageManifestDigest": "sha256:" + "b" * 64,
        },
        "image": {"digest": image_digest, "platform": "linux/amd64"},
        "artifacts": {
            "dependencyScan": dep,
            "imageScan": scan,
            "imageSbom": sbom,
            "provenance": provenance,
            "signatureVerification": sig,
        },
        "redis": {
            "rateLimit": {
                "tlsVerified": True,
                "nonDefaultIdentity": True,
                "leastPrivilegeReviewed": True,
                "persistenceVerified": False,
                "replicationVerified": False,
                "backupRestoreTestedAt": None,
            },
            "share": {
                "tlsVerified": True,
                "nonDefaultIdentity": True,
                "leastPrivilegeReviewed": True,
                "persistenceVerified": True,
                "replicationVerified": True,
                "backupRestoreTestedAt": recent_backup,
            },
            "contribution": {
                "tlsVerified": True,
                "nonDefaultIdentity": True,
                "leastPrivilegeReviewed": True,
                "persistenceVerified": True,
                "replicationVerified": True,
                "backupRestoreTestedAt": recent_backup,
            },
        },
        "logging": {
            "requestBodyLogging": False,
            "authorizationHeaderLogging": False,
            "capabilityHeaderLogging": False,
            "queryStringLogging": False,
            "wafBodyCapture": False,
            "apmBodyCapture": False,
            "thirdPartyTelemetryExport": False,
            "reviewedAt": fmt(now - timedelta(minutes=4)),
        },
        "riskExceptions": [],
    }
    target = tmp_path / "release-evidence.json"
    target.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return target


def _mutate(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))


def test_release_evidence_binds_exact_source_image_provenance_and_privacy(tmp_path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)
    path = _evidence(tmp_path, now)
    out = verify_mod.verify(path, now=now)
    assert out["ok"] is True
    assert out["image_digest"] == "sha256:" + "a" * 64
    assert out["logging_privacy"] == "verified"
    assert out["redis_planes"] == ["rateLimit", "share", "contribution"]
    serialized = json.dumps(out).lower()
    for forbidden in ("redis://", "rediss://", "password", "authorization", "hostname", "username"):
        assert forbidden not in serialized


def test_release_evidence_rejects_stale_wrong_source_and_artifact_tampering(tmp_path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)
    stale = _evidence(tmp_path / "stale", now)
    _mutate(stale, lambda d: d["release"].update(generatedAt="2026-08-20T00:00:00Z", expiresAt="2026-08-31T00:00:00Z"))
    with pytest.raises(verify_mod.EvidenceError, match="RELEASE_EVIDENCE_STALE"):
        verify_mod.verify(stale, now=now)

    source = _evidence(tmp_path / "source", now)
    _mutate(source, lambda d: d["source"].update(requirementsLockSha256="0" * 64))
    with pytest.raises(verify_mod.EvidenceError, match="LOCK_EVIDENCE_SOURCE_MISMATCH"):
        verify_mod.verify(source, now=now)

    tamper = _evidence(tmp_path / "tamper", now)
    (tamper.parent / "image-scan.json").write_text("tampered")
    with pytest.raises(verify_mod.EvidenceError, match="IMAGE_SCAN_HASH_MISMATCH"):
        verify_mod.verify(tamper, now=now)


def test_release_evidence_rejects_unbound_provenance_and_unverified_signature(tmp_path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)
    wrong = _evidence(tmp_path / "wrong", now)
    prov_path = wrong.parent / "provenance.json"
    prov = json.loads(prov_path.read_text())
    prov["subject"][0]["digest"]["sha256"] = "c" * 64
    prov_path.write_text(json.dumps(prov, sort_keys=True))
    _mutate(wrong, lambda d: d["artifacts"]["provenance"].update(sha256=_sha(prov_path)))
    with pytest.raises(verify_mod.EvidenceError, match="PROVENANCE_SUBJECT_IMAGE_MISMATCH"):
        verify_mod.verify(wrong, now=now)

    unsigned = _evidence(tmp_path / "unsigned", now)
    _mutate(unsigned, lambda d: d["artifacts"]["provenance"].update(signatureVerified=False))
    with pytest.raises(verify_mod.EvidenceError, match="PROVENANCE_SIGNATURE_NOT_VERIFIED"):
        verify_mod.verify(unsigned, now=now)


def test_release_evidence_rejects_logging_telemetry_and_redis_paper_claims(tmp_path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)
    telemetry = _evidence(tmp_path / "telemetry", now)
    _mutate(telemetry, lambda d: d["logging"].update(thirdPartyTelemetryExport=True))
    with pytest.raises(verify_mod.EvidenceError, match="LOGGING_THIRDPARTYTELEMETRYEXPORT_POLICY_MISMATCH"):
        verify_mod.verify(telemetry, now=now)

    default_user = _evidence(tmp_path / "default", now)
    _mutate(default_user, lambda d: d["redis"]["share"].update(nonDefaultIdentity=False))
    with pytest.raises(verify_mod.EvidenceError, match="REDIS_SHARE_DEFAULT_IDENTITY"):
        verify_mod.verify(default_user, now=now)

    stale_backup = _evidence(tmp_path / "backup", now)
    _mutate(stale_backup, lambda d: d["redis"]["contribution"].update(backupRestoreTestedAt="2025-01-01T00:00:00Z"))
    with pytest.raises(verify_mod.EvidenceError, match="REDIS_CONTRIBUTION_BACKUP_RESTORE_STALE"):
        verify_mod.verify(stale_backup, now=now)


def test_release_evidence_rejects_secret_like_values_path_escape_and_risk_waivers(tmp_path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)
    secret = _evidence(tmp_path / "secret", now)
    _mutate(secret, lambda d: d.update(note="rediss://user:password@cache.internal/0"))
    with pytest.raises(verify_mod.EvidenceError, match="EVIDENCE_SECRET_LIKE_VALUE_FORBIDDEN"):
        verify_mod.verify(secret, now=now)

    escape = _evidence(tmp_path / "escape", now)
    _mutate(escape, lambda d: d["artifacts"]["imageScan"].update(path="../image-scan.json"))
    with pytest.raises(verify_mod.EvidenceError, match="ARTIFACT_PATH_TRAVERSAL"):
        verify_mod.verify(escape, now=now)

    exception = _evidence(tmp_path / "exception", now)
    _mutate(exception, lambda d: d.update(riskExceptions=[{"id": "skip-scan"}]))
    with pytest.raises(verify_mod.EvidenceError, match="RISK_EXCEPTIONS_FORBIDDEN"):
        verify_mod.verify(exception, now=now)


class _FakeRedis:
    def ping(self):
        return True

    def execute_command(self, *args):
        assert args == ("ACL", "WHOAMI")
        return b"ai-proxy-runtime"

    def info(self, section):
        if section == "persistence":
            return {"aof_enabled": 1, "loading": 0, "rdb_last_save_time": 123456789}
        if section == "replication":
            return {"role": "master", "connected_slaves": 2, "master_repl_offset": 999999}
        raise AssertionError(section)


def test_redis_probe_is_explicit_sanitized_and_does_not_output_authority(monkeypatch):
    secret_url = "rediss://runtime-user:super-secret@cache.internal.example:6380/4"
    monkeypatch.setenv("TEST_REDIS_URL", secret_url)
    out = probe_mod.collect(plane="share", url_env="TEST_REDIS_URL", client=_FakeRedis())
    assert out["transport"]["tls"] is True
    assert out["nonDefaultIdentity"] is True
    assert out["aofPersistenceObserved"] is True
    assert out["replicationObserved"] is True
    serialized = json.dumps(out).lower()
    for forbidden in ("runtime-user", "super-secret", "cache.internal", "6380", "999999", "123456789", "rediss://"):
        assert forbidden not in serialized
    with pytest.raises(RuntimeError, match="URL_ENV_NAME_INVALID"):
        probe_mod.collect(plane="share", url_env="bad-name", client=_FakeRedis())


def test_release_evidence_policy_is_fail_closed_and_example_is_non_authoritative():
    policy = verify_mod.POLICY
    assert policy["max_age_hours"] <= 72
    assert policy["require_slsa_provenance"] is True
    assert policy["require_signature_verification"] is True
    assert policy["forbid_unexpired_risk_exceptions"] is True
    assert policy["logging"]["request_body_logging"] is False
    assert policy["logging"]["third_party_telemetry_export"] is False
    example = (SECURITY / "release-evidence.example.json").read_text()
    assert "example-only-not-production" in example
    assert "REPLACE_WITH_64_HEX" in example
    assert "rediss://" not in example



def test_combined_release_gate_requires_both_source_policy_and_bound_evidence(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    path = _evidence(tmp_path, now)
    out = gate_mod.verify(path)
    assert out["ok"] is True
    assert out["source_policy"]["release_evidence_policy"] is True
    assert out["release_evidence"]["image_digest"] == "sha256:" + "a" * 64


def test_evidence_and_artifact_symlinks_are_forbidden(tmp_path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)
    path = _evidence(tmp_path / "real", now)
    link = tmp_path / "evidence-link.json"
    try:
        link.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(verify_mod.EvidenceError, match="EVIDENCE_SYMLINK_FORBIDDEN"):
        verify_mod.verify(link, now=now)

    artifact_case = _evidence(tmp_path / "artifact", now)
    original = artifact_case.parent / "image-scan.json"
    moved = artifact_case.parent / "image-scan-real.json"
    original.rename(moved)
    original.symlink_to(moved.name)
    _mutate(artifact_case, lambda d: d["artifacts"]["imageScan"].update(sha256=_sha(moved)))
    with pytest.raises(verify_mod.EvidenceError, match="ARTIFACT_SYMLINK_FORBIDDEN"):
        verify_mod.verify(artifact_case, now=now)



def test_release_subject_printer_exposes_only_non_secret_content_addressed_inputs():
    out = subjects_mod.subjects()
    assert out["proxy_version"] == "7.3.0"
    assert out["target_platform"] == "linux/amd64"
    assert len(out["requirements_lock_sha256"]) == 64
    assert len(out["python_sbom_sha256"]) == 64
    assert len(out["runtime_source_sha256"]) == 64
    assert str(out["base_image_index_digest"]).startswith("sha256:")
    serialized = json.dumps(out).lower()
    for forbidden in ("redis://", "rediss://", "password", "authorization", "hostname", "username", "token", "cookie"):
        assert forbidden not in serialized



def test_release_evidence_rejects_wrong_proxy_version_unknown_fields_and_bad_tool(tmp_path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)

    version = _evidence(tmp_path / "version", now)
    _mutate(version, lambda d: d["release"].update(proxyVersion="6.8.0"))
    with pytest.raises(verify_mod.EvidenceError, match="RELEASE_PROXY_VERSION_MISMATCH"):
        verify_mod.verify(version, now=now)

    extra = _evidence(tmp_path / "extra", now)
    _mutate(extra, lambda d: d.update(comment="harmless but not schema-v1"))
    with pytest.raises(verify_mod.EvidenceError, match="EVIDENCE_SCHEMA_FIELDS_INVALID"):
        verify_mod.verify(extra, now=now)

    nested = _evidence(tmp_path / "nested", now)
    _mutate(nested, lambda d: d["release"].update(buildNumber="123"))
    with pytest.raises(verify_mod.EvidenceError, match="RELEASE_SCHEMA_INVALID"):
        verify_mod.verify(nested, now=now)

    tool = _evidence(tmp_path / "tool", now)
    _mutate(tool, lambda d: d["artifacts"]["imageScan"].update(tool={"name": "", "version": "1"}))
    with pytest.raises(verify_mod.EvidenceError, match="IMAGE_SCAN_TOOL_INVALID"):
        verify_mod.verify(tool, now=now)


def test_release_evidence_rejects_oversize_manifest_old_sbom_and_unresolved_base(tmp_path, monkeypatch):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)

    oversize = _evidence(tmp_path / "oversize", now)
    monkeypatch.setitem(verify_mod.POLICY, "max_manifest_bytes", 64)
    with pytest.raises(verify_mod.EvidenceError, match="EVIDENCE_MANIFEST_TOO_LARGE"):
        verify_mod.verify(oversize, now=now)
    monkeypatch.setitem(verify_mod.POLICY, "max_manifest_bytes", 262144)

    old_sbom = _evidence(tmp_path / "old-sbom", now)
    sbom_path = old_sbom.parent / "image.cdx.json"
    sbom = json.loads(sbom_path.read_text())
    sbom["specVersion"] = "1.5"
    sbom_path.write_text(json.dumps(sbom, sort_keys=True))
    _mutate(old_sbom, lambda d: d["artifacts"]["imageSbom"].update(sha256=_sha(sbom_path)))
    with pytest.raises(verify_mod.EvidenceError, match="IMAGE_SBOM_SPEC_VERSION_TOO_OLD"):
        verify_mod.verify(old_sbom, now=now)

    unresolved = _evidence(tmp_path / "unresolved", now)
    index_digest = verify_mod.SUPPLY["base_image"]["index_digest"]
    _mutate(unresolved, lambda d: d["source"].update(baseImageManifestDigest=index_digest))
    with pytest.raises(verify_mod.EvidenceError, match="BASE_IMAGE_MANIFEST_UNRESOLVED"):
        verify_mod.verify(unresolved, now=now)


def test_runtime_source_digest_includes_non_python_utils_files(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    for name in verify_mod.RUNTIME_SOURCE_FILES:
        (root / name).write_text(f"{name}\n")
    utils = root / "_utils"
    utils.mkdir()
    (utils / "module.py").write_text("VALUE = 1\n")
    extra = utils / "runtime-policy.dat"
    extra.write_text("policy-a\n")
    before = verify_mod._runtime_source_sha256(root)
    extra.write_text("policy-b\n")
    after = verify_mod._runtime_source_sha256(root)
    assert before != after


def test_proxy_version_ratchets_to_b39_release():
    shared_path = PROXY / "_utils/_shared_logic.py"
    assert 'PROXY_VERSION: str = "7.3.0"' in shared_path.read_text()
