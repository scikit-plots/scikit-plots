from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT, TESTS_ROOT

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import stat

import pytest

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
CI = ROOT / "_hf_spaces_proxy" / "ci"
SECURITY = ROOT / "_hf_spaces_proxy" / "security"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


attest = _load("run148_attestation", CI / "redis_chaos_attestation.py")
source_tree = _load("run148_source_tree", SECURITY / "source_tree.py")
run20 = _load(
    "run148_release_fixture", TESTS_ROOT / "_hf_spaces_proxy" / "security" / "test_verify_release_evidence.py"
)
verify = run20.verify_mod

REDIS7 = "redis:7.4.11-bookworm@sha256:71da9275c5f3fcb97d0fa0c8c5b36cc995327265420f17a04bfd544f458059f7"
REDIS8 = "redis:8.2.9-bookworm@sha256:7d1e4ce8b9395088377ab382d1f6cfdbd13b3690795198a0399ab8d683064d6d"
REVISION = "a" * 40


def _mutate(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))


def test_run148_source_tree_subject_binds_content_mode_and_ignores_test_caches(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    a = root / "a.py"
    b = root / "sub" / "b.txt"
    b.parent.mkdir()
    a.write_text("print('a')\n")
    b.write_text("b\n")
    first = source_tree.source_tree_sha256(root)

    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "ignored").write_text("ignored")
    assert source_tree.source_tree_sha256(root) == first

    b.write_text("changed\n")
    assert source_tree.source_tree_sha256(root) != first
    b.write_text("b\n")
    b.chmod((b.stat().st_mode & 0o777) ^ stat.S_IXUSR)
    assert source_tree.source_tree_sha256(root) != first


def test_run148_attestation_is_digest_pinned_revision_bound_and_outside_source_tree(tmp_path: Path):
    payload = attest.build_attestation(
        mode="cluster",
        redis_major=8,
        redis_image=REDIS8,
        source_revision=REVISION,
        generated_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
    )
    assert payload["predicateType"] == attest.PREDICATE_TYPE
    assert payload["subject"]["sourceRevision"] == REVISION
    assert payload["subject"]["sourceTreeSha256"] == source_tree.source_tree_sha256(ROOT)
    assert payload["redis"] == {"major": 8, "image": REDIS8, "mode": "cluster"}
    assert payload["result"] == {"status": "pass"}

    with pytest.raises(attest.AttestationError, match="REDIS_IMAGE_NOT_DIGEST_PINNED"):
        attest.build_attestation(
            mode="standalone",
            redis_major=7,
            redis_image="redis:7.4.11-bookworm",
            source_revision=REVISION,
        )
    with pytest.raises(attest.AttestationError, match="SOURCE_REVISION_INVALID"):
        attest.build_attestation(
            mode="standalone",
            redis_major=7,
            redis_image=REDIS7,
            source_revision="main",
        )
    with pytest.raises(attest.AttestationError, match="ATTESTATION_OUTPUT_INSIDE_SOURCE_TREE"):
        attest.write_attestation(
            ROOT / "_hf_spaces_proxy" / "ci" / "bad-output",
            mode="standalone",
            redis_major=7,
            redis_image=REDIS7,
            source_revision=REVISION,
        )


def test_run148_signature_record_binds_exact_attestation_hash_and_revision(tmp_path: Path):
    attestation = attest.write_attestation(
        tmp_path / "attestations",
        mode="standalone",
        redis_major=7,
        redis_image=REDIS7,
        source_revision=REVISION,
    )
    record = attest.write_signature_verification_record(
        attestation,
        tmp_path / "verification" / "redis7-standalone.signature-verification.json",
        source_revision=REVISION,
    )
    doc = json.loads(record.read_text())
    assert doc["verified"] is True
    assert doc["signerIdentityVerified"] is True
    assert doc["sourceRevision"] == REVISION
    assert doc["attestationSha256"] == attest._sha256(attestation)
    assert "signature" not in json.dumps(doc).lower()


def test_run148_release_evidence_rejects_wrong_tree_and_unsigned_chaos(tmp_path: Path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)
    wrong_tree = run20._evidence(tmp_path / "tree", now)
    _mutate(wrong_tree, lambda d: d["source"].update(sourceTreeSha256="0" * 64))
    with pytest.raises(verify.EvidenceError, match="SOURCE_TREE_EVIDENCE_MISMATCH"):
        verify.verify(wrong_tree, now=now)

    unsigned = run20._evidence(tmp_path / "unsigned", now)
    release = json.loads(unsigned.read_text())
    entry = release["redisChaos"]["redis7"]["standalone"]
    signature_path = unsigned.parent / entry["signatureVerification"]["path"]
    signature = json.loads(signature_path.read_text())
    signature["signerIdentityVerified"] = False
    signature_path.write_text(json.dumps(signature, sort_keys=True))
    entry["signatureVerification"]["sha256"] = run20._sha(signature_path)
    unsigned.write_text(json.dumps(release, indent=2, sort_keys=True))
    with pytest.raises(verify.EvidenceError, match="REDIS_CHAOS_REDIS7_STANDALONE_SIGNER_IDENTITY_UNVERIFIED"):
        verify.verify(unsigned, now=now)


def test_run148_release_evidence_rejects_wrong_pinned_image_and_stale_attestation(tmp_path: Path):
    now = datetime(2026, 8, 30, 1, 0, tzinfo=timezone.utc)

    def rewrite_attestation(evidence: Path, *, major: int, mode: str, mutate) -> None:
        release = json.loads(evidence.read_text())
        entry = release["redisChaos"][f"redis{major}"][mode]
        att_path = evidence.parent / entry["attestation"]["path"]
        payload = json.loads(att_path.read_text())
        mutate(payload)
        att_path.write_text(json.dumps(payload, sort_keys=True))
        att_sha = run20._sha(att_path)
        entry["attestation"]["sha256"] = att_sha
        sig_path = evidence.parent / entry["signatureVerification"]["path"]
        sig = json.loads(sig_path.read_text())
        sig["attestationSha256"] = att_sha
        sig_path.write_text(json.dumps(sig, sort_keys=True))
        entry["signatureVerification"]["sha256"] = run20._sha(sig_path)
        entry["signatureVerification"]["subject"] = "sha256:" + att_sha
        evidence.write_text(json.dumps(release, indent=2, sort_keys=True))

    wrong_image = run20._evidence(tmp_path / "image", now)
    rewrite_attestation(
        wrong_image,
        major=8,
        mode="cluster",
        mutate=lambda p: p["redis"].update(image="redis:8.2.9-bookworm@sha256:" + "f" * 64),
    )
    with pytest.raises(verify.EvidenceError, match="REDIS_CHAOS_REDIS8_CLUSTER_IMAGE_MISMATCH"):
        verify.verify(wrong_image, now=now)

    stale = run20._evidence(tmp_path / "stale", now)
    rewrite_attestation(
        stale,
        major=7,
        mode="standalone",
        mutate=lambda p: p.update(generatedAt="2026-08-20T00:00:00Z"),
    )
    with pytest.raises(verify.EvidenceError, match="REDIS_CHAOS_REDIS7_STANDALONE_STALE"):
        verify.verify(stale, now=now)


def test_run148_ci_references_pin_redis_digests_and_github_signing_actions() -> None:
    dockerfile = (CI / "redis-chaos.Dockerfile").read_text()
    github = (CI / "github-actions.redis-chaos.reference.yml").read_text()
    circle = (CI / "circleci.redis-chaos.reference.yml").read_text()
    policy = (SECURITY / "release_evidence_policy.toml").read_text()

    for image in (REDIS7, REDIS8):
        assert image in github
        assert image in circle or image == REDIS8 and image in dockerfile
        assert image in policy
    assert "ARG REDIS_IMAGE=redis:8.2.9-bookworm@sha256:" in dockerfile
    assert "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6 # v4.2.2" in github
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1" in github
    assert "--attestation-dir /evidence" in github
    assert "--source-revision '${{ github.sha }}'" in github
    assert "gh attestation verify" in github
    assert '--source-digest "$GITHUB_SHA"' in github
    assert '--source-ref "$GITHUB_REF"' in github
    assert '--signer-workflow "$signer_workflow"' in github
    assert '--signer-digest "$GITHUB_SHA"' in github
    assert '--deny-self-hosted-runners' in github
    assert 'signer_workflow="${GITHUB_WORKFLOW_REF%@*}"' in github
    assert "signature-record" in github
    assert "redis:" in github and "@sha256:" in github
