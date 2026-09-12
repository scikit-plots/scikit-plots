from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SECURITY = ROOT / "_hf_spaces_proxy" / "security"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


publish = _load("run151_publish", SECURITY / "publish_release.py")
finalize = _load("run151_finalize", SECURITY / "finalize_publication.py")
REVISION = "a" * 40
PUB_NOW = datetime(2026, 9, 5, 4, 30, tzinfo=timezone.utc)
SIG_NOW = PUB_NOW + timedelta(minutes=1)
FINAL_NOW = PUB_NOW + timedelta(minutes=2)
VERIFIER_IDENTITY = "ci/post-publication-verifier"
SIGNER_IDENTITY = "https://github.com/scikit-plots/scikit-plots/.github/workflows/release.yml@refs/heads/main"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _promoted(tmp_path: Path) -> Path:
    root = tmp_path / "promoted"
    root.mkdir(parents=True)
    zip_path = root / "run151-test.zip"
    patch_path = root / "run150-to-run151.patch"
    zip_path.write_bytes(b"zip payload\n")
    patch_path.write_bytes(b"patch payload\n")
    sboms = {
        "pythonRuntime": {"name": "python-runtime.cdx.json", "sha256": "1" * 64},
        "image": {"name": "image.cdx.json", "sha256": "2" * 64, "subject": "sha256:" + "3" * 64},
    }
    statement = {
        "schemaVersion": 1,
        "predicateType": publish.PROMOTION_PREDICATE_TYPE,
        "generatedAt": "2026-09-05T04:00:00Z",
        "release": {"releaseId": "run151-test", "proxyVersion": "7.4.0", "sourceRevision": REVISION},
        "subject": {"sourceTreeSha256": "4" * 64, "evidenceSha256": "5" * 64},
        "artifacts": {
            "zip": {"name": zip_path.name, "sha256": _sha(zip_path), "size": zip_path.stat().st_size, "fileCount": 349},
            "patch": {"name": patch_path.name, "sha256": _sha(patch_path), "size": patch_path.stat().st_size},
            "baselineZip": {"name": "run150.zip", "sha256": "6" * 64, "size": 12345},
        },
        "sbomReferences": sboms,
        "verification": {
            "evidenceVerified": True,
            "patchRecreatesSourceTree": True,
            "zipRecreatesSourceTree": True,
            "deterministicZip": True,
        },
    }
    statement_path = root / "release-statement.json"
    statement_path.write_bytes(publish._canonical_bytes(statement))
    signature = {
        "schemaVersion": 1,
        "verified": True,
        "verifiedAt": "2026-09-05T04:01:00Z",
        "releaseStatementSha256": _sha(statement_path),
        "sourceRevision": REVISION,
        "signerIdentityVerified": True,
        "verifierEvidenceSha256": "7" * 64,
        "verifier": {"name": "release-attestation-verifier", "version": "1.0"},
    }
    signature_path = root / "release-statement.signature-verification.json"
    signature_path.write_bytes(publish._canonical_bytes(signature))
    payloads = [zip_path, patch_path, statement_path, signature_path]
    receipt = {
        "schemaVersion": 1,
        "finalizedAt": "2026-09-05T04:02:00Z",
        "releaseId": "run151-test",
        "sourceRevision": REVISION,
        "sourceTreeSha256": "4" * 64,
        "evidenceSha256": "5" * 64,
        "releaseStatementSha256": _sha(statement_path),
        "signatureVerificationSha256": _sha(signature_path),
        "signatureVerifierEvidenceSha256": "7" * 64,
        "publish": [{"name": path.name, "sha256": _sha(path), "size": path.stat().st_size} for path in payloads],
        "sbomReferences": sboms,
        "status": "promoted",
    }
    (root / "promotion-receipt.json").write_bytes(publish._canonical_bytes(receipt))
    return root


class FakePublisher:
    def __init__(self, now: datetime = PUB_NOW):
        self.now = now
        self.remote: dict[str, bytes] = {}

    def __call__(self, request: dict) -> dict:
        art = request["artifact"]
        name = art["name"]
        if request["operation"] == "publish":
            data = Path(art["localPath"]).read_bytes()
            if name in self.remote:
                assert self.remote[name] == data
                status = "present"
            else:
                self.remote[name] = data
                status = "created"
        else:
            data = self.remote[name]
            assert hashlib.sha256(data).hexdigest() == art["sha256"]
            assert len(data) == art["size"]
            status = "present"
        return {
            "schemaVersion": 1,
            "operation": request["operation"],
            "publicationId": request["publicationId"],
            "status": status,
            "target": {
                "publisher": request["target"]["publisher"],
                "targetId": request["target"]["targetId"],
                "locator": f"mock://release/{name}",
            },
            "artifact": {"name": name, "sha256": art["sha256"], "size": art["size"]},
            "guarantees": {
                "createOnly": True,
                "overwrite": False,
                "remoteReadbackVerified": True,
                "immutability": "object-lock",
            },
            "verifiedAt": self.now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


class FakeVerifier:
    def __init__(self, remote: dict[str, bytes], *, now: datetime = PUB_NOW):
        self.remote = remote
        self.now = now
        self.identity_override: str | None = None
        self.read_only = True
        self.publisher_credentials_reused = False
        self.locator_override: str | None = None
        self.hash_override: str | None = None
        self.size_override: int | None = None
        self.calls: list[dict] = []

    def __call__(self, request: dict) -> dict:
        self.calls.append(request)
        art = request["artifact"]
        name = art["name"]
        data = self.remote[name]
        sha = hashlib.sha256(data).hexdigest()
        size = len(data)
        return {
            "schemaVersion": 1,
            "operation": "verify",
            "verificationId": request["verificationId"],
            "publicationId": request["publicationId"],
            "status": "present",
            "verifier": {
                "identity": self.identity_override or request["verifier"]["identity"],
                "readOnly": self.read_only,
                "publisherCredentialsReused": self.publisher_credentials_reused,
            },
            "target": dict(request["target"]),
            "artifact": {"name": name, "sha256": art["sha256"], "size": art["size"]},
            "remote": {
                "locator": self.locator_override or art["locator"],
                "sha256": self.hash_override or sha,
                "size": self.size_override if self.size_override is not None else size,
                "immutability": art["immutability"],
            },
            "verifiedAt": self.now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


class FakeBinder:
    def __init__(self, *, now: datetime = FINAL_NOW):
        self.now = now
        self.remote: dict[str, bytes] = {}
        self.target_override: dict | None = None
        self.locator_override: str | None = None
        self.create_only = True
        self.overwrite = False
        self.readback = True
        self.immutability = "object-lock"
        self.binding_type = "release-asset"
        self.record_override: dict | None = None
        self.mutate_local = False
        self.calls: list[dict] = []

    def __call__(self, request: dict) -> dict:
        self.calls.append(request)
        record = request["record"]
        name = record["name"]
        if request["operation"] == "bind":
            path = Path(record["localPath"])
            data = path.read_bytes()
            if self.mutate_local:
                path.chmod(0o644)
                path.write_bytes(data + b"tamper")
            if name in self.remote:
                assert self.remote[name] == data
                status = "present"
            else:
                self.remote[name] = data
                status = "created"
        else:
            data = self.remote[name]
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
            assert len(data) == record["size"]
            status = "present"
        target = self.target_override or {
            "publisher": request["target"]["publisher"],
            "targetId": request["target"]["targetId"],
            "locator": self.locator_override or "mock://release/release-publication-record.json",
        }
        response_record = self.record_override or {"name": name, "sha256": record["sha256"], "size": record["size"]}
        return {
            "schemaVersion": 1,
            "operation": request["operation"],
            "bindingId": request["bindingId"],
            "publicationId": request["publicationId"],
            "status": status,
            "target": target,
            "record": response_record,
            "guarantees": {
                "createOnly": self.create_only,
                "overwrite": self.overwrite,
                "remoteReadbackVerified": self.readback,
                "immutability": self.immutability,
                "bindingType": self.binding_type,
            },
            "verifiedAt": self.now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


def _publication(tmp_path: Path):
    promoted = _promoted(tmp_path)
    publisher = FakePublisher()
    output = tmp_path / "publication"
    publish.publish_release(
        promotion_dir=promoted,
        output_dir=output,
        publisher="test-publisher",
        target_id="release/run151-test",
        adapter=publisher,
        now=PUB_NOW,
    )
    return output, publisher


def _prepared(tmp_path: Path):
    publication, publisher = _publication(tmp_path)
    verifier = FakeVerifier(publisher.remote, now=PUB_NOW)
    prepared = tmp_path / "prepared"
    out = finalize.prepare_attestation(
        publication_dir=publication,
        output_dir=prepared,
        verifier_identity=VERIFIER_IDENTITY,
        verifier=verifier,
        now=PUB_NOW,
    )
    evidence = tmp_path / "signature-verifier-output.json"
    evidence.write_text('{"verified":true,"source":"external-test-verifier"}\n')
    signature = tmp_path / "publication-attestation.signature-verification.json"
    finalize.write_signature_verification_record(
        publication_dir=publication,
        attestation=prepared / "publication-attestation.json",
        verifier_evidence=evidence,
        output=signature,
        signer_identity=SIGNER_IDENTITY,
        verifier_name="external-signature-verifier",
        verifier_version="1.0",
        verified_at=SIG_NOW,
    )
    return publication, publisher, prepared, evidence, signature, out


def test_run151_end_to_end_attests_reverifies_and_binds_final_record(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, prepared_result = _prepared(tmp_path)
    assert prepared_result["phase"] == "attestation-prepared"
    verifier = FakeVerifier(publisher.remote, now=FINAL_NOW)
    binder = FakeBinder(now=FINAL_NOW)
    output = tmp_path / "finalized"
    result = finalize.finalize_publication(
        publication_dir=publication,
        prepared_dir=prepared,
        signature_record=signature,
        signature_verifier_evidence=evidence,
        output_dir=output,
        expected_signer_identity=SIGNER_IDENTITY,
        verifier_identity=VERIFIER_IDENTITY,
        verifier=verifier,
        binder=binder,
        now=FINAL_NOW,
    )
    assert result["phase"] == "bound" and result["artifact_count"] == 4
    assert binder.remote[finalize.RECORD_NAME] == (output / finalize.RECORD_NAME).read_bytes()
    record = json.loads((output / finalize.RECORD_NAME).read_text())
    assert record["signer"]["identity"] == SIGNER_IDENTITY
    assert record["independentVerifier"] == {
        "identity": VERIFIER_IDENTITY,
        "verificationId": prepared_result["verification_id"],
        "readOnly": True,
        "publisherCredentialsReused": False,
    }
    assert record["verification"]["independentRemoteReadbackVerifiedAfterSignature"] is True
    assert record["target"] == {"publisher": "test-publisher", "targetId": "release/run151-test"}
    text = "\n".join(p.read_text() for p in output.rglob("*.json"))
    assert str(tmp_path) not in text and "localPath" not in text
    assert (output / "publication-transparency.json").read_bytes() == (publication / "publication-transparency.json").read_bytes()
    assert (output / "publication-attestation.json").read_bytes() == (prepared / "publication-attestation.json").read_bytes()


def test_run151_final_record_is_deterministic_and_binding_retry_is_resumable(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, _ = _prepared(tmp_path)
    binder = FakeBinder(now=FINAL_NOW)
    first = tmp_path / "first"
    finalize.finalize_publication(
        publication_dir=publication,
        prepared_dir=prepared,
        signature_record=signature,
        signature_verifier_evidence=evidence,
        output_dir=first,
        expected_signer_identity=SIGNER_IDENTITY,
        verifier_identity=VERIFIER_IDENTITY,
        verifier=FakeVerifier(publisher.remote, now=FINAL_NOW),
        binder=binder,
        now=FINAL_NOW,
    )
    first_sha = _sha(first / finalize.RECORD_NAME)
    binder.now = FINAL_NOW + timedelta(minutes=1)
    second = tmp_path / "second"
    finalize.finalize_publication(
        publication_dir=publication,
        prepared_dir=prepared,
        signature_record=signature,
        signature_verifier_evidence=evidence,
        output_dir=second,
        expected_signer_identity=SIGNER_IDENTITY,
        verifier_identity=VERIFIER_IDENTITY,
        verifier=FakeVerifier(publisher.remote, now=FINAL_NOW + timedelta(minutes=1)),
        binder=binder,
        now=FINAL_NOW + timedelta(minutes=1),
    )
    assert _sha(second / finalize.RECORD_NAME) == first_sha
    bind_results = [json.loads(p.read_text()) for p in (second / "binding-results").glob("*.bind.json")]
    assert bind_results[0]["status"] == "present"


def test_run151_rejects_tampered_noncanonical_or_extra_publication_evidence(tmp_path: Path):
    publication, _ = _publication(tmp_path)
    transparency = json.loads((publication / "publication-transparency.json").read_text())
    (publication / "publication-transparency.json").write_text(json.dumps(transparency, indent=2))
    with pytest.raises(finalize.TransparencyError, match="PUBLICATION_TRANSPARENCY_NOT_CANONICAL"):
        finalize.prepare_attestation(
            publication_dir=publication,
            output_dir=tmp_path / "bad1",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=lambda _: {},
            now=PUB_NOW,
        )

    publication2, _ = _publication(tmp_path / "extra")
    (publication2 / "extra.json").write_text("{}")
    with pytest.raises(finalize.TransparencyError, match="PUBLICATION_DIRECTORY_ALLOWLIST_MISMATCH"):
        finalize.prepare_attestation(
            publication_dir=publication2,
            output_dir=tmp_path / "bad2",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=lambda _: {},
            now=PUB_NOW,
        )

    publication3, _ = _publication(tmp_path / "publisher-drift")
    result_path = next((publication3 / "publisher-results").glob("*.verify.json"))
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(finalize.TransparencyError, match="PUBLICATION_PUBLISHER_RESULT_NOT_CANONICAL|PUBLICATION_PUBLISHER_RESULT_HASH_MISMATCH"):
        finalize.prepare_attestation(
            publication_dir=publication3,
            output_dir=tmp_path / "bad3",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=lambda _: {},
            now=PUB_NOW,
        )


def test_run151_requires_distinct_read_only_verifier_without_publisher_credentials(tmp_path: Path):
    publication, publisher = _publication(tmp_path)
    with pytest.raises(finalize.TransparencyError, match="INDEPENDENT_VERIFIER_NOT_DISTINCT"):
        finalize.prepare_attestation(
            publication_dir=publication,
            output_dir=tmp_path / "same",
            verifier_identity="test-publisher",
            verifier=FakeVerifier(publisher.remote),
            now=PUB_NOW,
        )

    verifier = FakeVerifier(publisher.remote)
    verifier.read_only = False
    with pytest.raises(finalize.TransparencyError, match="INDEPENDENT_VERIFIER_AUTHORITY_INVALID"):
        finalize.prepare_attestation(
            publication_dir=publication,
            output_dir=tmp_path / "rw",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=verifier,
            now=PUB_NOW,
        )

    verifier = FakeVerifier(publisher.remote)
    verifier.publisher_credentials_reused = True
    with pytest.raises(finalize.TransparencyError, match="INDEPENDENT_VERIFIER_AUTHORITY_INVALID"):
        finalize.prepare_attestation(
            publication_dir=publication,
            output_dir=tmp_path / "reused",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=verifier,
            now=PUB_NOW,
        )


def test_run151_rejects_remote_rebinding_wrong_verifier_and_stale_verification(tmp_path: Path):
    publication, publisher = _publication(tmp_path)
    verifier = FakeVerifier(publisher.remote)
    verifier.identity_override = "ci/other-verifier"
    with pytest.raises(finalize.TransparencyError, match="INDEPENDENT_VERIFIER_IDENTITY_MISMATCH"):
        finalize.prepare_attestation(
            publication_dir=publication,
            output_dir=tmp_path / "identity",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=verifier,
            now=PUB_NOW,
        )

    verifier = FakeVerifier(publisher.remote)
    verifier.hash_override = "0" * 64
    with pytest.raises(finalize.TransparencyError, match="INDEPENDENT_VERIFIER_REMOTE_MISMATCH"):
        finalize.prepare_attestation(
            publication_dir=publication,
            output_dir=tmp_path / "hash",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=verifier,
            now=PUB_NOW,
        )

    verifier = FakeVerifier(publisher.remote, now=PUB_NOW - timedelta(hours=2))
    with pytest.raises(finalize.TransparencyError, match="INDEPENDENT_VERIFIER_TIME_STALE"):
        finalize.prepare_attestation(
            publication_dir=publication,
            output_dir=tmp_path / "stale",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=verifier,
            now=PUB_NOW,
        )


def test_run151_signature_record_rebinds_attestation_signer_and_external_verifier_evidence(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, _ = _prepared(tmp_path)
    with pytest.raises(finalize.TransparencyError, match="SIGNATURE_RECORD_SIGNER_IDENTITY_MISMATCH"):
        finalize.finalize_publication(
            publication_dir=publication,
            prepared_dir=prepared,
            signature_record=signature,
            signature_verifier_evidence=evidence,
            output_dir=tmp_path / "wrong-signer",
            expected_signer_identity="https://github.com/scikit-plots/scikit-plots/.github/workflows/other.yml@refs/heads/main",
            verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW),
            binder=FakeBinder(),
            now=FINAL_NOW,
        )

    evidence.write_text("changed verifier evidence\n")
    with pytest.raises(finalize.TransparencyError, match="SIGNATURE_VERIFIER_EVIDENCE_HASH_MISMATCH"):
        finalize.finalize_publication(
            publication_dir=publication,
            prepared_dir=prepared,
            signature_record=signature,
            signature_verifier_evidence=evidence,
            output_dir=tmp_path / "wrong-evidence",
            expected_signer_identity=SIGNER_IDENTITY,
            verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW),
            binder=FakeBinder(),
            now=FINAL_NOW,
        )


def test_run151_detects_attestation_tamper_after_signature(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, _ = _prepared(tmp_path)
    att_path = prepared / "publication-attestation.json"
    att = json.loads(att_path.read_text())
    att["predicate"]["generatedAt"] = "2026-09-05T04:30:30Z"
    att_path.write_bytes(finalize._canonical_bytes(att))
    with pytest.raises(finalize.TransparencyError, match="SIGNATURE_RECORD_SUBJECT_MISMATCH"):
        finalize.finalize_publication(
            publication_dir=publication,
            prepared_dir=prepared,
            signature_record=signature,
            signature_verifier_evidence=evidence,
            output_dir=tmp_path / "out",
            expected_signer_identity=SIGNER_IDENTITY,
            verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW),
            binder=FakeBinder(),
            now=FINAL_NOW,
        )


def test_run151_finalization_rechecks_remote_objects_after_signature(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, _ = _prepared(tmp_path)
    first = next(iter(publisher.remote))
    publisher.remote[first] += b"tampered after signing"
    with pytest.raises(finalize.TransparencyError, match="INDEPENDENT_VERIFIER_REMOTE_MISMATCH"):
        finalize.finalize_publication(
            publication_dir=publication,
            prepared_dir=prepared,
            signature_record=signature,
            signature_verifier_evidence=evidence,
            output_dir=tmp_path / "out",
            expected_signer_identity=SIGNER_IDENTITY,
            verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW),
            binder=FakeBinder(),
            now=FINAL_NOW,
        )


def test_run151_rejects_final_verification_that_predates_signature(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, _ = _prepared(tmp_path)
    very_old = SIG_NOW - timedelta(minutes=11)
    with pytest.raises(finalize.TransparencyError, match="FINALIZATION_VERIFICATION_PREDATES_SIGNATURE"):
        finalize.finalize_publication(
            publication_dir=publication,
            prepared_dir=prepared,
            signature_record=signature,
            signature_verifier_evidence=evidence,
            output_dir=tmp_path / "out",
            expected_signer_identity=SIGNER_IDENTITY,
            verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=very_old),
            binder=FakeBinder(),
            now=FINAL_NOW,
        )


def test_run151_binder_cannot_switch_target_overwrite_or_hide_locator_secret(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, _ = _prepared(tmp_path)

    binder = FakeBinder()
    binder.target_override = {"publisher": "other", "targetId": "release/run151-test", "locator": "mock://release/record"}
    with pytest.raises(finalize.TransparencyError, match="RELEASE_BINDER_TARGET_MISMATCH"):
        finalize.finalize_publication(
            publication_dir=publication, prepared_dir=prepared, signature_record=signature,
            signature_verifier_evidence=evidence, output_dir=tmp_path / "target",
            expected_signer_identity=SIGNER_IDENTITY, verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW), binder=binder, now=FINAL_NOW,
        )

    binder = FakeBinder()
    binder.overwrite = True
    with pytest.raises(finalize.TransparencyError, match="RELEASE_BINDER_AUTHORITY_INVALID"):
        finalize.finalize_publication(
            publication_dir=publication, prepared_dir=prepared, signature_record=signature,
            signature_verifier_evidence=evidence, output_dir=tmp_path / "overwrite",
            expected_signer_identity=SIGNER_IDENTITY, verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW), binder=binder, now=FINAL_NOW,
        )

    binder = FakeBinder()
    binder.locator_override = "https://example.invalid/release?token=secret"
    with pytest.raises(finalize.TransparencyError, match="RELEASE_BINDER_LOCATOR_INVALID"):
        finalize.finalize_publication(
            publication_dir=publication, prepared_dir=prepared, signature_record=signature,
            signature_verifier_evidence=evidence, output_dir=tmp_path / "secret",
            expected_signer_identity=SIGNER_IDENTITY, verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW), binder=binder, now=FINAL_NOW,
        )


def test_run151_detects_final_record_mutation_during_remote_bind(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, _ = _prepared(tmp_path)
    binder = FakeBinder()
    binder.mutate_local = True
    with pytest.raises(finalize.TransparencyError, match="FINAL_RECORD_CHANGED_DURING_BIND"):
        finalize.finalize_publication(
            publication_dir=publication, prepared_dir=prepared, signature_record=signature,
            signature_verifier_evidence=evidence, output_dir=tmp_path / "out",
            expected_signer_identity=SIGNER_IDENTITY, verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW), binder=binder, now=FINAL_NOW,
        )


def test_run151_command_adapters_bound_stdout_and_reject_duplicate_json_keys(tmp_path: Path, monkeypatch):
    real_popen = finalize.subprocess.Popen
    seen = []

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen.append(proc)
        return proc

    monkeypatch.setattr(finalize.subprocess, "Popen", tracking_popen)
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setattr(finalize.os, "defpath", str(empty_path))
    noisy = tmp_path / "noisy.py"
    noisy.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(b'x' * 400000)\n"
        "sys.stdout.flush()\n"
    )
    noisy.chmod(0o755)
    adapter = finalize.command_verifier([sys.executable, str(noisy)])
    with pytest.raises(finalize.TransparencyError, match="INDEPENDENT_VERIFIER_COMMAND_OUTPUT_TOO_LARGE"):
        adapter({"schemaVersion": 1})
    assert seen[-1].poll() is not None
    assert seen[-1].stdin is not None and seen[-1].stdin.closed
    assert seen[-1].stdout is not None and seen[-1].stdout.closed

    duplicate = tmp_path / "duplicate.py"
    duplicate.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.write('{\\\"schemaVersion\\\":1,\\\"schemaVersion\\\":1}')\n"
    )
    duplicate.chmod(0o755)
    adapter = finalize.command_binder([sys.executable, str(duplicate)])
    with pytest.raises(finalize.TransparencyError, match="RELEASE_BINDER_COMMAND_OUTPUT_DUPLICATE_KEY"):
        adapter({"schemaVersion": 1})
    assert seen[-1].poll() is not None
    assert seen[-1].stdin is not None and seen[-1].stdin.closed
    assert seen[-1].stdout is not None and seen[-1].stdout.closed


def test_run151_finalization_output_cannot_live_inside_publication_or_prepared(tmp_path: Path):
    publication, publisher, prepared, evidence, signature, _ = _prepared(tmp_path)
    with pytest.raises(finalize.TransparencyError, match="FINALIZATION_OUTPUT_INSIDE_INPUT"):
        finalize.finalize_publication(
            publication_dir=publication,
            prepared_dir=prepared,
            signature_record=signature,
            signature_verifier_evidence=evidence,
            output_dir=prepared / "final",
            expected_signer_identity=SIGNER_IDENTITY,
            verifier_identity=VERIFIER_IDENTITY,
            verifier=FakeVerifier(publisher.remote, now=FINAL_NOW),
            binder=FakeBinder(),
            now=FINAL_NOW,
        )


def test_run151_rejects_stale_unsigned_attestation_before_signature_record(tmp_path: Path):
    publication, publisher = _publication(tmp_path)
    prepared = tmp_path / "prepared"
    finalize.prepare_attestation(
        publication_dir=publication,
        output_dir=prepared,
        verifier_identity=VERIFIER_IDENTITY,
        verifier=FakeVerifier(publisher.remote, now=PUB_NOW),
        now=PUB_NOW,
    )
    att_path = prepared / "publication-attestation.json"
    att = json.loads(att_path.read_text())
    att["predicate"]["generatedAt"] = "2026-09-01T04:30:00Z"
    att_path.write_bytes(finalize._canonical_bytes(att))
    evidence = tmp_path / "external.json"
    evidence.write_text("{}\n")
    with pytest.raises(finalize.TransparencyError, match="POST_PUBLICATION_ATTESTATION_STALE"):
        finalize.write_signature_verification_record(
            publication_dir=publication,
            attestation=att_path,
            verifier_evidence=evidence,
            output=tmp_path / "signature.json",
            signer_identity=SIGNER_IDENTITY,
            verifier_name="external-signature-verifier",
            verifier_version="1.0",
            verified_at=SIG_NOW,
        )
