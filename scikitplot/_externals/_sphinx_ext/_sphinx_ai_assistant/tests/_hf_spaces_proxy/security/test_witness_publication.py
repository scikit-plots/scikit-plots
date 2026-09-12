from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

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


witness = _load("run152_witness", SECURITY / "witness_publication.py")
NOW = datetime(2026, 9, 5, 5, 30, tzinfo=timezone.utc)
LOG_ID = "sigstore/rekor-production"
SUBMITTER = "ci/transparency-submitter"
PRIMARY = "ci/transparency-readonly-verifier"
WITNESSES = [("witness/a", "operator-a"), ("witness/b", "operator-b")]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical(value: dict) -> bytes:
    return witness._canonical_bytes(value)


def _finalized(tmp_path: Path) -> Path:
    root = tmp_path / "run151-finalized"
    root.mkdir()
    for dirname in ("final-verifier-results", "binding-results"):
        (root / dirname).mkdir()
    (root / "final-verifier-results" / "01-run152-test.zip.verify.json").write_bytes(_canonical({"ok": True, "artifact": "run152-test.zip"}))
    (root / "final-verifier-results" / "02-release-statement.json.verify.json").write_bytes(_canonical({"ok": True, "artifact": "release-statement.json"}))
    (root / "binding-results" / "release-publication-record.json.bind.json").write_bytes(_canonical({"ok": True, "operation": "bind"}))
    (root / "binding-results" / "release-publication-record.json.verify.json").write_bytes(_canonical({"ok": True, "operation": "verify"}))

    publication_transparency = {"schemaVersion": 1, "status": "published"}
    publication_receipt = {"schemaVersion": 1, "status": "published"}
    attestation = {"_type": "https://in-toto.io/Statement/v1", "subject": []}
    signature = {"schemaVersion": 1, "verified": True}
    files = {
        "publication-transparency.json": publication_transparency,
        "publication-receipt.json": publication_receipt,
        "publication-attestation.json": attestation,
        "publication-attestation.signature-verification.json": signature,
    }
    for name, value in files.items():
        (root / name).write_bytes(_canonical(value))

    record = {
        "schemaVersion": 1,
        "predicateType": "https://scikit-plots.org/attestations/release-final-record/v1",
        "status": "finalized",
        "publicationId": "pub-run152-test",
        "release": {"releaseId": "run152-test", "sourceRevision": "a" * 40},
        "subject": {
            "promotionReceiptSha256": "1" * 64,
            "publicationReceiptSha256": _sha(root / "publication-receipt.json"),
            "publicationTransparencySha256": _sha(root / "publication-transparency.json"),
            "postPublicationAttestationSha256": _sha(root / "publication-attestation.json"),
            "signatureVerificationSha256": _sha(root / "publication-attestation.signature-verification.json"),
            "signatureVerifierEvidenceSha256": "2" * 64,
        },
        "target": {"publisher": "test-publisher", "targetId": "release/run152-test"},
        "signer": {"identity": "release/signer", "verifiedAt": "2026-09-05T05:00:00Z"},
        "independentVerifier": {"identity": "ci/post-publication-verifier", "verificationId": "verify-1", "readOnly": True, "publisherCredentialsReused": False},
        "artifacts": [
            {"name": "run152-test.zip", "sha256": "3" * 64, "size": 123, "locator": "mock://release/run152-test.zip", "immutability": "object-lock"},
            {"name": "release-statement.json", "sha256": "4" * 64, "size": 45, "locator": "mock://release/release-statement.json", "immutability": "object-lock"},
        ],
        "verification": {"publicationEvidenceVerified": True, "signedAttestationVerified": True, "signerIdentityVerified": True, "independentRemoteReadbackVerifiedAfterSignature": True, "allPublishedObjectsPresent": True},
    }
    record_path = root / "release-publication-record.json"
    record_path.write_bytes(_canonical(record))
    receipt = {
        "schemaVersion": 1,
        "status": "bound",
        "publicationId": "pub-run152-test",
        "releaseId": "run152-test",
        "publicationTransparencySha256": record["subject"]["publicationTransparencySha256"],
        "postPublicationAttestationSha256": record["subject"]["postPublicationAttestationSha256"],
        "signatureVerificationSha256": record["subject"]["signatureVerificationSha256"],
        "finalRecord": {"name": record_path.name, "sha256": _sha(record_path), "size": record_path.stat().st_size},
        "binding": {
            "bindingId": "binding-1",
            "locator": "mock://release/release-publication-record.json",
            "immutability": "object-lock",
            "bindingType": "release-asset",
            "verifiedAt": "2026-09-05T05:02:00Z",
            "bindEvidenceSha256": _sha(root / "binding-results" / "release-publication-record.json.bind.json"),
            "verifyEvidenceSha256": _sha(root / "binding-results" / "release-publication-record.json.verify.json"),
        },
        "finalVerificationEvidence": [
            {"name": "run152-test.zip", "sha256": _sha(root / "final-verifier-results" / "01-run152-test.zip.verify.json")},
            {"name": "release-statement.json", "sha256": _sha(root / "final-verifier-results" / "02-release-statement.json.verify.json")},
        ],
    }
    (root / "release-publication-binding-receipt.json").write_bytes(_canonical(receipt))
    return root


def _previous(tmp_path: Path) -> Path:
    path = tmp_path / "previous-checkpoint.json"
    path.write_bytes(_canonical({
        "schemaVersion": 1,
        "logId": LOG_ID,
        "checkpoint": {"treeSize": 41, "rootHash": "7" * 64, "signedCheckpointSha256": "8" * 64},
    }))
    return path


class FakeLog:
    def __init__(self, *, now: datetime = NOW):
        self.now = now
        self.remote: dict[str, bytes] = {}
        self.log_id = LOG_ID
        self.entry_locator = "https://transparency.example/entry/42"
        self.checkpoint = {"treeSize": 43, "rootHash": "9" * 64, "signedCheckpointSha256": "a" * 64}
        self.mutate_local = False

    def __call__(self, request: dict) -> dict:
        subject = request["subject"]
        data = Path(subject["localPath"]).read_bytes()
        if self.mutate_local:
            Path(subject["localPath"]).write_bytes(data + b"tamper")
        if subject["sha256"] in self.remote:
            assert self.remote[subject["sha256"]] == data
            status = "present"
        else:
            self.remote[subject["sha256"]] = data
            status = "created"
        return {
            "schemaVersion": 1,
            "operation": "submit",
            "transparencyId": request["transparencyId"],
            "status": status,
            "log": {"logId": self.log_id, "entryId": "entry-42", "entryIndex": 42, "entryLocator": self.entry_locator},
            "subject": {k: subject[k] for k in ("name", "sha256", "size")},
            "checkpoint": dict(self.checkpoint),
            "proof": {"appendOnly": True, "overwrite": False, "integratedEntryVerified": True},
            "integratedAt": self.now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


class FakeObserver:
    def __init__(self, identity: str, operator: str | None = None, *, now: datetime = NOW):
        self.identity = identity
        self.operator = operator
        self.now = now
        self.read_only = True
        self.log_credentials_reused = False
        self.primary_credentials_reused = False
        self.proof_override: dict | None = None
        self.checkpoint_override: dict | None = None
        self.identity_override: str | None = None

    def __call__(self, request: dict) -> dict:
        observer = {
            "identity": self.identity_override or self.identity,
            "readOnly": self.read_only,
            "logCredentialsReused": self.log_credentials_reused,
        }
        if self.operator is not None:
            observer["operator"] = self.operator
            observer["primaryVerifierCredentialsReused"] = self.primary_credentials_reused
        return {
            "schemaVersion": request["schemaVersion"],
            "operation": "verify",
            "verificationId": request["verificationId"],
            "transparencyId": request["transparencyId"],
            "status": "included",
            "observer": observer,
            "log": dict(request["log"]),
            "subject": dict(request["subject"]),
            "checkpoint": self.checkpoint_override or dict(request["checkpoint"]),
            "previousCheckpoint": dict(request["previousCheckpoint"]),
            "proof": self.proof_override or {"checkpointSignatureVerified": True, "inclusionVerified": True, "consistencyVerified": True, "integratedEntryVerified": True},
            "verifiedAt": self.now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


class FakeAnchor:
    def __init__(self, *, now: datetime = NOW):
        self.now = now
        self.remote: dict[str, bytes] = {}
        self.locator = "mock://release/release-transparency-witness-record.json"
        self.create_only = True
        self.readback = True
        self.mutate_local = False

    def __call__(self, request: dict) -> dict:
        record = request["record"]
        if request["operation"] == "bind":
            path = Path(record["localPath"])
            data = path.read_bytes()
            if self.mutate_local:
                path.chmod(0o644)
                path.write_bytes(data + b"tamper")
            if record["name"] in self.remote:
                assert self.remote[record["name"]] == data
                status = "present"
            else:
                self.remote[record["name"]] = data
                status = "created"
        else:
            data = self.remote[record["name"]]
            assert _sha_bytes(data) == record["sha256"]
            assert len(data) == record["size"]
            status = "present"
        return {
            "schemaVersion": 1,
            "operation": request["operation"],
            "anchorId": request["anchorId"],
            "status": status,
            "target": dict(request["target"]),
            "record": {k: record[k] for k in ("name", "sha256", "size")},
            "guarantees": {"createOnly": self.create_only, "overwrite": False, "remoteReadbackVerified": self.readback, "immutability": "object-lock", "bindingType": "release-asset", "locator": self.locator},
            "verifiedAt": self.now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


def _run(tmp_path: Path, *, log=None, primary=None, witness_defs=None, anchor=None, output_name="out"):
    final = _finalized(tmp_path)
    prev = _previous(tmp_path)
    log = log or FakeLog()
    primary = primary or FakeObserver(PRIMARY)
    anchor = anchor or FakeAnchor()
    defs = witness_defs or [(identity, operator, FakeObserver(identity, operator)) for identity, operator in WITNESSES]
    out = tmp_path / output_name
    result = witness.witness_publication(
        finalized_dir=final,
        previous_checkpoint=prev,
        output_dir=out,
        log_id=LOG_ID,
        submitter_identity=SUBMITTER,
        log_adapter=log,
        verifier_identity=PRIMARY,
        verifier=primary,
        witnesses=defs,
        anchor=anchor,
        now=NOW,
    )
    return result, out, final, prev, log, anchor


def test_run152_end_to_end_requires_append_only_consistency_and_witness_quorum(tmp_path: Path):
    result, out, final, prev, log, anchor = _run(tmp_path)
    assert result["phase"] == "witnessed"
    assert result["witness_count"] == 2 and result["operator_count"] == 2
    record = json.loads((out / witness.RECORD_NAME).read_text())
    assert record["subject"]["finalPublicationRecordSha256"] == _sha(final / "release-publication-record.json")
    assert record["transparencyLog"]["checkpoint"] == log.checkpoint
    assert record["witnessQuorum"]["witnesses"] == [
        {"identity": "witness/a", "operator": "operator-a"},
        {"identity": "witness/b", "operator": "operator-b"},
    ]
    assert record["verification"] == {"primaryVerifierIdentity": PRIMARY, "checkpointSignatureVerified": True, "inclusionVerified": True, "consistencyVerified": True, "integratedEntryVerified": True}
    assert anchor.remote[witness.RECORD_NAME] == (out / witness.RECORD_NAME).read_bytes()
    accepted = json.loads((out / "accepted-transparency-checkpoint.json").read_text())
    assert accepted == {"schemaVersion": 1, "logId": LOG_ID, "checkpoint": log.checkpoint}
    receipt = json.loads((out / "release-transparency-witness-receipt.json").read_text())
    assert receipt["transparency"]["acceptedCheckpointSha256"] == _sha(out / "accepted-transparency-checkpoint.json")
    text = "\n".join(path.read_text() for path in out.rglob("*.json"))
    assert str(tmp_path) not in text and "localPath" not in text


def test_run152_witness_record_is_deterministic_and_anchor_retry_is_resumable(tmp_path: Path):
    final = _finalized(tmp_path)
    prev = _previous(tmp_path)
    log = FakeLog()
    anchor = FakeAnchor()
    defs = [(identity, operator, FakeObserver(identity, operator)) for identity, operator in WITNESSES]
    first = tmp_path / "first"
    witness.witness_publication(finalized_dir=final, previous_checkpoint=prev, output_dir=first, log_id=LOG_ID, submitter_identity=SUBMITTER, log_adapter=log, verifier_identity=PRIMARY, verifier=FakeObserver(PRIMARY), witnesses=defs, anchor=anchor, now=NOW)
    first_sha = _sha(first / witness.RECORD_NAME)
    later = NOW + timedelta(minutes=1)
    log.now = later; anchor.now = later
    defs2 = [(identity, operator, FakeObserver(identity, operator, now=later)) for identity, operator in WITNESSES]
    second = tmp_path / "second"
    witness.witness_publication(finalized_dir=final, previous_checkpoint=prev, output_dir=second, log_id=LOG_ID, submitter_identity=SUBMITTER, log_adapter=log, verifier_identity=PRIMARY, verifier=FakeObserver(PRIMARY, now=later), witnesses=defs2, anchor=anchor, now=later)
    assert _sha(second / witness.RECORD_NAME) == first_sha
    assert json.loads((second / "anchor-results" / (witness.RECORD_NAME + ".bind.json")).read_text())["status"] == "present"
    assert json.loads((second / "log-results" / "submit.json").read_text())["status"] == "present"


@pytest.mark.parametrize("mutator,code", [
    (lambda d: (d / "unexpected.json").write_text("{}\n"), "RUN151_DIRECTORY_ALLOWLIST_MISMATCH"),
    (lambda d: (d / "release-publication-record.json").write_text((d / "release-publication-record.json").read_text() + " \n"), "RUN151_FINAL_RECORD_NOT_CANONICAL"),
])
def test_run152_rejects_run151_directory_drift_and_noncanonical_evidence(tmp_path: Path, mutator, code: str):
    final = _finalized(tmp_path); prev = _previous(tmp_path); mutator(final)
    with pytest.raises(witness.WitnessError, match=code):
        witness.witness_publication(finalized_dir=final, previous_checkpoint=prev, output_dir=tmp_path / "out", log_id=LOG_ID, submitter_identity=SUBMITTER, log_adapter=FakeLog(), verifier_identity=PRIMARY, verifier=FakeObserver(PRIMARY), witnesses=[(a, b, FakeObserver(a, b)) for a, b in WITNESSES], anchor=FakeAnchor(), now=NOW)



def test_run152_rejects_tampered_run151_binding_or_final_verifier_sidecar(tmp_path: Path):
    final = _finalized(tmp_path); prev = _previous(tmp_path)
    bind = final / "binding-results" / "release-publication-record.json.bind.json"
    bind.write_bytes(_canonical({"ok": False, "operation": "bind"}))
    with pytest.raises(witness.WitnessError, match="RUN151_BINDING_EVIDENCE_HASH_MISMATCH"):
        witness.witness_publication(finalized_dir=final, previous_checkpoint=prev, output_dir=tmp_path / "out1", log_id=LOG_ID, submitter_identity=SUBMITTER, log_adapter=FakeLog(), verifier_identity=PRIMARY, verifier=FakeObserver(PRIMARY), witnesses=[(a, b, FakeObserver(a, b)) for a, b in WITNESSES], anchor=FakeAnchor(), now=NOW)

    case2 = tmp_path / "case2"; case2.mkdir()
    final2 = _finalized(case2); prev2 = _previous(case2)
    sidecar = final2 / "final-verifier-results" / "01-run152-test.zip.verify.json"
    sidecar.write_bytes(_canonical({"ok": False, "artifact": "run152-test.zip"}))
    with pytest.raises(witness.WitnessError, match="RUN151_FINAL_VERIFICATION_EVIDENCE_HASH_MISMATCH"):
        witness.witness_publication(finalized_dir=final2, previous_checkpoint=prev2, output_dir=case2 / "out2", log_id=LOG_ID, submitter_identity=SUBMITTER, log_adapter=FakeLog(), verifier_identity=PRIMARY, verifier=FakeObserver(PRIMARY), witnesses=[(a, b, FakeObserver(a, b)) for a, b in WITNESSES], anchor=FakeAnchor(), now=NOW)

def test_run152_rejects_previous_checkpoint_for_another_log(tmp_path: Path):
    final = _finalized(tmp_path); prev = _previous(tmp_path)
    doc = json.loads(prev.read_text()); doc["logId"] = "other/log"; prev.write_bytes(_canonical(doc))
    with pytest.raises(witness.WitnessError, match="PREVIOUS_CHECKPOINT_LOG_MISMATCH"):
        witness.witness_publication(finalized_dir=final, previous_checkpoint=prev, output_dir=tmp_path / "out", log_id=LOG_ID, submitter_identity=SUBMITTER, log_adapter=FakeLog(), verifier_identity=PRIMARY, verifier=FakeObserver(PRIMARY), witnesses=[(a, b, FakeObserver(a, b)) for a, b in WITNESSES], anchor=FakeAnchor(), now=NOW)


def test_run152_rejects_log_checkpoint_that_does_not_advance(tmp_path: Path):
    log = FakeLog(); log.checkpoint["treeSize"] = 41
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_LOG_CHECKPOINT_ORDER_INVALID"):
        _run(tmp_path, log=log)


def test_run152_rejects_primary_verifier_reusing_submitter_identity(tmp_path: Path):
    final = _finalized(tmp_path); prev = _previous(tmp_path)
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_VERIFIER_REUSES_SUBMITTER_IDENTITY"):
        witness.witness_publication(finalized_dir=final, previous_checkpoint=prev, output_dir=tmp_path / "out", log_id=LOG_ID, submitter_identity=SUBMITTER, log_adapter=FakeLog(), verifier_identity=SUBMITTER, verifier=FakeObserver(SUBMITTER), witnesses=[(a, b, FakeObserver(a, b)) for a, b in WITNESSES], anchor=FakeAnchor(), now=NOW)


def test_run152_rejects_witness_identity_or_operator_non_quorum(tmp_path: Path):
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_WITNESS_IDENTITY_NOT_DISTINCT"):
        _run(tmp_path, witness_defs=[("witness/a", "operator-a", FakeObserver("witness/a", "operator-a")), ("witness/a", "operator-b", FakeObserver("witness/a", "operator-b"))])
    tmp2 = tmp_path / "case2"; tmp2.mkdir()
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_WITNESS_OPERATOR_QUORUM_INVALID"):
        _run(tmp2, witness_defs=[("witness/a", "operator-a", FakeObserver("witness/a", "operator-a")), ("witness/b", "operator-a", FakeObserver("witness/b", "operator-a"))])


def test_run152_rejects_witness_without_readonly_separation(tmp_path: Path):
    bad = FakeObserver("witness/a", "operator-a"); bad.primary_credentials_reused = True
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_WITNESS_OBSERVER_SEPARATION_INVALID"):
        _run(tmp_path, witness_defs=[("witness/a", "operator-a", bad), ("witness/b", "operator-b", FakeObserver("witness/b", "operator-b"))])


def test_run152_rejects_failed_inclusion_consistency_or_checkpoint_signature(tmp_path: Path):
    bad = FakeObserver(PRIMARY)
    bad.proof_override = {"checkpointSignatureVerified": True, "inclusionVerified": True, "consistencyVerified": False, "integratedEntryVerified": True}
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_PRIMARY_VERIFIER_PROOF_INVALID"):
        _run(tmp_path, primary=bad)


def test_run152_rejects_observer_checkpoint_split_view(tmp_path: Path):
    bad = FakeObserver("witness/a", "operator-a")
    bad.checkpoint_override = {"treeSize": 43, "rootHash": "b" * 64, "signedCheckpointSha256": "a" * 64}
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_WITNESS_CHECKPOINT_MISMATCH"):
        _run(tmp_path, witness_defs=[("witness/a", "operator-a", bad), ("witness/b", "operator-b", FakeObserver("witness/b", "operator-b"))])


def test_run152_rejects_stale_observer_evidence(tmp_path: Path):
    old = FakeObserver(PRIMARY, now=NOW - timedelta(hours=1))
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_PRIMARY_VERIFIER_VERIFIED_AT_STALE"):
        _run(tmp_path, primary=old)


def test_run152_rejects_mutation_of_final_record_during_log_submit(tmp_path: Path):
    log = FakeLog(); log.mutate_local = True
    with pytest.raises(witness.WitnessError, match="RUN151_FINAL_RECORD_CHANGED_DURING_LOG_SUBMIT"):
        _run(tmp_path, log=log)


def test_run152_rejects_anchor_mutation_or_locator_collision(tmp_path: Path):
    anchor = FakeAnchor(); anchor.mutate_local = True
    with pytest.raises(witness.WitnessError, match="WITNESS_RECORD_CHANGED_DURING_ANCHOR"):
        _run(tmp_path, anchor=anchor)
    tmp2 = tmp_path / "case2"; tmp2.mkdir()
    collision = FakeAnchor(); collision.locator = "mock://release/release-publication-record.json"
    with pytest.raises(witness.WitnessError, match="TRANSPARENCY_ANCHOR_LOCATOR_COLLISION"):
        _run(tmp2, anchor=collision)


def test_run152_policy_and_documentation_require_external_log_and_distinct_witnesses():
    policy = (SECURITY / "release_witness_policy.toml").read_text()
    assert "min_witnesses = 2" in policy and "min_witness_operators = 2" in policy
    guide = (SECURITY / "RELEASE_WITNESS_GUIDE.md").read_text()
    for phrase in ("append-only", "previous checkpoint", "distinct operators", "split-view", "private key"):
        assert phrase.lower() in guide.lower()
