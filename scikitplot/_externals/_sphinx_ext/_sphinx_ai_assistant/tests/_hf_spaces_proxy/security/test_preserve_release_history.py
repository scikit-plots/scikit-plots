from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

from datetime import datetime, timezone
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


history = _load("run153_history", SECURITY / "preserve_release_history.py")
run152_fixture = _load("run153_run152_fixture", HERE / "test_witness_publication.py")
NOW = datetime(2026, 9, 5, 6, 0, tzinfo=timezone.utc)
HISTORY_ID = "scikit-plots/release-history"
COLLECTOR = "ci/history-collector"
ACTIVE_KEY = "rekor/key-2026-a"
NEW_KEY = "rekor/key-2026-b"
AUTHORITY = "release/history-policy-authority"
REPLICAS = [("history/replica-a", "replica-operator-a"), ("history/replica-b", "replica-operator-b"), ("history/replica-c", "replica-operator-c")]
ARCHIVES = [("history/archive-a", "archive-operator-a"), ("history/archive-b", "archive-operator-b")]


def _canonical(value: dict) -> bytes:
    return history._canonical_bytes(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _witnessed(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return run152_fixture._run(tmp_path, output_name="run152-witnessed")[1]


def _genesis(tmp_path: Path, witnessed: Path, *, key_id: str = ACTIVE_KEY) -> tuple[Path, Path]:
    previous = json.loads((witnessed / "previous-transparency-checkpoint.json").read_text())
    bundle = {
        "schemaVersion": 1,
        "predicateType": "https://scikit-plots.org/attestations/release-history/v1",
        "status": "preserved-history",
        "historyId": HISTORY_ID,
        "genesis": {
            "logId": previous["logId"],
            "checkpoint": previous["checkpoint"],
            "activeLogKeyId": key_id,
            "policyAuthorityIdentity": AUTHORITY,
            "bootstrapEvidenceSha256": "b" * 64,
        },
        "entries": [],
    }
    bundle_path = tmp_path / "previous-release-history-bundle.json"
    bundle_path.write_bytes(_canonical(bundle))
    state = {
        "schemaVersion": 1,
        "historyId": HISTORY_ID,
        "sequence": 0,
        "logId": previous["logId"],
        "activeLogKeyId": key_id,
        "policyAuthorityIdentity": AUTHORITY,
        "checkpoint": previous["checkpoint"],
        "bundleSha256": _sha(bundle_path),
        "chainHeadSha256": history._history_head(HISTORY_ID, bundle["genesis"], []),
        "revokedLogKeyIds": [],
    }
    state_path = tmp_path / "previous-trusted-history-state.json"
    state_path.write_bytes(_canonical(state))
    return state_path, bundle_path


class FakeReplica:
    def __init__(self, identity: str, operator: str, *, key_id: str = ACTIVE_KEY):
        self.identity = identity
        self.operator = operator
        self.key_id = key_id
        self.unavailable = False
        self.checkpoint_override = None
        self.previous_override = None
        self.history_override = None
        self.read_only = True
        self.log_reused = False
        self.publisher_reused = False
        self.writer_reused = False
        self.proof_override = None
        self.witness_anchor_override = None

    def __call__(self, request: dict) -> dict:
        replica = {
            "identity": self.identity,
            "operator": self.operator,
            "readOnly": self.read_only,
            "logCredentialsReused": self.log_reused,
            "publisherCredentialsReused": self.publisher_reused,
            "historyWriterCredentialsReused": self.writer_reused,
        }
        base = {
            "schemaVersion": 1,
            "operation": "observe",
            "observationId": request["observationId"],
            "historyId": request["historyId"],
            "replica": replica,
            "observedAt": NOW.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        if self.unavailable:
            return {**base, "status": "unavailable", "reason": "temporarily-unreachable"}
        return {
            **base,
            "status": "observed",
            "history": self.history_override or {
                "sequence": request["sequence"] - 1,
                "previousBundleSha256": request["history"]["previousBundleSha256"],
                "previousChainHeadSha256": request["history"]["previousChainHeadSha256"],
            },
            "log": {"logId": request["log"]["logId"], "keyId": self.key_id},
            "checkpoint": self.checkpoint_override or dict(request["checkpoint"]),
            "previousCheckpoint": self.previous_override or dict(request["previousCheckpoint"]),
            "witnessAnchor": self.witness_anchor_override or dict(request["witnessAnchor"]),
            "proof": self.proof_override or {
                "currentCheckpointSignatureVerified": True,
                "consistencyFromPreviousVerified": True,
                "priorHistoryHeadVerified": True,
                "independentGossipSource": True,
                "logKeyIdentityVerified": True,
                "witnessRecordRemoteReadbackVerified": True,
            },
        }


class FakeArchive:
    def __init__(self, identity: str, operator: str):
        self.identity = identity
        self.operator = operator
        self.remote: dict[str, bytes] = {}
        self.locator = f"mock://{operator}/release-history-bundle.json"
        self.mutate_local = False
        self.writer_reused = False
        self.log_reused = False
        self.create_only = True
        self.readback = True

    def __call__(self, request: dict) -> dict:
        artifact = request["artifact"]
        if request["operation"] == "bind":
            path = Path(artifact["localPath"])
            data = path.read_bytes()
            if self.mutate_local:
                path.chmod(0o644)
                path.write_bytes(data + b"tamper")
            if artifact["name"] in self.remote:
                assert self.remote[artifact["name"]] == data
                status = "present"
            else:
                self.remote[artifact["name"]] = data
                status = "created"
        else:
            data = self.remote[artifact["name"]]
            assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
            assert len(data) == artifact["size"]
            status = "present"
        return {
            "schemaVersion": 1,
            "operation": request["operation"],
            "archiveId": request["archiveId"],
            "historyId": request["historyId"],
            "sequence": request["sequence"],
            "status": status,
            "archive": {
                "identity": self.identity,
                "operator": self.operator,
                "historyWriterCredentialsReused": self.writer_reused,
                "transparencyLogCredentialsReused": self.log_reused,
            },
            "artifact": {k: artifact[k] for k in ("name", "sha256", "size")},
            "guarantees": {
                "createOnly": self.create_only,
                "overwrite": False,
                "remoteReadbackVerified": self.readback,
                "immutability": "object-lock",
                "locator": self.locator,
            },
            "verifiedAt": NOW.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


def _transition(tmp_path: Path, *, reason: str = "scheduled-rotation", old_continuity: bool = True, emergency: bool = False) -> Path:
    path = tmp_path / "key-transition.json"
    path.write_bytes(_canonical({
        "schemaVersion": 1,
        "historyId": HISTORY_ID,
        "logId": run152_fixture.LOG_ID,
        "transitionId": "rotate-2026-09-b",
        "fromKeyId": ACTIVE_KEY,
        "toKeyId": NEW_KEY,
        "reason": reason,
        "effectiveTreeSize": 43,
        "revokeFromKey": True,
        "authority": {
            "identity": AUTHORITY,
            "policySignatureVerified": True,
            "oldKeyContinuityVerified": old_continuity,
            "newKeyProofVerified": True,
            "emergencyRecoveryAuthorized": emergency,
        },
        "evidenceSha256": "e" * 64,
    }))
    return path


def _run(tmp_path: Path, *, replicas=None, quorum=2, archives=None, key_transition=None, output_name="run153-history"):
    witnessed = _witnessed(tmp_path / "witness-source")
    state, bundle = _genesis(tmp_path, witnessed)
    replica_defs = replicas or [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    archive_defs = archives or [(identity, operator, FakeArchive(identity, operator)) for identity, operator in ARCHIVES]
    out = tmp_path / output_name
    result = history.preserve_release_history(
        witnessed_dir=witnessed,
        previous_state=state,
        previous_bundle=bundle,
        output_dir=out,
        collector_identity=COLLECTOR,
        replicas=replica_defs,
        replica_quorum=quorum,
        archives=archive_defs,
        key_transition=key_transition,
        now=NOW,
    )
    return result, out, witnessed, state, bundle, replica_defs, archive_defs


def test_run153_end_to_end_requires_gossip_quorum_and_independent_archives(tmp_path: Path):
    result, out, *_ = _run(tmp_path)
    assert result["ok"] is True
    assert result["sequence"] == 1
    assert result["replica_observed"] == 3
    assert result["archive_count"] == 2
    verified = history.verify_history_bundle(bundle_path=out / "release-history-bundle.json", state_path=out / "trusted-history-state.json")
    assert verified["sequence"] == 1
    receipt = json.loads((out / "release-history-preservation-receipt.json").read_text())
    assert receipt["replicaQuorum"]["threshold"] == 2
    assert len(receipt["archives"]) == 2


def test_run153_bundle_and_state_are_deterministic_across_create_only_retry(tmp_path: Path):
    archives = [(identity, operator, FakeArchive(identity, operator)) for identity, operator in ARCHIVES]
    result1, out1, *_ = _run(tmp_path / "a", archives=archives, output_name="out-a")
    result2, out2, *_ = _run(tmp_path / "b", archives=archives, output_name="out-b")
    assert result1["bundle_sha256"] == result2["bundle_sha256"]
    assert result1["state_sha256"] == result2["state_sha256"]
    assert (out1 / "release-history-bundle.json").read_bytes() == (out2 / "release-history-bundle.json").read_bytes()
    bind_results = [json.loads(p.read_text())["status"] for p in sorted((out2 / "archive-results").glob("*.bind.json"))]
    assert bind_results == ["present", "present"]


def test_run153_allows_one_explicitly_unavailable_replica_when_quorum_still_holds(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    defs[2][2].unavailable = True
    result, out, *_ = _run(tmp_path, replicas=defs, quorum=2)
    assert result["replica_observed"] == 2
    receipt = json.loads((out / "release-history-preservation-receipt.json").read_text())
    assert receipt["replicaQuorum"]["unavailable"] == 1


def test_run153_rejects_missing_replica_quorum(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    defs[1][2].unavailable = True
    defs[2][2].unavailable = True
    with pytest.raises(history.HistoryError, match="HISTORY_REPLICA_QUORUM_NOT_MET"):
        _run(tmp_path, replicas=defs, quorum=2)


@pytest.mark.parametrize("field,code", [("checkpoint", "HISTORY_REPLICA_CURRENT_CHECKPOINT_MISMATCH"), ("previous", "HISTORY_REPLICA_PREVIOUS_CHECKPOINT_MISMATCH")])
def test_run153_rejects_gossip_fork_or_history_split_view(tmp_path: Path, field: str, code: str):
    defs = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    if field == "checkpoint":
        defs[1][2].checkpoint_override = {"treeSize": 43, "rootHash": "f" * 64, "signedCheckpointSha256": "a" * 64}
    else:
        defs[1][2].previous_override = {"treeSize": 41, "rootHash": "f" * 64, "signedCheckpointSha256": "8" * 64}
    with pytest.raises(history.HistoryError, match=code):
        _run(tmp_path, replicas=defs)


def test_run153_rejects_log_key_equivocation_between_replicas(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    defs[2][2].key_id = NEW_KEY
    with pytest.raises(history.HistoryError, match="HISTORY_REPLICA_LOG_KEY_EQUIVOCATION"):
        _run(tmp_path, replicas=defs)


def test_run153_requires_explicit_authorized_key_transition(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator, key_id=NEW_KEY)) for identity, operator in REPLICAS]
    with pytest.raises(history.HistoryError, match="HISTORY_LOG_KEY_CHANGE_WITHOUT_TRANSITION"):
        _run(tmp_path, replicas=defs)


def test_run153_scheduled_key_rotation_revokes_old_key_and_is_offline_verifiable(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator, key_id=NEW_KEY)) for identity, operator in REPLICAS]
    transition = _transition(tmp_path)
    result, out, *_ = _run(tmp_path, replicas=defs, key_transition=transition)
    assert result["ok"] is True
    state = json.loads((out / "trusted-history-state.json").read_text())
    assert state["activeLogKeyId"] == NEW_KEY
    assert state["revokedLogKeyIds"] == [ACTIVE_KEY]
    history.verify_history_bundle(bundle_path=out / "release-history-bundle.json", state_path=out / "trusted-history-state.json")


def test_run153_compromise_recovery_requires_emergency_authority(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator, key_id=NEW_KEY)) for identity, operator in REPLICAS]
    transition = _transition(tmp_path, reason="compromise-recovery", old_continuity=False, emergency=False)
    with pytest.raises(history.HistoryError, match="KEY_TRANSITION_EMERGENCY_AUTHORITY_INVALID"):
        _run(tmp_path, replicas=defs, key_transition=transition)


def test_run153_rejects_replay_of_revoked_key_in_offline_history(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator, key_id=NEW_KEY)) for identity, operator in REPLICAS]
    _, out, *_ = _run(tmp_path, replicas=defs, key_transition=_transition(tmp_path))
    bundle = json.loads((out / "release-history-bundle.json").read_text())
    prior = bundle["entries"][0]
    replay = json.loads(json.dumps(prior))
    replay["sequence"] = 2
    replay["release"] = {"releaseId": "run153-next", "publicationId": "pub-run153-next", "sourceRevision": "c" * 40}
    replay["subjects"] = {"finalPublicationRecordSha256": "1" * 64, "witnessRecordSha256": "2" * 64, "witnessReceiptSha256": "3" * 64}
    replay["log"]["logKeyId"] = ACTIVE_KEY
    replay["log"]["previousCheckpointSha256"] = hashlib.sha256(_canonical({"schemaVersion": 1, "logId": run152_fixture.LOG_ID, "checkpoint": prior["log"]["checkpoint"]})).hexdigest()
    replay["log"]["checkpoint"] = {"treeSize": 45, "rootHash": "4" * 64, "signedCheckpointSha256": "5" * 64}
    replay["log"]["acceptedCheckpointSha256"] = hashlib.sha256(_canonical({"schemaVersion": 1, "logId": run152_fixture.LOG_ID, "checkpoint": replay["log"]["checkpoint"]})).hexdigest()
    replay["keyTransition"] = None
    bundle["entries"].append(replay)
    with pytest.raises(history.HistoryError, match="HISTORY_ENTRY_REVOKED_KEY_REUSE"):
        history._validate_bundle(bundle)


def test_run153_rejects_release_or_witness_replay_in_offline_history(tmp_path: Path):
    _, out, *_ = _run(tmp_path)
    bundle = json.loads((out / "release-history-bundle.json").read_text())
    replay = json.loads(json.dumps(bundle["entries"][0]))
    replay["sequence"] = 2
    replay["log"]["previousCheckpointSha256"] = hashlib.sha256(_canonical({"schemaVersion": 1, "logId": run152_fixture.LOG_ID, "checkpoint": bundle["entries"][0]["log"]["checkpoint"]})).hexdigest()
    replay["log"]["checkpoint"] = {"treeSize": 45, "rootHash": "4" * 64, "signedCheckpointSha256": "5" * 64}
    replay["log"]["acceptedCheckpointSha256"] = hashlib.sha256(_canonical({"schemaVersion": 1, "logId": run152_fixture.LOG_ID, "checkpoint": replay["log"]["checkpoint"]})).hexdigest()
    bundle["entries"].append(replay)
    with pytest.raises(history.HistoryError, match="HISTORY_ENTRY_RELEASE_REPLAY"):
        history._validate_bundle(bundle)


def test_run153_rejects_previous_history_tamper(tmp_path: Path):
    witnessed = _witnessed(tmp_path / "source")
    state, bundle = _genesis(tmp_path, witnessed)
    doc = json.loads(bundle.read_text())
    doc["genesis"]["bootstrapEvidenceSha256"] = "c" * 64
    bundle.write_bytes(_canonical(doc))
    replicas = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    archives = [(identity, operator, FakeArchive(identity, operator)) for identity, operator in ARCHIVES]
    with pytest.raises(history.HistoryError, match="PREVIOUS_HISTORY_HASH_MISMATCH"):
        history.preserve_release_history(witnessed_dir=witnessed, previous_state=state, previous_bundle=bundle, output_dir=tmp_path / "out", collector_identity=COLLECTOR, replicas=replicas, replica_quorum=2, archives=archives, now=NOW)


def test_run153_rejects_run152_sidecar_tamper_before_gossip(tmp_path: Path):
    witnessed = _witnessed(tmp_path / "source")
    state, bundle = _genesis(tmp_path, witnessed)
    sidecar = next((witnessed / "witness-results").iterdir())
    sidecar.write_bytes(_canonical({"tampered": True}))
    replicas = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    archives = [(identity, operator, FakeArchive(identity, operator)) for identity, operator in ARCHIVES]
    with pytest.raises(history.HistoryError, match="RUN152_WITNESS_EVIDENCE_HASH_MISMATCH"):
        history.preserve_release_history(witnessed_dir=witnessed, previous_state=state, previous_bundle=bundle, output_dir=tmp_path / "out", collector_identity=COLLECTOR, replicas=replicas, replica_quorum=2, archives=archives, now=NOW)


def test_run153_rejects_archive_mutation_and_locator_collision(tmp_path: Path):
    bad = [(identity, operator, FakeArchive(identity, operator)) for identity, operator in ARCHIVES]
    bad[0][2].mutate_local = True
    with pytest.raises(history.HistoryError, match="HISTORY_BUNDLE_CHANGED_DURING_ARCHIVE"):
        _run(tmp_path / "mutation", archives=bad)

    collision = [(identity, operator, FakeArchive(identity, operator)) for identity, operator in ARCHIVES]
    collision[1][2].locator = collision[0][2].locator
    with pytest.raises(history.HistoryError, match="HISTORY_ARCHIVE_LOCATOR_COLLISION"):
        _run(tmp_path / "collision", archives=collision)


def test_run153_enforces_identity_and_operator_separation(tmp_path: Path):
    replicas = [
        ("history/replica-a", "same-operator", FakeReplica("history/replica-a", "same-operator")),
        ("history/replica-b", "same-operator", FakeReplica("history/replica-b", "same-operator")),
        ("history/replica-c", "same-operator", FakeReplica("history/replica-c", "same-operator")),
    ]
    with pytest.raises(history.HistoryError, match="HISTORY_REPLICA_OPERATOR_COUNT_INVALID"):
        _run(tmp_path / "replica-ops", replicas=replicas)

    archives = [
        ("history/archive-a", "same-archive", FakeArchive("history/archive-a", "same-archive")),
        ("history/archive-b", "same-archive", FakeArchive("history/archive-b", "same-archive")),
    ]
    with pytest.raises(history.HistoryError, match="HISTORY_ARCHIVE_OPERATOR_COUNT_INVALID"):
        _run(tmp_path / "archive-ops", archives=archives)


def test_run153_offline_verifier_detects_bundle_or_state_drift(tmp_path: Path):
    _, out, *_ = _run(tmp_path)
    state_path = out / "trusted-history-state.json"
    state = json.loads(state_path.read_text())
    state["chainHeadSha256"] = "0" * 64
    state_path.write_bytes(_canonical(state))
    with pytest.raises(history.HistoryError, match="HISTORY_STATE_HASH_MISMATCH"):
        history.verify_history_bundle(bundle_path=out / "release-history-bundle.json", state_path=state_path)



def test_run153_rejects_replica_that_cannot_rebind_witness_anchor(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    defs[1][2].witness_anchor_override = {
        "locator": "mock://attacker/release-transparency-witness-record.json",
        "record": {"name": "release-transparency-witness-record.json", "sha256": "0" * 64, "size": 1},
    }
    with pytest.raises(history.HistoryError, match="HISTORY_REPLICA_WITNESS_ANCHOR_MISMATCH"):
        _run(tmp_path, replicas=defs)


def test_run153_offline_bundle_recomputes_previous_checkpoint_hash(tmp_path: Path):
    _, out, *_ = _run(tmp_path)
    bundle = json.loads((out / "release-history-bundle.json").read_text())
    bundle["entries"][0]["log"]["previousCheckpointSha256"] = "0" * 64
    with pytest.raises(history.HistoryError, match="HISTORY_ENTRY_PREVIOUS_CHECKPOINT_HASH_MISMATCH"):
        history._validate_bundle(bundle)


def test_run153_key_transition_cannot_claim_effective_time_before_previous_checkpoint(tmp_path: Path):
    defs = [(identity, operator, FakeReplica(identity, operator, key_id=NEW_KEY)) for identity, operator in REPLICAS]
    transition = _transition(tmp_path)
    doc = json.loads(transition.read_text())
    doc["effectiveTreeSize"] = 41
    transition.write_bytes(_canonical(doc))
    with pytest.raises(history.HistoryError, match="KEY_TRANSITION_NOT_EFFECTIVE"):
        _run(tmp_path, replicas=defs, key_transition=transition)


def test_run153_rejects_secret_bearing_archive_locator(tmp_path: Path):
    archives = [(identity, operator, FakeArchive(identity, operator)) for identity, operator in ARCHIVES]
    archives[1][2].locator = "https://archive.example/history?token=secret"
    with pytest.raises(history.HistoryError, match="HISTORY_ARCHIVE_LOCATOR_INVALID"):
        _run(tmp_path, archives=archives)


def test_run153_bundle_is_stable_when_replica_recovers_between_retries(tmp_path: Path):
    archives = [(identity, operator, FakeArchive(identity, operator)) for identity, operator in ARCHIVES]
    first_replicas = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    first_replicas[2][2].unavailable = True
    result1, out1, *_ = _run(tmp_path / "a", replicas=first_replicas, quorum=2, archives=archives, output_name="out-a")
    second_replicas = [(identity, operator, FakeReplica(identity, operator)) for identity, operator in REPLICAS]
    result2, out2, *_ = _run(tmp_path / "b", replicas=second_replicas, quorum=2, archives=archives, output_name="out-b")
    assert result1["bundle_sha256"] == result2["bundle_sha256"]
    assert result1["state_sha256"] == result2["state_sha256"]
    assert (out1 / "release-history-bundle.json").read_bytes() == (out2 / "release-history-bundle.json").read_bytes()
    assert [json.loads(p.read_text())["status"] for p in sorted((out2 / "archive-results").glob("*.bind.json"))] == ["present", "present"]

def test_run153_policy_and_documentation_define_durable_history_boundary():
    policy = (SECURITY / "release_history_policy.toml").read_text()
    guide = (SECURITY / "RELEASE_HISTORY_GUIDE.md").read_text()
    gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    assert "min_replicas = 3" in policy
    assert "min_archives = 2" in policy
    assert "allow_log_key_change_without_transition = false" in policy
    assert "N-of-M" in guide
    assert "offline" in guide.lower()
    assert "compromise-recovery" in guide
    assert "Run 153" in gates
