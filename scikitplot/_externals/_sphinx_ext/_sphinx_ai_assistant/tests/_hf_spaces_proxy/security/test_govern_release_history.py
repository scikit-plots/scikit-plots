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


gov = _load("run154_governance", SECURITY / "govern_release_history.py")
run153 = _load("run154_run153_fixture", HERE / "test_preserve_release_history.py")
NOW = datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc)
GOVERNANCE_ID = "scikit-plots/release-governance"
POLICY_AUTHORITIES = [
    {"identity": "governance/policy-a", "operator": "governance-operator-a", "keyId": "governance/key-a"},
    {"identity": "governance/policy-b", "operator": "governance-operator-b", "keyId": "governance/key-b"},
    {"identity": "governance/policy-c", "operator": "governance-operator-c", "keyId": "governance/key-c"},
]
EMERGENCY_AUTHORITIES = [
    {"identity": "governance/emergency-a", "operator": "emergency-operator-a", "keyId": "emergency/key-a"},
    {"identity": "governance/emergency-b", "operator": "emergency-operator-b", "keyId": "emergency/key-b"},
    {"identity": "governance/emergency-c", "operator": "emergency-operator-c", "keyId": "emergency/key-c"},
]
GOV_ARCHIVES = [("governance/archive-a", "governance-archive-operator-a"), ("governance/archive-b", "governance-archive-operator-b")]
RECOVERY_SOURCES = [("recovery/source-a", "recovery-operator-a"), ("recovery/source-b", "recovery-operator-b"), ("recovery/source-c", "recovery-operator-c")]


def _canonical(value: dict) -> bytes:
    return gov._canonical_bytes(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history(tmp_path: Path, *, alternate_key: bool = False) -> Path:
    if not alternate_key:
        return run153._run(tmp_path, output_name="run153-history")[1]
    tmp_path.mkdir(parents=True, exist_ok=True)
    replicas = [(identity, operator, run153.FakeReplica(identity, operator, key_id=run153.NEW_KEY)) for identity, operator in run153.REPLICAS]
    transition = run153._transition(tmp_path, reason="scheduled-rotation", old_continuity=True, emergency=False)
    return run153._run(tmp_path, replicas=replicas, key_transition=transition, output_name="run153-history-alt")[1]


def _history_binding(history_dir: Path) -> dict:
    state = json.loads((history_dir / "trusted-history-state.json").read_text())
    return {
        "sequence": state["sequence"],
        "stateSha256": _sha(history_dir / "trusted-history-state.json"),
        "bundleSha256": _sha(history_dir / "release-history-bundle.json"),
        "chainHeadSha256": state["chainHeadSha256"],
    }


def _base_policy(*, version: int = 1, policy_members=None, emergency_members=None, replicas=None, archives=None) -> dict:
    policy_members = policy_members or POLICY_AUTHORITIES
    emergency_members = emergency_members or EMERGENCY_AUTHORITIES
    replicas = replicas or [{"identity": i, "operator": o} for i, o in run153.REPLICAS]
    archives = archives or [{"identity": i, "operator": o} for i, o in run153.ARCHIVES]
    return {
        "policyVersion": version,
        "policyAuthority": {"threshold": 2, "members": policy_members},
        "emergencyAuthority": {"threshold": 2, "members": emergency_members},
        "historyReplicas": {"threshold": 2, "members": replicas},
        "historyArchives": {"threshold": 2, "members": archives},
    }


def _genesis(tmp_path: Path, history_dir: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "governance-genesis.json"
    path.write_bytes(_canonical({
        "schemaVersion": 1,
        "governanceId": GOVERNANCE_ID,
        "historyId": run153.HISTORY_ID,
        "bootstrapEvidenceSha256": "f" * 64,
        "policy": _base_policy(),
    }))
    return path, _sha(path)


def _initialize(tmp_path: Path, history_dir: Path) -> Path:
    genesis, expected = _genesis(tmp_path, history_dir)
    out = tmp_path / "initialized-governance"
    result = gov.initialize_governance(history_dir=history_dir, genesis_path=genesis, expected_genesis_sha256=expected, output_dir=out)
    assert result["ok"] is True
    return out


def _proposal(tmp_path: Path, history_dir: Path, init_dir: Path, *, reason="membership-change", selected=None, revocations=None, next_policy=None, transition_id="governance-2026-09-01") -> Path:
    selected = selected or ["governance/key-a", "governance/key-b"]
    revocations = revocations or []
    if next_policy is None:
        archives = [{"identity": i, "operator": o} for i, o in run153.ARCHIVES] + [{"identity": "history/archive-c", "operator": "archive-operator-c"}]
        next_policy = _base_policy(version=2, archives=archives)
    previous_state = init_dir / "trusted-governance-state.json"
    path = tmp_path / "governance-transition-proposal.json"
    path.write_bytes(_canonical({
        "schemaVersion": 1,
        "governanceId": GOVERNANCE_ID,
        "transitionId": transition_id,
        "fromEpoch": json.loads(previous_state.read_text())["epoch"],
        "toEpoch": json.loads(previous_state.read_text())["epoch"] + 1,
        "reason": reason,
        "history": _history_binding(history_dir),
        "previousGovernanceStateSha256": _sha(previous_state),
        "selectedApproverKeyIds": selected,
        "revokedAuthorityKeyIds": revocations,
        "nextPolicy": next_policy,
    }))
    return path


def _approval(tmp_path: Path, proposal: Path, member: dict, *, role="policy", suffix="") -> Path:
    pdoc = json.loads(proposal.read_text())
    path = tmp_path / f"approval-{member['keyId'].replace('/', '-')}{suffix}.json"
    path.write_bytes(_canonical({
        "schemaVersion": 1,
        "governanceId": GOVERNANCE_ID,
        "transitionId": pdoc["transitionId"],
        "proposalSha256": _sha(proposal),
        "approver": member,
        "role": role,
        "decision": "approve",
        "signatureVerified": True,
        "signatureEvidenceSha256": hashlib.sha256((member["keyId"] + pdoc["transitionId"]).encode()).hexdigest(),
        "signedAt": NOW.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }))
    return path


class FakeGovernanceArchive:
    def __init__(self, identity: str, operator: str):
        self.identity = identity
        self.operator = operator
        self.remote: dict[str, bytes] = {}
        self.locator = f"mock://{operator}/release-governance-recovery-snapshot.json"
        self.mutate_local = False
        self.writer_reused = False
        self.history_writer_reused = False
        self.readback = True
        self.create_only = True

    def __call__(self, request: dict) -> dict:
        artifact = request["artifact"]
        if request["operation"] == "bind":
            path = Path(artifact["localPath"])
            data = path.read_bytes()
            if self.mutate_local:
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
            "governanceId": request["governanceId"],
            "epoch": request["epoch"],
            "status": status,
            "archive": {
                "identity": self.identity,
                "operator": self.operator,
                "governanceWriterCredentialsReused": self.writer_reused,
                "historyWriterCredentialsReused": self.history_writer_reused,
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


class FakeRecoverySource:
    def __init__(self, identity: str, operator: str, history_dir: Path):
        self.identity = identity
        self.operator = operator
        self.state = json.loads((history_dir / "trusted-history-state.json").read_text())
        self.bundle = json.loads((history_dir / "release-history-bundle.json").read_text())
        self.unavailable = False
        self.read_only = True
        self.history_writer_reused = False
        self.governance_writer_reused = False
        self.locator = f"mock://{operator}/history-recovery-snapshot.json"

    def __call__(self, request: dict) -> dict:
        base = {
            "schemaVersion": 1,
            "operation": "recover-history",
            "recoveryId": request["recoveryId"],
            "historyId": request["historyId"],
            "source": {
                "identity": self.identity,
                "operator": self.operator,
                "readOnly": self.read_only,
                "historyWriterCredentialsReused": self.history_writer_reused,
                "governanceWriterCredentialsReused": self.governance_writer_reused,
            },
            "observedAt": NOW.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        if self.unavailable:
            return {**base, "status": "unavailable", "reason": "archive-temporarily-unreachable"}
        return {
            **base,
            "status": "recovered",
            "snapshot": {"schemaVersion": 1, "historyState": self.state, "historyBundle": self.bundle},
            "locator": self.locator,
            "proof": {"immutableArchiveReadbackVerified": True, "independentArchive": True, "canonicalBytesReturned": True},
        }


def _transition_run(tmp_path: Path, *, history_dir=None, reason="membership-change", selected=None, revocations=None, next_policy=None, approval_members=None, role="policy", archives=None, transition_id="governance-2026-09-01"):
    history_dir = history_dir or _history(tmp_path / "history")
    init_dir = _initialize(tmp_path / "init", history_dir)
    proposal = _proposal(tmp_path, history_dir, init_dir, reason=reason, selected=selected, revocations=revocations, next_policy=next_policy, transition_id=transition_id)
    approval_members = approval_members or POLICY_AUTHORITIES[:2]
    approval_paths = [_approval(tmp_path, proposal, m, role=role) for m in approval_members]
    archives = archives or [(i, o, FakeGovernanceArchive(i, o)) for i, o in GOV_ARCHIVES]
    out = tmp_path / "governed"
    result = gov.apply_governance_transition(
        history_dir=history_dir,
        previous_state=init_dir / "trusted-governance-state.json",
        previous_bundle=init_dir / "release-governance-bundle.json",
        proposal_path=proposal,
        approval_paths=approval_paths,
        output_dir=out,
        governance_archives=archives,
        now=NOW,
    )
    return result, out, history_dir, init_dir, proposal, approval_paths, archives


def test_run154_genesis_requires_explicit_external_hash_pin(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    genesis, _ = _genesis(tmp_path, history_dir)
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_GENESIS_PIN_MISMATCH"):
        gov.initialize_governance(history_dir=history_dir, genesis_path=genesis, expected_genesis_sha256="0" * 64, output_dir=tmp_path / "bad")


def test_run154_genesis_rebinds_run153_replica_and_archive_membership(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    genesis, expected = _genesis(tmp_path, history_dir)
    doc = json.loads(genesis.read_text())
    doc["policy"]["historyReplicas"]["members"][0]["identity"] = "history/other-replica"
    genesis.write_bytes(_canonical(doc)); expected = _sha(genesis)
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_GENESIS_REPLICA_MEMBERSHIP_MISMATCH"):
        gov.initialize_governance(history_dir=history_dir, genesis_path=genesis, expected_genesis_sha256=expected, output_dir=tmp_path / "bad")


def test_run154_threshold_governed_membership_change_and_offline_verification(tmp_path: Path):
    result, out, *_ = _transition_run(tmp_path)
    assert result["ok"] is True
    assert result["epoch"] == 1
    assert result["policy_version"] == 2
    verified = gov.verify_governance_bundle(bundle_path=out / "release-governance-bundle.json", state_path=out / "trusted-governance-state.json")
    assert verified["epoch"] == 1
    state = json.loads((out / "trusted-governance-state.json").read_text())
    assert len(state["policy"]["historyArchives"]["members"]) == 3


def test_run154_proposal_is_bound_to_exact_history_and_previous_governance_state(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    init_dir = _initialize(tmp_path / "init", history_dir)
    proposal = _proposal(tmp_path, history_dir, init_dir)
    doc = json.loads(proposal.read_text()); doc["history"]["chainHeadSha256"] = "0" * 64; proposal.write_bytes(_canonical(doc))
    approvals = [_approval(tmp_path, proposal, m) for m in POLICY_AUTHORITIES[:2]]
    archives = [(i, o, FakeGovernanceArchive(i, o)) for i, o in GOV_ARCHIVES]
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_PROPOSAL_HISTORY_MISMATCH"):
        gov.apply_governance_transition(history_dir=history_dir, previous_state=init_dir / "trusted-governance-state.json", previous_bundle=init_dir / "release-governance-bundle.json", proposal_path=proposal, approval_paths=approvals, output_dir=tmp_path / "bad", governance_archives=archives, now=NOW)


def test_run154_rejects_missing_or_swapped_threshold_approval(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    init_dir = _initialize(tmp_path / "init", history_dir)
    proposal = _proposal(tmp_path, history_dir, init_dir)
    approval = _approval(tmp_path, proposal, POLICY_AUTHORITIES[0])
    archives = [(i, o, FakeGovernanceArchive(i, o)) for i, o in GOV_ARCHIVES]
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_APPROVAL_COUNT_INVALID"):
        gov.apply_governance_transition(history_dir=history_dir, previous_state=init_dir / "trusted-governance-state.json", previous_bundle=init_dir / "release-governance-bundle.json", proposal_path=proposal, approval_paths=[approval], output_dir=tmp_path / "bad", governance_archives=archives, now=NOW)


def test_run154_approval_cannot_be_replayed_for_another_proposal(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    init_dir = _initialize(tmp_path / "init", history_dir)
    proposal1 = _proposal(tmp_path, history_dir, init_dir, transition_id="transition-one")
    approval = _approval(tmp_path, proposal1, POLICY_AUTHORITIES[0])
    proposal2 = _proposal(tmp_path, history_dir, init_dir, transition_id="transition-two")
    doc, _ = gov._read(approval, "TEST")
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_APPROVAL_SCHEMA_INVALID"):
        gov._validate_approval(doc, governance_id=GOVERNANCE_ID, transition_id="transition-two", proposal_sha=_sha(proposal2), role="policy", allowed_members={m["keyId"]: m for m in POLICY_AUTHORITIES}, revoked=set(), now=NOW)


def test_run154_compromise_recovery_uses_emergency_council_and_revokes_compromised_policy_key(tmp_path: Path):
    new_policy_members = [POLICY_AUTHORITIES[1], POLICY_AUTHORITIES[2], {"identity": "governance/policy-d", "operator": "governance-operator-d", "keyId": "governance/key-d"}]
    next_policy = _base_policy(version=2, policy_members=new_policy_members)
    result, out, *_ = _transition_run(tmp_path, reason="authority-compromise-recovery", selected=["emergency/key-a", "emergency/key-b"], revocations=["governance/key-a"], next_policy=next_policy, approval_members=EMERGENCY_AUTHORITIES[:2], role="emergency", transition_id="compromise-recovery-1")
    assert result["ok"] is True
    state = json.loads((out / "trusted-governance-state.json").read_text())
    assert state["revokedAuthorityKeyIds"] == ["governance/key-a"]
    assert "governance/key-a" not in {m["keyId"] for m in state["policy"]["policyAuthority"]["members"]}


def test_run154_compromise_recovery_cannot_use_normal_policy_quorum(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    init_dir = _initialize(tmp_path / "init", history_dir)
    next_policy_members = [POLICY_AUTHORITIES[1], POLICY_AUTHORITIES[2], {"identity": "governance/policy-d", "operator": "governance-operator-d", "keyId": "governance/key-d"}]
    proposal = _proposal(tmp_path, history_dir, init_dir, reason="authority-compromise-recovery", selected=["emergency/key-a", "emergency/key-b"], revocations=["governance/key-a"], next_policy=_base_policy(version=2, policy_members=next_policy_members), transition_id="compromise-recovery-2")
    approvals = [_approval(tmp_path, proposal, m, role="policy") for m in POLICY_AUTHORITIES[:2]]
    archives = [(i, o, FakeGovernanceArchive(i, o)) for i, o in GOV_ARCHIVES]
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_APPROVAL_DECISION_INVALID"):
        gov.apply_governance_transition(history_dir=history_dir, previous_state=init_dir/"trusted-governance-state.json", previous_bundle=init_dir/"release-governance-bundle.json", proposal_path=proposal, approval_paths=approvals, output_dir=tmp_path/"bad", governance_archives=archives, now=NOW)


def test_run154_compromise_recovery_must_revoke_current_policy_key(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    init_dir = _initialize(tmp_path / "init", history_dir)
    proposal = _proposal(tmp_path, history_dir, init_dir, reason="authority-compromise-recovery", selected=["emergency/key-a", "emergency/key-b"], revocations=[], next_policy=_base_policy(version=2), transition_id="bad-recovery")
    approvals = [_approval(tmp_path, proposal, m, role="emergency") for m in EMERGENCY_AUTHORITIES[:2]]
    archives = [(i, o, FakeGovernanceArchive(i, o)) for i, o in GOV_ARCHIVES]
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_COMPROMISE_RECOVERY_MUST_REVOKE_POLICY_KEY"):
        gov.apply_governance_transition(history_dir=history_dir, previous_state=init_dir/"trusted-governance-state.json", previous_bundle=init_dir/"release-governance-bundle.json", proposal_path=proposal, approval_paths=approvals, output_dir=tmp_path/"bad", governance_archives=archives, now=NOW)


def test_run154_offline_verifier_rejects_revoked_key_reintroduced_later(tmp_path: Path):
    new_policy_members = [POLICY_AUTHORITIES[1], POLICY_AUTHORITIES[2], {"identity": "governance/policy-d", "operator": "governance-operator-d", "keyId": "governance/key-d"}]
    _, out, *_ = _transition_run(tmp_path, reason="authority-compromise-recovery", selected=["emergency/key-a", "emergency/key-b"], revocations=["governance/key-a"], next_policy=_base_policy(version=2, policy_members=new_policy_members), approval_members=EMERGENCY_AUTHORITIES[:2], role="emergency", transition_id="compromise-recovery-3")
    bundle = json.loads((out / "release-governance-bundle.json").read_text())
    entry = bundle["entries"][0]
    entry["nextPolicy"]["emergencyAuthority"]["members"][0]["keyId"] = "governance/key-a"
    entry["proposal"]["nextPolicy"] = json.loads(json.dumps(entry["nextPolicy"]))
    entry["proposalSha256"] = hashlib.sha256(_canonical(entry["proposal"])).hexdigest()
    for approval in entry["approvals"]:
        approval["proposalSha256"] = entry["proposalSha256"]
    bad = tmp_path / "tampered-bundle.json"; bad.write_bytes(_canonical(bundle))
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_REVOKED_KEY_IN_NEXT_POLICY"):
        gov.verify_governance_bundle(bundle_path=bad)


def test_run154_recovery_accepts_two_of_three_matching_immutable_archives(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    defs = [(i, o, FakeRecoverySource(i, o, history_dir)) for i, o in RECOVERY_SOURCES]
    defs[2][2].unavailable = True
    out = tmp_path / "recovered"
    result = gov.recover_history(history_id=run153.HISTORY_ID, expected_sequence=1, expected_chain_head_sha256=_history_binding(history_dir)["chainHeadSha256"], output_dir=out, recovery_sources=defs, recovery_quorum=2, now=NOW)
    assert result["ok"] is True
    assert result["observed"] == 2
    assert (out / "recovered-trusted-history-state.json").read_bytes() == (history_dir / "trusted-history-state.json").read_bytes()
    assert (out / "recovered-release-history-bundle.json").read_bytes() == (history_dir / "release-history-bundle.json").read_bytes()


def test_run154_recovery_fails_closed_on_any_available_archive_disagreement(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    alternate = _history(tmp_path / "alternate", alternate_key=True)
    defs = [(RECOVERY_SOURCES[0][0], RECOVERY_SOURCES[0][1], FakeRecoverySource(*RECOVERY_SOURCES[0], history_dir)), (RECOVERY_SOURCES[1][0], RECOVERY_SOURCES[1][1], FakeRecoverySource(*RECOVERY_SOURCES[1], history_dir)), (RECOVERY_SOURCES[2][0], RECOVERY_SOURCES[2][1], FakeRecoverySource(*RECOVERY_SOURCES[2], alternate))]
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_RECOVERY_ARCHIVE_DISAGREEMENT"):
        gov.recover_history(history_id=run153.HISTORY_ID, expected_sequence=1, expected_chain_head_sha256=_history_binding(history_dir)["chainHeadSha256"], output_dir=tmp_path/"bad", recovery_sources=defs, recovery_quorum=2, now=NOW)


def test_run154_recovery_rejects_writer_credentials_or_non_readonly_source(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    defs = [(i, o, FakeRecoverySource(i, o, history_dir)) for i, o in RECOVERY_SOURCES]
    defs[0][2].history_writer_reused = True
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_RECOVERY_SOURCE_AUTHORITY_INVALID"):
        gov.recover_history(history_id=run153.HISTORY_ID, expected_sequence=1, expected_chain_head_sha256=_history_binding(history_dir)["chainHeadSha256"], output_dir=tmp_path/"bad", recovery_sources=defs, recovery_quorum=2, now=NOW)


def test_run154_recovery_requires_operator_diversity(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    defs = [("recovery/source-a", "same-operator", FakeRecoverySource("recovery/source-a", "same-operator", history_dir)), ("recovery/source-b", "same-operator", FakeRecoverySource("recovery/source-b", "same-operator", history_dir)), ("recovery/source-c", "same-operator", FakeRecoverySource("recovery/source-c", "same-operator", history_dir))]
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_RECOVERY_OPERATOR_COUNT_INVALID"):
        gov.recover_history(history_id=run153.HISTORY_ID, expected_sequence=1, expected_chain_head_sha256=_history_binding(history_dir)["chainHeadSha256"], output_dir=tmp_path/"bad", recovery_sources=defs, recovery_quorum=2, now=NOW)


def test_run154_governance_snapshot_is_create_only_archived_and_retry_stable(tmp_path: Path):
    shared = [(i, o, FakeGovernanceArchive(i, o)) for i, o in GOV_ARCHIVES]
    result1, out1, history_dir, init_dir, proposal, approvals, _ = _transition_run(tmp_path/"first", archives=shared)
    # Rebuild identical inputs in another directory but keep exact proposal/approval bytes.
    out2 = tmp_path / "retry" / "governed"; out2.parent.mkdir(parents=True)
    result2 = gov.apply_governance_transition(history_dir=history_dir, previous_state=init_dir/"trusted-governance-state.json", previous_bundle=init_dir/"release-governance-bundle.json", proposal_path=proposal, approval_paths=approvals, output_dir=out2, governance_archives=shared, now=NOW)
    assert result1["bundle_sha256"] == result2["bundle_sha256"]
    assert result1["state_sha256"] == result2["state_sha256"]
    assert result1["recovery_snapshot_sha256"] == result2["recovery_snapshot_sha256"]
    statuses = [json.loads(p.read_text())["status"] for p in sorted((out2/"archive-results").glob("*.bind.json"))]
    assert statuses == ["present", "present"]


def test_run154_rejects_archive_mutation_locator_collision_and_credential_reuse(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    init_dir = _initialize(tmp_path / "init", history_dir)
    proposal = _proposal(tmp_path, history_dir, init_dir)
    approvals = [_approval(tmp_path, proposal, m) for m in POLICY_AUTHORITIES[:2]]
    a = FakeGovernanceArchive(*GOV_ARCHIVES[0]); b = FakeGovernanceArchive(*GOV_ARCHIVES[1]); b.locator = a.locator
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_ARCHIVE_LOCATOR_COLLISION"):
        gov.apply_governance_transition(history_dir=history_dir, previous_state=init_dir/"trusted-governance-state.json", previous_bundle=init_dir/"release-governance-bundle.json", proposal_path=proposal, approval_paths=approvals, output_dir=tmp_path/"bad-locator", governance_archives=[(*GOV_ARCHIVES[0], a), (*GOV_ARCHIVES[1], b)], now=NOW)
    a = FakeGovernanceArchive(*GOV_ARCHIVES[0]); b = FakeGovernanceArchive(*GOV_ARCHIVES[1]); a.writer_reused = True
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_ARCHIVE_AUTHORITY_INVALID"):
        gov.apply_governance_transition(history_dir=history_dir, previous_state=init_dir/"trusted-governance-state.json", previous_bundle=init_dir/"release-governance-bundle.json", proposal_path=proposal, approval_paths=approvals, output_dir=tmp_path/"bad-creds", governance_archives=[(*GOV_ARCHIVES[0], a), (*GOV_ARCHIVES[1], b)], now=NOW)


def test_run154_detects_run153_sidecar_tamper_before_governance(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    sidecar = next((history_dir / "archive-results").glob("*.verify.json"))
    doc = json.loads(sidecar.read_text()); doc["verifiedAt"] = "2026-09-05T09:00:00Z"; sidecar.write_bytes(_canonical(doc))
    genesis, expected = _genesis(tmp_path, history_dir)
    with pytest.raises(gov.GovernanceError, match="RUN153_ARCHIVE_EVIDENCE_REBIND_FAILED"):
        gov.initialize_governance(history_dir=history_dir, genesis_path=genesis, expected_genesis_sha256=expected, output_dir=tmp_path/"bad")


def test_run154_policy_and_documentation_define_governance_and_recovery_boundary():
    policy = gov.POLICY
    assert policy["min_policy_authorities"] >= 3
    assert policy["min_policy_threshold"] >= 2
    assert policy["min_emergency_authorities"] >= 3
    assert policy["min_recovery_sources"] >= 3
    assert policy["min_recovery_quorum"] >= 2
    assert policy["min_governance_archives"] >= 2
    guide = (SECURITY / "RELEASE_GOVERNANCE_GUIDE.md").read_text()
    gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    assert "threshold" in guide.lower() and "compromise" in guide.lower() and "disaster" in guide.lower()
    assert "Run 154" in gates and "governance" in gates.lower()

class FakeGovernanceRecoverySource:
    def __init__(self, identity: str, operator: str, governed_dir: Path):
        self.identity = identity
        self.operator = operator
        self.snapshot = json.loads((governed_dir / "release-governance-recovery-snapshot.json").read_text())
        self.unavailable = False
        self.read_only = True
        self.history_writer_reused = False
        self.governance_writer_reused = False
        self.locator = f"mock://{operator}/release-governance-recovery-snapshot.json"

    def __call__(self, request: dict) -> dict:
        base = {
            "schemaVersion": 1,
            "operation": "recover-governance",
            "recoveryId": request["recoveryId"],
            "governanceId": request["governanceId"],
            "source": {
                "identity": self.identity,
                "operator": self.operator,
                "readOnly": self.read_only,
                "historyWriterCredentialsReused": self.history_writer_reused,
                "governanceWriterCredentialsReused": self.governance_writer_reused,
            },
            "observedAt": NOW.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        if self.unavailable:
            return {**base, "status": "unavailable", "reason": "archive-temporarily-unreachable"}
        return {
            **base,
            "status": "recovered",
            "snapshot": self.snapshot,
            "locator": self.locator,
            "proof": {"immutableArchiveReadbackVerified": True, "independentArchive": True, "canonicalBytesReturned": True},
        }


def test_run154_history_recovery_requires_out_of_band_chain_head_pin(tmp_path: Path):
    history_dir = _history(tmp_path / "history")
    defs = [(i, o, FakeRecoverySource(i, o, history_dir)) for i, o in RECOVERY_SOURCES]
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_RECOVERY_CHAIN_HEAD_PIN_MISMATCH"):
        gov.recover_history(
            history_id=run153.HISTORY_ID,
            expected_sequence=1,
            expected_chain_head_sha256="0" * 64,
            output_dir=tmp_path / "bad-pin",
            recovery_sources=defs,
            recovery_quorum=2,
            now=NOW,
        )


def test_run154_governance_recovery_accepts_two_of_three_matching_snapshots(tmp_path: Path):
    _, governed, *_ = _transition_run(tmp_path / "source")
    state = json.loads((governed / "trusted-governance-state.json").read_text())
    defs = [(i, o, FakeGovernanceRecoverySource(i, o, governed)) for i, o in RECOVERY_SOURCES]
    defs[2][2].unavailable = True
    out = tmp_path / "recovered-governance"
    result = gov.recover_governance(
        governance_id=GOVERNANCE_ID,
        expected_epoch=1,
        expected_chain_head_sha256=state["chainHeadSha256"],
        output_dir=out,
        recovery_sources=defs,
        recovery_quorum=2,
        now=NOW,
    )
    assert result["ok"] is True
    assert result["observed"] == 2
    assert (out / "recovered-trusted-governance-state.json").read_bytes() == (governed / "trusted-governance-state.json").read_bytes()
    assert (out / "recovered-release-governance-bundle.json").read_bytes() == (governed / "release-governance-bundle.json").read_bytes()
    assert (out / "recovered-trusted-history-state.json").read_bytes() == (governed / "trusted-history-state.json").read_bytes()
    assert (out / "recovered-release-history-bundle.json").read_bytes() == (governed / "release-history-bundle.json").read_bytes()


def test_run154_governance_recovery_rejects_wrong_chain_head_pin(tmp_path: Path):
    _, governed, *_ = _transition_run(tmp_path / "source")
    defs = [(i, o, FakeGovernanceRecoverySource(i, o, governed)) for i, o in RECOVERY_SOURCES]
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_SNAPSHOT_RECOVERY_CHAIN_HEAD_PIN_MISMATCH"):
        gov.recover_governance(
            governance_id=GOVERNANCE_ID,
            expected_epoch=1,
            expected_chain_head_sha256="0" * 64,
            output_dir=tmp_path / "bad-pin",
            recovery_sources=defs,
            recovery_quorum=2,
            now=NOW,
        )


def test_run154_governance_recovery_rejects_writer_credential_reuse(tmp_path: Path):
    _, governed, *_ = _transition_run(tmp_path / "source")
    state = json.loads((governed / "trusted-governance-state.json").read_text())
    defs = [(i, o, FakeGovernanceRecoverySource(i, o, governed)) for i, o in RECOVERY_SOURCES]
    defs[0][2].governance_writer_reused = True
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_SNAPSHOT_RECOVERY_SOURCE_AUTHORITY_INVALID"):
        gov.recover_governance(
            governance_id=GOVERNANCE_ID,
            expected_epoch=1,
            expected_chain_head_sha256=state["chainHeadSha256"],
            output_dir=tmp_path / "bad-creds",
            recovery_sources=defs,
            recovery_quorum=2,
            now=NOW,
        )


def test_run154_offline_bundle_recomputes_previous_governance_state_hash(tmp_path: Path):
    _, out, *_ = _transition_run(tmp_path)
    bundle = json.loads((out / "release-governance-bundle.json").read_text())
    entry = bundle["entries"][0]
    entry["proposal"]["previousGovernanceStateSha256"] = "0" * 64
    entry["proposalSha256"] = hashlib.sha256(_canonical(entry["proposal"])).hexdigest()
    for approval in entry["approvals"]:
        approval["proposalSha256"] = entry["proposalSha256"]
    bad = tmp_path / "bad-prior-state.json"
    bad.write_bytes(_canonical(bundle))
    with pytest.raises(gov.GovernanceError, match="GOVERNANCE_ENTRY_PREVIOUS_STATE_HASH_MISMATCH"):
        gov.verify_governance_bundle(bundle_path=bad)
