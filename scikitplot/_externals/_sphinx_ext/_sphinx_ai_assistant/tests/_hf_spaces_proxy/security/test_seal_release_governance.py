from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SECURITY = ROOT / "_hf_spaces_proxy" / "security"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


seal = _load("run155_root_seal", SECURITY / "seal_release_governance.py")
run154 = _load("run155_run154_fixture", HERE / "test_govern_release_history.py")
NOW = datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc)
ISSUED = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)
EXPIRES = datetime(2027, 9, 5, 8, 0, tzinfo=timezone.utc)

ROOT_MEMBERS = [
    {"keyId": "root/key-a", "identity": "root/offline-a", "operator": "root-operator-a"},
    {"keyId": "root/key-b", "identity": "root/offline-b", "operator": "root-operator-b"},
    {"keyId": "root/key-c", "identity": "root/offline-c", "operator": "root-operator-c"},
]
NEW_ROOT_MEMBERS = [
    {"keyId": "root2/key-a", "identity": "root2/offline-a", "operator": "root2-operator-a"},
    {"keyId": "root2/key-b", "identity": "root2/offline-b", "operator": "root2-operator-b"},
    {"keyId": "root2/key-c", "identity": "root2/offline-c", "operator": "root2-operator-c"},
]
NEW_POLICY_AUTHORITIES = [
    {"identity": "governance2/policy-a", "operator": "governance2-operator-a", "keyId": "governance2/key-a"},
    {"identity": "governance2/policy-b", "operator": "governance2-operator-b", "keyId": "governance2/key-b"},
    {"identity": "governance2/policy-c", "operator": "governance2-operator-c", "keyId": "governance2/key-c"},
]


def _priv(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(("run155:" + label).encode()).digest())


def _public_b64(private: Ed25519PrivateKey) -> str:
    raw = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def _sig_b64(private: Ed25519PrivateKey, raw: bytes) -> str:
    return base64.b64encode(private.sign(raw)).decode("ascii")


def _ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical(value: dict) -> bytes:
    return seal._canonical_bytes(value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_members(policy_members=None, emergency_members=None, root_members=None):
    return (root_members or ROOT_MEMBERS) + (policy_members or run154.POLICY_AUTHORITIES) + (emergency_members or run154.EMERGENCY_AUTHORITIES)


def _keys(policy_members=None, emergency_members=None, root_members=None, *, key_expiry=EXPIRES):
    members = _all_members(policy_members, emergency_members, root_members)
    private = {m["keyId"]: _priv(m["keyId"]) for m in members}
    public = {
        m["keyId"]: {
            "keytype": "ed25519",
            "scheme": "ed25519",
            "identity": m["identity"],
            "operator": m["operator"],
            "expires": _ts(key_expiry),
            "keyval": {"public": _public_b64(private[m["keyId"]])},
        }
        for m in members
    }
    return private, public


def _root_envelope(
    *, version=1, policy_members=None, emergency_members=None, root_members=None,
    issued=ISSUED, expires=EXPIRES, old_private=None, old_root_members=None,
    omit_new_signer: str | None = None, omit_old_signer: str | None = None,
):
    root_members = root_members or ROOT_MEMBERS
    policy_members = policy_members or run154.POLICY_AUTHORITIES
    emergency_members = emergency_members or run154.EMERGENCY_AUTHORITIES
    private, keys = _keys(policy_members, emergency_members, root_members, key_expiry=expires)
    signed = {
        "_type": "root",
        "specVersion": "1.0.0",
        "version": version,
        "governanceId": run154.GOVERNANCE_ID,
        "issuedAt": _ts(issued),
        "expires": _ts(expires),
        "keys": {k: keys[k] for k in sorted(keys)},
        "roles": {
            "root": {"keyids": sorted(m["keyId"] for m in root_members), "threshold": 2},
            "governance": {"keyids": sorted(m["keyId"] for m in policy_members), "threshold": 2},
            "emergency": {"keyids": sorted(m["keyId"] for m in emergency_members), "threshold": 2},
        },
    }
    signed_raw = _canonical(signed)
    sigs = []
    # New-root signatures prove self-authorization. Two are sufficient.
    for member in root_members[:2]:
        if member["keyId"] != omit_new_signer:
            sigs.append({"keyid": member["keyId"], "sig": _sig_b64(private[member["keyId"]], signed_raw)})
    # Rotation envelopes additionally contain old-root signatures over the exact new root.
    if old_private is not None and old_root_members is not None:
        for member in old_root_members[:2]:
            if member["keyId"] != omit_old_signer:
                sigs.append({"keyid": member["keyId"], "sig": _sig_b64(old_private[member["keyId"]], signed_raw)})
    envelope = {"signatures": sorted(sigs, key=lambda x: x["keyid"]), "signed": signed}
    return envelope, private


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(doc))
    return path


def _candidate(tmp_path: Path, *, next_policy=None, reason="membership-change", selected=None, revocations=None, approval_members=None, role="policy") -> Path:
    result, out, *_ = run154._transition_run(
        tmp_path / "run154",
        next_policy=next_policy,
        reason=reason,
        selected=selected,
        revocations=revocations,
        approval_members=approval_members,
        role=role,
    )
    assert result["ok"] is True
    return out


def _bootstrap(tmp_path: Path, envelope: dict) -> tuple[Path, str]:
    path = _write(tmp_path / "root-v1.json", envelope)
    return path, _sha(path)


def _authorization_files(tmp_path: Path, candidate: Path, root_envelope: dict, private: dict[str, Ed25519PrivateKey], *, signed_at=NOW, tamper_key: str | None = None):
    candidate_info = seal._validate_run154_dir(candidate)
    root_info = seal._validate_root_envelope(root_envelope)
    subject = seal._subject(candidate_info, root_info)
    docs = []
    selected = subject["selectedKeyIds"]
    previous_role = "emergencyAuthority" if subject["role"] == "emergency" else "policyAuthority"
    members = {m["keyId"]: m for m in candidate_info["previous_policy"][previous_role]["members"]}
    for index, key_id in enumerate(selected, start=1):
        member = members[key_id]
        signed = {
            "schemaVersion": 1,
            "keyId": key_id,
            "identity": member["identity"],
            "operator": member["operator"],
            "decision": "approve",
            "rootVersion": root_info["version"],
            "subjectSha256": hashlib.sha256(_canonical(subject)).hexdigest(),
            "signedAt": _ts(signed_at),
        }
        signing_key = private[tamper_key] if tamper_key is not None and index == 1 else private[key_id]
        doc = {"signature": _sig_b64(signing_key, _canonical(signed)), "signed": signed}
        docs.append(_write(tmp_path / f"crypto-approval-{index}.json", doc))
    return docs


def _seal(tmp_path: Path, *, candidate=None, envelope=None, private=None, next_root=None, previous=None, signed_at=NOW):
    candidate = candidate or _candidate(tmp_path / "candidate")
    if envelope is None:
        envelope, private = _root_envelope()
    assert private is not None
    signatures = _authorization_files(tmp_path / "sigs", candidate, envelope, private, signed_at=signed_at)
    out = tmp_path / "sealed"
    kwargs = dict(governance_dir=candidate, authorization_signature_paths=signatures, output_dir=out, now=NOW)
    if previous is None:
        root_path, pin = _bootstrap(tmp_path / "bootstrap", envelope)
        kwargs.update(bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin)
    else:
        previous_bundle_doc = json.loads((previous / "release-root-bundle.json").read_text())
        kwargs.update(previous_root_state=previous / "trusted-release-root-state.json", previous_root_bundle=previous / "release-root-bundle.json", expected_bootstrap_root_sha256=previous_bundle_doc["bootstrapRootSha256"])
    if next_root is not None:
        next_path = _write(tmp_path / "next-root.json", next_root)
        kwargs["next_root_path"] = next_path
    result = seal.seal_governance(**kwargs)
    return result, out, candidate, envelope, private, signatures


def test_run155_bootstrap_seals_run154_with_real_threshold_signatures(tmp_path: Path):
    result, out, *_ = _seal(tmp_path)
    assert result["ok"] is True
    assert result["authorizing_root_version"] == 1
    assert result["active_root_version"] == 1
    auth = json.loads((out / "cryptographic-governance-authorization.json").read_text())
    assert auth["status"] == "cryptographically-authorized"
    assert auth["selectedKeyIds"] == ["governance/key-a", "governance/key-b"]
    assert len(auth["signatures"]) == 2


def test_run155_exact_bootstrap_hash_pin_is_mandatory(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope()
    root_path, _ = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    with pytest.raises(seal.RootTrustError, match="ROOT_BOOTSTRAP_PIN_MISMATCH"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256="0" * 64, now=NOW)


def test_run155_bootstrap_root_requires_real_self_signature_threshold(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope(omit_new_signer="root/key-b")
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    with pytest.raises(seal.RootTrustError, match="ROOT_BOOTSTRAP_SELF_SIGNATURE_THRESHOLD_NOT_MET"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_candidate_signature_is_cryptographically_verified_not_boolean_trusted(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope()
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private, tamper_key="governance/key-c")
    with pytest.raises(seal.RootTrustError, match="ROOT_AUTH_SIGNATURE_INVALID"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_signature_set_must_equal_run154_selected_keys(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope()
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    sigs.pop()
    with pytest.raises(seal.RootTrustError, match="ROOT_AUTH_SIGNATURE_COUNT_INVALID"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_signature_subject_binds_exact_candidate_snapshot(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope()
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    # Mutating any Run 154 sidecar causes candidate validation/rebinding failure before sealing.
    archive = next((candidate / "archive-results").iterdir())
    changed = json.loads(archive.read_text())
    changed["verifiedAt"] = "2026-09-05T09:00:01Z"
    archive.write_bytes(_canonical(changed))
    with pytest.raises(seal.RootTrustError):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_authorization_rejects_stale_signature(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope()
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private, signed_at=NOW - timedelta(hours=169))
    with pytest.raises(seal.RootTrustError, match="ROOT_AUTH_SIGNATURE_STALE"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_authorization_rejects_future_signature(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope()
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private, signed_at=NOW + timedelta(minutes=11))
    with pytest.raises(seal.RootTrustError, match="ROOT_AUTH_SIGNATURE_FROM_FUTURE"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_expired_root_is_freeze_protected(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope(expires=NOW + timedelta(minutes=30))
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    with pytest.raises(seal.RootTrustError, match="ROOT_EXPIRED_OR_FREEZE_RISK"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_key_expiry_must_cover_root_lifetime(tmp_path: Path):
    envelope, _ = _root_envelope()
    envelope["signed"]["keys"]["root/key-a"]["expires"] = _ts(NOW + timedelta(days=1))
    with pytest.raises(seal.RootTrustError, match="ROOT_KEY_EXPIRES_BEFORE_ROOT"):
        seal._validate_root_envelope(envelope)


def test_run155_root_role_is_offline_and_disjoint(tmp_path: Path):
    envelope, _ = _root_envelope()
    envelope["signed"]["roles"]["root"]["keyids"] = ["governance/key-a", "root/key-a", "root/key-b", "root/key-c"]
    with pytest.raises(seal.RootTrustError, match="ROOT_ROLE_NOT_DISJOINT"):
        seal._validate_root_envelope(envelope)


def test_run155_root_role_must_match_prior_governance_authority(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    wrong = [dict(x) for x in run154.POLICY_AUTHORITIES]
    wrong[2] = {"identity": "wrong/policy-c", "operator": "wrong-operator-c", "keyId": "wrong/key-c"}
    envelope, private = _root_envelope(policy_members=wrong)
    root_path, pin = _bootstrap(tmp_path, envelope)
    # Candidate selected keys a/b are present, but complete authority parity still fails.
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    with pytest.raises(seal.RootTrustError, match="ROOT_AUTHORIZING_ROLE_KEY_SET_MISMATCH"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_authority_change_requires_root_rotation(tmp_path: Path):
    next_policy = run154._base_policy(version=2, policy_members=NEW_POLICY_AUTHORITIES)
    candidate = _candidate(tmp_path / "candidate", next_policy=next_policy)
    envelope, private = _root_envelope()
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    with pytest.raises(seal.RootTrustError, match="ROOT_ROTATION_REQUIRED_FOR_AUTHORITY_CHANGE"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_rotation_requires_old_and_new_root_thresholds(tmp_path: Path):
    next_policy = run154._base_policy(version=2, policy_members=NEW_POLICY_AUTHORITIES)
    candidate = _candidate(tmp_path / "candidate", next_policy=next_policy)
    old_envelope, old_private = _root_envelope()
    new_envelope, _ = _root_envelope(version=2, policy_members=NEW_POLICY_AUTHORITIES, root_members=NEW_ROOT_MEMBERS, issued=NOW, expires=EXPIRES + timedelta(days=30), old_private=old_private, old_root_members=ROOT_MEMBERS, omit_old_signer="root/key-b")
    root_path, pin = _bootstrap(tmp_path / "bootstrap", old_envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, old_envelope, old_private)
    next_path = _write(tmp_path / "root-v2.json", new_envelope)
    with pytest.raises(seal.RootTrustError, match="ROOT_ROTATION_OLD_ROOT_THRESHOLD_NOT_MET"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, next_root_path=next_path, now=NOW)


def test_run155_rotation_requires_new_root_self_threshold(tmp_path: Path):
    next_policy = run154._base_policy(version=2, policy_members=NEW_POLICY_AUTHORITIES)
    candidate = _candidate(tmp_path / "candidate", next_policy=next_policy)
    old_envelope, old_private = _root_envelope()
    new_envelope, _ = _root_envelope(version=2, policy_members=NEW_POLICY_AUTHORITIES, root_members=NEW_ROOT_MEMBERS, issued=NOW, expires=EXPIRES + timedelta(days=30), old_private=old_private, old_root_members=ROOT_MEMBERS, omit_new_signer="root2/key-b")
    root_path, pin = _bootstrap(tmp_path / "bootstrap", old_envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, old_envelope, old_private)
    next_path = _write(tmp_path / "root-v2.json", new_envelope)
    with pytest.raises(seal.RootTrustError, match="ROOT_ROTATION_NEW_ROOT_THRESHOLD_NOT_MET"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, next_root_path=next_path, now=NOW)


def test_run155_rotation_rejects_version_skip(tmp_path: Path):
    next_policy = run154._base_policy(version=2, policy_members=NEW_POLICY_AUTHORITIES)
    candidate = _candidate(tmp_path / "candidate", next_policy=next_policy)
    old_envelope, old_private = _root_envelope()
    new_envelope, _ = _root_envelope(version=3, policy_members=NEW_POLICY_AUTHORITIES, root_members=NEW_ROOT_MEMBERS, issued=NOW, expires=EXPIRES + timedelta(days=30), old_private=old_private, old_root_members=ROOT_MEMBERS)
    root_path, pin = _bootstrap(tmp_path / "bootstrap", old_envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, old_envelope, old_private)
    next_path = _write(tmp_path / "root-v3.json", new_envelope)
    with pytest.raises(seal.RootTrustError, match="ROOT_ROTATION_VERSION_NOT_CONSECUTIVE"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, next_root_path=next_path, now=NOW)


def test_run155_valid_dual_signed_rotation_updates_active_root(tmp_path: Path):
    next_policy = run154._base_policy(version=2, policy_members=NEW_POLICY_AUTHORITIES)
    candidate = _candidate(tmp_path / "candidate", next_policy=next_policy)
    old_envelope, old_private = _root_envelope()
    new_envelope, _ = _root_envelope(version=2, policy_members=NEW_POLICY_AUTHORITIES, root_members=NEW_ROOT_MEMBERS, issued=NOW, expires=EXPIRES + timedelta(days=30), old_private=old_private, old_root_members=ROOT_MEMBERS)
    result, out, *_ = _seal(tmp_path / "seal", candidate=candidate, envelope=old_envelope, private=old_private, next_root=new_envelope)
    assert result["active_root_version"] == 2
    state = json.loads((out / "trusted-release-root-state.json").read_text())
    assert state["rootVersion"] == 2
    pin = json.loads((out / "release-root-bundle.json").read_text())["bootstrapRootSha256"]
    seal.verify_root_bundle(bundle_path=out / "release-root-bundle.json", state_path=out / "trusted-release-root-state.json", expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_next_root_must_match_final_authority_membership(tmp_path: Path):
    next_policy = run154._base_policy(version=2, policy_members=NEW_POLICY_AUTHORITIES)
    candidate = _candidate(tmp_path / "candidate", next_policy=next_policy)
    old_envelope, old_private = _root_envelope()
    # Validly rotated root but still carrying old governance authority.
    new_envelope, _ = _root_envelope(version=2, root_members=NEW_ROOT_MEMBERS, issued=NOW, expires=EXPIRES + timedelta(days=30), old_private=old_private, old_root_members=ROOT_MEMBERS)
    root_path, pin = _bootstrap(tmp_path / "bootstrap", old_envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, old_envelope, old_private)
    next_path = _write(tmp_path / "root-v2.json", new_envelope)
    with pytest.raises(seal.RootTrustError, match="ROOT_FINAL_GOVERNANCE_ROLE_KEY_SET_MISMATCH"):
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out", bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, next_root_path=next_path, now=NOW)


def test_run155_emergency_transition_requires_emergency_crypto_role(tmp_path: Path):
    next_policy = run154._base_policy(version=2, policy_members=[run154.POLICY_AUTHORITIES[1], run154.POLICY_AUTHORITIES[2], {"identity": "governance/policy-d", "operator": "governance-operator-d", "keyId": "governance/key-d"}])
    candidate = _candidate(
        tmp_path / "candidate", next_policy=next_policy, reason="authority-compromise-recovery",
        selected=["emergency/key-a", "emergency/key-b"], revocations=["governance/key-a"],
        approval_members=run154.EMERGENCY_AUTHORITIES[:2], role="emergency",
    )
    envelope, private = _root_envelope()
    new_root_members = NEW_ROOT_MEMBERS
    # New root final policy contains policy b/c/d; build matching public keys.
    final_members = next_policy["policyAuthority"]["members"]
    next_envelope, _ = _root_envelope(version=2, policy_members=final_members, root_members=new_root_members, issued=NOW, expires=EXPIRES + timedelta(days=30), old_private=private, old_root_members=ROOT_MEMBERS)
    result, out, *_ = _seal(tmp_path / "seal", candidate=candidate, envelope=envelope, private=private, next_root=next_envelope)
    auth = json.loads((out / "cryptographic-governance-authorization.json").read_text())
    assert result["ok"] is True
    assert auth["role"] == "emergency"
    assert auth["selectedKeyIds"] == ["emergency/key-a", "emergency/key-b"]


def test_run155_rejects_revoked_key_in_active_root_final_roles(tmp_path: Path):
    next_policy = run154._base_policy(version=2)
    # Normal transition cannot revoke while leaving key in next policy; Run154 itself rejects it.
    with pytest.raises(run154.gov.GovernanceError):
        _candidate(tmp_path / "candidate", next_policy=next_policy, revocations=["governance/key-a"])


def test_run155_root_bundle_offline_verifier_detects_signature_mutation(tmp_path: Path):
    _, out, *_ = _seal(tmp_path)
    bundle = json.loads((out / "release-root-bundle.json").read_text())
    sig = bundle["roots"][0]["signatures"][0]["sig"]
    bundle["roots"][0]["signatures"][0]["sig"] = ("A" if sig[0] != "A" else "B") + sig[1:]
    bad = _write(tmp_path / "bad-bundle.json", bundle)
    with pytest.raises(seal.RootTrustError):
        seal.verify_root_bundle(bundle_path=bad, now=NOW)



def _candidate_after(tmp_path: Path, previous_candidate: Path, history_dir: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    previous_state = json.loads((previous_candidate / "trusted-governance-state.json").read_text())
    next_policy = json.loads(json.dumps(previous_state["policy"]))
    next_policy["policyVersion"] += 1
    proposal = run154._proposal(
        tmp_path, history_dir, previous_candidate, next_policy=next_policy,
        transition_id="governance-2026-09-05-epoch-2", reason="scheduled-change",
    )
    approvals = [run154._approval(tmp_path, proposal, member, role="policy", suffix="-epoch2") for member in run154.POLICY_AUTHORITIES[:2]]
    archives = [(i, o, run154.FakeGovernanceArchive(i, o)) for i, o in run154.GOV_ARCHIVES]
    out = tmp_path / "run154-epoch2"
    result = run154.gov.apply_governance_transition(
        history_dir=history_dir,
        previous_state=previous_candidate / "trusted-governance-state.json",
        previous_bundle=previous_candidate / "release-governance-bundle.json",
        proposal_path=proposal,
        approval_paths=approvals,
        output_dir=out,
        governance_archives=archives,
        now=NOW,
    )
    assert result["ok"] is True
    assert result["epoch"] == 2
    return out


def test_run155_previous_root_state_successfully_advances_epoch_two(tmp_path: Path):
    transition_result = run154._transition_run(tmp_path / "run154-first")
    first_candidate = transition_result[1]
    history_dir = transition_result[2]
    envelope, private = _root_envelope()
    first_result, first_sealed, *_ = _seal(tmp_path / "first-seal", candidate=first_candidate, envelope=envelope, private=private)
    assert first_result["active_root_version"] == 1
    second_candidate = _candidate_after(tmp_path / "run154-second", first_candidate, history_dir)
    signatures = _authorization_files(tmp_path / "second-sigs", second_candidate, envelope, private)
    pin = json.loads((first_sealed / "release-root-bundle.json").read_text())["bootstrapRootSha256"]
    second_out = tmp_path / "second-sealed"
    result = seal.seal_governance(
        governance_dir=second_candidate,
        authorization_signature_paths=signatures,
        output_dir=second_out,
        previous_root_state=first_sealed / "trusted-release-root-state.json",
        previous_root_bundle=first_sealed / "release-root-bundle.json",
        expected_bootstrap_root_sha256=pin,
        now=NOW,
    )
    assert result["ok"] is True
    assert result["epoch"] == 2
    assert result["active_root_version"] == 1
    verified = seal.verify_sealed_governance(sealed_dir=second_out, expected_bootstrap_root_sha256=pin, now=NOW)
    assert verified["epoch"] == 2


def test_run155_seal_is_deterministic_for_identical_authority_bytes(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope()
    root_path, pin = _bootstrap(tmp_path / "bootstrap", envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    outputs = []
    for name in ("out-a", "out-b"):
        out = tmp_path / name
        seal.seal_governance(
            governance_dir=candidate, authorization_signature_paths=sigs, output_dir=out,
            bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW,
        )
        outputs.append(out)
    left = {p.relative_to(outputs[0]): p.read_bytes() for p in outputs[0].rglob("*") if p.is_file()}
    right = {p.relative_to(outputs[1]): p.read_bytes() for p in outputs[1].rglob("*") if p.is_file()}
    assert left == right


def test_run155_offline_seal_rejects_embedded_history_mutation(tmp_path: Path):
    _, out, *_ = _seal(tmp_path)
    pin = json.loads((out / "release-root-bundle.json").read_text())["bootstrapRootSha256"]
    snapshot_path = out / "release-governance-recovery-snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["historyState"]["sequence"] += 1
    snapshot_path.write_bytes(_canonical(snapshot))
    with pytest.raises(seal.RootTrustError):
        seal.verify_sealed_governance(sealed_dir=out, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_previous_root_state_prevents_governance_rollback_or_skip(tmp_path: Path):
    _, first, *_ = _seal(tmp_path / "first")
    candidate = _candidate(tmp_path / "second-candidate")  # another epoch-1 candidate, not epoch 2
    envelope = json.loads((first / "active-root.json").read_text())
    # Private keys are deterministic from key ids, so use the same root private set for test signing.
    private, _ = _keys()
    sigs = _authorization_files(tmp_path / "second-sigs", candidate, envelope, private)
    with pytest.raises(seal.RootTrustError, match="ROOT_GOVERNANCE_SEQUENCE_OR_PREVIOUS_STATE_MISMATCH"):
        pin = json.loads((first / "release-root-bundle.json").read_text())["bootstrapRootSha256"]
        seal.seal_governance(governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "second", previous_root_state=first / "trusted-release-root-state.json", previous_root_bundle=first / "release-root-bundle.json", expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_duplicate_json_keys_are_rejected(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('{"signatures":[],"signed":{},"signed":{}}\n')
    with pytest.raises(seal.RootTrustError, match="DUPLICATE_KEY"):
        seal._read(p, "ROOT_TEST")


def test_run155_output_contains_no_private_key_material(tmp_path: Path):
    _, out, *_ = _seal(tmp_path)
    combined = b"\n".join(p.read_bytes() for p in out.rglob("*") if p.is_file())
    assert b"private" not in combined.lower()
    assert b"seed" not in combined.lower()



def test_run155_offline_complete_seal_verifies_real_authorization_signatures(tmp_path: Path):
    _, out, *_ = _seal(tmp_path)
    pin = json.loads((out / "release-root-bundle.json").read_text())["bootstrapRootSha256"]
    result = seal.verify_sealed_governance(sealed_dir=out, expected_bootstrap_root_sha256=pin, now=NOW)
    assert result["ok"] is True
    assert result["epoch"] == 1


def test_run155_offline_seal_rejects_authorization_signature_mutation(tmp_path: Path):
    _, out, *_ = _seal(tmp_path)
    pin = json.loads((out / "release-root-bundle.json").read_text())["bootstrapRootSha256"]
    auth_path = out / "cryptographic-governance-authorization.json"
    auth = json.loads(auth_path.read_text())
    sig = auth["signatures"][0]["signature"]
    auth["signatures"][0]["signature"] = ("A" if sig[0] != "A" else "B") + sig[1:]
    auth_path.write_bytes(_canonical(auth))
    with pytest.raises(seal.RootTrustError):
        seal.verify_sealed_governance(sealed_dir=out, expected_bootstrap_root_sha256=pin, now=NOW)


def test_run155_offline_verification_requires_out_of_band_bootstrap_pin(tmp_path: Path):
    _, out, *_ = _seal(tmp_path)
    with pytest.raises(seal.RootTrustError, match="ROOT_EXPECTED_BOOTSTRAP_PIN_REQUIRED"):
        seal.verify_root_bundle(bundle_path=out / "release-root-bundle.json", state_path=out / "trusted-release-root-state.json", now=NOW)


def test_run155_root_operators_must_be_disjoint_from_online_authority(tmp_path: Path):
    overlap = [dict(x) for x in ROOT_MEMBERS]
    overlap[0]["operator"] = run154.POLICY_AUTHORITIES[0]["operator"]
    envelope, _ = _root_envelope(root_members=overlap)
    with pytest.raises(seal.RootTrustError, match="ROOT_OPERATOR_NOT_DISJOINT"):
        seal._validate_root_envelope(envelope)


def test_run155_unreferenced_run154_archive_sidecar_is_rejected(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    extra = candidate / "archive-results" / "99-unreferenced.json"
    original = next((candidate / "archive-results").iterdir())
    extra.write_bytes(original.read_bytes())
    with pytest.raises(seal.RootTrustError, match="RUN154_ARCHIVE_EVIDENCE_SET_MISMATCH"):
        seal._validate_run154_dir(candidate)



def test_run155_bootstrap_rejects_non_root_signature_malleability(tmp_path: Path):
    candidate = _candidate(tmp_path / "candidate")
    envelope, private = _root_envelope()
    signed_raw = _canonical(envelope["signed"])
    envelope["signatures"].append({
        "keyid": "governance/key-a",
        "sig": _sig_b64(private["governance/key-a"], signed_raw),
    })
    envelope["signatures"] = sorted(envelope["signatures"], key=lambda x: x["keyid"])
    root_path, pin = _bootstrap(tmp_path, envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, envelope, private)
    with pytest.raises(seal.RootTrustError, match="ROOT_BOOTSTRAP_UNAUTHORIZED_SIGNATURE_KEY"):
        seal.seal_governance(
            governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out",
            bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin, now=NOW,
        )


def test_run155_rotation_rejects_non_root_signature_malleability(tmp_path: Path):
    next_policy = run154._base_policy(version=2, policy_members=NEW_POLICY_AUTHORITIES)
    candidate = _candidate(tmp_path / "candidate", next_policy=next_policy)
    old_envelope, old_private = _root_envelope()
    new_envelope, new_private = _root_envelope(
        version=2, policy_members=NEW_POLICY_AUTHORITIES, root_members=NEW_ROOT_MEMBERS,
        issued=NOW, expires=EXPIRES + timedelta(days=30),
        old_private=old_private, old_root_members=ROOT_MEMBERS,
    )
    signed_raw = _canonical(new_envelope["signed"])
    new_envelope["signatures"].append({
        "keyid": "governance2/key-a",
        "sig": _sig_b64(new_private["governance2/key-a"], signed_raw),
    })
    new_envelope["signatures"] = sorted(new_envelope["signatures"], key=lambda x: x["keyid"])
    root_path, pin = _bootstrap(tmp_path / "bootstrap", old_envelope)
    sigs = _authorization_files(tmp_path / "sigs", candidate, old_envelope, old_private)
    next_path = _write(tmp_path / "root-v2.json", new_envelope)
    with pytest.raises(seal.RootTrustError, match="ROOT_ROTATION_UNAUTHORIZED_SIGNATURE_KEY"):
        seal.seal_governance(
            governance_dir=candidate, authorization_signature_paths=sigs, output_dir=tmp_path / "out",
            bootstrap_root_path=root_path, expected_bootstrap_root_sha256=pin,
            next_root_path=next_path, now=NOW,
        )


def test_run155_live_bundle_rejects_future_dated_current_root(tmp_path: Path):
    issued = NOW + timedelta(minutes=11)
    expires = issued + timedelta(days=365)
    envelope, _ = _root_envelope(issued=issued, expires=expires)
    root_path, pin = _bootstrap(tmp_path, envelope)
    bundle = {
        "schemaVersion": 1,
        "predicateType": seal.PREDICATE_TYPE,
        "status": "trusted-release-root-chain",
        "governanceId": run154.GOVERNANCE_ID,
        "bootstrapRootSha256": pin,
        "roots": [envelope],
    }
    bundle_path = _write(tmp_path / "bundle.json", bundle)
    with pytest.raises(seal.RootTrustError, match="ROOT_CURRENT_ISSUED_FROM_FUTURE"):
        seal.verify_root_bundle(
            bundle_path=bundle_path, expected_bootstrap_root_sha256=pin, now=NOW,
            require_current_fresh=True,
        )

def test_run155_documentation_describes_cryptographic_boundary():
    guide = (SECURITY / "RELEASE_ROOT_GUIDE.md").read_text()
    gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    for needle in ("Ed25519", "old-root", "new-root", "freeze", "private key", "Run 155"):
        assert needle.lower() in guide.lower()
    assert "Run 155" in gates
    assert "seal_release_governance.py" in gates
