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


trust = _load("run156_trust", SECURITY / "maintain_release_trust.py")
run155 = _load("run156_run155_fixture", HERE / "test_seal_release_governance.py")
NOW = run155.NOW

SNAPSHOT_MEMBERS = [
    {"keyId": "snapshot/key-a", "identity": "snapshot/a", "operator": "snapshot-op-a", "signerProfile": "offline"},
    {"keyId": "snapshot/key-b", "identity": "snapshot/b", "operator": "snapshot-op-b", "signerProfile": "hsm"},
    {"keyId": "snapshot/key-c", "identity": "snapshot/c", "operator": "snapshot-op-c", "signerProfile": "offline"},
]
TIMESTAMP_MEMBERS = [
    {"keyId": "timestamp/key-a", "identity": "timestamp/a", "operator": "timestamp-op-a", "signerProfile": "hsm"},
    {"keyId": "timestamp/key-b", "identity": "timestamp/b", "operator": "timestamp-op-b", "signerProfile": "offline"},
    {"keyId": "timestamp/key-c", "identity": "timestamp/c", "operator": "timestamp-op-c", "signerProfile": "offline"},
]
RECOVERY_MEMBERS = [
    {"keyId": "recovery/key-a", "identity": "recovery/a", "operator": "recovery-op-a", "signerProfile": "hsm"},
    {"keyId": "recovery/key-b", "identity": "recovery/b", "operator": "recovery-op-b", "signerProfile": "offline"},
    {"keyId": "recovery/key-c", "identity": "recovery/c", "operator": "recovery-op-c", "signerProfile": "offline"},
]
RECOVERED_ROOT_MEMBERS = [
    {"keyId": "recovered-root/key-a", "identity": "recovered-root/a", "operator": "recovered-root-op-a"},
    {"keyId": "recovered-root/key-b", "identity": "recovered-root/b", "operator": "recovered-root-op-b"},
    {"keyId": "recovered-root/key-c", "identity": "recovered-root/c", "operator": "recovered-root-op-c"},
]


def _priv(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(("run156:" + label).encode()).digest())


def _pub(priv: Ed25519PrivateKey) -> str:
    raw = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def _sig(priv: Ed25519PrivateKey, raw: bytes) -> str:
    return base64.b64encode(priv.sign(raw)).decode("ascii")


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(trust._canonical(doc))
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base(tmp_path: Path):
    result, sealed, _candidate, root_envelope, root_private, _sigs = run155._seal(tmp_path / "run155")
    assert result["ok"] is True
    bundle = json.loads((sealed / "release-root-bundle.json").read_text())
    return sealed, bundle["bootstrapRootSha256"], root_envelope, root_private


def _delegated_keys(members):
    private = {m["keyId"]: _priv(m["keyId"]) for m in members}
    expiry = _ts(NOW + timedelta(days=30))
    public = {
        m["keyId"]: {
            "keytype": "ed25519", "scheme": "ed25519", "identity": m["identity"], "operator": m["operator"],
            "expires": expiry, "signerProfile": m["signerProfile"], "keyval": {"public": _pub(private[m["keyId"]])},
        }
        for m in members
    }
    return private, public


def _delegation(tmp_path: Path, sealed: Path, pin: str, root_private: dict, *, version: int = 1, issued=None, expires=None, overlap=False, bad_sig=False):
    effective = trust._effective_root_from_run155(sealed, pin, now=NOW)
    members = SNAPSHOT_MEMBERS + TIMESTAMP_MEMBERS
    private, keys = _delegated_keys(members)
    snap_ids = sorted(m["keyId"] for m in SNAPSHOT_MEMBERS)
    ts_ids = sorted(m["keyId"] for m in TIMESTAMP_MEMBERS)
    if overlap:
        ts_ids[0] = snap_ids[0]; ts_ids = sorted(ts_ids)
    signed = {
        "_type": "delegations", "specVersion": "1.0.0", "schemaVersion": 1, "version": version,
        "governanceId": effective["info"]["governance_id"],
        "issuedAt": _ts(issued or (NOW - timedelta(minutes=5))),
        "expires": _ts(expires or (NOW + timedelta(days=20))),
        "run155Root": {"version": effective["version"], "sha256": effective["sha256"], "chainHeadSha256": effective["chainHeadSha256"]},
        "keys": {k: keys[k] for k in sorted(keys)},
        "roles": {"snapshot": {"keyids": snap_ids, "threshold": 2}, "timestamp": {"keyids": ts_ids, "threshold": 2}},
    }
    raw = trust._canonical(signed)
    root_ids = effective["info"]["roles"]["root"]["keyids"][:2]
    sigs = []
    for i, kid in enumerate(root_ids):
        signing = _priv("wrong-root") if bad_sig and i == 0 else root_private[kid]
        sigs.append({"keyid": kid, "sig": _sig(signing, raw)})
    doc = {"signatures": sorted(sigs, key=lambda x: x["keyid"]), "signed": signed}
    return _write(tmp_path / "delegation.json", doc), private


def _metadata(tmp_path: Path, sealed: Path, pin: str, delegation_path: Path, private: dict, *, snap_version=1, ts_version=1, snap_body_mutate=False, timestamp_body_mutate=False, snap_issued=None, snap_expires=None, timestamp_issued=None, timestamp_expires=None, bad_snapshot_sig=False, bad_timestamp_sig=False):
    effective = trust._effective_root_from_run155(sealed, pin, now=NOW)
    delegation = trust._verify_delegation_root(delegation_path, effective, now=NOW)
    snap_body = {
        "run155Root": {"version": effective["version"], "sha256": effective["sha256"], "chainHeadSha256": effective["chainHeadSha256"]},
        "sealedArtifacts": trust._sealed_artifacts(sealed),
    }
    if snap_body_mutate:
        snap_body["sealedArtifacts"]["active-root.json"]["sha256"] = "0" * 64
    snap_signed = {
        "_type": "snapshot", "specVersion": "1.0.0", "schemaVersion": 1, "version": snap_version,
        "governanceId": effective["info"]["governance_id"], "issuedAt": _ts(snap_issued or (NOW - timedelta(minutes=4))),
        "expires": _ts(snap_expires or (NOW + timedelta(hours=12))), "delegationRoot": {"version": delegation["version"], "sha256": delegation["sha256"]},
        "body": snap_body,
    }
    snap_raw = trust._canonical(snap_signed)
    snap_sigs = []
    for i, kid in enumerate(delegation["roles"]["snapshot"]["keyids"][:2]):
        signing = _priv("wrong-snapshot") if bad_snapshot_sig and i == 0 else private[kid]
        snap_sigs.append({"keyid": kid, "sig": _sig(signing, snap_raw)})
    snap_doc = {"signatures": sorted(snap_sigs, key=lambda x: x["keyid"]), "signed": snap_signed}
    snap_path = _write(tmp_path / "snapshot.json", snap_doc)
    snap_sha = _sha(snap_path)
    ts_body = {
        "snapshot": {"version": snap_version, "sha256": snap_sha, "size": snap_path.stat().st_size, "expires": snap_signed["expires"]},
        "run155Root": {"version": effective["version"], "sha256": effective["sha256"]},
    }
    if timestamp_body_mutate:
        ts_body["snapshot"]["sha256"] = "f" * 64
    ts_signed = {
        "_type": "timestamp", "specVersion": "1.0.0", "schemaVersion": 1, "version": ts_version,
        "governanceId": effective["info"]["governance_id"], "issuedAt": _ts(timestamp_issued or (NOW - timedelta(minutes=2))),
        "expires": _ts(timestamp_expires or (NOW + timedelta(hours=1))), "delegationRoot": {"version": delegation["version"], "sha256": delegation["sha256"]},
        "body": ts_body,
    }
    ts_raw = trust._canonical(ts_signed)
    ts_sigs = []
    for i, kid in enumerate(delegation["roles"]["timestamp"]["keyids"][:2]):
        signing = _priv("wrong-timestamp") if bad_timestamp_sig and i == 0 else private[kid]
        ts_sigs.append({"keyid": kid, "sig": _sig(signing, ts_raw)})
    ts_doc = {"signatures": sorted(ts_sigs, key=lambda x: x["keyid"]), "signed": ts_signed}
    return snap_path, _write(tmp_path / "timestamp.json", ts_doc)


def _refresh(tmp_path: Path, *, previous=None, **meta_kwargs):
    sealed, pin, _root_env, root_private = _base(tmp_path / "base")
    delegation, delegated_private = _delegation(tmp_path / "metadata", sealed, pin, root_private, version=1 if previous is None else previous[2])
    snapshot, timestamp = _metadata(tmp_path / "metadata", sealed, pin, delegation, delegated_private, snap_version=1 if previous is None else previous[0], ts_version=1 if previous is None else previous[1], **meta_kwargs)
    out = tmp_path / "out"
    kwargs = {}
    if previous is not None:
        kwargs = {"previous_state_path": previous[3] / "trusted-delegated-metadata-state.json", "previous_bundle_path": previous[3] / "release-delegated-metadata-bundle.json"}
    result = trust.refresh_delegated_metadata(sealed_dir=sealed, bootstrap_root_sha256=pin, delegation_root_path=delegation, snapshot_path=snapshot, timestamp_path=timestamp, output_dir=out, now=NOW, **kwargs)
    return result, out, sealed, pin, delegation, snapshot, timestamp


def _recovery_root(tmp_path: Path, governance_id: str, *, profiles=None, omit_sig=False, overlap_operator=False, duplicate_channels=False, key_expiry_days=730):
    profiles = profiles or [m["signerProfile"] for m in RECOVERY_MEMBERS]
    members = [dict(m, signerProfile=profiles[i], recoveryChannel=("recovery-channel-1" if duplicate_channels else f"recovery-channel-{i+1}")) for i, m in enumerate(RECOVERY_MEMBERS)]
    if overlap_operator:
        members[0]["operator"] = "root-operator-a"
    private = {m["keyId"]: _priv(m["keyId"]) for m in members}
    key_expiry = _ts(NOW + timedelta(days=key_expiry_days))
    keys = {
        m["keyId"]: {
            "keytype": "ed25519", "scheme": "ed25519", "identity": m["identity"], "operator": m["operator"],
            "expires": key_expiry, "signerProfile": m["signerProfile"], "recoveryChannel": m["recoveryChannel"],
            "keyval": {"public": _pub(private[m["keyId"]])},
        }
        for m in members
    }
    signed = {
        "_type": "recovery-root", "specVersion": "1.0.0", "schemaVersion": 1, "version": 1, "governanceId": governance_id,
        "issuedAt": _ts(NOW - timedelta(minutes=5)), "expires": _ts(NOW + timedelta(days=365)),
        "keys": {k: keys[k] for k in sorted(keys)},
        "role": {"keyids": sorted(m["keyId"] for m in members), "threshold": 2}, "minChannels": 2,
    }
    raw = trust._canonical(signed)
    sigs = []
    for m in members[: (1 if omit_sig else 2)]:
        sigs.append({"keyid": m["keyId"], "sig": _sig(private[m["keyId"]], raw)})
    doc = {"signatures": sorted(sigs, key=lambda x: x["keyid"]), "signed": signed}
    path = _write(tmp_path / "recovery-root.json", doc)
    return path, _sha(path), private, members


def _replacement(tmp_path: Path, *, compromised_reused=False, change_governance=False, omit_self=False, recovery_operator_overlap=False):
    roots = RECOVERED_ROOT_MEMBERS.copy()
    if compromised_reused:
        roots[0] = run155.ROOT_MEMBERS[0]
    if recovery_operator_overlap:
        roots[0] = dict(roots[0], operator="recovery-op-a")
    policy_members = run155.NEW_POLICY_AUTHORITIES if change_governance else run155.run154.POLICY_AUTHORITIES
    env, private = run155._root_envelope(
        version=2, root_members=roots, policy_members=policy_members,
        issued=NOW - timedelta(minutes=1), expires=run155.EXPIRES,
        omit_new_signer=roots[1]["keyId"] if omit_self else None,
    )
    return _write(tmp_path / "replacement-root.json", env), env, private


def _recovery_signatures(tmp_path: Path, subject: dict, private: dict, members: list[dict], *, same_channel=False, wrong_sig=False):
    paths = []
    for i, member in enumerate(members[:2]):
        signed = {
            "schemaVersion": 1, "keyId": member["keyId"], "identity": member["identity"], "operator": member["operator"],
            "channel": "recovery-channel-1" if same_channel else member["recoveryChannel"], "signerProfile": member["signerProfile"],
            "decision": "recover", "subjectSha256": hashlib.sha256(trust._canonical(subject)).hexdigest(), "signedAt": _ts(NOW - timedelta(minutes=1)),
        }
        signing = _priv("wrong-recovery") if wrong_sig and i == 0 else private[member["keyId"]]
        paths.append(_write(tmp_path / f"recovery-sig-{i+1}.json", {"signature": _sig(signing, trust._canonical(signed)), "signed": signed}))
    return paths


def _recover(tmp_path: Path, *, same_channel=False, wrong_sig=False, compromised_reused=False, change_governance=False, omit_self=False, profiles=None, wrong_pin=False, recovery_operator_overlap=False, duplicate_channels=False, recovery_key_expiry_days=730, replacement_operator_overlap=False):
    sealed, pin, _env, _root_private = _base(tmp_path / "base")
    effective = trust._effective_root_from_run155(sealed, pin, now=NOW)
    rr_path, rr_pin, rr_private, rr_members = _recovery_root(tmp_path / "recovery", effective["info"]["governance_id"], profiles=profiles, overlap_operator=recovery_operator_overlap, duplicate_channels=duplicate_channels, key_expiry_days=recovery_key_expiry_days)
    repl_path, repl_doc, _ = _replacement(tmp_path / "recovery", compromised_reused=compromised_reused, change_governance=change_governance, omit_self=omit_self, recovery_operator_overlap=replacement_operator_overlap)
    replacement = trust._replacement_root(repl_path, effective, ["root/key-a"], now=NOW) if not (compromised_reused or change_governance or omit_self) else None
    if replacement is not None:
        selected = sorted(trust._verify_recovery_root(rr_path, rr_pin, now=NOW)["role"]["keyids"][:2])
        subject = {"schemaVersion": 1, "predicateType": trust.PREDICATE_TYPE + "/root-recovery", "governanceId": effective["info"]["governance_id"], "incidentId": "incident-2026-09-05", "previousRoot": {"version": effective["version"], "sha256": effective["sha256"], "chainHeadSha256": effective["chainHeadSha256"]}, "replacementRoot": {"version": replacement["version"], "sha256": replacement["sha256"]}, "recoveryRoot": {"version": 1, "sha256": rr_pin}, "compromisedRootKeyIds": ["root/key-a"], "selectedRecoveryKeyIds": selected}
        sigs = _recovery_signatures(tmp_path / "recovery", subject, rr_private, rr_members, same_channel=same_channel, wrong_sig=wrong_sig)
    else:
        sigs = []
    out = tmp_path/"out"
    result = trust.recover_root(sealed_dir=sealed, bootstrap_root_sha256=pin, recovery_root_path=rr_path, expected_recovery_root_sha256=("0" * 64 if wrong_pin else rr_pin), replacement_root_path=repl_path, compromised_root_key_ids=["root/key-a"], incident_id="incident-2026-09-05", recovery_signature_paths=sigs, output_dir=out, now=NOW)
    return result, out, sealed, pin, rr_pin


def test_run156_refresh_accepts_threshold_snapshot_and_timestamp(tmp_path: Path):
    result, out, sealed, pin, *_ = _refresh(tmp_path)
    assert result["sequence"] == 1
    verified = trust.verify_delegated_output(output_dir=out, sealed_dir=sealed, bootstrap_root_sha256=pin, now=NOW)
    assert verified["ok"] is True


def test_run156_delegation_root_requires_real_root_threshold(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path)
    delegation, _ = _delegation(tmp_path, sealed, pin, root_private, bad_sig=True)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATION_ROOT_SIGNATURE_INVALID"):
        trust._verify_delegation_root(
            delegation,
            trust._effective_root_from_run155(sealed, pin, now=NOW),
            now=NOW,
        )


def test_run156_delegated_roles_must_be_disjoint(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path)
    delegation, _ = _delegation(tmp_path, sealed, pin, root_private, overlap=True)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATION_ROLE_KEY_OVERLAP"):
        trust._verify_delegation_root(
            delegation,
            trust._effective_root_from_run155(sealed, pin, now=NOW),
            now=NOW,
        )


def test_run156_delegation_root_expiry_is_freeze_protected(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path)
    delegation, _ = _delegation(
        tmp_path, sealed, pin, root_private, expires=NOW + timedelta(minutes=30)
    )
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATION_ROOT_FREEZE_RISK"):
        trust._verify_delegation_root(
            delegation,
            trust._effective_root_from_run155(sealed, pin, now=NOW),
            now=NOW,
        )


def test_run156_snapshot_signature_is_cryptographically_verified(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path)
    delegation, priv = _delegation(tmp_path, sealed, pin, root_private)
    snap, ts = _metadata(
        tmp_path, sealed, pin, delegation, priv, bad_snapshot_sig=True
    )
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_SNAPSHOT_SIGNATURE_INVALID"):
        trust.refresh_delegated_metadata(
            sealed_dir=sealed,
            bootstrap_root_sha256=pin,
            delegation_root_path=delegation,
            snapshot_path=snap,
            timestamp_path=ts,
            output_dir=tmp_path / "out",
            now=NOW,
        )


def test_run156_snapshot_binds_exact_run155_seal(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path); delegation, priv = _delegation(tmp_path, sealed, pin, root_private)
    snap, ts = _metadata(tmp_path, sealed, pin, delegation, priv, snap_body_mutate=True)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_SNAPSHOT_BODY_MISMATCH"):
        trust.refresh_delegated_metadata(sealed_dir=sealed, bootstrap_root_sha256=pin, delegation_root_path=delegation, snapshot_path=snap, timestamp_path=ts, output_dir=tmp_path / "out", now=NOW)


def test_run156_timestamp_signature_is_cryptographically_verified(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path); delegation, priv = _delegation(tmp_path, sealed, pin, root_private)
    snap, ts = _metadata(tmp_path, sealed, pin, delegation, priv, bad_timestamp_sig=True)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_TIMESTAMP_SIGNATURE_INVALID"):
        trust.refresh_delegated_metadata(sealed_dir=sealed, bootstrap_root_sha256=pin, delegation_root_path=delegation, snapshot_path=snap, timestamp_path=ts, output_dir=tmp_path / "out", now=NOW)


def test_run156_timestamp_rebinds_exact_snapshot(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path); delegation, priv = _delegation(tmp_path, sealed, pin, root_private)
    snap, ts = _metadata(tmp_path, sealed, pin, delegation, priv, timestamp_body_mutate=True)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_TIMESTAMP_BODY_MISMATCH"):
        trust.refresh_delegated_metadata(sealed_dir=sealed, bootstrap_root_sha256=pin, delegation_root_path=delegation, snapshot_path=snap, timestamp_path=ts, output_dir=tmp_path / "out", now=NOW)


def test_run156_timestamp_short_expiry_is_freeze_protected(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path)
    delegation, priv = _delegation(tmp_path, sealed, pin, root_private)
    snap, ts = _metadata(
        tmp_path,
        sealed,
        pin,
        delegation,
        priv,
        timestamp_expires=NOW + timedelta(minutes=5),
    )
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_TIMESTAMP_FREEZE_RISK"):
        trust.refresh_delegated_metadata(
            sealed_dir=sealed,
            bootstrap_root_sha256=pin,
            delegation_root_path=delegation,
            snapshot_path=snap,
            timestamp_path=ts,
            output_dir=tmp_path / "out",
            now=NOW,
        )


def test_run156_previous_state_rejects_metadata_version_skip(tmp_path: Path):
    _, out, *_ = _refresh(tmp_path / "first")
    previous = (3, 3, 1, out)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_METADATA_VERSION_ROLLBACK_OR_SKIP"):
        _refresh(tmp_path/"second", previous=previous)


def test_run156_chain_rejects_previous_bundle_mutation(tmp_path: Path):
    _, out, *_ = _refresh(tmp_path)
    doc = json.loads((out / "release-delegated-metadata-bundle.json").read_text())
    doc["entries"][0]["chainHeadSha256"] = "0" * 64
    _write(out / "release-delegated-metadata-bundle.json", doc)
    with pytest.raises(
        trust.DelegatedTrustError,
        match="DELEGATED_PREVIOUS_BUNDLE_HASH_MISMATCH|DELEGATED_PREVIOUS_CHAIN_HASH_INVALID",
    ):
        trust._previous_state(
            out / "trusted-delegated-metadata-state.json",
            out / "release-delegated-metadata-bundle.json",
        )


def test_run156_refresh_is_deterministic_for_identical_inputs(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path / "base"); delegation, priv = _delegation(tmp_path / "inputs", sealed, pin, root_private); snap, ts = _metadata(tmp_path / "inputs", sealed, pin, delegation, priv)
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name; trust.refresh_delegated_metadata(sealed_dir=sealed, bootstrap_root_sha256=pin, delegation_root_path=delegation, snapshot_path=snap, timestamp_path=ts, output_dir=out, now=NOW); outs.append(out)
    for filename in sorted(p.name for p in outs[0].iterdir()):
        assert (outs[0]/filename).read_bytes() == (outs[1]/filename).read_bytes()


def test_run156_verify_detects_receipt_mutation(tmp_path: Path):
    _, out, sealed, pin, *_ = _refresh(tmp_path)
    doc = json.loads((out / "release-delegated-metadata-receipt.json").read_text())
    doc["sequence"] = 99
    _write(out / "release-delegated-metadata-receipt.json", doc)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_VERIFY_RECEIPT_REBIND_FAILED"):
        trust.verify_delegated_output(output_dir=out, sealed_dir=sealed, bootstrap_root_sha256=pin, now=NOW)


def test_run156_duplicate_json_keys_are_rejected(tmp_path: Path):
    p = tmp_path/"bad.json"; p.write_text('{"a":1,"a":2}\n')
    with pytest.raises(trust.DelegatedTrustError, match="DUPLICATE_KEY"):
        trust._read(p, "TEST")


def test_run156_recovery_root_is_independently_hash_pinned(tmp_path: Path):
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_ROOT_PIN_MISMATCH"):
        _recover(tmp_path, wrong_pin=True)


def test_run156_recovery_root_requires_self_threshold(tmp_path: Path):
    sealed, pin, _, _ = _base(tmp_path); eff = trust._effective_root_from_run155(sealed, pin, now=NOW); rr, rrpin, _, _ = _recovery_root(tmp_path, eff["info"]["governance_id"], omit_sig=True)
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_ROOT_THRESHOLD_NOT_MET"):
        trust._verify_recovery_root(rr, rrpin, now=NOW)


def test_run156_recovery_requires_hardware_profile_in_recovery_authority(tmp_path: Path):
    profiles = ["offline", "offline", "offline"]
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_ROOT_HARDWARE_PROFILE_REQUIRED"):
        _recover(tmp_path, profiles=profiles)


def test_run156_recovery_rejects_invalid_signature(tmp_path: Path):
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_SIGNATURE_CRYPTO_INVALID"):
        _recover(tmp_path, wrong_sig=True)


def test_run156_recovery_requires_distinct_channels(tmp_path: Path):
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_SIGNATURE_CHANNEL_MISMATCH"):
        _recover(tmp_path, same_channel=True)


def test_run156_recovery_rejects_compromised_key_reuse(tmp_path: Path):
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_COMPROMISED_KEY_REUSED"):
        _recover(tmp_path, compromised_reused=True)


def test_run156_recovery_cannot_silently_change_governance_role(tmp_path: Path):
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_REPLACEMENT_POLICY_ROLE_CHANGED"):
        _recover(tmp_path, change_governance=True)


def test_run156_replacement_root_requires_new_root_self_threshold(tmp_path: Path):
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_REPLACEMENT_NEW_ROOT_THRESHOLD_NOT_MET"):
        _recover(tmp_path, omit_self=True)


def test_run156_valid_multi_channel_recovery_and_offline_verification(tmp_path: Path):
    result, out, sealed, pin, rrpin = _recover(tmp_path)
    assert result["channels"] == 2
    verified = trust.verify_recovery_output(output_dir=out, sealed_dir=sealed, bootstrap_root_sha256=pin, expected_recovery_root_sha256=rrpin, now=NOW)
    assert verified["ok"] is True and verified["replacement_root_version"] == 2


def test_run156_recovery_output_is_deterministic(tmp_path: Path):
    # Independently identical fixtures yield the same canonical trust artifacts.
    r1, o1, *_ = _recover(tmp_path / "a")
    r2, o2, *_ = _recover(tmp_path / "b")
    assert r1["replacement_root_sha256"] == r2["replacement_root_sha256"]
    for f in (
        "release-root-recovery-record.json",
        "recovered-effective-root.json",
        "release-root-recovery-receipt.json",
    ):
        assert (o1 / f).read_bytes() == (o2 / f).read_bytes()


def test_run156_offline_recovery_verifier_detects_signature_mutation(tmp_path: Path):
    _, out, sealed, pin, rrpin = _recover(tmp_path)
    doc = json.loads((out / "release-root-recovery-record.json").read_text())
    doc["signatures"][0]["signature"] = "A" * 88
    _write(out / "release-root-recovery-record.json", doc)
    with pytest.raises(trust.DelegatedTrustError):
        trust.verify_recovery_output(
            output_dir=out,
            sealed_dir=sealed,
            bootstrap_root_sha256=pin,
            expected_recovery_root_sha256=rrpin,
            now=NOW,
        )


def test_run156_outputs_contain_no_private_key_material(tmp_path: Path):
    _, out, *_ = _recover(tmp_path)
    text = "\n".join(p.read_text() for p in out.iterdir())
    for needle in ("privateKey", "private key", "seed", "hsmHandle", "secretKey"):
        assert needle.lower() not in text.lower()


def test_run156_same_delegation_version_cannot_change_bytes(tmp_path: Path):
    _, previous_out, *_ = _refresh(tmp_path / "first")
    sealed, pin, _, root_private = _base(tmp_path / "second-base")
    delegation, private = _delegation(tmp_path / "second-input", sealed, pin, root_private, version=1, expires=NOW + timedelta(days=21))
    snapshot, timestamp = _metadata(tmp_path / "second-input", sealed, pin, delegation, private, snap_version=2, ts_version=2)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_SAME_VERSION_DELEGATION_CHANGED"):
        trust.refresh_delegated_metadata(
            sealed_dir=sealed, bootstrap_root_sha256=pin, delegation_root_path=delegation, snapshot_path=snapshot, timestamp_path=timestamp,
            previous_state_path=previous_out/"trusted-delegated-metadata-state.json", previous_bundle_path=previous_out/"release-delegated-metadata-bundle.json",
            output_dir=tmp_path/"out", now=NOW,
        )


def test_run156_previous_root_history_fork_is_rejected(tmp_path: Path):
    _, previous_out, *_ = _refresh(tmp_path / "first")
    bundle = json.loads((previous_out / "release-delegated-metadata-bundle.json").read_text())
    state = json.loads((previous_out / "trusted-delegated-metadata-state.json").read_text())
    entry = bundle["entries"][0]
    entry["effectiveRoot"]["sha256"] = "f" * 64
    body = {k: v for k, v in entry.items() if k != "chainHeadSha256"}
    head = hashlib.sha256((("0" * 64) + "\n").encode() + trust._canonical(body)).hexdigest()
    entry["chainHeadSha256"] = head
    _write(previous_out / "release-delegated-metadata-bundle.json", bundle)
    state["effectiveRoot"] = entry["effectiveRoot"]
    state["chainHeadSha256"] = head
    state["bundleSha256"] = _sha(previous_out / "release-delegated-metadata-bundle.json")
    _write(previous_out / "trusted-delegated-metadata-state.json", state)
    sealed, pin, _, root_private = _base(tmp_path / "second-base")
    delegation, private = _delegation(tmp_path / "second-input", sealed, pin, root_private, version=1)
    snapshot, timestamp = _metadata(
        tmp_path / "second-input", sealed, pin, delegation, private, snap_version=2, ts_version=2
    )
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_ROOT_HISTORY_FORK"):
        trust.refresh_delegated_metadata(
            sealed_dir=sealed,
            bootstrap_root_sha256=pin,
            delegation_root_path=delegation,
            snapshot_path=snapshot,
            timestamp_path=timestamp,
            previous_state_path=previous_out / "trusted-delegated-metadata-state.json",
            previous_bundle_path=previous_out / "release-delegated-metadata-bundle.json",
            output_dir=tmp_path / "out",
            now=NOW,
        )


def test_run156_timestamp_may_not_precede_snapshot(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path); delegation, private = _delegation(tmp_path, sealed, pin, root_private); snapshot, timestamp = _metadata(tmp_path, sealed, pin, delegation, private, timestamp_issued=NOW - timedelta(minutes=6))
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_TIMESTAMP_PRECEDES_SNAPSHOT"):
        trust.refresh_delegated_metadata(sealed_dir=sealed, bootstrap_root_sha256=pin, delegation_root_path=delegation, snapshot_path=snapshot, timestamp_path=timestamp, output_dir=tmp_path / "out", now=NOW)


def test_run156_timestamp_expiry_may_not_extend_beyond_snapshot(tmp_path: Path):
    sealed, pin, _, root_private = _base(tmp_path); delegation, private = _delegation(tmp_path, sealed, pin, root_private); snapshot, timestamp = _metadata(tmp_path, sealed, pin, delegation, private, snap_expires=NOW + timedelta(hours=1), timestamp_expires=NOW + timedelta(hours=2))
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_TIMESTAMP_EXPIRES_AFTER_SNAPSHOT"):
        trust.refresh_delegated_metadata(sealed_dir=sealed, bootstrap_root_sha256=pin, delegation_root_path=delegation, snapshot_path=snapshot, timestamp_path=timestamp, output_dir=tmp_path / "out", now=NOW)


def test_run156_historical_metadata_verification_survives_later_expiry(tmp_path: Path):
    _, out, sealed, pin, *_ = _refresh(tmp_path)
    assert trust.verify_delegated_output(
        output_dir=out,
        sealed_dir=sealed,
        bootstrap_root_sha256=pin,
        now=NOW + timedelta(days=2),
        require_fresh=False,
    )["ok"] is True


def test_run156_live_metadata_verification_rejects_expired_timestamp(tmp_path: Path):
    _, out, sealed, pin, *_ = _refresh(tmp_path)
    with pytest.raises(trust.DelegatedTrustError, match="DELEGATED_TIMESTAMP_FREEZE_RISK"):
        trust.verify_delegated_output(
            output_dir=out,
            sealed_dir=sealed,
            bootstrap_root_sha256=pin,
            now=NOW + timedelta(hours=2),
            require_fresh=True,
        )


def test_run156_recovery_root_channels_are_pinned_and_diverse(tmp_path: Path):
    sealed, pin, _, _ = _base(tmp_path)
    eff = trust._effective_root_from_run155(sealed, pin, now=NOW)
    rr, rrpin, _, _ = _recovery_root(
        tmp_path, eff["info"]["governance_id"], duplicate_channels=True
    )
    with pytest.raises(
        trust.DelegatedTrustError, match="RECOVERY_ROOT_CHANNEL_DIVERSITY_INVALID"
    ):
        trust._verify_recovery_root(rr, rrpin, now=NOW)


def test_run156_recovery_root_keys_must_cover_recovery_root_lifetime(tmp_path: Path):
    sealed, pin, _, _ = _base(tmp_path); eff = trust._effective_root_from_run155(sealed, pin, now=NOW); rr, rrpin, _, _ = _recovery_root(tmp_path, eff["info"]["governance_id"], key_expiry_days=30)
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_ROOT_KEY_EXPIRES_BEFORE_ROOT"):
        trust._verify_recovery_root(rr, rrpin, now=NOW)


def test_run156_recovery_authority_operator_must_be_independent(tmp_path: Path):
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_AUTHORITY_OPERATOR_OVERLAP"):
        _recover(tmp_path, recovery_operator_overlap=True)


def test_run156_replacement_root_cannot_be_operated_by_recovery_council(tmp_path: Path):
    with pytest.raises(trust.DelegatedTrustError, match="RECOVERY_REPLACEMENT_OPERATOR_OVERLAP"):
        _recover(tmp_path, replacement_operator_overlap=True)


def test_run156_historical_recovery_verification_survives_later_expiry(tmp_path: Path):
    _, out, sealed, pin, rrpin = _recover(tmp_path)
    assert trust.verify_recovery_output(output_dir=out, sealed_dir=sealed, bootstrap_root_sha256=pin, expected_recovery_root_sha256=rrpin, now=NOW + timedelta(days=800), historical=True)["ok"] is True


def test_run156_documentation_describes_delegation_and_recovery():
    guide = (SECURITY / "RELEASE_DELEGATED_TRUST_GUIDE.md").read_text()
    gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    for needle in (
        "Run 156", "snapshot", "timestamp", "recovery root", "multi-channel",
        "Ed25519", "rollback", "freeze",
    ):
        assert needle.lower() in guide.lower()
    assert "Run 156" in gates and "maintain_release_trust.py" in gates
