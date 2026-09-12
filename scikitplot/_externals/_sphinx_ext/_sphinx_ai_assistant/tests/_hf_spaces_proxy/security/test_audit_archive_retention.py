from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SECURITY = ROOT / "_hf_spaces_proxy" / "security"

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod

health = _load("run161_archive_health", SECURITY / "audit_archive_retention.py")
run160_tests = _load("run160_helpers_for_run161", HERE / "test_archive_native_status_evidence.py")
NOW = run160_tests.NOW

def _canonical(v): return (json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n").encode()
def _pub(priv): return base64.b64encode(priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
def _sig(priv, doc): return base64.b64encode(priv.sign(_canonical(doc))).decode()
def _write(path, doc): path.write_bytes(_canonical(doc)); return path

def _key(priv, identity, operator, expires=None):
    return {"identity": identity, "operator": operator, "expires": health._ts(expires or (NOW + timedelta(days=900))), "publicKey": _pub(priv)}

def _root(tmp_path, *, selected=None, duplicate_operator=False):
    keys = {}
    privs = {}
    for i in range(3):
        kid = f"ret-root-{i+1}"
        priv = Ed25519PrivateKey.generate()
        privs[kid] = priv
        keys[kid] = _key(
            priv,
            f"ret-root-id-{i+1}",
            "ret-root-op-a" if duplicate_operator else f"ret-root-op-{i+1}",
            NOW + timedelta(days=800),
        )
    selected = selected or ["ret-root-1", "ret-root-2"]
    signed = {
        "_type": "archive-retention-root",
        "specVersion": "1.0.0",
        "schemaVersion": 1,
        "rootId": "archive-retention-root/main",
        "version": 1,
        "issuedAt": health._ts(NOW - timedelta(minutes=1)),
        "expires": health._ts(NOW + timedelta(days=700)),
        "threshold": 2,
        "selectedSignerKeyIds": sorted(selected),
        "keys": {k: keys[k] for k in sorted(keys)},
    }
    doc = {
        "signed": signed,
        "signatures": [
            {"keyId": k, "signature": _sig(privs[k], signed)}
            for k in sorted(selected)
        ],
    }
    path = _write(tmp_path / "retention-root.json", doc)
    return doc, path, health._sha_bytes(_canonical(doc)), privs

def _run160(tmp_path):
    setup, _, out, _, _ = run160_tests._preserved(tmp_path / "run160")
    receipt = json.loads(
        (out / "release-native-evidence-archive-receipt.json").read_text()
    )
    payload = (out / "release-native-evidence-archive.json").read_bytes()
    return setup, out, receipt, {
        "name": "release-native-evidence-archive.json",
        "sha256": health._sha_bytes(payload),
        "size": len(payload),
    }

def _member_from_entry(entry, idx, *, provider_priv=None, auditor_priv=None):
    pp = provider_priv or Ed25519PrivateKey.generate()
    ap = auditor_priv or Ed25519PrivateKey.generate()
    member = {
        "archiveIdentity": entry["archiveIdentity"],
        "archiveOperator": entry["archiveOperator"],
        "archiveId": entry["archiveId"],
        "locator": entry["locator"],
        "immutability": entry["immutability"],
        "providerKeyId": f"provider-key-{idx}",
        "providerKey": _key(pp, f"provider-{idx}", entry["archiveOperator"]),
        "auditorIdentity": f"health-auditor-{idx}",
        "auditorKeyId": f"auditor-key-{idx}",
        "auditorKey": _key(
            ap,
            f"auditor-key-identity-{idx}",
            f"health-audit-op-{idx}",
        ),
    }
    return member, pp, ap

def _new_member(idx=3):
    pp = Ed25519PrivateKey.generate()
    ap = Ed25519PrivateKey.generate()
    member = {
        "archiveIdentity": f"archive-new-{idx}",
        "archiveOperator": f"archive-op-new-{idx}",
        "archiveId": f"archive-id-new-{idx}",
        "locator": f"mem+immutable://archive-op-new-{idx}/archive-new-{idx}/native-evidence",
        "immutability": "object-lock",
        "providerKeyId": f"provider-key-new-{idx}",
        "providerKey": _key(pp, f"provider-new-{idx}", f"archive-op-new-{idx}"),
        "auditorIdentity": f"health-auditor-new-{idx}",
        "auditorKeyId": f"auditor-key-new-{idx}",
        "auditorKey": _key(
            ap,
            f"auditor-key-identity-new-{idx}",
            f"health-audit-op-new-{idx}",
        ),
    }
    return member, pp, ap

def _membership(
    tmp_path,
    root_doc,
    root_privs,
    run160_out,
    _receipt,
    members,
    *,
    sequence=1,
    previous_head=None,
    previous_members=None,
    issued=NOW,
    selected=None,
    minimum=2,
    transition_override=None,
    name=None,
):
    payload = (run160_out / "release-native-evidence-archive.json").read_bytes()
    state = json.loads(
        (run160_out / "trusted-native-evidence-archive-state.json").read_text()
    )
    old = {x["archiveId"] for x in (previous_members or [])}
    new = {x["archiveId"] for x in members}
    added = sorted(new - old)
    retired = sorted(old - new)
    kind = "bootstrap" if previous_members is None else (
        "migration" if added or retired else "audit"
    )
    transition = transition_override or {
        "kind": kind,
        "addedArchiveIds": added,
        "retiredArchiveIds": retired,
    }
    selected = selected or ["ret-root-1", "ret-root-2"]
    signed = {
        "_type": "archive-health-membership",
        "specVersion": "1.0.0",
        "schemaVersion": 1,
        "rootSha256": health._sha_bytes(_canonical(root_doc)),
        "sequence": sequence,
        "issuedAt": health._ts(issued),
        "previousHealthChainHeadSha256": previous_head,
        "run160ArchiveArtifact": {
            "name": "release-native-evidence-archive.json",
            "sha256": health._sha_bytes(payload),
            "size": len(payload),
        },
        "nativeStatusChainHeadSha256": state["nativeStatusChainHeadSha256"],
        "minimumDurableCopies": minimum,
        "selectedSignerKeyIds": sorted(selected),
        "members": sorted(members, key=lambda x: x["archiveId"]),
        "transition": transition,
    }
    doc = {
        "signed": signed,
        "signatures": [
            {"keyId": k, "signature": _sig(root_privs[k], signed)}
            for k in sorted(selected)
        ],
    }
    return _write(tmp_path/(name or f"membership-{sequence}.json"), doc), doc

class Provider:
    def __init__(
        self,
        member,
        priv,
        now=NOW,
        *,
        retention_days=365,
        version=None,
        bad_sig=False,
        bad_artifact=False,
        bad_challenge=False,
        legal_hold=False,
        mode=None,
    ):
        self.member = member
        self.priv = priv
        self.now = now
        self.retention_days = retention_days
        self.version = version or f"version/{member['archiveId']}/1"
        self.bad_sig = bad_sig
        self.bad_artifact = bad_artifact
        self.bad_challenge = bad_challenge
        self.legal_hold = legal_hold
        self.mode = mode or {
            "object-lock": "object-lock-compliance",
            "content-addressed": "content-addressed",
            "versioned-create-only": "versioned-create-only",
            "release-create-only": "release-create-only",
        }[member["immutability"]]

    def __call__(self, request):
        artifact = dict(request["artifact"])
        if self.bad_artifact:
            artifact["size"] += 1
        signed = {
            "schemaVersion": 1,
            "operation": "audit-retention",
            "auditId": request["auditId"],
            "archiveId": self.member["archiveId"],
            "archiveIdentity": self.member["archiveIdentity"],
            "locator": self.member["locator"],
            "immutableVersionId": self.version,
            "artifact": artifact,
            "immutability": self.member["immutability"],
            "retentionMode": self.mode,
            "retentionUntil": health._ts(
                self.now + timedelta(days=self.retention_days)
            ),
            "legalHold": self.legal_hold,
            "challenge": "0" * 64 if self.bad_challenge else request["challenge"],
            "remoteReadbackVerified": True,
            "observedAt": health._ts(self.now),
        }
        sig = _sig(self.priv, signed)
        if self.bad_sig:
            sig = base64.b64encode(b"x" * 64).decode()
        return {
            "signed": signed,
            "signature": {"keyId": self.member["providerKeyId"], "signature": sig},
        }

class Auditor:
    def __init__(
        self,
        member,
        priv,
        provider,
        now=NOW,
        *,
        bad_sig=False,
        bad_version=False,
        bad_readback=False,
    ):
        self.member = member
        self.priv = priv
        self.provider = provider
        self.now = now
        self.bad_sig = bad_sig
        self.bad_version = bad_version
        self.bad_readback = bad_readback

    def __call__(self, request):
        signed = {
            "schemaVersion": 1,
            "operation": "verify-retention",
            "auditId": request["auditId"],
            "auditorIdentity": self.member["auditorIdentity"],
            "archiveId": self.member["archiveId"],
            "archiveIdentity": self.member["archiveIdentity"],
            "locator": self.member["locator"],
            "immutableVersionId": (
                "wrong/version"
                if self.bad_version
                else request["immutableVersionId"]
            ),
            "artifact": request["artifact"],
            "challenge": request["challenge"],
            "providerResponseSha256": request["providerResponseSha256"],
            "readOnly": True,
            "remoteReadbackVerified": not self.bad_readback,
            "observedAt": health._ts(self.now),
        }
        sig = _sig(self.priv, signed)
        if self.bad_sig:
            sig = base64.b64encode(b"y" * 64).decode()
        return {
            "signed": signed,
            "signature": {
                "keyId": self.member["auditorKeyId"],
                "signature": sig,
            },
        }

def _setup(tmp_path):
    setup, run160_out, receipt, _ = _run160(tmp_path)
    root_doc, root_path, root_pin, root_privs = _root(tmp_path)
    members = []
    privmap = {}
    for i, entry in enumerate(receipt["archives"], start=1):
        m, pp, ap = _member_from_entry(entry, i)
        members.append(m)
        privmap[m["archiveId"]] = (pp, ap)
    mp, _ = _membership(tmp_path, root_doc, root_privs, run160_out, receipt, members)
    targets = []
    for m in members:
        pp, ap = privmap[m["archiveId"]]
        p = Provider(m, pp)
        targets.append((m["archiveId"], p, Auditor(m, ap, p)))
    return (
        setup,
        run160_out,
        receipt,
        root_doc,
        root_path,
        root_pin,
        root_privs,
        members,
        privmap,
        mp,
        targets,
    )

def _audit(
    tmp_path, *, targets_mutator=None, membership_mutator=None
):
    (
        setup,
        run160_out,
        _,
        root_doc,
        root_path,
        root_pin,
        root_privs,
        members,
        privmap,
        mp,
        targets,
    ) = _setup(tmp_path)
    if membership_mutator:
        doc = json.loads(mp.read_text())
        membership_mutator(doc, members, root_privs)
        _write(mp, doc)
    if targets_mutator:
        targets = targets_mutator(targets, members, privmap)
    out = tmp_path / "health"
    result = health.audit_archive_health(
        run160_dir=run160_out,
        output_dir=out,
        retention_root_path=root_path,
        membership_path=mp,
        targets=targets,
        expected_retention_root_sha256=root_pin,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
    )
    return (
        setup,
        run160_out,
        root_doc,
        root_path,
        root_pin,
        root_privs,
        members,
        privmap,
        out,
        result,
    )

def test_run161_bootstrap_and_offline_verify(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, _, out, result = _audit(
        tmp_path
    )
    assert result["ok"] and result["active_archive_count"] == 2
    verify = health.verify_archive_health(
        run160_dir=run160_out,
        output_dir=out,
        retention_root_path=root_path,
        expected_retention_root_sha256=root_pin,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
        historical=True,
    )
    assert verify["sequence"] == 1

def test_run161_requires_exact_run160_bootstrap_membership(tmp_path):
    setup, run160_out, _, _, root_path, root_pin, root_privs, _, _, mp, targets = _setup(
        tmp_path
    )
    doc = json.loads(mp.read_text())
    doc["signed"]["members"][0]["locator"] = "mem+immutable://other/place"
    doc["signatures"] = [
        {"keyId": k, "signature": _sig(root_privs[k], doc["signed"])}
        for k in doc["signed"]["selectedSignerKeyIds"]
    ]
    _write(mp, doc)
    with pytest.raises(health.ArchiveHealthError, match="BOOTSTRAP_RUN160_MEMBERSHIP_MISMATCH"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp,
            targets=targets,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )

def test_run161_rejects_wrong_root_pin(tmp_path):
    setup, run160_out, _, _, root_path, _, _, _, _, mp, targets = _setup(tmp_path)
    with pytest.raises(health.ArchiveHealthError, match="ROOT_PIN_MISMATCH"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp,
            targets=targets,
            expected_retention_root_sha256="0" * 64,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )

def test_run161_root_threshold_requires_independent_operators(tmp_path):
    _, path, pin, _ = _root(tmp_path, duplicate_operator=True)
    with pytest.raises(health.ArchiveHealthError, match="ROOT_OPERATOR_QUORUM_INVALID"):
        health._verify_root(json.loads(path.read_text()), pin, now=NOW)

def test_run161_root_signature_set_is_exact(tmp_path):
    doc, _, pin, privs = _root(tmp_path)
    signed = doc["signed"]
    doc["signatures"].append(
        {
            "keyId": "ret-root-3",
            "signature": _sig(privs["ret-root-3"], signed),
        }
    )
    with pytest.raises(health.ArchiveHealthError, match="ROOT_SIGNATURE_SET_INVALID"):
        health._verify_root(doc, pin, now=NOW)

def test_run161_membership_signature_mutation_fails(tmp_path):
    def mutate(doc, _members, _privs):
        doc["signatures"][0]["signature"] = base64.b64encode(b"z" * 64).decode()

    with pytest.raises(
        health.ArchiveHealthError, match="MEMBERSHIP_SIGNATURE_INVALID"
    ):
        _audit(tmp_path, membership_mutator=mutate)

def test_run161_provider_signature_mutation_fails(tmp_path):
    def mt(targets, members, privmap):
        out = []
        for aid, p, a in targets:
            if aid == members[0]["archiveId"]:
                p = Provider(members[0], privmap[aid][0], bad_sig=True)
                a = Auditor(members[0], privmap[aid][1], p)
            out.append((aid, p, a))
        return out
    with pytest.raises(
        health.ArchiveHealthError, match="PROVIDER_SIGNATURE_INVALID"
    ):
        _audit(tmp_path, targets_mutator=mt)

def test_run161_provider_artifact_rebinding(tmp_path):
    def mt(targets, members, privmap):
        out = []
        for aid, p, a in targets:
            if aid == members[0]["archiveId"]:
                p = Provider(
                    members[0], privmap[aid][0], bad_artifact=True
                )
                a = Auditor(members[0], privmap[aid][1], p)
            out.append((aid, p, a))
        return out
    with pytest.raises(
        health.ArchiveHealthError, match="PROVIDER_ARTIFACT_BINDING_INVALID"
    ):
        _audit(tmp_path, targets_mutator=mt)

def test_run161_provider_challenge_rebinding(tmp_path):
    def mt(targets, members, privmap):
        out = []
        for aid, p, a in targets:
            if aid == members[0]["archiveId"]:
                p = Provider(members[0], privmap[aid][0], bad_challenge=True)
                a = Auditor(members[0], privmap[aid][1], p)
            out.append((aid, p, a))
        return out
    with pytest.raises(
        health.ArchiveHealthError, match="PROVIDER_PROOF_INVALID"
    ):
        _audit(tmp_path, targets_mutator=mt)

def test_run161_auditor_signature_mutation_fails(tmp_path):
    def mt(targets, members, privmap):
        return [
            (
                aid,
                p,
                Auditor(
                    m,
                    privmap[aid][1],
                    p,
                    bad_sig=(aid == members[0]["archiveId"]),
                ),
            )
            for (aid, p, a), m in zip(targets, members)
        ]

    with pytest.raises(
        health.ArchiveHealthError, match="AUDITOR_SIGNATURE_INVALID"
    ):
        _audit(tmp_path, targets_mutator=mt)

def test_run161_auditor_must_read_same_immutable_version(tmp_path):
    def mt(targets, members, privmap):
        out = []
        for aid, p, a in targets:
            m = next(x for x in members if x["archiveId"] == aid)
            out.append(
                (
                    aid,
                    p,
                    Auditor(
                        m,
                        privmap[aid][1],
                        p,
                        bad_version=(aid == members[0]["archiveId"]),
                    ),
                )
            )
        return out
    with pytest.raises(
        health.ArchiveHealthError, match="AUDITOR_PROOF_BINDING_INVALID"
    ):
        _audit(tmp_path, targets_mutator=mt)

def test_run161_auditor_readback_is_mandatory(tmp_path):
    def mt(targets, members, privmap):
        out = []
        for aid, p, a in targets:
            m = next(x for x in members if x["archiveId"] == aid)
            out.append(
                (
                    aid,
                    p,
                    Auditor(
                        m,
                        privmap[aid][1],
                        p,
                        bad_readback=(aid == members[0]["archiveId"]),
                    ),
                )
            )
        return out

    with pytest.raises(
        health.ArchiveHealthError, match="AUDITOR_PROOF_INVALID"
    ):
        _audit(tmp_path, targets_mutator=mt)

def test_run161_minimum_retention_remaining_is_enforced(tmp_path):
    def mt(targets, members, privmap):
        out = []
        for aid, p, a in targets:
            m = next(x for x in members if x["archiveId"] == aid)
            p = (
                Provider(m, privmap[aid][0], retention_days=2)
                if aid == members[0]["archiveId"]
                else p
            )
            out.append((aid, p, Auditor(m, privmap[aid][1], p)))
        return out

    with pytest.raises(
        health.ArchiveHealthError, match="RETENTION_TOO_SHORT"
    ):
        _audit(tmp_path, targets_mutator=mt)

def test_run161_legal_hold_can_cover_short_retention(tmp_path):
    def mt(targets, members, privmap):
        out = []
        for aid, p, a in targets:
            m = next(x for x in members if x["archiveId"] == aid)
            p = Provider(m, privmap[aid][0], retention_days=0, legal_hold=True)
            out.append((aid, p, Auditor(m, privmap[aid][1], p)))
        return out

    *_, result = _audit(tmp_path, targets_mutator=mt)
    assert result["ok"]

def test_run161_rejects_archive_auditor_operator_overlap(tmp_path):
    setup, run160_out, _, _, root_path, root_pin, root_privs, _, _, mp, targets = _setup(
        tmp_path
    )
    doc = json.loads(mp.read_text())
    doc["signed"]["members"][0]["auditorKey"]["operator"] = doc["signed"]["members"][0]["archiveOperator"]
    doc["signatures"] = [
        {"keyId": k, "signature": _sig(root_privs[k], doc["signed"])}
        for k in doc["signed"]["selectedSignerKeyIds"]
    ]
    _write(mp, doc)
    with pytest.raises(
        health.ArchiveHealthError,
        match="AUDITOR_NOT_INDEPENDENT|PLANES_OVERLAP",
    ):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp,
            targets=targets,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )

def test_run161_migration_proves_new_set_before_retirement(tmp_path):
    setup, run160_out, root_doc, root_path, root_pin, root_privs, members, privmap, out, _ = _audit(
        tmp_path / "first"
    )
    state = json.loads((out / "trusted-archive-health-state.json").read_text())
    newm, pp, ap = _new_member(3)
    members2 = [members[1], newm]
    receipt = json.loads(
        (run160_out / "release-native-evidence-archive-receipt.json").read_text()
    )
    mp2, _ = _membership(
        tmp_path,
        root_doc,
        root_privs,
        run160_out,
        receipt,
        members2,
        sequence=2,
        previous_head=state["healthChainHeadSha256"],
        previous_members=members,
        issued=NOW + timedelta(minutes=5),
        name="membership-2.json",
    )
    targets = []
    for m in members2:
        if m["archiveId"] == newm["archiveId"]:
            p = Provider(m, pp, now=NOW + timedelta(minutes=5))
            a = Auditor(m, ap, p, now=NOW + timedelta(minutes=5))
        else:
            ppriv, apriv = privmap[m["archiveId"]]
            p = Provider(m, ppriv, now=NOW + timedelta(minutes=5))
            a = Auditor(m, apriv, p, now=NOW + timedelta(minutes=5))
        targets.append((m["archiveId"], p, a))
    out2 = tmp_path / "second"
    result = health.audit_archive_health(
        run160_dir=run160_out,
        output_dir=out2,
        retention_root_path=root_path,
        membership_path=mp2,
        targets=targets,
        expected_retention_root_sha256=root_pin,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        previous_output_dir=out,
        now=NOW + timedelta(minutes=5),
    )
    assert result["retirement_authorized"] == [members[0]["archiveId"]]
    active = json.loads((out2 / "active-archive-health-evidence.json").read_text())
    assert active["retirementAuthorizedArchiveIds"] == [members[0]["archiveId"]]

def test_run161_migration_rejects_missing_new_member_audit(tmp_path):
    setup, run160_out, root_doc, root_path, root_pin, root_privs, members, privmap, out, _ = _audit(
        tmp_path / "first"
    )
    state = json.loads((out / "trusted-archive-health-state.json").read_text())
    newm, _, _ = _new_member(3)
    members2 = [members[1], newm]
    receipt = json.loads(
        (run160_out / "release-native-evidence-archive-receipt.json").read_text()
    )
    mp2, _ = _membership(
        tmp_path,
        root_doc,
        root_privs,
        run160_out,
        receipt,
        members2,
        sequence=2,
        previous_head=state["healthChainHeadSha256"],
        previous_members=members,
        issued=NOW + timedelta(minutes=5),
        name="membership-2.json",
    )
    m = members[1]
    ppriv, apriv = privmap[m["archiveId"]]
    p = Provider(m, ppriv, now=NOW + timedelta(minutes=5))
    targets = [
        (
            m["archiveId"],
            p,
            Auditor(m, apriv, p, now=NOW + timedelta(minutes=5)),
        )
    ]
    with pytest.raises(health.ArchiveHealthError, match="TARGET_SET_MISMATCH"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp2,
            targets=targets,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            previous_output_dir=out,
            now=NOW + timedelta(minutes=5),
        )

def test_run161_migration_cannot_drop_below_durable_copy_minimum(tmp_path):
    setup, run160_out, root_doc, root_path, root_pin, root_privs, members, privmap, out, _ = _audit(
        tmp_path / "first"
    )
    state = json.loads((out / "trusted-archive-health-state.json").read_text())
    receipt = json.loads(
        (run160_out / "release-native-evidence-archive-receipt.json").read_text()
    )
    mp2, _ = _membership(
        tmp_path,
        root_doc,
        root_privs,
        run160_out,
        receipt,
        [members[0]],
        sequence=2,
        previous_head=state["healthChainHeadSha256"],
        previous_members=members,
        issued=NOW + timedelta(minutes=5),
        minimum=2,
        name="membership-2.json",
    )
    ppriv, apriv = privmap[members[0]["archiveId"]]
    p = Provider(members[0], ppriv, now=NOW + timedelta(minutes=5))
    with pytest.raises(health.ArchiveHealthError, match="MEMBER_COUNT_INVALID"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp2,
            targets=[
                (
                    members[0]["archiveId"],
                    p,
                    Auditor(
                        members[0], apriv, p, now=NOW + timedelta(minutes=5)
                    ),
                )
            ],
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            previous_output_dir=out,
            now=NOW + timedelta(minutes=5),
        )

def test_run161_transition_lists_are_cryptographically_exact(tmp_path):
    def mutate(doc, _members, privs):
        doc["signed"]["transition"] = {
            "kind": "bootstrap",
            "addedArchiveIds": ["fake"],
            "retiredArchiveIds": [],
        }
        doc["signatures"] = [
            {
                "keyId": key,
                "signature": _sig(privs[key], doc["signed"]),
            }
            for key in doc["signed"]["selectedSignerKeyIds"]
        ]

    with pytest.raises(
        health.ArchiveHealthError, match="TRANSITION_MISMATCH"
    ):
        _audit(tmp_path, membership_mutator=mutate)

def test_run161_membership_extra_authorized_signature_is_rejected(tmp_path):
    def mutate(doc, _members, privs):
        doc["signatures"].append(
            {
                "keyId": "ret-root-3",
                "signature": _sig(privs["ret-root-3"], doc["signed"]),
            }
        )

    with pytest.raises(
        health.ArchiveHealthError, match="MEMBERSHIP_SIGNATURE_SET_INVALID"
    ):
        _audit(tmp_path, membership_mutator=mutate)

def test_run161_historical_replay_survives_old_retention_expiry(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, _, out, _ = _audit(tmp_path)
    future = NOW + timedelta(days=500)
    verified = health.verify_archive_health(
        run160_dir=run160_out,
        output_dir=out,
        retention_root_path=root_path,
        expected_retention_root_sha256=root_pin,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=future,
        historical=True,
    )
    assert verified["ok"]

def test_run161_live_replay_rejects_expired_current_retention(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, _, out, _ = _audit(tmp_path)
    with pytest.raises(
        health.ArchiveHealthError,
        match="RETENTION_TOO_SHORT|ROOT_EXPIRED|PROVIDER_OBSERVATION_NOT_FRESH|ACTIVE_AUDIT_STALE",
    ):
        health.verify_archive_health(
            run160_dir=run160_out,
            output_dir=out,
            retention_root_path=root_path,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW + timedelta(days=690),
            historical=False,
        )

def test_run161_previous_event_expiry_does_not_freeze_new_current_event(tmp_path):
    setup, run160_out, root_doc, root_path, root_pin, root_privs, members, privmap, out, _ = _audit(tmp_path / "first")
    state = json.loads((out / "trusted-archive-health-state.json").read_text())
    receipt = json.loads((run160_out / "release-native-evidence-archive-receipt.json").read_text())
    later = NOW + timedelta(days=20)
    mp2, _ = _membership(
        tmp_path,
        root_doc,
        root_privs,
        run160_out,
        receipt,
        members,
        sequence=2,
        previous_head=state["healthChainHeadSha256"],
        previous_members=members,
        issued=later,
        name="membership-2.json",
    )
    targets = []
    for m in members:
        pp, ap = privmap[m["archiveId"]]
        p = Provider(m, pp, now=later)
        targets.append((m["archiveId"], p, Auditor(m, ap, p, now=later)))
    result = health.audit_archive_health(
        run160_dir=run160_out,
        output_dir=tmp_path / "second",
        retention_root_path=root_path,
        membership_path=mp2,
        targets=targets,
        expected_retention_root_sha256=root_pin,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        previous_output_dir=out,
        now=later,
    )
    assert result["sequence"] == 2

def test_run161_audit_interval_is_bounded(tmp_path):
    setup, run160_out, root_doc, root_path, root_pin, root_privs, members, privmap, out, _ = _audit(tmp_path / "first")
    state = json.loads((out / "trusted-archive-health-state.json").read_text())
    receipt = json.loads((run160_out / "release-native-evidence-archive-receipt.json").read_text())
    later = NOW + timedelta(days=40)
    mp2, _ = _membership(
        tmp_path,
        root_doc,
        root_privs,
        run160_out,
        receipt,
        members,
        sequence=2,
        previous_head=state["healthChainHeadSha256"],
        previous_members=members,
        issued=later,
        name="membership-2.json",
    )
    targets = []
    for m in members:
        pp, ap = privmap[m["archiveId"]]
        p = Provider(m, pp, now=later)
        targets.append((m["archiveId"], p, Auditor(m, ap, p, now=later)))
    with pytest.raises(health.ArchiveHealthError, match="AUDIT_INTERVAL_EXCEEDED"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp2,
            targets=targets,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            previous_output_dir=out,
            now=later,
        )

def test_run161_bundle_mutation_is_detected(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, _, out, _ = _audit(tmp_path)
    doc = json.loads((out / "release-archive-health-bundle.json").read_text())
    doc["events"][0]["audits"][0]["immutableVersionId"] = "mutated"
    _write(out / "release-archive-health-bundle.json", doc)
    with pytest.raises(health.ArchiveHealthError):
        health.verify_archive_health(run160_dir=run160_out, output_dir=out, retention_root_path=root_path, expected_retention_root_sha256=root_pin, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)

def test_run161_receipt_mutation_is_detected(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, _, out, _ = _audit(tmp_path)
    doc = json.loads((out / "release-archive-health-receipt.json").read_text())
    doc["events"][0]["providerResults"][0]["response"]["signed"]["legalHold"] = True
    _write(out / "release-archive-health-receipt.json", doc)
    with pytest.raises(health.ArchiveHealthError):
        health.verify_archive_health(run160_dir=run160_out, output_dir=out, retention_root_path=root_path, expected_retention_root_sha256=root_pin, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)

def test_run161_duplicate_json_keys_are_rejected(tmp_path):
    p = tmp_path / "dup.json"
    p.write_text('{"a":1,"a":2}\n')
    with pytest.raises(health.ArchiveHealthError, match="DUPLICATE_KEY"):
        health._read_json(p, "DUP")

def test_run161_outputs_contain_no_private_keys_or_local_paths(tmp_path):
    *_, out, _ = _audit(tmp_path)
    text = "\n".join(p.read_text() for p in out.iterdir()).lower()
    assert "privatekey" not in text and "private key" not in text and str(tmp_path).lower() not in text

def test_run161_output_is_create_only(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, _, out, _ = _audit(tmp_path)
    # second attempt to same path is rejected before overwrite
    _, _, _, _, _, _, _, _, _, mp, targets = _setup(tmp_path / "again")
    with pytest.raises(health.ArchiveHealthError, match="OUTPUT_EXISTS"):
        health.audit_archive_health(run160_dir=run160_out, output_dir=out, retention_root_path=root_path, membership_path=mp, targets=targets, expected_retention_root_sha256=root_pin, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW)

def test_run161_command_adapter_bounds_output_while_produced(tmp_path):
    script = tmp_path / "noisy.py"
    script.write_text("import sys\nsys.stdout.write('x' * (9 * 1024 * 1024))\n")
    adapter = health.command_provider([sys.executable, str(script)])
    with pytest.raises(health.ArchiveHealthError, match="OUTPUT_TOO_LARGE"):
        adapter({"x": 1})

def test_run161_documentation_describes_retention_challenges_migration_and_retirement():
    guide = (SECURITY / "RELEASE_ARCHIVE_HEALTH_GUIDE.md").read_text()
    gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    for phrase in ("provider-signed retention", "independent challenge", "retirement authorization", "minimum durable copies", "out-of-band"):
        assert phrase in guide.lower()
    assert "Run 161" in gates

def _prepare_second(tmp_path, *, sequence=2, members_transform=None, issued_delta=timedelta(minutes=5)):
    setup, run160_out, root_doc, root_path, root_pin, root_privs, members, privmap, out, _ = _audit(tmp_path / "first")
    state = json.loads((out / "trusted-archive-health-state.json").read_text())
    receipt = json.loads((run160_out / "release-native-evidence-archive-receipt.json").read_text())
    members2 = list(members)
    if members_transform:
        members2 = members_transform(members2)
    later = NOW + issued_delta
    mp2, _ = _membership(
        tmp_path,
        root_doc,
        root_privs,
        run160_out,
        receipt,
        members2,
        sequence=sequence,
        previous_head=state["healthChainHeadSha256"],
        previous_members=members,
        issued=later,
        name=f"membership-{sequence}.json",
    )
    return (
        setup,
        run160_out,
        root_doc,
        root_path,
        root_pin,
        root_privs,
        members,
        members2,
        privmap,
        out,
        mp2,
        later,
    )

def test_run161_live_checkpoint_remains_valid_within_audit_interval(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, _, out, _ = _audit(tmp_path)
    result = health.verify_archive_health(
        run160_dir=run160_out,
        output_dir=out,
        retention_root_path=root_path,
        expected_retention_root_sha256=root_pin,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW + timedelta(days=1),
        historical=False,
    )
    assert result["ok"]

def test_run161_sequence_skip_is_rejected_before_adapters(tmp_path):
    setup, run160_out, root_doc, root_path, root_pin, root_privs, members, members2, privmap, out, mp2, later = _prepare_second(tmp_path, sequence=3)
    targets = []
    for m in members2:
        pp, ap = privmap[m["archiveId"]]
        p = Provider(m, pp, now=later)
        targets.append((m["archiveId"], p, Auditor(m, ap, p, now=later)))
    with pytest.raises(health.ArchiveHealthError, match="SEQUENCE_NOT_CONSECUTIVE"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp2,
            targets=targets,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            previous_output_dir=out,
            now=later,
        )

def test_run161_existing_archive_id_cannot_be_rebound(tmp_path):
    setup, run160_out, root_doc, root_path, root_pin, root_privs, members, _, out, _ = _audit(tmp_path / "first")
    state = json.loads((out / "trusted-archive-health-state.json").read_text())
    receipt = json.loads((run160_out / "release-native-evidence-archive-receipt.json").read_text())
    later = NOW + timedelta(minutes=5)
    changed = json.loads(json.dumps(members))
    changed[0]["locator"] = "mem+immutable://changed/provider/object"
    mp2, _ = _membership(
        tmp_path,
        root_doc,
        root_privs,
        run160_out,
        receipt,
        changed,
        sequence=2,
        previous_head=state["healthChainHeadSha256"],
        previous_members=members,
        issued=later,
        name="membership-rebound.json",
    )
    with pytest.raises(health.ArchiveHealthError, match="MEMBER_ID_REBOUND"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp2,
            targets=[],
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            previous_output_dir=out,
            now=later,
        )

def test_run161_immutable_version_must_not_change_for_same_archive(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, members2, privmap, out, mp2, later = _prepare_second(tmp_path)
    targets = []
    for i, m in enumerate(members2):
        pp, ap = privmap[m["archiveId"]]
        version = f"version/{m['archiveId']}/2" if i == 0 else None
        p = Provider(m, pp, now=later, version=version)
        targets.append((m["archiveId"], p, Auditor(m, ap, p, now=later)))
    target = tmp_path / "bad-version"
    with pytest.raises(health.ArchiveHealthError, match="IMMUTABLE_VERSION_CHANGED"):
        health.audit_archive_health(run160_dir=run160_out, output_dir=target, retention_root_path=root_path, membership_path=mp2, targets=targets, expected_retention_root_sha256=root_pin, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], previous_output_dir=out, now=later)
    assert not target.exists()

def test_run161_retention_expiry_cannot_move_backward(tmp_path):
    setup, run160_out, _, root_path, root_pin, _, _, members2, privmap, out, mp2, later = _prepare_second(tmp_path)
    targets = []
    for i, m in enumerate(members2):
        pp, ap = privmap[m["archiveId"]]
        p = Provider(m, pp, now=later, retention_days=300 if i == 0 else 365)
        targets.append((m["archiveId"], p, Auditor(m, ap, p, now=later)))
    target = tmp_path / "bad-retention"
    with pytest.raises(health.ArchiveHealthError, match="RETENTION_ROLLBACK"):
        health.audit_archive_health(run160_dir=run160_out, output_dir=target, retention_root_path=root_path, membership_path=mp2, targets=targets, expected_retention_root_sha256=root_pin, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], previous_output_dir=out, now=later)
    assert not target.exists()

def test_run161_retention_mode_must_match_immutability_class(tmp_path):
    def mt(targets, members, privmap):
        out = []
        for aid, _, _ in targets:
            m = next(x for x in members if x["archiveId"] == aid)
            wrong = (
                "object-lock-compliance"
                if m["immutability"] != "object-lock"
                else "content-addressed"
            )
            p = Provider(m, privmap[aid][0], mode=wrong)
            out.append((aid, p, Auditor(m, privmap[aid][1], p)))
        return out

    with pytest.raises(
        health.ArchiveHealthError,
        match="RETENTION_MODE_IMMUTABILITY_MISMATCH",
    ):
        _audit(tmp_path, targets_mutator=mt)

def test_run161_governance_archive_auditor_operator_planes_are_separate(tmp_path):
    setup, run160_out, _, _, root_path, root_pin, root_privs, _, _, mp, targets = _setup(
        tmp_path
    )
    doc = json.loads(mp.read_text())
    m = doc["signed"]["members"][0]
    m["archiveOperator"] = "ret-root-op-1"
    m["providerKey"]["operator"] = "ret-root-op-1"
    doc["signatures"] = [
        {"keyId": k, "signature": _sig(root_privs[k], doc["signed"])}
        for k in doc["signed"]["selectedSignerKeyIds"]
    ]
    _write(mp, doc)
    with pytest.raises(health.ArchiveHealthError, match="GOVERNANCE_OPERATOR_OVERLAP"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp,
            targets=targets,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )

def test_run161_membership_signature_order_is_canonical(tmp_path):
    setup, run160_out, _, _, root_path, root_pin, _, _, _, mp, targets = _setup(tmp_path)
    doc = json.loads(mp.read_text())
    doc["signatures"] = list(reversed(doc["signatures"]))
    _write(mp, doc)
    with pytest.raises(health.ArchiveHealthError, match="SIGNATURES_NOT_SORTED"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp,
            targets=targets,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )

def test_run161_detects_input_directory_drift_during_external_call(tmp_path):
    setup, run160_out, _, _, root_path, root_pin, _, _, _, mp, targets = _setup(tmp_path)
    original = targets[0][1]

    class MutatingProvider:
        def __call__(self, request):
            (run160_out / "injected-during-audit.txt").write_text("drift")
            return original(request)

    targets[0] = (targets[0][0], MutatingProvider(), targets[0][2])
    with pytest.raises(health.ArchiveHealthError, match="INPUT_DRIFT"):
        health.audit_archive_health(
            run160_dir=run160_out,
            output_dir=tmp_path / "bad",
            retention_root_path=root_path,
            membership_path=mp,
            targets=targets,
            expected_retention_root_sha256=root_pin,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )
