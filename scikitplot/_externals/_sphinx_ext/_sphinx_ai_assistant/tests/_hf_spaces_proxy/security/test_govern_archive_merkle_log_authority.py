from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
import copy
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest  # type: ignore[import-not-found]
from cryptography.hazmat.primitives import serialization  # type: ignore[import-not-found]
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # type: ignore[import-not-found]

TESTS = Path(__file__).resolve().parent
SEC = RUNTIME_ROOT / "_hf_spaces_proxy" / "security"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SEC))
import test_verify_archive_merkle_transparency as t164  # noqa: E402
import govern_archive_merkle_log_authority as auth  # noqa: E402
import verify_archive_merkle_transparency as merkle  # noqa: E402

NOW = t164.NOW


def _pub(priv):
    return base64.b64encode(priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def _sig(priv, obj):
    return base64.b64encode(priv.sign(auth._canonical(obj))).decode()


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(auth._canonical(obj))
    return path


def _control_root(tmp, *, recovery=False, name="gov"):
    privs = {}; keys = {}
    for i in range(3):
        kid = f"{name}-key-{i+1}"; priv = Ed25519PrivateKey.generate(); privs[kid] = priv
        row = {"identity": f"{name}-id-{i+1}", "operator": f"{name}-op-{i+1}", "expires": auth._ts(NOW + timedelta(days=800)), "publicKey": _pub(priv)}
        if recovery:
            row["recoveryChannel"] = f"{name}-channel-{1 if i == 2 else i+1}"
        keys[kid] = row
    selected = [f"{name}-key-1", f"{name}-key-2"]
    signed = {
        "_type": "archive-log-recovery-root" if recovery else "archive-log-governance-root",
        "specVersion": "1.0.0", "schemaVersion": 1, "rootId": f"archive-log/{name}", "version": 1,
        "issuedAt": auth._ts(NOW), "expires": auth._ts(NOW + timedelta(days=700)), "threshold": 2,
        "selectedSignerKeyIds": selected, "keys": {k: keys[k] for k in sorted(keys)},
    }
    doc = {"signed": signed, "signatures": [{"keyId": k, "signature": _sig(privs[k], signed)} for k in selected]}
    path = _write(tmp / f"{name}-root.json", doc)
    return doc, path, auth._sha_bytes(auth._canonical(doc)), privs


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run165-base")
    f = t164._fixture(tmp / "pred")
    run164 = tmp / "run164"
    merkle.anchor_merkle_transparency(**t164._kwargs(f, run164), adapters=f["adapters"])
    gov, gp, gpin, gpriv = _control_root(tmp, name="gov")
    rec, rp, rpin, rpriv = _control_root(tmp, recovery=True, name="recovery")
    return {"tmp": tmp, "f": f, "run164": run164, "gov": gov, "gp": gp, "gpin": gpin, "gpriv": gpriv,
            "rec": rec, "rp": rp, "rpin": rpin, "rpriv": rpriv}


def _run164_info(base, now=NOW):
    f = base["f"]
    return auth._load_run164_offline(base["run164"], f["rp"], f["rpin"], now=now, historical=False)


def _transition(base, tmp, *, sequence, kind, current, nxt, previous_state_path=None, compromised=None,
                selected=None, issued=None, gov_privs=None, rec_privs=None):
    if issued is None:
        issued = NOW + timedelta(seconds=sequence - 1)
    info = _run164_info(base, now=NOW)
    changed = sorted(lid for lid in current if current[lid] != nxt[lid])
    previous_revoked = []
    previous_sha = None
    if previous_state_path is not None:
        state_raw = Path(previous_state_path).read_bytes(); state = json.loads(state_raw)
        previous_revoked = list(state["revokedKeyFingerprints"]); previous_sha = auth._sha_bytes(state_raw)
    if kind == "scheduled-rotation":
        replaced = set()
        for lid in changed:
            if current[lid]["publicKey"] != nxt[lid]["publicKey"]:
                replaced.add(auth._pub_fingerprint(current[lid]["publicKey"]))
            if current[lid]["gossipPublicKey"] != nxt[lid]["gossipPublicKey"]:
                replaced.add(auth._pub_fingerprint(current[lid]["gossipPublicKey"]))
        revoked = sorted(set(previous_revoked) | replaced)
        compromised = []
        role = "governance"
        selected = selected or ["gov-key-1", "gov-key-2"]
        privs = gov_privs or base["gpriv"]
    elif kind == "compromise-recovery":
        compromised = sorted(compromised or [])
        replaced = set()
        for lid in changed:
            replaced.add(auth._pub_fingerprint(current[lid]["publicKey"])); replaced.add(auth._pub_fingerprint(current[lid]["gossipPublicKey"]))
        revoked = sorted(set(previous_revoked) | replaced)
        role = "recovery"
        selected = selected or ["recovery-key-1", "recovery-key-2"]
        privs = rec_privs or base["rpriv"]
    else:
        changed = []; compromised = []; revoked = []; role = "governance"; selected = selected or ["gov-key-1", "gov-key-2"]; privs = gov_privs or base["gpriv"]
    signed = {
        "_type": "archive-merkle-log-authority-transition", "specVersion": "1.0.0", "schemaVersion": 1,
        "transitionId": f"transition-{sequence}-{kind}", "sequence": sequence, "kind": kind, "issuedAt": auth._ts(issued),
        "run164Sequence": info["replay"]["sequence"], "merkleConsensusHeadSha256": info["replay"]["merkleConsensusHeadSha256"],
        "run164Artifacts": info["artifacts"], "previousAuthorityStateSha256": previous_sha,
        "currentAuthority": current, "nextAuthority": nxt, "changedLogIds": changed,
        "compromisedKeyFingerprints": compromised, "revokedKeyFingerprints": revoked,
        "authorizationRole": role, "selectedSignerKeyIds": selected, "handoffSubjectSha256s": {},
    }
    subjects = {lid: auth._handoff_subject(signed=signed, log_id=lid, checkpoint=info["checkpoints"][lid]) for lid in sorted(current)}
    signed["handoffSubjectSha256s"] = {lid: auth._sha_bytes(auth._canonical(subjects[lid])) for lid in sorted(subjects)}
    doc = {"signed": signed, "signatures": [{"keyId": k, "signature": _sig(privs[k], signed)} for k in selected]}
    return doc, info


def _handoff(base, transition, info, *, new_privs, old_privs=None):
    signed = transition["signed"]; proofs = []
    for lid in sorted(signed["nextAuthority"]):
        subject = auth._handoff_subject(signed=signed, log_id=lid, checkpoint=info["checkpoints"][lid])
        changed = lid in signed["changedLogIds"]
        old_log = old_gossip = None
        if signed["kind"] == "scheduled-rotation" and changed:
            old_log = _sig(old_privs["log"][lid], subject); old_gossip = _sig(old_privs["gossip"][lid], subject)
        proofs.append({"logId": lid, "subjectSha256": auth._sha_bytes(auth._canonical(subject)),
                       "newLogSignature": _sig(new_privs["log"][lid], subject), "newGossipSignature": _sig(new_privs["gossip"][lid], subject),
                       "oldLogSignature": old_log, "oldGossipSignature": old_gossip})
    return {"schemaVersion": 1, "transitionId": signed["transitionId"], "sequence": signed["sequence"], "proofs": proofs}


def _bootstrap(base, tmp):
    info = _run164_info(base); current = copy.deepcopy(base["f"]["root"]["signed"]["logs"])
    tr, info = _transition(base, tmp, sequence=1, kind="bootstrap", current=current, nxt=copy.deepcopy(current))
    hf = _handoff(base, tr, info, new_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]})
    tp = _write(tmp / "bootstrap-transition.json", tr); hp = _write(tmp / "bootstrap-handoff.json", hf); out = tmp / "out1"
    auth.apply_log_authority_transition(
        run164_dir=base["run164"],
        transparency_root_path=base["f"]["rp"],
        transparency_root_pin=base["f"]["rpin"],
        governance_root_path=base["gp"],
        governance_root_pin=base["gpin"],
        recovery_root_path=base["rp"],
        recovery_root_pin=base["rpin"],
        transition_path=tp,
        handoff_path=hp,
        output_dir=out,
        now=NOW,
    )
    return out, tr, hf


def _kwargs(base, out, now=NOW, historical=False):
    return dict(run164_dir=base["run164"], transparency_root_path=base["f"]["rp"], transparency_root_pin=base["f"]["rpin"],
                governance_root_path=base["gp"], governance_root_pin=base["gpin"], recovery_root_path=base["rp"], recovery_root_pin=base["rpin"],
                output_dir=out, now=now, historical=historical)


def test_run165_bootstrap_and_offline_verify(base, tmp_path):
    out, _, _ = _bootstrap(base, tmp_path)
    result = auth.verify_log_authority_history(**_kwargs(base, out))
    assert result["ok"] and result["sequence"] == 1


def test_run165_wrong_governance_pin_rejected(base, tmp_path):
    out, _, _ = _bootstrap(base, tmp_path)
    kw = _kwargs(base, out); kw["governance_root_pin"] = "0" * 64
    with pytest.raises(auth.ArchiveLogAuthorityError):
        auth.verify_log_authority_history(**kw)


def test_run165_extra_root_signature_rejected(base, tmp_path):
    doc = copy.deepcopy(base["gov"])
    doc["signatures"].append(
        {
            "keyId": "gov-key-3",
            "signature": _sig(base["gpriv"]["gov-key-3"], doc["signed"]),
        },
    )
    _write(tmp_path / "bad-root.json", doc)
    with pytest.raises(auth.ArchiveLogAuthorityError):
        auth._verify_control_root(doc, auth._sha_bytes(auth._canonical(doc)), now=NOW, recovery=False, historical=False)


def test_run165_bootstrap_requires_exact_run164_authority(base, tmp_path):
    current = copy.deepcopy(base["f"]["root"]["signed"]["logs"]); nxt = copy.deepcopy(current)
    lid = sorted(nxt)[0]; nxt[lid]["operator"] = "other-op"
    tr, info = _transition(base, tmp_path, sequence=1, kind="bootstrap", current=current, nxt=nxt)
    hf = _handoff(base, tr, info, new_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]})
    with pytest.raises(auth.ArchiveLogAuthorityError):
        auth.apply_log_authority_transition(
            run164_dir=base["run164"],
            transparency_root_path=base["f"]["rp"],
            transparency_root_pin=base["f"]["rpin"],
            governance_root_path=base["gp"],
            governance_root_pin=base["gpin"],
            recovery_root_path=base["rp"],
            recovery_root_pin=base["rpin"],
            transition_path=_write(tmp_path/"t.json", tr),
            handoff_path=_write(tmp_path/"h.json", hf),
            output_dir=tmp_path/"out",
            now=NOW,
        )


def test_run165_handoff_signature_mutation_rejected(base, tmp_path):
    current = copy.deepcopy(base["f"]["root"]["signed"]["logs"])
    tr, info = _transition(base, tmp_path, sequence=1, kind="bootstrap", current=current, nxt=copy.deepcopy(current))
    hf = _handoff(base, tr, info, new_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]}); hf["proofs"][0]["newLogSignature"] = base64.b64encode(b"x"*64).decode()
    with pytest.raises(auth.ArchiveLogAuthorityError):
        auth._verify_handoff_document(
            hf,
            transition=auth._verify_transition_document(
                tr,
                governance=auth._verify_control_root(
                    base["gov"],
                    base["gpin"],
                    now=NOW,
                    recovery=False,
                    historical=False,
                ),
                recovery=auth._verify_control_root(
                    base["rec"],
                    base["rpin"],
                    now=NOW,
                    recovery=True,
                    historical=False,
                ),
                run164=info,
                previous_state=None,
                previous_state_raw=None,
                now=NOW,
                creation=True,
            ),
        )


def test_run165_scheduled_rotation_requires_old_and_new_handoff(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    state_path = out1 / "trusted-archive-log-authority-state.json"; state = json.loads(state_path.read_bytes())
    current = copy.deepcopy(state["activeAuthority"]); nxt = copy.deepcopy(current); lid = sorted(nxt)[0]
    new_lp = Ed25519PrivateKey.generate(); new_gp = Ed25519PrivateKey.generate(); nxt[lid]["publicKey"] = _pub(new_lp); nxt[lid]["gossipPublicKey"] = _pub(new_gp)
    tr, info = _transition(base, tmp_path, sequence=2, kind="scheduled-rotation", current=current, nxt=nxt, previous_state_path=state_path)
    logs = dict(base["f"]["lpriv"]); gossip = dict(base["f"]["gpriv"]); logs[lid] = new_lp; gossip[lid] = new_gp
    hf = _handoff(base, tr, info, new_privs={"log": logs, "gossip": gossip}, old_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]})
    out2 = tmp_path / "out2"
    auth.apply_log_authority_transition(
        run164_dir=base["run164"],
        transparency_root_path=base["f"]["rp"],
        transparency_root_pin=base["f"]["rpin"],
        governance_root_path=base["gp"],
        governance_root_pin=base["gpin"],
        recovery_root_path=base["rp"],
        recovery_root_pin=base["rpin"],
        transition_path=_write(tmp_path/"t2.json", tr),
        handoff_path=_write(tmp_path/"h2.json", hf),
        previous_output_dir=out1,
        output_dir=out2,
        now=NOW,
    )
    result = auth.verify_log_authority_history(**_kwargs(base, out2)); assert result["sequence"] == 2
    active = json.loads((out2/"active-archive-log-authority.json").read_bytes())
    assert auth._pub_fingerprint(current[lid]["publicKey"]) in active["revokedKeyFingerprints"]


def test_run165_scheduled_rotation_missing_old_signature_rejected(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    state_path = out1/"trusted-archive-log-authority-state.json"
    state = json.loads(state_path.read_bytes())
    current = copy.deepcopy(state["activeAuthority"])
    nxt = copy.deepcopy(current)
    lid = sorted(nxt)[0]
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    nxt[lid]["gossipPublicKey"] = _pub(gp)
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="scheduled-rotation",
        current=current,
        nxt=nxt,
        previous_state_path=state_path
    )
    logs = dict(base["f"]["lpriv"])
    gossip = dict(base["f"]["gpriv"])
    logs[lid] = lp
    gossip[lid] = gp
    hf = _handoff(
        base,
        tr,
        info,
        new_privs={"log": logs, "gossip": gossip},
        old_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]},
    )
    hf["proofs"][0 if hf["proofs"][0]["logId"] == lid else 1]["oldLogSignature"] = None
    with pytest.raises(auth.ArchiveLogAuthorityError):
        auth._verify_handoff_document(
            hf,
            transition=auth._verify_transition_document(
                tr,
                governance=auth._verify_control_root(
                    base["gov"],
                    base["gpin"],
                    now=NOW,
                    recovery=False,
                    historical=False,
                ),
                recovery=auth._verify_control_root(
                    base["rec"],
                    base["rpin"],
                    now=NOW,
                    recovery=True,
                    historical=False,
                ),
                run164=info,
                previous_state=state,
                previous_state_raw=state_path.read_bytes(),
                now=NOW,
                creation=True,
            ),
        )


def test_run165_compromise_recovery_replaces_both_keys_without_old_signatures(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    state_path = out1 / "trusted-archive-log-authority-state.json"
    state = json.loads(state_path.read_bytes())
    current = copy.deepcopy(state["activeAuthority"])
    nxt = copy.deepcopy(current)
    lid = sorted(nxt)[1]
    oldfp = auth._pub_fingerprint(current[lid]["publicKey"])
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    nxt[lid]["gossipPublicKey"] = _pub(gp)
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="compromise-recovery",
        current=current,
        nxt=nxt,
        previous_state_path=state_path,
        compromised=[oldfp],
    )
    logs = dict(base["f"]["lpriv"])
    gossip = dict(base["f"]["gpriv"])
    logs[lid] = lp
    gossip[lid] = gp
    hf = _handoff(base, tr, info, new_privs={"log": logs, "gossip": gossip})
    out2 = tmp_path / "out2"
    auth.apply_log_authority_transition(
        run164_dir=base["run164"],
        transparency_root_path=base["f"]["rp"],
        transparency_root_pin=base["f"]["rpin"],
        governance_root_path=base["gp"],
        governance_root_pin=base["gpin"],
        recovery_root_path=base["rp"],
        recovery_root_pin=base["rpin"],
        transition_path=_write(tmp_path / "rt.json", tr),
        handoff_path=_write(tmp_path / "rh.json", hf),
        previous_output_dir=out1,
        output_dir=out2,
        now=NOW,
    )
    active = json.loads(
        (out2 / "active-archive-log-authority.json").read_bytes()
    )
    assert oldfp in active["revokedKeyFingerprints"] and active["sequence"] == 2


def test_run165_recovery_requires_full_log_and_gossip_rotation(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[1]
    lp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    fp = auth._pub_fingerprint(cur[lid]["publicKey"])
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="compromise-recovery",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
        compromised=[fp],
    )
    with pytest.raises(auth.ArchiveLogAuthorityError):
        auth._verify_transition_document(
            tr,
            governance=auth._verify_control_root(
                base["gov"],
                base["gpin"],
                now=NOW,
                recovery=False,
                historical=False,
            ),
            recovery=auth._verify_control_root(
                base["rec"],
                base["rpin"],
                now=NOW,
                recovery=True,
                historical=False,
            ),
            run164=info,
            previous_state=st,
            previous_state_raw=sp.read_bytes(),
            now=NOW,
            creation=True,
        )


def test_run165_recovery_transition_requires_channel_diversity(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[1]
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    nxt[lid]["gossipPublicKey"] = _pub(gp)
    fp = auth._pub_fingerprint(cur[lid]["publicKey"])
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="compromise-recovery",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
        compromised=[fp],
        selected=["recovery-key-1", "recovery-key-3"],
    )
    with pytest.raises(
        auth.ArchiveLogAuthorityError,
        match="RECOVERY_QUORUM",
    ):
        auth._verify_transition_document(
            tr,
            governance=auth._verify_control_root(
                base["gov"],
                base["gpin"],
                now=NOW,
                recovery=False,
                historical=False,
            ),
            recovery=auth._verify_control_root(
                base["rec"],
                base["rpin"],
                now=NOW,
                recovery=True,
                historical=False,
            ),
            run164=info,
            previous_state=st,
            previous_state_raw=sp.read_bytes(),
            now=NOW,
            creation=True,
        )


def test_run165_recovery_changed_set_must_match_compromised_keys(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lids = sorted(nxt)
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lids[0]]["publicKey"] = _pub(lp)
    nxt[lids[0]]["gossipPublicKey"] = _pub(gp)
    fp = auth._pub_fingerprint(cur[lids[1]]["publicKey"])
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="compromise-recovery",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
        compromised=[fp],
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="CHANGED_SET"):
        auth._verify_transition_document(
            tr,
            governance=auth._verify_control_root(
                base["gov"],
                base["gpin"],
                now=NOW,
                recovery=False,
                historical=False,
            ),
            recovery=auth._verify_control_root(
                base["rec"],
                base["rpin"],
                now=NOW,
                recovery=True,
                historical=False,
            ),
            run164=info,
            previous_state=st,
            previous_state_raw=sp.read_bytes(),
            now=NOW,
            creation=True,
        )


def test_run165_operator_rebind_without_key_rotation_rejected(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[0]
    nxt[lid]["operator"] = "new-operator"
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="scheduled-rotation",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="REBOUND"):
        auth._verify_transition_document(
            tr,
            governance=auth._verify_control_root(
                base["gov"], base["gpin"], now=NOW, recovery=False, historical=False
            ),
            recovery=auth._verify_control_root(
                base["rec"], base["rpin"], now=NOW, recovery=True, historical=False
            ),
            run164=info,
            previous_state=st,
            previous_state_raw=sp.read_bytes(),
            now=NOW,
            creation=True,
        )


def test_run165_control_plane_overlap_rejected(base, tmp_path):
    gov = copy.deepcopy(base["gov"])
    gov["signed"]["keys"]["gov-key-1"]["publicKey"] = base["f"]["root"]["signed"]["logs"][sorted(base["f"]["root"]["signed"]["logs"])[0]]["publicKey"]
    # Re-sign to prove the overlap check, not signature failure.
    gov["signatures"] = [
        {"keyId": k, "signature": _sig(base["gpriv"][k], gov["signed"])}
        for k in gov["signed"]["selectedSignerKeyIds"]
    ]
    # Public key replacement invalidates signer 1 by design; verify plane separation directly with a parsed fake control root.
    parsed = {"keys": {k: dict(v) for k, v in base["gov"]["signed"]["keys"].items()}}
    parsed["keys"]["gov-key-1"]["publicKey"] = gov["signed"]["keys"]["gov-key-1"]["publicKey"]
    recovery = auth._verify_control_root(
        base["rec"], base["rpin"], now=NOW, recovery=True, historical=False
    )
    run164 = _run164_info(base)
    with pytest.raises(auth.ArchiveLogAuthorityError, match="OVERLAP"):
        auth._enforce_plane_separation(
            parsed, recovery, run164["root"], run164["root"]["logs"]
        )


def test_run165_stale_transition_rejected_but_historical_output_survives(base, tmp_path):
    current = copy.deepcopy(base["f"]["root"]["signed"]["logs"])
    old = NOW - timedelta(hours=2)
    tr, info = _transition(
        base,
        tmp_path,
        sequence=1,
        kind="bootstrap",
        current=current,
        nxt=copy.deepcopy(current),
        issued=old,
    )
    hf = _handoff(
        base,
        tr,
        info,
        new_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]},
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="STALE"):
        auth.apply_log_authority_transition(
            run164_dir=base["run164"],
            transparency_root_path=base["f"]["rp"],
            transparency_root_pin=base["f"]["rpin"],
            governance_root_path=base["gp"],
            governance_root_pin=base["gpin"],
            recovery_root_path=base["rp"],
            recovery_root_pin=base["rpin"],
            transition_path=_write(tmp_path / "t.json", tr),
            handoff_path=_write(tmp_path / "h.json", hf),
            output_dir=tmp_path / "out",
            now=NOW,
        )
    out, _, _ = _bootstrap(base, tmp_path / "fresh")
    assert auth.verify_log_authority_history(
        **_kwargs(base, out, now=NOW + timedelta(days=750), historical=True)
    )["ok"]


def test_run165_run164_mutation_detected(base, tmp_path):
    out, _, _ = _bootstrap(base, tmp_path / "b")
    bad = tmp_path / "run164"
    import shutil

    shutil.copytree(base["run164"], bad)
    p = bad / "active-archive-merkle-evidence.json"
    doc = json.loads(p.read_bytes())
    doc["leafHash"] = "0" * 64
    p.write_bytes(auth._canonical(doc))
    kw = _kwargs(base, out)
    kw["run164_dir"] = bad
    with pytest.raises(auth.ArchiveLogAuthorityError):
        auth.verify_log_authority_history(**kw)


def test_run165_duplicate_json_keys_rejected(tmp_path):
    p = tmp_path / "dup.json"
    p.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(auth.ArchiveLogAuthorityError, match="DUPLICATE"):
        auth._read_json(p, "X")


def test_run165_output_exists_rejected(base, tmp_path):
    out, _, _ = _bootstrap(base, tmp_path / "b")
    current = copy.deepcopy(base["f"]["root"]["signed"]["logs"])
    tr, info = _transition(
        base,
        tmp_path,
        sequence=1,
        kind="bootstrap",
        current=current,
        nxt=copy.deepcopy(current),
    )
    hf = _handoff(
        base,
        tr,
        info,
        new_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]},
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="OUTPUT_EXISTS"):
        auth.apply_log_authority_transition(
            run164_dir=base["run164"],
            transparency_root_path=base["f"]["rp"],
            transparency_root_pin=base["f"]["rpin"],
            governance_root_path=base["gp"],
            governance_root_pin=base["gpin"],
            recovery_root_path=base["rp"],
            recovery_root_pin=base["rpin"],
            transition_path=_write(tmp_path / "t.json", tr),
            handoff_path=_write(tmp_path / "h.json", hf),
            output_dir=out,
            now=NOW,
        )


def test_run165_sequence_skip_rejected(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[0]
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    nxt[lid]["gossipPublicKey"] = _pub(gp)
    tr, info = _transition(
        base,
        tmp_path,
        sequence=3,
        kind="scheduled-rotation",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="SEQUENCE"):
        auth._verify_transition_document(
            tr,
            governance=auth._verify_control_root(
                base["gov"],
                base["gpin"],
                now=NOW,
                recovery=False,
                historical=False,
            ),
            recovery=auth._verify_control_root(
                base["rec"],
                base["rpin"],
                now=NOW,
                recovery=True,
                historical=False,
            ),
            run164=info,
            previous_state=st,
            previous_state_raw=sp.read_bytes(),
            now=NOW,
            creation=True,
        )


def test_run165_revoked_key_cannot_be_reintroduced(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[0]
    old_pub = cur[lid]["publicKey"]
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    nxt[lid]["gossipPublicKey"] = _pub(gp)
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="scheduled-rotation",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
    )
    logs = dict(base["f"]["lpriv"])
    goss = dict(base["f"]["gpriv"])
    logs[lid] = lp
    goss[lid] = gp
    hf = _handoff(
        base,
        tr,
        info,
        new_privs={"log": logs, "gossip": goss},
        old_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]},
    )
    out2 = tmp_path / "out2"
    auth.apply_log_authority_transition(
        run164_dir=base["run164"],
        transparency_root_path=base["f"]["rp"],
        transparency_root_pin=base["f"]["rpin"],
        governance_root_path=base["gp"],
        governance_root_pin=base["gpin"],
        recovery_root_path=base["rp"],
        recovery_root_pin=base["rpin"],
        transition_path=_write(tmp_path / "t2.json", tr),
        handoff_path=_write(tmp_path / "h2.json", hf),
        previous_output_dir=out1,
        output_dir=out2,
        now=NOW,
    )
    sp2 = out2 / "trusted-archive-log-authority-state.json"
    st2 = json.loads(sp2.read_bytes())
    cur2 = copy.deepcopy(st2["activeAuthority"])
    nxt2 = copy.deepcopy(cur2)
    nxt2[lid]["publicKey"] = old_pub
    tr3, info3 = _transition(
        base,
        tmp_path,
        sequence=3,
        kind="scheduled-rotation",
        current=cur2,
        nxt=nxt2,
        previous_state_path=sp2,
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="REINTRODUCED|ACTIVE"):
        auth._verify_transition_document(
            tr3,
            governance=auth._verify_control_root(
                base["gov"], base["gpin"], now=NOW, recovery=False, historical=False
            ),
            recovery=auth._verify_control_root(
                base["rec"], base["rpin"], now=NOW, recovery=True, historical=False
            ),
            run164=info3,
            previous_state=st2,
            previous_state_raw=sp2.read_bytes(),
            now=NOW,
            creation=True,
        )


def test_run165_transition_artifact_is_deterministic(base, tmp_path):
    out, _, _ = _bootstrap(base, tmp_path / "b")
    raw1 = (out / "release-archive-log-authority-bundle.json").read_bytes()
    assert raw1 == auth._canonical(json.loads(raw1))


def test_run165_documentation_mentions_old_new_handoff_recovery_and_revocation():
    text = (SEC / "RELEASE_ARCHIVE_LOG_AUTHORITY_GUIDE.md").read_text().lower()
    for phrase in ("old + new", "compromise recovery", "permanent revocation", "run 164"):
        assert phrase in text


def test_run165_live_control_root_expiry_fails_but_historical_replay_survives(base, tmp_path):
    out, _, _ = _bootstrap(base, tmp_path / "b")
    with pytest.raises(auth.ArchiveLogAuthorityError, match="EXPIRED"):
        auth.verify_log_authority_history(
            **_kwargs(base, out, now=NOW + timedelta(days=750), historical=False)
        )
    assert auth.verify_log_authority_history(
        **_kwargs(base, out, now=NOW + timedelta(days=750), historical=True)
    )["ok"]


def test_run165_state_mutation_detected(base, tmp_path):
    out, _, _ = _bootstrap(base, tmp_path / "b")
    p = out / "trusted-archive-log-authority-state.json"
    doc = json.loads(p.read_bytes())
    doc["logAuthorityChainHeadSha256"] = "0" * 64
    p.write_bytes(auth._canonical(doc))
    with pytest.raises(auth.ArchiveLogAuthorityError, match="STATE_MISMATCH"):
        auth.verify_log_authority_history(**_kwargs(base, out))


def test_run165_transition_id_reuse_rejected_on_replay(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[0]
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    nxt[lid]["gossipPublicKey"] = _pub(gp)
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="scheduled-rotation",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
    )
    tr["signed"]["transitionId"] = "transition-1-bootstrap"
    # Recompute subjects and governance signatures for the reused ID.
    subjects = {
        x: auth._handoff_subject(
            signed=tr["signed"], log_id=x, checkpoint=info["checkpoints"][x]
        )
        for x in sorted(cur)
    }
    tr["signed"]["handoffSubjectSha256s"] = {
        x: auth._sha_bytes(auth._canonical(subjects[x])) for x in sorted(subjects)
    }
    tr["signatures"] = [
        {"keyId": k, "signature": _sig(base["gpriv"][k], tr["signed"])}
        for k in tr["signed"]["selectedSignerKeyIds"]
    ]
    logs = dict(base["f"]["lpriv"])
    goss = dict(base["f"]["gpriv"])
    logs[lid] = lp
    goss[lid] = gp
    hf = _handoff(
        base,
        tr,
        info,
        new_privs={"log": logs, "gossip": goss},
        old_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]},
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="ID_REUSED"):
        # Apply stages then full replay catches cross-event reuse before commit.
        auth.apply_log_authority_transition(
            run164_dir=base["run164"],
            transparency_root_path=base["f"]["rp"],
            transparency_root_pin=base["f"]["rpin"],
            governance_root_path=base["gp"],
            governance_root_pin=base["gpin"],
            recovery_root_path=base["rp"],
            recovery_root_pin=base["rpin"],
            transition_path=_write(tmp_path / "t.json", tr),
            handoff_path=_write(tmp_path / "h.json", hf),
            previous_output_dir=out1,
            output_dir=tmp_path / "out2",
            now=NOW + timedelta(seconds=1),
        )
    assert not (tmp_path/"out2").exists()


def test_run165_transition_time_must_increase(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[0]
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    nxt[lid]["gossipPublicKey"] = _pub(gp)
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="scheduled-rotation",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
        issued=NOW,
    )
    logs = dict(base["f"]["lpriv"])
    goss = dict(base["f"]["gpriv"])
    logs[lid] = lp
    goss[lid] = gp
    hf = _handoff(
        base,
        tr,
        info,
        new_privs={"log": logs, "gossip": goss},
        old_privs={
            "log": base["f"]["lpriv"],
            "gossip": base["f"]["gpriv"],
        },
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="TIME_NOT_MONOTONIC"):
        auth.apply_log_authority_transition(
            run164_dir=base["run164"],
            transparency_root_path=base["f"]["rp"],
            transparency_root_pin=base["f"]["rpin"],
            governance_root_path=base["gp"],
            governance_root_pin=base["gpin"],
            recovery_root_path=base["rp"],
            recovery_root_pin=base["rpin"],
            transition_path=_write(tmp_path / "t.json", tr),
            handoff_path=_write(tmp_path / "h.json", hf),
            previous_output_dir=out1,
            output_dir=tmp_path / "out2",
            now=NOW + timedelta(seconds=1),
        )


def test_run165_handoff_proof_membership_is_exact(base, tmp_path):
    current = copy.deepcopy(base["f"]["root"]["signed"]["logs"])
    tr, info = _transition(
        base,
        tmp_path,
        sequence=1,
        kind="bootstrap",
        current=current,
        nxt=copy.deepcopy(current),
    )
    hf = _handoff(
        base,
        tr,
        info,
        new_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]},
    )
    hf["proofs"].pop()
    parsed = auth._verify_transition_document(
        tr,
        governance=auth._verify_control_root(
            base["gov"],
            base["gpin"],
            now=NOW,
            recovery=False,
            historical=False,
        ),
        recovery=auth._verify_control_root(
            base["rec"],
            base["rpin"],
            now=NOW,
            recovery=True,
            historical=False,
        ),
        run164=info,
        previous_state=None,
        previous_state_raw=None,
        now=NOW,
        creation=True,
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="PROOF_SET"):
        auth._verify_handoff_document(hf, transition=parsed)


def test_run165_recovery_forbids_old_handoff_signatures(base, tmp_path):
    out1, _, _ = _bootstrap(base, tmp_path / "b")
    sp = out1 / "trusted-archive-log-authority-state.json"
    st = json.loads(sp.read_bytes())
    cur = copy.deepcopy(st["activeAuthority"])
    nxt = copy.deepcopy(cur)
    lid = sorted(nxt)[1]
    lp = Ed25519PrivateKey.generate()
    gp = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(lp)
    nxt[lid]["gossipPublicKey"] = _pub(gp)
    fp = auth._pub_fingerprint(cur[lid]["publicKey"])
    tr, info = _transition(
        base,
        tmp_path,
        sequence=2,
        kind="compromise-recovery",
        current=cur,
        nxt=nxt,
        previous_state_path=sp,
        compromised=[fp],
    )
    logs = dict(base["f"]["lpriv"])
    goss = dict(base["f"]["gpriv"])
    logs[lid] = lp
    goss[lid] = gp
    hf = _handoff(base, tr, info, new_privs={"log": logs, "gossip": goss})
    subject = auth._handoff_subject(
        signed=tr["signed"], log_id=lid, checkpoint=info["checkpoints"][lid]
    )
    row = next(x for x in hf["proofs"] if x["logId"] == lid)
    row["oldLogSignature"] = _sig(base["f"]["lpriv"][lid], subject)
    parsed = auth._verify_transition_document(
        tr,
        governance=auth._verify_control_root(
            base["gov"], base["gpin"], now=NOW, recovery=False, historical=False
        ),
        recovery=auth._verify_control_root(
            base["rec"], base["rpin"], now=NOW, recovery=True, historical=False
        ),
        run164=info,
        previous_state=st,
        previous_state_raw=sp.read_bytes(),
        now=NOW + timedelta(seconds=1),
        creation=True,
    )
    with pytest.raises(auth.ArchiveLogAuthorityError, match="FORBIDDEN"):
        auth._verify_handoff_document(hf, transition=parsed)


def test_run165_authority_input_drift_fails_before_commit(base, tmp_path, monkeypatch):
    current = copy.deepcopy(base["f"]["root"]["signed"]["logs"])
    tr, info = _transition(
        base,
        tmp_path,
        sequence=1,
        kind="bootstrap",
        current=current,
        nxt=copy.deepcopy(current),
    )
    hf = _handoff(
        base,
        tr,
        info,
        new_privs={"log": base["f"]["lpriv"], "gossip": base["f"]["gpriv"]},
    )
    tp = _write(tmp_path / "t.json", tr)
    hp = _write(tmp_path / "h.json", hf)
    out = tmp_path / "out"
    original = auth._verify_handoff_document

    def mutate_after_verify(doc, *, transition):
        result = original(doc, transition=transition)
        tp.write_bytes(tp.read_bytes() + b" ")
        return result

    monkeypatch.setattr(auth, "_verify_handoff_document", mutate_after_verify)
    with pytest.raises(auth.ArchiveLogAuthorityError, match="INPUT_DRIFT"):
        auth.apply_log_authority_transition(
            run164_dir=base["run164"],
            transparency_root_path=base["f"]["rp"],
            transparency_root_pin=base["f"]["rpin"],
            governance_root_path=base["gp"],
            governance_root_pin=base["gpin"],
            recovery_root_path=base["rp"],
            recovery_root_pin=base["rpin"],
            transition_path=tp,
            handoff_path=hp,
            output_dir=out,
            now=NOW,
        )
    assert not out.exists()
