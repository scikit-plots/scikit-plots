from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
import copy
import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

TESTS = Path(__file__).resolve().parent
SEC = RUNTIME_ROOT / "_hf_spaces_proxy" / "security"
sys.path.insert(0, str(TESTS)); sys.path.insert(0, str(SEC))
import test_verify_archive_merkle_transparency as t164
import test_govern_archive_merkle_log_authority as t165
import test_continue_archive_merkle_authority as t166
import continue_archive_merkle_authority as cont
import govern_archive_merkle_log_authority as auth
import rebridge_archive_merkle_authority as rb
import verify_archive_merkle_transparency as merkle

NOW = t164.NOW
T2 = NOW + timedelta(minutes=5)
T3 = NOW + timedelta(minutes=10)
T4 = NOW + timedelta(minutes=15)
T5 = NOW + timedelta(minutes=20)


def _pub(p):
    return base64.b64encode(p.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def _sig(p, obj):
    return base64.b64encode(p.sign(rb._canonical(obj))).decode()


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rb._canonical(obj))
    return path


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run167-base")
    f = t164._fixture(tmp / "pred")
    run164 = tmp / "run164"
    merkle.anchor_merkle_transparency(**t164._kwargs(f, run164), adapters=f["adapters"])
    gov, gp, gpin, gpriv = t165._control_root(tmp, name="gov")
    rec, rp, rpin, rpriv = t165._control_root(tmp, recovery=True, name="recovery")
    b = {"tmp": tmp, "f": f, "run164": run164, "gov": gov, "gp": gp, "gpin": gpin, "gpriv": gpriv,
         "rec": rec, "rp": rp, "rpin": rpin, "rpriv": rpriv}
    run165, _, _ = t165._bootstrap(b, tmp / "bootstrap")
    r = f["r"]
    run161b, wout2, aout2 = t166._advance_run163(f, tmp / "advance2", seq=2, run161_prev=r["out"], wout_prev=f["wout"], aout_prev=f["aout"], now=T2)
    b.update(run165=run165, run161b=run161b, wout2=wout2, aout2=aout2)
    ads166 = t166._adapters(b, now=T2)
    run166 = tmp / "run166"
    cont.continue_merkle_authority(**t166._kwargs(b, run166, now=T2), adapters=ads166)
    leaves = {lid: list(log.leaves) for lid, log, _ in ads166}
    run161c, wout3, aout3 = t166._advance_run163(f, tmp / "advance3", seq=3, run161_prev=run161b, wout_prev=wout2, aout_prev=aout2, now=T3)
    b.update(run166=run166, leaves=leaves, run161c=run161c, wout3=wout3, aout3=aout3)
    return b


def _kwargs(base, out, *, run161=None, wout=None, aout=None, now=T3, historical=None):
    f = base["f"]; r = f["r"]
    d = dict(
        run160_dir=r["run160"], run166_run161_dir=base["run161b"], run161_dir=run161 or base["run161c"],
        retention_root_path=r["root_path"], retention_root_pin=r["root_pin"], witness_root_path=f["wrp"],
        witness_root_pin=f["wrpin"], bootstrap_pin=r["setup"]["pin"], recovery_pin=r["setup"]["rr_pin"],
        attestation_pins=[r["setup"]["ca_pin"]], run166_run162_dir=base["wout2"], run162_dir=wout or base["wout3"],
        anchor_plan_path=f["pp"], run166_run163_dir=base["aout2"], run163_dir=aout or base["aout3"],
        run164_dir=base["run164"], transparency_root_path=f["rp"], transparency_root_pin=f["rpin"],
        governance_root_path=base["gp"], governance_root_pin=base["gpin"], recovery_root_path=base["rp"],
        recovery_root_pin=base["rpin"], run165_dir=base["run165"], run166_dir=base["run166"], output_dir=out, now=now,
    )
    if historical is not None:
        d["historical"] = historical
    return d


def _source(base, previous=None):
    if previous is None:
        run165_docs, run165_raws = auth._load_output(base["run165"])
        run166_docs, run166_raws = cont._load_output(base["run166"])
        run165_info = {"docs": run165_docs, "raws": run165_raws, "artifacts": rb._artifact_map(run165_raws), "state": run165_docs["trusted-archive-log-authority-state.json"]}
        run166_info = {"docs": run166_docs, "raws": run166_raws, "artifacts": rb._artifact_map(run166_raws), "state": run166_docs["trusted-archive-merkle-continuity-state.json"], "active": run166_docs["active-archive-merkle-continuity.json"]}
        return rb._source_from_base(run166_info, run165_info)
    state_raw = (previous / "trusted-archive-merkle-rebridge-state.json").read_bytes()
    st = json.loads(state_raw)
    return {
        "sourceKind": "run167", "stateRaw": state_raw, "stateSha256": rb._sha_bytes(state_raw),
        "merkleSequence": st["sequence"], "merkleHead": st["merkleRebridgeContinuityHeadSha256"],
        "checkpoints": st["lastCheckpoints"], "authority": st["activeAuthority"], "revoked": st["revokedKeyFingerprints"],
        "rebridgeSequence": st["rebridgeSequence"], "authorityHead": st["archiveMerkleRebridgeAuthorityHeadSha256"],
        "lastTransitionIssued": None if st["lastTransitionIssuedAt"] is None else rb._dt(st["lastTransitionIssuedAt"], "X"),
    }


def _transition(base, tmp, *, source=None, kind="scheduled-rotation", issued=T3, lid=None, current_privs=None):
    source = source or _source(base)
    cur = copy.deepcopy(source["authority"]); nxt = copy.deepcopy(cur)
    lid = lid or sorted(nxt)[0]
    current_privs = current_privs or {"log": dict(base["f"]["lpriv"]), "gossip": dict(base["f"]["gpriv"])}
    new_log = Ed25519PrivateKey.generate(); new_gossip = Ed25519PrivateKey.generate()
    nxt[lid]["publicKey"] = _pub(new_log); nxt[lid]["gossipPublicKey"] = _pub(new_gossip)
    new_privs = {"log": dict(current_privs["log"]), "gossip": dict(current_privs["gossip"])}
    new_privs["log"][lid] = new_log; new_privs["gossip"][lid] = new_gossip
    changed = [lid]
    if kind == "scheduled-rotation":
        compromised = []
        replaced = {auth._pub_fingerprint(cur[lid]["publicKey"]), auth._pub_fingerprint(cur[lid]["gossipPublicKey"])}
        role = "governance"; selected = ["gov-key-1", "gov-key-2"]; root_privs = base["gpriv"]
    else:
        compromised = [auth._pub_fingerprint(cur[lid]["publicKey"])]
        replaced = {auth._pub_fingerprint(cur[lid]["publicKey"]), auth._pub_fingerprint(cur[lid]["gossipPublicKey"])}
        role = "recovery"; selected = ["recovery-key-1", "recovery-key-2"]; root_privs = base["rpriv"]
    revoked = sorted(set(source["revoked"]) | replaced)
    signed = {
        "_type": "archive-merkle-authority-rebridge-transition", "specVersion": "1.0.0", "schemaVersion": 1,
        "transitionId": f"run167-{source['rebridgeSequence']+1}-{kind}-{lid}", "rebridgeSequence": source["rebridgeSequence"] + 1,
        "kind": kind, "issuedAt": rb._ts(issued), "sourceKind": source["sourceKind"],
        "sourceTrustedStateSha256": source["stateSha256"], "sourceMerkleSequence": source["merkleSequence"],
        "sourceMerkleContinuityHeadSha256": source["merkleHead"], "previousAuthorityBridgeHeadSha256": source["authorityHead"],
        "currentAuthority": cur, "nextAuthority": nxt, "changedLogIds": changed,
        "compromisedKeyFingerprints": compromised, "revokedKeyFingerprints": revoked,
        "authorizationRole": role, "selectedSignerKeyIds": selected, "handoffSubjectSha256s": {},
    }
    subjects = {x: rb._handoff_subject(signed=signed, log_id=x, checkpoint=source["checkpoints"][x]) for x in sorted(cur)}
    signed["handoffSubjectSha256s"] = {x: rb._sha_bytes(rb._canonical(subjects[x])) for x in sorted(subjects)}
    tr = {"signed": signed, "signatures": [{"keyId": k, "signature": _sig(root_privs[k], signed)} for k in selected]}
    proofs = []
    for x in sorted(cur):
        subject = subjects[x]; changed_here = x == lid
        proofs.append({
            "logId": x, "subjectSha256": rb._sha_bytes(rb._canonical(subject)),
            "newLogSignature": _sig(new_privs["log"][x], subject), "newGossipSignature": _sig(new_privs["gossip"][x], subject),
            "oldLogSignature": (_sig(current_privs["log"][x], subject) if kind == "scheduled-rotation" and changed_here else None),
            "oldGossipSignature": (_sig(current_privs["gossip"][x], subject) if kind == "scheduled-rotation" and changed_here else None),
        })
    hf = {"schemaVersion": 1, "transitionId": signed["transitionId"], "rebridgeSequence": signed["rebridgeSequence"], "proofs": proofs}
    return _write(tmp / "transition.json", tr), _write(tmp / "handoff.json", hf), nxt, new_privs, lid


def _adapters(base, authority, privs, *, leaves=None, now=T3, mut_gossip=None):
    leaves = leaves or base["leaves"]
    out = []
    for lid, cfg in authority.items():
        la = t164.LogAdapter(lid, cfg, privs["log"][lid], now); la.leaves = list(leaves[lid])
        ga = t164.GossipAdapter(lid, cfg, privs["gossip"][lid], now, mut=(mut_gossip if lid == sorted(authority)[-1] else None))
        out.append((lid, la, ga))
    return out


def _advance(base, tmp, *, previous=None, transition=True, kind="scheduled-rotation", source=None,
             current_privs=None, run161=None, wout=None, aout=None, now=T3, leaves=None, adapters=None):
    tmp.mkdir(parents=True, exist_ok=True)
    tp = hp = None
    if transition:
        tp, hp, authority, privs, lid = _transition(base, tmp, source=source, kind=kind, issued=now, current_privs=current_privs)
    else:
        source = source or _source(base, previous)
        authority = source["authority"]; privs = current_privs; lid = None
    ads = adapters or _adapters(base, authority, privs, leaves=leaves, now=now)
    out = tmp / "out"
    rb.advance_archive_merkle_rebridge(**_kwargs(base, out, run161=run161, wout=wout, aout=aout, now=now), adapters=ads,
                                       transition_path=tp, handoff_path=hp, previous_output_dir=previous)
    return out, ads, authority, privs, lid


def _leaves(adapters):
    return {lid: list(log.leaves) for lid, log, _ in adapters}


def test_run167_scheduled_rebridge_appends_from_exact_run166_checkpoint(base, tmp_path):
    out, ads, authority, privs, lid = _advance(base, tmp_path)
    res = rb.verify_archive_merkle_rebridge(**_kwargs(base, out))
    assert res["sequence"] == 3 and res["rebridge_sequence"] == 1
    state166 = json.loads((base["run166"] / "trusted-archive-merkle-continuity-state.json").read_text())
    receipt = json.loads((out / "release-archive-merkle-rebridge-receipt.json").read_text())
    for log_id, doc in receipt["events"][0]["logResponses"].items():
        assert doc["signed"]["previousTreeSize"] == state166["lastCheckpoints"][log_id]["treeSize"]
        assert doc["signed"]["previousRootHash"] == state166["lastCheckpoints"][log_id]["rootHash"]
        assert doc["signed"]["consistencyProof"]


def test_run167_scheduled_rotation_requires_old_and_new_handoff(base, tmp_path):
    tp, hp, authority, privs, lid = _transition(base, tmp_path)
    d = json.loads(hp.read_text()); row = next(x for x in d["proofs"] if x["logId"] == lid); row["oldLogSignature"] = None
    _write(hp, d)
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="OLD_HANDOFF_REQUIRED"):
        rb.advance_archive_merkle_rebridge(**_kwargs(base, tmp_path / "out"), adapters=_adapters(base, authority, privs), transition_path=tp, handoff_path=hp)


def test_run167_compromise_recovery_forbids_old_handoff_and_appends(base, tmp_path):
    out, ads, authority, privs, lid = _advance(base, tmp_path, kind="compromise-recovery")
    state = json.loads((out / "trusted-archive-merkle-rebridge-state.json").read_text())
    assert auth._pub_fingerprint(base["f"]["root"]["signed"]["logs"][lid]["publicKey"]) in state["revokedKeyFingerprints"]
    assert rb.verify_archive_merkle_rebridge(**_kwargs(base, out))["ok"]


def test_run167_append_only_epoch_under_rebridged_authority(base, tmp_path):
    out1, ads1, authority, privs, _ = _advance(base, tmp_path / "first")
    run161d, wout4, aout4 = t166._advance_run163(base["f"], tmp_path / "advance4", seq=4, run161_prev=base["run161c"], wout_prev=base["wout3"], aout_prev=base["aout3"], now=T4)
    source = _source(base, out1)
    leaves = _leaves(ads1)
    out2, ads2, _, _, _ = _advance(base, tmp_path / "second", previous=out1, transition=False, source=source,
                                    current_privs=privs, run161=run161d, wout=wout4, aout=aout4, now=T4, leaves=leaves)
    res = rb.verify_archive_merkle_rebridge(**_kwargs(base, out2, run161=run161d, wout=wout4, aout=aout4, now=T4))
    assert res["sequence"] == 4 and res["rebridge_sequence"] == 1
    bundle = json.loads((out2 / "release-archive-merkle-rebridge-bundle.json").read_text())
    assert [x["action"] for x in bundle["events"]] == ["rebridge", "append"]


def test_run167_second_rebridge_uses_latest_run167_checkpoint_not_run166(base, tmp_path):
    out1, ads1, authority1, privs1, _ = _advance(base, tmp_path / "first")
    run161d, wout4, aout4 = t166._advance_run163(base["f"], tmp_path / "advance4", seq=4, run161_prev=base["run161c"], wout_prev=base["wout3"], aout_prev=base["aout3"], now=T4)
    source = _source(base, out1); leaves = _leaves(ads1)
    out2, _, _, _, lid2 = _advance(
        base,
        tmp_path / "second",
        previous=out1,
        transition=True,
        source=source,
        current_privs=privs1,
        run161=run161d,
        wout=wout4,
        aout=aout4,
        now=T4,
        leaves=leaves,
    )
    rec = json.loads((out2 / "release-archive-merkle-rebridge-receipt.json").read_text())
    prior = json.loads((out1 / "trusted-archive-merkle-rebridge-state.json").read_text())["lastCheckpoints"]
    subject = rec["events"][1]["transitionDocument"]["signed"]
    assert subject["sourceKind"] == "run167" and subject["sourceMerkleSequence"] == 3
    row = rec["events"][1]["logResponses"][lid2]["signed"]
    assert row["previousTreeSize"] == prior[lid2]["treeSize"]


def test_run167_recovery_old_signature_is_rejected(base, tmp_path):
    tp, hp, authority, privs, lid = _transition(base, tmp_path, kind="compromise-recovery")
    d = json.loads(hp.read_text()); row = next(x for x in d["proofs"] if x["logId"] == lid)
    subject = rb._handoff_subject(signed=json.loads(tp.read_text())["signed"], log_id=lid, checkpoint=_source(base)["checkpoints"][lid])
    row["oldLogSignature"] = _sig(base["f"]["lpriv"][lid], subject); _write(hp, d)
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="OLD_HANDOFF_FORBIDDEN"):
        rb.advance_archive_merkle_rebridge(**_kwargs(base, tmp_path / "out"), adapters=_adapters(base, authority, privs), transition_path=tp, handoff_path=hp)


def test_run167_source_state_binding_rejects_stale_run166_state(base, tmp_path):
    tp, hp, authority, privs, _ = _transition(base, tmp_path)
    d = json.loads(tp.read_text()); d["signed"]["sourceTrustedStateSha256"] = "0" * 64
    d["signatures"] = [{"keyId": k, "signature": _sig(base["gpriv"][k], d["signed"])} for k in d["signed"]["selectedSignerKeyIds"]]
    _write(tp, d)
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="SOURCE_STATE"):
        rb.advance_archive_merkle_rebridge(**_kwargs(base, tmp_path / "out"), adapters=_adapters(base, authority, privs), transition_path=tp, handoff_path=hp)


def test_run167_same_run163_epoch_cannot_repeat(base, tmp_path):
    out1, ads, authority, privs, _ = _advance(base, tmp_path / "first")
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="SEQUENCE_INVALID"):
        _advance(base, tmp_path / "again", previous=out1, transition=False, source=_source(base, out1), current_privs=privs, leaves=_leaves(ads))


def test_run167_gossip_split_view_rejected(base, tmp_path):
    tp, hp, authority, privs, _ = _transition(base, tmp_path)
    def mut(s):
        s["checkpointSha256s"][sorted(s["checkpointSha256s"])[0]] = "0" * 64
    ads = _adapters(base, authority, privs, now=T3, mut_gossip=mut)
    with pytest.raises(rb.ArchiveMerkleRebridgeError):
        rb.advance_archive_merkle_rebridge(**_kwargs(base, tmp_path / "out"), adapters=ads, transition_path=tp, handoff_path=hp)


def test_run167_log_signature_mutation_detected_offline(base, tmp_path):
    out, _, _, _, _ = _advance(base, tmp_path)
    p = out / "release-archive-merkle-rebridge-receipt.json"; d = json.loads(p.read_text())
    d["events"][0]["logResponses"][sorted(d["events"][0]["logResponses"])[0]]["signature"] = "A" * 88; _write(p, d)
    with pytest.raises(rb.ArchiveMerkleRebridgeError):
        rb.verify_archive_merkle_rebridge(**_kwargs(base, out))


def test_run167_transition_id_reuse_rejected(base, tmp_path):
    out1, ads1, authority1, privs1, _ = _advance(base, tmp_path / "first")
    run161d, wout4, aout4 = t166._advance_run163(base["f"], tmp_path / "advance4", seq=4, run161_prev=base["run161c"], wout_prev=base["wout3"], aout_prev=base["aout3"], now=T4)
    source = _source(base, out1); tp, hp, authority2, privs2, _ = _transition(base, tmp_path / "second", source=source, issued=T4, current_privs=privs1)
    d = json.loads(tp.read_text()); old = json.loads((out1 / "release-archive-merkle-rebridge-receipt.json").read_text())["events"][0]["transitionDocument"]["signed"]["transitionId"]
    d["signed"]["transitionId"] = old
    subjects = {x: rb._handoff_subject(signed=d["signed"], log_id=x, checkpoint=source["checkpoints"][x]) for x in sorted(authority2)}
    d["signed"]["handoffSubjectSha256s"] = {x: rb._sha_bytes(rb._canonical(subjects[x])) for x in sorted(subjects)}
    d["signatures"] = [{"keyId": k, "signature": _sig(base["gpriv"][k], d["signed"])} for k in d["signed"]["selectedSignerKeyIds"]]; _write(tp, d)
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="ID_REUSED"):
        rb.advance_archive_merkle_rebridge(**_kwargs(base, tmp_path / "out", run161=run161d, wout=wout4, aout=aout4, now=T4), adapters=_adapters(base, authority2, privs2, leaves=_leaves(ads1), now=T4), transition_path=tp, handoff_path=hp, previous_output_dir=out1)


def test_run167_revoked_key_cannot_be_reintroduced(base, tmp_path):
    out1, ads1, authority1, privs1, lid = _advance(base, tmp_path / "first")
    source = _source(base, out1); cur = copy.deepcopy(source["authority"]); nxt = copy.deepcopy(cur)
    nxt[lid]["publicKey"] = base["f"]["root"]["signed"]["logs"][lid]["publicKey"]
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="REINTRODUCED"):
        # Directly exercise the transition verifier with a fully signed document assembled from the stale key.
        tp, hp, _, _, _ = _transition(base, tmp_path / "x", source=source, issued=T4, current_privs=privs1)
        d = json.loads(tp.read_text()); d["signed"]["nextAuthority"] = nxt
        subjects = {x: rb._handoff_subject(signed=d["signed"], log_id=x, checkpoint=source["checkpoints"][x]) for x in sorted(cur)}
        d["signed"]["handoffSubjectSha256s"] = {x: rb._sha_bytes(rb._canonical(subjects[x])) for x in sorted(subjects)}
        d["signatures"] = [{"keyId": k, "signature": _sig(base["gpriv"][k], d["signed"])} for k in d["signed"]["selectedSignerKeyIds"]]
        gov = rb._verify_control_root(base["gp"], base["gpin"], recovery=False, now=T4, historical=True)
        rec = rb._verify_control_root(base["rp"], base["rpin"], recovery=True, now=T4, historical=True)
        rb._verify_transition(d, source=source, governance=gov, recovery=rec, expected_log_ids=set(cur), seen_ids=set(), previous_transition_issued=source["lastTransitionIssued"], now=T4, creation=True)


def test_run167_duplicate_json_keys_rejected(tmp_path):
    p = tmp_path / "bad.json"; p.write_text('{"x":1,"x":2}\n')
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="DUPLICATE_KEY"):
        rb._read_json(p, "BAD")


def test_run167_input_drift_fails_before_commit(base, tmp_path, monkeypatch):
    tp, hp, authority, privs, _ = _transition(base, tmp_path)
    original = rb._verify_handoff
    def mutate(doc, *, transition):
        result = original(doc, transition=transition); tp.write_bytes(tp.read_bytes() + b" "); return result
    monkeypatch.setattr(rb, "_verify_handoff", mutate)
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="INPUT_DRIFT"):
        rb.advance_archive_merkle_rebridge(**_kwargs(base, tmp_path / "out"), adapters=_adapters(base, authority, privs), transition_path=tp, handoff_path=hp)
    assert not (tmp_path / "out").exists()


def test_run167_live_epoch_expires_but_historical_replay_survives(base, tmp_path):
    out, _, _, _, _ = _advance(base, tmp_path)
    far = T3 + timedelta(days=40)
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="STALE"):
        rb.verify_archive_merkle_rebridge(**_kwargs(base, out, now=far))
    assert rb.verify_archive_merkle_rebridge(**_kwargs(base, out, now=far, historical=True))["ok"]


def test_run167_output_is_canonical_and_allowlisted(base, tmp_path):
    out, _, _, _, _ = _advance(base, tmp_path)
    assert {p.name for p in out.iterdir()} == rb._OUTPUT_NAMES
    for p in out.iterdir():
        assert p.read_bytes() == rb._canonical(json.loads(p.read_bytes()))


def test_run167_documentation_mentions_recursive_rebridge_old_new_recovery_and_rfc6962():
    guide = (SEC / "RELEASE_ARCHIVE_MERKLE_REBRIDGE_GUIDE.md").read_text().lower()
    gates = (SEC / "SECURITY_RELEASE_GATES.md").read_text()
    for phrase in ("recursive", "old + new", "compromise recovery", "rfc6962", "run 166", "append-only"):
        assert phrase in guide
    assert "Run 167" in gates


def test_run167_first_event_cannot_be_append_only(base, tmp_path):
    source = _source(base)
    privs = {"log": dict(base["f"]["lpriv"]), "gossip": dict(base["f"]["gpriv"])}
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="FIRST_EVENT_REQUIRES_TRANSITION"):
        _advance(base, tmp_path, transition=False, source=source, current_privs=privs)


def test_run167_handoff_membership_is_exact(base, tmp_path):
    tp, hp, authority, privs, _ = _transition(base, tmp_path)
    d = json.loads(hp.read_text()); d["proofs"].pop(); _write(hp, d)
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="PROOF_SET"):
        rb.advance_archive_merkle_rebridge(**_kwargs(base, tmp_path / "out"), adapters=_adapters(base, authority, privs), transition_path=tp, handoff_path=hp)


def test_run167_transition_time_must_strictly_increase(base, tmp_path):
    out1, ads1, _, privs1, _ = _advance(base, tmp_path / "first")
    run161d, wout4, aout4 = t166._advance_run163(base["f"], tmp_path / "advance4", seq=4, run161_prev=base["run161c"], wout_prev=base["wout3"], aout_prev=base["aout3"], now=T4)
    source = _source(base, out1)
    tp, hp, authority2, privs2, _ = _transition(base, tmp_path / "second", source=source, issued=T3, current_privs=privs1)
    with pytest.raises(rb.ArchiveMerkleRebridgeError, match="TIME_NOT_MONOTONIC"):
        rb.advance_archive_merkle_rebridge(
            **_kwargs(base, tmp_path / "out2", run161=run161d, wout=wout4, aout=aout4, now=T4),
            adapters=_adapters(base, authority2, privs2, leaves=_leaves(ads1), now=T4), transition_path=tp, handoff_path=hp, previous_output_dir=out1)
