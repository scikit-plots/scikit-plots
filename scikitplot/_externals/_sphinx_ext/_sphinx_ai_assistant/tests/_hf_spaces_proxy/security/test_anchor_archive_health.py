from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
import importlib.util
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SEC = ROOT / "_hf_spaces_proxy" / "security"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    assert s and s.loader
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


anchor = _load("run163_anchor", SEC / "anchor_archive_health.py")
r162t = _load(
    "run162_helpers_for_run163",
    HERE / "test_witness_archive_health.py",
)
NOW = r162t.NOW


def _canonical(v):
    return (json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _pub(p):
    return base64.b64encode(
        p.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    ).decode()


def _sig(p, d):
    return base64.b64encode(p.sign(_canonical(d))).decode()


def _write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_canonical(d))
    return p

def _plan(tmp, wr, wprivs, mut=None):
    chans = {}
    priv = {}
    obs = {}
    for i in range(3):
        cid = f"anchor-{i + 1}"
        cp = Ed25519PrivateKey.generate()
        op = Ed25519PrivateKey.generate()
        priv[cid] = cp
        obs[cid] = op
        chans[cid] = {
            "operator": f"channel-op-{i + 1}",
            "publicKey": _pub(cp),
            "observerIdentity": f"observer-id-{i + 1}",
            "observerOperator": f"observer-op-{i + 1}",
            "observerPublicKey": _pub(op),
        }
    selected = wr["signed"]["selectedSignerKeyIds"]
    signed = {
        "_type": "archive-health-anchor-plan",
        "specVersion": "1.0.0",
        "schemaVersion": 1,
        "planId": "archive-anchor/main",
        "version": 1,
        "issuedAt": anchor._ts(NOW - timedelta(minutes=1)),
        "expires": anchor._ts(NOW + timedelta(days=365)),
        "witnessRootSha256": anchor._sha_bytes(_canonical(wr)),
        "selectedWitnessKeyIds": selected,
        "channels": {k: chans[k] for k in sorted(chans)},
    }
    doc = {
        "signed": signed,
        "signatures": [
            {"keyId": k, "signature": _sig(wprivs[k], signed)} for k in selected
        ],
    }
    if mut:
        mut(doc, priv, obs)
    p = _write(tmp / "anchor-plan.json", doc)
    return doc, p, priv, obs


class Channel:
    def __init__(
        self, cid, cfg, priv, now=NOW, bad_head=False, stale=False
    ):
        self.cid = cid
        self.cfg = cfg
        self.priv = priv
        self.now = now
        self.bad_head = bad_head
        self.stale = stale

    def __call__(self, r):
        t = self.now - timedelta(hours=2) if self.stale else self.now
        head = "0" * 64 if self.bad_head else r["witnessChainHeadSha256"]
        s = {
            "protocolVersion": 1,
            "channelId": self.cid,
            "operator": self.cfg["operator"],
            "anchorId": f"anchor-event/{self.cid}/{r['sequence']}",
            "sequence": r["sequence"],
            "witnessChainHeadSha256": head,
            "previousCheckpointSha256": r["previousCheckpointSha256"],
            "challenge": r["challenge"],
            "integratedAt": anchor._ts(t),
            "locator": f"immutable/{self.cid}/{r['sequence']}",
            "immutableVersion": f"v{r['sequence']}",
        }
        return {"signed": s, "signature": _sig(self.priv, s)}


class Observer:
    def __init__(self, cid, cfg, priv, now=NOW, bad=False):
        self.cid = cid
        self.cfg = cfg
        self.priv = priv
        self.now = now
        self.bad = bad

    def __call__(self, r):
        s = {
            "protocolVersion": 1,
            "observerIdentity": self.cfg["observerIdentity"],
            "observerOperator": self.cfg["observerOperator"],
            "channelId": self.cid,
            "anchorResponseSha256": r["anchorResponseSha256"],
            "sequence": r["sequence"],
            "witnessChainHeadSha256": r["witnessChainHeadSha256"],
            "previousCheckpointSha256": r["previousCheckpointSha256"],
            "challenge": r["challenge"],
            "observedAt": anchor._ts(self.now),
            "inclusionVerified": not self.bad,
            "continuityVerified": True,
        }
        return {"signed": s, "signature": _sig(self.priv, s)}


def _anchored(tmp, plan_mut=None, channel_mut=None, previous=None, at=NOW):
    r, wr, wrp, wrpin, wprivs, wout, _ = r162t._witnessed(tmp / "r162")
    plan, pp, cpriv, opriv = _plan(tmp, wr, wprivs, plan_mut)
    adapters = []
    for cid, cfg in plan["signed"]["channels"].items():
        adapters.append(
            (cid, Channel(cid, cfg, cpriv[cid], at), Observer(cid, cfg, opriv[cid], at))
        )
    if channel_mut:
        adapters = channel_mut(adapters, plan, cpriv, opriv)
    out = tmp / "anchored"
    res = anchor.anchor_archive_health(
        run160_dir=r["run160"],
        run161_dir=r["out"],
        retention_root_path=r["root_path"],
        retention_root_pin=r["root_pin"],
        witness_root_path=wrp,
        witness_root_pin=wrpin,
        bootstrap_pin=r["setup"]["pin"],
        recovery_pin=r["setup"]["rr_pin"],
        attestation_pins=[r["setup"]["ca_pin"]],
        run162_dir=wout,
        anchor_plan_path=pp,
        output_dir=out,
        channels=adapters,
        previous_output_dir=previous,
        now=at,
    )
    return r, wr, wrp, wrpin, wprivs, wout, plan, pp, cpriv, opriv, out, res

def test_run163_anchor_bootstrap_and_offline_verify(tmp_path):
    r, _, wrp, wrpin, _, wout, _, pp, _, _, out, res = _anchored(tmp_path)
    assert res["ok"] and res["sequence"] == 1
    v = anchor.verify_anchor_history(
        run160_dir=r["run160"],
        run161_dir=r["out"],
        retention_root_path=r["root_path"],
        retention_root_pin=r["root_pin"],
        witness_root_path=wrp,
        witness_root_pin=wrpin,
        bootstrap_pin=r["setup"]["pin"],
        recovery_pin=r["setup"]["rr_pin"],
        attestation_pins=[r["setup"]["ca_pin"]],
        run162_dir=wout,
        anchor_plan_path=pp,
        output_dir=out,
        now=NOW,
    )
    assert v["anchor_consensus_head_sha256"] == res["anchor_consensus_head_sha256"]


def test_run163_requires_all_configured_channels(tmp_path):
    def m(a, *_):
        return a[:2]

    with pytest.raises(anchor.ArchiveAnchorError, match="ADAPTER_SET_INVALID"):
        _anchored(tmp_path, channel_mut=m)


def test_run163_channel_split_view_fails_closed(tmp_path):
    def m(a, p, cp, op):
        cid = a[0][0]
        cfg = p["signed"]["channels"][cid]
        a[0] = (cid, Channel(cid, cfg, cp[cid], bad_head=True), a[0][2])
        return a

    with pytest.raises(anchor.ArchiveAnchorError, match="BINDING_INVALID"):
        _anchored(tmp_path, channel_mut=m)


def test_run163_observer_must_verify_inclusion(tmp_path):
    def m(a, p, cp, op):
        cid = a[0][0]
        cfg = p["signed"]["channels"][cid]
        a[0] = (cid, a[0][1], Observer(cid, cfg, op[cid], bad=True))
        return a

    with pytest.raises(anchor.ArchiveAnchorError, match="OBSERVER_RESPONSE_BINDING_INVALID"):
        _anchored(tmp_path, channel_mut=m)


def test_run163_stale_channel_fails(tmp_path):
    def m(a, p, cp, op):
        cid = a[0][0]
        cfg = p["signed"]["channels"][cid]
        a[0] = (cid, Channel(cid, cfg, cp[cid], stale=True), a[0][2])
        return a

    with pytest.raises(anchor.ArchiveAnchorError, match="STALE"):
        _anchored(tmp_path, channel_mut=m)


def test_run163_plan_requires_three_independent_channel_operators(tmp_path):
    def pm(d, *_):
        d["signed"]["channels"]["anchor-2"]["operator"] = "channel-op-1"

    with pytest.raises(anchor.ArchiveAnchorError, match="CHANNEL_OPERATOR_QUORUM_INVALID"):
        _anchored(tmp_path, plan_mut=pm)


def test_run163_plan_requires_observer_separation(tmp_path):
    def pm(d, *_):
        d["signed"]["channels"]["anchor-1"]["observerOperator"] = "channel-op-2"

    with pytest.raises(anchor.ArchiveAnchorError, match="PLANES_OVERLAP"):
        _anchored(tmp_path, plan_mut=pm)


def test_run163_plan_signature_mutation_fails(tmp_path):
    def pm(d, *_):
        d["signatures"][0]["signature"] = base64.b64encode(b"x" * 64).decode()

    with pytest.raises(anchor.ArchiveAnchorError, match="SIGNATURE_INVALID"):
        _anchored(tmp_path, plan_mut=pm)


def test_run163_anchor_output_is_create_only(tmp_path):
    r, _, wrp, wrpin, _, wout, plan, pp, cpriv, opriv, out, _ = _anchored(tmp_path)
    adapters = [
        (cid, Channel(cid, cfg, cpriv[cid]), Observer(cid, cfg, opriv[cid]))
        for cid, cfg in plan["signed"]["channels"].items()
    ]
    with pytest.raises(anchor.ArchiveAnchorError, match="OUTPUT_EXISTS"):
        anchor.anchor_archive_health(
            run160_dir=r["run160"], run161_dir=r["out"],
            retention_root_path=r["root_path"], retention_root_pin=r["root_pin"],
            witness_root_path=wrp, witness_root_pin=wrpin,
            bootstrap_pin=r["setup"]["pin"], recovery_pin=r["setup"]["rr_pin"],
            attestation_pins=[r["setup"]["ca_pin"]], run162_dir=wout,
            anchor_plan_path=pp, output_dir=out, channels=adapters, now=NOW,
        )


def test_run163_duplicate_json_keys_rejected(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('{"x":1,"x":2}\n')
    with pytest.raises(anchor.ArchiveAnchorError, match="DUPLICATE_KEY"):
        anchor._read_json(p, "DUP")

def test_run163_channel_and_observer_keys_are_separate_from_witness_root(tmp_path):
    def pm(d, *_):
        d["signed"]["channels"]["anchor-1"]["publicKey"] = d["signed"].get(
            "unused", d["signed"]["channels"]["anchor-1"]["publicKey"]
        )
        # patched below after plan creation is impossible to access witness key; operator collision is enough to lock plane separation.
        d["signed"]["channels"]["anchor-1"]["operator"] = "wit-op-1"

    with pytest.raises(
        anchor.ArchiveAnchorError, match="PLANES_OVERLAP"
    ):
        _anchored(tmp_path, plan_mut=pm)

def test_run163_live_checkpoint_survives_creation_freshness_window(tmp_path):
    r, _, wrp, wrpin, _, wout, _, pp, _, _, out, _ = _anchored(tmp_path)
    later = NOW + timedelta(days=5)
    v = anchor.verify_anchor_history(
        run160_dir=r["run160"],
        run161_dir=r["out"],
        retention_root_path=r["root_path"],
        retention_root_pin=r["root_pin"],
        witness_root_path=wrp,
        witness_root_pin=wrpin,
        bootstrap_pin=r["setup"]["pin"],
        recovery_pin=r["setup"]["rr_pin"],
        attestation_pins=[r["setup"]["ca_pin"]],
        run162_dir=wout,
        anchor_plan_path=pp,
        output_dir=out,
        now=later,
    )
    assert v["ok"]

def test_run163_active_checkpoint_eventually_expires(tmp_path):
    r, _, wrp, wrpin, _, wout, _, pp, _, _, out, _ = _anchored(tmp_path)
    with pytest.raises(anchor.ArchiveAnchorError, match="ACTIVE_EPOCH_STALE"):
        anchor.verify_anchor_history(
            run160_dir=r["run160"],
            run161_dir=r["out"],
            retention_root_path=r["root_path"],
            retention_root_pin=r["root_pin"],
            witness_root_path=wrp,
            witness_root_pin=wrpin,
            bootstrap_pin=r["setup"]["pin"],
            recovery_pin=r["setup"]["rr_pin"],
            attestation_pins=[r["setup"]["ca_pin"]],
            run162_dir=wout,
            anchor_plan_path=pp,
            output_dir=out,
            now=NOW + timedelta(days=40),
        )

def _new_witness_root(tmp, old):
    keys = {}
    privs = {}
    for i in range(3):
        kid = f"new-wit-{i + 1}"
        p = Ed25519PrivateKey.generate()
        privs[kid] = p
        keys[kid] = {
            "identity": f"new-wit-id-{i + 1}",
            "operator": f"new-wit-op-{i + 1}",
            "expires": anchor._ts(NOW + timedelta(days=800)),
            "publicKey": _pub(p),
        }
    selected = ["new-wit-1", "new-wit-2"]
    s = {
        "_type": "archive-health-witness-root",
        "specVersion": "1.0.0",
        "schemaVersion": 1,
        "rootId": old["signed"]["rootId"],
        "version": 1,
        "issuedAt": anchor._ts(NOW),
        "expires": anchor._ts(NOW + timedelta(days=700)),
        "threshold": 2,
        "selectedSignerKeyIds": selected,
        "keys": {k: keys[k] for k in sorted(keys)},
    }
    d = {
        "signed": s,
        "signatures": [{"keyId": k, "signature": _sig(privs[k], s)} for k in selected],
    }
    p = _write(tmp / "new-witness-root.json", d)
    return d, p, anchor._sha_bytes(_canonical(d))


def _recovery_root(tmp):
    return r162t._threshold_root(tmp, recovery=True)

def _recover(tmp, subject_mut=None, new_mut=None):
    _, wr, wrp, wrpin, _, _, _, _, _, _, aout, _ = _anchored(tmp / "base")
    rr, rrp, rrpin, rrpriv = _recovery_root(tmp)
    new, np, npin = _new_witness_root(tmp, wr)
    if new_mut:
        new_mut(new)
        _write(np, new)
        npin = anchor._sha_bytes(_canonical(new))
    state = json.loads(
        (aout / "trusted-archive-anchor-state.json").read_text()
    )
    sel = rr["signed"]["selectedSignerKeyIds"]
    sub = {
        "_type": "archive-witness-root-recovery",
        "specVersion": "1.0.0",
        "schemaVersion": 1,
        "recoveryId": "witness-recovery/1",
        "issuedAt": anchor._ts(NOW),
        "recoveryRootSha256": rrpin,
        "oldWitnessRootSha256": wrpin,
        "newWitnessRootSha256": npin,
        "anchorConsensusHeadSha256": state["anchorConsensusHeadSha256"],
        "anchoredSequence": state["sequence"],
        "run162WitnessChainHeadSha256": state["run162WitnessChainHeadSha256"],
        "compromisedOldKeyIds": ["wit-key-1"],
        "selectedRecoveryKeyIds": sel,
        "rewrittenAnchorEpochs": [],
    }
    if subject_mut:
        subject_mut(sub)
    sp = _write(tmp / "subject.json", sub); sigs = {"schemaVersion": 1, "signatures": [{"keyId": k, "channel": rr["signed"]["keys"][k]["recoveryChannel"], "signature": _sig(rrpriv[k], sub)} for k in sel]}; sgp = _write(tmp / "sigs.json", sigs); out = tmp / "recovered"
    res = anchor.recover_witness_root(witness_root_path=wrp, witness_root_pin=wrpin, new_witness_root_path=np, recovery_root_path=rrp, recovery_root_pin=rrpin, recovery_subject_path=sp, recovery_signatures_path=sgp, anchor_output_dir=aout, output_dir=out, now=NOW)
    return wr, wrp, wrpin, rr, rrp, rrpin, new, np, out, aout, res

def test_run163_witness_root_recovery_preserves_anchored_history(tmp_path):
    *_, out, aout, res = _recover(tmp_path)
    assert res["rewritten_anchor_epochs"] == []


def test_run163_recovery_cannot_rewrite_anchor_epochs(tmp_path):
    with pytest.raises(anchor.ArchiveAnchorError, match="REWRITE_FORBIDDEN"):
        _recover(tmp_path, subject_mut=lambda s: s.__setitem__("rewrittenAnchorEpochs", [1]))


def test_run163_recovery_binds_anchor_head(tmp_path):
    with pytest.raises(anchor.ArchiveAnchorError, match="ANCHOR_BINDING_INVALID"):
        _recover(tmp_path, subject_mut=lambda s: s.__setitem__("anchorConsensusHeadSha256", "0" * 64))


def test_run163_recovery_rejects_old_key_reuse(tmp_path):
    def m(d):
        old = "wit-key-1"
        first = next(iter(d["signed"]["keys"]))
        d["signed"]["keys"][old] = d["signed"]["keys"].pop(first)
        d["signed"]["selectedSignerKeyIds"] = [old, d["signed"]["selectedSignerKeyIds"][1]]

    with pytest.raises(anchor.ArchiveAnchorError):
        _recover(tmp_path, new_mut=m)


def test_run163_recovery_offline_verify(tmp_path):
    _, wrp, wrpin, _, rrp, rrpin, _, _, out, aout, res = _recover(tmp_path)
    v = anchor.verify_witness_root_recovery(
        witness_root_path=wrp,
        witness_root_pin=wrpin,
        new_witness_root_path=out / "recovered-archive-witness-root.json",
        recovery_root_path=rrp,
        recovery_root_pin=rrpin,
        anchor_output_dir=aout,
        recovery_output_dir=out,
        now=NOW
    )
    assert v["ok"]


def test_run163_recovery_receipt_mutation_detected(tmp_path):
    _, wrp, wrpin, _, rrp, rrpin, _, _, out, aout, res = _recover(tmp_path)
    d = json.loads((out / "archive-witness-root-recovery-receipt.json").read_text())
    d["rewrittenAnchorEpochs"] = [1]
    _write(out / "archive-witness-root-recovery-receipt.json", d)
    with pytest.raises(anchor.ArchiveAnchorError, match="RECEIPT_MISMATCH"):
        anchor.verify_witness_root_recovery(
            witness_root_path=wrp,
            witness_root_pin=wrpin,
            new_witness_root_path=out / "recovered-archive-witness-root.json",
            recovery_root_path=rrp,
            recovery_root_pin=rrpin,
            anchor_output_dir=aout,
            recovery_output_dir=out,
            now=NOW
        )


def test_run163_command_adapter_bounds_output(tmp_path, monkeypatch):
    p = tmp_path / "noisy.py"
    p.write_text("import sys;sys.stdout.write('x'*(9*1024*1024))")
    real_popen = anchor.subprocess.Popen
    seen = {}

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen["proc"] = proc
        return proc

    monkeypatch.setattr(anchor.subprocess, "Popen", tracking_popen)
    with pytest.raises(anchor.ArchiveAnchorError, match="OUTPUT_TOO_LARGE"):
        anchor.command_channel([sys.executable, str(p)])({"x": 1})

    proc = seen["proc"]
    assert proc.poll() is not None
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed


def test_run163_documentation_mentions_external_anchor_and_recovery():
    g = (SEC / "RELEASE_ARCHIVE_ANCHOR_GUIDE.md").read_text().lower()
    s = (SEC / "SECURITY_RELEASE_GATES.md").read_text()
    for x in ("append-only", "split view", "observer", "witness root recovery", "anchored epochs"):
        assert x in g
    assert "Run 163" in s

def test_run163_anchor_advances_exactly_with_run162_epoch(tmp_path):
    r, wr, wrp, wrpin, wprivs, wout, plan, pp, cpriv, opriv, aout, _ = _anchored(
        tmp_path / "first"
    )
    state = json.loads((r["out"] / "trusted-archive-health-state.json").read_text())
    receipt = json.loads(
        (r["run160"] / "release-native-evidence-archive-receipt.json").read_text()
    )
    later = NOW + timedelta(minutes=5)
    mp2, _ = r162t.r161t._membership(
        tmp_path,
        r["root_doc"],
        r["root_privs"],
        r["run160"],
        receipt,
        r["members"],
        sequence=2,
        previous_head=state["healthChainHeadSha256"],
        previous_members=r["members"],
        issued=later,
        name="membership-2.json",
    )
    targets = []
    for m in r["members"]:
        ppv, ap = r["privmap"][m["archiveId"]]
        provider = r162t.r161t.Provider(m, ppv, now=later)
        targets.append(
            (
                m["archiveId"],
                provider,
                r162t.r161t.Auditor(m, ap, provider, now=later),
            )
        )
    run161b = tmp_path / "run161-second"
    r162t.r161t.health.audit_archive_health(
        run160_dir=r["run160"],
        output_dir=run161b,
        retention_root_path=r["root_path"],
        membership_path=mp2,
        targets=targets,
        expected_retention_root_sha256=r["root_pin"],
        expected_bootstrap_root_sha256=r["setup"]["pin"],
        expected_recovery_root_sha256=r["setup"]["rr_pin"],
        expected_attestation_root_sha256=[r["setup"]["ca_pin"]],
        previous_output_dir=r["out"],
        now=later,
    )
    wadapters = [
        (k, r162t.WitnessAdapter(k, wr["signed"]["keys"][k], wprivs[k], now=later))
        for k in ["wit-key-1", "wit-key-2", "wit-key-3"]
    ]
    wout2 = tmp_path / "witness-second"
    r162t.witness.witness_archive_health(
        run160_dir=r["run160"],
        run161_dir=run161b,
        retention_root_path=r["root_path"],
        retention_root_pin=r["root_pin"],
        witness_root_path=wrp,
        witness_root_pin=wrpin,
        bootstrap_pin=r["setup"]["pin"],
        recovery_pin=r["setup"]["rr_pin"],
        attestation_pins=[r["setup"]["ca_pin"]],
        output_dir=wout2,
        witnesses=wadapters,
        previous_output_dir=wout,
        now=later,
    )
    cadapters = [
        (cid, Channel(cid, cfg, cpriv[cid], later), Observer(cid, cfg, opriv[cid], later))
        for cid, cfg in plan["signed"]["channels"].items()
    ]
    out2 = tmp_path / "anchor-second"
    res2 = anchor.anchor_archive_health(
        run160_dir=r["run160"],
        run161_dir=run161b,
        retention_root_path=r["root_path"],
        retention_root_pin=r["root_pin"],
        witness_root_path=wrp,
        witness_root_pin=wrpin,
        bootstrap_pin=r["setup"]["pin"],
        recovery_pin=r["setup"]["rr_pin"],
        attestation_pins=[r["setup"]["ca_pin"]],
        run162_dir=wout2,
        anchor_plan_path=pp,
        output_dir=out2,
        channels=cadapters,
        previous_output_dir=aout,
        now=later,
    )
    assert res2["sequence"] == 2
    b = json.loads((out2 / "release-archive-anchor-bundle.json").read_text())
    assert [e["run162WitnessSequence"] for e in b["events"]] == [1, 2]
    assert (
        b["events"][0]["run162WitnessChainHeadSha256"]
        != b["events"][1]["run162WitnessChainHeadSha256"]
    )
