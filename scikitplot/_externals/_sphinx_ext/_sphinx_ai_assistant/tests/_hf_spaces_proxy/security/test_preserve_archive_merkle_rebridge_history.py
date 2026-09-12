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

TESTS = Path(__file__).resolve().parent
SEC = RUNTIME_ROOT / "_hf_spaces_proxy" / "security"
sys.path.insert(0, str(TESTS)); sys.path.insert(0, str(SEC))

import test_continue_archive_merkle_authority as t166
import test_rebridge_archive_merkle_authority as t167
import preserve_archive_merkle_rebridge_history as rec

NOW = t167.T5


def _write(path: Path, obj):
    path.write_bytes(rec._canonical(obj)); return path


class ArchiveAdapter:
    def __init__(self, identity, operator, *, now=NOW, status="created", mutate=None):
        self.identity = identity; self.operator = operator; self.now = now; self.status = status; self.mutate = mutate; self.requests = []

    def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        value = {
            "schemaVersion": 1, "operation": "preserve", "status": self.status,
            "archiveId": request["archiveId"], "archiveIdentity": self.identity, "archiveOperator": self.operator,
            "artifact": request["artifact"], "locator": f"s3://{self.identity}/run168/checkpoint.json",
            "immutableVersionId": f"version-{self.identity}", "immutabilityClass": "object-lock", "overwrite": False,
            "readBackSha256": request["artifact"]["sha256"], "readBackSize": request["artifact"]["size"],
            "observedAt": rec.run167._ts(self.now),
        }
        if self.mutate:
            self.mutate(value, request)
        return value


class VerifierAdapter:
    def __init__(self, identity, operator, *, now=NOW, mutate=None):
        self.identity = identity; self.operator = operator; self.now = now; self.mutate = mutate; self.requests = []

    def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        value = {
            "schemaVersion": 1, "status": "verified", "verifierIdentity": self.identity, "verifierOperator": self.operator,
            "archiveId": request["archiveId"], "archiveIdentity": request["archiveIdentity"], "archiveOperator": request["archiveOperator"],
            "artifact": request["artifact"], "locator": request["locator"], "immutableVersionId": request["immutableVersionId"],
            "readBackSha256": request["artifact"]["sha256"], "readBackSize": request["artifact"]["size"],
            "readOnly": True, "archiveCredentialsReused": False, "observedAt": rec.run167._ts(self.now),
        }
        if self.mutate:
            self.mutate(value, request)
        return value


class RecoveryAdapter:
    def __init__(self, identity, operator, raw: bytes | None, *, now=NOW, locator=None, mutate=None):
        self.identity = identity; self.operator = operator; self.raw = raw; self.now = now; self.locator = locator or f"s3://{identity}/run168/checkpoint.json"; self.mutate = mutate

    def __call__(self, request):
        if self.raw is None:
            value = {"schemaVersion": 1, "status": "unavailable", "sourceIdentity": self.identity, "sourceOperator": self.operator,
                     "observedAt": rec.run167._ts(self.now), "readOnly": True, "writerCredentialsReused": False, "reason": "offline"}
        else:
            value = {"schemaVersion": 1, "status": "observed", "sourceIdentity": self.identity, "sourceOperator": self.operator,
                     "observedAt": rec.run167._ts(self.now), "readOnly": True, "writerCredentialsReused": False,
                     "locator": self.locator, "artifact": rec._artifact("release-archive-merkle-recovery-checkpoint.json", self.raw),
                     "payloadBase64": base64.b64encode(self.raw).decode("ascii")}
        if self.mutate:
            self.mutate(value, request)
        return value


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run168-base")
    b = t167.base.__wrapped__(tmp_path_factory) if hasattr(t167.base, "__wrapped__") else None
    # pytest fixture functions are wrapped; calling __wrapped__ gives us the same module-scoped predecessor once.
    if b is None:
        raise RuntimeError("run167 fixture unavailable")
    out1, ads1, authority1, privs1, _ = t167._advance(b, tmp / "r167-1")
    run161d, wout4, aout4 = t166._advance_run163(b["f"], tmp / "advance4", seq=4, run161_prev=b["run161c"], wout_prev=b["wout3"], aout_prev=b["aout3"], now=t167.T4)
    source1 = t167._source(b, out1); leaves1 = t167._leaves(ads1)
    out2, ads2, _, _, _ = t167._advance(b, tmp / "r167-2", previous=out1, transition=False, source=source1, current_privs=privs1,
                                         run161=run161d, wout=wout4, aout=aout4, now=t167.T4, leaves=leaves1)
    run161e, wout5, aout5 = t166._advance_run163(b["f"], tmp / "advance5", seq=5, run161_prev=run161d, wout_prev=wout4, aout_prev=aout4, now=t167.T5)
    source2 = t167._source(b, out2); leaves2 = t167._leaves(ads2)
    out3, ads3, authority3, privs3, lid3 = t167._advance(b, tmp / "r167-3", previous=out2, transition=True, kind="compromise-recovery",
                                                         source=source2, current_privs=privs1, run161=run161e, wout=wout5, aout=aout5,
                                                         now=t167.T5, leaves=leaves2)
    kwargs = t167._kwargs(b, out3, run161=run161e, wout=wout5, aout=aout5, now=t167.T5)
    kwargs.pop("output_dir", None); kwargs.pop("now", None)
    return {"tmp": tmp, "b": b, "run167": out3, "kwargs": kwargs, "authority": authority3, "privs": privs3, "changed": lid3}


def _targets(*, now=NOW, archive_mut=None, verifier_mut=None, statuses=None):
    statuses = statuses or ["created", "created", "created"]
    rows = []
    for i, suffix in enumerate(("a", "b", "c")):
        am = archive_mut if i == 0 else None; vm = verifier_mut if i == 0 else None
        rows.append((f"archive-{suffix}", f"archive-op-{suffix}", ArchiveAdapter(f"archive-{suffix}", f"archive-op-{suffix}", now=now, status=statuses[i], mutate=am),
                     f"verifier-{suffix}", f"verifier-op-{suffix}", VerifierAdapter(f"verifier-{suffix}", f"verifier-op-{suffix}", now=now, mutate=vm)))
    return rows


def _preserve(base, target: Path, **kwargs):
    return rec.preserve_rebridge_history(run167_dir=base["run167"], output_dir=target, verify_kwargs=base["kwargs"], targets=_targets(**kwargs), now=NOW)


def _checkpoint(out: Path):
    raw = (out / "release-archive-merkle-recovery-checkpoint.json").read_bytes()
    return json.loads(raw), raw


def _pins(out: Path):
    cp, raw = _checkpoint(out); s = cp["summary"]
    return dict(expected_checkpoint_sha256=rec._sha_bytes(raw), expected_authority_head_sha256=s["authorityHead"], expected_merkle_head_sha256=s["merkleHead"],
                expected_active_authority_sha256=s["activeAuthoritySha256"], expected_sequence=s["sequence"])


def test_run168_preserves_three_generation_run167_history(base, tmp_path):
    out = tmp_path / "out"; result = _preserve(base, out)
    assert result["ok"] and result["archives"] == 3
    cp = json.loads((out / "release-archive-merkle-recovery-checkpoint.json").read_text())
    assert [x["action"] for x in cp["run167Documents"]["release-archive-merkle-rebridge-bundle.json"]["events"]] == ["rebridge", "append", "rebridge"]
    assert cp["summary"]["rebridgeSequence"] == 2
    assert base["changed"] is not None


def test_run168_compact_summary_reconstructs_active_authority_and_revocations(base, tmp_path):
    out = tmp_path / "out"; _preserve(base, out)
    cp = json.loads((out / "release-archive-merkle-recovery-checkpoint.json").read_text())
    state = json.loads((base["run167"] / "trusted-archive-merkle-rebridge-state.json").read_text())
    assert cp["summary"]["activeAuthority"] == state["activeAuthority"]
    assert cp["summary"]["revokedKeyFingerprints"] == state["revokedKeyFingerprints"]
    assert cp["summary"]["lastCheckpoints"] == state["lastCheckpoints"]
    assert cp["summary"]["authorityHead"] == state["archiveMerkleRebridgeAuthorityHeadSha256"]


def test_run168_verify_preserved_output(base, tmp_path):
    out = tmp_path / "out"; _preserve(base, out)
    result = rec.verify_recovery_archive(output_dir=out, now=NOW)
    assert result["ok"] and result["sequence"] == 5 and result["rebridge_sequence"] == 2


def test_run168_create_only_present_retry_produces_same_checkpoint(base, tmp_path):
    out1 = tmp_path / "a"; out2 = tmp_path / "b"
    _preserve(base, out1, statuses=["created", "created", "created"])
    _preserve(base, out2, statuses=["present", "present", "present"])
    assert (out1 / "release-archive-merkle-recovery-checkpoint.json").read_bytes() == (out2 / "release-archive-merkle-recovery-checkpoint.json").read_bytes()
    assert (out1 / "trusted-archive-merkle-recovery-state.json").read_bytes() == (out2 / "trusted-archive-merkle-recovery-state.json").read_bytes()


def test_run168_requires_three_configured_archives(base, tmp_path):
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="TARGET_PLAN_TOO_SMALL"):
        rec.preserve_rebridge_history(run167_dir=base["run167"], output_dir=tmp_path / "out", verify_kwargs=base["kwargs"], targets=_targets()[:2], now=NOW)


def test_run168_rejects_archive_verifier_operator_overlap(base, tmp_path):
    rows = _targets(); ai, ao, aa, vi, vo, va = rows[0]
    rows[0] = (ai, ao, aa, vi, ao, va)
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="AUTHORITY_PLANE_OVERLAP"):
        rec.preserve_rebridge_history(run167_dir=base["run167"], output_dir=tmp_path / "out", verify_kwargs=base["kwargs"], targets=rows, now=NOW)


def test_run168_archive_hash_mismatch_rejected(base, tmp_path):
    def mut(v, req): v["readBackSha256"] = "0" * 64
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="ARCHIVE_READBACK_INVALID"):
        _preserve(base, tmp_path / "out", archive_mut=mut)


def test_run168_verifier_credential_reuse_rejected(base, tmp_path):
    def mut(v, req): v["archiveCredentialsReused"] = True
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="VERIFIER_AUTHORITY_INVALID"):
        _preserve(base, tmp_path / "out", verifier_mut=mut)


def test_run168_locator_collision_rejected(base, tmp_path):
    rows = _targets()
    common = "s3://shared/run168/checkpoint.json"
    for row in rows:
        adapter = row[2]
        adapter.mutate = lambda v, req, c=common: v.__setitem__("locator", c)
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="LOCATOR_COLLISION"):
        rec.preserve_rebridge_history(run167_dir=base["run167"], output_dir=tmp_path / "out", verify_kwargs=base["kwargs"], targets=rows, now=NOW)


def test_run168_run167_mutation_detected_offline(base, tmp_path):
    out = tmp_path / "out"; _preserve(base, out)
    p = out / "release-archive-merkle-recovery-checkpoint.json"; d = json.loads(p.read_text())
    d["run167Documents"]["trusted-archive-merkle-rebridge-state.json"]["sequence"] += 1
    # Recompute outer head to ensure the inner Run 167 checks, not only the outer hash, catch it.
    body = {k: d[k] for k in d if k != "recoveryCheckpointHeadSha256"}; d["recoveryCheckpointHeadSha256"] = rec._checkpoint_head(body); _write(p, d)
    with pytest.raises(rec.ArchiveMerkleRecoveryError):
        rec.verify_recovery_archive(output_dir=out, now=NOW)


def test_run168_recovery_two_of_three_with_one_unavailable(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); cp, raw = _checkpoint(out)
    sources = [("source-a", "source-op-a", RecoveryAdapter("source-a", "source-op-a", raw)),
               ("source-b", "source-op-b", RecoveryAdapter("source-b", "source-op-b", raw)),
               ("source-c", "source-op-c", RecoveryAdapter("source-c", "source-op-c", None))]
    recovered = tmp_path / "recovered"
    result = rec.recover_rebridge_history(sources=sources, output_dir=recovered, now=NOW, **_pins(out))
    assert result["observed"] == 2 and result["unavailable"] == 1
    for name in rec._RUN167_NAMES:
        assert (recovered / "recovered-run167" / name).read_bytes() == (base["run167"] / name).read_bytes()


def test_run168_recovered_active_projection_is_self_contained(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); _, raw = _checkpoint(out)
    src = [(f"source-{x}", f"source-op-{x}", RecoveryAdapter(f"source-{x}", f"source-op-{x}", raw)) for x in "abc"]
    recovered = tmp_path / "recovered"; rec.recover_rebridge_history(sources=src, output_dir=recovered, now=NOW, **_pins(out))
    active = json.loads((recovered / "recovered-active-archive-merkle-authority.json").read_text())
    state = json.loads((base["run167"] / "trusted-archive-merkle-rebridge-state.json").read_text())
    assert active["authority"] == state["activeAuthority"]
    assert active["revokedKeyFingerprints"] == state["revokedKeyFingerprints"]
    assert active["lastCheckpoints"] == state["lastCheckpoints"]


def test_run168_any_observed_recovery_equivocation_fails_even_with_quorum(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); cp, raw = _checkpoint(out)
    bad = copy.deepcopy(cp); bad["summary"]["sequence"] -= 1
    body = {k: bad[k] for k in bad if k != "recoveryCheckpointHeadSha256"}; bad["recoveryCheckpointHeadSha256"] = rec._checkpoint_head(body); badraw = rec._canonical(bad)
    src = [("source-a", "source-op-a", RecoveryAdapter("source-a", "source-op-a", raw)),
           ("source-b", "source-op-b", RecoveryAdapter("source-b", "source-op-b", raw)),
           ("source-c", "source-op-c", RecoveryAdapter("source-c", "source-op-c", badraw))]
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="EQUIVOCATION"):
        rec.recover_rebridge_history(sources=src, output_dir=tmp_path / "recovered", now=NOW, **_pins(out))


def test_run168_unanimous_old_checkpoint_rejected_by_out_of_band_pin(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); _, raw = _checkpoint(out)
    src = [(f"source-{x}", f"source-op-{x}", RecoveryAdapter(f"source-{x}", f"source-op-{x}", raw)) for x in "abc"]
    pins = _pins(out); pins["expected_checkpoint_sha256"] = "0" * 64
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="ROLLBACK_CHECKPOINT_PIN_MISMATCH"):
        rec.recover_rebridge_history(sources=src, output_dir=tmp_path / "recovered", now=NOW, **pins)


def test_run168_authority_head_pin_mismatch_rejected(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); _, raw = _checkpoint(out)
    src = [(f"source-{x}", f"source-op-{x}", RecoveryAdapter(f"source-{x}", f"source-op-{x}", raw)) for x in "abc"]
    pins = _pins(out); pins["expected_authority_head_sha256"] = "0" * 64
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="ROLLBACK_HEAD_PIN_MISMATCH"):
        rec.recover_rebridge_history(sources=src, output_dir=tmp_path / "recovered", now=NOW, **pins)


def test_run168_merkle_head_pin_mismatch_rejected(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); _, raw = _checkpoint(out)
    src = [(f"source-{x}", f"source-op-{x}", RecoveryAdapter(f"source-{x}", f"source-op-{x}", raw)) for x in "abc"]
    pins = _pins(out); pins["expected_merkle_head_sha256"] = "0" * 64
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="ROLLBACK_HEAD_PIN_MISMATCH"):
        rec.recover_rebridge_history(sources=src, output_dir=tmp_path / "recovered", now=NOW, **pins)


def test_run168_active_authority_pin_mismatch_rejected(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); _, raw = _checkpoint(out)
    src = [(f"source-{x}", f"source-op-{x}", RecoveryAdapter(f"source-{x}", f"source-op-{x}", raw)) for x in "abc"]
    pins = _pins(out); pins["expected_active_authority_sha256"] = "0" * 64
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="ROLLBACK_HEAD_PIN_MISMATCH"):
        rec.recover_rebridge_history(sources=src, output_dir=tmp_path / "recovered", now=NOW, **pins)


def test_run168_sequence_pin_mismatch_rejected(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); _, raw = _checkpoint(out)
    src = [(f"source-{x}", f"source-op-{x}", RecoveryAdapter(f"source-{x}", f"source-op-{x}", raw)) for x in "abc"]
    pins = _pins(out); pins["expected_sequence"] -= 1
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="ROLLBACK_HEAD_PIN_MISMATCH"):
        rec.recover_rebridge_history(sources=src, output_dir=tmp_path / "recovered", now=NOW, **pins)


def test_run168_recovery_locator_collision_rejected(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); _, raw = _checkpoint(out)
    loc = "s3://same/recovery/checkpoint.json"
    src = [(f"source-{x}", f"source-op-{x}", RecoveryAdapter(f"source-{x}", f"source-op-{x}", raw, locator=loc)) for x in "abc"]
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="SOURCE_LOCATOR_COLLISION"):
        rec.recover_rebridge_history(sources=src, output_dir=tmp_path / "recovered", now=NOW, **_pins(out))


def test_run168_recovery_requires_three_configured_sources(base, tmp_path):
    out = tmp_path / "preserved"; _preserve(base, out); _, raw = _checkpoint(out)
    src = [("source-a", "source-op-a", RecoveryAdapter("source-a", "source-op-a", raw)), ("source-b", "source-op-b", RecoveryAdapter("source-b", "source-op-b", raw))]
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="SOURCE_PLAN_INVALID"):
        rec.recover_rebridge_history(sources=src, output_dir=tmp_path / "recovered", now=NOW, **_pins(out))


def test_run168_duplicate_json_keys_rejected(tmp_path):
    p = tmp_path / "bad.json"; p.write_text('{"x":1,"x":2}\n')
    with pytest.raises(rec.ArchiveMerkleRecoveryError, match="DUPLICATE_KEY"):
        rec._read_json(p, "BAD")


def test_run168_input_drift_detected_before_commit(base, tmp_path, monkeypatch):
    rows = _targets(); original = rows[0][2].__call__
    class MutatingArchive:
        def __init__(self, inner): self.inner = inner; self.done = False
        def __call__(self, request):
            value = self.inner(request)
            if not self.done:
                self.done = True
                p = base["run167"] / "active-archive-merkle-rebridge.json"
                p.write_bytes(p.read_bytes() + b" ")
            return value
    ai, ao, aa, vi, vo, va = rows[0];
    rows[0] = (ai, ao, MutatingArchive(aa), vi, vo, va)
    backup = tmp_path / "backup"; shutil.copytree(base["run167"], backup)
    try:
        with pytest.raises(rec.ArchiveMerkleRecoveryError, match="INPUT_DRIFT"):
            rec.preserve_rebridge_history(run167_dir=base["run167"], output_dir=tmp_path / "out", verify_kwargs=base["kwargs"], targets=rows, now=NOW)
        assert not (tmp_path / "out").exists()
    finally:
        shutil.rmtree(base["run167"]); shutil.copytree(backup, base["run167"])


def test_run168_documentation_and_release_gates_are_wired():
    guide = SEC / "RELEASE_ARCHIVE_MERKLE_RECOVERY_GUIDE.md"
    assert guide.exists()
    text = guide.read_text().lower()
    for phrase in ("run 168", "out-of-band", "equivocation", "rollback", "recovered-active-archive-merkle-authority.json"):
        assert phrase in text
    gates = (SEC / "SECURITY_RELEASE_GATES.md").read_text().lower()
    evidence = (SEC / "RELEASE_EVIDENCE_GUIDE.md").read_text().lower()
    assert "run 168" in gates and "run 168" in evidence
