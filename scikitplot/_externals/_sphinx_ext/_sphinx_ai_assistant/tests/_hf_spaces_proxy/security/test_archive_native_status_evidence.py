from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SECURITY = ROOT / "_hf_spaces_proxy" / "security"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


archive = _load("run160_native_archive", SECURITY / "archive_native_status_evidence.py")
run159 = _load("run159_helpers_for_run160", HERE / "test_verify_native_status_provenance.py")
NOW = run159.NOW


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class MemoryArchive:
    def __init__(self, store: dict[str, bytes], identity: str, operator: str, *, locator: str | None = None, mutate_local=False, bad_flags=False, artifact_override=None):
        self.store = store; self.identity = identity; self.operator = operator
        self.locator = locator or f"mem+immutable://{operator}/{identity}/native-evidence"
        self.mutate_local = mutate_local; self.bad_flags = bad_flags; self.artifact_override = artifact_override

    def __call__(self, request):
        artifact = dict(request["artifact"]); local = artifact.pop("localPath", None)
        if request["operation"] == "bind":
            assert local
            p = Path(local); raw = p.read_bytes()
            if self.mutate_local:
                p.write_bytes(raw + b"x")
            status = "present" if self.locator in self.store else "created"
            if self.locator in self.store and self.store[self.locator] != raw:
                # A real create-only backend would reject this collision.  The fixture emits
                # a structurally valid response so the client-side readback verifier catches it.
                status = "present"
            else:
                self.store.setdefault(self.locator, raw)
        else:
            status = "present"
        result_artifact = self.artifact_override or artifact
        return {
            "schemaVersion": 1,
            "operation": request["operation"],
            "archiveId": request["archiveId"],
            "status": status,
            "archive": {"identity": self.identity, "operator": self.operator, "nativeStatusCredentialsReused": False, "verifierCredentialsReused": False},
            "artifact": result_artifact,
            "guarantees": {"createOnly": not self.bad_flags, "overwrite": False, "remoteReadbackVerified": True, "immutability": "content-addressed", "locator": self.locator},
            "verifiedAt": run159.native._ts(NOW),
        }


class MemoryVerifier:
    def __init__(self, store: dict[str, bytes], identity: str, operator: str, *, bad_proof=False, artifact_override=None, reused=False):
        self.store = store
        self.identity = identity
        self.operator = operator
        self.bad_proof = bad_proof
        self.artifact_override = artifact_override
        self.reused = reused

    def __call__(self, request):
        raw = self.store[request["locator"]]
        art = {"name": request["artifact"]["name"], "sha256": archive._sha_bytes(raw), "size": len(raw)}
        if self.artifact_override is not None:
            art = self.artifact_override
        proof = {"remoteReadbackVerified": True, "sha256Verified": True, "sizeVerified": True, "independentReader": True}
        if self.bad_proof:
            proof["independentReader"] = False
        return {
            "schemaVersion": 1, "operation": "verify", "archiveId": request["archiveId"], "status": "present",
            "verifier": {"identity": self.identity, "operator": self.operator, "readOnly": True, "archiveCredentialsReused": self.reused, "nativeStatusCredentialsReused": False},
            "archiveIdentity": request["archiveIdentity"], "locator": request["locator"], "artifact": art, "proof": proof,
            "verifiedAt": run159.native._ts(NOW),
        }


class RecoverySource:
    def __init__(
        self,
        raw: bytes | None,
        identity: str,
        operator: str,
        *,
        locator: str | None = None,
        unavailable=False,
        mutate=False,
        observed_at=None,
    ):
        self.raw = raw
        self.identity = identity
        self.operator = operator
        self.locator = locator or f"mem+recovery://{operator}/{identity}"
        self.unavailable = unavailable
        self.mutate = mutate
        self.observed_at = observed_at or NOW

    def __call__(self, _request):
        common = {
            "schemaVersion": 1,
            "operation": "recover",
            "source": {
                "identity": self.identity,
                "operator": self.operator,
                "readOnly": True,
                "archiveWriterCredentialsReused": False,
                "verifierCredentialsReused": False,
            },
            "observedAt": run159.native._ts(self.observed_at),
        }
        if self.unavailable:
            return dict(common, status="unavailable", reason="synthetic outage")
        assert self.raw is not None
        raw = self.raw + (b" " if self.mutate else b"")
        return dict(
            common,
            status="observed",
            locator=self.locator,
            artifact={
                "name": "release-native-evidence-archive.json",
                "sha256": archive._sha_bytes(raw),
                "size": len(raw),
            },
            payloadBase64=base64.b64encode(raw).decode(),
        )


def _native(tmp_path: Path):
    setup, _, native_dir, *_ = run159._init_native(tmp_path / "run159")
    return setup, native_dir


def _targets(store: dict[str, bytes], *, same_archive_operator=False, same_verifier_operator=False, overlap_planes=False, same_locator=False):
    ao1 = "archive-op-a"
    ao2 = ao1 if same_archive_operator else "archive-op-b"
    vo1 = "verify-op-a"
    vo2 = vo1 if same_verifier_operator else "verify-op-b"
    if overlap_planes:
        vo1 = ao1
    loc = "mem+immutable://shared/native" if same_locator else None
    a1 = MemoryArchive(store, "archive-a", ao1, locator=loc)
    a2 = MemoryArchive(store, "archive-b", ao2, locator=loc)
    v1 = MemoryVerifier(store, "verify-a", vo1)
    v2 = MemoryVerifier(store, "verify-b", vo2)
    return [
        ("archive-a", ao1, a1, "verify-a", vo1, v1),
        ("archive-b", ao2, a2, "verify-b", vo2, v2),
    ]


def _preserved(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    store: dict[str, bytes] = {}
    out = tmp_path / "archived"
    result = archive.preserve_native_status(native_dir=native_dir, output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], targets=_targets(store), now=NOW)
    return setup, native_dir, out, store, result


def test_run160_preserves_and_offline_verifies_complete_run159_output(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path)
    assert result["ok"] is True and result["archive_count"] == 2
    verified = archive.verify_native_archive(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)
    assert verified["ok"] is True and verified["source_count"] >= 2


def test_run160_archive_embeds_exact_run159_canonical_documents(tmp_path: Path):
    _, native_dir, out, _, _ = _preserved(tmp_path)
    payload = json.loads((out / "release-native-evidence-archive.json").read_text())
    embedded = payload["nativeStatus"]["output"]
    for name in archive._NATIVE_NAMES:
        assert archive._canonical(embedded[name]) == (native_dir/name).read_bytes()


def test_run160_source_inventory_rebinds_raw_der_hashes(tmp_path: Path):
    _, _, out, _, _ = _preserved(tmp_path)
    payload = json.loads((out / "release-native-evidence-archive.json").read_text())
    assert {x["type"] for x in payload["sourceInventory"]} >= {"crl", "ocsp"}
    assert {x["type"] for x in payload["sourceInventory"]} >= {"crl", "ocsp"}
    assert all(len(x["sha256"]) == 64 and x["size"] > 0 for x in payload["sourceInventory"])


def test_run160_requires_two_archives(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    store = {}
    t = _targets(store)[:1]
    with pytest.raises(archive.NativeArchiveError, match="TARGETS_INVALID"):
        archive.preserve_native_status(
            native_dir=native_dir,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            targets=t,
            now=NOW,
        )


def test_run160_requires_independent_archive_operators(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    store = {}
    with pytest.raises(archive.NativeArchiveError, match="OPERATOR_QUORUM"):
        archive.preserve_native_status(native_dir=native_dir, output_dir=tmp_path / "bad", expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], targets=_targets(store, same_archive_operator=True), now=NOW)


def test_run160_requires_independent_verifier_operators(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    store = {}
    with pytest.raises(archive.NativeArchiveError, match="OPERATOR_QUORUM"):
        archive.preserve_native_status(native_dir=native_dir, output_dir=tmp_path / "bad", expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], targets=_targets(store, same_verifier_operator=True), now=NOW)


def test_run160_archive_and_verifier_operator_planes_cannot_overlap(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    store = {}
    with pytest.raises(
        archive.NativeArchiveError, match="INDEPENDENCE|PLANES_OVERLAP"
    ):
        archive.preserve_native_status(
            native_dir=native_dir,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            targets=_targets(store, overlap_planes=True),
            now=NOW,
        )


def test_run160_archive_and_verifier_identity_must_be_distinct(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    store = {}
    t = _targets(store)
    a = t[0]
    t[0] = (
        a[0],
        a[1],
        a[2],
        a[0],
        a[4],
        MemoryVerifier(store, a[0], a[4]),
    )
    with pytest.raises(archive.NativeArchiveError, match="INDEPENDENCE"):
        archive.preserve_native_status(
            native_dir=native_dir,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            targets=t,
            now=NOW,
        )


def test_run160_rejects_archive_without_create_only_guarantee(tmp_path: Path):
    setup, native_dir = _native(tmp_path); store = {}; t = _targets(store)
    t[0] = (t[0][0], t[0][1], MemoryArchive(store, t[0][0], t[0][1], bad_flags=True), *t[0][3:])
    with pytest.raises(archive.NativeArchiveError, match="GUARANTEES_INVALID"):
        archive.preserve_native_status(native_dir=native_dir, output_dir=tmp_path / "bad", expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], targets=t, now=NOW)


def test_run160_rejects_independent_verifier_without_readback_proof(tmp_path: Path):
    setup, native_dir = _native(tmp_path); store = {}; t = _targets(store)
    t[0] = (*t[0][:5], MemoryVerifier(store, t[0][3], t[0][4], bad_proof=True))
    with pytest.raises(archive.NativeArchiveError, match="VERIFIER_PROOF_INVALID"):
        archive.preserve_native_status(native_dir=native_dir, output_dir=tmp_path / "bad", expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], targets=t, now=NOW)


def test_run160_rejects_verifier_credential_reuse(tmp_path: Path):
    setup, native_dir = _native(tmp_path); store = {}; t = _targets(store)
    t[0] = (*t[0][:5], MemoryVerifier(store, t[0][3], t[0][4], reused=True))
    with pytest.raises(archive.NativeArchiveError, match="CREDENTIAL_REUSE"):
        archive.preserve_native_status(native_dir=native_dir, output_dir=tmp_path / "bad", expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], targets=t, now=NOW)


def test_run160_rejects_remote_locator_collision(tmp_path: Path):
    setup, native_dir = _native(tmp_path); store = {}
    with pytest.raises(archive.NativeArchiveError, match="LOCATOR_COLLISION"):
        archive.preserve_native_status(native_dir=native_dir, output_dir=tmp_path / "bad", expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], targets=_targets(store, same_locator=True), now=NOW)


def test_run160_rejects_archive_artifact_mutation_during_bind(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    store = {}
    t = _targets(store)
    t[0] = (
        t[0][0],
        t[0][1],
        MemoryArchive(store, t[0][0], t[0][1], mutate_local=True),
        *t[0][3:],
    )
    with pytest.raises(archive.NativeArchiveError, match="CHANGED_DURING_BIND"):
        archive.preserve_native_status(
            native_dir=native_dir,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            targets=t,
            now=NOW,
        )


def test_run160_rejects_remote_artifact_hash_mismatch(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    store = {}
    t = _targets(store)
    bad = {
        "name": "release-native-evidence-archive.json",
        "sha256": "0" * 64,
        "size": 1,
    }
    t[0] = (
        t[0][0],
        t[0][1],
        MemoryArchive(store, t[0][0], t[0][1], artifact_override=bad),
        *t[0][3:],
    )
    with pytest.raises(archive.NativeArchiveError, match="ARTIFACT_MISMATCH"):
        archive.preserve_native_status(
            native_dir=native_dir,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            targets=t,
            now=NOW,
        )


def test_run160_archive_payload_and_state_are_deterministic(tmp_path: Path):
    setup, native_dir = _native(tmp_path)
    outs = []
    for n in ("one", "two"):
        store = {}
        out = tmp_path / n
        archive.preserve_native_status(
            native_dir=native_dir,
            output_dir=out,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            targets=_targets(store),
            now=NOW,
        )
        outs.append(out)
    for name in (
        "release-native-evidence-archive.json",
        "trusted-native-evidence-archive-state.json",
    ):
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()


def test_run160_archive_offline_verifier_detects_embedded_run159_mutation(tmp_path: Path):
    setup, _, out, _, _ = _preserved(tmp_path)
    p = out / "release-native-evidence-archive.json"
    doc = json.loads(p.read_text())
    doc["nativeStatus"]["output"]["active-native-status-evidence.json"]["nativeStatusChainHeadSha256"] = "0" * 64
    p.write_bytes(_canonical(doc))
    with pytest.raises(archive.NativeArchiveError):
        archive.verify_native_archive(
            output_dir=out,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
            historical=True,
        )


def test_run160_archive_offline_verifier_detects_receipt_mutation(tmp_path: Path):
    setup, _, out, _, _ = _preserved(tmp_path)
    p = out / "release-native-evidence-archive-receipt.json"
    doc = json.loads(p.read_text())
    doc["archiveCount"] = 99
    p.write_bytes(_canonical(doc))
    with pytest.raises(archive.NativeArchiveError, match="RECEIPT"):
        archive.verify_native_archive(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)


def _recovery_sources(raw: bytes, *, third_unavailable=False, conflict=False, observed_at=None):
    a = RecoverySource(raw, "recover-a", "recovery-op-a", observed_at=observed_at)
    b = RecoverySource(raw, "recover-b", "recovery-op-b", observed_at=observed_at)
    sources = [("recover-a", "recovery-op-a", a), ("recover-b", "recovery-op-b", b)]
    if third_unavailable:
        c = RecoverySource(None, "recover-c", "recovery-op-c", unavailable=True, observed_at=observed_at)
        sources.append(("recover-c", "recovery-op-c", c))
    if conflict:
        bad_doc = json.loads(raw)
        bad_doc["status"] = "tampered-native-status-evidence"
        bad = _canonical(bad_doc)
        c = RecoverySource(bad, "recover-c", "recovery-op-c", observed_at=observed_at)
        sources.append(("recover-c", "recovery-op-c", c))
    return sources


def test_run160_recovers_exact_run159_from_two_independent_sources(tmp_path: Path):
    setup, native_dir, out, _, result = _preserved(tmp_path)
    raw = (out / "release-native-evidence-archive.json").read_bytes()
    recovered = tmp_path / "recovered"
    r = archive.recover_native_status(sources=_recovery_sources(raw), output_dir=recovered, expected_archive_sha256=result["archive_sha256"], expected_native_status_chain_head_sha256=result["native_status_chain_head_sha256"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)
    assert r["ok"] is True and r["observed"] == 2
    for name in archive._NATIVE_NAMES:
        assert (recovered/"recovered-native-status"/name).read_bytes() == (native_dir/name).read_bytes()


def test_run160_recovery_allows_one_unavailable_source_with_two_source_quorum(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path)
    raw = (out / "release-native-evidence-archive.json").read_bytes()
    r = archive.recover_native_status(sources=_recovery_sources(raw, third_unavailable=True), output_dir=tmp_path / "r", expected_archive_sha256=result["archive_sha256"], expected_native_status_chain_head_sha256=result["native_status_chain_head_sha256"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)
    assert r["observed"] == 2 and r["unavailable"] == 1


def test_run160_recovery_rejects_any_observed_equivocation_even_with_two_matching(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path); raw = (out / "release-native-evidence-archive.json").read_bytes()
    with pytest.raises(archive.NativeArchiveError, match="EQUIVOCATION"):
        archive.recover_native_status(sources=_recovery_sources(raw, conflict=True), output_dir=tmp_path / "r", expected_archive_sha256=result["archive_sha256"], expected_native_status_chain_head_sha256=result["native_status_chain_head_sha256"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)


def test_run160_recovery_rejects_unanimous_but_wrong_archive_pin(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path); raw = (out / "release-native-evidence-archive.json").read_bytes()
    with pytest.raises(archive.NativeArchiveError, match="ROLLBACK_PIN_MISMATCH"):
        archive.recover_native_status(sources=_recovery_sources(raw), output_dir=tmp_path / "r", expected_archive_sha256="0" * 64, expected_native_status_chain_head_sha256=result["native_status_chain_head_sha256"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)


def test_run160_recovery_rejects_unanimous_but_wrong_chain_head_pin(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path)
    raw = (out / "release-native-evidence-archive.json").read_bytes()
    with pytest.raises(archive.NativeArchiveError, match="CHAIN_HEAD_PIN_MISMATCH"):
        archive.recover_native_status(
            sources=_recovery_sources(raw),
            output_dir=tmp_path / "r",
            expected_archive_sha256=result["archive_sha256"],
            expected_native_status_chain_head_sha256="0" * 64,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
            historical=True,
        )


def test_run160_recovery_requires_independent_operators(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path)
    raw = (out / "release-native-evidence-archive.json").read_bytes()
    a = RecoverySource(raw, "a", "same")
    b = RecoverySource(raw, "b", "same")
    with pytest.raises(archive.NativeArchiveError, match="QUORUM_PLAN_INVALID"):
        archive.recover_native_status(
            sources=[("a", "same", a), ("b", "same", b)],
            output_dir=tmp_path / "r",
            expected_archive_sha256=result["archive_sha256"],
            expected_native_status_chain_head_sha256=result[
                "native_status_chain_head_sha256"
            ],
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
            historical=True,
        )


def test_run160_historical_recovery_survives_native_status_expiry(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path)
    raw = (out / "release-native-evidence-archive.json").read_bytes()
    r = archive.recover_native_status(
        sources=_recovery_sources(raw, observed_at=NOW + timedelta(days=400)),
        output_dir=tmp_path / "r",
        expected_archive_sha256=result["archive_sha256"],
        expected_native_status_chain_head_sha256=result[
            "native_status_chain_head_sha256"
        ],
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW + timedelta(days=400),
        historical=True,
    )
    assert r["ok"] is True


def test_run160_live_archive_verification_retains_run159_freeze_protection(tmp_path: Path):
    setup, _, out, _, _ = _preserved(tmp_path)
    with pytest.raises(archive.NativeArchiveError):
        archive.verify_native_archive(
            output_dir=out,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW + timedelta(days=2),
            historical=False,
        )


def test_run160_recovery_rejects_duplicate_locator_sources(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path)
    raw = (out / "release-native-evidence-archive.json").read_bytes()
    loc = "mem+recovery://same/object"
    a = RecoverySource(raw, "a", "op-a", locator=loc)
    b = RecoverySource(raw, "b", "op-b", locator=loc)
    with pytest.raises(archive.NativeArchiveError, match="LOCATOR_COLLISION"):
        archive.recover_native_status(
            sources=[("a", "op-a", a), ("b", "op-b", b)],
            output_dir=tmp_path / "r",
            expected_archive_sha256=result["archive_sha256"],
            expected_native_status_chain_head_sha256=result[
                "native_status_chain_head_sha256"
            ],
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
            historical=True,
        )


def test_run160_duplicate_json_keys_are_rejected(tmp_path: Path):
    p = tmp_path / "dup.json"; p.write_text('{"a":1,"a":2}\n')
    with pytest.raises(archive.NativeArchiveError, match="DUPLICATE_KEY"):
        archive._read_json(p, "RUN160_DUP")


def test_run160_persistent_output_contains_no_local_paths_or_private_key_markers(tmp_path: Path):
    _, native_dir, out, _, _ = _preserved(tmp_path); raw = b"\n".join(p.read_bytes() for p in out.iterdir()).lower()
    assert str(native_dir).encode().lower() not in raw
    assert b"private_key" not in raw and b"privatekey" not in raw and b"localpath" not in raw and b"seed" not in raw


def test_run160_documentation_describes_replication_verifier_diversity_and_recovery():
    guide = (SECURITY / "RELEASE_NATIVE_ARCHIVE_GUIDE.md").read_text(); gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    assert "Run 160" in guide and "immutable" in guide.lower() and "independent" in guide.lower() and "recovery" in guide.lower() and "equivocation" in guide.lower()
    assert "Run 160" in gates and "native" in gates.lower() and "archive" in gates.lower()


def test_run160_receipt_replays_exact_preserved_adapter_results(tmp_path: Path):
    setup, _, out, _, _ = _preserved(tmp_path)
    p = out / "release-native-evidence-archive-receipt.json"; doc = json.loads(p.read_text())
    doc["archives"][0]["bindResult"]["guarantees"]["remoteReadbackVerified"] = False
    p.write_bytes(_canonical(doc))
    with pytest.raises(archive.NativeArchiveError, match="RESULT_REBIND|GUARANTEES"):
        archive.verify_native_archive(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW + timedelta(days=400), historical=True)


def test_run160_historical_archive_verification_does_not_expire_old_receipt_observations(tmp_path: Path):
    setup, _, out, _, _ = _preserved(tmp_path)
    result = archive.verify_native_archive(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW + timedelta(days=400), historical=True)
    assert result["ok"] is True


def test_run160_create_only_retry_accepts_present_only_for_same_artifact(tmp_path: Path):
    setup, native_dir = _native(tmp_path); store = {}; outs = []
    for name in ("one", "two"):
        out = tmp_path / name
        archive.preserve_native_status(native_dir=native_dir, output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], targets=_targets(store), now=NOW)
        outs.append(out)
    assert (outs[0] / "release-native-evidence-archive.json").read_bytes() == (outs[1] / "release-native-evidence-archive.json").read_bytes()
    receipt = json.loads((outs[1] / "release-native-evidence-archive-receipt.json").read_text())
    assert all(x["bindResult"]["status"] == "present" for x in receipt["archives"])


def test_run160_recovery_rejects_stale_reader_observation(tmp_path: Path):
    setup, _, out, _, result = _preserved(tmp_path)
    raw = (out / "release-native-evidence-archive.json").read_bytes()
    with pytest.raises(archive.NativeArchiveError, match="OBSERVED_AT_STALE"):
        archive.recover_native_status(
            sources=_recovery_sources(raw),
            output_dir=tmp_path / "r",
            expected_archive_sha256=result["archive_sha256"],
            expected_native_status_chain_head_sha256=result["native_status_chain_head_sha256"],
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW + timedelta(hours=1),
            historical=True,
        )


def test_run160_command_adapter_bounds_output_while_produced(tmp_path: Path, monkeypatch):
    script = tmp_path / "noisy.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(b'x' * 4096)\n"
    )
    script.chmod(0o755)
    real_popen = archive.subprocess.Popen
    seen = {}

    def tracking_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        seen["process"] = process
        seen["stdin"] = process.stdin
        return process

    monkeypatch.setattr(archive.subprocess, "Popen", tracking_popen)
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setattr(archive.os, "defpath", str(empty_path))
    old = archive.POLICY["max_adapter_output_bytes"]
    archive.POLICY["max_adapter_output_bytes"] = 1024
    try:
        adapter = archive.command_archive([sys.executable, str(script)])
        with pytest.raises(archive.NativeArchiveError, match="OUTPUT_TOO_LARGE"):
            adapter({"x": 1})
    finally:
        archive.POLICY["max_adapter_output_bytes"] = old

    process = seen["process"]
    assert process.poll() is not None
    assert seen["stdin"] is not None and seen["stdin"].closed
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
