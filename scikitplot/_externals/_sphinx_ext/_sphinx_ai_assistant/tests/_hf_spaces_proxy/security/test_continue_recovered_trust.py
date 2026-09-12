from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SECURITY = ROOT / "_hf_spaces_proxy" / "security"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


continuity = _load("run157_continuity", SECURITY / "continue_recovered_trust.py")
run156 = _load("run157_run156_fixture", HERE / "test_maintain_release_trust.py")
run155 = run156.run155
NOW = run155.NOW

NEXT_ROOT_MEMBERS = [
    {"keyId": "root3/key-a", "identity": "root3/offline-a", "operator": "root3-operator-a"},
    {"keyId": "root3/key-b", "identity": "root3/offline-b", "operator": "root3-operator-b"},
    {"keyId": "root3/key-c", "identity": "root3/offline-c", "operator": "root3-operator-c"},
]


def _priv(label: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(("run157:" + label).encode()).digest())


def _canonical(value) -> bytes:
    return continuity._canonical(value)


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(doc))
    return path


def _ca(tmp_path: Path, *, label="attestation-root"):
    key = _priv(label)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Run157 Attestation Root")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(1001)
        .not_valid_before(NOW - timedelta(days=30)).not_valid_after(NOW + timedelta(days=800))
        .add_extension(x509.BasicConstraints(ca=True, path_length=2), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False, key_cert_sign=True, crl_sign=True, encipher_only=None, decipher_only=None), critical=True)
        .sign(key, algorithm=None)
    )
    der = cert.public_bytes(serialization.Encoding.DER)
    path = tmp_path / "attestation-root.der"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(der)
    return path, hashlib.sha256(der).hexdigest(), key, cert


def _leaf(ca_key, ca_cert, key_id: str):
    key = _priv("attestation-leaf:" + key_id)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Attester " + key_id)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(ca_cert.subject).public_key(key.public_key())
        .serial_number(int.from_bytes(hashlib.sha256(key_id.encode()).digest()[:16], "big"))
        .not_valid_before(NOW - timedelta(days=1)).not_valid_after(NOW + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False, encipher_only=None, decipher_only=None), critical=True)
        .sign(ca_key, algorithm=None)
    )
    return key, cert


def _attestation_docs(tmp_path: Path, root_envelope: dict, context_id: str, ca_key, ca_cert, *, tamper_sig=False, omit_key=None, wrong_public=False, bad_device=False):
    info = continuity._root_info(root_envelope, "TEST_ROOT")
    paths = []
    for index, key_id in enumerate(info["roles"]["root"]["keyids"], start=1):
        if key_id == omit_key:
            continue
        leaf_key, leaf_cert = _leaf(ca_key, ca_cert, key_id)
        signed = {
            "schemaVersion": 1,
            "predicateType": continuity.PREDICATE_TYPE + "/x509-key-attestation",
            "governanceId": info["governance_id"],
            "contextId": context_id,
            "rootVersion": info["version"],
            "rootSha256": info["sha256"],
            "keyId": key_id,
            "publicKey": ("A" * 44 if wrong_public and index == 1 else info["keys"][key_id]["keyval"]["public"]),
            "deviceClass": ("laptop" if bad_device and index == 1 else "hsm"),
            "manufacturer": "example-attestation-ca",
            "model": "test-hsm-1",
            "attestedAt": run155._ts(NOW - timedelta(minutes=2)),
            "expires": run155._ts(NOW + timedelta(days=30)),
            "certificateChainDer": [base64.b64encode(leaf_cert.public_bytes(serialization.Encoding.DER)).decode("ascii")],
        }
        sig_key = _priv("wrong-attester") if tamper_sig and index == 1 else leaf_key
        doc = {"signature": base64.b64encode(sig_key.sign(_canonical(signed))).decode("ascii"), "signed": signed}
        paths.append(_write(tmp_path / f"attestation-{index}.json", doc))
    return paths


def _recovered_setup(tmp_path: Path, *, attestation_kwargs=None):
    transition = run155.run154._transition_run(tmp_path / "run154-first")
    first_candidate, history_dir = transition[1], transition[2]
    root_env, root_private = run155._root_envelope()
    _, sealed, *_ = run155._seal(tmp_path / "seal-first", candidate=first_candidate, envelope=root_env, private=root_private)
    pin = json.loads((sealed / "release-root-bundle.json").read_text())["bootstrapRootSha256"]
    effective = run156.trust._effective_root_from_run155(sealed, pin, now=NOW)
    rr_path, rr_pin, rr_private, rr_members = run156._recovery_root(tmp_path / "recovery-input", effective["info"]["governance_id"])
    repl_path, repl_doc, repl_private = run156._replacement(tmp_path / "recovery-input")
    replacement = run156.trust._replacement_root(repl_path, effective, ["root/key-a"], now=NOW)
    selected = sorted(run156.trust._verify_recovery_root(rr_path, rr_pin, now=NOW)["role"]["keyids"][:2])
    subject = {
        "schemaVersion": 1,
        "predicateType": run156.trust.PREDICATE_TYPE + "/root-recovery",
        "governanceId": effective["info"]["governance_id"],
        "incidentId": "incident-2026-09-05",
        "previousRoot": {
            "version": effective["version"],
            "sha256": effective["sha256"],
            "chainHeadSha256": effective["chainHeadSha256"],
        },
        "replacementRoot": {
            "version": replacement["version"],
            "sha256": replacement["sha256"],
        },
        "recoveryRoot": {"version": 1, "sha256": rr_pin},
        "compromisedRootKeyIds": ["root/key-a"],
        "selectedRecoveryKeyIds": selected,
    }
    recovery_sigs = run156._recovery_signatures(tmp_path / "recovery-input", subject, rr_private, rr_members)
    recovery_out = tmp_path / "run156-recovery"
    run156.trust.recover_root(
        sealed_dir=sealed,
        bootstrap_root_sha256=pin,
        recovery_root_path=rr_path,
        expected_recovery_root_sha256=rr_pin,
        replacement_root_path=repl_path,
        compromised_root_key_ids=["root/key-a"],
        incident_id="incident-2026-09-05",
        recovery_signature_paths=recovery_sigs,
        output_dir=recovery_out,
        now=NOW,
    )
    ca_path, ca_pin, ca_key, ca_cert = _ca(tmp_path / "attestation-ca")
    attestations = _attestation_docs(tmp_path / "attestations", repl_doc, "incident-2026-09-05", ca_key, ca_cert, **(attestation_kwargs or {}))
    activated = tmp_path / "activated"
    continuity.activate_recovery(
        sealed_dir=sealed,
        bootstrap_root_sha256=pin,
        recovery_dir=recovery_out,
        expected_recovery_root_sha256=rr_pin,
        attestation_paths=attestations,
        attestation_trust_root_paths=[ca_path],
        expected_attestation_root_sha256=[ca_pin],
        output_dir=activated,
        now=NOW,
    )
    return {
        "activated": activated, "sealed": sealed, "pin": pin, "rr_pin": rr_pin, "ca_path": ca_path, "ca_pin": ca_pin, "ca_key": ca_key, "ca_cert": ca_cert,
        "replacement": repl_doc, "replacement_private": repl_private, "first_candidate": first_candidate, "history_dir": history_dir,
    }


def test_run157_activates_recovered_root_with_x509_attestation(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    verified = continuity.verify_continuity(output_dir=setup["activated"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW)
    assert verified["ok"] is True
    assert verified["active_root_version"] == 2


def test_run157_attestation_root_requires_exact_out_of_band_pin(tmp_path: Path):
    transition = run155.run154._transition_run(tmp_path / "run154-first")
    first_candidate = transition[1]
    root_env, root_private = run155._root_envelope()
    _, sealed, *_ = run155._seal(tmp_path / "seal", candidate=first_candidate, envelope=root_env, private=root_private)
    pin = json.loads((sealed / "release-root-bundle.json").read_text())["bootstrapRootSha256"]
    effective = run156.trust._effective_root_from_run155(sealed, pin, now=NOW)
    rr_path, rr_pin, rr_private, rr_members = run156._recovery_root(tmp_path / "ri", effective["info"]["governance_id"])
    repl_path, repl_doc, _ = run156._replacement(tmp_path / "ri")
    replacement = run156.trust._replacement_root(repl_path, effective, ["root/key-a"], now=NOW)
    selected = sorted(run156.trust._verify_recovery_root(rr_path, rr_pin, now=NOW)["role"]["keyids"][:2])
    subject = {
        "schemaVersion": 1,
        "predicateType": run156.trust.PREDICATE_TYPE + "/root-recovery",
        "governanceId": effective["info"]["governance_id"],
        "incidentId": "incident-2026-09-05",
        "previousRoot": {
            "version": effective["version"],
            "sha256": effective["sha256"],
            "chainHeadSha256": effective["chainHeadSha256"],
        },
        "replacementRoot": {
            "version": replacement["version"],
            "sha256": replacement["sha256"],
        },
        "recoveryRoot": {"version": 1, "sha256": rr_pin},
        "compromisedRootKeyIds": ["root/key-a"],
        "selectedRecoveryKeyIds": selected,
    }
    sigs = run156._recovery_signatures(tmp_path / "ri", subject, rr_private, rr_members)
    recovery = tmp_path / "recovery"
    run156.trust.recover_root(
        sealed_dir=sealed,
        bootstrap_root_sha256=pin,
        recovery_root_path=rr_path,
        expected_recovery_root_sha256=rr_pin,
        replacement_root_path=repl_path,
        compromised_root_key_ids=["root/key-a"],
        incident_id="incident-2026-09-05",
        recovery_signature_paths=sigs,
        output_dir=recovery,
        now=NOW,
    )
    ca_path, _, ca_key, ca_cert = _ca(tmp_path / "ca")
    attest = _attestation_docs(
        tmp_path / "att",
        repl_doc,
        "incident-2026-09-05",
        ca_key,
        ca_cert,
    )
    with pytest.raises(continuity.RootContinuityError, match="ATTESTATION_ROOT_PIN_MISMATCH"):
        continuity.activate_recovery(
            sealed_dir=sealed,
            bootstrap_root_sha256=pin,
            recovery_dir=recovery,
            expected_recovery_root_sha256=rr_pin,
            attestation_paths=attest,
            attestation_trust_root_paths=[ca_path],
            expected_attestation_root_sha256=["0" * 64],
            output_dir=tmp_path / "out",
            now=NOW,
        )


def test_run157_all_recovered_root_keys_must_be_attested(tmp_path: Path):
    with pytest.raises(continuity.RootContinuityError, match="ATTESTATION_ROOT_KEY_SET_MISMATCH"):
        _recovered_setup(tmp_path, attestation_kwargs={"omit_key": "recovered-root/key-c"})


def test_run157_attestation_leaf_signature_is_verified(tmp_path: Path):
    with pytest.raises(continuity.RootContinuityError, match="ATTESTATION_SIGNATURE_INVALID"):
        _recovered_setup(tmp_path, attestation_kwargs={"tamper_sig": True})


def test_run157_attestation_binds_exact_recovered_public_key(tmp_path: Path):
    with pytest.raises(continuity.RootContinuityError, match="ATTESTATION_PUBLIC_KEY_MISMATCH"):
        _recovered_setup(tmp_path, attestation_kwargs={"wrong_public": True})


def test_run157_attestation_device_class_is_policy_bounded(tmp_path: Path):
    with pytest.raises(continuity.RootContinuityError, match="ATTESTATION_DEVICE_CLASS_INVALID"):
        _recovered_setup(tmp_path, attestation_kwargs={"bad_device": True})


def test_run157_activation_is_deterministic(tmp_path: Path):
    setup = _recovered_setup(tmp_path / "one")
    base = setup["activated"]
    copy = tmp_path / "copy"
    copy.mkdir()
    for p in base.iterdir():
        (copy / p.name).write_bytes(p.read_bytes())
    assert (copy / "release-root-continuity-bundle.json").read_bytes() == (base / "release-root-continuity-bundle.json").read_bytes()
    assert continuity.verify_continuity(output_dir=copy, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW)["ok"]


def test_run157_offline_verifier_detects_recovery_record_mutation(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    bundle_path = setup["activated"] / "release-root-continuity-bundle.json"
    bundle = json.loads(bundle_path.read_text()); bundle["recoveryOutput"]["release-root-recovery-record.json"]["observedChannels"][0] = "evil-channel"
    bundle_path.write_bytes(_canonical(bundle))
    with pytest.raises(continuity.RootContinuityError):
        continuity.verify_continuity(output_dir=setup["activated"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW)


def test_run157_next_governance_epoch_authorizes_from_recovered_root(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    candidate2 = run155._candidate_after(
        tmp_path / "epoch2", setup["first_candidate"], setup["history_dir"]
    )
    sigs = run155._authorization_files(
        tmp_path / "epoch2-sigs", candidate2, setup["replacement"], setup["replacement_private"]
    )
    out = tmp_path / "advanced"
    result = continuity.advance_governance(
        previous_dir=setup["activated"],
        governance_dir=candidate2,
        authorization_signature_paths=sigs,
        output_dir=out,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
    )
    assert result["authorizing_root_version"] == 2
    assert result["active_root_version"] == 2
    assert result["governance_epoch"] == 2


def test_run157_post_recovery_root_rotation_uses_recovered_old_threshold(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    candidate2 = run155._candidate_after(
        tmp_path / "epoch2", setup["first_candidate"], setup["history_dir"]
    )
    sigs = run155._authorization_files(
        tmp_path / "epoch2-sigs", candidate2, setup["replacement"], setup["replacement_private"]
    )
    next_root, _ = run155._root_envelope(
        version=3,
        root_members=NEXT_ROOT_MEMBERS,
        old_private=setup["replacement_private"],
        old_root_members=run156.RECOVERED_ROOT_MEMBERS,
        issued=NOW,
        expires=run155.EXPIRES,
    )
    next_path = _write(tmp_path / "next-root.json", next_root)
    attest = _attestation_docs(tmp_path / "next-att", next_root, "governance-epoch-2", setup["ca_key"], setup["ca_cert"])
    out = tmp_path / "advanced"
    result = continuity.advance_governance(
        previous_dir=setup["activated"],
        governance_dir=candidate2,
        authorization_signature_paths=sigs,
        output_dir=out,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        next_root_path=next_path,
        root_attestation_paths=attest,
        now=NOW,
    )
    assert result["authorizing_root_version"] == 2
    assert result["active_root_version"] == 3
    assert continuity.verify_continuity(
        output_dir=out,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
    )["active_root_version"] == 3


def test_run157_compromised_pre_recovery_root_cannot_authorize_future_rotation(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    candidate2 = run155._candidate_after(
        tmp_path / "epoch2", setup["first_candidate"], setup["history_dir"]
    )
    sigs = run155._authorization_files(tmp_path / "epoch2-sigs", candidate2, setup["replacement"], setup["replacement_private"])
    old_private, _ = run155._keys()
    bad_next, _ = run155._root_envelope(
        version=3,
        root_members=NEXT_ROOT_MEMBERS,
        old_private=old_private,
        old_root_members=run155.ROOT_MEMBERS,
        issued=NOW,
        expires=run155.EXPIRES,
    )
    bad_path = _write(tmp_path / "bad-next.json", bad_next)
    attest = _attestation_docs(tmp_path / "next-att", bad_next, "governance-epoch-2", setup["ca_key"], setup["ca_cert"])
    with pytest.raises(continuity.RootContinuityError, match="ROTATION_INVALID"):
        continuity.advance_governance(
            previous_dir=setup["activated"],
            governance_dir=candidate2,
            authorization_signature_paths=sigs,
            output_dir=tmp_path / "out",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            next_root_path=bad_path,
            root_attestation_paths=attest,
            now=NOW,
        )


def test_run157_future_rotation_requires_attestation_for_every_new_root_key(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    candidate2 = run155._candidate_after(
        tmp_path / "epoch2", setup["first_candidate"], setup["history_dir"]
    )
    sigs = run155._authorization_files(
        tmp_path / "sigs", candidate2, setup["replacement"], setup["replacement_private"]
    )
    next_root, _ = run155._root_envelope(
        version=3,
        root_members=NEXT_ROOT_MEMBERS,
        old_private=setup["replacement_private"],
        old_root_members=run156.RECOVERED_ROOT_MEMBERS,
        issued=NOW,
        expires=run155.EXPIRES,
    )
    path = _write(tmp_path / "next.json", next_root)
    attest = _attestation_docs(
        tmp_path / "att",
        next_root,
        "governance-epoch-2",
        setup["ca_key"],
        setup["ca_cert"],
        omit_key="root3/key-c",
    )
    with pytest.raises(continuity.RootContinuityError, match="ATTESTATION_ROOT_KEY_SET_MISMATCH"):
        continuity.advance_governance(
            previous_dir=setup["activated"],
            governance_dir=candidate2,
            authorization_signature_paths=sigs,
            output_dir=tmp_path / "out",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            next_root_path=path,
            root_attestation_paths=attest,
            now=NOW,
        )


def test_run157_governance_signature_must_match_recovered_root_role(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    candidate2 = run155._candidate_after(
        tmp_path / "epoch2", setup["first_candidate"], setup["history_dir"]
    )
    sigs = run155._authorization_files(
        tmp_path / "sigs",
        candidate2,
        setup["replacement"],
        setup["replacement_private"],
        tamper_key="governance/key-c",
    )
    with pytest.raises(continuity.RootContinuityError, match="AUTH_SIGNATURE_INVALID"):
        continuity.advance_governance(
            previous_dir=setup["activated"],
            governance_dir=candidate2,
            authorization_signature_paths=sigs,
            output_dir=tmp_path / "out",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )


def test_run157_offline_verifier_detects_epoch_chain_head_mutation(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    candidate2 = run155._candidate_after(
        tmp_path / "epoch2", setup["first_candidate"], setup["history_dir"]
    )
    sigs = run155._authorization_files(
        tmp_path / "sigs",
        candidate2,
        setup["replacement"],
        setup["replacement_private"],
    )
    out = tmp_path / "advanced"
    continuity.advance_governance(
        previous_dir=setup["activated"],
        governance_dir=candidate2,
        authorization_signature_paths=sigs,
        output_dir=out,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
    )
    p = out / "release-root-continuity-bundle.json"
    doc = json.loads(p.read_text())
    doc["epochs"][0]["chainHeadSha256"] = "0" * 64
    p.write_bytes(_canonical(doc))
    with pytest.raises(continuity.RootContinuityError):
        continuity.verify_continuity(
            output_dir=out,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )


def test_run157_duplicate_json_keys_are_rejected(tmp_path: Path):
    p = tmp_path / "dup.json"
    p.write_text('{"a":1,"a":2}\n')
    with pytest.raises(continuity.RootContinuityError, match="DUPLICATE_KEY"):
        continuity._read(p, "TEST_DUP")


def test_run157_outputs_contain_no_private_keys(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    raw = b"\n".join(p.read_bytes() for p in setup["activated"].iterdir())
    lowered = raw.lower()
    assert b"privatekey" not in lowered and b"private_key" not in lowered and b"seed" not in lowered


def test_run157_documentation_describes_recovered_continuity_and_x509_attestation():
    guide = (SECURITY / "RELEASE_ROOT_CONTINUITY_GUIDE.md").read_text()
    gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    assert "Run 157" in guide and "X.509" in guide and "old recovered-root" in guide
    assert "Run 157" in gates and "recovered-root" in gates


def test_run157_stored_authorization_age_does_not_freeze_live_continuity(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    candidate2 = run155._candidate_after(
        tmp_path / "epoch2", setup["first_candidate"], setup["history_dir"]
    )
    sigs = run155._authorization_files(
        tmp_path / "sigs",
        candidate2,
        setup["replacement"],
        setup["replacement_private"],
    )
    out = tmp_path / "advanced"
    continuity.advance_governance(
        previous_dir=setup["activated"],
        governance_dir=candidate2,
        authorization_signature_paths=sigs,
        output_dir=out,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
    )
    # Run 155's live authorization-age limit is 168h. Accepted history remains valid later.
    verified = continuity.verify_continuity(
        output_dir=out,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW + timedelta(days=8),
    )
    assert verified["governance_epoch"] == 2


def test_run157_expired_historical_attestation_does_not_erase_accepted_history(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    # Attestation statements expire after 30 days, but their historical proof remains valid.
    verified = continuity.verify_continuity(output_dir=setup["activated"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW + timedelta(days=31))
    assert verified["active_root_version"] == 2


def test_run157_current_active_root_still_has_live_freeze_protection(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    with pytest.raises(continuity.RootContinuityError, match="ACTIVE_ROOT_EXPIRED_OR_FREEZE_RISK"):
        continuity.verify_continuity(output_dir=setup["activated"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW + timedelta(days=366))


def test_run157_recovery_revoked_root_key_cannot_reappear_later(tmp_path: Path):
    setup = _recovered_setup(tmp_path)
    candidate2 = run155._candidate_after(
        tmp_path / "epoch2", setup["first_candidate"], setup["history_dir"]
    )
    sigs = run155._authorization_files(
        tmp_path / "sigs",
        candidate2,
        setup["replacement"],
        setup["replacement_private"],
    )
    bad_members = [run155.ROOT_MEMBERS[0], NEXT_ROOT_MEMBERS[1], NEXT_ROOT_MEMBERS[2]]
    bad_root, _ = run155._root_envelope(
        version=3,
        root_members=bad_members,
        old_private=setup["replacement_private"],
        old_root_members=run156.RECOVERED_ROOT_MEMBERS,
        issued=NOW,
        expires=run155.EXPIRES,
    )
    bad_path = _write(tmp_path / "revoked-reuse.json", bad_root)
    with pytest.raises(continuity.RootContinuityError, match="RECOVERY_REVOKED_KEY_REINTRODUCED"):
        continuity.advance_governance(
            previous_dir=setup["activated"],
            governance_dir=candidate2,
            authorization_signature_paths=sigs,
            output_dir=tmp_path / "out",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            next_root_path=bad_path,
            root_attestation_paths=[],
            now=NOW,
        )


def test_run157_embedded_attestation_trust_root_is_hash_rebound(tmp_path: Path):
    setup = _recovered_setup(tmp_path); p = setup["activated"] / "release-root-continuity-bundle.json"; doc = json.loads(p.read_text())
    raw = base64.b64decode(doc["attestationTrustRoots"][0]["der"]); doc["attestationTrustRoots"][0]["der"] = base64.b64encode(raw[:-1] + bytes([raw[-1] ^ 1])).decode("ascii"); p.write_bytes(_canonical(doc))
    with pytest.raises(continuity.RootContinuityError):
        continuity.verify_continuity(output_dir=setup["activated"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW)
