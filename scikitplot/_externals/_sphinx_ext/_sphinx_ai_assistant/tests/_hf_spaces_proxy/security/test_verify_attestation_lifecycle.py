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
from cryptography.x509.oid import NameOID, ObjectIdentifier

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SECURITY = ROOT / "_hf_spaces_proxy" / "security"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lifecycle = _load("run158_lifecycle", SECURITY / "verify_attestation_lifecycle.py")
run157 = _load("run157_test_helpers_for_run158", HERE / "test_continue_recovered_trust.py")
run155 = run157.run155
continuity = run157.continuity
NOW = run157.NOW


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(doc))
    return path


def _profiled_attestations(tmp_path: Path, root_envelope: dict, context_id: str, ca_key, ca_cert, *, claim_mutator=None):
    info = continuity._root_info(root_envelope, "RUN158_TEST_ROOT")
    paths = []
    oid = ObjectIdentifier(str(lifecycle.POLICY["vendor_claim_extension_oid"]))
    for index, key_id in enumerate(info["roles"]["root"]["keyids"], start=1):
        leaf_key = run157._priv("run158-attestation-leaf:" + key_id)
        device_hash = hashlib.sha256(("device:" + key_id).encode()).hexdigest()
        claim = {
            "schemaVersion": 1,
            "profile": "hsm-x509-v1",
            "deviceClass": "hsm",
            "manufacturer": "example-attestation-ca",
            "model": "test-hsm-158",
            "hardwareBacked": True,
            "nonExportable": True,
            "keyId": key_id,
            "publicKeySha256": hashlib.sha256(base64.b64decode(info["keys"][key_id]["keyval"]["public"])).hexdigest(),
            "deviceIdHash": device_hash,
        }
        if claim_mutator:
            claim_mutator(claim, index, key_id)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Run158 Attester " + key_id)])
        leaf = (
            x509.CertificateBuilder()
            .subject_name(subject).issuer_name(ca_cert.subject).public_key(leaf_key.public_key())
            .serial_number(int.from_bytes(hashlib.sha256(("run158:" + key_id).encode()).digest()[:16], "big"))
            .not_valid_before(NOW - timedelta(days=1)).not_valid_after(NOW + timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False, key_cert_sign=False, crl_sign=False, encipher_only=None, decipher_only=None), critical=True)
            .add_extension(x509.UnrecognizedExtension(oid, _canonical(claim)), critical=False)
            .sign(ca_key, algorithm=None)
        )
        signed = {
            "schemaVersion": 1,
            "predicateType": continuity.PREDICATE_TYPE + "/x509-key-attestation",
            "governanceId": info["governance_id"],
            "contextId": context_id,
            "rootVersion": info["version"],
            "rootSha256": info["sha256"],
            "keyId": key_id,
            "publicKey": info["keys"][key_id]["keyval"]["public"],
            "deviceClass": "hsm",
            "manufacturer": "example-attestation-ca",
            "model": "test-hsm-158",
            "attestedAt": run155._ts(NOW - timedelta(minutes=2)),
            "expires": run155._ts(NOW + timedelta(days=30)),
            "certificateChainDer": [base64.b64encode(leaf.public_bytes(serialization.Encoding.DER)).decode()],
        }
        doc = {"signature": base64.b64encode(leaf_key.sign(_canonical(signed))).decode(), "signed": signed}
        paths.append(_write(tmp_path / f"attestation-{index}.json", doc))
    return paths


def _profiled_continuity(tmp_path: Path, *, claim_mutator=None):
    seed = run157._recovered_setup(tmp_path / "seed")
    seed_bundle = json.loads((seed["activated"] / "release-root-continuity-bundle.json").read_text())
    sealed = tmp_path / "sealed"; recovery = tmp_path / "recovery"
    continuity._write_docs(sealed, seed_bundle["baseSeal"])
    continuity._write_docs(recovery, seed_bundle["recoveryOutput"])
    ca_path, ca_pin, ca_key, ca_cert = run157._ca(tmp_path / "attestation-ca")
    attest = _profiled_attestations(tmp_path / "attestations", seed["replacement"], "incident-2026-09-05", ca_key, ca_cert, claim_mutator=claim_mutator)
    out = tmp_path / "continuity"
    continuity.activate_recovery(
        sealed_dir=sealed, bootstrap_root_sha256=seed["pin"], recovery_dir=recovery,
        expected_recovery_root_sha256=seed["rr_pin"], attestation_paths=attest,
        attestation_trust_root_paths=[ca_path], expected_attestation_root_sha256=[ca_pin],
        output_dir=out, now=NOW,
    )
    return {**seed, "activated": out, "ca_path": ca_path, "ca_pin": ca_pin, "ca_key": ca_key, "ca_cert": ca_cert}


def _status_keys(label="status"):
    private = {}
    keys = {}
    for suffix in ("a", "b", "c"):
        kid = f"{label}/key-{suffix}"
        key = run157._priv("run158:" + kid); private[kid] = key
        pub = base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
        keys[kid] = {"keytype": "ed25519", "scheme": "ed25519", "identity": kid, "operator": f"{label}-operator-{suffix}", "expires": run155._ts(NOW + timedelta(days=500)), "keyval": {"public": pub}}
    return private, keys


def _continuity_docs(path: Path):
    return lifecycle._continuity_docs(path)


def _ca_set(tmp_path: Path, setup, *, version=1, previous=None, ca_cert=None, ca_pin=None, transition_type=None, retired=None, revoked=None, status_label="status", status_private=None, status_keys=None, tamper_root_sig=False):
    docs = _continuity_docs(setup["activated"])
    if ca_cert is None:
        ca_cert = setup["ca_cert"]
    der = ca_cert.public_bytes(serialization.Encoding.DER)
    if ca_pin is None:
        ca_pin = hashlib.sha256(der).hexdigest()
    if status_private is None or status_keys is None:
        status_private, status_keys = _status_keys(status_label)
    if version == 1:
        prev_sha = "0" * 64
        transition_type = transition_type or "bootstrap"
    else:
        assert previous is not None
        prev_sha = hashlib.sha256(_canonical(previous)).hexdigest()
        transition_type = transition_type or "scheduled-rotation"
    root_ids = setup["replacement"]["signed"]["roles"]["root"]["keyids"]
    root_ids = setup["replacement"]["signed"]["roles"]["root"]["keyids"]
    selected_root = sorted(root_ids[:2])
    signed = {
        "_type": "attestation-ca-set",
        "specVersion": "1.0.0",
        "version": version,
        "governanceId": setup["replacement"]["signed"]["governanceId"],
        "issuedAt": run155._ts(NOW),
        "expires": run155._ts(NOW + timedelta(days=300)),
        "continuity": lifecycle._continuity_binding(docs),
        "previousCaSetSha256": prev_sha,
        "transition": {"type": transition_type, "retiredCaSha256": sorted(retired or []), "revokedCaSha256": sorted(revoked or [])},
        "trustRoots": [{"sha256": ca_pin, "der": base64.b64encode(der).decode(), "profile": "hsm-x509-v1"}],
        "statusAuthority": {"keys": {k: status_keys[k] for k in sorted(status_keys)}, "role": {"keyids": sorted(status_keys), "threshold": 2}},
        "requiredProfilesByDeviceClass": {"hsm": "hsm-x509-v1", "kms-hsm": "kms-hsm-x509-v1", "secure-element": "secure-element-x509-v1", "tpm": "tpm-x509-v1"},
        "selectedRootKeyIds": selected_root,
    }
    sigs = []
    for kid in selected_root:
        sig = setup["replacement_private"][kid].sign(_canonical(signed))
        if tamper_root_sig and not sigs:
            sig = bytes([sig[0] ^ 1]) + sig[1:]
        sigs.append({"keyid": kid, "sig": base64.b64encode(sig).decode()})
    doc = {"signatures": sigs, "signed": signed}
    return _write(tmp_path / f"ca-set-v{version}.json", doc), doc, status_private, status_keys


def _status_snapshot(
    tmp_path: Path,
    setup,
    ca_set: dict,
    status_private: dict,
    *,
    version=1,
    previous=None,
    revoked=None,
    omit=None,
    tamper_sig=False,
    issued=NOW,
    next_update=None,
):
    docs = _continuity_docs(setup["activated"])
    inventory, _ = lifecycle._inventory(
        docs["release-root-continuity-bundle.json"], ca_set=ca_set
    )
    revoked = revoked or {}
    entries = []
    for cert in inventory:
        if cert["sha256"] == omit:
            continue
        if cert["sha256"] in revoked:
            at, reason = revoked[cert["sha256"]]
            entries.append(
                {
                    "sha256": cert["sha256"],
                    "serialNumber": cert["serialNumber"],
                    "status": "revoked",
                    "revokedAt": run155._ts(at),
                    "reason": reason,
                }
            )
        else:
            entries.append(
                {
                    "sha256": cert["sha256"],
                    "serialNumber": cert["serialNumber"],
                    "status": "good",
                    "revokedAt": None,
                    "reason": None,
                }
            )
    selected_status = ca_set["signed"]["statusAuthority"]["role"]["keyids"][:2]
    signed = {
        "_type": "attestation-status",
        "specVersion": "1.0.0",
        "version": version,
        "governanceId": ca_set["signed"]["governanceId"],
        "caSetVersion": ca_set["signed"]["version"],
        "caSetSha256": hashlib.sha256(_canonical(ca_set)).hexdigest(),
        "previousStatusSha256": (
            "0" * 64
            if previous is None
            else hashlib.sha256(_canonical(previous)).hexdigest()
        ),
        "continuityRootChainHeadSha256": ca_set["signed"]["continuity"][
            "rootChainHeadSha256"
        ],
        "issuedAt": run155._ts(issued),
        "nextUpdate": run155._ts(next_update or (issued + timedelta(hours=12))),
        "certificates": entries,
        "selectedStatusKeyIds": selected_status,
    }
    sigs = []
    for kid in selected_status:
        sig = status_private[kid].sign(_canonical(signed))
        if tamper_sig and not sigs:
            sig = bytes([sig[0] ^ 1]) + sig[1:]
        sigs.append({"keyid": kid, "sig": base64.b64encode(sig).decode()})
    doc = {"signatures": sigs, "signed": signed}
    return _write(tmp_path / f"status-v{version}.json", doc), doc


def _initialized(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre")
    ca_path, ca_doc, status_private, status_keys = _ca_set(
        tmp_path / "inputs", setup
    )
    status_path, status_doc = _status_snapshot(
        tmp_path / "inputs", setup, ca_doc, status_private
    )
    out = tmp_path / "lifecycle"
    lifecycle.initialize_lifecycle(
        continuity_dir=setup["activated"],
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        ca_set_path=ca_path,
        status_snapshot_path=status_path,
        output_dir=out,
        now=NOW,
    )
    return setup, out, ca_doc, status_doc, status_private, status_keys


def test_run158_initializes_and_verifies_attestation_lifecycle(tmp_path: Path):
    setup, out, _, _, _, _ = _initialized(tmp_path)
    result = lifecycle.verify_lifecycle(
        output_dir=out,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
    )
    assert result["ca_set_version"] == 1 and result["status_version"] == 1


def test_run158_ca_set_requires_release_root_threshold(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre")
    ca_path, _, _, _ = _ca_set(tmp_path / "i", setup, tamper_root_sig=True)
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="SIGNATURE_INVALID"
    ):
        lifecycle._verify_ca_set(
            json.loads(ca_path.read_text()),
            continuity_docs=_continuity_docs(setup["activated"]),
            previous=None,
            now=NOW,
            historical=False,
        )


def test_run158_status_snapshot_requires_threshold_signature(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre")
    _, ca, sp, _ = _ca_set(tmp_path / "i", setup)
    status, _ = _status_snapshot(tmp_path / "i", setup, ca, sp, tamper_sig=True)
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="SIGNATURE_INVALID"
    ):
        lifecycle._verify_status_snapshot(
            json.loads(status.read_text()),
            ca_set=ca,
            continuity_bundle=_continuity_docs(setup["activated"])[
                "release-root-continuity-bundle.json"
            ],
            previous=None,
            now=NOW,
            historical=False,
        )


def test_run158_status_snapshot_must_cover_every_attestation_certificate(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre")
    _, ca, sp, _ = _ca_set(tmp_path / "i", setup)
    inventory, _ = lifecycle._inventory(
        _continuity_docs(setup["activated"])[
            "release-root-continuity-bundle.json"
        ],
        ca_set=ca,
    )
    p, _ = _status_snapshot(
        tmp_path / "i", setup, ca, sp, omit=inventory[0]["sha256"]
    )
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="COVERAGE_INCOMPLETE"
    ):
        lifecycle._verify_status_snapshot(
            json.loads(p.read_text()),
            ca_set=ca,
            continuity_bundle=_continuity_docs(setup["activated"])[
                "release-root-continuity-bundle.json"
            ],
            previous=None,
            now=NOW,
            historical=False,
        )


def test_run158_vendor_claim_requires_hardware_backed_and_non_exportable(tmp_path: Path):
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="HARDWARE_PROPERTIES_INVALID"
    ):
        setup = _profiled_continuity(
            tmp_path / "pre",
            claim_mutator=lambda c, _i, _k: c.__setitem__(
                "hardwareBacked", False
            ),
        )
        _, ca, _, _ = _ca_set(tmp_path / "i", setup)
        lifecycle._inventory(
            _continuity_docs(setup["activated"])[
                "release-root-continuity-bundle.json"
            ],
            ca_set=ca,
        )


def test_run158_vendor_claim_binds_exact_release_root_public_key(tmp_path: Path):
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="ROOT_PUBLIC_MISMATCH"
    ):
        setup = _profiled_continuity(
            tmp_path / "pre",
            claim_mutator=lambda c, _i, _k: c.__setitem__(
                "publicKeySha256", "0" * 64
            ),
        )
        _, ca, _, _ = _ca_set(tmp_path / "i", setup)
        lifecycle._inventory(
            _continuity_docs(setup["activated"])[
                "release-root-continuity-bundle.json"
            ],
            ca_set=ca,
        )


def test_run158_vendor_profile_is_device_class_specific(tmp_path: Path):
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="SEMANTICS_MISMATCH"
    ):
        setup = _profiled_continuity(
            tmp_path / "pre",
            claim_mutator=lambda c, _i, _k: c.__setitem__(
                "profile", "tpm-x509-v1"
            ),
        )
        _, ca, _, _ = _ca_set(tmp_path / "i", setup)
        lifecycle._inventory(
            _continuity_docs(setup["activated"])[
                "release-root-continuity-bundle.json"
            ],
            ca_set=ca,
        )


def test_run158_status_authority_is_disjoint_from_release_root(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre")
    priv, keys = _status_keys()
    root_kid = setup["replacement"]["signed"]["roles"]["root"]["keyids"][0]
    keys[root_kid] = keys.pop(sorted(keys)[0])
    priv[root_kid] = priv.pop(sorted(priv)[0])
    _, doc, _, _ = _ca_set(
        tmp_path / "i", setup, status_private=priv, status_keys=keys
    )
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="OVERLAPS_RELEASE_ROOT"
    ):
        lifecycle._verify_ca_set(
            doc,
            continuity_docs=_continuity_docs(setup["activated"]),
            previous=None,
            now=NOW,
            historical=False,
        )


def test_run158_status_authority_operator_is_disjoint_from_release_root(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre")
    priv, keys = _status_keys()
    first = sorted(keys)[0]
    root_kid = setup["replacement"]["signed"]["roles"]["root"]["keyids"][0]
    keys[first]["operator"] = setup["replacement"]["signed"]["keys"][root_kid]["operator"]
    _, doc, _, _ = _ca_set(tmp_path / "i", setup, status_private=priv, status_keys=keys)
    with pytest.raises(
        lifecycle.AttestationLifecycleError,
        match="OPERATOR_OVERLAPS_RELEASE_ROOT",
    ):
        lifecycle._verify_ca_set(
            doc,
            continuity_docs=_continuity_docs(setup["activated"]),
            previous=None,
            now=NOW,
            historical=False,
        )


def test_run158_live_status_has_freeze_protection(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre")
    _, ca, sp, _ = _ca_set(tmp_path / "i", setup)
    p, _ = _status_snapshot(
        tmp_path / "i", setup, ca, sp, next_update=NOW + timedelta(minutes=5)
    )
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="EXPIRED_OR_FREEZE_RISK"
    ):
        lifecycle._verify_status_snapshot(
            json.loads(p.read_text()),
            ca_set=ca,
            continuity_bundle=_continuity_docs(setup["activated"])[
                "release-root-continuity-bundle.json"
            ],
            previous=None,
            now=NOW,
            historical=False,
        )


def test_run158_historical_status_survives_later_expiry(tmp_path: Path):
    setup, out, _, _, _, _ = _initialized(tmp_path)
    result = lifecycle.verify_lifecycle(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW + timedelta(days=10), historical=True)
    assert result["ok"] is True


def test_run158_later_revocation_preserves_historical_acceptance_but_blocks_live_use(tmp_path: Path):
    setup, out, ca, old_status, sp, _ = _initialized(tmp_path)
    inventory, _ = lifecycle._inventory(_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], ca_set=ca)
    target = inventory[0]["sha256"]
    p, _ = _status_snapshot(tmp_path / "next", setup, ca, sp, version=2, previous=old_status, revoked={target: (NOW + timedelta(hours=1), "key-compromise")}, issued=NOW + timedelta(hours=2), next_update=NOW + timedelta(hours=12))
    # Advance at a time after revocation: this intentionally refuses continued live use.
    with pytest.raises(lifecycle.AttestationLifecycleError, match="ACTIVE_CERT_REVOKED"):
        lifecycle.advance_lifecycle(previous_dir=out, status_snapshot_path=p, output_dir=tmp_path / "blocked", expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW + timedelta(hours=2))
    # The signed status itself still proves the release attestation was good at its earlier attestation time.
    status = lifecycle._verify_status_snapshot(json.loads(p.read_text()), ca_set=ca, continuity_bundle=_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], previous=old_status, now=NOW + timedelta(days=20), historical=True)
    lifecycle._evaluate_status(continuity_bundle=_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], ca_set=ca, status=status, now=NOW + timedelta(days=20), historical=True)


def test_run158_revocation_effective_before_attestation_invalidates_historical_proof(tmp_path: Path):
    setup, _, ca, old_status, sp, _ = _initialized(tmp_path)
    inventory, _ = lifecycle._inventory(
        _continuity_docs(setup["activated"])["release-root-continuity-bundle.json"],
        ca_set=ca,
    )
    target = inventory[0]["sha256"]
    p, _ = _status_snapshot(
        tmp_path / "next",
        setup,
        ca,
        sp,
        version=2,
        previous=old_status,
        revoked={target: (NOW - timedelta(hours=1), "key-compromise")},
        issued=NOW,
        next_update=NOW + timedelta(hours=12),
    )
    status = lifecycle._verify_status_snapshot(
        json.loads(p.read_text()),
        ca_set=ca,
        continuity_bundle=_continuity_docs(setup["activated"])[
            "release-root-continuity-bundle.json"
        ],
        previous=old_status,
        now=NOW,
        historical=True,
    )
    with pytest.raises(lifecycle.AttestationLifecycleError, match="REVOKED_AT_ATTESTATION_TIME"):
        lifecycle._evaluate_status(continuity_bundle=_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], ca_set=ca, status=status, now=NOW, historical=True)


def test_run158_revocation_is_sticky(tmp_path: Path):
    setup, _, ca, old_status, sp, _ = _initialized(tmp_path)
    inventory, _ = lifecycle._inventory(
        _continuity_docs(setup["activated"])["release-root-continuity-bundle.json"],
        ca_set=ca,
    )
    target = inventory[0]["sha256"]
    _, s2 = _status_snapshot(
        tmp_path / "s2",
        setup,
        ca,
        sp,
        version=2,
        previous=old_status,
        revoked={target: (NOW + timedelta(minutes=1), "key-compromise")},
        issued=NOW + timedelta(minutes=2),
    )
    verified = lifecycle._verify_status_snapshot(
        s2,
        ca_set=ca,
        continuity_bundle=_continuity_docs(setup["activated"])[
            "release-root-continuity-bundle.json"
        ],
        previous=old_status,
        now=NOW + timedelta(minutes=2),
        historical=True,
    )
    _, s3 = _status_snapshot(
        tmp_path / "s3",
        setup,
        ca,
        sp,
        version=3,
        previous=verified,
        issued=NOW + timedelta(minutes=3),
    )
    with pytest.raises(lifecycle.AttestationLifecycleError, match="REVOCATION_NOT_STICKY"):
        lifecycle._verify_status_snapshot(s3, ca_set=ca, continuity_bundle=_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], previous=verified, now=NOW + timedelta(minutes=3), historical=True)


def _new_ca(tmp_path: Path, label="new-ca"):
    return run157._ca(tmp_path, label=label)


def test_run158_scheduled_ca_rotation_is_root_threshold_governed(tmp_path: Path):
    setup, out, old_ca, old_status, sp, sk = _initialized(tmp_path)
    _, new_pin, _, new_cert = _new_ca(tmp_path / "ca2")
    ca_path, ca2, _, _ = _ca_set(
        tmp_path / "rotate",
        setup,
        version=2,
        previous=old_ca,
        ca_cert=new_cert,
        ca_pin=new_pin,
        transition_type="scheduled-rotation",
        retired=[setup["ca_pin"]],
        status_private=sp,
        status_keys=sk,
    )
    status_path, _ = _status_snapshot(
        tmp_path / "rotate", setup, ca2, sp, version=2, previous=old_status
    )
    advanced = tmp_path / "advanced"
    result = lifecycle.advance_lifecycle(
        previous_dir=out,
        status_snapshot_path=status_path,
        next_ca_set_path=ca_path,
        output_dir=advanced,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
    )
    assert result["ca_set_version"] == 2
    assert result["status_version"] == 2


def test_run158_ca_rotation_requires_complete_removal_accounting(tmp_path: Path):
    setup, _, old_ca, _, sp, sk = _initialized(tmp_path)
    _, new_pin, _, new_cert = _new_ca(tmp_path / "ca2")
    _, doc, _, _ = _ca_set(
        tmp_path / "r",
        setup,
        version=2,
        previous=old_ca,
        ca_cert=new_cert,
        ca_pin=new_pin,
        status_private=sp,
        status_keys=sk,
    )
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="REMOVAL_ACCOUNTING_INVALID"
    ):
        lifecycle._verify_ca_set(
            doc,
            continuity_docs=_continuity_docs(setup["activated"]),
            previous=old_ca,
            now=NOW,
            historical=False,
        )


def test_run158_compromise_ca_rotation_requires_explicit_revocation(tmp_path: Path):
    setup, _, old_ca, _, sp, sk = _initialized(tmp_path)
    _, new_pin, _, new_cert = _new_ca(tmp_path / "ca2")
    _, doc, _, _ = _ca_set(
        tmp_path / "r",
        setup,
        version=2,
        previous=old_ca,
        ca_cert=new_cert,
        ca_pin=new_pin,
        transition_type="compromise-recovery",
        retired=[setup["ca_pin"]],
        status_private=sp,
        status_keys=sk,
    )
    with pytest.raises(
        lifecycle.AttestationLifecycleError,
        match="COMPROMISE_REQUIRES_REVOCATION",
    ):
        lifecycle._verify_ca_set(
            doc,
            continuity_docs=_continuity_docs(setup["activated"]),
            previous=old_ca,
            now=NOW,
            historical=False,
        )


def test_run158_compromised_ca_cannot_be_reintroduced(tmp_path: Path):
    setup, out, old_ca, old_status, sp, sk = _initialized(tmp_path)
    _, new_pin, _, new_cert = _new_ca(tmp_path / "ca2")
    ca2p, ca2, _, _ = _ca_set(
        tmp_path / "r2",
        setup,
        version=2,
        previous=old_ca,
        ca_cert=new_cert,
        ca_pin=new_pin,
        transition_type="compromise-recovery",
        revoked=[setup["ca_pin"]],
        status_private=sp,
        status_keys=sk,
    )
    st2p, st2 = _status_snapshot(
        tmp_path / "r2", setup, ca2, sp, version=2, previous=old_status
    )
    adv = tmp_path / "adv"
    lifecycle.advance_lifecycle(
        previous_dir=out,
        status_snapshot_path=st2p,
        next_ca_set_path=ca2p,
        output_dir=adv,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=NOW,
    )
    ca3p, ca3, _, _ = _ca_set(
        tmp_path / "r3",
        setup,
        version=3,
        previous=ca2,
        ca_cert=setup["ca_cert"],
        ca_pin=setup["ca_pin"],
        transition_type="scheduled-rotation",
        retired=[new_pin],
        status_private=sp,
        status_keys=sk,
    )
    st3p, _ = _status_snapshot(
        tmp_path / "r3", setup, ca3, sp, version=3, previous=st2
    )
    with pytest.raises(
        lifecycle.AttestationLifecycleError, match="REVOKED_CA_REINTRODUCED"
    ):
        lifecycle.advance_lifecycle(
            previous_dir=adv,
            status_snapshot_path=st3p,
            next_ca_set_path=ca3p,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
        )


def test_run158_status_versions_cannot_skip(tmp_path: Path):
    setup, out, ca, old_status, sp, _ = _initialized(tmp_path)
    p, _ = _status_snapshot(
        tmp_path / "s", setup, ca, sp, version=3, previous=old_status,
    )
    with pytest.raises(lifecycle.AttestationLifecycleError, match="VERSION_NOT_CONSECUTIVE"):
        lifecycle.advance_lifecycle(
            previous_dir=out, status_snapshot_path=p,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW,
        )


def test_run158_ca_set_versions_cannot_skip(tmp_path: Path):
    setup, _, old_ca, _, sp, sk = _initialized(tmp_path)
    _, new_pin, _, new_cert = _new_ca(tmp_path / "ca2")
    _, doc, _, _ = _ca_set(
        tmp_path / "r", setup, version=3, previous=old_ca, ca_cert=new_cert,
        ca_pin=new_pin, retired=[setup["ca_pin"]], status_private=sp,
        status_keys=sk,
    )
    with pytest.raises(lifecycle.AttestationLifecycleError, match="ROTATION_CONTINUITY_INVALID"):
        lifecycle._verify_ca_set(
            doc, continuity_docs=_continuity_docs(setup["activated"]),
            previous=old_ca, now=NOW, historical=False,
        )


def test_run158_bundle_mutation_is_detected_offline(tmp_path: Path):
    setup, out, _, _, _, _ = _initialized(tmp_path); p = out / "release-attestation-lifecycle-bundle.json"; doc = json.loads(p.read_text()); doc["events"][0]["inventorySha256"] = "0" * 64; p.write_bytes(_canonical(doc))
    with pytest.raises(lifecycle.AttestationLifecycleError):
        lifecycle.verify_lifecycle(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW, historical=True)


def test_run158_duplicate_json_keys_are_rejected(tmp_path: Path):
    p = tmp_path / "dup.json"; p.write_text('{"a":1,"a":2}\n')
    with pytest.raises(lifecycle.AttestationLifecycleError, match="DUPLICATE_KEY"):
        lifecycle._read(p, "RUN158_DUP")


def test_run158_output_contains_no_private_key_material(tmp_path: Path):
    _, out, _, _, _, _ = _initialized(tmp_path); raw = b"\n".join(p.read_bytes() for p in out.iterdir()).lower()
    assert b"privatekey" not in raw and b"private_key" not in raw and b"seed" not in raw


def test_run158_initialization_is_deterministic(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre"); ca_path, ca, sp, _ = _ca_set(tmp_path / "i", setup); status_path, _ = _status_snapshot(tmp_path / "i", setup, ca, sp)
    outs = []
    for name in ("one", "two"):
        out = tmp_path / name; lifecycle.initialize_lifecycle(continuity_dir=setup["activated"], expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], ca_set_path=ca_path, status_snapshot_path=status_path, output_dir=out, now=NOW); outs.append(out)
    for name in ["release-attestation-lifecycle-bundle.json", "trusted-attestation-lifecycle-state.json", "active-attestation-ca-set.json", "active-attestation-status.json", "release-attestation-lifecycle-receipt.json"]:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()


def test_run158_documentation_describes_status_revocation_ca_rotation_and_vendor_semantics():
    guide = (SECURITY / "RELEASE_ATTESTATION_LIFECYCLE_GUIDE.md").read_text()
    gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    assert "Run 158" in guide and "revocation" in guide.lower() and "CA rotation" in guide and "vendor" in guide.lower()
    assert "Run 158" in gates and "attestation lifecycle" in gates.lower()


def _resign_ca_set(doc: dict, setup) -> dict:
    signed = doc["signed"]
    sigs = []
    for kid in signed["selectedRootKeyIds"]:
        sig = setup["replacement_private"][kid].sign(_canonical(signed))
        sigs.append({"keyid": kid, "sig": base64.b64encode(sig).decode()})
    return {"signatures": sorted(sigs, key=lambda x: x["keyid"]), "signed": signed}


def _resign_status(doc: dict, private: dict) -> dict:
    signed = doc["signed"]
    sigs = []
    for kid in signed["selectedStatusKeyIds"]:
        sig = private[kid].sign(_canonical(signed))
        sigs.append({"keyid": kid, "sig": base64.b64encode(sig).decode()})
    return {"signatures": sorted(sigs, key=lambda x: x["keyid"]), "signed": signed}


def test_run158_selected_status_signers_must_span_independent_operators(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre"); priv, keys = _status_keys(); ids = sorted(keys); keys[ids[1]]["operator"] = keys[ids[0]]["operator"]
    _, ca, _, _ = _ca_set(tmp_path / "i", setup, status_private=priv, status_keys=keys)
    # Three configured keys still span two operators, but the selected 2-of-3 subset does not.
    _, doc = _status_snapshot(tmp_path / "i", setup, ca, priv)
    with pytest.raises(lifecycle.AttestationLifecycleError, match="SIGNER_OPERATOR_QUORUM_INVALID"):
        lifecycle._verify_status_snapshot(doc, ca_set=ca, continuity_bundle=_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], previous=None, now=NOW, historical=False)


def test_run158_ca_set_rejects_extra_authorized_signature_malleability(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre"); _, doc, _, _ = _ca_set(tmp_path / "i", setup)
    third = setup["replacement"]["signed"]["roles"]["root"]["keyids"][2]
    sig = setup["replacement_private"][third].sign(_canonical(doc["signed"]))
    doc["signatures"].append({"keyid": third, "sig": base64.b64encode(sig).decode()}); doc["signatures"].sort(key=lambda x: x["keyid"])
    with pytest.raises(lifecycle.AttestationLifecycleError, match="SIGNATURE_SET_MISMATCH"):
        lifecycle._verify_ca_set(doc, continuity_docs=_continuity_docs(setup["activated"]), previous=None, now=NOW, historical=False)


def test_run158_status_rejects_extra_authorized_signature_malleability(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre"); _, ca, priv, _ = _ca_set(tmp_path / "i", setup); _, doc = _status_snapshot(tmp_path / "i", setup, ca, priv)
    third = ca["signed"]["statusAuthority"]["role"]["keyids"][2]
    sig = priv[third].sign(_canonical(doc["signed"])); doc["signatures"].append({"keyid": third, "sig": base64.b64encode(sig).decode()}); doc["signatures"].sort(key=lambda x: x["keyid"])
    with pytest.raises(lifecycle.AttestationLifecycleError, match="SIGNATURE_KEY_UNAUTHORIZED|SIGNATURE_SET_MISMATCH"):
        lifecycle._verify_status_snapshot(doc, ca_set=ca, continuity_bundle=_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], previous=None, now=NOW, historical=False)


def test_run158_status_binds_exact_previous_status_hash(tmp_path: Path):
    setup, out, ca, old, priv, _ = _initialized(tmp_path); _, doc = _status_snapshot(tmp_path / "s2", setup, ca, priv, version=2, previous=old)
    doc["signed"]["previousStatusSha256"] = "0" * 64; doc = _resign_status(doc, priv)
    with pytest.raises(lifecycle.AttestationLifecycleError, match="PREVIOUS_HASH_MISMATCH"):
        lifecycle._verify_status_snapshot(doc, ca_set=ca, continuity_bundle=_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], previous=old, now=NOW, historical=False)


def test_run158_status_cannot_predate_governing_ca_set(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre"); _, ca, priv, _ = _ca_set(tmp_path / "i", setup); _, doc = _status_snapshot(tmp_path / "i", setup, ca, priv, issued=NOW - timedelta(minutes=1), next_update=NOW + timedelta(hours=1))
    with pytest.raises(lifecycle.AttestationLifecycleError, match="PREDATES_CA_SET"):
        lifecycle._verify_status_snapshot(doc, ca_set=ca, continuity_bundle=_continuity_docs(setup["activated"])["release-root-continuity-bundle.json"], previous=None, now=NOW, historical=False)


def test_run158_live_verification_retains_run157_active_root_freeze_check(tmp_path: Path):
    setup, out, _, _, _, _ = _initialized(tmp_path)
    with pytest.raises(lifecycle.AttestationLifecycleError, match="EMBEDDED_CONTINUITY_INVALID:CONTINUITY_VERIFY_ACTIVE_ROOT_EXPIRED_OR_FREEZE_RISK"):
        lifecycle.verify_lifecycle(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW + timedelta(days=366))


def test_run158_ca_set_cannot_outlive_active_release_root(tmp_path: Path):
    setup = _profiled_continuity(tmp_path / "pre"); _, doc, _, _ = _ca_set(tmp_path / "i", setup); doc["signed"]["expires"] = run155._ts(NOW + timedelta(days=400)); doc = _resign_ca_set(doc, setup)
    with pytest.raises(lifecycle.AttestationLifecycleError, match="OUTLIVES_ACTIVE_ROOT"):
        lifecycle._verify_ca_set(doc, continuity_docs=_continuity_docs(setup["activated"]), previous=None, now=NOW, historical=False)
