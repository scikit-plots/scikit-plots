from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509 import ocsp

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


native = _load("run159_native_status", SECURITY / "verify_native_status_provenance.py")
run158 = _load("run158_test_helpers_for_run159", HERE / "test_verify_attestation_lifecycle.py")
NOW = run158.NOW + timedelta(hours=1)


def _canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _lifecycle(tmp_path: Path):
    setup, out, ca, status, sp, sk = run158._initialized(tmp_path)
    return setup, out, ca, status, sp, sk


def _leafs(lifecycle_dir: Path):
    docs = native._lifecycle_docs(lifecycle_dir)
    return native._leaf_inventory(docs["release-attestation-lifecycle-bundle.json"], docs["active-attestation-ca-set.json"]), docs


def _crl(path: Path, setup, leaves, *, number=1, revoked=None, this_update=None, next_update=None, sign_key=None):
    this_update = this_update or NOW
    next_update = next_update or (this_update + timedelta(hours=24))
    builder = x509.CertificateRevocationListBuilder().issuer_name(setup["ca_cert"].subject).last_update(this_update).next_update(next_update)
    builder = builder.add_extension(x509.CRLNumber(number), critical=False)
    for leaf in leaves:
        if revoked and leaf["sha256"] in revoked:
            when, reason = revoked[leaf["sha256"]]
            entry = x509.RevokedCertificateBuilder().serial_number(leaf["cert"].serial_number).revocation_date(when)
            entry = entry.add_extension(x509.CRLReason(reason), critical=False).build()
            builder = builder.add_revoked_certificate(entry)
    crl = builder.sign(private_key=sign_key or setup["ca_key"], algorithm=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(crl.public_bytes(serialization.Encoding.DER))
    return path


def _der_tlv(raw: bytes | bytearray, offset: int):
    tag = raw[offset]
    i = offset + 1
    first = raw[i]; i += 1
    if first < 0x80:
        length = first
    else:
        count = first & 0x7F
        if count == 0 or count > 4:
            raise ValueError("unsupported DER length")
        length = int.from_bytes(raw[i:i + count], "big"); i += count
    start = i; end = start + length
    if end > len(raw):
        raise ValueError("truncated DER")
    return tag, offset, start, end, end


def _normalize_ocsp_produced_at(raw: bytes, signing_key, produced_at: datetime) -> bytes:
    """Make synthetic OCSP producedAt deterministic, then re-sign the response."""
    buf = bytearray(raw)
    tag, _, outer_start, outer_end, _ = _der_tlv(buf, 0)
    if tag != 0x30 or outer_end != len(buf):
        raise ValueError("invalid OCSP response DER")
    pos = outer_start
    tag, _, _, _, pos = _der_tlv(buf, pos)
    if tag != 0x0A:
        raise ValueError("invalid OCSP status")
    tag, _, rb_start, rb_end, _ = _der_tlv(buf, pos)
    if tag != 0xA0:
        raise ValueError("missing OCSP responseBytes")
    tag, _, seq_start, seq_end, _ = _der_tlv(buf, rb_start)
    if tag != 0x30 or seq_end != rb_end:
        raise ValueError("invalid OCSP responseBytes")
    pos = seq_start
    tag, _, _, _, pos = _der_tlv(buf, pos)
    if tag != 0x06:
        raise ValueError("missing OCSP response type")
    tag, _, octet_start, octet_end, pos2 = _der_tlv(buf, pos)
    if tag != 0x04 or pos2 != seq_end:
        raise ValueError("invalid OCSP basic response wrapper")
    tag, _, basic_start, basic_end, _ = _der_tlv(buf, octet_start)
    if tag != 0x30 or basic_end != octet_end:
        raise ValueError("invalid BasicOCSPResponse")
    pos = basic_start
    tag, tbs_tag, tbs_start, tbs_end, pos = _der_tlv(buf, pos)
    if tag != 0x30:
        raise ValueError("missing OCSP responseData")
    inner = tbs_start
    tag, _, _, _, next_inner = _der_tlv(buf, inner)
    if tag == 0xA0:
        inner = next_inner
    tag, _, _, _, inner = _der_tlv(buf, inner)
    if tag not in (0xA1, 0xA2):
        raise ValueError("invalid OCSP responderId")
    tag, _, produced_start, produced_end, _ = _der_tlv(buf, inner)
    if tag != 0x18:
        raise ValueError("missing OCSP producedAt")
    fixed = produced_at.astimezone(timezone.utc).strftime("%Y%m%d%H%M%SZ").encode("ascii")
    if len(fixed) != produced_end - produced_start:
        raise ValueError("unexpected OCSP producedAt encoding")
    buf[produced_start:produced_end] = fixed
    signature = signing_key.sign(bytes(buf[tbs_tag:tbs_end]))
    tag, _, _, _, pos = _der_tlv(buf, pos)
    if tag != 0x30:
        raise ValueError("missing OCSP signature algorithm")
    tag, _, sig_start, sig_end, _ = _der_tlv(buf, pos)
    if tag != 0x03 or buf[sig_start] != 0 or sig_end - sig_start - 1 != len(signature):
        raise ValueError("unexpected OCSP signature encoding")
    buf[sig_start + 1:sig_end] = signature
    return bytes(buf)


def _write_ocsp_response(path: Path, response, signing_key, produced_at: datetime):
    raw = _normalize_ocsp_produced_at(response.public_bytes(serialization.Encoding.DER), signing_key, produced_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _ocsp_one(path: Path, setup, leaf, *, status=ocsp.OCSPCertStatus.GOOD, this_update=None, next_update=None, revocation_time=None, reason=None, responder_cert=None, responder_key=None):
    this_update = this_update or NOW
    next_update = next_update or (this_update + timedelta(hours=12))
    responder_cert = responder_cert or setup["ca_cert"]
    responder_key = responder_key or setup["ca_key"]
    builder = ocsp.OCSPResponseBuilder().add_response(
        cert=leaf["cert"], issuer=setup["ca_cert"], algorithm=hashes.SHA256(), cert_status=status,
        this_update=this_update, next_update=next_update,
        revocation_time=revocation_time, revocation_reason=reason,
    ).responder_id(ocsp.OCSPResponderEncoding.NAME, responder_cert)
    if responder_cert != setup["ca_cert"]:
        builder = builder.certificates([responder_cert])
    response = builder.sign(private_key=responder_key, algorithm=None)
    return _write_ocsp_response(path, response, responder_key, this_update)


def _evidence(tmp_path: Path, setup, lifecycle_dir: Path, *, crl_number=1, crl_revoked=None, ocsp_revoked=None, this_update=None):
    leaves, _ = _leafs(lifecycle_dir)
    crl = _crl(tmp_path / "status.crl", setup, leaves, number=crl_number, revoked=crl_revoked, this_update=this_update)
    ocsps = []
    for i, leaf in enumerate(leaves):
        if ocsp_revoked and leaf["sha256"] in ocsp_revoked:
            when, reason = ocsp_revoked[leaf["sha256"]]
            p = _ocsp_one(tmp_path / f"status-{i}.ocsp", setup, leaf, status=ocsp.OCSPCertStatus.REVOKED, this_update=this_update, revocation_time=when, reason=reason)
        else:
            p = _ocsp_one(tmp_path / f"status-{i}.ocsp", setup, leaf, this_update=this_update)
        ocsps.append(p)
    return crl, ocsps, leaves


def _init_native(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    crl, ocsps, leaves = _evidence(tmp_path / "evidence", setup, lifecycle_dir)
    out = tmp_path / "native"
    native.initialize_native_status(
        lifecycle_dir=lifecycle_dir,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        crl_paths=[crl], ocsp_paths=ocsps, output_dir=out, now=NOW,
    )
    return setup, lifecycle_dir, out, crl, ocsps, leaves


def test_run159_initializes_and_offline_verifies_raw_crl_ocsp(tmp_path: Path):
    setup, _, out, *_ = _init_native(tmp_path)
    result = native.verify_native_status(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW)
    assert result["ok"] is True and result["sequence"] == 1


def test_run159_preserves_exact_raw_der_bytes(tmp_path: Path):
    _, _, out, crl, ocsps, _ = _init_native(tmp_path)
    event = json.loads((out / "active-native-status-evidence.json").read_text())
    by_hash = {s["sha256"]: base64.b64decode(s["der"]) for s in event["sources"]}
    assert by_hash[hashlib.sha256(crl.read_bytes()).hexdigest()] == crl.read_bytes()
    for p in ocsps:
        assert by_hash[hashlib.sha256(p.read_bytes()).hexdigest()] == p.read_bytes()


def test_run159_requires_both_crl_and_ocsp_for_every_leaf(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    leaves, _ = _leafs(lifecycle_dir)
    crl = _crl(tmp_path / "e/status.crl", setup, leaves)
    with pytest.raises(native.NativeStatusError, match="SOURCE_COVERAGE_INCOMPLETE"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[crl], ocsp_paths=[], output_dir=tmp_path/"bad", now=NOW)


def test_run159_rejects_crl_ocsp_equivocation_even_if_one_matches_run158(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    crl, _, leaves = _evidence(tmp_path / "base", setup, lifecycle_dir)
    target = leaves[0]; when = NOW - timedelta(minutes=2)
    ocsps = []
    for i, leaf in enumerate(leaves):
        if leaf["sha256"] == target["sha256"]:
            ocsps.append(_ocsp_one(tmp_path/f"e/o{i}.ocsp", setup, leaf, status=ocsp.OCSPCertStatus.REVOKED, revocation_time=when, reason=x509.ReasonFlags.key_compromise))
        else:
            ocsps.append(_ocsp_one(tmp_path/f"e/o{i}.ocsp", setup, leaf))
    with pytest.raises(native.NativeStatusError, match="SOURCE_CONFLICT"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[crl], ocsp_paths=ocsps, output_dir=tmp_path/"bad", now=NOW)


def test_run159_rejects_native_consensus_that_disagrees_with_run158(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    leaves, _ = _leafs(lifecycle_dir); target = leaves[0]; when = NOW - timedelta(minutes=2)
    revoked = {target["sha256"]: (when, x509.ReasonFlags.key_compromise)}
    crl, ocsps, _ = _evidence(tmp_path/"e", setup, lifecycle_dir, crl_revoked=revoked, ocsp_revoked=revoked)
    with pytest.raises(native.NativeStatusError, match="RUN158_MISMATCH"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[crl], ocsp_paths=ocsps, output_dir=tmp_path/"bad", now=NOW)


def test_run159_crl_signature_is_verified(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred"); leaves, _ = _leafs(lifecycle_dir)
    wrong = run158.run157._priv("run159-wrong-crl")
    crl = _crl(tmp_path/"bad.crl", setup, leaves, sign_key=wrong)
    _, ocsps, _ = _evidence(tmp_path/"e", setup, lifecycle_dir)
    with pytest.raises(native.NativeStatusError, match="CRL_SIGNATURE_INVALID"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[crl], ocsp_paths=ocsps, output_dir=tmp_path/"bad", now=NOW)


def test_run159_ocsp_signature_is_verified(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred"); crl, ocsps, leaves = _evidence(tmp_path/"e", setup, lifecycle_dir)
    raw = bytearray(ocsps[0].read_bytes()); raw[-1] ^= 1; ocsps[0].write_bytes(bytes(raw))
    with pytest.raises(native.NativeStatusError):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[crl], ocsp_paths=ocsps, output_dir=tmp_path/"bad", now=NOW)


def test_run159_crl_number_is_required(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred"); leaves, _ = _leafs(lifecycle_dir)
    b = x509.CertificateRevocationListBuilder().issuer_name(setup["ca_cert"].subject).last_update(NOW).next_update(NOW+timedelta(hours=24))
    p = tmp_path/"no-number.crl"; p.write_bytes(b.sign(setup["ca_key"], algorithm=None).public_bytes(serialization.Encoding.DER))
    _, ocsps, _ = _evidence(tmp_path/"e", setup, lifecycle_dir)
    with pytest.raises(native.NativeStatusError, match="CRL_NUMBER_MISSING"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[p], ocsp_paths=ocsps, output_dir=tmp_path/"bad", now=NOW)


def test_run159_crl_freeze_window_is_enforced(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred"); leaves, _ = _leafs(lifecycle_dir)
    crl = _crl(tmp_path / "e/s.crl", setup, leaves, next_update=NOW + timedelta(minutes=5)); _, ocsps, _ = _evidence(tmp_path / "o", setup, lifecycle_dir)
    with pytest.raises(native.NativeStatusError, match="CRL_EXPIRED_OR_FREEZE_RISK"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[crl], ocsp_paths=ocsps, output_dir=tmp_path / "bad", now=NOW)


def test_run159_ocsp_next_update_is_required(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred"); crl, ocsps, leaves = _evidence(tmp_path / "e", setup, lifecycle_dir)
    leaf = leaves[0]
    response = ocsp.OCSPResponseBuilder().add_response(cert=leaf["cert"], issuer=setup["ca_cert"], algorithm=hashes.SHA256(), cert_status=ocsp.OCSPCertStatus.GOOD, this_update=NOW, next_update=None, revocation_time=None, revocation_reason=None).responder_id(ocsp.OCSPResponderEncoding.NAME, setup["ca_cert"]).sign(setup["ca_key"], algorithm=None)
    _write_ocsp_response(ocsps[0], response, setup["ca_key"], NOW)
    with pytest.raises(native.NativeStatusError, match="OCSP_NEXT_UPDATE_MISSING"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[crl], ocsp_paths=ocsps, output_dir=tmp_path / "bad", now=NOW)


def test_run159_ocsp_issuer_hash_is_rebound_to_exact_leaf_issuer(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    crl, ocsps, leaves = _evidence(tmp_path / "e", setup, lifecycle_dir)
    _, _, other_key, other_cert = run158._new_ca(
        tmp_path / "other-ca", label="run159-other-ca"
    )
    leaf = leaves[0]
    response = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=leaf["cert"],
            issuer=other_cert,
            algorithm=hashes.SHA256(),
            cert_status=ocsp.OCSPCertStatus.GOOD,
            this_update=NOW,
            next_update=NOW + timedelta(hours=12),
            revocation_time=None,
            revocation_reason=None,
        )
        .responder_id(ocsp.OCSPResponderEncoding.NAME, other_cert)
        .sign(other_key, algorithm=None)
    )
    _write_ocsp_response(ocsps[0], response, other_key, NOW)
    with pytest.raises(native.NativeStatusError, match="ISSUER_HASH_MISMATCH|RESPONDER"):
        native.initialize_native_status(
            lifecycle_dir=lifecycle_dir,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            crl_paths=[crl],
            ocsp_paths=ocsps,
            output_dir=tmp_path / "bad",
            now=NOW,
        )


def test_run159_advance_requires_monotonic_crl_number_and_time(tmp_path: Path):
    setup, lifecycle_dir, out, *_ = _init_native(tmp_path)
    crl, ocsps, _ = _evidence(
        tmp_path / "next",
        setup,
        lifecycle_dir,
        crl_number=1,
        this_update=NOW + timedelta(minutes=1),
    )
    with pytest.raises(native.NativeStatusError, match="CRL_NUMBER_NOT_MONOTONIC"):
        native.advance_native_status(
            previous_dir=out,
            crl_paths=[crl],
            ocsp_paths=ocsps,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW + timedelta(minutes=1),
        )


def test_run159_advance_accepts_monotonic_crl_and_ocsp_evidence(tmp_path: Path):
    setup, lifecycle_dir, out, *_ = _init_native(tmp_path)
    later = NOW + timedelta(minutes=2)
    crl, ocsps, _ = _evidence(
        tmp_path / "next", setup, lifecycle_dir, crl_number=2, this_update=later
    )
    result = native.advance_native_status(
        previous_dir=out,
        crl_paths=[crl],
        ocsp_paths=ocsps,
        output_dir=tmp_path / "next-out",
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=later,
    )
    assert result["sequence"] == 2


def test_run159_bundle_raw_evidence_mutation_is_detected_offline(tmp_path: Path):
    setup, _, out, *_ = _init_native(tmp_path)
    p = out / "release-native-status-bundle.json"
    doc = json.loads(p.read_text())
    doc["events"][0]["sources"][0]["der"] = base64.b64encode(
        b"not-der"
    ).decode()
    p.write_bytes(_canonical(doc))
    with pytest.raises(native.NativeStatusError):
        native.verify_native_status(
            output_dir=out,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=NOW,
            historical=True,
        )


def test_run159_historical_raw_evidence_survives_later_freshness_expiry(tmp_path: Path):
    setup, _, out, *_ = _init_native(tmp_path)
    assert native.verify_native_status(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW+timedelta(days=400), historical=True)["ok"] is True


def test_run159_live_raw_evidence_retains_freeze_protection(tmp_path: Path):
    setup, _, out, *_ = _init_native(tmp_path)
    with pytest.raises(native.NativeStatusError, match="EXPIRED_OR_FREEZE_RISK|PREDECESSOR_INVALID"):
        native.verify_native_status(output_dir=out, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW+timedelta(days=2))


def test_run159_vendor_native_profile_fails_closed_without_verifier(tmp_path: Path):
    manifest = {"schemaVersion": 1, "profile": "tpm2-quote-v1", "evidenceType": "tpm2-quote", "rawEvidenceBase64": base64.b64encode(b"quote").decode(), "expectedKeyId": "root/key-a", "expectedPublicKeySha256": "1" * 64}
    p = tmp_path / "vendor.json"; p.write_bytes(_canonical(manifest))
    with pytest.raises(native.NativeStatusError, match="VERIFIER_NOT_CONFIGURED"):
        native._verify_vendor_manifest(p, {})


def test_run159_vendor_verifier_result_is_bound_to_exact_raw_bytes(tmp_path: Path, monkeypatch):
    manifest = {"schemaVersion": 1, "profile": "test-vendor-v1", "evidenceType": "vendor-proof", "rawEvidenceBase64": base64.b64encode(b"raw-vendor-proof").decode(), "expectedKeyId": "root/key-a", "expectedPublicKeySha256": "1" * 64}
    p = tmp_path / "vendor.json"; p.write_bytes(_canonical(manifest))
    verifier = tmp_path / "verifier.py"; verifier.write_text(f"""#!{sys.executable}\nimport sys,json,hashlib\nr=json.loads(sys.stdin.read())\no={{'schemaVersion':1,'profile':r['profile'],'verifierIdentity':'synthetic-vendor-verifier','rawEvidenceSha256':r['rawEvidenceSha256'],'keyId':r['expectedKeyId'],'publicKeySha256':r['expectedPublicKeySha256'],'verified':True}}\nprint(json.dumps(o,sort_keys=True,separators=(',',':')))\n"""); verifier.chmod(0o755)
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setattr(native.os, "defpath", str(empty_path))
    result = native._verify_vendor_manifest(p, {"test-vendor-v1": verifier})
    assert result["rawEvidenceSha256"] == hashlib.sha256(b"raw-vendor-proof").hexdigest()
    assert result["verifierExecutableSha256"] == native._sha(verifier)


def test_run159_duplicate_json_keys_are_rejected(tmp_path: Path):
    p = tmp_path / "dup.json"; p.write_text('{"a":1,"a":2}\n')
    with pytest.raises(native.NativeStatusError, match="DUPLICATE_KEY"):
        native._read_json(p, "RUN159_DUP")


def test_run159_output_contains_no_private_key_material(tmp_path: Path):
    _, _, out, *_ = _init_native(tmp_path); raw = b"\n".join(p.read_bytes() for p in out.iterdir()).lower()
    assert b"privatekey" not in raw and b"private_key" not in raw and b"seed" not in raw


def test_run159_initialization_is_deterministic(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred"); crl, ocsps, _ = _evidence(tmp_path / "e", setup, lifecycle_dir)
    outs = []
    for name in ("one", "two"):
        out = tmp_path / name
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[crl], ocsp_paths=ocsps, output_dir=out, now=NOW); outs.append(out)
    for name in ["release-native-status-bundle.json", "trusted-native-status-state.json", "active-native-status-evidence.json", "release-native-status-receipt.json"]:
        assert (outs[0]/name).read_bytes() == (outs[1]/name).read_bytes()


def test_run159_documentation_describes_native_status_and_anti_equivocation():
    guide = (SECURITY / "RELEASE_NATIVE_STATUS_GUIDE.md").read_text()
    gates = (SECURITY / "SECURITY_RELEASE_GATES.md").read_text()
    assert "Run 159" in guide and "OCSP" in guide and "CRL" in guide and "equivocation" in guide.lower() and "offline" in guide.lower()
    assert "Run 159" in gates and "native status" in gates.lower()


def _delegated_responder(setup, *, eku=True, digital_signature=True):
    key = run158.run157._priv("run159-delegated-ocsp")
    subject = x509.Name(
        [
            x509.NameAttribute(
                x509.oid.NameOID.COMMON_NAME, "Run159 Delegated OCSP"
            )
        ]
    )
    b = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(setup["ca_cert"].subject)
        .public_key(key.public_key())
        .serial_number(159001)
        .not_valid_before(NOW - timedelta(days=1))
        .not_valid_after(NOW + timedelta(days=30))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=digital_signature,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
    )
    if eku:
        b = b.add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.OCSP_SIGNING]),
            critical=False,
        )
    cert = b.sign(setup["ca_key"], algorithm=None)
    return key, cert


def test_run159_accepts_delegated_ocsp_responder_with_ocsp_signing_eku(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    crl, ocsps, leaves = _evidence(tmp_path / "e", setup, lifecycle_dir)
    key, cert = _delegated_responder(setup)
    ocsps[0] = _ocsp_one(
        ocsps[0], setup, leaves[0], responder_cert=cert, responder_key=key
    )
    out = tmp_path / "out"
    native.initialize_native_status(
        lifecycle_dir=lifecycle_dir,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        crl_paths=[crl],
        ocsp_paths=ocsps,
        output_dir=out,
        now=NOW,
    )
    event = json.loads((out / "active-native-status-evidence.json").read_text())
    assert any(s.get("responderKind") == "delegated" for s in event["sources"])


def test_run159_rejects_delegated_ocsp_responder_without_eku(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    crl, ocsps, leaves = _evidence(tmp_path / "e", setup, lifecycle_dir)
    key, cert = _delegated_responder(setup, eku=False)
    ocsps[0] = _ocsp_one(
        ocsps[0], setup, leaves[0], responder_cert=cert, responder_key=key
    )
    with pytest.raises(
        native.NativeStatusError, match="RESPONDER_CERT_INVALID|EKU_INVALID"
    ):
        native.initialize_native_status(
            lifecycle_dir=lifecycle_dir,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            crl_paths=[crl],
            ocsp_paths=ocsps,
            output_dir=tmp_path / "bad",
            now=NOW,
        )


def test_run159_rejects_delta_crl_as_incomplete_good_evidence(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    _leafs(lifecycle_dir)
    b = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(setup["ca_cert"].subject)
        .last_update(NOW)
        .next_update(NOW + timedelta(hours=24))
        .add_extension(x509.CRLNumber(2), critical=False)
        .add_extension(x509.DeltaCRLIndicator(1), critical=True)
    )
    p = tmp_path / "delta.crl"
    p.write_bytes(
        b.sign(setup["ca_key"], algorithm=None).public_bytes(serialization.Encoding.DER)
    )
    _, ocsps, _ = _evidence(tmp_path / "e", setup, lifecycle_dir)
    with pytest.raises(native.NativeStatusError, match="CRL_DELTA_UNSUPPORTED"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[p], ocsp_paths=ocsps, output_dir=tmp_path / "bad", now=NOW)


def test_run159_rejects_two_crls_for_same_issuer_in_one_event(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    leaves, _ = _leafs(lifecycle_dir)
    p1 = _crl(tmp_path / "a.crl", setup, leaves, number=1)
    p2 = _crl(
        tmp_path / "b.crl",
        setup,
        leaves,
        number=2,
        this_update=NOW + timedelta(seconds=1),
    )
    _, ocsps, _ = _evidence(tmp_path / "e", setup, lifecycle_dir)
    with pytest.raises(native.NativeStatusError, match="CRL_SOURCE_AMBIGUOUS"):
        native.initialize_native_status(lifecycle_dir=lifecycle_dir, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], crl_paths=[p1, p2], ocsp_paths=ocsps, output_dir=tmp_path / "bad", now=NOW + timedelta(seconds=1))


def test_run159_advance_rejects_ocsp_this_update_rollback(tmp_path: Path):
    setup, lifecycle_dir, out, *_ = _init_native(tmp_path)
    later = NOW + timedelta(minutes=2)
    crl, ocsps, leaves = _evidence(
        tmp_path / "next", setup, lifecycle_dir, crl_number=2, this_update=later
    )
    # Replace one OCSP response with an older thisUpdate under the same responder/certificate identity.
    ocsps[0] = _ocsp_one(
        ocsps[0],
        setup,
        leaves[0],
        this_update=NOW - timedelta(minutes=1),
        next_update=later + timedelta(hours=1),
    )
    with pytest.raises(native.NativeStatusError, match="OCSP_TIME_NOT_MONOTONIC"):
        native.advance_native_status(
            previous_dir=out,
            crl_paths=[crl],
            ocsp_paths=ocsps,
            output_dir=tmp_path / "bad",
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            now=later,
        )


def test_run159_rejects_crl_revocation_time_after_this_update(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    leaves, _ = _leafs(lifecycle_dir)
    target = leaves[0]
    future = NOW + timedelta(hours=1)
    revoked = {target["sha256"]: (future, x509.ReasonFlags.key_compromise)}
    crl, ocsps, _ = _evidence(
        tmp_path / "e", setup, lifecycle_dir, crl_revoked=revoked
    )
    with pytest.raises(native.NativeStatusError, match="CRL_REVOCATION_FROM_FUTURE"):
        native.initialize_native_status(
            lifecycle_dir=lifecycle_dir,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            crl_paths=[crl],
            ocsp_paths=ocsps,
            output_dir=tmp_path / "bad",
            now=NOW,
        )


def test_run159_vendor_verifier_output_is_bounded_while_produced(tmp_path: Path, monkeypatch):
    manifest = {"schemaVersion": 1, "profile": "noisy-v1", "evidenceType": "vendor-proof", "rawEvidenceBase64": base64.b64encode(b"raw").decode(), "expectedKeyId": "root/key-a", "expectedPublicKeySha256": "1" * 64}
    p = tmp_path / "vendor.json"; p.write_bytes(_canonical(manifest))
    verifier = tmp_path / "noisy.py"; verifier.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write('x' * 200000)\nsys.stdout.flush()\n"); verifier.chmod(0o755)
    real_popen = native.subprocess.Popen
    seen = {}

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen["proc"] = proc
        return proc

    monkeypatch.setattr(native.subprocess, "Popen", tracking_popen)
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setattr(native.os, "defpath", str(empty_path))
    monkeypatch.setitem(native.POLICY, "vendor_verifier_max_output_bytes", 1024)
    with pytest.raises(native.NativeStatusError, match="OUTPUT_TOO_LARGE"):
        native._verify_vendor_manifest(p, {"noisy-v1": verifier})
    proc = seen["proc"]
    assert proc.poll() is not None
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed


def test_run159_live_replay_treats_prior_events_historically_but_active_event_fresh(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    leaves, _ = _leafs(lifecycle_dir)
    crl1 = _crl(
        tmp_path / "e1/status.crl",
        setup,
        leaves,
        number=1,
        this_update=NOW,
        next_update=NOW + timedelta(minutes=16),
    )
    ocsp1 = [
        _ocsp_one(
            tmp_path / f"e1/o{i}.ocsp",
            setup,
            leaf,
            this_update=NOW,
            next_update=NOW + timedelta(minutes=16),
        )
        for i, leaf in enumerate(leaves)
    ]
    out = tmp_path / "one"
    native.initialize_native_status(
        lifecycle_dir=lifecycle_dir,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        crl_paths=[crl1],
        ocsp_paths=ocsp1,
        output_dir=out,
        now=NOW,
    )
    later = NOW + timedelta(minutes=5)
    crl2 = _crl(
        tmp_path / "e2/status.crl",
        setup,
        leaves,
        number=2,
        this_update=later,
        next_update=NOW + timedelta(hours=1),
    )
    ocsp2 = [
        _ocsp_one(
            tmp_path / f"e2/o{i}.ocsp",
            setup,
            leaf,
            this_update=later,
            next_update=NOW + timedelta(hours=1),
        )
        for i, leaf in enumerate(leaves)
    ]
    out2 = tmp_path / "two"
    native.advance_native_status(
        previous_dir=out,
        crl_paths=[crl2],
        ocsp_paths=ocsp2,
        output_dir=out2,
        expected_bootstrap_root_sha256=setup["pin"],
        expected_recovery_root_sha256=setup["rr_pin"],
        expected_attestation_root_sha256=[setup["ca_pin"]],
        now=later,
    )
    # Event 1 has expired here, while event 2 and Run 158 remain fresh.
    assert native.verify_native_status(output_dir=out2, expected_bootstrap_root_sha256=setup["pin"], expected_recovery_root_sha256=setup["rr_pin"], expected_attestation_root_sha256=[setup["ca_pin"]], now=NOW + timedelta(minutes=17))["ok"] is True


def test_run159_rejects_unsigned_extraneous_ocsp_certificate_list_malleability(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    crl, ocsps, leaves = _evidence(tmp_path / "e", setup, lifecycle_dir)
    _, extra_cert = _delegated_responder(setup)
    leaf = leaves[0]
    response = (
        ocsp.OCSPResponseBuilder()
        .add_response(
            cert=leaf["cert"],
            issuer=setup["ca_cert"],
            algorithm=hashes.SHA256(),
            cert_status=ocsp.OCSPCertStatus.GOOD,
            this_update=NOW,
            next_update=NOW + timedelta(hours=12),
            revocation_time=None,
            revocation_reason=None,
        )
        .responder_id(ocsp.OCSPResponderEncoding.NAME, setup["ca_cert"])
        .certificates([extra_cert])
        .sign(setup["ca_key"], algorithm=None)
    )
    _write_ocsp_response(ocsps[0], response, setup["ca_key"], NOW)
    with pytest.raises(
        native.NativeStatusError, match="EXTRANEOUS_RESPONDER_CERTIFICATES"
    ):
        native.initialize_native_status(
            lifecycle_dir=lifecycle_dir,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            crl_paths=[crl],
            ocsp_paths=ocsps,
            output_dir=tmp_path / "bad",
            now=NOW,
        )


def test_run159_aggregate_raw_evidence_is_bounded_before_parsing(tmp_path: Path, monkeypatch):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    crl, ocsps, _ = _evidence(tmp_path / "e", setup, lifecycle_dir)
    monkeypatch.setitem(native.POLICY, "max_total_raw_evidence_bytes", 1)
    with pytest.raises(native.NativeStatusError, match="TOTAL_SIZE_INVALID"):
        native.initialize_native_status(
            lifecycle_dir=lifecycle_dir,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            crl_paths=[crl],
            ocsp_paths=ocsps,
            output_dir=tmp_path / "bad",
            now=NOW,
        )


def test_run159_rejects_ocsp_revocation_time_after_this_update(tmp_path: Path):
    setup, lifecycle_dir, *_ = _lifecycle(tmp_path / "pred")
    crl, ocsps, leaves = _evidence(tmp_path / "e", setup, lifecycle_dir)
    ocsps[0] = _ocsp_one(
        ocsps[0],
        setup,
        leaves[0],
        status=ocsp.OCSPCertStatus.REVOKED,
        this_update=NOW,
        revocation_time=NOW + timedelta(hours=1),
        reason=x509.ReasonFlags.key_compromise,
    )
    with pytest.raises(native.NativeStatusError, match="OCSP_REVOCATION_FROM_FUTURE"):
        native.initialize_native_status(
            lifecycle_dir=lifecycle_dir,
            expected_bootstrap_root_sha256=setup["pin"],
            expected_recovery_root_sha256=setup["rr_pin"],
            expected_attestation_root_sha256=[setup["ca_pin"]],
            crl_paths=[crl],
            ocsp_paths=ocsps,
            output_dir=tmp_path / "bad",
            now=NOW,
        )
