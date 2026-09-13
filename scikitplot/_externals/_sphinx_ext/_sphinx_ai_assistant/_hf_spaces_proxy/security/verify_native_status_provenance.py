"""
Run 159: independently verify and preserve native certificate-status evidence.

The Run 158 threshold-signed status snapshot is a policy assertion.  This layer
recomputes the status of every attestation leaf from raw DER CRLs and raw DER OCSP
responses, preserves those exact bytes, rejects *any* source disagreement, and
requires the resulting decision to match Run 158 exactly.

Optional vendor-native evidence is accepted only through an explicitly configured
profile verifier adapter.  There are intentionally no permissive built-in vendor
adapters: an unconfigured profile fails closed.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import tomllib

try:
    from . import verify_attestation_lifecycle as lifecycle
except (ImportError, ValueError) as exc:
    import importlib.util

    _here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_run158_lifecycle_for_native_status", _here / "verify_attestation_lifecycle.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("verify_attestation_lifecycle.py") from exc
    lifecycle = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lifecycle
    spec.loader.exec_module(lifecycle)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_native_status_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])
_ZERO_HASH = "0" * 64
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_CHUNK = 1024 * 1024
logger = logging.getLogger(__name__)


class NativeStatusError(RuntimeError):
    """Native status provenance invariant failed."""


def _fail(code: str) -> None:
    raise NativeStatusError(code)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(_canonical(value))


def _loads(raw: bytes, code: str) -> dict[str, Any]:
    def hook(pairs):
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                _fail(code + "_DUPLICATE_KEY")
            out[key] = value
        return out

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except NativeStatusError:
        raise
    except Exception as exc:
        raise NativeStatusError(code + "_JSON_INVALID") from exc
    if not isinstance(value, dict):
        _fail(code + "_SCHEMA_INVALID")
    return value


def _read_json(path: Path, code: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(code + "_INVALID")
    if path.stat().st_size > int(POLICY["max_json_bytes"]):
        _fail(code + "_TOO_LARGE")
    raw = path.read_bytes()
    doc = _loads(raw, code)
    if raw != _canonical(doc):
        _fail(code + "_NOT_CANONICAL")
    return doc, raw


def _raw(path: Path, code: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(code + "_INVALID")
    size = path.stat().st_size
    if size <= 0 or size > int(POLICY["max_raw_evidence_bytes"]):
        _fail(code + "_SIZE_INVALID")
    return path.read_bytes()


def _identity(value: Any, code: str) -> str:
    if not isinstance(value, str):
        _fail(code)
    value = value.strip()
    if (
        not value
        or len(value) > 512  # ruff: ignore[magic-value-comparison]
        or ".." in value
        or "?" in value
        or "#" in value
        or "\x00" in value
    ):
        _fail(code)
    if (
        any(
            ord(c) < 32  # ruff: ignore[magic-value-comparison]
            or ord(c) == 127  # ruff: ignore[magic-value-comparison]
            for c in value
        )
        or _ID.fullmatch(value) is None
    ):
        _fail(code)
    return value


def _hex(value: Any, code: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(code)
    return value


def _dt(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise NativeStatusError(code) from exc


def _ts(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _cert_time(value: Any) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _regular_dir(path: Path, code: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        _fail(code)
    return path.resolve()


def _outside(path: Path, protected: Iterable[Path], code: str) -> Path:
    target = path.expanduser().resolve()
    for item in protected:
        root = item.resolve()
        if target == root or root in target.parents:
            _fail(code)
    return target


def _artifact(name: str, raw: bytes) -> dict[str, Any]:
    return {"name": name, "sha256": _sha_bytes(raw), "size": len(raw)}


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_ATTESTATION_LIFECYCLE_STATE = "trusted-attestation-lifecycle-state.json"
_DOC_NATIVE_STATUS_STATE = "trusted-native-status-state.json"


def _lifecycle_docs(root: Path) -> dict[str, dict[str, Any]]:
    names = {
        "release-attestation-lifecycle-bundle.json",
        _DOC_ATTESTATION_LIFECYCLE_STATE,
        "active-attestation-ca-set.json",
        "active-attestation-status.json",
        "release-attestation-lifecycle-receipt.json",
    }
    if {p.name for p in root.iterdir()} != names:
        _fail("NATIVE_PREDECESSOR_ALLOWLIST_MISMATCH")
    return {
        name: _read_json(
            root / name,
            "NATIVE_PREDECESSOR_" + name.upper().replace("-", "_").replace(".", "_"),
        )[0]
        for name in sorted(names)
    }


def _verify_predecessor(
    root: Path,
    *,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    now: datetime,
    historical: bool,
) -> dict[str, dict[str, Any]]:
    try:
        lifecycle.verify_lifecycle(
            output_dir=root,
            expected_bootstrap_root_sha256=bootstrap_pin,
            expected_recovery_root_sha256=recovery_pin,
            expected_attestation_root_sha256=attestation_pins,
            now=now,
            historical=historical,
        )
    except lifecycle.AttestationLifecycleError as exc:
        raise NativeStatusError("NATIVE_PREDECESSOR_INVALID:" + str(exc)) from exc
    return _lifecycle_docs(root)


def _verify_signature(
    public_key: Any, signature: bytes, data: bytes, hash_algorithm: Any, code: str
) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric import (  # ruff: ignore[import-outside-top-level]
            ec,
            ed448,
            ed25519,
            padding,
            rsa,
        )

        if isinstance(public_key, (ed25519.Ed25519PublicKey, ed448.Ed448PublicKey)):
            public_key.verify(signature, data)
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, data, padding.PKCS1v15(), hash_algorithm)
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, data, ec.ECDSA(hash_algorithm))
        else:
            _fail(code + "_UNSUPPORTED_KEY")
    except NativeStatusError:
        raise
    except Exception as exc:
        raise NativeStatusError(code + "_SIGNATURE_INVALID") from exc


def _ca_records(ca_set: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]
    except Exception as exc:
        raise NativeStatusError("NATIVE_CRYPTOGRAPHY_UNAVAILABLE") from exc
    out = []
    for item in ca_set["signed"]["trustRoots"]:
        raw = base64.b64decode(item["der"], validate=True)
        cert = x509.load_der_x509_certificate(raw)
        out.append(
            {
                "sha256": item["sha256"],
                "der": raw,
                "cert": cert,
                "profile": item["profile"],
            }
        )
    return out


def _leaf_inventory(
    bundle: dict[str, Any], ca_set: dict[str, Any]
) -> list[dict[str, Any]]:
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]
    except Exception as exc:
        raise NativeStatusError("NATIVE_CRYPTOGRAPHY_UNAVAILABLE") from exc
    continuity_bundle = bundle["continuityOutput"][
        "release-root-continuity-bundle.json"
    ]
    ca_records = _ca_records(ca_set)
    run158_inventory, _ = lifecycle._inventory(continuity_bundle, ca_set=ca_set)
    run158_by_sha = {x["sha256"]: x for x in run158_inventory}
    leaves: dict[str, dict[str, Any]] = {}
    for att in lifecycle._attestations_from_continuity(continuity_bundle):
        s = att["signed"]
        chain = s.get("certificateChainDer")
        if not isinstance(chain, list) or not chain:
            _fail("NATIVE_ATTESTATION_CHAIN_INVALID")
        raw = base64.b64decode(chain[0], validate=True)
        cert = x509.load_der_x509_certificate(raw)
        sha = _sha_bytes(raw)
        if sha not in run158_by_sha:
            _fail("NATIVE_LEAF_NOT_IN_RUN158_INVENTORY")
        matches = [ca for ca in ca_records if cert.issuer == ca["cert"].subject]
        valid = []
        for ca in matches:
            try:
                lifecycle.continuity._verify_cert_signature(
                    cert, ca["cert"].public_key(), "NATIVE_LEAF_CA"
                )
                valid.append(ca)
            except Exception:  # ruff: ignore[blind-except, try-except-in-loop]
                pass
        if len(valid) != 1:
            _fail("NATIVE_LEAF_ISSUER_AMBIGUOUS")
        leaves[sha] = {
            "sha256": sha,
            "der": raw,
            "cert": cert,
            "serialNumber": format(cert.serial_number, "x"),
            "keyId": s["keyId"],
            "attestedAt": s["attestedAt"],
            "issuerCaSha256": valid[0]["sha256"],
            "issuerCert": valid[0]["cert"],
            "issuerDer": valid[0]["der"],
        }
    return [leaves[k] for k in sorted(leaves)]


def _reason_name(reason: Any) -> str | None:
    if reason is None:
        return None
    name = getattr(reason, "name", str(reason)).replace("_", "-")
    return name  # ruff: ignore[unnecessary-assign]


def _run158_decisions(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for item in status["signed"]["certificates"]:
        out[item["sha256"]] = {
            "status": item["status"],
            "revokedAt": item["revokedAt"],
            "reason": item["reason"],
        }
    return out


def _verify_crl(  # ruff: ignore[too-many-branches]
    raw: bytes,
    *,
    leaves: list[dict[str, Any]],
    ca_records: list[dict[str, Any]],
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]

        crl = x509.load_der_x509_crl(raw)
    except Exception as exc:
        raise NativeStatusError("NATIVE_CRL_DER_INVALID") from exc
    matches = [ca for ca in ca_records if crl.issuer == ca["cert"].subject]
    if len(matches) != 1:
        _fail("NATIVE_CRL_ISSUER_INVALID")
    ca = matches[0]
    try:
        crl.extensions.get_extension_for_class(x509.DeltaCRLIndicator)
    except x509.ExtensionNotFound:
        pass
    else:
        _fail("NATIVE_CRL_DELTA_UNSUPPORTED")
    try:
        crl.extensions.get_extension_for_class(x509.IssuingDistributionPoint)
    except x509.ExtensionNotFound:
        pass
    else:
        # A scoped/indirect CRL cannot prove that absence means good for every leaf.
        _fail("NATIVE_CRL_SCOPED_OR_INDIRECT_UNSUPPORTED")
    _verify_signature(
        ca["cert"].public_key(),
        crl.signature,
        crl.tbs_certlist_bytes,
        crl.signature_hash_algorithm,
        "NATIVE_CRL",
    )
    this_update = _cert_time(crl.last_update_utc)
    next_update_raw = crl.next_update_utc
    if next_update_raw is None:
        _fail("NATIVE_CRL_NEXT_UPDATE_MISSING")
    next_update = _cert_time(next_update_raw)
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if this_update > now + skew:
        _fail("NATIVE_CRL_FROM_FUTURE")
    if next_update <= this_update or next_update - this_update > timedelta(
        hours=int(POLICY["max_crl_lifetime_hours"])
    ):
        _fail("NATIVE_CRL_LIFETIME_INVALID")
    ca_nb = lifecycle.continuity._cert_time(ca["cert"], "not_valid_before")
    ca_na = lifecycle.continuity._cert_time(ca["cert"], "not_valid_after")
    if this_update < ca_nb or next_update > ca_na:
        _fail("NATIVE_CRL_OUTSIDE_ISSUER_VALIDITY")
    if not historical and next_update < now + timedelta(
        minutes=int(POLICY["min_native_remaining_minutes"])
    ):
        _fail("NATIVE_CRL_EXPIRED_OR_FREEZE_RISK")
    try:
        number = crl.extensions.get_extension_for_class(x509.CRLNumber).value.crl_number
    except x509.ExtensionNotFound:
        if bool(POLICY["require_crl_number"]):
            _fail("NATIVE_CRL_NUMBER_MISSING")
        number = 0
    if not isinstance(number, int) or number < 0:
        _fail("NATIVE_CRL_NUMBER_INVALID")
    revoked_entries = list(crl)
    if len(revoked_entries) > int(POLICY["max_crl_entries"]):
        _fail("NATIVE_CRL_ENTRY_COUNT_INVALID")
    serials = [entry.serial_number for entry in revoked_entries]
    if len(set(serials)) != len(serials):
        _fail("NATIVE_CRL_DUPLICATE_SERIAL")
    revoked_by_serial = {entry.serial_number: entry for entry in revoked_entries}
    decisions = []
    for leaf in leaves:
        if leaf["issuerCaSha256"] != ca["sha256"]:
            continue
        entry = revoked_by_serial.get(leaf["cert"].serial_number)
        if entry is None:
            decision = {
                "certSha256": leaf["sha256"],
                "serialNumber": leaf["serialNumber"],
                "status": "good",
                "revokedAt": None,
                "reason": None,
            }
        else:
            when = _cert_time(entry.revocation_date_utc)
            if when > this_update + skew:
                _fail("NATIVE_CRL_REVOCATION_FROM_FUTURE")
            reason = None
            try:
                reason = _reason_name(
                    entry.extensions.get_extension_for_class(
                        x509.CRLReason
                    ).value.reason
                )
            except x509.ExtensionNotFound:
                reason = "unspecified"
            if reason == "remove-from-crl":
                _fail("NATIVE_CRL_REMOVE_FROM_CRL_INVALID_IN_FULL_CRL")
            decision = {
                "certSha256": leaf["sha256"],
                "serialNumber": leaf["serialNumber"],
                "status": "revoked",
                "revokedAt": _ts(when),
                "reason": reason,
            }
        decisions.append(decision)
    decisions.sort(key=lambda x: x["certSha256"])
    return {
        "type": "crl",
        "sha256": _sha_bytes(raw),
        "der": base64.b64encode(raw).decode("ascii"),
        "issuerCaSha256": ca["sha256"],
        "sourceId": "crl:" + ca["sha256"],
        "thisUpdate": _ts(this_update),
        "nextUpdate": _ts(next_update),
        "crlNumber": number,
        "decisions": decisions,
    }


def _ocsp_status_name(value: Any) -> str:
    from cryptography.x509 import ocsp  # ruff: ignore[import-outside-top-level]

    if value == ocsp.OCSPCertStatus.GOOD:
        return "good"
    if value == ocsp.OCSPCertStatus.REVOKED:
        return "revoked"
    _fail("NATIVE_OCSP_CERT_STATUS_UNKNOWN")
    raise AssertionError


def _verify_ocsp(  # ruff: ignore[too-many-branches]
    raw: bytes,
    *,
    leaves: list[dict[str, Any]],
    ca_records: list[dict[str, Any]],
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]
        from cryptography.x509 import ocsp  # ruff: ignore[import-outside-top-level]

        response = ocsp.load_der_ocsp_response(raw)
    except Exception as exc:
        raise NativeStatusError("NATIVE_OCSP_DER_INVALID") from exc
    if response.response_status != ocsp.OCSPResponseStatus.SUCCESSFUL:
        _fail("NATIVE_OCSP_RESPONSE_NOT_SUCCESSFUL")
    responses = list(response.responses)
    if len(responses) != 1:
        _fail("NATIVE_OCSP_SINGLE_RESPONSE_REQUIRED")
    single = responses[0]
    leaf_matches = [
        leaf for leaf in leaves if leaf["cert"].serial_number == single.serial_number
    ]
    if len(leaf_matches) != 1:
        _fail("NATIVE_OCSP_SERIAL_NOT_IN_INVENTORY")
    leaf = leaf_matches[0]
    ca_matches = [ca for ca in ca_records if ca["sha256"] == leaf["issuerCaSha256"]]
    if len(ca_matches) != 1:
        _fail("NATIVE_OCSP_ISSUER_CA_MISSING")
    ca = ca_matches[0]
    try:
        req = (
            ocsp.OCSPRequestBuilder()
            .add_certificate(leaf["cert"], ca["cert"], single.hash_algorithm)
            .build()
        )
        if (
            req.issuer_name_hash != single.issuer_name_hash
            or req.issuer_key_hash != single.issuer_key_hash
        ):
            _fail("NATIVE_OCSP_ISSUER_HASH_MISMATCH")
    except NativeStatusError:
        raise
    except Exception as exc:
        raise NativeStatusError("NATIVE_OCSP_ISSUER_HASH_INVALID") from exc
    produced = _cert_time(response.produced_at_utc)
    this_update = _cert_time(single.this_update_utc)
    next_raw = single.next_update_utc
    if next_raw is None and bool(POLICY["require_ocsp_next_update"]):
        _fail("NATIVE_OCSP_NEXT_UPDATE_MISSING")
    next_update = _cert_time(next_raw) if next_raw is not None else produced
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if produced > now + skew or this_update > now + skew:
        _fail("NATIVE_OCSP_FROM_FUTURE")
    if produced < this_update - skew:
        _fail("NATIVE_OCSP_PRODUCED_BEFORE_THIS_UPDATE")
    if next_update <= this_update or next_update - this_update > timedelta(
        hours=int(POLICY["max_ocsp_lifetime_hours"])
    ):
        _fail("NATIVE_OCSP_LIFETIME_INVALID")
    ca_nb = lifecycle.continuity._cert_time(ca["cert"], "not_valid_before")
    ca_na = lifecycle.continuity._cert_time(ca["cert"], "not_valid_after")
    if produced < ca_nb or next_update > ca_na:
        _fail("NATIVE_OCSP_OUTSIDE_ISSUER_VALIDITY")
    if not historical and next_update < now + timedelta(
        minutes=int(POLICY["min_native_remaining_minutes"])
    ):
        _fail("NATIVE_OCSP_EXPIRED_OR_FREEZE_RISK")
    responder_name = response.responder_name
    if bool(POLICY["require_ocsp_responder_name"]) and responder_name is None:
        _fail("NATIVE_OCSP_RESPONDER_NAME_REQUIRED")
    signer = None
    responder_der = None
    responder_kind = None
    embedded_all = list(response.certificates)
    if responder_name == ca["cert"].subject:
        if embedded_all:
            _fail("NATIVE_OCSP_EXTRANEOUS_RESPONDER_CERTIFICATES")
        signer = ca["cert"]
        responder_der = ca["der"]
        responder_kind = "issuer-ca"
    else:
        embedded = [
            cert
            for cert in embedded_all
            if responder_name is not None and cert.subject == responder_name
        ]
        if len(embedded_all) != 1 or len(embedded) != 1:
            _fail("NATIVE_OCSP_RESPONDER_CERT_MISSING_OR_AMBIGUOUS")
        signer = embedded[0]
        responder_der = (
            signer.public_bytes(lifecycle.continuity.serialization.Encoding.DER)
            if hasattr(lifecycle.continuity, "serialization")
            else None
        )
        if responder_der is None:
            from cryptography.hazmat.primitives import (  # ruff: ignore[import-outside-top-level]
                serialization,
            )

            responder_der = signer.public_bytes(serialization.Encoding.DER)
        try:
            lifecycle.continuity._verify_cert_signature(
                signer, ca["cert"].public_key(), "NATIVE_OCSP_RESPONDER"
            )
            bc = signer.extensions.get_extension_for_class(x509.BasicConstraints).value
            if bc.ca:
                _fail("NATIVE_OCSP_RESPONDER_CA_INVALID")
            eku = signer.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
            if x509.oid.ExtendedKeyUsageOID.OCSP_SIGNING not in eku:
                _fail("NATIVE_OCSP_RESPONDER_EKU_INVALID")
            ku = signer.extensions.get_extension_for_class(x509.KeyUsage).value
            if not ku.digital_signature:
                _fail("NATIVE_OCSP_RESPONDER_KEY_USAGE_INVALID")
            nb = lifecycle.continuity._cert_time(signer, "not_valid_before")
            na = lifecycle.continuity._cert_time(signer, "not_valid_after")
            if not (nb <= produced <= na):
                _fail("NATIVE_OCSP_RESPONDER_CERT_NOT_VALID_AT_PRODUCTION")
        except NativeStatusError:
            raise
        except Exception as exc:
            raise NativeStatusError("NATIVE_OCSP_RESPONDER_CERT_INVALID") from exc
        responder_kind = "delegated"
    _verify_signature(
        signer.public_key(),
        response.signature,
        response.tbs_response_bytes,
        response.signature_hash_algorithm,
        "NATIVE_OCSP",
    )
    status = _ocsp_status_name(single.certificate_status)
    revoked_at = None
    reason = None
    if status == "revoked":
        if single.revocation_time_utc is None:
            _fail("NATIVE_OCSP_REVOCATION_TIME_MISSING")
        revoked_dt = _cert_time(single.revocation_time_utc)
        if revoked_dt > this_update + skew:
            _fail("NATIVE_OCSP_REVOCATION_FROM_FUTURE")
        revoked_at = _ts(revoked_dt)
        reason = _reason_name(single.revocation_reason) or "unspecified"
    decision = {
        "certSha256": leaf["sha256"],
        "serialNumber": leaf["serialNumber"],
        "status": status,
        "revokedAt": revoked_at,
        "reason": reason,
    }
    return {
        "type": "ocsp",
        "sha256": _sha_bytes(raw),
        "der": base64.b64encode(raw).decode("ascii"),
        "issuerCaSha256": ca["sha256"],
        "sourceId": "ocsp:" + _sha_bytes(responder_der),
        "responderKind": responder_kind,
        "responderCertSha256": _sha_bytes(responder_der),
        "responderCertDer": base64.b64encode(responder_der).decode("ascii"),
        "producedAt": _ts(produced),
        "thisUpdate": _ts(this_update),
        "nextUpdate": _ts(next_update),
        "decisions": [decision],
    }


def _normalize_decision(
    status: str, revoked_at: str | None, reason: str | None
) -> tuple[str, str | None, str | None]:
    if status == "good":
        return ("good", None, None)
    return ("revoked", revoked_at, (reason or "unspecified").replace("_", "-"))


def _aggregate(
    *,
    leaves: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    run158_status: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = _run158_decisions(run158_status)
    required_kinds = {str(x) for x in POLICY["required_source_kinds"]}
    result = []
    for leaf in leaves:
        observations = []
        for source in sources:
            observations.extend(
                {
                    "kind": source["type"],
                    "sourceId": source["sourceId"],
                    **decision,
                }
                for decision in source["decisions"]
                if decision["certSha256"] == leaf["sha256"]
            )
        kinds = {x["kind"] for x in observations}
        if bool(POLICY["require_all_active_leaf_certificates"]) and (
            len(kinds) < int(POLICY["min_source_kinds_per_leaf"])
            or not required_kinds.issubset(kinds)
        ):
            _fail("NATIVE_STATUS_SOURCE_COVERAGE_INCOMPLETE")
        normalized = {
            _normalize_decision(x["status"], x["revokedAt"], x["reason"])
            for x in observations
        }
        if len(normalized) != 1:
            _fail("NATIVE_STATUS_SOURCE_CONFLICT")
        decision = next(iter(normalized))
        prior = expected.get(leaf["sha256"])
        if prior is None:
            _fail("NATIVE_STATUS_RUN158_DECISION_MISSING")
        prior_norm = _normalize_decision(
            prior["status"], prior["revokedAt"], prior["reason"]
        )
        # Run 158 reason strings are policy labels and may be more specific than a native source.
        if bool(POLICY["require_native_match_run158"]):  # ruff: ignore[collapsible-if]
            if decision[0] != prior_norm[0] or decision[1] != prior_norm[1]:
                _fail("NATIVE_STATUS_RUN158_MISMATCH")
        result.append(
            {
                "certSha256": leaf["sha256"],
                "serialNumber": leaf["serialNumber"],
                "keyId": leaf["keyId"],
                "status": decision[0],
                "revokedAt": decision[1],
                "sourceKinds": sorted(kinds),
                "sourceIds": sorted({x["sourceId"] for x in observations}),
                "observationCount": len(observations),
            }
        )
    result.sort(key=lambda x: x["certSha256"])
    return result


def _verify_source_continuity(
    previous_event: dict[str, Any] | None, sources: list[dict[str, Any]]
) -> None:
    if previous_event is None:
        return
    old_sources = previous_event.get("sources")
    if not isinstance(old_sources, list):
        _fail("NATIVE_PREVIOUS_SOURCES_INVALID")
    old_crl = {x["issuerCaSha256"]: x for x in old_sources if x.get("type") == "crl"}
    old_ocsp = {}
    for x in old_sources:
        if x.get("type") != "ocsp":
            continue
        for d in x.get("decisions", []):
            old_ocsp[(x["sourceId"], d["certSha256"])] = x
    for source in sources:
        if source["type"] == "crl":
            old = old_crl.get(source["issuerCaSha256"])
            if old is not None:
                if source["crlNumber"] <= old["crlNumber"]:
                    _fail("NATIVE_CRL_NUMBER_NOT_MONOTONIC")
                if _dt(source["thisUpdate"], "NATIVE_CRL_THIS_UPDATE_INVALID") <= _dt(
                    old["thisUpdate"], "NATIVE_OLD_CRL_THIS_UPDATE_INVALID"
                ):
                    _fail("NATIVE_CRL_TIME_NOT_MONOTONIC")
        else:
            d = source["decisions"][0]
            old = old_ocsp.get((source["sourceId"], d["certSha256"]))
            if old is not None and _dt(
                source["thisUpdate"], "NATIVE_OCSP_THIS_UPDATE_INVALID"
            ) <= _dt(old["thisUpdate"], "NATIVE_OLD_OCSP_THIS_UPDATE_INVALID"):
                _fail("NATIVE_OCSP_TIME_NOT_MONOTONIC")


def _vendor_verifier(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        _fail("NATIVE_VENDOR_VERIFIER_SPEC_INVALID")
    profile, raw_path = spec.split("=", 1)
    profile = _identity(profile, "NATIVE_VENDOR_PROFILE_INVALID")
    path = Path(raw_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        _fail("NATIVE_VENDOR_VERIFIER_EXECUTABLE_INVALID")
    return profile, path


def _verify_vendor_manifest(  # ruff: ignore[too-many-branches]
    path: Path,
    verifiers: dict[str, Path],
) -> dict[str, Any]:
    doc, raw = _read_json(path, "NATIVE_VENDOR_MANIFEST")
    if (
        set(doc)
        != {
            "schemaVersion",
            "profile",
            "evidenceType",
            "rawEvidenceBase64",
            "expectedKeyId",
            "expectedPublicKeySha256",
        }
        or doc["schemaVersion"] != 1
    ):
        _fail("NATIVE_VENDOR_MANIFEST_SCHEMA_INVALID")
    profile = _identity(doc["profile"], "NATIVE_VENDOR_PROFILE_INVALID")
    verifier = verifiers.get(profile)
    if verifier is None:
        _fail("NATIVE_VENDOR_PROFILE_VERIFIER_NOT_CONFIGURED")
    try:
        evidence = base64.b64decode(doc["rawEvidenceBase64"], validate=True)
    except Exception as exc:
        raise NativeStatusError("NATIVE_VENDOR_RAW_EVIDENCE_INVALID") from exc
    if not evidence or len(evidence) > int(POLICY["max_raw_evidence_bytes"]):
        _fail("NATIVE_VENDOR_RAW_EVIDENCE_SIZE_INVALID")
    request = _canonical(
        {
            "profile": profile,
            "evidenceType": _identity(
                doc["evidenceType"], "NATIVE_VENDOR_EVIDENCE_TYPE_INVALID"
            ),
            "rawEvidenceSha256": _sha_bytes(evidence),
            "rawEvidenceBase64": base64.b64encode(evidence).decode("ascii"),
            "expectedKeyId": _identity(
                doc["expectedKeyId"], "NATIVE_VENDOR_KEY_ID_INVALID"
            ),
            "expectedPublicKeySha256": _hex(
                doc["expectedPublicKeySha256"], "NATIVE_VENDOR_PUBLIC_HASH_INVALID"
            ),
        }
    )
    limit = int(POLICY["vendor_verifier_max_output_bytes"])
    proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        [str(verifier)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
    )
    assert proc.stdin is not None  # ruff: ignore[assert]
    assert proc.stdout is not None  # ruff: ignore[assert]
    chunks: list[bytes] = []
    state: dict[str, Any] = {"size": 0, "overflow": False, "error": None}

    def drain() -> None:
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    return
                state["size"] += len(chunk)
                if state["size"] > limit:
                    state["overflow"] = True
                    try:  # ruff: ignore[suppressible-exception]
                        proc.kill()
                    except OSError:
                        pass
                    return
                chunks.append(chunk)
        # defensive pipe failure
        except BaseException as exc:  # pragma: no cover  # ruff: ignore[blind-except]
            state["error"] = exc
            with contextlib.suppress(OSError):
                proc.kill()

    reader = threading.Thread(
        target=drain, name="native-vendor-verifier-stdout", daemon=True
    )
    reader.start()
    try:
        try:
            proc.stdin.write(request)
            proc.stdin.close()
            proc.wait(timeout=int(POLICY["vendor_verifier_timeout_seconds"]))
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            reader.join(timeout=2)
            raise NativeStatusError("NATIVE_VENDOR_VERIFIER_TIMEOUT") from exc
        finally:
            try:  # ruff: ignore[suppressible-exception]
                proc.stdin.close()
            except OSError:
                pass
        reader.join(timeout=2)
        if reader.is_alive() or state["error"] is not None:
            _fail("NATIVE_VENDOR_VERIFIER_PIPE_FAILURE")
        if state["overflow"]:
            _fail("NATIVE_VENDOR_VERIFIER_OUTPUT_TOO_LARGE")
        if proc.returncode != 0:
            _fail("NATIVE_VENDOR_VERIFIER_FAILED")
        stdout = b"".join(chunks)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if reader.is_alive():
            reader.join(timeout=2)
        try:  # ruff: ignore[suppressible-exception]
            proc.stdout.close()
        except OSError:
            pass
    result = _loads(stdout, "NATIVE_VENDOR_VERIFIER_RESULT")
    expected = {
        "schemaVersion",
        "profile",
        "verifierIdentity",
        "rawEvidenceSha256",
        "keyId",
        "publicKeySha256",
        "verified",
    }
    if (
        set(result) != expected
        or result["schemaVersion"] != 1
        or result["verified"] is not True
    ):
        _fail("NATIVE_VENDOR_VERIFIER_RESULT_INVALID")
    if stdout != _canonical(result):
        _fail("NATIVE_VENDOR_VERIFIER_RESULT_NOT_CANONICAL")
    if (
        result["profile"] != profile
        or result["rawEvidenceSha256"] != _sha_bytes(evidence)
        or result["keyId"] != doc["expectedKeyId"]
        or result["publicKeySha256"] != doc["expectedPublicKeySha256"]
    ):
        _fail("NATIVE_VENDOR_VERIFIER_RESULT_MISMATCH")
    return {
        "manifestSha256": _sha_bytes(raw),
        "profile": profile,
        "evidenceType": doc["evidenceType"],
        "rawEvidenceSha256": _sha_bytes(evidence),
        "rawEvidenceBase64": base64.b64encode(evidence).decode("ascii"),
        "verifierExecutableSha256": _sha(verifier),
        "result": result,
    }


def _validate_preserved_vendor(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        _fail("NATIVE_VENDOR_EVIDENCE_LIST_INVALID")
    normalized = []
    for item in items:
        expected = {
            "manifestSha256",
            "profile",
            "evidenceType",
            "rawEvidenceSha256",
            "rawEvidenceBase64",
            "verifierExecutableSha256",
            "result",
        }
        if not isinstance(item, dict) or set(item) != expected:
            _fail("NATIVE_VENDOR_EVIDENCE_SCHEMA_INVALID")
        raw = base64.b64decode(item["rawEvidenceBase64"], validate=True)
        if _sha_bytes(raw) != _hex(
            item["rawEvidenceSha256"], "NATIVE_VENDOR_EVIDENCE_HASH_INVALID"
        ):
            _fail("NATIVE_VENDOR_EVIDENCE_HASH_MISMATCH")
        _hex(item["manifestSha256"], "NATIVE_VENDOR_MANIFEST_HASH_INVALID")
        _hex(item["verifierExecutableSha256"], "NATIVE_VENDOR_VERIFIER_HASH_INVALID")
        _identity(item["profile"], "NATIVE_VENDOR_PROFILE_INVALID")
        _identity(item["evidenceType"], "NATIVE_VENDOR_EVIDENCE_TYPE_INVALID")
        result = item["result"]
        if (
            not isinstance(result, dict)
            or result.get("verified") is not True
            or result.get("rawEvidenceSha256") != item["rawEvidenceSha256"]
            or result.get("profile") != item["profile"]
        ):
            _fail("NATIVE_VENDOR_PRESERVED_RESULT_INVALID")
        normalized.append(item)
    expected_order = sorted(
        normalized, key=lambda x: (x["profile"], x["rawEvidenceSha256"])
    )
    if normalized != expected_order:
        _fail("NATIVE_VENDOR_EVIDENCE_NOT_NORMALIZED")
    return normalized


def _event_head(previous: str, body: dict[str, Any]) -> str:
    return _sha_bytes(_canonical({"previousChainHeadSha256": previous, "body": body}))


def _input_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(p.resolve()): _sha(p) for p in paths}


def _check_inputs(before: dict[str, str]) -> None:
    for raw, expected in before.items():
        path = Path(raw)
        if not path.is_file() or path.is_symlink() or _sha(path) != expected:
            _fail("NATIVE_INPUT_DRIFT_DETECTED")


def _write_output(
    target: Path,
    *,
    bundle: dict[str, Any],
    state: dict[str, Any],
    event: dict[str, Any],
    inputs: dict[str, str],
    phase: str,
) -> None:
    _check_inputs(inputs)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".native-status-", dir=parent))
    try:
        _write(tmp / "release-native-status-bundle.json", bundle)
        _write(tmp / _DOC_NATIVE_STATUS_STATE, state)
        _write(tmp / "active-native-status-evidence.json", event)
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "phase": phase,
            "bundle": _artifact(
                "release-native-status-bundle.json", _canonical(bundle)
            ),
            "state": _artifact(_DOC_NATIVE_STATUS_STATE, _canonical(state)),
            "activeEvidence": _artifact(
                "active-native-status-evidence.json", _canonical(event)
            ),
        }
        _write(tmp / "release-native-status-receipt.json", receipt)
        _check_inputs(inputs)
        os.replace(tmp, target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _build_sources(
    *,
    crl_paths: list[Path],
    ocsp_paths: list[Path],
    leaves: list[dict[str, Any]],
    ca_records: list[dict[str, Any]],
    now: datetime,
    historical: bool,
) -> list[dict[str, Any]]:
    if len(crl_paths) + len(ocsp_paths) > int(POLICY["max_raw_evidence_files"]):
        _fail("NATIVE_EVIDENCE_FILE_COUNT_INVALID")
    total = 0
    for path in [*crl_paths, *ocsp_paths]:
        if path.is_symlink() or not path.is_file():
            _fail("NATIVE_EVIDENCE_FILE_INVALID")
        total += path.stat().st_size
        if total > int(POLICY["max_total_raw_evidence_bytes"]):
            _fail("NATIVE_EVIDENCE_TOTAL_SIZE_INVALID")
    sources = [
        _verify_crl(
            _raw(path, "NATIVE_CRL_FILE"),
            leaves=leaves,
            ca_records=ca_records,
            now=now,
            historical=historical,
        )
        for path in crl_paths
    ]
    sources.extend(
        _verify_ocsp(
            _raw(path, "NATIVE_OCSP_FILE"),
            leaves=leaves,
            ca_records=ca_records,
            now=now,
            historical=historical,
        )
        for path in ocsp_paths
    )
    sources.sort(key=lambda x: (x["type"], x["sourceId"], x["sha256"]))
    if len({(x["type"], x["sourceId"], x["sha256"]) for x in sources}) != len(sources):
        _fail("NATIVE_EVIDENCE_DUPLICATE")
    crl_ids = [x["sourceId"] for x in sources if x["type"] == "crl"]
    if len(set(crl_ids)) != len(crl_ids):
        _fail("NATIVE_CRL_SOURCE_AMBIGUOUS")
    ocsp_ids = [
        (x["sourceId"], x["decisions"][0]["certSha256"])
        for x in sources
        if x["type"] == "ocsp"
    ]
    if len(set(ocsp_ids)) != len(ocsp_ids):
        _fail("NATIVE_OCSP_SOURCE_AMBIGUOUS")
    return sources


def _predecessor_binding(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    state = docs[_DOC_ATTESTATION_LIFECYCLE_STATE]
    bundle = docs["release-attestation-lifecycle-bundle.json"]
    return {
        "bundle": _artifact(
            "release-attestation-lifecycle-bundle.json", _canonical(bundle)
        ),
        "state": _artifact(_DOC_ATTESTATION_LIFECYCLE_STATE, _canonical(state)),
        "lifecycleChainHeadSha256": _hex(
            state["lifecycleChainHeadSha256"], "NATIVE_LIFECYCLE_HEAD_INVALID"
        ),
        "activeCaSetSha256": _sha_bytes(
            _canonical(docs["active-attestation-ca-set.json"])
        ),
        "activeStatusSha256": _sha_bytes(
            _canonical(docs["active-attestation-status.json"])
        ),
    }


def initialize_native_status(  # ruff: ignore[undocumented-public-function]
    *,
    lifecycle_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    crl_paths: list[Path],
    ocsp_paths: list[Path],
    output_dir: Path,
    vendor_manifest_paths: list[Path] | None = None,
    vendor_verifier_specs: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    pred = _regular_dir(lifecycle_dir, "NATIVE_LIFECYCLE_DIR_INVALID")
    docs = _verify_predecessor(
        pred,
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=expected_attestation_root_sha256,
        now=current,
        historical=False,
    )
    lifecycle_bundle = docs["release-attestation-lifecycle-bundle.json"]
    ca_set = docs["active-attestation-ca-set.json"]
    leaves = _leaf_inventory(lifecycle_bundle, ca_set)
    ca_records = _ca_records(ca_set)
    sources = _build_sources(
        crl_paths=crl_paths,
        ocsp_paths=ocsp_paths,
        leaves=leaves,
        ca_records=ca_records,
        now=current,
        historical=False,
    )
    decisions = _aggregate(
        leaves=leaves,
        sources=sources,
        run158_status=docs["active-attestation-status.json"],
    )
    _verify_source_continuity(None, sources)
    verifiers = dict(_vendor_verifier(spec) for spec in vendor_verifier_specs or [])
    vendor = [
        _verify_vendor_manifest(p, verifiers) for p in vendor_manifest_paths or []
    ]
    vendor.sort(key=lambda x: (x["profile"], x["rawEvidenceSha256"]))
    predecessor = _predecessor_binding(docs)
    body = {
        "sequence": 1,
        "type": "native-status-bootstrap",
        "previousChainHeadSha256": predecessor["lifecycleChainHeadSha256"],
        "predecessor": predecessor,
        "sources": sources,
        "decisions": decisions,
        "vendorEvidence": vendor,
    }
    head = _event_head(body["previousChainHeadSha256"], body)
    event = dict(body, nativeStatusChainHeadSha256=head)
    bundle = {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "trusted-native-status-provenance",
        "bootstrapRootSha256": _hex(
            expected_bootstrap_root_sha256, "NATIVE_BOOTSTRAP_PIN_INVALID"
        ),
        "recoveryRootSha256": _hex(
            expected_recovery_root_sha256, "NATIVE_RECOVERY_PIN_INVALID"
        ),
        "attestationRootSha256": sorted(
            _hex(x, "NATIVE_ATTESTATION_PIN_INVALID")
            for x in expected_attestation_root_sha256
        ),
        "attestationLifecycleOutput": docs,
        "events": [event],
    }
    state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-native-status-provenance",
        "sequence": 1,
        "lifecycleChainHeadSha256": predecessor["lifecycleChainHeadSha256"],
        "nativeStatusChainHeadSha256": head,
        "activeEvidenceSha256": _sha_bytes(_canonical(event)),
        "bundleSha256": _sha_bytes(_canonical(bundle)),
    }
    protected = [pred, *crl_paths, *ocsp_paths, *(vendor_manifest_paths or [])]
    target = _outside(output_dir, protected, "NATIVE_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("NATIVE_OUTPUT_ALREADY_EXISTS")
    inputs = (
        [p for p in pred.rglob("*") if p.is_file()]
        + list(crl_paths)
        + list(ocsp_paths)
        + list(vendor_manifest_paths or [])
        + list(verifiers.values())
    )
    before = _input_hashes(inputs)
    _write_output(
        target,
        bundle=bundle,
        state=state,
        event=event,
        inputs=before,
        phase="native-status-initialized",
    )
    verify_native_status(
        output_dir=target,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    return {
        "ok": True,
        "phase": "native-status-initialized",
        "sequence": 1,
        "native_status_chain_head_sha256": head,
    }


def _native_docs(root: Path) -> dict[str, dict[str, Any]]:
    names = {
        "release-native-status-bundle.json",
        _DOC_NATIVE_STATUS_STATE,
        "active-native-status-evidence.json",
        "release-native-status-receipt.json",
    }
    if {p.name for p in root.iterdir()} != names:
        _fail("NATIVE_OUTPUT_ALLOWLIST_MISMATCH")
    return {
        name: _read_json(
            root / name,
            "NATIVE_OUTPUT_" + name.upper().replace("-", "_").replace(".", "_"),
        )[0]
        for name in sorted(names)
    }


def _verify_bundle(  # ruff: ignore[too-many-branches]
    bundle: dict[str, Any],
    *,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    now: datetime,
    historical: bool,
) -> tuple[dict[str, Any], str]:
    expected_top = {
        "schemaVersion",
        "predicateType",
        "status",
        "bootstrapRootSha256",
        "recoveryRootSha256",
        "attestationRootSha256",
        "attestationLifecycleOutput",
        "events",
    }
    if (
        set(bundle) != expected_top
        or bundle["schemaVersion"] != int(POLICY["bundle_schema_version"])
        or bundle["predicateType"] != PREDICATE_TYPE
        or bundle["status"] != "trusted-native-status-provenance"
    ):
        _fail("NATIVE_BUNDLE_SCHEMA_INVALID")
    if (
        bundle["bootstrapRootSha256"] != bootstrap_pin
        or bundle["recoveryRootSha256"] != recovery_pin
        or bundle["attestationRootSha256"] != sorted(attestation_pins)
    ):
        _fail("NATIVE_BUNDLE_PIN_MISMATCH")
    # Rehydrate and independently re-run Run 158 from canonical embedded bytes.
    tmp = Path(tempfile.mkdtemp(prefix=".native-predecessor-"))
    try:
        docs = bundle["attestationLifecycleOutput"]
        if not isinstance(docs, dict):
            _fail("NATIVE_EMBEDDED_PREDECESSOR_INVALID")
        for name, doc in docs.items():
            _write(tmp / name, doc)
        pred_docs = _verify_predecessor(
            tmp,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            now=now,
            historical=historical,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    events = bundle["events"]
    if (
        not isinstance(events, list)
        or not events
        or len(events) > int(POLICY["max_events"])
    ):
        _fail("NATIVE_EVENTS_INVALID")
    ca_set = pred_docs["active-attestation-ca-set.json"]
    lifecycle_bundle = pred_docs["release-attestation-lifecycle-bundle.json"]
    leaves = _leaf_inventory(lifecycle_bundle, ca_set)
    ca_records = _ca_records(ca_set)
    previous_head = pred_docs[_DOC_ATTESTATION_LIFECYCLE_STATE][
        "lifecycleChainHeadSha256"
    ]
    previous_event = None
    final_event = None
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict) or set(event) != {
            "sequence",
            "type",
            "previousChainHeadSha256",
            "predecessor",
            "sources",
            "decisions",
            "vendorEvidence",
            "nativeStatusChainHeadSha256",
        }:
            _fail("NATIVE_EVENT_SCHEMA_INVALID")
        if event["sequence"] != index or event["type"] not in {
            "native-status-bootstrap",
            "native-status-advance",
        }:
            _fail("NATIVE_EVENT_SEQUENCE_INVALID")
        if event["previousChainHeadSha256"] != previous_head:
            _fail("NATIVE_EVENT_PREVIOUS_HEAD_MISMATCH")
        if event["predecessor"] != _predecessor_binding(pred_docs):
            _fail("NATIVE_EVENT_PREDECESSOR_MISMATCH")
        verified_sources = []
        event_historical = historical or index < len(events)
        for source in event["sources"]:
            if not isinstance(source, dict) or source.get("type") not in {
                "crl",
                "ocsp",
            }:
                _fail("NATIVE_EVENT_SOURCE_SCHEMA_INVALID")
            raw = base64.b64decode(source.get("der", ""), validate=True)
            if source["type"] == "crl":
                verified = _verify_crl(
                    raw,
                    leaves=leaves,
                    ca_records=ca_records,
                    now=now,
                    historical=event_historical,
                )
            else:
                verified = _verify_ocsp(
                    raw,
                    leaves=leaves,
                    ca_records=ca_records,
                    now=now,
                    historical=event_historical,
                )
            if verified != source:
                _fail("NATIVE_EVENT_SOURCE_REBIND_MISMATCH")
            verified_sources.append(verified)
        _verify_source_continuity(previous_event, verified_sources)
        decisions = _aggregate(
            leaves=leaves,
            sources=verified_sources,
            run158_status=pred_docs["active-attestation-status.json"],
        )
        if decisions != event["decisions"]:
            _fail("NATIVE_EVENT_DECISION_MISMATCH")
        # Vendor evidence is preserved and hash-bound. Re-running a vendor binary is intentionally
        # not required for historical replay; its exact executable hash and canonical result are kept.
        _validate_preserved_vendor(event["vendorEvidence"])
        body = {k: event[k] for k in event if k != "nativeStatusChainHeadSha256"}
        head = _event_head(previous_head, body)
        if event["nativeStatusChainHeadSha256"] != head:
            _fail("NATIVE_EVENT_CHAIN_HEAD_MISMATCH")
        previous_head = head
        previous_event = event
        final_event = event
    assert final_event is not None  # ruff: ignore[assert]
    return final_event, previous_head


def verify_native_status(  # ruff: ignore[undocumented-public-function]
    *,
    output_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root = _regular_dir(output_dir, "NATIVE_OUTPUT_DIR_INVALID")
    docs = _native_docs(root)
    final_event, head = _verify_bundle(
        docs["release-native-status-bundle.json"],
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=sorted(expected_attestation_root_sha256),
        now=current,
        historical=historical,
    )
    state = docs[_DOC_NATIVE_STATUS_STATE]
    bundle = docs["release-native-status-bundle.json"]
    expected_state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-native-status-provenance",
        "sequence": len(bundle["events"]),
        "lifecycleChainHeadSha256": bundle["attestationLifecycleOutput"][
            _DOC_ATTESTATION_LIFECYCLE_STATE
        ]["lifecycleChainHeadSha256"],
        "nativeStatusChainHeadSha256": head,
        "activeEvidenceSha256": _sha_bytes(_canonical(final_event)),
        "bundleSha256": _sha_bytes(_canonical(bundle)),
    }
    if (
        state != expected_state
        or docs["active-native-status-evidence.json"] != final_event
    ):
        _fail("NATIVE_STATE_OR_ACTIVE_EVIDENCE_MISMATCH")
    receipt = docs["release-native-status-receipt.json"]
    expected_receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "phase": (
            "native-status-initialized"
            if len(bundle["events"]) == 1
            else "native-status-advanced"
        ),
        "bundle": _artifact("release-native-status-bundle.json", _canonical(bundle)),
        "state": _artifact(_DOC_NATIVE_STATUS_STATE, _canonical(state)),
        "activeEvidence": _artifact(
            "active-native-status-evidence.json", _canonical(final_event)
        ),
    }
    if receipt != expected_receipt:
        _fail("NATIVE_RECEIPT_MISMATCH")
    return {
        "ok": True,
        "sequence": len(bundle["events"]),
        "native_status_chain_head_sha256": head,
    }


def advance_native_status(  # ruff: ignore[undocumented-public-function]
    *,
    previous_dir: Path,
    crl_paths: list[Path],
    ocsp_paths: list[Path],
    output_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    prev = _regular_dir(previous_dir, "NATIVE_PREVIOUS_DIR_INVALID")
    verify_native_status(
        output_dir=prev,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    docs = _native_docs(prev)
    bundle = json.loads(json.dumps(docs["release-native-status-bundle.json"]))
    pred_docs = bundle["attestationLifecycleOutput"]
    ca_set = pred_docs["active-attestation-ca-set.json"]
    lifecycle_bundle = pred_docs["release-attestation-lifecycle-bundle.json"]
    leaves = _leaf_inventory(lifecycle_bundle, ca_set)
    ca_records = _ca_records(ca_set)
    sources = _build_sources(
        crl_paths=crl_paths,
        ocsp_paths=ocsp_paths,
        leaves=leaves,
        ca_records=ca_records,
        now=current,
        historical=False,
    )
    previous_event = bundle["events"][-1]
    _verify_source_continuity(previous_event, sources)
    decisions = _aggregate(
        leaves=leaves,
        sources=sources,
        run158_status=pred_docs["active-attestation-status.json"],
    )
    seq = len(bundle["events"]) + 1
    predecessor = _predecessor_binding(pred_docs)
    body = {
        "sequence": seq,
        "type": "native-status-advance",
        "previousChainHeadSha256": docs[_DOC_NATIVE_STATUS_STATE][
            "nativeStatusChainHeadSha256"
        ],
        "predecessor": predecessor,
        "sources": sources,
        "decisions": decisions,
        "vendorEvidence": [],
    }
    head = _event_head(body["previousChainHeadSha256"], body)
    event = dict(body, nativeStatusChainHeadSha256=head)
    bundle["events"].append(event)
    state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-native-status-provenance",
        "sequence": seq,
        "lifecycleChainHeadSha256": predecessor["lifecycleChainHeadSha256"],
        "nativeStatusChainHeadSha256": head,
        "activeEvidenceSha256": _sha_bytes(_canonical(event)),
        "bundleSha256": _sha_bytes(_canonical(bundle)),
    }
    target = _outside(
        output_dir,
        [prev, *crl_paths, *ocsp_paths],
        "NATIVE_ADVANCE_OUTPUT_INSIDE_INPUT",
    )
    if target.exists() or target.is_symlink():
        _fail("NATIVE_ADVANCE_OUTPUT_ALREADY_EXISTS")
    inputs = (
        [p for p in prev.rglob("*") if p.is_file()] + list(crl_paths) + list(ocsp_paths)
    )
    before = _input_hashes(inputs)
    _write_output(
        target,
        bundle=bundle,
        state=state,
        event=event,
        inputs=before,
        phase="native-status-advanced",
    )
    verify_native_status(
        output_dir=target,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    return {
        "ok": True,
        "phase": "native-status-advanced",
        "sequence": seq,
        "native_status_chain_head_sha256": head,
    }


def _paths(values: list[str]) -> list[Path]:
    return [Path(x) for x in values]


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bootstrap-root-sha256", required=True)
    common.add_argument("--recovery-root-sha256", required=True)
    common.add_argument("--attestation-root-sha256", action="append", required=True)
    p = sub.add_parser("initialize", parents=[common])
    p.add_argument("--lifecycle-dir", type=Path, required=True)
    p.add_argument("--crl", action="append", default=[])
    p.add_argument("--ocsp", action="append", default=[])
    p.add_argument("--vendor-manifest", action="append", default=[])
    p.add_argument("--vendor-verifier", action="append", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("advance", parents=[common])
    p.add_argument("--previous-dir", type=Path, required=True)
    p.add_argument("--crl", action="append", default=[])
    p.add_argument("--ocsp", action="append", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("verify", parents=[common])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--historical", action="store_true")
    args = parser.parse_args(argv)
    kwargs = {
        "expected_bootstrap_root_sha256": args.bootstrap_root_sha256,
        "expected_recovery_root_sha256": args.recovery_root_sha256,
        "expected_attestation_root_sha256": args.attestation_root_sha256,
    }
    try:
        if args.command == "initialize":
            result = initialize_native_status(
                lifecycle_dir=args.lifecycle_dir,
                crl_paths=_paths(args.crl),
                ocsp_paths=_paths(args.ocsp),
                vendor_manifest_paths=_paths(args.vendor_manifest),
                vendor_verifier_specs=args.vendor_verifier,
                output_dir=args.output_dir,
                **kwargs,
            )
        elif args.command == "advance":
            result = advance_native_status(
                previous_dir=args.previous_dir,
                crl_paths=_paths(args.crl),
                ocsp_paths=_paths(args.ocsp),
                output_dir=args.output_dir,
                **kwargs,
            )
        else:
            result = verify_native_status(
                output_dir=args.output_dir, historical=args.historical, **kwargs
            )
    except NativeStatusError as exc:
        logger.error(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    logger.info(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
