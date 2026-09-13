"""
Run 158: preserve and verify hardware-attestation lifecycle status.

This layer consumes a verified Run 157 recovered-root continuity directory.  It does
not replace Run 157 certificate-chain verification.  Instead it adds four properties
that X.509 path validation alone does not provide:

* root-threshold governed attestation-CA/status-authority metadata;
* short-lived threshold-signed certificate status snapshots;
* certificate-bound vendor/device semantic claims; and
* deterministic historical proof of status at the exact attestation time.

No signing private key is read by this module.  Every signature is supplied as
canonical external evidence and verified in-process with Ed25519.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import tomllib

logger = logging.getLogger(__name__)

try:
    from . import continue_recovered_trust as continuity
    from . import seal_release_governance as rootseal
except (ImportError, ValueError):
    import importlib.util

    _here = Path(__file__).resolve().parent

    def _load(name: str, filename: str):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, _here / filename)
        if spec is None or spec.loader is None:
            raise ImportError(filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    continuity = _load(
        "_run157_continuity_for_lifecycle", "continue_recovered_trust.py"
    )
    rootseal = _load("_run155_root_for_lifecycle", "seal_release_governance.py")

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_attestation_lifecycle_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])
_ZERO_HASH = "0" * 64
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_CHUNK = 1024 * 1024


class AttestationLifecycleError(RuntimeError):
    """Attestation lifecycle, status, or vendor semantic invariant failed."""


def _fail(code: str) -> None:
    raise AttestationLifecycleError(code)


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
    except AttestationLifecycleError:
        raise
    except Exception as exc:
        raise AttestationLifecycleError(code + "_JSON_INVALID") from exc
    if not isinstance(value, dict):
        _fail(code + "_SCHEMA_INVALID")
    return value


def _read(
    path: Path, code: str, *, canonical: bool = True
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(code + "_INVALID")
    if path.stat().st_size > int(POLICY["max_json_bytes"]):
        _fail(code + "_TOO_LARGE")
    raw = path.read_bytes()
    doc = _loads(raw, code)
    if canonical and raw != _canonical(doc):
        _fail(code + "_NOT_CANONICAL")
    return doc, raw


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


def _int(value: Any, code: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail(code)
    return value


def _dt(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise AttestationLifecycleError(code) from exc


def _ts(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _b64(value: Any, code: str, *, expected_len: int | None = None) -> bytes:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > int(POLICY["max_json_bytes"])
    ):
        _fail(code)
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise AttestationLifecycleError(code) from exc
    if base64.b64encode(raw).decode("ascii") != value or (
        expected_len is not None and len(raw) != expected_len
    ):
        _fail(code)
    return raw


def _ed25519_verify(public: str, signature: str, message: bytes, code: str) -> None:
    try:
        rootseal._ed25519_verify(
            rootseal._b64(public, code + "_PUBLIC", expected_len=32),
            rootseal._b64(signature, code + "_SIGNATURE", expected_len=64),
            message,
            code,
        )
    except rootseal.RootTrustError as exc:
        raise AttestationLifecycleError(str(exc)) from exc


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


def _artifact_raw(name: str, raw: bytes) -> dict[str, Any]:
    return {"name": name, "sha256": _sha_bytes(raw), "size": len(raw)}


def _docs_from_dir(root: Path, names: set[str], code: str) -> dict[str, dict[str, Any]]:
    if {p.name for p in root.iterdir()} != names:
        _fail(code + "_ALLOWLIST_MISMATCH")
    out = {}
    for name in sorted(names):
        out[name] = _read(
            root / name, code + "_" + name.upper().replace("-", "_").replace(".", "_")
        )[0]
    return out


def _write_docs(root: Path, docs: dict[str, dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, doc in sorted(docs.items()):
        _write(root / name, doc)


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_ATTESTATION_LIFECYCLE_STATE = "trusted-attestation-lifecycle-state.json"
_DOC_ROOT_CONTINUITY_STATE = "trusted-root-continuity-state.json"


def _continuity_docs(root: Path) -> dict[str, dict[str, Any]]:
    names = {
        "release-root-continuity-bundle.json",
        _DOC_ROOT_CONTINUITY_STATE,
        "active-root.json",
        "release-root-continuity-receipt.json",
    }
    return _docs_from_dir(root, names, "ATTESTATION_LIFECYCLE_CONTINUITY")


def _root_info(envelope: dict[str, Any], code: str) -> dict[str, Any]:
    try:
        return rootseal._validate_root_envelope(envelope)
    except rootseal.RootTrustError as exc:
        raise AttestationLifecycleError(code + ":" + str(exc)) from exc


def _cert_der(value: str, code: str):
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]

        der = _b64(value, code + "_DER_INVALID")
        cert = x509.load_der_x509_certificate(der)
        return der, cert
    except AttestationLifecycleError:
        raise
    except Exception as exc:
        raise AttestationLifecycleError(code + "_CERT_INVALID") from exc


def _verify_ca_item(item: Any, *, now: datetime, historical: bool) -> dict[str, Any]:
    if not isinstance(item, dict) or set(item) != {"sha256", "der", "profile"}:
        _fail("ATTESTATION_CA_ITEM_SCHEMA_INVALID")
    der, cert = _cert_der(item["der"], "ATTESTATION_CA")
    sha = _hex(item["sha256"], "ATTESTATION_CA_SHA_INVALID")
    if _sha_bytes(der) != sha:
        _fail("ATTESTATION_CA_HASH_MISMATCH")
    profile = _identity(item["profile"], "ATTESTATION_CA_PROFILE_INVALID")
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]

        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        ku = cert.extensions.get_extension_for_class(x509.KeyUsage).value
        if not bc.ca or not ku.key_cert_sign or not ku.crl_sign:
            _fail("ATTESTATION_CA_CONSTRAINTS_INVALID")
        if cert.issuer != cert.subject:
            _fail("ATTESTATION_CA_NOT_SELF_ISSUED")
        continuity._verify_cert_signature(
            cert, cert.public_key(), "ATTESTATION_CA_SELF"
        )
        nb = continuity._cert_time(cert, "not_valid_before")
        na = continuity._cert_time(cert, "not_valid_after")
        if not historical and not (nb <= now <= na):
            _fail("ATTESTATION_CA_NOT_CURRENT")
    except AttestationLifecycleError:
        raise
    except continuity.RootContinuityError as exc:
        raise AttestationLifecycleError(str(exc)) from exc
    except Exception as exc:
        raise AttestationLifecycleError("ATTESTATION_CA_CERT_INVALID") from exc
    return {
        "sha256": sha,
        "der": base64.b64encode(der).decode("ascii"),
        "profile": profile,
    }


def _status_key(value: Any, code: str) -> dict[str, Any]:
    expected = {"keytype", "scheme", "identity", "operator", "expires", "keyval"}
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("keytype") != "ed25519"
        or value.get("scheme") != "ed25519"
    ):
        _fail(code + "_SCHEMA_INVALID")
    keyval = value.get("keyval")
    if not isinstance(keyval, dict) or set(keyval) != {"public"}:
        _fail(code + "_KEYVAL_INVALID")
    rootseal._b64(keyval["public"], code + "_PUBLIC_INVALID", expected_len=32)
    return {
        "keytype": "ed25519",
        "scheme": "ed25519",
        "identity": _identity(value["identity"], code + "_IDENTITY_INVALID"),
        "operator": _identity(value["operator"], code + "_OPERATOR_INVALID"),
        "expires": _ts(_dt(value["expires"], code + "_EXPIRES_INVALID")),
        "keyval": {"public": keyval["public"]},
    }


def _status_authority(
    value: Any, *, active_root: dict[str, Any], ca_expires: datetime
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"keys", "role"}:
        _fail("ATTESTATION_STATUS_AUTHORITY_SCHEMA_INVALID")
    raw_keys = value["keys"]
    if not isinstance(raw_keys, dict) or not (
        int(POLICY["min_status_keys"])
        <= len(raw_keys)
        <= int(POLICY["max_status_keys"])
    ):
        _fail("ATTESTATION_STATUS_KEY_COUNT_INVALID")
    keys = {
        _identity(k, "ATTESTATION_STATUS_KEY_ID_INVALID"): _status_key(
            v, "ATTESTATION_STATUS_KEY"
        )
        for k, v in raw_keys.items()
    }
    role = value["role"]
    if not isinstance(role, dict) or set(role) != {"keyids", "threshold"}:
        _fail("ATTESTATION_STATUS_ROLE_SCHEMA_INVALID")
    keyids = (
        [_identity(x, "ATTESTATION_STATUS_ROLE_KEY_INVALID") for x in role["keyids"]]
        if isinstance(role["keyids"], list)
        else []
    )
    if (
        len(set(keyids)) != len(keyids)
        or sorted(keyids) != keyids
        or set(keyids) != set(keys)
    ):
        _fail("ATTESTATION_STATUS_ROLE_KEY_SET_INVALID")
    threshold = _int(
        role["threshold"], "ATTESTATION_STATUS_THRESHOLD_INVALID", minimum=1
    )
    if threshold < int(POLICY["min_status_threshold"]) or threshold > len(keyids):
        _fail("ATTESTATION_STATUS_THRESHOLD_INVALID")
    identities = {keys[k]["identity"] for k in keyids}
    if len(identities) != len(keyids):
        _fail("ATTESTATION_STATUS_IDENTITY_DUPLICATE")
    operators = {keys[k]["operator"] for k in keyids}
    if len(operators) < int(POLICY["min_status_operators"]):
        _fail("ATTESTATION_STATUS_OPERATOR_QUORUM_INVALID")
    active_root_ids = set(active_root["roles"]["root"]["keyids"])
    active_root_ops = {active_root["keys"][k]["operator"] for k in active_root_ids}
    if (
        bool(POLICY["require_status_key_disjoint_from_release_root"])
        and set(keyids) & active_root_ids
    ):
        _fail("ATTESTATION_STATUS_KEY_OVERLAPS_RELEASE_ROOT")
    if (
        bool(POLICY["require_status_operator_disjoint_from_release_root"])
        and operators & active_root_ops
    ):
        _fail("ATTESTATION_STATUS_OPERATOR_OVERLAPS_RELEASE_ROOT")
    for key in keys.values():
        if _dt(key["expires"], "ATTESTATION_STATUS_KEY_EXPIRES_INVALID") < ca_expires:
            _fail("ATTESTATION_STATUS_KEY_EXPIRES_BEFORE_CA_SET")
    return {
        "keys": {k: keys[k] for k in sorted(keys)},
        "role": {"keyids": sorted(keyids), "threshold": threshold},
    }


def _continuity_binding(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    state = docs[_DOC_ROOT_CONTINUITY_STATE]
    bundle = docs["release-root-continuity-bundle.json"]
    active = _root_info(docs["active-root.json"], "ATTESTATION_ACTIVE_ROOT_INVALID")
    if state.get("rootChainHeadSha256") != bundle.get("recoveryEvent", {}).get(
        "chainHeadSha256"
    ) and not bundle.get("epochs"):
        _fail("ATTESTATION_CONTINUITY_HEAD_MISMATCH")
    return {
        "state": _artifact_raw(_DOC_ROOT_CONTINUITY_STATE, _canonical(state)),
        "bundle": _artifact_raw(
            "release-root-continuity-bundle.json", _canonical(bundle)
        ),
        "rootChainHeadSha256": _hex(
            state.get("rootChainHeadSha256"), "ATTESTATION_CONTINUITY_HEAD_INVALID"
        ),
        "activeRoot": {"version": active["version"], "sha256": active["sha256"]},
    }


def _verify_ca_set(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    *,
    continuity_docs: dict[str, dict[str, Any]],
    previous: dict[str, Any] | None,
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    if set(doc) != {"signatures", "signed"} or not isinstance(doc.get("signed"), dict):
        _fail("ATTESTATION_CA_SET_SCHEMA_INVALID")
    s = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "version",
        "governanceId",
        "issuedAt",
        "expires",
        "continuity",
        "previousCaSetSha256",
        "transition",
        "trustRoots",
        "statusAuthority",
        "requiredProfilesByDeviceClass",
        "selectedRootKeyIds",
    }
    if (
        set(s) != expected
        or s["_type"] != "attestation-ca-set"
        or s["specVersion"] != str(POLICY["spec_version"])
    ):
        _fail("ATTESTATION_CA_SET_SIGNED_SCHEMA_INVALID")
    version = _int(s["version"], "ATTESTATION_CA_SET_VERSION_INVALID", minimum=1)
    active = _root_info(
        continuity_docs["active-root.json"], "ATTESTATION_CA_SET_ACTIVE_ROOT_INVALID"
    )
    if s["governanceId"] != active["governance_id"]:
        _fail("ATTESTATION_CA_SET_GOVERNANCE_ID_MISMATCH")
    binding = _continuity_binding(continuity_docs)
    if s["continuity"] != binding:
        _fail("ATTESTATION_CA_SET_CONTINUITY_MISMATCH")
    issued = _dt(s["issuedAt"], "ATTESTATION_CA_SET_ISSUED_INVALID")
    expires = _dt(s["expires"], "ATTESTATION_CA_SET_EXPIRES_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if issued > now + skew:
        _fail("ATTESTATION_CA_SET_FROM_FUTURE")
    if expires <= issued or expires - issued > timedelta(
        days=int(POLICY["max_ca_set_lifetime_days"])
    ):
        _fail("ATTESTATION_CA_SET_LIFETIME_INVALID")
    if not historical and expires < now + timedelta(
        minutes=int(POLICY["min_ca_set_remaining_minutes"])
    ):
        _fail("ATTESTATION_CA_SET_EXPIRED_OR_FREEZE_RISK")
    if issued < active["issued"]:
        _fail("ATTESTATION_CA_SET_PREDATES_ACTIVE_ROOT")
    if expires > active["expires"]:
        _fail("ATTESTATION_CA_SET_OUTLIVES_ACTIVE_ROOT")
    roots_raw = s["trustRoots"]
    if not isinstance(roots_raw, list) or not (
        1 <= len(roots_raw) <= int(POLICY["max_ca_roots"])
    ):
        _fail("ATTESTATION_CA_SET_ROOTS_INVALID")
    roots = [_verify_ca_item(x, now=now, historical=historical) for x in roots_raw]
    roots.sort(key=lambda x: x["sha256"])
    if roots != roots_raw or len({x["sha256"] for x in roots}) != len(roots):
        _fail("ATTESTATION_CA_SET_ROOTS_NOT_NORMALIZED")
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]

        for item in roots:
            cert = x509.load_der_x509_certificate(
                _b64(item["der"], "ATTESTATION_CA_SET_ROOT_DER_INVALID")
            )
            if issued < continuity._cert_time(
                cert, "not_valid_before"
            ) or expires > continuity._cert_time(cert, "not_valid_after"):
                _fail("ATTESTATION_CA_SET_ROOT_VALIDITY_MISMATCH")
    except AttestationLifecycleError:
        raise
    except Exception as exc:
        raise AttestationLifecycleError("ATTESTATION_CA_SET_ROOT_CERT_INVALID") from exc
    profile_map = s["requiredProfilesByDeviceClass"]
    required_map = {
        str(k): str(v) for k, v in POLICY["required_profile_by_device_class"].items()
    }
    if profile_map != required_map:
        _fail("ATTESTATION_CA_SET_PROFILE_POLICY_MISMATCH")
    allowed_profiles = set(required_map.values())
    if any(item["profile"] not in allowed_profiles for item in roots):
        _fail("ATTESTATION_CA_SET_ROOT_PROFILE_INVALID")
    authority = _status_authority(
        s["statusAuthority"], active_root=active, ca_expires=expires
    )
    transition = s["transition"]
    if not isinstance(transition, dict) or set(transition) != {
        "type",
        "retiredCaSha256",
        "revokedCaSha256",
    }:
        _fail("ATTESTATION_CA_SET_TRANSITION_SCHEMA_INVALID")
    typ = transition["type"]
    if typ not in {"bootstrap", "scheduled-rotation", "compromise-recovery"}:
        _fail("ATTESTATION_CA_SET_TRANSITION_TYPE_INVALID")
    retired = (
        [
            _hex(x, "ATTESTATION_CA_SET_RETIRED_CA_INVALID")
            for x in transition["retiredCaSha256"]
        ]
        if isinstance(transition["retiredCaSha256"], list)
        else []
    )
    revoked = (
        [
            _hex(x, "ATTESTATION_CA_SET_REVOKED_CA_INVALID")
            for x in transition["revokedCaSha256"]
        ]
        if isinstance(transition["revokedCaSha256"], list)
        else []
    )
    if (
        sorted(set(retired)) != retired
        or sorted(set(revoked)) != revoked
        or set(retired) & set(revoked)
    ):
        _fail("ATTESTATION_CA_SET_TRANSITION_CA_SET_INVALID")
    current_set = {x["sha256"] for x in roots}
    if previous is None:
        if (
            version != 1
            or s["previousCaSetSha256"] != _ZERO_HASH
            or typ != "bootstrap"
            or retired
            or revoked
        ):
            _fail("ATTESTATION_CA_SET_BOOTSTRAP_INVALID")
        embedded = continuity_docs["release-root-continuity-bundle.json"].get(
            "attestationTrustRoots"
        )
        embedded_pins = (
            sorted(x.get("sha256") for x in embedded)
            if isinstance(embedded, list)
            else []
        )
        if sorted(current_set) != embedded_pins:
            _fail("ATTESTATION_CA_SET_BOOTSTRAP_ROOT_SET_MISMATCH")
    else:
        prev_signed = previous["signed"]
        prev_sha = _sha_bytes(_canonical(previous))
        if (
            version != prev_signed["version"] + 1
            or s["previousCaSetSha256"] != prev_sha
            or typ == "bootstrap"
        ):
            _fail("ATTESTATION_CA_SET_ROTATION_CONTINUITY_INVALID")
        if issued < _dt(
            prev_signed["issuedAt"], "ATTESTATION_PREVIOUS_CA_SET_ISSUED_INVALID"
        ):
            _fail("ATTESTATION_CA_SET_ISSUED_AT_ROLLBACK")
        old = {x["sha256"] for x in prev_signed["trustRoots"]}
        removed = old - current_set
        if removed != set(retired) | set(revoked):
            _fail("ATTESTATION_CA_SET_REMOVAL_ACCOUNTING_INVALID")
        if typ == "compromise-recovery" and not revoked:
            _fail("ATTESTATION_CA_SET_COMPROMISE_REQUIRES_REVOCATION")
        if typ == "scheduled-rotation" and revoked:
            _fail("ATTESTATION_CA_SET_SCHEDULED_ROTATION_CANNOT_REVOKE")
    normalized_signed = dict(s)
    normalized_signed["trustRoots"] = roots
    normalized_signed["statusAuthority"] = authority
    normalized_signed["issuedAt"] = _ts(issued)
    normalized_signed["expires"] = _ts(expires)
    normalized_signed["transition"] = {
        "type": typ,
        "retiredCaSha256": retired,
        "revokedCaSha256": revoked,
    }
    if normalized_signed != s:
        _fail("ATTESTATION_CA_SET_NOT_NORMALIZED")
    role = active["roles"]["root"]
    selected = s["selectedRootKeyIds"]
    if not isinstance(selected, list):
        _fail("ATTESTATION_CA_SET_SELECTED_ROOT_KEYS_INVALID")
    selected = [
        _identity(x, "ATTESTATION_CA_SET_SELECTED_ROOT_KEY_INVALID") for x in selected
    ]
    if (
        sorted(set(selected)) != selected
        or len(selected) != role["threshold"]
        or not set(selected).issubset(set(role["keyids"]))
    ):
        _fail("ATTESTATION_CA_SET_SELECTED_ROOT_KEYS_INVALID")
    if len({active["keys"][kid]["operator"] for kid in selected}) < int(
        POLICY["min_root_signer_operators"]
    ):
        _fail("ATTESTATION_CA_SET_ROOT_SIGNER_OPERATOR_QUORUM_INVALID")
    sigs = doc["signatures"]
    if not isinstance(sigs, list):
        _fail("ATTESTATION_CA_SET_SIGNATURES_INVALID")
    sigmap: dict[str, str] = {}
    normalized_sigs = []
    for item in sigs:
        if not isinstance(item, dict) or set(item) != {"keyid", "sig"}:
            _fail("ATTESTATION_CA_SET_SIGNATURE_SCHEMA_INVALID")
        kid = _identity(item["keyid"], "ATTESTATION_CA_SET_SIGNATURE_KEY_INVALID")
        if kid in sigmap:
            _fail("ATTESTATION_CA_SET_SIGNATURE_DUPLICATE")
        rootseal._b64(
            item["sig"],
            "ATTESTATION_CA_SET_SIGNATURE_ENCODING_INVALID",
            expected_len=64,
        )
        sigmap[kid] = item["sig"]
        normalized_sigs.append({"keyid": kid, "sig": item["sig"]})
    normalized_sigs.sort(key=lambda x: x["keyid"])
    if normalized_sigs != sigs:
        _fail("ATTESTATION_CA_SET_SIGNATURES_NOT_NORMALIZED")
    if set(sigmap) != set(selected):
        _fail("ATTESTATION_CA_SET_SIGNATURE_SET_MISMATCH")
    valid = []
    for kid in role["keyids"]:
        if kid not in sigmap:
            continue
        key = active["keys"][kid]
        if _dt(key["expires"], "ATTESTATION_CA_SET_ROOT_KEY_EXPIRES_INVALID") < issued:
            continue
        _ed25519_verify(
            key["keyval"]["public"],
            sigmap[kid],
            _canonical(s),
            "ATTESTATION_CA_SET_ROOT_SIGNATURE_INVALID",
        )
        valid.append(kid)
    if len(valid) < role["threshold"]:
        _fail("ATTESTATION_CA_SET_ROOT_THRESHOLD_NOT_MET")
    if len({active["keys"][kid]["operator"] for kid in valid}) < int(
        POLICY["min_root_signer_operators"]
    ):
        _fail("ATTESTATION_CA_SET_ROOT_SIGNER_OPERATOR_QUORUM_INVALID")
    return {"signatures": normalized_sigs, "signed": normalized_signed}


def _claims_from_attestation(
    attestation: dict[str, Any], *, ca_set: dict[str, Any]
) -> dict[str, Any]:
    s = attestation["signed"]
    chain = s.get("certificateChainDer")
    if not isinstance(chain, list) or not chain:
        _fail("ATTESTATION_VENDOR_CERT_CHAIN_INVALID")
    der, cert = _cert_der(chain[0], "ATTESTATION_VENDOR_LEAF")
    required_map = ca_set["signed"]["requiredProfilesByDeviceClass"]
    device = s.get("deviceClass")
    if device not in required_map:
        _fail("ATTESTATION_VENDOR_DEVICE_CLASS_UNSUPPORTED")
    required_profile = required_map[device]
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]

        oid = x509.ObjectIdentifier(str(POLICY["vendor_claim_extension_oid"]))
        ext = cert.extensions.get_extension_for_oid(oid)
        raw = bytes(ext.value.value)
    except Exception as exc:
        raise AttestationLifecycleError(
            "ATTESTATION_VENDOR_CLAIM_EXTENSION_MISSING"
        ) from exc
    claim = _loads(raw, "ATTESTATION_VENDOR_CLAIM")
    expected = {
        "schemaVersion",
        "profile",
        "deviceClass",
        "manufacturer",
        "model",
        "hardwareBacked",
        "nonExportable",
        "keyId",
        "publicKeySha256",
        "deviceIdHash",
    }
    if set(claim) != expected or claim["schemaVersion"] != int(
        POLICY["vendor_claim_schema_version"]
    ):
        _fail("ATTESTATION_VENDOR_CLAIM_SCHEMA_INVALID")
    if raw != _canonical(claim):
        _fail("ATTESTATION_VENDOR_CLAIM_NOT_CANONICAL")
    if (
        claim["profile"] != required_profile
        or claim["deviceClass"] != device
        or claim["manufacturer"] != s["manufacturer"]
        or claim["model"] != s["model"]
    ):
        _fail("ATTESTATION_VENDOR_CLAIM_SEMANTICS_MISMATCH")
    if claim["hardwareBacked"] is not True or claim["nonExportable"] is not True:
        _fail("ATTESTATION_VENDOR_CLAIM_HARDWARE_PROPERTIES_INVALID")
    if claim["keyId"] != s["keyId"]:
        _fail("ATTESTATION_VENDOR_CLAIM_KEYID_MISMATCH")
    public_raw = rootseal._b64(
        s["publicKey"], "ATTESTATION_VENDOR_ROOT_PUBLIC_INVALID", expected_len=32
    )
    if claim["publicKeySha256"] != _sha_bytes(public_raw):
        _fail("ATTESTATION_VENDOR_CLAIM_ROOT_PUBLIC_MISMATCH")
    _hex(claim["deviceIdHash"], "ATTESTATION_VENDOR_DEVICE_ID_INVALID")
    return {
        "keyId": s["keyId"],
        "deviceIdHash": claim["deviceIdHash"],
        "profile": claim["profile"],
        "leafSha256": _sha_bytes(der),
    }


def _attestations_from_continuity(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    recovery = bundle.get("recoveryAttestations")
    if not isinstance(recovery, list):
        _fail("ATTESTATION_CONTINUITY_ATTESTATIONS_INVALID")
    out.extend(recovery)
    epochs = bundle.get("epochs")
    if not isinstance(epochs, list):
        _fail("ATTESTATION_CONTINUITY_EPOCHS_INVALID")
    for entry in epochs:
        if not isinstance(entry, dict):
            _fail("ATTESTATION_CONTINUITY_EPOCH_INVALID")
        docs = entry.get("rootAttestations")
        if docs:
            if not isinstance(docs, list):
                _fail("ATTESTATION_CONTINUITY_EPOCH_ATTESTATIONS_INVALID")
            out.extend(docs)
    return out


def _inventory(
    continuity_bundle: dict[str, Any], *, ca_set: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    certs: dict[str, dict[str, Any]] = {}
    bindings: dict[str, str] = {}
    for attestation in _attestations_from_continuity(continuity_bundle):
        if not isinstance(attestation, dict) or not isinstance(
            attestation.get("signed"), dict
        ):
            _fail("ATTESTATION_CONTINUITY_ATTESTATION_SCHEMA_INVALID")
        s = attestation["signed"]
        vendor = _claims_from_attestation(attestation, ca_set=ca_set)
        previous = bindings.get(vendor["keyId"])
        if previous is not None and previous != vendor["deviceIdHash"]:
            _fail("ATTESTATION_DEVICE_IDENTITY_CHANGED")
        bindings[vendor["keyId"]] = vendor["deviceIdHash"]
        attested = _dt(s["attestedAt"], "ATTESTATION_INVENTORY_ATTESTED_AT_INVALID")
        chain = s["certificateChainDer"]
        for _index, text in enumerate(chain):
            der, cert = _cert_der(text, "ATTESTATION_INVENTORY_CERT")
            sha = _sha_bytes(der)
            item = certs.setdefault(
                sha,
                {
                    "sha256": sha,
                    "serialNumber": format(cert.serial_number, "x"),
                    "keyIds": [],
                    "attestedAt": [],
                },
            )
            if item["serialNumber"] != format(cert.serial_number, "x"):
                _fail("ATTESTATION_INVENTORY_CERT_COLLISION")
            if s["keyId"] not in item["keyIds"]:
                item["keyIds"].append(s["keyId"])
            ts = _ts(attested)
            if ts not in item["attestedAt"]:
                item["attestedAt"].append(ts)
    normalized = []
    for item in certs.values():
        item["keyIds"].sort()
        item["attestedAt"].sort()
        normalized.append(item)
    normalized.sort(key=lambda x: x["sha256"])
    if len(normalized) > int(POLICY["max_status_entries"]):
        _fail("ATTESTATION_INVENTORY_TOO_LARGE")
    return normalized, {k: bindings[k] for k in sorted(bindings)}


def _verify_status_snapshot(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    *,
    ca_set: dict[str, Any],
    continuity_bundle: dict[str, Any],
    previous: dict[str, Any] | None,
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    if set(doc) != {"signatures", "signed"} or not isinstance(doc.get("signed"), dict):
        _fail("ATTESTATION_STATUS_SCHEMA_INVALID")
    s = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "version",
        "governanceId",
        "caSetVersion",
        "caSetSha256",
        "previousStatusSha256",
        "continuityRootChainHeadSha256",
        "issuedAt",
        "nextUpdate",
        "certificates",
        "selectedStatusKeyIds",
    }
    if (
        set(s) != expected
        or s["_type"] != "attestation-status"
        or s["specVersion"] != str(POLICY["spec_version"])
    ):
        _fail("ATTESTATION_STATUS_SIGNED_SCHEMA_INVALID")
    ca_signed = ca_set["signed"]
    if (
        s["governanceId"] != ca_signed["governanceId"]
        or s["caSetVersion"] != ca_signed["version"]
        or s["caSetSha256"] != _sha_bytes(_canonical(ca_set))
    ):
        _fail("ATTESTATION_STATUS_CA_SET_MISMATCH")
    expected_head = ca_signed["continuity"]["rootChainHeadSha256"]
    if s["continuityRootChainHeadSha256"] != expected_head:
        _fail("ATTESTATION_STATUS_CONTINUITY_MISMATCH")
    version = _int(s["version"], "ATTESTATION_STATUS_VERSION_INVALID", minimum=1)
    if previous is None:
        if version != 1 or s["previousStatusSha256"] != _ZERO_HASH:
            _fail("ATTESTATION_STATUS_BOOTSTRAP_VERSION_INVALID")
    else:
        if version != previous["signed"]["version"] + 1:
            _fail("ATTESTATION_STATUS_VERSION_NOT_CONSECUTIVE")
        if s["previousStatusSha256"] != _sha_bytes(_canonical(previous)):
            _fail("ATTESTATION_STATUS_PREVIOUS_HASH_MISMATCH")
    issued = _dt(s["issuedAt"], "ATTESTATION_STATUS_ISSUED_INVALID")
    next_update = _dt(s["nextUpdate"], "ATTESTATION_STATUS_NEXT_UPDATE_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if issued > now + skew:
        _fail("ATTESTATION_STATUS_FROM_FUTURE")
    if next_update <= issued or next_update - issued > timedelta(
        hours=int(POLICY["max_status_lifetime_hours"])
    ):
        _fail("ATTESTATION_STATUS_LIFETIME_INVALID")
    if not historical and next_update < now + timedelta(
        minutes=int(POLICY["min_status_remaining_minutes"])
    ):
        _fail("ATTESTATION_STATUS_EXPIRED_OR_FREEZE_RISK")
    ca_issued = _dt(ca_signed["issuedAt"], "ATTESTATION_STATUS_CA_ISSUED_INVALID")
    ca_expires = _dt(ca_signed["expires"], "ATTESTATION_STATUS_CA_EXPIRES_INVALID")
    if issued < ca_issued:
        _fail("ATTESTATION_STATUS_PREDATES_CA_SET")
    if next_update > ca_expires:
        _fail("ATTESTATION_STATUS_OUTLIVES_CA_SET")
    inventory, _ = _inventory(continuity_bundle, ca_set=ca_set)
    expected_by_sha = {x["sha256"]: x for x in inventory}
    raw_entries = s["certificates"]
    if not isinstance(raw_entries, list) or len(raw_entries) > int(
        POLICY["max_status_entries"]
    ):
        _fail("ATTESTATION_STATUS_CERTIFICATES_INVALID")
    entries = []
    seen = set()
    for entry in raw_entries:
        if not isinstance(entry, dict) or set(entry) != {
            "sha256",
            "serialNumber",
            "status",
            "revokedAt",
            "reason",
        }:
            _fail("ATTESTATION_STATUS_ENTRY_SCHEMA_INVALID")
        sha = _hex(entry["sha256"], "ATTESTATION_STATUS_CERT_SHA_INVALID")
        if sha in seen:
            _fail("ATTESTATION_STATUS_CERT_DUPLICATE")
        seen.add(sha)
        if (
            sha not in expected_by_sha
            or entry["serialNumber"] != expected_by_sha[sha]["serialNumber"]
        ):
            _fail("ATTESTATION_STATUS_CERT_INVENTORY_MISMATCH")
        status = entry["status"]
        if status not in {"good", "revoked"}:
            _fail("ATTESTATION_STATUS_VALUE_INVALID")
        if status == "good":
            if entry["revokedAt"] is not None or entry["reason"] is not None:
                _fail("ATTESTATION_STATUS_GOOD_FIELDS_INVALID")
        else:
            revoked = _dt(entry["revokedAt"], "ATTESTATION_STATUS_REVOKED_AT_INVALID")
            if revoked > issued + skew:
                _fail("ATTESTATION_STATUS_REVOCATION_FROM_FUTURE")
            _identity(entry["reason"], "ATTESTATION_STATUS_REASON_INVALID")
        entries.append(entry)
    entries.sort(key=lambda x: x["sha256"])
    if entries != raw_entries:
        _fail("ATTESTATION_STATUS_ENTRIES_NOT_NORMALIZED")
    if (
        bool(POLICY["require_all_attestation_certificates_covered"])
        and set(expected_by_sha) != seen
    ):
        _fail("ATTESTATION_STATUS_COVERAGE_INCOMPLETE")
    if previous is not None:
        old = {x["sha256"]: x for x in previous["signed"]["certificates"]}
        for entry in entries:
            prior = old.get(entry["sha256"])
            if prior is None:
                continue
            if prior["status"] == "revoked" and entry != prior:
                _fail("ATTESTATION_STATUS_REVOCATION_NOT_STICKY")
    authority = ca_signed["statusAuthority"]
    role = authority["role"]
    keys = authority["keys"]
    selected = s["selectedStatusKeyIds"]
    if not isinstance(selected, list):
        _fail("ATTESTATION_STATUS_SELECTED_KEYS_INVALID")
    selected = [
        _identity(x, "ATTESTATION_STATUS_SELECTED_KEY_INVALID") for x in selected
    ]
    if (
        sorted(set(selected)) != selected
        or len(selected) != role["threshold"]
        or not set(selected).issubset(set(role["keyids"]))
    ):
        _fail("ATTESTATION_STATUS_SELECTED_KEYS_INVALID")
    if len({keys[kid]["operator"] for kid in selected}) < int(
        POLICY["min_status_operators"]
    ):
        _fail("ATTESTATION_STATUS_SIGNER_OPERATOR_QUORUM_INVALID")
    sigs = doc["signatures"]
    if not isinstance(sigs, list):
        _fail("ATTESTATION_STATUS_SIGNATURES_INVALID")
    sigmap = {}
    normalized_sigs = []
    for item in sigs:
        if not isinstance(item, dict) or set(item) != {"keyid", "sig"}:
            _fail("ATTESTATION_STATUS_SIGNATURE_SCHEMA_INVALID")
        kid = _identity(item["keyid"], "ATTESTATION_STATUS_SIGNATURE_KEY_INVALID")
        if kid in sigmap or kid not in selected:
            _fail("ATTESTATION_STATUS_SIGNATURE_KEY_UNAUTHORIZED")
        rootseal._b64(
            item["sig"],
            "ATTESTATION_STATUS_SIGNATURE_ENCODING_INVALID",
            expected_len=64,
        )
        sigmap[kid] = item["sig"]
        normalized_sigs.append({"keyid": kid, "sig": item["sig"]})
    normalized_sigs.sort(key=lambda x: x["keyid"])
    if normalized_sigs != sigs:
        _fail("ATTESTATION_STATUS_SIGNATURES_NOT_NORMALIZED")
    if set(sigmap) != set(selected):
        _fail("ATTESTATION_STATUS_SIGNATURE_SET_MISMATCH")
    valid = []
    for kid in role["keyids"]:
        if kid not in sigmap:
            continue
        if (
            _dt(keys[kid]["expires"], "ATTESTATION_STATUS_SIGNER_EXPIRES_INVALID")
            < issued
        ):
            continue
        _ed25519_verify(
            keys[kid]["keyval"]["public"],
            sigmap[kid],
            _canonical(s),
            "ATTESTATION_STATUS_SIGNATURE_INVALID",
        )
        valid.append(kid)
    if len(valid) < role["threshold"]:
        _fail("ATTESTATION_STATUS_THRESHOLD_NOT_MET")
    if len({keys[kid]["operator"] for kid in valid}) < int(
        POLICY["min_status_operators"]
    ):
        _fail("ATTESTATION_STATUS_SIGNER_OPERATOR_QUORUM_INVALID")
    normalized_signed = dict(s)
    normalized_signed["issuedAt"] = _ts(issued)
    normalized_signed["nextUpdate"] = _ts(next_update)
    normalized_signed["certificates"] = entries
    if normalized_signed != s:
        _fail("ATTESTATION_STATUS_NOT_NORMALIZED")
    return {"signatures": normalized_sigs, "signed": normalized_signed}


def _evaluate_status(
    *,
    continuity_bundle: dict[str, Any],
    ca_set: dict[str, Any],
    status: dict[str, Any],
    now: datetime,
    historical: bool,
) -> None:
    inventory, _ = _inventory(continuity_bundle, ca_set=ca_set)
    by_sha = {x["sha256"]: x for x in status["signed"]["certificates"]}
    for cert in inventory:
        decision = by_sha[cert["sha256"]]
        if decision["status"] != "revoked":
            continue
        revoked = _dt(decision["revokedAt"], "ATTESTATION_STATUS_REVOKED_AT_INVALID")
        attested_times = [
            _dt(x, "ATTESTATION_INVENTORY_TIME_INVALID") for x in cert["attestedAt"]
        ]
        if any(revoked <= t for t in attested_times):
            _fail("ATTESTATION_CERT_REVOKED_AT_ATTESTATION_TIME")
        if not historical and revoked <= now:
            _fail("ATTESTATION_ACTIVE_CERT_REVOKED")


def _ca_revoked_union(history: list[dict[str, Any]]) -> list[str]:
    result = set()
    for envelope in history:
        result.update(envelope["signed"]["transition"]["revokedCaSha256"])
    return sorted(result)


def _lifecycle_head(previous: str, body: dict[str, Any]) -> str:
    return _sha_bytes(_canonical({"previousChainHeadSha256": previous, "event": body}))


def _state(
    bundle: dict[str, Any],
    ca_set: dict[str, Any],
    status: dict[str, Any],
    chain: str,
    device_bindings: dict[str, str],
) -> dict[str, Any]:
    return {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-attestation-lifecycle",
        "governanceId": bundle["governanceId"],
        "lifecycleSequence": len(bundle["events"]),
        "continuityRootChainHeadSha256": bundle["continuityBinding"][
            "rootChainHeadSha256"
        ],
        "activeCaSet": {
            "version": ca_set["signed"]["version"],
            "sha256": _sha_bytes(_canonical(ca_set)),
            "expires": ca_set["signed"]["expires"],
        },
        "activeStatus": {
            "version": status["signed"]["version"],
            "sha256": _sha_bytes(_canonical(status)),
            "nextUpdate": status["signed"]["nextUpdate"],
        },
        "revokedCaSha256": _ca_revoked_union(bundle["caSets"]),
        "deviceBindings": device_bindings,
        "lifecycleChainHeadSha256": chain,
        "bundleSha256": _sha_bytes(_canonical(bundle)),
    }


def _write_output(
    target: Path,
    *,
    bundle: dict[str, Any],
    state: dict[str, Any],
    ca_set: dict[str, Any],
    status: dict[str, Any],
    input_files: list[Path],
    before: dict[str, str],
    phase: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="run158-lifecycle-", dir=target.parent
    ) as td:
        stage = Path(td) / "lifecycle"
        stage.mkdir()
        _write(stage / "release-attestation-lifecycle-bundle.json", bundle)
        _write(stage / _DOC_ATTESTATION_LIFECYCLE_STATE, state)
        _write(stage / "active-attestation-ca-set.json", ca_set)
        _write(stage / "active-attestation-status.json", status)
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": phase,
            "governanceId": bundle["governanceId"],
            "bundle": _artifact_raw(
                "release-attestation-lifecycle-bundle.json", _canonical(bundle)
            ),
            "state": _artifact_raw(_DOC_ATTESTATION_LIFECYCLE_STATE, _canonical(state)),
            "caSet": _artifact_raw(
                "active-attestation-ca-set.json", _canonical(ca_set)
            ),
            "attestationStatus": _artifact_raw(
                "active-attestation-status.json", _canonical(status)
            ),
        }
        _write(stage / "release-attestation-lifecycle-receipt.json", receipt)
        for p in input_files:
            if _sha(p) != before[str(p.resolve())]:
                _fail("ATTESTATION_LIFECYCLE_INPUT_CHANGED")
        temp = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, temp, copy_function=shutil.copy2)
        os.replace(temp, target)


def initialize_lifecycle(  # ruff: ignore[undocumented-public-function]
    *,
    continuity_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    ca_set_path: Path,
    status_snapshot_path: Path,
    output_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cont = _regular_dir(continuity_dir, "ATTESTATION_CONTINUITY_DIR_INVALID")
    try:
        continuity.verify_continuity(
            output_dir=cont,
            expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
            expected_recovery_root_sha256=expected_recovery_root_sha256,
            expected_attestation_root_sha256=expected_attestation_root_sha256,
            now=current,
        )
    except continuity.RootContinuityError as exc:
        raise AttestationLifecycleError(
            "ATTESTATION_PREDECESSOR_INVALID:" + str(exc)
        ) from exc
    docs = _continuity_docs(cont)
    ca_doc, _ = _read(ca_set_path, "ATTESTATION_CA_SET")
    ca_set = _verify_ca_set(
        ca_doc, continuity_docs=docs, previous=None, now=current, historical=False
    )
    status_doc, _ = _read(status_snapshot_path, "ATTESTATION_STATUS")
    status = _verify_status_snapshot(
        status_doc,
        ca_set=ca_set,
        continuity_bundle=docs["release-root-continuity-bundle.json"],
        previous=None,
        now=current,
        historical=False,
    )
    _evaluate_status(
        continuity_bundle=docs["release-root-continuity-bundle.json"],
        ca_set=ca_set,
        status=status,
        now=current,
        historical=False,
    )
    inventory, bindings = _inventory(
        docs["release-root-continuity-bundle.json"], ca_set=ca_set
    )
    ca_sha = _sha_bytes(_canonical(ca_set))
    status_sha = _sha_bytes(_canonical(status))
    body = {
        "sequence": 1,
        "type": "lifecycle-bootstrap",
        "previousChainHeadSha256": docs[_DOC_ROOT_CONTINUITY_STATE][
            "rootChainHeadSha256"
        ],
        "caSetSha256": ca_sha,
        "statusSha256": status_sha,
        "inventorySha256": _sha_bytes(_canonical({"certificates": inventory})),
        "deviceBindingsSha256": _sha_bytes(_canonical(bindings)),
    }
    chain = _lifecycle_head(body["previousChainHeadSha256"], body)
    event = dict(body, lifecycleChainHeadSha256=chain)
    binding = _continuity_binding(docs)
    bundle = {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "trusted-attestation-lifecycle-chain",
        "governanceId": ca_set["signed"]["governanceId"],
        "bootstrapRootSha256": _hex(
            expected_bootstrap_root_sha256, "ATTESTATION_BOOTSTRAP_PIN_INVALID"
        ),
        "recoveryRootSha256": _hex(
            expected_recovery_root_sha256, "ATTESTATION_RECOVERY_PIN_INVALID"
        ),
        "attestationRootSha256": sorted(
            _hex(x, "ATTESTATION_ROOT_PIN_INVALID")
            for x in expected_attestation_root_sha256
        ),
        "continuityOutput": docs,
        "continuityBinding": binding,
        "caSets": [ca_set],
        "statusSnapshots": [status],
        "events": [event],
    }
    state = _state(bundle, ca_set, status, chain, bindings)
    target = _outside(
        output_dir,
        [cont, ca_set_path, status_snapshot_path],
        "ATTESTATION_OUTPUT_INSIDE_INPUT",
    )
    if target.exists() or target.is_symlink():
        _fail("ATTESTATION_OUTPUT_ALREADY_EXISTS")
    inputs = [p for p in cont.rglob("*") if p.is_file()] + [
        ca_set_path,
        status_snapshot_path,
    ]
    before = {str(p.resolve()): _sha(p) for p in inputs}
    _write_output(
        target,
        bundle=bundle,
        state=state,
        ca_set=ca_set,
        status=status,
        input_files=inputs,
        before=before,
        phase="attestation-lifecycle-initialized",
    )
    verify_lifecycle(
        output_dir=target,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    return {
        "ok": True,
        "phase": "attestation-lifecycle-initialized",
        "ca_set_version": 1,
        "status_version": 1,
        "lifecycle_chain_head_sha256": chain,
    }


def _verify_bundle(  # ruff: ignore[too-many-branches]
    bundle: dict[str, Any],
    *,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    now: datetime,
    historical: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, str]]:
    expected = {
        "schemaVersion",
        "predicateType",
        "status",
        "governanceId",
        "bootstrapRootSha256",
        "recoveryRootSha256",
        "attestationRootSha256",
        "continuityOutput",
        "continuityBinding",
        "caSets",
        "statusSnapshots",
        "events",
    }
    if (
        set(bundle) != expected
        or bundle["schemaVersion"] != int(POLICY["bundle_schema_version"])
        or bundle["predicateType"] != PREDICATE_TYPE
        or bundle["status"] != "trusted-attestation-lifecycle-chain"
    ):
        _fail("ATTESTATION_BUNDLE_SCHEMA_INVALID")
    if (
        bundle["bootstrapRootSha256"]
        != _hex(bootstrap_pin, "ATTESTATION_BOOTSTRAP_PIN_INVALID")
        or bundle["recoveryRootSha256"]
        != _hex(recovery_pin, "ATTESTATION_RECOVERY_PIN_INVALID")
        or bundle["attestationRootSha256"] != sorted(attestation_pins)
    ):
        _fail("ATTESTATION_BUNDLE_PIN_MISMATCH")
    docs = bundle["continuityOutput"]
    if not isinstance(docs, dict):
        _fail("ATTESTATION_BUNDLE_CONTINUITY_INVALID")
    with tempfile.TemporaryDirectory(prefix="run158-continuity-") as td:
        d = Path(td) / "continuity"
        _write_docs(d, docs)
        try:
            continuity.verify_continuity(
                output_dir=d,
                expected_bootstrap_root_sha256=bootstrap_pin,
                expected_recovery_root_sha256=recovery_pin,
                expected_attestation_root_sha256=attestation_pins,
                now=now,
                historical=historical,
            )
        except continuity.RootContinuityError as exc:
            raise AttestationLifecycleError(
                "ATTESTATION_EMBEDDED_CONTINUITY_INVALID:" + str(exc)
            ) from exc
    if bundle["continuityBinding"] != _continuity_binding(docs):
        _fail("ATTESTATION_BUNDLE_CONTINUITY_BINDING_INVALID")
    ca_sets = bundle["caSets"]
    statuses = bundle["statusSnapshots"]
    events = bundle["events"]
    if (
        not isinstance(ca_sets, list)
        or not isinstance(statuses, list)
        or not isinstance(events, list)
        or not ca_sets
        or not statuses
        or len(statuses) != len(events)
        or len(events) > int(POLICY["max_lifecycle_events"])
    ):
        _fail("ATTESTATION_BUNDLE_HISTORY_INVALID")
    prev_ca = None
    verified_ca = []
    revoked = set()
    current_ca = None
    for raw in ca_sets:
        ca = _verify_ca_set(
            raw, continuity_docs=docs, previous=prev_ca, now=now, historical=True
        )
        verified_ca.append(ca)
        prev_ca = ca
        current_ca = ca
        revoked.update(ca["signed"]["transition"]["revokedCaSha256"])
        active_pins = {x["sha256"] for x in ca["signed"]["trustRoots"]}
        if active_pins & revoked:
            _fail("ATTESTATION_REVOKED_CA_REINTRODUCED")
    if verified_ca != ca_sets:
        _fail("ATTESTATION_CA_SET_HISTORY_NOT_NORMALIZED")
    ca_by_sha = {_sha_bytes(_canonical(x)): x for x in ca_sets}
    referenced_ca = [raw.get("signed", {}).get("caSetSha256") for raw in statuses]
    if any(sha not in referenced_ca for sha in ca_by_sha) or referenced_ca[
        -1
    ] != _sha_bytes(_canonical(ca_sets[-1])):
        _fail("ATTESTATION_CA_SET_HISTORY_UNREFERENCED")
    prev_status = None
    chain = docs[_DOC_ROOT_CONTINUITY_STATE]["rootChainHeadSha256"]
    bindings_final = {}
    for idx, (raw, event) in enumerate(zip(statuses, events), start=1):
        ca = ca_by_sha.get(raw.get("signed", {}).get("caSetSha256"))
        if ca is None:
            _fail("ATTESTATION_STATUS_REFERENCES_UNKNOWN_CA_SET")
        status = _verify_status_snapshot(
            raw,
            ca_set=ca,
            continuity_bundle=docs["release-root-continuity-bundle.json"],
            previous=prev_status,
            now=now,
            historical=True,
        )
        prev_status = status
        _evaluate_status(
            continuity_bundle=docs["release-root-continuity-bundle.json"],
            ca_set=ca,
            status=status,
            now=now,
            historical=True,
        )
        inventory, bindings = _inventory(
            docs["release-root-continuity-bundle.json"], ca_set=ca
        )
        bindings_final = bindings
        body = {
            "sequence": idx,
            "type": "lifecycle-bootstrap" if idx == 1 else "lifecycle-advance",
            "previousChainHeadSha256": chain,
            "caSetSha256": _sha_bytes(_canonical(ca)),
            "statusSha256": _sha_bytes(_canonical(status)),
            "inventorySha256": _sha_bytes(_canonical({"certificates": inventory})),
            "deviceBindingsSha256": _sha_bytes(_canonical(bindings)),
        }
        next_chain = _lifecycle_head(chain, body)
        expected_event = dict(body, lifecycleChainHeadSha256=next_chain)
        if event != expected_event:
            _fail("ATTESTATION_LIFECYCLE_EVENT_REBIND_FAILED")
        chain = next_chain
    if not historical:
        # Live policy applies only to the current CA/status; historical entries replay at their signing times.
        current_ca = _verify_ca_set(
            ca_sets[-1],
            continuity_docs=docs,
            previous=(ca_sets[-2] if len(ca_sets) > 1 else None),
            now=now,
            historical=False,
        )
        current_status = _verify_status_snapshot(
            statuses[-1],
            ca_set=current_ca,
            continuity_bundle=docs["release-root-continuity-bundle.json"],
            previous=(statuses[-2] if len(statuses) > 1 else None),
            now=now,
            historical=False,
        )
        _evaluate_status(
            continuity_bundle=docs["release-root-continuity-bundle.json"],
            ca_set=current_ca,
            status=current_status,
            now=now,
            historical=False,
        )
    return ca_sets[-1], statuses[-1], chain, bindings_final


def verify_lifecycle(  # ruff: ignore[undocumented-public-function]
    *,
    output_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root = _regular_dir(output_dir, "ATTESTATION_VERIFY_DIR_INVALID")
    names = {
        "release-attestation-lifecycle-bundle.json",
        _DOC_ATTESTATION_LIFECYCLE_STATE,
        "active-attestation-ca-set.json",
        "active-attestation-status.json",
        "release-attestation-lifecycle-receipt.json",
    }
    docs = _docs_from_dir(root, names, "ATTESTATION_VERIFY")
    bundle = docs["release-attestation-lifecycle-bundle.json"]
    state = docs[_DOC_ATTESTATION_LIFECYCLE_STATE]
    ca, status, chain, bindings = _verify_bundle(
        bundle,
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=expected_attestation_root_sha256,
        now=current,
        historical=historical,
    )
    expected_state = _state(bundle, ca, status, chain, bindings)
    if state != expected_state:
        _fail("ATTESTATION_STATE_REBIND_FAILED")
    if (
        docs["active-attestation-ca-set.json"] != ca
        or docs["active-attestation-status.json"] != status
    ):
        _fail("ATTESTATION_ACTIVE_ARTIFACT_REBIND_FAILED")
    receipt = docs["release-attestation-lifecycle-receipt.json"]
    expected_receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": receipt.get("status"),
        "governanceId": bundle["governanceId"],
        "bundle": _artifact_raw(
            "release-attestation-lifecycle-bundle.json", _canonical(bundle)
        ),
        "state": _artifact_raw(_DOC_ATTESTATION_LIFECYCLE_STATE, _canonical(state)),
        "caSet": _artifact_raw("active-attestation-ca-set.json", _canonical(ca)),
        "attestationStatus": _artifact_raw(
            "active-attestation-status.json", _canonical(status)
        ),
    }
    if receipt not in [
        dict(expected_receipt, status="attestation-lifecycle-initialized"),
        dict(expected_receipt, status="attestation-lifecycle-advanced"),
    ]:
        _fail("ATTESTATION_RECEIPT_REBIND_FAILED")
    return {
        "ok": True,
        "phase": "attestation-lifecycle-verified",
        "ca_set_version": ca["signed"]["version"],
        "status_version": status["signed"]["version"],
        "lifecycle_chain_head_sha256": chain,
    }


def advance_lifecycle(  # ruff: ignore[undocumented-public-function]
    *,
    previous_dir: Path,
    status_snapshot_path: Path,
    output_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    next_ca_set_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    prev = _regular_dir(previous_dir, "ATTESTATION_PREVIOUS_DIR_INVALID")
    verify_lifecycle(
        output_dir=prev,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    docs = _docs_from_dir(
        prev,
        {
            "release-attestation-lifecycle-bundle.json",
            _DOC_ATTESTATION_LIFECYCLE_STATE,
            "active-attestation-ca-set.json",
            "active-attestation-status.json",
            "release-attestation-lifecycle-receipt.json",
        },
        "ATTESTATION_PREVIOUS",
    )
    bundle = json.loads(json.dumps(docs["release-attestation-lifecycle-bundle.json"]))
    continuity_docs = bundle["continuityOutput"]
    previous_ca = docs["active-attestation-ca-set.json"]
    if next_ca_set_path is not None:
        raw, _ = _read(next_ca_set_path, "ATTESTATION_NEXT_CA_SET")
        ca = _verify_ca_set(
            raw,
            continuity_docs=continuity_docs,
            previous=previous_ca,
            now=current,
            historical=False,
        )
        revoked_before = set(_ca_revoked_union(bundle["caSets"]))
        new_pins = {x["sha256"] for x in ca["signed"]["trustRoots"]}
        if revoked_before & new_pins:
            _fail("ATTESTATION_REVOKED_CA_REINTRODUCED")
        bundle["caSets"].append(ca)
    else:
        ca = previous_ca
    status_raw, _ = _read(status_snapshot_path, "ATTESTATION_NEXT_STATUS")
    status = _verify_status_snapshot(
        status_raw,
        ca_set=ca,
        continuity_bundle=continuity_docs["release-root-continuity-bundle.json"],
        previous=docs["active-attestation-status.json"],
        now=current,
        historical=False,
    )
    _evaluate_status(
        continuity_bundle=continuity_docs["release-root-continuity-bundle.json"],
        ca_set=ca,
        status=status,
        now=current,
        historical=False,
    )
    inventory, bindings = _inventory(
        continuity_docs["release-root-continuity-bundle.json"], ca_set=ca
    )
    previous_bindings = docs[_DOC_ATTESTATION_LIFECYCLE_STATE]["deviceBindings"]
    for kid, value in previous_bindings.items():
        if kid in bindings and bindings[kid] != value:
            _fail("ATTESTATION_DEVICE_IDENTITY_CHANGED")
    bundle["statusSnapshots"].append(status)
    seq = len(bundle["events"]) + 1
    previous_head = docs[_DOC_ATTESTATION_LIFECYCLE_STATE]["lifecycleChainHeadSha256"]
    body = {
        "sequence": seq,
        "type": "lifecycle-advance",
        "previousChainHeadSha256": previous_head,
        "caSetSha256": _sha_bytes(_canonical(ca)),
        "statusSha256": _sha_bytes(_canonical(status)),
        "inventorySha256": _sha_bytes(_canonical({"certificates": inventory})),
        "deviceBindingsSha256": _sha_bytes(_canonical(bindings)),
    }
    chain = _lifecycle_head(previous_head, body)
    bundle["events"].append(dict(body, lifecycleChainHeadSha256=chain))
    state = _state(bundle, ca, status, chain, bindings)
    protected = [prev, status_snapshot_path] + (
        [next_ca_set_path] if next_ca_set_path else []
    )
    target = _outside(
        output_dir,
        [p for p in protected if p is not None],
        "ATTESTATION_ADVANCE_OUTPUT_INSIDE_INPUT",
    )
    if target.exists() or target.is_symlink():
        _fail("ATTESTATION_ADVANCE_OUTPUT_ALREADY_EXISTS")
    inputs = (
        [p for p in prev.rglob("*") if p.is_file()]
        + [status_snapshot_path]
        + ([next_ca_set_path] if next_ca_set_path else [])
    )
    before = {str(p.resolve()): _sha(p) for p in inputs}
    _write_output(
        target,
        bundle=bundle,
        state=state,
        ca_set=ca,
        status=status,
        input_files=inputs,
        before=before,
        phase="attestation-lifecycle-advanced",
    )
    final = verify_lifecycle(
        output_dir=target,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    return {
        "ok": True,
        "phase": "attestation-lifecycle-advanced",
        "ca_set_version": ca["signed"]["version"],
        "status_version": status["signed"]["version"],
        "lifecycle_chain_head_sha256": final["lifecycle_chain_head_sha256"],
    }


def _paths(values: list[str]) -> list[Path]:
    return [Path(x) for x in values]


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Verify release attestation lifecycle and certificate status"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def pins(p):
        p.add_argument("--bootstrap-root-sha256", required=True)
        p.add_argument("--recovery-root-sha256", required=True)
        p.add_argument("--attestation-root-sha256", action="append", required=True)

    p = sub.add_parser("initialize")
    pins(p)
    p.add_argument("--continuity-dir", type=Path, required=True)
    p.add_argument("--ca-set", type=Path, required=True)
    p.add_argument("--status", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("advance")
    pins(p)
    p.add_argument("--previous-dir", type=Path, required=True)
    p.add_argument("--status", type=Path, required=True)
    p.add_argument("--next-ca-set", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("verify")
    pins(p)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--historical", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "initialize":
        result = initialize_lifecycle(
            continuity_dir=args.continuity_dir,
            expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            expected_recovery_root_sha256=args.recovery_root_sha256,
            expected_attestation_root_sha256=args.attestation_root_sha256,
            ca_set_path=args.ca_set,
            status_snapshot_path=args.status,
            output_dir=args.output_dir,
        )
    elif args.cmd == "advance":
        result = advance_lifecycle(
            previous_dir=args.previous_dir,
            status_snapshot_path=args.status,
            output_dir=args.output_dir,
            expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            expected_recovery_root_sha256=args.recovery_root_sha256,
            expected_attestation_root_sha256=args.attestation_root_sha256,
            next_ca_set_path=args.next_ca_set,
        )
    else:
        result = verify_lifecycle(
            output_dir=args.output_dir,
            expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            expected_recovery_root_sha256=args.recovery_root_sha256,
            expected_attestation_root_sha256=args.attestation_root_sha256,
            historical=args.historical,
        )
    logger.info("%s", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
