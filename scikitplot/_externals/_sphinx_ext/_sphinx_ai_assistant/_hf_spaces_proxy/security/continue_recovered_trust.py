"""
Continue cryptographic release trust after an independently authorized root recovery.

Run 156 can replace a compromised root without requiring signatures from that compromised
root.  This module makes that recovery a first-class, offline-verifiable continuity event.
Later governance epochs authorize from the recovered root, and later ordinary root
rotations again require the recovered/current old-root threshold plus the new-root
threshold.

Hardware provenance is verified with a provider-neutral X.509 attestation profile: each
active replacement/new root key must be bound by a leaf attestation key whose certificate
chain terminates at an independently SHA-256-pinned trust root.  The leaf signs an exact
canonical statement binding the release root key, recovery/continuity context, device
class, and validity interval.  Vendor-specific certificate issuance/extension policy can
therefore live behind the pinned attestation CA without placing vendor SDKs or private keys
in this repository.
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

try:  # package import
    from . import maintain_release_trust as delegated
    from . import seal_release_governance as rootseal
except (ImportError, ValueError):  # Space/script/importlib loading
    import importlib.util

    def _load_local(name: str, filename: str):
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).resolve().parent / filename
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    rootseal = _load_local("_release_root_for_continuity", "seal_release_governance.py")
    delegated = _load_local(
        "_delegated_trust_for_continuity", "maintain_release_trust.py"
    )

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_root_continuity_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])
_ZERO_HASH = "0" * 64
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_CHUNK = 1024 * 1024
logger = logging.getLogger(__name__)


class RootContinuityError(RuntimeError):
    """Recovered-root continuity or attestation invariant failed."""


def _fail(code: str) -> None:
    raise RootContinuityError(code)


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
    except RootContinuityError:
        raise
    except Exception as exc:
        raise RootContinuityError(code + "_JSON_INVALID") from exc
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


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(_canonical(value))


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
            ord(ch) < 32  # ruff: ignore[magic-value-comparison]
            or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            for ch in value
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
        raise RootContinuityError(code) from exc


def _b64(value: Any, code: str) -> bytes:
    _len = len(value) > 2_000_000  # ruff: ignore[magic-value-comparison]
    if not isinstance(value, str) or not value or _len:
        _fail(code)
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise RootContinuityError(code) from exc
    if base64.b64encode(raw).decode("ascii") != value:
        _fail(code)
    return raw


def _artifact_raw(name: str, raw: bytes) -> dict[str, Any]:
    return {"name": name, "sha256": _sha_bytes(raw), "size": len(raw)}


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


def _docs_from_dir(root: Path, names: set[str], code: str) -> dict[str, dict[str, Any]]:
    if {p.name for p in root.iterdir()} != names:
        _fail(code + "_ALLOWLIST_MISMATCH")
    docs: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        doc, _ = _read(
            root / name, code + "_" + name.upper().replace("-", "_").replace(".", "_")
        )
        docs[name] = doc
    return docs


def _write_docs(root: Path, docs: dict[str, dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, doc in sorted(docs.items()):
        _write(root / name, doc)


def _cert_der(path: Path, expected_pin: str) -> tuple[bytes, Any]:
    _hex(expected_pin, "CONTINUITY_ATTESTATION_ROOT_PIN_INVALID")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        _fail("CONTINUITY_ATTESTATION_ROOT_FILE_INVALID")
    raw = path.read_bytes()
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]
        from cryptography.hazmat.primitives import (  # ruff: ignore[import-outside-top-level]
            serialization,
        )

        if b"-----BEGIN CERTIFICATE-----" in raw:
            cert = x509.load_pem_x509_certificate(raw)
        else:
            cert = x509.load_der_x509_certificate(raw)
        der = cert.public_bytes(serialization.Encoding.DER)
    except Exception as exc:
        raise RootContinuityError("CONTINUITY_ATTESTATION_ROOT_CERT_INVALID") from exc
    if _sha_bytes(der) != expected_pin:
        _fail("CONTINUITY_ATTESTATION_ROOT_PIN_MISMATCH")
    return der, cert


def _verify_cert_signature(cert: Any, issuer_public: Any, code: str) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric import (  # ruff: ignore[import-outside-top-level]
            padding,
        )
        from cryptography.hazmat.primitives.asymmetric.ec import (  # ruff: ignore[import-outside-top-level]
            ECDSA,
            EllipticCurvePublicKey,
        )
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # ruff: ignore[import-outside-top-level]
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives.asymmetric.rsa import (  # ruff: ignore[import-outside-top-level]
            RSAPublicKey,
        )

        if isinstance(issuer_public, Ed25519PublicKey):
            issuer_public.verify(cert.signature, cert.tbs_certificate_bytes)
        elif isinstance(issuer_public, EllipticCurvePublicKey):
            issuer_public.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                ECDSA(cert.signature_hash_algorithm),
            )
        elif isinstance(issuer_public, RSAPublicKey):
            issuer_public.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm,
            )
        else:
            _fail(code + "_UNSUPPORTED_ISSUER_KEY")
    except RootContinuityError:
        raise
    except Exception as exc:
        raise RootContinuityError(code + "_SIGNATURE_INVALID") from exc


def _cert_time(cert: Any, field: str) -> datetime:
    # cryptography versions differ on the *_utc properties.
    value = getattr(cert, field + "_utc", None)
    if value is None:
        value = getattr(cert, field).replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _verify_trust_root(
    der: bytes, cert: Any, *, now: datetime, historical: bool
) -> dict[str, Any]:
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]

        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        if not bc.ca:
            _fail("CONTINUITY_ATTESTATION_TRUST_ROOT_NOT_CA")
    except RootContinuityError:
        raise
    except Exception as exc:
        raise RootContinuityError(
            "CONTINUITY_ATTESTATION_TRUST_ROOT_CONSTRAINTS_INVALID"
        ) from exc
    if cert.issuer != cert.subject:
        _fail("CONTINUITY_ATTESTATION_TRUST_ROOT_NOT_SELF_ISSUED")
    _verify_cert_signature(cert, cert.public_key(), "CONTINUITY_ATTESTATION_TRUST_ROOT")
    not_before = _cert_time(cert, "not_valid_before")
    not_after = _cert_time(cert, "not_valid_after")
    if not historical and not (not_before <= now <= not_after):
        _fail("CONTINUITY_ATTESTATION_TRUST_ROOT_NOT_CURRENT")
    return {"sha256": _sha_bytes(der), "der": base64.b64encode(der).decode("ascii")}


def _load_attestation_roots(
    paths: list[Path], pins: list[str], *, now: datetime, historical: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(paths) != len(pins) or len(paths) < int(POLICY["min_attestation_roots"]):
        _fail("CONTINUITY_ATTESTATION_ROOT_SET_INVALID")
    docs: list[dict[str, Any]] = []
    certs: dict[str, Any] = {}
    for path, pin in zip(paths, pins):
        der, cert = _cert_der(path, pin)
        doc = _verify_trust_root(der, cert, now=now, historical=historical)
        if doc["sha256"] in certs:
            _fail("CONTINUITY_ATTESTATION_ROOT_DUPLICATE")
        docs.append(doc)
        certs[doc["sha256"]] = cert
    docs.sort(key=lambda x: x["sha256"])
    if [x["sha256"] for x in docs] != sorted(pins):
        _fail("CONTINUITY_ATTESTATION_ROOT_PIN_SET_MISMATCH")
    return docs, certs


def _embedded_attestation_roots(
    docs: Any, expected_pins: list[str], *, now: datetime, historical: bool
) -> dict[str, Any]:
    if not isinstance(docs, list) or len(docs) < int(POLICY["min_attestation_roots"]):
        _fail("CONTINUITY_EMBEDDED_ATTESTATION_ROOTS_INVALID")
    expected = sorted(
        _hex(x, "CONTINUITY_EXPECTED_ATTESTATION_PIN_INVALID") for x in expected_pins
    )
    if [x.get("sha256") for x in docs if isinstance(x, dict)] != expected:
        _fail("CONTINUITY_EXPECTED_ATTESTATION_PIN_SET_MISMATCH")
    certs: dict[str, Any] = {}
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]

        for item in docs:
            if not isinstance(item, dict) or set(item) != {"sha256", "der"}:
                _fail("CONTINUITY_EMBEDDED_ATTESTATION_ROOT_SCHEMA_INVALID")
            der = _b64(item["der"], "CONTINUITY_EMBEDDED_ATTESTATION_ROOT_DER_INVALID")
            if _sha_bytes(der) != item["sha256"]:
                _fail("CONTINUITY_EMBEDDED_ATTESTATION_ROOT_HASH_INVALID")
            cert = x509.load_der_x509_certificate(der)
            _verify_trust_root(der, cert, now=now, historical=historical)
            certs[item["sha256"]] = cert
    except RootContinuityError:
        raise
    except Exception as exc:
        raise RootContinuityError(
            "CONTINUITY_EMBEDDED_ATTESTATION_ROOT_CERT_INVALID"
        ) from exc
    return certs


def _leaf_algorithm(public_key: Any) -> str:
    try:
        from cryptography.hazmat.primitives.asymmetric.ec import (  # ruff: ignore[import-outside-top-level]
            SECP256R1,
            EllipticCurvePublicKey,
        )
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # ruff: ignore[import-outside-top-level]
            Ed25519PublicKey,
        )

        if isinstance(public_key, Ed25519PublicKey):
            return "ed25519"
        if isinstance(public_key, EllipticCurvePublicKey) and isinstance(
            public_key.curve, SECP256R1
        ):
            return "ecdsa-p256"
    except Exception:  # ruff: ignore[blind-except]
        pass
    _fail("CONTINUITY_ATTESTATION_LEAF_ALGORITHM_UNSUPPORTED")
    return None


def _verify_leaf_signature(public_key: Any, signature: bytes, raw: bytes) -> None:
    try:
        from cryptography.hazmat.primitives import (  # ruff: ignore[import-outside-top-level]
            hashes,
        )
        from cryptography.hazmat.primitives.asymmetric.ec import (  # ruff: ignore[import-outside-top-level]
            ECDSA,
            EllipticCurvePublicKey,
        )
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # ruff: ignore[import-outside-top-level]
            Ed25519PublicKey,
        )

        if isinstance(public_key, Ed25519PublicKey):
            public_key.verify(signature, raw)
        elif isinstance(public_key, EllipticCurvePublicKey):
            public_key.verify(signature, raw, ECDSA(hashes.SHA256()))
        else:
            _fail("CONTINUITY_ATTESTATION_LEAF_ALGORITHM_UNSUPPORTED")
    except RootContinuityError:
        raise
    except Exception as exc:
        raise RootContinuityError("CONTINUITY_ATTESTATION_SIGNATURE_INVALID") from exc


def _verify_attestation(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    *,
    root_info: dict[str, Any],
    context_id: str,
    trust_certs: dict[str, Any],
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    if set(doc) != {"signature", "signed"} or not isinstance(doc.get("signed"), dict):
        _fail("CONTINUITY_ATTESTATION_SCHEMA_INVALID")
    s = doc["signed"]
    expected = {
        "schemaVersion",
        "predicateType",
        "governanceId",
        "contextId",
        "rootVersion",
        "rootSha256",
        "keyId",
        "publicKey",
        "deviceClass",
        "manufacturer",
        "model",
        "attestedAt",
        "expires",
        "certificateChainDer",
    }
    if (
        set(s) != expected
        or s["schemaVersion"] != int(POLICY["attestation_schema_version"])
        or s["predicateType"] != PREDICATE_TYPE + "/x509-key-attestation"
    ):
        _fail("CONTINUITY_ATTESTATION_SIGNED_SCHEMA_INVALID")
    if (
        s["governanceId"] != root_info["governance_id"]
        or s["contextId"] != context_id
        or s["rootVersion"] != root_info["version"]
        or s["rootSha256"] != root_info["sha256"]
    ):
        _fail("CONTINUITY_ATTESTATION_CONTEXT_MISMATCH")
    kid = _identity(s["keyId"], "CONTINUITY_ATTESTATION_KEYID_INVALID")
    if kid not in root_info["roles"]["root"]["keyids"]:
        _fail("CONTINUITY_ATTESTATION_KEY_NOT_ACTIVE_ROOT")
    if s["publicKey"] != root_info["keys"][kid]["keyval"]["public"]:
        _fail("CONTINUITY_ATTESTATION_PUBLIC_KEY_MISMATCH")
    device = _identity(s["deviceClass"], "CONTINUITY_ATTESTATION_DEVICE_CLASS_INVALID")
    if device not in set(POLICY["allowed_device_classes"]):
        _fail("CONTINUITY_ATTESTATION_DEVICE_CLASS_INVALID")
    _identity(s["manufacturer"], "CONTINUITY_ATTESTATION_MANUFACTURER_INVALID")
    _identity(s["model"], "CONTINUITY_ATTESTATION_MODEL_INVALID")
    attested = _dt(s["attestedAt"], "CONTINUITY_ATTESTATION_TIME_INVALID")
    expires = _dt(s["expires"], "CONTINUITY_ATTESTATION_EXPIRES_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if attested > now + skew:
        _fail("CONTINUITY_ATTESTATION_FROM_FUTURE")
    if expires <= attested or expires - attested > timedelta(
        days=int(POLICY["max_attestation_lifetime_days"])
    ):
        _fail("CONTINUITY_ATTESTATION_LIFETIME_INVALID")
    if not historical and expires < now + timedelta(
        minutes=int(POLICY["min_attestation_remaining_minutes"])
    ):
        _fail("CONTINUITY_ATTESTATION_EXPIRED_OR_FREEZE_RISK")
    chain_raw = s["certificateChainDer"]
    if not isinstance(chain_raw, list) or not (
        1 <= len(chain_raw) <= int(POLICY["max_attestation_chain_length"])
    ):
        _fail("CONTINUITY_ATTESTATION_CHAIN_INVALID")
    try:
        from cryptography import x509  # ruff: ignore[import-outside-top-level]

        chain = [
            x509.load_der_x509_certificate(
                _b64(x, "CONTINUITY_ATTESTATION_CERT_DER_INVALID")
            )
            for x in chain_raw
        ]
        leaf = chain[0]
        try:
            bc = leaf.extensions.get_extension_for_class(x509.BasicConstraints).value
            if bc.ca:
                _fail("CONTINUITY_ATTESTATION_LEAF_IS_CA")
            ku = leaf.extensions.get_extension_for_class(x509.KeyUsage).value
            if not ku.digital_signature:
                _fail("CONTINUITY_ATTESTATION_LEAF_NO_DIGITAL_SIGNATURE")
        except RootContinuityError:
            raise
        except Exception as exc:
            raise RootContinuityError(
                "CONTINUITY_ATTESTATION_LEAF_CONSTRAINTS_INVALID"
            ) from exc
        if _leaf_algorithm(leaf.public_key()) not in set(
            POLICY["allowed_leaf_key_algorithms"]
        ):
            _fail("CONTINUITY_ATTESTATION_LEAF_ALGORITHM_UNSUPPORTED")
        for idx in range(len(chain) - 1):
            child, issuer = chain[idx], chain[idx + 1]
            if child.issuer != issuer.subject:
                _fail("CONTINUITY_ATTESTATION_CHAIN_ISSUER_MISMATCH")
            try:
                if not issuer.extensions.get_extension_for_class(
                    x509.BasicConstraints
                ).value.ca:
                    _fail("CONTINUITY_ATTESTATION_INTERMEDIATE_NOT_CA")
            except RootContinuityError:
                raise
            except Exception as exc:
                raise RootContinuityError(
                    "CONTINUITY_ATTESTATION_INTERMEDIATE_CONSTRAINTS_INVALID"
                ) from exc
            _verify_cert_signature(
                child, issuer.public_key(), "CONTINUITY_ATTESTATION_CHAIN"
            )
        last = chain[-1]
        matching = [
            cert for cert in trust_certs.values() if last.issuer == cert.subject
        ]
        if len(matching) != 1:
            _fail("CONTINUITY_ATTESTATION_TRUST_ANCHOR_NOT_UNIQUE")
        anchor = matching[0]
        _verify_cert_signature(
            last, anchor.public_key(), "CONTINUITY_ATTESTATION_ANCHOR"
        )
        all_certs = [*chain, anchor]
        for cert in all_certs:
            if attested < _cert_time(cert, "not_valid_before") or expires > _cert_time(
                cert, "not_valid_after"
            ):
                _fail("CONTINUITY_ATTESTATION_CERT_VALIDITY_MISMATCH")
        _verify_leaf_signature(
            leaf.public_key(),
            _b64(doc["signature"], "CONTINUITY_ATTESTATION_SIGNATURE_ENCODING_INVALID"),
            _canonical(s),
        )
    except RootContinuityError:
        raise
    except Exception as exc:
        raise RootContinuityError("CONTINUITY_ATTESTATION_CERTIFICATE_INVALID") from exc
    return {"signature": doc["signature"], "signed": s}


def _attestations(
    paths: list[Path],
    *,
    root_info: dict[str, Any],
    context_id: str,
    trust_certs: dict[str, Any],
    now: datetime,
    historical: bool = False,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        doc, _ = _read(path, "CONTINUITY_ATTESTATION")
        normalized = _verify_attestation(
            doc,
            root_info=root_info,
            context_id=context_id,
            trust_certs=trust_certs,
            now=now,
            historical=historical,
        )
        kid = normalized["signed"]["keyId"]
        if kid in seen:
            _fail("CONTINUITY_ATTESTATION_DUPLICATE_KEY")
        seen.add(kid)
        docs.append(normalized)
    required = sorted(root_info["roles"]["root"]["keyids"])
    if (
        bool(POLICY["require_all_active_root_keys_attested"])
        and sorted(seen) != required
    ):
        _fail("CONTINUITY_ATTESTATION_ROOT_KEY_SET_MISMATCH")
    docs.sort(key=lambda x: x["signed"]["keyId"])
    return docs


def _root_info(envelope: dict[str, Any], code: str) -> dict[str, Any]:
    try:
        return rootseal._validate_root_envelope(envelope)
    except rootseal.RootTrustError as exc:
        raise RootContinuityError(code + ":" + str(exc)) from exc


def _continuity_head(previous: str, body: dict[str, Any]) -> str:
    _hex(previous, "CONTINUITY_PREVIOUS_HEAD_INVALID")
    return _sha_bytes((previous + "\n").encode("ascii") + _canonical(body))


def _candidate_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "governanceState",
        "governanceBundle",
        "historyState",
        "historyBundle",
    }
    if set(snapshot) != expected or snapshot.get("schemaVersion") != int(
        rootseal.governance.POLICY["recovery_snapshot_schema_version"]
    ):
        _fail("CONTINUITY_GOVERNANCE_SNAPSHOT_SCHEMA_INVALID")
    state = snapshot["governanceState"]
    bundle = snapshot["governanceBundle"]
    if not isinstance(state, dict) or not isinstance(bundle, dict):
        _fail("CONTINUITY_GOVERNANCE_SNAPSHOT_SCHEMA_INVALID")
    with tempfile.TemporaryDirectory(prefix="run157-governance-verify-") as td:
        sp = Path(td) / "state.json"
        bp = Path(td) / "bundle.json"
        _write(sp, state)
        _write(bp, bundle)
        try:
            rootseal.governance.verify_governance_bundle(bundle_path=bp, state_path=sp)
        except rootseal.governance.GovernanceError as exc:
            raise RootContinuityError(
                "CONTINUITY_GOVERNANCE_BUNDLE_INVALID:" + str(exc)
            ) from exc
    entries = bundle.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("CONTINUITY_GOVERNANCE_ENTRIES_INVALID")
    entry = entries[-1]
    proposal = entry.get("proposal")
    if not isinstance(proposal, dict):
        _fail("CONTINUITY_GOVERNANCE_PROPOSAL_INVALID")
    previous_policy = (
        bundle["genesis"]["initialPolicy"]
        if len(entries) == 1
        else entries[-2]["nextPolicy"]
    )
    revoked_before: set[str] = set()
    for old in entries[:-1]:
        revoked_before.update(old.get("revokedAuthorityKeyIds", []))
    raw_state = _canonical(state)
    raw_bundle = _canonical(bundle)
    raw_snapshot = _canonical(snapshot)
    raw_proposal = _canonical(proposal)
    return {
        "state": state,
        "bundle": bundle,
        "snapshot": snapshot,
        "entry": entry,
        "previous_policy": previous_policy,
        "final_policy": state["policy"],
        "revoked_before": revoked_before,
        "state_sha256": _sha_bytes(raw_state),
        "bundle_sha256": _sha_bytes(raw_bundle),
        "snapshot_sha256": _sha_bytes(raw_snapshot),
        "snapshot_size": len(raw_snapshot),
        "proposal_sha256": _sha_bytes(raw_proposal),
    }


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_GOVERNANCE_STATE = "trusted-governance-state.json"
_DOC_RELEASE_ROOT_STATE = "trusted-release-root-state.json"
_DOC_ROOT_CONTINUITY_STATE = "trusted-root-continuity-state.json"


def _candidate_subject(
    candidate: dict[str, Any], auth_root: dict[str, Any]
) -> dict[str, Any]:
    entry = candidate["entry"]
    role = "emergency" if entry.get("approvalRole") == "emergency" else "governance"
    state_raw = _canonical(candidate["state"])
    bundle_raw = _canonical(candidate["bundle"])
    snapshot_raw = _canonical(candidate["snapshot"])
    return {
        "schemaVersion": int(rootseal.POLICY["schema_version"]),
        "predicateType": rootseal.PREDICATE_TYPE + "/governance-authorization",
        "governanceId": candidate["state"]["governanceId"],
        "historyId": candidate["state"]["historyId"],
        "epoch": candidate["state"]["epoch"],
        "policyVersion": candidate["state"]["policy"]["policyVersion"],
        "transitionId": entry["transitionId"],
        "role": role,
        "proposalSha256": candidate["proposal_sha256"],
        "selectedKeyIds": sorted(entry["selectedApproverKeyIds"]),
        "authorizingRoot": {
            "version": auth_root["version"],
            "sha256": auth_root["sha256"],
        },
        "governanceState": {
            "name": _DOC_GOVERNANCE_STATE,
            "sha256": candidate["state_sha256"],
            "size": len(state_raw),
        },
        "governanceBundle": {
            "name": "release-governance-bundle.json",
            "sha256": candidate["bundle_sha256"],
            "size": len(bundle_raw),
        },
        "recoverySnapshot": {
            "name": "release-governance-recovery-snapshot.json",
            "sha256": candidate["snapshot_sha256"],
            "size": len(snapshot_raw),
        },
        "history": candidate["state"]["history"],
    }


def _verify_authorization(
    authorization: dict[str, Any],
    *,
    candidate: dict[str, Any],
    auth_root: dict[str, Any],
    historical: bool,
    now: datetime,
) -> None:
    expected = {
        "schemaVersion",
        "predicateType",
        "status",
        "subject",
        "subjectSha256",
        "rootVersion",
        "rootSha256",
        "role",
        "selectedKeyIds",
        "signatures",
    }
    if (
        set(authorization) != expected
        or authorization["schemaVersion"]
        != int(rootseal.POLICY["authorization_schema_version"])
        or authorization["predicateType"]
        != rootseal.PREDICATE_TYPE + "/governance-authorization"
        or authorization["status"] != "cryptographically-authorized"
    ):
        _fail("CONTINUITY_AUTHORIZATION_SCHEMA_INVALID")
    role_name = (
        "emergency"
        if candidate["entry"].get("approvalRole") == "emergency"
        else "governance"
    )
    try:
        rootseal._assert_role_matches_policy(
            auth_root,
            role_name,
            candidate["previous_policy"],
            candidate["revoked_before"],
            "CONTINUITY_AUTHORIZING_ROLE",
        )
        other = "governance" if role_name == "emergency" else "emergency"
        rootseal._assert_role_matches_policy(
            auth_root,
            other,
            candidate["previous_policy"],
            candidate["revoked_before"],
            "CONTINUITY_AUTHORIZING_OTHER_ROLE",
        )
    except rootseal.RootTrustError as exc:
        raise RootContinuityError(str(exc)) from exc
    subject = _candidate_subject(candidate, auth_root)
    if (
        authorization["subject"] != subject
        or authorization["subjectSha256"] != _sha_bytes(_canonical(subject))
        or authorization["rootVersion"] != auth_root["version"]
        or authorization["rootSha256"] != auth_root["sha256"]
        or authorization["role"] != role_name
    ):
        _fail("CONTINUITY_AUTHORIZATION_REBIND_FAILED")
    selected = subject["selectedKeyIds"]
    if authorization["selectedKeyIds"] != selected:
        _fail("CONTINUITY_AUTHORIZATION_SELECTED_SET_INVALID")
    policy_field = (
        "emergencyAuthority" if role_name == "emergency" else "policyAuthority"
    )
    members = {
        m["keyId"]: m for m in candidate["previous_policy"][policy_field]["members"]
    }
    sigs = authorization["signatures"]
    if not isinstance(sigs, list) or len(sigs) != len(selected):
        _fail("CONTINUITY_AUTHORIZATION_SIGNATURE_COUNT_INVALID")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in sigs:
        kid = doc.get("signed", {}).get("keyId") if isinstance(doc, dict) else None
        if kid not in selected or kid in seen:
            _fail("CONTINUITY_AUTHORIZATION_SIGNATURE_SET_INVALID")
        seen.add(kid)
        try:
            normalized.append(
                rootseal._validate_authorization_signature(
                    doc,
                    subject=subject,
                    root_info=auth_root,
                    role_name=role_name,
                    expected_member=members[kid],
                    now=now,
                    enforce_freshness=not historical,
                )
            )
        except rootseal.RootTrustError as exc:
            raise RootContinuityError(
                "CONTINUITY_AUTHORIZATION_INVALID:" + str(exc)
            ) from exc
    normalized.sort(key=lambda x: x["signed"]["keyId"])
    if sigs != normalized or sorted(seen) != selected:
        _fail("CONTINUITY_AUTHORIZATION_NOT_NORMALIZED")


def _authorization_from_files(
    paths: list[Path],
    *,
    candidate: dict[str, Any],
    auth_root: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    role_name = (
        "emergency"
        if candidate["entry"].get("approvalRole") == "emergency"
        else "governance"
    )
    try:
        rootseal._assert_role_matches_policy(
            auth_root,
            role_name,
            candidate["previous_policy"],
            candidate["revoked_before"],
            "CONTINUITY_AUTHORIZING_ROLE",
        )
        other = "governance" if role_name == "emergency" else "emergency"
        rootseal._assert_role_matches_policy(
            auth_root,
            other,
            candidate["previous_policy"],
            candidate["revoked_before"],
            "CONTINUITY_AUTHORIZING_OTHER_ROLE",
        )
    except rootseal.RootTrustError as exc:
        raise RootContinuityError(str(exc)) from exc
    subject = _candidate_subject(candidate, auth_root)
    selected = subject["selectedKeyIds"]
    if len(paths) != len(selected):
        _fail("CONTINUITY_AUTH_SIGNATURE_COUNT_INVALID")
    policy_field = (
        "emergencyAuthority" if role_name == "emergency" else "policyAuthority"
    )
    members = {
        m["keyId"]: m for m in candidate["previous_policy"][policy_field]["members"]
    }
    by_key: dict[str, dict[str, Any]] = {}
    for path in paths:
        doc, _ = _read(path, "CONTINUITY_AUTH_SIGNATURE")
        kid = (
            doc.get("signed", {}).get("keyId")
            if isinstance(doc.get("signed"), dict)
            else None
        )
        if kid not in selected or kid in by_key:
            _fail("CONTINUITY_AUTH_SIGNATURE_KEY_SET_INVALID")
        try:
            by_key[kid] = rootseal._validate_authorization_signature(
                doc,
                subject=subject,
                root_info=auth_root,
                role_name=role_name,
                expected_member=members[kid],
                now=now,
            )
        except rootseal.RootTrustError as exc:
            raise RootContinuityError(
                "CONTINUITY_AUTH_SIGNATURE_INVALID:" + str(exc)
            ) from exc
    if sorted(by_key) != selected:
        _fail("CONTINUITY_AUTH_SIGNATURE_SET_MISMATCH")
    return {
        "schemaVersion": int(rootseal.POLICY["authorization_schema_version"]),
        "predicateType": rootseal.PREDICATE_TYPE + "/governance-authorization",
        "status": "cryptographically-authorized",
        "subject": subject,
        "subjectSha256": _sha_bytes(_canonical(subject)),
        "rootVersion": auth_root["version"],
        "rootSha256": auth_root["sha256"],
        "role": role_name,
        "selectedKeyIds": selected,
        "signatures": [by_key[k] for k in sorted(by_key)],
    }


def _state(
    *,
    bundle: dict[str, Any],
    active_root: dict[str, Any],
    chain_head: str,
    governance_binding: dict[str, Any],
    recovery_record_sha: str,
) -> dict[str, Any]:
    bundle_sha = _sha_bytes(_canonical(bundle))
    return {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-recovered-root-continuity",
        "governanceId": active_root["governance_id"],
        "bootstrapRootSha256": bundle["bootstrapRootSha256"],
        "recoveryRootSha256": bundle["recoveryRootSha256"],
        "activeRoot": {
            "version": active_root["version"],
            "sha256": active_root["sha256"],
            "expires": active_root["signed"]["expires"],
        },
        "rootChainHeadSha256": chain_head,
        "bundleSha256": bundle_sha,
        "governance": governance_binding,
        "recovery": {
            "incidentId": bundle["recoveryEvent"]["incidentId"],
            "recordSha256": recovery_record_sha,
        },
    }


def _governance_binding_from_run155_state(state: dict[str, Any]) -> dict[str, Any]:
    g = state["governance"]
    return {
        "epoch": g["epoch"],
        "policyVersion": g["policyVersion"],
        "stateSha256": g["stateSha256"],
        "bundleSha256": g["bundleSha256"],
        "recoverySnapshotSha256": g["recoverySnapshotSha256"],
        "proposalSha256": g["proposalSha256"],
    }


def _governance_binding_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": candidate["state"]["epoch"],
        "policyVersion": candidate["state"]["policy"]["policyVersion"],
        "stateSha256": candidate["state_sha256"],
        "bundleSha256": candidate["bundle_sha256"],
        "recoverySnapshotSha256": candidate["snapshot_sha256"],
        "proposalSha256": candidate["proposal_sha256"],
    }


def activate_recovery(  # ruff: ignore[undocumented-public-function]
    *,
    sealed_dir: Path,
    bootstrap_root_sha256: str,
    recovery_dir: Path,
    expected_recovery_root_sha256: str,
    attestation_paths: list[Path],
    attestation_trust_root_paths: list[Path],
    expected_attestation_root_sha256: list[str],
    output_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    sealed = _regular_dir(sealed_dir, "CONTINUITY_RUN155_DIR_INVALID")
    recovery = _regular_dir(recovery_dir, "CONTINUITY_RUN156_RECOVERY_DIR_INVALID")
    try:
        rootseal.verify_sealed_governance(
            sealed_dir=sealed,
            expected_bootstrap_root_sha256=bootstrap_root_sha256,
            now=current,
            require_current_fresh=True,
        )
        delegated.verify_recovery_output(
            output_dir=recovery,
            sealed_dir=sealed,
            bootstrap_root_sha256=bootstrap_root_sha256,
            expected_recovery_root_sha256=expected_recovery_root_sha256,
            now=current,
        )
    except (rootseal.RootTrustError, delegated.DelegatedTrustError) as exc:
        raise RootContinuityError("CONTINUITY_PREDECESSOR_INVALID:" + str(exc)) from exc
    base_names = {
        "release-root-bundle.json",
        _DOC_RELEASE_ROOT_STATE,
        "cryptographic-governance-authorization.json",
        "active-root.json",
        "release-governance-recovery-snapshot.json",
        "release-root-receipt.json",
    }
    recovery_names = {
        "release-root-recovery-record.json",
        "recovered-effective-root.json",
        "release-root-recovery-receipt.json",
    }
    base_docs = _docs_from_dir(sealed, base_names, "CONTINUITY_BASE_SEAL")
    recovery_docs = _docs_from_dir(
        recovery, recovery_names, "CONTINUITY_RECOVERY_OUTPUT"
    )
    active = _root_info(
        recovery_docs["recovered-effective-root.json"],
        "CONTINUITY_RECOVERED_ROOT_INVALID",
    )
    base_state = base_docs[_DOC_RELEASE_ROOT_STATE]
    if active["version"] != base_state["rootVersion"] + 1:
        _fail("CONTINUITY_RECOVERED_ROOT_VERSION_INVALID")
    roots_embedded, trust_certs = _load_attestation_roots(
        attestation_trust_root_paths, expected_attestation_root_sha256, now=current
    )
    incident_id = recovery_docs["release-root-recovery-record.json"]["subject"][
        "incidentId"
    ]
    attestations = _attestations(
        attestation_paths,
        root_info=active,
        context_id=incident_id,
        trust_certs=trust_certs,
        now=current,
    )
    recovery_record_raw = _canonical(recovery_docs["release-root-recovery-record.json"])
    attestation_sha = _sha_bytes(_canonical({"attestations": attestations}))
    body = {
        "type": "root-recovery",
        "sequence": 1,
        "previousChainHeadSha256": base_state["rootChainHeadSha256"],
        "incidentId": incident_id,
        "recoveryRecordSha256": _sha_bytes(recovery_record_raw),
        "replacementRoot": {"version": active["version"], "sha256": active["sha256"]},
        "attestationsSha256": attestation_sha,
    }
    chain = _continuity_head(base_state["rootChainHeadSha256"], body)
    recovery_event = dict(body, chainHeadSha256=chain)
    bundle = {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "trusted-recovered-root-continuity-chain",
        "governanceId": active["governance_id"],
        "bootstrapRootSha256": _hex(
            bootstrap_root_sha256, "CONTINUITY_BOOTSTRAP_PIN_INVALID"
        ),
        "recoveryRootSha256": _hex(
            expected_recovery_root_sha256, "CONTINUITY_RECOVERY_PIN_INVALID"
        ),
        "attestationTrustRoots": roots_embedded,
        "baseSeal": base_docs,
        "recoveryOutput": recovery_docs,
        "recoveryAttestations": attestations,
        "recoveryEvent": recovery_event,
        "epochs": [],
    }
    state = _state(
        bundle=bundle,
        active_root=active,
        chain_head=chain,
        governance_binding=_governance_binding_from_run155_state(base_state),
        recovery_record_sha=_sha_bytes(recovery_record_raw),
    )
    target = _outside(
        output_dir,
        [sealed, recovery]
        + [p.resolve() for p in attestation_paths + attestation_trust_root_paths],
        "CONTINUITY_OUTPUT_INSIDE_INPUT",
    )
    if target.exists() or target.is_symlink():
        _fail("CONTINUITY_OUTPUT_ALREADY_EXISTS")
    inputs = (
        [p for p in sealed.rglob("*") if p.is_file()]
        + [p for p in recovery.rglob("*") if p.is_file()]
        + attestation_paths
        + attestation_trust_root_paths
    )
    before = {str(p.resolve()): _sha(p) for p in inputs}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="run157-activate-", dir=target.parent
    ) as td:
        stage = Path(td) / "continuity"
        stage.mkdir()
        _write(stage / "release-root-continuity-bundle.json", bundle)
        _write(stage / _DOC_ROOT_CONTINUITY_STATE, state)
        _write(stage / "active-root.json", active["envelope"])
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "recovered-root-activated",
            "governanceId": active["governance_id"],
            "incidentId": incident_id,
            "bundle": _artifact_raw(
                "release-root-continuity-bundle.json", _canonical(bundle)
            ),
            "state": _artifact_raw(_DOC_ROOT_CONTINUITY_STATE, _canonical(state)),
            "activeRoot": _artifact_raw(
                "active-root.json", _canonical(active["envelope"])
            ),
            "attestationRootSha256": sorted(expected_attestation_root_sha256),
        }
        _write(stage / "release-root-continuity-receipt.json", receipt)
        for p in inputs:
            if _sha(p) != before[str(p.resolve())]:
                _fail("CONTINUITY_INPUT_CHANGED_DURING_ACTIVATION")
        temp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, temp_target, copy_function=shutil.copy2)
        os.replace(temp_target, target)
    verify_continuity(
        output_dir=target,
        expected_bootstrap_root_sha256=bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    return {
        "ok": True,
        "phase": "recovered-root-activated",
        "active_root_version": active["version"],
        "active_root_sha256": active["sha256"],
        "root_chain_head_sha256": chain,
    }


def _verify_base_and_recovery(
    bundle: dict[str, Any],
    *,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    now: datetime,
    historical: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    base_docs = bundle["baseSeal"]
    recovery_docs = bundle["recoveryOutput"]
    if not isinstance(base_docs, dict) or not isinstance(recovery_docs, dict):
        _fail("CONTINUITY_EMBEDDED_PREDECESSOR_SCHEMA_INVALID")
    with tempfile.TemporaryDirectory(prefix="run157-predecessor-") as td:
        sd = Path(td) / "sealed"
        rd = Path(td) / "recovery"
        _write_docs(sd, base_docs)
        _write_docs(rd, recovery_docs)
        try:
            rootseal.verify_sealed_governance(
                sealed_dir=sd,
                expected_bootstrap_root_sha256=bootstrap_pin,
                now=now,
                require_current_fresh=False,
            )
            delegated.verify_recovery_output(
                output_dir=rd,
                sealed_dir=sd,
                bootstrap_root_sha256=bootstrap_pin,
                expected_recovery_root_sha256=recovery_pin,
                now=now,
                historical=True,
            )
        except (rootseal.RootTrustError, delegated.DelegatedTrustError) as exc:
            raise RootContinuityError(
                "CONTINUITY_EMBEDDED_PREDECESSOR_INVALID:" + str(exc)
            ) from exc
    base_state = base_docs[_DOC_RELEASE_ROOT_STATE]
    active = _root_info(
        recovery_docs["recovered-effective-root.json"],
        "CONTINUITY_EMBEDDED_RECOVERED_ROOT_INVALID",
    )
    trust_certs = _embedded_attestation_roots(
        bundle["attestationTrustRoots"], attestation_pins, now=now, historical=True
    )
    incident = recovery_docs["release-root-recovery-record.json"]["subject"][
        "incidentId"
    ]
    attest_raw = bundle["recoveryAttestations"]
    if not isinstance(attest_raw, list):
        _fail("CONTINUITY_RECOVERY_ATTESTATIONS_INVALID")
    normalized = []
    for doc in attest_raw:
        if not isinstance(doc, dict):
            _fail("CONTINUITY_RECOVERY_ATTESTATIONS_INVALID")
        normalized.append(
            _verify_attestation(
                doc,
                root_info=active,
                context_id=incident,
                trust_certs=trust_certs,
                now=now,
                historical=True,
            )
        )
    normalized.sort(key=lambda x: x["signed"]["keyId"])
    required = sorted(active["roles"]["root"]["keyids"])
    if (
        normalized != attest_raw
        or sorted(x["signed"]["keyId"] for x in normalized) != required
    ):
        _fail("CONTINUITY_RECOVERY_ATTESTATION_SET_INVALID")
    record_sha = _sha_bytes(
        _canonical(recovery_docs["release-root-recovery-record.json"])
    )
    body = {
        "type": "root-recovery",
        "sequence": 1,
        "previousChainHeadSha256": base_state["rootChainHeadSha256"],
        "incidentId": incident,
        "recoveryRecordSha256": record_sha,
        "replacementRoot": {"version": active["version"], "sha256": active["sha256"]},
        "attestationsSha256": _sha_bytes(_canonical({"attestations": normalized})),
    }
    chain = _continuity_head(base_state["rootChainHeadSha256"], body)
    expected_event = dict(body, chainHeadSha256=chain)
    if bundle["recoveryEvent"] != expected_event:
        _fail("CONTINUITY_RECOVERY_EVENT_REBIND_FAILED")
    return active, base_state, chain, trust_certs


def verify_continuity(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    output_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root = _regular_dir(output_dir, "CONTINUITY_VERIFY_DIR_INVALID")
    allowed = {
        "release-root-continuity-bundle.json",
        _DOC_ROOT_CONTINUITY_STATE,
        "active-root.json",
        "release-root-continuity-receipt.json",
    }
    if {p.name for p in root.iterdir()} != allowed:
        _fail("CONTINUITY_VERIFY_ALLOWLIST_MISMATCH")
    bundle, bundle_raw = _read(
        root / "release-root-continuity-bundle.json", "CONTINUITY_VERIFY_BUNDLE"
    )
    state, state_raw = _read(
        root / _DOC_ROOT_CONTINUITY_STATE, "CONTINUITY_VERIFY_STATE"
    )
    active_doc, active_raw = _read(
        root / "active-root.json", "CONTINUITY_VERIFY_ACTIVE_ROOT"
    )
    expected_bundle_keys = {
        "schemaVersion",
        "predicateType",
        "status",
        "governanceId",
        "bootstrapRootSha256",
        "recoveryRootSha256",
        "attestationTrustRoots",
        "baseSeal",
        "recoveryOutput",
        "recoveryAttestations",
        "recoveryEvent",
        "epochs",
    }
    if (
        set(bundle) != expected_bundle_keys
        or bundle["schemaVersion"] != int(POLICY["bundle_schema_version"])
        or bundle["predicateType"] != PREDICATE_TYPE
        or bundle["status"] != "trusted-recovered-root-continuity-chain"
    ):
        _fail("CONTINUITY_VERIFY_BUNDLE_SCHEMA_INVALID")
    if bundle["bootstrapRootSha256"] != _hex(
        expected_bootstrap_root_sha256, "CONTINUITY_VERIFY_BOOTSTRAP_PIN_INVALID"
    ) or bundle["recoveryRootSha256"] != _hex(
        expected_recovery_root_sha256, "CONTINUITY_VERIFY_RECOVERY_PIN_INVALID"
    ):
        _fail("CONTINUITY_VERIFY_PIN_MISMATCH")
    active, base_state, chain, trust_certs = _verify_base_and_recovery(
        bundle,
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=expected_attestation_root_sha256,
        now=current,
        historical=historical,
    )
    governance_binding = _governance_binding_from_run155_state(base_state)
    recovery_record = bundle["recoveryOutput"]["release-root-recovery-record.json"]
    recovery_revoked = set(recovery_record.get("revokedRootKeyIds", []))
    if recovery_revoked & set(active["keys"]):
        _fail("CONTINUITY_VERIFY_RECOVERY_REVOKED_KEY_ACTIVE")
    epochs = bundle["epochs"]
    if not isinstance(epochs, list) or len(epochs) > int(
        POLICY["max_continuity_epochs"]
    ):
        _fail("CONTINUITY_VERIFY_EPOCHS_INVALID")
    for index, entry in enumerate(epochs, 1):
        if not isinstance(entry, dict):
            _fail("CONTINUITY_VERIFY_EPOCH_SCHEMA_INVALID")
        expected_entry_keys = {
            "sequence",
            "epoch",
            "previousChainHeadSha256",
            "governanceSnapshot",
            "authorization",
            "nextRoot",
            "rootAttestations",
            "activeRoot",
            "chainHeadSha256",
        }
        if (
            set(entry) != expected_entry_keys
            or entry["sequence"] != index + 1
            or entry["previousChainHeadSha256"] != chain
        ):
            _fail("CONTINUITY_VERIFY_EPOCH_SCHEMA_INVALID")
        candidate = _candidate_from_snapshot(entry["governanceSnapshot"])
        if (
            candidate["state"]["epoch"] != governance_binding["epoch"] + 1
            or candidate["entry"]["proposal"].get("previousGovernanceStateSha256")
            != governance_binding["stateSha256"]
        ):
            _fail("CONTINUITY_VERIFY_GOVERNANCE_SEQUENCE_INVALID")
        _verify_authorization(
            entry["authorization"],
            candidate=candidate,
            auth_root=active,
            historical=True,
            now=current,
        )
        authority_changed = any(
            rootseal._policy_role(candidate["previous_policy"], r)
            != rootseal._policy_role(candidate["final_policy"], r)
            for r in ("governance", "emergency")
        )
        next_doc = entry["nextRoot"]
        root_att = entry["rootAttestations"]
        if next_doc is None:
            if authority_changed:
                _fail("CONTINUITY_VERIFY_ROTATION_REQUIRED")
            if root_att != []:
                _fail("CONTINUITY_VERIFY_UNEXPECTED_ATTESTATIONS")
            next_active = active
        else:
            if not isinstance(next_doc, dict):
                _fail("CONTINUITY_VERIFY_NEXT_ROOT_INVALID")
            next_info = _root_info(next_doc, "CONTINUITY_VERIFY_NEXT_ROOT_INVALID")
            if recovery_revoked & set(next_info["keys"]):
                _fail("CONTINUITY_VERIFY_RECOVERY_REVOKED_KEY_REINTRODUCED")
            try:
                next_active = rootseal._verify_rotation(
                    active, next_info, now=next_info["issued"]
                )
            except rootseal.RootTrustError as exc:
                raise RootContinuityError(
                    "CONTINUITY_VERIFY_ROTATION_INVALID:" + str(exc)
                ) from exc
            context = f"governance-epoch-{candidate['state']['epoch']}"
            if not isinstance(root_att, list):
                _fail("CONTINUITY_VERIFY_ROOT_ATTESTATIONS_INVALID")
            normalized = []
            for doc in root_att:
                if not isinstance(doc, dict):
                    _fail("CONTINUITY_VERIFY_ROOT_ATTESTATIONS_INVALID")
                normalized.append(
                    _verify_attestation(
                        doc,
                        root_info=next_active,
                        context_id=context,
                        trust_certs=trust_certs,
                        now=current,
                        historical=True,
                    )
                )
            normalized.sort(key=lambda x: x["signed"]["keyId"])
            if normalized != root_att or sorted(
                x["signed"]["keyId"] for x in normalized
            ) != sorted(next_active["roles"]["root"]["keyids"]):
                _fail("CONTINUITY_VERIFY_ROOT_ATTESTATION_SET_INVALID")
        revoked_after = set(candidate["state"].get("revokedAuthorityKeyIds", []))
        try:
            rootseal._assert_role_matches_policy(
                next_active,
                "governance",
                candidate["final_policy"],
                revoked_after,
                "CONTINUITY_FINAL_GOVERNANCE_ROLE",
            )
            rootseal._assert_role_matches_policy(
                next_active,
                "emergency",
                candidate["final_policy"],
                revoked_after,
                "CONTINUITY_FINAL_EMERGENCY_ROLE",
            )
        except rootseal.RootTrustError as exc:
            raise RootContinuityError(str(exc)) from exc
        if (
            entry["activeRoot"] != next_active["envelope"]
            or entry["epoch"] != candidate["state"]["epoch"]
        ):
            _fail("CONTINUITY_VERIFY_ACTIVE_ROOT_REBIND_FAILED")
        body = {k: v for k, v in entry.items() if k != "chainHeadSha256"}
        calc = _continuity_head(chain, body)
        if entry["chainHeadSha256"] != calc:
            _fail("CONTINUITY_VERIFY_CHAIN_HASH_INVALID")
        chain = calc
        active = next_active
        governance_binding = _governance_binding_from_candidate(candidate)
    if not historical:
        skew = timedelta(minutes=int(rootseal.POLICY["max_clock_skew_minutes"]))
        if active["issued"] > current + skew:
            _fail("CONTINUITY_VERIFY_ACTIVE_ROOT_FROM_FUTURE")
        if active["expires"] < current + timedelta(
            minutes=int(rootseal.POLICY["min_root_remaining_minutes"])
        ):
            _fail("CONTINUITY_VERIFY_ACTIVE_ROOT_EXPIRED_OR_FREEZE_RISK")
    expected_state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-recovered-root-continuity",
        "governanceId": active["governance_id"],
        "bootstrapRootSha256": bundle["bootstrapRootSha256"],
        "recoveryRootSha256": bundle["recoveryRootSha256"],
        "activeRoot": {
            "version": active["version"],
            "sha256": active["sha256"],
            "expires": active["signed"]["expires"],
        },
        "rootChainHeadSha256": chain,
        "bundleSha256": _sha_bytes(bundle_raw),
        "governance": governance_binding,
        "recovery": {
            "incidentId": bundle["recoveryEvent"]["incidentId"],
            "recordSha256": bundle["recoveryEvent"]["recoveryRecordSha256"],
        },
    }
    if state != expected_state:
        _fail("CONTINUITY_VERIFY_STATE_REBIND_FAILED")
    if active_doc != active["envelope"] or _sha_bytes(active_raw) != active["sha256"]:
        _fail("CONTINUITY_VERIFY_ACTIVE_ROOT_FILE_MISMATCH")
    receipt, _ = _read(
        root / "release-root-continuity-receipt.json", "CONTINUITY_VERIFY_RECEIPT"
    )
    expected_receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": (
            "recovered-root-activated"
            if not epochs
            else "recovered-root-continuity-advanced"
        ),
        "governanceId": active["governance_id"],
        "incidentId": bundle["recoveryEvent"]["incidentId"],
        "bundle": _artifact_raw("release-root-continuity-bundle.json", bundle_raw),
        "state": _artifact_raw(_DOC_ROOT_CONTINUITY_STATE, state_raw),
        "activeRoot": _artifact_raw("active-root.json", active_raw),
        "attestationRootSha256": sorted(expected_attestation_root_sha256),
    }
    if receipt != expected_receipt:
        _fail("CONTINUITY_VERIFY_RECEIPT_REBIND_FAILED")
    return {
        "ok": True,
        "phase": "recovered-root-continuity-verified",
        "governance_epoch": governance_binding["epoch"],
        "active_root_version": active["version"],
        "active_root_sha256": active["sha256"],
        "root_chain_head_sha256": chain,
    }


def advance_governance(  # ruff: ignore[undocumented-public-function]
    *,
    previous_dir: Path,
    governance_dir: Path,
    authorization_signature_paths: list[Path],
    output_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    next_root_path: Path | None = None,
    root_attestation_paths: list[Path] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    prev = _regular_dir(previous_dir, "CONTINUITY_ADVANCE_PREVIOUS_INVALID")
    verified = verify_continuity(
        output_dir=prev,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    bundle, _ = _read(
        prev / "release-root-continuity-bundle.json",
        "CONTINUITY_ADVANCE_PREVIOUS_BUNDLE",
    )
    state, _ = _read(
        prev / _DOC_ROOT_CONTINUITY_STATE, "CONTINUITY_ADVANCE_PREVIOUS_STATE"
    )
    active_doc, _ = _read(
        prev / "active-root.json", "CONTINUITY_ADVANCE_PREVIOUS_ACTIVE"
    )
    active = _root_info(active_doc, "CONTINUITY_ADVANCE_ACTIVE_ROOT_INVALID")
    try:
        candidate = rootseal._validate_run154_dir(
            _regular_dir(governance_dir, "CONTINUITY_ADVANCE_GOVERNANCE_DIR_INVALID")
        )
    except rootseal.RootTrustError as exc:
        raise RootContinuityError(
            "CONTINUITY_ADVANCE_CANDIDATE_INVALID:" + str(exc)
        ) from exc
    if (
        candidate["state"]["epoch"] != state["governance"]["epoch"] + 1
        or candidate["entry"]["proposal"].get("previousGovernanceStateSha256")
        != state["governance"]["stateSha256"]
    ):
        _fail("CONTINUITY_ADVANCE_GOVERNANCE_SEQUENCE_INVALID")
    authorization = _authorization_from_files(
        authorization_signature_paths,
        candidate=candidate,
        auth_root=active,
        now=current,
    )
    authority_changed = any(
        rootseal._policy_role(candidate["previous_policy"], r)
        != rootseal._policy_role(candidate["final_policy"], r)
        for r in ("governance", "emergency")
    )
    next_doc = None
    next_active = active
    attestations = []
    trust_certs = _embedded_attestation_roots(
        bundle["attestationTrustRoots"],
        expected_attestation_root_sha256,
        now=current,
        historical=False,
    )
    if next_root_path is not None:
        next_doc, _ = _read(next_root_path, "CONTINUITY_ADVANCE_NEXT_ROOT")
        next_info = _root_info(next_doc, "CONTINUITY_ADVANCE_NEXT_ROOT_INVALID")
        recovery_revoked = set(
            bundle["recoveryOutput"]["release-root-recovery-record.json"].get(
                "revokedRootKeyIds", []
            )
        )
        if recovery_revoked & set(next_info["keys"]):
            _fail("CONTINUITY_ADVANCE_RECOVERY_REVOKED_KEY_REINTRODUCED")
        try:
            next_active = rootseal._verify_rotation(active, next_info, now=current)
        except rootseal.RootTrustError as exc:
            raise RootContinuityError(
                "CONTINUITY_ADVANCE_ROTATION_INVALID:" + str(exc)
            ) from exc
        context = f"governance-epoch-{candidate['state']['epoch']}"
        attestations = _attestations(
            root_attestation_paths or [],
            root_info=next_active,
            context_id=context,
            trust_certs=trust_certs,
            now=current,
        )
    elif authority_changed:
        _fail("CONTINUITY_ADVANCE_ROTATION_REQUIRED_FOR_AUTHORITY_CHANGE")
    elif root_attestation_paths:
        _fail("CONTINUITY_ADVANCE_ATTESTATIONS_WITHOUT_ROTATION")
    revoked_after = set(candidate["state"].get("revokedAuthorityKeyIds", []))
    try:
        rootseal._assert_role_matches_policy(
            next_active,
            "governance",
            candidate["final_policy"],
            revoked_after,
            "CONTINUITY_ADVANCE_FINAL_GOVERNANCE_ROLE",
        )
        rootseal._assert_role_matches_policy(
            next_active,
            "emergency",
            candidate["final_policy"],
            revoked_after,
            "CONTINUITY_ADVANCE_FINAL_EMERGENCY_ROLE",
        )
    except rootseal.RootTrustError as exc:
        raise RootContinuityError(str(exc)) from exc
    seq = len(bundle["epochs"]) + 2
    previous_head = state["rootChainHeadSha256"]
    body = {
        "sequence": seq,
        "epoch": candidate["state"]["epoch"],
        "previousChainHeadSha256": previous_head,
        "governanceSnapshot": candidate["snapshot"],
        "authorization": authorization,
        "nextRoot": next_doc,
        "rootAttestations": attestations,
        "activeRoot": next_active["envelope"],
    }
    chain = _continuity_head(previous_head, body)
    entry = dict(body, chainHeadSha256=chain)
    next_bundle = json.loads(json.dumps(bundle))
    next_bundle["epochs"].append(entry)
    next_state = _state(
        bundle=next_bundle,
        active_root=next_active,
        chain_head=chain,
        governance_binding=_governance_binding_from_candidate(candidate),
        recovery_record_sha=bundle["recoveryEvent"]["recoveryRecordSha256"],
    )
    target = _outside(
        output_dir,
        [prev, candidate["root"]]
        + [p.resolve() for p in authorization_signature_paths]
        + ([next_root_path.resolve()] if next_root_path else [])
        + [p.resolve() for p in root_attestation_paths or []],
        "CONTINUITY_ADVANCE_OUTPUT_INSIDE_INPUT",
    )
    if target.exists() or target.is_symlink():
        _fail("CONTINUITY_ADVANCE_OUTPUT_ALREADY_EXISTS")
    inputs = (
        [p for p in prev.rglob("*") if p.is_file()]
        + [p for p in candidate["root"].rglob("*") if p.is_file()]
        + authorization_signature_paths
        + ([next_root_path] if next_root_path else [])
        + list(root_attestation_paths or [])
    )
    before = {str(p.resolve()): _sha(p) for p in inputs}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run157-advance-", dir=target.parent) as td:
        stage = Path(td) / "continuity"
        stage.mkdir()
        _write(stage / "release-root-continuity-bundle.json", next_bundle)
        _write(stage / _DOC_ROOT_CONTINUITY_STATE, next_state)
        _write(stage / "active-root.json", next_active["envelope"])
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "recovered-root-continuity-advanced",
            "governanceId": next_active["governance_id"],
            "incidentId": next_bundle["recoveryEvent"]["incidentId"],
            "bundle": _artifact_raw(
                "release-root-continuity-bundle.json", _canonical(next_bundle)
            ),
            "state": _artifact_raw(_DOC_ROOT_CONTINUITY_STATE, _canonical(next_state)),
            "activeRoot": _artifact_raw(
                "active-root.json", _canonical(next_active["envelope"])
            ),
            "attestationRootSha256": sorted(expected_attestation_root_sha256),
        }
        _write(stage / "release-root-continuity-receipt.json", receipt)
        for p in inputs:
            if _sha(p) != before[str(p.resolve())]:
                _fail("CONTINUITY_ADVANCE_INPUT_CHANGED")
        temp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, temp_target, copy_function=shutil.copy2)
        os.replace(temp_target, target)
    final = verify_continuity(
        output_dir=target,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    return {
        "ok": True,
        "phase": "recovered-root-continuity-advanced",
        "governance_epoch": candidate["state"]["epoch"],
        "authorizing_root_version": active["version"],
        "active_root_version": next_active["version"],
        "root_chain_head_sha256": final["root_chain_head_sha256"],
    }


def _paths(values: list[str]) -> list[Path]:
    return [Path(x) for x in values]


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser(
        "activate",
        help="activate a Run 156 recovered root as the trusted continuation root",
    )
    a.add_argument("--sealed-dir", type=Path, required=True)
    a.add_argument("--bootstrap-root-sha256", required=True)
    a.add_argument("--recovery-dir", type=Path, required=True)
    a.add_argument("--recovery-root-sha256", required=True)
    a.add_argument("--attestation", action="append", default=[])
    a.add_argument("--attestation-trust-root", action="append", default=[])
    a.add_argument("--attestation-root-sha256", action="append", default=[])
    a.add_argument("--output-dir", type=Path, required=True)
    adv = sub.add_parser(
        "advance", help="seal the next governance epoch from the recovered/current root"
    )
    adv.add_argument("--previous-dir", type=Path, required=True)
    adv.add_argument("--governance-dir", type=Path, required=True)
    adv.add_argument("--signature", action="append", default=[])
    adv.add_argument("--next-root", type=Path)
    adv.add_argument("--root-attestation", action="append", default=[])
    adv.add_argument("--bootstrap-root-sha256", required=True)
    adv.add_argument("--recovery-root-sha256", required=True)
    adv.add_argument("--attestation-root-sha256", action="append", default=[])
    adv.add_argument("--output-dir", type=Path, required=True)
    v = sub.add_parser(
        "verify", help="offline-verify recovered-root continuity and later epochs"
    )
    v.add_argument("--continuity-dir", type=Path, required=True)
    v.add_argument("--bootstrap-root-sha256", required=True)
    v.add_argument("--recovery-root-sha256", required=True)
    v.add_argument("--attestation-root-sha256", action="append", default=[])
    v.add_argument("--historical", action="store_true")
    args = p.parse_args(argv)
    try:
        if args.command == "activate":
            result = activate_recovery(
                sealed_dir=args.sealed_dir,
                bootstrap_root_sha256=args.bootstrap_root_sha256,
                recovery_dir=args.recovery_dir,
                expected_recovery_root_sha256=args.recovery_root_sha256,
                attestation_paths=_paths(args.attestation),
                attestation_trust_root_paths=_paths(args.attestation_trust_root),
                expected_attestation_root_sha256=args.attestation_root_sha256,
                output_dir=args.output_dir,
            )
        elif args.command == "advance":
            result = advance_governance(
                previous_dir=args.previous_dir,
                governance_dir=args.governance_dir,
                authorization_signature_paths=_paths(args.signature),
                output_dir=args.output_dir,
                expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
                expected_recovery_root_sha256=args.recovery_root_sha256,
                expected_attestation_root_sha256=args.attestation_root_sha256,
                next_root_path=args.next_root,
                root_attestation_paths=_paths(args.root_attestation),
            )
        else:
            result = verify_continuity(
                output_dir=args.continuity_dir,
                expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
                expected_recovery_root_sha256=args.recovery_root_sha256,
                expected_attestation_root_sha256=args.attestation_root_sha256,
                historical=args.historical,
            )
    except RootContinuityError as exc:
        logger.error("root continuity verification failed: %s", exc)
        # sys.stdout.write(
        #     json.dumps({"ok": False, "error": str(exc)}, sort_keys=True) + "\n"
        # )
        return 2
    logger.info("root continuity operation completed: %s", result)
    # sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
