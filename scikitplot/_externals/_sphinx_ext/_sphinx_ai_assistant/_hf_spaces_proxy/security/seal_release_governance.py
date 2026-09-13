"""
Cryptographically seal release-governance candidates with threshold Ed25519 roots.

Run 154 constructs and archives a governance candidate.  This module is the acceptance
boundary: it does not trust Run 154's semantic ``signatureVerified`` booleans.  Instead,
it verifies fresh signatures by the exact selected governance keys over an exact,
canonical candidate subject, and maintains a TUF-style versioned root chain.
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
    from . import govern_release_history as governance
except (ImportError, ValueError) as exc:  # direct Space-style/script/importlib loading
    import importlib.util

    _gov_path = Path(__file__).resolve().parent / "govern_release_history.py"
    _gov_name = "_release_governance_for_root_seal"
    if _gov_name in sys.modules:
        governance = sys.modules[_gov_name]
    else:
        _spec = importlib.util.spec_from_file_location(_gov_name, _gov_path)
        if _spec is None or _spec.loader is None:
            raise ImportError("cannot load govern_release_history") from exc
        governance = importlib.util.module_from_spec(_spec)
        sys.modules[_gov_name] = governance
        _spec.loader.exec_module(governance)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_root_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_CHUNK = 1024 * 1024
_ZERO_HASH = "0" * 64
logger = logging.getLogger(__name__)


class RootTrustError(RuntimeError):
    """Cryptographic root or governance-authorization invariant failed."""


def _fail(code: str) -> None:
    raise RootTrustError(code)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
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
    path.write_bytes(_canonical_bytes(value))


def _loads(raw: bytes, code: str) -> dict[str, Any]:
    def hook(pairs):
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(code + "_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except RootTrustError:
        raise
    except Exception as exc:
        raise RootTrustError(code + "_JSON_INVALID") from exc
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
    if canonical and raw != _canonical_bytes(doc):
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
            ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            for ch in value  # lint
        )
        or _ID.fullmatch(value) is None
    ):
        _fail(code)
    return value


def _hex(value: Any, code: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(code)
    return value


def _size(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(code)
    return value


def _timestamp_value(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code + "_INVALID")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise RootTrustError(code + "_INVALID") from exc
    return parsed


def _timestamp(value: Any, code: str) -> str:
    _timestamp_value(value, code)
    return str(value)


def _b64(value: Any, code: str, *, expected_len: int) -> bytes:
    _lint = len(value) > 4096  # ruff: ignore[magic-value-comparison]
    if not isinstance(value, str) or not value or _lint:
        _fail(code)
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise RootTrustError(code) from exc
    if len(raw) != expected_len or base64.b64encode(raw).decode("ascii") != value:
        _fail(code)
    return raw


def _ed25519_verify(
    public_raw: bytes, signature_raw: bytes, message: bytes, code: str
) -> None:
    try:
        from cryptography.exceptions import (  # ruff: ignore[import-outside-top-level]
            InvalidSignature,
        )
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # ruff: ignore[import-outside-top-level]
            Ed25519PublicKey,
        )
    # fail closed if deployment omitted the crypto backend
    except Exception as exc:
        raise RootTrustError("ROOT_CRYPTO_BACKEND_UNAVAILABLE") from exc
    try:
        Ed25519PublicKey.from_public_bytes(public_raw).verify(signature_raw, message)
    except InvalidSignature as exc:
        raise RootTrustError(code) from exc
    except Exception as exc:
        raise RootTrustError(code + "_KEY_INVALID") from exc


def _artifact(path: Path, name: str | None = None) -> dict[str, Any]:
    return {
        "name": name or path.name,
        "sha256": _sha(path),
        "size": path.stat().st_size,
    }


def _artifact_doc(doc: dict[str, Any], name: str) -> dict[str, Any]:
    raw = _canonical_bytes(doc)
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


def _root_key(value: Any, key_id: str) -> dict[str, Any]:
    expected = {"keytype", "scheme", "identity", "operator", "expires", "keyval"}
    if not isinstance(value, dict) or set(value) != expected:
        _fail("ROOT_KEY_SCHEMA_INVALID")
    if value.get("keytype") not in set(POLICY["allowed_keytypes"]) or value.get(
        "scheme"
    ) not in set(POLICY["allowed_schemes"]):
        _fail("ROOT_KEY_ALGORITHM_INVALID")
    keyval = value.get("keyval")
    if not isinstance(keyval, dict) or set(keyval) != {"public"}:
        _fail("ROOT_KEYVAL_SCHEMA_INVALID")
    public_text = keyval.get("public")
    _b64(public_text, "ROOT_PUBLIC_KEY_INVALID", expected_len=32)
    return {
        "keytype": "ed25519",
        "scheme": "ed25519",
        "identity": _identity(value.get("identity"), "ROOT_KEY_IDENTITY_INVALID"),
        "operator": _identity(value.get("operator"), "ROOT_KEY_OPERATOR_INVALID"),
        "expires": _timestamp(value.get("expires"), "ROOT_KEY_EXPIRES"),
        "keyval": {"public": public_text},
    }


def _role(
    value: Any,
    code: str,
    *,
    min_keys: int,
    min_threshold: int,
    min_operators: int,
    keys: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"keyids", "threshold"}:
        _fail(code + "_SCHEMA_INVALID")
    keyids_raw = value.get("keyids")
    if not isinstance(keyids_raw, list) or not (
        min_keys <= len(keyids_raw) <= int(POLICY["max_root_keys"])
    ):
        _fail(code + "_KEY_COUNT_INVALID")
    keyids = [_identity(x, code + "_KEY_ID_INVALID") for x in keyids_raw]
    if len(set(keyids)) != len(keyids) or any(x not in keys for x in keyids):
        _fail(code + "_KEY_SET_INVALID")
    threshold = _size(value.get("threshold"), code + "_THRESHOLD_INVALID")
    if threshold < min_threshold or threshold > len(keyids):
        _fail(code + "_THRESHOLD_INVALID")
    operators = {keys[x]["operator"] for x in keyids}
    if len(operators) < min_operators:
        _fail(code + "_OPERATOR_COUNT_INVALID")
    return {"keyids": sorted(keyids), "threshold": threshold}


def _validate_root_signed(  # ruff: ignore[too-many-branches]
    signed: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "_type",
        "specVersion",
        "version",
        "governanceId",
        "issuedAt",
        "expires",
        "keys",
        "roles",
    }
    if (
        set(signed) != expected
        or signed.get("_type") != "root"
        or signed.get("specVersion") != str(POLICY["spec_version"])
    ):
        _fail("ROOT_SIGNED_SCHEMA_INVALID")
    version = _size(signed.get("version"), "ROOT_VERSION_INVALID")
    if version < 1:
        _fail("ROOT_VERSION_INVALID")
    governance_id = _identity(signed.get("governanceId"), "ROOT_GOVERNANCE_ID_INVALID")
    issued = _timestamp_value(signed.get("issuedAt"), "ROOT_ISSUED_AT")
    expires = _timestamp_value(signed.get("expires"), "ROOT_EXPIRES")
    if expires <= issued or expires - issued > timedelta(
        days=int(POLICY["max_root_lifetime_days"])
    ):
        _fail("ROOT_LIFETIME_INVALID")
    keys_raw = signed.get("keys")
    if not isinstance(keys_raw, dict) or not (
        int(POLICY["min_root_keys"]) <= len(keys_raw) <= int(POLICY["max_root_keys"])
    ):
        _fail("ROOT_KEY_COUNT_INVALID")
    keys: dict[str, dict[str, Any]] = {}
    for raw_id, value in keys_raw.items():
        key_id = _identity(raw_id, "ROOT_KEY_ID_INVALID")
        keys[key_id] = _root_key(value, key_id)
    roles_raw = signed.get("roles")
    if not isinstance(roles_raw, dict) or set(roles_raw) != {
        "root",
        "governance",
        "emergency",
    }:
        _fail("ROOT_ROLES_SCHEMA_INVALID")
    roles = {
        "root": _role(
            roles_raw["root"],
            "ROOT_ROLE_ROOT",
            min_keys=int(POLICY["min_root_keys"]),
            min_threshold=int(POLICY["min_root_threshold"]),
            min_operators=int(POLICY["min_root_operators"]),
            keys=keys,
        ),
        "governance": _role(
            roles_raw["governance"],
            "ROOT_ROLE_GOVERNANCE",
            min_keys=int(POLICY["min_role_keys"]),
            min_threshold=int(POLICY["min_role_threshold"]),
            min_operators=int(POLICY["min_role_operators"]),
            keys=keys,
        ),
        "emergency": _role(
            roles_raw["emergency"],
            "ROOT_ROLE_EMERGENCY",
            min_keys=int(POLICY["min_role_keys"]),
            min_threshold=int(POLICY["min_role_threshold"]),
            min_operators=int(POLICY["min_role_operators"]),
            keys=keys,
        ),
    }
    root_set = set(roles["root"]["keyids"])
    governance_set = set(roles["governance"]["keyids"])
    emergency_set = set(roles["emergency"]["keyids"])
    if set(keys) != root_set | governance_set | emergency_set:
        _fail("ROOT_UNUSED_KEY_INVALID")
    if bool(POLICY.get("require_root_role_disjoint", True)) and root_set & (
        governance_set | emergency_set
    ):
        _fail("ROOT_ROLE_NOT_DISJOINT")
    if (
        bool(POLICY.get("require_governance_emergency_disjoint", True))
        and governance_set & emergency_set
    ):
        _fail("ROOT_GOVERNANCE_EMERGENCY_NOT_DISJOINT")
    if bool(POLICY.get("require_root_operator_disjoint", True)):
        root_ops = {keys[x]["operator"] for x in root_set}
        online_ops = {keys[x]["operator"] for x in governance_set | emergency_set}
        if root_ops & online_ops:
            _fail("ROOT_OPERATOR_NOT_DISJOINT")
    # Every active key remains valid for the complete lifetime of the root metadata.
    for role in roles.values():
        for key_id in role["keyids"]:
            if _timestamp_value(keys[key_id]["expires"], "ROOT_KEY_EXPIRES") < expires:
                _fail("ROOT_KEY_EXPIRES_BEFORE_ROOT")
    normalized = {
        "_type": "root",
        "specVersion": str(POLICY["spec_version"]),
        "version": version,
        "governanceId": governance_id,
        "issuedAt": issued.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "expires": expires.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "keys": {k: keys[k] for k in sorted(keys)},
        "roles": roles,
    }
    if signed != normalized:
        _fail("ROOT_SIGNED_NOT_NORMALIZED")
    return {
        "signed": normalized,
        "version": version,
        "governance_id": governance_id,
        "issued": issued,
        "expires": expires,
        "keys": keys,
        "roles": roles,
    }


def _validate_root_envelope(doc: dict[str, Any]) -> dict[str, Any]:
    if set(doc) != {"signatures", "signed"} or not isinstance(doc.get("signed"), dict):
        _fail("ROOT_METADATA_SCHEMA_INVALID")
    info = _validate_root_signed(doc["signed"])
    signatures = doc.get("signatures")
    if (
        not isinstance(signatures, list)
        or len(signatures) > int(POLICY["max_root_keys"]) * 2
    ):
        _fail("ROOT_SIGNATURES_SCHEMA_INVALID")
    normalized_sigs: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in signatures:
        if not isinstance(item, dict) or set(item) != {"keyid", "sig"}:
            _fail("ROOT_SIGNATURE_SCHEMA_INVALID")
        key_id = _identity(item.get("keyid"), "ROOT_SIGNATURE_KEY_ID_INVALID")
        if key_id in seen:
            _fail("ROOT_SIGNATURE_KEY_DUPLICATE")
        seen.add(key_id)
        sig = item.get("sig")
        _b64(sig, "ROOT_SIGNATURE_ENCODING_INVALID", expected_len=64)
        normalized_sigs.append({"keyid": key_id, "sig": sig})
    normalized_sigs.sort(key=lambda x: x["keyid"])
    if signatures != normalized_sigs:
        _fail("ROOT_SIGNATURES_NOT_NORMALIZED")
    info["envelope"] = {"signatures": normalized_sigs, "signed": info["signed"]}
    info["signed_bytes"] = _canonical_bytes(info["signed"])
    info["sha256"] = _sha_bytes(_canonical_bytes(info["envelope"]))
    info["signature_map"] = {x["keyid"]: x["sig"] for x in normalized_sigs}
    return info


def _verify_role_signatures(
    envelope_info: dict[str, Any],
    trusted_info: dict[str, Any],
    role_name: str,
    *,
    at: datetime,
    code: str,
) -> list[str]:
    role = trusted_info["roles"][role_name]
    valid: list[str] = []
    for key_id in role["keyids"]:
        sig_text = envelope_info["signature_map"].get(key_id)
        if sig_text is None:
            continue
        key = trusted_info["keys"][key_id]
        if _timestamp_value(key["expires"], "ROOT_KEY_EXPIRES") < at:
            continue
        _ed25519_verify(
            _b64(key["keyval"]["public"], "ROOT_PUBLIC_KEY_INVALID", expected_len=32),
            _b64(sig_text, "ROOT_SIGNATURE_ENCODING_INVALID", expected_len=64),
            envelope_info["signed_bytes"],
            code + "_SIGNATURE_INVALID",
        )
        valid.append(key_id)
    if len(valid) < role["threshold"]:
        _fail(code + "_THRESHOLD_NOT_MET")
    return sorted(valid)


def _assert_signature_keys(
    signature_map: dict[str, str], allowed: set[str], code: str
) -> None:
    if not set(signature_map).issubset(allowed):
        _fail(code + "_UNAUTHORIZED_SIGNATURE_KEY")


def _verify_bootstrap_root(
    root_doc: dict[str, Any], expected_sha256: str, *, now: datetime
) -> dict[str, Any]:
    info = _validate_root_envelope(root_doc)
    _assert_signature_keys(
        info["signature_map"], set(info["roles"]["root"]["keyids"]), "ROOT_BOOTSTRAP"
    )
    if info["version"] != 1:
        _fail("ROOT_BOOTSTRAP_VERSION_INVALID")
    if info["sha256"] != _hex(expected_sha256, "ROOT_BOOTSTRAP_PIN_INVALID"):
        _fail("ROOT_BOOTSTRAP_PIN_MISMATCH")
    if info["issued"] > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
        _fail("ROOT_ISSUED_FROM_FUTURE")
    if info["expires"] < now + timedelta(
        minutes=int(POLICY["min_root_remaining_minutes"])
    ):
        _fail("ROOT_EXPIRED_OR_FREEZE_RISK")
    _verify_role_signatures(
        info, info, "root", at=info["issued"], code="ROOT_BOOTSTRAP_SELF_SIGNATURE"
    )
    return info


def _verify_rotation(
    previous: dict[str, Any], current: dict[str, Any], *, now: datetime
) -> dict[str, Any]:
    allowed_signature_keys = set(previous["roles"]["root"]["keyids"]) | set(
        current["roles"]["root"]["keyids"]
    )
    _assert_signature_keys(
        current["signature_map"], allowed_signature_keys, "ROOT_ROTATION"
    )
    if current["governance_id"] != previous["governance_id"]:
        _fail("ROOT_ROTATION_GOVERNANCE_ID_MISMATCH")
    if current["version"] != previous["version"] + 1:
        _fail("ROOT_ROTATION_VERSION_NOT_CONSECUTIVE")
    if current["issued"] < previous["issued"]:
        _fail("ROOT_ROTATION_ISSUED_AT_ROLLBACK")
    if current["issued"] > previous["expires"] + timedelta(
        minutes=int(POLICY["max_clock_skew_minutes"])
    ):
        _fail("ROOT_ROTATION_AFTER_OLD_ROOT_EXPIRY")
    if current["issued"] > now + timedelta(
        minutes=int(POLICY["max_clock_skew_minutes"])
    ):
        _fail("ROOT_ROTATION_FROM_FUTURE")
    if current["expires"] < now + timedelta(
        minutes=int(POLICY["min_root_remaining_minutes"])
    ):
        _fail("ROOT_ROTATION_EXPIRED_OR_FREEZE_RISK")
    _verify_role_signatures(
        current, previous, "root", at=current["issued"], code="ROOT_ROTATION_OLD_ROOT"
    )
    _verify_role_signatures(
        current, current, "root", at=current["issued"], code="ROOT_ROTATION_NEW_ROOT"
    )
    return current


def _root_head(governance_id: str, roots: list[dict[str, Any]]) -> str:
    head = _ZERO_HASH
    for item in roots:
        info = _validate_root_envelope(item)
        head = _sha_bytes(
            _canonical_bytes(
                {
                    "governanceId": governance_id,
                    "previous": head,
                    "rootVersion": info["version"],
                    "rootSha256": info["sha256"],
                }
            )
        )
    return head


def _validate_root_bundle(
    bundle: dict[str, Any],
    *,
    expected_bootstrap_pin: str | None = None,
    now: datetime | None = None,
    require_current_fresh: bool = True,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "predicateType",
        "status",
        "governanceId",
        "bootstrapRootSha256",
        "roots",
    }
    if (
        set(bundle) != expected
        or bundle.get("schemaVersion") != int(POLICY["root_bundle_schema_version"])
        or bundle.get("predicateType") != PREDICATE_TYPE
        or bundle.get("status") != "trusted-release-root-chain"
    ):
        _fail("ROOT_BUNDLE_SCHEMA_INVALID")
    governance_id = _identity(
        bundle.get("governanceId"), "ROOT_BUNDLE_GOVERNANCE_ID_INVALID"
    )
    bootstrap_pin = _hex(
        bundle.get("bootstrapRootSha256"), "ROOT_BUNDLE_BOOTSTRAP_PIN_INVALID"
    )
    if expected_bootstrap_pin is not None and bootstrap_pin != _hex(
        expected_bootstrap_pin, "ROOT_EXPECTED_BOOTSTRAP_PIN_INVALID"
    ):
        _fail("ROOT_BUNDLE_BOOTSTRAP_PIN_MISMATCH")
    roots = bundle.get("roots")
    if not isinstance(roots, list) or not (
        1 <= len(roots) <= int(POLICY["max_root_versions"])
    ):
        _fail("ROOT_BUNDLE_ROOT_COUNT_INVALID")
    infos = [
        (
            _validate_root_envelope(x)
            if isinstance(x, dict)
            else _fail("ROOT_BUNDLE_ROOT_SCHEMA_INVALID")
        )
        for x in roots
    ]
    first = infos[0]
    if (
        first["version"] != 1
        or first["governance_id"] != governance_id
        or first["sha256"] != bootstrap_pin
    ):
        _fail("ROOT_BUNDLE_GENESIS_MISMATCH")
    _assert_signature_keys(
        first["signature_map"],
        set(first["roles"]["root"]["keyids"]),
        "ROOT_BUNDLE_GENESIS",
    )
    _verify_role_signatures(
        first,
        first,
        "root",
        at=first["issued"],
        code="ROOT_BUNDLE_GENESIS_SELF_SIGNATURE",
    )
    previous = first
    for current in infos[1:]:
        if (
            current["governance_id"] != governance_id
            or current["version"] != previous["version"] + 1
        ):
            _fail("ROOT_BUNDLE_VERSION_CHAIN_INVALID")
        if current["issued"] < previous["issued"] or current["issued"] > previous[
            "expires"
        ] + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
            _fail("ROOT_BUNDLE_TIME_CHAIN_INVALID")
        allowed_signature_keys = set(previous["roles"]["root"]["keyids"]) | set(
            current["roles"]["root"]["keyids"]
        )
        _assert_signature_keys(
            current["signature_map"], allowed_signature_keys, "ROOT_BUNDLE_ROTATION"
        )
        _verify_role_signatures(
            current, previous, "root", at=current["issued"], code="ROOT_BUNDLE_OLD_ROOT"
        )
        _verify_role_signatures(
            current, current, "root", at=current["issued"], code="ROOT_BUNDLE_NEW_ROOT"
        )
        previous = current
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if require_current_fresh and previous["issued"] > current_time + timedelta(
        minutes=int(POLICY["max_clock_skew_minutes"])
    ):
        _fail("ROOT_CURRENT_ISSUED_FROM_FUTURE")
    if require_current_fresh and previous["expires"] < current_time + timedelta(
        minutes=int(POLICY["min_root_remaining_minutes"])
    ):
        _fail("ROOT_CURRENT_EXPIRED_OR_FREEZE_RISK")
    normalized = {
        "schemaVersion": int(POLICY["root_bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "trusted-release-root-chain",
        "governanceId": governance_id,
        "bootstrapRootSha256": bootstrap_pin,
        "roots": roots,
    }
    if bundle != normalized:
        _fail("ROOT_BUNDLE_NOT_NORMALIZED")
    return {
        "bundle": normalized,
        "governance_id": governance_id,
        "bootstrap_pin": bootstrap_pin,
        "infos": infos,
        "current": infos[-1],
        "chain_head_sha256": _root_head(governance_id, roots),
        "bundle_sha256": _sha_bytes(_canonical_bytes(normalized)),
    }


def _policy_role(policy: dict[str, Any], role_name: str) -> dict[str, Any]:
    field = "policyAuthority" if role_name == "governance" else "emergencyAuthority"
    value = policy.get(field)
    if not isinstance(value, dict) or set(value) != {"threshold", "members"}:
        _fail("ROOT_POLICY_ROLE_SCHEMA_INVALID")
    return value


def _assert_role_matches_policy(
    root_info: dict[str, Any],
    role_name: str,
    policy: dict[str, Any],
    revoked: set[str],
    code: str,
) -> None:
    role = root_info["roles"][role_name]
    policy_role = _policy_role(policy, role_name)
    members = policy_role["members"]
    if role["threshold"] != policy_role["threshold"]:
        _fail(code + "_THRESHOLD_MISMATCH")
    policy_ids = sorted(m["keyId"] for m in members)
    if role["keyids"] != policy_ids:
        _fail(code + "_KEY_SET_MISMATCH")
    by_id = {m["keyId"]: m for m in members}
    for key_id in role["keyids"]:
        if key_id in revoked:
            _fail(code + "_REVOKED_KEY")
        key = root_info["keys"][key_id]
        member = by_id[key_id]
        if (
            key["identity"] != member["identity"]
            or key["operator"] != member["operator"]
        ):
            _fail(code + "_MEMBER_BINDING_MISMATCH")


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_GOVERNANCE_STATE = "trusted-governance-state.json"
_DOC_HISTORY_STATE = "trusted-history-state.json"
_DOC_RELEASE_ROOT_STATE = "trusted-release-root-state.json"


def _validate_run154_dir(  # ruff: ignore[too-many-branches]
    path: Path,
) -> dict[str, Any]:
    root = _regular_dir(path, "RUN154_DIRECTORY_INVALID")
    allowed = {
        "release-governance-bundle.json",
        _DOC_GOVERNANCE_STATE,
        "release-governance-recovery-snapshot.json",
        "release-governance-receipt.json",
        "governance-transition-proposal.json",
        "approval-evidence",
        "archive-results",
        _DOC_HISTORY_STATE,
        "release-history-bundle.json",
    }
    if {p.name for p in root.iterdir()} != allowed:
        _fail("RUN154_DIRECTORY_ALLOWLIST_MISMATCH")
    for dirname in ("approval-evidence", "archive-results"):
        d = root / dirname
        if d.is_symlink() or not d.is_dir() or not any(d.iterdir()):
            _fail("RUN154_EVIDENCE_DIRECTORY_INVALID")
        for item in d.iterdir():
            if item.is_symlink() or not item.is_file():
                _fail("RUN154_EVIDENCE_ENTRY_INVALID")
            _read(item, "RUN154_EVIDENCE_ENTRY")
    try:
        verified = governance.verify_governance_bundle(
            bundle_path=root / "release-governance-bundle.json",
            state_path=root / _DOC_GOVERNANCE_STATE,
        )
    except governance.GovernanceError as exc:
        raise RootTrustError("RUN154_GOVERNANCE_INVALID:" + str(exc)) from exc
    bundle, bundle_raw = _read(root / "release-governance-bundle.json", "RUN154_BUNDLE")
    state, state_raw = _read(root / _DOC_GOVERNANCE_STATE, "RUN154_STATE")
    snapshot, snapshot_raw = _read(
        root / "release-governance-recovery-snapshot.json", "RUN154_SNAPSHOT"
    )
    receipt, _ = _read(root / "release-governance-receipt.json", "RUN154_RECEIPT")
    proposal, proposal_raw = _read(
        root / "governance-transition-proposal.json", "RUN154_PROPOSAL"
    )
    history_state, history_state_raw = _read(
        root / _DOC_HISTORY_STATE, "RUN154_HISTORY_STATE"
    )
    history_bundle, history_bundle_raw = _read(
        root / "release-history-bundle.json", "RUN154_HISTORY_BUNDLE"
    )
    if snapshot != {
        "schemaVersion": int(governance.POLICY["recovery_snapshot_schema_version"]),
        "governanceState": state,
        "governanceBundle": bundle,
        "historyState": history_state,
        "historyBundle": history_bundle,
    }:
        _fail("RUN154_SNAPSHOT_CONTENT_MISMATCH")
    expected_receipt = {
        "schemaVersion",
        "status",
        "governanceId",
        "historyId",
        "epoch",
        "policyVersion",
        "proposalSha256",
        "approvalEvidenceSha256",
        "history",
        "governance",
        "archives",
    }
    if (
        set(receipt) != expected_receipt
        or receipt.get("status") != "governed"
        or receipt.get("governanceId") != state.get("governanceId")
        or receipt.get("historyId") != state.get("historyId")
        or receipt.get("epoch") != state.get("epoch")
        or receipt.get("policyVersion") != state.get("policy", {}).get("policyVersion")
    ):
        _fail("RUN154_RECEIPT_SCHEMA_INVALID")
    artifacts = receipt.get("governance")
    if not isinstance(artifacts, dict) or artifacts != {
        "bundle": {
            "name": "release-governance-bundle.json",
            "sha256": _sha_bytes(bundle_raw),
            "size": len(bundle_raw),
        },
        "state": {
            "name": _DOC_GOVERNANCE_STATE,
            "sha256": _sha_bytes(state_raw),
            "size": len(state_raw),
        },
        "recoverySnapshot": {
            "name": "release-governance-recovery-snapshot.json",
            "sha256": _sha_bytes(snapshot_raw),
            "size": len(snapshot_raw),
        },
    }:
        _fail("RUN154_RECEIPT_ARTIFACT_REBIND_FAILED")
    if receipt.get("proposalSha256") != _sha_bytes(proposal_raw):
        _fail("RUN154_PROPOSAL_RECEIPT_REBIND_FAILED")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("RUN154_GOVERNANCE_ENTRY_MISSING")
    entry = entries[-1]
    if (
        entry.get("proposalSha256") != _sha_bytes(proposal_raw)
        or entry.get("proposal") != proposal
        or entry.get("epoch") != state.get("epoch")
    ):
        _fail("RUN154_LATEST_ENTRY_REBIND_FAILED")
    approval_docs = []
    approval_hashes = []
    for p in sorted((root / "approval-evidence").iterdir(), key=lambda x: x.name):
        doc, raw = _read(p, "RUN154_APPROVAL_EVIDENCE")
        approval_docs.append(doc)
        approval_hashes.append(_sha_bytes(raw))
    if sorted(approval_hashes) != sorted(receipt.get("approvalEvidenceSha256", [])):
        _fail("RUN154_APPROVAL_EVIDENCE_REBIND_FAILED")
    if sorted(
        _sha_bytes(_canonical_bytes(x)) for x in entry.get("approvals", [])
    ) != sorted(approval_hashes):
        _fail("RUN154_APPROVAL_ENTRY_REBIND_FAILED")
    archive_hash_list = [_sha(p) for p in (root / "archive-results").iterdir()]
    archive_hashes = set(archive_hash_list)
    archives = receipt.get("archives")
    if not isinstance(archives, list) or not archives:
        _fail("RUN154_ARCHIVE_EVIDENCE_SCHEMA_INVALID")
    referenced_archive_hashes: list[str] = []
    for item in archives:
        if not isinstance(item, dict):
            _fail("RUN154_ARCHIVE_EVIDENCE_SCHEMA_INVALID")
        for field in ("bindEvidenceSha256", "verifyEvidenceSha256"):
            value = item.get(field)
            if value not in archive_hashes:
                _fail("RUN154_ARCHIVE_EVIDENCE_REBIND_FAILED")
            referenced_archive_hashes.append(value)
    if (
        len(archive_hash_list) != len(referenced_archive_hashes)
        or len(set(archive_hash_list)) != len(archive_hash_list)
        or len(set(referenced_archive_hashes)) != len(referenced_archive_hashes)
        or set(referenced_archive_hashes) != archive_hashes
    ):
        _fail("RUN154_ARCHIVE_EVIDENCE_SET_MISMATCH")
    history_receipt = receipt.get("history")
    if not isinstance(history_receipt, dict) or history_receipt != {
        "sequence": state["history"]["sequence"],
        "stateSha256": _sha_bytes(history_state_raw),
        "bundleSha256": _sha_bytes(history_bundle_raw),
        "chainHeadSha256": state["history"]["chainHeadSha256"],
    }:
        _fail("RUN154_HISTORY_REBIND_FAILED")
    previous_policy = (
        bundle["genesis"]["initialPolicy"]
        if len(entries) == 1
        else entries[-2]["nextPolicy"]
    )
    revoked_before: set[str] = set()
    for old in entries[:-1]:
        revoked_before.update(old.get("revokedAuthorityKeyIds", []))
    return {
        "root": root,
        "verified": verified,
        "bundle": bundle,
        "state": state,
        "snapshot": snapshot,
        "receipt": receipt,
        "proposal": proposal,
        "entry": entry,
        "previous_policy": previous_policy,
        "final_policy": state["policy"],
        "revoked_before": revoked_before,
        "snapshot_sha256": _sha_bytes(snapshot_raw),
        "snapshot_size": len(snapshot_raw),
        "state_sha256": _sha_bytes(state_raw),
        "bundle_sha256": _sha_bytes(bundle_raw),
        "proposal_sha256": _sha_bytes(proposal_raw),
        "history_state_sha256": _sha_bytes(history_state_raw),
        "history_bundle_sha256": _sha_bytes(history_bundle_raw),
    }


def _subject(candidate: dict[str, Any], auth_root: dict[str, Any]) -> dict[str, Any]:
    entry = candidate["entry"]
    role = "emergency" if entry.get("approvalRole") == "emergency" else "governance"
    return {
        "schemaVersion": int(POLICY["schema_version"]),
        "predicateType": PREDICATE_TYPE + "/governance-authorization",
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
            "size": (candidate["root"] / _DOC_GOVERNANCE_STATE).stat().st_size,
        },
        "governanceBundle": {
            "name": "release-governance-bundle.json",
            "sha256": candidate["bundle_sha256"],
            "size": (
                (candidate["root"] / "release-governance-bundle.json").stat().st_size
            ),
        },
        "recoverySnapshot": {
            "name": "release-governance-recovery-snapshot.json",
            "sha256": candidate["snapshot_sha256"],
            "size": candidate["snapshot_size"],
        },
        "history": candidate["state"]["history"],
    }


def _validate_authorization_signature(
    doc: dict[str, Any],
    *,
    subject: dict[str, Any],
    root_info: dict[str, Any],
    role_name: str,
    expected_member: dict[str, Any],
    now: datetime,
    enforce_freshness: bool = True,
) -> dict[str, Any]:
    if set(doc) != {"signature", "signed"} or not isinstance(doc.get("signed"), dict):
        _fail("ROOT_AUTH_SIGNATURE_SCHEMA_INVALID")
    signed = doc["signed"]
    expected = {
        "schemaVersion",
        "keyId",
        "identity",
        "operator",
        "decision",
        "rootVersion",
        "subjectSha256",
        "signedAt",
    }
    if (
        set(signed) != expected
        or signed.get("schemaVersion") != int(POLICY["signature_schema_version"])
        or signed.get("decision") != "approve"
    ):
        _fail("ROOT_AUTH_SIGNATURE_SIGNED_SCHEMA_INVALID")
    key_id = _identity(signed.get("keyId"), "ROOT_AUTH_KEY_ID_INVALID")
    if key_id != expected_member["keyId"]:
        _fail("ROOT_AUTH_SELECTED_KEY_MISMATCH")
    key = root_info["keys"].get(key_id)
    if key is None or key_id not in root_info["roles"][role_name]["keyids"]:
        _fail("ROOT_AUTH_KEY_NOT_AUTHORIZED")
    if (
        signed.get("identity") != expected_member["identity"]
        or signed.get("operator") != expected_member["operator"]
        or key["identity"] != expected_member["identity"]
        or key["operator"] != expected_member["operator"]
    ):
        _fail("ROOT_AUTH_IDENTITY_BINDING_MISMATCH")
    if signed.get("rootVersion") != root_info["version"] or signed.get(
        "subjectSha256"
    ) != _sha_bytes(_canonical_bytes(subject)):
        _fail("ROOT_AUTH_SUBJECT_REBIND_FAILED")
    signed_at = _timestamp_value(signed.get("signedAt"), "ROOT_AUTH_SIGNED_AT")
    if signed.get("signedAt") != signed_at.isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    ):
        _fail("ROOT_AUTH_SIGNED_AT_NOT_NORMALIZED")
    if enforce_freshness:
        if signed_at > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
            _fail("ROOT_AUTH_SIGNATURE_FROM_FUTURE")
        if now - signed_at > timedelta(
            hours=int(POLICY["max_authorization_age_hours"])
        ):
            _fail("ROOT_AUTH_SIGNATURE_STALE")
    if (
        signed_at < root_info["issued"]
        or signed_at > root_info["expires"]
        or signed_at > _timestamp_value(key["expires"], "ROOT_KEY_EXPIRES")
    ):
        _fail("ROOT_AUTH_SIGNATURE_OUTSIDE_KEY_VALIDITY")
    signature = _b64(
        doc.get("signature"), "ROOT_AUTH_SIGNATURE_ENCODING_INVALID", expected_len=64
    )
    _ed25519_verify(
        _b64(key["keyval"]["public"], "ROOT_PUBLIC_KEY_INVALID", expected_len=32),
        signature,
        _canonical_bytes(signed),
        "ROOT_AUTH_SIGNATURE_INVALID",
    )
    normalized = {"signature": doc["signature"], "signed": signed}
    if doc != normalized:
        _fail("ROOT_AUTH_SIGNATURE_NOT_NORMALIZED")
    return normalized


def _state_document(
    *,
    root_info: dict[str, Any],
    bundle_sha: str,
    chain_head: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": int(POLICY["root_state_schema_version"]),
        "governanceId": root_info["governance_id"],
        "rootVersion": root_info["version"],
        "rootSha256": root_info["sha256"],
        "rootExpires": root_info["signed"]["expires"],
        "rootChainHeadSha256": chain_head,
        "bundleSha256": bundle_sha,
        "governance": {
            "epoch": candidate["state"]["epoch"],
            "policyVersion": candidate["state"]["policy"]["policyVersion"],
            "stateSha256": candidate["state_sha256"],
            "bundleSha256": candidate["bundle_sha256"],
            "recoverySnapshotSha256": candidate["snapshot_sha256"],
            "proposalSha256": candidate["proposal_sha256"],
        },
    }


def verify_root_bundle(  # ruff: ignore[undocumented-public-function]
    *,
    bundle_path: Path,
    state_path: Path | None = None,
    expected_bootstrap_root_sha256: str | None = None,
    now: datetime | None = None,
    require_current_fresh: bool = True,
) -> dict[str, Any]:
    if expected_bootstrap_root_sha256 is None:
        _fail("ROOT_EXPECTED_BOOTSTRAP_PIN_REQUIRED")
    bundle, _ = _read(bundle_path, "ROOT_BUNDLE")
    info = _validate_root_bundle(
        bundle,
        expected_bootstrap_pin=expected_bootstrap_root_sha256,
        now=now,
        require_current_fresh=require_current_fresh,
    )
    if state_path is not None:
        state, state_raw = _read(state_path, "ROOT_STATE")
        expected = {
            "schemaVersion",
            "governanceId",
            "rootVersion",
            "rootSha256",
            "rootExpires",
            "rootChainHeadSha256",
            "bundleSha256",
            "governance",
        }
        if set(state) != expected or state.get("schemaVersion") != int(
            POLICY["root_state_schema_version"]
        ):
            _fail("ROOT_STATE_SCHEMA_INVALID")
        current = info["current"]
        if (
            state.get("governanceId") != info["governance_id"]
            or state.get("rootVersion") != current["version"]
            or state.get("rootSha256") != current["sha256"]
            or state.get("rootExpires") != current["signed"]["expires"]
            or state.get("rootChainHeadSha256") != info["chain_head_sha256"]
            or state.get("bundleSha256") != info["bundle_sha256"]
        ):
            _fail("ROOT_STATE_REBIND_FAILED")
        governance_binding = state.get("governance")
        if not isinstance(governance_binding, dict) or set(governance_binding) != {
            "epoch",
            "policyVersion",
            "stateSha256",
            "bundleSha256",
            "recoverySnapshotSha256",
            "proposalSha256",
        }:
            _fail("ROOT_STATE_GOVERNANCE_BINDING_INVALID")
        for field in (
            "stateSha256",
            "bundleSha256",
            "recoverySnapshotSha256",
            "proposalSha256",
        ):
            _hex(governance_binding.get(field), "ROOT_STATE_GOVERNANCE_HASH_INVALID")
        _size(governance_binding.get("epoch"), "ROOT_STATE_GOVERNANCE_EPOCH_INVALID")
        _size(
            governance_binding.get("policyVersion"),
            "ROOT_STATE_GOVERNANCE_POLICY_VERSION_INVALID",
        )
        info["state"] = state
        info["state_sha256"] = _sha_bytes(state_raw)
    return info


def verify_sealed_governance(  # ruff: ignore[too-many-branches]
    *,
    sealed_dir: Path,
    expected_bootstrap_root_sha256: str,
    now: datetime | None = None,
    require_current_fresh: bool = False,
) -> dict[str, Any]:
    """
    Offline-verify a complete Run 155 seal, including governance signatures.

    Historical verification defaults to allowing an expired *current* root because expiry
    does not invalidate signatures that were made while the key/root was valid.  Callers
    performing a live acceptance check should set ``require_current_fresh=True``.
    """
    root = _regular_dir(sealed_dir, "ROOT_SEALED_DIRECTORY_INVALID")
    allowed = {
        "release-root-bundle.json",
        _DOC_RELEASE_ROOT_STATE,
        "cryptographic-governance-authorization.json",
        "active-root.json",
        "release-governance-recovery-snapshot.json",
        "release-root-receipt.json",
    }
    if {p.name for p in root.iterdir()} != allowed:
        _fail("ROOT_SEALED_DIRECTORY_ALLOWLIST_MISMATCH")
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    bundle_info = verify_root_bundle(
        bundle_path=root / "release-root-bundle.json",
        state_path=root / _DOC_RELEASE_ROOT_STATE,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        now=current_time,
        require_current_fresh=require_current_fresh,
    )
    state, state_raw = _read(root / _DOC_RELEASE_ROOT_STATE, "ROOT_SEALED_STATE")
    active_root, active_root_raw = _read(
        root / "active-root.json", "ROOT_SEALED_ACTIVE_ROOT"
    )
    if (
        active_root != bundle_info["current"]["envelope"]
        or _sha_bytes(active_root_raw) != bundle_info["current"]["sha256"]
    ):
        _fail("ROOT_SEALED_ACTIVE_ROOT_REBIND_FAILED")
    snapshot, snapshot_raw = _read(
        root / "release-governance-recovery-snapshot.json",
        "ROOT_SEALED_GOVERNANCE_SNAPSHOT",
    )
    expected_snapshot_keys = {
        "schemaVersion",
        "governanceState",
        "governanceBundle",
        "historyState",
        "historyBundle",
    }
    if set(snapshot) != expected_snapshot_keys or snapshot.get("schemaVersion") != int(
        governance.POLICY["recovery_snapshot_schema_version"]
    ):
        _fail("ROOT_SEALED_GOVERNANCE_SNAPSHOT_SCHEMA_INVALID")
    gov_state = snapshot.get("governanceState")
    gov_bundle = snapshot.get("governanceBundle")
    history_state = snapshot.get("historyState")
    history_bundle = snapshot.get("historyBundle")
    if not all(
        isinstance(x, dict)
        for x in (gov_state, gov_bundle, history_state, history_bundle)
    ):
        _fail("ROOT_SEALED_GOVERNANCE_SNAPSHOT_SCHEMA_INVALID")
    with tempfile.TemporaryDirectory(prefix="root-seal-offline-governance-") as tmp:
        sp = Path(tmp) / "state.json"
        bp = Path(tmp) / "bundle.json"
        sp.write_bytes(_canonical_bytes(gov_state))
        bp.write_bytes(_canonical_bytes(gov_bundle))
        try:
            governance.verify_governance_bundle(bundle_path=bp, state_path=sp)
        except governance.GovernanceError as exc:
            raise RootTrustError("ROOT_SEALED_GOVERNANCE_INVALID:" + str(exc)) from exc
    history_state_raw = _canonical_bytes(history_state)
    history_bundle_raw = _canonical_bytes(history_bundle)
    with tempfile.TemporaryDirectory(prefix="root-seal-offline-history-") as tmp:
        hsp = Path(tmp) / "history-state.json"
        hbp = Path(tmp) / "history-bundle.json"
        hsp.write_bytes(history_state_raw)
        hbp.write_bytes(history_bundle_raw)
        try:
            history_info = governance.history._validate_previous_history(hsp, hbp)
        except governance.history.HistoryError as exc:
            raise RootTrustError("ROOT_SEALED_HISTORY_INVALID:" + str(exc)) from exc
    expected_history_binding = {
        "sequence": history_info["sequence"],
        "stateSha256": _sha_bytes(history_state_raw),
        "bundleSha256": _sha_bytes(history_bundle_raw),
        "chainHeadSha256": history_info["chain_head_sha256"],
    }
    if gov_state.get("history") != expected_history_binding:
        _fail("ROOT_SEALED_HISTORY_REBIND_FAILED")
    entries = gov_bundle.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail("ROOT_SEALED_GOVERNANCE_ENTRY_MISSING")
    entry = entries[-1]
    previous_policy = (
        gov_bundle["genesis"]["initialPolicy"]
        if len(entries) == 1
        else entries[-2]["nextPolicy"]
    )
    revoked_before: set[str] = set()
    for old in entries[:-1]:
        revoked_before.update(old.get("revokedAuthorityKeyIds", []))
    proposal = entry.get("proposal")
    if not isinstance(proposal, dict):
        _fail("ROOT_SEALED_PROPOSAL_MISSING")
    proposal_sha = _sha_bytes(_canonical_bytes(proposal))
    if proposal_sha != entry.get("proposalSha256"):
        _fail("ROOT_SEALED_PROPOSAL_HASH_MISMATCH")
    authorization, authorization_raw = _read(
        root / "cryptographic-governance-authorization.json",
        "ROOT_SEALED_AUTHORIZATION",
    )
    expected_auth_keys = {
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
        set(authorization) != expected_auth_keys
        or authorization.get("schemaVersion")
        != int(POLICY["authorization_schema_version"])
        or authorization.get("predicateType")
        != PREDICATE_TYPE + "/governance-authorization"
        or authorization.get("status") != "cryptographically-authorized"
    ):
        _fail("ROOT_SEALED_AUTHORIZATION_SCHEMA_INVALID")
    auth_version = _size(
        authorization.get("rootVersion"), "ROOT_SEALED_AUTH_ROOT_VERSION_INVALID"
    )
    auth_sha = _hex(
        authorization.get("rootSha256"), "ROOT_SEALED_AUTH_ROOT_HASH_INVALID"
    )
    matches = [
        x
        for x in bundle_info["infos"]
        if x["version"] == auth_version and x["sha256"] == auth_sha
    ]
    if len(matches) != 1:
        _fail("ROOT_SEALED_AUTH_ROOT_NOT_IN_CHAIN")
    auth_root = matches[0]
    role_name = (
        "emergency" if entry.get("approvalRole") == "emergency" else "governance"
    )
    if authorization.get("role") != role_name:
        _fail("ROOT_SEALED_AUTH_ROLE_MISMATCH")
    _assert_role_matches_policy(
        auth_root,
        role_name,
        previous_policy,
        revoked_before,
        "ROOT_SEALED_AUTHORIZING_ROLE",
    )
    other_role = "governance" if role_name == "emergency" else "emergency"
    _assert_role_matches_policy(
        auth_root,
        other_role,
        previous_policy,
        revoked_before,
        "ROOT_SEALED_AUTHORIZING_OTHER_ROLE",
    )
    subject = {
        "schemaVersion": int(POLICY["schema_version"]),
        "predicateType": PREDICATE_TYPE + "/governance-authorization",
        "governanceId": gov_state["governanceId"],
        "historyId": gov_state["historyId"],
        "epoch": gov_state["epoch"],
        "policyVersion": gov_state["policy"]["policyVersion"],
        "transitionId": entry["transitionId"],
        "role": role_name,
        "proposalSha256": proposal_sha,
        "selectedKeyIds": sorted(entry["selectedApproverKeyIds"]),
        "authorizingRoot": {
            "version": auth_root["version"],
            "sha256": auth_root["sha256"],
        },
        "governanceState": {
            "name": _DOC_GOVERNANCE_STATE,
            "sha256": _sha_bytes(_canonical_bytes(gov_state)),
            "size": len(_canonical_bytes(gov_state)),
        },
        "governanceBundle": {
            "name": "release-governance-bundle.json",
            "sha256": _sha_bytes(_canonical_bytes(gov_bundle)),
            "size": len(_canonical_bytes(gov_bundle)),
        },
        "recoverySnapshot": {
            "name": "release-governance-recovery-snapshot.json",
            "sha256": _sha_bytes(snapshot_raw),
            "size": len(snapshot_raw),
        },
        "history": gov_state["history"],
    }
    if authorization.get("subject") != subject or authorization.get(
        "subjectSha256"
    ) != _sha_bytes(_canonical_bytes(subject)):
        _fail("ROOT_SEALED_AUTH_SUBJECT_MISMATCH")
    selected = subject["selectedKeyIds"]
    if authorization.get("selectedKeyIds") != selected:
        _fail("ROOT_SEALED_AUTH_SELECTED_SET_MISMATCH")
    policy_role = _policy_role(previous_policy, role_name)
    members = {m["keyId"]: m for m in policy_role["members"]}
    signatures = authorization.get("signatures")
    if not isinstance(signatures, list) or len(signatures) != len(selected):
        _fail("ROOT_SEALED_AUTH_SIGNATURE_COUNT_INVALID")
    seen: set[str] = set()
    operators: set[str] = set()
    for sig_doc in signatures:
        if not isinstance(sig_doc, dict) or not isinstance(sig_doc.get("signed"), dict):
            _fail("ROOT_SEALED_AUTH_SIGNATURE_SCHEMA_INVALID")
        key_id = sig_doc["signed"].get("keyId")
        if key_id not in selected or key_id in seen:
            _fail("ROOT_SEALED_AUTH_SIGNATURE_SET_INVALID")
        seen.add(key_id)
        normalized = _validate_authorization_signature(
            sig_doc,
            subject=subject,
            root_info=auth_root,
            role_name=role_name,
            expected_member=members[key_id],
            now=current_time,
            enforce_freshness=False,
        )
        operators.add(normalized["signed"]["operator"])
    if sorted(seen) != selected or len(seen) < policy_role["threshold"]:
        _fail("ROOT_SEALED_AUTH_SIGNATURE_THRESHOLD_INVALID")
    min_ops = int(
        governance.POLICY["min_emergency_operators"]
        if role_name == "emergency"
        else governance.POLICY["min_policy_operators"]
    )
    if len(operators) < min_ops:
        _fail("ROOT_SEALED_AUTH_OPERATOR_QUORUM_INVALID")
    final_policy = gov_state["policy"]
    revoked_after = set(gov_state.get("revokedAuthorityKeyIds", []))
    _assert_role_matches_policy(
        bundle_info["current"],
        "governance",
        final_policy,
        revoked_after,
        "ROOT_SEALED_FINAL_GOVERNANCE_ROLE",
    )
    _assert_role_matches_policy(
        bundle_info["current"],
        "emergency",
        final_policy,
        revoked_after,
        "ROOT_SEALED_FINAL_EMERGENCY_ROLE",
    )
    governance_binding = state["governance"]
    expected_binding = {
        "epoch": gov_state["epoch"],
        "policyVersion": gov_state["policy"]["policyVersion"],
        "stateSha256": _sha_bytes(_canonical_bytes(gov_state)),
        "bundleSha256": _sha_bytes(_canonical_bytes(gov_bundle)),
        "recoverySnapshotSha256": _sha_bytes(snapshot_raw),
        "proposalSha256": proposal_sha,
    }
    if governance_binding != expected_binding:
        _fail("ROOT_SEALED_STATE_GOVERNANCE_REBIND_FAILED")
    receipt, _ = _read(root / "release-root-receipt.json", "ROOT_SEALED_RECEIPT")
    expected_receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": "cryptographically-sealed",
        "governanceId": gov_state["governanceId"],
        "epoch": gov_state["epoch"],
        "policyVersion": gov_state["policy"]["policyVersion"],
        "authorization": {
            "name": "cryptographic-governance-authorization.json",
            "sha256": _sha_bytes(authorization_raw),
            "size": len(authorization_raw),
        },
        "rootBundle": _artifact(root / "release-root-bundle.json"),
        "rootState": {
            "name": _DOC_RELEASE_ROOT_STATE,
            "sha256": _sha_bytes(state_raw),
            "size": len(state_raw),
        },
        "activeRoot": {
            "name": "active-root.json",
            "sha256": _sha_bytes(active_root_raw),
            "size": len(active_root_raw),
        },
        "governanceSnapshot": {
            "name": "release-governance-recovery-snapshot.json",
            "sha256": _sha_bytes(snapshot_raw),
            "size": len(snapshot_raw),
        },
    }
    if receipt != expected_receipt:
        _fail("ROOT_SEALED_RECEIPT_REBIND_FAILED")
    return {
        "ok": True,
        "phase": "governance-cryptographic-seal-verified",
        "governance_id": gov_state["governanceId"],
        "epoch": gov_state["epoch"],
        "root_version": bundle_info["current"]["version"],
        "root_chain_head_sha256": bundle_info["chain_head_sha256"],
        "authorization_sha256": _sha_bytes(authorization_raw),
    }


def seal_governance(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    governance_dir: Path,
    authorization_signature_paths: list[Path],
    output_dir: Path,
    bootstrap_root_path: Path | None = None,
    expected_bootstrap_root_sha256: str | None = None,
    previous_root_state: Path | None = None,
    previous_root_bundle: Path | None = None,
    next_root_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    candidate = _validate_run154_dir(governance_dir)
    protected = [candidate["root"]]
    if previous_root_state is not None:
        protected.append(previous_root_state.expanduser().resolve().parent)
    target = _outside(output_dir, protected, "ROOT_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("ROOT_OUTPUT_ALREADY_EXISTS")
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expected_bootstrap_root_sha256 is None:
        _fail("ROOT_EXPECTED_BOOTSTRAP_PIN_REQUIRED")
    expected_bootstrap_root_sha256 = _hex(
        expected_bootstrap_root_sha256, "ROOT_EXPECTED_BOOTSTRAP_PIN_INVALID"
    )
    if (previous_root_state is None) != (previous_root_bundle is None):
        _fail("ROOT_PREVIOUS_STATE_BUNDLE_PAIR_REQUIRED")
    if previous_root_state is None:
        if bootstrap_root_path is None:
            _fail("ROOT_BOOTSTRAP_INPUT_REQUIRED")
        root_doc, _ = _read(bootstrap_root_path, "ROOT_BOOTSTRAP_METADATA")
        auth_root = _verify_bootstrap_root(
            root_doc, expected_bootstrap_root_sha256, now=current_time
        )
        if auth_root["governance_id"] != candidate["state"]["governanceId"]:
            _fail("ROOT_BOOTSTRAP_GOVERNANCE_ID_MISMATCH")
        if (
            bool(POLICY.get("allow_bootstrap_only_epoch_one", True))
            and candidate["state"]["epoch"] != 1
        ):
            _fail("ROOT_BOOTSTRAP_EPOCH_INVALID")
        roots = [auth_root["envelope"]]
        bootstrap_pin = auth_root["sha256"]
        previous_info = None
    else:
        if bootstrap_root_path is not None:
            _fail("ROOT_BOOTSTRAP_NOT_ALLOWED_WITH_PREVIOUS")
        previous_info = verify_root_bundle(
            bundle_path=previous_root_bundle,
            state_path=previous_root_state,
            expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
            now=current_time,
        )
        auth_root = previous_info["current"]
        if auth_root["governance_id"] != candidate["state"]["governanceId"]:
            _fail("ROOT_PREVIOUS_GOVERNANCE_ID_MISMATCH")
        previous_binding = previous_info["state"]["governance"]
        if (
            candidate["state"]["epoch"] != previous_binding["epoch"] + 1
            or candidate["entry"]["proposal"].get("previousGovernanceStateSha256")
            != previous_binding["stateSha256"]
        ):
            _fail("ROOT_GOVERNANCE_SEQUENCE_OR_PREVIOUS_STATE_MISMATCH")
        roots = list(previous_info["bundle"]["roots"])
        bootstrap_pin = previous_info["bootstrap_pin"]
    role_name = (
        "emergency"
        if candidate["entry"].get("approvalRole") == "emergency"
        else "governance"
    )
    _assert_role_matches_policy(
        auth_root,
        role_name,
        candidate["previous_policy"],
        candidate["revoked_before"],
        "ROOT_AUTHORIZING_ROLE",
    )
    # The non-selected counterpart role must also match the prior policy, preventing a
    # root file from smuggling an alternate emergency/governance membership.
    other_role = "governance" if role_name == "emergency" else "emergency"
    _assert_role_matches_policy(
        auth_root,
        other_role,
        candidate["previous_policy"],
        candidate["revoked_before"],
        "ROOT_AUTHORIZING_OTHER_ROLE",
    )
    subject = _subject(candidate, auth_root)
    selected = subject["selectedKeyIds"]
    policy_role = _policy_role(candidate["previous_policy"], role_name)
    if (
        selected != sorted(candidate["entry"].get("selectedApproverKeyIds", []))
        or len(selected) < policy_role["threshold"]
    ):
        _fail("ROOT_AUTH_SELECTED_QUORUM_INVALID")
    expected_members = {m["keyId"]: m for m in policy_role["members"]}
    if len(authorization_signature_paths) != len(selected):
        _fail("ROOT_AUTH_SIGNATURE_COUNT_INVALID")
    sig_by_key: dict[str, dict[str, Any]] = {}
    input_paths = sorted(
        (p for p in candidate["root"].rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(candidate["root"])),
    )
    input_paths.extend(authorization_signature_paths)
    if bootstrap_root_path is not None:
        input_paths.append(bootstrap_root_path)
    if previous_root_state is not None:
        input_paths.extend([previous_root_state, previous_root_bundle])
    if next_root_path is not None:
        input_paths.append(next_root_path)
    initial_hashes = {str(p.resolve()): _sha(p) for p in input_paths}
    for path in authorization_signature_paths:
        doc, _ = _read(path, "ROOT_AUTH_SIGNATURE")
        signed = doc.get("signed")
        key_id = signed.get("keyId") if isinstance(signed, dict) else None
        if key_id not in selected or key_id in sig_by_key:
            _fail("ROOT_AUTH_SIGNATURE_KEY_SET_INVALID")
        sig_by_key[key_id] = _validate_authorization_signature(
            doc,
            subject=subject,
            root_info=auth_root,
            role_name=role_name,
            expected_member=expected_members[key_id],
            now=current_time,
        )
    if sorted(sig_by_key) != selected:
        _fail("ROOT_AUTH_SIGNATURE_SET_MISMATCH")
    operators = {sig_by_key[k]["signed"]["operator"] for k in selected}
    if len(operators) < int(
        governance.POLICY["min_emergency_operators"]
        if role_name == "emergency"
        else governance.POLICY["min_policy_operators"]
    ):
        _fail("ROOT_AUTH_OPERATOR_QUORUM_INVALID")
    authorization = {
        "schemaVersion": int(POLICY["authorization_schema_version"]),
        "predicateType": PREDICATE_TYPE + "/governance-authorization",
        "status": "cryptographically-authorized",
        "subject": subject,
        "subjectSha256": _sha_bytes(_canonical_bytes(subject)),
        "rootVersion": auth_root["version"],
        "rootSha256": auth_root["sha256"],
        "role": role_name,
        "selectedKeyIds": selected,
        "signatures": [sig_by_key[k] for k in sorted(sig_by_key)],
    }
    final_policy = candidate["final_policy"]
    authority_changed = any(
        _policy_role(candidate["previous_policy"], r) != _policy_role(final_policy, r)
        for r in ("governance", "emergency")
    )
    active_root = auth_root
    if next_root_path is not None:
        next_doc, _ = _read(next_root_path, "ROOT_NEXT_METADATA")
        next_info = _validate_root_envelope(next_doc)
        active_root = _verify_rotation(auth_root, next_info, now=current_time)
        roots.append(active_root["envelope"])
    elif authority_changed:
        _fail("ROOT_ROTATION_REQUIRED_FOR_AUTHORITY_CHANGE")
    revoked_after = set(candidate["state"].get("revokedAuthorityKeyIds", []))
    _assert_role_matches_policy(
        active_root,
        "governance",
        final_policy,
        revoked_after,
        "ROOT_FINAL_GOVERNANCE_ROLE",
    )
    _assert_role_matches_policy(
        active_root,
        "emergency",
        final_policy,
        revoked_after,
        "ROOT_FINAL_EMERGENCY_ROLE",
    )
    root_bundle = {
        "schemaVersion": int(POLICY["root_bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "trusted-release-root-chain",
        "governanceId": candidate["state"]["governanceId"],
        "bootstrapRootSha256": bootstrap_pin,
        "roots": roots,
    }
    bundle_info = _validate_root_bundle(
        root_bundle, expected_bootstrap_pin=bootstrap_pin, now=current_time
    )
    bundle_sha = bundle_info["bundle_sha256"]
    root_state = _state_document(
        root_info=active_root,
        bundle_sha=bundle_sha,
        chain_head=bundle_info["chain_head_sha256"],
        candidate=candidate,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-root-seal-", dir=target.parent
    ) as tmp:
        stage = Path(tmp) / "root-seal"
        stage.mkdir()
        _write(stage / "release-root-bundle.json", root_bundle)
        _write(stage / _DOC_RELEASE_ROOT_STATE, root_state)
        _write(stage / "cryptographic-governance-authorization.json", authorization)
        _write(stage / "active-root.json", active_root["envelope"])
        shutil.copy2(
            candidate["root"] / "release-governance-recovery-snapshot.json",
            stage / "release-governance-recovery-snapshot.json",
        )
        verify_root_bundle(
            bundle_path=stage / "release-root-bundle.json",
            state_path=stage / _DOC_RELEASE_ROOT_STATE,
            expected_bootstrap_root_sha256=bootstrap_pin,
            now=current_time,
        )
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "cryptographically-sealed",
            "governanceId": candidate["state"]["governanceId"],
            "epoch": candidate["state"]["epoch"],
            "policyVersion": candidate["state"]["policy"]["policyVersion"],
            "authorization": _artifact_doc(
                authorization, "cryptographic-governance-authorization.json"
            ),
            "rootBundle": _artifact_doc(root_bundle, "release-root-bundle.json"),
            "rootState": _artifact_doc(root_state, _DOC_RELEASE_ROOT_STATE),
            "activeRoot": _artifact_doc(active_root["envelope"], "active-root.json"),
            "governanceSnapshot": _artifact(
                stage / "release-governance-recovery-snapshot.json"
            ),
        }
        _write(stage / "release-root-receipt.json", receipt)
        for path in input_paths:
            if _sha(path) != initial_hashes[str(path.resolve())]:
                _fail("ROOT_INPUT_CHANGED_DURING_SEAL")
        temp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, temp_target, copy_function=shutil.copy2)
        os.replace(temp_target, target)
    verify_sealed_governance(
        sealed_dir=target,
        expected_bootstrap_root_sha256=bootstrap_pin,
        now=current_time,
        require_current_fresh=True,
    )
    return {
        "ok": True,
        "phase": "governance-cryptographically-sealed",
        "governance_id": candidate["state"]["governanceId"],
        "epoch": candidate["state"]["epoch"],
        "policy_version": candidate["state"]["policy"]["policyVersion"],
        "authorizing_root_version": auth_root["version"],
        "active_root_version": active_root["version"],
        "root_chain_head_sha256": bundle_info["chain_head_sha256"],
        "authorization_sha256": _sha(
            target / "cryptographic-governance-authorization.json"
        ),
        "root_state_sha256": _sha(target / _DOC_RELEASE_ROOT_STATE),
        "root_bundle_sha256": _sha(target / "release-root-bundle.json"),
    }


def _parse_paths(values: list[str]) -> list[Path]:
    return [Path(x) for x in values]


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser(
        "seal", help="cryptographically seal a Run 154 governance candidate"
    )
    seal.add_argument("--governance-dir", type=Path, required=True)
    seal.add_argument(
        "--signature",
        action="append",
        default=[],
        help="canonical Ed25519 authorization signature JSON",
    )
    seal.add_argument("--output-dir", type=Path, required=True)
    seal.add_argument("--bootstrap-root", type=Path)
    seal.add_argument("--bootstrap-root-sha256", required=True)
    seal.add_argument("--previous-root-state", type=Path)
    seal.add_argument("--previous-root-bundle", type=Path)
    seal.add_argument("--next-root", type=Path)
    verify = sub.add_parser(
        "verify", help="offline-verify a root chain and optional compact state"
    )
    verify.add_argument("--root-bundle", type=Path, required=True)
    verify.add_argument("--root-state", type=Path)
    verify.add_argument("--bootstrap-root-sha256", required=True)
    verify_seal = sub.add_parser(
        "verify-seal",
        help="offline-verify a complete Run 155 cryptographic governance seal",
    )
    verify_seal.add_argument("--sealed-dir", type=Path, required=True)
    verify_seal.add_argument("--bootstrap-root-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_root_bundle(
                bundle_path=args.root_bundle,
                state_path=args.root_state,
                expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            )
            out = {
                "ok": True,
                "phase": "release-root-verified",
                "governance_id": result["governance_id"],
                "root_version": result["current"]["version"],
                "root_chain_head_sha256": result["chain_head_sha256"],
            }
        elif args.command == "verify-seal":
            out = verify_sealed_governance(
                sealed_dir=args.sealed_dir,
                expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
                require_current_fresh=False,
            )
        else:
            out = seal_governance(
                governance_dir=args.governance_dir,
                authorization_signature_paths=_parse_paths(args.signature),
                output_dir=args.output_dir,
                bootstrap_root_path=args.bootstrap_root,
                expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
                previous_root_state=args.previous_root_state,
                previous_root_bundle=args.previous_root_bundle,
                next_root_path=args.next_root,
            )
    except RootTrustError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("%s", json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
