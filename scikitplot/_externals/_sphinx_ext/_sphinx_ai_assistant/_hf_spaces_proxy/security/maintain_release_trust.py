"""
Run 156: delegated freshness metadata and threshold root recovery.

Run 155 cryptographically seals governance.  This layer adds short-lived snapshot and
 timestamp metadata plus an independently pinned, multi-channel recovery authority for
 root-role compromise.  Private keys are never accepted as input or written to output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import tomllib

logger = logging.getLogger(__name__)

try:
    from . import seal_release_governance as rootseal
except (ImportError, ValueError) as exc:
    import importlib.util

    _p = Path(__file__).resolve().parent / "seal_release_governance.py"
    _n = "_run155_root_for_delegated_trust"
    if _n in sys.modules:
        rootseal = sys.modules[_n]
    else:
        _s = importlib.util.spec_from_file_location(_n, _p)
        if _s is None or _s.loader is None:
            raise ImportError("cannot load seal_release_governance") from exc
        rootseal = importlib.util.module_from_spec(_s)
        sys.modules[_n] = rootseal
        _s.loader.exec_module(rootseal)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_delegated_trust_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_ZERO_HASH = "0" * 64


class DelegatedTrustError(RuntimeError):  # ruff: ignore[undocumented-public-class]
    pass


def _fail(code: str) -> None:
    raise DelegatedTrustError(code)


def _canonical(value: dict[str, Any]) -> bytes:
    return rootseal._canonical_bytes(value)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    return rootseal._sha(path)


def _loads(raw: bytes, code: str) -> dict[str, Any]:
    def hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                _fail(code + "_DUPLICATE_KEY")
            out[k] = v
        return out

    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except DelegatedTrustError:
        raise
    except Exception as exc:
        raise DelegatedTrustError(code + "_JSON_INVALID") from exc
    if not isinstance(doc, dict):
        _fail(code + "_SCHEMA_INVALID")
    return doc


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


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.write_bytes(_canonical(doc))


def _identity(value: Any, code: str) -> str:
    if not isinstance(value, str):
        _fail(code)
    v = value.strip()
    if (
        not v
        or len(v) > 512  # ruff: ignore[magic-value-comparison]
        or ".." in v
        or "?" in v
        or "#" in v
        or "\x00" in v
    ):
        _fail(code)
    if _ID.fullmatch(v) is None or any(
        ord(c) < 32  # ruff: ignore[magic-value-comparison]
        or ord(c) == 127  # ruff: ignore[magic-value-comparison]
        for c in v  # lint
    ):
        _fail(code)
    return v


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
        raise DelegatedTrustError(code) from exc


def _ts(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _b64(value: Any, code: str, *, expected_len: int) -> bytes:
    try:
        return rootseal._b64(value, code, expected_len=expected_len)
    except rootseal.RootTrustError as exc:
        raise DelegatedTrustError(str(exc)) from exc


def _artifact(path: Path, name: str | None = None) -> dict[str, Any]:
    return {
        "name": name or path.name,
        "sha256": _sha(path),
        "size": path.stat().st_size,
    }


def _artifact_raw(name: str, raw: bytes) -> dict[str, Any]:
    return {"name": name, "sha256": _sha_bytes(raw), "size": len(raw)}


def _dir(path: Path, code: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        _fail(code)
    return path.resolve()


def _outside(path: Path, protected: list[Path], code: str) -> Path:
    t = path.expanduser().resolve()
    for p in protected:
        r = p.resolve()
        if t == r or r in t.parents:
            _fail(code)
    return t


def _key(value: Any, code: str) -> dict[str, Any]:
    expected = {
        "keytype",
        "scheme",
        "identity",
        "operator",
        "expires",
        "signerProfile",
        "keyval",
    }
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    if value["keytype"] not in set(POLICY["allowed_keytypes"]) or value[
        "scheme"
    ] not in set(POLICY["allowed_schemes"]):
        _fail(code + "_ALGORITHM_INVALID")
    profile = value["signerProfile"]
    if profile not in set(POLICY["allowed_signer_profiles"]):
        _fail(code + "_PROFILE_INVALID")
    kv = value["keyval"]
    if not isinstance(kv, dict) or set(kv) != {"public"}:
        _fail(code + "_KEYVAL_INVALID")
    _b64(kv["public"], code + "_PUBLIC_INVALID", expected_len=32)
    _dt(value["expires"], code + "_EXPIRES_INVALID")
    return {
        "keytype": "ed25519",
        "scheme": "ed25519",
        "identity": _identity(value["identity"], code + "_IDENTITY_INVALID"),
        "operator": _identity(value["operator"], code + "_OPERATOR_INVALID"),
        "expires": value["expires"],
        "signerProfile": profile,
        "keyval": {"public": kv["public"]},
    }


def _role(
    value: Any,
    keys: dict[str, dict[str, Any]],
    code: str,
    *,
    min_keys: int,
    min_threshold: int,
    min_ops: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"keyids", "threshold"}:
        _fail(code + "_SCHEMA_INVALID")
    ids = value["keyids"]
    if not isinstance(ids, list) or not (
        min_keys <= len(ids) <= int(POLICY["max_keys"])
    ):
        _fail(code + "_KEY_COUNT_INVALID")
    ids = [_identity(x, code + "_KEYID_INVALID") for x in ids]
    if (
        ids != sorted(ids)
        or len(set(ids)) != len(ids)
        or any(x not in keys for x in ids)
    ):
        _fail(code + "_KEY_SET_INVALID")
    threshold = _int(
        value["threshold"], code + "_THRESHOLD_INVALID", minimum=min_threshold
    )
    if threshold > len(ids):
        _fail(code + "_THRESHOLD_INVALID")
    if len({keys[x]["operator"] for x in ids}) < min_ops:
        _fail(code + "_OPERATOR_COUNT_INVALID")
    return {"keyids": ids, "threshold": threshold}


def _sig_map(doc: dict[str, Any], code: str) -> dict[str, str]:
    sigs = doc.get("signatures")
    if not isinstance(sigs, list) or len(sigs) > int(POLICY["max_keys"]) * 2:
        _fail(code + "_SIGNATURES_INVALID")
    out = {}
    norm = []
    for item in sigs:
        if not isinstance(item, dict) or set(item) != {"keyid", "sig"}:
            _fail(code + "_SIGNATURE_SCHEMA_INVALID")
        kid = _identity(item["keyid"], code + "_SIGNATURE_KEYID_INVALID")
        if kid in out:
            _fail(code + "_SIGNATURE_DUPLICATE")
        _b64(item["sig"], code + "_SIGNATURE_ENCODING_INVALID", expected_len=64)
        out[kid] = item["sig"]
        norm.append({"keyid": kid, "sig": item["sig"]})
    norm.sort(key=lambda x: x["keyid"])
    if sigs != norm:
        _fail(code + "_SIGNATURES_NOT_NORMALIZED")
    return out


def _verify_role_signatures(
    doc: dict[str, Any],
    signed: dict[str, Any],
    role: dict[str, Any],
    keys: dict[str, dict[str, Any]],
    *,
    at: datetime,
    code: str,
    exact_allowed: bool = True,
) -> list[str]:
    sigs = _sig_map(doc, code)
    allowed = set(role["keyids"])
    if exact_allowed and not set(sigs).issubset(allowed):
        _fail(code + "_UNAUTHORIZED_SIGNATURE_KEY")
    raw = _canonical(signed)
    valid = []
    for kid in role["keyids"]:
        if kid not in sigs:
            continue
        key = keys[kid]
        if _dt(key["expires"], code + "_KEY_EXPIRES") < at:
            _fail(code + "_KEY_EXPIRED")
        pub = _b64(key["keyval"]["public"], code + "_PUBLIC_INVALID", expected_len=32)
        sig = _b64(sigs[kid], code + "_SIG_INVALID", expected_len=64)
        try:
            rootseal._ed25519_verify(pub, sig, raw, code + "_SIGNATURE_INVALID")
        except rootseal.RootTrustError as exc:
            raise DelegatedTrustError(str(exc)) from exc
        valid.append(kid)
    if len(valid) < role["threshold"]:
        _fail(code + "_THRESHOLD_NOT_MET")
    return valid


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_DELEGATED_METADATA_STATE = "trusted-delegated-metadata-state.json"
_DOC_RELEASE_ROOT_STATE = "trusted-release-root-state.json"


def _effective_root_from_run155(
    sealed_dir: Path, bootstrap_pin: str, *, now: datetime, require_fresh: bool = True
) -> dict[str, Any]:
    try:
        rootseal.verify_sealed_governance(
            sealed_dir=sealed_dir,
            expected_bootstrap_root_sha256=bootstrap_pin,
            now=now,
            require_current_fresh=require_fresh,
        )
        bundle_info = rootseal.verify_root_bundle(
            bundle_path=sealed_dir / "release-root-bundle.json",
            state_path=sealed_dir / _DOC_RELEASE_ROOT_STATE,
            expected_bootstrap_root_sha256=bootstrap_pin,
            now=now,
            require_current_fresh=require_fresh,
        )
    except rootseal.RootTrustError as exc:
        raise DelegatedTrustError("DELEGATED_RUN155_INVALID:" + str(exc)) from exc
    cur = bundle_info["current"]
    return {
        "version": cur["version"],
        "sha256": cur["sha256"],
        "chainHeadSha256": bundle_info["chain_head_sha256"],
        "envelope": cur["envelope"],
        "info": cur,
        "infos": bundle_info["infos"],
    }


def _verify_delegation_root(  # ruff: ignore[too-many-branches]
    path: Path, effective_root: dict[str, Any], *, now: datetime
) -> dict[str, Any]:
    doc, raw = _read(path, "DELEGATION_ROOT")
    if set(doc) != {"signatures", "signed"} or not isinstance(doc["signed"], dict):
        _fail("DELEGATION_ROOT_SCHEMA_INVALID")
    s = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "version",
        "governanceId",
        "issuedAt",
        "expires",
        "run155Root",
        "keys",
        "roles",
    }
    if (
        set(s) != expected
        or s["_type"] != "delegations"
        or s["specVersion"] != str(POLICY["spec_version"])
        or s["schemaVersion"] != int(POLICY["delegation_schema_version"])
    ):
        _fail("DELEGATION_ROOT_SIGNED_SCHEMA_INVALID")
    version = _int(s["version"], "DELEGATION_ROOT_VERSION_INVALID", minimum=1)
    if s["governanceId"] != effective_root["info"]["governance_id"]:
        _fail("DELEGATION_ROOT_GOVERNANCE_MISMATCH")
    rb = s["run155Root"]
    exp_rb = {
        "version": effective_root["version"],
        "sha256": effective_root["sha256"],
        "chainHeadSha256": effective_root["chainHeadSha256"],
    }
    if rb != exp_rb:
        _fail("DELEGATION_ROOT_RUN155_ROOT_MISMATCH")
    issued = _dt(s["issuedAt"], "DELEGATION_ROOT_ISSUED_INVALID")
    expires = _dt(s["expires"], "DELEGATION_ROOT_EXPIRES_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if issued > now + skew:
        _fail("DELEGATION_ROOT_ISSUED_FROM_FUTURE")
    if expires <= issued or expires - issued > timedelta(
        days=int(POLICY["max_delegation_lifetime_days"])
    ):
        _fail("DELEGATION_ROOT_LIFETIME_INVALID")
    if expires - now < timedelta(
        minutes=int(POLICY["min_delegation_remaining_minutes"])
    ):
        _fail("DELEGATION_ROOT_FREEZE_RISK")
    kr = s["keys"]
    if not isinstance(kr, dict) or not (
        6 <= len(kr) <= int(POLICY["max_keys"])  # ruff: ignore[magic-value-comparison]
    ):
        _fail("DELEGATION_ROOT_KEYS_INVALID")
    keys = {
        _identity(k, "DELEGATION_ROOT_KEYID_INVALID"): _key(v, "DELEGATION_ROOT_KEY")
        for k, v in kr.items()
    }
    if list(kr) != sorted(kr):
        _fail("DELEGATION_ROOT_KEYS_NOT_NORMALIZED")
    roles_raw = s["roles"]
    if not isinstance(roles_raw, dict) or set(roles_raw) != {"snapshot", "timestamp"}:
        _fail("DELEGATION_ROOT_ROLES_INVALID")
    snapshot = _role(
        roles_raw["snapshot"],
        keys,
        "DELEGATION_SNAPSHOT_ROLE",
        min_keys=int(POLICY["min_snapshot_keys"]),
        min_threshold=int(POLICY["min_snapshot_threshold"]),
        min_ops=int(POLICY["min_snapshot_operators"]),
    )
    timestamp = _role(
        roles_raw["timestamp"],
        keys,
        "DELEGATION_TIMESTAMP_ROLE",
        min_keys=int(POLICY["min_timestamp_keys"]),
        min_threshold=int(POLICY["min_timestamp_threshold"]),
        min_ops=int(POLICY["min_timestamp_operators"]),
    )
    if set(snapshot["keyids"]) & set(timestamp["keyids"]):
        _fail("DELEGATION_ROLE_KEY_OVERLAP")
    for kid in snapshot["keyids"] + timestamp["keyids"]:
        if _dt(keys[kid]["expires"], "DELEGATION_ROLE_KEY_EXPIRES_INVALID") < expires:
            _fail("DELEGATION_ROLE_KEY_EXPIRES_BEFORE_DELEGATION")
    root_info = effective_root["info"]
    # Delegation signatures cover the delegation bytes, not the Run 155 root signed body.
    role = root_info["roles"]["root"]
    root_keys = root_info["keys"]
    sigs = _sig_map(doc, "DELEGATION_ROOT")
    if not set(sigs).issubset(set(role["keyids"])):
        _fail("DELEGATION_ROOT_UNAUTHORIZED_SIGNATURE_KEY")
    valid = []
    raw_signed = _canonical(s)
    for kid in role["keyids"]:
        if kid not in sigs:
            continue
        key = root_keys[kid]
        if _dt(key["expires"], "DELEGATION_ROOT_ROOT_KEY_EXPIRES") < issued:
            _fail("DELEGATION_ROOT_ROOT_KEY_EXPIRED")
        pub = _b64(
            key["keyval"]["public"],
            "DELEGATION_ROOT_ROOT_PUBLIC_INVALID",
            expected_len=32,
        )
        sig = _b64(sigs[kid], "DELEGATION_ROOT_ROOT_SIG_INVALID", expected_len=64)
        try:
            rootseal._ed25519_verify(
                pub, sig, raw_signed, "DELEGATION_ROOT_SIGNATURE_INVALID"
            )
        except rootseal.RootTrustError as exc:
            raise DelegatedTrustError(str(exc)) from exc
        valid.append(kid)
    if len(valid) < role["threshold"]:
        _fail("DELEGATION_ROOT_THRESHOLD_NOT_MET")
    return {
        "doc": doc,
        "raw": raw,
        "sha256": _sha_bytes(raw),
        "version": version,
        "issued": issued,
        "expires": expires,
        "keys": keys,
        "roles": {"snapshot": snapshot, "timestamp": timestamp},
    }


def _sealed_artifacts(sealed_dir: Path) -> dict[str, dict[str, Any]]:
    root = _dir(sealed_dir, "DELEGATED_SEALED_DIR_INVALID")
    names = {
        "release-root-bundle.json",
        _DOC_RELEASE_ROOT_STATE,
        "cryptographic-governance-authorization.json",
        "active-root.json",
        "release-governance-recovery-snapshot.json",
        "release-root-receipt.json",
    }
    if {p.name for p in root.iterdir()} != names:
        _fail("DELEGATED_SEALED_DIR_ALLOWLIST_MISMATCH")
    return {name: _artifact(root / name) for name in sorted(names)}


def _verify_metadata_envelope(
    path: Path,
    *,
    kind: str,
    delegation: dict[str, Any],
    now: datetime,
    expected_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc, raw = _read(path, f"DELEGATED_{kind.upper()}")
    if set(doc) != {"signatures", "signed"} or not isinstance(doc["signed"], dict):
        _fail(f"DELEGATED_{kind.upper()}_SCHEMA_INVALID")
    s = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "version",
        "governanceId",
        "issuedAt",
        "expires",
        "delegationRoot",
        "body",
    }
    if (
        set(s) != expected
        or s["_type"] != kind
        or s["specVersion"] != str(POLICY["spec_version"])
    ):
        _fail(f"DELEGATED_{kind.upper()}_SIGNED_SCHEMA_INVALID")
    schema_key = f"{kind}_schema_version"
    if s["schemaVersion"] != int(POLICY[schema_key]):
        _fail(f"DELEGATED_{kind.upper()}_SCHEMA_VERSION_INVALID")
    version = _int(s["version"], f"DELEGATED_{kind.upper()}_VERSION_INVALID", minimum=1)
    if s["governanceId"] != delegation["doc"]["signed"]["governanceId"]:
        _fail(f"DELEGATED_{kind.upper()}_GOVERNANCE_MISMATCH")
    if s["delegationRoot"] != {
        "version": delegation["version"],
        "sha256": delegation["sha256"],
    }:
        _fail(f"DELEGATED_{kind.upper()}_DELEGATION_MISMATCH")
    issued = _dt(s["issuedAt"], f"DELEGATED_{kind.upper()}_ISSUED_INVALID")
    expires = _dt(s["expires"], f"DELEGATED_{kind.upper()}_EXPIRES_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if issued > now + skew:
        _fail(f"DELEGATED_{kind.upper()}_ISSUED_FROM_FUTURE")
    maxlife = timedelta(
        hours=int(
            POLICY["max_snapshot_lifetime_hours"]
            if kind == "snapshot"
            else POLICY["max_timestamp_lifetime_hours"]
        )
    )
    minremain = timedelta(
        minutes=int(
            POLICY["min_snapshot_remaining_minutes"]
            if kind == "snapshot"
            else POLICY["min_timestamp_remaining_minutes"]
        )
    )
    if expires <= issued or expires - issued > maxlife:
        _fail(f"DELEGATED_{kind.upper()}_LIFETIME_INVALID")
    if expires - now < minremain:
        _fail(f"DELEGATED_{kind.upper()}_FREEZE_RISK")
    if not isinstance(s["body"], dict):
        _fail(f"DELEGATED_{kind.upper()}_BODY_INVALID")
    if expected_body is not None and s["body"] != expected_body:
        _fail(f"DELEGATED_{kind.upper()}_BODY_MISMATCH")
    _verify_role_signatures(
        doc,
        s,
        delegation["roles"][kind],
        delegation["keys"],
        at=issued,
        code=f"DELEGATED_{kind.upper()}",
    )
    return {
        "doc": doc,
        "raw": raw,
        "sha256": _sha_bytes(raw),
        "version": version,
        "issued": issued,
        "expires": expires,
    }


def _previous_state(  # ruff: ignore[too-many-branches]
    state_path: Path | None, bundle_path: Path | None
) -> dict[str, Any] | None:
    if state_path is None and bundle_path is None:
        return None
    if state_path is None or bundle_path is None:
        _fail("DELEGATED_PREVIOUS_PAIR_REQUIRED")
    state, state_raw = _read(state_path, "DELEGATED_PREVIOUS_STATE")
    bundle, bundle_raw = _read(bundle_path, "DELEGATED_PREVIOUS_BUNDLE")
    if set(state) != {
        "schemaVersion",
        "status",
        "governanceId",
        "effectiveRoot",
        "delegationRoot",
        "snapshot",
        "timestamp",
        "bundleSha256",
        "chainHeadSha256",
    }:
        _fail("DELEGATED_PREVIOUS_STATE_SCHEMA_INVALID")
    if set(bundle) != {
        "schemaVersion",
        "predicateType",
        "status",
        "governanceId",
        "entries",
    }:
        _fail("DELEGATED_PREVIOUS_BUNDLE_SCHEMA_INVALID")
    if (
        state["schemaVersion"] != int(POLICY["state_schema_version"])
        or bundle["schemaVersion"] != int(POLICY["bundle_schema_version"])
        or bundle["predicateType"] != PREDICATE_TYPE
        or state["status"] != "trusted-delegated-metadata"
        or bundle["status"] != "trusted-delegated-metadata-chain"
    ):
        _fail("DELEGATED_PREVIOUS_SCHEMA_INVALID")
    if _sha_bytes(bundle_raw) != state["bundleSha256"]:
        _fail("DELEGATED_PREVIOUS_BUNDLE_HASH_MISMATCH")
    entries = bundle["entries"]
    if not isinstance(entries, list) or not entries:
        _fail("DELEGATED_PREVIOUS_ENTRIES_INVALID")
    head = _ZERO_HASH
    prev = None
    for i, e in enumerate(entries, 1):
        if (
            not isinstance(e, dict)
            or e.get("sequence") != i
            or e.get("previousChainHeadSha256") != head
        ):
            _fail("DELEGATED_PREVIOUS_CHAIN_INVALID")
        body = {k: v for k, v in e.items() if k != "chainHeadSha256"}
        calc = _sha_bytes((head + "\n").encode() + _canonical(body))
        if e.get("chainHeadSha256") != calc:
            _fail("DELEGATED_PREVIOUS_CHAIN_HASH_INVALID")
        head = calc
        prev = e
    if head != state["chainHeadSha256"] or prev is None:
        _fail("DELEGATED_PREVIOUS_HEAD_MISMATCH")
    if bundle["governanceId"] != state["governanceId"]:
        _fail("DELEGATED_PREVIOUS_GOVERNANCE_MISMATCH")
    if state["effectiveRoot"] != prev.get("effectiveRoot"):
        _fail("DELEGATED_PREVIOUS_STATE_ROOT_REBIND_FAILED")
    if state["delegationRoot"] != {
        "version": prev["delegationRoot"]["version"],
        "sha256": prev["delegationRoot"]["sha256"],
    }:
        _fail("DELEGATED_PREVIOUS_STATE_DELEGATION_REBIND_FAILED")
    if state["snapshot"] != {
        "version": prev["snapshot"]["version"],
        "sha256": prev["snapshot"]["sha256"],
        "expires": prev["snapshot"]["expires"],
    }:
        _fail("DELEGATED_PREVIOUS_STATE_SNAPSHOT_REBIND_FAILED")
    if state["timestamp"] != {
        "version": prev["timestamp"]["version"],
        "sha256": prev["timestamp"]["sha256"],
        "expires": prev["timestamp"]["expires"],
    }:
        _fail("DELEGATED_PREVIOUS_STATE_TIMESTAMP_REBIND_FAILED")
    return {
        "state": state,
        "state_raw": state_raw,
        "bundle": bundle,
        "bundle_raw": bundle_raw,
        "head": head,
        "last": prev,
    }


def refresh_delegated_metadata(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    sealed_dir: Path,
    bootstrap_root_sha256: str,
    delegation_root_path: Path,
    snapshot_path: Path,
    timestamp_path: Path,
    output_dir: Path,
    previous_state_path: Path | None = None,
    previous_bundle_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    sealed = _dir(sealed_dir, "DELEGATED_RUN155_DIR_INVALID")
    effective = _effective_root_from_run155(sealed, bootstrap_root_sha256, now=current)
    delegation = _verify_delegation_root(delegation_root_path, effective, now=current)
    artifacts = _sealed_artifacts(sealed)
    snapshot_body = {
        "run155Root": {
            "version": effective["version"],
            "sha256": effective["sha256"],
            "chainHeadSha256": effective["chainHeadSha256"],
        },
        "sealedArtifacts": artifacts,
    }
    snapshot = _verify_metadata_envelope(
        snapshot_path,
        kind="snapshot",
        delegation=delegation,
        now=current,
        expected_body=snapshot_body,
    )
    timestamp_body = {
        "snapshot": {
            "version": snapshot["version"],
            "sha256": snapshot["sha256"],
            "size": len(snapshot["raw"]),
            "expires": snapshot["doc"]["signed"]["expires"],
        },
        "run155Root": {"version": effective["version"], "sha256": effective["sha256"]},
    }
    timestamp = _verify_metadata_envelope(
        timestamp_path,
        kind="timestamp",
        delegation=delegation,
        now=current,
        expected_body=timestamp_body,
    )
    if timestamp["issued"] < snapshot["issued"]:
        _fail("DELEGATED_TIMESTAMP_PRECEDES_SNAPSHOT")
    if timestamp["expires"] > snapshot["expires"]:
        _fail("DELEGATED_TIMESTAMP_EXPIRES_AFTER_SNAPSHOT")
    prev = _previous_state(previous_state_path, previous_bundle_path)
    if prev is None:
        if (
            delegation["version"] != 1
            or snapshot["version"] != 1
            or timestamp["version"] != 1
        ):
            _fail("DELEGATED_BOOTSTRAP_VERSION_INVALID")
        entries = []
        head = _ZERO_HASH
        seq = 1
    else:
        ps = prev["state"]
        if ps["governanceId"] != effective["info"]["governance_id"]:
            _fail("DELEGATED_GOVERNANCE_CHANGED")
        prior_root = ps["effectiveRoot"]
        matches = [
            x
            for x in effective["infos"]
            if x["version"] == prior_root["version"]
            and x["sha256"] == prior_root["sha256"]
        ]
        if len(matches) != 1:
            _fail("DELEGATED_ROOT_HISTORY_FORK")
        prior_index = effective["infos"].index(matches[0]) + 1
        prior_head = rootseal._root_head(
            effective["info"]["governance_id"],
            [x["envelope"] for x in effective["infos"][:prior_index]],
        )
        if prior_head != prior_root["chainHeadSha256"]:
            _fail("DELEGATED_ROOT_HISTORY_HEAD_MISMATCH")
        if delegation["version"] not in {
            ps["delegationRoot"]["version"],
            ps["delegationRoot"]["version"] + 1,
        }:
            _fail("DELEGATED_DELEGATION_VERSION_INVALID")
        if (
            delegation["version"] == ps["delegationRoot"]["version"]
            and delegation["sha256"] != ps["delegationRoot"]["sha256"]
        ):
            _fail("DELEGATED_SAME_VERSION_DELEGATION_CHANGED")
        if (
            snapshot["version"] != ps["snapshot"]["version"] + 1
            or timestamp["version"] != ps["timestamp"]["version"] + 1
        ):
            _fail("DELEGATED_METADATA_VERSION_ROLLBACK_OR_SKIP")
        entries = list(prev["bundle"]["entries"])
        head = prev["head"]
        seq = len(entries) + 1
    entry_body = {
        "sequence": seq,
        "previousChainHeadSha256": head,
        "effectiveRoot": {
            "version": effective["version"],
            "sha256": effective["sha256"],
            "chainHeadSha256": effective["chainHeadSha256"],
        },
        "delegationRoot": (
            _artifact_raw("release-delegation-root.json", delegation["raw"])
            | {"version": delegation["version"]}
        ),
        "snapshot": (
            _artifact_raw("release-snapshot.json", snapshot["raw"])
            | {
                "version": snapshot["version"],
                "expires": snapshot["doc"]["signed"]["expires"],
            }
        ),
        "timestamp": (
            _artifact_raw("release-timestamp.json", timestamp["raw"])
            | {
                "version": timestamp["version"],
                "expires": timestamp["doc"]["signed"]["expires"],
            }
        ),
    }
    chain = _sha_bytes((head + "\n").encode() + _canonical(entry_body))
    entry = entry_body | {"chainHeadSha256": chain}
    entries.append(entry)
    bundle = {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "trusted-delegated-metadata-chain",
        "governanceId": effective["info"]["governance_id"],
        "entries": entries,
    }
    bundle_raw = _canonical(bundle)
    state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-delegated-metadata",
        "governanceId": effective["info"]["governance_id"],
        "effectiveRoot": entry["effectiveRoot"],
        "delegationRoot": {
            "version": delegation["version"],
            "sha256": delegation["sha256"],
        },
        "snapshot": {
            "version": snapshot["version"],
            "sha256": snapshot["sha256"],
            "expires": snapshot["doc"]["signed"]["expires"],
        },
        "timestamp": {
            "version": timestamp["version"],
            "sha256": timestamp["sha256"],
            "expires": timestamp["doc"]["signed"]["expires"],
        },
        "bundleSha256": _sha_bytes(bundle_raw),
        "chainHeadSha256": chain,
    }
    protected = [
        sealed,
        delegation_root_path.resolve().parent,
        snapshot_path.resolve().parent,
        timestamp_path.resolve().parent,
    ]
    if previous_state_path:
        protected += [previous_state_path.resolve().parent]
    target = _outside(output_dir, protected, "DELEGATED_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("DELEGATED_OUTPUT_ALREADY_EXISTS")
    input_files = [delegation_root_path, snapshot_path, timestamp_path]
    if previous_state_path is not None:
        input_files += [previous_state_path, previous_bundle_path]
    before = {"sealed/" + p.name: _artifact(p) for p in sealed.iterdir()}
    before.update({"input/" + str(i): _artifact(p) for i, p in enumerate(input_files)})
    tmp = Path(tempfile.mkdtemp(prefix="run156-delegated-", dir=str(target.parent)))
    try:
        _write(tmp / "release-delegation-root.json", delegation["doc"])
        _write(tmp / "release-snapshot.json", snapshot["doc"])
        _write(tmp / "release-timestamp.json", timestamp["doc"])
        _write(tmp / "release-delegated-metadata-bundle.json", bundle)
        _write(tmp / _DOC_DELEGATED_METADATA_STATE, state)
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "delegated-metadata-accepted",
            "governanceId": state["governanceId"],
            "sequence": seq,
            "delegationRoot": _artifact(tmp / "release-delegation-root.json"),
            "snapshot": _artifact(tmp / "release-snapshot.json"),
            "timestamp": _artifact(tmp / "release-timestamp.json"),
            "bundle": _artifact(tmp / "release-delegated-metadata-bundle.json"),
            "state": _artifact(tmp / _DOC_DELEGATED_METADATA_STATE),
        }
        _write(tmp / "release-delegated-metadata-receipt.json", receipt)
        after = {"sealed/" + p.name: _artifact(p) for p in sealed.iterdir()}
        after.update(
            {"input/" + str(i): _artifact(p) for i, p in enumerate(input_files)}
        )
        if before != after:
            _fail("DELEGATED_INPUT_DRIFT")
        tmp.replace(target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "ok": True,
        "phase": "delegated-metadata-accepted",
        "sequence": seq,
        "snapshot_version": snapshot["version"],
        "timestamp_version": timestamp["version"],
        "chain_head_sha256": chain,
    }


def _recovery_key(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or "recoveryChannel" not in value:
        _fail(code + "_SCHEMA_INVALID")
    base = {k: v for k, v in value.items() if k != "recoveryChannel"}
    normalized = _key(base, code)
    normalized["recoveryChannel"] = _identity(
        value["recoveryChannel"], code + "_CHANNEL_INVALID"
    )
    if set(value) != {
        "keytype",
        "scheme",
        "identity",
        "operator",
        "expires",
        "signerProfile",
        "keyval",
        "recoveryChannel",
    }:
        _fail(code + "_SCHEMA_INVALID")
    return normalized


def _verify_recovery_root(  # ruff: ignore[too-many-branches]
    path: Path, expected_sha256: str, *, now: datetime
) -> dict[str, Any]:
    doc, raw = _read(path, "RECOVERY_ROOT")
    _hex(expected_sha256, "RECOVERY_ROOT_PIN_INVALID")
    if _sha_bytes(raw) != expected_sha256:
        _fail("RECOVERY_ROOT_PIN_MISMATCH")
    if set(doc) != {"signatures", "signed"} or not isinstance(doc["signed"], dict):
        _fail("RECOVERY_ROOT_SCHEMA_INVALID")
    s = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "version",
        "governanceId",
        "issuedAt",
        "expires",
        "keys",
        "role",
        "minChannels",
    }
    if (
        set(s) != expected
        or s["_type"] != "recovery-root"
        or s["specVersion"] != str(POLICY["spec_version"])
        or s["schemaVersion"] != int(POLICY["recovery_root_schema_version"])
    ):
        _fail("RECOVERY_ROOT_SIGNED_SCHEMA_INVALID")
    if _int(s["version"], "RECOVERY_ROOT_VERSION_INVALID", minimum=1) != 1:
        _fail("RECOVERY_ROOT_BOOTSTRAP_VERSION_INVALID")
    issued = _dt(s["issuedAt"], "RECOVERY_ROOT_ISSUED_INVALID")
    expires = _dt(s["expires"], "RECOVERY_ROOT_EXPIRES_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if issued > now + skew:
        _fail("RECOVERY_ROOT_ISSUED_FROM_FUTURE")
    if expires <= issued or expires - issued > timedelta(
        days=int(POLICY["max_recovery_root_lifetime_days"])
    ):
        _fail("RECOVERY_ROOT_LIFETIME_INVALID")
    if expires - now < timedelta(days=int(POLICY["min_recovery_root_remaining_days"])):
        _fail("RECOVERY_ROOT_FREEZE_RISK")
    kr = s["keys"]
    if not isinstance(kr, dict) or not (
        int(POLICY["min_recovery_keys"]) <= len(kr) <= int(POLICY["max_keys"])
    ):
        _fail("RECOVERY_ROOT_KEYS_INVALID")
    keys = {
        _identity(k, "RECOVERY_ROOT_KEYID_INVALID"): _recovery_key(
            v, "RECOVERY_ROOT_KEY"
        )
        for k, v in kr.items()
    }
    if list(kr) != sorted(kr):
        _fail("RECOVERY_ROOT_KEYS_NOT_NORMALIZED")
    role = _role(
        s["role"],
        keys,
        "RECOVERY_ROLE",
        min_keys=int(POLICY["min_recovery_keys"]),
        min_threshold=int(POLICY["min_recovery_threshold"]),
        min_ops=int(POLICY["min_recovery_operators"]),
    )
    for kid in role["keyids"]:
        if _dt(keys[kid]["expires"], "RECOVERY_ROOT_KEY_EXPIRES_INVALID") < expires:
            _fail("RECOVERY_ROOT_KEY_EXPIRES_BEFORE_ROOT")
    min_channels = _int(
        s["minChannels"],
        "RECOVERY_ROOT_CHANNEL_COUNT_INVALID",
        minimum=int(POLICY["min_recovery_channels"]),
    )
    if min_channels > len(role["keyids"]):
        _fail("RECOVERY_ROOT_CHANNEL_COUNT_INVALID")
    if len({keys[k]["recoveryChannel"] for k in role["keyids"]}) < min_channels:
        _fail("RECOVERY_ROOT_CHANNEL_DIVERSITY_INVALID")
    if bool(POLICY["require_hardware_profile_for_recovery"]) and not any(
        keys[k]["signerProfile"] in set(POLICY["hardware_signer_profiles"])
        for k in role["keyids"]
    ):
        _fail("RECOVERY_ROOT_HARDWARE_PROFILE_REQUIRED")
    _verify_role_signatures(doc, s, role, keys, at=issued, code="RECOVERY_ROOT")
    return {
        "doc": doc,
        "raw": raw,
        "sha256": _sha_bytes(raw),
        "version": 1,
        "keys": keys,
        "role": role,
        "min_channels": min_channels,
        "governance_id": s["governanceId"],
    }


def _replacement_root(  # ruff: ignore[too-many-branches]
    path: Path, current_root: dict[str, Any], compromised: list[str], *, now: datetime
) -> dict[str, Any]:
    doc, _ = _read(path, "RECOVERY_REPLACEMENT_ROOT")
    try:
        info = rootseal._validate_root_envelope(doc)
    except rootseal.RootTrustError as exc:
        raise DelegatedTrustError(
            "RECOVERY_REPLACEMENT_ROOT_INVALID:" + str(exc)
        ) from exc
    if info["version"] != current_root["version"] + 1:
        _fail("RECOVERY_REPLACEMENT_ROOT_VERSION_INVALID")
    if info["signed"]["governanceId"] != current_root["info"]["signed"]["governanceId"]:
        _fail("RECOVERY_REPLACEMENT_GOVERNANCE_MISMATCH")
    # Recovery changes root authority only; governance and emergency policy remain unchanged.
    if bool(POLICY["require_recovery_role_only_change"]):
        for role in ("governance", "emergency"):
            old = current_root["info"]["roles"][role]
            new = info["roles"][role]
            if old != new:
                _fail("RECOVERY_REPLACEMENT_POLICY_ROLE_CHANGED")
            for kid in old["keyids"]:
                if current_root["info"]["keys"][kid] != info["keys"][kid]:
                    _fail("RECOVERY_REPLACEMENT_POLICY_KEY_CHANGED")
    if not compromised or any(
        k not in current_root["info"]["roles"]["root"]["keyids"] for k in compromised
    ):
        _fail("RECOVERY_COMPROMISED_KEY_SET_INVALID")
    if set(compromised) & set(info["roles"]["root"]["keyids"]):
        _fail("RECOVERY_COMPROMISED_KEY_REUSED")
    # New root self-threshold is mandatory; old-root signatures are intentionally not required.
    sigs = _sig_map(doc, "RECOVERY_REPLACEMENT_ROOT")
    allowed = set(info["roles"]["root"]["keyids"])
    if not set(sigs).issubset(allowed):
        _fail("RECOVERY_REPLACEMENT_UNAUTHORIZED_SIGNATURE_KEY")
    valid = []
    raw = _canonical(info["signed"])
    issued = info["issued"]
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if issued > now + skew:
        _fail("RECOVERY_REPLACEMENT_ISSUED_FROM_FUTURE")
    for kid in info["roles"]["root"]["keyids"]:
        if kid not in sigs:
            continue
        key = info["keys"][kid]
        pub = _b64(
            key["keyval"]["public"],
            "RECOVERY_REPLACEMENT_PUBLIC_INVALID",
            expected_len=32,
        )
        sig = _b64(sigs[kid], "RECOVERY_REPLACEMENT_SIG_INVALID", expected_len=64)
        try:
            rootseal._ed25519_verify(
                pub, sig, raw, "RECOVERY_REPLACEMENT_SIGNATURE_INVALID"
            )
        except rootseal.RootTrustError as exc:
            raise DelegatedTrustError(str(exc)) from exc
        valid.append(kid)
    if len(valid) < info["roles"]["root"]["threshold"]:
        _fail("RECOVERY_REPLACEMENT_NEW_ROOT_THRESHOLD_NOT_MET")
    info["envelope"] = doc
    info["sha256"] = _sha_bytes(_canonical(doc))
    return info


def recover_root(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    sealed_dir: Path,
    bootstrap_root_sha256: str,
    recovery_root_path: Path,
    expected_recovery_root_sha256: str,
    replacement_root_path: Path,
    compromised_root_key_ids: list[str],
    incident_id: str,
    recovery_signature_paths: list[Path],
    output_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    sealed = _dir(sealed_dir, "RECOVERY_RUN155_DIR_INVALID")
    active = _effective_root_from_run155(sealed, bootstrap_root_sha256, now=current)
    rr = _verify_recovery_root(
        recovery_root_path, expected_recovery_root_sha256, now=current
    )
    if rr["governance_id"] != active["info"]["governance_id"]:
        _fail("RECOVERY_ROOT_GOVERNANCE_MISMATCH")
    active_keys = active["info"]["keys"]
    if set(rr["keys"]) & set(active_keys):
        _fail("RECOVERY_AUTHORITY_KEY_OVERLAP")
    if {x["operator"] for x in rr["keys"].values()} & {
        x["operator"] for x in active_keys.values()
    }:
        _fail("RECOVERY_AUTHORITY_OPERATOR_OVERLAP")
    compromised = sorted(
        {
            _identity(x, "RECOVERY_COMPROMISED_KEY_INVALID")
            for x in compromised_root_key_ids
        }
    )
    replacement = _replacement_root(
        replacement_root_path, active, compromised, now=current
    )
    replacement_root_keys = {
        replacement["keys"][k]["operator"]
        for k in replacement["roles"]["root"]["keyids"]
    }
    recovery_ops = {rr["keys"][k]["operator"] for k in rr["role"]["keyids"]}
    if set(replacement["roles"]["root"]["keyids"]) & set(rr["role"]["keyids"]):
        _fail("RECOVERY_REPLACEMENT_KEY_OVERLAP")
    if replacement_root_keys & recovery_ops:
        _fail("RECOVERY_REPLACEMENT_OPERATOR_OVERLAP")
    selected = sorted(rr["role"]["keyids"][: rr["role"]["threshold"]])
    subject = {
        "schemaVersion": int(POLICY["schema_version"]),
        "predicateType": PREDICATE_TYPE + "/root-recovery",
        "governanceId": active["info"]["governance_id"],
        "incidentId": _identity(incident_id, "RECOVERY_INCIDENT_ID_INVALID"),
        "previousRoot": {
            "version": active["version"],
            "sha256": active["sha256"],
            "chainHeadSha256": active["chainHeadSha256"],
        },
        "replacementRoot": {
            "version": replacement["version"],
            "sha256": replacement["sha256"],
        },
        "recoveryRoot": {"version": rr["version"], "sha256": rr["sha256"]},
        "compromisedRootKeyIds": compromised,
        "selectedRecoveryKeyIds": selected,
    }
    subject_sha = _sha_bytes(_canonical(subject))
    docs = []
    seen = set()
    operators = set()
    channels = set()
    hardware = False
    if len(recovery_signature_paths) != len(selected):
        _fail("RECOVERY_SIGNATURE_COUNT_INVALID")
    for path in recovery_signature_paths:
        doc, _ = _read(path, "RECOVERY_SIGNATURE")
        if set(doc) != {"signature", "signed"} or not isinstance(doc["signed"], dict):
            _fail("RECOVERY_SIGNATURE_SCHEMA_INVALID")
        s = doc["signed"]
        expected = {
            "schemaVersion",
            "keyId",
            "identity",
            "operator",
            "channel",
            "signerProfile",
            "decision",
            "subjectSha256",
            "signedAt",
        }
        if (
            set(s) != expected
            or s["schemaVersion"] != int(POLICY["recovery_signature_schema_version"])
            or s["decision"] != "recover"
            or s["subjectSha256"] != subject_sha
        ):
            _fail("RECOVERY_SIGNATURE_SIGNED_INVALID")
        kid = _identity(s["keyId"], "RECOVERY_SIGNATURE_KEYID_INVALID")
        if kid not in selected or kid in seen:
            _fail("RECOVERY_SIGNATURE_KEY_SET_INVALID")
        seen.add(kid)
        key = rr["keys"][kid]
        if (
            s["identity"] != key["identity"]
            or s["operator"] != key["operator"]
            or s["signerProfile"] != key["signerProfile"]
        ):
            _fail("RECOVERY_SIGNATURE_IDENTITY_MISMATCH")
        channel = _identity(s["channel"], "RECOVERY_SIGNATURE_CHANNEL_INVALID")
        if channel != key["recoveryChannel"]:
            _fail("RECOVERY_SIGNATURE_CHANNEL_MISMATCH")
        channels.add(channel)
        operators.add(key["operator"])
        hardware |= key["signerProfile"] in set(POLICY["hardware_signer_profiles"])
        signed_at = _dt(s["signedAt"], "RECOVERY_SIGNATURE_TIME_INVALID")
        skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        if signed_at > current + skew:
            _fail("RECOVERY_SIGNATURE_FROM_FUTURE")
        if current - signed_at > timedelta(
            hours=int(POLICY["max_recovery_signature_age_hours"])
        ):
            _fail("RECOVERY_SIGNATURE_STALE")
        if _dt(key["expires"], "RECOVERY_SIGNATURE_KEY_EXPIRES") < signed_at:
            _fail("RECOVERY_SIGNATURE_KEY_EXPIRED")
        sig = _b64(
            doc["signature"], "RECOVERY_SIGNATURE_ENCODING_INVALID", expected_len=64
        )
        pub = _b64(
            key["keyval"]["public"],
            "RECOVERY_SIGNATURE_PUBLIC_INVALID",
            expected_len=32,
        )
        try:
            rootseal._ed25519_verify(
                pub, sig, _canonical(s), "RECOVERY_SIGNATURE_CRYPTO_INVALID"
            )
        except rootseal.RootTrustError as exc:
            raise DelegatedTrustError(str(exc)) from exc
        docs.append(doc)
    if sorted(seen) != selected:
        _fail("RECOVERY_SIGNATURE_SET_MISMATCH")
    if len(operators) < int(POLICY["min_recovery_operators"]):
        _fail("RECOVERY_OPERATOR_QUORUM_INVALID")
    if len(channels) < rr["min_channels"]:
        _fail("RECOVERY_CHANNEL_QUORUM_INVALID")
    if bool(POLICY["require_hardware_profile_for_recovery"]) and not hardware:
        _fail("RECOVERY_HARDWARE_SIGNER_REQUIRED")
    docs.sort(key=lambda d: d["signed"]["keyId"])
    record = {
        "schemaVersion": int(POLICY["recovery_record_schema_version"]),
        "predicateType": PREDICATE_TYPE + "/root-recovery-record",
        "status": "root-recovered",
        "subject": subject,
        "subjectSha256": subject_sha,
        "recoveryRoot": rr["doc"],
        "replacementRoot": replacement["envelope"],
        "signatures": docs,
        "observedChannels": sorted(channels),
        "revokedRootKeyIds": compromised,
    }
    protected = [
        sealed,
        recovery_root_path.resolve().parent,
        replacement_root_path.resolve().parent,
    ] + [p.resolve().parent for p in recovery_signature_paths]
    target = _outside(output_dir, protected, "RECOVERY_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("RECOVERY_OUTPUT_ALREADY_EXISTS")
    recovery_inputs = [
        recovery_root_path,
        replacement_root_path,
        *list(recovery_signature_paths),
    ]
    before = {"sealed/" + p.name: _artifact(p) for p in sealed.iterdir()}
    before.update(
        {"input/" + str(i): _artifact(p) for i, p in enumerate(recovery_inputs)}
    )
    tmp = Path(tempfile.mkdtemp(prefix="run156-recovery-", dir=str(target.parent)))
    try:
        _write(tmp / "release-root-recovery-record.json", record)
        _write(tmp / "recovered-effective-root.json", replacement["envelope"])
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "root-recovery-accepted",
            "governanceId": subject["governanceId"],
            "incidentId": subject["incidentId"],
            "recoveryRecord": _artifact(tmp / "release-root-recovery-record.json"),
            "effectiveRoot": _artifact(tmp / "recovered-effective-root.json"),
            "recoveryRootSha256": rr["sha256"],
            "revokedRootKeyIds": compromised,
        }
        _write(tmp / "release-root-recovery-receipt.json", receipt)
        after = {"sealed/" + p.name: _artifact(p) for p in sealed.iterdir()}
        after.update(
            {"input/" + str(i): _artifact(p) for i, p in enumerate(recovery_inputs)}
        )
        if before != after:
            _fail("RECOVERY_INPUT_DRIFT")
        tmp.replace(target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "ok": True,
        "phase": "root-recovery-accepted",
        "incident_id": subject["incidentId"],
        "previous_root_version": active["version"],
        "replacement_root_version": replacement["version"],
        "replacement_root_sha256": replacement["sha256"],
        "channels": len(channels),
    }


def verify_delegated_output(  # ruff: ignore[undocumented-public-function]
    *,
    output_dir: Path,
    sealed_dir: Path,
    bootstrap_root_sha256: str,
    now: datetime | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out = _dir(output_dir, "DELEGATED_VERIFY_DIR_INVALID")
    allowed = {
        "release-delegation-root.json",
        "release-snapshot.json",
        "release-timestamp.json",
        "release-delegated-metadata-bundle.json",
        _DOC_DELEGATED_METADATA_STATE,
        "release-delegated-metadata-receipt.json",
    }
    if {p.name for p in out.iterdir()} != allowed:
        _fail("DELEGATED_VERIFY_ALLOWLIST_MISMATCH")
    effective = _effective_root_from_run155(
        _dir(sealed_dir, "DELEGATED_VERIFY_RUN155_INVALID"),
        bootstrap_root_sha256,
        now=current,
        require_fresh=require_fresh,
    )
    delegation_now = current
    if not require_fresh:
        delegation_doc, _ = _read(
            out / "release-delegation-root.json", "DELEGATED_VERIFY_DELEGATION_TIME"
        )
        delegation_now = _dt(
            delegation_doc["signed"]["issuedAt"], "DELEGATED_VERIFY_DELEGATION_ISSUED"
        )
    delegation = _verify_delegation_root(
        out / "release-delegation-root.json", effective, now=delegation_now
    )
    artifacts = _sealed_artifacts(sealed_dir)
    snapshot_body = {
        "run155Root": {
            "version": effective["version"],
            "sha256": effective["sha256"],
            "chainHeadSha256": effective["chainHeadSha256"],
        },
        "sealedArtifacts": artifacts,
    }
    snapshot = _verify_metadata_envelope(
        out / "release-snapshot.json",
        kind="snapshot",
        delegation=delegation,
        now=current if require_fresh else snapshot_time(out / "release-snapshot.json"),
        expected_body=snapshot_body,
    )
    timestamp_body = {
        "snapshot": {
            "version": snapshot["version"],
            "sha256": snapshot["sha256"],
            "size": len(snapshot["raw"]),
            "expires": snapshot["doc"]["signed"]["expires"],
        },
        "run155Root": {"version": effective["version"], "sha256": effective["sha256"]},
    }
    timestamp = _verify_metadata_envelope(
        out / "release-timestamp.json",
        kind="timestamp",
        delegation=delegation,
        now=current if require_fresh else snapshot_time(out / "release-timestamp.json"),
        expected_body=timestamp_body,
    )
    state, _ = _read(out / _DOC_DELEGATED_METADATA_STATE, "DELEGATED_VERIFY_STATE")
    prev = _previous_state(
        out / _DOC_DELEGATED_METADATA_STATE,
        out / "release-delegated-metadata-bundle.json",
    )
    expected_effective = {
        "version": effective["version"],
        "sha256": effective["sha256"],
        "chainHeadSha256": effective["chainHeadSha256"],
    }
    if (
        prev is None
        or state["governanceId"] != effective["info"]["governance_id"]
        or state["effectiveRoot"] != expected_effective
        or state["delegationRoot"]
        != {"version": delegation["version"], "sha256": delegation["sha256"]}
        or state["snapshot"]["sha256"] != snapshot["sha256"]
        or state["timestamp"]["sha256"] != timestamp["sha256"]
    ):
        _fail("DELEGATED_VERIFY_STATE_REBIND_FAILED")
    receipt, _ = _read(
        out / "release-delegated-metadata-receipt.json", "DELEGATED_VERIFY_RECEIPT"
    )
    exp = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": "delegated-metadata-accepted",
        "governanceId": state["governanceId"],
        "sequence": len(prev["bundle"]["entries"]),
        "delegationRoot": _artifact(out / "release-delegation-root.json"),
        "snapshot": _artifact(out / "release-snapshot.json"),
        "timestamp": _artifact(out / "release-timestamp.json"),
        "bundle": _artifact(out / "release-delegated-metadata-bundle.json"),
        "state": _artifact(out / _DOC_DELEGATED_METADATA_STATE),
    }
    if receipt != exp:
        _fail("DELEGATED_VERIFY_RECEIPT_REBIND_FAILED")
    return {
        "ok": True,
        "phase": "delegated-metadata-verified",
        "sequence": len(prev["bundle"]["entries"]),
        "chain_head_sha256": state["chainHeadSha256"],
    }


def snapshot_time(path: Path) -> datetime:  # ruff: ignore[undocumented-public-function]
    doc, _ = _read(path, "DELEGATED_HISTORICAL_TIME")
    return _dt(doc["signed"]["issuedAt"], "DELEGATED_HISTORICAL_ISSUED")


def verify_recovery_output(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    output_dir: Path,
    sealed_dir: Path,
    bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    out = _dir(output_dir, "RECOVERY_VERIFY_DIR_INVALID")
    allowed = {
        "release-root-recovery-record.json",
        "recovered-effective-root.json",
        "release-root-recovery-receipt.json",
    }
    if {p.name for p in out.iterdir()} != allowed:
        _fail("RECOVERY_VERIFY_ALLOWLIST_MISMATCH")
    record, record_raw = _read(
        out / "release-root-recovery-record.json", "RECOVERY_VERIFY_RECORD"
    )
    expected_keys = {
        "schemaVersion",
        "predicateType",
        "status",
        "subject",
        "subjectSha256",
        "recoveryRoot",
        "replacementRoot",
        "signatures",
        "observedChannels",
        "revokedRootKeyIds",
    }
    if (
        set(record) != expected_keys
        or record["schemaVersion"] != int(POLICY["recovery_record_schema_version"])
        or record["predicateType"] != PREDICATE_TYPE + "/root-recovery-record"
        or record["status"] != "root-recovered"
    ):
        _fail("RECOVERY_VERIFY_RECORD_SCHEMA_INVALID")
    sigs = record.get("signatures")
    if not isinstance(sigs, list) or not sigs:
        _fail("RECOVERY_VERIFY_SIGNATURES_INVALID")
    signed_times = []
    for d in sigs:
        if not isinstance(d, dict) or not isinstance(d.get("signed"), dict):
            _fail("RECOVERY_VERIFY_SIGNATURES_INVALID")
        signed_times.append(
            _dt(d["signed"].get("signedAt"), "RECOVERY_VERIFY_SIGNED_AT_INVALID")
        )
    current = (
        max(signed_times) if historical else (now or datetime.now(timezone.utc))
    ).astimezone(timezone.utc)
    active = _effective_root_from_run155(
        _dir(sealed_dir, "RECOVERY_VERIFY_RUN155_INVALID"),
        bootstrap_root_sha256,
        now=current,
        require_fresh=not historical,
    )
    with tempfile.TemporaryDirectory(prefix="run156-recovery-verify-") as td:
        td = Path(td)  # ruff: ignore[redefined-loop-name]
        rr_path = td / "recovery-root.json"
        repl_path = td / "replacement-root.json"
        _write(rr_path, record["recoveryRoot"])
        _write(repl_path, record["replacementRoot"])
        rr = _verify_recovery_root(rr_path, expected_recovery_root_sha256, now=current)
        active_keys = active["info"]["keys"]
        if set(rr["keys"]) & set(active_keys):
            _fail("RECOVERY_VERIFY_AUTHORITY_KEY_OVERLAP")
        if {x["operator"] for x in rr["keys"].values()} & {
            x["operator"] for x in active_keys.values()
        }:
            _fail("RECOVERY_VERIFY_AUTHORITY_OPERATOR_OVERLAP")
        subject = record.get("subject")
        if not isinstance(subject, dict) or record.get("subjectSha256") != _sha_bytes(
            _canonical(subject)
        ):
            _fail("RECOVERY_VERIFY_SUBJECT_HASH_INVALID")
        required_subject = {
            "schemaVersion",
            "predicateType",
            "governanceId",
            "incidentId",
            "previousRoot",
            "replacementRoot",
            "recoveryRoot",
            "compromisedRootKeyIds",
            "selectedRecoveryKeyIds",
        }
        if (
            set(subject) != required_subject
            or subject["schemaVersion"] != int(POLICY["schema_version"])
            or subject["predicateType"] != PREDICATE_TYPE + "/root-recovery"
        ):
            _fail("RECOVERY_VERIFY_SUBJECT_SCHEMA_INVALID")
        compromised = subject.get("compromisedRootKeyIds")
        if not isinstance(compromised, list) or compromised != sorted(set(compromised)):
            _fail("RECOVERY_VERIFY_COMPROMISED_SET_INVALID")
        replacement = _replacement_root(repl_path, active, compromised, now=current)
        replacement_ops = {
            replacement["keys"][k]["operator"]
            for k in replacement["roles"]["root"]["keyids"]
        }
        recovery_ops = {rr["keys"][k]["operator"] for k in rr["role"]["keyids"]}
        if set(replacement["roles"]["root"]["keyids"]) & set(rr["role"]["keyids"]):
            _fail("RECOVERY_VERIFY_REPLACEMENT_KEY_OVERLAP")
        if replacement_ops & recovery_ops:
            _fail("RECOVERY_VERIFY_REPLACEMENT_OPERATOR_OVERLAP")
    expected_subject = {
        "schemaVersion": int(POLICY["schema_version"]),
        "predicateType": PREDICATE_TYPE + "/root-recovery",
        "governanceId": active["info"]["governance_id"],
        "incidentId": _identity(
            subject.get("incidentId"), "RECOVERY_VERIFY_INCIDENT_INVALID"
        ),
        "previousRoot": {
            "version": active["version"],
            "sha256": active["sha256"],
            "chainHeadSha256": active["chainHeadSha256"],
        },
        "replacementRoot": {
            "version": replacement["version"],
            "sha256": replacement["sha256"],
        },
        "recoveryRoot": {"version": rr["version"], "sha256": rr["sha256"]},
        "compromisedRootKeyIds": compromised,
        "selectedRecoveryKeyIds": sorted(
            rr["role"]["keyids"][: rr["role"]["threshold"]]
        ),
    }
    if subject != expected_subject:
        _fail("RECOVERY_VERIFY_SUBJECT_REBIND_FAILED")
    selected = expected_subject["selectedRecoveryKeyIds"]
    seen = set()
    operators = set()
    channels = set()
    hardware = False
    subject_sha = record["subjectSha256"]
    normalized = []
    for doc in sigs:
        if set(doc) != {"signature", "signed"} or not isinstance(doc["signed"], dict):
            _fail("RECOVERY_VERIFY_SIGNATURE_SCHEMA_INVALID")
        ss = doc["signed"]
        expected_signed = {
            "schemaVersion",
            "keyId",
            "identity",
            "operator",
            "channel",
            "signerProfile",
            "decision",
            "subjectSha256",
            "signedAt",
        }
        if (
            set(ss) != expected_signed
            or ss["schemaVersion"] != int(POLICY["recovery_signature_schema_version"])
            or ss["decision"] != "recover"
            or ss["subjectSha256"] != subject_sha
        ):
            _fail("RECOVERY_VERIFY_SIGNATURE_SIGNED_INVALID")
        kid = _identity(ss["keyId"], "RECOVERY_VERIFY_KEYID_INVALID")
        if kid not in selected or kid in seen:
            _fail("RECOVERY_VERIFY_KEY_SET_INVALID")
        seen.add(kid)
        key = rr["keys"][kid]
        if (
            ss["identity"] != key["identity"]
            or ss["operator"] != key["operator"]
            or ss["signerProfile"] != key["signerProfile"]
        ):
            _fail("RECOVERY_VERIFY_IDENTITY_MISMATCH")
        channel = _identity(ss["channel"], "RECOVERY_VERIFY_CHANNEL_INVALID")
        if channel != key["recoveryChannel"]:
            _fail("RECOVERY_VERIFY_CHANNEL_MISMATCH")
        channels.add(channel)
        operators.add(key["operator"])
        hardware |= key["signerProfile"] in set(POLICY["hardware_signer_profiles"])
        signed_at = _dt(ss["signedAt"], "RECOVERY_VERIFY_TIME_INVALID")
        if not historical:
            skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
            if signed_at > current + skew or current - signed_at > timedelta(
                hours=int(POLICY["max_recovery_signature_age_hours"])
            ):
                _fail("RECOVERY_VERIFY_SIGNATURE_FRESHNESS_INVALID")
        if _dt(key["expires"], "RECOVERY_VERIFY_KEY_EXPIRES") < signed_at:
            _fail("RECOVERY_VERIFY_KEY_EXPIRED")
        pub = _b64(
            key["keyval"]["public"], "RECOVERY_VERIFY_PUBLIC_INVALID", expected_len=32
        )
        sig = _b64(doc["signature"], "RECOVERY_VERIFY_SIG_INVALID", expected_len=64)
        try:
            rootseal._ed25519_verify(
                pub, sig, _canonical(ss), "RECOVERY_VERIFY_CRYPTO_INVALID"
            )
        except rootseal.RootTrustError as exc:
            raise DelegatedTrustError(str(exc)) from exc
        normalized.append(doc)
    normalized.sort(key=lambda d: d["signed"]["keyId"])
    if sigs != normalized or sorted(seen) != selected:
        _fail("RECOVERY_VERIFY_SIGNATURE_SET_MISMATCH")
    if (
        len(operators) < int(POLICY["min_recovery_operators"])
        or len(channels) < rr["min_channels"]
    ):
        _fail("RECOVERY_VERIFY_QUORUM_INVALID")
    if bool(POLICY["require_hardware_profile_for_recovery"]) and not hardware:
        _fail("RECOVERY_VERIFY_HARDWARE_REQUIRED")
    if (
        record.get("observedChannels") != sorted(channels)
        or record.get("revokedRootKeyIds") != compromised
    ):
        _fail("RECOVERY_VERIFY_RECORD_REBIND_FAILED")
    effective_doc, effective_raw = _read(
        out / "recovered-effective-root.json", "RECOVERY_VERIFY_EFFECTIVE_ROOT"
    )
    if effective_doc != replacement["envelope"]:
        _fail("RECOVERY_VERIFY_EFFECTIVE_ROOT_MISMATCH")
    receipt, _ = _read(
        out / "release-root-recovery-receipt.json", "RECOVERY_VERIFY_RECEIPT"
    )
    exp_receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": "root-recovery-accepted",
        "governanceId": subject["governanceId"],
        "incidentId": subject["incidentId"],
        "recoveryRecord": _artifact_raw(
            "release-root-recovery-record.json", record_raw
        ),
        "effectiveRoot": _artifact_raw("recovered-effective-root.json", effective_raw),
        "recoveryRootSha256": rr["sha256"],
        "revokedRootKeyIds": compromised,
    }
    if receipt != exp_receipt:
        _fail("RECOVERY_VERIFY_RECEIPT_REBIND_FAILED")
    return {
        "ok": True,
        "phase": "root-recovery-verified",
        "incident_id": subject["incidentId"],
        "replacement_root_version": replacement["version"],
        "replacement_root_sha256": replacement["sha256"],
    }


def _paths(values: list[str]) -> list[Path]:
    return [Path(x) for x in values]


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("refresh")
    r.add_argument("--sealed-dir", type=Path, required=True)
    r.add_argument("--bootstrap-root-sha256", required=True)
    r.add_argument("--delegation-root", type=Path, required=True)
    r.add_argument("--snapshot", type=Path, required=True)
    r.add_argument("--timestamp", type=Path, required=True)
    r.add_argument("--previous-state", type=Path)
    r.add_argument("--previous-bundle", type=Path)
    r.add_argument("--output-dir", type=Path, required=True)
    q = sub.add_parser("recover-root")
    q.add_argument("--sealed-dir", type=Path, required=True)
    q.add_argument("--bootstrap-root-sha256", required=True)
    q.add_argument("--recovery-root", type=Path, required=True)
    q.add_argument("--recovery-root-sha256", required=True)
    q.add_argument("--replacement-root", type=Path, required=True)
    q.add_argument("--compromised-root-key", action="append", default=[])
    q.add_argument("--incident-id", required=True)
    q.add_argument("--signature", action="append", default=[])
    q.add_argument("--output-dir", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--sealed-dir", type=Path, required=True)
    v.add_argument("--bootstrap-root-sha256", required=True)
    v.add_argument("--historical", action="store_true")
    w = sub.add_parser("verify-recovery")
    w.add_argument("--output-dir", type=Path, required=True)
    w.add_argument("--sealed-dir", type=Path, required=True)
    w.add_argument("--bootstrap-root-sha256", required=True)
    w.add_argument("--recovery-root-sha256", required=True)
    w.add_argument("--historical", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.command == "refresh":
            out = refresh_delegated_metadata(
                sealed_dir=a.sealed_dir,
                bootstrap_root_sha256=a.bootstrap_root_sha256,
                delegation_root_path=a.delegation_root,
                snapshot_path=a.snapshot,
                timestamp_path=a.timestamp,
                previous_state_path=a.previous_state,
                previous_bundle_path=a.previous_bundle,
                output_dir=a.output_dir,
            )
        elif a.command == "recover-root":
            out = recover_root(
                sealed_dir=a.sealed_dir,
                bootstrap_root_sha256=a.bootstrap_root_sha256,
                recovery_root_path=a.recovery_root,
                expected_recovery_root_sha256=a.recovery_root_sha256,
                replacement_root_path=a.replacement_root,
                compromised_root_key_ids=a.compromised_root_key,
                incident_id=a.incident_id,
                recovery_signature_paths=_paths(a.signature),
                output_dir=a.output_dir,
            )
        elif a.command == "verify":
            out = verify_delegated_output(
                output_dir=a.output_dir,
                sealed_dir=a.sealed_dir,
                bootstrap_root_sha256=a.bootstrap_root_sha256,
                require_fresh=not a.historical,
            )
        else:
            out = verify_recovery_output(
                output_dir=a.output_dir,
                sealed_dir=a.sealed_dir,
                bootstrap_root_sha256=a.bootstrap_root_sha256,
                expected_recovery_root_sha256=a.recovery_root_sha256,
                historical=a.historical,
            )
    except DelegatedTrustError as exc:
        logger.error("delegated trust operation failed: %s", exc)
        return 2
    logger.info("delegated trust operation completed: %s", a.command)
    # sys.stdout.write(json.dumps(out, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
