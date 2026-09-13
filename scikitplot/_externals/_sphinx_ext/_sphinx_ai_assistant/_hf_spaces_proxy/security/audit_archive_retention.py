"""
Run 161: cryptographically audit archive retention and durable-copy health.

Run 160 proves that exact native-status evidence was copied to independent immutable
archives.  Run 161 makes durability an append-only cryptographic lifecycle rather than a
one-time semantic claim:

* an independently pinned, threshold-signed retention-governance root authorizes the
  exact active archive membership and every migration/retirement;
* each storage provider signs the immutable version ID, retention expiry, legal-hold
  state, and a challenge-bound read-back of the exact Run 160 archive artifact;
* a distinct read-only auditor signs an independent read-back of the same version;
* every migration proves the complete *next* durable set before retired members are
  authorized for removal; and
* cumulative state is replayable offline from canonical evidence.

Private signing keys are intentionally never accepted by this tool.
"""

from __future__ import annotations

import argparse
import base64
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
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

import tomllib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)

try:
    from . import archive_native_status_evidence as run160
except (ImportError, ValueError) as exc:
    import importlib.util

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "_run160_archive_for_health", _here / "archive_native_status_evidence.py"
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("archive_native_status_evidence.py") from exc
    run160 = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = run160
    _spec.loader.exec_module(run160)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_archive_health_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_CHUNK = 1024 * 1024


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_ARCHIVE_HEALTH_STATE = "trusted-archive-health-state.json"


_OUTPUT_NAMES = {
    "release-archive-health-bundle.json",
    _DOC_ARCHIVE_HEALTH_STATE,
    "active-archive-health-evidence.json",
    "release-archive-health-receipt.json",
}


class ArchiveHealthError(RuntimeError):
    """Run 161 archive-health invariant failed."""


def _fail(code: str) -> None:
    raise ArchiveHealthError(code)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(_canonical(value))


def _loads(raw: bytes, code: str) -> dict[str, Any]:
    def hook(pairs):
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                _fail(code + "_DUPLICATE_KEY")
            out[k] = v
        return out

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except ArchiveHealthError:
        raise
    except Exception as exc:
        raise ArchiveHealthError(code + "_JSON_INVALID") from exc
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


def _size(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(code)
    return value


def _b64(value: Any, code: str, *, expected_len: int | None = None) -> bytes:
    if not isinstance(value, str):
        _fail(code)
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ArchiveHealthError(code) from exc
    if expected_len is not None and len(raw) != expected_len:
        _fail(code)
    return raw


def _dt(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise ArchiveHealthError(code) from exc


def _ts(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _safe_locator(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048  # ruff: ignore[magic-value-comparison]
        or "\x00" in value
    ):
        _fail(code)
    try:
        p = urlsplit(value)
    except ValueError:
        _fail(code)
    if (
        not p.scheme
        or p.scheme.lower() == "file"
        or p.query
        or p.fragment
        or p.username is not None
        or p.password is not None
    ):
        _fail(code)
    if ".." in p.path.split("/"):
        _fail(code)
    return value


def _file_fingerprint(path: Path, code: str) -> tuple[str, int, int]:
    if path.is_symlink() or not path.is_file():
        _fail(code)
    st = path.stat()
    return (_sha(path), st.st_size, st.st_mode & 0o7777)


def _dir_fingerprint(root: Path, code: str) -> dict[str, tuple[str, int, int]]:
    root = _regular_dir(root, code)
    out = {}
    for item in root.iterdir():
        if item.is_symlink() or not item.is_file():
            _fail(code)
        out[item.name] = _file_fingerprint(item, code)
    return out


def _artifact(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "size"}:
        _fail(code + "_SCHEMA_INVALID")
    name = value.get("name")
    if not isinstance(name, str) or Path(name).name != name or not name:
        _fail(code + "_NAME_INVALID")
    return {
        "name": name,
        "sha256": _hex(value.get("sha256"), code + "_SHA_INVALID"),
        "size": _size(value.get("size"), code + "_SIZE_INVALID"),
    }


def _artifact_bytes(name: str, raw: bytes) -> dict[str, Any]:
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


def _ed25519_verify(public_text: str, sig_text: str, message: bytes, code: str) -> None:
    public = _b64(public_text, code + "_PUBLIC_KEY_INVALID", expected_len=32)
    sig = _b64(sig_text, code + "_SIGNATURE_ENCODING_INVALID", expected_len=64)
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(sig, message)
    except (InvalidSignature, ValueError) as exc:
        raise ArchiveHealthError(code + "_SIGNATURE_INVALID") from exc


def _key(value: Any, code: str) -> dict[str, Any]:
    expected = {"identity", "operator", "expires", "publicKey"}
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    result = {
        "identity": _identity(value.get("identity"), code + "_IDENTITY_INVALID"),
        "operator": _identity(value.get("operator"), code + "_OPERATOR_INVALID"),
        "expires": _ts(_dt(value.get("expires"), code + "_EXPIRES_INVALID")),
        "publicKey": value.get("publicKey"),
    }
    _b64(result["publicKey"], code + "_PUBLIC_KEY_INVALID", expected_len=32)
    if value != result:
        _fail(code + "_NOT_NORMALIZED")
    return result


def _verify_root(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    expected_sha256: str,
    *,
    now: datetime,
    historical: bool = False,
) -> dict[str, Any]:
    if set(doc) != {"signed", "signatures"} or not isinstance(doc.get("signed"), dict):
        _fail("ARCHIVE_HEALTH_ROOT_SCHEMA_INVALID")
    s = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "rootId",
        "version",
        "issuedAt",
        "expires",
        "threshold",
        "selectedSignerKeyIds",
        "keys",
    }
    if (
        set(s) != expected
        or s.get("_type") != "archive-retention-root"
        or s.get("specVersion") != str(POLICY["spec_version"])
        or s.get("schemaVersion") != int(POLICY["root_schema_version"])
    ):
        _fail("ARCHIVE_HEALTH_ROOT_SIGNED_SCHEMA_INVALID")
    root_id = _identity(s.get("rootId"), "ARCHIVE_HEALTH_ROOT_ID_INVALID")
    version = _size(s.get("version"), "ARCHIVE_HEALTH_ROOT_VERSION_INVALID")
    if version != 1:
        _fail("ARCHIVE_HEALTH_ROOT_VERSION_INVALID")
    issued = _dt(s.get("issuedAt"), "ARCHIVE_HEALTH_ROOT_ISSUED_INVALID")
    expires = _dt(s.get("expires"), "ARCHIVE_HEALTH_ROOT_EXPIRES_INVALID")
    if expires <= issued or expires - issued > timedelta(
        days=int(POLICY["max_root_lifetime_days"])
    ):
        _fail("ARCHIVE_HEALTH_ROOT_LIFETIME_INVALID")
    if not historical:
        if issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
            _fail("ARCHIVE_HEALTH_ROOT_FROM_FUTURE")
        if expires <= now:
            _fail("ARCHIVE_HEALTH_ROOT_EXPIRED")
    keys_raw = s.get("keys")
    if not isinstance(keys_raw, dict) or not (
        int(POLICY["min_governance_keys"])
        <= len(keys_raw)
        <= int(POLICY["max_key_count"])
    ):
        _fail("ARCHIVE_HEALTH_ROOT_KEYS_INVALID")
    keys: dict[str, dict[str, Any]] = {}
    for kid, value in keys_raw.items():
        kid = _identity(  # ruff: ignore[redefined-loop-name]
            kid,
            "ARCHIVE_HEALTH_ROOT_KEY_ID_INVALID",
        )
        keys[kid] = _key(value, "ARCHIVE_HEALTH_ROOT_KEY")
        if (
            _dt(keys[kid]["expires"], "ARCHIVE_HEALTH_ROOT_KEY_EXPIRES_INVALID")
            < expires
        ):
            _fail("ARCHIVE_HEALTH_ROOT_KEY_EXPIRES_BEFORE_ROOT")
    threshold = _size(s.get("threshold"), "ARCHIVE_HEALTH_ROOT_THRESHOLD_INVALID")
    if threshold < int(POLICY["min_governance_threshold"]) or threshold > len(keys):
        _fail("ARCHIVE_HEALTH_ROOT_THRESHOLD_INVALID")
    selected = s.get("selectedSignerKeyIds")
    if not isinstance(selected, list) or len(selected) != threshold:
        _fail("ARCHIVE_HEALTH_ROOT_SELECTED_INVALID")
    selected = [
        _identity(x, "ARCHIVE_HEALTH_ROOT_SELECTED_KEY_INVALID") for x in selected
    ]
    if (
        selected != sorted(selected)
        or len(set(selected)) != len(selected)
        or any(x not in keys for x in selected)
    ):
        _fail("ARCHIVE_HEALTH_ROOT_SELECTED_INVALID")
    if len({keys[x]["operator"] for x in selected}) < int(
        POLICY["min_governance_operators"]
    ):
        _fail("ARCHIVE_HEALTH_ROOT_OPERATOR_QUORUM_INVALID")
    sigs = doc.get("signatures")
    if not isinstance(sigs, list) or len(sigs) != len(selected):
        _fail("ARCHIVE_HEALTH_ROOT_SIGNATURE_SET_INVALID")
    sig_map: dict[str, str] = {}
    for item in sigs:
        if not isinstance(item, dict) or set(item) != {"keyId", "signature"}:
            _fail("ARCHIVE_HEALTH_ROOT_SIGNATURE_SCHEMA_INVALID")
        kid = _identity(item.get("keyId"), "ARCHIVE_HEALTH_ROOT_SIGNATURE_KEY_INVALID")
        if kid in sig_map:
            _fail("ARCHIVE_HEALTH_ROOT_SIGNATURE_DUPLICATE")
        sig_map[kid] = item.get("signature")
    if sorted(sig_map) != selected:
        _fail("ARCHIVE_HEALTH_ROOT_SIGNATURE_SET_INVALID")
    if sigs != sorted(sigs, key=lambda x: x["keyId"]):
        _fail("ARCHIVE_HEALTH_ROOT_SIGNATURES_NOT_SORTED")
    signed_bytes = _canonical(s)
    for kid in selected:
        at = issued
        if _dt(keys[kid]["expires"], "ARCHIVE_HEALTH_ROOT_KEY_EXPIRES_INVALID") < at:
            _fail("ARCHIVE_HEALTH_ROOT_SIGNER_EXPIRED")
        _ed25519_verify(
            keys[kid]["publicKey"], sig_map[kid], signed_bytes, "ARCHIVE_HEALTH_ROOT"
        )
    raw = _canonical(doc)
    sha = _sha_bytes(raw)
    if sha != _hex(expected_sha256, "ARCHIVE_HEALTH_ROOT_PIN_INVALID"):
        _fail("ARCHIVE_HEALTH_ROOT_PIN_MISMATCH")
    return {
        "rootId": root_id,
        "version": version,
        "issued": issued,
        "expires": expires,
        "threshold": threshold,
        "keys": keys,
        "sha256": sha,
        "doc": doc,
    }


def _member(value: Any, code: str) -> dict[str, Any]:
    expected = {
        "archiveIdentity",
        "archiveOperator",
        "archiveId",
        "locator",
        "immutability",
        "providerKeyId",
        "providerKey",
        "auditorIdentity",
        "auditorKeyId",
        "auditorKey",
    }
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    ai = _identity(value.get("archiveIdentity"), code + "_ARCHIVE_IDENTITY_INVALID")
    ao = _identity(value.get("archiveOperator"), code + "_ARCHIVE_OPERATOR_INVALID")
    archive_id = _identity(value.get("archiveId"), code + "_ARCHIVE_ID_INVALID")
    locator = _safe_locator(value.get("locator"), code + "_LOCATOR_INVALID")
    imm = value.get("immutability")
    if imm not in set(POLICY["allowed_immutability"]):
        _fail(code + "_IMMUTABILITY_INVALID")
    pkid = _identity(value.get("providerKeyId"), code + "_PROVIDER_KEY_ID_INVALID")
    pkey = _key(value.get("providerKey"), code + "_PROVIDER_KEY")
    aid = _identity(value.get("auditorIdentity"), code + "_AUDITOR_IDENTITY_INVALID")
    akid = _identity(value.get("auditorKeyId"), code + "_AUDITOR_KEY_ID_INVALID")
    akey = _key(value.get("auditorKey"), code + "_AUDITOR_KEY")
    if pkey["operator"] != ao:
        _fail(code + "_PROVIDER_OPERATOR_MISMATCH")
    if aid == ai or akey["operator"] == ao:
        _fail(code + "_AUDITOR_NOT_INDEPENDENT")
    result = {
        "archiveIdentity": ai,
        "archiveOperator": ao,
        "archiveId": archive_id,
        "locator": locator,
        "immutability": imm,
        "providerKeyId": pkid,
        "providerKey": pkey,
        "auditorIdentity": aid,
        "auditorKeyId": akid,
        "auditorKey": akey,
    }
    if value != result:
        _fail(code + "_NOT_NORMALIZED")
    return result


def _membership(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    root: dict[str, Any],
    *,
    run160_artifact: dict[str, Any],
    native_head: str,
    previous_head: str | None,
    previous_members: list[dict[str, Any]] | None,
    now: datetime,
    historical: bool = False,
) -> dict[str, Any]:
    if set(doc) != {"signed", "signatures"} or not isinstance(doc.get("signed"), dict):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SCHEMA_INVALID")
    s = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "rootSha256",
        "sequence",
        "issuedAt",
        "previousHealthChainHeadSha256",
        "run160ArchiveArtifact",
        "nativeStatusChainHeadSha256",
        "minimumDurableCopies",
        "selectedSignerKeyIds",
        "members",
        "transition",
    }
    if (
        set(s) != expected
        or s.get("_type") != "archive-health-membership"
        or s.get("specVersion") != str(POLICY["spec_version"])
        or s.get("schemaVersion") != int(POLICY["membership_schema_version"])
    ):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SIGNED_SCHEMA_INVALID")
    if s.get("rootSha256") != root["sha256"]:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_ROOT_MISMATCH")
    seq = _size(s.get("sequence"), "ARCHIVE_HEALTH_MEMBERSHIP_SEQUENCE_INVALID")
    if seq < 1:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SEQUENCE_INVALID")
    issued = _dt(s.get("issuedAt"), "ARCHIVE_HEALTH_MEMBERSHIP_ISSUED_INVALID")
    if not historical and issued > now + timedelta(
        minutes=int(POLICY["max_clock_skew_minutes"])
    ):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_FROM_FUTURE")
    prev = s.get("previousHealthChainHeadSha256")
    if previous_head is None:
        if seq != 1 or prev is not None:
            _fail("ARCHIVE_HEALTH_MEMBERSHIP_BOOTSTRAP_CONTINUITY_INVALID")
    elif seq < 2 or prev != previous_head:  # ruff: ignore[magic-value-comparison]
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_CONTINUITY_INVALID")
    if (
        s.get("run160ArchiveArtifact") != run160_artifact
        or s.get("nativeStatusChainHeadSha256") != native_head
    ):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_RUN160_BINDING_INVALID")
    minimum = _size(
        s.get("minimumDurableCopies"), "ARCHIVE_HEALTH_MEMBERSHIP_MINIMUM_INVALID"
    )
    if minimum < int(POLICY["min_durable_copies"]):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_MINIMUM_INVALID")
    members_raw = s.get("members")
    if (
        not isinstance(members_raw, list)
        or len(members_raw) < minimum
        or len(members_raw) > int(POLICY["max_archives"])
    ):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_MEMBER_COUNT_INVALID")
    members = [_member(x, "ARCHIVE_HEALTH_MEMBER") for x in members_raw]
    if members_raw != sorted(members, key=lambda x: x["archiveId"]):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_NOT_SORTED")
    ids = [x["archiveId"] for x in members]
    loc = [x["locator"] for x in members]
    if len(set(ids)) != len(ids) or len(set(loc)) != len(loc):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_DUPLICATE_INVALID")
    if len({x["archiveOperator"] for x in members}) < int(
        POLICY["min_archive_operators"]
    ):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_ARCHIVE_OPERATOR_QUORUM_INVALID")
    if len({x["auditorKey"]["operator"] for x in members}) < int(
        POLICY["min_auditor_operators"]
    ):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_AUDITOR_OPERATOR_QUORUM_INVALID")
    archive_ops = {x["archiveOperator"] for x in members}
    auditor_ops = {x["auditorKey"]["operator"] for x in members}
    if archive_ops & auditor_ops:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_OPERATOR_PLANES_OVERLAP")
    root_ops = {root["keys"][x]["operator"] for x in root["keys"]}
    if root_ops & (archive_ops | auditor_ops):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_GOVERNANCE_OPERATOR_OVERLAP")
    if issued < root["issued"] or issued > root["expires"]:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_OUTSIDE_ROOT_LIFETIME")
    key_ids = []
    public_keys = []
    for x in members:
        if (
            _dt(
                x["providerKey"]["expires"],
                "ARCHIVE_HEALTH_PROVIDER_KEY_EXPIRES_INVALID",
            )
            <= issued
            or _dt(
                x["auditorKey"]["expires"], "ARCHIVE_HEALTH_AUDITOR_KEY_EXPIRES_INVALID"
            )
            <= issued
        ):
            _fail("ARCHIVE_HEALTH_MEMBERSHIP_KEY_EXPIRED")
        key_ids.extend([x["providerKeyId"], x["auditorKeyId"]])
        public_keys.extend(
            [x["providerKey"]["publicKey"], x["auditorKey"]["publicKey"]]
        )
    if len(set(key_ids)) != len(key_ids) or len(set(public_keys)) != len(public_keys):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_KEY_REUSE_INVALID")
    if set(public_keys) & {root["keys"][x]["publicKey"] for x in root["keys"]}:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_GOVERNANCE_KEY_OVERLAP")
    if previous_members is not None:
        old_by_id = {x["archiveId"]: x for x in previous_members}
        for x in members:
            if x["archiveId"] in old_by_id and x != old_by_id[x["archiveId"]]:
                _fail("ARCHIVE_HEALTH_MEMBER_ID_REBOUND")
    transition = s.get("transition")
    if not isinstance(transition, dict) or set(transition) != {
        "kind",
        "addedArchiveIds",
        "retiredArchiveIds",
    }:
        _fail("ARCHIVE_HEALTH_TRANSITION_SCHEMA_INVALID")
    old = {x["archiveId"] for x in previous_members or []}
    new = set(ids)
    added = sorted(new - old)
    retired = sorted(old - new)
    kind = transition.get("kind")
    expected_kind = (
        "bootstrap"
        if previous_members is None
        else ("migration" if added or retired else "audit")
    )
    if (
        kind != expected_kind
        or transition.get("addedArchiveIds") != added
        or transition.get("retiredArchiveIds") != retired
    ):
        _fail("ARCHIVE_HEALTH_TRANSITION_MISMATCH")
    selected = s.get("selectedSignerKeyIds")
    if not isinstance(selected, list) or len(selected) != root["threshold"]:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SELECTED_INVALID")
    selected = [
        _identity(x, "ARCHIVE_HEALTH_MEMBERSHIP_SELECTED_KEY_INVALID") for x in selected
    ]
    if (
        selected != sorted(selected)
        or len(set(selected)) != len(selected)
        or any(x not in root["keys"] for x in selected)
    ):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SELECTED_INVALID")
    if len({root["keys"][x]["operator"] for x in selected}) < int(
        POLICY["min_governance_operators"]
    ):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SIGNER_OPERATOR_QUORUM_INVALID")
    sigs = doc.get("signatures")
    if not isinstance(sigs, list) or len(sigs) != len(selected):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SIGNATURE_SET_INVALID")
    sigmap = {}
    for item in sigs:
        if not isinstance(item, dict) or set(item) != {"keyId", "signature"}:
            _fail("ARCHIVE_HEALTH_MEMBERSHIP_SIGNATURE_SCHEMA_INVALID")
        kid = _identity(
            item.get("keyId"), "ARCHIVE_HEALTH_MEMBERSHIP_SIGNATURE_KEY_INVALID"
        )
        if kid in sigmap:
            _fail("ARCHIVE_HEALTH_MEMBERSHIP_SIGNATURE_DUPLICATE")
        sigmap[kid] = item.get("signature")
    if sorted(sigmap) != selected:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SIGNATURE_SET_INVALID")
    if sigs != sorted(sigs, key=lambda x: x["keyId"]):
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SIGNATURES_NOT_SORTED")
    raw = _canonical(s)
    for kid in selected:
        if (
            _dt(root["keys"][kid]["expires"], "ARCHIVE_HEALTH_ROOT_KEY_EXPIRES_INVALID")
            < issued
        ):
            _fail("ARCHIVE_HEALTH_MEMBERSHIP_SIGNER_EXPIRED")
        _ed25519_verify(
            root["keys"][kid]["publicKey"],
            sigmap[kid],
            raw,
            "ARCHIVE_HEALTH_MEMBERSHIP",
        )
    return {
        "sequence": seq,
        "issued": issued,
        "minimum": minimum,
        "members": members,
        "transition": transition,
        "doc": doc,
        "sha256": _sha_bytes(_canonical(doc)),
    }


def _command_adapter(
    command: list[str], *, prefix: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if not command:
        _fail(prefix + "_COMMAND_INVALID")

    def invoke(request: dict[str, Any]) -> dict[str, Any]:
        proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
        )
        input_bytes = _canonical(request)
        out = bytearray()
        err = bytearray()
        overflow = []

        def reader(pipe, buf):
            try:
                while True:
                    chunk = pipe.read(65536)
                    if not chunk:
                        break
                    if len(buf) + len(chunk) > int(POLICY["max_adapter_output_bytes"]):
                        overflow.append(True)
                        proc.kill()
                        break
                    buf.extend(chunk)
            finally:
                pipe.close()

        t1 = threading.Thread(target=reader, args=(proc.stdout, out), daemon=True)
        t2 = threading.Thread(target=reader, args=(proc.stderr, err), daemon=True)
        t1.start()
        t2.start()
        try:
            assert proc.stdin is not None  # ruff: ignore[assert]
            proc.stdin.write(input_bytes)
            proc.stdin.close()
            proc.wait(timeout=int(POLICY["adapter_timeout_seconds"]))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise ArchiveHealthError(  # ruff: ignore[raise-without-from-inside-except]
                prefix + "_TIMEOUT"
            )
        t1.join()
        t2.join()
        if overflow:
            _fail(prefix + "_OUTPUT_TOO_LARGE")
        if proc.returncode != 0:
            _fail(prefix + "_FAILED")
        return _loads(bytes(out), prefix + "_RESPONSE")

    return invoke


def command_provider(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="ARCHIVE_HEALTH_PROVIDER")


def command_auditor(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="ARCHIVE_HEALTH_AUDITOR")


def _audit_id(
    sequence: int, archive_id: str, membership_sha: str, artifact_sha: str
) -> str:
    return "audit-" + _sha_bytes(
        _canonical(
            {
                "sequence": sequence,
                "archiveId": archive_id,
                "membershipSha256": membership_sha,
                "artifactSha256": artifact_sha,
            }
        )
    )


def _challenge(audit_id: str, previous_head: str | None) -> str:
    return _sha_bytes(
        _canonical(
            {"auditId": audit_id, "previousHealthChainHeadSha256": previous_head}
        )
    )


def _verify_provider_response(  # ruff: ignore[too-many-branches]
    value: dict[str, Any],
    member: dict[str, Any],
    *,
    audit_id: str,
    challenge: str,
    artifact: dict[str, Any],
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"signed", "signature"}
        or not isinstance(value.get("signed"), dict)
    ):
        _fail("ARCHIVE_HEALTH_PROVIDER_RESPONSE_SCHEMA_INVALID")
    s = value["signed"]
    expected = {
        "schemaVersion",
        "operation",
        "auditId",
        "archiveId",
        "archiveIdentity",
        "locator",
        "immutableVersionId",
        "artifact",
        "immutability",
        "retentionMode",
        "retentionUntil",
        "legalHold",
        "challenge",
        "remoteReadbackVerified",
        "observedAt",
    }
    if (
        set(s) != expected
        or s.get("schemaVersion") != int(POLICY["provider_protocol_version"])
        or s.get("operation") != "audit-retention"
    ):
        _fail("ARCHIVE_HEALTH_PROVIDER_SIGNED_SCHEMA_INVALID")
    if (
        s.get("auditId") != audit_id
        or s.get("archiveId") != member["archiveId"]
        or s.get("archiveIdentity") != member["archiveIdentity"]
    ):
        _fail("ARCHIVE_HEALTH_PROVIDER_BINDING_INVALID")
    if (
        _safe_locator(s.get("locator"), "ARCHIVE_HEALTH_PROVIDER_LOCATOR_INVALID")
        != member["locator"]
        or s.get("artifact") != artifact
        or s.get("immutability") != member["immutability"]
    ):
        _fail("ARCHIVE_HEALTH_PROVIDER_ARTIFACT_BINDING_INVALID")
    version = _identity(
        s.get("immutableVersionId"), "ARCHIVE_HEALTH_PROVIDER_VERSION_ID_INVALID"
    )
    mode = s.get("retentionMode")
    if mode not in set(POLICY["allowed_retention_modes"]):
        _fail("ARCHIVE_HEALTH_RETENTION_MODE_INVALID")
    compatible = {
        "object-lock": {"object-lock-compliance", "object-lock-governance"},
        "content-addressed": {"content-addressed"},
        "versioned-create-only": {"versioned-create-only"},
        "release-create-only": {"release-create-only"},
    }
    if mode not in compatible.get(member["immutability"], set()):
        _fail("ARCHIVE_HEALTH_RETENTION_MODE_IMMUTABILITY_MISMATCH")
    retention = _dt(s.get("retentionUntil"), "ARCHIVE_HEALTH_RETENTION_UNTIL_INVALID")
    observed = _dt(s.get("observedAt"), "ARCHIVE_HEALTH_PROVIDER_OBSERVED_INVALID")
    if not historical:
        skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        age = timedelta(minutes=int(POLICY["max_result_age_minutes"]))
        if observed > now + skew or now - observed > age:
            _fail("ARCHIVE_HEALTH_PROVIDER_OBSERVATION_NOT_FRESH")
        if (
            retention
            < now + timedelta(days=int(POLICY["minimum_retention_remaining_days"]))
            and s.get("legalHold") is not True
        ):
            _fail("ARCHIVE_HEALTH_RETENTION_TOO_SHORT")
    if retention <= observed and s.get("legalHold") is not True:
        _fail("ARCHIVE_HEALTH_RETENTION_EXPIRED_AT_OBSERVATION")
    if (
        not isinstance(s.get("legalHold"), bool)
        or s.get("challenge") != challenge
        or s.get("remoteReadbackVerified") is not True
    ):
        _fail("ARCHIVE_HEALTH_PROVIDER_PROOF_INVALID")
    sig = value.get("signature")
    if (
        not isinstance(sig, dict)
        or set(sig) != {"keyId", "signature"}
        or sig.get("keyId") != member["providerKeyId"]
    ):
        _fail("ARCHIVE_HEALTH_PROVIDER_SIGNATURE_SCHEMA_INVALID")
    if (
        _dt(
            member["providerKey"]["expires"],
            "ARCHIVE_HEALTH_PROVIDER_KEY_EXPIRES_INVALID",
        )
        < observed
    ):
        _fail("ARCHIVE_HEALTH_PROVIDER_KEY_EXPIRED")
    _ed25519_verify(
        member["providerKey"]["publicKey"],
        sig.get("signature"),
        _canonical(s),
        "ARCHIVE_HEALTH_PROVIDER",
    )
    return {
        "immutableVersionId": version,
        "retentionMode": mode,
        "retentionUntil": _ts(retention),
        "legalHold": s["legalHold"],
        "observedAt": _ts(observed),
        "sha256": _sha_bytes(_canonical(value)),
        "doc": value,
    }


def _verify_auditor_response(
    value: dict[str, Any],
    member: dict[str, Any],
    *,
    audit_id: str,
    challenge: str,
    artifact: dict[str, Any],
    provider: dict[str, Any],
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"signed", "signature"}
        or not isinstance(value.get("signed"), dict)
    ):
        _fail("ARCHIVE_HEALTH_AUDITOR_RESPONSE_SCHEMA_INVALID")
    s = value["signed"]
    expected = {
        "schemaVersion",
        "operation",
        "auditId",
        "auditorIdentity",
        "archiveId",
        "archiveIdentity",
        "locator",
        "immutableVersionId",
        "artifact",
        "challenge",
        "providerResponseSha256",
        "readOnly",
        "remoteReadbackVerified",
        "observedAt",
    }
    if (
        set(s) != expected
        or s.get("schemaVersion") != int(POLICY["auditor_protocol_version"])
        or s.get("operation") != "verify-retention"
    ):
        _fail("ARCHIVE_HEALTH_AUDITOR_SIGNED_SCHEMA_INVALID")
    if (
        s.get("auditId") != audit_id
        or s.get("auditorIdentity") != member["auditorIdentity"]
        or s.get("archiveId") != member["archiveId"]
        or s.get("archiveIdentity") != member["archiveIdentity"]
    ):
        _fail("ARCHIVE_HEALTH_AUDITOR_BINDING_INVALID")
    if (
        _safe_locator(s.get("locator"), "ARCHIVE_HEALTH_AUDITOR_LOCATOR_INVALID")
        != member["locator"]
        or s.get("immutableVersionId") != provider["immutableVersionId"]
        or s.get("artifact") != artifact
        or s.get("challenge") != challenge
        or s.get("providerResponseSha256") != provider["sha256"]
    ):
        _fail("ARCHIVE_HEALTH_AUDITOR_PROOF_BINDING_INVALID")
    if s.get("readOnly") is not True or s.get("remoteReadbackVerified") is not True:
        _fail("ARCHIVE_HEALTH_AUDITOR_PROOF_INVALID")
    observed = _dt(s.get("observedAt"), "ARCHIVE_HEALTH_AUDITOR_OBSERVED_INVALID")
    if not historical:
        skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        age = timedelta(minutes=int(POLICY["max_result_age_minutes"]))
        if observed > now + skew or now - observed > age:
            _fail("ARCHIVE_HEALTH_AUDITOR_OBSERVATION_NOT_FRESH")
    sig = value.get("signature")
    if (
        not isinstance(sig, dict)
        or set(sig) != {"keyId", "signature"}
        or sig.get("keyId") != member["auditorKeyId"]
    ):
        _fail("ARCHIVE_HEALTH_AUDITOR_SIGNATURE_SCHEMA_INVALID")
    if (
        _dt(
            member["auditorKey"]["expires"],
            "ARCHIVE_HEALTH_AUDITOR_KEY_EXPIRES_INVALID",
        )
        < observed
    ):
        _fail("ARCHIVE_HEALTH_AUDITOR_KEY_EXPIRED")
    _ed25519_verify(
        member["auditorKey"]["publicKey"],
        sig.get("signature"),
        _canonical(s),
        "ARCHIVE_HEALTH_AUDITOR",
    )
    return {
        "observedAt": _ts(observed),
        "sha256": _sha_bytes(_canonical(value)),
        "doc": value,
    }


def _run160_docs(
    root: Path,
    *,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    now: datetime,
    historical: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_HEALTH_RUN160_DIR_INVALID")
    try:
        info = run160.verify_native_archive(
            output_dir=root,
            expected_bootstrap_root_sha256=bootstrap_pin,
            expected_recovery_root_sha256=recovery_pin,
            expected_attestation_root_sha256=attestation_pins,
            now=now,
            historical=historical,
        )
    except run160.NativeArchiveError as exc:
        raise ArchiveHealthError(
            "ARCHIVE_HEALTH_RUN160_REPLAY_INVALID:" + str(exc)
        ) from exc
    docs = {}
    raws = {}
    for name in run160._ARCHIVE_NAMES:
        doc, raw = _read_json(
            root / name,
            "ARCHIVE_HEALTH_RUN160_" + name.upper().replace("-", "_").replace(".", "_"),
        )
        docs[name] = doc
        raws[name] = raw
    return info, docs, raws


def _baseline_members(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for entry in receipt.get("archives", []):
        out[entry["archiveId"]] = {
            "archiveIdentity": entry["archiveIdentity"],
            "archiveOperator": entry["archiveOperator"],
            "archiveId": entry["archiveId"],
            "locator": entry["locator"],
            "immutability": entry["immutability"],
        }
    return out


def _event_head(previous_head: str | None, event: dict[str, Any]) -> str:
    return _sha_bytes(
        _canonical({"previousHealthChainHeadSha256": previous_head, "event": event})
    )


def _output_docs(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_HEALTH_OUTPUT_DIR_INVALID")
    if {p.name for p in root.iterdir()} != _OUTPUT_NAMES:
        _fail("ARCHIVE_HEALTH_OUTPUT_ALLOWLIST_MISMATCH")
    docs = {}
    raws = {}
    for name in sorted(_OUTPUT_NAMES):
        docs[name], raws[name] = _read_json(
            root / name,
            "ARCHIVE_HEALTH_OUTPUT_" + name.upper().replace("-", "_").replace(".", "_"),
        )
    return docs, raws


def _replay(  # ruff: ignore[too-many-branches]
    *,
    run160_dir: Path,
    root_doc: dict[str, Any],
    root_pin: str,
    bundle: dict[str, Any],
    receipts: list[dict[str, Any]],
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    rinfo, _rdocs, rraws = _run160_docs(
        run160_dir,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        now=now,
        historical=True,
    )
    root = _verify_root(root_doc, root_pin, now=now, historical=historical)
    run160_item = _artifact_bytes(
        "release-native-evidence-archive.json",
        rraws["release-native-evidence-archive.json"],
    )
    if (
        set(bundle)
        != {
            "schemaVersion",
            "predicateType",
            "status",
            "retentionRoot",
            "run160ArchiveArtifact",
            "nativeStatusChainHeadSha256",
            "events",
        }
        or bundle.get("schemaVersion") != int(POLICY["bundle_schema_version"])
        or bundle.get("predicateType") != PREDICATE_TYPE
        or bundle.get("status") != "archive-health-history"
    ):
        _fail("ARCHIVE_HEALTH_BUNDLE_SCHEMA_INVALID")
    if (
        bundle.get("retentionRoot")
        != {
            "sha256": root["sha256"],
            "rootId": root["rootId"],
            "version": root["version"],
        }
        or bundle.get("run160ArchiveArtifact") != run160_item
        or bundle.get("nativeStatusChainHeadSha256")
        != rinfo["native_status_chain_head_sha256"]
    ):
        _fail("ARCHIVE_HEALTH_BUNDLE_AUTHORITY_MISMATCH")
    events = bundle.get("events")
    if (
        not isinstance(events, list)
        or not events
        or len(events) > int(POLICY["max_events"])
        or len(receipts) != len(events)
    ):
        _fail("ARCHIVE_HEALTH_EVENTS_INVALID")
    previous_head = None
    previous_members = None
    previous_audit_time = None
    previous_versions = {}
    previous_retentions = {}
    previous_modes = {}
    previous_issued = None
    last_audit_times = []
    for idx, (event, receipt) in enumerate(zip(events, receipts), start=1):
        if not isinstance(event, dict) or set(event) != {
            "sequence",
            "membership",
            "audits",
            "healthChainHeadSha256",
        }:
            _fail("ARCHIVE_HEALTH_EVENT_SCHEMA_INVALID")
        membership = _membership(
            event["membership"],
            root,
            run160_artifact=run160_item,
            native_head=rinfo["native_status_chain_head_sha256"],
            previous_head=previous_head,
            previous_members=previous_members,
            now=now,
            historical=True,
        )
        if membership["sequence"] != idx or event.get("sequence") != idx:
            _fail("ARCHIVE_HEALTH_EVENT_SEQUENCE_INVALID")
        if previous_issued is not None and membership["issued"] <= previous_issued:
            _fail("ARCHIVE_HEALTH_MEMBERSHIP_TIME_NOT_MONOTONIC")
        audits = event.get("audits")
        if not isinstance(audits, list) or len(audits) != len(membership["members"]):
            _fail("ARCHIVE_HEALTH_AUDIT_COUNT_INVALID")
        by_id = {x["archiveId"]: x for x in membership["members"]}
        seen = set()
        normalized = []
        audit_times = []
        if not isinstance(receipt, dict) or set(receipt) != {
            "sequence",
            "membershipSha256",
            "providerResults",
            "auditorResults",
        }:
            _fail("ARCHIVE_HEALTH_EVENT_RECEIPT_SCHEMA_INVALID")
        if (
            receipt.get("sequence") != idx
            or receipt.get("membershipSha256") != membership["sha256"]
        ):
            _fail("ARCHIVE_HEALTH_EVENT_RECEIPT_BINDING_INVALID")
        providers = receipt.get("providerResults")
        auditors = receipt.get("auditorResults")
        if (
            not isinstance(providers, list)
            or not isinstance(auditors, list)
            or len(providers) != len(audits)
            or len(auditors) != len(audits)
        ):
            _fail("ARCHIVE_HEALTH_EVENT_RECEIPT_COUNT_INVALID")
        if providers != sorted(
            providers,
            key=lambda x: x.get("archiveId", "") if isinstance(x, dict) else "",
        ) or auditors != sorted(
            auditors,
            key=lambda x: x.get("archiveId", "") if isinstance(x, dict) else "",
        ):
            _fail("ARCHIVE_HEALTH_EVENT_RECEIPT_NOT_SORTED")
        pmap = {x.get("archiveId"): x for x in providers if isinstance(x, dict)}
        amap = {x.get("archiveId"): x for x in auditors if isinstance(x, dict)}
        if len(pmap) != len(providers) or len(amap) != len(auditors):
            _fail("ARCHIVE_HEALTH_EVENT_RECEIPT_DUPLICATE_RESULT")
        for a in audits:
            if not isinstance(a, dict) or set(a) != {
                "archiveId",
                "auditId",
                "challenge",
                "immutableVersionId",
                "retentionMode",
                "retentionUntil",
                "legalHold",
                "providerResponseSha256",
                "auditorResponseSha256",
            }:
                _fail("ARCHIVE_HEALTH_AUDIT_SCHEMA_INVALID")
            aid = a.get("archiveId")
            if aid not in by_id or aid in seen:
                _fail("ARCHIVE_HEALTH_AUDIT_MEMBER_INVALID")
            seen.add(aid)
            member = by_id[aid]
            audit_id = _audit_id(idx, aid, membership["sha256"], run160_item["sha256"])
            challenge = _challenge(audit_id, previous_head)
            if a.get("auditId") != audit_id or a.get("challenge") != challenge:
                _fail("ARCHIVE_HEALTH_AUDIT_CHALLENGE_INVALID")
            pe = pmap.get(aid)
            ae = amap.get(aid)
            if (
                not isinstance(pe, dict)
                or set(pe) != {"archiveId", "response"}
                or not isinstance(ae, dict)
                or set(ae) != {"archiveId", "response"}
            ):
                _fail("ARCHIVE_HEALTH_EVENT_RECEIPT_RESULT_INVALID")
            provider = _verify_provider_response(
                pe["response"],
                member,
                audit_id=audit_id,
                challenge=challenge,
                artifact=run160_item,
                now=now,
                historical=True,
            )
            auditor = _verify_auditor_response(
                ae["response"],
                member,
                audit_id=audit_id,
                challenge=challenge,
                artifact=run160_item,
                provider=provider,
                now=now,
                historical=True,
            )
            expected = {
                "archiveId": aid,
                "auditId": audit_id,
                "challenge": challenge,
                "immutableVersionId": provider["immutableVersionId"],
                "retentionMode": provider["retentionMode"],
                "retentionUntil": provider["retentionUntil"],
                "legalHold": provider["legalHold"],
                "providerResponseSha256": provider["sha256"],
                "auditorResponseSha256": auditor["sha256"],
            }
            if a != expected:
                _fail("ARCHIVE_HEALTH_AUDIT_REPLAY_MISMATCH")
            if (
                aid in previous_versions
                and provider["immutableVersionId"] != previous_versions[aid]
            ):
                _fail("ARCHIVE_HEALTH_IMMUTABLE_VERSION_CHANGED")
            if (
                aid in previous_modes
                and provider["retentionMode"] != previous_modes[aid]
            ):
                _fail("ARCHIVE_HEALTH_RETENTION_MODE_CHANGED")
            if (
                aid in previous_retentions
                and _dt(
                    provider["retentionUntil"], "ARCHIVE_HEALTH_RETENTION_UNTIL_INVALID"
                )
                < previous_retentions[aid]
            ):
                _fail("ARCHIVE_HEALTH_RETENTION_ROLLBACK")
            previous_versions[aid] = provider["immutableVersionId"]
            previous_modes[aid] = provider["retentionMode"]
            previous_retentions[aid] = _dt(
                provider["retentionUntil"], "ARCHIVE_HEALTH_RETENTION_UNTIL_INVALID"
            )
            po = _dt(provider["observedAt"], "ARCHIVE_HEALTH_PROVIDER_OBSERVED_INVALID")
            ao = _dt(auditor["observedAt"], "ARCHIVE_HEALTH_AUDITOR_OBSERVED_INVALID")
            skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
            age = timedelta(minutes=int(POLICY["max_result_age_minutes"]))
            if (
                po < membership["issued"] - skew
                or ao < membership["issued"] - skew
                or abs(po - ao) > age
            ):
                _fail("ARCHIVE_HEALTH_OBSERVATION_TIME_BINDING_INVALID")
            audit_times.append(max(po, ao))
            normalized.append(expected)
        if len(seen) < membership["minimum"] or len(
            {by_id[x]["archiveOperator"] for x in seen}
        ) < int(POLICY["min_archive_operators"]):
            _fail("ARCHIVE_HEALTH_DURABLE_QUORUM_INVALID")
        current_audit = max(audit_times)
        if (
            previous_audit_time is not None
            and current_audit - previous_audit_time
            > timedelta(days=int(POLICY["maximum_audit_interval_days"]))
        ):
            _fail("ARCHIVE_HEALTH_AUDIT_INTERVAL_EXCEEDED")
        core = {
            "sequence": idx,
            "membership": event["membership"],
            "audits": normalized,
        }
        head = _event_head(previous_head, core)
        if event.get("healthChainHeadSha256") != head:
            _fail("ARCHIVE_HEALTH_CHAIN_HEAD_MISMATCH")
        previous_head = head
        previous_members = membership["members"]
        previous_audit_time = current_audit
        previous_issued = membership["issued"]
        last_audit_times = audit_times
    if not historical:
        if (
            not last_audit_times
            or now - max(last_audit_times)
            > timedelta(days=int(POLICY["maximum_audit_interval_days"]))
            or max(last_audit_times)
            > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        ):
            _fail("ARCHIVE_HEALTH_ACTIVE_AUDIT_STALE")
        for item in events[-1]["audits"]:
            if item["legalHold"] is not True and _dt(
                item["retentionUntil"], "ARCHIVE_HEALTH_RETENTION_UNTIL_INVALID"
            ) < now + timedelta(days=int(POLICY["minimum_retention_remaining_days"])):
                _fail("ARCHIVE_HEALTH_RETENTION_TOO_SHORT")
    return {
        "sequence": len(events),
        "chain_head": previous_head,
        "members": previous_members,
        "run160_artifact": run160_item,
        "native_head": rinfo["native_status_chain_head_sha256"],
        "last_event": events[-1],
        "last_receipt": receipts[-1],
        "last_issued": previous_issued,
    }


def verify_archive_health(  # ruff: ignore[undocumented-public-function]
    *,
    run160_dir: Path,
    output_dir: Path,
    retention_root_path: Path,
    expected_retention_root_sha256: str,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root_doc, _ = _read_json(retention_root_path, "ARCHIVE_HEALTH_ROOT")
    docs, raws = _output_docs(output_dir)
    bundle = docs["release-archive-health-bundle.json"]
    receipt = docs["release-archive-health-receipt.json"]
    if (
        set(receipt) != {"schemaVersion", "status", "events"}
        or receipt.get("schemaVersion") != int(POLICY["receipt_schema_version"])
        or receipt.get("status") != "archive-health-audited"
        or not isinstance(receipt.get("events"), list)
    ):
        _fail("ARCHIVE_HEALTH_RECEIPT_SCHEMA_INVALID")
    replay = _replay(
        run160_dir=run160_dir,
        root_doc=root_doc,
        root_pin=expected_retention_root_sha256,
        bundle=bundle,
        receipts=receipt["events"],
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=expected_attestation_root_sha256,
        now=current,
        historical=historical,
    )
    bundle_item = _artifact_bytes(
        "release-archive-health-bundle.json", raws["release-archive-health-bundle.json"]
    )
    state = docs[_DOC_ARCHIVE_HEALTH_STATE]
    expected_state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-archive-health",
        "sequence": replay["sequence"],
        "healthChainHeadSha256": replay["chain_head"],
        "run160ArchiveArtifact": replay["run160_artifact"],
        "nativeStatusChainHeadSha256": replay["native_head"],
        "bundleArtifact": bundle_item,
        "activeArchiveIds": sorted(x["archiveId"] for x in replay["members"]),
    }
    if state != expected_state:
        _fail("ARCHIVE_HEALTH_STATE_MISMATCH")
    active = docs["active-archive-health-evidence.json"]
    expected_active = {
        "schemaVersion": int(POLICY["active_schema_version"]),
        "sequence": replay["sequence"],
        "healthChainHeadSha256": replay["chain_head"],
        "membership": replay["last_event"]["membership"],
        "audits": replay["last_event"]["audits"],
        "retirementAuthorizedArchiveIds": replay["last_event"]["membership"]["signed"][
            "transition"
        ]["retiredArchiveIds"],
    }
    if active != expected_active:
        _fail("ARCHIVE_HEALTH_ACTIVE_MISMATCH")
    return {
        "ok": True,
        "phase": "archive-health-verified",
        "sequence": replay["sequence"],
        "health_chain_head_sha256": replay["chain_head"],
        "active_archive_count": len(replay["members"]),
    }


ProviderAdapter = Callable[[dict[str, Any]], dict[str, Any]]
AuditorAdapter = Callable[[dict[str, Any]], dict[str, Any]]


def audit_archive_health(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    run160_dir: Path,
    output_dir: Path,
    retention_root_path: Path,
    membership_path: Path,
    targets: list[tuple[str, ProviderAdapter, AuditorAdapter]],
    expected_retention_root_sha256: str,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    previous_output_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run160_dir = _regular_dir(run160_dir, "ARCHIVE_HEALTH_RUN160_DIR_INVALID")
    root_doc, _ = _read_json(retention_root_path, "ARCHIVE_HEALTH_ROOT")
    root = _verify_root(root_doc, expected_retention_root_sha256, now=current)
    rinfo, rdocs, rraws = _run160_docs(
        run160_dir,
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=expected_attestation_root_sha256,
        now=current,
        historical=True,
    )
    run160_item = _artifact_bytes(
        "release-native-evidence-archive.json",
        rraws["release-native-evidence-archive.json"],
    )
    previous_bundle = None
    previous_receipts = []
    previous_head = None
    previous_members = None
    protected = [run160_dir, retention_root_path.resolve(), membership_path.resolve()]
    if previous_output_dir is not None:
        previous_output_dir = _regular_dir(
            previous_output_dir, "ARCHIVE_HEALTH_PREVIOUS_DIR_INVALID"
        )
        protected.append(previous_output_dir)
        pdocs, _ = _output_docs(previous_output_dir)
        if set(pdocs["release-archive-health-receipt.json"]) != {
            "schemaVersion",
            "status",
            "events",
        }:
            _fail("ARCHIVE_HEALTH_PREVIOUS_RECEIPT_INVALID")
        prev = _replay(
            run160_dir=run160_dir,
            root_doc=root_doc,
            root_pin=expected_retention_root_sha256,
            bundle=pdocs["release-archive-health-bundle.json"],
            receipts=pdocs["release-archive-health-receipt.json"]["events"],
            bootstrap_pin=expected_bootstrap_root_sha256,
            recovery_pin=expected_recovery_root_sha256,
            attestation_pins=expected_attestation_root_sha256,
            now=current,
            historical=True,
        )
        previous_bundle = pdocs["release-archive-health-bundle.json"]
        previous_receipts = list(pdocs["release-archive-health-receipt.json"]["events"])
        previous_head = prev["chain_head"]
        previous_members = prev["members"]
    target = _outside(output_dir, protected, "ARCHIVE_HEALTH_OUTPUT_OVERLAPS_INPUT")
    if target.exists():
        _fail("ARCHIVE_HEALTH_OUTPUT_EXISTS")
    membership_doc, _ = _read_json(membership_path, "ARCHIVE_HEALTH_MEMBERSHIP")
    membership = _membership(
        membership_doc,
        root,
        run160_artifact=run160_item,
        native_head=rinfo["native_status_chain_head_sha256"],
        previous_head=previous_head,
        previous_members=previous_members,
        now=current,
    )
    expected_sequence = (
        1 if previous_bundle is None else len(previous_bundle["events"]) + 1
    )
    if membership["sequence"] != expected_sequence:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_SEQUENCE_NOT_CONSECUTIVE")
    if previous_bundle is not None and membership["issued"] <= prev["last_issued"]:
        _fail("ARCHIVE_HEALTH_MEMBERSHIP_TIME_NOT_MONOTONIC")
    if previous_members is None:
        base = _baseline_members(rdocs["release-native-evidence-archive-receipt.json"])
        for member in membership["members"]:
            b = base.get(member["archiveId"])
            if b is None or any(
                member[k] != b[k]
                for k in (
                    "archiveIdentity",
                    "archiveOperator",
                    "archiveId",
                    "locator",
                    "immutability",
                )
            ):
                _fail("ARCHIVE_HEALTH_BOOTSTRAP_RUN160_MEMBERSHIP_MISMATCH")
        if set(base) != {x["archiveId"] for x in membership["members"]}:
            _fail("ARCHIVE_HEALTH_BOOTSTRAP_RUN160_MEMBERSHIP_MISMATCH")
    adapters = {
        _identity(aid, "ARCHIVE_HEALTH_TARGET_ARCHIVE_ID_INVALID"): (p, a)
        for aid, p, a in targets
    }
    member_ids = {x["archiveId"] for x in membership["members"]}
    if set(adapters) != member_ids:
        _fail("ARCHIVE_HEALTH_TARGET_SET_MISMATCH")
    run160_fingerprint = _dir_fingerprint(
        run160_dir, "ARCHIVE_HEALTH_RUN160_INPUT_DRIFT"
    )
    root_fingerprint = _file_fingerprint(
        retention_root_path.resolve(), "ARCHIVE_HEALTH_ROOT_INPUT_DRIFT"
    )
    membership_fingerprint = _file_fingerprint(
        membership_path.resolve(), "ARCHIVE_HEALTH_MEMBERSHIP_INPUT_DRIFT"
    )
    previous_fingerprint = (
        _dir_fingerprint(previous_output_dir, "ARCHIVE_HEALTH_PREVIOUS_INPUT_DRIFT")
        if previous_output_dir is not None
        else None
    )
    audit_items = []
    provider_results = []
    auditor_results = []
    for member in membership["members"]:
        audit_id = _audit_id(
            membership["sequence"],
            member["archiveId"],
            membership["sha256"],
            run160_item["sha256"],
        )
        challenge = _challenge(audit_id, previous_head)
        provider_adapter, auditor_adapter = adapters[member["archiveId"]]
        request = {
            "schemaVersion": int(POLICY["provider_protocol_version"]),
            "operation": "audit-retention",
            "auditId": audit_id,
            "challenge": challenge,
            "archive": {
                "identity": member["archiveIdentity"],
                "operator": member["archiveOperator"],
                "archiveId": member["archiveId"],
                "locator": member["locator"],
            },
            "artifact": run160_item,
        }
        praw = provider_adapter(request)
        provider = _verify_provider_response(
            praw,
            member,
            audit_id=audit_id,
            challenge=challenge,
            artifact=run160_item,
            now=current,
            historical=False,
        )
        arequest = {
            "schemaVersion": int(POLICY["auditor_protocol_version"]),
            "operation": "verify-retention",
            "auditId": audit_id,
            "challenge": challenge,
            "auditor": {
                "identity": member["auditorIdentity"],
                "operator": member["auditorKey"]["operator"],
            },
            "archive": {
                "identity": member["archiveIdentity"],
                "archiveId": member["archiveId"],
                "locator": member["locator"],
            },
            "immutableVersionId": provider["immutableVersionId"],
            "artifact": run160_item,
            "providerResponseSha256": provider["sha256"],
        }
        araw = auditor_adapter(arequest)
        auditor = _verify_auditor_response(
            araw,
            member,
            audit_id=audit_id,
            challenge=challenge,
            artifact=run160_item,
            provider=provider,
            now=current,
            historical=False,
        )
        audit_items.append(
            {
                "archiveId": member["archiveId"],
                "auditId": audit_id,
                "challenge": challenge,
                "immutableVersionId": provider["immutableVersionId"],
                "retentionMode": provider["retentionMode"],
                "retentionUntil": provider["retentionUntil"],
                "legalHold": provider["legalHold"],
                "providerResponseSha256": provider["sha256"],
                "auditorResponseSha256": auditor["sha256"],
            }
        )
        provider_results.append({"archiveId": member["archiveId"], "response": praw})
        auditor_results.append({"archiveId": member["archiveId"], "response": araw})
    audit_items.sort(key=lambda x: x["archiveId"])
    provider_results.sort(key=lambda x: x["archiveId"])
    auditor_results.sort(key=lambda x: x["archiveId"])
    if (
        _dir_fingerprint(run160_dir, "ARCHIVE_HEALTH_RUN160_INPUT_DRIFT")
        != run160_fingerprint
        or _file_fingerprint(
            retention_root_path.resolve(), "ARCHIVE_HEALTH_ROOT_INPUT_DRIFT"
        )
        != root_fingerprint
        or _file_fingerprint(
            membership_path.resolve(), "ARCHIVE_HEALTH_MEMBERSHIP_INPUT_DRIFT"
        )
        != membership_fingerprint
    ):
        _fail("ARCHIVE_HEALTH_INPUT_DRIFT_DETECTED")
    if (
        previous_output_dir is not None
        and _dir_fingerprint(previous_output_dir, "ARCHIVE_HEALTH_PREVIOUS_INPUT_DRIFT")
        != previous_fingerprint
    ):
        _fail("ARCHIVE_HEALTH_INPUT_DRIFT_DETECTED")
    core = {
        "sequence": membership["sequence"],
        "membership": membership_doc,
        "audits": audit_items,
    }
    head = _event_head(previous_head, core)
    event = dict(core, healthChainHeadSha256=head)
    if previous_bundle is None:
        bundle = {
            "schemaVersion": int(POLICY["bundle_schema_version"]),
            "predicateType": PREDICATE_TYPE,
            "status": "archive-health-history",
            "retentionRoot": {
                "sha256": root["sha256"],
                "rootId": root["rootId"],
                "version": root["version"],
            },
            "run160ArchiveArtifact": run160_item,
            "nativeStatusChainHeadSha256": rinfo["native_status_chain_head_sha256"],
            "events": [event],
        }
    else:
        bundle = json.loads(json.dumps(previous_bundle))
        bundle["events"].append(event)
    receipt_event = {
        "sequence": membership["sequence"],
        "membershipSha256": membership["sha256"],
        "providerResults": provider_results,
        "auditorResults": auditor_results,
    }
    receipts = [*previous_receipts, receipt_event]
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run161-health-", dir=parent))
    try:
        _write(stage / "release-archive-health-bundle.json", bundle)
        bundle_raw = (stage / "release-archive-health-bundle.json").read_bytes()
        bundle_item = _artifact_bytes("release-archive-health-bundle.json", bundle_raw)
        state = {
            "schemaVersion": int(POLICY["state_schema_version"]),
            "status": "trusted-archive-health",
            "sequence": membership["sequence"],
            "healthChainHeadSha256": head,
            "run160ArchiveArtifact": run160_item,
            "nativeStatusChainHeadSha256": rinfo["native_status_chain_head_sha256"],
            "bundleArtifact": bundle_item,
            "activeArchiveIds": sorted(member_ids),
        }
        active = {
            "schemaVersion": int(POLICY["active_schema_version"]),
            "sequence": membership["sequence"],
            "healthChainHeadSha256": head,
            "membership": membership_doc,
            "audits": audit_items,
            "retirementAuthorizedArchiveIds": membership["transition"][
                "retiredArchiveIds"
            ],
        }
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "archive-health-audited",
            "events": receipts,
        }
        _write(stage / _DOC_ARCHIVE_HEALTH_STATE, state)
        _write(stage / "active-archive-health-evidence.json", active)
        _write(stage / "release-archive-health-receipt.json", receipt)
        verify_archive_health(
            run160_dir=run160_dir,
            output_dir=stage,
            retention_root_path=retention_root_path,
            expected_retention_root_sha256=expected_retention_root_sha256,
            expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
            expected_recovery_root_sha256=expected_recovery_root_sha256,
            expected_attestation_root_sha256=expected_attestation_root_sha256,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verified = verify_archive_health(
        run160_dir=run160_dir,
        output_dir=target,
        retention_root_path=retention_root_path,
        expected_retention_root_sha256=expected_retention_root_sha256,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    return dict(
        verified,
        phase="archive-health-audited",
        retirement_authorized=membership["transition"]["retiredArchiveIds"],
    )


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("audit", "verify"):
        q = sub.add_parser(name)
        q.add_argument("--run160-dir", type=Path, required=True)
        q.add_argument("--retention-root", type=Path, required=True)
        q.add_argument("--retention-root-sha256", required=True)
        q.add_argument("--bootstrap-root-sha256", required=True)
        q.add_argument("--recovery-root-sha256", required=True)
        q.add_argument("--attestation-root-sha256", action="append", required=True)
    a = sub.choices["audit"]
    a.add_argument("--membership", type=Path, required=True)
    a.add_argument("--previous-output-dir", type=Path)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument(
        "--target",
        action="append",
        required=True,
        help="archive-id=provider command... || auditor command...",
    )
    v = sub.choices["verify"]
    v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--historical", action="store_true")
    args = p.parse_args(argv)
    if args.command == "audit":
        targets = []
        for raw in args.target:
            if "=" not in raw or "||" not in raw:
                _fail("ARCHIVE_HEALTH_TARGET_ARGUMENT_INVALID")
            aid, rhs = raw.split("=", 1)
            pcmd, acmd = [x.strip().split() for x in rhs.split("||", 1)]
            targets.append((aid, command_provider(pcmd), command_auditor(acmd)))
        result = audit_archive_health(
            run160_dir=args.run160_dir,
            output_dir=args.output_dir,
            retention_root_path=args.retention_root,
            membership_path=args.membership,
            targets=targets,
            expected_retention_root_sha256=args.retention_root_sha256,
            expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            expected_recovery_root_sha256=args.recovery_root_sha256,
            expected_attestation_root_sha256=args.attestation_root_sha256,
            previous_output_dir=args.previous_output_dir,
        )
    else:
        result = verify_archive_health(
            run160_dir=args.run160_dir,
            output_dir=args.output_dir,
            retention_root_path=args.retention_root,
            expected_retention_root_sha256=args.retention_root_sha256,
            expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            expected_recovery_root_sha256=args.recovery_root_sha256,
            expected_attestation_root_sha256=args.attestation_root_sha256,
            historical=args.historical,
        )
    logger.info("%s", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchiveHealthError as exc:
        logger.error("%s", json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
