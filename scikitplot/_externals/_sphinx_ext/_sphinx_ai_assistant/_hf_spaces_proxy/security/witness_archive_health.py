"""
Run 162: witness archive-health epochs and recover retention governance.

Run 161 proves provider retention and independent challenge/read-back for the active
archive set.  Run 162 adds two independent trust planes:

* an out-of-band pinned witness root signs cross-auditor views of the exact Run 161
  archive-health state.  Any observed split view fails closed; and
* a separately pinned recovery root can replace a compromised Run 161 retention root
  without using the compromised root to authorize archive retirement.

Private signing keys are never accepted by this tool.
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

import tomllib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)

try:
    from . import audit_archive_retention as run161
except (ImportError, ValueError) as exc:
    import importlib.util

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "_run161_health_for_witness", _here / "audit_archive_retention.py"
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("audit_archive_retention.py") from exc
    run161 = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = run161
    _spec.loader.exec_module(run161)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_archive_witness_policy.toml").read_text())
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
_DOC_ARCHIVE_WITNESS_STATE = "trusted-archive-witness-state.json"


_WITNESS_OUTPUT_NAMES = {
    "release-archive-witness-bundle.json",
    _DOC_ARCHIVE_WITNESS_STATE,
    "active-archive-witness-evidence.json",
    "release-archive-witness-receipt.json",
}
_RECOVERY_OUTPUT_NAMES = {
    "recovered-retention-root.json",
    "retention-root-recovery-record.json",
    "retention-root-recovery-receipt.json",
}


class ArchiveWitnessError(RuntimeError):
    """Run 162 archive witness/recovery invariant failed."""


def _fail(code: str) -> None:
    raise ArchiveWitnessError(code)


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
    except ArchiveWitnessError:
        raise
    except Exception as exc:
        raise ArchiveWitnessError(code + "_JSON_INVALID") from exc
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
        raise ArchiveWitnessError(code) from exc
    if expected_len is not None and len(raw) != expected_len:
        _fail(code)
    return raw


def _dt(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise ArchiveWitnessError(code) from exc


def _ts(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _ed25519_verify(
    public_text: str, signature_text: str, message: bytes, code: str
) -> None:
    public = _b64(public_text, code + "_PUBLIC_KEY_INVALID", expected_len=32)
    signature = _b64(
        signature_text, code + "_SIGNATURE_ENCODING_INVALID", expected_len=64
    )
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
    except (InvalidSignature, ValueError) as exc:
        raise ArchiveWitnessError(code + "_SIGNATURE_INVALID") from exc


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


def _file_fingerprint(path: Path, code: str) -> tuple[str, int, int]:
    if path.is_symlink() or not path.is_file():
        _fail(code)
    st = path.stat()
    return (_sha(path), st.st_size, st.st_mode & 0o7777)


def _dir_fingerprint(root: Path, code: str) -> dict[str, tuple[str, int, int]]:
    root = _regular_dir(root, code)
    out = {}
    for p in root.iterdir():
        if p.is_symlink() or not p.is_file():
            _fail(code)
        out[p.name] = _file_fingerprint(p, code)
    return out


def _key(value: Any, code: str, *, channel: bool = False) -> dict[str, Any]:
    expected = {"identity", "operator", "expires", "publicKey"} | (
        {"recoveryChannel"} if channel else set()
    )
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    out = {
        "identity": _identity(value.get("identity"), code + "_IDENTITY_INVALID"),
        "operator": _identity(value.get("operator"), code + "_OPERATOR_INVALID"),
        "expires": _ts(_dt(value.get("expires"), code + "_EXPIRES_INVALID")),
        "publicKey": value.get("publicKey"),
    }
    _b64(out["publicKey"], code + "_PUBLIC_KEY_INVALID", expected_len=32)
    if channel:
        out["recoveryChannel"] = _identity(
            value.get("recoveryChannel"), code + "_CHANNEL_INVALID"
        )
    if value != out:
        _fail(code + "_NOT_NORMALIZED")
    return out


def _verify_threshold_root(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    expected_sha256: str,
    *,
    root_type: str,
    schema_key: str,
    min_keys_key: str,
    min_threshold_key: str,
    min_operators_key: str,
    now: datetime,
    historical: bool = False,
    channel: bool = False,
) -> dict[str, Any]:
    if set(doc) != {"signed", "signatures"} or not isinstance(doc.get("signed"), dict):
        _fail("ARCHIVE_WITNESS_ROOT_SCHEMA_INVALID")
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
        or s.get("_type") != root_type
        or s.get("specVersion") != str(POLICY["spec_version"])
        or s.get("schemaVersion") != int(POLICY[schema_key])
    ):
        _fail("ARCHIVE_WITNESS_ROOT_SIGNED_SCHEMA_INVALID")
    root_id = _identity(s.get("rootId"), "ARCHIVE_WITNESS_ROOT_ID_INVALID")
    version = _size(s.get("version"), "ARCHIVE_WITNESS_ROOT_VERSION_INVALID")
    if version != 1:
        _fail("ARCHIVE_WITNESS_ROOT_VERSION_INVALID")
    issued = _dt(s.get("issuedAt"), "ARCHIVE_WITNESS_ROOT_ISSUED_INVALID")
    expires = _dt(s.get("expires"), "ARCHIVE_WITNESS_ROOT_EXPIRES_INVALID")
    if expires <= issued or expires - issued > timedelta(
        days=int(POLICY["max_root_lifetime_days"])
    ):
        _fail("ARCHIVE_WITNESS_ROOT_LIFETIME_INVALID")
    if not historical:
        if issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
            _fail("ARCHIVE_WITNESS_ROOT_FROM_FUTURE")
        if expires <= now:
            _fail("ARCHIVE_WITNESS_ROOT_EXPIRED")
    raw_keys = s.get("keys")
    if not isinstance(raw_keys, dict) or not (
        int(POLICY[min_keys_key]) <= len(raw_keys) <= int(POLICY["max_key_count"])
    ):
        _fail("ARCHIVE_WITNESS_ROOT_KEYS_INVALID")
    keys = {}
    for kid, value in raw_keys.items():
        kid = _identity(  # ruff: ignore[redefined-loop-name]
            kid,
            "ARCHIVE_WITNESS_ROOT_KEY_ID_INVALID",
        )
        keys[kid] = _key(value, "ARCHIVE_WITNESS_ROOT_KEY", channel=channel)
        if (
            _dt(keys[kid]["expires"], "ARCHIVE_WITNESS_ROOT_KEY_EXPIRES_INVALID")
            < expires
        ):
            _fail("ARCHIVE_WITNESS_ROOT_KEY_EXPIRES_BEFORE_ROOT")
    threshold = _size(s.get("threshold"), "ARCHIVE_WITNESS_ROOT_THRESHOLD_INVALID")
    if threshold < int(POLICY[min_threshold_key]) or threshold > len(keys):
        _fail("ARCHIVE_WITNESS_ROOT_THRESHOLD_INVALID")
    selected = s.get("selectedSignerKeyIds")
    if not isinstance(selected, list) or len(selected) != threshold:
        _fail("ARCHIVE_WITNESS_ROOT_SELECTED_INVALID")
    selected = [_identity(x, "ARCHIVE_WITNESS_ROOT_SELECTED_INVALID") for x in selected]
    if (
        selected != sorted(selected)
        or len(set(selected)) != len(selected)
        or any(x not in keys for x in selected)
    ):
        _fail("ARCHIVE_WITNESS_ROOT_SELECTED_INVALID")
    if len({keys[x]["operator"] for x in selected}) < int(POLICY[min_operators_key]):
        _fail("ARCHIVE_WITNESS_ROOT_OPERATOR_QUORUM_INVALID")
    sigs = doc.get("signatures")
    if not isinstance(sigs, list) or len(sigs) != len(selected):
        _fail("ARCHIVE_WITNESS_ROOT_SIGNATURE_SET_INVALID")
    sig_map = {}
    for item in sigs:
        if not isinstance(item, dict) or set(item) != {"keyId", "signature"}:
            _fail("ARCHIVE_WITNESS_ROOT_SIGNATURE_SCHEMA_INVALID")
        kid = _identity(item.get("keyId"), "ARCHIVE_WITNESS_ROOT_SIGNATURE_KEY_INVALID")
        if kid in sig_map:
            _fail("ARCHIVE_WITNESS_ROOT_SIGNATURE_DUPLICATE")
        sig_map[kid] = item.get("signature")
    if sorted(sig_map) != selected or sigs != sorted(sigs, key=lambda x: x["keyId"]):
        _fail("ARCHIVE_WITNESS_ROOT_SIGNATURE_SET_INVALID")
    signed_bytes = _canonical(s)
    for kid in selected:
        _ed25519_verify(
            keys[kid]["publicKey"], sig_map[kid], signed_bytes, "ARCHIVE_WITNESS_ROOT"
        )
    raw = _canonical(doc)
    sha = _sha_bytes(raw)
    if sha != _hex(expected_sha256, "ARCHIVE_WITNESS_ROOT_PIN_INVALID"):
        _fail("ARCHIVE_WITNESS_ROOT_PIN_MISMATCH")
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


def _verify_run161(
    *,
    run160_dir: Path,
    run161_dir: Path,
    retention_root_path: Path,
    retention_root_pin: str,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    try:
        verified = run161.verify_archive_health(
            run160_dir=run160_dir,
            output_dir=run161_dir,
            retention_root_path=retention_root_path,
            expected_retention_root_sha256=retention_root_pin,
            expected_bootstrap_root_sha256=bootstrap_pin,
            expected_recovery_root_sha256=recovery_pin,
            expected_attestation_root_sha256=attestation_pins,
            now=now,
            historical=historical,
        )
    except Exception as exc:
        raise ArchiveWitnessError("ARCHIVE_WITNESS_RUN161_INVALID:" + str(exc)) from exc
    docs = {}
    raws = {}
    for name in sorted(
        _WITNESS_OUTPUT_NAMES
        | {
            "release-archive-health-bundle.json",
            _DOC_ARCHIVE_HEALTH_STATE,
            "active-archive-health-evidence.json",
            "release-archive-health-receipt.json",
        }
    ):
        p = run161_dir / name
        if p.exists() and name.startswith(
            (
                "release-archive-health",
                "trusted-archive-health",
                "active-archive-health",
            )
        ):
            doc, raw = _read_json(
                p,
                "ARCHIVE_WITNESS_RUN161_"
                + name.upper().replace("-", "_").replace(".", "_"),
            )
            docs[name] = doc
            raws[name] = raw
    retention_doc, _ = _read_json(retention_root_path, "ARCHIVE_WITNESS_RETENTION_ROOT")
    try:
        retention_root = run161._verify_root(
            retention_doc, retention_root_pin, now=now, historical=historical
        )
    except Exception as exc:
        raise ArchiveWitnessError(
            "ARCHIVE_WITNESS_RETENTION_ROOT_INVALID:" + str(exc)
        ) from exc
    return {
        "verified": verified,
        "docs": docs,
        "raws": raws,
        "retention_root": retention_root,
    }


def _run161_view(run161_docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    state = run161_docs[_DOC_ARCHIVE_HEALTH_STATE]
    active = run161_docs["active-archive-health-evidence.json"]
    membership = active.get("membership", {}).get("signed", {})
    members = membership.get("members")
    audits = active.get("audits")
    if not isinstance(members, list) or not isinstance(audits, list):
        _fail("ARCHIVE_WITNESS_RUN161_VIEW_INVALID")
    by_audit = {x.get("archiveId"): x for x in audits if isinstance(x, dict)}
    rows = []
    for m in members:
        if not isinstance(m, dict):
            _fail("ARCHIVE_WITNESS_RUN161_VIEW_INVALID")
        aid = m.get("archiveId")
        a = by_audit.get(aid)
        if not isinstance(a, dict):
            _fail("ARCHIVE_WITNESS_RUN161_VIEW_INVALID")
        rows.append(
            {
                "archiveId": aid,
                "archiveOperator": m.get("archiveOperator"),
                "locator": m.get("locator"),
                "immutability": m.get("immutability"),
                "immutableVersionId": a.get("immutableVersionId"),
                "retentionMode": a.get("retentionMode"),
                "retentionUntil": a.get("retentionUntil"),
                "legalHold": a.get("legalHold"),
                "providerResponseSha256": a.get("providerResponseSha256"),
                "auditorResponseSha256": a.get("auditorResponseSha256"),
            }
        )
    rows.sort(key=lambda x: x["archiveId"])
    return {
        "run161Sequence": state.get("sequence"),
        "healthChainHeadSha256": state.get("healthChainHeadSha256"),
        "activeArchiveIds": state.get("activeArchiveIds"),
        "archives": rows,
    }


def _enforce_witness_separation(
    root: dict[str, Any],
    retention_root: dict[str, Any],
    run161_docs: dict[str, dict[str, Any]],
) -> None:
    active = run161_docs["active-archive-health-evidence.json"]
    members = active.get("membership", {}).get("signed", {}).get("members")
    if not isinstance(members, list):
        _fail("ARCHIVE_WITNESS_AUTHORITY_PLANES_INVALID")
    witness_ops = {v["operator"] for v in root["keys"].values()}
    witness_pubs = {v["publicKey"] for v in root["keys"].values()}
    other_ops = {v["operator"] for v in retention_root["keys"].values()}
    other_pubs = {v["publicKey"] for v in retention_root["keys"].values()}
    for m in members:
        if not isinstance(m, dict):
            _fail("ARCHIVE_WITNESS_AUTHORITY_PLANES_INVALID")
        other_ops.update(
            [m.get("archiveOperator"), m.get("auditorKey", {}).get("operator")]
        )
        other_pubs.update(
            [
                m.get("providerKey", {}).get("publicKey"),
                m.get("auditorKey", {}).get("publicKey"),
            ]
        )
    other_ops.discard(None)
    other_pubs.discard(None)
    if witness_ops & other_ops or witness_pubs & other_pubs:
        _fail("ARCHIVE_WITNESS_AUTHORITY_PLANES_OVERLAP")


def _challenge(  # ruff: ignore[too-many-positional-arguments]
    sequence: int,
    previous_head: str | None,
    run161_head: str,
    view_sha: str,
    state_sha: str,
    active_sha: str,
    selected_witness_key_ids: list[str],
) -> str:
    return _sha_bytes(
        _canonical(
            {
                "sequence": sequence,
                "previousWitnessChainHeadSha256": previous_head,
                "run161HealthChainHeadSha256": run161_head,
                "viewSha256": view_sha,
                "run161HealthStateSha256": state_sha,
                "run161ActiveEvidenceSha256": active_sha,
                "selectedWitnessKeyIds": selected_witness_key_ids,
            }
        )
    )


def _command_adapter(
    command: list[str], *, prefix: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if not command:
        _fail(prefix + "_COMMAND_EMPTY")

    def call(request: dict[str, Any]) -> dict[str, Any]:
        proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
        )
        payload = _canonical(request)
        out_parts = []
        err_parts = []
        total = [0]
        lock = threading.Lock()
        too_large = [False]

        def drain(stream, parts):
            try:
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    with lock:
                        total[0] += len(chunk)
                        if total[0] > int(POLICY["max_adapter_output_bytes"]):
                            too_large[0] = True
                            proc.kill()
                            return
                    parts.append(chunk)
            finally:
                stream.close()

        assert proc.stdin  # ruff: ignore[assert]
        assert proc.stdout  # ruff: ignore[assert]
        assert proc.stderr  # ruff: ignore[assert]
        t1 = threading.Thread(target=drain, args=(proc.stdout, out_parts), daemon=True)
        t2 = threading.Thread(target=drain, args=(proc.stderr, err_parts), daemon=True)
        t1.start()
        t2.start()
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
            try:
                rc = proc.wait(timeout=int(POLICY["adapter_timeout_seconds"]))
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.wait()
                raise ArchiveWitnessError(prefix + "_TIMEOUT") from exc
            t1.join()
            t2.join()
            if too_large[0]:
                _fail(prefix + "_OUTPUT_TOO_LARGE")
            if rc != 0:
                _fail(prefix + "_FAILED")
            return _loads(b"".join(out_parts), prefix + "_RESPONSE")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if t1.is_alive():
                t1.join()
            if t2.is_alive():
                t2.join()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:  # ruff: ignore[suppressible-exception]
                    stream.close()
                except OSError:  # ruff: ignore[try-except-in-loop]
                    pass

    return call


def command_witness(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="ARCHIVE_WITNESS_ADAPTER")


def _verify_witness_response(
    value: dict[str, Any],
    *,
    key_id: str,
    key: dict[str, Any],
    sequence: int,
    challenge: str,
    expected_view: dict[str, Any],
    expected_artifacts: dict[str, Any],
    selected_witness_key_ids: list[str],
    root_issued: datetime,
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    if (
        set(value) != {"signed", "signature"}
        or not isinstance(value.get("signed"), dict)
        or not isinstance(value.get("signature"), dict)
    ):
        _fail("ARCHIVE_WITNESS_RESPONSE_SCHEMA_INVALID")
    s = value["signed"]
    expected = {
        "schemaVersion",
        "operation",
        "sequence",
        "witnessIdentity",
        "witnessOperator",
        "challenge",
        "selectedWitnessKeyIds",
        "run161Artifacts",
        "view",
        "readOnly",
        "credentialsReused",
        "observedAt",
    }
    if (
        set(s) != expected
        or s.get("schemaVersion") != int(POLICY["witness_protocol_version"])
        or s.get("operation") != "witness-archive-health"
    ):
        _fail("ARCHIVE_WITNESS_RESPONSE_SIGNED_SCHEMA_INVALID")
    if (
        s.get("sequence") != sequence
        or s.get("witnessIdentity") != key["identity"]
        or s.get("witnessOperator") != key["operator"]
        or s.get("challenge") != challenge
        or s.get("selectedWitnessKeyIds") != selected_witness_key_ids
        or s.get("run161Artifacts") != expected_artifacts
    ):
        _fail("ARCHIVE_WITNESS_RESPONSE_BINDING_INVALID")
    if s.get("readOnly") is not True or s.get("credentialsReused") is not False:
        _fail("ARCHIVE_WITNESS_RESPONSE_AUTHORITY_INVALID")
    if s.get("view") != expected_view:
        _fail("ARCHIVE_WITNESS_SPLIT_VIEW_DETECTED")
    observed = _dt(s.get("observedAt"), "ARCHIVE_WITNESS_OBSERVED_AT_INVALID")
    if observed < root_issued - timedelta(
        minutes=int(POLICY["max_clock_skew_minutes"])
    ):
        _fail("ARCHIVE_WITNESS_OBSERVATION_BEFORE_ROOT")
    if not historical and (
        observed > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        or now - observed
        > timedelta(minutes=int(POLICY["max_observation_age_minutes"]))
    ):
        _fail("ARCHIVE_WITNESS_OBSERVATION_NOT_FRESH")
    sig = value["signature"]
    if set(sig) != {"keyId", "signature"} or sig.get("keyId") != key_id:
        _fail("ARCHIVE_WITNESS_SIGNATURE_BINDING_INVALID")
    if _dt(key["expires"], "ARCHIVE_WITNESS_KEY_EXPIRES_INVALID") < observed:
        _fail("ARCHIVE_WITNESS_KEY_EXPIRED")
    _ed25519_verify(
        key["publicKey"], sig.get("signature"), _canonical(s), "ARCHIVE_WITNESS"
    )
    return {"observed": observed, "sha256": _sha_bytes(_canonical(value)), "doc": value}


def _event_head(previous: str | None, event_core: dict[str, Any]) -> str:
    return _sha_bytes(
        _canonical({"previousWitnessChainHeadSha256": previous, "event": event_core})
    )


def _load_witness_output(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_WITNESS_OUTPUT_DIR_INVALID")
    docs = {}
    raws = {}
    for name in _WITNESS_OUTPUT_NAMES:
        doc, raw = _read_json(
            root / name,
            "ARCHIVE_WITNESS_OUTPUT_"
            + name.upper().replace("-", "_").replace(".", "_"),
        )
        docs[name] = doc
        raws[name] = raw
    if {p.name for p in root.iterdir()} != _WITNESS_OUTPUT_NAMES:
        _fail("ARCHIVE_WITNESS_OUTPUT_ALLOWLIST_INVALID")
    return docs, raws


def verify_witness_history(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    run160_dir: Path,
    run161_dir: Path,
    retention_root_path: Path,
    retention_root_pin: str,
    witness_root_path: Path,
    witness_root_pin: str,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    output_dir: Path,
    now: datetime | None = None,
    historical: bool = False,
    require_current_match: bool = True,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run160_dir = _regular_dir(run160_dir, "ARCHIVE_WITNESS_RUN160_DIR_INVALID")
    run161_dir = _regular_dir(run161_dir, "ARCHIVE_WITNESS_RUN161_DIR_INVALID")
    root_doc, _ = _read_json(witness_root_path, "ARCHIVE_WITNESS_ROOT")
    root = _verify_threshold_root(
        root_doc,
        witness_root_pin,
        root_type="archive-health-witness-root",
        schema_key="witness_root_schema_version",
        min_keys_key="min_witness_keys",
        min_threshold_key="min_witness_threshold",
        min_operators_key="min_witness_operators",
        now=current,
        historical=historical,
    )
    r161 = _verify_run161(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        now=current,
        historical=bool(historical),
    )
    expected_view = _run161_view(r161["docs"])
    expected_view_sha = _sha_bytes(_canonical(expected_view))
    _enforce_witness_separation(root, r161["retention_root"], r161["docs"])
    docs, raws = _load_witness_output(output_dir)
    bundle = docs["release-archive-witness-bundle.json"]
    receipt = docs["release-archive-witness-receipt.json"]
    if (
        set(bundle)
        != {"schemaVersion", "predicateType", "status", "witnessRoot", "events"}
        or bundle.get("schemaVersion") != int(POLICY["witness_bundle_schema_version"])
        or bundle.get("predicateType") != PREDICATE_TYPE
        or bundle.get("status") != "archive-health-witness-history"
    ):
        _fail("ARCHIVE_WITNESS_BUNDLE_SCHEMA_INVALID")
    if bundle.get("witnessRoot") != {
        "sha256": root["sha256"],
        "rootId": root["rootId"],
        "version": root["version"],
    }:
        _fail("ARCHIVE_WITNESS_BUNDLE_ROOT_BINDING_INVALID")
    if (
        set(receipt) != {"schemaVersion", "status", "events"}
        or receipt.get("schemaVersion") != int(POLICY["witness_receipt_schema_version"])
        or receipt.get("status") != "archive-health-witnessed"
    ):
        _fail("ARCHIVE_WITNESS_RECEIPT_SCHEMA_INVALID")
    events = bundle.get("events")
    receipts = receipt.get("events")
    if (
        not isinstance(events, list)
        or not events
        or not isinstance(receipts, list)
        or len(receipts) != len(events)
    ):
        _fail("ARCHIVE_WITNESS_HISTORY_LENGTH_INVALID")
    previous = None
    last_time = None
    for idx, (event, rec) in enumerate(zip(events, receipts), start=1):
        if not isinstance(event, dict) or set(event) != {
            "sequence",
            "run161HealthChainHeadSha256",
            "run161HealthStateArtifact",
            "run161ActiveArtifact",
            "view",
            "viewSha256",
            "challenge",
            "witnessKeyIds",
            "witnessResponseSha256s",
            "witnessChainHeadSha256",
        }:
            _fail("ARCHIVE_WITNESS_EVENT_SCHEMA_INVALID")
        if event.get("sequence") != idx:
            _fail("ARCHIVE_WITNESS_SEQUENCE_INVALID")
        if (
            not isinstance(rec, dict)
            or set(rec)
            != {"sequence", "run161HealthState", "run161ActiveEvidence", "responses"}
            or rec.get("sequence") != idx
            or not isinstance(rec.get("responses"), list)
        ):
            _fail("ARCHIVE_WITNESS_RECEIPT_EVENT_INVALID")
        embedded_state = rec["run161HealthState"]
        embedded_active = rec["run161ActiveEvidence"]
        if not isinstance(embedded_state, dict) or not isinstance(
            embedded_active, dict
        ):
            _fail("ARCHIVE_WITNESS_EMBEDDED_RUN161_INVALID")
        state_raw = _canonical(embedded_state)
        active_raw = _canonical(embedded_active)
        if event.get("run161HealthStateArtifact") != {
            "name": _DOC_ARCHIVE_HEALTH_STATE,
            "sha256": _sha_bytes(state_raw),
            "size": len(state_raw),
        } or event.get("run161ActiveArtifact") != {
            "name": "active-archive-health-evidence.json",
            "sha256": _sha_bytes(active_raw),
            "size": len(active_raw),
        }:
            _fail("ARCHIVE_WITNESS_RUN161_ARTIFACT_BINDING_INVALID")
        embedded_view = _run161_view(
            {
                _DOC_ARCHIVE_HEALTH_STATE: embedded_state,
                "active-archive-health-evidence.json": embedded_active,
            }
        )
        embedded_view_sha = _sha_bytes(_canonical(embedded_view))
        if (
            event.get("view") != embedded_view
            or event.get("run161HealthChainHeadSha256")
            != embedded_view["healthChainHeadSha256"]
            or event.get("viewSha256") != embedded_view_sha
        ):
            _fail("ARCHIVE_WITNESS_EVENT_VIEW_BINDING_INVALID")
        if (
            idx == len(events)
            and require_current_match
            and (
                embedded_state != r161["docs"][_DOC_ARCHIVE_HEALTH_STATE]
                or embedded_active
                != r161["docs"]["active-archive-health-evidence.json"]
            )
        ):
            _fail("ARCHIVE_WITNESS_CURRENT_RUN161_MISMATCH")
        artifacts = {
            "run161HealthStateArtifact": event["run161HealthStateArtifact"],
            "run161ActiveArtifact": event["run161ActiveArtifact"],
        }
        selected_event = event.get("witnessKeyIds")
        if (
            not isinstance(selected_event, list)
            or selected_event != sorted(selected_event)
            or len(set(selected_event)) != len(selected_event)
        ):
            _fail("ARCHIVE_WITNESS_EVENT_WITNESS_SET_INVALID")
        challenge = _challenge(
            idx,
            previous,
            embedded_view["healthChainHeadSha256"],
            embedded_view_sha,
            event["run161HealthStateArtifact"]["sha256"],
            event["run161ActiveArtifact"]["sha256"],
            selected_event,
        )
        if event.get("challenge") != challenge:
            _fail("ARCHIVE_WITNESS_CHALLENGE_INVALID")
        responses = rec["responses"]
        key_ids = []
        response_hashes = []
        observed_times = []
        operators = set()
        for item in responses:
            if not isinstance(item, dict) or set(item) != {"keyId", "response"}:
                _fail("ARCHIVE_WITNESS_RECEIPT_RESPONSE_INVALID")
            kid = _identity(item.get("keyId"), "ARCHIVE_WITNESS_KEY_ID_INVALID")
            if kid not in root["keys"] or kid in key_ids:
                _fail("ARCHIVE_WITNESS_KEY_ID_INVALID")
            verified = _verify_witness_response(
                item["response"],
                key_id=kid,
                key=root["keys"][kid],
                sequence=idx,
                challenge=challenge,
                expected_view=embedded_view,
                expected_artifacts=artifacts,
                selected_witness_key_ids=selected_event,
                root_issued=root["issued"],
                now=current,
                historical=True if idx < len(events) else historical,
            )
            key_ids.append(kid)
            response_hashes.append(verified["sha256"])
            observed_times.append(verified["observed"])
            operators.add(root["keys"][kid]["operator"])
        if len(key_ids) < root["threshold"] or len(operators) < int(
            POLICY["min_witness_operators"]
        ):
            _fail("ARCHIVE_WITNESS_QUORUM_INVALID")
        if key_ids != sorted(key_ids):
            _fail("ARCHIVE_WITNESS_RESPONSES_NOT_SORTED")
        if (
            event.get("witnessKeyIds") != key_ids
            or event.get("witnessResponseSha256s") != response_hashes
        ):
            _fail("ARCHIVE_WITNESS_RESPONSE_BINDING_INVALID")
        core = {k: event[k] for k in event if k != "witnessChainHeadSha256"}
        head = _event_head(previous, core)
        if event.get("witnessChainHeadSha256") != head:
            _fail("ARCHIVE_WITNESS_CHAIN_HEAD_INVALID")
        if last_time is not None and min(observed_times) - last_time > timedelta(
            days=int(POLICY["maximum_witness_interval_days"])
        ):
            _fail("ARCHIVE_WITNESS_INTERVAL_EXCEEDED")
        last_time = max(observed_times)
        previous = head
    state = docs[_DOC_ARCHIVE_WITNESS_STATE]
    active = docs["active-archive-witness-evidence.json"]
    bundle_item = {
        "name": "release-archive-witness-bundle.json",
        "sha256": _sha_bytes(raws["release-archive-witness-bundle.json"]),
        "size": len(raws["release-archive-witness-bundle.json"]),
    }
    expected_state = {
        "schemaVersion": int(POLICY["witness_state_schema_version"]),
        "status": "trusted-archive-witness",
        "sequence": len(events),
        "witnessChainHeadSha256": previous,
        "run161HealthChainHeadSha256": events[-1]["run161HealthChainHeadSha256"],
        "bundleArtifact": bundle_item,
        "activeArchiveIds": events[-1]["view"]["activeArchiveIds"],
    }
    expected_active = {
        "schemaVersion": int(POLICY["witness_active_schema_version"]),
        "sequence": len(events),
        "witnessChainHeadSha256": previous,
        "challenge": events[-1]["challenge"],
        "view": events[-1]["view"],
        "witnessKeyIds": events[-1]["witnessKeyIds"],
    }
    if state != expected_state or active != expected_active:
        _fail("ARCHIVE_WITNESS_STATE_BINDING_INVALID")
    if (
        not historical
        and last_time is not None
        and current - last_time
        > timedelta(days=int(POLICY["maximum_witness_interval_days"]))
    ):
        _fail("ARCHIVE_WITNESS_ACTIVE_EPOCH_STALE")
    return {
        "ok": True,
        "sequence": len(events),
        "witness_chain_head_sha256": previous,
        "witness_count": len(events[-1]["witnessKeyIds"]),
        "view": expected_view,
    }


def witness_archive_health(  # ruff: ignore[undocumented-public-function]
    *,
    run160_dir: Path,
    run161_dir: Path,
    retention_root_path: Path,
    retention_root_pin: str,
    witness_root_path: Path,
    witness_root_pin: str,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    output_dir: Path,
    witnesses: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]],
    previous_output_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run160_dir = _regular_dir(run160_dir, "ARCHIVE_WITNESS_RUN160_DIR_INVALID")
    run161_dir = _regular_dir(run161_dir, "ARCHIVE_WITNESS_RUN161_DIR_INVALID")
    root_doc, _ = _read_json(witness_root_path, "ARCHIVE_WITNESS_ROOT")
    root = _verify_threshold_root(
        root_doc,
        witness_root_pin,
        root_type="archive-health-witness-root",
        schema_key="witness_root_schema_version",
        min_keys_key="min_witness_keys",
        min_threshold_key="min_witness_threshold",
        min_operators_key="min_witness_operators",
        now=current,
    )
    r161 = _verify_run161(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        now=current,
        historical=False,
    )
    _enforce_witness_separation(root, r161["retention_root"], r161["docs"])
    expected_view = _run161_view(r161["docs"])
    view_sha = _sha_bytes(_canonical(expected_view))
    run161_head = expected_view["healthChainHeadSha256"]
    state_doc = r161["docs"][_DOC_ARCHIVE_HEALTH_STATE]
    active_doc = r161["docs"]["active-archive-health-evidence.json"]
    state_raw = _canonical(state_doc)
    active_raw = _canonical(active_doc)
    state_item = {
        "name": _DOC_ARCHIVE_HEALTH_STATE,
        "sha256": _sha_bytes(state_raw),
        "size": len(state_raw),
    }
    active_item = {
        "name": "active-archive-health-evidence.json",
        "sha256": _sha_bytes(active_raw),
        "size": len(active_raw),
    }
    artifacts = {
        "run161HealthStateArtifact": state_item,
        "run161ActiveArtifact": active_item,
    }
    previous_bundle = None
    previous_receipts = []
    previous_head = None
    protected = [
        run160_dir,
        run161_dir,
        retention_root_path.resolve(),
        witness_root_path.resolve(),
    ]
    if previous_output_dir is not None:
        previous_output_dir = _regular_dir(
            previous_output_dir, "ARCHIVE_WITNESS_PREVIOUS_DIR_INVALID"
        )
        protected.append(previous_output_dir)
        prev = verify_witness_history(
            run160_dir=run160_dir,
            run161_dir=run161_dir,
            retention_root_path=retention_root_path,
            retention_root_pin=retention_root_pin,
            witness_root_path=witness_root_path,
            witness_root_pin=witness_root_pin,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            output_dir=previous_output_dir,
            now=current,
            historical=True,
            require_current_match=False,
        )
        pdocs, _ = _load_witness_output(previous_output_dir)
        previous_bundle = pdocs["release-archive-witness-bundle.json"]
        previous_receipts = list(
            pdocs["release-archive-witness-receipt.json"]["events"]
        )
        previous_head = prev["witness_chain_head_sha256"]
    target = _outside(output_dir, protected, "ARCHIVE_WITNESS_OUTPUT_OVERLAPS_INPUT")
    if target.exists():
        _fail("ARCHIVE_WITNESS_OUTPUT_EXISTS")
    adapters = {
        _identity(k, "ARCHIVE_WITNESS_ADAPTER_KEY_INVALID"): a for k, a in witnesses
    }
    if len(adapters) != len(witnesses) or any(k not in root["keys"] for k in adapters):
        _fail("ARCHIVE_WITNESS_ADAPTER_SET_INVALID")
    if len(adapters) < root["threshold"] or len(
        {root["keys"][k]["operator"] for k in adapters}
    ) < int(POLICY["min_witness_operators"]):
        _fail("ARCHIVE_WITNESS_QUORUM_INVALID")
    sequence = 1 if previous_bundle is None else len(previous_bundle["events"]) + 1
    selected_witness_key_ids = sorted(adapters)
    challenge = _challenge(
        sequence,
        previous_head,
        run161_head,
        view_sha,
        state_item["sha256"],
        active_item["sha256"],
        selected_witness_key_ids,
    )
    run160_fp = _dir_fingerprint(run160_dir, "ARCHIVE_WITNESS_RUN160_INPUT_DRIFT")
    run161_fp = _dir_fingerprint(run161_dir, "ARCHIVE_WITNESS_RUN161_INPUT_DRIFT")
    rr_fp = _file_fingerprint(
        retention_root_path.resolve(), "ARCHIVE_WITNESS_RETENTION_ROOT_DRIFT"
    )
    wr_fp = _file_fingerprint(
        witness_root_path.resolve(), "ARCHIVE_WITNESS_WITNESS_ROOT_DRIFT"
    )
    prev_fp = (
        _dir_fingerprint(previous_output_dir, "ARCHIVE_WITNESS_PREVIOUS_DRIFT")
        if previous_output_dir is not None
        else None
    )
    responses = []
    hashes = []
    request = {
        "schemaVersion": int(POLICY["witness_protocol_version"]),
        "operation": "witness-archive-health",
        "sequence": sequence,
        "challenge": challenge,
        "selectedWitnessKeyIds": selected_witness_key_ids,
        "run161Artifacts": artifacts,
        "expectedView": expected_view,
    }
    for kid in sorted(adapters):
        raw = adapters[kid](request)
        verified = _verify_witness_response(
            raw,
            key_id=kid,
            key=root["keys"][kid],
            sequence=sequence,
            challenge=challenge,
            expected_view=expected_view,
            expected_artifacts=artifacts,
            selected_witness_key_ids=selected_witness_key_ids,
            root_issued=root["issued"],
            now=current,
            historical=False,
        )
        responses.append({"keyId": kid, "response": raw})
        hashes.append(verified["sha256"])
    if (
        _dir_fingerprint(run160_dir, "ARCHIVE_WITNESS_RUN160_INPUT_DRIFT") != run160_fp
        or _dir_fingerprint(run161_dir, "ARCHIVE_WITNESS_RUN161_INPUT_DRIFT")
        != run161_fp
        or _file_fingerprint(
            retention_root_path.resolve(), "ARCHIVE_WITNESS_RETENTION_ROOT_DRIFT"
        )
        != rr_fp
        or _file_fingerprint(
            witness_root_path.resolve(), "ARCHIVE_WITNESS_WITNESS_ROOT_DRIFT"
        )
        != wr_fp
    ):
        _fail("ARCHIVE_WITNESS_INPUT_DRIFT_DETECTED")
    if (
        previous_output_dir is not None
        and _dir_fingerprint(previous_output_dir, "ARCHIVE_WITNESS_PREVIOUS_DRIFT")
        != prev_fp
    ):
        _fail("ARCHIVE_WITNESS_INPUT_DRIFT_DETECTED")
    core = {
        "sequence": sequence,
        "run161HealthChainHeadSha256": run161_head,
        "viewSha256": view_sha,
        "challenge": challenge,
        "witnessKeyIds": [x["keyId"] for x in responses],
        "witnessResponseSha256s": hashes,
    }
    core.update(
        {
            "run161HealthStateArtifact": state_item,
            "run161ActiveArtifact": active_item,
            "view": expected_view,
        }
    )
    head = _event_head(previous_head, core)
    event = dict(core, witnessChainHeadSha256=head)
    if previous_bundle is None:
        bundle = {
            "schemaVersion": int(POLICY["witness_bundle_schema_version"]),
            "predicateType": PREDICATE_TYPE,
            "status": "archive-health-witness-history",
            "witnessRoot": {
                "sha256": root["sha256"],
                "rootId": root["rootId"],
                "version": root["version"],
            },
            "events": [event],
        }
    else:
        bundle = json.loads(json.dumps(previous_bundle))
        bundle["events"].append(event)
    receipt = {
        "schemaVersion": int(POLICY["witness_receipt_schema_version"]),
        "status": "archive-health-witnessed",
        "events": [
            *previous_receipts,
            {
                "sequence": sequence,
                "run161HealthState": state_doc,
                "run161ActiveEvidence": active_doc,
                "responses": responses,
            },
        ],
    }
    bundle_raw = _canonical(bundle)
    bundle_item = {
        "name": "release-archive-witness-bundle.json",
        "sha256": _sha_bytes(bundle_raw),
        "size": len(bundle_raw),
    }
    state = {
        "schemaVersion": int(POLICY["witness_state_schema_version"]),
        "status": "trusted-archive-witness",
        "sequence": sequence,
        "witnessChainHeadSha256": head,
        "run161HealthChainHeadSha256": run161_head,
        "bundleArtifact": bundle_item,
        "activeArchiveIds": expected_view["activeArchiveIds"],
    }
    active = {
        "schemaVersion": int(POLICY["witness_active_schema_version"]),
        "sequence": sequence,
        "witnessChainHeadSha256": head,
        "challenge": challenge,
        "view": expected_view,
        "witnessKeyIds": [x["keyId"] for x in responses],
    }
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run162-witness-", dir=parent))
    try:
        _write(stage / "release-archive-witness-bundle.json", bundle)
        _write(stage / _DOC_ARCHIVE_WITNESS_STATE, state)
        _write(stage / "active-archive-witness-evidence.json", active)
        _write(stage / "release-archive-witness-receipt.json", receipt)
        verify_witness_history(
            run160_dir=run160_dir,
            run161_dir=run161_dir,
            retention_root_path=retention_root_path,
            retention_root_pin=retention_root_pin,
            witness_root_path=witness_root_path,
            witness_root_pin=witness_root_pin,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            output_dir=stage,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_witness_history(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        witness_root_path=witness_root_path,
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        output_dir=target,
        now=current,
    )


def _recovery_subject(  # ruff: ignore[too-many-branches, too-many-positional-arguments]
    doc: dict[str, Any],
    recovery_root: dict[str, Any],
    old_root: dict[str, Any],
    new_root: dict[str, Any],
    active_state: dict[str, Any],
    now: datetime,
    historical: bool = False,
) -> dict[str, Any]:
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "recoveryId",
        "issuedAt",
        "recoveryRootSha256",
        "oldRetentionRootSha256",
        "newRetentionRootSha256",
        "run161HealthChainHeadSha256",
        "activeArchiveIds",
        "compromisedOldKeyIds",
        "selectedRecoveryKeyIds",
        "retirementAuthorizedArchiveIds",
    }
    if (
        set(doc) != expected
        or doc.get("_type") != "archive-retention-root-recovery"
        or doc.get("specVersion") != str(POLICY["spec_version"])
        or doc.get("schemaVersion") != int(POLICY["recovery_record_schema_version"])
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_SUBJECT_SCHEMA_INVALID")
    _identity(doc.get("recoveryId"), "ARCHIVE_WITNESS_RECOVERY_ID_INVALID")
    issued = _dt(doc.get("issuedAt"), "ARCHIVE_WITNESS_RECOVERY_ISSUED_INVALID")
    if not historical and issued > now + timedelta(
        minutes=int(POLICY["max_clock_skew_minutes"])
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_FROM_FUTURE")
    if issued < recovery_root["issued"] or issued > recovery_root["expires"]:
        _fail("ARCHIVE_WITNESS_RECOVERY_OUTSIDE_RECOVERY_ROOT_LIFETIME")
    if issued < new_root["issued"] or issued > new_root["expires"]:
        _fail("ARCHIVE_WITNESS_RECOVERY_OUTSIDE_NEW_ROOT_LIFETIME")
    if (
        doc.get("recoveryRootSha256") != recovery_root["sha256"]
        or doc.get("oldRetentionRootSha256") != old_root["sha256"]
        or doc.get("newRetentionRootSha256") != new_root["sha256"]
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_ROOT_BINDING_INVALID")
    if doc.get("run161HealthChainHeadSha256") != active_state.get(
        "healthChainHeadSha256"
    ) or doc.get("activeArchiveIds") != active_state.get("activeArchiveIds"):
        _fail("ARCHIVE_WITNESS_RECOVERY_HEALTH_BINDING_INVALID")
    if doc.get("retirementAuthorizedArchiveIds") != []:
        _fail("ARCHIVE_WITNESS_RECOVERY_RETIREMENT_FORBIDDEN")
    compromised = doc.get("compromisedOldKeyIds")
    if (
        not isinstance(compromised, list)
        or not compromised
        or compromised != sorted(compromised)
        or len(set(compromised)) != len(compromised)
        or any(k not in old_root["keys"] for k in compromised)
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_COMPROMISED_KEYS_INVALID")
    selected = doc.get("selectedRecoveryKeyIds")
    if (
        not isinstance(selected, list)
        or len(selected) != recovery_root["threshold"]
        or selected != sorted(selected)
        or len(set(selected)) != len(selected)
        or any(k not in recovery_root["keys"] for k in selected)
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_SELECTED_INVALID")
    if len({recovery_root["keys"][k]["operator"] for k in selected}) < int(
        POLICY["min_recovery_operators"]
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_OPERATOR_QUORUM_INVALID")
    if len({recovery_root["keys"][k]["recoveryChannel"] for k in selected}) < int(
        POLICY["min_recovery_channels"]
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_CHANNEL_QUORUM_INVALID")
    if new_root["rootId"] != old_root["rootId"]:
        _fail("ARCHIVE_WITNESS_RECOVERY_ROOT_ID_MISMATCH")
    old_pubs = {v["publicKey"] for v in old_root["keys"].values()}
    new_pubs = {v["publicKey"] for v in new_root["keys"].values()}
    recovery_pubs = {v["publicKey"] for v in recovery_root["keys"].values()}
    old_ops = {v["operator"] for v in old_root["keys"].values()}
    new_ops = {v["operator"] for v in new_root["keys"].values()}
    recovery_ops = {v["operator"] for v in recovery_root["keys"].values()}
    if (
        set(old_root["keys"]) & set(new_root["keys"])
        or old_pubs & new_pubs
        or old_ops & new_ops
        or recovery_pubs & (old_pubs | new_pubs)
        or recovery_ops & (old_ops | new_ops)
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_AUTHORITY_SEPARATION_INVALID")
    if any(k in new_root["keys"] for k in compromised):
        _fail("ARCHIVE_WITNESS_RECOVERY_COMPROMISED_KEY_REINTRODUCED")
    return {"issued": issued, "selected": selected, "compromised": compromised}


def verify_retention_root_recovery(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    run160_dir: Path,
    run161_dir: Path,
    old_retention_root_path: Path,
    old_retention_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    output_dir: Path,
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run160_dir = _regular_dir(run160_dir, "ARCHIVE_WITNESS_RUN160_DIR_INVALID")
    run161_dir = _regular_dir(run161_dir, "ARCHIVE_WITNESS_RUN161_DIR_INVALID")
    output_dir = _regular_dir(output_dir, "ARCHIVE_WITNESS_RECOVERY_OUTPUT_DIR_INVALID")
    if {p.name for p in output_dir.iterdir()} != _RECOVERY_OUTPUT_NAMES:
        _fail("ARCHIVE_WITNESS_RECOVERY_OUTPUT_ALLOWLIST_INVALID")
    old_doc, _ = _read_json(
        old_retention_root_path, "ARCHIVE_WITNESS_OLD_RETENTION_ROOT"
    )
    try:
        old = run161._verify_root(
            old_doc, old_retention_root_pin, now=current, historical=True
        )
    except Exception as exc:
        raise ArchiveWitnessError(
            "ARCHIVE_WITNESS_OLD_RETENTION_ROOT_INVALID:" + str(exc)
        ) from exc
    new_doc, new_raw = _read_json(
        output_dir / "recovered-retention-root.json", "ARCHIVE_WITNESS_RECOVERED_ROOT"
    )
    new_sha = _sha_bytes(new_raw)
    try:
        new = run161._verify_root(new_doc, new_sha, now=current, historical=historical)
    except Exception as exc:
        raise ArchiveWitnessError(
            "ARCHIVE_WITNESS_NEW_RETENTION_ROOT_INVALID:" + str(exc)
        ) from exc
    recovery_doc, _ = _read_json(recovery_root_path, "ARCHIVE_WITNESS_RECOVERY_ROOT")
    rr = _verify_threshold_root(
        recovery_doc,
        recovery_root_pin,
        root_type="archive-retention-recovery-root",
        schema_key="recovery_root_schema_version",
        min_keys_key="min_recovery_keys",
        min_threshold_key="min_recovery_threshold",
        min_operators_key="min_recovery_operators",
        now=current,
        historical=historical,
        channel=True,
    )
    _verify_run161(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        retention_root_path=old_retention_root_path,
        retention_root_pin=old_retention_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        now=current,
        historical=True,
    )
    active, _ = _read_json(
        run161_dir / _DOC_ARCHIVE_HEALTH_STATE, "ARCHIVE_WITNESS_RUN161_STATE"
    )
    record, record_raw = _read_json(
        output_dir / "retention-root-recovery-record.json",
        "ARCHIVE_WITNESS_RECOVERY_RECORD",
    )
    receipt, _ = _read_json(
        output_dir / "retention-root-recovery-receipt.json",
        "ARCHIVE_WITNESS_RECOVERY_RECEIPT",
    )
    if (
        set(record)
        != {
            "schemaVersion",
            "status",
            "subject",
            "signatures",
            "oldRetentionRoot",
            "newRetentionRoot",
            "recoveryRoot",
        }
        or record.get("schemaVersion") != int(POLICY["recovery_record_schema_version"])
        or record.get("status") != "retention-root-recovered"
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_RECORD_SCHEMA_INVALID")
    subject = record.get("subject")
    parsed = _recovery_subject(
        subject, rr, old, new, active, current, historical=historical
    )
    sigs = record.get("signatures")
    if not isinstance(sigs, list) or len(sigs) != len(parsed["selected"]):
        _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURE_SET_INVALID")
    sig_map = {}
    for item in sigs:
        if not isinstance(item, dict) or set(item) != {"keyId", "channel", "signature"}:
            _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURE_SCHEMA_INVALID")
        kid = _identity(
            item.get("keyId"), "ARCHIVE_WITNESS_RECOVERY_SIGNATURE_KEY_INVALID"
        )
        if (
            kid in sig_map
            or kid not in rr["keys"]
            or item.get("channel") != rr["keys"][kid]["recoveryChannel"]
        ):
            _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURE_BINDING_INVALID")
        sig_map[kid] = item.get("signature")
    if sorted(sig_map) != parsed["selected"] or sigs != sorted(
        sigs, key=lambda x: x["keyId"]
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURE_SET_INVALID")
    for kid in parsed["selected"]:
        if (
            _dt(
                rr["keys"][kid]["expires"],
                "ARCHIVE_WITNESS_RECOVERY_KEY_EXPIRES_INVALID",
            )
            < parsed["issued"]
        ):
            _fail("ARCHIVE_WITNESS_RECOVERY_KEY_EXPIRED")
        _ed25519_verify(
            rr["keys"][kid]["publicKey"],
            sig_map[kid],
            _canonical(subject),
            "ARCHIVE_WITNESS_RECOVERY",
        )
    expected_record = {
        "schemaVersion": int(POLICY["recovery_record_schema_version"]),
        "status": "retention-root-recovered",
        "subject": subject,
        "signatures": sigs,
        "oldRetentionRoot": {"sha256": old["sha256"], "rootId": old["rootId"]},
        "newRetentionRoot": {"sha256": new["sha256"], "rootId": new["rootId"]},
        "recoveryRoot": {"sha256": rr["sha256"], "rootId": rr["rootId"]},
    }
    if record != expected_record:
        _fail("ARCHIVE_WITNESS_RECOVERY_RECORD_BINDING_INVALID")
    expected_receipt = {
        "schemaVersion": int(POLICY["recovery_receipt_schema_version"]),
        "status": "retention-root-recovery-accepted",
        "run161HealthChainHeadSha256": active["healthChainHeadSha256"],
        "activeArchiveIds": active["activeArchiveIds"],
        "retirementAuthorizedArchiveIds": [],
        "recordSha256": _sha_bytes(record_raw),
        "recoveredRetentionRootSha256": new["sha256"],
    }
    if receipt != expected_receipt:
        _fail("ARCHIVE_WITNESS_RECOVERY_RECEIPT_MISMATCH")
    return {
        "ok": True,
        "phase": "retention-root-recovery-verified",
        "recovered_retention_root_sha256": new["sha256"],
        "run161_health_chain_head_sha256": active["healthChainHeadSha256"],
        "retirement_authorized": [],
    }


def recover_retention_root(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    run160_dir: Path,
    run161_dir: Path,
    old_retention_root_path: Path,
    old_retention_root_pin: str,
    new_retention_root_path: Path,
    recovery_root_path: Path,
    recovery_root_pin: str,
    recovery_subject_path: Path,
    recovery_signatures_path: Path,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    output_dir: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run160_dir = _regular_dir(run160_dir, "ARCHIVE_WITNESS_RUN160_DIR_INVALID")
    run161_dir = _regular_dir(run161_dir, "ARCHIVE_WITNESS_RUN161_DIR_INVALID")
    old_doc, _ = _read_json(
        old_retention_root_path, "ARCHIVE_WITNESS_OLD_RETENTION_ROOT"
    )
    try:
        old = run161._verify_root(
            old_doc, old_retention_root_pin, now=current, historical=True
        )
    except Exception as exc:
        raise ArchiveWitnessError(
            "ARCHIVE_WITNESS_OLD_RETENTION_ROOT_INVALID:" + str(exc)
        ) from exc
    new_doc, new_raw = _read_json(
        new_retention_root_path, "ARCHIVE_WITNESS_NEW_RETENTION_ROOT"
    )
    new_sha = _sha_bytes(new_raw)
    try:
        new = run161._verify_root(new_doc, new_sha, now=current, historical=False)
    except Exception as exc:
        raise ArchiveWitnessError(
            "ARCHIVE_WITNESS_NEW_RETENTION_ROOT_INVALID:" + str(exc)
        ) from exc
    recovery_doc, _ = _read_json(recovery_root_path, "ARCHIVE_WITNESS_RECOVERY_ROOT")
    rr = _verify_threshold_root(
        recovery_doc,
        recovery_root_pin,
        root_type="archive-retention-recovery-root",
        schema_key="recovery_root_schema_version",
        min_keys_key="min_recovery_keys",
        min_threshold_key="min_recovery_threshold",
        min_operators_key="min_recovery_operators",
        now=current,
        channel=True,
    )
    _verify_run161(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        retention_root_path=old_retention_root_path,
        retention_root_pin=old_retention_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        now=current,
        historical=True,
    )
    active, _ = _read_json(
        run161_dir / _DOC_ARCHIVE_HEALTH_STATE, "ARCHIVE_WITNESS_RUN161_STATE"
    )
    subject, _ = _read_json(recovery_subject_path, "ARCHIVE_WITNESS_RECOVERY_SUBJECT")
    parsed = _recovery_subject(subject, rr, old, new, active, current)
    signatures, _ = _read_json(
        recovery_signatures_path, "ARCHIVE_WITNESS_RECOVERY_SIGNATURES"
    )
    if (
        set(signatures) != {"schemaVersion", "signatures"}
        or signatures.get("schemaVersion")
        != int(POLICY["recovery_record_schema_version"])
        or not isinstance(signatures.get("signatures"), list)
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURES_SCHEMA_INVALID")
    sigs = signatures["signatures"]
    if len(sigs) != len(parsed["selected"]):
        _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURE_SET_INVALID")
    sig_map = {}
    for item in sigs:
        if not isinstance(item, dict) or set(item) != {"keyId", "channel", "signature"}:
            _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURE_SCHEMA_INVALID")
        kid = _identity(
            item.get("keyId"), "ARCHIVE_WITNESS_RECOVERY_SIGNATURE_KEY_INVALID"
        )
        if (
            kid in sig_map
            or kid not in rr["keys"]
            or item.get("channel") != rr["keys"][kid]["recoveryChannel"]
        ):
            _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURE_BINDING_INVALID")
        sig_map[kid] = item.get("signature")
    if sorted(sig_map) != parsed["selected"] or sigs != sorted(
        sigs, key=lambda x: x["keyId"]
    ):
        _fail("ARCHIVE_WITNESS_RECOVERY_SIGNATURE_SET_INVALID")
    for kid in parsed["selected"]:
        if (
            _dt(
                rr["keys"][kid]["expires"],
                "ARCHIVE_WITNESS_RECOVERY_KEY_EXPIRES_INVALID",
            )
            < parsed["issued"]
        ):
            _fail("ARCHIVE_WITNESS_RECOVERY_KEY_EXPIRED")
        _ed25519_verify(
            rr["keys"][kid]["publicKey"],
            sig_map[kid],
            _canonical(subject),
            "ARCHIVE_WITNESS_RECOVERY",
        )
    protected = [
        run160_dir,
        run161_dir,
        old_retention_root_path.resolve(),
        new_retention_root_path.resolve(),
        recovery_root_path.resolve(),
        recovery_subject_path.resolve(),
        recovery_signatures_path.resolve(),
    ]
    target = _outside(
        output_dir, protected, "ARCHIVE_WITNESS_RECOVERY_OUTPUT_OVERLAPS_INPUT"
    )
    if target.exists():
        _fail("ARCHIVE_WITNESS_RECOVERY_OUTPUT_EXISTS")
    fps = [
        _dir_fingerprint(run160_dir, "ARCHIVE_WITNESS_RECOVERY_RUN160_DRIFT"),
        _dir_fingerprint(run161_dir, "ARCHIVE_WITNESS_RECOVERY_RUN161_DRIFT"),
        _file_fingerprint(
            old_retention_root_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_OLD_ROOT_DRIFT"
        ),
        _file_fingerprint(
            new_retention_root_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_NEW_ROOT_DRIFT"
        ),
        _file_fingerprint(
            recovery_root_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_ROOT_DRIFT"
        ),
        _file_fingerprint(
            recovery_subject_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_SUBJECT_DRIFT"
        ),
        _file_fingerprint(
            recovery_signatures_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_SIGS_DRIFT"
        ),
    ]
    fps2 = [
        _dir_fingerprint(run160_dir, "ARCHIVE_WITNESS_RECOVERY_RUN160_DRIFT"),
        _dir_fingerprint(run161_dir, "ARCHIVE_WITNESS_RECOVERY_RUN161_DRIFT"),
        _file_fingerprint(
            old_retention_root_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_OLD_ROOT_DRIFT"
        ),
        _file_fingerprint(
            new_retention_root_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_NEW_ROOT_DRIFT"
        ),
        _file_fingerprint(
            recovery_root_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_ROOT_DRIFT"
        ),
        _file_fingerprint(
            recovery_subject_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_SUBJECT_DRIFT"
        ),
        _file_fingerprint(
            recovery_signatures_path.resolve(), "ARCHIVE_WITNESS_RECOVERY_SIGS_DRIFT"
        ),
    ]
    if fps != fps2:
        _fail("ARCHIVE_WITNESS_RECOVERY_INPUT_DRIFT_DETECTED")
    record = {
        "schemaVersion": int(POLICY["recovery_record_schema_version"]),
        "status": "retention-root-recovered",
        "subject": subject,
        "signatures": sigs,
        "oldRetentionRoot": {"sha256": old["sha256"], "rootId": old["rootId"]},
        "newRetentionRoot": {"sha256": new["sha256"], "rootId": new["rootId"]},
        "recoveryRoot": {"sha256": rr["sha256"], "rootId": rr["rootId"]},
    }
    receipt = {
        "schemaVersion": int(POLICY["recovery_receipt_schema_version"]),
        "status": "retention-root-recovery-accepted",
        "run161HealthChainHeadSha256": active["healthChainHeadSha256"],
        "activeArchiveIds": active["activeArchiveIds"],
        "retirementAuthorizedArchiveIds": [],
        "recordSha256": _sha_bytes(_canonical(record)),
        "recoveredRetentionRootSha256": new["sha256"],
    }
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run162-recovery-", dir=parent))
    try:
        (stage / "recovered-retention-root.json").write_bytes(new_raw)
        _write(stage / "retention-root-recovery-record.json", record)
        _write(stage / "retention-root-recovery-receipt.json", receipt)
        if {p.name for p in stage.iterdir()} != _RECOVERY_OUTPUT_NAMES:
            _fail("ARCHIVE_WITNESS_RECOVERY_OUTPUT_ALLOWLIST_INVALID")
        verify_retention_root_recovery(
            run160_dir=run160_dir,
            run161_dir=run161_dir,
            old_retention_root_path=old_retention_root_path,
            old_retention_root_pin=old_retention_root_pin,
            recovery_root_path=recovery_root_path,
            recovery_root_pin=recovery_root_pin,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            output_dir=stage,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verified = verify_retention_root_recovery(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        old_retention_root_path=old_retention_root_path,
        old_retention_root_pin=old_retention_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        output_dir=target,
        now=current,
    )
    return dict(verified, old_retention_root_sha256=old["sha256"])


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    # The Python API is the release integration surface; CLI intentionally supports offline verification only here.
    v = sub.add_parser("verify-witness")
    v.add_argument("--run160-dir", type=Path, required=True)
    v.add_argument("--run161-dir", type=Path, required=True)
    v.add_argument("--retention-root", type=Path, required=True)
    v.add_argument("--retention-root-pin", required=True)
    v.add_argument("--witness-root", type=Path, required=True)
    v.add_argument("--witness-root-pin", required=True)
    v.add_argument("--bootstrap-pin", required=True)
    v.add_argument("--recovery-pin", required=True)
    v.add_argument("--attestation-pin", action="append", required=True)
    v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--historical", action="store_true")
    args = p.parse_args(argv)
    if args.command == "verify-witness":
        result = verify_witness_history(
            run160_dir=args.run160_dir,
            run161_dir=args.run161_dir,
            retention_root_path=args.retention_root,
            retention_root_pin=args.retention_root_pin,
            witness_root_path=args.witness_root,
            witness_root_pin=args.witness_root_pin,
            bootstrap_pin=args.bootstrap_pin,
            recovery_pin=args.recovery_pin,
            attestation_pins=args.attestation_pin,
            output_dir=args.output_dir,
            historical=args.historical,
        )
        logger.info("%s", json.dumps(result, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
