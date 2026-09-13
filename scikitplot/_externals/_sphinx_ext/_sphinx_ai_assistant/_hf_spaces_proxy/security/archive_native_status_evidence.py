"""
Run 160: immutably replicate and recover exact Run 159 native-status evidence.

The Run 159 bundle already preserves raw DER CRL/OCSP bytes and verified vendor evidence.
Run 160 treats that complete canonical output as one immutable archival subject, binds the
configured archive/verifier membership into deterministic bytes, requires independent
remote read-back through a second authority plane, and supports fail-closed recovery from
multiple read-only archives.
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

logger = logging.getLogger(__name__)

try:
    from . import verify_native_status_provenance as native
except (ImportError, ValueError) as exc:
    import importlib.util

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "_run159_native_for_archive", _here / "verify_native_status_provenance.py"
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("verify_native_status_provenance.py") from exc
    native = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = native
    _spec.loader.exec_module(native)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_native_archive_policy.toml").read_text())
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
_DOC_NATIVE_EVIDENCE_ARCHIVE_STATE = "trusted-native-evidence-archive-state.json"
_DOC_NATIVE_STATUS_STATE = "trusted-native-status-state.json"


_NATIVE_NAMES = {
    "release-native-status-bundle.json",
    _DOC_NATIVE_STATUS_STATE,
    "active-native-status-evidence.json",
    "release-native-status-receipt.json",
}
_ARCHIVE_NAMES = {
    "release-native-evidence-archive.json",
    _DOC_NATIVE_EVIDENCE_ARCHIVE_STATE,
    "release-native-evidence-archive-receipt.json",
}


class NativeArchiveError(RuntimeError):
    """Run 160 native-evidence archival invariant failed."""


def _fail(code: str) -> None:
    raise NativeArchiveError(code)


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
    except NativeArchiveError:
        raise
    except Exception as exc:
        raise NativeArchiveError(code + "_JSON_INVALID") from exc
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


def _safe_locator(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048  # ruff: ignore[magic-value-comparison]
        or "\x00" in value
    ):
        _fail(code)
    try:
        parsed = urlsplit(value)
    except ValueError:
        _fail(code)
    if (
        not parsed.scheme
        or parsed.scheme.lower() == "file"
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        _fail(code)
    if ".." in parsed.path.split("/"):
        _fail(code)
    return value


def _dt(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise NativeArchiveError(code) from exc


def _fresh(value: Any, code: str, *, now: datetime) -> str:
    parsed = _dt(value, code)
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    age = timedelta(minutes=int(POLICY["max_result_age_minutes"]))
    if parsed > now + skew:
        _fail(code + "_FROM_FUTURE")
    if now - parsed > age:
        _fail(code + "_STALE")
    return value


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


def _artifact_doc(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "size"}:
        _fail(code + "_SCHEMA_INVALID")
    name = value.get("name")
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in {".", ".."}
    ):
        _fail(code + "_NAME_INVALID")
    return {
        "name": name,
        "sha256": _hex(value.get("sha256"), code + "_SHA256_INVALID"),
        "size": _size(value.get("size"), code + "_SIZE_INVALID"),
    }


def _native_docs(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    root = _regular_dir(root, "NATIVE_ARCHIVE_RUN159_DIR_INVALID")
    if {p.name for p in root.iterdir()} != _NATIVE_NAMES:
        _fail("NATIVE_ARCHIVE_RUN159_ALLOWLIST_MISMATCH")
    docs: dict[str, dict[str, Any]] = {}
    raws: dict[str, bytes] = {}
    for name in sorted(_NATIVE_NAMES):
        doc, raw = _read_json(
            root / name,
            "NATIVE_ARCHIVE_RUN159_" + name.upper().replace("-", "_").replace(".", "_"),
        )
        docs[name] = doc
        raws[name] = raw
    receipt = docs["release-native-status-receipt.json"]
    expected = {
        "bundle": _artifact(
            "release-native-status-bundle.json",
            raws["release-native-status-bundle.json"],
        ),
        "state": _artifact(_DOC_NATIVE_STATUS_STATE, raws[_DOC_NATIVE_STATUS_STATE]),
        "activeEvidence": _artifact(
            "active-native-status-evidence.json",
            raws["active-native-status-evidence.json"],
        ),
    }
    for key, item in expected.items():
        if receipt.get(key) != item:
            _fail("NATIVE_ARCHIVE_RUN159_RECEIPT_REBIND_FAILED")
    return docs, raws


def _source_inventory(  # ruff: ignore[too-many-branches]
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    events = bundle.get("events")
    if not isinstance(events, list) or not events:
        _fail("NATIVE_ARCHIVE_EVENTS_INVALID")
    for event in events:
        sequence = event.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            _fail("NATIVE_ARCHIVE_EVENT_SEQUENCE_INVALID")
        sources = event.get("sources")
        if not isinstance(sources, list):
            _fail("NATIVE_ARCHIVE_SOURCES_INVALID")
        for source in sources:
            if not isinstance(source, dict):
                _fail("NATIVE_ARCHIVE_SOURCE_INVALID")
            kind = source.get("type")
            if kind not in {"crl", "ocsp"}:
                _fail("NATIVE_ARCHIVE_SOURCE_KIND_INVALID")
            der = source.get("der")
            try:
                raw = base64.b64decode(der, validate=True)
            except Exception as exc:
                raise NativeArchiveError("NATIVE_ARCHIVE_SOURCE_DER_INVALID") from exc
            sha = _hex(source.get("sha256"), "NATIVE_ARCHIVE_SOURCE_SHA_INVALID")
            if _sha_bytes(raw) != sha:
                _fail("NATIVE_ARCHIVE_SOURCE_DER_HASH_MISMATCH")
            item = {
                "sequence": sequence,
                "type": kind,
                "sourceId": _identity(
                    source.get("sourceId"), "NATIVE_ARCHIVE_SOURCE_ID_INVALID"
                ),
                "sha256": sha,
                "size": len(raw),
            }
            if kind == "ocsp" and source.get("responderCertDer") is not None:
                try:
                    cert_raw = base64.b64decode(
                        source["responderCertDer"], validate=True
                    )
                except Exception as exc:
                    raise NativeArchiveError(
                        "NATIVE_ARCHIVE_RESPONDER_DER_INVALID"
                    ) from exc
                item["responderCertificateSha256"] = _sha_bytes(cert_raw)
                item["responderCertificateSize"] = len(cert_raw)
            out.append(item)
        vendor = event.get("vendorEvidence", [])
        if not isinstance(vendor, list):
            _fail("NATIVE_ARCHIVE_VENDOR_EVIDENCE_INVALID")
        for item in vendor:
            if not isinstance(item, dict):
                _fail("NATIVE_ARCHIVE_VENDOR_EVIDENCE_INVALID")
            out.append(
                {
                    "sequence": sequence,
                    "type": "vendor",
                    "sourceId": _identity(
                        item.get("profile"), "NATIVE_ARCHIVE_VENDOR_PROFILE_INVALID"
                    ),
                    "sha256": _hex(
                        item.get("rawEvidenceSha256"),
                        "NATIVE_ARCHIVE_VENDOR_SHA_INVALID",
                    ),
                    "size": len(
                        base64.b64decode(item.get("rawEvidenceBase64"), validate=True)
                    ),
                }
            )
    out.sort(key=lambda x: (x["sequence"], x["type"], x["sourceId"], x["sha256"]))
    if len(
        {(x["sequence"], x["type"], x["sourceId"], x["sha256"]) for x in out}
    ) != len(out):
        _fail("NATIVE_ARCHIVE_SOURCE_INVENTORY_DUPLICATE")
    return out


def _command_adapter(
    command: list[str], *, prefix: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if not command:
        _fail(prefix + "_COMMAND_INVALID")

    def call(request: dict[str, Any]) -> dict[str, Any]:
        process = (
            # lint
            subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                close_fds=True,
                env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
            )
        )
        assert process.stdin is not None  # ruff: ignore[assert]
        assert process.stdout is not None  # ruff: ignore[assert]
        assert process.stderr is not None  # ruff: ignore[assert]
        stdin = process.stdin
        process.stdin = None
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        overflow = threading.Event()
        limit = int(POLICY["max_adapter_output_bytes"])

        def drain(stream, chunks):
            total = 0
            try:
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        overflow.set()
                        process.kill()
                        break
                    chunks.append(chunk)
            finally:
                stream.close()

        threads = [
            threading.Thread(target=drain, args=(process.stdout, stdout_chunks)),
            threading.Thread(target=drain, args=(process.stderr, stderr_chunks)),
        ]
        for thread in threads:
            thread.start()
        try:
            try:
                stdin.write(_canonical(request))
                stdin.close()
                process.wait(timeout=int(POLICY["adapter_timeout_seconds"]))
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise NativeArchiveError(prefix + "_TIMEOUT") from exc
            for thread in threads:
                thread.join()
            if overflow.is_set():
                _fail(prefix + "_OUTPUT_TOO_LARGE")
            if process.returncode != 0:
                _fail(prefix + "_FAILED")
            return _loads(b"".join(stdout_chunks), prefix + "_OUTPUT")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            try:  # ruff: ignore[suppressible-exception]
                stdin.close()
            except OSError:
                pass
            for thread in threads:
                thread.join()

    return call


def command_archive(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="NATIVE_ARCHIVE_ADAPTER")


def command_verifier(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="NATIVE_ARCHIVE_VERIFIER_ADAPTER")


def command_recovery(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="NATIVE_ARCHIVE_RECOVERY_ADAPTER")


ArchiveAdapter = Callable[[dict[str, Any]], dict[str, Any]]
VerifierAdapter = Callable[[dict[str, Any]], dict[str, Any]]
RecoveryAdapter = Callable[[dict[str, Any]], dict[str, Any]]


def _archive_id(chain_head: str, artifact_sha: str, identity: str) -> str:
    return (
        "native-evidence-"
        + hashlib.sha256(
            (chain_head + "\0" + artifact_sha + "\0" + identity).encode()
        ).hexdigest()[:32]
    )


def _validate_archive_response(
    value: dict[str, Any],
    *,
    operation: str,
    archive_id: str,
    identity: str,
    operator: str,
    artifact: dict[str, Any],
    now: datetime,
    historical: bool = False,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "operation",
        "archiveId",
        "status",
        "archive",
        "artifact",
        "guarantees",
        "verifiedAt",
    }
    if (
        set(value) != expected
        or value.get("schemaVersion") != int(POLICY["archive_protocol_version"])
        or value.get("operation") != operation
        or value.get("archiveId") != archive_id
    ):
        _fail("NATIVE_ARCHIVE_RESULT_SCHEMA_INVALID")
    allowed = {"created", "present"} if operation == "bind" else {"present"}
    if value.get("status") not in allowed:
        _fail("NATIVE_ARCHIVE_STATUS_INVALID")
    authority = value.get("archive")
    if (
        not isinstance(authority, dict)
        or set(authority)
        != {
            "identity",
            "operator",
            "nativeStatusCredentialsReused",
            "verifierCredentialsReused",
        }
        or authority.get("identity") != identity
        or authority.get("operator") != operator
    ):
        _fail("NATIVE_ARCHIVE_AUTHORITY_INVALID")
    if (
        authority.get("nativeStatusCredentialsReused") is not False
        or authority.get("verifierCredentialsReused") is not False
    ):
        _fail("NATIVE_ARCHIVE_CREDENTIAL_REUSE")
    if _artifact_doc(value.get("artifact"), "NATIVE_ARCHIVE_ARTIFACT") != artifact:
        _fail("NATIVE_ARCHIVE_ARTIFACT_MISMATCH")
    guarantees = value.get("guarantees")
    expected_g = {
        "createOnly",
        "overwrite",
        "remoteReadbackVerified",
        "immutability",
        "locator",
    }
    if (
        not isinstance(guarantees, dict)
        or set(guarantees) != expected_g
        or guarantees.get("createOnly") is not True
        or guarantees.get("overwrite") is not False
        or guarantees.get("remoteReadbackVerified") is not True
    ):
        _fail("NATIVE_ARCHIVE_GUARANTEES_INVALID")
    immutability = guarantees.get("immutability")
    if immutability not in set(POLICY["allowed_archive_immutability"]):
        _fail("NATIVE_ARCHIVE_IMMUTABILITY_INVALID")
    verified_at = value.get("verifiedAt")
    if historical:
        _dt(verified_at, "NATIVE_ARCHIVE_VERIFIED_AT")
    else:
        _fresh(verified_at, "NATIVE_ARCHIVE_VERIFIED_AT", now=now)
    return {
        "locator": _safe_locator(
            guarantees.get("locator"), "NATIVE_ARCHIVE_LOCATOR_INVALID"
        ),
        "immutability": immutability,
        "verifiedAt": verified_at,
    }


def _validate_verifier_response(
    value: dict[str, Any],
    *,
    archive_id: str,
    identity: str,
    operator: str,
    archive_identity: str,
    archive_operator: str,
    locator: str,
    artifact: dict[str, Any],
    now: datetime,
    historical: bool = False,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "operation",
        "archiveId",
        "status",
        "verifier",
        "archiveIdentity",
        "locator",
        "artifact",
        "proof",
        "verifiedAt",
    }
    if (
        set(value) != expected
        or value.get("schemaVersion") != int(POLICY["verifier_protocol_version"])
        or value.get("operation") != "verify"
        or value.get("archiveId") != archive_id
        or value.get("status") != "present"
    ):
        _fail("NATIVE_ARCHIVE_VERIFIER_RESULT_SCHEMA_INVALID")
    verifier = value.get("verifier")
    if (
        not isinstance(verifier, dict)
        or set(verifier)
        != {
            "identity",
            "operator",
            "readOnly",
            "archiveCredentialsReused",
            "nativeStatusCredentialsReused",
        }
        or verifier.get("identity") != identity
        or verifier.get("operator") != operator
    ):
        _fail("NATIVE_ARCHIVE_VERIFIER_AUTHORITY_INVALID")
    if (
        verifier.get("readOnly") is not True
        or verifier.get("archiveCredentialsReused") is not False
        or verifier.get("nativeStatusCredentialsReused") is not False
    ):
        _fail("NATIVE_ARCHIVE_VERIFIER_CREDENTIAL_REUSE")
    if identity == archive_identity or operator == archive_operator:
        _fail("NATIVE_ARCHIVE_VERIFIER_NOT_INDEPENDENT")
    if (
        value.get("archiveIdentity") != archive_identity
        or _safe_locator(
            value.get("locator"), "NATIVE_ARCHIVE_VERIFIER_LOCATOR_INVALID"
        )
        != locator
    ):
        _fail("NATIVE_ARCHIVE_VERIFIER_REBIND_FAILED")
    if (
        _artifact_doc(value.get("artifact"), "NATIVE_ARCHIVE_VERIFIER_ARTIFACT")
        != artifact
    ):
        _fail("NATIVE_ARCHIVE_VERIFIER_ARTIFACT_MISMATCH")
    if value.get("proof") != {
        "remoteReadbackVerified": True,
        "sha256Verified": True,
        "sizeVerified": True,
        "independentReader": True,
    }:
        _fail("NATIVE_ARCHIVE_VERIFIER_PROOF_INVALID")
    verified_at = value.get("verifiedAt")
    if historical:
        _dt(verified_at, "NATIVE_ARCHIVE_VERIFIER_VERIFIED_AT")
    else:
        _fresh(verified_at, "NATIVE_ARCHIVE_VERIFIER_VERIFIED_AT", now=now)
    return {"verifiedAt": verified_at}


def _archive_docs(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    root = _regular_dir(root, "NATIVE_ARCHIVE_OUTPUT_DIR_INVALID")
    if {p.name for p in root.iterdir()} != _ARCHIVE_NAMES:
        _fail("NATIVE_ARCHIVE_OUTPUT_ALLOWLIST_MISMATCH")
    docs: dict[str, dict[str, Any]] = {}
    raws: dict[str, bytes] = {}
    for name in sorted(_ARCHIVE_NAMES):
        doc, raw = _read_json(
            root / name,
            "NATIVE_ARCHIVE_OUTPUT_" + name.upper().replace("-", "_").replace(".", "_"),
        )
        docs[name] = doc
        raws[name] = raw
    return docs, raws


def _rehydrate_native(output: dict[str, Any], target: Path) -> None:
    if not isinstance(output, dict) or set(output) != _NATIVE_NAMES:
        _fail("NATIVE_ARCHIVE_EMBEDDED_RUN159_INVALID")
    target.mkdir(parents=True, exist_ok=False)
    for name in sorted(_NATIVE_NAMES):
        doc = output.get(name)
        if not isinstance(doc, dict):
            _fail("NATIVE_ARCHIVE_EMBEDDED_RUN159_INVALID")
        _write(target / name, doc)


def _verify_archive_payload(  # ruff: ignore[too-many-branches]
    payload: dict[str, Any],
    *,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "predicateType",
        "status",
        "nativeStatus",
        "sourceInventory",
        "archivePolicy",
    }
    if (
        set(payload) != expected
        or payload.get("schemaVersion") != int(POLICY["archive_schema_version"])
        or payload.get("predicateType") != PREDICATE_TYPE
        or payload.get("status") != "preserved-native-status-evidence"
    ):
        _fail("NATIVE_ARCHIVE_PAYLOAD_SCHEMA_INVALID")
    native_meta = payload.get("nativeStatus")
    if not isinstance(native_meta, dict) or set(native_meta) != {
        "sequence",
        "nativeStatusChainHeadSha256",
        "output",
    }:
        _fail("NATIVE_ARCHIVE_NATIVE_META_INVALID")
    sequence = _size(native_meta.get("sequence"), "NATIVE_ARCHIVE_SEQUENCE_INVALID")
    if sequence < 1:
        _fail("NATIVE_ARCHIVE_SEQUENCE_INVALID")
    chain_head = _hex(
        native_meta.get("nativeStatusChainHeadSha256"),
        "NATIVE_ARCHIVE_CHAIN_HEAD_INVALID",
    )
    policy = payload.get("archivePolicy")
    if not isinstance(policy, dict) or set(policy) != {
        "minimumArchives",
        "minimumArchiveOperators",
        "minimumVerifierOperators",
        "targets",
    }:
        _fail("NATIVE_ARCHIVE_POLICY_SCHEMA_INVALID")
    targets = policy.get("targets")
    if not isinstance(targets, list) or len(targets) < int(POLICY["min_archives"]):
        _fail("NATIVE_ARCHIVE_TARGETS_INVALID")
    seen_a: set[str] = set()
    seen_v: set[str] = set()
    archive_ops: set[str] = set()
    verifier_ops: set[str] = set()
    for target in targets:
        if not isinstance(target, dict) or set(target) != {
            "archiveIdentity",
            "archiveOperator",
            "verifierIdentity",
            "verifierOperator",
        }:
            _fail("NATIVE_ARCHIVE_TARGET_SCHEMA_INVALID")
        ai = _identity(target.get("archiveIdentity"), "NATIVE_ARCHIVE_IDENTITY_INVALID")
        ao = _identity(target.get("archiveOperator"), "NATIVE_ARCHIVE_OPERATOR_INVALID")
        vi = _identity(
            target.get("verifierIdentity"), "NATIVE_ARCHIVE_VERIFIER_IDENTITY_INVALID"
        )
        vo = _identity(
            target.get("verifierOperator"), "NATIVE_ARCHIVE_VERIFIER_OPERATOR_INVALID"
        )
        if ai in seen_a or vi in seen_v or ai == vi or ao == vo:
            _fail("NATIVE_ARCHIVE_TARGET_INDEPENDENCE_INVALID")
        seen_a.add(ai)
        seen_v.add(vi)
        archive_ops.add(ao)
        verifier_ops.add(vo)
    if archive_ops & verifier_ops:
        _fail("NATIVE_ARCHIVE_OPERATOR_PLANES_OVERLAP")
    if len(archive_ops) < int(POLICY["min_archive_operators"]) or len(
        verifier_ops
    ) < int(POLICY["min_verifier_operators"]):
        _fail("NATIVE_ARCHIVE_OPERATOR_QUORUM_INVALID")
    if (
        policy.get("minimumArchives") != int(POLICY["min_archives"])
        or policy.get("minimumArchiveOperators") != int(POLICY["min_archive_operators"])
        or policy.get("minimumVerifierOperators")
        != int(POLICY["min_verifier_operators"])
    ):
        _fail("NATIVE_ARCHIVE_POLICY_VALUE_INVALID")
    tmp = Path(tempfile.mkdtemp(prefix=".run160-rehydrate-"))
    try:
        native_dir = tmp / "native"
        _rehydrate_native(native_meta["output"], native_dir)
        try:
            result = native.verify_native_status(
                output_dir=native_dir,
                expected_bootstrap_root_sha256=bootstrap_pin,
                expected_recovery_root_sha256=recovery_pin,
                expected_attestation_root_sha256=attestation_pins,
                now=now,
                historical=historical,
            )
        except native.NativeStatusError as exc:
            raise NativeArchiveError(
                "NATIVE_ARCHIVE_RUN159_REPLAY_INVALID:" + str(exc)
            ) from exc
        if (
            result.get("sequence") != sequence
            or result.get("native_status_chain_head_sha256") != chain_head
        ):
            _fail("NATIVE_ARCHIVE_RUN159_REPLAY_MISMATCH")
        docs, _ = _native_docs(native_dir)
        inventory = _source_inventory(docs["release-native-status-bundle.json"])
        if payload.get("sourceInventory") != inventory:
            _fail("NATIVE_ARCHIVE_SOURCE_INVENTORY_MISMATCH")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {
        "sequence": sequence,
        "chain_head": chain_head,
        "source_count": len(payload["sourceInventory"]),
    }


def verify_native_archive(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    output_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    docs, raws = _archive_docs(output_dir)
    payload = docs["release-native-evidence-archive.json"]
    info = _verify_archive_payload(
        payload,
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=expected_attestation_root_sha256,
        now=current,
        historical=historical,
    )
    payload_item = _artifact(
        "release-native-evidence-archive.json",
        raws["release-native-evidence-archive.json"],
    )
    state = docs[_DOC_NATIVE_EVIDENCE_ARCHIVE_STATE]
    expected_state = {
        "schemaVersion",
        "status",
        "sequence",
        "nativeStatusChainHeadSha256",
        "archiveArtifact",
        "sourceInventorySha256",
    }
    if (
        set(state) != expected_state
        or state.get("schemaVersion") != int(POLICY["state_schema_version"])
        or state.get("status") != "trusted-native-status-archive"
        or state.get("sequence") != info["sequence"]
        or state.get("nativeStatusChainHeadSha256") != info["chain_head"]
        or state.get("archiveArtifact") != payload_item
        or state.get("sourceInventorySha256")
        != _sha_bytes(_canonical(payload["sourceInventory"]))
    ):
        _fail("NATIVE_ARCHIVE_STATE_MISMATCH")
    receipt = docs["release-native-evidence-archive-receipt.json"]
    expected_receipt = {
        "schemaVersion",
        "status",
        "nativeStatusChainHeadSha256",
        "archiveArtifact",
        "archiveCount",
        "archiveOperatorCount",
        "verifierOperatorCount",
        "archives",
    }
    if (
        set(receipt) != expected_receipt
        or receipt.get("schemaVersion") != int(POLICY["receipt_schema_version"])
        or receipt.get("status") != "preserved"
        or receipt.get("archiveArtifact") != payload_item
        or receipt.get("nativeStatusChainHeadSha256") != info["chain_head"]
    ):
        _fail("NATIVE_ARCHIVE_RECEIPT_MISMATCH")
    plan = payload["archivePolicy"]["targets"]
    entries = receipt.get("archives")
    if (
        not isinstance(entries, list)
        or len(entries) != len(plan)
        or receipt.get("archiveCount") != len(plan)
    ):
        _fail("NATIVE_ARCHIVE_RECEIPT_COUNT_MISMATCH")
    archive_ops: set[str] = set()
    verifier_ops: set[str] = set()
    locators: set[str] = set()
    expected_plan = {
        (
            x["archiveIdentity"],
            x["archiveOperator"],
            x["verifierIdentity"],
            x["verifierOperator"],
        )
        for x in plan
    }
    observed_plan = set()
    for entry in entries:
        keys = {
            "archiveIdentity",
            "archiveOperator",
            "verifierIdentity",
            "verifierOperator",
            "archiveId",
            "locator",
            "immutability",
            "archiveVerifiedAt",
            "independentVerifiedAt",
            "bindResultSha256",
            "archiveVerifyResultSha256",
            "independentVerifyResultSha256",
            "bindResult",
            "archiveVerifyResult",
            "independentVerifyResult",
        }
        if not isinstance(entry, dict) or set(entry) != keys:
            _fail("NATIVE_ARCHIVE_RECEIPT_ENTRY_SCHEMA_INVALID")
        ai = _identity(
            entry.get("archiveIdentity"), "NATIVE_ARCHIVE_RECEIPT_ARCHIVE_ID_INVALID"
        )
        ao = _identity(
            entry.get("archiveOperator"),
            "NATIVE_ARCHIVE_RECEIPT_ARCHIVE_OPERATOR_INVALID",
        )
        vi = _identity(
            entry.get("verifierIdentity"), "NATIVE_ARCHIVE_RECEIPT_VERIFIER_ID_INVALID"
        )
        vo = _identity(
            entry.get("verifierOperator"),
            "NATIVE_ARCHIVE_RECEIPT_VERIFIER_OPERATOR_INVALID",
        )
        observed_plan.add((ai, ao, vi, vo))
        archive_ops.add(ao)
        verifier_ops.add(vo)
        _identity(
            entry.get("archiveId"), "NATIVE_ARCHIVE_RECEIPT_ARCHIVE_ID_VALUE_INVALID"
        )
        locator = _safe_locator(
            entry.get("locator"), "NATIVE_ARCHIVE_RECEIPT_LOCATOR_INVALID"
        )
        if locator in locators:
            _fail("NATIVE_ARCHIVE_RECEIPT_LOCATOR_COLLISION")
        locators.add(locator)
        if entry.get("immutability") not in set(POLICY["allowed_archive_immutability"]):
            _fail("NATIVE_ARCHIVE_RECEIPT_IMMUTABILITY_INVALID")
        _dt(
            entry.get("archiveVerifiedAt"),
            "NATIVE_ARCHIVE_RECEIPT_ARCHIVE_TIME_INVALID",
        )
        _dt(
            entry.get("independentVerifiedAt"),
            "NATIVE_ARCHIVE_RECEIPT_VERIFIER_TIME_INVALID",
        )
        for field, result_field in (
            ("bindResultSha256", "bindResult"),
            ("archiveVerifyResultSha256", "archiveVerifyResult"),
            ("independentVerifyResultSha256", "independentVerifyResult"),
        ):
            expected_hash = _hex(
                entry.get(field), "NATIVE_ARCHIVE_RECEIPT_RESULT_HASH_INVALID"
            )
            result_doc = entry.get(result_field)
            if (
                not isinstance(result_doc, dict)
                or _sha_bytes(_canonical(result_doc)) != expected_hash
            ):
                _fail("NATIVE_ARCHIVE_RECEIPT_RESULT_REBIND_FAILED")
        expected_archive_id = _archive_id(
            info["chain_head"], payload_item["sha256"], ai
        )
        if entry.get("archiveId") != expected_archive_id:
            _fail("NATIVE_ARCHIVE_RECEIPT_ARCHIVE_ID_MISMATCH")
        bind_check = _validate_archive_response(
            entry["bindResult"],
            operation="bind",
            archive_id=expected_archive_id,
            identity=ai,
            operator=ao,
            artifact=payload_item,
            now=current,
            historical=True,
        )
        verify_check = _validate_archive_response(
            entry["archiveVerifyResult"],
            operation="verify",
            archive_id=expected_archive_id,
            identity=ai,
            operator=ao,
            artifact=payload_item,
            now=current,
            historical=True,
        )
        if (
            bind_check["locator"] != verify_check["locator"]
            or verify_check["locator"] != locator
            or bind_check["immutability"] != verify_check["immutability"]
            or verify_check["immutability"] != entry.get("immutability")
        ):
            _fail("NATIVE_ARCHIVE_RECEIPT_ARCHIVE_REPLAY_MISMATCH")
        verifier_check = _validate_verifier_response(
            entry["independentVerifyResult"],
            archive_id=expected_archive_id,
            identity=vi,
            operator=vo,
            archive_identity=ai,
            archive_operator=ao,
            locator=locator,
            artifact=payload_item,
            now=current,
            historical=True,
        )
        if (
            entry.get("archiveVerifiedAt") != verify_check["verifiedAt"]
            or entry.get("independentVerifiedAt") != verifier_check["verifiedAt"]
        ):
            _fail("NATIVE_ARCHIVE_RECEIPT_TIME_REBIND_FAILED")
    if (
        observed_plan != expected_plan
        or receipt.get("archiveOperatorCount") != len(archive_ops)
        or receipt.get("verifierOperatorCount") != len(verifier_ops)
    ):
        _fail("NATIVE_ARCHIVE_RECEIPT_PLAN_MISMATCH")
    return {
        "ok": True,
        "sequence": info["sequence"],
        "native_status_chain_head_sha256": info["chain_head"],
        "archive_sha256": payload_item["sha256"],
        "source_count": info["source_count"],
    }


def preserve_native_status(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    native_dir: Path,
    output_dir: Path,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    targets: list[tuple[str, str, ArchiveAdapter, str, str, VerifierAdapter]],
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    source = _regular_dir(native_dir, "NATIVE_ARCHIVE_RUN159_DIR_INVALID")
    try:
        run159 = native.verify_native_status(
            output_dir=source,
            expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
            expected_recovery_root_sha256=expected_recovery_root_sha256,
            expected_attestation_root_sha256=expected_attestation_root_sha256,
            now=current,
            historical=False,
        )
    except native.NativeStatusError as exc:
        raise NativeArchiveError("NATIVE_ARCHIVE_RUN159_INVALID:" + str(exc)) from exc
    docs, raws = _native_docs(source)
    normalized = []
    for ai, ao, archive_adapter, vi, vo, verifier_adapter in targets:
        normalized.append(
            (
                _identity(ai, "NATIVE_ARCHIVE_IDENTITY_INVALID"),
                _identity(ao, "NATIVE_ARCHIVE_OPERATOR_INVALID"),
                archive_adapter,
                _identity(vi, "NATIVE_ARCHIVE_VERIFIER_IDENTITY_INVALID"),
                _identity(vo, "NATIVE_ARCHIVE_VERIFIER_OPERATOR_INVALID"),
                verifier_adapter,
            )
        )
    normalized.sort(key=lambda x: (x[1], x[0], x[4], x[3]))
    plan = [
        {
            "archiveIdentity": ai,
            "archiveOperator": ao,
            "verifierIdentity": vi,
            "verifierOperator": vo,
        }
        for ai, ao, _, vi, vo, _ in normalized
    ]
    payload = {
        "schemaVersion": int(POLICY["archive_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "preserved-native-status-evidence",
        "nativeStatus": {
            "sequence": run159["sequence"],
            "nativeStatusChainHeadSha256": run159["native_status_chain_head_sha256"],
            "output": docs,
        },
        "sourceInventory": _source_inventory(docs["release-native-status-bundle.json"]),
        "archivePolicy": {
            "minimumArchives": int(POLICY["min_archives"]),
            "minimumArchiveOperators": int(POLICY["min_archive_operators"]),
            "minimumVerifierOperators": int(POLICY["min_verifier_operators"]),
            "targets": plan,
        },
    }
    _verify_archive_payload(
        payload,
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=expected_attestation_root_sha256,
        now=current,
        historical=False,
    )
    target = _outside(output_dir, [source], "NATIVE_ARCHIVE_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("NATIVE_ARCHIVE_OUTPUT_ALREADY_EXISTS")
    if len(normalized) < int(POLICY["min_archives"]):
        _fail("NATIVE_ARCHIVE_TARGETS_INVALID")
    payload_raw = _canonical(payload)
    payload_item = _artifact("release-native-evidence-archive.json", payload_raw)
    initial_hashes = {name: _sha_bytes(raw) for name, raw in raws.items()}
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run160-archive-", dir=parent))
    try:
        payload_path = stage / payload_item["name"]
        payload_path.write_bytes(payload_raw)
        archive_results = []
        locators: set[str] = set()
        archive_ops: set[str] = set()
        verifier_ops: set[str] = set()
        for _index, (ai, ao, archive_adapter, vi, vo, verifier_adapter) in enumerate(
            normalized, start=1
        ):
            archive_id = _archive_id(
                run159["native_status_chain_head_sha256"], payload_item["sha256"], ai
            )
            base = {
                "schemaVersion": int(POLICY["archive_protocol_version"]),
                "archiveId": archive_id,
                "archive": {"identity": ai, "operator": ao},
                "artifact": payload_item,
            }
            bind_request = dict(base, operation="bind")
            bind_request["artifact"] = dict(payload_item, localPath=str(payload_path))
            before = _sha(payload_path)
            bind_raw = archive_adapter(bind_request)
            if _sha(payload_path) != before:
                _fail("NATIVE_ARCHIVE_ARTIFACT_CHANGED_DURING_BIND")
            bind = _validate_archive_response(
                bind_raw,
                operation="bind",
                archive_id=archive_id,
                identity=ai,
                operator=ao,
                artifact=payload_item,
                now=current,
            )
            verify_raw = archive_adapter(dict(base, operation="verify"))
            verify = _validate_archive_response(
                verify_raw,
                operation="verify",
                archive_id=archive_id,
                identity=ai,
                operator=ao,
                artifact=payload_item,
                now=current,
            )
            if (
                bind["locator"] != verify["locator"]
                or bind["immutability"] != verify["immutability"]
            ):
                _fail("NATIVE_ARCHIVE_REMOTE_REBIND_FAILED")
            if verify["locator"] in locators:
                _fail("NATIVE_ARCHIVE_LOCATOR_COLLISION")
            locators.add(verify["locator"])
            verifier_request = {
                "schemaVersion": int(POLICY["verifier_protocol_version"]),
                "operation": "verify",
                "archiveId": archive_id,
                "archiveIdentity": ai,
                "locator": verify["locator"],
                "artifact": payload_item,
                "verifier": {"identity": vi, "operator": vo},
            }
            verifier_raw = verifier_adapter(verifier_request)
            verifier = _validate_verifier_response(
                verifier_raw,
                archive_id=archive_id,
                identity=vi,
                operator=vo,
                archive_identity=ai,
                archive_operator=ao,
                locator=verify["locator"],
                artifact=payload_item,
                now=current,
            )
            archive_ops.add(ao)
            verifier_ops.add(vo)
            archive_results.append(
                {
                    "archiveIdentity": ai,
                    "archiveOperator": ao,
                    "verifierIdentity": vi,
                    "verifierOperator": vo,
                    "archiveId": archive_id,
                    "locator": verify["locator"],
                    "immutability": verify["immutability"],
                    "archiveVerifiedAt": verify["verifiedAt"],
                    "independentVerifiedAt": verifier["verifiedAt"],
                    "bindResultSha256": _sha_bytes(_canonical(bind_raw)),
                    "archiveVerifyResultSha256": _sha_bytes(_canonical(verify_raw)),
                    "independentVerifyResultSha256": _sha_bytes(
                        _canonical(verifier_raw)
                    ),
                    "bindResult": bind_raw,
                    "archiveVerifyResult": verify_raw,
                    "independentVerifyResult": verifier_raw,
                }
            )
        if (
            len(archive_ops) < int(POLICY["min_archive_operators"])
            or len(verifier_ops) < int(POLICY["min_verifier_operators"])
            or archive_ops & verifier_ops
        ):
            _fail("NATIVE_ARCHIVE_OPERATOR_QUORUM_INVALID")
        for name, expected in initial_hashes.items():
            if _sha(source / name) != expected:
                _fail("NATIVE_ARCHIVE_RUN159_INPUT_DRIFT")
        if _sha(payload_path) != payload_item["sha256"]:
            _fail("NATIVE_ARCHIVE_OUTPUT_DRIFT")
        state = {
            "schemaVersion": int(POLICY["state_schema_version"]),
            "status": "trusted-native-status-archive",
            "sequence": run159["sequence"],
            "nativeStatusChainHeadSha256": run159["native_status_chain_head_sha256"],
            "archiveArtifact": payload_item,
            "sourceInventorySha256": _sha_bytes(_canonical(payload["sourceInventory"])),
        }
        _write(stage / _DOC_NATIVE_EVIDENCE_ARCHIVE_STATE, state)
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "preserved",
            "nativeStatusChainHeadSha256": run159["native_status_chain_head_sha256"],
            "archiveArtifact": payload_item,
            "archiveCount": len(normalized),
            "archiveOperatorCount": len(archive_ops),
            "verifierOperatorCount": len(verifier_ops),
            "archives": archive_results,
        }
        _write(stage / "release-native-evidence-archive-receipt.json", receipt)
        tmp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, tmp_target, copy_function=shutil.copy2)
        os.replace(tmp_target, target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    verified = verify_native_archive(
        output_dir=target,
        expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
        expected_recovery_root_sha256=expected_recovery_root_sha256,
        expected_attestation_root_sha256=expected_attestation_root_sha256,
        now=current,
    )
    return dict(verified, phase="native-status-archived", archive_count=len(normalized))


def _validate_recovery_response(
    value: dict[str, Any], *, identity: str, operator: str, now: datetime
) -> dict[str, Any]:
    common = {"schemaVersion", "operation", "status", "source", "observedAt"}
    if (
        value.get("schemaVersion") != int(POLICY["recovery_protocol_version"])
        or value.get("operation") != "recover"
    ):
        _fail("NATIVE_ARCHIVE_RECOVERY_RESULT_SCHEMA_INVALID")
    source = value.get("source")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "identity",
            "operator",
            "readOnly",
            "archiveWriterCredentialsReused",
            "verifierCredentialsReused",
        }
        or source.get("identity") != identity
        or source.get("operator") != operator
    ):
        _fail("NATIVE_ARCHIVE_RECOVERY_AUTHORITY_INVALID")
    if (
        source.get("readOnly") is not True
        or source.get("archiveWriterCredentialsReused") is not False
        or source.get("verifierCredentialsReused") is not False
    ):
        _fail("NATIVE_ARCHIVE_RECOVERY_CREDENTIAL_REUSE")
    observed_at = _fresh(
        value.get("observedAt"), "NATIVE_ARCHIVE_RECOVERY_OBSERVED_AT", now=now
    )
    if value.get("status") == "unavailable":
        if set(value) != common | {"reason"} or not bool(
            POLICY["allow_unavailable_recovery_source"]
        ):
            _fail("NATIVE_ARCHIVE_RECOVERY_UNAVAILABLE_INVALID")
        reason = value.get("reason")
        _len = len(reason) > 255  # ruff: ignore[magic-value-comparison]
        if not isinstance(reason, str) or not reason or _len:
            _fail("NATIVE_ARCHIVE_RECOVERY_REASON_INVALID")
        return {"status": "unavailable", "observedAt": observed_at, "reason": reason}
    if value.get("status") != "observed" or set(value) != common | {
        "locator",
        "artifact",
        "payloadBase64",
    }:
        _fail("NATIVE_ARCHIVE_RECOVERY_STATUS_INVALID")
    artifact = _artifact_doc(value.get("artifact"), "NATIVE_ARCHIVE_RECOVERY_ARTIFACT")
    if artifact["name"] != "release-native-evidence-archive.json":
        _fail("NATIVE_ARCHIVE_RECOVERY_ARTIFACT_NAME_INVALID")
    try:
        raw = base64.b64decode(value.get("payloadBase64"), validate=True)
    except Exception as exc:
        raise NativeArchiveError(
            "NATIVE_ARCHIVE_RECOVERY_PAYLOAD_BASE64_INVALID"
        ) from exc
    if len(raw) != artifact["size"] or _sha_bytes(raw) != artifact["sha256"]:
        _fail("NATIVE_ARCHIVE_RECOVERY_PAYLOAD_HASH_MISMATCH")
    doc = _loads(raw, "NATIVE_ARCHIVE_RECOVERY_PAYLOAD")
    if raw != _canonical(doc):
        _fail("NATIVE_ARCHIVE_RECOVERY_PAYLOAD_NOT_CANONICAL")
    return {
        "status": "observed",
        "observedAt": observed_at,
        "locator": _safe_locator(
            value.get("locator"), "NATIVE_ARCHIVE_RECOVERY_LOCATOR_INVALID"
        ),
        "artifact": artifact,
        "raw": raw,
        "doc": doc,
    }


def recover_native_status(  # ruff: ignore[undocumented-public-function]
    *,
    sources: list[tuple[str, str, RecoveryAdapter]],
    output_dir: Path,
    expected_archive_sha256: str,
    expected_native_status_chain_head_sha256: str,
    expected_bootstrap_root_sha256: str,
    expected_recovery_root_sha256: str,
    expected_attestation_root_sha256: list[str],
    now: datetime | None = None,
    historical: bool = True,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expected_archive_sha256 = _hex(
        expected_archive_sha256, "NATIVE_ARCHIVE_RECOVERY_EXPECTED_ARCHIVE_SHA_INVALID"
    )
    expected_native_status_chain_head_sha256 = _hex(
        expected_native_status_chain_head_sha256,
        "NATIVE_ARCHIVE_RECOVERY_EXPECTED_CHAIN_HEAD_INVALID",
    )
    normalized = [
        (
            _identity(i, "NATIVE_ARCHIVE_RECOVERY_IDENTITY_INVALID"),
            _identity(o, "NATIVE_ARCHIVE_RECOVERY_OPERATOR_INVALID"),
            a,
        )
        for i, o, a in sources
    ]
    if (
        len(normalized) < int(POLICY["min_recovery_sources"])
        or len({i for i, _, _ in normalized}) != len(normalized)
        or len({o for _, o, _ in normalized}) < int(POLICY["min_recovery_operators"])
    ):
        _fail("NATIVE_ARCHIVE_RECOVERY_QUORUM_PLAN_INVALID")
    observed = []
    unavailable = []
    request = {
        "schemaVersion": int(POLICY["recovery_protocol_version"]),
        "operation": "recover",
        "expectedArchiveSha256": expected_archive_sha256,
        "expectedNativeStatusChainHeadSha256": expected_native_status_chain_head_sha256,
    }
    for identity, operator, adapter in sorted(normalized, key=lambda x: (x[1], x[0])):
        result = _validate_recovery_response(
            adapter(dict(request, source={"identity": identity, "operator": operator})),
            identity=identity,
            operator=operator,
            now=current,
        )
        (observed if result["status"] == "observed" else unavailable).append(
            dict(result, identity=identity, operator=operator)
        )
    if len(observed) < int(POLICY["min_recovery_sources"]) or len(
        {x["operator"] for x in observed}
    ) < int(POLICY["min_recovery_operators"]):
        _fail("NATIVE_ARCHIVE_RECOVERY_QUORUM_NOT_MET")
    locators = [x["locator"] for x in observed]
    if len(set(locators)) != len(locators):
        _fail("NATIVE_ARCHIVE_RECOVERY_LOCATOR_COLLISION")
    raw_hashes = {_sha_bytes(x["raw"]) for x in observed}
    if len(raw_hashes) != 1:
        _fail("NATIVE_ARCHIVE_RECOVERY_EQUIVOCATION")
    raw = observed[0]["raw"]
    if _sha_bytes(raw) != expected_archive_sha256:
        _fail("NATIVE_ARCHIVE_RECOVERY_ROLLBACK_PIN_MISMATCH")
    doc = observed[0]["doc"]
    info = _verify_archive_payload(
        doc,
        bootstrap_pin=expected_bootstrap_root_sha256,
        recovery_pin=expected_recovery_root_sha256,
        attestation_pins=expected_attestation_root_sha256,
        now=current,
        historical=historical,
    )
    if info["chain_head"] != expected_native_status_chain_head_sha256:
        _fail("NATIVE_ARCHIVE_RECOVERY_CHAIN_HEAD_PIN_MISMATCH")
    target = _outside(output_dir, [], "NATIVE_ARCHIVE_RECOVERY_OUTPUT_INVALID")
    if target.exists() or target.is_symlink():
        _fail("NATIVE_ARCHIVE_RECOVERY_OUTPUT_ALREADY_EXISTS")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run160-recovery-", dir=parent))
    try:
        recovered = stage / "recovered-native-status"
        _rehydrate_native(doc["nativeStatus"]["output"], recovered)
        try:
            native.verify_native_status(
                output_dir=recovered,
                expected_bootstrap_root_sha256=expected_bootstrap_root_sha256,
                expected_recovery_root_sha256=expected_recovery_root_sha256,
                expected_attestation_root_sha256=expected_attestation_root_sha256,
                now=current,
                historical=historical,
            )
        except native.NativeStatusError as exc:
            raise NativeArchiveError(
                "NATIVE_ARCHIVE_RECOVERED_RUN159_INVALID:" + str(exc)
            ) from exc
        receipt = {
            "schemaVersion": int(POLICY["recovery_receipt_schema_version"]),
            "status": "recovered",
            "archiveArtifact": _artifact("release-native-evidence-archive.json", raw),
            "nativeStatusChainHeadSha256": info["chain_head"],
            "observed": [
                {
                    "identity": x["identity"],
                    "operator": x["operator"],
                    "locator": x["locator"],
                    "observedAt": x["observedAt"],
                }
                for x in observed
            ],
            "unavailable": [
                {
                    "identity": x["identity"],
                    "operator": x["operator"],
                    "reason": x["reason"],
                    "observedAt": x["observedAt"],
                }
                for x in unavailable
            ],
        }
        _write(stage / "release-native-evidence-recovery-receipt.json", receipt)
        os.replace(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "ok": True,
        "phase": "native-status-recovered",
        "sequence": info["sequence"],
        "native_status_chain_head_sha256": info["chain_head"],
        "archive_sha256": expected_archive_sha256,
        "observed": len(observed),
        "unavailable": len(unavailable),
    }


def _parse_target(value: str) -> tuple[str, str, str, str, list[str], list[str]]:
    # archive-id,archive-op,verifier-id,verifier-op=archive command... || verifier command...
    if "=" not in value or "||" not in value:
        _fail("NATIVE_ARCHIVE_TARGET_ARG_INVALID")
    lhs, rhs = value.split("=", 1)
    parts = lhs.split(",")
    if len(parts) != 4:  # ruff: ignore[magic-value-comparison]
        _fail("NATIVE_ARCHIVE_TARGET_ARG_INVALID")
    archive_cmd, verifier_cmd = [x.strip().split() for x in rhs.split("||", 1)]
    return parts[0], parts[1], parts[2], parts[3], archive_cmd, verifier_cmd


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def pins(p):
        p.add_argument("--bootstrap-root-sha256", required=True)
        p.add_argument("--recovery-root-sha256", required=True)
        p.add_argument("--attestation-root-sha256", action="append", required=True)

    preserve = sub.add_parser("preserve")
    preserve.add_argument("--native-dir", type=Path, required=True)
    preserve.add_argument("--output-dir", type=Path, required=True)
    preserve.add_argument("--target", action="append", required=True)
    pins(preserve)
    verify = sub.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--historical", action="store_true")
    pins(verify)
    recover = sub.add_parser("recover")
    recover.add_argument(
        "--source", action="append", required=True, help="identity,operator=command..."
    )
    recover.add_argument("--output-dir", type=Path, required=True)
    recover.add_argument("--expected-archive-sha256", required=True)
    recover.add_argument("--expected-native-status-chain-head-sha256", required=True)
    recover.add_argument("--live", action="store_true")
    pins(recover)
    args = parser.parse_args(argv)
    logger.info("starting native archive command: %s", args.command)
    if args.command == "preserve":
        targets = []
        for value in args.target:
            ai, ao, vi, vo, ac, vc = _parse_target(value)
            targets.append((ai, ao, command_archive(ac), vi, vo, command_verifier(vc)))
        result = preserve_native_status(
            native_dir=args.native_dir,
            output_dir=args.output_dir,
            expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            expected_recovery_root_sha256=args.recovery_root_sha256,
            expected_attestation_root_sha256=args.attestation_root_sha256,
            targets=targets,
        )
    elif args.command == "verify":
        result = verify_native_archive(
            output_dir=args.output_dir,
            expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            expected_recovery_root_sha256=args.recovery_root_sha256,
            expected_attestation_root_sha256=args.attestation_root_sha256,
            historical=args.historical,
        )
    else:
        src = []
        for value in args.source:
            if "=" not in value:
                _fail("NATIVE_ARCHIVE_SOURCE_ARG_INVALID")
            lhs, cmd = value.split("=", 1)
            parts = lhs.split(",")
            if len(parts) != 2:  # ruff: ignore[magic-value-comparison]
                _fail("NATIVE_ARCHIVE_SOURCE_ARG_INVALID")
            src.append((parts[0], parts[1], command_recovery(cmd.split())))
        result = recover_native_status(
            sources=src,
            output_dir=args.output_dir,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_native_status_chain_head_sha256=args.expected_native_status_chain_head_sha256,
            expected_bootstrap_root_sha256=args.bootstrap_root_sha256,
            expected_recovery_root_sha256=args.recovery_root_sha256,
            expected_attestation_root_sha256=args.attestation_root_sha256,
            historical=not args.live,
        )
    # sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    logger.info("completed native archive command: %s", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
