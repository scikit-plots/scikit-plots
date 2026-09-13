"""
Independently attest, sign-verify, and bind a Run 150 publication to its final release record.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Callable

import tomllib

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_transparency_policy.toml").read_text())
PUBLICATION_PREDICATE_TYPE = str(POLICY["publication_predicate_type"])
ATTESTATION_PREDICATE_TYPE = str(POLICY["attestation_predicate_type"])
FINAL_RECORD_PREDICATE_TYPE = str(POLICY["final_record_predicate_type"])
RECORD_NAME = str(POLICY["record_name"])
IN_TOTO_STATEMENT = "https://in-toto.io/Statement/v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_CHUNK = 1024 * 1024


class TransparencyError(RuntimeError):
    """Post-publication transparency processing violated a fail-closed invariant."""


def _fail(code: str) -> None:
    raise TransparencyError(code)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(_canonical_bytes(value))


def _loads_json(raw: bytes, code: str) -> dict[str, Any]:
    def object_pairs(pairs):
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                _fail(code + "_DUPLICATE_KEY")
            out[key] = value
        return out

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    except TransparencyError:
        raise
    except Exception as exc:
        raise TransparencyError(code + "_JSON_INVALID") from exc
    if not isinstance(value, dict):
        _fail(code + "_SCHEMA_INVALID")
    return value


def _utc(value: datetime | None = None) -> str:
    now = (value or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_time(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        out = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail(code)
    return out.astimezone(timezone.utc)


def _fresh_time(value: Any, code: str, *, now: datetime) -> str:
    parsed = _parse_time(value, code + "_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    age = timedelta(minutes=int(POLICY["max_result_age_minutes"]))
    if parsed > now + skew:
        _fail(code + "_FROM_FUTURE")
    if now - parsed > age:
        _fail(code + "_STALE")
    return value


def _safe_name(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        _fail(code)
    if Path(value).name != value or "/" in value or "\\" in value or "\x00" in value:
        _fail(code)
    if any(
        ord(ch) < 32  # ruff: ignore[magic-value-comparison]
        or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in value
    ):
        _fail(code)
    return value


def _safe_id(value: Any, code: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None or ".." in value:
        _fail(code)
    return value


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
    if any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in value  # lint
    ):
        _fail(code)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}", value) is None:
        _fail(code)
    return value


def _bounded_text(
    value: Any, code: str, *, limit: int = 1024, allow_at: bool = False
) -> str:
    if not isinstance(value, str):
        _fail(code)
    value = value.strip()
    if not value or len(value) > limit:
        _fail(code)
    if any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in value  # lint
    ):
        _fail(code)
    forbidden = ("?", "#") if allow_at else ("?", "#", "@")
    if any(marker in value for marker in forbidden):
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


def _regular_file(path: Path, *, max_bytes: int, code: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_file():
        _fail(code)
    resolved = raw.resolve()
    if resolved.stat().st_size > max_bytes:
        _fail(code + "_TOO_LARGE")
    return resolved


def _regular_dir(path: Path, code: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_dir():
        _fail(code)
    return raw.resolve()


def _outside(path: Path, protected: tuple[Path, ...], code: str) -> Path:
    target = path.expanduser().resolve()
    for item in protected:
        root = item.resolve()
        if target == root or root in target.parents:
            _fail(code)
    return target


def _read_json_exact(
    path: Path, *, max_bytes: int, code: str, canonical: bool = True
) -> tuple[dict[str, Any], bytes]:
    path = _regular_file(path, max_bytes=max_bytes, code=code)
    raw = path.read_bytes()
    doc = _loads_json(raw, code)
    if canonical and raw != _canonical_bytes(doc):
        _fail(code + "_NOT_CANONICAL")
    return doc, raw


def _artifact(value: Any, code: str, *, remote: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code + "_SCHEMA_INVALID")
    expected = {"name", "sha256", "size"}
    if remote:
        expected |= {"locator", "immutability"}
    if set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    out = {
        "name": _safe_name(value.get("name"), code + "_NAME_INVALID"),
        "sha256": _hex(value.get("sha256"), code + "_SHA256_INVALID"),
        "size": _size(value.get("size"), code + "_SIZE_INVALID"),
    }
    if remote:
        out["locator"] = _bounded_text(value.get("locator"), code + "_LOCATOR_INVALID")
        immutability = value.get("immutability")
        if immutability not in set(POLICY["allowed_immutability"]):
            _fail(code + "_IMMUTABILITY_INVALID")
        out["immutability"] = immutability
    return out


def _validate_publisher_result(
    path: Path,
    *,
    operation: str,
    expected: dict[str, Any],
    publication_id: str,
    publisher: str,
    target_id: str,
    locator: str,
    immutability: str,
    expected_sha: str,
) -> None:
    doc, raw = _read_json_exact(
        path,
        max_bytes=int(POLICY["max_publisher_result_bytes"]),
        code="PUBLICATION_PUBLISHER_RESULT",
        canonical=True,
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        _fail("PUBLICATION_PUBLISHER_RESULT_HASH_MISMATCH")
    keys = {
        "schemaVersion",
        "operation",
        "publicationId",
        "status",
        "target",
        "artifact",
        "guarantees",
        "verifiedAt",
    }
    if (
        set(doc) != keys
        or doc.get("schemaVersion") != 1
        or doc.get("operation") != operation
    ):
        _fail("PUBLICATION_PUBLISHER_RESULT_SCHEMA_INVALID")
    if doc.get("publicationId") != publication_id:
        _fail("PUBLICATION_PUBLISHER_RESULT_ID_MISMATCH")
    allowed_status = {"created", "present"} if operation == "publish" else {"present"}
    if doc.get("status") not in allowed_status:
        _fail("PUBLICATION_PUBLISHER_RESULT_STATUS_INVALID")
    target = doc.get("target")
    if not isinstance(target, dict) or set(target) != {
        "publisher",
        "targetId",
        "locator",
    }:
        _fail("PUBLICATION_PUBLISHER_RESULT_TARGET_SCHEMA_INVALID")
    if (
        target.get("publisher") != publisher
        or target.get("targetId") != target_id
        or target.get("locator") != locator
    ):
        _fail("PUBLICATION_PUBLISHER_RESULT_TARGET_MISMATCH")
    if doc.get("artifact") != {
        "name": expected["name"],
        "sha256": expected["sha256"],
        "size": expected["size"],
    }:
        _fail("PUBLICATION_PUBLISHER_RESULT_ARTIFACT_MISMATCH")
    guarantees = doc.get("guarantees")
    if not isinstance(guarantees, dict) or set(guarantees) != {
        "createOnly",
        "overwrite",
        "remoteReadbackVerified",
        "immutability",
    }:
        _fail("PUBLICATION_PUBLISHER_RESULT_GUARANTEES_SCHEMA_INVALID")
    if guarantees != {
        "createOnly": True,
        "overwrite": False,
        "remoteReadbackVerified": True,
        "immutability": immutability,
    }:
        _fail("PUBLICATION_PUBLISHER_RESULT_GUARANTEES_INVALID")
    _parse_time(doc.get("verifiedAt"), "PUBLICATION_PUBLISHER_RESULT_TIME_INVALID")


def _validate_publication(  # ruff: ignore[too-many-branches]
    root: Path,
) -> dict[str, Any]:
    root = _regular_dir(root, "PUBLICATION_DIRECTORY_INVALID")
    transparency, transparency_raw = _read_json_exact(
        root / "publication-transparency.json",
        max_bytes=int(POLICY["max_publication_json_bytes"]),
        code="PUBLICATION_TRANSPARENCY",
        canonical=True,
    )
    receipt, receipt_raw = _read_json_exact(
        root / "publication-receipt.json",
        max_bytes=int(POLICY["max_publication_json_bytes"]),
        code="PUBLICATION_RECEIPT",
        canonical=True,
    )
    transparency_sha = hashlib.sha256(transparency_raw).hexdigest()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()

    transparency_keys = {
        "schemaVersion",
        "predicateType",
        "generatedAt",
        "publicationId",
        "release",
        "subject",
        "target",
        "artifacts",
        "verification",
    }
    if (
        set(transparency) != transparency_keys
        or transparency.get("schemaVersion") != 1
        or transparency.get("predicateType") != PUBLICATION_PREDICATE_TYPE
    ):
        _fail("PUBLICATION_TRANSPARENCY_SCHEMA_INVALID")
    _parse_time(
        transparency.get("generatedAt"), "PUBLICATION_TRANSPARENCY_TIME_INVALID"
    )
    publication_id = _safe_id(
        transparency.get("publicationId"), "PUBLICATION_ID_INVALID"
    )
    release = transparency.get("release")
    if not isinstance(release, dict) or set(release) != {"releaseId", "sourceRevision"}:
        _fail("PUBLICATION_RELEASE_SCHEMA_INVALID")
    release_id = release.get("releaseId")
    if not isinstance(release_id, str) or _RELEASE_ID.fullmatch(release_id) is None:
        _fail("PUBLICATION_RELEASE_ID_INVALID")
    revision = release.get("sourceRevision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        _fail("PUBLICATION_SOURCE_REVISION_INVALID")
    subject = transparency.get("subject")
    if not isinstance(subject, dict) or set(subject) != {
        "promotionReceiptSha256",
        "sourceTreeSha256",
        "evidenceSha256",
        "releaseStatementSha256",
    }:
        _fail("PUBLICATION_SUBJECT_SCHEMA_INVALID")
    for field in (
        "promotionReceiptSha256",
        "sourceTreeSha256",
        "evidenceSha256",
        "releaseStatementSha256",
    ):
        _hex(subject.get(field), "PUBLICATION_SUBJECT_HASH_INVALID")
    target = transparency.get("target")
    if not isinstance(target, dict) or set(target) != {"publisher", "targetId"}:
        _fail("PUBLICATION_TARGET_SCHEMA_INVALID")
    publisher = _safe_id(target.get("publisher"), "PUBLICATION_PUBLISHER_INVALID")
    target_id = _safe_id(target.get("targetId"), "PUBLICATION_TARGET_ID_INVALID")
    verification = transparency.get("verification")
    if verification != {
        "promotionReceiptVerified": True,
        "localSnapshotVerified": True,
        "createOnly": True,
        "remoteReadbackVerified": True,
        "allReceiptObjectsPublished": True,
    }:
        _fail("PUBLICATION_VERIFICATION_INVALID")

    values = transparency.get("artifacts")
    if (
        not isinstance(values, list)
        or not values
        or len(values) > int(POLICY["max_artifacts"])
    ):
        _fail("PUBLICATION_ARTIFACTS_INVALID")
    artifacts: list[dict[str, Any]] = []
    locators: set[str] = set()
    names: set[str] = set()
    results_dir = root / "publisher-results"
    if results_dir.is_symlink() or not results_dir.is_dir():
        _fail("PUBLICATION_PUBLISHER_RESULTS_DIRECTORY_INVALID")
    expected_result_names: set[str] = set()
    for index, value in enumerate(values, 1):
        if not isinstance(value, dict) or set(value) != {
            "name",
            "sha256",
            "size",
            "remote",
            "publisherEvidence",
        }:
            _fail("PUBLICATION_ARTIFACT_SCHEMA_INVALID")
        item = {
            "name": _safe_name(value.get("name"), "PUBLICATION_ARTIFACT_NAME_INVALID"),
            "sha256": _hex(value.get("sha256"), "PUBLICATION_ARTIFACT_SHA256_INVALID"),
            "size": _size(value.get("size"), "PUBLICATION_ARTIFACT_SIZE_INVALID"),
        }
        if item["name"] in names:
            _fail("PUBLICATION_ARTIFACT_DUPLICATE_NAME")
        names.add(item["name"])
        remote = value.get("remote")
        if not isinstance(remote, dict) or set(remote) != {
            "locator",
            "sha256",
            "size",
            "immutability",
            "verifiedAt",
        }:
            _fail("PUBLICATION_REMOTE_SCHEMA_INVALID")
        locator = _bounded_text(
            remote.get("locator"), "PUBLICATION_REMOTE_LOCATOR_INVALID"
        )
        if locator in locators:
            _fail("PUBLICATION_REMOTE_LOCATOR_COLLISION")
        locators.add(locator)
        immutability = remote.get("immutability")
        if immutability not in set(POLICY["allowed_immutability"]):
            _fail("PUBLICATION_REMOTE_IMMUTABILITY_INVALID")
        if remote.get("sha256") != item["sha256"] or remote.get("size") != item["size"]:
            _fail("PUBLICATION_REMOTE_SUBJECT_MISMATCH")
        _parse_time(remote.get("verifiedAt"), "PUBLICATION_REMOTE_TIME_INVALID")
        publisher_evidence = value.get("publisherEvidence")
        if not isinstance(publisher_evidence, dict) or set(publisher_evidence) != {
            "publishSha256",
            "verifySha256",
        }:
            _fail("PUBLICATION_PUBLISHER_EVIDENCE_SCHEMA_INVALID")
        publish_sha = _hex(
            publisher_evidence.get("publishSha256"),
            "PUBLICATION_PUBLISH_EVIDENCE_HASH_INVALID",
        )
        verify_sha = _hex(
            publisher_evidence.get("verifySha256"),
            "PUBLICATION_VERIFY_EVIDENCE_HASH_INVALID",
        )
        pub_name = f"{index:02d}-{item['name']}.publish.json"
        ver_name = f"{index:02d}-{item['name']}.verify.json"
        expected_result_names.update({pub_name, ver_name})
        _validate_publisher_result(
            results_dir / pub_name,
            operation="publish",
            expected=item,
            publication_id=publication_id,
            publisher=publisher,
            target_id=target_id,
            locator=locator,
            immutability=immutability,
            expected_sha=publish_sha,
        )
        _validate_publisher_result(
            results_dir / ver_name,
            operation="verify",
            expected=item,
            publication_id=publication_id,
            publisher=publisher,
            target_id=target_id,
            locator=locator,
            immutability=immutability,
            expected_sha=verify_sha,
        )
        artifacts.append({**item, "locator": locator, "immutability": immutability})
    actual_result_names = {p.name for p in results_dir.iterdir()}
    if actual_result_names != expected_result_names or any(
        p.is_symlink() or not p.is_file() for p in results_dir.iterdir()
    ):
        _fail("PUBLICATION_PUBLISHER_RESULTS_ALLOWLIST_MISMATCH")

    receipt_keys = {
        "schemaVersion",
        "status",
        "completedAt",
        "publicationId",
        "releaseId",
        "sourceRevision",
        "promotionReceiptSha256",
        "transparencySha256",
        "target",
        "artifacts",
    }
    if (
        set(receipt) != receipt_keys
        or receipt.get("schemaVersion") != 1
        or receipt.get("status") != "published"
    ):
        _fail("PUBLICATION_RECEIPT_SCHEMA_INVALID")
    _parse_time(receipt.get("completedAt"), "PUBLICATION_RECEIPT_TIME_INVALID")
    if (
        receipt.get("publicationId") != publication_id
        or receipt.get("releaseId") != release_id
        or receipt.get("sourceRevision") != revision
    ):
        _fail("PUBLICATION_RECEIPT_IDENTITY_MISMATCH")
    if (
        receipt.get("promotionReceiptSha256") != subject["promotionReceiptSha256"]
        or receipt.get("transparencySha256") != transparency_sha
    ):
        _fail("PUBLICATION_RECEIPT_SUBJECT_MISMATCH")
    if receipt.get("target") != {"publisher": publisher, "targetId": target_id}:
        _fail("PUBLICATION_RECEIPT_TARGET_MISMATCH")
    receipt_artifacts = receipt.get("artifacts")
    if not isinstance(receipt_artifacts, list) or len(receipt_artifacts) != len(
        artifacts
    ):
        _fail("PUBLICATION_RECEIPT_ARTIFACTS_INVALID")
    normalized_receipt = [
        _artifact(item, "PUBLICATION_RECEIPT_ARTIFACT", remote=True)
        for item in receipt_artifacts
    ]
    if normalized_receipt != artifacts:
        _fail("PUBLICATION_RECEIPT_ARTIFACT_MISMATCH")

    root_entries = {p.name for p in root.iterdir()}
    if root_entries != {
        "publication-transparency.json",
        "publication-receipt.json",
        "publisher-results",
    }:
        _fail("PUBLICATION_DIRECTORY_ALLOWLIST_MISMATCH")
    return {
        "root": root,
        "transparency": transparency,
        "receipt": receipt,
        "transparency_sha256": transparency_sha,
        "receipt_sha256": receipt_sha,
        "publication_id": publication_id,
        "release_id": release_id,
        "source_revision": revision,
        "publisher": publisher,
        "target_id": target_id,
        "promotion_receipt_sha256": subject["promotionReceiptSha256"],
        "artifacts": artifacts,
    }


def _verification_id(transparency_sha: str, verifier_identity: str) -> str:
    h = hashlib.sha256()
    h.update(b"release-independent-verification-v1\0")
    for value in (transparency_sha, verifier_identity):
        data = value.encode("utf-8")
        h.update(len(data).to_bytes(4, "big"))
        h.update(data)
    return "verify-" + h.hexdigest()[:32]


def _binding_id(
    publication_id: str, record_sha: str, publisher: str, target_id: str
) -> str:
    h = hashlib.sha256()
    h.update(b"release-final-record-binding-v1\0")
    for value in (publication_id, record_sha, publisher, target_id):
        data = value.encode("utf-8")
        h.update(len(data).to_bytes(4, "big"))
        h.update(data)
    return "bind-" + h.hexdigest()[:32]


Verifier = Callable[[dict[str, Any]], dict[str, Any]]
Binder = Callable[[dict[str, Any]], dict[str, Any]]


def _command_adapter(
    command: list[str], *, prefix: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if not command or any(not isinstance(part, str) or not part for part in command):
        _fail(prefix + "_COMMAND_INVALID")
    executable = (
        shutil.which(command[0]) if not os.path.isabs(command[0]) else command[0]
    )
    if not executable or not Path(executable).is_file():
        _fail(prefix + "_COMMAND_UNAVAILABLE")
    argv = [executable, *command[1:]]

    def invoke(request: dict[str, Any]) -> dict[str, Any]:
        payload = _canonical_bytes(request)
        limit = int(POLICY["max_adapter_output_bytes"])
        proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            argv,
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
                        with contextlib.suppress(OSError):
                            proc.kill()
                        return
                    chunks.append(chunk)
            # defensive pipe failure
            except (
                BaseException  # pragma: no cover  # ruff: ignore[blind-except]
            ) as exc:
                state["error"] = exc
                try:  # ruff: ignore[suppressible-exception]
                    proc.kill()
                except OSError:
                    pass

        reader = threading.Thread(
            target=drain, name=prefix.lower() + "-stdout", daemon=True
        )
        reader.start()
        try:
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
                proc.wait(timeout=int(POLICY["adapter_timeout_seconds"]))
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.wait()
                reader.join(timeout=2)
                raise TransparencyError(prefix + "_COMMAND_TIMEOUT") from exc
            finally:
                try:  # ruff: ignore[suppressible-exception]
                    proc.stdin.close()
                except OSError:
                    pass
            reader.join(timeout=2)
            if reader.is_alive() or state["error"] is not None:
                proc.kill()
                _fail(prefix + "_COMMAND_OUTPUT_DRAIN_FAILED")
            if state["overflow"]:
                _fail(prefix + "_COMMAND_OUTPUT_TOO_LARGE")
            if proc.returncode != 0:
                _fail(prefix + "_COMMAND_FAILED")
            return _loads_json(b"".join(chunks), prefix + "_COMMAND_OUTPUT")
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

    return invoke


def command_verifier(  # ruff: ignore[undocumented-public-function]
    command: list[str],
) -> Verifier:
    return _command_adapter(command, prefix="INDEPENDENT_VERIFIER")


def command_binder(  # ruff: ignore[undocumented-public-function]
    command: list[str],
) -> Binder:
    return _command_adapter(command, prefix="RELEASE_BINDER")


def _validate_verifier_response(
    value: Any,
    *,
    item: dict[str, Any],
    info: dict[str, Any],
    verifier_identity: str,
    verification_id: str,
    now: datetime,
) -> dict[str, Any]:
    keys = {
        "schemaVersion",
        "operation",
        "verificationId",
        "publicationId",
        "status",
        "verifier",
        "target",
        "artifact",
        "remote",
        "verifiedAt",
    }
    if not isinstance(value, dict) or set(value) != keys:
        _fail("INDEPENDENT_VERIFIER_RESPONSE_SCHEMA_INVALID")
    if (
        value.get("schemaVersion") != int(POLICY["verifier_protocol_version"])
        or value.get("operation") != "verify"
    ):
        _fail("INDEPENDENT_VERIFIER_RESPONSE_VERSION_INVALID")
    if (
        value.get("verificationId") != verification_id
        or value.get("publicationId") != info["publication_id"]
        or value.get("status") != "present"
    ):
        _fail("INDEPENDENT_VERIFIER_RESPONSE_IDENTITY_MISMATCH")
    verifier = value.get("verifier")
    if not isinstance(verifier, dict) or set(verifier) != {
        "identity",
        "readOnly",
        "publisherCredentialsReused",
    }:
        _fail("INDEPENDENT_VERIFIER_AUTHORITY_SCHEMA_INVALID")
    if (
        _identity(verifier.get("identity"), "INDEPENDENT_VERIFIER_IDENTITY_INVALID")
        != verifier_identity
    ):
        _fail("INDEPENDENT_VERIFIER_IDENTITY_MISMATCH")
    if (
        verifier.get("readOnly") is not True
        or verifier.get("publisherCredentialsReused") is not False
    ):
        _fail("INDEPENDENT_VERIFIER_AUTHORITY_INVALID")
    target = value.get("target")
    if target != {"publisher": info["publisher"], "targetId": info["target_id"]}:
        _fail("INDEPENDENT_VERIFIER_TARGET_MISMATCH")
    if value.get("artifact") != {
        "name": item["name"],
        "sha256": item["sha256"],
        "size": item["size"],
    }:
        _fail("INDEPENDENT_VERIFIER_ARTIFACT_MISMATCH")
    remote = value.get("remote")
    expected_remote = {
        "locator": item["locator"],
        "sha256": item["sha256"],
        "size": item["size"],
        "immutability": item["immutability"],
    }
    if remote != expected_remote:
        _fail("INDEPENDENT_VERIFIER_REMOTE_MISMATCH")
    verified_at = _fresh_time(
        value.get("verifiedAt"), "INDEPENDENT_VERIFIER_TIME", now=now
    )
    return {"verifiedAt": verified_at, "remote": expected_remote}


def _verify_remote_set(
    *,
    info: dict[str, Any],
    verifier_identity: str,
    adapter: Verifier,
    output_dir: Path,
    now: datetime,
) -> tuple[str, list[dict[str, Any]]]:
    verification_id = _verification_id(info["transparency_sha256"], verifier_identity)
    output_dir.mkdir(parents=True, exist_ok=False)
    out: list[dict[str, Any]] = []
    for index, item in enumerate(info["artifacts"], 1):
        request = {
            "schemaVersion": int(POLICY["verifier_protocol_version"]),
            "operation": "verify",
            "verificationId": verification_id,
            "publicationId": info["publication_id"],
            "releaseId": info["release_id"],
            "publicationTransparencySha256": info["transparency_sha256"],
            "verifier": {
                "identity": verifier_identity,
                "requiredAuthority": {
                    "readOnly": True,
                    "publisherCredentialsReused": False,
                },
            },
            "target": {"publisher": info["publisher"], "targetId": info["target_id"]},
            "artifact": {
                "name": item["name"],
                "sha256": item["sha256"],
                "size": item["size"],
                "locator": item["locator"],
                "immutability": item["immutability"],
            },
        }
        raw = adapter(request)
        normalized = _validate_verifier_response(
            raw,
            item=item,
            info=info,
            verifier_identity=verifier_identity,
            verification_id=verification_id,
            now=now,
        )
        path = output_dir / f"{index:02d}-{item['name']}.verify.json"
        _write_canonical(path, raw)
        out.append(
            {
                "name": item["name"],
                "sha256": item["sha256"],
                "size": item["size"],
                "locator": item["locator"],
                "immutability": item["immutability"],
                "verifiedAt": normalized["verifiedAt"],
                "verifierEvidenceSha256": _sha256(path),
            }
        )
    return verification_id, out


def prepare_attestation(  # ruff: ignore[undocumented-public-function]
    *,
    publication_dir: Path,
    output_dir: Path,
    verifier_identity: str,
    verifier: Verifier,
    now: datetime | None = None,
) -> dict[str, Any]:
    publication_dir = _regular_dir(publication_dir, "PUBLICATION_DIRECTORY_INVALID")
    target = _outside(
        output_dir, (publication_dir,), "ATTESTATION_OUTPUT_INSIDE_PUBLICATION"
    )
    if target.exists() or target.is_symlink():
        _fail("ATTESTATION_OUTPUT_ALREADY_EXISTS")
    verifier_identity = _identity(
        verifier_identity, "INDEPENDENT_VERIFIER_IDENTITY_INVALID"
    )
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    info = _validate_publication(publication_dir)
    if verifier_identity == info["publisher"]:
        _fail("INDEPENDENT_VERIFIER_NOT_DISTINCT")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-post-publication-", dir=target.parent
    ) as temp_raw:
        stage = Path(temp_raw) / "prepared"
        stage.mkdir()
        verification_id, results = _verify_remote_set(
            info=info,
            verifier_identity=verifier_identity,
            adapter=verifier,
            output_dir=stage / "independent-verifier-results",
            now=current,
        )
        after = _validate_publication(publication_dir)
        if (
            after["transparency_sha256"] != info["transparency_sha256"]
            or after["receipt_sha256"] != info["receipt_sha256"]
            or after["artifacts"] != info["artifacts"]
        ):
            _fail("PUBLICATION_CHANGED_DURING_ATTESTATION")
        attestation = {
            "_type": IN_TOTO_STATEMENT,
            "subject": [
                {
                    "name": "publication-transparency.json",
                    "digest": {"sha256": info["transparency_sha256"]},
                }
            ],
            "predicateType": ATTESTATION_PREDICATE_TYPE,
            "predicate": {
                "schemaVersion": int(POLICY["attestation_schema_version"]),
                "generatedAt": _utc(current),
                "publication": {
                    "publicationId": info["publication_id"],
                    "publicationReceiptSha256": info["receipt_sha256"],
                    "promotionReceiptSha256": info["promotion_receipt_sha256"],
                },
                "release": {
                    "releaseId": info["release_id"],
                    "sourceRevision": info["source_revision"],
                },
                "target": {
                    "publisher": info["publisher"],
                    "targetId": info["target_id"],
                },
                "independentVerifier": {
                    "identity": verifier_identity,
                    "verificationId": verification_id,
                    "readOnly": True,
                    "publisherCredentialsReused": False,
                },
                "artifacts": results,
                "verification": {
                    "publicationEvidenceVerified": True,
                    "publisherEvidenceRebound": True,
                    "independentRemoteReadbackVerified": True,
                    "allPublishedObjectsPresent": True,
                },
            },
        }
        attestation_path = stage / "publication-attestation.json"
        _write_canonical(attestation_path, attestation)
        tmp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, tmp_target, copy_function=shutil.copy2)
        os.replace(tmp_target, target)
    return {
        "ok": True,
        "phase": "attestation-prepared",
        "publication_id": info["publication_id"],
        "publication_transparency_sha256": info["transparency_sha256"],
        "attestation_sha256": _sha256(target / "publication-attestation.json"),
        "verification_id": verification_id,
        "artifact_count": len(info["artifacts"]),
    }


def _validate_attestation(  # ruff: ignore[too-many-branches]
    path: Path,
    *,
    info: dict[str, Any],
    prepared_dir: Path | None = None,
) -> dict[str, Any]:
    doc, raw = _read_json_exact(
        path,
        max_bytes=int(POLICY["max_publication_json_bytes"]),
        code="POST_PUBLICATION_ATTESTATION",
        canonical=True,
    )
    if (
        set(doc) != {"_type", "subject", "predicateType", "predicate"}
        or doc.get("_type") != IN_TOTO_STATEMENT
        or doc.get("predicateType") != ATTESTATION_PREDICATE_TYPE
    ):
        _fail("POST_PUBLICATION_ATTESTATION_SCHEMA_INVALID")
    if doc.get("subject") != [
        {
            "name": "publication-transparency.json",
            "digest": {"sha256": info["transparency_sha256"]},
        }
    ]:
        _fail("POST_PUBLICATION_ATTESTATION_SUBJECT_MISMATCH")
    predicate = doc.get("predicate")
    if not isinstance(predicate, dict) or set(predicate) != {
        "schemaVersion",
        "generatedAt",
        "publication",
        "release",
        "target",
        "independentVerifier",
        "artifacts",
        "verification",
    }:
        _fail("POST_PUBLICATION_ATTESTATION_PREDICATE_SCHEMA_INVALID")
    if predicate.get("schemaVersion") != int(POLICY["attestation_schema_version"]):
        _fail("POST_PUBLICATION_ATTESTATION_VERSION_INVALID")
    generated = _parse_time(
        predicate.get("generatedAt"), "POST_PUBLICATION_ATTESTATION_TIME_INVALID"
    )
    publication = predicate.get("publication")
    expected_publication = {
        "publicationId": info["publication_id"],
        "publicationReceiptSha256": info["receipt_sha256"],
        "promotionReceiptSha256": info["promotion_receipt_sha256"],
    }
    if publication != expected_publication:
        _fail("POST_PUBLICATION_ATTESTATION_PUBLICATION_MISMATCH")
    if predicate.get("release") != {
        "releaseId": info["release_id"],
        "sourceRevision": info["source_revision"],
    }:
        _fail("POST_PUBLICATION_ATTESTATION_RELEASE_MISMATCH")
    if predicate.get("target") != {
        "publisher": info["publisher"],
        "targetId": info["target_id"],
    }:
        _fail("POST_PUBLICATION_ATTESTATION_TARGET_MISMATCH")
    verifier = predicate.get("independentVerifier")
    if not isinstance(verifier, dict) or set(verifier) != {
        "identity",
        "verificationId",
        "readOnly",
        "publisherCredentialsReused",
    }:
        _fail("POST_PUBLICATION_ATTESTATION_VERIFIER_SCHEMA_INVALID")
    verifier_identity = _identity(
        verifier.get("identity"),
        "POST_PUBLICATION_ATTESTATION_VERIFIER_IDENTITY_INVALID",
    )
    if (
        verifier_identity == info["publisher"]
        or verifier.get("readOnly") is not True
        or verifier.get("publisherCredentialsReused") is not False
    ):
        _fail("POST_PUBLICATION_ATTESTATION_VERIFIER_AUTHORITY_INVALID")
    expected_verification_id = _verification_id(
        info["transparency_sha256"], verifier_identity
    )
    if verifier.get("verificationId") != expected_verification_id:
        _fail("POST_PUBLICATION_ATTESTATION_VERIFICATION_ID_MISMATCH")
    if predicate.get("verification") != {
        "publicationEvidenceVerified": True,
        "publisherEvidenceRebound": True,
        "independentRemoteReadbackVerified": True,
        "allPublishedObjectsPresent": True,
    }:
        _fail("POST_PUBLICATION_ATTESTATION_VERIFICATION_INVALID")
    values = predicate.get("artifacts")
    if not isinstance(values, list) or len(values) != len(info["artifacts"]):
        _fail("POST_PUBLICATION_ATTESTATION_ARTIFACTS_INVALID")
    normalized: list[dict[str, Any]] = []
    for index, (value, expected) in enumerate(zip(values, info["artifacts"]), 1):
        keys = {
            "name",
            "sha256",
            "size",
            "locator",
            "immutability",
            "verifiedAt",
            "verifierEvidenceSha256",
        }
        if not isinstance(value, dict) or set(value) != keys:
            _fail("POST_PUBLICATION_ATTESTATION_ARTIFACT_SCHEMA_INVALID")
        invariant = {
            key: value[key]
            for key in ("name", "sha256", "size", "locator", "immutability")
        }
        if invariant != expected:
            _fail("POST_PUBLICATION_ATTESTATION_ARTIFACT_MISMATCH")
        _parse_time(
            value.get("verifiedAt"),
            "POST_PUBLICATION_ATTESTATION_ARTIFACT_TIME_INVALID",
        )
        evidence_sha = _hex(
            value.get("verifierEvidenceSha256"),
            "POST_PUBLICATION_ATTESTATION_EVIDENCE_HASH_INVALID",
        )
        if prepared_dir is not None:
            evidence_path = (
                prepared_dir
                / "independent-verifier-results"
                / f"{index:02d}-{expected['name']}.verify.json"
            )
            if (
                _sha256(
                    _regular_file(
                        evidence_path,
                        max_bytes=int(POLICY["max_adapter_output_bytes"]),
                        code="POST_PUBLICATION_VERIFIER_EVIDENCE_INVALID",
                    )
                )
                != evidence_sha
            ):
                _fail("POST_PUBLICATION_ATTESTATION_EVIDENCE_HASH_MISMATCH")
        normalized.append(dict(value))
    return {
        "doc": doc,
        "raw": raw,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "generated": generated,
        "verifier_identity": verifier_identity,
        "verification_id": expected_verification_id,
        "artifacts": normalized,
    }


def write_signature_verification_record(  # ruff: ignore[undocumented-public-function]
    *,
    publication_dir: Path,
    attestation: Path,
    verifier_evidence: Path,
    output: Path,
    signer_identity: str,
    verifier_name: str,
    verifier_version: str,
    verified_at: datetime | None = None,
) -> Path:
    info = _validate_publication(publication_dir)
    att = _validate_attestation(attestation, info=info)
    verifier_evidence = _regular_file(
        verifier_evidence,
        max_bytes=int(POLICY["max_external_verifier_evidence_bytes"]),
        code="SIGNATURE_VERIFIER_EVIDENCE_INVALID",
    )
    signer_identity = _identity(signer_identity, "SIGNER_IDENTITY_INVALID")
    for value, code in (
        (verifier_name, "SIGNATURE_VERIFIER_NAME_INVALID"),
        (verifier_version, "SIGNATURE_VERIFIER_VERSION_INVALID"),
    ):
        _bounded_text(value, code, limit=128, allow_at=True)
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _fail("SIGNATURE_RECORD_OUTPUT_EXISTS")
    current = (verified_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if att["generated"] > current + skew:
        _fail("POST_PUBLICATION_ATTESTATION_FROM_FUTURE")
    if current - att["generated"] > timedelta(
        hours=int(POLICY["max_attestation_age_hours"])
    ):
        _fail("POST_PUBLICATION_ATTESTATION_STALE")
    if current + skew < att["generated"]:
        _fail("SIGNATURE_RECORD_PREDATES_ATTESTATION")
    payload = {
        "schemaVersion": int(POLICY["signature_record_schema_version"]),
        "verified": True,
        "verifiedAt": _utc(current),
        "publicationId": info["publication_id"],
        "publicationTransparencySha256": info["transparency_sha256"],
        "attestationSha256": att["sha256"],
        "signerIdentity": signer_identity,
        "signerIdentityVerified": True,
        "verifierEvidenceSha256": _sha256(verifier_evidence),
        "verifier": {
            "name": verifier_name.strip(),
            "version": verifier_version.strip(),
        },
    }
    _write_canonical(target, payload)
    return target


def _validate_signature_record(
    path: Path,
    *,
    info: dict[str, Any],
    att: dict[str, Any],
    verifier_evidence: Path,
    expected_signer_identity: str,
    now: datetime,
) -> dict[str, Any]:
    doc, _ = _read_json_exact(
        path,
        max_bytes=int(POLICY["max_publication_json_bytes"]),
        code="SIGNATURE_RECORD",
        canonical=True,
    )
    keys = {
        "schemaVersion",
        "verified",
        "verifiedAt",
        "publicationId",
        "publicationTransparencySha256",
        "attestationSha256",
        "signerIdentity",
        "signerIdentityVerified",
        "verifierEvidenceSha256",
        "verifier",
    }
    if set(doc) != keys or doc.get("schemaVersion") != int(
        POLICY["signature_record_schema_version"]
    ):
        _fail("SIGNATURE_RECORD_SCHEMA_INVALID")
    if doc.get("verified") is not True or doc.get("signerIdentityVerified") is not True:
        _fail("SIGNATURE_RECORD_NOT_VERIFIED")
    if (
        doc.get("publicationId") != info["publication_id"]
        or doc.get("publicationTransparencySha256") != info["transparency_sha256"]
        or doc.get("attestationSha256") != att["sha256"]
    ):
        _fail("SIGNATURE_RECORD_SUBJECT_MISMATCH")
    identity = _identity(
        doc.get("signerIdentity"), "SIGNATURE_RECORD_SIGNER_IDENTITY_INVALID"
    )
    if identity != expected_signer_identity:
        _fail("SIGNATURE_RECORD_SIGNER_IDENTITY_MISMATCH")
    evidence = _regular_file(
        verifier_evidence,
        max_bytes=int(POLICY["max_external_verifier_evidence_bytes"]),
        code="SIGNATURE_VERIFIER_EVIDENCE_INVALID",
    )
    if doc.get("verifierEvidenceSha256") != _sha256(evidence):
        _fail("SIGNATURE_VERIFIER_EVIDENCE_HASH_MISMATCH")
    verifier = doc.get("verifier")
    if not isinstance(verifier, dict) or set(verifier) != {"name", "version"}:
        _fail("SIGNATURE_RECORD_VERIFIER_SCHEMA_INVALID")
    for field in ("name", "version"):
        _bounded_text(
            verifier.get(field),
            "SIGNATURE_RECORD_VERIFIER_FIELD_INVALID",
            limit=128,
            allow_at=True,
        )
    verified = _parse_time(doc.get("verifiedAt"), "SIGNATURE_RECORD_TIME_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if att["generated"] > now + skew:
        _fail("POST_PUBLICATION_ATTESTATION_FROM_FUTURE")
    if now - att["generated"] > timedelta(
        hours=int(POLICY["max_attestation_age_hours"])
    ):
        _fail("POST_PUBLICATION_ATTESTATION_STALE")
    if verified > now + skew:
        _fail("SIGNATURE_RECORD_FROM_FUTURE")
    if now - verified > timedelta(hours=int(POLICY["max_attestation_age_hours"])):
        _fail("SIGNATURE_RECORD_STALE")
    if verified + skew < att["generated"]:
        _fail("SIGNATURE_RECORD_PREDATES_ATTESTATION")
    return {"doc": doc, "sha256": _sha256(path), "verified": verified}


def _validate_prepared_dir(
    prepared_dir: Path, *, info: dict[str, Any]
) -> dict[str, Any]:
    prepared_dir = _regular_dir(prepared_dir, "PREPARED_ATTESTATION_DIRECTORY_INVALID")
    att = _validate_attestation(
        prepared_dir / "publication-attestation.json",
        info=info,
        prepared_dir=prepared_dir,
    )
    results = prepared_dir / "independent-verifier-results"
    if results.is_symlink() or not results.is_dir():
        _fail("PREPARED_VERIFIER_RESULTS_DIRECTORY_INVALID")
    expected = {
        f"{index:02d}-{item['name']}.verify.json"
        for index, item in enumerate(info["artifacts"], 1)
    }
    actual = {p.name for p in results.iterdir()}
    if actual != expected or any(
        p.is_symlink() or not p.is_file() for p in results.iterdir()
    ):
        _fail("PREPARED_VERIFIER_RESULTS_ALLOWLIST_MISMATCH")
    if {p.name for p in prepared_dir.iterdir()} != {
        "publication-attestation.json",
        "independent-verifier-results",
    }:
        _fail("PREPARED_ATTESTATION_DIRECTORY_ALLOWLIST_MISMATCH")
    return att


def _normalized_final_artifacts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: item[key]
            for key in ("name", "sha256", "size", "locator", "immutability")
        }
        for item in results
    ]


def _validate_binder_response(
    value: Any,
    *,
    operation: str,
    info: dict[str, Any],
    binding_id: str,
    record_item: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    keys = {
        "schemaVersion",
        "operation",
        "bindingId",
        "publicationId",
        "status",
        "target",
        "record",
        "guarantees",
        "verifiedAt",
    }
    if not isinstance(value, dict) or set(value) != keys:
        _fail("RELEASE_BINDER_RESPONSE_SCHEMA_INVALID")
    if (
        value.get("schemaVersion") != int(POLICY["binder_protocol_version"])
        or value.get("operation") != operation
    ):
        _fail("RELEASE_BINDER_RESPONSE_VERSION_INVALID")
    if (
        value.get("bindingId") != binding_id
        or value.get("publicationId") != info["publication_id"]
    ):
        _fail("RELEASE_BINDER_RESPONSE_ID_MISMATCH")
    allowed_status = {"created", "present"} if operation == "bind" else {"present"}
    if value.get("status") not in allowed_status:
        _fail("RELEASE_BINDER_RESPONSE_STATUS_INVALID")
    target = value.get("target")
    if not isinstance(target, dict) or set(target) != {
        "publisher",
        "targetId",
        "locator",
    }:
        _fail("RELEASE_BINDER_TARGET_SCHEMA_INVALID")
    if (
        target.get("publisher") != info["publisher"]
        or target.get("targetId") != info["target_id"]
    ):
        _fail("RELEASE_BINDER_TARGET_MISMATCH")
    locator = _bounded_text(target.get("locator"), "RELEASE_BINDER_LOCATOR_INVALID")
    if value.get("record") != record_item:
        _fail("RELEASE_BINDER_RECORD_MISMATCH")
    guarantees = value.get("guarantees")
    if not isinstance(guarantees, dict) or set(guarantees) != {
        "createOnly",
        "overwrite",
        "remoteReadbackVerified",
        "immutability",
        "bindingType",
    }:
        _fail("RELEASE_BINDER_GUARANTEES_SCHEMA_INVALID")
    if (
        guarantees.get("createOnly") is not True
        or guarantees.get("overwrite") is not False
        or guarantees.get("remoteReadbackVerified") is not True
    ):
        _fail("RELEASE_BINDER_AUTHORITY_INVALID")
    immutability = guarantees.get("immutability")
    if immutability not in set(POLICY["allowed_immutability"]):
        _fail("RELEASE_BINDER_IMMUTABILITY_INVALID")
    binding_type = guarantees.get("bindingType")
    if binding_type not in set(POLICY["allowed_binding_types"]):
        _fail("RELEASE_BINDER_TYPE_INVALID")
    verified_at = _fresh_time(value.get("verifiedAt"), "RELEASE_BINDER_TIME", now=now)
    return {
        "locator": locator,
        "immutability": immutability,
        "bindingType": binding_type,
        "verifiedAt": verified_at,
        "status": value["status"],
    }


def finalize_publication(  # ruff: ignore[undocumented-public-function]
    *,
    publication_dir: Path,
    prepared_dir: Path,
    signature_record: Path,
    signature_verifier_evidence: Path,
    output_dir: Path,
    expected_signer_identity: str,
    verifier_identity: str,
    verifier: Verifier,
    binder: Binder,
    now: datetime | None = None,
) -> dict[str, Any]:
    publication_dir = _regular_dir(publication_dir, "PUBLICATION_DIRECTORY_INVALID")
    prepared_dir = _regular_dir(prepared_dir, "PREPARED_ATTESTATION_DIRECTORY_INVALID")
    target = _outside(
        output_dir, (publication_dir, prepared_dir), "FINALIZATION_OUTPUT_INSIDE_INPUT"
    )
    if target.exists() or target.is_symlink():
        _fail("FINALIZATION_OUTPUT_ALREADY_EXISTS")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expected_signer_identity = _identity(
        expected_signer_identity, "EXPECTED_SIGNER_IDENTITY_INVALID"
    )
    verifier_identity = _identity(
        verifier_identity, "INDEPENDENT_VERIFIER_IDENTITY_INVALID"
    )
    info = _validate_publication(publication_dir)
    att = _validate_prepared_dir(prepared_dir, info=info)
    if verifier_identity != att["verifier_identity"]:
        _fail("FINALIZATION_VERIFIER_IDENTITY_MISMATCH")
    signature_record = _regular_file(
        signature_record,
        max_bytes=int(POLICY["max_publication_json_bytes"]),
        code="SIGNATURE_RECORD_INVALID",
    )
    sig = _validate_signature_record(
        signature_record,
        info=info,
        att=att,
        verifier_evidence=signature_verifier_evidence,
        expected_signer_identity=expected_signer_identity,
        now=current,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-final-record-", dir=target.parent
    ) as temp_raw:
        temp = Path(temp_raw)
        stage = temp / "finalized"
        stage.mkdir()
        verification_id, final_results = _verify_remote_set(
            info=info,
            verifier_identity=verifier_identity,
            adapter=verifier,
            output_dir=stage / "final-verifier-results",
            now=current,
        )
        if verification_id != att["verification_id"]:
            _fail("FINALIZATION_VERIFICATION_ID_MISMATCH")
        skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        for result in final_results:
            verified_at = _parse_time(
                result["verifiedAt"], "FINALIZATION_VERIFIER_TIME_INVALID"
            )
            if verified_at + skew < sig["verified"]:
                _fail("FINALIZATION_VERIFICATION_PREDATES_SIGNATURE")
        normalized_final = _normalized_final_artifacts(final_results)
        if normalized_final != info["artifacts"]:
            _fail("FINALIZATION_REMOTE_SET_MISMATCH")

        final_record = {
            "schemaVersion": int(POLICY["final_record_schema_version"]),
            "predicateType": FINAL_RECORD_PREDICATE_TYPE,
            "status": "finalized",
            "publicationId": info["publication_id"],
            "release": {
                "releaseId": info["release_id"],
                "sourceRevision": info["source_revision"],
            },
            "subject": {
                "promotionReceiptSha256": info["promotion_receipt_sha256"],
                "publicationReceiptSha256": info["receipt_sha256"],
                "publicationTransparencySha256": info["transparency_sha256"],
                "postPublicationAttestationSha256": att["sha256"],
                "signatureVerificationSha256": sig["sha256"],
                "signatureVerifierEvidenceSha256": sig["doc"]["verifierEvidenceSha256"],
            },
            "target": {"publisher": info["publisher"], "targetId": info["target_id"]},
            "signer": {
                "identity": expected_signer_identity,
                "verifiedAt": sig["doc"]["verifiedAt"],
            },
            "independentVerifier": {
                "identity": verifier_identity,
                "verificationId": verification_id,
                "readOnly": True,
                "publisherCredentialsReused": False,
            },
            "artifacts": normalized_final,
            "verification": {
                "publicationEvidenceVerified": True,
                "signedAttestationVerified": True,
                "signerIdentityVerified": True,
                "independentRemoteReadbackVerifiedAfterSignature": True,
                "allPublishedObjectsPresent": True,
            },
        }
        record_path = stage / RECORD_NAME
        _write_canonical(record_path, final_record)
        os.chmod(record_path, 0o444)
        record_item = {
            "name": RECORD_NAME,
            "sha256": _sha256(record_path),
            "size": record_path.stat().st_size,
        }
        binding_id = _binding_id(
            info["publication_id"],
            record_item["sha256"],
            info["publisher"],
            info["target_id"],
        )
        binding_results = stage / "binding-results"
        binding_results.mkdir()
        base_request = {
            "schemaVersion": int(POLICY["binder_protocol_version"]),
            "bindingId": binding_id,
            "publicationId": info["publication_id"],
            "releaseId": info["release_id"],
            "publicationTransparencySha256": info["transparency_sha256"],
            "postPublicationAttestationSha256": att["sha256"],
            "target": {"publisher": info["publisher"], "targetId": info["target_id"]},
            "record": record_item,
        }
        bind_request = dict(base_request)
        bind_request["operation"] = "bind"
        bind_request["record"] = dict(record_item, localPath=str(record_path))
        before = _sha256(record_path)
        bind_raw = binder(bind_request)
        if _sha256(record_path) != before or before != record_item["sha256"]:
            _fail("FINAL_RECORD_CHANGED_DURING_BIND")
        bind_result = _validate_binder_response(
            bind_raw,
            operation="bind",
            info=info,
            binding_id=binding_id,
            record_item=record_item,
            now=current,
        )
        verify_request = dict(base_request)
        verify_request["operation"] = "verify"
        verify_raw = binder(verify_request)
        verify_result = _validate_binder_response(
            verify_raw,
            operation="verify",
            info=info,
            binding_id=binding_id,
            record_item=record_item,
            now=current,
        )
        if any(
            bind_result[field] != verify_result[field]
            for field in ("locator", "immutability", "bindingType")
        ):
            _fail("RELEASE_BINDER_VERIFY_REBIND_FAILED")
        if verify_result["locator"] in {item["locator"] for item in info["artifacts"]}:
            _fail("RELEASE_BINDER_LOCATOR_COLLISION")
        bind_path = binding_results / (RECORD_NAME + ".bind.json")
        verify_path = binding_results / (RECORD_NAME + ".verify.json")
        _write_canonical(bind_path, bind_raw)
        _write_canonical(verify_path, verify_raw)

        # Rebind immutable inputs after all remote calls. Final record bytes do not include
        # fresh verifier timestamps, so an interrupted create-only binding is reproducible.
        info_after = _validate_publication(publication_dir)
        att_after = _validate_prepared_dir(prepared_dir, info=info_after)
        if (
            info_after["transparency_sha256"] != info["transparency_sha256"]
            or info_after["receipt_sha256"] != info["receipt_sha256"]
            or att_after["sha256"] != att["sha256"]
        ):
            _fail("FINALIZATION_INPUT_CHANGED_DURING_BIND")
        if _sha256(signature_record) != sig["sha256"]:
            _fail("SIGNATURE_RECORD_CHANGED_DURING_BIND")

        shutil.copy2(
            publication_dir / "publication-transparency.json",
            stage / "publication-transparency.json",
        )
        shutil.copy2(
            publication_dir / "publication-receipt.json",
            stage / "publication-receipt.json",
        )
        shutil.copy2(
            prepared_dir / "publication-attestation.json",
            stage / "publication-attestation.json",
        )
        shutil.copy2(
            signature_record,
            stage / "publication-attestation.signature-verification.json",
        )
        binding_receipt = {
            "schemaVersion": int(POLICY["binding_receipt_schema_version"]),
            "status": "bound",
            "publicationId": info["publication_id"],
            "releaseId": info["release_id"],
            "publicationTransparencySha256": info["transparency_sha256"],
            "postPublicationAttestationSha256": att["sha256"],
            "signatureVerificationSha256": sig["sha256"],
            "finalRecord": record_item,
            "binding": {
                "bindingId": binding_id,
                "locator": verify_result["locator"],
                "immutability": verify_result["immutability"],
                "bindingType": verify_result["bindingType"],
                "verifiedAt": verify_result["verifiedAt"],
                "bindEvidenceSha256": _sha256(bind_path),
                "verifyEvidenceSha256": _sha256(verify_path),
            },
            "finalVerificationEvidence": [
                {"name": result["name"], "sha256": result["verifierEvidenceSha256"]}
                for result in final_results
            ],
        }
        receipt_path = stage / "release-publication-binding-receipt.json"
        _write_canonical(receipt_path, binding_receipt)
        tmp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, tmp_target, copy_function=shutil.copy2)
        os.replace(tmp_target, target)
    return {
        "ok": True,
        "phase": "bound",
        "publication_id": info["publication_id"],
        "release_id": info["release_id"],
        "attestation_sha256": att["sha256"],
        "final_record_sha256": _sha256(target / RECORD_NAME),
        "binding_receipt_sha256": _sha256(
            target / "release-publication-binding-receipt.json"
        ),
        "binding_id": binding_id,
        "artifact_count": len(info["artifacts"]),
    }


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--publication-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--verifier-identity", required=True)
    prepare.add_argument("--verifier-executable", required=True)
    prepare.add_argument("--verifier-arg", action="append", default=[])

    sig = sub.add_parser("signature-record")
    sig.add_argument("--publication-dir", type=Path, required=True)
    sig.add_argument("--attestation", type=Path, required=True)
    sig.add_argument("--verifier-evidence", type=Path, required=True)
    sig.add_argument("--output", type=Path, required=True)
    sig.add_argument("--signer-identity", required=True)
    sig.add_argument("--verifier-name", required=True)
    sig.add_argument("--verifier-version", required=True)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--publication-dir", type=Path, required=True)
    finalize.add_argument("--prepared-dir", type=Path, required=True)
    finalize.add_argument("--signature-record", type=Path, required=True)
    finalize.add_argument("--signature-verifier-evidence", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--expected-signer-identity", required=True)
    finalize.add_argument("--verifier-identity", required=True)
    finalize.add_argument("--verifier-executable", required=True)
    finalize.add_argument("--verifier-arg", action="append", default=[])
    finalize.add_argument("--binder-executable", required=True)
    finalize.add_argument("--binder-arg", action="append", default=[])

    args = parser.parse_args(argv)
    logger.info("Running publication command: %s", args.command)
    try:
        if args.command == "prepare":
            result = prepare_attestation(
                publication_dir=args.publication_dir,
                output_dir=args.output_dir,
                verifier_identity=args.verifier_identity,
                verifier=command_verifier(
                    [args.verifier_executable, *args.verifier_arg]
                ),
            )
        elif args.command == "signature-record":
            path = write_signature_verification_record(
                publication_dir=args.publication_dir,
                attestation=args.attestation,
                verifier_evidence=args.verifier_evidence,
                output=args.output,
                signer_identity=args.signer_identity,
                verifier_name=args.verifier_name,
                verifier_version=args.verifier_version,
            )
            result = {
                "ok": True,
                "phase": "signature-record",
                "path": path.name,
                "sha256": _sha256(path),
            }
        elif args.command == "finalize":
            result = finalize_publication(
                publication_dir=args.publication_dir,
                prepared_dir=args.prepared_dir,
                signature_record=args.signature_record,
                signature_verifier_evidence=args.signature_verifier_evidence,
                output_dir=args.output_dir,
                expected_signer_identity=args.expected_signer_identity,
                verifier_identity=args.verifier_identity,
                verifier=command_verifier(
                    [args.verifier_executable, *args.verifier_arg]
                ),
                binder=command_binder([args.binder_executable, *args.binder_arg]),
            )
        else:  # pragma: no cover
            _fail("COMMAND_INVALID")
    except (TransparencyError, OSError) as exc:
        logger.error("Publication finalization failed: %s", exc)
        # sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True) + "\n")
        return 2
    logger.info("Publication command completed successfully: %s", args.command)
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
