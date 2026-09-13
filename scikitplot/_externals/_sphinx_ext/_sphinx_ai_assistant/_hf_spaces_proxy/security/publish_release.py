"""
Publish only Run 149 promotion-receipt objects and emit bounded transparency evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys  # ruff: ignore[unused-import]
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import tomllib

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_publication_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])
PROMOTION_PREDICATE_TYPE = str(POLICY["promotion_predicate_type"])
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_CHUNK = 1024 * 1024
logger = logging.getLogger(__name__)


class PublicationError(RuntimeError):
    """Publication transaction violated a fail-closed invariant."""


def _fail(code: str) -> None:
    raise PublicationError(code)


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
    except PublicationError:
        raise
    except Exception as exc:
        raise PublicationError(code + "_JSON_INVALID") from exc
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


def _hex(value: Any, code: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(code)
    return value


def _positive_size(value: Any, code: str) -> int:
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


def _outside(path: Path, protected: Path, code: str) -> Path:
    target = path.expanduser().resolve()
    root = protected.resolve()
    if target == root or root in target.parents:
        _fail(code)
    return target


def _read_json_exact(
    path: Path, *, max_bytes: int, code: str, canonical: bool = False
) -> tuple[dict[str, Any], bytes]:
    path = _regular_file(path, max_bytes=max_bytes, code=code)
    raw = path.read_bytes()
    doc = _loads_json(raw, code)
    if canonical and raw != _canonical_bytes(doc):
        _fail(code + "_NOT_CANONICAL")
    return doc, raw


def _publish_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "size"}:
        _fail("PROMOTION_PUBLISH_ITEM_SCHEMA_INVALID")
    return {
        "name": _safe_name(value.get("name"), "PROMOTION_PUBLISH_NAME_INVALID"),
        "sha256": _hex(value.get("sha256"), "PROMOTION_PUBLISH_SHA256_INVALID"),
        "size": _positive_size(value.get("size"), "PROMOTION_PUBLISH_SIZE_INVALID"),
    }


def _validate_promotion_bindings(  # ruff: ignore[too-many-branches]
    root: Path, receipt: dict[str, Any], items: list[dict[str, Any]]
) -> None:
    statement_path = root / "release-statement.json"
    signature_path = root / "release-statement.signature-verification.json"
    statement, statement_raw = _read_json_exact(
        statement_path,
        max_bytes=1024 * 1024,
        code="PROMOTION_STATEMENT",
        canonical=True,
    )
    signature, signature_raw = _read_json_exact(
        signature_path,
        max_bytes=1024 * 1024,
        code="PROMOTION_SIGNATURE",
        canonical=True,
    )
    if hashlib.sha256(statement_raw).hexdigest() != receipt["releaseStatementSha256"]:
        _fail("PROMOTION_STATEMENT_HASH_REBIND_FAILED")
    if (
        hashlib.sha256(signature_raw).hexdigest()
        != receipt["signatureVerificationSha256"]
    ):
        _fail("PROMOTION_SIGNATURE_HASH_REBIND_FAILED")

    expected_statement = {
        "schemaVersion",
        "predicateType",
        "generatedAt",
        "release",
        "subject",
        "artifacts",
        "sbomReferences",
        "verification",
    }
    if (
        set(statement) != expected_statement
        or statement.get("schemaVersion") != 1
        or statement.get("predicateType") != PROMOTION_PREDICATE_TYPE
    ):
        _fail("PROMOTION_STATEMENT_SCHEMA_INVALID")
    _parse_time(statement.get("generatedAt"), "PROMOTION_STATEMENT_TIME_INVALID")
    release = statement.get("release")
    if not isinstance(release, dict) or set(release) != {
        "releaseId",
        "proxyVersion",
        "sourceRevision",
    }:
        _fail("PROMOTION_STATEMENT_RELEASE_SCHEMA_INVALID")
    if (
        release.get("releaseId") != receipt["releaseId"]
        or release.get("sourceRevision") != receipt["sourceRevision"]
    ):
        _fail("PROMOTION_STATEMENT_RELEASE_REBIND_FAILED")
    if (
        not isinstance(release.get("proxyVersion"), str)
        or not release["proxyVersion"]
        or len(release["proxyVersion"]) > 64  # ruff: ignore[magic-value-comparison]
    ):
        _fail("PROMOTION_STATEMENT_PROXY_VERSION_INVALID")
    subject = statement.get("subject")
    if subject != {
        "sourceTreeSha256": receipt["sourceTreeSha256"],
        "evidenceSha256": receipt["evidenceSha256"],
    }:
        _fail("PROMOTION_STATEMENT_SUBJECT_REBIND_FAILED")
    if statement.get("sbomReferences") != receipt["sbomReferences"]:
        _fail("PROMOTION_STATEMENT_SBOM_REBIND_FAILED")
    if statement.get("verification") != {
        "evidenceVerified": True,
        "patchRecreatesSourceTree": True,
        "zipRecreatesSourceTree": True,
        "deterministicZip": True,
    }:
        _fail("PROMOTION_STATEMENT_VERIFICATION_INVALID")
    artifacts = statement.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "zip",
        "patch",
        "baselineZip",
    }:
        _fail("PROMOTION_STATEMENT_ARTIFACTS_SCHEMA_INVALID")
    by_name = {item["name"]: item for item in items}
    for field, suffix in (("zip", ".zip"), ("patch", ".patch")):
        value = artifacts.get(field)
        expected_keys = (
            {"name", "sha256", "size", "fileCount"}
            if field == "zip"
            else {"name", "sha256", "size"}
        )
        if not isinstance(value, dict) or set(value) != expected_keys:
            _fail("PROMOTION_STATEMENT_ARTIFACT_SCHEMA_INVALID")
        name = _safe_name(
            value.get("name"), "PROMOTION_STATEMENT_ARTIFACT_NAME_INVALID"
        )
        if not name.endswith(suffix) or name not in by_name:
            _fail("PROMOTION_STATEMENT_ARTIFACT_REBIND_FAILED")
        if (
            value.get("sha256") != by_name[name]["sha256"]
            or value.get("size") != by_name[name]["size"]
        ):
            _fail("PROMOTION_STATEMENT_ARTIFACT_REBIND_FAILED")
        if field == "zip" and (
            not isinstance(value.get("fileCount"), int)
            or isinstance(value.get("fileCount"), bool)
            or value["fileCount"] < 1
        ):
            _fail("PROMOTION_STATEMENT_ZIP_FILE_COUNT_INVALID")
    baseline = artifacts.get("baselineZip")
    if not isinstance(baseline, dict) or set(baseline) != {"name", "sha256", "size"}:
        _fail("PROMOTION_STATEMENT_BASELINE_SCHEMA_INVALID")
    _safe_name(baseline.get("name"), "PROMOTION_STATEMENT_BASELINE_NAME_INVALID")
    _hex(baseline.get("sha256"), "PROMOTION_STATEMENT_BASELINE_SHA_INVALID")
    _positive_size(baseline.get("size"), "PROMOTION_STATEMENT_BASELINE_SIZE_INVALID")

    expected_signature = {
        "schemaVersion",
        "verified",
        "verifiedAt",
        "releaseStatementSha256",
        "sourceRevision",
        "signerIdentityVerified",
        "verifierEvidenceSha256",
        "verifier",
    }
    if set(signature) != expected_signature or signature.get("schemaVersion") != 1:
        _fail("PROMOTION_SIGNATURE_SCHEMA_INVALID")
    if (
        signature.get("verified") is not True
        or signature.get("signerIdentityVerified") is not True
    ):
        _fail("PROMOTION_SIGNATURE_NOT_VERIFIED")
    if (
        signature.get("releaseStatementSha256") != receipt["releaseStatementSha256"]
        or signature.get("sourceRevision") != receipt["sourceRevision"]
    ):
        _fail("PROMOTION_SIGNATURE_SUBJECT_REBIND_FAILED")
    if (
        signature.get("verifierEvidenceSha256")
        != receipt["signatureVerifierEvidenceSha256"]
    ):
        _fail("PROMOTION_SIGNATURE_VERIFIER_REBIND_FAILED")
    _parse_time(signature.get("verifiedAt"), "PROMOTION_SIGNATURE_TIME_INVALID")
    verifier = signature.get("verifier")
    if not isinstance(verifier, dict) or set(verifier) != {"name", "version"}:
        _fail("PROMOTION_SIGNATURE_VERIFIER_SCHEMA_INVALID")
    for field in ("name", "version"):
        value = verifier.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 128  # ruff: ignore[magic-value-comparison]
            or any(ord(ch) < 32 for ch in value)  # ruff: ignore[magic-value-comparison]
        ):
            _fail("PROMOTION_SIGNATURE_VERIFIER_FIELD_INVALID")


def _validate_promotion(  # ruff: ignore[too-many-branches]
    promotion_dir: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    root = promotion_dir.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        _fail("PROMOTION_DIRECTORY_INVALID")
    receipt_path = root / "promotion-receipt.json"
    doc, raw = _read_json_exact(
        receipt_path,
        max_bytes=int(POLICY["max_promotion_receipt_bytes"]),
        code="PROMOTION_RECEIPT",
        canonical=True,
    )
    expected = {
        "schemaVersion",
        "finalizedAt",
        "releaseId",
        "sourceRevision",
        "sourceTreeSha256",
        "evidenceSha256",
        "releaseStatementSha256",
        "signatureVerificationSha256",
        "signatureVerifierEvidenceSha256",
        "publish",
        "sbomReferences",
        "status",
    }
    if (
        set(doc) != expected
        or doc.get("schemaVersion") != 1
        or doc.get("status") != "promoted"
    ):
        _fail("PROMOTION_RECEIPT_SCHEMA_INVALID")
    if (
        not isinstance(doc.get("releaseId"), str)
        or _RELEASE_ID.fullmatch(doc["releaseId"]) is None
    ):
        _fail("PROMOTION_RELEASE_ID_INVALID")
    if (
        not isinstance(doc.get("sourceRevision"), str)
        or _REVISION.fullmatch(doc["sourceRevision"]) is None
    ):
        _fail("PROMOTION_SOURCE_REVISION_INVALID")
    for key in (
        "sourceTreeSha256",
        "evidenceSha256",
        "releaseStatementSha256",
        "signatureVerificationSha256",
        "signatureVerifierEvidenceSha256",
    ):
        _hex(doc.get(key), "PROMOTION_RECEIPT_HASH_INVALID")
    _parse_time(doc.get("finalizedAt"), "PROMOTION_FINALIZED_TIME_INVALID")
    if not isinstance(doc.get("sbomReferences"), dict):
        _fail("PROMOTION_SBOM_REFERENCES_INVALID")
    values = doc.get("publish")
    if (
        not isinstance(values, list)
        or not values
        or len(values) > int(POLICY["max_publish_artifacts"])
    ):
        _fail("PROMOTION_PUBLISH_LIST_INVALID")
    items = [_publish_item(item) for item in values]
    names = [item["name"] for item in items]
    if len(names) != len(set(names)):
        _fail("PROMOTION_PUBLISH_DUPLICATE_NAME")
    if (
        "release-statement.json" not in names
        or "release-statement.signature-verification.json" not in names
    ):
        _fail("PROMOTION_REQUIRED_STATEMENT_MISSING")
    if (
        sum(name.endswith(".zip") for name in names) != 1
        or sum(name.endswith(".patch") for name in names) != 1
    ):
        _fail("PROMOTION_PRIMARY_ARTIFACT_SET_INVALID")
    total = 0
    for item in items:
        if item["size"] > int(POLICY["max_artifact_bytes"]):
            _fail("PROMOTION_ARTIFACT_TOO_LARGE")
        total += item["size"]
        if total > int(POLICY["max_total_publish_bytes"]):
            _fail("PROMOTION_TOTAL_TOO_LARGE")
        path = root / item["name"]
        if path.is_symlink() or not path.is_file():
            _fail("PROMOTION_ARTIFACT_MISSING")
        if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            _fail("PROMOTION_ARTIFACT_REBIND_FAILED")
    by_name = {item["name"]: item for item in items}
    if by_name["release-statement.json"]["sha256"] != doc["releaseStatementSha256"]:
        _fail("PROMOTION_STATEMENT_HASH_REBIND_FAILED")
    if (
        by_name["release-statement.signature-verification.json"]["sha256"]
        != doc["signatureVerificationSha256"]
    ):
        _fail("PROMOTION_SIGNATURE_HASH_REBIND_FAILED")
    _validate_promotion_bindings(root, doc, items)
    actual = sorted(p.name for p in root.iterdir() if not p.is_dir())
    expected_files = sorted([*names, "promotion-receipt.json"])
    if actual != expected_files or any(
        p.is_symlink() or p.is_dir() for p in root.iterdir()
    ):
        _fail("PROMOTION_DIRECTORY_CONTAINS_UNAUTHORIZED_OBJECT")
    return doc, hashlib.sha256(raw).hexdigest(), items


def _publication_id(receipt_sha: str, publisher: str, target_id: str) -> str:
    h = hashlib.sha256()
    h.update(b"release-publication-v1\0")
    for value in (receipt_sha, publisher, target_id):
        data = value.encode("utf-8")
        h.update(len(data).to_bytes(4, "big"))
        h.update(data)
    return "pub-" + h.hexdigest()[:32]


def _snapshot_artifacts(
    promotion_dir: Path, items: list[dict[str, Any]], target: Path
) -> dict[str, Path]:
    target.mkdir(parents=True, exist_ok=False)
    out: dict[str, Path] = {}
    for item in items:
        source = promotion_dir / item["name"]
        dest = target / item["name"]
        with source.open("rb") as src, dest.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=_CHUNK)
        os.chmod(dest, 0o444)
        if dest.stat().st_size != item["size"] or _sha256(dest) != item["sha256"]:
            _fail("PUBLICATION_SNAPSHOT_REBIND_FAILED")
        out[item["name"]] = dest
    return out


def _bounded_text(value: Any, code: str, *, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        _fail(code)
    if any(
        ord(ch) < 32  # ruff: ignore[magic-value-comparison]
        or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in value
    ):
        _fail(code)
    if any(marker in value for marker in ("?", "#", "@")):
        _fail(code)
    return value.strip()


def _validate_adapter_response(  # ruff: ignore[too-many-branches]
    value: Any,
    *,
    operation: str,
    item: dict[str, Any],
    publisher: str,
    target_id: str,
    publication_id: str,
    now: datetime,
    required_immutability: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schemaVersion",
        "operation",
        "publicationId",
        "status",
        "target",
        "artifact",
        "guarantees",
        "verifiedAt",
    }:
        _fail("PUBLISHER_RESPONSE_SCHEMA_INVALID")
    if (
        value.get("schemaVersion") != int(POLICY["adapter_protocol_version"])
        or value.get("operation") != operation
    ):
        _fail("PUBLISHER_RESPONSE_VERSION_INVALID")
    if value.get("publicationId") != publication_id:
        _fail("PUBLISHER_RESPONSE_PUBLICATION_ID_MISMATCH")
    allowed_status = {"created", "present"} if operation == "publish" else {"present"}
    if value.get("status") not in allowed_status:
        _fail("PUBLISHER_RESPONSE_STATUS_INVALID")
    target = value.get("target")
    if not isinstance(target, dict) or set(target) != {
        "publisher",
        "targetId",
        "locator",
    }:
        _fail("PUBLISHER_RESPONSE_TARGET_SCHEMA_INVALID")
    if target.get("publisher") != publisher or target.get("targetId") != target_id:
        _fail("PUBLISHER_RESPONSE_TARGET_MISMATCH")
    locator = _bounded_text(
        target.get("locator"), "PUBLISHER_RESPONSE_LOCATOR_INVALID", limit=1024
    )
    artifact = value.get("artifact")
    if artifact != {
        "name": item["name"],
        "sha256": item["sha256"],
        "size": item["size"],
    }:
        _fail("PUBLISHER_RESPONSE_ARTIFACT_MISMATCH")
    guarantees = value.get("guarantees")
    expected_guarantees = {
        "createOnly",
        "overwrite",
        "remoteReadbackVerified",
        "immutability",
    }
    if not isinstance(guarantees, dict) or set(guarantees) != expected_guarantees:
        _fail("PUBLISHER_RESPONSE_GUARANTEES_SCHEMA_INVALID")
    if (
        guarantees.get("createOnly") is not True
        or guarantees.get("overwrite") is not False
        or guarantees.get("remoteReadbackVerified") is not True
    ):
        _fail("PUBLISHER_RESPONSE_AUTHORITY_INVALID")
    immutability = guarantees.get("immutability")
    allowed = set(POLICY["allowed_immutability"])
    if not isinstance(immutability, str) or immutability not in allowed:
        _fail("PUBLISHER_RESPONSE_IMMUTABILITY_INVALID")
    if required_immutability and immutability not in required_immutability:
        _fail("PUBLISHER_RESPONSE_IMMUTABILITY_INSUFFICIENT")
    verified_at = _parse_time(
        value.get("verifiedAt"), "PUBLISHER_RESPONSE_TIME_INVALID"
    )
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    max_age = timedelta(minutes=int(POLICY["max_result_age_minutes"]))
    if verified_at > now + skew:
        _fail("PUBLISHER_RESPONSE_FROM_FUTURE")
    if now - verified_at > max_age:
        _fail("PUBLISHER_RESPONSE_STALE")
    return {
        "status": value["status"],
        "locator": locator,
        "immutability": immutability,
        "verifiedAt": value["verifiedAt"],
    }


Publisher = Callable[[dict[str, Any]], dict[str, Any]]


def command_publisher(  # ruff: ignore[undocumented-public-function]
    command: list[str],
) -> Publisher:
    if not command or any(not isinstance(part, str) or not part for part in command):
        _fail("PUBLISHER_COMMAND_INVALID")
    executable = (
        shutil.which(command[0]) if not os.path.isabs(command[0]) else command[0]
    )
    if not executable or not Path(executable).is_file():
        _fail("PUBLISHER_COMMAND_UNAVAILABLE")
    argv = [executable, *command[1:]]

    def invoke(  # ruff: ignore[too-many-branches]
        request: dict[str, Any],
    ) -> dict[str, Any]:
        payload = _canonical_bytes(request)
        limit = int(POLICY["max_adapter_output_bytes"])
        proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
        )
        assert (  # ruff: ignore[assert, pytest-composite-assertion]
            proc.stdin is not None and proc.stdout is not None
        )
        chunks: list[bytes] = []
        state = {"size": 0, "overflow": False, "error": None}

        def drain_stdout() -> None:
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
            except (
                BaseException  # ruff: ignore[blind-except]
            ) as exc:  # pragma: no cover
                state["error"] = exc
                try:  # ruff: ignore[suppressible-exception]
                    proc.kill()
                except OSError:
                    pass

        reader = threading.Thread(
            target=drain_stdout, name="release-publisher-stdout", daemon=True
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
                raise PublicationError("PUBLISHER_COMMAND_TIMEOUT") from exc
            finally:
                try:  # ruff: ignore[suppressible-exception]
                    proc.stdin.close()
                except OSError:
                    pass

            reader.join(timeout=2)
            if reader.is_alive():
                proc.kill()
                proc.wait()
                reader.join(timeout=2)
                _fail("PUBLISHER_COMMAND_OUTPUT_DRAIN_FAILED")
            if state["error"] is not None:
                _fail("PUBLISHER_COMMAND_OUTPUT_DRAIN_FAILED")
            if state["overflow"]:
                _fail("PUBLISHER_COMMAND_OUTPUT_TOO_LARGE")
            if proc.returncode != 0:
                _fail("PUBLISHER_COMMAND_FAILED")
            stdout = b"".join(chunks)
            try:
                return _loads_json(stdout, "PUBLISHER_COMMAND_OUTPUT")
            except PublicationError as exc:
                if str(exc).startswith("PUBLISHER_COMMAND_OUTPUT"):
                    raise
                raise PublicationError("PUBLISHER_COMMAND_OUTPUT_INVALID") from exc
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


def publish_release(  # ruff: ignore[undocumented-public-function]
    *,
    promotion_dir: Path,
    output_dir: Path,
    publisher: str,
    target_id: str,
    adapter: Publisher,
    required_immutability: Iterable[str] = (),
    now: datetime | None = None,
) -> dict[str, Any]:
    promotion_dir = promotion_dir.expanduser().resolve()
    publisher = _safe_id(publisher, "PUBLICATION_PUBLISHER_INVALID")
    target_id = _safe_id(target_id, "PUBLICATION_TARGET_ID_INVALID")
    target = _outside(output_dir, promotion_dir, "PUBLICATION_OUTPUT_INSIDE_PROMOTION")
    if target.exists() or target.is_symlink():
        _fail("PUBLICATION_OUTPUT_ALREADY_EXISTS")
    required = set(required_immutability)
    if not required.issubset(set(POLICY["allowed_immutability"])):
        _fail("PUBLICATION_REQUIRED_IMMUTABILITY_INVALID")
    fixed_now = now.astimezone(timezone.utc) if now is not None else None

    def clock_now() -> datetime:
        return fixed_now or datetime.now(timezone.utc)

    receipt, receipt_sha, items = _validate_promotion(promotion_dir)
    publication_id = _publication_id(receipt_sha, publisher, target_id)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-publication-", dir=target.parent
    ) as temp_raw:
        temp = Path(temp_raw)
        snapshots = _snapshot_artifacts(promotion_dir, items, temp / "snapshot")
        evidence_dir = temp / "evidence"
        results_dir = evidence_dir / "publisher-results"
        results_dir.mkdir(parents=True)
        transparency_artifacts: list[dict[str, Any]] = []
        receipt_artifacts: list[dict[str, Any]] = []
        locators: set[str] = set()
        for index, item in enumerate(items, 1):
            snapshot = snapshots[item["name"]]
            base_request = {
                "schemaVersion": int(POLICY["adapter_protocol_version"]),
                "publicationId": publication_id,
                "releaseId": receipt["releaseId"],
                "promotionReceiptSha256": receipt_sha,
                "target": {"publisher": publisher, "targetId": target_id},
                "artifact": {
                    "name": item["name"],
                    "sha256": item["sha256"],
                    "size": item["size"],
                },
            }
            publish_request = dict(base_request)
            publish_request["operation"] = "publish"
            publish_request["artifact"] = dict(
                base_request["artifact"], localPath=str(snapshot)
            )
            before = _sha256(snapshot)
            publish_raw = adapter(publish_request)
            if _sha256(snapshot) != before or before != item["sha256"]:
                _fail("PUBLICATION_LOCAL_ARTIFACT_CHANGED_DURING_UPLOAD")
            publish_result = _validate_adapter_response(
                publish_raw,
                operation="publish",
                item=item,
                publisher=publisher,
                target_id=target_id,
                publication_id=publication_id,
                now=clock_now(),
                required_immutability=required,
            )
            verify_request = dict(base_request)
            verify_request["operation"] = "verify"
            verify_raw = adapter(verify_request)
            verify_result = _validate_adapter_response(
                verify_raw,
                operation="verify",
                item=item,
                publisher=publisher,
                target_id=target_id,
                publication_id=publication_id,
                now=clock_now(),
                required_immutability=required,
            )
            if (
                publish_result["locator"] != verify_result["locator"]
                or publish_result["immutability"] != verify_result["immutability"]
            ):
                _fail("PUBLISHER_VERIFY_RESULT_REBIND_FAILED")
            locator = verify_result["locator"]
            if locator in locators:
                _fail("PUBLISHER_LOCATOR_COLLISION")
            locators.add(locator)
            if (
                _sha256(snapshot) != item["sha256"]
                or snapshot.stat().st_size != item["size"]
            ):
                _fail("PUBLICATION_LOCAL_SNAPSHOT_CHANGED")
            pub_path = results_dir / f"{index:02d}-{item['name']}.publish.json"
            ver_path = results_dir / f"{index:02d}-{item['name']}.verify.json"
            _write_canonical(pub_path, publish_raw)
            _write_canonical(ver_path, verify_raw)
            transparency_artifacts.append(
                {
                    "name": item["name"],
                    "sha256": item["sha256"],
                    "size": item["size"],
                    "remote": {
                        "locator": locator,
                        "sha256": item["sha256"],
                        "size": item["size"],
                        "immutability": verify_result["immutability"],
                        "verifiedAt": verify_result["verifiedAt"],
                    },
                    "publisherEvidence": {
                        "publishSha256": _sha256(pub_path),
                        "verifySha256": _sha256(ver_path),
                    },
                }
            )
            receipt_artifacts.append(
                {
                    "name": item["name"],
                    "sha256": item["sha256"],
                    "size": item["size"],
                    "locator": locator,
                    "immutability": verify_result["immutability"],
                }
            )

        # The original promotion directory may be mutable, but every published byte came from the
        # verified server-owned snapshot. Re-validate it anyway to surface operator-side drift.
        receipt_after, receipt_sha_after, items_after = _validate_promotion(
            promotion_dir
        )
        if (
            receipt_sha_after != receipt_sha
            or receipt_after != receipt
            or items_after != items
        ):
            _fail("PROMOTION_CHANGED_DURING_PUBLICATION")

        transparency = {
            "schemaVersion": int(POLICY["transparency_schema_version"]),
            "predicateType": PREDICATE_TYPE,
            "generatedAt": _utc(clock_now()),
            "publicationId": publication_id,
            "release": {
                "releaseId": receipt["releaseId"],
                "sourceRevision": receipt["sourceRevision"],
            },
            "subject": {
                "promotionReceiptSha256": receipt_sha,
                "sourceTreeSha256": receipt["sourceTreeSha256"],
                "evidenceSha256": receipt["evidenceSha256"],
                "releaseStatementSha256": receipt["releaseStatementSha256"],
            },
            "target": {"publisher": publisher, "targetId": target_id},
            "artifacts": transparency_artifacts,
            "verification": {
                "promotionReceiptVerified": True,
                "localSnapshotVerified": True,
                "createOnly": True,
                "remoteReadbackVerified": True,
                "allReceiptObjectsPublished": True,
            },
        }
        transparency_path = evidence_dir / "publication-transparency.json"
        _write_canonical(transparency_path, transparency)
        publication_receipt = {
            "schemaVersion": int(POLICY["publication_receipt_schema_version"]),
            "status": "published",
            "completedAt": _utc(clock_now()),
            "publicationId": publication_id,
            "releaseId": receipt["releaseId"],
            "sourceRevision": receipt["sourceRevision"],
            "promotionReceiptSha256": receipt_sha,
            "transparencySha256": _sha256(transparency_path),
            "target": {"publisher": publisher, "targetId": target_id},
            "artifacts": receipt_artifacts,
        }
        receipt_path = evidence_dir / "publication-receipt.json"
        _write_canonical(receipt_path, publication_receipt)
        tmp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(evidence_dir, tmp_target, copy_function=shutil.copy2)
        os.replace(tmp_target, target)
    return {
        "ok": True,
        "phase": "published",
        "publication_id": publication_id,
        "release_id": receipt["releaseId"],
        "promotion_receipt_sha256": receipt_sha,
        "transparency_sha256": _sha256(target / "publication-transparency.json"),
        "publication_receipt_sha256": _sha256(target / "publication-receipt.json"),
        "artifact_count": len(items),
    }


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--publisher-executable", required=True)
    parser.add_argument("--publisher-arg", action="append", default=[])
    parser.add_argument("--require-immutability", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        adapter = command_publisher([args.publisher_executable, *args.publisher_arg])
        result = publish_release(
            promotion_dir=args.promotion_dir,
            output_dir=args.output_dir,
            publisher=args.publisher,
            target_id=args.target_id,
            adapter=adapter,
            required_immutability=args.require_immutability,
        )
    except PublicationError as exc:
        logger.error(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    logger.info(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
