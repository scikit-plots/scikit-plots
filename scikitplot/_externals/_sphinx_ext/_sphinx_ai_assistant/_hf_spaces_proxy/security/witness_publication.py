"""
Anchor a Run 151 final release record in append-only transparency and require a witness quorum.
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

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_witness_policy.toml").read_text())
RUN151_POLICY = tomllib.loads((HERE / "release_transparency_policy.toml").read_text())
RECORD_NAME = str(POLICY["record_name"])
WITNESS_RECORD_PREDICATE_TYPE = str(POLICY["witness_record_predicate_type"])
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_CHUNK = 1024 * 1024


class WitnessError(RuntimeError):
    """Transparency logging or quorum verification violated a fail-closed invariant."""


def _fail(code: str) -> None:
    raise WitnessError(code)


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
    def pairs_hook(pairs):
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                _fail(code + "_DUPLICATE_KEY")
            out[key] = value
        return out

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
    except WitnessError:
        raise
    except Exception as exc:
        raise WitnessError(code + "_JSON_INVALID") from exc
    if not isinstance(value, dict):
        _fail(code + "_SCHEMA_INVALID")
    return value


def _read_json(
    path: Path, code: str, *, canonical: bool = True
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(code + "_INVALID")
    if path.stat().st_size > int(POLICY["max_json_bytes"]):
        _fail(code + "_TOO_LARGE")
    raw = path.read_bytes()
    doc = _loads_json(raw, code)
    if canonical and raw != _canonical_bytes(doc):
        _fail(code + "_NOT_CANONICAL")
    return doc, raw


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


def _hex(value: Any, code: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
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
    if (
        any(
            ord(ch) < 32  # ruff: ignore[magic-value-comparison]
            or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            for ch in value  # lint
        )
        or _ID.fullmatch(value) is None
    ):
        _fail(code)
    return value


def _safe_text(value: Any, code: str, *, limit: int = 1024) -> str:
    if not isinstance(value, str):
        _fail(code)
    value = value.strip()
    if (
        not value
        or len(value) > limit
        or "?" in value
        or "#" in value
        or "\x00" in value
    ):
        _fail(code)
    if any(
        ord(ch) < 32  # ruff: ignore[magic-value-comparison]
        or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in value  # lint
    ):
        _fail(code)
    return value


def _safe_name(value: Any, code: str) -> str:
    value = _safe_text(value, code, limit=255)
    if (
        value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        _fail(code)
    return value


def _size(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(code)
    return value


def _fresh(value: Any, code: str, *, now: datetime) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code + "_INVALID")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError:
        _fail(code + "_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    age = timedelta(minutes=int(POLICY["max_result_age_minutes"]))
    if parsed > now + skew:
        _fail(code + "_FROM_FUTURE")
    if now - parsed > age:
        _fail(code + "_STALE")
    return value


def _artifact(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "size"}:
        _fail(code + "_SCHEMA_INVALID")
    return {
        "name": _safe_name(value.get("name"), code + "_NAME_INVALID"),
        "sha256": _hex(value.get("sha256"), code + "_SHA256_INVALID"),
        "size": _size(value.get("size"), code + "_SIZE_INVALID"),
    }


def _checkpoint(value: Any, code: str) -> dict[str, Any]:
    expected = {"treeSize", "rootHash", "signedCheckpointSha256"}
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    tree_size = _size(value.get("treeSize"), code + "_TREE_SIZE_INVALID")
    if tree_size < 1:
        _fail(code + "_TREE_SIZE_INVALID")
    return {
        "treeSize": tree_size,
        "rootHash": _hex(value.get("rootHash"), code + "_ROOT_HASH_INVALID"),
        "signedCheckpointSha256": _hex(
            value.get("signedCheckpointSha256"), code + "_SIGNED_HASH_INVALID"
        ),
    }


def _validate_finalized(  # ruff: ignore[too-many-branches]
    root: Path,
) -> dict[str, Any]:
    root = _regular_dir(root, "RUN151_DIRECTORY_INVALID")
    allowed = {
        "publication-transparency.json",
        "publication-receipt.json",
        "publication-attestation.json",
        "publication-attestation.signature-verification.json",
        "release-publication-record.json",
        "release-publication-binding-receipt.json",
        "final-verifier-results",
        "binding-results",
    }
    if {p.name for p in root.iterdir()} != allowed:
        _fail("RUN151_DIRECTORY_ALLOWLIST_MISMATCH")
    for dirname in ("final-verifier-results", "binding-results"):
        directory = root / dirname
        if directory.is_symlink() or not directory.is_dir():
            _fail("RUN151_EVIDENCE_DIRECTORY_INVALID")
        for item in directory.iterdir():
            if item.is_symlink() or not item.is_file():
                _fail("RUN151_EVIDENCE_ENTRY_INVALID")
            _read_json(item, "RUN151_EVIDENCE_ENTRY")

    record, record_raw = _read_json(
        root / "release-publication-record.json", "RUN151_FINAL_RECORD"
    )
    receipt, receipt_raw = _read_json(
        root / "release-publication-binding-receipt.json", "RUN151_BINDING_RECEIPT"
    )
    if set(record) != {
        "schemaVersion",
        "predicateType",
        "status",
        "publicationId",
        "release",
        "subject",
        "target",
        "signer",
        "independentVerifier",
        "artifacts",
        "verification",
    }:
        _fail("RUN151_FINAL_RECORD_SCHEMA_INVALID")
    if (
        record.get("schemaVersion") != int(RUN151_POLICY["final_record_schema_version"])
        or record.get("predicateType")
        != str(RUN151_POLICY["final_record_predicate_type"])
        or record.get("status") != "finalized"
    ):
        _fail("RUN151_FINAL_RECORD_SCHEMA_INVALID")
    publication_id = _safe_text(
        record.get("publicationId"), "RUN151_PUBLICATION_ID_INVALID", limit=255
    )
    release = record.get("release")
    if not isinstance(release, dict) or set(release) != {"releaseId", "sourceRevision"}:
        _fail("RUN151_RELEASE_SCHEMA_INVALID")
    release_id = _safe_text(
        release.get("releaseId"), "RUN151_RELEASE_ID_INVALID", limit=127
    )
    source_revision = _safe_text(
        release.get("sourceRevision"), "RUN151_SOURCE_REVISION_INVALID", limit=128
    )
    if _REVISION.fullmatch(source_revision) is None:
        _fail("RUN151_SOURCE_REVISION_INVALID")
    target = record.get("target")
    if not isinstance(target, dict) or set(target) != {"publisher", "targetId"}:
        _fail("RUN151_TARGET_SCHEMA_INVALID")
    publisher = _safe_text(
        target.get("publisher"), "RUN151_PUBLISHER_INVALID", limit=255
    )
    target_id = _safe_text(
        target.get("targetId"), "RUN151_TARGET_ID_INVALID", limit=255
    )
    signer = record.get("signer")
    if not isinstance(signer, dict) or set(signer) != {"identity", "verifiedAt"}:
        _fail("RUN151_SIGNER_SCHEMA_INVALID")
    _identity(signer.get("identity"), "RUN151_SIGNER_IDENTITY_INVALID")
    if not isinstance(signer.get("verifiedAt"), str):
        _fail("RUN151_SIGNER_TIME_INVALID")
    independent = record.get("independentVerifier")
    if not isinstance(independent, dict) or set(independent) != {
        "identity",
        "verificationId",
        "readOnly",
        "publisherCredentialsReused",
    }:
        _fail("RUN151_INDEPENDENT_VERIFIER_SCHEMA_INVALID")
    _identity(
        independent.get("identity"), "RUN151_INDEPENDENT_VERIFIER_IDENTITY_INVALID"
    )
    _safe_text(
        independent.get("verificationId"),
        "RUN151_INDEPENDENT_VERIFICATION_ID_INVALID",
        limit=255,
    )
    if (
        independent.get("readOnly") is not True
        or independent.get("publisherCredentialsReused") is not False
    ):
        _fail("RUN151_INDEPENDENT_VERIFIER_AUTHORITY_INVALID")
    expected_verification = {
        "publicationEvidenceVerified": True,
        "signedAttestationVerified": True,
        "signerIdentityVerified": True,
        "independentRemoteReadbackVerifiedAfterSignature": True,
        "allPublishedObjectsPresent": True,
    }
    if record.get("verification") != expected_verification:
        _fail("RUN151_VERIFICATION_FLAGS_INVALID")
    subject = record.get("subject")
    expected_subject = {
        "promotionReceiptSha256",
        "publicationReceiptSha256",
        "publicationTransparencySha256",
        "postPublicationAttestationSha256",
        "signatureVerificationSha256",
        "signatureVerifierEvidenceSha256",
    }
    if not isinstance(subject, dict) or set(subject) != expected_subject:
        _fail("RUN151_SUBJECT_SCHEMA_INVALID")
    for key in expected_subject:
        _hex(subject.get(key), "RUN151_SUBJECT_HASH_INVALID")
    if (
        _sha256(root / "publication-transparency.json")
        != subject["publicationTransparencySha256"]
    ):
        _fail("RUN151_TRANSPARENCY_HASH_MISMATCH")
    if (
        _sha256(root / "publication-receipt.json")
        != subject["publicationReceiptSha256"]
    ):
        _fail("RUN151_PUBLICATION_RECEIPT_HASH_MISMATCH")
    if (
        _sha256(root / "publication-attestation.json")
        != subject["postPublicationAttestationSha256"]
    ):
        _fail("RUN151_ATTESTATION_HASH_MISMATCH")
    if (
        _sha256(root / "publication-attestation.signature-verification.json")
        != subject["signatureVerificationSha256"]
    ):
        _fail("RUN151_SIGNATURE_RECORD_HASH_MISMATCH")

    receipt_keys = {
        "schemaVersion",
        "status",
        "publicationId",
        "releaseId",
        "publicationTransparencySha256",
        "postPublicationAttestationSha256",
        "signatureVerificationSha256",
        "finalRecord",
        "binding",
        "finalVerificationEvidence",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != receipt_keys
        or receipt.get("schemaVersion")
        != int(RUN151_POLICY["binding_receipt_schema_version"])
        or receipt.get("status") != "bound"
    ):
        _fail("RUN151_BINDING_RECEIPT_SCHEMA_INVALID")
    if (
        receipt.get("publicationId") != publication_id
        or receipt.get("releaseId") != release_id
    ):
        _fail("RUN151_BINDING_RECEIPT_ID_MISMATCH")
    if (
        receipt.get("publicationTransparencySha256")
        != subject["publicationTransparencySha256"]
    ):
        _fail("RUN151_BINDING_RECEIPT_TRANSPARENCY_MISMATCH")
    if (
        receipt.get("postPublicationAttestationSha256")
        != subject["postPublicationAttestationSha256"]
    ):
        _fail("RUN151_BINDING_RECEIPT_ATTESTATION_MISMATCH")
    if (
        receipt.get("signatureVerificationSha256")
        != subject["signatureVerificationSha256"]
    ):
        _fail("RUN151_BINDING_RECEIPT_SIGNATURE_MISMATCH")
    record_item = _artifact(receipt.get("finalRecord"), "RUN151_BOUND_FINAL_RECORD")
    actual_record_item = {
        "name": "release-publication-record.json",
        "sha256": hashlib.sha256(record_raw).hexdigest(),
        "size": len(record_raw),
    }
    if record_item != actual_record_item:
        _fail("RUN151_BOUND_FINAL_RECORD_MISMATCH")
    binding = receipt.get("binding")
    if not isinstance(binding, dict) or set(binding) != {
        "bindingId",
        "locator",
        "immutability",
        "bindingType",
        "verifiedAt",
        "bindEvidenceSha256",
        "verifyEvidenceSha256",
    }:
        _fail("RUN151_BINDING_SCHEMA_INVALID")
    binding_locator = _safe_text(
        binding.get("locator"), "RUN151_BINDING_LOCATOR_INVALID"
    )
    if binding.get("immutability") not in set(POLICY["allowed_immutability"]):
        _fail("RUN151_BINDING_IMMUTABILITY_INVALID")
    if binding.get("bindingType") not in set(POLICY["allowed_binding_types"]):
        _fail("RUN151_BINDING_TYPE_INVALID")
    bind_evidence_sha = _hex(
        binding.get("bindEvidenceSha256"), "RUN151_BIND_EVIDENCE_HASH_INVALID"
    )
    verify_evidence_sha = _hex(
        binding.get("verifyEvidenceSha256"), "RUN151_VERIFY_EVIDENCE_HASH_INVALID"
    )
    bind_evidence = (
        root / "binding-results" / "release-publication-record.json.bind.json"
    )
    verify_evidence = (
        root / "binding-results" / "release-publication-record.json.verify.json"
    )
    if (
        _sha256(bind_evidence) != bind_evidence_sha
        or _sha256(verify_evidence) != verify_evidence_sha
    ):
        _fail("RUN151_BINDING_EVIDENCE_HASH_MISMATCH")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _fail("RUN151_ARTIFACTS_INVALID")
    remote_locators: set[str] = {binding_locator}
    artifact_names: set[str] = set()
    for _index, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != {
            "name",
            "sha256",
            "size",
            "locator",
            "immutability",
        }:
            _fail("RUN151_ARTIFACT_SCHEMA_INVALID")
        artifact_name = _safe_name(item.get("name"), "RUN151_ARTIFACT_NAME_INVALID")
        if artifact_name in artifact_names:
            _fail("RUN151_ARTIFACT_NAME_DUPLICATE")
        artifact_names.add(artifact_name)
        _hex(item.get("sha256"), "RUN151_ARTIFACT_HASH_INVALID")
        _size(item.get("size"), "RUN151_ARTIFACT_SIZE_INVALID")
        locator = _safe_text(item.get("locator"), "RUN151_ARTIFACT_LOCATOR_INVALID")
        if locator in remote_locators:
            _fail("RUN151_REMOTE_LOCATOR_COLLISION")
        remote_locators.add(locator)
        if item.get("immutability") not in set(POLICY["allowed_immutability"]):
            _fail("RUN151_ARTIFACT_IMMUTABILITY_INVALID")
    final_evidence = receipt.get("finalVerificationEvidence")
    if not isinstance(final_evidence, list) or len(final_evidence) != len(artifacts):
        _fail("RUN151_FINAL_VERIFICATION_EVIDENCE_INVALID")
    result_files = sorted((root / "final-verifier-results").iterdir())
    if len(result_files) != len(artifacts):
        _fail("RUN151_FINAL_VERIFICATION_EVIDENCE_COUNT_MISMATCH")
    for index, (artifact_item, evidence_item, evidence_path) in enumerate(
        zip(artifacts, final_evidence, result_files), start=1
    ):
        if not isinstance(evidence_item, dict) or set(evidence_item) != {
            "name",
            "sha256",
        }:
            _fail("RUN151_FINAL_VERIFICATION_EVIDENCE_SCHEMA_INVALID")
        if evidence_item.get("name") != artifact_item["name"]:
            _fail("RUN151_FINAL_VERIFICATION_EVIDENCE_NAME_MISMATCH")
        evidence_sha = _hex(
            evidence_item.get("sha256"),
            "RUN151_FINAL_VERIFICATION_EVIDENCE_HASH_INVALID",
        )
        expected_filename = f"{index:02d}-{artifact_item['name']}.verify.json"
        if (
            evidence_path.name != expected_filename
            or _sha256(evidence_path) != evidence_sha
        ):
            _fail("RUN151_FINAL_VERIFICATION_EVIDENCE_HASH_MISMATCH")
    return {
        "root": root,
        "publication_id": publication_id,
        "release_id": release_id,
        "source_revision": source_revision,
        "publisher": publisher,
        "target_id": target_id,
        "record": record,
        "record_item": actual_record_item,
        "record_sha256": actual_record_item["sha256"],
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "binding_locator": binding_locator,
        "remote_locators": remote_locators,
        "attestation_sha256": subject["postPublicationAttestationSha256"],
        "signature_sha256": subject["signatureVerificationSha256"],
        "transparency_sha256": subject["publicationTransparencySha256"],
    }


def _validate_previous_checkpoint(path: Path, *, log_id: str) -> dict[str, Any]:
    doc, raw = _read_json(path, "PREVIOUS_CHECKPOINT")
    if set(doc) != {"schemaVersion", "logId", "checkpoint"} or doc.get(
        "schemaVersion"
    ) != int(POLICY["previous_checkpoint_schema_version"]):
        _fail("PREVIOUS_CHECKPOINT_SCHEMA_INVALID")
    if (
        _safe_text(doc.get("logId"), "PREVIOUS_CHECKPOINT_LOG_ID_INVALID", limit=255)
        != log_id
    ):
        _fail("PREVIOUS_CHECKPOINT_LOG_MISMATCH")
    checkpoint = _checkpoint(doc.get("checkpoint"), "PREVIOUS_CHECKPOINT")
    return {
        "doc": doc,
        "checkpoint": checkpoint,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _transparency_id(record_sha: str, log_id: str) -> str:
    return (
        "tw-" + hashlib.sha256((record_sha + "\0" + log_id).encode()).hexdigest()[:40]
    )


def _verification_id(transparency_id: str, identity: str, role: str) -> str:
    return (
        role
        + "-"
        + hashlib.sha256((transparency_id + "\0" + identity).encode()).hexdigest()[:32]
    )


def _anchor_id(record_sha: str, publisher: str, target_id: str) -> str:
    return (
        "anchor-"
        + hashlib.sha256(
            (record_sha + "\0" + publisher + "\0" + target_id).encode()
        ).hexdigest()[:40]
    )


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
        assert (  # ruff: ignore[assert, pytest-composite-assertion]
            process.stdin is not None
            and process.stdout is not None
            and process.stderr is not None
        )
        stdin = process.stdin
        process.stdin = None
        stderr_chunks: list[bytes] = []
        stdout_chunks: list[bytes] = []
        overflow = threading.Event()
        limit = int(POLICY["max_adapter_output_bytes"])

        def drain(stream, chunks):
            total = 0
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

        threads = [
            threading.Thread(target=drain, args=(process.stdout, stdout_chunks)),
            threading.Thread(target=drain, args=(process.stderr, stderr_chunks)),
        ]
        for thread in threads:
            thread.start()
        try:
            stdin.write(_canonical_bytes(request))
            stdin.close()
            process.wait(timeout=int(POLICY["adapter_timeout_seconds"]))
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise WitnessError(prefix + "_TIMEOUT") from exc
        finally:
            for thread in threads:
                thread.join()
        if overflow.is_set():
            _fail(prefix + "_OUTPUT_TOO_LARGE")
        if process.returncode != 0:
            _fail(prefix + "_FAILED")
        return _loads_json(b"".join(stdout_chunks), prefix + "_OUTPUT")

    return call


def command_log(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="TRANSPARENCY_LOG_ADAPTER")


def command_verifier(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="TRANSPARENCY_VERIFIER_ADAPTER")


def command_witness(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="TRANSPARENCY_WITNESS_ADAPTER")


def command_anchor(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="TRANSPARENCY_ANCHOR_ADAPTER")


def _validate_log_response(
    value: dict[str, Any],
    *,
    info: dict[str, Any],
    transparency_id: str,
    log_id: str,
    previous: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "operation",
        "transparencyId",
        "status",
        "log",
        "subject",
        "checkpoint",
        "proof",
        "integratedAt",
    }
    if (
        set(value) != expected
        or value.get("schemaVersion") != int(POLICY["log_protocol_version"])
        or value.get("operation") != "submit"
    ):
        _fail("TRANSPARENCY_LOG_RESULT_SCHEMA_INVALID")
    if value.get("transparencyId") != transparency_id or value.get("status") not in {
        "created",
        "present",
    }:
        _fail("TRANSPARENCY_LOG_RESULT_STATUS_INVALID")
    log = value.get("log")
    if not isinstance(log, dict) or set(log) != {
        "logId",
        "entryId",
        "entryIndex",
        "entryLocator",
    }:
        _fail("TRANSPARENCY_LOG_RESULT_LOG_SCHEMA_INVALID")
    if log.get("logId") != log_id:
        _fail("TRANSPARENCY_LOG_ID_MISMATCH")
    entry_id = _safe_text(
        log.get("entryId"), "TRANSPARENCY_ENTRY_ID_INVALID", limit=255
    )
    entry_index = _size(log.get("entryIndex"), "TRANSPARENCY_ENTRY_INDEX_INVALID")
    entry_locator = _safe_text(
        log.get("entryLocator"), "TRANSPARENCY_ENTRY_LOCATOR_INVALID"
    )
    subject = _artifact(value.get("subject"), "TRANSPARENCY_LOG_SUBJECT")
    if subject != info["record_item"]:
        _fail("TRANSPARENCY_LOG_SUBJECT_MISMATCH")
    checkpoint = _checkpoint(value.get("checkpoint"), "TRANSPARENCY_LOG_CHECKPOINT")
    if (
        checkpoint["treeSize"] <= previous["checkpoint"]["treeSize"]
        or entry_index >= checkpoint["treeSize"]
    ):
        _fail("TRANSPARENCY_LOG_CHECKPOINT_ORDER_INVALID")
    proof = value.get("proof")
    if not isinstance(proof, dict) or set(proof) != {
        "appendOnly",
        "overwrite",
        "integratedEntryVerified",
    }:
        _fail("TRANSPARENCY_LOG_PROOF_SCHEMA_INVALID")
    if proof != {
        "appendOnly": True,
        "overwrite": False,
        "integratedEntryVerified": True,
    }:
        _fail("TRANSPARENCY_LOG_GUARANTEES_INVALID")
    integrated_at = _fresh(
        value.get("integratedAt"), "TRANSPARENCY_LOG_INTEGRATED_AT", now=now
    )
    return {
        "entryId": entry_id,
        "entryIndex": entry_index,
        "entryLocator": entry_locator,
        "checkpoint": checkpoint,
        "integratedAt": integrated_at,
    }


def _validate_observer_response(
    value: dict[str, Any],
    *,
    role: str,
    protocol_version: int,
    expected_id: str,
    identity: str,
    operator: str | None,
    info: dict[str, Any],
    transparency_id: str,
    log_id: str,
    log_result: dict[str, Any],
    previous: dict[str, Any],
    primary_identity: str | None,
    now: datetime,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "operation",
        "verificationId",
        "transparencyId",
        "status",
        "observer",
        "log",
        "subject",
        "checkpoint",
        "previousCheckpoint",
        "proof",
        "verifiedAt",
    }
    if (
        set(value) != expected
        or value.get("schemaVersion") != protocol_version
        or value.get("operation") != "verify"
    ):
        _fail(role + "_RESULT_SCHEMA_INVALID")
    if (
        value.get("verificationId") != expected_id
        or value.get("transparencyId") != transparency_id
        or value.get("status") != "included"
    ):
        _fail(role + "_RESULT_ID_MISMATCH")
    observer = value.get("observer")
    observer_keys = {"identity", "readOnly", "logCredentialsReused"} | (
        {"operator", "primaryVerifierCredentialsReused"}
        if operator is not None
        else set()
    )
    if not isinstance(observer, dict) or set(observer) != observer_keys:
        _fail(role + "_OBSERVER_SCHEMA_INVALID")
    if (
        observer.get("identity") != identity
        or observer.get("readOnly") is not True
        or observer.get("logCredentialsReused") is not False
    ):
        _fail(role + "_OBSERVER_AUTHORITY_INVALID")
    if operator is not None:
        if (
            observer.get("operator") != operator
            or observer.get("primaryVerifierCredentialsReused") is not False
        ):
            _fail(role + "_OBSERVER_SEPARATION_INVALID")
        if primary_identity is not None and identity == primary_identity:
            _fail(role + "_IDENTITY_REUSES_PRIMARY_VERIFIER")
    log = value.get("log")
    expected_log = {
        "logId": log_id,
        "entryId": log_result["entryId"],
        "entryIndex": log_result["entryIndex"],
        "entryLocator": log_result["entryLocator"],
    }
    if log != expected_log:
        _fail(role + "_LOG_REBIND_FAILED")
    if _artifact(value.get("subject"), role + "_SUBJECT") != info["record_item"]:
        _fail(role + "_SUBJECT_MISMATCH")
    if (
        _checkpoint(value.get("checkpoint"), role + "_CHECKPOINT")
        != log_result["checkpoint"]
    ):
        _fail(role + "_CHECKPOINT_MISMATCH")
    if (
        _checkpoint(value.get("previousCheckpoint"), role + "_PREVIOUS_CHECKPOINT")
        != previous["checkpoint"]
    ):
        _fail(role + "_PREVIOUS_CHECKPOINT_MISMATCH")
    proof = value.get("proof")
    expected_proof = {
        "checkpointSignatureVerified": True,
        "inclusionVerified": True,
        "consistencyVerified": True,
        "integratedEntryVerified": True,
    }
    if proof != expected_proof:
        _fail(role + "_PROOF_INVALID")
    verified_at = _fresh(value.get("verifiedAt"), role + "_VERIFIED_AT", now=now)
    return {"identity": identity, "operator": operator, "verifiedAt": verified_at}


def _validate_anchor_response(
    value: dict[str, Any],
    *,
    operation: str,
    info: dict[str, Any],
    record_item: dict[str, Any],
    anchor_id: str,
    now: datetime,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "operation",
        "anchorId",
        "status",
        "target",
        "record",
        "guarantees",
        "verifiedAt",
    }
    if (
        set(value) != expected
        or value.get("schemaVersion") != int(POLICY["anchor_protocol_version"])
        or value.get("operation") != operation
    ):
        _fail("TRANSPARENCY_ANCHOR_RESULT_SCHEMA_INVALID")
    if value.get("anchorId") != anchor_id or value.get("status") not in (
        {"created", "present"} if operation == "bind" else {"present"}
    ):
        _fail("TRANSPARENCY_ANCHOR_STATUS_INVALID")
    target = value.get("target")
    if target != {"publisher": info["publisher"], "targetId": info["target_id"]}:
        _fail("TRANSPARENCY_ANCHOR_TARGET_MISMATCH")
    if _artifact(value.get("record"), "TRANSPARENCY_ANCHOR_RECORD") != record_item:
        _fail("TRANSPARENCY_ANCHOR_RECORD_MISMATCH")
    guarantees = value.get("guarantees")
    if not isinstance(guarantees, dict) or set(guarantees) != {
        "createOnly",
        "overwrite",
        "remoteReadbackVerified",
        "immutability",
        "bindingType",
        "locator",
    }:
        _fail("TRANSPARENCY_ANCHOR_GUARANTEES_SCHEMA_INVALID")
    if (
        guarantees.get("createOnly") is not True
        or guarantees.get("overwrite") is not False
        or guarantees.get("remoteReadbackVerified") is not True
    ):
        _fail("TRANSPARENCY_ANCHOR_GUARANTEES_INVALID")
    immutability = guarantees.get("immutability")
    binding_type = guarantees.get("bindingType")
    locator = _safe_text(
        guarantees.get("locator"), "TRANSPARENCY_ANCHOR_LOCATOR_INVALID"
    )
    if immutability not in set(
        POLICY["allowed_immutability"]
    ) or binding_type not in set(POLICY["allowed_binding_types"]):
        _fail("TRANSPARENCY_ANCHOR_CLASS_INVALID")
    if locator in info["remote_locators"]:
        _fail("TRANSPARENCY_ANCHOR_LOCATOR_COLLISION")
    verified_at = _fresh(
        value.get("verifiedAt"), "TRANSPARENCY_ANCHOR_VERIFIED_AT", now=now
    )
    return {
        "locator": locator,
        "immutability": immutability,
        "bindingType": binding_type,
        "verifiedAt": verified_at,
    }


LogAdapter = Callable[[dict[str, Any]], dict[str, Any]]
ObserverAdapter = Callable[[dict[str, Any]], dict[str, Any]]
AnchorAdapter = Callable[[dict[str, Any]], dict[str, Any]]


def witness_publication(  # ruff: ignore[undocumented-public-function]
    *,
    finalized_dir: Path,
    previous_checkpoint: Path,
    output_dir: Path,
    log_id: str,
    submitter_identity: str,
    log_adapter: LogAdapter,
    verifier_identity: str,
    verifier: ObserverAdapter,
    witnesses: list[tuple[str, str, ObserverAdapter]],
    anchor: AnchorAdapter,
    now: datetime | None = None,
) -> dict[str, Any]:
    finalized_dir = _regular_dir(finalized_dir, "RUN151_DIRECTORY_INVALID")
    target = _outside(output_dir, (finalized_dir,), "WITNESS_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("WITNESS_OUTPUT_ALREADY_EXISTS")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    info = _validate_finalized(finalized_dir)
    log_id = _safe_text(log_id, "TRANSPARENCY_LOG_ID_INVALID", limit=255)
    submitter_identity = _identity(
        submitter_identity, "TRANSPARENCY_SUBMITTER_IDENTITY_INVALID"
    )
    verifier_identity = _identity(
        verifier_identity, "TRANSPARENCY_VERIFIER_IDENTITY_INVALID"
    )
    if verifier_identity == submitter_identity:
        _fail("TRANSPARENCY_VERIFIER_REUSES_SUBMITTER_IDENTITY")
    previous = _validate_previous_checkpoint(previous_checkpoint, log_id=log_id)
    if (
        not witnesses
        or len(witnesses) < int(POLICY["min_witnesses"])
        or len(witnesses) > int(POLICY["max_witnesses"])
    ):
        _fail("TRANSPARENCY_WITNESS_COUNT_INVALID")
    normalized_witnesses: list[tuple[str, str, ObserverAdapter]] = []
    identities: set[str] = set()
    operators: set[str] = set()
    for identity_raw, operator_raw, adapter in witnesses:
        identity = _identity(identity_raw, "TRANSPARENCY_WITNESS_IDENTITY_INVALID")
        operator = _identity(operator_raw, "TRANSPARENCY_WITNESS_OPERATOR_INVALID")
        if identity in identities or identity in {
            submitter_identity,
            verifier_identity,
        }:
            _fail("TRANSPARENCY_WITNESS_IDENTITY_NOT_DISTINCT")
        identities.add(identity)
        operators.add(operator)
        normalized_witnesses.append((identity, operator, adapter))
    if len(operators) < int(POLICY["min_witness_operators"]):
        _fail("TRANSPARENCY_WITNESS_OPERATOR_QUORUM_INVALID")

    initial_hashes = {
        name: _sha256(finalized_dir / name)
        for name in (
            "release-publication-record.json",
            "release-publication-binding-receipt.json",
            "publication-attestation.json",
            "publication-attestation.signature-verification.json",
        )
    }
    transparency_id = _transparency_id(info["record_sha256"], log_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-witness-", dir=target.parent
    ) as temp_raw:
        stage = Path(temp_raw) / "witnessed"
        stage.mkdir()
        log_results = stage / "log-results"
        witness_results = stage / "witness-results"
        anchor_results = stage / "anchor-results"
        log_results.mkdir()
        witness_results.mkdir()
        anchor_results.mkdir()
        subject_with_path = dict(
            info["record_item"],
            localPath=str(finalized_dir / "release-publication-record.json"),
        )
        log_request = {
            "schemaVersion": int(POLICY["log_protocol_version"]),
            "operation": "submit",
            "transparencyId": transparency_id,
            "release": {
                "releaseId": info["release_id"],
                "publicationId": info["publication_id"],
            },
            "submitter": {"identity": submitter_identity},
            "logId": log_id,
            "subject": subject_with_path,
            "previousCheckpoint": previous["checkpoint"],
        }
        before = _sha256(finalized_dir / "release-publication-record.json")
        log_raw = log_adapter(log_request)
        if _sha256(finalized_dir / "release-publication-record.json") != before:
            _fail("RUN151_FINAL_RECORD_CHANGED_DURING_LOG_SUBMIT")
        log_result = _validate_log_response(
            log_raw,
            info=info,
            transparency_id=transparency_id,
            log_id=log_id,
            previous=previous,
            now=current,
        )
        _write_canonical(log_results / "submit.json", log_raw)

        base_observer_request = {
            "schemaVersion": int(POLICY["verifier_protocol_version"]),
            "operation": "verify",
            "transparencyId": transparency_id,
            "release": {
                "releaseId": info["release_id"],
                "publicationId": info["publication_id"],
            },
            "log": {
                "logId": log_id,
                "entryId": log_result["entryId"],
                "entryIndex": log_result["entryIndex"],
                "entryLocator": log_result["entryLocator"],
            },
            "subject": info["record_item"],
            "checkpoint": log_result["checkpoint"],
            "previousCheckpoint": previous["checkpoint"],
        }
        primary_id = _verification_id(transparency_id, verifier_identity, "primary")
        primary_request = dict(base_observer_request)
        primary_request["verificationId"] = primary_id
        primary_request["observer"] = {"identity": verifier_identity}
        primary_raw = verifier(primary_request)
        _validate_observer_response(
            primary_raw,
            role="TRANSPARENCY_PRIMARY_VERIFIER",
            protocol_version=int(POLICY["verifier_protocol_version"]),
            expected_id=primary_id,
            identity=verifier_identity,
            operator=None,
            info=info,
            transparency_id=transparency_id,
            log_id=log_id,
            log_result=log_result,
            previous=previous,
            primary_identity=None,
            now=current,
        )
        primary_path = log_results / "primary.verify.json"
        _write_canonical(primary_path, primary_raw)

        witness_meta: list[dict[str, str]] = []
        witness_evidence: list[dict[str, str]] = []
        for index, (identity, operator, adapter) in enumerate(
            sorted(normalized_witnesses, key=lambda x: (x[1], x[0])), start=1
        ):
            wid = _verification_id(transparency_id, identity, "witness")
            request = dict(base_observer_request)
            request["schemaVersion"] = int(POLICY["witness_protocol_version"])
            request["verificationId"] = wid
            request["observer"] = {"identity": identity, "operator": operator}
            raw = adapter(request)
            _validate_observer_response(
                raw,
                role="TRANSPARENCY_WITNESS",
                protocol_version=int(POLICY["witness_protocol_version"]),
                expected_id=wid,
                identity=identity,
                operator=operator,
                info=info,
                transparency_id=transparency_id,
                log_id=log_id,
                log_result=log_result,
                previous=previous,
                primary_identity=verifier_identity,
                now=current,
            )
            path = (
                witness_results
                / f"{index:02d}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}.verify.json"
            )
            _write_canonical(path, raw)
            witness_meta.append({"identity": identity, "operator": operator})
            witness_evidence.append(
                {"identity": identity, "operator": operator, "sha256": _sha256(path)}
            )

        witnessed_record = {
            "schemaVersion": int(POLICY["witness_record_schema_version"]),
            "predicateType": WITNESS_RECORD_PREDICATE_TYPE,
            "status": "witnessed",
            "release": {
                "releaseId": info["release_id"],
                "publicationId": info["publication_id"],
                "sourceRevision": info["source_revision"],
            },
            "subject": {
                "finalPublicationRecordSha256": info["record_sha256"],
                "finalBindingReceiptSha256": info["receipt_sha256"],
                "postPublicationAttestationSha256": info["attestation_sha256"],
                "signatureVerificationSha256": info["signature_sha256"],
            },
            "target": {"publisher": info["publisher"], "targetId": info["target_id"]},
            "transparencyLog": {
                "logId": log_id,
                "entryId": log_result["entryId"],
                "entryIndex": log_result["entryIndex"],
                "entryLocator": log_result["entryLocator"],
                "checkpoint": log_result["checkpoint"],
                "previousCheckpointSha256": previous["sha256"],
            },
            "verification": {
                "primaryVerifierIdentity": verifier_identity,
                "checkpointSignatureVerified": True,
                "inclusionVerified": True,
                "consistencyVerified": True,
                "integratedEntryVerified": True,
            },
            "witnessQuorum": {
                "threshold": int(POLICY["min_witnesses"]),
                "minimumOperators": int(POLICY["min_witness_operators"]),
                "witnesses": witness_meta,
            },
        }
        record_path = stage / RECORD_NAME
        _write_canonical(record_path, witnessed_record)
        os.chmod(record_path, 0o444)
        record_item = {
            "name": RECORD_NAME,
            "sha256": _sha256(record_path),
            "size": record_path.stat().st_size,
        }
        anchor_id = _anchor_id(
            record_item["sha256"], info["publisher"], info["target_id"]
        )
        base_anchor = {
            "schemaVersion": int(POLICY["anchor_protocol_version"]),
            "anchorId": anchor_id,
            "releaseId": info["release_id"],
            "publicationId": info["publication_id"],
            "target": {"publisher": info["publisher"], "targetId": info["target_id"]},
            "record": record_item,
        }
        bind_request = dict(base_anchor, operation="bind")
        bind_request["record"] = dict(record_item, localPath=str(record_path))
        record_before = _sha256(record_path)
        bind_raw = anchor(bind_request)
        if _sha256(record_path) != record_before:
            _fail("WITNESS_RECORD_CHANGED_DURING_ANCHOR")
        bind_result = _validate_anchor_response(
            bind_raw,
            operation="bind",
            info=info,
            record_item=record_item,
            anchor_id=anchor_id,
            now=current,
        )
        verify_raw = anchor(dict(base_anchor, operation="verify"))
        verify_result = _validate_anchor_response(
            verify_raw,
            operation="verify",
            info=info,
            record_item=record_item,
            anchor_id=anchor_id,
            now=current,
        )
        if any(
            bind_result[key] != verify_result[key]
            for key in ("locator", "immutability", "bindingType")
        ):
            _fail("TRANSPARENCY_ANCHOR_VERIFY_REBIND_FAILED")
        bind_path = anchor_results / (RECORD_NAME + ".bind.json")
        verify_path = anchor_results / (RECORD_NAME + ".verify.json")
        _write_canonical(bind_path, bind_raw)
        _write_canonical(verify_path, verify_raw)

        info_after = _validate_finalized(finalized_dir)
        final_hashes = {name: _sha256(finalized_dir / name) for name in initial_hashes}
        if (
            initial_hashes != final_hashes
            or info_after["record_sha256"] != info["record_sha256"]
            or info_after["receipt_sha256"] != info["receipt_sha256"]
        ):
            _fail("RUN151_INPUT_CHANGED_DURING_WITNESSING")
        if _sha256(previous_checkpoint) != previous["sha256"]:
            _fail("PREVIOUS_CHECKPOINT_CHANGED_DURING_WITNESSING")

        shutil.copy2(
            finalized_dir / "release-publication-record.json",
            stage / "release-publication-record.json",
        )
        shutil.copy2(
            finalized_dir / "release-publication-binding-receipt.json",
            stage / "release-publication-binding-receipt.json",
        )
        shutil.copy2(
            previous_checkpoint, stage / "previous-transparency-checkpoint.json"
        )
        accepted_checkpoint_path = stage / "accepted-transparency-checkpoint.json"
        _write_canonical(
            accepted_checkpoint_path,
            {
                "schemaVersion": int(POLICY["previous_checkpoint_schema_version"]),
                "logId": log_id,
                "checkpoint": log_result["checkpoint"],
            },
        )
        receipt = {
            "schemaVersion": int(POLICY["witness_receipt_schema_version"]),
            "status": "witnessed",
            "releaseId": info["release_id"],
            "publicationId": info["publication_id"],
            "finalPublicationRecord": info["record_item"],
            "finalBindingReceiptSha256": info["receipt_sha256"],
            "witnessRecord": record_item,
            "transparency": {
                "transparencyId": transparency_id,
                "logId": log_id,
                "entryId": log_result["entryId"],
                "entryIndex": log_result["entryIndex"],
                "entryLocator": log_result["entryLocator"],
                "checkpoint": log_result["checkpoint"],
                "previousCheckpointSha256": previous["sha256"],
                "acceptedCheckpointSha256": _sha256(accepted_checkpoint_path),
                "submitEvidenceSha256": _sha256(log_results / "submit.json"),
                "primaryVerifierEvidenceSha256": _sha256(primary_path),
            },
            "witnessEvidence": witness_evidence,
            "anchor": {
                "anchorId": anchor_id,
                "locator": verify_result["locator"],
                "immutability": verify_result["immutability"],
                "bindingType": verify_result["bindingType"],
                "verifiedAt": verify_result["verifiedAt"],
                "bindEvidenceSha256": _sha256(bind_path),
                "verifyEvidenceSha256": _sha256(verify_path),
            },
        }
        receipt_path = stage / "release-transparency-witness-receipt.json"
        _write_canonical(receipt_path, receipt)
        tmp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, tmp_target, copy_function=shutil.copy2)
        os.replace(tmp_target, target)
    return {
        "ok": True,
        "phase": "witnessed",
        "release_id": info["release_id"],
        "publication_id": info["publication_id"],
        "transparency_id": transparency_id,
        "witness_count": len(witness_meta),
        "operator_count": len(operators),
        "witness_record_sha256": _sha256(target / RECORD_NAME),
        "witness_receipt_sha256": _sha256(
            target / "release-transparency-witness-receipt.json"
        ),
        "anchor_id": anchor_id,
    }


def _parse_witness(value: str) -> tuple[str, str, list[str]]:
    parts = value.split("=", 1)
    if len(parts) != 2 or "@" not in parts[0]:  # ruff: ignore[magic-value-comparison]
        raise argparse.ArgumentTypeError(
            "witness must be IDENTITY@OPERATOR=EXECUTABLE[,ARG...]"
        )
    identity, operator = parts[0].split("@", 1)
    command = [part for part in parts[1].split(",") if part]
    if not identity or not operator or not command:
        raise argparse.ArgumentTypeError("invalid witness specification")
    return identity, operator, command


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalized-dir", type=Path, required=True)
    parser.add_argument("--previous-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-id", required=True)
    parser.add_argument("--submitter-identity", required=True)
    parser.add_argument("--log-executable", required=True)
    parser.add_argument("--log-arg", action="append", default=[])
    parser.add_argument("--verifier-identity", required=True)
    parser.add_argument("--verifier-executable", required=True)
    parser.add_argument("--verifier-arg", action="append", default=[])
    parser.add_argument(
        "--witness",
        action="append",
        type=_parse_witness,
        required=True,
        help="IDENTITY@OPERATOR=EXECUTABLE[,ARG...]",
    )
    parser.add_argument("--anchor-executable", required=True)
    parser.add_argument("--anchor-arg", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        witnesses = [
            (identity, operator, command_witness(command))
            for identity, operator, command in args.witness
        ]
        result = witness_publication(
            finalized_dir=args.finalized_dir,
            previous_checkpoint=args.previous_checkpoint,
            output_dir=args.output_dir,
            log_id=args.log_id,
            submitter_identity=args.submitter_identity,
            log_adapter=command_log([args.log_executable, *args.log_arg]),
            verifier_identity=args.verifier_identity,
            verifier=command_verifier([args.verifier_executable, *args.verifier_arg]),
            witnesses=witnesses,
            anchor=command_anchor([args.anchor_executable, *args.anchor_arg]),
        )
    except (WitnessError, OSError) as exc:
        logger.error(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    logger.info(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
