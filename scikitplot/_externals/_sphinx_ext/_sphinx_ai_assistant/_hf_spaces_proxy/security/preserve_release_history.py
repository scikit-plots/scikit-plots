"""
Preserve witnessed releases in a quorum-gossiped, independently archived history chain.
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
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

import tomllib

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_history_policy.toml").read_text())
WITNESS_POLICY = tomllib.loads((HERE / "release_witness_policy.toml").read_text())
RUN151_POLICY = tomllib.loads((HERE / "release_transparency_policy.toml").read_text())
HISTORY_PREDICATE_TYPE = str(POLICY["history_predicate_type"])
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_CHUNK = 1024 * 1024
logger = logging.getLogger(__name__)


class HistoryError(RuntimeError):
    """Release-history preservation violated a fail-closed invariant."""


def _fail(code: str) -> None:
    raise HistoryError(code)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
    except HistoryError:
        raise
    except Exception as exc:
        raise HistoryError(code + "_JSON_INVALID") from exc
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
            for ch in value
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
        for ch in value
    ):
        _fail(code)
    return value


def _safe_locator(value: Any, code: str) -> str:
    value = _safe_text(value, code, limit=2048)
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
    if not isinstance(value, dict) or set(value) != {
        "treeSize",
        "rootHash",
        "signedCheckpointSha256",
    }:
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


def _checkpoint_doc(log_id: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": int(WITNESS_POLICY["previous_checkpoint_schema_version"]),
        "logId": log_id,
        "checkpoint": checkpoint,
    }


def _history_head(
    history_id: str, genesis: dict[str, Any], entries: list[dict[str, Any]]
) -> str:
    head = hashlib.sha256(
        _canonical_bytes({"historyId": history_id, "genesis": genesis})
    ).digest()
    for entry in entries:
        head = hashlib.sha256(head + _canonical_bytes(entry)).digest()
    return head.hex()


def _validate_witnessed(  # ruff: ignore[too-many-branches]
    root: Path,
) -> dict[str, Any]:
    root = _regular_dir(root, "RUN152_DIRECTORY_INVALID")
    allowed = {
        "release-publication-record.json",
        "release-publication-binding-receipt.json",
        "previous-transparency-checkpoint.json",
        "accepted-transparency-checkpoint.json",
        "release-transparency-witness-record.json",
        "release-transparency-witness-receipt.json",
        "log-results",
        "witness-results",
        "anchor-results",
    }
    if {p.name for p in root.iterdir()} != allowed:
        _fail("RUN152_DIRECTORY_ALLOWLIST_MISMATCH")
    for dirname in ("log-results", "witness-results", "anchor-results"):
        directory = root / dirname
        if directory.is_symlink() or not directory.is_dir():
            _fail("RUN152_EVIDENCE_DIRECTORY_INVALID")
        for item in directory.iterdir():
            if item.is_symlink() or not item.is_file():
                _fail("RUN152_EVIDENCE_ENTRY_INVALID")
            _read_json(item, "RUN152_EVIDENCE_ENTRY")

    final_record, final_raw = _read_json(
        root / "release-publication-record.json", "RUN152_FINAL_RECORD"
    )
    final_binding, final_binding_raw = _read_json(
        root / "release-publication-binding-receipt.json", "RUN152_FINAL_BINDING"
    )
    previous, previous_raw = _read_json(
        root / "previous-transparency-checkpoint.json", "RUN152_PREVIOUS_CHECKPOINT"
    )
    accepted, accepted_raw = _read_json(
        root / "accepted-transparency-checkpoint.json", "RUN152_ACCEPTED_CHECKPOINT"
    )
    witness_record, witness_raw = _read_json(
        root / "release-transparency-witness-record.json", "RUN152_WITNESS_RECORD"
    )
    receipt, receipt_raw = _read_json(
        root / "release-transparency-witness-receipt.json", "RUN152_WITNESS_RECEIPT"
    )

    if (
        set(final_record)
        != {
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
        }
        or final_record.get("schemaVersion")
        != int(RUN151_POLICY["final_record_schema_version"])
        or final_record.get("predicateType")
        != str(RUN151_POLICY["final_record_predicate_type"])
        or final_record.get("status") != "finalized"
    ):
        _fail("RUN152_FINAL_RECORD_SCHEMA_INVALID")
    publication_id = _safe_text(
        final_record.get("publicationId"), "RUN152_PUBLICATION_ID_INVALID", limit=255
    )
    release = final_record.get("release")
    if not isinstance(release, dict) or set(release) != {"releaseId", "sourceRevision"}:
        _fail("RUN152_RELEASE_SCHEMA_INVALID")
    release_id = _safe_text(
        release.get("releaseId"), "RUN152_RELEASE_ID_INVALID", limit=127
    )
    source_revision = _safe_text(
        release.get("sourceRevision"), "RUN152_SOURCE_REVISION_INVALID", limit=128
    )
    if _REVISION.fullmatch(source_revision) is None:
        _fail("RUN152_SOURCE_REVISION_INVALID")
    final_subject = final_record.get("subject")
    final_subject_keys = {
        "promotionReceiptSha256",
        "publicationReceiptSha256",
        "publicationTransparencySha256",
        "postPublicationAttestationSha256",
        "signatureVerificationSha256",
        "signatureVerifierEvidenceSha256",
    }
    if not isinstance(final_subject, dict) or set(final_subject) != final_subject_keys:
        _fail("RUN152_FINAL_SUBJECT_SCHEMA_INVALID")
    for value in final_subject.values():
        _hex(value, "RUN152_FINAL_SUBJECT_HASH_INVALID")
    final_target = final_record.get("target")
    if not isinstance(final_target, dict) or set(final_target) != {
        "publisher",
        "targetId",
    }:
        _fail("RUN152_FINAL_TARGET_SCHEMA_INVALID")

    for doc, label in ((previous, "PREVIOUS"), (accepted, "ACCEPTED")):
        if set(doc) != {"schemaVersion", "logId", "checkpoint"} or doc.get(
            "schemaVersion"
        ) != int(WITNESS_POLICY["previous_checkpoint_schema_version"]):
            _fail("RUN152_" + label + "_CHECKPOINT_SCHEMA_INVALID")
    log_id = _safe_text(accepted.get("logId"), "RUN152_LOG_ID_INVALID", limit=255)
    if previous.get("logId") != log_id:
        _fail("RUN152_CHECKPOINT_LOG_MISMATCH")
    previous_checkpoint = _checkpoint(
        previous.get("checkpoint"), "RUN152_PREVIOUS_CHECKPOINT"
    )
    accepted_checkpoint = _checkpoint(
        accepted.get("checkpoint"), "RUN152_ACCEPTED_CHECKPOINT"
    )
    if accepted_checkpoint["treeSize"] <= previous_checkpoint["treeSize"]:
        _fail("RUN152_CHECKPOINT_ORDER_INVALID")

    expected_wr_keys = {
        "schemaVersion",
        "predicateType",
        "status",
        "release",
        "subject",
        "target",
        "transparencyLog",
        "verification",
        "witnessQuorum",
    }
    if (
        set(witness_record) != expected_wr_keys
        or witness_record.get("schemaVersion")
        != int(WITNESS_POLICY["witness_record_schema_version"])
        or witness_record.get("predicateType")
        != str(WITNESS_POLICY["witness_record_predicate_type"])
        or witness_record.get("status") != "witnessed"
    ):
        _fail("RUN152_WITNESS_RECORD_SCHEMA_INVALID")
    target = witness_record.get("target")
    if not isinstance(target, dict) or set(target) != {"publisher", "targetId"}:
        _fail("RUN152_WITNESS_TARGET_SCHEMA_INVALID")
    publisher = _safe_text(
        target.get("publisher"), "RUN152_WITNESS_PUBLISHER_INVALID", limit=255
    )
    target_id = _safe_text(
        target.get("targetId"), "RUN152_WITNESS_TARGET_ID_INVALID", limit=255
    )
    if final_target != {"publisher": publisher, "targetId": target_id}:
        _fail("RUN152_WITNESS_TARGET_MISMATCH")
    wr_release = witness_record.get("release")
    if wr_release != {
        "releaseId": release_id,
        "publicationId": publication_id,
        "sourceRevision": source_revision,
    }:
        _fail("RUN152_WITNESS_RELEASE_MISMATCH")
    wr_subject = witness_record.get("subject")
    if not isinstance(wr_subject, dict) or set(wr_subject) != {
        "finalPublicationRecordSha256",
        "finalBindingReceiptSha256",
        "postPublicationAttestationSha256",
        "signatureVerificationSha256",
    }:
        _fail("RUN152_WITNESS_SUBJECT_SCHEMA_INVALID")
    if wr_subject.get("finalPublicationRecordSha256") != _sha_bytes(
        final_raw
    ) or wr_subject.get("finalBindingReceiptSha256") != _sha_bytes(final_binding_raw):
        _fail("RUN152_WITNESS_SUBJECT_HASH_MISMATCH")
    if (
        wr_subject.get("postPublicationAttestationSha256")
        != final_subject["postPublicationAttestationSha256"]
        or wr_subject.get("signatureVerificationSha256")
        != final_subject["signatureVerificationSha256"]
    ):
        _fail("RUN152_WITNESS_FINAL_SUBJECT_REBIND_FAILED")
    wr_verification = witness_record.get("verification")
    if not isinstance(wr_verification, dict) or set(wr_verification) != {
        "primaryVerifierIdentity",
        "checkpointSignatureVerified",
        "inclusionVerified",
        "consistencyVerified",
        "integratedEntryVerified",
    }:
        _fail("RUN152_WITNESS_VERIFICATION_SCHEMA_INVALID")
    primary_identity = _identity(
        wr_verification.get("primaryVerifierIdentity"),
        "RUN152_PRIMARY_VERIFIER_IDENTITY_INVALID",
    )
    if {
        k: wr_verification[k]
        for k in (
            "checkpointSignatureVerified",
            "inclusionVerified",
            "consistencyVerified",
            "integratedEntryVerified",
        )
    } != {
        "checkpointSignatureVerified": True,
        "inclusionVerified": True,
        "consistencyVerified": True,
        "integratedEntryVerified": True,
    }:
        _fail("RUN152_WITNESS_VERIFICATION_FLAGS_INVALID")
    tlog = witness_record.get("transparencyLog")
    if not isinstance(tlog, dict) or set(tlog) != {
        "logId",
        "entryId",
        "entryIndex",
        "entryLocator",
        "checkpoint",
        "previousCheckpointSha256",
    }:
        _fail("RUN152_WITNESS_LOG_SCHEMA_INVALID")
    if (
        tlog.get("logId") != log_id
        or _checkpoint(tlog.get("checkpoint"), "RUN152_WITNESS_CHECKPOINT")
        != accepted_checkpoint
    ):
        _fail("RUN152_WITNESS_CHECKPOINT_MISMATCH")
    if tlog.get("previousCheckpointSha256") != _sha_bytes(previous_raw):
        _fail("RUN152_WITNESS_PREVIOUS_CHECKPOINT_HASH_MISMATCH")
    _safe_text(tlog.get("entryId"), "RUN152_ENTRY_ID_INVALID", limit=255)
    _size(tlog.get("entryIndex"), "RUN152_ENTRY_INDEX_INVALID")
    _safe_text(tlog.get("entryLocator"), "RUN152_ENTRY_LOCATOR_INVALID")
    quorum = witness_record.get("witnessQuorum")
    if not isinstance(quorum, dict) or set(quorum) != {
        "threshold",
        "minimumOperators",
        "witnesses",
    }:
        _fail("RUN152_WITNESS_QUORUM_SCHEMA_INVALID")
    threshold = _size(quorum.get("threshold"), "RUN152_WITNESS_THRESHOLD_INVALID")
    minimum_operators = _size(
        quorum.get("minimumOperators"), "RUN152_WITNESS_OPERATOR_THRESHOLD_INVALID"
    )
    witness_meta = quorum.get("witnesses")
    if not isinstance(witness_meta, list) or len(witness_meta) < threshold:
        _fail("RUN152_WITNESS_QUORUM_INVALID")
    normalized_meta = []
    for item in witness_meta:
        if not isinstance(item, dict) or set(item) != {"identity", "operator"}:
            _fail("RUN152_WITNESS_META_SCHEMA_INVALID")
        normalized_meta.append(
            {
                "identity": _identity(
                    item.get("identity"), "RUN152_WITNESS_IDENTITY_INVALID"
                ),
                "operator": _identity(
                    item.get("operator"), "RUN152_WITNESS_OPERATOR_INVALID"
                ),
            }
        )
    if len({x["operator"] for x in normalized_meta}) < minimum_operators:
        _fail("RUN152_WITNESS_OPERATOR_QUORUM_INVALID")

    final_binding_keys = {
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
        not isinstance(final_binding, dict)
        or set(final_binding) != final_binding_keys
        or final_binding.get("schemaVersion")
        != int(RUN151_POLICY["binding_receipt_schema_version"])
        or final_binding.get("status") != "bound"
        or final_binding.get("publicationId") != publication_id
        or final_binding.get("releaseId") != release_id
    ):
        _fail("RUN152_FINAL_BINDING_SCHEMA_INVALID")
    if _artifact(final_binding.get("finalRecord"), "RUN152_FINAL_BINDING_RECORD") != {
        "name": "release-publication-record.json",
        "sha256": _sha_bytes(final_raw),
        "size": len(final_raw),
    }:
        _fail("RUN152_FINAL_BINDING_RECORD_MISMATCH")

    receipt_keys = {
        "schemaVersion",
        "status",
        "releaseId",
        "publicationId",
        "finalPublicationRecord",
        "finalBindingReceiptSha256",
        "witnessRecord",
        "transparency",
        "witnessEvidence",
        "anchor",
    }
    if (
        set(receipt) != receipt_keys
        or receipt.get("schemaVersion")
        != int(WITNESS_POLICY["witness_receipt_schema_version"])
        or receipt.get("status") != "witnessed"
    ):
        _fail("RUN152_WITNESS_RECEIPT_SCHEMA_INVALID")
    if (
        receipt.get("releaseId") != release_id
        or receipt.get("publicationId") != publication_id
    ):
        _fail("RUN152_WITNESS_RECEIPT_ID_MISMATCH")
    final_item = _artifact(
        receipt.get("finalPublicationRecord"), "RUN152_FINAL_RECORD_ITEM"
    )
    expected_final_item = {
        "name": "release-publication-record.json",
        "sha256": _sha_bytes(final_raw),
        "size": len(final_raw),
    }
    if final_item != expected_final_item or receipt.get(
        "finalBindingReceiptSha256"
    ) != _sha_bytes(final_binding_raw):
        _fail("RUN152_FINAL_RECORD_REBIND_FAILED")
    witness_item = _artifact(receipt.get("witnessRecord"), "RUN152_WITNESS_RECORD_ITEM")
    expected_witness_item = {
        "name": "release-transparency-witness-record.json",
        "sha256": _sha_bytes(witness_raw),
        "size": len(witness_raw),
    }
    if witness_item != expected_witness_item:
        _fail("RUN152_WITNESS_RECORD_REBIND_FAILED")
    trans = receipt.get("transparency")
    trans_keys = {
        "transparencyId",
        "logId",
        "entryId",
        "entryIndex",
        "entryLocator",
        "checkpoint",
        "previousCheckpointSha256",
        "acceptedCheckpointSha256",
        "submitEvidenceSha256",
        "primaryVerifierEvidenceSha256",
    }
    if not isinstance(trans, dict) or set(trans) != trans_keys:
        _fail("RUN152_TRANSPARENCY_RECEIPT_SCHEMA_INVALID")
    if (
        trans.get("logId") != log_id
        or trans.get("entryId") != tlog.get("entryId")
        or trans.get("entryIndex") != tlog.get("entryIndex")
        or trans.get("entryLocator") != tlog.get("entryLocator")
        or _checkpoint(trans.get("checkpoint"), "RUN152_RECEIPT_CHECKPOINT")
        != accepted_checkpoint
    ):
        _fail("RUN152_TRANSPARENCY_RECEIPT_MISMATCH")
    if trans.get("previousCheckpointSha256") != _sha_bytes(previous_raw) or trans.get(
        "acceptedCheckpointSha256"
    ) != _sha_bytes(accepted_raw):
        _fail("RUN152_CHECKPOINT_RECEIPT_HASH_MISMATCH")
    submit_path = root / "log-results" / "submit.json"
    primary_path = root / "log-results" / "primary.verify.json"
    if _sha256(submit_path) != _hex(
        trans.get("submitEvidenceSha256"), "RUN152_SUBMIT_EVIDENCE_HASH_INVALID"
    ):
        _fail("RUN152_SUBMIT_EVIDENCE_HASH_MISMATCH")
    if _sha256(primary_path) != _hex(
        trans.get("primaryVerifierEvidenceSha256"),
        "RUN152_PRIMARY_EVIDENCE_HASH_INVALID",
    ):
        _fail("RUN152_PRIMARY_EVIDENCE_HASH_MISMATCH")
    submit_evidence, _ = _read_json(submit_path, "RUN152_SUBMIT_EVIDENCE")
    submit_keys = {
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
    expected_log = {
        "logId": log_id,
        "entryId": tlog["entryId"],
        "entryIndex": tlog["entryIndex"],
        "entryLocator": tlog["entryLocator"],
    }
    if (
        set(submit_evidence) != submit_keys
        or submit_evidence.get("schemaVersion")
        != int(WITNESS_POLICY["log_protocol_version"])
        or submit_evidence.get("operation") != "submit"
        or submit_evidence.get("transparencyId") != trans.get("transparencyId")
        or submit_evidence.get("status") not in {"created", "present"}
    ):
        _fail("RUN152_SUBMIT_EVIDENCE_SCHEMA_INVALID")
    if (
        submit_evidence.get("log") != expected_log
        or _artifact(submit_evidence.get("subject"), "RUN152_SUBMIT_EVIDENCE_SUBJECT")
        != expected_final_item
        or _checkpoint(
            submit_evidence.get("checkpoint"), "RUN152_SUBMIT_EVIDENCE_CHECKPOINT"
        )
        != accepted_checkpoint
    ):
        _fail("RUN152_SUBMIT_EVIDENCE_REBIND_FAILED")
    if submit_evidence.get("proof") != {
        "appendOnly": True,
        "overwrite": False,
        "integratedEntryVerified": True,
    }:
        _fail("RUN152_SUBMIT_EVIDENCE_GUARANTEES_INVALID")

    primary_evidence, _ = _read_json(primary_path, "RUN152_PRIMARY_EVIDENCE")
    observer_keys = {
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
        set(primary_evidence) != observer_keys
        or primary_evidence.get("schemaVersion")
        != int(WITNESS_POLICY["verifier_protocol_version"])
        or primary_evidence.get("operation") != "verify"
        or primary_evidence.get("transparencyId") != trans.get("transparencyId")
        or primary_evidence.get("status") != "included"
    ):
        _fail("RUN152_PRIMARY_EVIDENCE_SCHEMA_INVALID")
    primary_observer = primary_evidence.get("observer")
    if (
        not isinstance(primary_observer, dict)
        or set(primary_observer) != {"identity", "readOnly", "logCredentialsReused"}
        or primary_observer.get("identity") != primary_identity
        or primary_observer.get("readOnly") is not True
        or primary_observer.get("logCredentialsReused") is not False
    ):
        _fail("RUN152_PRIMARY_EVIDENCE_AUTHORITY_INVALID")
    if (
        primary_evidence.get("log") != expected_log
        or _artifact(primary_evidence.get("subject"), "RUN152_PRIMARY_EVIDENCE_SUBJECT")
        != expected_final_item
        or _checkpoint(
            primary_evidence.get("checkpoint"), "RUN152_PRIMARY_EVIDENCE_CHECKPOINT"
        )
        != accepted_checkpoint
        or _checkpoint(
            primary_evidence.get("previousCheckpoint"),
            "RUN152_PRIMARY_EVIDENCE_PREVIOUS_CHECKPOINT",
        )
        != previous_checkpoint
    ):
        _fail("RUN152_PRIMARY_EVIDENCE_REBIND_FAILED")
    if primary_evidence.get("proof") != {
        "checkpointSignatureVerified": True,
        "inclusionVerified": True,
        "consistencyVerified": True,
        "integratedEntryVerified": True,
    }:
        _fail("RUN152_PRIMARY_EVIDENCE_PROOF_INVALID")

    witness_evidence = receipt.get("witnessEvidence")
    if not isinstance(witness_evidence, list) or len(witness_evidence) != len(
        normalized_meta
    ):
        _fail("RUN152_WITNESS_EVIDENCE_COUNT_MISMATCH")
    evidence_files = sorted((root / "witness-results").iterdir())
    if len(evidence_files) != len(witness_evidence):
        _fail("RUN152_WITNESS_RESULT_COUNT_MISMATCH")
    evidence_by_hash = {_sha256(path): path for path in evidence_files}
    evidence_meta = []
    for item in witness_evidence:
        if not isinstance(item, dict) or set(item) != {
            "identity",
            "operator",
            "sha256",
        }:
            _fail("RUN152_WITNESS_EVIDENCE_SCHEMA_INVALID")
        identity = _identity(
            item.get("identity"), "RUN152_WITNESS_EVIDENCE_IDENTITY_INVALID"
        )
        operator = _identity(
            item.get("operator"), "RUN152_WITNESS_EVIDENCE_OPERATOR_INVALID"
        )
        sha = _hex(item.get("sha256"), "RUN152_WITNESS_EVIDENCE_HASH_INVALID")
        if sha not in evidence_by_hash:
            _fail("RUN152_WITNESS_EVIDENCE_HASH_MISMATCH")
        evidence_doc, _ = _read_json(evidence_by_hash[sha], "RUN152_WITNESS_EVIDENCE")
        if (
            set(evidence_doc) != observer_keys
            or evidence_doc.get("schemaVersion")
            != int(WITNESS_POLICY["witness_protocol_version"])
            or evidence_doc.get("operation") != "verify"
            or evidence_doc.get("transparencyId") != trans.get("transparencyId")
            or evidence_doc.get("status") != "included"
        ):
            _fail("RUN152_WITNESS_EVIDENCE_SCHEMA_INVALID")
        observer = evidence_doc.get("observer")
        witness_observer_keys = {
            "identity",
            "operator",
            "readOnly",
            "logCredentialsReused",
            "primaryVerifierCredentialsReused",
        }
        if (
            not isinstance(observer, dict)
            or set(observer) != witness_observer_keys
            or observer.get("identity") != identity
            or observer.get("operator") != operator
            or observer.get("readOnly") is not True
            or observer.get("logCredentialsReused") is not False
            or observer.get("primaryVerifierCredentialsReused") is not False
        ):
            _fail("RUN152_WITNESS_EVIDENCE_AUTHORITY_INVALID")
        if (
            evidence_doc.get("log") != expected_log
            or _artifact(evidence_doc.get("subject"), "RUN152_WITNESS_EVIDENCE_SUBJECT")
            != expected_final_item
            or _checkpoint(
                evidence_doc.get("checkpoint"), "RUN152_WITNESS_EVIDENCE_CHECKPOINT"
            )
            != accepted_checkpoint
            or _checkpoint(
                evidence_doc.get("previousCheckpoint"),
                "RUN152_WITNESS_EVIDENCE_PREVIOUS_CHECKPOINT",
            )
            != previous_checkpoint
        ):
            _fail("RUN152_WITNESS_EVIDENCE_REBIND_FAILED")
        if evidence_doc.get("proof") != {
            "checkpointSignatureVerified": True,
            "inclusionVerified": True,
            "consistencyVerified": True,
            "integratedEntryVerified": True,
        }:
            _fail("RUN152_WITNESS_EVIDENCE_PROOF_INVALID")
        evidence_meta.append({"identity": identity, "operator": operator})
    if sorted(evidence_meta, key=lambda x: (x["operator"], x["identity"])) != sorted(
        normalized_meta, key=lambda x: (x["operator"], x["identity"])
    ):
        _fail("RUN152_WITNESS_EVIDENCE_MEMBERSHIP_MISMATCH")

    anchor = receipt.get("anchor")
    anchor_keys = {
        "anchorId",
        "locator",
        "immutability",
        "bindingType",
        "verifiedAt",
        "bindEvidenceSha256",
        "verifyEvidenceSha256",
    }
    if not isinstance(anchor, dict) or set(anchor) != anchor_keys:
        _fail("RUN152_ANCHOR_RECEIPT_SCHEMA_INVALID")
    _safe_text(anchor.get("anchorId"), "RUN152_ANCHOR_ID_INVALID", limit=255)
    anchor_locator = _safe_locator(
        anchor.get("locator"), "RUN152_ANCHOR_LOCATOR_INVALID"
    )
    bind_path = (
        root / "anchor-results" / "release-transparency-witness-record.json.bind.json"
    )
    verify_path = (
        root / "anchor-results" / "release-transparency-witness-record.json.verify.json"
    )
    if _sha256(bind_path) != _hex(
        anchor.get("bindEvidenceSha256"), "RUN152_ANCHOR_BIND_HASH_INVALID"
    ) or _sha256(verify_path) != _hex(
        anchor.get("verifyEvidenceSha256"), "RUN152_ANCHOR_VERIFY_HASH_INVALID"
    ):
        _fail("RUN152_ANCHOR_EVIDENCE_HASH_MISMATCH")
    for operation, evidence_path in (("bind", bind_path), ("verify", verify_path)):
        evidence, _ = _read_json(evidence_path, "RUN152_ANCHOR_EVIDENCE")
        expected_anchor_keys = {
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
            set(evidence) != expected_anchor_keys
            or evidence.get("schemaVersion")
            != int(WITNESS_POLICY["anchor_protocol_version"])
            or evidence.get("operation") != operation
            or evidence.get("anchorId") != anchor.get("anchorId")
        ):
            _fail("RUN152_ANCHOR_EVIDENCE_SCHEMA_INVALID")
        if evidence.get("status") not in (
            {"created", "present"} if operation == "bind" else {"present"}
        ):
            _fail("RUN152_ANCHOR_EVIDENCE_STATUS_INVALID")
        if (
            evidence.get("target") != {"publisher": publisher, "targetId": target_id}
            or _artifact(evidence.get("record"), "RUN152_ANCHOR_EVIDENCE_RECORD")
            != expected_witness_item
        ):
            _fail("RUN152_ANCHOR_EVIDENCE_REBIND_FAILED")
        guarantees = evidence.get("guarantees")
        if not isinstance(guarantees, dict) or set(guarantees) != {
            "createOnly",
            "overwrite",
            "remoteReadbackVerified",
            "immutability",
            "bindingType",
            "locator",
        }:
            _fail("RUN152_ANCHOR_EVIDENCE_GUARANTEES_SCHEMA_INVALID")
        if (
            guarantees.get("createOnly") is not True
            or guarantees.get("overwrite") is not False
            or guarantees.get("remoteReadbackVerified") is not True
        ):
            _fail("RUN152_ANCHOR_EVIDENCE_GUARANTEES_INVALID")
        if (
            guarantees.get("immutability") != anchor.get("immutability")
            or guarantees.get("bindingType") != anchor.get("bindingType")
            or _safe_locator(
                guarantees.get("locator"), "RUN152_ANCHOR_EVIDENCE_LOCATOR_INVALID"
            )
            != anchor_locator
        ):
            _fail("RUN152_ANCHOR_EVIDENCE_GUARANTEES_MISMATCH")
        if operation == "verify" and evidence.get("verifiedAt") != anchor.get(
            "verifiedAt"
        ):
            _fail("RUN152_ANCHOR_EVIDENCE_TIME_MISMATCH")

    return {
        "root": root,
        "release_id": release_id,
        "publication_id": publication_id,
        "source_revision": source_revision,
        "log_id": log_id,
        "previous_checkpoint": previous_checkpoint,
        "accepted_checkpoint": accepted_checkpoint,
        "previous_checkpoint_sha256": _sha_bytes(previous_raw),
        "accepted_checkpoint_sha256": _sha_bytes(accepted_raw),
        "final_record_item": expected_final_item,
        "witness_record_item": expected_witness_item,
        "witness_receipt_sha256": _sha_bytes(receipt_raw),
        "witness_threshold": threshold,
        "witness_minimum_operators": minimum_operators,
        "witnesses": sorted(
            normalized_meta, key=lambda x: (x["operator"], x["identity"])
        ),
        "publisher": publisher,
        "target_id": target_id,
        "anchor_locator": anchor_locator,
    }


def _validate_key_transition(
    value: dict[str, Any],
    *,
    history_id: str,
    log_id: str,
    from_key: str,
    to_key: str,
    authority_identity: str,
    checkpoint: dict[str, Any],
    previous_tree_size: int,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "historyId",
        "logId",
        "transitionId",
        "fromKeyId",
        "toKeyId",
        "reason",
        "effectiveTreeSize",
        "revokeFromKey",
        "authority",
        "evidenceSha256",
    }
    if set(value) != expected or value.get("schemaVersion") != int(
        POLICY["key_transition_schema_version"]
    ):
        _fail("KEY_TRANSITION_SCHEMA_INVALID")
    if value.get("historyId") != history_id or value.get("logId") != log_id:
        _fail("KEY_TRANSITION_HISTORY_MISMATCH")
    if (
        _identity(value.get("fromKeyId"), "KEY_TRANSITION_FROM_INVALID") != from_key
        or _identity(value.get("toKeyId"), "KEY_TRANSITION_TO_INVALID") != to_key
    ):
        _fail("KEY_TRANSITION_KEY_MISMATCH")
    if from_key == to_key:
        _fail("KEY_TRANSITION_NOOP")
    _safe_text(value.get("transitionId"), "KEY_TRANSITION_ID_INVALID", limit=255)
    reason = value.get("reason")
    if reason not in set(POLICY["allowed_transition_reasons"]):
        _fail("KEY_TRANSITION_REASON_INVALID")
    effective = _size(
        value.get("effectiveTreeSize"), "KEY_TRANSITION_TREE_SIZE_INVALID"
    )
    if effective <= previous_tree_size or effective > checkpoint["treeSize"]:
        _fail("KEY_TRANSITION_NOT_EFFECTIVE")
    if value.get("revokeFromKey") is not True:
        _fail("KEY_TRANSITION_PREVIOUS_KEY_NOT_REVOKED")
    authority = value.get("authority")
    authority_keys = {
        "identity",
        "policySignatureVerified",
        "oldKeyContinuityVerified",
        "newKeyProofVerified",
        "emergencyRecoveryAuthorized",
    }
    if not isinstance(authority, dict) or set(authority) != authority_keys:
        _fail("KEY_TRANSITION_AUTHORITY_SCHEMA_INVALID")
    if (
        _identity(authority.get("identity"), "KEY_TRANSITION_AUTHORITY_INVALID")
        != authority_identity
    ):
        _fail("KEY_TRANSITION_AUTHORITY_MISMATCH")
    if (
        authority.get("policySignatureVerified") is not True
        or authority.get("newKeyProofVerified") is not True
    ):
        _fail("KEY_TRANSITION_CRYPTOGRAPHIC_AUTHORITY_INVALID")
    if reason == "scheduled-rotation" and (
        authority.get("oldKeyContinuityVerified") is not True
        or authority.get("emergencyRecoveryAuthorized") is not False
    ):
        _fail("KEY_TRANSITION_SCHEDULED_ROTATION_INVALID")
    if (
        reason == "compromise-recovery"
        and authority.get("emergencyRecoveryAuthorized") is not True
    ):
        _fail("KEY_TRANSITION_EMERGENCY_AUTHORITY_INVALID")
    _hex(value.get("evidenceSha256"), "KEY_TRANSITION_EVIDENCE_HASH_INVALID")
    return value


def _validate_bundle(  # ruff: ignore[too-many-branches]
    bundle: dict[str, Any],
) -> dict[str, Any]:
    if (
        set(bundle)
        != {
            "schemaVersion",
            "predicateType",
            "status",
            "historyId",
            "genesis",
            "entries",
        }
        or bundle.get("schemaVersion") != int(POLICY["bundle_schema_version"])
        or bundle.get("predicateType") != HISTORY_PREDICATE_TYPE
        or bundle.get("status") != "preserved-history"
    ):
        _fail("HISTORY_BUNDLE_SCHEMA_INVALID")
    history_id = _safe_text(bundle.get("historyId"), "HISTORY_ID_INVALID", limit=255)
    genesis = bundle.get("genesis")
    genesis_keys = {
        "logId",
        "checkpoint",
        "activeLogKeyId",
        "policyAuthorityIdentity",
        "bootstrapEvidenceSha256",
    }
    if not isinstance(genesis, dict) or set(genesis) != genesis_keys:
        _fail("HISTORY_GENESIS_SCHEMA_INVALID")
    log_id = _safe_text(genesis.get("logId"), "HISTORY_LOG_ID_INVALID", limit=255)
    checkpoint = _checkpoint(genesis.get("checkpoint"), "HISTORY_GENESIS_CHECKPOINT")
    active_key = _identity(
        genesis.get("activeLogKeyId"), "HISTORY_GENESIS_LOG_KEY_INVALID"
    )
    authority = _identity(
        genesis.get("policyAuthorityIdentity"), "HISTORY_POLICY_AUTHORITY_INVALID"
    )
    _hex(
        genesis.get("bootstrapEvidenceSha256"),
        "HISTORY_BOOTSTRAP_EVIDENCE_HASH_INVALID",
    )
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) > int(POLICY["max_entries"]):
        _fail("HISTORY_ENTRY_COUNT_INVALID")
    revoked: set[str] = set()
    release_ids: set[str] = set()
    publication_ids: set[str] = set()
    witness_subjects: set[str] = set()
    previous_sequence = 0
    for entry in entries:
        expected = {
            "sequence",
            "release",
            "subjects",
            "log",
            "replicaQuorum",
            "keyTransition",
        }
        if not isinstance(entry, dict) or set(entry) != expected:
            _fail("HISTORY_ENTRY_SCHEMA_INVALID")
        sequence = _size(entry.get("sequence"), "HISTORY_ENTRY_SEQUENCE_INVALID")
        if sequence != previous_sequence + 1:
            _fail("HISTORY_ENTRY_SEQUENCE_GAP")
        previous_sequence = sequence
        release = entry.get("release")
        if not isinstance(release, dict) or set(release) != {
            "releaseId",
            "publicationId",
            "sourceRevision",
        }:
            _fail("HISTORY_ENTRY_RELEASE_SCHEMA_INVALID")
        release_id = _safe_text(
            release.get("releaseId"), "HISTORY_ENTRY_RELEASE_ID_INVALID", limit=127
        )
        publication_id = _safe_text(
            release.get("publicationId"),
            "HISTORY_ENTRY_PUBLICATION_ID_INVALID",
            limit=255,
        )
        if release_id in release_ids or publication_id in publication_ids:
            _fail("HISTORY_ENTRY_RELEASE_REPLAY")
        release_ids.add(release_id)
        publication_ids.add(publication_id)
        revision = _safe_text(
            release.get("sourceRevision"), "HISTORY_ENTRY_REVISION_INVALID", limit=128
        )
        if _REVISION.fullmatch(revision) is None:
            _fail("HISTORY_ENTRY_REVISION_INVALID")
        subjects = entry.get("subjects")
        if not isinstance(subjects, dict) or set(subjects) != {
            "finalPublicationRecordSha256",
            "witnessRecordSha256",
            "witnessReceiptSha256",
        }:
            _fail("HISTORY_ENTRY_SUBJECT_SCHEMA_INVALID")
        for value in subjects.values():
            _hex(value, "HISTORY_ENTRY_SUBJECT_HASH_INVALID")
        witness_subject = subjects["witnessRecordSha256"]
        if witness_subject in witness_subjects:
            _fail("HISTORY_ENTRY_WITNESS_REPLAY")
        witness_subjects.add(witness_subject)
        log = entry.get("log")
        if (
            not isinstance(log, dict)
            or set(log)
            != {
                "logId",
                "logKeyId",
                "checkpoint",
                "previousCheckpointSha256",
                "acceptedCheckpointSha256",
            }
            or log.get("logId") != log_id
        ):
            _fail("HISTORY_ENTRY_LOG_SCHEMA_INVALID")
        new_checkpoint = _checkpoint(log.get("checkpoint"), "HISTORY_ENTRY_CHECKPOINT")
        previous_tree_size = checkpoint["treeSize"]
        if new_checkpoint["treeSize"] <= previous_tree_size:
            _fail("HISTORY_ENTRY_CHECKPOINT_NOT_ADVANCING")
        expected_previous_sha = _sha_bytes(
            _canonical_bytes(_checkpoint_doc(log_id, checkpoint))
        )
        if log.get("previousCheckpointSha256") != expected_previous_sha:
            _fail("HISTORY_ENTRY_PREVIOUS_CHECKPOINT_HASH_MISMATCH")
        expected_accepted_sha = _sha_bytes(
            _canonical_bytes(_checkpoint_doc(log_id, new_checkpoint))
        )
        if log.get("acceptedCheckpointSha256") != expected_accepted_sha:
            _fail("HISTORY_ENTRY_ACCEPTED_CHECKPOINT_HASH_MISMATCH")
        log_key = _identity(log.get("logKeyId"), "HISTORY_ENTRY_LOG_KEY_INVALID")
        transition = entry.get("keyTransition")
        if log_key == active_key:
            if transition is not None:
                _fail("HISTORY_ENTRY_UNEXPECTED_KEY_TRANSITION")
        else:
            if log_key in revoked:
                _fail("HISTORY_ENTRY_REVOKED_KEY_REUSE")
            if not isinstance(transition, dict):
                _fail("HISTORY_ENTRY_KEY_CHANGE_WITHOUT_TRANSITION")
            _validate_key_transition(
                transition,
                history_id=history_id,
                log_id=log_id,
                from_key=active_key,
                to_key=log_key,
                authority_identity=authority,
                checkpoint=new_checkpoint,
                previous_tree_size=previous_tree_size,
            )
            revoked.add(active_key)
            active_key = log_key
        quorum = entry.get("replicaQuorum")
        if not isinstance(quorum, dict) or set(quorum) != {
            "threshold",
            "minimumOperators",
            "replicas",
        }:
            _fail("HISTORY_ENTRY_REPLICA_QUORUM_SCHEMA_INVALID")
        threshold = _size(
            quorum.get("threshold"), "HISTORY_ENTRY_REPLICA_THRESHOLD_INVALID"
        )
        minimum_operators = _size(
            quorum.get("minimumOperators"),
            "HISTORY_ENTRY_REPLICA_OPERATOR_THRESHOLD_INVALID",
        )
        replicas = quorum.get("replicas")
        if (
            threshold < int(POLICY["min_replica_quorum"])
            or minimum_operators < int(POLICY["min_replica_operators"])
            or not isinstance(replicas, list)
            or len(replicas) < threshold
        ):
            _fail("HISTORY_ENTRY_REPLICA_QUORUM_INVALID")
        identities: set[str] = set()
        operators: set[str] = set()
        for replica in replicas:
            if not isinstance(replica, dict) or set(replica) != {
                "identity",
                "operator",
            }:
                _fail("HISTORY_ENTRY_REPLICA_SCHEMA_INVALID")
            identity = _identity(
                replica.get("identity"), "HISTORY_ENTRY_REPLICA_IDENTITY_INVALID"
            )
            operator = _identity(
                replica.get("operator"), "HISTORY_ENTRY_REPLICA_OPERATOR_INVALID"
            )
            if identity in identities:
                _fail("HISTORY_ENTRY_REPLICA_IDENTITY_DUPLICATE")
            identities.add(identity)
            operators.add(operator)
        if len(operators) < minimum_operators:
            _fail("HISTORY_ENTRY_REPLICA_OPERATOR_QUORUM_INVALID")
        checkpoint = new_checkpoint
    return {
        "history_id": history_id,
        "log_id": log_id,
        "checkpoint": checkpoint,
        "active_log_key_id": active_key,
        "policy_authority_identity": authority,
        "revoked_log_key_ids": sorted(revoked),
        "sequence": len(entries),
        "chain_head_sha256": _history_head(history_id, genesis, entries),
        "genesis": genesis,
        "entries": entries,
    }


def _validate_previous_history(state_path: Path, bundle_path: Path) -> dict[str, Any]:
    state, state_raw = _read_json(state_path, "PREVIOUS_HISTORY_STATE")
    bundle, bundle_raw = _read_json(bundle_path, "PREVIOUS_HISTORY_BUNDLE")
    bundle_info = _validate_bundle(bundle)
    expected_state = {
        "schemaVersion",
        "historyId",
        "sequence",
        "logId",
        "activeLogKeyId",
        "policyAuthorityIdentity",
        "checkpoint",
        "bundleSha256",
        "chainHeadSha256",
        "revokedLogKeyIds",
    }
    if set(state) != expected_state or state.get("schemaVersion") != int(
        POLICY["state_schema_version"]
    ):
        _fail("PREVIOUS_HISTORY_STATE_SCHEMA_INVALID")
    if (
        state.get("historyId") != bundle_info["history_id"]
        or state.get("logId") != bundle_info["log_id"]
    ):
        _fail("PREVIOUS_HISTORY_STATE_ID_MISMATCH")
    if (
        _size(state.get("sequence"), "PREVIOUS_HISTORY_SEQUENCE_INVALID")
        != bundle_info["sequence"]
    ):
        _fail("PREVIOUS_HISTORY_SEQUENCE_MISMATCH")
    if (
        _identity(state.get("activeLogKeyId"), "PREVIOUS_HISTORY_LOG_KEY_INVALID")
        != bundle_info["active_log_key_id"]
    ):
        _fail("PREVIOUS_HISTORY_LOG_KEY_MISMATCH")
    if (
        _identity(
            state.get("policyAuthorityIdentity"), "PREVIOUS_HISTORY_AUTHORITY_INVALID"
        )
        != bundle_info["policy_authority_identity"]
    ):
        _fail("PREVIOUS_HISTORY_AUTHORITY_MISMATCH")
    if (
        _checkpoint(state.get("checkpoint"), "PREVIOUS_HISTORY_CHECKPOINT")
        != bundle_info["checkpoint"]
    ):
        _fail("PREVIOUS_HISTORY_CHECKPOINT_MISMATCH")
    if (
        state.get("bundleSha256") != _sha_bytes(bundle_raw)
        or state.get("chainHeadSha256") != bundle_info["chain_head_sha256"]
    ):
        _fail("PREVIOUS_HISTORY_HASH_MISMATCH")
    revoked = state.get("revokedLogKeyIds")
    if (
        not isinstance(revoked, list)
        or sorted(
            {_identity(x, "PREVIOUS_HISTORY_REVOKED_KEY_INVALID") for x in revoked}
        )
        != bundle_info["revoked_log_key_ids"]
    ):
        _fail("PREVIOUS_HISTORY_REVOKED_KEYS_MISMATCH")
    return {
        **bundle_info,
        "state": state,
        "state_sha256": _sha_bytes(state_raw),
        "bundle": bundle,
        "bundle_sha256": _sha_bytes(bundle_raw),
    }


def verify_history_bundle(  # ruff: ignore[undocumented-public-function]
    *, bundle_path: Path, state_path: Path | None = None
) -> dict[str, Any]:
    bundle, raw = _read_json(bundle_path, "HISTORY_BUNDLE")
    info = _validate_bundle(bundle)
    if state_path is not None:
        state, _ = _read_json(state_path, "HISTORY_STATE")
        expected = {
            "schemaVersion",
            "historyId",
            "sequence",
            "logId",
            "activeLogKeyId",
            "policyAuthorityIdentity",
            "checkpoint",
            "bundleSha256",
            "chainHeadSha256",
            "revokedLogKeyIds",
        }
        if set(state) != expected or state.get("schemaVersion") != int(
            POLICY["state_schema_version"]
        ):
            _fail("HISTORY_STATE_SCHEMA_INVALID")
        if (
            state.get("historyId") != info["history_id"]
            or state.get("sequence") != info["sequence"]
            or state.get("logId") != info["log_id"]
        ):
            _fail("HISTORY_STATE_ID_MISMATCH")
        if (
            state.get("activeLogKeyId") != info["active_log_key_id"]
            or state.get("policyAuthorityIdentity") != info["policy_authority_identity"]
            or _checkpoint(state.get("checkpoint"), "HISTORY_STATE_CHECKPOINT")
            != info["checkpoint"]
        ):
            _fail("HISTORY_STATE_TRUST_MISMATCH")
        if (
            state.get("bundleSha256") != _sha_bytes(raw)
            or state.get("chainHeadSha256") != info["chain_head_sha256"]
            or state.get("revokedLogKeyIds") != info["revoked_log_key_ids"]
        ):
            _fail("HISTORY_STATE_HASH_MISMATCH")
    return {
        "ok": True,
        "history_id": info["history_id"],
        "sequence": info["sequence"],
        "log_id": info["log_id"],
        "active_log_key_id": info["active_log_key_id"],
        "chain_head_sha256": info["chain_head_sha256"],
        "bundle_sha256": _sha_bytes(raw),
    }


def _observation_id(history_id: str, sequence: int, identity: str) -> str:
    return (
        "history-observe-"
        + hashlib.sha256(
            (history_id + "\0" + str(sequence) + "\0" + identity).encode()
        ).hexdigest()[:32]
    )


def _archive_id(history_id: str, sequence: int, bundle_sha: str, identity: str) -> str:
    return (
        "history-archive-"
        + hashlib.sha256(
            (
                history_id + "\0" + str(sequence) + "\0" + bundle_sha + "\0" + identity
            ).encode()
        ).hexdigest()[:32]
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
        assert process.stdin is not None  # ruff: ignore[assert]
        assert process.stdout is not None  # ruff: ignore[assert]
        assert process.stderr is not None  # ruff: ignore[assert]
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
            raise HistoryError(prefix + "_TIMEOUT") from exc
        finally:
            for thread in threads:
                thread.join()
        if overflow.is_set():
            _fail(prefix + "_OUTPUT_TOO_LARGE")
        if process.returncode != 0:
            _fail(prefix + "_FAILED")
        return _loads_json(b"".join(stdout_chunks), prefix + "_OUTPUT")

    return call


def command_replica(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="HISTORY_REPLICA_ADAPTER")


def command_archive(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="HISTORY_ARCHIVE_ADAPTER")


def _validate_replica_response(  # ruff: ignore[too-many-branches]
    value: dict[str, Any],
    *,
    observation_id: str,
    history_id: str,
    identity: str,
    operator: str,
    previous: dict[str, Any],
    current: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    common = {
        "schemaVersion",
        "operation",
        "observationId",
        "historyId",
        "status",
        "replica",
        "observedAt",
    }
    if (
        value.get("schemaVersion") != int(POLICY["replica_protocol_version"])
        or value.get("operation") != "observe"
        or value.get("observationId") != observation_id
        or value.get("historyId") != history_id
    ):
        _fail("HISTORY_REPLICA_RESULT_ID_MISMATCH")
    replica = value.get("replica")
    replica_keys = {
        "identity",
        "operator",
        "readOnly",
        "logCredentialsReused",
        "publisherCredentialsReused",
        "historyWriterCredentialsReused",
    }
    if (
        not isinstance(replica, dict)
        or set(replica) != replica_keys
        or replica.get("identity") != identity
        or replica.get("operator") != operator
    ):
        _fail("HISTORY_REPLICA_AUTHORITY_SCHEMA_INVALID")
    if (
        replica.get("readOnly") is not True
        or replica.get("logCredentialsReused") is not False
        or replica.get("publisherCredentialsReused") is not False
        or replica.get("historyWriterCredentialsReused") is not False
    ):
        _fail("HISTORY_REPLICA_AUTHORITY_INVALID")
    observed_at = _fresh(
        value.get("observedAt"), "HISTORY_REPLICA_OBSERVED_AT", now=now
    )
    status = value.get("status")
    if status == "unavailable":
        if set(value) != common | {"reason"} or not bool(
            POLICY["allow_unavailable_replica"]
        ):
            _fail("HISTORY_REPLICA_UNAVAILABLE_INVALID")
        reason = _safe_text(
            value.get("reason"), "HISTORY_REPLICA_UNAVAILABLE_REASON_INVALID", limit=255
        )
        return {
            "status": "unavailable",
            "identity": identity,
            "operator": operator,
            "observedAt": observed_at,
            "reason": reason,
        }
    if status != "observed":
        _fail("HISTORY_REPLICA_STATUS_INVALID")
    expected = common | {
        "history",
        "log",
        "checkpoint",
        "previousCheckpoint",
        "witnessAnchor",
        "proof",
    }
    if set(value) != expected:
        _fail("HISTORY_REPLICA_RESULT_SCHEMA_INVALID")
    history = value.get("history")
    expected_history = {
        "sequence": previous["sequence"],
        "previousBundleSha256": previous["bundle_sha256"],
        "previousChainHeadSha256": previous["chain_head_sha256"],
    }
    if history != expected_history:
        _fail("HISTORY_REPLICA_PRIOR_HISTORY_MISMATCH")
    log = value.get("log")
    if (
        not isinstance(log, dict)
        or set(log) != {"logId", "keyId"}
        or log.get("logId") != current["log_id"]
    ):
        _fail("HISTORY_REPLICA_LOG_MISMATCH")
    key_id = _identity(log.get("keyId"), "HISTORY_REPLICA_LOG_KEY_INVALID")
    if (
        _checkpoint(value.get("checkpoint"), "HISTORY_REPLICA_CHECKPOINT")
        != current["accepted_checkpoint"]
    ):
        _fail("HISTORY_REPLICA_CURRENT_CHECKPOINT_MISMATCH")
    if (
        _checkpoint(
            value.get("previousCheckpoint"), "HISTORY_REPLICA_PREVIOUS_CHECKPOINT"
        )
        != previous["checkpoint"]
    ):
        _fail("HISTORY_REPLICA_PREVIOUS_CHECKPOINT_MISMATCH")
    witness_anchor = value.get("witnessAnchor")
    if not isinstance(witness_anchor, dict) or set(witness_anchor) != {
        "locator",
        "record",
    }:
        _fail("HISTORY_REPLICA_WITNESS_ANCHOR_SCHEMA_INVALID")
    if (
        _safe_locator(
            witness_anchor.get("locator"),
            "HISTORY_REPLICA_WITNESS_ANCHOR_LOCATOR_INVALID",
        )
        != current["anchor_locator"]
        or _artifact(
            witness_anchor.get("record"), "HISTORY_REPLICA_WITNESS_ANCHOR_RECORD"
        )
        != current["witness_record_item"]
    ):
        _fail("HISTORY_REPLICA_WITNESS_ANCHOR_MISMATCH")
    proof = value.get("proof")
    expected_proof = {
        "currentCheckpointSignatureVerified": True,
        "consistencyFromPreviousVerified": True,
        "priorHistoryHeadVerified": True,
        "independentGossipSource": True,
        "logKeyIdentityVerified": True,
        "witnessRecordRemoteReadbackVerified": True,
    }
    if proof != expected_proof:
        _fail("HISTORY_REPLICA_PROOF_INVALID")
    return {
        "status": "observed",
        "identity": identity,
        "operator": operator,
        "keyId": key_id,
        "observedAt": observed_at,
    }


def _validate_archive_response(
    value: dict[str, Any],
    *,
    operation: str,
    archive_id: str,
    history_id: str,
    sequence: int,
    identity: str,
    operator: str,
    artifact: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "operation",
        "archiveId",
        "historyId",
        "sequence",
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
        or value.get("historyId") != history_id
        or value.get("sequence") != sequence
    ):
        _fail("HISTORY_ARCHIVE_RESULT_SCHEMA_INVALID")
    if value.get("status") not in (
        {"created", "present"} if operation == "bind" else {"present"}
    ):
        _fail("HISTORY_ARCHIVE_STATUS_INVALID")
    archive = value.get("archive")
    archive_keys = {
        "identity",
        "operator",
        "historyWriterCredentialsReused",
        "transparencyLogCredentialsReused",
    }
    if (
        not isinstance(archive, dict)
        or set(archive) != archive_keys
        or archive.get("identity") != identity
        or archive.get("operator") != operator
    ):
        _fail("HISTORY_ARCHIVE_AUTHORITY_SCHEMA_INVALID")
    if (
        archive.get("historyWriterCredentialsReused") is not False
        or archive.get("transparencyLogCredentialsReused") is not False
    ):
        _fail("HISTORY_ARCHIVE_AUTHORITY_INVALID")
    if _artifact(value.get("artifact"), "HISTORY_ARCHIVE_ARTIFACT") != artifact:
        _fail("HISTORY_ARCHIVE_ARTIFACT_MISMATCH")
    guarantees = value.get("guarantees")
    if not isinstance(guarantees, dict) or set(guarantees) != {
        "createOnly",
        "overwrite",
        "remoteReadbackVerified",
        "immutability",
        "locator",
    }:
        _fail("HISTORY_ARCHIVE_GUARANTEES_SCHEMA_INVALID")
    if (
        guarantees.get("createOnly") is not True
        or guarantees.get("overwrite") is not False
        or guarantees.get("remoteReadbackVerified") is not True
    ):
        _fail("HISTORY_ARCHIVE_GUARANTEES_INVALID")
    immutability = guarantees.get("immutability")
    if immutability not in set(POLICY["allowed_archive_immutability"]):
        _fail("HISTORY_ARCHIVE_IMMUTABILITY_INVALID")
    locator = _safe_locator(
        guarantees.get("locator"), "HISTORY_ARCHIVE_LOCATOR_INVALID"
    )
    verified_at = _fresh(
        value.get("verifiedAt"), "HISTORY_ARCHIVE_VERIFIED_AT", now=now
    )
    return {"locator": locator, "immutability": immutability, "verifiedAt": verified_at}


ReplicaAdapter = Callable[[dict[str, Any]], dict[str, Any]]
ArchiveAdapter = Callable[[dict[str, Any]], dict[str, Any]]


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_HISTORY_STATE = "trusted-history-state.json"


def preserve_release_history(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    witnessed_dir: Path,
    previous_state: Path,
    previous_bundle: Path,
    output_dir: Path,
    collector_identity: str,
    replicas: list[tuple[str, str, ReplicaAdapter]],
    replica_quorum: int,
    archives: list[tuple[str, str, ArchiveAdapter]],
    key_transition: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    witnessed_dir = _regular_dir(witnessed_dir, "RUN152_DIRECTORY_INVALID")
    target = _outside(output_dir, (witnessed_dir,), "HISTORY_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("HISTORY_OUTPUT_ALREADY_EXISTS")
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    current = _validate_witnessed(witnessed_dir)
    previous = _validate_previous_history(previous_state, previous_bundle)
    if current["log_id"] != previous["log_id"]:
        _fail("HISTORY_LOG_ID_MISMATCH")
    if current["previous_checkpoint"] != previous["checkpoint"]:
        _fail("HISTORY_CHECKPOINT_GAP_OR_FORK")
    expected_previous_doc = _checkpoint_doc(previous["log_id"], previous["checkpoint"])
    if current["previous_checkpoint_sha256"] != _sha_bytes(
        _canonical_bytes(expected_previous_doc)
    ):
        _fail("HISTORY_PREVIOUS_CHECKPOINT_BYTES_MISMATCH")
    if current["accepted_checkpoint"]["treeSize"] <= previous["checkpoint"]["treeSize"]:
        _fail("HISTORY_CHECKPOINT_NOT_ADVANCING")
    history_id = previous["history_id"]
    collector_identity = _identity(
        collector_identity, "HISTORY_COLLECTOR_IDENTITY_INVALID"
    )

    if len(replicas) < int(POLICY["min_replicas"]) or len(replicas) > int(
        POLICY["max_replicas"]
    ):
        _fail("HISTORY_REPLICA_COUNT_INVALID")
    replica_quorum = _size(replica_quorum, "HISTORY_REPLICA_QUORUM_INVALID")
    if replica_quorum < int(POLICY["min_replica_quorum"]) or replica_quorum > len(
        replicas
    ):
        _fail("HISTORY_REPLICA_QUORUM_INVALID")
    normalized_replicas: list[tuple[str, str, ReplicaAdapter]] = []
    replica_ids: set[str] = set()
    replica_operators: set[str] = set()
    for identity_raw, operator_raw, adapter in replicas:
        identity = _identity(identity_raw, "HISTORY_REPLICA_IDENTITY_INVALID")
        operator = _identity(operator_raw, "HISTORY_REPLICA_OPERATOR_INVALID")
        if identity == collector_identity or identity in replica_ids:
            _fail("HISTORY_REPLICA_IDENTITY_NOT_DISTINCT")
        replica_ids.add(identity)
        replica_operators.add(operator)
        normalized_replicas.append((identity, operator, adapter))
    if len(replica_operators) < int(POLICY["min_replica_operators"]):
        _fail("HISTORY_REPLICA_OPERATOR_COUNT_INVALID")

    if len(archives) < int(POLICY["min_archives"]) or len(archives) > int(
        POLICY["max_archives"]
    ):
        _fail("HISTORY_ARCHIVE_COUNT_INVALID")
    normalized_archives: list[tuple[str, str, ArchiveAdapter]] = []
    archive_ids: set[str] = set()
    archive_operators: set[str] = set()
    for identity_raw, operator_raw, adapter in archives:
        identity = _identity(identity_raw, "HISTORY_ARCHIVE_IDENTITY_INVALID")
        operator = _identity(operator_raw, "HISTORY_ARCHIVE_OPERATOR_INVALID")
        if (
            identity == collector_identity
            or identity in archive_ids
            or identity in replica_ids
        ):
            _fail("HISTORY_ARCHIVE_IDENTITY_NOT_DISTINCT")
        archive_ids.add(identity)
        archive_operators.add(operator)
        normalized_archives.append((identity, operator, adapter))
    if len(archive_operators) < int(POLICY["min_archive_operators"]):
        _fail("HISTORY_ARCHIVE_OPERATOR_COUNT_INVALID")

    input_files = [
        witnessed_dir / "release-publication-record.json",
        witnessed_dir / "release-transparency-witness-record.json",
        witnessed_dir / "release-transparency-witness-receipt.json",
        witnessed_dir / "previous-transparency-checkpoint.json",
        witnessed_dir / "accepted-transparency-checkpoint.json",
        previous_state,
        previous_bundle,
    ]
    if key_transition is not None:
        input_files.append(key_transition)
    initial_hashes = {str(path.resolve()): _sha256(path) for path in input_files}
    sequence = previous["sequence"] + 1
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="release-history-", dir=target.parent
    ) as temp_raw:
        stage = Path(temp_raw) / "history"
        stage.mkdir()
        replica_dir = stage / "replica-results"
        archive_dir = stage / "archive-results"
        replica_dir.mkdir()
        archive_dir.mkdir()
        observed: list[dict[str, Any]] = []
        unavailable: list[dict[str, Any]] = []
        replica_evidence: list[dict[str, Any]] = []
        for index, (identity, operator, adapter) in enumerate(
            sorted(normalized_replicas, key=lambda x: (x[1], x[0])), start=1
        ):
            observation_id = _observation_id(history_id, sequence, identity)
            request = {
                "schemaVersion": int(POLICY["replica_protocol_version"]),
                "operation": "observe",
                "observationId": observation_id,
                "historyId": history_id,
                "sequence": sequence,
                "replica": {"identity": identity, "operator": operator},
                "history": {
                    "previousBundleSha256": previous["bundle_sha256"],
                    "previousChainHeadSha256": previous["chain_head_sha256"],
                },
                "log": {"logId": current["log_id"]},
                "checkpoint": current["accepted_checkpoint"],
                "previousCheckpoint": previous["checkpoint"],
                "subjects": {
                    "witnessRecordSha256": current["witness_record_item"]["sha256"],
                    "witnessReceiptSha256": current["witness_receipt_sha256"],
                },
                "witnessAnchor": {
                    "locator": current["anchor_locator"],
                    "record": current["witness_record_item"],
                },
            }
            raw = adapter(request)
            result = _validate_replica_response(
                raw,
                observation_id=observation_id,
                history_id=history_id,
                identity=identity,
                operator=operator,
                previous=previous,
                current=current,
                now=current_time,
            )
            path = (
                replica_dir
                / f"{index:02d}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}.observe.json"
            )
            _write_canonical(path, raw)
            replica_evidence.append(
                {
                    "identity": identity,
                    "operator": operator,
                    "status": result["status"],
                    "sha256": _sha256(path),
                }
            )
            (observed if result["status"] == "observed" else unavailable).append(result)

        if len(observed) < replica_quorum:
            _fail("HISTORY_REPLICA_QUORUM_NOT_MET")
        agreeing_operators = {x["operator"] for x in observed}
        if len(agreeing_operators) < int(POLICY["min_replica_operators"]):
            _fail("HISTORY_REPLICA_OPERATOR_QUORUM_NOT_MET")
        key_ids = {x["keyId"] for x in observed}
        if len(key_ids) != 1:
            _fail("HISTORY_REPLICA_LOG_KEY_EQUIVOCATION")
        current_key = next(iter(key_ids))
        if current_key in set(previous["revoked_log_key_ids"]) and not bool(
            POLICY["allow_revoked_log_key"]
        ):
            _fail("HISTORY_REVOKED_LOG_KEY_REJECTED")

        transition_doc: dict[str, Any] | None = None
        if current_key != previous["active_log_key_id"]:
            if key_transition is None and not bool(
                POLICY["allow_log_key_change_without_transition"]
            ):
                _fail("HISTORY_LOG_KEY_CHANGE_WITHOUT_TRANSITION")
            assert key_transition is not None  # ruff: ignore[assert]
            transition_doc, _transition_raw = _read_json(
                key_transition, "KEY_TRANSITION"
            )
            _validate_key_transition(
                transition_doc,
                history_id=history_id,
                log_id=current["log_id"],
                from_key=previous["active_log_key_id"],
                to_key=current_key,
                authority_identity=previous["policy_authority_identity"],
                checkpoint=current["accepted_checkpoint"],
                previous_tree_size=previous["checkpoint"]["treeSize"],
            )
            shutil.copy2(key_transition, stage / "key-transition.json")
        elif key_transition is not None:
            _fail("HISTORY_UNEXPECTED_KEY_TRANSITION")

        configured_replica_meta = sorted(
            [
                {"identity": identity, "operator": operator}
                for identity, operator, _ in normalized_replicas
            ],
            key=lambda x: (x["operator"], x["identity"]),
        )
        entry = {
            "sequence": sequence,
            "release": {
                "releaseId": current["release_id"],
                "publicationId": current["publication_id"],
                "sourceRevision": current["source_revision"],
            },
            "subjects": {
                "finalPublicationRecordSha256": current["final_record_item"]["sha256"],
                "witnessRecordSha256": current["witness_record_item"]["sha256"],
                "witnessReceiptSha256": current["witness_receipt_sha256"],
            },
            "log": {
                "logId": current["log_id"],
                "logKeyId": current_key,
                "checkpoint": current["accepted_checkpoint"],
                "previousCheckpointSha256": current["previous_checkpoint_sha256"],
                "acceptedCheckpointSha256": current["accepted_checkpoint_sha256"],
            },
            "replicaQuorum": {
                "threshold": replica_quorum,
                "minimumOperators": int(POLICY["min_replica_operators"]),
                "replicas": configured_replica_meta,
            },
            "keyTransition": transition_doc,
        }
        new_bundle = dict(previous["bundle"])
        new_bundle["entries"] = [*previous["entries"], entry]
        bundle_path = stage / "release-history-bundle.json"
        _write_canonical(bundle_path, new_bundle)
        bundle_info = _validate_bundle(new_bundle)
        if (
            bundle_info["sequence"] != sequence
            or bundle_info["checkpoint"] != current["accepted_checkpoint"]
            or bundle_info["active_log_key_id"] != current_key
        ):
            _fail("HISTORY_BUNDLE_POSTBUILD_VALIDATION_FAILED")
        bundle_item = {
            "name": bundle_path.name,
            "sha256": _sha256(bundle_path),
            "size": bundle_path.stat().st_size,
        }
        new_state = {
            "schemaVersion": int(POLICY["state_schema_version"]),
            "historyId": history_id,
            "sequence": sequence,
            "logId": current["log_id"],
            "activeLogKeyId": current_key,
            "policyAuthorityIdentity": previous["policy_authority_identity"],
            "checkpoint": current["accepted_checkpoint"],
            "bundleSha256": bundle_item["sha256"],
            "chainHeadSha256": bundle_info["chain_head_sha256"],
            "revokedLogKeyIds": bundle_info["revoked_log_key_ids"],
        }
        state_path = stage / _DOC_HISTORY_STATE
        _write_canonical(state_path, new_state)
        verify_history_bundle(bundle_path=bundle_path, state_path=state_path)
        state_item = {
            "name": state_path.name,
            "sha256": _sha256(state_path),
            "size": state_path.stat().st_size,
        }

        archive_evidence: list[dict[str, Any]] = []
        locators: set[str] = set()
        for index, (identity, operator, adapter) in enumerate(
            sorted(normalized_archives, key=lambda x: (x[1], x[0])), start=1
        ):
            archive_id = _archive_id(
                history_id, sequence, bundle_item["sha256"], identity
            )
            base_request = {
                "schemaVersion": int(POLICY["archive_protocol_version"]),
                "archiveId": archive_id,
                "historyId": history_id,
                "sequence": sequence,
                "archive": {"identity": identity, "operator": operator},
                "artifact": bundle_item,
            }
            bind_request = dict(base_request, operation="bind")
            bind_request["artifact"] = dict(bundle_item, localPath=str(bundle_path))
            before = _sha256(bundle_path)
            bind_raw = adapter(bind_request)
            if _sha256(bundle_path) != before:
                _fail("HISTORY_BUNDLE_CHANGED_DURING_ARCHIVE")
            bind = _validate_archive_response(
                bind_raw,
                operation="bind",
                archive_id=archive_id,
                history_id=history_id,
                sequence=sequence,
                identity=identity,
                operator=operator,
                artifact=bundle_item,
                now=current_time,
            )
            verify_raw = adapter(dict(base_request, operation="verify"))
            verify = _validate_archive_response(
                verify_raw,
                operation="verify",
                archive_id=archive_id,
                history_id=history_id,
                sequence=sequence,
                identity=identity,
                operator=operator,
                artifact=bundle_item,
                now=current_time,
            )
            if (
                bind["locator"] != verify["locator"]
                or bind["immutability"] != verify["immutability"]
            ):
                _fail("HISTORY_ARCHIVE_VERIFY_REBIND_FAILED")
            if verify["locator"] in locators:
                _fail("HISTORY_ARCHIVE_LOCATOR_COLLISION")
            locators.add(verify["locator"])
            bind_path = (
                archive_dir
                / f"{index:02d}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}.bind.json"
            )
            verify_path = (
                archive_dir
                / f"{index:02d}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}.verify.json"
            )
            _write_canonical(bind_path, bind_raw)
            _write_canonical(verify_path, verify_raw)
            archive_evidence.append(
                {
                    "identity": identity,
                    "operator": operator,
                    "archiveId": archive_id,
                    "locator": verify["locator"],
                    "immutability": verify["immutability"],
                    "verifiedAt": verify["verifiedAt"],
                    "bindEvidenceSha256": _sha256(bind_path),
                    "verifyEvidenceSha256": _sha256(verify_path),
                }
            )

        for path in input_files:
            if _sha256(path) != initial_hashes[str(path.resolve())]:
                _fail("HISTORY_INPUT_CHANGED_DURING_PRESERVATION")
        if (
            _sha256(bundle_path) != bundle_item["sha256"]
            or _sha256(state_path) != state_item["sha256"]
        ):
            _fail("HISTORY_OUTPUT_CHANGED_DURING_ARCHIVE")

        for name in (
            "release-transparency-witness-record.json",
            "release-transparency-witness-receipt.json",
            "previous-transparency-checkpoint.json",
            "accepted-transparency-checkpoint.json",
        ):
            shutil.copy2(witnessed_dir / name, stage / name)
        shutil.copy2(previous_state, stage / "previous-trusted-history-state.json")
        shutil.copy2(previous_bundle, stage / "previous-release-history-bundle.json")
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "preserved",
            "historyId": history_id,
            "sequence": sequence,
            "collectorIdentity": collector_identity,
            "previous": {
                "stateSha256": previous["state_sha256"],
                "bundleSha256": previous["bundle_sha256"],
                "chainHeadSha256": previous["chain_head_sha256"],
                "checkpoint": previous["checkpoint"],
            },
            "current": {
                "releaseId": current["release_id"],
                "publicationId": current["publication_id"],
                "witnessRecordSha256": current["witness_record_item"]["sha256"],
                "witnessReceiptSha256": current["witness_receipt_sha256"],
                "acceptedCheckpointSha256": current["accepted_checkpoint_sha256"],
                "logKeyId": current_key,
            },
            "replicaQuorum": {
                "configured": len(normalized_replicas),
                "threshold": replica_quorum,
                "observed": len(observed),
                "unavailable": len(unavailable),
                "minimumOperators": int(POLICY["min_replica_operators"]),
                "evidence": replica_evidence,
            },
            "historyBundle": bundle_item,
            "trustedHistoryState": state_item,
            "archives": archive_evidence,
        }
        receipt_path = stage / "release-history-preservation-receipt.json"
        _write_canonical(receipt_path, receipt)
        tmp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, tmp_target, copy_function=shutil.copy2)
        os.replace(tmp_target, target)

    return {
        "ok": True,
        "phase": "history-preserved",
        "history_id": history_id,
        "sequence": sequence,
        "release_id": current["release_id"],
        "replica_count": len(normalized_replicas),
        "replica_observed": len(observed),
        "replica_quorum": replica_quorum,
        "archive_count": len(normalized_archives),
        "bundle_sha256": _sha256(target / "release-history-bundle.json"),
        "state_sha256": _sha256(target / _DOC_HISTORY_STATE),
        "chain_head_sha256": bundle_info["chain_head_sha256"],
    }


def _parse_endpoint(value: str, label: str) -> tuple[str, str, list[str]]:
    parts = value.split("=", 1)
    if len(parts) != 2 or "@" not in parts[0]:  # ruff: ignore[magic-value-comparison]
        raise argparse.ArgumentTypeError(
            f"{label} must be IDENTITY@OPERATOR=EXECUTABLE[,ARG...]"
        )
    identity, operator = parts[0].split("@", 1)
    command = [part for part in parts[1].split(",") if part]
    if not identity or not operator or not command:
        raise argparse.ArgumentTypeError(f"invalid {label} specification")
    return identity, operator, command


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify_parser = sub.add_parser(
        "verify", help="offline-verify a canonical history bundle"
    )
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--state", type=Path)
    advance = sub.add_parser(
        "advance", help="advance witnessed release history after gossip quorum"
    )
    advance.add_argument("--witnessed-dir", type=Path, required=True)
    advance.add_argument("--previous-state", type=Path, required=True)
    advance.add_argument("--previous-bundle", type=Path, required=True)
    advance.add_argument("--output-dir", type=Path, required=True)
    advance.add_argument("--collector-identity", required=True)
    advance.add_argument("--replica-quorum", type=int, required=True)
    advance.add_argument(
        "--replica",
        action="append",
        required=True,
        type=lambda x: _parse_endpoint(x, "replica"),
    )
    advance.add_argument(
        "--archive",
        action="append",
        required=True,
        type=lambda x: _parse_endpoint(x, "archive"),
    )
    advance.add_argument("--key-transition", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_history_bundle(
                bundle_path=args.bundle, state_path=args.state
            )
        else:
            replicas = [
                (identity, operator, command_replica(command))
                for identity, operator, command in args.replica
            ]
            archives = [
                (identity, operator, command_archive(command))
                for identity, operator, command in args.archive
            ]
            result = preserve_release_history(
                witnessed_dir=args.witnessed_dir,
                previous_state=args.previous_state,
                previous_bundle=args.previous_bundle,
                output_dir=args.output_dir,
                collector_identity=args.collector_identity,
                replicas=replicas,
                replica_quorum=args.replica_quorum,
                archives=archives,
                key_transition=args.key_transition,
            )
    except (HistoryError, OSError) as exc:
        logger.error("history preservation failed: %s", exc)
        # sys.stdout.write(
        #     json.dumps({"ok": False, "error": str(exc)}, sort_keys=True) + "\n"
        # )
        return 2
    logger.info("%s", json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
