"""
Run 168: durably preserve and recover complete Run 167 recursive Merkle authority state.

Preservation first executes the full Run 167 verifier.  It then freezes the exact four
canonical Run 167 artifacts into one deterministic recovery checkpoint, replicates that
checkpoint to multiple independently operated immutable archives, and requires separate
read-only verification of every accepted copy.

Recovery does not trust majority voting over differing bytes.  Every observed archive
must return the same canonical checkpoint and callers must additionally pin the checkpoint
SHA-256, Run 167 authority head, Merkle continuity head, active-authority SHA-256, and
sequence out of band.  A recovered environment can therefore reconstruct the current
log/gossip authority, permanent revocations, last RFC6962 checkpoints, and all Run 167
history without any intermediate online service state.
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
import stat
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
    from . import rebridge_archive_merkle_authority as run167
except (ImportError, ValueError) as exc:
    import importlib.util

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "_run167_rebridge_for_recovery", _here / "rebridge_archive_merkle_authority.py"
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("rebridge_archive_merkle_authority.py") from exc
    run167 = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = run167
    _spec.loader.exec_module(run167)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads(
    (HERE / "release_archive_merkle_recovery_policy.toml").read_text()
)
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
_DOC_ARCHIVE_LOG_AUTHORITY_STATE = "trusted-archive-log-authority-state.json"
_DOC_ARCHIVE_MERKLE_CONTINUITY_STATE = "trusted-archive-merkle-continuity-state.json"
_DOC_ARCHIVE_MERKLE_REBRIDGE_STATE = "trusted-archive-merkle-rebridge-state.json"
_DOC_ARCHIVE_MERKLE_RECOVERY_STATE = "trusted-archive-merkle-recovery-state.json"


_RUN167_NAMES = {
    "release-archive-merkle-rebridge-bundle.json",
    _DOC_ARCHIVE_MERKLE_REBRIDGE_STATE,
    "active-archive-merkle-rebridge.json",
    "release-archive-merkle-rebridge-receipt.json",
}
_OUTPUT_NAMES = {
    "release-archive-merkle-recovery-checkpoint.json",
    _DOC_ARCHIVE_MERKLE_RECOVERY_STATE,
    "release-archive-merkle-recovery-receipt.json",
}

ArchiveAdapter = Callable[[dict[str, Any]], dict[str, Any]]
VerifierAdapter = Callable[[dict[str, Any]], dict[str, Any]]
RecoveryAdapter = Callable[[dict[str, Any]], dict[str, Any]]


class ArchiveMerkleRecoveryError(RuntimeError):
    """Run 168 recovery durability invariant failed."""


def _fail(code: str) -> None:
    raise ArchiveMerkleRecoveryError(code)


def _canonical(value: Any) -> bytes:
    return run167._canonical(value)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value))


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            _fail("ARCHIVE_MERKLE_RECOVERY_JSON_DUPLICATE_KEY")
        out[k] = v
    return out


def _loads(raw: bytes, code: str) -> dict[str, Any]:
    if len(raw) > int(POLICY["max_json_bytes"]):
        _fail(code + "_TOO_LARGE")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes)
    except ArchiveMerkleRecoveryError:
        raise
    except Exception as exc:
        raise ArchiveMerkleRecoveryError(code + "_JSON_INVALID") from exc
    if not isinstance(value, dict):
        _fail(code + "_SCHEMA_INVALID")
    if _canonical(value) != raw:
        _fail(code + "_NOT_CANONICAL")
    return value


def _read_json(path: Path, code: str) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    try:
        st = path.lstat()
    except OSError:
        _fail(code + "_MISSING")
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        _fail(code + "_NOT_REGULAR")
    raw = path.read_bytes()
    return _loads(raw, code), raw


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


def _positive_int(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(code)
    return value


def _size(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(code)
    return value


def _dt(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ArchiveMerkleRecoveryError(code) from exc
    if parsed.tzinfo is None:
        _fail(code)
    return parsed.astimezone(timezone.utc)


def _fresh(value: Any, code: str, *, now: datetime, historical: bool = False) -> str:
    parsed = _dt(value, code)
    if historical:
        return value
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    age = timedelta(minutes=int(POLICY["max_result_age_minutes"]))
    if parsed > now + skew:
        _fail(code + "_FROM_FUTURE")
    if now - parsed > age:
        _fail(code + "_STALE")
    return value


def _regular_dir(path: Path, code: str) -> Path:
    path = Path(path)
    try:
        st = path.lstat()
    except OSError:
        _fail(code)
    if not stat.S_ISDIR(st.st_mode):
        _fail(code)
    return path.resolve()


def _outside(path: Path, protected: Iterable[Path], code: str) -> Path:
    target = Path(path).expanduser().absolute()
    for item in protected:
        root = Path(item).absolute()
        if target == root or root in target.parents or target in root.parents:
            _fail(code)
    return target


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


def _artifact(name: str, raw: bytes) -> dict[str, Any]:
    return {"name": name, "sha256": _sha_bytes(raw), "size": len(raw)}


def _artifact_doc(value: Any, code: str) -> dict[str, Any]:
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


def _dir_fingerprint(root: Path, code: str) -> tuple[tuple[str, str, int, int], ...]:
    root = _regular_dir(root, code)
    rows = []
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        if stat.S_ISDIR(st.st_mode):
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            _fail(code)
        rows.append(
            (
                p.relative_to(root).as_posix(),
                _sha(p),
                st.st_size,
                stat.S_IMODE(st.st_mode),
            )
        )
    return tuple(rows)


def _authority_fingerprint(
    run167_dir: Path, verify_kwargs: dict[str, Any]
) -> tuple[Any, ...]:
    rows: list[Any] = [
        (
            "run167",
            _dir_fingerprint(run167_dir, "ARCHIVE_MERKLE_RECOVERY_INPUT_INVALID"),
        )
    ]
    seen: set[str] = set()
    for key, value in sorted(verify_kwargs.items()):
        if key in {"output_dir", "now", "historical"}:
            continue
        values = value if isinstance(value, list) else [value]
        for idx, item in enumerate(values):
            if not isinstance(item, Path):
                continue
            ap = str(item.absolute())
            token = f"{key}:{idx}:{ap}"
            if token in seen:
                continue
            seen.add(token)
            if item.is_dir():
                rows.append(
                    (
                        key,
                        idx,
                        ap,
                        "dir",
                        _dir_fingerprint(
                            item, "ARCHIVE_MERKLE_RECOVERY_AUTHORITY_INPUT_INVALID"
                        ),
                    )
                )
            else:
                try:
                    st = item.lstat()
                except OSError:
                    _fail("ARCHIVE_MERKLE_RECOVERY_AUTHORITY_INPUT_INVALID")
                if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                    _fail("ARCHIVE_MERKLE_RECOVERY_AUTHORITY_INPUT_INVALID")
                rows.append(
                    (
                        key,
                        idx,
                        ap,
                        "file",
                        _sha(item),
                        st.st_size,
                        stat.S_IMODE(st.st_mode),
                    )
                )
    return tuple(rows)


def _run167_docs(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_MERKLE_RECOVERY_RUN167_DIR_INVALID")
    if {p.name for p in root.iterdir()} != _RUN167_NAMES:
        _fail("ARCHIVE_MERKLE_RECOVERY_RUN167_ALLOWLIST_INVALID")
    docs: dict[str, dict[str, Any]] = {}
    raws: dict[str, bytes] = {}
    for name in sorted(_RUN167_NAMES):
        docs[name], raws[name] = _read_json(
            root / name, "ARCHIVE_MERKLE_RECOVERY_RUN167"
        )
    return docs, raws


def _artifact_map(raws: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {name: _artifact(name, raws[name]) for name in sorted(raws)}


def _verify_snapshot_documents(  # ruff: ignore[too-many-branches]
    docs: dict[str, dict[str, Any]], raws: dict[str, bytes]
) -> dict[str, Any]:
    if set(docs) != _RUN167_NAMES or set(raws) != _RUN167_NAMES:
        _fail("ARCHIVE_MERKLE_RECOVERY_RUN167_DOCUMENT_SET_INVALID")
    bundle = docs["release-archive-merkle-rebridge-bundle.json"]
    state = docs[_DOC_ARCHIVE_MERKLE_REBRIDGE_STATE]
    active = docs["active-archive-merkle-rebridge.json"]
    receipt = docs["release-archive-merkle-rebridge-receipt.json"]
    events = bundle.get("events")
    recs = receipt.get("events")
    if (
        not isinstance(events, list)
        or not events
        or not isinstance(recs, list)
        or len(events) != len(recs)
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_HISTORY_INVALID")
    braw = raws["release-archive-merkle-rebridge-bundle.json"]
    if state.get("bundleArtifact") != _artifact(
        "release-archive-merkle-rebridge-bundle.json", braw
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_BUNDLE_REBIND_INVALID")
    last = events[-1]
    if active != run167._make_active(state, last):
        _fail("ARCHIVE_MERKLE_RECOVERY_ACTIVE_REBIND_INVALID")
    sequence = _positive_int(
        state.get("sequence"), "ARCHIVE_MERKLE_RECOVERY_SEQUENCE_INVALID"
    )
    if last.get("sequence") != sequence or active.get("sequence") != sequence:
        _fail("ARCHIVE_MERKLE_RECOVERY_SEQUENCE_MISMATCH")
    base_sequence = _positive_int(
        bundle.get("baseRun166Sequence"),
        "ARCHIVE_MERKLE_RECOVERY_BASE_SEQUENCE_INVALID",
    )
    if sequence != base_sequence + len(events):
        _fail("ARCHIVE_MERKLE_RECOVERY_SEQUENCE_CHAIN_INVALID")
    run165_docs = receipt.get("run165Documents")
    run166_docs = receipt.get("run166Documents")
    if not isinstance(run165_docs, dict) or not isinstance(run166_docs, dict):
        _fail("ARCHIVE_MERKLE_RECOVERY_BASE_DOCUMENTS_INVALID")
    try:
        r165_state = run165_docs[_DOC_ARCHIVE_LOG_AUTHORITY_STATE]
        r166_state = run166_docs[_DOC_ARCHIVE_MERKLE_CONTINUITY_STATE]
        r166_active = run166_docs["active-archive-merkle-continuity.json"]
    except Exception:  # ruff: ignore[blind-except]
        _fail("ARCHIVE_MERKLE_RECOVERY_BASE_DOCUMENTS_INVALID")
    r165_raws = {name: _canonical(value) for name, value in run165_docs.items()}
    r166_raws = {name: _canonical(value) for name, value in run166_docs.items()}
    if bundle.get("run165Artifacts") != run167._artifact_map(r165_raws) or bundle.get(
        "baseRun166Artifacts"
    ) != run167._artifact_map(r166_raws):
        _fail("ARCHIVE_MERKLE_RECOVERY_BASE_ARTIFACT_REBIND_INVALID")
    if bundle.get("run165Sequence") != r165_state.get("sequence") or bundle.get(
        "run165LogAuthorityChainHeadSha256"
    ) != r165_state.get("logAuthorityChainHeadSha256"):
        _fail("ARCHIVE_MERKLE_RECOVERY_RUN165_REBIND_INVALID")
    if bundle.get("baseRun166Sequence") != r166_state.get("sequence") or bundle.get(
        "baseRun166MerkleAuthorityContinuityHeadSha256"
    ) != r166_state.get("merkleAuthorityContinuityHeadSha256"):
        _fail("ARCHIVE_MERKLE_RECOVERY_RUN166_REBIND_INVALID")
    r165_info = {"state": r165_state}
    r166_info = {"state": r166_state, "active": r166_active, "raws": r166_raws}
    authority_head = run167._base_authority_head(r165_info, r166_info)
    merkle_head = bundle["baseRun166MerkleAuthorityContinuityHeadSha256"]
    for idx, (event, rec) in enumerate(zip(events, recs), 1):
        if (
            not isinstance(event, dict)
            or not isinstance(rec, dict)
            or event.get("sequence") != base_sequence + idx
            or rec.get("sequence") != event.get("sequence")
        ):
            _fail("ARCHIVE_MERKLE_RECOVERY_EVENT_SEQUENCE_INVALID")
        action = event.get("action")
        if action == "rebridge":
            td = rec.get("transitionDocument")
            hd = rec.get("handoffProofDocument")
            if not isinstance(td, dict) or not isinstance(hd, dict):
                _fail("ARCHIVE_MERKLE_RECOVERY_HANDOFF_DOCUMENT_INVALID")
            tsha = _sha_bytes(_canonical(td))
            hsha = _sha_bytes(_canonical(hd))
            if (
                event.get("transitionSha256") != tsha
                or event.get("handoffSha256") != hsha
            ):
                _fail("ARCHIVE_MERKLE_RECOVERY_HANDOFF_HASH_INVALID")
            authority_head = run167._bridge_head(
                authority_head,
                tsha,
                hsha,
                _hex(
                    event.get("sourceTrustedStateSha256"),
                    "ARCHIVE_MERKLE_RECOVERY_SOURCE_STATE_INVALID",
                ),
            )
        elif action == "append":
            if (
                rec.get("transitionDocument") is not None
                or rec.get("handoffProofDocument") is not None
            ):
                _fail("ARCHIVE_MERKLE_RECOVERY_APPEND_HANDOFF_INVALID")
        else:
            _fail("ARCHIVE_MERKLE_RECOVERY_ACTION_INVALID")
        if event.get("archiveMerkleRebridgeAuthorityHeadSha256") != authority_head:
            _fail("ARCHIVE_MERKLE_RECOVERY_AUTHORITY_HEAD_INVALID")
        bare = {k: event[k] for k in event if k != "merkleRebridgeContinuityHeadSha256"}
        merkle_head = run167._merkle_head(merkle_head, bare)
        if event.get("merkleRebridgeContinuityHeadSha256") != merkle_head:
            _fail("ARCHIVE_MERKLE_RECOVERY_MERKLE_HEAD_INVALID")
    if (
        state.get("archiveMerkleRebridgeAuthorityHeadSha256") != authority_head
        or active.get("archiveMerkleRebridgeAuthorityHeadSha256") != authority_head
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_FINAL_AUTHORITY_HEAD_INVALID")
    if (
        state.get("merkleRebridgeContinuityHeadSha256") != merkle_head
        or active.get("merkleRebridgeContinuityHeadSha256") != merkle_head
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_FINAL_MERKLE_HEAD_INVALID")
    authority = state.get("activeAuthority")
    if not isinstance(authority, dict) or active.get("authority") != authority:
        _fail("ARCHIVE_MERKLE_RECOVERY_ACTIVE_AUTHORITY_INVALID")
    active_sha = _sha_bytes(_canonical(authority))
    if (
        state.get("activeAuthoritySha256") != active_sha
        or active.get("activeAuthoritySha256") != active_sha
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_ACTIVE_AUTHORITY_HASH_INVALID")
    revoked = state.get("revokedKeyFingerprints")
    if (
        not isinstance(revoked, list)
        or revoked != sorted(set(revoked))
        or any(_HEX64.fullmatch(x) is None for x in revoked if isinstance(x, str))
        or any(not isinstance(x, str) for x in revoked)
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_REVOKED_SET_INVALID")
    checkpoints = state.get("lastCheckpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(authority):
        _fail("ARCHIVE_MERKLE_RECOVERY_CHECKPOINT_SET_INVALID")
    log_rows = last.get("logs")
    if not isinstance(log_rows, list):
        _fail("ARCHIVE_MERKLE_RECOVERY_LAST_LOG_ROWS_INVALID")
    from_rows = {
        r.get("logId"): {
            k: r.get(k) for k in ("checkpointSha256", "treeSize", "rootHash")
        }
        for r in log_rows
        if isinstance(r, dict)
    }
    if checkpoints != from_rows:
        _fail("ARCHIVE_MERKLE_RECOVERY_LAST_CHECKPOINT_REBIND_INVALID")
    return {
        "sequence": sequence,
        "postRun166Sequence": state.get("postRun166Sequence"),
        "rebridgeSequence": state.get("rebridgeSequence"),
        "authorityHead": authority_head,
        "merkleHead": merkle_head,
        "activeAuthority": authority,
        "activeAuthoritySha256": active_sha,
        "revokedKeyFingerprints": revoked,
        "lastCheckpoints": checkpoints,
        "run165Sequence": state.get("run165Sequence"),
        "run165LogAuthorityChainHeadSha256": state.get(
            "run165LogAuthorityChainHeadSha256"
        ),
    }


def _checkpoint_head(checkpoint_without_head: dict[str, Any]) -> str:
    return _sha_bytes(
        _canonical(
            {
                "domain": "run168-archive-merkle-recovery-checkpoint",
                "checkpoint": checkpoint_without_head,
            }
        )
    )


def _make_checkpoint(
    docs: dict[str, dict[str, Any]],
    raws: dict[str, bytes],
    summary: dict[str, Any],
    membership: list[dict[str, str]],
) -> dict[str, Any]:
    body = {
        "schemaVersion": int(POLICY["checkpoint_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "archive-merkle-rebridge-recovery-checkpoint",
        "run167Artifacts": _artifact_map(raws),
        "run167Documents": {name: docs[name] for name in sorted(docs)},
        "summary": summary,
        "archiveMembership": membership,
    }
    return {**body, "recoveryCheckpointHeadSha256": _checkpoint_head(body)}


def _verify_checkpoint(doc: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "predicateType",
        "status",
        "run167Artifacts",
        "run167Documents",
        "summary",
        "archiveMembership",
        "recoveryCheckpointHeadSha256",
    }
    if (
        set(doc) != expected
        or doc.get("schemaVersion") != int(POLICY["checkpoint_schema_version"])
        or doc.get("predicateType") != PREDICATE_TYPE
        or doc.get("status") != "archive-merkle-rebridge-recovery-checkpoint"
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_CHECKPOINT_SCHEMA_INVALID")
    body = {k: doc[k] for k in doc if k != "recoveryCheckpointHeadSha256"}
    if doc["recoveryCheckpointHeadSha256"] != _checkpoint_head(body):
        _fail("ARCHIVE_MERKLE_RECOVERY_CHECKPOINT_HEAD_INVALID")
    documents = doc.get("run167Documents")
    if not isinstance(documents, dict) or set(documents) != _RUN167_NAMES:
        _fail("ARCHIVE_MERKLE_RECOVERY_CHECKPOINT_DOCUMENTS_INVALID")
    raws = {name: _canonical(documents[name]) for name in sorted(documents)}
    if doc.get("run167Artifacts") != _artifact_map(raws):
        _fail("ARCHIVE_MERKLE_RECOVERY_CHECKPOINT_ARTIFACTS_INVALID")
    summary = _verify_snapshot_documents(documents, raws)
    if doc.get("summary") != summary:
        _fail("ARCHIVE_MERKLE_RECOVERY_CHECKPOINT_SUMMARY_INVALID")
    membership = doc.get("archiveMembership")
    if not isinstance(membership, list) or len(membership) < int(
        POLICY["min_configured_archives"]
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_MEMBERSHIP_INVALID")
    expected_keys = {
        "archiveIdentity",
        "archiveOperator",
        "verifierIdentity",
        "verifierOperator",
    }
    normalized = []
    for row in membership:
        if not isinstance(row, dict) or set(row) != expected_keys:
            _fail("ARCHIVE_MERKLE_RECOVERY_MEMBERSHIP_INVALID")
        normalized.append(
            {
                k: _identity(
                    row[k], "ARCHIVE_MERKLE_RECOVERY_MEMBERSHIP_IDENTITY_INVALID"
                )
                for k in sorted(row)
            }
        )
    if membership != sorted(
        normalized,
        key=lambda x: (
            x["archiveOperator"],
            x["archiveIdentity"],
            x["verifierOperator"],
            x["verifierIdentity"],
        ),
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_MEMBERSHIP_NOT_NORMALIZED")
    if len({r["archiveIdentity"] for r in membership}) != len(membership) or len(
        {r["verifierIdentity"] for r in membership}
    ) != len(membership):
        _fail("ARCHIVE_MERKLE_RECOVERY_MEMBERSHIP_IDENTITY_REUSE")
    if len({r["archiveOperator"] for r in membership}) < int(
        POLICY["min_archive_operators"]
    ) or len({r["verifierOperator"] for r in membership}) < int(
        POLICY["min_verifier_operators"]
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_MEMBERSHIP_OPERATOR_QUORUM_INVALID")
    if {r["archiveIdentity"] for r in membership} & {
        r["verifierIdentity"] for r in membership
    } or {r["archiveOperator"] for r in membership} & {
        r["verifierOperator"] for r in membership
    }:
        _fail("ARCHIVE_MERKLE_RECOVERY_AUTHORITY_PLANE_OVERLAP")
    return {
        "summary": summary,
        "raws": raws,
        "documents": documents,
        "membership": membership,
    }


def _archive_id(checkpoint_sha: str, identity: str) -> str:
    return _sha_bytes(
        _canonical({"checkpointSha256": checkpoint_sha, "archiveIdentity": identity})
    )


def _validate_archive_response(
    value: Any,
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
        "status",
        "archiveId",
        "archiveIdentity",
        "archiveOperator",
        "artifact",
        "locator",
        "immutableVersionId",
        "immutabilityClass",
        "overwrite",
        "readBackSha256",
        "readBackSize",
        "observedAt",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schemaVersion") != int(POLICY["archive_protocol_version"])
        or value.get("operation") != operation
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_ARCHIVE_RESPONSE_SCHEMA_INVALID")
    if (
        value.get("status") not in {"created", "present"}
        or value.get("archiveId") != archive_id
        or value.get("archiveIdentity") != identity
        or value.get("archiveOperator") != operator
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_ARCHIVE_RESPONSE_BINDING_INVALID")
    if (
        value.get("artifact") != artifact
        or value.get("overwrite") is not False
        or value.get("readBackSha256") != artifact["sha256"]
        or value.get("readBackSize") != artifact["size"]
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_ARCHIVE_READBACK_INVALID")
    imm = value.get("immutabilityClass")
    if imm not in set(POLICY["allowed_archive_immutability"]):
        _fail("ARCHIVE_MERKLE_RECOVERY_ARCHIVE_IMMUTABILITY_INVALID")
    return {
        "schemaVersion": value["schemaVersion"],
        "operation": operation,
        "status": value["status"],
        "archiveId": archive_id,
        "archiveIdentity": identity,
        "archiveOperator": operator,
        "artifact": artifact,
        "locator": _safe_locator(
            value.get("locator"), "ARCHIVE_MERKLE_RECOVERY_ARCHIVE_LOCATOR_INVALID"
        ),
        "immutableVersionId": _identity(
            value.get("immutableVersionId"),
            "ARCHIVE_MERKLE_RECOVERY_ARCHIVE_VERSION_INVALID",
        ),
        "immutabilityClass": imm,
        "overwrite": False,
        "readBackSha256": artifact["sha256"],
        "readBackSize": artifact["size"],
        "observedAt": _fresh(
            value.get("observedAt"),
            "ARCHIVE_MERKLE_RECOVERY_ARCHIVE_OBSERVED_AT",
            now=now,
            historical=historical,
        ),
    }


def _validate_verifier_response(
    value: Any,
    *,
    archive: dict[str, Any],
    identity: str,
    operator: str,
    artifact: dict[str, Any],
    now: datetime,
    historical: bool = False,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "status",
        "verifierIdentity",
        "verifierOperator",
        "archiveId",
        "archiveIdentity",
        "archiveOperator",
        "artifact",
        "locator",
        "immutableVersionId",
        "readBackSha256",
        "readBackSize",
        "readOnly",
        "archiveCredentialsReused",
        "observedAt",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schemaVersion") != int(POLICY["verifier_protocol_version"])
        or value.get("status") != "verified"
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_VERIFIER_RESPONSE_SCHEMA_INVALID")
    if (
        value.get("verifierIdentity") != identity
        or value.get("verifierOperator") != operator
        or value.get("archiveId") != archive["archiveId"]
        or value.get("archiveIdentity") != archive["archiveIdentity"]
        or value.get("archiveOperator") != archive["archiveOperator"]
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_VERIFIER_RESPONSE_BINDING_INVALID")
    if (
        value.get("artifact") != artifact
        or value.get("locator") != archive["locator"]
        or value.get("immutableVersionId") != archive["immutableVersionId"]
        or value.get("readBackSha256") != artifact["sha256"]
        or value.get("readBackSize") != artifact["size"]
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_VERIFIER_READBACK_INVALID")
    if (
        value.get("readOnly") is not True
        or value.get("archiveCredentialsReused") is not False
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_VERIFIER_AUTHORITY_INVALID")
    return {
        "schemaVersion": value["schemaVersion"],
        "status": "verified",
        "verifierIdentity": identity,
        "verifierOperator": operator,
        "archiveId": archive["archiveId"],
        "archiveIdentity": archive["archiveIdentity"],
        "archiveOperator": archive["archiveOperator"],
        "artifact": artifact,
        "locator": archive["locator"],
        "immutableVersionId": archive["immutableVersionId"],
        "readBackSha256": artifact["sha256"],
        "readBackSize": artifact["size"],
        "readOnly": True,
        "archiveCredentialsReused": False,
        "observedAt": _fresh(
            value.get("observedAt"),
            "ARCHIVE_MERKLE_RECOVERY_VERIFIER_OBSERVED_AT",
            now=now,
            historical=historical,
        ),
    }


def _load_output(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_MERKLE_RECOVERY_OUTPUT_INVALID")
    if {p.name for p in root.iterdir()} != _OUTPUT_NAMES:
        _fail("ARCHIVE_MERKLE_RECOVERY_OUTPUT_ALLOWLIST_INVALID")
    docs = {}
    raws = {}
    for name in sorted(_OUTPUT_NAMES):
        docs[name], raws[name] = _read_json(
            root / name, "ARCHIVE_MERKLE_RECOVERY_OUTPUT"
        )
    return docs, raws


def verify_recovery_archive(  # ruff: ignore[undocumented-public-function]
    *, output_dir: Path, now: datetime | None = None, historical: bool = False
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    docs, raws = _load_output(output_dir)
    checkpoint = docs["release-archive-merkle-recovery-checkpoint.json"]
    info = _verify_checkpoint(checkpoint)
    artifact = _artifact(
        "release-archive-merkle-recovery-checkpoint.json",
        raws["release-archive-merkle-recovery-checkpoint.json"],
    )
    state = docs[_DOC_ARCHIVE_MERKLE_RECOVERY_STATE]
    expected_state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-archive-merkle-recovery",
        "sequence": info["summary"]["sequence"],
        "rebridgeSequence": info["summary"]["rebridgeSequence"],
        "archiveMerkleRebridgeAuthorityHeadSha256": info["summary"]["authorityHead"],
        "merkleRebridgeContinuityHeadSha256": info["summary"]["merkleHead"],
        "activeAuthoritySha256": info["summary"]["activeAuthoritySha256"],
        "recoveryCheckpointHeadSha256": checkpoint["recoveryCheckpointHeadSha256"],
        "checkpointArtifact": artifact,
        "archiveMembership": checkpoint["archiveMembership"],
    }
    if state != expected_state:
        _fail("ARCHIVE_MERKLE_RECOVERY_STATE_INVALID")
    receipt = docs["release-archive-merkle-recovery-receipt.json"]
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"schemaVersion", "status", "checkpointArtifact", "archives"}
        or receipt.get("schemaVersion") != int(POLICY["receipt_schema_version"])
        or receipt.get("status") != "archive-merkle-recovery-preserved"
        or receipt.get("checkpointArtifact") != artifact
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_RECEIPT_SCHEMA_INVALID")
    rows = receipt.get("archives")
    if not isinstance(rows, list) or len(rows) < int(POLICY["min_successful_archives"]):
        _fail("ARCHIVE_MERKLE_RECOVERY_RECEIPT_QUORUM_INVALID")
    member_map = {r["archiveIdentity"]: r for r in checkpoint["archiveMembership"]}
    locators = set()
    archive_ops = set()
    verifier_ops = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "archiveResponse",
            "verifierResponse",
        }:
            _fail("ARCHIVE_MERKLE_RECOVERY_RECEIPT_ROW_INVALID")
        ar = row["archiveResponse"]
        identity = ar.get("archiveIdentity") if isinstance(ar, dict) else None
        if identity not in member_map:
            _fail("ARCHIVE_MERKLE_RECOVERY_RECEIPT_MEMBER_INVALID")
        member = member_map[identity]
        parsed_a = _validate_archive_response(
            ar,
            operation="preserve",
            archive_id=_archive_id(artifact["sha256"], member["archiveIdentity"]),
            identity=member["archiveIdentity"],
            operator=member["archiveOperator"],
            artifact=artifact,
            now=current,
            historical=historical,
        )
        parsed_v = _validate_verifier_response(
            row["verifierResponse"],
            archive=parsed_a,
            identity=member["verifierIdentity"],
            operator=member["verifierOperator"],
            artifact=artifact,
            now=current,
            historical=historical,
        )
        if parsed_a["locator"] in locators:
            _fail("ARCHIVE_MERKLE_RECOVERY_LOCATOR_COLLISION")
        locators.add(parsed_a["locator"])
        archive_ops.add(parsed_a["archiveOperator"])
        verifier_ops.add(parsed_v["verifierOperator"])
    if (
        len(rows) < int(POLICY["min_successful_archives"])
        or len(archive_ops) < int(POLICY["min_archive_operators"])
        or len(verifier_ops) < int(POLICY["min_verifier_operators"])
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_RECEIPT_OPERATOR_QUORUM_INVALID")
    return {
        "ok": True,
        "sequence": info["summary"]["sequence"],
        "rebridge_sequence": info["summary"]["rebridgeSequence"],
        "authority_head_sha256": info["summary"]["authorityHead"],
        "merkle_head_sha256": info["summary"]["merkleHead"],
        "active_authority_sha256": info["summary"]["activeAuthoritySha256"],
        "checkpoint_sha256": artifact["sha256"],
        "checkpoint_head_sha256": checkpoint["recoveryCheckpointHeadSha256"],
    }


def preserve_rebridge_history(  # ruff: ignore[undocumented-public-function]
    *,
    run167_dir: Path,
    output_dir: Path,
    verify_kwargs: dict[str, Any],
    targets: list[tuple[str, str, ArchiveAdapter, str, str, VerifierAdapter]],
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run167_dir = _regular_dir(run167_dir, "ARCHIVE_MERKLE_RECOVERY_RUN167_DIR_INVALID")
    target = _outside(
        output_dir, [run167_dir], "ARCHIVE_MERKLE_RECOVERY_OUTPUT_INVALID"
    )
    if target.exists() or target.is_symlink():
        _fail("ARCHIVE_MERKLE_RECOVERY_OUTPUT_ALREADY_EXISTS")
    normalized = []
    for ai, ao, aa, vi, vo, va in targets:
        normalized.append(
            (
                _identity(ai, "ARCHIVE_MERKLE_RECOVERY_ARCHIVE_IDENTITY_INVALID"),
                _identity(ao, "ARCHIVE_MERKLE_RECOVERY_ARCHIVE_OPERATOR_INVALID"),
                aa,
                _identity(vi, "ARCHIVE_MERKLE_RECOVERY_VERIFIER_IDENTITY_INVALID"),
                _identity(vo, "ARCHIVE_MERKLE_RECOVERY_VERIFIER_OPERATOR_INVALID"),
                va,
            )
        )
    if len(normalized) < int(POLICY["min_configured_archives"]):
        _fail("ARCHIVE_MERKLE_RECOVERY_TARGET_PLAN_TOO_SMALL")
    membership = sorted(
        [
            {
                "archiveIdentity": x[0],
                "archiveOperator": x[1],
                "verifierIdentity": x[3],
                "verifierOperator": x[4],
            }
            for x in normalized
        ],
        key=lambda x: (
            x["archiveOperator"],
            x["archiveIdentity"],
            x["verifierOperator"],
            x["verifierIdentity"],
        ),
    )
    # Reuse the checkpoint membership validator for identity/operator separation.
    probe = {
        "schemaVersion": int(POLICY["checkpoint_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "archive-merkle-rebridge-recovery-checkpoint",
        "run167Artifacts": {},
        "run167Documents": {},
        "summary": {},
        "archiveMembership": membership,
    }
    # Validate plan directly because a full checkpoint does not exist yet.
    if len({r["archiveIdentity"] for r in membership}) != len(membership) or len(
        {r["verifierIdentity"] for r in membership}
    ) != len(membership):
        _fail("ARCHIVE_MERKLE_RECOVERY_MEMBERSHIP_IDENTITY_REUSE")
    if len({r["archiveOperator"] for r in membership}) < int(
        POLICY["min_archive_operators"]
    ) or len({r["verifierOperator"] for r in membership}) < int(
        POLICY["min_verifier_operators"]
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_MEMBERSHIP_OPERATOR_QUORUM_INVALID")
    if {r["archiveIdentity"] for r in membership} & {
        r["verifierIdentity"] for r in membership
    } or {r["archiveOperator"] for r in membership} & {
        r["verifierOperator"] for r in membership
    }:
        _fail("ARCHIVE_MERKLE_RECOVERY_AUTHORITY_PLANE_OVERLAP")
    before = _authority_fingerprint(run167_dir, verify_kwargs)
    kwargs = dict(verify_kwargs)
    kwargs["output_dir"] = run167_dir
    kwargs["now"] = current
    kwargs["historical"] = False
    try:
        run167.verify_archive_merkle_rebridge(**kwargs)
    except Exception as exc:
        raise ArchiveMerkleRecoveryError(
            "ARCHIVE_MERKLE_RECOVERY_RUN167_VERIFY_FAILED:" + str(exc)
        ) from exc
    docs, raws = _run167_docs(run167_dir)
    summary = _verify_snapshot_documents(docs, raws)
    checkpoint = _make_checkpoint(docs, raws, summary, membership)
    craw = _canonical(checkpoint)
    artifact = _artifact("release-archive-merkle-recovery-checkpoint.json", craw)
    successes = []
    locators = set()
    archive_ops = set()
    verifier_ops = set()
    for ai, ao, adapter, vi, vo, verifier in sorted(
        normalized, key=lambda x: (x[1], x[0], x[4], x[3])
    ):
        archive_id = _archive_id(artifact["sha256"], ai)
        request = {
            "schemaVersion": int(POLICY["archive_protocol_version"]),
            "operation": "preserve",
            "archiveId": archive_id,
            "archiveIdentity": ai,
            "archiveOperator": ao,
            "artifact": artifact,
            "payloadBase64": base64.b64encode(craw).decode("ascii"),
            "overwrite": False,
            "recoveryCheckpointHeadSha256": checkpoint["recoveryCheckpointHeadSha256"],
            "authorityHeadSha256": summary["authorityHead"],
            "merkleHeadSha256": summary["merkleHead"],
        }
        parsed_a = _validate_archive_response(
            adapter(dict(request)),
            operation="preserve",
            archive_id=archive_id,
            identity=ai,
            operator=ao,
            artifact=artifact,
            now=current,
        )
        vreq = {
            "schemaVersion": int(POLICY["verifier_protocol_version"]),
            "operation": "verify",
            "archiveId": archive_id,
            "archiveIdentity": ai,
            "archiveOperator": ao,
            "verifierIdentity": vi,
            "verifierOperator": vo,
            "artifact": artifact,
            "locator": parsed_a["locator"],
            "immutableVersionId": parsed_a["immutableVersionId"],
            "expectedSha256": artifact["sha256"],
            "expectedSize": artifact["size"],
        }
        parsed_v = _validate_verifier_response(
            verifier(dict(vreq)),
            archive=parsed_a,
            identity=vi,
            operator=vo,
            artifact=artifact,
            now=current,
        )
        if parsed_a["locator"] in locators:
            _fail("ARCHIVE_MERKLE_RECOVERY_LOCATOR_COLLISION")
        locators.add(parsed_a["locator"])
        archive_ops.add(ao)
        verifier_ops.add(vo)
        successes.append({"archiveResponse": parsed_a, "verifierResponse": parsed_v})
    if (
        len(successes) < int(POLICY["min_successful_archives"])
        or len(archive_ops) < int(POLICY["min_archive_operators"])
        or len(verifier_ops) < int(POLICY["min_verifier_operators"])
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_PRESERVATION_QUORUM_NOT_MET")
    if before != _authority_fingerprint(run167_dir, verify_kwargs):
        _fail("ARCHIVE_MERKLE_RECOVERY_INPUT_DRIFT")
    state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-archive-merkle-recovery",
        "sequence": summary["sequence"],
        "rebridgeSequence": summary["rebridgeSequence"],
        "archiveMerkleRebridgeAuthorityHeadSha256": summary["authorityHead"],
        "merkleRebridgeContinuityHeadSha256": summary["merkleHead"],
        "activeAuthoritySha256": summary["activeAuthoritySha256"],
        "recoveryCheckpointHeadSha256": checkpoint["recoveryCheckpointHeadSha256"],
        "checkpointArtifact": artifact,
        "archiveMembership": membership,
    }
    receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": "archive-merkle-recovery-preserved",
        "checkpointArtifact": artifact,
        "archives": successes,
    }
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run168-preserve-", dir=parent))
    try:
        _write(stage / "release-archive-merkle-recovery-checkpoint.json", checkpoint)
        _write(stage / _DOC_ARCHIVE_MERKLE_RECOVERY_STATE, state)
        _write(stage / "release-archive-merkle-recovery-receipt.json", receipt)
        verify_recovery_archive(output_dir=stage, now=current)
        os.replace(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "ok": True,
        "phase": "archive-merkle-rebridge-preserved",
        "sequence": summary["sequence"],
        "checkpoint_sha256": artifact["sha256"],
        "checkpoint_head_sha256": checkpoint["recoveryCheckpointHeadSha256"],
        "authority_head_sha256": summary["authorityHead"],
        "merkle_head_sha256": summary["merkleHead"],
        "archives": len(successes),
    }


def _validate_recovery_response(
    value: Any, *, identity: str, operator: str, now: datetime
) -> dict[str, Any]:
    common = {
        "schemaVersion",
        "status",
        "sourceIdentity",
        "sourceOperator",
        "observedAt",
        "readOnly",
        "writerCredentialsReused",
    }
    if (
        not isinstance(value, dict)
        or value.get("schemaVersion") != int(POLICY["recovery_protocol_version"])
        or value.get("sourceIdentity") != identity
        or value.get("sourceOperator") != operator
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_SCHEMA_INVALID")
    if (
        value.get("readOnly") is not True
        or value.get("writerCredentialsReused") is not False
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_AUTHORITY_INVALID")
    observed_at = _fresh(
        value.get("observedAt"), "ARCHIVE_MERKLE_RECOVERY_SOURCE_OBSERVED_AT", now=now
    )
    if value.get("status") == "unavailable":
        if set(value) != common | {"reason"} or not bool(
            POLICY["allow_unavailable_recovery_source"]
        ):
            _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_UNAVAILABLE_INVALID")
        reason = value.get("reason")
        _len = len(reason) > 255  # ruff: ignore[magic-value-comparison]
        if not isinstance(reason, str) or not reason or _len:
            _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_REASON_INVALID")
        return {"status": "unavailable", "observedAt": observed_at, "reason": reason}
    if value.get("status") != "observed" or set(value) != common | {
        "locator",
        "artifact",
        "payloadBase64",
    }:
        _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_STATUS_INVALID")
    artifact = _artifact_doc(
        value.get("artifact"), "ARCHIVE_MERKLE_RECOVERY_SOURCE_ARTIFACT"
    )
    if artifact["name"] != "release-archive-merkle-recovery-checkpoint.json":
        _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_ARTIFACT_NAME_INVALID")
    try:
        raw = base64.b64decode(value.get("payloadBase64"), validate=True)
    except Exception as exc:
        raise ArchiveMerkleRecoveryError(
            "ARCHIVE_MERKLE_RECOVERY_SOURCE_PAYLOAD_INVALID"
        ) from exc
    if len(raw) != artifact["size"] or _sha_bytes(raw) != artifact["sha256"]:
        _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_PAYLOAD_HASH_INVALID")
    doc = _loads(raw, "ARCHIVE_MERKLE_RECOVERY_SOURCE_PAYLOAD")
    return {
        "status": "observed",
        "observedAt": observed_at,
        "locator": _safe_locator(
            value.get("locator"), "ARCHIVE_MERKLE_RECOVERY_SOURCE_LOCATOR_INVALID"
        ),
        "artifact": artifact,
        "raw": raw,
        "doc": doc,
    }


def recover_rebridge_history(  # ruff: ignore[undocumented-public-function]
    *,
    sources: list[tuple[str, str, RecoveryAdapter]],
    output_dir: Path,
    expected_checkpoint_sha256: str,
    expected_authority_head_sha256: str,
    expected_merkle_head_sha256: str,
    expected_active_authority_sha256: str,
    expected_sequence: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expected_checkpoint_sha256 = _hex(
        expected_checkpoint_sha256,
        "ARCHIVE_MERKLE_RECOVERY_EXPECTED_CHECKPOINT_INVALID",
    )
    expected_authority_head_sha256 = _hex(
        expected_authority_head_sha256,
        "ARCHIVE_MERKLE_RECOVERY_EXPECTED_AUTHORITY_HEAD_INVALID",
    )
    expected_merkle_head_sha256 = _hex(
        expected_merkle_head_sha256,
        "ARCHIVE_MERKLE_RECOVERY_EXPECTED_MERKLE_HEAD_INVALID",
    )
    expected_active_authority_sha256 = _hex(
        expected_active_authority_sha256,
        "ARCHIVE_MERKLE_RECOVERY_EXPECTED_ACTIVE_AUTHORITY_INVALID",
    )
    expected_sequence = _positive_int(
        expected_sequence, "ARCHIVE_MERKLE_RECOVERY_EXPECTED_SEQUENCE_INVALID"
    )
    normalized = [
        (
            _identity(i, "ARCHIVE_MERKLE_RECOVERY_SOURCE_IDENTITY_INVALID"),
            _identity(o, "ARCHIVE_MERKLE_RECOVERY_SOURCE_OPERATOR_INVALID"),
            a,
        )
        for i, o, a in sources
    ]
    if (
        len(normalized) < int(POLICY["min_configured_recovery_sources"])
        or len({i for i, _, _ in normalized}) != len(normalized)
        or len({o for _, o, _ in normalized}) < int(POLICY["min_recovery_operators"])
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_PLAN_INVALID")
    request = {
        "schemaVersion": int(POLICY["recovery_protocol_version"]),
        "operation": "recover",
        "expectedCheckpointSha256": expected_checkpoint_sha256,
        "expectedAuthorityHeadSha256": expected_authority_head_sha256,
        "expectedMerkleHeadSha256": expected_merkle_head_sha256,
        "expectedActiveAuthoritySha256": expected_active_authority_sha256,
        "expectedSequence": expected_sequence,
    }
    observed = []
    unavailable = []
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
    if len(observed) < int(POLICY["min_observed_recovery_sources"]) or len(
        {x["operator"] for x in observed}
    ) < int(POLICY["min_recovery_operators"]):
        _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_QUORUM_NOT_MET")
    locators = [x["locator"] for x in observed]
    if len(set(locators)) != len(locators):
        _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_LOCATOR_COLLISION")
    hashes = {_sha_bytes(x["raw"]) for x in observed}
    if len(hashes) != 1:
        _fail("ARCHIVE_MERKLE_RECOVERY_EQUIVOCATION")
    raw = observed[0]["raw"]
    if _sha_bytes(raw) != expected_checkpoint_sha256:
        _fail("ARCHIVE_MERKLE_RECOVERY_ROLLBACK_CHECKPOINT_PIN_MISMATCH")
    info = _verify_checkpoint(observed[0]["doc"])
    summary = info["summary"]
    if (
        summary["authorityHead"] != expected_authority_head_sha256
        or summary["merkleHead"] != expected_merkle_head_sha256
        or summary["activeAuthoritySha256"] != expected_active_authority_sha256
        or summary["sequence"] != expected_sequence
    ):
        _fail("ARCHIVE_MERKLE_RECOVERY_ROLLBACK_HEAD_PIN_MISMATCH")
    target = _outside(output_dir, [], "ARCHIVE_MERKLE_RECOVERY_RECOVERY_OUTPUT_INVALID")
    if target.exists() or target.is_symlink():
        _fail("ARCHIVE_MERKLE_RECOVERY_RECOVERY_OUTPUT_ALREADY_EXISTS")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run168-recover-", dir=parent))
    try:
        (stage / "recovered-archive-merkle-recovery-checkpoint.json").write_bytes(raw)
        recovered = stage / "recovered-run167"
        recovered.mkdir()
        for name in sorted(_RUN167_NAMES):
            _write(recovered / name, info["documents"][name])
        # Re-check the exact reconstructed byte set using the compact offline snapshot verifier.
        rdocs, rraws = _run167_docs(recovered)
        rsummary = _verify_snapshot_documents(rdocs, rraws)
        if rsummary != summary:
            _fail("ARCHIVE_MERKLE_RECOVERY_REHYDRATED_MISMATCH")
        active = {
            "schemaVersion": int(POLICY["active_recovery_schema_version"]),
            "status": "recovered-active-archive-merkle-authority",
            "sequence": summary["sequence"],
            "rebridgeSequence": summary["rebridgeSequence"],
            "authority": summary["activeAuthority"],
            "activeAuthoritySha256": summary["activeAuthoritySha256"],
            "revokedKeyFingerprints": summary["revokedKeyFingerprints"],
            "lastCheckpoints": summary["lastCheckpoints"],
            "archiveMerkleRebridgeAuthorityHeadSha256": summary["authorityHead"],
            "merkleRebridgeContinuityHeadSha256": summary["merkleHead"],
        }
        _write(stage / "recovered-active-archive-merkle-authority.json", active)
        receipt = {
            "schemaVersion": int(POLICY["recovery_receipt_schema_version"]),
            "status": "archive-merkle-rebridge-recovered",
            "checkpointArtifact": _artifact(
                "release-archive-merkle-recovery-checkpoint.json", raw
            ),
            "sequence": summary["sequence"],
            "archiveMerkleRebridgeAuthorityHeadSha256": summary["authorityHead"],
            "merkleRebridgeContinuityHeadSha256": summary["merkleHead"],
            "activeAuthoritySha256": summary["activeAuthoritySha256"],
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
        _write(stage / "release-archive-merkle-recovery-recovery-receipt.json", receipt)
        os.replace(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "ok": True,
        "phase": "archive-merkle-rebridge-recovered",
        "sequence": summary["sequence"],
        "checkpoint_sha256": expected_checkpoint_sha256,
        "authority_head_sha256": summary["authorityHead"],
        "merkle_head_sha256": summary["merkleHead"],
        "active_authority_sha256": summary["activeAuthoritySha256"],
        "observed": len(observed),
        "unavailable": len(unavailable),
    }


def _command_adapter(
    command: list[str], *, prefix: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if not command:
        _fail(prefix + "_COMMAND_EMPTY")
    limit = int(POLICY["max_adapter_output_bytes"])
    timeout = int(POLICY["adapter_timeout_seconds"])

    def call(request: dict[str, Any]) -> dict[str, Any]:
        try:
            proc = (
                # lint
                subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
                )
            )
        except OSError as exc:
            raise ArchiveMerkleRecoveryError(prefix + "_START_FAILED") from exc
        out = bytearray()
        err = bytearray()
        overflow = threading.Event()
        total = [0]
        lock = threading.Lock()

        def drain(stream, buf):
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                with lock:
                    total[0] += len(chunk)
                    if total[0] > limit:
                        overflow.set()
                        proc.kill()
                        break
                buf.extend(chunk)

        ts = [
            threading.Thread(target=drain, args=(proc.stdout, out), daemon=True),
            threading.Thread(target=drain, args=(proc.stderr, err), daemon=True),
        ]
        for t in ts:
            t.start()
        try:
            assert proc.stdin is not None  # ruff: ignore[assert]
            proc.stdin.write(_canonical(request))
            proc.stdin.close()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            _fail(prefix + "_TIMEOUT")
        finally:
            for t in ts:
                t.join(timeout=1)
        if overflow.is_set():
            _fail(prefix + "_OUTPUT_TOO_LARGE")
        if proc.returncode != 0:
            _fail(prefix + "_FAILED")
        return _loads(bytes(out), prefix + "_OUTPUT")

    return call


def command_archive(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="ARCHIVE_MERKLE_RECOVERY_ARCHIVE_ADAPTER")


def command_verifier(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="ARCHIVE_MERKLE_RECOVERY_VERIFIER_ADAPTER")


def command_recovery(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(command, prefix="ARCHIVE_MERKLE_RECOVERY_SOURCE_ADAPTER")


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-checkpoint")
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--historical", action="store_true")
    recover = sub.add_parser("recover")
    recover.add_argument(
        "--source", action="append", required=True, help="identity,operator=command..."
    )
    recover.add_argument("--output-dir", type=Path, required=True)
    recover.add_argument("--expected-checkpoint-sha256", required=True)
    recover.add_argument("--expected-authority-head-sha256", required=True)
    recover.add_argument("--expected-merkle-head-sha256", required=True)
    recover.add_argument("--expected-active-authority-sha256", required=True)
    recover.add_argument("--expected-sequence", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify-checkpoint":
        result = verify_recovery_archive(
            output_dir=args.output_dir, historical=args.historical
        )
    else:
        sources = []
        for value in args.source:
            if "=" not in value:
                _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_ARG_INVALID")
            lhs, cmd = value.split("=", 1)
            parts = lhs.split(",")
            if len(parts) != 2:  # ruff: ignore[magic-value-comparison]
                _fail("ARCHIVE_MERKLE_RECOVERY_SOURCE_ARG_INVALID")
            sources.append((parts[0], parts[1], command_recovery(cmd.strip().split())))
        result = recover_rebridge_history(
            sources=sources,
            output_dir=args.output_dir,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_authority_head_sha256=args.expected_authority_head_sha256,
            expected_merkle_head_sha256=args.expected_merkle_head_sha256,
            expected_active_authority_sha256=args.expected_active_authority_sha256,
            expected_sequence=args.expected_sequence,
        )
    logger.info(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
