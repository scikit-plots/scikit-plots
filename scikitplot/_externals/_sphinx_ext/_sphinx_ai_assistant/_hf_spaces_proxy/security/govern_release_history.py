"""
Threshold-govern release-history trust roots and recover them from immutable archives.
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
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

import tomllib

logger = logging.getLogger(__name__)

try:  # package import
    from . import preserve_release_history as history
except (ImportError, ValueError) as exc:  # direct Space-style/script/importlib loading
    import importlib.util

    _history_path = Path(__file__).resolve().parent / "preserve_release_history.py"
    _history_name = "_release_history_for_governance"
    if _history_name in sys.modules:
        history = sys.modules[_history_name]
    else:
        _spec = importlib.util.spec_from_file_location(_history_name, _history_path)
        if _spec is None or _spec.loader is None:
            raise ImportError("cannot load preserve_release_history") from exc
        history = importlib.util.module_from_spec(_spec)
        sys.modules[_history_name] = history
        _spec.loader.exec_module(history)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_governance_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}$")
_CHUNK = 1024 * 1024


class GovernanceError(RuntimeError):
    """Governance or disaster-recovery evidence violated a fail-closed invariant."""


def _fail(code: str) -> None:
    raise GovernanceError(code)


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
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                _fail(code + "_DUPLICATE_KEY")
            out[key] = value
        return out

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except GovernanceError:
        raise
    except Exception as exc:
        raise GovernanceError(code + "_JSON_INVALID") from exc
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
            ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
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
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in value  # lint
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


def _size(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(code)
    return value


def _timestamp(value: Any, code: str, *, now: datetime | None = None) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code + "_INVALID")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError:
        _fail(code + "_INVALID")
    if now is not None and parsed > now + timedelta(
        minutes=int(POLICY["max_clock_skew_minutes"])
    ):
        _fail(code + "_FROM_FUTURE")
    return value


def _artifact_doc(doc: dict[str, Any], name: str) -> dict[str, Any]:
    raw = _canonical_bytes(doc)
    return {"name": name, "sha256": _sha_bytes(raw), "size": len(raw)}


def _member(value: Any, code: str, *, keyed: bool) -> dict[str, str]:
    expected = {"identity", "operator", "keyId"} if keyed else {"identity", "operator"}
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    out = {
        "identity": _identity(value.get("identity"), code + "_IDENTITY_INVALID"),
        "operator": _identity(value.get("operator"), code + "_OPERATOR_INVALID"),
    }
    if keyed:
        out["keyId"] = _identity(value.get("keyId"), code + "_KEY_ID_INVALID")
    return out


def _membership(
    value: Any,
    code: str,
    *,
    keyed: bool,
    min_members: int,
    max_members: int,
    min_threshold: int,
    min_operators: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"threshold", "members"}:
        _fail(code + "_SCHEMA_INVALID")
    threshold = _size(value.get("threshold"), code + "_THRESHOLD_INVALID")
    members_raw = value.get("members")
    if not isinstance(members_raw, list) or not (
        min_members <= len(members_raw) <= max_members
    ):
        _fail(code + "_MEMBER_COUNT_INVALID")
    members = [_member(x, code + "_MEMBER", keyed=keyed) for x in members_raw]
    identities = [x["identity"] for x in members]
    operators = {x["operator"] for x in members}
    if len(set(identities)) != len(identities):
        _fail(code + "_IDENTITY_DUPLICATE")
    if keyed:
        keys = [x["keyId"] for x in members]
        if len(set(keys)) != len(keys):
            _fail(code + "_KEY_DUPLICATE")
    if threshold < min_threshold or threshold > len(members):
        _fail(code + "_THRESHOLD_INVALID")
    if len(operators) < min_operators:
        _fail(code + "_OPERATOR_COUNT_INVALID")
    members = sorted(
        members, key=lambda x: (x["operator"], x["identity"], x.get("keyId", ""))
    )
    return {"threshold": threshold, "members": members}


def _policy(value: Any, code: str) -> dict[str, Any]:
    expected = {
        "policyVersion",
        "policyAuthority",
        "emergencyAuthority",
        "historyReplicas",
        "historyArchives",
    }
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    version = _size(value.get("policyVersion"), code + "_VERSION_INVALID")
    if version < 1:
        _fail(code + "_VERSION_INVALID")
    result = {
        "policyVersion": version,
        "policyAuthority": _membership(
            value.get("policyAuthority"),
            code + "_POLICY_AUTHORITY",
            keyed=True,
            min_members=int(POLICY["min_policy_authorities"]),
            max_members=int(POLICY["max_policy_authorities"]),
            min_threshold=int(POLICY["min_policy_threshold"]),
            min_operators=int(POLICY["min_policy_operators"]),
        ),
        "emergencyAuthority": _membership(
            value.get("emergencyAuthority"),
            code + "_EMERGENCY_AUTHORITY",
            keyed=True,
            min_members=int(POLICY["min_emergency_authorities"]),
            max_members=int(POLICY["max_emergency_authorities"]),
            min_threshold=int(POLICY["min_emergency_threshold"]),
            min_operators=int(POLICY["min_emergency_operators"]),
        ),
        "historyReplicas": _membership(
            value.get("historyReplicas"),
            code + "_HISTORY_REPLICAS",
            keyed=False,
            min_members=int(POLICY["min_history_replicas"]),
            max_members=int(POLICY["max_history_replicas"]),
            min_threshold=int(POLICY["min_history_replica_threshold"]),
            min_operators=int(POLICY["min_history_replica_operators"]),
        ),
        "historyArchives": _membership(
            value.get("historyArchives"),
            code + "_HISTORY_ARCHIVES",
            keyed=False,
            min_members=int(POLICY["min_history_archives"]),
            max_members=int(POLICY["max_history_archives"]),
            min_threshold=int(POLICY["min_history_archive_threshold"]),
            min_operators=int(POLICY["min_history_archive_operators"]),
        ),
    }
    policy_ids = {m["identity"] for m in result["policyAuthority"]["members"]}
    emergency_ids = {m["identity"] for m in result["emergencyAuthority"]["members"]}
    policy_keys = {m["keyId"] for m in result["policyAuthority"]["members"]}
    emergency_keys = {m["keyId"] for m in result["emergencyAuthority"]["members"]}
    if policy_ids & emergency_ids or policy_keys & emergency_keys:
        _fail(code + "_EMERGENCY_AUTHORITY_NOT_DISTINCT")
    if bool(POLICY.get("require_emergency_operator_disjoint", False)):
        policy_ops = {m["operator"] for m in result["policyAuthority"]["members"]}
        emergency_ops = {m["operator"] for m in result["emergencyAuthority"]["members"]}
        if policy_ops & emergency_ops:
            _fail(code + "_EMERGENCY_OPERATOR_NOT_DISTINCT")
    return result


def _governance_head(
    governance_id: str, genesis: dict[str, Any], entries: list[dict[str, Any]]
) -> str:
    head = hashlib.sha256(
        _canonical_bytes({"governanceId": governance_id, "genesis": genesis})
    ).digest()
    for entry in entries:
        head = hashlib.sha256(head + _canonical_bytes(entry)).digest()
    return head.hex()


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


def _validate_run153_dir(  # ruff: ignore[too-many-branches]
    root: Path,
) -> dict[str, Any]:
    root = _regular_dir(root, "RUN153_DIRECTORY_INVALID")
    allowed = {
        "release-history-bundle.json",
        _DOC_HISTORY_STATE,
        "release-history-preservation-receipt.json",
        "previous-release-history-bundle.json",
        "previous-trusted-history-state.json",
        "release-transparency-witness-record.json",
        "release-transparency-witness-receipt.json",
        "previous-transparency-checkpoint.json",
        "accepted-transparency-checkpoint.json",
        "replica-results",
        "archive-results",
    }
    if {p.name for p in root.iterdir()} != allowed:
        _fail("RUN153_DIRECTORY_ALLOWLIST_MISMATCH")
    for dirname in ("replica-results", "archive-results"):
        d = root / dirname
        if d.is_symlink() or not d.is_dir():
            _fail("RUN153_EVIDENCE_DIRECTORY_INVALID")
        for item in d.iterdir():
            if item.is_symlink() or not item.is_file():
                _fail("RUN153_EVIDENCE_ENTRY_INVALID")
            _read(item, "RUN153_EVIDENCE_ENTRY")
    try:
        info = history._validate_previous_history(
            root / _DOC_HISTORY_STATE, root / "release-history-bundle.json"
        )
    except history.HistoryError as exc:
        raise GovernanceError("RUN153_HISTORY_INVALID:" + str(exc)) from exc
    receipt, _ = _read(
        root / "release-history-preservation-receipt.json", "RUN153_RECEIPT"
    )
    expected_receipt = {
        "schemaVersion",
        "status",
        "historyId",
        "sequence",
        "collectorIdentity",
        "previous",
        "current",
        "replicaQuorum",
        "historyBundle",
        "trustedHistoryState",
        "archives",
    }
    if (
        set(receipt) != expected_receipt
        or receipt.get("status") != "preserved"
        or receipt.get("historyId") != info["history_id"]
        or receipt.get("sequence") != info["sequence"]
    ):
        _fail("RUN153_RECEIPT_SCHEMA_INVALID")
    bundle_item = receipt.get("historyBundle")
    state_item = receipt.get("trustedHistoryState")
    expected_bundle = {
        "name": "release-history-bundle.json",
        "sha256": info["bundle_sha256"],
        "size": (root / "release-history-bundle.json").stat().st_size,
    }
    expected_state = {
        "name": _DOC_HISTORY_STATE,
        "sha256": info["state_sha256"],
        "size": (root / _DOC_HISTORY_STATE).stat().st_size,
    }
    if bundle_item != expected_bundle or state_item != expected_state:
        _fail("RUN153_RECEIPT_ARTIFACT_REBIND_FAILED")
    replica_evidence = (
        receipt.get("replicaQuorum", {}).get("evidence")
        if isinstance(receipt.get("replicaQuorum"), dict)
        else None
    )
    if not isinstance(replica_evidence, list):
        _fail("RUN153_REPLICA_EVIDENCE_SCHEMA_INVALID")
    replica_hashes = {_sha(p) for p in (root / "replica-results").iterdir()}
    for item in replica_evidence:
        if (
            not isinstance(item, dict)
            or _hex(item.get("sha256"), "RUN153_REPLICA_EVIDENCE_HASH_INVALID")
            not in replica_hashes
        ):
            _fail("RUN153_REPLICA_EVIDENCE_REBIND_FAILED")
    archives = receipt.get("archives")
    if not isinstance(archives, list):
        _fail("RUN153_ARCHIVE_EVIDENCE_SCHEMA_INVALID")
    archive_hashes = {_sha(p) for p in (root / "archive-results").iterdir()}
    archive_members: list[dict[str, str]] = []
    for item in archives:
        if not isinstance(item, dict):
            _fail("RUN153_ARCHIVE_EVIDENCE_SCHEMA_INVALID")
        for field in ("bindEvidenceSha256", "verifyEvidenceSha256"):
            if (
                _hex(item.get(field), "RUN153_ARCHIVE_EVIDENCE_HASH_INVALID")
                not in archive_hashes
            ):
                _fail("RUN153_ARCHIVE_EVIDENCE_REBIND_FAILED")
        archive_members.append(
            {
                "identity": _identity(
                    item.get("identity"), "RUN153_ARCHIVE_IDENTITY_INVALID"
                ),
                "operator": _identity(
                    item.get("operator"), "RUN153_ARCHIVE_OPERATOR_INVALID"
                ),
            }
        )
    last_entry = info["entries"][-1] if info["entries"] else None
    replicas = last_entry["replicaQuorum"] if last_entry is not None else None
    return {
        **info,
        "root": root,
        "receipt": receipt,
        "archive_members": sorted(
            archive_members, key=lambda x: (x["operator"], x["identity"])
        ),
        "replica_membership": replicas,
    }


def _validate_genesis(
    doc: dict[str, Any], *, history_info: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "governanceId",
        "historyId",
        "bootstrapEvidenceSha256",
        "policy",
    }
    if set(doc) != expected or doc.get("schemaVersion") != int(
        POLICY["genesis_schema_version"]
    ):
        _fail("GOVERNANCE_GENESIS_SCHEMA_INVALID")
    governance_id = _safe_text(
        doc.get("governanceId"), "GOVERNANCE_ID_INVALID", limit=255
    )
    if doc.get("historyId") != history_info["history_id"]:
        _fail("GOVERNANCE_GENESIS_HISTORY_ID_MISMATCH")
    _hex(doc.get("bootstrapEvidenceSha256"), "GOVERNANCE_BOOTSTRAP_EVIDENCE_INVALID")
    policy = _policy(doc.get("policy"), "GOVERNANCE_GENESIS_POLICY")
    return {"governance_id": governance_id, "policy": policy}


def _entry_policy(entry: dict[str, Any]) -> dict[str, Any]:
    return _policy(entry.get("nextPolicy"), "GOVERNANCE_ENTRY_NEXT_POLICY")


def _validate_approval(
    doc: dict[str, Any],
    *,
    governance_id: str,
    transition_id: str,
    proposal_sha: str,
    role: str,
    allowed_members: dict[str, dict[str, str]],
    revoked: set[str],
    now: datetime | None,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "governanceId",
        "transitionId",
        "proposalSha256",
        "approver",
        "role",
        "decision",
        "signatureVerified",
        "signatureEvidenceSha256",
        "signedAt",
    }
    if (
        set(doc) != expected
        or doc.get("schemaVersion") != int(POLICY["approval_schema_version"])
        or doc.get("governanceId") != governance_id
        or doc.get("transitionId") != transition_id
        or doc.get("proposalSha256") != proposal_sha
    ):
        _fail("GOVERNANCE_APPROVAL_SCHEMA_INVALID")
    if (
        doc.get("role") != role
        or doc.get("decision") != "approve"
        or doc.get("signatureVerified") is not True
    ):
        _fail("GOVERNANCE_APPROVAL_DECISION_INVALID")
    approver = _member(doc.get("approver"), "GOVERNANCE_APPROVER", keyed=True)
    expected_member = allowed_members.get(approver["keyId"])
    if expected_member != approver:
        _fail("GOVERNANCE_APPROVER_NOT_AUTHORIZED")
    if approver["keyId"] in revoked:
        _fail("GOVERNANCE_REVOKED_AUTHORITY_KEY_REJECTED")
    _hex(
        doc.get("signatureEvidenceSha256"),
        "GOVERNANCE_APPROVAL_SIGNATURE_EVIDENCE_INVALID",
    )
    _timestamp(doc.get("signedAt"), "GOVERNANCE_APPROVAL_SIGNED_AT", now=now)
    return approver


def _history_binding(value: Any, code: str) -> dict[str, Any]:
    expected = {"sequence", "stateSha256", "bundleSha256", "chainHeadSha256"}
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    return {
        "sequence": _size(value.get("sequence"), code + "_SEQUENCE_INVALID"),
        "stateSha256": _hex(value.get("stateSha256"), code + "_STATE_HASH_INVALID"),
        "bundleSha256": _hex(value.get("bundleSha256"), code + "_BUNDLE_HASH_INVALID"),
        "chainHeadSha256": _hex(
            value.get("chainHeadSha256"), code + "_CHAIN_HEAD_INVALID"
        ),
    }


def _state_document(
    *,
    governance_id: str,
    history_id: str,
    epoch: int,
    policy: dict[str, Any],
    revoked: list[str],
    history_binding: dict[str, Any],
    bundle_sha256: str,
    chain_head_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "governanceId": governance_id,
        "historyId": history_id,
        "epoch": epoch,
        "policy": policy,
        "revokedAuthorityKeyIds": revoked,
        "history": history_binding,
        "bundleSha256": bundle_sha256,
        "chainHeadSha256": chain_head_sha256,
    }


def _validate_bundle(  # ruff: ignore[too-many-branches]
    bundle: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "predicateType",
        "status",
        "governanceId",
        "historyId",
        "genesis",
        "entries",
    }
    if (
        set(bundle) != expected
        or bundle.get("schemaVersion") != int(POLICY["bundle_schema_version"])
        or bundle.get("predicateType") != PREDICATE_TYPE
        or bundle.get("status") != "governed-history"
    ):
        _fail("GOVERNANCE_BUNDLE_SCHEMA_INVALID")
    governance_id = _safe_text(
        bundle.get("governanceId"), "GOVERNANCE_ID_INVALID", limit=255
    )
    history_id = _safe_text(
        bundle.get("historyId"), "GOVERNANCE_HISTORY_ID_INVALID", limit=255
    )
    genesis = bundle.get("genesis")
    if not isinstance(genesis, dict) or set(genesis) != {
        "genesisSha256",
        "bootstrapEvidenceSha256",
        "initialPolicy",
        "initialHistory",
    }:
        _fail("GOVERNANCE_BUNDLE_GENESIS_SCHEMA_INVALID")
    _hex(genesis.get("genesisSha256"), "GOVERNANCE_GENESIS_HASH_INVALID")
    _hex(genesis.get("bootstrapEvidenceSha256"), "GOVERNANCE_BOOTSTRAP_HASH_INVALID")
    current_policy = _policy(genesis.get("initialPolicy"), "GOVERNANCE_INITIAL_POLICY")
    current_history = _history_binding(
        genesis.get("initialHistory"), "GOVERNANCE_INITIAL_HISTORY"
    )
    revoked: set[str] = set()
    entries = bundle.get("entries")
    _len = len(entries) > 4096  # ruff: ignore[magic-value-comparison]
    if not isinstance(entries, list) or _len:
        _fail("GOVERNANCE_ENTRY_COUNT_INVALID")
    previous_epoch = 0
    transition_ids: set[str] = set()
    previous_history_sequence = current_history["sequence"]
    for index, entry in enumerate(entries, start=1):
        expected_entry = {
            "epoch",
            "transitionId",
            "reason",
            "history",
            "proposal",
            "proposalSha256",
            "approvalRole",
            "selectedApproverKeyIds",
            "approvals",
            "revokedAuthorityKeyIds",
            "nextPolicy",
        }
        if not isinstance(entry, dict) or set(entry) != expected_entry:
            _fail("GOVERNANCE_ENTRY_SCHEMA_INVALID")
        epoch = _size(entry.get("epoch"), "GOVERNANCE_ENTRY_EPOCH_INVALID")
        if epoch != previous_epoch + 1:
            _fail("GOVERNANCE_ENTRY_EPOCH_GAP")
        # Reconstruct the exact prior governance state from the bundle prefix.
        prefix_bundle = dict(bundle)
        prefix_bundle["entries"] = entries[: index - 1]
        prefix_sha = _sha_bytes(_canonical_bytes(prefix_bundle))
        prefix_head = _governance_head(governance_id, genesis, entries[: index - 1])
        prior_state = _state_document(
            governance_id=governance_id,
            history_id=history_id,
            epoch=previous_epoch,
            policy=current_policy,
            revoked=sorted(revoked),
            history_binding=current_history,
            bundle_sha256=prefix_sha,
            chain_head_sha256=prefix_head,
        )
        previous_epoch = epoch
        transition_id = _identity(
            entry.get("transitionId"), "GOVERNANCE_TRANSITION_ID_INVALID"
        )
        if transition_id in transition_ids:
            _fail("GOVERNANCE_TRANSITION_REPLAY")
        transition_ids.add(transition_id)
        reason = entry.get("reason")
        if reason not in set(POLICY["allowed_transition_reasons"]):
            _fail("GOVERNANCE_REASON_INVALID")
        history_binding = _history_binding(
            entry.get("history"), "GOVERNANCE_HISTORY_BINDING"
        )
        if history_binding["sequence"] < previous_history_sequence:
            _fail("GOVERNANCE_HISTORY_ROLLBACK")
        previous_history_sequence = history_binding["sequence"]
        proposal = entry.get("proposal")
        proposal_expected = {
            "schemaVersion",
            "governanceId",
            "transitionId",
            "fromEpoch",
            "toEpoch",
            "reason",
            "history",
            "previousGovernanceStateSha256",
            "selectedApproverKeyIds",
            "revokedAuthorityKeyIds",
            "nextPolicy",
        }
        if (
            not isinstance(proposal, dict)
            or set(proposal) != proposal_expected
            or proposal.get("schemaVersion") != int(POLICY["proposal_schema_version"])
        ):
            _fail("GOVERNANCE_ENTRY_PROPOSAL_SCHEMA_INVALID")
        proposal_sha = _hex(
            entry.get("proposalSha256"), "GOVERNANCE_PROPOSAL_HASH_INVALID"
        )
        if _sha_bytes(_canonical_bytes(proposal)) != proposal_sha:
            _fail("GOVERNANCE_ENTRY_PROPOSAL_HASH_MISMATCH")
        if (
            proposal.get("governanceId") != governance_id
            or proposal.get("transitionId") != transition_id
            or proposal.get("fromEpoch") != epoch - 1
            or proposal.get("toEpoch") != epoch
            or proposal.get("reason") != reason
        ):
            _fail("GOVERNANCE_ENTRY_PROPOSAL_REBIND_FAILED")
        if (
            _history_binding(proposal.get("history"), "GOVERNANCE_PROPOSAL_HISTORY")
            != history_binding
        ):
            _fail("GOVERNANCE_ENTRY_PROPOSAL_HISTORY_MISMATCH")
        if proposal.get("previousGovernanceStateSha256") != _sha_bytes(
            _canonical_bytes(prior_state)
        ):
            _fail("GOVERNANCE_ENTRY_PREVIOUS_STATE_HASH_MISMATCH")
        role = entry.get("approvalRole")
        expected_role = (
            "emergency" if reason == "authority-compromise-recovery" else "policy"
        )
        if role != expected_role:
            _fail("GOVERNANCE_APPROVAL_ROLE_INVALID")
        authority = (
            current_policy["emergencyAuthority"]
            if role == "emergency"
            else current_policy["policyAuthority"]
        )
        members = {m["keyId"]: m for m in authority["members"]}
        selected = entry.get("selectedApproverKeyIds")
        if (
            not isinstance(selected, list)
            or len(selected) < authority["threshold"]
            or len(set(selected)) != len(selected)
        ):
            _fail("GOVERNANCE_SELECTED_APPROVER_QUORUM_INVALID")
        selected = [
            _identity(x, "GOVERNANCE_SELECTED_APPROVER_KEY_INVALID") for x in selected
        ]
        proposal_selected = proposal.get("selectedApproverKeyIds")
        if not isinstance(proposal_selected, list) or sorted(
            proposal_selected
        ) != sorted(selected):
            _fail("GOVERNANCE_ENTRY_PROPOSAL_APPROVER_SET_MISMATCH")
        if any(x not in members or x in revoked for x in selected):
            _fail("GOVERNANCE_SELECTED_APPROVER_NOT_AUTHORIZED")
        approvals = entry.get("approvals")
        if not isinstance(approvals, list) or len(approvals) != len(selected):
            _fail("GOVERNANCE_APPROVAL_COUNT_INVALID")
        approval_keys: list[str] = []
        operators: set[str] = set()
        for approval in approvals:
            approver = _validate_approval(
                approval,
                governance_id=governance_id,
                transition_id=transition_id,
                proposal_sha=proposal_sha,
                role=role,
                allowed_members=members,
                revoked=revoked,
                now=None,
            )
            approval_keys.append(approver["keyId"])
            operators.add(approver["operator"])
        if sorted(approval_keys) != sorted(selected):
            _fail("GOVERNANCE_APPROVAL_SET_MISMATCH")
        min_ops = int(
            POLICY["min_emergency_operators"]
            if role == "emergency"
            else POLICY["min_policy_operators"]
        )
        if len(operators) < min_ops:
            _fail("GOVERNANCE_APPROVAL_OPERATOR_QUORUM_INVALID")
        revocations = entry.get("revokedAuthorityKeyIds")
        if not isinstance(revocations, list) or len(set(revocations)) != len(
            revocations
        ):
            _fail("GOVERNANCE_REVOCATION_SCHEMA_INVALID")
        revocations = [
            _identity(x, "GOVERNANCE_REVOCATION_KEY_INVALID") for x in revocations
        ]
        proposal_revocations = proposal.get("revokedAuthorityKeyIds")
        if not isinstance(proposal_revocations, list) or sorted(
            proposal_revocations
        ) != sorted(revocations):
            _fail("GOVERNANCE_ENTRY_PROPOSAL_REVOCATION_MISMATCH")
        current_authority_keys = {
            m["keyId"] for m in current_policy["policyAuthority"]["members"]
        } | {m["keyId"] for m in current_policy["emergencyAuthority"]["members"]}
        if any(x not in current_authority_keys or x in revoked for x in revocations):
            _fail("GOVERNANCE_REVOCATION_KEY_NOT_CURRENT")
        if reason == "authority-compromise-recovery":
            current_policy_keys = {
                m["keyId"] for m in current_policy["policyAuthority"]["members"]
            }
            if not set(revocations) & current_policy_keys:
                _fail("GOVERNANCE_COMPROMISE_RECOVERY_MUST_REVOKE_POLICY_KEY")
        next_policy = _entry_policy(entry)
        proposal_next_policy = _policy(
            proposal.get("nextPolicy"), "GOVERNANCE_PROPOSAL_NEXT_POLICY"
        )
        if proposal_next_policy != next_policy:
            _fail("GOVERNANCE_ENTRY_PROPOSAL_NEXT_POLICY_MISMATCH")
        if next_policy["policyVersion"] != current_policy["policyVersion"] + 1:
            _fail("GOVERNANCE_POLICY_VERSION_NOT_ADVANCING")
        next_keys = {m["keyId"] for m in next_policy["policyAuthority"]["members"]} | {
            m["keyId"] for m in next_policy["emergencyAuthority"]["members"]
        }
        if set(revocations) & next_keys:
            _fail("GOVERNANCE_REVOKED_KEY_IN_NEXT_POLICY")
        if revoked & next_keys:
            _fail("GOVERNANCE_PREVIOUSLY_REVOKED_KEY_REUSED")
        revoked.update(revocations)
        current_policy = next_policy
        current_history = history_binding
    return {
        "governance_id": governance_id,
        "history_id": history_id,
        "epoch": len(entries),
        "policy": current_policy,
        "history": current_history,
        "revoked_authority_key_ids": sorted(revoked),
        "chain_head_sha256": _governance_head(governance_id, genesis, entries),
        "entries": entries,
        "genesis": genesis,
    }


def verify_governance_bundle(  # ruff: ignore[undocumented-public-function]
    *, bundle_path: Path, state_path: Path | None = None
) -> dict[str, Any]:
    bundle, bundle_raw = _read(bundle_path, "GOVERNANCE_BUNDLE")
    info = _validate_bundle(bundle)
    if state_path is not None:
        state, state_raw = _read(state_path, "GOVERNANCE_STATE")
        expected = {
            "schemaVersion",
            "governanceId",
            "historyId",
            "epoch",
            "policy",
            "revokedAuthorityKeyIds",
            "history",
            "bundleSha256",
            "chainHeadSha256",
        }
        if set(state) != expected or state.get("schemaVersion") != int(
            POLICY["state_schema_version"]
        ):
            _fail("GOVERNANCE_STATE_SCHEMA_INVALID")
        if (
            state.get("governanceId") != info["governance_id"]
            or state.get("historyId") != info["history_id"]
            or state.get("epoch") != info["epoch"]
        ):
            _fail("GOVERNANCE_STATE_ID_MISMATCH")
        if (
            _policy(state.get("policy"), "GOVERNANCE_STATE_POLICY") != info["policy"]
            or state.get("revokedAuthorityKeyIds") != info["revoked_authority_key_ids"]
        ):
            _fail("GOVERNANCE_STATE_POLICY_MISMATCH")
        hist = _history_binding(state.get("history"), "GOVERNANCE_STATE_HISTORY")
        if hist != info["history"]:
            _fail("GOVERNANCE_STATE_HISTORY_MISMATCH")
        if (
            state.get("bundleSha256") != _sha_bytes(bundle_raw)
            or state.get("chainHeadSha256") != info["chain_head_sha256"]
        ):
            _fail("GOVERNANCE_STATE_HASH_MISMATCH")
        if _sha_bytes(state_raw) == state.get("bundleSha256"):
            _fail("GOVERNANCE_STATE_SELF_BIND_INVALID")
    return {
        "ok": True,
        "governance_id": info["governance_id"],
        "history_id": info["history_id"],
        "epoch": info["epoch"],
        "policy_version": info["policy"]["policyVersion"],
        "bundle_sha256": _sha_bytes(bundle_raw),
        "chain_head_sha256": info["chain_head_sha256"],
    }


def _state_for(
    *, info: dict[str, Any], history_info: dict[str, Any], bundle_sha: str
) -> dict[str, Any]:
    history_binding = {
        "sequence": history_info["sequence"],
        "stateSha256": history_info["state_sha256"],
        "bundleSha256": history_info["bundle_sha256"],
        "chainHeadSha256": history_info["chain_head_sha256"],
    }
    return _state_document(
        governance_id=info["governance_id"],
        history_id=info["history_id"],
        epoch=info["epoch"],
        policy=info["policy"],
        revoked=info["revoked_authority_key_ids"],
        history_binding=history_binding,
        bundle_sha256=bundle_sha,
        chain_head_sha256=info["chain_head_sha256"],
    )


def initialize_governance(  # ruff: ignore[undocumented-public-function]
    *,
    history_dir: Path,
    genesis_path: Path,
    expected_genesis_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    current = _validate_run153_dir(history_dir)
    target = _outside(output_dir, (current["root"],), "GOVERNANCE_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("GOVERNANCE_OUTPUT_ALREADY_EXISTS")
    expected_genesis_sha256 = _hex(
        expected_genesis_sha256, "GOVERNANCE_EXPECTED_GENESIS_HASH_INVALID"
    )
    genesis, genesis_raw = _read(genesis_path, "GOVERNANCE_GENESIS")
    if _sha_bytes(genesis_raw) != expected_genesis_sha256:
        _fail("GOVERNANCE_GENESIS_PIN_MISMATCH")
    g = _validate_genesis(genesis, history_info=current)
    init_inputs = [p for p in current["root"].rglob("*") if p.is_file()] + [
        genesis_path
    ]
    init_hashes = {str(p.resolve()): _sha(p) for p in init_inputs}
    policy = g["policy"]
    current_replicas = current["replica_membership"]
    if not isinstance(current_replicas, dict) or set(current_replicas) != {
        "threshold",
        "minimumOperators",
        "replicas",
    }:
        _fail("GOVERNANCE_RUN153_REPLICA_MEMBERSHIP_MISSING")
    normalized_current_replicas = _membership(
        {
            "threshold": current_replicas["threshold"],
            "members": current_replicas["replicas"],
        },
        "GOVERNANCE_RUN153_REPLICAS",
        keyed=False,
        min_members=int(POLICY["min_history_replicas"]),
        max_members=int(POLICY["max_history_replicas"]),
        min_threshold=int(POLICY["min_history_replica_threshold"]),
        min_operators=int(POLICY["min_history_replica_operators"]),
    )
    if policy["historyReplicas"] != normalized_current_replicas:
        _fail("GOVERNANCE_GENESIS_REPLICA_MEMBERSHIP_MISMATCH")
    expected_archives = {
        tuple(sorted(x.items())) for x in policy["historyArchives"]["members"]
    }
    current_archives = {tuple(sorted(x.items())) for x in current["archive_members"]}
    if (
        current_archives != expected_archives
        or len(current_archives) < policy["historyArchives"]["threshold"]
    ):
        _fail("GOVERNANCE_GENESIS_ARCHIVE_MEMBERSHIP_MISMATCH")
    initial_history = {
        "sequence": current["sequence"],
        "stateSha256": current["state_sha256"],
        "bundleSha256": current["bundle_sha256"],
        "chainHeadSha256": current["chain_head_sha256"],
    }
    bundle = {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "governed-history",
        "governanceId": g["governance_id"],
        "historyId": current["history_id"],
        "genesis": {
            "genesisSha256": expected_genesis_sha256,
            "bootstrapEvidenceSha256": genesis["bootstrapEvidenceSha256"],
            "initialPolicy": policy,
            "initialHistory": initial_history,
        },
        "entries": [],
    }
    info = _validate_bundle(bundle)
    bundle_raw = _canonical_bytes(bundle)
    bundle_sha = _sha_bytes(bundle_raw)
    state = _state_for(info=info, history_info=current, bundle_sha=bundle_sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-governance-init-", dir=target.parent
    ) as tmp:
        stage = Path(tmp) / "governance"
        stage.mkdir()
        _write(stage / "release-governance-bundle.json", bundle)
        _write(stage / _DOC_GOVERNANCE_STATE, state)
        verify_governance_bundle(
            bundle_path=stage / "release-governance-bundle.json",
            state_path=stage / _DOC_GOVERNANCE_STATE,
        )
        shutil.copy2(genesis_path, stage / "governance-genesis.json")
        for input_path in init_inputs:
            if _sha(input_path) != init_hashes[str(input_path.resolve())]:
                _fail("GOVERNANCE_INITIALIZATION_INPUT_CHANGED")
        shutil.copy2(
            current["root"] / _DOC_HISTORY_STATE,
            stage / _DOC_HISTORY_STATE,
        )
        shutil.copy2(
            current["root"] / "release-history-bundle.json",
            stage / "release-history-bundle.json",
        )
        os.replace(stage, target)
    return {
        "ok": True,
        "phase": "governance-initialized",
        "governance_id": g["governance_id"],
        "epoch": 0,
        "policy_version": policy["policyVersion"],
        "bundle_sha256": bundle_sha,
        "state_sha256": _sha(target / _DOC_GOVERNANCE_STATE),
    }


def _validate_previous_governance(
    state_path: Path, bundle_path: Path
) -> dict[str, Any]:
    result = verify_governance_bundle(bundle_path=bundle_path, state_path=state_path)
    state, state_raw = _read(state_path, "PREVIOUS_GOVERNANCE_STATE")
    bundle, bundle_raw = _read(bundle_path, "PREVIOUS_GOVERNANCE_BUNDLE")
    info = _validate_bundle(bundle)
    return {
        **result,
        **info,
        "state": state,
        "state_sha256": _sha_bytes(state_raw),
        "bundle": bundle,
        "bundle_sha256": _sha_bytes(bundle_raw),
        "history": state["history"],
    }


def _validate_proposal(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "governanceId",
        "transitionId",
        "fromEpoch",
        "toEpoch",
        "reason",
        "history",
        "previousGovernanceStateSha256",
        "selectedApproverKeyIds",
        "revokedAuthorityKeyIds",
        "nextPolicy",
    }
    if (
        set(doc) != expected
        or doc.get("schemaVersion") != int(POLICY["proposal_schema_version"])
        or doc.get("governanceId") != previous["governance_id"]
    ):
        _fail("GOVERNANCE_PROPOSAL_SCHEMA_INVALID")
    transition_id = _identity(
        doc.get("transitionId"), "GOVERNANCE_TRANSITION_ID_INVALID"
    )
    if (
        _size(doc.get("fromEpoch"), "GOVERNANCE_FROM_EPOCH_INVALID")
        != previous["epoch"]
        or _size(doc.get("toEpoch"), "GOVERNANCE_TO_EPOCH_INVALID")
        != previous["epoch"] + 1
    ):
        _fail("GOVERNANCE_PROPOSAL_EPOCH_MISMATCH")
    reason = doc.get("reason")
    if reason not in set(POLICY["allowed_transition_reasons"]):
        _fail("GOVERNANCE_REASON_INVALID")
    if doc.get("previousGovernanceStateSha256") != previous["state_sha256"]:
        _fail("GOVERNANCE_PROPOSAL_PREVIOUS_STATE_MISMATCH")
    history_binding = doc.get("history")
    expected_history = {
        "sequence": current["sequence"],
        "stateSha256": current["state_sha256"],
        "bundleSha256": current["bundle_sha256"],
        "chainHeadSha256": current["chain_head_sha256"],
    }
    if history_binding != expected_history:
        _fail("GOVERNANCE_PROPOSAL_HISTORY_MISMATCH")
    selected = doc.get("selectedApproverKeyIds")
    if not isinstance(selected, list) or len(set(selected)) != len(selected):
        _fail("GOVERNANCE_SELECTED_APPROVER_SET_INVALID")
    selected = [
        _identity(x, "GOVERNANCE_SELECTED_APPROVER_KEY_INVALID") for x in selected
    ]
    role = "emergency" if reason == "authority-compromise-recovery" else "policy"
    authority = (
        previous["policy"]["emergencyAuthority"]
        if role == "emergency"
        else previous["policy"]["policyAuthority"]
    )
    allowed = {x["keyId"]: x for x in authority["members"]}
    if len(selected) < authority["threshold"] or any(
        x not in allowed or x in set(previous["revoked_authority_key_ids"])
        for x in selected
    ):
        _fail("GOVERNANCE_SELECTED_APPROVER_QUORUM_INVALID")
    revocations = doc.get("revokedAuthorityKeyIds")
    if not isinstance(revocations, list) or len(set(revocations)) != len(revocations):
        _fail("GOVERNANCE_REVOCATION_SCHEMA_INVALID")
    revocations = [
        _identity(x, "GOVERNANCE_REVOCATION_KEY_INVALID") for x in revocations
    ]
    current_authority_keys = {
        m["keyId"] for m in previous["policy"]["policyAuthority"]["members"]
    } | {m["keyId"] for m in previous["policy"]["emergencyAuthority"]["members"]}
    if any(
        x not in current_authority_keys
        or x in set(previous["revoked_authority_key_ids"])
        for x in revocations
    ):
        _fail("GOVERNANCE_REVOCATION_KEY_NOT_CURRENT")
    if reason == "authority-compromise-recovery":
        current_policy_keys = {
            m["keyId"] for m in previous["policy"]["policyAuthority"]["members"]
        }
        if not set(revocations) & current_policy_keys:
            _fail("GOVERNANCE_COMPROMISE_RECOVERY_MUST_REVOKE_POLICY_KEY")
    next_policy = _policy(doc.get("nextPolicy"), "GOVERNANCE_PROPOSAL_NEXT_POLICY")
    if next_policy["policyVersion"] != previous["policy"]["policyVersion"] + 1:
        _fail("GOVERNANCE_POLICY_VERSION_NOT_ADVANCING")
    next_keys = {m["keyId"] for m in next_policy["policyAuthority"]["members"]} | {
        m["keyId"] for m in next_policy["emergencyAuthority"]["members"]
    }
    if (set(revocations) | set(previous["revoked_authority_key_ids"])) & next_keys:
        _fail("GOVERNANCE_REVOKED_KEY_IN_NEXT_POLICY")
    return {
        "transition_id": transition_id,
        "reason": reason,
        "role": role,
        "selected": selected,
        "allowed": allowed,
        "revocations": revocations,
        "next_policy": next_policy,
        "history": expected_history,
    }


def _archive_id(
    governance_id: str, epoch: int, snapshot_sha: str, identity: str
) -> str:
    return (
        "governance-archive-"
        + hashlib.sha256(
            (
                governance_id
                + "\0"
                + str(epoch)
                + "\0"
                + snapshot_sha
                + "\0"
                + identity
            ).encode()
        ).hexdigest()[:32]
    )


def _recovery_id(history_id: str, sequence: int, identity: str) -> str:
    return (
        "history-recovery-"
        + hashlib.sha256(
            (history_id + "\0" + str(sequence) + "\0" + identity).encode()
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
        chunks: list[bytes] = []
        err_chunks: list[bytes] = []
        overflow = threading.Event()
        limit = int(POLICY["max_adapter_output_bytes"])

        def drain(stream, sink):
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
                sink.append(chunk)

        threads = [
            threading.Thread(target=drain, args=(process.stdout, chunks)),
            threading.Thread(target=drain, args=(process.stderr, err_chunks)),
        ]
        for t in threads:
            t.start()
        try:
            stdin.write(_canonical_bytes(request))
            stdin.close()
            process.wait(timeout=int(POLICY["adapter_timeout_seconds"]))
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise GovernanceError(prefix + "_TIMEOUT") from exc
        finally:
            for t in threads:
                t.join()
        if overflow.is_set():
            _fail(prefix + "_OUTPUT_TOO_LARGE")
        if process.returncode != 0:
            _fail(prefix + "_FAILED")
        return _loads(b"".join(chunks), prefix + "_OUTPUT")

    return call


def command_recovery_source(  # ruff: ignore[undocumented-public-function]
    command: list[str],
):
    return _command_adapter(command, prefix="GOVERNANCE_RECOVERY_ADAPTER")


def command_archive(  # ruff: ignore[undocumented-public-function]
    command: list[str],
):
    return _command_adapter(command, prefix="GOVERNANCE_ARCHIVE_ADAPTER")


RecoveryAdapter = Callable[[dict[str, Any]], dict[str, Any]]
ArchiveAdapter = Callable[[dict[str, Any]], dict[str, Any]]


def _validate_recovery_response(
    value: dict[str, Any],
    *,
    recovery_id: str,
    history_id: str,
    identity: str,
    operator: str,
    now: datetime,
) -> dict[str, Any]:
    base = {
        "schemaVersion",
        "operation",
        "recoveryId",
        "historyId",
        "status",
        "source",
        "observedAt",
    }
    if (
        value.get("schemaVersion") != int(POLICY["recovery_protocol_version"])
        or value.get("operation") != "recover-history"
        or value.get("recoveryId") != recovery_id
        or value.get("historyId") != history_id
    ):
        _fail("GOVERNANCE_RECOVERY_RESULT_ID_MISMATCH")
    source = value.get("source")
    source_keys = {
        "identity",
        "operator",
        "readOnly",
        "historyWriterCredentialsReused",
        "governanceWriterCredentialsReused",
    }
    if (
        not isinstance(source, dict)
        or set(source) != source_keys
        or source.get("identity") != identity
        or source.get("operator") != operator
        or source.get("readOnly") is not True
        or source.get("historyWriterCredentialsReused") is not False
        or source.get("governanceWriterCredentialsReused") is not False
    ):
        _fail("GOVERNANCE_RECOVERY_SOURCE_AUTHORITY_INVALID")
    _timestamp(value.get("observedAt"), "GOVERNANCE_RECOVERY_OBSERVED_AT", now=now)
    if value.get("status") == "unavailable":
        if set(value) != base | {"reason"} or not bool(
            POLICY["allow_unavailable_recovery_source"]
        ):
            _fail("GOVERNANCE_RECOVERY_UNAVAILABLE_INVALID")
        return {"status": "unavailable", "identity": identity, "operator": operator}
    if value.get("status") != "recovered" or set(value) != base | {
        "snapshot",
        "locator",
        "proof",
    }:
        _fail("GOVERNANCE_RECOVERY_RESULT_SCHEMA_INVALID")
    locator = _safe_locator(value.get("locator"), "GOVERNANCE_RECOVERY_LOCATOR_INVALID")
    proof = value.get("proof")
    if proof != {
        "immutableArchiveReadbackVerified": True,
        "independentArchive": True,
        "canonicalBytesReturned": True,
    }:
        _fail("GOVERNANCE_RECOVERY_PROOF_INVALID")
    snapshot = value.get("snapshot")
    expected_snapshot = {"schemaVersion", "historyState", "historyBundle"}
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != expected_snapshot
        or snapshot.get("schemaVersion")
        != int(POLICY["recovery_snapshot_schema_version"])
    ):
        _fail("GOVERNANCE_RECOVERY_SNAPSHOT_SCHEMA_INVALID")
    state = snapshot.get("historyState")
    bundle = snapshot.get("historyBundle")
    if not isinstance(state, dict) or not isinstance(bundle, dict):
        _fail("GOVERNANCE_RECOVERY_SNAPSHOT_SCHEMA_INVALID")
    state_raw = _canonical_bytes(state)
    bundle_raw = _canonical_bytes(bundle)
    with tempfile.TemporaryDirectory(prefix="governance-recover-validate-") as tmp:
        sp = Path(tmp) / "state.json"
        bp = Path(tmp) / "bundle.json"
        sp.write_bytes(state_raw)
        bp.write_bytes(bundle_raw)
        try:
            info = history._validate_previous_history(sp, bp)
        except history.HistoryError as exc:
            raise GovernanceError(
                "GOVERNANCE_RECOVERY_HISTORY_INVALID:" + str(exc)
            ) from exc
    if info["history_id"] != history_id:
        _fail("GOVERNANCE_RECOVERY_HISTORY_ID_MISMATCH")
    return {
        "status": "recovered",
        "identity": identity,
        "operator": operator,
        "locator": locator,
        "state": state,
        "bundle": bundle,
        "state_sha256": _sha_bytes(state_raw),
        "bundle_sha256": _sha_bytes(bundle_raw),
        "sequence": info["sequence"],
        "chain_head_sha256": info["chain_head_sha256"],
    }


def recover_history(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    history_id: str,
    expected_sequence: int,
    expected_chain_head_sha256: str,
    output_dir: Path,
    recovery_sources: list[tuple[str, str, RecoveryAdapter]],
    recovery_quorum: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    history_id = _safe_text(
        history_id, "GOVERNANCE_RECOVERY_HISTORY_ID_INVALID", limit=255
    )
    expected_sequence = _size(expected_sequence, "GOVERNANCE_RECOVERY_SEQUENCE_INVALID")
    expected_chain_head_sha256 = _hex(
        expected_chain_head_sha256, "GOVERNANCE_RECOVERY_EXPECTED_CHAIN_HEAD_INVALID"
    )
    target = output_dir.expanduser().resolve()
    if target.exists() or target.is_symlink():
        _fail("GOVERNANCE_RECOVERY_OUTPUT_ALREADY_EXISTS")
    if not (
        int(POLICY["min_recovery_sources"])
        <= len(recovery_sources)
        <= int(POLICY["max_recovery_sources"])
    ):
        _fail("GOVERNANCE_RECOVERY_SOURCE_COUNT_INVALID")
    recovery_quorum = _size(recovery_quorum, "GOVERNANCE_RECOVERY_QUORUM_INVALID")
    if recovery_quorum < int(POLICY["min_recovery_quorum"]) or recovery_quorum > len(
        recovery_sources
    ):
        _fail("GOVERNANCE_RECOVERY_QUORUM_INVALID")
    normalized = []
    ids = set()
    ops = set()
    for ir, oraw, adapter in recovery_sources:
        i = _identity(ir, "GOVERNANCE_RECOVERY_IDENTITY_INVALID")
        o = _identity(oraw, "GOVERNANCE_RECOVERY_OPERATOR_INVALID")
        if i in ids:
            _fail("GOVERNANCE_RECOVERY_IDENTITY_DUPLICATE")
        ids.add(i)
        ops.add(o)
        normalized.append((i, o, adapter))
    if len(ops) < int(POLICY["min_recovery_operators"]):
        _fail("GOVERNANCE_RECOVERY_OPERATOR_COUNT_INVALID")
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    observed = []
    unavailable = []
    evidence = []
    locators = set()
    for i, o, adapter in sorted(normalized, key=lambda x: (x[1], x[0])):
        rid = _recovery_id(history_id, expected_sequence, i)
        request = {
            "schemaVersion": int(POLICY["recovery_protocol_version"]),
            "operation": "recover-history",
            "recoveryId": rid,
            "historyId": history_id,
            "expectedSequence": expected_sequence,
            "source": {"identity": i, "operator": o},
        }
        raw = adapter(request)
        result = _validate_recovery_response(
            raw,
            recovery_id=rid,
            history_id=history_id,
            identity=i,
            operator=o,
            now=current_time,
        )
        evidence.append(
            {
                "identity": i,
                "operator": o,
                "status": result["status"],
                "sha256": _sha_bytes(_canonical_bytes(raw)),
            }
        )
        if result["status"] == "recovered":
            if result["sequence"] != expected_sequence:
                _fail("GOVERNANCE_RECOVERY_SEQUENCE_MISMATCH")
            if result["locator"] in locators:
                _fail("GOVERNANCE_RECOVERY_LOCATOR_COLLISION")
            locators.add(result["locator"])
            observed.append(result)
        else:
            unavailable.append(result)
    if len(observed) < recovery_quorum:
        _fail("GOVERNANCE_RECOVERY_QUORUM_NOT_MET")
    if len({x["operator"] for x in observed}) < int(POLICY["min_recovery_operators"]):
        _fail("GOVERNANCE_RECOVERY_OPERATOR_QUORUM_NOT_MET")
    views = {
        (x["state_sha256"], x["bundle_sha256"], x["chain_head_sha256"])
        for x in observed
    }
    if len(views) != 1:
        _fail("GOVERNANCE_RECOVERY_ARCHIVE_DISAGREEMENT")
    chosen = min(observed, key=lambda x: (x["operator"], x["identity"]))
    if chosen["chain_head_sha256"] != expected_chain_head_sha256:
        _fail("GOVERNANCE_RECOVERY_CHAIN_HEAD_PIN_MISMATCH")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-recovery-", dir=target.parent
    ) as tmp:
        stage = Path(tmp) / "recovery"
        stage.mkdir()
        _write(stage / "recovered-trusted-history-state.json", chosen["state"])
        _write(stage / "recovered-release-history-bundle.json", chosen["bundle"])
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "recovered",
            "historyId": history_id,
            "sequence": expected_sequence,
            "expectedChainHeadSha256": expected_chain_head_sha256,
            "quorum": {
                "configured": len(normalized),
                "threshold": recovery_quorum,
                "observed": len(observed),
                "unavailable": len(unavailable),
                "minimumOperators": int(POLICY["min_recovery_operators"]),
            },
            "recovered": {
                "stateSha256": chosen["state_sha256"],
                "bundleSha256": chosen["bundle_sha256"],
                "chainHeadSha256": chosen["chain_head_sha256"],
            },
            "evidence": evidence,
        }
        _write(stage / "release-history-recovery-receipt.json", receipt)
        os.replace(stage, target)
    return {
        "ok": True,
        "phase": "history-recovered",
        "history_id": history_id,
        "sequence": expected_sequence,
        "observed": len(observed),
        "quorum": recovery_quorum,
        "state_sha256": chosen["state_sha256"],
        "bundle_sha256": chosen["bundle_sha256"],
        "chain_head_sha256": chosen["chain_head_sha256"],
    }


def _governance_recovery_id(governance_id: str, epoch: int, identity: str) -> str:
    return (
        "governance-recovery-"
        + hashlib.sha256(
            (governance_id + "\0" + str(epoch) + "\0" + identity).encode()
        ).hexdigest()[:32]
    )


def _validate_governance_recovery_response(
    value: dict[str, Any],
    *,
    recovery_id: str,
    governance_id: str,
    identity: str,
    operator: str,
    now: datetime,
) -> dict[str, Any]:
    base = {
        "schemaVersion",
        "operation",
        "recoveryId",
        "governanceId",
        "status",
        "source",
        "observedAt",
    }
    if (
        value.get("schemaVersion") != int(POLICY["recovery_protocol_version"])
        or value.get("operation") != "recover-governance"
        or value.get("recoveryId") != recovery_id
        or value.get("governanceId") != governance_id
    ):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_RESULT_ID_MISMATCH")
    source = value.get("source")
    source_keys = {
        "identity",
        "operator",
        "readOnly",
        "historyWriterCredentialsReused",
        "governanceWriterCredentialsReused",
    }
    if (
        not isinstance(source, dict)
        or set(source) != source_keys
        or source.get("identity") != identity
        or source.get("operator") != operator
    ):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_SOURCE_SCHEMA_INVALID")
    if (
        source.get("readOnly") is not True
        or source.get("historyWriterCredentialsReused") is not False
        or source.get("governanceWriterCredentialsReused") is not False
    ):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_SOURCE_AUTHORITY_INVALID")
    _timestamp(
        value.get("observedAt"), "GOVERNANCE_SNAPSHOT_RECOVERY_OBSERVED_AT", now=now
    )
    if value.get("status") == "unavailable":
        if set(value) != base | {"reason"} or not bool(
            POLICY["allow_unavailable_recovery_source"]
        ):
            _fail("GOVERNANCE_SNAPSHOT_RECOVERY_UNAVAILABLE_INVALID")
        return {"status": "unavailable", "identity": identity, "operator": operator}
    if value.get("status") != "recovered" or set(value) != base | {
        "snapshot",
        "locator",
        "proof",
    }:
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_RESULT_SCHEMA_INVALID")
    locator = _safe_locator(
        value.get("locator"), "GOVERNANCE_SNAPSHOT_RECOVERY_LOCATOR_INVALID"
    )
    if value.get("proof") != {
        "immutableArchiveReadbackVerified": True,
        "independentArchive": True,
        "canonicalBytesReturned": True,
    }:
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_PROOF_INVALID")
    snapshot = value.get("snapshot")
    expected = {
        "schemaVersion",
        "governanceState",
        "governanceBundle",
        "historyState",
        "historyBundle",
    }
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != expected
        or snapshot.get("schemaVersion")
        != int(POLICY["recovery_snapshot_schema_version"])
    ):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_SCHEMA_INVALID")
    gstate = snapshot.get("governanceState")
    gbundle = snapshot.get("governanceBundle")
    hstate = snapshot.get("historyState")
    hbundle = snapshot.get("historyBundle")
    if not all(isinstance(x, dict) for x in (gstate, gbundle, hstate, hbundle)):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_SCHEMA_INVALID")
    gstate_raw = _canonical_bytes(gstate)
    gbundle_raw = _canonical_bytes(gbundle)
    hstate_raw = _canonical_bytes(hstate)
    hbundle_raw = _canonical_bytes(hbundle)
    with tempfile.TemporaryDirectory(prefix="governance-snapshot-validate-") as tmp:
        root = Path(tmp)
        gsp = root / "governance-state.json"
        gbp = root / "governance-bundle.json"
        hsp = root / "history-state.json"
        hbp = root / "history-bundle.json"
        gsp.write_bytes(gstate_raw)
        gbp.write_bytes(gbundle_raw)
        hsp.write_bytes(hstate_raw)
        hbp.write_bytes(hbundle_raw)
        ginfo = verify_governance_bundle(bundle_path=gbp, state_path=gsp)
        try:
            hinfo = history._validate_previous_history(hsp, hbp)
        except history.HistoryError as exc:
            raise GovernanceError(
                "GOVERNANCE_SNAPSHOT_RECOVERY_HISTORY_INVALID:" + str(exc)
            ) from exc
    if (
        ginfo["governance_id"] != governance_id
        or gstate.get("historyId") != hinfo["history_id"]
    ):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_ID_MISMATCH")
    expected_history = {
        "sequence": hinfo["sequence"],
        "stateSha256": hinfo["state_sha256"],
        "bundleSha256": hinfo["bundle_sha256"],
        "chainHeadSha256": hinfo["chain_head_sha256"],
    }
    if gstate.get("history") != expected_history:
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_HISTORY_REBIND_FAILED")
    return {
        "status": "recovered",
        "identity": identity,
        "operator": operator,
        "locator": locator,
        "snapshot": snapshot,
        "snapshot_sha256": _sha_bytes(_canonical_bytes(snapshot)),
        "governance_state_sha256": _sha_bytes(gstate_raw),
        "governance_bundle_sha256": _sha_bytes(gbundle_raw),
        "governance_chain_head_sha256": ginfo["chain_head_sha256"],
        "epoch": ginfo["epoch"],
        "history_state_sha256": hinfo["state_sha256"],
        "history_bundle_sha256": hinfo["bundle_sha256"],
        "history_chain_head_sha256": hinfo["chain_head_sha256"],
        "history_sequence": hinfo["sequence"],
    }


def recover_governance(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    governance_id: str,
    expected_epoch: int,
    expected_chain_head_sha256: str,
    output_dir: Path,
    recovery_sources: list[tuple[str, str, RecoveryAdapter]],
    recovery_quorum: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    governance_id = _safe_text(
        governance_id, "GOVERNANCE_SNAPSHOT_RECOVERY_ID_INVALID", limit=255
    )
    expected_epoch = _size(expected_epoch, "GOVERNANCE_SNAPSHOT_RECOVERY_EPOCH_INVALID")
    expected_chain_head_sha256 = _hex(
        expected_chain_head_sha256,
        "GOVERNANCE_SNAPSHOT_RECOVERY_EXPECTED_CHAIN_HEAD_INVALID",
    )
    target = output_dir.expanduser().resolve()
    if target.exists() or target.is_symlink():
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_OUTPUT_ALREADY_EXISTS")
    if not (
        int(POLICY["min_recovery_sources"])
        <= len(recovery_sources)
        <= int(POLICY["max_recovery_sources"])
    ):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_SOURCE_COUNT_INVALID")
    recovery_quorum = _size(
        recovery_quorum, "GOVERNANCE_SNAPSHOT_RECOVERY_QUORUM_INVALID"
    )
    if recovery_quorum < int(POLICY["min_recovery_quorum"]) or recovery_quorum > len(
        recovery_sources
    ):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_QUORUM_INVALID")
    normalized = []
    ids = set()
    operators = set()
    for identity_raw, operator_raw, adapter in recovery_sources:
        identity = _identity(
            identity_raw, "GOVERNANCE_SNAPSHOT_RECOVERY_IDENTITY_INVALID"
        )
        operator = _identity(
            operator_raw, "GOVERNANCE_SNAPSHOT_RECOVERY_OPERATOR_INVALID"
        )
        if identity in ids:
            _fail("GOVERNANCE_SNAPSHOT_RECOVERY_IDENTITY_DUPLICATE")
        ids.add(identity)
        operators.add(operator)
        normalized.append((identity, operator, adapter))
    if len(operators) < int(POLICY["min_recovery_operators"]):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_OPERATOR_COUNT_INVALID")
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    observed = []
    unavailable = []
    evidence = []
    locators = set()
    for identity, operator, adapter in sorted(normalized, key=lambda x: (x[1], x[0])):
        recovery_id = _governance_recovery_id(governance_id, expected_epoch, identity)
        request = {
            "schemaVersion": int(POLICY["recovery_protocol_version"]),
            "operation": "recover-governance",
            "recoveryId": recovery_id,
            "governanceId": governance_id,
            "expectedEpoch": expected_epoch,
            "source": {"identity": identity, "operator": operator},
        }
        raw = adapter(request)
        result = _validate_governance_recovery_response(
            raw,
            recovery_id=recovery_id,
            governance_id=governance_id,
            identity=identity,
            operator=operator,
            now=current_time,
        )
        evidence.append(
            {
                "identity": identity,
                "operator": operator,
                "status": result["status"],
                "sha256": _sha_bytes(_canonical_bytes(raw)),
            }
        )
        if result["status"] == "recovered":
            if result["epoch"] != expected_epoch:
                _fail("GOVERNANCE_SNAPSHOT_RECOVERY_EPOCH_MISMATCH")
            if result["locator"] in locators:
                _fail("GOVERNANCE_SNAPSHOT_RECOVERY_LOCATOR_COLLISION")
            locators.add(result["locator"])
            observed.append(result)
        else:
            unavailable.append(result)
    if len(observed) < recovery_quorum:
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_QUORUM_NOT_MET")
    if len({x["operator"] for x in observed}) < int(POLICY["min_recovery_operators"]):
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_OPERATOR_QUORUM_NOT_MET")
    views = {
        (
            x["snapshot_sha256"],
            x["governance_state_sha256"],
            x["governance_bundle_sha256"],
            x["governance_chain_head_sha256"],
            x["history_state_sha256"],
            x["history_bundle_sha256"],
            x["history_chain_head_sha256"],
        )
        for x in observed
    }
    if len(views) != 1:
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_ARCHIVE_DISAGREEMENT")
    chosen = min(observed, key=lambda x: (x["operator"], x["identity"]))
    if chosen["governance_chain_head_sha256"] != expected_chain_head_sha256:
        _fail("GOVERNANCE_SNAPSHOT_RECOVERY_CHAIN_HEAD_PIN_MISMATCH")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="governance-snapshot-recovery-", dir=target.parent
    ) as tmp:
        stage = Path(tmp) / "recovery"
        stage.mkdir()
        snapshot = chosen["snapshot"]
        _write(
            stage / "recovered-trusted-governance-state.json",
            snapshot["governanceState"],
        )
        _write(
            stage / "recovered-release-governance-bundle.json",
            snapshot["governanceBundle"],
        )
        _write(stage / "recovered-trusted-history-state.json", snapshot["historyState"])
        _write(
            stage / "recovered-release-history-bundle.json", snapshot["historyBundle"]
        )
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "governance-recovered",
            "governanceId": governance_id,
            "epoch": expected_epoch,
            "expectedChainHeadSha256": expected_chain_head_sha256,
            "quorum": {
                "configured": len(normalized),
                "threshold": recovery_quorum,
                "observed": len(observed),
                "unavailable": len(unavailable),
                "minimumOperators": int(POLICY["min_recovery_operators"]),
            },
            "recovered": {
                "snapshotSha256": chosen["snapshot_sha256"],
                "governanceStateSha256": chosen["governance_state_sha256"],
                "governanceBundleSha256": chosen["governance_bundle_sha256"],
                "governanceChainHeadSha256": chosen["governance_chain_head_sha256"],
                "historyStateSha256": chosen["history_state_sha256"],
                "historyBundleSha256": chosen["history_bundle_sha256"],
                "historyChainHeadSha256": chosen["history_chain_head_sha256"],
            },
            "evidence": evidence,
        }
        _write(stage / "release-governance-recovery-receipt.json", receipt)
        os.replace(stage, target)
    return {
        "ok": True,
        "phase": "governance-recovered",
        "governance_id": governance_id,
        "epoch": expected_epoch,
        "observed": len(observed),
        "quorum": recovery_quorum,
        "snapshot_sha256": chosen["snapshot_sha256"],
        "state_sha256": chosen["governance_state_sha256"],
        "bundle_sha256": chosen["governance_bundle_sha256"],
        "chain_head_sha256": chosen["governance_chain_head_sha256"],
    }


def _validate_archive_response(
    value: dict[str, Any],
    *,
    operation: str,
    archive_id: str,
    governance_id: str,
    epoch: int,
    identity: str,
    operator: str,
    artifact: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    expected = {
        "schemaVersion",
        "operation",
        "archiveId",
        "governanceId",
        "epoch",
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
        or value.get("governanceId") != governance_id
        or value.get("epoch") != epoch
    ):
        _fail("GOVERNANCE_ARCHIVE_RESULT_SCHEMA_INVALID")
    if value.get("status") not in (
        {"created", "present"} if operation == "bind" else {"present"}
    ):
        _fail("GOVERNANCE_ARCHIVE_STATUS_INVALID")
    archive = value.get("archive")
    keys = {
        "identity",
        "operator",
        "governanceWriterCredentialsReused",
        "historyWriterCredentialsReused",
    }
    if (
        not isinstance(archive, dict)
        or set(archive) != keys
        or archive.get("identity") != identity
        or archive.get("operator") != operator
        or archive.get("governanceWriterCredentialsReused") is not False
        or archive.get("historyWriterCredentialsReused") is not False
    ):
        _fail("GOVERNANCE_ARCHIVE_AUTHORITY_INVALID")
    if value.get("artifact") != artifact:
        _fail("GOVERNANCE_ARCHIVE_ARTIFACT_MISMATCH")
    g = value.get("guarantees")
    if (
        not isinstance(g, dict)
        or set(g)
        != {
            "createOnly",
            "overwrite",
            "remoteReadbackVerified",
            "immutability",
            "locator",
        }
        or g.get("createOnly") is not True
        or g.get("overwrite") is not False
        or g.get("remoteReadbackVerified") is not True
    ):
        _fail("GOVERNANCE_ARCHIVE_GUARANTEES_INVALID")
    if g.get("immutability") not in set(POLICY["allowed_archive_immutability"]):
        _fail("GOVERNANCE_ARCHIVE_IMMUTABILITY_INVALID")
    locator = _safe_locator(g.get("locator"), "GOVERNANCE_ARCHIVE_LOCATOR_INVALID")
    _timestamp(value.get("verifiedAt"), "GOVERNANCE_ARCHIVE_VERIFIED_AT", now=now)
    return {"locator": locator, "immutability": g["immutability"]}


def apply_governance_transition(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    history_dir: Path,
    previous_state: Path,
    previous_bundle: Path,
    proposal_path: Path,
    approval_paths: list[Path],
    output_dir: Path,
    governance_archives: list[tuple[str, str, ArchiveAdapter]],
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _validate_run153_dir(history_dir)
    previous = _validate_previous_governance(previous_state, previous_bundle)
    if current["history_id"] != previous["history_id"]:
        _fail("GOVERNANCE_HISTORY_ID_MISMATCH")
    if current["sequence"] < previous["history"]["sequence"]:
        _fail("GOVERNANCE_HISTORY_ROLLBACK")
    target = _outside(output_dir, (current["root"],), "GOVERNANCE_OUTPUT_INSIDE_INPUT")
    if target.exists() or target.is_symlink():
        _fail("GOVERNANCE_OUTPUT_ALREADY_EXISTS")
    proposal, proposal_raw = _read(proposal_path, "GOVERNANCE_PROPOSAL")
    p = _validate_proposal(proposal, previous=previous, current=current)
    proposal_sha = _sha_bytes(proposal_raw)
    role = p["role"]
    if len(approval_paths) != len(p["selected"]):
        _fail("GOVERNANCE_APPROVAL_COUNT_INVALID")
    approvals = []
    keys = []
    operators = set()
    hashes = []
    for path in approval_paths:
        doc, raw = _read(path, "GOVERNANCE_APPROVAL")
        approver = _validate_approval(
            doc,
            governance_id=previous["governance_id"],
            transition_id=p["transition_id"],
            proposal_sha=proposal_sha,
            role=role,
            allowed_members=p["allowed"],
            revoked=set(previous["revoked_authority_key_ids"]),
            now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc),
        )
        approvals.append(doc)
        keys.append(approver["keyId"])
        operators.add(approver["operator"])
        hashes.append(_sha_bytes(raw))
    if sorted(keys) != sorted(p["selected"]):
        _fail("GOVERNANCE_APPROVAL_SET_MISMATCH")
    min_ops = int(
        POLICY["min_emergency_operators"]
        if role == "emergency"
        else POLICY["min_policy_operators"]
    )
    if len(operators) < min_ops:
        _fail("GOVERNANCE_APPROVAL_OPERATOR_QUORUM_INVALID")
    approvals = sorted(
        approvals,
        key=lambda x: (
            x["approver"]["operator"],
            x["approver"]["identity"],
            x["approver"]["keyId"],
        ),
    )
    entry = {
        "epoch": previous["epoch"] + 1,
        "transitionId": p["transition_id"],
        "reason": p["reason"],
        "history": p["history"],
        "proposal": proposal,
        "proposalSha256": proposal_sha,
        "approvalRole": role,
        "selectedApproverKeyIds": sorted(p["selected"]),
        "approvals": approvals,
        "revokedAuthorityKeyIds": sorted(p["revocations"]),
        "nextPolicy": p["next_policy"],
    }
    new_bundle = dict(previous["bundle"])
    new_bundle["entries"] = [*previous["entries"], entry]
    info = _validate_bundle(new_bundle)
    bundle_raw = _canonical_bytes(new_bundle)
    bundle_sha = _sha_bytes(bundle_raw)
    history_info = {
        "sequence": current["sequence"],
        "state_sha256": current["state_sha256"],
        "bundle_sha256": current["bundle_sha256"],
        "chain_head_sha256": current["chain_head_sha256"],
    }
    new_state = _state_for(info=info, history_info=history_info, bundle_sha=bundle_sha)
    snapshot = {
        "schemaVersion": int(POLICY["recovery_snapshot_schema_version"]),
        "governanceState": new_state,
        "governanceBundle": new_bundle,
        "historyState": current["state"],
        "historyBundle": current["bundle"],
    }
    snapshot_item = _artifact_doc(snapshot, "release-governance-recovery-snapshot.json")
    if not (
        int(POLICY["min_governance_archives"])
        <= len(governance_archives)
        <= int(POLICY["max_governance_archives"])
    ):
        _fail("GOVERNANCE_ARCHIVE_COUNT_INVALID")
    normalized = []
    ids = set()
    ops = set()
    for ir, oraw, adapter in governance_archives:
        i = _identity(ir, "GOVERNANCE_ARCHIVE_IDENTITY_INVALID")
        o = _identity(oraw, "GOVERNANCE_ARCHIVE_OPERATOR_INVALID")
        if i in ids:
            _fail("GOVERNANCE_ARCHIVE_IDENTITY_DUPLICATE")
        ids.add(i)
        ops.add(o)
        normalized.append((i, o, adapter))
    if len(ops) < int(POLICY["min_governance_archive_operators"]):
        _fail("GOVERNANCE_ARCHIVE_OPERATOR_COUNT_INVALID")
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run153_inputs = sorted(
        (p for p in current["root"].rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(current["root"])),
    )
    input_paths = [
        *run153_inputs,
        previous_state,
        previous_bundle,
        proposal_path,
        *approval_paths,
    ]
    initial = {str(x.resolve()): _sha(x) for x in input_paths}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-governance-", dir=target.parent
    ) as tmp:
        stage = Path(tmp) / "governance"
        stage.mkdir()
        archive_dir = stage / "archive-results"
        archive_dir.mkdir()
        bundle_path = stage / "release-governance-bundle.json"
        state_path = stage / _DOC_GOVERNANCE_STATE
        snapshot_path = stage / "release-governance-recovery-snapshot.json"
        _write(bundle_path, new_bundle)
        _write(state_path, new_state)
        _write(snapshot_path, snapshot)
        verify_governance_bundle(bundle_path=bundle_path, state_path=state_path)
        archive_evidence = []
        locators = set()
        for index, (i, o, adapter) in enumerate(
            sorted(normalized, key=lambda x: (x[1], x[0])), start=1
        ):
            aid = _archive_id(
                previous["governance_id"], info["epoch"], snapshot_item["sha256"], i
            )
            base = {
                "schemaVersion": int(POLICY["archive_protocol_version"]),
                "archiveId": aid,
                "governanceId": previous["governance_id"],
                "epoch": info["epoch"],
                "archive": {"identity": i, "operator": o},
                "artifact": snapshot_item,
            }
            bind_req = dict(base, operation="bind")
            bind_req["artifact"] = dict(snapshot_item, localPath=str(snapshot_path))
            before = _sha(snapshot_path)
            bind_raw = adapter(bind_req)
            if _sha(snapshot_path) != before:
                _fail("GOVERNANCE_SNAPSHOT_CHANGED_DURING_ARCHIVE")
            bind = _validate_archive_response(
                bind_raw,
                operation="bind",
                archive_id=aid,
                governance_id=previous["governance_id"],
                epoch=info["epoch"],
                identity=i,
                operator=o,
                artifact=snapshot_item,
                now=current_time,
            )
            verify_raw = adapter(dict(base, operation="verify"))
            verify = _validate_archive_response(
                verify_raw,
                operation="verify",
                archive_id=aid,
                governance_id=previous["governance_id"],
                epoch=info["epoch"],
                identity=i,
                operator=o,
                artifact=snapshot_item,
                now=current_time,
            )
            if bind != verify:
                _fail("GOVERNANCE_ARCHIVE_VERIFY_REBIND_FAILED")
            if verify["locator"] in locators:
                _fail("GOVERNANCE_ARCHIVE_LOCATOR_COLLISION")
            locators.add(verify["locator"])
            bp = (
                archive_dir
                / f"{index:02d}-{hashlib.sha256(i.encode()).hexdigest()[:12]}.bind.json"
            )
            vp = (
                archive_dir
                / f"{index:02d}-{hashlib.sha256(i.encode()).hexdigest()[:12]}.verify.json"
            )
            _write(bp, bind_raw)
            _write(vp, verify_raw)
            archive_evidence.append(
                {
                    "identity": i,
                    "operator": o,
                    "archiveId": aid,
                    "locator": verify["locator"],
                    "immutability": verify["immutability"],
                    "bindEvidenceSha256": _sha(bp),
                    "verifyEvidenceSha256": _sha(vp),
                }
            )
        for path in input_paths:
            if _sha(path) != initial[str(path.resolve())]:
                _fail("GOVERNANCE_INPUT_CHANGED_DURING_TRANSITION")
        receipt = {
            "schemaVersion": int(POLICY["receipt_schema_version"]),
            "status": "governed",
            "governanceId": previous["governance_id"],
            "historyId": current["history_id"],
            "epoch": info["epoch"],
            "policyVersion": info["policy"]["policyVersion"],
            "proposalSha256": proposal_sha,
            "approvalEvidenceSha256": sorted(hashes),
            "history": {
                "sequence": current["sequence"],
                "stateSha256": current["state_sha256"],
                "bundleSha256": current["bundle_sha256"],
                "chainHeadSha256": current["chain_head_sha256"],
            },
            "governance": {
                "bundle": _artifact_doc(new_bundle, "release-governance-bundle.json"),
                "state": _artifact_doc(new_state, _DOC_GOVERNANCE_STATE),
                "recoverySnapshot": snapshot_item,
            },
            "archives": archive_evidence,
        }
        _write(stage / "release-governance-receipt.json", receipt)
        shutil.copy2(proposal_path, stage / "governance-transition-proposal.json")
        approvals_dir = stage / "approval-evidence"
        approvals_dir.mkdir()
        for index, path in enumerate(
            sorted(approval_paths, key=lambda p: p.name), start=1
        ):
            shutil.copy2(path, approvals_dir / f"{index:02d}-{path.name}")
        shutil.copy2(
            current["root"] / _DOC_HISTORY_STATE,
            stage / _DOC_HISTORY_STATE,
        )
        shutil.copy2(
            current["root"] / "release-history-bundle.json",
            stage / "release-history-bundle.json",
        )
        tmp_target = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, tmp_target, copy_function=shutil.copy2)
        os.replace(tmp_target, target)
    return {
        "ok": True,
        "phase": "governance-transition-applied",
        "governance_id": previous["governance_id"],
        "epoch": info["epoch"],
        "policy_version": info["policy"]["policyVersion"],
        "history_sequence": current["sequence"],
        "bundle_sha256": _sha(target / "release-governance-bundle.json"),
        "state_sha256": _sha(target / _DOC_GOVERNANCE_STATE),
        "recovery_snapshot_sha256": _sha(
            target / "release-governance-recovery-snapshot.json"
        ),
        "archive_count": len(normalized),
    }


def _parse_endpoint(value: str, label: str) -> tuple[str, str, list[str]]:
    parts = value.split("=", 1)
    if len(parts) != 2 or "@" not in parts[0]:  # ruff: ignore[magic-value-comparison]
        raise argparse.ArgumentTypeError(
            f"{label} must be IDENTITY@OPERATOR=EXECUTABLE[,ARG...]"
        )
    identity, operator = parts[0].split("@", 1)
    command = [x for x in parts[1].split(",") if x]
    if not identity or not operator or not command:
        raise argparse.ArgumentTypeError(f"invalid {label} specification")
    return identity, operator, command


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--state", type=Path)
    init = sub.add_parser("initialize")
    init.add_argument("--history-dir", type=Path, required=True)
    init.add_argument("--genesis", type=Path, required=True)
    init.add_argument("--expected-genesis-sha256", required=True)
    init.add_argument("--output-dir", type=Path, required=True)
    recover = sub.add_parser("recover-history")
    recover.add_argument("--history-id", required=True)
    recover.add_argument("--expected-sequence", type=int, required=True)
    recover.add_argument("--expected-chain-head-sha256", required=True)
    recover.add_argument("--output-dir", type=Path, required=True)
    recover.add_argument("--recovery-quorum", type=int, required=True)
    recover.add_argument(
        "--source",
        action="append",
        required=True,
        type=lambda x: _parse_endpoint(x, "source"),
    )
    recover_gov = sub.add_parser("recover-governance")
    recover_gov.add_argument("--governance-id", required=True)
    recover_gov.add_argument("--expected-epoch", type=int, required=True)
    recover_gov.add_argument("--expected-chain-head-sha256", required=True)
    recover_gov.add_argument("--output-dir", type=Path, required=True)
    recover_gov.add_argument("--recovery-quorum", type=int, required=True)
    recover_gov.add_argument(
        "--source",
        action="append",
        required=True,
        type=lambda x: _parse_endpoint(x, "source"),
    )
    advance = sub.add_parser("transition")
    advance.add_argument("--history-dir", type=Path, required=True)
    advance.add_argument("--previous-state", type=Path, required=True)
    advance.add_argument("--previous-bundle", type=Path, required=True)
    advance.add_argument("--proposal", type=Path, required=True)
    advance.add_argument("--approval", action="append", required=True, type=Path)
    advance.add_argument("--output-dir", type=Path, required=True)
    advance.add_argument(
        "--archive",
        action="append",
        required=True,
        type=lambda x: _parse_endpoint(x, "archive"),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_governance_bundle(
                bundle_path=args.bundle, state_path=args.state
            )
        elif args.command == "initialize":
            result = initialize_governance(
                history_dir=args.history_dir,
                genesis_path=args.genesis,
                expected_genesis_sha256=args.expected_genesis_sha256,
                output_dir=args.output_dir,
            )
        elif args.command == "recover-history":
            sources = [(i, o, command_recovery_source(c)) for i, o, c in args.source]
            result = recover_history(
                history_id=args.history_id,
                expected_sequence=args.expected_sequence,
                expected_chain_head_sha256=args.expected_chain_head_sha256,
                output_dir=args.output_dir,
                recovery_sources=sources,
                recovery_quorum=args.recovery_quorum,
            )
        elif args.command == "recover-governance":
            sources = [(i, o, command_recovery_source(c)) for i, o, c in args.source]
            result = recover_governance(
                governance_id=args.governance_id,
                expected_epoch=args.expected_epoch,
                expected_chain_head_sha256=args.expected_chain_head_sha256,
                output_dir=args.output_dir,
                recovery_sources=sources,
                recovery_quorum=args.recovery_quorum,
            )
        else:
            archives = [(i, o, command_archive(c)) for i, o, c in args.archive]
            result = apply_governance_transition(
                history_dir=args.history_dir,
                previous_state=args.previous_state,
                previous_bundle=args.previous_bundle,
                proposal_path=args.proposal,
                approval_paths=args.approval,
                output_dir=args.output_dir,
                governance_archives=archives,
            )
    except (GovernanceError, history.HistoryError, OSError) as exc:
        logger.error("governance command failed: %s", exc)
        # sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True) + "\n")
        return 2
    logger.info("governance command completed successfully")
    # sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
