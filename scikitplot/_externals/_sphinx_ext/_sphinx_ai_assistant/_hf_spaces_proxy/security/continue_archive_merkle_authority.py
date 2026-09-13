"""
Run 166: continue RFC6962 Merkle epochs under Run 165 active authority.

This layer bridges the last accepted Run 164 Merkle checkpoint into later epochs
signed by the rotated/recovered log and gossip keys accepted by Run 165.  The
retired pre-handoff key is never required again.  Every post-handoff event still
uses the Run 164 RFC6962 inclusion/consistency proof verifier; only authority
selection and cumulative continuity are upgraded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import tomllib

logger = logging.getLogger(__name__)

try:
    from . import govern_archive_merkle_log_authority as authority
    from . import verify_archive_merkle_transparency as merkle
except (ImportError, ValueError) as exc:
    import importlib.util

    _here = Path(__file__).resolve().parent

    def _load(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(name, _here / filename)
        if spec is None or spec.loader is None:
            raise ImportError(filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    merkle = _load("_run164_merkle_for_run166", "verify_archive_merkle_transparency.py")
    authority = _load(
        "_run165_authority_for_run166", "govern_archive_merkle_log_authority.py"
    )

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads(
    (HERE / "release_archive_merkle_continuity_policy.toml").read_text()
)
PREDICATE_TYPE = str(POLICY["predicate_type"])


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_ARCHIVE_ANCHOR_STATE = "trusted-archive-anchor-state.json"
_DOC_ARCHIVE_LOG_AUTHORITY_STATE = "trusted-archive-log-authority-state.json"
_DOC_ARCHIVE_MERKLE_CONTINUITY_STATE = "trusted-archive-merkle-continuity-state.json"
_DOC_ARCHIVE_MERKLE_STATE = "trusted-archive-merkle-state.json"


_OUTPUT_NAMES = {
    "release-archive-merkle-continuity-bundle.json",
    _DOC_ARCHIVE_MERKLE_CONTINUITY_STATE,
    "active-archive-merkle-continuity.json",
    "release-archive-merkle-continuity-receipt.json",
}
_RUN163_NAMES = {
    "release-archive-anchor-bundle.json",
    _DOC_ARCHIVE_ANCHOR_STATE,
    "active-archive-anchor-evidence.json",
    "release-archive-anchor-receipt.json",
}
_RUN164_NAMES = {
    "release-archive-merkle-bundle.json",
    _DOC_ARCHIVE_MERKLE_STATE,
    "active-archive-merkle-evidence.json",
    "release-archive-merkle-receipt.json",
}
_RUN165_NAMES = {
    "release-archive-log-authority-bundle.json",
    _DOC_ARCHIVE_LOG_AUTHORITY_STATE,
    "active-archive-log-authority.json",
    "release-archive-log-authority-receipt.json",
}
_CHUNK = 1024 * 1024


class ArchiveMerkleContinuityError(  # ruff: ignore[undocumented-public-class]
    RuntimeError
):
    pass


def _fail(code: str) -> None:
    raise ArchiveMerkleContinuityError(code)


def _canonical(value: Any) -> bytes:
    return merkle._canonical(value)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value))


def _loads(raw: bytes, code: str) -> dict[str, Any]:
    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                _fail(code + "_DUPLICATE_KEY")
            out[key] = value
        return out

    if len(raw) > int(POLICY["max_json_bytes"]):
        _fail(code + "_TOO_LARGE")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except ArchiveMerkleContinuityError:
        raise
    except Exception as exc:
        raise ArchiveMerkleContinuityError(code + "_JSON_INVALID") from exc
    if not isinstance(value, dict):
        _fail(code + "_SCHEMA_INVALID")
    if raw != _canonical(value):
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


def _regular_dir(path: Path, code: str) -> Path:
    path = Path(path)
    try:
        st = path.lstat()
    except OSError:
        _fail(code)
    if not stat.S_ISDIR(st.st_mode):
        _fail(code)
    return path.resolve()


def _outside(path: Path, protected: list[Path], code: str) -> Path:
    p = Path(path).absolute()
    for item in protected:
        q = Path(item).absolute()
        if p == q or q in p.parents or p in q.parents:
            _fail(code)
    return p


def _dir_fingerprint(root: Path, code: str) -> tuple[tuple[str, str, int, int], ...]:
    root = _regular_dir(root, code)
    rows = []
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        if stat.S_ISDIR(st.st_mode):
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            _fail(code)
        rows.append(
            (
                path.relative_to(root).as_posix(),
                _sha(path),
                st.st_size,
                stat.S_IMODE(st.st_mode),
            )
        )
    return tuple(rows)


def _authority_fingerprint(paths: list[Path], code: str) -> tuple[Any, ...]:
    rows = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            rows.append((str(p.absolute()), "dir", _dir_fingerprint(p, code)))
        else:
            try:
                st = p.lstat()
            except OSError:
                _fail(code)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                _fail(code)
            rows.append(
                (
                    str(p.absolute()),
                    "file",
                    _sha(p),
                    st.st_size,
                    stat.S_IMODE(st.st_mode),
                )
            )
    return tuple(rows)


def _artifact_map(raws: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": _sha_bytes(raws[name]), "size": len(raws[name])}
        for name in sorted(raws)
    }


def _load_run164(
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    *,
    now: datetime,
) -> dict[str, Any]:
    try:
        return authority._load_run164_offline(
            Path(run164_dir),
            Path(transparency_root_path),
            transparency_root_pin,
            now=now,
            historical=True,
        )
    except Exception as exc:
        raise ArchiveMerkleContinuityError(
            "ARCHIVE_MERKLE_CONTINUITY_RUN164_INVALID:" + str(exc)
        ) from exc


def _load_run165(
    *,
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    governance_root_path: Path,
    governance_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    run165_dir: Path,
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    try:
        authority.verify_log_authority_history(
            run164_dir=run164_dir,
            transparency_root_path=transparency_root_path,
            transparency_root_pin=transparency_root_pin,
            governance_root_path=governance_root_path,
            governance_root_pin=governance_root_pin,
            recovery_root_path=recovery_root_path,
            recovery_root_pin=recovery_root_pin,
            output_dir=run165_dir,
            now=now,
            historical=True,
        )
        docs, raws = authority._load_output(Path(run165_dir))
    except Exception as exc:
        raise ArchiveMerkleContinuityError(
            "ARCHIVE_MERKLE_CONTINUITY_RUN165_INVALID:" + str(exc)
        ) from exc
    state = docs[_DOC_ARCHIVE_LOG_AUTHORITY_STATE]
    active = docs["active-archive-log-authority.json"]
    if state.get("sequence") != active.get("sequence") or state.get(
        "logAuthorityChainHeadSha256"
    ) != active.get("logAuthorityChainHeadSha256"):
        _fail("ARCHIVE_MERKLE_CONTINUITY_RUN165_STATE_ACTIVE_MISMATCH")
    authority_map = active.get("authority")
    if authority_map != state.get("activeAuthority"):
        _fail("ARCHIVE_MERKLE_CONTINUITY_RUN165_AUTHORITY_MISMATCH")
    if active.get("continuityCheckpoints") != state.get("continuityCheckpoints"):
        _fail("ARCHIVE_MERKLE_CONTINUITY_RUN165_CHECKPOINTS_MISMATCH")
    bundle = docs["release-archive-log-authority-bundle.json"]
    receipt = docs["release-archive-log-authority-receipt.json"]
    last_transition = receipt["events"][-1]["transitionDocument"]["signed"]
    issued = merkle._dt(
        last_transition["issuedAt"], "ARCHIVE_MERKLE_CONTINUITY_RUN165_ISSUED_INVALID"
    )
    if not historical:
        if issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
            _fail("ARCHIVE_MERKLE_CONTINUITY_RUN165_FROM_FUTURE")
        if now - issued > timedelta(days=int(POLICY["max_active_authority_age_days"])):
            _fail("ARCHIVE_MERKLE_CONTINUITY_RUN165_AUTHORITY_STALE")
    return {
        "docs": docs,
        "raws": raws,
        "state": state,
        "active": active,
        "authority": authority_map,
        "artifacts": _artifact_map(raws),
        "issued": issued,
        "bundle": bundle,
    }


def _load_run163_current(
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
    run162_dir: Path,
    anchor_plan_path: Path,
    run163_dir: Path,
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    try:
        return merkle._verify_run163_current(
            run160_dir=run160_dir,
            run161_dir=run161_dir,
            retention_root_path=retention_root_path,
            retention_root_pin=retention_root_pin,
            witness_root_path=witness_root_path,
            witness_root_pin=witness_root_pin,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            run162_dir=run162_dir,
            anchor_plan_path=anchor_plan_path,
            run163_dir=run163_dir,
            now=now,
            historical=historical,
        )
    except Exception as exc:
        raise ArchiveMerkleContinuityError(
            "ARCHIVE_MERKLE_CONTINUITY_RUN163_INVALID:" + str(exc)
        ) from exc


def _enforce_external_separation(
    active_authority: dict[str, Any], witness_root_path: Path, anchor_plan_path: Path
) -> None:
    try:
        wdoc, _ = merkle._read_json(
            Path(witness_root_path), "ARCHIVE_MERKLE_CONTINUITY_WITNESS_ROOT"
        )
        pdoc, _ = merkle._read_json(
            Path(anchor_plan_path), "ARCHIVE_MERKLE_CONTINUITY_ANCHOR_PLAN"
        )
    except Exception as exc:
        raise ArchiveMerkleContinuityError(
            "ARCHIVE_MERKLE_CONTINUITY_EXTERNAL_AUTHORITY_INVALID:" + str(exc)
        ) from exc
    other_ops = set()
    other_pubs = set()
    for value in wdoc.get("signed", {}).get("keys", {}).values():
        if isinstance(value, dict):
            other_ops.add(value.get("operator"))
            other_pubs.add(value.get("publicKey"))
    for value in pdoc.get("signed", {}).get("channels", {}).values():
        if isinstance(value, dict):
            other_ops.update([value.get("operator"), value.get("observerOperator")])
            other_pubs.update([value.get("publicKey"), value.get("observerPublicKey")])
    other_ops.discard(None)
    other_pubs.discard(None)
    own_ops = {v["operator"] for v in active_authority.values()} | {
        v["gossipOperator"] for v in active_authority.values()
    }
    own_pubs = {v["publicKey"] for v in active_authority.values()} | {
        v["gossipPublicKey"] for v in active_authority.values()
    }
    if own_ops & other_ops or own_pubs & other_pubs:
        _fail("ARCHIVE_MERKLE_CONTINUITY_EXTERNAL_AUTHORITY_OVERLAP")


def _load_output(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_MERKLE_CONTINUITY_OUTPUT_INVALID")
    if {p.name for p in root.iterdir()} != _OUTPUT_NAMES:
        _fail("ARCHIVE_MERKLE_CONTINUITY_OUTPUT_ALLOWLIST_INVALID")
    docs = {}
    raws = {}
    for name in sorted(_OUTPUT_NAMES):
        docs[name], raws[name] = _read_json(
            root / name, "ARCHIVE_MERKLE_CONTINUITY_OUTPUT"
        )
    return docs, raws


def _event_head(previous_head: str, event_without_head: dict[str, Any]) -> str:
    return _sha_bytes(
        _canonical(
            {
                "previousMerkleAuthorityContinuityHeadSha256": previous_head,
                "event": event_without_head,
            }
        )
    )


def _log_challenge(
    *,
    sequence: int,
    run163_head: str,
    leaf_hash: str,
    previous_head: str,
    authority_head: str,
    active_authority_sha: str,
) -> str:
    return _sha_bytes(
        _canonical(
            {
                "sequence": sequence,
                "run163AnchorConsensusHeadSha256": run163_head,
                "leafHash": leaf_hash,
                "previousMerkleAuthorityContinuityHeadSha256": previous_head,
                "run165LogAuthorityChainHeadSha256": authority_head,
                "activeAuthoritySha256": active_authority_sha,
            }
        )
    )


def _gossip_challenge(
    *,
    sequence: int,
    run163_head: str,
    leaf_hash: str,
    checkpoints: dict[str, str],
    authority_head: str,
    active_authority_sha: str,
) -> str:
    return _sha_bytes(
        _canonical(
            {
                "sequence": sequence,
                "run163AnchorConsensusHeadSha256": run163_head,
                "leafHash": leaf_hash,
                "checkpointSha256s": checkpoints,
                "run165LogAuthorityChainHeadSha256": authority_head,
                "activeAuthoritySha256": active_authority_sha,
            }
        )
    )


def _normalize_merkle_error(prefix: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ArchiveMerkleContinuityError:
        raise
    except Exception as exc:
        raise ArchiveMerkleContinuityError(prefix + ":" + str(exc)) from exc


def _base_checkpoints(run165: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cps = run165["state"].get("continuityCheckpoints")
    if not isinstance(cps, dict) or set(cps) != set(run165["authority"]):
        _fail("ARCHIVE_MERKLE_CONTINUITY_BASE_CHECKPOINTS_INVALID")
    out = {}
    for lid in sorted(cps):
        row = cps[lid]
        if not isinstance(row, dict) or set(row) != {
            "checkpointSha256",
            "treeSize",
            "rootHash",
        }:
            _fail("ARCHIVE_MERKLE_CONTINUITY_BASE_CHECKPOINT_INVALID")
        out[lid] = {
            "logId": lid,
            **row,
            "leafIndex": row["treeSize"] - 1,
            "leafHash": None,
        }
    return out


def _replay(  # ruff: ignore[too-many-branches]
    *,
    run164: dict[str, Any],
    run165: dict[str, Any],
    bundle: dict[str, Any],
    receipt: dict[str, Any],
    now: datetime,
    historical: bool,
    current_run163_docs: dict[str, Any] | None,
    current_run163_raws: dict[str, bytes] | None,
) -> dict[str, Any]:
    expected_bundle = {
        "schemaVersion",
        "predicateType",
        "status",
        "run164Artifacts",
        "run165Artifacts",
        "baseRun164Sequence",
        "baseMerkleConsensusHeadSha256",
        "run165Sequence",
        "run165LogAuthorityChainHeadSha256",
        "activeAuthoritySha256",
        "events",
    }
    if (
        set(bundle) != expected_bundle
        or bundle.get("schemaVersion") != int(POLICY["bundle_schema_version"])
        or bundle.get("predicateType") != PREDICATE_TYPE
        or bundle.get("status") != "archive-merkle-authority-continuity-history"
    ):
        _fail("ARCHIVE_MERKLE_CONTINUITY_BUNDLE_SCHEMA_INVALID")
    if (
        bundle["run164Artifacts"] != run164["artifacts"]
        or bundle["run165Artifacts"] != run165["artifacts"]
    ):
        _fail("ARCHIVE_MERKLE_CONTINUITY_BASE_ARTIFACT_MISMATCH")
    base_seq = run164["replay"]["sequence"]
    base_head = run164["replay"]["merkleConsensusHeadSha256"]
    authority_sha = _sha_bytes(_canonical(run165["authority"]))
    if (
        bundle["baseRun164Sequence"] != base_seq
        or bundle["baseMerkleConsensusHeadSha256"] != base_head
    ):
        _fail("ARCHIVE_MERKLE_CONTINUITY_BASE_MISMATCH")
    if (
        bundle["run165Sequence"] != run165["state"]["sequence"]
        or bundle["run165LogAuthorityChainHeadSha256"]
        != run165["state"]["logAuthorityChainHeadSha256"]
        or bundle["activeAuthoritySha256"] != authority_sha
    ):
        _fail("ARCHIVE_MERKLE_CONTINUITY_AUTHORITY_BINDING_INVALID")
    if (
        set(receipt)
        != {"schemaVersion", "status", "run164Documents", "run165Documents", "events"}
        or receipt.get("schemaVersion") != int(POLICY["receipt_schema_version"])
        or receipt.get("status") != "archive-merkle-authority-continuity-accepted"
    ):
        _fail("ARCHIVE_MERKLE_CONTINUITY_RECEIPT_SCHEMA_INVALID")
    if (
        receipt["run164Documents"] != run164["docs"]
        or receipt["run165Documents"] != run165["docs"]
    ):
        _fail("ARCHIVE_MERKLE_CONTINUITY_EMBEDDED_BASE_MISMATCH")
    events = bundle["events"]
    recs = receipt["events"]
    if (
        not isinstance(events, list)
        or not events
        or not isinstance(recs, list)
        or len(events) != len(recs)
    ):
        _fail("ARCHIVE_MERKLE_CONTINUITY_HISTORY_LENGTH_INVALID")
    previous_head = base_head
    previous_by_log = _base_checkpoints(run165)
    last = None
    for idx, (event, rec) in enumerate(zip(events, recs), 1):
        sequence = base_seq + idx
        expected_event_keys = {
            "sequence",
            "postHandoffSequence",
            "run163Sequence",
            "run163AnchorConsensusHeadSha256",
            "run163Artifacts",
            "leafHash",
            "run165Sequence",
            "run165LogAuthorityChainHeadSha256",
            "activeAuthoritySha256",
            "logs",
            "logChallenge",
            "gossipChallenge",
            "gossipResponseSha256s",
            "merkleAuthorityContinuityHeadSha256",
        }
        if (
            not isinstance(event, dict)
            or set(event) != expected_event_keys
            or event["sequence"] != sequence
            or event["postHandoffSequence"] != idx
            or event["run163Sequence"] != sequence
        ):
            _fail("ARCHIVE_MERKLE_CONTINUITY_EVENT_SCHEMA_INVALID")
        if (
            event["run165Sequence"] != run165["state"]["sequence"]
            or event["run165LogAuthorityChainHeadSha256"]
            != run165["state"]["logAuthorityChainHeadSha256"]
            or event["activeAuthoritySha256"] != authority_sha
        ):
            _fail("ARCHIVE_MERKLE_CONTINUITY_EVENT_AUTHORITY_INVALID")
        if (
            not isinstance(rec, dict)
            or set(rec)
            != {"sequence", "run163Documents", "logResponses", "gossipResponses"}
            or rec["sequence"] != sequence
        ):
            _fail("ARCHIVE_MERKLE_CONTINUITY_RECEIPT_EVENT_INVALID")
        rdocs = rec["run163Documents"]
        if not isinstance(rdocs, dict) or set(rdocs) != _RUN163_NAMES:
            _fail("ARCHIVE_MERKLE_CONTINUITY_RUN163_DOCUMENTS_INVALID")
        rraws = {name: _canonical(rdocs[name]) for name in sorted(rdocs)}
        artifacts = _artifact_map(rraws)
        if event["run163Artifacts"] != artifacts:
            _fail("ARCHIVE_MERKLE_CONTINUITY_RUN163_ARTIFACT_INVALID")
        state163 = rdocs[_DOC_ARCHIVE_ANCHOR_STATE]
        if (
            state163.get("sequence") != sequence
            or state163.get("anchorConsensusHeadSha256")
            != event["run163AnchorConsensusHeadSha256"]
        ):
            _fail("ARCHIVE_MERKLE_CONTINUITY_RUN163_BINDING_INVALID")
        leaf_doc = merkle._leaf_document(rdocs, rraws)
        leaf_hash = merkle.merkle_leaf_hash(_canonical(leaf_doc))
        if event["leafHash"] != leaf_hash:
            _fail("ARCHIVE_MERKLE_CONTINUITY_LEAF_INVALID")
        log_challenge = _log_challenge(
            sequence=sequence,
            run163_head=event["run163AnchorConsensusHeadSha256"],
            leaf_hash=leaf_hash,
            previous_head=previous_head,
            authority_head=run165["state"]["logAuthorityChainHeadSha256"],
            active_authority_sha=authority_sha,
        )
        if event["logChallenge"] != log_challenge:
            _fail("ARCHIVE_MERKLE_CONTINUITY_LOG_CHALLENGE_INVALID")
        log_docs = rec["logResponses"]
        if not isinstance(log_docs, dict) or set(log_docs) != set(run165["authority"]):
            _fail("ARCHIVE_MERKLE_CONTINUITY_LOG_RESPONSE_SET_INVALID")
        rows = []
        next_by_log = {}
        for lid in sorted(run165["authority"]):
            parsed = _normalize_merkle_error(
                "ARCHIVE_MERKLE_CONTINUITY_CHECKPOINT_INVALID",
                merkle._verify_checkpoint,
                log_docs[lid],
                log_id=lid,
                config=run165["authority"][lid],
                sequence=sequence,
                challenge=log_challenge,
                expected_leaf_hash=leaf_hash,
                previous=previous_by_log[lid],
                now=now,
                creation=False,
                historical=True,
            )
            row = {
                k: parsed[k]
                for k in (
                    "logId",
                    "checkpointSha256",
                    "treeSize",
                    "rootHash",
                    "leafIndex",
                    "leafHash",
                )
            }
            rows.append(row)
            next_by_log[lid] = row
        if event["logs"] != rows:
            _fail("ARCHIVE_MERKLE_CONTINUITY_LOG_ROWS_INVALID")
        checkpoints = {row["logId"]: row["checkpointSha256"] for row in rows}
        gossip_challenge = _gossip_challenge(
            sequence=sequence,
            run163_head=event["run163AnchorConsensusHeadSha256"],
            leaf_hash=leaf_hash,
            checkpoints=checkpoints,
            authority_head=run165["state"]["logAuthorityChainHeadSha256"],
            active_authority_sha=authority_sha,
        )
        if event["gossipChallenge"] != gossip_challenge:
            _fail("ARCHIVE_MERKLE_CONTINUITY_GOSSIP_CHALLENGE_INVALID")
        gossip_docs = rec["gossipResponses"]
        if not isinstance(gossip_docs, dict) or set(gossip_docs) != set(
            run165["authority"]
        ):
            _fail("ARCHIVE_MERKLE_CONTINUITY_GOSSIP_RESPONSE_SET_INVALID")
        gossip_hashes = []
        for lid in sorted(run165["authority"]):
            parsed = _normalize_merkle_error(
                "ARCHIVE_MERKLE_CONTINUITY_GOSSIP_INVALID",
                merkle._verify_gossip,
                gossip_docs[lid],
                log_id=lid,
                config=run165["authority"][lid],
                sequence=sequence,
                challenge=gossip_challenge,
                run163_head=event["run163AnchorConsensusHeadSha256"],
                checkpoints=checkpoints,
                now=now,
                creation=False,
                historical=True,
            )
            gossip_hashes.append(parsed["gossipSha256"])
        if event["gossipResponseSha256s"] != gossip_hashes:
            _fail("ARCHIVE_MERKLE_CONTINUITY_GOSSIP_HASHES_INVALID")
        bare = {
            k: event[k] for k in event if k != "merkleAuthorityContinuityHeadSha256"
        }
        head = _event_head(previous_head, bare)
        if event["merkleAuthorityContinuityHeadSha256"] != head:
            _fail("ARCHIVE_MERKLE_CONTINUITY_CHAIN_HEAD_INVALID")
        previous_head = head
        previous_by_log = next_by_log
        last = event
    if current_run163_docs is not None:
        latest_docs = receipt["events"][-1]["run163Documents"]
        if (
            latest_docs != current_run163_docs
            or {n: _canonical(latest_docs[n]) for n in sorted(latest_docs)}
            != current_run163_raws
        ):
            _fail("ARCHIVE_MERKLE_CONTINUITY_CURRENT_RUN163_MISMATCH")
    return {
        "sequence": base_seq + len(events),
        "postHandoffSequence": len(events),
        "head": previous_head,
        "last": last,
        "checkpoints": previous_by_log,
    }


def verify_merkle_authority_continuity(  # ruff: ignore[undocumented-public-function]
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
    run162_dir: Path,
    anchor_plan_path: Path,
    run163_dir: Path,
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    governance_root_path: Path,
    governance_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    run165_dir: Path,
    output_dir: Path,
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run164 = _load_run164(
        run164_dir, transparency_root_path, transparency_root_pin, now=current
    )
    run165 = _load_run165(
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        run165_dir=run165_dir,
        now=current,
        historical=historical,
    )
    _enforce_external_separation(
        run165["authority"], witness_root_path, anchor_plan_path
    )
    current163 = _load_run163_current(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        witness_root_path=witness_root_path,
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        run162_dir=run162_dir,
        anchor_plan_path=anchor_plan_path,
        run163_dir=run163_dir,
        now=current,
        historical=historical,
    )
    docs, raws = _load_output(output_dir)
    replay = _replay(
        run164=run164,
        run165=run165,
        bundle=docs["release-archive-merkle-continuity-bundle.json"],
        receipt=docs["release-archive-merkle-continuity-receipt.json"],
        now=current,
        historical=historical,
        current_run163_docs=current163["docs"],
        current_run163_raws=current163["raws"],
    )
    bundle_raw = raws["release-archive-merkle-continuity-bundle.json"]
    last = replay["last"]
    expected_state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-archive-merkle-authority-continuity",
        "sequence": replay["sequence"],
        "postHandoffSequence": replay["postHandoffSequence"],
        "baseRun164Sequence": run164["replay"]["sequence"],
        "run165Sequence": run165["state"]["sequence"],
        "run165LogAuthorityChainHeadSha256": run165["state"][
            "logAuthorityChainHeadSha256"
        ],
        "activeAuthoritySha256": _sha_bytes(_canonical(run165["authority"])),
        "merkleAuthorityContinuityHeadSha256": replay["head"],
        "lastRun163AnchorConsensusHeadSha256": last["run163AnchorConsensusHeadSha256"],
        "lastCheckpoints": {
            lid: {
                k: replay["checkpoints"][lid][k]
                for k in ("checkpointSha256", "treeSize", "rootHash")
            }
            for lid in sorted(replay["checkpoints"])
        },
        "bundleArtifact": {
            "name": "release-archive-merkle-continuity-bundle.json",
            "sha256": _sha_bytes(bundle_raw),
            "size": len(bundle_raw),
        },
    }
    if docs[_DOC_ARCHIVE_MERKLE_CONTINUITY_STATE] != expected_state:
        _fail("ARCHIVE_MERKLE_CONTINUITY_STATE_MISMATCH")
    expected_active = {
        "schemaVersion": int(POLICY["active_schema_version"]),
        "status": "active-archive-merkle-authority-continuity",
        "sequence": replay["sequence"],
        "postHandoffSequence": replay["postHandoffSequence"],
        "run165Sequence": run165["state"]["sequence"],
        "run165LogAuthorityChainHeadSha256": run165["state"][
            "logAuthorityChainHeadSha256"
        ],
        "authority": run165["authority"],
        "merkleAuthorityContinuityHeadSha256": replay["head"],
        "run163AnchorConsensusHeadSha256": last["run163AnchorConsensusHeadSha256"],
        "leafHash": last["leafHash"],
        "logs": last["logs"],
        "gossipResponseSha256s": last["gossipResponseSha256s"],
    }
    if docs["active-archive-merkle-continuity.json"] != expected_active:
        _fail("ARCHIVE_MERKLE_CONTINUITY_ACTIVE_MISMATCH")
    if not historical:
        # Accepted history uses historical signature semantics, while the newest epoch stays live.
        latest = docs["release-archive-merkle-continuity-receipt.json"]["events"][-1]
        for lid in sorted(run165["authority"]):
            integrated = merkle._dt(
                latest["logResponses"][lid]["signed"]["integratedAt"],
                "ARCHIVE_MERKLE_CONTINUITY_ACTIVE_TIME_INVALID",
            )
            observed = merkle._dt(
                latest["gossipResponses"][lid]["signed"]["observedAt"],
                "ARCHIVE_MERKLE_CONTINUITY_ACTIVE_TIME_INVALID",
            )
            max_age = timedelta(days=int(POLICY["max_active_epoch_age_days"]))
            if current - integrated > max_age or current - observed > max_age:
                _fail("ARCHIVE_MERKLE_CONTINUITY_ACTIVE_EPOCH_STALE")
    return {
        "ok": True,
        "sequence": replay["sequence"],
        "post_handoff_sequence": replay["postHandoffSequence"],
        "merkle_authority_continuity_head_sha256": replay["head"],
    }


def continue_merkle_authority(  # ruff: ignore[too-many-branches, undocumented-public-function]
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
    run162_dir: Path,
    anchor_plan_path: Path,
    run163_dir: Path,
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    governance_root_path: Path,
    governance_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    run165_dir: Path,
    output_dir: Path,
    adapters: list[
        tuple[
            str,
            Callable[[dict[str, Any]], dict[str, Any]],
            Callable[[dict[str, Any]], dict[str, Any]],
        ]
    ],
    previous_output_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    authority_paths = [
        Path(run160_dir),
        Path(run161_dir),
        Path(retention_root_path),
        Path(witness_root_path),
        Path(run162_dir),
        Path(anchor_plan_path),
        Path(run163_dir),
        Path(run164_dir),
        Path(transparency_root_path),
        Path(governance_root_path),
        Path(recovery_root_path),
        Path(run165_dir),
    ]
    if previous_output_dir is not None:
        authority_paths.append(Path(previous_output_dir))
    before = _authority_fingerprint(
        authority_paths, "ARCHIVE_MERKLE_CONTINUITY_INPUT_DRIFT"
    )
    run164 = _load_run164(
        run164_dir, transparency_root_path, transparency_root_pin, now=current
    )
    run165 = _load_run165(
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        run165_dir=run165_dir,
        now=current,
        historical=False,
    )
    _enforce_external_separation(
        run165["authority"], witness_root_path, anchor_plan_path
    )
    current163 = _load_run163_current(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        witness_root_path=witness_root_path,
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        run162_dir=run162_dir,
        anchor_plan_path=anchor_plan_path,
        run163_dir=run163_dir,
        now=current,
        historical=False,
    )
    state163 = current163["docs"][_DOC_ARCHIVE_ANCHOR_STATE]
    sequence = state163.get("sequence")
    base_seq = run164["replay"]["sequence"]
    if not isinstance(sequence, int) or sequence <= base_seq:
        _fail("ARCHIVE_MERKLE_CONTINUITY_SEQUENCE_INVALID")
    old_events = []
    old_receipts = []
    previous_head = run164["replay"]["merkleConsensusHeadSha256"]
    previous_by_log = _base_checkpoints(run165)
    if previous_output_dir is None:
        if sequence != base_seq + 1:
            _fail("ARCHIVE_MERKLE_CONTINUITY_PREVIOUS_OUTPUT_REQUIRED")
    else:
        pdocs, _ = _load_output(previous_output_dir)
        state_prev = pdocs[_DOC_ARCHIVE_MERKLE_CONTINUITY_STATE]
        if state_prev.get("run165LogAuthorityChainHeadSha256") != run165["state"][
            "logAuthorityChainHeadSha256"
        ] or state_prev.get("activeAuthoritySha256") != _sha_bytes(
            _canonical(run165["authority"])
        ):
            _fail("ARCHIVE_MERKLE_CONTINUITY_AUTHORITY_CHANGED_REQUIRES_NEW_BRIDGE")
        replay = _replay(
            run164=run164,
            run165=run165,
            bundle=pdocs["release-archive-merkle-continuity-bundle.json"],
            receipt=pdocs["release-archive-merkle-continuity-receipt.json"],
            now=current,
            historical=True,
            current_run163_docs=None,
            current_run163_raws=None,
        )
        if sequence != replay["sequence"] + 1:
            _fail("ARCHIVE_MERKLE_CONTINUITY_SEQUENCE_INVALID")
        previous_head = replay["head"]
        previous_by_log = replay["checkpoints"]
        old_events = list(
            pdocs["release-archive-merkle-continuity-bundle.json"]["events"]
        )
        old_receipts = list(
            pdocs["release-archive-merkle-continuity-receipt.json"]["events"]
        )
    if {lid for lid, _, _ in adapters} != set(run165["authority"]) or len(
        adapters
    ) != len(run165["authority"]):
        _fail("ARCHIVE_MERKLE_CONTINUITY_ADAPTER_SET_INVALID")
    amap = {lid: (log, gossip) for lid, log, gossip in adapters}
    leaf_doc = merkle._leaf_document(current163["docs"], current163["raws"])
    leaf_hash = merkle.merkle_leaf_hash(_canonical(leaf_doc))
    authority_sha = _sha_bytes(_canonical(run165["authority"]))
    log_challenge = _log_challenge(
        sequence=sequence,
        run163_head=state163["anchorConsensusHeadSha256"],
        leaf_hash=leaf_hash,
        previous_head=previous_head,
        authority_head=run165["state"]["logAuthorityChainHeadSha256"],
        active_authority_sha=authority_sha,
    )
    log_docs = {}
    rows = []
    next_by_log = {}
    for lid in sorted(run165["authority"]):
        prev = previous_by_log[lid]
        request = {
            "operation": "append-merkle-leaf-authority-continuity",
            "protocolVersion": 1,
            "logId": lid,
            "sequence": sequence,
            "challenge": log_challenge,
            "leaf": leaf_doc,
            "leafHash": leaf_hash,
            "previousTreeSize": prev["treeSize"],
            "previousRootHash": prev["rootHash"],
            "run165LogAuthorityChainHeadSha256": run165["state"][
                "logAuthorityChainHeadSha256"
            ],
            "activeAuthoritySha256": authority_sha,
        }
        doc = amap[lid][0](request)
        parsed = _normalize_merkle_error(
            "ARCHIVE_MERKLE_CONTINUITY_CHECKPOINT_INVALID",
            merkle._verify_checkpoint,
            doc,
            log_id=lid,
            config=run165["authority"][lid],
            sequence=sequence,
            challenge=log_challenge,
            expected_leaf_hash=leaf_hash,
            previous=prev,
            now=current,
            creation=True,
            historical=False,
        )
        row = {
            k: parsed[k]
            for k in (
                "logId",
                "checkpointSha256",
                "treeSize",
                "rootHash",
                "leafIndex",
                "leafHash",
            )
        }
        log_docs[lid] = doc
        rows.append(row)
        next_by_log[lid] = row
    checkpoints = {row["logId"]: row["checkpointSha256"] for row in rows}
    gossip_challenge = _gossip_challenge(
        sequence=sequence,
        run163_head=state163["anchorConsensusHeadSha256"],
        leaf_hash=leaf_hash,
        checkpoints=checkpoints,
        authority_head=run165["state"]["logAuthorityChainHeadSha256"],
        active_authority_sha=authority_sha,
    )
    gossip_docs = {}
    gossip_hashes = []
    for lid in sorted(run165["authority"]):
        request = {
            "operation": "gossip-merkle-authority-continuity",
            "protocolVersion": 1,
            "sequence": sequence,
            "challenge": gossip_challenge,
            "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
            "leafHash": leaf_hash,
            "checkpointSha256s": checkpoints,
            "run165LogAuthorityChainHeadSha256": run165["state"][
                "logAuthorityChainHeadSha256"
            ],
            "activeAuthoritySha256": authority_sha,
        }
        doc = amap[lid][1](request)
        parsed = _normalize_merkle_error(
            "ARCHIVE_MERKLE_CONTINUITY_GOSSIP_INVALID",
            merkle._verify_gossip,
            doc,
            log_id=lid,
            config=run165["authority"][lid],
            sequence=sequence,
            challenge=gossip_challenge,
            run163_head=state163["anchorConsensusHeadSha256"],
            checkpoints=checkpoints,
            now=current,
            creation=True,
            historical=False,
        )
        gossip_docs[lid] = doc
        gossip_hashes.append(parsed["gossipSha256"])
    post_seq = sequence - base_seq
    event_bare = {
        "sequence": sequence,
        "postHandoffSequence": post_seq,
        "run163Sequence": sequence,
        "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
        "run163Artifacts": _artifact_map(current163["raws"]),
        "leafHash": leaf_hash,
        "run165Sequence": run165["state"]["sequence"],
        "run165LogAuthorityChainHeadSha256": run165["state"][
            "logAuthorityChainHeadSha256"
        ],
        "activeAuthoritySha256": authority_sha,
        "logs": rows,
        "logChallenge": log_challenge,
        "gossipChallenge": gossip_challenge,
        "gossipResponseSha256s": gossip_hashes,
    }
    head = _event_head(previous_head, event_bare)
    event = dict(event_bare)
    event["merkleAuthorityContinuityHeadSha256"] = head
    receipt_event = {
        "sequence": sequence,
        "run163Documents": current163["docs"],
        "logResponses": log_docs,
        "gossipResponses": gossip_docs,
    }
    bundle = {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "archive-merkle-authority-continuity-history",
        "run164Artifacts": run164["artifacts"],
        "run165Artifacts": run165["artifacts"],
        "baseRun164Sequence": base_seq,
        "baseMerkleConsensusHeadSha256": run164["replay"]["merkleConsensusHeadSha256"],
        "run165Sequence": run165["state"]["sequence"],
        "run165LogAuthorityChainHeadSha256": run165["state"][
            "logAuthorityChainHeadSha256"
        ],
        "activeAuthoritySha256": authority_sha,
        "events": [*old_events, event],
    }
    receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": "archive-merkle-authority-continuity-accepted",
        "run164Documents": run164["docs"],
        "run165Documents": run165["docs"],
        "events": [*old_receipts, receipt_event],
    }
    bundle_raw = _canonical(bundle)
    state_doc = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-archive-merkle-authority-continuity",
        "sequence": sequence,
        "postHandoffSequence": post_seq,
        "baseRun164Sequence": base_seq,
        "run165Sequence": run165["state"]["sequence"],
        "run165LogAuthorityChainHeadSha256": run165["state"][
            "logAuthorityChainHeadSha256"
        ],
        "activeAuthoritySha256": authority_sha,
        "merkleAuthorityContinuityHeadSha256": head,
        "lastRun163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
        "lastCheckpoints": {
            lid: {
                k: next_by_log[lid][k]
                for k in ("checkpointSha256", "treeSize", "rootHash")
            }
            for lid in sorted(next_by_log)
        },
        "bundleArtifact": {
            "name": "release-archive-merkle-continuity-bundle.json",
            "sha256": _sha_bytes(bundle_raw),
            "size": len(bundle_raw),
        },
    }
    active = {
        "schemaVersion": int(POLICY["active_schema_version"]),
        "status": "active-archive-merkle-authority-continuity",
        "sequence": sequence,
        "postHandoffSequence": post_seq,
        "run165Sequence": run165["state"]["sequence"],
        "run165LogAuthorityChainHeadSha256": run165["state"][
            "logAuthorityChainHeadSha256"
        ],
        "authority": run165["authority"],
        "merkleAuthorityContinuityHeadSha256": head,
        "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
        "leafHash": leaf_hash,
        "logs": rows,
        "gossipResponseSha256s": gossip_hashes,
    }
    if before != _authority_fingerprint(
        authority_paths, "ARCHIVE_MERKLE_CONTINUITY_INPUT_DRIFT"
    ):
        _fail("ARCHIVE_MERKLE_CONTINUITY_INPUT_DRIFT")
    protected = list(authority_paths)
    target = _outside(
        output_dir, protected, "ARCHIVE_MERKLE_CONTINUITY_OUTPUT_OVERLAPS_INPUT"
    )
    if target.exists():
        _fail("ARCHIVE_MERKLE_CONTINUITY_OUTPUT_EXISTS")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".run166-merkle-continuity-", dir=target.parent)
    )
    try:
        (stage / "release-archive-merkle-continuity-bundle.json").write_bytes(
            bundle_raw
        )
        _write(stage / _DOC_ARCHIVE_MERKLE_CONTINUITY_STATE, state_doc)
        _write(stage / "active-archive-merkle-continuity.json", active)
        _write(stage / "release-archive-merkle-continuity-receipt.json", receipt)
        verify_merkle_authority_continuity(
            run160_dir=run160_dir,
            run161_dir=run161_dir,
            retention_root_path=retention_root_path,
            retention_root_pin=retention_root_pin,
            witness_root_path=witness_root_path,
            witness_root_pin=witness_root_pin,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            run162_dir=run162_dir,
            anchor_plan_path=anchor_plan_path,
            run163_dir=run163_dir,
            run164_dir=run164_dir,
            transparency_root_path=transparency_root_path,
            transparency_root_pin=transparency_root_pin,
            governance_root_path=governance_root_path,
            governance_root_pin=governance_root_pin,
            recovery_root_path=recovery_root_path,
            recovery_root_pin=recovery_root_pin,
            run165_dir=run165_dir,
            output_dir=stage,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_merkle_authority_continuity(
        run160_dir=run160_dir,
        run161_dir=run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        witness_root_path=witness_root_path,
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        run162_dir=run162_dir,
        anchor_plan_path=anchor_plan_path,
        run163_dir=run163_dir,
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        run165_dir=run165_dir,
        output_dir=target,
        now=current,
    )


def command_log(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return merkle.command_log(command)


def command_gossip(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return merkle.command_gossip(command)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run160-dir", type=Path, required=True)
    parser.add_argument("--run161-dir", type=Path, required=True)
    parser.add_argument("--retention-root-path", type=Path, required=True)
    parser.add_argument("--retention-root-pin", required=True)
    parser.add_argument("--witness-root-path", type=Path, required=True)
    parser.add_argument("--witness-root-pin", required=True)
    parser.add_argument("--bootstrap-pin", required=True)
    parser.add_argument("--recovery-pin", required=True)
    parser.add_argument(
        "--attestation-pin", action="append", dest="attestation_pins", required=True
    )
    parser.add_argument("--run162-dir", type=Path, required=True)
    parser.add_argument("--anchor-plan-path", type=Path, required=True)
    parser.add_argument("--run163-dir", type=Path, required=True)
    parser.add_argument("--run164-dir", type=Path, required=True)
    parser.add_argument("--transparency-root-path", type=Path, required=True)
    parser.add_argument("--transparency-root-pin", required=True)
    parser.add_argument("--governance-root-path", type=Path, required=True)
    parser.add_argument("--governance-root-pin", required=True)
    parser.add_argument("--recovery-root-path", type=Path, required=True)
    parser.add_argument("--recovery-root-pin", required=True)
    parser.add_argument("--run165-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def main(  # ruff: ignore[undocumented-public-function]
    argv=None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    p = subs.add_parser("verify")
    _common(p)
    p.add_argument("--historical", action="store_true")
    a = subs.add_parser("append")
    _common(a)
    a.add_argument("--previous-output-dir", type=Path)
    a.add_argument("--log-command", action="append", required=True)
    a.add_argument("--gossip-command", action="append", required=True)
    ns = parser.parse_args(argv)
    kwargs = vars(ns)
    cmd = kwargs.pop("cmd")
    if cmd == "verify":
        historical = kwargs.pop("historical")
        result = verify_merkle_authority_continuity(**kwargs, historical=historical)
    else:
        previous = kwargs.pop("previous_output_dir")
        log_cmds = kwargs.pop("log_command")
        gossip_cmds = kwargs.pop("gossip_command")
        if len(log_cmds) != len(gossip_cmds):
            _fail("ARCHIVE_MERKLE_CONTINUITY_COMMAND_COUNT_INVALID")

        # CLI format: LOG_ID::program arg... ; one command per log and gossip side.
        def parse(items, builder):
            out = {}
            for raw in items:
                if "::" not in raw:
                    _fail("ARCHIVE_MERKLE_CONTINUITY_COMMAND_INVALID")
                lid, command = raw.split("::", 1)
                out[lid] = builder(command.split())
            return out

        logs = parse(log_cmds, command_log)
        goss = parse(gossip_cmds, command_gossip)
        if set(logs) != set(goss):
            _fail("ARCHIVE_MERKLE_CONTINUITY_COMMAND_SET_INVALID")
        adapters = [(lid, logs[lid], goss[lid]) for lid in sorted(logs)]
        result = continue_merkle_authority(
            **kwargs, adapters=adapters, previous_output_dir=previous
        )
    logger.info("Merkle authority %s completed", cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
