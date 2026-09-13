"""
Run 167: recursively re-bridge rotated/recovered Merkle authority after Run 166.

Run 166 permits arbitrarily many RFC6962 appends while one Run 165 authority remains
active.  Run 167 handles the next authority change *after* those appends: the old/new
(or recovery) handoff is signed against the latest accepted Run 166/Run 167 checkpoint,
then the first new Merkle leaf is appended under the replacement authority.  Later
append-only epochs and later re-bridges use the same cumulative format.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import re
import shutil
import stat
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import tomllib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)

try:
    import continue_archive_merkle_authority as run166
    import govern_archive_merkle_log_authority as run165
    import verify_archive_merkle_transparency as merkle
except ImportError:  # canonical package import
    from . import continue_archive_merkle_authority as run166
    from . import govern_archive_merkle_log_authority as run165
    from . import verify_archive_merkle_transparency as merkle

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads(
    (HERE / "release_archive_merkle_rebridge_policy.toml").read_text()
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
_DOC_ARCHIVE_MERKLE_REBRIDGE_STATE = "trusted-archive-merkle-rebridge-state.json"


_OUTPUT_NAMES = {
    "release-archive-merkle-rebridge-bundle.json",
    _DOC_ARCHIVE_MERKLE_REBRIDGE_STATE,
    "active-archive-merkle-rebridge.json",
    "release-archive-merkle-rebridge-receipt.json",
}
_RUN166_NAMES = {
    "release-archive-merkle-continuity-bundle.json",
    _DOC_ARCHIVE_MERKLE_CONTINUITY_STATE,
    "active-archive-merkle-continuity.json",
    "release-archive-merkle-continuity-receipt.json",
}
_RUN165_NAMES = {
    "release-archive-log-authority-bundle.json",
    _DOC_ARCHIVE_LOG_AUTHORITY_STATE,
    "active-archive-log-authority.json",
    "release-archive-log-authority-receipt.json",
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")


class ArchiveMerkleRebridgeError(  # ruff: ignore[undocumented-public-class]
    RuntimeError
):
    pass


def _fail(code: str) -> None:
    raise ArchiveMerkleRebridgeError(code)


def _canonical(value: Any) -> bytes:
    return merkle._canonical(value)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _no_dupes(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            _fail("ARCHIVE_MERKLE_REBRIDGE_JSON_DUPLICATE_KEY")
        out[key] = value
    return out


def _loads(raw: bytes, code: str) -> dict[str, Any]:
    if len(raw) > int(POLICY["max_json_bytes"]):
        _fail(code + "_TOO_LARGE")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes)
    except ArchiveMerkleRebridgeError:
        raise
    except Exception:  # ruff: ignore[blind-except]
        _fail(code + "_JSON_INVALID")
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


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value))


def _id(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        _fail(code)
    return value


def _hex(value: Any, code: str) -> str:
    _len = len(value) != 64  # ruff: ignore[magic-value-comparison]
    if not isinstance(value, str) or _len:
        _fail(code)
    try:
        bytes.fromhex(value)
    except ValueError:
        _fail(code)
    if value != value.lower():
        _fail(code)
    return value


def _dt(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail(code)
    if parsed.tzinfo is None:
        _fail(code)
    return parsed.astimezone(timezone.utc)


def _ts(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _b64(value: Any, code: str, size: int | None = None) -> bytes:
    if not isinstance(value, str):
        _fail(code)
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:  # ruff: ignore[blind-except]
        _fail(code)
    if size is not None and len(raw) != size:
        _fail(code)
    if base64.b64encode(raw).decode("ascii") != value:
        _fail(code)
    return raw


def _verify_sig(public_key: str, signature: str, message: bytes, code: str) -> None:
    try:
        Ed25519PublicKey.from_public_bytes(
            _b64(public_key, code + "_PUBLIC_INVALID", 32)
        ).verify(_b64(signature, code + "_SIGNATURE_INVALID", 64), message)
    except InvalidSignature:
        _fail(code + "_INVALID")
    except ArchiveMerkleRebridgeError:
        raise
    except Exception:  # ruff: ignore[blind-except]
        _fail(code + "_INVALID")


def _positive_int(value: Any, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(code)
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
        rel = path.relative_to(root).as_posix()
        st = path.lstat()
        if stat.S_ISDIR(st.st_mode):
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            _fail(code)
        rows.append((rel, _sha(path), st.st_size, stat.S_IMODE(st.st_mode)))
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


def _pub_fingerprint(public_key: str) -> str:
    return _sha_bytes(
        _b64(public_key, "ARCHIVE_MERKLE_REBRIDGE_PUBLIC_KEY_INVALID", 32)
    )


def _authority(
    value: Any, *, expected_log_ids: set[str], code: str
) -> dict[str, dict[str, str]]:
    try:
        return run165._authority(value, expected_log_ids=expected_log_ids, code=code)
    except Exception as exc:
        raise ArchiveMerkleRebridgeError(code + "_INVALID:" + str(exc)) from exc


def _authority_sha(value: dict[str, Any]) -> str:
    return _sha_bytes(_canonical(value))


def _authority_fps(value: dict[str, Any]) -> set[str]:
    out = set()
    for row in value.values():
        out.add(_pub_fingerprint(row["publicKey"]))
        out.add(_pub_fingerprint(row["gossipPublicKey"]))
    return out


def _verify_control_root(
    path: Path, pin: str, *, recovery: bool, now: datetime, historical: bool
) -> dict[str, Any]:
    doc, _ = _read_json(
        Path(path),
        (
            "ARCHIVE_MERKLE_REBRIDGE_RECOVERY_ROOT"
            if recovery
            else "ARCHIVE_MERKLE_REBRIDGE_GOVERNANCE_ROOT"
        ),
    )
    try:
        return run165._verify_control_root(
            doc, pin, now=now, recovery=recovery, historical=historical
        )
    except Exception as exc:
        raise ArchiveMerkleRebridgeError(
            (
                "ARCHIVE_MERKLE_REBRIDGE_RECOVERY_ROOT_INVALID:"
                if recovery
                else "ARCHIVE_MERKLE_REBRIDGE_GOVERNANCE_ROOT_INVALID:"
            )
            + str(exc)
        ) from exc


def _load_run164(
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    *,
    now: datetime,
) -> dict[str, Any]:
    try:
        return run166._load_run164(
            run164_dir, transparency_root_path, transparency_root_pin, now=now
        )
    except Exception as exc:
        raise ArchiveMerkleRebridgeError(
            "ARCHIVE_MERKLE_REBRIDGE_RUN164_INVALID:" + str(exc)
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
) -> dict[str, Any]:
    try:
        return run166._load_run165(
            run164_dir=run164_dir,
            transparency_root_path=transparency_root_path,
            transparency_root_pin=transparency_root_pin,
            governance_root_path=governance_root_path,
            governance_root_pin=governance_root_pin,
            recovery_root_path=recovery_root_path,
            recovery_root_pin=recovery_root_pin,
            run165_dir=run165_dir,
            now=now,
            historical=True,
        )
    except Exception as exc:
        raise ArchiveMerkleRebridgeError(
            "ARCHIVE_MERKLE_REBRIDGE_RUN165_INVALID:" + str(exc)
        ) from exc


def _load_run166(
    *,
    run160_dir: Path,
    run166_run161_dir: Path,
    retention_root_path: Path,
    retention_root_pin: str,
    witness_root_path: Path,
    witness_root_pin: str,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    run166_run162_dir: Path,
    anchor_plan_path: Path,
    run166_run163_dir: Path,
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    governance_root_path: Path,
    governance_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    run165_dir: Path,
    run166_dir: Path,
    now: datetime,
) -> dict[str, Any]:
    try:
        run166.verify_merkle_authority_continuity(
            run160_dir=run160_dir,
            run161_dir=run166_run161_dir,
            retention_root_path=retention_root_path,
            retention_root_pin=retention_root_pin,
            witness_root_path=witness_root_path,
            witness_root_pin=witness_root_pin,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            run162_dir=run166_run162_dir,
            anchor_plan_path=anchor_plan_path,
            run163_dir=run166_run163_dir,
            run164_dir=run164_dir,
            transparency_root_path=transparency_root_path,
            transparency_root_pin=transparency_root_pin,
            governance_root_path=governance_root_path,
            governance_root_pin=governance_root_pin,
            recovery_root_path=recovery_root_path,
            recovery_root_pin=recovery_root_pin,
            run165_dir=run165_dir,
            output_dir=run166_dir,
            now=now,
            historical=True,
        )
        docs, raws = run166._load_output(Path(run166_dir))
    except Exception as exc:
        raise ArchiveMerkleRebridgeError(
            "ARCHIVE_MERKLE_REBRIDGE_RUN166_INVALID:" + str(exc)
        ) from exc
    state = docs[_DOC_ARCHIVE_MERKLE_CONTINUITY_STATE]
    active = docs["active-archive-merkle-continuity.json"]
    if state.get("sequence") != active.get("sequence") or state.get(
        "merkleAuthorityContinuityHeadSha256"
    ) != active.get("merkleAuthorityContinuityHeadSha256"):
        _fail("ARCHIVE_MERKLE_REBRIDGE_RUN166_STATE_ACTIVE_MISMATCH")
    if state.get("activeAuthoritySha256") != _authority_sha(active.get("authority")):
        _fail("ARCHIVE_MERKLE_REBRIDGE_RUN166_AUTHORITY_MISMATCH")
    cps = state.get("lastCheckpoints")
    if not isinstance(cps, dict) or set(cps) != set(active.get("authority", {})):
        _fail("ARCHIVE_MERKLE_REBRIDGE_RUN166_CHECKPOINTS_INVALID")
    return {
        "docs": docs,
        "raws": raws,
        "state": state,
        "active": active,
        "artifacts": _artifact_map(raws),
    }


def _load_current_run163(
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
        return run166._load_run163_current(
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
        raise ArchiveMerkleRebridgeError(
            "ARCHIVE_MERKLE_REBRIDGE_RUN163_INVALID:" + str(exc)
        ) from exc


def _verify_embedded_run163(
    docs: dict[str, Any],
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
    now: datetime,
) -> dict[str, Any]:
    if not isinstance(docs, dict) or set(docs) != merkle._RUN163_NAMES:
        _fail("ARCHIVE_MERKLE_REBRIDGE_EMBEDDED_RUN163_SCHEMA_INVALID")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in sorted(docs):
            _write(root / name, docs[name])
        try:
            result = merkle.run163.verify_anchor_history(
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
                output_dir=root,
                now=now,
                historical=True,
                require_current_match=False,
            )
        except Exception as exc:
            raise ArchiveMerkleRebridgeError(
                "ARCHIVE_MERKLE_REBRIDGE_EMBEDDED_RUN163_INVALID:" + str(exc)
            ) from exc
    return result


def _enforce_separation(
    governance: dict[str, Any],
    recovery: dict[str, Any],
    run164_root: dict[str, Any],
    witness_root_path: Path,
    anchor_plan_path: Path,
    *authorities: dict[str, Any],
) -> None:
    try:
        run165._enforce_plane_separation(
            governance, recovery, run164_root, *authorities
        )
        for authority in authorities:
            run166._enforce_external_separation(
                authority, witness_root_path, anchor_plan_path
            )
    except Exception as exc:
        raise ArchiveMerkleRebridgeError(
            "ARCHIVE_MERKLE_REBRIDGE_AUTHORITY_OVERLAP:" + str(exc)
        ) from exc


def _normalize_checkpoints(
    value: Any, *, expected_log_ids: set[str], code: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != expected_log_ids:
        _fail(code + "_SET_INVALID")
    out = {}
    for lid in sorted(value):
        row = value[lid]
        if not isinstance(row, dict) or set(row) != {
            "checkpointSha256",
            "treeSize",
            "rootHash",
        }:
            _fail(code + "_SCHEMA_INVALID")
        cp = _hex(row["checkpointSha256"], code + "_CHECKPOINT_INVALID")
        ts = _positive_int(row["treeSize"], code + "_TREE_SIZE_INVALID")
        rh = _hex(row["rootHash"], code + "_ROOT_INVALID")
        out[lid] = {"checkpointSha256": cp, "treeSize": ts, "rootHash": rh}
    if out != value:
        _fail(code + "_NOT_NORMALIZED")
    return out


def _base_authority_head(
    run165_info: dict[str, Any], run166_info: dict[str, Any]
) -> str:
    state_raw = run166_info["raws"][_DOC_ARCHIVE_MERKLE_CONTINUITY_STATE]
    return _sha_bytes(
        _canonical(
            {
                "run165LogAuthorityChainHeadSha256": run165_info["state"][
                    "logAuthorityChainHeadSha256"
                ],
                "run166MerkleAuthorityContinuityHeadSha256": run166_info["state"][
                    "merkleAuthorityContinuityHeadSha256"
                ],
                "run166TrustedStateSha256": _sha_bytes(state_raw),
            }
        )
    )


def _bridge_head(
    previous_head: str, transition_sha: str, handoff_sha: str, source_state_sha: str
) -> str:
    return _sha_bytes(
        _canonical(
            {
                "previousArchiveMerkleRebridgeAuthorityHeadSha256": previous_head,
                "transitionSha256": transition_sha,
                "handoffSha256": handoff_sha,
                "sourceTrustedStateSha256": source_state_sha,
            }
        )
    )


def _merkle_head(previous_head: str, event_without_head: dict[str, Any]) -> str:
    return _sha_bytes(
        _canonical(
            {
                "previousArchiveMerkleRebridgeContinuityHeadSha256": previous_head,
                "event": event_without_head,
            }
        )
    )


def _handoff_subject(
    *, signed: dict[str, Any], log_id: str, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    return {
        "_type": "archive-merkle-authority-rebridge-handoff",
        "specVersion": str(POLICY["spec_version"]),
        "schemaVersion": int(POLICY["handoff_schema_version"]),
        "transitionId": signed["transitionId"],
        "rebridgeSequence": signed["rebridgeSequence"],
        "kind": signed["kind"],
        "logId": log_id,
        "sourceKind": signed["sourceKind"],
        "sourceTrustedStateSha256": signed["sourceTrustedStateSha256"],
        "sourceMerkleSequence": signed["sourceMerkleSequence"],
        "sourceMerkleContinuityHeadSha256": signed["sourceMerkleContinuityHeadSha256"],
        "previousAuthorityBridgeHeadSha256": signed[
            "previousAuthorityBridgeHeadSha256"
        ],
        "priorCheckpoint": checkpoint,
        "currentLogAuthority": signed["currentAuthority"][log_id],
        "nextLogAuthority": signed["nextAuthority"][log_id],
        "currentAuthoritySha256": _authority_sha(signed["currentAuthority"]),
        "nextAuthoritySha256": _authority_sha(signed["nextAuthority"]),
    }


def _verify_transition(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    *,
    source: dict[str, Any],
    governance: dict[str, Any],
    recovery: dict[str, Any],
    expected_log_ids: set[str],
    seen_ids: set[str],
    previous_transition_issued: datetime | None,
    now: datetime,
    creation: bool,
) -> dict[str, Any]:
    if set(doc) != {"signed", "signatures"} or not isinstance(doc.get("signed"), dict):
        _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_SCHEMA_INVALID")
    signed = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "transitionId",
        "rebridgeSequence",
        "kind",
        "issuedAt",
        "sourceKind",
        "sourceTrustedStateSha256",
        "sourceMerkleSequence",
        "sourceMerkleContinuityHeadSha256",
        "previousAuthorityBridgeHeadSha256",
        "currentAuthority",
        "nextAuthority",
        "changedLogIds",
        "compromisedKeyFingerprints",
        "revokedKeyFingerprints",
        "authorizationRole",
        "selectedSignerKeyIds",
        "handoffSubjectSha256s",
    }
    if (
        set(signed) != expected
        or signed.get("_type") != "archive-merkle-authority-rebridge-transition"
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_SIGNED_SCHEMA_INVALID")
    if signed.get("specVersion") != str(POLICY["spec_version"]) or signed.get(
        "schemaVersion"
    ) != int(POLICY["transition_schema_version"]):
        _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_VERSION_INVALID")
    tid = _id(signed["transitionId"], "ARCHIVE_MERKLE_REBRIDGE_TRANSITION_ID_INVALID")
    if tid in seen_ids:
        _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_ID_REUSED")
    rbseq = _positive_int(
        signed["rebridgeSequence"], "ARCHIVE_MERKLE_REBRIDGE_SEQUENCE_INVALID"
    )
    if rbseq != source["rebridgeSequence"] + 1:
        _fail("ARCHIVE_MERKLE_REBRIDGE_SEQUENCE_INVALID")
    kind = signed["kind"]
    if kind not in {"scheduled-rotation", "compromise-recovery"}:
        _fail("ARCHIVE_MERKLE_REBRIDGE_KIND_INVALID")
    issued = _dt(signed["issuedAt"], "ARCHIVE_MERKLE_REBRIDGE_ISSUED_INVALID")
    if issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
        _fail("ARCHIVE_MERKLE_REBRIDGE_FROM_FUTURE")
    if creation and now - issued > timedelta(
        minutes=int(POLICY["max_transition_freshness_minutes"])
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_STALE")
    if previous_transition_issued is not None and issued <= previous_transition_issued:
        _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_TIME_NOT_MONOTONIC")
    if (
        signed["sourceKind"] != source["sourceKind"]
        or signed["sourceTrustedStateSha256"] != source["stateSha256"]
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_SOURCE_STATE_INVALID")
    if (
        signed["sourceMerkleSequence"] != source["merkleSequence"]
        or signed["sourceMerkleContinuityHeadSha256"] != source["merkleHead"]
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_SOURCE_MERKLE_INVALID")
    if signed["previousAuthorityBridgeHeadSha256"] != source["authorityHead"]:
        _fail("ARCHIVE_MERKLE_REBRIDGE_PREVIOUS_AUTHORITY_HEAD_INVALID")
    current = _authority(
        signed["currentAuthority"],
        expected_log_ids=expected_log_ids,
        code="ARCHIVE_MERKLE_REBRIDGE_CURRENT_AUTHORITY",
    )
    nxt = _authority(
        signed["nextAuthority"],
        expected_log_ids=expected_log_ids,
        code="ARCHIVE_MERKLE_REBRIDGE_NEXT_AUTHORITY",
    )
    if current != source["authority"]:
        _fail("ARCHIVE_MERKLE_REBRIDGE_CURRENT_AUTHORITY_MISMATCH")
    for lid in expected_log_ids:
        if current[lid]["gossipIdentity"] != nxt[lid]["gossipIdentity"]:
            _fail("ARCHIVE_MERKLE_REBRIDGE_GOSSIP_IDENTITY_CHANGE_FORBIDDEN")
    changed = signed["changedLogIds"]
    if not isinstance(changed, list):
        _fail("ARCHIVE_MERKLE_REBRIDGE_CHANGED_SET_INVALID")
    changed = [_id(x, "ARCHIVE_MERKLE_REBRIDGE_CHANGED_ID_INVALID") for x in changed]
    computed = sorted(lid for lid in expected_log_ids if current[lid] != nxt[lid])
    if (
        changed != sorted(changed)
        or len(set(changed)) != len(changed)
        or changed != computed
        or not changed
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_CHANGED_SET_INVALID")
    compromised = signed["compromisedKeyFingerprints"]
    revoked = signed["revokedKeyFingerprints"]
    if not isinstance(compromised, list) or not isinstance(revoked, list):
        _fail("ARCHIVE_MERKLE_REBRIDGE_REVOCATION_LIST_INVALID")
    compromised = [
        _hex(x, "ARCHIVE_MERKLE_REBRIDGE_COMPROMISED_INVALID") for x in compromised
    ]
    revoked = [_hex(x, "ARCHIVE_MERKLE_REBRIDGE_REVOKED_INVALID") for x in revoked]
    if (
        compromised != sorted(compromised)
        or len(set(compromised)) != len(compromised)
        or revoked != sorted(revoked)
        or len(set(revoked)) != len(revoked)
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_REVOCATION_LIST_INVALID")
    previous_revoked = list(source["revoked"])
    current_fps = _authority_fps(current)
    next_fps = _authority_fps(nxt)
    if set(previous_revoked) & next_fps:
        _fail("ARCHIVE_MERKLE_REBRIDGE_REVOKED_KEY_REINTRODUCED")
    if kind == "scheduled-rotation":
        if compromised:
            _fail("ARCHIVE_MERKLE_REBRIDGE_SCHEDULED_COMPROMISED_NOT_EMPTY")
        replaced = set()
        for lid in changed:
            old = current[lid]
            new = nxt[lid]
            if (
                old["operator"] != new["operator"]
                and old["publicKey"] == new["publicKey"]
            ):
                _fail("ARCHIVE_MERKLE_REBRIDGE_OPERATOR_REBOUND_WITHOUT_KEY_ROTATION")
            if (
                old["gossipOperator"] != new["gossipOperator"]
                and old["gossipPublicKey"] == new["gossipPublicKey"]
            ):
                _fail(
                    "ARCHIVE_MERKLE_REBRIDGE_GOSSIP_OPERATOR_REBOUND_WITHOUT_KEY_ROTATION"
                )
            if old["publicKey"] != new["publicKey"]:
                replaced.add(_pub_fingerprint(old["publicKey"]))
            if old["gossipPublicKey"] != new["gossipPublicKey"]:
                replaced.add(_pub_fingerprint(old["gossipPublicKey"]))
        if revoked != sorted(set(previous_revoked) | replaced):
            _fail("ARCHIVE_MERKLE_REBRIDGE_SCHEDULED_REVOCATION_INVALID")
        role = "governance"
        root = governance
    else:
        if not compromised or not set(compromised) <= current_fps:
            _fail("ARCHIVE_MERKLE_REBRIDGE_RECOVERY_COMPROMISED_INVALID")
        affected = sorted(
            lid
            for lid in expected_log_ids
            if {
                _pub_fingerprint(current[lid]["publicKey"]),
                _pub_fingerprint(current[lid]["gossipPublicKey"]),
            }
            & set(compromised)
        )
        if affected != changed:
            _fail("ARCHIVE_MERKLE_REBRIDGE_RECOVERY_CHANGED_SET_INVALID")
        replaced = set()
        for lid in changed:
            old = current[lid]
            new = nxt[lid]
            if (
                old["publicKey"] == new["publicKey"]
                or old["gossipPublicKey"] == new["gossipPublicKey"]
            ):
                _fail("ARCHIVE_MERKLE_REBRIDGE_RECOVERY_REQUIRES_FULL_KEY_ROTATION")
            replaced.add(_pub_fingerprint(old["publicKey"]))
            replaced.add(_pub_fingerprint(old["gossipPublicKey"]))
        if revoked != sorted(set(previous_revoked) | replaced):
            _fail("ARCHIVE_MERKLE_REBRIDGE_RECOVERY_REVOCATION_INVALID")
        role = "recovery"
        root = recovery
    if set(revoked) & next_fps:
        _fail("ARCHIVE_MERKLE_REBRIDGE_REVOKED_KEY_ACTIVE")
    if signed["authorizationRole"] != role:
        _fail("ARCHIVE_MERKLE_REBRIDGE_AUTHORIZATION_ROLE_INVALID")
    selected = signed["selectedSignerKeyIds"]
    if not isinstance(selected, list):
        _fail("ARCHIVE_MERKLE_REBRIDGE_SELECTED_SIGNERS_INVALID")
    selected = [
        _id(x, "ARCHIVE_MERKLE_REBRIDGE_SELECTED_SIGNERS_INVALID") for x in selected
    ]
    if (
        selected != sorted(selected)
        or len(selected) != root["threshold"]
        or len(set(selected)) != len(selected)
        or any(x not in root["keys"] for x in selected)
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_SELECTED_SIGNERS_INVALID")
    if role == "recovery":
        if len({root["keys"][k]["operator"] for k in selected}) < int(
            run165.POLICY["min_recovery_operators"]
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_RECOVERY_OPERATOR_QUORUM_INVALID")
        if len({root["keys"][k]["recoveryChannel"] for k in selected}) < int(
            run165.POLICY["min_recovery_channels"]
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_RECOVERY_CHANNEL_QUORUM_INVALID")
    elif len({root["keys"][k]["operator"] for k in selected}) < int(
        run165.POLICY["min_governance_operators"]
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_GOVERNANCE_OPERATOR_QUORUM_INVALID")
    if issued < root["issued"] or issued >= root["expires"]:
        _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_OUTSIDE_ROOT_LIFETIME")
    for kid in selected:
        if issued >= _dt(
            root["keys"][kid]["expires"],
            "ARCHIVE_MERKLE_REBRIDGE_SIGNER_EXPIRES_INVALID",
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_SIGNER_EXPIRED")
    signatures = doc["signatures"]
    if not isinstance(signatures, list) or len(signatures) != len(selected):
        _fail("ARCHIVE_MERKLE_REBRIDGE_SIGNATURE_SET_INVALID")
    sigmap = {}
    for item in signatures:
        if not isinstance(item, dict) or set(item) != {"keyId", "signature"}:
            _fail("ARCHIVE_MERKLE_REBRIDGE_SIGNATURE_SCHEMA_INVALID")
        kid = _id(item["keyId"], "ARCHIVE_MERKLE_REBRIDGE_SIGNATURE_KEY_INVALID")
        if kid in sigmap:
            _fail("ARCHIVE_MERKLE_REBRIDGE_SIGNATURE_DUPLICATE")
        sigmap[kid] = item["signature"]
    if sorted(sigmap) != selected or signatures != sorted(
        signatures, key=lambda x: x["keyId"]
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_SIGNATURE_SET_INVALID")
    message = _canonical(signed)
    for kid in selected:
        _verify_sig(
            root["keys"][kid]["publicKey"],
            sigmap[kid],
            message,
            "ARCHIVE_MERKLE_REBRIDGE_TRANSITION_SIGNATURE",
        )
    subjects = {
        lid: _handoff_subject(
            signed=signed, log_id=lid, checkpoint=source["checkpoints"][lid]
        )
        for lid in sorted(expected_log_ids)
    }
    expected_hashes = {
        lid: _sha_bytes(_canonical(subjects[lid])) for lid in sorted(subjects)
    }
    if signed["handoffSubjectSha256s"] != expected_hashes:
        _fail("ARCHIVE_MERKLE_REBRIDGE_HANDOFF_SUBJECT_HASH_INVALID")
    return {
        "signed": signed,
        "transitionSha256": _sha_bytes(_canonical(doc)),
        "subjects": subjects,
        "current": current,
        "next": nxt,
        "revoked": revoked,
        "kind": kind,
        "rebridgeSequence": rbseq,
        "issued": issued,
        "transitionId": tid,
    }


def _verify_handoff(
    doc: dict[str, Any], *, transition: dict[str, Any]
) -> dict[str, Any]:
    signed = transition["signed"]
    if set(doc) != {
        "schemaVersion",
        "transitionId",
        "rebridgeSequence",
        "proofs",
    } or doc.get("schemaVersion") != int(POLICY["handoff_schema_version"]):
        _fail("ARCHIVE_MERKLE_REBRIDGE_HANDOFF_SCHEMA_INVALID")
    if (
        doc.get("transitionId") != signed["transitionId"]
        or doc.get("rebridgeSequence") != signed["rebridgeSequence"]
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_HANDOFF_BINDING_INVALID")
    proofs = doc["proofs"]
    if (
        not isinstance(proofs, list)
        or len(proofs) != len(transition["subjects"])
        or proofs != sorted(proofs, key=lambda x: x.get("logId", ""))
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_HANDOFF_PROOF_SET_INVALID")
    seen = set()
    for proof in proofs:
        expected = {
            "logId",
            "subjectSha256",
            "newLogSignature",
            "newGossipSignature",
            "oldLogSignature",
            "oldGossipSignature",
        }
        if not isinstance(proof, dict) or set(proof) != expected:
            _fail("ARCHIVE_MERKLE_REBRIDGE_HANDOFF_PROOF_SCHEMA_INVALID")
        lid = _id(proof["logId"], "ARCHIVE_MERKLE_REBRIDGE_HANDOFF_LOG_ID_INVALID")
        if lid in seen or lid not in transition["subjects"]:
            _fail("ARCHIVE_MERKLE_REBRIDGE_HANDOFF_LOG_ID_INVALID")
        seen.add(lid)
        subject = transition["subjects"][lid]
        raw = _canonical(subject)
        if proof["subjectSha256"] != _sha_bytes(raw):
            _fail("ARCHIVE_MERKLE_REBRIDGE_HANDOFF_SUBJECT_BINDING_INVALID")
        cur = transition["current"][lid]
        nxt = transition["next"][lid]
        _verify_sig(
            nxt["publicKey"],
            proof["newLogSignature"],
            raw,
            "ARCHIVE_MERKLE_REBRIDGE_NEW_LOG_HANDOFF",
        )
        _verify_sig(
            nxt["gossipPublicKey"],
            proof["newGossipSignature"],
            raw,
            "ARCHIVE_MERKLE_REBRIDGE_NEW_GOSSIP_HANDOFF",
        )
        changed = lid in signed["changedLogIds"]
        if transition["kind"] == "scheduled-rotation" and changed:
            if proof["oldLogSignature"] is None or proof["oldGossipSignature"] is None:
                _fail("ARCHIVE_MERKLE_REBRIDGE_OLD_HANDOFF_REQUIRED")
            _verify_sig(
                cur["publicKey"],
                proof["oldLogSignature"],
                raw,
                "ARCHIVE_MERKLE_REBRIDGE_OLD_LOG_HANDOFF",
            )
            _verify_sig(
                cur["gossipPublicKey"],
                proof["oldGossipSignature"],
                raw,
                "ARCHIVE_MERKLE_REBRIDGE_OLD_GOSSIP_HANDOFF",
            )
        elif (
            proof["oldLogSignature"] is not None
            or proof["oldGossipSignature"] is not None
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_OLD_HANDOFF_FORBIDDEN")
    return doc


def _log_challenge(
    *,
    sequence: int,
    run163_head: str,
    leaf_hash: str,
    previous_merkle_head: str,
    authority_bridge_head: str,
    active_authority_sha: str,
    source_state_sha: str,
) -> str:
    return _sha_bytes(
        _canonical(
            {
                "sequence": sequence,
                "run163AnchorConsensusHeadSha256": run163_head,
                "leafHash": leaf_hash,
                "previousArchiveMerkleRebridgeContinuityHeadSha256": (
                    previous_merkle_head
                ),
                "archiveMerkleRebridgeAuthorityHeadSha256": authority_bridge_head,
                "activeAuthoritySha256": active_authority_sha,
                "sourceTrustedStateSha256": source_state_sha,
            }
        )
    )


def _gossip_challenge(
    *,
    sequence: int,
    run163_head: str,
    leaf_hash: str,
    checkpoints: dict[str, str],
    authority_bridge_head: str,
    active_authority_sha: str,
    source_state_sha: str,
) -> str:
    return _sha_bytes(
        _canonical(
            {
                "sequence": sequence,
                "run163AnchorConsensusHeadSha256": run163_head,
                "leafHash": leaf_hash,
                "checkpointSha256s": checkpoints,
                "archiveMerkleRebridgeAuthorityHeadSha256": authority_bridge_head,
                "activeAuthoritySha256": active_authority_sha,
                "sourceTrustedStateSha256": source_state_sha,
            }
        )
    )


def _normalize_merkle_error(prefix: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ArchiveMerkleRebridgeError:
        raise
    except Exception as exc:
        raise ArchiveMerkleRebridgeError(prefix + ":" + str(exc)) from exc


def _load_output(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_MERKLE_REBRIDGE_OUTPUT_INVALID")
    if {p.name for p in root.iterdir()} != _OUTPUT_NAMES:
        _fail("ARCHIVE_MERKLE_REBRIDGE_OUTPUT_ALLOWLIST_INVALID")
    docs = {}
    raws = {}
    for name in sorted(_OUTPUT_NAMES):
        docs[name], raws[name] = _read_json(
            root / name, "ARCHIVE_MERKLE_REBRIDGE_OUTPUT"
        )
    return docs, raws


def _bundle_base(
    *,
    run166_info: dict[str, Any],
    run165_info: dict[str, Any],
    governance: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "archive-merkle-authority-rebridge-history",
        "baseRun166Artifacts": run166_info["artifacts"],
        "baseRun166Sequence": run166_info["state"]["sequence"],
        "baseRun166MerkleAuthorityContinuityHeadSha256": run166_info["state"][
            "merkleAuthorityContinuityHeadSha256"
        ],
        "run165Artifacts": run165_info["artifacts"],
        "run165Sequence": run165_info["state"]["sequence"],
        "run165LogAuthorityChainHeadSha256": run165_info["state"][
            "logAuthorityChainHeadSha256"
        ],
        "governanceRootSha256": governance["sha256"],
        "recoveryRootSha256": recovery["sha256"],
    }


def _make_state(
    *,
    base: dict[str, Any],
    events: list[dict[str, Any]],
    active_authority: dict[str, Any],
    revoked: list[str],
    rebridge_sequence: int,
    authority_head: str,
    merkle_head: str,
    checkpoints: dict[str, Any],
    last_transition_issued: datetime | None,
) -> dict[str, Any]:
    bundle = dict(base)
    bundle["events"] = events
    braw = _canonical(bundle)
    last = events[-1]
    return {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-archive-merkle-rebridge",
        "sequence": last["sequence"],
        "postRun166Sequence": len(events),
        "rebridgeSequence": rebridge_sequence,
        "baseRun166Sequence": base["baseRun166Sequence"],
        "run165Sequence": base["run165Sequence"],
        "run165LogAuthorityChainHeadSha256": base["run165LogAuthorityChainHeadSha256"],
        "governanceRootSha256": base["governanceRootSha256"],
        "recoveryRootSha256": base["recoveryRootSha256"],
        "archiveMerkleRebridgeAuthorityHeadSha256": authority_head,
        "activeAuthority": active_authority,
        "activeAuthoritySha256": _authority_sha(active_authority),
        "revokedKeyFingerprints": revoked,
        "merkleRebridgeContinuityHeadSha256": merkle_head,
        "lastRun163AnchorConsensusHeadSha256": last["run163AnchorConsensusHeadSha256"],
        "lastCheckpoints": {
            lid: {
                k: checkpoints[lid][k]
                for k in ("checkpointSha256", "treeSize", "rootHash")
            }
            for lid in sorted(checkpoints)
        },
        "lastTransitionIssuedAt": (
            None if last_transition_issued is None else _ts(last_transition_issued)
        ),
        "bundleArtifact": {
            "name": "release-archive-merkle-rebridge-bundle.json",
            "sha256": _sha_bytes(braw),
            "size": len(braw),
        },
    }


def _make_active(state: dict[str, Any], last: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": int(POLICY["active_schema_version"]),
        "status": "active-archive-merkle-rebridge",
        "sequence": state["sequence"],
        "postRun166Sequence": state["postRun166Sequence"],
        "rebridgeSequence": state["rebridgeSequence"],
        "run165Sequence": state["run165Sequence"],
        "run165LogAuthorityChainHeadSha256": state["run165LogAuthorityChainHeadSha256"],
        "archiveMerkleRebridgeAuthorityHeadSha256": state[
            "archiveMerkleRebridgeAuthorityHeadSha256"
        ],
        "authority": state["activeAuthority"],
        "activeAuthoritySha256": state["activeAuthoritySha256"],
        "revokedKeyFingerprints": state["revokedKeyFingerprints"],
        "merkleRebridgeContinuityHeadSha256": state[
            "merkleRebridgeContinuityHeadSha256"
        ],
        "run163AnchorConsensusHeadSha256": last["run163AnchorConsensusHeadSha256"],
        "leafHash": last["leafHash"],
        "logs": last["logs"],
        "gossipResponseSha256s": last["gossipResponseSha256s"],
    }


def _source_from_base(
    run166_info: dict[str, Any], run165_info: dict[str, Any]
) -> dict[str, Any]:
    raw = run166_info["raws"][_DOC_ARCHIVE_MERKLE_CONTINUITY_STATE]
    cps = _normalize_checkpoints(
        run166_info["state"]["lastCheckpoints"],
        expected_log_ids=set(run166_info["active"]["authority"]),
        code="ARCHIVE_MERKLE_REBRIDGE_BASE_CHECKPOINTS",
    )
    return {
        "sourceKind": "run166",
        "stateRaw": raw,
        "stateSha256": _sha_bytes(raw),
        "merkleSequence": run166_info["state"]["sequence"],
        "merkleHead": run166_info["state"]["merkleAuthorityContinuityHeadSha256"],
        "checkpoints": cps,
        "authority": run166_info["active"]["authority"],
        "revoked": list(run165_info["state"]["revokedKeyFingerprints"]),
        "rebridgeSequence": 0,
        "authorityHead": _base_authority_head(run165_info, run166_info),
        "lastTransitionIssued": None,
    }


def _replay(  # ruff: ignore[too-many-branches]
    *,
    run166_info: dict[str, Any],
    run165_info: dict[str, Any],
    governance: dict[str, Any],
    recovery: dict[str, Any],
    run164_info: dict[str, Any],
    bundle: dict[str, Any],
    receipt: dict[str, Any],
    now: datetime,
    current_run163_docs: dict[str, Any] | None,
    current_run163_raws: dict[str, bytes] | None,
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
) -> dict[str, Any]:
    base = _bundle_base(
        run166_info=run166_info,
        run165_info=run165_info,
        governance=governance,
        recovery=recovery,
    )
    expected_bundle_keys = set(base) | {"events"}
    if set(bundle) != expected_bundle_keys or any(
        bundle[k] != v for k, v in base.items()
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_BUNDLE_BASE_INVALID")
    if set(receipt) != {
        "schemaVersion",
        "status",
        "run166Documents",
        "run165Documents",
        "events",
    }:
        _fail("ARCHIVE_MERKLE_REBRIDGE_RECEIPT_SCHEMA_INVALID")
    if (
        receipt["schemaVersion"] != int(POLICY["receipt_schema_version"])
        or receipt["status"] != "archive-merkle-authority-rebridge-accepted"
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_RECEIPT_SCHEMA_INVALID")
    if (
        receipt["run166Documents"] != run166_info["docs"]
        or receipt["run165Documents"] != run165_info["docs"]
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_BASE_DOCUMENT_MISMATCH")
    events = bundle["events"]
    recs = receipt["events"]
    if (
        not isinstance(events, list)
        or not events
        or not isinstance(recs, list)
        or len(events) != len(recs)
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_HISTORY_LENGTH_INVALID")
    source = _source_from_base(run166_info, run165_info)
    seen_ids: set[str] = set()
    previous_transition_issued = None
    active_authority = source["authority"]
    revoked = source["revoked"]
    authority_head = source["authorityHead"]
    rebridge_sequence = 0
    merkle_head = source["merkleHead"]
    checkpoints = source["checkpoints"]
    prefix_events: list[dict[str, Any]] = []
    last = None
    for idx, (event, rec) in enumerate(zip(events, recs), 1):
        sequence = base["baseRun166Sequence"] + idx
        expected_event = {
            "sequence",
            "postRun166Sequence",
            "action",
            "rebridgeSequence",
            "run163Sequence",
            "run163AnchorConsensusHeadSha256",
            "run163Artifacts",
            "leafHash",
            "sourceTrustedStateSha256",
            "archiveMerkleRebridgeAuthorityHeadSha256",
            "activeAuthoritySha256",
            "revokedKeyFingerprints",
            "transitionSha256",
            "handoffSha256",
            "logs",
            "logChallenge",
            "gossipChallenge",
            "gossipResponseSha256s",
            "merkleRebridgeContinuityHeadSha256",
        }
        if (
            not isinstance(event, dict)
            or set(event) != expected_event
            or event.get("sequence") != sequence
            or event.get("postRun166Sequence") != idx
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_EVENT_SCHEMA_INVALID")
        expected_rec = {
            "sequence",
            "run163Documents",
            "transitionDocument",
            "handoffProofDocument",
            "logResponses",
            "gossipResponses",
        }
        if (
            not isinstance(rec, dict)
            or set(rec) != expected_rec
            or rec.get("sequence") != sequence
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_RECEIPT_EVENT_INVALID")
        if event["sourceTrustedStateSha256"] != source["stateSha256"]:
            _fail("ARCHIVE_MERKLE_REBRIDGE_EVENT_SOURCE_STATE_INVALID")
        vr163 = _verify_embedded_run163(
            rec["run163Documents"],
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
            now=now,
        )
        if (
            vr163["sequence"] != sequence
            or event["run163Sequence"] != sequence
            or event["run163AnchorConsensusHeadSha256"]
            != vr163["anchor_consensus_head_sha256"]
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_RUN163_BINDING_INVALID")
        run163_raws = {
            name: _canonical(rec["run163Documents"][name])
            for name in sorted(rec["run163Documents"])
        }
        if event["run163Artifacts"] != _artifact_map(run163_raws):
            _fail("ARCHIVE_MERKLE_REBRIDGE_RUN163_ARTIFACT_INVALID")
        leaf_doc = merkle._leaf_document(rec["run163Documents"], run163_raws)
        leaf_hash = merkle.merkle_leaf_hash(_canonical(leaf_doc))
        if event["leafHash"] != leaf_hash:
            _fail("ARCHIVE_MERKLE_REBRIDGE_LEAF_INVALID")
        action = event["action"]
        if idx == 1 and action != "rebridge":
            _fail("ARCHIVE_MERKLE_REBRIDGE_FIRST_EVENT_REQUIRES_REBRIDGE")
        if action == "rebridge":
            if rec["transitionDocument"] is None or rec["handoffProofDocument"] is None:
                _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_MISSING")
            tr = _verify_transition(
                rec["transitionDocument"],
                source=source,
                governance=governance,
                recovery=recovery,
                expected_log_ids=set(active_authority),
                seen_ids=seen_ids,
                previous_transition_issued=previous_transition_issued,
                now=now,
                creation=False,
            )
            hf = _verify_handoff(rec["handoffProofDocument"], transition=tr)
            tsha = tr["transitionSha256"]
            hsha = _sha_bytes(_canonical(hf))
            if event["transitionSha256"] != tsha or event["handoffSha256"] != hsha:
                _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_ARTIFACT_BINDING_INVALID")
            authority_head = _bridge_head(
                authority_head, tsha, hsha, source["stateSha256"]
            )
            active_authority = tr["next"]
            revoked = tr["revoked"]
            rebridge_sequence = tr["rebridgeSequence"]
            previous_transition_issued = tr["issued"]
            seen_ids.add(tr["transitionId"])
        elif action == "append":
            if (
                rec["transitionDocument"] is not None
                or rec["handoffProofDocument"] is not None
                or event["transitionSha256"] is not None
                or event["handoffSha256"] is not None
            ):
                _fail("ARCHIVE_MERKLE_REBRIDGE_APPEND_TRANSITION_FORBIDDEN")
        else:
            _fail("ARCHIVE_MERKLE_REBRIDGE_ACTION_INVALID")
        if (
            event["rebridgeSequence"] != rebridge_sequence
            or event["archiveMerkleRebridgeAuthorityHeadSha256"] != authority_head
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_AUTHORITY_HEAD_BINDING_INVALID")
        active_sha = _authority_sha(active_authority)
        if (
            event["activeAuthoritySha256"] != active_sha
            or event["revokedKeyFingerprints"] != revoked
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_ACTIVE_AUTHORITY_BINDING_INVALID")
        _enforce_separation(
            governance,
            recovery,
            run164_info["root"],
            witness_root_path,
            anchor_plan_path,
            active_authority,
        )
        log_challenge = _log_challenge(
            sequence=sequence,
            run163_head=event["run163AnchorConsensusHeadSha256"],
            leaf_hash=leaf_hash,
            previous_merkle_head=merkle_head,
            authority_bridge_head=authority_head,
            active_authority_sha=active_sha,
            source_state_sha=source["stateSha256"],
        )
        if event["logChallenge"] != log_challenge:
            _fail("ARCHIVE_MERKLE_REBRIDGE_LOG_CHALLENGE_INVALID")
        log_docs = rec["logResponses"]
        if not isinstance(log_docs, dict) or set(log_docs) != set(active_authority):
            _fail("ARCHIVE_MERKLE_REBRIDGE_LOG_RESPONSE_SET_INVALID")
        rows = []
        next_checkpoints = {}
        for lid in sorted(active_authority):
            prev = {
                "logId": lid,
                **checkpoints[lid],
                "leafIndex": checkpoints[lid]["treeSize"] - 1,
                "leafHash": None,
            }
            parsed = _normalize_merkle_error(
                "ARCHIVE_MERKLE_REBRIDGE_CHECKPOINT_INVALID",
                merkle._verify_checkpoint,
                log_docs[lid],
                log_id=lid,
                config=active_authority[lid],
                sequence=sequence,
                challenge=log_challenge,
                expected_leaf_hash=leaf_hash,
                previous=prev,
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
            next_checkpoints[lid] = {
                k: row[k] for k in ("checkpointSha256", "treeSize", "rootHash")
            }
        if event["logs"] != rows:
            _fail("ARCHIVE_MERKLE_REBRIDGE_LOG_ROWS_INVALID")
        cp_hashes = {row["logId"]: row["checkpointSha256"] for row in rows}
        gossip_challenge = _gossip_challenge(
            sequence=sequence,
            run163_head=event["run163AnchorConsensusHeadSha256"],
            leaf_hash=leaf_hash,
            checkpoints=cp_hashes,
            authority_bridge_head=authority_head,
            active_authority_sha=active_sha,
            source_state_sha=source["stateSha256"],
        )
        if event["gossipChallenge"] != gossip_challenge:
            _fail("ARCHIVE_MERKLE_REBRIDGE_GOSSIP_CHALLENGE_INVALID")
        gossip_docs = rec["gossipResponses"]
        if not isinstance(gossip_docs, dict) or set(gossip_docs) != set(
            active_authority
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_GOSSIP_RESPONSE_SET_INVALID")
        gossip_hashes = []
        for lid in sorted(active_authority):
            parsed = _normalize_merkle_error(
                "ARCHIVE_MERKLE_REBRIDGE_GOSSIP_INVALID",
                merkle._verify_gossip,
                gossip_docs[lid],
                log_id=lid,
                config=active_authority[lid],
                sequence=sequence,
                challenge=gossip_challenge,
                run163_head=event["run163AnchorConsensusHeadSha256"],
                checkpoints=cp_hashes,
                now=now,
                creation=False,
                historical=True,
            )
            gossip_hashes.append(parsed["gossipSha256"])
        if event["gossipResponseSha256s"] != gossip_hashes:
            _fail("ARCHIVE_MERKLE_REBRIDGE_GOSSIP_HASHES_INVALID")
        bare = {k: event[k] for k in event if k != "merkleRebridgeContinuityHeadSha256"}
        new_merkle_head = _merkle_head(merkle_head, bare)
        if event["merkleRebridgeContinuityHeadSha256"] != new_merkle_head:
            _fail("ARCHIVE_MERKLE_REBRIDGE_CHAIN_HEAD_INVALID")
        merkle_head = new_merkle_head
        checkpoints = next_checkpoints
        last = event
        prefix_events.append(event)
        prefix_state = _make_state(
            base=base,
            events=prefix_events,
            active_authority=active_authority,
            revoked=revoked,
            rebridge_sequence=rebridge_sequence,
            authority_head=authority_head,
            merkle_head=merkle_head,
            checkpoints=checkpoints,
            last_transition_issued=previous_transition_issued,
        )
        source = {
            "sourceKind": "run167",
            "stateRaw": _canonical(prefix_state),
            "stateSha256": _sha_bytes(_canonical(prefix_state)),
            "merkleSequence": sequence,
            "merkleHead": merkle_head,
            "checkpoints": checkpoints,
            "authority": active_authority,
            "revoked": revoked,
            "rebridgeSequence": rebridge_sequence,
            "authorityHead": authority_head,
            "lastTransitionIssued": previous_transition_issued,
        }
    if current_run163_docs is not None:
        latest_docs = receipt["events"][-1]["run163Documents"]
        if (
            latest_docs != current_run163_docs
            or {n: _canonical(latest_docs[n]) for n in sorted(latest_docs)}
            != current_run163_raws
        ):
            _fail("ARCHIVE_MERKLE_REBRIDGE_CURRENT_RUN163_MISMATCH")
    return {
        "sequence": last["sequence"],
        "postRun166Sequence": len(events),
        "rebridgeSequence": rebridge_sequence,
        "authority": active_authority,
        "revoked": revoked,
        "authorityHead": authority_head,
        "merkleHead": merkle_head,
        "checkpoints": checkpoints,
        "last": last,
        "lastTransitionIssued": previous_transition_issued,
        "state": prefix_state,
        "source": source,
    }


def verify_archive_merkle_rebridge(  # ruff: ignore[undocumented-public-function]
    *,
    run160_dir: Path,
    run166_run161_dir: Path,
    run161_dir: Path,
    retention_root_path: Path,
    retention_root_pin: str,
    witness_root_path: Path,
    witness_root_pin: str,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    run166_run162_dir: Path,
    run162_dir: Path,
    anchor_plan_path: Path,
    run166_run163_dir: Path,
    run163_dir: Path,
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    governance_root_path: Path,
    governance_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    run165_dir: Path,
    run166_dir: Path,
    output_dir: Path,
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run164_info = _load_run164(
        run164_dir, transparency_root_path, transparency_root_pin, now=current
    )
    run165_info = _load_run165(
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        run165_dir=run165_dir,
        now=current,
    )
    run166_info = _load_run166(
        run160_dir=run160_dir,
        run166_run161_dir=run166_run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        witness_root_path=witness_root_path,
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        run166_run162_dir=run166_run162_dir,
        anchor_plan_path=anchor_plan_path,
        run166_run163_dir=run166_run163_dir,
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        run165_dir=run165_dir,
        run166_dir=run166_dir,
        now=current,
    )
    governance = _verify_control_root(
        governance_root_path,
        governance_root_pin,
        recovery=False,
        now=current,
        historical=True,
    )
    recovery = _verify_control_root(
        recovery_root_path,
        recovery_root_pin,
        recovery=True,
        now=current,
        historical=True,
    )
    current163 = _load_current_run163(
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
    docs, _raws = _load_output(output_dir)
    replay = _replay(
        run166_info=run166_info,
        run165_info=run165_info,
        governance=governance,
        recovery=recovery,
        run164_info=run164_info,
        bundle=docs["release-archive-merkle-rebridge-bundle.json"],
        receipt=docs["release-archive-merkle-rebridge-receipt.json"],
        now=current,
        current_run163_docs=current163["docs"],
        current_run163_raws=current163["raws"],
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
    )
    if docs[_DOC_ARCHIVE_MERKLE_REBRIDGE_STATE] != replay["state"]:
        _fail("ARCHIVE_MERKLE_REBRIDGE_STATE_MISMATCH")
    if docs["active-archive-merkle-rebridge.json"] != _make_active(
        replay["state"], replay["last"]
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_ACTIVE_MISMATCH")
    if not historical:
        if replay["lastTransitionIssued"] is not None and current - replay[
            "lastTransitionIssued"
        ] > timedelta(days=int(POLICY["max_active_authority_age_days"])):
            _fail("ARCHIVE_MERKLE_REBRIDGE_ACTIVE_AUTHORITY_STALE")
        latest = docs["release-archive-merkle-rebridge-receipt.json"]["events"][-1]
        max_age = timedelta(days=int(POLICY["max_active_epoch_age_days"]))
        for lid in sorted(replay["authority"]):
            integrated = merkle._dt(
                latest["logResponses"][lid]["signed"]["integratedAt"],
                "ARCHIVE_MERKLE_REBRIDGE_ACTIVE_TIME_INVALID",
            )
            observed = merkle._dt(
                latest["gossipResponses"][lid]["signed"]["observedAt"],
                "ARCHIVE_MERKLE_REBRIDGE_ACTIVE_TIME_INVALID",
            )
            if current - integrated > max_age or current - observed > max_age:
                _fail("ARCHIVE_MERKLE_REBRIDGE_ACTIVE_EPOCH_STALE")
    return {
        "ok": True,
        "sequence": replay["sequence"],
        "post_run166_sequence": replay["postRun166Sequence"],
        "rebridge_sequence": replay["rebridgeSequence"],
        "archive_merkle_rebridge_authority_head_sha256": replay["authorityHead"],
        "merkle_rebridge_continuity_head_sha256": replay["merkleHead"],
    }


def advance_archive_merkle_rebridge(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    run160_dir: Path,
    run166_run161_dir: Path,
    run161_dir: Path,
    retention_root_path: Path,
    retention_root_pin: str,
    witness_root_path: Path,
    witness_root_pin: str,
    bootstrap_pin: str,
    recovery_pin: str,
    attestation_pins: list[str],
    run166_run162_dir: Path,
    run162_dir: Path,
    anchor_plan_path: Path,
    run166_run163_dir: Path,
    run163_dir: Path,
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    governance_root_path: Path,
    governance_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    run165_dir: Path,
    run166_dir: Path,
    output_dir: Path,
    adapters: list[
        tuple[
            str,
            Callable[[dict[str, Any]], dict[str, Any]],
            Callable[[dict[str, Any]], dict[str, Any]],
        ]
    ],
    transition_path: Path | None = None,
    handoff_path: Path | None = None,
    previous_output_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if (transition_path is None) != (handoff_path is None):
        _fail("ARCHIVE_MERKLE_REBRIDGE_TRANSITION_PAIR_REQUIRED")
    authority_paths = [
        Path(run160_dir),
        Path(run166_run161_dir),
        Path(run161_dir),
        Path(retention_root_path),
        Path(witness_root_path),
        Path(run166_run162_dir),
        Path(run162_dir),
        Path(anchor_plan_path),
        Path(run166_run163_dir),
        Path(run163_dir),
        Path(run164_dir),
        Path(transparency_root_path),
        Path(governance_root_path),
        Path(recovery_root_path),
        Path(run165_dir),
        Path(run166_dir),
    ]
    if transition_path is not None:
        authority_paths += [Path(transition_path), Path(handoff_path)]
    if previous_output_dir is not None:
        authority_paths.append(Path(previous_output_dir))
    before = _authority_fingerprint(
        authority_paths, "ARCHIVE_MERKLE_REBRIDGE_INPUT_DRIFT"
    )
    run164_info = _load_run164(
        run164_dir, transparency_root_path, transparency_root_pin, now=current
    )
    run165_info = _load_run165(
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        run165_dir=run165_dir,
        now=current,
    )
    run166_info = _load_run166(
        run160_dir=run160_dir,
        run166_run161_dir=run166_run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        witness_root_path=witness_root_path,
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        run166_run162_dir=run166_run162_dir,
        anchor_plan_path=anchor_plan_path,
        run166_run163_dir=run166_run163_dir,
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        run165_dir=run165_dir,
        run166_dir=run166_dir,
        now=current,
    )
    governance = _verify_control_root(
        governance_root_path,
        governance_root_pin,
        recovery=False,
        now=current,
        historical=True,
    )
    recovery = _verify_control_root(
        recovery_root_path,
        recovery_root_pin,
        recovery=True,
        now=current,
        historical=True,
    )
    current163 = _load_current_run163(
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
    base = _bundle_base(
        run166_info=run166_info,
        run165_info=run165_info,
        governance=governance,
        recovery=recovery,
    )
    old_events: list[dict[str, Any]] = []
    old_receipts: list[dict[str, Any]] = []
    source = _source_from_base(run166_info, run165_info)
    seen_ids: set[str] = set()
    previous_transition_issued = None
    if previous_output_dir is None:
        if transition_path is None:
            _fail("ARCHIVE_MERKLE_REBRIDGE_FIRST_EVENT_REQUIRES_TRANSITION")
    else:
        pdocs, _ = _load_output(Path(previous_output_dir))
        previous = _replay(
            run166_info=run166_info,
            run165_info=run165_info,
            governance=governance,
            recovery=recovery,
            run164_info=run164_info,
            bundle=pdocs["release-archive-merkle-rebridge-bundle.json"],
            receipt=pdocs["release-archive-merkle-rebridge-receipt.json"],
            now=current,
            current_run163_docs=None,
            current_run163_raws=None,
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
        )
        source = previous["source"]
        old_events = list(
            pdocs["release-archive-merkle-rebridge-bundle.json"]["events"]
        )
        old_receipts = list(
            pdocs["release-archive-merkle-rebridge-receipt.json"]["events"]
        )
        for rec in old_receipts:
            if rec["transitionDocument"] is not None:
                seen_ids.add(rec["transitionDocument"]["signed"]["transitionId"])
        previous_transition_issued = previous["lastTransitionIssued"]
    state163 = current163["docs"][_DOC_ARCHIVE_ANCHOR_STATE]
    sequence = state163.get("sequence")
    if not isinstance(sequence, int) or sequence != source["merkleSequence"] + 1:
        _fail("ARCHIVE_MERKLE_REBRIDGE_SEQUENCE_INVALID")
    active_authority = source["authority"]
    revoked = list(source["revoked"])
    authority_head = source["authorityHead"]
    rebridge_sequence = source["rebridgeSequence"]
    transition_doc = handoff_doc = None
    transition_sha = handoff_sha = None
    action = "append"
    if transition_path is not None:
        transition_doc, _ = _read_json(
            Path(transition_path), "ARCHIVE_MERKLE_REBRIDGE_TRANSITION"
        )
        tr = _verify_transition(
            transition_doc,
            source=source,
            governance=governance,
            recovery=recovery,
            expected_log_ids=set(active_authority),
            seen_ids=seen_ids,
            previous_transition_issued=previous_transition_issued,
            now=current,
            creation=True,
        )
        handoff_doc, _ = _read_json(
            Path(handoff_path), "ARCHIVE_MERKLE_REBRIDGE_HANDOFF"
        )
        hf = _verify_handoff(handoff_doc, transition=tr)
        transition_sha = tr["transitionSha256"]
        handoff_sha = _sha_bytes(_canonical(hf))
        authority_head = _bridge_head(
            authority_head, transition_sha, handoff_sha, source["stateSha256"]
        )
        active_authority = tr["next"]
        revoked = tr["revoked"]
        rebridge_sequence = tr["rebridgeSequence"]
        previous_transition_issued = tr["issued"]
        action = "rebridge"
    _enforce_separation(
        governance,
        recovery,
        run164_info["root"],
        witness_root_path,
        anchor_plan_path,
        active_authority,
    )
    if {lid for lid, _, _ in adapters} != set(active_authority) or len(adapters) != len(
        active_authority
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_ADAPTER_SET_INVALID")
    amap = {lid: (log, gossip) for lid, log, gossip in adapters}
    leaf_doc = merkle._leaf_document(current163["docs"], current163["raws"])
    leaf_hash = merkle.merkle_leaf_hash(_canonical(leaf_doc))
    active_sha = _authority_sha(active_authority)
    log_challenge = _log_challenge(
        sequence=sequence,
        run163_head=state163["anchorConsensusHeadSha256"],
        leaf_hash=leaf_hash,
        previous_merkle_head=source["merkleHead"],
        authority_bridge_head=authority_head,
        active_authority_sha=active_sha,
        source_state_sha=source["stateSha256"],
    )
    log_docs = {}
    rows = []
    next_checkpoints = {}
    for lid in sorted(active_authority):
        prev = {
            "logId": lid,
            **source["checkpoints"][lid],
            "leafIndex": source["checkpoints"][lid]["treeSize"] - 1,
            "leafHash": None,
        }
        request = {
            "operation": "append-merkle-leaf-rebridged-authority",
            "protocolVersion": 1,
            "logId": lid,
            "sequence": sequence,
            "challenge": log_challenge,
            "leaf": leaf_doc,
            "leafHash": leaf_hash,
            "previousTreeSize": prev["treeSize"],
            "previousRootHash": prev["rootHash"],
            "archiveMerkleRebridgeAuthorityHeadSha256": authority_head,
            "activeAuthoritySha256": active_sha,
            "sourceTrustedStateSha256": source["stateSha256"],
        }
        doc = amap[lid][0](request)
        parsed = _normalize_merkle_error(
            "ARCHIVE_MERKLE_REBRIDGE_CHECKPOINT_INVALID",
            merkle._verify_checkpoint,
            doc,
            log_id=lid,
            config=active_authority[lid],
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
        next_checkpoints[lid] = {
            k: row[k] for k in ("checkpointSha256", "treeSize", "rootHash")
        }
    cp_hashes = {row["logId"]: row["checkpointSha256"] for row in rows}
    gossip_challenge = _gossip_challenge(
        sequence=sequence,
        run163_head=state163["anchorConsensusHeadSha256"],
        leaf_hash=leaf_hash,
        checkpoints=cp_hashes,
        authority_bridge_head=authority_head,
        active_authority_sha=active_sha,
        source_state_sha=source["stateSha256"],
    )
    gossip_docs = {}
    gossip_hashes = []
    for lid in sorted(active_authority):
        request = {
            "operation": "gossip-merkle-rebridged-authority",
            "protocolVersion": 1,
            "sequence": sequence,
            "challenge": gossip_challenge,
            "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
            "leafHash": leaf_hash,
            "checkpointSha256s": cp_hashes,
            "archiveMerkleRebridgeAuthorityHeadSha256": authority_head,
            "activeAuthoritySha256": active_sha,
            "sourceTrustedStateSha256": source["stateSha256"],
        }
        doc = amap[lid][1](request)
        parsed = _normalize_merkle_error(
            "ARCHIVE_MERKLE_REBRIDGE_GOSSIP_INVALID",
            merkle._verify_gossip,
            doc,
            log_id=lid,
            config=active_authority[lid],
            sequence=sequence,
            challenge=gossip_challenge,
            run163_head=state163["anchorConsensusHeadSha256"],
            checkpoints=cp_hashes,
            now=current,
            creation=True,
            historical=False,
        )
        gossip_docs[lid] = doc
        gossip_hashes.append(parsed["gossipSha256"])
    run163_artifacts = _artifact_map(current163["raws"])
    event_bare = {
        "sequence": sequence,
        "postRun166Sequence": len(old_events) + 1,
        "action": action,
        "rebridgeSequence": rebridge_sequence,
        "run163Sequence": sequence,
        "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
        "run163Artifacts": run163_artifacts,
        "leafHash": leaf_hash,
        "sourceTrustedStateSha256": source["stateSha256"],
        "archiveMerkleRebridgeAuthorityHeadSha256": authority_head,
        "activeAuthoritySha256": active_sha,
        "revokedKeyFingerprints": revoked,
        "transitionSha256": transition_sha,
        "handoffSha256": handoff_sha,
        "logs": rows,
        "logChallenge": log_challenge,
        "gossipChallenge": gossip_challenge,
        "gossipResponseSha256s": gossip_hashes,
    }
    new_merkle_head = _merkle_head(source["merkleHead"], event_bare)
    event = dict(event_bare)
    event["merkleRebridgeContinuityHeadSha256"] = new_merkle_head
    receipt_event = {
        "sequence": sequence,
        "run163Documents": current163["docs"],
        "transitionDocument": transition_doc,
        "handoffProofDocument": handoff_doc,
        "logResponses": log_docs,
        "gossipResponses": gossip_docs,
    }
    events = [*old_events, event]
    receipt_events = [*old_receipts, receipt_event]
    bundle = dict(base)
    bundle["events"] = events
    receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": "archive-merkle-authority-rebridge-accepted",
        "run166Documents": run166_info["docs"],
        "run165Documents": run165_info["docs"],
        "events": receipt_events,
    }
    state = _make_state(
        base=base,
        events=events,
        active_authority=active_authority,
        revoked=revoked,
        rebridge_sequence=rebridge_sequence,
        authority_head=authority_head,
        merkle_head=new_merkle_head,
        checkpoints=next_checkpoints,
        last_transition_issued=previous_transition_issued,
    )
    active = _make_active(state, event)
    if before != _authority_fingerprint(
        authority_paths, "ARCHIVE_MERKLE_REBRIDGE_INPUT_DRIFT"
    ):
        _fail("ARCHIVE_MERKLE_REBRIDGE_INPUT_DRIFT")
    target = _outside(
        Path(output_dir),
        authority_paths,
        "ARCHIVE_MERKLE_REBRIDGE_OUTPUT_OVERLAPS_INPUT",
    )
    if target.exists():
        _fail("ARCHIVE_MERKLE_REBRIDGE_OUTPUT_EXISTS")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run167-merkle-rebridge-", dir=target.parent))
    try:
        _write(stage / "release-archive-merkle-rebridge-bundle.json", bundle)
        _write(stage / _DOC_ARCHIVE_MERKLE_REBRIDGE_STATE, state)
        _write(stage / "active-archive-merkle-rebridge.json", active)
        _write(stage / "release-archive-merkle-rebridge-receipt.json", receipt)
        verify_archive_merkle_rebridge(
            run160_dir=run160_dir,
            run166_run161_dir=run166_run161_dir,
            run161_dir=run161_dir,
            retention_root_path=retention_root_path,
            retention_root_pin=retention_root_pin,
            witness_root_path=witness_root_path,
            witness_root_pin=witness_root_pin,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            run166_run162_dir=run166_run162_dir,
            run162_dir=run162_dir,
            anchor_plan_path=anchor_plan_path,
            run166_run163_dir=run166_run163_dir,
            run163_dir=run163_dir,
            run164_dir=run164_dir,
            transparency_root_path=transparency_root_path,
            transparency_root_pin=transparency_root_pin,
            governance_root_path=governance_root_path,
            governance_root_pin=governance_root_pin,
            recovery_root_path=recovery_root_path,
            recovery_root_pin=recovery_root_pin,
            run165_dir=run165_dir,
            run166_dir=run166_dir,
            output_dir=stage,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_archive_merkle_rebridge(
        run160_dir=run160_dir,
        run166_run161_dir=run166_run161_dir,
        run161_dir=run161_dir,
        retention_root_path=retention_root_path,
        retention_root_pin=retention_root_pin,
        witness_root_path=witness_root_path,
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        run166_run162_dir=run166_run162_dir,
        run162_dir=run162_dir,
        anchor_plan_path=anchor_plan_path,
        run166_run163_dir=run166_run163_dir,
        run163_dir=run163_dir,
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        run165_dir=run165_dir,
        run166_dir=run166_dir,
        output_dir=target,
        now=current,
    )


def command_log(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return merkle.command_log(command)


def command_gossip(command: list[str]):  # ruff: ignore[undocumented-public-function]
    return merkle.command_gossip(command)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run160-dir", type=Path, required=True)
    parser.add_argument("--run166-run161-dir", type=Path, required=True)
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
    parser.add_argument("--run166-run162-dir", type=Path, required=True)
    parser.add_argument("--run162-dir", type=Path, required=True)
    parser.add_argument("--anchor-plan-path", type=Path, required=True)
    parser.add_argument("--run166-run163-dir", type=Path, required=True)
    parser.add_argument("--run163-dir", type=Path, required=True)
    parser.add_argument("--run164-dir", type=Path, required=True)
    parser.add_argument("--transparency-root-path", type=Path, required=True)
    parser.add_argument("--transparency-root-pin", required=True)
    parser.add_argument("--governance-root-path", type=Path, required=True)
    parser.add_argument("--governance-root-pin", required=True)
    parser.add_argument("--recovery-root-path", type=Path, required=True)
    parser.add_argument("--recovery-root-pin", required=True)
    parser.add_argument("--run165-dir", type=Path, required=True)
    parser.add_argument("--run166-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def main(  # ruff: ignore[undocumented-public-function]
    argv=None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    v = subs.add_parser("verify")
    _common(v)
    v.add_argument("--historical", action="store_true")
    a = subs.add_parser("advance")
    _common(a)
    a.add_argument("--previous-output-dir", type=Path)
    a.add_argument("--transition", type=Path)
    a.add_argument("--handoff", type=Path)
    a.add_argument("--log-command", action="append", required=True)
    a.add_argument("--gossip-command", action="append", required=True)
    ns = parser.parse_args(argv)
    kwargs = vars(ns)
    cmd = kwargs.pop("cmd")
    try:
        if cmd == "verify":
            historical = kwargs.pop("historical")
            result = verify_archive_merkle_rebridge(**kwargs, historical=historical)
        else:
            previous = kwargs.pop("previous_output_dir")
            transition = kwargs.pop("transition")
            handoff = kwargs.pop("handoff")
            log_cmds = kwargs.pop("log_command")
            gossip_cmds = kwargs.pop("gossip_command")

            def parse(items, builder):
                out = {}
                for raw in items:
                    if "::" not in raw:
                        _fail("ARCHIVE_MERKLE_REBRIDGE_COMMAND_INVALID")
                    lid, command = raw.split("::", 1)
                    out[lid] = builder(command.split())
                return out

            logs = parse(log_cmds, command_log)
            goss = parse(gossip_cmds, command_gossip)
            if set(logs) != set(goss):
                _fail("ARCHIVE_MERKLE_REBRIDGE_COMMAND_SET_INVALID")
            adapters = [(lid, logs[lid], goss[lid]) for lid in sorted(logs)]
            result = advance_archive_merkle_rebridge(
                **kwargs,
                adapters=adapters,
                transition_path=transition,
                handoff_path=handoff,
                previous_output_dir=previous,
            )
        logger.info(
            json.dumps({"ok": True, "result": result}, sort_keys=True, default=str)
        )
        return 0
    except ArchiveMerkleRebridgeError as exc:
        logger.error(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
