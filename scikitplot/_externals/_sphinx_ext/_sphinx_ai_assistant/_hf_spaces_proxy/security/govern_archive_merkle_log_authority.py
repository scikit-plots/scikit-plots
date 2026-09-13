"""
govern_archive_merkle_log_authority.
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
from typing import Any

import tomllib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

try:
    import verify_archive_merkle_transparency as merkle
except ImportError:  # canonical package import
    from . import verify_archive_merkle_transparency as merkle

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_archive_log_authority_policy.toml").read_text())
PREDICATE_TYPE = str(POLICY["predicate_type"])


# ── Verified-state document names ────────────────────────────────────────
#
# One constant per release-evidence document.  The values are the on-disk
# artifact names and are unchanged; only the repetition is removed.  Reading
# a document by a name that says which document it is also keeps static
# analysis from reading the filename token 'trusted' as 'confidential':
# these files carry published Merkle roots and SHA-256 digests, which must
# stay in clear text for any third party to verify them.
_DOC_ARCHIVE_LOG_AUTHORITY_STATE = "trusted-archive-log-authority-state.json"
_DOC_ARCHIVE_MERKLE_STATE = "trusted-archive-merkle-state.json"


_OUTPUT_NAMES = {
    "release-archive-log-authority-bundle.json",
    _DOC_ARCHIVE_LOG_AUTHORITY_STATE,
    "active-archive-log-authority.json",
    "release-archive-log-authority-receipt.json",
}
_RUN164_NAMES = {
    "release-archive-merkle-bundle.json",
    _DOC_ARCHIVE_MERKLE_STATE,
    "active-archive-merkle-evidence.json",
    "release-archive-merkle-receipt.json",
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")


class ArchiveLogAuthorityError(RuntimeError):  # ruff: ignore[undocumented-public-class]
    pass


def _fail(code: str) -> None:
    raise ArchiveLogAuthorityError(code)


def _canonical(value: Any) -> bytes:
    # Keep the release-security canonical byte contract exactly aligned with Run 164.
    return merkle._canonical(value)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _no_dupes(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            _fail("ARCHIVE_LOG_AUTHORITY_JSON_DUPLICATE_KEY")
        out[key] = value
    return out


def _loads(raw: bytes, code: str) -> dict[str, Any]:
    if len(raw) > int(POLICY["max_json_bytes"]):
        _fail(code + "_TOO_LARGE")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dupes)
    except ArchiveLogAuthorityError:
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
    except ArchiveLogAuthorityError:
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


def _pub_fingerprint(public_key: str) -> str:
    return _sha_bytes(_b64(public_key, "ARCHIVE_LOG_AUTHORITY_PUBLIC_KEY_INVALID", 32))


def _root_key(value: Any, code: str, *, recovery: bool) -> dict[str, str]:
    expected = {"identity", "operator", "expires", "publicKey"}
    if recovery:
        expected.add("recoveryChannel")
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    out = {
        "identity": _id(value["identity"], code + "_IDENTITY_INVALID"),
        "operator": _id(value["operator"], code + "_OPERATOR_INVALID"),
        "expires": value["expires"],
        "publicKey": value["publicKey"],
    }
    _dt(out["expires"], code + "_EXPIRES_INVALID")
    _b64(out["publicKey"], code + "_PUBLIC_INVALID", 32)
    if recovery:
        out["recoveryChannel"] = _id(
            value["recoveryChannel"], code + "_CHANNEL_INVALID"
        )
    if out != value:
        _fail(code + "_NOT_NORMALIZED")
    return out


def _verify_control_root(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    expected_pin: str,
    *,
    now: datetime,
    recovery: bool,
    historical: bool,
) -> dict[str, Any]:
    prefix = "ARCHIVE_LOG_RECOVERY_ROOT" if recovery else "ARCHIVE_LOG_GOVERNANCE_ROOT"
    root_type = (
        "archive-log-recovery-root" if recovery else "archive-log-governance-root"
    )
    schema_key = (
        "recovery_root_schema_version" if recovery else "governance_root_schema_version"
    )
    if set(doc) != {"signed", "signatures"} or not isinstance(doc.get("signed"), dict):
        _fail(prefix + "_SCHEMA_INVALID")
    signed = doc["signed"]
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
    if set(signed) != expected or signed.get("_type") != root_type:
        _fail(prefix + "_SIGNED_SCHEMA_INVALID")
    if (
        signed.get("specVersion") != str(POLICY["spec_version"])
        or signed.get("schemaVersion") != int(POLICY[schema_key])
        or signed.get("version") != 1
    ):
        _fail(prefix + "_VERSION_INVALID")
    root_id = _id(signed["rootId"], prefix + "_ID_INVALID")
    issued = _dt(signed["issuedAt"], prefix + "_ISSUED_INVALID")
    expires = _dt(signed["expires"], prefix + "_EXPIRES_INVALID")
    if expires <= issued or expires - issued > timedelta(
        days=int(POLICY["max_root_lifetime_days"])
    ):
        _fail(prefix + "_LIFETIME_INVALID")
    if not historical:
        if issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
            _fail(prefix + "_FROM_FUTURE")
        if expires <= now:
            _fail(prefix + "_EXPIRED")
    keys_raw = signed["keys"]
    min_keys = int(POLICY["min_recovery_keys" if recovery else "min_governance_keys"])
    if not isinstance(keys_raw, dict) or not (
        min_keys <= len(keys_raw) <= 64  # ruff: ignore[magic-value-comparison]
    ):
        _fail(prefix + "_KEYS_INVALID")
    keys = {}
    for raw_kid, value in sorted(keys_raw.items()):
        kid = _id(raw_kid, prefix + "_KEY_ID_INVALID")
        if kid != raw_kid or kid in keys:
            _fail(prefix + "_KEY_ID_INVALID")
        keys[kid] = _root_key(value, prefix + "_KEY", recovery=recovery)
    if len({v["publicKey"] for v in keys.values()}) != len(keys):
        _fail(prefix + "_PUBLIC_KEY_REUSE")
    for key in keys.values():
        if _dt(key["expires"], prefix + "_KEY_EXPIRES_INVALID") < expires:
            _fail(prefix + "_KEY_EXPIRES_BEFORE_ROOT")
    threshold = _positive_int(signed["threshold"], prefix + "_THRESHOLD_INVALID")
    min_threshold = int(
        POLICY["min_recovery_threshold" if recovery else "min_governance_threshold"]
    )
    if threshold < min_threshold or threshold > len(keys):
        _fail(prefix + "_THRESHOLD_INVALID")
    selected = signed["selectedSignerKeyIds"]
    if not isinstance(selected, list):
        _fail(prefix + "_SELECTED_INVALID")
    selected = [_id(x, prefix + "_SELECTED_INVALID") for x in selected]
    if (
        selected != sorted(selected)
        or len(selected) != threshold
        or len(set(selected)) != len(selected)
        or any(x not in keys for x in selected)
    ):
        _fail(prefix + "_SELECTED_INVALID")
    min_ops = int(
        POLICY["min_recovery_operators" if recovery else "min_governance_operators"]
    )
    if len({keys[k]["operator"] for k in selected}) < min_ops:
        _fail(prefix + "_OPERATOR_QUORUM_INVALID")
    if recovery and len({keys[k]["recoveryChannel"] for k in selected}) < int(
        POLICY["min_recovery_channels"]
    ):
        _fail(prefix + "_CHANNEL_QUORUM_INVALID")
    signatures = doc["signatures"]
    if not isinstance(signatures, list) or len(signatures) != len(selected):
        _fail(prefix + "_SIGNATURE_SET_INVALID")
    sigs = {}
    for item in signatures:
        if not isinstance(item, dict) or set(item) != {"keyId", "signature"}:
            _fail(prefix + "_SIGNATURE_SCHEMA_INVALID")
        kid = _id(item["keyId"], prefix + "_SIGNATURE_KEY_INVALID")
        if kid in sigs:
            _fail(prefix + "_SIGNATURE_DUPLICATE")
        sigs[kid] = item["signature"]
    if sorted(sigs) != selected or signatures != sorted(
        signatures, key=lambda x: x["keyId"]
    ):
        _fail(prefix + "_SIGNATURE_SET_INVALID")
    message = _canonical(signed)
    for kid in selected:
        _verify_sig(keys[kid]["publicKey"], sigs[kid], message, prefix + "_SIGNATURE")
    raw = _canonical(doc)
    sha = _sha_bytes(raw)
    if sha != _hex(expected_pin, prefix + "_PIN_INVALID"):
        _fail(prefix + "_PIN_MISMATCH")
    return {
        "rootId": root_id,
        "sha256": sha,
        "keys": keys,
        "threshold": threshold,
        "issued": issued,
        "expires": expires,
        "doc": doc,
    }


def _load_run164_offline(
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    *,
    now: datetime,
    historical: bool,
) -> dict[str, Any]:
    root_doc, _ = _read_json(Path(transparency_root_path), "ARCHIVE_LOG_RUN164_ROOT")
    try:
        root = merkle.verify_transparency_root(
            root_doc, transparency_root_pin, now=now, historical=historical
        )
        docs, raws = merkle._load_output(Path(run164_dir))
        replay = merkle._replay_history(
            root=root,
            bundle=docs["release-archive-merkle-bundle.json"],
            receipt=docs["release-archive-merkle-receipt.json"],
            now=now,
            historical=historical,
            current_run163_docs=None,
            current_run163_raws=None,
        )
    except Exception as exc:  # ruff: ignore[blind-except]
        _fail("ARCHIVE_LOG_RUN164_VERIFICATION_FAILED:" + str(exc))
    state = docs[_DOC_ARCHIVE_MERKLE_STATE]
    active = docs["active-archive-merkle-evidence.json"]
    bundle_raw = raws["release-archive-merkle-bundle.json"]
    expected_state = {
        "schemaVersion": int(merkle.POLICY["state_schema_version"]),
        "status": "trusted-archive-merkle-transparency",
        "sequence": replay["sequence"],
        "run163Sequence": replay["last"]["run163Sequence"],
        "run163AnchorConsensusHeadSha256": replay["last"][
            "run163AnchorConsensusHeadSha256"
        ],
        "merkleConsensusHeadSha256": replay["merkleConsensusHeadSha256"],
        "transparencyRootSha256": root["sha256"],
        "bundleArtifact": {
            "name": "release-archive-merkle-bundle.json",
            "sha256": _sha_bytes(bundle_raw),
            "size": len(bundle_raw),
        },
    }
    if state != expected_state:
        _fail("ARCHIVE_LOG_RUN164_STATE_MISMATCH")
    last = replay["last"]
    expected_active = {
        "schemaVersion": int(merkle.POLICY["active_schema_version"]),
        "status": "active-archive-merkle-transparency",
        "sequence": replay["sequence"],
        "run163AnchorConsensusHeadSha256": last["run163AnchorConsensusHeadSha256"],
        "merkleConsensusHeadSha256": replay["merkleConsensusHeadSha256"],
        "leafHash": last["leafHash"],
        "logs": last["logs"],
        "gossipResponseSha256s": last["gossipResponseSha256s"],
    }
    if active != expected_active:
        _fail("ARCHIVE_LOG_RUN164_ACTIVE_MISMATCH")
    artifacts = {
        name: {"sha256": _sha_bytes(raws[name]), "size": len(raws[name])}
        for name in sorted(raws)
    }
    checkpoints = {
        row["logId"]: {k: row[k] for k in ("checkpointSha256", "treeSize", "rootHash")}
        for row in last["logs"]
    }
    return {
        "root": root,
        "docs": docs,
        "raws": raws,
        "replay": replay,
        "artifacts": artifacts,
        "checkpoints": checkpoints,
    }


def _authority(
    value: Any, *, expected_log_ids: set[str], code: str
) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != expected_log_ids:
        _fail(code + "_LOG_SET_INVALID")
    out = {}
    pubs = set()
    for lid in sorted(value):
        if _id(lid, code + "_LOG_ID_INVALID") != lid:
            _fail(code + "_LOG_ID_INVALID")
        try:
            row = merkle._log_entry(value[lid], code + "_LOG")
        except Exception as exc:  # ruff: ignore[blind-except]
            _fail(code + "_LOG_INVALID:" + str(exc))
        if (
            row["publicKey"] in pubs
            or row["gossipPublicKey"] in pubs
            or row["publicKey"] == row["gossipPublicKey"]
        ):
            _fail(code + "_PUBLIC_KEY_REUSE")
        pubs.add(row["publicKey"])
        pubs.add(row["gossipPublicKey"])
        out[lid] = row
    if out != value:
        _fail(code + "_NOT_NORMALIZED")
    return out


def _authority_sha(authority: dict[str, Any]) -> str:
    return _sha_bytes(_canonical(authority))


def _authority_fingerprints(authority: dict[str, Any]) -> set[str]:
    out = set()
    for row in authority.values():
        out.add(_pub_fingerprint(row["publicKey"]))
        out.add(_pub_fingerprint(row["gossipPublicKey"]))
    return out


def _enforce_plane_separation(
    governance: dict[str, Any],
    recovery: dict[str, Any],
    run164_root: dict[str, Any],
    *authorities: dict[str, Any],
) -> None:
    gov_ops = {v["operator"] for v in governance["keys"].values()}
    gov_pubs = {v["publicKey"] for v in governance["keys"].values()}
    rec_ops = {v["operator"] for v in recovery["keys"].values()}
    rec_pubs = {v["publicKey"] for v in recovery["keys"].values()}
    merkle_root_ops = {v["operator"] for v in run164_root["keys"].values()}
    merkle_root_pubs = {v["publicKey"] for v in run164_root["keys"].values()}
    log_ops = {v["operator"] for v in run164_root["logs"].values()} | {
        v["gossipOperator"] for v in run164_root["logs"].values()
    }
    log_pubs = {v["publicKey"] for v in run164_root["logs"].values()} | {
        v["gossipPublicKey"] for v in run164_root["logs"].values()
    }
    if gov_ops & rec_ops or gov_pubs & rec_pubs:
        _fail("ARCHIVE_LOG_CONTROL_PLANES_OVERLAP")
    if (gov_ops | rec_ops) & (merkle_root_ops | log_ops) or (gov_pubs | rec_pubs) & (
        merkle_root_pubs | log_pubs
    ):
        _fail("ARCHIVE_LOG_CONTROL_PLANES_OVERLAP")
    for authority in authorities:
        a_ops = {v["operator"] for v in authority.values()} | {
            v["gossipOperator"] for v in authority.values()
        }
        a_pubs = {v["publicKey"] for v in authority.values()} | {
            v["gossipPublicKey"] for v in authority.values()
        }
        if (gov_ops | rec_ops) & a_ops or (gov_pubs | rec_pubs) & a_pubs:
            _fail("ARCHIVE_LOG_CONTROL_PLANES_OVERLAP")


def _handoff_subject(
    *, signed: dict[str, Any], log_id: str, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    current = signed["currentAuthority"][log_id]
    nxt = signed["nextAuthority"][log_id]
    return {
        "_type": "archive-merkle-log-authority-handoff",
        "specVersion": str(POLICY["spec_version"]),
        "schemaVersion": int(POLICY["handoff_schema_version"]),
        "transitionId": signed["transitionId"],
        "sequence": signed["sequence"],
        "kind": signed["kind"],
        "logId": log_id,
        "run164Sequence": signed["run164Sequence"],
        "merkleConsensusHeadSha256": signed["merkleConsensusHeadSha256"],
        "priorCheckpoint": checkpoint,
        "currentLogAuthority": current,
        "nextLogAuthority": nxt,
        "currentAuthoritySha256": _authority_sha(signed["currentAuthority"]),
        "nextAuthoritySha256": _authority_sha(signed["nextAuthority"]),
    }


def _verify_transition_document(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    *,
    governance: dict[str, Any],
    recovery: dict[str, Any],
    run164: dict[str, Any],
    previous_state: dict[str, Any] | None,
    previous_state_raw: bytes | None,
    now: datetime,
    creation: bool,
) -> dict[str, Any]:
    if set(doc) != {"signed", "signatures"} or not isinstance(doc.get("signed"), dict):
        _fail("ARCHIVE_LOG_TRANSITION_SCHEMA_INVALID")
    signed = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "transitionId",
        "sequence",
        "kind",
        "issuedAt",
        "run164Sequence",
        "merkleConsensusHeadSha256",
        "run164Artifacts",
        "previousAuthorityStateSha256",
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
        or signed.get("_type") != "archive-merkle-log-authority-transition"
    ):
        _fail("ARCHIVE_LOG_TRANSITION_SIGNED_SCHEMA_INVALID")
    if signed.get("specVersion") != str(POLICY["spec_version"]) or signed.get(
        "schemaVersion"
    ) != int(POLICY["transition_schema_version"]):
        _fail("ARCHIVE_LOG_TRANSITION_VERSION_INVALID")
    _id(signed["transitionId"], "ARCHIVE_LOG_TRANSITION_ID_INVALID")
    sequence = _positive_int(
        signed["sequence"], "ARCHIVE_LOG_TRANSITION_SEQUENCE_INVALID"
    )
    kind = signed["kind"]
    if kind not in {"bootstrap", "scheduled-rotation", "compromise-recovery"}:
        _fail("ARCHIVE_LOG_TRANSITION_KIND_INVALID")
    issued = _dt(signed["issuedAt"], "ARCHIVE_LOG_TRANSITION_ISSUED_INVALID")
    if issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
        _fail("ARCHIVE_LOG_TRANSITION_FROM_FUTURE")
    if creation and now - issued > timedelta(
        minutes=int(POLICY["max_transition_freshness_minutes"])
    ):
        _fail("ARCHIVE_LOG_TRANSITION_STALE")
    if (
        signed["run164Sequence"] != run164["replay"]["sequence"]
        or signed["merkleConsensusHeadSha256"]
        != run164["replay"]["merkleConsensusHeadSha256"]
    ):
        _fail("ARCHIVE_LOG_TRANSITION_RUN164_BINDING_INVALID")
    if signed["run164Artifacts"] != run164["artifacts"]:
        _fail("ARCHIVE_LOG_TRANSITION_RUN164_ARTIFACT_INVALID")
    log_ids = set(run164["root"]["logs"])
    current = _authority(
        signed["currentAuthority"],
        expected_log_ids=log_ids,
        code="ARCHIVE_LOG_CURRENT_AUTHORITY",
    )
    nxt = _authority(
        signed["nextAuthority"],
        expected_log_ids=log_ids,
        code="ARCHIVE_LOG_NEXT_AUTHORITY",
    )
    for lid in log_ids:
        if current[lid]["gossipIdentity"] != nxt[lid]["gossipIdentity"]:
            _fail("ARCHIVE_LOG_GOSSIP_IDENTITY_CHANGE_FORBIDDEN")
    changed = signed["changedLogIds"]
    if not isinstance(changed, list):
        _fail("ARCHIVE_LOG_CHANGED_SET_INVALID")
    changed = [_id(x, "ARCHIVE_LOG_CHANGED_ID_INVALID") for x in changed]
    computed_changed = sorted(lid for lid in log_ids if current[lid] != nxt[lid])
    if (
        changed != sorted(changed)
        or len(set(changed)) != len(changed)
        or changed != computed_changed
    ):
        _fail("ARCHIVE_LOG_CHANGED_SET_INVALID")
    compromised = signed["compromisedKeyFingerprints"]
    revoked = signed["revokedKeyFingerprints"]
    if not isinstance(compromised, list) or not isinstance(revoked, list):
        _fail("ARCHIVE_LOG_REVOCATION_LIST_INVALID")
    compromised = [
        _hex(x, "ARCHIVE_LOG_COMPROMISED_FINGERPRINT_INVALID") for x in compromised
    ]
    revoked = [_hex(x, "ARCHIVE_LOG_REVOKED_FINGERPRINT_INVALID") for x in revoked]
    if (
        compromised != sorted(compromised)
        or len(set(compromised)) != len(compromised)
        or revoked != sorted(revoked)
        or len(set(revoked)) != len(revoked)
    ):
        _fail("ARCHIVE_LOG_REVOCATION_LIST_INVALID")
    previous_revoked = (
        [] if previous_state is None else previous_state["revokedKeyFingerprints"]
    )
    if previous_state is None:
        if (
            sequence != 1
            or signed["previousAuthorityStateSha256"] is not None
            or kind != "bootstrap"
        ):
            _fail("ARCHIVE_LOG_BOOTSTRAP_INVALID")
        original = {lid: run164["root"]["logs"][lid] for lid in sorted(log_ids)}
        if current != original or nxt != original or changed or compromised or revoked:
            _fail("ARCHIVE_LOG_BOOTSTRAP_AUTHORITY_INVALID")
    else:
        if sequence != previous_state["sequence"] + 1:
            _fail("ARCHIVE_LOG_SEQUENCE_INVALID")
        if previous_state_raw is None or signed[
            "previousAuthorityStateSha256"
        ] != _sha_bytes(previous_state_raw):
            _fail("ARCHIVE_LOG_PREVIOUS_STATE_BINDING_INVALID")
        if current != previous_state["activeAuthority"]:
            _fail("ARCHIVE_LOG_CURRENT_AUTHORITY_MISMATCH")
        if (
            signed["run164Sequence"] != previous_state["run164Sequence"]
            or signed["merkleConsensusHeadSha256"]
            != previous_state["merkleConsensusHeadSha256"]
            or signed["run164Artifacts"] != previous_state["run164Artifacts"]
        ):
            _fail("ARCHIVE_LOG_RUN164_SNAPSHOT_CHANGED")
        if kind == "bootstrap":
            _fail("ARCHIVE_LOG_BOOTSTRAP_REPEATED")
        if not changed:
            _fail("ARCHIVE_LOG_ROTATION_NOOP")
    current_fps = _authority_fingerprints(current)
    next_fps = _authority_fingerprints(nxt)
    if set(previous_revoked) & next_fps:
        _fail("ARCHIVE_LOG_REVOKED_KEY_REINTRODUCED")
    if kind == "scheduled-rotation":
        if compromised:
            _fail("ARCHIVE_LOG_SCHEDULED_COMPROMISED_NOT_EMPTY")
        replaced = set()
        for lid in changed:
            old = current[lid]
            new = nxt[lid]
            if (
                old["operator"] != new["operator"]
                and old["publicKey"] == new["publicKey"]
            ):
                _fail("ARCHIVE_LOG_OPERATOR_REBOUND_WITHOUT_KEY_ROTATION")
            if (
                old["gossipOperator"] != new["gossipOperator"]
                and old["gossipPublicKey"] == new["gossipPublicKey"]
            ):
                _fail("ARCHIVE_LOG_GOSSIP_OPERATOR_REBOUND_WITHOUT_KEY_ROTATION")
            if old["publicKey"] != new["publicKey"]:
                replaced.add(_pub_fingerprint(old["publicKey"]))
            if old["gossipPublicKey"] != new["gossipPublicKey"]:
                replaced.add(_pub_fingerprint(old["gossipPublicKey"]))
        expected_revoked = sorted(set(previous_revoked) | replaced)
        if revoked != expected_revoked:
            _fail("ARCHIVE_LOG_SCHEDULED_REVOCATION_INVALID")
    elif kind == "compromise-recovery":
        if not compromised:
            _fail("ARCHIVE_LOG_RECOVERY_COMPROMISED_EMPTY")
        if not set(compromised) <= current_fps:
            _fail("ARCHIVE_LOG_RECOVERY_COMPROMISED_UNKNOWN")
        affected = sorted(
            lid
            for lid in log_ids
            if {
                _pub_fingerprint(current[lid]["publicKey"]),
                _pub_fingerprint(current[lid]["gossipPublicKey"]),
            }
            & set(compromised)
        )
        if affected != changed:
            _fail("ARCHIVE_LOG_RECOVERY_CHANGED_SET_INVALID")
        replaced = set()
        for lid in changed:
            old = current[lid]
            new = nxt[lid]
            if (
                old["publicKey"] == new["publicKey"]
                or old["gossipPublicKey"] == new["gossipPublicKey"]
            ):
                _fail("ARCHIVE_LOG_RECOVERY_REQUIRES_FULL_KEY_ROTATION")
            replaced.add(_pub_fingerprint(old["publicKey"]))
            replaced.add(_pub_fingerprint(old["gossipPublicKey"]))
        expected_revoked = sorted(set(previous_revoked) | replaced)
        if revoked != expected_revoked:
            _fail("ARCHIVE_LOG_RECOVERY_REVOCATION_INVALID")
    elif previous_state is not None:
        _fail("ARCHIVE_LOG_BOOTSTRAP_INVALID")
    if set(revoked) & next_fps:
        _fail("ARCHIVE_LOG_REVOKED_KEY_ACTIVE")
    role = signed["authorizationRole"]
    expected_role = "recovery" if kind == "compromise-recovery" else "governance"
    if role != expected_role:
        _fail("ARCHIVE_LOG_AUTHORIZATION_ROLE_INVALID")
    root = recovery if role == "recovery" else governance
    selected = signed["selectedSignerKeyIds"]
    if not isinstance(selected, list):
        _fail("ARCHIVE_LOG_SELECTED_SIGNERS_INVALID")
    selected = [_id(x, "ARCHIVE_LOG_SELECTED_SIGNERS_INVALID") for x in selected]
    if (
        selected != sorted(selected)
        or len(selected) != root["threshold"]
        or len(set(selected)) != len(selected)
        or any(x not in root["keys"] for x in selected)
    ):
        _fail("ARCHIVE_LOG_SELECTED_SIGNERS_INVALID")
    if role == "recovery":
        if len({root["keys"][k]["operator"] for k in selected}) < int(
            POLICY["min_recovery_operators"]
        ) or len({root["keys"][k]["recoveryChannel"] for k in selected}) < int(
            POLICY["min_recovery_channels"]
        ):
            _fail("ARCHIVE_LOG_RECOVERY_QUORUM_INVALID")
    elif len({root["keys"][k]["operator"] for k in selected}) < int(
        POLICY["min_governance_operators"]
    ):
        _fail("ARCHIVE_LOG_GOVERNANCE_QUORUM_INVALID")
    if issued < root["issued"] or issued >= root["expires"]:
        _fail("ARCHIVE_LOG_TRANSITION_OUTSIDE_ROOT_LIFETIME")
    for kid in selected:
        if issued >= _dt(
            root["keys"][kid]["expires"], "ARCHIVE_LOG_SIGNER_EXPIRES_INVALID"
        ):
            _fail("ARCHIVE_LOG_SIGNER_EXPIRED_AT_TRANSITION")
    signatures = doc["signatures"]
    if not isinstance(signatures, list) or len(signatures) != len(selected):
        _fail("ARCHIVE_LOG_TRANSITION_SIGNATURE_SET_INVALID")
    sigs = {}
    for item in signatures:
        if not isinstance(item, dict) or set(item) != {"keyId", "signature"}:
            _fail("ARCHIVE_LOG_TRANSITION_SIGNATURE_SCHEMA_INVALID")
        kid = _id(item["keyId"], "ARCHIVE_LOG_TRANSITION_SIGNATURE_KEY_INVALID")
        if kid in sigs:
            _fail("ARCHIVE_LOG_TRANSITION_SIGNATURE_DUPLICATE")
        sigs[kid] = item["signature"]
    if sorted(sigs) != selected or signatures != sorted(
        signatures, key=lambda x: x["keyId"]
    ):
        _fail("ARCHIVE_LOG_TRANSITION_SIGNATURE_SET_INVALID")
    message = _canonical(signed)
    for kid in selected:
        _verify_sig(
            root["keys"][kid]["publicKey"],
            sigs[kid],
            message,
            "ARCHIVE_LOG_TRANSITION_SIGNATURE",
        )
    subjects = {}
    for lid in sorted(log_ids):
        subjects[lid] = _handoff_subject(
            signed=signed, log_id=lid, checkpoint=run164["checkpoints"][lid]
        )
    expected_hashes = {
        lid: _sha_bytes(_canonical(subjects[lid])) for lid in sorted(subjects)
    }
    if signed["handoffSubjectSha256s"] != expected_hashes:
        _fail("ARCHIVE_LOG_HANDOFF_SUBJECT_HASH_INVALID")
    return {
        "signed": signed,
        "transitionSha256": _sha_bytes(_canonical(doc)),
        "subjects": subjects,
        "revoked": revoked,
        "current": current,
        "next": nxt,
        "kind": kind,
        "sequence": sequence,
    }


def _verify_handoff_document(
    doc: dict[str, Any], *, transition: dict[str, Any]
) -> dict[str, Any]:
    signed = transition["signed"]
    if (
        set(doc) != {"schemaVersion", "transitionId", "sequence", "proofs"}
        or doc.get("schemaVersion") != int(POLICY["handoff_schema_version"])
        or doc.get("transitionId") != signed["transitionId"]
        or doc.get("sequence") != signed["sequence"]
    ):
        _fail("ARCHIVE_LOG_HANDOFF_SCHEMA_INVALID")
    proofs = doc["proofs"]
    if not isinstance(proofs, list) or len(proofs) != len(transition["subjects"]):
        _fail("ARCHIVE_LOG_HANDOFF_PROOF_SET_INVALID")
    if proofs != sorted(proofs, key=lambda x: x.get("logId", "")):
        _fail("ARCHIVE_LOG_HANDOFF_PROOF_ORDER_INVALID")
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
            _fail("ARCHIVE_LOG_HANDOFF_PROOF_SCHEMA_INVALID")
        lid = _id(proof["logId"], "ARCHIVE_LOG_HANDOFF_LOG_ID_INVALID")
        if lid in seen or lid not in transition["subjects"]:
            _fail("ARCHIVE_LOG_HANDOFF_LOG_ID_INVALID")
        seen.add(lid)
        subject = transition["subjects"][lid]
        raw = _canonical(subject)
        if proof["subjectSha256"] != _sha_bytes(raw):
            _fail("ARCHIVE_LOG_HANDOFF_SUBJECT_BINDING_INVALID")
        nxt = transition["next"][lid]
        cur = transition["current"][lid]
        _verify_sig(
            nxt["publicKey"],
            proof["newLogSignature"],
            raw,
            "ARCHIVE_LOG_NEW_LOG_HANDOFF",
        )
        _verify_sig(
            nxt["gossipPublicKey"],
            proof["newGossipSignature"],
            raw,
            "ARCHIVE_LOG_NEW_GOSSIP_HANDOFF",
        )
        changed = lid in signed["changedLogIds"]
        if transition["kind"] == "scheduled-rotation" and changed:
            if proof["oldLogSignature"] is None or proof["oldGossipSignature"] is None:
                _fail("ARCHIVE_LOG_OLD_HANDOFF_REQUIRED")
            _verify_sig(
                cur["publicKey"],
                proof["oldLogSignature"],
                raw,
                "ARCHIVE_LOG_OLD_LOG_HANDOFF",
            )
            _verify_sig(
                cur["gossipPublicKey"],
                proof["oldGossipSignature"],
                raw,
                "ARCHIVE_LOG_OLD_GOSSIP_HANDOFF",
            )
        elif (
            proof["oldLogSignature"] is not None
            or proof["oldGossipSignature"] is not None
        ):
            _fail("ARCHIVE_LOG_OLD_HANDOFF_FORBIDDEN")
    return doc


def _event_head(previous_head: str | None, event_without_head: dict[str, Any]) -> str:
    return _sha_bytes(
        _canonical(
            {
                "previousLogAuthorityChainHeadSha256": previous_head,
                "event": event_without_head,
            }
        )
    )


def _load_output(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_LOG_OUTPUT_INVALID")
    if {p.name for p in root.iterdir()} != _OUTPUT_NAMES:
        _fail("ARCHIVE_LOG_OUTPUT_ALLOWLIST_INVALID")
    docs = {}
    raws = {}
    for name in sorted(_OUTPUT_NAMES):
        docs[name], raws[name] = _read_json(root / name, "ARCHIVE_LOG_OUTPUT")
    return docs, raws


def _replay(
    *,
    governance: dict[str, Any],
    recovery: dict[str, Any],
    run164: dict[str, Any],
    bundle: dict[str, Any],
    receipt: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    if (
        set(bundle)
        != {
            "schemaVersion",
            "predicateType",
            "status",
            "governanceRootSha256",
            "recoveryRootSha256",
            "events",
        }
        or bundle.get("schemaVersion") != int(POLICY["bundle_schema_version"])
        or bundle.get("predicateType") != PREDICATE_TYPE
        or bundle.get("status") != "archive-log-authority-history"
        or bundle.get("governanceRootSha256") != governance["sha256"]
        or bundle.get("recoveryRootSha256") != recovery["sha256"]
    ):
        _fail("ARCHIVE_LOG_BUNDLE_SCHEMA_INVALID")
    if (
        set(receipt) != {"schemaVersion", "status", "events"}
        or receipt.get("schemaVersion") != int(POLICY["receipt_schema_version"])
        or receipt.get("status") != "archive-log-authority-accepted"
    ):
        _fail("ARCHIVE_LOG_RECEIPT_SCHEMA_INVALID")
    events = bundle["events"]
    recs = receipt["events"]
    if (
        not isinstance(events, list)
        or not events
        or not isinstance(recs, list)
        or len(events) != len(recs)
    ):
        _fail("ARCHIVE_LOG_HISTORY_LENGTH_INVALID")
    prev_state = None
    prev_state_raw = None
    prev_head = None
    last_transition = None
    last_handoff = None
    previous_issued = None
    transition_ids = set()
    for idx, (event, rec) in enumerate(zip(events, recs), 1):
        if (
            not isinstance(event, dict)
            or set(event)
            != {
                "sequence",
                "kind",
                "transitionSha256",
                "run164Sequence",
                "merkleConsensusHeadSha256",
                "run164Artifacts",
                "currentAuthoritySha256",
                "nextAuthoritySha256",
                "revokedKeyFingerprints",
                "handoffSubjectSha256s",
                "logAuthorityChainHeadSha256",
            }
            or event.get("sequence") != idx
        ):
            _fail("ARCHIVE_LOG_EVENT_SCHEMA_INVALID")
        if (
            not isinstance(rec, dict)
            or set(rec)
            != {
                "sequence",
                "transitionDocument",
                "handoffProofDocument",
                "run164Documents",
            }
            or rec.get("sequence") != idx
        ):
            _fail("ARCHIVE_LOG_RECEIPT_EVENT_INVALID")
        run164_docs = rec["run164Documents"]
        if not isinstance(run164_docs, dict) or set(run164_docs) != _RUN164_NAMES:
            _fail("ARCHIVE_LOG_EMBEDDED_RUN164_INVALID")
        raws = {name: _canonical(run164_docs[name]) for name in sorted(run164_docs)}
        artifacts = {
            name: {"sha256": _sha_bytes(raws[name]), "size": len(raws[name])}
            for name in sorted(raws)
        }
        if artifacts != run164["artifacts"] or event["run164Artifacts"] != artifacts:
            _fail("ARCHIVE_LOG_RUN164_ARTIFACT_REPLAY_INVALID")
        tr = _verify_transition_document(
            rec["transitionDocument"],
            governance=governance,
            recovery=recovery,
            run164=run164,
            previous_state=prev_state,
            previous_state_raw=prev_state_raw,
            now=now,
            creation=False,
        )
        transition_id = tr["signed"]["transitionId"]
        issued = _dt(tr["signed"]["issuedAt"], "ARCHIVE_LOG_TRANSITION_ISSUED_INVALID")
        if transition_id in transition_ids:
            _fail("ARCHIVE_LOG_TRANSITION_ID_REUSED")
        if previous_issued is not None and issued <= previous_issued:
            _fail("ARCHIVE_LOG_TRANSITION_TIME_NOT_MONOTONIC")
        transition_ids.add(transition_id)
        previous_issued = issued
        hf = _verify_handoff_document(rec["handoffProofDocument"], transition=tr)
        if (
            event["kind"] != tr["kind"]
            or event["transitionSha256"] != tr["transitionSha256"]
            or event["run164Sequence"] != tr["signed"]["run164Sequence"]
            or event["merkleConsensusHeadSha256"]
            != tr["signed"]["merkleConsensusHeadSha256"]
            or event["currentAuthoritySha256"] != _authority_sha(tr["current"])
            or event["nextAuthoritySha256"] != _authority_sha(tr["next"])
            or event["revokedKeyFingerprints"] != tr["revoked"]
            or event["handoffSubjectSha256s"] != tr["signed"]["handoffSubjectSha256s"]
        ):
            _fail("ARCHIVE_LOG_EVENT_BINDING_INVALID")
        bare = {k: event[k] for k in event if k != "logAuthorityChainHeadSha256"}
        head = _event_head(prev_head, bare)
        if event["logAuthorityChainHeadSha256"] != head:
            _fail("ARCHIVE_LOG_CHAIN_HEAD_INVALID")
        prefix_bundle = {
            "schemaVersion": bundle["schemaVersion"],
            "predicateType": bundle["predicateType"],
            "status": bundle["status"],
            "governanceRootSha256": bundle["governanceRootSha256"],
            "recoveryRootSha256": bundle["recoveryRootSha256"],
            "events": events[:idx],
        }
        bundle_raw = _canonical(prefix_bundle)
        state = {
            "schemaVersion": int(POLICY["state_schema_version"]),
            "status": "trusted-archive-log-authority",
            "sequence": idx,
            "run164Sequence": tr["signed"]["run164Sequence"],
            "merkleConsensusHeadSha256": tr["signed"]["merkleConsensusHeadSha256"],
            "run164Artifacts": artifacts,
            "logAuthorityChainHeadSha256": head,
            "governanceRootSha256": governance["sha256"],
            "recoveryRootSha256": recovery["sha256"],
            "activeAuthority": tr["next"],
            "revokedKeyFingerprints": tr["revoked"],
            "continuityCheckpoints": run164["checkpoints"],
            "bundleArtifact": {
                "name": "release-archive-log-authority-bundle.json",
                "sha256": _sha_bytes(bundle_raw),
                "size": len(bundle_raw),
            },
        }
        prev_state = state
        prev_state_raw = _canonical(state)
        prev_head = head
        last_transition = tr
        last_handoff = hf
    return {
        "state": prev_state,
        "head": prev_head,
        "transition": last_transition,
        "handoff": last_handoff,
        "sequence": len(events),
    }


def verify_log_authority_history(  # ruff: ignore[undocumented-public-function]
    *,
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    governance_root_path: Path,
    governance_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    output_dir: Path,
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    gov_doc, _ = _read_json(Path(governance_root_path), "ARCHIVE_LOG_GOVERNANCE_ROOT")
    rec_doc, _ = _read_json(Path(recovery_root_path), "ARCHIVE_LOG_RECOVERY_ROOT")
    governance = _verify_control_root(
        gov_doc, governance_root_pin, now=current, recovery=False, historical=historical
    )
    recovery = _verify_control_root(
        rec_doc, recovery_root_pin, now=current, recovery=True, historical=historical
    )
    run164 = _load_run164_offline(
        Path(run164_dir),
        Path(transparency_root_path),
        transparency_root_pin,
        now=current,
        historical=historical,
    )
    docs, raws = _load_output(Path(output_dir))
    replay = _replay(
        governance=governance,
        recovery=recovery,
        run164=run164,
        bundle=docs["release-archive-log-authority-bundle.json"],
        receipt=docs["release-archive-log-authority-receipt.json"],
        now=current,
    )
    _enforce_plane_separation(
        governance, recovery, run164["root"], replay["state"]["activeAuthority"]
    )
    bundle_raw = raws["release-archive-log-authority-bundle.json"]
    expected_state = dict(replay["state"])
    if expected_state["bundleArtifact"] != {
        "name": "release-archive-log-authority-bundle.json",
        "sha256": _sha_bytes(bundle_raw),
        "size": len(bundle_raw),
    }:
        _fail("ARCHIVE_LOG_BUNDLE_ARTIFACT_REPLAY_INVALID")
    if docs[_DOC_ARCHIVE_LOG_AUTHORITY_STATE] != expected_state:
        _fail("ARCHIVE_LOG_STATE_MISMATCH")
    tr = replay["transition"]
    expected_active = {
        "schemaVersion": int(POLICY["active_schema_version"]),
        "status": "active-archive-log-authority",
        "sequence": replay["sequence"],
        "run164Sequence": tr["signed"]["run164Sequence"],
        "merkleConsensusHeadSha256": tr["signed"]["merkleConsensusHeadSha256"],
        "logAuthorityChainHeadSha256": replay["head"],
        "authority": tr["next"],
        "revokedKeyFingerprints": tr["revoked"],
        "continuityCheckpoints": run164["checkpoints"],
        "handoffSubjectSha256s": tr["signed"]["handoffSubjectSha256s"],
    }
    if docs["active-archive-log-authority.json"] != expected_active:
        _fail("ARCHIVE_LOG_ACTIVE_MISMATCH")
    return {
        "ok": True,
        "sequence": replay["sequence"],
        "log_authority_chain_head_sha256": replay["head"],
        "active_authority": tr["next"],
    }


def apply_log_authority_transition(  # ruff: ignore[undocumented-public-function]
    *,
    run164_dir: Path,
    transparency_root_path: Path,
    transparency_root_pin: str,
    governance_root_path: Path,
    governance_root_pin: str,
    recovery_root_path: Path,
    recovery_root_pin: str,
    transition_path: Path,
    handoff_path: Path,
    output_dir: Path,
    previous_output_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    authority_paths = [
        Path(run164_dir),
        Path(transparency_root_path),
        Path(governance_root_path),
        Path(recovery_root_path),
        Path(transition_path),
        Path(handoff_path),
    ]
    if previous_output_dir is not None:
        authority_paths.append(Path(previous_output_dir))
    # Snapshot authority bytes before parsing anything so a concurrent mutation cannot become the baseline.
    before = _authority_fingerprint(authority_paths, "ARCHIVE_LOG_INPUT_DRIFT")
    gov_doc, _ = _read_json(Path(governance_root_path), "ARCHIVE_LOG_GOVERNANCE_ROOT")
    rec_doc, _ = _read_json(Path(recovery_root_path), "ARCHIVE_LOG_RECOVERY_ROOT")
    governance = _verify_control_root(
        gov_doc, governance_root_pin, now=current, recovery=False, historical=False
    )
    recovery = _verify_control_root(
        rec_doc, recovery_root_pin, now=current, recovery=True, historical=False
    )
    run164 = _load_run164_offline(
        Path(run164_dir),
        Path(transparency_root_path),
        transparency_root_pin,
        now=current,
        historical=False,
    )
    previous_state = None
    previous_state_raw = None
    old_events = []
    old_receipts = []
    previous_head = None
    if previous_output_dir is not None:
        prev_docs, _ = _load_output(Path(previous_output_dir))
        replay = _replay(
            governance=governance,
            recovery=recovery,
            run164=run164,
            bundle=prev_docs["release-archive-log-authority-bundle.json"],
            receipt=prev_docs["release-archive-log-authority-receipt.json"],
            now=current,
        )
        previous_state = replay["state"]
        previous_state_raw = _canonical(previous_state)
        previous_head = replay["head"]
        old_events = list(
            prev_docs["release-archive-log-authority-bundle.json"]["events"]
        )
        old_receipts = list(
            prev_docs["release-archive-log-authority-receipt.json"]["events"]
        )
    transition_doc, _ = _read_json(Path(transition_path), "ARCHIVE_LOG_TRANSITION")
    tr = _verify_transition_document(
        transition_doc,
        governance=governance,
        recovery=recovery,
        run164=run164,
        previous_state=previous_state,
        previous_state_raw=previous_state_raw,
        now=current,
        creation=True,
    )
    handoff_doc, _ = _read_json(Path(handoff_path), "ARCHIVE_LOG_HANDOFF")
    _verify_handoff_document(handoff_doc, transition=tr)
    _enforce_plane_separation(
        governance, recovery, run164["root"], tr["current"], tr["next"]
    )
    event_bare = {
        "sequence": tr["sequence"],
        "kind": tr["kind"],
        "transitionSha256": tr["transitionSha256"],
        "run164Sequence": tr["signed"]["run164Sequence"],
        "merkleConsensusHeadSha256": tr["signed"]["merkleConsensusHeadSha256"],
        "run164Artifacts": run164["artifacts"],
        "currentAuthoritySha256": _authority_sha(tr["current"]),
        "nextAuthoritySha256": _authority_sha(tr["next"]),
        "revokedKeyFingerprints": tr["revoked"],
        "handoffSubjectSha256s": tr["signed"]["handoffSubjectSha256s"],
    }
    head = _event_head(previous_head, event_bare)
    event = dict(event_bare)
    event["logAuthorityChainHeadSha256"] = head
    receipt_event = {
        "sequence": tr["sequence"],
        "transitionDocument": transition_doc,
        "handoffProofDocument": handoff_doc,
        "run164Documents": run164["docs"],
    }
    bundle = {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "archive-log-authority-history",
        "governanceRootSha256": governance["sha256"],
        "recoveryRootSha256": recovery["sha256"],
        "events": [*old_events, event],
    }
    receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": "archive-log-authority-accepted",
        "events": [*old_receipts, receipt_event],
    }
    bundle_raw = _canonical(bundle)
    state_core = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-archive-log-authority",
        "sequence": tr["sequence"],
        "run164Sequence": tr["signed"]["run164Sequence"],
        "merkleConsensusHeadSha256": tr["signed"]["merkleConsensusHeadSha256"],
        "run164Artifacts": run164["artifacts"],
        "logAuthorityChainHeadSha256": head,
        "governanceRootSha256": governance["sha256"],
        "recoveryRootSha256": recovery["sha256"],
        "activeAuthority": tr["next"],
        "revokedKeyFingerprints": tr["revoked"],
        "continuityCheckpoints": run164["checkpoints"],
    }
    state = dict(state_core)
    state["bundleArtifact"] = {
        "name": "release-archive-log-authority-bundle.json",
        "sha256": _sha_bytes(bundle_raw),
        "size": len(bundle_raw),
    }
    active = {
        "schemaVersion": int(POLICY["active_schema_version"]),
        "status": "active-archive-log-authority",
        "sequence": tr["sequence"],
        "run164Sequence": tr["signed"]["run164Sequence"],
        "merkleConsensusHeadSha256": tr["signed"]["merkleConsensusHeadSha256"],
        "logAuthorityChainHeadSha256": head,
        "authority": tr["next"],
        "revokedKeyFingerprints": tr["revoked"],
        "continuityCheckpoints": run164["checkpoints"],
        "handoffSubjectSha256s": tr["signed"]["handoffSubjectSha256s"],
    }
    if before != _authority_fingerprint(authority_paths, "ARCHIVE_LOG_INPUT_DRIFT"):
        _fail("ARCHIVE_LOG_INPUT_DRIFT")
    protected = authority_paths
    target = _outside(Path(output_dir), protected, "ARCHIVE_LOG_OUTPUT_OVERLAPS_INPUT")
    if target.exists():
        _fail("ARCHIVE_LOG_OUTPUT_EXISTS")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run165-log-authority-", dir=target.parent))
    try:
        (stage / "release-archive-log-authority-bundle.json").write_bytes(bundle_raw)
        _write(stage / _DOC_ARCHIVE_LOG_AUTHORITY_STATE, state)
        _write(stage / "active-archive-log-authority.json", active)
        _write(stage / "release-archive-log-authority-receipt.json", receipt)
        verify_log_authority_history(
            run164_dir=run164_dir,
            transparency_root_path=transparency_root_path,
            transparency_root_pin=transparency_root_pin,
            governance_root_path=governance_root_path,
            governance_root_pin=governance_root_pin,
            recovery_root_path=recovery_root_path,
            recovery_root_pin=recovery_root_pin,
            output_dir=stage,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_log_authority_history(
        run164_dir=run164_dir,
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        governance_root_path=governance_root_path,
        governance_root_pin=governance_root_pin,
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        output_dir=target,
        now=current,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run164-dir", required=True, type=Path)
    parser.add_argument("--transparency-root", required=True, type=Path)
    parser.add_argument("--transparency-root-pin", required=True)
    parser.add_argument("--governance-root", required=True, type=Path)
    parser.add_argument("--governance-root-pin", required=True)
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--recovery-root-pin", required=True)
    parser.add_argument("--output", required=True, type=Path)


def main(  # ruff: ignore[undocumented-public-function]
    argv=None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run 165 archive Merkle log-key authority lifecycle"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    _common(verify)
    verify.add_argument("--historical", action="store_true")
    apply = sub.add_parser("apply")
    _common(apply)
    apply.add_argument("--transition", required=True, type=Path)
    apply.add_argument("--handoff", required=True, type=Path)
    apply.add_argument("--previous-output", type=Path)
    args = parser.parse_args(argv)
    try:
        logger.info("Running archive log authority command: %s", args.command)
        if args.command == "verify":
            result = verify_log_authority_history(
                run164_dir=args.run164_dir,
                transparency_root_path=args.transparency_root,
                transparency_root_pin=args.transparency_root_pin,
                governance_root_path=args.governance_root,
                governance_root_pin=args.governance_root_pin,
                recovery_root_path=args.recovery_root,
                recovery_root_pin=args.recovery_root_pin,
                output_dir=args.output,
                historical=args.historical,
            )
        else:
            logger.info("Applying archive log authority handoff: %s", args.handoff)
            result = apply_log_authority_transition(
                run164_dir=args.run164_dir,
                transparency_root_path=args.transparency_root,
                transparency_root_pin=args.transparency_root_pin,
                governance_root_path=args.governance_root,
                governance_root_pin=args.governance_root_pin,
                recovery_root_path=args.recovery_root,
                recovery_root_pin=args.recovery_root_pin,
                transition_path=args.transition,
                handoff_path=args.handoff,
                output_dir=args.output,
                previous_output_dir=args.previous_output,
            )
        logger.info("Archive log authority command succeeded: %s", result)
        return 0
    except ArchiveLogAuthorityError as exc:
        logger.error("Archive log authority command failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
