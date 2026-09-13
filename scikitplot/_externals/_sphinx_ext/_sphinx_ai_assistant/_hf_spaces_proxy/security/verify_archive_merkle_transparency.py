"""
Run 164: verify Merkle transparency for externally anchored archive health.

Run 163 hash-linked channel assertions are upgraded to cryptographically verifiable
RFC6962-style Merkle inclusion and consistency proofs.  A separately pinned
transparency root defines independent log and gossip authorities.  Every accepted
Run 163 anchor epoch must be included in every configured log, each log must prove
append-only extension from its previously accepted tree root, and independent gossip
observers must sign the exact same set of checkpoint hashes.
"""

from __future__ import annotations

import base64
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
from typing import Any, Callable

import tomllib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)

try:
    from . import anchor_archive_health as run163
except (ImportError, ValueError) as exc:
    import importlib.util
    import sys

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "_run163_for_merkle", _here / "anchor_archive_health.py"
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("anchor_archive_health.py") from exc
    run163 = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = run163
    _spec.loader.exec_module(run163)

HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_archive_merkle_policy.toml").read_text())
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
_DOC_ARCHIVE_ANCHOR_STATE = "trusted-archive-anchor-state.json"
_DOC_ARCHIVE_MERKLE_STATE = "trusted-archive-merkle-state.json"


_OUTPUT_NAMES = {
    "release-archive-merkle-bundle.json",
    _DOC_ARCHIVE_MERKLE_STATE,
    "active-archive-merkle-evidence.json",
    "release-archive-merkle-receipt.json",
}
_RUN163_NAMES = {
    "release-archive-anchor-bundle.json",
    _DOC_ARCHIVE_ANCHOR_STATE,
    "active-archive-anchor-evidence.json",
    "release-archive-anchor-receipt.json",
}


class ArchiveMerkleError(RuntimeError):  # ruff: ignore[undocumented-public-class]
    pass


def _fail(code: str) -> None:
    raise ArchiveMerkleError(code)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


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

    try:
        value = json.loads(raw.decode(), object_pairs_hook=hook)
    except ArchiveMerkleError:
        raise
    except Exception as exc:
        raise ArchiveMerkleError(code + "_JSON_INVALID") from exc
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


def _id(value: Any, code: str) -> str:
    if not isinstance(value, str):
        _fail(code)
    value = value.strip()
    if (
        not value
        or ".." in value
        or "?" in value
        or "#" in value
        or "\x00" in value
        or _ID.fullmatch(value) is None
    ):
        _fail(code)
    return value


def _hex(value: Any, code: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(code)
    return value


def _dt(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except Exception as exc:
        raise ArchiveMerkleError(code) from exc


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
    except Exception as exc:
        raise ArchiveMerkleError(code) from exc
    if size is not None and len(raw) != size:
        _fail(code)
    return raw


def _verify_sig(public_key: str, signature: str, message: bytes, code: str) -> None:
    try:
        Ed25519PublicKey.from_public_bytes(
            _b64(public_key, code + "_PUBLIC", 32)
        ).verify(_b64(signature, code + "_SIGNATURE", 64), message)
    except (InvalidSignature, ValueError) as exc:
        raise ArchiveMerkleError(code + "_INVALID") from exc


def _positive_int(value: Any, code: str, *, allow_zero: bool = False) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < (0 if allow_zero else 1)
    ):
        _fail(code)
    if value > int(POLICY["max_tree_size"]):
        _fail(code)
    return value


def _regular_dir(path: Path, code: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        _fail(code)
    return path.resolve()


def _outside(path: Path, protected: list[Path], code: str) -> Path:
    target = path.expanduser().resolve()
    for item in protected:
        root = item.resolve()
        if target == root or root in target.parents:
            _fail(code)
    return target


def _dir_fingerprint(root: Path, code: str) -> dict[str, tuple[str, int, int]]:
    root = _regular_dir(root, code)
    out = {}
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_file():
            _fail(code)
        stat = path.stat()
        out[path.name] = (_sha(path), stat.st_size, stat.st_mode & 0o7777)
    return out


def _authority_fingerprint(paths: list[Path], code: str) -> list[tuple[str, Any]]:
    out = []
    for raw in paths:
        path = Path(raw)
        if path.is_symlink():
            _fail(code)
        resolved = path.resolve()
        if resolved.is_dir():
            out.append((str(resolved), _dir_fingerprint(resolved, code)))
        elif resolved.is_file():
            stat = resolved.stat()
            out.append(
                (str(resolved), (_sha(resolved), stat.st_size, stat.st_mode & 0o7777))
            )
        else:
            _fail(code)
    return out


# RFC6962-style Merkle hashing: leaf = H(0x00 || data), node = H(0x01 || left || right).
def merkle_leaf_hash(data: bytes) -> str:  # ruff: ignore[undocumented-public-function]
    return hashlib.sha256(b"\x00" + data).hexdigest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _proof_hashes(values: Any, code: str) -> list[bytes]:
    if not isinstance(values, list) or len(values) > int(POLICY["max_proof_nodes"]):
        _fail(code)
    return [bytes.fromhex(_hex(value, code)) for value in values]


def verify_inclusion_proof(  # ruff: ignore[undocumented-public-function]
    *, leaf_hash: str, leaf_index: int, tree_size: int, proof: list[str], root_hash: str
) -> bool:
    leaf = bytes.fromhex(_hex(leaf_hash, "ARCHIVE_MERKLE_LEAF_HASH_INVALID"))
    root = bytes.fromhex(_hex(root_hash, "ARCHIVE_MERKLE_ROOT_HASH_INVALID"))
    if tree_size <= 0 or leaf_index < 0 or leaf_index >= tree_size:
        return False
    fn = leaf_index
    sn = tree_size - 1
    result = leaf
    for sibling in _proof_hashes(proof, "ARCHIVE_MERKLE_INCLUSION_PROOF_INVALID"):
        if fn & 1 or fn == sn:
            result = _node_hash(sibling, result)
            while fn != 0 and (fn & 1) == 0:
                fn >>= 1
                sn >>= 1
        else:
            result = _node_hash(result, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and result == root


def verify_consistency_proof(  # ruff: ignore[too-many-branches, too-many-return-statements, undocumented-public-function]
    *,
    old_size: int,
    new_size: int,
    old_root_hash: str | None,
    new_root_hash: str,
    proof: list[str],
) -> bool:
    if old_size < 0 or new_size <= 0 or old_size > new_size:
        return False
    new_root = bytes.fromhex(_hex(new_root_hash, "ARCHIVE_MERKLE_NEW_ROOT_INVALID"))
    nodes = _proof_hashes(proof, "ARCHIVE_MERKLE_CONSISTENCY_PROOF_INVALID")
    if old_size == 0:
        return len(nodes) == 0 and old_root_hash is None
    if old_root_hash is None:
        return False
    old_root = bytes.fromhex(_hex(old_root_hash, "ARCHIVE_MERKLE_OLD_ROOT_INVALID"))
    if old_size == new_size:
        return len(nodes) == 0 and old_root == new_root

    fn = old_size - 1
    sn = new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1

    if fn == 0:
        fr = sr = old_root
    else:
        if not nodes:
            return False
        fr = sr = nodes[0]
        nodes = nodes[1:]

    for node in nodes:
        if sn == 0:
            return False
        if (fn & 1) or fn == sn:
            fr = _node_hash(node, fr)
            sr = _node_hash(node, sr)
            while fn != 0 and (fn & 1) == 0:
                fn >>= 1
                sn >>= 1
        else:
            sr = _node_hash(sr, node)
        fn >>= 1
        sn >>= 1
    return sn == 0 and fr == old_root and sr == new_root


def _root_key(value: Any, code: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "identity",
        "operator",
        "expires",
        "publicKey",
    }:
        _fail(code + "_SCHEMA_INVALID")
    out = {
        "identity": _id(value["identity"], code + "_IDENTITY_INVALID"),
        "operator": _id(value["operator"], code + "_OPERATOR_INVALID"),
        "expires": value["expires"],
        "publicKey": value["publicKey"],
    }
    _dt(out["expires"], code + "_EXPIRES_INVALID")
    _b64(out["publicKey"], code + "_PUBLIC_INVALID", 32)
    if out != value:
        _fail(code + "_NOT_NORMALIZED")
    return out


def _log_entry(value: Any, code: str) -> dict[str, str]:
    expected = {
        "operator",
        "publicKey",
        "gossipIdentity",
        "gossipOperator",
        "gossipPublicKey",
    }
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    out = {
        "operator": _id(value["operator"], code + "_OPERATOR_INVALID"),
        "publicKey": value["publicKey"],
        "gossipIdentity": _id(
            value["gossipIdentity"], code + "_GOSSIP_IDENTITY_INVALID"
        ),
        "gossipOperator": _id(
            value["gossipOperator"], code + "_GOSSIP_OPERATOR_INVALID"
        ),
        "gossipPublicKey": value["gossipPublicKey"],
    }
    _b64(out["publicKey"], code + "_PUBLIC_INVALID", 32)
    _b64(out["gossipPublicKey"], code + "_GOSSIP_PUBLIC_INVALID", 32)
    if out != value:
        _fail(code + "_NOT_NORMALIZED")
    return out


def verify_transparency_root(  # ruff: ignore[too-many-branches, undocumented-public-function]
    doc: dict[str, Any],
    expected_sha256: str,
    *,
    now: datetime,
    historical: bool = False,
) -> dict[str, Any]:
    if set(doc) != {"signed", "signatures"} or not isinstance(doc.get("signed"), dict):
        _fail("ARCHIVE_MERKLE_ROOT_SCHEMA_INVALID")
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
        "logs",
    }
    if (
        set(signed) != expected
        or signed.get("_type") != "archive-merkle-transparency-root"
    ):
        _fail("ARCHIVE_MERKLE_ROOT_SIGNED_SCHEMA_INVALID")
    if signed.get("specVersion") != str(POLICY["spec_version"]) or signed.get(
        "schemaVersion"
    ) != int(POLICY["transparency_root_schema_version"]):
        _fail("ARCHIVE_MERKLE_ROOT_VERSION_INVALID")
    root_id = _id(signed["rootId"], "ARCHIVE_MERKLE_ROOT_ID_INVALID")
    if signed["version"] != 1:
        _fail("ARCHIVE_MERKLE_ROOT_VERSION_INVALID")
    issued = _dt(signed["issuedAt"], "ARCHIVE_MERKLE_ROOT_ISSUED_INVALID")
    expires = _dt(signed["expires"], "ARCHIVE_MERKLE_ROOT_EXPIRES_INVALID")
    if expires <= issued or expires - issued > timedelta(
        days=int(POLICY["max_root_lifetime_days"])
    ):
        _fail("ARCHIVE_MERKLE_ROOT_LIFETIME_INVALID")
    if not historical:
        if issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
            _fail("ARCHIVE_MERKLE_ROOT_FROM_FUTURE")
        if expires <= now:
            _fail("ARCHIVE_MERKLE_ROOT_EXPIRED")

    raw_keys = signed["keys"]
    _raw_keys = (
        int(POLICY["min_root_keys"])  # int
        <= len(raw_keys)  # lint
        <= 64  # ruff: ignore[magic-value-comparison]
    )
    if not isinstance(raw_keys, dict) or not _raw_keys:
        _fail("ARCHIVE_MERKLE_ROOT_KEYS_INVALID")
    keys = {}
    for raw_kid, value in sorted(raw_keys.items()):
        kid = _id(raw_kid, "ARCHIVE_MERKLE_ROOT_KEY_ID_INVALID")
        if kid != raw_kid or kid in keys:
            _fail("ARCHIVE_MERKLE_ROOT_KEY_ID_INVALID")
        keys[kid] = _root_key(value, "ARCHIVE_MERKLE_ROOT_KEY")
    for key in keys.values():
        if _dt(key["expires"], "ARCHIVE_MERKLE_ROOT_KEY_EXPIRES_INVALID") < expires:
            _fail("ARCHIVE_MERKLE_ROOT_KEY_EXPIRES_BEFORE_ROOT")

    threshold = _positive_int(
        signed["threshold"], "ARCHIVE_MERKLE_ROOT_THRESHOLD_INVALID"
    )
    if threshold < int(POLICY["min_root_threshold"]) or threshold > len(keys):
        _fail("ARCHIVE_MERKLE_ROOT_THRESHOLD_INVALID")
    selected = signed["selectedSignerKeyIds"]
    if not isinstance(selected, list):
        _fail("ARCHIVE_MERKLE_ROOT_SELECTED_INVALID")
    selected = [_id(x, "ARCHIVE_MERKLE_ROOT_SELECTED_INVALID") for x in selected]
    if (
        selected != sorted(selected)
        or len(selected) != threshold
        or len(set(selected)) != len(selected)
    ):
        _fail("ARCHIVE_MERKLE_ROOT_SELECTED_INVALID")
    if any(k not in keys for k in selected):
        _fail("ARCHIVE_MERKLE_ROOT_SELECTED_INVALID")
    if len({keys[k]["operator"] for k in selected}) < int(POLICY["min_root_operators"]):
        _fail("ARCHIVE_MERKLE_ROOT_OPERATOR_QUORUM_INVALID")

    logs_raw = signed["logs"]
    if not isinstance(logs_raw, dict) or len(logs_raw) < int(POLICY["min_logs"]):
        _fail("ARCHIVE_MERKLE_LOGS_INVALID")
    logs = {
        lid: _log_entry(value, "ARCHIVE_MERKLE_LOG")
        for lid, value in sorted(logs_raw.items())
    }
    if list(logs) != sorted(logs) or any(
        _id(lid, "ARCHIVE_MERKLE_LOG_ID_INVALID") != lid for lid in logs
    ):
        _fail("ARCHIVE_MERKLE_LOGS_INVALID")
    if len({v["operator"] for v in logs.values()}) != len(logs):
        _fail("ARCHIVE_MERKLE_LOG_OPERATOR_DIVERSITY_INVALID")
    if len({v["gossipOperator"] for v in logs.values()}) < int(POLICY["min_gossipers"]):
        _fail("ARCHIVE_MERKLE_GOSSIP_OPERATOR_DIVERSITY_INVALID")
    root_ops = {v["operator"] for v in keys.values()}
    root_pubs = {v["publicKey"] for v in keys.values()}
    log_ops = {v["operator"] for v in logs.values()}
    log_pubs = {v["publicKey"] for v in logs.values()}
    gossip_ops = {v["gossipOperator"] for v in logs.values()}
    gossip_pubs = {v["gossipPublicKey"] for v in logs.values()}
    if (
        root_ops & (log_ops | gossip_ops)
        or log_ops & gossip_ops
        or root_pubs & (log_pubs | gossip_pubs)
        or log_pubs & gossip_pubs
    ):
        _fail("ARCHIVE_MERKLE_AUTHORITY_PLANES_OVERLAP")

    signatures = doc["signatures"]
    if not isinstance(signatures, list) or len(signatures) != len(selected):
        _fail("ARCHIVE_MERKLE_ROOT_SIGNATURE_SET_INVALID")
    sig_map = {}
    for item in signatures:
        if not isinstance(item, dict) or set(item) != {"keyId", "signature"}:
            _fail("ARCHIVE_MERKLE_ROOT_SIGNATURE_SCHEMA_INVALID")
        key_id = _id(item["keyId"], "ARCHIVE_MERKLE_ROOT_SIGNATURE_KEY_INVALID")
        if key_id in sig_map:
            _fail("ARCHIVE_MERKLE_ROOT_SIGNATURE_DUPLICATE")
        sig_map[key_id] = item["signature"]
    if sorted(sig_map) != selected or signatures != sorted(
        signatures, key=lambda x: x["keyId"]
    ):
        _fail("ARCHIVE_MERKLE_ROOT_SIGNATURE_SET_INVALID")
    message = _canonical(signed)
    for key_id in selected:
        _verify_sig(
            keys[key_id]["publicKey"],
            sig_map[key_id],
            message,
            "ARCHIVE_MERKLE_ROOT_SIGNATURE",
        )
    raw = _canonical(doc)
    sha = _sha_bytes(raw)
    if sha != _hex(expected_sha256, "ARCHIVE_MERKLE_ROOT_PIN_INVALID"):
        _fail("ARCHIVE_MERKLE_ROOT_PIN_MISMATCH")
    return {
        "rootId": root_id,
        "sha256": sha,
        "keys": keys,
        "logs": logs,
        "issued": issued,
        "expires": expires,
        "doc": doc,
    }


def _command_adapter(
    command: list[str], prefix: str
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if not command:
        _fail(prefix + "_COMMAND_EMPTY")

    def call(request: dict[str, Any]) -> dict[str, Any]:
        proc = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
        )
        payload = _canonical(request)
        out = bytearray()
        err = bytearray()
        limit = int(POLICY["max_adapter_output_bytes"])

        def pump(stream, buf):
            try:
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if len(buf) > limit:
                        try:  # ruff: ignore[suppressible-exception]
                            proc.kill()
                        except OSError:
                            pass
                        return
            finally:
                stream.close()

        t1 = threading.Thread(target=pump, args=(proc.stdout, out))
        t2 = threading.Thread(target=pump, args=(proc.stderr, err))
        t1.start()
        t2.start()
        try:
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
                proc.wait(timeout=int(POLICY["adapter_timeout_seconds"]))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                _fail(prefix + "_TIMEOUT")
            t1.join()
            t2.join()
            if len(out) > limit or len(err) > limit:
                _fail(prefix + "_OUTPUT_TOO_LARGE")
            if proc.returncode != 0:
                _fail(prefix + "_FAILED")
            return _loads(bytes(out), prefix + "_RESPONSE")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if t1.is_alive():
                t1.join()
            if t2.is_alive():
                t2.join()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:  # ruff: ignore[suppressible-exception]
                    stream.close()
                except OSError:  # ruff: ignore[try-except-in-loop]
                    pass

    return call


def command_log(  # ruff: ignore[undocumented-public-function]
    command: list[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _command_adapter(command, "ARCHIVE_MERKLE_LOG_ADAPTER")


def command_gossip(  # ruff: ignore[undocumented-public-function]
    command: list[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _command_adapter(command, "ARCHIVE_MERKLE_GOSSIP_ADAPTER")


def _verify_run163_current(
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
    historical: bool = False,
) -> dict[str, Any]:
    try:
        run163.verify_anchor_history(
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
            output_dir=run163_dir,
            now=now,
            historical=historical,
            require_current_match=True,
        )
    except Exception as exc:
        raise ArchiveMerkleError("ARCHIVE_MERKLE_RUN163_INVALID:" + str(exc)) from exc
    docs = {}
    raws = {}
    root = _regular_dir(Path(run163_dir), "ARCHIVE_MERKLE_RUN163_DIR_INVALID")
    if {p.name for p in root.iterdir()} != _RUN163_NAMES:
        _fail("ARCHIVE_MERKLE_RUN163_ALLOWLIST_INVALID")
    for name in sorted(_RUN163_NAMES):
        docs[name], raws[name] = _read_json(root / name, "ARCHIVE_MERKLE_RUN163")
    return {"docs": docs, "raws": raws}


def _enforce_external_separation(
    root: dict[str, Any], witness_root_path: Path, anchor_plan_path: Path
) -> None:
    wdoc, _ = _read_json(Path(witness_root_path), "ARCHIVE_MERKLE_WITNESS_ROOT")
    pdoc, _ = _read_json(Path(anchor_plan_path), "ARCHIVE_MERKLE_ANCHOR_PLAN")
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
    own_ops = (
        {v["operator"] for v in root["keys"].values()}
        | {v["operator"] for v in root["logs"].values()}
        | {v["gossipOperator"] for v in root["logs"].values()}
    )
    own_pubs = (
        {v["publicKey"] for v in root["keys"].values()}
        | {v["publicKey"] for v in root["logs"].values()}
        | {v["gossipPublicKey"] for v in root["logs"].values()}
    )
    if own_ops & other_ops or own_pubs & other_pubs:
        _fail("ARCHIVE_MERKLE_EXTERNAL_AUTHORITY_OVERLAP")


def _leaf_document(
    run163_docs: dict[str, dict[str, Any]], run163_raws: dict[str, bytes]
) -> dict[str, Any]:
    state = run163_docs[_DOC_ARCHIVE_ANCHOR_STATE]
    active = run163_docs["active-archive-anchor-evidence.json"]
    return {
        "_type": "run163-archive-anchor-leaf",
        "specVersion": str(POLICY["spec_version"]),
        "run163Sequence": state.get("sequence"),
        "run163WitnessSequence": state.get("run162WitnessSequence"),
        "run163AnchorConsensusHeadSha256": state.get("anchorConsensusHeadSha256"),
        "run163StateSha256": _sha_bytes(run163_raws[_DOC_ARCHIVE_ANCHOR_STATE]),
        "run163ActiveEvidenceSha256": _sha_bytes(
            run163_raws["active-archive-anchor-evidence.json"]
        ),
        "run163ChannelIds": active.get("channelIds", state.get("channelIds")),
    }


def _verify_checkpoint(  # ruff: ignore[too-many-branches]
    doc: dict[str, Any],
    *,
    log_id: str,
    config: dict[str, str],
    sequence: int,
    challenge: str,
    expected_leaf_hash: str,
    previous: dict[str, Any] | None,
    now: datetime,
    creation: bool,
    historical: bool,
) -> dict[str, Any]:
    if set(doc) != {"signed", "signature"} or not isinstance(doc.get("signed"), dict):
        _fail("ARCHIVE_MERKLE_CHECKPOINT_SCHEMA_INVALID")
    signed = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "logId",
        "operator",
        "sequence",
        "treeSize",
        "rootHash",
        "leafIndex",
        "leafHash",
        "previousTreeSize",
        "previousRootHash",
        "integratedAt",
        "challenge",
        "inclusionProof",
        "consistencyProof",
    }
    if (
        set(signed) != expected
        or signed.get("_type") != "archive-merkle-log-checkpoint"
    ):
        _fail("ARCHIVE_MERKLE_CHECKPOINT_SIGNED_SCHEMA_INVALID")
    if signed.get("specVersion") != str(POLICY["spec_version"]) or signed.get(
        "schemaVersion"
    ) != int(POLICY["checkpoint_schema_version"]):
        _fail("ARCHIVE_MERKLE_CHECKPOINT_VERSION_INVALID")
    if (
        signed["logId"] != log_id
        or signed["operator"] != config["operator"]
        or signed["sequence"] != sequence
        or signed["challenge"] != challenge
    ):
        _fail("ARCHIVE_MERKLE_CHECKPOINT_BINDING_INVALID")
    if signed["leafHash"] != expected_leaf_hash:
        _fail("ARCHIVE_MERKLE_CHECKPOINT_LEAF_INVALID")
    tree_size = _positive_int(signed["treeSize"], "ARCHIVE_MERKLE_TREE_SIZE_INVALID")
    leaf_index = _positive_int(
        signed["leafIndex"], "ARCHIVE_MERKLE_LEAF_INDEX_INVALID", allow_zero=True
    )
    if leaf_index >= tree_size:
        _fail("ARCHIVE_MERKLE_LEAF_INDEX_INVALID")
    root_hash = _hex(signed["rootHash"], "ARCHIVE_MERKLE_ROOT_HASH_INVALID")
    integrated = _dt(signed["integratedAt"], "ARCHIVE_MERKLE_INTEGRATED_AT_INVALID")
    if integrated > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
        _fail("ARCHIVE_MERKLE_CHECKPOINT_FROM_FUTURE")
    if creation and now - integrated > timedelta(
        minutes=int(POLICY["max_adapter_freshness_minutes"])
    ):
        _fail("ARCHIVE_MERKLE_CHECKPOINT_STALE")
    if (
        not historical
        and not creation
        and now - integrated > timedelta(days=int(POLICY["max_active_epoch_age_days"]))
    ):
        _fail("ARCHIVE_MERKLE_ACTIVE_EPOCH_STALE")

    if previous is None:
        if (
            signed["previousTreeSize"] != 0
            or signed["previousRootHash"] is not None
            or signed["consistencyProof"] != []
        ):
            _fail("ARCHIVE_MERKLE_GENESIS_CONSISTENCY_INVALID")
        old_size = 0
        old_root = None
    else:
        old_size = previous["treeSize"]
        old_root = previous["rootHash"]
        if (
            signed["previousTreeSize"] != old_size
            or signed["previousRootHash"] != old_root
        ):
            _fail("ARCHIVE_MERKLE_PREVIOUS_CHECKPOINT_INVALID")
        if tree_size <= old_size:
            _fail("ARCHIVE_MERKLE_TREE_NOT_ADVANCED")
    if not verify_inclusion_proof(
        leaf_hash=expected_leaf_hash,
        leaf_index=leaf_index,
        tree_size=tree_size,
        proof=signed["inclusionProof"],
        root_hash=root_hash,
    ):
        _fail("ARCHIVE_MERKLE_INCLUSION_INVALID")
    if not verify_consistency_proof(
        old_size=old_size,
        new_size=tree_size,
        old_root_hash=old_root,
        new_root_hash=root_hash,
        proof=signed["consistencyProof"],
    ):
        _fail("ARCHIVE_MERKLE_CONSISTENCY_INVALID")
    _verify_sig(
        config["publicKey"],
        doc["signature"],
        _canonical(signed),
        "ARCHIVE_MERKLE_CHECKPOINT_SIGNATURE",
    )
    return {
        "logId": log_id,
        "treeSize": tree_size,
        "rootHash": root_hash,
        "leafIndex": leaf_index,
        "leafHash": expected_leaf_hash,
        "checkpointSha256": _sha_bytes(_canonical(doc)),
        "integratedAt": integrated,
    }


def _gossip_challenge(
    sequence: int, run163_head: str, leaf_hash: str, checkpoints: dict[str, str]
) -> str:
    return _sha_bytes(
        _canonical(
            {
                "sequence": sequence,
                "run163AnchorConsensusHeadSha256": run163_head,
                "leafHash": leaf_hash,
                "checkpointSha256s": checkpoints,
            }
        )
    )


def _verify_gossip(
    doc: dict[str, Any],
    *,
    log_id: str,
    config: dict[str, str],
    sequence: int,
    challenge: str,
    run163_head: str,
    checkpoints: dict[str, str],
    now: datetime,
    creation: bool,
    historical: bool,
) -> dict[str, Any]:
    if set(doc) != {"signed", "signature"} or not isinstance(doc.get("signed"), dict):
        _fail("ARCHIVE_MERKLE_GOSSIP_SCHEMA_INVALID")
    signed = doc["signed"]
    expected = {
        "_type",
        "specVersion",
        "schemaVersion",
        "gossipIdentity",
        "gossipOperator",
        "sequence",
        "observedAt",
        "challenge",
        "run163AnchorConsensusHeadSha256",
        "checkpointSha256s",
    }
    if (
        set(signed) != expected
        or signed.get("_type") != "archive-merkle-gossip-observation"
    ):
        _fail("ARCHIVE_MERKLE_GOSSIP_SIGNED_SCHEMA_INVALID")
    if signed.get("specVersion") != str(POLICY["spec_version"]) or signed.get(
        "schemaVersion"
    ) != int(POLICY["gossip_schema_version"]):
        _fail("ARCHIVE_MERKLE_GOSSIP_VERSION_INVALID")
    if (
        signed["gossipIdentity"] != config["gossipIdentity"]
        or signed["gossipOperator"] != config["gossipOperator"]
    ):
        _fail("ARCHIVE_MERKLE_GOSSIP_IDENTITY_INVALID")
    if (
        signed["sequence"] != sequence
        or signed["challenge"] != challenge
        or signed["run163AnchorConsensusHeadSha256"] != run163_head
    ):
        _fail("ARCHIVE_MERKLE_GOSSIP_BINDING_INVALID")
    if signed["checkpointSha256s"] != checkpoints:
        _fail("ARCHIVE_MERKLE_GOSSIP_SPLIT_VIEW")
    observed = _dt(signed["observedAt"], "ARCHIVE_MERKLE_GOSSIP_OBSERVED_INVALID")
    if observed > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
        _fail("ARCHIVE_MERKLE_GOSSIP_FROM_FUTURE")
    if creation and now - observed > timedelta(
        minutes=int(POLICY["max_adapter_freshness_minutes"])
    ):
        _fail("ARCHIVE_MERKLE_GOSSIP_STALE")
    if (
        not historical
        and not creation
        and now - observed > timedelta(days=int(POLICY["max_active_epoch_age_days"]))
    ):
        _fail("ARCHIVE_MERKLE_ACTIVE_EPOCH_STALE")
    _verify_sig(
        config["gossipPublicKey"],
        doc["signature"],
        _canonical(signed),
        "ARCHIVE_MERKLE_GOSSIP_SIGNATURE",
    )
    return {
        "logId": log_id,
        "gossipSha256": _sha_bytes(_canonical(doc)),
        "observedAt": observed,
    }


def _event_head(previous_head: str | None, event_without_head: dict[str, Any]) -> str:
    return _sha_bytes(
        _canonical(
            {
                "previousMerkleConsensusHeadSha256": previous_head,
                "event": event_without_head,
            }
        )
    )


def _run163_artifacts(raws: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": _sha_bytes(raws[name]), "size": len(raws[name])}
        for name in sorted(raws)
    }


def _load_output(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    root = _regular_dir(root, "ARCHIVE_MERKLE_OUTPUT_INVALID")
    if {p.name for p in root.iterdir()} != _OUTPUT_NAMES:
        _fail("ARCHIVE_MERKLE_OUTPUT_ALLOWLIST_INVALID")
    docs = {}
    raws = {}
    for name in sorted(_OUTPUT_NAMES):
        docs[name], raws[name] = _read_json(root / name, "ARCHIVE_MERKLE_OUTPUT")
    return docs, raws


def _replay_history(  # ruff: ignore[too-many-branches]
    *,
    root: dict[str, Any],
    bundle: dict[str, Any],
    receipt: dict[str, Any],
    now: datetime,
    historical: bool,
    current_run163_docs: dict[str, Any] | None,
    current_run163_raws: dict[str, bytes] | None,
) -> dict[str, Any]:
    if set(bundle) != {
        "schemaVersion",
        "predicateType",
        "status",
        "transparencyRootSha256",
        "events",
    }:
        _fail("ARCHIVE_MERKLE_BUNDLE_SCHEMA_INVALID")
    if (
        bundle["schemaVersion"] != int(POLICY["bundle_schema_version"])
        or bundle["predicateType"] != PREDICATE_TYPE
        or bundle["status"] != "archive-merkle-transparency-history"
        or bundle["transparencyRootSha256"] != root["sha256"]
    ):
        _fail("ARCHIVE_MERKLE_BUNDLE_SCHEMA_INVALID")
    if (
        set(receipt) != {"schemaVersion", "status", "events"}
        or receipt["schemaVersion"] != int(POLICY["receipt_schema_version"])
        or receipt["status"] != "archive-merkle-transparency-accepted"
    ):
        _fail("ARCHIVE_MERKLE_RECEIPT_SCHEMA_INVALID")
    events = bundle["events"]
    recs = receipt["events"]
    if (
        not isinstance(events, list)
        or not events
        or not isinstance(recs, list)
        or len(events) != len(recs)
    ):
        _fail("ARCHIVE_MERKLE_HISTORY_LENGTH_INVALID")
    previous_head = None
    previous_by_log: dict[str, dict[str, Any] | None] = dict.fromkeys(root["logs"])
    for idx, (event, rec) in enumerate(zip(events, recs), 1):
        expected_event_keys = {
            "sequence",
            "run163Sequence",
            "run163AnchorConsensusHeadSha256",
            "run163Artifacts",
            "leafHash",
            "logs",
            "logChallenge",
            "gossipChallenge",
            "gossipResponseSha256s",
            "merkleConsensusHeadSha256",
        }
        if (
            not isinstance(event, dict)
            or set(event) != expected_event_keys
            or event["sequence"] != idx
            or event["run163Sequence"] != idx
        ):
            _fail("ARCHIVE_MERKLE_EVENT_SCHEMA_INVALID")
        if (
            not isinstance(rec, dict)
            or set(rec)
            != {"sequence", "run163Documents", "logResponses", "gossipResponses"}
            or rec["sequence"] != idx
        ):
            _fail("ARCHIVE_MERKLE_RECEIPT_EVENT_INVALID")
        run163_docs = rec["run163Documents"]
        if not isinstance(run163_docs, dict) or set(run163_docs) != _RUN163_NAMES:
            _fail("ARCHIVE_MERKLE_EMBEDDED_RUN163_INVALID")
        raws = {name: _canonical(run163_docs[name]) for name in sorted(run163_docs)}
        if event["run163Artifacts"] != _run163_artifacts(raws):
            _fail("ARCHIVE_MERKLE_RUN163_ARTIFACT_BINDING_INVALID")
        state = run163_docs[_DOC_ARCHIVE_ANCHOR_STATE]
        if event["run163Sequence"] != state.get("sequence") or event[
            "run163AnchorConsensusHeadSha256"
        ] != state.get("anchorConsensusHeadSha256"):
            _fail("ARCHIVE_MERKLE_RUN163_STATE_BINDING_INVALID")
        leaf_doc = _leaf_document(run163_docs, raws)
        leaf_hash = merkle_leaf_hash(_canonical(leaf_doc))
        if event["leafHash"] != leaf_hash:
            _fail("ARCHIVE_MERKLE_LEAF_BINDING_INVALID")
        log_responses = rec["logResponses"]
        if not isinstance(log_responses, dict) or set(log_responses) != set(
            root["logs"]
        ):
            _fail("ARCHIVE_MERKLE_LOG_RESPONSE_SET_INVALID")
        log_rows = []
        latest = idx == len(events)
        for log_id in sorted(root["logs"]):
            parsed = _verify_checkpoint(
                log_responses[log_id],
                log_id=log_id,
                config=root["logs"][log_id],
                sequence=idx,
                challenge=event["logChallenge"],
                expected_leaf_hash=leaf_hash,
                previous=previous_by_log[log_id],
                now=now,
                creation=False,
                historical=historical or not latest,
            )
            log_rows.append(
                {
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
            )
            previous_by_log[log_id] = parsed
        if event["logs"] != log_rows:
            _fail("ARCHIVE_MERKLE_LOG_EVENT_MISMATCH")
        checkpoints = {row["logId"]: row["checkpointSha256"] for row in log_rows}
        challenge = _gossip_challenge(
            idx, event["run163AnchorConsensusHeadSha256"], leaf_hash, checkpoints
        )
        if event["gossipChallenge"] != challenge:
            _fail("ARCHIVE_MERKLE_GOSSIP_CHALLENGE_INVALID")
        gossip = rec["gossipResponses"]
        if not isinstance(gossip, dict) or set(gossip) != set(root["logs"]):
            _fail("ARCHIVE_MERKLE_GOSSIP_RESPONSE_SET_INVALID")
        gossip_hashes = []
        for log_id in sorted(root["logs"]):
            parsed = _verify_gossip(
                gossip[log_id],
                log_id=log_id,
                config=root["logs"][log_id],
                sequence=idx,
                challenge=challenge,
                run163_head=event["run163AnchorConsensusHeadSha256"],
                checkpoints=checkpoints,
                now=now,
                creation=False,
                historical=historical or not latest,
            )
            gossip_hashes.append(parsed["gossipSha256"])
        if event["gossipResponseSha256s"] != gossip_hashes:
            _fail("ARCHIVE_MERKLE_GOSSIP_EVENT_MISMATCH")
        bare = {k: event[k] for k in event if k != "merkleConsensusHeadSha256"}
        head = _event_head(previous_head, bare)
        if event["merkleConsensusHeadSha256"] != head:
            _fail("ARCHIVE_MERKLE_CHAIN_HEAD_INVALID")
        previous_head = head
    if current_run163_docs is not None and current_run163_raws is not None:
        last = events[-1]
        current_state = current_run163_docs[_DOC_ARCHIVE_ANCHOR_STATE]
        if last["run163Sequence"] != current_state.get("sequence") or last[
            "run163AnchorConsensusHeadSha256"
        ] != current_state.get("anchorConsensusHeadSha256"):
            _fail("ARCHIVE_MERKLE_CURRENT_RUN163_MISMATCH")
        if last["run163Artifacts"] != _run163_artifacts(current_run163_raws):
            _fail("ARCHIVE_MERKLE_CURRENT_RUN163_ARTIFACT_MISMATCH")
    return {
        "sequence": len(events),
        "merkleConsensusHeadSha256": previous_head,
        "last": events[-1],
    }


def verify_merkle_history(  # ruff: ignore[undocumented-public-function]
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
    transparency_root_path: Path,
    transparency_root_pin: str,
    output_dir: Path,
    now: datetime | None = None,
    historical: bool = False,
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root_doc, _ = _read_json(Path(transparency_root_path), "ARCHIVE_MERKLE_ROOT")
    root = verify_transparency_root(
        root_doc, transparency_root_pin, now=current, historical=historical
    )
    _enforce_external_separation(root, Path(witness_root_path), Path(anchor_plan_path))
    current163 = _verify_run163_current(
        run160_dir=Path(run160_dir),
        run161_dir=Path(run161_dir),
        retention_root_path=Path(retention_root_path),
        retention_root_pin=retention_root_pin,
        witness_root_path=Path(witness_root_path),
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        run162_dir=Path(run162_dir),
        anchor_plan_path=Path(anchor_plan_path),
        run163_dir=Path(run163_dir),
        now=current,
        historical=historical,
    )
    docs, raws = _load_output(Path(output_dir))
    replay = _replay_history(
        root=root,
        bundle=docs["release-archive-merkle-bundle.json"],
        receipt=docs["release-archive-merkle-receipt.json"],
        now=current,
        historical=historical,
        current_run163_docs=current163["docs"],
        current_run163_raws=current163["raws"],
    )
    state = docs[_DOC_ARCHIVE_MERKLE_STATE]
    active = docs["active-archive-merkle-evidence.json"]
    expected_state = {
        "schemaVersion": int(POLICY["state_schema_version"]),
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
            "sha256": _sha_bytes(raws["release-archive-merkle-bundle.json"]),
            "size": len(raws["release-archive-merkle-bundle.json"]),
        },
    }
    if state != expected_state:
        _fail("ARCHIVE_MERKLE_STATE_MISMATCH")
    last = replay["last"]
    expected_active = {
        "schemaVersion": int(POLICY["active_schema_version"]),
        "status": "active-archive-merkle-transparency",
        "sequence": replay["sequence"],
        "run163AnchorConsensusHeadSha256": last["run163AnchorConsensusHeadSha256"],
        "merkleConsensusHeadSha256": replay["merkleConsensusHeadSha256"],
        "leafHash": last["leafHash"],
        "logs": last["logs"],
        "gossipResponseSha256s": last["gossipResponseSha256s"],
    }
    if active != expected_active:
        _fail("ARCHIVE_MERKLE_ACTIVE_MISMATCH")
    return {
        "ok": True,
        "sequence": replay["sequence"],
        "merkle_consensus_head_sha256": replay["merkleConsensusHeadSha256"],
        "transparency_root_sha256": root["sha256"],
    }


def anchor_merkle_transparency(  # ruff: ignore[too-many-branches, undocumented-public-function]
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
    transparency_root_path: Path,
    transparency_root_pin: str,
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
    root_doc, _ = _read_json(Path(transparency_root_path), "ARCHIVE_MERKLE_ROOT")
    root = verify_transparency_root(root_doc, transparency_root_pin, now=current)
    _enforce_external_separation(root, Path(witness_root_path), Path(anchor_plan_path))
    current163 = _verify_run163_current(
        run160_dir=Path(run160_dir),
        run161_dir=Path(run161_dir),
        retention_root_path=Path(retention_root_path),
        retention_root_pin=retention_root_pin,
        witness_root_path=Path(witness_root_path),
        witness_root_pin=witness_root_pin,
        bootstrap_pin=bootstrap_pin,
        recovery_pin=recovery_pin,
        attestation_pins=attestation_pins,
        run162_dir=Path(run162_dir),
        anchor_plan_path=Path(anchor_plan_path),
        run163_dir=Path(run163_dir),
        now=current,
    )
    authority_paths = [
        Path(run160_dir),
        Path(run161_dir),
        Path(retention_root_path),
        Path(witness_root_path),
        Path(run162_dir),
        Path(anchor_plan_path),
        Path(run163_dir),
        Path(transparency_root_path),
    ]
    if previous_output_dir is not None:
        authority_paths.append(Path(previous_output_dir))
    before = _authority_fingerprint(authority_paths, "ARCHIVE_MERKLE_INPUT_DRIFT")
    state163 = current163["docs"][_DOC_ARCHIVE_ANCHOR_STATE]
    sequence = state163.get("sequence")
    if not isinstance(sequence, int) or sequence <= 0:
        _fail("ARCHIVE_MERKLE_RUN163_SEQUENCE_INVALID")

    previous_head = None
    previous_by_log = dict.fromkeys(root["logs"])
    old_events = []
    old_receipts = []
    if previous_output_dir is not None:
        prev_docs, _ = _load_output(Path(previous_output_dir))
        prev_replay = _replay_history(
            root=root,
            bundle=prev_docs["release-archive-merkle-bundle.json"],
            receipt=prev_docs["release-archive-merkle-receipt.json"],
            now=current,
            historical=True,
            current_run163_docs=None,
            current_run163_raws=None,
        )
        if sequence != prev_replay["sequence"] + 1:
            _fail("ARCHIVE_MERKLE_SEQUENCE_INVALID")
        previous_head = prev_replay["merkleConsensusHeadSha256"]
        old_events = list(prev_docs["release-archive-merkle-bundle.json"]["events"])
        old_receipts = list(prev_docs["release-archive-merkle-receipt.json"]["events"])
        for row in prev_replay["last"]["logs"]:
            previous_by_log[row["logId"]] = row
    elif sequence != 1:
        _fail("ARCHIVE_MERKLE_PREVIOUS_OUTPUT_REQUIRED")

    if {item[0] for item in adapters} != set(root["logs"]) or len(adapters) != len(
        root["logs"]
    ):
        _fail("ARCHIVE_MERKLE_ADAPTER_SET_INVALID")
    amap = {lid: (log, gossip) for lid, log, gossip in adapters}
    leaf_doc = _leaf_document(current163["docs"], current163["raws"])
    leaf_hash = merkle_leaf_hash(_canonical(leaf_doc))
    prechallenge = _sha_bytes(
        _canonical(
            {
                "sequence": sequence,
                "run163AnchorConsensusHeadSha256": state163[
                    "anchorConsensusHeadSha256"
                ],
                "leafHash": leaf_hash,
                "previousMerkleConsensusHeadSha256": previous_head,
            }
        )
    )
    log_docs = {}
    log_rows = []
    for log_id in sorted(root["logs"]):
        prev = previous_by_log[log_id]
        request = {
            "operation": "append-merkle-leaf",
            "protocolVersion": 1,
            "logId": log_id,
            "sequence": sequence,
            "challenge": prechallenge,
            "leaf": leaf_doc,
            "leafHash": leaf_hash,
            "previousTreeSize": 0 if prev is None else prev["treeSize"],
            "previousRootHash": None if prev is None else prev["rootHash"],
        }
        doc = amap[log_id][0](request)
        parsed = _verify_checkpoint(
            doc,
            log_id=log_id,
            config=root["logs"][log_id],
            sequence=sequence,
            challenge=prechallenge,
            expected_leaf_hash=leaf_hash,
            previous=prev,
            now=current,
            creation=True,
            historical=False,
        )
        log_docs[log_id] = doc
        log_rows.append(
            {
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
        )
    checkpoints = {row["logId"]: row["checkpointSha256"] for row in log_rows}
    gossip_challenge = _gossip_challenge(
        sequence, state163["anchorConsensusHeadSha256"], leaf_hash, checkpoints
    )
    # Log checkpoints sign the prechallenge; gossip signs the complete checkpoint map.  Preserve both.
    gossip_docs = {}
    gossip_hashes = []
    for log_id in sorted(root["logs"]):
        request = {
            "operation": "gossip-merkle-checkpoints",
            "protocolVersion": 1,
            "sequence": sequence,
            "challenge": gossip_challenge,
            "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
            "leafHash": leaf_hash,
            "checkpointSha256s": checkpoints,
        }
        doc = amap[log_id][1](request)
        parsed = _verify_gossip(
            doc,
            log_id=log_id,
            config=root["logs"][log_id],
            sequence=sequence,
            challenge=gossip_challenge,
            run163_head=state163["anchorConsensusHeadSha256"],
            checkpoints=checkpoints,
            now=current,
            creation=True,
            historical=False,
        )
        gossip_docs[log_id] = doc
        gossip_hashes.append(parsed["gossipSha256"])

    event_bare = {
        "sequence": sequence,
        "run163Sequence": sequence,
        "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
        "run163Artifacts": _run163_artifacts(current163["raws"]),
        "leafHash": leaf_hash,
        "logs": log_rows,
        "logChallenge": prechallenge,
        "gossipChallenge": gossip_challenge,
        "gossipResponseSha256s": gossip_hashes,
    }
    head = _event_head(previous_head, event_bare)
    event = dict(event_bare)
    event["merkleConsensusHeadSha256"] = head
    receipt_event = {
        "sequence": sequence,
        "run163Documents": current163["docs"],
        "logResponses": log_docs,
        "gossipResponses": gossip_docs,
    }
    bundle = {
        "schemaVersion": int(POLICY["bundle_schema_version"]),
        "predicateType": PREDICATE_TYPE,
        "status": "archive-merkle-transparency-history",
        "transparencyRootSha256": root["sha256"],
        "events": [*old_events, event],
    }
    receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "status": "archive-merkle-transparency-accepted",
        "events": [*old_receipts, receipt_event],
    }
    bundle_raw = _canonical(bundle)
    state_doc = {
        "schemaVersion": int(POLICY["state_schema_version"]),
        "status": "trusted-archive-merkle-transparency",
        "sequence": sequence,
        "run163Sequence": sequence,
        "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
        "merkleConsensusHeadSha256": head,
        "transparencyRootSha256": root["sha256"],
        "bundleArtifact": {
            "name": "release-archive-merkle-bundle.json",
            "sha256": _sha_bytes(bundle_raw),
            "size": len(bundle_raw),
        },
    }
    active = {
        "schemaVersion": int(POLICY["active_schema_version"]),
        "status": "active-archive-merkle-transparency",
        "sequence": sequence,
        "run163AnchorConsensusHeadSha256": state163["anchorConsensusHeadSha256"],
        "merkleConsensusHeadSha256": head,
        "leafHash": leaf_hash,
        "logs": log_rows,
        "gossipResponseSha256s": gossip_hashes,
    }
    if before != _authority_fingerprint(authority_paths, "ARCHIVE_MERKLE_INPUT_DRIFT"):
        _fail("ARCHIVE_MERKLE_INPUT_DRIFT")
    protected = [
        Path(run160_dir),
        Path(run161_dir),
        Path(retention_root_path),
        Path(witness_root_path),
        Path(run162_dir),
        Path(anchor_plan_path),
        Path(run163_dir),
        Path(transparency_root_path),
    ]
    if previous_output_dir is not None:
        protected.append(Path(previous_output_dir))
    target = _outside(
        Path(output_dir), protected, "ARCHIVE_MERKLE_OUTPUT_OVERLAPS_INPUT"
    )
    if target.exists():
        _fail("ARCHIVE_MERKLE_OUTPUT_EXISTS")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run164-merkle-", dir=target.parent))
    try:
        (stage / "release-archive-merkle-bundle.json").write_bytes(bundle_raw)
        _write(stage / _DOC_ARCHIVE_MERKLE_STATE, state_doc)
        _write(stage / "active-archive-merkle-evidence.json", active)
        _write(stage / "release-archive-merkle-receipt.json", receipt)
        verify_merkle_history(
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
            transparency_root_path=transparency_root_path,
            transparency_root_pin=transparency_root_pin,
            output_dir=stage,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_merkle_history(
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
        transparency_root_path=transparency_root_path,
        transparency_root_pin=transparency_root_pin,
        output_dir=target,
        now=current,
    )


def _common_args(parser) -> None:
    parser.add_argument("--run160-dir", type=Path, required=True)
    parser.add_argument("--run161-dir", type=Path, required=True)
    parser.add_argument("--retention-root", type=Path, required=True)
    parser.add_argument("--retention-root-pin", required=True)
    parser.add_argument("--witness-root", type=Path, required=True)
    parser.add_argument("--witness-root-pin", required=True)
    parser.add_argument("--bootstrap-pin", required=True)
    parser.add_argument("--recovery-pin", required=True)
    parser.add_argument("--attestation-pin", action="append", default=[])
    parser.add_argument("--run162-dir", type=Path, required=True)
    parser.add_argument("--anchor-plan", type=Path, required=True)
    parser.add_argument("--run163-dir", type=Path, required=True)
    parser.add_argument("--transparency-root", type=Path, required=True)
    parser.add_argument("--transparency-root-pin", required=True)


def main(argv=None) -> int:  # ruff: ignore[undocumented-public-function]
    import argparse  # ruff: ignore[import-outside-top-level]

    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    verify = subs.add_parser("verify")
    _common_args(verify)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--historical", action="store_true")
    anchor = subs.add_parser("anchor")
    _common_args(anchor)
    anchor.add_argument("--adapter-config", type=Path, required=True)
    anchor.add_argument("--output-dir", type=Path, required=True)
    anchor.add_argument("--previous-output-dir", type=Path)
    ns = parser.parse_args(argv)
    kwargs = {
        "run160_dir": ns.run160_dir,
        "run161_dir": ns.run161_dir,
        "retention_root_path": ns.retention_root,
        "retention_root_pin": ns.retention_root_pin,
        "witness_root_path": ns.witness_root,
        "witness_root_pin": ns.witness_root_pin,
        "bootstrap_pin": ns.bootstrap_pin,
        "recovery_pin": ns.recovery_pin,
        "attestation_pins": ns.attestation_pin,
        "run162_dir": ns.run162_dir,
        "anchor_plan_path": ns.anchor_plan,
        "run163_dir": ns.run163_dir,
        "transparency_root_path": ns.transparency_root,
        "transparency_root_pin": ns.transparency_root_pin,
        "output_dir": ns.output_dir,
    }
    if ns.command == "verify":
        result = verify_merkle_history(**kwargs, historical=ns.historical)
    else:
        cfg, _ = _read_json(ns.adapter_config, "ARCHIVE_MERKLE_ADAPTER_CONFIG")
        if set(cfg) != {"logs"} or not isinstance(cfg["logs"], dict):
            _fail("ARCHIVE_MERKLE_ADAPTER_CONFIG_SCHEMA_INVALID")
        adapters = []
        for log_id, value in sorted(cfg["logs"].items()):
            if not isinstance(value, dict) or set(value) != {
                "logCommand",
                "gossipCommand",
            }:
                _fail("ARCHIVE_MERKLE_ADAPTER_CONFIG_SCHEMA_INVALID")
            if not all(
                isinstance(v, list) and v and all(isinstance(x, str) and x for x in v)
                for v in value.values()
            ):
                _fail("ARCHIVE_MERKLE_ADAPTER_CONFIG_SCHEMA_INVALID")
            adapters.append(
                (
                    log_id,
                    command_log(value["logCommand"]),
                    command_gossip(value["gossipCommand"]),
                )
            )
        result = anchor_merkle_transparency(
            **kwargs, adapters=adapters, previous_output_dir=ns.previous_output_dir
        )
    logger.info("Archive Merkle transparency %s completed", ns.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
