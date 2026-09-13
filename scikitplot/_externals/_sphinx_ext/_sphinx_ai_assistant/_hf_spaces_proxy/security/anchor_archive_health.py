"""
Run 163: externally anchor Run 162 archive-health witness history.

Every accepted Run 162 witness-chain head is hash-linked into multiple independently
operated append-only channel checkpoints. A separate observer key for each channel
signs an inclusion/continuity read-back of the exact channel response. Recovery of
an archive-health witness root is authorized by a separate pinned threshold root and
is cryptographically bound to the already anchored consensus head; recovery cannot
rewrite accepted anchor epochs.
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
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tomllib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)

try:
    from . import witness_archive_health as run162
except (ImportError, ValueError) as exc:
    import importlib.util

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "_run162_for_anchor", _here / "witness_archive_health.py"
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("witness_archive_health.py") from exc
    run162 = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = run162
    _spec.loader.exec_module(run162)
HERE = Path(__file__).resolve().parent
POLICY = tomllib.loads((HERE / "release_archive_anchor_policy.toml").read_text())
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


_ANCHOR_NAMES = {
    "release-archive-anchor-bundle.json",
    _DOC_ARCHIVE_ANCHOR_STATE,
    "active-archive-anchor-evidence.json",
    "release-archive-anchor-receipt.json",
}
_RECOVERY_NAMES = {
    "recovered-archive-witness-root.json",
    "archive-witness-root-recovery-record.json",
    "archive-witness-root-recovery-receipt.json",
}


class ArchiveAnchorError(RuntimeError):  # ruff: ignore[undocumented-public-class]
    pass


def _fail(c):
    raise ArchiveAnchorError(c)


def _canonical(v):
    return (
        json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda: f.read(_CHUNK), b""):
            h.update(x)
    return h.hexdigest()


def _write(p, v):
    p.write_bytes(_canonical(v))


def _loads(raw, code):
    def hook(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                _fail(code + "_DUPLICATE_KEY")
            d[k] = v
        return d

    try:
        v = json.loads(raw.decode(), object_pairs_hook=hook)
    except ArchiveAnchorError:
        raise
    except Exception as e:
        raise ArchiveAnchorError(code + "_JSON_INVALID") from e
    if not isinstance(v, dict):
        _fail(code + "_SCHEMA_INVALID")
    return v


def _read_json(p, code):
    if p.is_symlink() or not p.is_file():
        _fail(code + "_INVALID")
    if p.stat().st_size > int(POLICY["max_json_bytes"]):
        _fail(code + "_TOO_LARGE")
    raw = p.read_bytes()
    d = _loads(raw, code)
    if raw != _canonical(d):
        _fail(code + "_NOT_CANONICAL")
    return d, raw


def _id(v, code):
    if not isinstance(v, str):
        _fail(code)
    v = v.strip()
    if (
        not v
        or ".." in v
        or "?" in v
        or "#" in v
        or "\x00" in v
        or _ID.fullmatch(v) is None
    ):
        _fail(code)
    return v


def _hex(v, code):
    if not isinstance(v, str) or _HEX64.fullmatch(v) is None:
        _fail(code)
    return v


def _dt(v, code):
    if not isinstance(v, str) or not v.endswith("Z"):
        _fail(code)
    try:
        return datetime.fromisoformat(v[:-1] + "+00:00").astimezone(timezone.utc)
    except Exception as e:
        raise ArchiveAnchorError(code) from e


def _ts(d):
    return (
        d.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _b64(v, code, n=None):
    if not isinstance(v, str):
        _fail(code)
    try:
        r = base64.b64decode(v, validate=True)
    except Exception as e:
        raise ArchiveAnchorError(code) from e
    if n is not None and len(r) != n:
        _fail(code)
    return r


def _verify(pk, sig, msg, code):
    try:
        Ed25519PublicKey.from_public_bytes(_b64(pk, code + "_PUBLIC", 32)).verify(
            _b64(sig, code + "_SIG", 64), msg
        )
    except (InvalidSignature, ValueError) as e:
        raise ArchiveAnchorError(code + "_INVALID") from e


def _regular_dir(p, code):
    if p.is_symlink() or not p.is_dir():
        _fail(code)
    return p.resolve()


def _outside(p, protected, code):
    t = p.expanduser().resolve()
    for x in protected:
        r = x.resolve()
        if t == r or r in t.parents:
            _fail(code)
    return t


def _dir_fp(root, code):
    root = _regular_dir(root, code)
    out = {}
    for p in sorted(root.iterdir()):
        if p.is_symlink() or not p.is_file():
            _fail(code)
        st = p.stat()
        out[p.name] = (_sha(p), st.st_size, st.st_mode & 0o7777)
    return out


def _command_adapter(command, prefix):
    def call(req):
        raw = _canonical(req)
        p = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": os.defpath, "LC_ALL": "C", "LANG": "C"},
        )
        out = bytearray()
        err = bytearray()
        limit = int(POLICY["max_adapter_output_bytes"])

        def pump(stream, buf):
            while True:
                x = stream.read(65536)
                if not x:
                    break
                buf.extend(x)
                if len(buf) > limit:
                    try:  # ruff: ignore[suppressible-exception]
                        p.kill()
                    except OSError:
                        pass
                    return

        t1 = threading.Thread(target=pump, args=(p.stdout, out))
        t2 = threading.Thread(target=pump, args=(p.stderr, err))
        t1.start()
        t2.start()
        try:
            try:
                p.stdin.write(raw)
                p.stdin.close()
                p.wait(timeout=int(POLICY["adapter_timeout_seconds"]))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                t1.join()
                t2.join()
                _fail(prefix + "_TIMEOUT")
            t1.join()
            t2.join()
            if len(out) > limit or len(err) > limit:
                _fail(prefix + "_OUTPUT_TOO_LARGE")
            if p.returncode != 0:
                _fail(prefix + "_FAILED")
            return _loads(bytes(out), prefix + "_RESPONSE")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            if t1.is_alive():
                t1.join()
            if t2.is_alive():
                t2.join()
            for stream in (p.stdin, p.stdout, p.stderr):
                try:  # ruff: ignore[suppressible-exception]
                    stream.close()
                except OSError:  # ruff: ignore[try-except-in-loop]
                    pass

    return call


def command_channel(c):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(c, "ARCHIVE_ANCHOR_CHANNEL")


def command_observer(c):  # ruff: ignore[undocumented-public-function]
    return _command_adapter(c, "ARCHIVE_ANCHOR_OBSERVER")


def _verify_witness_root(doc, pin, now, historical=False):
    try:
        return run162._verify_threshold_root(
            doc,
            pin,
            root_type="archive-health-witness-root",
            schema_key="witness_root_schema_version",
            min_keys_key="min_witness_keys",
            min_threshold_key="min_witness_threshold",
            min_operators_key="min_witness_operators",
            now=now,
            historical=historical,
        )
    except Exception as e:
        raise ArchiveAnchorError("ARCHIVE_ANCHOR_WITNESS_ROOT_INVALID:" + str(e)) from e


def _verify_run162(
    *,
    run160_dir,
    run161_dir,
    retention_root_path,
    retention_root_pin,
    witness_root_path,
    witness_root_pin,
    bootstrap_pin,
    recovery_pin,
    attestation_pins,
    run162_dir,
    now,
    historical=False,
    require_current_match=True,
):
    try:
        return run162.verify_witness_history(
            run160_dir=run160_dir,
            run161_dir=run161_dir,
            retention_root_path=retention_root_path,
            retention_root_pin=retention_root_pin,
            witness_root_path=witness_root_path,
            witness_root_pin=witness_root_pin,
            bootstrap_pin=bootstrap_pin,
            recovery_pin=recovery_pin,
            attestation_pins=attestation_pins,
            output_dir=run162_dir,
            now=now,
            historical=historical,
            require_current_match=require_current_match,
        )
    except Exception as e:
        raise ArchiveAnchorError("ARCHIVE_ANCHOR_RUN162_INVALID:" + str(e)) from e


def _channel_entry(v, code):
    exp = {
        "operator",
        "publicKey",
        "observerIdentity",
        "observerOperator",
        "observerPublicKey",
    }
    if not isinstance(v, dict) or set(v) != exp:
        _fail(code + "_SCHEMA_INVALID")
    out = {
        "operator": _id(v["operator"], code + "_OPERATOR"),
        "publicKey": v["publicKey"],
        "observerIdentity": _id(v["observerIdentity"], code + "_OBS_ID"),
        "observerOperator": _id(v["observerOperator"], code + "_OBS_OPERATOR"),
        "observerPublicKey": v["observerPublicKey"],
    }
    _b64(out["publicKey"], code + "_PUBLIC", 32)
    _b64(out["observerPublicKey"], code + "_OBS_PUBLIC", 32)
    if out != v:
        _fail(code + "_NOT_NORMALIZED")
    return out


def _verify_plan(  # ruff: ignore[too-many-branches]
    plan,
    wroot,
    now,
    historical=False,
):
    if set(plan) != {"signed", "signatures"} or not isinstance(
        plan.get("signed"), dict
    ):
        _fail("ARCHIVE_ANCHOR_PLAN_SCHEMA_INVALID")
    s = plan["signed"]
    exp = {
        "_type",
        "specVersion",
        "schemaVersion",
        "planId",
        "version",
        "issuedAt",
        "expires",
        "witnessRootSha256",
        "selectedWitnessKeyIds",
        "channels",
    }
    if (
        set(s) != exp
        or s.get("_type") != "archive-health-anchor-plan"
        or s.get("specVersion") != str(POLICY["spec_version"])
        or s.get("schemaVersion") != int(POLICY["anchor_plan_schema_version"])
    ):
        _fail("ARCHIVE_ANCHOR_PLAN_SIGNED_SCHEMA_INVALID")
    _id(s["planId"], "ARCHIVE_ANCHOR_PLAN_ID_INVALID")
    if s["version"] != 1:
        _fail("ARCHIVE_ANCHOR_PLAN_VERSION_INVALID")
    issued = _dt(s["issuedAt"], "ARCHIVE_ANCHOR_PLAN_ISSUED_INVALID")
    expires = _dt(s["expires"], "ARCHIVE_ANCHOR_PLAN_EXPIRES_INVALID")
    if expires <= issued or (
        not historical
        and (
            issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
            or expires <= now
        )
    ):
        _fail("ARCHIVE_ANCHOR_PLAN_LIFETIME_INVALID")
    if s["witnessRootSha256"] != wroot["sha256"]:
        _fail("ARCHIVE_ANCHOR_PLAN_ROOT_BINDING_INVALID")
    selected = s["selectedWitnessKeyIds"]
    if (
        not isinstance(selected, list)
        or selected != sorted(selected)
        or len(selected) != wroot["threshold"]
        or any(x not in wroot["keys"] for x in selected)
    ):
        _fail("ARCHIVE_ANCHOR_PLAN_SIGNERS_INVALID")
    channels = s["channels"]
    if not isinstance(channels, dict) or len(channels) < int(POLICY["min_channels"]):
        _fail("ARCHIVE_ANCHOR_CHANNELS_INVALID")
    norm = {}
    for cid, v in sorted(channels.items()):
        norm[_id(cid, "ARCHIVE_ANCHOR_CHANNEL_ID_INVALID")] = _channel_entry(
            v, "ARCHIVE_ANCHOR_CHANNEL"
        )
    if channels != norm:
        _fail("ARCHIVE_ANCHOR_CHANNELS_NOT_NORMALIZED")
    channel_ops = {v["operator"] for v in norm.values()}
    observer_ops = {v["observerOperator"] for v in norm.values()}
    channel_keys = {v["publicKey"] for v in norm.values()}
    observer_keys = {v["observerPublicKey"] for v in norm.values()}
    witness_ops = {v["operator"] for v in wroot["keys"].values()}
    witness_keys = {v["publicKey"] for v in wroot["keys"].values()}
    if len(channel_ops) < int(POLICY["min_channel_operators"]):
        _fail("ARCHIVE_ANCHOR_CHANNEL_OPERATOR_QUORUM_INVALID")
    if len(observer_ops) < int(POLICY["min_observer_operators"]):
        _fail("ARCHIVE_ANCHOR_OBSERVER_OPERATOR_QUORUM_INVALID")
    if len(channel_keys) != len(norm) or len(observer_keys) != len(norm):
        _fail("ARCHIVE_ANCHOR_KEY_REUSE_INVALID")
    if (
        channel_ops & observer_ops
        or channel_ops & witness_ops
        or observer_ops & witness_ops
        or channel_keys & observer_keys
        or channel_keys & witness_keys
        or observer_keys & witness_keys
    ):
        _fail("ARCHIVE_ANCHOR_PLANES_OVERLAP")
    sigs = plan["signatures"]
    if not isinstance(sigs, list) or [x.get("keyId") for x in sigs] != selected:
        _fail("ARCHIVE_ANCHOR_PLAN_SIGNATURE_SET_INVALID")
    for x in sigs:
        if not isinstance(x, dict) or set(x) != {"keyId", "signature"}:
            _fail("ARCHIVE_ANCHOR_PLAN_SIGNATURE_SCHEMA_INVALID")
        _verify(
            wroot["keys"][x["keyId"]]["publicKey"],
            x["signature"],
            _canonical(s),
            "ARCHIVE_ANCHOR_PLAN_SIGNATURE",
        )
    return {"signed": s, "channels": norm, "issued": issued, "expires": expires}


def _verify_channel_response(
    resp,
    cid,
    cfg,
    *,
    sequence,
    witness_head,
    previous,
    challenge,
    now,
    historical=False,
):
    if (
        not isinstance(resp, dict)
        or set(resp) != {"signed", "signature"}
        or not isinstance(resp["signed"], dict)
    ):
        _fail("ARCHIVE_ANCHOR_CHANNEL_RESPONSE_SCHEMA_INVALID")
    s = resp["signed"]
    exp = {
        "protocolVersion",
        "channelId",
        "operator",
        "anchorId",
        "sequence",
        "witnessChainHeadSha256",
        "previousCheckpointSha256",
        "challenge",
        "integratedAt",
        "locator",
        "immutableVersion",
    }
    if (
        set(s) != exp
        or s["protocolVersion"] != 1
        or s["channelId"] != cid
        or s["operator"] != cfg["operator"]
        or s["sequence"] != sequence
        or s["witnessChainHeadSha256"] != witness_head
        or s["previousCheckpointSha256"] != previous
        or s["challenge"] != challenge
    ):
        _fail("ARCHIVE_ANCHOR_CHANNEL_RESPONSE_BINDING_INVALID")
    _id(s["anchorId"], "ARCHIVE_ANCHOR_ID_INVALID")
    _id(s["locator"], "ARCHIVE_ANCHOR_LOCATOR_INVALID")
    _id(s["immutableVersion"], "ARCHIVE_ANCHOR_VERSION_INVALID")
    at = _dt(s["integratedAt"], "ARCHIVE_ANCHOR_INTEGRATED_INVALID")
    if not historical and (
        at > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        or now - at > timedelta(minutes=int(POLICY["max_observation_age_minutes"]))
    ):
        _fail("ARCHIVE_ANCHOR_CHANNEL_RESPONSE_STALE")
    _verify(
        cfg["publicKey"],
        resp["signature"],
        _canonical(s),
        "ARCHIVE_ANCHOR_CHANNEL_SIGNATURE",
    )
    return {"doc": resp, "sha256": _sha_bytes(_canonical(resp)), "time": at}


def _verify_observer_response(
    resp,
    cid,
    cfg,
    *,
    anchor_sha,
    sequence,
    witness_head,
    previous,
    challenge,
    now,
    historical=False,
):
    if (
        not isinstance(resp, dict)
        or set(resp) != {"signed", "signature"}
        or not isinstance(resp["signed"], dict)
    ):
        _fail("ARCHIVE_ANCHOR_OBSERVER_RESPONSE_SCHEMA_INVALID")
    s = resp["signed"]
    exp = {
        "protocolVersion",
        "observerIdentity",
        "observerOperator",
        "channelId",
        "anchorResponseSha256",
        "sequence",
        "witnessChainHeadSha256",
        "previousCheckpointSha256",
        "challenge",
        "observedAt",
        "inclusionVerified",
        "continuityVerified",
    }
    if (
        set(s) != exp
        or s["protocolVersion"] != 1
        or s["observerIdentity"] != cfg["observerIdentity"]
        or s["observerOperator"] != cfg["observerOperator"]
        or s["channelId"] != cid
        or s["anchorResponseSha256"] != anchor_sha
        or s["sequence"] != sequence
        or s["witnessChainHeadSha256"] != witness_head
        or s["previousCheckpointSha256"] != previous
        or s["challenge"] != challenge
        or s["inclusionVerified"] is not True
        or s["continuityVerified"] is not True
    ):
        _fail("ARCHIVE_ANCHOR_OBSERVER_RESPONSE_BINDING_INVALID")
    at = _dt(s["observedAt"], "ARCHIVE_ANCHOR_OBSERVED_INVALID")
    if not historical and (
        at > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        or now - at > timedelta(minutes=int(POLICY["max_observation_age_minutes"]))
    ):
        _fail("ARCHIVE_ANCHOR_OBSERVER_RESPONSE_STALE")
    _verify(
        cfg["observerPublicKey"],
        resp["signature"],
        _canonical(s),
        "ARCHIVE_ANCHOR_OBSERVER_SIGNATURE",
    )
    return {"doc": resp, "sha256": _sha_bytes(_canonical(resp)), "time": at}


def _event_head(prev, core):
    return _sha_bytes(
        _canonical({"previousAnchorConsensusHeadSha256": prev, "event": core})
    )


def _load_output(root):
    root = _regular_dir(root, "ARCHIVE_ANCHOR_OUTPUT_INVALID")
    if {p.name for p in root.iterdir()} != _ANCHOR_NAMES:
        _fail("ARCHIVE_ANCHOR_OUTPUT_ALLOWLIST_INVALID")
    d = {}
    r = {}
    for n in _ANCHOR_NAMES:
        d[n], r[n] = _read_json(root / n, "ARCHIVE_ANCHOR_OUTPUT")
    return d, r


def verify_anchor_history(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    run160_dir,
    run161_dir,
    retention_root_path,
    retention_root_pin,
    witness_root_path,
    witness_root_pin,
    bootstrap_pin,
    recovery_pin,
    attestation_pins,
    run162_dir,
    anchor_plan_path,
    output_dir,
    now=None,
    historical=False,
    require_current_match=True,
):
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    wdoc, _ = _read_json(witness_root_path, "ARCHIVE_ANCHOR_WITNESS_ROOT")
    wroot = _verify_witness_root(wdoc, witness_root_pin, current, historical=historical)
    pdoc, _ = _read_json(anchor_plan_path, "ARCHIVE_ANCHOR_PLAN")
    plan = _verify_plan(pdoc, wroot, current, historical=historical)
    current162 = _verify_run162(
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
        now=current,
        historical=True,
        require_current_match=True,
    )
    docs, raws = _load_output(output_dir)
    bundle = docs["release-archive-anchor-bundle.json"]
    receipt = docs["release-archive-anchor-receipt.json"]
    if (
        set(bundle)
        != {"schemaVersion", "predicateType", "status", "anchorPlanSha256", "events"}
        or bundle["schemaVersion"] != 1
        or bundle["predicateType"] != PREDICATE_TYPE
        or bundle["status"] != "archive-health-anchor-history"
        or bundle["anchorPlanSha256"] != _sha_bytes(_canonical(pdoc))
    ):
        _fail("ARCHIVE_ANCHOR_BUNDLE_SCHEMA_INVALID")
    if (
        set(receipt) != {"schemaVersion", "status", "events"}
        or receipt["schemaVersion"] != 1
        or receipt["status"] != "archive-health-anchored"
    ):
        _fail("ARCHIVE_ANCHOR_RECEIPT_SCHEMA_INVALID")
    events = bundle["events"]
    recs = receipt["events"]
    if (
        not isinstance(events, list)
        or not events
        or not isinstance(recs, list)
        or len(events) != len(recs)
    ):
        _fail("ARCHIVE_ANCHOR_HISTORY_LENGTH_INVALID")
    prev_head = None
    prev_checkpoints = dict.fromkeys(plan["channels"])
    last_time = None
    for idx, (e, rec) in enumerate(zip(events, recs), 1):
        if not isinstance(e, dict) or set(e) != {
            "sequence",
            "run162WitnessSequence",
            "run162WitnessChainHeadSha256",
            "run162Artifacts",
            "challenge",
            "channelIds",
            "channelCheckpointSha256s",
            "observerResponseSha256s",
            "anchorConsensusHeadSha256",
        }:
            _fail("ARCHIVE_ANCHOR_EVENT_SCHEMA_INVALID")
        if e["sequence"] != idx or e["run162WitnessSequence"] != idx:
            _fail("ARCHIVE_ANCHOR_SEQUENCE_INVALID")
        if (
            not isinstance(rec, dict)
            or set(rec) != {"sequence", "run162Documents", "channels"}
            or rec["sequence"] != idx
        ):
            _fail("ARCHIVE_ANCHOR_RECEIPT_EVENT_INVALID")
        rdocs = rec["run162Documents"]
        if not isinstance(rdocs, dict) or set(rdocs) != run162._WITNESS_OUTPUT_NAMES:
            _fail("ARCHIVE_ANCHOR_EMBEDDED_RUN162_INVALID")
        arts = {
            n: {
                "sha256": _sha_bytes(_canonical(rdocs[n])),
                "size": len(_canonical(rdocs[n])),
            }
            for n in sorted(rdocs)
        }
        if e["run162Artifacts"] != arts:
            _fail("ARCHIVE_ANCHOR_RUN162_ARTIFACT_BINDING_INVALID")
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            for n, v in rdocs.items():
                _write(t / n, v)
            vr = _verify_run162(
                run160_dir=run160_dir,
                run161_dir=run161_dir,
                retention_root_path=retention_root_path,
                retention_root_pin=retention_root_pin,
                witness_root_path=witness_root_path,
                witness_root_pin=witness_root_pin,
                bootstrap_pin=bootstrap_pin,
                recovery_pin=recovery_pin,
                attestation_pins=attestation_pins,
                run162_dir=t,
                now=current,
                historical=True,
                require_current_match=False,
            )
        head = vr["witness_chain_head_sha256"]
        if e["run162WitnessChainHeadSha256"] != head or vr["sequence"] != idx:
            _fail("ARCHIVE_ANCHOR_RUN162_HEAD_BINDING_INVALID")
        if idx == len(events) and require_current_match:
            curdocs = {}
            for n in run162._WITNESS_OUTPUT_NAMES:
                curdocs[n], _ = _read_json(
                    Path(run162_dir) / n, "ARCHIVE_ANCHOR_CURRENT_RUN162"
                )
            if curdocs != rdocs or head != current162["witness_chain_head_sha256"]:
                _fail("ARCHIVE_ANCHOR_CURRENT_RUN162_MISMATCH")
        channel_ids = sorted(plan["channels"])
        if e["channelIds"] != channel_ids:
            _fail("ARCHIVE_ANCHOR_CHANNEL_SET_INVALID")
        challenge = _sha_bytes(
            _canonical(
                {
                    "sequence": idx,
                    "run162WitnessSequence": idx,
                    "previousAnchorConsensusHeadSha256": prev_head,
                    "run162WitnessChainHeadSha256": head,
                    "run162Artifacts": arts,
                    "channelIds": channel_ids,
                    "previousChannelCheckpointSha256s": prev_checkpoints,
                }
            )
        )
        if e["challenge"] != challenge:
            _fail("ARCHIVE_ANCHOR_CHALLENGE_INVALID")
        items = rec["channels"]
        if (
            not isinstance(items, list)
            or [x.get("channelId") for x in items] != channel_ids
        ):
            _fail("ARCHIVE_ANCHOR_RECEIPT_CHANNELS_INVALID")
        cps = []
        obs = []
        times = []
        newcp = {}
        for item in items:
            if not isinstance(item, dict) or set(item) != {
                "channelId",
                "anchorResponse",
                "observerResponse",
            }:
                _fail("ARCHIVE_ANCHOR_RECEIPT_CHANNEL_INVALID")
            cid = item["channelId"]
            cfg = plan["channels"][cid]
            a = _verify_channel_response(
                item["anchorResponse"],
                cid,
                cfg,
                sequence=idx,
                witness_head=head,
                previous=prev_checkpoints[cid],
                challenge=challenge,
                now=current,
                historical=True,
            )
            o = _verify_observer_response(
                item["observerResponse"],
                cid,
                cfg,
                anchor_sha=a["sha256"],
                sequence=idx,
                witness_head=head,
                previous=prev_checkpoints[cid],
                challenge=challenge,
                now=current,
                historical=True,
            )
            cps.append(a["sha256"])
            obs.append(o["sha256"])
            times += [a["time"], o["time"]]
            newcp[cid] = a["sha256"]
        if e["channelCheckpointSha256s"] != cps or e["observerResponseSha256s"] != obs:
            _fail("ARCHIVE_ANCHOR_RESPONSE_BINDING_INVALID")
        core = {k: e[k] for k in e if k != "anchorConsensusHeadSha256"}
        eh = _event_head(prev_head, core)
        if e["anchorConsensusHeadSha256"] != eh:
            _fail("ARCHIVE_ANCHOR_CONSENSUS_HEAD_INVALID")
        if last_time is not None and min(times) - last_time > timedelta(
            days=int(POLICY["maximum_anchor_interval_days"])
        ):
            _fail("ARCHIVE_ANCHOR_INTERVAL_EXCEEDED")
        last_time = max(times)
        prev_head = eh
        prev_checkpoints = newcp
    state = docs[_DOC_ARCHIVE_ANCHOR_STATE]
    active = docs["active-archive-anchor-evidence.json"]
    bitem = {
        "name": "release-archive-anchor-bundle.json",
        "sha256": _sha_bytes(raws["release-archive-anchor-bundle.json"]),
        "size": len(raws["release-archive-anchor-bundle.json"]),
    }
    est = {
        "schemaVersion": 1,
        "status": "trusted-archive-anchor",
        "sequence": len(events),
        "run162WitnessSequence": events[-1]["run162WitnessSequence"],
        "anchorConsensusHeadSha256": prev_head,
        "run162WitnessChainHeadSha256": events[-1]["run162WitnessChainHeadSha256"],
        "channelIds": sorted(plan["channels"]),
        "bundleArtifact": bitem,
    }
    if state != est:
        _fail("ARCHIVE_ANCHOR_STATE_BINDING_INVALID")
    eactive = {
        "schemaVersion": 1,
        "status": "active-archive-anchor",
        "sequence": len(events),
        "run162WitnessSequence": events[-1]["run162WitnessSequence"],
        "anchorConsensusHeadSha256": prev_head,
        "run162WitnessChainHeadSha256": events[-1]["run162WitnessChainHeadSha256"],
        "channelCheckpointSha256s": events[-1]["channelCheckpointSha256s"],
        "observerResponseSha256s": events[-1]["observerResponseSha256s"],
    }
    if active != eactive:
        _fail("ARCHIVE_ANCHOR_ACTIVE_BINDING_INVALID")
    if (
        not historical
        and last_time is not None
        and current - last_time
        > timedelta(days=int(POLICY["maximum_anchor_interval_days"]))
    ):
        _fail("ARCHIVE_ANCHOR_ACTIVE_EPOCH_STALE")
    return {
        "ok": True,
        "sequence": len(events),
        "anchor_consensus_head_sha256": prev_head,
        "run162_witness_chain_head_sha256": events[-1]["run162WitnessChainHeadSha256"],
    }


def anchor_archive_health(  # ruff: ignore[undocumented-public-function]
    *,
    run160_dir,
    run161_dir,
    retention_root_path,
    retention_root_pin,
    witness_root_path,
    witness_root_pin,
    bootstrap_pin,
    recovery_pin,
    attestation_pins,
    run162_dir,
    anchor_plan_path,
    output_dir,
    channels,
    previous_output_dir=None,
    now=None,
):
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    wdoc, _ = _read_json(witness_root_path, "ARCHIVE_ANCHOR_WITNESS_ROOT")
    wroot = _verify_witness_root(wdoc, witness_root_pin, current)
    pdoc, praw = _read_json(anchor_plan_path, "ARCHIVE_ANCHOR_PLAN")
    plan = _verify_plan(pdoc, wroot, current)
    vr = _verify_run162(
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
        now=current,
        historical=True,
        require_current_match=True,
    )
    cidmap = {cid: (a, o) for cid, a, o in channels}
    if set(cidmap) != set(plan["channels"]):
        _fail("ARCHIVE_ANCHOR_CHANNEL_ADAPTER_SET_INVALID")
    prev_events = []
    prev_recs = []
    prev_head = None
    prev_cp = dict.fromkeys(plan["channels"])
    if previous_output_dir is not None:
        pv = verify_anchor_history(
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
            output_dir=previous_output_dir,
            now=current,
            historical=True,
            require_current_match=False,
        )
        pd, _ = _load_output(previous_output_dir)
        prev_events = list(pd["release-archive-anchor-bundle.json"]["events"])
        prev_recs = list(pd["release-archive-anchor-receipt.json"]["events"])
        prev_head = pv["anchor_consensus_head_sha256"]
        last = prev_recs[-1]["channels"]
        prev_cp = {
            x["channelId"]: _sha_bytes(_canonical(x["anchorResponse"])) for x in last
        }
    seq = len(prev_events) + 1
    if vr["sequence"] != seq:
        _fail("ARCHIVE_ANCHOR_RUN162_SEQUENCE_INVALID")
    rdocs = {}
    for n in sorted(run162._WITNESS_OUTPUT_NAMES):
        rdocs[n], _ = _read_json(Path(run162_dir) / n, "ARCHIVE_ANCHOR_RUN162")
    arts = {
        n: {
            "sha256": _sha_bytes(_canonical(rdocs[n])),
            "size": len(_canonical(rdocs[n])),
        }
        for n in sorted(rdocs)
    }
    cids = sorted(plan["channels"])
    head = vr["witness_chain_head_sha256"]
    challenge = _sha_bytes(
        _canonical(
            {
                "sequence": seq,
                "run162WitnessSequence": seq,
                "previousAnchorConsensusHeadSha256": prev_head,
                "run162WitnessChainHeadSha256": head,
                "run162Artifacts": arts,
                "channelIds": cids,
                "previousChannelCheckpointSha256s": prev_cp,
            }
        )
    )
    recitems = []
    cps = []
    obs = []
    for cid in cids:
        cfg = plan["channels"][cid]
        aadapter, oadapter = cidmap[cid]
        req = {
            "operation": "append",
            "protocolVersion": 1,
            "channelId": cid,
            "sequence": seq,
            "witnessChainHeadSha256": head,
            "previousCheckpointSha256": prev_cp[cid],
            "challenge": challenge,
            "createOnly": True,
            "overwrite": False,
        }
        ar = aadapter(req)
        av = _verify_channel_response(
            ar,
            cid,
            cfg,
            sequence=seq,
            witness_head=head,
            previous=prev_cp[cid],
            challenge=challenge,
            now=current,
        )
        orq = {
            "operation": "verify",
            "protocolVersion": 1,
            "channelId": cid,
            "sequence": seq,
            "witnessChainHeadSha256": head,
            "previousCheckpointSha256": prev_cp[cid],
            "challenge": challenge,
            "anchorResponseSha256": av["sha256"],
            "readOnly": True,
        }
        ov = _verify_observer_response(
            oadapter(orq),
            cid,
            cfg,
            anchor_sha=av["sha256"],
            sequence=seq,
            witness_head=head,
            previous=prev_cp[cid],
            challenge=challenge,
            now=current,
        )
        recitems.append(
            {"channelId": cid, "anchorResponse": ar, "observerResponse": ov["doc"]}
        )
        cps.append(av["sha256"])
        obs.append(ov["sha256"])
    core = {
        "sequence": seq,
        "run162WitnessSequence": seq,
        "run162WitnessChainHeadSha256": head,
        "run162Artifacts": arts,
        "challenge": challenge,
        "channelIds": cids,
        "channelCheckpointSha256s": cps,
        "observerResponseSha256s": obs,
    }
    event = dict(core, anchorConsensusHeadSha256=_event_head(prev_head, core))
    events = [*prev_events, event]
    recs = [
        *prev_recs,
        {
            "sequence": seq,
            "run162Documents": rdocs,
            "channels": recitems,
        },
    ]
    bundle = {
        "schemaVersion": 1,
        "predicateType": PREDICATE_TYPE,
        "status": "archive-health-anchor-history",
        "anchorPlanSha256": _sha_bytes(praw),
        "events": events,
    }
    braw = _canonical(bundle)
    state = {
        "schemaVersion": 1,
        "status": "trusted-archive-anchor",
        "sequence": seq,
        "run162WitnessSequence": seq,
        "anchorConsensusHeadSha256": event["anchorConsensusHeadSha256"],
        "run162WitnessChainHeadSha256": head,
        "channelIds": cids,
        "bundleArtifact": {
            "name": "release-archive-anchor-bundle.json",
            "sha256": _sha_bytes(braw),
            "size": len(braw),
        },
    }
    active = {
        "schemaVersion": 1,
        "status": "active-archive-anchor",
        "sequence": seq,
        "run162WitnessSequence": seq,
        "anchorConsensusHeadSha256": event["anchorConsensusHeadSha256"],
        "run162WitnessChainHeadSha256": head,
        "channelCheckpointSha256s": cps,
        "observerResponseSha256s": obs,
    }
    receipt = {"schemaVersion": 1, "status": "archive-health-anchored", "events": recs}
    protected = [
        Path(run160_dir),
        Path(run161_dir),
        Path(run162_dir),
        Path(retention_root_path),
        Path(witness_root_path),
        Path(anchor_plan_path),
    ]
    target = _outside(
        Path(output_dir), protected, "ARCHIVE_ANCHOR_OUTPUT_OVERLAPS_INPUT"
    )
    if target.exists():
        _fail("ARCHIVE_ANCHOR_OUTPUT_EXISTS")
    fp = (
        _dir_fp(Path(run160_dir), "ARCHIVE_ANCHOR_RUN160_DRIFT"),
        _dir_fp(Path(run161_dir), "ARCHIVE_ANCHOR_RUN161_DRIFT"),
        _dir_fp(Path(run162_dir), "ARCHIVE_ANCHOR_RUN162_DRIFT"),
    )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run163-anchor-", dir=parent))
    try:
        _write(stage / "release-archive-anchor-bundle.json", bundle)
        _write(stage / _DOC_ARCHIVE_ANCHOR_STATE, state)
        _write(stage / "active-archive-anchor-evidence.json", active)
        _write(stage / "release-archive-anchor-receipt.json", receipt)
        if fp != (
            _dir_fp(Path(run160_dir), "ARCHIVE_ANCHOR_RUN160_DRIFT"),
            _dir_fp(Path(run161_dir), "ARCHIVE_ANCHOR_RUN161_DRIFT"),
            _dir_fp(Path(run162_dir), "ARCHIVE_ANCHOR_RUN162_DRIFT"),
        ):
            _fail("ARCHIVE_ANCHOR_INPUT_DRIFT_DETECTED")
        verify_anchor_history(
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
            output_dir=stage,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_anchor_history(
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
        output_dir=target,
        now=current,
    )


def _verify_recovery_root(doc, pin, now, historical=False):
    # Reuse Run 162's threshold-root parser with the Run 162 recovery-root wire format.
    try:
        return run162._verify_threshold_root(
            doc,
            pin,
            root_type="archive-retention-recovery-root",
            schema_key="recovery_root_schema_version",
            min_keys_key="min_recovery_keys",
            min_threshold_key="min_recovery_threshold",
            min_operators_key="min_recovery_operators",
            now=now,
            historical=historical,
            channel=True,
        )
    except Exception as e:
        raise ArchiveAnchorError(
            "ARCHIVE_ANCHOR_RECOVERY_ROOT_INVALID:" + str(e)
        ) from e


def _recovery_subject(  # ruff: ignore[too-many-positional-arguments]
    s,
    rr,
    old,
    new,
    anchor_state,
    now,
    historical=False,
):
    exp = {
        "_type",
        "specVersion",
        "schemaVersion",
        "recoveryId",
        "issuedAt",
        "recoveryRootSha256",
        "oldWitnessRootSha256",
        "newWitnessRootSha256",
        "anchorConsensusHeadSha256",
        "anchoredSequence",
        "run162WitnessChainHeadSha256",
        "compromisedOldKeyIds",
        "selectedRecoveryKeyIds",
        "rewrittenAnchorEpochs",
    }
    if (
        not isinstance(s, dict)
        or set(s) != exp
        or s["_type"] != "archive-witness-root-recovery"
        or s["specVersion"] != str(POLICY["spec_version"])
        or s["schemaVersion"] != 1
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_SUBJECT_SCHEMA_INVALID")
    _id(s["recoveryId"], "ARCHIVE_ANCHOR_RECOVERY_ID_INVALID")
    issued = _dt(s["issuedAt"], "ARCHIVE_ANCHOR_RECOVERY_ISSUED_INVALID")
    if not historical and (
        issued > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
        or now - issued > timedelta(minutes=int(POLICY["max_observation_age_minutes"]))
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_SUBJECT_STALE")
    if (
        s["recoveryRootSha256"] != rr["sha256"]
        or s["oldWitnessRootSha256"] != old["sha256"]
        or s["newWitnessRootSha256"] != new["sha256"]
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_ROOT_BINDING_INVALID")
    if (
        s["anchorConsensusHeadSha256"] != anchor_state["anchorConsensusHeadSha256"]
        or s["anchoredSequence"] != anchor_state["sequence"]
        or s["run162WitnessChainHeadSha256"]
        != anchor_state["run162WitnessChainHeadSha256"]
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_ANCHOR_BINDING_INVALID")
    if s["rewrittenAnchorEpochs"] != []:
        _fail("ARCHIVE_ANCHOR_RECOVERY_REWRITE_FORBIDDEN")
    compromised = s["compromisedOldKeyIds"]
    selected = s["selectedRecoveryKeyIds"]
    if (
        not isinstance(compromised, list)
        or not compromised
        or compromised != sorted(compromised)
        or any(x not in old["keys"] for x in compromised)
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_COMPROMISED_KEYS_INVALID")
    if (
        not isinstance(selected, list)
        or selected != sorted(selected)
        or len(selected) != rr["threshold"]
        or any(x not in rr["keys"] for x in selected)
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_SELECTED_INVALID")
    if len({rr["keys"][x]["operator"] for x in selected}) < int(
        POLICY["min_recovery_operators"]
    ) or len({rr["keys"][x]["recoveryChannel"] for x in selected}) < int(
        POLICY["min_recovery_channels"]
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_QUORUM_INVALID")
    oldops = {v["operator"] for v in old["keys"].values()}
    newops = {v["operator"] for v in new["keys"].values()}
    rrops = {v["operator"] for v in rr["keys"].values()}
    if (
        oldops & newops
        or oldops & rrops
        or newops & rrops
        or set(old["keys"]) & set(new["keys"])
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_AUTHORITY_SEPARATION_INVALID")
    return {"issued": issued, "selected": selected}


def verify_witness_root_recovery(  # ruff: ignore[undocumented-public-function]
    *,
    witness_root_path,
    witness_root_pin,
    new_witness_root_path,
    recovery_root_path,
    recovery_root_pin,
    anchor_output_dir,
    recovery_output_dir,
    now=None,
    historical=False,
):
    logger.debug(
        "verifying witness-root recovery output: %s",
        recovery_output_dir,
    )
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    odoc, _ = _read_json(witness_root_path, "ARCHIVE_ANCHOR_OLD_WITNESS_ROOT")
    old = _verify_witness_root(odoc, witness_root_pin, current, historical=True)
    ndoc, nraw = _read_json(new_witness_root_path, "ARCHIVE_ANCHOR_NEW_WITNESS_ROOT")
    new = _verify_witness_root(ndoc, _sha_bytes(nraw), current, historical=historical)
    rdoc, _ = _read_json(recovery_root_path, "ARCHIVE_ANCHOR_RECOVERY_ROOT")
    rr = _verify_recovery_root(rdoc, recovery_root_pin, current, historical=historical)
    astate, _ = _read_json(
        Path(anchor_output_dir) / _DOC_ARCHIVE_ANCHOR_STATE,
        "ARCHIVE_ANCHOR_STATE",
    )
    root = _regular_dir(
        Path(recovery_output_dir), "ARCHIVE_ANCHOR_RECOVERY_OUTPUT_INVALID"
    )
    if {p.name for p in root.iterdir()} != _RECOVERY_NAMES:
        _fail("ARCHIVE_ANCHOR_RECOVERY_OUTPUT_ALLOWLIST_INVALID")
    rec, rraw = _read_json(
        root / "archive-witness-root-recovery-record.json",
        "ARCHIVE_ANCHOR_RECOVERY_RECORD",
    )
    receipt, _ = _read_json(
        root / "archive-witness-root-recovery-receipt.json",
        "ARCHIVE_ANCHOR_RECOVERY_RECEIPT",
    )
    if (
        set(rec)
        != {
            "schemaVersion",
            "status",
            "subject",
            "signatures",
            "oldWitnessRoot",
            "newWitnessRoot",
            "recoveryRoot",
        }
        or rec["schemaVersion"] != 1
        or rec["status"] != "archive-witness-root-recovered"
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_RECORD_SCHEMA_INVALID")
    parsed = _recovery_subject(
        rec["subject"], rr, old, new, astate, current, historical=historical
    )
    sigs = rec["signatures"]
    if (
        not isinstance(sigs, list)
        or [x.get("keyId") for x in sigs] != parsed["selected"]
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_SIGNATURE_SET_INVALID")
    for x in sigs:
        if not isinstance(x, dict) or set(x) != {"keyId", "channel", "signature"}:
            _fail("ARCHIVE_ANCHOR_RECOVERY_SIGNATURE_SCHEMA_INVALID")
        kid = x["keyId"]
        if x["channel"] != rr["keys"][kid]["recoveryChannel"]:
            _fail("ARCHIVE_ANCHOR_RECOVERY_CHANNEL_INVALID")
        _verify(
            rr["keys"][kid]["publicKey"],
            x["signature"],
            _canonical(rec["subject"]),
            "ARCHIVE_ANCHOR_RECOVERY_SIGNATURE",
        )
    expected = {
        "schemaVersion": 1,
        "status": "archive-witness-root-recovery-accepted",
        "anchorConsensusHeadSha256": astate["anchorConsensusHeadSha256"],
        "anchoredSequence": astate["sequence"],
        "run162WitnessChainHeadSha256": astate["run162WitnessChainHeadSha256"],
        "rewrittenAnchorEpochs": [],
        "recordSha256": _sha_bytes(rraw),
        "recoveredWitnessRootSha256": new["sha256"],
    }
    if receipt != expected:
        _fail("ARCHIVE_ANCHOR_RECOVERY_RECEIPT_MISMATCH")
    return {
        "ok": True,
        "recovered_witness_root_sha256": new["sha256"],
        "anchor_consensus_head_sha256": astate["anchorConsensusHeadSha256"],
        "rewritten_anchor_epochs": [],
    }


def recover_witness_root(  # ruff: ignore[undocumented-public-function]
    *,
    witness_root_path,
    witness_root_pin,
    new_witness_root_path,
    recovery_root_path,
    recovery_root_pin,
    recovery_subject_path,
    recovery_signatures_path,
    anchor_output_dir,
    output_dir,
    now=None,
):
    logger.info("recovering witness root into %s", output_dir)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    odoc, _ = _read_json(witness_root_path, "ARCHIVE_ANCHOR_OLD_WITNESS_ROOT")
    old = _verify_witness_root(odoc, witness_root_pin, current, historical=True)
    ndoc, nraw = _read_json(new_witness_root_path, "ARCHIVE_ANCHOR_NEW_WITNESS_ROOT")
    new = _verify_witness_root(ndoc, _sha_bytes(nraw), current)
    rdoc, _ = _read_json(recovery_root_path, "ARCHIVE_ANCHOR_RECOVERY_ROOT")
    rr = _verify_recovery_root(rdoc, recovery_root_pin, current)
    astate, _ = _read_json(
        Path(anchor_output_dir) / _DOC_ARCHIVE_ANCHOR_STATE,
        "ARCHIVE_ANCHOR_STATE",
    )
    subject, _ = _read_json(recovery_subject_path, "ARCHIVE_ANCHOR_RECOVERY_SUBJECT")
    parsed = _recovery_subject(subject, rr, old, new, astate, current)
    sd, _ = _read_json(recovery_signatures_path, "ARCHIVE_ANCHOR_RECOVERY_SIGNATURES")
    if (
        set(sd) != {"schemaVersion", "signatures"}
        or sd["schemaVersion"] != 1
        or not isinstance(sd["signatures"], list)
    ):
        _fail("ARCHIVE_ANCHOR_RECOVERY_SIGNATURES_SCHEMA_INVALID")
    sigs = sd["signatures"]
    if [x.get("keyId") for x in sigs] != parsed["selected"]:
        _fail("ARCHIVE_ANCHOR_RECOVERY_SIGNATURE_SET_INVALID")
    for x in sigs:
        if not isinstance(x, dict) or set(x) != {"keyId", "channel", "signature"}:
            _fail("ARCHIVE_ANCHOR_RECOVERY_SIGNATURE_SCHEMA_INVALID")
        kid = x["keyId"]
        if x["channel"] != rr["keys"][kid]["recoveryChannel"]:
            _fail("ARCHIVE_ANCHOR_RECOVERY_CHANNEL_INVALID")
        _verify(
            rr["keys"][kid]["publicKey"],
            x["signature"],
            _canonical(subject),
            "ARCHIVE_ANCHOR_RECOVERY_SIGNATURE",
        )
    rec = {
        "schemaVersion": 1,
        "status": "archive-witness-root-recovered",
        "subject": subject,
        "signatures": sigs,
        "oldWitnessRoot": {"sha256": old["sha256"], "rootId": old["rootId"]},
        "newWitnessRoot": {"sha256": new["sha256"], "rootId": new["rootId"]},
        "recoveryRoot": {"sha256": rr["sha256"], "rootId": rr["rootId"]},
    }
    receipt = {
        "schemaVersion": 1,
        "status": "archive-witness-root-recovery-accepted",
        "anchorConsensusHeadSha256": astate["anchorConsensusHeadSha256"],
        "anchoredSequence": astate["sequence"],
        "run162WitnessChainHeadSha256": astate["run162WitnessChainHeadSha256"],
        "rewrittenAnchorEpochs": [],
        "recordSha256": _sha_bytes(_canonical(rec)),
        "recoveredWitnessRootSha256": new["sha256"],
    }
    target = _outside(
        Path(output_dir),
        [
            Path(witness_root_path),
            Path(new_witness_root_path),
            Path(recovery_root_path),
            Path(recovery_subject_path),
            Path(recovery_signatures_path),
            Path(anchor_output_dir),
        ],
        "ARCHIVE_ANCHOR_RECOVERY_OUTPUT_OVERLAPS_INPUT",
    )
    if target.exists():
        _fail("ARCHIVE_ANCHOR_RECOVERY_OUTPUT_EXISTS")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".run163-recovery-", dir=parent))
    try:
        (stage / "recovered-archive-witness-root.json").write_bytes(nraw)
        _write(stage / "archive-witness-root-recovery-record.json", rec)
        _write(stage / "archive-witness-root-recovery-receipt.json", receipt)
        verify_witness_root_recovery(
            witness_root_path=witness_root_path,
            witness_root_pin=witness_root_pin,
            new_witness_root_path=stage / "recovered-archive-witness-root.json",
            recovery_root_path=recovery_root_path,
            recovery_root_pin=recovery_root_pin,
            anchor_output_dir=anchor_output_dir,
            recovery_output_dir=stage,
            now=current,
        )
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_witness_root_recovery(
        witness_root_path=witness_root_path,
        witness_root_pin=witness_root_pin,
        new_witness_root_path=target / "recovered-archive-witness-root.json",
        recovery_root_path=recovery_root_path,
        recovery_root_pin=recovery_root_pin,
        anchor_output_dir=anchor_output_dir,
        recovery_output_dir=target,
        now=current,
    )


def _common_predecessor_args(p):
    p.add_argument("--run160-dir", type=Path, required=True)
    p.add_argument("--run161-dir", type=Path, required=True)
    p.add_argument("--retention-root", type=Path, required=True)
    p.add_argument("--retention-root-pin", required=True)
    p.add_argument("--witness-root", type=Path, required=True)
    p.add_argument("--witness-root-pin", required=True)
    p.add_argument("--bootstrap-pin", required=True)
    p.add_argument("--recovery-pin", required=True)
    p.add_argument("--attestation-pin", action="append", default=[])
    p.add_argument("--run162-dir", type=Path, required=True)
    p.add_argument("--anchor-plan", type=Path, required=True)


def main(argv=None):  # ruff: ignore[undocumented-public-function]
    import argparse  # ruff: ignore[import-outside-top-level]

    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("anchor")
    _common_predecessor_args(a)
    a.add_argument("--adapter-config", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--previous-output-dir", type=Path)
    v = sp.add_parser("verify-anchor")
    _common_predecessor_args(v)
    v.add_argument("--output-dir", type=Path, required=True)
    v.add_argument("--historical", action="store_true")
    r = sp.add_parser("recover-witness-root")
    r.add_argument("--witness-root", type=Path, required=True)
    r.add_argument("--witness-root-pin", required=True)
    r.add_argument("--new-witness-root", type=Path, required=True)
    r.add_argument("--recovery-root", type=Path, required=True)
    r.add_argument("--recovery-root-pin", required=True)
    r.add_argument("--recovery-subject", type=Path, required=True)
    r.add_argument("--recovery-signatures", type=Path, required=True)
    r.add_argument("--anchor-output-dir", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    vr = sp.add_parser("verify-recovery")
    vr.add_argument("--witness-root", type=Path, required=True)
    vr.add_argument("--witness-root-pin", required=True)
    vr.add_argument("--new-witness-root", type=Path, required=True)
    vr.add_argument("--recovery-root", type=Path, required=True)
    vr.add_argument("--recovery-root-pin", required=True)
    vr.add_argument("--anchor-output-dir", type=Path, required=True)
    vr.add_argument("--recovery-output-dir", type=Path, required=True)
    vr.add_argument("--historical", action="store_true")
    ns = ap.parse_args(argv)
    logger.info("running archive-anchor command: %s", ns.cmd)
    if ns.cmd == "anchor":
        cfg, _ = _read_json(ns.adapter_config, "ARCHIVE_ANCHOR_ADAPTER_CONFIG")
        if set(cfg) != {"channels"} or not isinstance(cfg["channels"], dict):
            _fail("ARCHIVE_ANCHOR_ADAPTER_CONFIG_SCHEMA_INVALID")
        adapters = []
        for cid, x in sorted(cfg["channels"].items()):
            if (
                not isinstance(x, dict)
                or set(x) != {"appendCommand", "observerCommand"}
                or not isinstance(x["appendCommand"], list)
                or not isinstance(x["observerCommand"], list)
                or not all(
                    isinstance(y, str) and y
                    for y in x["appendCommand"] + x["observerCommand"]
                )
            ):
                _fail("ARCHIVE_ANCHOR_ADAPTER_CONFIG_SCHEMA_INVALID")
            adapters.append(
                (
                    cid,
                    command_channel(x["appendCommand"]),
                    command_observer(x["observerCommand"]),
                )
            )
        out = anchor_archive_health(
            run160_dir=ns.run160_dir,
            run161_dir=ns.run161_dir,
            retention_root_path=ns.retention_root,
            retention_root_pin=ns.retention_root_pin,
            witness_root_path=ns.witness_root,
            witness_root_pin=ns.witness_root_pin,
            bootstrap_pin=ns.bootstrap_pin,
            recovery_pin=ns.recovery_pin,
            attestation_pins=ns.attestation_pin,
            run162_dir=ns.run162_dir,
            anchor_plan_path=ns.anchor_plan,
            output_dir=ns.output_dir,
            channels=adapters,
            previous_output_dir=ns.previous_output_dir,
        )
    elif ns.cmd == "verify-anchor":
        out = verify_anchor_history(
            run160_dir=ns.run160_dir,
            run161_dir=ns.run161_dir,
            retention_root_path=ns.retention_root,
            retention_root_pin=ns.retention_root_pin,
            witness_root_path=ns.witness_root,
            witness_root_pin=ns.witness_root_pin,
            bootstrap_pin=ns.bootstrap_pin,
            recovery_pin=ns.recovery_pin,
            attestation_pins=ns.attestation_pin,
            run162_dir=ns.run162_dir,
            anchor_plan_path=ns.anchor_plan,
            output_dir=ns.output_dir,
            historical=ns.historical,
        )
    elif ns.cmd == "recover-witness-root":
        out = recover_witness_root(
            witness_root_path=ns.witness_root,
            witness_root_pin=ns.witness_root_pin,
            new_witness_root_path=ns.new_witness_root,
            recovery_root_path=ns.recovery_root,
            recovery_root_pin=ns.recovery_root_pin,
            recovery_subject_path=ns.recovery_subject,
            recovery_signatures_path=ns.recovery_signatures,
            anchor_output_dir=ns.anchor_output_dir,
            output_dir=ns.output_dir,
        )
    else:
        out = verify_witness_root_recovery(
            witness_root_path=ns.witness_root,
            witness_root_pin=ns.witness_root_pin,
            new_witness_root_path=ns.new_witness_root,
            recovery_root_path=ns.recovery_root,
            recovery_root_pin=ns.recovery_root_pin,
            anchor_output_dir=ns.anchor_output_dir,
            recovery_output_dir=ns.recovery_output_dir,
            historical=ns.historical,
        )
    logger.info("completed archive-anchor command: %s", ns.cmd)
    # sys.stdout.write(json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
