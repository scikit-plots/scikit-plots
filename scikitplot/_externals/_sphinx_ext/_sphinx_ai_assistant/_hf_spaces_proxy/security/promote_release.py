"""
Prepare and finalize one immutable, evidence-bound extension release transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXTENSION_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import verify_release_evidence  # noqa: E402
from source_tree import SourceTreeError, source_tree_sha256  # noqa: E402

POLICY = tomllib.loads((HERE / "release_promotion_policy.toml").read_text())
ARCHIVE_PREFIX = str(POLICY["archive_prefix"])
PREDICATE_TYPE = str(POLICY["predicate_type"])
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_IGNORED_DIRS = {".git", ".pytest_cache", "__pycache__"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_MAX_ZIP_RATIO = 500
_CHUNK = 1024 * 1024


class PromotionError(RuntimeError):
    """Release transaction violated a fail-closed promotion invariant."""


def _fail(code: str) -> None:
    raise PromotionError(code)


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


def _regular_file(path: Path, *, max_bytes: int, code: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_file():
        _fail(code)
    resolved = raw.resolve()
    if resolved.stat().st_size > max_bytes:
        _fail(code + "_TOO_LARGE")
    return resolved


def _outside_source(path: Path, source_root: Path) -> Path:
    target = path.expanduser().resolve()
    root = source_root.resolve()
    if target == root or root in target.parents:
        _fail("PROMOTION_OUTPUT_INSIDE_SOURCE_TREE")
    return target


def _input_outside_source(path: Path, source_root: Path, code: str) -> Path:
    resolved = path.resolve()
    root = source_root.resolve()
    if resolved == root or root in resolved.parents:
        _fail(code)
    return resolved


def _release_files(root: Path) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []
    total = 0
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        rendered = rel.as_posix()
        if "\\" in rendered or any(
            ord(ch) < 32  # ruff: ignore[magic-value-comparison]
            or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            or 0x202A <= ord(ch) <= 0x202E  # ruff: ignore[magic-value-comparison]
            or 0x2066 <= ord(ch) <= 0x2069  # ruff: ignore[magic-value-comparison]
            for ch in rendered
        ):
            _fail("SOURCE_TREE_PATH_UNSAFE")
        if any(part in _IGNORED_DIRS for part in rel.parts):
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        if path.is_symlink():
            _fail("SOURCE_TREE_SYMLINK_FORBIDDEN")
        if path.is_file():
            total += path.stat().st_size
            if total > int(POLICY["max_source_total_bytes"]):
                _fail("SOURCE_TREE_TOTAL_TOO_LARGE")
            files.append(path)
            if len(files) > int(POLICY["max_source_files"]):
                _fail("SOURCE_TREE_TOO_MANY_FILES")
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _copy_snapshot(source_root: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for source in _release_files(source_root):
        rel = source.relative_to(source_root)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, dest.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=_CHUNK)
        os.chmod(dest, source.stat().st_mode & 0o777)
    if source_tree_sha256(target) != source_tree_sha256(source_root):
        _fail("SOURCE_SNAPSHOT_HASH_MISMATCH")


def _compare_trees(left: Path, right: Path) -> None:
    lfiles = _release_files(left)
    rfiles = _release_files(right)
    lmap = {p.relative_to(left).as_posix(): p for p in lfiles}
    rmap = {p.relative_to(right).as_posix(): p for p in rfiles}
    if list(lmap) != list(rmap):
        _fail("PATCH_RESULT_TREE_MISMATCH")
    for rel, lp in lmap.items():
        rp = rmap[rel]
        if (lp.stat().st_mode & 0o777) != (rp.stat().st_mode & 0o777):
            _fail("PATCH_RESULT_MODE_MISMATCH")
        if lp.stat().st_size != rp.stat().st_size or _sha256(lp) != _sha256(rp):
            _fail("PATCH_RESULT_CONTENT_MISMATCH")


def _zip_rel(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        _fail("BASELINE_ZIP_PATH_INVALID")
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        _fail("BASELINE_ZIP_PATH_INVALID")
    return rel


def _safe_extract_baseline(  # ruff: ignore[too-many-branches]
    archive: Path,
    destination: Path,
) -> Path:
    prefix = PurePosixPath(ARCHIVE_PREFIX)
    total = 0
    count = 0
    seen: set[str] = set()
    destination.mkdir(parents=True, exist_ok=False)
    try:
        zf = zipfile.ZipFile(archive, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise PromotionError("BASELINE_ZIP_INVALID") from exc
    with zf:
        for info in zf.infolist():
            rel = _zip_rel(info.filename)
            key = rel.as_posix()
            if key in seen:
                _fail("BASELINE_ZIP_DUPLICATE_PATH")
            seen.add(key)
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                _fail("BASELINE_ZIP_COMPRESSION_UNSUPPORTED")
            if not info.is_dir():
                count += 1
                total += info.file_size
                if count > int(POLICY["max_source_files"]):
                    _fail("BASELINE_ZIP_TOO_MANY_FILES")
                if total > int(POLICY["max_source_total_bytes"]):
                    _fail("BASELINE_ZIP_TOTAL_TOO_LARGE")
                if (
                    info.compress_size
                    and info.file_size > info.compress_size * _MAX_ZIP_RATIO
                ):
                    _fail("BASELINE_ZIP_COMPRESSION_RATIO")
            mode = (info.external_attr >> 16) & 0xFFFF
            ftype = stat.S_IFMT(mode)
            if ftype not in {0, stat.S_IFREG, stat.S_IFDIR}:
                _fail("BASELINE_ZIP_SPECIAL_FILE")
            if info.is_dir() and ftype == stat.S_IFREG:
                _fail("BASELINE_ZIP_TYPE_PATH_MISMATCH")
            if not info.is_dir() and ftype == stat.S_IFDIR:
                _fail("BASELINE_ZIP_TYPE_PATH_MISMATCH")
            target = destination.joinpath(*rel.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info, "r") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=_CHUNK)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise PromotionError("BASELINE_ZIP_PAYLOAD_INVALID") from exc
            file_mode = mode & 0o777 or 0o644
            os.chmod(target, file_mode)
    extension = destination.joinpath(*prefix.parts)
    if not extension.is_dir() or extension.is_symlink():
        _fail("BASELINE_EXTENSION_ROOT_MISSING")
    return extension


def _validate_patch_headers(patch: Path) -> None:
    text = patch.read_text(encoding="utf-8", errors="strict")
    saw = False
    for line in text.splitlines():
        if not line.startswith("diff --git "):
            continue
        saw = True
        parts = line.split(" ")
        if len(parts) != 4:  # ruff: ignore[magic-value-comparison]
            _fail("PATCH_HEADER_INVALID")
        for raw, marker in ((parts[2], "a/"), (parts[3], "b/")):
            if not raw.startswith(marker):
                _fail("PATCH_HEADER_INVALID")
            path = raw[2:]
            rel = PurePosixPath(path)
            if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
                _fail("PATCH_PATH_INVALID")
            if (
                any(part in _IGNORED_DIRS for part in rel.parts)
                or rel.suffix in _IGNORED_SUFFIXES
            ):
                _fail("PATCH_IGNORED_ARTIFACT_FORBIDDEN")
    if not saw:
        _fail("PATCH_EMPTY_OR_INVALID")


def _trusted_git_executable() -> str:
    """Resolve the Git binary without trusting the ambient process PATH."""
    override = os.environ.get("SCIKITPLOT_RELEASE_GIT_EXECUTABLE")
    if override is not None:
        raw = Path(override).expanduser()
        if not raw.is_absolute():
            _fail("GIT_EXECUTABLE_PIN_INVALID")
        candidate = raw.resolve()
    else:
        found = shutil.which("git", path=os.defpath)
        if not found:
            _fail("GIT_REQUIRED")
        candidate = Path(found).resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        _fail("GIT_EXECUTABLE_PIN_INVALID" if override is not None else "GIT_REQUIRED")
    return str(candidate)


def _git_apply_environment(root: Path) -> dict[str, str]:
    """Return a secret-free deterministic environment for ``git apply``."""
    home = root / "home"
    xdg = root / "xdg"
    home.mkdir()
    xdg.mkdir()
    empty_config = root / "empty.gitconfig"
    empty_config.write_text("")
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(xdg),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": str(empty_config),
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "LC_ALL": "C",
        "LANG": "C",
    }
    # Windows process creation needs a small OS bootstrap set; these values are not
    # release credentials and do not grant Git configuration authority.
    for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _apply_patch(baseline_extension: Path, patch: Path) -> None:
    _validate_patch_headers(patch)
    git = _trusted_git_executable()
    common = [
        git,
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.filemode=true",
        "-c",
        "core.ignorecase=false",
        "-c",
        "core.safecrlf=false",
        "apply",
        "--binary",
        "--whitespace=nowarn",
    ]
    with tempfile.TemporaryDirectory(prefix="scikitplot-release-git-") as scratch:
        env = _git_apply_environment(Path(scratch))
        check = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [*common, "--check", str(patch)],
            check=False,
            cwd=baseline_extension,
            env=env,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            _fail("PATCH_APPLY_CHECK_FAILED")
        applied = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [*common, str(patch)],
            check=False,
            cwd=baseline_extension,
            env=env,
            capture_output=True,
            text=True,
        )
        if applied.returncode != 0:
            _fail("PATCH_APPLY_FAILED")


def _build_deterministic_zip(snapshot: Path, target: Path) -> tuple[str, int, int]:
    prefix = PurePosixPath(ARCHIVE_PREFIX)
    files = _release_files(snapshot)
    with zipfile.ZipFile(
        target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=False
    ) as zf:
        for path in files:
            rel = PurePosixPath(path.relative_to(snapshot).as_posix())
            arcname = (prefix / rel).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=_ZIP_EPOCH)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | (path.stat().st_mode & 0o777)) << 16
            info.flag_bits |= 0x800
            with path.open("rb") as src, zf.open(info, "w", force_zip64=False) as dst:
                shutil.copyfileobj(src, dst, length=_CHUNK)
    _verify_release_zip(target, snapshot)
    return _sha256(target), target.stat().st_size, len(files)


def _verify_release_zip(archive: Path, snapshot: Path) -> None:
    prefix = PurePosixPath(ARCHIVE_PREFIX)
    expected = {p.relative_to(snapshot).as_posix(): p for p in _release_files(snapshot)}
    seen: list[str] = []
    with zipfile.ZipFile(archive, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            _fail("RELEASE_ZIP_CRC_FAILED")
        for info in zf.infolist():
            rel = _zip_rel(info.filename)
            try:
                local = rel.relative_to(prefix).as_posix()
            except ValueError:
                _fail("RELEASE_ZIP_PREFIX_INVALID")
            if local not in expected or info.is_dir():
                _fail("RELEASE_ZIP_TREE_MISMATCH")
            if local in seen:
                _fail("RELEASE_ZIP_DUPLICATE_PATH")
            seen.append(local)
            source = expected[local]
            mode = ((info.external_attr >> 16) & 0o777) or 0o644
            if mode != (source.stat().st_mode & 0o777):
                _fail("RELEASE_ZIP_MODE_MISMATCH")
            h = hashlib.sha256()
            with zf.open(info, "r") as handle:
                for chunk in iter(lambda: handle.read(_CHUNK), b""):
                    h.update(chunk)
            if h.hexdigest() != _sha256(source):
                _fail("RELEASE_ZIP_CONTENT_MISMATCH")
    if seen != list(expected):
        _fail("RELEASE_ZIP_ORDER_OR_TREE_MISMATCH")


def _read_verified_evidence(
    evidence: Path, verify_result: dict[str, Any]
) -> dict[str, Any]:
    try:
        doc = json.loads(evidence.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PromotionError("EVIDENCE_JSON_INVALID") from exc
    if doc["source"]["sourceTreeSha256"] != verify_result["source_tree_sha256"]:
        _fail("EVIDENCE_RESULT_REBIND_FAILED")
    return doc


def _artifact_ref_from_evidence(
    evidence: Path, doc: dict[str, Any], name: str
) -> dict[str, Any]:
    item = doc["artifacts"][name]
    path = evidence.parent / item["path"]
    if path.is_symlink() or not path.is_file():
        _fail("SBOM_REFERENCE_FILE_INVALID")
    if _sha256(path) != item["sha256"]:
        _fail("SBOM_REFERENCE_HASH_MISMATCH")
    return {
        "sha256": item["sha256"],
        "subject": item["subject"],
        "name": Path(item["path"]).name,
    }


def prepare_release(  # ruff: ignore[undocumented-public-function]
    *,
    evidence: Path,
    baseline_zip: Path,
    patch: Path,
    output_dir: Path,
    source_root: Path = EXTENSION_ROOT,
    evidence_verifier: Callable[..., dict[str, Any]] = verify_release_evidence.verify,
    now: datetime | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    evidence = _regular_file(
        evidence,
        max_bytes=int(verify_release_evidence.POLICY["max_manifest_bytes"]),
        code="EVIDENCE_FILE_INVALID",
    )
    baseline_zip = _regular_file(
        baseline_zip,
        max_bytes=int(POLICY["max_baseline_zip_bytes"]),
        code="BASELINE_ZIP_FILE_INVALID",
    )
    patch = _regular_file(
        patch, max_bytes=int(POLICY["max_patch_bytes"]), code="PATCH_FILE_INVALID"
    )
    evidence = _input_outside_source(
        evidence, source_root, "EVIDENCE_INSIDE_SOURCE_TREE"
    )
    baseline_zip = _input_outside_source(
        baseline_zip, source_root, "BASELINE_ZIP_INSIDE_SOURCE_TREE"
    )
    patch = _input_outside_source(patch, source_root, "PATCH_INSIDE_SOURCE_TREE")
    target = _outside_source(output_dir, source_root)
    if target.exists():
        _fail("PREPARE_OUTPUT_ALREADY_EXISTS")
    verified = evidence_verifier(evidence, now=now)
    source_sha = source_tree_sha256(source_root)
    if verified.get("source_tree_sha256") != source_sha:
        _fail("VERIFIED_SOURCE_SUBJECT_MISMATCH")
    revision = verified.get("source_revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        _fail("VERIFIED_SOURCE_REVISION_INVALID")
    release_id = verified.get("release_id")
    if not isinstance(release_id, str) or _RELEASE_ID.fullmatch(release_id) is None:
        _fail("VERIFIED_RELEASE_ID_INVALID")
    evidence_doc = _read_verified_evidence(evidence, verified)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-prepare-", dir=target.parent
    ) as temp_raw:
        temp = Path(temp_raw)
        snapshot = temp / "snapshot"
        _copy_snapshot(source_root, snapshot)
        if source_tree_sha256(snapshot) != source_sha:
            _fail("SOURCE_SNAPSHOT_SUBJECT_MISMATCH")
        extracted = temp / "baseline"
        baseline_extension = _safe_extract_baseline(baseline_zip, extracted)
        _apply_patch(baseline_extension, patch)
        _compare_trees(baseline_extension, snapshot)

        stage = temp / "stage"
        stage.mkdir()
        zip_name = f"{release_id}.zip"
        patch_name = f"{release_id}.patch"
        zip_path = stage / zip_name
        patch_out = stage / patch_name
        zip_sha, zip_size, file_count = _build_deterministic_zip(snapshot, zip_path)
        shutil.copyfile(patch, patch_out)
        os.chmod(patch_out, 0o644)

        # Package determinism is part of the transaction contract.
        second = stage / ".determinism-check.zip"
        second_sha, _, _ = _build_deterministic_zip(snapshot, second)
        if second_sha != zip_sha or second.read_bytes() != zip_path.read_bytes():
            _fail("RELEASE_ZIP_NONDETERMINISTIC")
        second.unlink()

        if source_tree_sha256(source_root) != source_sha:
            _fail("SOURCE_TREE_CHANGED_DURING_PREPARE")
        if source_tree_sha256(snapshot) != source_sha:
            _fail("SOURCE_SNAPSHOT_CHANGED_DURING_PREPARE")

        statement = {
            "schemaVersion": int(POLICY["statement_schema_version"]),
            "predicateType": PREDICATE_TYPE,
            "generatedAt": _utc(now),
            "release": {
                "releaseId": release_id,
                "proxyVersion": verified["proxy_version"],
                "sourceRevision": revision,
            },
            "subject": {
                "sourceTreeSha256": source_sha,
                "evidenceSha256": verified["evidence_sha256"],
            },
            "artifacts": {
                "zip": {
                    "name": zip_name,
                    "sha256": zip_sha,
                    "size": zip_size,
                    "fileCount": file_count,
                },
                "patch": {
                    "name": patch_name,
                    "sha256": _sha256(patch_out),
                    "size": patch_out.stat().st_size,
                },
                "baselineZip": {
                    "name": baseline_zip.name,
                    "sha256": _sha256(baseline_zip),
                    "size": baseline_zip.stat().st_size,
                },
            },
            "sbomReferences": {
                "pythonRuntime": {
                    "name": Path(verify_release_evidence.SUPPLY["sbom_file"]).name,
                    "sha256": evidence_doc["source"]["pythonSbomSha256"],
                },
                "image": _artifact_ref_from_evidence(
                    evidence, evidence_doc, "imageSbom"
                ),
            },
            "verification": {
                "evidenceVerified": True,
                "patchRecreatesSourceTree": True,
                "zipRecreatesSourceTree": True,
                "deterministicZip": True,
            },
        }
        statement_path = stage / "release-statement.json"
        _write_canonical(statement_path, statement)
        summary = {
            "ok": True,
            "phase": "prepared",
            "release_id": release_id,
            "source_tree_sha256": source_sha,
            "source_revision": revision,
            "release_statement_sha256": _sha256(statement_path),
            "zip_sha256": zip_sha,
            "patch_sha256": _sha256(patch_out),
        }
        target_tmp = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        if target_tmp.exists():
            _fail("PREPARE_TEMP_COLLISION")
        shutil.copytree(stage, target_tmp, copy_function=shutil.copy2)
        os.replace(target_tmp, target)
    return summary


def write_signature_verification_record(  # ruff: ignore[undocumented-public-function]
    *,
    statement: Path,
    verifier_evidence: Path,
    output: Path,
    source_revision: str,
    verifier_name: str,
    verifier_version: str,
    verified_at: datetime | None = None,
) -> Path:
    statement = _regular_file(
        statement, max_bytes=1024 * 1024, code="RELEASE_STATEMENT_INVALID"
    )
    verifier_evidence = _regular_file(
        verifier_evidence,
        max_bytes=1024 * 1024,
        code="SIGNATURE_VERIFIER_EVIDENCE_INVALID",
    )
    if _REVISION.fullmatch(str(source_revision or "")) is None:
        _fail("SOURCE_REVISION_INVALID")
    for value, code in (
        (verifier_name, "SIGNATURE_VERIFIER_NAME_INVALID"),
        (verifier_version, "SIGNATURE_VERIFIER_VERSION_INVALID"),
    ):
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 128  # ruff: ignore[magic-value-comparison]
            or any(ord(ch) < 32 for ch in value)  # ruff: ignore[magic-value-comparison]
        ):
            _fail(code)
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _fail("SIGNATURE_RECORD_OUTPUT_EXISTS")
    payload = {
        "schemaVersion": int(POLICY["signature_record_schema_version"]),
        "verified": True,
        "verifiedAt": _utc(verified_at),
        "releaseStatementSha256": _sha256(statement),
        "sourceRevision": source_revision,
        "signerIdentityVerified": True,
        "verifierEvidenceSha256": _sha256(verifier_evidence),
        "verifier": {
            "name": verifier_name.strip(),
            "version": verifier_version.strip(),
        },
    }
    _write_canonical(target, payload)
    return target


def _safe_artifact_name(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        _fail(code)
    if "/" in value or "\\" in value or "\x00" in value or Path(value).name != value:
        _fail(code)
    return value


def _statement_file_item(
    value: Any, code: str, *, file_count: bool = False
) -> dict[str, Any]:
    expected = {"name", "sha256", "size"} | ({"fileCount"} if file_count else set())
    if not isinstance(value, dict) or set(value) != expected:
        _fail(code + "_SCHEMA_INVALID")
    _safe_artifact_name(value.get("name"), code + "_NAME_INVALID")
    if (
        not isinstance(value.get("sha256"), str)
        or _HEX64.fullmatch(value["sha256"]) is None
    ):
        _fail(code + "_SHA256_INVALID")
    if (
        not isinstance(value.get("size"), int)
        or isinstance(value.get("size"), bool)
        or value["size"] < 0
    ):
        _fail(code + "_SIZE_INVALID")
    if file_count and (
        not isinstance(value.get("fileCount"), int)
        or isinstance(value.get("fileCount"), bool)
        or value["fileCount"] < 1
    ):
        _fail(code + "_FILE_COUNT_INVALID")
    return value


def _validate_statement(  # ruff: ignore[too-many-branches]
    statement_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    if statement_path.is_symlink() or not statement_path.is_file():
        _fail("RELEASE_STATEMENT_FILE_INVALID")
    try:
        doc = json.loads(statement_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PromotionError("RELEASE_STATEMENT_JSON_INVALID") from exc
    expected_root = {
        "schemaVersion",
        "predicateType",
        "generatedAt",
        "release",
        "subject",
        "artifacts",
        "sbomReferences",
        "verification",
    }
    if not isinstance(doc, dict) or set(doc) != expected_root:
        _fail("RELEASE_STATEMENT_SCHEMA_INVALID")
    if (
        doc.get("schemaVersion") != int(POLICY["statement_schema_version"])
        or doc.get("predicateType") != PREDICATE_TYPE
    ):
        _fail("RELEASE_STATEMENT_VERSION_INVALID")
    _parse_time(doc.get("generatedAt"), "RELEASE_STATEMENT_TIME_INVALID")
    release = doc.get("release")
    if not isinstance(release, dict) or set(release) != {
        "releaseId",
        "proxyVersion",
        "sourceRevision",
    }:
        _fail("RELEASE_STATEMENT_RELEASE_SCHEMA_INVALID")
    if (
        not isinstance(release.get("releaseId"), str)
        or _RELEASE_ID.fullmatch(release["releaseId"]) is None
    ):
        _fail("RELEASE_STATEMENT_RELEASE_ID_INVALID")
    if not isinstance(release.get("proxyVersion"), str) or not release["proxyVersion"]:
        _fail("RELEASE_STATEMENT_PROXY_VERSION_INVALID")
    if (
        not isinstance(release.get("sourceRevision"), str)
        or _REVISION.fullmatch(release["sourceRevision"]) is None
    ):
        _fail("RELEASE_STATEMENT_REVISION_INVALID")
    subject = doc.get("subject")
    if not isinstance(subject, dict) or set(subject) != {
        "sourceTreeSha256",
        "evidenceSha256",
    }:
        _fail("RELEASE_STATEMENT_SUBJECT_SCHEMA_INVALID")
    for key in ("sourceTreeSha256", "evidenceSha256"):
        if (
            not isinstance(subject.get(key), str)
            or _HEX64.fullmatch(subject[key]) is None
        ):
            _fail("RELEASE_STATEMENT_SUBJECT_HASH_INVALID")
    if subject["sourceTreeSha256"] != source_tree_sha256(source_root):
        _fail("RELEASE_STATEMENT_SOURCE_MISMATCH")
    artifacts = doc.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "zip",
        "patch",
        "baselineZip",
    }:
        _fail("RELEASE_STATEMENT_ARTIFACTS_SCHEMA_INVALID")
    zip_item = _statement_file_item(
        artifacts.get("zip"), "RELEASE_STATEMENT_ZIP", file_count=True
    )
    patch_item = _statement_file_item(artifacts.get("patch"), "RELEASE_STATEMENT_PATCH")
    baseline_item = _statement_file_item(
        artifacts.get("baselineZip"), "RELEASE_STATEMENT_BASELINE"
    )
    if (
        not zip_item["name"].endswith(".zip")
        or not baseline_item["name"].endswith(".zip")
        or not patch_item["name"].endswith(".patch")
    ):
        _fail("RELEASE_STATEMENT_ARTIFACT_EXTENSION_INVALID")
    if (
        len(
            {
                zip_item["name"],
                patch_item["name"],
                baseline_item["name"],
            }
        )
        != 3  # ruff: ignore[magic-value-comparison]
    ):  # ruff: ignore[magic-value-comparison]
        _fail("RELEASE_STATEMENT_ARTIFACT_NAME_COLLISION")
    sboms = doc.get("sbomReferences")
    if not isinstance(sboms, dict) or set(sboms) != {"pythonRuntime", "image"}:
        _fail("RELEASE_STATEMENT_SBOM_SCHEMA_INVALID")
    python_sbom = sboms.get("pythonRuntime")
    if not isinstance(python_sbom, dict) or set(python_sbom) != {"name", "sha256"}:
        _fail("RELEASE_STATEMENT_PYTHON_SBOM_SCHEMA_INVALID")
    _safe_artifact_name(
        python_sbom.get("name"), "RELEASE_STATEMENT_PYTHON_SBOM_NAME_INVALID"
    )
    if (
        not isinstance(python_sbom.get("sha256"), str)
        or _HEX64.fullmatch(python_sbom["sha256"]) is None
    ):
        _fail("RELEASE_STATEMENT_PYTHON_SBOM_HASH_INVALID")
    image_sbom = sboms.get("image")
    if not isinstance(image_sbom, dict) or set(image_sbom) != {
        "name",
        "sha256",
        "subject",
    }:
        _fail("RELEASE_STATEMENT_IMAGE_SBOM_SCHEMA_INVALID")
    _safe_artifact_name(
        image_sbom.get("name"), "RELEASE_STATEMENT_IMAGE_SBOM_NAME_INVALID"
    )
    if (
        not isinstance(image_sbom.get("sha256"), str)
        or _HEX64.fullmatch(image_sbom["sha256"]) is None
    ):
        _fail("RELEASE_STATEMENT_IMAGE_SBOM_HASH_INVALID")
    if not isinstance(image_sbom.get("subject"), str) or not image_sbom[
        "subject"
    ].startswith("sha256:"):
        _fail("RELEASE_STATEMENT_IMAGE_SBOM_SUBJECT_INVALID")
    verification = doc.get("verification")
    expected_verification = {
        "evidenceVerified": True,
        "patchRecreatesSourceTree": True,
        "zipRecreatesSourceTree": True,
        "deterministicZip": True,
    }
    if verification != expected_verification:
        _fail("RELEASE_STATEMENT_VERIFICATION_INVALID")
    return doc


def _validate_signature_record(  # ruff: ignore[too-many-branches]
    record_path: Path,
    *,
    statement_sha: str,
    revision: str,
    verifier_evidence: Path,
    statement_generated: datetime,
    now: datetime | None,
) -> str:
    try:
        doc = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PromotionError("SIGNATURE_RECORD_JSON_INVALID") from exc
    if not isinstance(doc, dict) or set(doc) != {
        "schemaVersion",
        "verified",
        "verifiedAt",
        "releaseStatementSha256",
        "sourceRevision",
        "signerIdentityVerified",
        "verifierEvidenceSha256",
        "verifier",
    }:
        _fail("SIGNATURE_RECORD_SCHEMA_INVALID")
    if doc.get("schemaVersion") != int(POLICY["signature_record_schema_version"]):
        _fail("SIGNATURE_RECORD_VERSION_INVALID")
    if doc.get("verified") is not True or doc.get("signerIdentityVerified") is not True:
        _fail("SIGNATURE_RECORD_NOT_VERIFIED")
    if doc.get("releaseStatementSha256") != statement_sha:
        _fail("SIGNATURE_RECORD_SUBJECT_MISMATCH")
    if doc.get("sourceRevision") != revision:
        _fail("SIGNATURE_RECORD_REVISION_MISMATCH")
    if doc.get("verifierEvidenceSha256") != _sha256(verifier_evidence):
        _fail("SIGNATURE_VERIFIER_EVIDENCE_HASH_MISMATCH")
    verifier = doc.get("verifier")
    if not isinstance(verifier, dict) or set(verifier) != {"name", "version"}:
        _fail("SIGNATURE_VERIFIER_SCHEMA_INVALID")
    for field in ("name", "version"):
        value = verifier.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 128  # ruff: ignore[magic-value-comparison]
            or any(ord(ch) < 32 for ch in value)  # ruff: ignore[magic-value-comparison]
        ):
            _fail("SIGNATURE_VERIFIER_FIELD_INVALID")
    verified_at = _parse_time(doc.get("verifiedAt"), "SIGNATURE_RECORD_TIME_INVALID")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if verified_at > current + timedelta(minutes=10):
        _fail("SIGNATURE_RECORD_FROM_FUTURE")
    if current - verified_at > timedelta(hours=int(POLICY["max_signature_age_hours"])):
        _fail("SIGNATURE_RECORD_STALE")
    if verified_at < statement_generated - timedelta(minutes=10):
        _fail("SIGNATURE_RECORD_PREDATES_STATEMENT")
    return _sha256(record_path)


def finalize_release(  # ruff: ignore[too-many-branches, undocumented-public-function]
    *,
    evidence: Path,
    prepared_dir: Path,
    baseline_zip: Path,
    signature_record: Path,
    signature_verifier_evidence: Path,
    promotion_dir: Path,
    source_root: Path = EXTENSION_ROOT,
    evidence_verifier: Callable[..., dict[str, Any]] = verify_release_evidence.verify,
    now: datetime | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    prepared = _outside_source(prepared_dir, source_root)
    if not prepared.is_dir() or prepared.is_symlink():
        _fail("PREPARED_DIR_INVALID")
    target = _outside_source(promotion_dir, source_root)
    if target.exists():
        _fail("PROMOTION_OUTPUT_ALREADY_EXISTS")
    evidence = _regular_file(
        evidence,
        max_bytes=int(verify_release_evidence.POLICY["max_manifest_bytes"]),
        code="EVIDENCE_FILE_INVALID",
    )
    baseline_zip = _regular_file(
        baseline_zip,
        max_bytes=int(POLICY["max_baseline_zip_bytes"]),
        code="BASELINE_ZIP_FILE_INVALID",
    )
    signature_record = _regular_file(
        signature_record, max_bytes=1024 * 1024, code="SIGNATURE_RECORD_INVALID"
    )
    signature_verifier_evidence = _regular_file(
        signature_verifier_evidence,
        max_bytes=1024 * 1024,
        code="SIGNATURE_VERIFIER_EVIDENCE_INVALID",
    )
    evidence = _input_outside_source(
        evidence, source_root, "EVIDENCE_INSIDE_SOURCE_TREE"
    )
    baseline_zip = _input_outside_source(
        baseline_zip, source_root, "BASELINE_ZIP_INSIDE_SOURCE_TREE"
    )
    signature_record = _input_outside_source(
        signature_record, source_root, "SIGNATURE_RECORD_INSIDE_SOURCE_TREE"
    )
    signature_verifier_evidence = _input_outside_source(
        signature_verifier_evidence,
        source_root,
        "SIGNATURE_VERIFIER_EVIDENCE_INSIDE_SOURCE_TREE",
    )

    verified = evidence_verifier(evidence, now=now)
    statement_path = prepared / "release-statement.json"
    statement = _validate_statement(statement_path, source_root)
    statement_sha = _sha256(statement_path)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    statement_generated = _parse_time(
        statement["generatedAt"], "RELEASE_STATEMENT_TIME_INVALID"
    )
    if statement_generated > current + timedelta(minutes=10):
        _fail("RELEASE_STATEMENT_FROM_FUTURE")
    if current - statement_generated > timedelta(
        hours=int(POLICY["max_signature_age_hours"])
    ):
        _fail("RELEASE_STATEMENT_STALE")
    if statement["subject"].get("evidenceSha256") != verified["evidence_sha256"]:
        _fail("RELEASE_STATEMENT_EVIDENCE_MISMATCH")
    if (
        statement["release"].get("releaseId") != verified["release_id"]
        or statement["release"].get("sourceRevision") != verified["source_revision"]
    ):
        _fail("RELEASE_STATEMENT_RELEASE_MISMATCH")
    revision = verified["source_revision"]
    signature_sha = _validate_signature_record(
        signature_record,
        statement_sha=statement_sha,
        revision=revision,
        verifier_evidence=signature_verifier_evidence,
        statement_generated=statement_generated,
        now=now,
    )
    if (
        statement["artifacts"]["baselineZip"]["name"] != baseline_zip.name
        or statement["artifacts"]["baselineZip"]["sha256"] != _sha256(baseline_zip)
        or statement["artifacts"]["baselineZip"]["size"] != baseline_zip.stat().st_size
    ):
        _fail("PROMOTION_BASELINE_HASH_MISMATCH")

    evidence_doc = _read_verified_evidence(evidence, verified)
    expected_python_sbom = {
        "name": Path(verify_release_evidence.SUPPLY["sbom_file"]).name,
        "sha256": evidence_doc["source"]["pythonSbomSha256"],
    }
    expected_image_sbom = _artifact_ref_from_evidence(
        evidence, evidence_doc, "imageSbom"
    )
    if (
        statement["sbomReferences"]["pythonRuntime"] != expected_python_sbom
        or statement["sbomReferences"]["image"] != expected_image_sbom
    ):
        _fail("PROMOTION_SBOM_REFERENCE_MISMATCH")

    zip_item = statement["artifacts"]["zip"]
    patch_item = statement["artifacts"]["patch"]
    zip_path = prepared / zip_item["name"]
    patch_path = prepared / patch_item["name"]
    for path, item, code in (
        (zip_path, zip_item, "ZIP"),
        (patch_path, patch_item, "PATCH"),
    ):
        if path.is_symlink() or not path.is_file():
            _fail(f"PROMOTION_{code}_MISSING")
        if _sha256(path) != item.get("sha256") or path.stat().st_size != item.get(
            "size"
        ):
            _fail(f"PROMOTION_{code}_HASH_MISMATCH")
    _verify_release_zip(zip_path, source_root)
    if zip_item["fileCount"] != len(_release_files(source_root)):
        _fail("PROMOTION_ZIP_FILE_COUNT_MISMATCH")
    with tempfile.TemporaryDirectory(prefix="release-finalize-patch-") as temp_raw:
        baseline_extension = _safe_extract_baseline(
            baseline_zip, Path(temp_raw) / "baseline"
        )
        _apply_patch(baseline_extension, patch_path)
        _compare_trees(baseline_extension, source_root)
    if source_tree_sha256(source_root) != verified["source_tree_sha256"]:
        _fail("SOURCE_TREE_CHANGED_BEFORE_FINALIZE")

    receipt = {
        "schemaVersion": int(POLICY["receipt_schema_version"]),
        "finalizedAt": _utc(now),
        "releaseId": verified["release_id"],
        "sourceRevision": revision,
        "sourceTreeSha256": verified["source_tree_sha256"],
        "evidenceSha256": verified["evidence_sha256"],
        "releaseStatementSha256": statement_sha,
        "signatureVerificationSha256": signature_sha,
        "signatureVerifierEvidenceSha256": _sha256(signature_verifier_evidence),
        "publish": [
            {
                "name": zip_item["name"],
                "sha256": zip_item["sha256"],
                "size": zip_item["size"],
            },
            {
                "name": patch_item["name"],
                "sha256": patch_item["sha256"],
                "size": patch_item["size"],
            },
            {
                "name": "release-statement.json",
                "sha256": statement_sha,
                "size": statement_path.stat().st_size,
            },
            {
                "name": "release-statement.signature-verification.json",
                "sha256": signature_sha,
                "size": signature_record.stat().st_size,
            },
        ],
        "sbomReferences": statement["sbomReferences"],
        "status": "promoted",
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-finalize-", dir=target.parent
    ) as temp_raw:
        stage = Path(temp_raw) / "promoted"
        stage.mkdir()
        shutil.copy2(zip_path, stage / zip_path.name)
        shutil.copy2(patch_path, stage / patch_path.name)
        shutil.copy2(statement_path, stage / "release-statement.json")
        shutil.copy2(
            signature_record, stage / "release-statement.signature-verification.json"
        )
        receipt_path = stage / "promotion-receipt.json"
        _write_canonical(receipt_path, receipt)
        # Rebind every publishable byte immediately before the atomic directory move.
        for item in receipt["publish"]:
            path = stage / item["name"]
            if _sha256(path) != item["sha256"] or path.stat().st_size != item["size"]:
                _fail("PROMOTION_STAGE_REBIND_FAILED")
        if source_tree_sha256(source_root) != verified["source_tree_sha256"]:
            _fail("SOURCE_TREE_CHANGED_DURING_FINALIZE")
        final_tmp = target.with_name(target.name + ".tmp-" + os.urandom(6).hex())
        shutil.copytree(stage, final_tmp, copy_function=shutil.copy2)
        os.replace(final_tmp, target)
    return {
        "ok": True,
        "phase": "promoted",
        "release_id": verified["release_id"],
        "source_tree_sha256": verified["source_tree_sha256"],
        "release_statement_sha256": statement_sha,
        "promotion_receipt_sha256": _sha256(target / "promotion-receipt.json"),
    }


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--evidence", type=Path, required=True)
    prepare.add_argument("--baseline-zip", type=Path, required=True)
    prepare.add_argument("--patch", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    sig = sub.add_parser("signature-record")
    sig.add_argument("--statement", type=Path, required=True)
    sig.add_argument("--verifier-evidence", type=Path, required=True)
    sig.add_argument("--output", type=Path, required=True)
    sig.add_argument("--source-revision", required=True)
    sig.add_argument("--verifier-name", required=True)
    sig.add_argument("--verifier-version", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--evidence", type=Path, required=True)
    finalize.add_argument("--prepared-dir", type=Path, required=True)
    finalize.add_argument("--baseline-zip", type=Path, required=True)
    finalize.add_argument("--signature-record", type=Path, required=True)
    finalize.add_argument("--signature-verifier-evidence", type=Path, required=True)
    finalize.add_argument("--promotion-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_release(
                evidence=args.evidence,
                baseline_zip=args.baseline_zip,
                patch=args.patch,
                output_dir=args.output_dir,
            )
        elif args.command == "signature-record":
            path = write_signature_verification_record(
                statement=args.statement,
                verifier_evidence=args.verifier_evidence,
                output=args.output,
                source_revision=args.source_revision,
                verifier_name=args.verifier_name,
                verifier_version=args.verifier_version,
            )
            result = {
                "ok": True,
                "phase": "signature-record",
                "path": path.name,
                "sha256": _sha256(path),
            }
        else:
            result = finalize_release(
                evidence=args.evidence,
                prepared_dir=args.prepared_dir,
                baseline_zip=args.baseline_zip,
                signature_record=args.signature_record,
                signature_verifier_evidence=args.signature_verifier_evidence,
                promotion_dir=args.promotion_dir,
            )
    except (
        PromotionError,
        verify_release_evidence.EvidenceError,
        SourceTreeError,
        OSError,
        UnicodeError,
        zipfile.BadZipFile,
    ) as exc:
        sys.stderr.write(
            json.dumps({"ok": False, "code": str(exc)}, sort_keys=True) + "\n"
        )
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
