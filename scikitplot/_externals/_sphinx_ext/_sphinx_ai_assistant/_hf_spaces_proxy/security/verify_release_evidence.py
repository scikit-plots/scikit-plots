#!/usr/bin/env python3
"""
Verify production release evidence without externalizing secrets.

This verifier binds time-sensitive CI/deployment evidence to the exact B39
source lock, Python SBOM, immutable base-image policy, target platform and final
OCI image digest.  It deliberately does not contact registries, scanners,
Redis, or telemetry services; evidence collection is a separate explicit CI or
operator action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import tomllib

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POLICY = tomllib.loads((HERE / "release_evidence_policy.toml").read_text())
SUPPLY = tomllib.loads((HERE / "supply_chain_policy.toml").read_text())
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")
SLSA_PREDICATE = "https://slsa.dev/provenance/v1"
RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
RUNTIME_SOURCE_FILES = (
    "Dockerfile",
    ".dockerignore",
    "requirements.lock",
    "app.py",
    "deduplicate_dataset.py",
)
PROXY_VERSION_RE = re.compile(
    r'^PROXY_VERSION: str = "([0-9]+\.[0-9]+\.[0-9]+)"$', re.MULTILINE
)
CYCLONEDX_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+$")
FORBIDDEN_SECRET_KEYS = {
    "authorization",
    "cookie",
    "password",
    "passwd",
    "secret",
    "token",
    "apikey",
    "api_key",
    "privatekey",
    "private_key",
    "redisurl",
    "redis_url",
    "connectionstring",
    "connection_string",
    "hostname",
    "username",
    "host",
}
SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"(?i)\b(?:redis|rediss|https?)://[^\s/@:]+:[^\s/@]+@"),
)


class EvidenceError(RuntimeError):  # ruff: ignore[undocumented-public-class]
    pass


def _fail(code: str) -> None:
    raise EvidenceError(code)


def _runtime_source_sha256(root: Path = ROOT) -> str:
    """Hash exact application/container inputs with stable names and lengths."""
    paths = [root / name for name in RUNTIME_SOURCE_FILES]
    utils_root = root / "_utils"
    if not utils_root.is_dir() or utils_root.is_symlink():
        _fail("RUNTIME_SOURCE_UTILS_INVALID")
    paths.extend(
        sorted(
            path
            for path in utils_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    )
    h = hashlib.sha256()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            _fail("RUNTIME_SOURCE_FILE_INVALID")
        rel = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_time(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(code)
    try:
        dt = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail(code)
    if dt.tzinfo is None:
        _fail(code)
    return dt.astimezone(timezone.utc)


def _hex(value: Any, code: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        _fail(code)
    return value


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
        _fail(code)
    return value


def _mapping(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _fail(code)


def _tool(value: Any, code: str) -> None:
    tool = _mapping(value, code)
    _exact_keys(tool, {"name", "version"}, code)
    for field in ("name", "version"):
        item = tool.get(field)
        if (
            not isinstance(item, str)
            or not item.strip()
            or len(item) > 128  # ruff: ignore[magic-value-comparison]
            or any(ord(ch) < 32 for ch in item)  # ruff: ignore[magic-value-comparison]
        ):
            _fail(code)


def _current_proxy_version() -> str:
    shared = ROOT / "_utils" / "_shared_logic.py"
    try:
        text = shared.read_text(encoding="utf-8")
    except OSError:
        _fail("PROXY_VERSION_SOURCE_UNREADABLE")
    match = PROXY_VERSION_RE.search(text)
    if match is None:
        _fail("PROXY_VERSION_SOURCE_INVALID")
    return match.group(1)


def _version_tuple(value: Any, code: str) -> tuple[int, int]:
    if not isinstance(value, str) or CYCLONEDX_VERSION_RE.fullmatch(value) is None:
        _fail(code)
    major, minor = value.split(".", 1)
    return int(major), int(minor)


def _bool(value: Any, expected: bool, code: str) -> None:
    if value is not expected:
        _fail(code)


def _walk_values(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).replace("-", "_").lower() in FORBIDDEN_SECRET_KEYS:
                _fail("EVIDENCE_SECRET_FIELD_FORBIDDEN")
            _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            _walk_values(child)
    elif isinstance(value, str):
        if "\x00" in value or len(value) > 4096:  # ruff: ignore[magic-value-comparison]
            _fail("EVIDENCE_STRING_UNSAFE")
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                _fail("EVIDENCE_SECRET_LIKE_VALUE_FORBIDDEN")


def _artifact_path(evidence_dir: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw or raw.startswith(("/", "~")) or "\\" in raw:
        _fail("ARTIFACT_PATH_INVALID")
    rel = Path(raw)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        _fail("ARTIFACT_PATH_TRAVERSAL")
    root = evidence_dir.resolve()
    candidate = root / rel
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            _fail("ARTIFACT_SYMLINK_FORBIDDEN")
    path = candidate.resolve()
    if root not in path.parents:
        _fail("ARTIFACT_PATH_ESCAPE")
    if not path.is_file():
        _fail("ARTIFACT_FILE_MISSING")
    if path.stat().st_size > int(POLICY["max_evidence_file_bytes"]):
        _fail("ARTIFACT_FILE_TOO_LARGE")
    return path


def _verify_artifact(
    evidence_dir: Path,
    obj: Any,
    name: str,
    *,
    subject: str,
    extra_keys: set[str] | None = None,
) -> Path:
    item = _mapping(obj, f"{name.upper()}_INVALID")
    expected_keys = {"path", "sha256", "status", "subject", "tool"} | (
        extra_keys or set()
    )
    _exact_keys(item, expected_keys, f"{name.upper()}_SCHEMA_INVALID")
    _tool(item.get("tool"), f"{name.upper()}_TOOL_INVALID")
    if item.get("status") != "pass":
        _fail(f"{name.upper()}_NOT_PASS")
    if item.get("subject") != subject:
        _fail(f"{name.upper()}_SUBJECT_MISMATCH")
    path = _artifact_path(evidence_dir, item.get("path"))
    expected = _hex(item.get("sha256"), f"{name.upper()}_SHA256_INVALID")
    if _sha256(path) != expected:
        _fail(f"{name.upper()}_HASH_MISMATCH")
    return path


def _verify_provenance(
    path: Path, image_digest: str, base_manifest_digest: str, item: dict[str, Any]
) -> None:
    if item.get("predicateType") != SLSA_PREDICATE:
        _fail("PROVENANCE_PREDICATE_UNSUPPORTED")
    _bool(item.get("signatureVerified"), True, "PROVENANCE_SIGNATURE_NOT_VERIFIED")
    try:
        statement = json.loads(path.read_text())
    except Exception as exc:
        raise EvidenceError("PROVENANCE_JSON_INVALID") from exc
    if (
        not isinstance(statement, dict)
        or statement.get("_type") != "https://in-toto.io/Statement/v1"
    ):
        _fail("PROVENANCE_STATEMENT_INVALID")
    if statement.get("predicateType") != SLSA_PREDICATE:
        _fail("PROVENANCE_STATEMENT_PREDICATE_MISMATCH")
    wanted = image_digest.split(":", 1)[1]
    subjects = statement.get("subject")
    if not isinstance(subjects, list) or not any(
        isinstance(s, dict)
        and isinstance(s.get("digest"), dict)
        and s["digest"].get("sha256") == wanted
        for s in subjects
    ):
        _fail("PROVENANCE_SUBJECT_IMAGE_MISMATCH")
    base_wanted = base_manifest_digest.split(":", 1)[1]
    predicate = statement.get("predicate")
    definition = (
        predicate.get("buildDefinition") if isinstance(predicate, dict) else None
    )
    dependencies = (
        definition.get("resolvedDependencies") if isinstance(definition, dict) else None
    )
    if not isinstance(dependencies, list) or not any(
        isinstance(dep, dict)
        and isinstance(dep.get("digest"), dict)
        and dep["digest"].get("sha256") == base_wanted
        for dep in dependencies
    ):
        _fail("PROVENANCE_BASE_MANIFEST_UNBOUND")


def _verify_redis(doc: dict[str, Any], now: datetime) -> None:
    redis_policy = POLICY["redis"]
    planes = _mapping(doc.get("redis"), "REDIS_EVIDENCE_MISSING")
    _exact_keys(planes, {"rateLimit", "share", "contribution"}, "REDIS_SCHEMA_INVALID")
    for name in ("rateLimit", "share", "contribution"):
        item = _mapping(planes.get(name), f"REDIS_{name.upper()}_MISSING")
        _exact_keys(
            item,
            {
                "tlsVerified",
                "nonDefaultIdentity",
                "leastPrivilegeReviewed",
                "persistenceVerified",
                "replicationVerified",
                "backupRestoreTestedAt",
            },
            f"REDIS_{name.upper()}_SCHEMA_INVALID",
        )
        if redis_policy["require_tls"]:
            _bool(item.get("tlsVerified"), True, f"REDIS_{name.upper()}_TLS_UNVERIFIED")
        if redis_policy["require_non_default_identity"]:
            _bool(
                item.get("nonDefaultIdentity"),
                True,
                f"REDIS_{name.upper()}_DEFAULT_IDENTITY",
            )
        if redis_policy["require_least_privilege_review"]:
            _bool(
                item.get("leastPrivilegeReviewed"),
                True,
                f"REDIS_{name.upper()}_ACL_UNREVIEWED",
            )
    for name, prefix in (("share", "share"), ("contribution", "contribution")):
        item = planes[name]
        if redis_policy[f"{prefix}_require_persistence"]:
            _bool(
                item.get("persistenceVerified"),
                True,
                f"REDIS_{name.upper()}_PERSISTENCE_UNVERIFIED",
            )
        if redis_policy[f"{prefix}_require_replication"]:
            _bool(
                item.get("replicationVerified"),
                True,
                f"REDIS_{name.upper()}_REPLICATION_UNVERIFIED",
            )
        tested = _parse_time(
            item.get("backupRestoreTestedAt"),
            f"REDIS_{name.upper()}_BACKUP_TIME_INVALID",
        )
        if tested > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
            _fail(f"REDIS_{name.upper()}_BACKUP_TIME_FUTURE")
        if now - tested > timedelta(
            days=int(redis_policy["backup_restore_max_age_days"])
        ):
            _fail(f"REDIS_{name.upper()}_BACKUP_RESTORE_STALE")


def _verify_logging(doc: dict[str, Any], now: datetime) -> None:
    logging = _mapping(doc.get("logging"), "LOGGING_EVIDENCE_MISSING")
    _exact_keys(
        logging,
        {
            "requestBodyLogging",
            "authorizationHeaderLogging",
            "capabilityHeaderLogging",
            "queryStringLogging",
            "wafBodyCapture",
            "apmBodyCapture",
            "thirdPartyTelemetryExport",
            "reviewedAt",
        },
        "LOGGING_SCHEMA_INVALID",
    )
    mapping = {
        "requestBodyLogging": "request_body_logging",
        "authorizationHeaderLogging": "authorization_header_logging",
        "capabilityHeaderLogging": "capability_header_logging",
        "queryStringLogging": "query_string_logging",
        "wafBodyCapture": "waf_body_capture",
        "apmBodyCapture": "apm_body_capture",
        "thirdPartyTelemetryExport": "third_party_telemetry_export",
    }
    for evidence_name, policy_name in mapping.items():
        _bool(
            logging.get(evidence_name),
            bool(POLICY["logging"][policy_name]),
            f"LOGGING_{evidence_name.upper()}_POLICY_MISMATCH",
        )
    reviewed = _parse_time(logging.get("reviewedAt"), "LOGGING_REVIEW_TIME_INVALID")
    if reviewed > now + timedelta(minutes=int(POLICY["max_clock_skew_minutes"])):
        _fail("LOGGING_REVIEW_TIME_FUTURE")
    if now - reviewed > timedelta(hours=int(POLICY["max_age_hours"])):
        _fail("LOGGING_REVIEW_STALE")


def verify(  # ruff: ignore[too-many-branches, undocumented-public-function]
    evidence_file: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if evidence_file.is_symlink():
        _fail("EVIDENCE_SYMLINK_FORBIDDEN")
    if not evidence_file.exists() or not evidence_file.is_file():
        _fail("EVIDENCE_FILE_MISSING")
    if evidence_file.stat().st_size > int(POLICY["max_manifest_bytes"]):
        _fail("EVIDENCE_MANIFEST_TOO_LARGE")
    evidence_file = evidence_file.resolve()
    try:
        doc = json.loads(evidence_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvidenceError("EVIDENCE_JSON_INVALID") from exc
    doc = _mapping(doc, "EVIDENCE_ROOT_INVALID")
    _walk_values(doc)
    _exact_keys(
        doc,
        {
            "schemaVersion",
            "release",
            "source",
            "image",
            "artifacts",
            "redis",
            "logging",
            "riskExceptions",
        },
        "EVIDENCE_SCHEMA_FIELDS_INVALID",
    )
    if doc.get("schemaVersion") != POLICY["evidence_schema_version"]:
        _fail("EVIDENCE_SCHEMA_UNSUPPORTED")
    release = _mapping(doc.get("release"), "RELEASE_BLOCK_MISSING")
    _exact_keys(
        release,
        {"releaseId", "generatedAt", "expiresAt", "proxyVersion", "targetPlatform"},
        "RELEASE_SCHEMA_INVALID",
    )
    if (
        not isinstance(release.get("releaseId"), str)
        or RELEASE_ID.fullmatch(release["releaseId"]) is None
    ):
        _fail("RELEASE_ID_INVALID")
    if release.get("targetPlatform") != POLICY["target_platform"]:
        _fail("RELEASE_PLATFORM_MISMATCH")
    if release.get("proxyVersion") != _current_proxy_version():
        _fail("RELEASE_PROXY_VERSION_MISMATCH")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    generated = _parse_time(release.get("generatedAt"), "RELEASE_GENERATED_AT_INVALID")
    expires = _parse_time(release.get("expiresAt"), "RELEASE_EXPIRES_AT_INVALID")
    skew = timedelta(minutes=int(POLICY["max_clock_skew_minutes"]))
    if generated > current + skew:
        _fail("RELEASE_EVIDENCE_FROM_FUTURE")
    if current - generated > timedelta(hours=int(POLICY["max_age_hours"])):
        _fail("RELEASE_EVIDENCE_STALE")
    if expires <= current or expires <= generated:
        _fail("RELEASE_EVIDENCE_EXPIRED")
    if expires - generated > timedelta(hours=int(POLICY["max_age_hours"])):
        _fail("RELEASE_EVIDENCE_EXPIRY_TOO_LONG")

    source = _mapping(doc.get("source"), "SOURCE_BLOCK_MISSING")
    _exact_keys(
        source,
        {
            "requirementsLockSha256",
            "pythonSbomSha256",
            "runtimeSourceSha256",
            "baseImageIndexDigest",
            "baseImageManifestDigest",
        },
        "SOURCE_SCHEMA_INVALID",
    )
    lock_sha = _hex(source.get("requirementsLockSha256"), "LOCK_SHA256_INVALID")
    sbom_sha = _hex(source.get("pythonSbomSha256"), "PYTHON_SBOM_SHA256_INVALID")
    if lock_sha != _sha256(ROOT / SUPPLY["lock_file"]):
        _fail("LOCK_EVIDENCE_SOURCE_MISMATCH")
    if sbom_sha != _sha256(ROOT / SUPPLY["sbom_file"]):
        _fail("PYTHON_SBOM_EVIDENCE_SOURCE_MISMATCH")
    runtime_sha = _hex(
        source.get("runtimeSourceSha256"), "RUNTIME_SOURCE_SHA256_INVALID"
    )
    if runtime_sha != _runtime_source_sha256():
        _fail("RUNTIME_SOURCE_EVIDENCE_MISMATCH")
    if source.get("baseImageIndexDigest") != SUPPLY["base_image"]["index_digest"]:
        _fail("BASE_IMAGE_INDEX_MISMATCH")
    base_manifest_digest = _digest(
        source.get("baseImageManifestDigest"), "BASE_IMAGE_MANIFEST_DIGEST_INVALID"
    )
    if base_manifest_digest == source.get("baseImageIndexDigest"):
        _fail("BASE_IMAGE_MANIFEST_UNRESOLVED")

    image = _mapping(doc.get("image"), "IMAGE_BLOCK_MISSING")
    _exact_keys(image, {"digest", "platform"}, "IMAGE_SCHEMA_INVALID")
    image_digest = _digest(image.get("digest"), "IMAGE_DIGEST_INVALID")
    if image.get("platform") != POLICY["target_platform"]:
        _fail("IMAGE_PLATFORM_MISMATCH")

    artifacts = _mapping(doc.get("artifacts"), "ARTIFACTS_BLOCK_MISSING")
    _exact_keys(
        artifacts,
        {
            "dependencyScan",
            "imageScan",
            "imageSbom",
            "provenance",
            "signatureVerification",
        },
        "ARTIFACTS_SCHEMA_INVALID",
    )
    if POLICY["require_dependency_scan"]:
        _verify_artifact(
            evidence_file.parent,
            artifacts.get("dependencyScan"),
            "dependency_scan",
            subject="sha256:" + lock_sha,
        )
    if POLICY["require_image_vulnerability_scan"]:
        _verify_artifact(
            evidence_file.parent,
            artifacts.get("imageScan"),
            "image_scan",
            subject=image_digest,
        )
    if POLICY["require_full_image_sbom"]:
        sbom_path = _verify_artifact(
            evidence_file.parent,
            artifacts.get("imageSbom"),
            "image_sbom",
            subject=image_digest,
        )
        try:
            image_sbom = json.loads(sbom_path.read_text())
        except Exception as exc:
            raise EvidenceError("IMAGE_SBOM_JSON_INVALID") from exc
        if (
            not isinstance(image_sbom, dict)
            or image_sbom.get("bomFormat") != "CycloneDX"
        ):
            _fail("IMAGE_SBOM_NOT_CYCLONEDX")
        if _version_tuple(
            image_sbom.get("specVersion"), "IMAGE_SBOM_SPEC_VERSION_INVALID"
        ) < _version_tuple(
            POLICY["minimum_cyclonedx_spec_version"], "POLICY_CYCLONEDX_VERSION_INVALID"
        ):
            _fail("IMAGE_SBOM_SPEC_VERSION_TOO_OLD")
    if POLICY["require_slsa_provenance"]:
        provenance_item = _mapping(artifacts.get("provenance"), "PROVENANCE_INVALID")
        provenance_path = _verify_artifact(
            evidence_file.parent,
            provenance_item,
            "provenance",
            subject=image_digest,
            extra_keys={"predicateType", "signatureVerified"},
        )
        _verify_provenance(
            provenance_path, image_digest, base_manifest_digest, provenance_item
        )
    if POLICY["require_signature_verification"]:
        _verify_artifact(
            evidence_file.parent,
            artifacts.get("signatureVerification"),
            "signature_verification",
            subject=image_digest,
        )

    _verify_redis(doc, current)
    if POLICY["require_log_privacy_attestation"]:
        _verify_logging(doc, current)
    exceptions = doc.get("riskExceptions")
    if not isinstance(exceptions, list):
        _fail("RISK_EXCEPTIONS_INVALID")
    if POLICY["forbid_unexpired_risk_exceptions"] and exceptions:
        _fail("RISK_EXCEPTIONS_FORBIDDEN")

    return {
        "ok": True,
        "schema_version": doc["schemaVersion"],
        "release_id": release["releaseId"],
        "proxy_version": release["proxyVersion"],
        "target_platform": release["targetPlatform"],
        "image_digest": image_digest,
        "runtime_source_sha256": runtime_sha,
        "evidence_sha256": _sha256(evidence_file),
        "evidence_expires_at": release["expiresAt"],
        "redis_planes": ["rateLimit", "share", "contribution"],
        "logging_privacy": "verified",
    }


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="Path to release-evidence.json")
    args = parser.parse_args(argv)
    try:
        result = verify(args.evidence)
    except EvidenceError as exc:
        sys.stderr.write(
            json.dumps({"ok": False, "code": str(exc)}, sort_keys=True) + "\n"
        )
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
