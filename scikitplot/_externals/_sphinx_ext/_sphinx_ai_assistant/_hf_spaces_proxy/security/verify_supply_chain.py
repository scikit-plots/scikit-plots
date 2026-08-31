#!/usr/bin/env python3
"""
Offline structural verifier for the B38/B39 supply-chain policy.

This verifier proves committed-file consistency and downgrade ratchets.  It
cannot prove that today's advisory database has no newer finding; networked
release scanning remains a separate mandatory gate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import tomllib

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
POLICY = tomllib.loads((ROOT / "security/supply_chain_policy.toml").read_text())
LOCK = ROOT / POLICY["lock_file"]
SBOM = ROOT / POLICY["sbom_file"]
DIRECT = ROOT / "requirements.txt"
DOCKER = ROOT / "Dockerfile"
IGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.hardened.reference.yml"

LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\s]+) "
    r"--hash=sha256:(?P<digest>[0-9a-f]{64})$"
)


def _version_tuple(text: str) -> tuple[int | str, ...]:
    # Floors in this policy are simple numeric PEP-440 releases. Keep this
    # parser intentionally narrow so unusual versions fail review rather than
    # gaining surprising ordering semantics.
    out: list[int | str] = []
    for part in re.split(r"[.-]", text):
        out.append(int(part) if part.isdigit() else part)
    return tuple(out)


def _locked() -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for raw in LOCK.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE_RE.fullmatch(line)
        if not match:
            raise AssertionError(f"LOCK_LINE_NOT_EXACT_HASHED:{line}")
        name = match.group("name").lower().replace("_", "-")
        if name in result:
            raise AssertionError(f"LOCK_DUPLICATE:{name}")
        result[name] = (match.group("version"), match.group("digest"))
    if not result:
        raise AssertionError("LOCK_EMPTY")
    return result


def verify() -> dict[str, object]:  # ruff: ignore[too-many-branches]
    """Verify."""
    locked = _locked()

    direct: dict[str, str] = {}
    for raw in DIRECT.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if any(token in line for token in (">", "<", "~=", "[", "]", "@", ";")):
            raise AssertionError(f"DIRECT_REQUIREMENT_NOT_EXACT_MINIMAL:{line}")
        if line.count("==") != 1:
            raise AssertionError(f"DIRECT_REQUIREMENT_NOT_EXACT:{line}")
        name, version = line.split("==", 1)
        direct[name.lower().replace("_", "-")] = version
    for name, version in direct.items():
        if locked.get(name, (None,))[0] != version:
            raise AssertionError(f"DIRECT_LOCK_DRIFT:{name}")

    for name, floor in POLICY["advisory_floors"].items():
        normalized = name.lower().replace("_", "-")
        if normalized not in locked:
            raise AssertionError(f"ADVISORY_FLOOR_PACKAGE_MISSING:{normalized}")
        if _version_tuple(locked[normalized][0]) < _version_tuple(str(floor)):
            raise AssertionError(f"ADVISORY_FLOOR_REGRESSION:{normalized}")

    docker = DOCKER.read_text()
    image = POLICY["base_image"]
    expected = f"{image['repository']}:{image['tag']}@{image['index_digest']}"
    required_docker = (
        expected,
        "--require-hashes",
        "--only-binary=:all:",
        "USER 1000:1000",
        "DEPLOYMENT_PROFILE=strict",
        "COPY --from=builder /opt/venv /opt/venv",
        "FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS builder",
        "FROM --platform=linux/amd64 ${PYTHON_IMAGE} AS runtime",
    )
    for marker in required_docker:
        if marker not in docker:
            raise AssertionError(f"DOCKER_HARDENING_MISSING:{marker}")
    if "uvicorn[standard]" in docker or "fastapi[standard]" in docker:
        raise AssertionError("DOCKER_BROAD_EXTRAS_FORBIDDEN")

    ignore_lines = {
        line.strip()
        for line in IGNORE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    if "*" not in ignore_lines:
        raise AssertionError("DOCKERIGNORE_NOT_DENY_BY_DEFAULT")
    for required in ("!requirements.lock", "!app.py", "!_utils/**"):
        if required not in ignore_lines:
            raise AssertionError(f"DOCKERIGNORE_RUNTIME_ALLOWLIST_MISSING:{required}")
    for required in (
        "_utils/__pycache__/",
        "_utils/**/__pycache__/",
        "_utils/**/*.pyc",
        "_utils/**/*.pyo",
    ):
        if required not in ignore_lines:
            raise AssertionError(
                f"DOCKERIGNORE_GENERATED_BYTECODE_EXCLUSION_MISSING:{required}"
            )

    compose = COMPOSE.read_text()
    for marker in (
        "read_only: true",
        'user: "1000:1000"',
        "no-new-privileges:true",
        "cap_drop:",
        "DEPLOYMENT_PROFILE: strict",
    ):
        if marker not in compose:
            raise AssertionError(f"HARDENED_REFERENCE_MISSING:{marker}")

    release_policy = POLICY.get("release_evidence", {})
    for field in (
        "policy_file",
        "example_file",
        "verifier_file",
        "combined_gate_file",
        "redis_probe_file",
        "subject_printer_file",
    ):
        raw = release_policy.get(field)
        if not isinstance(raw, str) or not raw:
            raise AssertionError(f"RELEASE_EVIDENCE_POLICY_MISSING:{field}")
        path = ROOT / raw
        if not path.is_file():
            raise AssertionError(f"RELEASE_EVIDENCE_FILE_MISSING:{field}")
    evidence_policy = tomllib.loads((ROOT / release_policy["policy_file"]).read_text())
    if (int(evidence_policy.get("max_age_hours", 0)) <= 0) or int(
        evidence_policy.get("max_age_hours", 999)
    ) > (
        72  # ruff: ignore[magic-value-comparison]
    ):
        raise AssertionError("RELEASE_EVIDENCE_MAX_AGE_UNSAFE")
    manifest_cap = int(evidence_policy.get("max_manifest_bytes", 0))
    if manifest_cap <= 0 or manifest_cap > 1024 * 1024:
        raise AssertionError("RELEASE_EVIDENCE_MANIFEST_CAP_UNSAFE")
    minimum_cdx = str(evidence_policy.get("minimum_cyclonedx_spec_version", ""))
    if not re.fullmatch(r"[0-9]+\.[0-9]+", minimum_cdx) or _version_tuple(
        minimum_cdx
    ) < _version_tuple("1.6"):
        raise AssertionError("RELEASE_EVIDENCE_CYCLONEDX_FLOOR_UNSAFE")
    if evidence_policy.get("target_platform") != POLICY.get("target_platform"):
        raise AssertionError("RELEASE_EVIDENCE_PLATFORM_DRIFT")
    if evidence_policy.get("forbid_unexpired_risk_exceptions") is not True:
        raise AssertionError("RELEASE_EVIDENCE_RISK_EXCEPTION_BYPASS")
    if evidence_policy.get("logging", {}).get("request_body_logging") is not False:
        raise AssertionError("RELEASE_EVIDENCE_BODY_LOGGING_NOT_FORBIDDEN")
    if (
        evidence_policy.get("logging", {}).get("third_party_telemetry_export")
        is not False
    ):
        raise AssertionError("RELEASE_EVIDENCE_TELEMETRY_NOT_FORBIDDEN")

    sbom = json.loads(SBOM.read_text())
    components = {
        c["name"].lower().replace("_", "-"): c for c in sbom.get("components", [])
    }
    if set(components) != set(locked):
        missing = sorted(set(locked) - set(components))
        extra = sorted(set(components) - set(locked))
        raise AssertionError(f"SBOM_LOCK_SET_DRIFT:missing={missing}:extra={extra}")
    for name, (version, digest) in locked.items():
        comp = components[name]
        if comp.get("version") != version:
            raise AssertionError(f"SBOM_VERSION_DRIFT:{name}")
        hashes = {
            h.get("content")
            for h in comp.get("hashes", [])
            if h.get("alg") == "SHA-256"
        }
        if digest not in hashes:
            raise AssertionError(f"SBOM_HASH_DRIFT:{name}")

    return {
        "ok": True,
        "locked_packages": len(locked),
        "direct_packages": len(direct),
        "base_index_digest": image["index_digest"],
        "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
        "sbom_sha256": hashlib.sha256(SBOM.read_bytes()).hexdigest(),
        "release_evidence_policy": True,
    }


if __name__ == "__main__":
    try:
        sys.stdout.write(json.dumps(verify(), sort_keys=True) + "\n")
    except Exception as exc:  # ruff: ignore[blind-except]
        sys.stderr.write(
            json.dumps({"ok": False, "error": str(exc)}, sort_keys=True) + "\n"
        )
        raise SystemExit(1) from exc
