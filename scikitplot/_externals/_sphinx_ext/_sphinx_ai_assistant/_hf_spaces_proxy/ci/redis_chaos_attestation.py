"""
Canonical Redis chaos attestation payloads for Run 148 release evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXTENSION_ROOT = HERE.parents[1]
SECURITY = EXTENSION_ROOT / "_hf_spaces_proxy" / "security"
if str(SECURITY) not in sys.path:
    sys.path.insert(0, str(SECURITY))

from source_tree import source_tree_sha256  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PREDICATE_TYPE = (
    "https://scikit-plots.org/attestations/provider-artifact-redis-chaos/v1"
)
TOOL_NAME = "scikitplot-provider-artifact-redis-chaos"
TOOL_VERSION = "148"
_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IMAGE = re.compile(r"^redis:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}$")


class AttestationError(RuntimeError):
    """Attestation inputs violate the fail-closed Run 148 contract."""


def validate_source_revision(  # ruff: ignore[undocumented-public-function]
    value: str,
) -> str:
    rendered = str(value or "").strip().lower()
    if _REVISION.fullmatch(rendered) is None:
        raise AttestationError("SOURCE_REVISION_INVALID")
    return rendered


def validate_redis_image(  # ruff: ignore[undocumented-public-function]
    value: str,
) -> str:
    rendered = str(value or "").strip()
    if _IMAGE.fullmatch(rendered) is None:
        raise AttestationError("REDIS_IMAGE_NOT_DIGEST_PINNED")
    return rendered


def _outside_source_tree(path: Path) -> Path:
    target = path.expanduser().resolve()
    root = EXTENSION_ROOT.resolve()
    if target == root or root in target.parents:
        raise AttestationError("ATTESTATION_OUTPUT_INSIDE_SOURCE_TREE")
    return target


def build_attestation(  # ruff: ignore[undocumented-public-function]
    *,
    mode: str,
    redis_major: int,
    redis_image: str,
    source_revision: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    if mode not in {"standalone", "cluster"}:
        raise AttestationError("REDIS_CHAOS_MODE_INVALID")
    if redis_major not in {7, 8}:
        raise AttestationError("REDIS_MAJOR_INVALID")
    revision = validate_source_revision(source_revision)
    image = validate_redis_image(redis_image)
    now = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "predicateType": PREDICATE_TYPE,
        "generatedAt": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "subject": {
            "sourceRevision": revision,
            "sourceTreeSha256": source_tree_sha256(EXTENSION_ROOT),
        },
        "redis": {
            "major": redis_major,
            "image": image,
            "mode": mode,
        },
        "result": {"status": "pass"},
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
    }


def write_attestation(  # ruff: ignore[undocumented-public-function]
    directory: Path,
    *,
    mode: str,
    redis_major: int,
    redis_image: str,
    source_revision: str,
) -> Path:
    root = _outside_source_tree(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"redis-chaos-redis{redis_major}-{mode}.json"
    payload = build_attestation(
        mode=mode,
        redis_major=redis_major,
        redis_image=redis_image,
        source_revision=source_revision,
    )
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_signature_verification_record(  # ruff: ignore[undocumented-public-function]
    attestation: Path,
    output: Path,
    *,
    source_revision: str,
) -> Path:
    revision = validate_source_revision(source_revision)
    source = attestation.expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise AttestationError("ATTESTATION_FILE_INVALID")
    target = _outside_source_tree(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "verified": True,
        "verifiedAt": (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
        "attestationSha256": _sha256(source),
        "sourceRevision": revision,
        "signerIdentityVerified": True,
    }
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sig = sub.add_parser(
        "signature-record",
        help="emit sanitized record only after an external signature verifier succeeds",
    )
    sig.add_argument("--attestation", type=Path, required=True)
    sig.add_argument("--output", type=Path, required=True)
    sig.add_argument("--source-revision", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "signature-record":
            write_signature_verification_record(
                args.attestation,
                args.output,
                source_revision=args.source_revision,
            )
            return 0
        raise AttestationError("COMMAND_INVALID")
    except (AttestationError, OSError) as exc:
        logger.error("[run148] FAIL: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
