# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
First-class resource descriptors for ``scikitplot-chat-v1``.

The browser supplies names/MIME/modality only as hints.  The proxy owns byte
measurement, hashing and bounded signature classification before any provider
adapter receives authority over an uploaded resource.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

MAX_RESOURCE_COUNT = 256
MAX_RESOURCE_NAME_CHARS = 240
MAX_RESOURCE_MIME_CHARS = 120
MAX_RESOURCE_PATH_CHARS = 1024
MAX_RESOURCE_ID_CHARS = 48
MAX_RESOURCE_DECLARED_BYTES = 2**63 - 1
RESOURCE_SIGNATURE_BYTES = 4096

RESOURCE_MODALITIES = frozenset(
    {
        "text",
        "image",
        "animated_image",
        "vector_image",
        "audio",
        "video",
        "document",
        "archive",
        "data",
        "binary",
    }
)
RESOURCE_INTENTS = frozenset({"auto", "raw", "extract", "context"})
_RESOURCE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,47}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".py",
        ".pyi",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".json",
        ".jsonl",
        ".ipynb",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".tsv",
        ".xml",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".less",
        ".ini",
        ".cfg",
        ".conf",
        ".log",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".bat",
        ".cmd",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".java",
        ".kt",
        ".kts",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".scala",
        ".r",
        ".jl",
    }
)
_DATA_EXTENSIONS = frozenset(
    {".parquet", ".arrow", ".feather", ".xlsx", ".xls", ".ods"}
)


class ResourceContractError(ValueError):
    """A client supplied malformed or contradictory resource metadata."""


@dataclass(frozen=True)
class ResourceDescriptor:
    """Untrusted browser declaration, normalized before byte verification."""

    id: str
    name: str
    mime_type: str
    size: int
    modality: str
    intent: str
    relative_path: str = ""
    archive_name: str = ""


@dataclass(frozen=True)
class VerifiedResource:
    """Proxy-owned resource identity after byte measurement/signature sniffing."""

    descriptor: ResourceDescriptor
    actual_size: int
    sha256: str
    detected_mime: str
    detected_modality: str
    signature: str

    @property
    def id(self) -> str:
        return self.descriptor.id

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def intent(self) -> str:
        return self.descriptor.intent


def _safe_name(value: Any) -> str:
    name = str(value or "").replace("/", " ").replace("\\", " ")
    name = _CONTROL_RE.sub(" ", name).strip()
    if not name:
        raise ResourceContractError("resource name is required")
    return name[:MAX_RESOURCE_NAME_CHARS]


def _safe_relative_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/")
    if not raw:
        return ""
    if len(raw) > MAX_RESOURCE_PATH_CHARS or _CONTROL_RE.search(raw):
        raise ResourceContractError("resource relative_path is invalid")
    if raw.startswith(("/", "//")) or re.match(r"^[A-Za-z]:/", raw):
        raise ResourceContractError("resource relative_path must be relative")
    parts = raw.split("/")

    def _too_long(value: str) -> bool:
        return len(value) > 255  # ruff: ignore[magic-value-comparison]

    if any(not p or p in {".", ".."} or _too_long(p) for p in parts):
        raise ResourceContractError("resource relative_path is unsafe")
    return "/".join(parts)


def _extension(name: str) -> str:
    idx = name.rfind(".")
    return name[idx:].lower() if idx > 0 else ""


def classify_resource(  # ruff: ignore[too-many-branches, too-many-return-statements]
    *, name: str, mime_type: str, prefix: bytes = b""
) -> tuple[str, str, str]:
    """
    Return ``(modality, canonical_mime, signature_label)``.

    Prefix signatures outrank MIME/name hints.  Classification is intentionally
    broad: provider adapters decide semantic support for a specific model.
    """
    sample = bytes(prefix[:RESOURCE_SIGNATURE_BYTES])
    lower = sample.lstrip().lower()
    declared = str(mime_type or "").strip().lower()[:MAX_RESOURCE_MIME_CHARS]
    ext = _extension(name)

    if sample.startswith(b"%PDF-"):
        return "document", "application/pdf", "pdf"
    if sample.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "archive", "application/zip", "zip"
    if sample.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png", "png"
    if sample.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg", "jpeg"
    if sample.startswith((b"GIF87a", b"GIF89a")):
        return "animated_image", "image/gif", "gif"
    if sample.startswith(b"RIFF") and sample[8:12] == b"WEBP":
        return "image", "image/webp", "webp"
    if sample.startswith(b"RIFF") and sample[8:12] == b"WAVE":
        return "audio", "audio/wav", "wav"
    if sample.startswith(b"OggS"):
        return (
            "audio",
            declared if declared.startswith("audio/") else "audio/ogg",
            "ogg",
        )
    if sample.startswith(b"fLaC"):
        return "audio", "audio/flac", "flac"
    if sample.startswith(b"ID3"):
        return "audio", "audio/mpeg", "mp3"
    _len = len(sample) >= 12  # ruff: ignore[magic-value-comparison]
    if _len and sample[4:8] == b"ftyp":  # ruff: ignore[collapsible-if]
        # ISO BMFF covers MP4/MOV and some image formats.  A video declaration or
        # common movie extension is enough to choose video; adapters still own
        # exact codec/container support.
        if declared.startswith("video/") or ext in {
            ".mp4",
            ".m4v",
            ".mov",
            ".3gp",
            ".3gpp",
        }:
            return "video", declared or "video/mp4", "iso-bmff"
    if sample.startswith(b"\x1a\x45\xdf\xa3"):
        if declared.startswith("audio/"):
            return "audio", declared, "ebml"
        return (
            "video",
            declared if declared.startswith("video/") else "video/webm",
            "ebml",
        )
    if b"<svg" in lower[:1024]:
        return "vector_image", "image/svg+xml", "svg"

    if declared == "image/svg+xml" or ext == ".svg":
        return "vector_image", "image/svg+xml", "svg-hint"
    if declared.startswith("image/"):
        return (
            ("animated_image" if declared == "image/gif" else "image"),
            declared,
            "mime",
        )
    if declared.startswith("audio/"):
        return "audio", declared, "mime"
    if declared.startswith("video/"):
        return "video", declared, "mime"
    if declared == "application/pdf" or ext == ".pdf":
        return "document", "application/pdf", "mime"
    if declared in {"application/zip", "application/x-zip-compressed"} or ext == ".zip":
        return "archive", "application/zip", "mime"
    if declared.startswith("text/") or ext in _TEXT_EXTENSIONS:
        return "text", declared or "text/plain", "text-hint"
    if declared in {
        "application/json",
        "application/ld+json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
    }:
        return "text", declared, "structured-text"
    if ext in _DATA_EXTENSIONS or declared in {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.apache.parquet",
    }:
        return "data", declared or "application/octet-stream", "data-hint"
    return "binary", declared or "application/octet-stream", "unknown"


def parse_resource_descriptors(value: Any) -> tuple[ResourceDescriptor, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ResourceContractError("resources must be an array")
    if len(value) > MAX_RESOURCE_COUNT:
        raise ResourceContractError(
            f"resources exceed maximum count {MAX_RESOURCE_COUNT}"
        )

    out: list[ResourceDescriptor] = []
    seen: set[str] = set()
    allowed = {
        "id",
        "name",
        "mime_type",
        "size",
        "modality",
        "intent",
        "relative_path",
        "archive_name",
    }
    for raw in value:
        if not isinstance(raw, dict):
            raise ResourceContractError("each resource must be an object")
        unknown = set(raw) - allowed
        if unknown:
            raise ResourceContractError(
                "unsupported resource field(s): " + ", ".join(sorted(unknown))
            )
        rid = str(raw.get("id") or "")
        if len(rid) > MAX_RESOURCE_ID_CHARS or not _RESOURCE_ID_RE.fullmatch(rid):
            raise ResourceContractError("resource id is invalid")
        if rid in seen:
            raise ResourceContractError("resource id must be unique")
        seen.add(rid)
        name = _safe_name(raw.get("name"))
        mime = str(raw.get("mime_type") or "").strip().lower()
        if len(mime) > MAX_RESOURCE_MIME_CHARS or _CONTROL_RE.search(mime):
            raise ResourceContractError("resource mime_type is invalid")
        size = raw.get("size", 0)
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > MAX_RESOURCE_DECLARED_BYTES
        ):
            raise ResourceContractError("resource size is invalid")
        modality = str(raw.get("modality") or "binary")
        if modality not in RESOURCE_MODALITIES:
            raise ResourceContractError("resource modality is invalid")
        intent = str(raw.get("intent") or "auto")
        if intent not in RESOURCE_INTENTS:
            raise ResourceContractError("resource intent is invalid")
        relative_path = _safe_relative_path(raw.get("relative_path"))
        archive_name = (
            _safe_name(raw.get("archive_name")) if raw.get("archive_name") else ""
        )
        out.append(
            ResourceDescriptor(
                rid, name, mime, size, modality, intent, relative_path, archive_name
            )
        )
    return tuple(out)


def verified_descriptor(
    descriptor: ResourceDescriptor,
    *,
    actual_size: int,
    sha256: str,
    prefix: bytes,
    actual_mime: str = "",
) -> VerifiedResource:
    modality, detected_mime, signature = classify_resource(
        name=descriptor.name,
        mime_type=actual_mime or descriptor.mime_type,
        prefix=prefix,
    )
    # Declared modality is advisory, never authoritative. Keep the original
    # descriptor intact for diagnostics while routing on detected_modality.
    return VerifiedResource(
        descriptor=replace(descriptor),
        actual_size=max(0, int(actual_size)),
        sha256=str(sha256),
        detected_mime=detected_mime,
        detected_modality=modality,
        signature=signature,
    )


def resource_summary(resources: Iterable[VerifiedResource]) -> list[dict[str, Any]]:
    """Privacy-minimal diagnostics for Stub Mirror/receipts; never includes bytes."""
    return [
        {
            "id": row.id,
            "name": row.name,
            "intent": row.intent,
            "declared_modality": row.descriptor.modality,
            "detected_modality": row.detected_modality,
            "declared_mime": row.descriptor.mime_type,
            "detected_mime": row.detected_mime,
            "declared_size": row.descriptor.size,
            "actual_size": row.actual_size,
            "sha256": row.sha256,
            "signature": row.signature,
            "relative_path": row.descriptor.relative_path,
            "archive_name": row.descriptor.archive_name,
        }
        for row in resources
    ]
