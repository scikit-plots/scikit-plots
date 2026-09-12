# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Provider-generated binary artifact contract.

Run 143 keeps generated binary bytes outside chat text and outside ZIP tree
authority.  A provider may generate one bounded artifact, but the browser must
review/accept it separately before Run 139 may authorize the corresponding
existing archive path.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

PROVIDER_ARTIFACT_CONTRACT = "scikitplot-provider-artifact-output-v1"
PROVIDER_ARTIFACT_RECEIPT_CONTRACT = "scikitplot-provider-artifact-output-receipt-v1"
PROVIDER_ARTIFACT_MAX_REQUEST_BYTES = 64 * 1024
PROVIDER_ARTIFACT_MAX_PROMPT_CHARS = 12_000
PROVIDER_ARTIFACT_MAX_INSTRUCTIONS_CHARS = 4_000
PROVIDER_ARTIFACT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
PROVIDER_ARTIFACT_MAX_RECEIPT_HEADER_BYTES = 4 * 1024

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GENERATOR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_CANCEL_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_LIFECYCLE_ID = re.compile(r"^[0-9a-f]{32}$")
_DEDUPE_KEY = re.compile(r"^[0-9a-f]{32}$")
_ALLOWED_ROOT = frozenset(
    {
        "contract",
        "generator_id",
        "kind",
        "prompt",
        "mime_type",
        "options",
        "cancel_token",
        "regenerate_of",
        "dedupe_key",
    }
)
_ALLOWED_IMAGE_OPTIONS = frozenset({"size", "quality"})
_ALLOWED_AUDIO_OPTIONS = frozenset({"voice", "instructions"})
_IMAGE_SIZES = frozenset(
    {
        "auto",
        "1024x1024",
        "1536x1024",
        "1024x1536",
        "2048x2048",
        "2048x1152",
        "3840x2160",
        "2160x3840",
    }
)
_IMAGE_QUALITIES = frozenset({"auto", "low", "medium", "high"})
_OPENAI_TTS_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }
)


class ProviderArtifactError(RuntimeError):
    """One provider-artifact request or output crossed a fail-closed boundary."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code or "PROVIDER_ARTIFACT_INVALID")
        super().__init__(message or self.code)


@dataclass(frozen=True)
class ProviderArtifactRequest:
    generator_id: str
    kind: str
    prompt: str
    mime_type: str
    options: Mapping[str, str]
    cancel_token: str = ""
    regenerate_of: str = ""
    dedupe_key: str = ""

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderArtifactGeneratorSpec:
    id: str
    provider: str
    model: str
    kind: str
    mime_types: tuple[str, ...]
    max_output_bytes: int
    option_schema: Mapping[str, Any]
    diagnostic: bool = False

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "kind": self.kind,
            "mime_types": list(self.mime_types),
            "max_output_bytes": int(self.max_output_bytes),
            "options": dict(self.option_schema),
            "diagnostic": bool(self.diagnostic),
        }


@dataclass(frozen=True)
class ProviderArtifactReceipt:
    provider: str
    model: str
    generator_id: str
    kind: str
    mime_type: str
    output_size: int
    output_sha256: str
    prompt_sha256: str
    created_at: int
    lifecycle_id: str = ""
    expires_at: int = 0
    regeneration_of: str = ""
    state: str = "ready"

    def with_lifecycle(
        self,
        *,
        lifecycle_id: str,
        expires_at: int,
        regeneration_of: str = "",
        state: str = "ready",
    ) -> ProviderArtifactReceipt:
        return replace(
            self,
            lifecycle_id=lifecycle_id,
            expires_at=int(expires_at),
            regeneration_of=regeneration_of,
            state=state,
        )

    def as_public_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "contract": PROVIDER_ARTIFACT_RECEIPT_CONTRACT,
            "provider": self.provider,
            "model": self.model,
            "generator_id": self.generator_id,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "output_size": int(self.output_size),
            "output_sha256": self.output_sha256,
            "prompt_sha256": self.prompt_sha256,
            "created_at": int(self.created_at),
        }
        if self.lifecycle_id:
            out.update(
                {
                    "lifecycle_id": self.lifecycle_id,
                    "expires_at": int(self.expires_at),
                    "regeneration_of": self.regeneration_of or None,
                    "state": self.state,
                }
            )
        return out


def _text(value: object, *, field: str, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProviderArtifactError(
            "PROVIDER_ARTIFACT_REQUEST_INVALID", f"{field} must be a string"
        )
    if "\x00" in value or any(
        ord(ch) < 32 and ch not in "\t\n\r"  # ruff: ignore[magic-value-comparison]
        for ch in value
    ):
        raise ProviderArtifactError(
            "PROVIDER_ARTIFACT_REQUEST_INVALID", f"{field} contains control characters"
        )
    raw = value.strip()
    if not allow_empty and not raw:
        raise ProviderArtifactError(
            "PROVIDER_ARTIFACT_REQUEST_INVALID", f"{field} is required"
        )
    if len(raw) > maximum:
        raise ProviderArtifactError(
            "PROVIDER_ARTIFACT_REQUEST_TOO_LARGE", f"{field} is too large"
        )
    return raw


def parse_provider_artifact_request(  # ruff: ignore[too-many-branches]
    body: bytes,
) -> ProviderArtifactRequest:
    if len(body) > PROVIDER_ARTIFACT_MAX_REQUEST_BYTES:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_TOO_LARGE")
    try:
        raw = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID") from exc
    if not isinstance(raw, dict) or set(raw) - _ALLOWED_ROOT:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
    if raw.get("contract") != PROVIDER_ARTIFACT_CONTRACT:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
    generator_id = _text(raw.get("generator_id"), field="generator_id", maximum=128)
    if not _GENERATOR_ID.fullmatch(generator_id):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
    kind = _text(raw.get("kind"), field="kind", maximum=16).lower()
    if kind not in {"image", "audio"}:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_KIND_UNSUPPORTED")
    prompt = _text(
        raw.get("prompt"), field="prompt", maximum=PROVIDER_ARTIFACT_MAX_PROMPT_CHARS
    )
    mime_type = _text(raw.get("mime_type"), field="mime_type", maximum=64).lower()
    allowed_mimes = {
        "image": {"image/png", "image/jpeg", "image/webp"},
        "audio": {"audio/mpeg", "audio/wav"},
    }[kind]
    if mime_type not in allowed_mimes:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_MIME_UNSUPPORTED")
    options_raw = raw.get("options", {})
    if not isinstance(options_raw, dict):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
    allowed_options = (
        _ALLOWED_IMAGE_OPTIONS if kind == "image" else _ALLOWED_AUDIO_OPTIONS
    )
    if set(options_raw) - allowed_options:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
    options: dict[str, str] = {}
    if kind == "image":
        if "size" in options_raw:
            size = _text(options_raw["size"], field="options.size", maximum=32).lower()
            if size not in _IMAGE_SIZES:
                raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
            options["size"] = size
        if "quality" in options_raw:
            quality = _text(
                options_raw["quality"], field="options.quality", maximum=16
            ).lower()
            if quality not in _IMAGE_QUALITIES:
                raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
            options["quality"] = quality
    else:
        if "voice" in options_raw:
            voice = _text(
                options_raw["voice"], field="options.voice", maximum=32
            ).lower()
            if voice not in _OPENAI_TTS_VOICES:
                raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
            options["voice"] = voice
        if "instructions" in options_raw:
            options["instructions"] = _text(
                options_raw["instructions"],
                field="options.instructions",
                maximum=PROVIDER_ARTIFACT_MAX_INSTRUCTIONS_CHARS,
                allow_empty=True,
            )
    cancel_token = str(raw.get("cancel_token") or "").strip().lower()
    if cancel_token and not _CANCEL_TOKEN.fullmatch(cancel_token):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
    regenerate_of = str(raw.get("regenerate_of") or "").strip().lower()
    if regenerate_of and not _LIFECYCLE_ID.fullmatch(regenerate_of):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REGENERATION_INVALID")
    dedupe_key = str(raw.get("dedupe_key") or "").strip().lower()
    if dedupe_key and not _DEDUPE_KEY.fullmatch(dedupe_key):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_REQUEST_INVALID")
    return ProviderArtifactRequest(
        generator_id=generator_id,
        kind=kind,
        prompt=prompt,
        mime_type=mime_type,
        options=options,
        cancel_token=cancel_token,
        regenerate_of=regenerate_of,
        dedupe_key=dedupe_key,
    )


def build_provider_artifact_receipt_from_digest(
    *,
    request: ProviderArtifactRequest,
    spec: ProviderArtifactGeneratorSpec,
    output_size: int,
    output_sha256: str,
) -> ProviderArtifactReceipt:
    if (
        request.generator_id != spec.id
        or request.kind != spec.kind
        or request.mime_type not in spec.mime_types
    ):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_CAPABILITY_MISMATCH")
    if output_size <= 0 or output_size > min(
        spec.max_output_bytes, PROVIDER_ARTIFACT_MAX_OUTPUT_BYTES
    ):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_TOO_LARGE")
    digest = str(output_sha256 or "").lower()
    if not _HEX64.fullmatch(digest):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_INVALID")
    return ProviderArtifactReceipt(
        provider=spec.provider,
        model=spec.model,
        generator_id=spec.id,
        kind=spec.kind,
        mime_type=request.mime_type,
        output_size=output_size,
        output_sha256=digest,
        prompt_sha256=request.prompt_sha256,
        created_at=int(time.time()),
    )


def build_provider_artifact_receipt(
    *,
    request: ProviderArtifactRequest,
    spec: ProviderArtifactGeneratorSpec,
    output: bytes,
) -> ProviderArtifactReceipt:
    if not output:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_INVALID")
    return build_provider_artifact_receipt_from_digest(
        request=request,
        spec=spec,
        output_size=len(output),
        output_sha256=hashlib.sha256(output).hexdigest(),
    )


def encode_provider_artifact_receipt_header(receipt: ProviderArtifactReceipt) -> str:
    payload = json.dumps(
        receipt.as_public_dict(), sort_keys=True, separators=(",", ":")
    )
    raw = payload.encode("utf-8")
    if len(raw) > PROVIDER_ARTIFACT_MAX_RECEIPT_HEADER_BYTES:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_RECEIPT_INVALID")
    return payload


def validate_provider_artifact_signature(
    fileobj: Any, mime_type: str, output_size: int
) -> None:
    """Fail closed when generated bytes do not match the advertised media type."""
    if output_size <= 0 or output_size > PROVIDER_ARTIFACT_MAX_OUTPUT_BYTES:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_TOO_LARGE")
    try:
        pos = fileobj.tell()
        fileobj.seek(0)
        head = fileobj.read(32)
        tail = b""
        if output_size >= 16:  # ruff: ignore[magic-value-comparison]
            fileobj.seek(max(0, output_size - 16))
            tail = fileobj.read(16)
        fileobj.seek(pos)
    except Exception as exc:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_INVALID") from exc
    mime = str(mime_type or "").lower()
    valid = False
    if mime == "image/png":
        valid = head.startswith(b"\x89PNG\r\n\x1a\n") and tail.endswith(
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
    elif mime == "image/jpeg":
        valid = head.startswith(b"\xff\xd8\xff") and tail.endswith(b"\xff\xd9")
    elif mime == "image/webp":
        riff_size = int.from_bytes(head[4:8], "little") + (
            8 if len(head) >= 12 else 0  # ruff: ignore[magic-value-comparison]
        )
        valid = (
            len(head) >= 12  # ruff: ignore[magic-value-comparison]
            and head[:4] == b"RIFF"
            and head[8:12] == b"WEBP"
            and riff_size == output_size
        )
    elif mime == "audio/wav":
        riff_size = int.from_bytes(head[4:8], "little") + (
            8 if len(head) >= 12 else 0  # ruff: ignore[magic-value-comparison]
        )
        valid = (
            len(head) >= 12  # ruff: ignore[magic-value-comparison]
            and head[:4] == b"RIFF"
            and head[8:12] == b"WAVE"
            and riff_size == output_size
        )
    elif mime == "audio/mpeg":
        valid = head.startswith(b"ID3") or (
            len(head) >= 2  # ruff: ignore[magic-value-comparison]
            and head[0] == 0xFF  # ruff: ignore[magic-value-comparison]
            and (head[1] & 0xE0) == 0xE0  # ruff: ignore[magic-value-comparison]
        )
    if not valid:
        raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_TYPE_MISMATCH")


def validate_receipt_shape(receipt: Mapping[str, Any]) -> bool:
    """Small helper used by tests/clients when validating bounded receipts."""
    try:
        return bool(
            receipt.get("contract") == PROVIDER_ARTIFACT_RECEIPT_CONTRACT
            and _GENERATOR_ID.fullmatch(str(receipt.get("generator_id", "")))
            and str(receipt.get("kind", "")) in {"image", "audio"}
            and int(receipt.get("output_size", 0)) > 0
            and _HEX64.fullmatch(str(receipt.get("output_sha256", "")))
            and _HEX64.fullmatch(str(receipt.get("prompt_sha256", "")))
            and (
                not receipt.get("lifecycle_id")
                or _LIFECYCLE_ID.fullmatch(str(receipt.get("lifecycle_id", "")))
            )
            and (
                not receipt.get("regeneration_of")
                or _LIFECYCLE_ID.fullmatch(str(receipt.get("regeneration_of", "")))
            )
        )
    except (TypeError, ValueError):
        return False
