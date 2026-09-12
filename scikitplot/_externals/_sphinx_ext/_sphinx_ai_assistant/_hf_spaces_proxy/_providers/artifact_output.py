# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
First-class provider binary artifact output executors.

Output generation is intentionally separate from chat/resource input routing.
Input modality support never implies write/output authority.  Only executors
registered here may produce bytes for ``/v1/artifacts/provider-output``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import struct
import tempfile
import wave
import zlib
from dataclasses import dataclass
from typing import Protocol, Sequence

import httpx

try:
    from .._utils._provider_artifact import (
        PROVIDER_ARTIFACT_MAX_OUTPUT_BYTES,
        ProviderArtifactError,
        ProviderArtifactGeneratorSpec,
        ProviderArtifactReceipt,
        ProviderArtifactRequest,
        build_provider_artifact_receipt_from_digest,
        validate_provider_artifact_signature,
    )
except ImportError:  # standalone HF Space: _providers is top-level
    from _utils._provider_artifact import (
        PROVIDER_ARTIFACT_MAX_OUTPUT_BYTES,
        ProviderArtifactError,
        ProviderArtifactGeneratorSpec,
        ProviderArtifactReceipt,
        ProviderArtifactRequest,
        build_provider_artifact_receipt_from_digest,
        validate_provider_artifact_signature,
    )

_OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"
_OPENAI_SPEECH_URL = "https://api.openai.com/v1/audio/speech"
_OPENAI_IMAGE_MODEL = "gpt-image-2"
_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
_IMAGE_MAX_OUTPUT = 16 * 1024 * 1024
_AUDIO_MAX_OUTPUT = 32 * 1024 * 1024
_IMAGE_JSON_MAX = 24 * 1024 * 1024
_COPY_CHUNK = 1024 * 1024


@dataclass
class GeneratedProviderArtifact:
    file: tempfile.SpooledTemporaryFile
    receipt: ProviderArtifactReceipt
    filename: str

    def close(self) -> None:
        self.file.close()


class ProviderArtifactOutputExecutor(Protocol):
    name: str
    enabled: bool

    def specs(self) -> Sequence[ProviderArtifactGeneratorSpec]: ...

    async def generate(
        self, request: ProviderArtifactRequest
    ) -> GeneratedProviderArtifact: ...


class ProviderArtifactOutputRegistry:
    """Generator-id registry separated from resource/chat execution authority."""

    def __init__(self) -> None:
        self._executors: dict[str, ProviderArtifactOutputExecutor] = {}
        self._specs: dict[str, ProviderArtifactGeneratorSpec] = {}

    def clear(self) -> None:
        self._executors.clear()
        self._specs.clear()

    def register(self, executor: ProviderArtifactOutputExecutor) -> None:
        if not bool(getattr(executor, "enabled", False)):
            raise ValueError("provider artifact executor is disabled")
        rows = tuple(executor.specs())
        if not rows:
            raise ValueError("provider artifact executor exposes no generators")
        for spec in rows:
            if spec.id in self._specs:
                raise ValueError("duplicate provider artifact generator id")
            self._specs[spec.id] = spec
            self._executors[spec.id] = executor

    def public_specs(self) -> tuple[ProviderArtifactGeneratorSpec, ...]:
        return tuple(self._specs[key] for key in sorted(self._specs))

    def spec(self, generator_id: str) -> ProviderArtifactGeneratorSpec | None:
        return self._specs.get(str(generator_id or ""))

    async def generate(
        self, request: ProviderArtifactRequest
    ) -> GeneratedProviderArtifact:
        executor = self._executors.get(request.generator_id)
        spec = self._specs.get(request.generator_id)
        if executor is None or spec is None:
            raise ProviderArtifactError("PROVIDER_ARTIFACT_GENERATOR_UNAVAILABLE")
        if request.kind != spec.kind or request.mime_type not in spec.mime_types:
            raise ProviderArtifactError("PROVIDER_ARTIFACT_CAPABILITY_MISMATCH")
        artifact = await executor.generate(request)
        try:
            validate_provider_artifact_signature(
                artifact.file, request.mime_type, artifact.receipt.output_size
            )
        except Exception:
            artifact.close()
            raise
        artifact.file.seek(0)
        return artifact


def _png_bytes(prompt: str) -> bytes:
    """Return a deterministic 32x32 RGB PNG without third-party image libraries."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    width = height = 32
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(
                (
                    digest[(x + y) % 32],
                    digest[(x * 3 + y) % 32],
                    digest[(x + y * 5) % 32],
                )
            )
    raw = zlib.compress(bytes(rows), level=9)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", raw)
        + chunk(b"IEND", b"")
    )


def _wav_bytes(prompt: str) -> bytes:
    """Return a deterministic short PCM WAV fixture derived from the prompt."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    rate = 8000
    duration = 0.20
    frames = int(rate * duration)
    freq = 300 + digest[0]
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        samples = bytearray()
        for i in range(frames):
            value = int(8000 * math.sin(2 * math.pi * freq * i / rate))
            samples.extend(struct.pack("<h", value))
        wav.writeframes(bytes(samples))
    return out.getvalue()


def _artifact_from_bytes(
    *,
    request: ProviderArtifactRequest,
    spec: ProviderArtifactGeneratorSpec,
    payload: bytes,
    filename: str,
) -> GeneratedProviderArtifact:
    if not payload or len(payload) > min(
        spec.max_output_bytes, PROVIDER_ARTIFACT_MAX_OUTPUT_BYTES
    ):
        raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_TOO_LARGE")
    spool = (
        # lint
        tempfile.SpooledTemporaryFile(  # ruff: ignore[open-file-with-context-handler]
            max_size=8 * 1024 * 1024,
            mode="w+b",
        )
    )
    spool.write(payload)
    spool.seek(0)
    receipt = build_provider_artifact_receipt_from_digest(
        request=request,
        spec=spec,
        output_size=len(payload),
        output_sha256=hashlib.sha256(payload).hexdigest(),
    )
    return GeneratedProviderArtifact(file=spool, receipt=receipt, filename=filename)


class StubProviderArtifactOutputExecutor:
    """Deterministic credential-free generator used only when the stub rig is enabled."""

    name = "stub"
    enabled = True

    _SPECS = (
        ProviderArtifactGeneratorSpec(
            id="stub/generated-png",
            provider="stub",
            model="deterministic-png",
            kind="image",
            diagnostic=True,
            mime_types=("image/png",),
            max_output_bytes=1024 * 1024,
            option_schema={},
        ),
        ProviderArtifactGeneratorSpec(
            id="stub/generated-wav",
            provider="stub",
            model="deterministic-wav",
            kind="audio",
            diagnostic=True,
            mime_types=("audio/wav",),
            max_output_bytes=1024 * 1024,
            option_schema={},
        ),
    )

    def specs(self) -> Sequence[ProviderArtifactGeneratorSpec]:
        return self._SPECS

    async def generate(
        self, request: ProviderArtifactRequest
    ) -> GeneratedProviderArtifact:
        spec = next(
            (row for row in self._SPECS if row.id == request.generator_id), None
        )
        if spec is None or request.mime_type not in spec.mime_types:
            raise ProviderArtifactError("PROVIDER_ARTIFACT_CAPABILITY_MISMATCH")
        if spec.kind == "image":
            return _artifact_from_bytes(
                request=request,
                spec=spec,
                payload=_png_bytes(request.prompt),
                filename="generated.png",
            )
        return _artifact_from_bytes(
            request=request,
            spec=spec,
            payload=_wav_bytes(request.prompt),
            filename="generated.wav",
        )


async def _read_limited(response: httpx.Response, maximum: int) -> bytes:
    declared = response.headers.get("content-length")
    if declared:
        try:
            if int(declared) > maximum:
                raise ProviderArtifactError("PROVIDER_ARTIFACT_UPSTREAM_TOO_LARGE")
        except ValueError:
            pass
    out = bytearray()
    async for chunk in response.aiter_bytes():
        out.extend(chunk)
        if len(out) > maximum:
            raise ProviderArtifactError("PROVIDER_ARTIFACT_UPSTREAM_TOO_LARGE")
    return bytes(out)


async def _stream_limited_to_spool(
    response: httpx.Response, maximum: int
) -> tuple[tempfile.SpooledTemporaryFile, int, str]:
    declared = response.headers.get("content-length")
    if declared:
        try:
            if int(declared) > maximum:
                raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_TOO_LARGE")
        except ValueError:
            pass
    spool = (
        # lint
        tempfile.SpooledTemporaryFile(  # ruff: ignore[open-file-with-context-handler]
            max_size=8 * 1024 * 1024,
            mode="w+b",
        )
    )
    digest = hashlib.sha256()
    total = 0
    try:
        async for chunk in response.aiter_bytes(chunk_size=_COPY_CHUNK):
            total += len(chunk)
            if total > maximum:
                raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_TOO_LARGE")
            digest.update(chunk)
            spool.write(chunk)
        if total <= 0:
            raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_INVALID")
        spool.seek(0)
        return spool, total, digest.hexdigest()
    except Exception:
        spool.close()
        raise


class OpenAIProviderArtifactOutputExecutor:
    """Official OpenAI GPT Image / Speech API output executor."""

    name = "openai"
    enabled = True

    _SPECS = (
        ProviderArtifactGeneratorSpec(
            id="openai/gpt-image-2",
            provider="openai",
            model=_OPENAI_IMAGE_MODEL,
            kind="image",
            mime_types=("image/png", "image/jpeg", "image/webp"),
            max_output_bytes=_IMAGE_MAX_OUTPUT,
            option_schema={
                "size": [
                    "auto",
                    "1024x1024",
                    "1536x1024",
                    "1024x1536",
                    "2048x2048",
                    "2048x1152",
                    "3840x2160",
                    "2160x3840",
                ],
                "quality": ["auto", "low", "medium", "high"],
            },
        ),
        ProviderArtifactGeneratorSpec(
            id="openai/gpt-4o-mini-tts",
            provider="openai",
            model=_OPENAI_TTS_MODEL,
            kind="audio",
            mime_types=("audio/mpeg", "audio/wav"),
            max_output_bytes=_AUDIO_MAX_OUTPUT,
            option_schema={
                "voice": [
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
                ],
                "instructions": "string",
            },
        ),
    )

    def __init__(
        self, *, api_key: str, client: httpx.AsyncClient, timeout_seconds: float
    ) -> None:
        token = str(api_key or "").strip()
        if not token:
            raise ValueError("OpenAI artifact output requires a non-empty API key")
        self._api_key = token
        self._client = client
        self._timeout = max(10.0, min(float(timeout_seconds), 600.0))

    def specs(self) -> Sequence[ProviderArtifactGeneratorSpec]:
        return self._SPECS

    def _spec(self, request: ProviderArtifactRequest) -> ProviderArtifactGeneratorSpec:
        spec = next(
            (row for row in self._SPECS if row.id == request.generator_id), None
        )
        if (
            spec is None
            or request.mime_type not in spec.mime_types
            or request.kind != spec.kind
        ):
            raise ProviderArtifactError("PROVIDER_ARTIFACT_CAPABILITY_MISMATCH")
        return spec

    async def generate(
        self, request: ProviderArtifactRequest
    ) -> GeneratedProviderArtifact:
        spec = self._spec(request)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(connect=10.0, read=self._timeout, write=30.0, pool=10.0)
        if spec.kind == "image":
            fmt = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp"}[
                request.mime_type
            ]
            payload = {
                "model": _OPENAI_IMAGE_MODEL,
                "prompt": request.prompt,
                "size": request.options.get("size", "auto"),
                "quality": request.options.get("quality", "auto"),
                "output_format": fmt,
            }
            try:
                response = await self._client.post(
                    _OPENAI_IMAGE_URL, headers=headers, json=payload, timeout=timeout
                )
            except httpx.RequestError as exc:
                raise ProviderArtifactError(
                    "PROVIDER_ARTIFACT_UPSTREAM_UNAVAILABLE"
                ) from exc
            try:
                if response.status_code != 200:  # ruff: ignore[magic-value-comparison]
                    raise ProviderArtifactError("PROVIDER_ARTIFACT_UPSTREAM_REJECTED")
                raw = await _read_limited(response, _IMAGE_JSON_MAX)
            finally:
                await response.aclose()
            try:
                doc = json.loads(raw)
                encoded = doc["data"][0]["b64_json"]
                if not isinstance(encoded, str) or not encoded:
                    raise ValueError
                # Refuse impossible base64 lengths before decoding into process memory.
                if len(encoded) > ((_IMAGE_MAX_OUTPUT + 2) // 3) * 4 + 16:
                    raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_TOO_LARGE")
                binary = base64.b64decode(encoded, validate=True)
            except ProviderArtifactError:
                raise
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ProviderArtifactError("PROVIDER_ARTIFACT_OUTPUT_INVALID") from exc
            suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[
                request.mime_type
            ]
            return _artifact_from_bytes(
                request=request,
                spec=spec,
                payload=binary,
                filename="generated" + suffix,
            )

        response_format = "wav" if request.mime_type == "audio/wav" else "mp3"
        payload = {
            "model": _OPENAI_TTS_MODEL,
            "input": request.prompt,
            "voice": request.options.get("voice", "marin"),
            "response_format": response_format,
        }
        instructions = request.options.get("instructions", "")
        if instructions:
            payload["instructions"] = instructions
        try:
            req = self._client.build_request(
                "POST",
                _OPENAI_SPEECH_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response = await self._client.send(req, stream=True)
        except httpx.RequestError as exc:
            raise ProviderArtifactError(
                "PROVIDER_ARTIFACT_UPSTREAM_UNAVAILABLE"
            ) from exc
        try:
            if response.status_code != 200:  # ruff: ignore[magic-value-comparison]
                raise ProviderArtifactError("PROVIDER_ARTIFACT_UPSTREAM_REJECTED")
            spool, size, digest = await _stream_limited_to_spool(
                response, _AUDIO_MAX_OUTPUT
            )
        finally:
            await response.aclose()
        receipt = build_provider_artifact_receipt_from_digest(
            request=request,
            spec=spec,
            output_size=size,
            output_sha256=digest,
        )
        suffix = ".wav" if request.mime_type == "audio/wav" else ".mp3"
        return GeneratedProviderArtifact(
            file=spool, receipt=receipt, filename="generated" + suffix
        )
