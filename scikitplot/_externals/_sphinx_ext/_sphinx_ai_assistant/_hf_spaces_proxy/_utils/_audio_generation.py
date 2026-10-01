# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_audio_generation.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Server-owned asynchronous Audio generation over provider-artifact executors.

The browser requests a workflow, never a renderer/provider/upstream.  This
module deliberately reuses ``ProviderArtifactOutputRegistry`` as the binary
execution primitive while owning a separate job/artifact lifecycle suitable
for conversational and AI Learn clients.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ._provider_artifact import (
    PROVIDER_ARTIFACT_MAX_PROMPT_CHARS,
    ProviderArtifactError,
    ProviderArtifactRequest,
)

AUDIO_GENERATION_REQUEST_CONTRACT = "assistant.audio-generation-request.v1"
GENERATION_JOB_CONTRACT = "scikitplot-generation-job.v2"
AUDIO_GENERATION_MAX_REQUEST_BYTES = 64 * 1024
AUDIO_GENERATION_CAPABILITY_VERSION = 1
AUDIO_GENERATION_JOB_TTL_SECONDS = 60 * 60
AUDIO_GENERATION_ARTIFACT_TTL_SECONDS = 60 * 60
AUDIO_GENERATION_MAX_JOBS = 256
AUDIO_GENERATION_MAX_TOTAL_ARTIFACT_BYTES = 64 * 1024 * 1024


class AudioGenerationError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        retryable: bool = False,
    ) -> None:
        self.code = str(code or "AUDIO_GENERATION_ERROR")
        self.retryable = bool(retryable)
        super().__init__(message or self.code)


@dataclass(frozen=True)
class AudioGenerationRequest:
    text: str
    selected_model: str
    mode: str = "narration"

    def canonical_sha256(self) -> str:
        raw = json.dumps(
            {
                "contract": AUDIO_GENERATION_REQUEST_CONTRACT,
                "mode": self.mode,
                "selected_model": self.selected_model,
                "text": self.text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass
class AudioArtifact:
    artifact_id: str
    capability: str
    mime_type: str
    data: bytes
    sha256: str
    created_at: float
    expires_at: float


@dataclass
class AudioGenerationJob:
    generation_id: str
    capability: str
    request_sha256: str
    selected_model: str
    state: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    strategy: str = "delegated"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    artifact_id: str = ""
    error_code: str = ""
    error_message: str = ""
    retryable: bool = False

    def public(self, artifact: AudioArtifact | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "contract": GENERATION_JOB_CONTRACT,
            "generation_id": self.generation_id,
            "generation_capability": self.capability,
            "workflow": "assistant_audio_narration",
            "execution": {
                "state": self.state,
                "stage": self.stage,
                "progress": float(self.progress),
            },
            "plan": {"strategy": self.strategy},
            "selected_model": self.selected_model,
            "created_at": int(self.created_at),
            "updated_at": int(self.updated_at),
            "artifacts": [],
        }
        if artifact is not None and self.state == "ready":
            out["artifacts"] = [
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_capability": artifact.capability,
                    "modality": "audio",
                    "mime_type": artifact.mime_type,
                    "size_bytes": len(artifact.data),
                    "sha256": artifact.sha256,
                }
            ]
        if self.error_code:
            out["error"] = {
                "type": "generation_error",
                "code": self.error_code,
                "layer": "execution",
                "message": self.error_message or "Audio generation failed.",
                "retryable": bool(self.retryable),
            }
        return out


def parse_audio_generation_request(body: bytes) -> AudioGenerationRequest:
    if len(body) > AUDIO_GENERATION_MAX_REQUEST_BYTES:
        raise AudioGenerationError("REQUEST_TOO_LARGE")
    try:
        raw = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AudioGenerationError("REQUEST_INVALID") from exc
    if not isinstance(raw, dict):
        raise AudioGenerationError("REQUEST_INVALID")
    allowed = {"contract", "text", "selected_model", "mode"}
    if set(raw) - allowed or raw.get("contract") != AUDIO_GENERATION_REQUEST_CONTRACT:
        raise AudioGenerationError("REQUEST_INVALID")
    text = raw.get("text")
    selected_model = raw.get("selected_model")
    mode = raw.get("mode", "narration")
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text.strip()) > PROVIDER_ARTIFACT_MAX_PROMPT_CHARS
        or "\x00" in text
    ):
        raise AudioGenerationError("REQUEST_INVALID")
    if (
        not isinstance(selected_model, str)
        or not selected_model.strip()
        or len(selected_model.strip()) > 256  # ruff: ignore[magic-value-comparison]
    ):
        raise AudioGenerationError("MODEL_UNKNOWN")
    if mode != "narration":
        raise AudioGenerationError("REQUEST_INVALID")
    return AudioGenerationRequest(
        text=text.strip(), selected_model=selected_model.strip(), mode=mode
    )


class AudioGenerationService:
    """Bounded process-local job/artifact authority for Audio v1."""

    def __init__(
        self,
        *,
        registry: Any,
        model_allowed: Callable[[str], bool],
        max_jobs: int = AUDIO_GENERATION_MAX_JOBS,
        max_total_artifact_bytes: int = AUDIO_GENERATION_MAX_TOTAL_ARTIFACT_BYTES,
        job_ttl_seconds: int = AUDIO_GENERATION_JOB_TTL_SECONDS,
        artifact_ttl_seconds: int = AUDIO_GENERATION_ARTIFACT_TTL_SECONDS,
    ) -> None:
        self._registry = registry
        self._model_allowed = model_allowed
        self._max_jobs = max(8, int(max_jobs))
        self._max_total_artifact_bytes = max(1024 * 1024, int(max_total_artifact_bytes))
        self._job_ttl = max(60, int(job_ttl_seconds))
        self._artifact_ttl = max(60, int(artifact_ttl_seconds))
        self._jobs: dict[str, AudioGenerationJob] = {}
        self._artifacts: dict[str, AudioArtifact] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": "memory",
            "shared": False,
            "durable": False,
            "max_jobs": self._max_jobs,
            "job_ttl_seconds": self._job_ttl,
            "artifact_ttl_seconds": self._artifact_ttl,
            "max_total_artifact_bytes": self._max_total_artifact_bytes,
        }

    def _choose_spec(self) -> Any:
        rows = [row for row in self._registry.public_specs() if row.kind == "audio"]
        live = sorted(
            (row for row in rows if not row.diagnostic),
            key=lambda row: row.id,
        )
        diagnostic = sorted(
            (row for row in rows if row.diagnostic),
            key=lambda row: row.id,
        )
        if live:
            return live[0]
        if diagnostic:
            return diagnostic[0]
        return None

    async def _cleanup_locked(self) -> None:
        now = time.time()
        for artifact_id, artifact in list(self._artifacts.items()):
            if artifact.expires_at <= now:
                self._artifacts.pop(artifact_id, None)
        for generation_id, job in list(self._jobs.items()):
            if (
                job.state in {"ready", "failed", "cancelled"}
                and job.updated_at + self._job_ttl <= now
            ):
                self._jobs.pop(generation_id, None)
                self._tasks.pop(generation_id, None)
                for key, (_digest, gid) in list(self._idempotency.items()):
                    if gid == generation_id:
                        self._idempotency.pop(key, None)

    async def submit(
        self,
        request: AudioGenerationRequest,
        idempotency_key: str,
    ) -> AudioGenerationJob:
        key = str(idempotency_key or "").strip()
        if (
            not key  # lint
            or len(key) > 128  # ruff: ignore[magic-value-comparison]
            or any(
                (
                    ord(ch) < 33  # ruff: ignore[magic-value-comparison]
                    or ord(ch) > 126  # ruff: ignore[magic-value-comparison]
                )
                for ch in key
            )
        ):
            raise AudioGenerationError("IDEMPOTENCY_INVALID")
        digest = request.canonical_sha256()
        async with self._lock:
            await self._cleanup_locked()
            previous = self._idempotency.get(key)
            if previous:
                old_digest, generation_id = previous
                if old_digest != digest:
                    raise AudioGenerationError("IDEMPOTENCY_CONFLICT")
                job = self._jobs.get(generation_id)
                if job is not None:
                    return job
            if not self._model_allowed(request.selected_model):
                raise AudioGenerationError("MODEL_UNKNOWN")
            spec = self._choose_spec()
            if spec is None:
                raise AudioGenerationError("NO_COMPATIBLE_DEPLOYMENT")
            if len(self._jobs) >= self._max_jobs:
                raise AudioGenerationError("QUEUE_FULL", retryable=True)
            generation_id = "gen_" + secrets.token_hex(16)
            capability = secrets.token_urlsafe(32)
            strategy = (
                "stub"
                if bool(spec.diagnostic)
                else ("native" if spec.model == request.selected_model else "delegated")
            )
            job = AudioGenerationJob(
                generation_id=generation_id,
                capability=capability,
                request_sha256=digest,
                selected_model=request.selected_model,
                strategy=strategy,
            )
            self._jobs[generation_id] = job
            self._idempotency[key] = (digest, generation_id)
            task = asyncio.create_task(self._run(job, request, spec))
            self._tasks[generation_id] = task
            return job

    async def _run(
        self,
        job: AudioGenerationJob,
        request: AudioGenerationRequest,
        spec: Any,
    ) -> None:
        try:
            async with self._lock:
                if job.state == "cancelled":
                    return
                job.state = "running"
                job.stage = "synthesizing"
                job.progress = 0.25
                job.updated_at = time.time()
            mime_type = "audio/mpeg" if "audio/mpeg" in spec.mime_types else "audio/wav"
            low = ProviderArtifactRequest(
                generator_id=spec.id,
                kind="audio",
                prompt=request.text,
                mime_type=mime_type,
                options={},
            )
            generated = await self._registry.generate(low)
            try:
                generated.file.seek(0)
                data = generated.file.read()
            finally:
                generated.close()
            if not data:
                raise AudioGenerationError("OUTPUT_INVALID")
            async with self._lock:
                if job.state == "cancelled":
                    return
                await self._cleanup_locked()
                total = sum(len(row.data) for row in self._artifacts.values())
                if total + len(data) > self._max_total_artifact_bytes:
                    raise AudioGenerationError("QUEUE_FULL", retryable=True)
                artifact_id = "art_" + secrets.token_hex(16)
                artifact = AudioArtifact(
                    artifact_id=artifact_id,
                    capability=secrets.token_urlsafe(32),
                    mime_type=mime_type,
                    data=data,
                    sha256=hashlib.sha256(data).hexdigest(),
                    created_at=time.time(),
                    expires_at=time.time() + self._artifact_ttl,
                )
                self._artifacts[artifact_id] = artifact
                job.artifact_id = artifact_id
                job.state = "ready"
                job.stage = "ready"
                job.progress = 1.0
                job.updated_at = time.time()
        except asyncio.CancelledError:
            async with self._lock:
                if job.state not in {"ready", "failed"}:
                    job.state = "cancelled"
                    job.stage = "cancelled"
                    job.updated_at = time.time()
        except (ProviderArtifactError, AudioGenerationError) as exc:
            async with self._lock:
                if job.state == "cancelled":
                    return
                code = getattr(exc, "code", "PROVIDER_UNAVAILABLE")
                public = {
                    "PROVIDER_ARTIFACT_UPSTREAM_UNAVAILABLE": (
                        "The speech renderer is temporarily unavailable."
                    ),
                    "PROVIDER_ARTIFACT_UPSTREAM_REJECTED": (
                        "The speech renderer rejected the request."
                    ),
                    "PROVIDER_ARTIFACT_OUTPUT_TOO_LARGE": (
                        "The generated audio exceeded the configured limit."
                    ),
                    "QUEUE_FULL": (
                        "The audio generation service is temporarily at capacity."
                    ),
                }.get(
                    code,
                    "Audio generation could not complete.",
                )
                job.state = "failed"
                job.stage = "failed"
                job.progress = max(job.progress, 0.25)
                job.error_code = (
                    "PROVIDER_UNAVAILABLE"
                    if code.startswith("PROVIDER_ARTIFACT_UPSTREAM")
                    else code
                )
                job.error_message = public
                job.retryable = code in {
                    "PROVIDER_ARTIFACT_UPSTREAM_UNAVAILABLE",
                    "QUEUE_FULL",
                }
                job.updated_at = time.time()

    async def get(
        self,
        generation_id: str,
        capability: str,
    ) -> tuple[AudioGenerationJob, AudioArtifact | None]:
        async with self._lock:
            await self._cleanup_locked()
            job = self._jobs.get(str(generation_id or ""))
            if job is None or not secrets.compare_digest(
                job.capability,
                str(capability or ""),
            ):
                raise AudioGenerationError("JOB_NOT_ACCESSIBLE")
            artifact = self._artifacts.get(job.artifact_id) if job.artifact_id else None
            return job, artifact

    async def cancel(
        self,
        generation_id: str,
        capability: str,
    ) -> AudioGenerationJob:
        async with self._lock:
            await self._cleanup_locked()
            job = self._jobs.get(str(generation_id or ""))
            if job is None or not secrets.compare_digest(
                job.capability,
                str(capability or ""),
            ):
                raise AudioGenerationError("JOB_NOT_ACCESSIBLE")
            if job.state in {"ready", "failed", "cancelled"}:
                raise AudioGenerationError("JOB_NOT_CANCELLABLE")
            job.state = "cancelled"
            job.stage = "cancelled"
            job.updated_at = time.time()
            task = self._tasks.get(job.generation_id)
            if task is not None and not task.done():
                task.cancel()
            return job

    async def artifact(self, artifact_id: str, capability: str) -> AudioArtifact:
        async with self._lock:
            await self._cleanup_locked()
            artifact = self._artifacts.get(str(artifact_id or ""))
            if artifact is None or not secrets.compare_digest(
                artifact.capability, str(capability or "")
            ):
                raise AudioGenerationError("ARTIFACT_NOT_ACCESSIBLE")
            return artifact
