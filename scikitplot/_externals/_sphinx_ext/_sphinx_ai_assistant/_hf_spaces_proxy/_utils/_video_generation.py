# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_video_generation.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Video-generation contracts and a deterministic test backend.

The production proxy owns transport/authentication while this module owns the
bounded public contract, validation, and an in-memory stub lifecycle used only
when ``VIDEO_GENERATION_MODE=stub``.  The stub never publishes media and never
claims to have uploaded to YouTube; it exists so the browser lifecycle can be
exercised end to end before a real renderer/publisher is configured.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import re
import time
import uuid
from collections import OrderedDict
from typing import Any
from urllib.parse import urlsplit, urlunsplit

VIDEO_GENERATION_REQUEST_CONTRACT = "learn.video-generation-request.v1"
VIDEO_GENERATION_JOB_CONTRACT = "learn.video-generation-job.v1"
VIDEO_GENERATION_CAPABILITY_VERSION = 2
VIDEO_GENERATION_MAX_BODY_BYTES = 128 * 1024
VIDEO_GENERATION_MAX_JOBS = 256
VIDEO_GENERATION_RESULT_PROVIDERS = ("youtube", "hls", "file", "external")
VIDEO_GENERATION_ACTIONS = ("cancel", "retry", "archive", "restore")
VIDEO_GENERATION_STATUSES = (
    "draft",
    "submitted",
    "queued",
    "running",
    "ready",
    "failed",
    "cancelled",
    "archived",
)

_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_MODE_SET = frozenset({"topic", "source", "url", "prompt"})
_STYLE_SET = frozenset({"explainer", "lecture", "storyboard", "comparison"})
_LENGTH_SET = frozenset({"short", "standard", "deep"})
_ASPECT_SET = frozenset({"16:9", "1:1", "9:16"})


class VideoGenerationError(ValueError):
    """
    Bounded contract/configuration error with a stable public code.

    ``message`` is the authored, client-safe sentence passed at the raise
    site. Responses expose ``code`` and ``message`` only - never ``str(exc)``,
    whose content is whatever the exception machinery makes of the instance
    (arguments, notes, a chained cause) rather than what the author approved.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _bounded_text(value: Any, limit: int, *, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise VideoGenerationError(
            "VIDEO_FIELD_REQUIRED",
            "A required video generation field is empty.",
        )
    if len(text) > limit:
        raise VideoGenerationError(
            "VIDEO_FIELD_TOO_LONG",
            "A video generation field exceeds its size limit.",
        )
    return text


def _public_https_url(
    value: Any,
    *,
    allow_empty: bool = False,
) -> str:
    raw = _bounded_text(value, 2048)
    if not raw and allow_empty:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise VideoGenerationError(
            "VIDEO_URL_INVALID",
            "The video generation URL is invalid.",
        ) from exc
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise VideoGenerationError(
            "VIDEO_URL_INVALID",
            "Video generation URLs must use public HTTPS.",
        )
    if parsed.username or parsed.password or parsed.fragment:
        raise VideoGenerationError(
            "VIDEO_URL_INVALID",
            "Video generation URLs cannot contain credentials or fragments.",
        )
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise VideoGenerationError(
            "VIDEO_URL_PRIVATE",
            "Local/private video source URLs are not allowed.",
        )
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not literal_ip.is_global:
        raise VideoGenerationError(
            "VIDEO_URL_PRIVATE",
            "Local/private video source URLs are not allowed.",
        )
    # DNS rebinding/private-address resolution remains the upstream fetcher's
    # responsibility; this boundary rejects obvious local literals early.
    return urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))


def normalize_video_upstream_url(value: str) -> str:
    """Validate an operator-configured exact upstream collection endpoint."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise VideoGenerationError(
            "VIDEO_UPSTREAM_INVALID",
            "VIDEO_GENERATION_UPSTREAM_URL must be an absolute HTTPS URL.",
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise VideoGenerationError(
            "VIDEO_UPSTREAM_INVALID",
            "VIDEO_GENERATION_UPSTREAM_URL cannot contain credentials, query parameters, or fragments.",
        )
    path = (parsed.path or "").rstrip("/")
    if not path:
        raise VideoGenerationError(
            "VIDEO_UPSTREAM_INVALID",
            "VIDEO_GENERATION_UPSTREAM_URL must name the video-generation collection endpoint.",
        )
    return urlunsplit(("https", parsed.netloc, path, "", ""))


def validate_generation_id(value: Any) -> str:
    generation_id = _bounded_text(value, 160, required=True)
    if not _ID_RE.fullmatch(generation_id):
        raise VideoGenerationError(
            "VIDEO_JOB_ID_INVALID",
            "Video generation id contains unsupported characters.",
        )
    return generation_id


def validate_idempotency_key(value: Any) -> str:
    key = _bounded_text(value, 160, required=True)
    if not _ID_RE.fullmatch(key):
        raise VideoGenerationError(
            "VIDEO_IDEMPOTENCY_INVALID",
            "Idempotency-Key contains unsupported characters.",
        )
    return key


def _source_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VideoGenerationError(
            "VIDEO_SOURCE_INVALID",
            "The Source snapshot is invalid.",
        )
    out = {
        "id": _bounded_text(value.get("id"), 160, required=True),
        "title": _bounded_text(value.get("title"), 240, required=True),
        "summary": _bounded_text(value.get("summary"), 2000),
        "publisher": _bounded_text(value.get("publisher"), 160),
        "format": _bounded_text(value.get("format"), 120),
        "url": _public_https_url(value.get("url"), allow_empty=True),
    }
    return out  # ruff: ignore[unnecessary-assign]


def _topic_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VideoGenerationError(
            "VIDEO_TOPIC_INVALID",
            "The Topic snapshot is invalid.",
        )
    domains = value.get("domains") if isinstance(value.get("domains"), list) else []
    sources = value.get("sources") if isinstance(value.get("sources"), list) else []
    return {
        "id": _bounded_text(value.get("id"), 160, required=True),
        "title": _bounded_text(value.get("title"), 240, required=True),
        "summary": _bounded_text(value.get("summary"), 2000),
        "domains": [
            _bounded_text(item, 80) for item in domains[:24] if str(item or "").strip()
        ],
        "sources": [_source_snapshot(item) for item in sources[:24]],
    }


def _model_snapshot(value: Any) -> dict[str, Any] | None:
    if value in (None, {}):
        return None
    if not isinstance(value, dict):
        raise VideoGenerationError(
            "VIDEO_MODEL_INVALID",
            "The requested model selection is invalid.",
        )
    effort = value.get("effort") if isinstance(value.get("effort"), dict) else {}
    return {
        "source": _bounded_text(value.get("source"), 80) or "ai-assistant",
        "id": _bounded_text(value.get("id"), 160, required=True),
        "label": _bounded_text(value.get("label"), 240),
        "provider": _bounded_text(value.get("provider"), 80),
        "model": _bounded_text(value.get("model"), 240),
        "effort": {
            "id": _bounded_text(effort.get("id"), 80) or "default",
            "label": _bounded_text(effort.get("label"), 120) or "Default",
            "supported": bool(effort.get("supported")),
        },
    }


def parse_video_generation_request(  # ruff: ignore[too-many-branches]
    raw: bytes,
) -> dict[str, Any]:
    if not raw:
        raise VideoGenerationError(
            "VIDEO_REQUEST_EMPTY",
            "Video generation request body is empty.",
        )
    if len(raw) > VIDEO_GENERATION_MAX_BODY_BYTES:
        raise VideoGenerationError(
            "VIDEO_REQUEST_TOO_LARGE",
            "Video generation request body is too large.",
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VideoGenerationError(
            "VIDEO_REQUEST_JSON",
            "Video generation request must be valid UTF-8 JSON.",
        ) from exc
    if not isinstance(value, dict):
        raise VideoGenerationError(
            "VIDEO_REQUEST_INVALID",
            "Video generation request must be a JSON object.",
        )
    if value.get("contract") != VIDEO_GENERATION_REQUEST_CONTRACT:
        raise VideoGenerationError(
            "VIDEO_CONTRACT_UNSUPPORTED",
            "Unsupported video generation request contract.",
        )

    client_request_id = _bounded_text(
        value.get("client_request_id"), 160, required=True
    )
    if not _ID_RE.fullmatch(client_request_id):
        raise VideoGenerationError(
            "VIDEO_REQUEST_ID_INVALID",
            "Video generation request id is invalid.",
        )

    raw_input = value.get("input")
    if not isinstance(raw_input, dict):
        raise VideoGenerationError(
            "VIDEO_INPUT_INVALID",
            "Video generation input is invalid.",
        )
    mode = _bounded_text(raw_input.get("mode"), 32, required=True).lower()
    if mode not in _MODE_SET:
        raise VideoGenerationError(
            "VIDEO_INPUT_MODE",
            "Unsupported video generation input mode.",
        )
    input_doc: dict[str, Any] = {"mode": mode}
    if mode == "topic":
        input_doc["topic"] = _topic_snapshot(raw_input.get("topic"))
    elif mode == "source":
        input_doc["source"] = _source_snapshot(raw_input.get("source"))
    elif mode == "url":
        input_doc["url"] = _public_https_url(raw_input.get("url"))
    else:
        input_doc["prompt"] = _bounded_text(
            raw_input.get("prompt"), 6000, required=True
        )

    presentation = (
        value.get("presentation") if isinstance(value.get("presentation"), dict) else {}
    )
    style = _bounded_text(presentation.get("style"), 40) or "explainer"
    length = _bounded_text(presentation.get("length"), 40) or "standard"
    aspect = _bounded_text(presentation.get("aspect_ratio"), 20) or "16:9"
    if (
        style not in _STYLE_SET
        or length not in _LENGTH_SET
        or aspect not in _ASPECT_SET
    ):
        raise VideoGenerationError(
            "VIDEO_PRESENTATION_INVALID",
            "Unsupported video presentation option.",
        )
    presentation_doc = {
        "style": style,
        "length": length,
        "language": _bounded_text(presentation.get("language"), 32) or "en",
        "voice": _bounded_text(presentation.get("voice"), 160) or "default",
        "aspect_ratio": aspect,
        "captions": bool(presentation.get("captions", True)),
        "branding": bool(presentation.get("branding", True)),
    }

    provenance = (
        value.get("provenance") if isinstance(value.get("provenance"), dict) else {}
    )
    source_page = _bounded_text(provenance.get("source_page"), 2048)
    if source_page:
        source_page = _public_https_url(source_page)
    provenance_doc = {
        "site_id": _bounded_text(provenance.get("site_id"), 160),
        "catalog_revision": _bounded_text(provenance.get("catalog_revision"), 160),
        "source_page": source_page,
    }

    out: dict[str, Any] = {
        "contract": VIDEO_GENERATION_REQUEST_CONTRACT,
        "client_request_id": client_request_id,
        "input": input_doc,
        "instructions": _bounded_text(value.get("instructions"), 4000),
        "presentation": presentation_doc,
        "model_selection": _model_snapshot(value.get("model_selection")),
        "provenance": provenance_doc,
    }
    derived = _bounded_text(value.get("derived_from_video_id"), 160)
    if derived:
        if not _ID_RE.fullmatch(derived):
            raise VideoGenerationError(
                "VIDEO_DERIVED_ID_INVALID",
                "Derived video id is invalid.",
            )
        out["derived_from_video_id"] = derived
    return out


def normalize_video_job(value: Any) -> dict[str, Any]:
    """Validate and privacy-minimize one backend job receipt."""
    if not isinstance(value, dict):
        raise VideoGenerationError(
            "VIDEO_JOB_INVALID",
            "Video generation backend returned an invalid job.",
        )
    generation_id = _bounded_text(
        value.get("generation_id") or value.get("id"), 160, required=True
    )
    if not _ID_RE.fullmatch(generation_id):
        raise VideoGenerationError(
            "VIDEO_JOB_INVALID",
            "Video generation backend returned an invalid job id.",
        )
    status = _bounded_text(value.get("status"), 40, required=True)
    if status not in VIDEO_GENERATION_STATUSES:
        raise VideoGenerationError(
            "VIDEO_JOB_INVALID",
            "Video generation backend returned an unsupported status.",
        )
    try:
        progress_raw = value.get("progress")
        progress = (
            None if progress_raw is None else max(0.0, min(1.0, float(progress_raw)))
        )
    except (TypeError, ValueError):
        progress = None
    out: dict[str, Any] = {
        "contract": VIDEO_GENERATION_JOB_CONTRACT,
        "generation_id": generation_id,
        "status": status,
        "stage": _bounded_text(value.get("stage"), 80) or status,
        "progress": progress,
        "title": _bounded_text(value.get("title"), 240) or "Generated video",
        "created_at": _bounded_text(value.get("created_at"), 80),
        "updated_at": _bounded_text(value.get("updated_at"), 80),
    }
    error = value.get("error")
    if isinstance(error, dict):
        out["error"] = {
            "code": _bounded_text(error.get("code"), 120),
            "message": (
                _bounded_text(error.get("message"), 600)
                or "Generation could not complete."
            ),
            "retryable": error.get("retryable") is not False,
        }
    result = value.get("result")
    if isinstance(result, dict):
        provider = _bounded_text(result.get("provider"), 40)
        if provider and provider not in VIDEO_GENERATION_RESULT_PROVIDERS:
            raise VideoGenerationError(
                "VIDEO_JOB_INVALID",
                "Video generation backend returned an unsupported result provider.",
            )
        duration = result.get("duration_seconds")
        try:
            duration_number = max(0, min(24 * 3600, int(float(duration or 0))))
        except (TypeError, ValueError):
            duration_number = 0
        out["result"] = {
            "provider": provider,
            "provider_id": _bounded_text(result.get("provider_id"), 240),
            "url": _public_https_url(result.get("url"), allow_empty=True),
            "thumbnail_url": _public_https_url(
                result.get("thumbnail_url") or result.get("poster"), allow_empty=True
            ),
            "duration_seconds": duration_number,
            "test_mode": bool(result.get("test_mode")),
        }
    execution = value.get("execution")
    if isinstance(execution, dict):
        # Deliberately bounded provenance only; never pass provider credentials,
        # raw prompts, internal traces, or arbitrary backend objects.
        out["execution"] = {
            "pipeline_version": _bounded_text(execution.get("pipeline_version"), 160),
            "planner_model": _bounded_text(execution.get("planner_model"), 240),
            "renderer": _bounded_text(execution.get("renderer"), 160),
            "publish_provider": _bounded_text(execution.get("publish_provider"), 80),
            "fallback_used": bool(execution.get("fallback_used")),
        }
    return out


def request_title(request_doc: dict[str, Any]) -> str:
    input_doc = request_doc.get("input") or {}
    if isinstance(input_doc.get("topic"), dict):
        return str(input_doc["topic"].get("title") or "Generated video")[:200]
    if isinstance(input_doc.get("source"), dict):
        return str(input_doc["source"].get("title") or "Generated video")[:200]
    if input_doc.get("url"):
        return str(input_doc["url"])[:200]
    return str(input_doc.get("prompt") or "Generated video")[:200]


def _iso(now: float | None = None) -> str:
    # UTC ISO-8601 without importing datetime keeps this module stdlib-light.
    stamp = time.gmtime(time.time() if now is None else now)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", stamp)


class StubVideoGenerationStore:
    """Bounded deterministic lifecycle for browser/proxy integration tests."""

    _RUN_STAGES = (
        ("queued", 0.05),
        ("preparing", 0.14),
        ("grounding", 0.28),
        ("outline", 0.42),
        ("script", 0.58),
        ("visuals", 0.74),
        ("rendering", 0.90),
        ("verifying", 0.97),
    )

    def __init__(
        self, *, max_jobs: int = VIDEO_GENERATION_MAX_JOBS, ready_seconds: float = 3.0
    ) -> None:
        self.max_jobs = max(8, min(4096, int(max_jobs)))
        self.ready_seconds = max(0.0, min(300.0, float(ready_seconds)))
        self._jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._idempotency: dict[str, str] = {}

    def _trim(self) -> None:
        while len(self._jobs) > self.max_jobs:
            generation_id, _job = self._jobs.popitem(last=False)
            stale = [
                key
                for key, value in self._idempotency.items()
                if value == generation_id
            ]
            for key in stale:
                self._idempotency.pop(key, None)

    def create(
        self, request_doc: dict[str, Any], idempotency_key: str
    ) -> dict[str, Any]:
        existing = self._idempotency.get(idempotency_key)
        if existing and existing in self._jobs:
            return self.get(existing)
        now = time.time()
        generation_id = "vg_stub_" + uuid.uuid4().hex[:20]
        job = {
            "contract": VIDEO_GENERATION_JOB_CONTRACT,
            "generation_id": generation_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.05,
            "title": request_title(request_doc),
            "created_at": _iso(now),
            "updated_at": _iso(now),
            "_started_epoch": now,
            "_request": copy.deepcopy(request_doc),
            "_archived_from": "",
        }
        self._jobs[generation_id] = job
        self._idempotency[idempotency_key] = generation_id
        self._trim()
        return self.get(generation_id)

    def _materialize(self, job: dict[str, Any]) -> dict[str, Any]:
        if job.get("status") in {"cancelled", "failed", "archived"}:
            return job
        elapsed = max(
            0.0, time.time() - float(job.get("_started_epoch") or time.time())
        )
        if self.ready_seconds <= 0 or elapsed >= self.ready_seconds:
            job.update(
                status="ready",
                stage="ready",
                progress=1.0,
                updated_at=_iso(),
                result={
                    "provider": "external",
                    "provider_id": "stub:" + str(job["generation_id"]),
                    "url": "",
                    "thumbnail_url": "",
                    "duration_seconds": 0,
                    "test_mode": True,
                },
            )
            return job
        fraction = elapsed / self.ready_seconds
        index = min(len(self._RUN_STAGES) - 1, int(fraction * len(self._RUN_STAGES)))
        stage, floor = self._RUN_STAGES[index]
        next_floor = self._RUN_STAGES[min(index + 1, len(self._RUN_STAGES) - 1)][1]
        local = (fraction * len(self._RUN_STAGES)) - index
        progress = min(
            0.99, floor + max(0.0, next_floor - floor) * max(0.0, min(1.0, local))
        )
        job.update(
            status=("queued" if stage == "queued" else "running"),
            stage=stage,
            progress=progress,
            updated_at=_iso(),
        )
        return job

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in job.items()
            if not key.startswith("_")
        }

    def get(self, generation_id: str) -> dict[str, Any]:
        job = self._jobs.get(generation_id)
        if not job:
            raise VideoGenerationError(
                "VIDEO_JOB_NOT_FOUND",
                "Video generation was not found.",
            )
        self._materialize(job)
        self._jobs.move_to_end(generation_id)
        return self._public(job)

    def list(self, *, include_archived: bool = True) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for generation_id in reversed(list(self._jobs.keys())):
            item = self.get(generation_id)
            if not include_archived and item.get("status") == "archived":
                continue
            out.append(item)
        return out

    def action(self, generation_id: str, action: str) -> dict[str, Any]:
        if action not in VIDEO_GENERATION_ACTIONS:
            raise VideoGenerationError(
                "VIDEO_ACTION_UNSUPPORTED",
                "Unsupported video generation lifecycle action.",
            )
        job = self._jobs.get(generation_id)
        if not job:
            raise VideoGenerationError(
                "VIDEO_JOB_NOT_FOUND",
                "Video generation was not found.",
            )
        self._materialize(job)
        status = str(job.get("status") or "")
        if action == "cancel":
            if status not in {"submitted", "queued", "running"}:
                raise VideoGenerationError(
                    "VIDEO_ACTION_INVALID_STATE",
                    "Only an active generation can be cancelled.",
                )
            job.update(
                status="cancelled",
                stage="cancelled",
                progress=job.get("progress"),
                updated_at=_iso(),
            )
        elif action == "retry":
            if status not in {"failed", "cancelled"}:
                raise VideoGenerationError(
                    "VIDEO_ACTION_INVALID_STATE",
                    "Only a failed or cancelled generation can be retried.",
                )
            job.pop("error", None)
            job.pop("result", None)
            job.update(
                status="queued",
                stage="queued",
                progress=0.05,
                updated_at=_iso(),
                _started_epoch=time.time(),
            )
        elif action == "archive":
            if status in {"submitted", "queued", "running", "archived"}:
                raise VideoGenerationError(
                    "VIDEO_ACTION_INVALID_STATE",
                    "Active or already archived generations cannot be archived.",
                )
            job["_archived_from"] = status
            job.update(status="archived", stage="archived", updated_at=_iso())
        elif action == "restore":
            if status != "archived":
                raise VideoGenerationError(
                    "VIDEO_ACTION_INVALID_STATE",
                    "Only an archived generation can be restored.",
                )
            restored = str(
                job.get("_archived_from")
                or ("ready" if job.get("result") else "cancelled")
            )
            job.update(status=restored, stage=restored, updated_at=_iso())
        return self._public(job)
