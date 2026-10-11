# scikitplot/corpus/_readers/_whisper.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Whisper ASR backend adapters for AudioReader and VideoReader."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .._backends import (
    BackendCandidate,
    BackendOutcome,
    BackendPlan,
    BackendPolicy,
    backend_policy,
    plan_backend_chain,
    run_backend_chain,
)
from .._capabilities import CapabilityRegistry
from .._schema import ErrorPolicy

__all__ = [
    "WHISPER_MODELS",
    "ASRBackend",
    "ASRRequest",
    "asr_segment_metadata",
    "plan_whisper_backends",
    "transcribe_whisper",
]

WHISPER_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
)

_BACKEND_FASTER = "faster-whisper"
_BACKEND_OPENAI = "openai-whisper"


@dataclasses.dataclass(frozen=True)
class ASRRequest:
    """Stable immutable request passed to user-provided ASR backends.

    ``allow_network`` and ``allow_download`` communicate orchestration policy to
    trusted custom backends; they are requirements, not suggestions, when a
    backend declares itself ``offline_capable``. ``policy_name`` is provenance
    for diagnostics and backend-specific tuning, not an authorization token.
    """

    media_path: Path
    model_size: str
    language: str | None
    component: str
    include_confidence: bool = False
    allow_network: bool = True
    allow_download: bool = True
    policy_name: str = "resilient"


@dataclasses.dataclass(frozen=True)
class ASRBackend:
    """User-defined ASR backend adapter.

    Parameters
    ----------
    name : str
        Stable name used by :class:`BackendPolicy` ordering and diagnostics.
    transcribe : callable
        ``transcribe(request: ASRRequest) -> iterable[Mapping]``.  Every segment
        must provide ``text``, ``timecode_start`` and ``timecode_end``. Optional
        metadata is preserved.
    capability : str or None, optional
        Capability-registry identifier used by readiness-aware policies.
    fallback_exceptions : tuple, optional
        Exceptions safe to convert into fallback diagnostics.
    requires_network : bool, optional
        Declare a network requirement for offline policy enforcement.
    may_download : bool, optional
        Declare that first use may download model/assets.
    offline_capable : bool, optional
        Whether the backend can honour ``ASRRequest.allow_network=False`` and
        ``allow_download=False`` without attempting a download. Custom
        backends must opt in explicitly; the orchestrator trusts this contract.
    degrades_on_failure : bool, optional
        Whether failure followed by another backend success is DEGRADED.

    Notes
    -----
    User callables receive one immutable request object rather than an expanding
    keyword signature, so future request fields can be added compatibly.
    """

    name: str
    transcribe: Callable[[ASRRequest], Iterable[Mapping[str, Any]]]
    capability: str | None = None
    fallback_exceptions: tuple[type[BaseException], ...] = (Exception,)
    requires_network: bool = False
    may_download: bool = False
    offline_capable: bool = False
    degrades_on_failure: bool = True

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ASRBackend.name must be a non-empty string")
        if self.name in {_BACKEND_FASTER, _BACKEND_OPENAI}:
            raise ValueError(
                f"ASRBackend name {self.name!r} is reserved by a built-in backend"
            )
        if not callable(self.transcribe):
            raise TypeError("ASRBackend.transcribe must be callable")
        if not self.fallback_exceptions:
            raise ValueError("ASRBackend.fallback_exceptions must not be empty")


def _normalize_user_segments(
    segments: Iterable[Mapping[str, Any]],
    *,
    backend: str,
) -> list[dict[str, Any]]:
    """Validate custom ASR output without inventing missing timing metadata."""
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(segments):
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"ASR backend {backend!r} segment {index} must be a mapping; "
                f"got {type(raw).__name__}"
            )
        if (
            "text" not in raw
            or "timecode_start" not in raw
            or "timecode_end" not in raw
        ):
            raise ValueError(
                f"ASR backend {backend!r} segment {index} must define text, "
                "timecode_start and timecode_end"
            )
        text = str(raw["text"]).strip()
        if not text:
            continue
        start = float(raw["timecode_start"])
        end = float(raw["timecode_end"])
        if start < 0 or end < start:
            raise ValueError(
                f"ASR backend {backend!r} segment {index} has invalid timing "
                f"start={start!r}, end={end!r}"
            )
        segment = dict(raw)
        segment["text"] = text
        segment.setdefault("raw_text", text)
        segment["timecode_start"] = round(start, 3)
        segment["timecode_end"] = round(end, 3)
        result.append(segment)
    return result


def _faster_segments(
    media_path: Path,
    model_size: str,
    language: str | None,
    *,
    model_kwargs: dict[str, Any] | None,
    transcribe_kwargs: dict[str, Any] | None,
    include_confidence: bool,
) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel  # noqa: PLC0415

    model = WhisperModel(model_size, **(model_kwargs or {}))
    kwargs: dict[str, Any] = dict(transcribe_kwargs or {})
    kwargs["language"] = language
    segments, _info = model.transcribe(str(media_path), **kwargs)

    result: list[dict[str, Any]] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        chunk: dict[str, Any] = {
            "text": text,
            "raw_text": text,
            "timecode_start": round(seg.start, 3),
            "timecode_end": round(seg.end, 3),
        }
        if include_confidence and hasattr(seg, "avg_logprob"):
            chunk["confidence"] = round(math.exp(seg.avg_logprob), 4)
        result.append(chunk)
    return result


def _openai_segments(
    media_path: Path,
    model_size: str,
    language: str | None,
    *,
    include_confidence: bool,
) -> list[dict[str, Any]]:
    import whisper  # noqa: PLC0415

    model = whisper.load_model(model_size)
    wresult = model.transcribe(str(media_path), language=language)

    result: list[dict[str, Any]] = []
    for seg in wresult.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        chunk: dict[str, Any] = {
            "text": text,
            "raw_text": text,
            "timecode_start": round(seg["start"], 3),
            "timecode_end": round(seg["end"], 3),
        }
        if include_confidence and "avg_logprob" in seg:
            chunk["confidence"] = round(math.exp(seg["avg_logprob"]), 4)
        result.append(chunk)
    return result


def _failure_details(outcome: BackendOutcome[list[dict[str, Any]]]) -> str:
    details = [
        f"{record.details.get('backend', 'unknown')}: "
        f"{record.exception_type}: {record.message}"
        for record in outcome.errors
    ]
    if outcome.skipped:
        details.append("skipped=" + ",".join(outcome.skipped))
    if outcome.missed:
        details.append("missed=" + ",".join(outcome.missed))
    return "; ".join(details) or f"status={outcome.status.value}"


_SEGMENT_CORE_KEYS = frozenset(
    {
        "text",
        "raw_text",
        "timecode_start",
        "timecode_end",
        "confidence",
        "asr_backend",
    }
)


def asr_segment_metadata(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Return user/backend metadata without allowing core-field shadowing.

    Custom ASR backends may attach provider-specific information to a segment.
    The reader stores those values under ``asr_metadata`` rather than merging
    arbitrary keys into the raw-chunk namespace, where names such as
    ``source_type`` or ``section_type`` could otherwise override reader-owned
    provenance.
    """
    return {
        key: value for key, value in segment.items() if key not in _SEGMENT_CORE_KEYS
    }


def _resolve_asr_policy(
    policy: BackendPolicy | str | Mapping[str, Any] | None,
    *,
    strict: bool,
) -> BackendPolicy:
    """Resolve ASR policy once for both preflight and execution."""
    resolved = backend_policy(policy)
    if strict and not resolved.raise_on_exhausted:
        resolved = dataclasses.replace(
            resolved,
            name=f"{resolved.name}+strict",
            on_exhausted=ErrorPolicy.RAISE,
        )
    return resolved


def _asr_candidates(
    media_path: Path,
    model_size: str,
    language: str | None,
    *,
    component: str,
    resolved_policy: BackendPolicy,
    faster_model_kwargs: dict[str, Any] | None,
    faster_transcribe_kwargs: dict[str, Any] | None,
    include_confidence: bool,
    custom_backends: Sequence[ASRBackend],
) -> list[BackendCandidate[list[dict[str, Any]]]]:
    """Build the single canonical ASR candidate set used by plan and run."""
    request = ASRRequest(
        media_path=media_path,
        model_size=model_size,
        language=language,
        component=component,
        include_confidence=include_confidence,
        allow_network=resolved_policy.allow_network,
        allow_download=resolved_policy.allow_download,
        policy_name=resolved_policy.name,
    )

    faster_options = dict(faster_model_kwargs or {})
    if not resolved_policy.allow_download:
        # Local-only execution makes UNKNOWN cache readiness safe to attempt:
        # a cache miss becomes a local backend failure, never a download.
        faster_options["local_files_only"] = True

    candidates: list[BackendCandidate[list[dict[str, Any]]]] = [
        BackendCandidate(
            name=_BACKEND_FASTER,
            run=lambda: _faster_segments(
                media_path,
                model_size,
                language,
                model_kwargs=faster_options,
                transcribe_kwargs=faster_transcribe_kwargs,
                include_confidence=include_confidence,
            ),
            capability="asr:faster-whisper",
            may_download=True,
            offline_capable=True,
        ),
        BackendCandidate(
            name=_BACKEND_OPENAI,
            run=lambda: _openai_segments(
                media_path,
                model_size,
                language,
                include_confidence=include_confidence,
            ),
            capability="asr:openai-whisper",
            may_download=True,
        ),
    ]
    known_names = {candidate.name for candidate in candidates}
    for custom in custom_backends:
        if not isinstance(custom, ASRBackend):
            raise TypeError(
                "custom_backends must contain ASRBackend instances; "
                f"got {type(custom).__name__}"
            )
        if custom.name in known_names:
            raise ValueError(f"duplicate ASR backend name {custom.name!r}")
        known_names.add(custom.name)
        candidates.append(
            BackendCandidate(
                name=custom.name,
                run=lambda custom=custom: _normalize_user_segments(
                    custom.transcribe(request), backend=custom.name
                ),
                fallback_exceptions=custom.fallback_exceptions,
                degrades_on_failure=custom.degrades_on_failure,
                capability=custom.capability,
                requires_network=custom.requires_network,
                may_download=custom.may_download,
                offline_capable=custom.offline_capable,
            )
        )
    return candidates


def plan_whisper_backends(
    media_path: Path,
    model_size: str,
    language: str | None,
    *,
    component: str,
    strict: bool = False,
    faster_model_kwargs: dict[str, Any] | None = None,
    faster_transcribe_kwargs: dict[str, Any] | None = None,
    include_confidence: bool = False,
    policy: BackendPolicy | str | Mapping[str, Any] | None = None,
    custom_backends: Sequence[ASRBackend] = (),
    capability_registry: CapabilityRegistry | None = None,
) -> BackendPlan:
    """Plan the exact ASR chain without importing/loading/running a model."""
    resolved_policy = _resolve_asr_policy(policy, strict=strict)
    candidates = _asr_candidates(
        media_path,
        model_size,
        language,
        component=component,
        resolved_policy=resolved_policy,
        faster_model_kwargs=faster_model_kwargs,
        faster_transcribe_kwargs=faster_transcribe_kwargs,
        include_confidence=include_confidence,
        custom_backends=custom_backends,
    )
    return plan_backend_chain(
        candidates,
        policy=resolved_policy,
        capability_registry=capability_registry,
    )


def transcribe_whisper(
    media_path: Path,
    model_size: str,
    language: str | None,
    *,
    component: str,
    logger: Any,
    strict: bool = False,
    faster_model_kwargs: dict[str, Any] | None = None,
    faster_transcribe_kwargs: dict[str, Any] | None = None,
    include_confidence: bool = False,
    report: Callable[[BackendOutcome[list[dict[str, Any]]]], None] | None = None,
    policy: BackendPolicy | str | Mapping[str, Any] | None = None,
    custom_backends: Sequence[ASRBackend] = (),
    capability_registry: CapabilityRegistry | None = None,
) -> list[dict[str, Any]]:
    """Transcribe media with a shared, policy-driven ASR backend chain.

    Preflight and execution are built from the same candidate factory.  Runtime
    failures therefore cannot silently change the order/readiness contract that
    :func:`plan_whisper_backends` reports.
    """
    resolved_policy = _resolve_asr_policy(policy, strict=strict)
    candidates = _asr_candidates(
        media_path,
        model_size,
        language,
        component=component,
        resolved_policy=resolved_policy,
        faster_model_kwargs=faster_model_kwargs,
        faster_transcribe_kwargs=faster_transcribe_kwargs,
        include_confidence=include_confidence,
        custom_backends=custom_backends,
    )

    outcome = run_backend_chain(
        candidates,
        default=[],
        logger=logger,
        component=component,
        operation="Whisper transcription",
        subject=media_path.name,
        error_code="ASR_BACKEND_FAILED",
        stage="transcribe",
        is_empty=lambda value: not value,
        policy=resolved_policy,
        capability_registry=capability_registry,
    )

    if outcome.backend is not None:
        for segment in outcome.value:
            segment["asr_backend"] = outcome.backend

    if report is not None:
        report(outcome)

    if resolved_policy.raise_on_exhausted and not outcome.succeeded:
        if outcome.errors and all(
            bool(record.details.get("is_import_error")) for record in outcome.errors
        ):
            raise ImportError(
                f"{component}: transcribe=True requires either faster-whisper "
                "or openai-whisper.\n"
                "Install one of:\n"
                "  pip install faster-whisper   # recommended (faster, lower VRAM)\n"
                "  pip install openai-whisper   # reference implementation\n"
            )
        raise RuntimeError(
            f"{component}: all Whisper backends failed for "
            f"{media_path.name!r}: {_failure_details(outcome)}"
        )

    return outcome.value
