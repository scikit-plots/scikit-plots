# scikitplot/corpus/_readers/_whisper.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Whisper ASR backend adapters for AudioReader and VideoReader."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from .._backends import (
    BackendCandidate,
    BackendOutcome,
    BackendStatus,
    run_backend_chain,
)

__all__ = [
    "WHISPER_MODELS",
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
    return "; ".join(
        f"{record.details.get('backend', 'unknown')}: "
        f"{record.exception_type}: {record.message}"
        for record in outcome.errors
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
) -> list[dict[str, Any]]:
    """Transcribe media with a shared faster-whisper -> openai-whisper chain.

    The chain catches ordinary backend failures, including lazy decode/inference
    failures, rather than only import errors.  ``MemoryError`` propagates.  A
    successful empty transcription is accepted and does not trigger fallback.

    ``strict=False`` returns an empty list after total backend failure;
    ``strict=True`` raises after the same fallback chain is exhausted.
    Structured backend diagnostics can be captured with ``report``.
    """
    outcome = run_backend_chain(
        (
            BackendCandidate(
                name=_BACKEND_FASTER,
                run=lambda: _faster_segments(
                    media_path,
                    model_size,
                    language,
                    model_kwargs=faster_model_kwargs,
                    transcribe_kwargs=faster_transcribe_kwargs,
                    include_confidence=include_confidence,
                ),
            ),
            BackendCandidate(
                name=_BACKEND_OPENAI,
                run=lambda: _openai_segments(
                    media_path,
                    model_size,
                    language,
                    include_confidence=include_confidence,
                ),
            ),
        ),
        default=[],
        logger=logger,
        component=component,
        operation="Whisper transcription",
        subject=media_path.name,
        error_code="ASR_BACKEND_FAILED",
        stage="transcribe",
        is_empty=lambda value: not value,
    )

    if report is not None:
        report(outcome)

    if strict and outcome.status is BackendStatus.FAILED:
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
