"""Focused tests for the shared optional-backend orchestration layer."""

from __future__ import annotations

import json
import logging

import pytest

from ..._backends import (
    BackendCandidate,
    BackendStatus,
    run_backend_chain,
)

logger = logging.getLogger(__name__)


def test_runtime_failure_falls_through_and_records_degraded() -> None:
    def broken() -> str:
        raise TypeError("backend skew")

    outcome = run_backend_chain(
        (
            BackendCandidate("primary", broken),
            BackendCandidate("fallback", lambda: "ok"),
        ),
        default="",
        logger=logger,
        component="Reader",
        operation="extract",
        subject="sample.bin",
    )

    assert outcome.status is BackendStatus.DEGRADED
    assert outcome.value == "ok"
    assert outcome.backend == "fallback"
    assert outcome.attempted == ("primary", "fallback")
    assert outcome.errors[0].exception_type == "TypeError"
    assert outcome.errors[0].details["backend"] == "primary"
    json.dumps(outcome.to_dict())


def test_normal_miss_tries_next_without_error_record() -> None:
    outcome = run_backend_chain(
        (
            BackendCandidate("first", lambda: None, accept=lambda value: value is not None),
            BackendCandidate("second", lambda: 3.5),
        ),
        default=None,
        logger=logger,
        component="Reader",
        operation="probe",
    )

    assert outcome.status is BackendStatus.SUCCESS
    assert outcome.value == 3.5
    assert outcome.missed == ("first",)
    assert outcome.errors == ()


def test_successful_empty_result_is_not_fallback_failure() -> None:
    called = {"fallback": False}

    def fallback() -> list[str]:
        called["fallback"] = True
        return ["unexpected"]

    outcome = run_backend_chain(
        (
            BackendCandidate("primary", lambda: []),
            BackendCandidate("fallback", fallback),
        ),
        default=[],
        logger=logger,
        component="Reader",
        operation="transcribe",
        is_empty=lambda value: not value,
    )

    assert outcome.status is BackendStatus.EMPTY
    assert outcome.value == []
    assert called["fallback"] is False


def test_expected_optional_primary_can_fallback_without_degraded_status() -> None:
    def absent() -> str:
        raise ImportError("optional parser absent")

    outcome = run_backend_chain(
        (
            BackendCandidate(
                "optional",
                absent,
                fallback_exceptions=(ImportError,),
                degrades_on_failure=False,
            ),
            BackendCandidate("stdlib", lambda: "parsed"),
        ),
        default="",
        logger=logger,
        component="XMLReader",
        operation="parse",
    )

    assert outcome.status is BackendStatus.SUCCESS
    assert outcome.value == "parsed"
    assert outcome.errors[0].details["degrades"] is False
    assert outcome.errors[0].details["is_import_error"] is True


def test_non_fallback_exception_propagates() -> None:
    def malformed() -> str:
        raise ValueError("bad document")

    with pytest.raises(ValueError, match="bad document"):
        run_backend_chain(
            (
                BackendCandidate(
                    "parser",
                    malformed,
                    fallback_exceptions=(ImportError,),
                ),
            ),
            default="",
            logger=logger,
            component="Reader",
            operation="parse",
        )


def test_memory_error_always_propagates() -> None:
    def oom() -> str:
        raise MemoryError("out of memory")

    with pytest.raises(MemoryError):
        run_backend_chain(
            (BackendCandidate("backend", oom),),
            default="",
            logger=logger,
            component="Reader",
            operation="extract",
        )
