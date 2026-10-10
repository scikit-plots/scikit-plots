"""Focused tests for the shared optional-backend orchestration layer."""

from __future__ import annotations

import json
import logging

import pytest

from ..._backends import (
    BackendCandidate,
    BackendPolicy,
    BackendStatus,
    plan_backend_chain,
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


def test_all_misses_report_exhausted() -> None:
    outcome = run_backend_chain(
        (
            BackendCandidate("one", lambda: None, accept=lambda value: value is not None),
            BackendCandidate("two", lambda: None, accept=lambda value: value is not None),
        ),
        default=None,
        logger=logger,
        component="Reader",
        operation="probe",
    )
    assert outcome.status is BackendStatus.EXHAUSTED
    assert not outcome.succeeded


def test_empty_candidate_set_reports_unavailable() -> None:
    outcome = run_backend_chain(
        (),
        default=[],
        logger=logger,
        component="Reader",
        operation="probe",
    )
    assert outcome.status is BackendStatus.UNAVAILABLE
    assert not outcome.succeeded


def test_unavailable_outcome_explains_policy_skip() -> None:
    logger = logging.getLogger("test-backend-skip-details")
    outcome = run_backend_chain(
        [
            BackendCandidate(
                name="remote",
                run=lambda: "never",
                requires_network=True,
            )
        ],
        default="",
        logger=logger,
        component="test",
        operation="probe",
        policy=BackendPolicy.offline(),
    )
    assert outcome.status is BackendStatus.UNAVAILABLE
    assert outcome.skipped == ("remote",)
    assert outcome.skip_details[0].backend == "remote"
    assert "network disabled" in outcome.skip_details[0].reason
    assert outcome.to_dict()["skip_details"][0]["backend"] == "remote"


def test_skip_detail_includes_capability_readiness() -> None:
    from scikitplot.corpus import CapabilityRegistry, CapabilitySpec

    registry = CapabilityRegistry([
        CapabilitySpec(
            "asr:missing-model",
            "asr",
            assets_required=True,
            asset_probe=lambda: False,
        )
    ])
    outcome = run_backend_chain(
        [
            BackendCandidate(
                name="local-model",
                run=lambda: "never",
                capability="asr:missing-model",
            )
        ],
        default="",
        logger=logging.getLogger("test-backend-capability-skip-details"),
        component="test",
        operation="probe",
        policy=BackendPolicy(require_ready=True),
        capability_registry=registry,
    )
    detail = outcome.skip_details[0]
    assert detail.capability == "asr:missing-model"
    assert detail.capability_status == "misconfigured"
    assert detail.ready is False


def test_backend_plan_is_side_effect_free_and_matches_runtime_selection() -> None:
    calls: list[str] = []
    candidates = (
        BackendCandidate("primary", lambda: calls.append("primary") or "a"),
        BackendCandidate("fallback", lambda: calls.append("fallback") or "b"),
    )
    policy = BackendPolicy.first_available().with_order("fallback", "primary")

    plan = plan_backend_chain(candidates, policy=policy)

    assert calls == []
    assert plan.declared == ("primary", "fallback")
    assert plan.ordered == ("fallback", "primary")
    assert plan.eligible == ("fallback", "primary")
    assert plan.selected == ("fallback",)
    json.dumps(plan.to_dict())

    outcome = run_backend_chain(
        candidates,
        default="",
        logger=logger,
        component="Reader",
        operation="probe",
        policy=policy,
    )
    assert calls == ["fallback"]
    assert outcome.backend == plan.selected[0]


def test_backend_plan_offline_allows_unknown_cache_only_when_backend_can_enforce_local() -> None:
    from scikitplot.corpus import CapabilityRegistry, CapabilitySpec

    registry = CapabilityRegistry([
        CapabilitySpec(
            "asr:cached-unknown",
            "asr",
            module=None,
            assets_required=True,
            may_download=True,
        )
    ])
    candidates = (
        BackendCandidate(
            "local-aware",
            lambda: "ok",
            capability="asr:cached-unknown",
            may_download=True,
            offline_capable=True,
        ),
        BackendCandidate(
            "cannot-enforce-offline",
            lambda: "never",
            capability="asr:cached-unknown",
            may_download=True,
            offline_capable=False,
        ),
    )

    plan = plan_backend_chain(
        candidates,
        policy=BackendPolicy.offline(),
        capability_registry=registry,
    )

    assert plan.selected == ("local-aware",)
    assert plan.skipped == ("cannot-enforce-offline",)
    assert "cannot enforce local-only" in plan.skip_details[0].reason


def test_plan_and_outcome_capability_views_mark_selected_and_active() -> None:
    from scikitplot.corpus import CapabilityRegistry, CapabilitySpec

    registry = CapabilityRegistry([
        CapabilitySpec("backend:one", "demo"),
        CapabilitySpec("backend:two", "demo"),
    ])
    candidates = (
        BackendCandidate("one", lambda: "ok", capability="backend:one"),
        BackendCandidate("two", lambda: "fallback", capability="backend:two"),
    )
    plan = plan_backend_chain(candidates, capability_registry=registry)
    plan_view = plan.capability_view(registry=registry)
    assert plan_view["backend:one"]["selected"] is True
    assert plan_view["backend:one"]["active"] is False
    assert plan_view["backend:two"]["selected"] is True

    outcome = run_backend_chain(
        candidates,
        default="",
        logger=logger,
        component="Reader",
        operation="probe",
        capability_registry=registry,
    )
    runtime_view = outcome.capability_view(registry=registry)
    assert runtime_view["backend:one"]["selected"] is True
    assert runtime_view["backend:one"]["active"] is True
    assert runtime_view["backend:two"]["selected"] is True
    assert runtime_view["backend:two"]["active"] is False


def test_audio_reader_preflight_uses_real_configuration_without_running_backend(tmp_path) -> None:
    from scikitplot.corpus import ASRBackend, AudioReader

    calls: list[str] = []
    backend = ASRBackend(
        "local-asr",
        lambda request: calls.append(request.component) or [],
        offline_capable=True,
    )
    policy = BackendPolicy.first_available().with_order(
        "local-asr", include_unlisted=False
    )
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"")
    reader = AudioReader(
        path,
        transcribe=True,
        backend_policy=policy,
        asr_backends=(backend,),
    )

    plan = reader.plan_asr_backends()

    assert calls == []
    assert plan.selected == ("local-asr",)
    assert plan.policy == "first-available"


def test_video_reader_preflight_uses_real_configuration_without_running_backend(tmp_path) -> None:
    from scikitplot.corpus import ASRBackend, VideoReader

    calls: list[str] = []
    backend = ASRBackend(
        "local-video-asr",
        lambda request: calls.append(request.component) or [],
        offline_capable=True,
    )
    policy = BackendPolicy.offline().with_order(
        "local-video-asr", include_unlisted=False
    )
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"")
    reader = VideoReader(
        path,
        transcribe=True,
        backend_policy=policy,
        asr_backends=(backend,),
    )

    plan = reader.plan_asr_backends()

    assert calls == []
    assert plan.selected == ("local-video-asr",)
    assert plan.policy == "offline"
