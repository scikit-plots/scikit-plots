from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

from pathlib import Path

import pytest
from fastapi import HTTPException

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app as proxy_app


ROOT = RUNTIME_ROOT


def test_sphinx_config_exposes_reader_initial_defaults() -> None:
    src = (ROOT / "__init__.py").read_text(encoding="utf-8")
    expected = {
        'ai_assistant_panel_feedback_telemetry_default", False': "panelFeedbackTelemetryDefault",
        'ai_assistant_panel_feedback_review_default", True': "panelFeedbackReviewDefault",
        'ai_assistant_panel_page_integration_default", False': "panelPageIntegrationDefault",
        'ai_assistant_panel_streaming_default", True': "panelStreamingDefault",
        'ai_assistant_panel_remember_conversation", True': "panelRememberConversation",
        'ai_assistant_panel_current_page_context", True': "panelCurrentPageContext",
    }
    for config_fragment, js_key in expected.items():
        assert config_fragment in src
        assert f'"{js_key}"' in src


def test_feedback_lineage_aliases_are_rejected_not_migrated() -> None:
    canonical = {
        "feedbackId": "f1",
        "feedbackChainId": "f1",
        "prevFeedbackId": None,
        "prevFeedbackIds": [],
        "editCount": 0,
    }
    proxy_app._validate_feedback_lineage_fields(canonical, label="x")

    for alias, value in (("sessionId", "f1"), ("prevSessionId", "f0")):
        bad = dict(canonical)
        bad[alias] = value
        with pytest.raises(HTTPException) as exc:
            proxy_app._validate_feedback_lineage_fields(bad, label="x")
        assert exc.value.status_code == 422
        assert "retired feedback lineage aliases" in str(exc.value.detail)


def test_scalar_only_current_lineage_is_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        proxy_app._validate_feedback_lineage_fields(
            {
                "feedbackId": "f2",
                "feedbackChainId": "f1",
                "prevFeedbackId": "f1",
                "editCount": 1,
            },
            label="x",
        )
    assert exc.value.status_code == 422
    assert "prevFeedbackIds is required" in str(exc.value.detail)
