from __future__ import annotations

import json

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _dataset_schema as schema


def test_feedback_telemetry_and_review_are_distinct_saved_row_families() -> None:
    detail = {
        "feedbackId": "fb-event",
        "answerIndex": 0,
        "ratingValue": 1,
        "ratingLabel": "helpful",
        "ratingTitle": "Helpful",
        "ratingMode": "quick",
        "query": "Q",
        "answer": "A",
        "message": "useful",
        "model": {"id": "m", "provider": "p", "model": "owner/m"},
        "page": "https://docs.example/page",
        "consentVersion": "2.0.0",
        "trainingConsentVersion": "1.0.0",
        "ratingScaleMin": -1,
        "ratingScaleMax": 1,
        "ts": 123,
    }
    telemetry = schema.normalize_feedback_record(detail, server_ts_ms=1000)
    review = schema.normalize_feedback_review_record(
        detail, server_ts_ms=1000, receipt_id="receipt"
    )

    assert telemetry["_dedup_key"] == "fb-event:feedback"
    assert telemetry["trainingStatus"] == "telemetry"
    assert telemetry["query"] == telemetry["answer"] == telemetry["page"] == ""
    assert telemetry["message"] == ""
    assert telemetry["model"] is None
    assert telemetry["consentVersion"] is None

    assert review["_dedup_key"] == "receipt:feedback"
    assert review["recordType"] == "qa"
    assert review["action"] == "review"
    assert review["trainingStatus"] == "eligible"
    assert review["query"] == "Q" and review["answer"] == "A"
    assert review["message"] == "useful"
    assert review["model"]["model"] == "owner/m"
    assert review["page"] == "https://docs.example/page"
    assert review["consentVersion"] == "2.0.0"


def test_contribution_future_repository_row_is_eligible() -> None:
    row = schema.normalize_contribution_record(
        {
            "recordType": "conversation",
            "messages": [
                {"role": "user", "content": "Q", "ts": 1},
                {
                    "role": "assistant",
                    "content": "A",
                    "ts": 2,
                    "model": {"id": "m", "provider": "p", "model": "owner/m"},
                },
            ],
            "ts": 2,
        },
        envelope={"page": "https://docs.example/page", "consentVersion": "2.0.0"},
        server_ts_ms=1000,
        training_status="eligible",
        submission_id="receipt",
    )
    encoded = json.dumps(row, ensure_ascii=False)

    assert row["_dedup_key"] == "receipt:conversation"
    assert row["trainingStatus"] == "eligible"
    assert row["modelEvidence"] == "client_reported_per_message"
    assert '"trainingStatus": "eligible"' in encoded
