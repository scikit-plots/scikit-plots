from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _dataset_schema as schema


EXPECTED_CANONICAL_COLUMNS = [
    "schemaVersion",
    "_source",
    "_ts",
    "_dedup_key",
    "conversationId",
    "feedbackId",
    "feedbackChainId",
    "recordType",
    "answerIndex",
    "action",
    "prevFeedbackId",
    "prevFeedbackIds",
    "editCount",
    "status",
    "trainingStatus",
    "ratingValue",
    "ratingSlug",
    "ratingTitle",
    "ratingMode",
    "ratingScaleMin",
    "ratingScaleMax",
    "qualityScore",
    "qualityPercent",
    "message",
    "query",
    "answer",
    "messages",
    "model",
    "modelEvidence",
    "page",
    "consentVersion",
    "trainingConsentVersion",
    "ts",
]


def test_saved_preview_column_contract_matches_server_schema_v5() -> None:
    assert schema.SCHEMA_VERSION == 5
    assert schema.CANONICAL_COLUMNS == EXPECTED_CANONICAL_COLUMNS


def test_feedback_review_saved_shape_assumptions_match_normalizer() -> None:
    payload = {
        "feedbackId": "fb-2",
        "feedbackChainId": "fb-0",
        "prevFeedbackId": "fb-1",
        "prevFeedbackIds": ["fb-0", "fb-1"],
        "editCount": 2,
        "answerIndex": 0,
        "ratingValue": 1,
        "ratingLabel": "helpful",
        "ratingTitle": "Helpful",
        "ratingMode": "quick",
        "ratingScaleMin": -1,
        "ratingScaleMax": 1,
        "message": "useful note",
        "query": "Q",
        "answer": "A",
        "model": {"id": "m", "provider": "p", "model": "owner/m"},
        "page": "https://docs.example/page",
        "consentVersion": "1.0.0",
        "trainingConsentVersion": "1.0.0",
        "ts": 20,
    }
    row = schema.normalize_feedback_review_record(
        payload, server_ts_ms=1234, receipt_id="receipt"
    )
    assert list(row) == EXPECTED_CANONICAL_COLUMNS + ["feedbackReview"]
    assert row["_source"] == "feedback"
    assert row["_ts"] == 1234
    assert row["_dedup_key"] == "receipt:feedback"
    assert row["recordType"] == "qa"
    assert row["action"] == "review"
    assert row["trainingStatus"] == "eligible"
    assert row["ratingSlug"] == "helpful"
    assert row["qualityScore"] == 1.0
    assert row["qualityPercent"] == 100.0
    assert row["modelEvidence"] == "client_selected"
    assert list(row["model"]) == schema.MODEL_KEYS
    assert row["feedbackReview"] is True


def test_conversation_contribution_saved_shape_assumptions_match_normalizer() -> None:
    rec = {
        "recordType": "conversation",
        "message": "why useful",
        "ts": 30,
        "messages": [
            {"role": "user", "content": "Q", "ts": 1},
            {
                "role": "assistant",
                "content": "A",
                "ts": 2,
                "model": {"id": "m", "provider": "p", "model": "owner/m"},
                "feedback": {
                    "feedbackId": "fb-2",
                    "feedbackChainId": "fb-0",
                    "prevFeedbackId": "fb-1",
                    "prevFeedbackIds": ["fb-0", "fb-1"],
                    "editCount": 2,
                    "ratingValue": 1,
                    "ratingLabel": "helpful",
                    "ratingTitle": "Helpful",
                    "ratingMode": "quick",
                    "note": "good",
                },
            },
        ],
    }
    envelope = {
        "page": "https://docs.example/page",
        "consentVersion": "2.0.0",
        "model": None,
    }
    row = schema.normalize_contribution_record(
        rec,
        envelope=envelope,
        server_ts_ms=1234,
        training_status="quarantined",
        submission_id="receipt",
    )
    assert list(row) == EXPECTED_CANONICAL_COLUMNS
    assert row["_source"] == "contribution"
    assert row["_ts"] == 1234
    assert row["_dedup_key"] == "receipt:conversation"
    assert row["recordType"] == "conversation"
    assert row["trainingStatus"] == "quarantined"
    assert row["model"] is None
    assert row["modelEvidence"] == "client_reported_per_message"
    assert row["messages"][1]["feedback"]["ratingSlug"] == "helpful"
    assert row["messages"][1]["feedback"]["feedbackChainId"] == "fb-0"
    assert row["messages"][1]["feedback"]["prevFeedbackIds"] == ["fb-0", "fb-1"]
    assert list(row["messages"][1]["model"]) == schema.MODEL_KEYS
