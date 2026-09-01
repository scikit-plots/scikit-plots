from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _dataset_schema as schema


def test_feedback_and_conversation_use_receipt_type_dedup_shape() -> None:
    feedback = schema.normalize_feedback_review_record(
        {
            "feedbackId": "fb-1",
            "ratingValue": 1,
            "ratingLabel": "helpful",
            "ratingTitle": "Helpful",
            "ratingMode": "quick",
            "ratingScaleMin": -1,
            "ratingScaleMax": 1,
            "query": "Q",
            "answer": "A",
            "model": {"id": "m", "provider": "p", "model": "owner/m"},
        },
        server_ts_ms=1234,
        receipt_id="receipt",
    )
    conversation = schema.normalize_contribution_record(
        {
            "recordType": "conversation",
            "messages": [{"role": "user", "content": "Q", "ts": 1}],
        },
        envelope={"consentVersion": "2.0.0"},
        server_ts_ms=1234,
        submission_id="receipt",
    )

    assert feedback["_dedup_key"] == "receipt:feedback"
    assert conversation["_dedup_key"] == "receipt:conversation"
    assert feedback["_dedup_key"].split(":", 1)[0] == conversation["_dedup_key"].split(":", 1)[0]


def test_ordinary_feedback_telemetry_uses_identifier_first_dedup_shape() -> None:
    row = schema.normalize_feedback_record(
        {
            "feedbackId": "fb-event",
            "answerIndex": 0,
            "ratingValue": 1,
            "ratingLabel": "helpful",
            "ratingTitle": "Helpful",
            "ratingMode": "quick",
            "ts": 123,
        },
        server_ts_ms=456,
    )
    assert row["_dedup_key"] == "fb-event:feedback"
    assert row["trainingStatus"] == "telemetry"
    assert row["query"] == row["answer"] == row["page"] == ""
    assert row["model"] is None
