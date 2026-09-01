from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app as proxy_app


def _payload() -> dict:
    return {
        "schemaVersion": 1,
        "consentFlag": True,
        "consentVersion": "2.0.0",
        "trainingConsentFlag": True,
        "trainingConsentVersion": "1.0.0",
        "feedbackId": "feedback-event-1",
        "answerIndex": 0,
        "ratingValue": 1,
        "ratingLabel": "helpful",
        "ratingTitle": "Helpful",
        "ratingMode": "quick",
        "ratingScaleMin": -1,
        "ratingScaleMax": 1,
        "message": "",
        "query": "What does this page cover?",
        "answer": "It covers the documented API.",
        "model": {
            "id": "qwen-7b",
            "provider": "huggingface",
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "label": "Qwen 7B",
        },
        "page": "https://docs.example.test/page",
        "ts": 1_788_138_000_000,
    }


def _assert_422(payload: dict, fragment: str) -> None:
    try:
        proxy_app._validate_feedback_review_payload(payload)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 422
        assert fragment.lower() in str(getattr(exc, "detail", "")).lower()
    else:
        raise AssertionError("payload must fail review validation")


def test_feedback_review_requires_originating_model_attribution() -> None:
    payload = _payload()
    payload["model"] = None
    _assert_422(payload, "originating model")


def test_feedback_review_requires_provider_and_model_name() -> None:
    payload = _payload()
    payload["model"] = {"provider": "huggingface", "model": ""}
    _assert_422(payload, "originating model")


def test_feedback_review_normalizer_preserves_model_evidence() -> None:
    record = proxy_app.normalize_feedback_review_record(
        _payload(), server_ts_ms=1_788_138_000_000, receipt_id="a" * 32
    )
    assert record["query"] == "What does this page cover?"
    assert record["answer"] == "It covers the documented API."
    assert record["model"]["provider"] == "huggingface"
    assert record["model"]["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert record["model"]["id"] == "qwen-7b"
    assert record["modelEvidence"] == "client_selected"


def test_feedback_rating_title_length_is_validated_independently() -> None:
    payload = _payload()
    payload["ratingLabel"] = "ok"
    payload["ratingTitle"] = "x" * 129
    try:
        proxy_app._validate_feedback_review_payload(payload)
    except Exception as exc:  # FastAPI HTTPException without binding test storage.
        assert getattr(exc, "status_code", None) == 422
        assert "ratingtitle" in str(getattr(exc, "detail", "")).lower()
    else:
        raise AssertionError("overlong ratingTitle must be rejected")
