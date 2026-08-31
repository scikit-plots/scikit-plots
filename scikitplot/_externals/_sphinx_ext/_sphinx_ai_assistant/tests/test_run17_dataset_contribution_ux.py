from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

schema = importlib.import_module("_utils._dataset_schema")
proxy_app = importlib.import_module("app")


def _v4_conversation_payload(*, consent: str = "2.0.0") -> dict:
    return {
        "schemaVersion": 4,
        "consentFlag": True,
        "consentVersion": consent,
        "page": "https://example.test/docs/page",
        "model": None,
        "records": [
            {
                "recordType": "conversation",
                "message": "Useful multi-turn debugging example",
                "messages": [
                    {"role": "user", "content": "How do I debug this?", "ts": 10},
                    {
                        "role": "assistant",
                        "content": "Start with the failing boundary.",
                        "ts": 20,
                        "model": {"id": "m1", "provider": "p1", "model": "owner/m1"},
                        "feedback": {
                            "ratingValue": 1,
                            "ratingLabel": "helpful",
                            "ratingMode": "quick",
                            "note": "good diagnostic sequence",
                        },
                    },
                    # Error/tool/system material is not part of the conversation record family.
                    {"role": "error", "content": "trace with accidental secret", "ts": 25},
                    {"role": "user", "content": "Then what?", "ts": 30},
                    {
                        "role": "assistant",
                        "content": "Verify the exact packaged bytes.",
                        "ts": 40,
                        "model": {"id": "m2", "provider": "p2", "model": "owner/m2"},
                        "feedback": None,
                    },
                ],
                # Unknown client fields must not become stored columns/capabilities.
                "deleteToken": "must-not-persist",
                "editToken": "must-not-persist",
            }
        ],
    }


def _legacy_v3_payload() -> dict:
    return {
        "schemaVersion": 3,
        "consentFlag": True,
        "consentVersion": "1.0.0",
        "page": "https://example.test/docs/page",
        "model": {"id": "legacy", "provider": "test", "model": "legacy/model"},
        "records": [
            {
                "answerIndex": 0,
                "query": "q",
                "answer": "a",
                "ratingValue": 1,
                "ratingLabel": "helpful",
                "ratingMode": "quick",
            }
        ],
    }


def _reset_state() -> None:
    proxy_app._contrib_quarantine.clear()
    proxy_app._contrib_rl.clear()


def test_schema_v4_conversation_is_one_ordered_record_with_per_message_evidence():
    payload = _v4_conversation_payload()
    row = schema.normalize_contribution_record(
        payload["records"][0],
        envelope=payload,
        server_ts_ms=100,
        submission_id="receipt",
    )
    assert schema.SCHEMA_VERSION == 4
    assert schema.RESERVED_CONSENT_VERSION == "2.0.0"
    assert row["recordType"] == "conversation"
    assert row["_dedup_key"] == "receipt:conversation"
    assert row["trainingStatus"] == "quarantined"
    assert row["query"] == ""
    assert row["answer"] == ""
    assert row["model"] is None
    assert row["modelEvidence"] == "client_reported_per_message"
    assert row["message"] == "Useful multi-turn debugging example"
    assert [m["role"] for m in row["messages"]] == ["user", "assistant", "user", "assistant"]
    assert row["messages"][1]["model"]["id"] == "m1"
    assert row["messages"][3]["model"]["id"] == "m2"
    assert row["messages"][1]["feedback"]["ratingSlug"] == "helpful"
    assert row["messages"][1]["feedback"]["note"] == "good diagnostic sequence"
    assert "deleteToken" not in row and "editToken" not in row


def test_schema_v4_qa_preserves_existing_family_with_explicit_record_type():
    payload = {
        "schemaVersion": 4,
        "consentFlag": True,
        "consentVersion": "2.0.0",
        "page": "https://example.test/docs",
        "model": {"id": "m", "provider": "p"},
        "records": [{"recordType": "qa", "answerIndex": 7, "query": "q", "answer": "a"}],
    }
    row = schema.normalize_contribution_record(
        payload["records"][0], envelope=payload, server_ts_ms=1, submission_id="r"
    )
    assert row["recordType"] == "qa"
    assert row["answerIndex"] == 7
    assert row["messages"] is None
    assert row["query"] == "q"
    assert row["answer"] == "a"
    assert row["modelEvidence"] == "client_reported"


def test_app_accepts_legacy_v3_consent_but_requires_v2_consent_for_v4():
    _reset_state()
    with TestClient(proxy_app.app) as client:
        legacy = client.post("/v1/contribute", json=_legacy_v3_payload())
        assert legacy.status_code == 200, legacy.text

        wrong = client.post("/v1/contribute", json=_v4_conversation_payload(consent="1.0.0"))
        assert wrong.status_code == 422
        assert "Consent text changed" in wrong.text

        current = client.post("/v1/contribute", json=_v4_conversation_payload())
        assert current.status_code == 200, current.text
        assert current.json()["consentVersion"] == "2.0.0"


def test_app_quarantines_whole_conversation_as_exactly_one_record_and_delete_removes_active_copy():
    _reset_state()
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_v4_conversation_payload())
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["rows"] == 1
        entry = proxy_app._contrib_quarantine[body["receiptId"]]
        assert len(entry["records"]) == 1
        row = entry["records"][0]
        assert row["recordType"] == "conversation"
        assert row["trainingStatus"] == "quarantined"
        assert row["_dedup_key"].endswith(":conversation")

        deleted = client.delete(
            f"/v1/contribute/{body['receiptId']}",
            headers={"X-Contribution-Delete-Token": body["deleteToken"]},
        )
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["status"] == "deleted"
        assert proxy_app._contrib_quarantine[body["receiptId"]]["records"] == []


def test_empty_or_non_training_conversation_messages_fail_closed():
    _reset_state()
    payload = _v4_conversation_payload()
    payload["records"][0]["messages"] = [
        {"role": "error", "content": "error only"},
        {"role": "system", "content": "system only"},
    ]
    with TestClient(proxy_app.app) as client:
        response = client.post("/v1/contribute", json=payload)
    assert response.status_code == 422
    assert "No valid contribution records" in response.text


def test_conversation_withdrawal_tombstone_uses_same_receipt_scoped_dedup_key():
    row = schema.normalize_contribution_withdrawal_record(
        "receipt:conversation", server_ts_ms=500
    )
    assert row["_dedup_key"] == "receipt:conversation"
    assert row["action"] == "withdraw"
    assert row["trainingStatus"] == "withdrawn"
    assert row["answerIndex"] is None
    assert row["messages"] is None
