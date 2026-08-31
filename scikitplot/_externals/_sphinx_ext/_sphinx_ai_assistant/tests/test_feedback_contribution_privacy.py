from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

schema = importlib.import_module("_utils._dataset_schema")
dd = importlib.import_module("deduplicate_dataset")
proxy_app = importlib.import_module("app")


def _legacy_feedback_payload() -> dict:
    return {
        "schemaVersion": 2,
        "ratingValue": 1,
        "ratingLabel": "helpful",
        "ratingTitle": "Helpful",
        "ratingMode": "panel",
        "message": "private note SECRET",
        "query": "private user question",
        "answer": "private model answer",
        "model": {"id": "forged", "provider": "forged", "model": "forged"},
        "answerIndex": 4,
        "page": "https://example.test/private?token=secret",
        "ts": 123,
        "sessionId": "feedback-event-id",
        "conversationId": "stable-chat-id",
    }


def _telemetry_feedback_payload() -> dict:
    payload = _legacy_feedback_payload()
    payload.update({
        "schemaVersion": schema.FEEDBACK_TELEMETRY_SCHEMA_VERSION,
        "telemetryConsent": True,
        "telemetryConsentVersion": schema.FEEDBACK_TELEMETRY_CONSENT_VERSION,
        "telemetryConsentAt": 1_700_000_000_000,
    })
    return payload


def _contribution_payload(*, consent_version: str = "1.0.0") -> dict:
    return {
        "schemaVersion": 3,
        "consentFlag": True,
        "consentVersion": consent_version,
        "sessionId": "attacker-stable-session-must-not-store",
        "page": "https://example.test/docs/page",
        "model": {"id": "claimed-model", "provider": "claimed-provider", "model": "claimed/model"},
        "records": [
            {
                "answerIndex": 0,
                "query": "question",
                "answer": "answer",
                "ratingValue": 1,
                "ratingLabel": "helpful",
                "ratingTitle": "Helpful",
                "ratingMode": "quick",
                "message": "optional contribution note",
                "feedbackId": "must-not-cross-link",
                "prevFeedbackId": "must-not-cross-link-either",
                "editCount": 99,
                "ts": 123,
            }
        ],
    }


@pytest.fixture(autouse=True)
def _reset_collection_state(monkeypatch):
    proxy_app._contrib_quarantine.clear()
    proxy_app._contrib_rl.clear()
    proxy_app._feedback_rl.clear()
    monkeypatch.setattr(proxy_app, "FEEDBACK_PERSIST_ENABLED", False)
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_TOKEN", "")
    yield
    proxy_app._contrib_quarantine.clear()
    proxy_app._contrib_rl.clear()
    proxy_app._feedback_rl.clear()


def test_schema_v4_preserves_legacy_v3_consent_and_server_training_state():
    assert schema.SCHEMA_VERSION == 4
    assert schema.CONSENT_VERSION_ENABLED is True
    assert schema.RESERVED_CONSENT_VERSION == "2.0.0"
    assert "1.0.0" in schema.LEGACY_CONSENT_VERSIONS
    row = schema.normalize_contribution_record(
        _contribution_payload()["records"][0],
        envelope=_contribution_payload(),
        server_ts_ms=1000,
        submission_id="receipt",
    )
    assert row["trainingStatus"] == "quarantined"
    assert row["modelEvidence"] == "client_reported"
    assert row["conversationId"] is None
    assert row["feedbackId"] is None
    assert row["prevFeedbackId"] is None
    assert row["consentVersion"] == "1.0.0"


def test_feedback_normalizer_discards_content_identity_and_model_even_from_legacy_direct_caller():
    row = schema.normalize_feedback_record(_legacy_feedback_payload(), server_ts_ms=1000)
    assert row["trainingStatus"] == "telemetry"
    assert row["query"] == ""
    assert row["answer"] == ""
    assert row["message"] == ""
    assert row["model"] is None
    assert row["modelEvidence"] is None
    assert row["page"] == ""
    assert row["conversationId"] is None
    assert row["feedbackId"] == "feedback-event-id"
    assert row["ratingSlug"] == "helpful"


def test_training_builder_fails_closed_for_feedback_quarantine_and_legacy_rows():
    eligible = schema.normalize_contribution_record(
        _contribution_payload()["records"][0],
        envelope=_contribution_payload(),
        server_ts_ms=1000,
        training_status="eligible",
        submission_id="eligible",
    )
    quarantined = dict(eligible, trainingStatus="quarantined", _dedup_key="q:0")
    legacy = dict(eligible, trainingStatus="legacy_unreviewed", _dedup_key="l:0")
    feedback = schema.normalize_feedback_record(_legacy_feedback_payload(), server_ts_ms=1000)
    clean = dd.deduplicate([feedback, quarantined, legacy, eligible])
    assert [r["trainingStatus"] for r in clean] == ["eligible"]
    audit = dd.deduplicate([feedback, quarantined, legacy, eligible], include_unreviewed=True)
    assert {r["trainingStatus"] for r in audit} == {"eligible", "quarantined", "legacy_unreviewed"}
    assert all(r["_source"] == "contribution" for r in audit)


def test_contribution_enters_mutable_quarantine_and_delete_capability_physically_removes_it(monkeypatch):
    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_contribution_payload())
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["status"] == "quarantined"
        assert body["receiptId"] in proxy_app._contrib_quarantine
        assert body["deleteToken"] not in json.dumps(proxy_app._contrib_quarantine[body["receiptId"]])
        row = proxy_app._contrib_quarantine[body["receiptId"]]["records"][0]
        assert row["trainingStatus"] == "quarantined"
        assert row["conversationId"] is None
        assert row["modelEvidence"] == "client_reported"

        denied = client.delete(
            f"/v1/contribute/{body['receiptId']}",
            headers={"X-Contribution-Delete-Token": "wrong"},
        )
        assert denied.status_code == 403
        deleted = client.delete(
            f"/v1/contribute/{body['receiptId']}",
            headers={"X-Contribution-Delete-Token": body["deleteToken"]},
        )
        assert deleted.status_code == 200
        tombstone = proxy_app._contrib_quarantine[body["receiptId"]]
        assert tombstone["state"] == "deleted"
        assert tombstone["records"] == []
        assert tombstone["bytes"] == 0
        assert deleted.json()["contentRemovedFromActiveLedger"] is True
        assert deleted.json()["physicalErasureGuaranteed"] is False
        assert deleted.json()["physicalErasureScope"] == "not-guaranteed"


def test_only_authorized_review_can_promote_to_durable_eligible_storage(monkeypatch):
    captured: dict[str, object] = {}

    class FakeStorage:
        primary = object()
        def set_client(self, client): pass
        async def initialize(self): pass
        def manifest(self): return {"targets": []}
        async def close(self): pass

    async def fake_persist(
        *, kind: str, content: bytes, commit_message: str, path_timestamp: float | None = None
    ):
        captured["kind"] = kind
        captured["content"] = content
        captured["commit_message"] = commit_message
        captured["path_timestamp"] = path_timestamp
        return SimpleNamespace(record_id="record123", primary="fake", mirrors={})

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "_persist_storage_record", fake_persist)
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_TOKEN", "review-secret")

    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_contribution_payload()).json()
        rid = created["receiptId"]
        assert client.post(f"/v1/contribute/{rid}/promote").status_code == 401
        promoted = client.post(
            f"/v1/contribute/{rid}/promote",
            headers={"Authorization": "Bearer review-secret"},
        )
        assert promoted.status_code == 200, promoted.text
        assert promoted.json()["status"] == "eligible"
        lifecycle = proxy_app._contrib_quarantine[rid]
        assert lifecycle["state"] == "eligible"
        assert lifecycle["records"] == []  # raw pending content removed from mutable ledger
        assert lifecycle["bytes"] == 0
        assert lifecycle["storage"]["recordId"] == "record123"

    rows = [json.loads(line) for line in bytes(captured["content"]).decode().splitlines()]
    assert rows and all(row["trainingStatus"] == "eligible" for row in rows)
    assert all(row["modelEvidence"] == "client_reported" for row in rows)


def test_contribution_rejects_stale_consent_version():
    with TestClient(proxy_app.app) as client:
        stale = client.post("/v1/contribute", json=_contribution_payload(consent_version="old"))
    assert stale.status_code == 422


def test_feedback_endpoint_requires_explicit_versioned_telemetry_permission_before_rate_limit():
    with TestClient(proxy_app.app) as client:
        missing = client.post("/v1/feedback", json=_legacy_feedback_payload())
        stale = _telemetry_feedback_payload()
        stale["telemetryConsentVersion"] = "0.9.0"
        stale_response = client.post("/v1/feedback", json=stale)
    assert missing.status_code == 403
    assert stale_response.status_code == 403
    assert proxy_app._feedback_rl == {}


def test_feedback_endpoint_rejects_malformed_consent_contract():
    bad_schema = _telemetry_feedback_payload()
    bad_schema["schemaVersion"] = 3
    no_timestamp = _telemetry_feedback_payload()
    no_timestamp.pop("telemetryConsentAt")
    with TestClient(proxy_app.app) as client:
        assert client.post("/v1/feedback", json=bad_schema).status_code == 422
        assert client.post("/v1/feedback", json=no_timestamp).status_code == 422


def test_feedback_endpoint_persistence_can_only_store_minimal_telemetry(monkeypatch):
    captured: dict[str, object] = {}

    class FakeStorage:
        primary = object()
        def set_client(self, client): pass
        async def initialize(self): pass
        def manifest(self): return {"targets": []}
        async def close(self): pass

    async def fake_persist(*, kind: str, content: bytes, commit_message: str):
        captured["content"] = content
        return SimpleNamespace(record_id="fb", primary="fake", mirrors={})

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "_persist_storage_record", fake_persist)
    monkeypatch.setattr(proxy_app, "FEEDBACK_PERSIST_ENABLED", True)
    with TestClient(proxy_app.app) as client:
        response = client.post("/v1/feedback", json=_telemetry_feedback_payload())
        assert response.status_code == 200, response.text

    row = json.loads(bytes(captured["content"]).decode())
    assert row["trainingStatus"] == "telemetry"
    for key in ("query", "answer", "message", "page"):
        assert row[key] == ""
    assert row["model"] is None
    assert row["conversationId"] is None


def test_feedback_has_independent_small_body_limit(monkeypatch):
    monkeypatch.setattr(proxy_app, "FEEDBACK_MAX_BODY_BYTES", 100)
    with TestClient(proxy_app.app) as client:
        response = client.post("/v1/feedback", content=json.dumps({"ratingValue": 1, "junk": "x" * 500}))
    assert response.status_code == 413


def test_cloudflare_feedback_contract_is_minimal_and_opt_in():
    src = (ROOT / "_cf_worker" / "index.js").read_text(encoding="utf-8")
    start = src.index("// ── POST /v1/feedback")
    end = src.index("// ── Global Share", start)
    route = src[start:end]
    assert "const MAX_FB_BYTES = 16 * 1024" in route
    assert "trainingStatus: 'telemetry'" in route
    assert "FEEDBACK_PERSIST_ENABLED" in route
    assert "fb.telemetryConsent !== true" in route
    assert "fb.telemetryConsentVersion !== '1.0.0'" in route
    assert "fb.schemaVersion !== 4" in route
    assert "JSON.stringify({ ...fb" not in route
    for forbidden in ("query: fb.query", "answer: fb.answer", "message: fb.message", "sessionId:", "conversationId:"):
        assert forbidden not in route


def test_contribution_delete_header_is_cors_allowlisted():
    src = (PROXY / "app.py").read_text(encoding="utf-8")
    assert '"X-Contribution-Delete-Token"' in src
    assert '"FEEDBACK_PERSIST_ENABLED", "false"' in src
    assert 'CONTRIBUTION_REVIEW_TOKEN' in src
