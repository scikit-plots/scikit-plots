from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

schema = importlib.import_module("_utils._dataset_schema")
dd = importlib.import_module("deduplicate_dataset")
proxy_app = importlib.import_module("app")



def _contribution_payload(*, consent_version: str = "2.0.0") -> dict:
    return {
        "schemaVersion": 4,
        "consentFlag": True,
        "consentVersion": consent_version,
        "sessionId": "attacker-stable-session-must-not-store",
        "page": "https://example.test/docs/page",
        "model": {"id": "claimed-model", "provider": "claimed-provider", "model": "claimed/model"},
        "records": [
            {
                "recordType": "qa",
                "answerIndex": 0,
                "query": "question",
                "answer": "answer",
                "ratingValue": 1,
                "ratingLabel": "helpful",
                "ratingTitle": "Helpful",
                "ratingMode": "quick",
                "message": "optional contribution note",
                "feedbackId": "feedback-current",
                "feedbackChainId": "feedback-current",
                "prevFeedbackId": None,
                "prevFeedbackIds": [],
                "editCount": 0,
                "ts": 123,
            }
        ],
    }


@pytest.fixture(autouse=True)
def _reset_collection_state(monkeypatch):
    proxy_app._contrib_quarantine.clear()
    proxy_app._contrib_rl.clear()
    # These integration cases exercise the mutable-ledger contribution lifecycle.
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_MODE", "ledger")
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_TOKEN", "")
    yield
    proxy_app._contrib_quarantine.clear()
    proxy_app._contrib_rl.clear()


def test_schema_v5_current_contribution_consent_and_server_training_state():
    assert schema.SCHEMA_VERSION == 5
    assert schema.CONSENT_VERSION_ENABLED is True
    assert schema.RESERVED_CONSENT_VERSION == "2.0.0"
    row = schema.normalize_contribution_record(
        _contribution_payload()["records"][0],
        envelope=_contribution_payload(),
        server_ts_ms=1000,
        submission_id="receipt",
    )
    assert row["trainingStatus"] == "quarantined"
    assert row["modelEvidence"] == "client_reported"
    assert row["conversationId"] is None
    assert row["feedbackId"] == "feedback-current"
    assert row["feedbackChainId"] == "feedback-current"
    assert row["prevFeedbackId"] is None
    assert row["prevFeedbackIds"] == []
    assert row["consentVersion"] == "2.0.0"


def test_training_builder_fails_closed_for_unreviewed_contributions():
    eligible = schema.normalize_contribution_record(
        _contribution_payload()["records"][0],
        envelope=_contribution_payload(),
        server_ts_ms=1000,
        training_status="eligible",
        submission_id="eligible",
    )
    quarantined = dict(
        eligible,
        trainingStatus="quarantined",
        _dedup_key="q:0",
        feedbackId="feedback-q",
        feedbackChainId="feedback-q",
    )
    legacy = dict(
        eligible,
        trainingStatus="legacy_unreviewed",
        _dedup_key="l:0",
        feedbackId="feedback-l",
        feedbackChainId="feedback-l",
    )
    eligible = dict(
        eligible,
        feedbackId="feedback-e",
        feedbackChainId="feedback-e",
    )
    clean = dd.deduplicate([quarantined, legacy, eligible])
    assert [r["trainingStatus"] for r in clean] == ["eligible"]
    audit = dd.deduplicate([quarantined, legacy, eligible], include_unreviewed=True)
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


def test_contribution_rejects_noncurrent_consent_version():
    with TestClient(proxy_app.app) as client:
        stale = client.post("/v1/contribute", json=_contribution_payload(consent_version="old"))
    assert stale.status_code == 422


def test_generic_feedback_route_is_contract_strict_and_worker_has_no_legacy_feedback_route():
    with TestClient(proxy_app.app) as client:
        rejected = client.post("/v1/feedback", json={"ratingValue": 1})
    assert rejected.status_code == 422
    assert "unsupported field" in rejected.text.lower()
    assert "telemetry permission" not in rejected.text.lower()

    worker = (ROOT / "_cf_worker" / "index.js").read_text(encoding="utf-8")
    assert "url.pathname === '/v1/feedback'" not in worker
    assert "FEEDBACK_PERSIST_ENABLED" not in worker
    assert "telemetryConsent" not in worker


def test_contribution_delete_header_is_cors_allowlisted():
    src = (PROXY / "app.py").read_text(encoding="utf-8")
    assert '"X-Contribution-Delete-Token"' in src
    assert 'CONTRIBUTION_REVIEW_TOKEN' in src
