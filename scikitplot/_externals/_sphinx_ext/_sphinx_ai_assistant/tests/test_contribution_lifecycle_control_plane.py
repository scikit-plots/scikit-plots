from __future__ import annotations

import asyncio
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

from _utils._contribution_ledger import (  # noqa: E402
    ContributionLedgerError,
    MemoryContributionLedger,
    SQLiteContributionLedger,
)

schema = importlib.import_module("_utils._dataset_schema")
dd = importlib.import_module("deduplicate_dataset")
proxy_app = importlib.import_module("app")


def _payload() -> dict:
    return {
        "schemaVersion": 3,
        "consentFlag": True,
        "consentVersion": "1.0.0",
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
                "message": "optional note",
                "ts": 123,
            }
        ],
    }


def _entry(receipt_id: str = "receipt", token_hash: str = "h" * 64) -> dict:
    row = schema.normalize_contribution_record(
        _payload()["records"][0],
        envelope=_payload(),
        server_ts_ms=1000,
        submission_id=receipt_id,
    )
    return {
        "receiptId": receipt_id,
        "state": "quarantined",
        "records": [row],
        "bytes": len(json.dumps(row).encode()),
        "deleteTokenHash": token_hash,
        "expiresAt": 9_999_999_999.0,
        "receivedAt": 1000.0,
        "dedupKeys": [row["_dedup_key"]],
        "storage": {},
        "withdrawalStorage": {},
        "currentViewRemoval": {},
        "lastError": "",
    }


def test_memory_ledger_atomically_blocks_double_promotion():
    async def run():
        ledger = MemoryContributionLedger(max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32)
        await ledger.create(_entry())
        claimed = await ledger.begin_promotion("receipt")
        assert claimed["state"] == "promoting"
        with pytest.raises(ContributionLedgerError) as exc:
            await ledger.begin_promotion("receipt")
        assert exc.value.code == "PROMOTION_IN_PROGRESS"

    asyncio.run(run())


def test_sqlite_ledger_survives_reconstruction_and_drops_raw_rows_after_promotion(tmp_path):
    db = tmp_path / "lifecycle.sqlite3"

    async def run():
        first = SQLiteContributionLedger(
            str(db), max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32
        )
        await first.initialize()
        await first.create(_entry(token_hash="a" * 64))
        await first.close()

        second = SQLiteContributionLedger(
            str(db), max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32
        )
        await second.initialize()
        recovered = await second.get("receipt")
        assert recovered is not None
        assert recovered["state"] == "quarantined"
        assert recovered["records"][0]["query"] == "question"

        claimed = await second.begin_promotion("receipt")
        assert claimed["state"] == "promoting"
        await second.mark_promoted(
            "receipt",
            storage={
                "recordId": "record-1",
                "primary": "primary",
                "mirrors": {"mirror": "ok"},
                "paths": {"primary": "contributions/a.jsonl"},
            },
        )
        promoted = await second.get("receipt")
        assert promoted is not None
        assert promoted["state"] == "eligible"
        assert promoted["records"] == []
        assert promoted["bytes"] == 0
        assert promoted["dedupKeys"] == ["receipt:0"]
        assert promoted["deleteTokenHash"] == "a" * 64
        await second.close()

    asyncio.run(run())
    # Plain canonical contribution content is intentionally removed after promotion.
    raw = db.read_bytes()
    assert b'"query":"question"' not in raw
    assert b'"answer":"answer"' not in raw



def test_sqlite_restart_reclaims_interrupted_lifecycle_states(tmp_path):
    db = tmp_path / "recovery.sqlite3"

    async def run():
        first = SQLiteContributionLedger(
            str(db), max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32
        )
        await first.initialize()
        await first.create(_entry(token_hash="b" * 64))
        claimed = await first.begin_promotion("receipt")
        assert claimed["state"] == "promoting"
        await first.close()

        # Startup belongs to a new process lifecycle: stale promotion ownership
        # is reclaimed so review can replay to the receipt-stable provider path.
        second = SQLiteContributionLedger(
            str(db), max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32
        )
        await second.initialize()
        recovered = await second.get("receipt")
        assert recovered is not None
        assert recovered["state"] == "quarantined"
        assert recovered["records"][0]["query"] == "question"
        await second.begin_promotion("receipt")
        await second.mark_promoted(
            "receipt",
            storage={"recordId": "record-1", "primary": "primary", "mirrors": {}, "paths": {}},
        )
        await second.begin_withdrawal("receipt")
        await second.close()

        third = SQLiteContributionLedger(
            str(db), max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32
        )
        await third.initialize()
        recovered_withdrawal = await third.get("receipt")
        assert recovered_withdrawal is not None
        assert recovered_withdrawal["state"] == "eligible"
        assert recovered_withdrawal["lastError"] == "RECOVERED_AFTER_RESTART"
        await third.close()

    asyncio.run(run())


def test_memory_terminal_tombstones_are_bounded_by_retention():
    async def run():
        ledger = MemoryContributionLedger(
            max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32,
            terminal_retention_seconds=60,
        )
        old = _entry("old")
        old.update({"state": "deleted", "records": [], "bytes": 0, "updatedAt": 1.0})
        ledger.entries["old"] = old
        assert await ledger.get("old") is None
        assert "old" not in ledger.entries

    asyncio.run(run())

def test_withdrawal_tombstone_suppresses_previously_eligible_training_row():
    eligible = schema.normalize_contribution_record(
        _payload()["records"][0], envelope=_payload(), server_ts_ms=1000,
        training_status="eligible", submission_id="receipt",
    )
    withdrawal = schema.normalize_contribution_withdrawal_record("receipt:0", server_ts_ms=2000)
    assert withdrawal["action"] == "withdraw"
    assert withdrawal["trainingStatus"] == "withdrawn"
    for key in ("query", "answer", "message", "page"):
        assert withdrawal[key] == ""
    assert withdrawal["model"] is None
    assert dd.deduplicate([eligible, withdrawal]) == []
    # A stale/older withdrawal must not suppress a later eligible row.
    assert dd.deduplicate([dict(eligible, _ts=3000), withdrawal])[0]["trainingStatus"] == "eligible"


@pytest.fixture
def _clean_app(monkeypatch):
    if hasattr(proxy_app._CONTRIBUTION_LEDGER, "clear_for_tests"):
        proxy_app._CONTRIBUTION_LEDGER.clear_for_tests()
    proxy_app._contrib_rl.clear()
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REQUIRE_DURABLE", False)
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_TOKEN", "review-secret")
    yield
    if hasattr(proxy_app._CONTRIBUTION_LEDGER, "clear_for_tests"):
        proxy_app._CONTRIBUTION_LEDGER.clear_for_tests()
    proxy_app._contrib_rl.clear()


def test_post_promotion_delete_becomes_truthful_training_withdrawal(monkeypatch, _clean_app):
    persisted: list[dict] = []
    removals: list[dict] = []

    class FakeStorage:
        primary = object()
        def set_client(self, client): pass
        async def initialize(self): pass
        def manifest(self): return {"targets": []}
        async def close(self): pass
        async def remove_current_view(self, paths, *, record_id=None, commit_message=""):
            removals.append({"paths": dict(paths), "record_id": record_id})
            return {"primary": "removed-current-view", "mirror": "degraded"}

    async def fake_persist(*, kind: str, content: bytes, commit_message: str, path_timestamp=None):
        idx = len(persisted) + 1
        persisted.append({"kind": kind, "content": content, "message": commit_message})
        return SimpleNamespace(
            record_id=f"record-{idx}",
            primary="primary",
            mirrors={"mirror": "ok"},
            paths={
                "primary": f"contributions/record-{idx}.jsonl",
                "mirror": f"contributions/record-{idx}.jsonl",
            },
        )

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "_persist_storage_record", fake_persist)

    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_payload())
        assert created.status_code == 200, created.text
        receipt = created.json()
        rid = receipt["receiptId"]
        token = receipt["deleteToken"]

        denied_status = client.get(f"/v1/contribute/{rid}", headers={"X-Contribution-Delete-Token": "wrong"})
        assert denied_status.status_code == 403

        promoted = client.post(
            f"/v1/contribute/{rid}/promote",
            headers={"Authorization": "Bearer review-secret"},
        )
        assert promoted.status_code == 200, promoted.text

        status = client.get(
            f"/v1/contribute/{rid}",
            headers={"X-Contribution-Delete-Token": token},
        )
        assert status.status_code == 200
        assert status.json()["status"] == "eligible"
        assert status.json()["trainingEligible"] is True

        withdrawn = client.delete(
            f"/v1/contribute/{rid}",
            headers={"X-Contribution-Delete-Token": token},
        )
        assert withdrawn.status_code == 202, withdrawn.text
        body = withdrawn.json()
        assert body["status"] == "withdrawn"
        assert body["trainingWithdrawn"] is True
        assert body["trainingEligible"] is False
        assert body["physicalErasureGuaranteed"] is False
        assert body["physicalErasureScope"] == "not-guaranteed"
        assert body["currentViewRemoval"]["primary"] == "removed-current-view"
        assert body["currentViewRemoval"]["mirror"] == "degraded"

        again = client.delete(
            f"/v1/contribute/{rid}",
            headers={"X-Contribution-Delete-Token": token},
        )
        assert again.status_code == 200
        assert again.json()["status"] == "withdrawn"

    assert len(persisted) == 2
    promoted_rows = [json.loads(line) for line in persisted[0]["content"].decode().splitlines()]
    withdrawal_rows = [json.loads(line) for line in persisted[1]["content"].decode().splitlines()]
    assert promoted_rows[0]["trainingStatus"] == "eligible"
    assert withdrawal_rows[0]["trainingStatus"] == "withdrawn"
    assert withdrawal_rows[0]["action"] == "withdraw"
    for key in ("query", "answer", "message", "page"):
        assert withdrawal_rows[0][key] == ""
    assert withdrawal_rows[0]["model"] is None
    assert dd.deduplicate(promoted_rows + withdrawal_rows) == []
    assert removals == [{
        "paths": {"primary": "contributions/record-1.jsonl", "mirror": "contributions/record-1.jsonl"},
        "record_id": "record-1",
    }]


def test_durable_required_mode_fails_closed_on_memory_ledger(monkeypatch, _clean_app):
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REQUIRE_DURABLE", True)
    with TestClient(proxy_app.app) as client:
        response = client.post("/v1/contribute", json=_payload())
    assert response.status_code == 503
    assert "Durable contribution lifecycle storage is required" in response.text


def test_browser_wording_distinguishes_pending_delete_from_post_promotion_withdrawal():
    src = (ROOT / "_static" / "ai-assistant.js").read_text(encoding="utf-8")
    assert "Delete pending / withdraw training use" in src
    assert "versioned provider history is not claimed physically erased" in src
    assert "Pending contribution removed from the active review ledger before promotion." in src
