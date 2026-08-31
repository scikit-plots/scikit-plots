from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

proxy_app = importlib.import_module("app")
share_store_mod = importlib.import_module("_utils._share_store")


def _snapshot(text: str = "answer") -> dict:
    return {
        "schema_version": "2.0",
        "session": {
            "id": "session-run18",
            "page_url": "https://example.test/docs/page",
            "page_title": "Run 18",
            "assistant_name": "AI Assistant",
            "exported_at": 1,
            "exported_at_iso": "2026-08-30T00:00:00Z",
        },
        "turns": [{
            "turn_index": 0,
            "user": {"text": "question", "ts": 1, "ts_iso": "2026-08-30T00:00:00Z"},
            "assistant": {"text": text, "ts": 2, "ts_iso": "2026-08-30T00:00:01Z"},
        }],
        "records": [
            {
                "turn_index": 0,
                "message_index": 0,
                "role": "user",
                "text": "question",
                "ts": 1,
                "ts_iso": "2026-08-30T00:00:00Z",
                "model_id": None,
                "model_provider": None,
                "model_name": None,
                "feedback_rating_value": None,
                "feedback_rating_label": None,
                "feedback_message": None,
                "session_id": "session-run18",
                "page_url": "https://example.test/docs/page",
            },
            {
                "turn_index": 0,
                "message_index": 1,
                "role": "assistant",
                "text": text,
                "ts": 2,
                "ts_iso": "2026-08-30T00:00:01Z",
                "model_id": None,
                "model_provider": None,
                "model_name": None,
                "feedback_rating_value": None,
                "feedback_rating_label": None,
                "feedback_message": None,
                "session_id": "session-run18",
                "page_url": "https://example.test/docs/page",
            },
        ],
    }


def _contribution(answer: str = "answer") -> dict:
    return {
        "schemaVersion": 4,
        "consentFlag": True,
        "consentVersion": "2.0.0",
        "page": "https://example.test/docs/page",
        "model": None,
        "records": [{"recordType": "qa", "answerIndex": 0, "query": "question", "answer": answer}],
    }


def _operation_headers(resource_id: str, raw_token: str, operation_id: str = "o" * 32) -> dict[str, str]:
    return {
        "X-AI-Operation-Id": operation_id,
        "X-AI-Resource-Id": resource_id,
        "X-AI-Management-Token-Hash": hashlib.sha256(raw_token.encode("utf-8")).hexdigest(),
        "X-AI-Operation-Created-At": str(int(time.time() * 1000)),
    }


@pytest.fixture(autouse=True)
def _reset_runtime(monkeypatch):
    proxy_app._share_store.clear()
    proxy_app._share_rl.clear()
    proxy_app._contrib_quarantine.clear()
    proxy_app._contrib_rl.clear()
    monkeypatch.setattr(proxy_app, "SHARE_WRITE_TOKEN", "")
    monkeypatch.setattr(proxy_app, "SHARE_PUBLIC_BASE_URL", "https://share.example.test")
    monkeypatch.setattr(proxy_app, "SHARE_RATE_LIMIT_PER_HOUR", 10)
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_RATE_LIMIT_PER_HOUR", 5)
    yield
    proxy_app._share_store.clear()
    proxy_app._share_rl.clear()
    proxy_app._contrib_quarantine.clear()
    proxy_app._contrib_rl.clear()


def test_share_envelope_keeps_raw_revoke_capability_client_side_and_replays_create(monkeypatch):
    raw_token = "run18-private-share-capability-" + "x" * 24
    resource_id = "1" * 32
    headers = _operation_headers(resource_id, raw_token, "shareop" + "a" * 25)
    monkeypatch.setattr(proxy_app, "SHARE_RATE_LIMIT_PER_HOUR", 1)

    with TestClient(proxy_app.app) as client:
        created = client.post(
            "/v1/share", headers=headers,
            json={"snapshot": _snapshot("first"), "format": "json", "ttlDays": 7},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["uuid"] == resource_id
        assert "editToken" not in body
        entry = proxy_app._share_store[resource_id]
        assert entry["edit_hash"] == headers["X-AI-Management-Token-Hash"]
        assert raw_token not in json.dumps(entry)

        # Same identity + same reviewed content resolves the existing object even
        # though the normal rate limit is now exhausted.
        replay = client.post(
            "/v1/share", headers=headers,
            json={"snapshot": _snapshot("first"), "format": "json", "ttlDays": 7},
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["uuid"] == resource_id
        assert replay.json()["idempotentReplay"] is True
        assert "editToken" not in replay.json()
        assert len(proxy_app._share_store) == 1

        conflict = client.post(
            "/v1/share", headers=headers,
            json={"snapshot": _snapshot("different"), "format": "json", "ttlDays": 7},
        )
        assert conflict.status_code == 409

        wrong_capability = _operation_headers(resource_id, "different-share-capability-" + "z" * 32, headers["X-AI-Operation-Id"])
        wrong_capability["X-AI-Operation-Created-At"] = headers["X-AI-Operation-Created-At"]
        rejected = client.post(
            "/v1/share", headers=wrong_capability,
            json={"snapshot": _snapshot("first"), "format": "json", "ttlDays": 7},
        )
        assert rejected.status_code == 409

        revoked = client.post(
            "/v1/share/revoke",
            headers={"X-Share-Edit-Token": raw_token},
            json={"shareId": resource_id},
        )
        assert revoked.status_code == 200, revoked.text
        assert resource_id not in proxy_app._share_store


def test_contribution_envelope_keeps_raw_delete_capability_client_side_and_replays(monkeypatch):
    raw_token = "run18-private-contribution-capability-" + "y" * 20
    resource_id = "2" * 32
    headers = _operation_headers(resource_id, raw_token, "contribop" + "b" * 23)
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_RATE_LIMIT_PER_HOUR", 1)

    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", headers=headers, json=_contribution("first"))
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["receiptId"] == resource_id
        assert "deleteToken" not in body
        entry = proxy_app._contrib_quarantine[resource_id]
        assert entry["deleteTokenHash"] == headers["X-AI-Management-Token-Hash"]
        assert raw_token not in json.dumps(entry)

        replay = client.post("/v1/contribute", headers=headers, json=_contribution("first"))
        assert replay.status_code == 200, replay.text
        assert replay.json()["receiptId"] == resource_id
        assert replay.json()["idempotentReplay"] is True
        assert "deleteToken" not in replay.json()
        assert len(proxy_app._contrib_quarantine) == 1

        conflict = client.post("/v1/contribute", headers=headers, json=_contribution("different"))
        assert conflict.status_code == 409

        wrong_capability = _operation_headers(resource_id, "different-contribution-capability-" + "q" * 32, headers["X-AI-Operation-Id"])
        wrong_capability["X-AI-Operation-Created-At"] = headers["X-AI-Operation-Created-At"]
        rejected = client.post("/v1/contribute", headers=wrong_capability, json=_contribution("first"))
        assert rejected.status_code == 409

        deleted = client.delete(
            f"/v1/contribute/{resource_id}",
            headers={"X-Contribution-Delete-Token": raw_token},
        )
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["status"] == "deleted"


def test_schema_v4_rejects_reviewed_limits_instead_of_silent_truncation():
    too_many_records = _contribution()
    too_many_records["records"] = [
        {"recordType": "qa", "query": f"q{i}", "answer": "a"} for i in range(101)
    ]
    too_many_messages = _contribution()
    too_many_messages["records"] = [{
        "recordType": "conversation",
        "messages": [{"role": "user", "content": f"q{i}"} for i in range(101)],
    }]
    too_long = _contribution()
    too_long["records"][0]["answer"] = "a" * 100_001

    with TestClient(proxy_app.app) as client:
        assert client.post("/v1/contribute", json=too_many_records).status_code == 422
        assert client.post("/v1/contribute", json=too_many_messages).status_code == 422
        long_response = client.post("/v1/contribute", json=too_long)
        assert long_response.status_code == 422
        assert "100000" in long_response.text
    assert not proxy_app._contrib_quarantine


def test_health_reports_memory_share_storage_as_non_durable_non_shared():
    with TestClient(proxy_app.app) as client:
        health = client.get("/health")
    assert health.status_code == 200
    share = health.json()["share"]
    assert share["backend"] == "memory"
    assert share["durable"] is False
    assert share["shared"] is False
    assert share["consistency_scope"] == "process_local"


def test_sqlite_share_store_survives_reconstruction_without_raw_public_id_key(tmp_path):
    db = tmp_path / "shares.sqlite3"
    share_id = "abcdef0123456789abcdef0123456789"
    edit_hash = hashlib.sha256(b"private-edit-token").hexdigest()
    entry = {
        "snapshot": _snapshot("sqlite"),
        "format": "json",
        "edit_hash": edit_hash,
        "bytes": 123,
        "expiresAt_ts": time.time() + 3600,
        "expiresAt": "2026-08-30T02:00:00Z",
        "transport_version": 2,
        "operation_payload_digest": "d" * 64,
        "operation_id": "op" + "z" * 30,
    }

    async def exercise() -> None:
        first = share_store_mod.SQLiteShareStore(str(db), max_entries=10, max_total_bytes=100_000)
        await first.initialize()
        await first.create(share_id, entry)
        await first.close()
        second = share_store_mod.SQLiteShareStore(str(db), max_entries=10, max_total_bytes=100_000)
        await second.initialize()
        restored = await second.get(share_id)
        await second.close()
        assert restored is not None
        assert restored["snapshot"]["records"][1]["text"] == "sqlite"
        assert second.manifest()["durable"] is True
        assert second.manifest()["shared"] is False

    asyncio.run(exercise())

    # Public capability is SHA-256 pseudonymized as the DB key. The raw locator
    # must not appear in SQLite key material or journals generated by this test.
    for candidate in [db, Path(str(db) + "-wal"), Path(str(db) + "-shm")]:
        if candidate.exists():
            assert share_id.encode("ascii") not in candidate.read_bytes()
    with sqlite3.connect(db) as conn:
        key = conn.execute("SELECT share_key FROM global_shares").fetchone()[0]
    assert key == hashlib.sha256(share_id.encode("utf-8")).hexdigest()
    assert key != share_id


def test_redis_share_get_uses_one_atomic_expiry_script():
    entry = {
        "snapshot": _snapshot("redis"),
        "bytes": 10,
        "expiresAt_ts": time.time() + 60,
        "edit_hash": "e" * 64,
    }

    class ProbeStore(share_store_mod.RedisShareStore):
        def __init__(self, result):
            super().__init__(
                "redis://example.invalid/0",
                key_prefix="run18",
                max_entries=10,
                max_total_bytes=1000,
                client=object(),
            )
            self.result = result
            self.calls = []

        async def _eval(self, script, keys, args):
            self.calls.append((script, keys, args))
            return self.result

    async def exercise() -> None:
        live = ProbeStore((1, json.dumps(entry)))
        restored = await live.get("3" * 32)
        assert restored["snapshot"]["records"][1]["text"] == "redis"
        assert len(live.calls) == 1
        assert live.calls[0][0] == share_store_mod._REDIS_GET

        expired = ProbeStore((0, "EXPIRED"))
        with pytest.raises(share_store_mod.ShareStoreError, match="EXPIRED"):
            await expired.get("4" * 32)
        assert len(expired.calls) == 1

    asyncio.run(exercise())
    script = share_store_mod._REDIS_GET
    assert "redis.call('GET',KEYS[3])" in script
    assert "redis.call('DEL',KEYS[3])" in script
    assert "redis.call('ZREM',KEYS[1],member)" in script
    assert "redis.call('DECRBY',KEYS[2],n)" in script
