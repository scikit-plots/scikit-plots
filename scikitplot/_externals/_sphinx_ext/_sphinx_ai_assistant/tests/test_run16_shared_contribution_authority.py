from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))

from _utils import _contribution_ledger as ledger_mod  # noqa: E402
import app as proxy_app  # noqa: E402
from _utils._storage import StorageWriteError  # noqa: E402
from _utils._contribution_ledger import (  # noqa: E402
    ContributionLedgerError,
    RedisContributionLedger,
    build_contribution_ledger,
)




def _assert_run16_source_contract(ledger_src: str, app_src: str, storage_src: str) -> None:
    assert 'class RedisContributionLedger' in ledger_src
    assert 'receipt_id_externalized": "hmac_sha256"' in ledger_src
    assert "if state == 'promoting' and old_lease <= now then\n  entry.state='promotion_uncertain'" in ledger_src
    route_start = app_src.index('@app.post("/v1/contribute")')
    route_end = app_src.index('raw = await _read_limited_body', route_start)
    shared_route = app_src[route_start:route_end]
    assert 'if CONTRIBUTION_REQUIRE_SHARED and not (' in shared_route
    assert 'bool(ledger_manifest.get("shared"))' in shared_route
    assert 'bool(ledger_manifest.get("authoritative"))' in shared_route
    assert 'status_code=503' in shared_route
    assert 'detail="Shared contribution lifecycle authority is required."' in shared_route
    assert 'mark_promotion_uncertain' in app_src
    assert 'bool(getattr(exc, "transient", False))' in app_src
    transport_pos = storage_src.index('f"{target.provider.upper()}_TRANSPORT"')
    transport_block = storage_src[max(0, transport_pos - 160): transport_pos + 200]
    assert 'raise StorageWriteError(' in transport_block
    assert 'transient=True' in transport_block

def _entry(receipt_id: str = "receipt-a") -> dict:
    return {
        "receiptId": receipt_id,
        "state": "quarantined",
        "records": [{"query": "q", "answer": "a", "_dedup_key": f"{receipt_id}:0"}],
        "bytes": 64,
        "deleteTokenHash": "d" * 64,
        "expiresAt": 9_999_999_999.0,
        "receivedAt": 1000.0,
        "dedupKeys": [f"{receipt_id}:0"],
        "storage": {},
        "withdrawalStorage": {},
        "currentViewRemoval": {},
        "lastError": "",
    }


class FakeRedisContributionClient:
    """
    Small semantic fake for the Run 16 Redis scripts.

    The fake intentionally models only the lifecycle transitions exercised here;
    source assertions below pin the actual Redis Lua/index contract separately.
    Two ledger instances sharing this object therefore model independent proxy
    replicas talking to one consistency domain.
    """

    def __init__(self) -> None:
        self.receipts: dict[str, dict] = {}
        self.pending: dict[str, float] = {}
        self.pending_bytes: dict[str, int] = {}
        self.all_receipts: dict[str, float] = {}
        self.keys_seen: list[str] = []

    async def ping(self):
        return True

    @staticmethod
    def _out(ok: bool, value) -> list:
        if isinstance(value, dict):
            value = json.dumps(value, separators=(",", ":"))
        return [1 if ok else 0, value.encode() if isinstance(value, str) else value]

    async def eval(self, script, numkeys, *items):
        keys = [str(v) for v in items[:numkeys]]
        args = list(items[numkeys:])
        self.keys_seen.extend(keys)
        receipt_key = keys[-1]

        if script == ledger_mod._REDIS_CREATE_LUA:
            now, member, payload, expires_at, live_until, max_receipts, max_pending, max_bytes, byte_count, _ttl = args
            if receipt_key in self.receipts:
                return self._out(False, "DUPLICATE_RECEIPT")
            active_all = [m for m, until in self.all_receipts.items() if float(until) > float(now)]
            active_pending = [m for m, until in self.pending.items() if float(until) > float(now)]
            if len(active_all) >= int(max_receipts):
                return self._out(False, "RECEIPT_CAPACITY")
            if len(active_pending) >= int(max_pending):
                return self._out(False, "PENDING_CAPACITY")
            if sum(self.pending_bytes.get(m, 0) for m in active_pending) + int(byte_count) > int(max_bytes):
                return self._out(False, "PENDING_BYTE_CAPACITY")
            entry = json.loads(payload)
            self.receipts[receipt_key] = entry
            self.all_receipts[str(member)] = float(live_until)
            self.pending[str(member)] = float(expires_at)
            self.pending_bytes[str(member)] = int(byte_count)
            return self._out(True, entry)

        if script == ledger_mod._REDIS_GET_LUA:
            entry = self.receipts.get(receipt_key)
            return self._out(True, entry or "")

        if script == ledger_mod._REDIS_BEGIN_PROMOTION_LUA:
            now, member, claim_hash, lease_until, _retention = args
            entry = self.receipts.get(receipt_key)
            if entry is None:
                return self._out(False, "NOT_FOUND")
            state = entry.get("state")
            old_lease = float(entry.get("operationLeaseUntil") or 0)
            if state == "promoting" and old_lease > float(now):
                return self._out(False, "PROMOTION_IN_PROGRESS")
            if state == "promoting" and old_lease <= float(now):
                entry.update(
                    state="promotion_uncertain",
                    lastError="CLAIM_EXPIRED_RECONCILIATION_REQUIRED",
                    operationClaimHash="",
                    operationLeaseUntil=0,
                    updatedAt=float(now),
                )
                return self._out(False, "RECONCILIATION_REQUIRED")
            if state != "quarantined":
                return self._out(False, "NOT_PENDING")
            entry.update(
                state="promoting",
                operationClaimHash=str(claim_hash),
                operationLeaseUntil=float(lease_until),
                updatedAt=float(now),
            )
            self.pending[str(member)] = max(float(entry["expiresAt"]), float(lease_until))
            return self._out(True, entry)

        if script == ledger_mod._REDIS_MARK_PROMOTION_UNCERTAIN_LUA:
            now, member, claim_hash, code, immortal = args
            entry = self.receipts.get(receipt_key)
            if entry is None:
                return self._out(False, "NOT_FOUND")
            if entry.get("state") != "promoting":
                return self._out(False, "PROMOTION_STATE")
            if entry.get("operationClaimHash") != str(claim_hash):
                return self._out(False, "STALE_CLAIM")
            entry.update(
                state="promotion_uncertain",
                lastError=str(code),
                operationClaimHash="",
                operationLeaseUntil=0,
                updatedAt=float(now),
            )
            self.pending[str(member)] = float(immortal)
            self.all_receipts[str(member)] = float(immortal)
            return self._out(True, entry)

        if script == ledger_mod._REDIS_MARK_PROMOTED_LUA:
            now, member, claim_hash, storage_json, immortal = args
            entry = self.receipts.get(receipt_key)
            if entry is None:
                return self._out(False, "NOT_FOUND")
            if entry.get("state") != "promoting":
                return self._out(False, "PROMOTION_STATE")
            if entry.get("operationClaimHash") != str(claim_hash):
                return self._out(False, "STALE_CLAIM")
            entry.update(
                state="eligible",
                records=[],
                bytes=0,
                promotedAt=float(now),
                storage=json.loads(storage_json),
                operationClaimHash="",
                operationLeaseUntil=0,
                updatedAt=float(now),
            )
            self.pending.pop(str(member), None)
            self.pending_bytes.pop(str(member), None)
            self.all_receipts[str(member)] = float(immortal)
            return self._out(True, entry)

        if script == ledger_mod._REDIS_BEGIN_WITHDRAWAL_LUA:
            now, member, claim_hash, lease_until, immortal = args
            entry = self.receipts.get(receipt_key)
            if entry is None:
                return self._out(False, "NOT_FOUND")
            state = entry.get("state")
            old_lease = float(entry.get("operationLeaseUntil") or 0)
            if state == "withdrawn":
                return self._out(True, entry)
            if state == "withdrawing" and old_lease > float(now):
                return self._out(False, "WITHDRAWAL_IN_PROGRESS")
            if state == "withdrawing" and old_lease <= float(now):
                state = "withdrawal_uncertain"
            if state not in {"eligible", "promotion_uncertain", "withdrawal_uncertain"}:
                return self._out(False, "NOT_ELIGIBLE")
            entry.update(
                operationPriorState=state,
                state="withdrawing",
                operationClaimHash=str(claim_hash),
                operationLeaseUntil=float(lease_until),
                updatedAt=float(now),
            )
            self.all_receipts[str(member)] = float(immortal)
            return self._out(True, entry)

        if script == ledger_mod._REDIS_MARK_WITHDRAWN_LUA:
            now, member, claim_hash, withdrawal_json, removal_json, retention = args
            entry = self.receipts.get(receipt_key)
            if entry is None:
                return self._out(False, "NOT_FOUND")
            if entry.get("state") != "withdrawing":
                return self._out(False, "WITHDRAWAL_STATE")
            if entry.get("operationClaimHash") != str(claim_hash):
                return self._out(False, "STALE_CLAIM")
            entry.update(
                state="withdrawn",
                records=[],
                bytes=0,
                withdrawnAt=float(now),
                withdrawalStorage=json.loads(withdrawal_json),
                currentViewRemoval=json.loads(removal_json),
                operationClaimHash="",
                operationLeaseUntil=0,
                updatedAt=float(now),
            )
            self.all_receipts[str(member)] = float(now) + float(retention)
            return self._out(True, entry)

        if script in {ledger_mod._REDIS_PROMOTION_FAILED_LUA, ledger_mod._REDIS_WITHDRAWAL_FAILED_LUA}:
            return self._out(True, self.receipts.get(receipt_key) or "")

        if script == ledger_mod._REDIS_DELETE_PENDING_LUA:
            now, member, retention = args
            entry = self.receipts.get(receipt_key)
            if entry is None:
                return self._out(False, "NOT_FOUND")
            state = entry.get("state")
            lease_until = float(entry.get("operationLeaseUntil") or 0)
            if state == "promoting" and lease_until > float(now):
                return self._out(False, "BUSY")
            if state == "promoting" and lease_until <= float(now):
                entry.update(
                    state="promotion_uncertain",
                    lastError="CLAIM_EXPIRED_RECONCILIATION_REQUIRED",
                    operationClaimHash="",
                    operationLeaseUntil=0,
                    updatedAt=float(now),
                )
                return self._out(False, "RECONCILIATION_REQUIRED")
            if state != "quarantined":
                return self._out(False, "NOT_PENDING")
            entry.update(state="deleted", records=[], bytes=0, deletedAt=float(now), updatedAt=float(now))
            self.pending.pop(str(member), None)
            self.pending_bytes.pop(str(member), None)
            self.all_receipts[str(member)] = float(now) + float(retention)
            return self._out(True, entry)

        raise AssertionError("unexpected Redis script")


def _redis_ledger(client: FakeRedisContributionClient) -> RedisContributionLedger:
    return RedisContributionLedger(
        "redis://shared.example:6379/0",
        key_secret="k" * 32,
        key_prefix="test-assistant",
        max_pending_entries=8,
        max_pending_bytes=1_000_000,
        max_receipts=32,
        operation_lease_seconds=30,
        client=client,
    )


def test_two_replicas_share_one_atomic_promotion_claim():
    async def run():
        shared = FakeRedisContributionClient()
        a, b = _redis_ledger(shared), _redis_ledger(shared)
        await a.initialize(); await b.initialize()
        await a.create(_entry())

        results = await asyncio.gather(
            a.begin_promotion("receipt-a"),
            b.begin_promotion("receipt-a"),
            return_exceptions=True,
        )
        winners = [r for r in results if isinstance(r, dict)]
        losers = [r for r in results if isinstance(r, ContributionLedgerError)]
        assert len(winners) == 1
        assert len(losers) == 1
        assert losers[0].code == "PROMOTION_IN_PROGRESS"
        assert winners[0]["operationClaim"]

    asyncio.run(run())


def test_expired_promotion_lease_fails_closed_to_reconciliation(monkeypatch):
    async def run():
        shared = FakeRedisContributionClient()
        a, b = _redis_ledger(shared), _redis_ledger(shared)
        clock = {"now": 1_000.0}
        monkeypatch.setattr(ledger_mod, "_now", lambda: clock["now"])
        entry = _entry(); entry["expiresAt"] = 10_000.0
        await a.create(entry)
        first = await a.begin_promotion("receipt-a")
        clock["now"] += 31.0
        with pytest.raises(ContributionLedgerError) as blocked:
            await b.begin_promotion("receipt-a")
        assert blocked.value.code == "RECONCILIATION_REQUIRED"
        current = await b.get("receipt-a")
        assert current is not None
        assert current["state"] == "promotion_uncertain"

        with pytest.raises(ContributionLedgerError) as stale:
            await a.mark_promoted("receipt-a", storage={"recordId": "old"}, claim_token=first["operationClaim"])
        assert stale.value.code in {"PROMOTION_STATE", "STALE_CLAIM"}

        # The participant can always resolve the ambiguous promotion toward the
        # privacy-safe direction by writing a withdrawal tombstone.
        withdrawal = await b.begin_withdrawal("receipt-a")
        done = await b.mark_withdrawn(
            "receipt-a",
            withdrawal_storage={"recordId": "withdrawal"},
            current_view_removal={"primary": "unknown-path"},
            claim_token=withdrawal["operationClaim"],
        )
        assert done["state"] == "withdrawn"
        assert done["records"] == []

    asyncio.run(run())


def test_direct_pending_delete_cannot_bypass_expired_promotion_uncertainty(monkeypatch):
    async def run():
        shared = FakeRedisContributionClient()
        ledger = _redis_ledger(shared)
        clock = {"now": 1_500.0}
        monkeypatch.setattr(ledger_mod, "_now", lambda: clock["now"])
        entry = _entry(); entry["expiresAt"] = 20_000.0
        await ledger.create(entry)
        await ledger.begin_promotion("receipt-a")
        clock["now"] += 31.0
        with pytest.raises(ContributionLedgerError) as blocked:
            await ledger.delete_pending("receipt-a")
        assert blocked.value.code == "RECONCILIATION_REQUIRED"
        current = await ledger.get("receipt-a")
        assert current is not None
        assert current["state"] == "promotion_uncertain"
        assert current["records"]

    asyncio.run(run())


def test_withdrawal_claim_is_shared_and_stale_claim_cannot_finalize(monkeypatch):
    async def run():
        shared = FakeRedisContributionClient()
        a, b = _redis_ledger(shared), _redis_ledger(shared)
        clock = {"now": 2_000.0}
        monkeypatch.setattr(ledger_mod, "_now", lambda: clock["now"])
        entry = _entry(); entry["expiresAt"] = 20_000.0
        await a.create(entry)
        promotion = await a.begin_promotion("receipt-a")
        await a.mark_promoted("receipt-a", storage={"recordId": "r"}, claim_token=promotion["operationClaim"])
        first = await a.begin_withdrawal("receipt-a")
        with pytest.raises(ContributionLedgerError) as busy:
            await b.begin_withdrawal("receipt-a")
        assert busy.value.code == "WITHDRAWAL_IN_PROGRESS"
        clock["now"] += 31.0
        # Withdrawal is monotonic: after an expired claim, a new claimant may
        # continue toward withdrawn, never back toward training eligibility.
        second = await b.begin_withdrawal("receipt-a")
        with pytest.raises(ContributionLedgerError) as stale:
            await a.mark_withdrawn(
                "receipt-a", withdrawal_storage={}, current_view_removal={}, claim_token=first["operationClaim"]
            )
        assert stale.value.code == "STALE_CLAIM"
        done = await b.mark_withdrawn(
            "receipt-a",
            withdrawal_storage={"recordId": "w"},
            current_view_removal={"primary": "removed-current-view"},
            claim_token=second["operationClaim"],
        )
        assert done["state"] == "withdrawn"

    asyncio.run(run())


def test_redis_keyspace_never_externalizes_raw_receipt_identifier():
    async def run():
        shared = FakeRedisContributionClient()
        ledger = _redis_ledger(shared)
        await ledger.create(_entry("super-secret-receipt-id"))
        assert shared.keys_seen
        assert all("super-secret-receipt-id" not in key for key in shared.keys_seen)
        expected = hashlib.sha256  # keep the assertion explicit about a one-way external key shape
        assert ledger.manifest()["receipt_id_externalized"] == "hmac_sha256"
        assert expected is not None

    asyncio.run(run())


def test_redis_manifest_is_shared_authority_but_not_unverified_durability():
    ledger = _redis_ledger(FakeRedisContributionClient())
    manifest = ledger.manifest()
    assert manifest["shared"] is True
    assert manifest["authoritative"] is True
    assert manifest["durable"] is False
    assert manifest["durability"] == "shared_transactional_external"


def test_redis_backend_requires_separate_long_key_secret():
    with pytest.raises(ContributionLedgerError) as exc:
        RedisContributionLedger(
            "redis://example", key_secret="short", key_prefix="x",
            max_pending_entries=8, max_pending_bytes=1000, max_receipts=32,
        )
    assert exc.value.code == "REDIS_KEY_SECRET_TOO_SHORT"


def test_builder_exposes_redis_as_explicit_backend_only():
    ledger = build_contribution_ledger(
        "redis",
        sqlite_path="/tmp/unused.sqlite3",
        redis_url="redis://example",
        redis_key_secret="z" * 32,
        max_pending_entries=8,
        max_pending_bytes=1000,
        max_receipts=32,
    )
    assert isinstance(ledger, RedisContributionLedger)
    assert ledger.shared is True


def test_redis_scripts_pin_atomic_index_and_claim_contract():
    source = (PROXY / "_utils" / "_contribution_ledger.py").read_text(encoding="utf-8")
    assert "{contribution}" in source
    assert "operationClaimHash" in ledger_mod._REDIS_BEGIN_PROMOTION_LUA
    assert "STALE_CLAIM" in ledger_mod._REDIS_MARK_PROMOTED_LUA
    assert "promotion_uncertain" in ledger_mod._REDIS_MARK_PROMOTION_UNCERTAIN_LUA
    assert "RECONCILIATION_REQUIRED" in ledger_mod._REDIS_BEGIN_PROMOTION_LUA
    assert "STALE_CLAIM" in ledger_mod._REDIS_MARK_WITHDRAWN_LUA
    assert "ZCARD" in ledger_mod._REDIS_CREATE_LUA
    assert "HVALS" in ledger_mod._REDIS_CREATE_LUA
    assert "CONTRIBUTION_REQUIRE_SHARED" in (PROXY / "app.py").read_text(encoding="utf-8")


def _contribution_payload() -> dict:
    return {
        "schemaVersion": 3,
        "consentFlag": True,
        "consentVersion": "1.0.0",
        "page": "https://example.test/docs/page",
        "model": {"id": "claimed", "provider": "claimed", "model": "claimed/model"},
        "records": [{
            "answerIndex": 0,
            "query": "question",
            "answer": "answer",
            "ratingValue": 1,
            "ratingLabel": "helpful",
            "ratingTitle": "Helpful",
            "ratingMode": "quick",
            "message": "",
            "ts": 123,
        }],
    }


def test_shared_required_mode_fails_closed_on_local_ledger(monkeypatch):
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REQUIRE_SHARED", True)
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REQUIRE_DURABLE", False)
    monkeypatch.setattr(proxy_app, "_CONTRIBUTION_LEDGER_READY", True)
    monkeypatch.setattr(proxy_app, "_CONTRIBUTION_LEDGER_CONFIG_ERROR", "")
    with TestClient(proxy_app.app) as client:
        response = client.post("/v1/contribute", json={})
    assert response.status_code == 503
    assert "Shared contribution lifecycle authority is required" in response.text


def test_transient_promotion_write_becomes_uncertain_and_user_can_withdraw(monkeypatch):
    # The default imported app uses the memory ledger; this test pins the generic
    # lifecycle semantics independently of the Redis backend implementation.
    ledger = ledger_mod.MemoryContributionLedger(
        max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32
    )
    monkeypatch.setattr(proxy_app, "_CONTRIBUTION_LEDGER", ledger)
    monkeypatch.setattr(proxy_app, "_CONTRIBUTION_LEDGER_READY", True)
    monkeypatch.setattr(proxy_app, "_CONTRIBUTION_LEDGER_CONFIG_ERROR", "")
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REQUIRE_SHARED", False)
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REQUIRE_DURABLE", False)
    monkeypatch.setattr(proxy_app, "CONTRIBUTION_REVIEW_TOKEN", "review-secret")

    calls = {"count": 0}

    class FakeStorage:
        primary = object()
        def set_client(self, client): pass
        async def initialize(self): pass
        async def close(self): pass
        def manifest(self): return {"targets": []}
        def primary_ready(self): return True
        async def remove_current_view(self, paths, *, record_id=None, commit_message=""):
            return {"primary": "unknown-path"}

    async def persist(*, kind, content, commit_message, path_timestamp=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise StorageWriteError("GITHUB_WRITE", transient=True)
        return SimpleNamespace(
            record_id="withdrawal-record",
            primary="primary",
            mirrors={},
            paths={"primary": "contributions/withdrawal.jsonl"},
        )

    monkeypatch.setattr(proxy_app, "_STORAGE", FakeStorage())
    monkeypatch.setattr(proxy_app, "_persist_storage_record", persist)

    with TestClient(proxy_app.app) as client:
        created = client.post("/v1/contribute", json=_contribution_payload())
        assert created.status_code == 200, created.text
        rid = created.json()["receiptId"]
        delete_token = created.json()["deleteToken"]

        promoted = client.post(
            f"/v1/contribute/{rid}/promote",
            headers={"Authorization": "Bearer review-secret"},
        )
        assert promoted.status_code == 503
        assert "outcome is uncertain" in promoted.text

        status = client.get(
            f"/v1/contribute/{rid}",
            headers={"X-Contribution-Delete-Token": delete_token},
        )
        assert status.status_code == 200
        assert status.json()["status"] == "promotion_uncertain"
        assert status.json()["reconciliationRequired"] is True
        assert status.json()["trainingEligible"] is False

        retry = client.post(
            f"/v1/contribute/{rid}/promote",
            headers={"Authorization": "Bearer review-secret"},
        )
        assert retry.status_code == 409

        withdrawn = client.delete(
            f"/v1/contribute/{rid}",
            headers={"X-Contribution-Delete-Token": delete_token},
        )
        assert withdrawn.status_code == 202, withdrawn.text
        assert withdrawn.json()["status"] == "withdrawn"
        assert withdrawn.json()["trainingWithdrawn"] is True


def test_sqlite_promotion_uncertainty_survives_restart_and_resolves_to_withdrawal(tmp_path):
    db = tmp_path / "uncertain.sqlite3"

    async def run():
        ledger = ledger_mod.SQLiteContributionLedger(
            str(db), max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32
        )
        await ledger.initialize()
        await ledger.create(_entry("sqlite-uncertain"))
        await ledger.begin_promotion("sqlite-uncertain")
        uncertain = await ledger.mark_promotion_uncertain("sqlite-uncertain", "GITHUB_TRANSPORT")
        assert uncertain["state"] == "promotion_uncertain"
        assert uncertain["records"]
        await ledger.close()

        reopened = ledger_mod.SQLiteContributionLedger(
            str(db), max_pending_entries=8, max_pending_bytes=1_000_000, max_receipts=32
        )
        await reopened.initialize()
        current = await reopened.get("sqlite-uncertain")
        assert current is not None
        assert current["state"] == "promotion_uncertain"
        with pytest.raises(ContributionLedgerError) as blocked:
            await reopened.begin_promotion("sqlite-uncertain")
        assert blocked.value.code == "NOT_PENDING"

        await reopened.begin_withdrawal("sqlite-uncertain")
        await reopened.withdrawal_failed("sqlite-uncertain", "TEMPORARY")
        still_uncertain = await reopened.get("sqlite-uncertain")
        assert still_uncertain is not None
        assert still_uncertain["state"] == "promotion_uncertain"

        await reopened.begin_withdrawal("sqlite-uncertain")
        done = await reopened.mark_withdrawn(
            "sqlite-uncertain", withdrawal_storage={}, current_view_removal={}
        )
        assert done["state"] == "withdrawn"
        assert done["records"] == []
        await reopened.close()

    asyncio.run(run())


def test_run16_positive_control_mutant_shared_requirement_removed_is_caught():
    ledger_src = (PROXY / "_utils" / "_contribution_ledger.py").read_text(encoding="utf-8")
    app_src = (PROXY / "app.py").read_text(encoding="utf-8")
    storage_src = (PROXY / "_utils" / "_storage.py").read_text(encoding="utf-8")
    route_start = app_src.index('@app.post("/v1/contribute")')
    anchor = 'if CONTRIBUTION_REQUIRE_SHARED and not ('
    gate_start = app_src.index(anchor, route_start)
    mutated = app_src[:gate_start] + app_src[gate_start:].replace(anchor, 'if False and not (', 1)
    with pytest.raises(AssertionError):
        _assert_run16_source_contract(ledger_src, mutated, storage_src)


def test_run16_positive_control_mutant_receipt_hmac_removed_is_caught():
    ledger_src = (PROXY / "_utils" / "_contribution_ledger.py").read_text(encoding="utf-8")
    app_src = (PROXY / "app.py").read_text(encoding="utf-8")
    storage_src = (PROXY / "_utils" / "_storage.py").read_text(encoding="utf-8")
    anchor = '"receipt_id_externalized": "hmac_sha256"'
    assert ledger_src.count(anchor) == 1
    mutated = ledger_src.replace(anchor, '"receipt_id_externalized": "raw"', 1)
    with pytest.raises(AssertionError):
        _assert_run16_source_contract(mutated, app_src, storage_src)


def test_run16_positive_control_mutant_promotion_auto_reclaim_is_caught():
    ledger_src = (PROXY / "_utils" / "_contribution_ledger.py").read_text(encoding="utf-8")
    app_src = (PROXY / "app.py").read_text(encoding="utf-8")
    storage_src = (PROXY / "_utils" / "_storage.py").read_text(encoding="utf-8")
    anchor = "if state == 'promoting' and old_lease <= now then\n  entry.state='promotion_uncertain'"
    assert ledger_src.count(anchor) == 1
    mutated = ledger_src.replace(anchor, anchor.replace("entry.state='promotion_uncertain'", "entry.state='quarantined'"), 1)
    with pytest.raises(AssertionError):
        _assert_run16_source_contract(mutated, app_src, storage_src)


def test_run16_positive_control_mutant_transport_ambiguity_downgraded_is_caught():
    ledger_src = (PROXY / "_utils" / "_contribution_ledger.py").read_text(encoding="utf-8")
    app_src = (PROXY / "app.py").read_text(encoding="utf-8")
    storage_src = (PROXY / "_utils" / "_storage.py").read_text(encoding="utf-8")
    marker = 'f"{target.provider.upper()}_TRANSPORT"'
    transport_pos = storage_src.index(marker)
    transient_pos = storage_src.index('transient=True', transport_pos)
    assert transient_pos - transport_pos < 160
    mutated = storage_src[:transient_pos] + storage_src[transient_pos:].replace(
        'transient=True', 'transient=False', 1
    )
    with pytest.raises(AssertionError):
        _assert_run16_source_contract(ledger_src, app_src, mutated)
