# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Run 145 — shared atomic provider-artifact lifecycle authority."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import REPOSITORY_ROOT

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _provider_artifact_lifecycle as lifecycle_mod
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._provider_artifact import (
    PROVIDER_ARTIFACT_CONTRACT,
    ProviderArtifactError,
    ProviderArtifactGeneratorSpec,
    build_provider_artifact_receipt_from_digest,
    parse_provider_artifact_request,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._provider_artifact_lifecycle import (
    RedisProviderArtifactLifecycleRegistry,
    build_provider_artifact_lifecycle_registry,
)


def _request(*, key: str = "1" * 32, token: str = "2" * 64, regenerate_of: str = ""):
    raw = {
        "contract": PROVIDER_ARTIFACT_CONTRACT,
        "generator_id": "stub/generated-png",
        "kind": "image",
        "prompt": "shared lifecycle fixture",
        "mime_type": "image/png",
        "options": {},
        "cancel_token": token,
        "dedupe_key": key,
    }
    if regenerate_of:
        raw["regenerate_of"] = regenerate_of
    return parse_provider_artifact_request(json.dumps(raw).encode())


def _receipt(request, *, size: int = 17, sha: str = "a" * 64):
    spec = ProviderArtifactGeneratorSpec(
        id="stub/generated-png",
        provider="stub",
        model="deterministic-png",
        kind="image",
        mime_types=("image/png",),
        max_output_bytes=1024 * 1024,
        option_schema={},
        diagnostic=True,
    )
    return build_provider_artifact_receipt_from_digest(
        request=request, spec=spec, output_size=size, output_sha256=sha
    )


@dataclass
class _SharedRedisState:
    records: dict[str, dict] = field(default_factory=dict)
    expiry: dict[str, int] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    cancels: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _FakeRedis:
    """Tiny semantic Redis oracle shared by independent registry instances."""

    def __init__(self, state: _SharedRedisState) -> None:
        self.state = state
        self.closed = False

    async def ping(self):
        return True

    async def close(self):
        self.closed = True

    async def delete(self, *keys):
        async with self.state.lock:
            self.state.records.clear(); self.state.expiry.clear()
            self.state.fingerprints.clear(); self.state.cancels.clear()
        return len(keys)

    def _cleanup(self, now: int) -> None:
        for lifecycle_id in [row for row, deadline in self.state.expiry.items() if deadline <= now]:
            record = self.state.records.pop(lifecycle_id, None)
            self.state.expiry.pop(lifecycle_id, None)
            if record:
                if self.state.fingerprints.get(record.get("fingerprint")) == lifecycle_id:
                    self.state.fingerprints.pop(record.get("fingerprint"), None)
                if self.state.cancels.get(record.get("cancel_token_sha256")) == lifecycle_id:
                    self.state.cancels.pop(record.get("cancel_token_sha256"), None)

    async def eval(self, script, numkeys, *payload):
        del numkeys
        args = list(payload[4:])
        async with self.state.lock:
            if script == lifecycle_mod._REDIS_CLEANUP:
                before = len(self.state.records)
                self._cleanup(int(args[0]))
                return [1, str(before - len(self.state.records))]

            if script == lifecycle_mod._REDIS_BEGIN:
                now = int(args[0]); lifecycle_id = args[1]; fingerprint = args[2]; cancel_hash = args[3]
                generator, provider, model, kind, mime, prompt_sha = args[4:10]
                regenerate_of = args[10]; expires = int(args[11]); max_records = int(args[12]); dup_window = int(args[13])
                self._cleanup(now)
                parent_id = ""
                if regenerate_of:
                    parent = self.state.records.get(regenerate_of)
                    if not parent or int(parent["expires_at"]) <= now or parent["status"] not in {"ready", "delivered", "applied"} or (parent["generator_id"], parent["provider"], parent["model"], parent["kind"], parent["mime_type"]) != (generator, provider, model, kind, mime):
                        return [0, "REGENERATION_INVALID"]
                    parent_id = regenerate_of
                else:
                    existing_id = self.state.fingerprints.get(fingerprint)
                    existing = self.state.records.get(existing_id) if existing_id else None
                    if existing and existing["status"] in {"generating", "ready", "delivered", "reserved", "applied"} and now - int(existing["created_at"]) <= dup_window:
                        return [0, "DUPLICATE"]
                    if existing_id:
                        self.state.fingerprints.pop(fingerprint, None)
                if cancel_hash and cancel_hash in self.state.cancels:
                    return [0, "REQUEST_INVALID"]
                if len(self.state.records) >= max_records:
                    return [0, "BUSY"]
                if lifecycle_id in self.state.records:
                    prior = self.state.records[lifecycle_id]
                    if (
                        prior["status"] == "generating"
                        and prior["fingerprint"] == fingerprint
                        and prior["cancel_token_sha256"] == cancel_hash
                        and (prior["generator_id"], prior["provider"], prior["model"], prior["kind"], prior["mime_type"], prior["prompt_sha256"], prior["regeneration_of"])
                        == (generator, provider, model, kind, mime, prompt_sha, regenerate_of)
                        and int(prior["created_at"]) == now
                        and int(prior["expires_at"]) == expires
                    ):
                        return [1, prior["regeneration_of"]]
                    return [0, "COLLISION"]
                row = {
                    "lifecycle_id": lifecycle_id, "fingerprint": fingerprint, "cancel_token_sha256": cancel_hash,
                    "generator_id": generator, "provider": provider, "model": model, "kind": kind,
                    "mime_type": mime, "prompt_sha256": prompt_sha, "regeneration_of": parent_id,
                    "status": "generating", "created_at": now, "expires_at": expires,
                    "output_size": 0, "output_sha256": "", "reservation_id": "",
                    "reserved_from_status": "", "reservation_deadline": 0,
                    "applied_reservation_id": "",
                }
                self.state.records[lifecycle_id] = row; self.state.expiry[lifecycle_id] = expires
                self.state.fingerprints[fingerprint] = lifecycle_id
                if cancel_hash:
                    self.state.cancels[cancel_hash] = lifecycle_id
                return [1, parent_id]

            if script == lifecycle_mod._REDIS_COMPLETE:
                lifecycle_id, now, expires, size, sha, generator, provider, model, kind, mime, prompt_sha = args
                row = self.state.records.get(lifecycle_id)
                if not row:
                    return [0, "CANCELLED"]
                if (row["generator_id"], row["provider"], row["model"], row["kind"], row["mime_type"], row["prompt_sha256"]) != (generator, provider, model, kind, mime, prompt_sha):
                    return [0, "RECEIPT_INVALID"]
                if row["status"] != "generating":
                    if row["status"] in {"ready", "delivered", "reserved", "applied"} and int(row["output_size"]) == int(size) and row["output_sha256"] == sha:
                        return [1, json.dumps(row)]
                    return [0, "CANCELLED"]
                row.update(status="ready", output_size=int(size), output_sha256=sha, expires_at=int(expires), reservation_id="", reserved_from_status="", reservation_deadline=0, applied_reservation_id="")
                self.state.expiry[lifecycle_id] = int(expires)
                return [1, json.dumps(row)]

            if script == lifecycle_mod._REDIS_FAIL:
                lifecycle_id, status, expires = args; row = self.state.records.get(lifecycle_id)
                if row:
                    row.update(status=status, expires_at=int(expires), reservation_id="", reserved_from_status="", reservation_deadline=0)
                    self.state.expiry[lifecycle_id] = int(expires)
                return [1, "OK"]

            if script == lifecycle_mod._REDIS_CANCEL:
                token_hash, now, expires = args; del now
                lifecycle_id = self.state.cancels.get(token_hash); row = self.state.records.get(lifecycle_id) if lifecycle_id else None
                if not row:
                    return [0, "INVALID"]
                if row["status"] == "generating":
                    row["status"] = "cancelled"; row["expires_at"] = int(expires); self.state.expiry[lifecycle_id] = int(expires)
                return [1, json.dumps(row)]

            if script == lifecycle_mod._REDIS_MARK_DELIVERED:
                row = self.state.records.get(args[0])
                if row and row["status"] == "ready":
                    row["status"] = "delivered"
                return [1, "OK"]

            if script == lifecycle_mod._REDIS_GET:
                lifecycle_id, now = args[0], int(args[1]); self._cleanup(now)
                row = self.state.records.get(lifecycle_id)
                if not row:
                    return [0, "NOT_FOUND"]
                return [1, json.dumps(row)]

            if script == lifecycle_mod._REDIS_RESERVE:
                now, reservation, deadline = int(args[0]), args[1], int(args[2]); correlations = json.loads(args[3]); self._cleanup(now)
                ids = [str(c[0]) for c in correlations]
                if len(set(ids)) != len(ids):
                    return [0, "INVALID"]
                rows = []
                for lifecycle_id, sha, size in correlations:
                    row = self.state.records.get(lifecycle_id)
                    if not row or int(row["expires_at"]) <= now or row["output_sha256"] != sha or int(row["output_size"]) != int(size):
                        return [0, "INVALID"]
                    if not (row["status"] in {"ready", "delivered"} or (row["status"] == "reserved" and row["reservation_id"] == reservation)):
                        return [0, "INVALID"]
                    rows.append(row)
                for lifecycle_id, row in zip(ids, rows):
                    if row["status"] != "reserved":
                        row["reserved_from_status"] = row["status"]; row["status"] = "reserved"; row["reservation_id"] = reservation; row["reservation_deadline"] = deadline; self.state.expiry[lifecycle_id] = deadline
                return [1, "OK"]

            if script == lifecycle_mod._REDIS_RELEASE:
                now, reservation, ids = int(args[0]), args[1], json.loads(args[2])
                for lifecycle_id in ids:
                    row = self.state.records.get(lifecycle_id)
                    if not row or row["status"] != "reserved" or row["reservation_id"] != reservation:
                        continue
                    if int(row["expires_at"]) <= now:
                        self.state.records.pop(lifecycle_id, None); self.state.expiry.pop(lifecycle_id, None)
                    else:
                        row["status"] = row["reserved_from_status"] if row["reserved_from_status"] in {"ready", "delivered"} else "delivered"; row["reservation_id"] = ""; row["reserved_from_status"] = ""; row["reservation_deadline"] = 0; self.state.expiry[lifecycle_id] = int(row["expires_at"])
                return [1, "OK"]

            if script == lifecycle_mod._REDIS_APPLY:
                reservation, ids = args[0], json.loads(args[1]); rows = []
                for lifecycle_id in ids:
                    row = self.state.records.get(lifecycle_id)
                    if not row or not ((row["status"] == "reserved" and row["reservation_id"] == reservation) or (row["status"] == "applied" and row.get("applied_reservation_id") == reservation)):
                        return [0, "INVALID"]
                    rows.append(row)
                for lifecycle_id, row in zip(ids, rows):
                    if row["status"] != "applied":
                        row["status"] = "applied"; row["applied_reservation_id"] = reservation; row["reservation_id"] = ""; row["reserved_from_status"] = ""; row["reservation_deadline"] = 0; self.state.expiry[lifecycle_id] = int(row["expires_at"])
                return [1, "OK"]

            raise AssertionError("unexpected lifecycle Lua script")


def _redis_registry(state: _SharedRedisState, **kwargs):
    return RedisProviderArtifactLifecycleRegistry(
        "rediss://user:secret@example.test/1",
        key_prefix="test",
        client=_FakeRedis(state),
        require_tls=True,
        cancellation_poll_seconds=0.01,
        **kwargs,
    )


def test_factory_keeps_memory_compatibility_and_redis_is_shared_metadata_only() -> None:
    memory = build_provider_artifact_lifecycle_registry("memory")
    assert memory.manifest() == {
        "backend": "memory", "shared": False, "authoritative": True,
        "consistency_scope": "process_local", "persistence": "ephemeral_process_memory",
    }
    shared = _redis_registry(_SharedRedisState())
    manifest = shared.manifest()
    assert manifest["backend"] == "redis" and manifest["shared"] is True and manifest["authoritative"] is True
    assert manifest["persistence"] == "ephemeral_shared_metadata"
    rendered = json.dumps(manifest)
    assert "example.test" not in rendered and "secret" not in rendered


def test_two_workers_share_duplicate_and_regeneration_authority() -> None:
    async def run():
        state = _SharedRedisState(); worker_a = _redis_registry(state); worker_b = _redis_registry(state)
        await worker_a.initialize(); await worker_b.initialize()
        request = _request(token="3" * 64)
        first = await worker_a.begin(request, provider="stub", model="deterministic-png")
        try:
            await worker_b.begin(_request(token="4" * 64), provider="stub", model="deterministic-png")
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_DUPLICATE"
        else:
            raise AssertionError("cross-worker duplicate escaped shared authority")
        ready = await worker_a.complete(first, _receipt(request))
        child = await worker_b.begin(_request(token="5" * 64, regenerate_of=ready.lifecycle_id), provider="stub", model="deterministic-png")
        assert child != first
        assert state.records[child]["regeneration_of"] == first
        await worker_a.close(); await worker_b.close()
    asyncio.run(run())


def test_cross_worker_cancel_cancels_generation_owner_task() -> None:
    async def run():
        state = _SharedRedisState(); owner = _redis_registry(state); remote = _redis_registry(state)
        await owner.initialize(); await remote.initialize()
        request = _request(key="6" * 32, token="6" * 64)
        lifecycle_id = await owner.begin(request, provider="stub", model="deterministic-png")
        task = asyncio.create_task(asyncio.sleep(60)); await owner.attach_task(lifecycle_id, task)
        status = await remote.cancel(request.cancel_token)
        assert status["status"] == "cancelled"
        for _ in range(50):
            if task.cancelled():
                break
            await asyncio.sleep(0.01)
        assert task.cancelled(), "owner worker did not observe shared cancellation"
        await owner.close(); await remote.close()
    asyncio.run(run())



def test_cross_worker_cancel_before_task_attachment_fails_closed() -> None:
    async def run():
        state = _SharedRedisState(); owner = _redis_registry(state); remote = _redis_registry(state)
        await owner.initialize(); await remote.initialize()
        request = _request(key="c" * 32, token="c" * 64)
        lifecycle_id = await owner.begin(request, provider="stub", model="deterministic-png")
        status = await remote.cancel(request.cancel_token)
        assert status["status"] == "cancelled"
        task = asyncio.create_task(asyncio.sleep(60))
        try:
            await owner.attach_task(lifecycle_id, task)
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_CANCELLED"
        else:
            raise AssertionError("cancelled shared lifecycle accepted a late owner task")
        await asyncio.sleep(0)
        assert task.cancelled()
        await owner.close(); await remote.close()
    asyncio.run(run())

def test_two_workers_have_one_atomic_zip_reservation_winner_and_terminal_apply() -> None:
    async def run():
        state = _SharedRedisState(); a = _redis_registry(state); b = _redis_registry(state)
        await a.initialize(); await b.initialize()
        request = _request(key="7" * 32, token="7" * 64)
        lifecycle_id = await a.begin(request, provider="stub", model="deterministic-png")
        await a.complete(lifecycle_id, _receipt(request, size=23, sha="7" * 64)); await a.mark_delivered(lifecycle_id)
        correlation = ((lifecycle_id, "7" * 64, 23),)
        async def attempt(registry):
            try:
                return ("ok",) + await registry.reserve_zip_correlations(correlation)
            except ProviderArtifactError as exc:
                return ("err", exc.code, ())
        outcomes = await asyncio.gather(attempt(a), attempt(b))
        winners = [row for row in outcomes if row[0] == "ok"]; losers = [row for row in outcomes if row[0] == "err"]
        assert len(winners) == 1 and len(losers) == 1
        assert losers[0][1] == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"
        reservation, ids = winners[0][1], winners[0][2]
        await b.mark_applied(ids, reservation)
        try:
            await a.reserve_zip_correlations(correlation)
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"
        else:
            raise AssertionError("applied candidate replayed across workers")
        await a.close(); await b.close()
    asyncio.run(run())


def test_multi_candidate_reservation_is_all_or_nothing_across_workers() -> None:
    async def run():
        state = _SharedRedisState(); a = _redis_registry(state); b = _redis_registry(state)
        await a.initialize(); await b.initialize()
        req_a = _request(key="8" * 32, token="8" * 64); req_b = _request(key="9" * 32, token="9" * 64)
        id_a = await a.begin(req_a, provider="stub", model="deterministic-png"); id_b = await b.begin(req_b, provider="stub", model="deterministic-png")
        await a.complete(id_a, _receipt(req_a, size=31, sha="8" * 64)); await b.complete(id_b, _receipt(req_b, size=37, sha="9" * 64))
        await a.mark_delivered(id_a); await b.mark_delivered(id_b)
        try:
            await a.reserve_zip_correlations(((id_a, "8" * 64, 31), (id_b, "f" * 64, 37)))
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"
        else:
            raise AssertionError("partial cross-worker reservation was committed")
        assert state.records[id_a]["status"] == state.records[id_b]["status"] == "delivered"
        reservation, ids = await b.reserve_zip_correlations(((id_a, "8" * 64, 31), (id_b, "9" * 64, 37)))
        assert set(ids) == {id_a, id_b}; assert {state.records[x]["reservation_id"] for x in ids} == {reservation}
        await a.release_zip_reservation(ids, reservation)
        assert {state.records[x]["status"] for x in ids} == {"delivered"}
        await a.close(); await b.close()
    asyncio.run(run())



def test_expiring_older_same_fingerprint_cannot_delete_newer_dedupe_index(monkeypatch) -> None:
    clock = [1_000]
    monkeypatch.setattr(lifecycle_mod.time, "time", lambda: clock[0])

    async def run():
        state = _SharedRedisState(); registry = _redis_registry(state, candidate_ttl_seconds=30, duplicate_window_seconds=2)
        await registry.initialize()
        first_request = _request(key="d" * 32, token="d" * 64)
        first = await registry.begin(first_request, provider="stub", model="deterministic-png")
        await registry.complete(first, _receipt(first_request, size=11, sha="d" * 64))
        fingerprint = state.records[first]["fingerprint"]
        clock[0] = 1_003
        second_request = _request(key="d" * 32, token="e" * 64)
        second = await registry.begin(second_request, provider="stub", model="deterministic-png")
        assert second != first and state.fingerprints[fingerprint] == second
        # Expire only the older record, then trigger global cleanup through a
        # different generation. The newer fingerprint index must survive.
        clock[0] = 1_031
        await registry.begin(_request(key="f" * 32, token="f" * 64), provider="stub", model="deterministic-png")
        assert first not in state.records
        assert second in state.records
        assert state.fingerprints[fingerprint] == second
        assert "HGET',KEYS[3]" in lifecycle_mod._REDIS_BEGIN
        await registry.close()
    asyncio.run(run())


def test_shared_expiry_sweeper_removes_idle_expired_metadata(monkeypatch) -> None:
    clock = [2_000]
    monkeypatch.setattr(lifecycle_mod.time, "time", lambda: clock[0])

    async def run():
        state = _SharedRedisState()
        registry = RedisProviderArtifactLifecycleRegistry(
            "rediss://user:secret@example.test/1",
            key_prefix="test",
            client=_FakeRedis(state),
            require_tls=True,
            candidate_ttl_seconds=30,
            sweep_interval_seconds=0.01,
        )
        await registry.initialize()
        request = _request(key="3" * 32, token="3" * 64)
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        await registry.complete(lifecycle_id, _receipt(request, size=47, sha="3" * 64))
        assert lifecycle_id in state.records
        clock[0] = 2_031
        for _ in range(50):
            if lifecycle_id not in state.records:
                break
            await asyncio.sleep(0.01)
        assert lifecycle_id not in state.records
        assert not state.fingerprints and not state.cancels
        await registry.close()
    asyncio.run(run())

def test_redis_close_does_not_delete_shared_records() -> None:
    async def run():
        state = _SharedRedisState(); a = _redis_registry(state); b = _redis_registry(state)
        await a.initialize(); await b.initialize()
        request = _request(key="a" * 32, token="a" * 64)
        lifecycle_id = await a.begin(request, provider="stub", model="deterministic-png")
        await a.complete(lifecycle_id, _receipt(request, size=41, sha="a" * 64))
        await a.close()
        status = await b.public_status(lifecycle_id)
        assert status is not None and status["status"] == "ready"
        await b.close()
    asyncio.run(run())



def test_shared_clear_is_process_local_and_never_erases_redis_authority() -> None:
    async def run():
        state = _SharedRedisState(); a = _redis_registry(state); b = _redis_registry(state)
        await a.initialize(); await b.initialize()
        request = _request(key="0" * 32, token="0" * 64)
        lifecycle_id = await a.begin(request, provider="stub", model="deterministic-png")
        await a.complete(lifecycle_id, _receipt(request, size=43, sha="0" * 64))
        await a.clear()
        status = await b.public_status(lifecycle_id)
        assert status is not None and status["status"] == "ready"
        await a.close(); await b.close()
    asyncio.run(run())


def test_graceful_worker_shutdown_cancels_only_its_owned_generation() -> None:
    async def run():
        state = _SharedRedisState(); a = _redis_registry(state); b = _redis_registry(state)
        await a.initialize(); await b.initialize()
        req_a = _request(key="1" * 32, token="1" * 64)
        req_b = _request(key="2" * 32, token="2" * 64)
        id_a = await a.begin(req_a, provider="stub", model="deterministic-png")
        id_b = await b.begin(req_b, provider="stub", model="deterministic-png")
        task_a = asyncio.create_task(asyncio.sleep(60)); task_b = asyncio.create_task(asyncio.sleep(60))
        await a.attach_task(id_a, task_a); await b.attach_task(id_b, task_b)
        await a.close()
        assert state.records[id_a]["status"] == "cancelled"
        assert state.records[id_b]["status"] == "generating"
        assert task_a.cancelled() and not task_b.cancelled()
        await b.close()
        assert task_b.cancelled()
    asyncio.run(run())

def test_redis_transport_requires_verified_tls_when_requested() -> None:
    try:
        RedisProviderArtifactLifecycleRegistry("redis://example.test/0", require_tls=True, client=_FakeRedis(_SharedRedisState()))
    except ProviderArtifactError as exc:
        assert exc.code == "PROVIDER_ARTIFACT_REDIS_TLS_REQUIRED"
    else:
        raise AssertionError("plaintext Redis escaped strict lifecycle transport policy")


def test_health_exposes_only_coarse_lifecycle_authority(monkeypatch) -> None:
    app._provider_artifact_rl.clear(); monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        lifecycle = client.get("/health").json()["capabilities"]["provider_artifact_output"]["lifecycle_authority"]
    assert lifecycle == {"backend": "memory", "shared": False, "authoritative": True, "ready": True}
    rendered = json.dumps(lifecycle)
    assert "url" not in rendered and "host" not in rendered and "token" not in rendered and "prefix" not in rendered



def test_health_suppresses_generators_when_shared_lifecycle_is_required_but_unavailable(monkeypatch) -> None:
    spec = ProviderArtifactGeneratorSpec(
        id="test/public-png",
        provider="test",
        model="public-png",
        kind="image",
        mime_types=("image/png",),
        max_output_bytes=1024,
        option_schema={},
        diagnostic=False,
    )
    app._provider_artifact_rl.clear(); monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        monkeypatch.setattr(app._PROVIDER_ARTIFACT_OUTPUT_REGISTRY, "public_specs", lambda: (spec,))
        assert client.get("/health").json()["capabilities"]["provider_artifact_output"]["generators"]
        monkeypatch.setattr(app, "PROVIDER_ARTIFACT_LIFECYCLE_REQUIRE_SHARED", True)
        caps = client.get("/health").json()["capabilities"]["provider_artifact_output"]
        assert caps["lifecycle_authority"]["backend"] == "memory"
        assert caps["lifecycle_authority"]["ready"] is False
        assert caps["generators"] == []

def test_provider_output_fails_closed_when_lifecycle_authority_not_ready(monkeypatch) -> None:
    app._provider_artifact_rl.clear(); monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        monkeypatch.setattr(app, "_PROVIDER_ARTIFACT_LIFECYCLE_READY", False)
        response = client.post("/v1/artifacts/provider-output", content=b"{}", headers={"content-type": "application/json"})
    assert response.status_code == 503
    assert response.json()["code"] == "PROVIDER_ARTIFACT_LIFECYCLE_UNAVAILABLE"


def test_redis_records_remain_metadata_only() -> None:
    async def run():
        state = _SharedRedisState(); registry = _redis_registry(state); await registry.initialize()
        request = _request(key="b" * 32, token="b" * 64)
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        raw = json.dumps(state.records[lifecycle_id], sort_keys=True)
        assert "shared lifecycle fixture" not in raw
        assert request.cancel_token not in raw
        assert request.prompt_sha256 in raw
        assert hashlib.sha256(request.cancel_token.encode()).hexdigest() in raw
        assert "path" not in raw and "bytes" not in raw and "secret" not in raw
        await registry.close()
    asyncio.run(run())



def _import_app_with_lifecycle_env(**updates):
    env = os.environ.copy()
    env.update({key: str(value) for key, value in updates.items()})
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)
    code = (
        "import json; "
        "from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app; "
        "print(json.dumps({'manifest': app._PROVIDER_ARTIFACT_LIFECYCLE.manifest(), "
        "'error': app._PROVIDER_ARTIFACT_LIFECYCLE_CONFIG_ERROR}, sort_keys=True))"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1]), result


def test_app_environment_selects_shared_redis_lifecycle_without_echoing_authority() -> None:
    secret_url = "rediss://runtime-user:super-secret@cache.internal.example:6380/4"
    doc, result = _import_app_with_lifecycle_env(
        PROVIDER_ARTIFACT_LIFECYCLE_BACKEND="redis",
        PROVIDER_ARTIFACT_LIFECYCLE_REDIS_URL=secret_url,
        REDIS_REQUIRE_TLS="true",
    )
    assert doc["error"] == ""
    assert doc["manifest"]["backend"] == "redis"
    assert doc["manifest"]["shared"] is True
    rendered = result.stdout + result.stderr
    for forbidden in ("runtime-user", "super-secret", "cache.internal.example", "6380", "rediss://"):
        assert forbidden not in rendered


def test_app_environment_rejects_plaintext_redis_under_tls_policy_without_echoing_url() -> None:
    secret_url = "redis://runtime-user:super-secret@cache.internal.example:6379/4"
    doc, result = _import_app_with_lifecycle_env(
        PROVIDER_ARTIFACT_LIFECYCLE_BACKEND="redis",
        PROVIDER_ARTIFACT_LIFECYCLE_REDIS_URL=secret_url,
        REDIS_REQUIRE_TLS="true",
    )
    assert doc["manifest"]["backend"] == "memory"
    assert doc["error"] == "PROVIDER_ARTIFACT_REDIS_TLS_REQUIRED"
    rendered = result.stdout + result.stderr
    for forbidden in ("runtime-user", "super-secret", "cache.internal.example", "6379", "redis://"):
        assert forbidden not in rendered

def test_lua_scripts_use_one_hash_slot_and_do_not_embed_prompt_or_provider_secret_fields() -> None:
    scripts = [
        lifecycle_mod._REDIS_BEGIN, lifecycle_mod._REDIS_COMPLETE, lifecycle_mod._REDIS_FAIL,
        lifecycle_mod._REDIS_CANCEL, lifecycle_mod._REDIS_MARK_DELIVERED, lifecycle_mod._REDIS_GET,
        lifecycle_mod._REDIS_RESERVE, lifecycle_mod._REDIS_RELEASE, lifecycle_mod._REDIS_APPLY,
    ]
    rendered = "\n".join(scripts)
    assert "prompt_text" not in rendered and "artifact_bytes" not in rendered and "provider_token" not in rendered
    registry = _redis_registry(_SharedRedisState())
    assert all("{provider-artifact}" in key for key in registry._keys)
