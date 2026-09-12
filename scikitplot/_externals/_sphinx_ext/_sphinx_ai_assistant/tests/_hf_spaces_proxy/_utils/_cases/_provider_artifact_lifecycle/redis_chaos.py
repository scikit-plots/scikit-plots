# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Run 146 — Redis reconnect / cluster / live-Lua chaos boundary."""
from __future__ import annotations

import asyncio
import binascii
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import types
import time

import pytest

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import REPOSITORY_ROOT

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


def _request(*, key: str = "1" * 32, token: str = "2" * 64):
    raw = {
        "contract": PROVIDER_ARTIFACT_CONTRACT,
        "generator_id": "stub/generated-png",
        "kind": "image",
        "prompt": "run146 redis chaos fixture",
        "mime_type": "image/png",
        "options": {},
        "cancel_token": token,
        "dedupe_key": key,
    }
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


class _RetryClient:
    """Scripted client that drops the first response after accepting a command."""

    def __init__(self, fail_once_scripts=()):
        self.fail_once = set(fail_once_scripts)
        self.calls: list[tuple[str, tuple]] = []
        self.counts: dict[str, int] = {}

    async def ping(self):
        return True

    async def close(self):
        return None

    async def eval(self, script, numkeys, *payload):
        del numkeys
        args = tuple(payload[4:])
        self.calls.append((script, args))
        self.counts[script] = self.counts.get(script, 0) + 1
        if script in self.fail_once and self.counts[script] == 1:
            raise ConnectionError("simulated response loss after commit")
        if script == lifecycle_mod._REDIS_BEGIN:
            return [1, ""]
        if script == lifecycle_mod._REDIS_COMPLETE:
            lifecycle_id, now, expires, size, sha = args[:5]
            row = {
                "lifecycle_id": lifecycle_id,
                "status": "ready",
                "created_at": int(now),
                "expires_at": int(expires),
                "regeneration_of": "",
                "output_size": int(size),
                "output_sha256": sha,
            }
            return [1, json.dumps(row)]
        if script in {
            lifecycle_mod._REDIS_RESERVE,
            lifecycle_mod._REDIS_RELEASE,
            lifecycle_mod._REDIS_APPLY,
            lifecycle_mod._REDIS_MARK_DELIVERED,
            lifecycle_mod._REDIS_FAIL,
            lifecycle_mod._REDIS_CLEANUP,
        }:
            return [1, "OK"]
        if script == lifecycle_mod._REDIS_GET:
            return [0, "NOT_FOUND"]
        raise AssertionError("unexpected script")


def _registry(client, **kwargs):
    return RedisProviderArtifactLifecycleRegistry(
        "redis://127.0.0.1/0",
        client=client,
        require_tls=False,
        sweep_interval_seconds=60,
        **kwargs,
    )


def test_transport_retry_reuses_exact_begin_operation_identity() -> None:
    async def run():
        client = _RetryClient({lifecycle_mod._REDIS_BEGIN})
        registry = _registry(client, operation_attempts=2)
        request = _request()
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        calls = [args for script, args in client.calls if script == lifecycle_mod._REDIS_BEGIN]
        assert len(calls) == 2
        assert calls[0] == calls[1]
        assert calls[0][1] == lifecycle_id
    asyncio.run(run())


def test_transport_retry_covers_complete_reserve_and_apply_without_new_authority() -> None:
    async def run():
        client = _RetryClient({lifecycle_mod._REDIS_COMPLETE, lifecycle_mod._REDIS_RESERVE, lifecycle_mod._REDIS_APPLY})
        registry = _registry(client, operation_attempts=2)
        request = _request(key="3" * 32, token="4" * 64)
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        await registry.complete(lifecycle_id, _receipt(request, size=23, sha="b" * 64))
        reservation, ids = await registry.reserve_zip_correlations(((lifecycle_id, "b" * 64, 23),))
        await registry.mark_applied(ids, reservation)
        for script in (lifecycle_mod._REDIS_COMPLETE, lifecycle_mod._REDIS_RESERVE, lifecycle_mod._REDIS_APPLY):
            calls = [args for seen, args in client.calls if seen == script]
            assert len(calls) == 2 and calls[0] == calls[1]
    asyncio.run(run())


def test_lua_mutations_are_retry_idempotent_for_ambiguous_transport_outcomes() -> None:
    assert "prior.status == 'generating'" in lifecycle_mod._REDIS_BEGIN
    assert lifecycle_mod._REDIS_BEGIN.index("local retry_raw") < lifecycle_mod._REDIS_BEGIN.index("local existing=redis.call('HGET',KEYS[3],fingerprint)")
    assert "e.status == 'ready' or e.status == 'delivered' or e.status == 'reserved' or e.status == 'applied'" in lifecycle_mod._REDIS_COMPLETE
    assert "e.status == 'reserved' and e.reservation_id == reservation" in lifecycle_mod._REDIS_RESERVE
    assert "e.status == 'applied' and tostring(e.applied_reservation_id or '') == reservation" in lifecycle_mod._REDIS_APPLY


def test_cancellation_watcher_survives_brief_shared_authority_failover() -> None:
    async def run():
        registry = _registry(_RetryClient(), cancellation_poll_seconds=0.001, cancellation_watch_failure_limit=3)
        statuses = iter((
            {"status": "generating"},  # attach_task ownership check
            ProviderArtifactError("PROVIDER_ARTIFACT_REDIS_OPERATION_FAILED"),
            ProviderArtifactError("PROVIDER_ARTIFACT_REDIS_OPERATION_FAILED"),
            {"status": "generating"},
            {"status": "cancelled"},
        ))
        async def status(_):
            item = next(statuses, {"status": "cancelled"})
            if isinstance(item, Exception):
                raise item
            return item
        registry.public_status = status  # type: ignore[method-assign]
        task = asyncio.create_task(asyncio.sleep(60))
        await registry.attach_task("a" * 32, task)
        for _ in range(100):
            if task.cancelled():
                break
            await asyncio.sleep(0.002)
        assert task.cancelled()
        await registry.clear()
    asyncio.run(run())


def test_cancellation_watcher_fails_provider_spend_closed_after_bounded_authority_loss() -> None:
    async def run():
        registry = _registry(_RetryClient(), cancellation_poll_seconds=0.001, cancellation_watch_failure_limit=2)
        first = True
        async def status(_):
            nonlocal first
            if first:
                first = False
                return {"status": "generating"}
            raise ProviderArtifactError("PROVIDER_ARTIFACT_REDIS_OPERATION_FAILED")
        registry.public_status = status  # type: ignore[method-assign]
        task = asyncio.create_task(asyncio.sleep(60))
        await registry.attach_task("b" * 32, task)
        for _ in range(100):
            if task.cancelled():
                break
            await asyncio.sleep(0.002)
        assert task.cancelled()
        await registry.clear()
    asyncio.run(run())


def test_cluster_topology_is_explicit_and_database_zero_only() -> None:
    registry = RedisProviderArtifactLifecycleRegistry(
        "redis://cluster.example.test/0", client=_RetryClient(), cluster_mode=True
    )
    manifest = registry.manifest()
    assert manifest["topology"] == "cluster"
    assert manifest["shared"] is True and manifest["authoritative"] is True
    with pytest.raises(ProviderArtifactError, match="PROVIDER_ARTIFACT_REDIS_CLUSTER_DATABASE_INVALID"):
        RedisProviderArtifactLifecycleRegistry(
            "redis://cluster.example.test/2", client=_RetryClient(), cluster_mode=True
        )


def test_factory_threads_explicit_cluster_mode() -> None:
    registry = build_provider_artifact_lifecycle_registry(
        "redis",
        redis_url="redis://cluster.example.test/0",
        redis_client=_RetryClient(),
        redis_cluster_mode=True,
    )
    assert registry.manifest()["topology"] == "cluster"


def test_cluster_initialize_uses_rediscluster_client_not_standalone(monkeypatch) -> None:
    captured = {}

    class _ClusterClient(_RetryClient):
        async def aclose(self):
            return None

    class _RedisCluster:
        @classmethod
        def from_url(cls, url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = dict(kwargs)
            return _ClusterClient()

    redis_pkg = types.ModuleType("redis"); redis_pkg.__path__ = []
    asyncio_pkg = types.ModuleType("redis.asyncio"); asyncio_pkg.__path__ = []
    cluster_pkg = types.ModuleType("redis.asyncio.cluster"); cluster_pkg.RedisCluster = _RedisCluster
    monkeypatch.setitem(sys.modules, "redis", redis_pkg)
    monkeypatch.setitem(sys.modules, "redis.asyncio", asyncio_pkg)
    monkeypatch.setitem(sys.modules, "redis.asyncio.cluster", cluster_pkg)

    async def run():
        registry = RedisProviderArtifactLifecycleRegistry(
            "redis://cluster.example.test/0", cluster_mode=True, sweep_interval_seconds=60
        )
        await registry.initialize()
        assert captured["url"] == "redis://cluster.example.test/0"
        assert captured["kwargs"]["decode_responses"] is False
        assert registry.manifest()["topology"] == "cluster"
        await registry.close()
    asyncio.run(run())


def _cluster_slot(key: str) -> int:
    raw = key.encode()
    left = raw.find(b"{")
    if left >= 0:
        right = raw.find(b"}", left + 1)
        if right > left + 1:
            raw = raw[left + 1:right]
    return binascii.crc_hqx(raw, 0) % 16384


def test_all_lifecycle_keys_share_one_redis_cluster_slot() -> None:
    registry = RedisProviderArtifactLifecycleRegistry(
        "redis://cluster.example.test/0", client=_RetryClient(), cluster_mode=True
    )
    slots = {_cluster_slot(key) for key in registry._keys}
    assert len(slots) == 1
    assert all("{provider-artifact}" in key for key in registry._keys)


class _RespRedis:
    """Tiny test-only RESP2 client; opens a fresh socket for every command."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    async def ping(self):
        return await self._command("PING")

    async def close(self):
        return None

    async def eval(self, script, numkeys, *payload):
        return await self._command("EVAL", script, numkeys, *payload)

    async def _command(self, *parts):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            encoded = []
            for part in parts:
                value = part if isinstance(part, bytes) else str(part).encode()
                encoded.append(b"$" + str(len(value)).encode() + b"\r\n" + value + b"\r\n")
            writer.write(b"*" + str(len(encoded)).encode() + b"\r\n" + b"".join(encoded))
            await writer.drain()
            return await self._read(reader)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _read(self, reader):
        prefix = await reader.readexactly(1)
        if prefix == b"+":
            return (await reader.readline()).rstrip(b"\r\n")
        if prefix == b"-":
            raise RuntimeError((await reader.readline()).decode(errors="replace").strip())
        if prefix == b":":
            return int((await reader.readline()).strip())
        if prefix == b"$":
            size = int((await reader.readline()).strip())
            if size < 0:
                return None
            data = await reader.readexactly(size)
            await reader.readexactly(2)
            return data
        if prefix == b"*":
            count = int((await reader.readline()).strip())
            return [await self._read(reader) for _ in range(count)]
        raise RuntimeError("unsupported RESP response")


def _redis_server_path() -> str:
    path = os.environ.get("RUN146_REDIS_SERVER", "").strip() or shutil.which("redis-server") or ""
    if path:
        return path
    if os.environ.get("RUN146_REDIS_CHAOS_REQUIRED", "").strip().lower() in {"1", "true", "yes", "on"}:
        pytest.fail("RUN146_REDIS_CHAOS_REQUIRED=1 but redis-server is unavailable")
    pytest.skip("redis-server unavailable; set RUN146_REDIS_SERVER or RUN146_REDIS_CHAOS_REQUIRED=1 in Redis CI")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class _RedisProcess:
    def __init__(self, binary: str, port: int, directory: str):
        self.binary = binary
        self.port = port
        self.directory = directory
        self.proc: subprocess.Popen | None = None

    def start(self):
        self.proc = subprocess.Popen(
            [
                self.binary, "--bind", "127.0.0.1", "--port", str(self.port),
                "--protected-mode", "yes", "--save", "", "--appendonly", "no",
                "--dir", self.directory, "--loglevel", "warning",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("redis-server exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.03)
        raise RuntimeError("redis-server startup timed out")

    def stop(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait(timeout=3)
        self.proc = None


def test_live_redis_executes_actual_lua_and_single_winner_reservation() -> None:
    binary = _redis_server_path()
    with tempfile.TemporaryDirectory(prefix="run146-redis-") as directory:
        port = _free_port(); server = _RedisProcess(binary, port, directory); server.start()
        async def run():
            a = RedisProviderArtifactLifecycleRegistry("redis://127.0.0.1/0", client=_RespRedis("127.0.0.1", port), sweep_interval_seconds=60)
            b = RedisProviderArtifactLifecycleRegistry("redis://127.0.0.1/0", client=_RespRedis("127.0.0.1", port), sweep_interval_seconds=60)
            await a.initialize(); await b.initialize()
            request = _request(key="7" * 32, token="8" * 64)
            lifecycle_id = await a.begin(request, provider="stub", model="deterministic-png")
            await a.complete(lifecycle_id, _receipt(request, size=31, sha="c" * 64))
            await a.mark_delivered(lifecycle_id)
            results = await asyncio.gather(
                a.reserve_zip_correlations(((lifecycle_id, "c" * 64, 31),)),
                b.reserve_zip_correlations(((lifecycle_id, "c" * 64, 31),)),
                return_exceptions=True,
            )
            winners = [row for row in results if not isinstance(row, Exception)]
            losers = [row for row in results if isinstance(row, ProviderArtifactError)]
            assert len(winners) == 1 and len(losers) == 1
            reservation, ids = winners[0]
            await a.mark_applied(ids, reservation)
            # APPLY itself is retry-idempotent for the same reservation.
            await a.mark_applied(ids, reservation)
            status = await b.public_status(lifecycle_id)
            assert status is not None and status["status"] == "applied"
            await a.close(); await b.close()
        try:
            asyncio.run(run())
        finally:
            server.stop()


def test_live_redis_restart_invalidates_ephemeral_authority_and_recovers_new_work() -> None:
    binary = _redis_server_path()
    with tempfile.TemporaryDirectory(prefix="run146-redis-restart-") as directory:
        port = _free_port(); server = _RedisProcess(binary, port, directory); server.start()
        async def run():
            client = _RespRedis("127.0.0.1", port)
            registry = RedisProviderArtifactLifecycleRegistry(
                "redis://127.0.0.1/0", client=client, operation_attempts=2, sweep_interval_seconds=60
            )
            await registry.initialize()
            first = _request(key="9" * 32, token="a" * 64)
            first_id = await registry.begin(first, provider="stub", model="deterministic-png")
            await registry.complete(first_id, _receipt(first, size=37, sha="d" * 64))
            server.stop()
            with pytest.raises(ProviderArtifactError, match="PROVIDER_ARTIFACT_REDIS_OPERATION_FAILED"):
                await registry.public_status(first_id)
            server.start()
            # Ephemeral Redis loss fails old candidates closed; no authority is reconstructed.
            assert await registry.public_status(first_id) is None
            second = _request(key="b" * 32, token="c" * 64)
            second_id = await registry.begin(second, provider="stub", model="deterministic-png")
            assert second_id != first_id
            await registry.close()
        try:
            asyncio.run(run())
        finally:
            server.stop()


def _import_app_with_env(**updates):
    env = os.environ.copy()
    env.update({key: str(value) for key, value in updates.items()})
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)
    code = (
        "import json; "
        "from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app; "
        "print(json.dumps({'manifest': app._PROVIDER_ARTIFACT_LIFECYCLE.manifest(), "
        "'error': app._PROVIDER_ARTIFACT_LIFECYCLE_CONFIG_ERROR}, sort_keys=True))"
    )
    result = subprocess.run(
        [os.environ.get("PYTHON", "python"), "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1]), result


def test_app_cluster_topology_is_explicit_without_redis_authority_leakage() -> None:
    secret_url = "rediss://cluster-user:cluster-secret@redis-cluster.internal:6380/0"
    doc, result = _import_app_with_env(
        PROVIDER_ARTIFACT_LIFECYCLE_BACKEND="redis",
        PROVIDER_ARTIFACT_LIFECYCLE_REDIS_URL=secret_url,
        PROVIDER_ARTIFACT_LIFECYCLE_REDIS_TOPOLOGY="cluster",
        REDIS_REQUIRE_TLS="true",
    )
    assert doc["error"] == ""
    assert doc["manifest"]["backend"] == "redis"
    assert doc["manifest"]["topology"] == "cluster"
    rendered = result.stdout + result.stderr
    for forbidden in ("cluster-user", "cluster-secret", "redis-cluster.internal", "6380", "rediss://"):
        assert forbidden not in rendered


def test_app_cluster_topology_rejects_nonzero_database_fail_closed() -> None:
    secret_url = "rediss://cluster-user:cluster-secret@redis-cluster.internal:6380/2"
    doc, result = _import_app_with_env(
        PROVIDER_ARTIFACT_LIFECYCLE_BACKEND="redis",
        PROVIDER_ARTIFACT_LIFECYCLE_REDIS_URL=secret_url,
        PROVIDER_ARTIFACT_LIFECYCLE_REDIS_TOPOLOGY="cluster",
        REDIS_REQUIRE_TLS="true",
    )
    assert doc["error"] == "PROVIDER_ARTIFACT_REDIS_CLUSTER_DATABASE_INVALID"
    assert doc["manifest"]["backend"] == "memory"
    rendered = result.stdout + result.stderr
    for forbidden in ("cluster-user", "cluster-secret", "redis-cluster.internal", "6380", "rediss://"):
        assert forbidden not in rendered
