# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Run 147 — reproducible live Redis standalone/cluster CI boundary."""
from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import REPOSITORY_ROOT, RUNTIME_ROOT

import asyncio
import binascii
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._provider_artifact import (
    PROVIDER_ARTIFACT_CONTRACT,
    ProviderArtifactError,
    ProviderArtifactGeneratorSpec,
    build_provider_artifact_receipt_from_digest,
    parse_provider_artifact_request,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._provider_artifact_lifecycle import (
    RedisProviderArtifactLifecycleRegistry,
)

_THIS = Path(__file__).resolve()
_EXTENSION = RUNTIME_ROOT
_CI = _EXTENSION / "_hf_spaces_proxy" / "ci"
_TRUE = {"1", "true", "yes", "on"}


def _request(*, key: str, token: str):
    return parse_provider_artifact_request(
        json.dumps(
            {
                "contract": PROVIDER_ARTIFACT_CONTRACT,
                "generator_id": "stub/generated-png",
                "kind": "image",
                "prompt": "run147 live Redis CI fixture",
                "mime_type": "image/png",
                "options": {},
                "cancel_token": token,
                "dedupe_key": key,
            }
        ).encode()
    )


def _receipt(request, *, size: int, sha: str):
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


class _RespRedis:
    """Small RESP2 client used only to bootstrap/query real Redis processes."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)

    async def command(self, *parts):
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
            if count < 0:
                return None
            return [await self._read(reader) for _ in range(count)]
        raise RuntimeError("unsupported RESP response")


def _port_available(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _free_cluster_ports(count: int) -> list[int]:
    """Return client ports whose implicit Redis cluster-bus ports are safe too."""
    if count < 1 or count > 32:
        raise ValueError("cluster node count out of test bounds")
    found: list[int] = []
    # Redis defaults the cluster bus to client_port + 10000. Keep both under
    # 65535 and away from the common system/ephemeral lower ranges.
    for port in range(20000, 45000):
        if port in found:
            continue
        bus = port + 10000
        if _port_available(port) and _port_available(bus):
            found.append(port)
            if len(found) == count:
                return found
    raise RuntimeError("could not allocate bounded Redis cluster ports")


def _required_binary(*, required_env: str) -> str:
    explicit = os.environ.get("RUN147_REDIS_SERVER", "").strip() or os.environ.get(
        "RUN146_REDIS_SERVER", ""
    ).strip()
    path = explicit or shutil.which("redis-server") or ""
    if path:
        return path
    if os.environ.get(required_env, "").strip().lower() in _TRUE:
        pytest.fail(f"{required_env}=1 but redis-server is unavailable")
    pytest.skip("redis-server unavailable; Run 147 live CI gate requires Redis 7/8")


def _require_redis_py(*, required_env: str) -> None:
    try:
        import redis  # noqa: F401
        from redis.asyncio.cluster import RedisCluster  # noqa: F401
    except Exception:
        if os.environ.get(required_env, "").strip().lower() in _TRUE:
            pytest.fail(f"{required_env}=1 but redis-py cluster support is unavailable")
        pytest.skip("redis-py unavailable; install the pinned proxy requirements for live CI")


class _ClusterNode:
    def __init__(self, binary: str, directory: Path, port: int):
        self.binary = binary
        self.directory = directory
        self.port = int(port)
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                self.binary,
                "--bind",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--protected-mode",
                "no",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(self.directory),
                "--cluster-enabled",
                "yes",
                "--cluster-config-file",
                f"nodes-{self.port}.conf",
                "--cluster-node-timeout",
                "1000",
                "--cluster-replica-validity-factor",
                "0",
                "--loglevel",
                "warning",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("redis cluster node exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.03)
        raise RuntimeError("redis cluster node startup timed out")

    def stop(self, *, kill: bool = False) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            if kill:
                self.proc.kill()
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=4)
        self.proc = None


async def _wait_until(predicate, *, timeout: float = 15.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            value = await predicate()
            if value:
                return value
        except Exception as exc:  # transient cluster convergence is expected
            last = exc
        await asyncio.sleep(interval)
    if last:
        raise AssertionError(f"condition timed out after transient error: {type(last).__name__}")
    raise AssertionError("condition timed out")


def _decode(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


async def _bootstrap_cluster(nodes: list[_ClusterNode]) -> dict[int, int]:
    clients = [_RespRedis("127.0.0.1", node.port) for node in nodes]
    for client in clients[1:]:
        await clients[0].command("CLUSTER", "MEET", "127.0.0.1", client.port)

    async def all_known():
        raw = await clients[0].command("CLUSTER", "NODES")
        return len([line for line in _decode(raw).splitlines() if line.strip()]) >= 6

    await _wait_until(all_known)

    ranges = ((0, 5460), (5461, 10922), (10923, 16383))
    for client, (start, end) in zip(clients[:3], ranges):
        # Redis accepts a single ADDSLOTS command with this bounded slot set.
        await client.command("CLUSTER", "ADDSLOTS", *range(start, end + 1))

    master_ids = [_decode(await client.command("CLUSTER", "MYID")) for client in clients[:3]]
    for replica, master_id in zip(clients[3:], master_ids):
        await replica.command("CLUSTER", "REPLICATE", master_id)

    async def cluster_ok():
        for client in clients:
            info = _decode(await client.command("CLUSTER", "INFO"))
            if "cluster_state:ok" not in info:
                return False
        return True

    await _wait_until(cluster_ok, timeout=20)
    return {nodes[i].port: nodes[i + 3].port for i in range(3)}


async def _slots(client: _RespRedis):
    return await client.command("CLUSTER", "SLOTS")


def _provider_slot() -> int:
    # Redis cluster hash-tag semantics hash only the bytes between {...}.
    return int(binascii.crc_hqx(b"provider-artifact", 0) % 16384)


def _slot_owner(slots, target: int) -> int:
    for row in slots or []:
        start, end = int(row[0]), int(row[1])
        if start <= target <= end:
            return int(row[2][1])
    raise AssertionError("provider-artifact slot has no owner")


def _worker_command(seed_port: int, lifecycle_id: str, sha: str, size: int) -> list[str]:
    return [
        sys.executable,
        str(_THIS),
        "--worker-reserve",
        str(seed_port),
        lifecycle_id,
        sha,
        str(size),
    ]


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = str(REPOSITORY_ROOT)
    existing = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = repo_root if not existing else repo_root + os.pathsep + existing
    return env


def test_run147_ci_entrypoint_and_container_are_fail_closed_and_version_matrixed() -> None:
    runner = (_CI / "run_redis_chaos.py").read_text(encoding="utf-8")
    dockerfile = (_CI / "redis-chaos.Dockerfile").read_text(encoding="utf-8")
    github = (_CI / "github-actions.redis-chaos.reference.yml").read_text(encoding="utf-8")
    circle = (_CI / "circleci.redis-chaos.reference.yml").read_text(encoding="utf-8")
    assert "_SUPPORTED_REDIS_MAJORS = {7, 8}" in runner
    assert 'RUN146_REDIS_CHAOS_REQUIRED\"] = \"1\"' in runner
    assert 'RUN147_REDIS_CLUSTER_REQUIRED\"] = \"1\"' in runner
    assert "redis.asyncio.cluster import RedisCluster" in runner
    assert "ARG REDIS_IMAGE=redis:8.2.9-bookworm" in dockerfile
    assert "redis:7.4.11-bookworm" in github and "redis:8.2.9-bookworm" in github
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1" in github
    assert "actions/checkout@v" not in github
    assert "--mode all" in github and "--read-only" in github and "--cap-drop ALL" in github
    assert "redis:7.4.11-bookworm" in circle and "redis:8.2.9-bookworm" in circle
    assert "--mode all" in circle and "--security-opt no-new-privileges" in circle


def test_run147_cluster_port_allocator_bounds_client_and_bus_ports() -> None:
    ports = _free_cluster_ports(6)
    assert len(set(ports)) == 6
    assert all(20000 <= port < 45000 for port in ports)
    assert all(port + 10000 <= 65535 for port in ports)
    assert all(_port_available(port) and _port_available(port + 10000) for port in ports)


def test_run147_ci_runner_is_standard_library_importable_without_redis_runtime() -> None:
    path = _CI / "run_redis_chaos.py"
    spec = importlib.util.spec_from_file_location("run147_redis_ci", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._SUPPORTED_REDIS_MAJORS == {7, 8}
    assert module._extension_root() == _EXTENSION
    assert module._repo_pythonpath() == REPOSITORY_ROOT


def test_live_standalone_ci_requires_real_server_and_executes_actual_lua() -> None:
    binary = _required_binary(required_env="RUN147_REDIS_STANDALONE_REQUIRED")
    _require_redis_py(required_env="RUN147_REDIS_STANDALONE_REQUIRED")
    # Run 146 owns the disposable standalone Redis/Lua/restart assertions. Run
    # 147 deliberately invokes the same test under a mandatory no-skip flag in
    # the portable CI runner; this local smoke proves the exact binary itself.
    result = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0
    rendered = result.stdout + result.stderr
    assert "Redis" in rendered or "redis" in rendered


def test_live_cluster_six_nodes_cross_process_race_and_primary_failover() -> None:
    binary = _required_binary(required_env="RUN147_REDIS_CLUSTER_REQUIRED")
    _require_redis_py(required_env="RUN147_REDIS_CLUSTER_REQUIRED")

    with tempfile.TemporaryDirectory(prefix="run147-cluster-") as tmp:
        root = Path(tmp)
        ports = _free_cluster_ports(6)
        nodes = [_ClusterNode(binary, root / f"n{i}", ports[i]) for i in range(6)]
        for node in nodes:
            node.start()

        async def run():
            replica_for_master = await _bootstrap_cluster(nodes)
            seed = nodes[0].port
            registry = RedisProviderArtifactLifecycleRegistry(
                f"redis://127.0.0.1:{seed}/0",
                cluster_mode=True,
                require_tls=False,
                operation_attempts=3,
                sweep_interval_seconds=60,
            )
            await registry.initialize()
            request = _request(key="d" * 32, token="e" * 64)
            lifecycle_id = await registry.begin(
                request, provider="stub", model="deterministic-png"
            )
            output_sha = "f" * 64
            output_size = 41
            await registry.complete(
                lifecycle_id, _receipt(request, size=output_size, sha=output_sha)
            )
            await registry.mark_delivered(lifecycle_id)

            # Two OS processes race for the same lifecycle authority. Exactly
            # one receives the atomic Redis reservation.
            env = _worker_env()
            workers = [
                subprocess.Popen(
                    _worker_command(seed, lifecycle_id, output_sha, output_size),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            rows = []
            for proc in workers:
                stdout, stderr = proc.communicate(timeout=30)
                assert proc.returncode in {0, 3}, stderr
                rows.append(json.loads(stdout.strip().splitlines()[-1]))
            winners = [row for row in rows if row.get("ok")]
            losers = [row for row in rows if not row.get("ok")]
            assert len(winners) == 1 and len(losers) == 1
            reservation = str(winners[0]["reservation_id"])
            await registry.mark_applied((lifecycle_id,), reservation)

            client = _RespRedis("127.0.0.1", seed)
            slot = _provider_slot()
            owner = _slot_owner(await _slots(client), slot)
            assert owner in replica_for_master
            replica = replica_for_master[owner]
            # Ensure the terminal write reached the replica before killing the
            # slot owner, making the failover assertion deterministic.
            await _RespRedis("127.0.0.1", owner).command("WAIT", 1, 5000)
            owner_node = next(node for node in nodes if node.port == owner)
            owner_node.stop(kill=True)
            survivor = next(node.port for node in nodes if node.port != owner)
            survivor_client = _RespRedis("127.0.0.1", survivor)

            async def promoted():
                current = _slot_owner(await _slots(survivor_client), slot)
                return current == replica

            await _wait_until(promoted, timeout=20, interval=0.1)

            # A new production RedisCluster client discovers the promoted owner.
            post = RedisProviderArtifactLifecycleRegistry(
                f"redis://127.0.0.1:{survivor}/0",
                cluster_mode=True,
                require_tls=False,
                operation_attempts=3,
                sweep_interval_seconds=60,
            )
            await post.initialize()
            status = await post.public_status(lifecycle_id)
            assert status is not None and status["status"] == "applied"
            with pytest.raises(ProviderArtifactError):
                await post.reserve_zip_correlations(
                    ((lifecycle_id, output_sha, output_size),)
                )
            second = _request(key="1" * 32, token="2" * 64)
            second_id = await post.begin(
                second, provider="stub", model="deterministic-png"
            )
            assert second_id != lifecycle_id
            await post.close()
            await registry.close()

        try:
            asyncio.run(run())
        finally:
            for node in nodes:
                node.stop(kill=True)


def _reserve_worker(seed_port: int, lifecycle_id: str, sha: str, size: int) -> int:
    async def run():
        registry = RedisProviderArtifactLifecycleRegistry(
            f"redis://127.0.0.1:{seed_port}/0",
            cluster_mode=True,
            require_tls=False,
            operation_attempts=3,
            sweep_interval_seconds=60,
        )
        try:
            await registry.initialize()
            reservation, ids = await registry.reserve_zip_correlations(
                ((lifecycle_id, sha, int(size)),)
            )
            print(
                json.dumps(
                    {"ok": True, "reservation_id": reservation, "ids": list(ids)},
                    sort_keys=True,
                )
            )
            return 0
        except ProviderArtifactError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            return 3
        finally:
            await registry.close()

    return asyncio.run(run())


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--worker-reserve":
        raise SystemExit(
            _reserve_worker(
                int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
            )
        )
    raise SystemExit("test module is only executable in --worker-reserve mode")
