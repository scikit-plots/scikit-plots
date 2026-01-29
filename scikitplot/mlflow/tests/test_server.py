from __future__ import annotations

"""Tests for server command building and process spawning.

These tests keep subprocess usage fully mocked to avoid environment coupling.
"""

from types import SimpleNamespace

from scikitplot.mlflow import ServerConfig
from scikitplot.mlflow._server import build_server_args, spawn_server


def test_build_server_args_includes_common_flags() -> None:
    cfg = ServerConfig(
        host="127.0.0.1",
        port=5005,
        backend_store_uri="sqlite:///tmp.db",
        default_artifact_root="/tmp/artifacts",
        serve_artifacts=True,
        strict_cli_compat=False,
    )
    args = build_server_args(cfg)
    assert "--host" in args and "127.0.0.1" in args
    assert "--port" in args and "5005" in args
    assert "--backend-store-uri" in args
    assert "--default-artifact-root" in args
    assert "--serve-artifacts" in args


def test_spawn_server_calls_cli_compat(monkeypatch) -> None:
    """When strict_cli_compat=True, spawn_server must validate flags."""
    cfg = ServerConfig(host="127.0.0.1", port=5010, strict_cli_compat=True)

    import importlib
    s = importlib.import_module("scikitplot.mlflow._server")

    monkeypatch.setattr(s, "_is_port_free", lambda host, port: True)

    calls = {"checked": False}

    def _ensure(args, supported_flags, context):
        calls["checked"] = True

    monkeypatch.setattr(s, "ensure_flags_supported", _ensure)
    monkeypatch.setattr(
        s,
        "get_mlflow_server_cli_caps",
        lambda: SimpleNamespace(flags=frozenset({"--host", "--port"})),
    )

    class DummyPopen:
        def __init__(self, *a, **k):
            self.pid = 123
            self.returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True
            return None

        def wait(self, timeout=None):
            # Simulate a process that doesn't exit on terminate (forces kill path).
            raise s.subprocess.TimeoutExpired(cmd="x", timeout=timeout)

        def kill(self):
            self.killed = True
            return None

    dummy = DummyPopen()
    monkeypatch.setattr(s.subprocess, "Popen", lambda *a, **k: dummy)

    sv = spawn_server(cfg)
    assert calls["checked"] is True

    # terminate() should attempt graceful termination then kill on timeout
    sv.terminate()
    assert sv.process.terminated is True
    assert sv.process.killed is True
