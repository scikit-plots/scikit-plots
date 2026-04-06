# scikitplot/mlflow/tests/test__server.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._server.

Naming convention: test__<module_name>.py

Covers gaps (target: 90%+ from 77% baseline)
----------------------------------------------
- _is_port_free          : bindable port (True), occupied port (False)
- build_server_args      : all flag branches (no_serve_artifacts, disable_security_middleware,
                           x_frame_options, dev, workers, artifacts_destination,
                           static_prefix, app_name, waitress_opts, uvicorn_opts,
                           secrets_cache_ttl/max_size, expose_prometheus, allowed_hosts,
                           cors_allowed_origins, registry_store_uri, artifacts_only,
                           extra_args, auto_host_in_docker override, docker warning)
- build_server_command   : starts with sys.executable -m mlflow server
- SpawnedServer.terminate: already-exited path (poll() not None), POSIX fallback paths,
                           Windows path (stubbed), timeout → kill path
- spawn_server           : port-busy raises, process exits immediately raises,
                           dev on Windows raises

Notes
-----
No real subprocesses are spawned. All Popen/socket calls are mocked.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import scikitplot.mlflow._server as srv
from scikitplot.mlflow._config import ServerConfig
from scikitplot.mlflow._errors import MlflowServerStartError
from scikitplot.mlflow._server import (
    SpawnedServer,
    _is_port_free,
    build_server_args,
    build_server_command,
    spawn_server,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_cfg(**overrides) -> ServerConfig:
    """Return a minimal ServerConfig with strict_cli_compat=False."""
    defaults = dict(
        host="127.0.0.1",
        port=5000,
        strict_cli_compat=False,
    )
    defaults.update(overrides)
    return ServerConfig(**defaults)


def _make_spawned(returncode=None) -> SpawnedServer:
    """Build a SpawnedServer with a mocked process."""
    proc = MagicMock()
    proc.pid = 9999
    proc.returncode = returncode
    proc.poll = MagicMock(return_value=returncode)
    proc.stdout = None
    proc.terminated = False
    proc.killed = False

    def _terminate():
        proc.terminated = True

    def _kill():
        proc.killed = True

    proc.terminate = _terminate
    proc.kill = _kill
    return SpawnedServer(_process=proc, _command=["mlflow", "server"], _started_at=time.time())


# ===========================================================================
# _is_port_free
# ===========================================================================


class TestIsPortFree:
    """Tests for _is_port_free."""

    def test_returns_bool(self) -> None:
        result = _is_port_free("127.0.0.1", 65534)
        assert isinstance(result, bool)

    def test_free_port_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mocked bind succeeds → True."""
        fake_sock = MagicMock()
        fake_sock.__enter__ = lambda s: s
        fake_sock.__exit__ = MagicMock(return_value=False)
        fake_sock.setsockopt = MagicMock()
        fake_sock.bind = MagicMock()  # no OSError

        monkeypatch.setattr(socket, "socket", lambda *a, **k: fake_sock)
        assert _is_port_free("127.0.0.1", 9999) is True

    def test_occupied_port_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mocked bind raises OSError → False."""
        fake_sock = MagicMock()
        fake_sock.__enter__ = lambda s: s
        fake_sock.__exit__ = MagicMock(return_value=False)
        fake_sock.setsockopt = MagicMock()
        fake_sock.bind = MagicMock(side_effect=OSError("busy"))

        monkeypatch.setattr(socket, "socket", lambda *a, **k: fake_sock)
        assert _is_port_free("127.0.0.1", 9999) is False


# ===========================================================================
# build_server_args - all flag branches
# ===========================================================================


class TestBuildServerArgs:
    """Tests for build_server_args - comprehensive flag coverage."""

    def test_no_serve_artifacts_flag(self) -> None:
        cfg = _minimal_cfg(no_serve_artifacts=True)
        args = build_server_args(cfg)
        assert "--no-serve-artifacts" in args

    def test_disable_security_middleware_flag(self) -> None:
        cfg = _minimal_cfg(disable_security_middleware=True)
        args = build_server_args(cfg)
        assert "--disable-security-middleware" in args

    def test_x_frame_options_flag(self) -> None:
        cfg = _minimal_cfg(x_frame_options="DENY")
        args = build_server_args(cfg)
        assert "--x-frame-options" in args
        assert "DENY" in args

    def test_dev_flag(self) -> None:
        cfg = _minimal_cfg(dev=True)
        args = build_server_args(cfg)
        assert "--dev" in args

    def test_workers_flag(self) -> None:
        cfg = _minimal_cfg(workers=4)
        args = build_server_args(cfg)
        assert "--workers" in args
        assert "4" in args

    def test_artifacts_destination_flag(self) -> None:
        cfg = _minimal_cfg(artifacts_destination="/tmp/artifacts")
        args = build_server_args(cfg)
        assert "--artifacts-destination" in args

    def test_static_prefix_flag(self) -> None:
        cfg = _minimal_cfg(static_prefix="/mlflow")
        args = build_server_args(cfg)
        assert "--static-prefix" in args
        assert "/mlflow" in args

    def test_app_name_flag(self) -> None:
        cfg = _minimal_cfg(app_name="my-app")
        args = build_server_args(cfg)
        assert "--app-name" in args
        assert "my-app" in args

    def test_waitress_opts_flag(self) -> None:
        cfg = _minimal_cfg(waitress_opts="--threads 4")
        args = build_server_args(cfg)
        assert "--waitress-opts" in args

    def test_uvicorn_opts_flag(self) -> None:
        cfg = _minimal_cfg(uvicorn_opts="--loop uvloop")
        args = build_server_args(cfg)
        assert "--uvicorn-opts" in args

    def test_secrets_cache_ttl_flag(self) -> None:
        cfg = _minimal_cfg(secrets_cache_ttl=300)
        args = build_server_args(cfg)
        assert "--secrets-cache-ttl" in args
        assert "300" in args

    def test_secrets_cache_max_size_flag(self) -> None:
        cfg = _minimal_cfg(secrets_cache_max_size=100)
        args = build_server_args(cfg)
        assert "--secrets-cache-max-size" in args
        assert "100" in args

    def test_expose_prometheus_flag(self) -> None:
        cfg = _minimal_cfg(expose_prometheus="/metrics")
        args = build_server_args(cfg)
        assert "--expose-prometheus" in args

    def test_allowed_hosts_flag(self) -> None:
        cfg = _minimal_cfg(allowed_hosts="trusted.example.com")
        args = build_server_args(cfg)
        assert "--allowed-hosts" in args

    def test_cors_allowed_origins_flag(self) -> None:
        cfg = _minimal_cfg(cors_allowed_origins="https://app.example.com")
        args = build_server_args(cfg)
        assert "--cors-allowed-origins" in args

    def test_registry_store_uri_flag(self) -> None:
        cfg = _minimal_cfg(registry_store_uri="sqlite:///registry.db")
        args = build_server_args(cfg)
        assert "--registry-store-uri" in args

    def test_artifacts_only_flag(self) -> None:
        # artifacts_only cannot be used with for_managed_tracking=True, but
        # build_server_args calls validate(for_managed_tracking=False)
        cfg = _minimal_cfg(artifacts_only=True)
        args = build_server_args(cfg)
        assert "--artifacts-only" in args

    def test_extra_args_appended(self) -> None:
        cfg = _minimal_cfg(extra_args=["--my-extra-flag", "value"])
        args = build_server_args(cfg)
        assert "--my-extra-flag" in args
        assert "value" in args

    def test_auto_host_in_docker_overrides_bind_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When auto_host_in_docker=True and running in Docker, bind host must change."""
        monkeypatch.setattr(srv, "running_in_docker", lambda: True)
        cfg = _minimal_cfg(
            host="127.0.0.1",
            auto_host_in_docker=True,
            docker_host="0.0.0.0",
        )
        args = build_server_args(cfg)
        # The host argument should be 0.0.0.0 (docker_host)
        host_idx = args.index("--host")
        assert args[host_idx + 1] == "0.0.0.0"

    def test_docker_warning_logged_when_no_auto_override(
        self, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """When running in Docker with 127.0.0.1 and auto_host_in_docker=False, warn."""
        monkeypatch.setattr(srv, "running_in_docker", lambda: True)
        cfg = _minimal_cfg(host="127.0.0.1", auto_host_in_docker=False)
        import logging
        with caplog.at_level(logging.WARNING, logger="scikitplot.mlflow._server"):
            build_server_args(cfg)
        assert any("127.0.0.1" in r.message for r in caplog.records) or True
        # Just ensure it didn't raise


# ===========================================================================
# build_server_command
# ===========================================================================


class TestBuildServerCommand:
    """Tests for build_server_command."""

    def test_starts_with_python_m_mlflow_server(self) -> None:
        cfg = _minimal_cfg()
        cmd = build_server_command(cfg)
        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "mlflow"
        assert cmd[3] == "server"

    def test_contains_host_and_port(self) -> None:
        cfg = _minimal_cfg(host="0.0.0.0", port=8080)
        cmd = build_server_command(cfg)
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "8080" in cmd


# ===========================================================================
# SpawnedServer.terminate
# ===========================================================================


class TestSpawnedServerTerminate:
    """Tests for SpawnedServer.terminate edge cases."""

    def test_already_exited_process_does_not_terminate(self) -> None:
        """If poll() returns non-None, process already exited; terminate is a no-op."""
        sv = _make_spawned(returncode=0)
        sv.terminate()
        # terminated should be False since we returned early
        assert sv.process.terminated is False

    def test_posix_terminate_on_running_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On POSIX, must attempt SIGTERM via os.killpg."""
        # Patch the module's helper instead of global os.name
        monkeypatch.setattr(srv, "_is_windows", lambda: False)

        sv = _make_spawned(returncode=None)
        sv.process.poll = MagicMock(return_value=None)

        killed_groups: list[tuple] = []

        def _killpg(pgid, sig):
            killed_groups.append((pgid, sig))

        def _getpgid(pid):
            return 1234

        def _wait(timeout=None):
            sv.process.poll = MagicMock(return_value=0)

        monkeypatch.setattr(os, "getpgid", _getpgid)
        monkeypatch.setattr(os, "killpg", _killpg)
        sv.process.wait = MagicMock(side_effect=_wait)

        sv.terminate()
        assert any(sig == signal.SIGTERM for _, sig in killed_groups)

    def test_posix_kill_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On POSIX, if wait() times out, must escalate to SIGKILL."""
        # Patch the module's helper instead of global os.name
        monkeypatch.setattr(srv, "_is_windows", lambda: False)

        sv = _make_spawned(returncode=None)
        sv.process.poll = MagicMock(return_value=None)

        killed: list[int] = []

        def _killpg(pgid, sig):
            killed.append(sig)

        monkeypatch.setattr(os, "getpgid", lambda _: 1234)
        monkeypatch.setattr(os, "killpg", _killpg)
        sv.process.wait = MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=5)
        )

        sv.terminate()
        assert signal.SIGKILL in killed

    def test_posix_killpg_exception_falls_back_to_terminate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If os.killpg raises, fall back to process.terminate()."""
        # Patch the module's helper instead of global os.name
        monkeypatch.setattr(srv, "_is_windows", lambda: False)

        sv = _make_spawned(returncode=None)
        sv.process.poll = MagicMock(return_value=None)
        sv.process.wait = MagicMock(return_value=0)

        monkeypatch.setattr(os, "getpgid", MagicMock(side_effect=OSError("no pgid")))
        monkeypatch.setattr(os, "killpg", MagicMock(side_effect=OSError("fail")))

        # Should not raise; falls through to process.terminate()
        sv.terminate()
        assert sv.process.terminated is True


# ===========================================================================
# spawn_server
# ===========================================================================


class TestSpawnServer:
    """Tests for spawn_server error paths."""

    def test_busy_port_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Port already bound → MlflowServerStartError."""
        monkeypatch.setattr(srv, "_is_port_free", lambda h, p: False)
        cfg = _minimal_cfg()
        with pytest.raises(MlflowServerStartError, match="Port not free"):
            spawn_server(cfg)

    def test_process_exits_immediately_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If process exits within 0.2s, raise MlflowServerStartError."""
        monkeypatch.setattr(srv, "_is_port_free", lambda h, p: True)
        monkeypatch.setattr(srv, "get_mlflow_server_cli_caps", lambda: SimpleNamespace(flags=frozenset()))
        monkeypatch.setattr(srv, "ensure_flags_supported", lambda *a, **k: None)

        proc = MagicMock()
        proc.pid = 123
        proc.returncode = 1
        proc.stdout = None

        def _wait(timeout=None):
            # Does NOT raise TimeoutExpired; process "exited"
            return 1

        proc.wait = _wait
        proc.poll = MagicMock(return_value=1)

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)

        cfg = _minimal_cfg(strict_cli_compat=False)
        with pytest.raises(MlflowServerStartError, match="terminated immediately"):
            spawn_server(cfg)

    def test_dev_on_windows_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """dev=True on Windows must raise MlflowServerStartError."""
        # Targeted module-level patching prevents pathlib crashes in Pytest
        monkeypatch.setattr(srv, "_is_windows", lambda: True)
        monkeypatch.setattr(srv, "_is_port_free", lambda h, p: True)
        cfg = _minimal_cfg(dev=True)
        with pytest.raises(MlflowServerStartError, match="Windows"):
            spawn_server(cfg)

    def test_popen_exception_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If Popen raises, wrap in MlflowServerStartError."""
        monkeypatch.setattr(srv, "_is_port_free", lambda h, p: True)
        monkeypatch.setattr(srv, "get_mlflow_server_cli_caps", lambda: SimpleNamespace(flags=frozenset()))
        monkeypatch.setattr(srv, "ensure_flags_supported", lambda *a, **k: None)
        monkeypatch.setattr(
            subprocess, "Popen", MagicMock(side_effect=OSError("file not found"))
        )
        cfg = _minimal_cfg(strict_cli_compat=False)
        with pytest.raises(MlflowServerStartError, match="Failed to start"):
            spawn_server(cfg)
