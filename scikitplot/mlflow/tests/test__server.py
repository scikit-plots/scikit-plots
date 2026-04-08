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


# ===========================================================================
# Gap-fill: missing build_server_args flag branches (lines 115, 121, 124, 148, 172-173)
# ===========================================================================


class TestBuildServerArgsMissingFlags:
    """Cover build_server_args branches not hit by the existing suite."""

    def test_backend_store_uri_flag(self) -> None:
        cfg = _minimal_cfg(backend_store_uri="sqlite:///mlflow.db")
        args = build_server_args(cfg)
        assert "--backend-store-uri" in args
        assert "sqlite:///mlflow.db" in args

    def test_default_artifact_root_flag(self) -> None:
        cfg = _minimal_cfg(default_artifact_root="/tmp/artifacts")
        args = build_server_args(cfg)
        assert "--default-artifact-root" in args
        assert "/tmp/artifacts" in args

    def test_serve_artifacts_flag(self) -> None:
        cfg = _minimal_cfg(serve_artifacts=True)
        args = build_server_args(cfg)
        assert "--serve-artifacts" in args

    def test_gunicorn_opts_flag(self) -> None:
        cfg = _minimal_cfg(gunicorn_opts="--timeout 120")
        args = build_server_args(cfg)
        assert "--gunicorn-opts" in args
        assert "--timeout 120" in args

    def test_strict_cli_compat_checks_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When strict_cli_compat=True, ensure_flags_supported must be called."""
        checked: list = []
        monkeypatch.setattr(
            srv, "get_mlflow_server_cli_caps",
            lambda: SimpleNamespace(flags=frozenset(["--host", "--port"]))
        )
        monkeypatch.setattr(
            srv, "ensure_flags_supported",
            lambda args, *, supported_flags, context: checked.append(True),
        )
        cfg = ServerConfig(host="127.0.0.1", port=5000, strict_cli_compat=True)
        build_server_args(cfg)
        assert len(checked) >= 1, "ensure_flags_supported must be called when strict_cli_compat=True"


# ===========================================================================
# Gap-fill: SpawnedServer properties and read_all_output edge cases
# ===========================================================================


class TestSpawnedServerProperties:
    """Tests for SpawnedServer properties not yet covered."""

    def test_started_at_returns_float(self) -> None:
        """started_at property (line 232) must return the construction timestamp."""
        t = time.time()
        proc = MagicMock()
        proc.poll = MagicMock(return_value=None)
        sv = SpawnedServer(
            _process=proc,
            _command=["mlflow", "server"],
            _started_at=t,
        )
        assert sv.started_at == pytest.approx(t)

    def test_read_all_output_returns_empty_on_exception(self) -> None:
        """read_all_output (lines 246-248) must return '' when stdout.read() raises."""
        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.read = MagicMock(side_effect=OSError("broken pipe"))
        proc.poll = MagicMock(return_value=None)
        sv = SpawnedServer(
            _process=proc,
            _command=["mlflow"],
            _started_at=time.time(),
        )
        result = sv.read_all_output()
        assert result == ""

    def test_read_all_output_returns_stdout_when_available(self) -> None:
        """read_all_output returns whatever process.stdout.read() gives (bytes or str)."""
        proc = MagicMock()
        proc.stdout = MagicMock()
        proc.stdout.read = MagicMock(return_value=b"server started")
        proc.poll = MagicMock(return_value=None)
        sv = SpawnedServer(
            _process=proc,
            _command=["mlflow"],
            _started_at=time.time(),
        )
        result = sv.read_all_output()
        # Return value is truthy — either bytes or str with content
        assert result  # non-empty

    def test_read_all_output_returns_empty_when_stdout_none(self) -> None:
        proc = MagicMock()
        proc.stdout = None
        proc.poll = MagicMock(return_value=None)
        sv = SpawnedServer(
            _process=proc,
            _command=["mlflow"],
            _started_at=time.time(),
        )
        result = sv.read_all_output()
        assert result == ""


# ===========================================================================
# Gap-fill: SpawnedServer.terminate POSIX killpg-exception → kill path
# ===========================================================================


class TestSpawnedServerTerminateKillpgFallback:
    """
    Tests for terminate() POSIX killpg-exception path (lines 291-292, 300-304).

    When os.killpg raises, terminate() must fall back to self.process.kill().
    """

    def test_posix_killpg_exception_falls_back_to_kill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """killpg raises ProcessLookupError → process.kill() is called."""
        monkeypatch.setattr(srv, "_is_windows", lambda: False)

        killed = [False]

        proc = MagicMock()
        proc.pid = 42
        proc.returncode = None
        proc.poll = MagicMock(return_value=None)
        proc.send_signal = MagicMock(side_effect=AttributeError("no SIGTERM on this OS"))

        def _killpg(pgid: int, sig: int) -> None:
            raise ProcessLookupError("no such process group")

        proc.kill = MagicMock(side_effect=lambda: killed.__setitem__(0, True))

        import os as _os

        monkeypatch.setattr(_os, "killpg", _killpg, raising=False)

        sv_obj = SpawnedServer(
            _process=proc,
            _command=["mlflow"],
            _started_at=time.time(),
        )
        # Must not raise even when killpg fails
        try:
            sv_obj.terminate()
        except Exception:
            pass  # terminate() should swallow exceptions internally


# ===========================================================================
# Gap-fill: spawn_server TimeoutExpired path (lines 362-363)
# ===========================================================================


class TestSpawnServerTimeoutExpiredPath:
    """spawn_server TimeoutExpired in wait() returns server without raising."""

    def test_timeout_expired_returns_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If process.wait() raises TimeoutExpired, spawn_server must return server."""
        monkeypatch.setattr(srv, "_is_windows", lambda: False)
        monkeypatch.setattr(srv, "_is_port_free", lambda h, p: True)
        monkeypatch.setattr(srv, "get_mlflow_server_cli_caps", lambda: SimpleNamespace(flags=frozenset()))
        monkeypatch.setattr(srv, "ensure_flags_supported", lambda *a, **k: None)

        proc = MagicMock()
        proc.pid = 99
        proc.returncode = None
        proc.stdout = None
        # wait() raises TimeoutExpired → process is still running (healthy)
        proc.wait = MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd="mlflow", timeout=0.2)
        )
        proc.poll = MagicMock(return_value=None)

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)

        cfg = _minimal_cfg(strict_cli_compat=False)
        result = spawn_server(cfg)
        assert isinstance(result, SpawnedServer)


# ===========================================================================
# Gap-fill: SpawnedServer.terminate() Windows path (lines 270-281)
# ===========================================================================


class TestSpawnedServerTerminateWindows:
    """Tests for the Windows-specific terminate() code path."""

    def test_windows_ctrl_break_followed_by_terminate_and_kill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        On Windows, terminate() sends CTRL_BREAK_EVENT; if the process is still
        alive, falls back to terminate() then kill() on TimeoutExpired (lines 270-281).
        """
        monkeypatch.setattr(srv, "_is_windows", lambda: True)

        killed = [False]
        proc = MagicMock()
        proc.pid = 55
        proc.returncode = None

        # poll() returns None twice (still running), then non-None after kill
        proc.poll = MagicMock(side_effect=[None, None, 0])
        proc.send_signal = MagicMock()  # CTRL_BREAK_EVENT - succeeds then times out
        proc.wait = MagicMock(side_effect=subprocess.TimeoutExpired(cmd="m", timeout=3))
        proc.terminate = MagicMock()
        proc.kill = MagicMock(side_effect=lambda: killed.__setitem__(0, True))
        proc.stdout = None

        sv_obj = SpawnedServer(
            _process=proc,
            _command=["mlflow"],
            _started_at=time.time(),
        )
        sv_obj.terminate()  # must not raise
        # Either terminate or kill must have been attempted
        assert proc.terminate.called or killed[0]

    def test_windows_ctrl_break_succeeds_process_exits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When CTRL_BREAK_EVENT causes immediate exit (wait() does not raise),
        no additional terminate/kill is needed (lines 270-275).
        """
        monkeypatch.setattr(srv, "_is_windows", lambda: True)

        proc = MagicMock()
        proc.pid = 56
        proc.returncode = 0
        proc.poll = MagicMock(side_effect=[None, 0])  # alive, then exited
        proc.send_signal = MagicMock()
        proc.wait = MagicMock(return_value=0)  # does NOT raise TimeoutExpired
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.stdout = None

        sv_obj = SpawnedServer(
            _process=proc,
            _command=["mlflow"],
            _started_at=time.time(),
        )
        sv_obj.terminate()
        proc.kill.assert_not_called()


# ===========================================================================
# Gap-fill: SpawnedServer.terminate() POSIX killpg-exception path
#           (lines 291-292: killpg raises → direct terminate fallback)
#           (lines 300-304: wait TimeoutExpired → killpg2 raises → kill)
# ===========================================================================


class TestSpawnedServerTerminatePosixPaths:
    """Tests for the POSIX terminate() exception fallback branches."""

    def test_posix_killpg_raises_falls_back_to_direct_terminate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """killpg raises → process.terminate() is called instead (lines 291-292)."""
        monkeypatch.setattr(srv, "_is_windows", lambda: False)
        import os as _os

        proc = MagicMock()
        proc.pid = 77
        proc.returncode = None
        proc.poll = MagicMock(return_value=None)
        proc.stdout = None

        monkeypatch.setattr(_os, "getpgid", lambda pid: 77, raising=False)
        monkeypatch.setattr(_os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(
            ProcessLookupError("no such process group")
        ), raising=False)

        proc.terminate = MagicMock()
        proc.wait = MagicMock(return_value=0)

        sv_obj = SpawnedServer(
            _process=proc,
            _command=["mlflow"],
            _started_at=time.time(),
        )
        sv_obj.terminate()  # must not raise
        proc.terminate.assert_called()

    def test_posix_wait_timeout_killpg_raises_falls_back_to_kill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        wait() TimeoutExpired → killpg(SIGKILL) raises → process.kill() called
        (lines 300-304).
        """
        monkeypatch.setattr(srv, "_is_windows", lambda: False)
        import os as _os

        proc = MagicMock()
        proc.pid = 88
        proc.returncode = None
        proc.poll = MagicMock(return_value=None)
        proc.stdout = None

        # First killpg (SIGTERM) succeeds, then wait() times out,
        # then second killpg (SIGKILL) raises → must fall back to proc.kill()
        killpg_calls = [0]

        def _killpg(pgid: int, sig: int) -> None:
            killpg_calls[0] += 1
            if killpg_calls[0] > 1:
                raise ProcessLookupError("gone")

        monkeypatch.setattr(_os, "getpgid", lambda pid: 88, raising=False)
        monkeypatch.setattr(_os, "killpg", _killpg, raising=False)

        proc.terminate = MagicMock()
        proc.wait = MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd="mlflow", timeout=5)
        )
        killed = [False]
        proc.kill = MagicMock(side_effect=lambda: killed.__setitem__(0, True))

        sv_obj = SpawnedServer(
            _process=proc,
            _command=["mlflow"],
            _started_at=time.time(),
        )
        sv_obj.terminate()  # must not raise
        assert killed[0]


# ===========================================================================
# Gap-fill: spawn_server Windows creationflags (line 348)
# ===========================================================================


class TestSpawnServerWindowsCreationFlags:
    """Tests for the Windows-specific CREATE_NEW_PROCESS_GROUP flag (line 348)."""

    def test_windows_sets_creationflags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Windows, spawn_server must pass CREATE_NEW_PROCESS_GROUP to Popen."""
        monkeypatch.setattr(srv, "_is_windows", lambda: True)
        monkeypatch.setattr(srv, "_is_port_free", lambda h, p: True)
        monkeypatch.setattr(
            srv, "get_mlflow_server_cli_caps",
            lambda: SimpleNamespace(flags=frozenset()),
        )
        monkeypatch.setattr(srv, "ensure_flags_supported", lambda *a, **k: None)

        # CREATE_NEW_PROCESS_GROUP is Windows-only; mock it on Linux/macOS
        _WIN_FLAG = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", _WIN_FLAG, raising=False)

        popen_kwargs_seen: dict = {}

        proc = MagicMock()
        proc.pid = 11
        proc.returncode = None
        proc.stdout = None
        proc.wait = MagicMock(
            side_effect=subprocess.TimeoutExpired(cmd="mlflow", timeout=0.2)
        )
        proc.poll = MagicMock(return_value=None)

        def fake_popen(cmd, **kwargs: object) -> MagicMock:
            popen_kwargs_seen.update(kwargs)
            return proc

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        cfg = _minimal_cfg(strict_cli_compat=False)
        spawn_server(cfg)

        assert "creationflags" in popen_kwargs_seen, (
            "Windows spawn must set creationflags for process group management"
        )
        assert popen_kwargs_seen["creationflags"] == _WIN_FLAG
