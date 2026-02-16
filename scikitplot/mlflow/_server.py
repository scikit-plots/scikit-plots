# scikitplot/mlflow/_server.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_server.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


from ._cli_caps import ensure_flags_supported, get_mlflow_server_cli_caps
from ._config import ServerConfig
from ._container import running_in_docker
from ._errors import MlflowServerStartError


def _is_port_free(host: str, port: int) -> bool:
    """
    Return True if the (host, port) is bindable (port is free).

    Parameters
    ----------
    host : str
        Host interface to check.
    port : int
        TCP port to check.

    Returns
    -------
    bool
        True if the port appears free.

    Notes
    -----
    This is a strict preflight check; it does not reserve the port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def build_server_args(cfg: ServerConfig) -> list[str]:  # noqa: PLR0912
    """
    Build the CLI args for `mlflow server` from a ServerConfig.

    Parameters
    ----------
    cfg : ServerConfig
        Server configuration.

    Returns
    -------
    list[str]
        CLI arguments (excluding `python -m mlflow server` prefix).

    Raises
    ------
    ValueError
        If cfg violates generic constraints.
    """
    cfg.validate(for_managed_tracking=False)

    bind_host = cfg.host
    if bind_host == "127.0.0.1" and running_in_docker():
        if cfg.auto_host_in_docker:
            bind_host = cfg.docker_host
        else:
            logger.warning(
                "MLflow server host is 127.0.0.1 while running inside Docker; "
                "the UI is not reachable from outside the container unless you "
                "publish ports and bind to 0.0.0.0. Consider setting "
                "ServerConfig.auto_host_in_docker=True (or host='0.0.0.0') and "
                "SessionConfig.public_tracking_uri='http://localhost:<port>'."
            )

    args: list[str] = [
        "--host",
        bind_host,
        "--port",
        str(cfg.port),
    ]

    if cfg.workers is not None:
        args += ["--workers", str(cfg.workers)]

    if cfg.backend_store_uri is not None:
        args += ["--backend-store-uri", cfg.backend_store_uri]

    if cfg.registry_store_uri is not None:
        args += ["--registry-store-uri", cfg.registry_store_uri]

    if cfg.default_artifact_root is not None:
        args += ["--default-artifact-root", cfg.default_artifact_root]

    if cfg.serve_artifacts:
        args += ["--serve-artifacts"]

    if cfg.no_serve_artifacts:
        args += ["--no-serve-artifacts"]

    if cfg.artifacts_only:
        args += ["--artifacts-only"]

    if cfg.artifacts_destination is not None:
        args += ["--artifacts-destination", cfg.artifacts_destination]

    if cfg.allowed_hosts is not None:
        args += ["--allowed-hosts", cfg.allowed_hosts]
    if cfg.cors_allowed_origins is not None:
        args += ["--cors-allowed-origins", cfg.cors_allowed_origins]
    if cfg.x_frame_options is not None:
        args += ["--x-frame-options", cfg.x_frame_options]
    if cfg.disable_security_middleware:
        args += ["--disable-security-middleware"]

    if cfg.static_prefix is not None:
        args += ["--static-prefix", cfg.static_prefix]

    if cfg.gunicorn_opts is not None:
        args += ["--gunicorn-opts", cfg.gunicorn_opts]
    if cfg.waitress_opts is not None:
        args += ["--waitress-opts", cfg.waitress_opts]
    if cfg.uvicorn_opts is not None:
        args += ["--uvicorn-opts", cfg.uvicorn_opts]

    if cfg.expose_prometheus is not None:
        args += ["--expose-prometheus", cfg.expose_prometheus]

    if cfg.app_name is not None:
        args += ["--app-name", cfg.app_name]

    if cfg.dev:
        args += ["--dev"]

    if cfg.secrets_cache_ttl is not None:
        args += ["--secrets-cache-ttl", str(cfg.secrets_cache_ttl)]
    if cfg.secrets_cache_max_size is not None:
        args += ["--secrets-cache-max-size", str(cfg.secrets_cache_max_size)]

    if cfg.extra_args:
        args += list(cfg.extra_args)

    if cfg.strict_cli_compat:
        caps = get_mlflow_server_cli_caps()
        ensure_flags_supported(
            args, supported_flags=caps.flags, context="mlflow server command"
        )

    return args


def build_server_command(cfg: ServerConfig) -> list[str]:
    """
    Build a deterministic `mlflow server` command.

    Parameters
    ----------
    cfg : ServerConfig
        Server configuration.

    Returns
    -------
    list[str]
        Command list suitable for `subprocess.Popen`.

    Notes
    -----
    Uses `sys.executable -m mlflow` to ensure the server runs in the same environment.
    """
    args = build_server_args(cfg)
    return [sys.executable, "-m", "mlflow", "server", *args]


@dataclass
class SpawnedServer:
    """
    Spawned MLflow server process state.

    Attributes
    ----------
    process : subprocess.Popen
        The underlying server process.
    command : list[str]
        The command used to launch the server.
    """

    _process: subprocess.Popen
    _command: list[str]
    _started_at: float

    @property
    def process(self) -> subprocess.Popen:
        """The underlying server process."""
        return self._process

    @property
    def command(self) -> list[str]:
        """The command used to launch the server."""
        return self._command

    @property
    def started_at(self) -> float:
        """Timestamp (time.time()) when the server was started."""
        return self._started_at

    def read_all_output(self) -> str:
        """
        Read all remaining captured stdout (best-effort).

        Returns
        -------
        str
            Captured output text, or an empty string.
        """
        try:
            if self.process.stdout is None:
                return ""
            return self.process.stdout.read() or ""
        except Exception:
            return ""

    def terminate(self) -> None:
        """
        Terminate the spawned process deterministically.

        Returns
        -------
        None

        Notes
        -----
        Cross-platform strict teardown:
        - POSIX: terminates the whole process group (start_new_session=True).
        - Windows: attempts CTRL_BREAK_EVENT for process group, then terminate/kill.
        """
        if self.process.poll() is not None:
            _ = self.read_all_output()
            return

        if os.name == "nt":
            # Best-effort CTRL_BREAK_EVENT to the process group.
            try:
                self.process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                self.process.wait(timeout=3)
            except Exception:
                pass

            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        else:
            # POSIX: kill process group
            try:
                pgid = os.getpgid(self.process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                # Fallback to direct terminate
                try:  # noqa: SIM105
                    self.process.terminate()
                except Exception:
                    pass

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    try:  # noqa: SIM105
                        self.process.kill()
                    except Exception:
                        pass

        _ = self.read_all_output()


def spawn_server(cfg: ServerConfig) -> SpawnedServer:
    """
    Spawn an MLflow server subprocess.

    Parameters
    ----------
    cfg : ServerConfig
        Server configuration.

    Returns
    -------
    SpawnedServer
        Handle to the spawned server.

    Raises
    ------
    MlflowServerStartError
        If the configured port is not free or the process fails to start.
    """
    cfg.validate(for_managed_tracking=True)

    if cfg.dev and os.name == "nt":
        raise MlflowServerStartError(
            "MLflow server --dev is unsupported on Windows (per MLflow CLI)."
        )

    if not _is_port_free(cfg.host, cfg.port):
        raise MlflowServerStartError(f"Port not free: {cfg.host}:{cfg.port}")

    cmd = build_server_command(cfg)

    popen_kwargs = {
        "env": os.environ.copy(),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
    }

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]  # noqa: S603
    except Exception as e:
        raise MlflowServerStartError(f"Failed to start mlflow server: {e!r}") from e

    server = SpawnedServer(_process=proc, _command=cmd, _started_at=time.time())

    # Fast-fail if process exits immediately (bad flags, missing deps, etc.)
    try:
        proc.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        return server

    rc = proc.returncode
    out = server.read_all_output()
    raise MlflowServerStartError(
        "MLflow server terminated immediately "
        f"(returncode={rc}). Command: {cmd!r}. Output:\n{out}"
    )
