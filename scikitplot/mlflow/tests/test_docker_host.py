from __future__ import annotations

"""Tests for Docker-aware host warnings and overrides.

These tests must not depend on MLflow CLI probing. They validate only the Docker-host warning logic.
"""

import importlib

from scikitplot.mlflow import ServerConfig


def _capture_warning(monkeypatch, server_mod):  # noqa: ANN001
    """Capture warnings emitted via server_mod.logger.warning deterministically."""
    warnings: list[str] = []

    def _warn(msg: str, *args, **kwargs) -> None:  # noqa: ANN001
        # Logging normally formats later; capture the formatted string here.
        try:
            warnings.append(msg % args if args else msg)
        except Exception:
            warnings.append(str(msg))

    monkeypatch.setattr(server_mod.logger, "warning", _warn)
    return warnings


def test_docker_host_no_override_logs_warning(monkeypatch) -> None:
    """Warn when bind host remains 127.0.0.1 inside Docker and auto override is disabled."""
    s = importlib.import_module("scikitplot.mlflow._server")
    monkeypatch.setattr(s, "running_in_docker", lambda: True)

    warnings = _capture_warning(monkeypatch, s)

    cfg = ServerConfig(
        host="127.0.0.1",
        port=5000,
        auto_host_in_docker=False,
        docker_host="0.0.0.0",
        strict_cli_compat=False,
    )
    args = s.build_server_args(cfg)

    assert args[:4] == ["--host", "127.0.0.1", "--port", "5000"]
    assert len(warnings) == 1
    assert "while running inside Docker" in warnings[0]
    assert "0.0.0.0" in warnings[0]


def test_docker_host_override(monkeypatch) -> None:
    """When auto_host_in_docker=True, bind host is rewritten and no warning is emitted."""
    s = importlib.import_module("scikitplot.mlflow._server")
    monkeypatch.setattr(s, "running_in_docker", lambda: True)

    warnings = _capture_warning(monkeypatch, s)

    cfg = ServerConfig(
        host="127.0.0.1",
        port=5000,
        auto_host_in_docker=True,
        docker_host="0.0.0.0",
        strict_cli_compat=False,
    )
    args = s.build_server_args(cfg)

    hi = args.index("--host")
    assert args[hi + 1] == "0.0.0.0"
    assert warnings == []
