# tests/test_cli.py
# SPDX-License-Identifier: BSD-3-Clause
"""CLI and Docker profile tests for :mod:`scikitplot.mcp.__main__`."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PKG_ROOT))

from scikitplot.mcp import __main__ as cli  # noqa: E402


def _parse(*argv: str, environ=None):
    args = cli._parser().parse_args(list(argv))
    return cli._resolve_config(args, environ={} if environ is None else environ)


def test_default_profile_is_local_stdio():
    config = _parse()
    assert config.transport == "stdio"
    assert config.host == "127.0.0.1"
    assert config.docker is False
    assert config.health_path == "/healthz"


def test_docker_profile_uses_streamable_http_and_container_bind():
    config = _parse("--docker")
    assert config.transport == "streamable-http"
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.path == "/mcp"
    assert config.health_path == "/healthz"
    assert config.allow_unauthenticated_remote is True


def test_remote_bind_requires_explicit_acknowledgement():
    with pytest.raises(SystemExit, match="refusing unauthenticated non-local bind"):
        _parse("--transport", "streamable-http", "--host", "0.0.0.0")


def test_environment_config_is_supported():
    config = _parse(
        environ={
            "SCIKITPLOT_MCP_DOCKER": "true",
            "SCIKITPLOT_MCP_PORT": "8123",
            "SCIKITPLOT_MCP_PATH": "/rpc",
            "SCIKITPLOT_MCP_HEALTH_PATH": "/live",
            "SCIKITPLOT_MCP_LOG_LEVEL": "warning",
        }
    )
    assert config.docker is True
    assert config.port == 8123
    assert config.path == "/rpc"
    assert config.health_path == "/live"
    assert config.log_level == "WARNING"


@pytest.mark.parametrize("bad_path", ["mcp", "/mcp?x=1", "/mcp#frag", "/mcp\\x", "/mcp//x"])
def test_invalid_mcp_paths_rejected(bad_path):
    with pytest.raises(SystemExit):
        _parse("--docker", "--path", bad_path)


def test_health_and_mcp_paths_must_not_overlap():
    with pytest.raises(SystemExit, match="must not overlap"):
        _parse("--docker", "--path", "/mcp", "--health-path", "/mcp/healthz")


def test_probe_converts_wildcard_bind_to_loopback():
    config = _parse("--docker", "--probe")
    assert cli._health_url(config) == "http://127.0.0.1:8000/healthz"


def test_probe_success(monkeypatch, capsys):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, _limit):
            return b'{"status":"ok"}'

    class Opener:
        def open(self, url, timeout):
            assert url == "http://127.0.0.1:8000/healthz"
            assert timeout == 2.0
            return Response()

    monkeypatch.setattr(cli, "build_opener", lambda *_args: Opener())
    assert cli._probe_health(_parse("--docker", "--probe")) == 0
    assert "HEALTHY" in capsys.readouterr().out


def test_probe_rejects_invalid_payload(monkeypatch, capsys):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, _limit):
            return b'{"status":"starting"}'

    class Opener:
        def open(self, _url, timeout):
            assert timeout == 2.0
            return Response()

    monkeypatch.setattr(cli, "build_opener", lambda *_args: Opener())
    assert cli._probe_health(_parse("--docker", "--probe")) == 1
    assert "UNHEALTHY" in capsys.readouterr().err


def test_print_effective_config_avoids_server_creation(monkeypatch, caplog):
    monkeypatch.setattr(
        cli,
        "create_server",
        lambda *_args, **_kwargs: pytest.fail("server must not be created"),
    )
    output = io.StringIO()
    assert (
        cli.main(
            ["--docker", "--print-effective-config"],
            stdout=output,
        )
        == 0
    )
    data = json.loads(output.getvalue())
    assert data["transport"] == "streamable-http"
    assert data["host"] == "0.0.0.0"
    assert not any(
        '"transport"' in record.getMessage() for record in caplog.records
    )


def test_docker_main_passes_http_and_health_settings(monkeypatch):
    captured = {}

    class FakeServer:
        def run(self, **kwargs):
            captured["run"] = kwargs

    def fake_create_server(_retriever, **kwargs):
        captured["create"] = kwargs
        return FakeServer()

    monkeypatch.setattr(cli, "create_server", fake_create_server)
    assert cli.main(["--docker", "--port", "8123", "--path", "/rpc"]) == 0
    assert captured["create"]["health_path"] == "/healthz"
    assert captured["run"] == {
        "transport": "streamable-http",
        "host": "0.0.0.0",
        "port": 8123,
        "streamable_http_path": "/rpc",
        "json_response": True,
        "stateless_http": True,
        "max_request_body_size": 1024 * 1024,
    }


def test_backend_self_test_is_repeatable_and_avoids_server_creation(monkeypatch):
    monkeypatch.setattr(
        cli,
        "create_server",
        lambda *_args, **_kwargs: pytest.fail("server must not be created"),
    )

    def run_once():
        output = io.StringIO()
        assert (
            cli.main(
                ["--self-test", "--self-test-query", "transport"],
                stdout=output,
            )
            == 0
        )
        return json.loads(output.getvalue())

    first = run_once()
    second = run_once()
    assert first == second
    assert first["count"] == len(first["passages"]) == len(first["citations"])
    assert first["count"] >= 1


def test_probe_and_self_test_are_mutually_exclusive():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _parse("--docker", "--probe", "--self-test")


def test_effective_config_resolution_is_idempotent():
    argv = ("--docker", "--port", "8123", "--path", "/rpc")
    assert _parse(*argv) == _parse(*argv)


def test_backend_self_test_can_require_exact_canary(monkeypatch):
    monkeypatch.setattr(
        cli,
        "create_server",
        lambda *_args, **_kwargs: pytest.fail("server must not be created"),
    )
    output = io.StringIO()
    assert (
        cli.main(
            [
                "--self-test",
                "--self-test-query",
                "MCP_CANARY_7F3A91C2",
                "--self-test-require-match",
                "--self-test-expected-doc-id",
                "scikitplot-canary-001",
            ],
            stdout=output,
        )
        == 0
    )
    payload = json.loads(output.getvalue())
    assert payload["count"] >= 1
    assert any(
        citation["doc_id"] == "scikitplot-canary-001"
        for citation in payload["citations"]
    )


def test_backend_self_test_required_match_fails_closed():
    with pytest.raises(SystemExit, match="returned no matching documentation"):
        cli.main(
            [
                "--self-test",
                "--self-test-query",
                "zzzz_nonexistent_canary_query_8d9c45",
                "--self-test-require-match",
            ]
        )


def test_backend_self_test_expected_doc_id_fails_closed():
    with pytest.raises(SystemExit, match="expected doc_id"):
        cli.main(
            [
                "--self-test",
                "--self-test-query",
                "transport",
                "--self-test-expected-doc-id",
                "scikitplot-canary-001",
            ]
        )


def test_self_test_assertion_flags_require_self_test():
    with pytest.raises(SystemExit, match="require --self-test"):
        _parse("--self-test-require-match")
    with pytest.raises(SystemExit, match="require --self-test"):
        _parse("--self-test-expected-doc-id", "transport")
