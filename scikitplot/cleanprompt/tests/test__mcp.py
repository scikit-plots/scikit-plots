"""
Tests for :mod:`scikitplot.cleanprompt._mcp`.

Notes
-----
**Developer notes.** ``TestNothingReachesTheModel`` is the property the tool
list was designed around: in MCP every tool result enters the model's
context, so no result — success or failure, text or structured — may contain
a removed value, or the absolute path of the user's home.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from .. import FluentCleanPrompt
from .._exceptions import CleanPromptError
from .._guard import Guard
from .._mcp import PROTOCOL_VERSIONS, McpServer, serve
from .._runtime import Cleaner

SECRETS = (
    "Marion Holt",
    "ann@example.com",
    "00412345",
    "+1 555 010 4477",
    "hunter2hunter2",
)


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "records").mkdir()
    # Bytes, not text: a file written in text mode ends its lines with CRLF on
    # Windows, and the tools return a file's line endings as they are.
    (tmp_path / "records" / "patients.csv").write_bytes(
        b"name,email,mrn\nMarion Holt,ann@example.com,00412345\n"
    )
    (tmp_path / "records" / "note.txt").write_bytes(
        b"Call Marion Holt on +1 555 010 4477.\n"
    )
    (tmp_path / ".env").write_bytes(b"DB_PASSWORD=hunter2hunter2\n")
    return tmp_path


def _server(root, **plan):
    builder = FluentCleanPrompt().packs(plan.pop("packs", "all"))
    frozen = builder.plan()
    return McpServer(lambda: Guard(Cleaner(frozen)), [root])


def _call(server, name, **arguments):
    reply = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return reply["result"]


class TestProtocol:
    def test_initialize_negotiates_the_version(self, root):
        server = _server(root)
        for asked, answered in (
            ("2024-11-05", "2024-11-05"),
            ("2099-01-01", PROTOCOL_VERSIONS[0]),
        ):
            reply = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": asked},
                }
            )
            assert reply["result"]["protocolVersion"] == answered
            assert reply["result"]["capabilities"] == {"tools": {"listChanged": False}}

    def test_notifications_get_no_answer(self, root):
        assert (
            _server(root).handle(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            is None
        )

    def test_ping_and_unknown_method(self, root):
        server = _server(root)
        assert (
            server.handle({"jsonrpc": "2.0", "id": 1, "method": "ping"})["result"] == {}
        )
        assert (
            server.handle({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})[
                "error"
            ]["code"]
            == -32601
        )

    def test_invalid_requests(self, root):
        server = _server(root)
        assert server.handle({"id": 1, "method": "ping"})["error"]["code"] == -32600
        assert server.handle([])["error"]["code"] == -32600
        bad = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "nope"},
        }
        assert server.handle(bad)["error"]["code"] == -32602

    def test_every_tool_has_a_closed_schema(self, root):
        tools = _server(root).handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )["result"]["tools"]
        assert {tool["name"] for tool in tools} == {
            "cleanprompt_read_file",
            "cleanprompt_write_file",
            "cleanprompt_encode_text",
            "cleanprompt_encode_folder",
            "cleanprompt_inspect",
            "cleanprompt_forget",
        }
        for tool in tools:
            assert tool["inputSchema"]["type"] == "object"
            assert tool["inputSchema"]["additionalProperties"] is False

    def test_a_batch_is_answered_per_request(self, root):
        replies = _server(root).handle(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "method": "notifications/x"},
            ]
        )
        assert [reply["id"] for reply in replies] == [1]


class TestTools:
    def test_read_then_write_restores_locally(self, root):
        server = _server(root)
        read = _call(server, "cleanprompt_read_file", path="records/patients.csv")
        assert read["isError"] is False
        assert (
            read["content"][0]["text"]
            == "name,email,mrn\n[PERSON-1],[EMAIL-1],[MRN-1]\n"
        )
        written = _call(
            server,
            "cleanprompt_write_file",
            path="out/reply.txt",
            text="Dear [PERSON-1] ([MRN-1]) [X-9]",
        )
        assert written["isError"] is False
        assert (
            root / "out" / "reply.txt"
        ).read_text(encoding="utf-8") == "Dear Marion Holt (00412345) [X-9]"
        assert "[X-9]" in written["content"][0]["text"]

    def test_the_same_value_keeps_its_placeholder_across_calls(self, root):
        server = _server(root)
        _call(server, "cleanprompt_read_file", path="records/patients.csv")
        note = _call(server, "cleanprompt_read_file", path="records/note.txt")
        assert note["content"][0]["text"].startswith("Call [PERSON-1]")

    def test_writing_does_not_replace_a_file_unless_asked(self, root):
        server = _server(root)
        failed = _call(server, "cleanprompt_write_file", path=".env", text="x")
        assert failed["isError"] is True and (root / ".env").read_text(encoding="utf-8").startswith(
            "DB_PASSWORD"
        )
        assert (
            _call(
                server, "cleanprompt_write_file", path=".env", text="x", overwrite=True
            )["isError"]
            is False
        )

    @pytest.mark.parametrize(
        "path", ["/etc/passwd", "../outside.txt", "records/../../escape.txt"]
    )
    def test_paths_outside_the_roots_are_refused(self, root, path):
        for tool, arguments in (
            ("cleanprompt_read_file", {"path": path}),
            ("cleanprompt_write_file", {"path": path, "text": "x"}),
        ):
            result = _call(_server(root), tool, **arguments)
            assert (
                result["isError"] is True and "outside" in result["content"][0]["text"]
            )

    def test_a_symlink_out_of_the_root_is_refused(self, root, tmp_path_factory):
        outside = tmp_path_factory.mktemp("outside") / "secret.txt"
        outside.write_text("ann@example.com", encoding="utf-8")
        try:
            (root / "link.txt").symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable")
        assert (
            _call(_server(root), "cleanprompt_read_file", path="link.txt")["isError"]
            is True
        )

    def test_inspect_changes_nothing(self, root):
        server = _server(root)
        result = _call(server, "cleanprompt_inspect", path="records/patients.csv")
        assert result["structuredContent"]["kinds"] == {
            "EMAIL": 1,
            "MRN": 1,
            "PERSON": 1,
        }
        assert (
            _call(server, "cleanprompt_encode_text", text="ann@example.com")["content"][
                0
            ]["text"]
            == "[EMAIL-1]"
        )

    def test_inspect_a_folder(self, root):
        server = _server(root)
        result = _call(server, "cleanprompt_inspect", path="records")
        content = result["structuredContent"]
        assert content["kinds"] == {"EMAIL": 1, "MRN": 1, "PERSON": 2, "PHONE": 1}
        assert {entry["file"] for entry in content["files"]} == {
            "note.txt",
            "patients.csv",
        }
        assert result["content"][0]["text"].startswith("5 value(s) in 2 file(s)")
        # Inspecting a folder left no value behind in this session.
        written = _call(
            server, "cleanprompt_write_file", path="w.txt", text="[PERSON-1]"
        )
        assert written["structuredContent"]["unknown"] == ["[PERSON-1]"]

    def test_encode_folder_and_forget(self, root):
        server = _server(root)
        result = _call(
            server, "cleanprompt_encode_folder", source="records", target="safe"
        )
        assert result["structuredContent"]["encoded"] == 2
        assert "Marion Holt" not in (root / "safe" / "note.txt").read_text(encoding="utf-8")
        _call(server, "cleanprompt_forget")
        written = _call(
            server, "cleanprompt_write_file", path="after.txt", text="[PERSON-1]"
        )
        assert (root / "after.txt").read_text(encoding="utf-8") == "[PERSON-1]"
        assert written["structuredContent"]["unknown"] == ["[PERSON-1]"]

    def test_the_server_needs_a_real_root(self, tmp_path):
        with pytest.raises(CleanPromptError):
            McpServer(lambda: Guard(), [tmp_path / "missing"])


class TestNothingReachesTheModel:
    def test_no_tool_result_carries_a_value_or_the_home_path(self, root):
        server = _server(root)
        results = [
            _call(server, "cleanprompt_read_file", path="records/patients.csv"),
            _call(server, "cleanprompt_read_file", path="records/note.txt"),
            _call(server, "cleanprompt_read_file", path=".env"),
            _call(server, "cleanprompt_inspect", path="records/note.txt"),
            _call(server, "cleanprompt_inspect", path="records"),
            _call(server, "cleanprompt_inspect", path="."),
            _call(server, "cleanprompt_encode_folder", source="records", target="safe"),
            _call(
                server,
                "cleanprompt_write_file",
                path="reply.txt",
                text="To [PERSON-1] at [EMAIL-1]",
            ),
            _call(server, "cleanprompt_write_file", path="reply.txt", text="again"),
            _call(server, "cleanprompt_read_file", path="missing.txt"),
        ]
        dumped = json.dumps(results)
        for secret in SECRETS:
            assert secret not in dumped
        assert str(root) not in dumped


def test_stdio_round_trip(root):
    lines = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "cleanprompt_read_file",
                "arguments": {"path": "records/patients.csv"},
            },
        },
    ]
    stdin = io.StringIO("\n".join(json.dumps(line) for line in lines) + "\nnot json\n")
    stdout = io.StringIO()
    assert serve(_server(root), stdin, stdout) == 0
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [reply.get("id") for reply in replies] == [1, 2, None]
    assert replies[2]["error"]["code"] == -32700


def test_the_command_serves_real_stdio(root):
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "cleanprompt_read_file", "arguments": {"path": ".env"}},
        }
    )
    repo = Path(__file__).resolve().parents[3]
    done = subprocess.run(
        [sys.executable, "-m", "scikitplot.cleanprompt", "mcp", "--root", str(root)],
        input=request + "\n",
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=60,
        check=True,
    )
    reply = json.loads(done.stdout.strip())
    assert reply["result"]["content"][0]["text"] == "DB_PASSWORD=[SECRET-1]\n"
    assert "ready" in done.stderr
