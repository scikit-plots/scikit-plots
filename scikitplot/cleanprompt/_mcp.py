"""
A Model Context Protocol server that keeps the values on this machine.

Any agent that speaks MCP — a desktop assistant, a coding agent, an IDE — can
start this server and use cleanprompt without a line of code. It is standard
library only: JSON-RPC 2.0 over standard input and output, one message per
line.

Notes
-----
**User notes.** Register it with your agent. For a client configured by JSON::

    {
        "mcpServers": {
            "cleanprompt": {
                "command": "python",
                "args": [
                    "-m",
                    "scikitplot.cleanprompt",
                    "mcp",
                    "--root",
                    "/path/to/project",
                ],
            }
        }
    }

The agent then has these tools:

``cleanprompt_read_file``
    Read a file *through* cleanprompt. The agent — and the model behind it —
    receives the encoded text; the raw file never enters the conversation.
``cleanprompt_write_file``
    Write text the model produced to a file, with placeholders restored
    locally. The tool answers with a path and counts, so the restored values
    never enter the conversation either.
``cleanprompt_encode_text``
    Encode text before the agent passes it to another service.
``cleanprompt_encode_folder``
    Write a safe copy of a folder, as ``batch`` does, and report per file.
``cleanprompt_inspect``
    Count what a file or a text would have hidden, by kind.
``cleanprompt_forget``
    Drop every value this server holds.

**Developer notes — the rule that shaped the tool list.**

In MCP, *everything a tool returns is added to the model's context*. So a
``decode`` tool that returned restored text would hand the model the very
values it was built to withhold. No tool here returns a removed value: reading
returns encoded text, writing restores into a file and returns counts, and
inspecting returns kinds and numbers. The test suite asserts that property over
every tool.

*Paths are confined.* The server reads and writes only under the roots it was
started with (the working directory by default), after resolving symbolic
links; a path outside them is refused. A write never replaces an existing file
unless ``overwrite`` is true.

*One server is one conversation.* The vault lives in memory for the life of
the process and is never written anywhere, so a placeholder means the same
value across every tool call and the values end with the session.

*Protocol.* Requests are answered, notifications are not. A malformed line
gets a JSON-RPC parse error and the server keeps running; an unknown method
gets ``-32601``; a tool that fails reports ``isError`` in its result, as the
specification asks, rather than failing the request. Standard output carries
protocol messages only; diagnostics go to standard error.

See Also
--------
scikitplot.cleanprompt._guard.Guard : The gate every tool goes through.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import IO, Any

from ._exceptions import CleanPromptError, LeakError
from ._guard import Guard
from ._logging import audit

__all__ = [
    "PROTOCOL_VERSIONS",
    "McpServer",
    "serve",
]

#: MCP protocol versions this server speaks, newest first. A client asking for
#: another is answered with the newest, as the specification prescribes.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602


class _ToolError(Exception):
    """A tool failed for a reason the agent should be told."""


def _string(arguments: Mapping[str, Any], name: str, default: str | None = None) -> str:
    """Return a required (or defaulted) string argument."""
    value = arguments.get(name, default)
    if not isinstance(value, str) or (default is None and not value):
        raise _ToolError(f"argument {name!r} must be a non-empty string")
    return value


def _schema(
    properties: dict[str, Any],
    required: Iterable[str],
) -> dict[str, Any]:
    """Return a JSON Schema object for a tool's input."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_PATH = {"type": "string", "description": "A path under one of the server's roots."}
_FORMAT = {
    "type": "string",
    "description": "A format name or extension, e.g. 'csv', '.env', 'markdown'.",
}

_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "cleanprompt_read_file",
        "description": (
            "Read a local file through cleanprompt and return its text with every sensitive "
            "value replaced by a placeholder. Use this instead of reading the file directly: "
            "the raw values never enter the conversation. Office and PDF files return their "
            "extracted text."
        ),
        "inputSchema": _schema({"path": _PATH, "format": _FORMAT}, ["path"]),
    },
    {
        "name": "cleanprompt_write_file",
        "description": (
            "Write text containing placeholders to a local file, with the original values "
            "restored on this machine. Returns only the path and counts; the restored values "
            "are never returned."
        ),
        "inputSchema": _schema(
            {
                "path": _PATH,
                "text": {
                    "type": "string",
                    "description": "Text with placeholders such as [EMAIL-1].",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Replace an existing file. Default false.",
                },
            },
            ["path", "text"],
        ),
    },
    {
        "name": "cleanprompt_encode_text",
        "description": (
            "Replace sensitive values in a text with placeholders, before passing it to another "
            "service. The same value always gets the same placeholder in this session."
        ),
        "inputSchema": _schema(
            {"text": {"type": "string"}, "format": _FORMAT}, ["text"]
        ),
    },
    {
        "name": "cleanprompt_encode_folder",
        "description": (
            "Write an encoded copy of a folder: every file a selected format reads, under one "
            "vault. Unreadable files are skipped or refused and never written. Returns per-file "
            "statuses."
        ),
        "inputSchema": _schema(
            {"source": _PATH, "target": _PATH}, ["source", "target"]
        ),
    },
    {
        "name": "cleanprompt_inspect",
        "description": (
            "Report how many values of each kind a file, a folder or a text contains, without "
            "changing anything and without returning any value."
        ),
        "inputSchema": _schema(
            {"path": _PATH, "text": {"type": "string"}, "format": _FORMAT}, []
        ),
    },
    {
        "name": "cleanprompt_forget",
        "description": (
            "Drop every value this server holds. Placeholders issued earlier can no longer be restored."
        ),
        "inputSchema": _schema({}, []),
    },
)


def _folder_summary(items: list) -> tuple[str, dict[str, Any]]:
    """
    Summarise a folder survey for the model: kinds and counts, never values.

    Parameters
    ----------
    items : list of Item
        From :meth:`Cleaner.survey_tree`; paths are relative to the folder.

    Returns
    -------
    tuple of (str, dict)
        The text line and the structured content.
    """
    from ._runtime import kind_totals  # ruff: ignore[import-outside-top-level]

    kinds = kind_totals(items)
    summary = ", ".join(f"{kind}={count}" for kind, count in kinds.items()) or "nothing"
    encoded = sum(1 for item in items if item.status == "encoded")
    files = [
        {"file": item.relative, "status": item.status, "kinds": item.kinds}
        for item in items
    ]
    text = (
        f"{sum(kinds.values())} value(s) in {encoded} file(s) would be hidden: "
        f"{summary}"
    )
    return text, {"kinds": kinds, "files": files}


class McpServer:
    """
    Answer MCP requests with cleanprompt tools.

    Parameters
    ----------
    make_guard : callable
        Returns a fresh :class:`~scikitplot.cleanprompt.Guard`. Called at start
        and again by ``cleanprompt_forget``.
    roots : iterable of path-like
        Folders the tools may read and write under.
    name : str, default='cleanprompt'
        The server name announced to clients.
    version : str, default='1.0.0'
        The server version announced to clients.

    Notes
    -----
    **Developer notes.** :meth:`handle` is a pure function of one message and
    the server's state — no I/O — which is what makes the protocol testable
    without a subprocess; :func:`serve` only moves lines.
    """

    __slots__ = ("_guard", "_make_guard", "_protocol", "_roots", "name", "version")

    def __init__(
        self,
        make_guard: Callable[[], Guard],
        roots: Iterable[str | os.PathLike],
        name: str = "cleanprompt",
        version: str = "1.0.0",
    ) -> None:
        self._make_guard = make_guard
        self._guard = make_guard()
        self._roots = tuple(Path(root).resolve() for root in roots)
        if not self._roots:
            msg = "the MCP server needs at least one root folder"
            raise CleanPromptError(msg)
        for root in self._roots:
            if not root.is_dir():
                msg = f"root {str(root)!r} is not a folder"
                raise CleanPromptError(msg)
        self._protocol = PROTOCOL_VERSIONS[0]
        self.name = name
        self.version = version

    # -- protocol ---------------------------------------------------------

    def handle(  # ruff: ignore[too-many-return-statements]
        self,
        message: Any,
    ) -> Any:
        """
        Return the response to one JSON-RPC message, or ``None``.

        Parameters
        ----------
        message : object
            A decoded JSON-RPC request, notification or batch.

        Returns
        -------
        dict, list or None
            ``None`` for a notification, or a batch of notifications.
        """
        if isinstance(message, list):
            if not message:
                return _error(None, _INVALID_REQUEST, "an empty batch is not a request")
            replies = [
                reply
                for reply in (self.handle(one) for one in message)
                if reply is not None
            ]
            return replies or None
        if (
            not isinstance(message, dict)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
        ):
            return _error(
                message.get("id") if isinstance(message, dict) else None,
                _INVALID_REQUEST,
                "not a JSON-RPC 2.0 request",
            )
        is_request = "id" in message
        method = message["method"]
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return (
                _error(message.get("id"), _INVALID_PARAMS, "params must be an object")
                if is_request
                else None
            )
        if not is_request:
            return None  # notifications (initialized, cancelled, ...) need no answer
        request_id = message["id"]
        if method == "initialize":
            return _result(request_id, self._initialize(params))
        if method == "ping":
            return _result(request_id, {})
        if method == "tools/list":
            return _result(request_id, {"tools": [dict(tool) for tool in _TOOLS]})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or name not in {
                tool["name"] for tool in _TOOLS
            }:
                return _error(request_id, _INVALID_PARAMS, f"unknown tool {name!r}")
            if not isinstance(arguments, dict):
                return _error(
                    request_id, _INVALID_PARAMS, "arguments must be an object"
                )
            return _result(request_id, self._call(name, arguments))
        return _error(
            request_id, _METHOD_NOT_FOUND, f"method {method!r} is not supported"
        )

    def _initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Negotiate the protocol version and describe the server."""
        requested = params.get("protocolVersion")
        self._protocol = (
            requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        )
        return {
            "protocolVersion": self._protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": self.version},
            "instructions": (
                "Read local files with cleanprompt_read_file and write model output with "
                "cleanprompt_write_file, so personal data and secrets never enter the "
                "conversation. Placeholders such as [EMAIL-1] stand for values kept on this "
                "machine; never guess them."
            ),
        }

    def _call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one tool and wrap its outcome as an MCP tool result."""
        try:
            text, data = getattr(self, "_tool_" + name.replace("cleanprompt_", ""))(
                arguments
            )
        except (_ToolError, CleanPromptError, OSError, TypeError) as exc:
            # A leak refusal names kinds only; every other message here
            # names a path or a reason, never a value.
            audit("tool_failed", tool=name, error=type(exc).__name__)
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        audit("tool", tool=name)
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }
        if data is not None and self._protocol >= "2025-06-18":
            result["structuredContent"] = data
        return result

    # -- paths ----------------------------------------------------------

    def _path(self, raw: str, must_exist: bool = True) -> Path:
        """Resolve a path and refuse it unless it lies under a root."""
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self._roots[0] / candidate
        resolved = candidate.resolve()
        if not any(
            resolved == root or root in resolved.parents for root in self._roots
        ):
            raise _ToolError(f"{raw!r} is outside the folders this server may use")
        if must_exist and not resolved.exists():
            raise _ToolError(f"{raw!r} does not exist")
        return resolved

    def _shown(self, path: Path) -> str:
        """
        Return a path as the model may see it: relative to its root.

        Notes
        -----
        **Developer notes.** A tool result enters the model's context, and an
        absolute path usually carries a home directory — a user name. Paths
        are shown relative to the root they lie under.
        """
        for root in self._roots:
            if path == root or root in path.parents:
                return path.relative_to(root).as_posix() or "."
        return path.name

    # -- tools ----------------------------------------------------------

    def _tool_read_file(
        self, arguments: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        path = self._path(_string(arguments, "path"))
        fmt = arguments.get("format")
        encoded = self._guard.cleaner.encode_file(path, fmt)
        kinds = self._guard.cleaner.leaks(encoded.text)
        if kinds:
            raise LeakError(
                f"refused: {len(kinds)} removed value(s) remain in the output",
                kinds=kinds,
            )
        return encoded.text, {
            "format": encoded.format,
            "values": encoded.count,
            "kinds": encoded.report.get("kinds", {}),
        }

    def _tool_write_file(
        self, arguments: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        path = self._path(_string(arguments, "path"), must_exist=False)
        text = arguments.get("text")
        if not isinstance(text, str):
            raise _ToolError("argument 'text' must be a string")
        overwrite = arguments.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise _ToolError("argument 'overwrite' must be true or false")
        if path.exists() and not overwrite:
            raise _ToolError(
                f"{self._shown(path)!r} exists; pass overwrite: true to replace it"
            )
        if path.is_dir():
            raise _ToolError(f"{self._shown(path)!r} is a folder")
        outcome = self._guard.cleaner.decode_report(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(outcome.text.encode("utf-8"))
        data = {
            "path": self._shown(path),
            "restored": len(outcome.restored),
            "unknown": list(outcome.unknown),
        }
        note = f"wrote {self._shown(path)} ({len(outcome.restored)} placeholder(s) restored"
        note += (
            f"; not issued here and left as written: {', '.join(outcome.unknown)})"
            if outcome.unknown
            else ")"
        )
        return note, data

    def _tool_encode_text(
        self, arguments: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        text = arguments.get("text")
        if not isinstance(text, str):
            raise _ToolError("argument 'text' must be a string")
        fmt = _string(arguments, "format", "text")
        safe = self._guard.outgoing(text, fmt)
        return safe, None

    def _tool_encode_folder(
        self, arguments: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        source = self._path(_string(arguments, "source"))
        target = self._path(_string(arguments, "target"), must_exist=False)
        items = [
            {
                "status": item.status,
                "file": item.relative,
                "output": item.output,
                "values": item.count,
                "reason": item.reason,
            }
            for item in self._guard.cleaner.encode_tree(source, target)
        ]
        counts = {
            status: sum(1 for i in items if i["status"] == status)
            for status in ("encoded", "skipped", "refused")
        }
        lines = [
            f"{i['status']:<8} {i['file']}"
            + (f"  ({i['reason']})" if i["reason"] else "")
            for i in items
        ]
        return "\n".join(
            [
                f"wrote {self._shown(target)}: "
                + ", ".join(f"{v} {k}" for k, v in counts.items()),
                *lines,
            ]
        ), {"items": items, **counts}

    def _tool_inspect(self, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        from ._runtime import Cleaner  # ruff: ignore[import-outside-top-level]

        # A scratch cleaner: inspecting never changes this session's vault.
        probe = Cleaner(self._guard.cleaner.plan)
        try:
            path = (
                self._path(_string(arguments, "path")) if "path" in arguments else None
            )
            if path is not None and path.is_dir():
                return _folder_summary(probe.survey_tree(path))
            if path is not None:
                encoded = probe.encode_file(path, arguments.get("format"))
            elif isinstance(arguments.get("text"), str):
                encoded = probe.encode_text(
                    arguments["text"], _string(arguments, "format", "text")
                )
            else:
                raise _ToolError("give 'path' or 'text'")
            kinds = encoded.report.get("kinds", {})
        finally:
            probe.clear()
        summary = (
            ", ".join(f"{kind}={count}" for kind, count in kinds.items()) or "nothing"
        )
        return f"{sum(kinds.values())} value(s) would be hidden: {summary}", {
            "kinds": kinds
        }

    def _tool_forget(self, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        del arguments
        self._guard.clear()
        self._guard = self._make_guard()
        return "forgotten: every value this server held is gone", {"forgotten": True}

    def close(self) -> None:
        """Drop every value held."""
        self._guard.clear()


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def serve(server: McpServer, stdin: IO[str], stdout: IO[str]) -> int:
    """
    Serve MCP over newline-delimited JSON until standard input closes.

    Parameters
    ----------
    server : McpServer
        Answers each message.
    stdin, stdout : file-like
        The transport.

    Returns
    -------
    int
        ``0`` when the client closes the connection.

    Notes
    -----
    **Developer notes.** A line that is not JSON gets a parse error with a
    ``null`` id, as JSON-RPC requires, and the loop continues: one bad message
    from a client must not end the session and drop the vault. The vault is
    cleared when the loop ends, however it ends.
    """
    try:
        for line in stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except ValueError:
                reply: Any = _error(None, _PARSE_ERROR, "not valid JSON")
            else:
                reply = server.handle(message)
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()
    finally:
        server.close()
    return 0
