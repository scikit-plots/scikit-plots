# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Emit the gateway config that wires the scikit-plots docs MCP server into a host.

This is the recommended, zero-extra-code path: the read-only ``search_docs`` tool
becomes available to the agent host with no adapter code. Pure stdlib; imports no
framework and no MCP SDK.
"""

from __future__ import annotations

import json


class OpenClawMcpConfig:
    """
    Build the MCP-server config entry for the scikit-plots docs server.

    Parameters
    ----------
    name : str, optional
        Config key / server id (default ``"scikitplot-docs"``).
    command : str, optional
        Launch command (default ``"python"``).
    args : list of str, optional
        Launch arguments (default ``["-m", "scikitplot.mcp"]``). The equivalent
        console form is ``command="scikitplot", args=["mcp"]``.
    transport : str, optional
        MCP transport, ``"stdio"`` (default) or ``"streamable-http"``.

    Examples
    --------
    >>> print(OpenClawMcpConfig().to_json())  # doctest: +ELLIPSIS
    {
      "mcpServers": {
        "scikitplot-docs": {
    ...
    """

    def __init__(
        self,
        *,
        name: str = "scikitplot-docs",
        command: str = "python",
        args: list[str] | None = None,
        transport: str = "stdio",
    ) -> None:
        if transport not in ("stdio", "streamable-http"):
            raise ValueError("transport must be 'stdio' or 'streamable-http'")
        self.name = name
        self.command = command
        self.args = list(args) if args is not None else ["-m", "scikitplot.mcp"]
        self.transport = transport

    def to_dict(self) -> dict:
        """Return the config mapping (``{"mcpServers": {name: {...}}}``)."""
        return {
            "mcpServers": {
                self.name: {
                    "command": self.command,
                    "args": list(self.args),
                    "transport": self.transport,
                }
            }
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return the config as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
