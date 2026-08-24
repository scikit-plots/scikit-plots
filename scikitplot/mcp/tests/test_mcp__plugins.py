"""
Guard: every generated plugin bundle is valid JSON and points at our server.

Prevents client-config drift (wrong command/args) or invalid JSON shipping.
"""
import json
import pathlib

import pytest

_PLUGINS = pathlib.Path(__file__).resolve().parent.parent / "_plugins"
_JSON = sorted(_PLUGINS.rglob("*.json"))


@pytest.mark.skipif(not _JSON, reason="no plugin bundles present")
@pytest.mark.parametrize("path", _JSON, ids=lambda p: str(p.relative_to(_PLUGINS)))
def test_plugin_json_is_valid(path):
    json.loads(path.read_text())  # raises on invalid JSON


def _iter_server_entries(obj):
    """Yield every MCP server-config dict regardless of the client's wrapper key."""
    if isinstance(obj, dict):
        for key in ("mcpServers", "servers", "cline.mcpServers"):
            block = obj.get(key)
            if isinstance(block, dict):
                yield from block.values()
            elif isinstance(block, list):
                yield from block


def test_all_configs_launch_our_server():
    checked = 0
    for path in _JSON:
        data = json.loads(path.read_text())
        for entry in _iter_server_entries(data):
            args = entry.get("args", [])
            # Command must invoke this module (module form or console command).
            joined = f"{entry.get('command', '')} {' '.join(map(str, args))}"
            assert "scikitplot.mcp" in joined or "scikitplot mcp" in joined, (path, entry)
            checked += 1
    assert checked > 0, "no MCP server config entries found in _plugins/"
