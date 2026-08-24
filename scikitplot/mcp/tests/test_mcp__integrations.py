"""Integrations are read-only, SDK-free, and emit valid config for our server."""
import json

from .._integrations._agno import ScikitplotDocsToolkit
from .._integrations._openclaw import OpenClawMcpConfig


def test_docs_toolkit_is_sdk_free_and_read_only():
    tk = ScikitplotDocsToolkit()                       # no MCP SDK, no framework
    out = tk.search_docs("documentation", k=5)
    assert set(out) == {"query", "count", "passages", "citations", "message"}
    assert out["count"] == len(out["passages"]) == len(out["citations"])  # invariant
    assert isinstance(out["passages"], list)


def test_openclaw_config_targets_our_server():
    cfg = OpenClawMcpConfig()
    d = cfg.to_dict()
    entry = d["mcpServers"]["scikitplot-docs"]
    assert entry["command"] == "python"
    assert entry["args"] == ["-m", "scikitplot.mcp"]
    assert entry["transport"] == "stdio"
    json.loads(cfg.to_json())                          # valid JSON


def test_openclaw_config_rejects_bad_transport():
    import pytest
    with pytest.raises(ValueError):
        OpenClawMcpConfig(transport="carrier-pigeon")


def test_build_agno_toolkit_actionable_without_agno():
    # agno is not installed here; the builder must raise a clear ImportError.
    import pytest
    from .._integrations._agno import build_agno_toolkit
    with pytest.raises(ImportError, match="agno"):
        build_agno_toolkit()


def test_importing_integrations_pulls_no_sdk_or_framework():
    # Import the integration modules in a CLEAN interpreter and assert that neither
    # the MCP SDK nor any agent framework was imported as a side effect. Subprocess
    # isolation avoids sys.modules pollution from other tests.
    import pathlib
    import subprocess
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[3]  # dir containing scikitplot/
    probe = (
        "import sys\n"
        "import scikitplot.mcp._integrations._agno\n"
        "import scikitplot.mcp._integrations._openclaw\n"
        "leaked = [m for m in ('mcp', 'mcp_types', 'agno', 'starlette') "
        "if m in sys.modules]\n"
        "sys.stdout.write('LEAKED=' + ','.join(leaked))\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert result.returncode == 0, (
        f"importing integrations leaked SDK/framework modules: "
        f"{result.stdout!r} {result.stderr[-400:]!r}"
    )
