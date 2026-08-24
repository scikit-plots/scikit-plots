"""
Enforce the single-MCP-wire-protocol invariant (RULESET INV-8; policy #2/#6/#7).

scikit-plots SHALL have exactly ONE implementation of the MCP wire protocol: the
official-SDK integration in `_server.py`. Python 3.8/3.9 keep the SDK-free
retrieval core (Tier L) but MUST NOT gain a hand-rolled JSON-RPC "MCP fallback".
This test fails loudly if a future change reintroduces the rejected pattern.
"""
import ast
import pathlib

import pytest

_MCP_DIR = pathlib.Path(__file__).resolve().parent.parent  # scikitplot/mcp
_SDK_ROOTS = {"mcp", "mcp_types"}

# Names that would signal a resurrected hand-rolled fallback server.
_FORBIDDEN_MODULE_NAMES = {
    "_legacy_jsonrpc_server.py", "_py38_mcp.py", "_fallback_protocol.py",
    "_mcp_fallback.py", "_jsonrpc_server.py",
}


def _source_files():
    for p in _MCP_DIR.rglob("*.py"):
        parts = set(p.parts)
        if "tests" in parts or "_maintenance" in parts:
            continue
        yield p


def _imports_official_sdk(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _SDK_ROOTS:
                return True
        elif isinstance(node, ast.Import):
            if any(a.name.split(".")[0] in _SDK_ROOTS for a in node.names):
                return True
    return False


def test_no_forbidden_fallback_modules():
    present = {p.name for p in _MCP_DIR.rglob("*.py")} & _FORBIDDEN_MODULE_NAMES
    assert not present, f"forbidden hand-rolled MCP fallback module(s): {sorted(present)}"


def test_exactly_one_module_integrates_the_sdk():
    importers = sorted(p.name for p in _source_files() if _imports_official_sdk(p))
    assert importers == ["_server.py"], (
        f"the MCP SDK must be integrated in exactly one module (_server.py); "
        f"found: {importers}"
    )


def test_no_handrolled_jsonrpc_branded_as_mcp():
    # Rejected anti-pattern: an in-package JSON-RPC method table (protocolVersion +
    # tools/list dispatch) that duplicates the SDK. Reject it outside _server.py.
    offenders = []
    for p in _source_files():
        src = p.read_text()
        if '"jsonrpc"' in src and "protocolVersion" in src and "tools/list" in src:
            offenders.append(p.name)
    assert not offenders, f"hand-rolled JSON-RPC MCP dispatch found in: {offenders}"


def test_sdk_import_is_lazy_not_module_level():
    # INV-1: importing scikitplot.mcp must not import the SDK. The only SDK
    # importer (_server.py) must do so inside a function, never at module scope.
    server = _MCP_DIR / "_server.py"
    tree = ast.parse(server.read_text())
    for node in ast.walk(tree):
        is_sdk_import = (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] in _SDK_ROOTS
        )
        if is_sdk_import:
            assert node.col_offset > 0, (
                "MCP SDK import must be lazy (inside a function), not module-level"
            )
