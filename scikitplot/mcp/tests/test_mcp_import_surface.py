"""
Release-contract regression gates (MCP-CLOSE-001 / MCP-CLOSE-002).

The base import surface must not require pydantic (server-only dep), and the
advertised `[mcp]` extra must declare its real direct dependencies.
"""
import pathlib
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # contains scikitplot/


def test_import_and_discovery_without_pydantic():
    # MCP-CLOSE-002: `import scikitplot.mcp` + SDK-free discovery must work with
    # pydantic absent. Robust to a parent interpreter that preloaded pydantic: the
    # subprocess clears any preloaded pydantic modules, then installs an import
    # blocker, so a fresh import that needed pydantic would raise (and fail).
    probe = (
        "import builtins, sys\n"
        "for _m in [m for m in sys.modules if m == 'pydantic' or m.startswith('pydantic.')]:\n"
        "    del sys.modules[_m]\n"
        "_real = builtins.__import__\n"
        "def _block(name, *a, **k):\n"
        "    if name.split('.')[0] == 'pydantic':\n"
        "        raise ImportError('pydantic blocked for test')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _block\n"
        "import scikitplot.mcp as M\n"
        "assert M.server_runtime_status()['retrieval_available'] is True\n"
        "assert M.server_capabilities()['effect_class'] == 'read_only'\n"
        "from scikitplot.mcp._hybrid import HybridRetriever\n"
        "assert 'pydantic' not in sys.modules, 'pydantic was imported by the base surface'\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert r.returncode == 0, (
        f"base import surface required pydantic: {r.stdout!r} {r.stderr[-500:]!r}"
    )


def test_cli_introspection_without_pydantic():
    # MCP-CLOSE-R3: --help / --print-effective-config / --list-capabilities must run
    # without importing the pydantic-backed server tier.
    for argv in (["--help"], ["--print-effective-config"], ["--list-capabilities"]):
        probe = (
            "import builtins, sys\n"
            "for _m in [m for m in sys.modules if m == 'pydantic' or m.startswith('pydantic.')]:\n"
            "    del sys.modules[_m]\n"
            "_real = builtins.__import__\n"
            "def _block(name, *a, **k):\n"
            "    if name.split('.')[0] in ('pydantic', 'mcp', 'mcp_types'):\n"
            "        raise ImportError('server-tier blocked for test')\n"
            "    return _real(name, *a, **k)\n"
            "builtins.__import__ = _block\n"
            "from scikitplot.mcp.__main__ import main\n"
            f"sys.exit(main({argv!r}))\n"
        )
        r = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        assert r.returncode == 0, (
            f"`python -m scikitplot.mcp {' '.join(argv)}` needed the server tier: "
            f"{r.stdout[-200:]!r} {r.stderr[-500:]!r}"
        )


def test_searchservice_lazily_requires_pydantic_but_is_reachable():
    # SearchService is a server-tier export: reachable from the package, backed by
    # pydantic (present in this environment).
    import scikitplot.mcp as M
    from scikitplot.mcp._demo import builtin_demo_retriever
    out = M.SearchService(builtin_demo_retriever()).search("doc", 3)
    assert out.count == len(out.passages) == len(out.citations)


def test_mcp_extra_declares_server_dependencies():
    # MCP-CLOSE-001: the advertised extra must declare pydantic + the SDK range.
    pp = _REPO_ROOT / "pyproject.toml"
    if not pp.exists():
        pytest.skip("pyproject.toml not present in this layout")
    text = pp.read_text()
    start = text.find("\nmcp = [")
    if start == -1:
        pytest.fail("no [mcp] optional-dependency group found")
    # Terminate at the array-closing ']' on its own line, not at a ']' inside a
    # comment (e.g. the literal "[mcp]" mentioned in a comment).
    body = text[start:]
    end = body.find("\n]")
    block = body[: end + 2] if end != -1 else body
    assert "pydantic" in block, "[mcp] extra must declare pydantic (server model dep)"
    assert "mcp>=2.0.0,<3" in block, "[mcp] extra must pin the SDK range mcp>=2.0.0,<3"
    assert 'python_version < "3.10"' not in block, "no <3.10 SDK fallback allowed"
    # MCP-CLOSE-R2: the SDK line must carry NO python marker, so [mcp] fails to
    # resolve on <3.10 instead of leaving a partial (pydantic-only) extra.
    import re as _re
    # MCP-M00-10, adjudicated in run M01 and re-enabled here: a marker makes pip
    # DROP the SDK on 3.8/3.9 and install a pydantic-only extra that cannot
    # serve. Without it the SDK's own Requires-Python rejects the interpreter
    # and resolution fails loudly, which is the intended behaviour.
    sdk_lines = [l for l in block.splitlines() if 'mcp>=2.0.0,<3' in l]
    assert sdk_lines and all('python_version' not in l for l in sdk_lines), (
        'the mcp SDK requirement must not carry a python_version marker')
    # M01-01: starlette is imported directly by the server layer, so it must be
    # declared rather than inherited from the SDK transitively.
    assert any('starlette' in l for l in block.splitlines()), (
        'the [mcp] extra must declare starlette explicitly')
