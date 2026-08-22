"""
MCP-001 regression: create_server distinguishes Python-floor from missing SDK.

The MCP SDK v2 requires Python >= 3.10 (verified against official SDK docs), while
scikit-plots supports >= 3.8. create_server must fail with a version-named message
on < 3.10, and an install-named message on >= 3.10 when the SDK is absent.
"""
import sys

import pytest

from scikitplot.mcp import _server


def test_create_server_requires_python_310(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 9, 0, "final", 0))
    with pytest.raises(RuntimeError, match=r"Python >= 3\.10"):
        _server.create_server(object())


def test_create_server_missing_sdk_message_on_310(monkeypatch):
    # Force the SDK import to fail regardless of what's installed.
    import builtins

    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == "mcp.server" or name.split(".")[0] in {"mcp", "mcp_types"}:
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    # Ensure the version guard passes (this interpreter is >= 3.10 in CI).
    if sys.version_info < (3, 10):
        pytest.skip("interpreter below 3.10; version guard covers this case")

    # M07-03: blocking the import is not by itself "SDK missing" -- create_server
    # now consults the capability probe. Make the probe agree that the SDK is
    # absent, otherwise this simulates a BROKEN install, which is a different
    # message (asserted in the next test).
    monkeypatch.setattr(
        _server,
        "server_runtime_status",
        lambda: {
            "sdk_status": "absent",
            "sdk_version": None,
            "server_available": False,
            "reason": "mcp-sdk-not-installed",
        },
    )
    with pytest.raises(RuntimeError, match=r"mcp>=2\.0\.0,<3"):
        _server.create_server(object())


def test_create_server_distinguishes_broken_install_from_missing(monkeypatch):
    """M07-03: an ImportError with the SDK installed is BROKEN, not ABSENT."""
    import builtins

    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == "mcp.server" or name.split(".")[0] in {"mcp", "mcp_types"}:
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    if sys.version_info < (3, 10):
        pytest.skip("interpreter below 3.10; version guard covers this case")

    monkeypatch.setattr(
        _server,
        "server_runtime_status",
        lambda: {
            "sdk_status": "available",
            "sdk_version": "2.0.0",
            "server_available": True,
            "reason": None,
        },
    )
    with pytest.raises(RuntimeError, match="broken environment"):
        _server.create_server(object())


def test_create_server_reports_an_incompatible_sdk_version(monkeypatch):
    """M03/MCP-D01: an out-of-range SDK must not be reported as missing."""
    import builtins

    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == "mcp.server" or name.split(".")[0] in {"mcp", "mcp_types"}:
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    if sys.version_info < (3, 10):
        pytest.skip("interpreter below 3.10; version guard covers this case")

    monkeypatch.setattr(
        _server,
        "server_runtime_status",
        lambda: {
            "sdk_status": "incompatible",
            "sdk_version": "1.9.0",
            "server_available": False,
            "reason": "mcp-sdk-incompatible",
        },
    )
    with pytest.raises(RuntimeError, match="outside the"):
        _server.create_server(object())
