"""
SDK-free capability probe + guaranteed SDK-free retrieval (Tier-L, Python 3.8+).

The MCP *protocol server* is Tier S (Python >= 3.10 + mcp>=2,<3). The retrieval
layer is Tier L and must work with no MCP SDK present. `server_runtime_status()`
lets callers detect this without importing the SDK or catching an exception.
"""
import sys

from scikitplot.mcp._server import SearchService, server_runtime_status
from scikitplot.mcp._demo import builtin_demo_retriever


def test_status_shape_and_consistency():
    st = server_runtime_status()
    # M03/MCP-D01: the contract now carries version and compatibility, because
    # presence alone cannot decide whether the SDK is usable.
    assert set(st) == {
        "retrieval_available", "server_available", "python_ok",
        "python", "sdk_present", "sdk_version", "sdk_compatible",
        "sdk_status", "reason",
    }
    assert st["retrieval_available"] is True                       # always
    assert st["python_ok"] == (sys.version_info >= (3, 10))
    # Availability requires a COMPATIBLE sdk, not merely a present one.
    assert st["server_available"] == bool(
        st["python_ok"] and st["sdk_status"] == "available"
    )
    assert st["sdk_status"] in {
        "available", "absent", "broken", "incompatible",
        "misconfigured", "unreachable", "unknown",
    }
    if st["sdk_status"] == "available":
        assert st["sdk_compatible"] is True
        assert st["sdk_version"] is not None
    if st["server_available"]:
        assert st["reason"] is None
    else:
        assert st["reason"] in {
            "python<3.10", "mcp-sdk-not-installed", "mcp-sdk-incompatible",
            "mcp-sdk-broken", "mcp-sdk-status-unknown",
        }


def test_probe_is_immune_to_a_shadowing_directory(tmp_path, monkeypatch):
    """
    MCP-D01: a directory named ``mcp/`` on sys.path must not fake a server.

    ``find_spec`` answered "is a module of this name importable", which any such
    directory satisfies. Distribution metadata answers the question that matters.
    """
    (tmp_path / "mcp").mkdir()
    (tmp_path / "mcp" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    st = server_runtime_status()
    if st["sdk_status"] == "available":
        # A real SDK is installed; the shadow must not change the reported version.
        import importlib.metadata as md
        assert st["sdk_version"] == md.version("mcp")
    else:
        assert st["server_available"] is False


def test_failed_probe_reports_unknown_not_absent(monkeypatch):
    """M03-03: a probe that did not complete must not claim 'not installed'."""
    import importlib.metadata as md
    from scikitplot.mcp import _capabilities as caps

    def boom(name):
        raise OSError("distribution metadata unreadable")

    monkeypatch.setattr(md, "version", boom)
    status, detected = caps._probe_sdk()
    assert status == "unknown"
    assert detected is None


def test_probe_does_not_import_the_sdk():
    # Calling the probe must not import mcp/mcp_types (find_spec only).
    for mod in ("mcp", "mcp_types"):
        sys.modules.pop(mod, None)
    server_runtime_status()
    # If the SDK is genuinely absent, it must not have been imported by the probe.
    import importlib.util
    if importlib.util.find_spec("mcp") is None:
        assert "mcp" not in sys.modules


def test_sdk_free_retrieval_is_alive():
    # The native search API works with no MCP SDK installed (Tier L guarantee).
    out = SearchService(builtin_demo_retriever()).search("documentation", k=5)
    assert out.count == len(out.passages) == len(out.citations)   # invariant
    assert isinstance(out.passages, list)


def test_capability_vocabulary_matches_corpus_when_corpus_is_installed():
    """M03: MCP must consume Corpus's CapabilityStatus, not define a rival one."""
    import pytest

    corpus = pytest.importorskip("scikitplot.corpus")
    from scikitplot.mcp._capabilities import (
        assert_capability_vocabulary_matches_corpus,
    )

    assert_capability_vocabulary_matches_corpus()
    assert {m.value for m in corpus.CapabilityStatus} >= {
        "available",
        "absent",
        "broken",
        "incompatible",
        "unknown",
    }


def test_incompatible_sdk_is_not_reported_as_absent(monkeypatch):
    """MCP-D01: presence is not compatibility; an out-of-range SDK is INCOMPATIBLE."""
    import importlib.metadata as md
    from scikitplot.mcp import _capabilities as caps

    monkeypatch.setattr(md, "version", lambda name: "1.9.0" if name == "mcp" else "0")
    status, detected = caps._probe_sdk()
    assert status == "incompatible"
    assert detected == "1.9.0"

    monkeypatch.setattr(md, "version", lambda name: "3.0.0" if name == "mcp" else "0")
    assert caps._probe_sdk()[0] == "incompatible"
