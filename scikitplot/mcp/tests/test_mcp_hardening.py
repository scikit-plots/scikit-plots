"""
Redesign-wave regression gates: capabilities discovery (SDK-free), SDK-drift
fail-closed (M1/MCP-002), concurrency contract (M3), provenance survives fusion (A3).
All SDK-free so they run on the Legacy Retrieval tier.
"""
import threading
import time

import pytest

from scikitplot.mcp._core import RetrievedChunk
from scikitplot.mcp._server import (
    SearchService,
    server_capabilities,
    _forbid_unknown_tool_arguments,
)


# --- capabilities discovery (SDK-free) ---
def test_capabilities_is_read_only_and_sdk_free():
    import sys
    caps = server_capabilities()
    assert caps["effect_class"] == "read_only"
    assert [t["name"] for t in caps["tools"]] == ["search_docs"]
    assert all(t["read_only"] for t in caps["tools"])
    assert caps["resources"][0]["uri_template"] == "docs://chunk/{doc_id}"
    import importlib.util
    if importlib.util.find_spec("mcp") is None:
        assert "mcp" not in sys.modules  # discovery never imports the SDK


# --- M1 / MCP-002: SDK-drift must fail closed, never silently pass ---
def test_forbid_unknown_args_fails_closed_on_missing_tool_manager():
    class FakeServerNoManager:
        pass  # no _tool_manager -> the strict-args seam is gone
    with pytest.raises(RuntimeError, match="per-tool validation metadata"):
        _forbid_unknown_tool_arguments(FakeServerNoManager(), "search_docs")


def test_forbid_unknown_args_fails_closed_on_missing_arg_model():
    class _Tool:
        fn_metadata = object()          # has no arg_model
    class _Mgr:
        def get_tool(self, name):
            return _Tool()
    class FakeServer:
        _tool_manager = _Mgr()
    with pytest.raises(RuntimeError, match="Pydantic argument model"):
        _forbid_unknown_tool_arguments(FakeServer(), "search_docs")


# --- M3: concurrency is bounded; a failing/slow leg does not corrupt output ---
class _SlowRetriever:
    def __init__(self, delay=0.02):
        self.delay = delay
        self.peak = 0
        self._live = 0
        self._lock = threading.Lock()

    def search(self, query, k=5):
        with self._lock:
            self._live += 1
            self.peak = max(self.peak, self._live)
        try:
            time.sleep(self.delay)
            return [RetrievedChunk(text=f"hit for {query}", source_uri="https://example/x",
                                   score=1.0, doc_id="d1")]
        finally:
            with self._lock:
                self._live -= 1


def test_concurrency_is_bounded_by_max_concurrency():
    r = _SlowRetriever()
    svc = SearchService(r, max_concurrency=2, acquire_timeout_seconds=5.0)
    errors, results = [], []

    def worker():
        try:
            results.append(svc.search("q", 3))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 8
    assert r.peak <= 2, f"observed concurrency {r.peak} exceeded max_concurrency=2"


class _FailingRetriever:
    def search(self, query, k=5):
        raise RuntimeError("backend down")


def test_failing_leg_surfaces_cleanly_without_corruption():
    svc = SearchService(_FailingRetriever(), max_concurrency=2, acquire_timeout_seconds=5.0)
    with pytest.raises(RuntimeError):
        svc.search("q", 3)
    # a subsequent good call on a fresh service still works (no shared corruption)
    ok = SearchService(_SlowRetriever(delay=0.0)).search("q", 3)
    assert ok.count == len(ok.passages) == len(ok.citations)


# --- A3: provenance (doc_id, source_uri) survives fusion + dedup unchanged ---
def test_provenance_survives_fusion():
    from scikitplot.mcp._hybrid import HybridRetriever

    class _Leg:
        def __init__(self, chunks):
            self._chunks = chunks
        def search(self, query, k=5):
            return list(self._chunks[:k])

    leg1 = _Leg([
        RetrievedChunk(text="alpha", source_uri="https://docs/a", score=0.9, doc_id="A", title="A-title"),
        RetrievedChunk(text="beta",  source_uri="https://docs/b", score=0.8, doc_id="B"),
    ])
    leg2 = _Leg([
        RetrievedChunk(text="alpha", source_uri="https://docs/a", score=0.7, doc_id="A"),  # dup of A
        RetrievedChunk(text="gamma", source_uri="https://docs/c", score=0.6, doc_id="C"),
    ])
    fused = HybridRetriever([leg1, leg2]).search("q", k=10)
    by_id = {c.doc_id: c for c in fused}
    assert set(by_id) == {"A", "B", "C"}                 # dedup kept one A across legs
    assert by_id["A"].source_uri == "https://docs/a"     # provenance intact
    assert by_id["B"].source_uri == "https://docs/b"
    assert by_id["C"].source_uri == "https://docs/c"
    assert by_id["A"].title == "A-title"                 # richest copy retained


# --- --list-capabilities CLI (SDK-free discovery from the terminal) ---
def test_list_capabilities_cli_is_sdk_free(capsys):
    import json as _json
    from scikitplot.mcp.__main__ import main
    code = main(["--list-capabilities"])
    assert code == 0
    caps = _json.loads(capsys.readouterr().out)
    assert caps["effect_class"] == "read_only"
    assert [t["name"] for t in caps["tools"]] == ["search_docs"]


# --- R5: static vs effective capabilities (resource conditioned on document_reader) ---
def test_static_vs_effective_capabilities():
    from scikitplot.mcp._capabilities import (
        effective_server_capabilities,
        server_capabilities,
    )
    static = server_capabilities()
    assert static["kind"] == "static"
    # static advertises the resource as *potential*, gated by document_reader
    res = static["resources"][0]
    assert res["requires"] == ["document_reader"]

    # no document_reader -> effective surface has NO resource (matches create_server)
    eff_off = effective_server_capabilities(document_reader_enabled=False)
    assert eff_off["kind"] == "effective"
    assert eff_off["resources"] == []

    # with document_reader -> resource is present
    eff_on = effective_server_capabilities(document_reader_enabled=True)
    assert [r["uri_template"] for r in eff_on["resources"]] == ["docs://chunk/{doc_id}"]

    # health only for streamable-http
    assert effective_server_capabilities(transport="stdio")["health"] is None
    assert effective_server_capabilities(transport="streamable-http")["health"] == {"path": "/healthz"}
