"""Identical passages are sent once, citing every source (finding CX-02)."""

from __future__ import annotations

from scikitplot.mcp import _core


def _chunk(doc_id, text, uri=None, score=1.0):
    return _core.RetrievedChunk(
        doc_id=doc_id, title=doc_id, source_uri=uri or f"https://d.io/{doc_id}",
        anchor="", text=text, score=score,
    )


def _structured(chunks, **kwargs):
    return _core.build_search_docs_result("q", chunks, **kwargs)["structuredContent"]


def test_identical_text_is_sent_once_and_cites_every_source():
    out = _structured([_chunk("a", "Licensed under BSD."), _chunk("b", "Licensed under BSD.")])
    assert out["count"] == 1 and out["duplicates_merged"] == 1
    assert out["citations"][0]["doc_id"] == "a"
    assert out["citations"][0]["also_in"] == [{"source_uri": "https://d.io/b", "doc_id": "b"}]


def test_whitespace_differences_are_the_same_text():
    out = _structured([_chunk("a", "one  two\nthree"), _chunk("b", " one two three ")])
    assert out["count"] == 1 and out["duplicates_merged"] == 1


def test_a_merged_copy_does_not_use_up_the_limit():
    chunks = [_chunk("a", "same"), _chunk("b", "same"), _chunk("c", "other")]
    out = _structured(chunks, max_results=2)
    assert [c["doc_id"] for c in out["citations"]] == ["a", "c"]


def test_different_text_is_never_merged():
    out = _structured([_chunk("a", "roc_auc_score"), _chunk("b", "roc_auc_scores")])
    assert out["count"] == 2 and out["duplicates_merged"] == 0
    assert all("also_in" not in c for c in out["citations"])


def test_order_stays_best_first():
    chunks = [_chunk("a", "x"), _chunk("b", "y"), _chunk("c", "x"), _chunk("d", "z")]
    out = _structured(chunks)
    assert [c["doc_id"] for c in out["citations"]] == ["a", "b", "d"]
    assert [c["n"] for c in out["citations"]] == [1, 2, 3]


def test_the_typed_server_path_carries_the_merge():
    """The closed wire models accept ``also_in`` and ``duplicates_merged``."""
    import pytest

    server = pytest.importorskip("scikitplot.mcp._server")

    class Twice(_core.DocsRetriever):
        def search(self, query, k=5):
            return [_chunk("a", "Licensed under BSD."), _chunk("b", "Licensed under BSD."), _chunk("c", "x")]

    out = server.SearchService(Twice()).search("licence", 3)
    assert out.count == 2 and out.duplicates_merged == 1
    assert [a.doc_id for a in out.citations[0].also_in] == ["b"]
    assert out.citations[1].also_in == []
