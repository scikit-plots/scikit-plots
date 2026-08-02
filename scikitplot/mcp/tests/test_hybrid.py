# tests/test_hybrid.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scikitplot.mcp hybrid retrieval (RRF fusion, BM25 leg)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from scikitplot.mcp import (  # noqa: E402
    Bm25Retriever,
    HybridRetriever,
    RetrievedChunk,
    build_search_docs_result,
    reciprocal_rank_fusion,
)


def _chunk(doc_id, text="t", uri=None, title="", anchor=""):
    return RetrievedChunk(text=text, source_uri=uri or f"https://d.io/{doc_id}",
                          doc_id=doc_id, title=title, anchor=anchor)


class _Fixed:
    """Retriever returning a fixed ranked list."""
    def __init__(self, chunks):
        self._c = chunks

    def search(self, query, k=5):
        return self._c[:k]


class _Boom:
    def search(self, query, k=5):
        raise RuntimeError("backend down")


# ── RRF primitive ────────────────────────────────────────────────────────────
def test_rrf_sums_across_lists():
    a = [_chunk("x"), _chunk("y")]      # x rank1, y rank2
    b = [_chunk("y"), _chunk("z")]      # y rank1, z rank2
    fused = reciprocal_rank_fusion([(1.0, a), (1.0, b)], k=60)
    # y appears in both → highest fused score
    top = max(fused, key=fused.get)
    assert top == "y"
    assert fused["y"] > fused["x"] and fused["y"] > fused["z"]


# ── HybridRetriever ──────────────────────────────────────────────────────────
def test_hybrid_boosts_agreement():
    dense = _Fixed([_chunk("A"), _chunk("B"), _chunk("C")])
    bm25 = _Fixed([_chunk("C"), _chunk("A"), _chunk("D")])
    h = HybridRetriever([dense, bm25])
    out = h.search("q", k=3)
    ids = [c.doc_id for c in out]
    # A and C are found by both legs → they rank above B/D
    assert ids[0] in ("A", "C") and ids[1] in ("A", "C")
    assert len(out) == 3


def test_hybrid_dedup_by_doc_id():
    dense = _Fixed([_chunk("A", title="rich"), _chunk("B")])
    bm25 = _Fixed([_chunk("A"), _chunk("B")])
    h = HybridRetriever([dense, bm25])
    out = h.search("q", k=5)
    ids = [c.doc_id for c in out]
    assert ids.count("A") == 1 and ids.count("B") == 1, "no duplicate doc_ids"
    # richest-metadata representative kept
    a = next(c for c in out if c.doc_id == "A")
    assert a.title == "rich"


def test_hybrid_resilient_to_failing_leg():
    good = _Fixed([_chunk("A"), _chunk("B")])
    h = HybridRetriever([_Boom(), good])
    out = h.search("q", k=2)
    assert [c.doc_id for c in out] == ["A", "B"], "failing leg skipped, good leg used"


def test_hybrid_weighting_shifts_order():
    dense = _Fixed([_chunk("D1"), _chunk("D2")])
    bm25 = _Fixed([_chunk("B1"), _chunk("B2")])
    # Heavily weight bm25 → its top should win overall.
    h = HybridRetriever([dense, bm25], weights=[0.1, 10.0])
    out = h.search("q", k=1)
    assert out[0].doc_id == "B1"


def test_hybrid_empty_query():
    h = HybridRetriever([_Fixed([_chunk("A")])])
    assert h.search("  ", k=3) == []


def test_hybrid_weights_length_validated():
    with pytest.raises(ValueError):
        HybridRetriever([_Fixed([]), _Fixed([])], weights=[1.0])


def test_hybrid_fused_score_set_and_cited():
    dense = _Fixed([_chunk("A", uri="https://d.io/a", anchor="s1")])
    bm25 = _Fixed([_chunk("A", uri="https://d.io/a")])
    h = HybridRetriever([dense, bm25])
    out = h.search("q", k=1)
    assert out[0].score > 0.0, "fused RRF score assigned"
    res = build_search_docs_result("q", out)
    assert res["structuredContent"]["citations"][0]["source_uri"] == "https://d.io/a#s1"


# ── Bm25Retriever (DI) ───────────────────────────────────────────────────────
def test_bm25_retriever_di():
    def fts(query, k):
        return [("d1", 4.2), ("d2", 3.1)][:k]
    def lookup(did):
        return {"d1": {"text": "exact term match", "source_uri": "https://d.io/1", "title": "One"},
                "d2": {"text": "other", "source_uri": "https://d.io/2"}}.get(did, {})
    r = Bm25Retriever(fts, lookup)
    hits = r.search("configure_rate_limit", k=2)
    assert [h.doc_id for h in hits] == ["d1", "d2"]
    assert hits[0].title == "One" and hits[0].source_uri == "https://d.io/1"


def test_bm25_from_corpus_raises_without_deps():
    with pytest.raises(RuntimeError, match="scikitplot.corpus"):
        Bm25Retriever.from_corpus_sqlite("/tmp/store.db")


def test_bm25_empty_query():
    r = Bm25Retriever(lambda q, k: [("x", 1.0)], lambda d: {"text": "t", "source_uri": "https://d.io/x"})
    assert r.search("", k=3) == []
