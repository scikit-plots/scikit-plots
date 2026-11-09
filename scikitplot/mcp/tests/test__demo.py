"""Tests for scikitplot.mcp._demo."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from .._demo import DemoDocument, InMemoryBm25Retriever, _tokens

#: Documents chosen so that searches meet ties (d1 and d3 are equal), a term
#: held only by a title, repeated terms and a document without any query term.
_DOCS = [
    DemoDocument("d0", "alpha beta beta gamma", "a.md", title="Alpha guide"),
    DemoDocument("d1", "beta gamma", "b.md"),
    DemoDocument("d2", "alpha alpha alpha delta epsilon zeta eta theta", "c.md"),
    DemoDocument("d3", "beta gamma", "d.md"),
    DemoDocument("d4", "unrelated words only", "e.md", title="Omega"),
]

_SETTINGS = [
    {},
    {"k1": 0.5, "b": 0.0},
    {"k1": 2.0, "b": 1.0, "title_weight": 0.0},
    {"k1": 1.2, "b": 0.3, "title_weight": 5.5},
]

_QUERIES = [
    "alpha",
    "beta gamma",
    "gamma beta",
    "alpha alpha beta",
    # A term three times: tripling is not exact, so the order of the
    # multiplications shows in the last bit.
    "alpha alpha alpha beta",
    "gamma gamma gamma beta beta beta alpha alpha alpha",
    "guide omega",
    "alpha beta gamma delta epsilon zeta eta theta unrelated",
    "missing",
    "Alpha. BETA:",
]


def _scan(documents, query, *, k1=1.5, b=0.75, title_weight=2.0):
    """BM25 by its definition: every document, every query term, in order."""
    frequencies, lengths = {}, {}
    for document in documents:
        counts = {t: float(c) for t, c in Counter(_tokens(document.text)).items()}
        for token, count in Counter(_tokens(document.title)).items():
            counts[token] = counts.get(token, 0.0) + title_weight * count
        frequencies[document.doc_id] = counts
        lengths[document.doc_id] = max(1.0, sum(counts.values()))
    total = len(documents)
    average = sum(lengths.values()) / total
    scored = []
    for doc_id, counts in frequencies.items():
        score = 0.0
        norm = 1.0 - b + b * (lengths[doc_id] / average)
        for term, query_frequency in Counter(_tokens(query)).items():
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            df = sum(1 for other in frequencies.values() if other.get(term, 0) > 0)
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm) * query_frequency
        if score > 0 and math.isfinite(score):
            scored.append((score, doc_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(doc_id, score) for score, doc_id in scored]


class TestPostingsEqualTheDefinition:
    """CX-13: postings give the scan's scores and order, to the last bit."""

    @pytest.mark.parametrize("setting", _SETTINGS, ids=repr)
    @pytest.mark.parametrize("query", _QUERIES)
    def test_scores_and_order_equal_the_scan(self, setting, query) -> None:
        index = InMemoryBm25Retriever(_DOCS, **setting)
        want = _scan(_DOCS, query, **setting)
        got = [(hit.doc_id, hit.score) for hit in index.search(query, 50)]
        assert [(d, s.hex()) for d, s in got] == [(d, s.hex()) for d, s in want]

    def test_equal_scores_are_ordered_by_document_id(self) -> None:
        hits = {
            h.doc_id: h for h in InMemoryBm25Retriever(_DOCS).search("beta gamma", 50)
        }
        assert [doc_id for doc_id in hits if doc_id in ("d1", "d3")] == ["d1", "d3"]
        assert hits["d1"].score == hits["d3"].score

    def test_k_limits_the_hits(self) -> None:
        index = InMemoryBm25Retriever(_DOCS)
        want = [d for d, _ in _scan(_DOCS, "alpha beta gamma")]
        for k in (1, 2, 3):
            assert [h.doc_id for h in index.search("alpha beta gamma", k)] == want[:k]

    def test_postings_hold_only_the_documents_with_the_term(self) -> None:
        index = InMemoryBm25Retriever(_DOCS)
        assert index._postings["alpha"][0] == ["d0", "d2"]
        assert index._postings["unrelated"][0] == ["d4"]
        assert "missing" not in index._postings
        assert all(
            len(ids) == len(nums) == len(dens)
            for ids, nums, dens in index._postings.values()
        )

    def test_a_title_only_term_without_weight_is_not_indexed(self) -> None:
        index = InMemoryBm25Retriever(_DOCS, title_weight=0.0)
        assert "omega" not in index._postings
        assert index.search("omega guide", 5) == []
        assert [h.doc_id for h in InMemoryBm25Retriever(_DOCS).search("omega", 5)] == [
            "d4"
        ]
