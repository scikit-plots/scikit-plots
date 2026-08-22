# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Guard tests for the M04 invariant ``FAILED retrieval != EMPTY retrieval``.

These are Tier-L tests: they import no MCP SDK and no pydantic, so they collect
and run on a base install (Python 3.8+).

Notes
-----
**Developer-focused.** Every assertion here corresponds to a review finding, so
a regression shows up as a named failing test rather than as a silently wrong
answer on the wire. See ``_maintenance/checkpoints/M04_CORPUS_NEUTRAL_RESULT.md``.
"""

from __future__ import annotations

import pytest

from scikitplot.mcp._core import build_search_docs_result
from scikitplot.mcp._hybrid import Bm25Retriever, HybridRetriever
from scikitplot.mcp._outcome import (
    DEGRADED,
    EMPTY,
    FAILED,
    SUCCESS,
    LegRecord,
    RetrievalOutcome,
    status_of,
)


def _lookup(doc_id: str) -> dict:
    return {"text": "hello world", "source_uri": "https://example.test/a", "title": "T"}


def _boom(query: str, k: int):
    raise RuntimeError("index unavailable")


def _one_hit(query: str, k: int):
    return [("d1", 1.0)]


def _no_hits(query: str, k: int):
    return []


# --------------------------------------------------------------------------
# RetrievalOutcome behaves exactly like the list it replaced
# --------------------------------------------------------------------------


def test_outcome_is_a_real_list_so_existing_callers_are_unaffected():
    chunk = object()
    outcome = RetrievalOutcome([chunk])
    assert isinstance(outcome, list)
    assert len(outcome) == 1
    assert outcome[0] is chunk
    assert list(iter(outcome)) == [chunk]
    assert bool(outcome) is True
    assert bool(RetrievalOutcome([])) is False


def test_status_is_derived_not_assumed():
    assert RetrievalOutcome([]).status == EMPTY
    assert RetrievalOutcome([object()]).status == SUCCESS
    failed = LegRecord("dense", FAILED, error="boom")
    assert RetrievalOutcome([], legs=[failed]).status == FAILED
    ran = LegRecord("lexical", EMPTY)
    assert RetrievalOutcome([], legs=[ran, failed]).status == DEGRADED


def test_degraded_or_failed_leg_must_explain_itself():
    with pytest.raises(ValueError, match="without an error explanation"):
        LegRecord("dense", FAILED)
    with pytest.raises(ValueError, match="without an error explanation"):
        LegRecord("dense", DEGRADED)


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="unknown retrieval"):
        LegRecord("dense", "sort-of-ok")
    with pytest.raises(ValueError, match="unknown retrieval status"):
        RetrievalOutcome([], status="mostly-fine")


def test_status_of_tolerates_a_plain_list():
    """A retriever that still returns a bare list keeps its previous meaning."""
    assert status_of([]) == EMPTY
    assert status_of([object()]) == SUCCESS
    assert status_of(None) == EMPTY


# --------------------------------------------------------------------------
# The invariant itself
# --------------------------------------------------------------------------


def test_failed_retrieval_is_not_reported_as_empty():
    """MCP-D03 / M04: the wire must not claim 'no matching documentation'."""
    outcome = Bm25Retriever(_boom, _lookup).search("transport", 5)
    assert outcome.status == FAILED
    assert len(outcome) == 0

    result = build_search_docs_result("transport", outcome)
    message = result["structuredContent"]["message"]
    assert "No matching documentation was found for this query." not in message
    assert "failed" in message.lower()
    assert result["isError"] is True
    assert result["structuredContent"]["retrieval_status"] == FAILED
    assert result["structuredContent"]["retrieval_errors"]


def test_genuinely_empty_retrieval_keeps_the_original_message():
    outcome = Bm25Retriever(_no_hits, _lookup).search("transport", 5)
    assert outcome.status == EMPTY
    result = build_search_docs_result("transport", outcome)
    assert (
        result["structuredContent"]["message"]
        == "No matching documentation was found for this query."
    )
    assert result["isError"] is False


def test_legacy_bare_list_still_produces_the_original_response():
    """Backward compatibility: unmigrated callers must not change behaviour."""
    result = build_search_docs_result("transport", [])
    assert (
        result["structuredContent"]["message"]
        == "No matching documentation was found for this query."
    )
    assert result["isError"] is False
    assert result["structuredContent"]["retrieval_status"] == EMPTY


# --------------------------------------------------------------------------
# Fusion must not flatten a nested failure
# --------------------------------------------------------------------------


def test_fusion_reports_failed_only_when_every_leg_failed():
    fail = Bm25Retriever(_boom, _lookup)
    assert HybridRetriever([fail, fail]).search("q", 5).status == FAILED


def test_fusion_reports_degraded_when_some_legs_survive():
    fail = Bm25Retriever(_boom, _lookup)
    good = Bm25Retriever(_one_hit, _lookup)
    empty = Bm25Retriever(_no_hits, _lookup)
    assert HybridRetriever([fail, good]).search("q", 5).status == DEGRADED
    assert HybridRetriever([fail, empty]).search("q", 5).status == DEGRADED


def test_fusion_is_clean_when_all_legs_run():
    good = Bm25Retriever(_one_hit, _lookup)
    empty = Bm25Retriever(_no_hits, _lookup)
    assert HybridRetriever([good, good]).search("q", 5).status == SUCCESS
    assert HybridRetriever([empty, empty]).search("q", 5).status == EMPTY


def test_fusion_does_not_discard_a_falsy_failed_leg():
    """
    Regression: ``search(...) or []`` silently replaced a FAILED outcome.

    An empty :class:`RetrievalOutcome` is falsy, so the old ``or []`` guard
    swapped it for a plain list and lost the status. This is the exact
    flattening the M04 invariant forbids.
    """
    outcome = HybridRetriever([Bm25Retriever(_boom, _lookup)]).search("q", 5)
    assert outcome.status == FAILED, "a falsy failed leg was flattened to empty"


# --------------------------------------------------------------------------
# Vocabulary must not drift from Corpus
# --------------------------------------------------------------------------


def test_status_vocabulary_matches_corpus_when_corpus_is_installed():
    corpus = pytest.importorskip("scikitplot.corpus")
    from scikitplot.mcp._outcome import assert_matches_corpus, to_corpus_status

    assert_matches_corpus()
    assert to_corpus_status(FAILED) is corpus.RetrievalStatus.FAILED
    assert to_corpus_status(EMPTY) is corpus.RetrievalStatus.EMPTY
