"""
Structured truncation regressions (slice S-23).

Chunk text was capped and a resource was capped, and in both cases the only
signal was an appended ellipsis inside the text. A consumer cannot distinguish a
truncated passage from one that legitimately ends in an ellipsis, so a bounded
payload now states the limit it applied and what it applied it to.

See Also
--------
scikitplot.mcp._core.build_search_docs_result
"""

import pytest

from .. import _core, _server


def _chunk(text, doc_id="d"):
    return _core.RetrievedChunk(
        doc_id=doc_id, title="T", source_uri="docs://d", anchor="#a",
        text=text, score=0.5,
    )


def test_an_untruncated_result_says_so():
    """The flag states a fact, so it must be present and false when nothing was cut."""
    result = _core.build_search_docs_result("q", [_chunk("short")])
    structured = result["structuredContent"]
    assert structured["truncated"] is False
    assert structured["limits"]["chunk_chars"] == _core.MAX_CHUNK_CHARS


def test_a_truncated_passage_is_reported_structurally():
    """A consumer learns the text was cut without inspecting its last character."""
    result = _core.build_search_docs_result("q", [_chunk("x" * (_core.MAX_CHUNK_CHARS + 50))])
    structured = result["structuredContent"]
    assert structured["truncated"] is True
    assert structured["truncations"][0]["applied_to"] == "chunk_chars"
    assert structured["truncations"][0]["limit"] == _core.MAX_CHUNK_CHARS


def test_the_report_names_the_passage_that_was_cut():
    """With several passages, a caller must know which one lost text."""
    chunks = [_chunk("short", "a"), _chunk("y" * (_core.MAX_CHUNK_CHARS + 10), "b")]
    structured = _core.build_search_docs_result("q", chunks)["structuredContent"]
    reported = {entry["doc_id"] for entry in structured["truncations"]}
    assert reported == {"b"}


def test_a_query_cut_to_its_limit_is_reported():
    """The query bound is a bound like any other."""
    long_query = "q" * (_core.MAX_QUERY_CHARS + 20)
    structured = _core.build_search_docs_result(
        long_query, [_chunk("short")]
    )["structuredContent"]
    assert any(entry["applied_to"] == "query_chars"
               for entry in structured["truncations"])


def test_an_ellipsis_in_the_source_is_not_mistaken_for_truncation():
    """The defect this closes: the signal was indistinguishable from content."""
    structured = _core.build_search_docs_result(
        "q", [_chunk("a real ending\u2026")]
    )["structuredContent"]
    assert structured["truncated"] is False


def test_the_existing_fields_are_unchanged():
    """Truncation reporting is additional, not a reshaping of the result."""
    structured = _core.build_search_docs_result("q", [_chunk("short")])["structuredContent"]
    for field in ("query", "count", "passages", "citations", "message",
                  "retrieval_status", "security"):
        assert field in structured


def test_the_resource_reports_its_own_bound():
    """A rendered resource is a bounded payload too."""
    rendered = _server._read_resource(
        lambda doc_id: _chunk("z" * (_core.MAX_RESOURCE_CHARS * 2)), "big"
    )
    assert "truncated" in rendered.lower()
    assert str(_core.MAX_CHUNK_CHARS) in rendered or str(
        _core.MAX_RESOURCE_CHARS
    ) in rendered


def test_a_small_resource_makes_no_truncation_claim():
    """Absent when nothing was cut."""
    rendered = _server._read_resource(lambda doc_id: _chunk("short body"), "small")
    assert "truncated" not in rendered.lower()
