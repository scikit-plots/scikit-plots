"""
Argument validation at the public Tier-L entry point (slice S-25).

``build_search_docs_result`` is exported in ``_core.__all__``. Its bounds lived
only in ``SearchCoordinator.validate``, which a direct caller bypasses, so
``max_results=-1`` and ``max_results=0`` were accepted. The rule now comes from
:mod:`scikitplot._utils._indexing`, the same one the corpus artifact and the
lexical index apply.

See Also
--------
scikitplot.mcp._validation.require_count
scikitplot.mcp._core.build_search_docs_result
"""

import pytest

from .._validation import require_count
from .. import _core


def _chunk(doc_id="d"):
    """Return one retrieved chunk."""
    return _core.RetrievedChunk(
        doc_id=doc_id, title="T", source_uri="docs://d", anchor="#a",
        text="body", score=0.5,
    )


@pytest.mark.parametrize("value", [-1, -20])
def test_negative_result_counts_are_refused(value):
    """A negative bound was accepted and silently produced nothing."""
    with pytest.raises(ValueError):
        _core.build_search_docs_result("q", [_chunk()], max_results=value)


@pytest.mark.parametrize("value", [True, False])
def test_boolean_result_counts_are_refused(value):
    """``True`` is not a number of results."""
    with pytest.raises(TypeError):
        _core.build_search_docs_result("q", [_chunk()], max_results=value)


@pytest.mark.parametrize("value", [1.0, "5", None])
def test_non_integral_result_counts_are_refused(value):
    """A count is an integer or it is an error."""
    with pytest.raises(TypeError):
        _core.build_search_docs_result("q", [_chunk()], max_results=value)


def test_counts_above_the_declared_cap_are_clamped_not_refused():
    """
    Asking for more than the module serves is documented behaviour, not an error.

    MC04 was that values which are not counts at all were accepted. Clamping an
    over-cap request has its own test in the project suite, and tightening it
    here would break a documented contract the finding never questioned.
    """
    chunks = [_chunk(str(i)) for i in range(_core.MAX_RESULTS + 5)]
    result = _core.build_search_docs_result("q", chunks,
                                            max_results=_core.MAX_RESULTS + 1)
    assert result["structuredContent"]["count"] == _core.MAX_RESULTS


def test_zero_is_accepted_as_a_request_for_nothing():
    """Zero is a legitimate bound, not a mistake."""
    result = _core.build_search_docs_result("q", [_chunk()], max_results=0)
    assert result["structuredContent"]["count"] == 0


def test_ordinary_calls_are_unaffected():
    """The default path and an in-range bound behave as before."""
    chunks = [_chunk("a"), _chunk("b")]
    assert _core.build_search_docs_result("q", chunks)["structuredContent"]["count"] == 2
    assert _core.build_search_docs_result(
        "q", chunks, max_results=1)["structuredContent"]["count"] == 1


def test_the_builder_and_the_coordinator_agree():
    """The public function is no longer weaker than the path that wraps it."""
    for candidate in (-1, 0, 1, _core.MAX_RESULTS, True):
        try:
            require_count(candidate, name="max_results")
        except Exception as exc:  # noqa: BLE001
            with pytest.raises(type(exc)):
                _core.build_search_docs_result("q", [_chunk()], max_results=candidate)
        else:
            _core.build_search_docs_result("q", [_chunk()], max_results=candidate)
