"""
Result-binding regressions (slice S-17).

Results were bound to caller-supplied row order: ``get_top_n`` took a sequence
of documents and indexed it with the scorer's row offsets. A sequence of the
right length but the wrong order was accepted silently and returned the wrong
documents, and ties came back in descending row order because the sort was
reversed rather than tie-broken. Results now bind to identities the index holds.

See Also
--------
scikitplot.rank_bm25._rank_bm25.BM25.get_top_ids
"""

import pytest

from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

CORPUS = [["alpha", "beta"], ["beta"], ["gamma"]]
IDS = ["doc-a", "doc-b", "doc-c"]
CLASSES = (BM25Okapi, BM25L, BM25Plus)


@pytest.mark.parametrize("scorer", CLASSES)
def test_results_bind_to_identities_supplied_at_build_time(scorer):
    """A hit names a document, not a position into whatever the caller passed."""
    index = scorer(CORPUS, doc_ids=IDS)
    hits = index.get_top_ids(["alpha"], n=1)
    assert hits[0][0] == "doc-a"
    assert hits[0][1] > 0


@pytest.mark.parametrize("scorer", CLASSES)
def test_a_reordered_sequence_cannot_misbind(scorer):
    """The failure B04 recorded is unreachable once identities are held."""
    index = scorer(CORPUS, doc_ids=IDS)
    assert index.get_top_ids(["gamma"], n=1)[0][0] == "doc-c"


def test_ties_are_broken_deterministically_by_position():
    """Equal scores come back in row order, not reversed."""
    index = BM25Okapi([["x"], ["x"], ["x"]], doc_ids=["a", "b", "c"])
    hits = index.get_top_ids(["x"], n=3)
    assert [doc_id for doc_id, _ in hits] == ["a", "b", "c"]


def test_ties_are_broken_the_same_way_for_positional_results():
    """The positional path gets the same order, so the two cannot disagree."""
    documents = ["first", "second", "third"]
    index = BM25Okapi([["x"], ["x"], ["x"]])
    assert index.get_top_n(["x"], documents, n=3) == documents


@pytest.mark.parametrize("scorer", CLASSES)
def test_ids_must_match_the_corpus(scorer):
    """A mismatched identity list is refused where it is supplied."""
    with pytest.raises(ValueError):
        scorer(CORPUS, doc_ids=["only-one"])


@pytest.mark.parametrize("scorer", CLASSES)
def test_duplicate_ids_are_refused(scorer):
    """One identity cannot name two rows, as in the artifact sidecar."""
    with pytest.raises(ValueError):
        scorer(CORPUS, doc_ids=["a", "a", "b"])


def test_get_top_ids_requires_identities():
    """Asking for identities without supplying any is an error, not a guess."""
    index = BM25Okapi(CORPUS)
    with pytest.raises(ValueError) as excinfo:
        index.get_top_ids(["alpha"], n=1)
    assert "doc_ids" in str(excinfo.value)


def test_the_positional_path_still_works():
    """Existing callers are unaffected; the precondition is now documented."""
    documents = ["first", "second", "third"]
    assert BM25Okapi(CORPUS).get_top_n(["gamma"], documents, n=1) == ["third"]


def test_result_count_is_validated_on_both_paths():
    """The shared count contract covers the identity path too."""
    index = BM25Okapi(CORPUS, doc_ids=IDS)
    with pytest.raises(ValueError):
        index.get_top_ids(["alpha"], n=-1)
    assert index.get_top_ids(["alpha"], n=0) == []


def test_doc_ids_are_exposed_in_row_order():
    """A caller can see the mapping the index holds."""
    assert BM25Okapi(CORPUS, doc_ids=IDS).doc_ids == tuple(IDS)
