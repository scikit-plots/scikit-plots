"""
Persistence and sparse-evaluation regressions (slice S-18).

A lexical index held derived statistics in memory and offered no way to write
them down, so it could not be published, pinned to a build, or reloaded: every
process start rebuilt it. Scoring also materialised one value per document per
query term, so a query touching three documents in a million-document corpus
still paid for a million.

See Also
--------
scikitplot.rank_bm25._rank_bm25.BM25.save
scikitplot.rank_bm25._rank_bm25.BM25.load
"""

import json

import pytest

from .._identity import Analyzer
from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

CORPUS = [["alpha", "beta"], ["beta"], ["gamma"]]
IDS = ["doc-a", "doc-b", "doc-c"]
CLASSES = (BM25Okapi, BM25L, BM25Plus)


@pytest.mark.parametrize("scorer", CLASSES)
def test_an_index_round_trips(tmp_path, scorer):
    """What was saved is what loads: same scores, same identity."""
    index = scorer(CORPUS, doc_ids=IDS)
    index.save(tmp_path / "bm25")
    reloaded = scorer.load(tmp_path / "bm25")
    assert reloaded.identity == index.identity
    assert reloaded.get_top_ids(["beta"], n=3) == index.get_top_ids(["beta"], n=3)


def test_the_analyzer_declaration_survives(tmp_path):
    """A build that cannot say how it tokenised is a build nobody can repeat."""
    analyzer = Analyzer(name="whitespace", version="2", stopwords=frozenset({"the"}))
    index = BM25Okapi(["the alpha", "beta"], tokenizer=analyzer, doc_ids=["a", "b"])
    index.save(tmp_path / "bm25")
    reloaded = BM25Okapi.load(tmp_path / "bm25")
    assert reloaded.analyzer == analyzer
    assert reloaded.identity == index.identity


def test_publication_is_a_generation_behind_a_pointer(tmp_path):
    """The same publication contract the corpus artifact uses."""
    target = tmp_path / "bm25"
    BM25Okapi(CORPUS, doc_ids=IDS).save(target)
    assert (target / "current.json").is_file()
    generations = [p for p in target.iterdir() if p.name.startswith("generation-")]
    assert len(generations) == 1


def test_a_held_generation_survives_a_replacement(tmp_path):
    """A reader keeps the build it opened."""
    target = tmp_path / "bm25"
    first = BM25Okapi(CORPUS, doc_ids=IDS)
    first_generation = first.save(target)
    held = first_generation
    BM25Okapi([["delta"]], doc_ids=["doc-d"]).save(target)
    assert BM25Okapi.load(target).doc_ids == ("doc-d",)
    # The generation is named for the state it holds, not for `identity`: two
    # corpora built the same way share an identity but are different builds.
    assert held.is_dir()
    assert BM25Okapi.load(held).doc_ids == tuple(IDS)


def test_a_failed_save_leaves_the_previous_generation(tmp_path, monkeypatch):
    """Publication commits or leaves the previous build alone."""
    target = tmp_path / "bm25"
    BM25Okapi(CORPUS, doc_ids=IDS).save(target)

    def boom(*args, **kwargs):
        raise OSError("injected")

    monkeypatch.setattr(json, "dumps", boom)
    with pytest.raises(OSError):
        BM25Okapi([["delta"]], doc_ids=["doc-d"]).save(target)
    monkeypatch.undo()
    assert BM25Okapi.load(target).doc_ids == tuple(IDS)


def test_a_missing_pointer_is_refused(tmp_path):
    """A directory with no pointer is not guessed at."""
    target = tmp_path / "bm25"
    BM25Okapi(CORPUS, doc_ids=IDS).save(target)
    (target / "current.json").unlink()
    with pytest.raises(ValueError):
        BM25Okapi.load(target)


# -- sparse evaluation -----------------------------------------------------


def test_only_matching_documents_are_scored():
    """A query touching one document does not pay for the whole corpus."""
    corpus = [["alpha"]] + [["filler"]] * 200
    index = BM25Okapi(corpus, doc_ids=[f"d{i}" for i in range(201)])
    assert index.candidate_count(["alpha"]) == 1
    assert index.candidate_count(["filler"]) == 200
    assert index.candidate_count(["absent"]) == 0


def test_sparse_and_rank_all_agree():
    """The fast path and the diagnostic must not disagree about ranking."""
    index = BM25Okapi(CORPUS, doc_ids=IDS)
    scores = index.get_scores(["beta"])
    sparse = index.get_top_ids(["beta"], n=3)
    by_id = dict(sparse)
    for position, doc_id in enumerate(IDS):
        if doc_id in by_id:
            assert by_id[doc_id] == pytest.approx(float(scores[position]))


def test_an_out_of_vocabulary_query_returns_nothing():
    """A query matching no document returns no hits, not arbitrary zeros."""
    index = BM25Okapi(CORPUS, doc_ids=IDS)
    assert index.get_top_ids(["absent"], n=3) == []


def test_rank_all_is_still_available_as_a_diagnostic():
    """The whole-corpus vector remains, named for what it is."""
    scores = BM25Okapi(CORPUS).get_scores(["beta"])
    assert len(scores) == 3
