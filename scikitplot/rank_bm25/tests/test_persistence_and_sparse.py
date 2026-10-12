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
import random

import numpy as np
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


@pytest.mark.parametrize("scorer", CLASSES)
def test_sparse_ranking_and_scores_equal_the_dense_path(scorer):
    """
    ``get_top_ids`` is ``get_scores`` restricted to matching documents.

    BM25L and BM25+ lower-bound every term with ``delta``, so a term a document
    lacks still contributes. The sparse path used to drop those contributions:
    on random corpora it ranked differently from ``get_scores`` for about one
    query in nine and reported different scores for about two in three. Scores
    are now compared exactly, not within a tolerance.
    """
    rng = random.Random(20261012)  # noqa: S311 - seeded, reproducible
    vocabulary = list("abcdefgh")
    for _ in range(200):
        corpus = [rng.choices(vocabulary, k=rng.randint(1, 8)) for _ in range(6)]
        ids = [f"d{i}" for i in range(6)]
        index = scorer(corpus, doc_ids=ids)
        query = rng.sample(vocabulary, k=rng.randint(1, 3)) * rng.randint(1, 2)
        dense = index.get_scores(query)
        sparse = index.get_top_ids(query, n=len(ids))
        matching = {
            ids[row]
            for row, document in enumerate(corpus)
            if set(query) & set(document)
        }
        assert {doc_id for doc_id, _ in sparse} == matching
        for doc_id, score in sparse:
            assert score == dense[ids.index(doc_id)]  # bit-identical
        order = [ids[row] for row in np.argsort(-dense, kind="stable")]
        assert [doc_id for doc_id, _ in sparse] == [d for d in order if d in matching]


@pytest.mark.parametrize("scorer", CLASSES)
def test_batch_scores_equal_the_dense_path(scorer):
    index = scorer([["a", "b"], ["b", "c", "c"], ["d"]])
    dense = index.get_scores(["b", "c", "x"])
    batch = index.get_batch_scores(["b", "c", "x"], [2, 0])
    assert batch == [dense[2], dense[0]]


@pytest.mark.parametrize("scorer", [BM25L, BM25Plus])
def test_a_delta_recipe_scores_an_absent_term(scorer):
    """The premise of the fix: a lacking term still adds ``delta`` evidence."""
    index = scorer([["a"], ["b"]], doc_ids=["x", "y"])
    assert index.get_scores(["a", "b"])[0] > index.get_scores(["a"])[0]


def test_an_out_of_vocabulary_query_returns_nothing():
    """A query matching no document returns no hits, not arbitrary zeros."""
    index = BM25Okapi(CORPUS, doc_ids=IDS)
    assert index.get_top_ids(["absent"], n=3) == []


def test_rank_all_is_still_available_as_a_diagnostic():
    """The whole-corpus vector remains, named for what it is."""
    scores = BM25Okapi(CORPUS).get_scores(["beta"])
    assert len(scores) == 3
