# scikitplot/corpus/_similarity/tests/test__pairwise.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Rankers built from pairwise functions, and the Levenshtein ranker."""

from __future__ import annotations

import math
import pickle
import sys

import pytest

from ..._custom_hooks import CustomRetrievalIndex
from ..._metrics import edit_similarity
from ..._schema import CorpusDocument
from .._pairwise import (
    LevenshteinRanker,
    PairwiseRanker,
    make_levenshtein_ranker,
    make_pairwise_ranker,
)
from .._similarity import RetrievalConfig


def _docs(*texts, normalized=None):
    return [
        CorpusDocument(
            doc_id=f"d{i}",
            input_path=f"memory://d{i}",
            chunk_index=0,
            text=text,
            normalized_text=None if normalized is None else normalized[i],
        )
        for i, text in enumerate(texts)
    ]


def _ids(hits):
    return [hit.doc.doc_id for hit in hits]


class TestPairwiseRanker:
    def test_ranks_by_similarity(self):
        hits = make_pairwise_ranker(edit_similarity)("kitten", _docs("sitting", "kitten", "dog"), RetrievalConfig(top_k=2))
        assert _ids(hits) == ["d1", "d0"]
        assert [hit.rank for hit in hits] == [0, 1]
        assert hits[0].match_mode == "pairwise"
        assert hits[0].native_metric == "edit_similarity"
        assert hits[1].native_score == hits[1].score == 1 - 3 / 7

    def test_distance_with_greater_is_better_false(self):
        def distance(query, text):
            return abs(len(query) - len(text))

        hits = make_pairwise_ranker(distance, greater_is_better=False)("abc", _docs("a", "abcd", "abc"), RetrievalConfig(top_k=3))
        assert _ids(hits) == ["d2", "d1", "d0"]
        assert [hit.score for hit in hits] == [0.0, -1.0, -2.0]
        assert math.copysign(1.0, hits[0].score) == 1.0  # no negative zero
        assert [hit.native_score for hit in hits] == [0.0, 1.0, 2.0]

    def test_ties_follow_document_position(self):
        hits = make_pairwise_ranker(lambda q, t: 1.0)("q", _docs("x", "y", "z"), RetrievalConfig(top_k=3))
        assert _ids(hits) == ["d0", "d1", "d2"]

    def test_cutoff_can_return_fewer_than_top_k(self):
        ranker = make_pairwise_ranker(edit_similarity, score_cutoff=0.8)
        assert _ids(ranker("kitten", _docs("kitten", "xxxxxx"), RetrievalConfig(top_k=10))) == ["d0"]

    def test_kwargs_reach_the_function(self):
        ranker = make_pairwise_ranker(edit_similarity, normalize="basic")
        assert ranker("KITTEN", _docs("kitten"), RetrievalConfig(top_k=1))[0].score == 1.0
        assert "normalize='basic'" in repr(ranker)

    def test_normalized_text_is_preferred(self):
        docs = _docs("Kitten!", normalized=["kitten"])
        assert make_pairwise_ranker(edit_similarity)("kitten", docs, RetrievalConfig(top_k=1))[0].score == 1.0
        raw = make_pairwise_ranker(edit_similarity, use_normalized_text=False)
        assert raw("kitten", docs, RetrievalConfig(top_k=1))[0].score < 1.0

    @pytest.mark.parametrize(
        ("value", "error", "match"),
        [(math.nan, ValueError, "finite"), (math.inf, ValueError, "finite"), ("0.5", TypeError, "real number"), (True, TypeError, "real number")],
    )
    def test_bad_scores_name_the_document(self, value, error, match):
        ranker = make_pairwise_ranker(lambda q, t: value, name="broken")
        with pytest.raises(error, match=f"broken returned .*position 0.*{match}"):
            ranker("q", _docs("x"), RetrievalConfig(top_k=1))

    def test_top_k_is_checked(self):
        class Config:
            top_k = 0

        with pytest.raises(ValueError, match="top_k"):
            make_pairwise_ranker(edit_similarity)("q", _docs("x"), Config())

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"score_func": 3}, TypeError, "callable"),
            ({"greater_is_better": 1}, TypeError, "greater_is_better"),
            ({"use_normalized_text": None}, TypeError, "use_normalized_text"),
            ({"score_cutoff": math.inf}, ValueError, "finite"),
            ({"score_cutoff": "0.5"}, TypeError, "score_cutoff"),
            ({"name": 3}, TypeError, "name"),
            ({"tokens": 3}, TypeError, "does not accept"),
        ],
    )
    def test_factory_refusals(self, kwargs, error, match):
        options = {"score_func": edit_similarity, **kwargs}
        with pytest.raises(error, match=match):
            make_pairwise_ranker(**options)

    def test_repr_and_pickle(self):
        ranker = make_pairwise_ranker(edit_similarity, greater_is_better=False, score_cutoff=-0.5)
        assert repr(ranker) == "make_pairwise_ranker(edit_similarity, greater_is_better=False, score_cutoff=-0.5)"
        assert isinstance(pickle.loads(pickle.dumps(ranker)), PairwiseRanker)  # noqa: S301 - own object

    def test_drives_custom_retrieval_index(self):
        index = CustomRetrievalIndex(custom_scorer_fn=make_pairwise_ranker(edit_similarity))
        index.build(_docs("sitting", "kitten"))
        assert _ids(index.search("kitten")) == ["d1", "d0"]


class TestLevenshteinRanker:
    """The four corpus-adapter tests formerly in ``levenshtein/tests``."""

    def test_ranks_documents(self):
        hits = make_levenshtein_ranker(backend="python")("kitten", _docs("kitten", "sitting"), RetrievalConfig(top_k=2))
        assert [hit.doc.text for hit in hits] == ["kitten", "sitting"]
        assert hits[0].score == 1.0
        assert hits[0].native_metric == "normalized_levenshtein_similarity"
        assert hits[0].match_mode == "levenshtein"

    def test_provenance_names_the_backend_that_ran(self, monkeypatch):
        core = pytest.importorskip("scikitplot.levenshtein._core")
        monkeypatch.setattr(core, "_resolve_backend_name", lambda name, strict: "rapidfuzz")

        def broken_impl(_name):
            def fail(a, b):
                raise RuntimeError("accelerator broke")

            return fail

        monkeypatch.setattr(core, "_distance_impl", broken_impl)
        monkeypatch.setattr(core.logger, "warning", lambda *args, **kwargs: None)
        hits = make_levenshtein_ranker(backend="auto")("kitten", _docs("sitting"), RetrievalConfig(top_k=1))
        assert hits[0].backend == "levenshtein:python"

    def test_cutoff_can_return_fewer_than_top_k(self):
        ranker = make_levenshtein_ranker(backend="python", score_cutoff=0.8)
        hits = ranker("kitten", _docs("kitten", "xxxxxx"), RetrievalConfig(top_k=10))
        assert [hit.doc.text for hit in hits] == ["kitten"]

    def test_ties_follow_levenshtein_order(self):
        # "ab" vs "a" (d=1, sim 0.5) and "abcd" (d=2, sim 0.5): rank() orders by distance next.
        hits = make_levenshtein_ranker(backend="python")("ab", _docs("abcd", "a"), RetrievalConfig(top_k=2))
        assert [hit.doc.text for hit in hits] == ["a", "abcd"]

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [({"score_cutoff": 1.5}, ValueError), ({"score_cutoff": "1"}, TypeError), ({"strict": 1}, TypeError), ({"backend": 3}, TypeError)],
    )
    def test_factory_refusals(self, kwargs, error):
        with pytest.raises(error):
            make_levenshtein_ranker(**kwargs)

    def test_missing_levenshtein_names_the_alternative(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scikitplot.levenshtein", None)
        with pytest.raises(ImportError, match="make_pairwise_ranker"):
            make_levenshtein_ranker()

    def test_repr_and_pickle(self):
        ranker = make_levenshtein_ranker(backend="python", score_cutoff=0.5)
        assert repr(ranker) == (
            "make_levenshtein_ranker(backend='python', strict=False, use_normalized_text=True, score_cutoff=0.5)"
        )
        assert isinstance(pickle.loads(pickle.dumps(ranker)), LevenshteinRanker)  # noqa: S301 - own object



class TestReviewFixes:
    def test_integer_overflow_names_the_document(self):
        """Review F12: an int beyond the float range raised OverflowError."""
        ranker = make_pairwise_ranker(lambda q, t: 10**400, name="huge")
        with pytest.raises(ValueError, match=r"huge returned inf.*position 0"):
            ranker("q", _docs("x"), RetrievalConfig(top_k=1))

    def test_unknown_backend_fails_at_construction(self):
        """Review F5: 'nope' was accepted until the first search."""
        with pytest.raises(ValueError, match="unknown backend"):
            make_levenshtein_ranker(backend="nope")

    def test_unavailable_backend_warns_once_per_search(self, monkeypatch):
        """Review F5: one warning per document (2000 for 2000 documents)."""
        core = pytest.importorskip("scikitplot.levenshtein._core")
        real = core._backend_info

        def info(name):
            if name == "rapidfuzz":
                return core.BackendInfo("rapidfuzz", False, None, "MIT", "rapidfuzz", "simulated absence")
            return real(name)

        seen = []
        monkeypatch.setattr(core, "_backend_info", info)
        monkeypatch.setattr(core.logger, "warning", lambda *args, **kwargs: seen.append(args))
        ranker = make_levenshtein_ranker(backend="rapidfuzz")
        hits = ranker("kitten", _docs(*(["sitting"] * 2000)), RetrievalConfig(top_k=3))
        assert len(hits) == 3
        assert len(seen) == 1
        with pytest.raises(ImportError, match="unavailable"):
            make_levenshtein_ranker(backend="rapidfuzz", strict=True)
