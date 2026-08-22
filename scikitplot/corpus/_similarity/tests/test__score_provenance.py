# scikitplot/corpus/_similarity/tests/test__score_provenance.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-hit score provenance and the fusion policy (F-R07-03, ADR-R07-003)."""

from __future__ import annotations

import pytest

from ..._schema import CorpusDocument
from .._similarity import RetrievalConfig, RetrievalHit, RetrievalIndex

__all__: "list[str]" = [
    "TestScoreProvenance",
    "TestFusionPolicy",
]


def _docs():
    texts = ["quantum physics", "classical motion", "quantum computing"]
    return [
        CorpusDocument.create(
            input_path=f"f{i}.txt",
            chunk_index=i,
            text=text,
            embedding=[float(i), 1.0, 0.0],
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture(name="index")
def _index():
    index = RetrievalIndex()
    index.build(_docs())
    return index


class TestScoreProvenance:
    """F-R07-03 — ``score`` was a single unqualified float."""

    @pytest.mark.parametrize(
        ("mode", "expected_metric"),
        [
            ("strict", "exact_match"),
            ("keyword", "bm25"),
            ("semantic", "cosine_similarity"),
            ("hybrid", "reciprocal_rank_fusion"),
        ],
    )
    def test_every_mode_declares_its_metric(
        self, index, mode: str, expected_metric: str
    ) -> None:
        response = index.search(
            "quantum",
            config=RetrievalConfig(match_mode=mode, top_k=2),
            query_embedding=[1.0, 1.0, 0.0],
        )
        assert len(response) > 0
        assert response[0].native_metric == expected_metric

    def test_rank_is_zero_based_and_ordered(self, index) -> None:
        response = index.search(
            "quantum", config=RetrievalConfig(match_mode="keyword", top_k=3)
        )
        assert [hit.rank for hit in response] == list(range(len(response)))

    def test_native_score_matches_score_for_single_leg_modes(self, index) -> None:
        response = index.search(
            "quantum",
            config=RetrievalConfig(match_mode="semantic", top_k=1),
            query_embedding=[1.0, 1.0, 0.0],
        )
        hit = response[0]
        assert hit.native_score == pytest.approx(hit.score)

    def test_fused_hit_has_no_native_score(self, index) -> None:
        """A fused score belongs to no backend.

        Claiming a ``native_score`` for it would assert a scale the value does
        not have -- the fused number is not on any leg's scale.
        """
        response = index.search(
            "quantum",
            config=RetrievalConfig(match_mode="hybrid", top_k=2),
            query_embedding=[1.0, 1.0, 0.0],
        )
        hit = response[0]
        assert hit.native_score is None
        assert hit.native_metric == "reciprocal_rank_fusion"
        assert hit.score > 0

    def test_semantic_metric_follows_the_backend_declaration(self, index) -> None:
        """The metric is read from the backend, not hardcoded.

        IMPL-12 made ``score_semantics`` declarable precisely so this could not
        drift from what the backend actually returns.
        """
        index._backend.score_semantics = "bounded_inverse_distance"
        response = index.search(
            "quantum",
            config=RetrievalConfig(match_mode="semantic", top_k=1),
            query_embedding=[1.0, 1.0, 0.0],
        )
        assert response[0].native_metric == "bounded_inverse_distance"

    def test_provenance_is_excluded_from_equality(self) -> None:
        """Provenance says how a hit was produced, not what it is."""
        doc = _docs()[0]
        a = RetrievalHit(doc=doc, score=1.0, match_mode="keyword", rank=0)
        b = RetrievalHit(doc=doc, score=1.0, match_mode="keyword", rank=7)
        assert a == b


class TestFusionPolicy:
    """ADR-R07-003 — rank fusion is the default; score fusion is opt-in."""

    def test_mixed_metrics_forbid_score_fusion(self, index) -> None:
        """Adding a BM25 score to a cosine similarity yields a meaningless number."""
        keyword = index.search(
            "quantum", config=RetrievalConfig(match_mode="keyword")
        )
        semantic = index.search(
            "quantum",
            config=RetrievalConfig(match_mode="semantic"),
            query_embedding=[1.0, 1.0, 0.0],
        )
        combined = list(keyword) + list(semantic)
        assert RetrievalIndex.check_score_fusion_allowed(combined) is None

    def test_shared_cosine_metric_permits_score_fusion(self, index) -> None:
        semantic = index.search(
            "quantum",
            config=RetrievalConfig(match_mode="semantic"),
            query_embedding=[1.0, 1.0, 0.0],
        )
        assert (
            RetrievalIndex.check_score_fusion_allowed(list(semantic))
            == "cosine_similarity"
        )

    def test_shared_but_unnormalized_metric_still_forbids_score_fusion(
        self, index
    ) -> None:
        """A shared metric is necessary but not sufficient.

        Score fusion also needs a *validated normalization*, and R06 established
        none exists for any non-cosine metric today -- so BM25-only hits are
        refused even though they share a metric.
        """
        keyword = index.search(
            "quantum", config=RetrievalConfig(match_mode="keyword")
        )
        assert RetrievalIndex.check_score_fusion_allowed(list(keyword)) is None

    def test_empty_input_forbids_score_fusion(self) -> None:
        assert RetrievalIndex.check_score_fusion_allowed([]) is None

    def test_hits_without_declared_metrics_are_ignored_not_assumed(self) -> None:
        """An undeclared metric must not be treated as compatible."""
        doc = _docs()[0]
        undeclared = RetrievalHit(doc=doc, score=1.0, match_mode="keyword")
        assert RetrievalIndex.check_score_fusion_allowed([undeclared]) is None
