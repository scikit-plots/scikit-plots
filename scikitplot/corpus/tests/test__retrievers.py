# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot.corpus._retrievers` — legs and fusion provenance.

Covers findings F-R09-02 (per-retriever rank computed then discarded) and
F-R09-03 (no retriever abstraction, no reranker separation).
"""

from __future__ import annotations

import json

import pytest

from .._graph import GraphQuery, derive_graph
from .._retrieval import LegKind, LegStatus
from .._retrievers import (
    DenseRetriever,
    GraphRetriever,
    LegContribution,
    LexicalRetriever,
    fuse_by_rank,
)
from .._schema import CorpusDocument
from .._similarity._similarity import RetrievalConfig, RetrievalIndex

__all__: "list[str]" = [
    "TestLegs",
    "TestFusionProvenance",
    "TestGraphLeg",
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


@pytest.fixture(name="legs")
def _legs(index):
    lexical = LexicalRetriever(index).retrieve(
        "quantum", RetrievalConfig(top_k=5, match_mode="keyword")
    )
    dense = DenseRetriever(index).retrieve(
        "quantum",
        RetrievalConfig(top_k=5, match_mode="semantic"),
        query_embedding=[1.0, 1.0, 0.0],
    )
    return lexical, dense


class TestLegs:
    """F-R09-03 — legs are objects that can carry a status."""

    def test_each_leg_declares_its_kind(self, index) -> None:
        assert LexicalRetriever(index).leg is LegKind.LEXICAL
        assert DenseRetriever(index).leg is LegKind.DENSE

    def test_a_leg_returns_an_outcome_not_a_bare_list(self, legs) -> None:
        """A bare list cannot distinguish 'found nothing' from 'could not run'."""
        lexical, _ = legs
        assert lexical.outcome.status is LegStatus.SUCCESS
        assert lexical.outcome.hit_count == len(lexical)

    def test_leg_is_iterable_and_sized(self, legs) -> None:
        lexical, _ = legs
        assert len(list(lexical)) == len(lexical)

    def test_dense_leg_reports_why_it_could_not_run(self, index) -> None:
        result = DenseRetriever(index).retrieve(
            "quantum", RetrievalConfig(match_mode="keyword"), query_embedding=None
        )
        assert result.outcome.status is LegStatus.FAILED
        assert result.outcome.error.code == "NO_QUERY_EMBEDDING"
        assert len(result) == 0

    def test_legs_carry_generation_provenance(self, legs, index) -> None:
        lexical, dense = legs
        assert lexical.outcome.generation == index.index_generation
        assert dense.outcome.backend == index.backend_name


class TestFusionProvenance:
    """F-R09-02 — rank was computed in the fusion loop, then dropped."""

    def test_contributions_name_the_leg_and_rank(self, legs) -> None:
        fused = fuse_by_rank(list(legs), top_k=3)
        _, _, contributions = fused[0]
        assert all(isinstance(c, LegContribution) for c in contributions)
        assert {c.leg for c in contributions} <= {LegKind.LEXICAL, LegKind.DENSE}
        assert all(c.rank >= 0 for c in contributions)

    def test_a_both_legs_hit_is_distinguishable_from_a_one_leg_hit(
        self, legs
    ) -> None:
        """
        The signal that used to collapse into a single float.

        A document found by both legs and one found by only one are very
        different confidence claims; the fused score alone could not tell them
        apart.
        """
        fused = fuse_by_rank(list(legs), top_k=5)
        counts = {len(contribs) for _, _, contribs in fused}
        assert counts == {1, 2}

    def test_contributions_record_the_native_metric(self, legs) -> None:
        """So a consumer can verify no cross-metric comparison happened."""
        fused = fuse_by_rank(list(legs), top_k=3)
        metrics = {
            c.native_metric for _, _, contribs in fused for c in contribs
        }
        assert metrics == {"bm25", "cosine_similarity"}

    def test_fused_score_is_recomputable_from_contributions(self, legs) -> None:
        """The point of returning provenance: the score can be explained."""
        rrf_k = 60
        fused = fuse_by_rank(list(legs), rrf_k=rrf_k, top_k=5)
        for _, score, contributions in fused:
            expected = sum(1.0 / (rrf_k + c.rank + 1) for c in contributions)
            assert score == pytest.approx(expected)

    def test_weights_are_applied_per_leg(self, legs) -> None:
        unweighted = fuse_by_rank(list(legs), top_k=5)
        weighted = fuse_by_rank(list(legs), weights={LegKind.DENSE: 0.0}, top_k=5)
        assert [s for _, s, _ in unweighted] != [s for _, s, _ in weighted]

    def test_failed_legs_contribute_nothing_harmlessly(self, index, legs) -> None:
        lexical, _ = legs
        failed = DenseRetriever(index).retrieve(
            "quantum", RetrievalConfig(match_mode="keyword"), query_embedding=None
        )
        fused = fuse_by_rank([lexical, failed], top_k=5)
        assert all(
            all(c.leg is LegKind.LEXICAL for c in contribs)
            for _, _, contribs in fused
        )

    def test_fusion_keys_on_doc_id_not_row_offset(self, legs) -> None:
        """The property that kept F-R01-07's positional coupling out of fusion."""
        fused = fuse_by_rank(list(legs), top_k=5)
        ids = [hit.doc.doc_id for hit, _, _ in fused]
        assert len(ids) == len(set(ids))

    def test_contribution_is_serialisable(self, legs) -> None:
        _, _, contributions = fuse_by_rank(list(legs), top_k=1)[0]
        json.dumps([c.to_dict() for c in contributions])


class TestGraphLeg:
    """The graph as a peer evidence path."""

    def test_traversal_succeeds(self) -> None:
        docs = _docs()
        graph = derive_graph(docs)
        result = GraphRetriever(graph).retrieve(
            "q",
            RetrievalConfig(),
            graph_query=GraphQuery(seeds=[docs[0].doc_id]),
        )
        assert result.outcome.status is LegStatus.SUCCESS
        assert len(result) > 0

    def test_budget_exhaustion_degrades_the_leg(self) -> None:
        """The budget that stopped the traversal survives into the leg outcome."""
        docs = _docs()
        graph = derive_graph(docs)
        result = GraphRetriever(graph).retrieve(
            "q",
            RetrievalConfig(),
            graph_query=GraphQuery(seeds=[docs[0].doc_id], max_total_nodes=1),
        )
        assert result.outcome.status is LegStatus.DEGRADED
        assert result.outcome.error.details["budget"] == "max_total_nodes"

    def test_missing_seeds_fail_rather_than_return_empty(self) -> None:
        """The graph refines what already matched, so no seeds is not zero hits."""
        result = GraphRetriever(derive_graph(_docs())).retrieve(
            "q", RetrievalConfig()
        )
        assert result.outcome.status is LegStatus.FAILED
        assert result.outcome.error.code == "NO_GRAPH_SEEDS"
