# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._graph` — the G0 derived view (F-R08-01)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from .._graph import (
    G0_PRODUCER,
    EdgeTrust,
    GraphQuery,
    NodeKind,
    RelationType,
    derive_graph,
)
from .._schema import CorpusDocument

__all__: "list[str]" = [
    "TestDerivation",
    "TestEdgeProvenance",
    "TestTraversal",
    "TestBudgets",
]


def _doc(index: int, parent: "str | None" = None, path: str = "book.txt"):
    doc = CorpusDocument.create(
        input_path=path, chunk_index=index, text=f"text {index}"
    )
    return dataclasses.replace(doc, parent_doc_id=parent)


@pytest.fixture(name="docs")
def _docs():
    """root -> (c1, c2), all from one source."""
    root = _doc(0)
    return [root, _doc(1, root.doc_id), _doc(2, root.doc_id)]


@pytest.fixture(name="graph")
def _graph(docs):
    return derive_graph(docs, generation="g1")


class TestDerivation:
    """What G0 derives, and what it deliberately does not."""

    def test_node_kinds_follow_the_schema(self, graph, docs) -> None:
        """A chunk is a document with a parent, not a separate kind."""
        assert graph.node(docs[0].doc_id).kind is NodeKind.DOCUMENT
        assert graph.node(docs[1].doc_id).kind is NodeKind.SECTION

    def test_source_is_a_node_not_an_implicit_property(self, graph) -> None:
        """Modelling it implicitly would make same_source an N^2 edge set."""
        sources = [n for n in graph._nodes.values() if n.kind is NodeKind.SOURCE]
        assert len(sources) == 1

    def test_derives_three_relations(self, graph) -> None:
        relations = {
            edge.relation
            for edges in graph._out.values()
            for edge in edges
        }
        assert relations == {
            RelationType.SAME_SOURCE,
            RelationType.CONTAINS,
            RelationType.PARENT_OF,
            RelationType.PRECEDES,
        }

    def test_references_is_not_derived_at_g0(self, graph) -> None:
        """
        F-R08-03: CorpusDocument has its own url, not outbound links.

        Extracting body links feels like structure but is extraction, and needs
        provenance -- so it belongs to G1, not here.
        """
        relations = {
            edge.relation
            for edges in graph._out.values()
            for edge in edges
        }
        assert RelationType.REFERENCES not in relations

    def test_hierarchy_edges_are_empty_without_a_producer(self) -> None:
        """
        R13 / U-11: no pipeline stage sets parent_doc_id today.

        Recorded as a test so the limitation is visible rather than surprising:
        same_source and precedes carry the graph until a producer lands.
        """
        flat = [_doc(0), _doc(1)]
        graph = derive_graph(flat)
        relations = {
            edge.relation for edges in graph._out.values() for edge in edges
        }
        assert RelationType.CONTAINS not in relations
        assert RelationType.SAME_SOURCE in relations
        assert RelationType.PRECEDES in relations

    def test_precedes_follows_index_order_not_input_order(self) -> None:
        a, b, c = _doc(2), _doc(0), _doc(1)
        graph = derive_graph([a, b, c])
        precedes = [
            edge
            for edges in graph._out.values()
            for edge in edges
            if edge.relation is RelationType.PRECEDES
        ]
        assert (b.doc_id, c.doc_id) in {(e.source, e.target) for e in precedes}


class TestEdgeProvenance:
    """ADR-R08-003 — every edge, including a structural one."""

    def test_every_edge_answers_all_four_questions(self, graph) -> None:
        for edges in graph._out.values():
            for edge in edges:
                assert edge.relation is not None
                assert edge.evidence
                assert edge.producer == G0_PRODUCER
                assert edge.trust is EdgeTrust.DERIVED
                assert edge.generation == "g1"

    def test_no_generic_relation_exists(self) -> None:
        """An edge that does not say how two things relate cannot be filtered."""
        assert "related" not in {r.value for r in RelationType}

    def test_edges_are_serialisable(self, graph) -> None:
        edge = next(iter(graph._out.values()))[0]
        json.dumps(edge.to_dict())


class TestTraversal:
    """BFS, and only BFS."""

    def test_reaches_the_whole_component(self, graph, docs) -> None:
        response = graph.traverse(GraphQuery(seeds=[docs[0].doc_id], max_hops=3))
        assert response.status == "success"
        assert len(response.nodes) == graph.node_count

    def test_unknown_seed_yields_empty(self, graph) -> None:
        response = graph.traverse(GraphQuery(seeds=["nope"]))
        assert response.status == "empty"

    def test_relation_filter_is_the_constrained_mode(self, graph, docs) -> None:
        """Relation-constrained traversal is BFS with a filter, not an algorithm."""
        response = graph.traverse(
            GraphQuery(
                seeds=[docs[0].doc_id],
                allowed_relation_types=(RelationType.CONTAINS,),
            )
        )
        assert {e.relation for e in response.edges} == {RelationType.CONTAINS}

    def test_response_is_serialisable(self, graph, docs) -> None:
        response = graph.traverse(GraphQuery(seeds=[docs[0].doc_id]))
        json.dumps(response.to_dict())


class TestBudgets:
    """ADR-R08-004 — exhaustion degrades, never truncates silently."""

    @pytest.mark.parametrize(
        ("kwargs", "budget"),
        [
            ({"max_total_nodes": 1}, "max_total_nodes"),
            ({"max_total_edges": 1}, "max_total_edges"),
            ({"max_neighbors_per_node": 1}, "max_neighbors_per_node"),
            ({"max_evidence": 1}, "max_evidence"),
            ({"max_seed_nodes": 0}, "max_seed_nodes"),
        ],
    )
    def test_each_budget_degrades_and_names_itself(
        self, graph, docs, kwargs, budget: str
    ) -> None:
        response = graph.traverse(GraphQuery(seeds=[docs[0].doc_id], **kwargs))
        assert response.status == "degraded"
        assert budget in response.exhausted_budgets

    def test_degradation_always_carries_a_record(self, graph, docs) -> None:
        """A truncated traversal that looks complete is the F-R07-01 shape."""
        response = graph.traverse(
            GraphQuery(seeds=[docs[0].doc_id], max_total_nodes=1)
        )
        assert response.errors
        assert response.errors[0].code == "GRAPH_BUDGET_EXHAUSTED"
        assert response.errors[0].details["budget"] == "max_total_nodes"

    def test_deadline_is_enforced(self, graph, docs) -> None:
        response = graph.traverse(
            GraphQuery(seeds=[docs[0].doc_id], deadline_seconds=0.0, max_hops=3)
        )
        assert "deadline_seconds" in response.exhausted_budgets

    def test_an_unbudgeted_traversal_is_not_possible(self) -> None:
        """Every budget has a conservative default; none is optional."""
        query = GraphQuery()
        for field in (
            "max_seed_nodes",
            "max_hops",
            "max_neighbors_per_node",
            "max_total_nodes",
            "max_total_edges",
        ):
            assert getattr(query, field) > 0

    def test_min_trust_filters_edges(self, graph, docs) -> None:
        response = graph.traverse(
            GraphQuery(seeds=[docs[0].doc_id], min_trust=EdgeTrust.DERIVED)
        )
        assert all(e.trust is EdgeTrust.DERIVED for e in response.edges)
