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

    def test_a_deadline_of_zero_does_not_depend_on_the_clock_moving(
        self, graph, docs, monkeypatch
    ) -> None:
        """A coarse clock (Windows: 15.6 ms) may not tick during a traversal."""
        from .. import _graph

        monkeypatch.setattr(_graph.time, "monotonic", lambda: 1000.0)
        response = graph.traverse(
            GraphQuery(seeds=[docs[0].doc_id], deadline_seconds=0.0, max_hops=3)
        )
        assert "deadline_seconds" in response.exhausted_budgets

    def test_a_deadline_that_has_not_passed_is_not_exhausted(
        self, graph, docs, monkeypatch
    ) -> None:
        from .. import _graph

        monkeypatch.setattr(_graph.time, "monotonic", lambda: 1000.0)
        response = graph.traverse(
            GraphQuery(seeds=[docs[0].doc_id], deadline_seconds=60.0, max_hops=3)
        )
        assert "deadline_seconds" not in response.exhausted_budgets

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


# ===========================================================================
# D12: the outline, derived from metadata["heading_path"]
# ===========================================================================


def _headed(index, path, source="guide.md"):
    doc = CorpusDocument.create(input_path=source, chunk_index=index, text=f"text {index}")
    return dataclasses.replace(doc, metadata={"heading_path": path})


def _edges(graph):
    return sorted(
        (e.source, e.relation.value, e.target, e.evidence)
        for node_id in graph._nodes
        for e in graph.neighbors(node_id)
    )


def _heading(graph, label):
    found = [n for n in graph._nodes.values() if n.kind is NodeKind.HEADING and n.label == label]
    assert len(found) == 1, label
    return found[0].node_id


class TestOutline:
    def test_documents_without_a_heading_path_add_nothing(self, docs) -> None:
        plain = derive_graph(docs)
        assert not [n for n in plain._nodes.values() if n.kind is NodeKind.HEADING]
        assert all(e[3] != "metadata.heading_path" for e in _edges(plain))

    def test_the_tree_is_source_heading_subheading_document(self) -> None:
        docs = [
            _headed(0, ["Guide"]),
            _headed(1, ["Guide", "Install"]),
            _headed(2, ["Guide", "Install"]),
            _headed(3, ["Notes"]),
        ]
        graph = derive_graph(docs)
        guide, install, notes = (_heading(graph, t) for t in ("Guide", "Install", "Notes"))
        source = next(n.node_id for n in graph._nodes.values() if n.kind is NodeKind.SOURCE)
        outline = [e[:3] for e in _edges(graph) if e[3] == "metadata.heading_path"]
        ids = [d.doc_id for d in docs]
        want = [
            (guide, "same_source", source),
            (notes, "same_source", source),
            (guide, "contains", install),
            (install, "parent_of", guide),
            (guide, "contains", ids[0]),
            (ids[0], "parent_of", guide),
            (install, "contains", ids[1]),
            (ids[1], "parent_of", install),
            (install, "contains", ids[2]),
            (ids[2], "parent_of", install),
            (notes, "contains", ids[3]),
            (ids[3], "parent_of", notes),
        ]
        assert outline == sorted(want)
        assert sum(n.kind is NodeKind.HEADING for n in graph._nodes.values()) == len(
            {"Guide", "Install", "Notes"}
        )

    def test_the_same_titles_in_two_sources_are_two_headings(self) -> None:
        graph = derive_graph([_headed(0, ["Intro"], "a.md"), _headed(0, ["Intro"], "b.md")])
        headings = [n for n in graph._nodes.values() if n.kind is NodeKind.HEADING]
        assert [n.label for n in headings] == ["Intro", "Intro"]
        assert len({n.node_id for n in headings}) == len(headings)

    def test_a_title_holding_separators_does_not_merge_paths(self) -> None:
        docs = [_headed(0, ["a, b"]), _headed(1, ['a", "b']), _headed(2, ["a", "b"])]
        graph = derive_graph(docs)
        labels = sorted(n.label for n in graph._nodes.values() if n.kind is NodeKind.HEADING)
        assert labels == ["a", 'a", "b', "a, b", "b"]

    @pytest.mark.parametrize(
        "path", ["Guide", [], [""], ["ok", ""], [1], ("a", None), None, {"a": 1}],
        ids=repr,
    )
    def test_anything_but_a_sequence_of_titles_is_not_a_heading_path(self, path) -> None:
        graph = derive_graph([_headed(0, path)])
        assert not [n for n in graph._nodes.values() if n.kind is NodeKind.HEADING]

    def test_a_tuple_path_and_a_list_path_are_one_heading(self) -> None:
        graph = derive_graph([_headed(0, ("A", "B")), _headed(1, ["A", "B"])])
        assert sum(n.kind is NodeKind.HEADING for n in graph._nodes.values()) == len(("A", "B"))

    def test_traversal_reaches_a_sibling_through_its_heading(self) -> None:
        docs = [_headed(0, ["Guide", "Install"]), _headed(5, ["Guide", "Install"])]
        graph = derive_graph(docs)
        query = GraphQuery(
            seeds=(docs[0].doc_id,),
            max_hops=2,
            allowed_relation_types=(RelationType.CONTAINS, RelationType.PARENT_OF),
        )
        reached = {n.node_id for n in graph.traverse(query).nodes}
        assert docs[1].doc_id in reached

    def test_deriving_twice_gives_the_same_graph(self) -> None:
        docs = [_headed(i, ["G", f"S{i % 3}"]) for i in range(9)]
        first, second = derive_graph(docs), derive_graph(docs)
        assert list(first._nodes) == list(second._nodes)
        assert _edges(first) == _edges(second)

    def test_structure_chunks_carry_their_outline(self, tmp_path) -> None:
        from .._chunkers._structure import StructureChunker  # noqa: PLC0415
        from .._pipeline import CorpusPipeline  # noqa: PLC0415

        page = tmp_path / "page.md"
        page.write_text("# Guide\n\nIntro.\n\n## Install\n\nRun it.\n\n## Use\n\nCall it.\n", "utf-8")
        docs = CorpusPipeline(chunker=StructureChunker()).run(page).documents
        graph = derive_graph(docs)
        for doc in docs:
            path = doc.metadata.get("heading_path")
            if not path:
                continue
            parents = [
                graph.node(e.target).label
                for e in graph.neighbors(doc.doc_id)
                if e.relation is RelationType.PARENT_OF
            ]
            assert parents == [path[-1]]
        assert {"Guide", "Install", "Use"} <= {
            n.label for n in graph._nodes.values() if n.kind is NodeKind.HEADING
        }
