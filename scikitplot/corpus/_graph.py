# scikitplot/corpus/_graph.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Relationship-aware retrieval over a *derived* document graph.

Notes
-----
**User-focused.**

.. code-block:: python

    from scikitplot.corpus import GraphQuery, derive_graph

    graph = derive_graph(documents)
    response = graph.traverse(GraphQuery(seeds=[doc_id], max_hops=2))
    response.nodes, response.edges, response.status

**Developer-focused.**  Three decisions from the review campaign shape this
module, and each rules out a tempting alternative.

*The graph is derived, never stored* (ADR-R08-002).  Every G0 edge is a pure
function of fields Corpus already persists.  Materialising them would create a
second copy of the same truth, requiring invalidation on every document change
-- a cache-coherence problem in exchange for no new information.  A derived view
also inherits the storage layer's verified restart correctness for free, whereas
a persisted graph would need its own generation identity and atomic publication.

*A chunk is not a node kind* (ADR-R08-001).  A chunk **is** a
:class:`~scikitplot.corpus.CorpusDocument` with ``parent_doc_id`` set -- the
schema says so.  Two node kinds backed by one record type would force every
traversal to reconcile them, reintroducing the duplicate-identity problem
resolved in review run R01.

*Every edge carries provenance, including structural ones* (ADR-R08-003).
Exempting derived edges as "self-evident" was rejected: once G0, G1 and G2 edges
coexist in one traversal, a consumer must filter by trust, which is impossible
if the highest-trust tier is the one that omitted the field.

A caveat recorded rather than hidden: review run R13 established that **no
reader, chunker or pipeline stage currently populates** ``parent_doc_id``.  On
today's corpora the ``parent_of``/``contains`` edge set is therefore empty, and
``same_source`` and ``precedes`` carry the graph on their own.

See Also
--------
scikitplot.corpus._hierarchy : validation and traversal of the same relation.
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from enum import unique
from typing import Any, Iterable

from ._diagnostics import ErrorCategory, ErrorRecord
from ._schema import _StrEnumBase

__all__: list[str] = [
    "EdgeTrust",
    "GraphEdge",
    "GraphNode",
    "GraphQuery",
    "GraphResponse",
    "NodeKind",
    "RelationType",
    "derive_graph",
]


@unique
class NodeKind(_StrEnumBase):
    """What a graph node represents.

    Notes
    -----
    ``chunk`` is deliberately absent.  ``symbol`` and ``topic`` are also absent:
    neither has a producer, and a node kind with no producer is a schema for
    data that cannot be created.
    """

    DOCUMENT = "document"
    """A ``CorpusDocument`` with no parent."""

    SECTION = "section"
    """A ``CorpusDocument`` with ``parent_doc_id`` set."""

    SOURCE = "source"
    """An ``input_path`` + ``source_type`` pair: the physical origin.

    Modelled as a node rather than an implicit property so ``same_source`` is
    *N* edges to one node instead of an *N²* edge set over documents.
    """

    ENTITY = "entity"
    """Reserved for G1 extraction; no producer exists yet."""


@unique
class RelationType(_StrEnumBase):
    """The kind of relationship an edge asserts.

    Notes
    -----
    A generic ``related`` relation is deliberately absent: an edge that does not
    say *how* two things relate cannot be filtered, explained or trusted.
    """

    CONTAINS = "contains"
    """Parent to child, from ``parent_doc_id``."""

    PARENT_OF = "parent_of"
    """Alias direction of :attr:`CONTAINS`, kept explicit for readability."""

    SAME_SOURCE = "same_source"
    """Document to the source it was read from."""

    PRECEDES = "precedes"
    """Sequential order within one parent, from the index fields."""

    REFERENCES = "references"
    """Outbound reference. Requires G1 body-link extraction; not derived at G0."""


@unique
class EdgeTrust(_StrEnumBase):
    """How an edge came to exist."""

    DERIVED = "derived"
    """A pure function of stored fields. The G0 tier."""

    EXTRACTED = "extracted"
    """Produced by a deterministic extractor. The G1 tier."""

    CLAIMED = "claimed"
    """Asserted by a model. The G2 tier; never auto-promoted."""


@dataclasses.dataclass(frozen=True)
class GraphNode:
    """One node in the derived graph."""

    node_id: str
    kind: NodeKind
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {"node_id": self.node_id, "kind": self.kind.value, "label": self.label}


@dataclasses.dataclass(frozen=True)
class GraphEdge:
    """One relationship, with the provenance every edge must carry.

    Parameters
    ----------
    source : str
        Origin ``node_id``.
    target : str
        Destination ``node_id``.
    relation : RelationType
        How the two relate. Never generic.
    evidence : str
        The field(s) this edge was derived from.
    producer : str
        Identity and version of the rule, extractor or model that produced it.
    trust : EdgeTrust
        Which construction tier it belongs to.
    generation : str or None, optional
        Graph generation this edge belongs to.
    """

    source: str
    target: str
    relation: RelationType
    evidence: str
    producer: str
    trust: EdgeTrust
    generation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "evidence": self.evidence,
            "producer": self.producer,
            "trust": self.trust.value,
            "generation": self.generation,
        }


@dataclasses.dataclass(frozen=True)
class GraphQuery:
    """A bounded traversal request.

    Parameters
    ----------
    seeds : tuple of str
        Starting ``node_id`` values.
    max_seed_nodes : int, optional
    max_hops : int, optional
    max_neighbors_per_node : int, optional
    max_total_nodes : int, optional
    max_total_edges : int, optional
    allowed_relation_types : tuple of RelationType or None, optional
        ``None`` admits every relation.
    deadline_seconds : float or None, optional
    max_evidence : int or None, optional
        Evidence budget: cap on returned nodes carrying payload.
    min_trust : EdgeTrust or None, optional
        Exclude edges below this tier.

    Notes
    -----
    **Developer.**  All eight §20 budgets are present with conservative
    defaults, because an unbounded traversal over a derived graph is unbounded
    work.  ``allowed_relation_types`` doubles as the relation-constrained search
    mode, which is why only BFS is implemented: relation-constrained traversal
    is BFS with this field set, not a separate algorithm.
    """

    seeds: tuple[str, ...] = ()
    max_seed_nodes: int = 32
    max_hops: int = 2
    max_neighbors_per_node: int = 64
    max_total_nodes: int = 512
    max_total_edges: int = 2048
    allowed_relation_types: tuple[RelationType, ...] | None = None
    deadline_seconds: float | None = None
    max_evidence: int | None = None
    min_trust: EdgeTrust | None = None

    def __post_init__(self) -> None:
        """Normalise ``seeds`` to a tuple."""
        object.__setattr__(self, "seeds", tuple(self.seeds))


@dataclasses.dataclass(frozen=True)
class GraphResponse:
    """The outcome of a traversal, including why it stopped.

    Notes
    -----
    **Developer.**  A budget-exhausted traversal is ``DEGRADED`` with a naming
    :class:`ErrorRecord`, never a silently truncated result.  A truncated
    traversal that looks complete is the same defect class as a filter that
    returns everything (F-R07-01) -- and this codebase has now removed that
    shape from six separate sites.
    """

    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()
    hops_completed: int = 0

    @property
    def status(self) -> str:
        """``"success"``, ``"empty"`` or ``"degraded"``."""
        if self.errors:
            return "degraded"
        return "success" if self.nodes else "empty"

    @property
    def exhausted_budgets(self) -> list[str]:
        """Names of the budgets that stopped this traversal."""
        return [record.details.get("budget", "?") for record in self.errors]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "status": self.status,
            "hops_completed": self.hops_completed,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "errors": [e.to_dict() for e in self.errors],
        }


#: Identity of the G0 derivation rules, recorded on every edge they produce.
G0_PRODUCER = "corpus.g0/1"


def _source_node_id(doc: Any) -> str:
    """Return the node id for a document's physical origin."""
    source_type = getattr(getattr(doc, "source_type", None), "value", "unknown")
    return f"source:{source_type}:{getattr(doc, 'input_path', '')}"


def _order_key(doc: Any) -> tuple[int, int, int]:
    """Return a sequential ordering key from the index fields."""
    return (
        getattr(doc, "chunk_index", 0) or 0,
        getattr(doc, "paragraph_index", 0) or 0,
        getattr(doc, "frame_index", 0) or 0,
    )


class DerivedGraph:
    """An in-memory, read-only G0 view over a document collection.

    Notes
    -----
    Built once from the documents supplied; it holds no independent state and is
    never persisted.  Rebuild it when the documents change -- which is cheap,
    and cheaper than keeping a second copy correct.
    """

    __slots__ = ("_generation", "_nodes", "_out")

    def __init__(
        self,
        nodes: dict[str, GraphNode],
        edges: Iterable[GraphEdge],
        generation: str | None = None,
    ) -> None:
        self._nodes = dict(nodes)
        self._generation = generation
        self._out: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            self._out[edge.source].append(edge)

    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges."""
        return sum(len(v) for v in self._out.values())

    def node(self, node_id: str) -> GraphNode | None:
        """Return one node, or ``None``."""
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> list[GraphEdge]:
        """Return the outbound edges of ``node_id``."""
        return list(self._out.get(node_id, ()))

    def traverse(  # ruff: ignore[too-many-branches]
        self,
        query: GraphQuery,
    ) -> GraphResponse:
        """Breadth-first traversal under every budget in ``query``.

        Returns
        -------
        GraphResponse
            Nodes and edges reached, plus a record for each budget exhausted.

        Notes
        -----
        Only BFS is implemented.  Relation-constrained traversal is BFS with
        ``allowed_relation_types`` set; weighted best-first would need edge
        weights G0 does not produce, and bidirectional search would need a
        labelled connection query that does not exist.
        """
        started = time.monotonic()
        errors: list[ErrorRecord] = []

        def _exhausted(budget: str, limit: Any) -> None:
            errors.append(
                ErrorRecord(
                    code="GRAPH_BUDGET_EXHAUSTED",
                    category=ErrorCategory.RESOURCE,
                    message=f"traversal stopped: {budget} limit of {limit} reached",
                    stage="traverse",
                    details={"budget": budget, "limit": limit},
                )
            )

        seeds = [s for s in query.seeds if s in self._nodes]
        if len(seeds) > query.max_seed_nodes:
            _exhausted("max_seed_nodes", query.max_seed_nodes)
            seeds = seeds[: query.max_seed_nodes]

        visited: dict[str, GraphNode] = {}
        collected: list[GraphEdge] = []
        frontier = list(seeds)
        for node_id in frontier:
            visited[node_id] = self._nodes[node_id]

        allowed = (
            set(query.allowed_relation_types)
            if query.allowed_relation_types is not None
            else None
        )
        trust_order = {
            EdgeTrust.DERIVED: 3,
            EdgeTrust.EXTRACTED: 2,
            EdgeTrust.CLAIMED: 1,
        }
        floor = trust_order.get(query.min_trust, 0) if query.min_trust else 0

        hops = 0
        stop = False
        while frontier and hops < query.max_hops and not stop:
            hops += 1
            following: list[str] = []
            for node_id in frontier:
                if query.deadline_seconds is not None and (
                    time.monotonic() - started > query.deadline_seconds
                ):
                    _exhausted("deadline_seconds", query.deadline_seconds)
                    stop = True
                    break

                outgoing = self.neighbors(node_id)
                if allowed is not None:
                    outgoing = [e for e in outgoing if e.relation in allowed]
                if floor:
                    outgoing = [e for e in outgoing if trust_order[e.trust] >= floor]

                if len(outgoing) > query.max_neighbors_per_node:
                    _exhausted("max_neighbors_per_node", query.max_neighbors_per_node)
                    outgoing = outgoing[: query.max_neighbors_per_node]

                for edge in outgoing:
                    if len(collected) >= query.max_total_edges:
                        _exhausted("max_total_edges", query.max_total_edges)
                        stop = True
                        break
                    collected.append(edge)
                    if edge.target in visited:
                        continue
                    if len(visited) >= query.max_total_nodes:
                        _exhausted("max_total_nodes", query.max_total_nodes)
                        stop = True
                        break
                    target = self._nodes.get(edge.target)
                    if target is not None:
                        visited[edge.target] = target
                        following.append(edge.target)
                if stop:
                    break
            frontier = following

        if frontier and hops >= query.max_hops:
            _exhausted("max_hops", query.max_hops)

        nodes = list(visited.values())
        if query.max_evidence is not None and len(nodes) > query.max_evidence:
            _exhausted("max_evidence", query.max_evidence)
            nodes = nodes[: query.max_evidence]

        return GraphResponse(
            nodes=tuple(nodes),
            edges=tuple(collected),
            errors=tuple(errors),
            hops_completed=hops,
        )


def derive_graph(
    documents: Iterable[Any],
    *,
    generation: str | None = None,
) -> DerivedGraph:
    """Build the G0 view from document fields.

    Parameters
    ----------
    documents : Iterable[Any]
        iterable of CorpusDocument
    generation : str or None, optional
        Recorded on every edge for provenance.

    Returns
    -------
    DerivedGraph

    Notes
    -----
    Four relations are derived, each from fields Corpus already stores:

    ``contains`` / ``parent_of``
        from ``parent_doc_id``.  **Empty on today's corpora** -- no producer sets
        that field yet (R13 / unknown U-11).
    ``same_source``
        from ``input_path`` and ``source_type``.
    ``precedes``
        from ``chunk_index`` / ``paragraph_index`` / ``frame_index`` within one
        parent.

    ``references`` is *not* derived: ``CorpusDocument`` has a single ``url``
    field -- its own source URL, not outbound links -- so true reference edges
    need G1 body-link extraction (finding F-R08-03).
    """
    docs = list(documents)
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    def _edge(src: str, dst: str, relation: RelationType, evidence: str) -> None:
        edges.append(
            GraphEdge(
                source=src,
                target=dst,
                relation=relation,
                evidence=evidence,
                producer=G0_PRODUCER,
                trust=EdgeTrust.DERIVED,
                generation=generation,
            )
        )

    by_parent: dict[str, list[Any]] = defaultdict(list)
    for doc in docs:
        doc_id = getattr(doc, "doc_id", None)
        if not doc_id:
            continue
        parent_id = getattr(doc, "parent_doc_id", None)
        nodes[doc_id] = GraphNode(
            node_id=doc_id,
            kind=NodeKind.SECTION if parent_id else NodeKind.DOCUMENT,
            label=getattr(doc, "input_path", None),
        )
        source_id = _source_node_id(doc)
        nodes.setdefault(
            source_id,
            GraphNode(node_id=source_id, kind=NodeKind.SOURCE, label=source_id),
        )
        _edge(doc_id, source_id, RelationType.SAME_SOURCE, "input_path+source_type")
        by_parent[parent_id or source_id].append(doc)

    for doc in docs:
        doc_id = getattr(doc, "doc_id", None)
        parent_id = getattr(doc, "parent_doc_id", None)
        if doc_id and parent_id and parent_id in nodes:
            _edge(parent_id, doc_id, RelationType.CONTAINS, "parent_doc_id")
            _edge(doc_id, parent_id, RelationType.PARENT_OF, "parent_doc_id")

    for group in by_parent.values():
        ordered = sorted(group, key=_order_key)
        for earlier, later in zip(ordered, ordered[1:]):
            _edge(
                earlier.doc_id,
                later.doc_id,
                RelationType.PRECEDES,
                "chunk_index/paragraph_index/frame_index",
            )

    return DerivedGraph(nodes, edges, generation=generation)
