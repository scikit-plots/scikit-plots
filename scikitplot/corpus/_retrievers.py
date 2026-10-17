# scikitplot/corpus/_retrievers.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Independent evidence paths: retriever legs and deterministic rank fusion.

Notes
-----
**User-focused.**  Each leg answers a query on its own and reports how it went::

    lexical = LexicalRetriever(index)
    dense = DenseRetriever(index)

    result = lexical.retrieve("quantum", config)
    result.hits, result.outcome.status

    fused = fuse_by_rank([lexical_result, dense_result], top_k=5)
    hit, score, contributions = fused[0]  # which legs found it, and at what rank

**Developer-focused.**  Two findings shape this module.

*Legs were private methods* (finding F-R09-03).  ``_search_keyword`` and
``_search_semantic`` lived on :class:`~scikitplot.corpus.RetrievalIndex`, a class
that already owns document storage, embedding stacking, backend selection,
generation derivation and four search modes.  Adding the graph leg would have
made it a sixth concern.  Extraction is also what makes per-leg status natural: a
leg *object* can have a status, whereas a private method can only return a list.

*Per-retriever rank was computed and discarded* (finding F-R09-02).  The fusion
loop's ``enumerate`` calls already produced exactly which leg found a hit and at
what rank; both were used to compute the score and then dropped.  A caller could
not distinguish a hit that ranked #1 in both legs from one that ranked #1 and
#40 -- very different confidence signals collapsed into a single float, and a
fused score that could not be recomputed or explained from the result.

The fusion rule is :class:`~scikitplot.corpus.FusionPolicy` (R2-1), the one
weighted reciprocal rank fusion that :meth:`RetrievalIndex.search` and the MCP
hybrid retriever also call: ``w / (k + rank)`` with ranks from 1, keyed on
``doc_id`` rather than row offset, a document once per leg at its best rank,
weight-0 legs adding nothing, exact ties to the lower identity.

See Also
--------
scikitplot.corpus._retrieval.RetrievalResponse : the envelope legs report into.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable

from ._diagnostics import ErrorCategory, ErrorRecord
from ._fusion import FusionPolicy
from ._retrieval import LegKind, LegOutcome, LegStatus

__all__: list[str] = [
    "DenseRetriever",
    "GraphRetriever",
    "LegContribution",
    "LegResult",
    "LexicalRetriever",
    "Retriever",
    "fuse_by_rank",
]


@dataclasses.dataclass(frozen=True)
class LegContribution:
    """One leg's contribution to a fused hit.

    Parameters
    ----------
    leg : LegKind
        Which evidence path contributed.
    rank : int
        Zero-based position within that leg.
    native_score : float or None
        The score that leg assigned, on its own scale.
    native_metric : str or None
        The scale, so a consumer can verify no cross-metric comparison happened.

    Notes
    -----
    **Developer.**  This is propagation, not new computation: the fusion loop
    already had all four values.
    """

    leg: LegKind
    rank: int
    native_score: float | None = None
    native_metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "leg": self.leg.value,
            "rank": self.rank,
            "native_score": self.native_score,
            "native_metric": self.native_metric,
        }


@dataclasses.dataclass(frozen=True)
class LegResult:
    """What one leg produced, and how it went."""

    hits: tuple[Any, ...]
    outcome: LegOutcome

    def __iter__(self):
        """Iterate the hits."""
        return iter(self.hits)

    def __len__(self) -> int:
        """Return the number of hits."""
        return len(self.hits)


class Retriever:
    """Base class for an independent evidence path.

    Notes
    -----
    Subclasses implement :meth:`retrieve` and declare :attr:`leg`.  Every leg
    returns a :class:`LegResult` rather than a bare list, so a leg that could not
    run says so instead of returning an empty list indistinguishable from a
    genuine zero-hit answer.
    """

    #: Which evidence path this retriever represents.
    leg: LegKind = LegKind.LEXICAL

    def retrieve(self, query: str, config: Any, **kwargs: Any) -> LegResult:
        """Run this leg."""
        raise NotImplementedError

    def _ok(self, hits: Iterable[Any], **extra: Any) -> LegResult:
        """Build a successful or empty outcome."""
        materialised = tuple(hits)
        return LegResult(
            hits=materialised,
            outcome=LegOutcome(
                leg=self.leg,
                status=LegStatus.SUCCESS if materialised else LegStatus.EMPTY,
                hit_count=len(materialised),
                **extra,
            ),
        )

    def _failed(self, record: ErrorRecord, **extra: Any) -> LegResult:
        """Build a failed outcome carrying its explanation."""
        return LegResult(
            hits=(),
            outcome=LegOutcome(
                leg=self.leg, status=LegStatus.FAILED, error=record, **extra
            ),
        )


class LexicalRetriever(Retriever):
    """Keyword / BM25 evidence path."""

    leg = LegKind.LEXICAL

    def __init__(self, index: Any) -> None:
        self._index = index

    def retrieve(self, query: str, config: Any, **kwargs: Any) -> LegResult:
        """Run keyword retrieval against the index."""
        hits = self._index._search_keyword(query, config)
        return self._ok(hits, generation=self._index.index_generation)


class DenseRetriever(Retriever):
    """Vector-similarity evidence path."""

    leg = LegKind.DENSE

    def __init__(self, index: Any) -> None:
        self._index = index

    def retrieve(
        self, query: str, config: Any, *, query_embedding: Any = None, **kwargs: Any
    ) -> LegResult:
        """Run dense retrieval, reporting why it could not run when it cannot."""
        reason = self._index._dense_unavailable_reason(query_embedding)
        if reason is not None:
            return self._failed(reason, generation=self._index.index_generation)
        hits = self._index._search_semantic(query, query_embedding, config)
        # R2-5: the leg says what was asked for and why something else ran,
        # from the same source as RetrievalIndex's own dense legs.
        fields = getattr(self._index, "_backend_fields", None)
        extra = (
            fields()
            if callable(fields)
            else {"backend": getattr(self._index, "backend_name", None)}
        )
        return self._ok(hits, generation=self._index.index_generation, **extra)


class GraphRetriever(Retriever):
    """Relationship-traversal evidence path.

    Notes
    -----
    **Developer.**  Seeds come from another leg's hits: the graph answers *what
    is related to what already matched*, so it is a refinement path rather than a
    primary one.  A traversal that exhausts a budget produces a ``DEGRADED``
    leg carrying the traversal's own :class:`ErrorRecord`, so the budget that
    stopped it survives into the retrieval envelope.
    """

    leg = LegKind.GRAPH

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def retrieve(
        self, query: str, config: Any, *, graph_query: Any = None, **kwargs: Any
    ) -> LegResult:
        """Traverse the graph from the supplied seeds."""
        if graph_query is None:
            return self._failed(
                ErrorRecord(
                    code="NO_GRAPH_SEEDS",
                    category=ErrorCategory.CAPABILITY,
                    message=(
                        "graph retrieval requires seed nodes; none were supplied "
                        "and no prior leg produced any"
                    ),
                    stage="retrieve",
                )
            )
        response = self._graph.traverse(graph_query)
        if response.errors:
            return LegResult(
                hits=tuple(response.nodes),
                outcome=LegOutcome(
                    leg=self.leg,
                    status=LegStatus.DEGRADED,
                    hit_count=len(response.nodes),
                    error=response.errors[0],
                ),
            )
        return self._ok(response.nodes)


def _fusion_key(hit: Any, leg: Any, rank: int) -> str:
    """Return the identity a hit is fused under: ``doc_id``, else ``node_id``."""
    key = getattr(getattr(hit, "doc", hit), "doc_id", None) or getattr(
        hit, "node_id", None
    )
    if not isinstance(key, str) or not key:
        raise ValueError(
            f"leg {leg!s} rank {rank}: a hit has no doc_id or node_id, so it has "
            "no identity to fuse under"
        )
    return key


def fuse_by_rank(
    results: Iterable[LegResult],
    *,
    rrf_k: int = 60,
    weights: dict[LegKind, float] | None = None,
    top_k: int = 10,
) -> list[tuple[Any, float, tuple[LegContribution, ...]]]:
    """Combine leg results by weighted reciprocal rank fusion.

    Parameters
    ----------
    results : iterable of LegResult
        One entry per leg. Failed legs contribute nothing but are harmless.
    rrf_k : int, optional
        The standard RRF constant, at least 1. Default 60.
    weights : dict, optional
        Per-leg weight, finite and non-negative; unlisted legs weigh ``1.0``.
        A weight-0 leg contributes nothing and adds no document.
    top_k : int, optional
        Number of fused hits to return.

    Returns
    -------
    list of (hit, fused_score, contributions)
        Descending fused score; ties by ascending identity (``doc_id`` or
        ``node_id``). ``hit`` is the first hit seen for that identity.

    Raises
    ------
    ValueError
        If a hit has neither a ``doc_id`` nor a ``node_id``, or a weight or
        ``rrf_k`` is out of range.
    TypeError
        If a weight or ``rrf_k`` has the wrong type.

    Notes
    -----
    **Developer.**  R2-1: the arithmetic and the tie-break are
    :class:`~scikitplot.corpus.FusionPolicy`, the one rule shared with
    :meth:`RetrievalIndex.search` and the MCP hybrid retriever. Rank fusion is
    the default combiner because score-space fusion needs a validated
    normalization that does not exist for most metrics (ADR-R07-003). Each
    hit's contributions are returned, so the fused score can be recomputed and
    explained from the result.

    Fusion keys on ``doc_id``, never on a row offset -- the property that kept
    F-R01-07's positional coupling out of the fusion path. A hit without an
    identity is refused: an object address is not stable across processes.
    """
    policy = FusionPolicy(
        k=rrf_k,
        weights={LegKind(kind).value: w for kind, w in (weights or {}).items()},
        default_weight=1.0,
        top_k=top_k,
    )
    # A weight-0 leg adds nothing, so its hits are not asked for an identity.
    legs = [
        (
            result.outcome.leg,
            (
                list(result.hits)
                if policy.weight(LegKind(result.outcome.leg).value)
                else []
            ),
        )
        for result in results
    ]
    first_hit: list[dict[str, Any]] = []
    keys: list[list[str]] = []
    for kind, leg_hits in legs:
        by_key: dict[str, Any] = {}
        leg_keys = []
        for rank, hit in enumerate(leg_hits, start=1):
            key = _fusion_key(hit, kind, rank)
            leg_keys.append(key)
            by_key.setdefault(key, hit)
        first_hit.append(by_key)
        keys.append(leg_keys)
    fused = policy.fuse(
        (LegKind(kind).value, leg_keys) for (kind, _hits), leg_keys in zip(legs, keys)
    )
    out = []
    for item in fused:
        contributions = []
        for part in item.contributions:
            hit = first_hit[part.position][item.key]
            contributions.append(
                LegContribution(
                    leg=LegKind(part.leg),
                    rank=part.rank - 1,
                    native_score=getattr(hit, "native_score", None),
                    native_metric=getattr(hit, "native_metric", None),
                )
            )
        hit = first_hit[item.contributions[0].position][item.key]
        out.append((hit, item.score, tuple(contributions)))
    return out
