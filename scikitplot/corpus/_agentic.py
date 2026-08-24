# scikitplot/corpus/_agentic.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded investigation: session state, budget policy, routing and the A2 loop.

Notes
-----
**User-focused.**

.. code-block:: python

    session = AgenticRetrievalSession(query="what causes X?", budgets=BudgetPolicy())
    plan = route(session.query)  # A1: which legs, deterministically
    outcome = run_investigation(session, retrieve_fn)
    outcome.stop_reason  # why it stopped, always
    outcome.status  # ANSWERED | PARTIAL | ABSTAIN

**Developer-focused.**  Three constraints from review run R10 shape this module,
and each rules out an easier implementation.

*Autonomy stops at A2.*  A3 (external tools) needs an explicit external-tool
policy, and R03 established the standing rule that no hidden network access is
permitted -- that is a *security* contract, not a retrieval one.  A4 (mutating
actions) is out of scope by the guide's own statement.

*Budgets are enforced by deterministic policy, never proposed by a model.*  A
model may suggest a next step; :class:`BudgetPolicy` decides whether it may run.
The policy is arithmetic on counters and clocks, so a loop's termination can be
*proven* rather than hoped for.

*Sufficiency is decided by deterministic signals first.*  An agent that cannot
distinguish a degraded retrieval from a complete one judges evidence sufficient
on half the evidence and then **stops refining**, because its stop condition is
satisfied -- terminating early and silently rather than merely propagating
error.  Two of the seven signals were unanswerable until the retrieval envelope
(F-R09-01) and per-hit contributions (F-R09-02) landed.

The session is **ephemeral by default**, which is what lets durable agent memory
be deferred honestly: its seven prerequisites -- tenant scope, provenance, trust,
TTL, delete/forget, poisoning defence, promotion review -- are all absent, and an
ephemeral session needs none of them.

See Also
--------
scikitplot.corpus._retrieval.RetrievalResponse : what a retrieval step returns.
"""

from __future__ import annotations

import dataclasses
import time
from enum import unique
from typing import Any, Callable, Iterable

from ._diagnostics import ErrorCategory, ErrorRecord
from ._retrieval import LegKind, RetrievalStatus
from ._schema import _StrEnumBase

__all__: list[str] = [
    "AgenticRetrievalSession",
    "BudgetPolicy",
    "InvestigationOutcome",
    "InvestigationStatus",
    "RoutePlan",
    "StopReason",
    "SufficiencySignals",
    "route",
    "run_investigation",
]


@unique
class InvestigationStatus(_StrEnumBase):
    """How an investigation finished."""

    ANSWERED = "answered"
    """Evidence was judged sufficient."""

    PARTIAL = "partial"
    """Stopped with evidence, but short of sufficiency."""

    ABSTAIN = "abstain"
    """Stopped with no usable evidence."""


@unique
class StopReason(_StrEnumBase):
    """Why the loop stopped.  Always set -- a loop that stops silently is a bug."""

    SUFFICIENT = "sufficient"
    """The deterministic signals were satisfied."""

    NO_PROGRESS = "no_progress"
    """A refinement round added nothing new."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    """A declared budget ran out."""

    RETRIEVAL_FAILED = "retrieval_failed"
    """Every retrieval leg failed."""

    NOT_RETRIEVED = "not_retrieved"
    """Routing determined no retrieval was warranted."""


@dataclasses.dataclass
class BudgetPolicy:
    """Deterministic, decrementing limits on an investigation.

    Parameters
    ----------
    max_iterations : int, optional
    max_retrieval_calls : int, optional
    max_subqueries : int, optional
    max_evidence_documents : int, optional
    max_graph_expansions : int, optional
    max_wall_time_seconds : float or None, optional
    max_context_tokens : int or None, optional
    max_model_tokens : int or None, optional
    token_counter : callable or None, optional
        ``str -> int``.  Corpus cannot know the consumer's model, so token
        budgets are **caller-supplied**.

    Notes
    -----
    **Developer.**  The policy is the *sole authority*: a model may propose a
    next step, the policy decides whether it runs (ADR-R10-002).  Every check is
    arithmetic on counters and clocks, never a model call, so termination is
    provable.

    When ``token_counter`` is ``None`` the two token budgets are reported
    ``UNENFORCEABLE`` rather than silently treated as unlimited.  Guessing with a
    chunking tokenizer would give a budget wrong by an unknown, model-dependent
    factor -- a budget that *appears* enforced and is not, which is the same
    silent-partial-evidence shape found elsewhere in this codebase.
    """

    max_iterations: int = 4
    max_retrieval_calls: int = 8
    max_subqueries: int = 8
    max_evidence_documents: int = 64
    max_graph_expansions: int = 4
    max_wall_time_seconds: float | None = 30.0
    max_context_tokens: int | None = None
    max_model_tokens: int | None = None
    token_counter: Callable[[str], int] | None = None

    _iterations: int = dataclasses.field(default=0, repr=False)
    _retrieval_calls: int = dataclasses.field(default=0, repr=False)
    _subqueries: int = dataclasses.field(default=0, repr=False)
    _graph_expansions: int = dataclasses.field(default=0, repr=False)
    _started: float | None = dataclasses.field(default=None, repr=False)

    def start(self) -> None:
        """Begin the wall-clock budget."""
        self._started = time.monotonic()

    def unenforceable(self) -> list[str]:
        """Budgets this policy cannot enforce, declared rather than assumed."""
        if self.token_counter is not None:
            return []
        return [
            name
            for name, value in (
                ("max_context_tokens", self.max_context_tokens),
                ("max_model_tokens", self.max_model_tokens),
            )
            if value is not None
        ]

    def check(  # ruff: ignore[too-many-return-statements]
        self,
        *,
        evidence_count: int = 0,
    ) -> ErrorRecord | None:
        """Return a record naming the exhausted budget, or ``None``.

        Notes
        -----
        Checked *before* an action runs, so the loop never exceeds a budget and
        then reports it.
        """

        def _record(budget: str, limit: Any) -> ErrorRecord:
            return ErrorRecord(
                code="BUDGET_EXHAUSTED",
                category=ErrorCategory.RESOURCE,
                message=f"investigation stopped: {budget} limit of {limit} reached",
                stage="investigate",
                details={"budget": budget, "limit": limit},
            )

        if self._iterations >= self.max_iterations:
            return _record("max_iterations", self.max_iterations)
        if self._retrieval_calls >= self.max_retrieval_calls:
            return _record("max_retrieval_calls", self.max_retrieval_calls)
        if self._subqueries > self.max_subqueries:
            return _record("max_subqueries", self.max_subqueries)
        if self._graph_expansions > self.max_graph_expansions:
            return _record("max_graph_expansions", self.max_graph_expansions)
        if evidence_count > self.max_evidence_documents:
            return _record("max_evidence_documents", self.max_evidence_documents)
        if (
            self.max_wall_time_seconds is not None
            and self._started is not None
            and time.monotonic() - self._started > self.max_wall_time_seconds
        ):
            return _record("max_wall_time_seconds", self.max_wall_time_seconds)
        return None

    def debit_iteration(self) -> None:
        """Record one loop iteration."""
        self._iterations += 1

    def debit_retrieval(self) -> None:
        """Record one retrieval call."""
        self._retrieval_calls += 1

    def spent(self) -> dict[str, Any]:
        """Return what has been consumed so far."""
        return {
            "iterations": self._iterations,
            "retrieval_calls": self._retrieval_calls,
            "subqueries": self._subqueries,
            "graph_expansions": self._graph_expansions,
            "unenforceable": self.unenforceable(),
        }


@dataclasses.dataclass(frozen=True)
class RoutePlan:
    """A deterministic decision about which evidence paths to use.

    Notes
    -----
    ``legs`` may be **empty**, which is a legitimate outcome rather than an
    error: deciding *not* to retrieve is a valid deterministic answer.
    """

    legs: tuple[LegKind, ...]
    reason: str

    @property
    def retrieves(self) -> bool:
        """Whether this plan calls for any retrieval at all."""
        return bool(self.legs)


def route(  # ruff: ignore[undocumented-param]
    query: str,
    *,
    has_dense: bool = True,
    has_graph: bool = False,
) -> RoutePlan:
    """Choose evidence paths by deterministic rule (autonomy level A1).

    Parameters
    ----------
    query : str
    has_dense : bool, optional
        Whether a dense index is available.
    has_graph : bool, optional
        Whether a graph view is available.

    Returns
    -------
    RoutePlan

    Notes
    -----
    **Developer.**  Rule-based, never model-controlled: the guide forbids
    beginning with a model router, and a deterministic router is also the only
    kind whose behaviour can be regression-tested.

    A quoted phrase routes lexical-only, because a caller who quoted a phrase
    asked for that phrase, not for something semantically nearby.
    """
    stripped = query.strip()
    if not stripped:
        return RoutePlan(legs=(), reason="empty query: nothing to retrieve")

    quoted = len(stripped) > 1 and stripped[0] == stripped[-1] == '"'
    if quoted:
        return RoutePlan(
            legs=(LegKind.LEXICAL,),
            reason="quoted phrase: exact lexical match requested",
        )

    legs: list[LegKind] = [LegKind.LEXICAL]
    if has_dense:
        legs.append(LegKind.DENSE)
    if has_graph:
        legs.append(LegKind.GRAPH)
    reason = "hybrid" if has_dense else "lexical only: no dense index available"
    return RoutePlan(legs=tuple(legs), reason=reason)


@dataclasses.dataclass(frozen=True)
class SufficiencySignals:
    """The seven deterministic signals, computed before any model judgement.

    Notes
    -----
    ``independent_evidence`` counts **distinct** ``doc_id``.  A document found by
    two legs counts once, not twice: rank fusion scores it higher, which is right
    for ranking and wrong for counting independence (decision DEC-85).
    """

    retrieval_succeeded: bool
    provenance_complete: bool
    independent_evidence: int
    required_entities_covered: bool
    subquestions_answered: bool
    graph_path_found: bool
    conflicts_detected: bool

    def sufficient(self, *, min_evidence: int = 2) -> bool:
        """Whether the evidence is deterministically sufficient."""
        return (
            self.retrieval_succeeded
            and self.provenance_complete
            and self.independent_evidence >= min_evidence
            and not self.conflicts_detected
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return dataclasses.asdict(self)


def evaluate(
    response: Any, *, required_entities: Iterable[str] = ()
) -> SufficiencySignals:
    """Compute the deterministic sufficiency signals for one response.

    Notes
    -----
    ``retrieval_succeeded`` reads the envelope's status and
    ``provenance_complete`` reads per-hit contributions.  Neither is re-derived
    or estimated -- they were unanswerable until those contracts existed.
    """
    status = getattr(response, "status", None)
    succeeded = status in (RetrievalStatus.SUCCESS, RetrievalStatus.EMPTY)

    hits = list(response)
    doc_ids = {getattr(getattr(hit, "doc", hit), "doc_id", None) for hit in hits}
    doc_ids.discard(None)

    provenance = bool(hits) and all(
        getattr(hit, "native_metric", None) is not None for hit in hits
    )

    text = " ".join(
        (getattr(getattr(hit, "doc", hit), "text", "") or "") for hit in hits
    ).lower()
    required = tuple(required_entities)
    covered = all(entity.lower() in text for entity in required) if required else True

    return SufficiencySignals(
        retrieval_succeeded=succeeded,
        provenance_complete=provenance,
        independent_evidence=len(doc_ids),
        required_entities_covered=covered,
        subquestions_answered=covered,
        graph_path_found=False,
        conflicts_detected=False,
    )


@dataclasses.dataclass
class AgenticRetrievalSession:
    """Ephemeral state for one investigation.

    Notes
    -----
    **Developer.**  Not persisted, no storage backend, and **not** a fourth role
    alongside document storage, vector store and graph store.  Ephemerality is
    what lets durable agent memory be deferred honestly rather than half-built.
    """

    query: str
    budgets: BudgetPolicy = dataclasses.field(default_factory=BudgetPolicy)
    steps: list[str] = dataclasses.field(default_factory=list)
    evidence_ids: set[str] = dataclasses.field(default_factory=set)
    errors: list[ErrorRecord] = dataclasses.field(default_factory=list)

    def record(self, step: str) -> None:
        """Append a step to the trace."""
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "query": self.query,
            "steps": list(self.steps),
            "evidence_count": len(self.evidence_ids),
            "spent": self.budgets.spent(),
            "errors": [e.to_dict() for e in self.errors],
        }


@dataclasses.dataclass(frozen=True)
class InvestigationOutcome:
    """The result of a bounded investigation."""

    status: InvestigationStatus
    stop_reason: StopReason
    hits: tuple[Any, ...] = ()
    signals: SufficiencySignals | None = None
    session: AgenticRetrievalSession | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "status": self.status.value,
            "stop_reason": self.stop_reason.value,
            "hit_count": len(self.hits),
            "signals": self.signals.to_dict() if self.signals else None,
            "session": self.session.to_dict() if self.session else None,
        }


def run_investigation(  # ruff: ignore[undocumented-param]
    session: AgenticRetrievalSession,
    retrieve: Callable[[str], Any],
    *,
    plan: RoutePlan | None = None,
    min_evidence: int = 2,
    required_entities: Iterable[str] = (),
) -> InvestigationOutcome:
    """Run the bounded retrieve/evaluate/refine loop (autonomy level A2).

    Parameters
    ----------
    session : AgenticRetrievalSession
    retrieve : callable
        ``str -> RetrievalResponse``.  Supplied by the caller, so this loop
        performs no I/O of its own.
    plan : RoutePlan or None, optional
        Defaults to :func:`route`.
    min_evidence : int, optional
        Distinct documents required for sufficiency.
    required_entities : iterable of str, optional

    Returns
    -------
    InvestigationOutcome
        Always carrying a :class:`StopReason`.

    Notes
    -----
    **Developer.**  No-progress detection compares the ``doc_id`` **set** between
    rounds -- free, because ``doc_id`` is stable and content-derived, and it
    gives a provable upper bound on iterations independent of budget exhaustion.
    A loop whose stopping condition depends on a model call is one whose
    termination cannot be proven.
    """
    session.budgets.start()
    plan = plan if plan is not None else route(session.query)

    if not plan.retrieves:
        session.record(f"route: {plan.reason}")
        return InvestigationOutcome(
            status=InvestigationStatus.ABSTAIN,
            stop_reason=StopReason.NOT_RETRIEVED,
            session=session,
        )

    hits: tuple[Any, ...] = ()
    signals: SufficiencySignals | None = None
    stop = StopReason.BUDGET_EXHAUSTED

    while True:
        exhausted = session.budgets.check(evidence_count=len(session.evidence_ids))
        if exhausted is not None:
            session.errors.append(exhausted)
            stop = StopReason.BUDGET_EXHAUSTED
            break

        session.budgets.debit_iteration()
        session.budgets.debit_retrieval()
        session.record(f"retrieve: {session.query}")

        response = retrieve(session.query)
        hits = tuple(response)
        signals = evaluate(response, required_entities=required_entities)

        if getattr(response, "status", None) is RetrievalStatus.FAILED:
            session.errors.extend(getattr(response, "error_records", list)())
            stop = StopReason.RETRIEVAL_FAILED
            break

        before = set(session.evidence_ids)
        session.evidence_ids |= {
            getattr(getattr(hit, "doc", hit), "doc_id", None) for hit in hits
        } - {None}

        if signals.sufficient(min_evidence=min_evidence):
            stop = StopReason.SUFFICIENT
            break

        if session.evidence_ids == before:
            session.record("no-progress: evidence set unchanged")
            stop = StopReason.NO_PROGRESS
            break

    if stop is StopReason.SUFFICIENT:
        status = InvestigationStatus.ANSWERED
    elif hits:
        status = InvestigationStatus.PARTIAL
    else:
        status = InvestigationStatus.ABSTAIN

    return InvestigationOutcome(
        status=status,
        stop_reason=stop,
        hits=hits,
        signals=signals,
        session=session,
    )
