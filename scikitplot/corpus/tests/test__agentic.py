# scikitplot/corpus/tests/test__agentic.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._agentic` — routing, budgets and the A2 loop.

Covers findings F-R10-01 (entry condition) and F-R10-02 (no budget enforcement
point), and the decisions ADR-R10-001 through ADR-R10-004.
"""

from __future__ import annotations

import json

import pytest

from .._agentic import (
    AgenticRetrievalSession,
    BudgetPolicy,
    InvestigationStatus,
    StopReason,
    evaluate,
    route,
    run_investigation,
)
from .._retrieval import LegKind, RetrievalResponse, RetrievalStatus
from .._schema import CorpusDocument
from .._similarity._similarity import RetrievalConfig, RetrievalIndex

__all__: "list[str]" = [
    "TestRouter",
    "TestBudgetPolicy",
    "TestSufficiency",
    "TestInvestigationLoop",
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


@pytest.fixture(name="fetch")
def _fetch():
    index = RetrievalIndex()
    index.build(_docs())

    def _run(query: str):
        return index.search(
            query,
            config=RetrievalConfig(top_k=3, match_mode="hybrid"),
            query_embedding=[1.0, 1.0, 0.0],
        )

    return _run


class TestRouter:
    """A1 — rule-based, never model-controlled."""

    def test_default_is_hybrid(self) -> None:
        plan = route("quantum")
        assert plan.legs == (LegKind.LEXICAL, LegKind.DENSE)

    def test_quoted_phrase_routes_lexical_only(self) -> None:
        """A caller who quoted a phrase asked for that phrase."""
        plan = route('"exact phrase"')
        assert plan.legs == (LegKind.LEXICAL,)
        assert "quoted" in plan.reason

    def test_absent_dense_index_drops_the_dense_leg(self) -> None:
        assert route("quantum", has_dense=False).legs == (LegKind.LEXICAL,)

    def test_graph_leg_is_added_when_available(self) -> None:
        assert LegKind.GRAPH in route("quantum", has_graph=True).legs

    def test_routing_to_none_is_legitimate(self) -> None:
        """Deciding not to retrieve is a valid deterministic outcome, not an error."""
        plan = route("   ")
        assert plan.legs == ()
        assert plan.retrieves is False

    def test_is_deterministic(self) -> None:
        assert route("quantum") == route("quantum")


class TestBudgetPolicy:
    """F-R10-02 — there was no cross-call budget object at all."""

    def test_iteration_budget_stops_the_loop(self, fetch) -> None:
        session = AgenticRetrievalSession(
            query="quantum", budgets=BudgetPolicy(max_iterations=0)
        )
        outcome = run_investigation(session, fetch)
        assert outcome.stop_reason is StopReason.BUDGET_EXHAUSTED
        assert session.errors[0].details["budget"] == "max_iterations"

    def test_retrieval_call_budget_stops_the_loop(self, fetch) -> None:
        session = AgenticRetrievalSession(
            query="quantum",
            budgets=BudgetPolicy(max_iterations=9, max_retrieval_calls=0),
        )
        outcome = run_investigation(session, fetch)
        assert outcome.stop_reason is StopReason.BUDGET_EXHAUSTED

    def test_budget_is_checked_before_acting_not_after(self, fetch) -> None:
        """The loop must never exceed a budget and then report it."""
        session = AgenticRetrievalSession(
            query="quantum", budgets=BudgetPolicy(max_iterations=0)
        )
        run_investigation(session, fetch)
        assert session.budgets.spent()["retrieval_calls"] == 0

    def test_token_budgets_are_declared_unenforceable_not_assumed(self) -> None:
        """Corpus cannot know the consumer's model.

        Guessing with a chunking tokenizer would give a budget wrong by an
        unknown factor -- one that *appears* enforced and is not.
        """
        assert BudgetPolicy(max_context_tokens=1000).unenforceable() == [
            "max_context_tokens"
        ]
        assert (
            BudgetPolicy(max_context_tokens=1000, token_counter=len).unenforceable()
            == []
        )

    def test_no_token_budget_means_nothing_unenforceable(self) -> None:
        assert BudgetPolicy().unenforceable() == []

    def test_spent_is_serialisable(self, fetch) -> None:
        session = AgenticRetrievalSession(query="quantum")
        run_investigation(session, fetch)
        json.dumps(session.budgets.spent())


class TestSufficiency:
    """The two signals that were unanswerable before the retrieval envelope."""

    def test_reads_status_and_contributions_rather_than_estimating(
        self, fetch
    ) -> None:
        signals = evaluate(fetch("quantum"))
        assert signals.retrieval_succeeded is True
        assert signals.provenance_complete is True

    def test_a_degraded_retrieval_is_not_a_success(self) -> None:
        """The whole reason R10 blocked on F-R09-01.

        An agent that reads a degraded retrieval as complete judges evidence
        sufficient on half the evidence and then stops refining.
        """
        from .._diagnostics import ErrorCategory, ErrorRecord
        from .._retrieval import LegOutcome, LegStatus

        # One leg answered, the other could not: DEGRADED, not FAILED.
        degraded = RetrievalResponse(
            hits=[],
            legs=[
                LegOutcome(LegKind.LEXICAL, LegStatus.EMPTY),
                LegOutcome(
                    LegKind.DENSE,
                    LegStatus.FAILED,
                    error=ErrorRecord(
                        code="X", category=ErrorCategory.CAPABILITY, message="m"
                    ),
                ),
            ],
        )
        assert degraded.status is RetrievalStatus.DEGRADED
        assert evaluate(degraded).retrieval_succeeded is False

        # And an all-legs-empty response IS a success: nothing was prevented
        # from running, so zero hits is a real answer rather than a failure.
        empty = RetrievalResponse(
            hits=[], legs=[LegOutcome(LegKind.LEXICAL, LegStatus.EMPTY)]
        )
        assert empty.status is RetrievalStatus.EMPTY
        assert evaluate(empty).retrieval_succeeded is True

    def test_independent_evidence_counts_distinct_documents(self, fetch) -> None:
        """A document found by two legs counts once, not twice.

        Rank fusion scores it higher, which is right for ranking and wrong for
        counting independence.
        """
        signals = evaluate(fetch("quantum"))
        assert signals.independent_evidence == 3

    def test_sufficiency_requires_every_hard_signal(self) -> None:
        from .._agentic import SufficiencySignals

        base = dict(
            retrieval_succeeded=True,
            provenance_complete=True,
            independent_evidence=5,
            required_entities_covered=True,
            subquestions_answered=True,
            graph_path_found=False,
            conflicts_detected=False,
        )
        assert SufficiencySignals(**base).sufficient()
        assert not SufficiencySignals(**{**base, "retrieval_succeeded": False}).sufficient()
        assert not SufficiencySignals(**{**base, "provenance_complete": False}).sufficient()
        assert not SufficiencySignals(**{**base, "conflicts_detected": True}).sufficient()
        assert not SufficiencySignals(**{**base, "independent_evidence": 0}).sufficient()


class TestInvestigationLoop:
    """A2 — bounded retrieve / evaluate / refine."""

    def test_sufficient_evidence_answers(self, fetch) -> None:
        outcome = run_investigation(AgenticRetrievalSession(query="quantum"), fetch)
        assert outcome.status is InvestigationStatus.ANSWERED
        assert outcome.stop_reason is StopReason.SUFFICIENT

    def test_no_progress_terminates(self, fetch) -> None:
        """A provable upper bound on iterations, independent of budgets.

        A loop whose stopping condition depends on a model call is one whose
        termination cannot be proven.
        """
        session = AgenticRetrievalSession(query="quantum")
        outcome = run_investigation(session, fetch, min_evidence=99)
        assert outcome.stop_reason is StopReason.NO_PROGRESS
        assert outcome.status is InvestigationStatus.PARTIAL

    def test_routing_to_none_abstains_without_retrieving(self, fetch) -> None:
        session = AgenticRetrievalSession(query="")
        outcome = run_investigation(session, fetch)
        assert outcome.status is InvestigationStatus.ABSTAIN
        assert outcome.stop_reason is StopReason.NOT_RETRIEVED
        assert session.budgets.spent()["retrieval_calls"] == 0

    def test_a_stop_reason_is_always_present(self, fetch) -> None:
        """A loop that stops without saying why is a bug."""
        for query, kwargs in (("quantum", {}), ("quantum", {"min_evidence": 99}), ("", {})):
            outcome = run_investigation(
                AgenticRetrievalSession(query=query), fetch, **kwargs
            )
            assert isinstance(outcome.stop_reason, StopReason)

    def test_session_records_its_trace(self, fetch) -> None:
        session = AgenticRetrievalSession(query="quantum")
        run_investigation(session, fetch)
        assert any("retrieve" in step for step in session.steps)

    def test_outcome_is_serialisable(self, fetch) -> None:
        outcome = run_investigation(AgenticRetrievalSession(query="quantum"), fetch)
        json.dumps(outcome.to_dict())

    def test_loop_performs_no_io_of_its_own(self) -> None:
        """Retrieval is injected, so the loop never reaches the network itself."""
        calls = []

        def _spy(query: str):
            calls.append(query)
            return RetrievalResponse(hits=[])

        run_investigation(AgenticRetrievalSession(query="q"), _spy)
        assert calls == ["q"]
