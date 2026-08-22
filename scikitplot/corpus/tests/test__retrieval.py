# scikitplot/corpus/tests/test__retrieval.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._retrieval` and the F-R09-01 regression gate.

The gate at the end of this module is the important one: it reproduces the
defect exactly as review run R09 measured it and asserts the two outcomes are
now distinguishable.
"""

from __future__ import annotations

import json

import pytest

from .._diagnostics import ErrorCategory, ErrorRecord
from .._retrieval import (
    LegKind,
    LegOutcome,
    LegStatus,
    RetrievalResponse,
    RetrievalStatus,
)
from .._schema import CorpusDocument
from .._similarity._similarity import RetrievalConfig, RetrievalIndex

__all__: "list[str]" = [
    "TestLegOutcome",
    "TestStatusDerivation",
    "TestSequenceBehaviour",
    "TestHybridDegradationGate",
]


def _err(code: str = "X") -> ErrorRecord:
    return ErrorRecord(
        code=code, category=ErrorCategory.CAPABILITY, message="unavailable"
    )


class TestLegOutcome:
    """A degraded leg must explain itself."""

    @pytest.mark.parametrize("status", [LegStatus.DEGRADED, LegStatus.FAILED])
    def test_degraded_without_error_is_rejected(self, status: LegStatus) -> None:
        with pytest.raises(ValueError, match="no ErrorRecord"):
            LegOutcome(leg=LegKind.DENSE, status=status)

    @pytest.mark.parametrize(
        "status", [LegStatus.SUCCESS, LegStatus.EMPTY, LegStatus.SKIPPED]
    )
    def test_non_degraded_needs_no_error(self, status: LegStatus) -> None:
        assert LegOutcome(leg=LegKind.DENSE, status=status).error is None

    def test_ran_property(self) -> None:
        assert LegOutcome(LegKind.LEXICAL, LegStatus.SUCCESS).ran
        assert LegOutcome(LegKind.LEXICAL, LegStatus.EMPTY).ran
        assert not LegOutcome(LegKind.LEXICAL, LegStatus.SKIPPED).ran


class TestStatusDerivation:
    """The DEC-70 derivation rule, exhaustively."""

    def test_all_requested_failed_is_failed(self) -> None:
        resp = RetrievalResponse(
            hits=[],
            legs=[LegOutcome(LegKind.DENSE, LegStatus.FAILED, error=_err())],
        )
        assert resp.status is RetrievalStatus.FAILED

    def test_any_failed_leg_makes_it_degraded(self) -> None:
        resp = RetrievalResponse(
            hits=["h"],
            legs=[
                LegOutcome(LegKind.LEXICAL, LegStatus.SUCCESS, hit_count=1),
                LegOutcome(LegKind.DENSE, LegStatus.FAILED, error=_err()),
            ],
        )
        assert resp.status is RetrievalStatus.DEGRADED

    def test_empty_requires_every_requested_leg_to_have_run(self) -> None:
        """The binding rule from ADR-R02-002.

        Zero hits is only EMPTY when nothing was prevented from running;
        otherwise it is DEGRADED. This is the distinction the whole envelope
        exists to make.
        """
        all_ran = RetrievalResponse(
            hits=[],
            legs=[
                LegOutcome(LegKind.LEXICAL, LegStatus.EMPTY),
                LegOutcome(LegKind.DENSE, LegStatus.EMPTY),
            ],
        )
        assert all_ran.status is RetrievalStatus.EMPTY

        one_blocked = RetrievalResponse(
            hits=[],
            legs=[
                LegOutcome(LegKind.LEXICAL, LegStatus.EMPTY),
                LegOutcome(LegKind.DENSE, LegStatus.FAILED, error=_err()),
            ],
        )
        assert one_blocked.status is RetrievalStatus.DEGRADED

    def test_skipped_legs_do_not_degrade(self) -> None:
        """A leg the caller never asked for is not a failure.

        Without this, every keyword-only search would report DEGRADED.
        """
        resp = RetrievalResponse(
            hits=["h"],
            legs=[
                LegOutcome(LegKind.LEXICAL, LegStatus.SUCCESS, hit_count=1),
                LegOutcome(LegKind.DENSE, LegStatus.SKIPPED),
            ],
        )
        assert resp.status is RetrievalStatus.SUCCESS

    def test_explicit_terminal_status_is_respected(self) -> None:
        resp = RetrievalResponse(hits=[], status=RetrievalStatus.CANCELLED)
        assert resp.status is RetrievalStatus.CANCELLED

    def test_is_serialisable(self) -> None:
        resp = RetrievalResponse(
            hits=["h"],
            legs=[LegOutcome(LegKind.DENSE, LegStatus.FAILED, error=_err())],
            query="q",
        )
        assert json.dumps(resp.to_dict())


class TestSequenceBehaviour:
    """The envelope stands in for the list it replaced."""

    def test_len_iter_index_and_bool(self) -> None:
        resp = RetrievalResponse(hits=["a", "b"])
        assert len(resp) == 2
        assert list(resp) == ["a", "b"]
        assert resp[0] == "a"
        assert bool(resp) is True
        assert not RetrievalResponse(hits=[])


def _corpus():
    """Three documents; the third is reachable only through the dense leg."""
    texts = ["quantum physics theory", "classical physics motion", "quantum computing qubits"]
    return [
        CorpusDocument.create(
            input_path=f"f{i}.txt",
            chunk_index=i,
            text=text,
            embedding=[1.0 if i == 0 else 0.2, 0.5, 0.1],
        )
        for i, text in enumerate(texts)
    ]


class TestHybridDegradationGate:
    """Regression gate for F-R09-01."""

    def test_hybrid_without_embedding_is_degraded_not_silent(self) -> None:
        """The exact defect review run R09 measured.

        Before the envelope, both calls below returned ``match_mode="hybrid"``
        results with no status: the second had one fewer hit and every score
        exactly halved by the missing ``hybrid_alpha`` contribution, and nothing
        reported it.
        """
        index = RetrievalIndex()
        index.build(_corpus())
        cfg = RetrievalConfig(top_k=3, match_mode="hybrid")

        complete = index.search("quantum", config=cfg, query_embedding=[1.0, 0.5, 0.1])
        partial = index.search("quantum", config=cfg, query_embedding=None)

        # The observable difference that used to be invisible.
        assert len(complete) > len(partial)
        assert complete[0].score > partial[0].score

        # ...is now reported.
        assert complete.status is RetrievalStatus.SUCCESS
        assert partial.status is RetrievalStatus.DEGRADED
        assert complete.status is not partial.status

        dense = partial.leg(LegKind.DENSE)
        assert dense.status is LegStatus.FAILED
        assert dense.error.code == "NO_QUERY_EMBEDDING"
        assert partial.leg(LegKind.LEXICAL).status is LegStatus.SUCCESS

        # A degraded response always explains itself.
        assert partial.error_records()
        assert partial.degraded_legs

    def test_semantic_without_embedding_raises_but_no_index_degrades(self) -> None:
        """Caller error raises; corpus-state degrades.

        An explicit ``semantic`` query with no embedding is the caller failing to
        supply what they asked to search with, so it raises. A corpus with no
        dense index at all is a state condition and is reported through the
        envelope instead.
        """
        with_vectors = RetrievalIndex()
        with_vectors.build(_corpus())
        cfg = RetrievalConfig(match_mode="semantic")

        with pytest.raises(ValueError, match="query_embedding"):
            with_vectors.search("q", config=cfg, query_embedding=None)

        no_vectors = RetrievalIndex()
        no_vectors.build(
            [CorpusDocument.create(input_path="a.txt", chunk_index=0, text="text")]
        )
        response = no_vectors.search("q", config=cfg, query_embedding=None)
        assert response.status is RetrievalStatus.FAILED
        assert response.leg(LegKind.DENSE).error.code == "NO_DENSE_INDEX"

    def test_keyword_only_search_is_not_degraded(self) -> None:
        """A leg that was never requested must not force DEGRADED."""
        index = RetrievalIndex()
        index.build(_corpus())
        response = index.search(
            "quantum", config=RetrievalConfig(match_mode="keyword")
        )
        assert response.status in (RetrievalStatus.SUCCESS, RetrievalStatus.EMPTY)
        assert response.leg(LegKind.DENSE).status is LegStatus.SKIPPED
