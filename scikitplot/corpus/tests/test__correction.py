# scikitplot/corpus/tests/test__correction.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Bounded corrective retrieval (Run 5, R5-2): CorrectiveRetrieval.
"""

from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace as Ns

import pytest

from .. import (
    CorpusDocument,
    CorrectionResult,
    CorrectionVerdict,
    CorrectiveRetrieval,
    EvidenceGrade,
    FusionPolicy,
    GradeResult,
    RetrievalConfig,
    RetrievalExecutionPlan,
    RetrievalIndex,
    RetrievalStatus,
    TermCoverageGrader,
    evaluate_retrieval,
)
from .._diagnostics import ErrorCategory, ErrorRecord

TEXTS = (
    "quantum physics of light",
    "classical motion of bodies",
    "quantum computing with qubits",
    "light and optics",
    "bodies in motion and light",
)
KEYWORD = RetrievalConfig(top_k=5, match_mode="keyword")
#: Rounds the tests allow, and the deadline they set.
ROUNDS = 3
SECONDS = 4.0


@pytest.fixture(scope="module")
def index():
    built = RetrievalIndex(KEYWORD)
    built.build(
        [
            CorpusDocument.create(input_path=f"f{i}.txt", chunk_index=i, text=t)
            for i, t in enumerate(TEXTS)
        ]
    )
    return built


class Docs:
    """An index over ``{id: text}`` that returns documents holding every word."""

    def __init__(self, docs, fail_on=()):
        self.docs, self.fail_on, self.queries = docs, fail_on, []

    def search(self, query, **kwargs):
        self.queries.append((query, kwargs))
        if query in self.fail_on:
            raise RuntimeError(f"backend down for {query!r}")
        words = set(query.split())
        return [
            Ns(doc=Ns(doc_id=k, text=t), score=1.0)
            for k, t in self.docs.items()
            if words <= set(t.split())
        ]


def drop_last(query, rounds):
    """The previous round's query without its last word."""
    words = rounds[-1].query.split()[:-1]
    return " ".join(words) or None


def _ids(result):
    return [h.doc.doc_id for h in result]


# ------------------------------------------------------------------ grader


class TestTermCoverageGrader:
    def test_grades_by_coverage(self):
        hit = Ns(doc=Ns(text="Spans are slices of the source"))
        grader = TermCoverageGrader(top_k=1, accept=0.8, reject=0.3)
        assert grader("spans slices", [hit]).grade is EvidenceGrade.CORRECT
        mid = grader("spans slices cache", [hit])
        assert mid.grade is EvidenceGrade.AMBIGUOUS
        assert mid.score == pytest.approx(2 / 3)
        assert grader("cache keys digest", [hit]).grade is EvidenceGrade.INCORRECT

    def test_only_the_top_hits_are_read(self):
        hits = [Ns(doc=Ns(text="alpha")), Ns(doc=Ns(text="beta"))]
        one = TermCoverageGrader(top_k=1)("alpha beta", hits)
        assert one.score == 1 / 2
        assert TermCoverageGrader(top_k=2)("alpha beta", hits).score == 1.0

    @pytest.mark.parametrize(
        ("query", "hits"), [("spans", []), ("!!", [Ns(doc=Ns(text="x"))])]
    )
    def test_nothing_to_grade_is_incorrect(self, query, hits):
        result = TermCoverageGrader()(query, hits)
        assert result.grade is EvidenceGrade.INCORRECT
        assert result.score == 0.0

    def test_a_hit_without_text_contributes_nothing(self):
        assert TermCoverageGrader()("a", [Ns(doc=Ns(text=None))]).score == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"top_k": 0},
            {"top_k": True},
            {"accept": 0.3, "reject": 0.3},
            {"accept": 1.5},
            {"reject": -0.1},
            {"accept": "1"},
        ],
    )
    def test_validation(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            TermCoverageGrader(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"grade": "maybe", "score": 0.5},
            {"grade": "correct", "score": 1.5},
            {"grade": "correct", "score": math.nan},
            {"grade": "correct", "score": True},
            {"grade": "correct", "score": 0.5, "reason": 3},
        ],
    )
    def test_grade_result_validation(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            GradeResult(**kwargs)


# ------------------------------------------------------------------ verdicts


class TestVerdicts:
    def test_correct_evidence_is_accepted_at_once(self, index):
        result = CorrectiveRetrieval().run(index, "quantum light", config=KEYWORD)
        assert result.verdict is CorrectionVerdict.ACCEPTED
        assert len(result.rounds) == 1
        assert _ids(result) == _ids(index.search("quantum light", config=KEYWORD))
        assert result.status is RetrievalStatus.SUCCESS
        assert result.stop_reason == "accepted"

    def test_a_rewrite_corrects(self):
        docs = Docs({"b": "the cache key is a digest"})
        corrective = CorrectiveRetrieval(
            grader=TermCoverageGrader(accept=0.6), rewrite=drop_last, max_rounds=ROUNDS
        )
        result = corrective.run(docs, "cache key size")
        assert result.verdict is CorrectionVerdict.CORRECTED
        assert [r.query for r in result.rounds] == ["cache key size", "cache key"]
        assert [r.grade.grade for r in result.rounds] == [
            EvidenceGrade.INCORRECT,
            EvidenceGrade.CORRECT,
        ]
        assert _ids(result) == ["b"]

    def test_grades_are_against_the_original_query(self):
        # A rewrite that drifted to "cache" alone finds the document, but the
        # document holds one word of three: not accepted for the query.
        docs = Docs({"b": "cache"})
        corrective = CorrectiveRetrieval(
            rewrite=lambda q, rounds: "cache", max_rounds=ROUNDS
        )
        result = corrective.run(docs, "cache key size")
        assert result.rounds[1].grade.grade is EvidenceGrade.AMBIGUOUS
        assert result.verdict is CorrectionVerdict.PARTIAL

    def test_no_evidence_abstains(self):
        result = CorrectiveRetrieval(max_rounds=ROUNDS).run(Docs({}), "anything")
        assert result.verdict is CorrectionVerdict.ABSTAINED
        assert list(result) == [] and len(result) == 0  # noqa: PT018
        assert result.status is RetrievalStatus.EMPTY
        assert result.stop_reason == "no_rewrite"

    def test_ambiguous_rounds_are_fused(self):
        docs = Docs({"a": "alpha beta", "b": "alpha gamma", "c": "alpha delta"})
        queries = iter(["alpha", "alpha gamma"])
        corrective = CorrectiveRetrieval(
            grader=TermCoverageGrader(top_k=1, accept=1.0, reject=0.0),
            rewrite=lambda q, rounds: next(queries, None),
            max_rounds=ROUNDS,
        )
        result = corrective.run(docs, "alpha beta gamma")
        assert result.verdict is CorrectionVerdict.PARTIAL
        # Round 0 finds nothing (no document holds all three words); rounds
        # 1 and 2 are ambiguous (the top hit holds 2 of 3): they are fused.
        assert [r.grade.grade for r in result.rounds] == [
            EvidenceGrade.INCORRECT,
            EvidenceGrade.AMBIGUOUS,
            EvidenceGrade.AMBIGUOUS,
        ]
        legs = [("r1", ["a", "b", "c"]), ("r2", ["b"])]
        expected = [item.key for item in FusionPolicy().fuse(legs)]
        assert _ids(result) == expected

    def test_max_rounds_is_the_call_budget(self):
        docs = Docs({})
        corrective = CorrectiveRetrieval(
            rewrite=lambda q, rounds: f"{q} {len(rounds)}", max_rounds=ROUNDS
        )
        result = corrective.run(docs, "x")
        assert len(docs.queries) == ROUNDS == len(result.rounds)
        assert result.stop_reason == "max_rounds"

    def test_a_repeated_proposal_stops(self):
        docs = Docs({})
        result = CorrectiveRetrieval(
            rewrite=lambda q, rounds: q, max_rounds=ROUNDS
        ).run(docs, "x")
        assert len(docs.queries) == 1
        assert result.stop_reason == "no_rewrite"

    def test_the_deadline_stops_further_rounds(self):
        now = [0.0]

        class Slow(Docs):
            def search(self, query, **kwargs):
                now[0] += SECONDS
                return super().search(query, **kwargs)

        docs = Slow({})
        result = CorrectiveRetrieval(
            rewrite=lambda q, rounds: f"{q} {len(rounds)}",
            max_rounds=10,
            deadline_seconds=SECONDS,
            clock=lambda: now[0],
        ).run(docs, "x")
        assert len(docs.queries) == 1
        assert result.stop_reason == "deadline"


# ------------------------------------------------------------------ errors


class TestErrors:
    def test_a_failing_round_is_incorrect_and_degrades(self):
        docs = Docs({"a": "alpha"}, fail_on={"alpha beta"})
        result = CorrectiveRetrieval(
            rewrite=lambda q, rounds: "alpha", max_rounds=ROUNDS
        ).run(docs, "alpha beta")
        first = result.rounds[0]
        assert first.status is RetrievalStatus.FAILED
        assert first.error.code == "ROUND_FAILED"
        assert result.status is RetrievalStatus.DEGRADED
        assert result.error_records()[0].code == "ROUND_FAILED"

    @pytest.mark.parametrize(
        ("rewrite", "code"),
        [
            (lambda q, r: 1 / 0, "REWRITE_FAILED"),
            (lambda q, r: 3, "REWRITE_FAILED"),
            (lambda q, r: " ", "REWRITE_FAILED"),
        ],
    )
    def test_a_failing_rewrite_stops_and_degrades(self, rewrite, code):
        result = CorrectiveRetrieval(rewrite=rewrite, max_rounds=ROUNDS).run(
            Docs({}), "x"
        )
        assert result.stop_reason == "rewrite_failed"
        assert result.errors[0].code == code
        assert result.status is RetrievalStatus.DEGRADED

    @pytest.mark.parametrize("grader", [lambda q, h: 1 / 0, lambda q, h: "correct"])
    def test_a_failing_grader_is_incorrect(self, grader):
        result = CorrectiveRetrieval(grader=grader).run(Docs({"a": "x"}), "x")
        assert result.rounds[0].grade.grade is EvidenceGrade.INCORRECT
        assert result.errors[0].code == "GRADE_FAILED"
        assert result.verdict is CorrectionVerdict.ABSTAINED

    def test_raise_propagates(self):
        docs = Docs({}, fail_on={"x"})
        with pytest.raises(RuntimeError, match="backend down"):
            CorrectiveRetrieval(on_error="raise").run(docs, "x")
        with pytest.raises(ZeroDivisionError):
            CorrectiveRetrieval(rewrite=lambda q, r: 1 / 0, on_error="raise").run(
                Docs({}), "x"
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"plan": object()},
            {"grader": 3},
            {"rewrite": "halves"},
            {"max_rounds": 0},
            {"max_rounds": True},
            {"fusion": {"k": 60}},
            {"deadline_seconds": 0},
            {"deadline_seconds": math.inf},
            {"on_error": "ignore"},
        ],
    )
    def test_validation(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            CorrectiveRetrieval(**kwargs)

    @pytest.mark.parametrize("query", ["", "  ", None])
    def test_a_query_is_required(self, query):
        with pytest.raises(ValueError, match="non-empty"):
            CorrectiveRetrieval().run(Docs({}), query)


# ------------------------------------------------------------------ composition


class TestComposition:
    def test_a_plan_runs_each_round(self, index):
        plan = RetrievalExecutionPlan(retrieval=KEYWORD, top_k=2)
        result = CorrectiveRetrieval(plan=plan).run(index, "quantum light")
        assert len(result) == 2  # noqa: PLR2004
        assert result.rounds[0].status is RetrievalStatus.SUCCESS

    def test_search_kwargs_reach_every_round(self):
        docs = Docs({})
        CorrectiveRetrieval(
            rewrite=lambda q, rounds: f"{q} {len(rounds)}", max_rounds=2
        ).run(docs, "x", top_k=4)
        assert [kw for _, kw in docs.queries] == [{"top_k": 4}, {"top_k": 4}]

    def test_measured_like_an_index(self, index):
        corrective = CorrectiveRetrieval(max_rounds=1)
        report = evaluate_retrieval(
            corrective.bind(index, config=KEYWORD),
            queries={"q": "quantum light"},
            qrels={"q": [index.search("quantum light", config=KEYWORD)[0].doc.doc_id]},
            metrics=["recall@1"],
        )
        assert report.metrics == {"recall@1": 1.0}

    def test_to_dict_is_json(self, index):
        result = CorrectiveRetrieval().run(index, "quantum light", config=KEYWORD)
        data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert data["verdict"] == "accepted"
        assert data["rounds"][0]["grade"]["grader"] == "TermCoverageGrader"
        assert isinstance(result, CorrectionResult)


# ------------------------------------------------------------------ review 5b


class _Response(list):
    """A list of hits with a status and records, like a search response."""

    def __init__(self, hits, status, records=()):
        super().__init__(hits)
        self.status, self.records = status, list(records)

    def error_records(self):
        return list(self.records)


class TestReview:
    def test_hits_without_identity_do_not_raise_out_of_run(self):
        # 1: fusion needs an identity; degraded, the first round is kept.
        class NoIds:
            def search(self, query):
                return [Ns(doc=Ns(text="alpha"), score=1.0)]

        result = CorrectiveRetrieval(
            rewrite=lambda q, rounds: "alpha", max_rounds=2
        ).run(NoIds(), "alpha beta")
        assert result.verdict is CorrectionVerdict.PARTIAL
        assert result.errors[-1].code == "FUSE_FAILED"
        assert result.status is RetrievalStatus.DEGRADED
        with pytest.raises(ValueError, match="no doc_id"):
            CorrectiveRetrieval(
                rewrite=lambda q, rounds: "alpha", max_rounds=2, on_error="raise"
            ).run(NoIds(), "alpha beta")
        # One round needs no identity: returned as it is, without an error.
        single = CorrectiveRetrieval().run(NoIds(), "alpha beta")
        assert single.verdict is CorrectionVerdict.PARTIAL
        assert single.errors == ()
        assert single.status is RetrievalStatus.SUCCESS

    def test_a_cancelled_plan_is_not_an_empty_result(self, index):
        # 2: the harness's guard for incomplete responses still sees it.
        now = [0.0]

        def clock():
            now[0] += 1.0
            return now[0]

        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD,
            transform=lambda q: [q + " x"],
            deadline_seconds=0.5,
            clock=clock,
        )
        corrective = CorrectiveRetrieval(plan=plan, max_rounds=1)
        assert corrective.run(index, "quantum").status is RetrievalStatus.CANCELLED
        with pytest.raises(ValueError, match="cancelled"):
            evaluate_retrieval(
                corrective.bind(index),
                queries={"q": "quantum"},
                qrels={"q": ["x"]},
                metrics=["recall@1"],
            )

    @pytest.mark.parametrize(
        ("response", "code"),
        [
            (_Response([], RetrievalStatus.DEGRADED), "ROUND_INCOMPLETE"),
            (_Response([], "weird"), "STATUS_UNKNOWN"),
        ],
    )
    def test_an_incomplete_response_records_why(self, response, code):
        # 3: a degraded result always carries a record.
        class Index:
            def search(self, query):
                return response

        result = CorrectiveRetrieval().run(Index(), "x")
        assert result.status is RetrievalStatus.DEGRADED
        assert result.error_records()[0].code == code

    def test_a_responses_own_records_are_kept(self):
        record = ErrorRecord(
            code="LEG_DOWN", category=ErrorCategory.CAPABILITY, message="m"
        )

        class Index:
            def search(self, query):
                return _Response([], RetrievalStatus.DEGRADED, [record])

        assert CorrectiveRetrieval().run(Index(), "x").errors == (record,)

    def test_no_rewrite_is_asked_for_past_the_deadline(self):
        # 4: a model rewrite is a call; and the stop is recorded.
        now, asked = [0.0], []

        class Slow(Docs):
            def search(self, query, **kwargs):
                now[0] += SECONDS + 1
                return super().search(query, **kwargs)

        def rewrite(query, rounds):
            asked.append(query)
            return f"{query} {len(rounds)}"

        result = CorrectiveRetrieval(
            rewrite=rewrite,
            max_rounds=ROUNDS,
            deadline_seconds=SECONDS,
            clock=lambda: now[0],
        ).run(Slow({}), "x")
        assert asked == []
        assert result.stop_reason == "deadline"
        assert result.errors[-1].code == "DEADLINE_EXCEEDED"
        assert result.status is RetrievalStatus.DEGRADED

    def test_a_grade_subclass_is_checked(self):
        # 5: a subclass that skips the checks is rebuilt and refused.
        class Loose(GradeResult):
            def __post_init__(self):
                pass

        result = CorrectiveRetrieval(
            grader=lambda q, h: Loose("correct", math.nan)
        ).run(Docs({"a": "x"}), "x")
        assert result.errors[0].code == "GRADE_FAILED"
        json.dumps(result.to_dict())

    def test_one_ambiguous_round_is_fused_too(self):
        # 6: top_k and zero weights apply whatever the number of rounds.
        docs = Docs({"a": "alpha", "b": "alpha", "c": "alpha"})

        def grader(query, hits):
            return GradeResult("ambiguous", 0.5)

        one = CorrectiveRetrieval(grader=grader, fusion=FusionPolicy(top_k=1))
        assert len(one.run(docs, "alpha")) == 1
        none = CorrectiveRetrieval(
            grader=grader, fusion=FusionPolicy(weights={"r0": 0.0})
        )
        assert none.run(docs, "alpha").verdict is CorrectionVerdict.ABSTAINED

    def test_a_proposal_is_compared_stripped(self):
        # 7
        docs = Docs({})
        result = CorrectiveRetrieval(
            rewrite=lambda q, rounds: f" {q} ", max_rounds=ROUNDS
        ).run(docs, "x")
        assert len(docs.queries) == 1
        assert result.stop_reason == "no_rewrite"

    def test_a_graders_own_identity_is_kept(self):
        # 8
        result = CorrectiveRetrieval(
            grader=lambda q, h: GradeResult("correct", 1.0, grader="judge-v2")
        ).run(Docs({"a": "x"}), "x")
        assert result.rounds[0].grade.grader == "judge-v2"

    def test_numpy_scores_are_real_numbers(self):
        # 9
        np = pytest.importorskip("numpy")
        assert GradeResult("correct", np.float32(0.25)).score == 1 / 4
        TermCoverageGrader(accept=np.float64(0.9))


class TestReReview:
    def test_no_round_starts_past_the_deadline_after_a_slow_rewrite(self):
        now, started = [0.0], []

        class Timed(Docs):
            def search(self, query, **kwargs):
                started.append(now[0])
                return super().search(query, **kwargs)

        def slow(query, rounds):
            now[0] += 10.0
            return f"{query} {len(rounds)}"

        result = CorrectiveRetrieval(
            rewrite=slow, max_rounds=ROUNDS, deadline_seconds=1.0, clock=lambda: now[0]
        ).run(Timed({}), "x")
        assert started == [0.0]
        assert result.stop_reason == "deadline"

    def test_without_a_rewrite_a_deadline_stops_nothing(self):
        now = [0.0]

        class Slow(Docs):
            def search(self, query, **kwargs):
                now[0] += 10.0
                return super().search(query, **kwargs)

        result = CorrectiveRetrieval(
            grader=lambda q, h: GradeResult("ambiguous", 0.5),
            deadline_seconds=1.0,
            clock=lambda: now[0],
        ).run(Slow({"a": "x"}), "x")
        assert result.stop_reason == "no_rewrite"
        assert result.status is RetrievalStatus.SUCCESS

    def test_a_cancelled_round_is_not_evidence(self):
        class Index:
            def search(self, query):
                return _Response(
                    [Ns(doc=Ns(doc_id="a", text="x"), score=1.0)],
                    RetrievalStatus.CANCELLED,
                )

        result = CorrectiveRetrieval().run(Index(), "x")
        assert result.verdict is CorrectionVerdict.ABSTAINED
        assert result.status is RetrievalStatus.CANCELLED
        assert result.rounds[0].grade.reason == "the search cancelled"

    def test_an_error_inside_the_fusion_policy_propagates(self):
        class Broken(FusionPolicy):
            def fuse(self, legs):
                raise KeyError("bug")

        def grader(query, hits):
            return GradeResult("ambiguous", 0.5)

        docs = Docs({"a": "x"})
        with pytest.raises(KeyError, match="bug"):
            CorrectiveRetrieval(
                grader=grader, fusion=Broken(), rewrite=lambda q, r: "x", max_rounds=2
            ).run(docs, "x y")

    def test_thresholds_are_stored_as_floats(self):
        np = pytest.importorskip("numpy")
        grader = TermCoverageGrader(accept=np.float32(0.75))
        assert type(grader.accept) is float
        json.dumps(dataclasses.asdict(grader))
