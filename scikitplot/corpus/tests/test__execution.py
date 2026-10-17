# scikitplot/corpus/tests/test__execution.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Retrieval as typed, bounded stages (Run 4, R4-3): RetrievalExecutionPlan.
"""

from __future__ import annotations

import json
import math

import pytest

from .. import (
    CharBudgetCompressor,
    CorpusDocument,
    ExecutionResult,
    FusionPolicy,
    PlannedHit,
    RetrievalConfig,
    RetrievalExecutionPlan,
    RetrievalIndex,
    RetrievalStatus,
    StageBudget,
    StageKind,
    StageOutcome,
    StageStatus,
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
HYBRID = RetrievalConfig(top_k=5, match_mode="hybrid")
EMBEDDING = [1.0, 1.0, 0.0]
#: Budgets and sizes the tests set and then assert.
CALLS = 2
STEP = 5.0
SECONDS = 8
DEPTH = 2


def _docs():
    return [
        CorpusDocument.create(
            input_path=f"f{i}.txt",
            chunk_index=i,
            text=t,
            embedding=[float(i), 1.0, 0.0],
        )
        for i, t in enumerate(TEXTS)
    ]


@pytest.fixture(scope="module")
def index():
    built = RetrievalIndex()
    built.build(_docs())
    return built


class Clock:
    """A clock that moves only when told to."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TimedIndex:
    """Delegates to an index; each search advances a Clock by ``step``."""

    def __init__(self, index, clock, step=5.0, fail_on=()):
        self.index, self.clock, self.step, self.fail_on = index, clock, step, fail_on
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        self.clock.now += self.step
        if query in self.fail_on:
            raise RuntimeError(f"backend down for {query!r}")
        return self.index.search(query, **kwargs)


def _ids(hits):
    return [getattr(getattr(h, "doc", h), "doc_id", None) for h in hits]


def _stages(result):
    return {s.stage.value: s.status.value for s in result.stages}


# ---------------------------------------------------------------- validation


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"max_calls": 0}, ValueError),
            ({"max_calls": True}, TypeError),
            ({"max_calls": 1.5}, TypeError),
            ({"max_seconds": 0}, ValueError),
            ({"max_seconds": math.inf}, ValueError),
            ({"max_seconds": "1"}, TypeError),
        ],
    )
    def test_stage_budget(self, kwargs, error):
        with pytest.raises(error):
            StageBudget(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"transform": "rewrite"}, TypeError, "transform must be callable"),
            ({"keep_original": 1}, TypeError, "keep_original"),
            ({"max_queries": 0}, ValueError, "max_queries"),
            ({"fusion": {"k": 60}}, TypeError, "FusionPolicy"),
            ({"top_k": 0}, ValueError, "top_k"),
            ({"rerank": len, "rerank_depth": 3}, ValueError, "top_k must be set"),
            ({"rerank": len, "rerank_depth": 3, "top_k": 4}, ValueError, "at most"),
            ({"deadline_seconds": -1}, ValueError, "deadline_seconds"),
            ({"on_error": "ignore"}, ValueError, "on_error"),
            ({"budgets": {"search": StageBudget()}}, ValueError, "unknown stage"),
            ({"budgets": {"retrieve": 3}}, TypeError, "StageBudget"),
            ({"budgets": ["rerank"]}, TypeError, "mapping"),
            (
                {"budgets": (("retrieve", StageBudget()), ("retrieve", StageBudget()))},
                ValueError,
                "given twice",
            ),
        ],
    )
    def test_plan_fields(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            RetrievalExecutionPlan(**kwargs)

    def test_budgets_are_frozen_and_readable(self):
        plan = RetrievalExecutionPlan(budgets={"retrieve": StageBudget(max_calls=1)})
        assert plan.budgets == (("retrieve", StageBudget(max_calls=1)),)
        assert plan.budget(StageKind.RETRIEVE) == StageBudget(max_calls=1)
        assert plan.budget("rerank") is None
        hash(plan.budgets)

    @pytest.mark.parametrize("stage", ["transform", "fuse", "rerank", "compress"])
    def test_a_single_call_stage_refuses_a_budget(self, stage):
        # A budget is checked before calls; a stage of one call is never
        # stopped by it, so the plan refuses the promise (review 4c #1).
        with pytest.raises(ValueError, match="only 'retrieve' takes a StageBudget"):
            RetrievalExecutionPlan(budgets={stage: StageBudget(max_seconds=1.0)})

    def test_a_fusion_policy_keeping_nothing_is_refused(self):
        with pytest.raises(ValueError, match="top_k=0"):
            RetrievalExecutionPlan(fusion=FusionPolicy(top_k=0))

    @pytest.mark.parametrize(
        ("query", "error"), [(None, TypeError), ("", ValueError), ("  ", ValueError)]
    )
    def test_run_refuses_a_bad_query(self, index, query, error):
        with pytest.raises(error):
            RetrievalExecutionPlan().run(index, query)

    def test_run_refuses_an_index_without_search(self):
        with pytest.raises(TypeError, match="search method"):
            RetrievalExecutionPlan().run(object(), "q")

    def test_an_outcome_that_did_not_finish_explains_itself(self):
        for status in (StageStatus.DEGRADED, StageStatus.FAILED, StageStatus.CANCELLED):
            with pytest.raises(ValueError, match="no ErrorRecord"):
                StageOutcome(stage=StageKind.RERANK, status=status)


# ---------------------------------------------------------------- equivalence


class TestEmptyPlanIsTheSearch:
    """Gate: an empty plan returns exactly what index.search returns."""

    @pytest.mark.parametrize(
        ("config", "embedding"),
        [
            (RetrievalConfig(top_k=5, match_mode="strict"), None),
            (KEYWORD, None),
            (RetrievalConfig(top_k=5, match_mode="semantic"), EMBEDDING),
            (HYBRID, EMBEDDING),
            (HYBRID, None),  # dense leg cannot run: a degraded response
        ],
    )
    def test_same_hits_order_and_status(self, index, config, embedding):
        for query in ("quantum light", "motion", "nothing matches xyz"):
            base = index.search(query, config=config, query_embedding=embedding)
            result = RetrievalExecutionPlan(retrieval=config).run(
                index, query, query_embedding=embedding
            )
            assert _ids(result) == _ids(base)
            assert [h.score for h in result] == [h.score for h in base]
            assert result.status is base.status
            assert result.responses == (base,)
            assert _stages(result) == {
                "transform": "skipped",
                "retrieve": "success" if base.hits else "empty",
                "fuse": "skipped",
                "rerank": "skipped",
                "compress": "skipped",
            }

    def test_with_a_filter(self, index):
        expression = {"op": "eq", "field": "chunk_index", "value": 2}
        base = index.search("quantum", config=KEYWORD, filter=expression)
        result = RetrievalExecutionPlan(retrieval=KEYWORD).run(
            index, "quantum", filter=expression
        )
        assert _ids(result) == _ids(base)

    def test_a_search_that_raises_fails_the_plan(self, index):
        semantic = RetrievalConfig(match_mode="semantic")  # no embedding: raises
        result = RetrievalExecutionPlan(retrieval=semantic).run(index, "quantum")
        assert result.status is RetrievalStatus.FAILED
        retrieve = result.stage("retrieve")
        assert retrieve.status is StageStatus.FAILED
        assert retrieve.error.code == "QUERY_FAILED"
        assert retrieve.error.exception_type == "ValueError"
        with pytest.raises(ValueError, match="requires query_embedding"):
            RetrievalExecutionPlan(retrieval=semantic, on_error="raise").run(
                index, "quantum"
            )


# ---------------------------------------------------------------- transform


class TestTransform:
    def test_queries_original_first_duplicates_once(self, index):
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=lambda q: ["light", q, "light", "motion"]
        )
        result = plan.run(index, "quantum")
        assert result.queries == ("quantum", "light", "motion")
        transform = result.stage("transform")
        assert transform.status is StageStatus.SUCCESS
        # "quantum" repeats the original and "light" repeats itself: 4 - 2 run.
        assert dict(transform.details) == {"duplicates": 2, "proposed": 4}

    def test_without_the_original(self, index):
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=lambda q: ["light"], keep_original=False
        )
        assert plan.run(index, "quantum").queries == ("light",)

    def test_no_query_falls_back_to_the_original(self, index):
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=lambda q: [], keep_original=False
        )
        result = plan.run(index, "quantum")
        assert result.queries == ("quantum",)
        assert result.stage("transform").error.code == "NO_QUERY"
        assert result.status is RetrievalStatus.DEGRADED

    @pytest.mark.parametrize(
        "output", ["light", b"light", None, [1], ["ok", " "], {"a": 1}]
    )
    def test_a_bad_output_fails_the_stage_or_raises(self, index, output):
        plan = RetrievalExecutionPlan(retrieval=KEYWORD, transform=lambda q: output)
        result = plan.run(index, "quantum")
        assert result.queries == ("quantum",)
        assert result.stage("transform").status is StageStatus.FAILED
        assert result.status is RetrievalStatus.DEGRADED
        with pytest.raises((TypeError, ValueError)):
            RetrievalExecutionPlan(
                retrieval=KEYWORD, transform=lambda q: output, on_error="raise"
            ).run(index, "quantum")

    def test_queries_beyond_max_queries_are_dropped_and_reported(self, index):
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=lambda q: ["a", "b", "c"], max_queries=2
        )
        result = plan.run(index, "quantum")
        assert result.queries == ("quantum", "a")
        transform = result.stage("transform")
        assert transform.status is StageStatus.DEGRADED
        assert transform.error.code == "QUERIES_DROPPED"
        assert transform.error.details["dropped"] == ["b", "c"]


# ---------------------------------------------------------------- retrieve


class TestRetrieve:
    def test_embeddings_per_query(self, index):
        clock = Clock()
        timed = TimedIndex(index, clock)
        seen = []
        plan = RetrievalExecutionPlan(
            retrieval=HYBRID, transform=lambda q: ["light"], clock=clock
        )
        result = plan.run(
            timed,
            "quantum",
            query_embedding=EMBEDDING,
            embed_query=lambda text: seen.append(text) or [0.0, 1.0, 0.0],
        )
        assert seen == ["light"]  # the original used query_embedding
        assert timed.calls[0][1]["query_embedding"] == EMBEDDING
        assert result.status is RetrievalStatus.SUCCESS

    def test_a_transformed_query_without_an_embedding_is_degraded(self, index):
        plan = RetrievalExecutionPlan(retrieval=HYBRID, transform=lambda q: ["light"])
        result = plan.run(index, "quantum", query_embedding=EMBEDDING)
        assert [r.status.value for r in result.responses] == ["success", "degraded"]
        assert result.stage("retrieve").status is StageStatus.SUCCESS
        assert result.status is RetrievalStatus.DEGRADED
        assert any(r.code == "NO_QUERY_EMBEDDING" for r in result.error_records())

    def test_one_failing_query_degrades(self, index):
        clock = Clock()
        timed = TimedIndex(index, clock, fail_on=("light",))
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=lambda q: ["light", "motion"], clock=clock
        )
        result = plan.run(timed, "quantum")
        retrieve = result.stage("retrieve")
        assert retrieve.status is StageStatus.DEGRADED
        assert retrieve.error.code == "QUERY_FAILED"
        assert retrieve.error.details["query_index"] == 1
        assert result.responses[1] is None
        assert result.status is RetrievalStatus.DEGRADED
        with pytest.raises(RuntimeError, match="backend down"):
            RetrievalExecutionPlan(
                retrieval=KEYWORD, transform=lambda q: ["light"], on_error="raise"
            ).run(TimedIndex(index, Clock(), fail_on=("light",)), "quantum")

    def test_an_embedder_that_raises_fails_its_query(self, index):
        def embed(text):
            raise OSError("model offline")

        plan = RetrievalExecutionPlan(retrieval=HYBRID, transform=lambda q: ["light"])
        result = plan.run(
            index, "quantum", query_embedding=EMBEDDING, embed_query=embed
        )
        assert result.stage("retrieve").error.exception_type == "OSError"
        assert result.queries == ("quantum", "light")
        assert result.responses[1] is None

    def test_max_calls_stops_further_queries(self, index):
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD,
            transform=lambda q: ["light", "motion"],
            budgets={"retrieve": StageBudget(max_calls=CALLS)},
        )
        result = plan.run(index, "quantum")
        retrieve = result.stage("retrieve")
        assert retrieve.status is StageStatus.DEGRADED
        assert retrieve.calls == CALLS
        assert retrieve.error.code == "BUDGET_EXHAUSTED"
        assert retrieve.error.details == {"budget": "max_calls", "limit": CALLS}
        assert dict(retrieve.details)["not_started"] == 1
        assert result.responses[2] is None

    def test_max_seconds_stops_further_queries(self, index):
        clock = Clock()
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD,
            transform=lambda q: ["light", "motion"],
            budgets={"retrieve": StageBudget(max_seconds=SECONDS)},
            clock=clock,
        )
        result = plan.run(TimedIndex(index, clock, step=STEP), "quantum")
        retrieve = result.stage("retrieve")
        # Calls start at 0 s and 5 s; at 10 s (> 8 s) the budget stops the third.
        assert retrieve.calls == CALLS
        assert retrieve.error.details["budget"] == "max_seconds"
        assert retrieve.seconds == CALLS * STEP

    def test_a_passed_deadline_cancels_the_plan(self, index):
        clock = Clock()

        def slow(q):
            clock.now += 3
            return ["light"]

        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=slow, deadline_seconds=2, clock=clock
        )
        result = plan.run(TimedIndex(index, clock), "quantum")
        assert result.stage("retrieve").status is StageStatus.CANCELLED
        assert result.stage("retrieve").error.code == "DEADLINE_EXCEEDED"
        assert result.status is RetrievalStatus.CANCELLED
        assert not result

    def test_a_hit_without_identity_is_refused(self):
        class Anonymous:
            def search(self, query, **kwargs):
                return [object()]

        result = RetrievalExecutionPlan().run(Anonymous(), "q")
        assert result.stage("retrieve").status is StageStatus.FAILED
        assert "no doc_id or node_id" in result.stage("retrieve").error.message

    def test_an_index_cancelled_response_cancels_the_plan(self):
        # The index cancelled every query: the plan did not fail (review 4c #2).
        class Cancelled:
            status = RetrievalStatus.CANCELLED

            def __iter__(self):
                return iter(())

            def error_records(self):
                return [
                    ErrorRecord(
                        code="DEADLINE_EXCEEDED",
                        category=ErrorCategory.RESOURCE,
                        message="cancelled by the index",
                    )
                ]

        class Index:
            def search(self, query, **kwargs):
                return Cancelled()

        result = RetrievalExecutionPlan().run(Index(), "q")
        retrieve = result.stage("retrieve")
        assert retrieve.status is StageStatus.CANCELLED
        assert retrieve.error.code == "DEADLINE_EXCEEDED"
        assert retrieve.error.details["query_index"] == 0
        assert result.status is RetrievalStatus.CANCELLED
        assert result.to_dict()["responses"] == [
            {"status": "cancelled", "hit_count": None}
        ]

    @pytest.mark.parametrize(
        ("records", "code"),
        [
            (lambda: [{"code": "X"}], "QUERY_FAILED"),  # not an ErrorRecord
            (list, "QUERY_FAILED"),
            (None, "QUERY_FAILED"),  # no error_records at all
            (
                lambda: [
                    ErrorRecord(
                        code="INDEX_DOWN",
                        category=ErrorCategory.CAPABILITY,
                        message="m",
                        details=None,
                    )
                ],
                "INDEX_DOWN",
            ),
        ],
    )
    @pytest.mark.parametrize("status", [RetrievalStatus.FAILED, "failed"])
    def test_a_failed_response_with_any_records_degrades(self, records, code, status):
        # Re-review 4c: a malformed record must not crash a degrading plan.
        class Failed:
            def __iter__(self):
                return iter(())

        Failed.status = status
        if records is not None:
            Failed.error_records = lambda self: records()

        class Index:
            def search(self, query, **kwargs):
                return Failed()

        result = RetrievalExecutionPlan().run(Index(), "q")
        retrieve = result.stage("retrieve")
        assert retrieve.status is StageStatus.FAILED
        assert retrieve.error.code == code
        assert retrieve.error.details["query_index"] == 0

    def test_a_cancelled_status_given_as_its_value_cancels(self):
        class Cancelled:
            status = "cancelled"

            def __iter__(self):
                return iter(())

        class Index:
            def search(self, query, **kwargs):
                return Cancelled()

        result = RetrievalExecutionPlan().run(Index(), "q")
        assert result.stage("retrieve").status is StageStatus.CANCELLED
        assert result.status is RetrievalStatus.CANCELLED

    def test_an_unknown_response_status_is_not_success(self, index):
        class Odd(list):
            status = "partial"

        class Index:
            def search(self, query, **kwargs):
                return Odd(index.search(query, config=KEYWORD))

        result = RetrievalExecutionPlan().run(Index(), "light")
        assert result.stage("retrieve").status is StageStatus.SUCCESS
        assert result.status is RetrievalStatus.DEGRADED

    @pytest.mark.parametrize("score", [None, "0.5", True])
    def test_a_hit_without_a_real_score_fails_its_query(self, index, score):
        # Refused in retrieve, so fuse never sees it (review 4c #3).
        class Hit:
            def __init__(self, doc):
                self.doc, self.score = doc, score

        class Index:
            def search(self, query, **kwargs):
                if query == "quantum":
                    return index.search(query, config=KEYWORD)
                return [Hit(h.doc) for h in index.search("light", config=KEYWORD)]

        plan = RetrievalExecutionPlan(transform=lambda q: ["light"])
        result = plan.run(Index(), "quantum")
        assert result.stage("retrieve").status is StageStatus.DEGRADED
        assert "score must be a real number" in result.stage("retrieve").error.message
        assert _ids(result) == _ids(index.search("quantum", config=KEYWORD))

    def test_an_overrun_is_reported(self, index):
        clock = Clock()
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD,
            budgets={"retrieve": StageBudget(max_seconds=SECONDS)},
            clock=clock,
        )
        result = plan.run(TimedIndex(index, clock, step=SECONDS + 2), "light")
        details = dict(result.stage("retrieve").details)
        assert details["seconds_over_budget"] == pytest.approx(2.0)
        assert result.stage("retrieve").status is StageStatus.SUCCESS


# ---------------------------------------------------------------- fuse


class TestFuse:
    def test_fusion_is_the_one_policy(self, index):
        queries = ["quantum", "light", "motion"]
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=lambda q: queries[1:]
        )
        result = plan.run(index, "quantum")
        rankings = [
            (f"q{i}", _ids(index.search(q, config=KEYWORD)))
            for i, q in enumerate(queries)
        ]
        expected = FusionPolicy().fuse(rankings)
        assert [(h.key, h.score) for h in result] == [
            (i.key, i.score) for i in expected
        ]
        assert all(h.fused_score == h.score for h in result)
        assert result.stage("fuse").status is StageStatus.SUCCESS

    def test_query_ranks_are_each_querys_rank(self, index):
        plan = RetrievalExecutionPlan(retrieval=KEYWORD, transform=lambda q: ["light"])
        result = plan.run(index, "quantum")
        for i, q in enumerate(result.queries):
            ranking = _ids(index.search(q, config=KEYWORD))
            for hit in result:
                expected = ranking.index(hit.key) if hit.key in ranking else None
                assert hit.query_ranks[i] == expected

    def test_a_custom_policy(self, index):
        policy = FusionPolicy(k=1, weights={"q1": 0.0})
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=lambda q: ["light"], fusion=policy
        )
        result = plan.run(index, "quantum")
        assert _ids(result) == _ids(index.search("quantum", config=KEYWORD))
        assert dict(result.stage("fuse").details)["k"] == 1

    def test_a_policy_dropping_every_query_degrades(self, index):
        # Review 4c #5: hits in, none out, is not an empty corpus.
        policy = FusionPolicy(default_weight=0.0)
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, transform=lambda q: ["light"], fusion=policy
        )
        result = plan.run(index, "quantum")
        fuse = result.stage("fuse")
        assert fuse.status is StageStatus.DEGRADED
        assert fuse.error.code == "FUSION_DROPPED_ALL"
        assert result.status is RetrievalStatus.DEGRADED

    def test_fusion_runs_past_the_deadline(self, index):
        clock = Clock()
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD,
            transform=lambda q: ["light"],
            deadline_seconds=STEP + 1,
            clock=clock,
        )
        result = plan.run(TimedIndex(index, clock, step=STEP), "quantum")
        # Both queries ran (0 s and 5 s); fuse is in-process and always runs.
        assert result.stage("fuse").status is StageStatus.SUCCESS

    def test_fusion_is_skipped_with_one_query(self, index):
        result = RetrievalExecutionPlan(retrieval=KEYWORD).run(index, "quantum")
        assert result.stage("fuse").status is StageStatus.SKIPPED
        assert all(h.fused_score is None for h in result)


# ---------------------------------------------------------------- rerank


def _by_length(query, hits):
    return [float(len(h.doc.text)) for h in hits]


class TestRerank:
    def test_order_by_score_ties_keep_the_fused_order(self, index):
        base = RetrievalExecutionPlan(retrieval=KEYWORD).run(index, "light motion")
        result = RetrievalExecutionPlan(
            retrieval=KEYWORD, rerank=lambda q, hits: [1.0] * len(hits)
        ).run(index, "light motion")
        assert _ids(result) == _ids(base)  # all tied
        result = RetrievalExecutionPlan(retrieval=KEYWORD, rerank=_by_length).run(
            index, "light motion"
        )
        lengths = [len(h.doc.text) for h in result]
        assert lengths == sorted(lengths, reverse=True)
        assert all(h.rerank_score == h.score for h in result)

    def test_depth_limits_what_is_reranked(self, index):
        seen = []

        def rerank(query, hits):
            seen.append(len(hits))
            return [0.0] * len(hits)

        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, rerank=rerank, rerank_depth=DEPTH, top_k=DEPTH
        )
        result = plan.run(index, "light motion")
        assert seen == [DEPTH]
        assert len(result) == DEPTH
        assert dict(result.stage("rerank").details)["depth"] == DEPTH

    @pytest.mark.parametrize(
        "output",
        [
            lambda n: [1.0] * (n + 1),
            lambda n: [math.nan] * n,
            lambda n: [True] * n,
            lambda n: ["1"] * n,
            lambda n: "scores",
            lambda n: None,
            lambda n: set(range(n)),
            lambda n: dict.fromkeys(range(n), 1.0),
        ],
    )
    def test_a_bad_output_keeps_the_fused_order(self, index, output):
        base = RetrievalExecutionPlan(retrieval=KEYWORD).run(index, "light motion")
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, rerank=lambda q, hits: output(len(hits))
        )
        result = plan.run(index, "light motion")
        assert _ids(result) == _ids(base)
        assert result.stage("rerank").status is StageStatus.FAILED
        assert result.status is RetrievalStatus.DEGRADED
        with pytest.raises((TypeError, ValueError)):
            RetrievalExecutionPlan(
                retrieval=KEYWORD,
                rerank=lambda q, hits: output(len(hits)),
                on_error="raise",
            ).run(index, "light motion")

    def test_an_array_of_scores_is_accepted(self, index):
        np = pytest.importorskip("numpy")
        result = RetrievalExecutionPlan(
            retrieval=KEYWORD, rerank=lambda q, hits: np.asarray(_by_length(q, hits))
        ).run(index, "light motion")
        assert result.stage("rerank").status is StageStatus.SUCCESS

    def test_no_hits_skips_the_reranker(self, index):
        called = []
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, rerank=lambda q, h: called.append(1) or []
        )
        result = plan.run(index, "nothing matches xyz")
        assert called == []
        assert result.stage("rerank").status is StageStatus.SKIPPED
        assert result.status is RetrievalStatus.EMPTY

    def test_a_spent_budget_cancels_the_reranker(self, index):
        clock = Clock()
        plan = RetrievalExecutionPlan(
            retrieval=KEYWORD, rerank=_by_length, deadline_seconds=4, clock=clock
        )
        result = plan.run(TimedIndex(index, clock, step=5), "light")
        assert result.stage("rerank").status is StageStatus.CANCELLED
        assert all(h.rerank_score is None for h in result)
        assert result.status is RetrievalStatus.DEGRADED


# ---------------------------------------------------------------- compress


class TestCompress:
    def test_passages_are_spans_of_the_source(self, index):
        def first_word(query, hits):
            return [((0, h.doc.text.index(" ")),) for h in hits]

        result = RetrievalExecutionPlan(retrieval=KEYWORD, compress=first_word).run(
            index, "light"
        )
        for passage in result.passages:
            start, end = passage.spans[0]
            assert passage.text == passage.hit.doc.text[start:end]
        assert result.stage("compress").status is StageStatus.SUCCESS

    def test_an_empty_entry_drops_the_hit_from_the_context(self, index):
        result = RetrievalExecutionPlan(
            retrieval=KEYWORD, compress=lambda q, hits: [()] * len(hits)
        ).run(index, "light")
        assert result.passages == ()
        assert result.stage("compress").status is StageStatus.EMPTY
        assert len(result) > 0  # the ranking is unchanged

    def test_an_empty_entry_needs_no_text(self):
        # Review 4c #4: a hit whose text is unknown may still be dropped.
        class Hit:
            def __init__(self, i):
                self.doc_id, self.score = f"d{i}", 1.0

        class Index:
            def search(self, query, **kwargs):
                return [Hit(0), Hit(1)]

        result = RetrievalExecutionPlan(compress=lambda q, hits: [(), ()]).run(
            Index(), "q"
        )
        assert result.stage("compress").status is StageStatus.EMPTY

    def test_integer_like_offsets_are_accepted(self, index):
        np = pytest.importorskip("numpy")
        result = RetrievalExecutionPlan(
            retrieval=KEYWORD,
            compress=lambda q, hits: [((np.int64(0), np.int64(3)),)] * len(hits),
        ).run(index, "light")
        assert result.stage("compress").status is StageStatus.SUCCESS
        assert all(p.spans == ((0, 3),) for p in result.passages)
        assert all(type(v) is int for p in result.passages for v in p.spans[0])

    @pytest.mark.parametrize(
        "entry",
        [
            ((0, 999),),
            ((3, 2),),
            ((0, 4), (2, 6)),
            ((5, 6), (0, 2)),
            ((True, 3),),
            ((0, 1.5),),
            ((0, 1, 2),),
        ],
    )
    def test_a_bad_span_fails_the_stage(self, index, entry):
        result = RetrievalExecutionPlan(
            retrieval=KEYWORD, compress=lambda q, hits: [entry] * len(hits)
        ).run(index, "light")
        assert result.passages is None
        assert result.stage("compress").status is StageStatus.FAILED
        assert result.status is RetrievalStatus.DEGRADED

    def test_the_wrong_number_of_entries_fails(self, index):
        result = RetrievalExecutionPlan(
            retrieval=KEYWORD, compress=lambda q, hits: []
        ).run(index, "light")
        assert "entries for" in result.stage("compress").error.message


class TestCharBudgetCompressor:
    class _Doc:
        def __init__(self, text):
            self.text = text

    def _hits(self, *texts):
        return [type("Hit", (), {"doc": self._Doc(t)})() for t in texts]

    def test_whole_words_within_the_budget(self):
        spans = CharBudgetCompressor(12)("q", self._hits("alpha beta", "gamma delta"))
        assert spans == [((0, 10),), ()]  # "gamma" alone does not fit in 2

    def test_a_hard_cut(self):
        spans = CharBudgetCompressor(7, whole_words=False)(
            "q", self._hits("alpha beta")
        )
        assert spans == [((0, 7),)]

    def test_trailing_whitespace_is_not_kept(self):
        spans = CharBudgetCompressor(6)("q", self._hits("alpha beta"))
        assert spans == [((0, 5),)]

    @pytest.mark.parametrize("space", ["\t", "\u3000", "\u00a0", "\r\n"])
    def test_any_whitespace_is_a_word_boundary(self, space):
        # Review 4c #7: not only " " and "\n".
        text = f"alpha{space}beta"
        assert CharBudgetCompressor(7)("q", self._hits(text)) == [((0, 5),)]

    def test_text_without_whitespace_needs_hard_cuts(self):
        text = "\u91cf\u5b50\u7269\u7406\u5b66"  # five CJK characters
        assert CharBudgetCompressor(3)("q", self._hits(text)) == [()]
        hard = CharBudgetCompressor(3, whole_words=False)("q", self._hits(text))
        assert hard == [((0, 3),)]

    def test_a_first_word_longer_than_the_budget_keeps_nothing(self):
        assert CharBudgetCompressor(3)("q", self._hits("alphabet")) == [()]

    def test_the_total_never_exceeds_the_budget(self):
        texts = ["one two three", "four five", "six seven eight nine", "ten"]
        for budget in range(1, 60):
            spans = CharBudgetCompressor(budget)("q", self._hits(*texts))
            assert sum(e - s for entry in spans for s, e in entry) <= budget

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"max_chars": 0}, ValueError),
            ({"max_chars": True}, TypeError),
            ({"max_chars": 5, "whole_words": 1}, TypeError),
        ],
    )
    def test_validation(self, kwargs, error):
        with pytest.raises(error):
            CharBudgetCompressor(**kwargs)


# ---------------------------------------------------------------- result


class TestResult:
    def test_behaves_like_a_response(self, index):
        result = RetrievalExecutionPlan(retrieval=KEYWORD, top_k=DEPTH).run(
            index, "light"
        )
        assert isinstance(result, ExecutionResult)
        assert len(result) == DEPTH
        assert result[0] is result.hits[0]
        assert all(isinstance(h, PlannedHit) for h in result)
        assert result[0].doc is result[0].hit.doc
        assert result[0].doc_id == result[0].hit.doc.doc_id

    def test_to_dict_is_json(self, index):
        plan = RetrievalExecutionPlan(
            retrieval=HYBRID,
            transform=lambda q: ["light"],
            rerank=_by_length,
            compress=CharBudgetCompressor(40),
        )
        result = plan.run(index, "quantum", query_embedding=EMBEDDING)
        data = json.loads(json.dumps(result.to_dict()))
        assert [s["stage"] for s in data["stages"]] == [k.value for k in StageKind]
        assert data["status"] == result.status.value

    def test_a_list_response_records_its_size(self):
        class Hit:
            doc_id, score = "d0", 1.0

        class Index:
            def search(self, query, **kwargs):
                return [Hit()]

        data = RetrievalExecutionPlan().run(Index(), "q").to_dict()
        assert data["responses"] == [{"status": None, "hit_count": 1}]

    def test_status_derivation(self):
        record = ErrorRecord(code="X", category=ErrorCategory.INTERNAL, message="x")

        def result(retrieve, other=StageStatus.SKIPPED, hits=()):
            stages = (
                StageOutcome(
                    stage=StageKind.RETRIEVE,
                    status=retrieve,
                    error=(
                        record
                        if retrieve not in (StageStatus.SUCCESS, StageStatus.EMPTY)
                        else None
                    ),
                ),
                StageOutcome(
                    stage=StageKind.RERANK,
                    status=other,
                    error=(
                        record
                        if other in (
                            StageStatus.FAILED,
                            StageStatus.DEGRADED,
                            StageStatus.CANCELLED,
                        )
                        else None
                    ),
                ),
            )
            return ExecutionResult(
                hits=hits, stages=stages, queries=("q",), responses=(None,)
            ).status

        hit = PlannedHit(hit=None, score=1.0, key="d")
        assert result(StageStatus.CANCELLED) is RetrievalStatus.CANCELLED
        assert result(StageStatus.FAILED) is RetrievalStatus.FAILED
        assert (
            result(StageStatus.SUCCESS, StageStatus.FAILED, (hit,))
            is RetrievalStatus.DEGRADED
        )
        assert result(StageStatus.DEGRADED, hits=(hit,)) is RetrievalStatus.DEGRADED
        assert result(StageStatus.EMPTY) is RetrievalStatus.EMPTY
        assert result(StageStatus.SUCCESS, hits=(hit,)) is RetrievalStatus.SUCCESS


class TestHarness:
    def test_a_plan_is_measured_like_an_index(self, index):
        queries = {"q1": "quantum", "q2": "motion"}
        qrels = {
            "q1": [_ids(index.search("quantum", config=KEYWORD))[0]],
            "q2": ["none"],
        }
        plan = RetrievalExecutionPlan(retrieval=KEYWORD)
        bare = evaluate_retrieval(
            lambda q: index.search(q, config=KEYWORD),
            queries,
            qrels,
            metrics=["recall@1", "mrr"],
            zero_division=0.0,
        )
        planned = evaluate_retrieval(
            plan.bind(index),
            queries,
            qrels,
            metrics=["recall@1", "mrr"],
            zero_division=0.0,
        )
        assert planned.metrics == bare.metrics

    def test_a_degraded_plan_is_refused_by_default(self, index):
        plan = RetrievalExecutionPlan(retrieval=KEYWORD, rerank=lambda q, h: None)
        with pytest.raises(ValueError, match="degraded"):
            evaluate_retrieval(
                plan.bind(index),
                {"q": "light"},
                {"q": ["x"]},
                metrics=["mrr"],
                zero_division=0.0,
            )
