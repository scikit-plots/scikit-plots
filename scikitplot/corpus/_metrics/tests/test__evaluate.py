# scikitplot/corpus/_metrics/tests/test__evaluate.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation harness, exportable reports and paired comparison."""

from __future__ import annotations

import itertools
import json
import math
import warnings

import numpy as np
import pytest

from ..._diagnostics import ErrorCategory, ErrorRecord
from ..._retrieval import LegKind, LegOutcome, LegStatus, RetrievalResponse
from ..._schema import CorpusDocument
from ..._similarity import RetrievalConfig, RetrievalIndex
from .._evaluate import (
    EXACT_SIGN_FLIP_LIMIT,
    SCHEMA,
    EvaluationComparison,
    QueryEvaluation,
    RetrievalEvaluation,
    _sign_flip_p_value,
    evaluate_retrieval,
)
from .._inputs import UndefinedMetricWarning
from .._ranking import ranking_metrics

QUERIES = {"q1": "cat food", "q2": "dog toys", "q3": "bird seed"}
QRELS = {"q1": ["d2"], "q2": {"d4": 2, "d5": 1}, "q3": ["d9"]}
RUNS = {"q1": ["d1", "d2"], "q2": ["d5", "d4", "d6"], "q3": ["d7"]}
METRICS = ["recall@2", "mrr", "ndcg@3"]


def lookup(query):
    key = {text: key for key, text in QUERIES.items()}[query]
    return [(doc, 1.0) for doc in RUNS[key]]


def ticking_clock(step=0.25):
    counter = itertools.count()
    return lambda: next(counter) * step


@pytest.fixture
def report():
    return evaluate_retrieval(lookup, QUERIES, QRELS, metrics=METRICS, clock=ticking_clock(), metadata={"system": "lookup"})


class TestHarness:
    def test_metrics_equal_the_metric_functions(self, report):
        assert report.metrics == ranking_metrics(QRELS, RUNS, metrics=METRICS)

    def test_per_query_records(self, report):
        first = report.queries[0]
        assert first == QueryEvaluation(
            query_id="q1",
            metrics={"recall@2": 1.0, "mrr": 0.5, "ndcg@3": first.metrics["ndcg@3"]},
            n_retrieved=2,
            n_relevant=1,
            latency_s=0.25,
            status=None,
            retrieved=("d1", "d2"),
        )
        np.testing.assert_array_equal(report.per_query("recall@2"), [1.0, 1.0, 0.0])
        with pytest.raises(KeyError):
            report.per_query("map")

    def test_latency_and_statuses(self, report):
        assert report.latency == {"mean": 0.25, "p50": 0.25, "p95": 0.25, "max": 0.25, "total": 0.75}
        assert report.status_counts == {"unreported": 3}

    def test_settings_and_metadata(self, report):
        assert report.settings["metrics"] == METRICS
        assert report.settings["zero_division"] == "warn"
        assert report.settings["retriever"] == "function"
        assert report.metadata == {"system": "lookup"}

    def test_real_index_statuses(self):
        docs = [
            CorpusDocument(doc_id=key, input_path=f"memory://{key}", chunk_index=0, text=text)
            for key, text in {"d1": "cat food bowl", "d2": "dog toys ball"}.items()
        ]
        index = RetrievalIndex(RetrievalConfig(match_mode="keyword", top_k=2))
        index.build(docs)
        out = evaluate_retrieval(index, {"a": "cat food", "b": "dog toys"}, {"a": ["d1"], "b": ["d2"]}, metrics=["mrr"])
        assert out.metrics == {"mrr": 1.0}
        assert out.status_counts == {"success": 2}
        assert out.settings["retriever"] == "RetrievalIndex"

    def test_rankings_can_be_dropped(self):
        out = evaluate_retrieval(lookup, QUERIES, QRELS, metrics=["mrr"], keep_rankings=False)
        assert all(q.retrieved is None for q in out.queries)
        assert "retrieved" not in out.to_rows()[0]

    def test_id_getter_and_duplicates(self):
        def chunks(query):
            return [("d2#0", 1.0), ("d2#1", 0.9), ("d1#0", 0.5)]

        options = {"metrics": ["recall@2"], "id_getter": lambda hit: hit[0].split("#")[0]}
        with pytest.raises(ValueError, match="duplicates='first'"):
            evaluate_retrieval(chunks, {"q": "x"}, {"q": ["d1"]}, **options)
        out = evaluate_retrieval(chunks, {"q": "x"}, {"q": ["d1"]}, duplicates="first", **options)
        assert out.metrics == {"recall@2": 1.0}
        assert out.queries[0].retrieved == ("d2", "d1")
        assert out.settings["id_getter"] == "<lambda>"

    def test_undefined_values_under_nan(self):
        out = evaluate_retrieval(lookup, QUERIES, {**QRELS, "q3": []}, metrics=["recall@2"], zero_division=np.nan)
        assert out.metrics == {"recall@2": 1.0}
        assert math.isnan(out.queries[2].metrics["recall@2"])
        assert out.settings["zero_division"] == "nan"

    def test_undefined_values_warn_once(self):
        with pytest.warns(UndefinedMetricWarning) as record:
            evaluate_retrieval(lookup, QUERIES, {**QRELS, "q3": []}, metrics=["recall@2"])
        assert len(record) == 1
        assert record[0].filename == __file__


class TestHarnessRefusals:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"queries": {"q1": "a"}}, ValueError, "same keys"),
            ({"qrels": {}, "queries": {}}, ValueError, "at least one query"),
            ({"metrics": "mrr"}, TypeError, "collection of names"),
            ({"metrics": []}, ValueError, "at least one metric"),
            ({"metrics": ["mrr", "mrr"]}, ValueError, "twice"),
            ({"retriever": 3}, TypeError, "search\\(query\\)"),
            ({"id_getter": 3}, TypeError, "id_getter"),
            ({"keep_rankings": 1}, TypeError, "keep_rankings"),
            ({"clock": 3}, TypeError, "clock"),
            ({"queries": [("q1", "a")]}, TypeError, "mappings"),
            ({"on_degraded": "skip"}, ValueError, "on_degraded"),
        ],
    )
    def test_refused(self, kwargs, error, match):
        options = {"retriever": lookup, "queries": QUERIES, "qrels": QRELS}
        options.update(kwargs)
        with pytest.raises(error, match=match):
            evaluate_retrieval(options.pop("retriever"), options.pop("queries"), options.pop("qrels"), **options)


class TestDegradedResponses:
    @staticmethod
    def degraded(_query):
        error = ErrorRecord(code="NO_QUERY_EMBEDDING", category=ErrorCategory.CAPABILITY, message="no embedding")
        doc = CorpusDocument(doc_id="d1", input_path="memory://d1", chunk_index=0, text="t")
        from ..._similarity import RetrievalHit

        return RetrievalResponse(
            hits=[RetrievalHit(doc=doc, score=1.0, match_mode="hybrid")],
            legs=[
                LegOutcome(LegKind.LEXICAL, LegStatus.SUCCESS),
                LegOutcome(LegKind.DENSE, LegStatus.FAILED, error=error),
            ],
        )

    def test_raise_by_default(self):
        with pytest.raises(ValueError, match=r"query 'q'.*'degraded' response"):
            evaluate_retrieval(self.degraded, {"q": "x"}, {"q": ["d1"]})

    def test_warn_records_the_status(self):
        with pytest.warns(RuntimeWarning, match="1 of 1 responses were degraded"):
            out = evaluate_retrieval(self.degraded, {"q": "x"}, {"q": ["d1"]}, metrics=["mrr"], on_degraded="warn")
        assert out.status_counts == {"degraded": 1}
        assert out.metrics == {"mrr": 1.0}

    def test_ignore_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = evaluate_retrieval(self.degraded, {"q": "x"}, {"q": ["d1"]}, metrics=["mrr"], on_degraded="ignore")
        assert out.status_counts == {"degraded": 1}


class TestReports:
    @pytest.mark.parametrize("fmt", ["json", "yaml", "toml"])
    def test_round_trip(self, report, tmp_path, fmt):
        if fmt == "yaml":
            pytest.importorskip("yaml")
        if fmt == "toml":
            pytest.importorskip("tomli_w")
        path = report.write(tmp_path / f"report.{fmt}")
        again = RetrievalEvaluation.read(path)
        assert again.metrics == report.metrics
        assert again.queries == report.queries
        assert again.settings["metrics"] == report.settings["metrics"]
        assert again.metadata == report.metadata

    def test_nan_round_trips_through_strict_json(self, tmp_path):
        out = evaluate_retrieval(lookup, QUERIES, {**QRELS, "q3": []}, metrics=["recall@2"], zero_division=np.nan)
        text = out.dumps("json")
        assert "NaN" not in text
        assert '"recall@2": null' in text
        again = RetrievalEvaluation.from_dict(json.loads(text))
        assert math.isnan(again.queries[2].metrics["recall@2"])

    def test_jsonl_rows(self, report, tmp_path):
        path = report.write(tmp_path / "rows.jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [row["query_id"] for row in rows] == ["q1", "q2", "q3"]
        assert rows[0]["recall@2"] == 1.0
        assert rows[0]["retrieved"] == ["d1", "d2"]
        with pytest.raises(ValueError, match="per-query rows"):
            RetrievalEvaluation.read(path)

    def test_schema_is_checked(self, report):
        data = report.to_dict()
        assert data["schema"] == SCHEMA
        with pytest.raises(ValueError, match="Not a retrieval evaluation report"):
            RetrievalEvaluation.from_dict({**data, "schema": "other"})
        with pytest.raises(ValueError, match="schema version"):
            RetrievalEvaluation.from_dict({**data, "schema_version": 99})

    def test_unknown_suffix(self, report, tmp_path):
        with pytest.raises(ValueError, match="Cannot infer a format"):
            report.write(tmp_path / "report.txt")
        assert report.write(tmp_path / "report.txt", format="json").is_file()


class TestComparison:
    def _with_values(self, values):
        queries = tuple(
            QueryEvaluation(f"q{i}", {"m": v}, n_retrieved=1, n_relevant=1, latency_s=0.0)
            for i, v in enumerate(values)
        )
        return RetrievalEvaluation(metrics={"m": float(np.nanmean(values))}, queries=queries, settings={})

    def test_exact_p_values_by_hand(self):
        # d = (1, 1, 1): of the 8 sign flips only (+,+,+) and (-,-,-) reach |mean| = 1.
        out = self._with_values([1, 1, 1]).compare(self._with_values([0, 0, 0]))
        (row,) = out.metrics
        assert (row.delta, row.wins, row.ties, row.losses, row.n_pairs) == (1.0, 3, 0, 0, 3)
        assert row.p_value == 2 / 8
        # d = (1, 0): flipping the 0 changes nothing, so all four flips reach |mean| = 0.5.
        (row,) = self._with_values([1, 0]).compare(self._with_values([0, 0])).metrics
        assert row.p_value == 1.0
        assert row.ties == 1

    def test_identical_systems(self):
        (row,) = self._with_values([0.2, 0.4]).compare(self._with_values([0.2, 0.4])).metrics
        assert row.delta == 0.0
        assert row.p_value == 1.0
        assert row.ties == 2

    def test_undefined_pairs_are_excluded(self):
        (row,) = self._with_values([1.0, np.nan, 0.5]).compare(self._with_values([0.0, 0.0, np.nan])).metrics
        assert row.n_pairs == 1
        assert row.delta == 1.0

    def test_sampled_path_is_seeded(self):
        rng = np.random.default_rng(0)
        n = EXACT_SIGN_FLIP_LIMIT + 9
        a = self._with_values(rng.uniform(0.5, 1.0, n))
        b = self._with_values(rng.uniform(0.0, 0.5, n))
        first = a.compare(b, n_resamples=2000, seed=5).metrics[0].p_value
        assert first == a.compare(b, n_resamples=2000, seed=5).metrics[0].p_value
        assert first == pytest.approx(1 / 2001)  # every sampled flip falls short

    def test_sampled_p_value_bounds(self):
        diff = np.array([0.1, -0.1] * 20)
        p = _sign_flip_p_value(diff, n_resamples=500, seed=1)
        assert p == 1.0  # mean is exactly zero
        diff = np.array([0.1, -0.05] * 20)
        p = _sign_flip_p_value(diff, n_resamples=500, seed=1)
        assert 1 / 501 <= p <= 1.0

    def test_refusals(self, report):
        with pytest.raises(TypeError):
            report.compare({"m": 1})
        with pytest.raises(ValueError, match="same queries"):
            report.compare(evaluate_retrieval(lookup, {"q1": "cat food"}, {"q1": ["d2"]}, metrics=METRICS))
        other = evaluate_retrieval(lookup, QUERIES, QRELS, metrics=["map"])
        with pytest.raises(ValueError, match="share no metric"):
            report.compare(other)
        with pytest.raises(ValueError, match="n_resamples"):
            report.compare(report, n_resamples=0)

    def test_comparison_exports(self, report, tmp_path):
        out = report.compare(report)
        assert isinstance(out, EvaluationComparison)
        data = json.loads(out.dumps("json"))
        assert data["schema"] == "scikitplot.corpus.retrieval_comparison"
        assert [m["metric"] for m in data["metrics"]] == METRICS
        rows = out.dumps("jsonl").splitlines()
        assert len(rows) == len(METRICS)
        assert out.write(tmp_path / "cmp.json").is_file()



class TestReviewFixes:
    def test_latency_includes_a_lazy_retriever(self):
        """Review F3: a generator does its work while its results are read."""
        now = [0.0]

        def clock():
            return now[0]

        def lazy(query):
            for doc in ("d1", "d2"):
                now[0] += 1.0  # work happens here, during iteration
                yield (doc, 1.0)

        out = evaluate_retrieval(lazy, {"q": "x"}, {"q": ["d2"]}, metrics=["mrr"], clock=clock)
        assert out.queries[0].latency_s == 2.0

    def test_settings_hold_no_none(self, report):
        """Review F9: a None setting vanished in TOML."""
        assert "id_getter" not in report.settings
        assert None not in report.settings.values()

    def test_toml_refuses_none_in_metadata(self, tmp_path):
        out = evaluate_retrieval(lookup, QUERIES, QRELS, metrics=["mrr"], metadata={"note": None})
        with pytest.raises(ValueError, match="metadata holds None"):
            out.write(tmp_path / "r.toml")
        assert RetrievalEvaluation.read(out.write(tmp_path / "r.json")).metadata == {"note": None}

    def test_toml_round_trip_keeps_every_setting(self, report, tmp_path):
        pytest.importorskip("tomli_w")
        with_getter = evaluate_retrieval(lookup, QUERIES, QRELS, metrics=METRICS, id_getter=lambda hit: hit[0])
        for evaluation in (report, with_getter):
            again = RetrievalEvaluation.read(evaluation.write(tmp_path / "r.toml"))
            assert again.settings == evaluation.settings

    def test_negative_seed_is_refused_on_both_paths(self, report):
        """Review F18."""
        with pytest.raises(ValueError, match="seed must be >= 0"):
            report.compare(report, seed=-1)

    def _evaluation(self, values):
        queries = tuple(
            QueryEvaluation(f"q{i}", {"m": float(v)}, n_retrieved=1, n_relevant=1, latency_s=0.0)
            for i, v in enumerate(values)
        )
        return RetrievalEvaluation(metrics={"m": float(np.mean(values))}, queries=queries, settings={})

    def test_flips_equal_for_the_represented_values_count(self):
        """Review F10: 0.3 + 0.4 and 0.7 differ in binary floating point."""
        cand = [0.0, 0.3, 0.4, 0.001]
        base = [0.7, 0.0, 0.0, 0.0]
        (row,) = self._evaluation(cand).compare(self._evaluation(base)).metrics
        assert row.p_value == 1.0

    def test_exact_p_values_match_rational_arithmetic(self):
        """Review F10: against exact sums of the rationals the metrics represent."""
        import random
        from fractions import Fraction

        values = sorted({1.0 / r for r in range(1, 11)} | {a / b for b in range(1, 7) for a in range(b + 1)})
        rng = random.Random(7)  # noqa: S311 - seeded test data
        for _ in range(1500):
            n = rng.randint(2, 8)
            cand = [rng.choice(values) for _ in range(n)]
            base = [rng.choice(values) for _ in range(n)]
            d = [Fraction(c).limit_denominator(1000) - Fraction(b).limit_denominator(1000) for c, b in zip(cand, base)]
            observed = abs(sum(d))
            if observed == 0:
                expected = 1.0
            else:
                hits = sum(
                    1
                    for signs in itertools.product((1, -1), repeat=n)
                    if abs(sum(s * x for s, x in zip(signs, d))) >= observed
                )
                expected = hits / 2**n
            got = self._evaluation(cand).compare(self._evaluation(base)).metrics[0].p_value
            assert got == expected, (cand, base)


def test_compare_still_refuses_a_query_without_the_metric():
    """Review of 2b #10: the shared comparison skipped such a pair instead of raising KeyError."""
    from .._evaluate import QueryEvaluation, RetrievalEvaluation

    def report(values):
        queries = tuple(QueryEvaluation(q, metrics, 1, 1, 0.0) for q, metrics in values.items())
        return RetrievalEvaluation({"recall@2": 0.5}, queries, {})

    candidate = report({"a": {"recall@2": 1.0}, "b": {}})
    baseline = report({"a": {"recall@2": 0.0}, "b": {"recall@2": 0.0}})
    with pytest.raises(KeyError, match="recall@2"):
        candidate.compare(baseline)
