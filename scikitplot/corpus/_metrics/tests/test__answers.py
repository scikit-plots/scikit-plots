# scikitplot/corpus/_metrics/tests/test__answers.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The answer harness: inputs, judges and panels, failures, reports, comparison."""

from __future__ import annotations

import json
import math
import threading
import time

import pytest

from .._answers import (
    AnswerEvaluation,
    PreferenceEvaluation,
    evaluate_answers,
    evaluate_preference,
)
from .._judge import LLMJudge
from .._llm import CacheError, Completion, make_llm

RELEVANCE = '{"score": %d, "reason": "r"}'


def scripted(**by_metric):
    """A judge model that answers by the metric its prompt asks about."""
    calls = []
    lock = threading.Lock()

    def model(prompt, system):
        if '"claims"' in prompt:
            metric = "faithfulness"
        elif '"verdicts"' in prompt:
            metric = "context_precision"
        elif '"statements"' in prompt:
            metric = "context_recall"
        elif '"true_positives"' in prompt:
            metric = "answer_correctness"
        elif "<criteria>" in prompt:
            metric = "criteria"
        elif '"winner"' in prompt:
            metric = "preference"
        else:
            metric = "answer_relevance"
        with lock:
            calls.append((metric, prompt))
        out = by_metric[metric]
        out = out(prompt) if callable(out) else out
        if isinstance(out, BaseException):
            raise out
        return Completion(out, "callable", "m", input_tokens=10, output_tokens=2)

    model.calls = calls
    return model


DATA = [
    {"id": "q1", "question": "Capital of France?", "reference": "Paris", "contexts": ["Paris is the capital."]},
    {"id": "q2", "question": "Largest planet?", "reference": ["Jupiter", "planet Jupiter"]},
    {"id": "q3", "question": "Who wrote Hamlet?"},
]


class TestTextMetrics:
    def test_without_judges(self):
        answers = {"Capital of France?": "Paris", "Largest planet?": "Saturn", "Who wrote Hamlet?": "Shakespeare"}
        report = evaluate_answers(answers.get, DATA)
        assert list(report.metrics) == ["exact_match", "token_f1"]
        assert report.per_item("exact_match").tolist()[:2] == [1.0, 0.0]
        assert math.isnan(report.per_item("exact_match")[2])  # no reference
        assert report.metrics["exact_match"] == 0.5
        assert report.coverage == {"exact_match": 2, "token_f1": 2}

    def test_stored_answers(self):
        data = [dict(row, answer=a) for row, a in zip(DATA, ["Paris", "planet Jupiter", "W. Shakespeare"])]
        report = evaluate_answers(None, data)
        assert report.per_item("exact_match").tolist()[:2] == [1.0, 1.0]
        assert report.settings["system"] == "stored answers"
        assert all(item.latency_s == 0.0 for item in report.items)

    def test_stored_answer_missing(self):
        with pytest.raises(ValueError, match="no 'answer'"):
            evaluate_answers(None, DATA)


class TestJudges:
    def test_metrics_per_judge_and_panel(self):
        gpt = scripted(faithfulness='{"claims": [{"claim": "c", "supported": true}]}', answer_relevance=RELEVANCE % 5)
        claude = scripted(faithfulness='{"claims": [{"claim": "c", "supported": false}]}', answer_relevance=RELEVANCE % 3)
        report = evaluate_answers(lambda q: "Paris", DATA[:1], judges={"gpt": gpt, "claude": claude})
        assert report.metrics["faithfulness/gpt"] == 1.0
        assert report.metrics["faithfulness/claude"] == 0.0
        assert report.metrics["faithfulness"] == 0.5
        assert report.metrics["answer_relevance"] == (1.0 + 0.5) / 2
        assert set(report.judges) == {"gpt", "claude"}
        assert report.judges["gpt"]["llm"]["provider"] == "callable"
        assert report.usage["gpt"] == {"responses": 2, "cached_responses": 0, "input_tokens": 20, "output_tokens": 4,
                                       "responses_without_token_counts": 0}

    def test_missing_inputs_cost_no_call(self):
        model = scripted(faithfulness='{"claims": []}', answer_relevance=RELEVANCE % 4,
                         context_recall='{"statements": []}', answer_correctness='{"true_positives": ["t"], '
                         '"false_positives": [], "false_negatives": []}')
        report = evaluate_answers(
            lambda q: "x", DATA, judges=model,
            metrics=["faithfulness", "answer_relevance", "context_recall", "answer_correctness"],
        )
        undefined = {
            (item.item_id, metric): result.undefined
            for item in report.items for metric, by_judge in item.judgments.items() for result in by_judge.values()
        }
        assert undefined[("q2", "faithfulness")] == "no contexts"
        assert undefined[("q3", "context_recall")] == "no contexts or reference"
        assert undefined[("q2", "answer_correctness")] == "2 references; this judge metric takes one"
        assert undefined[("q3", "answer_correctness")] == "no reference"
        asked = [metric for metric, _ in model.calls]
        assert asked.count("faithfulness") == 1
        assert asked.count("answer_correctness") == 1
        assert asked.count("answer_relevance") == 3

    def test_system_contexts_replace_the_dataset_contexts(self):
        model = scripted(faithfulness='{"claims": [{"claim": "c", "supported": true}]}')
        system = lambda q: {"answer": "Paris", "contexts": [{"text": "retrieved"}]}  # noqa: E731
        report = evaluate_answers(system, DATA[:1], judges=model, metrics=["faithfulness"])
        assert "retrieved" in model.calls[0][1]
        assert "Paris is the capital." not in model.calls[0][1]
        assert report.items[0].n_contexts == 1

    def test_criteria(self):
        model = scripted(criteria='{"score": 5, "reason": "r"}')
        report = evaluate_answers(lambda q: "a", DATA[:1], judges=model, metrics=[], criteria={"concise": "Short."})
        assert report.metrics == {"exact_match": 0.0, "token_f1": 0.0, "criteria:concise/judge": 1.0,
                                  "criteria:concise": 1.0}

    def test_agreement_between_judges(self):
        def rating(values):
            queue = list(values)
            return lambda prompt: RELEVANCE % queue.pop(0)

        data = [{"question": f"q{i}"} for i in range(6)]
        same = {"a": scripted(answer_relevance=rating([1, 2, 3, 4, 5, 3])),
                "b": scripted(answer_relevance=rating([1, 2, 3, 4, 5, 3]))}
        report = evaluate_answers(lambda q: "x", data, judges=same, metrics=["answer_relevance"], text_metrics=[])
        assert report.agreement == {"answer_relevance": 1.0}
        single = evaluate_answers(lambda q: "x", data, judges=scripted(answer_relevance=RELEVANCE % 3),
                                  metrics=["answer_relevance"], text_metrics=[])
        assert single.agreement == {}

    def test_order_is_kept_with_threads(self):
        def slow(prompt):
            time.sleep(0.01 if "q0" in prompt else 0.0)
            return RELEVANCE % (int(prompt.split("<question>\nq")[1][0]) % 5 + 1)

        data = [{"question": f"q{i}"} for i in range(8)]
        report = evaluate_answers(lambda q: q, data, judges=scripted(answer_relevance=slow), metrics=["answer_relevance"],
                                  text_metrics=[], n_jobs=4)
        assert [item.item_id for item in report.items] == list(range(8))
        assert report.per_item("answer_relevance").tolist() == [((i % 5 + 1) - 1) / 4 for i in range(8)]


class TestFailures:
    def test_system_failure_is_recorded(self):
        def system(q):
            raise RuntimeError("model down")

        report = evaluate_answers(system, DATA[:1], judges=scripted(answer_relevance=RELEVANCE % 5),
                                  metrics=["answer_relevance"])
        item = report.items[0]
        assert item.answer is None
        assert item.error == "RuntimeError: model down"
        assert math.isnan(item.scores["answer_relevance"])
        assert report.errors == {"system": 1, "judge": 0}

    def test_system_failure_raises(self):
        with pytest.raises(ZeroDivisionError):
            evaluate_answers(lambda q: 1 / 0, DATA[:1], on_error="raise")

    def test_bad_system_output(self):
        report = evaluate_answers(lambda q: 42, DATA[:1])
        assert "must return a str" in report.items[0].error

    def test_judge_failure_is_recorded(self):
        report = evaluate_answers(lambda q: "x", DATA[:1], judges=scripted(answer_relevance="I think 4"),
                                  metrics=["answer_relevance"])
        result = report.items[0].judgments["answer_relevance"]["judge"]
        assert result.error.startswith("parse:")
        assert report.errors == {"system": 0, "judge": 1}
        assert math.isnan(report.metrics["answer_relevance"])

    def test_judge_failure_raises(self):
        with pytest.raises(ValueError, match="Invalid judge verdict"):
            evaluate_answers(lambda q: "x", DATA[:1], judges=scripted(answer_relevance="I think 4"),
                             metrics=["answer_relevance"], on_error="raise")

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"dataset": "q"}, TypeError, "dataset"),
            ({"dataset": []}, ValueError, "empty"),
            ({"dataset": [{"question": 1}]}, TypeError, "question"),
            ({"dataset": [{"question": "q", "extra": 1}]}, ValueError, "unknown keys"),
            ({"dataset": [{"question": "q", "id": 1}, {"question": "q", "id": 1}]}, ValueError, "duplicate id"),
            ({"dataset": [{"question": "q", "id": 1}, {"question": "q", "id": "1"}]}, TypeError, "mix"),
            ({"dataset": [{"question": "q", "id": True}]}, TypeError, "id"),
            ({"dataset": [{"question": "q", "reference": []}]}, TypeError, "reference"),
            ({"dataset": [{"question": "q", "contexts": "one"}]}, TypeError, "contexts"),
            ({"metrics": ["recall@10"]}, ValueError, "Unknown judge metric"),
            ({"metrics": ["criteria"]}, ValueError, "criteria="),
            ({"metrics": ["faithfulness", "faithfulness"]}, ValueError, "repeat"),
            ({"criteria": {"bad name": "x"}}, ValueError, "must match"),
            ({"criteria": {"ok": ""}}, TypeError, "non-empty"),
            ({"text_metrics": ["bleu"]}, ValueError, "Unknown text metric"),
            ({"judges": {"a b": lambda p, s: "x"}}, ValueError, "must match"),
            ({"judges": 3}, TypeError, "judges"),
            ({"on_error": "ignore"}, ValueError, "on_error"),
            ({"n_jobs": 0}, ValueError, "n_jobs"),
            ({"keep_verdicts": 1}, TypeError, "keep_verdicts"),
            ({"metadata": {"k": object()}}, TypeError, None),
        ],
    )
    def test_refusals(self, kwargs, error, match):
        options = {"dataset": DATA[:1], **kwargs}
        dataset = options.pop("dataset")
        with pytest.raises(error, match=match):
            evaluate_answers(lambda q: "x", dataset, **options)


class TestReport:
    @pytest.fixture
    def report(self):
        model = scripted(faithfulness='{"claims": [{"claim": "c", "supported": true}]}',
                         answer_relevance=lambda p: "oops" if "Largest" in p else RELEVANCE % 4)
        def system(question):
            if question.startswith("Who"):
                raise RuntimeError("down")
            return "Paris"

        return evaluate_answers(system, DATA, judges={"gpt": model}, metadata={"run": 3})

    @pytest.mark.parametrize("fmt", ["json", "yaml", "toml"])
    def test_round_trip(self, report, fmt, tmp_path):
        if fmt == "yaml":
            pytest.importorskip("yaml")
        if fmt == "toml":
            pytest.importorskip("tomli_w")
        path = report.write(tmp_path / f"r.{fmt}")
        back = AnswerEvaluation.read(path)
        assert _same(back, report)

    def test_json_has_no_nan_token(self, report):
        text = report.dumps("json")
        assert "NaN" not in text
        assert json.loads(text)["metrics"]["answer_relevance/gpt"] == 0.75

    def test_rows(self, report):
        rows = [json.loads(line) for line in report.dumps("jsonl").splitlines()]
        assert [row["item_id"] for row in rows] == ["q1", "q2", "q3"]
        assert rows[1]["answer_relevance/gpt"] is None

    def test_jsonl_is_not_a_report(self, report, tmp_path):
        path = report.write(tmp_path / "r.jsonl")
        with pytest.raises(ValueError, match="JSON Lines"):
            AnswerEvaluation.read(path)

    def test_toml_refuses_none_metadata(self, report):
        bad = AnswerEvaluation(report.metrics, report.items, report.judges, report.agreement, report.usage,
                               report.errors, report.settings, {"k": None})
        with pytest.raises(ValueError, match="metadata holds None"):
            bad.dumps("toml")

    def test_schema_is_checked(self):
        with pytest.raises(ValueError, match="Not an answer evaluation"):
            AnswerEvaluation.from_dict({"schema": "other"})

    def test_compare(self, report):
        worse = evaluate_answers(lambda q: "Lyon", DATA, judges={"gpt": scripted(
            faithfulness='{"claims": [{"claim": "c", "supported": false}]}', answer_relevance=RELEVANCE % 2)})
        comparison = report.compare(worse)
        rows = {row.metric: row for row in comparison.metrics}
        assert rows["exact_match"].delta == 0.5  # q1: Paris vs Lyon; q2 both wrong; q3 no reference
        assert rows["faithfulness/gpt"].n_pairs == 1
        assert rows["answer_relevance/gpt"].n_pairs == 1  # q2's verdict and q3's system failed in the candidate
        with pytest.raises(TypeError, match="AnswerEvaluation"):
            report.compare(object())

    def test_keep_verdicts_false(self):
        report = evaluate_answers(lambda q: "x", DATA[:1], judges=scripted(answer_relevance=RELEVANCE % 4),
                                  metrics=["answer_relevance"], keep_verdicts=False)
        assert report.items[0].judgments == {}
        assert report.usage["judge"]["responses"] == 1


def _same(a, b):
    def norm(value):
        return json.loads(json.dumps(value, allow_nan=True).replace("NaN", "null"))

    return norm(a.to_dict()) == norm(b.to_dict())


class TestPreference:
    def test_head_to_head(self):
        def forward_a_wins(prompt):
            first = prompt.split("<answer_a>\n")[1].split("\n</answer_a>")[0]
            return '{"winner": "%s", "reason": "r"}' % ("answer_a" if first == "good" else "answer_b")

        biased = scripted(preference='{"winner": "answer_a", "reason": "r"}')
        data = [{"id": i, "question": f"q{i}"} for i in range(3)]
        report = evaluate_preference(data, ["good"] * 3, ["bad"] * 3,
                                     judges={"fair": scripted(preference=forward_a_wins), "biased": biased},
                                     names=("mine", "gpt"))
        assert report.summary["fair"]["win_rate"] == 1.0
        assert report.summary["fair"]["position_consistency"] == 1.0
        assert report.summary["biased"]["ties"] == 3
        assert report.summary["biased"]["position_consistency"] == 0.0
        assert report.summary["panel"]["win_rate"] == 0.75
        assert report.names == ("mine", "gpt")
        assert report.agreement < 1.0
        rows = report.to_rows()
        assert rows[0] == {"item_id": 0, "score": 0.75, "winner/fair": "a", "winner/biased": "tie"}

    def test_mapping_answers_and_round_trip(self, tmp_path):
        data = [{"id": "x", "question": "q", "reference": "r"}]
        report = evaluate_preference(data, {"x": "A"}, {"x": "B"},
                                     judges=scripted(preference='{"winner": "tie", "reason": "r"}'))
        back = PreferenceEvaluation.read(report.write(tmp_path / "p.json"))
        assert _same(back, report)

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"answers_a": ["A", "A"]}, ValueError, "2 answers for 1"),
            ({"answers_a": {"other": "A"}}, ValueError, "no answer"),
            ({"answers_a": "A"}, TypeError, "sequence"),
            ({"answers_b": [3]}, TypeError, "str"),
            ({"judges": None}, ValueError, "at least one judge"),
            ({"names": ("a", "a")}, ValueError, "differ"),
            ({"names": ["a", "b"]}, TypeError, "tuple"),
        ],
    )
    def test_refusals(self, kwargs, error, match):
        options = {"answers_a": ["A"], "answers_b": ["B"], "judges": lambda p, s: "x", **kwargs}
        with pytest.raises(error, match=match):
            evaluate_preference([{"question": "q"}], options.pop("answers_a"), options.pop("answers_b"), **options)


def test_judge_instances_and_llms_are_accepted():
    llm = make_llm(scripted(answer_relevance=RELEVANCE % 5), model="m")
    for judge in (llm, LLMJudge(llm, on_error="raise")):
        report = evaluate_answers(lambda q: "x", DATA[:1], judges={"j": judge}, metrics=["answer_relevance"])
        assert report.metrics["answer_relevance/j"] == 1.0


class TestReviewFixes:
    """Increment 2b independent review."""

    def test_errors_are_counted_without_verdicts(self):
        """Review #4: keep_verdicts=False reported zero judge errors."""
        report = evaluate_answers(lambda q: "x", DATA, judges={"bad": scripted(answer_relevance="nope")},
                                  metrics=["answer_relevance"], keep_verdicts=False)
        assert report.errors == {"system": 0, "bad": 3}
        back = AnswerEvaluation.from_dict(json.loads(report.dumps("json")))
        assert back.errors == {"system": 0, "bad": 3}

    def test_cached_responses_cost_no_tokens(self):
        """Review #8: a fully cached rerun reported the first run's tokens."""
        llm = make_llm(scripted(answer_relevance=RELEVANCE % 4), model="m", cache={})
        first = evaluate_answers(lambda q: "x", DATA, judges={"j": llm}, metrics=["answer_relevance"])
        again = evaluate_answers(lambda q: "x", DATA, judges={"j": llm}, metrics=["answer_relevance"])
        assert first.usage["j"] == {"responses": 3, "cached_responses": 0, "input_tokens": 30, "output_tokens": 6,
                                    "responses_without_token_counts": 0}
        assert again.usage["j"] == {"responses": 3, "cached_responses": 3, "input_tokens": 0, "output_tokens": 0,
                                    "responses_without_token_counts": 0}

    def test_a_broken_cache_stops_the_run(self):
        """Review #1: a cache failure is a configuration error, not a recorded judge failure."""
        class Broken(dict):
            def get(self, key, default=None):
                raise RuntimeError("database is locked")

        llm = make_llm(scripted(answer_relevance=RELEVANCE % 4), model="m", cache=Broken())
        with pytest.raises(CacheError, match="database is locked"):
            evaluate_answers(lambda q: "x", DATA, judges={"j": llm}, metrics=["answer_relevance"], n_jobs=2)

    def test_preference_toml_round_trip_keeps_every_field(self, tmp_path):
        """Review #9: TOML dropped None-valued keys of preference items."""
        pytest.importorskip("tomli_w")
        report = evaluate_preference([{"id": "x", "question": "q"}], ["A"], ["B"],
                                     judges={"j": scripted(preference='{"winner": "tie", "reason": "r"}')})
        back = PreferenceEvaluation.read(report.write(tmp_path / "p.toml"))
        result = back.items[0]["judges"]["j"]
        assert result["error"] is None
        assert result["undefined"] is None
        assert _same(back, report)
