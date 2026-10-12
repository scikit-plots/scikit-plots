# scikitplot/corpus/_metrics/tests/test__ranking.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Ranking metrics: hand-computed tables, an independent reference, invariants.

The reference implementation below is written separately from the module
(full-list loops and NumPy arrays instead of the module's early exits), so a
shared misunderstanding would have to be made twice to pass.
"""

from __future__ import annotations

import math
import random
import warnings

import numpy as np
import pytest

from .._inputs import UndefinedMetricWarning
from .._ranking import (
    METRIC_NAMES,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    parse_metric,
    precision_at_k,
    r_precision,
    ranking_metrics,
    recall_at_k,
)

LOG2_3 = math.log2(3)


class TestBinaryTable:
    """R = {a, b, c}; ranking a x b y z (relevant at ranks 1 and 3)."""

    y_true = ({"a", "b", "c"},)
    y_pred = (("a", "x", "b", "y", "z"),)

    @pytest.mark.parametrize(("k", "expected"), [(1, 1.0), (2, 0.5), (3, 2 / 3), (5, 2 / 5), (10, 2 / 10)])
    def test_precision(self, k, expected):
        assert precision_at_k(self.y_true, self.y_pred, k=k) == pytest.approx(expected, abs=0)

    def test_precision_without_cutoff_divides_by_ranking_length(self):
        assert precision_at_k(self.y_true, self.y_pred, k=None) == 2 / 5

    @pytest.mark.parametrize(("k", "expected"), [(1, 1 / 3), (2, 1 / 3), (3, 2 / 3), (None, 2 / 3)])
    def test_recall(self, k, expected):
        assert recall_at_k(self.y_true, self.y_pred, k=k) == expected

    def test_hit_rate_and_mrr(self):
        assert hit_rate_at_k(self.y_true, self.y_pred, k=1) == 1.0
        assert mean_reciprocal_rank(self.y_true, self.y_pred) == 1.0

    @pytest.mark.parametrize(("k", "expected"), [(None, (1 + 2 / 3) / 3), (2, 1 / 3), (3, (1 + 2 / 3) / 3)])
    def test_average_precision(self, k, expected):
        assert mean_average_precision(self.y_true, self.y_pred, k=k) == pytest.approx(expected, rel=1e-15)

    def test_r_precision(self):
        assert r_precision(self.y_true, self.y_pred) == 2 / 3

    def test_ndcg(self):
        dcg = 1 / 1 + 1 / 2  # ranks 1 and 3: log2(2) = 1, log2(4) = 2
        idcg = 1 / 1 + 1 / LOG2_3 + 1 / 2
        assert ndcg_at_k(self.y_true, self.y_pred, k=5) == pytest.approx(dcg / idcg, rel=1e-15)


class TestGradedTable:
    """Grades a=3, b=2, c=0, d=1; ranking c b a e."""

    y_true = ({"a": 3, "b": 2, "c": 0, "d": 1},)
    y_pred = (("c", "b", "a", "e"),)

    def test_threshold_one(self):
        assert precision_at_k(self.y_true, self.y_pred, k=2) == 0.5
        assert recall_at_k(self.y_true, self.y_pred, k=3) == 2 / 3
        assert mean_reciprocal_rank(self.y_true, self.y_pred) == 0.5
        assert mean_average_precision(self.y_true, self.y_pred) == pytest.approx((1 / 2 + 2 / 3) / 3, rel=1e-15)

    def test_threshold_two(self):
        assert recall_at_k(self.y_true, self.y_pred, k=3, relevance_threshold=2) == 1.0

    def test_ndcg_linear(self):
        dcg = 0 / 1 + 2 / LOG2_3 + 3 / 2
        idcg = 3 / 1 + 2 / LOG2_3 + 1 / 2
        assert ndcg_at_k(self.y_true, self.y_pred, k=3) == pytest.approx(dcg / idcg, rel=1e-15)

    def test_ndcg_exponential(self):
        dcg = 0 / 1 + 3 / LOG2_3 + 7 / 2
        idcg = 7 / 1 + 3 / LOG2_3 + 1 / 2
        assert ndcg_at_k(self.y_true, self.y_pred, k=3, gain="exponential") == pytest.approx(
            dcg / idcg, rel=1e-15
        )

    def test_ndcg_uses_grades_below_the_threshold(self):
        # d has grade 1 < threshold 2 but still adds gain to nDCG.
        a = ndcg_at_k([{"d": 1}], [["d"]], relevance_threshold=2)
        assert a == 1.0


# ---------------------------------------------------------------------------
# Independent reference
# ---------------------------------------------------------------------------


def _reference(base, k, grades, ranked, threshold, gain):  # noqa: PLR0911, PLR0917 - one oracle per metric
    """Straightforward definitions over full lists; NaN when undefined."""
    relevant = {d for d, g in grades.items() if g >= threshold}
    cutoff = len(ranked) if k is None else k
    flags = np.array([d in relevant for d in ranked[:cutoff]], dtype=float)
    if base == "precision":
        return math.nan if cutoff == 0 else flags.sum() / cutoff
    if base == "recall":
        return math.nan if not relevant else flags.sum() / len(relevant)
    if base == "hit_rate":
        return float(flags.sum() > 0)
    if base == "mrr":
        positions = np.flatnonzero(flags)
        return 0.0 if positions.size == 0 else 1.0 / (positions[0] + 1)
    if base == "map":
        if not relevant:
            return math.nan
        precision_at = np.cumsum(flags) / np.arange(1, flags.size + 1)
        return float((precision_at * flags).sum() / len(relevant))
    if base == "r_precision":
        if not relevant:
            return math.nan
        top = ranked[: len(relevant)]
        return sum(d in relevant for d in top) / len(relevant)
    if base == "ndcg":
        g = (lambda x: x) if gain == "linear" else (lambda x: 2.0**x - 1.0)
        discounts = 1.0 / np.log2(np.arange(2, 2 + max(cutoff, len(grades)) + 1))
        got = np.array([g(grades.get(d, 0.0)) for d in ranked[:cutoff]])
        ideal_grades = sorted((v for v in grades.values() if v > 0), reverse=True)
        ideal = np.array([g(v) for v in ideal_grades[: (len(ideal_grades) if k is None else k)]])
        idcg = float((ideal * discounts[: ideal.size]).sum())
        if idcg == 0.0:
            return math.nan
        return float((got * discounts[: got.size]).sum()) / idcg
    raise AssertionError(base)


def _random_case(rng, n_queries):
    pool = [f"d{i}" for i in range(12)]
    y_true, y_pred = [], []
    for _ in range(n_queries):
        judged = rng.sample(pool, rng.randint(0, 6))
        y_true.append({d: rng.choice([0, 0.5, 1, 2, 3]) for d in judged})
        y_pred.append(rng.sample(pool, rng.randint(0, 10)))
    return y_true, y_pred


NAMES = [
    "precision", "precision@1", "precision@3", "recall", "recall@2", "recall@5",
    "hit_rate@1", "hit_rate@4", "mrr", "mrr@3", "map", "map@4", "r_precision",
    "ndcg", "ndcg@1", "ndcg@5",
]


@pytest.mark.parametrize("gain", ["linear", "exponential"])
@pytest.mark.parametrize("threshold", [0.5, 1, 2])
def test_matches_the_reference_on_random_queries(gain, threshold):
    rng = random.Random(1000 + int(threshold * 10) + len(gain))  # noqa: S311 - seeded test data
    y_true, y_pred = _random_case(rng, 200)
    got = ranking_metrics(
        y_true, y_pred, metrics=NAMES, relevance_threshold=threshold,
        zero_division=np.nan, gain=gain, average=None,
    )
    for name in NAMES:
        base, k = parse_metric(name)
        expected = np.array([_reference(base, k, t, p, threshold, gain) for t, p in zip(y_true, y_pred)])
        np.testing.assert_allclose(got[name], expected, rtol=1e-12, atol=0, equal_nan=True, err_msg=name)


def test_invariants_on_random_queries():
    rng = random.Random(7)  # noqa: S311 - seeded test data
    y_true, y_pred = _random_case(rng, 300)
    names = ["recall@1", "recall@3", "recall@10", "hit_rate@1", "hit_rate@3", "map@3", "ndcg@3"]
    got = ranking_metrics(y_true, y_pred, metrics=names, zero_division=np.nan, average=None)
    for values in got.values():
        defined = values[~np.isnan(values)]
        assert np.all(defined >= 0.0)
        assert np.all(defined <= 1.0 + 1e-12)
    for a, b in (("recall@1", "recall@3"), ("recall@3", "recall@10"), ("hit_rate@1", "hit_rate@3")):
        both = ~(np.isnan(got[a]) | np.isnan(got[b]))
        assert np.all(got[a][both] <= got[b][both]), (a, b)
    both = ~(np.isnan(got["map@3"]) | np.isnan(got["recall@3"]))
    assert np.all(got["map@3"][both] <= got["recall@3"][both] + 1e-15)


def test_the_ideal_ranking_has_ndcg_exactly_one():
    rng = random.Random(3)  # noqa: S311 - seeded test data
    for _ in range(200):
        grades = {f"d{i}": rng.choice([0.5, 1, 2, 3]) for i in range(rng.randint(1, 8))}
        ideal = sorted(grades, key=lambda d: -grades[d])
        assert ndcg_at_k([grades], [ideal], k=None) == 1.0
        assert ndcg_at_k([grades], [ideal], k=3) == 1.0


def test_r_precision_equals_precision_and_recall_at_r():
    y_true = [{"a", "b", "c"}]
    y_pred = [["a", "q", "c", "b"]]
    assert r_precision(y_true, y_pred) == precision_at_k(y_true, y_pred, k=3) == recall_at_k(y_true, y_pred, k=3)


class TestUndefinedValues:
    y_true = (("a",), ())
    y_pred = (("a",), ("b",))

    def test_recall_warns_and_scores_zero(self):
        with pytest.warns(UndefinedMetricWarning, match=r"recall@1 is undefined for 1 of 2"):
            assert recall_at_k(self.y_true, self.y_pred, k=1) == 0.5

    def test_warning_points_at_the_caller(self):
        with pytest.warns(UndefinedMetricWarning) as record:
            recall_at_k(self.y_true, self.y_pred, k=1)
        assert record[0].filename == __file__

    @pytest.mark.parametrize(("policy", "expected"), [(0.0, 0.5), (1.0, 1.0)])
    def test_explicit_values(self, policy, expected):
        assert recall_at_k(self.y_true, self.y_pred, k=1, zero_division=policy) == expected

    def test_nan_excludes(self):
        assert recall_at_k(self.y_true, self.y_pred, k=1, zero_division=np.nan) == 1.0
        per_query = recall_at_k(self.y_true, self.y_pred, k=1, zero_division=np.nan, average=None)
        assert per_query[0] == 1.0
        assert math.isnan(per_query[1])

    def test_defined_metrics_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert precision_at_k(self.y_true, self.y_pred, k=1) == 0.5
            assert hit_rate_at_k(self.y_true, self.y_pred, k=1) == 0.5
            assert mean_reciprocal_rank(self.y_true, self.y_pred) == 0.5

    def test_ndcg_undefined_when_every_grade_is_zero(self):
        out = ndcg_at_k([{"a": 0}], [["a"]], zero_division=np.nan, average=None)
        assert math.isnan(out[0])

    def test_precision_without_cutoff_on_an_empty_ranking(self):
        out = precision_at_k([["a"]], [[]], k=None, zero_division=np.nan, average=None)
        assert math.isnan(out[0])


class TestInputForms:
    def test_mappings_equal_sequences(self):
        qrels = {"q1": {"a": 2}, "q2": ["b", "c"]}
        runs = {"q2": ["c", "x"], "q1": ["x", "a"]}
        by_key = ranking_metrics(qrels, runs, metrics=["recall@2", "ndcg@2"], average=None)
        by_position = ranking_metrics(
            [qrels["q1"], qrels["q2"]], [runs["q1"], runs["q2"]],
            metrics=["recall@2", "ndcg@2"], average=None,
        )
        for name in by_key:
            np.testing.assert_array_equal(by_key[name], by_position[name])

    def test_sample_weight(self):
        assert recall_at_k([["a"], ["b"]], [["a"], ["x"]], k=1, sample_weight=[3, 1]) == 0.75

    def test_duplicates(self):
        with pytest.raises(ValueError, match="duplicates='first'"):
            recall_at_k([["a"]], [["x", "x", "a"]], k=2)
        assert recall_at_k([["a"]], [["x", "x", "a"]], k=2, duplicates="first") == 1.0

    def test_identifier_types_are_not_mixed(self):
        with pytest.raises(ValueError, match="mix str and int"):
            recall_at_k([{"1": 1}], [[1]])

    def test_a_string_is_not_a_ranking(self):
        with pytest.raises(TypeError, match="query at position 0"):
            recall_at_k([["a"]], ["abc"])

    def test_error_names_the_query_key(self):
        with pytest.raises(TypeError, match="query 'q9'"):
            recall_at_k({"q9": "a"}, {"q9": ["a"]})


class TestNames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("ndcg@10", ("ndcg", 10)), ("mrr", ("mrr", None)), ("r_precision", ("r_precision", None))],
    )
    def test_parse(self, name, expected):
        assert parse_metric(name) == expected

    @pytest.mark.parametrize(
        ("name", "match"),
        [
            ("precision@0", "Malformed"),
            ("recall@", "Malformed"),
            ("Recall@10", "Malformed"),
            ("f1@5", "Unknown ranking metric 'f1'"),
            ("r_precision@5", "takes no cutoff"),
            ("mrr\n", "Malformed"),
            ("recall@10\n", "Malformed"),
        ],
    )
    def test_refused(self, name, match):
        with pytest.raises(ValueError, match=match):
            parse_metric(name)

    def test_non_string(self):
        with pytest.raises(TypeError):
            parse_metric(10)

    def test_known_names(self):
        assert METRIC_NAMES == ("hit_rate", "map", "mrr", "ndcg", "precision", "r_precision", "recall")


class TestRankingMetrics:
    def test_equals_the_single_functions(self):
        rng = random.Random(11)  # noqa: S311 - seeded test data
        y_true, y_pred = _random_case(rng, 50)
        options = {"zero_division": 0.0}
        both = ranking_metrics(y_true, y_pred, metrics=["recall@3", "ndcg@3", "map", "mrr"], **options)
        assert both["recall@3"] == recall_at_k(y_true, y_pred, k=3, **options)
        assert both["ndcg@3"] == ndcg_at_k(y_true, y_pred, k=3, **options)
        assert both["map"] == mean_average_precision(y_true, y_pred, **options)
        assert both["mrr"] == mean_reciprocal_rank(y_true, y_pred, **options)

    def test_names_are_checked(self):
        with pytest.raises(TypeError, match="collection of names"):
            ranking_metrics([["a"]], [["a"]], metrics="mrr")
        with pytest.raises(ValueError, match="at least one"):
            ranking_metrics([["a"]], [["a"]], metrics=[])
        with pytest.raises(ValueError, match="twice"):
            ranking_metrics([["a"]], [["a"]], metrics=["mrr", "mrr"])

    def test_gain_is_checked(self):
        with pytest.raises(ValueError, match="gain"):
            ndcg_at_k([["a"]], [["a"]], gain="log")



class TestOverflow:
    """Review F12: huge grades raised OverflowError or scored 0 with a wrong reason."""

    def test_exponential_gain_overflow_is_a_clear_error(self):
        with pytest.raises(ValueError, match="overflows the exponential gain"):
            ndcg_at_k([{"a": 1100}], [["a"]], gain="exponential")

    @pytest.mark.parametrize(
        ("grades", "gain"),
        [({"a": 1023.5, "b": 1023.5}, "exponential"), ({"a": 1e308, "b": 1e308, "c": 1e308}, "linear")],
    )
    def test_ideal_dcg_overflow_is_a_clear_error(self, grades, gain):
        with pytest.raises(ValueError, match="ideal DCG overflows"):
            ndcg_at_k([grades], [list(grades)], gain=gain, k=None)

    def test_large_finite_grades_still_score(self):
        assert ndcg_at_k([{"a": 1e300, "b": 1e300}], [["a", "b"]], k=None) == 1.0
