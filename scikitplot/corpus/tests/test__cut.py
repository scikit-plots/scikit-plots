"""
CutPolicy: the three rules, clamps, evidence, refusals.
"""

from __future__ import annotations

import random

import pytest

from scikitplot.corpus import CutPolicy


def test_fixed():
    assert CutPolicy("fixed", max_k=2).cut([3, 2, 1]).k == 2
    assert CutPolicy("fixed", max_k=5).cut([3, 2]).k == 2


def test_threshold_is_k_star():
    result = CutPolicy("threshold", threshold=0.5).cut([0.9, 0.7, 0.5, 0.49, 0.1])
    assert (result.k, result.evidence, result.clamped) == (3, 0.5, False)


def test_threshold_clamps():
    low = CutPolicy("threshold", threshold=0.95, min_k=2).cut([0.9, 0.8, 0.1])
    assert (low.k, low.clamped) == (2, True)
    high = CutPolicy("threshold", threshold=0.1, max_k=1).cut([0.9, 0.8])
    assert (high.k, high.clamped) == (1, True)


def test_threshold_refuses_unscaled_scores():
    with pytest.raises(ValueError, match="calibrate them first"):
        CutPolicy("threshold", threshold=0.5).cut([12.4, 3.1])


def test_max_gap_and_ties():
    assert CutPolicy("max_gap").cut([1.0, 0.95, 0.4, 0.35]).k == 2
    tie = CutPolicy("max_gap").cut([1.0, 0.5, 0.0])  # two equal gaps: earliest
    assert (tie.k, tie.evidence) == (1, 0.5)


def test_max_gap_stays_inside_the_window():
    scores = [1.0, 0.1, 0.09, 0.0]
    assert CutPolicy("max_gap", min_k=2).cut(scores).k == 3  # gap 1->2 excluded
    assert CutPolicy("max_gap", max_k=1).cut(scores).k == 1


def test_degenerate_lists():
    assert CutPolicy("max_gap").cut([]).k == 0
    one = CutPolicy("max_gap").cut([0.3])
    assert (one.k, one.evidence) == (1, None)
    assert CutPolicy("threshold", threshold=0.5).cut([]).k == 0


@pytest.mark.parametrize("seed", range(30))
def test_max_gap_equals_brute_force(seed):
    rng = random.Random(seed)  # noqa: S311
    scores = sorted((round(rng.random(), 1) for _ in range(rng.randint(2, 12))), reverse=True)
    lo, hi = 1, len(scores)
    gaps = [(scores[k - 1] - scores[k], -k) for k in range(lo, hi)]
    want = -max(gaps)[1] if gaps else hi
    assert CutPolicy("max_gap").cut(scores).k == want


def test_apply_uses_hit_scores():
    class Hit:
        def __init__(self, score):
            self.score = score

    hits = [Hit(0.9), Hit(0.8), Hit(0.2)]
    assert CutPolicy("max_gap").apply(hits) == hits[:2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "top"},
        {"mode": "fixed"},
        {"mode": "threshold"},
        {"mode": "threshold", "threshold": 1.5},
        {"mode": "max_gap", "threshold": 0.5},
        {"mode": "max_gap", "min_k": -1},
        {"mode": "max_gap", "min_k": 3, "max_k": 2},
        {"mode": "max_gap", "min_k": True},
    ],
)
def test_policy_refusals(kwargs):
    with pytest.raises(ValueError):
        CutPolicy(**kwargs)


@pytest.mark.parametrize("scores", [[0.2, 0.5], [0.5, float("nan")]])
def test_score_refusals(scores):
    with pytest.raises(ValueError):
        CutPolicy("max_gap").cut(scores)
