"""
LogOddsFusion: Bayes arithmetic, contracts, diagnostics, and an empirical check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scikitplot.corpus import LogOddsFusion, binary_log_loss, expected_calibration_error


class _Identity:
    """A fitted calibrator whose scores already are probabilities."""

    fitted = True

    def __init__(self, prevalence=0.5):
        self.prevalence_ = prevalence

    def predict_proba(self, scores):
        return list(scores)


def _logit(p):
    return math.log(p / (1 - p))


def test_bayes_rule_with_unit_weights():
    fusion = LogOddsFusion(base_rate=0.2)
    result = fusion.fuse(
        [
            ("lex", [("d", 0.7)], _Identity(0.3)),
            ("dense", [("d", 0.6)], _Identity(0.4)),
        ]
    )
    (item,) = result.items
    want = _logit(0.2) + (_logit(0.7) - _logit(0.3)) + (_logit(0.6) - _logit(0.4))
    assert item.log_odds == pytest.approx(want, abs=1e-12)
    assert item.probability == pytest.approx(1 / (1 + math.exp(-want)), abs=1e-12)
    assert [c.leg for c in item.contributions] == ["lex", "dense"]


def test_a_missing_leg_is_no_evidence():
    result = LogOddsFusion(base_rate=0.3).fuse(
        [("lex", [("a", 0.9)], _Identity(0.3)), ("dense", [("b", 0.9)], _Identity(0.3))]
    )
    for item in result.items:
        assert len(item.contributions) == 1
        assert item.log_odds == pytest.approx(_logit(0.9), abs=1e-12)


def test_base_rate_moves_probabilities_not_order():
    legs = [
        ("lex", [("a", 0.9), ("b", 0.4), ("c", 0.6)], _Identity()),
        ("dense", [("b", 0.95), ("c", 0.2)], _Identity()),
    ]
    low = LogOddsFusion(base_rate=0.05).fuse(legs)
    high = LogOddsFusion(base_rate=0.6).fuse(legs)
    assert [i.key for i in low.items] == [i.key for i in high.items]
    assert all(x.probability < y.probability for x, y in zip(low.items, high.items))


def test_leg_order_does_not_change_the_result():
    a = ("lex", [("a", 0.9), ("b", 0.4)], _Identity(0.3))
    b = ("dense", [("b", 0.95), ("c", 0.2)], _Identity(0.25))
    one = LogOddsFusion(base_rate=0.1).fuse([a, b])
    two = LogOddsFusion(base_rate=0.1).fuse([b, a])
    assert [(i.key, i.log_odds) for i in one.items] == [(i.key, i.log_odds) for i in two.items]


def test_weights_scale_and_zero_removes():
    legs = [("lex", [("a", 0.8)], _Identity()), ("dense", [("z", 0.99)], _Identity())]
    result = LogOddsFusion(base_rate=0.5, weights={"dense": 0.0}).fuse(legs)
    assert [i.key for i in result.items] == ["a"]
    half = LogOddsFusion(base_rate=0.5, default_weight=0.5).fuse(legs[:1])
    assert half.items[0].log_odds == pytest.approx(0.5 * _logit(0.8), abs=1e-12)


def test_unfitted_weight_zero_leg_is_not_checked():
    class Unfitted:
        fitted = False

    LogOddsFusion(base_rate=0.5, weights={"x": 0.0}).fuse([("x", [("a", 1.0)], Unfitted())])


def test_ties_break_by_key_and_duplicates_count_once():
    result = LogOddsFusion(base_rate=0.5).fuse(
        [("lex", [("b", 0.7), ("a", 0.7), ("b", 0.99)], _Identity())]
    )
    assert [i.key for i in result.items] == ["a", "b"]
    assert result.items[1].contributions[0].score == 0.7


def test_isotonic_extremes_are_clipped_to_finite():
    result = LogOddsFusion(base_rate=0.5, clip=1e-3).fuse(
        [("lex", [("a", 1.0), ("b", 0.0)], _Identity())]
    )
    assert all(math.isfinite(i.log_odds) for i in result.items)
    assert result.items[0].log_odds == pytest.approx(_logit(1 - 1e-3), abs=1e-12)


def test_top_k_and_empty_legs():
    legs = [("lex", [("a", 0.9), ("b", 0.8), ("c", 0.7)], _Identity()), ("dense", [], _Identity())]
    assert [i.key for i in LogOddsFusion(base_rate=0.5, top_k=2).fuse(legs).items] == ["a", "b"]
    assert LogOddsFusion(base_rate=0.5).fuse([]).items == ()


def test_evidence_correlation_is_reported():
    keys = [f"d{i}" for i in range(20)]
    p = np.linspace(0.1, 0.9, 20)
    result = LogOddsFusion(base_rate=0.5).fuse(
        [
            ("lex", list(zip(keys, p)), _Identity()),
            ("dense", list(zip(keys, p * 0.9 + 0.05)), _Identity()),
            ("other", [("d0", 0.5)], _Identity()),
        ]
    )
    assert result.shared_counts[(0, 1)] == 20
    assert result.evidence_correlation[(0, 1)] > 0.99
    assert result.evidence_correlation[(0, 2)] is None  # one shared item


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"base_rate": 0.0}, ValueError),
        ({"base_rate": 1.0}, ValueError),
        ({"base_rate": True}, TypeError),
        ({"base_rate": 0.5, "clip": 0.5}, ValueError),
        ({"base_rate": 0.5, "clip": 0}, ValueError),
        ({"base_rate": 0.5, "weights": {"a": -1}}, ValueError),
        ({"base_rate": 0.5, "weights": {1: 1.0}}, TypeError),
        ({"base_rate": 0.5, "top_k": -1}, ValueError),
    ],
)
def test_policy_refusals(kwargs, error):
    with pytest.raises(error):
        LogOddsFusion(**kwargs)


def test_fuse_refusals():
    fusion = LogOddsFusion(base_rate=0.5)

    class Unfitted:
        fitted = False
        prevalence_ = 0.5

        def predict_proba(self, scores):
            return scores

    class NoPrevalence:
        fitted = True

        def predict_proba(self, scores):
            return scores

    class Bad(_Identity):
        def predict_proba(self, scores):
            return [1.5 for _ in scores]

    with pytest.raises(ValueError, match="not fitted"):
        fusion.fuse([("lex", [("a", 0.5)], Unfitted())])
    with pytest.raises(TypeError, match="prevalence_"):
        fusion.fuse([("lex", [("a", 0.5)], NoPrevalence())])
    with pytest.raises(ValueError, match="not a probability"):
        fusion.fuse([("lex", [("a", 0.5)], Bad())])
    with pytest.raises(TypeError):
        fusion.fuse([("lex", [("a", 0.5)])])
    with pytest.raises(ValueError, match="empty"):
        fusion.fuse([("lex", [("", 0.5)], _Identity())])
    with pytest.raises(ValueError, match="not finite"):
        fusion.fuse([("lex", [("a", float("nan"))], _Identity())])
    with pytest.raises(TypeError, match="must be a str"):
        fusion.fuse([("lex", [(3, 0.5)], _Identity())])


def test_fused_probability_beats_each_leg_on_independent_evidence():
    """Calibrated, conditionally independent legs: fusion should help, held out."""
    rank_bm25 = pytest.importorskip("scikitplot.rank_bm25")
    rng = np.random.default_rng(0)

    def draw(n):
        y = (rng.random(n) < 0.3).astype(int)
        lex = rng.normal(1.5 * y, 1.0)  # independent given y
        dense = rng.normal(1.2 * y, 1.0)
        return y, lex, dense

    y, lex, dense = draw(4000)
    qids = [str(i // 20) for i in range(4000)]
    cal_lex = rank_bm25.ScoreCalibrator().fit(lex, y, query_ids=qids, population="pool")
    cal_dense = rank_bm25.ScoreCalibrator().fit(dense, y, query_ids=qids, population="pool")
    yt, lext, denset = draw(4000)
    keys = [f"d{i}" for i in range(4000)]
    result = LogOddsFusion(base_rate=float(y.mean())).fuse(
        [
            ("lex", list(zip(keys, lext.tolist())), cal_lex),
            ("dense", list(zip(keys, denset.tolist())), cal_dense),
        ]
    )
    by_key = {item.key: item.probability for item in result.items}
    fused = np.array([by_key[k] for k in keys])
    loss_fused = binary_log_loss(yt, fused)
    assert loss_fused < binary_log_loss(yt, cal_lex.predict_proba(lext))
    assert loss_fused < binary_log_loss(yt, cal_dense.predict_proba(denset))
    assert expected_calibration_error(yt, fused) < 0.03
    # Marginal correlation is positive although the legs are independent
    # given relevance: both rise with it. It flags duplicates, not dependence.
    assert 0.0 < result.evidence_correlation[(0, 1)] < 0.5
