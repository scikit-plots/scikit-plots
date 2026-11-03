"""
ScoreCalibrator: agreement with reference fits, leakage guard, prior shift.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from .._calibration import ScoreCalibrator
from .._rank_bm25 import BM25Okapi


def _data(seed, n=400, slope=1.5, shift=-3.0):
    rng = np.random.default_rng(seed)
    scores = rng.gamma(2.0, 2.0, size=n)
    p = 1.0 / (1.0 + np.exp(-(slope * scores + shift)))
    labels = (rng.random(n) < p).astype(int)
    labels[:2] = [0, 1]
    qids = [f"q{i // 10}" for i in range(n)]  # 10 pairs per query
    return scores, labels, qids


def _fit(method, seed=0, **kw):
    s, y, q = _data(seed)
    return ScoreCalibrator(method, **kw).fit(s, y, query_ids=q, population="test"), s, y


@pytest.mark.parametrize("seed", range(6))
def test_sigmoid_matches_scikit_learn(seed):
    calibration = pytest.importorskip("sklearn.calibration")
    cal, s, y = _fit("sigmoid", seed)
    a, b = calibration._sigmoid_calibration(s, y)  # p = 1 / (1 + exp(a s + b))
    assert cal.slope_ == pytest.approx(-a, rel=1e-5)
    assert cal.intercept_ == pytest.approx(-b, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_isotonic_matches_scikit_learn(seed):
    isotonic = pytest.importorskip("sklearn.isotonic")
    s, y, q = _data(seed)
    s = np.round(s, 1)  # many tied scores
    cal = ScoreCalibrator("isotonic").fit(s, y, query_ids=q, population="test")
    ref = isotonic.IsotonicRegression(out_of_bounds="clip").fit(s, y)
    probe = np.concatenate([s, [-5.0, 0.0, 100.0], np.linspace(0, 20, 50)])
    np.testing.assert_allclose(cal.predict_proba(probe), ref.predict(probe), atol=1e-12)


def test_sigmoid_recovers_known_parameters():
    s, y, q = _data(0, n=20000, slope=1.5, shift=-3.0)
    cal = ScoreCalibrator("sigmoid", platt_targets=False).fit(
        s, y, query_ids=q, population="test"
    )
    assert cal.slope_ == pytest.approx(1.5, rel=0.08)
    assert cal.intercept_ == pytest.approx(-3.0, rel=0.08)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_monotone_so_ranking_is_kept(method):
    cal, s, _ = _fit(method)
    probe = np.sort(np.random.default_rng(1).uniform(-1, 30, 500))
    p = cal.predict_proba(probe)
    assert np.all(np.diff(p) >= 0)
    if method == "sigmoid":
        assert np.all(np.diff(p[probe < 8]) > 0)  # strictly, where not saturated


def test_sigmoid_keeps_bm25_ranking_exactly():
    corpus = [["a", "b"], ["a"], ["b", "c", "c"], ["c"], ["a", "c"]]
    index = BM25Okapi(corpus, doc_ids=list("vwxyz"))
    scores = index.get_scores(["a", "c"])
    cal = ScoreCalibrator("sigmoid").fit(
        [0.1, 0.2, 1.0, 1.2, 0.4, 0.9],
        [0, 0, 1, 1, 0, 1],
        query_ids=list("aabbcc"),
        population="toy",
    )
    probs = cal.predict_proba(scores)
    assert np.argsort(-probs, kind="stable").tolist() == np.argsort(
        -scores, kind="stable"
    ).tolist()


def test_require_held_out():
    cal, _, _ = _fit("sigmoid")
    cal.require_held_out(["new-1", "new-2"])
    with pytest.raises(ValueError, match="2 evaluation queries were used to fit"):
        cal.require_held_out(["q0", "q3", "fresh"])


def test_query_ids_are_stored_as_digests_only():
    cal, _, _ = _fit("sigmoid")
    text = json.dumps(cal.to_dict())
    assert '"q0"' not in text and "q0" not in cal.to_dict()["query_digests"]


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_json_round_trip_predicts_identically(method):
    cal, s, _ = _fit(method)
    again = ScoreCalibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert np.array_equal(again.predict_proba(s), cal.predict_proba(s))
    again.require_held_out(["unseen"])
    with pytest.raises(ValueError):
        again.require_held_out(["q1"])


def test_shift_prior_is_the_bayes_correction():
    cal, s, _ = _fit("sigmoid")
    shifted = cal.shift_prior(0.05, population="corpus")
    p, q = cal.predict_proba(s), shifted.predict_proba(s)
    logit = lambda v: np.log(v) - np.log1p(-v)  # noqa: E731
    offset = math.log(0.05 / 0.95) - math.log(cal.prevalence_ / (1 - cal.prevalence_))
    # Away from saturation, where recomputing a logit from p is accurate.
    ok = (p > 1e-6) & (p < 1 - 1e-6) & (q > 1e-6) & (q < 1 - 1e-6)
    assert ok.sum() > 300
    np.testing.assert_allclose(logit(q[ok]) - logit(p[ok]), offset, rtol=1e-9, atol=1e-9)
    assert shifted.population == "corpus" and shifted.prevalence_ == 0.05
    assert cal.population == "test"  # original unchanged
    round_trip = ScoreCalibrator.from_dict(shifted.to_dict())
    assert np.array_equal(round_trip.predict_proba(s), q)


def test_shift_prior_keeps_isotonic_extremes():
    cal, s, _ = _fit("isotonic")
    shifted = cal.shift_prior(0.3, population="other")
    base, moved = cal.predict_proba(s), shifted.predict_proba(s)
    assert np.array_equal(moved[base == 0.0], base[base == 0.0])
    assert np.array_equal(moved[base == 1.0], base[base == 1.0])


def test_separable_labels():
    s = np.array([0.1, 0.2, 0.3, 2.0, 2.1, 2.2])
    y = np.array([0, 0, 0, 1, 1, 1])
    q = list("abcdef")
    cal = ScoreCalibrator("sigmoid").fit(s, y, query_ids=q, population="t")
    assert np.isfinite(cal.slope_) and cal.slope_ > 0
    with pytest.raises(ValueError, match="no finite solution"):
        ScoreCalibrator("sigmoid", platt_targets=False).fit(s, y, query_ids=q, population="t")


def test_constant_scores_give_flat_prevalence():
    cal = ScoreCalibrator("sigmoid").fit(
        [1.0] * 4, [0, 1, 1, 0], query_ids=list("abcd"), population="t"
    )
    assert cal.slope_ == 0.0
    assert cal.predict_proba([1.0, 5.0]) == pytest.approx([0.5, 0.5], abs=1e-9)


def test_reversed_scores_are_refused():
    s, y, q = _data(0)
    with pytest.raises(ValueError, match="reverse the ranking"):
        ScoreCalibrator("sigmoid").fit(-s, y, query_ids=q, population="t")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"scores": [1, 2], "labels": [0, 1], "query_ids": ["a"]}, ValueError),
        ({"scores": [1, 2], "labels": [0, 2], "query_ids": ["a", "b"]}, ValueError),
        ({"scores": [1, 2], "labels": [1, 1], "query_ids": ["a", "b"]}, ValueError),
        ({"scores": [1, float("inf")], "labels": [0, 1], "query_ids": ["a", "b"]}, ValueError),
        ({"scores": [[1, 2]], "labels": [[0, 1]], "query_ids": ["a", "b"]}, ValueError),
        ({"scores": ["x", "y"], "labels": [0, 1], "query_ids": ["a", "b"]}, TypeError),
        ({"scores": [1, 2], "labels": [0, 1], "query_ids": ["a", "b"], "population": " "}, ValueError),
    ],
)
def test_fit_refusals(kwargs, error):
    kwargs = {"population": "t", **kwargs}
    with pytest.raises(error):
        ScoreCalibrator("sigmoid").fit(
            kwargs.pop("scores"), kwargs.pop("labels"), **kwargs
        )


def test_contract_refusals():
    with pytest.raises(ValueError, match="method must be one of"):
        ScoreCalibrator("beta")
    with pytest.raises(RuntimeError, match="not fitted"):
        ScoreCalibrator().predict_proba([1.0])
    cal, _, _ = _fit("sigmoid")
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            cal.shift_prior(bad, population="x")
    with pytest.raises(ValueError, match="not a 'score-calibrator/1'"):
        ScoreCalibrator.from_dict({"schema": "other"})
    state = cal.to_dict()
    del state["slope"]
    with pytest.raises(ValueError, match="missing 'slope'"):
        ScoreCalibrator.from_dict(state)
