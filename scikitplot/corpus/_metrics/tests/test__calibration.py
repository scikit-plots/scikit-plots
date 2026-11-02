"""
Calibration metrics: hand-checked values, scikit-learn agreement, contracts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from .._calibration import (
    binary_log_loss,
    brier_score,
    calibration_report,
    expected_calibration_error,
    reliability_table,
)

LABELS = [0, 1, 1, 0]
PROBS = [0.1, 0.9, 0.8, 0.3]


def test_hand_values():
    assert brier_score(LABELS, PROBS) == pytest.approx((0.01 + 0.01 + 0.04 + 0.09) / 4)
    want = -(math.log(0.9) + math.log(0.9) + math.log(0.8) + math.log(0.7)) / 4
    assert binary_log_loss(LABELS, PROBS) == pytest.approx(want)
    # bin [0, .5): mean .2, observed 0 ; bin [.5, 1]: mean .85, observed 1
    assert expected_calibration_error(LABELS, PROBS, n_bins=2) == pytest.approx(0.175)


def test_perfect_and_worst():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert expected_calibration_error([1, 0], [1.0, 0.0], n_bins=5) == 0.0
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0
    loss = binary_log_loss([1], [0.0])
    assert math.isfinite(loss) and loss > 30  # clipped, not infinite


def test_bool_labels_equal_int_labels():
    assert brier_score([True, False], [0.7, 0.2]) == brier_score([1, 0], [0.7, 0.2])


def test_edges_belong_to_lower_bin_and_one_to_last():
    table = reliability_table([0, 1, 1], [0.5, 1.0, 0.0], n_bins=2)
    assert table["count"] == [2, 1]  # 0.5 and 0.0 low, 1.0 high


def test_empty_bins_are_kept_as_nan():
    table = reliability_table([0, 1], [0.05, 0.95], n_bins=4)
    assert table["count"] == [1, 0, 0, 1]
    assert math.isnan(table["mean_predicted"][1])
    assert len(table["edges"]) == 5


def test_report_contents():
    report = calibration_report(LABELS, PROBS, n_bins=2)
    assert report["n_samples"] == 4
    assert report["prevalence"] == 0.5
    assert report["reliability"]["strategy"] == "uniform"
    assert report["expected_calibration_error"] == pytest.approx(0.175)


@pytest.mark.parametrize(
    "y_true, y_prob, error",
    [
        ([0, 1], [0.5], ValueError),
        ([], [], ValueError),
        ([0, 2], [0.1, 0.2], ValueError),
        ([0, 0.5], [0.1, 0.2], ValueError),
        ([0, 1], [0.1, 1.2], ValueError),
        ([0, 1], [-0.1, 0.2], ValueError),
        ([0, 1], [0.1, float("nan")], ValueError),
        ([0, 1], ["a", "b"], TypeError),
        ([[0, 1], [1, 0]], [[0.2, 0.8], [0.9, 0.1]], ValueError),
        (1, 0.5, ValueError),
    ],
)
def test_input_refusals(y_true, y_prob, error):
    with pytest.raises(error):
        brier_score(y_true, y_prob)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_bins": 0}, ValueError),
        ({"n_bins": 2.0}, TypeError),
        ({"n_bins": True}, TypeError),
        ({"strategy": "kmeans"}, ValueError),
    ],
)
def test_binning_refusals(kwargs, error):
    with pytest.raises(error):
        reliability_table(LABELS, PROBS, **kwargs)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_matches_scikit_learn(seed, strategy):
    metrics = pytest.importorskip("sklearn.metrics")
    calibration = pytest.importorskip("sklearn.calibration")
    rng = np.random.default_rng(seed)
    probs = rng.random(200)
    if seed % 2:
        probs = np.round(probs, 1)  # values exactly on bin edges
    labels = (rng.random(200) < probs).astype(int)
    labels[:2] = [0, 1]
    assert brier_score(labels, probs) == pytest.approx(
        metrics.brier_score_loss(labels, probs), abs=1e-15
    )
    assert binary_log_loss(labels, probs) == pytest.approx(
        metrics.log_loss(labels, probs, labels=[0, 1]), rel=1e-12
    )
    table = reliability_table(labels, probs, n_bins=7, strategy=strategy)
    prob_true, prob_pred = calibration.calibration_curve(
        labels, probs, n_bins=7, strategy=strategy
    )
    used = np.asarray(table["count"]) > 0
    np.testing.assert_allclose(np.asarray(table["observed_frequency"])[used], prob_true)
    np.testing.assert_allclose(np.asarray(table["mean_predicted"])[used], prob_pred)
