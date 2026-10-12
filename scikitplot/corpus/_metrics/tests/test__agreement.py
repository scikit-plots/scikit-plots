# scikitplot/corpus/_metrics/tests/test__agreement.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Agreement statistics: published values, reference implementations, edge cases."""

from __future__ import annotations

import math
import random
import warnings

import numpy as np
import pytest

from .._agreement import cohen_kappa, fleiss_kappa, krippendorff_alpha
from .._inputs import UndefinedMetricWarning

#: Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 coders,
#: 12 units, with gaps. Published alphas: nominal 0.743, ordinal 0.815,
#: interval 0.849, ratio 0.797.
TEXTBOOK = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
]


@pytest.mark.parametrize(("level", "published"), [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849), ("ratio", 0.797)])
def test_krippendorff_textbook(level, published):
    assert round(krippendorff_alpha(TEXTBOOK, level=level), 3) == published


def test_krippendorff_nan_marks_missing():
    as_nan = [[math.nan if v is None else v for v in row] for row in TEXTBOOK]
    assert krippendorff_alpha(as_nan, level="interval") == krippendorff_alpha(TEXTBOOK, level="interval")


def test_krippendorff_nominal_labels_need_no_order():
    assert krippendorff_alpha([["x", "y", "x"], ["x", "y", None]]) == 1.0


def test_cohen_hand_computed():
    # confusion [[1, 0], [1, 1]]: po = 2/3, pe = (1*2 + 2*1)/9 = 4/9, kappa = (2/3 - 4/9) / (5/9) = 0.4
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx(0.4, rel=1e-15)


def test_cohen_label_order_matters_for_weights():
    a, b = ["lo", "mid", "hi", "mid"], ["mid", "mid", "hi", "lo"]
    ordered = cohen_kappa(a, b, labels=["lo", "mid", "hi"], weights="linear")
    alphabetical = cohen_kappa(a, b, weights="linear")  # hi < lo < mid
    assert ordered != alphabetical


def test_fleiss_hand_computed():
    # items x raters; per-item agreement (1/3, 1, 0, 1) -> 7/12; shares (3/12, 5/12, 4/12) -> chance 50/144
    expected = (7 / 12 - 50 / 144) / (1 - 50 / 144)
    assert fleiss_kappa([[1, 1, 2], [2, 2, 2], [1, 2, 3], [3, 3, 3]]) == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: cohen_kappa([1, 1], [1, 1]), "single label"),
        (lambda: fleiss_kappa([[1, 1], [1, 1]]), "same label"),
        (lambda: krippendorff_alpha([[1, 1], [1, 1]]), "only one value"),
        (lambda: krippendorff_alpha([[1, None], [None, 2]]), "no unit has two ratings"),
    ],
)
def test_undefined_is_nan_with_a_warning(call, match):
    with pytest.warns(UndefinedMetricWarning, match=match):
        assert math.isnan(call())


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: cohen_kappa([1, 2], [1]), ValueError, "different numbers"),
        (lambda: cohen_kappa([1, None], [1, 2]), ValueError, "missing"),
        (lambda: cohen_kappa([1, 2], [1, 2], weights="cubic"), ValueError, "weights"),
        (lambda: cohen_kappa([1, 2], [1, 3], labels=[1, 2]), ValueError, "outside labels"),
        (lambda: cohen_kappa([1, 2], [1, 2], labels=[1, 1, 2]), ValueError, "unique"),
        (lambda: cohen_kappa("ab", "ab"), TypeError, "sequence"),
        (lambda: cohen_kappa([1, "a"], [1, "a"]), TypeError, "labels="),
        (lambda: fleiss_kappa([[1], [2]]), ValueError, "two ratings"),
        (lambda: fleiss_kappa([[1, None], [2, 2]]), ValueError, "krippendorff_alpha"),
        (lambda: fleiss_kappa([[1, 2], [2]]), ValueError, "rectangular"),
        (lambda: fleiss_kappa([]), ValueError, "empty"),
        (lambda: krippendorff_alpha([[1, 2]], level="ranked"), ValueError, "level"),
        (lambda: krippendorff_alpha([["a", "b"], ["a", "b"]], level="interval"), ValueError, "real-number"),
        (lambda: krippendorff_alpha([[True, False], [True, True]], level="interval"), ValueError, "real-number"),
        (lambda: krippendorff_alpha([[-1, 2], [1, 2]], level="ratio"), ValueError, "non-negative"),
    ],
)
def test_refusals(call, error, match):
    with pytest.raises(error, match=match):
        call()


def _random_cases(seed, n_cases):
    rng = random.Random(seed)  # noqa: S311 - seeded test data
    for _ in range(n_cases):
        n, k, r = rng.randint(2, 25), rng.randint(2, 5), rng.randint(2, 5)
        truth = [rng.randint(1, k) for _ in range(n)]
        rows = [[x if rng.random() < 0.7 else rng.randint(1, k) for _ in range(r)] for x in truth]
        yield rng, truth, rows


def test_parity_with_scikit_learn():
    metrics = pytest.importorskip("sklearn.metrics")
    for _rng, truth, rows in _random_cases(1, 300):
        other = [row[0] for row in rows]
        for weights in (None, "linear", "quadratic"):
            with _quiet():
                ref = metrics.cohen_kappa_score(truth, other, weights=weights)
                mine = cohen_kappa(truth, other, weights=weights)
            assert (math.isnan(ref) and math.isnan(mine)) or mine == pytest.approx(ref, abs=1e-12)


def test_parity_with_statsmodels():
    inter_rater = pytest.importorskip("statsmodels.stats.inter_rater")
    for _, _, rows in _random_cases(2, 300):
        table, _ = inter_rater.aggregate_raters(np.array(rows))
        with _quiet():
            ref = inter_rater.fleiss_kappa(table, method="fleiss")
            mine = fleiss_kappa(rows)
        assert (math.isnan(ref) and math.isnan(mine)) or mine == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("level", ["nominal", "ordinal", "interval", "ratio"])
def test_parity_with_krippendorff(level):
    krippendorff = pytest.importorskip("krippendorff")
    for rng, _, rows in _random_cases(3, 200):
        data = np.array(rows, dtype=float).T
        data[np.array([[rng.random() < 0.2 for _ in range(data.shape[1])] for _ in range(data.shape[0])])] = np.nan
        with _quiet():
            try:
                ref = krippendorff.alpha(reliability_data=data, level_of_measurement=level)
            except ValueError:
                ref = math.nan
            mine = krippendorff_alpha(data, level=level)
        assert (math.isnan(ref) and math.isnan(mine)) or mine == pytest.approx(ref, abs=1e-9)


class _quiet:  # noqa: N801 - used as a context manager
    """Silence warnings (the references warn on undefined values too)."""

    def __enter__(self):
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
