"""
Tests for the term-specific ``k1`` estimators (``_adaptive.py``).

Each estimator is checked against an independent, deliberately naive
re-derivation from its definition: loops instead of vectorised counting, and
an exhaustive scan instead of the fixed minimiser.

See Also
--------
scikitplot.rank_bm25._adaptive
"""

import math

import numpy as np
import pytest

from .._adaptive import (
    GRID_POINTS,
    MAX_OCCURRENCE,
    estimate_information_gain_k1,
    estimate_log_logistic_k1,
    normalised_counts,
)


def _postings(columns):
    """``(indptr, c)`` from one list of ``c'`` values per term."""
    indptr = np.zeros(len(columns) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(col) for col in columns])
    return indptr, np.concatenate([np.asarray(col, float) for col in columns])


def _random_columns(seed, n_terms=40, n_docs=200):
    rng = np.random.default_rng(seed)
    columns = []
    for _ in range(n_terms):
        df = int(rng.integers(1, n_docs // 2))
        counts = rng.geometric(rng.uniform(0.2, 0.9), size=df).astype(float)
        columns.append(counts / rng.uniform(0.4, 2.5, size=df))
    return columns, n_docs


# -- normalised counts ---------------------------------------------------------


def test_normalised_counts_follow_the_definition():
    rows = np.array([0, 2, 1])
    tfs = np.array([3, 1, 2])
    doc_len = [4, 6, 2]
    avgdl = 4.0
    c = normalised_counts(rows, tfs, doc_len, avgdl, 0.5)
    expected = [t / (1 - 0.5 + 0.5 * doc_len[r] / avgdl) for r, t in zip(rows, tfs)]
    assert c.tolist() == expected


# -- log-logistic --------------------------------------------------------------


def _g(k):
    return 1.0 if k == 1 else k * math.log(k) / (k - 1)


@pytest.mark.parametrize("seed", range(5))
def test_log_logistic_k1_solves_the_moment_equation(seed):
    columns, _ = _random_columns(seed)
    indptr, c = _postings(columns)
    k1 = estimate_log_logistic_k1(indptr, c)
    for column, k in zip(columns, k1.tolist()):
        target = sum(math.log1p(x) for x in column) / len(column)
        assert k > 0
        # The root is bracketed to adjacent doubles in ln k, so the residual is
        # at rounding level, relative to the target.
        assert abs(_g(k) - target) <= 1e-12 * max(1.0, target)


def test_log_logistic_g_is_strictly_increasing_so_the_root_is_unique():
    k = np.logspace(-8, 8, 20001)
    g = np.array([_g(x) for x in k])
    assert np.all(np.diff(g) > 0)


def test_log_logistic_k1_is_one_where_the_moment_is_one():
    # mean ln(1 + c) == 1 exactly when every c == e - 1, and g(1) == 1.
    indptr, c = _postings([[math.e - 1.0] * 4])
    (k,) = estimate_log_logistic_k1(indptr, c).tolist()
    assert abs(k - 1.0) <= 1e-12


def test_log_logistic_k1_is_deterministic():
    indptr, c = _postings(_random_columns(7)[0])
    first = estimate_log_logistic_k1(indptr, c)
    assert first.tobytes() == estimate_log_logistic_k1(indptr, c).tobytes()


def test_log_logistic_k1_of_an_extreme_term_is_still_bracketed():
    indptr, c = _postings([[1e-9], [1e9]])
    k1 = estimate_log_logistic_k1(indptr, c)
    assert np.isfinite(k1).all() and (k1 > 0).all()


# -- information gain: an independent re-derivation ----------------------------


def _naive_information_gain(column, n_docs, k1_min, k1_max, fallback):
    """The definition, by loops: df_r, G_r, R, then an exhaustive scan."""
    df = len(column)
    top = min(max(int(math.floor(x + 0.5)) for x in column), MAX_OCCURRENCE)

    def df_r(r):
        if r == 0:
            return n_docs
        if r == 1:
            return df
        return sum(1 for x in column if x >= r - 0.5)

    base = math.log2((df + 0.5) / (n_docs + 1))
    gain = [math.log2((df_r(r + 1) + 0.5) / (df_r(r) + 1)) - base for r in range(top + 1)]
    weight = gain[1]
    stop = top
    for r in range(1, top):
        if gain[r] > gain[r + 1]:
            stop = r
            break
    if stop < 2 or weight <= 0:
        return fallback, weight, None
    y = [gain[r] / weight for r in range(2, stop + 1)]
    rs = list(range(2, stop + 1))

    def objective(k):
        return sum((yy - (k + 1) * r / (k + r)) ** 2 for yy, r in zip(y, rs))

    scan = np.exp(np.linspace(math.log(k1_min), math.log(k1_max), 200001))
    values = [objective(k) for k in scan]
    return None, weight, (min(values), objective)


@pytest.mark.parametrize("seed", range(4))
def test_information_gain_matches_the_definition(seed):
    columns, n_docs = _random_columns(seed, n_terms=25)
    indptr, c = _postings(columns)
    k1, weight = estimate_information_gain_k1(
        indptr, c, n_docs, k1_min=0.001, k1_max=10.0, k1_fallback=0.001
    )
    fitted = 0
    for column, k, w in zip(columns, k1.tolist(), weight.tolist()):
        fallback, expected_weight, scan = _naive_information_gain(
            column, n_docs, 0.001, 10.0, 0.001
        )
        assert w == pytest.approx(expected_weight, rel=0, abs=1e-12)
        if scan is None:
            assert k == fallback
            continue
        fitted += 1
        best, objective = scan
        assert 0.001 <= k <= 10.0
        # Never worse than an exhaustive 200001-point scan, to rounding.
        assert objective(k) <= best + 1e-12
    assert fitted, "the random corpora must exercise the fit"


def test_a_never_repeated_term_takes_the_fallback():
    indptr, c = _postings([[1.0, 1.2, 0.9]])
    k1, weight = estimate_information_gain_k1(
        indptr, c, 50, k1_min=0.001, k1_max=10.0, k1_fallback=0.25
    )
    assert k1.tolist() == [0.25]
    expected = math.log2(0.5 / 4) - math.log2(3.5 / 51)
    assert weight[0] == pytest.approx(expected, rel=0, abs=1e-12)


def test_the_r_minus_half_threshold_is_inclusive():
    # c' == 1.5 exactly counts towards df_2; 1.5 - ulp does not.
    below = np.nextafter(1.5, 0.0)
    indptr, c = _postings([[1.5], [below]])
    _, weight = estimate_information_gain_k1(
        indptr, c, 10, k1_min=0.001, k1_max=10.0, k1_fallback=0.001
    )
    base = math.log2(1.5 / 11)
    expected = [math.log2(1.5 / 2) - base, math.log2(0.5 / 2) - base]
    assert weight.tolist() == pytest.approx(expected, rel=0, abs=1e-12)


def test_a_non_positive_first_gain_takes_the_fallback():
    # Held by every document, repeated in one: G_1 <= 0, the ratio is undefined.
    indptr, c = _postings([[1.0] * 9 + [3.0]])
    k1, weight = estimate_information_gain_k1(
        indptr, c, 10, k1_min=0.001, k1_max=10.0, k1_fallback=0.001
    )
    assert weight[0] <= 0
    assert k1.tolist() == [0.001]


def test_the_fit_is_deterministic_and_within_bounds():
    columns, n_docs = _random_columns(11)
    indptr, c = _postings(columns)
    first = estimate_information_gain_k1(
        indptr, c, n_docs, k1_min=0.05, k1_max=3.0, k1_fallback=0.5
    )
    again = estimate_information_gain_k1(
        indptr, c, n_docs, k1_min=0.05, k1_max=3.0, k1_fallback=0.5
    )
    assert first[0].tobytes() == again[0].tobytes()
    assert ((first[0] >= 0.05) & (first[0] <= 3.0)).all()


def test_the_grid_resolution_and_occurrence_cap_are_fixed():
    assert GRID_POINTS == 257
    assert MAX_OCCURRENCE == 255


def test_a_huge_normalised_count_is_bounded_by_the_occurrence_cap():
    # Without the cap the per-term arrays are sized by the largest count:
    # 1e12 entries here.
    column = [1e12, 2.0, 2.0, 3.0, 1.0, 1.0]
    indptr, c = _postings([column])
    k1, weight = estimate_information_gain_k1(
        indptr, c, 40, k1_min=0.001, k1_max=10.0, k1_fallback=0.001
    )
    fallback, expected_weight, scan = _naive_information_gain(
        column, 40, 0.001, 10.0, 0.001
    )
    assert weight[0] == pytest.approx(expected_weight, rel=0, abs=1e-12)
    if scan is None:
        assert k1[0] == fallback
    else:
        assert scan[1](k1[0]) <= scan[0] + 1e-12


# -- shared input checks -------------------------------------------------------


@pytest.mark.parametrize(
    "estimate",
    [
        estimate_log_logistic_k1,
        lambda i, c: estimate_information_gain_k1(
            i, c, 5, k1_min=0.001, k1_max=10.0, k1_fallback=0.001
        ),
    ],
)
def test_a_term_without_postings_is_refused(estimate):
    with pytest.raises(ValueError, match="at least one document"):
        estimate(np.array([0, 1, 1]), np.array([1.0]))
