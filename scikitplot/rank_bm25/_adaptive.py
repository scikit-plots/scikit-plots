# scikitplot/rank_bm25/_adaptive.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Term-specific ``k1`` estimators: information gain and log-logistic.

Both estimators read only the collection: no training data, no queries. They
are pure functions of the term-major postings and the document lengths, so a
built index and a reloaded one derive bit-identical values.

Notes
-----
**User-focused.** :class:`~scikitplot.rank_bm25.BM25Adpt` and
:class:`~scikitplot.rank_bm25.BM25T` call these when an index is built and
again when it is loaded. ``k1`` is not something you set for them; ``b`` is.

**Developer-focused.** Notation, shared by both estimators:

* ``c'(t, d) = tf(t, d) / (1 - b + b * dl(d) / avgdl)``, the length-normalised
  count of term ``t`` in document ``d``;
* ``df_t`` the number of documents holding ``t``; ``N`` the number of
  documents.

*Information gain.* ``df_0 = N``, ``df_1 = df_t`` and, for ``r >= 2``,
``df_r = |{d : c'(t, d) >= r - 0.5}|``. The gain of the ``r``-th occurrence is
``G_r = log2((df_{r+1} + 0.5) / (df_r + 1)) - log2((df_t + 0.5) / (N + 1))``.
``k1`` is the least-squares fit of ``(k + 1) r / (k + r)`` to ``G_r / G_1``
over ``r = 0..R``, and ``G_1`` replaces the idf. ``R`` is the first ``r >= 1``
with ``G_r > G_{r+1}``. The definition leaves three points open, which are
settled here and are part of the recipe version:

1. ``R`` is capped at ``min(r_top, MAX_OCCURRENCE)``, ``r_top`` being the
   largest ``r`` any document reaches: beyond ``r_top`` every ``df_r`` is 0
   and every ``G_r`` the same value, so "until the gain decreases" would
   never stop; :data:`MAX_OCCURRENCE` bounds the work and memory per term
   whatever a document's length (a single huge count would otherwise size
   the per-term arrays).
2. ``k1`` is undefined when ``R < 2`` (no repeated occurrence to fit) or
   ``G_1 <= 0`` (the ratio has no meaning); it is then ``k1_fallback``.
3. The minimiser is fixed: the objective on a log-spaced grid of
   :data:`GRID_POINTS` points over ``[k1_min, k1_max]``, then a golden-section
   refinement between the grid neighbours of the best point, keeping the
   refinement only if it is strictly better. The objective has no closed
   form, and nothing proves it unimodal, so the grid is what makes the
   result a property of the data rather than of a starting point.

*Log-logistic.* ``k1`` solves ``g(k) = s_t`` with
``g(k) = k ln(k) / (k - 1)`` (``g(1) = 1``) and ``s_t`` the mean of
``ln(1 + c')`` over the documents holding ``t``. ``g`` is strictly increasing
on ``(0, inf)`` (its derivative is ``(k - 1 - ln k) / (k - 1)**2 > 0``) with
range ``(0, inf)``, and ``s_t > 0``, so the root exists and is unique. It is
found by bisection on ``ln k`` until the bracket cannot shrink, which needs no
starting guess and returns the same double on every run.
"""

from __future__ import annotations

import math

import numpy as np

__all__: list[str] = [
    "GRID_POINTS",
    "MAX_OCCURRENCE",
    "estimate_information_gain_k1",
    "estimate_log_logistic_k1",
    "normalised_counts",
]

#: Grid resolution of the information-gain minimiser. Part of the recipe: a
#: change bumps the recipe version.
GRID_POINTS: int = 257

#: Largest ``r`` the information-gain fit considers. Part of the recipe.
MAX_OCCURRENCE: int = 255

#: Bound on ``|ln k|`` while bracketing; ``exp(700)`` is still finite.
_LOG_BOUND = 700.0

#: Iteration cap for the bisection and the golden section. Both stop earlier,
#: as soon as the interval cannot shrink in double precision.
_MAX_ITERATIONS = 400

_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0

#: The first repeated occurrence: ``df_r`` for ``r >= _FIRST_REPEAT`` is
#: counted from ``c'``, and a fit needs ``R >= _FIRST_REPEAT``.
_FIRST_REPEAT = 2


def normalised_counts(rows, tfs, doc_len, avgdl, b):
    """
    Return ``c'`` for every stored (row, term) pair.

    Parameters
    ----------
    rows : numpy.ndarray of int
        Row of each stored pair.
    tfs : numpy.ndarray of int
        Term frequency of each stored pair.
    doc_len : array-like of int
        Every document's length, in row order.
    avgdl : float
        Average document length.
    b : float
        Length-normalisation strength, in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray of float64
        ``tf / (1 - b + b * dl / avgdl)`` per stored pair.

    Notes
    -----
    **Developer.** A stored pair has ``tf >= 1``, so its document has length at
    least 1 and the normaliser is positive for every ``b`` in ``[0, 1]``.
    """
    lengths = np.asarray(doc_len, dtype=np.int64)[np.asarray(rows, dtype=np.int64)]
    norm = 1 - b + b * lengths / avgdl
    return np.asarray(tfs, dtype=np.float64) / norm


def _require_postings(indptr, c):
    """Check that every term column holds at least one pair."""
    df = np.diff(np.asarray(indptr, dtype=np.int64))
    if df.size and int(df.min()) <= 0:
        raise ValueError(
            "every vocabulary term must occur in at least one document; a term "
            "with no postings has no collection statistics to estimate k1 from."
        )
    if int(indptr[-1]) != c.size:
        raise ValueError("indptr and the per-pair counts disagree in length.")
    return df


def _log_logistic_g(k):
    """``k ln(k) / (k - 1)``, with its limit 1 at ``k == 1``."""
    d = k - 1.0
    zero = d == 0.0
    return np.where(zero, 1.0, k * np.log1p(d) / np.where(zero, 1.0, d))


def estimate_log_logistic_k1(indptr, c):
    """
    Estimate each term's ``k1`` by the log-logistic moment equation.

    Parameters
    ----------
    indptr : numpy.ndarray of int
        Term-major column offsets.
    c : numpy.ndarray of float64
        ``c'`` per stored pair, in the same order as the postings.

    Returns
    -------
    numpy.ndarray of float64
        One positive ``k1`` per term, the unique root of ``g(k) = s_t``.

    Raises
    ------
    ValueError
        If a term has no postings, or the arrays disagree.

    Notes
    -----
    **Developer.** Every term is solved at once: the bracket ``[lo, hi]`` in
    ``ln k`` keeps ``g(e**lo) < s_t <= g(e**hi)``, and the upper end is
    returned, the smallest ``ln k`` in the last bracket meeting the target.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    c = np.asarray(c, dtype=np.float64)
    df = _require_postings(indptr, c)
    if df.size == 0:
        return np.zeros(0)
    target = np.add.reduceat(np.log1p(c), indptr[:-1]) / df
    lo = np.full(df.size, -1.0)
    hi = np.full(df.size, 1.0)
    while True:
        low = _log_logistic_g(np.exp(lo)) >= target
        if not low.any():
            break
        if float(lo.min()) <= -_LOG_BOUND:
            raise RuntimeError("log-logistic k1: target too small to bracket.")
        lo = np.where(low, lo * 2.0, lo)
    while True:
        high = _log_logistic_g(np.exp(hi)) < target
        if not high.any():
            break
        if float(hi.max()) >= _LOG_BOUND:
            raise RuntimeError("log-logistic k1: target too large to bracket.")
        hi = np.where(high, hi * 2.0, hi)
    for _ in range(_MAX_ITERATIONS):
        mid = 0.5 * (lo + hi)
        active = (mid > lo) & (mid < hi)
        if not active.any():
            break
        below = _log_logistic_g(np.exp(mid)) < target
        lo = np.where(active & below, mid, lo)
        hi = np.where(active & ~below, mid, hi)
    return np.exp(hi)


def _fit_objective(u, y, r):
    """Sum of squared residuals at ``k = exp(u)``; ``u`` may be an array."""
    k = np.exp(np.asarray(u, dtype=np.float64))[..., None]
    return ((y - (k + 1.0) * r / (k + r)) ** 2).sum(axis=-1)


def _least_squares_k1(y, r, k1_min, k1_max):
    """Minimise the information-gain objective, by the fixed procedure."""
    u_min, u_max = math.log(k1_min), math.log(k1_max)
    grid = np.linspace(u_min, u_max, GRID_POINTS)
    values = _fit_objective(grid, y, r)
    best = int(np.argmin(values))
    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, GRID_POINTS - 1)]
    a, d = left, right
    b_ = d - _INV_PHI * (d - a)
    c_ = a + _INV_PHI * (d - a)
    fb, fc = float(_fit_objective(b_, y, r)), float(_fit_objective(c_, y, r))
    for _ in range(_MAX_ITERATIONS):
        if fb <= fc:
            d, c_, fc = c_, b_, fb
            b_ = d - _INV_PHI * (d - a)
            fb = float(_fit_objective(b_, y, r))
        else:
            a, b_, fb = b_, c_, fc
            c_ = a + _INV_PHI * (d - a)
            fc = float(_fit_objective(c_, y, r))
        if not (a < b_ < d and a < c_ < d):
            break
    refined, f_refined = (b_, fb) if fb <= fc else (c_, fc)
    if f_refined < float(values[best]):
        return min(max(math.exp(refined), k1_min), k1_max)
    return min(max(math.exp(float(grid[best])), k1_min), k1_max)


def estimate_information_gain_k1(
    indptr,
    c,
    n_docs,
    *,
    k1_min,
    k1_max,
    k1_fallback,
):
    """
    Estimate each term's ``k1`` and weight by information gain.

    Parameters
    ----------
    indptr : numpy.ndarray of int
        Term-major column offsets.
    c : numpy.ndarray of float64
        ``c'`` per stored pair, in the same order as the postings.
    n_docs : int
        Number of documents ``N``.
    k1_min, k1_max : float
        The interval the fit searches, ``0 < k1_min < k1_max``.
    k1_fallback : float
        ``k1`` of a term whose fit is undefined.

    Returns
    -------
    k1 : numpy.ndarray of float64
        One ``k1`` per term.
    weight : numpy.ndarray of float64
        ``G_1`` per term, used in place of the idf.

    Raises
    ------
    ValueError
        If a term has no postings, or the arrays disagree.

    Notes
    -----
    **Developer.** ``G_1`` is computed for every term in one vectorised pass.
    The fit runs only for terms that some document holds at ``c' >= 1.5``
    (``df_2 > 0``); every other term has ``R < 2`` and takes the fallback
    without a fit, which is most of a natural vocabulary.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    c = np.asarray(c, dtype=np.float64)
    df = _require_postings(indptr, c)
    n_terms = df.size
    if n_terms == 0:
        return np.zeros(0), np.zeros(0)
    # q = min(the largest integer r with c' >= r - 0.5, MAX_OCCURRENCE + 1):
    # the pair counts towards df_r for every 2 <= r <= q. floor(c + 0.5) is
    # exact for every r >= 2 (c >= 1.5) up to c < 2**51; below 1.5 a pair can
    # land in bucket 1 instead of 0, which is unused because df_1 is set to
    # df_t. For 1.5 <= c the real sum
    # c + 0.5 is a multiple of ulp(c). Within c's binade the sum is representable. Past it the
    # sum lies in [2**j, c + 0.5) with c < 2**j, and that interval holds no
    # integer above 2**j, so rounding cannot carry the sum across an integer.
    # Above the cap only the clamp matters.
    q = np.minimum(np.floor(c + 0.5), MAX_OCCURRENCE + 1.0).astype(np.int64)
    df_f = df.astype(np.float64)
    base = np.log2((df_f + 0.5) / (n_docs + 1.0))
    df2 = np.add.reduceat((q >= _FIRST_REPEAT).astype(np.int64), indptr[:-1])
    # The weight is computed per term with math.log2 (the platform libm, as
    # every other recipe's idf), so it does not depend on the NumPy build.
    weight = np.array(
        [
            math.log2((d2 + 0.5) / (d + 1.0)) - math.log2((d + 0.5) / (n_docs + 1.0))
            for d, d2 in zip(df.tolist(), df2.tolist())
        ],
        dtype=np.float64,
    )
    k1 = np.full(n_terms, float(k1_fallback))
    for t in np.flatnonzero((df2 > 0) & (weight > 0)).tolist():
        start, stop = int(indptr[t]), int(indptr[t + 1])
        top = min(int(q[start:stop].max()), MAX_OCCURRENCE)
        # dfr[r] for r = 0..top+1: the count of pairs with c' >= r - 0.5 (q is
        # clamped at MAX_OCCURRENCE + 1, which keeps dfr[top + 1] exact).
        counts = np.bincount(q[start:stop], minlength=top + 2)
        dfr = np.cumsum(counts[::-1])[::-1].astype(np.float64)
        dfr[0] = n_docs
        dfr[1] = df_f[t]
        r = np.arange(top + 1, dtype=np.float64)
        gain = np.log2((dfr[1 : top + 2] + 0.5) / (dfr[: top + 1] + 1.0)) - base[t]
        # One G_1 for the gate, the divisor and the persisted weight.
        gain[1] = weight[t]
        drops = np.flatnonzero(gain[1:top] > gain[2 : top + 1])
        stop_r = int(drops[0]) + 1 if drops.size else top
        if stop_r < _FIRST_REPEAT:
            continue
        y = gain[2 : stop_r + 1] / gain[1]
        k1[t] = _least_squares_k1(y, r[2 : stop_r + 1], k1_min, k1_max)
    return k1, weight
