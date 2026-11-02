# scikitplot/corpus/_metrics/_calibration.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Calibration of relevance probabilities: Brier score, log loss, reliability, ECE.

Use them to ask whether a number presented as ``P(relevant)`` behaves like
one: among hits given 0.8, are about 80 % relevant? Ranking metrics cannot
answer that -- any monotone rescaling of scores leaves a ranking unchanged --
and a threshold or a fusion of probabilities is only meaningful once it is
answered.

==================================  ==========================================
function                            what it measures
==================================  ==========================================
:func:`brier_score`                 mean squared error of the probabilities
                                    (lower is better; 0 is perfect)
:func:`binary_log_loss`             mean negative log-likelihood (lower is
                                    better; punishes confident mistakes)
:func:`reliability_table`           per-bin mean prediction against observed
                                    frequency, with bin counts
:func:`expected_calibration_error`  count-weighted mean gap between the two
                                    (0 is perfectly calibrated)
:func:`calibration_report`          all of the above, with sample size,
                                    prevalence and the binning used
==================================  ==========================================

Notes
-----
**User note.** Every function takes the scikit-learn order
``(y_true, y_prob)``: ``y_true`` is 0/1 (or ``bool``) relevance labels and
``y_prob`` the probabilities being judged, one per (query, document) pair.
Report the binning with every ECE: the value depends on ``n_bins`` and
``strategy``, and on which population was labelled (all documents, lexical
candidates, the rerank pool...), so an ECE without them cannot be compared.

**Developer note.** NumPy only; scikit-learn is not needed. The results equal
scikit-learn's ``brier_score_loss``, ``log_loss`` and ``calibration_curve``
(checked by the tests where scikit-learn is installed): the same probability
clipping for log loss (machine epsilon of float64) and the same bin
assignment (``searchsorted`` on the interior edges, so a value on an edge
belongs to the lower bin and 1.0 to the last).

References
----------
.. [1] Brier, G. W. (1950). "Verification of forecasts expressed in terms of
   probability". Monthly Weather Review, 78(1), 1-3.
.. [2] Naeini, M. P., Cooper, G. F. and Hauskrecht, M. (2015). "Obtaining
   well calibrated probabilities using Bayesian binning". AAAI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__: list[str] = [
    "CALIBRATION_STRATEGIES",
    "binary_log_loss",
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "reliability_table",
]

#: Accepted ``strategy`` values for binning.
CALIBRATION_STRATEGIES: tuple[str, ...] = ("uniform", "quantile")

_EPS = float(np.finfo(np.float64).eps)


def _check_inputs(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate labels and probabilities, and return them as float64 arrays.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels: 0/1 or ``bool``.
    y_prob : array-like of shape (n_samples,)
        Probabilities in ``[0, 1]``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(labels, probabilities)``, both float64 and one-dimensional.

    Raises
    ------
    ValueError
        For different lengths, no samples, a label other than 0/1, a
        non-finite probability or one outside ``[0, 1]``.
    TypeError
        For non-numeric input.

    Notes
    -----
    **Developer.** Labels are compared by value after conversion, so ``True``
    and ``1.0`` are both relevant; ``2`` or ``0.5`` is refused rather than
    thresholded, because a graded label silently binarised changes what the
    metric means.
    """
    try:
        labels = np.asarray(y_true, dtype=np.float64)
        probs = np.asarray(y_prob, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"y_true and y_prob must be numeric: {exc}") from exc
    # Refused, not flattened: a (n_samples, n_classes) probability matrix
    # flattened beside a matching label matrix would be scored as n*k
    # unrelated pairs without any error.
    for name, array in (("y_true", labels), ("y_prob", probs)):
        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional, got shape {array.shape}; pass "
                "one label and one probability of relevance per pair."
            )
    if labels.size != probs.size:
        raise ValueError(
            f"y_true and y_prob have different lengths: {labels.size} and {probs.size}."
        )
    if labels.size == 0:
        raise ValueError("calibration needs at least one sample; got none.")
    bad = ~np.isin(labels, (0.0, 1.0))
    if bad.any():
        raise ValueError(
            "y_true must be binary (0/1 or bool); found "
            f"{labels[bad][0]!r}. Binarise graded relevance explicitly first."
        )
    if not np.isfinite(probs).all():
        raise ValueError("y_prob contains a non-finite value.")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(
            "y_prob must lie in [0, 1]; scores are not probabilities until "
            "they are calibrated."
        )
    return labels, probs


def _check_bins(n_bins: Any, strategy: Any) -> int:
    """Validate ``n_bins`` and ``strategy``; return ``n_bins`` as int."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError(f"n_bins must be an int, got {type(n_bins).__name__}.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}.")
    if strategy not in CALIBRATION_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {CALIBRATION_STRATEGIES}, got {strategy!r}."
        )
    return int(n_bins)


def brier_score(y_true: Any, y_prob: Any) -> float:
    """
    Mean squared difference between probabilities and binary outcomes.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary relevance labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of relevance.

    Returns
    -------
    float
        In ``[0, 1]``; lower is better.

    Raises
    ------
    ValueError, TypeError
        See :func:`reliability_table`.

    See Also
    --------
    binary_log_loss : Penalises confident mistakes more strongly.

    Examples
    --------
    >>> brier_score([1, 0, 1], [0.9, 0.2, 0.6])
    0.07
    """
    labels, probs = _check_inputs(y_true, y_prob)
    return float(np.mean((probs - labels) ** 2))


def binary_log_loss(y_true: Any, y_prob: Any) -> float:
    """
    Mean negative log-likelihood of binary outcomes under the probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary relevance labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of relevance.

    Returns
    -------
    float
        Non-negative; lower is better.

    Raises
    ------
    ValueError, TypeError
        See :func:`reliability_table`.

    Notes
    -----
    **User.** A probability of exactly 0 or 1 is clipped to
    ``[eps, 1 - eps]`` (float64 machine epsilon) so one confident mistake
    gives a large finite loss rather than infinity; this is the clipping
    scikit-learn applies.

    Examples
    --------
    >>> round(binary_log_loss([1, 0], [0.8, 0.3]), 6)
    0.289909
    """
    labels, probs = _check_inputs(y_true, y_prob)
    p = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log1p(-p)))


def _edges(probs: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Bin edges over ``[0, 1]`` (uniform) or at quantiles of ``probs``."""
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    return np.percentile(probs, np.linspace(0.0, 1.0, n_bins + 1) * 100)


def reliability_table(
    y_true: Any, y_prob: Any, *, n_bins: int = 10, strategy: str = "uniform"
) -> dict[str, Any]:
    """
    Per-bin mean predicted probability against observed relevance frequency.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary relevance labels (0/1 or ``bool``).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of relevance, in ``[0, 1]``.
    n_bins : int, default=10
        Number of bins.
    strategy : {"uniform", "quantile"}, default="uniform"
        Equal-width bins over ``[0, 1]``, or edges at quantiles of
        ``y_prob`` (so bins hold similar counts).

    Returns
    -------
    dict
        ``edges`` (``n_bins + 1`` floats), and per bin ``count`` (int),
        ``mean_predicted`` and ``observed_frequency`` (``NaN`` for an empty
        bin), plus ``n_bins`` and ``strategy``.

    Raises
    ------
    ValueError
        For different lengths, no samples, a non-binary label, a probability
        outside ``[0, 1]`` or non-finite, ``n_bins < 1`` or an unknown
        ``strategy``.
    TypeError
        For non-numeric input or a non-integer ``n_bins``.

    Notes
    -----
    **User.** Empty bins are kept (count 0, ``NaN`` values) so tables from
    two systems line up bin for bin; scikit-learn's ``calibration_curve``
    drops them. Quantile edges can coincide when many probabilities are
    equal; such bins are empty.

    Examples
    --------
    >>> table = reliability_table([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3], n_bins=2)
    >>> table["count"], table["observed_frequency"]
    ([2, 2], [0.0, 1.0])
    """
    labels, probs = _check_inputs(y_true, y_prob)
    n_bins = _check_bins(n_bins, strategy)
    edges = _edges(probs, n_bins, strategy)
    ids = np.searchsorted(edges[1:-1], probs)
    count = np.bincount(ids, minlength=n_bins)
    sum_pred = np.bincount(ids, weights=probs, minlength=n_bins)
    sum_true = np.bincount(ids, weights=labels, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.where(count > 0, sum_pred / count, np.nan)
        observed = np.where(count > 0, sum_true / count, np.nan)
    return {
        "n_bins": n_bins,
        "strategy": strategy,
        "edges": edges.tolist(),
        "count": count.astype(int).tolist(),
        "mean_predicted": mean_pred.tolist(),
        "observed_frequency": observed.tolist(),
    }


def expected_calibration_error(
    y_true: Any, y_prob: Any, *, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """
    Count-weighted mean absolute gap between prediction and observation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary relevance labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of relevance.
    n_bins : int, default=10
        Number of bins.
    strategy : {"uniform", "quantile"}, default="uniform"
        See :func:`reliability_table`.

    Returns
    -------
    float
        ``sum_b (n_b / N) * |observed_b - mean_predicted_b|`` over non-empty
        bins; in ``[0, 1]``, 0 is perfectly calibrated.

    Raises
    ------
    ValueError, TypeError
        See :func:`reliability_table`.

    Examples
    --------
    >>> round(
    ...     expected_calibration_error([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3], n_bins=2), 4
    ... )
    0.175
    """
    table = reliability_table(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    count = np.asarray(table["count"], dtype=np.float64)
    used = count > 0
    gap = np.abs(
        np.asarray(table["observed_frequency"])[used]
        - np.asarray(table["mean_predicted"])[used]
    )
    return float(np.sum(count[used] * gap) / count.sum())


def calibration_report(
    y_true: Any, y_prob: Any, *, n_bins: int = 10, strategy: str = "uniform"
) -> dict[str, Any]:
    """
    Every calibration measure for one set of labelled probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary relevance labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities of relevance.
    n_bins : int, default=10
        Number of bins for the reliability table and ECE.
    strategy : {"uniform", "quantile"}, default="uniform"
        See :func:`reliability_table`.

    Returns
    -------
    dict
        ``n_samples``, ``prevalence`` (share of relevant labels),
        ``brier_score``, ``log_loss``, ``expected_calibration_error`` and
        ``reliability`` (the :func:`reliability_table`). JSON-serialisable
        except for ``NaN`` in empty bins.

    Raises
    ------
    ValueError, TypeError
        See :func:`reliability_table`.

    Notes
    -----
    **User.** Keep ``prevalence`` with the numbers: Brier score and log loss
    move with it, so two reports on populations of different prevalence are
    not directly comparable.
    """
    labels, probs = _check_inputs(y_true, y_prob)
    return {
        "n_samples": int(labels.size),
        "prevalence": float(labels.mean()),
        "brier_score": brier_score(labels, probs),
        "log_loss": binary_log_loss(labels, probs),
        "expected_calibration_error": expected_calibration_error(
            labels,
            probs,
            n_bins=n_bins,
            strategy=strategy,
        ),
        "reliability": reliability_table(
            labels,
            probs,
            n_bins=n_bins,
            strategy=strategy,
        ),
    }
