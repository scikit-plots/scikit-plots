# scikitplot/corpus/_metrics/_ranking.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Deterministic retrieval metrics over relevance judgments and rankings.

Each query contributes its judgments (``{doc_id: grade}``, or a collection of
relevant identifiers, each graded 1) and a ranking (identifiers, best first).
With ``R`` the relevant identifiers (grade >= ``relevance_threshold``) and
``top`` the first ``k`` ranked identifiers (the whole ranking when ``k`` is
``None``):

=====================  =====================================================
metric                 per-query value
=====================  =====================================================
``precision@k``        ``|R & top| / k``; without a cutoff, divided by the
                       ranking length (undefined for an empty ranking)
``recall@k``           ``|R & top| / |R|``
``hit_rate@k``         1 if ``R & top`` is not empty, else 0
``mrr@k``              ``1 / rank`` of the first relevant identifier in
                       ``top``, else 0
``map@k``              ``(1 / |R|) * sum(precision@i for each relevant
                       position i in top)``
``r_precision``        ``|R & first |R| ranked| / |R|`` (no cutoff)
``ndcg@k``             ``DCG@k / IDCG@k`` with
                       ``DCG@k = sum_i gain(grade_i) / log2(i + 1)``,
                       ``i = 1..k``; ``IDCG@k`` is the DCG of the judged
                       grades sorted in decreasing order
=====================  =====================================================

A query without a relevant identifier scores 0 on precision, hit rate and
MRR, whose formulas stay defined, and is *undefined* on recall, MAP and
R-precision (division by ``|R|``) and on nDCG when no judged grade is
positive. ``zero_division`` decides an undefined value, as in scikit-learn.

Notes
-----
**User note.** Unjudged identifiers are non-relevant (grade 0). nDCG uses the
grades themselves, so a grade below ``relevance_threshold`` still adds gain;
the other metrics use the threshold. ``gain="exponential"`` uses
``2 ** grade - 1`` (Burges et al., 2005) instead of the grade.

**Developer note.** :data:`_FORMULAS` holds the only per-query formula of each
metric. The public functions, :func:`ranking_metrics`, the scorer and the
evaluation harness all call it, so two paths cannot compute one metric two
ways (Rule 83).

References
----------
.. [1] Järvelin, K. and Kekäläinen, J. (2002). "Cumulated gain-based
   evaluation of IR techniques". ACM Transactions on Information Systems,
   20(4), 422-446.
.. [2] Burges, C. et al. (2005). "Learning to rank using gradient descent".
   Proceedings of the 22nd International Conference on Machine Learning.
.. [3] Voorhees, E. M. (1999). "The TREC-8 Question Answering Track Report".
   Proceedings of TREC-8.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ._inputs import (
    Judgments,
    aggregate,
    aligned,
    check_average,
    check_duplicates,
    check_id_types,
    check_k,
    check_relevance_threshold,
    check_sample_weight,
    check_zero_division,
    judgments,
    ranking,
    where,
)

__all__ = [
    "hit_rate_at_k",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "r_precision",
    "ranking_metrics",
    "recall_at_k",
]


def _linear_gain(grade: float) -> float:
    return grade


def _exponential_gain(grade: float) -> float:
    try:
        return 2.0**grade - 1.0
    except OverflowError:
        raise ValueError(
            f"The grade {grade!r} overflows the exponential gain 2 ** grade - 1; "
            "rescale the grades or use gain='linear'."
        ) from None


#: Gain functions for nDCG.
GAINS: Mapping[str, Callable[[float], float]] = {
    "linear": _linear_gain,
    "exponential": _exponential_gain,
}

_NAME_RE = re.compile(
    r"^(?P<base>[a-z][a-z0-9_]*)(?:@(?P<k>[1-9][0-9]*))?$",
)


# ---------------------------------------------------------------------------
# Per-query formulas (the only ones)
# ---------------------------------------------------------------------------


def _top(ranked: tuple, k: int | None) -> tuple:
    return ranked if k is None else ranked[:k]


def _precision(
    judged: Judgments,
    ranked: tuple,
    k: int | None,
    gain: Any,
) -> float:
    denominator = len(ranked) if k is None else k
    if denominator == 0:
        return math.nan
    relevant = judged.relevant
    return sum(1 for ident in _top(ranked, k) if ident in relevant) / denominator


def _recall(
    judged: Judgments,
    ranked: tuple,
    k: int | None,
    gain: Any,
) -> float:
    relevant = judged.relevant
    if not relevant:
        return math.nan
    return sum(1 for ident in _top(ranked, k) if ident in relevant) / len(relevant)


def _hit_rate(
    judged: Judgments,
    ranked: tuple,
    k: int | None,
    gain: Any,
) -> float:
    relevant = judged.relevant
    return 1.0 if any(ident in relevant for ident in _top(ranked, k)) else 0.0


def _reciprocal_rank(
    judged: Judgments,
    ranked: tuple,
    k: int | None,
    gain: Any,
) -> float:
    relevant = judged.relevant
    for position, ident in enumerate(_top(ranked, k), start=1):
        if ident in relevant:
            return 1.0 / position
    return 0.0


def _average_precision(
    judged: Judgments,
    ranked: tuple,
    k: int | None,
    gain: Any,
) -> float:
    relevant = judged.relevant
    if not relevant:
        return math.nan
    hits = 0
    total = 0.0
    for position, ident in enumerate(_top(ranked, k), start=1):
        if ident in relevant:
            hits += 1
            total += hits / position
    return total / len(relevant)


def _r_precision(
    judged: Judgments,
    ranked: tuple,
    k: int | None,
    gain: Any,
) -> float:
    relevant = judged.relevant
    if not relevant:
        return math.nan
    size = len(relevant)
    return sum(1 for ident in ranked[:size] if ident in relevant) / size


def _ndcg(
    judged: Judgments,
    ranked: tuple,
    k: int | None,
    gain: Any,
) -> float:
    ideal = sorted(
        (g for g in judged.grades.values() if g > 0.0),
        reverse=True,
    )
    if k is not None:
        ideal = ideal[:k]
    idcg = sum(gain(g) / math.log2(i + 2) for i, g in enumerate(ideal))
    if not math.isfinite(idcg):
        raise ValueError(
            "The ideal DCG overflows a float; rescale the grades (nDCG does not "
            "change when every grade is multiplied by the same positive number)."
        )
    if idcg == 0.0:
        return math.nan
    grades = judged.grades
    dcg = sum(
        gain(grades.get(ident, 0.0)) / math.log2(i + 2)
        for i, ident in enumerate(_top(ranked, k))
    )
    return dcg / idcg


@dataclass(frozen=True)
class _Formula:
    """One metric: its per-query formula, cutoff rule and undefined reason."""

    compute: Callable[[Judgments, tuple, Any, Any], float]
    accepts_k: bool
    undefined: str


#: The only per-query formula of each metric (see the module notes).
_FORMULAS: Mapping[str, _Formula] = {
    "precision": _Formula(
        _precision,
        True,
        "precision without a cutoff over an empty ranking",
    ),
    "recall": _Formula(_recall, True, "no relevant document"),
    "hit_rate": _Formula(_hit_rate, True, "never undefined"),
    "mrr": _Formula(_reciprocal_rank, True, "never undefined"),
    "map": _Formula(_average_precision, True, "no relevant document"),
    "r_precision": _Formula(_r_precision, False, "no relevant document"),
    "ndcg": _Formula(_ndcg, True, "no judged document has a positive grade"),
}

#: Metric names accepted by :func:`ranking_metrics` (``@k`` optional except
#: for ``r_precision``, which takes none).
METRIC_NAMES: tuple[str, ...] = tuple(sorted(_FORMULAS))


def parse_metric(name: Any) -> tuple[str, int | None]:
    """
    Split a metric name into its base and cutoff.

    Parameters
    ----------
    name : str
        ``"<base>"`` or ``"<base>@<k>"`` with ``k >= 1``, for example
        ``"recall@10"``, ``"mrr"`` or ``"ndcg@5"``.

    Returns
    -------
    base : str
    k : int or None

    Raises
    ------
    TypeError
        If ``name`` is not a ``str``.
    ValueError
        For an unknown base, a malformed cutoff, or a cutoff on
        ``r_precision``.

    Examples
    --------
    >>> parse_metric("ndcg@10")
    ('ndcg', 10)
    >>> parse_metric("mrr")
    ('mrr', None)
    """
    if not isinstance(name, str):
        raise TypeError(
            f"A metric name must be a str, got {type(name).__name__!r}.",
        )
    match = _NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Malformed metric name {name!r}: expected '<base>' or '<base>@<k>' "
            'with an integer k >= 1, such as "recall@10".'
        )
    base = match.group("base")
    if base not in _FORMULAS:
        raise ValueError(
            f"Unknown ranking metric {base!r} in {name!r}; use one of {METRIC_NAMES}.",
        )
    k = None if match.group("k") is None else int(match.group("k"))
    if k is not None and not _FORMULAS[base].accepts_k:
        raise ValueError(f"{base} takes no cutoff; got {name!r}.")
    return base, k


def _check_gain(gain: Any) -> Callable[[float], float]:
    try:
        return GAINS[gain]
    except (KeyError, TypeError):
        raise ValueError(
            f"gain must be one of {tuple(GAINS)}, got {gain!r}.",
        ) from None


def _compute(
    y_true: Any,
    y_pred: Any,
    specs: Sequence[tuple[str, str, int | None]],
    *,
    sample_weight: Any,
    relevance_threshold: Any,
    duplicates: Any,
    zero_division: Any,
    gain: Any,
    average: Any,
) -> dict[str, Any]:
    """Validate once, then apply each requested formula to every query."""
    threshold = check_relevance_threshold(relevance_threshold)
    policy = check_duplicates(duplicates)
    division = check_zero_division(zero_division)
    mode = check_average(average)
    gain_fn = _check_gain(gain)
    true_rows, pred_rows, keys = aligned(y_true, y_pred)
    n = len(true_rows)
    weights = check_sample_weight(sample_weight, n)
    labels = [where(i, None if keys is None else keys[i]) for i in range(n)]
    judged = [
        judgments(row, relevance_threshold=threshold, label=label)
        for row, label in zip(true_rows, labels)
    ]
    ranked = [
        ranking(row, duplicates=policy, label=label)
        for row, label in zip(pred_rows, labels)
    ]
    check_id_types(judged, ranked)
    out: dict[str, Any] = {}
    for (name, base, _), values in zip(
        specs,
        per_query(judged, ranked, specs, gain_fn),
    ):
        out[name] = aggregate(
            values,
            zero_division=division,
            sample_weight=weights,
            average=mode,
            metric=name,
            reason=_FORMULAS[base].undefined,
        )
    return out


def per_query(
    judged: Sequence[Judgments],
    ranked: Sequence[tuple],
    specs: Sequence[tuple[str, str, int | None]],
    gain: Callable[[float], float],
) -> list[np.ndarray]:
    """
    Apply each requested formula to every normalised query.

    Parameters
    ----------
    judged : sequence of Judgments
        Normalised judgments, one per query.
    ranked : sequence of tuple
        Normalised rankings, aligned with ``judged``.
    specs : sequence of (name, base, k)
        Parsed metric names.
    gain : callable
        nDCG gain function.

    Returns
    -------
    list of numpy.ndarray
        One float64 array per spec; ``NaN`` marks an undefined value.
    """
    return [
        np.fromiter(
            (_FORMULAS[base].compute(j, r, k, gain) for j, r in zip(judged, ranked)),
            dtype=np.float64,
            count=len(judged),
        )
        for _, base, k in specs
    ]


def undefined_reason(base: str) -> str:
    """Why ``base`` can be undefined, for warnings."""
    return _FORMULAS[base].undefined


def check_gain(gain: Any) -> Callable[[float], float]:
    """Validate an nDCG gain name and return its function."""
    return _check_gain(gain)


def _spec(base: str, k: Any) -> list[tuple[str, str, int | None]]:
    """Return the one-metric spec for a public function (validates ``k``)."""
    cutoff = check_k(k) if _FORMULAS[base].accepts_k else None
    return [(base if cutoff is None else f"{base}@{cutoff}", base, cutoff)]


def _only(result: dict[str, Any]) -> Any:
    """Return the single value of a one-metric result."""
    (value,) = result.values()
    return value


def precision_at_k(
    y_true: Any,
    y_pred: Any,
    *,
    k: int | None = 10,
    sample_weight: Any = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    average: str | None = "mean",
) -> Any:
    """
    Fraction of the first ``k`` ranked identifiers that are relevant.

    Parameters
    ----------
    y_true : sequence or mapping
        Relevance judgments per query (see :mod:`._ranking`).
    y_pred : sequence or mapping
        Rankings per query, best first.
    k : int or None, default=10
        Cutoff. The denominator is ``k`` even when fewer are ranked. ``None``
        divides by the ranking length instead.
    sample_weight : array-like of shape (n_queries,), default=None
        Per-query weights of the mean.
    relevance_threshold : float, default=1
        Grade from which a judged identifier is relevant.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Value when the metric is undefined (only without a cutoff, for an
        empty ranking).
    average : {"mean", None}, default="mean"
        ``None`` returns the per-query values.

    Returns
    -------
    float or numpy.ndarray

    Raises
    ------
    TypeError, ValueError
        For malformed inputs; the message names the query.

    See Also
    --------
    recall_at_k, ranking_metrics

    Examples
    --------
    >>> precision_at_k([{"a", "b"}], [["a", "x", "b", "y"]], k=2)
    0.5
    """
    return _only(
        _compute(
            y_true,
            y_pred,
            _spec("precision", k),
            sample_weight=sample_weight,
            relevance_threshold=relevance_threshold,
            duplicates=duplicates,
            zero_division=zero_division,
            gain="linear",
            average=average,
        )
    )


def recall_at_k(
    y_true: Any,
    y_pred: Any,
    *,
    k: int | None = 10,
    sample_weight: Any = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    average: str | None = "mean",
) -> Any:
    """
    Fraction of the relevant identifiers found in the first ``k``.

    Parameters
    ----------
    y_true : sequence or mapping
        Relevance judgments per query.
    y_pred : sequence or mapping
        Rankings per query, best first.
    k : int or None, default=10
        Cutoff; ``None`` uses the whole ranking.
    sample_weight : array-like of shape (n_queries,), default=None
        Per-query weights of the mean.
    relevance_threshold : float, default=1
        Grade from which a judged identifier is relevant.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Value for a query without a relevant identifier.
    average : {"mean", None}, default="mean"
        ``None`` returns the per-query values.

    Returns
    -------
    float or numpy.ndarray

    Raises
    ------
    TypeError, ValueError
        For malformed inputs; the message names the query.

    See Also
    --------
    precision_at_k, hit_rate_at_k, ranking_metrics

    Examples
    --------
    >>> recall_at_k([{"a", "b"}], [["a", "x", "b"]], k=2)
    0.5
    >>> recall_at_k({"q1": ["a"], "q2": ["b"]}, {"q1": ["a"], "q2": []}, average=None)
    array([1., 0.])
    """
    return _only(
        _compute(
            y_true,
            y_pred,
            _spec("recall", k),
            sample_weight=sample_weight,
            relevance_threshold=relevance_threshold,
            duplicates=duplicates,
            zero_division=zero_division,
            gain="linear",
            average=average,
        )
    )


def hit_rate_at_k(
    y_true: Any,
    y_pred: Any,
    *,
    k: int | None = 10,
    sample_weight: Any = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    average: str | None = "mean",
) -> Any:
    """
    Fraction of queries with at least one relevant identifier in the first ``k``.

    Parameters
    ----------
    y_true : sequence or mapping
        Relevance judgments per query.
    y_pred : sequence or mapping
        Rankings per query, best first.
    k : int or None, default=10
        Cutoff; ``None`` uses the whole ranking.
    sample_weight : array-like of shape (n_queries,), default=None
        Per-query weights of the mean.
    relevance_threshold : float, default=1
        Grade from which a judged identifier is relevant.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Accepted for a uniform signature; this metric is never undefined.
    average : {"mean", None}, default="mean"
        ``None`` returns the per-query values.

    Returns
    -------
    float or numpy.ndarray

    See Also
    --------
    recall_at_k, mean_reciprocal_rank

    Examples
    --------
    >>> hit_rate_at_k([["a"], ["b"]], [["x", "a"], ["x", "y"]], k=2)
    0.5
    """
    return _only(
        _compute(
            y_true,
            y_pred,
            _spec("hit_rate", k),
            sample_weight=sample_weight,
            relevance_threshold=relevance_threshold,
            duplicates=duplicates,
            zero_division=zero_division,
            gain="linear",
            average=average,
        )
    )


def mean_reciprocal_rank(
    y_true: Any,
    y_pred: Any,
    *,
    k: int | None = None,
    sample_weight: Any = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    average: str | None = "mean",
) -> Any:
    """
    Mean of ``1 / rank`` of the first relevant identifier (MRR).

    Parameters
    ----------
    y_true : sequence or mapping
        Relevance judgments per query.
    y_pred : sequence or mapping
        Rankings per query, best first.
    k : int or None, default=None
        Cutoff (MRR@k); a first relevant identifier beyond it scores 0.
    sample_weight : array-like of shape (n_queries,), default=None
        Per-query weights of the mean.
    relevance_threshold : float, default=1
        Grade from which a judged identifier is relevant.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Accepted for a uniform signature; this metric is never undefined.
    average : {"mean", None}, default="mean"
        ``None`` returns the per-query reciprocal ranks.

    Returns
    -------
    float or numpy.ndarray

    See Also
    --------
    hit_rate_at_k, mean_average_precision

    Examples
    --------
    >>> mean_reciprocal_rank([["a"], ["b"]], [["a", "x"], ["x", "b"]])
    0.75
    """
    return _only(
        _compute(
            y_true,
            y_pred,
            _spec("mrr", k),
            sample_weight=sample_weight,
            relevance_threshold=relevance_threshold,
            duplicates=duplicates,
            zero_division=zero_division,
            gain="linear",
            average=average,
        )
    )


def mean_average_precision(
    y_true: Any,
    y_pred: Any,
    *,
    k: int | None = None,
    sample_weight: Any = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    average: str | None = "mean",
) -> Any:
    """
    Mean average precision (MAP), normalised by all relevant identifiers.

    Parameters
    ----------
    y_true : sequence or mapping
        Relevance judgments per query.
    y_pred : sequence or mapping
        Rankings per query, best first.
    k : int or None, default=None
        Cutoff (MAP@k). The normaliser stays ``|R|``, so MAP@k <= recall@k.
    sample_weight : array-like of shape (n_queries,), default=None
        Per-query weights of the mean.
    relevance_threshold : float, default=1
        Grade from which a judged identifier is relevant.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Value for a query without a relevant identifier.
    average : {"mean", None}, default="mean"
        ``None`` returns the per-query average precisions.

    Returns
    -------
    float or numpy.ndarray

    See Also
    --------
    mean_reciprocal_rank, r_precision

    Examples
    --------
    >>> mean_average_precision([["a", "b"]], [["a", "x", "b"]])
    0.8333333333333333
    """
    return _only(
        _compute(
            y_true,
            y_pred,
            _spec("map", k),
            sample_weight=sample_weight,
            relevance_threshold=relevance_threshold,
            duplicates=duplicates,
            zero_division=zero_division,
            gain="linear",
            average=average,
        )
    )


def r_precision(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    average: str | None = "mean",
) -> Any:
    """
    Precision at ``|R|``, the number of relevant identifiers of each query.

    Parameters
    ----------
    y_true : sequence or mapping
        Relevance judgments per query.
    y_pred : sequence or mapping
        Rankings per query, best first.
    sample_weight : array-like of shape (n_queries,), default=None
        Per-query weights of the mean.
    relevance_threshold : float, default=1
        Grade from which a judged identifier is relevant.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Value for a query without a relevant identifier.
    average : {"mean", None}, default="mean"
        ``None`` returns the per-query values.

    Returns
    -------
    float or numpy.ndarray

    See Also
    --------
    precision_at_k, mean_average_precision

    Examples
    --------
    >>> r_precision([["a", "b"]], [["a", "x", "b"]])
    0.5
    """
    return _only(
        _compute(
            y_true,
            y_pred,
            _spec("r_precision", None),
            sample_weight=sample_weight,
            relevance_threshold=relevance_threshold,
            duplicates=duplicates,
            zero_division=zero_division,
            gain="linear",
            average=average,
        )
    )


def ndcg_at_k(
    y_true: Any,
    y_pred: Any,
    *,
    k: int | None = 10,
    gain: str = "linear",
    sample_weight: Any = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    average: str | None = "mean",
) -> Any:
    """
    Normalised discounted cumulative gain (nDCG@k) over graded judgments.

    Parameters
    ----------
    y_true : sequence or mapping
        Relevance judgments per query; grades are the gains.
    y_pred : sequence or mapping
        Rankings per query, best first.
    k : int or None, default=10
        Cutoff of both DCG and the ideal DCG; ``None`` uses everything.
    gain : {"linear", "exponential"}, default="linear"
        ``grade`` or ``2 ** grade - 1``.
    sample_weight : array-like of shape (n_queries,), default=None
        Per-query weights of the mean.
    relevance_threshold : float, default=1
        Accepted for a uniform signature; nDCG uses the grades, not the
        threshold.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Value for a query whose judged grades are all zero.
    average : {"mean", None}, default="mean"
        ``None`` returns the per-query values.

    Returns
    -------
    float or numpy.ndarray
        Values in [0, 1] (up to floating-point rounding).

    See Also
    --------
    mean_average_precision, ranking_metrics

    Notes
    -----
    This takes rankings of identifiers and graded judgments, the form a
    retrieval system produces. :func:`sklearn.metrics.ndcg_score` takes score
    matrices over a fixed label set instead.

    Examples
    --------
    >>> round(ndcg_at_k([{"a": 3, "b": 1}], [["b", "a"]], k=2), 4)
    0.7967
    """
    return _only(
        _compute(
            y_true,
            y_pred,
            _spec("ndcg", k),
            sample_weight=sample_weight,
            relevance_threshold=relevance_threshold,
            duplicates=duplicates,
            zero_division=zero_division,
            gain=gain,
            average=average,
        )
    )


def ranking_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    metrics: Iterable[str] = (
        "hit_rate@10",
        "precision@10",
        "recall@10",
        "mrr@10",
        "ndcg@10",
    ),
    sample_weight: Any = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    gain: str = "linear",
    average: str | None = "mean",
) -> dict[str, Any]:
    """
    Compute several ranking metrics in one validated pass.

    Parameters
    ----------
    y_true : sequence or mapping
        Relevance judgments per query.
    y_pred : sequence or mapping
        Rankings per query, best first.
    metrics : iterable of str, optional
        Default ``("hit_rate@10", "precision@10", "recall@10", "mrr@10",
        "ndcg@10")``. Names ``"<base>"`` or ``"<base>@<k>"``; bases are ``precision``,
        ``recall``, ``hit_rate``, ``mrr``, ``map``, ``ndcg`` and
        ``r_precision`` (no cutoff). Without ``@k`` the whole ranking is used.
    sample_weight : array-like of shape (n_queries,), default=None
        Per-query weights of the means.
    relevance_threshold : float, default=1
        Grade from which a judged identifier is relevant.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Value of an undefined metric.
    gain : {"linear", "exponential"}, default="linear"
        nDCG gain.
    average : {"mean", None}, default="mean"
        ``None`` returns per-query arrays.

    Returns
    -------
    dict
        Metric name to value (or array), in the order requested.

    Raises
    ------
    TypeError, ValueError
        For malformed inputs, an unknown name or a repeated name.

    Examples
    --------
    >>> ranking_metrics([["a"]], [["x", "a"]], metrics=["recall@1", "mrr"])
    {'recall@1': 0.0, 'mrr': 0.5}
    """
    if isinstance(metrics, str):
        raise TypeError('metrics is a collection of names; wrap one name as ["mrr"].')
    names = list(metrics)
    if not names:
        raise ValueError("metrics must name at least one metric.")
    if len(set(names)) != len(names):
        raise ValueError(f"metrics names a metric twice: {names!r}.")
    specs = [(name, *parse_metric(name)) for name in names]
    return _compute(
        y_true,
        y_pred,
        specs,
        sample_weight=sample_weight,
        relevance_threshold=relevance_threshold,
        duplicates=duplicates,
        zero_division=zero_division,
        gain=gain,
        average=average,
    )
