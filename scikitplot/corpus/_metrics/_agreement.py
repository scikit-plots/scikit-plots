# scikitplot/corpus/_metrics/_agreement.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Agreement between raters: Cohen's kappa, Fleiss' kappa, Krippendorff's alpha.

Use them to ask whether several judges (models, people, or both) agree enough
for their scores to mean something. 1 is perfect agreement, 0 is the
agreement expected by chance, and negative values are systematic
disagreement.

========================  ====================================================
statistic                 when
========================  ====================================================
:func:`cohen_kappa`       two raters, every item rated by both; optional
                          linear or quadratic weights for ordered labels
:func:`fleiss_kappa`      any number of raters, the same number per item,
                          nominal labels
:func:`krippendorff_alpha`  any number of raters, missing ratings allowed;
                          nominal, ordinal, interval or ratio values
========================  ====================================================

Notes
-----
**User note.** A statistic with no defined value (every rating identical, so
the chance agreement is already perfect) returns ``NaN`` and warns
:class:`~._inputs.UndefinedMetricWarning`; it does not raise. The results
equal scikit-learn's ``cohen_kappa_score``, statsmodels' ``fleiss_kappa`` and
the ``krippendorff`` package's ``alpha`` (checked by the tests where those
packages are installed); none of them is needed here.

**Developer note.** Krippendorff's alpha is computed from the coincidence
matrix: ``alpha = 1 - (n - 1) * sum(o * d2) / sum(n_c * n_k * d2)``, over
units with at least two ratings.

References
----------
.. [1] Cohen, J. (1960). "A coefficient of agreement for nominal scales".
   Educational and Psychological Measurement, 20(1), 37-46.
.. [2] Fleiss, J. L. (1971). "Measuring nominal scale agreement among many
   raters". Psychological Bulletin, 76(5), 378-382.
.. [3] Krippendorff, K. (2011). "Computing Krippendorff's Alpha-Reliability".
   University of Pennsylvania, Annenberg School for Communication.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Iterable, Mapping
from typing import Any, Sequence

import numpy as np

from ._inputs import UndefinedMetricWarning

__all__ = [
    "cohen_kappa",
    "fleiss_kappa",
    "krippendorff_alpha",
]

#: Weightings of :func:`cohen_kappa`.
KAPPA_WEIGHTS: tuple[Any, ...] = (None, "linear", "quadratic")

#: Measurement levels of :func:`krippendorff_alpha`.
ALPHA_LEVELS: tuple[str, ...] = ("nominal", "ordinal", "interval", "ratio")

#: Fewest ratings that can agree: per item (Fleiss) and per unit (Krippendorff).
MIN_RATINGS = 2


def _undefined(statistic: str, reason: str) -> float:
    warnings.warn(
        f"{statistic} is undefined: {reason}; returning nan.",
        UndefinedMetricWarning,
        stacklevel=3,
    )
    return math.nan


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isnan(value)
    )


def _rows(data: Any, name: str) -> list[list]:
    """Return a rectangular list of lists from a nested sequence or 2-D array."""
    if isinstance(data, (str, bytes, Mapping)) or not isinstance(data, Iterable):
        raise TypeError(
            f"{name} must be a 2-D array-like, got {type(data).__name__!r}."
        )
    rows = []
    for i, row in enumerate(data):
        if isinstance(row, (str, bytes, Mapping)) or not isinstance(row, Iterable):
            raise TypeError(
                f"{name}[{i}] must be a sequence, got {type(row).__name__!r}."
            )
        rows.append(list(row.tolist() if isinstance(row, np.ndarray) else row))
    if not rows:
        raise ValueError(f"{name} is empty.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(
            f"{name} must be rectangular; row lengths {sorted({len(r) for r in rows})}."
        )
    if width == 0:
        raise ValueError(f"{name} has no columns.")
    return rows


def _labels(values: Sequence, given: Any) -> list:
    if given is None:
        try:
            return sorted(set(values))
        except TypeError:
            raise TypeError(
                "labels of mixed, unorderable types; pass labels= in the intended order."
            ) from None
    labels = list(given)
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique.")
    unknown = sorted({repr(v) for v in values if v not in set(labels)})
    if unknown:
        raise ValueError(f"ratings outside labels: {unknown[:5]}.")
    return labels


def cohen_kappa(
    rater_a: Any, rater_b: Any, *, labels: Any = None, weights: str | None = None
) -> float:
    """
    Cohen's kappa between two raters of the same items.

    Parameters
    ----------
    rater_a, rater_b : sequence
        One label per item, in the same item order; no missing values.
    labels : sequence or None, default=None
        Label order (it matters for weights); ``None`` sorts the labels seen.
    weights : {None, "linear", "quadratic"}, default=None
        Disagreement weights: none (each disagreement counts 1), ``|i - j|``
        or ``(i - j) ** 2`` over label positions.

    Returns
    -------
    float
        ``NaN`` (with a warning) when chance agreement is already perfect.

    Raises
    ------
    ValueError
        For different lengths, no items, missing values or an unknown weight.

    Examples
    --------
    >>> round(cohen_kappa(["a", "b", "a"], ["a", "b", "b"]), 4)
    0.4
    """
    if weights not in KAPPA_WEIGHTS:
        raise ValueError(f"weights must be one of {KAPPA_WEIGHTS}, got {weights!r}.")
    a, b = (
        _rows([rater], name)[0]
        for rater, name in ((rater_a, "rater_a"), (rater_b, "rater_b"))
    )
    if len(a) != len(b):
        raise ValueError(
            f"rater_a and rater_b rate different numbers of items: {len(a)} and {len(b)}."
        )
    if any(_is_missing(v) for v in a + b):
        raise ValueError(
            "cohen_kappa needs every item rated by both raters; found a missing value."
        )
    order = _labels(a + b, labels)
    index = {label: i for i, label in enumerate(order)}
    k = len(order)
    confusion = np.zeros((k, k), dtype=np.float64)
    for x, y in zip(a, b):
        confusion[index[x], index[y]] += 1
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / confusion.sum()
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(np.float64)
    elif weights == "linear":
        w = np.abs(i - j).astype(np.float64)
    else:
        w = ((i - j) ** 2).astype(np.float64)
    denominator = float((w * expected).sum())
    if denominator == 0.0:
        return _undefined("cohen_kappa", "both raters used a single label")
    return 1.0 - float((w * confusion).sum()) / denominator


def fleiss_kappa(ratings: Any, *, labels: Any = None) -> float:
    """
    Fleiss' kappa for any number of raters with nominal labels.

    Parameters
    ----------
    ratings : array-like of shape (n_items, n_raters)
        One label per rater per item; every item has the same number of
        ratings (at least two) and none is missing.
    labels : sequence or None, default=None
        The label set; ``None`` uses the labels seen.

    Returns
    -------
    float
        ``NaN`` (with a warning) when every rating is the same label.

    Raises
    ------
    ValueError
        For fewer than two raters, a missing value or an unknown label.

    Examples
    --------
    >>> round(fleiss_kappa([[1, 1, 2], [2, 2, 2], [1, 2, 3], [3, 3, 3]]), 4)
    0.3617
    """
    rows = _rows(ratings, "ratings")
    n_raters = len(rows[0])
    if n_raters < MIN_RATINGS:
        raise ValueError("fleiss_kappa needs at least two ratings per item.")
    flat = [v for row in rows for v in row]
    if any(_is_missing(v) for v in flat):
        raise ValueError(
            "fleiss_kappa needs the same number of ratings per item; use krippendorff_alpha for gaps."
        )
    order = _labels(flat, labels)
    index = {label: i for i, label in enumerate(order)}
    counts = np.zeros((len(rows), len(order)), dtype=np.float64)
    for r, row in enumerate(rows):
        for v in row:
            counts[r, index[v]] += 1
    n_items = len(rows)
    per_item = ((counts**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    observed = float(per_item.mean())
    shares = counts.sum(axis=0) / (n_items * n_raters)
    chance = float((shares**2).sum())
    if chance == 1.0:
        return _undefined("fleiss_kappa", "every rating is the same label")
    return (observed - chance) / (1.0 - chance)


def _distance(level: str, values: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Squared difference function of Krippendorff's alpha for each value pair."""
    c = values[:, None]
    k = values[None, :]
    if level == "nominal":
        return (
            np.arange(values.size)[:, None] != np.arange(values.size)[None, :]
        ).astype(np.float64)
    if level == "interval":
        return (c - k) ** 2
    if level == "ratio":
        total = c + k
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                total == 0, 0.0, ((c - k) / np.where(total == 0, 1.0, total)) ** 2
            )
    cumulative = np.concatenate([[0.0], np.cumsum(totals)])
    lo = np.minimum(np.arange(values.size)[:, None], np.arange(values.size)[None, :])
    hi = np.maximum(np.arange(values.size)[:, None], np.arange(values.size)[None, :])
    between = cumulative[hi + 1] - cumulative[lo]
    return (between - (totals[lo] + totals[hi]) / 2.0) ** 2


def krippendorff_alpha(ratings: Any, *, level: str = "nominal") -> float:
    """
    Krippendorff's alpha for any number of raters, with missing ratings.

    Parameters
    ----------
    ratings : array-like of shape (n_raters, n_units)
        One row per rater, one column per unit (the ``krippendorff``
        package's ``reliability_data`` layout). ``None`` or ``NaN`` marks a
        missing rating. Units with fewer than two ratings are ignored.
    level : {"nominal", "ordinal", "interval", "ratio"}, default="nominal"
        Measurement level. Every level but nominal needs real numbers; ratio
        also needs non-negative values.

    Returns
    -------
    float
        ``NaN`` (with a warning) when fewer than two pairable ratings exist or
        only one value occurs.

    Raises
    ------
    ValueError
        For an unknown level, or values the level cannot use.

    Examples
    --------
    >>> round(krippendorff_alpha([[1, 2, 3, 3], [1, 2, 3, 4]], level="interval"), 4)
    0.8889
    """
    if level not in ALPHA_LEVELS:
        raise ValueError(f"level must be one of {ALPHA_LEVELS}, got {level!r}.")
    rows = _rows(ratings, "ratings")
    units = [
        [row[u] for row in rows if not _is_missing(row[u])] for u in range(len(rows[0]))
    ]
    units = [unit for unit in units if len(unit) >= MIN_RATINGS]
    seen = [v for unit in units for v in unit]
    if level != "nominal":
        bad = [
            v for v in seen if isinstance(v, bool) or not isinstance(v, numbers.Real)
        ]
        if bad:
            raise ValueError(
                f"level={level!r} needs real-number ratings; got {bad[0]!r}."
            )
        if level == "ratio" and any(v < 0 for v in seen):
            raise ValueError("level='ratio' needs non-negative ratings.")
    if not seen:
        return _undefined("krippendorff_alpha", "no unit has two ratings")
    order = _labels(seen, None) if level != "nominal" else list(dict.fromkeys(seen))
    index = {value: i for i, value in enumerate(order)}
    coincidence = np.zeros((len(order), len(order)), dtype=np.float64)
    for unit in units:
        counts = np.zeros(len(order), dtype=np.float64)
        for v in unit:
            counts[index[v]] += 1
        pairs = np.outer(counts, counts) - np.diag(counts)
        coincidence += pairs / (len(unit) - 1)
    totals = coincidence.sum(axis=1)
    n = float(totals.sum())
    values = np.array(
        order if level != "nominal" else range(len(order)), dtype=np.float64
    )
    d2 = _distance(level, values, totals)
    expected = float((np.outer(totals, totals) * d2).sum())
    if expected == 0.0:
        return _undefined("krippendorff_alpha", "only one value occurs")
    return 1.0 - (n - 1.0) * float((coincidence * d2).sum()) / expected
