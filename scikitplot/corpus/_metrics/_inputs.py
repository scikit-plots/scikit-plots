# scikitplot/corpus/_metrics/_inputs.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Validation and normalisation shared by every evaluation path.

Every metric function, the multi-metric pass, the scikit-learn scorer and the
evaluation harness read their inputs through this module, so a malformed input
fails the same way, with the same message, whichever entry point received it.

Notes
-----
**Developer note.** Identifiers are ``str`` or integers (normalised to
``int``), and one evaluation uses one of the two: ``"1"`` and ``1`` are
different documents, and a qrels file keyed by strings scored against integer
rankings would otherwise match nothing and report zeros. A ``str`` is never
read as a collection of identifiers, which is the bug class Run 1 removed
from ``rank_bm25``.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "UndefinedMetricWarning",
    "ranked_ids",
]

#: Accepted values of ``duplicates``.
DUPLICATE_POLICIES: tuple[str, ...] = ("raise", "first")

#: Accepted values of ``average``.
AVERAGE_OPTIONS: tuple[Any, ...] = ("mean", None)

#: Response statuses that mean a requested retrieval leg did not run normally
#: (``RetrievalStatus`` values).
DEGRADED_STATUSES: frozenset[str] = frozenset({"degraded", "failed", "cancelled"})

#: Accepted values of ``on_degraded``.
ON_DEGRADED_POLICIES: tuple[str, ...] = ("raise", "warn", "ignore")

#: How many offending keys a message lists before eliding the rest.
LISTED_KEYS = 5

#: Length of an ``(identifier, score)`` result pair.
_PAIR_LENGTH = 2


class UndefinedMetricWarning(UserWarning):
    """
    A metric is undefined for some queries and was scored by ``zero_division``.

    Notes
    -----
    Raised with ``zero_division="warn"``, which scores an undefined value as
    ``0.0``, as scikit-learn does for precision and recall. It is distinct from
    ``sklearn.exceptions.UndefinedMetricWarning``, so a filter for one does not
    silence the other.
    """


@dataclass(frozen=True)
class Judgments:
    """
    Relevance judgments of one query.

    Parameters
    ----------
    grades : mapping
        Every judged identifier and its grade (a finite number >= 0). An
        identifier that is not judged has grade 0.
    relevant : frozenset
        Identifiers whose grade is at least the relevance threshold.
    """

    grades: Mapping[Any, float]
    relevant: frozenset


def is_mapping_like(value: Any) -> bool:
    """
    Whether ``value`` has ``keys()`` and ``items()`` without being a mapping.

    A pandas ``Series`` is the common case. As judgments or a ranking it is
    ambiguous: its index or its values could hold the identifiers, and reading
    the wrong one gives plausible, wrong metrics. Rows refuse it; see
    :func:`judgments`, :func:`ranking` and :func:`aligned`.
    """
    return (
        not isinstance(value, (Mapping, str, bytes))
        and callable(getattr(value, "keys", None))
        and callable(getattr(value, "items", None))
    )


def where(position: int, key: Any = None) -> str:
    """Return the label used in messages for one query."""
    return f"query {key!r}" if key is not None else f"query at position {position}"


def check_k(k: Any, *, allow_none: bool = True, name: str = "k") -> int | None:
    """
    Validate a rank cutoff.

    Parameters
    ----------
    k : int or None
        Cutoff, at least 1. ``None`` means the whole ranking.
    allow_none : bool, optional
        Whether ``None`` is accepted.
    name : str, optional
        Parameter name for messages.

    Returns
    -------
    int or None

    Raises
    ------
    TypeError
        If ``k`` is not an integer (``bool`` included).
    ValueError
        If ``k`` is less than 1, or ``None`` where it is not allowed.
    """
    if k is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required for this metric; got None.")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise TypeError(
            f"{name} must be an int >= 1 or None, got {type(k).__name__!r}."
        )
    if k < 1:
        raise ValueError(f"{name} must be >= 1, got {k}.")
    return int(k)


def check_relevance_threshold(threshold: Any) -> float:
    """
    Validate the grade from which a judged identifier counts as relevant.

    Parameters
    ----------
    threshold : float
        Finite and strictly positive, so an unjudged identifier (grade 0) is
        never relevant.

    Returns
    -------
    float

    Raises
    ------
    TypeError
        If ``threshold`` is not a real number (``bool`` included).
    ValueError
        If it is not finite or not positive.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise TypeError(
            "relevance_threshold must be a real number, got "
            f"{type(threshold).__name__!r}."
        )
    value = float(threshold)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"relevance_threshold must be finite and > 0, got {threshold!r}."
        )
    return value


def check_duplicates(policy: Any) -> str:
    """Validate the duplicate-identifier policy of a ranking."""
    if policy not in DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicates must be one of {DUPLICATE_POLICIES}, got {policy!r}."
        )
    return policy


def check_zero_division(zero_division: Any) -> str | float:
    """
    Validate ``zero_division``.

    Parameters
    ----------
    zero_division : {"warn", 0.0, 1.0, np.nan}
        Value of an undefined metric. ``"warn"`` scores it ``0.0`` and warns;
        ``np.nan`` keeps it undefined and excludes it from the mean.

    Returns
    -------
    str or float

    Raises
    ------
    ValueError
        For any other value.
    """
    if isinstance(zero_division, str):
        if zero_division == "warn":
            return zero_division
    elif isinstance(zero_division, numbers.Real) and not isinstance(
        zero_division, bool
    ):
        value = float(zero_division)
        if math.isnan(value) or value in (0.0, 1.0):
            return value
    raise ValueError(
        f'zero_division must be "warn", 0.0, 1.0 or np.nan, got {zero_division!r}.'
    )


def check_average(average: Any) -> str | None:
    """Validate ``average`` (``"mean"`` or ``None`` for per-query values)."""
    if average not in AVERAGE_OPTIONS:
        raise ValueError(f'average must be "mean" or None, got {average!r}.')
    return average


def check_sample_weight(sample_weight: Any, n: int) -> np.ndarray | None:
    """
    Validate per-query weights.

    Parameters
    ----------
    sample_weight : array-like of shape (n,) or None
        Finite, non-negative weights in evaluation order.
    n : int
        Number of queries.

    Returns
    -------
    numpy.ndarray of float64 or None

    Raises
    ------
    ValueError
        If the shape is wrong, or a weight is negative or not finite.
    """
    if sample_weight is None:
        return None
    if isinstance(sample_weight, Mapping):
        raise TypeError(
            "sample_weight must be a sequence in evaluation order, not a mapping."
        )
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n,):
        raise ValueError(f"sample_weight must have shape ({n},), got {weights.shape}.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weight must be finite and non-negative.")
    return weights


def as_id(value: Any, *, label: str) -> str | int:
    """
    Return ``value`` as an identifier.

    Parameters
    ----------
    value : str or int
        A ``str`` (returned as a plain ``str``) or an integer that is not a
        ``bool`` (returned as ``int``, so ``numpy.int64(3)`` and ``3`` are one
        identifier).
    label : str
        Where the value came from, for messages.

    Returns
    -------
    str or int

    Raises
    ------
    TypeError
        For any other type.
    """
    if isinstance(value, str):
        return str(value)
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    raise TypeError(
        f"{label}: identifiers must be str or int, got {type(value).__name__!r} "
        f"({value!r})."
    )


def judgments(row: Any, *, relevance_threshold: float, label: str) -> Judgments:
    """
    Normalise the relevance judgments of one query.

    Parameters
    ----------
    row : mapping or collection
        ``{id: grade}`` with finite grades >= 0, or a collection of relevant
        identifiers (each graded 1).
    relevance_threshold : float
        Grade from which an identifier is relevant.
    label : str
        Query label for messages.

    Returns
    -------
    Judgments

    Raises
    ------
    TypeError
        For a ``str``/``bytes`` row, a mapping-like row (such as a pandas
        ``Series``, whose index or values could be the identifiers), a
        non-iterable row, a bad identifier or a grade that is not a real
        number.
    ValueError
        For a negative or non-finite grade, or an identifier listed twice.
    """
    grades: dict[Any, float] = {}
    if is_mapping_like(row):
        raise TypeError(
            f"{label}: a {type(row).__name__} is ambiguous as relevance judgments "
            "(its index or its values could be the identifiers). Pass .to_dict() "
            "for {doc_id: grade}, or .tolist() for a collection of relevant "
            "identifiers."
        )
    if isinstance(row, Mapping):
        for key, grade in row.items():
            ident = as_id(key, label=label)
            if isinstance(grade, bool) or not isinstance(grade, numbers.Real):
                raise TypeError(
                    f"{label}: the grade of {ident!r} must be a real number, got "
                    f"{type(grade).__name__!r}; write booleans as 0 and 1."
                )
            value = float(grade)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{label}: the grade of {ident!r} must be finite and >= 0, "
                    f"got {grade!r}."
                )
            if ident in grades:
                raise ValueError(f"{label}: {ident!r} is judged twice.")
            grades[ident] = value
    elif isinstance(row, (str, bytes)):
        raise TypeError(
            f"{label}: relevance judgments are a mapping or a collection of "
            f"identifiers; a {type(row).__name__} is not read as a collection of "
            "characters. Wrap a single identifier in a list."
        )
    elif isinstance(row, Iterable):
        for item in row:
            ident = as_id(item, label=label)
            if ident in grades:
                raise ValueError(f"{label}: {ident!r} is listed as relevant twice.")
            grades[ident] = 1.0
    else:
        raise TypeError(
            f"{label}: relevance judgments must be a mapping or a collection of "
            f"identifiers, got {type(row).__name__!r}."
        )
    relevant = frozenset(i for i, g in grades.items() if g >= relevance_threshold)
    return Judgments(grades=grades, relevant=relevant)


def ranking(row: Any, *, duplicates: str, label: str) -> tuple:
    """
    Normalise the ranking of one query.

    Parameters
    ----------
    row : sequence
        Identifiers, best first.
    duplicates : {"raise", "first"}
        ``"first"`` keeps the first (best) occurrence of a repeated identifier,
        as when several retrieved chunks map to one document.
    label : str
        Query label for messages.

    Returns
    -------
    tuple
        The identifiers, without repeats.

    Raises
    ------
    TypeError
        For a ``str``/``bytes``/mapping row, a mapping-like row (such as a
        pandas ``Series``), or a bad identifier.
    ValueError
        For a repeated identifier with ``duplicates="raise"``.
    """
    if is_mapping_like(row):
        raise TypeError(
            f"{label}: a {type(row).__name__} is ambiguous as a ranking (its index "
            "or its values could be the identifiers). Pass .tolist() with the "
            "identifiers in rank order."
        )
    if isinstance(row, (str, bytes, Mapping)):
        raise TypeError(
            f"{label}: a ranking is a sequence of identifiers, best first; got a "
            f"{type(row).__name__}. Wrap a single identifier in a list."
        )
    if not isinstance(row, Iterable):
        raise TypeError(
            f"{label}: a ranking must be a sequence of identifiers, got "
            f"{type(row).__name__!r}."
        )
    seen: set = set()
    out = []
    for item in row:
        ident = as_id(item, label=label)
        if ident in seen:
            if duplicates == "raise":
                raise ValueError(
                    f"{label}: {ident!r} appears more than once in the ranking. "
                    "Pass duplicates='first' to keep its best position, for "
                    "example when several chunks map to one document."
                )
            continue
        seen.add(ident)
        out.append(ident)
    return tuple(out)


def aligned(y_true: Any, y_pred: Any) -> tuple[list, list, list | None]:
    """
    Align judgments and predictions by query.

    Parameters
    ----------
    y_true, y_pred : sequence or mapping
        Either two sequences of equal length (query ``i`` is position ``i``),
        or two mappings keyed by query identifier with the same keys (the order
        of ``y_true`` is the evaluation order).

    Returns
    -------
    true_rows : list
    pred_rows : list
    keys : list or None
        Query identifiers for mappings, ``None`` for sequences.

    Raises
    ------
    TypeError
        If one is a mapping and the other is not, or either is a ``str``.
    ValueError
        If the lengths or key sets differ, if both are indexed (pandas
        ``Series``) with different indexes, or if there is no query.

    Notes
    -----
    An indexed sequence such as a pandas ``Series`` is aligned by position,
    which is scikit-learn's convention for ``y``. When both sides are indexed
    and the indexes differ, positional alignment would pair different queries,
    so it is refused instead of guessed.
    """
    true_is_map = isinstance(y_true, Mapping)
    pred_is_map = isinstance(y_pred, Mapping)
    if true_is_map != pred_is_map:
        raise TypeError(
            "y_true and y_pred must both be sequences (aligned by position) or "
            "both be mappings keyed by query identifier."
        )
    if true_is_map:
        keys = list(y_true)
        missing = [k for k in keys if k not in y_pred]
        extra = [k for k in y_pred if k not in y_true]
        if missing or extra:
            raise ValueError(
                "y_true and y_pred must have the same query identifiers; "
                f"missing from y_pred: {missing[:LISTED_KEYS]!r}"
                f"{' ...' if len(missing) > LISTED_KEYS else ''}, "
                f"unknown in y_pred: {extra[:LISTED_KEYS]!r}"
                f"{' ...' if len(extra) > LISTED_KEYS else ''}."
                " Give a query with no results an empty ranking."
            )
        true_rows = [y_true[k] for k in keys]
        pred_rows = [y_pred[k] for k in keys]
    else:
        for name, value in (("y_true", y_true), ("y_pred", y_pred)):
            if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
                raise TypeError(
                    f"{name} must be a sequence with one entry per query, got "
                    f"{type(value).__name__!r}."
                )
        if is_mapping_like(y_true) and is_mapping_like(y_pred):
            true_index = list(y_true.keys())
            pred_index = list(y_pred.keys())
            if true_index != pred_index:
                raise ValueError(
                    "y_true and y_pred are indexed differently, so aligning them by "
                    "position would pair different queries. Reorder one to the "
                    "other's index, or pass .to_dict() for both to align by key."
                )
        keys = None
        true_rows = list(y_true)
        pred_rows = list(y_pred)
        if len(true_rows) != len(pred_rows):
            raise ValueError(
                f"y_true has {len(true_rows)} queries but y_pred has "
                f"{len(pred_rows)}; they are aligned by position."
            )
    if not true_rows:
        raise ValueError("At least one query is required.")
    return true_rows, pred_rows, keys


def check_id_types(
    all_judgments: Sequence[Judgments], rankings: Sequence[tuple]
) -> None:
    """
    Require one identifier type (``str`` or ``int``) across an evaluation.

    Raises
    ------
    ValueError
        If both types occur, naming one identifier of each.
    """
    examples: dict[type, Any] = {}
    for judged in all_judgments:
        for ident in judged.grades:
            examples.setdefault(type(ident), ident)
    for row in rankings:
        for ident in row:
            examples.setdefault(type(ident), ident)
    if len(examples) > 1:
        raise ValueError(
            "Identifiers mix str and int (for example "
            f"{examples[str]!r} and {examples[int]!r}); one evaluation uses one "
            "type, so convert qrels and rankings to the same one."
        )


def aggregate(
    values: np.ndarray,
    *,
    zero_division: str | float,
    sample_weight: np.ndarray | None,
    average: str | None,
    metric: str,
    reason: str,
    stacklevel: int = 4,
) -> float | np.ndarray:
    """
    Apply ``zero_division`` to undefined (``NaN``) values, then average.

    Parameters
    ----------
    values : numpy.ndarray of float64
        Per-query values; ``NaN`` marks an undefined value.
    zero_division : {"warn", 0.0, 1.0, np.nan}
        Policy, already validated.
    sample_weight : numpy.ndarray or None
        Per-query weights, already validated.
    average : {"mean", None}
        ``None`` returns the per-query values.
    metric : str
        Metric name, for the warning.
    reason : str
        Why a value is undefined, for the warning.
    stacklevel : int, default=4
        Frames between this function and the user's call, for the warning
        (public function -> ``_compute`` -> here is 4).

    Returns
    -------
    float or numpy.ndarray
        The (weighted) mean over defined values, or per-query values. The mean
        is ``NaN`` when no value is defined under ``zero_division=np.nan``.

    Raises
    ------
    ValueError
        If the weights of the averaged queries sum to zero.
    """
    undefined = np.isnan(values)
    if undefined.any():
        if zero_division == "warn":
            warnings.warn(
                f"{metric} is undefined for {int(undefined.sum())} of "
                f"{values.size} queries ({reason}) and was scored 0.0. Set "
                "zero_division to choose the value explicitly.",
                UndefinedMetricWarning,
                stacklevel=stacklevel,
            )
            values = np.where(undefined, 0.0, values)
        elif not math.isnan(zero_division):
            values = np.where(undefined, float(zero_division), values)
    if average is None:
        return values
    defined = ~np.isnan(values)
    if not defined.any():
        return float("nan")
    weights = None if sample_weight is None else sample_weight[defined]
    if weights is not None and float(weights.sum()) == 0.0:
        raise ValueError(
            f"The sample weights of the queries averaged for {metric} sum to zero."
        )
    return float(np.average(values[defined], weights=weights))


def _ident_of(item: Any) -> Any:
    """Return the identifier of one retrieval result (see :func:`ranked_ids`)."""
    if isinstance(item, str) or (
        isinstance(item, numbers.Integral) and not isinstance(item, bool)
    ):
        return item
    if isinstance(item, tuple) and len(item) == _PAIR_LENGTH:
        return item[0]
    doc = getattr(item, "doc", None)
    if doc is not None:
        return getattr(doc, "doc_id", None)
    if hasattr(item, "doc_id"):
        return item.doc_id
    raise TypeError(
        f"Cannot take an identifier from a {type(item).__name__!r} result; "
        "pass id_getter= (for example lambda hit: hit.doc.input_path)."
    )


def ranked_ids(results: Any, *, id_getter: Callable[[Any], Any] | None = None) -> list:
    """
    Return the identifiers of retrieval results, in rank order.

    Parameters
    ----------
    results : iterable
        A :class:`~scikitplot.corpus.RetrievalResponse`, or any iterable of
        results, best first.
    id_getter : callable or None, default=None
        ``id_getter(result) -> str or int``. When ``None``, each result gives
        its identifier by this fixed rule, first match wins:

        1. a ``str`` or ``int`` is the identifier;
        2. a 2-tuple ``(identifier, score)`` gives its first item (the shape of
           ``RetrievalIndex.query``);
        3. an object with ``.doc`` gives ``.doc.doc_id``
           (:class:`~scikitplot.corpus.RetrievalHit`);
        4. an object with ``.doc_id`` gives it (a document).

    Returns
    -------
    list
        Identifiers (``str`` or ``int``), possibly repeated (several chunks of
        one document); the metrics decide what a repeat means.

    Raises
    ------
    TypeError
        If ``results`` is a ``str``, ``bytes`` or mapping, or a result yields no
        ``str``/``int`` identifier.

    Examples
    --------
    >>> ranked_ids([("d2", 0.9), ("d1", 0.4)])
    ['d2', 'd1']
    >>> ranked_ids([{"id": 7}], id_getter=lambda r: r["id"])
    [7]
    """
    if isinstance(results, (str, bytes, Mapping)):
        raise TypeError(
            "results must be an iterable of retrieval results, got "
            f"{type(results).__name__!r}."
        )
    if id_getter is not None and not callable(id_getter):
        raise TypeError("id_getter must be callable or None.")
    get = _ident_of if id_getter is None else id_getter
    return [
        as_id(get(item), label=f"result {rank}") for rank, item in enumerate(results)
    ]


def response_status(results: Any) -> str | None:
    """
    Return the status value of a retrieval response, or ``None`` if it has none.

    Parameters
    ----------
    results : object
        A :class:`~scikitplot.corpus.RetrievalResponse` (its ``status`` is an
        enum whose ``value`` is returned) or any other result.

    Returns
    -------
    str or None
    """
    status = getattr(results, "status", None)
    if status is None:
        return None
    return str(getattr(status, "value", status))


def check_on_degraded(
    policy: Any, *, allowed: Sequence[str] = ON_DEGRADED_POLICIES
) -> str:
    """Validate what a degraded retrieval response does."""
    if policy not in allowed:
        raise ValueError(
            f"on_degraded must be one of {tuple(allowed)}, got {policy!r}."
        )
    return policy
