# scikitplot/corpus/_metrics/_text.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Reference-based answer metrics: exact match, token F1, edit similarity, ANLS.

Each ``*_score`` function compares predicted answers with reference answers,
query by query, with the scikit-learn signature ``score(y_true, y_pred, ...)``,
so it can be wrapped by :func:`~scikitplot.corpus.make_corpus_scorer`. A
reference may be one string or a non-empty sequence of acceptable strings; the
best-matching reference counts.

Normalisation is explicit (``normalize=``):

``None``
    The strings are compared as given.
``"basic"``
    Unicode NFKC, case folding, and whitespace collapsed to single spaces.
    Language-neutral.
``"squad"``
    The SQuAD v1.1 evaluation normalisation: lower case, ASCII punctuation
    removed, the English articles *a*, *an*, *the* removed, whitespace
    collapsed. Use it to compare with published SQuAD-style numbers.

Notes
-----
**User note.** Token F1 splits on whitespace after normalisation. For scripts
written without spaces (Chinese, Japanese, Thai), pass a ``tokenizer``.

**Developer note.** :func:`edit_similarity` uses :mod:`scikitplot.levenshtein`
when it is importable and an exact dynamic-programming fallback otherwise.
Both compute the same Levenshtein distance; a randomised differential test
holds them equal. The fallback keeps this package usable on its own, as the
submodule independence rule requires.

References
----------
.. [1] Rajpurkar, P., Zhang, J., Lopyrev, K. and Liang, P. (2016). "SQuAD:
   100,000+ Questions for Machine Comprehension of Text". EMNLP.
.. [2] Biten, A. F. et al. (2019). "Scene Text Visual Question Answering".
   ICCV. (ANLS, threshold 0.5.)
"""

from __future__ import annotations

import math
import numbers
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Sequence

import numpy as np

from ._inputs import aggregate, aligned, check_average, check_sample_weight, where

__all__ = [
    "anls_score",
    "edit_similarity",
    "edit_similarity_score",
    "exact_match_score",
    "normalize_answer",
    "token_f1_score",
]

#: Accepted values of ``normalize``.
NORMALIZE_MODES: tuple[Any, ...] = (None, "basic", "squad")

#: Accepted values of ``backend``.
EDIT_BACKENDS: tuple[str, ...] = ("auto", "python")

_SQUAD_PUNCTUATION = frozenset(string.punctuation)
_SQUAD_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str, *, mode: str | None = "squad") -> str:
    """
    Normalise an answer string for comparison.

    Parameters
    ----------
    text : str
        Answer text.
    mode : {"squad", "basic", None}, default="squad"
        See the module documentation.

    Returns
    -------
    str

    Raises
    ------
    TypeError
        If ``text`` is not a ``str``.
    ValueError
        For an unknown ``mode``.

    Examples
    --------
    >>> normalize_answer("The  Eiffel Tower!")
    'eiffel tower'
    >>> normalize_answer("Straße", mode="basic")
    'strasse'
    """
    if not isinstance(text, str):
        raise TypeError(f"An answer must be a str, got {type(text).__name__!r}.")
    if mode is None:
        return text
    if mode == "basic":
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if mode == "squad":
        lowered = text.lower()
        no_punctuation = "".join(ch for ch in lowered if ch not in _SQUAD_PUNCTUATION)
        return " ".join(_SQUAD_ARTICLES.sub(" ", no_punctuation).split())
    raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {mode!r}.")


def _check_mode(mode: Any) -> Any:
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {mode!r}.")
    return mode


def _python_distance(a: str, b: str) -> int:
    """Exact Levenshtein distance by dynamic programming over code points."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _distance_function(backend: Any) -> Callable[[str, str], int]:
    """Resolve the distance implementation once per call (not per pair)."""
    if backend not in EDIT_BACKENDS:
        raise ValueError(f"backend must be one of {EDIT_BACKENDS}, got {backend!r}.")
    if backend == "auto":
        try:
            from scikitplot.levenshtein import distance  # noqa: PLC0415
        except ImportError:
            return _python_distance
        return distance
    return _python_distance


def _similarity(a: str, b: str, distance: Callable[[str, str], int]) -> float:
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - distance(a, b) / longest


def edit_similarity(
    a: str,
    b: str,
    *,
    normalize: str | None = None,
    backend: str = "auto",
) -> float:
    """
    Return ``1 - levenshtein(a, b) / max(len(a), len(b))``.

    Parameters
    ----------
    a, b : str
        Strings to compare (Unicode code points).
    normalize : {None, "basic", "squad"}, default=None
        Normalisation applied to both strings first.
    backend : {"auto", "python"}, default="auto"
        ``"auto"`` uses :mod:`scikitplot.levenshtein` when importable;
        ``"python"`` always uses the exact fallback. Both return the same value.

    Returns
    -------
    float
        In [0, 1]; 1.0 for two empty strings.

    Raises
    ------
    TypeError
        If ``a`` or ``b`` is not a ``str``.
    ValueError
        For an unknown ``normalize`` or ``backend``.

    See Also
    --------
    make_pairwise_ranker : Rank corpus documents with this function.

    Examples
    --------
    >>> edit_similarity("kitten", "sitting")
    0.5714285714285714
    """
    mode = _check_mode(normalize)
    distance = _distance_function(backend)
    return _similarity(
        normalize_answer(a, mode=mode), normalize_answer(b, mode=mode), distance
    )


def _references(row: Any, *, label: str) -> tuple[str, ...]:
    """Return the acceptable reference answers of one item."""
    if isinstance(row, str):
        return (row,)
    if isinstance(row, Mapping) or not isinstance(row, Iterable):
        raise TypeError(
            f"{label}: a reference is a str or a sequence of str, got "
            f"{type(row).__name__!r}."
        )
    refs = tuple(row)
    if not refs:
        raise ValueError(f"{label}: at least one reference answer is required.")
    for ref in refs:
        if not isinstance(ref, str):
            raise TypeError(
                f"{label}: reference answers must be str, got {type(ref).__name__!r}."
            )
    return refs


def _prediction(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{label}: a predicted answer must be a str, got {type(value).__name__!r}."
        )
    return value


def _score_items(
    y_true: Any,
    y_pred: Any,
    pair_score: Callable[[str, str], float],
    *,
    normalize: Any,
    sample_weight: Any,
    average: Any,
    metric: str,
) -> Any:
    """Validate, normalise, take the best reference per item, then average."""
    mode = _check_mode(normalize)
    averaging = check_average(average)
    true_rows, pred_rows, keys = aligned(y_true, y_pred)
    weights = check_sample_weight(sample_weight, len(true_rows))
    values = np.empty(len(true_rows), dtype=np.float64)
    for i, (refs_row, pred_row) in enumerate(zip(true_rows, pred_rows)):
        label = where(i, None if keys is None else keys[i])
        prediction = normalize_answer(_prediction(pred_row, label=label), mode=mode)
        values[i] = max(
            pair_score(normalize_answer(ref, mode=mode), prediction)
            for ref in _references(refs_row, label=label)
        )
    return aggregate(
        values,
        zero_division=0.0,
        sample_weight=weights,
        average=averaging,
        metric=metric,
        reason="never undefined",
    )


def exact_match_score(
    y_true: Any,
    y_pred: Any,
    *,
    normalize: str | None = "squad",
    sample_weight: Any = None,
    average: str | None = "mean",
) -> Any:
    """
    Fraction of predicted answers equal to a reference after normalisation.

    Parameters
    ----------
    y_true : sequence or mapping
        Reference answers per item: a ``str`` or a sequence of ``str``.
    y_pred : sequence or mapping
        Predicted answer per item (``str``), aligned with ``y_true``.
    normalize : {"squad", "basic", None}, default="squad"
        Normalisation of both sides.
    sample_weight : array-like of shape (n_items,), default=None
        Per-item weights of the mean.
    average : {"mean", None}, default="mean"
        ``None`` returns per-item values.

    Returns
    -------
    float or numpy.ndarray

    Raises
    ------
    TypeError, ValueError
        For malformed inputs; the message names the item.

    Examples
    --------
    >>> exact_match_score(
    ...     ["The Eiffel Tower", ["Paris", "City of Paris"]], ["eiffel tower.", "Lyon"]
    ... )
    0.5
    """
    return _score_items(
        y_true,
        y_pred,
        lambda ref, pred: float(ref == pred),
        normalize=normalize,
        sample_weight=sample_weight,
        average=average,
        metric="exact_match",
    )


def _token_f1(ref_tokens: Sequence[str], pred_tokens: Sequence[str]) -> float:
    if not ref_tokens or not pred_tokens:
        return float(list(ref_tokens) == list(pred_tokens))
    common = Counter(ref_tokens) & Counter(pred_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def token_f1_score(
    y_true: Any,
    y_pred: Any,
    *,
    normalize: str | None = "squad",
    tokenizer: Callable[[str], Sequence[str]] | None = None,
    sample_weight: Any = None,
    average: str | None = "mean",
) -> Any:
    """
    Token-overlap F1 between predicted and reference answers (SQuAD style).

    Parameters
    ----------
    y_true : sequence or mapping
        Reference answers per item: a ``str`` or a sequence of ``str``.
    y_pred : sequence or mapping
        Predicted answer per item (``str``), aligned with ``y_true``.
    normalize : {"squad", "basic", None}, default="squad"
        Normalisation before tokenisation.
    tokenizer : callable or None, default=None
        ``tokenizer(text) -> sequence of str``; ``None`` splits on whitespace.
    sample_weight : array-like of shape (n_items,), default=None
        Per-item weights of the mean.
    average : {"mean", None}, default="mean"
        ``None`` returns per-item values.

    Returns
    -------
    float or numpy.ndarray
        Per item, the best F1 over its references. Two empty token lists score
        1; one empty list scores 0.

    Raises
    ------
    TypeError, ValueError
        For malformed inputs or a tokenizer that does not return strings.

    Examples
    --------
    >>> token_f1_score(["the cat sat"], ["a cat sat down"])
    0.8
    """
    if tokenizer is not None and not callable(tokenizer):
        raise TypeError("tokenizer must be callable or None.")

    def tokens(text: str) -> list[str]:
        if tokenizer is None:
            return text.split()
        out = list(tokenizer(text))
        if not all(isinstance(tok, str) for tok in out):
            raise TypeError("tokenizer must return a sequence of str.")
        return out

    return _score_items(
        y_true,
        y_pred,
        lambda ref, pred: _token_f1(tokens(ref), tokens(pred)),
        normalize=normalize,
        sample_weight=sample_weight,
        average=average,
        metric="token_f1",
    )


def edit_similarity_score(
    y_true: Any,
    y_pred: Any,
    *,
    normalize: str | None = None,
    backend: str = "auto",
    sample_weight: Any = None,
    average: str | None = "mean",
) -> Any:
    """
    Mean normalised edit similarity between predictions and references.

    Parameters
    ----------
    y_true : sequence or mapping
        Reference answers per item: a ``str`` or a sequence of ``str``.
    y_pred : sequence or mapping
        Predicted answer per item (``str``), aligned with ``y_true``.
    normalize : {None, "basic", "squad"}, default=None
        Normalisation of both sides.
    backend : {"auto", "python"}, default="auto"
        Edit-distance implementation (both exact).
    sample_weight : array-like of shape (n_items,), default=None
        Per-item weights of the mean.
    average : {"mean", None}, default="mean"
        ``None`` returns per-item values.

    Returns
    -------
    float or numpy.ndarray

    See Also
    --------
    edit_similarity, anls_score

    Examples
    --------
    >>> edit_similarity_score(["kitten"], ["sitting"])
    0.5714285714285714
    """
    distance = _distance_function(backend)
    return _score_items(
        y_true,
        y_pred,
        lambda ref, pred: _similarity(ref, pred, distance),
        normalize=normalize,
        sample_weight=sample_weight,
        average=average,
        metric="edit_similarity",
    )


def anls_score(
    y_true: Any,
    y_pred: Any,
    *,
    threshold: float = 0.5,
    normalize: str | None = "basic",
    backend: str = "auto",
    sample_weight: Any = None,
    average: str | None = "mean",
) -> Any:
    """
    Average normalised Levenshtein similarity (ANLS).

    Per item, ``s = 1 - NL`` when the normalised Levenshtein distance ``NL``
    to a reference is below ``threshold`` and 0 otherwise; the best reference
    counts.

    Parameters
    ----------
    y_true : sequence or mapping
        Reference answers per item: a ``str`` or a sequence of ``str``.
    y_pred : sequence or mapping
        Predicted answer per item (``str``), aligned with ``y_true``.
    threshold : float, default=0.5
        In (0, 1]. Distances at or above it score 0, so an answer that is
        mostly wrong earns no partial credit.
    normalize : {"basic", "squad", None}, default="basic"
        Normalisation of both sides.
    backend : {"auto", "python"}, default="auto"
        Edit-distance implementation (both exact).
    sample_weight : array-like of shape (n_items,), default=None
        Per-item weights of the mean.
    average : {"mean", None}, default="mean"
        ``None`` returns per-item values.

    Returns
    -------
    float or numpy.ndarray

    Raises
    ------
    TypeError, ValueError
        For malformed inputs or a threshold outside (0, 1].

    Examples
    --------
    >>> anls_score(["kitten", "kitten"], ["kitten", "dog"])
    0.5
    """
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise TypeError(
            f"threshold must be a real number, got {type(threshold).__name__!r}.",
        )
    tau = float(threshold)
    if not (math.isfinite(tau) and 0.0 < tau <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}.")
    distance = _distance_function(backend)

    def score(ref: str, pred: str) -> float:
        longest = max(len(ref), len(pred))
        normalized_distance = 0.0 if longest == 0 else distance(ref, pred) / longest
        return 1.0 - normalized_distance if normalized_distance < tau else 0.0

    return _score_items(
        y_true,
        y_pred,
        score,
        normalize=normalize,
        sample_weight=sample_weight,
        average=average,
        metric="anls",
    )
