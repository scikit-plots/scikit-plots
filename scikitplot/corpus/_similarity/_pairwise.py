# scikitplot/corpus/_similarity/_pairwise.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Rank corpus documents with a pairwise relevance function.

The objects here are *rankers*: called as ``ranker(query, documents, config)``
they return :class:`~scikitplot.corpus.RetrievalHit` objects, best first, which
is the ``custom_scorer_fn`` contract of
:class:`~scikitplot.corpus.CustomRetrievalIndex`. They decide what is
retrieved. Evaluating what was retrieved is the job of
:func:`~scikitplot.corpus.make_corpus_scorer`.

:func:`make_pairwise_ranker`
    Any ``score_func(query, text, **kwargs) -> float``, with scikit-learn's
    sign convention: ``greater_is_better=False`` turns a distance into a score
    by negating it, as ``make_scorer`` does for a loss.
:func:`make_levenshtein_ranker`
    Normalised Levenshtein similarity through :mod:`scikitplot.levenshtein`,
    with the backend that actually ran recorded on each hit.

Notes
-----
**User note.** Ties are broken by document position, so the same query over
the same documents always returns the same order.

**Developer note.** This module replaces ``levenshtein.make_corpus_scorer``.
The adapter lives with the retrieval code it serves, and
:mod:`scikitplot.levenshtein` no longer imports corpus. Corpus imports
:mod:`scikitplot.levenshtein` only when a Levenshtein ranker is built, as the
submodule independence rule allows.
"""

from __future__ import annotations

import functools
import heapq
import inspect
import math
import numbers
from typing import Any, Callable, Mapping, Sequence

from ._similarity import RetrievalHit, _get_text

__all__ = [
    "LevenshteinRanker",
    "PairwiseRanker",
    "make_levenshtein_ranker",
    "make_pairwise_ranker",
]


def _top_k(config: Any) -> int:
    top_k = getattr(config, "top_k", None)
    if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral) or top_k < 1:
        raise ValueError(f"config.top_k must be an int >= 1, got {top_k!r}.")
    return int(top_k)


def _check_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__!r}.")
    return value


def _func_name(func: Any) -> str:
    if isinstance(func, functools.partial):
        return repr(func)
    return getattr(func, "__name__", repr(func))


class PairwiseRanker:
    """
    Ranker built from a pairwise relevance function.

    Parameters
    ----------
    score_func : callable
        ``score_func(query, text, **kwargs) -> float``; finite values only.
    greater_is_better : bool, default=True
        ``False`` for a distance: hits are scored ``-score_func(...)``.
    use_normalized_text : bool, default=True
        Compare with a document's ``normalized_text`` when it has one.
    score_cutoff : float or None, default=None
        Drop hits whose (signed) score is below it.
    name : str or None, default=None
        Recorded as ``native_metric``; defaults to the function name.
    kwargs : mapping or None, default=None
        Fixed keyword arguments of ``score_func``.

    Notes
    -----
    Build it with :func:`make_pairwise_ranker`, which validates the arguments.
    """

    def __init__(
        self,
        score_func: Callable[..., Any],
        *,
        greater_is_better: bool = True,
        use_normalized_text: bool = True,
        score_cutoff: float | None = None,
        name: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.score_func = score_func
        self.greater_is_better = greater_is_better
        self.use_normalized_text = use_normalized_text
        self.score_cutoff = score_cutoff
        self.name = _func_name(score_func) if name is None else name
        self.kwargs = {} if kwargs is None else dict(kwargs)

    def __call__(
        self,
        query: str,
        documents: Sequence[Any],
        config: Any,
    ) -> list[RetrievalHit]:
        """
        Return the best ``config.top_k`` documents for ``query``.

        Parameters
        ----------
        query : str
            Query text.
        documents : sequence of CorpusDocument
            Candidates.
        config : RetrievalConfig
            Supplies ``top_k``.

        Returns
        -------
        list of RetrievalHit
            ``match_mode="pairwise"``, ``native_score`` the raw value,
            ``native_metric`` the function name, ``rank`` from 0.

        Raises
        ------
        TypeError
            If ``score_func`` returns something other than a real number.
        ValueError
            If it returns a non-finite value, naming the document position.
        """
        top_k = _top_k(config)
        sign = 1.0 if self.greater_is_better else -1.0
        scored = []
        for position, doc in enumerate(documents):
            value = self.score_func(
                query, _get_text(doc, self.use_normalized_text), **self.kwargs
            )
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(
                    f"{self.name} returned {type(value).__name__!r} for the document "
                    f"at position {position}; a ranker needs a real number."
                )
            try:
                value = float(value)
            except OverflowError:  # an int beyond the float range
                value = math.inf
            if not math.isfinite(value):
                raise ValueError(
                    f"{self.name} returned {value!r} for the document at position "
                    f"{position}; a ranker needs finite scores."
                )
            score = sign * value + 0.0
            if self.score_cutoff is not None and score < self.score_cutoff:
                continue
            scored.append((score, position, value))
        best = heapq.nsmallest(top_k, scored, key=lambda item: (-item[0], item[1]))
        return [
            RetrievalHit(
                doc=documents[position],
                score=score,
                match_mode="pairwise",
                native_score=value,
                native_metric=self.name,
                rank=rank,
            )
            for rank, (score, position, value) in enumerate(best)
        ]

    def __repr__(self) -> str:
        parts = [self.name]
        if not self.greater_is_better:
            parts.append("greater_is_better=False")
        if not self.use_normalized_text:
            parts.append("use_normalized_text=False")
        if self.score_cutoff is not None:
            parts.append(f"score_cutoff={self.score_cutoff!r}")
        parts.extend(f"{key}={value!r}" for key, value in self.kwargs.items())
        return f"make_pairwise_ranker({', '.join(parts)})"


def make_pairwise_ranker(
    score_func: Callable[..., Any],
    *,
    greater_is_better: bool = True,
    use_normalized_text: bool = True,
    score_cutoff: float | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> PairwiseRanker:
    """
    Make a :class:`~scikitplot.corpus.CustomRetrievalIndex` ranker from a pairwise function.

    Parameters
    ----------
    score_func : callable
        ``score_func(query, text, **kwargs) -> float``, for example
        :func:`~scikitplot.corpus.edit_similarity`, a RapidFuzz ratio or a
        cross-encoder.
    greater_is_better : bool, default=True
        ``False`` for a distance, which is negated so higher is better.
    use_normalized_text : bool, default=True
        Compare with ``normalized_text`` when a document has one.
    score_cutoff : float or None, default=None
        Minimum (signed) score of a returned hit.
    name : str or None, default=None
        Recorded as each hit's ``native_metric``.
    **kwargs : dict
        Passed to ``score_func`` on every call.

    Returns
    -------
    PairwiseRanker

    Raises
    ------
    TypeError
        For a non-callable ``score_func``, non-bool flags, or ``kwargs`` that do
        not fit its signature.
    ValueError
        For a non-finite ``score_cutoff``.

    See Also
    --------
    make_levenshtein_ranker : The Levenshtein ranker with backend provenance.

    Examples
    --------
    >>> from scikitplot.corpus import (
    ...     CorpusDocument,
    ...     CustomRetrievalIndex,
    ...     edit_similarity,
    ...     make_pairwise_ranker,
    ... )
    >>> index = CustomRetrievalIndex(
    ...     custom_scorer_fn=make_pairwise_ranker(edit_similarity)
    ... )
    >>> index.build(
    ...     [
    ...         CorpusDocument.create("a.txt", 0, "kitten"),
    ...         CorpusDocument.create("b.txt", 0, "sitting"),
    ...     ]
    ... )
    >>> [hit.doc.text for hit in index.search("kitten")]
    ['kitten', 'sitting']
    """
    if not callable(score_func):
        raise TypeError(
            f"score_func must be callable, got {type(score_func).__name__!r}."
        )
    _check_bool("greater_is_better", greater_is_better)
    _check_bool("use_normalized_text", use_normalized_text)
    if score_cutoff is not None:
        if isinstance(score_cutoff, bool) or not isinstance(score_cutoff, numbers.Real):
            raise TypeError("score_cutoff must be a real number or None.")
        if not math.isfinite(float(score_cutoff)):
            raise ValueError(f"score_cutoff must be finite, got {score_cutoff!r}.")
        score_cutoff = float(score_cutoff)
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a str or None.")
    try:
        signature = inspect.signature(score_func)
    except (TypeError, ValueError):  # builtins without a signature
        signature = None
    if signature is not None:
        try:
            signature.bind_partial("", "", **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"{_func_name(score_func)} does not accept {sorted(kwargs)}: {exc}"
            ) from exc
    return PairwiseRanker(
        score_func,
        greater_is_better=greater_is_better,
        use_normalized_text=use_normalized_text,
        score_cutoff=score_cutoff,
        name=name,
        kwargs=kwargs,
    )


class LevenshteinRanker:
    """
    Ranker by normalised Levenshtein similarity, with backend provenance.

    Parameters
    ----------
    backend : str, default="auto"
        Backend of :func:`scikitplot.levenshtein.rank`.
    strict : bool, default=False
        Raise instead of falling back when ``backend`` is unavailable.
    use_normalized_text : bool, default=True
        Compare with ``normalized_text`` when a document has one.
    score_cutoff : float, default=0.0
        Minimum similarity in [0, 1].

    Notes
    -----
    Build it with :func:`make_levenshtein_ranker`. Each hit records
    ``backend="levenshtein:<name>"``, the backend that computed its distance,
    which can differ from the requested one after a runtime fallback.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        strict: bool = False,
        use_normalized_text: bool = True,
        score_cutoff: float = 0.0,
    ) -> None:
        self.backend = backend
        self.strict = strict
        self.use_normalized_text = use_normalized_text
        self.score_cutoff = score_cutoff

    def __call__(
        self,
        query: str,
        documents: Sequence[Any],
        config: Any,
    ) -> list[RetrievalHit]:
        """
        Return the best ``config.top_k`` documents for ``query``.

        Parameters
        ----------
        query : str
            Query text.
        documents : sequence of CorpusDocument
            Candidates.
        config : RetrievalConfig
            Supplies ``top_k``.

        Returns
        -------
        list of RetrievalHit
            Ordered as :func:`scikitplot.levenshtein.rank` orders them:
            similarity, then distance, then position.
        """
        from scikitplot.levenshtein import rank  # noqa: PLC0415

        ranked = rank(
            query,
            documents,
            key=lambda doc: _get_text(doc, self.use_normalized_text),
            limit=_top_k(config),
            score_cutoff=self.score_cutoff,
            backend=self.backend,
            strict=self.strict,
        )
        return [
            RetrievalHit(
                doc=match.choice,
                score=match.similarity,
                match_mode="levenshtein",
                backend=f"levenshtein:{match.backend}",
                native_score=match.similarity,
                native_metric="normalized_levenshtein_similarity",
                rank=position,
            )
            for position, match in enumerate(ranked)
        ]

    def __repr__(self) -> str:
        return (
            f"make_levenshtein_ranker(backend={self.backend!r}, strict={self.strict!r}, "
            f"use_normalized_text={self.use_normalized_text!r}, "
            f"score_cutoff={self.score_cutoff!r})"
        )


def make_levenshtein_ranker(
    *,
    backend: str = "auto",
    strict: bool = False,
    use_normalized_text: bool = True,
    score_cutoff: float = 0.0,
) -> LevenshteinRanker:
    """
    Make a :class:`~scikitplot.corpus.CustomRetrievalIndex` ranker by Levenshtein similarity.

    Parameters
    ----------
    backend : str, default="auto"
        A backend of :func:`scikitplot.levenshtein.available_backends`.
    strict : bool, default=False
        Raise instead of falling back when ``backend`` is unavailable.
    use_normalized_text : bool, default=True
        Compare with ``normalized_text`` when a document has one.
    score_cutoff : float, default=0.0
        Minimum normalised similarity in [0, 1]. ``config.top_k`` is an upper
        bound: a cutoff may return fewer hits.

    Returns
    -------
    LevenshteinRanker

    Raises
    ------
    ImportError
        If :mod:`scikitplot.levenshtein` is not installed, or ``strict=True``
        and ``backend`` is unavailable.
    TypeError, ValueError
        For an invalid argument, including an unknown backend name.

    Notes
    -----
    The backend name is checked here, and :func:`scikitplot.levenshtein.rank`
    resolves it once per search, so an unavailable backend warns once per
    search rather than once per document.

    See Also
    --------
    make_pairwise_ranker : Any pairwise function.

    Examples
    --------
    >>> from scikitplot.corpus import (
    ...     CorpusDocument,
    ...     RetrievalConfig,
    ...     make_levenshtein_ranker,
    ... )
    >>> ranker = make_levenshtein_ranker(backend="python", score_cutoff=0.5)
    >>> docs = [
    ...     CorpusDocument.create("a.txt", 0, "colour"),
    ...     CorpusDocument.create("b.txt", 0, "cloud"),
    ... ]
    >>> [
    ...     (hit.doc.text, hit.backend)
    ...     for hit in ranker("color", docs, RetrievalConfig(top_k=5))
    ... ]
    [('colour', 'levenshtein:python')]
    """
    try:
        from scikitplot.levenshtein import backend_info  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "make_levenshtein_ranker needs scikitplot.levenshtein, which ships "
            "with the full scikit-plots distribution; use make_pairwise_ranker "
            "with scikitplot.corpus.edit_similarity instead."
        ) from exc
    if not isinstance(backend, str):
        raise TypeError("backend must be a str.")
    _check_bool("strict", strict)
    info = backend_info(backend)  # ValueError for an unknown name
    if strict and not info.available:
        raise ImportError(
            f"Levenshtein backend {backend!r} is unavailable: {info.reason}"
        )
    _check_bool("use_normalized_text", use_normalized_text)
    if isinstance(score_cutoff, bool) or not isinstance(score_cutoff, numbers.Real):
        raise TypeError("score_cutoff must be a real number.")
    if not 0.0 <= float(score_cutoff) <= 1.0:
        raise ValueError(f"score_cutoff must be within [0, 1], got {score_cutoff!r}.")
    return LevenshteinRanker(
        backend=backend,
        strict=strict,
        use_normalized_text=use_normalized_text,
        score_cutoff=float(score_cutoff),
    )
