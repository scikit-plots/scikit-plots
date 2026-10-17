# scikitplot/corpus/_metrics/_scorer.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikit-learn compatible scorers for retrieval and answer evaluation.

A scorer is called as ``scorer(estimator, X, y_true)`` and returns one number,
greater is better, exactly as a scikit-learn scorer. It is accepted wherever
scikit-learn takes ``scoring=``: :func:`~sklearn.model_selection.cross_validate`,
:class:`~sklearn.model_selection.GridSearchCV` (including a dict of scorers),
:func:`~sklearn.metrics.check_scoring`, and metadata routing through
``set_score_request``.

What differs from :func:`sklearn.metrics.make_scorer` is how predictions are
obtained, because a corpus index is not a scikit-learn estimator:

``response_method="predict"``
    ``estimator.predict(X)`` returns one prediction per query (a ranking of
    identifiers, or an answer). A scikit-learn estimator goes through
    scikit-learn's response machinery, so from scikit-learn 1.4 several scorers
    in one ``cross_validate`` share one ``predict`` call (1.3 calls it once per
    scorer).
``response_method="search"``
    ``estimator.search(query)`` is called for each query in ``X`` (the
    :class:`~scikitplot.corpus.RetrievalIndex` and
    :class:`~scikitplot.corpus.CustomRetrievalIndex` interface), and each
    response becomes a ranking through :func:`ranked_ids`. A response whose
    status says a requested leg did not run raises, unless
    ``on_degraded="ignore"``.

Notes
-----
**User note.** :func:`scorer_backend_info` reports which base class is in use.
With scikit-learn installed it is scikit-learn's ``_BaseScorer`` (checked on
1.3.2, 1.4.2, 1.5.2, 1.6.1, 1.7.2, 1.8.0 and 1.9.1); without it, a vendored
copy of the same contract, so the scorers work in a ``scikit-plots-corpus``
install that has no scikit-learn.

**Developer note.** The base is resolved on first use, never at import, so
importing corpus does not import scikit-learn. A contract probe runs the
resolved class on a fixed input before it is accepted; a scikit-learn whose
private base changed shape is replaced by the vendored base, and the reason is
recorded rather than discovered inside a cross-validation loop.
:func:`sklearn.metrics.make_scorer` itself cannot score a corpus index: its
``_Scorer._score`` calls ``is_regressor(estimator)``, which on scikit-learn
1.9.1 raises ``AttributeError`` for any object without ``__sklearn_tags__``.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ._inputs import (
    DEGRADED_STATUSES,
    check_on_degraded,
    ranked_ids,
    response_status,
)
from ._ranking import (
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    parse_metric,
    precision_at_k,
    r_precision,
    recall_at_k,
)
from ._text import anls_score, edit_similarity_score, exact_match_score, token_f1_score

logger = logging.getLogger(__name__)

__all__ = [
    "ScorerBackendInfo",
    "get_corpus_scorer",
    "get_corpus_scorer_names",
    "make_corpus_scorer",
    "scorer_backend_info",
]

#: How a scorer obtains predictions from the estimator.
RESPONSE_METHODS: tuple[str, ...] = ("predict", "search")

#: What a scorer does with a degraded retrieval response.
SCORER_ON_DEGRADED: tuple[str, ...] = ("raise", "ignore")

#: Ranking metric base name -> public function.
_RANKING_FUNCTIONS: Mapping[str, Callable[..., Any]] = {
    "hit_rate": hit_rate_at_k,
    "map": mean_average_precision,
    "mrr": mean_reciprocal_rank,
    "ndcg": ndcg_at_k,
    "precision": precision_at_k,
    "r_precision": r_precision,
    "recall": recall_at_k,
}

#: Answer metric name -> public function.
_ANSWER_FUNCTIONS: Mapping[str, Callable[..., Any]] = {
    "anls": anls_score,
    "edit_similarity": edit_similarity_score,
    "exact_match": exact_match_score,
    "token_f1": token_f1_score,
}


@dataclass(frozen=True)
class ScorerBackendInfo:
    """
    Which base class the corpus scorers use in this process.

    Parameters
    ----------
    base : {"sklearn", "vendored"}
        ``"sklearn"`` when scikit-learn's ``_BaseScorer`` passed the contract
        probe, ``"vendored"`` otherwise.
    sklearn_version : str or None
        Installed scikit-learn version, ``None`` when it is not importable.
    reason : str or None
        Why the vendored base is in use; ``None`` for ``"sklearn"``.
    """

    base: str
    sklearn_version: str | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "base": self.base,
            "sklearn_version": self.sklearn_version,
            "reason": self.reason,
        }


def _func_name(func: Any) -> str:
    """Name of a score function, as scikit-learn's scorer ``repr`` shows it."""
    if isinstance(func, functools.partial):
        return repr(func)
    return getattr(func, "__name__", repr(func))


class CorpusScorerMixin:
    """
    The corpus half of a scorer: obtain predictions, then score them.

    The concrete class combines this mixin with a base that provides
    ``__call__``, ``_warn_overlap`` and metadata routing: scikit-learn's
    ``_BaseScorer`` or the vendored copy. Attributes are those of the base
    (``_score_func``, ``_sign``, ``_kwargs``, ``_response_method``) plus
    ``_id_getter`` and ``_on_degraded``.
    """

    _score_func: Callable[..., Any]
    _sign: int
    _kwargs: dict[str, Any]
    _response_method: str
    _id_getter: Callable[[Any], Any] | None
    _on_degraded: str

    def _score(
        self, method_caller: Any, estimator: Any, X: Any, y_true: Any, **kwargs: Any
    ) -> Any:
        """
        Obtain predictions for ``X`` and score them against ``y_true``.

        Parameters
        ----------
        method_caller : callable
            ``method_caller(estimator, name, X)``; scikit-learn passes a caching
            caller when several scorers share a response method.
        estimator : object
            Has ``predict(X)`` or ``search(query)``, per ``response_method``.
        X : sequence
            Queries.
        y_true : sequence
            Ground truth per query.
        **kwargs : dict
            Metadata routed to the score function (``sample_weight`` included).

        Returns
        -------
        float
            ``sign * score_func(y_true, y_pred, **kwargs)``.
        """
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer instance and"
                " passed metadata. Please pass them either as kwargs to"
                " `make_corpus_scorer` or metadata, but not both."
            ),
            kwargs=kwargs,
        )
        y_pred = self._predictions(method_caller, estimator, X)
        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)

    def _predictions(self, method_caller: Any, estimator: Any, X: Any) -> list:
        if self._response_method == "predict":
            if not callable(getattr(estimator, "predict", None)):
                # AttributeError, as scikit-learn's _check_response_method raises
                # for a missing response method.
                raise AttributeError(  # noqa: TRY004
                    f"{type(estimator).__name__} has no predict method; use "
                    "response_method='search' for a corpus index."
                )
            if _is_sklearn_estimator(estimator):
                return method_caller(estimator, "predict", X)
            return estimator.predict(X)
        search = getattr(estimator, "search", None)
        if not callable(search):
            raise AttributeError(  # noqa: TRY004 - a missing method, as above
                f"{type(estimator).__name__} has no search method; use "
                "response_method='predict' for an estimator with predict(X)."
            )
        if isinstance(X, (str, bytes)):
            raise TypeError(
                "X is a sequence of queries; wrap a single query in a list."
            )
        rankings = []
        for position, query in enumerate(X):
            response = search(query)
            status = response_status(response)
            if status in DEGRADED_STATUSES and self._on_degraded == "raise":
                raise RuntimeError(
                    f"Query at position {position} returned a {status!r} response: "
                    "a requested retrieval leg did not run, so its ranking is "
                    "partial. Fix the retriever (for example supply query "
                    "embeddings) or pass on_degraded='ignore' to score it anyway."
                )
            rankings.append(ranked_ids(response, id_getter=self._id_getter))
        return rankings

    def __repr__(self) -> str:
        parts = [_func_name(self._score_func)]
        if self._sign < 0:
            parts.append("greater_is_better=False")
        parts.append(f"response_method={self._response_method!r}")
        if self._id_getter is not None:
            parts.append(f"id_getter={_func_name(self._id_getter)}")
        if self._on_degraded != "raise":
            parts.append(f"on_degraded={self._on_degraded!r}")
        parts.extend(f"{key}={value!r}" for key, value in self._kwargs.items())
        return f"make_corpus_scorer({', '.join(parts)})"


def _is_sklearn_estimator(estimator: Any) -> bool:
    """
    Whether scikit-learn's response machinery accepts ``estimator``.

    ``False`` whenever ``sklearn.base`` cannot be imported, for any reason: a
    broken installation (an ABI mismatch raises ``ValueError``) leaves the
    estimator to be called directly instead of failing the score.
    """
    try:
        from sklearn.base import BaseEstimator  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return isinstance(estimator, BaseEstimator)


#: What the contract probe must return: plain, then with sample weights.
_PROBE_EXPECTED = (-0.5, -0.75)


class _ProbeIndex:
    """Fixed estimator for the contract probe: ``predict`` and ``search``."""

    def predict(self, X: Any) -> list:
        return [[query] for query in X]

    def search(self, query: Any) -> list:
        return [query]


def _probe_score(y_true: Any, y_pred: Any, sample_weight: Any = None) -> float:
    hits = [float(t in p) for t, p in zip(y_true, y_pred)]
    weights = [1.0] * len(hits) if sample_weight is None else list(sample_weight)
    return sum(h * w for h, w in zip(hits, weights)) / sum(weights)


def _probe_cases() -> list[tuple[str, Any]]:
    """Response methods and estimators the contract probe runs."""
    cases: list[tuple[str, Any]] = [
        ("predict", _ProbeIndex()),
        ("search", _ProbeIndex()),
    ]
    try:
        from sklearn.base import BaseEstimator  # noqa: PLC0415
    except ImportError:
        return cases

    class _SklearnProbe(BaseEstimator):
        """A scikit-learn estimator: ``predict`` goes through ``method_caller``."""

        def predict(self, X: Any) -> list:
            return [[query] for query in X]

    cases.append(("predict", _SklearnProbe()))
    return cases


def _contract_problem(cls: type) -> str | None:
    """Run ``cls`` on fixed inputs; return what failed, or ``None``."""
    try:
        for method, estimator in _probe_cases():
            scorer = cls(
                _probe_score, -1, {}, method, id_getter=None, on_degraded="raise"
            )
            plain = scorer(estimator, ["a", "b"], ["a", "x"])
            weighted = scorer(
                estimator, ["a", "b"], ["a", "x"], sample_weight=[3.0, 1.0]
            )
            if (plain, weighted) != _PROBE_EXPECTED:
                return (
                    f"contract probe ({method}, {type(estimator).__name__}) returned "
                    f"{plain!r} and {weighted!r} instead of -0.5 and -0.75"
                )
    except Exception as exc:  # noqa: BLE001 - every failure is the finding
        return f"contract probe raised {type(exc).__name__}: {exc}"
    return None


_RESOLVED: tuple[type, ScorerBackendInfo] | None = None
_RESOLVE_LOCK = threading.Lock()


def _resolve() -> tuple[type, ScorerBackendInfo]:
    """
    Resolve the scorer class once per process (see the module notes).

    Notes
    -----
    The first call holds a lock while it imports scikit-learn and runs the
    contract probe, so racing threads neither repeat the probe nor log the
    fallback twice; later calls read the stored result.
    """
    global _RESOLVED  # noqa: PLW0603 - the one process-wide resolution
    resolved = _RESOLVED
    if resolved is None:
        with _RESOLVE_LOCK:
            if _RESOLVED is None:
                _RESOLVED = _compute_resolution()
            resolved = _RESOLVED
    return resolved


def _compute_resolution() -> tuple[type, ScorerBackendInfo]:
    """Choose scikit-learn's base when it is usable, the vendored one otherwise."""
    try:
        import sklearn  # noqa: PLC0415
    except ImportError as exc:
        from ._sklearn_compat import CorpusScorer as Vendored  # noqa: PLC0415

        return Vendored, ScorerBackendInfo(
            "vendored", None, f"scikit-learn is not importable: {exc}"
        )
    # a broken installation is recorded, not fatal
    except Exception as exc:  # noqa: BLE001
        version: str | None = None
        reason = f"importing scikit-learn raised {type(exc).__name__}: {exc}"
    else:
        version = str(getattr(sklearn, "__version__", None))
        try:
            from ._sklearn_scorer import CorpusScorer as Native  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 - any failure here is the finding
            reason = (
                "scikit-learn's private _BaseScorer is not usable "
                f"({type(exc).__name__}: {exc})"
            )
        else:
            problem = _contract_problem(Native)
            if problem is None:
                return Native, ScorerBackendInfo("sklearn", version, None)
            reason = f"scikit-learn's _BaseScorer failed the {problem}"
    logger.warning("Corpus scorers use the vendored base: %s.", reason)
    from ._sklearn_compat import CorpusScorer as Vendored  # noqa: PLC0415

    return Vendored, ScorerBackendInfo("vendored", version, reason)


def scorer_backend_info() -> ScorerBackendInfo:
    """
    Report which base class corpus scorers use in this process.

    Returns
    -------
    ScorerBackendInfo

    Examples
    --------
    >>> scorer_backend_info().base in {"sklearn", "vendored"}
    True
    """
    return _resolve()[1]


def make_corpus_scorer(
    score_func: Callable[..., Any],
    *,
    response_method: str = "predict",
    greater_is_better: bool = True,
    id_getter: Callable[[Any], Any] | None = None,
    on_degraded: str = "raise",
    **kwargs: Any,
) -> Any:
    """
    Make a scikit-learn compatible scorer from a corpus metric.

    Parameters
    ----------
    score_func : callable
        ``score_func(y_true, y_pred, **kwargs) -> float``, for example
        :func:`recall_at_k` or :func:`token_f1_score`.
    response_method : {"predict", "search"}, default="predict"
        How predictions are obtained (see the module documentation).
    greater_is_better : bool, default=True
        ``False`` for a loss; the scorer then negates it, as scikit-learn does.
    id_getter : callable or None, default=None
        ``response_method="search"`` only: ``id_getter(result)`` maps a search
        result to the identifier used by ``y_true`` (for example a chunk to its
        source document). ``None`` applies the rule of :func:`ranked_ids`.
    on_degraded : {"raise", "ignore"}, default="raise"
        ``response_method="search"`` only: what a response whose requested leg
        did not run does.
    **kwargs : dict
        Passed to ``score_func`` on every call, such as ``k=10``.

    Returns
    -------
    scorer
        A ``_BaseScorer`` subclass instance (scikit-learn's, or the vendored
        copy; see :func:`scorer_backend_info`). It pickles.

    Raises
    ------
    TypeError
        If ``score_func`` or ``id_getter`` is not callable, or ``kwargs`` do not
        fit the signature of ``score_func``.
    ValueError
        For an unknown ``response_method`` or ``on_degraded``, or ``id_getter``
        with ``response_method="predict"``.

    See Also
    --------
    get_corpus_scorer : The same from a name such as ``"ndcg@10"``.
    sklearn.metrics.make_scorer : The scikit-learn original.

    Examples
    --------
    >>> from scikitplot.corpus import make_corpus_scorer, recall_at_k
    >>> scorer = make_corpus_scorer(recall_at_k, response_method="search", k=2)
    >>> class Index:
    ...     def search(self, query):
    ...         return [("d1", 0.9), ("d2", 0.5)]
    >>> scorer(Index(), ["first query", "second query"], [["d2"], ["d3"]])
    0.5
    """
    if not callable(score_func):
        raise TypeError(
            f"score_func must be callable, got {type(score_func).__name__!r}."
        )
    if response_method not in RESPONSE_METHODS:
        raise ValueError(
            f"response_method must be one of {RESPONSE_METHODS}, got {response_method!r}."
        )
    if not isinstance(greater_is_better, bool):
        raise TypeError("greater_is_better must be a bool.")
    if id_getter is not None:
        if not callable(id_getter):
            raise TypeError("id_getter must be callable or None.")
        if response_method != "search":
            raise ValueError("id_getter applies to response_method='search' only.")
    check_on_degraded(on_degraded, allowed=SCORER_ON_DEGRADED)
    try:
        signature = inspect.signature(score_func)
    except (TypeError, ValueError):  # builtins without a signature
        signature = None
    if signature is not None:
        try:
            signature.bind_partial(None, None, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"{_func_name(score_func)} does not accept {sorted(kwargs)}: {exc}"
            ) from exc
    cls = _resolve()[0]
    return cls(
        score_func,
        1 if greater_is_better else -1,
        dict(kwargs),
        response_method,
        id_getter=id_getter,
        on_degraded=on_degraded,
    )


def get_corpus_scorer_names() -> list[str]:
    """
    Return the names :func:`get_corpus_scorer` accepts.

    Returns
    -------
    list of str
        Sorted. Ranking names also take a cutoff, ``"<name>@<k>"``, except
        ``r_precision``.

    Examples
    --------
    >>> "ndcg" in get_corpus_scorer_names()
    True
    """
    return sorted([*_RANKING_FUNCTIONS, *_ANSWER_FUNCTIONS])


def get_corpus_scorer(
    name: str,
    *,
    response_method: str = "predict",
    id_getter: Callable[[Any], Any] | None = None,
    on_degraded: str = "raise",
    **kwargs: Any,
) -> Any:
    """
    Return a new scorer for a metric name.

    Parameters
    ----------
    name : str
        A name from :func:`get_corpus_scorer_names`; ranking names take an
        optional cutoff, ``"recall@10"``. A ranking name without a cutoff uses
        the whole ranking.
    response_method : {"predict", "search"}, default="predict"
        See :func:`make_corpus_scorer`.
    id_getter : callable or None, default=None
        See :func:`make_corpus_scorer`.
    on_degraded : {"raise", "ignore"}, default="raise"
        See :func:`make_corpus_scorer`.
    **kwargs : dict
        Further arguments of the metric, such as ``zero_division``.

    Returns
    -------
    scorer
        A new instance on every call, as :func:`sklearn.metrics.get_scorer`.

    Raises
    ------
    ValueError
        For an unknown name, or ``k`` given both in the name and as a keyword.

    Examples
    --------
    >>> get_corpus_scorer("ndcg@10", response_method="search")
    make_corpus_scorer(ndcg_at_k, response_method='search', k=10)
    """
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, got {type(name).__name__!r}.")
    if name in _ANSWER_FUNCTIONS:
        score_func = _ANSWER_FUNCTIONS[name]
        options = dict(kwargs)
    else:
        try:
            base, k = parse_metric(name)
        except ValueError:
            raise ValueError(
                f"{name!r} is not a corpus scorer name; use one of "
                f"{get_corpus_scorer_names()} (ranking names take '@k')."
            ) from None
        if "k" in kwargs:
            if base == "r_precision":
                raise ValueError("r_precision takes no cutoff; do not pass k.")
            if k is not None:
                raise ValueError(
                    f"k is given by the name {name!r}; do not pass it again."
                )
            raise ValueError(
                f"Give the cutoff in the name, as {base!r} + '@<k>' "
                f"(for example {base}@{kwargs['k']!r}), not as k=."
            )
        score_func = _RANKING_FUNCTIONS[base]
        options = dict(kwargs) if base == "r_precision" else {"k": k, **kwargs}
    return make_corpus_scorer(
        score_func,
        response_method=response_method,
        id_getter=id_getter,
        on_degraded=on_degraded,
        **options,
    )
