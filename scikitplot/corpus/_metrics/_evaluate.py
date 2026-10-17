# scikitplot/corpus/_metrics/_evaluate.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Retrieval evaluation harness, exportable reports and paired comparison.

:func:`evaluate_retrieval` runs a retriever over a query set, times each
query, and scores every ranking against the relevance judgments with the
formulas of :mod:`._ranking`. The result, :class:`RetrievalEvaluation`, keeps
the aggregate metrics, each query's metrics and ranking, the latency summary,
the response statuses and the settings, and writes them to JSON, JSON Lines
(one row per query), YAML or TOML.

:meth:`RetrievalEvaluation.compare` pairs two evaluations of the same queries
(two retrievers, two chunkers, a change against a baseline) and reports, per
metric, the mean difference, wins, ties and losses, and a paired sign-flip
randomisation p-value.

Notes
-----
**User note.** A response whose status says a requested leg did not run (for
example a hybrid search without a query embedding) is a partial ranking. The
harness raises on it by default, because metrics computed from it describe a
different system than the one configured; ``on_degraded="warn"`` evaluates it
anyway and records the statuses in the report.

**Developer note.** Latency is measured around obtaining the ranking (the
retriever call and reading its results, which is where a generator does its
work), with ``time.perf_counter`` unless a ``clock`` is injected. It is a measurement, not
a deterministic output: two runs agree on every metric and ranking, not on
latency.

References
----------
.. [1] Smucker, M. D., Allan, J. and Carterette, B. (2007). "A comparison of
   statistical significance tests for information retrieval evaluation".
   CIKM.
.. [2] Phipson, B. and Smyth, G. K. (2010). "Permutation p-values should never
   be zero". Statistical Applications in Genetics and Molecular Biology, 9(1).
"""

from __future__ import annotations

import math
import numbers
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .. import _serialize
from ._inputs import (
    DEGRADED_STATUSES,
    LISTED_KEYS,
    aggregate,
    as_id,
    check_duplicates,
    check_id_types,
    check_on_degraded,
    check_relevance_threshold,
    check_zero_division,
    judgments,
    ranked_ids,
    ranking,
    response_status,
    where,
)
from ._ranking import check_gain, parse_metric, per_query, undefined_reason

__all__ = [
    "EvaluationComparison",
    "MetricComparison",
    "QueryEvaluation",
    "RetrievalEvaluation",
    "evaluate_retrieval",
]

#: Metrics :func:`evaluate_retrieval` reports by default.
DEFAULT_METRICS: tuple[str, ...] = (
    "hit_rate@10",
    "precision@10",
    "recall@10",
    "mrr@10",
    "ndcg@10",
)

#: Schema identifier and version written into every report.
SCHEMA = "scikitplot.corpus.retrieval_evaluation"
SCHEMA_VERSION = 1
COMPARISON_SCHEMA = "scikitplot.corpus.retrieval_comparison"

#: Largest number of query pairs whose sign flips are enumerated exactly.
EXACT_SIGN_FLIP_LIMIT = 16

_UNREPORTED = "unreported"


def _zero_division_label(value: Any) -> Any:
    """Return the serialisable form of a validated ``zero_division``."""
    return "nan" if isinstance(value, float) and math.isnan(value) else value


def _contains_none(value: Any) -> bool:
    """Whether a JSON-like tree holds ``None`` anywhere."""
    if value is None:
        return True
    if isinstance(value, Mapping):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_none(item) for item in value)
    return False


def _float_or_nan(value: Any) -> float:
    """Read a metric value back: ``None`` (JSON null) is an undefined value."""
    return math.nan if value is None else float(value)


@dataclass(frozen=True)
class QueryEvaluation:
    """
    One query of a :class:`RetrievalEvaluation`.

    Parameters
    ----------
    query_id : str or int
        Query identifier.
    metrics : mapping
        Metric name to value; ``NaN`` when undefined under
        ``zero_division=np.nan``.
    n_retrieved : int
        Ranking length after duplicate handling.
    n_relevant : int
        Relevant identifiers in the judgments.
    latency_s : float
        Seconds to obtain the ranking: the retriever call and the consumption
        of its results (a lazy retriever does its work while they are read).
    status : str or None, default=None
        Response status (``"success"``, ``"empty"``, ``"degraded"`` ...), or
        ``None`` when the retriever returned none.
    retrieved : tuple or None, default=None
        The ranking, when kept.
    """

    query_id: Any
    metrics: Mapping[str, float]
    n_retrieved: int
    n_relevant: int
    latency_s: float
    status: str | None = None
    retrieved: tuple | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a mapping in the report schema."""
        return {
            "query_id": self.query_id,
            "metrics": dict(self.metrics),
            "n_retrieved": self.n_retrieved,
            "n_relevant": self.n_relevant,
            "latency_s": self.latency_s,
            "status": self.status,
            "retrieved": None if self.retrieved is None else list(self.retrieved),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QueryEvaluation:
        """
        Rebuild from :meth:`to_dict` output (``None`` metric values are ``NaN``).

        Raises
        ------
        KeyError
            If a required field is missing.
        """
        retrieved = data.get("retrieved")
        return cls(
            query_id=data["query_id"],
            metrics={k: _float_or_nan(v) for k, v in data["metrics"].items()},
            n_retrieved=int(data["n_retrieved"]),
            n_relevant=int(data["n_relevant"]),
            latency_s=float(data["latency_s"]),
            status=data.get("status"),
            retrieved=None if retrieved is None else tuple(retrieved),
        )


@dataclass(frozen=True)
class MetricComparison:
    """
    One metric of an :class:`EvaluationComparison`.

    Parameters
    ----------
    metric : str
        Metric name.
    candidate, baseline : float
        Means over the paired queries.
    delta : float
        ``candidate - baseline``.
    wins, ties, losses : int
        Queries where the candidate is higher, equal, lower.
    n_pairs : int
        Queries where both values are defined.
    p_value : float
        Two-sided paired sign-flip randomisation p-value; ``NaN`` without pairs.
    """

    metric: str
    candidate: float
    baseline: float
    delta: float
    wins: int
    ties: int
    losses: int
    n_pairs: int
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping (``NaN`` stays a float)."""
        return {
            "metric": self.metric,
            "candidate": self.candidate,
            "baseline": self.baseline,
            "delta": self.delta,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "n_pairs": self.n_pairs,
            "p_value": self.p_value,
        }


@dataclass(frozen=True)
class EvaluationComparison:
    """
    Paired comparison of two evaluations of the same queries.

    Parameters
    ----------
    metrics : tuple of MetricComparison
        One entry per metric both evaluations report, in the candidate's order.
    n_queries : int
        Queries (or answer items) in both evaluations.
    n_resamples : int
        Random sign flips drawn when the pairs exceed
        :data:`EXACT_SIGN_FLIP_LIMIT` (otherwise every flip is enumerated).
    seed : int
        Seed of those draws.
    """

    metrics: tuple
    n_queries: int
    n_resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a mapping in the comparison schema."""
        return {
            "schema": COMPARISON_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "method": "paired sign-flip randomisation test, two-sided",
            "n_queries": self.n_queries,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "metrics": [item.to_dict() for item in self.metrics],
        }

    def dumps(self, format: str = "json") -> str:  # noqa: A002
        """Serialise to ``"json"``, ``"jsonl"`` (one row per metric), ``"yaml"`` or ``"toml"``."""
        if format == "jsonl":
            return _serialize.dumps([item.to_dict() for item in self.metrics], "jsonl")
        return _serialize.dumps(self.to_dict(), format)

    def write(self, path: Any, format: str | None = None) -> Any:  # noqa: A002
        """Write atomically; the format defaults to the path suffix."""
        fmt = _serialize.resolve_format(path, format)
        return _serialize.write(path, self._payload(fmt), fmt)

    def _payload(self, fmt: str) -> Any:
        if fmt == "jsonl":
            return [item.to_dict() for item in self.metrics]
        return self.to_dict()


@dataclass(frozen=True)
class RetrievalEvaluation:
    """
    Result of :func:`evaluate_retrieval`.

    Parameters
    ----------
    metrics : mapping
        Metric name to the mean over queries (undefined values excluded under
        ``zero_division=np.nan``).
    queries : tuple of QueryEvaluation
        Per-query results, in evaluation order.
    settings : mapping
        The options the evaluation ran with.
    metadata : mapping, default={}
        Free-form, JSON-compatible description supplied by the caller.

    Notes
    -----
    **User note.** :meth:`write` picks the format from the file suffix:
    ``.json``, ``.jsonl``/``.ndjson`` (one row per query), ``.yaml``/``.yml``
    or ``.toml``. :meth:`read` restores a report from JSON, YAML or TOML.
    """

    metrics: Mapping[str, float]
    queries: tuple
    settings: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_queries(self) -> int:
        """Number of evaluated queries."""
        return len(self.queries)

    @property
    def latency(self) -> dict[str, float]:
        """Latency summary in seconds: mean, p50, p95, max and total."""
        values = np.array([q.latency_s for q in self.queries], dtype=np.float64)
        if values.size == 0:
            return {
                "mean": math.nan,
                "p50": math.nan,
                "p95": math.nan,
                "max": math.nan,
                "total": 0.0,
            }
        return {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
            "total": float(values.sum()),
        }

    @property
    def status_counts(self) -> dict[str, int]:
        """Number of queries per response status (``"unreported"`` for none)."""
        counts: dict[str, int] = {}
        for query in self.queries:
            key = _UNREPORTED if query.status is None else query.status
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def per_query(self, metric: str) -> np.ndarray:
        """
        Return one metric's per-query values, in evaluation order.

        Raises
        ------
        KeyError
            If the evaluation did not compute ``metric``.
        """
        if metric not in self.metrics:
            raise KeyError(f"{metric!r} was not evaluated; have {list(self.metrics)}.")
        return np.array([q.metrics[metric] for q in self.queries], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Return the full report as a mapping (schema versioned)."""
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "n_queries": self.n_queries,
            "metrics": dict(self.metrics),
            "latency_s": self.latency,
            "status_counts": self.status_counts,
            "settings": dict(self.settings),
            "metadata": dict(self.metadata),
            "queries": [query.to_dict() for query in self.queries],
        }

    def to_rows(self) -> list[dict[str, Any]]:
        """
        Return one flat row per query: identifier, each metric, counts, status.

        Returns
        -------
        list of dict
            The JSON Lines payload.
        """
        rows = []
        for query in self.queries:
            row: dict[str, Any] = {"query_id": query.query_id}
            row.update(query.metrics)
            row.update(
                n_retrieved=query.n_retrieved,
                n_relevant=query.n_relevant,
                latency_s=query.latency_s,
                status=query.status,
            )
            if query.retrieved is not None:
                row["retrieved"] = list(query.retrieved)
            rows.append(row)
        return rows

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalEvaluation:
        """
        Rebuild a report from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If the schema or its version is not this one.
        """
        if data.get("schema") != SCHEMA:
            raise ValueError(
                f"Not a retrieval evaluation report: schema {data.get('schema')!r}."
            )
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Report schema version {data.get('schema_version')!r} is not "
                f"{SCHEMA_VERSION}; re-run the evaluation."
            )
        settings = dict(data.get("settings", {}))
        return cls(
            metrics={k: _float_or_nan(v) for k, v in data["metrics"].items()},
            queries=tuple(QueryEvaluation.from_dict(q) for q in data["queries"]),
            settings=settings,
            metadata=dict(data.get("metadata", {})),
        )

    def dumps(self, format: str = "json") -> str:  # noqa: A002
        """Serialise to ``"json"``, ``"jsonl"``, ``"yaml"`` or ``"toml"``."""
        return _serialize.dumps(self._payload(format), format)

    def _payload(self, fmt: str) -> Any:
        if fmt == "toml" and _contains_none(self.metadata):
            raise ValueError(
                "metadata holds None, which TOML cannot represent and would drop; "
                "remove those entries or write json or yaml."
            )
        return self.to_rows() if fmt == "jsonl" else self.to_dict()

    def write(self, path: Any, format: str | None = None) -> Any:  # noqa: A002
        """
        Write the report atomically.

        Parameters
        ----------
        path : str or os.PathLike
            Destination; its suffix names the format unless ``format`` does.
        format : {"json", "jsonl", "yaml", "toml"} or None, default=None
            Explicit format.

        Returns
        -------
        pathlib.Path
        """
        fmt = _serialize.resolve_format(path, format)
        return _serialize.write(path, self._payload(fmt), fmt)

    @classmethod
    def read(
        cls,
        path: Any,
        format: str | None = None,
    ) -> RetrievalEvaluation:  # noqa: A002
        """
        Read a report written by :meth:`write` in JSON, YAML or TOML.

        Raises
        ------
        ValueError
            For JSON Lines, which holds the per-query rows only.
        """
        fmt = _serialize.resolve_format(path, format)
        if fmt == "jsonl":
            raise ValueError(
                "A JSON Lines file holds per-query rows, not a full report; "
                "read it with scikitplot.corpus._serialize.read, or write the "
                "report as json, yaml or toml."
            )
        return cls.from_dict(_serialize.read(path, fmt))

    def compare(
        self,
        baseline: RetrievalEvaluation,
        *,
        n_resamples: int = 10_000,
        seed: int = 0,
    ) -> EvaluationComparison:
        """
        Compare this evaluation with ``baseline`` over the same queries.

        Parameters
        ----------
        baseline : RetrievalEvaluation
            Evaluation of the same query identifiers.
        n_resamples : int, default=10000
            Random sign flips when the pairs exceed
            :data:`EXACT_SIGN_FLIP_LIMIT`; below it every flip is enumerated.
        seed : int, default=0
            Non-negative seed of the random flips, so the p-value is
            reproducible.

        Returns
        -------
        EvaluationComparison

        Raises
        ------
        TypeError
            If ``baseline`` is not a :class:`RetrievalEvaluation`.
        ValueError
            If the query sets differ or no metric is shared.

        Notes
        -----
        Per metric, only queries defined in both evaluations are paired. The
        statistic is ``|sum(d)|`` over the per-query differences ``d`` (the
        mean up to the constant number of pairs); the p-value counts sign flips
        whose statistic reaches the observed one (``+1`` in numerator and
        denominator when flips are sampled). A flip within
        ``4 * n * eps * sum(|d|)`` of the observed statistic counts as reaching
        it, so flips equal for the values the metrics represent are not lost to
        floating-point rounding; the p-value errs on the conservative side.
        """
        if not isinstance(baseline, RetrievalEvaluation):
            raise TypeError("baseline must be a RetrievalEvaluation.")
        return paired_comparison(
            [(q.query_id, q.metrics) for q in self.queries],
            [(q.query_id, q.metrics) for q in baseline.queries],
            list(self.metrics),
            list(baseline.metrics),
            n_resamples=n_resamples,
            seed=seed,
            unit="queries",
        )


def paired_comparison(
    candidate: list,
    baseline: list,
    candidate_metrics: list,
    baseline_metrics: list,
    *,
    n_resamples: int,
    seed: int,
    unit: str,
) -> EvaluationComparison:
    """
    Paired comparison shared by every evaluation report.

    Parameters
    ----------
    candidate, baseline : list of (identifier, mapping)
        Each unit's identifier and metric values, in the candidate's order.
    candidate_metrics, baseline_metrics : list of str
        Metric names each report holds, in report order.
    n_resamples, seed : int
        See :meth:`RetrievalEvaluation.compare`.
    unit : str
        Plural noun for messages (``"queries"``, ``"items"``).

    Returns
    -------
    EvaluationComparison

    Raises
    ------
    ValueError
        If the identifier sets differ, no metric is shared, or
        ``n_resamples``/``seed`` are out of range.
    TypeError
        If ``seed`` is not an int.
    """
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, numbers.Integral)
        or n_resamples < 1
    ):
        raise ValueError(f"n_resamples must be an int >= 1, got {n_resamples!r}.")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an int, got {type(seed).__name__!r}.")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}.")
    mine = dict(candidate)
    theirs = dict(baseline)
    if set(mine) != set(theirs):
        only_mine = [k for k in mine if k not in theirs][:LISTED_KEYS]
        only_theirs = [k for k in theirs if k not in mine][:LISTED_KEYS]
        raise ValueError(
            f"Both evaluations must cover the same {unit}; only in this one: "
            f"{only_mine!r}, only in the baseline: {only_theirs!r}."
        )
    shared = [name for name in candidate_metrics if name in set(baseline_metrics)]
    if not shared:
        raise ValueError("The evaluations share no metric.")
    order = [key for key, _ in candidate]
    rows = []
    for name in shared:
        cand = np.array([mine[k][name] for k in order], dtype=np.float64)
        base = np.array([theirs[k][name] for k in order], dtype=np.float64)
        both = ~(np.isnan(cand) | np.isnan(base))
        cand, base = cand[both], base[both]
        n_pairs = int(both.sum())
        if n_pairs == 0:
            rows.append(
                MetricComparison(
                    name, math.nan, math.nan, math.nan, 0, 0, 0, 0, math.nan
                )
            )
            continue
        diff = cand - base
        rows.append(
            MetricComparison(
                metric=name,
                candidate=float(cand.mean()),
                baseline=float(base.mean()),
                delta=float(diff.mean()),
                wins=int((diff > 0).sum()),
                ties=int((diff == 0).sum()),
                losses=int((diff < 0).sum()),
                n_pairs=n_pairs,
                p_value=_sign_flip_p_value(diff, int(n_resamples), int(seed)),
            )
        )
    return EvaluationComparison(
        metrics=tuple(rows),
        n_queries=len(order),
        n_resamples=int(n_resamples),
        seed=int(seed),
    )


def _sign_flip_p_value(diff: np.ndarray, n_resamples: int, seed: int) -> float:
    """
    Two-sided paired sign-flip randomisation p-value of ``sum(diff)``.

    The statistic is ``|sum(s * diff)|`` (the mean up to the constant ``n``).
    A flip reaches the observed statistic when it is at least the observed
    value minus ``4 * n * eps * sum(|diff|)``: the rounding error of a sum of
    ``n`` terms scales with the terms, not with the result, so two flips equal
    for the values the metrics represent (``0.3 + 0.4`` and ``0.7``) both
    count. Counting near-ties as reaching makes the p-value conservative.
    """
    n = diff.size
    observed = abs(math.fsum(diff))
    if observed == 0.0:
        return 1.0
    reach = observed - 4.0 * n * np.finfo(np.float64).eps * float(np.abs(diff).sum())
    if n <= EXACT_SIGN_FLIP_LIMIT:
        codes = np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)
        signs = (codes & 1) * 2.0 - 1.0
        stats = np.abs((signs * diff).sum(axis=1))
        return float(np.count_nonzero(stats >= reach) / stats.size)
    rng = np.random.default_rng(seed)
    count = 0
    remaining = n_resamples
    batch = max(1, min(n_resamples, 1_000_000 // n))
    while remaining:
        size = min(batch, remaining)
        signs = rng.integers(0, 2, size=(size, n), dtype=np.int8) * 2.0 - 1.0
        stats = np.abs((signs * diff).sum(axis=1))
        count += int(np.count_nonzero(stats >= reach))
        remaining -= size
    return float((count + 1) / (n_resamples + 1))


def _call_retriever(retriever: Any) -> Callable[[Any], Any]:
    """Return how to query ``retriever``: its ``search`` method, else itself."""
    search = getattr(retriever, "search", None)
    if callable(search):
        return search
    if callable(retriever):
        return retriever
    raise TypeError(
        "retriever must have a search(query) method or be callable as "
        f"retriever(query); got {type(retriever).__name__!r}."
    )


def _check_metric_names(metrics: Any) -> tuple[list[str], list[tuple[str, str, Any]]]:
    """Validate metric names; return them and their parsed specs."""
    if isinstance(metrics, str):
        raise TypeError('metrics is a collection of names; wrap one name as ["mrr"].')
    names = list(metrics)
    if not names:
        raise ValueError("metrics must name at least one metric.")
    if len(set(names)) != len(names):
        raise ValueError(f"metrics names a metric twice: {names!r}.")
    return names, [(name, *parse_metric(name)) for name in names]


def _check_query_sets(queries: Any, qrels: Any) -> list:
    """Require two mappings with the same keys; return the evaluation order."""
    if not isinstance(queries, Mapping) or not isinstance(qrels, Mapping):
        raise TypeError("queries and qrels must be mappings keyed by query identifier.")
    order = list(qrels)
    if not order:
        raise ValueError("qrels must judge at least one query.")
    missing = [k for k in order if k not in queries]
    extra = [k for k in queries if k not in qrels]
    if missing or extra:
        raise ValueError(
            "queries and qrels must have the same keys; without a query text: "
            f"{missing[:LISTED_KEYS]!r}, without judgments: {extra[:LISTED_KEYS]!r}."
        )
    for key in order:
        as_id(key, label="query identifier")
    return order


def evaluate_retrieval(
    retriever: Any,
    queries: Mapping[Any, Any],
    qrels: Mapping[Any, Any],
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    id_getter: Callable[[Any], Any] | None = None,
    relevance_threshold: float = 1,
    duplicates: str = "raise",
    zero_division: Any = "warn",
    gain: str = "linear",
    on_degraded: str = "raise",
    keep_rankings: bool = True,
    metadata: Mapping[str, Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> RetrievalEvaluation:
    """
    Run ``retriever`` on every query and score the rankings.

    Parameters
    ----------
    retriever : object
        Anything with ``search(query)`` (``RetrievalIndex``,
        ``CustomRetrievalIndex``) or a callable ``retriever(query)``. The
        result is read by :func:`ranked_ids`.
    queries : mapping
        Query identifier to query (usually text), passed to the retriever.
    qrels : mapping
        Query identifier to judgments (``{doc_id: grade}`` or a collection of
        relevant identifiers), with exactly the keys of ``queries``. Its order
        is the evaluation order.
    metrics : iterable of str, default=DEFAULT_METRICS
        Names as in :func:`ranking_metrics`.
    id_getter : callable or None, default=None
        Maps a result to its identifier (see :func:`ranked_ids`).
    relevance_threshold : float, default=1
        Grade from which a judged identifier is relevant.
    duplicates : {"raise", "first"}, default="raise"
        What a repeated identifier in a ranking does; ``"first"`` when several
        chunks map to one document through ``id_getter``.
    zero_division : {"warn", 0.0, 1.0, np.nan}, default="warn"
        Value of an undefined metric.
    gain : {"linear", "exponential"}, default="linear"
        nDCG gain.
    on_degraded : {"raise", "warn", "ignore"}, default="raise"
        What a degraded or failed response does (see the module notes).
    keep_rankings : bool, default=True
        Keep each ranking in the report, so it can be re-scored later.
    metadata : mapping or None, default=None
        Description stored with the report (system name, corpus version...).
    clock : callable or None, default=None
        ``clock() -> seconds``; ``None`` uses :func:`time.perf_counter`.

    Returns
    -------
    RetrievalEvaluation

    Raises
    ------
    TypeError
        For a retriever that cannot be called, or malformed inputs.
    ValueError
        If ``queries`` and ``qrels`` have different keys, a ranking repeats an
        identifier under ``duplicates="raise"``, or a response is degraded
        under ``on_degraded="raise"``.

    See Also
    --------
    ranking_metrics : The same metrics over precomputed rankings.
    make_corpus_scorer : One metric as a scikit-learn scorer.

    Examples
    --------
    >>> from scikitplot.corpus import evaluate_retrieval
    >>> def retriever(query):
    ...     return [("d1", 0.9), ("d2", 0.4)] if "cat" in query else [("d3", 0.7)]
    >>> report = evaluate_retrieval(
    ...     retriever,
    ...     queries={"q1": "cat food", "q2": "dog toys"},
    ...     qrels={"q1": ["d2"], "q2": ["d4"]},
    ...     metrics=["recall@2", "mrr"],
    ... )
    >>> report.metrics
    {'recall@2': 0.5, 'mrr': 0.25}
    """
    call = _call_retriever(retriever)
    names, specs = _check_metric_names(metrics)
    threshold = check_relevance_threshold(relevance_threshold)
    policy = check_duplicates(duplicates)
    division = check_zero_division(zero_division)
    gain_fn = check_gain(gain)
    degraded_policy = check_on_degraded(on_degraded)
    order = _check_query_sets(queries, qrels)
    if not isinstance(keep_rankings, bool):
        raise TypeError("keep_rankings must be a bool.")
    if id_getter is not None and not callable(id_getter):
        raise TypeError("id_getter must be callable or None.")
    tick = time.perf_counter if clock is None else clock
    if not callable(tick):
        raise TypeError("clock must be callable or None.")

    judged = []
    ranked = []
    latencies = []
    statuses = []
    for position, key in enumerate(order):
        label = where(position, key)
        judged.append(judgments(qrels[key], relevance_threshold=threshold, label=label))
        start = tick()
        response = call(queries[key])
        ids = ranked_ids(
            response, id_getter=id_getter
        )  # a lazy response is consumed here
        latencies.append(float(tick() - start))
        status = response_status(response)
        if status in DEGRADED_STATUSES and degraded_policy == "raise":
            raise ValueError(
                f"{label}: the retriever returned a {status!r} response, so a "
                "requested leg did not run and the ranking is partial. Fix the "
                "retriever (for example supply query embeddings), or pass "
                "on_degraded='warn' to evaluate it anyway."
            )
        statuses.append(status)
        ranked.append(ranking(ids, duplicates=policy, label=label))
    check_id_types(judged, ranked)

    degraded = sum(1 for s in statuses if s in DEGRADED_STATUSES)
    if degraded and degraded_policy == "warn":
        warnings.warn(
            f"{degraded} of {len(order)} responses were degraded or failed; the "
            "metrics include partial rankings (see status_counts).",
            RuntimeWarning,
            stacklevel=2,
        )

    values_by_metric = {}
    aggregates = {}
    for (name, base, _), values in zip(
        specs, per_query(judged, ranked, specs, gain_fn)
    ):
        settled = aggregate(
            values,
            zero_division=division,
            sample_weight=None,
            average=None,
            metric=name,
            reason=undefined_reason(base),
            stacklevel=3,
        )
        values_by_metric[name] = settled
        aggregates[name] = aggregate(
            settled,
            zero_division=math.nan,
            sample_weight=None,
            average="mean",
            metric=name,
            reason=undefined_reason(base),
        )

    query_results = tuple(
        QueryEvaluation(
            query_id=as_id(key, label="query identifier"),
            metrics={name: float(values_by_metric[name][i]) for name in names},
            n_retrieved=len(ranked[i]),
            n_relevant=len(judged[i].relevant),
            latency_s=latencies[i],
            status=statuses[i],
            retrieved=ranked[i] if keep_rankings else None,
        )
        for i, key in enumerate(order)
    )
    settings = {
        "metrics": [str(name) for name in names],
        "relevance_threshold": threshold,
        "duplicates": str(policy),
        "zero_division": _zero_division_label(division),
        "gain": str(gain),
        "on_degraded": str(degraded_policy),
        "retriever": type(retriever).__name__,
    }
    if id_getter is not None:  # absent means none, in every format (TOML has no null)
        settings["id_getter"] = str(getattr(id_getter, "__name__", repr(id_getter)))
    return RetrievalEvaluation(
        metrics=aggregates,
        queries=query_results,
        settings=settings,
        metadata={} if metadata is None else dict(metadata),
    )
