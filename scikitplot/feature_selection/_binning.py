# scikitplot/feature_selection/_binning.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Bin recommendations: *should* a numeric feature be binned, into *how many* bins, *where*.

Three decisions are kept apart:

1. **Suitability** -- an :class:`~scikitplot.feature_selection.AssessmentState`.
   Not numeric or declared categorical: ``not_applicable``; fewer finite values
   than ``min_count``: ``unknown``; two or fewer distinct values, zero range,
   or no more distinct values than bins: ``false``; otherwise ``true``.
2. **Count** -- a rule of :class:`~scikitplot.feature_selection.BinningPolicy`,
   clipped to ``[min_bins, max_bins]`` (the unclipped count is kept).
3. **Edges** -- equal-width or quantile, fitted on the training data and
   frozen in the recommendation.

Notes
-----
**User note.** A recommendation never transforms data. Use
:meth:`~scikitplot.feature_selection.FeatureSchema.binning_plan` to read all of
them as a table, or :func:`make_kbins_discretizer` to build a scikit-learn
``KBinsDiscretizer`` with the recommended per-feature counts; inside a pipeline
that discretizer re-learns its edges on each training fold, as it should.

**Developer note.** ``"knuth"`` is Knuth's rule [1]_ evaluated for *every*
integer count in ``[min_bins, max_bins]`` and the largest log posterior taken
(ties go to the smaller count), instead of a local optimiser started from
another rule; the answer is therefore exact and deterministic for the range.
The other rules are those of :func:`numpy.histogram_bin_edges` [2]_.

References
----------
.. [1] Knuth, K. H. (2006). "Optimal data-based binning for histograms".
   arXiv:physics/0605197. https://arxiv.org/abs/physics/0605197
.. [2] NumPy documentation, ``numpy.histogram_bin_edges``.
   https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._policy import BIN_RULES, BIN_STRATEGIES, BinningPolicy
from ._stats import FeatureStats, numeric_values
from ._types import AssessmentState, FeatureKind

__all__ = ["BinningRecommendation", "knuth_bin_count", "make_kbins_discretizer"]

_CATEGORICAL_KINDS = (
    FeatureKind.BINARY.value,
    FeatureKind.CATEGORICAL_NOMINAL.value,
    FeatureKind.CATEGORICAL_ORDINAL.value,
    FeatureKind.TEXT.value,
)


@dataclasses.dataclass(frozen=True)
class BinningRecommendation:
    """
    A frozen bin recommendation for one feature.

    Parameters
    ----------
    feature : str
        Column name.
    state : str
        Suitability, an :class:`~scikitplot.feature_selection.AssessmentState` value.
    rule : str or None
        Count rule used.
    n_bins : int or None
        Recommended count (after clipping and quantile merging).
    rule_n_bins : int or None
        Count the rule produced before clipping.
    strategy : str or None
        ``"uniform"`` or ``"quantile"``.
    edges : tuple of float
        ``n_bins + 1`` increasing edges fitted on training data; empty when
        not suitable.
    reasons : tuple of str
        Reason codes (``"clipped_to_max_bins"``, ``"tied_quantiles_merged"``).

    See Also
    --------
    BinningPolicy : The settings that produced this recommendation.
    """

    feature: str
    state: str
    rule: str | None = None
    n_bins: int | None = None
    rule_n_bins: int | None = None
    strategy: str | None = None
    edges: tuple[float, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def is_suitable(self) -> bool:
        """bool: The suitability state is ``"true"``."""
        return self.state == AssessmentState.TRUE.value

    def to_dict(self) -> dict[str, Any]:
        """Return the shared-data-model form."""
        data = dataclasses.asdict(self)
        data["edges"] = list(self.edges)
        data["reasons"] = list(self.reasons)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BinningRecommendation:
        """
        Rebuild from :meth:`to_dict` output, checking every field.

        Parameters
        ----------
        data : mapping
            Stored form.

        Returns
        -------
        BinningRecommendation
            The recommendation.

        Raises
        ------
        ValueError
            For an off-contract record.
        """
        allowed = {f.name for f in dataclasses.fields(cls)}
        if set(data) - allowed or not {"feature", "state"} <= set(data):
            raise ValueError(f"off-contract binning record: keys {sorted(data)}")
        AssessmentState(data["state"])
        for key in ("n_bins", "rule_n_bins"):
            value = data.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"binning {key} must be an int or None")
        edges = data.get("edges", [])
        if not isinstance(edges, list) or any(
            isinstance(e, bool) or not isinstance(e, (int, float)) for e in edges
        ):
            raise ValueError("binning edges must be a list of numbers")
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("binning edges must be strictly increasing")
        if not isinstance(data["feature"], str):
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                "binning feature must be a string",
            )
        rule, strategy = data.get("rule"), data.get("strategy")
        if rule is not None and rule not in BIN_RULES:
            raise ValueError(f"binning rule {rule!r} is not one of {BIN_RULES}")
        if strategy is not None and strategy not in BIN_STRATEGIES:
            raise ValueError(
                f"binning strategy {strategy!r} is not one of {BIN_STRATEGIES}"
            )
        reasons = data.get("reasons", [])
        if not isinstance(reasons, list) or not all(
            isinstance(r, str) for r in reasons
        ):
            raise ValueError("binning reasons must be a list of strings")
        if edges and len(edges) != (data.get("n_bins") or 0) + 1:
            raise ValueError("binning needs n_bins + 1 edges")
        return cls(
            feature=data["feature"],
            state=data["state"],
            rule=data.get("rule"),
            n_bins=data.get("n_bins"),
            rule_n_bins=data.get("rule_n_bins"),
            strategy=data.get("strategy"),
            edges=tuple(float(e) for e in edges),
            reasons=tuple(data.get("reasons", [])),
        )


def knuth_bin_count(values: Any, min_bins: int = 1, max_bins: int = 100) -> int:
    """
    Return the equal-width bin count that maximises Knuth's posterior.

    Parameters
    ----------
    values : array-like of float
        Finite sample values.
    min_bins, max_bins : int, default=1, 100
        Inclusive search range.

    Returns
    -------
    int
        The count with the largest log posterior; the smallest such count on
        an exact tie.

    Raises
    ------
    ValueError
        For an empty sample, non-finite values or an invalid range.

    Notes
    -----
    The log posterior of ``M`` equal-width bins with counts ``n_k`` over ``N``
    values is (Knuth 2006, eq. 31)::

        N log M + lgamma(M/2) - M lgamma(1/2) - lgamma(N + M/2)
            + sum_k lgamma(n_k + 1/2)

    Examples
    --------
    >>> import numpy as np
    >>> from scikitplot.feature_selection import knuth_bin_count
    >>> knuth_bin_count(np.random.default_rng(0).normal(size=1000), 1, 50)
    9
    """
    from scipy.special import gammaln  # noqa: PLC0415

    data = np.asarray(values, dtype="float64").ravel()
    if data.size == 0:
        raise ValueError("knuth_bin_count needs at least one value")
    if not np.all(np.isfinite(data)):
        raise ValueError("knuth_bin_count needs finite values")
    if not 1 <= min_bins <= max_bins:
        raise ValueError(f"need 1 <= min_bins <= max_bins, got {min_bins}, {max_bins}")
    n = data.size
    low, high = float(data.min()), float(data.max())
    if low == high:
        return int(min_bins)
    best_m, best_logp = int(min_bins), -np.inf
    for m in range(int(min_bins), int(max_bins) + 1):
        counts, _ = np.histogram(data, bins=m, range=(low, high))
        logp = (
            n * np.log(m)
            + gammaln(m / 2.0)
            - m * gammaln(0.5)
            - gammaln(n + m / 2.0)
            + gammaln(counts + 0.5).sum()
        )
        if logp > best_logp:
            best_m, best_logp = m, logp
    return best_m


def _rule_count(finite: np.ndarray, policy: BinningPolicy) -> int:
    if policy.rule == "fixed":
        return int(policy.n_bins)
    if policy.rule == "knuth":
        return knuth_bin_count(finite, policy.min_bins, policy.max_bins)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        edges = np.histogram_bin_edges(finite, bins=policy.rule)
    return max(int(len(edges) - 1), 1)


def recommend_bins(  # ruff: ignore[too-many-return-statements]
    series: pd.Series,
    stats: FeatureStats,
    policy: BinningPolicy,
    declared_kind: str | None = None,
) -> BinningRecommendation:
    """
    Recommend bins for one training column.

    Parameters
    ----------
    series : pandas.Series
        Training column.
    stats : FeatureStats
        Its statistics.
    policy : BinningPolicy
        Settings.
    declared_kind : str or None, default=None
        Kind declared in the feature policy.

    Returns
    -------
    BinningRecommendation
        The recommendation.
    """
    name = stats.name
    na, unknown = AssessmentState.NOT_APPLICABLE.value, AssessmentState.UNKNOWN.value
    false = AssessmentState.FALSE.value
    if not stats.is_numeric:
        return BinningRecommendation(name, na, reasons=("not_numeric",))
    if declared_kind in _CATEGORICAL_KINDS:
        return BinningRecommendation(name, na, reasons=("declared_categorical",))
    if (stats.n_finite or 0) < policy.min_count:
        return BinningRecommendation(name, unknown, reasons=("too_few_finite_values",))
    _n_unique = stats.n_unique <= 2  # ruff: ignore[magic-value-comparison]
    if stats.n_unique is not None and _n_unique:
        return BinningRecommendation(name, false, reasons=("two_or_fewer_values",))
    values = numeric_values(series)
    finite = values[np.isfinite(values)]
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        return BinningRecommendation(name, false, reasons=("zero_range",))
    reasons: list[str] = []
    raw = _rule_count(finite, policy)
    count = min(max(raw, policy.min_bins), policy.max_bins)
    if count != raw:
        reasons.append("clipped_to_min_bins" if raw < count else "clipped_to_max_bins")
    n_distinct = int(np.unique(finite).size)
    if n_distinct <= count:
        reasons.append("no_more_distinct_values_than_bins")
        return BinningRecommendation(
            name, false, policy.rule, count, raw, policy.strategy, (), tuple(reasons)
        )
    if policy.strategy == "uniform":
        edges = np.unique(np.linspace(low, high, count + 1))
        if len(edges) - 1 < count:  # bins narrower than the float spacing
            reasons.append("tied_edges_merged")
            count = int(len(edges) - 1)
    else:
        edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, count + 1)))
        if len(edges) - 1 < count:
            reasons.append("tied_quantiles_merged")
            count = int(len(edges) - 1)
    return BinningRecommendation(
        feature=name,
        state=AssessmentState.TRUE.value,
        rule=policy.rule,
        n_bins=count,
        rule_n_bins=raw,
        strategy=policy.strategy,
        edges=tuple(float(e) for e in edges),
        reasons=tuple(reasons) or ("suitable",),
    )


def make_kbins_discretizer(
    schema: Any,
    columns: Sequence[str] | None = None,
    *,
    encode: str = "ordinal",
    **kwargs: Any,
) -> tuple[list[str], Any]:
    """
    Build a ``KBinsDiscretizer`` with the recommended per-feature bin counts.

    Parameters
    ----------
    schema : FeatureSchema
        A fitted schema with binning recommendations.
    columns : sequence of str or None, default=None
        Columns to bin; None takes every suitable feature in schema order.
    encode : {"ordinal", "onehot", "onehot-dense"}, default="ordinal"
        Passed to ``KBinsDiscretizer``.
    **kwargs
        Other ``KBinsDiscretizer`` arguments (``subsample``, ``quantile_method``
        ...). ``strategy`` defaults to the policy's strategy.

    Returns
    -------
    columns : list of str
        Columns the discretizer applies to, in order (use them in a
        ``ColumnTransformer``).
    discretizer : sklearn.preprocessing.KBinsDiscretizer
        Unfitted discretizer with ``n_bins`` set per column.

    Raises
    ------
    ValueError
        If a requested column is not suitable for binning or no column is.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from scikitplot.feature_selection import (
    ...     profile_features,
    ...     make_kbins_discretizer,
    ... )
    >>> X = pd.DataFrame({"x": np.random.default_rng(0).gamma(2.0, size=500)})
    >>> cols, kb = make_kbins_discretizer(profile_features(X))
    >>> cols, int(kb.n_bins[0]) > 1
    (['x'], True)
    """
    from sklearn.preprocessing import KBinsDiscretizer  # noqa: PLC0415

    recs = {spec.name: spec.binning for spec in schema}
    if columns is None:
        columns = [n for n, r in recs.items() if r is not None and r.is_suitable]
    else:
        columns = list(columns)
        for column in columns:
            rec = recs.get(column)
            if rec is None or not rec.is_suitable:
                state = None if rec is None else rec.state
                raise ValueError(
                    f"column {column!r} has no suitable binning recommendation"
                    f" (state={state!r})"
                )
    if not columns:
        raise ValueError("no column is suitable for binning")
    kwargs.setdefault("strategy", schema.policy.binning.strategy)
    n_bins = np.asarray([recs[c].n_bins for c in columns], dtype=int)
    return columns, KBinsDiscretizer(n_bins=n_bins, encode=encode, **kwargs)
