# scikitplot/rank_bm25/_calibration.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Turn BM25 scores into calibrated probabilities of relevance.

A BM25 score ranks documents but is not a probability: its scale depends on
the corpus, the query length and the recipe. :class:`ScoreCalibrator` learns a
monotone map from scores to ``P(relevant | score)`` on labelled (query,
document) pairs, and refuses the two mistakes that make such a map look better
than it is: evaluating on the queries it was fitted on, and reusing it on a
population with a different share of relevant documents without saying so.

Notes
-----
**User-focused.**

.. code-block:: python

    cal = ScoreCalibrator("sigmoid").fit(
        train_scores, train_labels, query_ids=train_qids, population="bm25-top100"
    )
    cal.require_held_out(test_qids)  # raises if a test query was trained on
    probs = cal.predict_proba(test_scores)

Measure the result with :func:`scikitplot.corpus.calibration_report`. A
calibrator is only valid for the population it was fitted on (for example the
top 100 BM25 candidates per query); :meth:`ScoreCalibrator.shift_prior` moves
it to a population with a known different share of relevant documents.

**Developer-focused.**

*Sigmoid* (Platt scaling): ``p = sigmoid(a * s + b)``, fitted by Newton's
method on the convex log loss, after standardising the scores, with Platt's
target smoothing (positives at ``(n+ + 1) / (n+ + 2)``, negatives at
``1 / (n- + 2)``) by default, as scikit-learn's sigmoid calibration does. The
smoothing keeps the optimum finite when the labels are perfectly separable. A
fitted slope ``a <= 0`` is refused: it would reverse the ranking, which means
the score does not predict relevance on these labels.

*Isotonic*: the pool-adjacent-violators solution of non-decreasing least
squares, scores with equal values pooled first, predictions interpolated
linearly between fitted points and clipped outside them -- scikit-learn's
``IsotonicRegression(out_of_bounds="clip")``. It is monotone but not strictly:
it can map different scores to one probability, which merges their ranks.

*Prior shift*: for a population whose relevant share is ``pi'`` instead of the
training prevalence ``pi``, Bayes' rule with an unchanged score likelihood
gives ``logit p' = logit p + logit pi' - logit pi`` [1]_. It is exact under
that assumption, and the assumption is the caller's to check.

References
----------
.. [1] Saerens, M., Latinne, P. and Decaestecker, C. (2002). "Adjusting the
   outputs of a classifier to new a priori probabilities". Neural
   Computation, 14(1), 21-41.
.. [2] Platt, J. (1999). "Probabilistic outputs for support vector machines
   and comparisons to regularized likelihood methods". Advances in Large
   Margin Classifiers.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable

import numpy as np

__all__: list[str] = ["CALIBRATION_METHODS", "ScoreCalibrator"]

#: Accepted ``method`` values.
CALIBRATION_METHODS: tuple[str, ...] = ("sigmoid", "isotonic")

_SCHEMA = "score-calibrator/1"
_MAX_NEWTON_ITER = 100
_GRAD_TOL = 1e-10


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    e = np.exp(z[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check_scores(scores: Any, name: str = "scores") -> np.ndarray:
    try:
        array = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value.")
    return array


def _check_rate(rate: Any, name: str) -> float:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(rate).__name__}.")
    rate = float(rate)
    if not 0.0 < rate < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {rate!r}.")
    return rate


def _fit_sigmoid(scores, labels, platt_targets):
    """
    Return ``(a, b)`` minimising the log loss of ``sigmoid(a * s + b)``.

    Notes
    -----
    **Developer.** Newton's method on standardised scores with a halving line
    search: the objective is convex, so a step is accepted only if it lowers
    the loss, and convergence is declared on the gradient norm. Standardising
    keeps the Hessian well conditioned for scores of any scale; the solution is
    mapped back exactly afterwards.
    """
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    if not platt_targets and scores[labels == 0].max() < scores[labels == 1].min():
        # In one dimension, separable means every negative scores below every
        # positive; the 0/1 log loss then has no finite minimiser.
        raise ValueError(
            "sigmoid calibration has no finite solution: the labels are "
            "perfectly separated by the scores. Keep platt_targets=True, or add "
            "labelled pairs."
        )
    if platt_targets:
        targets = np.where(
            labels == 1.0,
            (positives + 1.0) / (positives + 2.0),
            1.0 / (negatives + 2.0),
        )
    else:
        targets = labels
    centre = float(scores.mean())
    spread = float(scores.std())
    x = (scores - centre) / spread if spread > 0 else np.zeros_like(scores)
    design = np.column_stack([x, np.ones_like(x)])

    def loss(theta):
        z = design @ theta
        # log(1 + e^z) - t * z, stably
        return float(np.sum(np.logaddexp(0.0, z) - targets * z))

    prior = min(max(targets.mean(), 1e-12), 1 - 1e-12)
    theta = np.array([0.0, _logit(prior)])
    if spread == 0:
        return 0.0, float(theta[1])
    current = loss(theta)
    eps = 1e-10
    for _ in range(_MAX_NEWTON_ITER):
        p = _sigmoid(design @ theta)
        gradient = design.T @ (p - targets)
        if float(np.max(np.abs(gradient))) <= _GRAD_TOL * labels.size:
            break
        weights = p * (1.0 - p)
        hessian = design.T @ (design * weights[:, None]) + 1e-12 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        size = 1.0
        while size > eps:
            candidate = theta - size * step
            value = loss(candidate)
            if value <= current:
                theta, current = candidate, value
                break
            size /= 2.0
        else:
            break  # no descent possible: at the optimum to working precision
        if not np.isfinite(theta).all():  # pragma: no cover - defensive
            raise ValueError("sigmoid calibration diverged to a non-finite value.")
    slope = float(theta[0]) / spread
    intercept = float(theta[1]) - float(theta[0]) * centre / spread
    return slope, intercept


def _fit_isotonic(scores, labels):
    """Return ``(thresholds, values)`` of the non-decreasing least-squares fit."""
    order = np.argsort(scores, kind="stable")
    xs, ys = scores[order], labels[order]
    unique, start = np.unique(xs, return_index=True)
    counts = np.diff(np.append(start, xs.size)).astype(np.float64)
    sums = np.add.reduceat(ys, start)
    # Pool adjacent violators over blocks of (sum, count).
    block_sum: list[float] = []
    block_count: list[float] = []
    block_width: list[int] = []
    for s, c in zip(sums.tolist(), counts.tolist()):
        block_sum.append(s)
        block_count.append(c)
        block_width.append(1)
        while len(block_sum) > 1 and (
            block_sum[-2] / block_count[-2] > block_sum[-1] / block_count[-1]
        ):
            s2, c2, w2 = block_sum.pop(), block_count.pop(), block_width.pop()
            block_sum[-1] += s2
            block_count[-1] += c2
            block_width[-1] += w2
    values = np.repeat(
        np.array(block_sum) / np.array(block_count), np.array(block_width)
    )
    return unique.astype(np.float64), values.astype(np.float64)


class ScoreCalibrator:
    """
    Monotone map from BM25 scores to probabilities of relevance.

    Parameters
    ----------
    method : {"sigmoid", "isotonic"}, default="sigmoid"
        Platt scaling (two parameters, data-efficient, strictly monotone) or
        isotonic regression (flexible, needs more labels, can merge ranks).
    platt_targets : bool, default=True
        For ``"sigmoid"``: fit to Platt's smoothed targets instead of 0/1,
        which keeps the fit finite on perfectly separable labels.

    Attributes
    ----------
    fitted : bool
        Whether :meth:`fit` has run.
    population : str or None
        The population the labels were drawn from, as declared at fit.
    prevalence_ : float
        Share of relevant labels in the population, after any
        :meth:`shift_prior`.
    n_samples_, n_queries_ : int
        Labelled pairs and distinct queries used to fit.
    slope_, intercept_ : float
        Sigmoid parameters (``"sigmoid"`` only).
    thresholds_, values_ : numpy.ndarray
        Fitted points (``"isotonic"`` only).

    Notes
    -----
    **User.** Fit on scores produced exactly as they will be at prediction
    time -- same recipe, same parameters, same analyzer, same candidate
    selection -- and evaluate only on queries not used to fit
    (:meth:`require_held_out`). A score transformed without fitting is not a
    probability; this class has no unfitted transform on purpose.
    """

    def __init__(self, method: str = "sigmoid", *, platt_targets: bool = True) -> None:
        if method not in CALIBRATION_METHODS:
            raise ValueError(
                f"method must be one of {CALIBRATION_METHODS}, got {method!r}."
            )
        if not isinstance(platt_targets, bool):
            raise TypeError("platt_targets must be a bool.")
        self.method = method
        self.platt_targets = platt_targets
        self.fitted = False
        self.population: str | None = None
        self._query_digests: frozenset[str] = frozenset()
        self._logit_offset = 0.0

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        scores: Any,
        labels: Any,
        *,
        query_ids: Iterable[Any],
        population: str,
    ) -> ScoreCalibrator:
        """
        Fit the map on labelled (query, document) pairs.

        Parameters
        ----------
        scores : array-like of shape (n_pairs,)
            BM25 scores, finite.
        labels : array-like of shape (n_pairs,)
            Binary relevance (0/1 or ``bool``).
        query_ids : iterable of str
            The query each pair belongs to; kept only as SHA-256 digests, for
            :meth:`require_held_out`.
        population : str
            What the pairs were drawn from, e.g. ``"bm25-top100"``; a
            calibrator is valid for that population only.

        Returns
        -------
        ScoreCalibrator
            ``self``.

        Raises
        ------
        ValueError
            For mismatched lengths, non-binary labels, labels of one class
            only, a non-finite score, an empty ``population``, a sigmoid slope
            ``<= 0``, or divergence on separable labels.
        TypeError
            For non-numeric scores or labels.
        """
        x = _check_scores(scores)
        y = _check_scores(labels, "labels")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("labels must be binary (0/1 or bool).")
        ids = [str(q) for q in query_ids]
        if not (x.size == y.size == len(ids)):
            raise ValueError(
                f"scores, labels and query_ids differ in length: {x.size}, "
                f"{y.size} and {len(ids)}."
            )
        positives = int(y.sum())
        if positives in (0, y.size):
            raise ValueError(
                "labels must contain both relevant and non-relevant pairs; "
                "one class alone cannot define a probability."
            )
        if not isinstance(population, str) or not population.strip():
            raise ValueError("population must be a non-empty string naming the source.")
        if self.method == "sigmoid":
            slope, intercept = _fit_sigmoid(x, y, self.platt_targets)
            # A zero slope is the honest answer for constant scores (no
            # information: predict the prevalence); otherwise it, like a
            # negative slope, means the scores do not order relevance.
            if slope < 0 or (slope == 0 and float(x.std()) > 0):
                raise ValueError(
                    f"fitted slope is {slope!r}: higher scores are not more "
                    "likely relevant on these labels, so a monotone calibration "
                    "would reverse the ranking. Check the labels and the "
                    "scoring configuration."
                )
            self.slope_, self.intercept_ = slope, intercept
        else:
            self.thresholds_, self.values_ = _fit_isotonic(x, y)
        self.population = population
        self.prevalence_ = positives / y.size
        self._train_prevalence = self.prevalence_
        self.n_samples_ = int(y.size)
        self.n_queries_ = len(set(ids))
        self._query_digests = frozenset(_digest(q) for q in ids)
        self._logit_offset = 0.0
        self.fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("this ScoreCalibrator is not fitted; call fit first.")

    # -- prediction ----------------------------------------------------------

    def predict_proba(self, scores: Any) -> np.ndarray:
        """
        Probability of relevance for each score.

        Parameters
        ----------
        scores : array-like of shape (n,)
            BM25 scores produced the way the training scores were.

        Returns
        -------
        numpy.ndarray of float64, shape (n,)
            One probability per score (one column, not scikit-learn's two).

        Raises
        ------
        RuntimeError
            If not fitted.
        ValueError, TypeError
            For non-finite, non-numeric or multi-dimensional scores.
        """
        self._require_fitted()
        x = _check_scores(scores)
        if self.method == "sigmoid":
            z = self.slope_ * x + self.intercept_ + self._logit_offset
            return _sigmoid(z)
        p = np.interp(x, self.thresholds_, self.values_)
        if self._logit_offset == 0.0:
            return p
        inside = (p > 0.0) & (p < 1.0)
        out = p.copy()
        with np.errstate(divide="ignore"):
            out[inside] = _sigmoid(
                np.log(p[inside]) - np.log1p(-p[inside]) + self._logit_offset
            )
        return out

    def shift_prior(self, base_rate: float, *, population: str) -> ScoreCalibrator:
        """
        Return a copy valid for a population with another relevant share.

        Parameters
        ----------
        base_rate : float
            Share of relevant pairs in the new population, in ``(0, 1)``.
        population : str
            Name of the new population.

        Returns
        -------
        ScoreCalibrator
            A new calibrator; ``self`` is unchanged.

        Notes
        -----
        **User.** Adds ``logit(base_rate) - logit(training prevalence)`` to
        every log-odds. Exact when the scores of relevant and of non-relevant
        pairs are distributed the same way in both populations (only their
        mix differs) -- true for a subsample, not for a different candidate
        generator. Isotonic outputs of exactly 0 or 1 stay 0 or 1.
        """
        self._require_fitted()
        rate = _check_rate(base_rate, "base_rate")
        if not isinstance(population, str) or not population.strip():
            raise ValueError("population must be a non-empty string naming the source.")
        clone = ScoreCalibrator.from_dict(self.to_dict())
        clone._logit_offset = _logit(rate) - _logit(self._train_prevalence)
        clone.prevalence_ = rate
        clone.population = population
        return clone

    def require_held_out(self, query_ids: Iterable[Any]) -> None:
        """
        Refuse evaluation queries that were used to fit.

        Parameters
        ----------
        query_ids : iterable of str
            The queries an evaluation will use.

        Raises
        ------
        ValueError
            Naming how many of them were seen during fit. Fit and evaluate on
            the same queries and calibration error is underestimated.
        """
        self._require_fitted()
        seen = sorted(
            {str(q) for q in query_ids if _digest(str(q)) in self._query_digests}
        )
        if seen:
            preview = ", ".join(repr(q) for q in seen[:3])
            raise ValueError(
                f"{len(seen)} evaluation queries were used to fit this "
                f"calibrator (e.g. {preview}); evaluate on held-out queries."
            )

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable state; query ids appear only as digests."""
        self._require_fitted()
        state: dict[str, Any] = {
            "schema": _SCHEMA,
            "method": self.method,
            "platt_targets": self.platt_targets,
            "population": self.population,
            "prevalence": self.prevalence_,
            "train_prevalence": self._train_prevalence,
            "logit_offset": self._logit_offset,
            "n_samples": self.n_samples_,
            "n_queries": self.n_queries_,
            "query_digests": sorted(self._query_digests),
        }
        if self.method == "sigmoid":
            state.update(slope=self.slope_, intercept=self.intercept_)
        else:
            state.update(
                thresholds=self.thresholds_.tolist(), values=self.values_.tolist()
            )
        return state

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> ScoreCalibrator:
        """
        Rebuild a fitted calibrator from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If the schema is not this one, or a required field is missing.
        """
        if not isinstance(state, dict) or state.get("schema") != _SCHEMA:
            raise ValueError(f"state is not a {_SCHEMA!r} calibrator.")
        try:
            obj = cls(state["method"], platt_targets=bool(state["platt_targets"]))
            obj.population = state["population"]
            obj.prevalence_ = float(state["prevalence"])
            obj._train_prevalence = float(state["train_prevalence"])
            obj._logit_offset = float(state["logit_offset"])
            obj.n_samples_ = int(state["n_samples"])
            obj.n_queries_ = int(state["n_queries"])
            obj._query_digests = frozenset(state["query_digests"])
            if obj.method == "sigmoid":
                obj.slope_ = float(state["slope"])
                obj.intercept_ = float(state["intercept"])
            else:
                obj.thresholds_ = np.asarray(state["thresholds"], dtype=np.float64)
                obj.values_ = np.asarray(state["values"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"calibrator state is missing {exc.args[0]!r}.") from exc
        obj.fitted = True
        return obj

    def __repr__(self) -> str:
        if not self.fitted:
            return f"ScoreCalibrator({self.method!r}, unfitted)"
        return (
            f"ScoreCalibrator({self.method!r}, population={self.population!r}, "
            f"n_samples={self.n_samples_}, n_queries={self.n_queries_})"
        )
