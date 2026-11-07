# scikitplot/corpus/_logodds_fusion.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Probability fusion: pool calibrated evidence from several retrievers in log-odds.

Reciprocal rank fusion (:class:`~scikitplot.corpus.FusionPolicy`) combines
ranks and needs no calibration, which is why it stays the default. When every
leg's scores have been calibrated into probabilities of relevance, the legs can
instead be combined as evidence, and the result is again a probability.

The rule
--------
For legs :math:`\ell` with weight :math:`w_\ell > 0`, each with a fitted
calibrator of prevalence :math:`\pi_\ell`, and a target prior
:math:`\pi_0` (``base_rate``), a document :math:`d` gets

.. math::

    z(d) = \operatorname{logit}\pi_0
         + \sum_{\ell \ni d} w_\ell
           \left(\operatorname{logit} p_\ell(d) - \operatorname{logit}\pi_\ell\right),
    \qquad P(d) = \sigma(z(d)),

where :math:`p_\ell(d)` is the leg's calibrated probability for the score it
gave :math:`d`. With unit weights this is Bayes' rule when the legs' evidence
is conditionally independent given relevance: each leg multiplies the prior
odds by its likelihood ratio :math:`\operatorname{odds}(p_\ell) /
\operatorname{odds}(\pi_\ell)`. A leg that does not list :math:`d` contributes
nothing: its likelihood ratio is taken as 1 (no evidence either way).

* Probabilities are clipped to ``[clip, 1 - clip]`` before the logit, so an
  isotonic 0 or 1 is a strong but finite vote.
* A document counts once per leg, at its first listing (as in rank fusion).
* Order: descending :math:`z`, compared as floats (each :math:`z` is a
  :func:`math.fsum`, independent of leg order), then the key ascending.

Notes
-----
**User.** Independence rarely holds: a lexical and a dense retriever often
agree because they see the same words. Correlated legs are counted twice and
the fused probability is over-confident. Independence *given relevance*
cannot be measured without labels, so the test is the fused probabilities'
:func:`~scikitplot.corpus.calibration_report` on held-out labelled queries; a
weight below 1 on correlated legs is the usual correction when it fails.
:meth:`LogOddsFusion.fuse` also returns the Pearson correlation of each pair
of legs' evidence over the documents both listed
(``LogOddsFusionResult.evidence_correlation``). It is a *marginal*
correlation: positive whenever both legs are informative, even when they are
conditionally independent (0.26 in a test with independent legs). It does not
test independence; a value near 1 flags legs that are nearly duplicates.

**Developer.** The legs are given *raw scores and their calibrator*, not
probabilities, so an uncalibrated number cannot enter by mistake: the
calibrator must report ``fitted is True`` and expose ``predict_proba`` and
``prevalence_`` (the contract of
:class:`scikitplot.rank_bm25.ScoreCalibrator`, duck-typed so this module
imports nothing from that distribution).
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import Any, Iterable, Mapping

from ._filters import SecurityFilter

__all__ = [
    "PROBABILITY_FUSION_LEGS",
    "LogOddsContribution",
    "LogOddsFusedItem",
    "LogOddsFusion",
    "LogOddsFusionResult",
    "ProbabilityFusion",
]

#: Leg names a :class:`ProbabilityFusion` can calibrate in hybrid search.
PROBABILITY_FUSION_LEGS: tuple[str, ...] = ("lexical", "dense")


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _require_rate(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value!r}")
    return value


def _require_weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


def _require_calibrator(calibrator: Any, leg: str) -> None:
    if getattr(calibrator, "fitted", False) is not True:
        raise ValueError(
            f"leg {leg!r}: its calibrator is not fitted. Probability fusion "
            "needs calibrated legs; use FusionPolicy (rank fusion) otherwise."
        )
    for attribute in ("predict_proba", "prevalence_"):
        if not hasattr(calibrator, attribute):
            raise TypeError(
                f"leg {leg!r}: calibrator has no {attribute!r}; expected a "
                "fitted ScoreCalibrator or an object with the same contract."
            )


@dataclasses.dataclass(frozen=True)
class LogOddsContribution:
    """
    One leg's evidence for one fused item.

    Parameters
    ----------
    leg : str
        The leg's name.
    position : int
        Zero-based position of the leg in the call.
    score : float
        The raw score the leg gave the item.
    probability : float
        The leg's calibrated probability for that score, before clipping.
    weight : float
        The leg's weight.
    evidence : float
        ``weight * (logit(clipped probability) - logit(leg prevalence))``.
    """

    leg: str
    position: int
    score: float
    probability: float
    weight: float
    evidence: float


@dataclasses.dataclass(frozen=True)
class LogOddsFusedItem:
    """
    One item of a probability-fused ranking.

    Parameters
    ----------
    key : str
        The item's identity.
    probability : float
        ``sigmoid(log_odds)``.
    log_odds : float
        ``logit(base_rate) + fsum(evidence)``.
    contributions : tuple of LogOddsContribution
        One per leg that listed the item, in leg order.
    """

    key: str
    probability: float
    log_odds: float
    contributions: tuple[LogOddsContribution, ...]


@dataclasses.dataclass(frozen=True)
class LogOddsFusionResult:
    """
    A fused ranking and the diagnostics needed to trust it.

    Parameters
    ----------
    items : tuple of LogOddsFusedItem
        Descending log-odds, ties by key, at most ``top_k``.
    evidence_correlation : dict
        For every pair of legs ``(a, b)`` (positions, ``a < b``): marginal
        Pearson correlation of their evidence over the items both listed, or
        ``None`` when fewer than three items are shared or either side is
        constant. Not a test of conditional independence (see the module
        notes); near 1 means near-duplicate legs.
    shared_counts : dict
        For the same pairs, how many items both legs listed.
    """

    items: tuple[LogOddsFusedItem, ...]
    evidence_correlation: dict[tuple[int, int], float | None]
    shared_counts: dict[tuple[int, int], int]


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:  # noqa: PLR2004 - two points always correlate perfectly
        return None
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


@dataclasses.dataclass(frozen=True)
class LogOddsFusion:
    """
    Pool calibrated legs as log-odds evidence into probabilities of relevance.

    Parameters
    ----------
    base_rate : float
        Prior probability of relevance for the fused population, in
        ``(0, 1)``. It shifts every log-odds equally, so it changes the
        probabilities but never the order. Required: there is no neutral
        default.
    weights : mapping of str to float, optional
        Weight per leg name; finite and non-negative; 0 removes the leg.
    default_weight : float, default=1.0
        Weight of a leg ``weights`` does not name.
    clip : float, default=1e-6
        Probabilities are clipped to ``[clip, 1 - clip]`` before the logit; in
        ``(0, 0.5)``.
    top_k : int or None, default=None
        Number of items to return; ``None`` returns all.

    Raises
    ------
    TypeError, ValueError
        For a parameter outside its domain.

    Examples
    --------
    >>> class Fitted:  # stands in for a fitted ScoreCalibrator
    ...     fitted, prevalence_ = True, 0.5
    ...
    ...     def predict_proba(self, scores):
    ...         return [min(max(s, 0.0), 1.0) for s in scores]
    >>> fusion = LogOddsFusion(base_rate=0.5)
    >>> result = fusion.fuse(
    ...     [
    ...         ("lexical", [("A", 0.9), ("B", 0.6)], Fitted()),
    ...         ("dense", [("B", 0.8), ("C", 0.7)], Fitted()),
    ...     ]
    ... )
    >>> [item.key for item in result.items]
    ['A', 'B', 'C']
    >>> round(result.items[1].probability, 4)  # odds 1.5 * 4 = 6
    0.8571
    """

    base_rate: float
    weights: Mapping[str, float] | tuple[tuple[str, float], ...] = ()
    default_weight: float = 1.0
    clip: float = 1e-6
    top_k: int | None = None

    def __post_init__(self) -> None:
        """Validate every field and freeze the weights."""
        object.__setattr__(
            self,
            "base_rate",
            _require_rate(self.base_rate, "base_rate"),
        )
        object.__setattr__(
            self,
            "default_weight",
            _require_weight(self.default_weight, "default_weight"),
        )
        clip = self.clip
        if isinstance(clip, bool) or not isinstance(clip, numbers.Real):
            raise TypeError(f"clip must be a real number, got {type(clip).__name__}")
        if not 0.0 < float(clip) < 0.5:  # noqa: PLR2004
            raise ValueError(f"clip must lie in (0, 0.5), got {clip!r}")
        object.__setattr__(self, "clip", float(clip))
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 0
        ):
            raise ValueError(f"top_k must be None or an int >= 0, got {self.top_k!r}")
        pairs = (
            list(self.weights.items())
            if isinstance(self.weights, Mapping)
            else list(self.weights)
        )
        checked: dict[str, float] = {}
        for leg, weight in pairs:
            if not isinstance(leg, str):
                raise TypeError(f"a leg name must be a str, got {type(leg).__name__}")
            if leg in checked:
                raise ValueError(f"leg {leg!r} is given two weights")
            checked[leg] = _require_weight(weight, f"weights[{leg!r}]")
        object.__setattr__(self, "weights", tuple(sorted(checked.items())))

    def weight(self, leg: str) -> float:
        """Return the weight of the leg named ``leg``."""
        return dict(self.weights).get(leg, self.default_weight)

    def fuse(  # ruff: ignore[too-many-branches]
        self,
        legs: Iterable[tuple[str, Iterable[tuple[str, float]], Any]],
    ) -> LogOddsFusionResult:
        """
        Fuse calibrated legs.

        Parameters
        ----------
        legs : iterable of (str, iterable of (str, float), calibrator)
            Each leg's name, its ``(key, raw score)`` hits best first, and the
            fitted calibrator for that leg's scores.

        Returns
        -------
        LogOddsFusionResult
            Items and evidence diagnostics.

        Raises
        ------
        TypeError
            For a malformed leg, a non-``str`` key or name, a non-numeric
            score, or a calibrator without the required contract.
        ValueError
            For an empty key, a non-finite score, an unfitted calibrator, or a
            calibrated probability outside ``[0, 1]``.
        """
        weights = dict(self.weights)
        low, high = self.clip, 1.0 - self.clip
        evidence: dict[str, list[float]] = {}
        parts: dict[str, list[LogOddsContribution]] = {}
        by_leg: list[dict[str, float]] = []
        for position, leg in enumerate(legs):
            if not isinstance(leg, tuple) or len(leg) != 3:  # noqa: PLR2004
                raise TypeError(
                    f"leg {position} must be a (name, hits, calibrator) triple"
                )
            name, hits, calibrator = leg
            if not isinstance(name, str):
                raise TypeError(f"leg {position} name must be a str")
            weight = weights.get(name, self.default_weight)
            leg_evidence: dict[str, float] = {}
            by_leg.append(leg_evidence)
            if weight == 0:
                continue
            _require_calibrator(calibrator, name)
            keys: list[str] = []
            scores: list[float] = []
            listed: set[str] = set()
            for rank, hit in enumerate(hits, start=1):
                if not isinstance(hit, tuple) or len(hit) != 2:  # noqa: PLR2004
                    raise TypeError(
                        f"leg {name!r} rank {rank}: a hit must be (key, score)",
                    )
                key, score = hit
                if not isinstance(key, str):
                    raise TypeError(f"leg {name!r} rank {rank}: a key must be a str")
                if not key:
                    raise ValueError(f"leg {name!r} rank {rank}: a key is empty")
                if isinstance(score, bool) or not isinstance(score, numbers.Real):
                    raise TypeError(
                        f"leg {name!r} rank {rank}: a score must be a number",
                    )
                if not math.isfinite(float(score)):
                    raise ValueError(
                        f"leg {name!r} rank {rank}: a score is not finite",
                    )
                if key in listed:
                    continue
                listed.add(key)
                keys.append(key)
                scores.append(float(score))
            if not keys:
                continue
            probabilities = [float(p) for p in calibrator.predict_proba(scores)]
            prior = _logit(min(max(float(calibrator.prevalence_), low), high))
            for key, score, p in zip(keys, scores, probabilities):
                if not 0.0 <= p <= 1.0:
                    raise ValueError(
                        f"leg {name!r}: calibrator returned {p!r} for score {score!r}, "
                        "which is not a probability"
                    )
                value = weight * (_logit(min(max(p, low), high)) - prior)
                leg_evidence[key] = value
                evidence.setdefault(key, []).append(value)
                parts.setdefault(key, []).append(
                    LogOddsContribution(name, position, score, p, weight, value)
                )
        anchor = _logit(self.base_rate)
        log_odds = {key: anchor + math.fsum(values) for key, values in evidence.items()}
        ordered = sorted(log_odds, key=lambda key: (-log_odds[key], key))
        if self.top_k is not None:
            ordered = ordered[: self.top_k]
        items = tuple(
            LogOddsFusedItem(
                key,
                _sigmoid(log_odds[key]),
                log_odds[key],
                tuple(parts[key]),
            )
            for key in ordered
        )
        correlation: dict[tuple[int, int], float | None] = {}
        shared: dict[tuple[int, int], int] = {}
        for a in range(len(by_leg)):
            for b in range(a + 1, len(by_leg)):
                common = sorted(set(by_leg[a]) & set(by_leg[b]))
                shared[(a, b)] = len(common)
                correlation[(a, b)] = _pearson(
                    [by_leg[a][k] for k in common], [by_leg[b][k] for k in common]
                )
        return LogOddsFusionResult(items, correlation, shared)


@dataclasses.dataclass(frozen=True, eq=False)
class ProbabilityFusion:
    """
    Opt-in probability fusion for hybrid search: a policy and a calibrator per leg.

    Parameters
    ----------
    policy : LogOddsFusion
        Weights, base rate and clipping. A leg runs only if its weight is
        above 0.
    calibrators : mapping of str to calibrator
        Fitted calibrator for each leg that runs, keyed by ``"lexical"``
        (BM25 scores) and ``"dense"`` (cosine similarity). Each must be fitted
        on the scores that leg produces for the hybrid candidate pool:
        ``3 * top_k`` hits per leg.
    scope : SecurityFilter or None, default=None
        The scope the lexical calibrator was fitted under, or None for an
        unscoped index. When the lexical leg runs, hybrid search fuses only
        on an index bound to a scope with the same
        :attr:`SecurityFilter.digest` (None: an index with no scope).

    Raises
    ------
    TypeError
        If ``policy`` is not a :class:`LogOddsFusion`, ``calibrators`` is not
        a mapping, a calibrator lacks the contract, or ``scope`` is not a
        :class:`SecurityFilter` or None.
    ValueError
        For an unknown leg name, a running leg without a calibrator, an
        unfitted calibrator, or no running leg.

    Notes
    -----
    **User.** Pass it as ``RetrievalConfig(match_mode="hybrid",
    probability_fusion=...)``; hits then carry ``score`` = the fused
    probability of relevance and ``native_metric="logodds_probability"``.

    BM25 document count, document frequencies and average length are computed
    over the documents a scope admits, so the same document scores
    differently under another scope. For a scoped index, fit the lexical
    calibrator on that index's scores and pass its ``SecurityFilter`` as
    ``scope``; any other scope, or none, is refused with both digests named.
    Cosine similarity does not depend on the scope, so a dense-only fusion
    (lexical weight 0) runs under any scope.

    **Developer.** The scope is matched by digest, never by label: two
    filters with one label are two scopes.
    """

    policy: LogOddsFusion
    calibrators: Mapping[str, Any]
    scope: SecurityFilter | None = None

    def __post_init__(self) -> None:
        """Validate the policy, the leg names and every running leg's calibrator."""
        if not isinstance(self.policy, LogOddsFusion):
            raise TypeError(
                f"policy must be a LogOddsFusion, got {type(self.policy).__name__}"
            )
        if not isinstance(self.calibrators, Mapping):
            raise TypeError("calibrators must be a mapping of leg name to calibrator")
        unknown = sorted(set(self.calibrators) - set(PROBABILITY_FUSION_LEGS))
        if unknown:
            raise ValueError(
                f"unknown leg names {unknown}; hybrid legs are {PROBABILITY_FUSION_LEGS}"
            )
        running = [
            leg for leg in PROBABILITY_FUSION_LEGS if self.policy.weight(leg) > 0
        ]
        if not running:
            raise ValueError("every leg has weight 0; nothing would be retrieved")
        for leg in running:
            if leg not in self.calibrators:
                raise ValueError(
                    f"leg {leg!r} has weight {self.policy.weight(leg)} but no calibrator"
                )
            _require_calibrator(self.calibrators[leg], leg)
        if self.scope is not None and not isinstance(self.scope, SecurityFilter):
            raise TypeError(
                f"scope must be a SecurityFilter or None, got {type(self.scope).__name__}"
            )
        object.__setattr__(self, "calibrators", dict(self.calibrators))

    @property
    def scope_digest(self) -> str | None:
        """The digest of :attr:`scope`, or None when fitted unscoped (str or None)."""
        return None if self.scope is None else self.scope.digest

    def require_scope(self, scope: SecurityFilter | None) -> None:
        """
        Refuse an index scope the lexical calibrator was not fitted under.

        Parameters
        ----------
        scope : SecurityFilter or None
            The scope the index is bound to; None for an unscoped index.

        Raises
        ------
        ValueError
            If the lexical leg runs and the digests differ, naming both.
        """
        if not self.runs("lexical"):
            return  # cosine similarity does not depend on the scope
        index = None if scope is None else scope.digest
        fitted = self.scope_digest
        if fitted != index:
            raise ValueError(
                "probability fusion: the lexical calibrator was fitted under scope "
                f"{fitted or 'unscoped'}, but this index is bound to "
                f"{index or 'unscoped'}. BM25 statistics depend on the scope: fit "
                "the calibrator on this index's scores and pass its SecurityFilter "
                "as ProbabilityFusion(scope=...), or use rank fusion."
            )

    def runs(self, leg: str) -> bool:
        """Whether the leg named ``leg`` is retrieved and fused."""
        return self.policy.weight(leg) > 0
