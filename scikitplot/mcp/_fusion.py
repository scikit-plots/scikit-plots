r"""
Rank-fusion contract shared by every hybrid retriever: weighted reciprocal rank fusion.

Three implementations of reciprocal rank fusion (RRF) existed in two submodules,
and they disagreed. Given the same legs they picked different winners when two
documents tied, when a document appeared twice in one leg, and when a leg had
weight 0, where two of them returned documents that no weighted leg had found.
The rule is therefore written once, here, and every hybrid path calls it.

The rule
--------
For legs :math:`\ell` with weight :math:`w_\ell > 0`, a document :math:`d`
scores

.. math::

    \operatorname{score}(d) = \sum_{\ell \ni d} \frac{w_\ell}{k + r_\ell(d)}

where :math:`r_\ell(d)` is the 1-based rank at which leg :math:`\ell` first
lists :math:`d`.

* ``k`` is an integer, at least 1.
* Weights are finite and non-negative. A weight-0 leg contributes nothing and
  adds no document.
* A document counts once per leg, at its best (first) rank; a repeat further
  down the same leg is ignored.
* Order: descending score, compared exactly as rational numbers, then the key
  in ascending order. Scores that are equal in exact arithmetic are ties, so
  floating-point rounding never decides an order.
* The reported ``score`` is :func:`math.fsum` of the per-leg terms, each
  ``weight / (k + rank)`` correctly rounded: it does not depend on leg order.
* Every fused item carries its per-leg contributions, so its score can be
  recomputed and explained from the result.

Notes
-----
**User.** Keys are the documents' identities (``doc_id``), so the tie-break is
"lowest ``doc_id`` first". Leg names are labels for weights and contributions;
two legs may share a name and then share its weight.

**Developer.** Each submodule that fuses ranks carries its own byte-identical
copy of this file, and ``scikitplot/tests/test_submodule_independence.py``
compares their digests, so the copies cannot drift into two answers again.
Only the standard library is imported. Items are sorted by their float scores
first; only items whose float scores lie within the rounding bound of each
other are compared again in exact arithmetic with :class:`fractions.Fraction`
(a float weight is a binary rational, so each term :math:`w / (k + r)` and
each sum is exact). Outside that bound the float order is the exact order, so
the result is exact at nearly float cost. Weights are stored as a sorted tuple
of ``(leg, weight)`` pairs, so a policy is hashable, picklable and copyable.

References
----------
.. [1] Cormack, G. V., Clarke, C. L. A., Buettcher, S. "Reciprocal Rank Fusion
   outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009.

Examples
--------
The worked example of the review: ``k = 60``; lexical ranks A, B, C; dense
ranks C, A, D. A scores 1/61 + 1/62 and C scores 1/63 + 1/61, so A wins.

>>> policy = FusionPolicy(k=60)
>>> fused = policy.fuse([("lexical", ["A", "B", "C"]), ("dense", ["C", "A", "D"])])
>>> [item.key for item in fused]
['A', 'C', 'B', 'D']
>>> [(c.leg, c.rank) for c in fused[0].contributions]
[('lexical', 1), ('dense', 2)]
"""

from __future__ import annotations

import dataclasses
import decimal
import math
import numbers
from fractions import Fraction
from typing import Any, Iterable, Mapping

__all__ = ["FusedItem", "FusionContribution", "FusionPolicy"]


def _require_int(value: Any, name: str, minimum: int) -> int:
    """Return ``value`` as an ``int`` of at least ``minimum``; refuse ``bool``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_weight(value: Any, name: str) -> float:
    """Return ``value`` as a finite, non-negative ``float``; refuse ``bool``."""
    if isinstance(value, bool) or not isinstance(
        value, (numbers.Real, decimal.Decimal)
    ):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    try:
        converted = float(value)
    except OverflowError:
        raise ValueError(f"{name} is too large for a float: {value!r}") from None
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return converted


#: Bound on the relative rounding error of a float sum of ``n`` correctly
#: rounded positive terms is ``n`` unit roundoffs; twice that, doubled again
#: for the two scores compared, is a safe tie window.
_ROUNDOFF = 2.0**-53


@dataclasses.dataclass(frozen=True)
class FusionContribution:
    """
    One leg's share of a fused item's score.

    Parameters
    ----------
    leg : str
        The leg's name.
    position : int
        Zero-based position of the leg in the fused call, which tells two legs
        of the same name apart.
    rank : int
        One-based rank at which the leg first listed the item.
    weight : float
        The leg's weight.
    value : float
        ``weight / (k + rank)``, correctly rounded.
    """

    leg: str
    position: int
    rank: int
    weight: float
    value: float


@dataclasses.dataclass(frozen=True)
class FusedItem:
    """
    One item of a fused ranking.

    Parameters
    ----------
    key : str
        The item's identity, as the legs listed it.
    score : float
        The fused score: :func:`math.fsum` of the contributions' values.
    contributions : tuple of FusionContribution
        One entry per leg that listed the item, in leg order.
    """

    key: str
    score: float
    contributions: tuple[FusionContribution, ...]


@dataclasses.dataclass(frozen=True)
class FusionPolicy:
    """
    Weighted reciprocal rank fusion with one tie-break and returned contributions.

    Parameters
    ----------
    k : int, optional
        The RRF constant, at least 1. Default 60.
    weights : mapping of str to float, or tuple of (str, float) pairs, optional
        Weight per leg name. Finite and non-negative; a weight of 0 removes the
        leg. Stored as a tuple of pairs sorted by leg name, so ``dict(weights)``
        reads it back and the policy is hashable.
    default_weight : float, optional
        Weight of a leg that ``weights`` does not name. Default 1.0.
    top_k : int or None, optional
        Number of items to return, at least 0; ``None`` returns every item.

    Raises
    ------
    TypeError
        If ``k`` or ``top_k`` is not an int (``bool`` included), a weight is not
        a real number, or a leg name is not a ``str``.
    ValueError
        If ``k < 1``, ``top_k < 0``, or a weight is negative or not finite.

    Examples
    --------
    >>> FusionPolicy.from_alpha(0.0).fuse([("lexical", ["a"]), ("dense", ["z"])])[0].key
    'a'
    >>> len(FusionPolicy.from_alpha(0.0).fuse([("lexical", ["a"]), ("dense", ["z"])]))
    1
    """

    k: int = 60
    weights: Mapping[str, float] | tuple[tuple[str, float], ...] = ()
    default_weight: float = 1.0
    top_k: int | None = None

    def __post_init__(self) -> None:
        """Validate every field and freeze the weights."""
        object.__setattr__(self, "k", _require_int(self.k, "k", 1))
        if self.top_k is not None:
            object.__setattr__(self, "top_k", _require_int(self.top_k, "top_k", 0))
        object.__setattr__(
            self,
            "default_weight",
            _require_weight(self.default_weight, "default_weight"),
        )
        if isinstance(self.weights, Mapping):
            pairs = list(self.weights.items())
        elif isinstance(self.weights, tuple) and all(
            isinstance(pair, tuple) and len(pair) == 2  # noqa: PLR2004
            for pair in self.weights
        ):
            pairs = list(self.weights)
        else:
            raise TypeError(
                "weights must be a mapping or a tuple of (leg, weight) pairs, got "
                f"{type(self.weights).__name__}"
            )
        checked: dict[str, float] = {}
        for leg, weight in pairs:
            if not isinstance(leg, str):
                raise TypeError(f"a leg name must be a str, got {type(leg).__name__}")
            if leg in checked:
                raise ValueError(f"leg {leg!r} is given two weights")
            checked[leg] = _require_weight(weight, f"weights[{leg!r}]")
        object.__setattr__(self, "weights", tuple(sorted(checked.items())))

    @classmethod
    def from_alpha(
        cls,
        alpha: float,
        *,
        k: int = 60,
        top_k: int | None = None,
        lexical: str = "lexical",
        dense: str = "dense",
    ) -> FusionPolicy:
        """
        Return the two-leg policy that ``alpha`` is shorthand for.

        Parameters
        ----------
        alpha : float
            Dense share in ``[0, 1]``: ``{lexical: 1 - alpha, dense: alpha}``.
            ``0`` is the lexical leg alone and ``1`` the dense leg alone.
        k : int, optional
            The RRF constant. Default 60.
        top_k : int or None, optional
            Number of items to return.
        lexical, dense : str, optional
            The two leg names.

        Returns
        -------
        FusionPolicy
            Every other leg name gets weight 0.

        Raises
        ------
        TypeError
            If ``alpha`` is not a real number.
        ValueError
            If ``alpha`` is outside ``[0, 1]`` or not finite.
        """
        alpha = _require_weight(alpha, "alpha")
        if alpha > 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        return cls(
            k=k,
            weights={lexical: 1.0 - alpha, dense: alpha},
            default_weight=0.0,
            top_k=top_k,
        )

    def weight(self, leg: str) -> float:
        """Return the weight of the leg named ``leg``."""
        return dict(self.weights).get(leg, self.default_weight)

    def fuse(self, legs: Iterable[tuple[str, Iterable[str]]]) -> list[FusedItem]:
        """
        Fuse ranked legs into one ranking.

        Parameters
        ----------
        legs : iterable of (str, iterable of str)
            Each leg's name and its keys, best first.

        Returns
        -------
        list of FusedItem
            Descending exact score, ties by ascending key, at most ``top_k``.

        Raises
        ------
        TypeError
            If a leg is not a ``(name, keys)`` pair, a name is not a ``str``, or
            a key is not a ``str``.
        ValueError
            If a key is the empty string.
        """
        weights = dict(self.weights)
        terms: dict[str, list[float]] = {}
        # (leg, position, rank, weight, value); dataclasses are built only for
        # the items returned, so a deep fan-out with a small top_k stays cheap.
        contributions: dict[str, list[tuple[str, int, int, float, float]]] = {}
        for position, leg in enumerate(legs):
            if not isinstance(leg, tuple) or len(leg) != 2:  # noqa: PLR2004
                raise TypeError(
                    f"leg {position} must be a (name, keys) pair, got "
                    f"{type(leg).__name__}"
                )
            name, keys = leg
            if not isinstance(name, str):
                raise TypeError(
                    f"leg {position} name must be a str, got {type(name).__name__}"
                )
            weight = weights.get(name, self.default_weight)
            if weight == 0:
                continue
            seen: set[str] = set()
            for rank, key in enumerate(keys, start=1):
                if not isinstance(key, str):
                    raise TypeError(
                        f"leg {name!r} rank {rank}: a key must be a str, got "
                        f"{type(key).__name__}"
                    )
                if not key:
                    raise ValueError(f"leg {name!r} rank {rank}: a key is empty")
                if key in seen:
                    continue
                seen.add(key)
                value = weight / (self.k + rank)  # one correctly rounded division
                terms.setdefault(key, []).append(value)
                contributions.setdefault(key, []).append(
                    (name, position, rank, weight, value)
                )
        score = {key: math.fsum(values) for key, values in terms.items()}
        ordered = sorted(score, key=lambda key: (-score[key], key))
        ordered = self._exact_ties(ordered, score, contributions)
        if self.top_k is not None:
            ordered = ordered[: self.top_k]
        return [
            FusedItem(
                key=key,
                score=score[key],
                contributions=tuple(
                    FusionContribution(*part) for part in contributions[key]
                ),
            )
            for key in ordered
        ]

    def _exact_ties(
        self,
        ordered: list[str],
        score: dict[str, float],
        contributions: dict[str, list[tuple[str, int, int, float, float]]],
    ) -> list[str]:
        """Reorder runs of float-close scores by exact score, then key.

        Two items whose float scores differ by more than the rounding bound
        are in exact order already; within a run of closer scores the exact
        sums decide, and equal exact sums fall to the key.
        """
        out: list[str] = []
        run: list[str] = []

        def exact(key: str) -> Fraction:
            return sum(
                (
                    Fraction(weight) / (self.k + rank)
                    for _leg, _position, rank, weight, _value in contributions[key]
                ),
                Fraction(0),
            )

        def flush() -> None:
            if len(run) > 1:
                run.sort(key=lambda key: (-exact(key), key))
            out.extend(run)
            run.clear()

        for key in ordered:
            if run:
                previous = run[-1]
                n = len(contributions[previous]) + len(contributions[key])
                window = 4 * n * _ROUNDOFF * max(score[previous], score[key])
                if score[previous] - score[key] > window:
                    flush()
            run.append(key)
        flush()
        return out
