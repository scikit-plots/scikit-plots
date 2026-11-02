# scikitplot/corpus/_cut.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
How many hits to keep: a fixed count, a probability threshold, or the largest gap.

Notes
-----
**User.** Retrieval returns ``top_k`` hits whether the 7th is as good as the
1st or useless. :class:`CutPolicy` decides how many of a ranked list to keep:

``"fixed"``
    The first ``max_k``.
``"threshold"``
    :math:`K^* = \max\{k : s_k \ge \tau\}` -- every hit scoring at least
    ``threshold``, clamped to ``[min_k, max_k]``. Scores must be
    probabilities (in ``[0, 1]``): a raw BM25, cosine or rank-fusion score
    has no scale shared across queries, so one threshold would mean something
    different for every query. Use it on calibrated scores, such as the hits
    of probability fusion.
``"max_gap"``
    Cut after the largest drop :math:`s_k - s_{k+1}`, looking only at cuts
    that keep between ``min_k`` and ``max_k`` hits; the earliest of equal
    gaps wins. A large gap is a ranking boundary, not proof that the hits
    above it answer the question.

**Developer.** Pure and deterministic: the decision depends only on the
scores and the policy, and :class:`CutResult` records the rule's evidence
(the threshold or the gap) so a cut can be explained afterwards.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import Any, Sequence

__all__ = ["CUT_MODES", "CutPolicy", "CutResult"]

#: Accepted ``mode`` values.
CUT_MODES: tuple[str, ...] = ("fixed", "threshold", "max_gap")


def _count(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class CutResult:
    """
    Where a list was cut, and why.

    Parameters
    ----------
    k : int
        Number of hits kept.
    mode : str
        The rule applied.
    evidence : float or None
        The threshold for ``"threshold"``, the gap cut at for ``"max_gap"``
        (``None`` when no gap could be measured), ``None`` for ``"fixed"``.
    clamped : bool
        Whether ``min_k`` or ``max_k`` overrode the rule.
    """

    k: int
    mode: str
    evidence: float | None
    clamped: bool


@dataclasses.dataclass(frozen=True)
class CutPolicy:
    """
    A rule for how many ranked hits to keep.

    Parameters
    ----------
    mode : {"fixed", "threshold", "max_gap"}
    threshold : float or None, default=None
        Required for ``"threshold"``, in ``[0, 1]``; refused otherwise.
    min_k : int, default=1
        Fewest hits kept (when there are that many).
    max_k : int or None, default=None
        Most hits kept; required for ``"fixed"``.

    Raises
    ------
    ValueError
        For an unknown mode, a missing or out-of-range threshold, a threshold
        given to another mode, ``min_k < 0``, ``max_k < min_k``, or
        ``"fixed"`` without ``max_k``.

    Examples
    --------
    >>> CutPolicy("max_gap").cut([0.9, 0.85, 0.8, 0.3, 0.25]).k
    3
    >>> CutPolicy("threshold", threshold=0.5).cut([0.9, 0.6, 0.4]).k
    2
    """

    mode: str
    threshold: float | None = None
    min_k: int = 1
    max_k: int | None = None

    def __post_init__(self) -> None:
        """Validate the policy."""
        if self.mode not in CUT_MODES:
            raise ValueError(f"mode must be one of {CUT_MODES}, got {self.mode!r}")
        object.__setattr__(self, "min_k", _count(self.min_k, "min_k", 0))
        if self.max_k is not None:
            object.__setattr__(self, "max_k", _count(self.max_k, "max_k", self.min_k))
        if self.mode == "fixed" and self.max_k is None:
            raise ValueError('mode "fixed" needs max_k')
        if self.mode == "threshold":
            t = self.threshold
            if (
                isinstance(t, bool)
                or not isinstance(t, numbers.Real)
                or not 0 <= t <= 1
            ):
                raise ValueError(
                    f'mode "threshold" needs a threshold in [0, 1], got {t!r}'
                )
            object.__setattr__(self, "threshold", float(t))
        elif self.threshold is not None:
            raise ValueError(
                f'threshold applies to mode "threshold" only, not {self.mode!r}',
            )

    def cut(self, scores: Sequence[float]) -> CutResult:
        """
        Decide how many of ``scores`` (best first) to keep.

        Parameters
        ----------
        scores : sequence of float
            Ranked scores, non-increasing and finite.

        Returns
        -------
        CutResult

        Raises
        ------
        ValueError
            If the scores increase anywhere, are not finite, or -- for
            ``"threshold"`` -- lie outside ``[0, 1]``.
        """
        values = [float(s) for s in scores]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("scores must be finite")
        if any(b > a for a, b in zip(values, values[1:])):
            raise ValueError("scores must be ranked best first (non-increasing)")
        n = len(values)
        low = min(self.min_k, n)
        high = n if self.max_k is None else min(self.max_k, n)
        if self.mode == "fixed":
            return CutResult(high, "fixed", None, False)
        if self.mode == "threshold":
            if any(v < 0 or v > 1 for v in values):
                raise ValueError(
                    'mode "threshold" needs probabilities in [0, 1]; these scores '
                    "are not on a shared scale (calibrate them first)"
                )
            k = sum(1 for v in values if v >= self.threshold)
            kept = max(low, min(k, high))
            return CutResult(kept, "threshold", self.threshold, kept != k)
        best_k, best_gap = high, None
        for k in range(max(low, 1), high):
            gap = values[k - 1] - values[k]
            if best_gap is None or gap > best_gap:
                best_k, best_gap = k, gap
        return CutResult(best_k, "max_gap", best_gap, False)

    def apply(self, hits: Sequence[Any]) -> list[Any]:
        """
        Keep the hits this policy keeps, scored by their ``score`` attribute.

        Parameters
        ----------
        hits : sequence
            Ranked hits with a ``score`` attribute (``RetrievalHit``).

        Returns
        -------
        list
            The first :attr:`CutResult.k` hits.
        """
        hits = list(hits)
        return hits[: self.cut([h.score for h in hits]).k]
