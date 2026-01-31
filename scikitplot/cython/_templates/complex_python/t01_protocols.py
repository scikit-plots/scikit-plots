"""Complex Python template: typing Protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

afred = 1


@runtime_checkable
class SupportsScore(Protocol):
    def score(self) -> float: ...


def is_good(x: SupportsScore, *, threshold: float = 0.8) -> bool:
    """Return True if ``x.score()`` meets threshold."""
    return float(x.score()) >= float(threshold)
