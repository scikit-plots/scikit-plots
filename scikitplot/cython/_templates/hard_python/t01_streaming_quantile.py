"""Hard Python template: deterministic reservoir-free approximate quantile stub.

Note: This is a *template stub* describing interfaces. Replace with a real
quantile sketch (e.g. t-digest) if needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantileSketch:
    """Template interface for a quantile sketch."""

    def update(self, x: float) -> None:
        raise NotImplementedError

    def quantile(self, q: float) -> float:
        raise NotImplementedError
