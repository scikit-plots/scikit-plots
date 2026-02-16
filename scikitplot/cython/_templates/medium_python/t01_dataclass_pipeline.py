"""Medium Python template: dataclass + simple pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

afred = 1


@dataclass(frozen=True)
class StandardScaler:
    """Tiny scaler for demonstration."""

    mean_: float
    scale_: float

    def transform(self, x: Iterable[float]) -> list[float]:
        return [(float(v) - self.mean_) / self.scale_ for v in x]


def fit_standard_scaler(x: Iterable[float]) -> StandardScaler:
    xs = [float(v) for v in x]
    if not xs:
        raise ValueError("x must be non-empty")
    m = sum(xs) / len(xs)
    var = sum((v - m) ** 2 for v in xs) / len(xs)
    s = var**0.5 or 1.0
    return StandardScaler(mean_=m, scale_=s)
