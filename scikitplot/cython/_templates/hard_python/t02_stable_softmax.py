"""Stable softmax (numerical stability)."""

from math import exp
from typing import Sequence


def softmax(x: Sequence[float]) -> list[float]:
    if not x:
        return []
    m = max(x)
    exps = [exp(v - m) for v in x]
    s = sum(exps)
    return [v / s for v in exps]
