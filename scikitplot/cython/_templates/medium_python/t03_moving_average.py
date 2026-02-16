"""Compute a simple moving average."""

from typing import List, Sequence


def moving_average(x: Sequence[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be > 0")
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(x):
        s += float(v)
        if i >= window:
            s -= float(x[i - window])
        if i >= window - 1:
            out.append(s / window)
    return out
