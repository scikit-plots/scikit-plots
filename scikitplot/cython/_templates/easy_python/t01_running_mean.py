"""Easy Python template: streaming statistic.

Shows stateful iteration and a small utility.
"""

from __future__ import annotations

from typing import Iterable, Iterator


def running_mean(values: Iterable[float]) -> Iterator[float]:
    """Yield the running mean of *values*."""
    total = 0.0
    n = 0
    for x in values:
        n += 1
        total += float(x)
        yield total / n
