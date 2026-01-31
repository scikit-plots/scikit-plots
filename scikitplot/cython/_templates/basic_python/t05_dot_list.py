"""Dot product on sequences (pure Python)."""

from typing import Sequence


def dot_list(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("length mismatch")
    return sum(a * b for a, b in zip(x, y))
