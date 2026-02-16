"""Devel NumPy Python template: optional NumPy usage."""

from __future__ import annotations

from typing import Sequence


def maybe_mean(x: Sequence[float]) -> float:
    """Compute mean; uses NumPy if installed, otherwise pure Python."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        xs = [float(v) for v in x]
        if not xs:
            raise ValueError("empty")
        return sum(xs) / len(xs)
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise ValueError("empty")
    return float(arr.mean())
