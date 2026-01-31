"""Devel Python template: NumPy usage (optional dependency)."""

from __future__ import annotations

from typing import Any


def as_float64_array(x: Any):
    """Convert input to a float64 NumPy array.

    Raises ImportError if NumPy is not installed.
    """
    import numpy as np  # noqa: PLC0415

    return np.asarray(x, dtype=np.float64)
