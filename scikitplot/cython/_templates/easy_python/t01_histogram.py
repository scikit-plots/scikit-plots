"""Easy Python template: small algorithm with clear loops."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def histogram(values: Iterable[int]) -> dict[int, int]:
    """Count integer occurrences.

    Parameters
    ----------
    values : Iterable[int]
        Input integers.

    Returns
    -------
    dict[int, int]
        Mapping value -> count.
    """
    return dict(Counter(values))
