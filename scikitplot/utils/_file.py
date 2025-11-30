# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikit-plots utility file helpers.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# import os
# import re
# import shutil

# # from pathlib import Path
# from datetime import datetime

SUFFIXES = ["B", "KB", "MB", "GB", "TB", "PB"]

__all__ = [
    "humansize",
    "humansize_vector",
]


def humansize(
    nbytes: float | object,
    suffixes: list[str] | None = None,
) -> str:
    """
    Convert a byte count into a human-readable string.

    Parameters
    ----------
    nbytes : int or float
        Number of bytes.
    suffixes : list[str], optional
        Custom suffix list.

    Returns
    -------
    str
        Human-readable size, e.g. "123 MB".

    Examples
    --------
    >>> humansize(2048)
    '2 KB'
    """
    suffixes = suffixes or SUFFIXES
    try:
        n = float(nbytes)
    except Exception:
        return str(nbytes)

    neg = n < 0
    n = abs(n)

    i = 0
    while n >= 1024 and i < len(suffixes) - 1:  # noqa: PLR2004
        n /= 1024
        i += 1

    f = f"{n:.2f}".rstrip("0").rstrip(".")
    return f"{'-' if neg else ''}{f} {suffixes[i]}"


def humansize_vector(values, suffixes=None):
    """
    Vectorized form of :func:`humansize`.

    Accepts scalars, lists, numpy arrays, or pandas Series.

    Examples
    --------
    >>> humansize_vector([1024, 10_000_000])
    array(['1 KB', '9.54 MB'], dtype=object)
    """
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return humansize(values, suffixes)

    arr = np.asarray(values, dtype="object")
    func = np.vectorize(lambda x: humansize(x, suffixes))
    return func(arr)
