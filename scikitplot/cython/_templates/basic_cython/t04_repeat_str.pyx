# cython: language_level=3
"""
Repeat a Python string.

User notes
----------
- Demonstrates working with Python `str` objects in Cython.
- Strings remain Python objects; C speedups come from type-stable loops.
"""

cpdef str repeat(str s, int n):
    """
    Repeat `s` `n` times.

    Parameters
    ----------
    s : str
        Input string.
    n : int
        Repeat count (must be >= 0).

    Returns
    -------
    str
        Repeated string.

    Raises
    ------
    ValueError
        If n is negative.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    return s * n
