# cython: language_level=3
"""
t18_histogram_int — histogram counting (NumPy int32).

What it demonstrates
--------------------
- Counting histogram bins with typed ndarrays.
- Strict bounds checks on bin range.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> import numpy as np
>>> m = compile_template("t18_histogram_int")
>>> x = np.array([0, 1, 1, 2], dtype=np.int32)
>>> m.hist_counts(x, 3).tolist()
[1, 2, 1]
"""

cimport numpy as cnp
import numpy as np


def hist_counts(np.ndarray[cnp.int32_t, ndim=1] x, int n_bins):
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    cdef np.ndarray[cnp.int64_t, ndim=1] out = np.zeros(n_bins, dtype=np.int64)
    cdef Py_ssize_t i, n = x.shape[0]
    cdef int v
    for i in range(n):
        v = x[i]
        if v < 0 or v >= n_bins:
            raise ValueError("value out of range")
        out[v] += 1
    return out
