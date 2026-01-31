# cython: language_level=3
"""Mean/variance (Welford) on double[:] memoryviews."""
include "common.pxi"

cpdef tuple mean_var(double[:] x):
    cdef Py_ssize_t n = x.shape[0]
    if n == 0:
        raise ValueError("empty")
    cdef double mean = 0.0
    cdef double m2 = 0.0
    cdef Py_ssize_t i
    cdef double delta
    for i in range(n):
        delta = x[i] - mean
        mean += delta / (i + 1)
        m2 += delta * (x[i] - mean)
    cdef double var = m2 / (n - 1) if n > 1 else 0.0
    return mean, var
