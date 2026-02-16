# cython: language_level=3
"""Reductions on sequences (no NumPy required)."""
include "common.pxi"

cpdef double sum_mem(double[:] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double acc = 0.0
    for i in range(n):
        acc += x[i]
    return acc

cpdef double mean_mem(double[:] x):
    if x.shape[0] == 0:
        raise ValueError("empty")
    return sum_mem(x) / x.shape[0]
