# cython: language_level=3
"""Vector ops on double[:] memoryviews (no NumPy required)."""
from libc.math cimport sqrt
include "common.pxi"

cpdef void add_inplace(double[:] x, double[:] y):
    cdef Py_ssize_t n = _require_same_len(x, y)
    cdef Py_ssize_t i
    for i in range(n):
        x[i] += y[i]

cpdef double dot(double[:] x, double[:] y):
    cdef Py_ssize_t n = _require_same_len(x, y)
    cdef Py_ssize_t i
    cdef double acc = 0.0
    for i in range(n):
        acc += x[i] * y[i]
    return acc

cpdef double norm2(double[:] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double acc = 0.0
    for i in range(n):
        acc += x[i] * x[i]
    return sqrt(acc)
