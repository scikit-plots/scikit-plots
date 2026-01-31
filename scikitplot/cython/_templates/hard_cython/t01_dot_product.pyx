# cython: language_level=3
"""Hard Cython template: dot product over memoryviews."""

from libc.stdint cimport int64_t


def dot_int64(long[:] a, long[:] b):
    cdef Py_ssize_t n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError("shape mismatch")
    cdef Py_ssize_t i
    cdef int64_t s = 0
    for i in range(n):
        s += <int64_t>a[i] * <int64_t>b[i]
    return s
