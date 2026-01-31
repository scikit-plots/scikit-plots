# cython: language_level=3
"""Easy Cython template: summing a 1D typed memoryview."""

from libc.stdint cimport int64_t


def sum_int64(long[:] x):
    """Sum a 1D buffer of integers.

    Notes
    -----
    Accepts objects exporting the buffer protocol (e.g., array('l'), NumPy arrays).
    """
    cdef Py_ssize_t i, n = x.shape[0]
    cdef int64_t s = 0
    for i in range(n):
        s += <int64_t>x[i]
    return s
