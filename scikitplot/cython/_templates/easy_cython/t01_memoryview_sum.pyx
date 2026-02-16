# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""Easy Cython template: typed memoryview loop."""

# import cython


def sum_doubles(double[:] x):
    """Sum a 1D array-like of doubles."""
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += x[i]
    return s
