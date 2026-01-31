# cython: language_level=3
"""
Compute L2 norm using a typed memoryview (no NumPy required).
"""

from libc.math cimport sqrt

cpdef double norm2(double[:] x):
    """Return sqrt(sum(x[i]^2))."""
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double acc = 0.0
    for i in range(n):
        acc += x[i] * x[i]
    return sqrt(acc)
