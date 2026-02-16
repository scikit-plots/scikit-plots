# cython: language_level=3
"""Devel NumPy C-API Cython template.

Requires NumPy at compile time.
"""

# import numpy as np
cimport numpy as cnp


def sum_float64(cnp.ndarray[cnp.float64_t, ndim=1] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += x[i]
    return s
