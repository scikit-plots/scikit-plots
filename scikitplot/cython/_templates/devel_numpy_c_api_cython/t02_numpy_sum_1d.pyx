# cython: language_level=3
"""
Sum a 1D NumPy array using typed memoryviews.

Requires NumPy (template metadata marks it).
"""

# import numpy as np
cimport numpy as cnp
cnp.import_array()


cpdef double sum1d(cnp.ndarray[cnp.double_t, ndim=1] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double acc = 0.0
    for i in range(n):
        acc += <double>x[i]
    return acc
