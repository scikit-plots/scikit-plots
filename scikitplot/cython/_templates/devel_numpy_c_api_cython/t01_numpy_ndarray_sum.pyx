# cython: language_level=3

"""Devel Cython template: NumPy ndarray typed access.

Requires NumPy.
"""

cimport numpy as cnp
import numpy as np


def sum_ndarray(object arr):
    cdef cnp.ndarray[cnp.double_t, ndim=1] x = np.asarray(arr, dtype=np.float64)
    cdef Py_ssize_t i
    cdef double s = 0.0
    for i in range(x.shape[0]):
        s += x[i]
    return s
