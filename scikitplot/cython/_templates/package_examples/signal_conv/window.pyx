# cython: language_level=3
"""Window functions."""
from libc.math cimport cos
cdef double PI = 3.141592653589793

cpdef void hann(double[:] out):
    cdef Py_ssize_t i, n = out.shape[0]
    if n <= 1:
        for i in range(n):
            out[i] = 1.0
        return
    for i in range(n):
        out[i] = 0.5 - 0.5*cos(2.0*PI*i/(n-1))
