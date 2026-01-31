# cython: language_level=3

"""Easy Cython template: sum a 1D double memoryview."""

import cython


@cython.boundscheck(False)
@cython.wraparound(False)
def sum1d(double[:] x):
    cdef Py_ssize_t i
    cdef double s = 0.0
    for i in range(x.shape[0]):
        s += x[i]
    return s
