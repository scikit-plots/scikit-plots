# cython: language_level=3
"""Simple histogram on integer data."""
include "common.pxi"

cpdef list hist_int(object xs, int nbins):
    if nbins <= 0:
        raise ValueError("nbins must be > 0")
    cdef list out = [0] * nbins
    cdef Py_ssize_t i, n = len(xs)
    cdef int v
    for i in range(n):
        v = <int>xs[i]
        if 0 <= v < nbins:
            out[v] += 1
    return out
