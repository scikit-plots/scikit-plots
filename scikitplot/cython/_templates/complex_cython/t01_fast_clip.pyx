# cython: language_level=3
# cython: boundscheck=False, wraparound=False
"""Complex Cython template: clip values in-place on a memoryview."""


def clip_inplace(double[:] x, double lo, double hi):
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        if x[i] < lo:
            x[i] = lo
        elif x[i] > hi:
            x[i] = hi
    return None
