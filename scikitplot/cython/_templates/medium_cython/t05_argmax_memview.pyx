# cython: language_level=3
"""
Argmax on double[:] memoryview.
"""

cpdef Py_ssize_t argmax(double[:] x):
    cdef Py_ssize_t n = x.shape[0]
    if n == 0:
        raise ValueError("empty")
    cdef Py_ssize_t i, best = 0
    cdef double bestv = x[0]
    for i in range(1, n):
        if x[i] > bestv:
            bestv = x[i]
            best = i
    return best
