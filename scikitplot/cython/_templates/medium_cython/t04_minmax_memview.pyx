# cython: language_level=3
"""
Compute min and max from a double[:] memoryview.
"""

cpdef tuple minmax(double[:] x):
    cdef Py_ssize_t n = x.shape[0]
    if n == 0:
        raise ValueError("empty")
    cdef Py_ssize_t i
    cdef double mn = x[0]
    cdef double mx = x[0]
    for i in range(1, n):
        if x[i] < mn:
            mn = x[i]
        if x[i] > mx:
            mx = x[i]
    return mn, mx
