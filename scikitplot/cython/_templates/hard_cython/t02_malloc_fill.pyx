# cython: language_level=3
"""
Manual allocation with malloc/free (advanced).
"""

from libc.stdlib cimport malloc, free


cpdef int sum_range(int n):

    cdef int* buf = <int*>malloc(n * sizeof(int))

    if buf == NULL:
        raise MemoryError()

    cdef int i
    cdef long long acc = 0

    try:
        for i in range(n):
            buf[i] = i
        for i in range(n):
            acc += buf[i]
        return <int>acc
    finally:
        free(buf)
