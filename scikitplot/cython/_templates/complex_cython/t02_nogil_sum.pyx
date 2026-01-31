# cython: language_level=3
"""
nogil numeric loop demo (no Python interaction in inner loop).
"""
cpdef double sum_squares(int n):
    cdef int i
    cdef double acc = 0.0
    with nogil:
        for i in range(n):
            acc += i * i
    return acc
