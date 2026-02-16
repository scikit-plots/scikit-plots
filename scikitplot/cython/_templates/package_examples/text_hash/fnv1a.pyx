# cython: language_level=3
"""FNV-1a hash for bytes."""
from libc.stdint cimport uint64_t

cpdef uint64_t fnv1a(bytes data):
    cdef uint64_t h = 1469598103934665603
    cdef uint64_t prime = 1099511628211
    cdef Py_ssize_t i, n = len(data)
    for i in range(n):
        h ^= data[i]
        h *= prime
    return h
