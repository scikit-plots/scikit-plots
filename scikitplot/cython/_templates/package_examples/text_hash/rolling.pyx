# cython: language_level=3
"""Rolling hash for bytes (simple)."""
from libc.stdint cimport uint64_t

cpdef uint64_t rolling_sum(bytes data, int window):
    if window <= 0:
        raise ValueError("window must be > 0")
    cdef Py_ssize_t n = len(data)
    if n < window:
        return 0
    cdef uint64_t s = 0
    cdef Py_ssize_t i
    for i in range(window):
        s += data[i]
    # Return only first window sum for simplicity (learning template)
    return s
