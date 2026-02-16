# cython: language_level=3
"""Compute degrees from adjacency list."""
cpdef list degrees(object adj):
    cdef Py_ssize_t n = len(adj)
    cdef list out = [0] * n
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = len(adj[i])
    return out
