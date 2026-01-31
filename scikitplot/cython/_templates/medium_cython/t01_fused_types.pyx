# cython: language_level=3
"""Medium Cython template: fused types.

Demonstrates writing one function that works for multiple numeric types.
"""

ctypedef fused num_t:
    int
    long
    double


def dot(num_t[:] a, num_t[:] b):
    """Return dot product of two 1D vectors."""
    cdef Py_ssize_t i, n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError("shape mismatch")
    cdef double s = 0.0
    for i in range(n):
        s += (<double>a[i]) * (<double>b[i])
    return s
