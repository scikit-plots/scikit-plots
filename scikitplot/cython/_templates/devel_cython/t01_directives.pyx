# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
"""Devel Cython template: compiler directives.

Shows how to use directives safely and explicitly.
"""


def axpy(double a, double[:] x, double[:] y):
    """Compute y <- a*x + y."""
    cdef Py_ssize_t i, n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("shape mismatch")
    for i in range(n):
        y[i] = a * x[i] + y[i]
    return None
