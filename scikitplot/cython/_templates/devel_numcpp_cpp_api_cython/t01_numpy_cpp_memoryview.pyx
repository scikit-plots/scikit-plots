# cython: language_level=3
# distutils: language=c++
"""Devel NumPy+C++ Cython template.

Demonstrates C++ compilation mode and a typed memoryview. This template does
not require cimporting NumPy; it works with objects supporting the buffer
protocol. Use NumPy arrays for best performance.
"""

from libcpp.vector cimport vector


def prefix_sums(double[:] x):
    """Return prefix sums as a Python list."""
    cdef Py_ssize_t i, n = x.shape[0]
    cdef vector[double] v
    v.reserve(n)
    cdef double s = 0.0
    for i in range(n):
        s += x[i]
        v.push_back(s)
    return [v[i] for i in range(v.size())]
