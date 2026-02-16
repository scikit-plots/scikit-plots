# distutils: language = c++
# cython: language_level=3

"""Devel Cython template: C++ + memoryview interop."""

from libcpp.vector cimport vector


def prefix_sums(double[:] x):
    cdef Py_ssize_t i
    cdef vector[double] out
    out.reserve(x.shape[0])
    cdef double s = 0.0
    for i in range(x.shape[0]):
        s += x[i]
        out.push_back(s)
    # Return as Python list for portability
    return [out[i] for i in range(out.size())]
