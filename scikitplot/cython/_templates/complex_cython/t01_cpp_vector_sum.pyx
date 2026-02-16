# cython: language_level=3
# distutils: language=c++
"""Complex Cython template: minimal C++ integration.

Demonstrates calling into a C++ standard library type.
Requires a working C++ toolchain.
"""

from libcpp.vector cimport vector


def sum_ints(list xs):
    cdef vector[int] v
    cdef int x
    for x in xs:
        v.push_back(<int>x)
    cdef long s = 0
    cdef size_t i
    for i in range(v.size()):
        s += v[i]
    return s
