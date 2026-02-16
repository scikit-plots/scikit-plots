# distutils: language = c++
# cython: language_level=3
"""
C++ std::vector interop demo (compile as C++).

This template is for advanced users; it requires a C++ compiler.
"""
from libcpp.vector cimport vector

cpdef int vector_size():
    cdef vector[int] v
    v.push_back(1)
    v.push_back(2)
    return <int>v.size()
