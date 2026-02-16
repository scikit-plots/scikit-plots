# distutils: language=c++
# cython: language_level=3
"""Developer Cython template: C++ language mode.

This template sets `language=c++` via distutils directive.
"""

cdef extern from "math.h":
    double sqrt(double)


def hypot(double a, double b):
    return sqrt(a*a + b*b)
