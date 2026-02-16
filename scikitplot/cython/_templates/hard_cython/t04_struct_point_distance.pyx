# cython: language_level=3
"""
C struct example: 2D point and distance.

Developer notes
--------------
- Demonstrates `cdef struct` and a `nogil` helper for numeric work.
"""
from libc.math cimport sqrt

cdef struct Point2:
    double x
    double y

cdef inline double _dist(Point2 a, Point2 b) nogil:
    cdef double dx = a.x - b.x
    cdef double dy = a.y - b.y
    return sqrt(dx*dx + dy*dy)

cpdef double dist(double ax, double ay, double bx, double by):
    cdef Point2 a
    cdef Point2 b
    a.x = ax
    a.y = ay
    b.x = bx
    b.y = by
    return _dist(a, b)
