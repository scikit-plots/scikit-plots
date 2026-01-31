# cython: language_level=3
"""
t08_struct_point — cdef struct.

What it demonstrates
--------------------
- Defining a ``cdef struct`` and using it for computation.
- Returning Python floats from C math.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> m = compile_template("t08_struct_point")
>>> m.dist2(0.0, 0.0, 3.0, 4.0)
25.0
"""


cdef struct Point:
    double x
    double y


def dist2(double x1, double y1, double x2, double y2):
    cdef Point a
    cdef Point b

    a.x = x1
    a.y = y1
    b.x = x2
    b.y = y2

    cdef double dx = a.x - b.x
    cdef double dy = a.y - b.y
    return dx*dx + dy*dy
