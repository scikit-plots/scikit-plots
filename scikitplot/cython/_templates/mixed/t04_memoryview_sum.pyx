# cython: language_level=3
"""
t04_memoryview_sum — sum a 1D typed memoryview.

What it demonstrates:

- Using typed memoryviews (buffer protocol) without importing NumPy.
- Strict typing for predictable performance.

How to run:

>>> from scikitplot.cython import compile_template
>>> m = compile_template("t04_memoryview_sum")
>>> m.sum_double([1.0, 2.0, 3.0])
6.0

Notes
-----
Python lists do not expose a buffer, so pass an ``array('d')`` or ``memoryview``
of a suitable buffer for best results:

>>> import array
>>> a = array.array('d', [1.0, 2.0, 3.0])
>>> m.sum_double(a)
6.0
"""


def sum_double(double[:] x):

    cdef Py_ssize_t i, n = x.shape[0]
    cdef double s = 0.0

    for i in range(n):
        s += x[i]
    return s
