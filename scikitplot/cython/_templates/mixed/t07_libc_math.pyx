# cython: language_level=3
"""
t07_libc_math — libc math usage.

What it demonstrates:

- ``cimport`` from ``libc.math`` for fast math operations.
- A pure C-style loop over a memoryview.

How to run:

>>> from scikitplot.cython import compile_template
>>> import array
>>> m = compile_template("t07_libc_math")
>>> a = array.array('d', [0.0, 1.0, 4.0])
>>> m.sqrt_sum(a)
3.0
"""

from libc.math cimport sqrt


def sqrt_sum(double[:] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += sqrt(x[i])
    return s
