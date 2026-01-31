# cython: language_level=3
"""
t17_kahan_sum — Kahan summation.

What it demonstrates
--------------------
- Numerically stable summation using compensation.
- Useful in ML metrics aggregation.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> import array
>>> m = compile_template("t17_kahan_sum")
>>> a = array.array('d', [1e100, 1.0, -1e100])
>>> m.kahan_sum(a)
1.0
"""


def kahan_sum(double[:] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double s = 0.0
    cdef double c = 0.0  # compensation
    cdef double y, t
    for i in range(n):
        y = x[i] - c
        t = s + y
        c = (t - s) - y
        s = t
    return s
