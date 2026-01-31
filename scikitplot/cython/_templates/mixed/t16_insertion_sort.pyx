# cython: language_level=3
"""
t16_insertion_sort — insertion sort on a typed memoryview.

What it demonstrates:

- In-place algorithms on typed memoryviews.
- Strict algorithm with deterministic output.

How to run:

>>> from scikitplot.cython import compile_template
>>> import array
>>> m = compile_template("t16_insertion_sort")
>>> a = array.array('d', [3.0, 1.0, 2.0])
>>> m.insertion_sort(a)
>>> list(a)
[1.0, 2.0, 3.0]
"""


def insertion_sort(double[:] x):
    cdef Py_ssize_t i, j, n = x.shape[0]
    cdef double key
    for i in range(1, n):
        key = x[i]
        j = i - 1
        while j >= 0 and x[j] > key:
            x[j + 1] = x[j]
            j -= 1
        x[j + 1] = key
