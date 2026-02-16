# cython: language_level=3
"""
t01_square_int — typed integer function (minimal).

What it demonstrates:

- Declaring C-typed arguments in a Python-visible function.
- Returning Python integers from C-typed arithmetic.

How to run:

>>> from scikitplot.cython import compile_template
>>> m = compile_template("t01_square_int")
>>> m.f(10)
100
"""


def f(int n):
    # A C-typed local variable is faster than Python ints for arithmetic.
    cdef int x = n
    return x * x
