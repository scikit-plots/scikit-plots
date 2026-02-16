# cython: language_level=3
"""
t02_fib_cpdef — cpdef Fibonacci (iterative).

What it demonstrates:

- ``cpdef``: a function that is callable efficiently from both Cython and Python.
- Strict input validation (no heuristics).

How to run:

>>> from scikitplot.cython import compile_template
>>> m = compile_template("t02_fib_cpdef")
>>> m.fib(10)
55
"""

cpdef long fib(int n):
    # Strict validation: negative inputs are rejected deterministically.
    if n < 0:
        raise ValueError("n must be >= 0")

    cdef long a = 0
    cdef long b = 1
    cdef int i

    for i in range(n):
        a, b = b, a + b
    return a
