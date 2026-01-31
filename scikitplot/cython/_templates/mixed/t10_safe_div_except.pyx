# cython: language_level=3
"""
t10_safe_div_except — exception-aware cpdef.

What it demonstrates
--------------------
- ``cpdef`` with an explicit exception behavior.
- Strict error handling (division by zero).

How to run
----------
>>> from scikitplot.cython import compile_template
>>> m = compile_template("t10_safe_div_except")
>>> m.safe_div(10, 2)
5
"""

cpdef int safe_div(int a, int b) except? -1:
    if b == 0:
        raise ZeroDivisionError("b must be non-zero")
    return a // b
