# cython: language_level=3
"""
t19_popcount — portable bit population count.

What it demonstrates
--------------------
- Bitwise operations on unsigned ints.
- Portability: no compiler-specific builtins.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> m = compile_template("t19_popcount")
>>> m.popcount32(0b1011)
3
"""

cpdef int popcount32(unsigned int x):
    cdef int count = 0
    while x:
        x &= x - 1  # clear lowest set bit
        count += 1
    return count
