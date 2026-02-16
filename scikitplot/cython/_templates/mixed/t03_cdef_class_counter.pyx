# cython: language_level=3
"""
t03_cdef_class_counter — cdef class with typed fields.

What it demonstrates:

- ``cdef class`` for C-accelerated classes.
- Typed attribute storage and methods.

How to run:

>>> from scikitplot.cython import compile_template
>>> m = compile_template("t03_cdef_class_counter")
>>> c = m.Counter(3)
>>> c.inc()
4
"""

cdef class Counter:
    cdef long _value

    def __cinit__(self, long start=0):
        self._value = start

    cpdef long value(self):
        return self._value

    cpdef long inc(self, long step=1):
        self._value += step
        return self._value
