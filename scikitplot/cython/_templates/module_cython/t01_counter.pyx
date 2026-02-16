# cython: language_level=3
"""Module Cython template: cdef class with Python-visible API."""

cdef class Counter:
    cdef long _n

    def __cinit__(self, long start=0):
        self._n = start

    def inc(self, long delta=1):
        self._n += delta
        return self._n

    def value(self):
        return self._n
