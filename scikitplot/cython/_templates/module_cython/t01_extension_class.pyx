# cython: language_level=3
"""Module Cython template: extension class."""

cdef class Counter:
    cdef long value

    def __cinit__(self, long start=0):
        self.value = start

    def inc(self, long by=1):
        self.value += by
        return self.value

    def get(self):
        return self.value
