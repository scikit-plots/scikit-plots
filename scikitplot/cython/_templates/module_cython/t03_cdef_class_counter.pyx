# cython: language_level=3
"""cdef class demo."""


cdef class Counter:
    cdef long long value

    def __cinit__(self, long long start=0):
        self.value = start

    cpdef void inc(self, long long step=1):
        self.value += step

    cpdef long long get(self):
        return self.value
