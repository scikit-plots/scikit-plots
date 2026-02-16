# cython: language_level=3

"""Hard Cython template: cdef class counter."""

cdef class Counter:
    cdef long value

    def __cinit__(self, long start=0):
        self.value = start

    cpdef long inc(self, long step=1):
        self.value += step
        return self.value
