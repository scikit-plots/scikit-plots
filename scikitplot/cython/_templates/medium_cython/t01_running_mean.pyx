# cython: language_level=3
"""Medium Cython template: a small stateful `cdef class`."""

cdef class RunningMean:
    cdef double _sum
    cdef long _n

    def __cinit__(self):
        self._sum = 0.0
        self._n = 0

    def update(self, double x):
        self._sum += x
        self._n += 1

    def value(self):
        if self._n == 0:
            return 0.0
        return self._sum / self._n
