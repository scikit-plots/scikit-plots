# cython: language_level=3
"""Hard Cython template: cdef class with manual memory.

This template shows a simple vector type backed by a C array.
"""

from libc.stdlib cimport malloc, free

cdef class IntVector:
    cdef int* data
    cdef Py_ssize_t n

    def __cinit__(self, Py_ssize_t n):
        if n < 0:
            raise ValueError("n must be >= 0")
        self.n = n
        self.data = <int*>malloc(n * sizeof(int)) if n else <int*>0
        if n and self.data == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.data != NULL:
            free(self.data)
            self.data = NULL

    def __len__(self):
        return self.n

    def set(self, Py_ssize_t i, int v):
        if i < 0 or i >= self.n:
            raise IndexError(i)
        self.data[i] = v

    def get(self, Py_ssize_t i):
        if i < 0 or i >= self.n:
            raise IndexError(i)
        return self.data[i]
