# cython: language_level=3
"""Naive 1D valid convolution."""
cpdef void convolve_valid(double[:] x, double[:] k, double[:] out):
    cdef Py_ssize_t nx = x.shape[0]
    cdef Py_ssize_t nk = k.shape[0]
    cdef Py_ssize_t no = out.shape[0]
    if no != nx - nk + 1:
        raise ValueError("out length must be len(x)-len(k)+1")
    cdef Py_ssize_t i, j
    cdef double acc
    for i in range(no):
        acc = 0.0
        for j in range(nk):
            acc += x[i+j] * k[j]
        out[i] = acc
