# cython: language_level=3
"""
t20_matmul_small — naive 2D matrix multiplication (memoryviews).

What it demonstrates:

- 2D typed memoryviews and triple loops.
- Strict dimension checks.

How to run:

>>> from scikitplot.cython import compile_template
>>> import numpy as np
>>> m = compile_template("t20_matmul_small")
>>> A = np.array([[1., 2.],[3., 4.]], dtype=np.float64)
>>> B = np.array([[5., 6.],[7., 8.]], dtype=np.float64)
>>> m.matmul(A, B)
array([[19., 22.],
       [43., 50.]])
"""

cimport numpy as cnp
import numpy as np


def matmul(np.ndarray[cnp.float64_t, ndim=2] A, np.ndarray[cnp.float64_t, ndim=2] B):
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t m = A.shape[0]
    cdef Py_ssize_t n = A.shape[1]
    if B.shape[0] != n:
        raise ValueError("shape mismatch")
    cdef Py_ssize_t p = B.shape[1]
    cdef np.ndarray[cnp.float64_t, ndim=2] C = np.zeros((m, p), dtype=np.float64)
    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]
    return C
