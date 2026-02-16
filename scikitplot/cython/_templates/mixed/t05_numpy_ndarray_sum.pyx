# cython: language_level=3
"""
t05_numpy_ndarray_sum — typed NumPy ndarray loop.

What it demonstrates:

- ``cimport numpy`` and typed ndarrays for fast loops.
- Enforcing dtype/ndim at call time.

How to run:

>>> from scikitplot.cython import compile_template
>>> import numpy as np
>>> m = compile_template("t05_numpy_ndarray_sum")
>>> m.sum_int32(np.array([1, 2, 3], dtype=np.int32))
6
"""

cimport numpy as cnp
import numpy as np


def sum_int32(np.ndarray[cnp.int32_t, ndim=1] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef long s = 0
    for i in range(n):
        s += x[i]
    return s
