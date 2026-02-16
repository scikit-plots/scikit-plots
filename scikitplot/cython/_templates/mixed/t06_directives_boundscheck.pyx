# cython: language_level=3
"""
t06_directives_boundscheck — directives for speed.

What it demonstrates
--------------------
- ``cython.boundscheck(False)`` and ``cython.wraparound(False)``.
- Local, explicit performance directives (no global magic).

How to run
----------
>>> from scikitplot.cython import compile_template
>>> import numpy as np
>>> m = compile_template("t06_directives_boundscheck")
>>> m.prefix_sum(np.array([1, 2, 3], dtype=np.int64)).tolist()
[1, 3, 6]
"""

import cython
cimport numpy as cnp
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def prefix_sum(np.ndarray[cnp.int64_t, ndim=1] x):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef np.ndarray[cnp.int64_t, ndim=1] out = np.empty(n, dtype=np.int64)
    cdef long long acc = 0
    for i in range(n):
        acc += x[i]
        out[i] = acc
    return out
