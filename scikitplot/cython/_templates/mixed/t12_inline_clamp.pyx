# cython: language_level=3
"""
t12_inline_clamp — cdef inline helper.

What it demonstrates
--------------------
- ``cdef inline`` helper to avoid repetition and keep hot loops tight.
- Clean separation of helper vs public function.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> import array
>>> m = compile_template("t12_inline_clamp")
>>> x = array.array('d', [-1.0, 0.5, 2.0])
>>> m.clamp01(x).tolist()
[0.0, 0.5, 1.0]
"""


cdef inline double clamp01_scalar(double x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def clamp01(double[:] x):
    cdef Py_ssize_t i, n = x.shape[0]
    # We return a Python list to keep this template dependency-free.
    out = [0.0] * n
    for i in range(n):
        out[i] = clamp01_scalar(x[i])
    return out
