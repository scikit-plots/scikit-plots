# cython: language_level=3
"""
t11_fused_types_dot — fused types dot product.

What it demonstrates
--------------------
- Fused types to generate specialized versions for multiple numeric types.
- Typed memoryviews for flexible inputs.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> import array
>>> m = compile_template("t11_fused_types_dot")
>>> a = array.array('d', [1.0, 2.0, 3.0])
>>> b = array.array('d', [4.0, 5.0, 6.0])
>>> m.dot(a, b)
32.0
"""

ctypedef fused floating:
    float
    double


def dot(floating[:] a, floating[:] b):
    cdef Py_ssize_t i, n = a.shape[0]
    if b.shape[0] != n:
        raise ValueError("shapes must match")
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s
