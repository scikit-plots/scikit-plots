# cython: language_level=3
"""
Add two numeric sequences and return a Python list.

Notes
-----
This is a learning template that demonstrates:
- typed loop counters (`Py_ssize_t`)
- explicit conversion to `double` for numeric stability.
"""

cpdef list add_list(object x, object y):
    cdef Py_ssize_t n = len(x)
    if len(y) != n:
        raise ValueError("length mismatch")
    cdef list out = [0.0] * n
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = <double>x[i] + <double>y[i]
    return out
