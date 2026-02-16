# cython: language_level=3
"""
Scale a vector in-place using a typed memoryview (no NumPy required).

User notes
----------
- Works with any object exposing a writable buffer of doubles, e.g.:
  `array('d')`, `memoryview(array('d'))`, some other buffer providers.
- This does *not* rely on NumPy C-API.

Examples
--------
>>> from array import array
>>> import scikitplot.cython as cy
>>> m = cy.compile_template("easy_cython/t05_vector_scale_memview")
>>> a = array('d', [1.0, 2.0, 3.0])
>>> m.scale(a, 10.0)
>>> list(a)
[10.0, 20.0, 30.0]
"""

cpdef void scale(double[:] x, double alpha):
    """Scale x[i] *= alpha in-place."""
    cdef Py_ssize_t i, n = x.shape[0]
    for i in range(n):
        x[i] = x[i] * alpha
