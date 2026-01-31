# cython: language_level=3
"""
Vector dot product on Python lists (no NumPy).

User notes
----------
- Accepts Python sequences (lists/tuples) of numbers.
- Converts elements to `double` inside a typed loop.

Developer notes
--------------
- This is a pedagogical template: speed depends on input types.
- For best performance, see memoryview-based templates.
"""

cpdef double dot_list(object x, object y):
    """
    Compute dot(x, y) for two equally-sized sequences.

    Parameters
    ----------
    x, y : sequence
        Input sequences of numbers.

    Returns
    -------
    double
        Dot product.

    Raises
    ------
    ValueError
        If lengths differ.
    """
    cdef Py_ssize_t n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")
    cdef Py_ssize_t i
    cdef double acc = 0.0
    for i in range(n):
        acc += <double>x[i] * <double>y[i]
    return acc
