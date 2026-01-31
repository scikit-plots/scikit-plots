# cython: language_level=3
"""
Parse a non-negative integer from ASCII bytes.

User notes
----------
- Useful micro-template for fast parsing in tight loops.
- Accepts `bytes` containing only digits (and optionally leading/trailing spaces).

Raises
------
ValueError on invalid input.
"""

cpdef int parse_uint(bytes s):
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = len(s)
    # skip leading spaces
    while i < n and s[i] in (32, 9, 10, 13):
        i += 1
    if i == n:
        raise ValueError("empty")
    cdef int acc = 0
    cdef int d
    while i < n and 48 <= s[i] <= 57:
        d = s[i] - 48
        acc = acc * 10 + d
        i += 1
    # skip trailing spaces
    while i < n and s[i] in (32, 9, 10, 13):
        i += 1
    if i != n:
        raise ValueError("invalid character")
    return acc
