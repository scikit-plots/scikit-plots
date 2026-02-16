# cython: language_level=3
"""
XOR two bytes objects (length must match).

User notes
----------
- Demonstrates working with `bytes` and indexing as integers 0..255.
- Output is a new `bytes` object.
"""

cpdef bytes xor_bytes(bytes a, bytes b):
    if len(a) != len(b):
        raise ValueError("length mismatch")
    cdef Py_ssize_t n = len(a)
    cdef bytearray out = bytearray(n)
    cdef Py_ssize_t i
    for i in range(n):
        out[i] = a[i] ^ b[i]
    return bytes(out)
