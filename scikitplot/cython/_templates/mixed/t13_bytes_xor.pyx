# cython: language_level=3
"""
t13_bytes_xor — bytes processing with memoryviews.

What it demonstrates:

- Using ``unsigned char[:]`` memoryviews.
- Creating output as ``bytes`` deterministically.

How to run:

>>> from scikitplot.cython import compile_template
>>> m = compile_template("t13_bytes_xor")
>>> m.xor_bytes(b"abc", 1)
b'`cb'
"""


def xor_bytes(bytes data, unsigned char key):
    cdef unsigned char[:] src = data
    cdef Py_ssize_t i, n = src.shape[0]
    out = bytearray(n)
    cdef unsigned char[:] dst = out
    for i in range(n):
        dst[i] = src[i] ^ key
    return bytes(out)
