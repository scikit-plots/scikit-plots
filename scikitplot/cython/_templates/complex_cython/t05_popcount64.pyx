# cython: language_level=3
"""
Popcount (count set bits) for 64-bit integers.

Notes
-----
Uses a classic bit-twiddling algorithm (Hacker's Delight).
"""
from libc.stdint cimport uint64_t

cpdef int popcount(uint64_t x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return <int>(x & 0x7F)
