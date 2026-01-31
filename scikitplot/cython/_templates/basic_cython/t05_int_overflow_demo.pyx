# cython: language_level=3
"""
Demonstrate fixed-width integer behavior.

Developer notes
--------------
- Uses `int32_t` from libc.stdint for predictable overflow semantics.
- This is a learning template; overflow is intentional.
"""

from libc.stdint cimport int32_t

cpdef int32_t add_wrap32(int32_t a, int32_t b):
    """Add two 32-bit ints (wraps on overflow)."""
    return <int32_t>(a + b)
