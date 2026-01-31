# cython: language_level=3
"""
Multi-file template: use a shipped header file.

User notes
----------
- Demonstrates `cdef extern from "header.h"` for C declarations.
- The header is copied into the build directory via metadata `support_paths`.
"""

cdef extern from "helper_add.h":
    int add_int(int a, int b)

cpdef int add(int a, int b):
    """Call `add_int` declared in an external header."""
    return add_int(a, b)
