# cython: language_level=3
"""
Toggle a boolean.

Notes
-----
Cython maps `bint` to a C integer used for booleans.
"""

cpdef bint toggle(bint flag):
    """Return logical not of flag (as bint)."""
    return not flag
