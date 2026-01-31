# cython: language_level=3
"""
Add two floats (basic Cython types).

User notes
----------
- Demonstrates `cpdef` for a Python-callable + C-callable function.
- Demonstrates `double` as a C floating point type.

Developer notes
--------------
- Keep signatures explicit for stable docs and predictable behavior.
"""

cpdef double add(double a, double b):
    """Return a + b as a C double."""
    return a + b
