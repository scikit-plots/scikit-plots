# cython: language_level=3

# scikitplot/random/_kiss/kiss_random.pxi

"""
Shared Cython implementation code (include file).

This file contains Cython code that can be shared between multiple .pyx files
via the 'include' directive. Think of .pxi as similar to .h files in C/C++,
but for Cython implementation code rather than just declarations.

WARNING FOR BEGINNERS
---------------------
.pxi files are generally DISCOURAGED in modern Cython development because:
- They use textual inclusion (like C #include), which can cause duplicate symbols
- They don't provide namespace isolation
- They make build dependencies harder to track
- They can lead to surprising compilation errors

RECOMMENDED ALTERNATIVE
-----------------------
Instead of .pxi files, prefer:
1. Put shared C-level code in .pxd files (declarations)
2. Put shared Python-level code in regular .py files
3. Use 'cimport' to import from .pxd files (better than 'include')
4. Create separate .pyx modules for shared functionality

WHY THIS FILE EXISTS
--------------------
This .pxi file is provided purely for educational purposes to demonstrate
the mechanism. In production code, you should avoid .pxi and use the
alternatives above.

Usage Example (NOT RECOMMENDED)
-------------------------------
In a .pyx file:
    include "kiss_random.pxi"  # Textually includes this file

Better Approach:
    from kiss_random cimport Kiss32Random  # Import from .pxd
    from kiss_random import Kiss32Random   # Import Python class from .pyx

Design Principles (if you must use .pxi)
----------------------------------------
- Keep .pxi files minimal and focused
- Only include pure implementation helpers
- Avoid defining classes or complex types
- Prefer inline functions for performance
- Document what's intended to be included where

Notes for Maintainers
---------------------
- Consider deprecating this file in favor of proper modularization
- If kept, ensure it's only included once per compilation unit
- Test that including this file doesn't cause symbol conflicts
- Keep content minimal and well-documented

References
----------
- Cython FAQ on .pxi: https://github.com/cython/cython/wiki/FAQ
- Best practices: https://github.com/cython/cython/issues/4310
"""

from libc.stdint cimport uint32_t, uint64_t
from libc.stddef cimport size_t

# ===========================================================================
# Shared helper functions (inline for performance)
# ===========================================================================

cdef inline bint is_power_of_two(size_t n) nogil:
    """
    Check if n is a power of 2.

    Parameters
    ----------
    n : size_t
        Value to check

    Returns
    -------
    bint
        True if n is a power of 2, False otherwise

    Notes
    -----
    - Uses bit manipulation trick: (n & (n-1)) == 0
    - Returns False for n==0
    - Useful for optimizing modulo operations

    Developer Notes
    ---------------
    This could be used to optimize index() method when n is a power of 2,
    replacing modulo with bitwise AND. However, current implementation
    keeps it simple with modulo for all cases.
    """
    return n != 0 and (n & (n - 1)) == 0


cdef inline uint32_t fast_modulo_pow2_32(uint32_t value, uint32_t n) nogil:
    """
    Fast modulo for power-of-2 divisors (32-bit).

    Parameters
    ----------
    value : uint32_t
        Value to reduce
    n : uint32_t
        Modulo divisor (MUST be a power of 2)

    Returns
    -------
    uint32_t
        value % n

    Notes
    -----
    - ONLY valid when n is a power of 2
    - Caller must verify n is power of 2 before calling
    - Uses bitwise AND instead of modulo for speed

    Developer Notes
    ---------------
    For n = 2^k, value % n == value & (n-1)
    Example: value % 8 == value & 7
    """
    return value & (n - 1)


cdef inline uint64_t fast_modulo_pow2_64(uint64_t value, uint64_t n) nogil:
    """
    Fast modulo for power-of-2 divisors (64-bit).

    See fast_modulo_pow2_32 for details.
    """
    return value & (n - 1)


# ===========================================================================
# Shared validation helpers
# ===========================================================================

cdef inline void validate_index_range(size_t n) except *:
    """
    Validate that n is a valid range for index generation.

    Parameters
    ----------
    n : size_t
        Range upper bound

    Raises
    ------
    ValueError
        If n would cause undefined behavior (currently never, as n==0 is handled)

    Notes
    -----
    - Current implementation allows n==0 (returns 0)
    - Could be extended to enforce n > 0 if needed
    - The 'except *' allows Python exceptions to propagate

    Developer Notes
    ---------------
    This is a placeholder for future validation logic.
    Consider adding overflow checks for very large n values.
    """
    # Current implementation: n==0 is explicitly handled, no validation needed
    pass


# ===========================================================================
# Shared constants and configuration
# ===========================================================================

# Maximum recommended data points for Kiss32Random
DEF MAX_RECOMMENDED_32BIT = 16777216  # 2^24

# Maximum recommended data points for Kiss64Random
DEF MAX_RECOMMENDED_64BIT = 1152921504606846976  # 2^60 (conservative estimate)

# Module metadata (can be shared across multiple .pyx files)
# DEF MODULE_NAME = "kiss_random"
# DEF MODULE_VERSION = "1.0.0"
# DEF MODULE_AUTHOR = "Contributors"


# ===========================================================================
# Shared documentation snippets (as compile-time strings)
# ===========================================================================

# These can be embedded in docstrings via string formatting if needed.

DEF SECURITY_WARNING = """
WARNING: This RNG is NOT cryptographically secure.
Do NOT use for:
- Password generation
- Security tokens
- Cryptographic keys
- Any security-sensitive randomness

For cryptographic randomness, use:
- secrets module (Python 3.6+)
- os.urandom()
- cryptography.hazmat.primitives.ciphers
"""

DEF PERFORMANCE_NOTE = """
Performance characteristics:
- Very fast generation (~1-2 CPU cycles per value)
- Small memory footprint (16-32 bytes of state)
- Good statistical properties for Monte Carlo simulations
- Suitable for parallel RNG with different seeds
"""


# ===========================================================================
# REMINDER: Avoid .pxi in production code
# ===========================================================================

# This file demonstrates .pxi functionality for learning purposes.
# In real projects, use one of these alternatives:
#
# 1. Put these functions in a separate .pyx module:
#    - Create kiss_random_helpers.pyx
#    - Cimport and use: from kiss_random_helpers cimport is_power_of_two
#
# 2. Put declarations in .pxd and implementations in .pyx:
#    - Declare in kiss_random.pxd: cdef inline bint is_power_of_two(size_t) nogil
#    - Implement in kiss_random.pyx
#    - Other modules cimport from kiss_random.pxd
#
# 3. For pure Python helpers, use regular .py files:
#    - Create kiss_random_utils.py
#    - Import normally: from kiss_random_utils import helper_func
#
# See: https://github.com/cython/cython/issues/4310
