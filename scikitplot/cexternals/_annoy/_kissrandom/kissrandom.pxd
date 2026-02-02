# cython: language_level=3
# distutils: language = c++

# scikitplot/cexternals/_annoy/_kissrandom/kissrandom.pxd

"""
Cython declarations file for kissrandom module.

This file contains C/C++ level type declarations and external definitions.
It serves as the interface between Cython code and the C++ header file.

Purpose
-------
- Declare C++ classes, structs, and functions from kissrandom.h
- Define cimport-able types for use in other .pyx files
- Expose C-level constants and inline functions
- NO Python-facing logic (that goes in .pyx)
- NO implementation code (declarations only)

Design Principles
-----------------
- Keep declarations minimal and exact (match C++ header precisely)
- Use 'cdef extern from' to import from C++ header
- Declare all public C++ members that Cython needs to access
- Use 'nogil' where appropriate for thread safety
- Follow Cython naming conventions (cdef for C-level, def/cpdef for Python)

Modern Cython Best Practices
-----------------------------
- Use DEF for compile-time constants (not IF/ELSE macros)
- Prefer 'cpdef' over 'cdef' when you want both C and Python access
- Use 'nogil' context managers for performance-critical sections
- Annotate with 'const' where C++ uses const
- Use specific integer types (uint32_t, uint64_t) not generic 'int'

Notes for Beginners
-------------------
- .pxd files are like C/C++ header files (.h) in Cython world
- They declare WHAT exists, not HOW it's implemented
- Other .pyx files can 'cimport' from this .pxd to use these types
- Changes here must match the C++ header exactly or compilation fails

Notes for Maintainers
---------------------
- Keep this file in sync with kissrandom.h
- Test compilation after any changes
- If C++ header changes, update declarations here
- Document any deviations from C++ header (e.g., name changes)

References
----------
- Cython docs on .pxd: https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html
- C++ interop: https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
- Type declarations: https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html
"""

from libc.stdint cimport uint32_t, uint64_t
from libc.stddef cimport size_t

# ===========================================================================
# Compile-time constants using DEF (preferred over IF in modern Cython)
# ===========================================================================
# DEF creates true compile-time constants that are substituted during Cython
# compilation, not runtime. Use DEF instead of IF/ELSE for feature flags.

DEF CYTHON_LANGUAGE_LEVEL = 3
DEF USE_CPLUSPLUS = True

# ===========================================================================
# C++ declarations from kissrandom.h
# ===========================================================================

# Import C++ classes from the header file.
# The 'extern from' directive tells Cython where to find the C++ definitions.
# The 'namespace' specifies the C++ namespace containing these classes.

cdef extern from "../src/kissrandom.h" namespace "Annoy" nogil:

    # const uint32_t Kiss32Random_default_seed "Annoy::Kiss32Random::default_seed"
    # const uint64_t Kiss64Random_default_seed "Annoy::Kiss64Random::default_seed"

    # -----------------------------------------------------------------------
    # Kiss32Random - 32-bit KISS RNG
    # -----------------------------------------------------------------------
    cdef cppclass Kiss32Random:

        # 🚫 must declare the function, not the constant.❌
        # Static constants
        # const uint32_t default_seed "Kiss32Random::default_seed"
        # uint32_t default_x "Kiss32Random::default_x"
        # const uint32_t default_y "Kiss32Random::default_y"
        # const uint32_t default_z "Kiss32Random::default_z"
        # const uint32_t default_c "Kiss32Random::default_c"
        # Public member variables (C++ struct fields)
        # uint32_t x
        # uint32_t y
        # uint32_t z
        # uint32_t c

        # Constructors
        # Note: Cython syntax for C++ constructors
        Kiss32Random() except +
        Kiss32Random(uint32_t seed) except +

        # Static methods
        @staticmethod
        uint32_t get_default_seed() nogil

        @staticmethod
        uint32_t get_seed() nogil

        @staticmethod
        uint32_t normalize_seed(uint32_t seed) nogil

        # Instance methods
        void reset(uint32_t seed) nogil
        void reset_default() nogil
        void set_seed(uint32_t seed) nogil
        uint32_t kiss() nogil
        int flip() nogil
        size_t index(size_t n) nogil

    # -----------------------------------------------------------------------
    # Kiss64Random - 64-bit KISS RNG
    # -----------------------------------------------------------------------
    cdef cppclass Kiss64Random:

        # 🚫 must declare the function, not the constant.❌
        # Static constants
        # const uint64_t default_seed "Kiss64Random::default_seed"
        # uint64_t default_x "Kiss32Random::default_x"
        # const uint64_t default_y "Kiss64Random::default_y"
        # const uint64_t default_z "Kiss64Random::default_z"
        # const uint64_t default_c "Kiss64Random::default_c"
        # Public member variables
        # uint64_t x
        # uint64_t y
        # uint64_t z
        # uint64_t c

        # Constructors
        Kiss64Random() except +
        Kiss64Random(uint64_t seed) except +

        # Static methods
        @staticmethod
        uint64_t get_default_seed() nogil

        @staticmethod
        uint64_t get_seed() nogil

        @staticmethod
        uint64_t normalize_seed(uint64_t seed) nogil

        # Instance methods
        void reset(uint64_t seed) nogil
        void reset_default() nogil
        void set_seed(uint64_t seed) nogil
        uint64_t kiss() nogil
        int flip() nogil
        size_t index(size_t n) nogil


# ===========================================================================
# Cython-level type aliases and helpers (optional but useful)
# ===========================================================================

# These are Cython-specific definitions that don't exist in C++.
# They can be used by other .pyx files that cimport this .pxd.

# Type alias for seed types (makes code more readable)
# ctypedef uint32_t seed32_t
# ctypedef uint64_t seed64_t

# Expose constants explicitly
# DEF Kiss32Random_default_seed = 123456789
# DEF Kiss64Random_default_seed = 1234567890987654321

# You can add additional helper functions here if needed for internal use.
# Example (commented out, add if needed):
# cdef inline uint32_t _safe_cast_to_uint32(object py_value) except? 0


# ===========================================================================
# Module-level constants accessible via cimport
# ===========================================================================

# These can be cimported by other Cython modules.
# Use DEF for compile-time constants, cdef for runtime constants.

# Compile-time version info
# DEF VERSION_MAJOR = 1
# DEF VERSION_MINOR = 0
# DEF VERSION_PATCH = 0

# Runtime accessible module info (optional, can be used in .pyx)
# cdef object __version__  # Defined in .pyx, declared here for cimport access
