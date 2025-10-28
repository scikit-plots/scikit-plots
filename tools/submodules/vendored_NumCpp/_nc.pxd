# _nc.pxd
# Purpose:
#   This file is similar to a C/C++ header (.h, .hpp). It declares C/C++ functions,
#   classes, and variables to be made available to Cython (.pyx) implementation files.
#   No implementation is written here—only declarations.
#
# When to use:
#   Use .pxd files when you want to "cimport" (not "import") C/C++ functionality
#   into a Cython .pyx file for high performance.

# Import reusable code from the .pxi file, If Needed avoid duplicates
# include "_nc.pxi"

# Import Python-level NumPy
# import numpy as np
# Import C-level NumPy typing for Cython
# Initialize NumPy C-API (required once)
# cimport numpy as cnp
# cnp.import_array()

# Import the necessary C++ standard library components
# from cython cimport floating, integral
# from libcpp cimport bool
# from libcpp.string cimport string

# ctypedef fused numeric_t:
#     floating
#     integral

# ctypedef fused lapack_t:
#     cnp.float32_t
#     cnp.float64_t
#     cnp.complex64_t
#     cnp.complex128_t


# Declare external C++ symbols from NumCpp
cdef extern from "NumCpp.hpp" namespace "nc":
    const char* VERSION   # Expose nc::VERSION constant to Cython
