# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False, wraparound=False

"""
:py:mod:`~._version` is a high-performance Python module that wraps the C++ NumCpp library.

Providing fast numerical functions with seamless NumPy array support across common numeric types.

.. seealso::
   * https://github.com/dpilger26/NumCpp
   * https://scikit-plots.github.io/dev/user_guide/cexternals/_numcpp/index.html
"""

# _version.pyx
# Purpose:
#   This is the main implementation file where you write Cython code that
#   interfaces with C++ (via .pxd) and Python. It compiles to a Python extension (.so/.pyd).
#
# Steps:
#   1. cimport declarations from .pxd (C-level access)
#   2. import Python modules normally
#   3. write optimized Cython functions or wrap C++ functions

# _version.pyx - Cython bindings for NumCpp NdArray
# This module provides Python access to selected functionality of the
# NumCpp C++ library (https://github.com/dpilger26/NumCpp), particularly
# the NdArray class and array operations such as `dot`.
#
# ======================================================================
# IMPORTANT NOTES:
#
# 1. C++ Templates Limitation:
#    - The original NumCpp library uses C++ templates (NdArray<T>) extensively.
#    - Cython cannot parse or instantiate C++ templates directly.
#    - Keywords like `explicit`, `std::initializer_list`, and rvalue references
#      (&&) are not supported in Cython .pxd files.
#
# 2. Type Specialization Requirement:
#    - Only specific type instantiations of NdArray can be exposed to Python.
#    - For example, we define NdArray_double for double, NdArray_complex_double
#      for std::complex<double>, etc.
#    - If you need other types (float, int, complex<float>), you must add
#      separate specializations in both the .pxd and .pyx files.
#
# 3. NumPy Conversion:
#    - Python functions use numpy.ndarray as input and output.
#    - Conversion helpers are implemented to map between NdArray<T> and
#      numpy.ndarray safely and efficiently.
#    - Users must pass arrays with the correct dtype (e.g., float64 for NdArray_double).
#
# 4. Wrapping Strategy:
#    - Each specialized C++ type is declared in the .pxd file.
#    - Python-callable functions in .pyx wrap these types and handle NumPy conversion.
#    - Example function: `dot_double(a: np.ndarray, b: np.ndarray)`.
#
# 5. Alternative Approaches:
#    - Pybind11 or Boost.Python fully support templates and require no specialization.
#    - For a full template coverage or minimal manual work, Pybind11 is recommended.
# ======================================================================

# Include reusable macros/helpers (optional)
# include "_version.pxi"

# Import Python-level NumPy
# import numpy as np
# Import C-level NumPy typing for Cython
# Initialize NumPy C-API (required once)
cimport numpy as cnp
cnp.import_array()

# Import C++ declarations from _version.pxd
from ._version cimport VERSION
# Exported Python variables
__version__ = VERSION.decode('utf-8')  # Convert const char* to Python str (safe)
