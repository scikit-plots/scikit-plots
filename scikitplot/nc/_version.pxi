# _version.pxi
# Purpose:
#   This is like a C/C++ #include file. Use it to share reusable code snippets,
#   macros, or inline Cython/C++ helper functions across multiple modules.
#   Unlike .pxd, this file is textually included using 'include' keyword.
#
# When to use:
#   Only when you have common code to reuse in multiple .pyx or .pxd files.
#   This is optional. (use: include "_version.pxi")

# Import the necessary C++ standard library components
# from libcpp.string cimport string

# Example (optional):
# cdef inline double square(double x):
#     return x * x
