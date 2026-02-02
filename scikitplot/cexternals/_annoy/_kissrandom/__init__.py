# scikitplot/cexternals/_annoy/_kissrandom/__init__.py

# Use cpdef for public API, cdef for internal helpers
# Always manage memory explicitly (__cinit__/__dealloc__)
# Use nogil when possible for performance
# Avoid .pxi files in modern code
# #
# kissrandom.h           # Pure C++ (logic, constants, inline funcs)
# kissrandom.pxd         # (like C headers) Cython C++ declarations ONLY (cppclass) (no Python-facing logic)
# kissrandom.pxi         # OPTIONAL thin helpers (rare, usually empty) Share Cython code between multiple .pyx for beginners: avoid .pxi
# kissrandom.pyx         # (like C source files) Cython Python implementation code wrapper class ONCE (Python-facing cdef class wrappers logic)
# kissrandom.pyi         # (for Python tooling) Python type hints (typing only for users, IDEs)
#
# C++ (kissrandom.h)
#         ↓
# Cython declarations (kissrandom.pxd)
#         ↓
# Python wrapper (kissrandom.pyx OR annoy_wrapper.pyx)
#
# cdef cppclass Kiss32Random:     # in .pxd
# cdef class PyKiss32Random:      # in .pyx
# Never both with the same name (.h -> .pxd -> .pyx)
# Either as a cppclass (C++ side)
# Or as a cdef class (Python wrapper)
