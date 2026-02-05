# scikitplot/cexternals/_annoy/_mman/__init__.py

# Use cpdef for public API, cdef for internal helpers
# Always manage memory explicitly (__cinit__/__dealloc__)
# Use nogil when possible for performance
# Avoid .pxi files in modern code
# #
# mman.h           # Pure C++ (logic, constants, inline funcs)
# mman.pxd         # (like C headers) Cython C++ declarations ONLY (cppclass) (no Python-facing logic)
# mman.pxi         # OPTIONAL thin helpers (rare, usually empty) Share Cython code between multiple .pyx for beginners: avoid .pxi
# mman.pyx         # (like C source files) Cython Python implementation code wrapper class ONCE (Python-facing cdef class wrappers logic)
# mman.pyi         # (for Python tooling) Python type hints (typing only for users, IDEs)
#
# C++ (mman.h)
#         ↓
# Cython declarations (mman.pxd)
#         ↓
# Python wrapper (mman.pyx OR annoy_wrapper.pyx)
#
# cdef cppclass Kiss32Random:     # in .pxd
# cdef class Kiss32Random:        # in .pyx
# Never both with the same name (.h -> .pxd -> .pyx)
# Either as a cppclass (C++ side)
# Or as a cdef class (Python wrapper)
"""
Cross-platform memory mapping module.

This module provides a unified Python interface for memory mapping
operations that work on both Windows and Unix/Linux/macOS systems.

Platform Support
----------------
- **Windows**: Uses custom implementation wrapping Windows API
  (CreateFileMapping, MapViewOfFile)
- **Unix/Linux/macOS**: Uses standard POSIX mmap functions

Architecture
------------
The module consists of several interconnected files:

**mman.h** (C/C++ header)
    Pure C/C++ code containing:
    - Platform detection (#if Windows vs POSIX)
    - Windows: Custom implementations of mmap, munmap, etc.
    - Unix/Linux: Includes standard <sys/mman.h>
    - Provides unified C API regardless of platform

**mman.pxd** (Cython declarations)
    C-level type declarations (like C headers for Cython):
    - Declares C functions from mman.h
    - Defines constants (PROT_*, MAP_*, MS_*)
    - Provides inline helper functions
    - NO Python-facing logic
    - NO implementation code (declarations only)
    - Used by other .pyx files via cimport

**mman.pyx** (Cython implementation)
    Python wrapper implementation:
    - Implements MemoryMap class (Python-facing API)
    - Wraps C functions from mman.pxd
    - Handles Python/C type conversions
    - Provides error handling and validation
    - Context manager support
    - Includes full NumPyDoc documentation

**mman.pyi** (Python type stubs)
    Type hints for static analysis:
    - Signatures for IDEs and type checkers
    - NO implementation
    - Used by mypy, pyright, etc.

**__init__.py** (this file)
    Package initialization:
    - Documentation
    - Module-level exports (if any)

Design Principles
-----------------
1. **Platform Abstraction**: Single API works everywhere
2. **No .pxi Files**: Modern Cython avoids .pxi includes
3. **Separation of Concerns**:
   - .h = C logic and platform-specific code
   - .pxd = C declarations for Cython
   - .pyx = Python wrapper and implementation
   - .pyi = Type hints for Python tooling

Best Practices
--------------
1. **Use cpdef** for public API (both C and Python access)
2. **Use cdef** for internal helpers (C-only, faster)
3. **Manage memory explicitly**: __cinit__/__dealloc__
4. **Use nogil** when possible for thread safety
5. **Validate inputs explicitly** before calling C code
6. **Handle errors properly** (check return values, errno)

Workflow
--------
When compiling Cython code:

1. **C++ compilation** (mman.h):
   - Platform detected at compile time
   - Correct headers included
   - Functions defined or imported

2. **Cython→C++ translation** (mman.pyx → mman.cpp):
   - Reads declarations from mman.pxd
   - Generates C++ code wrapping mman.h functions
   - Adds Python API glue code

3. **Final linking**:
   - Produces .so (Linux) or .pyd (Windows) extension module
   - Callable from Python as normal module

Common Patterns
---------------

**Declaring C++ Classes**::

    # In .pxd
    cdef cppclass MyCppClass:
        void method()

    # In .pyx
    cdef class PyMyCppClass:
        cdef MyCppClass* _ptr  # Wrap C++ object

**Wrapping C Functions**::

    # In .pxd
    cdef extern from "header.h":
        int c_function(int arg)

    # In .pyx
    def py_function(int arg):
        return c_function(arg)

**Context Managers**::

    # In .pyx
    cdef class Resource:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.cleanup()

Never Both
----------
NEVER have the same name in .pxd and .pyx as different things::

    # WRONG:
    # mman.pxd
    cdef cppclass MemoryMap:  # C++ class
        pass

    # mman.pyx
    cdef class MemoryMap:  # Python wrapper
        pass

This causes name conflicts. Instead::

    # CORRECT:
    # mman.h (C++)
    class MemoryMap { ... };

    # mman.pxd
    cdef cppclass CMemoryMap "MemoryMap":  # Rename for Cython
        pass

    # mman.pyx
    cdef class PyMemoryMap:  # Python wrapper with distinct name
        cdef CMemoryMap* _map

Notes for Maintainers
---------------------
- **Keep files in sync**: Changes to .h require updates to .pxd
- **Test on all platforms**: Windows and Unix/Linux behavior differs
- **Document everything**: Use NumPyDoc style in .pyx
- **Validate inputs**: Check parameters before calling C code
- **Handle edge cases**: Invalid fds, zero length, null pointers

References
----------
- Cython documentation: https://cython.readthedocs.io/
- POSIX mmap spec: https://pubs.opengroup.org/onlinepubs/9699919799/
- Windows Memory Mapping: https://docs.microsoft.com/en-us/windows/win32/memory/file-mapping
- mman-win32 project: https://code.google.com/archive/p/mman-win32/

Examples
--------
Creating anonymous memory mapping::

    from scikitplot.cexternals._annoy._mman.mman import (
        MemoryMap, PROT_READ, PROT_WRITE
    )

    # Create 4KB anonymous mapping
    with MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE) as m:
        m.write(b"Hello, World!")
        data = m.read(13)
        print(data)  # b'Hello, World!'

Mapping a file::

    with open("data.bin", "r+b") as f:
        with MemoryMap.create_file_mapping(
            f.fileno(), 0, 4096, PROT_READ
        ) as m:
            data = m.read(100)
"""

# Module version
__version__ = "1.0.1"

# Export nothing at package level
# Users should import from .mman module directly
__all__ = []
