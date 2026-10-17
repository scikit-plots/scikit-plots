# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# cython: binding=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

# scikitplot/memmap/_memmap/mem_map.pyx

"""
Windows Memory Mapping - Cython Implementation.

This module provides Python wrappers for Windows memory mapping functions,
offering a POSIX mmap-like interface on Windows platforms.

The implementation wraps the C functions from mman.h, which internally use
Windows CreateFileMapping, MapViewOfFile, and related APIs.

Classes
-------
MemoryMap
    Python wrapper for memory-mapped region with context manager support

Functions
---------
mmap_region
    Create a new memory-mapped region
munmap_region
    Unmap a memory-mapped region
mprotect_region
    Change protection of a memory-mapped region
msync_region
    Synchronize memory-mapped region with storage

Protection Flags
----------------
PROT_NONE : int
    No access
PROT_READ : int
    Read access
PROT_WRITE : int
    Write access
PROT_EXEC : int
    Execute access

Mapping Flags
-------------
MAP_SHARED : int
    Share mapping with other processes
MAP_PRIVATE : int
    Create private copy-on-write mapping
MAP_ANONYMOUS : int
    Map anonymous memory (not backed by file)
MAP_ANON : int
    Alias for MAP_ANONYMOUS

Sync Flags
----------
MS_ASYNC : int
    Asynchronous sync
MS_SYNC : int
    Synchronous sync
MS_INVALIDATE : int
    Invalidate cached pages

Design Principles
-----------------
- Thin Cython wrapper around C implementation in mman.h
- Expose Pythonic API while maintaining C performance
- Provide both C-level (cdef) and Python-level (def/cpdef) interfaces
- Handle Python/C type conversions safely and explicitly
- Validate inputs and provide clear error messages
- Support context managers for automatic cleanup
- Follow modern Cython best practices

Modern Cython Best Practices Applied
------------------------------------
1. Use DEF for compile-time constants (not IF/ELSE macros)
2. Use cpdef for methods that need both C and Python access
3. Add type annotations for all function signatures
4. Use nogil contexts where thread safety allows
5. Leverage embedsignature=True for automatic docstring signatures
6. Enable binding=True for better Python introspection
7. Use specific integer types (size_t, off_t) not generic int
8. Explicit memory management with __cinit__/__dealloc__
9. Properties with proper getter/setter implementations

Performance Optimizations
--------------------------
- boundscheck=False: Disable array bounds checking
- wraparound=False: Disable negative indexing
- cdivision=True: Use C division (faster, no zero-check)
- nogil: Release GIL where possible for thread safety

Security and Safety Notes
--------------------------
- Memory mapping requires careful resource management
- Always unmap regions when done (use context managers)
- Validate file descriptors and sizes
- Check return values for errors

Notes for Users
---------------
- Windows-only module (use standard mmap on Unix/Linux)
- Memory-mapped files can improve I/O performance
- Be careful with shared mappings (can affect other processes)
- Lock pages (mlock) may require elevated privileges

Notes for Developers
--------------------
- Keep this file in sync with mman.h and mem_map.pxd
- Add comprehensive docstrings following NumPy style
- Test all public functions with pytest
- Validate all input parameters explicitly
- Handle edge cases (invalid fds, zero length, etc.)

Build Requirements
------------------
- Cython >= 0.29.0 (3.0+ recommended)
- C++11 compatible compiler (MSVC on Windows)
- Python >= 3.7
- Windows OS (uses Windows.h internally)

Platform Notes
--------------
- Windows only - uses CreateFileMapping/MapViewOfFile
- On Unix/Linux, use Python's standard mmap module
- File descriptors on Windows are different from Unix

References
----------
.. [1] Windows Memory Mapping:
       https://docs.microsoft.com/en-us/windows/win32/memory/file-mapping
.. [2] POSIX mmap: https://man7.org/linux/man-pages/man2/mmap.2.html
.. [3] mman-win32: https://code.google.com/p/mman-win32/

Examples
--------
Create anonymous memory mapping:

>>> import mem_map
>>> mapping = MemoryMap.create_anonymous(4096, mem_map.PROT_READ | mem_map.PROT_WRITE)
>>> # Use mapping...
>>> mapping.close()

Using context manager (recommended):

>>> with MemoryMap.create_anonymous(4096, mem_map.PROT_READ | mem_map.PROT_WRITE) as mapping:
...     # Mapping automatically closed on exit
...     mapping.write(b"Hello, World!")
...     data = mapping.read(13)

Map a file into memory:

>>> with open("data.bin", "r+b") as f:
...     fd = f.fileno()
...     with MemoryMap.create_file_mapping(fd, 0, 4096, mem_map.PROT_READ) as mapping:
...         data = mapping.read(100)
"""

# ===========================================================================
# Imports
# ===========================================================================

from cpython.buffer cimport PyBUF_FORMAT, PyBUF_ND, PyBUF_STRIDES, PyBUF_WRITABLE
from cpython.bytearray cimport PyByteArray_AS_STRING
from cpython.bytes cimport PyBytes_AsStringAndSize

# Cython-level imports (C types and functions)
# from libc.stdint cimport int64_t
from libc.stdint cimport uintptr_t
from libc.stddef cimport size_t
from libc.errno cimport errno
from libc.string cimport memcpy  # , memset

# Python-level imports
# import os
# from sys import platform
# from typing import Final

# MAP_FAILED: Final[int] = ((void *)-1)  # == -1 means “mmap failed”
# Py_REFCNT and _Py_REFCNT are the same, except _Py_REFCNT takes
# a raw pointer and Py_REFCNT takes a normal Python object
# from cpython.ref cimport PyObject, _Py_REFCNT, Py_REFCNT

# Import C functions from our .pxd declarations
# Note: The "as mm" creates a namespace alias for cleaner code
# from scikitplot.memmap._memmap cimport mem_map as c_mmap
from scikitplot.memmap._memmap.mem_map cimport (
    mmap, munmap, mprotect, msync, mlock, munlock,
    off_t, is_map_failed, validate_prot_flags, validate_map_flags,
    get_page_size,
)

# ============================================================================
# Platform-Specific System APIs (Correct Pattern)
# ============================================================================

# Import platform-specific types
# IF UNAME_SYSNAME == "Windows":
# cdef extern from *:
# cdef extern from * nogil:
cdef extern from "../../cexternals/_annoy/src/mman.h" nogil:
    """
    #include <stddef.h>
    size_t cy_get_page_size(void);

    #ifdef _WIN32
        #include <windows.h>

        size_t cy_get_page_size(void) {
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            return (size_t)si.dwPageSize;
        }
    #else
        #include <unistd.h>

        size_t cy_get_page_size(void) {
            long tmp = sysconf(_SC_PAGESIZE);
            if (tmp <= 0)
                return 4096;
            return (size_t)tmp;
        }
    #endif
    """
    size_t cy_get_page_size() nogil


# The lock of one mapping: a PyMutex on a free-threaded CPython, nothing with
# the GIL (see the header). Sections under it are plain C only.
cdef extern from "mem_map_lock.h" nogil:
    ctypedef struct skplt_mm_lock:
        pass
    void skplt_mm_acquire(skplt_mm_lock* lock)
    void skplt_mm_release(skplt_mm_lock* lock)


# What MemoryMap._acquire found.
cdef enum:
    _ACQUIRED = 0
    _CLOSED = 1
    _NOT_READABLE = 2
    _NOT_WRITABLE = 3

# ===========================================================================
# Module metadata
# ===========================================================================

__all__ = [
    # flags
    "PY_PROT_NONE",
    "PY_PROT_READ",
    "PY_PROT_WRITE",
    "PY_PROT_EXEC",

    "PY_MAP_FILE",
    "PY_MAP_SHARED",
    "PY_MAP_PRIVATE",
    "PY_MAP_TYPE",
    "PY_MAP_FIXED",
    "PY_MAP_ANONYMOUS",
    "PY_MAP_ANON",

    "PY_MS_ASYNC",
    "PY_MS_SYNC",
    "PY_MS_INVALIDATE",

    "PY_FILE_MAP_EXECUTE",

    # Exceptions
    "MMapError",
    "MMapAllocationError",
    "MMapInvalidParameterError",

    # memmap
    "MemoryMap",
    "mmap_region",
    "py_validate_prot_flags",  # no-cython-lint
    "py_validate_map_flags",  # no-cython-lint
    "py_get_page_size",  # no-cython-lint
    "py_is_page_aligned",  # no-cython-lint
]

# ===========================================================================
# Compile-time constants using DEF
# ===========================================================================
# DEF VERSION_MAJOR = 1
# DEF VERSION_MINOR = 0
# DEF VERSION_PATCH = 1


# ===========================================================================
# Memory protection flags (POSIX-compatible)
# ===========================================================================

# These constants work on all platforms (Windows emulates them)
DEF PROT_NONE = 0
DEF PROT_READ = 1
DEF PROT_WRITE = 2
DEF PROT_EXEC = 4

# ===========================================================================
# Mapping flags (POSIX-compatible)
# ===========================================================================

DEF MAP_FILE = 0
DEF MAP_SHARED = 1
DEF MAP_PRIVATE = 2
DEF MAP_TYPE = 0xf
DEF MAP_FIXED = 0x10
DEF MAP_ANONYMOUS = 0x20
DEF MAP_ANON = 0x20  # Alias for MAP_ANONYMOUS

# ===========================================================================
# Sync flags (POSIX-compatible)
# ===========================================================================

DEF MS_ASYNC = 1
DEF MS_SYNC = 2
DEF MS_INVALIDATE = 4

# Windows compatibility
DEF FILE_MAP_EXECUTE = 0x0020
# MAP_FAILED: Final[int] = ((void *)-1)

PY_PROT_NONE = 0
PY_PROT_READ = 1
PY_PROT_WRITE = 2
PY_PROT_EXEC = 4

PY_MAP_FILE = 0
PY_MAP_SHARED = 1
PY_MAP_PRIVATE = 2
PY_MAP_TYPE = 0xf
PY_MAP_FIXED = 0x10
PY_MAP_ANONYMOUS = 0x20
PY_MAP_ANON = 0x20  # Alias for MAP_ANONYMOUS

PY_MS_ASYNC = 1
PY_MS_SYNC = 2
PY_MS_INVALIDATE = 4

# Windows native
PY_FILE_MAP_EXECUTE = 0x0020

# ===========================================================================
# Utility Functions Implementation (Thread-Safe, No Global State)
# ===========================================================================
# cdef size_t get_page_size() noexcept nogil:
#     return cy_get_page_size()


cdef size_t get_page_size() noexcept nogil:  # no-cython-lint
    """
    Get system page size.

    Returns
    -------
    size_t
        System page size in bytes. Falls back to 4096 on error.

    Notes
    -----
    Does not cache the result to avoid thread-safety issues under nogil.
    The sysconf/GetSystemInfo call overhead is negligible compared to
    actual mmap operations (~100 cycles vs 10,000+ cycles).

    On Windows, uses GetSystemInfo().dwPageSize.
    On POSIX systems, uses sysconf(_SC_PAGESIZE).

    Falls back to 4096 bytes (common page size) if system call fails.

    Developer Notes
    ---------------
    Previous implementations used a cached global variable (_PAGE_SIZE),
    which created a data race under nogil during first access. This
    version eliminates shared mutable state for correctness.

    Performance Impact
    ------------------
    - sysconf: ~100-200 CPU cycles (VDSO cached on Linux)
    - GetSystemInfo: ~50-100 CPU cycles
    - mmap syscall: ~10,000+ CPU cycles
    - Overhead: <2% of typical mmap operation

    Correctness > micro-optimization in systems code.

    Examples
    --------
    >>> page_size = get_page_size()
    >>> assert page_size > 0
    >>> assert page_size in [4096, 8192, 16384, 65536]  # Common sizes
    """
    return cy_get_page_size()


def py_get_page_size():
    """
    Get system page size in bytes.

    Returns
    -------
    int
        System page size.
    """
    cdef size_t ps

    with nogil:
        ps = get_page_size()

    return <int>ps


cdef bint is_page_aligned(size_t value) noexcept nogil:
    """
    Check if value is aligned to system page boundary.

    Parameters
    ----------
    value : size_t
        Value to check for alignment.

    Returns
    -------
    bool
        True if value is a multiple of page size, False otherwise.

    Notes
    -----
    Uses bitwise AND with (page_size - 1) for efficiency.
    Assumes page size is always a power of 2 (true on all major platforms).

    This check is equivalent to: (value % page_size) == 0
    But uses faster bitwise operation.

    Examples
    --------
    >>> page_size = get_page_size()
    >>> assert is_page_aligned(0)
    >>> assert is_page_aligned(page_size)
    >>> assert is_page_aligned(page_size * 2)
    >>> assert not is_page_aligned(page_size + 1)
    """
    cdef size_t page_size = get_page_size()

    # Optional safety check (can remove if guaranteed)
    if (page_size & (page_size - 1)) != 0:
        return (value % page_size) == 0

    return (value & (page_size - 1)) == 0


def py_is_page_aligned(int value):
    """
    Check whether value is page aligned.
    """
    if value < 0:
        raise ValueError("value must be non-negative")

    cdef bint result
    with nogil:
        result = is_page_aligned(<size_t>value)

    return bool(result)


cdef size_t align_to_page(size_t value) noexcept nogil:
    """
    Round value to page boundary.

    Parameters
    ----------
    value : size_t
        Value to align.

    Returns
    -------
    size_t
        Value rounded to nearest page boundary (multiple of page size).

    Notes
    -----
    Rounds DOWN to the nearest page boundary, not up.
    Uses efficient integer division: (value / page_size) * page_size

    Includes overflow protection for extremely large values.

    The formula ensures:
    - align_to_page(0) == 0
    - align_to_page(100) == 0 (assuming 4K pages)
    - align_to_page(4096) == 4096
    - align_to_page(4097) == 4096
    - align_to_page(8192) == 8192

    Developer Notes
    ---------------
    Division by power-of-2 is optimized to right shift by compiler.
    Overflow protection: if value is near SIZE_MAX, returns maximum
    aligned value instead of wrapping around.

    Examples
    --------
    >>> page_size = get_page_size()
    >>> assert align_to_page(0) == 0
    >>> assert align_to_page(100) == 0
    >>> assert align_to_page(page_size) == page_size
    >>> assert align_to_page(page_size + 1) == page_size
    >>> assert align_to_page(page_size * 2) == page_size * 2
    """
    cdef size_t page_size = get_page_size()
    return (value // page_size) * page_size


cdef int validate_prot_flags(int prot) except -1:  # no-cython-lint
    """
    Validate memory protection flags.

    Parameters
    ----------
    prot : int
        Protection flags (combination of PROT_* constants).

    Returns
    -------
    int
        0 on success.

    Raises
    ------
    ValueError
        If prot is negative, contains unknown flag bits, or is otherwise invalid.

    Notes
    -----
    Valid flags are: PROT_NONE (0), PROT_READ, PROT_WRITE, PROT_EXEC.
    These can be combined with bitwise OR.

    The validation checks:
    1. prot must be non-negative
    2. prot must only contain known PROT_* bits

    This prevents silent errors when invalid flags reach the C API boundary.

    Developer Notes
    ---------------
    Uses bitmask validation (prot & ~allowed) to detect any unknown bits.
    This is more robust than range checking (e.g., prot > 7) because:
    - Detects high bits: prot = 0x100 would pass range check but fail bitmask
    - Future-proof: works even if new flags are added
    - Clear error messages: reports exact unknown bits

    Migration Note
    --------------
    OLD: if prot < 0 or prot > 7  # Too permissive
    NEW: if (prot & ~allowed) != 0  # Detects all invalid bits

    Examples
    --------
    >>> validate_prot_flags(PROT_NONE)  # 0 - OK
    >>> validate_prot_flags(PROT_READ)  # 1 - OK
    >>> validate_prot_flags(PROT_READ | PROT_WRITE)  # 3 - OK
    >>> validate_prot_flags(-1)  # Raises ValueError
    >>> validate_prot_flags(0x100)  # Raises ValueError (unknown bit)
    """
    cdef int allowed = PROT_READ | PROT_WRITE | PROT_EXEC

    if prot < 0:
        raise ValueError(
            f"Protection flags cannot be negative: {prot}"
        )

    if (prot & ~allowed) != 0:
        raise ValueError(
            f"Invalid protection flags: {prot:#x}. "
            f"Unknown bits: {(prot & ~allowed):#x}. "
            f"Allowed flags: PROT_NONE (0), "
            f"PROT_READ ({PROT_READ}), "
            f"PROT_WRITE ({PROT_WRITE}), "
            f"PROT_EXEC ({PROT_EXEC})"
        )

    return 0

def py_validate_prot_flags(int prot):
    """
    Validate memory protection flags.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If flags are invalid.
    """
    validate_prot_flags(prot)


cdef int validate_map_flags(int flags) except -1:  # no-cython-lint
    """
    Validate memory mapping flags.

    Parameters
    ----------
    flags : int
        Mapping flags (combination of MAP_* constants).

    Returns
    -------
    int
        0 on success.

    Raises
    ------
    ValueError
        If flags are invalid, conflicting, or contain unknown bits.

    Notes
    -----
    Valid flags are: MAP_SHARED, MAP_PRIVATE, MAP_ANONYMOUS, MAP_FIXED.

    The validation enforces:
    1. No unknown flag bits are present (checked first for fail-fast)
    2. Exactly one of MAP_SHARED or MAP_PRIVATE must be set

    MAP_SHARED and MAP_PRIVATE are mutually exclusive by POSIX specification.

    Developer Notes
    ---------------
    Checks unknown bits first to fail fast on obvious errors.
    Uses explicit boolean logic for mutual exclusivity check for clarity.

    Migration Note
    --------------
    OLD: Only checked mutual exclusivity
    NEW: Also checks for unknown bits

    Examples
    --------
    >>> validate_map_flags(MAP_SHARED)  # OK
    >>> validate_map_flags(MAP_PRIVATE | MAP_ANONYMOUS)  # OK
    >>> validate_map_flags(MAP_SHARED | MAP_PRIVATE)  # Raises ValueError
    >>> validate_map_flags(0)  # Raises ValueError (missing SHARED/PRIVATE)
    >>> validate_map_flags(0x1000)  # Raises ValueError (unknown bit)
    """
    cdef int allowed = (MAP_SHARED | MAP_PRIVATE |
                        MAP_ANONYMOUS | MAP_FIXED)

    cdef bint has_shared = (flags & MAP_SHARED) != 0
    cdef bint has_private = (flags & MAP_PRIVATE) != 0

    # Check for unknown bits first (fail fast on obvious errors)
    if (flags & ~allowed) != 0:
        raise ValueError(
            f"Invalid mapping flags: {flags:#x}. "
            f"Unknown bits: {(flags & ~allowed):#x}. "
            f"Allowed flags: MAP_SHARED ({MAP_SHARED}), "
            f"MAP_PRIVATE ({MAP_PRIVATE}), "
            f"MAP_ANONYMOUS ({MAP_ANONYMOUS}), "
            f"MAP_FIXED ({MAP_FIXED})"
        )

    # Enforce exactly one of MAP_SHARED or MAP_PRIVATE
    # XOR logic: has_shared != has_private means exactly one is true
    if has_shared == has_private:  # Both true or both false
        if has_shared:
            raise ValueError(
                "MAP_SHARED and MAP_PRIVATE are mutually exclusive. "
                f"Received flags: {flags:#x}"
            )
        else:
            raise ValueError(
                "Must specify exactly one of MAP_SHARED or MAP_PRIVATE. "
                f"Received flags: {flags:#x}"
            )

    return 0


def py_validate_map_flags(int flags):
    """
    Validate memory mapping flags.

    Raises
    ------
    ValueError
        If flags are invalid.
    """
    validate_map_flags(flags)


cdef bint is_map_failed(void* ptr) noexcept nogil:  # no-cython-lint
    """
    Check if memory mapping operation failed.

    Parameters
    ----------
    ptr : void*
        Pointer returned from mmap() call.

    Returns
    -------
    bool
        True if ptr indicates mapping failure, False otherwise.

    Notes
    -----
    On both POSIX and Windows (via mman.h shim), MAP_FAILED is defined as
    ((void*)-1), i.e., all bits set to 1.

    This function provides a type-safe, explicit check instead of comparing
    directly with -1 or MAP_FAILED in application code.

    Developer Notes
    ---------------
    Cast to uintptr_t for safe integer comparison across platforms.
    Avoids pointer comparison warnings from strict compilers.

    Examples
    --------
    >>> ptr = mmap(NULL, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    >>> if is_map_failed(ptr):
    ...     # Handle error
    ...     pass
    """
    return ptr == <void*>-1

# End of utility functions
# ===========================================================================

# ===========================================================================
# Exception classes
# ===========================================================================

class MMapError(Exception):
    """Base exception for memory mapping errors."""
    pass


class MMapAllocationError(MMapError):
    """Raised when memory mapping allocation fails."""
    pass


class MMapInvalidParameterError(MMapError):
    """Raised when invalid parameters are provided."""
    pass


# ===========================================================================
# MemoryMap - Python wrapper for memory-mapped regions
# ===========================================================================

cdef class MemoryMap:
    """
    Memory-mapped region with automatic resource management.

    This class wraps a memory-mapped region and provides a Pythonic interface
    with context manager support for automatic cleanup.

    Attributes
    ----------
    addr : int (property, read-only)
        Memory address of the mapped region
    size : int (property, read-only)
        Size of the mapped region in bytes
    is_valid : bool (property, read-only)
        Whether the mapping is still valid

    Parameters
    ----------
    addr : int
        Memory address returned from mmap()
    size : int
        Size of the mapping in bytes

    Notes
    -----
    - Always use factory methods (create_*) to create instances
    - Use as context manager for automatic cleanup
    - Accessing closed mapping raises ValueError
    - The mapping exports its bytes through the buffer protocol, so
      ``memoryview(m)`` views it without copying. A ``memoryview`` and an
      array from :meth:`as_numpy_array` each hold an export, which keeps the
      mapped region alive: :meth:`close` ends this object's own API at once,
      and the region is unmapped when the last export is released. Using such
      a view after ``close()`` is therefore safe.
    - ``numpy.frombuffer(m)`` called directly holds an export from NumPy
      1.23 on (NumPy keeps a memoryview of its own as the array's base);
      older NumPy, still allowed on Python 3.8, releases it at once, and
      ``close()`` may then unmap under the array. :meth:`as_numpy_array`
      holds one with every NumPy.
    - An array's ``base`` memoryview is its export: calling
      ``arr.base.release()`` while the array is still used invalidates the
      array, as it does for any NumPy array over a released buffer.
    - A mapping without ``PROT_READ`` exports nothing (``BufferError``) and
      cannot be read; access through it would crash the process.
    - Methods that release the GIL around a system call or a copy
      (:meth:`read`, :meth:`write`, :meth:`msync`, :meth:`mlock`,
      :meth:`munlock`) hold the same kind of export while they run, so a
      :meth:`close` from another thread defers the unmap instead of unmapping
      under them. :meth:`mprotect` keeps the GIL for its system call, so the
      protection and the recorded flags change together, and it refuses to
      remove a protection bit while exports are alive (``BufferError``): a
      view exported as writable would crash the process on its next write.
    - Threads may share one mapping, on free-threaded CPython builds too:
      the open/closed state, the export count and the protection flags
      change under a lock of the mapping, so an export or a method never
      starts on a closed mapping, and the region is unmapped once, after
      the last export. What the threads write into the mapped bytes is
      theirs to order, as for any shared memory.

    Examples
    --------
    >>> with MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE) as m:
    ...     m.write(b"test")
    ...     data = m.read(4)
    """

    # C-level attributes
    cdef void* _addr       # ← C-owned, stable
    cdef size_t _size
    cdef bint _is_valid
    cdef int _prot
    cdef int _flags
    # Exports that still point at _addr: buffer exports (memoryviews, NumPy
    # arrays) and pins held by methods while they run without the GIL.
    # close() with exports alive defers munmap to the last release (_unpin).
    cdef Py_ssize_t _exports
    cdef bint _unmap_pending
    cdef Py_ssize_t _buffer_shape[1]
    # Guards _is_valid, _addr, _prot, _exports and _unmap_pending; _size,
    # _flags and _buffer_shape are fixed once the mapping exists. _addr
    # changes only while no hold is taken, so code holding one (an export,
    # or a method between _pin and _unpin) reads it without the lock.
    cdef skplt_mm_lock _lock

    def __cinit__(self):
        """
        C-level constructor.

        Initializes internal state.

        Notes
        -----
        - __cinit__ is guaranteed to be called exactly once
        - Actual mapping happens via factory methods
        """
        self._addr = NULL
        self._size = 0
        self._is_valid = False
        self._prot = 0
        self._flags = 0
        self._exports = 0
        self._unmap_pending = False

    def __dealloc__(self):
        """
        C-level destructor.

        Automatically unmaps the region if still valid.

        Notes
        -----
        - Called automatically when Python object is garbage collected
        - Must not raise exceptions
        - Unmaps whatever is still mapped, including a region whose
          ``close()`` was deferred (no view can outlive this object: every
          view holds a reference to it)
        """
        if self._addr is not NULL:
            munmap(self._addr, self._size)
            self._addr = NULL
            self._is_valid = False

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        """Export the mapped bytes as a 1-D unsigned-byte buffer."""
        cdef int prot = 0
        cdef int need = PROT_READ
        if (flags & PyBUF_WRITABLE) == PyBUF_WRITABLE:
            need = need | PROT_WRITE
        # The checks and the export count change together (_acquire): a
        # close() or mprotect() on another thread cannot come between them.
        cdef int status = self._acquire(need, &prot)
        if status == _CLOSED:
            raise BufferError("Mapping is closed")
        if status == _NOT_READABLE:
            raise BufferError("Mapping is not readable (created without PROT_READ)")
        if status == _NOT_WRITABLE:
            raise BufferError("Mapping is read-only (created without PROT_WRITE)")
        cdef bint writable = (prot & PROT_WRITE) != 0
        # From here on nothing can fail: the export is counted.
        buffer.buf = self._addr
        buffer.obj = self
        buffer.len = self._buffer_shape[0]
        buffer.readonly = not writable
        buffer.itemsize = 1
        buffer.ndim = 1
        buffer.format = NULL
        buffer.shape = NULL
        buffer.strides = NULL
        if (flags & PyBUF_FORMAT) == PyBUF_FORMAT:
            buffer.format = b"B"
        if (flags & PyBUF_ND) == PyBUF_ND:
            buffer.shape = self._buffer_shape
        if (flags & PyBUF_STRIDES) == PyBUF_STRIDES:
            buffer.strides = &buffer.itemsize
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        """Release one export; perform a deferred unmap after the last one."""
        self._unpin()

    cdef int _acquire(self, int need, int* prot) noexcept:
        """
        Hold the region mapped, if the mapping is open and has ``need``.

        A buffer export, or a method's GIL-free section. The checks and the
        count change under the lock, together: no ``close()`` or
        ``mprotect()`` comes between them. Returns ``_ACQUIRED`` (and the
        protection in ``prot``, unless NULL) or why not; raises nothing.
        """
        cdef int status = _ACQUIRED
        skplt_mm_acquire(&self._lock)
        if not self._is_valid:
            status = _CLOSED
        elif (need & PROT_READ) and not (self._prot & PROT_READ):
            status = _NOT_READABLE
        elif (need & PROT_WRITE) and not (self._prot & PROT_WRITE):
            status = _NOT_WRITABLE
        else:
            self._exports += 1
            if prot is not NULL:
                prot[0] = self._prot
        skplt_mm_release(&self._lock)
        return status

    cdef int _pin(self, int need) except -1:
        """Hold the region for a method; raise its documented error if not."""
        cdef int status = self._acquire(need, NULL)
        if status == _CLOSED:
            raise ValueError("Mapping is closed")
        if status == _NOT_READABLE:
            raise ValueError("Mapping is not readable")
        if status == _NOT_WRITABLE:
            raise ValueError("Mapping is not writable")
        return 0

    cdef void _unpin(self) noexcept:
        """Release one hold; perform a deferred unmap after the last one."""
        cdef void* addr = NULL
        skplt_mm_acquire(&self._lock)
        self._exports -= 1
        if self._exports == 0 and self._unmap_pending:
            self._unmap_pending = False
            addr = self._addr
            self._addr = NULL
        skplt_mm_release(&self._lock)
        # Nothing can raise here; a failed munmap gives the region back to
        # __dealloc__, which retries.
        if addr is not NULL and munmap(addr, self._size) != 0:
            skplt_mm_acquire(&self._lock)
            self._addr = addr
            skplt_mm_release(&self._lock)

    cdef bint _snapshot(self, int* prot) noexcept:
        """Return whether the mapping is open, and its protection in ``prot``."""
        cdef bint valid
        skplt_mm_acquire(&self._lock)
        valid = self._is_valid
        if prot is not NULL:
            prot[0] = self._prot
        skplt_mm_release(&self._lock)
        return valid

    def __init__(self):
        """
        Initialize MemoryMap.

        Notes
        -----
        - Do not call directly
        - Use factory methods: create_anonymous(), create_file_mapping()
        """
        pass

    @staticmethod
    def create_anonymous(
        size: int,
        prot: int = PROT_READ | PROT_WRITE,
        flags: int = MAP_PRIVATE
    ) -> MemoryMap:
        """
        Create anonymous memory mapping (not backed by file).

        Parameters
        ----------
        size : int
            Size of the mapping in bytes. Must be > 0.
        prot : int, optional
            Memory protection flags (PROT_READ, PROT_WRITE, etc.).
            Default: PROT_READ | PROT_WRITE
        flags : int, optional
            Mapping flags (should include MAP_PRIVATE or MAP_SHARED).
            Default: MAP_PRIVATE

        Returns
        -------
        MemoryMap
            New memory-mapped region

        Raises
        ------
        ValueError
            If size <= 0 or invalid flags
        MMapAllocationError
            If mapping allocation fails

        Notes
        -----
        - Memory is initially zero-filled
        - Anonymous mappings are not backed by any file
        - Useful for inter-process communication with MAP_SHARED

        Examples
        --------
        >>> m = MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE)
        >>> m.write(b"Hello")
        >>> m.close()

        Using context manager (recommended):

        >>> with MemoryMap.create_anonymous(4096) as m:
        ...     m.write(b"Hello, World!")
        """
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        cdef MemoryMap instance = MemoryMap()
        instance._create_mapping(
            NULL,  # Let system choose address
            size,
            prot,
            flags | MAP_ANONYMOUS,  # Add MAP_ANONYMOUS flag
            -1,  # No file descriptor for anonymous mapping
            0    # No offset
        )
        return instance

    @staticmethod
    def create_file_mapping(
        fd: int,
        offset: int,
        size: int,
        prot: int = PROT_READ,
        flags: int = MAP_PRIVATE
    ) -> MemoryMap:
        """
        Create file-backed memory mapping.

        Parameters
        ----------
        fd : int
            File descriptor of open file
        offset : int
            Offset in file to start mapping (must be page-aligned)
        size : int
            Size of the mapping in bytes. Must be > 0.
        prot : int, optional
            Memory protection flags.
            Default: PROT_READ
        flags : int, optional
            Mapping flags.
            Default: MAP_PRIVATE

        Returns
        -------
        MemoryMap
            New memory-mapped region

        Raises
        ------
        ValueError
            If fd < 0, size <= 0, or invalid flags
        MMapAllocationError
            If mapping allocation fails

        Notes
        -----
        - File must be opened with appropriate permissions
        - offset must be page-aligned (typically 4096 bytes)
        - MAP_SHARED changes are written back to file
        - MAP_PRIVATE creates copy-on-write mapping

        Examples
        --------
        >>> with open("data.bin", "r+b") as f:
        ...     m = MemoryMap.create_file_mapping(f.fileno(), 0, 4096, PROT_READ)
        ...     data = m.read(100)
        ...     m.close()
        """
        if fd < 0:
            raise ValueError(f"Invalid file descriptor: {fd}")
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")

        cdef MemoryMap instance = MemoryMap()
        instance._create_mapping(
            NULL,  # Let system choose address
            size,
            prot,
            flags,
            fd,
            offset
        )
        return instance

    cdef void _create_mapping(
        self,
        void* addr,
        size_t size,
        int prot,
        int flags,
        int fd,
        off_t offset
    ) except *:
        """
        Internal method to create memory mapping.

        Parameters
        ----------
        addr : void*
            Suggested address (usually NULL)
        size : size_t
            Size in bytes
        prot : int
            Protection flags
        flags : int
            Mapping flags
        fd : int
            File descriptor
        offset : off_t
            File offset

        Raises
        ------
        MMapAllocationError
            If mmap() fails

        Notes
        -----
        - This is a C-level method (cdef)
        - Validates inputs and calls C mmap()
        - Sets internal state on success
        """
        # Python-level validation flags (GIL held)
        validate_prot_flags(prot)
        validate_map_flags(flags)

        # Explicit C-level copies
        cdef void* c_addr = addr
        cdef size_t c_size = size
        cdef int c_prot = prot
        cdef int c_flags = flags
        cdef int c_fd = fd
        cdef off_t c_offset = offset

        # System Call C mmap function
        cdef void* result
        with nogil:
            result = mmap(c_addr, c_size, c_prot, c_flags, c_fd, c_offset)

        # Check for failure
        if is_map_failed(result):
            err = errno
            raise MMapAllocationError(
                f"Failed to create memory mapping: errno={err}, "
                f"size={size}, prot={prot}, flags={flags}"
            )

        # Store mapping info
        self._addr = result
        self._size = size
        self._prot = prot
        self._flags = flags
        self._buffer_shape[0] = <Py_ssize_t>size
        self._is_valid = True

    @property
    def addr(self) -> int:
        """
        Get memory address of mapped region.

        Returns
        -------
        int
            Memory address as integer

        Raises
        ------
        ValueError
            If mapping is closed
        """
        cdef bint valid
        cdef uintptr_t addr
        skplt_mm_acquire(&self._lock)
        valid = self._is_valid
        addr = <uintptr_t>self._addr
        skplt_mm_release(&self._lock)
        if not valid:
            raise ValueError("Mapping is closed")
        return addr

    @property
    def size(self) -> int:
        """
        Get size of mapped region.

        Returns
        -------
        int
            Size in bytes

        Raises
        ------
        ValueError
            If mapping is closed
        """
        if not self._snapshot(NULL):
            raise ValueError("Mapping is closed")
        return self._size

    @property
    def is_valid(self) -> bool:
        """
        Check if mapping is still valid.

        Returns
        -------
        bool
            True if valid, False if closed
        """
        return self._snapshot(NULL)

    def close(self) -> None:
        """
        Close the memory mapping.

        Unmaps the region and releases resources.

        Raises
        ------
        MMapError
            If munmap() fails; the mapping then stays open.

        Notes
        -----
        - After closing, the mapping cannot be used
        - Called automatically by __dealloc__ or context manager
        - Safe to call multiple times (idempotent)
        - If views (``as_numpy_array``, ``memoryview``) are still alive, or
          another thread is inside a method that runs without the GIL
          (``read``, ``write``, ...), the unmap is deferred until the last of
          them is released. It used to unmap at once: reading such a view --
          for instance an array returned from a ``with`` block -- or a
          ``read`` running on another thread killed the process with SIGSEGV.
        - If ``munmap`` fails, the mapping is open again when this raises.
          Until then other threads see it closed: their exports and methods
          are refused, and a second ``close()`` returns at once.

        Examples
        --------
        >>> m = MemoryMap.create_anonymous(4096)
        >>> m.close()
        >>> m.is_valid
        False
        """
        cdef void* addr = NULL
        cdef size_t size = self._size
        cdef bint take = False
        cdef int result
        cdef int err = 0

        # Mark closed, then defer or take the region, under the lock: a
        # concurrent close() finds the mapping closed instead of unmapping
        # the same address a second time, and an export or a method either
        # finished taking its hold before (the unmap is deferred to its
        # release) or finds the mapping closed.
        skplt_mm_acquire(&self._lock)
        if self._is_valid:
            self._is_valid = False
            if self._exports > 0:
                self._unmap_pending = True
            else:
                addr = self._addr
                self._addr = NULL
                take = True
        skplt_mm_release(&self._lock)
        if not take:
            return  # already closed (idempotent), or deferred

        with nogil:
            result = munmap(addr, size)
            if result != 0:
                err = errno

        if result != 0:
            skplt_mm_acquire(&self._lock)
            self._addr = addr
            self._is_valid = True
            skplt_mm_release(&self._lock)
            raise MMapError(f"Failed to unmap memory: errno={err}")

    def read(self, size: int, offset: int = 0) -> bytes:
        """
        Read bytes from mapped region.

        Parameters
        ----------
        size : int
            Number of bytes to read
        offset : int, optional
            Offset from start of mapping. Default: 0

        Returns
        -------
        bytes
            Data read from mapping

        Raises
        ------
        ValueError
            If mapping is closed, not readable, or parameters are invalid

        Examples
        --------
        >>> with MemoryMap.create_anonymous(4096) as m:
        ...     _ = m.write(b"Hello")
        ...     data = m.read(5)
        ...     print(data)
        b'Hello'
        """
        cdef int prot = 0
        if not self._snapshot(&prot):
            raise ValueError("Mapping is closed")
        if not (prot & PROT_READ):
            # Reading a mapping without PROT_READ faults (SIGSEGV); refuse it
            # the way write() refuses a mapping without PROT_WRITE.
            raise ValueError("Mapping is not readable")
        if size < 0:
            raise ValueError(f"Size must be non-negative, got {size}")
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")
        if offset + size > self._size:
            raise ValueError(
                f"Read beyond mapping bounds: "
                f"offset={offset}, size={size}, mapping_size={self._size}"
            )

        cdef size_t c_size = size
        cdef uintptr_t c_offset = offset
        cdef bytearray buf
        cdef char* dst
        cdef const char* src

        # Pin the region before anything can release the GIL (the allocation
        # may run a garbage collection; the copy releases it): a close() from
        # another thread then defers the unmap until this copy is done. It
        # used to unmap under the running memcpy (SIGSEGV). The pin checks
        # again, with the count: the checks above were a snapshot.
        self._pin(PROT_READ)
        try:
            buf = bytearray(c_size)
            dst = PyByteArray_AS_STRING(buf)
            src = <const char*>(<uintptr_t>self._addr + c_offset)
            with nogil:
                memcpy(dst, src, c_size)
        finally:
            self._unpin()

        return bytes(buf)

    def write(self, data: bytes, offset: int = 0) -> int:
        """
        Write bytes to mapped region.

        Parameters
        ----------
        data : bytes
            Data to write
        offset : int, optional
            Offset from start of mapping. Default: 0

        Returns
        -------
        int
            Number of bytes written

        Raises
        ------
        ValueError
            If mapping is closed, not writable, or parameters invalid

        Examples
        --------
        >>> with MemoryMap.create_anonymous(4096, PROT_WRITE) as m:
        ...     n = m.write(b"Hello, World!")
        ...     print(n)
        13
        """
        cdef int prot = 0
        if not self._snapshot(&prot):
            raise ValueError("Mapping is closed")
        if not (prot & PROT_WRITE):
            raise ValueError("Mapping is not writable")
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")

        # 1. Extract pointer safely (GIL held)
        cdef const char* src
        cdef Py_ssize_t n

        # ---- SAFE extraction (GIL held)
        PyBytes_AsStringAndSize(data, <char**>&src, &n)

        if offset + n > self._size:
            raise ValueError("Write beyond mapping bounds")

        cdef uintptr_t c_offset = offset
        cdef char* dst

        # Pinned for the GIL-free copy; see ``read``.
        self._pin(PROT_WRITE)
        try:
            dst = <char*>(<uintptr_t>self._addr + c_offset)
            with nogil:
                memcpy(dst, src, <size_t>n)
        finally:
            self._unpin()

        return n

    def mprotect(self, prot: int) -> None:
        """
        Change memory protection of mapped region.

        Parameters
        ----------
        prot : int
            New protection flags

        Raises
        ------
        ValueError
            If mapping is closed or flags invalid
        BufferError
            If *prot* removes a protection bit while views of the mapping
            (``memoryview``, :meth:`as_numpy_array`) are alive, or while
            another thread runs :meth:`read`, :meth:`write`, :meth:`msync`,
            :meth:`mlock` or :meth:`munlock` on it: a view exported as writable
            or readable, or the running copy, would crash the process on its
            next access. Adding bits is always allowed.
        MMapError
            If mprotect() fails

        Examples
        --------
        >>> with MemoryMap.create_anonymous(4096, PROT_READ) as m:
        ...     m.mprotect(PROT_READ | PROT_WRITE)
        ...     m.write(b"Now writable!")
        """
        if not self._snapshot(NULL):
            raise ValueError("Mapping is closed")

        validate_prot_flags(prot)

        cdef int c_prot = prot
        cdef int status = 0  # 0 done, 1 closed, 2 a right under exports, 3 failed
        cdef Py_ssize_t exports = 0
        cdef int err = 0

        # The check, the system call and the recorded flags change together,
        # under the lock (and with the GIL held, unlike the other methods).
        # Apart, another thread could export a writable view (or write) from
        # the stale flags into pages already read-only.
        skplt_mm_acquire(&self._lock)
        if not self._is_valid:
            status = 1
        elif self._exports > 0 and (c_prot & self._prot) != self._prot:
            status = 2
            exports = self._exports
        elif mprotect(self._addr, self._size, c_prot) != 0:
            status = 3
            err = errno
        else:
            self._prot = c_prot
        skplt_mm_release(&self._lock)

        if status == 1:
            raise ValueError("Mapping is closed")
        if status == 2:
            raise BufferError(
                f"Cannot remove protection while {exports} view(s) of the "
                "mapping or running operation(s) on it (read, write, msync, "
                "mlock, munlock) are alive: an access through one of them "
                "would crash the process. Release the views first."
            )
        if status == 3:
            raise MMapError(f"Failed to change protection: errno={err}")

    def msync(self, flags: int = MS_SYNC) -> None:
        """
        Synchronize mapped region with backing storage.

        Parameters
        ----------
        flags : int, optional
            Sync flags (MS_ASYNC, MS_SYNC, MS_INVALIDATE).
            Default: MS_SYNC

        Raises
        ------
        ValueError
            If mapping is closed
        MMapError
            If msync() fails

        Notes
        -----
        - Only meaningful for file-backed mappings
        - MS_SYNC blocks until sync complete
        - MS_ASYNC returns immediately

        Examples
        --------
        >>> with MemoryMap.create_file_mapping(fd, 0, 4096, PROT_WRITE, MAP_SHARED) as m:
        ...     m.write(b"Data")
        ...     m.msync(MS_SYNC)  # Ensure written to disk
        """
        if not self._snapshot(NULL):
            raise ValueError("Mapping is closed")

        cdef int c_flags = flags
        cdef void* addr
        cdef size_t size
        cdef int result
        cdef int err = 0
        self._pin(0)
        try:
            addr = self._addr
            size = self._size
            with nogil:
                result = msync(addr, size, c_flags)
                if result != 0:
                    err = errno  # before _unpin, whose munmap may set it
        finally:
            self._unpin()

        if result != 0:
            raise MMapError(f"Failed to sync: errno={err}")

    # ---------------------------------------------------------------
    # Page-size query (exposes the C-level helper to Python)
    # ---------------------------------------------------------------

    @property
    def page_size(self) -> int:
        """
        System page size in bytes.

        Returns
        -------
        int
            Page size as reported by the kernel (e.g. 4096 or 16384).

        Notes
        -----
        File-mapping offsets passed to ``create_file_mapping`` must be
        multiples of this value.  Use it for manual alignment::

            aligned = (raw_offset // m.page_size) * m.page_size
        """
        return <int>get_page_size()

    # ---------------------------------------------------------------
    # Page-locking (mlock / munlock)
    # ---------------------------------------------------------------

    def mlock(self) -> None:
        """
        Lock mapped pages in physical memory (prevent swapping).

        Raises
        ------
        ValueError
            If the mapping is already closed.
        MMapError
            If the underlying ``mlock()`` system call fails.  On Linux
            this commonly means the process has exceeded its
            ``RLIMIT_MEMLOCK`` soft limit; on Windows it may require
            ``SE_LOCK_MEMORY_NAME`` privilege.

        Notes
        -----
        Locking pages is useful for latency-sensitive code (e.g.
        real-time signal processing) that cannot tolerate page faults.
        Remember to call :py:meth:`munlock` when the guarantee is no
        longer needed; otherwise the locked pages count against the
        process resource limit for the lifetime of the mapping.

        Examples
        --------
        >>> with MemoryMap.create_anonymous(4096) as m:
        ...     m.mlock()       # pages will not be swapped out
        ...     m.write(b"latency-critical data")
        ...     m.munlock()     # release the lock
        """
        if not self._snapshot(NULL):
            raise ValueError("Mapping is closed")

        cdef void* c_addr
        cdef size_t c_size
        cdef int result
        cdef int err = 0
        self._pin(0)
        try:
            c_addr = self._addr
            c_size = self._size
            with nogil:
                result = mlock(c_addr, c_size)
                if result != 0:
                    err = errno  # before _unpin, whose munmap may set it
        finally:
            self._unpin()

        if result != 0:
            raise MMapError(
                f"mlock failed: errno={err} "
                f"(check RLIMIT_MEMLOCK / SeLockmemoryPrivilege)"
            )

    def munlock(self) -> None:
        """
        Unlock mapped pages (allow the kernel to swap them out again).

        Raises
        ------
        ValueError
            If the mapping is already closed.
        MMapError
            If the underlying ``munlock()`` system call fails.

        Notes
        -----
        This is the inverse of :py:meth:`mlock`.  Calling ``munlock``
        on pages that were never locked is a no-op on most platforms
        but the behaviour is technically undefined by POSIX; avoid it.

        Examples
        --------
        >>> with MemoryMap.create_anonymous(4096) as m:
        ...     m.mlock()
        ...     # … do latency-critical work …
        ...     m.munlock()
        """
        if not self._snapshot(NULL):
            raise ValueError("Mapping is closed")

        cdef void* c_addr
        cdef size_t c_size
        cdef int result
        cdef int err = 0
        self._pin(0)
        try:
            c_addr = self._addr
            c_size = self._size
            with nogil:
                result = munlock(c_addr, c_size)
                if result != 0:
                    err = errno  # before _unpin, whose munmap may set it
        finally:
            self._unpin()

        if result != 0:
            raise MMapError(f"munlock failed: errno={err}")

    # ---------------------------------------------------------------
    # Zero-copy NumPy view
    # ---------------------------------------------------------------

    def as_numpy_array(self, dtype=None):
        """
        Return a NumPy array that shares memory with this mapping.

        No data is copied. The array holds a buffer export of this mapping,
        so the mapped region stays alive as long as the array (or any view of
        it) does, even after :meth:`close`.

        Parameters
        ----------
        dtype : numpy.dtype or None, optional
            Desired element type of the output array.  When *None*
            (default) the raw view is returned as ``numpy.uint8``.
            Any dtype whose ``itemsize`` evenly divides ``self.size``
            is accepted.

        Returns
        -------
        numpy.ndarray
            A 1-D array viewing the mapped memory. It is writeable only when
            the mapping was created with ``PROT_WRITE``.

        Raises
        ------
        ValueError
            If the mapping is closed, or if ``dtype.itemsize`` does
            not evenly divide the mapping size.
        ImportError
            If NumPy is not installed.

        Notes
        -----
        **Developer.** The array is built with ``numpy.frombuffer`` over this
        object's buffer protocol, which counts exports; ``close`` defers the
        ``munmap`` while any export is alive. The previous implementation
        wrapped the raw address in a ``ctypes`` buffer: that kept this object
        alive but not the region, because ``close`` (called by every ``with``
        block) unmapped it regardless, and touching the array afterwards
        dereferenced unmapped memory (SIGSEGV).

        Examples
        --------
        >>> import numpy as np
        >>> with MemoryMap.create_anonymous(4096) as m:
        ...     arr = m.as_numpy_array()          # uint8 view
        ...     arr[:5] = [72, 101, 108, 108, 111]
        ...     print(m.read(5))
        b'Hello'
        >>> int(arr[0])  # still valid after the with block closed m
        72

        Reinterpret as 32-bit floats:

        >>> with MemoryMap.create_anonymous(4096) as m:
        ...     arr = m.as_numpy_array(dtype=np.float32)
        ...     arr[0] = 3.14
        """
        import numpy as np

        if not self._snapshot(NULL):
            raise ValueError("Mapping is closed")

        if dtype is None:
            dtype = np.dtype(np.uint8)
        else:
            dtype = np.dtype(dtype)

        if self._size % dtype.itemsize != 0:
            raise ValueError(
                f"Mapping size {self._size} is not evenly divisible "
                f"by dtype itemsize {dtype.itemsize}"
            )

        # NumPy before 1.23 releases the buffer frombuffer obtains at once and
        # keeps only a reference to its argument (1.23+ keeps a memoryview of
        # its own). Passing a memoryview makes the export holder explicit with
        # every supported NumPy: arr.base is the memoryview, which holds one
        # export of this mapping for as long as the array lives. Its readonly
        # flag follows PROT_WRITE, and so does the array's.
        return np.frombuffer(memoryview(self), dtype=dtype)

    def __enter__(self) -> MemoryMap:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - automatically closes mapping."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        cdef bint valid
        cdef size_t addr
        skplt_mm_acquire(&self._lock)
        valid = self._is_valid
        addr = <size_t>self._addr
        skplt_mm_release(&self._lock)
        if valid:
            return f"MemoryMap(addr=0x{addr:x}, size={self._size})"
        else:
            return "MemoryMap(closed)"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()


# ===========================================================================
# Module-level convenience functions
# ===========================================================================

def mmap_region(
    size: int,
    prot: int = PROT_READ | PROT_WRITE,
    flags: int = MAP_PRIVATE | MAP_ANONYMOUS,
    fd: int = -1,
    offset: int = 0
) -> MemoryMap:
    """
    Create a memory-mapped region (convenience function).

    Parameters
    ----------
    size : int
        Size of mapping in bytes
    prot : int, optional
        Protection flags. Default: PROT_READ | PROT_WRITE
    flags : int, optional
        Mapping flags. Default: MAP_PRIVATE | MAP_ANONYMOUS
    fd : int, optional
        File descriptor. Default: -1 (anonymous)
    offset : int, optional
        File offset. Default: 0

    Returns
    -------
    MemoryMap
        New memory mapping

    Examples
    --------
    >>> m = mmap_region(4096)
    >>> m.write(b"Hello")
    >>> m.close()
    """
    if flags & MAP_ANONYMOUS:
        return MemoryMap.create_anonymous(size, prot, flags)
    else:
        return MemoryMap.create_file_mapping(fd, offset, size, prot, flags)
