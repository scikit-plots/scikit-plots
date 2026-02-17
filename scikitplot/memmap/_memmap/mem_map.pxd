# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# scikitplot/memmap/_memmap/mem_map.pxd

"""
Memory Mapping Interface Declarations
======================================

Cross-platform memory mapping interface for POSIX and Windows systems.

This header file provides C-level declarations for memory mapping operations.
All implementations are in mem_map.pyx to maintain proper separation of
interface and implementation.

Notes
-----
This is a pure declaration file (.pxd). It contains NO implementations.
All function bodies are in the corresponding .pyx file.

Platform Support
----------------
- Windows: Uses custom implementation (CreateFileMapping wrapper)
- Unix/Linux/macOS: Uses standard POSIX <sys/mman.h>
- Unified interface regardless of platform

References
----------
- Cython .pxd files: https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html
- POSIX mmap: https://man7.org/linux/man-pages/man2/mmap.2.html
- Windows Memory Mapping: https://docs.microsoft.com/en-us/windows/win32/memory/file-mapping

See Also
--------
mem_map.pyx : Implementation of all declared functions
mman.h : Underlying C/C++ cross-platform memory mapping header
"""

from sys import platform

from libc.stdint cimport uintptr_t, int64_t
from libc.stddef cimport size_t
from libc.errno cimport errno

# ===========================================================================
# Platform-specific type definitions
# ===========================================================================

# off_t is platform-dependent:
# - Windows 64-bit: typically int64_t
# - Unix/Linux: defined in sys/types.h (often long or int64_t)
# We normalize to int64_t for consistency across platforms

ctypedef int64_t off_t

# ===========================================================================
# Type aliases for clarity
# ===========================================================================

ctypedef void* mmap_addr_t
ctypedef size_t mmap_size_t
ctypedef off_t mmap_offset_t


# ===========================================================================
# Cross-Platform Memory Mapping API
# ===========================================================================

cdef extern from "../../cexternals/_annoy/src/mman.h" nogil:

    # Memory protection flags
    int PROT_NONE
    int PROT_READ
    int PROT_WRITE
    int PROT_EXEC

    # Memory mapping flags
    int MAP_FILE
    int MAP_SHARED
    int MAP_PRIVATE
    int MAP_TYPE
    int MAP_FIXED
    int MAP_ANONYMOUS
    int MAP_ANON

    # Memory sync flags
    int MS_ASYNC
    int MS_SYNC
    int MS_INVALIDATE

    # Failed mapping sentinel
    # MAP_FAILED: Final[int] = ((void *)-1)
    void* MAP_FAILED
    int FILE_MAP_EXECUTE

    # Core memory mapping functions
    void* mmap(void* addr, size_t length, int prot, int flags,
               int fd, off_t offset) nogil
    int munmap(void* addr, size_t length) nogil
    int mprotect(void* addr, size_t length, int prot) nogil
    int msync(void* addr, size_t length, int flags) nogil
    int mlock(void* addr, size_t length) nogil
    int munlock(void* addr, size_t length) nogil

# ===========================================================================
# Version information
# ===========================================================================

DEF VERSION_MAJOR = 1
DEF VERSION_MINOR = 0
DEF VERSION_PATCH = 1

# Runtime-accessible version (defined in .pyx)
cdef object __version__

# ===========================================================================
# Utility Function Declarations (Implementations in .pyx)
# ===========================================================================

cdef size_t get_page_size() noexcept nogil

# """
# Get system page size.
#
# Returns
# -------
# size_t
#     System page size in bytes.
#
# Notes
# -----
# Implementation in mem_map.pyx.
# Thread-safe (no global state).
# """

cdef bint is_page_aligned(
    size_t addr,
    # size_t page_size
) noexcept nogil

# """
# Check if value is aligned to page boundary.
#
# Parameters
# ----------
# value : size_t
#     Value to check for alignment.
#
# Returns
# -------
# bool
#     True if value is page-aligned, False otherwise.
#
# Notes
# -----
# Implementation in mem_map.pyx.
# """

cdef size_t align_to_page(
    size_t value,
    # size_t page_size
) noexcept nogil

# """
# Round value to page boundary.
#
# Parameters
# ----------
# value : size_t
#     Value to align.
#
# Returns
# -------
# size_t
#     Value rounded to nearest page boundary.
#
# Notes
# -----
# Implementation in mem_map.pyx.
# Includes overflow protection.
# """

cdef int validate_prot_flags(int prot) except -1

# """
# Validate memory protection flags.
#
# Parameters
# ----------
# prot : int
#     Protection flags to validate.
#
# Returns
# -------
# int
#     0 on success.
#
# Raises
# ------
# ValueError
#     If flags are invalid or contain unknown bits.
#
# Notes
# -----
# Implementation in mem_map.pyx.
# Uses bitmask validation.
# """

cdef int validate_map_flags(int flags) except -1

# """
# Validate memory mapping flags.
#
# Parameters
# ----------
# flags : int
#     Mapping flags to validate.
#
# Returns
# -------
# int
#     0 on success.
#
# Raises
# ------
# ValueError
#     If flags are invalid, conflicting, or contain unknown bits.
#
# Notes
# -----
# Implementation in mem_map.pyx.
# Checks for unknown bits and mutual exclusivity.
# """

cdef bint is_map_failed(void* ptr) noexcept nogil

# """
# Check if memory mapping failed.
#
# Parameters
# ----------
# ptr : void*
#     Pointer returned from mmap.
#
# Returns
# -------
# bool
#     True if mapping failed, False otherwise.
#
# Notes
# -----
# Implementation in mem_map.pyx.
# MAP_FAILED is ((void*)-1) on all platforms.
# """
