# cython: language_level=3
# distutils: language = c++
"""
Cython declarations file for cross-platform memory mapping.

This file contains C-level type declarations and external definitions
for memory mapping functions (mmap, munmap, etc.) that work on both
Windows and Unix/Linux/macOS platforms.

Purpose
-------
- Declare C functions from mman.h (cross-platform)
- Define cimport-able types for use in other .pyx files
- Expose C-level constants and function signatures
- NO Python-facing logic (that goes in .pyx)
- NO implementation code (declarations only)

Design Principles
-----------------
- Platform-agnostic interface (abstracts Windows vs POSIX)
- Keep declarations minimal and exact
- Use 'cdef extern from' to import from C header
- Declare all public C functions that Cython needs
- Use 'nogil' for thread-safe operations

Modern Cython Best Practices
-----------------------------
- Use DEF for compile-time constants
- Use specific integer types (size_t, off_t, int)
- Annotate with 'const' where appropriate
- Use 'nogil' for functions that don't touch Python objects
- Platform detection via compile-time conditionals

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
"""

from libc.stdint cimport int64_t
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

# ===========================================================================
# Compile-time constants
# ===========================================================================

DEF CYTHON_LANGUAGE_LEVEL = 3

# ===========================================================================
# Platform-specific type definitions
# ===========================================================================

# off_t is platform-dependent:
# - Windows 64-bit: typically int64_t
# - Unix/Linux: defined in sys/types.h (often long or int64_t)
# We normalize to int64_t for consistency across platforms
ctypedef int64_t off_t

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

# DEF MAP_FAILED = ((void *)-1)
DEF FILE_MAP_EXECUTE = 0x0020

# ===========================================================================
# C function declarations from mman.h
# ===========================================================================

# Cross-platform memory mapping functions
# On Windows: implemented in mman.h using Windows API
# On Unix/Linux: standard POSIX functions from <sys/mman.h>
#
# IMPORTANT: Do NOT add docstrings inside cdef extern blocks!
# They cause C++ compilation errors ("Memory does not name a type")

# ---------------------------------------------------------------------------
# sysconf – required by get_page_size() on POSIX.
# On Windows sysconf does not exist, but the generated C++ never calls it
# there because get_page_size() falls back to 4096 when ps <= 0.
# ---------------------------------------------------------------------------
cdef extern from "<unistd.h>" nogil:
    long sysconf(int name) nogil

# _SC_PAGESIZE is a compile-time macro on every POSIX system.  Pull the real
# value via a verbatim C snippet; provide a Windows dummy so the translation
# unit parses on MSVC even though the value is never used there.
cdef extern from *:
    """
    #if defined(_WIN32) || defined(_WIN64)
        #ifndef _SC_PAGESIZE
            #define _SC_PAGESIZE 30
        #endif
    #endif
    """
    int _SC_PAGESIZE

cdef extern from "../src/mman.h" nogil:

    # --- Memory protection flags ---
    # int PROT_NONE
    # int PROT_READ
    # int PROT_WRITE
    # int PROT_EXEC

    # --- mmap flags ---
    # int MAP_FILE
    # int MAP_SHARED
    # int MAP_PRIVATE
    # int MAP_TYPE
    # int MAP_FIXED
    # int MAP_ANONYMOUS
    # int MAP_ANON

    # --- msync flags ---
    # int MS_ASYNC
    # int MS_SYNC
    # int MS_INVALIDATE

    # --- mmap failure sentinel ---
    # void* MAP_FAILED

    # --- Windows compatibility ---
    # int FILE_MAP_EXECUTE

    # Memory mapping function
    # Creates a new memory mapping in the virtual address space
    void* mmap(
        void* addr,    # Hint for address (NULL = system chooses)
        size_t len,    # Length of mapping in bytes
        int prot,      # Memory protection (PROT_READ | PROT_WRITE | PROT_EXEC)
        int flags,     # Mapping flags (MAP_SHARED | MAP_PRIVATE | MAP_ANONYMOUS)
        int fildes,    # File descriptor (-1 for anonymous)
        off_t off      # Offset in file (must be page-aligned)
    ) nogil

    # Unmap memory region
    # Returns 0 on success, -1 on error (errno set)
    int munmap(
        void* addr,    # Start address of mapping
        size_t len     # Length of mapping
    ) nogil

    # Change protection of memory region
    # Returns 0 on success, -1 on error (errno set)
    int mprotect(
        void* addr,    # Start address of region
        size_t len,    # Length of region
        int prot       # New protection flags
    ) nogil

    # Synchronize memory region with backing storage
    # Returns 0 on success, -1 on error (errno set)
    int msync(
        void* addr,    # Start address of region
        size_t len,    # Length of region
        int flags      # Sync flags (MS_ASYNC | MS_SYNC | MS_INVALIDATE)
    ) nogil

    # Lock pages in memory (prevent swapping)
    # Returns 0 on success, -1 on error (errno set)
    int mlock(
        const void* addr,  # Start address of region
        size_t len         # Length of region
    ) nogil

    # Unlock pages (allow swapping)
    # Returns 0 on success, -1 on error (errno set)
    int munlock(
        const void* addr,  # Start address of region
        size_t len         # Length of region
    ) nogil

    # Truncate file to specified size
    # Returns 0 on success, -1 on error (errno set)
    # On Windows: custom implementation
    # On Unix/Linux: standard POSIX function
    int ftruncate(
        const int fd,       # File descriptor
        const int64_t size  # New file size in bytes
    ) nogil

# ===========================================================================
# Type aliases for clarity
# ===========================================================================

ctypedef void* mmap_addr_t
ctypedef size_t mmap_size_t
ctypedef off_t mmap_offset_t

# ===========================================================================
# Version information
# ===========================================================================

DEF VERSION_MAJOR = 1
DEF VERSION_MINOR = 0
DEF VERSION_PATCH = 1

# Runtime-accessible version (defined in .pyx)
cdef object __version__

# ===========================================================================
# Helper functions (inline, can be used by other Cython modules)
# ===========================================================================

cdef inline bint is_map_failed(void* ptr) nogil:
    """
    Check if mmap returned MAP_FAILED.

    Parameters
    ----------
    ptr : void*
        Return value from mmap()

    Returns
    -------
    bint
        True if ptr is MAP_FAILED ((void*)-1), False otherwise

    Notes
    -----
    MAP_FAILED is defined as ((void*)-1) on all platforms.
    This is a special sentinel value indicating mmap failure.
    Always check mmap return value before using the pointer.

    Examples
    --------
    cdef void* addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0)
    if is_map_failed(addr):
        # Handle error (errno is set)
        raise OSError(errno)
    """
    return ptr == <void*>-1


cdef inline int validate_prot_flags(int prot) except -1:
    """
    Validate memory protection flags.

    Parameters
    ----------
    prot : int
        Protection flags to validate

    Returns
    -------
    int
        The validated flags (unchanged)

    Raises
    ------
    ValueError
        If flags are invalid or out of range

    Notes
    -----
    Valid protection flags:
    - PROT_NONE (0): No access
    - PROT_READ (1): Read access
    - PROT_WRITE (2): Write access
    - PROT_EXEC (4): Execute access
    - Combinations: PROT_READ | PROT_WRITE, etc.

    Maximum valid value is PROT_READ | PROT_WRITE | PROT_EXEC = 7.

    Examples
    --------
    validate_prot_flags(PROT_READ | PROT_WRITE)  # OK, returns 3
    validate_prot_flags(0xFF)                     # Error, too large
    validate_prot_flags(-1)                       # Error, negative
    """
    # Check range: must be between 0 and 7 (all flags set)
    if prot < 0 or prot > (PROT_READ | PROT_WRITE | PROT_EXEC):
        raise ValueError(f"Invalid protection flags: {prot} (must be 0-7)")
    return prot


cdef inline int validate_map_flags(int flags) except -1:
    """
    Validate memory mapping flags.

    Parameters
    ----------
    flags : int
        Mapping flags to validate

    Returns
    -------
    int
        The validated flags (unchanged)

    Raises
    ------
    ValueError
        If flags are invalid or contain conflicting options

    Notes
    -----
    Required: Must have exactly ONE of:
    - MAP_SHARED: Changes visible to other processes
    - MAP_PRIVATE: Copy-on-write, changes private

    Optional flags:
    - MAP_ANONYMOUS: Anonymous mapping (no file backing)
    - MAP_FIXED: Use exact address specified

    Invalid combinations:
    - MAP_SHARED | MAP_PRIVATE (mutually exclusive)
    - Neither MAP_SHARED nor MAP_PRIVATE

    Examples
    --------
    validate_map_flags(MAP_SHARED | MAP_ANONYMOUS)   # OK
    validate_map_flags(MAP_PRIVATE)                  # OK
    validate_map_flags(MAP_SHARED | MAP_PRIVATE)     # Error, conflict
    validate_map_flags(0)                            # Error, no type
    """
    # Extract sharing type flags
    cdef int shared = flags & MAP_SHARED
    cdef int private = flags & MAP_PRIVATE

    # Must have exactly one of MAP_SHARED or MAP_PRIVATE
    if shared and private:
        raise ValueError(
            "Invalid flags: Cannot specify both MAP_SHARED and MAP_PRIVATE"
        )

    if not shared and not private:
        raise ValueError(
            "Invalid flags: Must specify either MAP_SHARED or MAP_PRIVATE"
        )

    return flags


cdef inline size_t get_page_size() nogil:
    """
    Return the OS page size in bytes.

    Returns
    -------
    size_t
        Page size as reported by the kernel.  Typical values: 4096 (x86-64),
        16384 (macOS ARM64 / Apple Silicon).

    Notes
    -----
    Calls ``sysconf(_SC_PAGESIZE)`` on POSIX.  Falls back to 4096 when the
    call returns a non-positive value or on Windows where sysconf is
    unavailable.

    Examples
    --------
    cdef size_t page_size = get_page_size()
    cdef size_t aligned_offset = (offset / page_size) * page_size
    """
    cdef long ps = sysconf(_SC_PAGESIZE)
    if ps <= 0:
        return 4096          # fallback; also covers Windows
    return <size_t>ps


cdef inline bint is_page_aligned(size_t value) nogil:
    """
    Check if value is page-aligned.

    Parameters
    ----------
    value : size_t
        Value to check (typically an offset)

    Returns
    -------
    bint
        True if value is aligned to page boundary, False otherwise

    Notes
    -----
    A value is page-aligned if it's a multiple of the page size.
    This is required for the offset parameter of mmap.

    Examples
    --------
    is_page_aligned(0)      # True
    is_page_aligned(4096)   # True (assuming 4K pages)
    is_page_aligned(4097)   # False
    """
    cdef size_t page_size = get_page_size()
    return (value % page_size) == 0


cdef inline size_t align_to_page(size_t value) nogil:
    """
    Round down value to nearest page boundary.

    Parameters
    ----------
    value : size_t
        Value to align

    Returns
    -------
    size_t
        Value rounded down to page boundary

    Notes
    -----
    This is useful for ensuring offsets are page-aligned.
    Always rounds DOWN, never up.

    Examples
    --------
    align_to_page(0)      # 0
    align_to_page(100)    # 0 (assuming 4K pages)
    align_to_page(4096)   # 4096
    align_to_page(4097)   # 4096
    """
    cdef size_t page_size = get_page_size()
    return (value / page_size) * page_size
