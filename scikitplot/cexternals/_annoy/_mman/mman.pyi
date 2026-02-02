# scikitplot/cexternals/_annoy/_mman/mman.pyi
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

"""
Type stubs for mman module (Windows memory mapping).

This file provides type hints for Python type checkers (mypy, pyright, etc.)
and IDE autocompletion. It describes the public Python API exposed by the
Cython extension module.

Design Principles
-----------------
- Follow PEP 484 (Type Hints) and PEP 561 (Distributing Type Information)
- Mirror the exact public API exposed in mman.pyx
- Use precise types (no 'Any' unless unavoidable)
- Document all public classes and functions with docstrings
- Keep in sync with mman.pyx implementation

Best Practices
--------------
- Use `int` for C integer types in Python interface
- Use `Final` for constants that should not be reassigned
- Use `@overload` if methods have multiple signatures
- Keep stub file minimal - detailed docs go in .pyx docstrings

Notes for Maintainers
---------------------
- This file is NOT imported at runtime; it's only for static analysis
- Changes to public API in .pyx MUST be reflected here
- Test with `mypy --strict` and `pyright` before committing
"""

from typing import Final, Optional, Union, Literal
from types import TracebackType

# ===========================================================================
# Module Version
# ===========================================================================

__version__: Final[str]

# ===========================================================================
# Protection Flags
# ===========================================================================

PROT_NONE: Final[int]
PROT_READ: Final[int]
PROT_WRITE: Final[int]
PROT_EXEC: Final[int]

# ===========================================================================
# Mapping Flags
# ===========================================================================

MAP_SHARED: Final[int]
MAP_PRIVATE: Final[int]
MAP_ANONYMOUS: Final[int]
MAP_ANON: Final[int]
MAP_FIXED: Final[int]

# ===========================================================================
# Sync Flags
# ===========================================================================

MS_ASYNC: Final[int]
MS_SYNC: Final[int]
MS_INVALIDATE: Final[int]

# ===========================================================================
# Exception Classes
# ===========================================================================

class MMapError(OSError):
    """Base exception for memory mapping errors."""
    ...

class MMapAllocationError(MMapError):
    """Raised when memory mapping allocation fails."""
    ...

class MMapInvalidParameterError(MMapError):
    """Raised when invalid parameters are provided."""
    ...

# ===========================================================================
# MemoryMap Class
# ===========================================================================

class MemoryMap:
    """
    Memory-mapped region with automatic resource management.

    This class wraps a memory-mapped region and provides a Pythonic interface
    with context manager support for automatic cleanup.

    Attributes
    ----------
    addr : int
        Memory address of the mapped region (read-only)
    size : int
        Size of the mapped region in bytes (read-only)
    is_valid : bool
        Whether the mapping is still valid (read-only)

    Notes
    -----
    - Always use factory methods (create_*) to create instances
    - Use as context manager for automatic cleanup
    - Accessing closed mapping raises ValueError

    Examples
    --------
    >>> with MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE) as m:
    ...     m.write(b"test")
    ...     data = m.read(4)
    """

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
        ...

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
        ...

    @property
    def is_valid(self) -> bool:
        """
        Check if mapping is still valid.

        Returns
        -------
        bool
            True if valid, False if closed
        """
        ...

    @staticmethod
    def create_anonymous(
        size: int,
        prot: int = PROT_READ | PROT_WRITE,
        flags: int = MAP_PRIVATE,
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
        """
        ...

    @staticmethod
    def create_file_mapping(
        fd: int,
        offset: int,
        size: int,
        prot: int = PROT_READ,
        flags: int = MAP_PRIVATE,
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
        """
        ...

    def close(self) -> None:
        """
        Close the memory mapping.

        Unmaps the region and releases resources.

        Raises
        ------
        ValueError
            If mapping is already closed
        MMapError
            If munmap() fails

        Notes
        -----
        - After closing, the mapping cannot be used
        - Called automatically by __dealloc__ or context manager
        - Safe to call multiple times (idempotent)
        """
        ...

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
            If mapping is closed or parameters are invalid
        """
        ...

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
        """
        ...

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
        MMapError
            If mprotect() fails
        """
        ...

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
        """
        ...

    def __enter__(self) -> MemoryMap:
        """Context manager entry."""
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit - automatically closes mapping."""
        ...

    def __repr__(self) -> str:
        """String representation."""
        ...

    def __str__(self) -> str:
        """User-friendly string representation."""
        ...

# ===========================================================================
# Module-Level Functions
# ===========================================================================

def mmap_region(
    size: int,
    prot: int = PROT_READ | PROT_WRITE,
    flags: int = MAP_PRIVATE | MAP_ANONYMOUS,
    fd: int = -1,
    offset: int = 0,
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

    Raises
    ------
    ValueError
        If parameters are invalid
    MMapAllocationError
        If mapping fails
    """
    ...
