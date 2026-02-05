# scikitplot/memmap/_memmap/mem_map.pyi
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

"""
Type stubs for the mman Cython extension module.

Mirrors the public API exposed by ``mman.pyx`` exactly.  Every constant,
exception class, method, and property declared here must stay in lock-step
with the implementation.

Design Principles
-----------------
- PEP 484 / PEP 561 compliant.
- ``Final`` on every module-level constant.
- No ``Any`` unless truly unavoidable.
- Detailed docstrings live in ``.pyx``; stubs are intentionally terse.

Notes for Maintainers
---------------------
- This file is **never** imported at runtime — static analysis only.
- After any API change in ``.pyx`` run ``mypy --strict`` and ``pyright``
  against your consumer code before shipping.
"""

from typing import Final, Optional
from types import TracebackType

import numpy as np

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

MAP_FILE: Final[int]
MAP_SHARED: Final[int]
MAP_PRIVATE: Final[int]
MAP_TYPE: Final[int]
MAP_FIXED: Final[int]
MAP_ANONYMOUS: Final[int]
MAP_ANON: Final[int]

# ===========================================================================
# Sync Flags
# ===========================================================================

MS_ASYNC: Final[int]
MS_SYNC: Final[int]
MS_INVALIDATE: Final[int]

# ===========================================================================
# Windows Compatibility
# ===========================================================================

FILE_MAP_EXECUTE: Final[int]

# ===========================================================================
# Exception Classes
# ===========================================================================

class MMapError(OSError):
    """Base exception for memory mapping errors."""
    ...

class MMapAllocationError(MMapError):
    """Raised when mmap() fails to allocate."""
    ...

class MMapInvalidParameterError(MMapError):
    """Raised when caller supplies an invalid parameter."""
    ...

# ===========================================================================
# MemoryMap Class
# ===========================================================================

class MemoryMap:
    """
    Memory-mapped region with automatic resource management.

    Always construct via the static factory methods
    :py:meth:`create_anonymous` or :py:meth:`create_file_mapping`.
    Use as a context manager for deterministic cleanup.

    Attributes
    ----------
    addr : int
        Virtual address of the mapped region (read-only).
    size : int
        Size of the mapped region in bytes (read-only).
    is_valid : bool
        ``True`` while the mapping has not been closed (read-only).
    page_size : int
        OS page size in bytes (read-only).
    """

    # ----- read-only properties -----

    @property
    def addr(self) -> int: ...

    @property
    def size(self) -> int: ...

    @property
    def is_valid(self) -> bool: ...

    @property
    def page_size(self) -> int: ...

    # ----- factory methods -----

    @staticmethod
    def create_anonymous(
        size: int,
        prot: int = ...,
        flags: int = ...,
    ) -> MemoryMap:
        """
        Create an anonymous (RAM-only) mapping.

        Parameters
        ----------
        size : int
            Mapping size in bytes.  Must be > 0.
        prot : int, optional
            Protection flags.  Default ``PROT_READ | PROT_WRITE``.
        flags : int, optional
            Mapping flags.  Default ``MAP_PRIVATE``.

        Returns
        -------
        MemoryMap

        Raises
        ------
        ValueError
            *size* <= 0 or flags invalid.
        MMapAllocationError
            Kernel refused the mapping.
        """
        ...

    @staticmethod
    def create_file_mapping(
        fd: int,
        offset: int,
        size: int,
        prot: int = ...,
        flags: int = ...,
    ) -> MemoryMap:
        """
        Create a file-backed mapping.

        Parameters
        ----------
        fd : int
            Open file descriptor.  Must be >= 0.
        offset : int
            Byte offset into the file.  Must be page-aligned and >= 0.
        size : int
            Mapping size in bytes.  Must be > 0.
        prot : int, optional
            Protection flags.  Default ``PROT_READ``.
        flags : int, optional
            Mapping flags.  Default ``MAP_PRIVATE``.

        Returns
        -------
        MemoryMap

        Raises
        ------
        ValueError
            Invalid *fd*, *offset*, *size*, or *flags*.
        MMapAllocationError
            Kernel refused the mapping.
        """
        ...

    # ----- I/O -----

    def read(self, size: int, offset: int = 0) -> bytes:
        """Read *size* bytes starting at *offset*."""
        ...

    def write(self, data: bytes, offset: int = 0) -> int:
        """Write *data* at *offset*; return number of bytes written."""
        ...

    # ----- protection & sync -----

    def mprotect(self, prot: int) -> None:
        """Change memory-protection flags for the entire mapping."""
        ...

    def msync(self, flags: int = ...) -> None:
        """
        Flush dirty pages to the backing file.

        Parameters
        ----------
        flags : int, optional
            ``MS_SYNC`` (default), ``MS_ASYNC``, or ``MS_INVALIDATE``.
        """
        ...

    # ----- page locking -----

    def mlock(self) -> None:
        """
        Lock all mapped pages in physical RAM (no swapping).

        Raises
        ------
        MMapError
            When the kernel rejects the request (e.g. ``RLIMIT_MEMLOCK``
            exceeded on Linux).
        """
        ...

    def munlock(self) -> None:
        """Release page-lock set by :py:meth:`mlock`."""
        ...

    # ----- NumPy integration -----

    def as_numpy_array(self, dtype: "Optional[np.dtype]" = None) -> "np.ndarray":
        """
        Zero-copy NumPy view of the mapped memory.

        Parameters
        ----------
        dtype : numpy.dtype or None, optional
            Element type.  Defaults to ``numpy.uint8``.
            ``dtype.itemsize`` must evenly divide :py:attr:`size`.

        Returns
        -------
        numpy.ndarray
            1-D array sharing the mapped buffer.  ``WRITEABLE`` is set
            only when the mapping carries ``PROT_WRITE``.

        Raises
        ------
        ValueError
            Mapping closed, or *dtype* size incompatible.
        ImportError
            NumPy not available.
        """
        ...

    # ----- lifecycle -----

    def close(self) -> None:
        """
        Unmap and release the region.

        Idempotent: calling ``close()`` on an already-closed mapping is
        a silent no-op.
        """
        ...

    # ----- context-manager -----

    def __enter__(self) -> MemoryMap: ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None: ...

    # ----- string representations -----

    def __repr__(self) -> str: ...
    def __str__(self)  -> str: ...

# ===========================================================================
# Module-Level Convenience Functions
# ===========================================================================

def mmap_region(
    size: int,
    prot: int = ...,
    flags: int = ...,
    fd: int = ...,
    offset: int = ...,
) -> MemoryMap:
    """
    Create an anonymous or file-backed mapping in one call.

    Parameters
    ----------
    size : int
        Mapping size in bytes.
    prot : int, optional
        Protection flags.  Default ``PROT_READ | PROT_WRITE``.
    flags : int, optional
        Mapping flags.  Default ``MAP_PRIVATE | MAP_ANONYMOUS``.
    fd : int, optional
        File descriptor.  Default ``-1`` (anonymous).
    offset : int, optional
        File offset.  Default ``0``.

    Returns
    -------
    MemoryMap

    Raises
    ------
    ValueError
        Invalid parameters.
    MMapAllocationError
        Kernel refused the mapping.
    """
    ...
