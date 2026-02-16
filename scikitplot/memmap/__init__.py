# scikitplot/memmap/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# This is intentional:
# - numpy.memmap → memory abstraction
# - scipy.io → data transfer & parsing

"""
MemMap: file-backed or anonymous memory mapping.

When a file path is provided, the mapping is file-backed and reflects the
contents of the underlying file. When no file is provided, the module creates
an anonymous memory mapping backed by RAM only.

This module provides a high-level abstraction for file-backed memory mapping,
allowing data on disk to be accessed as memory without loading the entire file
into RAM. The design follows the same conceptual model as ``numpy.memmap``:
OS-agnostic, intent-based, and providing safer, on-demand access for Python
users.

While this module is implemented on top of Annoy's low-level ``mman`` layer,
which operates on raw memory and offsets for custom binary index formats, all
operating-system details are hidden from the public API. Raw pointers and
platform-specific constructs are not exposed to user code.

This module is not a file I/O or parsing interface and does not eagerly copy
file contents into memory. It is intended for efficient random access to large,
stable binary data stored on disk with explicit and deterministic lifetime
management.
"""

from __future__ import annotations

from . import _memmap
from ._memmap import *  # noqa: F403

__all__ = []
__all__ += _memmap.__all__
