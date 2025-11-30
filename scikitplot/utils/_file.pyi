# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from typing import Sequence, overload

# Compatibility shim for `TypeAlias` in Python < 3.10
try:
    # Python 3.10+ — native support
    from typing import TypeAlias
except ImportError:
    try:
        # Fallback for older Python using typing_extensions (must be installed)
        from typing_extensions import (
            TypeAlias,
        )
    except ImportError:
        # Final fallback: dummy placeholder (used only for type hints)
        TypeAlias = object

import numpy as np

SUFFIXES: list[str]

Scalar: TypeAlias = ...

def humansize(nbytes: Scalar, suffixes: Sequence[str] | None = None) -> str: ...
@overload
def humansize_vector(values: Scalar, suffixes: Sequence[str] | None = None) -> str: ...
@overload
def humansize_vector(
    values: Sequence[Scalar], suffixes: Sequence[str] | None = None
) -> np.ndarray: ...
@overload
def humansize_vector(
    values: np.ndarray, suffixes: Sequence[str] | None = None
) -> np.ndarray: ...
def humansize_vector(
    values: Scalar | Sequence[Scalar] | np.ndarray,
    suffixes: Sequence[str] | None = None,
) -> str | np.ndarray: ...
