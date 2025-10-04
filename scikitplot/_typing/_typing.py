# ruff: noqa: F401

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Typing Notes for Developers.

The ``typing`` module provides support for gradual typing as defined by PEP 484
and subsequent PEPs. It is central to static type checking in Python.

This module contains type aliases and helpers which are useful for scientific,
ML libraries (e.g. scikit-plot), and downstream projects.

.. admonition:: Provisional status of typing

    The ``typing`` module and type stubs have historically been considered
    provisional. While stability has greatly improved in recent releases,
    details may still change across versions without deprecation warnings.

----------------------------------------------------------------------
🔑 Key Evolution by Python Version
----------------------------------------------------------------------

- Python 3.7 - 3.9:
    * Must use ``typing.Optional`` or ``typing.Union``.
    * Example:
        from typing import Optional, Union
        def f(x: Optional[float] = None) -> Union[int, None]:
            ...
    * PEP 604 syntax (``float | None``) is **NOT supported**.

- Python 3.10 - 3.11:
    * PEP 604 introduces the ``|`` operator as shorthand for Union.
    * Example:
        def f(x: float | None = None) -> int | None:
            ...
    * ``Optional`` and ``Union`` still work, but ``|`` is now preferred.

- Python 3.12:
    * ``Optional[X]`` is fully equivalent to ``X | None``.
    * Advanced typing features added: ``TypeVarTuple``, ``Unpack``.

- Python 3.13 - 3.14 (preview):
    * ``|`` syntax is the de facto standard.
    * Legacy ``Optional`` / ``Union`` remain for backwards compatibility.
    * New features like ``TypeAliasType`` and ``TypeIs`` expand expressiveness.

✅ Guidelines:
    * If supporting < 3.10 → stick with ``typing.Optional`` / ``typing.Union``.
    * If supporting only 3.10+ → prefer PEP 604 ``|`` syntax.
    * For libraries with broad user bases → default to ``typing.Optional``.
    * Always use ``from __future__ import annotations`` (3.7+) to defer evaluation.

----------------------------------------------------------------------
📦 What the typing module includes
----------------------------------------------------------------------

- **Generic machinery**:
  * ``Generic`` and ``Protocol`` to define type-safe interfaces.
  * Internal support for *generic aliases* like ``list[int]`` or ``dict[str, float]``.

- **Special forms**:
  * Constructs with unique meanings: ``NoReturn``, ``Never``, ``ClassVar``,
    ``Self``, ``Concatenate``, ``Unpack`` and others.

- **Type variables**:
  * ``TypeVar`` - parameterize generics with constraints.
  * ``ParamSpec`` - capture callable signatures.
  * ``TypeVarTuple`` - variadic type variables.

- **Helper functions**:
  * ``get_type_hints``, ``overload``, ``cast``, ``final``, ``Literal``.

- **Duck-typing protocols**:
  * ``SupportsFloat``, ``SupportsIndex``, ``SupportsAbs``, etc.

- **Specialized constructs**:
  * ``NewType``, ``NamedTuple``, ``TypedDict``.

- **Deprecated aliases**:
  * Older shorthands for built-in and ``collections.abc`` types, retained for compatibility.

⚠️ Note:
    Any symbol not listed in ``typing.__all__`` is considered an implementation
    detail and may change without notice. Use such names at your own risk.

----------------------------------------------------------------------
📚 References
----------------------------------------------------------------------
- PEP 484: Type Hints
- PEP 604: Union types as X | Y
- PEP 585: Builtin collection generics (``list[int]`` instead of ``List[int]``)
- PEP 646: Variadic generics
- Official docs: https://docs.python.org/3/library/typing.html
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

# npt.ArrayLike list, tuple, np.ndarray
# npt.NDArray np.ndarray Series (vector, matrix, tensor)
# np.generic NumPy skaler (np.float64, np.int32 vb.)
# sp.spmatrix sparse matrix (csr, csc, coup, vs.)
import matplotlib.typing as mpt  # Typing support  # noqa: F401
import numpy.typing as npt  # Typing support  # noqa: F401
import pandas._typing as pdt  # Typing support  # noqa: F401

if TYPE_CHECKING:
    # ⚠️ "list[str]" use quotes to avoid runtime eval
    if sys.version_info >= (3, 9):
        StrList = list[str]  # PEP 585, Python 3.9+
    else:
        from typing import List  # Python <3.9

        StrList = List[str]  # typing.List, Python <3.9

if TYPE_CHECKING:
    # ⚠️ "float | None" use quotes to avoid runtime eval
    if sys.version_info >= (3, 10):
        FloatOrNone = float | None  # PEP 604 syntax >= py3.10
    else:
        from typing import Optional  # Python <3.10

        # equivalent to Union[float, None]
        FloatOrNone = Optional[float]  # PEP 484 syntax < py3.10

if TYPE_CHECKING:
    from collections.abc import (
        Coroutine,
        # Callable,
        Generator,
        Hashable,
        Iterable,
        # Sequence,
        Sequence,
    )
    from enum import Enum, EnumType
    from typing import (  # noqa: F401
        IO,
        Any,
        Callable,
        ClassVar,
        Literal,
        ModuleType,
        Optional,
        Protocol,
        TypeVar,
        Union,
    )

    from typing_extensions import NotRequired  # type: ignore[reportMissingModuleSource]

    # Compatibility shim for `TypeAlias` in Python < 3.10
    try:
        # Python 3.10+ — native support
        from typing import TypeAlias
    except ImportError:
        try:
            # Fallback for older Python using typing_extensions (must be installed)
            from typing_extensions import (  # type: ignore[reportMissingModuleSource]
                TypeAlias,
            )
        except ImportError:
            # Final fallback: dummy placeholder (used only for type hints)
            TypeAlias = object

    import numpy as np  # type: ignore[reportMissingImports]
    import pandas as pd  # type: ignore[reportMissingImports]
    from numpy.typing import NDArray  # type: ignore[reportMissingImports]

    ## Generic callable type
    F = TypeVar("F", bound="Callable[..., Any]")
    ## Hashable key type (e.g., dict keys, cache identifiers)
    HT = TypeVar("HT", bound="Hashable")

    ## NumPy scalar type (e.g., np.float64, np.int32)
    NPScalar = TypeVar("NPScalar", bound="np.generic")
    ## NumPy ndarray of any shape/dtype
    # NDArray = TypeVar("NDArray", bound=np.ndarray)
    ## Pandas DataFrame type
    PDDataFrame = TypeVar("PDDataFrame", bound=pd.DataFrame)
    ## Scalar: could be a Python scalar or NumPy scalar
    Scalar = Union[int, float, complex, str, bool, np.generic]
    ## Tabular data type: DataFrame, Series, or ndarray
    Tabular = Union[pd.DataFrame, pd.Series, np.ndarray]
    ## Data container (any 1D or 2D sequence or array-like)
    ArrayLike = Union[Sequence[Any], np.ndarray, pd.Series, pd.DataFrame]

    ## A type representing a function decorator
    FuncDec = TypeVar("FuncDec", bound="Callable[[F], F]")
    ## A generic type variable for class decorator
    from .._decorates import BaseDecorator

    ClsDec = TypeVar("ClsDec", bound="BaseDecorator")
    DecoratorLike = TypeVar("DecoratorLike", bound="Union[F, FuncDec, ClsDec]")

# Allows for the creation of enumerated constants
# Enum values are immutable after definition.
# __all__ = []
