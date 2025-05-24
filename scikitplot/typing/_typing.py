"""
The typing module: Support for gradual typing as defined by PEP 484 and subsequent PEPs.

This module contains Type aliases which are useful for scikitplot and potentially
downstream libraries.

.. admonition:: Provisional status of typing

    The ``typing`` module and type stub files are considered provisional and may change
    at any time without a deprecation period.

Among other things, the module includes the following:
* Generic, Protocol, and internal machinery to support generic aliases.
  All subscripted types like X[int], Union[int, str] are generic aliases.
* Various "special forms" that have unique meanings in type annotations:
  NoReturn, Never, ClassVar, Self, Concatenate, Unpack, and others.
* Classes whose instances can be type arguments to generic classes and functions:
  TypeVar, ParamSpec, TypeVarTuple.
* Public helper functions: get_type_hints, overload, cast, final, and others.
* Several protocols to support duck-typing:
  SupportsFloat, SupportsIndex, SupportsAbs, and others.
* Special types: NewType, NamedTuple, TypedDict.
* Deprecated aliases for builtin types and collections.abc ABCs.

Any name not present in __all__ is an implementation detail
that may be changed without notice. Use at your own risk!
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: F401

from __future__ import annotations

from typing import TYPE_CHECKING

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
