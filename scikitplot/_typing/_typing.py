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
from builtins import type as type_t
from typing import TYPE_CHECKING

# npt.ArrayLike list, tuple, np.ndarray
# npt.NDArray np.ndarray Series (vector, matrix, tensor)
# np.generic NumPy skaler (np.float64, np.int32 vb.)
# sp.spmatrix sparse matrix (csr, csc, coup, vs.)
import matplotlib as mpl
import matplotlib.typing as mpt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas._typing as pdt

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles


# <-- no longer imported at runtime
if TYPE_CHECKING:
    # ⚠️ "list[str]" use quotes to avoid runtime eval
    if sys.version_info >= (3, 9):
        StrList = list[str]  # PEP 585, Python 3.9+
    else:
        from typing import List  # Python <3.9

        StrList = List[str]  # typing.List, Python <3.9

    # ⚠️ "float | None" use quotes to avoid runtime eval
    if sys.version_info >= (3, 10):
        FloatOrNone = float | None  # PEP 604 syntax >= py3.10
    else:
        from typing import Optional  # Python <3.10

        # equivalent to Union[float, None]
        FloatOrNone = Optional[float]  # PEP 484 syntax < py3.10

    from collections.abc import (
        Coroutine,
        # Callable,
        Generator,
        Hashable,
        Iterable,
        Mapping,
        MutableMapping,
        # Sequence,
        Sequence,
    )
    from datetime import (
        date,
        datetime,
        timedelta,
        tzinfo,
    )
    from enum import Enum, EnumType
    from os import PathLike
    from typing import (  # noqa: F401
        IO,
        Any,
        Callable,
        ClassVar,
        Literal,
        ModuleType,
        Optional,
        ParamSpec,
        Protocol,
        SupportsIndex,
        # TypeAlias,  # 3.10+
        TypeVar,
        Union,
        overload,
    )

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

    from typing_extensions import NotRequired

    # to maintain type information across generic functions and parametrization
    T = TypeVar("T")
    TypeT = TypeVar("TypeT", bound=type)

    P = ParamSpec("P")

    RandomState: TypeAlias = (
        int
        | np.ndarray
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
    )

    Shape: TypeAlias = tuple[int, ...]
    Suffixes: TypeAlias = Sequence[str | None]
    Ordered: TypeAlias = bool | None

    HashableT = TypeVar("HashableT", bound=Hashable)
    HashableT2 = TypeVar("HashableT2", bound=Hashable)
    MutableMappingT = TypeVar("MutableMappingT", bound=MutableMapping)

    ScalarLike_co: TypeAlias = int | float | complex | str | bytes | np.generic

    # numpy compatible types
    NumpyValueArrayLike: TypeAlias = ScalarLike_co | npt.ArrayLike
    NumpySorter: TypeAlias = npt._ArrayLikeInt_co | None

    # array-like

    ArrayLike: TypeAlias = Union["pdt.ExtensionArray", np.ndarray]
    ArrayLikeT = TypeVar("ArrayLikeT", "pdt.ExtensionArray", np.ndarray)
    AnyArrayLike: TypeAlias = Union[ArrayLike, "pdt.Index", "pdt.Series"]
    TimeArrayLike: TypeAlias = Union["pdt.DatetimeArray", "pdt.TimedeltaArray"]

    # list-like

    # from https://github.com/hauntsaninja/useful_types
    # includes Sequence-like objects but excludes str and bytes
    _T_co = TypeVar("_T_co", covariant=True)  # noqa: PYI018
    ListLike: TypeAlias = AnyArrayLike | pdt.SequenceNotStr | range

    # scalars

    PythonScalar: TypeAlias = str | float | bool
    DatetimeLikeScalar: TypeAlias = Union[
        "pdt.Period", "pdt.Timestamp", "pdt.Timedelta"
    ]

    Timezone: TypeAlias = str | tzinfo
    TimeUnit: TypeAlias = Literal["s", "ms", "us", "ns"]

    # dtypes
    NpDtype: TypeAlias = str | np.dtype | type[str | complex | bool | object]
    Dtype: TypeAlias = Union["pdt.ExtensionDtype", NpDtype]
    AstypeArg: TypeAlias = Union["pdt.ExtensionDtype", npt.DTypeLike]
    # DtypeArg specifies all allowable dtypes in a functions its dtype argument
    DtypeArg: TypeAlias = Dtype | Mapping[Hashable, Dtype]
    DtypeObj: TypeAlias = Union[np.dtype, "pdt.ExtensionDtype"]

    # filenames and file-like-objects
    AnyStr_co = TypeVar("AnyStr_co", str, bytes, covariant=True)
    AnyStr_contra = TypeVar("AnyStr_contra", str, bytes, contravariant=True)

    FilePath: TypeAlias = str | PathLike[str]

    # for arbitrary kwargs passed during reading/writing files
    StorageOptions: TypeAlias = dict[str, Any] | None

    # compression keywords and compression
    CompressionDict: TypeAlias = dict[str, Any]
    CompressionOptions: TypeAlias = (
        Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"]
        | CompressionDict
        | None
    )
    ParquetCompressionOptions: TypeAlias = (
        Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
    )

    # read_html flavors
    HTMLFlavors: TypeAlias = Literal["lxml", "html5lib", "bs4"]

    # join
    JoinHow: TypeAlias = Literal["left", "right", "inner", "outer"]
    JoinValidate: TypeAlias = Literal[
        "one_to_one",
        "1:1",
        "one_to_many",
        "1:m",
        "many_to_one",
        "m:1",
        "many_to_many",
        "m:m",
    ]

    # plotting
    PlottingOrientation: TypeAlias = Literal["horizontal", "vertical"]

    # maintain the sub-type of any hashable sequence
    SequenceT = TypeVar("SequenceT", bound=Sequence[Hashable])

    SliceType: TypeAlias = Hashable | None

    PythonFuncType: TypeAlias = Callable[[Any], Any]

    ## A generic type variable for class decorator
    from .._decorates import BaseDecorator

    # used in decorators to preserve the signature of the function it decorates
    # see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
    # FuncType: TypeAlias = Callable[..., Any]
    # F = TypeVar("F", bound=FuncType)

    ## A type representing a class, function decorator
    ClsDec = TypeVar("ClsDec", bound="BaseDecorator")
    FuncDec = TypeVar("FuncDec", bound="Callable[[F], F]")
    DecoratorLike = TypeVar("DecoratorLike", bound="Union[F, FuncDec, ClsDec]")

    ## Generic callable type
    F = TypeVar("F", bound="Callable[..., Any]")
    ## Hashable key type (e.g., dict keys, cache identifiers)
    HT = TypeVar("HT", bound="Hashable")

# Allows for the creation of enumerated constants
# Enum values are immutable after definition.
# __all__ = ["type_t"]
