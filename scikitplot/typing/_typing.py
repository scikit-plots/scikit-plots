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
from __future__ import annotations

import abc
import numbers

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Hashable, Sequence
from datetime import date, datetime, timedelta
from fractions import Fraction

import numpy as np
from numpy import ndarray  # TODO use ArrayLike?
from pandas import Series, Index, Timestamp, Timedelta
from matplotlib.colors import Colormap, Normalize

import numpy.typing as np_typing
from numpy.typing import NDArray

from types import GenericAlias, NoneType, ModuleType
from types import DynamicClassAttribute, LambdaType

from typing import TYPE_CHECKING
from typing import TypeVar, TypeAlias, Generic, Any
from typing import Callable, Literal, Optional, Union
from typing import List, Tuple, Dict, NamedTuple, TypedDict

# Allows for the creation of enumerated constants
# Enum values are immutable after definition. 
from enum import Enum

_HT = TypeVar("_HT", bound=Hashable)
DT = TypeVar("DT", bound=np.generic)

from .._globals import (
  _DefaultType as Default,
  _NoValueType,
  _DeprecatedType as Deprecated,
)