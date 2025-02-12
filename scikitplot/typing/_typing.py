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

# Allows for the creation of enumerated constants
# Enum values are immutable after definition.

# _HT = TypeVar("_HT", bound=Hashable)
# DT = TypeVar("DT", bound=np.generic)


__all__ = []
