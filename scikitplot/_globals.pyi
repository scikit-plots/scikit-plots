# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This stub file covers all public and internal symbols in scikitplot._globals.
# Upstream reference:
#   https://github.com/numpy/numpy/blob/main/numpy/_globals.pyi

import enum
from typing import Final, final

__all__: list[str]

# ---------------------------------------------------------------------------
# _CopyMode
# ---------------------------------------------------------------------------

@final
class _CopyMode(enum.Enum):
    """Enumeration of array-copy modes."""

    ALWAYS: bool
    NEVER: bool
    IF_NEEDED: int

    def __bool__(self) -> bool: ...

# ---------------------------------------------------------------------------
# SingletonBase
# ---------------------------------------------------------------------------

class SingletonBase:
    """Base class implementing the singleton pattern."""

    _instance: SingletonBase | None

    def __init_subclass__(cls, **kwargs: object) -> None: ...
    def __new__(cls) -> SingletonBase: ...
    def __reduce__(self) -> tuple[type[SingletonBase], tuple[()]]: ...

# ---------------------------------------------------------------------------
# _DefaultType / _Default
# ---------------------------------------------------------------------------

@final
class _DefaultType(SingletonBase):
    """Singleton sentinel: use the default value."""

    def __repr__(self) -> str: ...

_Default: Final[_DefaultType]

# ---------------------------------------------------------------------------
# _DeprecatedType / _Deprecated
# ---------------------------------------------------------------------------

@final
class _DeprecatedType(SingletonBase):
    """Singleton sentinel: value or feature is deprecated."""

    def __repr__(self) -> str: ...

_Deprecated: Final[_DeprecatedType]

# ---------------------------------------------------------------------------
# _NoValueType / _NoValue
# ---------------------------------------------------------------------------

@final
class _NoValueType(SingletonBase):
    """Singleton sentinel: no user-supplied value."""

    def __repr__(self) -> str: ...

_NoValue: Final[_NoValueType]
