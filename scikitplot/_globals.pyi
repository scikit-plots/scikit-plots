# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/_globals.pyi

__all__ = ["_CopyMode", "_NoValue"]

import enum
from typing import Final, final

@final
class _CopyMode(enum.Enum):
    ALWAYS = True
    NEVER = False
    IF_NEEDED = 2

    def __bool__(self, /) -> bool: ...

@final
class _NoValueType: ...

_NoValue: Final[_NoValueType] = ...
