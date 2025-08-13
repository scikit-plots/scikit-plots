# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/exceptions.pyi

from typing import overload

__all__ = [
    "ComplexWarning",
    "VisibleDeprecationWarning",
    "ModuleDeprecationWarning",
    "TooHardError",
    "AxisError",
    "DTypePromotionError",
]

class ComplexWarning(RuntimeWarning): ...
class ModuleDeprecationWarning(DeprecationWarning): ...
class VisibleDeprecationWarning(UserWarning): ...
class RankWarning(RuntimeWarning): ...
class TooHardError(RuntimeError): ...
class DTypePromotionError(TypeError): ...

class AxisError(ValueError, IndexError):
    axis: int | None
    ndim: int | None
    @overload
    def __init__(self, axis: str, ndim: None = ..., msg_prefix: None = ...) -> None: ...
    @overload
    def __init__(self, axis: int, ndim: int, msg_prefix: str | None = ...) -> None: ...
