# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Copyright (c) 2023, NumPy Developers.
#
# from __future__ import annotations: makes annotations lazy (PEP 563/749).
# Safe here: no runtime annotation evaluation in this file.
# https://github.com/numpy/meson
from __future__ import annotations

from typing import TYPE_CHECKING

from .module import Module

if TYPE_CHECKING:
    from ...interpreter import Interpreter  # type: ignore[]


def initialize(interpreter: "Interpreter") -> Module:
    return Module()
