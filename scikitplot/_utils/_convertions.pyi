# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/_utils/

__all__ = ["asbytes", "asunicode"]

def asunicode(s: bytes | str) -> str: ...
def asbytes(s: bytes | str) -> str: ...
