# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/_utils/

"""
A set of methods retained from np.compat module that
are still used across codebase.
"""

__all__ = ["asunicode", "asbytes"]


def asunicode(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')
