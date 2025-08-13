# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# from typing_extensions import LiteralString
from typing import Final, LiteralString

__all__ = (
    "__git_hash__",
    "__version__",
    "_version",
    "full_version",
    "git_revision",
    "release",
    "short_git_revision",
    "short_version",
    "version",
)
__version__: Final[LiteralString]
version: Final[LiteralString]
full_version: Final[LiteralString]

_version: Final[LiteralString]
short_version: Final[LiteralString]

__git_hash__: Final[LiteralString]
git_revision: Final[LiteralString]
short_git_revision: Final[LiteralString]

release: Final[bool]
