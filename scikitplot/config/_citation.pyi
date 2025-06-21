# pylint: skip-file
# ruff: noqa: PGH004
# ruff: noqa
# flake8: noqa
# type: ignore
# mypy: ignore-errors

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Only imports when type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Heavy import, only for type checking
    # Only imports when type checking, not at runtime
    from typing import Final, LiteralString
    from typing_extensions import LiteralString

CITATION = Final[LiteralString]

# Set the bibtex entry to the article referenced in CITATION.
def _get_bibtex(
    preferred_type: str = "software",
) -> str: ...

__bibtex__: Final[LiteralString]
__citation__: Final[LiteralString]
