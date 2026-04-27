# pylint: skip-file
# ruff: noqa: PGH004
# ruff: noqa
# flake8: noqa
# type: ignore

# Only imports when type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final, LiteralString
    from typing_extensions import LiteralString

__version__: Final[LiteralString]
