from typing import Final, LiteralString

# from typing_extensions import LiteralString

__all__ = (
    "__git_hash__",
    "__version__",
    "full_version",
    "git_revision",
    "release",
    "short_git_revision",
    "short_version",
    "version",
)
version: Final[LiteralString]
__version__: Final[LiteralString]
full_version: Final[LiteralString]
short_version: Final[LiteralString]

__git_hash__: Final[LiteralString]
git_revision: Final[LiteralString]
short_git_revision: Final[LiteralString]

release: Final[bool]
