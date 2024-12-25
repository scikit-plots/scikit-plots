import os
import sys

from typing import TypeAlias, Final, LiteralString
from typing_extensions import LiteralString

__all__ = (
  '__version__', 'version', 'full_version',
  'short_version',
  '__git_hash__', 'git_revision',
  'short_git_revision',
  'release',
)
version            : Final[LiteralString]
__version__        : Final[LiteralString]
full_version       : Final[LiteralString]
short_version      : Final[LiteralString]

__git_hash__       : Final[LiteralString]
git_revision       : Final[LiteralString]
short_git_revision : Final[LiteralString]

release            : Final[bool]