
from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar

SKPLT_API_NAME = 'scikitplot'

# class ExportType(Protocol):

#   def __call__(
#       self,
#       *v2: str,
#       v1: Optional[Sequence[str]] = None,
#   ) -> api_export:
#     ...


# sp_export: ExportType = functools.partial(
#     api_export, api_name=SKPLT_API_NAME
# )