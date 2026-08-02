from __future__ import annotations

from . import (
    _backends,
    _similarity,
)

# --- Similarity index ---
from ._backends import *  # noqa: F403
from ._similarity import *  # noqa: F403

__all__ = []
__all__ += _backends.__all__
__all__ += _similarity.__all__
