from __future__ import annotations

from . import (
    _similarity,
)

# --- Similarity index ---
from ._similarity import *  # noqa: F403

__all__ = []
__all__ += _similarity.__all__
