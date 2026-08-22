"""NLP and dependency-free enrichment components for the corpus pipeline."""

from __future__ import annotations

from . import (
    _nlp_enricher,
    _simple,
)
from ._nlp_enricher import *  # noqa: F403
from ._simple import *  # noqa: F403

__all__ = []
__all__ += _nlp_enricher.__all__
__all__ += _simple.__all__
