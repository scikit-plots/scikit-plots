"""NLP enrichment components for the corpus pipeline."""

from __future__ import annotations

from . import (
    _nlp_enricher,
)
from ._nlp_enricher import *  # noqa: F403

__all__ = []
__all__ += _nlp_enricher.__all__
