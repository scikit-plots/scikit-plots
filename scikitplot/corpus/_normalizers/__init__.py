"""
scikitplot.corpus._normalizers
================================
Text normalisation pipeline for :class:`~scikitplot.corpus._schema.CorpusDocument`.

Each normaliser receives a ``CorpusDocument`` and returns a new instance
(via ``doc.replace()``) with the normalised text written into
``normalized_text``. The original ``text`` field is always preserved.
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _normalizer,
    _text_normalizer,
)
from ._normalizer import *  # noqa: F403
from ._text_normalizer import *  # noqa: F403

__all__ = []
__all__ += _normalizer.__all__
__all__ += _text_normalizer.__all__
