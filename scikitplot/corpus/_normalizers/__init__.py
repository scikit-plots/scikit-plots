"""
scikitplot.corpus._normalizers
================================
Text normalisation pipeline for :class:`~scikitplot.corpus._schema.CorpusDocument`.

Each normaliser receives a ``CorpusDocument`` and returns a new instance
(via ``doc.replace()``) with the normalised text written into
``normalized_text``. The original ``text`` field is always preserved.
"""  # noqa: D205, D400

from __future__ import annotations

from ._normalizer import (
    DedupLinesNormalizer,
    HTMLStripNormalizer,
    LanguageDetectionNormalizer,
    LowercaseNormalizer,
    NormalizationPipeline,
    NormalizerBase,
    UnicodeNormalizer,
    WhitespaceNormalizer,
)
from ._text_normalizer import (
    NormalizerConfig,
    TextNormalizer,
    normalize_text,
)

__all__ = [  # noqa: RUF022
    "DedupLinesNormalizer",
    "HTMLStripNormalizer",
    "LanguageDetectionNormalizer",
    "LowercaseNormalizer",
    "NormalizationPipeline",
    "NormalizerBase",
    "UnicodeNormalizer",
    "WhitespaceNormalizer",
    # Text normalisation
    "NormalizerConfig",
    "TextNormalizer",
    "normalize_text",
]
