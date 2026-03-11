"""
scikitplot.corpus._metadata
============================
Corpus-level metadata management: collection descriptors, statistics,
and provenance manifests.

:class:`CollectionManifest`
    Describes a named corpus collection: title, author, source info,
    expected file count, and per-file provenance overrides.

:class:`CorpusStats`
    Aggregate statistics computed over a list of
    :class:`~scikitplot.corpus._schema.CorpusDocument` instances.
    Document count, token counts, language distribution, section type
    distribution, source type distribution, date range.

:func:`compute_stats`
    Pure function that computes :class:`CorpusStats` from a document list.

:func:`provenance_from_filename`
    Heuristic provenance extraction from a source filename
    (author_title_year pattern used by many Project Gutenberg exports).
"""  # noqa: D205, D400

from __future__ import annotations

from ._metadata import (
    CollectionManifest,
    CorpusStats,
    compute_stats,
    provenance_from_filename,
)

__all__ = [
    "CollectionManifest",
    "CorpusStats",
    "compute_stats",
    "provenance_from_filename",
]
