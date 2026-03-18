"""
scikitplot.corpus._metadata._metadata
========================================
Corpus metadata types and statistics computation.

Design invariants
-----------------
* :class:`CollectionManifest` is a frozen dataclass — configuration is
  always validated at construction via :meth:`validate`.
* :class:`CorpusStats` is frozen and carries only JSON-safe values
  (no numpy arrays, no CorpusDocument references).
* :func:`compute_stats` is a pure function — same input always produces
  same output; no I/O, no mutation.
* :func:`provenance_from_filename` is a pure function — deterministic,
  no I/O. Returns a ``dict`` that can be passed directly as
  ``source_provenance`` to :class:`~scikitplot.corpus._sources.CorpusSource`.

Python compatibility
--------------------
Python 3.8-3.15. No external dependencies.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence  # noqa: F401

from .._schema import CorpusDocument

logger = logging.getLogger(__name__)


# ===========================================================================
# CollectionManifest
# ===========================================================================


@dataclass(frozen=True)
class CollectionManifest:
    """
    Descriptor for a named corpus collection.

    A ``CollectionManifest`` holds corpus-level provenance metadata
    (author, title, date, language) that is propagated into every
    :class:`~scikitplot.corpus._schema.CorpusDocument` produced from this
    collection.  It optionally carries per-file provenance overrides so
    that individual files within a multi-file corpus can have their own
    metadata.

    Parameters
    ----------
    collection_id : str
        Unique identifier for this collection. Must be non-empty. Used
        as ``CorpusDocument.collection_id`` in all produced documents.
    title : str or None, optional
        Human-readable title of the collection. Default: ``None``.
    author : str or None, optional
        Primary author or editor. Default: ``None``.
    source_date : str or None, optional
        Publication or creation date in ISO 8601 format. Default: ``None``.
    language : str or None, optional
        Default ISO 639-1 language code for all files. Default: ``None``.
    description : str, optional
        Free-text description of the corpus. Default: ``""``.
    source_type : str or None, optional
        Default ``SourceType`` value string for all files. Default: ``None``.
    file_provenance : dict[str, dict], optional
        Per-file provenance overrides. Keys are filenames (basename only);
        values are dicts with any :class:`~scikitplot.corpus._schema.CorpusDocument`
        provenance field names. Override values take precedence over the
        collection-level defaults. Default: ``{}``.
    tags : list[str], optional
        Arbitrary tags for search / filtering. Default: ``[]``.
    expected_file_count : int or None, optional
        Expected number of source files. Used for completeness validation.
        Default: ``None`` (no check).

    Raises
    ------
    ValueError
        If ``collection_id`` is empty or whitespace-only, or if
        ``expected_file_count`` is negative.

    Examples
    --------
    >>> manifest = CollectionManifest(
    ...     collection_id="gutenberg_shakespeare",
    ...     title="The Complete Works of Shakespeare",
    ...     author="Shakespeare, William",
    ...     source_date="1600",
    ...     language="en",
    ...     source_type="play",
    ... )
    >>> manifest.to_provenance()
    {'collection_id': 'gutenberg_shakespeare', 'source_title': '...', ...}
    """

    collection_id: str
    title: str | None = None
    author: str | None = None
    source_date: str | None = None
    language: str | None = None
    description: str = ""
    source_type: str | None = None
    file_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    expected_file_count: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Assert that all invariants hold.

        Raises
        ------
        ValueError
            If any invariant is violated.
        """
        if not self.collection_id or not self.collection_id.strip():
            raise ValueError(
                "CollectionManifest.collection_id must be a non-empty string."
            )
        if self.expected_file_count is not None and self.expected_file_count < 0:
            raise ValueError(
                f"CollectionManifest.expected_file_count must be >= 0; "
                f"got {self.expected_file_count!r}."
            )
        if not isinstance(self.file_provenance, dict):
            raise TypeError("CollectionManifest.file_provenance must be a dict.")

    def to_provenance(self) -> dict[str, Any]:
        """
        Return a provenance dict suitable for
        :meth:`~scikitplot.corpus._base.DocumentReader.create`.

        Only non-``None`` values are included.

        Returns
        -------
        dict[str, Any]
            Keys are ``CorpusDocument`` provenance field names.
        """  # noqa: D205
        prov: dict[str, Any] = {"collection_id": self.collection_id}
        if self.title is not None:
            prov["source_title"] = self.title
        if self.author is not None:
            prov["source_author"] = self.author
        if self.source_date is not None:
            prov["source_date"] = self.source_date
        if self.source_type is not None:
            prov["source_type"] = self.source_type
        return prov

    def provenance_for_file(self, filename: str) -> dict[str, Any]:
        """
        Return merged provenance for a specific file.

        Starts with collection-level defaults, then applies per-file
        overrides from ``file_provenance``. Basename matching only.

        Parameters
        ----------
        filename : str
            Source filename (basename). Matched against ``file_provenance``
            keys case-sensitively.

        Returns
        -------
        dict[str, Any]
            Merged provenance dict.

        Examples
        --------
        >>> manifest = CollectionManifest(
        ...     collection_id="c1",
        ...     author="Default Author",
        ...     file_provenance={"hamlet.xml": {"source_title": "Hamlet"}},
        ... )
        >>> manifest.provenance_for_file("hamlet.xml")
        {'collection_id': 'c1', 'source_author': 'Default Author',
         'source_title': 'Hamlet'}
        """
        base = self.to_provenance()
        override = self.file_provenance.get(pathlib.Path(filename).name, {})
        return {**base, **override}

    def check_completeness(self, actual_file_count: int) -> bool:
        """
        Return ``True`` if the actual file count matches ``expected_file_count``.

        Parameters
        ----------
        actual_file_count : int
            Number of files actually found in the collection directory.

        Returns
        -------
        bool
            Always ``True`` when ``expected_file_count`` is ``None``.
        """
        if self.expected_file_count is None:
            return True
        matches = actual_file_count == self.expected_file_count
        if not matches:
            logger.warning(
                "CollectionManifest %r: expected %d files, found %d.",
                self.collection_id,
                self.expected_file_count,
                actual_file_count,
            )
        return matches

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CollectionManifest("
            f"collection_id={self.collection_id!r}, "
            f"title={self.title!r}, "
            f"n_file_provenance={len(self.file_provenance)})"
        )


# ===========================================================================
# CorpusStats
# ===========================================================================


@dataclass(frozen=True)
class CorpusStats:
    """
    Aggregate statistics over a
    :class:`~scikitplot.corpus._schema.CorpusDocument` collection.

    Parameters
    ----------
    n_documents : int
        Total document count.
    n_tokens : int
        Total whitespace-delimited token count (sum of
        ``doc.word_count``).
    n_chars : int
        Total character count (sum of ``doc.char_count``).
    mean_tokens : float
        Average tokens per document. ``0.0`` when ``n_documents == 0``.
    median_tokens : float
        Median tokens per document. ``0.0`` when ``n_documents == 0``.
    min_tokens : int
        Minimum token count across documents.
    max_tokens : int
        Maximum token count across documents.
    language_counts : dict[str, int]
        Map of ISO 639-1 code → document count. ``None`` language stored
        as ``"unknown"``.
    section_type_counts : dict[str, int]
        Map of ``SectionType.value`` → document count.
    source_type_counts : dict[str, int]
        Map of ``SourceType.value`` → document count.
    source_file_counts : dict[str, int]
        Map of ``source_file`` → document count.
    collection_ids : list[str]
        Sorted unique ``collection_id`` values (``None`` excluded).
    has_embeddings : int
        Number of documents where ``embedding`` is not ``None``.
    date_range : tuple[str, str] or None
        (earliest, latest) ``source_date`` values, or ``None`` if no
        documents have dates.
    """  # noqa: D205

    n_documents: int
    n_tokens: int
    n_chars: int
    mean_tokens: float
    median_tokens: float
    min_tokens: int
    max_tokens: int
    language_counts: dict[str, int]
    section_type_counts: dict[str, int]
    source_type_counts: dict[str, int]
    source_file_counts: dict[str, int]
    collection_ids: list[str]
    has_embeddings: int
    date_range: tuple[str, str] | None

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-safe dictionary representation of the stats.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "n_documents": self.n_documents,
            "n_tokens": self.n_tokens,
            "n_chars": self.n_chars,
            "mean_tokens": self.mean_tokens,
            "median_tokens": self.median_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "language_counts": self.language_counts,
            "section_type_counts": self.section_type_counts,
            "source_type_counts": self.source_type_counts,
            "source_file_counts": self.source_file_counts,
            "collection_ids": self.collection_ids,
            "has_embeddings": self.has_embeddings,
            "date_range": list(self.date_range) if self.date_range else None,
        }

    def summary(self) -> str:
        """
        Return a human-readable one-page summary string.

        Returns
        -------
        str
        """
        lines = [
            "CorpusStats",
            f"  Documents  : {self.n_documents:,}",
            f"  Tokens     : {self.n_tokens:,} total "
            f"(mean {self.mean_tokens:.1f}, "
            f"median {self.median_tokens:.0f}, "
            f"min {self.min_tokens}, max {self.max_tokens})",
            f"  Characters : {self.n_chars:,}",
            f"  Embedded   : {self.has_embeddings:,}",
        ]
        if self.language_counts:
            top_lang = sorted(self.language_counts.items(), key=lambda x: -x[1])[:5]
            lines.append(
                "  Languages  : " + ", ".join(f"{k}={v:,}" for k, v in top_lang)
            )
        if self.collection_ids:
            lines.append(f"  Collections: {', '.join(self.collection_ids[:5])}")
        if self.date_range:
            lines.append(f"  Date range : {self.date_range[0]} - {self.date_range[1]}")
        return "\n".join(lines)


# ===========================================================================
# compute_stats — pure function
# ===========================================================================


def compute_stats(docs: Sequence[CorpusDocument]) -> CorpusStats:
    """
    Compute aggregate statistics over a document collection.

    Parameters
    ----------
    docs : sequence of CorpusDocument
        Documents to analyse. May be empty.

    Returns
    -------
    CorpusStats
        Frozen statistics object.

    Notes
    -----
    This is a pure function: same input → same output, no I/O, no
    mutation. Safe to call from multiple threads concurrently.

    Median is computed without NumPy using a sort-based O(n log n)
    algorithm so that this module has zero optional dependencies.

    Examples
    --------
    >>> stats = compute_stats(docs)
    >>> print(stats.summary())
    CorpusStats
      Documents  : 487
      Tokens     : 42,310 total (mean 86.9, ...)
    """
    n = len(docs)

    if n == 0:
        return CorpusStats(
            n_documents=0,
            n_tokens=0,
            n_chars=0,
            mean_tokens=0.0,
            median_tokens=0.0,
            min_tokens=0,
            max_tokens=0,
            language_counts={},
            section_type_counts={},
            source_type_counts={},
            source_file_counts={},
            collection_ids=[],
            has_embeddings=0,
            date_range=None,
        )

    token_counts: list[int] = []
    n_chars: int = 0
    lang_counts: dict[str, int] = {}
    section_counts: dict[str, int] = {}
    source_type_cnt: dict[str, int] = {}
    source_file_cnt: dict[str, int] = {}
    collection_ids_set: set[str] = set()
    has_embeddings: int = 0
    dates: list[str] = []

    for doc in docs:
        wc = doc.word_count
        token_counts.append(wc)
        n_chars += doc.char_count

        # Language distribution
        lang_key = doc.language or "unknown"
        lang_counts[lang_key] = lang_counts.get(lang_key, 0) + 1

        # Section type distribution
        st_key = doc.section_type.value
        section_counts[st_key] = section_counts.get(st_key, 0) + 1

        # Source type distribution
        src_key = doc.source_type.value
        source_type_cnt[src_key] = source_type_cnt.get(src_key, 0) + 1

        # Source file distribution
        sf = doc.source_file
        source_file_cnt[sf] = source_file_cnt.get(sf, 0) + 1

        # Collection IDs
        if doc.collection_id:
            collection_ids_set.add(doc.collection_id)

        # Embeddings
        if doc.has_embedding:
            has_embeddings += 1

        # Dates
        if doc.source_date:
            dates.append(doc.source_date)

    # Token statistics
    token_counts_sorted = sorted(token_counts)
    total_tokens = sum(token_counts)
    mean_tokens = total_tokens / n
    mid = n // 2
    if n % 2 == 0:
        median_tokens = (token_counts_sorted[mid - 1] + token_counts_sorted[mid]) / 2.0
    else:
        median_tokens = float(token_counts_sorted[mid])

    date_range: tuple[str, str] | None = None
    if dates:
        dates_sorted = sorted(dates)
        date_range = (dates_sorted[0], dates_sorted[-1])

    return CorpusStats(
        n_documents=n,
        n_tokens=total_tokens,
        n_chars=n_chars,
        mean_tokens=round(mean_tokens, 2),
        median_tokens=median_tokens,
        min_tokens=token_counts_sorted[0],
        max_tokens=token_counts_sorted[-1],
        language_counts=lang_counts,
        section_type_counts=section_counts,
        source_type_counts=source_type_cnt,
        source_file_counts=source_file_cnt,
        collection_ids=sorted(collection_ids_set),
        has_embeddings=has_embeddings,
        date_range=date_range,
    )


# ===========================================================================
# provenance_from_filename — pure heuristic
# ===========================================================================

# Patterns for common Gutenberg-style filename conventions:
#   Author_Title_Year.txt
#   author-title-year.txt
#   AUTHOR_title_YEAR.txt
_FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"""
    ^
    (?P<author>[A-Za-z][A-Za-z,._\- ]+?)  # Author part (surname, firstname)
    [_\-]                                   # Separator
    (?P<title>[A-Za-z0-9][A-Za-z0-9,._\- ]+?)  # Title part
    (?:[_\-](?P<year>\d{4}))?              # Optional year
    (?:\.[a-z]+)?$                          # Optional extension
    """,
    re.VERBOSE,
)


def provenance_from_filename(
    filename: str,
    source_type: str | None = None,
) -> dict[str, Any]:
    """
    Extract provenance metadata from a source filename using heuristics.

    Designed for corpora organised by the Project Gutenberg naming
    convention: ``Surname_Firstname_Title_Year.ext`` or
    ``Author-Title.ext``. Falls back gracefully — always returns a dict,
    even if no patterns are detected.

    Parameters
    ----------
    filename : str
        Source filename (basename or full path; only the basename is used).
    source_type : str or None, optional
        ``SourceType`` value string to include in the result. Default: ``None``.

    Returns
    -------
    dict[str, Any]
        Dict with zero or more of: ``"source_author"``, ``"source_title"``,
        ``"source_date"``, ``"source_type"``. Suitable for passing as
        ``source_provenance`` to
        :meth:`~scikitplot.corpus._sources.CorpusSource.from_file`.

    Examples
    --------
    >>> provenance_from_filename("Shakespeare_William_Hamlet_1603.xml")
    {'source_author': 'Shakespeare William', 'source_title': 'Hamlet',
     'source_date': '1603'}

    >>> provenance_from_filename("dickens-great-expectations.txt")
    {'source_author': 'Dickens', 'source_title': 'Great Expectations'}

    >>> provenance_from_filename("document.pdf")
    {}
    """
    stem = pathlib.Path(filename).stem
    # Normalise separators to spaces for pattern matching
    normalised = stem.replace("_", " ").replace("-", " ").strip()

    result: dict[str, Any] = {}

    # Attempt structured pattern match on original stem
    m = _FILENAME_PATTERN.match(stem)
    if m:
        author = m.group("author").replace("_", " ").replace("-", " ").strip()
        title = m.group("title").replace("_", " ").replace("-", " ").strip()
        year = m.group("year")

        if len(author) >= 3:  # noqa: PLR2004
            result["source_author"] = author.title()
        if len(title) >= 2:  # noqa: PLR2004
            result["source_title"] = title.title()
        if year:
            result["source_date"] = year
    else:
        # Fallback: if the normalised name has >= 2 words, use first word
        # as author guess and the rest as title guess.
        parts = [p for p in normalised.split() if p]
        if len(parts) >= 3:  # noqa: PLR2004
            result["source_author"] = parts[0].title()
            result["source_title"] = " ".join(parts[1:]).title()
        elif len(parts) == 2:  # noqa: PLR2004
            result["source_title"] = " ".join(parts).title()

    if source_type is not None:
        result["source_type"] = source_type

    if result:
        logger.debug("provenance_from_filename(%r) → %r.", filename, result)

    return result


__all__ = [
    "CollectionManifest",
    "CorpusStats",
    "compute_stats",
    "provenance_from_filename",
]
