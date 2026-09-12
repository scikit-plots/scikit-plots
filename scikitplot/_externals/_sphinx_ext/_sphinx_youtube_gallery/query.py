"""
YouTube-specific adapter over the shared selection engine.

Filtering, sorting, grouping and pagination are not YouTube concerns; they
are collection concerns. The implementation therefore lives in
the sibling ``_sphinx_collection.select`` module, which operates on plain
mappings and works in both supported package namespaces. This module contributes only what
is genuinely YouTube-specific:

* the mapping from :class:`~.model.VideoRecord` to a queryable record,
* the convenience filters a video page actually wants (``channel``,
  ``playlist``, a publication interval), expressed as ordinary filter terms.

Notes
-----
**Developer-focused.** Keeping this a thin adapter is what guarantees that
``youtube-gallery`` and ``gallery-grid`` sort and group identically. When
they shared no code they were free to diverge -- different tie-breaking,
different treatment of missing values -- and any such difference is a bug
that only shows up as two pages ordering the same data differently.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._sphinx_collection.select import (
    FilterError,
    FilterTerm,
    Selection,
    apply_selection,
)
from .._sphinx_collection.select import group_records as _group_records
from .model import CatalogError, CatalogRecord

__all__ = ["GROUP_KEYS", "SORT_KEYS", "Query", "apply_query", "group_records"]

#: Built-in sort keys. A syntactically safe field path is also accepted so
#: explicit ``fields:`` metadata has the same selection vocabulary as
#: ``gallery-grid``. ``position`` is the author's intended teaching sequence.
SORT_KEYS = (
    "title",
    "published",
    "duration",
    "position",
    "channel",
    "playlist",
    "none",
)

#: Built-in grouping keys. A syntactically safe custom field path is also
#: accepted so an explicit catalog ``fields:`` mapping can use the same
#: grouping vocabulary as ``gallery-grid``. Presence is validated once the
#: catalog is available, keeping typos build-visible rather than silently
#: producing a single ``Ungrouped`` section.
GROUP_KEYS = ("playlist", "channel", "year", "none")
_GROUP_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$")

# Full Python regular expressions can exhibit catastrophic backtracking.  The
# directive accepts a deliberately small, useful subset whose work is bounded:
# literals, character classes, anchors, dot, ``?`` and bounded ``{m,n}``
# repetition.  Groups, alternation, backreferences and unbounded repetition
# are rejected with guidance to use ``:match:`` for ordinary text search.
MAX_REGEX_LENGTH = 256
MAX_REGEX_REPEAT = 100
MAX_REGEX_TEXT = 131_072


def _compile_safe_regex(  # ruff: ignore[too-many-branches]
    pattern: str,
):
    """Compile the bounded ``:match-regex:`` subset or raise CatalogError."""

    if len(pattern) > MAX_REGEX_LENGTH:
        raise CatalogError(
            f"match-regex is {len(pattern)} characters; the limit is {MAX_REGEX_LENGTH}"
        )
    index = 0
    in_class = False
    previous_quantifier = False
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 1
            if index >= len(pattern):
                raise CatalogError("match-regex ends with an incomplete escape")
            if pattern[index].isdigit() or pattern[index] in {"g", "k"}:
                raise CatalogError("match-regex backreferences are not supported")
            previous_quantifier = False
        elif char == "[" and not in_class:
            in_class = True
            previous_quantifier = False
        elif char == "]" and in_class:
            in_class = False
            previous_quantifier = False
        elif not in_class and char in "()*+|":
            raise CatalogError(
                f"match-regex construct {char!r} is not supported; use literals, "
                "character classes, anchors, '.', '?', or bounded '{m,n}' repeats"
            )
        elif not in_class and char == "?":
            if previous_quantifier:
                raise CatalogError("adjacent match-regex quantifiers are not supported")
            previous_quantifier = True
        elif not in_class and char == "{":
            end = pattern.find("}", index + 1)
            if end < 0:
                raise CatalogError("match-regex has an unterminated bounded repeat")
            spec = pattern[index + 1 : end]
            match = re.fullmatch(r"(\d+)(?:,(\d*))?", spec)
            if not match:
                raise CatalogError(f"invalid bounded repeat {{{spec}}} in match-regex")
            lower = int(match.group(1))
            upper_text = match.group(2)
            upper = (
                lower
                if upper_text is None
                else (int(upper_text) if upper_text else MAX_REGEX_REPEAT + 1)
            )
            if lower > upper or upper > MAX_REGEX_REPEAT:
                raise CatalogError(
                    f"match-regex repeats must have 0 <= m <= n <= {MAX_REGEX_REPEAT}"
                )
            if previous_quantifier:
                raise CatalogError("adjacent match-regex quantifiers are not supported")
            previous_quantifier = True
            index = end
        else:
            previous_quantifier = False
        index += 1
    if in_class:
        raise CatalogError("match-regex has an unterminated character class")
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise CatalogError(f"invalid match-regex {pattern!r}: {exc}") from exc


def as_record(video: CatalogRecord) -> dict[str, Any]:
    """
    Project a normalized video or channel record into a queryable mapping.

    Parameters
    ----------
    video : CatalogRecord
        The normalized video or channel.

    Returns
    -------
    dict
        A mapping the shared engine can filter, sort and group. ``year`` is
        materialised as a field rather than computed during grouping so that
        it is also filterable and sortable, at no extra cost.

    Examples
    --------
    >>> from .model import normalize_record
    >>> record = as_record(normalize_record({"id": "dQw4w9WgXcQ", "title": "X"}))
    >>> record["title"], record["year"]
    ('X', 'unknown')
    """
    record = {
        "id": video.id,
        "title": video.title,
        "description": video.description,
        "channel": video.channel,
        "channel_id": video.channel_id,
        "handle": getattr(video, "handle", ""),
        "playlist": video.playlist,
        "playlist_id": video.playlist_id,
        "position": video.position,
        "published": video.published,
        "duration": video.duration,
        "tags": video.tags,
        "url": video.url,
        "year": video.year,
        "video_count": getattr(video, "video_count", None),
    }
    # ``fields:`` is an explicit escape hatch for domain-neutral gallery
    # metadata (category, audience, series, ...). Normalization prevents
    # collisions with every core/presentation key, so flattening here is safe
    # and lets the shared selection engine use ordinary field/dotted-path names.
    record.update(getattr(video, "fields", {}))
    record["_video"] = video
    return record


@dataclass(frozen=True)
class Query:
    """
    A declarative selection over a homogeneous YouTube catalog.

    A thin, YouTube-shaped face over the shared ``Selection`` type.
    All fields are optional; an empty :class:`Query` selects every record in
    catalog order.

    Attributes
    ----------
    channel, playlist : str
        Match against the display name or the stable id, case-insensitively.
    tags : sequence of str
        Keep records carrying **all** of these tags.
    match : str
        Case-insensitive substring over title and description.
    match_regex : str
        A case-insensitive expression in the bounded subset described below.
    since, until : datetime.datetime or None
        Half-open publication interval ``[since, until)``.
    sort_by : str
        A key from :data:`SORT_KEYS`, optionally prefixed with ``-``.
    group_by : str
        A built-in key from :data:`GROUP_KEYS` or a safe custom ``fields:``
        path such as ``category`` or ``audience.level``.
    limit : int or None
        Maximum records to return, applied after sorting.
    offset : int
        Records to skip, applied after sorting and before ``limit``.

    Notes
    -----
    ``match`` searches title *and* description, which the generic ``~``
    operator cannot express over two fields at once, so it is applied as a
    dedicated term rather than translated. ``match_regex`` is likewise kept
    here: the shared grammar deliberately excludes regular expressions,
    because a catastrophically backtracking pattern in a ``:filter:`` option
    would stall a documentation build. ``match_regex`` therefore accepts only
    literals, character classes, anchors, dot, ``?`` and bounded ``{m,n}``
    repeats. Groups, alternation, backreferences, ``*`` and ``+`` are rejected.
    """

    channel: str = ""
    playlist: str = ""
    tags: Sequence[str] = ()
    match: str = ""
    match_regex: str = ""
    since: _dt.datetime | None = None
    until: _dt.datetime | None = None
    sort_by: str = "none"
    group_by: str = "none"
    limit: int | None = None
    offset: int = 0

    def __post_init__(self) -> None:
        """
        Validate keys and bounds eagerly.

        Raises
        ------
        CatalogError
            If a sort or group key is unknown, both text matchers are given,
            a bound is negative, or the interval is empty. Reported by name
            with the valid alternatives, so a typo such as ``:sort: title``
            is an error rather than a silent fallback to catalog order.
        """
        key = self.sort_by.lstrip("-") or "none"
        if key not in SORT_KEYS and not _GROUP_FIELD_RE.fullmatch(key):
            raise CatalogError(
                f"invalid sort key {self.sort_by!r}; use one of "
                f"{sorted(SORT_KEYS)} or a field path such as 'category' "
                "(prefix with '-' for descending)"
            )
        if self.group_by not in GROUP_KEYS and not _GROUP_FIELD_RE.fullmatch(
            self.group_by
        ):
            raise CatalogError(
                f"invalid group key {self.group_by!r}; use one of "
                f"{sorted(GROUP_KEYS)} or a field path such as 'category'"
            )
        if self.match and self.match_regex:
            raise CatalogError(
                "'match' and 'match-regex' are mutually exclusive; use one"
            )
        if self.match_regex:
            _compile_safe_regex(self.match_regex)
        if self.limit is not None and self.limit < 0:
            raise CatalogError(f"limit must not be negative, got {self.limit}")
        if self.offset < 0:
            raise CatalogError(f"offset must not be negative, got {self.offset}")
        if self.since and self.until and self.since >= self.until:
            raise CatalogError(
                f"since ({self.since.date()}) must be earlier than "
                f"until ({self.until.date()})"
            )

    def to_selection(self) -> Selection:
        """
        Translate into a generic the shared ``Selection`` type.

        Returns
        -------
        Selection
            The equivalent selection. ``channel`` and ``playlist`` each
            become an "any of" term over their equivalent identity fields, so
            a page can filter by whichever identifier its catalog happens to
            record.

        Raises
        ------
        CatalogError
            If a value cannot be expressed as a filter term.
        """
        terms: list[FilterTerm] = []
        if self.channel:
            terms.append(_AnyOfTerm(("channel", "channel_id", "handle"), self.channel))
        if self.playlist:
            terms.append(_AnyOfTerm(("playlist", "playlist_id"), self.playlist))
        terms.extend(FilterTerm("tags", "=", tag) for tag in self.tags)
        if self.match:
            terms.append(_TextTerm(self.match))
        if self.match_regex:
            terms.append(_RegexTerm(self.match_regex))
        if self.since or self.until:
            terms.append(_IntervalTerm(self.since, self.until))

        sort_key = self.sort_by.lstrip("-") or "none"
        sort_text = ""
        if sort_key != "none":
            sort_text = f"-{sort_key}" if self.sort_by.startswith("-") else sort_key

        try:
            return Selection(
                terms=terms,
                sort_keys=(
                    [(sort_text.lstrip("-"), sort_text.startswith("-"))]
                    if sort_text
                    else []
                ),
                group_by="" if self.group_by == "none" else self.group_by,
                limit=self.limit,
                offset=self.offset,
            )
        except FilterError as exc:  # pragma: no cover - bounds already checked
            raise CatalogError(str(exc)) from exc


class _AnyOfTerm(FilterTerm):
    """Match one identity against any of several equivalent fields."""

    def __init__(self, fields: Sequence[str], wanted: str) -> None:
        if not fields:
            raise ValueError("fields must not be empty")
        super().__init__(field=fields[0], operator="=", operand=wanted)
        object.__setattr__(self, "_fields", tuple(fields))

    @staticmethod
    def _identity(value: Any) -> str:
        """Case-fold identity text and treat ``@handle``/``handle`` equally."""
        return str(value or "").strip().casefold().lstrip("@")

    def matches(self, record) -> bool:
        """
        Test the wanted identity against every equivalent field.

        Handles are compared with or without their leading ``@`` so a URL
        parsed to ``youtube`` still matches a catalog value ``@youtube``.
        """
        wanted = self._identity(self.operand)
        return wanted in {self._identity(record.get(field)) for field in self._fields}


class _TextTerm(FilterTerm):
    """Case-insensitive substring over title and description together."""

    def __init__(self, needle: str) -> None:
        super().__init__(field="title", operator="~", operand=needle)

    def matches(self, record) -> bool:
        """
        Test the record's title and description.

        Parameters
        ----------
        record : mapping
            The record to test.

        Returns
        -------
        bool
            ``True`` if the needle occurs in either field.
        """
        haystack = f"{record.get('title', '')}\n{record.get('description', '')}"
        return self.operand.casefold() in haystack.casefold()


class _RegexTerm(FilterTerm):
    """Case-insensitive regular expression over title and description."""

    def __init__(self, pattern: str) -> None:
        super().__init__(field="title", operator="~", operand=pattern)
        object.__setattr__(self, "_compiled", _compile_safe_regex(pattern))

    def matches(self, record) -> bool:
        """
        Test the record's title and description against the pattern.

        Parameters
        ----------
        record : mapping
            The record to test.

        Returns
        -------
        bool
            ``True`` if the pattern matches either field.
        """
        haystack = f"{record.get('title', '')}\n{record.get('description', '')}"
        return bool(self._compiled.search(haystack[:MAX_REGEX_TEXT]))


class _IntervalTerm(FilterTerm):
    """Half-open publication interval ``[since, until)``."""

    def __init__(self, since: _dt.datetime | None, until: _dt.datetime | None) -> None:
        super().__init__(field="published", operator="", operand="")
        object.__setattr__(self, "_since", since)
        object.__setattr__(self, "_until", until)

    def matches(self, record) -> bool:
        """
        Test whether the record falls inside the interval.

        Parameters
        ----------
        record : mapping
            The record to test.

        Returns
        -------
        bool
            ``True`` if the publication date lies in the interval. A video
            with no publication date is excluded: "unknown date" cannot be
            asserted to fall inside a requested range.
        """
        published = record.get("published")
        if published is None:
            return False
        if self._since and published < self._since:
            return False
        if self._until and published >= self._until:  # ruff: ignore[needless-bool]
            return False
        return True


def _path_exists(record: Mapping[str, Any], path: str) -> bool:
    """Return whether a dotted field path exists, even when its value is None."""
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _validate_selection_fields(
    records: Sequence[Mapping[str, Any]], query: Query
) -> None:
    """Reject typoed/custom sort or group paths once records are available."""
    for option, raw in (
        ("sort", query.sort_by.lstrip("-")),
        ("group-by", query.group_by),
    ):
        key = raw or "none"
        if key == "none" or not records:
            continue
        if not any(_path_exists(record, key) for record in records):
            raise CatalogError(
                f"option ':{option}:' field {key!r} is not present in this catalog; "
                "put domain-neutral metadata under each record's 'fields:' mapping"
            )


def apply_query(
    videos: Iterable[CatalogRecord], query: Query
) -> tuple[list[CatalogRecord], int]:
    """
    Filter, sort and paginate a video catalog.

    Parameters
    ----------
    videos : iterable of CatalogRecord
        The homogeneous catalog.
    query : Query
        The selection to apply.

    Returns
    -------
    selected : list of CatalogRecord
        Records after filtering, sorting, ``offset`` and ``limit``.
    total : int
        How many matched the filters *before* pagination, so a caller can
        report "showing 24 of 1043" rather than silently truncating.

    Examples
    --------
    >>> from .model import normalize_catalog
    >>> catalog = normalize_catalog(
    ...     [
    ...         {"id": "aaaaaaaaaaa", "title": "Beta", "playlist": "P1"},
    ...         {"id": "bbbbbbbbbbb", "title": "alpha", "playlist": "P2"},
    ...     ]
    ... )
    >>> selected, total = apply_query(catalog, Query(sort_by="title"))
    >>> [r.title for r in selected], total
    (['alpha', 'Beta'], 2)
    >>> selected, total = apply_query(catalog, Query(playlist="p1"))
    >>> [r.title for r in selected], total
    (['Beta'], 1)
    >>> selected, total = apply_query(catalog, Query(limit=1))
    >>> len(selected), total
    (1, 2)
    """
    records = [as_record(video) for video in videos]
    _validate_selection_fields(records, query)
    selected, total = apply_selection(records, query.to_selection())
    return [record["_video"] for record in selected], total


def group_records(
    videos: Sequence[CatalogRecord], query: Query
) -> list[tuple[str, list[CatalogRecord]]]:
    """
    Partition videos into labelled sections.

    Parameters
    ----------
    videos : sequence of CatalogRecord
        Already filtered, sorted and paginated records.
    query : Query
        Supplies ``group_by``.

    Returns
    -------
    list of (str, list of CatalogRecord)
        Section label paired with its records. With ``group_by="none"`` this
        is a single ``("", videos)`` pair, so callers have one code path for
        grouped and ungrouped rendering.

    Examples
    --------
    >>> from .model import normalize_catalog
    >>> catalog = normalize_catalog(
    ...     [
    ...         {"id": "aaaaaaaaaaa", "playlist": "Intro"},
    ...         {"id": "bbbbbbbbbbb"},
    ...         {"id": "ccccccccccc", "playlist": "Intro"},
    ...     ]
    ... )
    >>> [
    ...     (label, len(rs))
    ...     for label, rs in group_records(catalog, Query(group_by="playlist"))
    ... ]
    [('Intro', 2), ('Ungrouped', 1)]
    >>> [label for label, _ in group_records(catalog, Query())]
    ['']
    """
    records = [as_record(video) for video in videos]
    _validate_selection_fields(records, query)
    sections = _group_records(records, query.to_selection())
    return [
        (label, [record["_video"] for record in group]) for label, group in sections
    ]
