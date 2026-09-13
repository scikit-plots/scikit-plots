"""
Data model for the YouTube learning catalog.

This module owns the *only* definition of what a catalog record is. It is
deliberately free of Sphinx and network imports so that it can be exercised
by plain unit tests, by the offline sync tool, and by the Sphinx directive
alike.

Design contract
---------------
The catalog is the single source of truth consumed at documentation build
time. Acquisition (talking to YouTube) happens in a separate, explicitly
invoked step that *writes* a catalog; the Sphinx build only ever *reads*
one. That separation is what makes the docs build deterministic,
reproducible offline, and immune to the quota exhaustion, rate limiting and
socket timeouts that make build-time API calls a recurring source of broken
pipelines.

Notes
-----
**User-focused.** A catalog is an ordinary YAML file you can hand-edit.
Existing video catalogs keep their ``videos:`` shape; a channel collection
uses a separate ``channels:`` shape. One directive invocation is intentionally
homogeneous, so video-player cards and simple channel-link cards never become
ambiguous or leak presentation rules into one another.

**Developer-focused.** Normalization is total and deterministic: the same
input record always yields the same :class:`VideoRecord`, and any record
that cannot be normalized raises :class:`CatalogError` naming the offending
index and field. No field is ever silently dropped or guessed.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .._sphinx_collection._yaml import MAX_COLLECTION_ITEMS

__all__ = [
    "CatalogError",
    "CatalogRecord",
    "ChannelRecord",
    "VideoRecord",
    "derive_channel_records",
    "normalize_catalog",
    "normalize_channel_record",
    "normalize_gallery_catalog",
    "normalize_record",
    "parse_duration",
    "parse_timestamp",
]


class CatalogError(ValueError):
    """
    Raised when a catalog payload cannot be normalized.

    Carries a message that names the record index and the offending field so
    a maintainer can locate the problem in the YAML without a traceback.
    """


#: Canonical YouTube video ids are exactly 11 URL-safe base64 characters.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

#: ISO-8601 duration as returned by the YouTube Data API, e.g. ``PT1H2M30S``.
_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


def parse_video_id(value: Any) -> str:
    """
    Normalize any accepted video reference to its canonical 11-character id.

    Thin wrapper over :func:`~.reference.parse_video_reference`, which owns
    the URL grammar. Kept as a separate name because the catalog only ever
    needs the id, while the reference parser returns the full structure.

    Parameters
    ----------
    value : Any
        A bare id or any YouTube URL that names a single video, including
        watch URLs carrying ``list``/``index``/``t`` parameters, ``youtu.be``
        links, Shorts, live and embed URLs, and bare query fragments.

    Returns
    -------
    str
        The canonical video id.

    Raises
    ------
    CatalogError
        If the value is not a YouTube reference, is malformed, or names a
        playlist or channel rather than one video.

    Examples
    --------
    >>> parse_video_id("dQw4w9WgXcQ")
    'dQw4w9WgXcQ'
    >>> parse_video_id("https://www.youtube.com/watch?v=JXtISpdDPNY&list=PL1x")
    'JXtISpdDPNY'
    >>> parse_video_id("https://youtu.be/JXtISpdDPNY?t=30")
    'JXtISpdDPNY'
    >>> parse_video_id("https://www.youtube.com/shorts/hbT7vzCvEc8")
    'hbT7vzCvEc8'
    """
    from .._sphinx_youtube_core.reference import (  # ruff: ignore[import-outside-top-level]
        ReferenceError,
        parse_video_reference,
    )

    try:
        return parse_video_reference(value).video_id
    except ReferenceError as exc:
        raise CatalogError(str(exc)) from exc


def parse_timestamp(value: Any) -> _dt.datetime | None:
    """
    Parse a publication timestamp into a timezone-aware UTC datetime.

    Accepts what both the YouTube Data API (RFC 3339, ``Z``-suffixed) and a
    hand-written catalog (a bare ``YYYY-MM-DD`` date, or a value PyYAML has
    already turned into a ``date``/``datetime``) realistically produce.

    Parameters
    ----------
    value : Any
        ``None``, a string, a :class:`datetime.date`, or a
        :class:`datetime.datetime`.

    Returns
    -------
    datetime.datetime or None
        A timezone-aware UTC datetime, or ``None`` if ``value`` is ``None``.
        Naive inputs are interpreted as UTC, which is what the YouTube API
        reports and what makes ``:since:``/``:until:`` comparisons total.

    Raises
    ------
    CatalogError
        If the value is present but cannot be parsed.

    Examples
    --------
    >>> parse_timestamp("2024-03-01T10:00:00Z").isoformat()
    '2024-03-01T10:00:00+00:00'
    >>> parse_timestamp("2024-03-01").isoformat()
    '2024-03-01T00:00:00+00:00'
    >>> parse_timestamp(None) is None
    True
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        parsed = value
    elif isinstance(value, _dt.date):
        parsed = _dt.datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # `fromisoformat` gained full RFC 3339 'Z' support only in 3.11;
        # normalizing here keeps the module working on older interpreters
        # across the whole declared support range.
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise CatalogError(
                f"{value!r} is not a valid ISO-8601 date or timestamp: {exc}"
            ) from exc
    else:
        raise CatalogError(
            f"timestamp must be a string or date, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def parse_duration(value: Any) -> int | None:
    """
    Parse a video duration into whole seconds.

    Parameters
    ----------
    value : Any
        ``None``, an integer number of seconds, an ISO-8601 duration as
        returned by the YouTube Data API (``PT1H2M30S``), or a clock-style
        string (``1:02:30`` or ``2:30``).

    Returns
    -------
    int or None
        Duration in seconds, or ``None`` if ``value`` is ``None``.

    Raises
    ------
    CatalogError
        If the value is present but cannot be parsed.

    Examples
    --------
    >>> parse_duration("PT1H2M30S")
    3750
    >>> parse_duration("2:30")
    150
    >>> parse_duration(90)
    90
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise CatalogError("duration must not be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise CatalogError(f"duration must not be negative, got {value}")
        return value
    if not isinstance(value, str):
        raise CatalogError(
            f"duration must be an int or string, got {type(value).__name__}"
        )
    text = value.strip()
    match = _ISO_DURATION_RE.match(text)
    if match and any(match.groupdict().values()):
        parts = {k: int(v) for k, v in match.groupdict().items() if v}
        return (
            parts.get("days", 0) * 86400
            + parts.get("hours", 0) * 3600
            + parts.get("minutes", 0) * 60
            + parts.get("seconds", 0)
        )
    if ":" in text:
        chunks = text.split(":")
        _len = len(chunks) > 3  # ruff: ignore[magic-value-comparison]
        if _len or not all(c.isdigit() for c in chunks):
            raise CatalogError(f"{value!r} is not a valid clock duration")
        total = 0
        for chunk in chunks:
            total = total * 60 + int(chunk)
        return total
    raise CatalogError(
        f"{value!r} is not a valid duration "
        f"(expected seconds, 'PT1H2M30S', or '1:02:30')"
    )


def _as_str_list(value: Any, field_name: str) -> list[str]:
    """
    Coerce a scalar-or-list YAML value into a list of strings.

    Parameters
    ----------
    value : Any
        ``None``, a single string, or a list of strings.
    field_name : str
        Field name, used in the error message.

    Returns
    -------
    list of str
        Possibly empty list of stripped, non-empty strings.

    Raises
    ------
    CatalogError
        If the value is neither a string nor a list of strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise CatalogError(
            f"{field_name} must be a string or list of strings, "
            f"got {type(value).__name__}"
        )
    result = []
    for entry in value:
        if not isinstance(entry, str):
            raise CatalogError(
                f"{field_name} entries must be strings, got {type(entry).__name__}"
            )
        stripped = entry.strip()
        if stripped:
            result.append(stripped)
    return result


# Custom gallery metadata is opt-in and deliberately namespaced in authored
# catalogs under ``fields:``. During rendering/query projection those keys are
# flattened into the ordinary gallery record so authors can use the exact same
# ``:group-by:``, filter, sort, and search field names as ``gallery-grid``.
# Presentation/identity names are reserved so metadata can never silently turn
# into a Sphinx Design card option or overwrite YouTube identity.
_CUSTOM_FIELD_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_CUSTOM_FIELD_RESERVED = frozenset(
    {
        "id",
        "url",
        "title",
        "description",
        "channel",
        "channel_id",
        "handle",
        "playlist",
        "playlist_id",
        "position",
        "published",
        "duration",
        "tags",
        "year",
        "kind",
        "video_count",
        "fields",
        "content",
        "header",
        "image",
        "link",
        "link-alt",
        "link-type",
        "img-top",
        "img-bottom",
        "img-alt",
        "img-background",
        "class-body",
        "class-card",
        "class-footer",
        "class-header",
        "class-img-bottom",
        "class-img-top",
        "class-item",
        "class-title",
        "columns",
        "margin",
        "padding",
        "shadow",
        "text-align",
        "width",
        "name",
    }
)
_MAX_CUSTOM_FIELD_DEPTH = 6


def _custom_fields(value: Any, index: int) -> dict[str, Any]:
    """
    Validate explicit, presentation-neutral gallery metadata.

    Nested mappings are supported so the shared collection engine's dotted
    paths (for example ``audience.level``) keep working. Values remain ordinary
    YAML data, but non-finite floats and unsupported container/object types are
    rejected before they can reach YAML/JSON/browser serialization.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise CatalogError(
            f"record {index}: fields must be a mapping, got {type(value).__name__}"
        )

    def clean(node: Any, path: str, depth: int) -> Any:
        if depth > _MAX_CUSTOM_FIELD_DEPTH:
            raise CatalogError(
                f"record {index}: fields.{path} exceeds the maximum metadata "
                f"nesting depth of {_MAX_CUSTOM_FIELD_DEPTH}"
            )
        if node is None or isinstance(node, (str, bool, int, _dt.date, _dt.datetime)):
            return node
        if isinstance(node, float):
            if not math.isfinite(node):
                raise CatalogError(
                    f"record {index}: fields.{path} must be a finite number"
                )
            return node
        if isinstance(node, list):
            result = []
            for i, item in enumerate(node):
                if isinstance(item, (dict, list)):
                    raise CatalogError(
                        f"record {index}: fields.{path}[{i}] must be a scalar; "
                        "put nested structure in a mapping and address it with a dotted path"
                    )
                result.append(clean(item, f"{path}[{i}]", depth + 1))
            return result
        if isinstance(node, dict):
            result: dict[str, Any] = {}
            for key, child in node.items():
                if not isinstance(key, str) or not _CUSTOM_FIELD_NAME_RE.fullmatch(key):
                    raise CatalogError(
                        f"record {index}: fields.{path} key {key!r} must start with "
                        "a letter and contain only letters, digits, '_' or '-'"
                    )
                if depth == 0 and key in _CUSTOM_FIELD_RESERVED:
                    raise CatalogError(
                        f"record {index}: fields key {key!r} is reserved by "
                        "youtube-gallery/gallery-grid; choose a metadata-only name"
                    )
                result[key] = clean(child, f"{path}.{key}" if path else key, depth + 1)
            return result
        raise CatalogError(
            f"record {index}: fields.{path} has unsupported value type "
            f"{type(node).__name__}"
        )

    return clean(value, "", 0)


@dataclass(frozen=True)
class VideoRecord:
    """
    One normalized video in the learning catalog.

    All fields except ``id`` are optional so that a hand-written catalog can
    be as small as a list of URLs, while a generated one carries the full
    metadata needed for sorting, filtering and sectioning.

    Attributes
    ----------
    id : str
        Canonical 11-character YouTube video id.
    title : str
        Human-readable title. Defaults to the id when unknown, so a card is
        never rendered with an empty heading.
    description : str
        Free text used by build-time matching and optional browser search
        fields. It is metadata only; ``youtube-gallery`` does not add a
        description paragraph to the card body.
    channel : str
        Channel display name, used for grouping and filtering.
    channel_id : str
        Stable channel identifier (``UC...``), preferred over display metadata
        when a canonical channel identity is available.
    handle : str
        Optional YouTube handle without ``@``. This keeps human-facing
        ``channel`` text separate from linkable channel identity, allowing the
        same video catalog to group by a display name and project offline to
        channel cards without an API lookup.
    playlist : str
        Playlist display name, used for grouping and filtering.
    playlist_id : str
        Stable playlist identifier (``PL...``).
    position : int or None
        Zero-based position within its playlist. ``None`` when the record
        did not come from a playlist. This is the only ordering that
        reflects the *author's* intended teaching sequence, so it is the
        default sort for playlist-scoped queries.
    published : datetime.datetime or None
        Publication timestamp in UTC.
    duration : int or None
        Runtime in whole seconds.
    tags : list of str
        Free-form labels for topical filtering.
    fields : dict
        Optional custom gallery metadata authored under ``fields:``. It is
        never rendered as card content, but is exposed to the same grouping,
        filtering, sorting, and search-field machinery as ``gallery-grid``.
    url : str
        Canonical HTTPS watch URL. Normalization validates a supplied URL
        against the video ID and emits the canonical URL, dropping additional
        query parameters. Titles and descriptions are rendered as plain text.
    """

    id: str
    title: str = ""
    description: str = ""
    channel: str = ""
    channel_id: str = ""
    handle: str = ""
    playlist: str = ""
    playlist_id: str = ""
    position: int | None = None
    published: _dt.datetime | None = None
    duration: int | None = None
    tags: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    url: str = ""

    @property
    def year(self) -> str:
        """
        Publication year as a string, or ``"unknown"``.

        Returns
        -------
        str
            Four-digit year, or ``"unknown"`` when ``published`` is ``None``.
            A string (rather than ``None``) keeps year-grouping keys totally
            ordered and directly usable as section headings.
        """
        return str(self.published.year) if self.published else "unknown"


@dataclass(frozen=True)
class ChannelRecord:
    """
    One normalized YouTube channel card.

    Channel catalogs deliberately share the query-facing fields of
    :class:`VideoRecord` so ``youtube-gallery`` can pass both kinds through
    the same selection and ``gallery-grid`` rendering infrastructure. Fields
    that only make sense for videos are present with empty/``None`` values;
    they are never rendered on a channel card.

    Attributes
    ----------
    id : str
        Stable collection key: the canonical ``UC...`` id when known,
        otherwise ``@handle`` or the legacy channel name.
    title : str
        Visible card title. Defaults to ``@handle`` when available.
    channel : str
        Display/search value for the channel. Defaults to ``title``.
    channel_id : str
        Canonical ``UC...`` id when supplied.
    handle : str
        Handle without the leading ``@`` when supplied.
    url : str
        Canonical HTTPS channel-root URL. Channel tab suffixes are removed,
        because a channel card represents the channel itself rather than a
        dynamic video-section request.
    description, tags : data
        Search/filter metadata. They remain data only; the card body stays
        title-only to match a plain ``gallery-grid`` link card.
    fields : dict
        Optional custom gallery metadata authored under ``fields:`` for a
        native channel catalog. Derived channel projections intentionally do
        not guess how arbitrary per-video fields should aggregate.
    video_count : int or None
        Number of selected video records represented by a channel derived
        through ``:view: channels``. ``None`` for an authored channel catalog.
    published : datetime.datetime or None
        For a derived channel, the newest known publication time among the
        selected catalog videos. ``None`` for ordinary authored channel cards.
    """

    id: str
    title: str = ""
    description: str = ""
    channel: str = ""
    channel_id: str = ""
    handle: str = ""
    playlist: str = ""
    playlist_id: str = ""
    position: int | None = None
    published: _dt.datetime | None = None
    duration: int | None = None
    tags: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    url: str = ""
    video_count: int | None = None

    @property
    def year(self) -> str:
        """Return the latest catalog-video year for a derived channel, if known."""
        return str(self.published.year) if self.published else "unknown"


CatalogRecord = VideoRecord | ChannelRecord

_CHANNEL_KNOWN_KEYS = frozenset(
    {
        "id",
        "url",
        "title",
        "description",
        "channel",
        "channel_id",
        "handle",
        "tags",
        "fields",
    }
)


def normalize_channel_record(  # ruff: ignore[too-many-branches]
    raw: Any,
    index: int = 0,
) -> ChannelRecord:
    """
    Normalize one channel-catalog entry to a title-only link card.

    Parameters
    ----------
    raw : Any
        A channel reference string (``@handle``, ``UC...`` id, or YouTube
        channel URL), or a mapping carrying that reference in ``id``/``url``/
        ``handle``/``channel_id`` plus optional title/search metadata.
    index : int, optional
        Record position used in located validation messages.

    Returns
    -------
    ChannelRecord
        Canonical channel-root identity and metadata.

    Raises
    ------
    CatalogError
        If the record is malformed or names something other than a channel.
    """
    if isinstance(raw, str):
        raw = {"id": raw}
    if not isinstance(raw, dict):
        raise CatalogError(
            f"record {index}: expected a mapping or channel reference string, "
            f"got {type(raw).__name__}"
        )
    unknown = set(raw) - _CHANNEL_KNOWN_KEYS
    if unknown:
        raise CatalogError(
            f"record {index}: unknown channel key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(_CHANNEL_KNOWN_KEYS)}"
        )

    def _text(key: str) -> str:
        value = raw.get(key)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise CatalogError(
                f"record {index}: {key} must be a string, got {type(value).__name__}"
            )
        return value.strip()

    reference_value = _text("url") or _text("id")
    explicit_handle = _text("handle")
    explicit_channel_id = _text("channel_id")
    if explicit_handle:
        from .._sphinx_youtube_core.reference import (  # ruff: ignore[import-outside-top-level]
            ReferenceError,
            validate_handle,
        )

        try:
            explicit_handle = validate_handle(explicit_handle)
        except ReferenceError as exc:
            raise CatalogError(
                f"record {index}: invalid channel handle: {exc}"
            ) from exc
    if not reference_value:
        reference_value = (
            f"@{explicit_handle}" if explicit_handle else explicit_channel_id
        )
    if not reference_value:
        raise CatalogError(
            f"record {index}: missing channel reference "
            f"('id', 'url', 'handle', or 'channel_id')"
        )

    from .._sphinx_youtube_core.reference import (  # ruff: ignore[import-outside-top-level]
        CHANNEL,
        ReferenceError,
        parse_reference,
        validate_channel_id,
        validate_handle,
    )

    try:
        reference = parse_reference(reference_value)
    except ReferenceError as exc:
        raise CatalogError(f"record {index}: {exc}") from exc
    if reference.kind != CHANNEL:
        raise CatalogError(
            f"record {index}: expected a YouTube channel, got {reference.describe()}"
        )

    handle = explicit_handle or reference.handle
    channel_id = explicit_channel_id or reference.channel_id
    if handle:
        try:
            handle = validate_handle(handle)
        except ReferenceError as exc:
            raise CatalogError(
                f"record {index}: invalid channel handle: {exc}"
            ) from exc
    if channel_id:
        try:
            channel_id = validate_channel_id(channel_id)
        except ReferenceError as exc:
            raise CatalogError(f"record {index}: invalid channel_id: {exc}") from exc
    if (
        explicit_handle
        and reference.handle
        and handle.casefold() != reference.handle.casefold()
    ):
        raise CatalogError(
            f"record {index}: handle and channel URL name different channels"
        )
    if (
        explicit_channel_id
        and reference.channel_id
        and channel_id != reference.channel_id
    ):
        raise CatalogError(
            f"record {index}: channel_id and channel URL name different channels"
        )

    channel_name = reference.channel_name
    if channel_id:
        # Stable UC identity owns deduplication when available; an authored
        # handle remains the friendlier canonical URL and default label.
        canonical_url = (
            f"https://www.youtube.com/@{handle}"
            if handle
            else f"https://www.youtube.com/channel/{channel_id}"
        )
        identity = channel_id
        default_title = f"@{handle}" if handle else channel_id
    elif handle:
        canonical_url = f"https://www.youtube.com/@{handle}"
        identity = f"@{handle.casefold()}"
        default_title = f"@{handle}"
    elif channel_name:
        canonical_url = f"https://www.youtube.com/{channel_name}"
        identity = channel_name.casefold()
        default_title = channel_name
    else:
        raise CatalogError(f"record {index}: channel carries no usable identifier")

    title = _text("title") or default_title
    channel = _text("channel") or title
    try:
        tags = _as_str_list(raw.get("tags"), "tags")
        fields = _custom_fields(raw.get("fields"), index)
    except CatalogError as exc:
        # _custom_fields already locates its errors; avoid duplicating the prefix.
        if str(exc).startswith(f"record {index}:"):
            raise
        raise CatalogError(f"record {index}: {exc}") from exc

    return ChannelRecord(
        id=identity,
        title=title,
        description=_text("description"),
        channel=channel,
        channel_id=channel_id,
        handle=handle,
        tags=tags,
        fields=fields,
        url=canonical_url,
    )


def derive_channel_records(  # ruff: ignore[too-many-branches]
    records: list[VideoRecord],
) -> list[ChannelRecord]:
    """
    Project a video selection into unique, stable channel cards.

    The projection is deliberately offline: it uses only channel identity already
    present in the reviewed catalog. ``channel_id`` is preferred, then the
    explicit video-level ``handle`` field; an ``@handle`` or channel URL in
    ``channel`` remains a compatibility fallback. Plain display names are not
    guessed because they cannot produce a trustworthy link.

    Identity is order-independent. If any selected record pairs a handle with a
    canonical ``UC…`` id, other records carrying only that handle join the same
    canonical channel bucket even when they appear earlier in the catalog.

    Tags are unioned across the selected videos, ``video_count`` records how many
    selected videos contributed to the channel, and ``published`` becomes the
    newest known publication timestamp in that selected slice. This makes one
    video catalog useful for both video browsing and channel exploration without
    duplicating source data or adding build-time network access.
    """
    from .._sphinx_youtube_core.reference import (  # ruff: ignore[import-outside-top-level]
        CHANNEL,
        ReferenceError,
        is_reference_url,
        parse_reference,
    )

    def authored_handle(record: VideoRecord) -> str:
        if record.handle:
            return record.handle.lstrip("@")
        value = record.channel.strip()
        if value.startswith("@") and len(value) > 1:
            return value[1:]
        # Ask "is this written as a URL?" here and let ``parse_reference``
        # be the only place that judges the host.  A ``"youtube.com/" in
        # value`` test both accepted the allowed host at an arbitrary
        # position and missed ``youtu.be``, ``music.youtube.com`` and every
        # country domain, which then skipped validation entirely.
        if is_reference_url(value):
            try:
                ref = parse_reference(value)
            except ReferenceError:
                return ""
            return ref.handle.lstrip("@") if ref.kind == CHANNEL and ref.handle else ""
        return ""

    # Learn identity aliases before projection.  A historical handle may be
    # renamed or even reused, so only a handle that points to exactly one UC id
    # is strong enough to upgrade a handle-only record.  Stable channel ids are
    # never merged merely because an ambiguous historical handle matches.
    handle_channel_ids: dict[str, set[str]] = {}
    preferred_handle: dict[str, tuple[_dt.datetime | None, int, str]] = {}
    for index, record in enumerate(records):
        handle = authored_handle(record)
        if not (handle and record.channel_id):
            continue
        handle_channel_ids.setdefault(handle.casefold(), set()).add(record.channel_id)
        candidate = (record.published, index, handle)
        current = preferred_handle.get(record.channel_id)
        # Prefer the handle attached to the newest known video.  When dates are
        # missing/equal, later catalog order wins deterministically; this lets a
        # maintained historical catalog model a channel-handle rename without
        # splitting the stable UC identity.
        candidate_key = (
            record.published or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc),
            index,
        )
        current_key = (
            (
                current[0] or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc),
                current[1],
            )
            if current
            else None
        )
        if current_key is None or candidate_key > current_key:
            preferred_handle[record.channel_id] = candidate
    handle_to_channel_id = {
        handle_key: next(iter(channel_ids))
        for handle_key, channel_ids in handle_channel_ids.items()
        if len(channel_ids) == 1
    }

    def display_label(  # ruff: ignore[too-many-return-statements]
        record: VideoRecord,
    ) -> str:
        """Return a human-facing channel label, never a raw channel URL/id."""
        value = record.channel.strip()
        if not value or value.startswith("UC"):
            return ""
        if value.startswith("@"):
            return value
        # Anything written as a URL must survive host validation before it
        # can be shown.  The previous substring gate let a non-YouTube link
        # fall through to the ``return value`` below, so an unvalidated URL
        # was rendered verbatim as a channel label -- exactly what this
        # function's contract forbids.
        if is_reference_url(value):
            try:
                ref = parse_reference(value)
            except ReferenceError:
                return ""
            if ref.kind != CHANNEL:
                return ""
            if ref.handle:
                return f"@{ref.handle.lstrip('@')}"
            if ref.channel_name:
                return ref.channel_name
            return ""
        return value

    # Keep display naming as fresh/deterministic as handle selection.  A channel
    # can be renamed while retaining its UC identity; using the first historical
    # label would leave a projected card visibly stale even though its handle URL
    # had already advanced to the newest known identity metadata.
    preferred_label: dict[str, tuple[_dt.datetime | None, int, str]] = {}
    for index, record in enumerate(records):
        authored = authored_handle(record)
        effective_channel_id = record.channel_id or (
            handle_to_channel_id.get(authored.casefold(), "") if authored else ""
        )
        label = display_label(record)
        if not (effective_channel_id and label):
            continue
        candidate_key = (
            record.published or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc),
            index,
        )
        current = preferred_label.get(effective_channel_id)
        current_key = (
            (
                current[0] or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc),
                current[1],
            )
            if current
            else None
        )
        if current_key is None or candidate_key > current_key:
            preferred_label[effective_channel_id] = (record.published, index, label)

    buckets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        authored = authored_handle(record)
        effective_channel_id = record.channel_id or (
            handle_to_channel_id.get(authored.casefold(), "") if authored else ""
        )
        # When stable identity is known, prefer the handle associated with the
        # newest catalog record for that channel.  Keep the authored handle for
        # handle-only buckets and for unique alias resolution.
        handle = (
            preferred_handle.get(effective_channel_id, (None, -1, ""))[2]
            if effective_channel_id
            else authored
        ) or authored
        raw_reference = effective_channel_id or handle or record.channel
        if not raw_reference:
            continue
        try:
            if effective_channel_id:
                ref = parse_reference(effective_channel_id)
            elif handle:
                ref = parse_reference(f"@{handle}")
            elif record.channel.startswith("@") or is_reference_url(record.channel):
                ref = parse_reference(record.channel)
            else:
                # A display name alone is intentionally not resolvable offline.
                continue
        except ReferenceError:
            continue
        if ref.kind != CHANNEL:
            continue
        if ref.channel_id:
            identity = ref.channel_id
            url = (
                f"https://www.youtube.com/@{handle}"
                if handle
                else f"https://www.youtube.com/channel/{ref.channel_id}"
            )
        elif ref.handle:
            handle = ref.handle.lstrip("@")
            identity = f"@{handle.casefold()}"
            url = f"https://www.youtube.com/@{handle}"
        else:
            continue
        title = (
            preferred_label.get(effective_channel_id, (None, -1, ""))[2]
            if effective_channel_id
            else display_label(record)
        )
        if not title:
            title = f"@{handle}" if handle else identity
        entry = buckets.get(identity)
        if entry is None:
            entry = {
                "id": identity,
                "title": title,
                "channel": title,
                "channel_id": ref.channel_id or "",
                "handle": handle,
                "url": url,
                "tags": [],
                "seen_tags": set(),
                "published": record.published,
                "video_count": 0,
            }
            buckets[identity] = entry
            order.append(identity)
        else:
            # Later records may carry a friendlier handle/display label even
            # though identity was already established by a canonical UC id.
            if handle and not entry["handle"]:
                entry["handle"] = handle
                entry["url"] = f"https://www.youtube.com/@{handle}"
            if entry["title"] == identity and title != identity:
                entry["title"] = title
                entry["channel"] = title
        entry["video_count"] += 1
        if record.published and (
            entry["published"] is None or record.published > entry["published"]
        ):
            entry["published"] = record.published
        for tag in record.tags:
            key = tag.casefold()
            if key not in entry["seen_tags"]:
                entry["seen_tags"].add(key)
                entry["tags"].append(tag)

    result: list[ChannelRecord] = []
    for identity in order:
        entry = buckets[identity]
        entry.pop("seen_tags")
        result.append(ChannelRecord(**entry))
    return result


def normalize_gallery_catalog(  # ruff: ignore[too-many-branches]
    payload: Any,
    origin: str = "catalog",
) -> tuple[str, list[CatalogRecord]]:
    """
    Normalize one homogeneous ``youtube-gallery`` catalog.

    Existing bare lists and ``videos:`` mappings remain video catalogs.
    A mapping containing ``channels:`` becomes a channel catalog. Supplying
    both keys is rejected on purpose: one directive invocation must render
    one card kind, which keeps runtime additions and player/link behavior
    unambiguous.

    Parameters
    ----------
    payload : Any
        Parsed YAML/JSON catalog.
    origin : str, optional
        Human-readable source name for diagnostics.

    Returns
    -------
    (str, list of CatalogRecord)
        ``("video", records)`` or ``("channel", records)``.

    Raises
    ------
    CatalogError
        If both collection kinds are present or a channel catalog is invalid.
    """
    if isinstance(payload, dict) and "channels" in payload:
        if "videos" in payload:
            raise CatalogError(
                f"{origin}: use either 'videos' or 'channels' in one "
                f"youtube-gallery, not both; split mixed content into two directives"
            )
        entries = payload["channels"]
        if entries is None:
            return "channel", []
        if not isinstance(entries, list):
            raise CatalogError(
                f"{origin}: 'channels' must be a list, got {type(entries).__name__}"
            )
        if len(entries) > MAX_COLLECTION_ITEMS:
            raise CatalogError(
                f"{origin}: catalog contains {len(entries):,} records; the limit is "
                f"{MAX_COLLECTION_ITEMS:,}. Split very large catalogs across pages."
            )
        normalized: list[ChannelRecord] = []
        for index, raw in enumerate(entries):
            try:
                normalized.append(normalize_channel_record(raw, index))
            except CatalogError as exc:  # ruff: ignore[try-except-in-loop]
                raise CatalogError(f"{origin}: {exc}") from exc

        # Native channel catalogs are explicit declarations, so contradictory
        # identity pairs are configuration errors rather than something to
        # resolve by source order.  This prevents a duplicated/stale handle from
        # silently linking the wrong canonical channel card.
        handle_ids: dict[str, set[str]] = {}
        channel_handles: dict[str, set[str]] = {}
        for record in normalized:
            if not (record.handle and record.channel_id):
                continue
            handle_ids.setdefault(record.handle.casefold(), set()).add(
                record.channel_id
            )
            channel_handles.setdefault(record.channel_id, set()).add(record.handle)
        for handle_key, channel_ids in handle_ids.items():
            if len(channel_ids) > 1:
                rendered = ", ".join(sorted(channel_ids))
                raise CatalogError(
                    f"{origin}: channel handle @{handle_key} is paired with multiple "
                    f"channel_id values ({rendered}); keep one canonical identity"
                )
        for channel_id, handles in channel_handles.items():
            if len({value.casefold() for value in handles}) > 1:
                rendered = ", ".join(
                    f"@{value}" for value in sorted(handles, key=str.casefold)
                )
                raise CatalogError(
                    f"{origin}: channel_id {channel_id} is paired with multiple handles "
                    f"({rendered}); keep one current handle in a native channel catalog"
                )

        # Learn canonical aliases before de-duplication so source order cannot
        # decide whether the surviving card has a stable UC identity or a
        # friendlier authored @handle URL.
        handle_to_channel_id = {
            handle_key: next(iter(channel_ids))
            for handle_key, channel_ids in handle_ids.items()
        }
        channel_id_to_handle = {
            channel_id: next(iter(handles))
            for channel_id, handles in channel_handles.items()
        }

        from dataclasses import replace  # ruff: ignore[import-outside-top-level]

        records: list[CatalogRecord] = []
        seen: set[str] = set()
        for record in normalized:
            channel_id = record.channel_id or (
                handle_to_channel_id.get(record.handle.casefold(), "")
                if record.handle
                else ""
            )
            handle = record.handle or channel_id_to_handle.get(channel_id, "")
            identity = channel_id or record.id
            url = (
                f"https://www.youtube.com/@{handle}"
                if handle
                else (
                    f"https://www.youtube.com/channel/{channel_id}"
                    if channel_id
                    else record.url
                )
            )
            title = record.title
            if handle and title in {record.id, record.channel_id}:
                title = f"@{handle}"
            channel = record.channel
            if handle and channel in {record.id, record.channel_id}:
                channel = title
            record = replace(  # ruff: ignore[redefined-loop-name]
                record,
                id=identity,
                channel_id=channel_id,
                handle=handle,
                url=url,
                title=title,
                channel=channel,
            )
            if record.id in seen:
                continue
            seen.add(record.id)
            records.append(record)
        return "channel", records

    return "video", list(normalize_catalog(payload, origin))


#: Catalog keys that map onto a :class:`VideoRecord` field. Any other key in
#: a record is rejected rather than ignored: a typo such as ``title`` would
#: otherwise silently produce a card titled with the bare video id.
_KNOWN_KEYS = frozenset(
    {
        "id",
        "url",
        "title",
        "description",
        "channel",
        "channel_id",
        "handle",
        "playlist",
        "playlist_id",
        "position",
        "published",
        "duration",
        "tags",
        "fields",
    }
)


def normalize_record(  # ruff: ignore[too-many-branches]
    raw: Any,
    index: int = 0,
) -> VideoRecord:
    """
    Normalize one raw catalog entry into a :class:`VideoRecord`.

    Parameters
    ----------
    raw : Any
        A mapping of catalog keys, or a bare string treated as a video
        reference (so that the simplest possible catalog is a list of URLs).
    index : int, optional
        Position of this record in the catalog, used in error messages.

    Returns
    -------
    VideoRecord
        The normalized record.

    Raises
    ------
    CatalogError
        If the entry is not a mapping or string, carries an unknown key, or
        any field fails to parse. The message always names ``index``.

    Examples
    --------
    >>> normalize_record("https://youtu.be/JXtISpdDPNY").id
    'JXtISpdDPNY'
    >>> record = normalize_record({"id": "JXtISpdDPNY", "title": "PCA"})
    >>> record.title, record.url
    ('PCA', 'https://www.youtube.com/watch?v=JXtISpdDPNY')
    """
    if isinstance(raw, str):
        raw = {"id": raw}
    if not isinstance(raw, dict):
        raise CatalogError(
            f"record {index}: expected a mapping or a video URL string, "
            f"got {type(raw).__name__}"
        )

    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        raise CatalogError(
            f"record {index}: unknown key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(_KNOWN_KEYS)}"
        )

    reference = raw.get("id") or raw.get("url")
    if reference is None:
        raise CatalogError(f"record {index}: missing required key 'id' (or 'url')")

    try:
        video_id = parse_video_id(reference)
        published = parse_timestamp(raw.get("published"))
        duration = parse_duration(raw.get("duration"))
        tags = _as_str_list(raw.get("tags"), "tags")
        fields = _custom_fields(raw.get("fields"), index)
    except CatalogError as exc:
        if str(exc).startswith(f"record {index}:"):
            raise
        raise CatalogError(f"record {index}: {exc}") from exc

    position = raw.get("position")
    if position is not None:
        if isinstance(position, bool) or not isinstance(position, int):
            raise CatalogError(
                f"record {index}: position must be an integer, "
                f"got {type(position).__name__}"
            )
        if position < 0:
            raise CatalogError(
                f"record {index}: position must not be negative, got {position}"
            )

    def _text(key: str) -> str:
        """Return a stripped string for ``key``, rejecting non-strings."""
        value = raw.get(key)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise CatalogError(
                f"record {index}: {key} must be a string, got {type(value).__name__}"
            )
        return value.strip()

    # Never forward a raw URL into generated RST/MyST or HTML links. A record
    # may supply both id and url; validate they identify the same video.
    raw_url = _text("url")
    if raw_url and parse_video_id(raw_url) != video_id:
        raise CatalogError(f"record {index}: url and id name different videos")
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    handle = _text("handle")
    channel_id = _text("channel_id")
    playlist_id = _text("playlist_id")
    from .._sphinx_youtube_core.reference import (  # ruff: ignore[import-outside-top-level]
        ReferenceError,
        validate_channel_id,
        validate_handle,
        validate_playlist_id,
    )

    if handle:
        try:
            handle = validate_handle(handle)
        except ReferenceError as exc:
            raise CatalogError(
                f"record {index}: invalid channel handle: {exc}"
            ) from exc
    if channel_id:
        try:
            channel_id = validate_channel_id(channel_id)
        except ReferenceError as exc:
            raise CatalogError(f"record {index}: invalid channel_id: {exc}") from exc
    if playlist_id:
        try:
            playlist_id = validate_playlist_id(playlist_id)
        except ReferenceError as exc:
            raise CatalogError(f"record {index}: invalid playlist_id: {exc}") from exc

    return VideoRecord(
        id=video_id,
        # Falling back to the id keeps every card titled: an untitled card in
        # a grid is indistinguishable from a broken one.
        title=_text("title") or video_id,
        description=_text("description"),
        channel=_text("channel"),
        channel_id=channel_id,
        handle=handle,
        playlist=_text("playlist"),
        playlist_id=playlist_id,
        position=position,
        published=published,
        duration=duration,
        tags=tags,
        fields=fields,
        url=canonical_url,
    )


def normalize_catalog(payload: Any, origin: str = "catalog") -> list[VideoRecord]:
    """
    Normalize a whole catalog payload into records, de-duplicating by id.

    Accepts either a bare list of records or a mapping with a ``videos``
    key, so that a generated catalog can carry sibling metadata (such as the
    sync timestamp and source query) alongside the records without the
    reader needing to know which shape it was handed.

    Parameters
    ----------
    payload : Any
        The value parsed from a catalog YAML/JSON document.
    origin : str, optional
        Description of where the payload came from, used in error messages.

    Returns
    -------
    list of VideoRecord
        Records in source order. When the same video id appears more than
        once, the *first* occurrence wins and later duplicates are dropped:
        a video legitimately appears in several playlists, and merging those
        catalogs must stay deterministic and order-stable.

    Raises
    ------
    CatalogError
        If the payload shape is wrong or any record fails to normalize.

    Examples
    --------
    >>> len(normalize_catalog(["https://youtu.be/JXtISpdDPNY"]))
    1
    >>> len(normalize_catalog({"videos": ["JXtISpdDPNY", "JXtISpdDPNY"]}))
    1
    """
    if payload is None:
        return []
    if isinstance(payload, dict):
        if "videos" not in payload:
            raise CatalogError(
                f"{origin}: mapping payload must contain a 'videos' key, "
                f"found {sorted(payload)}"
            )
        payload = payload["videos"]
        if payload is None:
            return []
    if not isinstance(payload, list):
        raise CatalogError(
            f"{origin}: expected a list of records (or a mapping with a "
            f"'videos' list), got {type(payload).__name__}"
        )
    if len(payload) > MAX_COLLECTION_ITEMS:
        raise CatalogError(
            f"{origin}: catalog contains {len(payload):,} records; the limit is "
            f"{MAX_COLLECTION_ITEMS:,}. Split very large catalogs across pages."
        )

    records: list[VideoRecord] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload):
        try:
            record = normalize_record(raw, index)
        except CatalogError as exc:
            raise CatalogError(f"{origin}: {exc}") from exc
        if record.id in seen:
            continue
        seen.add(record.id)
        records.append(record)
    return records
