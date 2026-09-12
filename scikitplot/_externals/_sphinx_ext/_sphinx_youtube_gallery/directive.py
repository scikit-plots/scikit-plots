"""
The ``youtube-gallery`` Sphinx directive.

Renders a queried slice of a YouTube catalog as a Sphinx Design grid, a
thumbnail wall, or a compact link list, from either a ``.md`` (MyST) or a
``.rst`` page.

Architecture
------------
The directive is the *render* layer only. It performs no network access and
holds no knowledge of the YouTube API::

    tools/youtube_sync.py   -->   _data/youtube/*.yaml   -->   youtube-gallery
    (explicit, online)            (committed, reviewed)        (build, offline)

Keeping acquisition out of the build is what makes documentation builds
deterministic and reproducible: the same commit always produces the same
HTML, a Read the Docs build never fails because of an API quota or a socket
timeout, and a reviewer can see in the diff exactly which videos a pull
request adds.

Rendering is delegated to the existing ``gallery-grid`` directive rather
than emitting Sphinx Design markup a second time, so the grid stays
byte-identical to the rest of the site and the MyST fence-safety handling
lives in exactly one place.

Notes
-----
**User-focused.** One directive covers every scenario: the whole catalog
sorted and filtered, sections per playlist, a single channel or playlist, a
date interval, or an inline list of links written straight into the page.

**Developer-focused.** Every option is validated before any output is
produced, and every failure is reported through docutils with a file and
line number so a bad option never aborts the build.
"""

from __future__ import annotations

import datetime as _dt
import re
import string
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from yaml import safe_dump

from .._sphinx_collection import (
    CONTAINER_CLASS,
    SECTION_STYLES,
    render_sections,
    sections_allowed,
)
from .._sphinx_collection._browser import collection_id, field_names
from .._sphinx_collection._presentation import CARD_SPEC, GRID_SPEC
from .._sphinx_collection._yaml import (
    BoundedYAMLError,
    load_bounded_yaml,
    read_bounded_utf8,
)
from .._sphinx_youtube_core.video_options import VIDEO_SPEC, player_options
from .model import (
    CatalogError,
    CatalogRecord,
    ChannelRecord,
    VideoRecord,
    derive_channel_records,
    normalize_gallery_catalog,
)
from .query import Query, apply_query, group_records

logger = logging.getLogger(__name__)

__all__ = ["YouTubeGalleryDirective", "setup"]

#: Presentation modes, in increasing order of page weight.
#:
#: ``list``
#:     A plain bullet list of links. No images, no iframes. The only mode
#:     that stays usable at four-figure record counts.
#: ``thumbnail``
#:     A grid of cards, each a remote thumbnail image linking out to
#:     YouTube. One image request per card, no third-party player code.
#: ``embed``
#:     A grid of cards, each an inline ``youtube`` player.
#: ``auto``
#:     ``embed`` at or below :data:`DEFAULT_MAX_EMBEDS` records, otherwise
#:     ``thumbnail``.
MODES = ("auto", "embed", "thumbnail", "list")

#: Content projection. ``auto`` follows the catalog kind; ``channels`` may
#: project a video catalog into one offline, deduplicated card per known channel.
VIEWS = ("auto", "videos", "channels")

#: Default ceiling on inline players per page.
#:
#: Each embed is a third-party iframe that loads the YouTube player. Even
#: with ``loading="lazy"``, a page of several hundred embeds is slow to
#: parse, heavy on memory, and hostile to assistive technology, which has to
#: traverse every frame. Above this ceiling ``auto`` degrades to thumbnails,
#: which look the same in a grid but cost one static image each.
DEFAULT_MAX_EMBEDS = 24

#: Remote thumbnail URL template. ``hqdefault`` is used rather than
#: ``maxresdefault`` because it is generated for *every* video, including
#: older and lower-resolution uploads; ``maxresdefault`` 404s for many and
#: would leave broken images scattered through a large gallery.
THUMBNAIL_URL = "https://i.ytimg.com/vi/{id}/hqdefault.jpg"


def _literal_metadata(value: str) -> str:
    """
    Render catalog text literally in RST and MyST, never as directives.

    Collapse source whitespace to one line and escape ASCII punctuation.
    The generic gallery's explicitly authored content remains markup.
    """
    text = " ".join(value.split())
    return "".join("\\" + char if char in string.punctuation else char for char in text)


def _mode_choice(argument: str) -> str:
    """
    Validate the ``:mode:`` option.

    Parameters
    ----------
    argument : str
        Raw option text.

    Returns
    -------
    str
        One of :data:`MODES`.

    Raises
    ------
    ValueError
        If the value is not a known mode. Docutils turns this into a located
        directive error.
    """
    return directives.choice(argument.strip().lower(), MODES)


def _view_choice(argument: str) -> str:
    """Validate the ``:view:`` content projection."""
    return directives.choice(argument.strip().lower(), VIEWS)


def _comma_list(argument: str) -> list[str]:
    """
    Split a comma-separated option value into a list of trimmed strings.

    Parameters
    ----------
    argument : str
        Raw option text, e.g. ``"pca, clustering"``.

    Returns
    -------
    list of str
        Non-empty, whitespace-trimmed entries.
    """
    if not argument:
        return []
    return [part.strip() for part in argument.split(",") if part.strip()]


def _format_duration(seconds: int | None) -> str:
    """
    Render a duration in seconds as a compact clock string.

    Parameters
    ----------
    seconds : int or None
        Runtime in seconds.

    Returns
    -------
    str
        ``"H:MM:SS"``, ``"M:SS"``, or ``""`` when ``seconds`` is ``None``.

    Examples
    --------
    >>> _format_duration(3750)
    '1:02:30'
    >>> _format_duration(150)
    '2:30'
    >>> _format_duration(None)
    ''
    """
    if seconds is None:
        return ""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class YouTubeGalleryDirective(SphinxDirective):
    """
    Render a queried slice of a YouTube catalog.

    The directive takes an optional argument naming a catalog file, relative
    to the current document. With no argument it falls back to the
    ``youtube_catalog_path`` configuration value, and with directive content
    it treats that content as an inline catalog, which is how an author
    writes a short, explicit list of links without maintaining a data file.

    Options
    -------
    catalog : path
        Catalog file, equivalent to the positional argument.
    channel, playlist : str
        Restrict to one channel or playlist, by display name or by stable
        id.
    tags : comma-separated str
        Keep records carrying all of the given tags.
    match : str
        Case-insensitive substring over title and description.
    match-regex : str
        Case-insensitive bounded regular expression over title and description.
        For predictable build time, supports literals, classes, anchors, dot,
        ``?`` and bounded ``{m,n}`` repeats; use ``match`` for ordinary text.
    since, until : ISO date
        Half-open publication interval.
    sort : str
        ``title``, ``published``, ``duration``, ``position``, ``channel``,
        ``playlist`` or ``none``; prefix with ``-`` to reverse.
    group-by : str
        ``playlist``, ``channel``, ``year`` or ``none``.
    limit, offset : int
        Pagination over the sorted result.
    mode : str
        One of :data:`MODES`; controls presentation only.
    view : {"auto", "videos", "channels"}
        Content projection. ``auto`` follows the catalog kind. On a video
        catalog, ``channels`` creates one offline channel card per stable
        ``channel_id``/explicit ``@handle`` represented by the selected videos.
        This reuses one reviewed video catalog for channel exploration without
        build-time API access or a duplicate channel file.
    grid-columns, columns : str
        Responsive grid counts, e.g. ``1 1 2 2``. ``columns`` is the existing
        alias; specifying different values for both is a located error.
    show-duration : flag
        Append known video durations to card titles. Channel cards ignore it.
        Descriptions remain searchable metadata and are intentionally not
        rendered in card bodies, matching ``gallery-grid``.
    class-card, class-container : str
        Existing aliases for additional card and grid container CSS classes.
    section-style : {"auto", "section", "rubric"}
        Use real sections where allowed, otherwise rubrics. Explicit section
        mode warns when it must fall back; auto falls back quietly.
    searchable : flag
        Add local search to emitted cards without enabling facet/sort controls.
    interactive : flag
        Add local search, available field filters, sorting, counts and Reset.
        Controls act on rendered cards only and are not enabled in list mode.
    filter-fields : comma-separated field names
        Metadata fields for live dropdowns, e.g. ``channel,tags``. Fields with
        no values are omitted. Multiple selected fields combine with AND.
    sort-fields : comma-separated field names
        Live sort fields; defaults to title. Fields without values are omitted.
        Sorting stays within existing groups and places missing values last.
    search-fields : comma-separated field names
        Additional metadata to search, beyond card text and titles.
    search-label : str
        Accessible search name and placeholder for this gallery.
    collection-id : str
        Unique, stable per-page name enabling the visitor's optional saved
        additions. Starts with a letter; up to 64 letters, digits, _ or -.
    grid-* : validated Sphinx Design options
        Grid presentation, including gutter, margin, padding, outline, reverse,
        class-container and class-row. The underlying option registry is used.
    card-* : validated Sphinx Design options
        Shared card presentation, including shadow, padding and class-card.
        Namespaced card-class-card overrides the existing class-card alias.
    video-* : validated player options
        Shared width, height, aspect, align, title, privacy-mode and
        url-parameters. Player titles default to each record's title.

    Notes
    -----
    Inline content takes precedence over the positional file, then ``catalog``,
    then ``youtube_catalog_path``. A single source is selected, never merged.
    Explicit file references resolve relative to the document within the source
    tree; the configured fallback resolves from the source directory.

    Grid/card/player settings apply to the generated elements in this directive.
    Build-time selection remains owned by this directive, so it is not applied
    again through nested grid options. Use separate invocations for different
    presentation defaults. The generic ``gallery-grid`` directive remains the
    entry point for arbitrary authored markup inside a card.

    Controls do not send or persist search terms. Visitor-added links can be saved
    locally only after explicit opt-in when collection-id is configured. Provider frames and images
    retain their normal network behavior. Without JavaScript the static content
    remains available. Sorting can reload players when DOM-preserving moves
    are unavailable; hiding a player does not pause playback.
    """

    name = "youtube-gallery"
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec: ClassVar[dict[str, Any]] = {
        **GRID_SPEC,
        **CARD_SPEC,
        **VIDEO_SPEC,
        "catalog": directives.unchanged,
        "channel": directives.unchanged,
        "playlist": directives.unchanged,
        "tags": _comma_list,
        "match": directives.unchanged,
        "match-regex": directives.unchanged,
        "since": directives.unchanged,
        "until": directives.unchanged,
        "sort": directives.unchanged,
        "group-by": directives.unchanged,
        "limit": directives.nonnegative_int,
        "show-count": directives.flag,
        "offset": directives.nonnegative_int,
        "mode": _mode_choice,
        "view": _view_choice,
        "columns": directives.unchanged,
        "grid-columns": directives.unchanged,
        "interactive": directives.flag,
        "filter-fields": field_names,
        "sort-fields": field_names,
        "search-fields": field_names,
        "show-duration": directives.flag,
        "show-description": (  # legacy no-op: card bodies match gallery-grid
            directives.flag
        ),
        "class-card": directives.unchanged,
        "class-container": directives.unchanged,
        "section-style": lambda argument: directives.choice(
            (argument or "auto").strip().lower(), SECTION_STYLES
        ),
        "searchable": directives.flag,
        "search-label": directives.unchanged,
        "collection-id": collection_id,
    }

    # -- input -------------------------------------------------------------

    def _load_records(self) -> tuple[str, list[CatalogRecord]]:
        """
        Resolve and load the catalog for this directive invocation.

        Precedence is explicit and single-valued: directive content, then
        the positional argument, then the ``:catalog:`` option, then the
        ``youtube_catalog_path`` config value. Exactly one source is used,
        so a page never silently merges two catalogs.

        Returns
        -------
        (str, list of CatalogRecord)
            Collection kind (``"video"`` or ``"channel"``) and normalized
            homogeneous records.

        Raises
        ------
        CatalogError
            If no source is configured, the file is missing, the YAML is
            unparsable, or a record fails to normalize.
        """
        if self.content:
            payload = self._parse_yaml(
                "\n".join(self.content), "youtube-gallery directive content"
            )
            return normalize_gallery_catalog(
                payload, "youtube-gallery directive content"
            )

        reference = None
        if self.arguments:
            reference = self.arguments[0].strip()
        elif self.options.get("catalog"):
            reference = self.options["catalog"].strip()

        if reference:
            from .._sphinx_gallery_grid.directive import (  # ruff: ignore[import-outside-top-level]
                _confined_path,
            )

            try:
                path = _confined_path(self, reference)
            except ValueError as exc:
                raise CatalogError(str(exc)) from exc
        else:
            configured = getattr(self.env.config, "youtube_catalog_path", "")
            if not configured:
                raise CatalogError(
                    "no catalog given: pass a path as the directive argument, "
                    "set the ':catalog:' option, write the videos as directive "
                    "content, or set 'youtube_catalog_path' in conf.py"
                )
            path = (Path(self.env.srcdir) / configured).resolve()

        if not path.exists():
            raise CatalogError(f"catalog file not found: {path}")

        # Register before reading so that editing the catalog invalidates
        # this document on an incremental build. Without it the page keeps
        # serving stale HTML until someone runs a clean build -- the single
        # most confusing failure mode of a generated-data pipeline.
        self.env.note_dependency(str(path))
        # A catalog saved as latin-1 or cp1252 otherwise raised an unhandled
        # `UnicodeDecodeError` that aborted the whole build; it is an input
        # problem and belongs in a located directive error.
        try:
            text = read_bounded_utf8(path, str(path))
        except BoundedYAMLError as exc:
            raise CatalogError(str(exc)) from exc
        payload = self._parse_yaml(text, str(path))
        return normalize_gallery_catalog(payload, str(path))

    @staticmethod
    def _parse_yaml(text: str, origin: str) -> Any:
        """
        Parse YAML, converting parser errors into :class:`CatalogError`.

        Parameters
        ----------
        text : str
            YAML source.
        origin : str
            Description of where the text came from, for the error message.

        Returns
        -------
        Any
            The parsed payload.

        Raises
        ------
        CatalogError
        If the text is invalid YAML or exceeds the shared byte, alias,
        nesting, node, or scalar-text resource limits.
        """
        try:
            return load_bounded_yaml(text, origin)
        except BoundedYAMLError as exc:
            raise CatalogError(str(exc)) from exc

    def _build_query(self) -> Query:
        """
        Translate directive options into a :class:`Query`.

        Returns
        -------
        Query
            The validated query.

        Raises
        ------
        CatalogError
            If any option value is invalid.
        """
        from .model import parse_timestamp  # ruff: ignore[import-outside-top-level]

        def _bound(name: str) -> _dt.datetime | None:
            """Parse a date option, prefixing errors with the option name."""
            raw = self.options.get(name)
            if not raw:
                return None
            try:
                return parse_timestamp(raw.strip())
            except CatalogError as exc:
                raise CatalogError(f"option ':{name}:' -- {exc}") from exc

        def _identifier(name: str, attribute: str) -> str:
            """
            Read a channel/playlist option, accepting a URL or a bare name.

            A page author copying a link has no reason to know which
            substring of it is the id, so a URL is reduced to its identifier
            here. Anything that is not a URL is passed through untouched, so
            filtering by a human-readable display name keeps working.
            """
            raw = self.options.get(name, "").strip()
            if not raw or "/" not in raw:
                return raw
            from .._sphinx_youtube_core.reference import (  # ruff: ignore[import-outside-top-level]
                ReferenceError,
                parse_reference,
            )

            try:
                reference = parse_reference(raw)
            except ReferenceError as exc:
                raise CatalogError(f"option ':{name}:' -- {exc}") from exc
            resolved = getattr(reference, attribute, "")
            if not resolved and attribute == "channel_id":
                # A handle or vanity name is a perfectly good filter value
                # when the catalog records it; only a URL with no channel
                # component at all is an error.
                resolved = reference.handle or reference.channel_name
            if not resolved:
                raise CatalogError(
                    f"option ':{name}:' -- {raw!r} names {reference.describe()}, "
                    f"which carries no {name} identifier"
                )
            return resolved

        return Query(
            channel=_identifier("channel", "channel_id"),
            playlist=_identifier("playlist", "playlist_id"),
            tags=tuple(self.options.get("tags", ())),
            match=self.options.get("match", "").strip(),
            match_regex=self.options.get("match-regex", "").strip(),
            since=_bound("since"),
            until=_bound("until"),
            sort_by=self.options.get("sort", "none").strip() or "none",
            group_by=self.options.get("group-by", "none").strip() or "none",
            limit=self.options.get("limit"),
            offset=self.options.get("offset", 0),
        )

    # -- output ------------------------------------------------------------

    def _resolve_mode(self, count: int) -> str:
        """
        Decide the presentation mode for ``count`` records.

        Parameters
        ----------
        count : int
            Number of records that will be rendered.

        Returns
        -------
        str
            A concrete mode: never ``"auto"``.

        Notes
        -----
        An explicit ``:mode: embed`` is always honoured -- the author may
        know something the heuristic does not -- but exceeding the embed
        budget emits a warning naming the count and the ceiling, so the
        cost is visible in the build log rather than only in a browser
        profile.
        """
        budget = getattr(
            self.env.config, "youtube_catalog_max_embeds", DEFAULT_MAX_EMBEDS
        )
        requested = self.options.get("mode", "auto")
        if requested == "auto":
            return "embed" if count <= budget else "thumbnail"
        if requested == "embed" and count > budget:
            logger.warning(
                f"youtube-gallery: rendering {count} inline players exceeds "
                f"the budget of {budget} "
                f"(youtube_catalog_max_embeds). Consider ':mode: thumbnail', "
                f"':mode: list', or ':limit:' to keep the page responsive.",
                location=self.get_location(),
            )
        return requested

    def _card(self, record: CatalogRecord, mode: str, rst: bool) -> dict[str, Any]:
        """
        Build one ``gallery-grid`` item mapping for a record.

        Parameters
        ----------
        record : CatalogRecord
            The video or channel to render. Videos become the same nested
            ``youtube`` player content an author could place in
            ``gallery-grid``; channels become title-only stretched-link cards.
        mode : str
            ``"embed"`` or ``"thumbnail"``.
        rst : bool
            Whether the calling page is reStructuredText, which decides the
            directive syntax used for an embedded player.

        Returns
        -------
        dict
            A mapping consumed by ``gallery-grid``, including metadata for
            the shared browser controls. No youtube-gallery-only prose is
            inserted into the card body.
        """
        title = record.title
        if "show-duration" in self.options and record.duration is not None:
            title = f"{title} ({_format_duration(record.duration)})"

        item: dict[str, Any] = {
            "title": _literal_metadata(title),
            "_sk_collection_browser_title": " ".join(title.split()),
            "description": record.description,
            "channel": record.channel,
            "channel_id": record.channel_id,
            "handle": getattr(record, "handle", ""),
            "playlist": record.playlist,
            "playlist_id": record.playlist_id,
            "tags": list(record.tags),
            "published": record.published.isoformat() if record.published else None,
            "year": record.year,
            "duration": record.duration,
            "position": record.position,
            "kind": "channel" if isinstance(record, ChannelRecord) else "video",
            "video_count": getattr(record, "video_count", None),
        }
        # Explicit ``fields:`` metadata is intentionally presentation-neutral.
        # Model validation reserves every identity/card-option name, so these
        # keys can be flattened safely and then behave exactly like authored
        # gallery-grid metadata (including dotted nested paths).
        # Keep accessibility-only link prose out of the local search corpus.
        # The typed adapter's default searchable identity is the human title;
        # authors opt into description/channel/custom metadata explicitly via
        # ``:search-fields:`` just as they do for gallery-grid.
        item["_sk_collection_search_base"] = [record.title]

        custom_fields = getattr(record, "fields", {})
        item.update(custom_fields)
        if custom_fields:
            # Tell gallery-grid these flattened keys are data, never card
            # options. This remains true even if a future Sphinx Design
            # release introduces an option named like an authored field.
            item["_sk_collection_metadata_only"] = list(custom_fields)

        if isinstance(record, ChannelRecord):
            # Exactly the same authored shape as a plain gallery-grid channel
            # card: visible title only, with Sphinx Design's stretched link.
            # Description/tags stay metadata for search/filtering and never
            # become p.sd-card-text in the card body.
            item["link"] = record.url
            item["link-alt"] = f"Open {' '.join(record.title.split())} on YouTube"
            return item

        if mode == "embed":
            # `gallery-grid` treats `content` as opaque source text for the
            # page's own markup language, so the player is written in that
            # language rather than as pre-rendered HTML.
            video_options = player_options(self.options, record.title)

            def _leaf_option_line(key: str, value: Any, prefix: str = "") -> str:
                # privacy_mode is a true flag. Emit canonical valueless source
                # rather than relying on parsers to treat trailing whitespace
                # as an empty argument. Explicit false was already omitted by
                # player_options().
                if key == "privacy_mode" and value == "":
                    return f"{prefix}:{key}:\n"
                return f"{prefix}:{key}: {value}\n"

            if rst:
                option_text = "".join(
                    _leaf_option_line(key, value, "   ")
                    for key, value in video_options.items()
                )
                item["content"] = f".. youtube:: {record.id}\n{option_text}"
            else:
                option_text = "".join(
                    _leaf_option_line(key, value)
                    for key, value in video_options.items()
                )
                # Title/option text may contain fence characters; choose a safe fence.
                runs = re.findall(r"~+", option_text)
                fence = "~" * max(3, max((len(run) + 1 for run in runs), default=3))
                item["content"] = (
                    f"{fence}{{youtube}} {record.id}\n{option_text}{fence}\n"
                )
        else:
            item["link"] = record.url
            item["link-alt"] = f"Watch {' '.join(record.title.split())} on YouTube"
            item["img-top"] = THUMBNAIL_URL.format(id=record.id)
            # Without this the thumbnail is emitted as `alt=""`, which marks
            # it *decorative* -- so a screen reader announces nothing at all
            # for the one element that identifies the video. On a
            # thumbnail-mode page the image is the content, not decoration
            # (WCAG 2.1 SC 1.1.1).
            item["img-alt"] = f"Video thumbnail: {' '.join(record.title.split())}"
        return item

    def _render_grid(  # ruff: ignore[too-many-branches]
        self,
        records: list[CatalogRecord],
        mode: str,
        rst: bool,
        query: Query,
    ) -> str:
        """
        Emit one ``gallery-grid`` source block for the selected catalog.

        ``youtube-gallery`` deliberately does not create a second collection
        wrapper.  It turns typed YouTube records into ordinary gallery item
        mappings, then forwards presentation, grouping, pagination, and live
        browser controls to ``gallery-grid`` itself.  The resulting
        ``sk-collection`` node is therefore the same root used by a directly
        authored gallery.

        Parameters
        ----------
        records : list of CatalogRecord
            Filtered records in their final build-time sort order. Pagination
            is intentionally left to ``gallery-grid`` so its count/empty-state
            behavior remains authoritative.
        mode : str
            ``"embed"`` or ``"thumbnail"``.
        rst : bool
            Whether to emit reStructuredText rather than MyST.
        query : Query
            Supplies grouping and pagination options that ``gallery-grid``
            applies to the already-filtered records.

        Returns
        -------
        str
            Directive source ready for ``nested_parse``.
        """
        items = [self._card(record, mode, rst) for record in records]
        body = safe_dump(
            items, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        options: dict[str, Any] = {
            "grid-columns": self.options.get(
                "grid-columns", self.options.get("columns", "1 2 2 3")
            )
        }
        for key in ("class-card", "class-container"):
            if self.options.get(key):
                options[key] = self.options[key]
        for key in (*GRID_SPEC, *CARD_SPEC):
            if key in self.options:
                options[key] = self.options[key]

        # Build-time YouTube-specific predicates were already applied above.
        # Leave the generic collection operations to gallery-grid so grouping,
        # pagination, count text, and section fallback have one implementation.
        if query.group_by != "none":
            options["group-by"] = query.group_by
        if query.limit is not None:
            options["limit"] = query.limit
        if query.offset:
            options["offset"] = query.offset
        if "show-count" in self.options:
            options["show-count"] = None
        if "section-style" in self.options:
            options["section-style"] = self.options["section-style"]

        # Forward the exact same reader-control contract instead of wrapping
        # gallery-grid in a second enhanced collection. Flags are emitted as
        # valueless directive options, never as the string ``None``.
        for key in ("searchable", "interactive"):
            if key in self.options:
                options[key] = None
        for key in ("filter-fields", "sort-fields", "search-fields"):
            if key in self.options:
                options[key] = ",".join(self.options[key])
        for key in ("search-label", "collection-id"):
            if key in self.options:
                options[key] = self.options[key]

        def option_line(key: str, value: Any, indent: str = "") -> str:
            if value is None:
                return f"{indent}:{key}:"
            return f"{indent}:{key}: {value}"

        if rst:
            option_lines = "\n".join(
                option_line(key, value, "   ") for key, value in options.items()
            )
            indented = "\n".join(
                f"   {line}" if line.strip() else line for line in body.splitlines()
            )
            return f".. gallery-grid::\n{option_lines}\n\n{indented}\n"

        option_lines = "\n".join(
            option_line(key, value) for key, value in options.items()
        )
        # A colon fence cannot be closed by the backtick fences the embedded
        # players use, so the player markup cannot terminate this block early.
        return f":::::{{gallery-grid}}\n{option_lines}\n\n{body}\n:::::\n"

    def _parse_gallery_grid(self, source: str) -> list[nodes.Node]:
        """
        Parse delegated gallery source without losing section context.

        ``gallery-grid`` decides whether grouped output may use real sections
        by inspecting its parser parent.  A generic temporary container would
        therefore make a top-level youtube gallery look artificially nested.
        Mirror the original attachment capability in the temporary holder so
        gallery-grid makes the same section/rubric decision it would make when
        authored directly at this location.
        """
        holder: nodes.Element = (
            nodes.section() if sections_allowed(self) else nodes.container()
        )
        self.state.nested_parse(
            StringList(source.splitlines(), source="<youtube-gallery-grid>"),
            0,
            holder,
        )
        rendered = list(holder.children)
        self._mark_gallery_root(rendered)
        return rendered

    @staticmethod
    def _mark_gallery_root(rendered: list[nodes.Node]) -> None:
        """
        Tag gallery-grid's own collection root for compatibility.

        Older pages may target ``.youtube-gallery`` in custom CSS.  Preserve
        that class on the *same* ``sk-collection`` node returned by
        ``gallery-grid`` rather than keeping the former redundant outer root.
        """
        for node in rendered:
            if not isinstance(node, nodes.Element):
                continue
            classes = node.get("classes", [])
            if CONTAINER_CLASS in classes:
                if "youtube-gallery" not in classes:
                    classes.append("youtube-gallery")
                return

    def _render_list(self, records: list[CatalogRecord], rst: bool) -> str:
        """
        Emit a plain bullet list of links.

        Parameters
        ----------
        records : list of CatalogRecord
            Records to list.
        rst : bool
            Whether to emit reStructuredText rather than Markdown.

        Returns
        -------
        str
            Markup source ready for ``nested_parse``.

        Notes
        -----
        This mode exists for four-figure catalogs, where a grid of images is
        itself the performance problem. It renders one line per video with
        no external requests at all.
        """
        lines = []
        for record in records:
            suffix = ""
            if "show-duration" in self.options and record.duration is not None:
                suffix = f" — {_format_duration(record.duration)}"
            # Escape the delimiters that would otherwise terminate the link
            # label early when a title legitimately contains one.
            label = _literal_metadata(record.title)
            if rst:
                lines.append(f"* `{label} <{record.url}>`__{suffix}")
            else:
                lines.append(f"- [{label}]({record.url}){suffix}")
        return "\n".join(lines) + "\n"

    def _is_rst(self) -> bool:
        """
        Detect whether the calling document is reStructuredText.

        Uses Sphinx's own ``source_suffix`` mapping -- the same mechanism
        Sphinx uses to choose a parser -- so projects with custom suffix
        mappings are handled correctly.

        Returns
        -------
        bool
            ``True`` for a reStructuredText page.
        """
        from .._sphinx_gallery_grid.directive import (  # ruff: ignore[import-outside-top-level]
            RESTRUCTUREDTEXT,
            _source_format,
        )

        return _source_format(self.env) == RESTRUCTUREDTEXT

    def run(  # ruff: ignore[too-many-branches]
        self,
    ) -> list[nodes.Node]:
        """
        Execute the directive.

        Returns
        -------
        list of docutils.nodes.Node
            The rendered nodes, or a located error node for recognized catalog
            or option validation failures. Unexpected internal errors are not
            suppressed. Empty queries emit a paragraph and a build warning.
        """
        try:
            if (
                "columns" in self.options
                and "grid-columns" in self.options
                and self.options["columns"].split()
                != self.options["grid-columns"].split()
            ):
                raise CatalogError(
                    ":columns: and :grid-columns: disagree; use one value"
                )
            collection_kind, records = self._load_records()
            query = self._build_query()
            requested_view = self.options.get("view", "auto")
            if collection_kind == "channel" and requested_view == "videos":
                raise CatalogError(
                    ":view: videos cannot be built from a channels catalog; "
                    "use a videos catalog or omit :view:"
                )
            if self.options.get("mode") == "list":
                live_options = (
                    "interactive",
                    "searchable",
                    "filter-fields",
                    "sort-fields",
                    "search-fields",
                    "search-label",
                    "collection-id",
                )
                active_live = [key for key in live_options if key in self.options]
                if active_live:
                    rendered = ", ".join(f":{key}:" for key in active_live)
                    raise CatalogError(
                        ":mode: list is the static lightweight renderer and cannot "
                        f"use reader-side controls ({rendered}); use :mode: thumbnail, "
                        ":mode: embed, or :mode: auto, or remove those options"
                    )
        except CatalogError as exc:
            return [
                self.state_machine.reporter.error(
                    f"youtube-gallery: {exc}", line=self.lineno
                )
            ]

        # Selection has two phases when a video catalog is projected into
        # channels. YouTube-specific predicates describe *which videos* feed
        # the projection; sort/group/pagination describe the *visible content*
        # and therefore must apply after projection. Sorting videos first and
        # then deduplicating channels makes ``:sort: title`` mean video title,
        # which is observably wrong for a channel gallery.
        try:
            if requested_view == "channels" and collection_kind == "video":
                predicate_query = replace(
                    query, sort_by="none", group_by="none", limit=None, offset=0
                )
                matching_videos, _ = apply_query(records, predicate_query)
                projected = derive_channel_records(
                    [
                        record
                        for record in matching_videos
                        if isinstance(record, VideoRecord)
                    ]
                )
                if matching_videos and not projected:
                    raise CatalogError(
                        ":view: channels found matching videos but none carry a "
                        "stable channel_id or explicit @handle/channel URL; add "
                        "channel identity to the catalog"
                    )

                # Reuse the shared engine for channel-level ordering and field
                # validation, but do not reapply predicates that were defined
                # over the contributing videos. Pagination/grouping remain the
                # delegated gallery-grid's responsibility.
                presentation_query = replace(
                    query,
                    channel="",
                    playlist="",
                    tags=(),
                    match="",
                    match_regex="",
                    since=None,
                    until=None,
                    limit=None,
                    offset=0,
                )
                ordered, total = apply_query(projected, presentation_query)
                collection_kind = "channel"
            else:
                unpaged_query = replace(query, limit=None, offset=0)
                ordered, total = apply_query(records, unpaged_query)
                if requested_view == "channels":
                    collection_kind = "channel"
        except CatalogError as exc:
            return [
                self.state_machine.reporter.error(
                    f"youtube-gallery: {exc}", line=self.lineno
                )
            ]

        if total == 0:
            logger.warning(
                f"youtube-gallery: no items matched "
                f"({len(records)} in catalog, 0 after filtering).",
                location=self.get_location(),
            )
            # Do not create a youtube-gallery-specific empty-state branch for
            # card modes. Delegating an empty typed selection to gallery-grid
            # preserves the same collection root, metadata carrier, controls,
            # and ``No items matched.`` presentation as an authored empty
            # gallery. Explicit list mode remains the intentionally separate
            # lightweight renderer below.

        remaining = max(0, total - query.offset)
        visible_count = (
            min(remaining, query.limit) if query.limit is not None else remaining
        )
        rst = self._is_rst()
        if collection_kind == "channel":
            # A channel collection has one canonical card presentation:
            # gallery-grid's title-only stretched-link card. ``list`` remains
            # the intentionally lightweight non-card mode.
            mode = "list" if self.options.get("mode") == "list" else "thumbnail"
        else:
            mode = self._resolve_mode(visible_count)

        if mode == "list":
            selected = ordered[query.offset :]
            if query.limit is not None:
                selected = selected[: query.limit]
            if not selected:
                return [nodes.paragraph(text="No items matched.")]
            sections = group_records(selected, query)
            parts = [
                (label, self._render_list(group, rst)) for label, group in sections
            ]
            rendered = render_sections(
                self, parts, self.options.get("section-style", "auto"), logger
            )
            if "show-count" in self.options and total != len(selected):
                rendered.append(
                    nodes.paragraph(text=f"Showing {len(selected)} of {total} items.")
                )
            return rendered

        source = self._render_grid(
            ordered,
            mode,
            rst,
            query,
        )
        # One unlabelled parse returns gallery-grid's own sk-collection root.
        # No youtube-gallery-owned wrapper, metadata carrier, or controls are
        # created, so both directives literally share the same UI engine.
        return self._parse_gallery_grid(source)


#: Extensions this directive renders through. `youtube-gallery` emits
#: `gallery-grid` and `youtube` markup, so without these a page fails with
#: "Unknown directive type: 'gallery-grid'" -- an error naming a directive
#: the author never wrote, pointing at generated source they cannot see.
#:
#: Declaring them here means enabling `_sphinx_youtube_gallery` is enough: Sphinx
#: loads the rest. The two directives stay independently usable; it is only
#: this one that depends on them.
from importlib.util import resolve_name

from .._extension_setup import check_namespace

REQUIRED_EXTENSIONS = (
    resolve_name(".._sphinx_gallery_grid", __package__),
    resolve_name(".._sphinxcontrib_youtube", __package__),
)


def _ensure_extensions(app: Sphinx) -> None:
    """Load dependencies in the caller's namespace; propagate setup failures."""
    check_namespace(app, __package__.rsplit(".", 1)[0])
    for name in REQUIRED_EXTENSIONS:
        app.setup_extension(name)


def setup(app: Sphinx) -> dict[str, Any]:
    """
    Register the directive, its configuration values, and its dependencies.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application to extend.

    Returns
    -------
    dict
        Extension metadata declaring parallel read/write safety. The
        directive is pure with respect to the environment -- it only reads
        the catalog file and notes it as a dependency -- so both are safe.
    """
    _ensure_extensions(app)
    app.add_config_value("youtube_catalog_path", "", "env", types=[str])
    app.add_config_value(
        "youtube_catalog_max_embeds", DEFAULT_MAX_EMBEDS, "env", types=[int]
    )
    app.add_directive("youtube-gallery", YouTubeGalleryDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
