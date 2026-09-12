"""
Generic ``gallery-grid`` directive for structured card collections.

Generating a gallery of images that are all the same size is a common
pattern in documentation, and this can be cumbersome if the gallery is
generated programmatically. This directive wraps this particular use-case
in a helper-directive to generate it with a single YAML configuration file.

This vendored copy is maintained as a theme-independent gallery engine.
Domain adapters such as ``youtube-gallery`` normalize their records and delegate
rendering here so card structure and reader controls have one owner.

The directive renders its grid using `Sphinx Design
<https://sphinx-design.readthedocs.io>`__'s ``grid`` / ``grid-item-card``
directives. Sphinx Design's own directives are written once and consumed
identically from Markdown (MyST) and reStructuredText documents, so this
directive mirrors that behaviour: it looks at the markup language of the
page that invokes it and emits the matching directive syntax (fenced code
blocks with colon options for MyST, indented ``..`` directives with field
options for reStructuredText). Authors do not need to do anything special;
``{gallery-grid}`` in a ``.md`` file and ``.. gallery-grid::`` in a ``.rst``
file both work out of the box.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

# -- shared collection engine -------------------------------------------------
# Selection, grouping, browser metadata and assets are first-party siblings of
# this directive. There is intentionally no "upstream-only" fallback: one
# gallery-grid build must have one behavior, and youtube-gallery delegates to
# this exact implementation.
from .._sphinx_collection import (
    CONTAINER_CLASS,
    SEARCHABLE_CLASS,
    SECTION_STYLES,
    FilterError,
    Selection,
    apply_selection,
    ensure_assets,
    group_records,
    has_field,
    render_sections,
)
from .._sphinx_collection._browser import (
    collection_id,
    field_names,
    metadata_node,
    record_for_browser,
)
from .._sphinx_collection._presentation import CARD_SPEC, GRID_SPEC, forwarded
from .._sphinx_collection._yaml import (
    MAX_COLLECTION_ITEMS,
    BoundedYAMLError,
    load_bounded_yaml,
    read_bounded_utf8,
)

# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# -- Markup-language identifiers ---------------------------------------------
# These are the two markup languages this directive knows how to emit.
# They intentionally match the parser names Sphinx uses internally
# (see ``source_suffix`` in the Sphinx configuration docs) so that
# ``_get_source_format`` can return them directly.
MARKDOWN = "markdown"
RESTRUCTUREDTEXT = "restructuredtext"

# -- Templates for Markdown / MyST output ------------------------------------
# MyST fenced directives do not require their content to be indented, and
# directive options are written as a colon-fenced block immediately below
# the opening fence (enabled via the ``colon_fence`` MyST extension).
#
# The fence is a ``{fence}`` placeholder (tildes, e.g. ``~~~~``) rather
# than a fixed run of backticks: see :func:`_fence_for` for why a fixed
# *and* backtick-based fence is unsafe once title/option text or nested
# card content can itself contain backticks (e.g. inline code in a
# title, or a YouTube embed wrapped in a dropdown admonition).
TEMPLATE_GRID_MYST = """
{fence}{{grid}} {columns}
{options}

{content}

{fence}
"""

TEMPLATE_CARD_MYST = """
{fence}{{grid-item-card}} {title}
{options}

{content}
{fence}
"""

#: Minimum fence length for a ``grid-item-card`` block. Matches the
#: historical fixed length (four characters) so plain, unnested card
#: content renders byte-for-byte as before.
MIN_CARD_FENCE = 4

#: Minimum fence length for the outer ``grid`` block. Kept one longer
#: than :data:`MIN_CARD_FENCE` so a grid with a single, unnested card
#: still nests exactly as before.
MIN_GRID_FENCE = 5

# -- Templates for reStructuredText output -----------------------------------
# reStructuredText directives require their options and body to be
# consistently indented under the directive line, with a blank line
# separating the option block from the body. Nested directives (a
# ``grid-item-card`` inside a ``grid``) therefore need an *additional*
# level of indentation on top of their own internal indentation; this is
# applied once, uniformly, when the individual cards are assembled into
# the surrounding grid (see ``_indent`` and ``run``).
TEMPLATE_GRID_RST = """
.. grid:: {columns}
{options}

{content}
"""

TEMPLATE_CARD_RST = """
.. grid-item-card:: {title}
{options}

{content}
"""

#: Number of spaces used for each level of reStructuredText indentation.
RST_INDENT = "   "


def _strip_quotes(value: Any) -> Any:
    """
    Strip one layer of matching quote characters from a string, if present.

    MyST/Sphinx Design tolerates (and the theme's own MyST pages
    conventionally use) directive arguments and options wrapped in quotes,
    e.g. ``:grid-columns: "1 2 3 4"``. Plain reStructuredText directive
    arguments do not support that quoting and fail validation if the quotes
    are passed through literally, so any value taken from this directive's
    own options is normalized before being re-embedded in either output
    template.

    Parameters
    ----------
    value : Any
        The raw option value. Non-string values are returned unchanged.

    Returns
    -------
    Any
        ``value`` with a single matching pair of leading/trailing quotes
        removed, if there was one; otherwise ``value`` unchanged.
    """
    if (
        isinstance(value, str)
        and len(value) >= 2  # ruff: ignore[magic-value-comparison]
        and value[0] == value[-1]
        and value[0] in "'\""
    ):
        return value[1:-1]
    return value


def _indent(text: str, prefix: str = RST_INDENT) -> str:
    r"""
    Indent every non-blank line of ``text`` by ``prefix``.

    Blank lines are left untouched (no trailing whitespace is introduced)
    so the resulting reStructuredText stays clean and diff-friendly.

    Parameters
    ----------
    text : str
        The block of text to indent. May contain multiple lines.
    prefix : str, optional
        The string prepended to each non-blank line. Defaults to
        :data:`RST_INDENT` (three spaces), which lines up under a
        ``".. directive::"`` marker.

    Returns
    -------
    str
        The indented text, with the same number of lines as the input.

    Examples
    --------
    >>> _indent("a\n\nb")
    '   a\n\n   b'
    """
    return "\n".join(
        prefix + line if line.strip() else line for line in text.splitlines()
    )


def _max_run(text: str, char: str) -> int:
    """
    Find the length of the longest run of consecutive ``char`` in ``text``.

    Parameters
    ----------
    text : str
        Text to scan. May contain multiple lines.
    char : str
        The single character to look for runs of (e.g. ``"~"``).

    Returns
    -------
    int
        The length of the longest consecutive run of ``char``, or ``0``
        if there are none.
    """
    longest = 0
    current = 0
    for ch in text:
        if ch == char:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _fence_for(text: str, minimum: int, char: str = "~") -> str:
    """
    Build a fence long enough to safely wrap ``text``.

    A MyST/Markdown fenced block is closed by the first line whose run of
    fence characters is *at least as long* as the fence that opened it,
    and (per CommonMark) a **backtick**-fenced block's info string may
    not itself contain a backtick -- if it does, the opening line simply
    isn't recognized as a fence at all, and everything up to the next
    accidental fence-like line is silently mis-parsed as ordinary prose.
    That restriction does not apply to tilde fences, so this directive
    wraps its own ``grid``/``grid-item-card`` output in tildes rather
    than backticks: a card ``title`` (or a ``class-card``/
    ``class-container`` option) containing an inline code span like
    `` `code` `` -- entirely reasonable content for a title -- would
    otherwise silently break the surrounding card with no error raised.

    Using tildes also means content nested inside a card (for example a
    ``{youtube}`` embed wrapped in a ``{admonition}`` dropdown, which
    conventionally use backtick fences) can never prematurely close our
    fence by accident, since a fence can only be closed by a line of the
    *same* fence character. The length is still computed dynamically as a
    safety net for the rarer case of content that nests its own
    tilde-fenced directive: always at least ``minimum`` characters
    (matching this directive's historical fence lengths so simple,
    unnested content is unaffected), and always at least one character
    longer than the longest run of ``char`` already present in ``text``.

    Parameters
    ----------
    text : str
        The content (and any options/title) that will be placed inside
        the fence.
    minimum : int
        The smallest acceptable fence length, e.g. :data:`MIN_CARD_FENCE`
        or :data:`MIN_GRID_FENCE`.
    char : str, optional
        The fence character to use. Defaults to ``"~"``.

    Returns
    -------
    str
        A string of ``char`` safe to use as the opening/closing fence.
    """
    return char * max(minimum, _max_run(text, char) + 1)


def _coerce_items(payload: Any, origin: str) -> list[dict[str, Any]]:
    """
    Validate that a parsed YAML payload is a list of item mappings.

    The directive body is author-supplied YAML, so every shape a user can
    plausibly type has to be handled explicitly rather than assumed. Before
    this check existed, an empty directive body parsed to ``None`` and a
    mapping body parsed to a ``dict``; both were iterated directly and
    aborted the entire Sphinx build with an unlocated ``TypeError`` /
    ``AttributeError`` deep inside the directive.

    Parameters
    ----------
    payload : Any
        The value returned by ``yaml.safe_load``.
    origin : str
        Human-readable description of where the YAML came from (a file path
        or ``"directive content"``), used in the error message.

    Returns
    -------
    list of dict
        The validated item list. An empty body yields an empty list, which
        the caller renders as an empty grid rather than an error.

    Raises
    ------
    ValueError
        If ``payload`` is neither ``None`` nor a list of mappings. The
        message names the offending shape and index so the author can find
        it without reading a traceback.
    """
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            f"{origin}: expected a YAML list of items, got "
            f"{type(payload).__name__}. Each gallery item must be a "
            f"list entry, e.g. '- title: My card'."
        )
    if len(payload) > MAX_COLLECTION_ITEMS:
        raise ValueError(
            f"{origin}: gallery contains {len(payload):,} items; the limit is "
            f"{MAX_COLLECTION_ITEMS:,}. Split very large collections across pages."
        )
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                f"{origin}: item {index} is a {type(item).__name__}, "
                f"expected a mapping of option names to values, "
                f"e.g. '- title: My card'."
            )
    return payload


# -- scikit-plots local patch: card options vs item data -----------------------
# Upstream forwards *every* remaining YAML key to `grid-item-card` as a
# directive option. Sphinx Design rejects the ones it does not recognise, and
# the rejected card is then dropped from the grid entirely -- so a single
# data field such as `category:` silently removes that item from the page,
# with only a `myst.directive_option` warning that reads like a typo report.
#
# That also made the collection semantics impossible: filtering and sorting
# need per-item *data*, and there was no way to carry any without destroying
# the card. Splitting the two is what lets an item be both renderable and
# queryable.
#
# Sourced from Sphinx Design's `GridItemCardDirective.option_spec`. Read at
# runtime when Sphinx Design is importable, so the set tracks the installed
# version instead of drifting against this hard-coded fallback.
_FALLBACK_CARD_OPTIONS = frozenset(
    {
        "class-body",
        "class-card",
        "class-footer",
        "class-header",
        "class-img-bottom",
        "class-img-top",
        "class-item",
        "class-title",
        "columns",
        "img-alt",
        "img-background",
        "img-bottom",
        "img-top",
        "link",
        "link-alt",
        "link-type",
        "margin",
        "padding",
        "shadow",
        "text-align",
        "width",
        "name",
    }
)


@lru_cache(maxsize=1)
def _card_option_names() -> frozenset[str]:
    """
    Return the option names ``grid-item-card`` accepts.

    Returns
    -------
    frozenset of str
        Option names read from the installed Sphinx Design, falling back to
        :data:`_FALLBACK_CARD_OPTIONS` if its internals move. The fallback
        keeps this directive working against a future Sphinx Design whose
        module layout changed, at the cost of not seeing newly added
        options -- which degrades to the previous behaviour for those keys
        rather than breaking the build.
    """
    try:
        from sphinx_design.grids import (  # ruff: ignore[import-outside-top-level]
            GridItemCardDirective,
        )

        return frozenset(GridItemCardDirective.option_spec) | {"name"}
    # depends on sphinx_design internals
    except Exception:  # pragma: no cover  # ruff: ignore[blind-except]
        return _FALLBACK_CARD_OPTIONS


#: Item keys this directive consumes itself, before options are built.
_METADATA_ONLY_KEY = "_sk_collection_metadata_only"
_BROWSER_TITLE_KEY = "_sk_collection_browser_title"
_RESERVED_ITEM_KEYS = frozenset(
    {"title", "header", "image", "content", _METADATA_ONLY_KEY}
)
# -- end scikit-plots local patch ----------------------------------------------


# -- scikit-plots local patch: confine data files to the source tree ---------
def _confined_path(directive, reference: str) -> Path:
    """
    Resolve a data-file reference and confine it to the source tree.

    Parameters
    ----------
    directive : SphinxDirective
        The calling directive, used for its source info and environment.
    reference : str
        The path as written by the page author, relative to the current
        document.

    Returns
    -------
    pathlib.Path
        The resolved, confined path.

    Raises
    ------
    ValueError
        If the resolved path lies outside the Sphinx source directory. A
        documentation build has no reason to read anywhere else, and an
        unconfined path means any contributor who can edit a ``.md`` file
        can make the build read ``/etc/passwd``, a CI secrets file, or a
        private key -- and then quote it back inside a build error message
        that lands in a public log.

    Notes
    -----
    Symlinks are resolved *before* the check, so a symlink inside the source
    tree pointing outside it is rejected too. Checking the unresolved path
    would make the confinement trivially bypassable.
    """
    source, _ = directive.get_source_info()
    candidate = (Path(source).parent / reference).resolve()
    roots = [Path(directive.env.srcdir).resolve()]
    confdir = getattr(directive.env.app, "confdir", None)
    if confdir:
        roots.append(Path(confdir).resolve())
    for root in roots:
        if candidate == root or root in candidate.parents:
            return candidate
    raise ValueError(
        f"{reference!r} resolves to {candidate}, which is outside the "
        f"documentation source directory ({roots[0]}). Data files must live "
        f"inside the project."
    )


# -- end scikit-plots local patch --------------------------------------------


def _source_format(env: Any) -> str:
    """
    Return the generated-markup language implied by Sphinx ``source_suffix``.

    Mapping form is authoritative (including unusual mappings for ``.rst``).
    Legacy string/list forms mean Sphinx's default reStructuredText parser for
    every listed suffix.  Only when configuration is unavailable do we fall
    back to the conventional ``.rst``/other split.
    """
    source_path = Path(env.doc2path(env.docname))
    suffix = source_path.suffix
    source_suffix = env.config.source_suffix
    if isinstance(source_suffix, dict):
        parser_name = source_suffix.get(suffix)
        return RESTRUCTUREDTEXT if parser_name == RESTRUCTUREDTEXT else MARKDOWN
    if isinstance(source_suffix, (str, list, tuple)):
        return RESTRUCTUREDTEXT
    return RESTRUCTUREDTEXT if suffix == ".rst" else MARKDOWN


class GalleryGridDirective(SphinxDirective):
    """
    A directive to show a gallery of images and links in a Bootstrap grid.

    The grid can be generated from a YAML file that contains a list of items, or
    from the content of the directive (also formatted in YAML). Use the parameter
    "class-card" to add an additional CSS class to all cards. When specifying the grid
    items, you can use all parameters from "grid-item-card" directive to customize
    individual cards + ["image", "header", "content", "title"].

    This directive can be used from both MyST (Markdown) and reStructuredText
    pages. It detects which markup language the calling page uses and
    generates the matching Sphinx Design syntax automatically -- no extra
    configuration is required from the page author.
    """

    name = "gallery-grid"
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec: ClassVar[dict[str, Any]] = {
        **GRID_SPEC,
        **CARD_SPEC,
        # A class to be added to the resulting container
        "grid-columns": directives.unchanged,
        "class-container": directives.unchanged,
        "class-card": directives.unchanged,
        # -- scikit-plots local patch: generic selection ---------------------
        # Filtering, sorting, grouping and pagination are not gallery
        # concerns; they are collection concerns that every list-of-records
        # directive needs. The engine lives in `_sphinx_ext._sphinx_collection` and
        # knows nothing about galleries, so this directive, `youtube-gallery`
        # and any future collection directive share one implementation and
        # cannot drift apart in behaviour.
        "filter": directives.unchanged,
        "sort": directives.unchanged,
        "group-by": directives.unchanged,
        "limit": directives.nonnegative_int,
        "offset": directives.nonnegative_int,
        "show-count": directives.flag,
        "section-style": lambda argument: directives.choice(
            (argument or "auto").strip().lower(), SECTION_STYLES
        ),
        "searchable": directives.flag,
        "interactive": directives.flag,
        "filter-fields": field_names,
        "sort-fields": field_names,
        "search-label": directives.unchanged,
        "collection-id": collection_id,
        "search-fields": field_names,
        # -- end scikit-plots local patch ------------------------------------
    }

    def _get_source_format(self) -> str:
        """
        Determine the markup language of the document invoking this directive.

        Looks up the current document's file extension in
        ``conf.py``'s ``source_suffix`` mapping (the mechanism Sphinx itself
        uses to choose a parser) so that projects with non-default suffix
        mappings (e.g. treating ``.txt`` as reStructuredText) are still
        handled correctly. Falls back to a plain ``.rst`` / everything-else
        check if no mapping is configured.

        Returns
        -------
        str
            Either :data:`RESTRUCTUREDTEXT` or :data:`MARKDOWN`.
        """
        return _source_format(self.env)

    def _build_options_block(self, options: dict[str, Any], *, rst: bool) -> str:
        """
        Render a directive's field options (e.g. ``:link: ...``) as text.

        Parameters
        ----------
        options : dict
            Mapping of option name to option value.
        rst : bool
            Whether to indent the result for reStructuredText. When
            ``False``, the MyST colon-fence formatting is used instead.

        Returns
        -------
        str
            The formatted, ready-to-embed options block.
        """
        lines = "\n".join(
            f":{key}:" if value is None else f":{key}: {value}"
            for key, value in options.items()
        )
        if rst:
            return _indent(lines)
        # MyST colon-fence blocks need a trailing hard line-break so the
        # option block is kept apart from the body that follows it.
        return f"{lines}  \n"

    def _build_card(self, item: dict[str, Any], *, rst: bool) -> str:
        """
        Render a single gallery item as a ``grid-item-card`` directive.

        Parameters
        ----------
        item : dict
            One entry from the gallery YAML/content, e.g. with ``title``,
            ``header``, ``image``, and ``content`` keys plus any raw
            ``grid-item-card`` options (``link``, ``link-alt``, etc.).
        rst : bool
            Whether to render reStructuredText (``True``) or MyST/Markdown
            (``False``) syntax.

        Returns
        -------
        str
            The fully rendered card, ready to be embedded in the
            surrounding grid.
        """
        # Work on a copy: the item is also the record the selection engine
        # filtered and sorted on, and mutating it in place would make a card
        # unrenderable the second time it appears -- which a `:group-by:` on
        # a list field legitimately does.
        item = dict(item)
        token = f"sk-collection-item-{self.env.new_serialno('sk-collection-item')}"
        # Typed adapters may need source escaping for the nested directive
        # argument while still exposing the human title verbatim to local
        # search/sort metadata. Keep those representations separate so an
        # RST escape such as ``C\+\+`` never changes browser semantics.
        browser_item = dict(item)
        browser_title = browser_item.pop(_BROWSER_TITLE_KEY, None)
        if browser_title is not None:
            browser_item["title"] = browser_title
        self._browser_records[token] = record_for_browser(browser_item, self.options)
        item.pop(_BROWSER_TITLE_KEY, None)
        title = item.pop("title", "")
        header = item.pop("header", None)
        image = item.pop("image", None)
        content = item.pop("content", None)
        metadata_only = item.pop(_METADATA_ONLY_KEY, ())
        if not isinstance(metadata_only, (list, tuple, set, frozenset)):
            metadata_only = ()
        metadata_only = {str(key) for key in metadata_only}

        # scikit-plots local patch: keep only keys Sphinx Design accepts.
        # Everything else is item *data* -- available to `:filter:`,
        # `:sort:` and `:group-by:`, and simply not rendered. Forwarding it
        # would make Sphinx Design reject and then silently drop the card.
        allowed = _card_option_names()
        item = {
            key: value
            for key, value in item.items()
            if key in allowed and key not in metadata_only
        }

        # optional parameter that influences all cards
        if "class-card" in self.options:
            item["class-card"] = _strip_quotes(self.options["class-card"])

        item.update(forwarded(self.options, "card-"))
        item["class-card"] = (str(item.get("class-card", "")) + " " + token).strip()

        if rst:
            body_parts = []
            if header:
                body_parts.append(f"{header}\n\n^^^\n")
            if image:
                body_parts.append(f".. image:: {image}\n")
            if content:
                body_parts.append(f"{content}\n")
            body = _indent("\n".join(body_parts))
            options = self._build_options_block(item, rst=True)
            return TEMPLATE_CARD_RST.format(options=options, content=body, title=title)

        body_parts = []
        if header:
            body_parts.append(f"{header}  \n^^^  \n")
        if image:
            body_parts.append(f"![image]({image})  \n")
        if content:
            body_parts.append(f"{content}  \n")
        body = "".join(body_parts)
        options = self._build_options_block(item, rst=False)
        # The fence must be longer than any backtick run already present
        # in the title, options, or body -- see `_fence_for` for why a
        # fixed-length fence would be unsafe here.
        fence = _fence_for(f"{title}\n{options}\n{body}", MIN_CARD_FENCE)
        return TEMPLATE_CARD_MYST.format(
            fence=fence, options=options, content=body, title=title
        )

    def run(  # ruff: ignore[too-many-branches, too-many-return-statements]
        self,
    ) -> list[nodes.Node]:
        """Create the gallery grid."""
        self._browser_records = {}
        if self.arguments:
            # If an argument is given, assume it's a path to a YAML file
            # Parse it and load it into the directive content
            # scikit-plots local patch: confined resolution (see above).
            try:
                path_data = _confined_path(self, self.arguments[0])
            except ValueError as exc:
                return [
                    self.state_machine.reporter.error(
                        f"gallery-grid: {exc}", line=self.lineno
                    )
                ]
            if not path_data.exists():
                # A missing data file means the entire gallery silently
                # disappears, so this must be a warning (visible in CI, and
                # fatal under `-W`), never an info-level log line.
                logger.warning(
                    f"gallery-grid: no grid data found at {path_data}.",
                    location=self.get_location(),
                )
                return [nodes.paragraph(text=f"No grid data found at {path_data}.")]
            # Register the data file as a build dependency so that editing it
            # invalidates this document. Without this, Sphinx has no idea the
            # page derives from the YAML: an edit leaves
            # `updating environment: 0 changed` and the stale HTML is served
            # until someone runs a clean build.
            #
            # Only *existing* files are registered. Sphinx treats a dependency
            # it cannot stat as permanently out of date, so noting a missing
            # path would make this document re-read on every single build and
            # a no-op rebuild would stop being a no-op. A document whose data
            # file is absent is already re-read each build for that same
            # reason, so a file appearing later is still picked up.
            self.env.note_dependency(str(path_data))
            # scikit-plots local patch: a data file saved as latin-1 or
            # cp1252 otherwise raised an unhandled `UnicodeDecodeError` that
            # aborted the whole build with a traceback inviting the author to
            # file a Sphinx bug. It is an input problem and belongs in a
            # located directive error, like every other malformed input.
            try:
                yaml_string = read_bounded_utf8(path_data, str(path_data))
            except BoundedYAMLError as exc:
                return [
                    self.state_machine.reporter.error(
                        f"gallery-grid: {exc}",
                        line=self.lineno,
                    )
                ]
            origin = str(path_data)
        else:
            yaml_string = "\n".join(self.content)
            origin = "gallery-grid directive content"

        rst = self._get_source_format() == RESTRUCTUREDTEXT

        # Malformed YAML is author error, not an internal fault: report it
        # through docutils so the message carries a file and line number and
        # the rest of the build still completes.
        try:
            items = _coerce_items(load_bounded_yaml(yaml_string, origin), origin)
        except (BoundedYAMLError, ValueError) as exc:
            return [
                self.state_machine.reporter.error(
                    f"gallery-grid: {exc}", line=self.lineno
                )
            ]

        # -- scikit-plots local patch: apply the generic selection ----------
        # Reader-side field lists should fail where they are authored when a
        # path exists nowhere in a non-empty collection. A field may still be
        # absent from some records; heterogeneous metadata is normal.
        if items:
            for option_name in ("filter-fields", "sort-fields", "search-fields"):
                for field_name in self.options.get(option_name, ()):
                    if not any(has_field(item, field_name) for item in items):
                        return [
                            self.state_machine.reporter.error(
                                f"gallery-grid: :{option_name}: field {field_name!r} "
                                "does not exist in any gallery record",
                                line=self.lineno,
                            )
                        ]

        try:
            selection = Selection.from_text(
                filter_text=self.options.get("filter", ""),
                sort_text=self.options.get("sort", ""),
                group_by=self.options.get("group-by", ""),
                limit=self.options.get("limit"),
                offset=self.options.get("offset", 0),
            )
            selected, total = apply_selection(items, selection)
            sections = group_records(selected, selection)
        except FilterError as exc:
            return [
                self.state_machine.reporter.error(
                    f"gallery-grid: {exc}", line=self.lineno
                )
            ]

        if selection.terms and not selected:
            # An empty result is far more often a typo in a filter than a
            # deliberately empty gallery, so it is surfaced -- as a warning,
            # leaving the page renderable.
            logger.warning(
                f"gallery-grid: no items matched the filter ({len(items)} in source).",
                location=self.get_location(),
            )

        # Real document sections where the context allows one, rubrics where
        # it does not -- decided per invocation, never an error. See
        # `_sphinx_collection.sections`.
        parts = [
            (label, self._render_grid(group, rst=rst)) for label, group in sections
        ]
        rendered = render_sections(
            self, parts, self.options.get("section-style", "auto"), logger
        )

        if "show-count" in self.options and total != len(selected):
            rendered.append(
                nodes.paragraph(text=f"Showing {len(selected)} of {total} items.")
            )

        # An empty result previously rendered *nothing* -- the reader saw a
        # heading followed by blank space, indistinguishable from a broken
        # page. The build log said why; the reader never sees the build log.
        if not selected:
            rendered.append(
                nodes.paragraph(
                    text="No items matched.",
                    classes=["sk-collection-empty"],
                )
            )

        # Wrap in a marked container so the browser enhancements can scope
        # themselves. Harmless when the assets are absent: it is a plain div.
        classes = [CONTAINER_CLASS]
        if "searchable" in self.options or "interactive" in self.options:
            classes.append(SEARCHABLE_CLASS)
        wrapper = nodes.container(classes=classes)
        if "searchable" in self.options or "interactive" in self.options:
            # Carried in a hidden node, not a `data-` attribute: docutils'
            # HTML writer emits only known attributes on a container, so a
            # custom one is silently dropped.
            wrapper += nodes.paragraph(
                text=self.options.get("search-label") or "Filter this gallery",
                classes=["sk-collection-label"],
            )
        wrapper += metadata_node(self._browser_records, self.options)
        wrapper += rendered
        return [wrapper]
        # -- end scikit-plots local patch ------------------------------------

    def _render_grid(self, items: list[dict[str, Any]], rst: bool) -> str:
        """
        Emit Sphinx Design grid source for one list of items.

        Split out of :meth:`run` so a grouped gallery can emit one grid per
        section while every grid is built by exactly one code path.

        Parameters
        ----------
        items : list of dict
            The items for this grid.
        rst : bool
            Whether the calling page is reStructuredText.

        Returns
        -------
        str
            Directive source ready for ``nested_parse``.
        """
        grid_items = [self._build_card(item, rst=rst) for item in items]

        # Prep the options that influence the grid container overall
        class_container = _strip_quotes(self.options.get("class-container", ""))
        class_ = f"gallery-directive {class_container}"
        container_options = {"gutter": 2, "class-container": class_}
        grid_options = forwarded(self.options, "grid-")
        extra_classes = grid_options.pop("class-container", "")
        container_options.update(grid_options)
        container_options["class-container"] = (class_ + " " + extra_classes).strip()
        columns = _strip_quotes(self.options.get("grid-columns", "1 2 3 4"))

        if rst:
            options_str = _indent(
                "\n".join(
                    f":{key}:" if value is None else f":{key}: {value}"
                    for key, value in container_options.items()
                )
            )
            content_str = _indent("\n".join(grid_items))
            grid_directive = TEMPLATE_GRID_RST.format(
                columns=columns, options=options_str, content=content_str
            )
        else:
            options_str = (
                "\n".join(
                    f":{key}:" if value is None else f":{key}: {value}"
                    for key, value in container_options.items()
                )
                + "  \n"
            )
            content_str = "\n".join(grid_items)
            # `content_str` already contains each card's own fence (sized
            # by `_build_card`), so basing the grid's fence on this
            # combined text guarantees it is longer than every fence
            # nested inside it, however deep that nesting goes.
            fence = _fence_for(f"{options_str}\n{content_str}", MIN_GRID_FENCE)
            grid_directive = TEMPLATE_GRID_MYST.format(
                fence=fence, columns=columns, options=options_str, content=content_str
            )

        return grid_directive

    def _parse(self, grid_directive: str) -> list[nodes.Node]:
        """
        Parse generated markup in the calling page's own language.

        Parameters
        ----------
        grid_directive : str
            Generated directive source.

        Returns
        -------
        list of docutils.nodes.Node
            The parsed nodes.
        """
        # Parse content as a directive so Sphinx Design processes it, using
        # whichever parser (MyST or reStructuredText) the calling page uses.
        # ``nested_parse`` requires a proper ``StringList`` (one entry per
        # line): the MyST bridge tolerates a single multi-line string, but
        # docutils' own reStructuredText state machine does not, so a real
        # ``StringList`` is required for both to behave correctly.
        container = nodes.container()
        content = StringList(grid_directive.splitlines(), source="<gallery-grid>")
        self.state.nested_parse(content, 0, container)

        # scikit-plots local patch: a grouped gallery emits a rubric and a
        # grid per section, so every child is returned rather than only the
        # first. For the ungrouped case this is the single Sphinx Design
        # container it always was.
        return list(container.children)


def _register_assets(app: Sphinx) -> None:
    """
    Register the shared browser assets, if the engine is available.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application.

    Notes
    -----
    The collection engine is a required sibling of this gallery extension, so
    asset registration has one supported path.
    """
    ensure_assets(app)


def setup(app: Sphinx) -> dict[str, Any]:  # ruff: ignore[undocumented-param]
    """
    Add custom configuration to sphinx app.

    Parameters
    ----------
    app: the Sphinx application

    Returns
    -------
    the 2 parallel parameters set to ``True``.
    """
    from .._extension_setup import (  # ruff: ignore[import-outside-top-level]
        check_namespace,
    )

    check_namespace(app, __package__.rsplit(".", 1)[0])
    app.setup_extension("sphinx_design")

    app.add_directive("gallery-grid", GalleryGridDirective)
    # scikit-plots local patch: browser enhancements (lazy images, optional
    # reader-side filtering). Connected to `builder-inited` so `app.outdir`
    # exists by the time the assets are written.
    app.connect(
        "builder-inited",
        lambda a: _register_assets(a),  # ruff: ignore[unnecessary-lambda]
    )

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
