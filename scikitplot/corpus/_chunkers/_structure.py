# scikitplot/corpus/_chunkers/_structure.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Structure-aware chunking of Markdown and reStructuredText (Run 4, R4-2).

:class:`StructureChunker` reads a document's headings and code blocks and
chunks by them: code blocks are their own chunks, prose is chunked within
its section, and every chunk carries the path of headings above it.

Notes
-----
**User.** The markup is given, not guessed (``markup="markdown"`` or
``"rst"``). Recognised:

* Markdown (CommonMark subset): ATX headings (``# Title``, with the
  CommonMark closing-sequence rule), setext headings (a paragraph
  underlined with ``===`` or ``---``), fenced code blocks (```` ``` ```` or
  ``~~~``, closed by the same character at least as long; an unclosed fence
  runs to the end, as in CommonMark) and, with ``front_matter=True``, YAML
  front matter. Lines end at ``\n``, ``\r\n`` or ``\r`` only. Indented
  code blocks are prose (they cannot be told from an indented list
  continuation without a full parser); a thematic break (``***``,
  ``- - -``) ends a paragraph; HTML blocks are not parsed
  (a heading line inside one is a heading). Containers are not parsed, so a
  paragraph with a line that starts a block quote (``>``) or a list item
  (``-``, ``+``, ``*`` or ``1.``/``1)``, then a space) is never underlined
  into a heading, nor is one that starts indented by four columns: there
  the underline is a thematic break or text, and it stays prose.
* reStructuredText, following docutils 0.22: section titles (an
  underline, with or without an equal overline, of one punctuation
  character, at least as long as the title; levels in order of first
  appearance of each style). A line docutils reads as a bullet item, a
  field, an option with a description, a doctest, a line block, a table
  border or explicit markup is not a title, whatever follows it.
  ``.. code-block::`` / ``.. code::`` / ``.. sourcecode::`` directives are
  code. So are ``::`` literal blocks: the body indented deeper than the text
  of the line that ends with ``::`` (after the markers on a list item's
  first line; on the first line of explicit markup, a field or an option,
  deeper than the least indentation of the lines that follow), or else
  quoted lines that start with one punctuation character. The ``::`` of a
  bare directive line (``.. note::``) or of a directive option is syntax,
  and a comment, a line block or a doctest never starts a literal block.
  Explicit markup ends at the first line that is not indented deeper. Tabs
  are 8 columns; tables are prose; titles are their source text (inline
  markup is not rendered). Lines end where :meth:`str.splitlines` ends
  them, except at ``\v`` and ``\f``, which docutils reads as spaces.

No heading is dropped (unless ``min_length`` drops it): a heading is in the
text of the prose chunk that follows it (``include_heading=True``), or else a
chunk of its own.

**Developer.** Like every built-in chunker, spans first (R3-3): blocks are
spans of the source, prose is split by :func:`recursive_spans`, and each
chunk's text is ``source[context_span]``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

from .._schema import ChunkingStrategy
from .._types import Chunk, ChunkerConfig, ChunkResult
from ._recursive import (
    DEFAULT_SEPARATORS,
    _BatchMixin,
    _bounds,
    _coerce_separators,
    _positive,
    _text_or_raise,
    _validate_separators,
    recursive_spans,
)
from ._spans import Span, strip_span

logger = logging.getLogger(__name__)

__all__ = [
    "StructureChunker",
    "StructureChunkerConfig",
]

#: Accepted ``markup`` values.
MARKUPS: tuple[str, ...] = ("markdown", "rst")

#: Code is cut only at blank lines, then lines, then characters.
_CODE_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "")

# Only the opening sequence is a regex; the rest of an ATX line is handled
# by str methods, so a long run of spaces or "#" costs linear time.
_ATX_OPEN = re.compile(r" {0,3}(#{1,6})(?=[ \t]|$)")
_SETEXT = re.compile(r" {0,3}(=+|-+)[ \t]*")
_FENCE = re.compile(r"( {0,3})(`{3,}|~{3,})(.*)")
_RST_ADORNMENT = re.compile(r"([!-/:-@\[-`{-~])\1*")
_RST_CODE = re.compile(r"(\s*)\.\. (?:code-block|code|sourcecode)::.*")
_MD_BREAK = re.compile(r"\r\n|\r|\n")
_RST_BREAK = re.compile(r"\r\n|[\n\r\x1c\x1d\x1e\x85\u2028\u2029]")
# A line that opens a block quote or a list item (CommonMark containers).
_MD_CONTAINER = re.compile(r" {0,3}(?:>|[-+*](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$))")
_FRONT_MATTER_CLOSE = ("---", "...")
_MD_THEMATIC = re.compile(r" {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*")
#: A Markdown line indented this many columns opens indented code, so it
#: cannot start a paragraph that is underlined into a heading.
_INDENTED_CODE_COLUMNS = 4
_RST_EXPLICIT = re.compile(r"\s*\.\.(?:\s|$)")
# The docutils (0.22) Body-state patterns tried before a section title: a
# line matching one is a bullet item, field, option, doctest, line block,
# table, explicit markup or anonymous target, whatever follows it. (An
# enumerator is tried too, but over an underline docutils reads it as text;
# so is an option without a description, which has no body there.)
_RST_OPTARG = r"(?:[a-zA-Z][a-zA-Z0-9_-]*|<[^<>]+>)"
_RST_OPTION = (
    rf"(?:[-+][a-zA-Z0-9](?: ?{_RST_OPTARG})?"
    rf"|(?:--|/)[a-zA-Z0-9][a-zA-Z0-9_-]*(?:[ =]{_RST_OPTARG})?)"
)
_RST_FIELD_PAT = r":(?![: ])(?:[^:\\]|\\.|:(?!(?:[ `]|$)))*(?<! ):(?:[ \t]+|$)"
_RST_OPTION_ITEM_PAT = rf"{_RST_OPTION}(?:, {_RST_OPTION})*  +\S"
#: Alternatives of _RST_NOT_TITLE, in docutils' order.
_RST_NOT_TITLE_ALTERNATIVES: tuple[str, ...] = (
    r"[-+*\u2022\u2023\u2043](?:[ \t]+|$)",
    _RST_FIELD_PAT,
    _RST_OPTION_ITEM_PAT,
    r">>>(?:[ \t]+|$)",
    r"\|(?:[ \t]+|$)",
    r"\+-[-+]+-\+ *$",
    r"=+(?: +=+)+ *$",
    r"\.\.(?:[ \t]+|$)",
    r"__(?:[ \t]+|$)",
)
_RST_NOT_TITLE = re.compile("|".join(_RST_NOT_TITLE_ALTERNATIVES))
# A field or option item: docutils reads its body with the indentation of
# the lines that follow (get_first_known_indented), like explicit markup.
_RST_FIELD_OR_OPTION = re.compile(rf"\s*(?:{_RST_FIELD_PAT}|{_RST_OPTION_ITEM_PAT})")
# A line block or doctest: its "::" never introduces a literal block.
_RST_NO_LITERAL = re.compile(r"\s*(?:>>>|\|)(?:[ \t]+|$)")
# Explicit markup that is not a comment: a footnote or citation, a target, a
# substitution definition or a directive (docutils' simplename).
_RST_NAME = r"(?:(?!_)\w)+(?:[-._+:](?:(?!_)\w)+)*"
_RST_CONSTRUCT = re.compile(
    rf"\s*\.\.[ \t]+(?:\[[^\]\s]+\]|_|\|[^|]+\|[ \t]+{_RST_NAME} ?::|{_RST_NAME} ?::)"
)
# A directive's first line; the field lines that follow it are its options.
_RST_DIRECTIVE = re.compile(rf"\s*\.\.[ \t]+(?:\|[^|]+\|[ \t]+)?{_RST_NAME} ?::")
_RST_FIELD = re.compile(rf"\s*{_RST_FIELD_PAT}")
# A directive line with nothing after "::": the "::" is directive syntax.
_RST_BARE_DIRECTIVE = re.compile(
    rf"\s*\.\.[ \t]+(?:\|[^|]+\|[ \t]+)?{_RST_NAME} ?::[ \t]*"
)
# One or more bullet or enumerated list markers (nested items on one line):
# the item's text starts after the last.
_RST_ITEMS = re.compile(
    r"\s*(?:(?:[-*+\u2022\u2023\u2043]|\(?(?:\d+|#|[A-Za-z]|[ivxlcdmIVXLCDM]{2,})"
    r"[.)])[ \t]+)+"
)


def _rst_items_match(text: str) -> re.Match[str] | None:
    """Match a reStructuredText list marker after leading indentation."""
    start = len(text) - len(text.lstrip())
    return _RST_ITEMS.match(text, start)


class _Line(NamedTuple):
    start: int  # offset of the first character
    end: int  # offset after the line break
    text: str  # without the line break


class _Block(NamedTuple):
    kind: str  # "heading", "code", "prose" or "front_matter"
    start: int
    end: int
    level: int = 0
    title: str = ""


def _lines(text: str, breaks: re.Pattern[str]) -> list[_Line]:
    """Split *text* into lines at the matches of *breaks*."""
    out: list[_Line] = []
    position = 0
    for match in breaks.finditer(text):
        out.append(_Line(position, match.end(), text[position : match.start()]))
        position = match.end()
    if position < len(text):
        out.append(_Line(position, len(text), text[position:]))
    return out


def _md_blank(line: _Line) -> bool:
    # CommonMark: a blank line holds only spaces and tabs.
    return not line.text.strip(" \t")


def _indent_columns(text: str) -> int:
    """Leading indentation in columns, a tab advancing to the next 4."""
    columns = 0
    for char in text:
        if char == " ":
            columns += 1
        elif char == "\t":
            columns += 4 - columns % 4
        else:
            break
    return columns


def _atx(text: str) -> tuple[int, str] | None:
    """Return ``(level, title)`` of an ATX heading line, else ``None``.

    The closing sequence of ``#`` is dropped only when a space or tab
    precedes it, or when it is all that follows the opening (CommonMark).
    """
    opening = _ATX_OPEN.match(text)
    if opening is None:
        return None
    rest = text[opening.end() :].strip(" \t")
    body = rest.rstrip("#")
    if not body:
        title = ""
    elif len(body) < len(rest) and body[-1] in " \t":
        title = body.rstrip(" \t")
    else:
        title = rest
    return len(opening.group(1)), title


def _markdown_blocks(lines: list[_Line], front_matter: bool) -> list[_Block]:
    """Return heading, code, prose and front-matter blocks of Markdown."""
    blocks: list[_Block] = []
    k = 0
    paragraph_start: int | None = None  # line index of the open paragraph
    setext_ok = False  # the open paragraph may be underlined into a heading
    if (
        front_matter
        and len(lines) > 1
        and lines[0].text.rstrip(" \t") == "---"
        and not _md_blank(lines[1])  # Pandoc: no blank line after the opening
    ):
        for m in range(1, len(lines)):
            if lines[m].text.rstrip(" \t") in _FRONT_MATTER_CLOSE:
                blocks.append(_Block("front_matter", lines[0].start, lines[m].end))
                k = m + 1
                break

    def close_prose(upto: int) -> None:
        nonlocal paragraph_start
        if paragraph_start is not None and upto > paragraph_start:
            blocks.append(
                _Block("prose", lines[paragraph_start].start, lines[upto - 1].end)
            )
        paragraph_start = None

    while k < len(lines):
        line = lines[k]
        fence = _FENCE.fullmatch(line.text)
        if fence and not (fence.group(2)[0] == "`" and "`" in fence.group(3)):
            close_prose(k)
            char, size = fence.group(2)[0], len(fence.group(2))
            closing = re.compile(rf" {{0,3}}{re.escape(char)}{{{size},}}[ \t]*")
            m = k + 1
            while m < len(lines) and not closing.fullmatch(lines[m].text):
                m += 1
            last = min(m, len(lines) - 1)
            blocks.append(_Block("code", line.start, lines[last].end))
            k = m + 1
            continue
        atx = _atx(line.text)
        if atx is not None:
            close_prose(k)
            blocks.append(_Block("heading", line.start, line.end, *atx))
            k += 1
            continue
        if paragraph_start is not None and setext_ok and _SETEXT.fullmatch(line.text):
            title = " ".join(lines[i].text.strip() for i in range(paragraph_start, k))
            level = 1 if line.text.strip()[0] == "=" else 2
            blocks.append(
                _Block("heading", lines[paragraph_start].start, line.end, level, title)
            )
            paragraph_start = None
            k += 1
            continue
        if _MD_THEMATIC.fullmatch(line.text):
            # A thematic break ends a paragraph; it is prose of its own.
            close_prose(k)
            blocks.append(_Block("prose", line.start, line.end))
            k += 1
            continue
        if _md_blank(line):
            close_prose(k)
            # Blank lines belong to the prose around them only as separators.
            k += 1
            continue
        if paragraph_start is None:
            paragraph_start = k
            setext_ok = _indent_columns(line.text) < _INDENTED_CODE_COLUMNS
        setext_ok = setext_ok and not _MD_CONTAINER.match(line.text)
        k += 1
    close_prose(len(lines))
    return blocks


def _rst_blocks(lines: list[_Line]) -> list[_Block]:
    """Return heading, code and prose blocks of reStructuredText, in order."""
    blocks: list[_Block] = []
    styles: list[tuple[str, bool]] = []  # heading styles in order of appearance
    # docutils expands tabs to 8 columns before it measures indentation.
    expanded = [line.text.expandtabs(8) for line in lines]
    blank = [not text.strip() for text in expanded]
    indents = [len(text) - len(text.lstrip()) for text in expanded]
    k = 0
    paragraph_start: int | None = None
    explicit_indent: int | None = None  # the open paragraph is explicit markup
    literal_next = False  # the paragraph just closed ended with "::"
    literal_indent = 0  # a literal body is indented deeper than this

    def unmarked(index: int) -> str:
        """Line *index* with its list markers blanked (columns kept)."""
        text = expanded[index]
        # items = _RST_ITEMS.match(text)
        items = _rst_items_match(text)
        return " " * items.end() + text[items.end() :] if items else text

    def body_indent(after: int, indent: int) -> int | None:
        """Least indentation deeper than *indent* of the lines after *after*."""
        least: int | None = None
        for m in range(after + 1, len(lines)):
            if blank[m]:
                continue
            if indents[m] <= indent:
                break
            least = indents[m] if least is None else min(least, indents[m])
        return least

    def close_prose(upto: int) -> None:
        nonlocal paragraph_start, explicit_indent, literal_next, literal_indent
        if paragraph_start is not None and upto > paragraph_start:
            blocks.append(
                _Block("prose", lines[paragraph_start].start, lines[upto - 1].end)
            )
            head, last = unmarked(paragraph_start), expanded[upto - 1]
            single = upto - 1 == paragraph_start
            literal_next = last.rstrip().endswith("::")
            explicit = _RST_EXPLICIT.match(head) is not None
            if (
                explicit
                and (
                    # A comment's content is not parsed; "name::" alone is
                    # directive syntax.
                    not _RST_CONSTRUCT.match(head)
                    or (single and _RST_BARE_DIRECTIVE.fullmatch(head))
                )
            ) or _RST_NO_LITERAL.match(head):
                literal_next = False
            if not single and _RST_DIRECTIVE.match(head) and _RST_FIELD.match(last):
                # A field line in a directive's first paragraph is an option;
                # its "::" is part of the value.
                literal_next = False
            # The literal body is indented deeper than the text of the line
            # that ends with "::". On the first line of explicit markup, a
            # field or an option, docutils takes the least indentation of
            # the lines that follow; on a list item's first line the text
            # starts after the markers; otherwise it is the line's own.
            # items = _RST_ITEMS.match(last) if single else None
            items = _rst_items_match(last) if single else None
            if not literal_next:
                literal_indent = 0
            elif single and (explicit or _RST_FIELD_OR_OPTION.match(head)):
                head_indent = len(head) - len(head.lstrip())
                least = body_indent(upto - 1, head_indent)
                literal_next = least is not None
                literal_indent = least or 0
            elif items:
                literal_indent = items.end()
            else:
                literal_indent = indents[upto - 1]
        paragraph_start = None
        explicit_indent = None

    def indented_body(after: int, indent: int) -> int:
        """Index after the indented block that follows line ``after``."""
        m = after + 1
        seen = False
        while m < len(lines):
            if not blank[m]:
                if indents[m] <= indent:
                    break
                seen = True
            m += 1
        if not seen:
            return after + 1
        while m - 1 > after and blank[m - 1]:
            m -= 1
        return m

    def quoted_body(after: int, indent: int) -> int:
        """Index after a quoted literal block that follows line ``after``.

        Its lines are at *indent* and all start with one punctuation
        character (docutils' nonalphanum7bit), up to a blank line.
        """
        m = after + 1
        while m < len(lines) and blank[m]:
            m += 1
        if m == len(lines) or indents[m] != indent:
            return after + 1
        quote = expanded[m][indent]
        if not _RST_ADORNMENT.fullmatch(quote):
            return after + 1
        while (
            m < len(lines)
            and not blank[m]
            and indents[m] == indent
            and expanded[m][indent] == quote
        ):
            m += 1
        return m

    def level_of(char: str, over: bool) -> int:
        style = (char, over)
        if style not in styles:
            styles.append(style)
        return styles.index(style) + 1

    while k < len(lines):
        line = lines[k]
        text = line.text
        if (
            explicit_indent is not None
            and not blank[k]
            and indents[k] <= explicit_indent
        ):
            # Explicit markup ends at a line that is not indented deeper.
            close_prose(k)
        # Overline + title + underline.
        if (
            k + 2 < len(lines)
            and _RST_ADORNMENT.fullmatch(text.rstrip())
            and lines[k + 1].text.strip()
            and lines[k + 2].text.rstrip() == text.rstrip()
            and len(text.rstrip()) >= len(lines[k + 1].text.strip())
        ):
            close_prose(k)
            literal_next = False
            blocks.append(
                _Block(
                    "heading",
                    line.start,
                    lines[k + 2].end,
                    level_of(text.rstrip()[0], True),
                    lines[k + 1].text.strip(),
                )
            )
            k += 3
            continue
        # Title + underline (the title is a one-line paragraph, not indented).
        if (
            k + 1 < len(lines)
            and text.strip()
            and not text[:1].isspace()
            and paragraph_start is None
            and _RST_ADORNMENT.fullmatch(lines[k + 1].text.rstrip())
            and len(lines[k + 1].text.rstrip()) >= len(text.rstrip())
            and not _RST_ADORNMENT.fullmatch(text.rstrip())
            and not _RST_NOT_TITLE.match(text)
        ):
            close_prose(k)
            literal_next = False
            blocks.append(
                _Block(
                    "heading",
                    line.start,
                    lines[k + 1].end,
                    level_of(lines[k + 1].text.rstrip()[0], False),
                    text.strip(),
                )
            )
            k += 2
            continue
        directive = _RST_CODE.fullmatch(text)
        if directive:
            close_prose(k)
            literal_next = False
            end = indented_body(k, indents[k])
            blocks.append(_Block("code", line.start, lines[end - 1].end))
            k = end
            continue
        if blank[k]:
            close_prose(k)
            if literal_next:
                # The body indented deeper than "para::" is a literal block,
                # or else lines quoted with one punctuation character.
                end = indented_body(k, literal_indent)
                if end == k + 1:
                    end = quoted_body(k, literal_indent)
                if end > k + 1:
                    blocks.append(
                        _Block("code", lines[k + 1].start, lines[end - 1].end)
                    )
                    literal_next = False
                    k = end
                    continue
            literal_next = False
            k += 1
            continue
        if paragraph_start is None:
            paragraph_start = k
            head = unmarked(k)
            if _RST_EXPLICIT.match(head):
                explicit_indent = len(head) - len(head.lstrip())
        k += 1
    close_prose(len(lines))
    return blocks


@dataclass(frozen=True)
class StructureChunkerConfig(ChunkerConfig):
    r"""Configuration for :class:`StructureChunker`.

    Parameters
    ----------
    markup : {"markdown", "rst"}, default="markdown"
        The document's markup; given, never guessed.
    chunk_size : int, default=1000
        Largest chunk anchor, in characters, not counting the whitespace
        around it. Longer prose is split by :class:`RecursiveChunker`'s rule;
        a longer code block at blank lines, then lines.
    include_heading : bool, default=True
        When a heading is directly followed by prose, begin the context of
        that prose's first chunk at the heading, so the heading's words are
        part of the text that is retrieved. A heading that is not (it is
        followed by code, another heading or the end, or this is ``False``)
        is a chunk of its own, ``block="heading"``.
    separators : tuple of str, default=("\n\n", "\n", ". ", " ", "")
        Separators for prose, coarsest first, ending with ``""``.
    min_length : int, default=1
        Drop a chunk with fewer characters than this, not counting the
        whitespace around it.
    front_matter : bool, default=True
        Markdown only: a first line ``---``, followed by a line that is not
        blank and closed by a later line ``---`` or ``...``, is YAML front
        matter (Pandoc's rule), chunked like code with
        ``block="front_matter"``. Otherwise, or with ``False``, the ``---`` is
        read as CommonMark reads it (a thematic break).
    """

    markup: str = "markdown"
    chunk_size: int = 1000
    include_heading: bool = True
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    min_length: int = 1
    front_matter: bool = True

    def __post_init__(self) -> None:
        """Validate the fields."""
        super().__post_init__()
        _coerce_separators(self)
        if self.markup not in MARKUPS:
            raise ValueError(
                f"StructureChunkerConfig.markup must be one of {MARKUPS};"
                f" got {self.markup!r}."
            )
        _positive(self.chunk_size, "StructureChunkerConfig.chunk_size")
        if not isinstance(self.include_heading, bool):
            raise TypeError("StructureChunkerConfig.include_heading must be a bool.")
        _validate_separators(self.separators, "StructureChunkerConfig")
        _positive(self.min_length, "StructureChunkerConfig.min_length")
        if not isinstance(self.front_matter, bool):
            raise TypeError("StructureChunkerConfig.front_matter must be a bool.")


class StructureChunker(_BatchMixin):
    r"""Chunk Markdown or reStructuredText by its headings and code blocks.

    Parameters
    ----------
    config : StructureChunkerConfig or None, default=None
        ``None`` is ``StructureChunkerConfig()`` (Markdown).

    See Also
    --------
    RecursiveChunker : Splits prose that exceeds ``chunk_size``.

    Notes
    -----
    **User.** Every chunk's metadata holds ``heading_path`` (a tuple of the
    titles of the sections that contain it, outermost first), ``block``
    (``"prose"``, ``"code"``, ``"heading"`` or ``"front_matter"``) and
    ``section_level``; documents built from the chunks carry all three. With
    ``include_heading``, a heading directly followed by prose is in the text
    of that prose's first chunk; any other heading is a chunk of its own, so
    no heading's words are lost (unless ``min_length`` drops them). Code
    blocks are never merged with prose, and a code chunk starts at the
    beginning of its first line, so its indentation is kept.

    Examples
    --------
    >>> text = "# Guide\n\nIntro text.\n\n## Install\n\n```\npip install x\n```\n"
    >>> chunks = StructureChunker().chunk(text).chunks
    >>> [
    ...     (c.metadata["block"], c.metadata["heading_path"], c.text) for c in chunks
    ... ]  # doctest: +NORMALIZE_WHITESPACE
    [('prose', ('Guide',), '# Guide\n\nIntro text.'),
     ('heading', ('Guide', 'Install'), '## Install'),
     ('code', ('Guide', 'Install'), '```\npip install x\n```')]
    """

    config_class = StructureChunkerConfig
    text_is_source_slice = True
    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.BLOCK

    def __init__(self, config: StructureChunkerConfig | None = None) -> None:
        if config is not None and not isinstance(config, StructureChunkerConfig):
            raise TypeError(
                "StructureChunker: config must be a StructureChunkerConfig or"
                f" None, got {type(config).__name__}."
            )
        self._cfg = config if config is not None else StructureChunkerConfig()

    @property
    def config(self) -> StructureChunkerConfig:
        """The configuration (read-only)."""
        return self._cfg

    def blocks(self, text: str) -> list[tuple[str, Span, int, str]]:
        """Return the document's blocks: ``(kind, span, level, title)``.

        Parameters
        ----------
        text : str
            The source.

        Returns
        -------
        list of (str, (int, int), int, str)
            ``kind`` is ``"heading"`` (with its level and title), ``"code"``,
            ``"prose"`` or ``"front_matter"``; spans in source order, without
            blank lines between blocks.
        """
        if self._cfg.markup == "markdown":
            parsed = _markdown_blocks(
                _lines(text, _MD_BREAK), front_matter=self._cfg.front_matter
            )
        else:
            parsed = _rst_blocks(_lines(text, _RST_BREAK))
        return [(b.kind, (b.start, b.end), b.level, b.title) for b in parsed]

    def _regions(self, text: str) -> list[tuple[str, Span, int, str]]:
        """Join consecutive prose blocks of one section into one region.

        Paragraphs are separate prose blocks; chunking each alone would make
        one chunk per paragraph whatever ``chunk_size`` is. A region runs
        from its first paragraph to its last (the blank lines between
        included) and is split by the budget, blank lines first.
        """
        regions: list[tuple[str, Span, int, str]] = []
        for kind, span, level, title in self.blocks(text):
            if kind == "prose" and regions and regions[-1][0] == "prose":
                previous = regions[-1]
                regions[-1] = ("prose", (previous[1][0], span[1]), 0, "")
            else:
                regions.append((kind, span, level, title))
        return regions

    def chunk(
        self,
        text: str,
        doc_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Split *text* into prose and code chunks with their heading path.

        Parameters
        ----------
        text : str
            Raw document text.
        doc_id : str or None, default=None
            Stored in each chunk's metadata.
        extra_metadata : dict or None, default=None
            Merged into the result metadata.

        Returns
        -------
        ChunkResult
            Chunks in source order.

        Raises
        ------
        TypeError
            If *text* is not a ``str``.
        ValueError
            If *text* is empty or whitespace-only, or holds a NUL character
            or a lone surrogate.
        """
        _text_or_raise(text, "StructureChunker")
        cfg = self._cfg
        path: list[tuple[int, str]] = []
        pending: Span | None = None  # a heading not yet in any chunk's text
        chunks: list[Chunk] = []
        counts = {"code": 0, "heading": 0}

        def emit(kind: str, span: Span, separators: tuple[str, ...]) -> None:
            nonlocal pending
            verbatim = kind in ("code", "front_matter")
            for piece in recursive_spans(text, span, cfg.chunk_size, separators):
                p_start, p_end = strip_span(text, *piece)
                if verbatim and p_end > p_start:
                    # Code keeps the indentation of its first line: start at
                    # the beginning of that line, not at its first character.
                    p_start = max(piece[0], text.rfind("\n", 0, p_start) + 1)
                bounds = _bounds(text, p_start, p_end)
                if bounds is None or bounds[1] - bounds[0] < cfg.min_length:
                    continue
                context = p_start
                if kind == "prose" and pending is not None:
                    context = pending[0]
                    pending = None
                meta: dict[str, Any] = {
                    "chunk_index": len(chunks),
                    "heading_path": tuple(t for _, t in path),
                    "section_level": path[-1][0] if path else 0,
                    "block": kind,
                }
                if doc_id is not None:
                    meta["doc_id"] = doc_id
                chunks.append(
                    Chunk(
                        text=text[context:p_end],
                        start_char=p_start,
                        end_char=p_end,
                        metadata=meta,
                        context_start=context if context != p_start else None,
                    )
                )
                if kind in counts:
                    counts[kind] += 1

        def flush() -> None:
            nonlocal pending
            if pending is not None:
                span, pending = pending, None
                emit("heading", span, cfg.separators)

        for kind, span, level, title in self._regions(text):
            if kind == "heading":
                flush()
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, title))
                pending = span
                if not cfg.include_heading:
                    flush()
            elif kind == "prose":
                emit("prose", span, cfg.separators)
            else:
                flush()
                emit(kind, span, _CODE_SEPARATORS)
        flush()
        result_meta: dict[str, Any] = {
            "span_basis": "input",
            "chunker": "structure",
            "markup": cfg.markup,
            "chunk_size": cfg.chunk_size,
            "code_chunks": counts["code"],
            "heading_chunks": counts["heading"],
            "total_chunks": len(chunks),
        }
        if extra_metadata:
            result_meta.update(extra_metadata)
        return ChunkResult(chunks=chunks, metadata=result_meta)
