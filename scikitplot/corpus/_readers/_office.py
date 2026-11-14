# scikitplot/corpus/_readers/_office.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Office documents: Office Open XML (``.docx``, ``.pptx``, ``.xlsx``) and OpenDocument (``.odt``, ``.ods``, ``.odp``).

Both families are ZIP packages of XML parts (OpenDocument also has a flat,
single-XML form). This reader opens the package in memory, finds the parts
that hold text (Office Open XML: by relationships; OpenDocument:
``content.xml``) and parses each with the hardened XML parser. Nothing is
extracted to disk and nothing in the package (macros, OLE objects, external
links) is run. The content decides how a file is read, not its extension.

Python 3.8-3.15. Standard library only.
"""

from __future__ import annotations

import logging
import posixpath
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, Iterator

from .._base import DocumentReader
from .._schema import SectionType
from ._xml_safety import parse_stdlib_secure

logger = logging.getLogger(__name__)

__all__ = ["OfficeReader"]

_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

#: Strict-conformance namespaces, mapped to the transitional ones the reader
#: is written against. The element and attribute names are the same.
_STRICT_PREFIX = "http://purl.oclc.org/ooxml/"
_STRICT = {
    "http://purl.oclc.org/ooxml/wordprocessingml/main": _W,
    "http://purl.oclc.org/ooxml/drawingml/main": _A,
    "http://purl.oclc.org/ooxml/presentationml/main": _P,
    "http://purl.oclc.org/ooxml/spreadsheetml/main": _S,
    "http://purl.oclc.org/ooxml/officeDocument/relationships": _DOC_REL,
}

_ODF = "urn:oasis:names:tc:opendocument:xmlns:"
_O_OFFICE = _ODF + "office:1.0"
_O_TEXT = _ODF + "text:1.0"
_O_TABLE = _ODF + "table:1.0"
_O_DRAW = _ODF + "drawing:1.0"
_O_PRESENTATION = _ODF + "presentation:1.0"
_O_MANIFEST = _ODF + "manifest:1.0"
_O_STYLE = _ODF + "style:1.0"

#: Sheet size limits of the common spreadsheet applications; a repeat count
#: beyond them is refused instead of expanded.
_MAX_COLUMNS = 16_384
_MAX_ROWS = 1_048_576
#: Most spaces one ``text:s`` may stand for.
_MAX_SPACES = 1_000

#: Relationship types, by their last path segment.
_OFFICE_DOCUMENT = "officeDocument"
_SLIDE = "slide"
_NOTES_SLIDE = "notesSlide"
_WORKSHEET = "worksheet"
_SHARED_STRINGS = "sharedStrings"
_FOOTNOTES = "footnotes"
_ENDNOTES = "endnotes"
_STYLES = "styles"

#: Deepest heading level kept; a larger outline level is body text.
_MAX_HEADING_LEVEL = 10

#: Start of an OLE2 compound file: a legacy binary or an encrypted package.
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_WORD_EXTENSIONS = [".docx", ".docm", ".dotx", ".dotm"]
_POWERPOINT_EXTENSIONS = [".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm"]
_EXCEL_EXTENSIONS = [".xlsx", ".xlsm", ".xltx", ".xltm"]
_OPENDOCUMENT_EXTENSIONS = [
    ".odt",
    ".ott",
    ".ods",
    ".ots",
    ".odp",
    ".otp",
    ".odg",
    ".otg",
    ".odf",
    ".fodt",
    ".fods",
    ".fodp",
    ".fodg",
]


def _q(namespace: str, name: str) -> str:
    """Return the ElementTree (Clark) name ``{namespace}name``."""
    return f"{{{namespace}}}{name}"


def _transitional(root: Any) -> None:
    """Rename Strict-conformance names under *root* to the transitional ones."""

    def renamed(name: str) -> str:
        if not name.startswith("{" + _STRICT_PREFIX):
            return name
        namespace, _, local = name[1:].partition("}")
        return _q(_STRICT.get(namespace, namespace), local)

    for element in root.iter():
        if isinstance(element.tag, str):
            element.tag = renamed(element.tag)
        strict = [key for key in element.attrib if key.startswith("{" + _STRICT_PREFIX)]
        for key in strict:
            element.attrib[renamed(key)] = element.attrib.pop(key)


def _parse(data: bytes, what: str) -> Any:
    """
    Parse *data* with the hardened parser.

    Raises
    ------
    ValueError
        If it declares entities or is not well-formed; *what* names it.
    """
    try:
        return parse_stdlib_secure(data)
    except ValueError:
        raise
    except Exception as exc:  # expat errors are not ValueError
        raise ValueError(f"OfficeReader: {what} is not well-formed XML: {exc}") from exc


class _Package:
    """
    The parts of an Office Open XML package, read with limits.

    Parameters
    ----------
    zf : zipfile.ZipFile
        The open package.
    name : str
        The file's display name, for messages.
    max_part_bytes : int
        Largest XML part read, uncompressed.
    max_parts : int
        Most parts the package may hold.

    Raises
    ------
    ValueError
        If the package holds more than ``max_parts`` parts.
    """

    def __init__(
        self, zf: zipfile.ZipFile, name: str, max_part_bytes: int, max_parts: int
    ) -> None:
        self._zf = zf
        self._name = name
        self._max = max_part_bytes
        infos = zf.infolist()
        if len(infos) > max_parts:
            raise ValueError(
                f"OfficeReader: {name} holds {len(infos)} parts"
                f" (max_parts={max_parts})."
            )
        self._infos = {info.filename: info for info in infos}
        if len(self._infos) != len(infos):
            seen: set[str] = set()
            twice = next(
                i.filename for i in infos if i.filename in seen or seen.add(i.filename)
            )
            raise ValueError(
                f"OfficeReader: {name} holds the part {twice!r} more than once;"
                " which one a reader takes is not defined."
            )

    def names(self) -> Iterator[str]:
        """Yield every part name."""
        return iter(self._infos)

    def xml(self, part: str) -> Any:
        """
        Return the parsed root of *part*, or ``None`` if the package lacks it.

        Raises
        ------
        ValueError
            If the part is larger than the limit, or is not well-formed XML.
        """
        info = self._infos.get(part)
        if info is None:
            return None
        if info.file_size > self._max:
            raise ValueError(
                f"OfficeReader: part {part!r} of {self._name} is"
                f" {info.file_size:,} bytes (max_part_bytes={self._max:,})."
            )
        try:
            with self._zf.open(info) as src:
                data = src.read(self._max + 1)
        except (
            zipfile.BadZipFile,
            zlib.error,
            EOFError,
            NotImplementedError,
            RuntimeError,
            OSError,
        ) as exc:
            # A damaged member, a compression method Python lacks, or a
            # member encrypted at the ZIP level (RuntimeError).
            raise ValueError(
                f"OfficeReader: part {part!r} of {self._name} cannot be read"
                f" from the package ({type(exc).__name__}: {exc})."
            ) from exc
        if len(data) > self._max:
            # The declared size understated the content.
            raise ValueError(
                f"OfficeReader: part {part!r} of {self._name} is larger than"
                f" max_part_bytes={self._max:,}."
            )
        root = _parse(data, f"part {part!r} of {self._name}")
        if _STRICT_PREFIX.encode("ascii") in data:
            _transitional(root)
        return root

    def has(self, part: str) -> bool:
        """Whether the package holds *part*."""
        return part in self._infos

    def relationships(self, part: str) -> dict[str, tuple[str, str]]:
        """
        Return ``{id: (type, target part)}`` for *part*'s internal relationships.

        ``part`` is ``""`` for the package's own relationships. External
        targets (``TargetMode="External"``) and targets outside the package
        (``../`` above its root) are left out.
        """
        folder, base = posixpath.split(part)
        root = self.xml(posixpath.join(folder, "_rels", f"{base}.rels"))
        found: dict[str, tuple[str, str]] = {}
        if root is None:
            return found
        for rel in root.iter(_q(_PKG_REL, "Relationship")):
            if rel.get("TargetMode") == "External":
                continue
            target = rel.get("Target") or ""
            if target.startswith("/"):
                resolved = posixpath.normpath(target.lstrip("/"))
            else:
                resolved = posixpath.normpath(posixpath.join(folder, target))
            if resolved.startswith("../") or resolved in ("..", "."):
                logger.warning(
                    "OfficeReader: %s: relationship %r of %r points outside"
                    " the package (%r); ignored.",
                    self._name,
                    rel.get("Id"),
                    part or "the package",
                    target,
                )
                continue
            kind = (rel.get("Type") or "").rsplit("/", 1)[-1]
            found[rel.get("Id") or ""] = (kind, resolved)
        return found

    def first(self, part: str, kind: str) -> str | None:
        """Return the target of *part*'s first relationship of *kind*."""
        for rel_kind, target in self.relationships(part).values():
            if rel_kind == kind:
                return target
        return None


# ---------------------------------------------------------------------------
# Text of WordprocessingML and DrawingML
# ---------------------------------------------------------------------------


#: The fallback of a markup-compatibility block: the same content again, in
#: an older form, next to the ``mc:Choice`` a reader of this format takes.
_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def _walk(
    root: Any,
    *,
    skip: frozenset = frozenset(),
    leaves: frozenset = frozenset(),
    drop: Any = None,
) -> Iterator[Any]:
    """
    Yield the descendants of *root* in document order, each once.

    Parameters
    ----------
    root : Element
        Where to start; not yielded itself.
    skip : frozenset of str, optional
        Tags whose subtree is left out. ``mc:Fallback`` is always left out.
    leaves : frozenset of str, optional
        Tags that are yielded but not entered, so content nested in them
        (a table in a cell, a text box in a paragraph) is read by whoever
        reads the leaf, and only there.
    drop : callable or None, optional
        ``drop(element) -> bool``: an element for which it is true is left
        out with its subtree, like a ``skip`` tag.
    """
    stack = [iter(root)]
    while stack:
        node = next(stack[-1], None)
        if node is None:
            stack.pop()
            continue
        tag = node.tag
        if tag == _FALLBACK or tag in skip or (drop is not None and drop(node)):
            continue
        yield node
        if tag not in leaves:
            stack.append(iter(node))


#: Tracked changes whose text is not in the document as it now reads.
_WORD_REMOVED = frozenset({_q(_W, "del"), _q(_W, "moveFrom")})


class _Text(str):
    """Text that records whether it holds hidden text (``hidden``)."""

    __slots__ = ("hidden",)

    def __new__(cls, text: str, hidden: bool = False) -> Any:
        made = super().__new__(cls, text)
        made.hidden = hidden
        return made


def _vanish_of(properties: Any) -> bool | None:
    """Return what a ``w:rPr`` says about hiding: ``True``, ``False`` or ``None``."""
    vanish = None if properties is None else properties.find(_q(_W, "vanish"))
    if vanish is None:
        return None
    return (vanish.get(_q(_W, "val")) or "true").lower() not in ("0", "false", "off")


class _WordHiding:
    """
    Which styles hide text, read from a Word styles part.

    Parameters
    ----------
    styles : Element or None
        The root of ``word/styles.xml``; ``None`` for a document without one.

    Notes
    -----
    A run is hidden by, in order: its own ``w:rPr/w:vanish``; its
    character style (``w:rStyle``); its paragraph's style (``w:pStyle``);
    the document default (``w:docDefaults``). A style without ``w:vanish``
    takes its ``w:basedOn`` style's value; a cycle of styles ends.
    """

    def __init__(self, styles: Any = None) -> None:
        own: dict[str, bool | None] = {}
        parents: dict[str, str] = {}
        self.default = False
        if styles is None:
            self.styles: dict[str, bool] = {}
            return
        defaults = styles.find(f"{_q(_W, 'docDefaults')}/{_q(_W, 'rPrDefault')}")
        if defaults is not None:
            self.default = bool(_vanish_of(defaults.find(_q(_W, "rPr"))))
        for style in styles.iter(_q(_W, "style")):
            style_id = style.get(_q(_W, "styleId")) or ""
            own[style_id] = _vanish_of(style.find(_q(_W, "rPr")))
            based = style.find(_q(_W, "basedOn"))
            if based is not None:
                parents[style_id] = based.get(_q(_W, "val")) or ""
        self.styles = {}
        for style_id in own:
            seen: set[str] = set()
            current = style_id
            while current in own and current not in seen:
                if own[current] is not None:
                    self.styles[style_id] = bool(own[current])
                    break
                seen.add(current)
                current = parents.get(current, "")

    def paragraph(self, p: Any) -> bool | None:
        """Return what the paragraph's style says about hiding its runs."""
        style = p.find(f"{_q(_W, 'pPr')}/{_q(_W, 'pStyle')}")
        return (
            None if style is None else self.styles.get(style.get(_q(_W, "val")) or "")
        )

    def run(self, run: Any, paragraph: bool | None) -> bool:
        """Whether a ``w:r`` is hidden, given its paragraph's style setting."""
        properties = run.find(_q(_W, "rPr"))
        own = _vanish_of(properties)
        if own is not None:
            return own
        style = None if properties is None else properties.find(_q(_W, "rStyle"))
        if style is not None:
            by_style = self.styles.get(style.get(_q(_W, "val")) or "")
            if by_style is not None:
                return by_style
        return self.default if paragraph is None else paragraph


#: No styles: only a run's own ``w:vanish`` hides it.
_NO_STYLES = _WordHiding()


def _word_paragraph(
    p: Any, *, keep_hidden: bool = True, hiding: _WordHiding = _NO_STYLES
) -> Any:
    """
    Return the text of a ``w:p``: runs, tabs and breaks, as :class:`_Text`.

    Tracked deletions and the old place of moved text are dropped; a
    markup-compatibility fallback is not read a second time. Runs
    formatted as hidden (``w:vanish``, directly or by a style in
    *hiding*) are read when *keep_hidden* and left out otherwise; the
    result's ``hidden`` says whether any was read.
    """
    by_paragraph = hiding.paragraph(p)
    out: list[str] = []
    t, tab, br, cr, run = (_q(_W, n) for n in ("t", "tab", "br", "cr", "r"))
    hyphen = _q(_W, "noBreakHyphen")
    hidden = False

    def drop(node: Any) -> bool:
        nonlocal hidden
        if node.tag != run or not hiding.run(node, by_paragraph):
            return False
        if keep_hidden and any(True for _ in node.iter(t)):
            hidden = True
        return not keep_hidden

    for node in _walk(p, skip=_WORD_REMOVED, drop=drop):
        tag = node.tag
        if tag == t:
            out.append(node.text or "")
        elif tag == tab:
            out.append("\t")
        elif tag in (br, cr):
            out.append("\n")
        elif tag == hyphen:
            out.append("-")
    return _Text("".join(out), hidden)


def _word_paragraphs(
    container: Any, *, keep_hidden: bool = True, hiding: _WordHiding = _NO_STYLES
) -> list[Any]:
    """Return each ``w:p`` under *container* as :class:`_Text`, nested ones in their parent."""
    p = _q(_W, "p")
    found = _walk(container, skip=_WORD_REMOVED, leaves=frozenset({p}))
    return [
        _word_paragraph(node, keep_hidden=keep_hidden, hiding=hiding)
        for node in found
        if node.tag == p
    ]


def _joined(parts: list[Any], separator: str = "\n") -> Any:
    """Join :class:`_Text` parts, keeping whether any held hidden text."""
    return _Text(
        separator.join(parts), any(getattr(part, "hidden", False) for part in parts)
    )


def _outline_level(properties: Any) -> int | None:
    """Return the 1-based heading level a ``w:pPr`` sets directly, if any."""
    outline = None if properties is None else properties.find(_q(_W, "outlineLvl"))
    if outline is None:
        return None
    try:
        level = int(outline.get(_q(_W, "val")) or "") + 1
    except ValueError:
        return None
    # Level 10 (val 9) is "body text" in WordprocessingML.
    return level if 1 <= level < _MAX_HEADING_LEVEL else None


def _word_style_levels(styles: Any) -> dict[str, int]:
    """
    Return ``{paragraph style id: heading level}`` from a styles part.

    A style is a heading when it sets an outline level, is one of the
    built-in ``heading 1`` to ``heading 9`` styles (stored under that name
    in every language), or is based on a style that is.
    """
    levels: dict[str, int] = {}
    parents: dict[str, str] = {}
    for style in [] if styles is None else styles.iter(_q(_W, "style")):
        if style.get(_q(_W, "type")) not in (None, "paragraph"):
            continue
        style_id = style.get(_q(_W, "styleId")) or ""
        level = _outline_level(style.find(_q(_W, "pPr")))
        name = style.find(_q(_W, "name"))
        words = [] if name is None else (name.get(_q(_W, "val")) or "").lower().split()
        named = (
            level is None
            and words[:1] == ["heading"]
            and len(words) == 2  # noqa: PLR2004
            and words[1].isdigit()
        )
        if named and 1 <= int(words[1]) < _MAX_HEADING_LEVEL:
            level = int(words[1])
        if level is not None:
            levels[style_id] = level
        based = style.find(_q(_W, "basedOn"))
        if based is not None:
            parents[style_id] = based.get(_q(_W, "val")) or ""
    for style_id, first_parent in parents.items():
        seen = {style_id}
        parent = first_parent
        while style_id not in levels and parent and parent not in seen:
            if parent in levels:
                levels[style_id] = levels[parent]
            seen.add(parent)
            parent = parents.get(parent, "")
    return levels


def _word_heading(p: Any, levels: dict[str, int]) -> int | None:
    """Return the heading level of a ``w:p``, or ``None`` for body text."""
    properties = p.find(_q(_W, "pPr"))
    if properties is None:
        return None
    if properties.find(_q(_W, "outlineLvl")) is not None:
        return _outline_level(properties)
    style = properties.find(_q(_W, "pStyle"))
    return None if style is None else levels.get(style.get(_q(_W, "val")) or "")


def _word_table(
    tbl: Any, *, keep_hidden: bool = True, hiding: _WordHiding = _NO_STYLES
) -> Any:
    """
    Return a ``w:tbl`` as lines of tab-separated cells; empty rows dropped.

    A table nested in a cell is part of that cell's text and is not read
    again as rows of its own.
    """
    tr, tc = _q(_W, "tr"), _q(_W, "tc")
    rows = []
    for row in _walk(tbl, leaves=frozenset({tr})):
        if row.tag != tr:
            continue
        cells = []
        for cell in _walk(row, leaves=frozenset({tc})):
            if cell.tag == tc:
                text = _joined(
                    _word_paragraphs(cell, keep_hidden=keep_hidden, hiding=hiding)
                )
                cells.append(_Text(text.strip(), text.hidden))
        if any(cells):
            rows.append(_joined(cells, "\t"))
    return _joined(rows)


def _drawing_paragraph(p: Any) -> str:
    """Return the text of one DrawingML ``a:p``: its runs and line breaks."""
    t, br = _q(_A, "t"), _q(_A, "br")
    parts = []
    for node in _walk(p):
        if node.tag == t:
            parts.append(node.text or "")
        elif node.tag == br:
            parts.append("\n")
    return "".join(parts)


def _drawing_paragraphs(container: Any) -> list[str]:
    """Return the ``a:p`` paragraphs under *container* in document order."""
    p = _q(_A, "p")
    found = _walk(container, leaves=frozenset({p}))
    return [_drawing_paragraph(node) for node in found if node.tag == p]


def _slide_text(root: Any) -> str:
    """Return a slide's text: shapes in document order, tables as rows."""
    lines: list[str] = []
    tbl, tr, tc, p = (_q(_A, n) for n in ("tbl", "tr", "tc", "p"))
    for node in _walk(root, leaves=frozenset({tbl, p})):
        if node.tag == tbl:
            for row in _walk(node, leaves=frozenset({tr})):
                if row.tag != tr:
                    continue
                cells = [
                    "\n".join(_drawing_paragraphs(cell)).strip()
                    for cell in _walk(row, leaves=frozenset({tc}))
                    if cell.tag == tc
                ]
                if any(cells):
                    lines.append("\t".join(cells))
        elif node.tag == p:
            lines.append(_drawing_paragraph(node))
    return "\n".join(line for line in lines if line.strip()).strip()


# ---------------------------------------------------------------------------
# Text of SpreadsheetML
# ---------------------------------------------------------------------------


def _string_item(si: Any) -> str:
    """Return a shared or inline string: its ``t`` runs, phonetic runs left out."""
    t, phonetic = _q(_S, "t"), _q(_S, "rPh")
    out: list[str] = []
    stack = [iter(si)]
    while stack:
        node = next(stack[-1], None)
        if node is None:
            stack.pop()
            continue
        if node.tag == phonetic:
            continue
        if node.tag == t:
            out.append(node.text or "")
        stack.append(iter(node))
    return "".join(out)


def _column_index(ref: str) -> int | None:
    """Return the 0-based column of a cell reference (``"C7"`` -> 2)."""
    index = 0
    letters = 0
    for char in ref:
        if "A" <= char <= "Z":
            index = index * 26 + (ord(char) - 64)
            letters += 1
        else:
            break
    return index - 1 if letters else None


def _cell_text(cell: Any, shared: list[str]) -> str:
    """
    Return a cell's value as text: as stored, never computed.

    Shared strings are looked up, booleans read ``TRUE``/``FALSE``, a
    formula gives its cached result (none: empty), numbers and dates give
    the stored number.
    """
    kind = cell.get("t", "n")
    if kind == "inlineStr":
        inline = cell.find(_q(_S, "is"))
        return "" if inline is None else _string_item(inline)
    value = cell.find(_q(_S, "v"))
    text = "" if value is None or value.text is None else value.text
    if kind == "s":
        try:
            return shared[int(text)]
        except (ValueError, IndexError):
            return ""
    if kind == "b":
        return {"1": "TRUE", "0": "FALSE"}.get(text, text)
    return text


def _sheet_rows(root: Any, shared: list[str]) -> Iterator[tuple[int, str]]:
    """
    Yield ``(row number, line)`` for each row with a value.

    Cells are tab-separated at their column, so a gap keeps its place;
    trailing empty cells are dropped.
    """
    c_tag = _q(_S, "c")
    for position, row in enumerate(root.iter(_q(_S, "row")), start=1):
        try:
            number = int(row.get("r") or position)
        except ValueError:
            number = position
        cells: dict[int, str] = {}
        following = 0
        for cell in row.findall(c_tag):
            column = _column_index(cell.get("r") or "")
            if column is None:
                column = following
            following = column + 1
            cells[column] = _cell_text(cell, shared)
        values = [cells.get(i, "") for i in range(max(cells, default=-1) + 1)]
        while values and not values[-1].strip():
            values.pop()
        if values:
            yield number, "\t".join(values)


# ---------------------------------------------------------------------------
# Text of OpenDocument
# ---------------------------------------------------------------------------

#: Inline content that is not part of the running text.
_ODF_NOT_TEXT = frozenset(
    {
        _q(_O_TEXT, "note"),
        _q(_O_OFFICE, "annotation"),
        _q(_O_OFFICE, "annotation-end"),
        _q(_O_TEXT, "tracked-changes"),
    }
)


class _Heading(_Text):
    """A heading's text, with its outline ``level`` (1 is the top)."""

    __slots__ = ("level",)

    def __new__(cls, text: str, level: int, hidden: bool = False) -> Any:
        made = super().__new__(cls, text, hidden)
        made.level = min(max(level, 1), _MAX_HEADING_LEVEL)
        return made


def _repeat(element: Any, attribute: str) -> int:
    """Return a ``number-*-repeated`` count of *element*; 1 when absent or bad."""
    try:
        count = int(element.get(_q(_O_TABLE, attribute)) or 1)
    except ValueError:
        return 1
    return max(count, 1)


def _odf_hidden(element: Any) -> bool:
    """Whether a conditional ``text:hidden-*`` element currently hides."""
    return (element.get(_q(_O_TEXT, "is-hidden")) or "true").lower() != "false"


def _odf_hidden_section(element: Any) -> bool:
    """Whether a ``text:section`` is not displayed."""
    if element.tag != _q(_O_TEXT, "section"):
        return False
    display = (element.get(_q(_O_TEXT, "display")) or "true").lower()
    if display == "none":
        return True
    return display == "condition" and _odf_hidden(element)


class _OdfHiding:
    """
    Which OpenDocument styles hide text (``text:display="none"``).

    Parameters
    ----------
    *roots : Element or None
        Parts that hold styles: ``content.xml`` (automatic styles) and
        ``styles.xml``, or the one root of a flat file.

    Notes
    -----
    A ``style:style`` hides when its ``style:text-properties`` has
    ``text:display="none"``; one without the setting takes its
    ``style:parent-style-name`` style's value (cycles end). A style is
    looked up by family (``text`` for spans, ``paragraph`` for paragraphs
    and headings) and name.
    """

    def __init__(self, *roots: Any) -> None:
        own: dict[tuple[str, str], bool | None] = {}
        parents: dict[tuple[str, str], str] = {}
        family_attr, name_attr = _q(_O_STYLE, "family"), _q(_O_STYLE, "name")
        for root in roots:
            if root is None:
                continue
            for style in root.iter(_q(_O_STYLE, "style")):
                key = (style.get(family_attr) or "", style.get(name_attr) or "")
                props = style.find(_q(_O_STYLE, "text-properties"))
                display = None if props is None else props.get(_q(_O_TEXT, "display"))
                own[key] = None if display is None else display.lower() == "none"
                parent = style.get(_q(_O_STYLE, "parent-style-name"))
                if parent:
                    parents[key] = parent
        self.styles: dict[tuple[str, str], bool] = {}
        for key in own:
            seen: set[tuple[str, str]] = set()
            current = key
            while current in own and current not in seen:
                if own[current] is not None:
                    self.styles[key] = bool(own[current])
                    break
                seen.add(current)
                current = (key[0], parents.get(current, ""))

    def hides(self, element: Any, family: str) -> bool:
        """Whether *element*'s ``text:style-name`` style of *family* hides it."""
        name = element.get(_q(_O_TEXT, "style-name"))
        return bool(name) and self.styles.get((family, name), False)


#: No styles: only hiding marked on the elements themselves counts.
_NO_ODF_STYLES = _OdfHiding()


def _odf_text(  # noqa: PLR0912
    node: Any, *, keep_hidden: bool = True, hiding: _OdfHiding = _NO_ODF_STYLES
) -> Any:
    """
    Return the running text under *node* as :class:`_Text`.

    Spaces, tabs and breaks are expanded. A conditional hidden text
    (``text:hidden-text``, its ``text:string-value``) is read when
    *keep_hidden* and left out otherwise; ``hidden`` says whether any was.
    So is a ``text:span`` whose style in *hiding* is not displayed.
    """
    span = _q(_O_TEXT, "span")
    space, tab, line_break = (_q(_O_TEXT, n) for n in ("s", "tab", "line-break"))
    hidden_text = _q(_O_TEXT, "hidden-text")
    hidden = False
    out: list[str] = []
    stack: list[tuple[bool, Any]] = [(False, node)]
    while stack:
        is_text, item = stack.pop()
        if is_text:
            out.append(item)
            continue
        tag = item.tag
        if tag in _ODF_NOT_TEXT:
            continue
        if tag == span and hiding.hides(item, "text"):
            if not keep_hidden:
                continue
            if "".join(item.itertext()).strip():
                hidden = True
        if tag == hidden_text:
            value = item.get(_q(_O_TEXT, "string-value")) or ""
            if not _odf_hidden(item):
                out.append(value)
            elif keep_hidden and value:
                out.append(value)
                hidden = True
            continue
        if tag == space:
            try:
                count = int(item.get(_q(_O_TEXT, "c")) or 1)
            except ValueError:
                count = 1
            out.append(" " * max(1, min(count, _MAX_SPACES)))
        elif tag == tab:
            out.append("\t")
        elif tag == line_break:
            out.append("\n")
        else:
            items: list[tuple[bool, Any]] = []
            if item.text:
                items.append((True, item.text))
            for child in item:
                items.append((False, child))
                if child.tail:
                    items.append((True, child.tail))
            stack.extend(reversed(items))
    return _Text("".join(out), hidden)


def _odf_blocks(  # noqa: PLR0912
    container: Any,
    *,
    tables: bool,
    skip: frozenset = frozenset(),
    keep_hidden: bool = True,
    hiding: _OdfHiding = _NO_ODF_STYLES,
) -> Iterator[Any]:
    """
    Yield the paragraphs under *container* in document order.

    A paragraph or heading is yielded as its text (``str``) and not entered
    again, so a text box inside it is read once. With ``tables=True`` a
    ``table:table`` is yielded as its element instead of being entered.
    Subtrees in *skip*, tracked changes and annotations are left out. A
    section that is not displayed, and a paragraph marked hidden
    (``text:hidden-paragraph``), are read with ``hidden`` set when
    *keep_hidden* and left out otherwise.
    """
    heading = _q(_O_TEXT, "h")
    hidden_paragraph = _q(_O_TEXT, "hidden-paragraph")
    hidden_from: list[int] = []  # stack depths of hidden sections
    blocks = {_q(_O_TEXT, "p"), heading}
    table = _q(_O_TABLE, "table")
    stack = [iter(container)]
    while stack:
        node = next(stack[-1], None)
        if node is None:
            stack.pop()
            if hidden_from and hidden_from[-1] > len(stack):
                hidden_from.pop()
            continue
        tag = node.tag
        if tag in _ODF_NOT_TEXT or tag in skip:
            continue
        if _odf_hidden_section(node):
            if not keep_hidden:
                continue
            hidden_from.append(len(stack) + 1)
            stack.append(iter(node))
            continue
        if tag in blocks:
            marker = node.find(hidden_paragraph)
            hidden_paragraph_now = (
                marker is not None and _odf_hidden(marker)
            ) or hiding.hides(node, "paragraph")
            if hidden_paragraph_now and not keep_hidden:
                continue
            text = _odf_text(node, keep_hidden=keep_hidden, hiding=hiding)
            hidden = text.hidden or hidden_paragraph_now or bool(hidden_from)
            hidden = hidden and bool(text.strip())
            if tag == heading:
                try:
                    level = int(node.get(_q(_O_TEXT, "outline-level")) or 1)
                except ValueError:
                    level = 1
                yield _Heading(text, level, hidden)
            else:
                yield _Text(text, hidden)
        elif tables and tag == table:
            yield node
        else:
            stack.append(iter(node))


def _odf_rows(
    table: Any,
    what: str,
    *,
    keep_hidden: bool = True,
    hiding: _OdfHiding = _NO_ODF_STYLES,
) -> Iterator[tuple[int, str]]:
    """
    Yield ``(row number, line)`` for each row of a ``table:table`` with a value.

    Repeated cells and rows are expanded; runs of empty ones only move the
    position, so the usual "rest of the sheet" filler costs nothing.

    Raises
    ------
    ValueError
        If cells with a value repeat past ``_MAX_COLUMNS`` columns, or rows
        with a value past ``_MAX_ROWS`` rows; *what* names the table.
    """
    cell_tag = _q(_O_TABLE, "table-cell")
    covered = _q(_O_TABLE, "covered-table-cell")
    number = 0
    row_tag = _q(_O_TABLE, "table-row")
    for row in _walk(table, leaves=frozenset({row_tag})):
        # Rows are not entered: a table nested in a cell is that cell's text.
        if row.tag != row_tag:
            continue
        rows = _repeat(row, "number-rows-repeated")
        cells: list[str] = []
        gap = 0
        for cell in row:
            if cell.tag not in (cell_tag, covered):
                continue
            count = _repeat(cell, "number-columns-repeated")
            text = ""
            if cell.tag == cell_tag:
                text = "\n".join(
                    _odf_blocks(
                        cell, tables=False, keep_hidden=keep_hidden, hiding=hiding
                    )
                ).strip()
            if not text:
                gap += count
                continue
            if len(cells) + gap + count > _MAX_COLUMNS:
                raise ValueError(
                    f"OfficeReader: {what}: row {number + 1} repeats cells past"
                    f" column {_MAX_COLUMNS:,}."
                )
            cells.extend([""] * gap)
            cells.extend([text] * count)
            gap = 0
        if cells:
            if number + rows > _MAX_ROWS:
                raise ValueError(
                    f"OfficeReader: {what}: row {number + 1} is repeated"
                    f" {rows:,} times, past row {_MAX_ROWS:,}."
                )
            line = "\t".join(cells)
            for offset in range(rows):
                yield number + 1 + offset, line
        number += rows


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@dataclass
class OfficeReader(DocumentReader):
    r"""
    Read office documents: Office Open XML and OpenDocument.

    Word documents (``.docx``, ``.docm``, ``.dotx``, ``.dotm``) give their
    body in document order, a chunk per run of paragraphs between tables and
    a chunk per table, then footnotes and endnotes. Presentations
    (``.pptx``, ``.pptm``, ``.potx``, ``.potm``, ``.ppsx``, ``.ppsm``) give
    a chunk per slide in the order of the deck, then, if asked, its notes.
    Workbooks (``.xlsx``, ``.xlsm``, ``.xltx``, ``.xltm``) give each
    worksheet in workbook order, ``rows_per_chunk`` rows a chunk.

    OpenDocument files read the same way: text documents (``.odt``,
    ``.ott``) as paragraphs, tables and notes; spreadsheets (``.ods``,
    ``.ots``) as sheets of rows; presentations and drawings (``.odp``,
    ``.otp``, ``.odg``, ``.otg``) as a chunk per page, then its notes; a
    formula (``.odf``) as its annotation text; and the flat single-XML
    forms (``.fodt``, ``.fods``, ``.fodp``, ``.fodg``).

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the file.
    notes : bool, optional
        Read speaker notes (PowerPoint) and footnotes and endnotes (Word).
        Default: ``True``.
    hidden : bool, optional
        Read slides and sheets marked hidden, and Word text formatted as
        hidden (``w:vanish``). Their chunks carry ``hidden=True`` (slides,
        sheets) or ``has_hidden_text=True`` (Word). ``False`` leaves all of
        them out. Default: ``True``.
    rows_per_chunk : int, optional
        Most spreadsheet rows in one chunk; a longer sheet gives several
        chunks. Default: ``500``.
    headings : bool, optional
        Start a new chunk at each heading of a text document and record
        the heading on the chunks under it. ``False`` reads headings as
        paragraphs. Default: ``True``.
    max_file_bytes : int, optional
        Largest file read. Default: 500 MB.
    max_part_bytes : int, optional
        Largest XML part read, uncompressed: the bound on memory, and on a
        part that a small ZIP inflates into a huge one. Default: 256 MB.
    max_parts : int, optional
        Most parts a package may hold. Default: 100 000.
    chunker, filter\_, filename_override, default_language : optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.

    Attributes
    ----------
    file_types : list of str
        The Office Open XML and OpenDocument extensions above.

    Raises
    ------
    ValueError
        If a limit is not a positive int; at read time, if the file is too
        large, is not a package or a flat OpenDocument file (a legacy
        ``.doc``/``.ppt`` or an encrypted file is named as such), has no
        main part the reader knows, holds a part twice, over a limit or not
        well-formed XML, or repeats spreadsheet cells past a sheet's size.

    See Also
    --------
    scikitplot.corpus._readers.PDFReader : PDF documents.
    scikitplot.corpus._readers.ZipReader : Archives of other files.

    Notes
    -----
    **Chunks.** Every chunk has ``text``, ``raw_text``, ``section_type``,
    ``part`` (the package part it came from) and ``has_macros`` (the
    package holds a VBA project; it is never run). Slides add
    ``page_number`` (0-based position in the deck) and ``hidden``; notes
    add ``page_number`` of their slide. Word tables are ``section_type
    "table"``, cells separated by tabs and rows by lines; footnotes and
    endnotes are ``"footnote"``. Sheet chunks are ``"table"``, rows by line
    and cells by tab at their column, with ``page_number`` (0-based sheet
    position), ``sheet`` (its name), ``row_start``, ``row_end`` (1-based
    row numbers) and ``hidden``. A cell gives its stored value: the cached
    result of a formula (formulas are never computed), dates as their
    serial number, booleans as ``TRUE``/``FALSE``.

    **Hidden text.** Word text formatted as hidden is not shown to a
    reader but is in the file; text placed there to steer a model that
    reads the document is one reason to know. A chunk that holds any
    carries ``has_hidden_text=True`` (the key is absent otherwise), and
    ``hidden=False`` leaves the hidden runs out. OpenDocument text counts
    the same way: a conditional hidden text (``text:hidden-text``), a
    paragraph marked hidden (``text:hidden-paragraph``) and a section not
    displayed (``text:display="none"``, or ``"condition"`` while hidden).
    A table in a hidden section is left out with ``hidden=False`` but not
    flagged otherwise. In Word a style hides text too: a character style,
    the paragraph's style or the document default with ``w:vanish``, each
    through the styles it is based on. In OpenDocument a span or paragraph
    style with ``text:display="none"`` hides too, through its parent
    styles, from ``content.xml`` and ``styles.xml``.

    **Headings.** In Word and OpenDocument text, a heading starts a new
    chunk (its text is the chunk's first line), and that chunk and the
    tables under it carry ``heading`` and ``heading_level`` (1 is the top)
    until the next heading, so a chunk keeps the section it belongs to.
    Word headings are found by outline level, by the built-in ``heading
    1`` to ``heading 9`` styles, and by styles based on them.

    **What decides.** The content, not the extension: a ``.docx`` that is
    a presentation reads as one, and the Strict and Transitional
    conformance classes of Office Open XML read alike.

    **Order.** Slides follow the deck's list (``p:sldIdLst``), not their
    part names, so ``slide11.xml`` can come before ``slide2.xml``.

    **Security.** The package is read in memory; part names never become
    paths. Every XML part is parsed with entity declarations and external
    entities refused (:mod:`._xml_safety`). Relationships to external
    targets, or to targets above the package root, are not followed.
    Tracked deletions are dropped; fields read as their cached text.

    **Memory.** Each part is parsed whole, so a sheet's memory is bounded by
    ``max_part_bytes``, not streamed.

    **OpenDocument specifics.** Cells give their displayed text, so dates
    read as shown. Repeated cells and rows are expanded; a repeat of cells
    or rows *with a value* beyond 16 384 columns or 1 048 576 rows is
    refused. Hidden pages and sheets are not told apart (the flag lives in
    a style), so their chunks carry no ``hidden`` key. A package whose
    manifest lists encrypted parts is refused.

    **Not read (yet):** headers and footers, comments, charts, SmartArt,
    embedded images (no OCR), slide masters and layouts; number formats in
    Excel (dates read as numbers); chartsheets; legacy binary files
    (``.doc``, ``.xls``, ``.ppt``).

    Examples
    --------
    >>> from pathlib import Path
    >>> from scikitplot.corpus._base import DocumentReader
    >>> reader = DocumentReader.create(Path("deck.pptx"), notes=False)
    >>> docs = list(reader.get_documents())
    """

    file_types: ClassVar[list[str] | None] = (
        _WORD_EXTENSIONS
        + _POWERPOINT_EXTENSIONS
        + _EXCEL_EXTENSIONS
        + _OPENDOCUMENT_EXTENSIONS
    )

    notes: bool = field(default=True)
    """Read speaker notes, footnotes and endnotes. Default: ``True``."""

    hidden: bool = field(default=True)
    """Read hidden slides, sheets and Word hidden text. Default: ``True``."""

    rows_per_chunk: int = field(default=500)
    """Most spreadsheet rows in one chunk. Default: ``500``."""

    headings: bool = field(default=True)
    """Split text documents at headings and record them. Default: ``True``."""

    max_file_bytes: int = field(default=500 * 1024 * 1024)
    """Largest file read. Default: 500 MB."""

    max_part_bytes: int = field(default=256 * 1024 * 1024)
    """Largest XML part read, uncompressed. Default: 256 MB."""

    max_parts: int = field(default=100_000)
    """Most parts a package may hold. Default: 100 000."""

    def __post_init__(self) -> None:
        """Validate the options."""
        super().__post_init__()
        for name in ("notes", "hidden", "headings"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(
                    f"OfficeReader: {name} must be a bool; got {getattr(self, name)!r}."
                )
        for name in ("max_file_bytes", "max_part_bytes", "max_parts", "rows_per_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"OfficeReader: {name} must be an int > 0; got {value!r}."
                )

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:  # noqa: PLR0912
        """
        Yield the text of the document, a chunk at a time.

        Yields
        ------
        dict
            See the Notes of :class:`OfficeReader`.

        Raises
        ------
        ValueError
            See :class:`OfficeReader`.
        """
        size = self.input_path.stat().st_size
        if size > self.max_file_bytes:
            raise ValueError(
                f"OfficeReader: {self.file_name} is {size:,} bytes, which"
                f" exceeds max_file_bytes={self.max_file_bytes:,}."
            )
        with open(self.input_path, "rb") as fh:  # noqa: PTH123
            head = fh.read(len(_OLE2_MAGIC))
        if head == _OLE2_MAGIC:
            raise ValueError(
                f"OfficeReader: {self.file_name} is an OLE2 compound file, not"
                " an Office Open XML or OpenDocument package: a legacy binary"
                " document (.doc, .xls, .ppt), or a package encrypted with a"
                " password. Save it as .docx/.xlsx/.pptx or .odt/.ods/.odp"
                " without a password."
            )
        if head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"<"):
            chunks: Iterator[dict[str, Any]] = self._flat_opendocument(size)
            for chunk in chunks:
                chunk["raw_text"] = chunk["text"]
                yield chunk
            return
        try:
            zf = zipfile.ZipFile(self.input_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"OfficeReader: {self.file_name} is not an office document"
                f" package (not a ZIP file: {exc})."
            ) from exc
        with zf:
            pkg = _Package(zf, self.file_name, self.max_part_bytes, self.max_parts)
            main = pkg.first("", _OFFICE_DOCUMENT)
            root = pkg.xml(main) if main else None
            if root is not None:
                has_macros = any(
                    posixpath.basename(n).lower() == "vbaproject.bin"
                    for n in pkg.names()
                )
                if root.tag == _q(_W, "document"):
                    chunks = self._word(pkg, main, root)
                elif root.tag == _q(_P, "presentation"):
                    chunks = self._powerpoint(pkg, main, root)
                elif root.tag == _q(_S, "workbook"):
                    chunks = self._excel(pkg, main, root)
                else:
                    raise ValueError(
                        f"OfficeReader: {self.file_name}: the main part {main!r}"
                        f" is {root.tag!r}, not a Word document, a presentation"
                        " or a workbook."
                    )
            elif pkg.has("content.xml"):
                self._refuse_encrypted(pkg)
                has_macros = any(
                    n.startswith(("Basic/", "Scripts/")) for n in pkg.names()
                )
                chunks = self._opendocument(
                    pkg.xml("content.xml"), "content.xml", pkg.xml("styles.xml")
                )
            else:
                raise ValueError(
                    f"OfficeReader: {self.file_name} has no main document part"
                    " (no officeDocument relationship in _rels/.rels, and no"
                    " content.xml)."
                )
            for chunk in chunks:
                chunk["raw_text"] = chunk["text"]
                chunk["has_macros"] = has_macros
                yield chunk

    # -- OpenDocument --------------------------------------------------------

    def _refuse_encrypted(self, pkg: _Package) -> None:
        manifest = pkg.xml("META-INF/manifest.xml")
        if manifest is not None and any(
            True for _ in manifest.iter(_q(_O_MANIFEST, "encryption-data"))
        ):
            raise ValueError(
                f"OfficeReader: {self.file_name} is encrypted with a password"
                " (its manifest lists encrypted parts). Save it without a"
                " password."
            )

    def _flat_opendocument(self, size: int) -> Iterator[dict[str, Any]]:
        if size > self.max_part_bytes:
            raise ValueError(
                f"OfficeReader: {self.file_name} is {size:,} bytes of XML"
                f" (max_part_bytes={self.max_part_bytes:,})."
            )
        with open(self.input_path, "rb") as fh:  # noqa: PTH123
            root = _parse(fh.read(), self.file_name)
        if root.tag != _q(_O_OFFICE, "document"):
            raise ValueError(
                f"OfficeReader: {self.file_name} is XML with the root"
                f" {root.tag!r}, not a flat OpenDocument file"
                " (office:document) and not a ZIP package."
            )
        scripts = root.find(_q(_O_OFFICE, "scripts"))
        has_macros = scripts is not None and len(scripts) > 0
        for chunk in self._opendocument(root, "flat"):
            chunk["has_macros"] = has_macros
            yield chunk

    def _opendocument(
        self, root: Any, part: str, styles: Any = None
    ) -> Iterator[dict[str, Any]]:
        if root is not None and root.tag.endswith("}math"):
            # A formula: the annotation holds its source text.
            text = "\n".join(
                (a.text or "").strip()
                for a in root.iter()
                if a.tag.endswith("}annotation")
            ).strip()
            if text:
                yield {
                    "text": text,
                    "section_type": SectionType.TEXT.value,
                    "part": part,
                }
            return
        body = None if root is None else root.find(_q(_O_OFFICE, "body"))
        kind = None if body is None or not len(body) else body[0]
        hiding = _OdfHiding(root, styles)
        if kind is None:
            raise ValueError(
                f"OfficeReader: {self.file_name}: {part} has no office:body"
                " with a document in it."
            )
        if kind.tag == _q(_O_OFFICE, "text"):
            yield from self._odf_text_document(kind, part, hiding)
        elif kind.tag == _q(_O_OFFICE, "spreadsheet"):
            tables = kind.findall(_q(_O_TABLE, "table"))
            for number, table in enumerate(tables):
                name = table.get(_q(_O_TABLE, "name")) or f"Sheet{number + 1}"
                rows = _odf_rows(
                    table,
                    f"{self.file_name} sheet {name!r}",
                    keep_hidden=self.hidden,
                    hiding=hiding,
                )
                yield from self._row_chunks(rows, name, number, part, None)
        elif kind.tag in (_q(_O_OFFICE, "presentation"), _q(_O_OFFICE, "drawing")):
            notes_tag = _q(_O_PRESENTATION, "notes")
            for number, page in enumerate(kind.findall(_q(_O_DRAW, "page"))):
                shown = _odf_blocks(
                    page,
                    tables=False,
                    skip=frozenset({notes_tag}),
                    keep_hidden=self.hidden,
                    hiding=hiding,
                )
                text = "\n".join(line for line in shown if line.strip()).strip()
                if text:
                    yield {
                        "text": text,
                        "section_type": SectionType.TEXT.value,
                        "part": part,
                        "page_number": number,
                    }
                notes = page.find(notes_tag)
                if not self.notes or notes is None:
                    continue
                spoken = "\n".join(
                    line
                    for line in _odf_blocks(
                        notes, tables=False, keep_hidden=self.hidden, hiding=hiding
                    )
                    if line.strip()
                ).strip()
                if spoken:
                    yield {
                        "text": spoken,
                        "section_type": SectionType.FOOTNOTE.value,
                        "part": part,
                        "page_number": number,
                    }
        else:
            raise ValueError(
                f"OfficeReader: {self.file_name}: {part} holds {kind.tag!r},"
                " not a text document, a spreadsheet, a presentation or a"
                " drawing."
            )

    def _odf_text_document(
        self, body: Any, part: str, hiding: _OdfHiding = _NO_ODF_STYLES
    ) -> Iterator[dict[str, Any]]:
        def blocks() -> Iterator[Any]:
            for block in _odf_blocks(
                body, tables=True, keep_hidden=self.hidden, hiding=hiding
            ):
                if isinstance(block, str):
                    yield block
                else:
                    rows = _odf_rows(
                        block,
                        self.file_name,
                        keep_hidden=self.hidden,
                        hiding=hiding,
                    )
                    yield ("table", "\n".join(line for _, line in rows))

        yield from self._sections(blocks(), part)
        if not self.notes:
            return
        for note in body.iter(_q(_O_TEXT, "note")):
            note_body = note.find(_q(_O_TEXT, "note-body"))
            if note_body is None:
                continue
            joined = _joined(
                list(
                    _odf_blocks(
                        note_body,
                        tables=False,
                        keep_hidden=self.hidden,
                        hiding=hiding,
                    )
                )
            )
            if joined.strip():
                chunk = {
                    "text": joined.strip(),
                    "section_type": SectionType.FOOTNOTE.value,
                    "part": part,
                }
                if joined.hidden:
                    chunk["has_hidden_text"] = True
                yield chunk

    # -- Word ---------------------------------------------------------------

    def _sections(self, blocks: Iterator[Any], part: str) -> Iterator[dict[str, Any]]:
        """
        Yield body chunks from *blocks*: paragraphs, headings and tables.

        *blocks* holds ``str`` (a paragraph), :class:`_Heading` and
        ``("table", text)``. With ``headings=True`` a heading starts a new
        chunk and every chunk until the next heading carries ``heading``
        and ``heading_level``.
        """
        paragraphs: list[str] = []
        current: dict[str, Any] = {}

        def made(text: Any, kind: SectionType, hidden: bool) -> dict[str, Any]:
            chunk = {"text": str(text), "section_type": kind.value, "part": part}
            chunk.update(current)
            if hidden:
                chunk["has_hidden_text"] = True
            return chunk

        def flush() -> Iterator[dict[str, Any]]:
            joined = _joined(paragraphs)
            paragraphs.clear()
            if joined.strip():
                yield made(joined.strip(), SectionType.TEXT, joined.hidden)

        for block in blocks:
            if isinstance(block, tuple):
                yield from flush()
                if block[1]:
                    hidden = getattr(block[1], "hidden", False)
                    yield made(block[1], SectionType.TABLE, hidden)
                continue
            if self.headings and isinstance(block, _Heading) and block.strip():
                yield from flush()
                current = {"heading": block.strip(), "heading_level": block.level}
            paragraphs.append(block)
        yield from flush()

    def _word(self, pkg: _Package, main: str, root: Any) -> Iterator[dict[str, Any]]:
        body = root.find(_q(_W, "body"))
        styles_part = pkg.first(main, _STYLES)
        styles = pkg.xml(styles_part) if styles_part else None
        levels = (
            _word_style_levels(styles) if self.headings and styles is not None else {}
        )
        hiding = _WordHiding(styles)

        def paragraph(p: Any) -> Any:
            text = _word_paragraph(p, keep_hidden=self.hidden, hiding=hiding)
            level = _word_heading(p, levels) if self.headings else None
            return text if level is None else _Heading(text, level, text.hidden)

        def blocks() -> Iterator[Any]:
            for node in [] if body is None else body:
                if node.tag == _q(_W, "p"):
                    yield paragraph(node)
                elif node.tag == _q(_W, "tbl"):
                    table = _word_table(node, keep_hidden=self.hidden, hiding=hiding)
                    yield ("table", table)
                elif node.tag == _q(_W, "sdt"):
                    # A content control: its paragraphs read as body text.
                    p_tag = _q(_W, "p")
                    for p in _walk(node, skip=_WORD_REMOVED, leaves=frozenset({p_tag})):
                        if p.tag == p_tag:
                            yield paragraph(p)

        yield from self._sections(blocks(), main)
        if not self.notes:
            return
        for kind, element in ((_FOOTNOTES, "footnote"), (_ENDNOTES, "endnote")):
            part = pkg.first(main, kind)
            notes_root = pkg.xml(part) if part else None
            if notes_root is None:
                continue
            for note in notes_root.iter(_q(_W, element)):
                if note.get(_q(_W, "type")) in ("separator", "continuationSeparator"):
                    continue
                joined = _joined(
                    _word_paragraphs(note, keep_hidden=self.hidden, hiding=hiding)
                )
                if joined.strip():
                    chunk = {
                        "text": joined.strip(),
                        "section_type": SectionType.FOOTNOTE.value,
                        "part": part,
                    }
                    if joined.hidden:
                        chunk["has_hidden_text"] = True
                    yield chunk

    # -- PowerPoint ----------------------------------------------------------

    def _powerpoint(
        self, pkg: _Package, main: str, root: Any
    ) -> Iterator[dict[str, Any]]:
        rels = pkg.relationships(main)
        id_list = root.find(_q(_P, "sldIdLst"))
        order = [] if id_list is None else id_list.findall(_q(_P, "sldId"))
        slides = []
        for entry in order:
            kind, target = rels.get(entry.get(_q(_DOC_REL, "id")) or "", ("", ""))
            if kind != _SLIDE:
                logger.warning(
                    "OfficeReader: %s: slide id %r has no slide part; skipped.",
                    self.file_name,
                    entry.get("id"),
                )
                continue
            slides.append(target)
        for number, part in enumerate(slides):
            slide = pkg.xml(part)
            if slide is None:
                logger.warning(
                    "OfficeReader: %s: slide part %r is missing; skipped.",
                    self.file_name,
                    part,
                )
                continue
            hidden = slide.get("show") in ("0", "false")
            if hidden and not self.hidden:
                continue
            text = _slide_text(slide)
            if text:
                yield {
                    "text": text,
                    "section_type": SectionType.TEXT.value,
                    "part": part,
                    "page_number": number,
                    "hidden": hidden,
                }
            if not self.notes:
                continue
            notes_part = pkg.first(part, _NOTES_SLIDE)
            notes_root = pkg.xml(notes_part) if notes_part else None
            if notes_root is None:
                continue
            notes_text = _notes_text(notes_root)
            if notes_text:
                yield {
                    "text": notes_text,
                    "section_type": SectionType.FOOTNOTE.value,
                    "part": notes_part,
                    "page_number": number,
                    "hidden": hidden,
                }

    # -- Excel ---------------------------------------------------------------

    def _excel(self, pkg: _Package, main: str, root: Any) -> Iterator[dict[str, Any]]:
        rels = pkg.relationships(main)
        shared: list[str] = []
        for kind, target in rels.values():
            if kind == _SHARED_STRINGS:
                strings = pkg.xml(target)
                if strings is not None:
                    shared = [_string_item(si) for si in strings.iter(_q(_S, "si"))]
                break
        sheets = root.find(_q(_S, "sheets"))
        for number, sheet in enumerate([] if sheets is None else sheets):
            name = sheet.get("name") or f"Sheet{number + 1}"
            kind, part = rels.get(sheet.get(_q(_DOC_REL, "id")) or "", ("", ""))
            if kind != _WORKSHEET:
                # A chartsheet or a dialog sheet: no cells to read.
                continue
            hidden = sheet.get("state") in ("hidden", "veryHidden")
            if hidden and not self.hidden:
                continue
            ws = pkg.xml(part)
            if ws is None:
                logger.warning(
                    "OfficeReader: %s: sheet %r part %r is missing; skipped.",
                    self.file_name,
                    name,
                    part,
                )
                continue
            yield from self._row_chunks(
                _sheet_rows(ws, shared), name, number, part, hidden
            )

    def _row_chunks(
        self,
        rows: Iterator[tuple[int, str]],
        name: str,
        number: int,
        part: str,
        hidden: bool | None,
    ) -> Iterator[dict[str, Any]]:
        """Yield a sheet's rows as chunks of ``rows_per_chunk`` rows."""
        block: list[tuple[int, str]] = []

        def chunk() -> dict[str, Any]:
            made = {
                "text": "\n".join(line for _, line in block),
                "section_type": SectionType.TABLE.value,
                "part": part,
                "page_number": number,
                "sheet": name,
                "row_start": block[0][0],
                "row_end": block[-1][0],
            }
            if hidden is not None:
                made["hidden"] = hidden
            return made

        for row in rows:
            block.append(row)
            if len(block) == self.rows_per_chunk:
                yield chunk()
                block = []
        if block:
            yield chunk()


def _notes_text(root: Any) -> str:
    """Return the body text of a notes slide (not its slide image or number)."""
    lines: list[str] = []
    for sp in _walk(root):
        if sp.tag != _q(_P, "sp"):
            continue
        ph = sp.find(f"{_q(_P, 'nvSpPr')}/{_q(_P, 'nvPr')}/{_q(_P, 'ph')}")
        if ph is None or ph.get("type") != "body":
            continue
        lines.extend(_drawing_paragraphs(sp))
    return "\n".join(line for line in lines if line.strip()).strip()
