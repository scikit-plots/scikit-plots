# scikitplot/corpus/_readers/_office.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Office Open XML documents: Word and PowerPoint (``.docx``, ``.pptx`` and kin).

An Office Open XML file is a ZIP package of XML parts. This reader opens the
package in memory, follows its relationships to the parts that hold text and
parses each with the hardened XML parser. Nothing is extracted to disk and
nothing in the package (macros, OLE objects, external links) is run.

Python 3.8-3.15. Standard library only.
"""

from __future__ import annotations

import logging
import posixpath
import zipfile
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

#: Relationship types, by their last path segment.
_OFFICE_DOCUMENT = "officeDocument"
_SLIDE = "slide"
_NOTES_SLIDE = "notesSlide"
_FOOTNOTES = "footnotes"
_ENDNOTES = "endnotes"

#: Start of an OLE2 compound file: a legacy binary or an encrypted package.
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_WORD_EXTENSIONS = [".docx", ".docm", ".dotx", ".dotm"]
_POWERPOINT_EXTENSIONS = [".pptx", ".pptm", ".potx", ".potm", ".ppsx", ".ppsm"]


def _q(namespace: str, name: str) -> str:
    """Return the ElementTree (Clark) name ``{namespace}name``."""
    return f"{{{namespace}}}{name}"


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
        with self._zf.open(info) as src:
            data = src.read(self._max + 1)
        if len(data) > self._max:
            # The declared size understated the content.
            raise ValueError(
                f"OfficeReader: part {part!r} of {self._name} is larger than"
                f" max_part_bytes={self._max:,}."
            )
        try:
            return parse_stdlib_secure(data)
        except ValueError:
            raise
        except Exception as exc:  # expat errors are not ValueError
            raise ValueError(
                f"OfficeReader: part {part!r} of {self._name} is not"
                f" well-formed XML: {exc}"
            ) from exc

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


def _word_paragraph(p: Any) -> str:
    """Return the text of a ``w:p``: runs, tabs and breaks; deletions dropped."""
    out: list[str] = []
    t, tab, br, cr = (_q(_W, n) for n in ("t", "tab", "br", "cr"))
    hyphen, deleted = _q(_W, "noBreakHyphen"), _q(_W, "del")
    stack = [iter(p)]
    while stack:
        node = next(stack[-1], None)
        if node is None:
            stack.pop()
            continue
        tag = node.tag
        if tag == deleted:
            continue  # tracked deletion: not part of the current text
        if tag == t:
            out.append(node.text or "")
        elif tag == tab:
            out.append("\t")
        elif tag in (br, cr):
            out.append("\n")
        elif tag == hyphen:
            out.append("-")
        stack.append(iter(node))
    return "".join(out)


def _word_table(tbl: Any) -> str:
    """Return a ``w:tbl`` as lines of tab-separated cells; empty rows dropped."""
    rows = []
    for tr in tbl.iter(_q(_W, "tr")):
        cells = [
            "\n".join(_word_paragraph(p) for p in tc.iter(_q(_W, "p"))).strip()
            for tc in tr.findall(_q(_W, "tc"))
        ]
        if any(cells):
            rows.append("\t".join(cells))
    return "\n".join(rows)


def _drawing_paragraph(p: Any) -> str:
    """Return the text of one DrawingML ``a:p``: its runs and line breaks."""
    t, br = _q(_A, "t"), _q(_A, "br")
    parts = []
    for node in p.iter():
        if node.tag == t:
            parts.append(node.text or "")
        elif node.tag == br:
            parts.append("\n")
    return "".join(parts)


def _drawing_paragraphs(container: Any) -> list[str]:
    """Return the ``a:p`` paragraphs under *container* in document order."""
    return [_drawing_paragraph(p) for p in container.iter(_q(_A, "p"))]


def _slide_text(root: Any) -> str:
    """Return a slide's text: shapes in document order, tables as rows."""
    lines: list[str] = []
    tbl = _q(_A, "tbl")
    tables = list(root.iter(tbl))
    in_table = {id(p) for table in tables for p in table.iter(_q(_A, "p"))}
    for node in root.iter():
        if node.tag == tbl:
            for tr in node.iter(_q(_A, "tr")):
                cells = [
                    "\n".join(_drawing_paragraphs(tc)).strip()
                    for tc in tr.findall(_q(_A, "tc"))
                ]
                if any(cells):
                    lines.append("\t".join(cells))
        elif node.tag == _q(_A, "p") and id(node) not in in_table:
            lines.append(_drawing_paragraph(node))
    return "\n".join(line for line in lines if line.strip()).strip()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@dataclass
class OfficeReader(DocumentReader):
    r"""
    Read Word and PowerPoint files in the Office Open XML format.

    Word documents (``.docx``, ``.docm``, ``.dotx``, ``.dotm``) give their
    body in document order, a chunk per run of paragraphs between tables and
    a chunk per table, then footnotes and endnotes. Presentations
    (``.pptx``, ``.pptm``, ``.potx``, ``.potm``, ``.ppsx``, ``.ppsm``) give
    a chunk per slide in the order of the deck, then, if asked, its notes.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the file.
    notes : bool, optional
        Read speaker notes (PowerPoint) and footnotes and endnotes (Word).
        Default: ``True``.
    hidden_slides : bool, optional
        Read slides marked hidden. Their chunks carry ``hidden=True``.
        Default: ``True``.
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
        The Word and PowerPoint extensions above.

    Raises
    ------
    ValueError
        If a limit is not a positive int; at read time, if the file is too
        large, is not a ZIP package (a legacy ``.doc``/``.ppt`` or an
        encrypted file is named as such), has no Word or PowerPoint main
        part, or holds a part over a limit or not well-formed XML.

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
    endnotes are ``"footnote"``.

    **Order.** Slides follow the deck's list (``p:sldIdLst``), not their
    part names, so ``slide11.xml`` can come before ``slide2.xml``.

    **Security.** The package is read in memory; part names never become
    paths. Every XML part is parsed with entity declarations and external
    entities refused (:mod:`._xml_safety`). Relationships to external
    targets, or to targets above the package root, are not followed.
    Tracked deletions are dropped; fields read as their cached text.

    **Not read (yet):** headers and footers, comments, charts, SmartArt,
    embedded images (no OCR), slide masters and layouts.

    Examples
    --------
    >>> from pathlib import Path
    >>> from scikitplot.corpus._base import DocumentReader
    >>> reader = DocumentReader.create(Path("deck.pptx"), notes=False)
    >>> docs = list(reader.get_documents())
    """

    file_types: ClassVar[list[str] | None] = _WORD_EXTENSIONS + _POWERPOINT_EXTENSIONS

    notes: bool = field(default=True)
    """Read speaker notes, footnotes and endnotes. Default: ``True``."""

    hidden_slides: bool = field(default=True)
    """Read slides marked hidden. Default: ``True``."""

    max_file_bytes: int = field(default=500 * 1024 * 1024)
    """Largest file read. Default: 500 MB."""

    max_part_bytes: int = field(default=256 * 1024 * 1024)
    """Largest XML part read, uncompressed. Default: 256 MB."""

    max_parts: int = field(default=100_000)
    """Most parts a package may hold. Default: 100 000."""

    def __post_init__(self) -> None:
        """Validate the options."""
        super().__post_init__()
        for name in ("notes", "hidden_slides"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(
                    f"OfficeReader: {name} must be a bool; got {getattr(self, name)!r}."
                )
        for name in ("max_file_bytes", "max_part_bytes", "max_parts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"OfficeReader: {name} must be an int > 0; got {value!r}."
                )

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
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
                " an Office Open XML package: a legacy binary document (.doc,"
                " .ppt) under another name, or a package encrypted with a"
                " password. Save it as .docx/.pptx without a password."
            )
        try:
            zf = zipfile.ZipFile(self.input_path)
        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"OfficeReader: {self.file_name} is not an Office Open XML"
                f" package (not a ZIP file: {exc})."
            ) from exc
        with zf:
            pkg = _Package(zf, self.file_name, self.max_part_bytes, self.max_parts)
            has_macros = any(
                posixpath.basename(n).lower() == "vbaproject.bin" for n in pkg.names()
            )
            main = pkg.first("", _OFFICE_DOCUMENT)
            root = pkg.xml(main) if main else None
            if root is None:
                raise ValueError(
                    f"OfficeReader: {self.file_name} has no main document part"
                    " (no officeDocument relationship in _rels/.rels)."
                )
            if root.tag == _q(_W, "document"):
                chunks = self._word(pkg, main, root)
            elif root.tag == _q(_P, "presentation"):
                chunks = self._powerpoint(pkg, main, root)
            else:
                raise ValueError(
                    f"OfficeReader: {self.file_name}: the main part {main!r} is"
                    f" {root.tag!r}, not a Word document or a presentation."
                )
            for chunk in chunks:
                chunk["raw_text"] = chunk["text"]
                chunk["has_macros"] = has_macros
                yield chunk

    # -- Word ---------------------------------------------------------------

    def _word(self, pkg: _Package, main: str, root: Any) -> Iterator[dict[str, Any]]:
        body = root.find(_q(_W, "body"))
        paragraphs: list[str] = []

        def flush() -> Iterator[dict[str, Any]]:
            text = "\n".join(paragraphs).strip()
            paragraphs.clear()
            if text:
                yield {
                    "text": text,
                    "section_type": SectionType.TEXT.value,
                    "part": main,
                }

        for node in [] if body is None else body:
            if node.tag == _q(_W, "p"):
                paragraphs.append(_word_paragraph(node))
            elif node.tag == _q(_W, "tbl"):
                yield from flush()
                table = _word_table(node)
                if table:
                    yield {
                        "text": table,
                        "section_type": SectionType.TABLE.value,
                        "part": main,
                    }
            elif node.tag == _q(_W, "sdt"):
                # A content control: its paragraphs read as body text.
                paragraphs.extend(_word_paragraph(p) for p in node.iter(_q(_W, "p")))
        yield from flush()
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
                text = "\n".join(
                    _word_paragraph(p) for p in note.iter(_q(_W, "p"))
                ).strip()
                if text:
                    yield {
                        "text": text,
                        "section_type": SectionType.FOOTNOTE.value,
                        "part": part,
                    }

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
            if hidden and not self.hidden_slides:
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


def _notes_text(root: Any) -> str:
    """Return the body text of a notes slide (not its slide image or number)."""
    lines: list[str] = []
    for sp in root.iter(_q(_P, "sp")):
        ph = sp.find(f"{_q(_P, 'nvSpPr')}/{_q(_P, 'nvPr')}/{_q(_P, 'ph')}")
        if ph is None or ph.get("type") != "body":
            continue
        lines.extend(_drawing_paragraphs(sp))
    return "\n".join(line for line in lines if line.strip()).strip()
