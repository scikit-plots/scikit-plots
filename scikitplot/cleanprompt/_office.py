"""
Read the text out of Word, Excel and PowerPoint files with the standard library.

``.docx``, ``.xlsx`` and ``.pptx`` are zip archives of XML (ECMA-376, Office
Open XML). Their text is in a handful of well-known parts, so extracting it
needs :mod:`zipfile` and :mod:`xml.etree.ElementTree` and nothing else.

Notes
-----
**User notes.** The output is **text**, not a rebuilt Office file: paragraphs
one per line for a document, one tab-separated row per line for a spreadsheet
(with a ``## sheet`` heading per sheet), and one line per text box for a
presentation. That text is what gets redacted and sent. A spreadsheet's first
row is kept as its header, so a pack's field rules apply to each column by
name exactly as they do to a CSV.

**Developer notes — why this lives here and not in corpus.**

The plan was to borrow ``scikitplot.corpus``'s readers for binary formats, and
for PDF and archives that is what happens. But corpus registers no reader for
``.docx``, ``.xlsx`` or ``.pptx`` — its supported types were measured, not
assumed — and the user asked for all three. Rather than leave them unsupported
or add a compiled dependency, they are read here from the XML they are made
of. :func:`register_corpus_readers` in ``_corpus.py`` hands these same
extractors *to* corpus, so the gap closes on both sides.

**Developer notes — the XML is untrusted, and the defence is total.**

A crafted document can hold an XML entity bomb or an external entity. Office
Open XML never uses a DTD, so any part containing ``<!DOCTYPE`` or
``<!ENTITY`` markup is **refused** before it is parsed. Without a DTD there are
no entities to expand and nothing external to fetch; this is a structural
guarantee rather than a limit tuned to catch today's payloads.

The check reads the part with its NUL bytes removed. ECMA-376 allows a part to
be UTF-16 as well as UTF-8, and in UTF-16 the markup ``<!DOCTYPE`` is stored
with a NUL beside every character, so a search of the raw bytes saw nothing
and the parser then expanded the entities (measured: ``&e;`` came back as its
replacement text). Removing NULs makes both encodings read the same to the
check. Every other encoding the parser accepts keeps XML's own characters at
their ASCII values, so there is no third spelling to miss.

Size is bounded twice: a part's *declared* size is checked before it is
opened, and the bytes actually decompressed are counted while reading, because
the declared size is attacker-controlled. Parts are parsed incrementally with
:func:`xml.etree.ElementTree.iterparse` and cleared as they are consumed, so a
large sheet streams rather than sitting in memory as a tree.

See Also
--------
scikitplot.cleanprompt._corpus : PDF and media through corpus.
scikitplot.cleanprompt._structured : Reads the extracted spreadsheet rows as fields.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterator

# The standard-library parser is used deliberately. The base tier is standard
# library only (CP-054), and the hazard a linter names here - entity expansion
# and external entities - needs a DTD, which ``_DTD`` refuses in every part
# before the parser sees a byte. ``defusedxml`` would add an undeclared
# dependency to guard a door that is already shut.
from xml.etree import ElementTree as ET  # noqa: N817, S405

from ._exceptions import CleanPromptError

__all__ = [
    "OFFICE_EXTENSIONS",
    "OfficeLimits",
    "extract_office_text",
]

#: Extensions this module reads, mapped to the kind of package.
OFFICE_EXTENSIONS = {
    ".docx": "docx",
    ".docm": "docx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".pptx": "pptx",
}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

#: Markup that introduces a DTD. Office Open XML never contains one. Searched
#: in the part with NUL bytes removed, so UTF-16 is read as well as UTF-8.
_DTD = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)", re.IGNORECASE)

#: Word parts, in reading order. Comments are read because they are where
#: reviewers write names.
_DOCX_PARTS = (
    re.compile(r"^word/document\.xml$"),
    re.compile(r"^word/header\d*\.xml$"),
    re.compile(r"^word/footer\d*\.xml$"),
    re.compile(r"^word/footnotes\.xml$"),
    re.compile(r"^word/endnotes\.xml$"),
    re.compile(r"^word/comments\.xml$"),
)


class OfficeLimits:
    """
    Bounds on what one Office file may cost to read.

    Parameters
    ----------
    max_members : int, default=10000
        Parts in the package.
    max_part_bytes : int, default=64 MiB
        Uncompressed bytes in any one XML part.
    max_total_bytes : int, default=256 MiB
        Uncompressed bytes read across the whole package.

    Notes
    -----
    **Developer notes.** A class with slots rather than a dataclass so the
    defaults are visible in the signature a user reads, and so it can be passed
    through the fluent plan without becoming part of its fingerprint.
    """

    __slots__ = ("max_members", "max_part_bytes", "max_total_bytes")

    def __init__(
        self,
        max_members: int = 10_000,
        max_part_bytes: int = 64 * 1024 * 1024,
        max_total_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        for name, value in (
            ("max_members", max_members),
            ("max_part_bytes", max_part_bytes),
            ("max_total_bytes", max_total_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                msg = f"OfficeLimits.{name} must be a positive integer, got {value!r}"
                raise CleanPromptError(msg)
        self.max_members = max_members
        self.max_part_bytes = max_part_bytes
        self.max_total_bytes = max_total_bytes


class _Budget:
    """
    Count decompressed bytes across an archive and refuse past the limit.

    Notes
    -----
    **Developer notes.** Shared by the Office reader and by archive redaction
    in ``_runtime.py``: both read untrusted zip members, and a zip bomb is the
    same attack whichever of them opens it.
    """

    __slots__ = ("limits", "used")

    def __init__(self, limits: OfficeLimits) -> None:
        self.limits = limits
        self.used = 0

    def read(
        self, archive: zipfile.ZipFile, info: zipfile.ZipInfo, xml: bool = True
    ) -> bytes:
        """
        Return one member's bytes, counting them against the budget.

        Parameters
        ----------
        archive : zipfile.ZipFile
            The open archive.
        info : zipfile.ZipInfo
            The member.
        xml : bool, default=True
            Whether the member is an XML part, which must carry no DTD.

        Returns
        -------
        bytes
            The whole member.

        Raises
        ------
        CleanPromptError
            On a size limit, or a DTD in an XML part.
        """
        if info.file_size > self.limits.max_part_bytes:
            msg = (
                f"part {info.filename!r} declares {info.file_size} bytes, above "
                f"the limit of {self.limits.max_part_bytes}; nothing was read"
            )
            raise CleanPromptError(msg)
        chunks = []
        taken = 0
        with archive.open(info) as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                taken += len(chunk)
                self.used += len(chunk)
                if (
                    taken > self.limits.max_part_bytes
                    or self.used > self.limits.max_total_bytes
                ):
                    msg = (
                        f"part {info.filename!r} decompresses past the size limit; "
                        "the file was refused rather than partly read"
                    )
                    raise CleanPromptError(msg)
                chunks.append(chunk)
        data = b"".join(chunks)
        if xml and _DTD.search(data.replace(b"\x00", b"")):
            msg = (
                f"part {info.filename!r} contains a DTD, which Office Open XML "
                "never uses; the file was refused as potentially hostile"
            )
            raise CleanPromptError(msg)
        return data


def _iter(data: bytes, tag: str) -> Iterator[ET.Element]:
    """Yield each completed element named ``tag``, clearing it afterwards."""
    # ``data`` came through ``_Budget.read``, which refused any DTD (see _DTD).
    for _event, element in ET.iterparse(  # noqa: S314
        io.BytesIO(data), events=("end",)
    ):
        if element.tag == tag:
            yield element
            element.clear()


def _paragraph_text(paragraph: ET.Element) -> str:
    """Return a Word paragraph's text, with tabs and breaks kept."""
    parts = []
    for node in paragraph.iter():
        if node.tag == f"{_W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts)


def _docx_part(data: bytes) -> list[str]:
    """
    Return one Word part's lines: a paragraph per line, a table row per line.

    Notes
    -----
    **Developer notes.** A table row is written as its cells joined by a tab,
    because that is what keeps a two-column table readable as fields: a row
    ``DOB | 1984-03-02`` becomes ``DOB<tab>1984-03-02``, and the reader that
    follows treats the tab as a separator, so the date is found by its label.
    Emitting each cell as its own line would separate every label from its
    value.
    """
    lines: list[str] = []
    depth = 0
    row: list[str] = []
    cell: list[str] = []
    # ``data`` came through ``_Budget.read``, which refused any DTD (see _DTD).
    for event, element in ET.iterparse(  # noqa: S314
        io.BytesIO(data), events=("start", "end")
    ):
        tag = element.tag
        if event == "start":
            if tag == f"{_W}tr":
                depth += 1
                row = []
            elif tag == f"{_W}tc" and depth:
                cell = []
            continue
        if tag == f"{_W}p":
            text = _paragraph_text(element)
            if depth:
                if text.strip():
                    cell.append(text)
            elif text.strip():
                lines.append(text)
            element.clear()
        elif tag == f"{_W}tc" and depth:
            row.append(" ".join(cell).replace("\t", " "))
        elif tag == f"{_W}tr" and depth:
            depth -= 1
            if any(value.strip() for value in row):
                lines.append("\t".join(row))
            element.clear()
    return lines


def _docx(archive: zipfile.ZipFile, budget: _Budget) -> str:
    """Return a Word document's paragraphs and table rows, one per line."""
    names = archive.namelist()
    lines: list[str] = []
    for pattern in _DOCX_PARTS:
        for name in sorted(n for n in names if pattern.match(n)):
            lines.extend(_docx_part(budget.read(archive, archive.getinfo(name))))
    return "\n".join(lines) + ("\n" if lines else "")


def _column_index(reference: str) -> int:
    """Return the zero-based column of a cell reference such as ``AB12``."""
    letters = re.match(r"[A-Z]+", reference or "")
    if not letters:
        return -1
    index = 0
    for char in letters.group():
        index = index * 26 + (ord(char) - 64)
    return index - 1


def _sheet_number(name: str) -> int:
    """Return the number in ``xl/worksheets/sheetN.xml`` for ordering."""
    found = re.search(r"(\d+)\.xml$", name)
    return int(found.group(1)) if found else 0


def _xlsx(archive: zipfile.ZipFile, budget: _Budget) -> str:
    """Return every sheet as tab-separated rows under a ``## sheet`` heading."""
    names = archive.namelist()
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        shared.extend(
            "".join(node.text or "" for node in item.iter(f"{_S}t"))
            for item in _iter(
                budget.read(
                    archive,
                    archive.getinfo("xl/sharedStrings.xml"),
                ),
                f"{_S}si",
            )
        )
    sheets = sorted(
        (n for n in names if re.match(r"^xl/worksheets/sheet\d+\.xml$", n)),
        key=_sheet_number,
    )
    out: list[str] = []
    for sheet in sheets:
        out.append(f"## sheet {_sheet_number(sheet)}")
        for row in _iter(budget.read(archive, archive.getinfo(sheet)), f"{_S}row"):
            cells: dict[int, str] = {}
            for cell in row.iter(f"{_S}c"):
                kind = cell.get("t")
                value = cell.find(f"{_S}v")
                if kind == "s" and value is not None and value.text is not None:
                    position = int(value.text)
                    text = shared[position] if 0 <= position < len(shared) else ""
                elif kind == "inlineStr":
                    text = "".join(node.text or "" for node in cell.iter(f"{_S}t"))
                else:
                    text = value.text if value is not None and value.text else ""
                column = _column_index(cell.get("r", ""))
                if column < 0:
                    column = len(cells)
                # A tab or newline inside a cell would break the row it is in,
                # so it is folded to a space in the extracted text.
                cells[column] = re.sub(r"[\t\r\n]+", " ", text)
            if cells:
                width = max(cells) + 1
                out.append("\t".join(cells.get(index, "") for index in range(width)))
    return "\n".join(out) + ("\n" if out else "")


def _pptx(archive: zipfile.ZipFile, budget: _Budget) -> str:
    """Return every slide's text boxes, then its speaker notes."""
    names = archive.namelist()
    out: list[str] = []
    for folder, label in (
        ("ppt/slides/slide", "slide"),
        ("ppt/notesSlides/notesSlide", "notes"),
    ):
        parts = sorted(
            (n for n in names if n.startswith(folder) and n.endswith(".xml")),
            key=_sheet_number,
        )
        for part in parts:
            out.append(f"## {label} {_sheet_number(part)}")
            for paragraph in _iter(
                budget.read(archive, archive.getinfo(part)), f"{_A}p"
            ):
                text = "".join(node.text or "" for node in paragraph.iter(f"{_A}t"))
                if text.strip():
                    out.append(text)
    return "\n".join(out) + ("\n" if out else "")


def extract_office_text(
    data: bytes,
    kind: str,
    limits: OfficeLimits | None = None,
    budget: _Budget | None = None,
) -> str:
    r"""
    Return the text of an Office Open XML package.

    Parameters
    ----------
    data : bytes
        The file's bytes.
    kind : {'docx', 'xlsx', 'pptx'}
        Which kind of package it is.
    limits : OfficeLimits, optional
        Size bounds. Defaults are generous for documents people write and
        refuse the ones built to exhaust a machine.
    budget : _Budget, optional
        A byte budget already in use — an enclosing archive's. When given,
        the package's parts count against it, so a zip of Office files is
        bounded in total and not per member (``limits`` is then ignored).

    Returns
    -------
    str
        The extracted text, newline-terminated when non-empty.

    Raises
    ------
    CleanPromptError
        If the bytes are not a zip archive, a limit is exceeded, a part holds a
        DTD, or an XML part does not parse. Nothing partial is returned.

    Examples
    --------
    >>> import io, zipfile
    >>> buffer = io.BytesIO()
    >>> with zipfile.ZipFile(buffer, "w") as z:
    ...     z.writestr(
    ...         "word/document.xml",
    ...         '<w:document xmlns:w="http://schemas.openxmlformats.org/'
    ...         'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello '
    ...         "Marion</w:t></w:r></w:p></w:body></w:document>",
    ...     )
    >>> extract_office_text(buffer.getvalue(), "docx")
    'Hello Marion\n'
    """
    readers = {"docx": _docx, "xlsx": _xlsx, "pptx": _pptx}
    if kind not in readers:
        msg = f"unknown Office package kind {kind!r}; choose from {', '.join(sorted(readers))}"
        raise CleanPromptError(msg)
    budget = budget if budget is not None else _Budget(limits or OfficeLimits())
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if len(archive.infolist()) > budget.limits.max_members:
                msg = (
                    f"package has more than {budget.limits.max_members} parts; refused"
                )
                raise CleanPromptError(msg)
            return readers[kind](archive, budget)
    except zipfile.BadZipFile as exc:
        msg = f"not a valid {kind} file (it is not a zip archive): {exc}"
        raise CleanPromptError(msg) from exc
    except ET.ParseError as exc:
        msg = f"an XML part of this {kind} file does not parse: {exc}"
        raise CleanPromptError(msg) from exc
