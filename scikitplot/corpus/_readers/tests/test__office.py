# corpus/_readers/tests/test__office.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
OfficeReader: Word and PowerPoint in the Office Open XML format.

Every package is built here with ``zipfile``; no binary fixture is stored.

Run with::

    pytest scikitplot/corpus/_readers/tests/test__office.py -v
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ..._base import DocumentReader
from .._office import OfficeReader

PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _rels(*rels: tuple) -> str:
    """Relationships ``(id, type, target)`` or ``(id, type, target, external)``."""
    items = "".join(
        f'<Relationship Id="{rel[0]}" Type="{REL}/{rel[1]}" Target="{rel[2]}"'
        + (' TargetMode="External"' if rel[3:] and rel[3] else "")
        + "/>"
        for rel in rels
    )
    return f'<?xml version="1.0"?><Relationships xmlns="{PKG}">{items}</Relationships>'


def _write(path: Path, parts: dict) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return path


def _p(*runs: str) -> str:
    return "<w:p>" + "".join(f"<w:r><w:t>{r}</w:t></w:r>" for r in runs) + "</w:p>"


def _docx(path: Path, body: str, extra: dict | None = None, doc_rels=()) -> Path:
    parts = {
        "_rels/.rels": _rels(("rId1", "officeDocument", "word/document.xml")),
        "word/document.xml": (
            f'<?xml version="1.0"?><w:document xmlns:w="{W}">'
            f"<w:body>{body}</w:body></w:document>"
        ),
    }
    if doc_rels:
        parts["word/_rels/document.xml.rels"] = _rels(*doc_rels)
    parts.update(extra or {})
    return _write(path, parts)


def _slide(*paragraphs: str, show: str | None = None, table: str = "") -> str:
    attr = f' show="{show}"' if show is not None else ""
    paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in paragraphs)
    return (
        f'<?xml version="1.0"?><p:sld xmlns:p="{P}" xmlns:a="{A}"{attr}>'
        f"<p:cSld><p:spTree><p:sp><p:txBody>{paras}</p:txBody></p:sp>"
        f"{table}</p:spTree></p:cSld></p:sld>"
    )


def _notes(text: str) -> str:
    return (
        f'<?xml version="1.0"?><p:notes xmlns:p="{P}" xmlns:a="{A}"><p:cSld>'
        '<p:spTree><p:sp><p:nvSpPr><p:nvPr><p:ph type="sldNum"/></p:nvPr>'
        "</p:nvSpPr><p:txBody><a:p><a:r><a:t>7</a:t></a:r></a:p></p:txBody></p:sp>"
        '<p:sp><p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr>'
        f"<p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>"
        "</p:spTree></p:cSld></p:notes>"
    )


def _pptx(path: Path, slides: list, order: list, extra: dict | None = None) -> Path:
    """``slides``: (part file name, xml); ``order``: file names in deck order."""
    ids = "".join(
        f'<p:sldId id="{256 + i}" r:id="rS{name}"/>' for i, name in enumerate(order)
    )
    parts = {
        "_rels/.rels": _rels(("rId1", "officeDocument", "ppt/presentation.xml")),
        "ppt/presentation.xml": (
            f'<?xml version="1.0"?><p:presentation xmlns:p="{P}" xmlns:r="{REL}">'
            f"<p:sldIdLst>{ids}</p:sldIdLst></p:presentation>"
        ),
        "ppt/_rels/presentation.xml.rels": _rels(
            *[(f"rS{name}", "slide", f"slides/{name}") for name, _ in slides]
        ),
    }
    for name, xml in slides:
        parts[f"ppt/slides/{name}"] = xml
    parts.update(extra or {})
    return _write(path, parts)


def _chunks(path: Path, **kwargs) -> list:
    return list(OfficeReader(input_path=path, **kwargs).get_raw_chunks())


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix",
    [".docx", ".docm", ".dotx", ".dotm", ".pptx", ".pptm", ".potx", ".potm", ".ppsx"],
)
def test_every_extension_dispatches_to_the_office_reader(tmp_path, suffix):
    path = _docx(tmp_path / f"a{suffix}", _p("x"))
    assert isinstance(DocumentReader.create(path), OfficeReader)


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------


class TestWord:
    def test_paragraphs_and_tables_in_document_order(self, tmp_path):
        table = (
            "<w:tbl><w:tr><w:tc>" + _p("a") + "</w:tc><w:tc>" + _p("b") + "</w:tc>"
            "</w:tr><w:tr><w:tc>" + _p("") + "</w:tc><w:tc>" + _p("") + "</w:tc>"
            "</w:tr><w:tr><w:tc>" + _p("c") + "</w:tc><w:tc>" + _p("d") + "</w:tc>"
            "</w:tr></w:tbl>"
        )
        path = _docx(tmp_path / "d.docx", _p("one") + _p("two") + table + _p("three"))
        got = [(c["section_type"], c["text"]) for c in _chunks(path)]
        assert got == [
            ("text", "one\ntwo"),
            ("table", "a\tb\nc\td"),
            ("text", "three"),
        ]

    def test_runs_tabs_breaks_and_deletions(self, tmp_path):
        body = (
            "<w:p><w:r><w:t>al</w:t></w:r><w:r><w:t>pha</w:t><w:tab/>"
            "<w:t>b</w:t><w:br/><w:t>c</w:t><w:noBreakHyphen/><w:t>d</w:t></w:r>"
            "<w:del><w:r><w:t>gone</w:t></w:r></w:del>"
            "<w:ins><w:r><w:t>new</w:t></w:r></w:ins></w:p>"
        )
        (chunk,) = _chunks(_docx(tmp_path / "d.docx", body))
        assert chunk["text"] == "alpha\tb\nc-dnew"
        assert chunk["raw_text"] == chunk["text"]

    def test_footnotes_and_endnotes_unless_turned_off(self, tmp_path):
        notes = (
            f'<?xml version="1.0"?><w:footnotes xmlns:w="{W}">'
            '<w:footnote w:type="separator" w:id="-1">' + _p("---") + "</w:footnote>"
            '<w:footnote w:id="1">' + _p("a note") + "</w:footnote></w:footnotes>"
        )
        path = _docx(
            tmp_path / "d.docx",
            _p("body"),
            extra={"word/footnotes.xml": notes},
            doc_rels=[("rF", "footnotes", "footnotes.xml")],
        )
        got = [(c["section_type"], c["text"]) for c in _chunks(path)]
        assert got == [("text", "body"), ("footnote", "a note")]
        assert [c["text"] for c in _chunks(path, notes=False)] == ["body"]

    def test_macros_are_flagged_never_run(self, tmp_path):
        path = _docx(tmp_path / "d.docm", _p("x"), extra={"word/vbaProject.bin": b"\0"})
        assert [c["has_macros"] for c in _chunks(path)] == [True]
        plain = _docx(tmp_path / "e.docx", _p("x"))
        assert [c["has_macros"] for c in _chunks(plain)] == [False]

    def test_an_empty_document_gives_no_chunk(self, tmp_path):
        assert _chunks(_docx(tmp_path / "d.docx", _p("  "))) == []


# ---------------------------------------------------------------------------
# PowerPoint
# ---------------------------------------------------------------------------


class TestPowerPoint:
    def test_slides_follow_the_deck_not_the_part_names(self, tmp_path):
        slides = [(f"slide{i}.xml", _slide(f"s{i}")) for i in (1, 2, 11)]
        path = _pptx(
            tmp_path / "d.pptx", slides, ["slide2.xml", "slide11.xml", "slide1.xml"]
        )
        got = [(c["page_number"], c["text"]) for c in _chunks(path)]
        assert got == [(0, "s2"), (1, "s11"), (2, "s1")]

    def test_notes_body_only_and_switchable(self, tmp_path):
        slides = [("slide1.xml", _slide("title"))]
        extra = {
            "ppt/slides/_rels/slide1.xml.rels": _rels(
                ("rN", "notesSlide", "../notesSlides/notesSlide1.xml")
            ),
            "ppt/notesSlides/notesSlide1.xml": _notes("say this"),
        }
        path = _pptx(tmp_path / "d.pptx", slides, ["slide1.xml"], extra)
        got = [(c["section_type"], c["page_number"], c["text"]) for c in _chunks(path)]
        assert got == [("text", 0, "title"), ("footnote", 0, "say this")]
        assert [c["text"] for c in _chunks(path, notes=False)] == ["title"]

    def test_hidden_slides_are_marked_or_left_out(self, tmp_path):
        slides = [("slide1.xml", _slide("a")), ("slide2.xml", _slide("b", show="0"))]
        path = _pptx(tmp_path / "d.pptx", slides, ["slide1.xml", "slide2.xml"])
        assert [(c["text"], c["hidden"]) for c in _chunks(path)] == [
            ("a", False),
            ("b", True),
        ]
        got = [(c["page_number"], c["text"]) for c in _chunks(path, hidden=False)]
        assert got == [(0, "a")]

    def test_a_table_reads_as_rows_once(self, tmp_path):
        table = (
            "<p:graphicFrame><a:graphic><a:graphicData><a:tbl>"
            "<a:tr><a:tc><a:txBody><a:p><a:r><a:t>h1</a:t></a:r></a:p></a:txBody>"
            "</a:tc><a:tc><a:txBody><a:p><a:r><a:t>h2</a:t></a:r></a:p></a:txBody>"
            "</a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>"
        )
        slides = [("slide1.xml", _slide("title", table=table))]
        (chunk,) = _chunks(_pptx(tmp_path / "d.pptx", slides, ["slide1.xml"]))
        assert chunk["text"] == "title\nh1\th2"

    def test_a_slide_without_text_gives_no_chunk_but_keeps_numbering(self, tmp_path):
        slides = [("slide1.xml", _slide()), ("slide2.xml", _slide("b"))]
        path = _pptx(tmp_path / "d.pptx", slides, ["slide1.xml", "slide2.xml"])
        assert [(c["page_number"], c["text"]) for c in _chunks(path)] == [(1, "b")]


# ---------------------------------------------------------------------------
# Hostile and broken input
# ---------------------------------------------------------------------------


class TestUntrustedInput:
    def test_entity_declarations_are_refused(self, tmp_path):
        evil = (
            '<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
            f'<w:document xmlns:w="{W}"><w:body>{_p("&x;")}</w:body></w:document>'
        )
        path = _write(
            tmp_path / "d.docx",
            {
                "_rels/.rels": _rels(("rId1", "officeDocument", "word/document.xml")),
                "word/document.xml": evil,
            },
        )
        with pytest.raises(ValueError, match="entit"):
            _chunks(path)

    def test_a_relationship_outside_the_package_is_not_followed(self, tmp_path):
        slides = [("slide1.xml", _slide("a"))]
        extra = {
            "ppt/slides/_rels/slide1.xml.rels": _rels(
                ("rN", "notesSlide", "../../../outside.xml")
            ),
            # Present in the ZIP under the name the target resolves to.
            "../outside.xml": _notes("leaked"),
        }
        path = _pptx(tmp_path / "d.pptx", slides, ["slide1.xml"], extra)
        assert [c["text"] for c in _chunks(path)] == ["a"]

    def test_an_external_relationship_is_not_followed(self, tmp_path):
        slides = [("slide1.xml", _slide("a"))]
        extra = {
            "ppt/slides/_rels/slide1.xml.rels": _rels(
                ("rN", "notesSlide", "http://example.com/x.xml", True)
            ),
            # Present under the name the target would resolve to if followed.
            "ppt/slides/http:/example.com/x.xml": _notes("fetched"),
        }
        path = _pptx(tmp_path / "d.pptx", slides, ["slide1.xml"], extra)
        assert [c["text"] for c in _chunks(path)] == ["a"]

    def test_a_part_over_the_limit_is_refused(self, tmp_path):
        path = _docx(tmp_path / "d.docx", _p("x" * 5000))
        with pytest.raises(ValueError, match="max_part_bytes"):
            _chunks(path, max_part_bytes=1000)

    def test_too_many_parts_is_refused(self, tmp_path):
        extra = {f"junk/{i}.xml": "<a/>" for i in range(10)}
        path = _docx(tmp_path / "d.docx", _p("x"), extra=extra)
        with pytest.raises(ValueError, match="max_parts"):
            _chunks(path, max_parts=5)

    def test_a_legacy_or_encrypted_file_is_named(self, tmp_path):
        path = tmp_path / "d.docx"
        path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 100)
        with pytest.raises(ValueError, match="OLE2"):
            _chunks(path)

    def test_not_a_zip_is_a_value_error(self, tmp_path):
        path = tmp_path / "d.pptx"
        path.write_bytes(b"plain text")
        with pytest.raises(ValueError, match="not a ZIP"):
            _chunks(path)

    def test_a_package_without_a_main_part(self, tmp_path):
        path = _write(tmp_path / "d.docx", {"[Content_Types].xml": "<a/>"})
        with pytest.raises(ValueError, match="no main document part"):
            _chunks(path)

    def test_a_main_part_of_another_kind(self, tmp_path):
        path = _write(
            tmp_path / "d.docx",
            {
                "_rels/.rels": _rels(("rId1", "officeDocument", "x/other.xml")),
                "x/other.xml": '<?xml version="1.0"?><other/>',
            },
        )
        with pytest.raises(
            ValueError, match="not a Word document, a presentation or a workbook"
        ):
            _chunks(path)

    def test_malformed_xml_names_the_part(self, tmp_path):
        path = _write(
            tmp_path / "d.docx",
            {
                "_rels/.rels": _rels(("rId1", "officeDocument", "word/document.xml")),
                "word/document.xml": "<w:document",
            },
        )
        with pytest.raises(ValueError, match=r"word/document\.xml"):
            _chunks(path)

    def test_a_file_over_max_file_bytes(self, tmp_path):
        path = _docx(tmp_path / "d.docx", _p("x"))
        with pytest.raises(ValueError, match="max_file_bytes"):
            _chunks(path, max_file_bytes=10)


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("notes", "yes", TypeError),
        ("hidden", 1, TypeError),
        ("rows_per_chunk", 0, ValueError),
        ("max_part_bytes", 0, ValueError),
        ("max_parts", True, ValueError),
        ("max_file_bytes", -1, ValueError),
    ],
)
def test_bad_options(tmp_path, option, value, error):
    path = _docx(tmp_path / "d.docx", _p("x"))
    with pytest.raises(error):
        OfficeReader(input_path=path, **{option: value})


def test_two_reads_give_the_same_chunks(tmp_path):
    slides = [(f"slide{i}.xml", _slide(f"s{i}")) for i in range(1, 6)]
    path = _pptx(tmp_path / "d.pptx", slides, [n for n, _ in slides])
    assert _chunks(path) == _chunks(path)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _sheet(rows: str) -> str:
    return f'<?xml version="1.0"?><worksheet xmlns="{S}"><sheetData>{rows}</sheetData></worksheet>'


def _xlsx(path: Path, sheets: list, shared: list | None = None, extra=None) -> Path:
    """``sheets``: (name, xml or None for a chartsheet, state or None)."""
    entries = "".join(
        f'<sheet name="{name}" sheetId="{i + 1}" r:id="rW{i}"'
        + (f' state="{state}"' if state else "")
        + "/>"
        for i, (name, _, state) in enumerate(sheets)
    )
    rels = [
        (
            f"rW{i}",
            "worksheet" if xml is not None else "chartsheet",
            f"worksheets/s{i}.xml",
        )
        for i, (_, xml, _) in enumerate(sheets)
    ]
    parts = {
        "_rels/.rels": _rels(("rId1", "officeDocument", "xl/workbook.xml")),
        "xl/workbook.xml": (
            f'<?xml version="1.0"?><workbook xmlns="{S}" xmlns:r="{REL}">'
            f"<sheets>{entries}</sheets></workbook>"
        ),
    }
    if shared is not None:
        rels.append(("rSS", "sharedStrings", "sharedStrings.xml"))
        parts["xl/sharedStrings.xml"] = (
            f'<?xml version="1.0"?><sst xmlns="{S}">'
            + "".join(f"<si>{item}</si>" for item in shared)
            + "</sst>"
        )
    parts["xl/_rels/workbook.xml.rels"] = _rels(*rels)
    for i, (_, xml, _) in enumerate(sheets):
        if xml is not None:
            parts[f"xl/worksheets/s{i}.xml"] = xml
    parts.update(extra or {})
    return _write(path, parts)


class TestExcel:
    def test_cell_kinds_read_as_stored(self, tmp_path):
        rows = (
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2"><v>3.5</v></c><c r="B2" t="b"><v>1</v></c>'
            '<c r="C2" t="str"><f>A2*2</f><v>7</v></c>'
            '<c r="D2" t="inlineStr"><is><t>in</t></is></c>'
            '<c r="E2" t="e"><v>#DIV/0!</v></c><c r="F2"><f>NOW()</f></c></row>'
        )
        shared = ["<t>name</t>", "<r><t>ri</t></r><r><t>ch</t></r><rPh><t>x</t></rPh>"]
        path = _xlsx(tmp_path / "b.xlsx", [("Data", _sheet(rows), None)], shared)
        (chunk,) = _chunks(path)
        assert chunk["text"] == "name\trich\n3.5\tTRUE\t7\tin\t#DIV/0!"
        assert (chunk["section_type"], chunk["sheet"], chunk["page_number"]) == (
            "table",
            "Data",
            0,
        )
        assert (chunk["row_start"], chunk["row_end"]) == (1, 2)

    def test_a_gap_keeps_the_column_and_empty_rows_are_dropped(self, tmp_path):
        rows = (
            '<row r="1"><c r="A1"><v>1</v></c><c r="C1"><v>3</v></c></row>'
            '<row r="2"><c r="B2" t="inlineStr"><is><t> </t></is></c></row>'
            '<row r="5"><c r="B5"><v>2</v></c></row>'
        )
        (chunk,) = _chunks(_xlsx(tmp_path / "b.xlsx", [("S", _sheet(rows), None)]))
        assert chunk["text"] == "1\t\t3\n\t2"
        assert (chunk["row_start"], chunk["row_end"]) == (1, 5)

    def test_sheets_in_workbook_order_chartsheets_skipped(self, tmp_path):
        one = _sheet('<row r="1"><c r="A1"><v>1</v></c></row>')
        two = _sheet('<row r="1"><c r="A1"><v>2</v></c></row>')
        path = _xlsx(
            tmp_path / "b.xlsx",
            [("Z", one, None), ("Chart", None, None), ("A", two, None)],
        )
        assert [(c["page_number"], c["sheet"], c["text"]) for c in _chunks(path)] == [
            (0, "Z", "1"),
            (2, "A", "2"),
        ]

    def test_rows_per_chunk_splits_a_long_sheet(self, tmp_path):
        rows = "".join(
            f'<row r="{i}"><c r="A{i}"><v>{i}</v></c></row>' for i in range(1, 6)
        )
        path = _xlsx(tmp_path / "b.xlsx", [("S", _sheet(rows), None)])
        got = [
            (c["row_start"], c["row_end"], c["text"])
            for c in _chunks(path, rows_per_chunk=2)
        ]
        assert got == [(1, 2, "1\n2"), (3, 4, "3\n4"), (5, 5, "5")]

    @pytest.mark.parametrize("state", ["hidden", "veryHidden"])
    def test_hidden_sheets_are_marked_or_left_out(self, tmp_path, state):
        cell = _sheet('<row r="1"><c r="A1"><v>1</v></c></row>')
        path = _xlsx(tmp_path / "b.xlsx", [("V", cell, None), ("H", cell, state)])
        assert [(c["sheet"], c["hidden"]) for c in _chunks(path)] == [
            ("V", False),
            ("H", True),
        ]
        assert [c["sheet"] for c in _chunks(path, hidden=False)] == ["V"]

    def test_cells_without_references_follow_one_another(self, tmp_path):
        rows = '<row><c t="inlineStr"><is><t>a</t></is></c><c><v>2</v></c></row>'
        (chunk,) = _chunks(_xlsx(tmp_path / "b.xlsx", [("S", _sheet(rows), None)]))
        assert (chunk["text"], chunk["row_start"]) == ("a\t2", 1)

    def test_a_shared_string_index_out_of_range_reads_empty(self, tmp_path):
        rows = '<row r="1"><c r="A1" t="s"><v>9</v></c><c r="B1"><v>1</v></c></row>'
        path = _xlsx(tmp_path / "b.xlsx", [("S", _sheet(rows), None)], ["<t>a</t>"])
        assert [c["text"] for c in _chunks(path)] == ["\t1"]

    def test_an_empty_workbook_gives_no_chunk(self, tmp_path):
        path = _xlsx(tmp_path / "b.xlsx", [("S", _sheet(""), None)])
        assert _chunks(path) == []

    @pytest.mark.parametrize("suffix", [".xlsx", ".xlsm", ".xltx", ".xltm"])
    def test_every_excel_extension_dispatches(self, tmp_path, suffix):
        path = _xlsx(tmp_path / f"b{suffix}", [("S", _sheet(""), None)])
        assert isinstance(DocumentReader.create(path), OfficeReader)
