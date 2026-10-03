"""
Tests for :mod:`scikitplot.cleanprompt._office`.

Notes
-----
**Developer notes.** The packages are built here with :mod:`zipfile` and a few
lines of XML, so the suite needs neither python-docx nor openpyxl and every
byte the reader sees is visible in the test.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from .._exceptions import CleanPromptError
from .._office import OFFICE_EXTENSIONS, OfficeLimits, extract_office_text

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'


def _package(parts):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in parts.items():
            archive.writestr(name, body)
    return buffer.getvalue()


def _docx(body):
    return _package(
        {"word/document.xml": f"<w:document {_W}><w:body>{body}</w:body></w:document>"}
    )


def _p(text):
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


class TestDocx:
    def test_paragraphs_in_order(self):
        assert (
            extract_office_text(_docx(_p("Hello") + _p("MRN: 00412345")), "docx")
            == "Hello\nMRN: 00412345\n"
        )

    def test_table_row_cells_are_joined_by_tabs(self):
        row = "<w:tbl><w:tr><w:tc>{0}</w:tc><w:tc>{1}</w:tc></w:tr></w:tbl>".format(
            _p("DOB"), _p("1984-03-02")
        )
        assert extract_office_text(_docx(row), "docx") == "DOB\t1984-03-02\n"

    def test_headers_and_comments_are_read(self):
        data = _package(
            {
                "word/document.xml": f"<w:document {_W}><w:body>{_p('body')}</w:body></w:document>",
                "word/header1.xml": f"<w:hdr {_W}>{_p('Confidential')}</w:hdr>",
                "word/comments.xml": f"<w:comments {_W}><w:comment>{_p('call Ann')}</w:comment></w:comments>",
            }
        )
        text = extract_office_text(data, "docx")
        assert "Confidential" in text and "call Ann" in text and "body" in text

    def test_a_dtd_is_refused(self):
        hostile = (
            '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e "boom">]>'
            + f"<w:document {_W}/>"
        )
        with pytest.raises(CleanPromptError, match="DTD"):
            extract_office_text(_package({"word/document.xml": hostile}), "docx")

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-16-be"])
    def test_a_dtd_is_refused_in_utf16(self, encoding):
        """
        ECMA-376 allows UTF-16 parts, where every markup character has a NUL
        beside it. The refusal read raw bytes, found no ``<!DOCTYPE`` and let
        the parser expand the entity.
        """
        hostile = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE x [<!ENTITY e "boom">]>' + f"<w:document {_W}/>"
        ).encode(encoding)
        with pytest.raises(CleanPromptError, match="DTD"):
            extract_office_text(_package({"word/document.xml": hostile}), "docx")

    def test_a_utf16_part_without_a_dtd_is_still_read(self):
        """The refusal is about the DTD, not about the encoding."""
        body = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            f"<w:document {_W}><w:body>{_p('Hello')}</w:body></w:document>"
        ).encode("utf-16")
        assert extract_office_text(_package({"word/document.xml": body}), "docx") == "Hello\n"


class TestXlsx:
    def test_cells_align_by_column_letter_and_sheets_are_headed(self):
        shared = f"<sst {_S}><si><t>name</t></si><si><t>mrn</t></si><si><t>Ann</t></si></sst>"
        sheet = (
            f"<worksheet {_S}><sheetData>"
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>7</v></c></row>'
            "</sheetData></worksheet>"
        )
        data = _package(
            {
                "xl/sharedStrings.xml": shared,
                "xl/worksheets/sheet10.xml": sheet,
                "xl/worksheets/sheet2.xml": sheet,
            }
        )
        text = extract_office_text(data, "xlsx")
        assert text.startswith("## sheet 2\nname\tmrn\nAnn\t\t7\n")
        assert text.index("## sheet 2") < text.index("## sheet 10")


class TestPptx:
    def test_slides_are_read(self):
        slide = f"<p:sld xmlns:p='x' {_A}><a:p><a:r><a:t>Q3 for Ann</a:t></a:r></a:p></p:sld>"
        text = extract_office_text(_package({"ppt/slides/slide1.xml": slide}), "pptx")
        assert "Q3 for Ann" in text


class TestLimits:
    def test_limits_must_be_positive_integers(self):
        with pytest.raises(CleanPromptError):
            OfficeLimits(max_members=0)
        with pytest.raises(CleanPromptError):
            OfficeLimits(max_part_bytes=True)

    def test_an_oversized_part_is_refused_unread(self):
        data = _docx(_p("x" * 5000))
        with pytest.raises(CleanPromptError, match="limit"):
            extract_office_text(data, "docx", OfficeLimits(max_part_bytes=100))

    def test_too_many_members_is_refused(self):
        data = _package({f"word/x{i}.xml": "<a/>" for i in range(5)})
        with pytest.raises(CleanPromptError):
            extract_office_text(data, "docx", OfficeLimits(max_members=3))

    def test_not_a_zip_is_refused(self):
        with pytest.raises(CleanPromptError):
            extract_office_text(b"not a zip", "docx")


def test_extension_table_names_three_kinds():
    assert set(OFFICE_EXTENSIONS.values()) == {"docx", "xlsx", "pptx"}
