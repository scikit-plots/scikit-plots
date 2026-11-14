"""
The per-input text budget (``max_text_chars`` / ``on_text_limit``).

One bound on what a small input can expand into, enforced at the single
place every reader's chunks pass (``DocumentReader._iter_raw_chunks``),
custom extractors and archive members included (CX-35, CX-36).
"""

from __future__ import annotations

import logging
import zipfile
import zlib
from pathlib import Path

import pytest

from scikitplot.corpus._base import DocumentReader
from scikitplot.corpus._readers._office import OfficeReader
from scikitplot.corpus._readers._pdf import PDFReader
from scikitplot.corpus._readers._zip import ZipReader
from scikitplot.corpus._readers.tests.test__office import _cell, _odf, _row

_DEFAULT = 256 * 1024 * 1024


def _reader(tmp_path: Path, texts: list[str], **kwargs) -> DocumentReader:
    """A text-file reader whose chunks come from a custom extractor."""
    path = tmp_path / "in.txt"
    path.write_text("unused", encoding="utf-8")
    return DocumentReader.create(path, custom_extractor=lambda _p: list(texts), **kwargs)


def _texts(reader: DocumentReader) -> list[str]:
    return [chunk["text"] for chunk in reader._iter_raw_chunks()]


def _shared_stream_pdf(path: Path, chars: int, pages: int) -> Path:
    """One Flat stream of *chars* shown characters, shared by *pages* pages."""
    line = "BT /F1 1 Tf 0 0 Td (" + "x" * 100 + ") Tj ET\n"
    stream = zlib.compress((line * (chars // 100)).encode("ascii"), 9)
    kids = " ".join(f"{5 + i} 0 R" for i in range(pages))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
        + stream
        + b"\nendstream",
    ]
    objects += [
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 3 0 R >> >> >>"
    ] * pages
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(bytes(out))
    return path


class TestBase:
    def test_no_limit_by_default_for_formats_bounded_by_their_input(self, tmp_path):
        reader = _reader(tmp_path, ["a" * 1000] * 5)
        assert reader.max_text_chars is None
        assert _texts(reader) == ["a" * 1000] * 5

    def test_the_amplifying_formats_default_to_256_mi(self, tmp_path):
        assert OfficeReader(input_path=_odf(tmp_path / "b.ods", "spreadsheet", "")).max_text_chars == _DEFAULT
        pdf = _shared_stream_pdf(tmp_path / "b.pdf", 100, 1)
        assert PDFReader(input_path=pdf).max_text_chars == _DEFAULT

    def test_a_custom_extractor_is_bounded_too(self, tmp_path):
        reader = _reader(tmp_path, ["abc", "def"], max_text_chars=5)
        with pytest.raises(ValueError, match="max_text_chars=5"):
            _texts(reader)

    def test_exactly_the_limit_passes(self, tmp_path):
        assert _texts(_reader(tmp_path, ["abc", "def"], max_text_chars=6)) == ["abc", "def"]

    def test_truncate_keeps_exactly_the_limit(self, tmp_path, caplog, captured_logger):
        reader = _reader(
            tmp_path, ["abc", "def", "ghi"], max_text_chars=5, on_text_limit="truncate"
        )
        with captured_logger(logging.WARNING, "scikitplot.corpus._base"):
            chunks = list(reader._iter_raw_chunks())
        assert [c["text"] for c in chunks] == ["abc", "de"]
        assert [c.get("text_truncated", False) for c in chunks] == [False, True]
        (record,) = caplog.records
        assert "truncated" in record.getMessage()

    def test_the_budget_is_idempotent(self, tmp_path):
        reader = _reader(tmp_path, [], max_text_chars=5, on_text_limit="truncate")
        once = list(reader._within_text_limit(iter([{"text": "abcdefgh"}])))
        twice = list(reader._within_text_limit(iter(once)))
        assert [c["text"] for c in twice] == [c["text"] for c in once] == ["abcde"]

    def test_get_documents_marks_the_cut_document(self, tmp_path):
        one = "The first paragraph has enough plain words to pass the filter. " * 2
        two = "A second paragraph also has enough plain words to be kept here. " * 2
        reader = _reader(
            tmp_path, [one, two], max_text_chars=len(one) + 64, on_text_limit="truncate"
        )
        docs = list(reader.get_documents())
        assert [d.text for d in docs] == [one, two[:64]]
        assert "text_truncated" not in docs[0].metadata
        assert docs[-1].metadata.get("text_truncated") is True

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("max_text_chars", 0),
            ("max_text_chars", -1),
            ("max_text_chars", True),
            ("max_text_chars", 2.5),
            ("on_text_limit", "skip"),
        ],
    )
    def test_bad_options_are_refused(self, tmp_path, option, value):
        with pytest.raises(ValueError, match=option):
            _reader(tmp_path, ["x"], **{option: value})


class TestPDF:
    def test_a_shared_compressed_stream_is_bounded(self, tmp_path):
        """A few KB, 20 pages sharing one stream of 20,000 characters each."""
        pytest.importorskip("pypdf")
        path = _shared_stream_pdf(tmp_path / "b.pdf", 20_000, 20)
        assert path.stat().st_size < 5_000
        reader = PDFReader(input_path=path, prefer_backend="pypdf", max_text_chars=50_000)
        with pytest.raises(ValueError, match="b.pdf yields more than max_text_chars=50,000"):
            _texts(reader)


class TestZip:
    def _archive(self, tmp_path: Path) -> Path:
        hostile = '<table:table table:name="T">' + _row(_cell("x", 100), 1_000) + "</table:table>"
        ods = _odf(tmp_path / "hostile.ods", "spreadsheet", hostile)
        archive = tmp_path / "a.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(ods, "hostile.ods")
            zf.writestr("note.txt", "a plain note")
        return archive

    def test_a_member_is_bounded_by_its_own_reader(self, tmp_path):
        """A refused member is skipped like any member error; the rest is read."""
        archive = self._archive(tmp_path)
        reader = ZipReader(
            input_path=archive, reader_kwargs={".ods": {"max_text_chars": 1_000}}
        )
        assert _texts(reader) == ["a plain note"]

    def test_the_archive_limit_bounds_the_total(self, tmp_path):
        reader = ZipReader(input_path=self._archive(tmp_path), max_text_chars=1_000)
        with pytest.raises(ValueError, match="a.zip yields more than max_text_chars=1,000"):
            _texts(reader)
