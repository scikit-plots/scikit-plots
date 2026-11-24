"""Readers say what they covered: read, empty and skipped parts (CX-49)."""

from __future__ import annotations

import zipfile
from unittest.mock import patch

import pytest

from ... import _corpus_builder
from ..._base import DocumentReader, ReadCoverage
from .. import _pdf, _zip


def _warnings(module, call):
    with patch.object(module.logger, "warning") as warning:
        result = call()
    return result, [c.args[0] % c.args[1:] for c in warning.call_args_list]


# -- the record -------------------------------------------------------------


def test_record_counts_and_summary():
    cov = ReadCoverage(unit="member", read=2)
    cov.skip("unsupported", 2)
    cov.skip("hidden")
    other = ReadCoverage(unit="member", empty=1)
    other.skip("unsupported")
    cov.merge(other)
    assert cov.total == 7
    assert cov.summary() == "2 read, 1 empty, 4 skipped (1 hidden, 3 unsupported) of 7 member(s)"


@pytest.mark.parametrize(
    ("read", "empty", "skipped", "share", "poor"),
    [
        (0, 0, 0, 0.5, False),  # nothing seen: nothing to say
        (0, 1, 0, 0.5, True),  # seen, nothing read
        (1, 1, 0, 0.5, False),  # half: not above the share
        (1, 1, 1, 0.5, True),
        (1, 3, 0, 1.0, False),  # 1 warns only when nothing was read
    ],
)
def test_is_poor(read, empty, skipped, share, poor):
    cov = ReadCoverage(read=read, empty=empty)
    if skipped:
        cov.skip("unsupported", skipped)
    assert cov.is_poor(share) is poor


# -- archives ---------------------------------------------------------------


@pytest.fixture
def unsupported_zip(tmp_path):
    path = tmp_path / "only_unsupported.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("page.html", "<p>x</p>")
        zf.writestr("data.csv", "a,b")
        zf.writestr("script.py", "x = 1")
    return path


def test_an_archive_of_unsupported_members_warns_once_with_counts(unsupported_zip):
    reader = DocumentReader.create(unsupported_zip)
    chunks, warnings = _warnings(_zip, lambda: list(reader.get_raw_chunks()))
    assert chunks == []
    assert len(warnings) == 1 and "3 unsupported" in warnings[0]
    assert reader.coverage.skipped == {"unsupported": 3}


def test_nested_archives_roll_up_into_one_warning(unsupported_zip, tmp_path):
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("a.txt", "A sentence that is long enough to keep.")
        zf.writestr(".hidden", "x")
        zf.write(unsupported_zip, "inner.zip")
    reader = DocumentReader.create(outer)
    _, warnings = _warnings(_zip, lambda: list(reader.get_raw_chunks()))
    assert len(warnings) == 1
    assert reader.coverage.read == 1
    assert reader.coverage.skipped == {"hidden": 1, "unsupported": 3}
    first = reader.coverage
    list(reader.get_raw_chunks())  # a second read: a new, equal record
    assert reader.coverage == first and reader.coverage is not first


def test_a_mostly_read_archive_is_quiet(tmp_path):
    path = tmp_path / "fine.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(3):
            zf.writestr(f"{i}.txt", f"Sentence number {i} is long enough to keep.")
        zf.writestr("one.py", "x = 1")
    reader = DocumentReader.create(path)
    _, warnings = _warnings(_zip, lambda: list(reader.get_raw_chunks()))
    assert warnings == [] and reader.coverage.read == 3


def test_the_share_is_validated(unsupported_zip):
    with pytest.raises(ValueError, match="coverage_warn_share"):
        DocumentReader.create(unsupported_zip, coverage_warn_share=0)


# -- PDFs -------------------------------------------------------------------


@pytest.fixture
def scanned_pdf(tmp_path):
    """One page of text, seven pages that carry only a page number."""
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    pytest.importorskip("pdfminer.high_level")
    path = tmp_path / "scan.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, "A title page with a real sentence of text on it.")
    c.showPage()
    for number in range(2, 9):
        c.drawString(300, 40, str(number))
        c.showPage()
    c.save()
    return path


def test_a_pdf_of_page_numbers_warns(scanned_pdf):
    reader = _pdf.PDFReader(input_path=scanned_pdf)
    chunks, warnings = _warnings(_pdf, lambda: list(reader.get_raw_chunks()))
    assert len(chunks) == 8  # every page is still yielded
    assert (reader.coverage.read, reader.coverage.empty) == (1, 7)
    assert len(warnings) == 1 and "ocr='auto'" in warnings[0]


def test_the_page_threshold_is_configurable(scanned_pdf):
    reader = _pdf.PDFReader(input_path=scanned_pdf, coverage_min_chars=1)
    _, warnings = _warnings(_pdf, lambda: list(reader.get_raw_chunks()))
    assert warnings == [] and reader.coverage.read == 8


# -- folders ----------------------------------------------------------------


def test_a_folder_of_unsupported_files_warns_once(tmp_path):
    for name in ("a.unknownext", "b.unknownext", ".hidden.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    sources, warnings = _warnings(
        _corpus_builder, lambda: _corpus_builder.CorpusBuilder._expand_sources([tmp_path])
    )
    assert sources == []
    assert len(warnings) == 1 and "2 unsupported" in warnings[0] and "1 hidden" in warnings[0]
