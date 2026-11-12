# scikitplot/corpus/_readers/tests/test__pdf.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for PDFReader's built-in backends.

Most tests replace pdfminer.six and pypdf with small stand-ins, so they run
without either library and can count how often a file is opened. The
``TestRealLibraries`` class reads a hand-written PDF with the real ones and
is skipped where they are not installed.
"""

from __future__ import annotations

import pathlib
import sys
import types
from typing import Any, Sequence

import pytest

from .. import _pdf
from .._pdf import PDFReader

#: What the encrypted test files open with, and something that is not it.
_KEY = "open sesame"
_OTHER_KEY = "open barley"

#: Number of times a file is opened when one page fails or is read out of order.
_TWICE = 2

# ===========================================================================
# Stand-ins for the two libraries
# ===========================================================================


class _Box:
    """A text container as pdfminer's layout yields it."""

    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


class _Figure:
    """A layout element that is not text."""


class _Params:
    """Stand-in for ``pdfminer.layout.LAParams``: it knows two arguments."""

    def __init__(self, boxes_flow: Any = 0.5, line_margin: float = 0.5) -> None:
        self.given = {"boxes_flow": boxes_flow, "line_margin": line_margin}


class _FakePdfminer:
    """
    Stand-in for ``pdfminer.high_level`` and ``pdfminer.layout``.

    Parameters
    ----------
    pages : list
        One entry per page: a list of layout elements, or an exception to
        raise when that page is reached.
    """

    def __init__(self, pages: list) -> None:
        self.pages = pages
        self.opens: list[dict[str, Any]] = []
        self.open_now = 0
        self.laid_out: list[int] = []
        # Keeps every iterator alive, so only close() can end one early.
        self._iterators: list[Any] = []

    def extract_pages(
        self,
        pdf_file: str,
        password: str = "",
        page_numbers: Any = None,
        laparams: Any = None,
        caching: bool = True,
    ) -> Any:
        self.opens.append(
            {
                "file": pdf_file,
                "password": password,
                "page_numbers": page_numbers,
                "laparams": laparams,
                "caching": caching,
            }
        )
        iterator = self._walk(page_numbers)
        self._iterators.append(iterator)
        return iterator

    def _walk(self, page_numbers: Any) -> Any:
        self.open_now += 1
        try:
            for index, page in enumerate(self.pages):
                if page_numbers and index not in page_numbers:
                    continue
                self.laid_out.append(index)
                if isinstance(page, Exception):
                    raise page
                yield page
        finally:
            self.open_now -= 1

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        high_level = types.ModuleType("pdfminer.high_level")
        high_level.extract_pages = self.extract_pages
        layout = types.ModuleType("pdfminer.layout")
        layout.LTTextContainer = _Box
        layout.LAParams = _Params
        package = types.ModuleType("pdfminer")
        monkeypatch.setitem(sys.modules, "pdfminer", package)
        monkeypatch.setitem(sys.modules, "pdfminer.high_level", high_level)
        monkeypatch.setitem(sys.modules, "pdfminer.layout", layout)


class _FakePypdf:
    """
    Stand-in for ``pypdf``.

    Parameters
    ----------
    pages : list
        One entry per page: its text, ``None`` for no text, or an exception
        to raise when the page's text is asked for.
    password : str or None
        The password that decrypts the file; ``None`` for a plain file.
    open_error : Exception or None
        Raised when the file is opened.
    """

    def __init__(
        self,
        pages: list,
        password: str | None = None,
        open_error: Exception | None = None,
    ) -> None:
        self.pages = pages
        self.password = password
        self.open_error = open_error
        self.opens = 0
        self.decrypts: list[str] = []
        self.asked: list[int] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = self

        class _Page:
            def __init__(self, index: int) -> None:
                self._index = index

            def extract_text(self) -> Any:
                fake.asked.append(self._index)
                value = fake.pages[self._index]
                if isinstance(value, Exception):
                    raise value
                return value

        class PdfReader:
            def __init__(self, stream: str) -> None:
                fake.opens += 1
                if fake.open_error is not None:
                    raise fake.open_error
                self.is_encrypted = fake.password is not None
                self.pages = [_Page(i) for i in range(len(fake.pages))]

            def decrypt(self, password: str) -> int:
                fake.decrypts.append(password)
                return 1 if password == fake.password else 0

        module = types.ModuleType("pypdf")
        module.PdfReader = PdfReader
        monkeypatch.setitem(sys.modules, "pypdf", module)


def _missing(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make importing each of ``names`` raise ImportError."""
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def pdf_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """A file that exists; the stand-ins never read it."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    return path


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """The module's warning and debug messages, formatted, by level."""
    seen: dict[str, list[str]] = {"warning": [], "debug": []}
    for level, messages in seen.items():
        monkeypatch.setattr(
            _pdf.logger,
            level,
            lambda msg, *args, _messages=messages: _messages.append(msg % args),
        )
    return seen


def _reader(
    path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    n_pages: int | None,
    **kwargs: Any,
) -> PDFReader:
    """A reader whose page count is ``n_pages`` without opening the file."""
    monkeypatch.setattr(PDFReader, "_count_pages", lambda self: n_pages)
    return PDFReader(input_path=path, **kwargs)


def _texts(reader: PDFReader) -> list[tuple[int, str]]:
    return [(c["page_number"], c["text"]) for c in reader.get_raw_chunks()]


# ===========================================================================
# pdfminer: one pass over the file
# ===========================================================================


class TestPdfminerOnePass:
    def test_every_page_is_read_from_one_open_of_the_file(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box(f"page {i}\n")] for i in range(6)])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 6, prefer_backend="pdfminer")
        assert _texts(reader) == [(i, f"page {i}") for i in range(6)]
        assert len(fake.opens) == 1
        assert fake.opens[0]["file"] == str(pdf_path)
        assert fake.opens[0]["page_numbers"] is None
        assert fake.laid_out == [0, 1, 2, 3, 4, 5]

    def test_a_page_is_laid_out_only_when_it_is_asked_for(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box(f"page {i}")] for i in range(5)])
        fake.install(monkeypatch)
        chunks = _reader(
            pdf_path, monkeypatch, 5, prefer_backend="pdfminer"
        ).get_raw_chunks()
        assert next(chunks)["page_number"] == 0
        assert fake.laid_out == [0]
        assert next(chunks)["page_number"] == 1
        assert fake.laid_out == [0, 1]

    def test_the_file_is_closed_when_the_consumer_stops_early(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box(f"page {i}")] for i in range(5)])
        fake.install(monkeypatch)
        chunks = _reader(
            pdf_path, monkeypatch, 5, prefer_backend="pdfminer"
        ).get_raw_chunks()
        next(chunks)
        assert fake.open_now == 1
        chunks.close()
        assert fake.open_now == 0

    def test_the_file_is_closed_after_the_last_page(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("only")]])
        fake.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 1)) == [(0, "only")]
        assert fake.open_now == 0

    def test_only_text_containers_give_text_and_raw_text_keeps_whitespace(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("  a\n"), _Figure(), _Box("b  \n")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 1, prefer_backend="pdfminer")
        (chunk,) = list(reader.get_raw_chunks())
        assert chunk["raw_text"] == "  a\nb  \n"
        assert chunk["text"] == "a\nb"
        assert chunk["page_number"] == 0
        assert chunk["section_type"] == "text"

    def test_a_page_without_text_is_skipped_and_the_pass_goes_on(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")], [_Figure()], [_Box(" \n")], [_Box("d")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 4, prefer_backend="pdfminer")
        assert _texts(reader) == [(0, "a"), (3, "d")]
        assert len(fake.opens) == 1

    def test_the_password_reaches_pdfminer(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        reader = _reader(
            pdf_path, monkeypatch, 1, prefer_backend="pdfminer", password=_KEY
        )
        assert _texts(reader) == [(0, "a")]
        assert fake.opens[0]["password"] == _KEY

    def test_without_a_page_count_reading_stops_at_the_end_of_the_file(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        fake = _FakePdfminer([[_Box("a")], [_Box("b")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, None, prefer_backend="pdfminer")
        assert _texts(reader) == [(0, "a"), (1, "b")]
        assert len(fake.opens) == 1
        (warning,) = logged["warning"]
        assert "could not determine page count for doc.pdf" in warning

    def test_a_page_pdfminer_fails_on_ends_a_pdfminer_only_read(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")], ValueError("bad page"), [_Box("c")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 3, prefer_backend="pdfminer")
        assert _texts(reader) == [(0, "a")]
        assert fake.open_now == 0

    def test_missing_pdfminer_is_an_import_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _missing(monkeypatch, "pdfminer", "pdfminer.high_level", "pdfminer.layout")
        reader = _reader(pdf_path, monkeypatch, 1, prefer_backend="pdfminer")
        with pytest.raises(ImportError, match=r"pdfminer\.six is required"):
            list(reader.get_raw_chunks())


class TestPdfminerPages:
    def test_a_page_out_of_order_opens_the_file_at_that_page(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box(f"page {i}")] for i in range(6)])
        fake.install(monkeypatch)
        pages = _pdf._PdfminerPages(pdf_path)
        assert pages.text(4) == "page 4"
        assert pages.text(5) == "page 5"
        assert len(fake.opens) == 1
        wanted = fake.opens[0]["page_numbers"]
        assert [i for i in range(8) if i in wanted] == [4, 5, 6, 7]
        assert pages.text(2) == "page 2"
        assert len(fake.opens) == _TWICE
        assert fake.laid_out == [4, 5, 2]
        pages.close()
        assert fake.open_now == 0

    def test_a_page_past_the_end_is_none_every_time(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        pages = _pdf._PdfminerPages(pdf_path)
        assert pages.text(0) == "a"
        assert pages.text(1) is None
        assert pages.text(2) is None
        assert len(fake.opens) == 1

    def test_after_a_failed_page_the_next_one_is_read_from_a_new_open(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        fake = _FakePdfminer([[_Box("a")], ValueError("bad page"), [_Box("c")]])
        fake.install(monkeypatch)
        pages = _pdf._PdfminerPages(pdf_path)
        assert pages.text(0) == "a"
        assert pages.text(1) is None
        assert logged["debug"] == [
            "PDFReader: pdfminer failed on page 1 of doc.pdf: bad page"
        ]
        assert fake.open_now == 0
        assert pages.text(2) == "c"
        assert len(fake.opens) == _TWICE
        wanted = fake.opens[1]["page_numbers"]
        assert [i for i in range(5) if i in wanted] == [2, 3, 4]

    def test_close_twice_and_before_any_read(self, pdf_path: pathlib.Path) -> None:
        pages = _pdf._PdfminerPages(pdf_path)
        pages.close()
        pages.close()


# ===========================================================================
# pdfminer: backend keyword arguments
# ===========================================================================


class TestPdfminerKwargs:
    def test_without_kwargs_pdfminer_gets_its_defaults(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 1)
        assert reader.pdfminer_kwargs is None
        assert _texts(reader) == [(0, "a")]
        assert fake.opens[0]["laparams"] is None
        assert fake.opens[0]["caching"] is True

    @pytest.mark.parametrize("backend", [None, "pdfminer"])
    def test_kwargs_reach_extract_pages(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        backend: str | None,
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        reader = _reader(
            pdf_path,
            monkeypatch,
            1,
            prefer_backend=backend,
            pdfminer_kwargs={"caching": False},
        )
        assert _texts(reader) == [(0, "a")]
        (opened,) = fake.opens
        assert opened["caching"] is False
        assert opened["laparams"] is None

    def test_a_laparams_mapping_becomes_layout_parameters(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        given = {"laparams": {"boxes_flow": None}}
        reader = _reader(pdf_path, monkeypatch, 1, pdfminer_kwargs=given)
        assert _texts(reader) == [(0, "a")]
        (opened,) = fake.opens
        assert isinstance(opened["laparams"], _Params)
        assert opened["laparams"].given == {"boxes_flow": None, "line_margin": 0.5}
        assert given == {"laparams": {"boxes_flow": None}}

    def test_a_laparams_object_is_passed_as_it_is(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        params = _Params(line_margin=0.25)
        reader = _reader(pdf_path, monkeypatch, 1, pdfminer_kwargs={"laparams": params})
        assert _texts(reader) == [(0, "a")]
        assert fake.opens[0]["laparams"] is params

    def test_kwargs_are_used_again_when_the_file_is_reopened(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")], ValueError("bad page"), [_Box("c")]])
        fake.install(monkeypatch)
        pages = _pdf._PdfminerPages(
            pdf_path, "", {"caching": False, "laparams": {"line_margin": 0.25}}
        )
        assert [pages.text(i) for i in range(3)] == ["a", None, "c"]
        assert [o["caching"] for o in fake.opens] == [False, False]
        assert [o["laparams"].given["line_margin"] for o in fake.opens] == [
            0.25,
            0.25,
        ]

    def test_an_argument_extract_pages_does_not_know_is_a_value_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 1, pdfminer_kwargs={"cache": False})
        with pytest.raises(ValueError, match="pdfminer_kwargs is not accepted") as info:
            list(reader.get_raw_chunks())
        assert "extract_pages" in str(info.value)
        assert "cache" in str(info.value)
        assert isinstance(info.value.__cause__, TypeError)
        assert fake.opens == []

    def test_a_layout_argument_pdfminer_does_not_know_is_a_value_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        reader = _reader(
            pdf_path, monkeypatch, 1, pdfminer_kwargs={"laparams": {"box_flow": 1}}
        )
        with pytest.raises(ValueError, match="LAParams") as info:
            list(reader.get_raw_chunks())
        assert "pdfminer_kwargs['laparams']" in str(info.value)
        assert "box_flow" in str(info.value)
        assert isinstance(info.value.__cause__, TypeError)
        assert fake.opens == []

    @pytest.mark.parametrize(
        "bad", [["caching"], "caching=False", 0.5, {1: None}, {None: 1}]
    )
    def test_anything_but_a_mapping_of_names_is_a_type_error(
        self, pdf_path: pathlib.Path, bad: Any
    ) -> None:
        with pytest.raises(TypeError, match="pdfminer_kwargs must be a mapping"):
            PDFReader(input_path=pdf_path, pdfminer_kwargs=bad)

    @pytest.mark.parametrize("name", ["password", "page_numbers", "maxpages"])
    def test_an_argument_the_reader_sets_is_refused(
        self, pdf_path: pathlib.Path, name: str
    ) -> None:
        with pytest.raises(ValueError, match="the reader sets these") as info:
            PDFReader(input_path=pdf_path, pdfminer_kwargs={name: 1, "caching": 1})
        assert f"['{name}']" in str(info.value)

    def test_an_empty_mapping_changes_nothing(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 1, pdfminer_kwargs={})
        assert _texts(reader) == [(0, "a")]
        assert fake.opens[0]["laparams"] is None

    def test_pypdf_ignores_pdfminer_kwargs(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _missing(monkeypatch, "pdfminer", "pdfminer.high_level", "pdfminer.layout")
        fake = _FakePypdf(["A"])
        fake.install(monkeypatch)
        reader = _reader(
            pdf_path,
            monkeypatch,
            1,
            prefer_backend="pypdf",
            pdfminer_kwargs={"no_such_argument": 1},
        )
        assert _texts(reader) == [(0, "A")]


# ===========================================================================
# pypdf: one reader
# ===========================================================================


class TestPypdfOneReader:
    def test_every_page_is_read_from_one_reader(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePypdf([f"page {i}" for i in range(5)])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 5, prefer_backend="pypdf")
        assert _texts(reader) == [(i, f"page {i}") for i in range(5)]
        assert fake.opens == 1
        assert fake.asked == [0, 1, 2, 3, 4]
        assert fake.decrypts == []

    def test_an_encrypted_file_is_decrypted_once(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePypdf(["a", "b", "c"], password=_KEY)
        fake.install(monkeypatch)
        reader = _reader(
            pdf_path, monkeypatch, 3, prefer_backend="pypdf", password=_KEY
        )
        assert _texts(reader) == [(0, "a"), (1, "b"), (2, "c")]
        assert fake.decrypts == [_KEY]

    def test_a_wrong_password_warns_once_and_gives_nothing(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        fake = _FakePypdf(["a", "b"], password=_KEY)
        fake.install(monkeypatch)
        pages = _pdf._PypdfPages(pdf_path, _OTHER_KEY)
        assert pages.text(0) is None
        assert pages.text(1) is None
        assert logged["warning"] == [
            "PDFReader: failed to decrypt doc.pdf (wrong password?)."
        ]
        assert fake.opens == 1
        assert fake.asked == []

    def test_a_file_that_cannot_be_opened_is_not_tried_again(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePypdf(["a"], open_error=ValueError("not a PDF"))
        fake.install(monkeypatch)
        pages = _pdf._PypdfPages(pdf_path)
        assert pages.text(0) is None
        assert pages.text(1) is None
        assert fake.opens == 1

    def test_a_page_that_fails_leaves_the_others_readable(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePypdf(["a", ValueError("bad page"), "c"])
        fake.install(monkeypatch)
        pages = _pdf._PypdfPages(pdf_path)
        assert [pages.text(i) for i in range(3)] == ["a", None, "c"]
        assert fake.opens == 1

    def test_no_text_is_an_empty_string_and_past_the_end_is_none(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        fake = _FakePypdf([None, "b"])
        fake.install(monkeypatch)
        pages = _pdf._PypdfPages(pdf_path)
        assert pages.text(0) == ""
        assert pages.text(2) is None
        assert fake.asked == [0]
        assert logged["debug"] == []

    def test_missing_pypdf_is_an_import_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _missing(monkeypatch, "pypdf")
        reader = _reader(pdf_path, monkeypatch, 1, prefer_backend="pypdf")
        with pytest.raises(ImportError, match="pypdf is required"):
            list(reader.get_raw_chunks())


# ===========================================================================
# Auto mode: pdfminer, with pypdf for the pages it gives no text for
# ===========================================================================


class TestAutoMode:
    def test_pypdf_is_not_opened_when_pdfminer_reads_every_page(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        miner = _FakePdfminer([[_Box("a")], [_Box("b")]])
        miner.install(monkeypatch)
        backup = _FakePypdf(["A", "B"])
        backup.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 2)) == [(0, "a"), (1, "b")]
        assert backup.opens == 0

    def test_pages_without_text_come_from_one_pypdf_reader(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        miner = _FakePdfminer([[_Box("a")], [], [_Box(" ")], [_Box("d")], []])
        miner.install(monkeypatch)
        backup = _FakePypdf(["A", "B", "C", "D", ""])
        backup.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 5)) == [
            (0, "a"),
            (1, "B"),
            (2, "C"),
            (3, "d"),
        ]
        assert len(miner.opens) == 1
        assert backup.opens == 1
        assert backup.asked == [1, 2, 4]

    def test_a_page_pdfminer_fails_on_comes_from_pypdf_and_the_rest_follow(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        miner = _FakePdfminer([[_Box("a")], ValueError("bad page"), [_Box("c")]])
        miner.install(monkeypatch)
        backup = _FakePypdf(["A", "B", "C"])
        backup.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 3)) == [
            (0, "a"),
            (1, "B"),
            (2, "c"),
        ]
        assert len(miner.opens) == _TWICE
        assert miner.open_now == 0

    def test_a_page_neither_backend_can_read_ends_the_read(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        miner = _FakePdfminer([[_Box("a")], ValueError("bad page"), [_Box("c")]])
        miner.install(monkeypatch)
        backup = _FakePypdf(["A", ValueError("bad page"), "C"])
        backup.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 3)) == [(0, "a")]

    def test_without_pdfminer_every_page_comes_from_pypdf(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _missing(monkeypatch, "pdfminer", "pdfminer.high_level", "pdfminer.layout")
        backup = _FakePypdf(["A", "B"])
        backup.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 2)) == [(0, "A"), (1, "B")]
        assert backup.opens == 1

    def test_without_pypdf_a_page_without_text_is_skipped(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        miner = _FakePdfminer([[_Box("a")], [], [_Box("c")]])
        miner.install(monkeypatch)
        _missing(monkeypatch, "pypdf")
        assert _texts(_reader(pdf_path, monkeypatch, 3)) == [(0, "a"), (2, "c")]

    def test_without_either_library_the_error_names_both(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _missing(
            monkeypatch, "pdfminer", "pdfminer.high_level", "pdfminer.layout", "pypdf"
        )
        reader = _reader(pdf_path, monkeypatch, 1)
        with pytest.raises(ImportError, match=r"Neither pdfminer\.six nor pypdf"):
            list(reader.get_raw_chunks())

    def test_a_second_read_of_the_same_reader_starts_again(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        miner = _FakePdfminer([[_Box("a")], [_Box("b")]])
        miner.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 2, prefer_backend="pdfminer")
        assert _texts(reader) == _texts(reader) == [(0, "a"), (1, "b")]
        assert len(miner.opens) == _TWICE


# ===========================================================================
# The real libraries on a hand-written PDF
# ===========================================================================


def _write_pdf(path: pathlib.Path, texts: Sequence[str]) -> pathlib.Path:
    """Write a PDF with one page per entry of ``texts``; "" is a blank page."""
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(texts)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(texts)} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(texts):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
                f" /Contents {5 + 2 * i} 0 R"
                " /Resources << /Font << /F1 3 0 R >> >> >>"
            ).encode("ascii")
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream".encode("ascii")
        )
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


class TestRealLibraries:
    TEXTS = ("first page", "", "third page", "fourth page")
    EXPECTED = ((0, "first page"), (2, "third page"), (3, "fourth page"))

    def _locked(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """The test PDF, encrypted with ``_KEY``."""
        pypdf = pytest.importorskip("pypdf")
        plain = _write_pdf(tmp_path / "plain.pdf", self.TEXTS)
        writer = pypdf.PdfWriter()
        for page in pypdf.PdfReader(str(plain)).pages:
            writer.add_page(page)
        writer.encrypt(_KEY, algorithm="RC4-128")
        path = tmp_path / "locked.pdf"
        with path.open("wb") as handle:
            writer.write(handle)
        return path

    @pytest.mark.parametrize("backend", [None, "pdfminer"])
    def test_pdfminer_reads_each_page_in_order(
        self, tmp_path: pathlib.Path, backend: str | None
    ) -> None:
        pytest.importorskip("pdfminer.high_level")
        path = _write_pdf(tmp_path / "doc.pdf", self.TEXTS)
        reader = PDFReader(input_path=path, prefer_backend=backend)
        assert _texts(reader) == list(self.EXPECTED)

    def test_pdfminer_opens_the_file_once(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        high_level = pytest.importorskip("pdfminer.high_level")
        path = _write_pdf(tmp_path / "doc.pdf", self.TEXTS)
        opens = []
        real = high_level.extract_pages

        def counted(*args: Any, **kwargs: Any) -> Any:
            opens.append(kwargs.get("page_numbers"))
            return real(*args, **kwargs)

        monkeypatch.setattr(high_level, "extract_pages", counted)
        reader = PDFReader(input_path=path, prefer_backend="pdfminer")
        assert _texts(reader) == list(self.EXPECTED)
        assert opens == [None]

    def test_pdfminer_reads_with_layout_parameters(
        self, tmp_path: pathlib.Path
    ) -> None:
        pytest.importorskip("pdfminer.high_level")
        path = _write_pdf(tmp_path / "doc.pdf", self.TEXTS)
        reader = PDFReader(
            input_path=path,
            pdfminer_kwargs={"caching": False, "laparams": {"boxes_flow": None}},
        )
        assert _texts(reader) == list(self.EXPECTED)
        layout = pytest.importorskip("pdfminer.layout")
        with_object = PDFReader(
            input_path=path,
            pdfminer_kwargs={"laparams": layout.LAParams(boxes_flow=None)},
        )
        assert _texts(with_object) == list(self.EXPECTED)
        for bad in ({"no_such_argument": 1}, {"laparams": {"no_such_argument": 1}}):
            unknown = PDFReader(input_path=path, pdfminer_kwargs=bad)
            with pytest.raises(ValueError, match="no_such_argument"):
                list(unknown.get_raw_chunks())

    def test_pdfminer_reads_a_page_out_of_order(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("pdfminer.high_level")
        path = _write_pdf(tmp_path / "doc.pdf", self.TEXTS)
        pages = _pdf._PdfminerPages(path)
        try:
            assert pages.text(3).strip() == "fourth page"
            assert pages.text(0).strip() == "first page"
            assert pages.text(4) is None
        finally:
            pages.close()

    def test_the_page_count_of_an_encrypted_file_needs_its_password(
        self, tmp_path: pathlib.Path
    ) -> None:
        pytest.importorskip("pdfminer.high_level")
        path = self._locked(tmp_path)
        assert _pdf._count_pdf_pages_pdfminer(path, _KEY) == len(self.TEXTS)
        assert _pdf._count_pdf_pages_pdfminer(path, _OTHER_KEY) is None
        assert _pdf._count_pdf_pages_pdfminer(path) is None

    def test_pypdf_reads_each_page_in_order(self, tmp_path: pathlib.Path) -> None:
        pytest.importorskip("pypdf")
        path = _write_pdf(tmp_path / "doc.pdf", self.TEXTS)
        reader = PDFReader(input_path=path, prefer_backend="pypdf")
        assert _texts(reader) == list(self.EXPECTED)

    @pytest.mark.parametrize("backend", [None, "pdfminer", "pypdf"])
    def test_an_encrypted_file_is_read_with_its_password(
        self, tmp_path: pathlib.Path, backend: str | None
    ) -> None:
        if backend != "pypdf":
            pytest.importorskip("pdfminer.high_level")
        path = self._locked(tmp_path)
        reader = PDFReader(input_path=path, prefer_backend=backend, password=_KEY)
        assert _texts(reader) == list(self.EXPECTED)
        wrong = PDFReader(input_path=path, prefer_backend=backend, password=_OTHER_KEY)
        assert _texts(wrong) == []
