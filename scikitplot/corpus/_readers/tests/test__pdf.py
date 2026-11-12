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

from .. import _image, _pdf
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


class _Absent:
    """A meta path finder for which the named top-level packages do not exist."""

    def __init__(self, names: Sequence[str]) -> None:
        self._names = tuple(names)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        top = fullname.partition(".")[0]
        if top in self._names:
            # What Python raises when no distribution provides the package.
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)


def _missing(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """
    Make each of ``names`` an absent package or a missing module.

    A name without a dot is a package that is not installed: importing it or
    anything under it raises ``ModuleNotFoundError`` for the top-level name.
    A dotted name is a module missing from a package that is there.
    """
    absent = [name for name in names if "." not in name]
    for name in absent:
        for loaded in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            monkeypatch.delitem(sys.modules, loaded)
    if absent:
        monkeypatch.setattr(sys, "meta_path", [_Absent(absent), *sys.meta_path])
    for name in names:
        if "." in name:
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

    def test_a_page_pdfminer_fails_on_is_skipped_and_named(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        fake = _FakePdfminer([[_Box("a")], ValueError("bad page"), [_Box("c")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 3, prefer_backend="pdfminer")
        assert _texts(reader) == [(0, "a"), (2, "c")]
        assert fake.open_now == 0
        assert logged["warning"] == [
            (
                "PDFReader: 1 of 3 page(s) of doc.pdf could not be read and are"
                " skipped (pages 1; pdfminer: ValueError: bad page)."
            )
        ]

    def test_missing_pdfminer_is_an_import_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _missing(monkeypatch, "pdfminer")
        reader = _reader(pdf_path, monkeypatch, 1, prefer_backend="pdfminer")
        with pytest.raises(ImportError, match=r"pdfminer\.six is required"):
            list(reader.get_raw_chunks())


class TestDamagedPdfminer:
    @pytest.mark.parametrize("module", ["pdfminer.high_level", "pdfminer.layout"])
    def test_a_package_without_one_of_its_modules_is_named_as_incomplete(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, module: str
    ) -> None:
        _FakePdfminer([[_Box("a")]]).install(monkeypatch)
        monkeypatch.setitem(sys.modules, module, None)
        reader = _reader(pdf_path, monkeypatch, 1, prefer_backend="pdfminer")
        with pytest.raises(ImportError, match="is incomplete") as info:
            list(reader.get_raw_chunks())
        message = str(info.value)
        assert module in message
        assert "must not be installed together" in message
        assert "pip install pdfminer.six" in message
        assert isinstance(info.value.__cause__, ImportError)

    def test_a_module_without_a_name_is_named_as_incomplete(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakePdfminer([[_Box("a")]]).install(monkeypatch)
        monkeypatch.delattr(sys.modules["pdfminer.layout"], "LAParams")
        reader = _reader(pdf_path, monkeypatch, 1, prefer_backend="pdfminer")
        with pytest.raises(ImportError, match="is incomplete") as info:
            list(reader.get_raw_chunks())
        assert "LAParams" in str(info.value)

    def test_an_absent_package_is_an_install_hint_only(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _missing(monkeypatch, "pdfminer")
        reader = _reader(pdf_path, monkeypatch, 1, prefer_backend="pdfminer")
        with pytest.raises(ImportError, match=r"pdfminer\.six is required") as info:
            list(reader.get_raw_chunks())
        assert "incomplete" not in str(info.value)


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
# Files and pages that cannot be read, or hold no text
# ===========================================================================


class TestNothingSilent:
    def test_a_file_pdfminer_cannot_open_is_a_value_error_with_the_reason(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([KeyError("No /Root object")])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, None, prefer_backend="pdfminer")
        with pytest.raises(ValueError, match=r"cannot read doc\.pdf") as info:
            list(reader.get_raw_chunks())
        message = str(info.value)
        assert "(pdfminer: KeyError: 'No /Root object')" in message
        assert "complete PDF" in message
        assert "'password'" in message
        assert fake.open_now == 0

    def test_both_backends_are_named_when_neither_can_open_the_file(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakePdfminer([ValueError()]).install(monkeypatch)
        _FakePypdf(["A"], open_error=OSError("truncated")).install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, None)
        with pytest.raises(ValueError, match=r"cannot read doc\.pdf") as info:
            list(reader.get_raw_chunks())
        assert "(pdfminer: ValueError; pypdf: OSError: truncated)" in str(info.value)

    def test_a_wrong_password_is_a_value_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakePypdf(["A"], password=_KEY).install(monkeypatch)
        reader = _reader(
            pdf_path, monkeypatch, None, prefer_backend="pypdf", password=_OTHER_KEY
        )
        with pytest.raises(ValueError, match=r"cannot read doc\.pdf") as info:
            list(reader.get_raw_chunks())
        assert "pypdf: the password does not decrypt the file" in str(info.value)

    def test_every_page_failing_with_a_known_count_is_a_value_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakePdfminer([ValueError("one"), ValueError("two")]).install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 2, prefer_backend="pdfminer")
        with pytest.raises(ValueError, match="pdfminer: ValueError: one"):
            list(reader.get_raw_chunks())

    def test_pages_without_text_are_one_warning_and_no_error(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        _FakePdfminer([[_Figure()], [_Box(" \n")], []]).install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 3, prefer_backend="pdfminer")
        assert _texts(reader) == []
        assert logged["warning"] == [
            (
                "PDFReader: none of the 3 page(s) read from doc.pdf has extractable"
                " text. The file probably holds scanned images; pass ocr='auto'"
                " to read them with OCR."
            )
        ]

    def test_one_page_with_text_is_no_warning(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        _FakePdfminer([[_Figure()], [_Box("b")], []]).install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 3, prefer_backend="pdfminer")
        assert _texts(reader) == [(1, "b")]
        assert logged["warning"] == []

    def test_a_file_without_pages_is_neither_an_error_nor_a_scan(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        _FakePdfminer([]).install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, 0, prefer_backend="pdfminer")
        assert _texts(reader) == []
        (warning,) = logged["warning"]
        assert "could not determine page count" in warning

    def test_without_a_count_a_failed_page_ends_the_read_without_a_warning(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        fake = _FakePdfminer([[_Box("a")], ValueError("bad page"), [_Box("c")]])
        fake.install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, None, prefer_backend="pdfminer")
        assert _texts(reader) == [(0, "a")]
        (warning,) = logged["warning"]
        assert "could not determine page count" in warning

    def test_without_a_count_reading_is_not_cut_at_a_fixed_number_of_pages(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n_pages = 10_050
        _FakePypdf(["x"] * n_pages).install(monkeypatch)
        reader = _reader(pdf_path, monkeypatch, None, prefer_backend="pypdf")
        assert sum(1 for _ in reader.get_raw_chunks()) == n_pages

    def test_stopping_early_on_a_readable_file_raises_nothing(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakePdfminer([[_Box("a")], ValueError("bad page")])
        fake.install(monkeypatch)
        chunks = _reader(
            pdf_path, monkeypatch, 2, prefer_backend="pdfminer"
        ).get_raw_chunks()
        assert next(chunks)["text"] == "a"
        chunks.close()
        assert fake.open_now == 0


# ===========================================================================
# OCR of rendered pages
# ===========================================================================


class _FakeImage:
    """A rendered page as the engine sees it."""

    def __init__(self, page: int, mode: str = "RGB") -> None:
        self.page = page
        self.mode = mode

    def convert(self, mode: str) -> _FakeImage:
        return _FakeImage(self.page, mode)


class _FakePdfium:
    """
    Stand-in for ``pypdfium2``.

    Parameters
    ----------
    n_pages : int
        Pages in the document.
    fail : dict, optional
        Page -> exception raised when that page is rendered.
    open_error : Exception or None
        Raised when the document is opened.
    mode : str
        Mode of the rendered images.
    """

    def __init__(
        self,
        n_pages: int,
        fail: dict[int, Exception] | None = None,
        open_error: Exception | None = None,
        mode: str = "RGB",
    ) -> None:
        self.n_pages = n_pages
        self.fail = fail or {}
        self.open_error = open_error
        self.mode = mode
        self.opens: list[dict[str, Any]] = []
        self.renders: list[dict[str, Any]] = []
        self.open_now = 0
        self.pages_open = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = self

        class _Bitmap:
            def __init__(self, page: int) -> None:
                self._page = page

            def to_pil(self) -> _FakeImage:
                return _FakeImage(self._page, fake.mode)

            def close(self) -> None:
                pass

        class _Page:
            def __init__(self, index: int) -> None:
                self._index = index
                fake.pages_open += 1

            def render(self, scale: float = 1, **kwargs: Any) -> _Bitmap:
                fake.renders.append({"page": self._index, "scale": scale, **kwargs})
                if "no_such_option" in kwargs:
                    raise TypeError("render() got an unexpected keyword argument")
                if self._index in fake.fail:
                    raise fake.fail[self._index]
                return _Bitmap(self._index)

            def close(self) -> None:
                fake.pages_open -= 1

        class PdfDocument:
            def __init__(self, path: str, password: Any = None) -> None:
                fake.opens.append({"path": path, "password": password})
                if fake.open_error is not None:
                    raise fake.open_error
                fake.open_now += 1

            def __getitem__(self, index: int) -> _Page:
                if index >= fake.n_pages:
                    raise IndexError(index)
                return _Page(index)

            def close(self) -> None:
                fake.open_now -= 1

        module = types.ModuleType("pypdfium2")
        module.PdfDocument = PdfDocument
        monkeypatch.setitem(sys.modules, "pypdfium2", module)


def _engine(image: _FakeImage) -> tuple[str, float]:
    """An OCR engine that reads the page number."""
    return f"ocr text of page {image.page}\n", 0.75


def _ocr_reader(
    pdf_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    layers: list[Any],
    **kwargs: Any,
) -> tuple[PDFReader, _FakePdfium]:
    """A pdfminer-only reader over ``layers`` with OCR by ``_engine``."""
    _FakePdfminer(layers).install(monkeypatch)
    pdfium = _FakePdfium(len(layers))
    pdfium.install(monkeypatch)
    kwargs.setdefault("ocr_engine", _engine)
    reader = _reader(
        pdf_path, monkeypatch, len(layers), prefer_backend="pdfminer", **kwargs
    )
    return reader, pdfium


class TestOcrPolicy:
    def test_off_is_the_default_and_renders_nothing(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reader, pdfium = _ocr_reader(pdf_path, monkeypatch, [[], [_Box("b")]])
        assert reader.ocr == "off"
        chunks = list(reader.get_raw_chunks())
        assert [(c["page_number"], c["text"]) for c in chunks] == [(1, "b")]
        assert "ocr_engine" not in chunks[0]
        assert pdfium.opens == []

    def test_auto_reads_only_pages_without_a_text_layer(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        layers = [[], [_Box("text layer")], [_Box(" \n")], [_Figure()]]
        reader, pdfium = _ocr_reader(pdf_path, monkeypatch, layers, ocr="auto")
        chunks = list(reader.get_raw_chunks())
        assert [(c["page_number"], c["text"]) for c in chunks] == [
            (0, "ocr text of page 0"),
            (1, "text layer"),
            (2, "ocr text of page 2"),
            (3, "ocr text of page 3"),
        ]
        assert [r["page"] for r in pdfium.renders] == [0, 2, 3]
        assert chunks[0]["raw_text"] == "ocr text of page 0\n"
        assert chunks[0]["ocr_engine"] == "custom"
        assert chunks[0]["confidence"] == 0.75  # noqa: PLR2004
        assert "ocr_engine" not in chunks[1]
        assert "confidence" not in chunks[1]
        assert len(pdfium.opens) == 1
        assert pdfium.open_now == 0
        assert pdfium.pages_open == 0

    @pytest.mark.parametrize(
        ("min_chars", "read"), [(1, [1]), (7, [0, 1]), (8, [0, 1, 2])]
    )
    def test_ocr_min_chars_sets_how_short_a_text_layer_is_not_trusted(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        min_chars: int,
        read: list[int],
    ) -> None:
        layers = [[_Box("Page 1")], [], [_Box("7 chars")]]
        reader, pdfium = _ocr_reader(
            pdf_path, monkeypatch, layers, ocr="auto", ocr_min_chars=min_chars
        )
        list(reader.get_raw_chunks())
        assert [r["page"] for r in pdfium.renders] == read

    def test_always_replaces_the_text_layer_unless_ocr_finds_nothing(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def engine(image: _FakeImage) -> str:
            return "" if image.page == 1 else f"seen {image.page}"

        layers = [[_Box("layer 0")], [_Box("layer 1")], []]
        reader, _ = _ocr_reader(
            pdf_path, monkeypatch, layers, ocr="always", ocr_engine=engine
        )
        chunks = list(reader.get_raw_chunks())
        assert [(c["page_number"], c["text"]) for c in chunks] == [
            (0, "seen 0"),
            (1, "layer 1"),
            (2, "seen 2"),
        ]
        assert "confidence" not in chunks[0]
        assert "ocr_engine" not in chunks[1]

    def test_a_policy_callable_decides_from_page_number_and_text(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, str]] = []

        def policy(page_number: int, text: str) -> bool:
            calls.append((page_number, text))
            return page_number == 2  # noqa: PLR2004

        layers = [[_Box(" a \n")], [], [_Box("c")]]
        reader, pdfium = _ocr_reader(pdf_path, monkeypatch, layers, ocr=policy)
        assert _texts(reader) == [(0, "a"), (2, "ocr text of page 2")]
        assert calls == [(0, "a"), (1, ""), (2, "c")]
        assert [r["page"] for r in pdfium.renders] == [2]

    def test_a_policy_that_does_not_return_a_bool_is_a_type_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reader, _ = _ocr_reader(
            pdf_path, monkeypatch, [[_Box("a")]], ocr=lambda page, text: 1
        )
        with pytest.raises(TypeError, match="must return a bool; got 'int'"):
            list(reader.get_raw_chunks())

    def test_ocr_max_pages_bounds_the_pages_read_and_names_the_rest(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        layers = [[], [_Box("b")], [], [], [], [_Box("f")], []]
        reader, pdfium = _ocr_reader(
            pdf_path, monkeypatch, layers, ocr="auto", ocr_max_pages=2
        )
        assert _texts(reader) == [
            (0, "ocr text of page 0"),
            (1, "b"),
            (2, "ocr text of page 2"),
            (5, "f"),
        ]
        assert [r["page"] for r in pdfium.renders] == [0, 2]
        assert logged["warning"] == [
            (
                "PDFReader: 3 more page(s) of doc.pdf needed OCR after"
                " ocr_max_pages=2 was reached (pages 3-4, 6); their text layer"
                " is used. Pass ocr_max_pages=None to read them all."
            )
        ]

    def test_no_limit_reads_every_page_and_warns_nothing(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        reader, pdfium = _ocr_reader(
            pdf_path, monkeypatch, [[]] * 70, ocr="auto", ocr_max_pages=None
        )
        assert len(_texts(reader)) == 70  # noqa: PLR2004
        assert len(pdfium.renders) == 70  # noqa: PLR2004
        assert logged["warning"] == []

    def test_with_ocr_on_a_file_without_text_says_so_without_the_hint(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        reader, _ = _ocr_reader(
            pdf_path, monkeypatch, [[], []], ocr="auto", ocr_engine=lambda image: ""
        )
        assert _texts(reader) == []
        assert logged["warning"] == [
            (
                "PDFReader: none of the 2 page(s) read from doc.pdf has"
                " extractable text, with OCR as configured."
            )
        ]

    def test_a_custom_reader_ignores_ocr(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pdfium = _FakePdfium(1)
        pdfium.install(monkeypatch)
        reader = PDFReader(
            input_path=pdf_path,
            prefer_backend="custom",
            custom_extractor=lambda path, **kw: [{"text": "custom"}],
            ocr="always",
        )
        assert [c["text"] for c in reader.get_raw_chunks()] == ["custom"]
        assert pdfium.opens == []


class TestOcrRendering:
    def test_pages_are_rendered_at_ocr_dpi_with_pypdfium2_kwargs(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reader, pdfium = _ocr_reader(
            pdf_path,
            monkeypatch,
            [[]],
            ocr="auto",
            ocr_dpi=144,
            pypdfium2_kwargs={"grayscale": True},
        )
        list(reader.get_raw_chunks())
        assert pdfium.renders == [{"page": 0, "scale": 2.0, "grayscale": True}]

    def test_the_password_reaches_pypdfium2_and_an_empty_one_is_none(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for password, expected in (("", None), (_KEY, _KEY)):
            reader, pdfium = _ocr_reader(
                pdf_path, monkeypatch, [[]], ocr="auto", password=password
            )
            list(reader.get_raw_chunks())
            assert pdfium.opens == [{"path": str(pdf_path), "password": expected}]

    def test_an_image_in_another_mode_is_converted_to_rgb(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        modes: list[str] = []

        def engine(image: _FakeImage) -> str:
            modes.append(image.mode)
            return "x"

        _FakePdfminer([[], []]).install(monkeypatch)
        pdfium = _FakePdfium(2, mode="BGRA")
        pdfium.install(monkeypatch)
        reader = _reader(
            pdf_path,
            monkeypatch,
            2,
            prefer_backend="pdfminer",
            ocr="auto",
            ocr_engine=engine,
        )
        list(reader.get_raw_chunks())
        assert modes == ["RGB", "RGB"]
        pdfium.mode = "L"
        list(reader.get_raw_chunks())
        assert modes == ["RGB", "RGB", "L", "L"]

    def test_an_option_pypdfium2_rejects_is_a_value_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reader, pdfium = _ocr_reader(
            pdf_path,
            monkeypatch,
            [[]],
            ocr="auto",
            pypdfium2_kwargs={"no_such_option": 1},
        )
        with pytest.raises(
            ValueError, match="pypdfium2_kwargs is not accepted"
        ) as info:
            list(reader.get_raw_chunks())
        assert isinstance(info.value.__cause__, TypeError)
        assert pdfium.open_now == 0
        assert pdfium.pages_open == 0

    def test_a_page_that_cannot_be_rendered_keeps_its_text_layer(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        _FakePdfminer([[_Box("a")], [], [_Box("c")], []]).install(monkeypatch)
        pdfium = _FakePdfium(4, fail={0: OSError("bad stream"), 3: TypeError()})
        pdfium.install(monkeypatch)
        reader = _reader(
            pdf_path,
            monkeypatch,
            4,
            prefer_backend="pdfminer",
            ocr="always",
            ocr_engine=_engine,
        )
        assert _texts(reader) == [
            (0, "a"),
            (1, "ocr text of page 1"),
            (2, "ocr text of page 2"),
        ]
        assert logged["warning"] == [
            (
                "PDFReader: 2 page(s) of doc.pdf could not be rendered for OCR"
                " (pages 0, 3; OSError: bad stream); their text layer is used."
            )
        ]
        assert pdfium.pages_open == 0

    def test_a_file_pypdfium2_cannot_open_is_tried_once(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        _FakePdfminer([[_Box("a")], [], []]).install(monkeypatch)
        pdfium = _FakePdfium(3, open_error=ValueError("not a PDF"))
        pdfium.install(monkeypatch)
        reader = _reader(
            pdf_path,
            monkeypatch,
            3,
            prefer_backend="pdfminer",
            ocr="auto",
            ocr_engine=_engine,
        )
        assert _texts(reader) == [(0, "a")]
        assert len(pdfium.opens) == 1
        (warning,) = logged["warning"]
        assert "(pages 1-2; ValueError: not a PDF)" in warning

    def test_without_pypdfium2_ocr_is_an_import_error_when_a_page_needs_it(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakePdfminer([[_Box("a")], []]).install(monkeypatch)
        _missing(monkeypatch, "pypdfium2")
        reader = _reader(
            pdf_path, monkeypatch, 2, prefer_backend="pdfminer", ocr="auto"
        )
        with pytest.raises(ImportError, match="pip install pypdfium2"):
            list(reader.get_raw_chunks())

    def test_without_pypdfium2_a_file_with_text_is_read_as_before(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakePdfminer([[_Box("a")], [_Box("b")]]).install(monkeypatch)
        _missing(monkeypatch, "pypdfium2")
        reader = _reader(
            pdf_path, monkeypatch, 2, prefer_backend="pdfminer", ocr="auto"
        )
        assert _texts(reader) == [(0, "a"), (1, "b")]

    def test_the_document_is_closed_when_the_consumer_stops_early(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reader, pdfium = _ocr_reader(pdf_path, monkeypatch, [[], []], ocr="auto")
        chunks = reader.get_raw_chunks()
        next(chunks)
        assert pdfium.open_now == 1
        chunks.close()
        assert pdfium.open_now == 0


class TestOcrEngines:
    def test_tesseract_gets_the_image_and_the_language(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, Any]] = []

        def fake(image: _FakeImage, lang: Any) -> tuple[str, float]:
            calls.append((image.page, lang))
            return "read", 0.5

        monkeypatch.setattr(_image, "_ocr_tesseract", fake)
        reader, _ = _ocr_reader(
            pdf_path,
            monkeypatch,
            [[]],
            ocr="auto",
            ocr_engine="tesseract",
            ocr_lang="deu",
        )
        (chunk,) = list(reader.get_raw_chunks())
        assert calls == [(0, "deu")]
        assert chunk["ocr_engine"] == "tesseract"
        assert chunk["confidence"] == 0.5  # noqa: PLR2004

    def test_easyocr_reuses_its_reader_across_pages(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        given: list[Any] = []

        def fake(image: _FakeImage, lang: Any, reader: Any) -> tuple[str, float, Any]:
            given.append(reader)
            return f"page {image.page}", 0.9, "loaded model"

        monkeypatch.setattr(_image, "_ocr_easyocr", fake)
        reader, _ = _ocr_reader(
            pdf_path, monkeypatch, [[], [], []], ocr="auto", ocr_engine="easyocr"
        )
        chunks = list(reader.get_raw_chunks())
        assert given == [None, "loaded model", "loaded model"]
        assert {c["ocr_engine"] for c in chunks} == {"easyocr"}

    @pytest.mark.parametrize(
        "result",
        [None, 3, ("text",), ("text", 1.5), ("text", -0.1), ("t", True), (1, 0.5)],
    )
    def test_an_engine_returning_anything_else_is_a_type_error(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, result: Any
    ) -> None:
        reader, pdfium = _ocr_reader(
            pdf_path, monkeypatch, [[]], ocr="auto", ocr_engine=lambda image: result
        )
        with pytest.raises(TypeError, match="ocr_engine callable must return"):
            list(reader.get_raw_chunks())
        assert pdfium.open_now == 0

    @pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0])
    def test_a_confidence_on_the_bounds_is_kept(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, confidence: Any
    ) -> None:
        reader, _ = _ocr_reader(
            pdf_path,
            monkeypatch,
            [[]],
            ocr="auto",
            ocr_engine=lambda image: ("t", confidence),
        )
        (chunk,) = list(reader.get_raw_chunks())
        assert chunk["confidence"] == float(confidence)

    def test_an_engine_error_is_not_swallowed(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def engine(image: _FakeImage) -> str:
            raise RuntimeError("tesseract is not installed")

        reader, pdfium = _ocr_reader(
            pdf_path, monkeypatch, [[]], ocr="auto", ocr_engine=engine
        )
        with pytest.raises(RuntimeError, match="tesseract is not installed"):
            list(reader.get_raw_chunks())
        assert pdfium.open_now == 0


class TestOcrValidation:
    @pytest.mark.parametrize(
        ("field", "value", "error"),
        [
            ("ocr", "on", ValueError),
            ("ocr", None, ValueError),
            ("ocr", True, ValueError),
            ("ocr_engine", "paddle", ValueError),
            ("ocr_engine", None, ValueError),
            ("ocr_min_chars", 0, ValueError),
            ("ocr_min_chars", True, ValueError),
            ("ocr_min_chars", "3", ValueError),
            ("ocr_max_pages", 0, ValueError),
            ("ocr_max_pages", False, ValueError),
            ("ocr_max_pages", 2.0, ValueError),
            ("ocr_dpi", 0, ValueError),
            ("ocr_dpi", 1201, ValueError),
            ("ocr_dpi", 300.0, ValueError),
            ("ocr_dpi", True, ValueError),
            ("ocr_lang", 3, TypeError),
            ("ocr_lang", ["eng"], TypeError),
            ("pypdfium2_kwargs", ["grayscale"], TypeError),
            ("pypdfium2_kwargs", {1: True}, TypeError),
            ("pypdfium2_kwargs", {"scale": 2}, ValueError),
        ],
    )
    def test_a_value_outside_the_documented_ones_is_refused(
        self, pdf_path: pathlib.Path, field: str, value: Any, error: type
    ) -> None:
        with pytest.raises(error, match=field):
            PDFReader(input_path=pdf_path, **{field: value})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ocr": "auto", "ocr_dpi": 1, "ocr_max_pages": None},
            {"ocr": "always", "ocr_dpi": 1200, "ocr_min_chars": 10_000},
            {"ocr": lambda page, text: False, "ocr_engine": lambda image: ""},
            {"ocr_engine": "easyocr", "ocr_lang": "en", "pypdfium2_kwargs": {}},
        ],
    )
    def test_the_documented_values_are_accepted(
        self, pdf_path: pathlib.Path, kwargs: dict[str, Any]
    ) -> None:
        PDFReader(input_path=pdf_path, **kwargs)


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
        _missing(monkeypatch, "pdfminer")
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

    def test_a_page_neither_backend_can_read_is_skipped_and_named(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        miner = _FakePdfminer(
            [[_Box("a")], ValueError("bad page"), [_Box("c")], KeyError("font")]
        )
        miner.install(monkeypatch)
        backup = _FakePypdf(["A", OSError("bad stream"), "C", OSError("later")])
        backup.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 4)) == [(0, "a"), (2, "c")]
        assert logged["warning"] == [
            (
                "PDFReader: 2 of 4 page(s) of doc.pdf could not be read and are"
                " skipped (pages 1, 3; pdfminer: ValueError: bad page;"
                " pypdf: OSError: bad stream)."
            )
        ]

    def test_without_pdfminer_every_page_comes_from_pypdf(
        self, pdf_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _missing(monkeypatch, "pdfminer")
        backup = _FakePypdf(["A", "B"])
        backup.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 2)) == [(0, "A"), (1, "B")]
        assert backup.opens == 1

    def test_a_damaged_pdfminer_is_said_once_and_pypdf_reads(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        _FakePdfminer([]).install(monkeypatch)
        monkeypatch.setitem(sys.modules, "pdfminer.high_level", None)
        backup = _FakePypdf(["A", "B"])
        backup.install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 2)) == [(0, "A"), (1, "B")]
        (warning,) = logged["warning"]
        assert "reading with pypdf" in warning
        assert "the installed package 'pdfminer' is incomplete" in warning
        assert "pip uninstall pdfminer pdfminer.six" in warning

    def test_a_missing_pdfminer_is_not_a_warning(
        self,
        pdf_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        logged: dict[str, list[str]],
    ) -> None:
        _missing(monkeypatch, "pdfminer")
        _FakePypdf(["A"]).install(monkeypatch)
        assert _texts(_reader(pdf_path, monkeypatch, 1)) == [(0, "A")]
        assert logged["warning"] == []

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
        _missing(monkeypatch, "pdfminer", "pypdf")
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

    def test_a_scanned_page_is_read_with_tesseract(
        self, tmp_path: pathlib.Path
    ) -> None:
        pytest.importorskip("pypdfium2")
        pytest.importorskip("pytesseract")
        pil_image = pytest.importorskip("PIL.Image")
        pil_draw = pytest.importorskip("PIL.ImageDraw")
        pil_font = pytest.importorskip("PIL.ImageFont")
        try:
            font = pil_font.load_default(size=72)
        except TypeError:  # Pillow < 10.1 has no sized default font
            pytest.skip("Pillow's default font has no size here")
        scan = pil_image.new("RGB", (1700, 600), "white")
        pil_draw.Draw(scan).text((80, 200), "SCANNED PAGE 42", fill="black", font=font)
        path = tmp_path / "scan.pdf"
        scan.save(path, "PDF", resolution=200.0)
        reader = PDFReader(input_path=path, ocr="auto")
        try:
            chunks = list(reader.get_raw_chunks())
        except RuntimeError as exc:  # the Tesseract binary is not installed
            pytest.skip(str(exc))
        (chunk,) = chunks
        assert chunk["page_number"] == 0
        assert "SCANNED PAGE 42" in " ".join(chunk["text"].split())
        assert chunk["ocr_engine"] == "tesseract"
        assert 0.0 <= chunk["confidence"] <= 1.0
        assert _texts(PDFReader(input_path=path)) == []

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
        with pytest.raises(ValueError, match=r"cannot read locked\.pdf") as info:
            list(wrong.get_raw_chunks())
        assert "'password'" in str(info.value)

    @pytest.mark.parametrize("backend", [None, "pdfminer", "pypdf"])
    @pytest.mark.parametrize(
        "content", [b"", b"this is not a PDF\n" * 20, b"%PDF-1.4\n1 0 obj\n<< /Type"]
    )
    def test_a_file_that_is_not_a_complete_pdf_is_a_value_error(
        self, tmp_path: pathlib.Path, backend: str | None, content: bytes
    ) -> None:
        pytest.importorskip("pypdf" if backend == "pypdf" else "pdfminer.high_level")
        path = tmp_path / "broken.pdf"
        path.write_bytes(content)
        reader = PDFReader(input_path=path, prefer_backend=backend)
        with pytest.raises(ValueError, match=r"cannot read broken\.pdf"):
            list(reader.get_raw_chunks())

    @pytest.mark.parametrize("backend", [None, "pdfminer", "pypdf"])
    def test_a_file_of_blank_pages_is_a_warning_not_an_error(
        self,
        tmp_path: pathlib.Path,
        backend: str | None,
        logged: dict[str, list[str]],
    ) -> None:
        pytest.importorskip("pypdf" if backend == "pypdf" else "pdfminer.high_level")
        path = _write_pdf(tmp_path / "scan.pdf", ["", "", ""])
        assert _texts(PDFReader(input_path=path, prefer_backend=backend)) == []
        (warning,) = logged["warning"]
        assert "none of the 3 page(s) read from scan.pdf" in warning
