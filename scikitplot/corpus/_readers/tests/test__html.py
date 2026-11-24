# scikitplot/corpus/_readers/tests/test__html.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :class:`~scikitplot.corpus._readers._web.HTMLReader` (CX-50).

Local HTML is read offline through the parse path :class:`WebReader` uses.
The encoding order is BOM, UTF-8, a declared ``<meta>`` charset that
decodes the bytes, then the :class:`TextReader` fallbacks.
"""

from __future__ import annotations

import builtins
import time
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("bs4")

from ... import _readers  # noqa: F401  (registers the readers)
from ..._base import DocumentReader
from ..._schema import SectionType, SourceType
from .. import _web
from .._web import HTMLReader, _declared_charset, _html_chunks

_PAGE = (
    "<html><head><title>Guide</title><style>p{}</style>"
    "<script>var hidden = 1;</script></head><body>"
    "<h2>Install</h2><p>Run the <b>installer</b> now.</p>"
    "<noscript>enable js</noscript><ul><li>one</li><li>two</li></ul>"
    "</body></html>"
)


def _write(tmp_path: Path, data: bytes, name: str = "page.html") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _chunks(path: Path, **kwargs) -> list[dict]:
    return list(HTMLReader(input_path=path, **kwargs).get_raw_chunks())


@pytest.mark.parametrize("suffix", [".html", ".htm", ".xhtml", ".HTML"])
def test_local_html_extensions_create_the_html_reader(tmp_path, suffix):
    path = _write(tmp_path, _PAGE.encode(), "page" + suffix)
    assert type(DocumentReader.create(path)) is HTMLReader


def test_elements_are_extracted_and_scripts_dropped(tmp_path):
    chunks = _chunks(_write(tmp_path, _PAGE.encode()))
    texts = [c["text"] for c in chunks]
    assert texts == ["Guide", "Install", "Run the installer now.", "one", "two"]
    joined = " ".join(texts)
    assert "hidden" not in joined and "enable js" not in joined and "p{}" not in joined
    assert {c["source_type"] for c in chunks} == {SourceType.WEB.value}
    assert [c["element_index"] for c in chunks] == list(range(5))
    assert chunks[0]["section_type"] == SectionType.TITLE.value
    assert all("url" not in c for c in chunks)


def test_a_saved_page_gives_the_chunks_of_the_live_page(tmp_path):
    from bs4 import BeautifulSoup

    live = list(_html_chunks(_PAGE, None, BeautifulSoup))
    local = _chunks(_write(tmp_path, _PAGE.encode()))
    strip = lambda c: {k: v for k, v in c.items() if k != "encoding"}  # noqa: E731
    assert [strip(c) for c in local] == live


def test_extract_tags_narrows_the_elements(tmp_path):
    chunks = _chunks(_write(tmp_path, _PAGE.encode()), extract_tags=["li"])
    assert [c["text"] for c in chunks] == ["one", "two"]


def test_nested_elements_are_read_once(tmp_path):
    html = "<ul><li>outer<ul><li>inner</li></ul></li></ul><pre><code>x = 1</code></pre>"
    chunks = _chunks(_write(tmp_path, html.encode()))
    assert [c["text"] for c in chunks] == ["outer inner", "x = 1"]


def test_unclosed_nested_tags_are_read_in_linear_time(tmp_path):
    # Unclosed <li> nest under html.parser; reading every level repeated
    # the text quadratically (minutes for this 72 kB file before the fix).
    path = _write(tmp_path, b"<li>xy" * 12_000)
    start = time.perf_counter()
    chunks = _chunks(path)
    assert time.perf_counter() - start < 10
    assert len(chunks) == 1
    assert chunks[0]["text"].count("xy") == 12_000


@pytest.mark.parametrize(
    ("raw", "codec", "text"),
    [
        # Not UTF-8; the declared charset decodes it.
        (
            b'<meta charset="windows-1251"><p>\xcf\xf0\xe8\xe2\xe5\xf2</p>',
            "windows-1251",
            "Привет",
        ),
        # http-equiv form.
        (
            b'<meta http-equiv="Content-Type" content="text/html; charset=koi8-r">'
            b"<p>\xf0\xd2\xc9</p>",
            "koi8-r",
            "При",
        ),
        # A stale declaration on a UTF-8 page: UTF-8 wins.
        (
            '<meta charset="iso-8859-1"><p>café</p>'.encode(),
            "utf-8",
            "café",
        ),
        # A declaration that does not decode the bytes falls through.
        (b'<meta charset="ascii"><p>caf\xe9</p>', "cp1252", "café"),
        # An unknown codec name is ignored.
        (b'<meta charset="no-such-codec"><p>caf\xe9</p>', "cp1252", "café"),
        # A byte-order mark wins over a declaration.
        (
            b"\xef\xbb\xbf" + '<meta charset="koi8-r"><p>café</p>'.encode(),
            "utf-8-sig",
            "café",
        ),
    ],
)
def test_encoding_order(tmp_path, raw, codec, text):
    chunks = _chunks(_write(tmp_path, raw))
    assert [c["text"] for c in chunks] == [text]
    assert chunks[0]["encoding"] == codec


def test_utf16_with_bom_is_read(tmp_path):
    chunks = _chunks(_write(tmp_path, "<p>hello</p>".encode("utf-16")))
    assert [c["text"] for c in chunks] == ["hello"]


def test_an_explicit_encoding_is_used(tmp_path):
    raw = b'<meta charset="windows-1251"><p>\xe9</p>'
    chunks = _chunks(_write(tmp_path, raw), encoding="latin-1")
    assert chunks[0]["encoding"] == "latin-1"
    assert chunks[0]["text"] == "é"


def test_a_binary_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a text file"):
        _chunks(_write(tmp_path, b"<p>x</p>\x00\x01"))


def test_the_size_limit_applies(tmp_path):
    path = _write(tmp_path, b"<p>" + b"x" * 100 + b"</p>")
    with pytest.raises(ValueError, match="max_file_bytes"):
        _chunks(path, max_file_bytes=10)


def test_the_default_size_limit_is_lower_than_for_text():
    assert HTMLReader.__dataclass_fields__["max_file_bytes"].default == 50 * 1024**2


def test_a_page_without_text_warns_once(tmp_path):
    path = _write(tmp_path, b"<html><body><script>x()</script><div></div></body></html>")
    with mock.patch.object(_web, "logger") as logger:
        assert _chunks(path) == []
    logger.warning.assert_called_once()
    assert "no extractable text" in logger.warning.call_args.args[0]


def test_documents_carry_the_encoding(tmp_path):
    path = _write(tmp_path, b"<p>" + b"word " * 20 + b"</p>")
    docs = list(HTMLReader(input_path=path).get_documents())
    assert docs and docs[0].source_type == SourceType.WEB
    assert docs[0].metadata.get("encoding") == "utf-8"


def test_missing_beautifulsoup_names_the_reader(tmp_path):
    path = _write(tmp_path, _PAGE.encode())
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "bs4":
            raise ImportError("no bs4")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", refuse):
        with pytest.raises(ImportError, match="HTMLReader"):
            _chunks(path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'<META CHARSET=Shift_JIS>', "Shift_JIS"),
        (b"<meta charset='utf-8'>", "utf-8"),
        (b" " * 5000 + b'<meta charset="koi8-r">', None),  # past the prescan
        (b"<p>charset=koi8-r</p>", None),  # not in a meta tag
    ],
)
def test_declared_charset(raw, expected):
    assert _declared_charset(raw) == expected
