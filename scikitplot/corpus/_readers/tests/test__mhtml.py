# scikitplot/corpus/_readers/tests/test__mhtml.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :class:`~scikitplot.corpus._readers._mhtml.MHTMLReader` (CX-50).

Messages are written here the way a browser saves a page: a
``multipart/related`` message with a quoted-printable page, base64
resources and frame parts.
"""

from __future__ import annotations

import base64
import quopri
from pathlib import Path
from unittest import mock

import pytest

from ... import _readers  # noqa: F401  (registers the readers)
from ..._base import DocumentReader
from ..._schema import SourceType
from .. import _mhtml
from .._mhtml import MHTMLReader, _part_codec

pytest.importorskip("bs4")

_B = "----MultipartBoundary--abc123----"


def _part(body: bytes, *, ctype: str, location: str, cid: str = "", qp: bool = True):
    headers = [f"Content-Type: {ctype}", f"Content-Location: {location}"]
    if cid:
        headers.append(f"Content-ID: <{cid}>")
    if qp:
        headers.append("Content-Transfer-Encoding: quoted-printable")
        payload = quopri.encodestring(body)
    else:
        headers.append("Content-Transfer-Encoding: base64")
        payload = base64.encodebytes(body)
    return ("\r\n".join(headers) + "\r\n\r\n").encode() + payload


def _message(parts: list[bytes], *, start: str = "", snapshot: str = "") -> bytes:
    head = ["From: <Saved by Blink>"]
    if snapshot:
        head.append(f"Snapshot-Content-Location: {snapshot}")
    params = f'type="text/html";\r\n\tboundary="{_B}"'
    if start:
        params += f';\r\n\tstart="<{start}>"'
    head += ["MIME-Version: 1.0", f"Content-Type: multipart/related;\r\n\t{params}"]
    out = ("\r\n".join(head) + "\r\n\r\n").encode()
    for p in parts:
        out += f"--{_B}\r\n".encode() + p + b"\r\n"
    return out + f"--{_B}--\r\n".encode()


_PAGE = (
    "<html><head><title>Saved page</title><script>hidden()</script></head>"
    "<body><h1>Heading</h1><p>Caf\u00e9 paragraph with an = sign and a long line "
    + "word " * 30
    + "</p></body></html>"
).encode()
_FRAME = b"<html><body><p>Frame text</p></body></html>"


def _write(tmp_path: Path, data: bytes, name: str = "page.mhtml") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _saved(tmp_path: Path, **kwargs) -> Path:
    parts = [
        _part(_FRAME, ctype="text/html", location="cid:frame-1", cid="frame-1"),
        _part(b"\x89PNG\r\n", ctype="image/png", location="https://e.x/a.png", qp=False),
        _part(
            _PAGE,
            ctype='text/html; charset="utf-8"',
            location="https://example.com/article",
            cid="root",
        ),
    ]
    return _write(tmp_path, _message(parts, start="root", **kwargs))


def _chunks(path: Path, **kwargs) -> list[dict]:
    return list(MHTMLReader(input_path=path, **kwargs).get_raw_chunks())


@pytest.mark.parametrize("name", ["page.mhtml", "page.mht", "PAGE.MHT"])
def test_extensions_create_the_reader(tmp_path, name):
    path = _write(tmp_path, _message([_part(_FRAME, ctype="text/html", location="a")]), name)
    assert type(DocumentReader.create(path)) is MHTMLReader


def test_the_root_part_comes_first_then_frames(tmp_path):
    chunks = _chunks(_saved(tmp_path, snapshot="https://example.com/article"))
    texts = [c["text"] for c in chunks]
    assert texts[:2] == ["Saved page", "Heading"]
    assert texts[-1] == "Frame text"
    assert "Caf\u00e9 paragraph with an = sign" in texts[2]  # quoted-printable decoded
    assert [c["part_index"] for c in chunks] == [0, 0, 0, 1]
    assert chunks[0]["mhtml_part"] == "https://example.com/article"
    assert chunks[-1]["mhtml_part"] == "cid:frame-1"
    assert {c["url"] for c in chunks} == {"https://example.com/article"}
    assert {c["source_type"] for c in chunks} == {SourceType.WEB.value}
    assert not any("hidden" in t for t in texts)


def test_without_a_start_parameter_parts_keep_their_order(tmp_path):
    parts = [
        _part(b"<p>one</p>", ctype="text/html", location="a"),
        _part(b"<p>two</p>", ctype="text/html", location="b"),
    ]
    chunks = _chunks(_write(tmp_path, _message(parts)))
    assert [c["text"] for c in chunks] == ["one", "two"]


def test_the_url_falls_back_to_the_root_location_and_must_be_http(tmp_path):
    chunks = _chunks(_saved(tmp_path))
    assert chunks[0]["url"] == "https://example.com/article"
    parts = [_part(b"<p>x</p>", ctype="text/html", location="file:///C:/x.htm")]
    assert all("url" not in c for c in _chunks(_write(tmp_path, _message(parts), "b.mht")))


def test_base64_html_parts_are_decoded(tmp_path):
    parts = [_part(b"<p>encoded</p>", ctype="text/html", location="a", qp=False)]
    assert [c["text"] for c in _chunks(_write(tmp_path, _message(parts)))] == ["encoded"]


def test_a_single_part_page_is_read(tmp_path):
    data = b"MIME-Version: 1.0\r\nContent-Type: text/html\r\n\r\n<p>alone</p>\r\n"
    assert [c["text"] for c in _chunks(_write(tmp_path, data))] == ["alone"]


@pytest.mark.parametrize(
    ("body", "ctype", "codec", "text"),
    [
        (b"<p>caf\xe9</p>", 'text/html; charset="iso-8859-1"', "iso-8859-1", "caf\u00e9"),
        ("<p>caf\u00e9</p>".encode(), "text/html; charset=ascii", "utf-8", "caf\u00e9"),
        (b'<meta charset="koi8-r"><p>\xf0\xd2\xc9</p>', "text/html", "koi8-r", "\u041f\u0440\u0438"),
        (b"<p>caf\xe9</p>", "text/html; charset=no-such", "cp1252", "caf\u00e9"),
    ],
)
def test_part_encoding_order(tmp_path, body, ctype, codec, text):
    parts = [_part(body, ctype=ctype, location="a", qp=False)]
    [chunk] = _chunks(_write(tmp_path, _message(parts)))
    assert (chunk["encoding"], chunk["text"]) == (codec, text)


def test_an_explicit_encoding_applies(tmp_path):
    parts = [_part(b"<p>\xe9</p>", ctype="text/html; charset=utf-8", location="a", qp=False)]
    [chunk] = _chunks(_write(tmp_path, _message(parts)), encoding="latin-1")
    assert (chunk["encoding"], chunk["text"]) == ("latin-1", "\u00e9")


def test_a_message_without_html_is_refused(tmp_path):
    parts = [_part(b"\x89PNG", ctype="image/png", location="a", qp=False)]
    with pytest.raises(ValueError, match="no HTML part"):
        _chunks(_write(tmp_path, _message(parts)))


def test_deep_nesting_is_refused(tmp_path):
    data = b"MIME-Version: 1.0\r\n"
    depth = 60
    for i in range(depth):
        data += f'Content-Type: multipart/mixed; boundary="b{i}"\r\n\r\n--b{i}\r\n'.encode()
    data += b"Content-Type: text/html\r\n\r\n<p>deep</p>\r\n"
    for i in reversed(range(depth)):
        data += f"--b{i}--\r\n".encode()
    with pytest.raises(ValueError, match="nesting deeper"):
        _chunks(_write(tmp_path, data))


def test_limits(tmp_path):
    path = _saved(tmp_path)
    with pytest.raises(ValueError, match="max_parts=2"):
        _chunks(path, max_parts=2)
    with pytest.raises(ValueError, match="max_file_bytes"):
        _chunks(path, max_file_bytes=100)
    reader = MHTMLReader(input_path=path, max_part_bytes=100)
    assert [c["text"] for c in reader.get_raw_chunks()] == ["Frame text"]
    assert reader.coverage.skipped == {"too-large": 1}


@pytest.mark.parametrize(
    "kwargs", [{"max_parts": 0}, {"max_part_bytes": True}, {"encoding": "nope"}]
)
def test_bad_options_are_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        MHTMLReader(input_path=_saved(tmp_path), **kwargs)


def test_a_page_without_text_warns_once(tmp_path):
    parts = [_part(b"<script>x()</script>", ctype="text/html", location="a")]
    reader = MHTMLReader(input_path=_write(tmp_path, _message(parts)))
    with mock.patch.object(_mhtml, "logger") as logger:
        assert list(reader.get_raw_chunks()) == []
    logger.warning.assert_called_once()


def test_documents_are_built(tmp_path):
    docs = list(MHTMLReader(input_path=_saved(tmp_path)).get_documents())
    assert docs and docs[0].url == "https://example.com/article"


def test_part_codec_bom_wins():
    assert _part_codec(b"\xef\xbb\xbf<p>x</p>", "koi8-r") == "utf-8-sig"
