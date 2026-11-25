# scikitplot/corpus/_readers/tests/test__chm.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the CHM container, the LZX decoder and :class:`CHMReader`.

Archives are built in the test with ``_chm_build`` (a minimal LZX encoder),
so no binary fixture is needed and every block type is covered.
"""

from __future__ import annotations

import random
import struct
from pathlib import Path
from unittest import mock

import pytest

from ... import _readers  # noqa: F401  (registers the readers)
from ..._base import DocumentReader
from ..._schema import SourceType
from .. import _chm
from .._chm import CHMReader, _ansi_code_page, _toc
from .._chm_format import FRAME, ChmArchive, _decode_table, _LzxDecoder
from ._chm_build import ALIGNED, UNCOMPRESSED, VERBATIM, build_chm, compress

pytest.importorskip("bs4")

_RNG = random.Random(20261125)
_WORDS = [
    bytes(_RNG.choice(b"abcdefghij") for _ in range(_RNG.randint(2, 9)))
    for _ in range(400)
]


def _text(n: int) -> bytes:
    return b" ".join(_RNG.choice(_WORDS) for _ in range(n))


def _page(title: str, body: bytes, head: bytes = b"") -> bytes:
    return (
        b"<html><head>" + head + b"<title>" + title.encode() + b"</title>"
        b"<script>hidden()</script></head><body><p>" + body + b"</p></body></html>"
    )


_HHC = (
    b'<HTML><BODY><UL><LI><OBJECT type="text/sitemap">'
    b'<param name="Name" value="Second &amp; last"><param name="Local" value="b.htm">'
    b'</OBJECT><LI><OBJECT type="text/sitemap"><param name="Name" value="First">'
    b'<param name="Local" value="html\\a.htm#top"></OBJECT></UL></BODY></HTML>'
)


def _write(tmp_path: Path, data: bytes, name: str = "book.chm") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _all(archive: ChmArchive) -> dict[str, bytes]:
    wanted = [e for e in archive.entries if e.section == 1]
    return {e.name: d for e, d in archive.read_many(wanted, max_decoded_bytes=1 << 30)}


# ---------------------------------------------------------------------------
# Container and decoder
# ---------------------------------------------------------------------------


_FILES = {f"/p{i}.htm": _page(f"p{i}", _text(_RNG.randint(5, 9000))) for i in range(10)}


@pytest.mark.parametrize("kind", [VERBATIM, ALIGNED, UNCOMPRESSED])
@pytest.mark.parametrize(
    ("frames_per_block", "reset_frames"), [(1, 2), (3, 4), (1, 1), (5, 8)]
)
def test_every_block_type_round_trips(kind, frames_per_block, reset_frames):
    data = build_chm(
        _FILES, kind=kind, frames_per_block=frames_per_block, reset_frames=reset_frames
    )
    archive = ChmArchive(data, "t.chm")
    assert sum(map(len, _FILES.values())) > 4 * FRAME  # several frames and resets
    assert _all(archive) == _FILES


@pytest.mark.parametrize("name", ["/p0.htm", "/p5.htm", "/p9.htm"])
def test_one_file_is_read_from_its_reset_point(name):
    archive = ChmArchive(build_chm(_FILES, reset_frames=2), "t.chm")
    entry = archive.entry(name)
    calls = []
    real = _LzxDecoder.frame

    def counting(self, *args):
        calls.append(args[1])
        return real(self, *args)

    with mock.patch.object(_LzxDecoder, "frame", counting):
        [(_, data)] = archive.read_many([entry], max_decoded_bytes=1 << 30)
    assert data == _FILES[name]
    last = (entry.offset + entry.length - 1) // FRAME
    first = entry.offset // FRAME
    assert len(calls) == last - (first - first % 2) + 1


def test_files_sharing_a_frame_decode_it_once():
    small = {f"/s{i}.htm": b"<p>%d</p>" % i for i in range(50)}
    archive = ChmArchive(build_chm(small), "t.chm")
    with mock.patch.object(_LzxDecoder, "frame", side_effect=_LzxDecoder.frame, autospec=True) as frame:
        assert _all(archive) == small
    assert frame.call_count == 1


def test_lookup_is_case_insensitive():
    archive = ChmArchive(build_chm({"/Index.HTM": b"<p>x</p>"}), "t.chm")
    assert archive.entry("/index.htm").name == "/Index.HTM"
    assert archive.entry("/missing.htm") is None


def test_stored_files_are_read_as_is():
    archive = ChmArchive(
        build_chm({"/a.htm": b"<p>a</p>"}, stored={"/#SYSTEM": b"\x01\x02"}), "t.chm"
    )
    entry = archive.entry("/#SYSTEM")
    assert entry.section == 0
    assert dict(archive.read_many([entry], max_decoded_bytes=1)) == {entry: b"\x01\x02"}


def test_x86_call_translation_is_undone():
    # In an uncompressed frame E8 operands are stored as absolute targets;
    # the decoder turns those in range back into relative ones.
    size = 1 << 20
    frame = bytearray(b"\x90" * 64)
    struct.pack_into("<Bi", frame, 0, 0xE8, 100)  # at 0: 100 -> 100 - 0
    struct.pack_into("<Bi", frame, 10, 0xE8, 105)  # at 10: 105 -> 95
    struct.pack_into("<Bi", frame, 20, 0xE8, -5)  # at 20: -5 -> -5 + size
    struct.pack_into("<Bi", frame, 30, 0xE8, size)  # out of range: kept
    struct.pack_into("<Bi", frame, 58, 0xE8, 7)  # in the last 10 bytes: kept
    archive = ChmArchive(
        build_chm({"/x.bin": bytes(frame)}, kind=UNCOMPRESSED, intel_size=size), "t.chm"
    )
    out = _all(archive)["/x.bin"]
    values = [struct.unpack_from("<i", out, at + 1)[0] for at in (0, 10, 20, 30, 58)]
    assert values == [100, 95, size - 5, size, 7]


def test_without_the_e8_header_bytes_are_unchanged():
    frame = struct.pack("<Bi", 0xE8, 105) + b"\x90" * 40
    archive = ChmArchive(build_chm({"/x.bin": frame}, kind=UNCOMPRESSED), "t.chm")
    assert _all(archive)["/x.bin"] == frame


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: b"XXXX" + d[4:], "not a CHM file"),
        (lambda d: d[:0x50], "not a CHM file"),
        (lambda d: d[:4] + struct.pack("<I", 9) + d[8:], "unsupported ITSF version"),
        (lambda d: d[:0x60 + 0x10] + struct.pack("<I", 1000) + d[0x60 + 0x14 :], "chunk size"),
        (lambda d: d[: 0x60 + 0x54 + 0x10] + struct.pack("<i", 0) + d[0x60 + 0x54 + 0x14 :], "loop"),
        (lambda d: d[: 0x60 + 0x54] + b"QQQQ" + d[0x60 + 0x58 :], "not a PMGL"),
        (lambda d: d[:0x200], "truncated"),
    ],
)
def test_malformed_containers_raise_value_error(mutate, message):
    data = build_chm({"/a.htm": _page("a", _text(50))})
    with pytest.raises(ValueError, match=message):
        ChmArchive(mutate(data), "t.chm")


def test_corrupt_compressed_bytes_raise_only_value_error():
    data = bytearray(build_chm(_FILES))
    archive = ChmArchive(bytes(data), "t.chm")
    content = archive.entry("::DataSpace/Storage/MSCompressed/Content")
    start = archive._content + content.offset
    rng = random.Random(7)
    outcomes = set()
    for _ in range(60):
        broken = bytearray(data)
        for _ in range(rng.randint(1, 8)):
            broken[start + rng.randrange(content.length)] ^= 1 << rng.randrange(8)
        try:
            _all(ChmArchive(bytes(broken), "t.chm"))
            outcomes.add("read")
        except ValueError:
            outcomes.add("refused")
    assert "refused" in outcomes  # and nothing else escaped


def test_the_decoding_budget_stops_a_read():
    archive = ChmArchive(build_chm(_FILES), "t.chm")
    with pytest.raises(ValueError, match="decoding needs more than 65,536 bytes"):
        _all_budget = list(
            archive.read_many(
                [e for e in archive.entries if e.section == 1],
                max_decoded_bytes=2 * FRAME,
            )
        )
        del _all_budget


def test_an_entry_past_the_section_is_refused():
    data = build_chm({"/a.htm": b"<p>a</p>"})
    archive = ChmArchive(data, "t.chm")
    bad = type(archive.entries[0])("/a.htm", 1, 0, 10_000)
    with pytest.raises(ValueError, match="past the compressed section"):
        list(archive.read_many([bad], max_decoded_bytes=1 << 20))


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [([0, 0], 0), ([1, 1], 1), ([2, 2, 1], 2)],
)
def test_decode_table_shapes(lengths, expected):
    _, bits = _decode_table(lengths, "t")
    assert bits == expected


def test_decode_table_refuses_an_over_subscribed_code():
    with pytest.raises(ValueError, match="over-subscribe"):
        _decode_table([1, 1, 1], "t")


def test_compress_offsets_start_each_frame():
    content, offsets = compress(b"x" * (3 * FRAME + 5))
    assert len(offsets) == 4 and offsets[0] == 0 and offsets == sorted(offsets)
    assert offsets[-1] < len(content)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _book(**kwargs) -> bytes:
    files = {
        "/toc.hhc": _HHC,
        "/html/a.htm": _page("Alpha", b"first page words " * 4),
        "/z.htm": _page("Zulu", b"unlisted page words " * 4),
        "/b.htm": _page("Beta", b"second page words " * 4),
        "/img.gif": b"GIF89a",
    }
    return build_chm(files, **kwargs)


def test_chm_creates_the_chm_reader(tmp_path):
    assert type(DocumentReader.create(_write(tmp_path, _book()))) is CHMReader


def test_pages_follow_the_table_of_contents(tmp_path):
    reader = CHMReader(input_path=_write(tmp_path, _book()))
    chunks = list(reader.get_raw_chunks())
    pages = [c["chm_page"] for c in chunks if c["html_tag"] == "title"]
    assert pages == ["/b.htm", "/html/a.htm", "/z.htm"]  # contents, then unlisted
    assert [c["page_index"] for c in chunks if c["html_tag"] == "title"] == [0, 1, 2]
    titles = {c["chm_page"]: c.get("toc_title") for c in chunks}
    assert titles == {"/b.htm": "Second & last", "/html/a.htm": "First", "/z.htm": None}
    assert {c["source_type"] for c in chunks} == {SourceType.DOCUMENTATION.value}
    assert all(c["input_path"] == "book.chm" + c["chm_page"] for c in chunks)
    assert not any("hidden" in c["text"] for c in chunks)
    assert reader.coverage.summary() == "3 read, 0 empty of 3 page(s)"


def test_documents_are_built(tmp_path):
    docs = list(CHMReader(input_path=_write(tmp_path, _book())).get_documents())
    assert docs and all(d.source_type == SourceType.DOCUMENTATION for d in docs)


def test_without_contents_pages_follow_archive_order(tmp_path):
    data = build_chm({"/b.htm": _page("B", b"b " * 10), "/a.htm": _page("A", b"a " * 10)})
    chunks = list(CHMReader(input_path=_write(tmp_path, data)).get_raw_chunks())
    assert [c["chm_page"] for c in chunks if c["html_tag"] == "title"] == ["/b.htm", "/a.htm"]
    assert all("toc_title" not in c for c in chunks)


@pytest.mark.parametrize(
    ("raw", "lcid", "codec", "text"),
    [
        ("<p>café</p>".encode(), 0x0409, "utf-8", "café"),
        (b'<meta charset="koi8-r"><p>\xf0\xd2\xc9</p>', 0x0409, "koi8-r", "При"),
        (b"<p>\xcf\xf0\xe8</p>", 0x0419, "cp1251", "При"),
        (b"<p>caf\xe9</p>", 0x0409, "cp1252", "café"),
        (b"<p>\x81\x8d</p>", 0x0409, "latin-1", "\x81\x8d"),
    ],
)
def test_page_encoding_order(tmp_path, raw, lcid, codec, text):
    path = _write(tmp_path, build_chm({"/a.htm": raw}, lcid=lcid))
    [chunk] = CHMReader(input_path=path).get_raw_chunks()
    assert (chunk["encoding"], chunk["text"]) == (codec, text)


def test_an_explicit_encoding_applies_to_every_page(tmp_path):
    path = _write(tmp_path, build_chm({"/a.htm": b"<p>\xe9</p>"}))
    [chunk] = CHMReader(input_path=path, encoding="latin-1").get_raw_chunks()
    assert chunk["encoding"] == "latin-1" and chunk["text"] == "é"


def test_large_and_surplus_pages_are_skipped_and_counted(tmp_path):
    files = {f"/p{i}.htm": _page(f"p{i}", b"w " * 30) for i in range(4)}
    files["/big.htm"] = _page("big", b"w " * 5000)
    reader = CHMReader(
        input_path=_write(tmp_path, build_chm(files)), max_pages=2, max_page_bytes=1000
    )
    pages = {c["chm_page"] for c in reader.get_raw_chunks()}
    assert pages == {"/p0.htm", "/p1.htm"}
    assert reader.coverage.skipped == {"limit": 2, "too-large": 1}


def test_the_decoding_limit_names_its_option(tmp_path):
    path = _write(tmp_path, build_chm(_FILES))
    reader = CHMReader(input_path=path, max_total_bytes=FRAME, max_text_chars_per_byte=None)
    with pytest.raises(ValueError, match="max_total_bytes=32,768"):
        list(reader.get_raw_chunks())


def test_the_ratio_limit_follows_the_file_size():
    reader = CHMReader.__new__(CHMReader)
    reader.max_total_bytes = 1 << 40
    reader.max_text_chars_per_byte = 50
    assert reader._decode_limit(1000)[0] == 4 * 1024 * 1024  # the 1 Mi floor
    assert reader._decode_limit(1_000_000)[0] == 4 * 50 * 1_000_000


@pytest.mark.parametrize(
    ("data", "kwargs", "message"),
    [
        (b"", {}, "is empty"),
        (b"not a help file at all, just bytes" * 4, {}, "not a CHM file"),
        (b"ITSF" + b"\x00" * 200, {"max_file_bytes": 10}, "max_file_bytes"),
    ],
)
def test_files_that_are_refused(tmp_path, data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        list(CHMReader(input_path=_write(tmp_path, data), **kwargs).get_raw_chunks())


@pytest.mark.parametrize(
    "kwargs",
    [{"max_pages": 0}, {"max_page_bytes": -1}, {"max_total_bytes": True}, {"encoding": "nope"}],
)
def test_bad_options_are_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        CHMReader(input_path=_write(tmp_path, _book()), **kwargs)


def test_a_book_without_text_warns_once(tmp_path):
    path = _write(tmp_path, build_chm({"/a.htm": b"<script>x()</script>", "/b.htm": b"<div></div>"}))
    reader = CHMReader(input_path=path)
    with mock.patch.object(_chm, "logger") as logger:
        assert list(reader.get_raw_chunks()) == []
    logger.warning.assert_called_once()
    assert reader.coverage.empty == 2


def test_an_unreadable_contents_file_falls_back_to_archive_order(tmp_path):
    path = _write(tmp_path, _book())
    reader = CHMReader(input_path=path)
    real = ChmArchive.read_many

    def failing(self, entries, **kwargs):
        if entries and entries[0].name.endswith(".hhc"):
            raise ValueError("broken contents")
        return real(self, entries, **kwargs)

    with mock.patch.object(ChmArchive, "read_many", failing), mock.patch.object(
        _chm, "logger"
    ) as logger:
        chunks = list(reader.get_raw_chunks())
    assert [c["chm_page"] for c in chunks if c["html_tag"] == "title"] == [
        "/html/a.htm",
        "/z.htm",
        "/b.htm",
    ]
    assert "table of contents" in logger.warning.call_args.args[0]


def test_the_file_is_closed_after_a_read(tmp_path):
    # Strict mode turns an unclosed file or mmap into an error.
    reader = CHMReader(input_path=_write(tmp_path, _book()))
    list(reader.get_raw_chunks())
    (tmp_path / "book.chm").unlink()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "base", "expected"),
    [
        (
            '<object type="text/sitemap"><param name="Local" value="mk:@MSITStore:'
            'x.chm::/Sub/Page.htm"></object>',
            "/",
            [("/sub/page.htm", "")],
        ),
        (
            '<object><param name="Local" value="../a.htm"></object>'
            '<object><param name="Local" value="../A.htm#x"></object>',
            "/doc",
            [("/a.htm", "")],
        ),
        ('<object><param name="Local" value="https://example.com/a.htm"></object>', "/", []),
        ('<object><param name="Name" value="no link"></object>', "/", []),
    ],
)
def test_toc(text, base, expected):
    assert _toc(text, base) == expected


@pytest.mark.parametrize(
    ("lcid", "codec"),
    [(0x0411, "cp932"), (0x0804, "cp936"), (0x0C04, "cp950"), (0x0809, None), (0, None)],
)
def test_ansi_code_page(lcid, codec):
    assert _ansi_code_page(lcid) == codec


def test_object_blocks_pair_tags_in_one_pass():
    from .._chm import _object_blocks

    text = "<OBJECT type=x>a</OBJECT> junk </object> <object>b<object>c</object>"
    assert _object_blocks(text) == ["a", "c"]
    assert _object_blocks("<object>never closed") == []


def test_unclosed_object_tags_cost_linear_time():
    import time

    from .._chm import _object_blocks

    start = time.perf_counter()
    assert _object_blocks("<object>" * 100_000) == []
    # "<object[^>]*>(.*?)</object>" rescanned to the end from every tag.
    assert time.perf_counter() - start < 1.0


def test_toc_with_unclosed_objects_still_reads_the_closed_ones():
    text = '<object><object><param name="Local" value="a.htm"></object>'
    assert _toc(text, "/") == [("/a.htm", "")]
