# corpus/_readers/tests/test__tar.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
TarReader: tar archives and single compressed files
===================================================

Format by content, natural order, regular files only, last copy of a name,
the extraction budget (including the data a compressed scan decompresses),
nesting with ZIP, and damaged input. Archives are built in each test; no
test reaches the network.
"""

from __future__ import annotations

import bz2
import contextlib
import gzip
import io
import lzma
import tarfile
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from ... import _readers  # noqa: F401  (registers every reader)
from ..._base import DocumentReader
from .. import _tar as tar_module
from .. import _zip as zip_module
from .._tar import TarReader, _compression, _inner_name
from .._zip import ArchiveBudgetError, ArchiveNestingError, ZipReader, _archive_ctx

MB = 1024 * 1024


@pytest.fixture(autouse=True)
def _reset_ctx():
    token = _archive_ctx.set(None)
    yield
    _archive_ctx.reset(token)


def _add(tf, name, data=b"", kind=tarfile.REGTYPE, link=""):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = link
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    tf.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)


def _make(path, members, mode="w"):
    with tarfile.open(path, mode) as tf:
        for member in members:
            _add(tf, *member) if isinstance(member, tuple) else _add(tf, member)
    return path


def _paths(reader):
    return [c["input_path"] for c in reader.get_raw_chunks()]


# ---------------------------------------------------------------------------
# Registration and format detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["a.tar", "a.tgz", "a.tbz", "a.tbz2", "a.txz", "a.gz", "a.bz2", "a.xz"]
)
def test_every_extension_dispatches_to_tar_reader(tmp_path, name):
    reader = DocumentReader.create(tmp_path / name)
    assert type(reader) is TarReader
    assert isinstance(reader, ZipReader)  # same limits and nesting


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (gzip.compress(b"x"), "gz"),
        (bz2.compress(b"x"), "bz2"),
        (lzma.compress(b"x"), "xz"),
        (b"plain", None),
        (b"", None),
    ],
)
def test_compression_is_decided_by_leading_bytes(tmp_path, data, expected):
    path = tmp_path / "f.dat"
    path.write_bytes(data)
    assert _compression(path) == expected


def test_inner_name_drops_only_the_compression_suffix():
    assert _inner_name("notes.txt.gz") == "notes.txt"
    assert _inner_name("a.b.csv.xz") == "a.b.csv"


# ---------------------------------------------------------------------------
# Reading archives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("a.tar", "w"),
        ("a.tar.gz", "w:gz"),
        ("a.tgz", "w:gz"),
        ("a.tar.bz2", "w:bz2"),
        ("a.tar.xz", "w:xz"),
    ],
)
def test_members_are_read_in_natural_order(tmp_path, name, mode):
    path = _make(
        tmp_path / name,
        [("p10.txt", b"Page ten."), ("p2.txt", b"Page two."), ("p1.txt", b"Page one.")],
        mode,
    )
    reader = TarReader(input_path=path)
    assert _paths(reader) == [f"{name}/p1.txt", f"{name}/p2.txt", f"{name}/p10.txt"]
    assert reader.coverage.read == 3


def test_a_gz_holding_a_tar_is_an_archive_whatever_its_name(tmp_path):
    path = _make(tmp_path / "bundle.gz", [("doc.txt", b"Inside a tar.")], "w:gz")
    assert _paths(TarReader(input_path=path)) == ["bundle.gz/doc.txt"]


def test_documents_carry_the_member_label(tmp_path):
    path = _make(tmp_path / "a.tar.gz", [("dir/doc.txt", b"Some text here.")], "w:gz")
    docs = list(DocumentReader.create(path).get_documents())
    assert [d.input_path for d in docs] == ["a.tar.gz/dir/doc.txt"]


def test_reader_kwargs_reach_the_member_reader(tmp_path):
    path = _make(tmp_path / "a.tar", [("t.csv", b"a;b\n1;2\n")])
    with mock.patch.object(DocumentReader, "_create_one", wraps=DocumentReader._create_one) as made:
        list(TarReader(input_path=path, reader_kwargs={"csv": {"delimiter": ";"}}).get_raw_chunks())
    assert made.call_args.kwargs["delimiter"] == ";"


def test_an_empty_file_has_no_members(tmp_path):
    path = tmp_path / "a.tar"
    path.write_bytes(b"")
    reader = TarReader(input_path=path)
    assert _paths(reader) == []
    assert reader.coverage.summary().startswith("0 read")


def test_the_temporary_directory_is_removed(tmp_path):
    path = _make(tmp_path / "a.tar", [("doc.txt", b"Text.")])
    made = []
    real = tar_module.tempfile.mkdtemp

    def spy(**kwargs):
        made.append(real(**kwargs))
        return made[-1]

    with mock.patch.object(tar_module.tempfile, "mkdtemp", side_effect=spy):
        list(TarReader(input_path=path).get_raw_chunks())
    assert made
    assert not any(__import__("os").path.exists(p) for p in made)


# ---------------------------------------------------------------------------
# Single compressed files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [(".gz", gzip.compress), (".bz2", bz2.compress), (".xz", lzma.compress)],
)
def test_a_single_compressed_file_is_read_under_its_own_name(tmp_path, suffix, compress):
    path = tmp_path / f"notes.txt{suffix}"
    path.write_bytes(compress(b"Hello from a compressed file."))
    chunks = list(TarReader(input_path=path).get_raw_chunks())
    assert [c["input_path"] for c in chunks] == [f"notes.txt{suffix}/notes.txt"]
    assert "Hello from a compressed file." in chunks[0]["text"]


def test_a_single_file_of_an_unsupported_type_is_skipped_or_refused(tmp_path):
    path = tmp_path / "blob.bin.gz"
    path.write_bytes(gzip.compress(b"\0\1"))
    reader = TarReader(input_path=path)
    assert _paths(reader) == []
    assert reader.coverage.skipped == {"unsupported": 1}
    with pytest.raises(ValueError, match="unsupported extension"):
        list(TarReader(input_path=path, skip_unsupported=False).get_raw_chunks())


def test_a_single_file_is_bounded_by_the_budget(tmp_path):
    path = tmp_path / "big.txt.gz"
    path.write_bytes(gzip.compress(b"a" * (2 * MB)))
    with pytest.raises(ArchiveBudgetError, match="max_total_bytes"):
        list(
            TarReader(
                input_path=path, max_total_bytes=MB, max_text_chars_per_byte=None
            ).get_raw_chunks()
        )


# ---------------------------------------------------------------------------
# What is not read
# ---------------------------------------------------------------------------


def test_links_and_devices_are_skipped(tmp_path):
    path = _make(
        tmp_path / "a.tar",
        [
            ("doc.txt", b"Real text."),
            ("soft.txt", b"", tarfile.SYMTYPE, "/etc/passwd"),
            ("hard.txt", b"", tarfile.LNKTYPE, "doc.txt"),
            ("dev.txt", b"", tarfile.CHRTYPE),
            ("pipe.txt", b"", tarfile.FIFOTYPE),
        ],
    )
    reader = TarReader(input_path=path)
    assert _paths(reader) == ["a.tar/doc.txt"]
    assert reader.coverage.skipped == {"not-a-file": 4}


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("/abs/escape.txt", "unsafe-path"),
        # A ".." part starts with "." and is skipped as hidden, before the
        # path check, the same rule as in ZipReader.
        ("../escape.txt", "hidden"),
        ("a/../../escape.txt", "hidden"),
    ],
)
def test_paths_outside_the_archive_are_skipped(tmp_path, name, reason):
    path = _make(tmp_path / "a.tar", [(name, b"Escaping text.")])
    reader = TarReader(input_path=path)
    assert _paths(reader) == []
    assert reader.coverage.skipped == {reason: 1}
    assert not (tmp_path / "escape.txt").exists()


def test_hidden_members_are_skipped(tmp_path):
    path = _make(tmp_path / "a.tar", [(".secret.txt", b"x"), ("__MACOSX/a.txt", b"x")])
    reader = TarReader(input_path=path)
    assert _paths(reader) == []
    assert reader.coverage.skipped == {"hidden": 2}


def test_the_last_copy_of_a_name_wins_ignoring_case(tmp_path):
    path = _make(
        tmp_path / "a.tar",
        [("doc.txt", b"First copy."), ("DOC.TXT", b"Second copy."), ("doc.txt", b"Third copy.")],
    )
    reader = TarReader(input_path=path)
    chunks = list(reader.get_raw_chunks())
    assert len(chunks) == 1 and "Third copy." in chunks[0]["text"]
    assert reader.coverage.skipped == {"duplicate": 2}


def test_max_files_counts_every_entry_directories_too(tmp_path):
    files = _make(tmp_path / "a.tar", [(f"f{i}.txt", b"x") for i in range(4)])
    with pytest.raises(ValueError, match="more than 3 entries"):
        list(TarReader(input_path=files, max_files=3).get_raw_chunks())
    dirs = _make(
        tmp_path / "b.tar", [(f"d{i}", b"", tarfile.DIRTYPE) for i in range(4)]
    )
    with pytest.raises(ValueError, match="a tar counts directories too"):
        list(TarReader(input_path=dirs, max_files=3).get_raw_chunks())


# ---------------------------------------------------------------------------
# Extraction budget
# ---------------------------------------------------------------------------


def test_a_member_larger_than_the_budget_is_refused(tmp_path):
    path = _make(tmp_path / "a.tar", [("big.txt", b"a" * (2 * MB))])
    with pytest.raises(ArchiveBudgetError, match="declares"):
        list(
            TarReader(
                input_path=path, max_total_bytes=MB, max_text_chars_per_byte=None
            ).get_raw_chunks()
        )


def test_a_compressed_scan_counts_skipped_members(tmp_path):
    members = [("blob.bin", b"\0" * (2 * MB)), ("doc.txt", b"Small text.")]
    options = {"max_total_bytes": MB, "max_text_chars_per_byte": None}
    # Plain tar: the unsupported member is passed over by seeking.
    plain = _make(tmp_path / "a.tar", members)
    assert _paths(TarReader(input_path=plain, **options)) == ["a.tar/doc.txt"]
    # Compressed: reaching doc.txt decompresses the blob first.
    packed = _make(tmp_path / "a.tar.gz", members, "w:gz")
    with pytest.raises(ArchiveBudgetError, match="decompresses"):
        list(TarReader(input_path=packed, **options).get_raw_chunks())


def test_truncate_keeps_the_members_extracted_before_the_limit(tmp_path):
    path = _make(
        tmp_path / "a.tar",
        [("a.txt", b"a" * 600_000), ("b.txt", b"b" * 600_000), ("c.txt", b"Small.")],
    )
    reader = TarReader(
        input_path=path,
        max_total_bytes=MB,
        max_text_chars_per_byte=None,
        on_text_limit="truncate",
    )
    # Patch the logger: caplog depends on how the installed package
    # configures logging, which differs between distributions.
    with mock.patch.object(zip_module, "logger") as logger:
        got = _paths(reader)
    assert got == ["a.tar/a.txt"]  # b.txt is past the limit; c.txt after it
    assert reader.coverage.skipped == {"over-budget": 2}
    assert "not read" in logger.warning.call_args.args[0]


def test_members_are_extracted_in_one_forward_pass(tmp_path):
    path = _make(
        tmp_path / "a.tar.gz",
        [("p3.txt", b"Three."), ("p1.txt", b"One."), ("p2.txt", b"Two.")],
        "w:gz",
    )
    seen = []
    real = tarfile.TarFile.extractfile

    def spy(self, member):
        seen.append(member.name)
        return real(self, member)

    with mock.patch.object(tarfile.TarFile, "extractfile", spy):
        got = _paths(TarReader(input_path=path))
    assert seen == ["p3.txt", "p1.txt", "p2.txt"]  # stored order: no rewind
    assert got == ["a.tar.gz/p1.txt", "a.tar.gz/p2.txt", "a.tar.gz/p3.txt"]


# ---------------------------------------------------------------------------
# Nesting with ZIP
# ---------------------------------------------------------------------------


def test_a_zip_inside_a_tar_and_a_tar_inside_a_zip(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("z.txt", "Zip text.")
    outer = _make(tmp_path / "o.tar.gz", [("in.zip", inner.getvalue())], "w:gz")
    assert _paths(TarReader(input_path=outer)) == ["o.tar.gz/in.zip/z.txt"]

    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tf:
        _add(tf, "t.txt", b"Tar text.")
    zpath = tmp_path / "o.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("in.tgz", tar_bytes.getvalue())
    reader = ZipReader(input_path=zpath)
    assert _paths(reader) == ["o.zip/in.tgz/t.txt"]
    assert reader.coverage.read == 1


def test_nesting_depth_applies_across_formats(tmp_path):
    payload, name = b"Deep text.", "doc.txt"
    for i in range(4):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            _add(tf, name, payload)
        payload, name = buf.getvalue(), f"l{i}.tgz"
    path = tmp_path / name
    path.write_bytes(payload)
    assert len(_paths(TarReader(input_path=path))) == 1
    with pytest.raises(ArchiveNestingError):
        list(TarReader(input_path=path, max_depth=3).get_raw_chunks())


# ---------------------------------------------------------------------------
# Damaged and mislabelled input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("a.tar", b"not a tar archive at all", "holds no tar archive"),
        ("a.tgz", gzip.compress(b"just text"), "holds no tar archive"),
        ("a.gz", b"not compressed", "holds no tar archive"),
    ],
)
def test_input_that_is_not_what_its_name_says_is_refused(tmp_path, name, data, message):
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        list(TarReader(input_path=path).get_raw_chunks())


def test_a_damaged_stream_is_refused_with_its_name(tmp_path):
    good = _make(tmp_path / "a.tar.gz", [("doc.txt", b"x" * 50_000)], "w:gz").read_bytes()
    path = tmp_path / "cut.tar.gz"
    path.write_bytes(good[: len(good) // 2])
    with pytest.raises(ValueError, match="'cut.tar.gz' is damaged"):
        list(TarReader(input_path=path).get_raw_chunks())


def test_a_damaged_single_file_is_refused(tmp_path):
    path = tmp_path / "notes.txt.gz"
    path.write_bytes(gzip.compress(b"x" * 50_000)[:-30])
    with pytest.raises(ValueError, match="is damaged"):
        list(TarReader(input_path=path).get_raw_chunks())


def test_extraction_streams_in_bounded_memory(tmp_path):
    import tracemalloc

    big = b"line of text\n" * (32 * MB // 13)  # about 32 MiB
    path = _make(tmp_path / "a.tar.gz", [("a.txt", big)], "w:gz")
    reader = TarReader(input_path=path, max_text_chars_per_byte=None)
    # Extraction only: what the member's own reader holds is its concern.
    with mock.patch.object(TarReader, "_read_member", return_value=iter(())):
        tracemalloc.start()
        try:
            list(reader.get_raw_chunks())
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    # A fixed cost (one 1 MiB copy block and the decompressor's buffers,
    # about 4 MiB) that does not grow with the member: well under a quarter.
    assert peak < len(big) // 4, f"peak {peak:,} bytes for a {len(big):,}-byte member"


# ---------------------------------------------------------------------------
# Hostile headers: bounded before tarfile spends the memory
# ---------------------------------------------------------------------------


def _raw_header(name, kind, size):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.size = size
    return info.tobuf(format=tarfile.USTAR_FORMAT)


def _record(body):
    return body + b"\0" * (-len(body) % 512)


def _with_header(kind, body):
    """A tar whose only member is preceded by a pax or long-name header."""
    regular = tarfile.TarInfo("a.txt")
    regular.size = 5
    return (
        _raw_header("././@LongLink", kind, len(body))
        + _record(body)
        + regular.tobuf(format=tarfile.USTAR_FORMAT)
        + _record(b"hello")
        + b"\0" * 1024
    )


def _whole_header_reads():
    """Make ``tarfile`` ask for a long header in one read, as older releases do.

    Newer CPython releases read a pax or long-name header in bounded chunks
    (``tarfile._safe_read``); older ones ask for all of it at once. Both
    must be refused alike, so the test runs both on every Python.
    """
    if not hasattr(tarfile, "_safe_read"):
        return contextlib.nullcontext()  # this Python already reads it whole
    return mock.patch.object(tarfile, "_safe_read", lambda stream, size: stream.read(size))


@pytest.mark.parametrize("kind", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME])
@pytest.mark.parametrize("compress", [None, gzip.compress])
@pytest.mark.parametrize("reads", ["as-this-python", "whole"])
def test_an_oversized_header_is_refused_in_bounded_memory(
    tmp_path, kind, compress, reads
):
    import tracemalloc

    raw = _with_header(kind, b"x" * (40 * MB))
    path = tmp_path / ("a.tar.gz" if compress else "a.tar")
    path.write_bytes(compress(raw) if compress else raw)
    patched = _whole_header_reads() if reads == "whole" else contextlib.nullcontext()
    tracemalloc.start()
    try:
        with patched, pytest.raises(ArchiveBudgetError, match="a header in .* takes more than"):
            list(TarReader(input_path=path).get_raw_chunks())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16 * MB, f"peak {peak:,} bytes"


def test_a_header_of_ordinary_size_is_read(tmp_path):
    long_name = "d/" + "n" * 300 + ".txt"  # a GNU long-name record
    with tarfile.open(tmp_path / "a.tar", "w", format=tarfile.GNU_FORMAT) as tf:
        _add(tf, long_name, b"Text under a long name.")
    with tarfile.open(tmp_path / "b.tar.gz", "w:gz", format=tarfile.PAX_FORMAT) as tf:
        info = tarfile.TarInfo("p.txt")
        info.pax_headers = {"comment": "c" * 100_000}
        info.size = 4
        tf.addfile(info, io.BytesIO(b"Pax."))
    assert _paths(TarReader(input_path=tmp_path / "b.tar.gz")) == ["b.tar.gz/p.txt"]
    # The 306-character name is read whole from its header; whether the
    # file system can hold it decides only whether it is extracted.
    reader = TarReader(input_path=tmp_path / "a.tar")
    got = _paths(reader)
    assert got in ([f"a.tar/{long_name}"], [])
    if not got:
        assert reader.coverage.skipped == {"unwritable-name": 1}


@pytest.mark.parametrize("reader_cls", [TarReader, ZipReader])
def test_a_name_the_file_system_cannot_hold_skips_only_that_member(
    tmp_path, reader_cls
):
    too_long = "n" * 400 + ".txt"  # past every file system's name limit
    members = [("a.txt", b"First text."), (too_long, b"Lost."), ("b.txt", b"Last.")]
    if reader_cls is TarReader:
        path = _make(tmp_path / "a.tar", members)
    else:
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in members:
                zf.writestr(name, data)
    reader = reader_cls(input_path=path)
    assert [p.split("/", 1)[1] for p in _paths(reader)] == ["a.txt", "b.txt"]
    assert reader.coverage.skipped == {"unwritable-name": 1}


def test_other_write_errors_still_stop_the_read(tmp_path):
    import errno as errno_codes

    path = _make(tmp_path / "a.tar", [("a.txt", b"Text.")])
    real_open = open
    writes = []

    def full_disk(file, mode="r", *args, **kwargs):
        if "w" in mode:  # only the extracted member is written
            writes.append(file)
            raise OSError(errno_codes.ENOSPC, "No space left on device")
        return real_open(file, mode, *args, **kwargs)

    with mock.patch("builtins.open", full_disk):
        with pytest.raises(OSError, match="No space left"):
            list(TarReader(input_path=path).get_raw_chunks())
    assert [Path(w).name for w in writes] == ["a.txt"]


def test_a_compressed_scan_stops_at_its_position_bound(tmp_path):
    # Directory entries carry no data size, yet each takes a header to
    # decompress: the stream position bounds them.
    raw = b"".join(
        _raw_header(f"d{i}/", tarfile.DIRTYPE, 0) for i in range(20_000)
    ) + b"\0" * 1024
    path = tmp_path / "a.tar.gz"
    path.write_bytes(gzip.compress(raw))
    options = {"max_files": 1_000_000, "max_total_bytes": MB,
               "max_text_chars_per_byte": None}
    with mock.patch.object(tar_module, "_HEADER_BYTES_PER_FILE", 0):
        with pytest.raises(ArchiveBudgetError, match="decompress past"):
            list(TarReader(input_path=path, **options).get_raw_chunks())
