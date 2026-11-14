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
from scikitplot.corpus._readers._zip import ArchiveBudgetError, ZipReader, _archive_ctx
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


class TestRatio:
    """``max_text_chars_per_byte``: output in proportion to the input size."""

    FLOOR = 1024 * 1024

    def _big(self, tmp_path, size, texts, **kwargs):
        path = tmp_path / "in.txt"
        path.write_bytes(b"x" * size)
        return DocumentReader.create(
            path, custom_extractor=lambda _p: list(texts), **kwargs
        )

    def test_off_by_default_and_50_for_the_amplifying_formats(self, tmp_path):
        assert _reader(tmp_path, ["x"]).max_text_chars_per_byte is None
        ods = _odf(tmp_path / "b.ods", "spreadsheet", "")
        assert OfficeReader(input_path=ods).max_text_chars_per_byte == 50
        pdf = _shared_stream_pdf(tmp_path / "b.pdf", 100, 1)
        assert PDFReader(input_path=pdf).max_text_chars_per_byte == 50

    def test_a_small_file_may_always_yield_the_floor(self, tmp_path):
        """10 bytes x 1 char/byte = 10, but the floor of 1 Mi characters holds."""
        texts = ["a" * (self.FLOOR // 2)] * 2
        reader = self._big(tmp_path, 10, texts, max_text_chars_per_byte=1)
        assert _texts(reader) == texts
        over = self._big(tmp_path, 10, [*texts, "a"], max_text_chars_per_byte=1)
        with pytest.raises(ValueError, match=r"more than 1,048,576 characters.*10 bytes.*max_text_chars_per_byte=1\b"):
            _texts(over)

    def test_past_the_floor_the_limit_follows_the_file_size(self, tmp_path):
        size = self.FLOOR  # 1 Mi bytes x 2 chars/byte = 2 Mi characters
        texts = ["a" * self.FLOOR] * 2
        assert _texts(self._big(tmp_path, size, texts, max_text_chars_per_byte=2)) == texts
        with pytest.raises(ValueError, match="max_text_chars_per_byte=2"):
            _texts(self._big(tmp_path, size, [*texts, "a"], max_text_chars_per_byte=2))

    def test_the_smaller_of_the_two_limits_applies(self, tmp_path):
        reader = self._big(
            tmp_path, 10, ["abc", "def"], max_text_chars=5, max_text_chars_per_byte=1
        )
        with pytest.raises(ValueError, match="max_text_chars=5"):
            _texts(reader)

    def test_truncate_applies_to_the_ratio_too(self, tmp_path):
        reader = self._big(
            tmp_path,
            10,
            ["a" * self.FLOOR, "bcd"],
            max_text_chars_per_byte=1,
            on_text_limit="truncate",
        )
        chunks = list(reader._iter_raw_chunks())
        assert [len(c["text"]) for c in chunks] == [self.FLOOR]

    def test_the_hostile_pdf_stops_at_the_floor_by_default(self, tmp_path):
        """4 KB sharing one stream: refused after 1 Mi characters, not 256 Mi."""
        pytest.importorskip("pypdf")
        path = _shared_stream_pdf(tmp_path / "b.pdf", 100_000, 20)
        reader = PDFReader(input_path=path, prefer_backend="pypdf")
        seen = 0
        with pytest.raises(ValueError, match="max_text_chars_per_byte=50"):
            for chunk in reader._iter_raw_chunks():
                seen += len(chunk["text"])
        assert seen <= self.FLOOR

    def test_none_turns_the_ratio_off(self, tmp_path):
        ods = _odf(
            tmp_path / "b.ods",
            "spreadsheet",
            '<table:table table:name="T">' + _row(_cell("x", 1_000), 2_000) + "</table:table>",
        )
        with pytest.raises(ValueError, match="max_text_chars_per_byte=50"):
            _texts(OfficeReader(input_path=ods))
        texts = _texts(OfficeReader(input_path=ods, max_text_chars_per_byte=None))
        assert sum(map(len, texts)) > self.FLOOR

    @pytest.mark.parametrize("value", [0, -1, True, "5", float("inf"), float("nan")])
    def test_bad_ratios_are_refused(self, tmp_path, value):
        with pytest.raises(ValueError, match="max_text_chars_per_byte"):
            _reader(tmp_path, ["x"], max_text_chars_per_byte=value)


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


_MI = 1024 * 1024


def _zip_of(path: Path, members: dict, level: int = 9) -> Path:
    """An archive of ``{name: bytes}``, deflated (highly compressible content)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


class TestZipInProportionToTheArchive:
    """Defaults and the extraction bound derived from the ratio (CX-41)."""

    def test_defaults_are_those_of_the_amplifying_formats(self, tmp_path):
        reader = ZipReader(input_path=_zip_of(tmp_path / "a.zip", {"a.txt": b"x"}))
        assert reader.max_text_chars == _DEFAULT
        assert reader.max_text_chars_per_byte == 50

    @pytest.mark.parametrize(
        ("size", "kwargs", "limit", "names"),
        [
            (0, {}, 4 * _MI, "max_text_chars_per_byte=50"),
            (_MI // 50, {}, 4 * _MI, "max_text_chars_per_byte=50"),
            (_MI // 50 + 1, {}, 4 * 50 * (_MI // 50 + 1), "max_text_chars_per_byte=50"),
            (3 * _MI + 1, {"max_text_chars_per_byte": 0.5}, 4 * (3 * _MI // 2 + 1), "=0.5"),
            (100 * _MI, {}, 2 * 1024 * _MI, "max_total_bytes=2,147,483,648"),
            (_MI, {"max_total_bytes": 1000}, 1000, "max_total_bytes=1,000"),
            (1, {"max_text_chars_per_byte": None}, 2 * 1024 * _MI, "max_total_bytes="),
            # An absolute text limit says nothing about the members' size.
            (1, {"max_text_chars": 10}, 4 * _MI, "max_text_chars_per_byte=50"),
        ],
    )
    def test_the_extraction_limit_in_force(self, tmp_path, size, kwargs, limit, names):
        reader = ZipReader(input_path=_zip_of(tmp_path / "a.zip", {"a.txt": b"x"}), **kwargs)
        found, described = reader._extract_limit(size)
        assert found == limit
        assert names in described

    def test_a_member_past_the_limit_is_refused_before_it_is_extracted(
        self, tmp_path, monkeypatch
    ):
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": b"x" * (4 * _MI + 1)})
        assert archive.stat().st_size * 200 < 4 * _MI  # the floor decides
        copied = []
        monkeypatch.setattr(
            "scikitplot.corpus._archive_handler.stream_copy_bounded",
            lambda *a, **k: copied.append(k["member_name"]) or 0,
        )
        with pytest.raises(ValueError) as caught:
            _texts(ZipReader(input_path=archive))
        message = str(caught.value)
        assert "'a.txt' of 'a.zip' declares 4,194,305 bytes, 4,194,304 left" in message
        assert f"4,194,304 bytes for an archive of {archive.stat().st_size:,} bytes" in message
        assert "max_text_chars_per_byte=50" in message
        assert copied == []

    def test_a_member_of_exactly_the_limit_is_extracted(self, tmp_path):
        # Extracted (4 Mi bytes), then refused as text: 4 Mi characters > 1 Mi.
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": b"x" * (4 * _MI)})
        with pytest.raises(ValueError, match="a.zip yields more than 1,048,576 characters"):
            _texts(ZipReader(input_path=archive))

    def test_the_limit_is_cumulative_over_members(self, tmp_path):
        # Blank members yield no text, so only the byte bound can decide.
        members = {f"{i}.txt": b" " * _MI for i in range(5)}
        archive = _zip_of(tmp_path / "a.zip", members)
        with pytest.raises(ValueError) as caught:
            _texts(ZipReader(input_path=archive))
        message = str(caught.value)
        assert "'4.txt' of 'a.zip' declares 1,048,576 bytes, 0 left" in message

    def test_the_copy_is_given_what_is_left_of_the_limit(self, tmp_path, monkeypatch):
        # zipfile never yields more than a member declares, so the budget of
        # the copy (the guard against a container that lies) is checked here.
        import scikitplot.corpus._archive_handler as handler

        real, budgets = handler.stream_copy_bounded, []

        def recording(src, dst, *, max_bytes, **kwargs):
            budgets.append(max_bytes)
            return real(src, dst, max_bytes=max_bytes, **kwargs)

        monkeypatch.setattr(handler, "stream_copy_bounded", recording)
        archive = _zip_of(tmp_path / "a.zip", {"1.txt": b" " * 1000, "2.txt": b" " * 10})
        assert _texts(ZipReader(input_path=archive)) == []
        assert budgets == [4 * _MI, 4 * _MI - 1000]

    def test_text_at_the_floor_is_read(self, tmp_path):
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": b"x" * _MI})
        assert [len(t) for t in _texts(ZipReader(input_path=archive))] == [_MI]

    def test_one_character_past_the_floor_is_refused_as_text(self, tmp_path):
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": b"x" * (_MI + 1)})
        with pytest.raises(ValueError, match="max_text_chars_per_byte=50"):
            _texts(ZipReader(input_path=archive))

    def test_truncate_keeps_exactly_the_limit(self, tmp_path):
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": b"x" * (_MI + 5)})
        kept = _texts(ZipReader(input_path=archive, on_text_limit="truncate"))
        assert [len(t) for t in kept] == [_MI]

    def test_none_turns_both_bounds_off(self, tmp_path):
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": b"x" * (4 * _MI + 1)})
        kept = _texts(ZipReader(input_path=archive, max_text_chars_per_byte=None))
        assert [len(t) for t in kept] == [4 * _MI + 1]

    def test_an_archive_that_is_not_compressible_is_unaffected(self, tmp_path):
        import hashlib

        noise = b"".join(hashlib.sha256(bytes([i % 256, i // 256])).digest() for i in range(2048))
        text = noise.hex().encode("ascii")  # 131,072 characters, about 2:1
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": text})
        assert [len(t) for t in _texts(ZipReader(input_path=archive))] == [len(text)]


class TestZipTreeBudget:
    """One extraction budget for the whole archive tree, set by the top archive."""

    def test_nested_archives_draw_from_the_top_budget(self, tmp_path):
        inner = _zip_of(tmp_path / "inner.zip", {"blank.txt": b" " * (4 * _MI)})
        # Each inner archive is within its own bound; together they are not.
        members = {f"part{i}.zip": inner.read_bytes() for i in range(3)}
        archive = _zip_of(tmp_path / "a.zip", members)
        assert archive.stat().st_size * 200 < 4 * _MI
        with pytest.raises(ArchiveBudgetError) as caught:
            list(ZipReader(input_path=archive).get_documents())
        assert "'blank.txt' of 'part0.zip'" in str(caught.value)
        assert "4,194,304 bytes for an archive of" in str(caught.value)
        assert _archive_ctx.get() is None

    def test_a_small_nested_archive_keeps_its_own_tighter_bound(self, tmp_path):
        import hashlib

        # A large, honest outer archive (its budget: about 26 MB) holding a
        # small inner archive whose blank member is past the inner's own 4 Mi.
        noise = b"".join(hashlib.sha256(i.to_bytes(4, "big")).digest() for i in range(4096))
        inner = _zip_of(tmp_path / "inner.zip", {"blank.txt": b" " * (5 * _MI)})
        archive = _zip_of(
            tmp_path / "a.zip", {"noise.bin": noise, "inner.zip": inner.read_bytes()}
        )
        assert archive.stat().st_size * 200 > 6 * _MI  # the top budget would allow it
        with pytest.raises(ArchiveBudgetError) as caught:
            _texts(ZipReader(input_path=archive))
        assert f"for an archive of {inner.stat().st_size:,} bytes" in str(caught.value)

    def test_a_nested_refusal_is_not_swallowed_as_a_member_error(self, tmp_path):
        inner = _zip_of(tmp_path / "inner.zip", {"blank.txt": b" " * (5 * _MI)})
        archive = _zip_of(
            tmp_path / "a.zip", {"inner.zip": inner.read_bytes(), "note.txt": b"kept?"}
        )
        with pytest.raises(ArchiveBudgetError):
            _texts(ZipReader(input_path=archive))

    def test_truncate_ends_the_archive_and_keeps_what_was_read(self, tmp_path, caplog):
        archive = _zip_of(
            tmp_path / "a.zip", {"1.txt": b"first", "2.txt": b"x" * (5 * _MI)}
        )
        reader = ZipReader(input_path=archive, on_text_limit="truncate")
        with caplog.at_level("WARNING", logger="scikitplot.corpus._readers._zip"):
            assert _texts(reader) == ["first"]
        assert "on_text_limit='truncate'" in caplog.text
        assert "'2.txt'" in caplog.text

    def test_truncate_in_a_nested_reader_still_refuses_the_tree(self, tmp_path):
        # Only the top-level reader decides; a member's setting does not.
        inner = _zip_of(tmp_path / "inner.zip", {"blank.txt": b" " * (5 * _MI)})
        archive = _zip_of(tmp_path / "a.zip", {"inner.zip": inner.read_bytes()})
        reader = ZipReader(
            input_path=archive, reader_kwargs={".zip": {"on_text_limit": "truncate"}}
        )
        with pytest.raises(ArchiveBudgetError):
            _texts(reader)

    def test_content_past_its_header_is_a_budget_error(self, tmp_path, monkeypatch):
        def overflow(*args, **kwargs):
            raise ValueError("archive member 'a.txt' decompressed beyond the remaining budget")

        monkeypatch.setattr(
            "scikitplot.corpus._archive_handler.stream_copy_bounded", overflow
        )
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": b"x"})
        with pytest.raises(ArchiveBudgetError, match="max_text_chars_per_byte=50"):
            _texts(ZipReader(input_path=archive))

    def test_the_budget_holds_on_both_entry_points(self, tmp_path):
        archive = _zip_of(tmp_path / "a.zip", {"a.txt": b" " * (5 * _MI)})
        for read in (
            lambda r: list(r.get_raw_chunks()),
            lambda r: list(r.get_documents()),
        ):
            with pytest.raises(ArchiveBudgetError):
                read(ZipReader(input_path=archive))
            assert _archive_ctx.get() is None

    def test_a_budget_error_is_a_value_error(self):
        assert issubclass(ArchiveBudgetError, ValueError)
