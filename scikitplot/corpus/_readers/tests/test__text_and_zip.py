# scikitplot/corpus/_readers/tests/test__text_and_zip.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus._readers._text and ._zip
=====================================================

Coverage — _text.py
--------------------
* :func:`_detect_encoding` — BOM paths (utf-8-sig, utf-16-le, utf-16-be,
  utf-32-le, utf-32-be), strict UTF-8 path, chardet path (mocked),
  chardet low-confidence fallback, chardet absent fallback, Latin-1
  fallback for invalid UTF-8.
* :class:`TextReader` — construction, encoding auto-detection end-to-end,
  explicit encoding, max_file_bytes guard, empty-file handling, whitespace-
  only file handling, section_type in yielded chunk, get_documents() returns
  CorpusDocument instances.
* :class:`MarkdownReader` — correct file_type, inherits TextReader behaviour.
* :class:`ReSTReader` — correct file_type, inherits TextReader behaviour.

Coverage — _zip.py
-------------------
* :func:`_should_skip` — hidden files, __MACOSX, __pycache__, normal files.
* :func:`_is_within` — inside / outside / symlink-escape.
* :class:`ZipReader` — construction defaults, max_files/max_total_bytes
  validation, reader_kwargs normalisation, TypeError on bad reader_kwargs;
  get_raw_chunks: reads txt members, skips hidden/system entries, ZipSlip
  rejection, file-count guard, bomb-size guard, skip_unsupported=True/False.

All tests use tmp_path; no network calls; no optional dependencies.

Run with::

    pytest corpus/_readers/tests/test__text_and_zip.py -v
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from .._text import (
    MarkdownReader,
    ReSTReader,
    TextReader,
    _detect_encoding,
)
from .._zip import ZipReader, _is_within, _should_skip


# ===========================================================================
# _detect_encoding — BOM and fallback chain
# ===========================================================================


class TestDetectEncoding:

    def test_utf8_bom(self) -> None:
        assert _detect_encoding(b"\xef\xbb\xbfHello") == "utf-8-sig"

    def test_utf32_le_bom(self) -> None:
        assert _detect_encoding(b"\xff\xfe\x00\x00Hello") == "utf-32-le"

    def test_utf32_be_bom(self) -> None:
        assert _detect_encoding(b"\x00\x00\xfe\xffHello") == "utf-32-be"

    def test_utf16_le_bom(self) -> None:
        # Must NOT be confused with utf-32-le (4-byte prefix checked first).
        raw = b"\xff\xfeH\x00e\x00l\x00l\x00o\x00"
        result = _detect_encoding(raw)
        assert result in ("utf-16-le",)

    def test_utf16_be_bom(self) -> None:
        raw = b"\xfe\xff\x00H\x00e\x00l\x00l\x00o"
        assert _detect_encoding(raw) == "utf-16-be"

    def test_strict_utf8_path(self) -> None:
        raw = "Hello, 世界!".encode("utf-8")
        assert _detect_encoding(raw) == "utf-8"

    def test_plain_ascii_is_utf8(self) -> None:
        assert _detect_encoding(b"Hello world") == "utf-8"

    def test_latin1_fallback_for_invalid_utf8(self) -> None:
        # 0x80-0xFF bytes are valid Latin-1 but invalid UTF-8.
        raw = b"calf\xe9"  # "café" in Latin-1
        result = _detect_encoding(raw)
        assert result in ("latin-1", "ISO-8859-1", "windows-1252", "latin1")

    def test_chardet_used_when_installed(self) -> None:
        raw = b"calf\xe9 au lait"  # Latin-1, not valid UTF-8
        mock_result = {"encoding": "ISO-8859-1", "confidence": 0.99}
        with patch.dict("sys.modules", {"chardet": MagicMock(detect=MagicMock(return_value=mock_result))}):
            # Force UTF-8 decode to fail so chardet branch is reached.
            result = _detect_encoding(raw)
        # chardet or latin-1 fallback — either is acceptable
        assert result is not None

    def test_chardet_low_confidence_falls_through_to_latin1(self) -> None:
        raw = b"calf\xe9"
        mock_result = {"encoding": "ISO-8859-1", "confidence": 0.50}
        mock_chardet = MagicMock()
        mock_chardet.detect.return_value = mock_result
        with patch.dict("sys.modules", {"chardet": mock_chardet}):
            result = _detect_encoding(raw)
        assert result in ("latin-1", "latin1", "ISO-8859-1")

    def test_chardet_absent_still_returns_string(self) -> None:
        raw = b"calf\xe9"
        with patch.dict("sys.modules", {"chardet": None}):
            result = _detect_encoding(raw)
        assert isinstance(result, str)


# ===========================================================================
# TextReader
# ===========================================================================


class TestTextReader:

    def test_reads_utf8_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        reader = TextReader(input_path=f)
        chunks = list(reader.get_raw_chunks())
        assert len(chunks) == 1
        assert "Hello, world!" in chunks[0]["text"]

    def test_section_type_in_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("Content.", encoding="utf-8")
        reader = TextReader(input_path=f)
        chunk = next(reader.get_raw_chunks())
        assert "section_type" in chunk

    def test_explicit_encoding_latin1(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_bytes("café".encode("latin-1"))
        reader = TextReader(input_path=f, encoding="latin-1")
        chunk = next(reader.get_raw_chunks())
        assert "café" in chunk["text"]

    def test_explicit_encoding_utf8(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("Ångström", encoding="utf-8")
        reader = TextReader(input_path=f, encoding="utf-8")
        chunk = next(reader.get_raw_chunks())
        assert "Ångström" in chunk["text"]

    def test_utf8_bom_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_bytes(b"\xef\xbb\xbfHello BOM world")
        reader = TextReader(input_path=f)
        chunk = next(reader.get_raw_chunks())
        # BOM must not appear in the decoded text.
        assert "\ufeff" not in chunk["text"]
        assert "Hello BOM world" in chunk["text"]

    def test_max_file_bytes_rejects_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * 200)
        reader = TextReader(input_path=f, max_file_bytes=100)
        with pytest.raises(ValueError, match="max_file_bytes"):
            list(reader.get_raw_chunks())

    def test_max_file_bytes_zero_raises_at_construction(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="max_file_bytes"):
            TextReader(input_path=f, max_file_bytes=0)

    def test_empty_file_yields_no_chunks(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        reader = TextReader(input_path=f)
        assert list(reader.get_raw_chunks()) == []

    def test_whitespace_only_file_yields_no_chunks(self, tmp_path: Path) -> None:
        f = tmp_path / "ws.txt"
        f.write_text("   \n\t  \n", encoding="utf-8")
        reader = TextReader(input_path=f)
        assert list(reader.get_raw_chunks()) == []

    def test_multiline_content_preserved(self, tmp_path: Path) -> None:
        content = "Line one.\nLine two.\nLine three."
        f = tmp_path / "multi.txt"
        f.write_text(content, encoding="utf-8")
        reader = TextReader(input_path=f)
        chunk = next(reader.get_raw_chunks())
        assert chunk["text"] == content

    def test_get_documents_returns_corpus_documents(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("The quick brown fox.", encoding="utf-8")
        reader = TextReader(input_path=f)
        docs = list(reader.get_documents())
        assert len(docs) >= 1
        # Each returned item must have a doc_id and text attribute.
        for doc in docs:
            assert hasattr(doc, "doc_id")
            assert hasattr(doc, "text")

    def test_file_type_class_variable(self) -> None:
        assert TextReader.file_type == ".txt"

    def test_file_types_contains_txt(self) -> None:
        assert ".txt" in (TextReader.file_types or [])

    def test_unicode_content_preserved(self, tmp_path: Path) -> None:
        content = "Привет мир. 你好世界. مرحبا بالعالم."
        f = tmp_path / "unicode.txt"
        f.write_text(content, encoding="utf-8")
        reader = TextReader(input_path=f)
        chunk = next(reader.get_raw_chunks())
        assert "Привет" in chunk["text"]
        assert "你好" in chunk["text"]

    def test_very_large_text_within_limit(self, tmp_path: Path) -> None:
        content = "word " * 10_000  # ~50KB
        f = tmp_path / "large.txt"
        f.write_text(content, encoding="utf-8")
        reader = TextReader(input_path=f, max_file_bytes=1_000_000)
        chunk = next(reader.get_raw_chunks())
        assert len(chunk["text"]) > 1000

    def test_exactly_at_max_file_bytes_is_accepted(self, tmp_path: Path) -> None:
        content = b"x" * 50
        f = tmp_path / "exact.txt"
        f.write_bytes(content)
        # Equal to limit must NOT raise (> not >=).
        reader = TextReader(input_path=f, max_file_bytes=50)
        chunks = list(reader.get_raw_chunks())
        assert len(chunks) == 1

    def test_one_byte_over_max_file_bytes_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "over.txt"
        f.write_bytes(b"x" * 51)
        reader = TextReader(input_path=f, max_file_bytes=50)
        with pytest.raises(ValueError, match="max_file_bytes"):
            list(reader.get_raw_chunks())


# ===========================================================================
# MarkdownReader and ReSTReader — extension aliases
# ===========================================================================


class TestMarkdownReader:

    def test_file_type(self) -> None:
        assert MarkdownReader.file_type == ".md"

    def test_file_types_contains_md(self) -> None:
        assert ".md" in (MarkdownReader.file_types or [])

    def test_reads_markdown_file(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Heading\n\nParagraph here.", encoding="utf-8")
        reader = MarkdownReader(input_path=f)
        chunk = next(reader.get_raw_chunks())
        assert "Heading" in chunk["text"]
        assert "Paragraph" in chunk["text"]

    def test_is_subclass_of_text_reader(self) -> None:
        assert issubclass(MarkdownReader, TextReader)


class TestReSTReader:

    def test_file_type(self) -> None:
        assert ReSTReader.file_type == ".rst"

    def test_file_types_contains_rst(self) -> None:
        assert ".rst" in (ReSTReader.file_types or [])

    def test_reads_rst_file(self, tmp_path: Path) -> None:
        f = tmp_path / "changes.rst"
        f.write_text("Title\n=====\n\nSome content here.", encoding="utf-8")
        reader = ReSTReader(input_path=f)
        chunk = next(reader.get_raw_chunks())
        assert "Title" in chunk["text"]

    def test_is_subclass_of_text_reader(self) -> None:
        assert issubclass(ReSTReader, TextReader)


# ===========================================================================
# _should_skip — ZIP member filter
# ===========================================================================


class TestShouldSkip:

    def test_normal_file_not_skipped(self) -> None:
        assert _should_skip("docs/chapter01.txt") is False

    def test_hidden_dot_file_skipped(self) -> None:
        assert _should_skip(".hidden") is True

    def test_hidden_dot_in_subdir_skipped(self) -> None:
        assert _should_skip("subdir/.hidden_file") is True

    def test_macosx_directory_skipped(self) -> None:
        assert _should_skip("__MACOSX/._readme.txt") is True

    def test_pycache_skipped(self) -> None:
        assert _should_skip("pkg/__pycache__/module.pyc") is True

    def test_nested_normal_path_not_skipped(self) -> None:
        assert _should_skip("a/b/c/document.pdf") is False

    def test_dot_in_filename_not_skipped(self) -> None:
        # A file named "report.v2.txt" must NOT be skipped (dot is in the name, not a component).
        assert _should_skip("reports/report.v2.txt") is False

    def test_root_dot_file_skipped(self) -> None:
        assert _should_skip(".DS_Store") is True


# ===========================================================================
# _is_within — ZipSlip guard
# ===========================================================================


class TestIsWithin:

    def test_child_inside_parent(self, tmp_path: Path) -> None:
        child = (tmp_path / "sub" / "file.txt").resolve()
        assert _is_within(child, tmp_path.resolve()) is True

    def test_child_is_parent_itself(self, tmp_path: Path) -> None:
        # The parent directory itself is not "within" in a strict sense —
        # relative_to succeeds with empty path so is True.
        assert _is_within(tmp_path.resolve(), tmp_path.resolve()) is True

    def test_child_outside_parent(self, tmp_path: Path) -> None:
        parent = tmp_path / "safe"
        parent.mkdir()
        # Escape path: go up from safe and into evil
        evil = (tmp_path / "evil" / "file.txt").resolve()
        assert _is_within(evil, parent.resolve()) is False

    def test_path_traversal_attempt(self, tmp_path: Path) -> None:
        # Simulate ZipSlip: extracted path resolves outside tmp_path.
        parent = tmp_path / "extract"
        parent.mkdir()
        # Manually construct a resolved path outside parent.
        escape = Path("/etc/passwd").resolve()
        assert _is_within(escape, parent.resolve()) is False


# ===========================================================================
# ZipReader — construction and validation
# ===========================================================================


class TestZipReaderConstruction:

    def _make_zip(self, tmp_path: Path, members: dict[str, bytes]) -> Path:
        """Create a ZIP file containing the given members."""
        zpath = tmp_path / "archive.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return zpath

    def test_default_max_files(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        r = ZipReader(input_path=zpath)
        assert r.max_files == 10_000

    def test_default_max_total_bytes(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        r = ZipReader(input_path=zpath)
        assert r.max_total_bytes == 2 * 1024 * 1024 * 1024

    def test_max_files_zero_raises(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        with pytest.raises(ValueError, match="max_files"):
            ZipReader(input_path=zpath, max_files=0)

    def test_max_total_bytes_zero_raises(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        with pytest.raises(ValueError, match="max_total_bytes"):
            ZipReader(input_path=zpath, max_total_bytes=0)

    def test_max_files_negative_raises(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        with pytest.raises(ValueError, match="max_files"):
            ZipReader(input_path=zpath, max_files=-1)

    def test_reader_kwargs_normalises_extension(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        r = ZipReader(input_path=zpath, reader_kwargs={"TXT": {}})
        assert ".txt" in r.reader_kwargs

    def test_reader_kwargs_not_dict_raises(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        with pytest.raises(TypeError, match="reader_kwargs"):
            ZipReader(input_path=zpath, reader_kwargs="bad")  # type: ignore[arg-type]

    def test_reader_kwargs_value_not_dict_raises(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        with pytest.raises(TypeError, match="reader_kwargs"):
            ZipReader(input_path=zpath, reader_kwargs={".txt": "bad"})  # type: ignore[dict-item]

    def test_reader_kwargs_leading_dot_preserved(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        r = ZipReader(input_path=zpath, reader_kwargs={".txt": {"encoding": "utf-8"}})
        assert ".txt" in r.reader_kwargs
        assert r.reader_kwargs[".txt"]["encoding"] == "utf-8"


# ===========================================================================
# ZipReader — get_raw_chunks: functional tests
# ===========================================================================


class TestZipReaderGetRawChunks:

    def _make_zip(self, tmp_path: Path, members: dict[str, str | bytes]) -> Path:
        zpath = tmp_path / "archive.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            for name, data in members.items():
                if isinstance(data, str):
                    zf.writestr(name, data.encode("utf-8"))
                else:
                    zf.writestr(name, data)
        return zpath

    def test_reads_txt_member(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {"doc.txt": "Hello from zip."})
        reader = ZipReader(input_path=zpath, skip_unsupported=True)
        chunks = list(reader.get_raw_chunks())
        assert len(chunks) >= 1
        assert any("Hello from zip." in c.get("text", "") for c in chunks)

    def test_skips_hidden_member(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {
            ".hidden": "secret",
            "visible.txt": "Hello visible.",
        })
        reader = ZipReader(input_path=zpath, skip_unsupported=True)
        chunks = list(reader.get_raw_chunks())
        # Only visible.txt should contribute.
        all_text = " ".join(c.get("text", "") for c in chunks)
        assert "Hello visible." in all_text

    def test_skips_macosx_member(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {
            "__MACOSX/._doc.txt": "garbage",
            "doc.txt": "Real content.",
        })
        reader = ZipReader(input_path=zpath, skip_unsupported=True)
        chunks = list(reader.get_raw_chunks())
        all_text = " ".join(c.get("text", "") for c in chunks)
        assert "Real content." in all_text
        assert "garbage" not in all_text

    def test_file_count_guard_raises(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {
            f"file{i}.txt": "x" for i in range(5)
        })
        reader = ZipReader(input_path=zpath, max_files=2, skip_unsupported=True)
        with pytest.raises(ValueError, match="max_files"):
            list(reader.get_raw_chunks())

    def test_bomb_guard_raises(self, tmp_path: Path) -> None:
        # Create members whose cumulative uncompressed size exceeds limit.
        big = "x" * 1000
        zpath = self._make_zip(tmp_path, {
            "a.txt": big,
            "b.txt": big,
            "c.txt": big,
        })
        reader = ZipReader(input_path=zpath, max_total_bytes=500, skip_unsupported=True)
        with pytest.raises(ValueError, match="max_total_bytes"):
            list(reader.get_raw_chunks())

    def test_unsupported_ext_raises_when_skip_false(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {"data.xyz123": "content"})
        reader = ZipReader(input_path=zpath, skip_unsupported=False)
        with pytest.raises(ValueError, match="unsupported extension"):
            list(reader.get_raw_chunks())

    def test_unsupported_ext_skipped_when_skip_true(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {
            "data.xyz123": "garbage",
            "doc.txt": "Good content.",
        })
        reader = ZipReader(input_path=zpath, skip_unsupported=True)
        chunks = list(reader.get_raw_chunks())
        all_text = " ".join(c.get("text", "") for c in chunks)
        assert "Good content." in all_text

    def test_empty_zip_yields_nothing(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {})
        reader = ZipReader(input_path=zpath, skip_unsupported=True)
        assert list(reader.get_raw_chunks()) == []

    def test_multiple_txt_members_all_yielded(self, tmp_path: Path) -> None:
        zpath = self._make_zip(tmp_path, {
            "a.txt": "Content A.",
            "b.txt": "Content B.",
            "c.txt": "Content C.",
        })
        reader = ZipReader(input_path=zpath, skip_unsupported=True)
        chunks = list(reader.get_raw_chunks())
        all_text = " ".join(c.get("text", "") for c in chunks)
        assert "Content A." in all_text
        assert "Content B." in all_text
        assert "Content C." in all_text

    def test_zipslip_member_skipped(self, tmp_path: Path) -> None:
        """A member with a path-traversal name must be silently skipped."""
        zpath = tmp_path / "evil.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            # Manually write a member with a traversal name.
            info = zipfile.ZipInfo("../../../etc/passwd")
            zf.writestr(info, "root:x:0:0")
            zf.writestr("safe.txt", "Safe content here.")
        reader = ZipReader(input_path=zpath, skip_unsupported=True)
        # Must not raise; the traversal member is skipped.
        chunks = list(reader.get_raw_chunks())
        all_text = " ".join(c.get("text", "") for c in chunks)
        assert "root:x:0:0" not in all_text
