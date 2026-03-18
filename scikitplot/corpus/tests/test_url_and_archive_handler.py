# tests/test_url_and_archive_handler.py
#
# Unit tests for _url_handler and _archive_handler modules.
# These tests are self-contained — no network access required.

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Tests for _url_handler.classify_url
# ---------------------------------------------------------------------------

from scikitplot.corpus._url_handler import (
    URLKind,
    _infer_extension_from_headers,
    _make_temp_filename,
    _resolve_gdrive,
    _resolve_github_blob,
    classify_url,
    resolve_url,
)


class TestClassifyURL:
    """Tests for classify_url."""

    # -- YouTube --
    def test_youtube_standard(self):
        assert classify_url("https://www.youtube.com/watch?v=abc123") == URLKind.YOUTUBE

    def test_youtube_short(self):
        assert classify_url("https://youtu.be/abc123") == URLKind.YOUTUBE

    def test_youtube_shorts(self):
        assert classify_url("https://www.youtube.com/shorts/abc123") == URLKind.YOUTUBE

    def test_youtube_embed(self):
        assert classify_url("https://www.youtube.com/embed/abc123") == URLKind.YOUTUBE

    def test_youtube_live(self):
        assert classify_url("https://www.youtube.com/live/abc123") == URLKind.YOUTUBE

    # -- Google Drive --
    def test_gdrive_file(self):
        assert classify_url(
            "https://drive.google.com/file/d/1abc-DEF_xyz/view"
        ) == URLKind.GOOGLE_DRIVE

    def test_gdrive_open(self):
        assert classify_url(
            "https://drive.google.com/open?id=1abc-DEF_xyz"
        ) == URLKind.GOOGLE_DRIVE

    # -- GitHub --
    def test_github_blob(self):
        assert classify_url(
            "https://github.com/user/repo/blob/main/data/file.csv"
        ) == URLKind.GITHUB_BLOB

    def test_github_raw(self):
        assert classify_url(
            "https://raw.githubusercontent.com/user/repo/main/data.csv"
        ) == URLKind.GITHUB_RAW

    # -- Downloadable --
    def test_pdf_url(self):
        assert classify_url("https://example.com/report.pdf") == URLKind.DOWNLOADABLE

    def test_csv_url(self):
        assert classify_url("https://example.com/data.csv") == URLKind.DOWNLOADABLE

    def test_mp3_url(self):
        assert classify_url("https://archive.org/download/file.mp3") == URLKind.DOWNLOADABLE

    def test_tar_gz_url(self):
        assert classify_url("https://example.com/data.tar.gz") == URLKind.DOWNLOADABLE

    def test_zip_url(self):
        assert classify_url("https://example.com/corpus.zip") == URLKind.DOWNLOADABLE

    def test_image_url(self):
        assert classify_url("https://example.com/scan.png") == URLKind.DOWNLOADABLE

    def test_mp4_url(self):
        assert classify_url("https://example.com/video.mp4") == URLKind.DOWNLOADABLE

    # -- Web page (fallback) --
    def test_web_page_no_ext(self):
        assert classify_url("https://example.com/article") == URLKind.WEB_PAGE

    def test_web_page_html(self):
        # .html is not in _DOWNLOADABLE_EXTENSIONS — treated as web page
        # Actually .html is not in the set, so it falls to WEB_PAGE
        result = classify_url("https://example.com/page.html")
        # .html is not in _DOWNLOADABLE_EXTENSIONS
        assert result == URLKind.WEB_PAGE

    def test_web_page_news(self):
        assert classify_url(
            "https://www.who.int/europe/news/item/12-12-2023-article"
        ) == URLKind.WEB_PAGE

    # -- Error --
    def test_invalid_url(self):
        with pytest.raises(ValueError, match="must start with"):
            classify_url("ftp://example.com/file.pdf")

    def test_empty_string(self):
        with pytest.raises(ValueError):
            classify_url("")


class TestResolveURL:
    """Tests for resolve_url and internal resolvers."""

    def test_gdrive_resolve(self):
        url = "https://drive.google.com/file/d/1abc-DEF_xyz/view?usp=sharing"
        resolved = resolve_url(url)
        assert resolved == "https://drive.google.com/uc?export=download&id=1abc-DEF_xyz"

    def test_gdrive_open_resolve(self):
        url = "https://drive.google.com/open?id=1abc-DEF_xyz"
        resolved = resolve_url(url)
        assert resolved == "https://drive.google.com/uc?export=download&id=1abc-DEF_xyz"

    def test_github_blob_resolve(self):
        url = "https://github.com/user/repo/blob/main/data/file.csv"
        resolved = resolve_url(url)
        assert resolved == "https://raw.githubusercontent.com/user/repo/main/data/file.csv"

    def test_github_raw_passthrough(self):
        url = "https://raw.githubusercontent.com/user/repo/main/data.csv"
        assert resolve_url(url) == url

    def test_downloadable_passthrough(self):
        url = "https://example.com/report.pdf"
        assert resolve_url(url) == url

    def test_web_page_passthrough(self):
        url = "https://example.com/article"
        assert resolve_url(url) == url

    def test_youtube_passthrough(self):
        url = "https://youtu.be/abc123"
        assert resolve_url(url) == url

    def test_gdrive_bad_url(self):
        with pytest.raises(ValueError, match="cannot extract file ID"):
            _resolve_gdrive("https://drive.google.com/nope")

    def test_github_blob_bad_url(self):
        with pytest.raises(ValueError, match="cannot parse"):
            _resolve_github_blob("https://github.com/nope")


class TestInferExtension:
    """Tests for _infer_extension_from_headers."""

    def test_from_url_path(self):
        assert _infer_extension_from_headers({}, "https://example.com/report.pdf") == ".pdf"

    def test_from_content_type(self):
        headers = {"Content-Type": "application/pdf; charset=utf-8"}
        assert _infer_extension_from_headers(headers, "https://example.com/download") == ".pdf"

    def test_from_content_disposition(self):
        headers = {"Content-Disposition": 'attachment; filename="report.pdf"'}
        assert _infer_extension_from_headers(headers, "https://example.com/download") == ".pdf"

    def test_compound_extension(self):
        assert _infer_extension_from_headers({}, "https://example.com/data.tar.gz") == ".tar.gz"

    def test_fallback_bin(self):
        assert _infer_extension_from_headers({}, "https://example.com/download") == ".bin"


class TestMakeTempFilename:
    """Tests for _make_temp_filename."""

    def test_deterministic(self):
        a = _make_temp_filename("https://example.com/file.pdf", ".pdf")
        b = _make_temp_filename("https://example.com/file.pdf", ".pdf")
        assert a == b

    def test_different_urls(self):
        a = _make_temp_filename("https://a.com/x.pdf", ".pdf")
        b = _make_temp_filename("https://b.com/x.pdf", ".pdf")
        assert a != b

    def test_format(self):
        name = _make_temp_filename("https://example.com/file.pdf", ".pdf")
        assert name.startswith("skplt_")
        assert name.endswith(".pdf")


# ---------------------------------------------------------------------------
# Tests for _archive_handler
# ---------------------------------------------------------------------------

from scikitplot.corpus._archive_handler import (
    extract_archive,
    is_archive,
)


class TestIsArchive:
    """Tests for is_archive."""

    def test_zip(self):
        assert is_archive("data.zip") is True

    def test_tar(self):
        assert is_archive("data.tar") is True

    def test_tar_gz(self):
        assert is_archive("data.tar.gz") is True

    def test_tgz(self):
        assert is_archive("data.tgz") is True

    def test_pdf_not_archive(self):
        assert is_archive("data.pdf") is False

    def test_txt_not_archive(self):
        assert is_archive("data.txt") is False


class TestExtractArchive:
    """Tests for extract_archive with real ZIP files."""

    def test_extract_simple_zip(self, tmp_path):
        """Create a simple ZIP with 2 text files and extract."""
        archive = tmp_path / "test.zip"
        dest = tmp_path / "extracted"

        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("file1.txt", "Hello world")
            zf.writestr("subdir/file2.txt", "Second file")

        files = extract_archive(archive, dest)
        assert len(files) == 2
        assert all(f.exists() for f in files)
        assert (dest / "file1.txt").read_text() == "Hello world"
        assert (dest / "subdir" / "file2.txt").read_text() == "Second file"

    def test_skip_hidden_files(self, tmp_path):
        """Hidden files inside archives should be skipped."""
        archive = tmp_path / "test.zip"
        dest = tmp_path / "extracted"

        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("visible.txt", "OK")
            zf.writestr(".hidden.txt", "SKIP")
            zf.writestr("__MACOSX/file.txt", "SKIP")

        files = extract_archive(archive, dest)
        assert len(files) == 1
        assert files[0].name == "visible.txt"

    def test_zipslip_prevention(self, tmp_path):
        """Path traversal entries should be skipped."""
        archive = tmp_path / "evil.zip"
        dest = tmp_path / "extracted"

        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("safe.txt", "OK")
            zf.writestr("../../etc/passwd", "EVIL")

        files = extract_archive(archive, dest)
        assert len(files) == 1
        assert files[0].name == "safe.txt"
        # The evil file should NOT exist outside dest
        assert not (tmp_path / "etc" / "passwd").exists()

    def test_max_files_limit(self, tmp_path):
        """Archives with too many files should be rejected."""
        archive = tmp_path / "big.zip"
        dest = tmp_path / "extracted"

        with zipfile.ZipFile(archive, "w") as zf:
            for i in range(50):
                zf.writestr(f"file_{i:04d}.txt", f"content {i}")

        with pytest.raises(ValueError, match="max_files"):
            extract_archive(archive, dest, max_files=10)

    def test_extension_filter(self, tmp_path):
        """Only files with supported extensions should be extracted."""
        archive = tmp_path / "mixed.zip"
        dest = tmp_path / "extracted"

        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("data.txt", "text content")
            zf.writestr("image.png", b"PNG data")
            zf.writestr("script.py", "print('hello')")

        files = extract_archive(
            archive, dest,
            supported_extensions=frozenset({".txt", ".py"}),
        )
        names = {f.name for f in files}
        assert "data.txt" in names
        assert "script.py" in names
        assert "image.png" not in names

    def test_unsupported_format(self, tmp_path):
        """Non-archive files should raise ValueError."""
        fake = tmp_path / "not_archive.txt"
        fake.write_text("I am not an archive")
        dest = tmp_path / "extracted"

        with pytest.raises(ValueError, match="not a recognised archive"):
            extract_archive(fake, dest)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
