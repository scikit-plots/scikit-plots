# tests/test_url_and_archive_handler.py
#
# Unit tests for _url_handler and _archive_handler modules.
# These tests are self-contained — no network access required.

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Tests for _url_handler.classify_url
# ---------------------------------------------------------------------------

from scikitplot.corpus._url_handler import (
    URLKind,
    _classify_content_type,
    _infer_extension_from_headers,
    _make_temp_filename,
    _resolve_gdrive,
    _resolve_github_blob,
    classify_url,
    probe_url_kind,
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
        result = classify_url("https://example.com/page.html")
        assert result == URLKind.WEB_PAGE

    def test_web_page_news(self):
        assert classify_url(
            "https://www.who.int/europe/news/item/12-12-2023-article"
        ) == URLKind.WEB_PAGE

    # -- API endpoints without extension (stage 1 → WEB_PAGE) --
    def test_api_endpoint_no_extension_is_webpage_before_probe(self):
        """API endpoint with no extension classifies as WEB_PAGE (stage 1 only)."""
        assert classify_url(
            "https://iris.who.int/server/api/core/bitstreams/abc/content"
        ) == URLKind.WEB_PAGE

    def test_api_download_no_extension_is_webpage_before_probe(self):
        """Download endpoints without extension are WEB_PAGE at stage 1."""
        assert classify_url("https://example.com/download") == URLKind.WEB_PAGE

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

    def test_image_content_type(self):
        headers = {"Content-Type": "image/jpeg"}
        assert _infer_extension_from_headers(headers, "https://example.com/img") == ".jpg"

    def test_audio_content_type(self):
        headers = {"Content-Type": "audio/mpeg"}
        assert _infer_extension_from_headers(headers, "https://example.com/podcast") == ".mp3"

    def test_video_content_type(self):
        headers = {"Content-Type": "video/mp4"}
        assert _infer_extension_from_headers(headers, "https://example.com/clip") == ".mp4"

    def test_zip_content_type(self):
        headers = {"Content-Type": "application/zip"}
        assert _infer_extension_from_headers(headers, "https://example.com/bundle") == ".zip"


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
# Tests for _classify_content_type (unit tests for the new helper)
# ---------------------------------------------------------------------------


class TestClassifyContentType:
    """Tests for _classify_content_type mapping MIME → URLKind."""

    def test_pdf_is_downloadable(self):
        assert _classify_content_type("application/pdf") == URLKind.DOWNLOADABLE

    def test_zip_is_downloadable(self):
        assert _classify_content_type("application/zip") == URLKind.DOWNLOADABLE

    def test_image_jpeg_is_downloadable(self):
        assert _classify_content_type("image/jpeg") == URLKind.DOWNLOADABLE

    def test_image_png_is_downloadable(self):
        assert _classify_content_type("image/png") == URLKind.DOWNLOADABLE

    def test_audio_mpeg_is_downloadable(self):
        assert _classify_content_type("audio/mpeg") == URLKind.DOWNLOADABLE

    def test_audio_wav_is_downloadable(self):
        assert _classify_content_type("audio/wav") == URLKind.DOWNLOADABLE

    def test_video_mp4_is_downloadable(self):
        assert _classify_content_type("video/mp4") == URLKind.DOWNLOADABLE

    def test_video_webm_is_downloadable(self):
        assert _classify_content_type("video/webm") == URLKind.DOWNLOADABLE

    def test_text_csv_is_downloadable(self):
        assert _classify_content_type("text/csv") == URLKind.DOWNLOADABLE

    def test_text_plain_is_downloadable(self):
        # Plain text (transcripts, code) routes to TextReader not WebReader.
        assert _classify_content_type("text/plain") == URLKind.DOWNLOADABLE

    def test_json_is_downloadable(self):
        assert _classify_content_type("application/json") == URLKind.DOWNLOADABLE

    def test_octet_stream_is_downloadable(self):
        assert _classify_content_type("application/octet-stream") == URLKind.DOWNLOADABLE

    def test_xlsx_is_downloadable(self):
        assert _classify_content_type(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ) == URLKind.DOWNLOADABLE

    def test_html_is_web_page(self):
        assert _classify_content_type("text/html") == URLKind.WEB_PAGE

    def test_empty_is_web_page(self):
        # Empty content-type → fail-safe: treat as web page.
        assert _classify_content_type("") == URLKind.WEB_PAGE

    def test_unknown_mime_is_web_page(self):
        assert _classify_content_type("x-custom/unknown-type") == URLKind.WEB_PAGE


# ---------------------------------------------------------------------------
# Tests for probe_url_kind (mocked — no real network calls)
# ---------------------------------------------------------------------------


class TestProbeUrlKind:
    """Tests for probe_url_kind with mocked HTTP responses."""

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="must start with"):
            probe_url_kind("ftp://example.com/file")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            probe_url_kind("")

    def _make_mock_response(self, content_type: str, status_code: int = 200):
        """Build a minimal mock requests.Response."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.headers = {"Content-Type": content_type}
        mock_resp.url = "https://example.com/resource"
        return mock_resp

    def test_pdf_content_type_returns_downloadable(self):
        """An API endpoint returning application/pdf → DOWNLOADABLE."""
        mock_resp = self._make_mock_response("application/pdf")
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="application/pdf"),
        ):
            result = probe_url_kind("https://iris.who.int/api/bitstreams/abc/content")
        assert result == URLKind.DOWNLOADABLE

    def test_image_jpeg_returns_downloadable(self):
        """An endpoint serving image/jpeg → DOWNLOADABLE."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="image/jpeg"),
        ):
            result = probe_url_kind("https://example.com/api/image")
        assert result == URLKind.DOWNLOADABLE

    def test_audio_mpeg_returns_downloadable(self):
        """An endpoint serving audio/mpeg → DOWNLOADABLE."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="audio/mpeg"),
        ):
            result = probe_url_kind("https://archive.org/stream/podcast")
        assert result == URLKind.DOWNLOADABLE

    def test_video_mp4_returns_downloadable(self):
        """An endpoint serving video/mp4 → DOWNLOADABLE."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="video/mp4"),
        ):
            result = probe_url_kind("https://cdn.example.com/stream")
        assert result == URLKind.DOWNLOADABLE

    def test_text_html_returns_web_page(self):
        """An endpoint serving text/html → WEB_PAGE."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="text/html"),
        ):
            result = probe_url_kind("https://example.com/article")
        assert result == URLKind.WEB_PAGE

    def test_empty_content_type_falls_back_to_web_page(self):
        """When probe returns empty string → fail-safe WEB_PAGE."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value=""),
        ):
            result = probe_url_kind("https://example.com/resource")
        assert result == URLKind.WEB_PAGE

    def test_probe_network_error_falls_back_to_web_page(self):
        """On network error in probe → fail-safe WEB_PAGE (no exception raised)."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value=""),
        ):
            result = probe_url_kind("https://example.com/resource")
        assert result == URLKind.WEB_PAGE

    def test_skip_ssrf_check_skips_validation(self):
        """skip_ssrf_check=True must not call _validate_url_security."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security") as mock_validate,
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="application/pdf"),
        ):
            probe_url_kind(
                "https://example.com/content",
                skip_ssrf_check=True,
            )
        mock_validate.assert_not_called()

    def test_ssrf_check_called_by_default(self):
        """SSRF check is called when skip_ssrf_check=False (default)."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security") as mock_validate,
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="text/html"),
        ):
            probe_url_kind("https://example.com/content")
        mock_validate.assert_called_once()

    def test_zip_content_type_returns_downloadable(self):
        """An endpoint serving application/zip → DOWNLOADABLE."""
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="application/zip"),
        ):
            result = probe_url_kind("https://example.com/bundle")
        assert result == URLKind.DOWNLOADABLE

    def test_who_iris_pdf_endpoint_pattern(self):
        """Simulate the exact WHO IRIS PDF endpoint from the notebook TODO."""
        # https://iris.who.int/server/api/core/bitstreams/7ad66865-7f23-4485-8cf5-7b3d78bdf4f9/content
        # → Content-Type: application/pdf → DOWNLOADABLE → PDFReader
        url = (
            "https://iris.who.int/server/api/core/bitstreams/"
            "7ad66865-7f23-4485-8cf5-7b3d78bdf4f9/content"
        )
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="application/pdf"),
        ):
            result = probe_url_kind(url)
        assert result == URLKind.DOWNLOADABLE

    def test_who_iris_image_endpoint_pattern(self):
        """Simulate the WHO IRIS image endpoint from the notebook TODO."""
        # https://iris.who.int/server/api/core/bitstreams/d57241c0-512d-4cfc-9ead-91a83eea83f0/content
        # → Content-Type: image/jpeg → DOWNLOADABLE → ImageReader
        url = (
            "https://iris.who.int/server/api/core/bitstreams/"
            "d57241c0-512d-4cfc-9ead-91a83eea83f0/content"
        )
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="image/jpeg"),
        ):
            result = probe_url_kind(url)
        assert result == URLKind.DOWNLOADABLE

    def test_archive_org_audio_endpoint_pattern(self):
        """Simulate the archive.org MP3 endpoint from the notebook TODO."""
        # https://archive.org/download/makingcon-241016/makingcon-241016_promo.mp3
        # Already has .mp3 extension → classify_url would catch it.
        # Test the extensionless variant: https://archive.org/stream/makingcon-241016
        url = "https://archive.org/stream/makingcon-241016"
        with (
            patch("scikitplot.corpus._url_handler._validate_url_security"),
            patch("scikitplot.corpus._url_handler._probe_with_requests",
                  return_value="audio/mpeg"),
        ):
            result = probe_url_kind(url)
        assert result == URLKind.DOWNLOADABLE


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
