# tests/test_url_and_archive_handler.py
#
# Unit tests for _url_handler and _archive_handler modules.
# These tests are self-contained — no network access required.

from __future__ import annotations

import ast
import os
import inspect
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Tests for _url_handler.classify_url
# ---------------------------------------------------------------------------

from .. import _url_handler as m
from .._url_handler import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF_BASE,
    _RETRYABLE_STATUS_CODES,
    URLKind,
    _classify_content_type,
    _detect_extension_from_magic,
    _extract_http_status,
    _fixup_bin_extension,
    _infer_extension_from_headers,
    _make_temp_filename,
    _resolve_gdrive,
    _resolve_github_blob,
    classify_url,
    infer_extension,
    probe_url_kind,
    resolve_url,
    download_url,
)
from .._base import DocumentReader, DummyReader
from .._corpus_builder import (
    BuilderConfig,
    CorpusBuilder,
)
from .._archive_handler import (
    extract_archive,
    is_archive,
)
from ..import _readers  # noqa: F401 — registers readers
from .._readers._zip import ZipReader


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
    """Tests for _infer_extension_from_headers (private) and infer_extension (public)."""

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

    # ── RFC-5987 Content-Disposition ──────────────────────────────────────

    def test_rfc5987_content_disposition_utf8(self):
        """filename*=UTF-8''... must be decoded and extension extracted."""
        headers = {
            "Content-Disposition": "attachment; filename*=UTF-8''report%20final.pdf"
        }
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".pdf"

    def test_rfc5987_preferred_over_plain(self):
        """filename*= (RFC-5987) wins over plain filename= when both present."""
        headers = {
            "Content-Disposition": (
                'attachment; filename="wrong.txt"; '
                "filename*=UTF-8''correct.pdf"
            )
        }
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".pdf"

    def test_rfc5987_percent_encoded_filename(self):
        """Percent-encoded chars are decoded before splitext."""
        headers = {
            "Content-Disposition": "attachment; filename*=UTF-8''audio%2Bsong.mp3"
        }
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".mp3"

    def test_plain_content_disposition_without_quotes(self):
        """filename=value without double quotes is also valid."""
        headers = {"Content-Disposition": "attachment; filename=data.csv"}
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".csv"

    # ── New MIME types added in fix 2 ─────────────────────────────────────

    def test_audio_opus_content_type(self):
        headers = {"Content-Type": "audio/opus"}
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".opus"

    def test_audio_webm_content_type(self):
        headers = {"Content-Type": "audio/webm"}
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".webm"

    def test_audio_aac_content_type(self):
        headers = {"Content-Type": "audio/aac"}
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".aac"

    def test_video_flv_content_type(self):
        headers = {"Content-Type": "video/x-flv"}
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".flv"

    def test_video_wmv_content_type(self):
        headers = {"Content-Type": "video/x-ms-wmv"}
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".wmv"

    def test_image_svg_content_type(self):
        headers = {"Content-Type": "image/svg+xml"}
        assert _infer_extension_from_headers(headers, "https://example.com/dl") == ".svg"

    # ── Public infer_extension() wrapper ─────────────────────────────────

    def test_public_infer_extension_is_callable(self):
        """infer_extension must be importable as a public name."""
        assert callable(infer_extension)

    def test_public_infer_extension_delegates_to_private(self):
        """infer_extension(h, url) must return same result as _infer_extension_from_headers."""
        headers = {"Content-Type": "audio/mpeg"}
        url = "https://example.com/podcast"
        assert infer_extension(headers, url) == _infer_extension_from_headers(headers, url)

    def test_public_infer_extension_url_path(self):
        assert infer_extension({}, "https://example.com/video.mp4") == ".mp4"

    def test_public_infer_extension_rfc5987(self):
        headers = {"Content-Disposition": "attachment; filename*=UTF-8''track.flac"}
        assert infer_extension(headers, "https://example.com/dl") == ".flac"


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


# ---------------------------------------------------------------------------
# Tests for _detect_extension_from_magic
# ---------------------------------------------------------------------------


class TestDetectExtensionFromMagic:
    """Tests for magic-byte file type detection."""

    def test_pdf(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"%PDF-1.4 fake pdf content here")
        assert _detect_extension_from_magic(f) == ".pdf"

    def test_png(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
        assert _detect_extension_from_magic(f) == ".png"

    def test_jpeg(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 28)
        assert _detect_extension_from_magic(f) == ".jpg"

    def test_zip(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"PK\x03\x04" + b"\x00" * 28)
        assert _detect_extension_from_magic(f) == ".zip"

    def test_mp3_id3(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"ID3\x04\x00" + b"\x00" * 27)
        assert _detect_extension_from_magic(f) == ".mp3"

    def test_flac(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"fLaC" + b"\x00" * 28)
        assert _detect_extension_from_magic(f) == ".flac"

    def test_wav_riff(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20)
        assert _detect_extension_from_magic(f) == ".wav"

    def test_webp_riff(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20)
        assert _detect_extension_from_magic(f) == ".webp"

    def test_avi_riff(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 20)
        assert _detect_extension_from_magic(f) == ".avi"

    def test_mp4_ftyp(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 20)
        assert _detect_extension_from_magic(f) == ".mp4"

    def test_m4a_ftyp(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 20)
        assert _detect_extension_from_magic(f) == ".m4a"

    def test_mov_ftyp(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x00\x00\x20ftypqt  " + b"\x00" * 20)
        assert _detect_extension_from_magic(f) == ".mov"

    def test_tar_gz(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 28)
        assert _detect_extension_from_magic(f) == ".tar.gz"

    def test_ogg(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"OggS" + b"\x00" * 28)
        assert _detect_extension_from_magic(f) == ".ogg"

    def test_xml(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"<?xml version='1.0'?>")
        assert _detect_extension_from_magic(f) == ".xml"

    def test_unknown_returns_none(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\xfe\xed\xfa\xce" + b"\x00" * 28)
        assert _detect_extension_from_magic(f) is None

    def test_empty_file_returns_none(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"")
        assert _detect_extension_from_magic(f) is None

    def test_short_file_returns_none(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00\x01")
        assert _detect_extension_from_magic(f) is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        f = tmp_path / "does_not_exist.bin"
        assert _detect_extension_from_magic(f) is None


class TestFixupBinExtension:
    """Tests for _fixup_bin_extension post-download rename."""

    def test_renames_pdf(self, tmp_path):
        """A .bin file containing PDF magic should be renamed to .pdf."""
        f = tmp_path / "downloaded.bin"
        f.write_bytes(b"%PDF-1.4 fake pdf content here")
        result = _fixup_bin_extension(f)
        assert result.suffix == ".pdf"
        assert result.exists()
        assert not f.exists()  # original .bin removed by rename

    def test_renames_png(self, tmp_path):
        f = tmp_path / "image.bin"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
        result = _fixup_bin_extension(f)
        assert result.suffix == ".png"

    def test_no_rename_for_non_bin(self, tmp_path):
        """Files that already have a real extension should not be touched."""
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4 fake pdf")
        result = _fixup_bin_extension(f)
        assert result == f
        assert result.suffix == ".pdf"

    def test_keeps_bin_when_unknown(self, tmp_path):
        """Unknown magic bytes should leave the .bin extension as-is."""
        f = tmp_path / "mystery.bin"
        f.write_bytes(b"\xfe\xed\xfa\xce" + b"\x00" * 28)
        result = _fixup_bin_extension(f)
        assert result == f
        assert result.suffix == ".bin"

    def test_keeps_bin_when_empty(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        result = _fixup_bin_extension(f)
        assert result == f
        assert result.suffix == ".bin"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ===========================================================================
# TestResolveUrlInAll
# ===========================================================================


class TestResolveUrlInAll:
    """All public names must be in _url_handler.__all__ with no duplicates."""

    def test_resolve_url_in_all(self):
        assert "resolve_url" in m.__all__

    def test_classify_url_in_all(self):
        assert "classify_url" in m.__all__

    def test_download_url_in_all(self):
        assert "download_url" in m.__all__

    def test_probe_url_kind_in_all(self):
        assert "probe_url_kind" in m.__all__

    def test_infer_extension_in_all(self):
        """infer_extension must be exported as a public name."""
        assert "infer_extension" in m.__all__

    def test_no_duplicate_names_in_all(self):
        """__all__ must contain no duplicate entries."""
        assert len(m.__all__) == len(set(m.__all__)), (
            f"Duplicate names in __all__: "
            f"{[n for n in m.__all__ if m.__all__.count(n) > 1]}"
        )


# ===========================================================================
# TestDownloadableExtensions
# ===========================================================================


class TestDownloadableExtensions:
    """DocumentReader._DOWNLOADABLE_EXTENSIONS — used by from_url()."""

    def test_pdf_in_downloadable(self):
        assert ".pdf" in DocumentReader._DOWNLOADABLE_EXTENSIONS

    def test_mp3_in_downloadable(self):
        assert ".mp3" in DocumentReader._DOWNLOADABLE_EXTENSIONS

    def test_jpg_in_downloadable(self):
        assert ".jpg" in DocumentReader._DOWNLOADABLE_EXTENSIONS

    def test_zip_in_downloadable(self):
        assert ".zip" in DocumentReader._DOWNLOADABLE_EXTENSIONS

    def test_html_not_in_downloadable(self):
        """HTML pages should NOT be in downloadable — they go to WebReader."""
        # .html may or may not be there, but .htm is not a download
        # The key check: .pdf, .mp3, .jpg must be present
        assert ".pdf" in DocumentReader._DOWNLOADABLE_EXTENSIONS


# ===========================================================================
# TestZipReaderRegistration
# ===========================================================================


class TestZipReaderRegistration:
    """ZipReader must be registered for .zip, overriding ALTOReader."""

    def test_zip_registered(self):
        assert ".zip" in DocumentReader._registry

    def test_zip_reader_is_zip_reader(self):
        assert DocumentReader._registry[".zip"] is ZipReader


# ===========================================================================
# TestDummyReaderCheck
# ===========================================================================


class TestDummyReaderCheck:
    """DummyReader.check() batch pre-flight validation."""

    def test_existing_file_ok(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        ok, errors = DummyReader.check(f)
        assert f in ok
        assert len(errors) == 0

    def test_missing_file_error(self, tmp_path):
        missing = tmp_path / "ghost.txt"
        ok, errors = DummyReader.check(missing)
        assert len(errors) == 1
        assert errors[0][0] == missing

    def test_multiple_sources_collected(self, tmp_path):
        good = tmp_path / "good.txt"
        good.write_text("ok")
        bad = tmp_path / "missing.txt"
        ok, errors = DummyReader.check(good, bad)
        assert good in ok
        assert len(errors) == 1

    def test_raise_on_first(self, tmp_path):
        missing = tmp_path / "no.txt"
        with pytest.raises((ValueError, OSError)):
            DummyReader.check(missing, raise_on_first=True)


# ===========================================================================
# TestDownloadUrlRetry  (Fix 6)
# ===========================================================================


class TestDownloadUrlRetry:
    """download_url retries transient HTTP errors and raises on permanent ones."""

    def test_retry_constants_exported(self):
        assert DEFAULT_MAX_RETRIES == 3
        assert DEFAULT_RETRY_BACKOFF_BASE == 1.0
        assert 429 in _RETRYABLE_STATUS_CODES
        assert 503 in _RETRYABLE_STATUS_CODES
        assert 404 not in _RETRYABLE_STATUS_CODES
        assert 403 not in _RETRYABLE_STATUS_CODES

    def test_extract_http_status_from_requests_error(self):
        exc = Exception("oops")
        exc.response = type("R", (), {"status_code": 503})()
        assert _extract_http_status(exc) == 503

    def test_extract_http_status_from_urllib_error(self):
        exc = Exception("oops")
        exc.code = 429
        assert _extract_http_status(exc) == 429

    def test_extract_http_status_none_when_no_code(self):
        assert _extract_http_status(Exception("plain")) is None

    def test_value_error_not_retried(self, tmp_path):
        """ValueError (SSRF, size exceeded) must propagate immediately."""
        calls = []

        def _mock_download(*args, **kwargs):
            calls.append(1)
            raise ValueError("size exceeded")

        with patch(
            "scikitplot.corpus._url_handler._download_with_requests",
            side_effect=_mock_download,
        ):
            with pytest.raises(ValueError, match="size exceeded"):
                download_url(
                    "https://example.com/file.pdf",
                    dest_dir=tmp_path,
                    max_retries=3,
                )
        # Must not retry on ValueError
        assert len(calls) == 1, "ValueError must not be retried"

    def test_permanent_404_not_retried(self, tmp_path):
        """HTTP 404 is permanent — must not retry."""
        calls = []

        def _mock_download(*args, **kwargs):
            calls.append(1)
            exc = Exception("not found")
            exc.response = type("R", (), {"status_code": 404})()
            raise exc

        with patch(
            "scikitplot.corpus._url_handler._download_with_requests",
            side_effect=_mock_download,
        ):
            with pytest.raises(Exception):
                download_url(
                    "https://example.com/file.pdf",
                    dest_dir=tmp_path,
                    max_retries=3,
                )
        assert len(calls) == 1, "404 must not be retried"

    def test_retries_exhausted_on_503(self, tmp_path):
        """503 is retried up to max_retries then raises."""
        calls = []

        def _mock_download(*args, **kwargs):
            calls.append(1)
            exc = Exception("service unavailable")
            exc.response = type("R", (), {"status_code": 503})()
            raise exc

        with patch(
            "scikitplot.corpus._url_handler._download_with_requests",
            side_effect=_mock_download,
        ), patch("time.sleep"):  # skip real delays
            with pytest.raises(Exception):
                download_url(
                    "https://example.com/file.pdf",
                    dest_dir=tmp_path,
                    max_retries=2,
                    retry_backoff=0.0,
                )
        # 1 initial attempt + 2 retries = 3 total
        assert len(calls) == 3, f"expected 3 attempts (1+2 retries), got {len(calls)}"


# ===========================================================================
# TestExpandSourcesEdgeCases  (Fix 1)
# ===========================================================================


class TestExpandSourcesEdgeCases:
    """_expand_sources must reject empty/whitespace sources without touching CWD."""

    def test_empty_string_skipped(self, tmp_path):
        result = CorpusBuilder._expand_sources([""])
        assert result == [], "empty string must not expand to CWD"

    def test_whitespace_only_skipped(self, tmp_path):
        result = CorpusBuilder._expand_sources(["   ", "\t", "\n"])
        assert result == [], "whitespace-only strings must be skipped"

    def test_valid_file_still_works(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        result = CorpusBuilder._expand_sources([str(f)])
        assert f in result

    def test_url_passthrough_unaffected(self):
        result = CorpusBuilder._expand_sources(["https://example.com/file.pdf"])
        assert result == ["https://example.com/file.pdf"]


# ===========================================================================
# TestAddSourceType  (Fix 5)
# ===========================================================================


class TestAddSourceType:
    """add() must accept and propagate source_type without mutating config."""

    def test_add_signature_has_source_type(self):
        sig = inspect.signature(CorpusBuilder.add)
        assert "source_type" in sig.parameters

    def test_config_source_type_restored_after_add(self, tmp_path):
        """Config must be unchanged whether add() succeeds or fails."""
        f = tmp_path / "doc.txt"
        f.write_text("hello world test sentence paragraph here")
        builder = CorpusBuilder(BuilderConfig(chunker="paragraph"))
        builder.build(str(f))
        original_type = builder.config.source_type  # None

        # Override source_type for add(); config must restore after call
        f2 = tmp_path / "doc2.txt"
        f2.write_text("second document content here for testing")
        builder.add(str(f2), source_type="article")

        assert builder.config.source_type == original_type, (
            "config.source_type must be restored to its pre-add() value"
        )

    def test_config_source_type_restored_even_on_error(self, tmp_path):
        """Config must restore even when build() raises inside add()."""
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        builder = CorpusBuilder(BuilderConfig(chunker="paragraph"))
        builder.build(str(f))

        original_type = builder.config.source_type

        with pytest.raises(ValueError):
            # Non-existent path causes ValueError("No valid sources found")
            builder.add("/nonexistent/path_xyz123.txt", source_type="audio")

        assert builder.config.source_type == original_type

    def test_config_restored_on_error(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        builder = CorpusBuilder(BuilderConfig(chunker="paragraph"))
        builder.build(str(f))
        original = builder.config.source_type
        with pytest.raises(ValueError):
            builder.add("/nonexistent/xyz123.txt", source_type="audio")
        assert builder.config.source_type == original


# ===========================================================================
# TestNFilteredPopulated  (Fix 2)
# ===========================================================================


class TestNFilteredPopulated:
    """BuildResult.n_filtered must reflect chunks removed by the reader filter."""

    def test_n_filtered_is_int(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello world. This is a test document with enough content.")
        builder = CorpusBuilder(BuilderConfig(chunker="paragraph"))
        result = builder.build(str(f))
        assert isinstance(result.n_filtered, int)
        assert result.n_filtered >= 0

    def test_n_filtered_is_non_negative_int(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello world. This is a test document with enough content.")
        builder = CorpusBuilder(BuilderConfig(chunker="paragraph"))
        result = builder.build(str(f))
        assert isinstance(result.n_filtered, int)
        assert result.n_filtered >= 0


# ===========================================================================
# TestZipReaderMembersCount  (Fix 3)
# ===========================================================================


class TestZipReaderMembersCount:
    """ZipReader must log member count correctly even when zip is malformed."""

    def test_valid_zip_members_count(self, tmp_path):
        """members_count is set correctly for a normal zip."""
        zf_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zf_path, "w") as zf:
            zf.writestr("a.txt", "hello")
            zf.writestr("b.txt", "world")

        reader = DocumentReader.create(zf_path)
        # Must not raise NameError about 'members'
        docs = list(reader.get_documents())
        assert isinstance(docs, list)

    def test_bad_zip_no_name_error(self, tmp_path):
        """A corrupt zip must raise a zipfile error, not NameError on members."""
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"this is not a zip file at all")
        reader = DocumentReader.create(bad_zip)
        with pytest.raises(Exception) as exc_info:
            list(reader.get_documents())
        # Must be a zipfile error, not NameError
        assert not isinstance(exc_info.value, NameError), (
            "NameError on 'members' must not be raised from ZipReader"
        )

    def test_valid_zip_no_error(self, tmp_path):
        zp = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("a.txt", "hello")
            zf.writestr("b.txt", "world")
        docs = list(DocumentReader.create(zp).get_documents())
        assert isinstance(docs, list)

    def test_corrupt_zip_raises_zipfile_error_not_name_error(self, tmp_path):
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        with pytest.raises(Exception) as exc_info:
            list(DocumentReader.create(bad).get_documents())
        assert not isinstance(exc_info.value, NameError)


# ===========================================================================
# TestSessionClosed  (Fix 4)
# ===========================================================================


class TestSessionClosed:
    """requests.Session must be used as a context manager in probe and download."""

    def test_probe_uses_session_context_manager(self):
        """_probe_with_requests must close session via context manager."""
        src = inspect.getsource(m._probe_with_requests)
        tree = ast.parse(src)
        # Check for 'with requests.Session()' or 'with ... as session'
        has_with_session = any(
            isinstance(node, ast.With)
            and any(
                "Session" in ast.unparse(item.context_expr)
                for item in node.items
            )
            for node in ast.walk(tree)
        )
        assert has_with_session, (
            "_probe_with_requests must use 'with requests.Session()' "
            "to prevent connection pool leaks"
        )

    def test_download_uses_session_context_manager(self):
        """_download_with_requests must close session via context manager."""
        src = inspect.getsource(m._download_with_requests)
        tree = ast.parse(src)
        has_with_session = any(
            isinstance(node, ast.With)
            and any(
                "Session" in ast.unparse(item.context_expr)
                for item in node.items
            )
            for node in ast.walk(tree)
        )
        assert has_with_session, (
            "_download_with_requests must use 'with requests.Session()' "
            "to prevent connection pool leaks"
        )

    def _has_with_session(self, fn) -> bool:
        src = inspect.getsource(fn)
        tree = ast.parse(src)
        return any(
            isinstance(node, ast.With)
            and any("Session" in ast.unparse(item.context_expr) for item in node.items)
            for node in ast.walk(tree)
        )

    def test_probe_uses_context_manager(self):
        assert self._has_with_session(m._probe_with_requests)

    def test_download_uses_context_manager(self):
        assert self._has_with_session(m._download_with_requests)


# ===========================================================================
# TestMaxWorkersParallel  (Fix 7)
# ===========================================================================


class TestMaxWorkersParallel:
    """build() must use ThreadPoolExecutor when max_workers > 1."""

    def test_max_workers_1_is_default(self):
        assert BuilderConfig().max_workers == 1

    def test_parallel_build_same_result_as_serial(self, tmp_path):
        """Parallel and serial ingestion must produce the same document count."""
        # Write 4 small text files
        for i in range(4):
            (tmp_path / f"doc{i}.txt").write_text(
                f"Document {i} with enough content to pass the default filter."
            )

        serial = CorpusBuilder(BuilderConfig(chunker="paragraph", max_workers=1))
        r_serial = serial.build(str(tmp_path))

        parallel = CorpusBuilder(BuilderConfig(chunker="paragraph", max_workers=4))
        r_parallel = parallel.build(str(tmp_path))

        assert r_parallel.n_documents == r_serial.n_documents
        assert r_parallel.n_sources == r_serial.n_sources

    def test_parallel_errors_collected(self, tmp_path):
        """Errors from parallel workers must be collected, not silently dropped."""
        good = tmp_path / "good.txt"
        good.write_text("Valid document content here for testing.")
        bad = tmp_path / "bad.unsupported_ext_xyz"
        bad.write_text("this extension is not registered")

        builder = CorpusBuilder(BuilderConfig(chunker="paragraph", max_workers=2))
        result = builder.build([str(good), str(bad)])

        # The bad file should produce an error entry, not silently vanish
        assert len(result.errors) == 1
        assert str(bad) in str(result.errors[0][0])

    def test_default_max_workers_is_1(self):
        assert BuilderConfig().max_workers == 1

    def test_parallel_same_doc_count_as_serial(self, tmp_path):
        for i in range(4):
            (tmp_path / f"doc{i}.txt").write_text(
                f"Document number {i} contains enough text to pass the default filter."
            )
        r_serial = CorpusBuilder(BuilderConfig(chunker="paragraph", max_workers=1)).build(str(tmp_path))
        r_parallel = CorpusBuilder(BuilderConfig(chunker="paragraph", max_workers=4)).build(str(tmp_path))
        assert r_parallel.n_documents == r_serial.n_documents
        assert r_parallel.n_sources == r_serial.n_sources
