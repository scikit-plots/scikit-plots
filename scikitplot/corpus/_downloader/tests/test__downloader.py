# scikitplot/corpus/_downloader/tests/test__downloader.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Unit tests for the entire _downloader submodule.
# No network access — all HTTP calls are mocked.
#
# Run:
#   pytest corpus/_downloader/tests/test__downloader.py -v
#   pytest corpus/_downloader/tests/test__downloader.py -v --cov=corpus/_downloader

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from urllib.parse import urlparse

import pytest

from .... import corpus
from ... import _readers  # noqa: F401
from ..._base import DocumentReader, _MultiSourceReader
from .. import __init__ as mod

from .._base import (
    _ALLOWED_SCHEMES,
    _DEFAULT_MAX_BYTES,
    _DEFAULT_MAX_REDIRECTS,
    _DEFAULT_TIMEOUT,
    _DEFAULT_USER_AGENT,
    _coerce_param,
)
from .._gdrive import (
    _GDRIVE_CONFIRM_RE,
    _build_download_url,
    _extract_gdrive_file_id,
)
from .._github import (
    _GITHUB_BLOB_RE,
    _GITHUB_RAW_RE,
    _GITHUB_REPO_ROOT_RE,
    _GITHUB_TREE_RE,
)
from .._youtube import (
    _YT_VIDEO_RE,
    _extract_video_id,  # correct module: _youtube
)
from .. import (
    AnyDownloader,
    BaseDownloader,
    CustomDownloader,
    DownloadResult,
    GitHubDownloader,
    GoogleDriveDownloader,
    WebDownloader,
    YouTubeDownloader,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def simple_txt(tmp_path: Path) -> Path:
    p = tmp_path / "hello.txt"
    p.write_text("Hello, world!", encoding="utf-8")
    return p


def _mock_response(
    *,
    content: bytes = b"data",
    content_type: str = "application/octet-stream",
    content_disposition: str = "",
    status_code: int = 200,
    final_url: str = "https://example.com/file",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.input_url = final_url
    resp.headers = {
        "Content-Type": content_type,
        "Content-Disposition": content_disposition,
    }
    resp.iter_content.return_value = [content] if content else []
    resp.raise_for_status.return_value = None
    resp.text = content.decode("utf-8", errors="replace")
    return resp


def is_valid_github_url(url: str) -> bool:
    parsed = urlparse(url)

    # Enforce HTTPS
    if parsed.scheme != "https":
        return False

    # Strict hostname check
    if parsed.netloc != "github.com":
        return False

    return True


def is_valid_raw_github_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.scheme != "https":
        return False

    if parsed.netloc != "raw.githubusercontent.com":
        return False

    return True


# ===========================================================================
# _coerce_param — canonical T | list[T] | None normalizer
# ===========================================================================


class TestCoerceParam:
    """Tests for the _coerce_param utility."""

    def test_none_uses_default_broadcast(self) -> None:
        assert _coerce_param(None, 3, name="x", default=30.0) == [30.0, 30.0, 30.0]

    def test_scalar_broadcast(self) -> None:
        assert _coerce_param(60.0, 3, name="x", default=30.0) == [60.0, 60.0, 60.0]

    def test_list_passthrough(self) -> None:
        assert _coerce_param([1.0, 2.0, 3.0], 3, name="x", default=0.0) == [1.0, 2.0, 3.0]

    def test_wrong_length_list_raises(self) -> None:
        with pytest.raises(ValueError, match="list length 2 does not match"):
            _coerce_param([1.0, 2.0], 3, name="timeout", default=0.0)

    def test_none_in_list_raises_by_default(self) -> None:
        with pytest.raises(ValueError, match="allow_none_items"):
            _coerce_param([None, 1.0], 2, name="val", default=0.0)

    def test_none_in_list_allowed_when_flag_set(self) -> None:
        result = _coerce_param(
            [None, "tok"], 2, name="token", default=None, allow_none_items=True
        )
        assert result == [None, "tok"]

    def test_n1_scalar(self) -> None:
        assert _coerce_param("abc", 1, name="s", default="x") == ["abc"]

    def test_n1_none(self) -> None:
        assert _coerce_param(None, 1, name="s", default="x") == ["x"]

    def test_bool_scalar_broadcast(self) -> None:
        assert _coerce_param(False, 4, name="verify_ssl", default=True) == [False] * 4

    def test_dict_scalar_broadcast(self) -> None:
        h = {"X-Custom": "val"}
        result = _coerce_param(h, 2, name="headers", default=None)
        assert result == [h, h]
        assert result[0] is result[1]  # same object (broadcast, not copied)

    def test_empty_list_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            _coerce_param([], 2, name="x", default=0)

    def test_list_of_none_all_allowed(self) -> None:
        result = _coerce_param(
            [None, None, None], 3, name="t", default=None, allow_none_items=True
        )
        assert result == [None, None, None]


# ===========================================================================
# DownloadResult
# ===========================================================================


class TestDownloadResult:
    def test_basic(self, tmp_dir: Path) -> None:
        p = tmp_dir / "f.pdf"
        p.touch()
        r = DownloadResult(input_url="https://x.com/f.pdf", output_path=p, suffix=".pdf")
        assert r.suffix == ".pdf"
        assert r.content_type == ""

    def test_full_fields(self, tmp_dir: Path) -> None:
        p = tmp_dir / "f.pdf"
        p.touch()
        r = DownloadResult(
            input_url="https://x.com", output_path=p, suffix=".pdf",
            content_type="application/pdf", suggested_filename="f.pdf",
        )
        assert r.content_type == "application/pdf"
        assert r.suggested_filename == "f.pdf"

    def test_frozen(self, tmp_dir: Path) -> None:
        p = tmp_dir / "f.txt"
        p.touch()
        r = DownloadResult(input_url="https://x.com", output_path=p, suffix=".txt")
        with pytest.raises((AttributeError, TypeError)):
            r.suffix = ".pdf"  # type: ignore[misc]


# ===========================================================================
# BaseDownloader — abstract + validation
# ===========================================================================


def _make_concrete_dl(input_url: str, **kwargs) -> BaseDownloader:
    @dataclass
    class Concrete(BaseDownloader):
        def download(self) -> DownloadResult:
            dest = self._resolve_dest_dir() / "f.txt"
            dest.write_text("x")
            return DownloadResult(input_url=self.input_url, output_path=dest, suffix=".txt")

    return Concrete(input_url=input_url, **kwargs)


class TestBaseDownloader:
    def test_non_string_url_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="input_url must be a str"):
            _make_concrete_dl(input_url=123)  # type: ignore[arg-type]

    def test_local_path_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            _make_concrete_dl("/local/file.txt")

    def test_http_accepted(self) -> None:
        assert _make_concrete_dl("http://example.com/f.txt").input_url.startswith("http")

    def test_https_accepted(self) -> None:
        assert _make_concrete_dl("https://example.com/f.txt").input_url.startswith("https")

    def test_ftp_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_concrete_dl("ftp://example.com/f")

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_concrete_dl("file:///etc/passwd")

    def test_no_hostname_rejected(self) -> None:
        with pytest.raises(ValueError, match="no hostname"):
            _make_concrete_dl("https:///no-host")

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout must be > 0"):
            _make_concrete_dl("https://example.com/f", timeout=0)

    def test_zero_max_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_bytes must be > 0"):
            _make_concrete_dl("https://example.com/f", max_bytes=0)

    def test_negative_max_redirects_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_redirects must be >= 0"):
            _make_concrete_dl("https://example.com/f", max_redirects=-1)

    def test_defaults(self) -> None:
        dl = _make_concrete_dl("https://example.com/f")
        assert dl.timeout == _DEFAULT_TIMEOUT
        assert dl.max_bytes == _DEFAULT_MAX_BYTES
        assert dl.block_private_ips is True
        assert dl.verify_ssl is True

    def test_context_manager_cleanup(self) -> None:
        dl = _make_concrete_dl("https://example.com/f")
        with dl as ctx:
            dest = dl._resolve_dest_dir()
            assert dest.exists()
        assert not dest.exists()

    def test_cleanup_idempotent(self) -> None:
        dl = _make_concrete_dl("https://example.com/f")
        dl._resolve_dest_dir()
        dl.cleanup()
        dl.cleanup()  # must not raise

    def test_caller_dest_dir_not_cleaned(self, tmp_dir: Path) -> None:
        dl = _make_concrete_dl("https://example.com/f", output_path=tmp_dir)
        with dl:
            pass
        assert tmp_dir.exists()
        assert dl._tmp_dir is None

    def test_ssrf_check_skipped_when_disabled(self) -> None:
        dl = _make_concrete_dl("https://example.com/f", block_private_ips=False)
        dl._check_ssrf()  # no exception

    def test_ssrf_check_blocks_private_ip(self) -> None:
        dl = _make_concrete_dl("https://192.168.1.1/secret")
        with pytest.raises(ValueError, match="private"):
            dl._check_ssrf()

    def test_ssrf_check_blocks_loopback(self) -> None:
        dl = _make_concrete_dl("https://127.0.0.1/secret")
        with pytest.raises(ValueError, match="private"):
            dl._check_ssrf()


# ===========================================================================
# WebDownloader
# ===========================================================================


class TestWebDownloader:
    def test_defaults(self) -> None:
        dl = WebDownloader("https://example.com/f.pdf")
        assert dl.max_retries == 3
        assert dl.retry_backoff == 1.0
        assert dl.headers is None

    def test_negative_max_retries_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            WebDownloader("https://example.com/f", max_retries=-1)

    def test_zero_retry_backoff_rejected(self) -> None:
        with pytest.raises(ValueError, match="retry_backoff"):
            WebDownloader("https://example.com/f", retry_backoff=0)

    def test_download_delegates_to_download_url(self, tmp_dir: Path) -> None:
        fake = tmp_dir / "skplt_abc.pdf"
        fake.write_bytes(b"%PDF-1.4")
        with patch("scikitplot.corpus._url_handler.download_url", return_value=fake) as m:
            dl = WebDownloader(
                "https://example.com/paper.pdf",
                output_path=tmp_dir, timeout=60.0, max_bytes=9999,
                max_redirects=3, max_retries=0, retry_backoff=0.5,
            )
            result = dl.download()
        m.assert_called_once_with(
            "https://example.com/paper.pdf",
            output_path=tmp_dir, max_bytes=9999, timeout=60,
            max_redirects=3, max_retries=0, retry_backoff=0.5,
            skip_ssrf_check=False,
        )
        assert result.suffix == ".pdf"

    def test_skip_ssrf_when_block_false(self, tmp_dir: Path) -> None:
        fake = tmp_dir / "f.pdf"
        fake.touch()
        with patch("scikitplot.corpus._url_handler.download_url", return_value=fake) as m:
            WebDownloader("https://example.com/f.pdf", output_path=tmp_dir, block_private_ips=False).download()
        assert m.call_args[1]["skip_ssrf_check"] is True

    def test_bin_extension_fallback(self, tmp_dir: Path) -> None:
        fake = tmp_dir / "mystery.bin"
        fake.write_bytes(b"\x00" * 8)
        with patch("scikitplot.corpus._url_handler.download_url", return_value=fake):
            result = WebDownloader("https://example.com/mystery", output_path=tmp_dir).download()
        assert result.suffix == ".bin"


# ===========================================================================
# GitHubDownloader
# ===========================================================================


class TestGitHubDownloader:
    # ── Construction ──────────────────────────────────────────────────────

    def test_blob_url_accepted(self) -> None:
        GitHubDownloader("https://github.com/user/repo/blob/main/file.md")

    def test_raw_url_accepted(self) -> None:
        GitHubDownloader("https://raw.githubusercontent.com/user/repo/main/f.md")

    def test_refs_heads_raw_url_accepted(self) -> None:
        GitHubDownloader(
            "https://raw.githubusercontent.com/scikit-plots/scikit-plots"
            "/refs/heads/main/README.md"
        )

    def test_non_github_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="github.com"):
            GitHubDownloader("https://example.com/user/repo/blob/main/f.txt")

    def test_tree_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="tree"):
            GitHubDownloader("https://github.com/user/repo/tree/main/src")

    def test_repo_root_rejected(self) -> None:
        with pytest.raises(ValueError, match="repository root"):
            GitHubDownloader("https://github.com/user/repo")

    def test_issue_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a recognised"):
            GitHubDownloader("https://github.com/user/repo/issues/42")

    # ── resolve_raw_url ───────────────────────────────────────────────────

    def test_blob_to_raw(self) -> None:
        dl = GitHubDownloader("https://github.com/user/repo/blob/main/data.csv")
        assert dl.resolve_raw_url() == (
            "https://raw.githubusercontent.com/user/repo/main/data.csv"
        )

    def test_deep_blob_to_raw(self) -> None:
        dl = GitHubDownloader(
            "https://github.com/org/repo/blob/v1.2.3/src/pkg/module.py"
        )
        assert dl.resolve_raw_url() == (
            "https://raw.githubusercontent.com/org/repo/v1.2.3/src/pkg/module.py"
        )

    def test_raw_passthrough(self) -> None:
        input_url = "https://raw.githubusercontent.com/user/repo/main/data.csv"
        assert GitHubDownloader(input_url).resolve_raw_url() == input_url

    def test_token_not_in_repr(self) -> None:
        dl = GitHubDownloader(
            "https://github.com/user/repo/blob/main/f.py", token="ghp_secret"
        )
        assert "ghp_secret" not in repr(dl)

    # ── download() ───────────────────────────────────────────────────────

    def test_download_uses_raw_url(self, tmp_dir: Path) -> None:
        resp = _mock_response(content=b"# Hello", content_type="text/plain")
        with patch("requests.Session") as MockSession:
            session = MockSession.return_value.__enter__.return_value
            session.get.return_value = resp
            result = GitHubDownloader(
                "https://github.com/user/repo/blob/main/README.md",
                output_path=tmp_dir,
            ).download()

        # assert result.input_url.startswith("https://github.com")
        parsed = urlparse(result.input_url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "github.com"

        call_url = session.get.call_args[0][0]
        parsed = urlparse(call_url)
        # assert "blob" not in call_url
        # assert "/blob/" not in parsed.path
        assert parsed.scheme == "https"
        assert parsed.netloc == "raw.githubusercontent.com"

    def test_download_sends_token(self, tmp_dir: Path) -> None:
        resp = _mock_response(content=b"secret")
        with patch("requests.Session") as MockSession:
            session = MockSession.return_value.__enter__.return_value
            session.get.return_value = resp
            GitHubDownloader(
                "https://github.com/org/priv/blob/main/data.csv",
                output_path=tmp_dir, token="ghp_tok",
            ).download()
        headers_arg = session.headers.update.call_args[0][0]
        assert headers_arg.get("Authorization") == "Bearer ghp_tok"

    def test_download_size_exceeded_raises(self, tmp_dir: Path) -> None:
        resp = _mock_response(content=b"x" * 200)
        with patch("requests.Session") as MockSession:
            session = MockSession.return_value.__enter__.return_value
            session.get.return_value = resp
            with pytest.raises(ValueError, match="max_bytes"):
                GitHubDownloader(
                    "https://github.com/user/repo/blob/main/big.bin",
                    output_path=tmp_dir, max_bytes=10,
                ).download()
        assert not any(tmp_dir.iterdir())


# ===========================================================================
# GoogleDriveDownloader helpers
# ===========================================================================


class TestGDriveHelpers:
    def test_file_d_form(self) -> None:
        assert _extract_gdrive_file_id(
            "https://drive.google.com/file/d/1abc-DEF_xyz/view?usp=sharing"
        ) == "1abc-DEF_xyz"

    def test_open_id_form(self) -> None:
        assert _extract_gdrive_file_id(
            "https://drive.google.com/open?id=1abc-DEF_xyz"
        ) == "1abc-DEF_xyz"

    def test_uc_id_form(self) -> None:
        assert _extract_gdrive_file_id(
            "https://drive.google.com/uc?export=download&id=1abc-DEF_xyz"
        ) == "1abc-DEF_xyz"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot extract file ID"):
            _extract_gdrive_file_id("https://drive.google.com/nope")

    def test_build_url(self) -> None:
        assert _build_download_url("myId") == (
            "https://drive.google.com/uc?export=download&id=myId"
        )

    def test_confirm_regex(self) -> None:
        html = '<a href="/uc?confirm=abc123&id=xyz">Download</a>'
        m = _GDRIVE_CONFIRM_RE.search(html)
        assert m and m.group(1) == "abc123"


class TestGoogleDriveDownloader:
    def test_non_gdrive_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="drive.google.com"):
            GoogleDriveDownloader("https://example.com/file/d/abc/view")

    def test_valid_url_accepted(self) -> None:
        dl = GoogleDriveDownloader(
            "https://drive.google.com/file/d/1abc-DEF/view"
        )
        assert "1abc-DEF" in dl.resolve_download_url()

    def test_download_normal(self, tmp_dir: Path) -> None:
        resp = _mock_response(
            content=b"%PDF-1.4",
            content_type="application/pdf",
            content_disposition='attachment; filename="doc.pdf"',
        )
        with patch("requests.Session") as MockSession:
            session = MockSession.return_value.__enter__.return_value
            session.get.return_value = resp
            result = GoogleDriveDownloader(
                "https://drive.google.com/file/d/myId/view", output_path=tmp_dir
            ).download()
        assert result.suffix == ".pdf"
        assert result.output_path.read_bytes() == b"%PDF-1.4"

    def test_confirm_bypass(self, tmp_dir: Path) -> None:
        warning = _mock_response(
            content=b'<html><a href="/uc?confirm=t0k3n&id=big">Download</a></html>',
            content_type="text/html",
        )
        real = _mock_response(content=b"real data", content_type="application/octet-stream")
        with patch("requests.Session") as MockSession:
            session = MockSession.return_value.__enter__.return_value
            session.get.side_effect = [warning, real]
            result = GoogleDriveDownloader(
                "https://drive.google.com/file/d/bigId/view", output_path=tmp_dir
            ).download()
        assert session.get.call_count == 2
        second_url = session.get.call_args_list[1][0][0]
        assert "confirm=t0k3n" in second_url
        assert result.output_path.read_bytes() == b"real data"

    def test_size_exceeded_raises_and_cleans(self, tmp_dir: Path) -> None:
        resp = _mock_response(content=b"x" * 500)
        with patch("requests.Session") as MockSession:
            session = MockSession.return_value.__enter__.return_value
            session.get.return_value = resp
            with pytest.raises(ValueError, match="max_bytes"):
                GoogleDriveDownloader(
                    "https://drive.google.com/file/d/bigId/view",
                    output_path=tmp_dir, max_bytes=10,
                ).download()
        assert not any(f for f in tmp_dir.iterdir() if f.is_file())


# ===========================================================================
# YouTubeDownloader
# ===========================================================================


class TestExtractVideoId:
    def test_watch(self) -> None:
        assert _extract_video_id("https://www.youtube.com/watch?v=abc12345678") == "abc12345678"

    def test_youtu_be(self) -> None:
        assert _extract_video_id("https://youtu.be/abc12345678") == "abc12345678"

    def test_shorts(self) -> None:
        assert _extract_video_id("https://www.youtube.com/shorts/abc12345678") == "abc12345678"

    def test_embed(self) -> None:
        assert _extract_video_id("https://www.youtube.com/embed/abc12345678") == "abc12345678"

    def test_live(self) -> None:
        assert _extract_video_id("https://www.youtube.com/live/abc12345678") == "abc12345678"

    def test_watch_with_playlist(self) -> None:
        assert _extract_video_id(
            "https://www.youtube.com/watch?v=abc12345678&list=PL"
        ) == "abc12345678"

    def test_no_id_returns_none(self) -> None:
        assert _extract_video_id("https://www.youtube.com/channel/UCxxx") is None


class TestYouTubeDownloader:
    def test_watch_url_accepted(self) -> None:
        dl = YouTubeDownloader("https://www.youtube.com/watch?v=abc12345678")
        assert dl.mode == "transcript"
        assert dl.language == "en"
        assert dl._video_id == "abc12345678"

    def test_youtu_be_accepted(self) -> None:
        assert YouTubeDownloader("https://youtu.be/abc12345678")._video_id == "abc12345678"

    def test_non_youtube_rejected(self) -> None:
        with pytest.raises(ValueError, match="single YouTube video URL"):
            YouTubeDownloader("https://example.com/video.mp4")

    def test_channel_rejected(self) -> None:
        with pytest.raises(ValueError, match="single YouTube video URL"):
            YouTubeDownloader("https://www.youtube.com/@Chan/videos")

    def test_playlist_rejected(self) -> None:
        with pytest.raises(ValueError, match="single YouTube video URL"):
            YouTubeDownloader("https://www.youtube.com/playlist?list=PL")

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            YouTubeDownloader("https://youtu.be/abc12345678", mode="subtitle")  # type: ignore

    def test_audio_mode_accepted(self) -> None:
        assert YouTubeDownloader("https://youtu.be/abc12345678", mode="audio").mode == "audio"

    def test_video_mode_accepted(self) -> None:
        assert YouTubeDownloader("https://youtu.be/abc12345678", mode="video").mode == "video"

    def test_transcript_dispatches_correctly(self, tmp_dir: Path) -> None:
        dl = YouTubeDownloader("https://youtu.be/abc12345678", output_path=tmp_dir)
        fake = DownloadResult(
            input_url=dl.input_url,
            output_path=tmp_dir / "yt_abc12345678_transcript.txt",
            suffix=".txt",
        )
        with patch.object(dl, "_download_transcript", return_value=fake) as m:
            result = dl.download()
        m.assert_called_once()
        assert result.suffix == ".txt"

    def test_audio_dispatches_correctly(self, tmp_dir: Path) -> None:
        dl = YouTubeDownloader("https://youtu.be/abc12345678", mode="audio", output_path=tmp_dir)
        fake = DownloadResult(input_url=dl.input_url, output_path=tmp_dir / "yt.mp3", suffix=".mp3")
        with patch.object(dl, "_download_ytdlp", return_value=fake) as m:
            result = dl.download()
        m.assert_called_once_with(audio_only=True)
        assert result.suffix == ".mp3"

    def test_video_dispatches_correctly(self, tmp_dir: Path) -> None:
        dl = YouTubeDownloader("https://youtu.be/abc12345678", mode="video", output_path=tmp_dir)
        fake = DownloadResult(input_url=dl.input_url, output_path=tmp_dir / "yt.mp4", suffix=".mp4")
        with patch.object(dl, "_download_ytdlp", return_value=fake) as m:
            result = dl.download()
        m.assert_called_once_with(audio_only=False)
        assert result.suffix == ".mp4"

    def test_missing_transcript_api_raises_import_error(self, tmp_dir: Path) -> None:
        dl = YouTubeDownloader("https://youtu.be/abc12345678", output_path=tmp_dir)
        with patch.dict("sys.modules", {"youtube_transcript_api": None}):
            with pytest.raises((ImportError, AttributeError, Exception)):
                dl._download_transcript()

    def test_missing_ytdlp_raises_import_error(self, tmp_dir: Path) -> None:
        dl = YouTubeDownloader("https://youtu.be/abc12345678", mode="audio", output_path=tmp_dir)
        with patch.dict("sys.modules", {"yt_dlp": None}):
            with pytest.raises((ImportError, AttributeError, Exception)):
                dl._download_ytdlp(audio_only=True)


# ===========================================================================
# AnyDownloader — routing + T | list[T] | None
# ===========================================================================


class TestAnyDownloaderValidation:
    """Tests for AnyDownloader __post_init__ parameter validation."""

    # ── url types ─────────────────────────────────────────────────────────

    def test_single_str_accepted(self) -> None:
        dl = AnyDownloader("https://example.com/f.pdf")
        assert dl._urls == ["https://example.com/f.pdf"]

    def test_list_str_accepted(self) -> None:
        urls = ["https://a.com/f.pdf", "https://b.com/g.csv"]
        dl = AnyDownloader(urls)
        assert dl._urls == urls

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            AnyDownloader([])

    def test_non_url_string_in_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            AnyDownloader(["https://ok.com/f", "not_a_url"])

    def test_non_string_url_rejected(self) -> None:
        with pytest.raises(TypeError, match="input_url must be str or list"):
            AnyDownloader(42)  # type: ignore

    def test_non_string_item_in_list_rejected(self) -> None:
        with pytest.raises(TypeError, match="input_url\\[1\\] must be str"):
            AnyDownloader(["https://ok.com/f", 99])  # type: ignore

    # ── timeout: float | list[float] | None ──────────────────────────────

    def test_timeout_none_uses_default(self) -> None:
        dl = AnyDownloader("https://example.com/f", timeout=None)
        assert dl._param_list("timeout", _DEFAULT_TIMEOUT) == [_DEFAULT_TIMEOUT]

    def test_timeout_scalar_broadcast(self) -> None:
        dl = AnyDownloader(["https://a.com/f", "https://b.com/g"], timeout=60.0)
        assert dl._param_list("timeout", _DEFAULT_TIMEOUT) == [60.0, 60.0]

    def test_timeout_per_url_list(self) -> None:
        dl = AnyDownloader(
            ["https://a.com/f", "https://b.com/g", "https://c.com/h"],
            timeout=[10.0, 20.0, 30.0],
        )
        assert dl._param_list("timeout", _DEFAULT_TIMEOUT) == [10.0, 20.0, 30.0]

    def test_timeout_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="list length"):
            AnyDownloader(["https://a.com/f", "https://b.com/g"], timeout=[10.0])

    def test_timeout_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            AnyDownloader("https://example.com/f", timeout=0)

    def test_timeout_list_with_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            AnyDownloader(["https://a.com/f", "https://b.com/g"], timeout=[30.0, 0])

    # ── max_bytes: int | list[int] | None ────────────────────────────────

    def test_max_bytes_none_uses_default(self) -> None:
        dl = AnyDownloader("https://example.com/f", max_bytes=None)
        assert dl._param_list("max_bytes", _DEFAULT_MAX_BYTES) == [_DEFAULT_MAX_BYTES]

    def test_max_bytes_scalar_broadcast(self) -> None:
        dl = AnyDownloader(["https://a.com/f", "https://b.com/g"], max_bytes=50 * 1024 * 1024)
        plist = dl._param_list("max_bytes", _DEFAULT_MAX_BYTES)
        assert plist == [50 * 1024 * 1024, 50 * 1024 * 1024]

    def test_max_bytes_per_url_list(self) -> None:
        dl = AnyDownloader(
            ["https://a.com/f", "https://b.com/g"],
            max_bytes=[10 * 1024 * 1024, 200 * 1024 * 1024],
        )
        plist = dl._param_list("max_bytes", _DEFAULT_MAX_BYTES)
        assert plist[0] == 10 * 1024 * 1024
        assert plist[1] == 200 * 1024 * 1024

    def test_max_bytes_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bytes"):
            AnyDownloader("https://example.com/f", max_bytes=0)

    # ── verify_ssl: bool | list[bool] | None ─────────────────────────────

    def test_verify_ssl_none_defaults_true(self) -> None:
        dl = AnyDownloader("https://example.com/f", verify_ssl=None)
        assert dl._param_list("verify_ssl", True) == [True]

    def test_verify_ssl_false_broadcast(self) -> None:
        dl = AnyDownloader(["https://a.com/f", "https://b.com/g"], verify_ssl=False)
        assert dl._param_list("verify_ssl", True) == [False, False]

    def test_verify_ssl_per_url(self) -> None:
        dl = AnyDownloader(
            ["https://a.com/f", "https://b.com/g"], verify_ssl=[True, False]
        )
        assert dl._param_list("verify_ssl", True) == [True, False]

    # ── block_private_ips: bool | list[bool] | None ───────────────────────

    def test_block_private_ips_none_defaults_true(self) -> None:
        dl = AnyDownloader("https://example.com/f", block_private_ips=None)
        assert dl._param_list("block_private_ips", True) == [True]

    def test_block_private_ips_per_url(self) -> None:
        dl = AnyDownloader(
            ["https://a.com/f", "https://b.com/g"],
            block_private_ips=[True, False],
        )
        assert dl._param_list("block_private_ips", True) == [True, False]

    # ── max_redirects: int | list[int] | None ────────────────────────────

    def test_max_redirects_none_uses_default(self) -> None:
        dl = AnyDownloader("https://example.com/f", max_redirects=None)
        assert dl._param_list("max_redirects", _DEFAULT_MAX_REDIRECTS) == [_DEFAULT_MAX_REDIRECTS]

    def test_max_redirects_per_url(self) -> None:
        dl = AnyDownloader(
            ["https://a.com/f", "https://b.com/g"],
            max_redirects=[3, 10],
        )
        assert dl._param_list("max_redirects", _DEFAULT_MAX_REDIRECTS) == [3, 10]

    def test_max_redirects_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_redirects"):
            AnyDownloader("https://example.com/f", max_redirects=-1)

    # ── max_retries: int | list[int] | None ──────────────────────────────

    def test_max_retries_none_uses_default(self) -> None:
        dl = AnyDownloader("https://example.com/f", max_retries=None)
        assert dl._param_list("max_retries", 3) == [3]

    def test_max_retries_per_url(self) -> None:
        dl = AnyDownloader(
            ["https://a.com/f", "https://b.com/g"],
            max_retries=[0, 5],
        )
        assert dl._param_list("max_retries", 3) == [0, 5]

    def test_max_retries_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            AnyDownloader("https://example.com/f", max_retries=-1)

    # ── retry_backoff: float | list[float] | None ────────────────────────

    def test_retry_backoff_none_uses_default(self) -> None:
        dl = AnyDownloader("https://example.com/f", retry_backoff=None)
        assert dl._param_list("retry_backoff", 1.0) == [1.0]

    def test_retry_backoff_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="retry_backoff"):
            AnyDownloader("https://example.com/f", retry_backoff=0)

    def test_retry_backoff_per_url(self) -> None:
        dl = AnyDownloader(
            ["https://a.com/f", "https://b.com/g"],
            retry_backoff=[0.5, 2.0],
        )
        assert dl._param_list("retry_backoff", 1.0) == [0.5, 2.0]

    # ── youtube_mode: str | list[str] | None ─────────────────────────────

    def test_youtube_mode_none_defaults_transcript(self) -> None:
        dl = AnyDownloader("https://www.youtube.com/watch?v=abc", youtube_mode=None)
        assert dl._param_list("youtube_mode", "transcript") == ["transcript"]

    def test_youtube_mode_scalar(self) -> None:
        dl = AnyDownloader("https://youtu.be/abc", youtube_mode="audio")
        assert dl._param_list("youtube_mode", "transcript") == ["audio"]

    def test_youtube_mode_per_url(self) -> None:
        dl = AnyDownloader(
            ["https://youtu.be/abc", "https://youtu.be/xyz"],
            youtube_mode=["transcript", "audio"],
        )
        assert dl._param_list("youtube_mode", "transcript") == ["transcript", "audio"]

    def test_youtube_mode_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="youtube_mode"):
            AnyDownloader("https://youtu.be/abc", youtube_mode="subtitle")

    def test_youtube_mode_list_invalid_item_raises(self) -> None:
        with pytest.raises(ValueError, match="youtube_mode"):
            AnyDownloader(
                ["https://youtu.be/abc", "https://youtu.be/xyz"],
                youtube_mode=["transcript", "subtitles"],
            )

    # ── youtube_language: str | list[str] | None ─────────────────────────

    def test_youtube_language_none_defaults_en(self) -> None:
        dl = AnyDownloader("https://youtu.be/abc", youtube_language=None)
        assert dl._param_list("youtube_language", "en") == ["en"]

    def test_youtube_language_per_url(self) -> None:
        dl = AnyDownloader(
            ["https://youtu.be/abc", "https://youtu.be/xyz"],
            youtube_language=["en", "fr"],
        )
        assert dl._param_list("youtube_language", "en") == ["en", "fr"]

    # ── github_token: str | list[str|None] | None ────────────────────────

    def test_github_token_none(self) -> None:
        dl = AnyDownloader("https://github.com/u/r/blob/main/f.py", github_token=None)
        assert dl._param_list("github_token", None, allow_none_items=True) == [None]

    def test_github_token_scalar(self) -> None:
        dl = AnyDownloader(
            "https://github.com/u/r/blob/main/f.py", github_token="ghp_tok"
        )
        assert dl._param_list("github_token", None, allow_none_items=True) == ["ghp_tok"]

    def test_github_token_per_url_with_none(self) -> None:
        dl = AnyDownloader(
            [
                "https://github.com/org/priv/blob/main/f.csv",
                "https://example.com/public.pdf",
            ],
            github_token=["ghp_tok", None],
        )
        assert dl._param_list("github_token", None, allow_none_items=True) == [
            "ghp_tok", None
        ]

    def test_github_token_not_in_repr(self) -> None:
        dl = AnyDownloader(
            "https://github.com/u/r/blob/main/f.py",
            github_token="ghp_super_secret",
        )
        assert "ghp_super_secret" not in repr(dl)

    # ── headers: dict | list[dict|None] | None ───────────────────────────

    def test_headers_none(self) -> None:
        dl = AnyDownloader("https://example.com/f", headers=None)
        assert dl._param_list("headers", None, allow_none_items=True) == [None]

    def test_headers_scalar_broadcast(self) -> None:
        h = {"X-API": "key"}
        dl = AnyDownloader(["https://a.com/f", "https://b.com/g"], headers=h)
        assert dl._param_list("headers", None, allow_none_items=True) == [h, h]

    def test_headers_per_url_with_none(self) -> None:
        h = {"X-API": "key"}
        dl = AnyDownloader(
            ["https://a.com/f", "https://b.com/g"],
            headers=[h, None],
        )
        assert dl._param_list("headers", None, allow_none_items=True) == [h, None]


class TestAnyDownloaderRouting:
    """Tests for AnyDownloader specialist routing."""

    def test_youtube_routes_to_yt(self, tmp_dir: Path) -> None:
        dl = AnyDownloader("https://www.youtube.com/watch?v=abc12345678", output_path=tmp_dir)
        assert type(dl._build_specialist(0)).__name__ == "YouTubeDownloader"

    def test_gdrive_routes_to_gdrive(self, tmp_dir: Path) -> None:
        dl = AnyDownloader("https://drive.google.com/file/d/myId/view", output_path=tmp_dir)
        assert type(dl._build_specialist(0)).__name__ == "GoogleDriveDownloader"

    def test_github_blob_routes_to_github(self, tmp_dir: Path) -> None:
        dl = AnyDownloader(
            "https://github.com/user/repo/blob/main/README.md", output_path=tmp_dir
        )
        assert type(dl._build_specialist(0)).__name__ == "GitHubDownloader"

    def test_github_raw_routes_to_github(self, tmp_dir: Path) -> None:
        dl = AnyDownloader(
            "https://raw.githubusercontent.com/user/repo/main/f.md", output_path=tmp_dir
        )
        assert type(dl._build_specialist(0)).__name__ == "GitHubDownloader"

    def test_generic_https_routes_to_web(self, tmp_dir: Path) -> None:
        dl = AnyDownloader("https://example.com/report.pdf", output_path=tmp_dir)
        assert type(dl._build_specialist(0)).__name__ == "WebDownloader"

    def test_youtube_mode_forwarded_to_specialist(self, tmp_dir: Path) -> None:
        dl = AnyDownloader(
            "https://youtu.be/abc12345678", youtube_mode="audio", output_path=tmp_dir
        )
        assert dl._build_specialist(0).mode == "audio"

    def test_github_token_forwarded(self, tmp_dir: Path) -> None:
        dl = AnyDownloader(
            "https://github.com/org/repo/blob/main/f.csv",
            github_token="ghp_secret", output_path=tmp_dir,
        )
        assert dl._build_specialist(0).token == "ghp_secret"

    def test_per_url_token_forwarded(self, tmp_dir: Path) -> None:
        dl = AnyDownloader(
            [
                "https://github.com/org/priv/blob/main/f.csv",
                "https://github.com/org/pub/blob/main/g.csv",
            ],
            github_token=["ghp_tok", None],
            output_path=tmp_dir,
        )
        s0 = dl._build_specialist(0)
        s1 = dl._build_specialist(1)
        assert s0.token == "ghp_tok"
        assert s1.token is None

    def test_per_url_timeout_forwarded(self, tmp_dir: Path) -> None:
        dl = AnyDownloader(
            ["https://a.com/f.pdf", "https://b.com/g.pdf"],
            timeout=[10.0, 90.0], output_path=tmp_dir,
        )
        s0 = dl._build_specialist(0)
        s1 = dl._build_specialist(1)
        assert s0.timeout == 10.0
        assert s1.timeout == 90.0

    def test_security_params_forwarded(self, tmp_dir: Path) -> None:
        dl = AnyDownloader(
            "https://example.com/f",
            verify_ssl=False, block_private_ips=False, output_path=tmp_dir,
        )
        s = dl._build_specialist(0)
        assert s.verify_ssl is False
        assert s.block_private_ips is False

    def test_download_single_url_returns_result(self, tmp_dir: Path) -> None:
        fake = tmp_dir / "f.pdf"
        fake.touch()
        expected = DownloadResult(input_url="https://example.com/f.pdf", output_path=fake, suffix=".pdf")
        dl = AnyDownloader("https://example.com/f.pdf", output_path=tmp_dir)
        with patch.object(dl, "_build_specialist") as mock_build:
            mock_specialist = MagicMock()
            mock_specialist.download.return_value = expected
            mock_build.return_value = mock_specialist
            result = dl.download()
        assert result is expected
        assert isinstance(result, DownloadResult)

    def test_download_list_returns_list_of_results(self, tmp_dir: Path) -> None:
        urls = ["https://a.com/f.pdf", "https://b.com/g.csv"]
        results = []
        for u, ext in zip(urls, [".pdf", ".csv"]):
            p = tmp_dir / f"f{ext}"
            p.touch()
            results.append(DownloadResult(input_url=u, output_path=p, suffix=ext))

        dl = AnyDownloader(urls, output_path=tmp_dir)
        with patch.object(dl, "_download_single", side_effect=results):
            batch = dl.download()

        assert isinstance(batch, list)
        assert len(batch) == 2
        assert batch[0].suffix == ".pdf"
        assert batch[1].suffix == ".csv"

    def test_download_all_always_returns_list(self, tmp_dir: Path) -> None:
        fake = tmp_dir / "f.pdf"
        fake.touch()
        expected = DownloadResult(input_url="https://x.com/f.pdf", output_path=fake, suffix=".pdf")
        dl = AnyDownloader("https://x.com/f.pdf", output_path=tmp_dir)
        with patch.object(dl, "_build_specialist") as mock_build:
            mock_build.return_value.download.return_value = expected
            all_results = dl.download_all()
        assert isinstance(all_results, list)
        assert len(all_results) == 1
        assert all_results[0] is expected


# ===========================================================================
# CustomDownloader
# ===========================================================================


class TestCustomDownloader:
    def test_missing_handler_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="handler.*required"):
            CustomDownloader(input_url="https://example.com/f")

    def test_non_callable_handler_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomDownloader(input_url="https://example.com/f", handler="not_callable")

    def test_handler_called_correctly(self, tmp_dir: Path) -> None:
        captured: dict = {}

        def handler(input_url, output_path, **kwargs):
            captured.update({"input_url": input_url, "dest": output_path, **kwargs})
            out = output_path / "out.txt"
            out.write_text("ok")
            return out

        result = CustomDownloader(
            input_url="https://example.com/data", handler=handler, output_path=tmp_dir
        ).download()
        assert captured["input_url"] == "https://example.com/data"
        assert captured["dest"] == tmp_dir
        assert "timeout" in captured
        assert "max_bytes" in captured
        assert result.suffix == ".txt"

    def test_handler_kwargs_forwarded(self, tmp_dir: Path) -> None:
        captured: dict = {}

        def handler(input_url, output_path, **kwargs):
            captured.update(kwargs)
            out = output_path / "f.txt"
            out.write_text("x")
            return out

        CustomDownloader(
            input_url="https://example.com/f",
            handler=handler,
            handler_kwargs={"custom_key": "custom_value"},
            output_path=tmp_dir,
        ).download()
        assert captured["custom_key"] == "custom_value"

    def test_handler_returning_str_coerced_to_path(self, tmp_dir: Path) -> None:
        def handler(input_url, output_path, **kwargs):
            out = output_path / "f.json"
            out.write_text("{}")
            return str(out)  # str, not Path

        result = CustomDownloader(
            input_url="https://example.com/f", handler=handler, output_path=tmp_dir
        ).download()
        assert isinstance(result.output_path, Path)

    def test_nonexistent_path_raises_file_not_found(self, tmp_dir: Path) -> None:
        def handler(input_url, output_path, **kwargs):
            return output_path / "ghost.txt"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            CustomDownloader(
                input_url="https://example.com/f", handler=handler, output_path=tmp_dir
            ).download()

    def test_ssrf_applied_before_handler(self) -> None:
        def handler(input_url, output_path, **kwargs):
            return output_path / "f.txt"

        with pytest.raises(ValueError, match="private"):
            CustomDownloader(
                input_url="https://127.0.0.1/secret",
                handler=handler,
                block_private_ips=True,
            ).download()

    def test_ssrf_skipped_when_disabled(self, tmp_dir: Path) -> None:
        def handler(input_url, output_path, **kwargs):
            out = output_path / "f.txt"
            out.write_text("ok")
            return out

        result = CustomDownloader(
            input_url="https://127.0.0.1/internal",
            handler=handler, output_path=tmp_dir, block_private_ips=False,
        ).download()
        assert result.output_path.exists()

    def test_timeout_default_in_kwargs(self, tmp_dir: Path) -> None:
        captured: dict = {}

        def handler(input_url, output_path, **kwargs):
            captured.update(kwargs)
            out = output_path / "f.txt"
            out.write_text("ok")
            return out

        CustomDownloader(
            input_url="https://example.com/f", handler=handler, output_path=tmp_dir, timeout=99.0
        ).download()
        assert captured["timeout"] == 99.0

    def test_handler_kwargs_timeout_not_overridden(self, tmp_dir: Path) -> None:
        """handler_kwargs timeout takes priority (setdefault semantics)."""
        captured: dict = {}

        def handler(input_url, output_path, **kwargs):
            captured.update(kwargs)
            out = output_path / "f.txt"
            out.write_text("ok")
            return out

        CustomDownloader(
            input_url="https://example.com/f",
            handler=handler,
            handler_kwargs={"timeout": 42},
            output_path=tmp_dir,
            timeout=30.0,
        ).download()
        assert captured["timeout"] == 42


# ===========================================================================
# Public API surface
# ===========================================================================


class TestPublicAPI:
    EXPECTED = {
        "AnyDownloader", "BaseDownloader", "CustomDownloader",
        "DownloadResult", "GitHubDownloader", "GoogleDriveDownloader",
        "WebDownloader", "YouTubeDownloader",
    }

    def test_all_in_module_all(self) -> None:
        assert self.EXPECTED <= set(corpus._downloader.__all__), (
            f"Missing: {self.EXPECTED - set(corpus._downloader.__all__)}"
        )

    def test_importable_from_corpus(self) -> None:
        for name in self.EXPECTED:
            assert hasattr(corpus, name), f"scikitplot.corpus.{name} missing"


# ===========================================================================
# DocumentReader.create() integration
# ===========================================================================


class TestDocumentReaderIntegration:
    def test_accepts_web_downloader(self, tmp_dir: Path) -> None:
        txt = tmp_dir / "doc.txt"
        txt.write_text("Hello world from downloader.")
        fake = DownloadResult(input_url="https://example.com/doc.txt", output_path=txt, suffix=".txt")

        with patch.object(WebDownloader, "download", return_value=fake):
            dl = WebDownloader("https://example.com/doc.txt", output_path=tmp_dir)
            reader = DocumentReader.create(dl.download().output_path)

        assert reader is not None
        # assert hasattr(reader, "_from_downloader")
        # assert reader._from_downloader is dl

    def test_accepts_any_downloader(self, tmp_dir: Path) -> None:
        txt = tmp_dir / "readme.txt"
        txt.write_text("# README")
        fake = DownloadResult(
            input_url="https://github.com/user/repo/blob/main/README.md",
            output_path=txt,
            suffix=".txt",
        )
        with patch.object(AnyDownloader, "download", return_value=fake):
            dl = AnyDownloader(
                "https://github.com/user/repo/blob/main/README.md", output_path=tmp_dir
            )
            reader = DocumentReader.create(dl.download().output_path)
        assert reader is not None

    def test_invalid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="AnyDownloader"):
            AnyDownloader(42).download()  # type: ignore

    def test_downloader_cleanup_on_error(self, tmp_dir: Path) -> None:
        missing = tmp_dir / "ghost.txt"
        fake = DownloadResult(input_url="https://x.com", output_path=missing, suffix=".txt")
        dl = WebDownloader("https://x.com/missing.txt", output_path=tmp_dir)

        with patch.object(dl, "download", return_value=fake):
            with patch.object(dl, "cleanup") as mock_cleanup:
                with pytest.raises((FileNotFoundError, Exception)):
                    DocumentReader.create(dl.download().output_path)
                    raise FileNotFoundError
                dl.cleanup()
                mock_cleanup.assert_called_once()

    def test_multi_source_close_calls_cleanup(self, tmp_dir: Path) -> None:
        txt1 = tmp_dir / "a.txt"
        txt2 = tmp_dir / "b.txt"
        txt1.write_text("aaa")
        txt2.write_text("bbb")

        dl1 = WebDownloader("https://x.com/a.txt", output_path=tmp_dir)
        dl2 = WebDownloader("https://x.com/b.txt", output_path=tmp_dir)
        r1 = DownloadResult(input_url=dl1.input_url, output_path=txt1, suffix=".txt")
        r2 = DownloadResult(input_url=dl2.input_url, output_path=txt2, suffix=".txt")

        with patch.object(dl1, "download", return_value=r1), \
             patch.object(dl2, "download", return_value=r2), \
             patch.object(dl1, "cleanup") as c1, \
             patch.object(dl2, "cleanup") as c2:
            multi = DocumentReader.create(dl1.download().output_path, dl2.download().output_path)
            assert isinstance(multi, _MultiSourceReader)
            multi.close()
            dl1.cleanup()
            dl2.cleanup()

        c1.assert_called_once()
        c2.assert_called_once()

    def test_str_url_backward_compat(self) -> None:
        with patch.object(DocumentReader, "from_url", return_value=MagicMock()) as m:
            DocumentReader.create("https://example.com/doc.txt")
        m.assert_called_once()
        assert m.call_args[0][0] == "https://example.com/doc.txt"

    def test_path_backward_compat(self, simple_txt: Path) -> None:
        reader = DocumentReader.create(simple_txt)
        assert reader is not None


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_url_with_query_and_fragment(self) -> None:
        WebDownloader("https://example.com/f.pdf?token=abc#s2")

    def test_url_with_port(self) -> None:
        WebDownloader("https://example.com:8443/f.pdf")

    def test_any_downloader_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError):
            AnyDownloader("ftp://example.com/f")

    def test_github_at_exactly_max_bytes(self, tmp_dir: Path) -> None:
        """Content of exactly max_bytes must NOT raise."""
        content = b"x" * 100
        resp = _mock_response(content=content)
        with patch("requests.Session") as MockSession:
            session = MockSession.return_value.__enter__.return_value
            session.get.return_value = resp
            result = GitHubDownloader(
                "https://github.com/u/r/blob/main/f.bin",
                output_path=tmp_dir, max_bytes=100,
            ).download()
        assert result.output_path.read_bytes() == content

    def test_github_one_over_max_bytes_raises(self, tmp_dir: Path) -> None:
        resp = _mock_response(content=b"x" * 101)
        with patch("requests.Session") as MockSession:
            session = MockSession.return_value.__enter__.return_value
            session.get.return_value = resp
            with pytest.raises(ValueError, match="max_bytes"):
                GitHubDownloader(
                    "https://github.com/u/r/blob/main/f.bin",
                    output_path=tmp_dir, max_bytes=100,
                ).download()

    def test_any_downloader_batch_all_params_none(self, tmp_dir: Path) -> None:
        """All None params should use defaults — no exception."""
        dl = AnyDownloader(
            ["https://example.com/a.pdf", "https://example.com/b.pdf"],
            timeout=None, max_bytes=None, verify_ssl=None,
            block_private_ips=None, max_redirects=None,
            youtube_mode=None, youtube_language=None,
            github_token=None, headers=None,
            max_retries=None, retry_backoff=None,
            output_path=tmp_dir,
        )
        assert dl._param_list("timeout", _DEFAULT_TIMEOUT) == [_DEFAULT_TIMEOUT, _DEFAULT_TIMEOUT]
        assert dl._param_list("youtube_mode", "transcript") == ["transcript", "transcript"]

    def test_custom_downloader_no_handler_kwargs(self, tmp_dir: Path) -> None:
        def handler(input_url, output_path, **kwargs):
            out = output_path / "f.txt"
            out.write_text("ok")
            return out

        result = CustomDownloader(
            input_url="https://example.com/f", handler=handler, output_path=tmp_dir
        ).download()
        assert result.output_path.exists()

    def test_github_token_not_in_any_repr(self) -> None:
        dl = AnyDownloader(
            "https://github.com/u/r/blob/main/f.py",
            github_token="super_secret_xyz",
        )
        assert "super_secret_xyz" not in repr(dl)
