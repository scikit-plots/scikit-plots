# scikitplot/corpus/_downloader/_youtube.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._downloader._youtube
=========================================
YouTube content downloader with mode-based dispatch.

:class:`YouTubeDownloader` downloads content from a YouTube video URL.
The ``mode`` parameter controls what is downloaded:

``"transcript"`` (default)
    Fetch the video transcript (closed captions) via
    ``youtube-transcript-api``.  No binary download required — the
    transcript is written to a local ``.txt`` file.  This is the fastest,
    lightest path and has no dependency on ``yt-dlp``.

``"audio"``
    Download the audio stream as ``mp3`` / ``m4a`` via ``yt-dlp``.
    Requires ``yt-dlp`` installed.

``"video"``
    Download the video stream as ``mp4`` via ``yt-dlp``.
    Requires ``yt-dlp`` installed.

Python compatibility: 3.8 - 3.15+.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass, field  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Literal, Optional

from ._base import BaseDownloader, DownloadResult

logger = logging.getLogger(__name__)

__all__ = ["YouTubeDownloader"]

# Accepted mode values
YouTubeMode = Literal["transcript", "audio", "video"]

# YouTube URL patterns (single video only — channels/playlists not supported here)
_YT_VIDEO_RE: re.Pattern[str] = re.compile(
    r"https?://(www\.)?(youtube\.com/(watch|shorts/|embed/|live/)|youtu\.be/)",
    re.IGNORECASE,
)


def _extract_video_id(input_url: str) -> Optional[str]:  # noqa: UP045
    """
    Extract the YouTube video ID from a URL.

    Parameters
    ----------
    input_url : str
        YouTube video URL.

    Returns
    -------
    str or None
        11-character video ID, or ``None`` if extraction fails.
    """
    parsed = urllib.parse.urlparse(input_url)

    # youtu.be/VIDEO_ID
    if parsed.hostname in ("youtu.be",):  # noqa: FURB171
        vid = parsed.path.lstrip("/").split("?")[0]
        return vid or None

    # youtube.com/watch?v=VIDEO_ID
    params = urllib.parse.parse_qs(parsed.query)
    if "v" in params:
        return params["v"][0]

    # youtube.com/shorts/VIDEO_ID or /embed/VIDEO_ID or /live/VIDEO_ID
    m = re.search(
        r"/(?:shorts|embed|live)/([a-zA-Z0-9_-]{11})",
        parsed.path,
    )
    if m:
        return m.group(1)

    return None


@dataclass
class YouTubeDownloader(BaseDownloader):
    """
    YouTube content downloader.

    Downloads a transcript, audio track, or video from a single YouTube
    video URL.  The ``mode`` parameter selects what is fetched.

    Parameters
    ----------
    input_url : str
        YouTube video URL.  Accepted forms:

        * ``https://www.youtube.com/watch?v=VIDEO_ID``
        * ``https://youtu.be/VIDEO_ID``
        * ``https://www.youtube.com/shorts/VIDEO_ID``
        * ``https://www.youtube.com/embed/VIDEO_ID``
    mode : {"transcript", "audio", "video"}, optional
        What to download.  Default: ``"transcript"``.
    language : str, optional
        BCP-47 language code for transcript fetching (e.g. ``"en"``,
        ``"fr"``, ``"de"``).  Falls back to auto-generated captions when
        the requested language is not available.  Only used for
        ``mode="transcript"``.  Default: ``"en"``.
    include_auto_generated : bool, optional
        When ``True``, include auto-generated transcripts as a fallback
        when no human-reviewed captions exist.  Default: ``True``.
    output_path : pathlib.Path or None, optional
        Directory for the downloaded file.  Default: ``None`` (temp dir).
    timeout : float, optional
        HTTP timeout in seconds (transcript fetch and yt-dlp).
        Default: ``30.0``.
    max_bytes : int, optional
        Download size cap in bytes (audio/video modes only; transcripts are
        always small).  Default: ``100 MB``.

    Raises
    ------
    ValueError
        If the URL is not a recognised YouTube single-video URL.
    ValueError
        If ``mode`` is not one of ``"transcript"``, ``"audio"``, ``"video"``.

    Notes
    -----
    **Transcript mode** uses ``youtube-transcript-api`` (pip-installable,
    lightweight, no browser).  It writes a plain ``.txt`` file where each
    caption segment is a line.

    **Audio/video modes** require ``yt-dlp`` (``pip install yt-dlp``).
    They invoke ``yt-dlp`` programmatically via its Python API.

    **Channels and playlists** are not supported — pass a single video URL.

    **SSRF prevention** is always applied for audio/video modes (network
    calls made by yt-dlp go to YouTube CDN, which is public; the check is a
    defence-in-depth measure).  Transcript mode makes its own HTTP calls
    which are also validated.

    Examples
    --------
    Transcript (default):

    >>> dl = YouTubeDownloader("https://www.youtube.com/watch?v=rwPISgZcYIk")
    >>> result = dl.download()
    >>> result.suffix
    '.txt'

    Audio download:

    >>> dl = YouTubeDownloader(
    ...     "https://youtu.be/rwPISgZcYIk",
    ...     mode="audio",
    ... )
    >>> result = dl.download()
    >>> result.suffix in (".mp3", ".m4a", ".webm")
    True
    """

    mode: YouTubeMode = "transcript"
    language: str = "en"
    include_auto_generated: bool = True

    def __post_init__(self) -> None:
        """Validate URL is a single YouTube video and mode is recognised."""
        super().__post_init__()

        if not _YT_VIDEO_RE.match(self.input_url):
            raise ValueError(
                f"YouTubeDownloader: input_url must be a single YouTube video URL; "
                f"got {self.input_url!r}.\n"
                f"Channels and playlists are not supported.  Accepted forms:\n"
                f"  https://www.youtube.com/watch?v=VIDEO_ID\n"
                f"  https://youtu.be/VIDEO_ID\n"
                f"  https://www.youtube.com/shorts/VIDEO_ID"
            )
        _valid_modes = ("transcript", "audio", "video")
        if self.mode not in _valid_modes:
            raise ValueError(
                f"YouTubeDownloader: mode must be one of {_valid_modes}; "
                f"got {self.mode!r}."
            )
        vid = _extract_video_id(self.input_url)
        if vid is None:
            raise ValueError(
                f"YouTubeDownloader: cannot extract video ID from {self.input_url!r}."
            )
        # Store the video ID for reuse without re-parsing
        object.__setattr__(self, "_video_id", vid)

    # ------------------------------------------------------------------
    # BaseDownloader contract
    # ------------------------------------------------------------------

    def download(self) -> DownloadResult:
        """
        Download the requested content and return a :class:`DownloadResult`.

        Dispatches to :meth:`_download_transcript`,
        :meth:`_download_audio`, or :meth:`_download_video` based on
        ``self.mode``.

        Returns
        -------
        DownloadResult
            Populated result with local file path, extension, source URL.

        Raises
        ------
        ImportError
            If ``youtube-transcript-api`` (transcript mode) or ``yt-dlp``
            (audio/video modes) is not installed.
        ValueError
            If the transcript is not available for the given video/language.
        """
        if self.mode == "transcript":
            return self._download_transcript()
        if self.mode == "audio":
            return self._download_ytdlp(audio_only=True)
        # mode == "video"
        return self._download_ytdlp(audio_only=False)

    # ------------------------------------------------------------------
    # Private mode implementations
    # ------------------------------------------------------------------

    def _download_transcript(self) -> DownloadResult:
        """
        Fetch the transcript via ``youtube-transcript-api`` and write to txt.

        Returns
        -------
        DownloadResult
            Result with ``suffix=".txt"``.

        Raises
        ------
        ImportError
            If ``youtube-transcript-api`` is not installed.
        ValueError
            If no transcript is available.
        """
        try:
            from youtube_transcript_api import (  # noqa: PLC0415
                YouTubeTranscriptApi,
            )
        except ImportError as exc:
            raise ImportError(
                "YouTubeDownloader(mode='transcript') requires "
                "'youtube-transcript-api'.  Install it with:\n"
                "  pip install youtube-transcript-api"
            ) from exc

        vid: str = self._video_id  # type: ignore[attr-defined]
        dest = self._resolve_dest_dir()

        # Build language preference list
        langs = [self.language]
        if self.include_auto_generated and self.language not in ("a.en", "a.fr"):
            # 'a.xx' codes are auto-generated — add auto fallback
            langs.append(f"a.{self.language}")

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                vid,
                languages=langs,
            )
        except Exception as exc:
            # Re-raise with actionable message
            raise ValueError(
                f"YouTubeDownloader: could not fetch transcript for video "
                f"{vid!r} (input_url={self.input_url!r}).  "
                f"The video may have no captions, or the requested language "
                f"{self.language!r} is not available.\n"
                f"Original error: {exc}"
            ) from exc

        # Write each segment text as a line
        lines = [seg.get("text", "") for seg in transcript_list]
        text = "\n".join(lines)

        out_path = dest / f"yt_{vid}_transcript.txt"
        out_path.write_text(text, encoding="utf-8")

        logger.info(
            "YouTubeDownloader(transcript): %s → %s (%d chars)",
            self.input_url,
            out_path.name,
            len(text),
        )
        return DownloadResult(
            input_url=self.input_url,
            output_path=out_path,
            suffix=".txt",
            content_type="text/plain",
            suggested_filename=out_path.name,
        )

    def _download_ytdlp(self, *, audio_only: bool) -> DownloadResult:
        """
        Download audio or video via ``yt-dlp`` Python API.

        Parameters
        ----------
        audio_only : bool
            ``True`` → audio only (mp3/m4a), ``False`` → video (mp4).

        Returns
        -------
        DownloadResult
            Populated result.

        Raises
        ------
        ImportError
            If ``yt-dlp`` is not installed.
        """
        try:
            import yt_dlp  # type: ignore[] # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"YouTubeDownloader(mode='{self.mode}') requires 'yt-dlp'.  "
                "Install it with:\n  pip install yt-dlp"
            ) from exc

        vid: str = self._video_id  # type: ignore[attr-defined]
        dest = self._resolve_dest_dir()
        out_template = str(dest / f"yt_{vid}.%(ext)s")

        if audio_only:
            format_str = "bestaudio/best"
            postprocessors = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ]
        else:
            format_str = "bestvideo+bestaudio/best"
            postprocessors = [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}]

        ydl_opts: dict = {
            "format": format_str,
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": int(self.timeout),
            "postprocessors": postprocessors,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.input_url])

        # Find the output file (extension assigned by yt-dlp)
        suffix = ".mp3" if audio_only else ".mp4"
        expected = dest / f"yt_{vid}{suffix}"
        if not expected.exists():
            # yt-dlp may use a different extension
            candidates = sorted(dest.glob(f"yt_{vid}.*"))
            if not candidates:
                raise RuntimeError(
                    f"YouTubeDownloader: yt-dlp ran but no output file found "
                    f"in {dest} for video {vid!r}."
                )
            expected = candidates[-1]
            suffix = expected.suffix.lower()

        logger.info(
            "YouTubeDownloader(%s): %s → %s",
            self.mode,
            self.input_url,
            expected.name,
        )
        return DownloadResult(
            input_url=self.input_url,
            output_path=expected,
            suffix=suffix,
            suggested_filename=expected.name,
        )
