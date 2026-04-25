# scikitplot/corpus/_url_handler.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._url_handler
===============================
URL classification, resolution, and secure download for the corpus pipeline.

This module bridges the gap between arbitrary URLs and the local-file-based
:class:`~scikitplot.corpus._base.DocumentReader` system.  Given a URL, it:

1. **Classifies** the URL into one of several kinds (web page, YouTube,
   downloadable file, Google Drive share, GitHub blob/raw).
2. **Resolves** provider-specific URLs to direct download links (e.g. Google
   Drive share → ``/uc?export=download``, GitHub blob → raw).
3. **Downloads** the resource to a local temporary file with security guards
   (SSRF prevention, size limits, timeout, streaming).

The caller — typically
:meth:`~scikitplot.corpus._corpus_builder.CorpusBuilder._ingest_source` —
receives a local :class:`pathlib.Path` that can be passed directly to
:meth:`~scikitplot.corpus._base.DocumentReader.create`.

Design invariants:

* **Zero optional dependencies at import time.** ``urllib`` and ``http``
  are stdlib. ``requests`` is imported lazily in ``_download_with_requests``
  and falls back to ``urllib.request`` if unavailable.
* **SSRF prevention:** Before connecting, resolved IP addresses are checked
  against RFC-1918 / loopback / link-local ranges.  This blocks redirect-based
  SSRF attacks where an attacker's URL 302-redirects to ``http://169.254.169.254``.
* **Deterministic temp paths:** Downloaded files use a SHA-256 prefix of the URL
  as the filename stem so repeated downloads of the same URL hit the same path
  (enables naive caching when ``output_path`` is reused across builds).
* **Content-type fallback:** When the URL path has no recognisable extension,
  the ``Content-Type`` header is used to infer one.
* **Caller owns cleanup:** The caller is responsible for deleting the returned
  temp file or the ``output_path`` after processing.  ``CorpusBuilder`` uses a
  ``tempfile.TemporaryDirectory`` context manager for this.

Python compatibility:

Python 3.8 through 3.15. ``from __future__ import annotations`` for PEP-604.
"""  # noqa: D205, D400

from __future__ import annotations

import hashlib
import ipaddress
import logging
import mimetypes
import os
import re
import socket
import sys
import tempfile
import urllib.error  # noqa: F401
import urllib.parse
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple  # noqa: F401

if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnumBase
else:

    class _StrEnumBase(str, Enum):  # type: ignore[no-redef]
        """Backport of ``enum.StrEnum`` for Python < 3.11.

        Enables direct string comparison: ``URLKind.WEB_PAGE == "web_page"``.
        On Python 3.11+ the stdlib ``enum.StrEnum`` is used instead.
        """


logger = logging.getLogger(__name__)

__all__ = [
    "URLKind",
    "classify_url",
    "download_url",
    "infer_extension",
    "probe_url_kind",
    "resolve_url",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default maximum download size in bytes (500 MB).
DEFAULT_MAX_DOWNLOAD_BYTES: int = 500 * 1024 * 1024

#: Default HTTP timeout in seconds.
DEFAULT_TIMEOUT_SECONDS: int = 120

#: Default maximum number of HTTP redirects to follow.
DEFAULT_MAX_REDIRECTS: int = 10

#: Default maximum retry attempts for transient HTTP errors.
DEFAULT_MAX_RETRIES: int = 3

#: Default base delay in seconds between retry attempts (exponential back-off).
DEFAULT_RETRY_BACKOFF_BASE: float = 1.0

#: HTTP status codes that indicate a transient server-side error and should
#: trigger an automatic retry.  Client errors (4xx except 429) are *not*
#: retried — they indicate a permanent problem with the request itself.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset(
    {
        429,  # Too Many Requests  — back off and retry
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    }
)

#: User-Agent header used for downloads.
_USER_AGENT: str = (
    "Mozilla/5.0 (compatible; scikitplot-corpus/1.0; "
    "+https://github.com/scikit-plots/scikit-plots)"
)

#: Content-Type → extension fallback mapping.
_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/zip": ".zip",
    "application/x-tar": ".tar",
    "application/gzip": ".tar.gz",
    "application/x-gzip": ".tar.gz",
    "application/x-bzip2": ".tar.bz2",
    "application/x-xz": ".tar.xz",
    "application/x-7z-compressed": ".7z",
    "text/plain": ".txt",
    "text/html": ".html",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "text/xml": ".xml",
    "application/xml": ".xml",
    "application/json": ".json",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/x-aac": ".aac",
    "audio/x-m4a": ".m4a",
    "video/mp4": ".mp4",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-flv": ".flv",
    "video/x-m4v": ".m4v",
    "video/x-ms-wmv": ".wmv",
    "video/3gpp": ".3gp",
    "image/svg+xml": ".svg",
}

#: Default HTTP timeout for probing in seconds (short — just HEAD).
DEFAULT_PROBE_TIMEOUT_SECONDS: int = 15

#: Content-Type MIME type prefixes that indicate a downloadable (non-HTML) resource.
#: The value ``True`` means "treat as DOWNLOADABLE regardless of subtype".
#: Checked by prefix so ``audio/mpeg`` matches the ``audio/`` key.
_DOWNLOADABLE_MIME_PREFIXES: tuple[str, ...] = (
    "application/pdf",
    "application/zip",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "application/x-bzip2",
    "application/x-xz",
    "application/x-7z-compressed",
    "application/vnd.openxmlformats-officedocument",
    "application/msword",
    "application/vnd.ms-",
    "application/octet-stream",
    "application/json",
    "application/xml",
    "text/csv",
    "text/plain",
    "text/markdown",
    "text/xml",
    "image/",
    "audio/",
    "video/",
)

# ---------------------------------------------------------------------------
# URL regex patterns
# ---------------------------------------------------------------------------

# Individual video: watch, shorts, embed, live, youtu.be
_YOUTUBE_VIDEO_RE: re.Pattern[str] = re.compile(
    r"https?://(www\.)?(youtube\.com/(watch|shorts/|embed/|live/)|youtu\.be/)",
    re.IGNORECASE,
)

# Channel / handle pages (all tab variants + legacy URL schemes)
#   @Handle, @Handle/shorts, @Handle/videos, @Handle/podcasts,
#   @Handle/streams, @Handle/community, @Handle/about, @Handle/featured
#   channel/UCxxx, c/Name (legacy), user/Name (legacy)
_YOUTUBE_CHANNEL_RE: re.Pattern[str] = re.compile(
    r"https?://(www\.)?youtube\.com/"
    r"(@[^/?#]+"
    r"|channel/[^/?#]+"
    r"|c/[^/?#]+"
    r"|user/[^/?#]+)"
    r"(/shorts|/videos|/podcasts|/streams|/community|/about|/featured)?"
    r"/?$",
    re.IGNORECASE,
)

# Pure playlist page (no video ID — only list= param)
_YOUTUBE_PLAYLIST_RE: re.Pattern[str] = re.compile(
    r"https?://(www\.)?youtube\.com/playlist\?",
    re.IGNORECASE,
)

_GDRIVE_FILE_RE: re.Pattern[str] = re.compile(
    r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)

_GDRIVE_OPEN_RE: re.Pattern[str] = re.compile(
    r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)

_GITHUB_BLOB_RE: re.Pattern[str] = re.compile(
    r"https?://github\.com/([^/]+/[^/]+)/blob/(.+)",
    re.IGNORECASE,
)

_GITHUB_RAW_RE: re.Pattern[str] = re.compile(
    r"https?://raw\.githubusercontent\.com/",
    re.IGNORECASE,
)

# File extensions that indicate a downloadable resource (not a web page).
_DOWNLOADABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Documents
        ".pdf",
        ".doc",
        ".docx",
        ".odt",
        ".rtf",
        # Spreadsheets
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls",
        ".ods",
        # Text / markup
        ".txt",
        ".md",
        ".rst",
        ".xml",
        ".json",
        ".jsonl",
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".tiff",
        ".tif",
        ".bmp",
        ".svg",
        # Audio
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".wma",
        ".aac",
        ".aiff",
        ".opus",
        ".wv",
        # Video
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".webm",
        ".m4v",
        ".wmv",
        ".flv",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".tar.gz",
        ".tar.bz2",
        ".tar.xz",
        # Code
        ".py",
        ".js",
        ".ts",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
    }
)


# ===========================================================================
# URLKind — classification result
# ===========================================================================


class URLKind(_StrEnumBase):
    """
    Classification of a URL for routing to the correct handler.

    Attributes
    ----------
    WEB_PAGE
        General web page — route to ``WebReader``.
    YOUTUBE
        Single YouTube video (watch, shorts, embed, live, youtu.be)
        — route to ``YouTubeReader``.
    YOUTUBE_CHANNEL
        YouTube channel or handle page (``@Handle``, ``/channel/``,
        ``/c/``, ``/user/``), with any tab suffix (``/shorts``,
        ``/videos``, ``/podcasts``, etc.)
        — enumerate videos via ``yt-dlp`` then route each to
        ``YouTubeReader``.
    YOUTUBE_PLAYLIST
        Pure playlist page (``/playlist?list=…``, no ``v=`` param)
        — enumerate videos via ``yt-dlp`` then route each to
        ``YouTubeReader``.
    DOWNLOADABLE
        Direct-download file (PDF, image, audio, etc.) — download
        locally, then route to ``DocumentReader.create()``.
    GOOGLE_DRIVE
        Google Drive share link — resolve to direct download URL,
        then download and route to ``DocumentReader.create()``.
    GITHUB_RAW
        GitHub raw URL (``raw.githubusercontent.com``) — treat as
        direct download.
    GITHUB_BLOB
        GitHub blob URL (``github.com/.../blob/...``) — resolve to
        raw URL, then download.
    """

    WEB_PAGE = "web_page"
    YOUTUBE = "youtube"
    YOUTUBE_CHANNEL = "youtube_channel"
    YOUTUBE_PLAYLIST = "youtube_playlist"
    DOWNLOADABLE = "downloadable"
    GOOGLE_DRIVE = "google_drive"
    GITHUB_RAW = "github_raw"
    GITHUB_BLOB = "github_blob"


# ===========================================================================
# classify_url
# ===========================================================================


def classify_url(url: str) -> URLKind:  # noqa: PLR0911
    """
    Classify a URL into one of the known :class:`URLKind` categories.

    Parameters
    ----------
    url : str
        Full URL string. Must start with ``http://`` or ``https://``.

    Returns
    -------
    URLKind
        Classification result.

    Raises
    ------
    ValueError
        If *url* does not start with ``http://`` or ``https://``.

    Notes
    -----
    **Classification order matters.** The check sequence is:

    1. YouTube channel / handle (checked before video — the ``@Handle``
       path would otherwise fall through to the web-page fallback).
    2. YouTube playlist (``/playlist?list=…``).
    3. YouTube single video (``watch``, ``shorts``, ``embed``, ``live``,
       ``youtu.be``).  A ``watch?v=…&list=…`` URL is classified as a
       single video — the ``list=`` param is contextual and the reader
       extracts only the ``v=`` video ID.
    4. Google Drive share links.
    5. GitHub blob URLs (must check before raw, since blob URLs
       are on ``github.com``).
    6. GitHub raw URLs.
    7. Downloadable file (extension-based heuristic on the URL path).
    8. Web page (default fallback).

    Examples
    --------
    >>> classify_url("https://youtu.be/rwPISgZcYIk")
    <URLKind.YOUTUBE: 'youtube'>
    >>> classify_url("https://www.youtube.com/watch?v=4nMSvDEYl1c")
    <URLKind.YOUTUBE: 'youtube'>
    >>> classify_url("https://www.youtube.com/shorts/-6hoqujlmfU")
    <URLKind.YOUTUBE: 'youtube'>
    >>> classify_url("https://www.youtube.com/watch?v=AAk3pi15Zn4&list=PLL4_zLP7J")
    <URLKind.YOUTUBE: 'youtube'>
    >>> classify_url("https://www.youtube.com/@WHO/videos")
    <URLKind.YOUTUBE_CHANNEL: 'youtube_channel'>
    >>> classify_url("https://www.youtube.com/@WHO/shorts")
    <URLKind.YOUTUBE_CHANNEL: 'youtube_channel'>
    >>> classify_url("https://www.youtube.com/playlist?list=PLL4_zLP7J")
    <URLKind.YOUTUBE_PLAYLIST: 'youtube_playlist'>
    >>> classify_url("https://example.com/report.pdf")
    <URLKind.DOWNLOADABLE: 'downloadable'>
    >>> classify_url("https://drive.google.com/file/d/abc123/view")
    <URLKind.GOOGLE_DRIVE: 'google_drive'>
    >>> classify_url("https://example.com/article")
    <URLKind.WEB_PAGE: 'web_page'>
    """
    if not isinstance(url, str) or not re.match(r"https?://", url, re.IGNORECASE):
        raise ValueError(
            f"classify_url: url must start with 'http://' or 'https://'; got {url!r}."
        )

    # 1. YouTube channel / handle — must come before video: @Handle paths
    #    share the youtube.com domain with watch/shorts but are not videos.
    if _YOUTUBE_CHANNEL_RE.match(url):
        return URLKind.YOUTUBE_CHANNEL

    # 2. YouTube pure playlist (no v= param)
    if _YOUTUBE_PLAYLIST_RE.match(url):
        return URLKind.YOUTUBE_PLAYLIST

    # 3. YouTube single video (watch, shorts, embed, live, youtu.be)
    #    watch?v=…&list=… is intentionally classified as YOUTUBE (single
    #    video); the reader extracts only the v= param.
    if _YOUTUBE_VIDEO_RE.match(url):
        return URLKind.YOUTUBE

    # 4. Google Drive
    if _GDRIVE_FILE_RE.match(url) or _GDRIVE_OPEN_RE.match(url):
        return URLKind.GOOGLE_DRIVE

    # 5. GitHub blob
    if _GITHUB_BLOB_RE.match(url):
        return URLKind.GITHUB_BLOB

    # 6. GitHub raw
    if _GITHUB_RAW_RE.match(url):
        return URLKind.GITHUB_RAW

    # 5. Downloadable file (extension-based)
    parsed = urllib.parse.urlparse(url)
    path_lower = parsed.path.lower()
    # Check for compound extensions first (e.g. ".tar.gz")
    for compound_ext in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if path_lower.endswith(compound_ext):
            return URLKind.DOWNLOADABLE
    # Then single extensions
    _, ext = os.path.splitext(path_lower)
    if ext and ext in _DOWNLOADABLE_EXTENSIONS:
        return URLKind.DOWNLOADABLE

    # 6. Default: web page
    return URLKind.WEB_PAGE


# ===========================================================================
# probe_url_kind — HEAD-request-based classification for extensionless URLs
# ===========================================================================


def probe_url_kind(
    url: str,
    *,
    timeout: int = DEFAULT_PROBE_TIMEOUT_SECONDS,
    skip_ssrf_check: bool = False,
) -> URLKind:
    """Probe a URL with a HEAD request to classify by Content-Type.

    Use this when :func:`classify_url` returns :attr:`URLKind.WEB_PAGE`
    but the URL has no file extension in the path (e.g. API endpoints
    like ``/content``, ``/download``, ``/bitstream``).  A HEAD request
    is sent first; if that fails a GET with ``stream=True`` is attempted
    (some servers reject HEAD).  The ``Content-Type`` response header is
    read and mapped to the correct :class:`URLKind`.

    Parameters
    ----------
    url : str
        HTTP/HTTPS URL to probe.
    timeout : int, optional
        Connection + read timeout in seconds.  Default: 15.
    skip_ssrf_check : bool, optional
        Skip SSRF prevention check.  Only for trusted internal URLs.
        Default: ``False``.

    Returns
    -------
    URLKind
        The inferred classification:

        - :attr:`URLKind.DOWNLOADABLE` — Content-Type indicates a
          non-HTML binary or structured file (PDF, image, audio, video,
          archive, CSV, JSON, plain text, etc.).
        - :attr:`URLKind.WEB_PAGE` — Content-Type is ``text/html`` or
          the probe failed (fail-safe: treat as web page so the caller
          can still attempt WebReader).

    Raises
    ------
    ValueError
        If *url* does not start with ``http://`` or ``https://``.

    Notes
    -----
    **When to call this**: Only when :func:`classify_url` returns
    ``WEB_PAGE`` *and* the URL path has no recognisable file extension.
    For URLs that already have a known extension or are already
    classified as YOUTUBE / GOOGLE_DRIVE / GITHUB_*, call the faster
    :func:`classify_url` directly.

    **Network cost**: One HEAD request (no body download).  Adds
    ~50-500 ms of latency depending on server and network.

    **Thread safety**: This function is stateless and safe to call
    from multiple threads.

    Developer note:

    The function tries ``requests`` first (better redirect + timeout
    handling).  It falls back to ``urllib.request`` if requests is not
    installed.  The SSRF check is applied before connecting when
    ``skip_ssrf_check=False``.

    Examples
    --------
    >>> # An API endpoint returning a PDF with no extension in path
    >>> kind = probe_url_kind(
    ...     "https://iris.who.int/server/api/core/bitstreams/abc/content"
    ... )
    >>> kind == URLKind.DOWNLOADABLE
    True  # Content-Type: application/pdf

    >>> # A normal web page
    >>> kind = probe_url_kind("https://www.example.com/about")
    >>> kind == URLKind.WEB_PAGE
    True  # Content-Type: text/html
    """
    if not isinstance(url, str) or not re.match(r"https?://", url, re.IGNORECASE):
        raise ValueError(
            f"probe_url_kind: url must start with 'http://' or 'https://'; got {url!r}."
        )

    if not skip_ssrf_check:
        _validate_url_security(url)

    content_type = _probe_content_type(url, timeout=timeout)
    return _classify_content_type(content_type)


def _probe_content_type(url: str, *, timeout: int) -> str:
    """Send a HEAD (or fallback GET) request and return the Content-Type.

    Parameters
    ----------
    url : str
        URL to probe.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Content-Type header value, lower-cased and stripped.
        Returns ``""`` on any network error.

    Notes
    -----
    HEAD is tried first.  If the server returns 405 Method Not Allowed
    or a connection error, a streaming GET is attempted and immediately
    closed after reading the headers.  Both paths respect *timeout*.

    Developer note:

    ``requests`` is preferred over ``urllib`` because it handles
    redirects automatically (important for CDN/API links that redirect
    to the actual file).  ``urllib`` is used as a stdlib fallback.
    """
    try:
        return _probe_with_requests(url, timeout=timeout)
    except ImportError:
        return _probe_with_urllib(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.debug("probe_url_kind: requests probe failed for %s: %s", url, exc)
        return ""


def _probe_with_requests(url: str, *, timeout: int) -> str:
    """Probe with the ``requests`` library.

    Parameters
    ----------
    url : str
        URL to probe.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Lower-cased Content-Type or ``""`` on failure.

    Raises
    ------
    ImportError
        If ``requests`` is not installed.
    """
    import requests  # noqa: PLC0415

    # Use the session as a context manager so the underlying urllib3
    # connection pool is released even if an exception propagates.
    with requests.Session() as session:
        session.headers["User-Agent"] = _USER_AGENT

        # Try HEAD first — no body, minimal bandwidth
        try:
            resp = session.head(url, allow_redirects=True, timeout=timeout)
            if resp.status_code not in (405, 501):
                ct = resp.headers.get("Content-Type", "")
                return ct.split(";")[0].strip().lower()
        except Exception:  # noqa: BLE001
            pass

        # Fallback: GET with stream — read headers only, close immediately
        try:
            resp = session.get(url, stream=True, allow_redirects=True, timeout=timeout)
            resp.close()
            ct = resp.headers.get("Content-Type", "")
            return ct.split(";")[0].strip().lower()
        except Exception as exc:  # noqa: BLE001
            logger.debug("_probe_with_requests: GET stream probe failed: %s", exc)
            return ""


def _probe_with_urllib(url: str, *, timeout: int) -> str:
    """Probe with ``urllib.request`` (stdlib fallback).

    Parameters
    ----------
    url : str
        URL to probe.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Lower-cased Content-Type or ``""`` on failure.
    """
    try:
        req = urllib.request.Request(  # noqa: S310
            url,
            method="HEAD",
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            ct = resp.headers.get("Content-Type", "")
            return ct.split(";")[0].strip().lower()
    except Exception:  # noqa: BLE001
        pass

    # HEAD failed — try GET and read only headers
    try:
        req = urllib.request.Request(  # noqa: S310
            url,
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            ct = resp.headers.get("Content-Type", "")
            return ct.split(";")[0].strip().lower()
    except Exception as exc:  # noqa: BLE001
        logger.debug("_probe_with_urllib: probe failed for %s: %s", url, exc)
        return ""


def _classify_content_type(content_type: str) -> URLKind:
    """Map a MIME content-type string to a :class:`URLKind`.

    Parameters
    ----------
    content_type : str
        Lower-cased MIME type (e.g. ``"application/pdf"``).
        May be empty string.

    Returns
    -------
    URLKind
        :attr:`URLKind.DOWNLOADABLE` if the type matches any prefix in
        :data:`_DOWNLOADABLE_MIME_PREFIXES`; :attr:`URLKind.WEB_PAGE`
        otherwise (including when *content_type* is empty — fail-safe).

    Notes
    -----
    ``text/plain`` is treated as DOWNLOADABLE rather than WEB_PAGE
    because plain-text files (transcripts, code, data) should be
    routed to :class:`TextReader`, not :class:`WebReader`.
    ``text/html`` stays WEB_PAGE — HTML is a web page by definition.
    """
    if not content_type:
        return URLKind.WEB_PAGE

    for prefix in _DOWNLOADABLE_MIME_PREFIXES:
        if content_type.startswith(prefix):
            logger.debug(
                "_classify_content_type: %r → DOWNLOADABLE (prefix=%r)",
                content_type,
                prefix,
            )
            return URLKind.DOWNLOADABLE

    logger.debug(
        "_classify_content_type: %r → WEB_PAGE (no matching prefix)",
        content_type,
    )
    return URLKind.WEB_PAGE


def resolve_url(url: str, kind: URLKind | None = None) -> str:
    """
    Resolve a provider-specific URL to a direct-download URL.

    Parameters
    ----------
    url : str
        Original URL.
    kind : URLKind or None, optional
        Pre-computed classification. If ``None``, :func:`classify_url`
        is called. Default: ``None``.

    Returns
    -------
    str
        Direct-download URL. For ``WEB_PAGE`` and ``YOUTUBE`` kinds,
        the original URL is returned unchanged (they are not
        download targets). For ``DOWNLOADABLE`` and ``GITHUB_RAW``,
        the URL is returned as-is (already direct). For
        ``GOOGLE_DRIVE`` and ``GITHUB_BLOB``, the resolved URL
        is returned.

    Raises
    ------
    ValueError
        If the URL cannot be resolved (e.g. malformed Google Drive link).

    Examples
    --------
    >>> resolve_url("https://drive.google.com/file/d/abc123/view")
    'https://drive.google.com/uc?export=download&id=abc123'
    >>> resolve_url("https://github.com/user/repo/blob/main/data.csv")
    'https://raw.githubusercontent.com/user/repo/main/data.csv'
    """
    if kind is None:
        kind = classify_url(url)

    if kind in (
        URLKind.WEB_PAGE,
        URLKind.YOUTUBE,
        URLKind.DOWNLOADABLE,
        URLKind.GITHUB_RAW,
    ):
        return url

    if kind == URLKind.GOOGLE_DRIVE:
        return _resolve_gdrive(url)

    if kind == URLKind.GITHUB_BLOB:
        return _resolve_github_blob(url)

    return url  # pragma: no cover — defensive


def _resolve_gdrive(url: str) -> str:
    """
    Resolve a Google Drive share link to a direct download URL.

    Parameters
    ----------
    url : str
        Google Drive share URL.

    Returns
    -------
    str
        Direct download URL via ``/uc?export=download``.

    Raises
    ------
    ValueError
        If the file ID cannot be extracted.
    """
    match = _GDRIVE_FILE_RE.search(url) or _GDRIVE_OPEN_RE.search(url)
    if match is None:
        raise ValueError(
            f"_resolve_gdrive: cannot extract file ID from {url!r}. "
            f"Expected format: https://drive.google.com/file/d/FILE_ID/..."
        )
    file_id = match.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _resolve_github_blob(url: str) -> str:
    """
    Resolve a GitHub blob URL to a raw.githubusercontent.com URL.

    Parameters
    ----------
    url : str
        GitHub blob URL.

    Returns
    -------
    str
        Raw content URL.

    Raises
    ------
    ValueError
        If the URL does not match the expected blob pattern.
    """
    match = _GITHUB_BLOB_RE.match(url)
    if match is None:
        raise ValueError(
            f"_resolve_github_blob: cannot parse {url!r}. "
            f"Expected: https://github.com/OWNER/REPO/blob/REF/PATH"
        )
    repo = match.group(1)  # "owner/repo"
    rest = match.group(2)  # "main/path/to/file.txt"
    return f"https://raw.githubusercontent.com/{repo}/{rest}"


# ===========================================================================
# SSRF prevention
# ===========================================================================


def _is_private_ip(hostname: str) -> bool:
    """
    Check if a hostname resolves to a private/loopback/link-local IP.

    Parameters
    ----------
    hostname : str
        Hostname or IP address string.

    Returns
    -------
    bool
        ``True`` if any resolved address is private, loopback,
        link-local, or reserved.

    Notes
    -----
    This is the SSRF prevention gate. It is called before every HTTP
    request to block attempts to reach internal services via redirects.
    """
    try:
        # getaddrinfo returns [(family, type, proto, canonname, sockaddr), ...]
        results = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, OSError):
        # DNS resolution failed — allow the caller to handle the error
        return False

    for _family, _type, _proto, _canonname, sockaddr in results:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        ):
            return True

    return False


def _validate_url_security(url: str) -> None:
    """
    Validate that a URL does not target a private/internal network.

    Parameters
    ----------
    url : str
        URL to validate.

    Raises
    ------
    ValueError
        If the URL targets a private, loopback, or link-local address.
    ValueError
        If the URL scheme is not ``http`` or ``https``.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"_validate_url_security: unsupported scheme {parsed.scheme!r}. "
            f"Only 'http' and 'https' are allowed."
        )
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"_validate_url_security: no hostname in URL {url!r}.")
    if _is_private_ip(hostname):
        raise ValueError(
            f"_validate_url_security: URL {url!r} resolves to a private "
            f"or internal IP address. Refusing to connect (SSRF prevention)."
        )


# ===========================================================================
# download_url — secure download to local temp file
# ===========================================================================


def _infer_extension_from_headers(  # noqa: PLR0911, PLR0912
    headers: Any,
    url: str,
) -> str:
    """
    Infer file extension from HTTP response headers or URL path.

    The resolution order is deliberately:

    1. **URL path** — cheapest and most reliable when present.
    2. **Content-Disposition** — servers explicitly declare the filename;
       this is authoritative when the URL path has no extension.
    3. **Content-Type** — MIME-based inference; skips ``application/octet-stream``
       because that type is deliberately generic ("unknown binary") and would
       produce ``.bin`` prematurely, preventing step 2 from being reached.
    4. **Fallback** — ``.bin`` when nothing can be inferred.

    Parameters
    ----------
    headers : http.client.HTTPMessage or dict-like
        Response headers.
    url : str
        Original URL (used as fallback for path-based extension).

    Returns
    -------
    str
        File extension including leading dot (e.g. ``".pdf"``).
        Returns ``".bin"`` if nothing can be inferred.
    """
    # ── 1. URL path ──────────────────────────────────────────────────────
    parsed = urllib.parse.urlparse(url)
    path_lower = parsed.path.lower()
    for compound_ext in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if path_lower.endswith(compound_ext):
            return compound_ext
    _, ext = os.path.splitext(path_lower)
    if ext and ext in _DOWNLOADABLE_EXTENSIONS:
        return ext

    # ── 2. Content-Disposition (most authoritative when URL has no ext) ──
    cd = None
    if hasattr(headers, "get"):
        cd = headers.get("Content-Disposition", "")
    if cd:
        # RFC 5987 encoded form: filename*=UTF-8''report%20final.pdf
        # Must be checked FIRST because the plain-form regex also partially
        # matches the star variant (captures the charset prefix as fname).
        m5987 = re.search(
            r"filename\*=[^']*''([^;\s]+)",
            cd,
            re.IGNORECASE,
        )
        if m5987:
            try:
                import urllib.parse as _up  # noqa: PLC0415

                fname = _up.unquote(m5987.group(1).strip().rstrip('"'))
                _, cd_ext = os.path.splitext(fname)
                if cd_ext:
                    return cd_ext.lower()
            except Exception:  # noqa: BLE001
                pass
        # Plain form: filename="report.pdf" or filename=report.pdf
        m_plain = re.search(r'filename="?([^";]+)"?', cd, re.IGNORECASE)
        if m_plain:
            fname = m_plain.group(1).strip()
            _, cd_ext = os.path.splitext(fname)
            if cd_ext:
                return cd_ext.lower()

    # ── 3. Content-Type (skip octet-stream — it means "unknown binary") ─
    ct = None
    if hasattr(headers, "get"):
        ct = headers.get("Content-Type", "")
    if ct:
        # "application/pdf; charset=utf-8" → "application/pdf"
        mime = ct.split(";")[0].strip().lower()
        # Skip octet-stream: it's deliberately generic and would produce
        # ".bin" via mimetypes, preventing downstream reader dispatch.
        if mime != "application/octet-stream":
            if mime in _CONTENT_TYPE_TO_EXT:
                return _CONTENT_TYPE_TO_EXT[mime]
            # Try mimetypes stdlib
            guessed = mimetypes.guess_extension(mime)
            if guessed:
                return guessed

    # ── 4. Fallback ──────────────────────────────────────────────────────
    return ".bin"


def _make_temp_filename(url: str, extension: str) -> str:
    """
    Generate a deterministic temp filename from URL + extension.

    Parameters
    ----------
    url : str
        Source URL.
    extension : str
        File extension including leading dot.

    Returns
    -------
    str
        Filename of the form ``"skplt_{hash}{ext}"``.
    """
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"skplt_{url_hash}{extension}"


# ---------------------------------------------------------------------------
# Magic-byte detection — last-resort extension inference
# ---------------------------------------------------------------------------

#: Mapping of file-header magic bytes → extension.
#: Checked in order; first match wins.  Each entry is
#: ``(byte_prefix, extension)``.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    # Documents
    (b"%PDF", ".pdf"),
    # Archives
    (b"PK\x03\x04", ".zip"),
    (b"PK\x05\x06", ".zip"),
    (b"\x1f\x8b", ".tar.gz"),
    (b"BZh", ".tar.bz2"),
    (b"\xfd7zXZ\x00", ".tar.xz"),
    (b"7z\xbc\xaf\x27\x1c", ".7z"),
    # Images
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"RIFF", ".webp"),  # RIFF....WEBP — checked below with content
    (b"II\x2a\x00", ".tiff"),
    (b"MM\x00\x2a", ".tiff"),
    (b"BM", ".bmp"),
    # Audio
    (b"ID3", ".mp3"),
    (b"\xff\xfb", ".mp3"),
    (b"\xff\xf3", ".mp3"),
    (b"\xff\xf2", ".mp3"),
    (b"fLaC", ".flac"),
    (b"OggS", ".ogg"),
    (b"RIFF", ".wav"),  # RIFF....WAVE — disambiguated below
    # Video
    (b"\x1a\x45\xdf\xa3", ".mkv"),
    # XML / HTML (text-based, check late)
    (b"<?xml", ".xml"),
    (b"<!DOCTYPE html", ".html"),
    (b"<html", ".html"),
)


def _detect_extension_from_magic(path: Path) -> str | None:  # noqa: PLR0911
    """Detect file extension from the first bytes of a file.

    Parameters
    ----------
    path : Path
        Path to the file to inspect.

    Returns
    -------
    str or None
        Detected extension (e.g. ``".pdf"``, ``".png"``), or ``None``
        if no known signature matches.

    Notes
    -----
    **RIFF disambiguation:** Both WAV and WEBP use the ``RIFF`` magic
    prefix.  The file sub-type at offset 8 distinguishes them:
    ``WAVE`` → ``.wav``, ``WEBP`` → ``.webp``.

    **MP4/MOV/M4A detection:** The ISO base media file format (MP4,
    MOV, M4A, M4V) uses ``ftyp`` at offset 4.  The brand bytes at
    offset 8 distinguish sub-types.

    This function reads at most 32 bytes and is safe to call on any
    file, including empty files (returns ``None``).
    """
    try:
        with open(path, "rb") as f:
            header = f.read(32)
    except OSError:
        return None

    if len(header) < 4:  # noqa: PLR2004
        return None

    # ── RIFF disambiguation (WAV vs WEBP) ────────────────────────
    if header[:4] == b"RIFF" and len(header) >= 12:  # noqa: PLR2004
        sub_type = header[8:12]
        if sub_type == b"WEBP":
            return ".webp"
        if sub_type == b"WAVE":
            return ".wav"
        if sub_type == b"AVI ":
            return ".avi"

    # ── ISO base media (MP4/MOV/M4A) ─────────────────────────────
    if len(header) >= 8 and header[4:8] == b"ftyp":  # noqa: PLR2004
        brand = header[8:12] if len(header) >= 12 else b""  # noqa: PLR2004
        if brand in (b"M4A ", b"M4B "):
            return ".m4a"
        if brand in (b"qt  ",):  # noqa: FURB171
            return ".mov"
        # Default: MP4 covers isom, mp41, mp42, avc1, etc.
        return ".mp4"

    # ── Standard prefix matching ─────────────────────────────────
    for magic, ext in _MAGIC_SIGNATURES:
        if header[: len(magic)] == magic:
            # Skip RIFF here — already handled above
            if magic == b"RIFF":
                continue
            return ext

    return None


def _fixup_bin_extension(dest_path: Path) -> Path:
    """Rename a ``.bin`` file to its correct extension via magic-byte detection.

    Parameters
    ----------
    dest_path : Path
        Path to the downloaded file.  If the suffix is not ``.bin``,
        the file is returned unchanged.

    Returns
    -------
    Path
        The (possibly renamed) file path.

    Notes
    -----
    This is the safety net for URLs served with ``Content-Type:
    application/octet-stream`` and no extension in the URL path or
    Content-Disposition header.  Without this, the file would be named
    ``*.bin`` and ``DocumentReader.create()`` would fail with
    "No DocumentReader registered for extension '.bin'".
    """
    if dest_path.suffix.lower() != ".bin":
        return dest_path

    detected = _detect_extension_from_magic(dest_path)
    if detected is None:
        logger.debug(
            "_fixup_bin_extension: no magic signature detected in %s; "
            "keeping .bin extension.",
            dest_path.name,
        )
        return dest_path

    new_path = dest_path.with_suffix(detected)
    try:
        dest_path.rename(new_path)
        logger.info(
            "_fixup_bin_extension: renamed %s → %s (detected %s via magic bytes)",
            dest_path.name,
            new_path.name,
            detected,
        )
        return new_path
    except OSError as exc:
        logger.warning(
            "_fixup_bin_extension: failed to rename %s → %s: %s",
            dest_path.name,
            new_path.name,
            exc,
        )
        return dest_path


def infer_extension(headers: Any, url: str) -> str:
    """Infer a file extension from HTTP response headers and URL path.

    Public wrapper around :func:`_infer_extension_from_headers`.  Call
    this when you already hold response headers (e.g. after a manual
    ``requests.head()``) and want to know what extension to use for the
    downloaded file.

    The resolution order is:

    1. URL path extension (cheapest, most reliable when present).
    2. ``Content-Disposition`` ``filename*=`` (RFC 5987 encoded form).
    3. ``Content-Disposition`` ``filename=`` (plain form).
    4. ``Content-Type`` MIME mapping (skips ``application/octet-stream``).
    5. ``mimetypes.guess_extension`` stdlib fallback.
    6. ``".bin"`` when nothing can be inferred.

    Parameters
    ----------
    headers : dict-like or http.client.HTTPMessage
        HTTP response headers that support ``.get(key, default)``.
    url : str
        Original request URL.  Used for path-based extension lookup
        and as a logging label.

    Returns
    -------
    str
        File extension including leading dot, e.g. ``".pdf"``.
        Returns ``".bin"`` if nothing can be inferred.

    Examples
    --------
    >>> infer_extension({"Content-Type": "audio/mpeg"}, "https://host/dl")
    '.mp3'
    >>> infer_extension({}, "https://host/report.pdf")
    '.pdf'
    >>> infer_extension(
    ...     {"Content-Disposition": "attachment; filename*=UTF-8''report%20final.pdf"},
    ...     "https://host/dl",
    ... )
    '.pdf'
    """
    return _infer_extension_from_headers(headers, url)


def _extract_http_status(exc: BaseException) -> int | None:
    """Extract the HTTP status code from a requests or urllib exception.

    Parameters
    ----------
    exc : BaseException
        Exception to inspect.

    Returns
    -------
    int or None
        HTTP status code if detectable, ``None`` otherwise.
    """
    # requests.HTTPError stores the response on exc.response
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    # urllib.error.HTTPError subclasses URLError and exposes .code
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return None


def download_url(
    url: str,
    *,
    output_path: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF_BASE,
    skip_ssrf_check: bool = False,
) -> Path:
    """
    Download a URL to a local file.

    Parameters
    ----------
    url : str
        URL to download. Must be ``http://`` or ``https://``.
    output_path : str, Path, or None, optional
        Directory to write the downloaded file into. If ``None``,
        ``tempfile.gettempdir()`` is used. Default: ``None``.
    max_bytes : int, optional
        Maximum download size in bytes. Default: 500 MB.
    timeout : int, optional
        HTTP timeout in seconds. Default: 120.
    max_redirects : int, optional
        Maximum number of HTTP redirects to follow. Default: 10.
    max_retries : int, optional
        Maximum retry attempts for transient HTTP errors (429, 500,
        502, 503, 504). Each attempt waits
        ``retry_backoff * 2 ** attempt`` seconds before retrying.
        Set to ``0`` to disable retries. Default: 3.
    retry_backoff : float, optional
        Base delay in seconds for exponential back-off. The actual
        wait before attempt *n* (0-indexed) is
        ``retry_backoff * 2 ** n`` seconds. Default: 1.0.
    skip_ssrf_check : bool, optional
        Skip SSRF prevention check. **Only** for trusted internal
        URLs. Default: ``False``.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file. The caller is responsible for
        cleanup.

    Raises
    ------
    ValueError
        If the URL is invalid, targets a private IP (SSRF), or the
        response exceeds *max_bytes*.
    urllib.error.URLError
        If the download fails due to a network error and all retries
        are exhausted.
    TimeoutError
        If the download exceeds *timeout* seconds.

    Notes
    -----
    **Security:** The URL is validated against private IP ranges before
    connecting. This prevents SSRF attacks where an attacker's URL
    redirects to an internal service.

    **Deterministic filenames:** The downloaded file uses a SHA-256
    prefix of the URL as the filename stem, so repeated downloads of
    the same URL produce the same filename.

    **Retry policy:** Only transient server-side errors trigger a retry
    (HTTP 429, 500, 502, 503, 504). Client errors (4xx except 429) and
    ``ValueError`` (SSRF, size exceeded) are *not* retried.

    Examples
    --------
    >>> path = download_url("https://example.com/report.pdf")
    >>> path.suffix
    '.pdf'
    >>> path.exists()
    True
    """
    import time  # noqa: PLC0415

    if not isinstance(url, str) or not re.match(r"https?://", url, re.IGNORECASE):
        raise ValueError(
            f"download_url: url must start with 'http://' or 'https://'; got {url!r}."
        )

    # SSRF check — performed once before any network I/O.
    if not skip_ssrf_check:
        _validate_url_security(url)

    # Resolve output_path
    if output_path is None:  # noqa: SIM108
        output_path = Path(tempfile.gettempdir())
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    total_attempts = max(1, max_retries + 1)

    for attempt in range(total_attempts):
        if attempt > 0:
            delay = retry_backoff * (2 ** (attempt - 1))
            logger.warning(
                "download_url: attempt %d/%d for %s after %.1fs back-off "
                "(previous: %s).",
                attempt + 1,
                total_attempts,
                url,
                delay,
                last_exc,
            )
            time.sleep(delay)

        try:
            # Try requests first (better redirect handling + streaming),
            # fall back to urllib when requests is not installed.
            try:
                dest_path = _download_with_requests(
                    url,
                    output_path=output_path,
                    max_bytes=max_bytes,
                    timeout=timeout,
                    max_redirects=max_redirects,
                    skip_ssrf_check=skip_ssrf_check,
                )
            except ImportError:
                logger.debug("requests library not available; falling back to urllib.")
                dest_path = _download_with_urllib(
                    url,
                    output_path=output_path,
                    max_bytes=max_bytes,
                    timeout=timeout,
                )

            # Post-download: rename .bin files whose real format can be
            # detected from magic bytes, so DocumentReader.create() does
            # not fail with "No reader for .bin".
            return _fixup_bin_extension(dest_path)

        except ValueError:
            # ValueError = permanent problem (SSRF, size exceeded).
            # Never retry — raise immediately.
            raise
        except Exception as exc:  # noqa: BLE001
            status = _extract_http_status(exc)
            if status is not None and status not in _RETRYABLE_STATUS_CODES:
                # Permanent HTTP error (e.g. 404, 403) — raise immediately.
                raise
            last_exc = exc
            if attempt >= max_retries:
                logger.error(
                    "download_url: all %d attempts exhausted for %s.",
                    total_attempts,
                    url,
                )
                raise

    # Unreachable: loop always returns or raises.
    raise RuntimeError("download_url: unexpected loop exit.")  # pragma: no cover


def _download_with_requests(
    url: str,
    *,
    output_path: Path,
    max_bytes: int,
    timeout: int,
    max_redirects: int,
    skip_ssrf_check: bool,
) -> Path:
    """
    Download using the ``requests`` library (preferred).

    Parameters
    ----------
    url : str
        URL to download.
    output_path : Path
        Destination directory.
    max_bytes : int
        Maximum download size.
    timeout : int
        HTTP timeout.
    max_redirects : int
        Maximum redirects.
    skip_ssrf_check : bool
        Skip SSRF check on redirects.

    Returns
    -------
    Path
        Downloaded file path.

    Raises
    ------
    ImportError
        If ``requests`` is not installed.
    ValueError
        If download exceeds *max_bytes* or SSRF detected on redirect.
    """
    import requests  # noqa: PLC0415

    # Use the session as a context manager so the underlying urllib3
    # connection pool is always released, even when an exception raises
    # mid-download (size exceeded, SSRF detected, network error, etc.).
    with requests.Session() as session:
        session.max_redirects = max_redirects
        session.headers["User-Agent"] = _USER_AGENT

        response = session.get(
            url,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()

        # SSRF check on final URL after redirects
        if not skip_ssrf_check and response.url != url:
            _validate_url_security(response.url)

        # Check Content-Length if available
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > max_bytes:
            raise ValueError(
                f"download_url: Content-Length {content_length} exceeds "
                f"max_bytes={max_bytes}."
            )

        # Infer extension
        ext = _infer_extension_from_headers(response.headers, url)
        filename = _make_temp_filename(url, ext)
        dest_path = output_path / filename

        # Stream download with size guard
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        # Clean up partial file before raising
                        f.close()
                        dest_path.unlink(missing_ok=True)
                        raise ValueError(
                            f"download_url: download exceeded max_bytes="
                            f"{max_bytes} (downloaded {downloaded} bytes so far)."
                        )
                    f.write(chunk)

        logger.info(
            "download_url: downloaded %s → %s (%d bytes)",
            url,
            dest_path,
            downloaded,
        )
        return dest_path


def _download_with_urllib(
    url: str,
    *,
    output_path: Path,
    max_bytes: int,
    timeout: int,
) -> Path:
    """
    Download using ``urllib.request`` (stdlib fallback).

    Parameters
    ----------
    url : str
        URL to download.
    output_path : Path
        Destination directory.
    max_bytes : int
        Maximum download size.
    timeout : int
        HTTP timeout.

    Returns
    -------
    Path
        Downloaded file path.

    Raises
    ------
    ValueError
        If download exceeds *max_bytes*.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    response = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310

    # Check Content-Length
    content_length = response.headers.get("Content-Length")
    if content_length is not None and int(content_length) > max_bytes:
        raise ValueError(
            f"download_url: Content-Length {content_length} exceeds "
            f"max_bytes={max_bytes}."
        )

    # Infer extension
    ext = _infer_extension_from_headers(response.headers, url)
    filename = _make_temp_filename(url, ext)
    dest_path = output_path / filename

    # Stream download
    downloaded = 0
    with open(dest_path, "wb") as f:
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            downloaded += len(chunk)
            if downloaded > max_bytes:
                f.close()
                dest_path.unlink(missing_ok=True)
                raise ValueError(
                    f"download_url: download exceeded max_bytes="
                    f"{max_bytes} (downloaded {downloaded} bytes so far)."
                )
            f.write(chunk)

    logger.info(
        "download_url: downloaded %s → %s (%d bytes)",
        url,
        dest_path,
        downloaded,
    )
    return dest_path
