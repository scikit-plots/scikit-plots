# scikitplot/corpus/_url_handler.py
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

Design invariants
-----------------
* **Zero optional dependencies at import time.** ``urllib`` and ``http``
  are stdlib. ``requests`` is imported lazily in ``_download_with_requests``
  and falls back to ``urllib.request`` if unavailable.
* **SSRF prevention:** Before connecting, resolved IP addresses are checked
  against RFC-1918 / loopback / link-local ranges.  This blocks redirect-based
  SSRF attacks where an attacker's URL 302-redirects to ``http://169.254.169.254``.
* **Deterministic temp paths:** Downloaded files use a SHA-256 prefix of the URL
  as the filename stem so repeated downloads of the same URL hit the same path
  (enables naive caching when ``dest_dir`` is reused across builds).
* **Content-type fallback:** When the URL path has no recognisable extension,
  the ``Content-Type`` header is used to infer one.
* **Caller owns cleanup:** The caller is responsible for deleting the returned
  temp file or the ``dest_dir`` after processing.  ``CorpusBuilder`` uses a
  ``tempfile.TemporaryDirectory`` context manager for this.

Python compatibility
--------------------
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
        """Backport of ``enum.StrEnum`` for Python < 3.11."""


logger = logging.getLogger(__name__)

__all__ = [
    "URLKind",
    "classify_url",
    "download_url",
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
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "video/mp4": ".mp4",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
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

_YOUTUBE_RE: re.Pattern[str] = re.compile(
    r"https?://(www\.)?(youtube\.com/(watch|shorts|embed|live)|youtu\.be/)",
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

    Values
    ------
    WEB_PAGE
        General web page — route to ``WebReader``.
    YOUTUBE
        YouTube video — route to ``YouTubeReader``.
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

    1. YouTube (highest priority — YouTube URLs look like web pages
       but need special transcript handling).
    2. Google Drive share links.
    3. GitHub blob URLs (must check before raw, since blob URLs
       are on ``github.com``).
    4. GitHub raw URLs.
    5. Downloadable file (extension-based heuristic on the URL path).
    6. Web page (default fallback).

    Examples
    --------
    >>> classify_url("https://youtu.be/dQw4w9WgXcQ")
    <URLKind.YOUTUBE: 'youtube'>
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

    # 1. YouTube
    if _YOUTUBE_RE.match(url):
        return URLKind.YOUTUBE

    # 2. Google Drive
    if _GDRIVE_FILE_RE.match(url) or _GDRIVE_OPEN_RE.match(url):
        return URLKind.GOOGLE_DRIVE

    # 3. GitHub blob
    if _GITHUB_BLOB_RE.match(url):
        return URLKind.GITHUB_BLOB

    # 4. GitHub raw
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
    """
    Probe a URL with a HEAD request to classify by Content-Type.

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

    Developer note
    --------------
    The function tries ``requests`` first (better redirect + timeout
    handling).  It falls back to ``urllib.request`` if requests is not
    installed.  The SSRF check is applied before connecting when
    ``skip_ssrf_check=False``.
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
    """
    Send a HEAD (or fallback GET) request and return the Content-Type.

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

    Developer note
    --------------
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
    """
    Probe with the ``requests`` library.

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

    session = requests.Session()
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
    """
    Probe with ``urllib.request`` (stdlib fallback).

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
    """
    Map a MIME content-type string to a :class:`URLKind`.

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


def _infer_extension_from_headers(
    headers: Any,
    url: str,
) -> str:
    """
    Infer file extension from HTTP response headers or URL path.

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
    # Try URL path first
    parsed = urllib.parse.urlparse(url)
    path_lower = parsed.path.lower()
    for compound_ext in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if path_lower.endswith(compound_ext):
            return compound_ext
    _, ext = os.path.splitext(path_lower)
    if ext and ext in _DOWNLOADABLE_EXTENSIONS:
        return ext

    # Try Content-Type header
    ct = None
    if hasattr(headers, "get"):
        ct = headers.get("Content-Type", "")
    if ct:
        # "application/pdf; charset=utf-8" → "application/pdf"
        mime = ct.split(";")[0].strip().lower()
        if mime in _CONTENT_TYPE_TO_EXT:
            return _CONTENT_TYPE_TO_EXT[mime]
        # Try mimetypes stdlib
        guessed = mimetypes.guess_extension(mime)
        if guessed:
            return guessed

    # Try Content-Disposition header
    cd = None
    if hasattr(headers, "get"):
        cd = headers.get("Content-Disposition", "")
    if cd:
        # attachment; filename="report.pdf"
        match = re.search(r'filename[*]?="?([^";]+)"?', cd, re.IGNORECASE)
        if match:
            fname = match.group(1).strip()
            _, cd_ext = os.path.splitext(fname)
            if cd_ext:
                return cd_ext.lower()

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


def download_url(
    url: str,
    *,
    dest_dir: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    skip_ssrf_check: bool = False,
) -> Path:
    """
    Download a URL to a local file.

    Parameters
    ----------
    url : str
        URL to download. Must be ``http://`` or ``https://``.
    dest_dir : str, Path, or None, optional
        Directory to write the downloaded file into. If ``None``,
        ``tempfile.gettempdir()`` is used. Default: ``None``.
    max_bytes : int, optional
        Maximum download size in bytes. Default: 500 MB.
    timeout : int, optional
        HTTP timeout in seconds. Default: 120.
    max_redirects : int, optional
        Maximum number of HTTP redirects to follow. Default: 10.
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
        If the download fails due to a network error.
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

    Examples
    --------
    >>> path = download_url("https://example.com/report.pdf")
    >>> path.suffix
    '.pdf'
    >>> path.exists()
    True
    """
    if not isinstance(url, str) or not re.match(r"https?://", url, re.IGNORECASE):
        raise ValueError(
            f"download_url: url must start with 'http://' or 'https://'; got {url!r}."
        )

    # SSRF check
    if not skip_ssrf_check:
        _validate_url_security(url)

    # Resolve dest_dir
    if dest_dir is None:  # noqa: SIM108
        dest_dir = Path(tempfile.gettempdir())
    else:
        dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try requests first (better redirect handling, streaming), fall back
    # to urllib.
    try:
        return _download_with_requests(
            url,
            dest_dir=dest_dir,
            max_bytes=max_bytes,
            timeout=timeout,
            max_redirects=max_redirects,
            skip_ssrf_check=skip_ssrf_check,
        )
    except ImportError:
        logger.debug("requests library not available; falling back to urllib.")
        return _download_with_urllib(
            url,
            dest_dir=dest_dir,
            max_bytes=max_bytes,
            timeout=timeout,
        )


def _download_with_requests(
    url: str,
    *,
    dest_dir: Path,
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
    dest_dir : Path
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

    session = requests.Session()
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
    dest_path = dest_dir / filename

    # Stream download with size guard
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    # Clean up partial file
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
    dest_dir: Path,
    max_bytes: int,
    timeout: int,
) -> Path:
    """
    Download using ``urllib.request`` (stdlib fallback).

    Parameters
    ----------
    url : str
        URL to download.
    dest_dir : Path
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
    dest_path = dest_dir / filename

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
