# scikitplot/corpus/_readers/_web.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._readers._web
================================
Text extraction from web URLs and YouTube videos.

Two readers are provided:

:class:`WebReader`
    Fetches any ``http://`` / ``https://`` URL, parses the HTML with
    BeautifulSoup, and yields structured text sections (title, headings,
    paragraphs, list items). Respects ``robots.txt`` by default.

:class:`YouTubeReader`
    Extracts the transcript/caption track for a YouTube video using
    ``youtube-transcript-api``. Tries to retrieve a manually-created
    caption track first; falls back to auto-generated captions.
    No audio download or transcription is performed.

Registration
------------
Both readers are registered under special ``":"`` keys
(``":url"`` and ``":youtube"``) rather than file extensions.
They are accessed exclusively via
:meth:`~scikitplot.corpus._base.DocumentReader.from_url`.

Dependencies (all optional lazy imports)
-----------------------------------------
- ``requests`` — HTTP fetch for :class:`WebReader`
- ``beautifulsoup4`` — HTML parsing for :class:`WebReader`
- ``youtube-transcript-api`` — YouTube transcript for :class:`YouTubeReader`

None of these are imported at module level. ``ImportError`` fires only
at first ``get_raw_chunks()`` call.

Python compatibility
--------------------
Python 3.8-3.15. Zero stdlib dependencies beyond what is already used
in ``scikitplot.corpus._base``.
"""  # noqa: D205, D400

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Generator, List, Optional, Tuple  # noqa: F401

from .._base import DocumentReader
from .._schema import SectionType, SourceType

logger = logging.getLogger(__name__)

__all__ = [
    "WebReader",
    "YouTubeReader",
    # "validate_url_safety",
]

# with patch("scikitplot.corpus._readers._web._load_youtube_api",
#            side_effect=ImportError):
# def _load_youtube_api():
#     from youtube_transcript_api import (  # noqa: PLC0415
#         CouldNotRetrieveTranscript,
#         NoTranscriptFound,
#         TranscriptsDisabled,
#         VideoUnavailable,
#         YouTubeTranscriptApi,
#     )
#     return (
#         CouldNotRetrieveTranscript,
#         NoTranscriptFound,
#         TranscriptsDisabled,
#         VideoUnavailable,
#         YouTubeTranscriptApi,
#     )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HTML tag pattern for stripping tags from cue text (reuse from module level)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# HTML tags whose text content maps to section types
_HEADING_TAGS: tuple[str, ...] = ("h1", "h2", "h3", "h4", "h5", "h6")
_TITLE_TAGS: tuple[str, ...] = ("title",)
_BODY_TAGS: tuple[str, ...] = ("p", "li", "td", "blockquote", "pre", "code")

# Regex for YouTube URL / video ID extraction
# Matches every single-video YouTube URL form:
#   watch?v=…, shorts/ID, embed/ID, live/ID, youtu.be/ID
_YT_URL_RE = re.compile(
    r"https?://(www\.)?(youtube\.com/(watch|shorts/|embed/|live/)|youtu\.be/)",
    re.IGNORECASE,
)
# Extracts the 11-char video ID from any of the above URL forms.
_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})")

# Default HTTP request timeout in seconds
_DEFAULT_TIMEOUT: int = 30

# Default maximum response size (avoid streaming huge pages into memory)
_DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB


def _extract_youtube_id(url: str) -> str | None:
    """
    Extract the 11-character YouTube video ID from a URL.

    Parameters
    ----------
    url : str
        YouTube URL. Handles both ``youtube.com/watch?v=`` and
        ``youtu.be/`` forms.

    Returns
    -------
    str or None
        Video ID, or ``None`` if not found.

    Examples
    --------
    >>> # "https://youtu.be/rwPISgZcYIk",  # https://www.youtube.com/watch?v=rwPISgZcYIk
    >>> _extract_youtube_id("https://www.youtube.com/watch?v=rwPISgZcYIk")
    'rwPISgZcYIk'
    >>> _extract_youtube_id("https://youtu.be/rwPISgZcYIk")
    'rwPISgZcYIk'
    """
    m = _YT_ID_RE.search(url)
    return m.group(1) if m else None


def _section_type_for_tag(tag_name: str) -> str:
    """
    Map an HTML tag name to a :class:`~scikitplot.corpus._schema.SectionType` value.

    Parameters
    ----------
    tag_name : str
        Lowercase HTML tag name.

    Returns
    -------
    str
        ``SectionType`` string value.
    """
    if tag_name in _TITLE_TAGS or tag_name == "h1":
        return SectionType.TITLE.value
    if tag_name in _HEADING_TAGS:
        return SectionType.HEADER.value
    return SectionType.TEXT.value


# SSRF Protection may _ssrf_guard.py
def validate_url_safety(
    url: str,
    *,
    allow_private_networks: bool = False,
    max_content_bytes: int = 50_000_000,  # 50 MB
) -> None:
    """Validate URL against SSRF attacks.

    Parameters
    ----------
    url : str
        URL to validate.
    allow_private_networks : bool
        If False, reject private/loopback/link-local IP addresses.
    max_content_bytes : int
        Maximum content size in bytes.

    Raises
    ------
    ValueError
        If the URL resolves to a blocked IP address.

    Notes
    -----
    **Developer note:** This function resolves the hostname to an IP
    address and checks it against RFC 1918 (private), RFC 3927
    (link-local), and loopback ranges. Cloud metadata endpoints
    (169.254.169.254) are explicitly blocked.

    Call this BEFORE making any HTTP request in WebReader and
    YouTubeReader.
    """
    if allow_private_networks:
        return

    from urllib.parse import urlparse  # noqa: PLC0415

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Cannot extract hostname from URL: {url!r}")

    try:
        # Resolve hostname to IP
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
        for _family, _type, _proto, _canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)

            if ip.is_loopback:
                raise ValueError(
                    f"URL {url!r} resolves to loopback address {ip}. "
                    f"Set allow_private_networks=True to allow."
                )

            if ip.is_private:
                raise ValueError(
                    f"URL {url!r} resolves to private address {ip}. "
                    f"Set allow_private_networks=True to allow."
                )

            if ip.is_link_local:
                raise ValueError(
                    f"URL {url!r} resolves to link-local address {ip}. "
                    f"Set allow_private_networks=True to allow."
                )

            # Cloud metadata endpoint
            if ip_str == "169.254.169.254":
                raise ValueError(
                    f"URL {url!r} resolves to cloud metadata endpoint. "
                    f"This is blocked for security."
                )

    except socket.gaierror:
        # Cannot resolve — let the HTTP library handle it
        logger.debug("Cannot resolve hostname %r; skipping SSRF check.", hostname)


# ===========================================================================
# WebReader
# ===========================================================================


@dataclass
class WebReader(DocumentReader):
    """
    Fetch a web page and extract structured text via BeautifulSoup.

    Each HTML element (title, headings, paragraphs, list items) is
    yielded as a separate raw chunk with its section type and the source
    URL as metadata. JavaScript-rendered content is **not** supported
    (use Playwright or Selenium for that).

    Parameters
    ----------
    input_file : pathlib.Path
        Wrap the URL string as ``pathlib.Path(url)`` when constructing
        directly. Use :meth:`~scikitplot.corpus._base.DocumentReader.from_url`
        for the canonical construction path.
    source_uri : str or None, optional
        The original URL string. Set automatically by ``from_url()``.
        If ``None``, ``str(input_file)`` is used as the URL.
    timeout : int, optional
        HTTP request timeout in seconds. Default: 30.
    max_response_bytes : int, optional
        Maximum response body size. Responses larger than this raise
        ``ValueError``. Default: 10 MB.
    headers : dict or None, optional
        Extra HTTP headers to include in the request. A sensible
        ``User-Agent`` is added automatically when not supplied.
        Default: ``None``.
    extract_tags : list of str or None, optional
        HTML tags to extract text from. When ``None`` (default), uses
        the built-in set: ``["title", "h1"-"h6", "p", "li", "blockquote",
        "pre", "td"]``. Override to narrow or expand extraction.
    chunker : ChunkerBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filter_ : FilterBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    default_language : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    allow_private_networks : bool
        If False, reject private/loopback/link-local IP addresses.
    max_content_bytes : int
        Maximum content size in bytes.

    Attributes
    ----------
    file_type : str
        Class variable. Registry key ``":url"``.
    file_types : list of str
        Class variable. Registered extensions:
        ``[":url"]``.

    Raises
    ------
    ImportError
        If ``requests`` or ``beautifulsoup4`` is not installed.
    ValueError
        If the response exceeds ``max_response_bytes``.
    RuntimeError
        If the HTTP request returns a non-2xx status code.

    See Also
    --------
    scikitplot.corpus._readers.YouTubeReader : YouTube transcript extraction.

    Notes
    -----
    **robots.txt:** This reader does not enforce ``robots.txt``. Callers
    are responsible for checking ``/robots.txt`` before scraping at
    scale.

    **Rate limiting:** For bulk URL ingestion, add delays between calls
    or use a polite scraping library (``scrapy``, ``httpx`` with backoff).

    **JavaScript-rendered pages:** ``requests`` fetches only the initial
    HTML. SPAs (React, Vue, Angular) will yield little or no text. Use
    ``playwright`` or ``selenium`` to render and then pass the final HTML
    to BeautifulSoup manually.

    Examples
    --------
    Via factory (recommended):

    >>> from scikitplot.corpus._base import DocumentReader
    >>> import scikitplot.corpus._readers
    >>> reader = DocumentReader.from_url("https://en.wikipedia.org/wiki/Python")
    >>> docs = list(reader.get_documents())
    >>> print(f"Extracted {len(docs)} text sections")

    Direct construction:

    >>> from pathlib import Path
    >>> url = "https://en.wikipedia.org/wiki/Python"
    >>> reader = WebReader(input_file=Path(url), source_uri=url)
    """

    file_type: ClassVar[str] = ":url"
    file_types: ClassVar[list[str] | None] = [":url"]

    timeout: int = field(default=_DEFAULT_TIMEOUT)
    """HTTP request timeout in seconds."""

    max_response_bytes: int = field(default=_DEFAULT_MAX_BYTES)
    """Maximum response body size. Default: 10 MB."""

    headers: dict[str, str] | None = field(default=None)
    """Extra HTTP request headers."""

    extract_tags: list[str] | None = field(default=None)
    """HTML tags to extract. ``None`` uses the built-in defaults."""

    allow_private_networks: bool = field(default=False)
    """If False, reject private/loopback/link-local IP addresses."""

    max_content_bytes: int = field(default=50_000_000)  # 50 MB
    """Maximum content size in bytes."""

    def __post_init__(self) -> None:
        """Initialise the WebReader and configure HTTP session settings.

        Notes
        -----
        URL-based readers do not call ``super().__post_init__()`` because
        ``validate_input`` for URLs performs a network check, not a
        filesystem existence check.  Validation is deferred to the first
        ``get_raw_chunks`` call.

        Raises
        ------
        ValueError
            If ``timeout <= 0`` or ``max_response_bytes <= 0``.
        """
        # URL readers do NOT call super().__post_init__ for file validation —
        # we handle validation ourselves in validate_input().
        from .._base import DefaultFilter  # noqa: PLC0415

        if self.filter_ is None:
            object.__setattr__(self, "filter_", DefaultFilter())

    def validate_input(self) -> None:
        """
        Validate the URL format instead of checking for a local file.

        Raises
        ------
        ValueError
            If the URL does not start with ``http://`` or ``https://``.
        """
        url = self._effective_url()
        if not re.match(r"https?://", url):
            raise ValueError(
                f"WebReader: source_uri must start with 'http://' or"
                f" 'https://'; got {url!r}."
            )

    @property
    def file_name(self) -> str:
        """Return the URL as the effective file name."""
        return self.filename_override or self._effective_url()

    def _effective_url(self) -> str:
        """Return the URL to fetch: ``source_uri`` or ``str(input_file)``.

        Returns
        -------
        str
            The URL used for the HTTP request.
        """
        return self.source_uri or str(self.input_file)

    def _build_headers(self) -> dict[str, str]:
        """Build the request headers dict with a default User-Agent."""
        defaults = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; scikitplot-corpus/1.0;"
                " +https://github.com/scikit-plot/scikit-plot)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        if self.headers:
            defaults.update(self.headers)
        return defaults

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Fetch the URL and yield one chunk per HTML text element.

        Yields
        ------
        dict
            Keys:

            ``"text"``
                Extracted text for this element.
            ``"section_type"``
                :attr:`SectionType.TITLE`, :attr:`SectionType.HEADER`,
                or :attr:`SectionType.TEXT`.
            ``"source_type"``
                Always :attr:`SourceType.WEB`; promoted to
                :attr:`CorpusDocument.source_type`.
            ``"html_tag"``
                Original HTML tag name (e.g. ``"p"``, ``"h2"``).
            ``"url"``
                Source URL; promoted to :attr:`CorpusDocument.url`.
            ``"element_index"``
                Zero-based position of this element in the extraction order.

        Raises
        ------
        ImportError
            If ``requests`` or ``beautifulsoup4`` is not installed.
        ValueError
            If the response body exceeds ``max_response_bytes``.
        RuntimeError
            If the HTTP response status is not 2xx.
        """
        url = self._effective_url()
        self.validate_input()

        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "requests is required for WebReader."
                " Install it with:\n"
                "  pip install requests"
            ) from exc

        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 is required for WebReader."
                " Install it with:\n"
                "  pip install beautifulsoup4"
            ) from exc

        logger.info("WebReader: fetching %s.", url)
        validate_url_safety(
            self.source_uri or str(self.input_file),
            allow_private_networks=self.allow_private_networks,
        )
        response = requests.get(
            url,
            headers=self._build_headers(),
            timeout=self.timeout,
            stream=True,
        )
        content_length = int(response.headers.get("Content-Length", 0))
        if content_length > self.max_content_bytes:
            raise ValueError(
                f"Content-Length {content_length} exceeds limit "
                f"{self.max_content_bytes}."
            )

        if not response.ok:
            raise RuntimeError(
                f"WebReader: HTTP {response.status_code} fetching {url}."
            )

        # Stream-read up to max_response_bytes
        chunks_raw = []
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=65536):
            total_bytes += len(chunk)
            if total_bytes > self.max_response_bytes:
                logger.warning(
                    "WebReader: response for %s exceeds max_response_bytes=%d;"
                    " truncating.",
                    url,
                    self.max_response_bytes,
                )
                break
            chunks_raw.append(chunk)

        html = b"".join(chunks_raw).decode(
            response.encoding or "utf-8", errors="replace"
        )

        logger.info("WebReader: fetched %s (%d bytes).", url, total_bytes)

        tags = self.extract_tags or list(_TITLE_TAGS + _HEADING_TAGS + _BODY_TAGS)
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style elements before extraction
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        element_index = 0
        for element in soup.find_all(tags):
            text = element.get_text(separator=" ", strip=True)
            if not text.strip():
                continue

            sec_type = _section_type_for_tag(element.name)

            yield {
                "text": text,
                "section_type": sec_type,
                # promoted → CorpusDocument.source_type
                "source_type": SourceType.WEB.value,
                "html_tag": element.name,
                "url": url,
                "element_index": element_index,
            }
            element_index += 1

        logger.info(
            "WebReader: extracted %d elements from %s.",
            element_index,
            url,
        )


# ===========================================================================
# YouTubeReader
# ===========================================================================


@dataclass
class YouTubeReader(DocumentReader):
    """
    Extract the transcript of a YouTube video using ``youtube-transcript-api``.

    Tries to retrieve manually-created captions first (preferred — often
    higher quality), then falls back to auto-generated captions.  No audio
    is downloaded; no local model is required.

    Parameters
    ----------
    input_file : pathlib.Path
        Wrap the YouTube URL as ``pathlib.Path(url)`` when constructing
        directly. Use :meth:`~scikitplot.corpus._base.DocumentReader.from_url`
        for the canonical construction path.
    source_uri : str or None, optional
        The original YouTube URL string. Set automatically by ``from_url()``.
    preferred_language : str or None, optional
        ISO 639-1 language code for the preferred transcript (e.g. ``"en"``,
        ``"de"``). When ``None``, the API returns the first available track.
        Default: ``None``.
    include_auto_generated : bool, optional
        When ``True`` (default), accept auto-generated captions as a fallback
        if no manual caption track is found.
    merge_short_cues : bool, optional
        When ``True``, consecutive cues shorter than ``min_cue_chars`` are
        merged with the following cue.  This prevents very short subtitle
        segments (e.g. ``"[Music]"``) from becoming separate documents.
        Default: ``True``.
    min_cue_chars : int, optional
        Minimum character count per cue when ``merge_short_cues=True``.
        Default: 20.
    chunker : ChunkerBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filter_ : FilterBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    default_language : str or None, optional
        ISO 639-1 language code for output documents. If ``None``, the
        transcript language detected by the API is used. Default: ``None``.

    Attributes
    ----------
    file_type : str
        Class variable. Registry key ``":youtube"``.
    file_types : list of str
        Class variable. Registered extensions:
        ``[":youtube"]``.

    Raises
    ------
    ImportError
        If ``youtube-transcript-api`` is not installed.
    ValueError
        If the video ID cannot be extracted from the URL.
    RuntimeError
        If no transcript is available for the video (private, disabled, etc.).

    See Also
    --------
    scikitplot.corpus._readers.WebReader : General web page reader.
    scikitplot.corpus._readers.VideoReader : Local video file reader.

    Notes
    -----
    **Privacy:** ``youtube-transcript-api`` fetches transcripts by making
    HTTP requests to YouTube's timedtext API endpoint. It does not log in
    and does not require a Google API key for public videos.  Private videos
    and videos with disabled captions will raise an error.

    **Quota:** YouTube's timedtext endpoint is rate-limited. For bulk
    ingestion, add delays between calls.

    Examples
    --------
    Via factory (recommended):

    >>> from scikitplot.corpus._base import DocumentReader
    >>> import scikitplot.corpus._readers
    >>> reader = DocumentReader.from_url(
    ...     "https://www.youtube.com/watch?v=rwPISgZcYIk",
    ...     default_language="en",
    ... )
    >>> docs = list(reader.get_documents())
    >>> print(f"Transcript cues: {len(docs)}")
    """

    file_type: ClassVar[str] = ":youtube"
    file_types: ClassVar[list[str] | None] = [":youtube"]

    preferred_language: str | None = field(default=None)
    """ISO 639-1 language code for the preferred transcript track."""

    include_auto_generated: bool = field(default=True)
    """Accept auto-generated captions as fallback."""

    merge_short_cues: bool = field(default=True)
    """Merge consecutive short cues into longer chunks."""

    min_cue_chars: int = field(default=20)
    """Minimum cue length for ``merge_short_cues``."""

    def __post_init__(self) -> None:
        """Initialise the YouTubeReader and parse the video ID from the URL.

        Raises
        ------
        ValueError
            If the URL does not match any recognised YouTube pattern
            (watch, shorts, embed, live, youtu.be).
        """
        from .._base import DefaultFilter  # noqa: PLC0415

        if self.filter_ is None:
            object.__setattr__(self, "filter_", DefaultFilter())

    def validate_input(self) -> None:
        """
        Validate that the source URI is a recognisable YouTube URL.

        Raises
        ------
        ValueError
            If the URL is not a YouTube URL or the video ID cannot be
            extracted.
        """
        url = self._effective_url()
        if not _YT_URL_RE.match(url):
            raise ValueError(
                f"YouTubeReader: URL does not look like a YouTube link: {url!r}. "
                f"Supported forms: "
                f"https://www.youtube.com/watch?v=ID, "
                f"https://www.youtube.com/shorts/ID, "
                f"https://www.youtube.com/embed/ID, "
                f"https://www.youtube.com/live/ID, "
                f"https://youtu.be/ID"
            )
        if _extract_youtube_id(url) is None:
            raise ValueError(f"YouTubeReader: could not extract video ID from {url!r}.")

    @property
    def file_name(self) -> str:
        """Return the URL as the effective file name."""
        return self.filename_override or self._effective_url()

    def _effective_url(self) -> str:
        """Return the YouTube URL for this reader.

        Returns
        -------
        str
            ``source_uri`` when set, otherwise ``str(input_file)``.
        """
        return self.source_uri or str(self.input_file)

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Fetch the YouTube transcript and yield one chunk per cue.

        Yields
        ------
        dict
            Keys:

            ``"text"``
                Caption cue text.
            ``"section_type"``
                Always :attr:`SectionType.TEXT`.
            ``"timecode_start"``
                Cue start time in seconds (float); promoted to
                :attr:`CorpusDocument.timecode_start`.
            ``"timecode_end"``
                Cue end time in seconds (= start + duration); promoted to
                :attr:`CorpusDocument.timecode_end`.
            ``"source_type"``
                Always :attr:`SourceType.VIDEO`; promoted to
                :attr:`CorpusDocument.source_type`.
            ``"transcript_type"``
                ``"manual"`` or ``"auto_generated"``; non-promoted, goes
                to ``metadata``.
            ``"video_id"``
                YouTube video ID string.
            ``"transcript_language"``
                Language code of the retrieved transcript.

        Raises
        ------
        ImportError
            If ``youtube-transcript-api`` is not installed.
        ValueError
            If the video ID cannot be extracted from the URL.
        RuntimeError
            If no transcript is available.
        """
        self.validate_input()

        try:
            # pip install youtube-transcript-api
            from youtube_transcript_api import (  # noqa: PLC0415
                CouldNotRetrieveTranscript,
                NoTranscriptFound,  # noqa: F401
                TranscriptsDisabled,
                VideoUnavailable,
                YouTubeTranscriptApi,
            )
        except ImportError as exc:
            raise ImportError(
                "youtube-transcript-api is required for YouTubeReader."
                " Install it with:\n"
                "  pip install youtube-transcript-api"
            ) from exc

        url = self._effective_url()
        video_id = _extract_youtube_id(url)
        if video_id is None:
            raise ValueError(f"YouTubeReader: could not extract video ID from {url!r}.")

        logger.info("YouTubeReader: fetching transcript for video %s.", video_id)

        # Retrieve available transcripts
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
        except (
            TranscriptsDisabled,
            VideoUnavailable,
            CouldNotRetrieveTranscript,
        ) as exc:
            raise RuntimeError(
                f"YouTubeReader: could not retrieve transcripts for video"
                f" {video_id!r}: {exc}"
            ) from exc

        # Select transcript: manual preferred, auto as fallback
        transcript = None
        transcript_type = (
            "manual"  # sub-type detail → metadata, not promoted source_type
        )
        lang_codes = [self.preferred_language] if self.preferred_language else []

        try:
            # or just filter for manually created transcripts
            transcript = transcript_list.find_manually_created_transcript(
                lang_codes or ["en", "en-US", "en-GB"]
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "YouTubeReader: manual transcript not found for %s;"
                " trying auto-generated.",
                video_id,
            )
            if self.include_auto_generated:
                try:
                    transcript = transcript_list.find_generated_transcript(
                        lang_codes or ["en", "en-US", "en-GB"]
                    )
                    transcript_type = "auto_generated"
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "YouTubeReader: auto-generated transcript not found for %s.",
                        video_id,
                    )

        if transcript is None:
            # Final fallback: take whatever is available
            all_transcripts = list(transcript_list)
            if not all_transcripts:
                raise RuntimeError(
                    f"YouTubeReader: no transcripts available for video {video_id!r}."
                )
            transcript = all_transcripts[0]
            transcript_type = "auto_generated" if transcript.is_generated else "manual"

        transcript_language = transcript.language_code
        logger.info(
            "YouTubeReader: using %s %s transcript for video %s.",
            transcript_type,
            transcript_language,
            video_id,
        )

        # Fetch the actual cues
        try:
            cues = transcript.fetch()
        except Exception as exc:
            raise RuntimeError(
                f"YouTubeReader: failed to fetch transcript for video"
                f" {video_id!r}: {exc}"
            ) from exc

        logger.info(
            "YouTubeReader: retrieved %d cues for video %s.", len(cues), video_id
        )

        # Optionally merge short cues
        if self.merge_short_cues:
            cues = self._merge_short_cues(cues)

        # Determine language for documents
        lang = self.default_language or transcript_language

        for cue in cues:
            # youtube-transcript-api returns dict or snippet objects
            text = cue.get("text", "") if isinstance(cue, dict) else str(cue.text)
            text = _HTML_TAG_RE.sub("", text).strip()
            if not text:
                continue

            start = float(cue.get("start", 0.0) if isinstance(cue, dict) else cue.start)
            duration = float(
                cue.get("duration", 0.0)
                if isinstance(cue, dict)
                else getattr(cue, "duration", 0.0)
            )

            yield {
                "text": text,
                "section_type": SectionType.TEXT.value,
                "timecode_start": round(
                    start, 3
                ),  # promoted → CorpusDocument.timecode_start
                "timecode_end": round(
                    start + duration, 3
                ),  # promoted → CorpusDocument.timecode_end
                # promoted → CorpusDocument.source_type
                "source_type": SourceType.VIDEO.value,
                # non-promoted → metadata ("manual"/"auto_generated")
                "transcript_type": transcript_type,
                "video_id": video_id,
                "transcript_language": transcript_language,
            }

    def _merge_short_cues(self, cues: list[Any]) -> list[dict[str, Any]]:
        """
        Merge consecutive cues shorter than ``min_cue_chars`` into the
        following cue.

        Parameters
        ----------
        cues : list
            Raw cue objects from youtube-transcript-api.

        Returns
        -------
        list of dict
            Normalised cue dicts after merging.
        """  # noqa: D205

        def _normalise(c: Any) -> dict[str, Any]:
            """Normalise a raw content block to ``{"text": str}``.

            Parameters
            ----------
            c : dict or any
                Raw content block from BeautifulSoup extraction.

            Returns
            -------
            dict[str, Any]
                Dict with at least ``"text"`` key.
            """
            if isinstance(c, dict):
                return c
            return {
                "text": str(getattr(c, "text", "")),
                "start": float(getattr(c, "start", 0.0)),
                "duration": float(getattr(c, "duration", 0.0)),
            }

        normalised = [_normalise(c) for c in cues]
        merged: list[dict[str, Any]] = []
        pending: dict[str, Any] | None = None

        for cue in normalised:
            text = cue.get("text", "").strip()
            if pending is None:
                pending = dict(cue)
            else:  # noqa: PLR5501
                if len(pending.get("text", "")) < self.min_cue_chars:
                    # Merge current into pending
                    dur = pending.get("duration", 0.0) + cue.get("duration", 0.0)
                    pending["text"] = pending["text"].rstrip() + " " + text
                    pending["duration"] = dur
                else:
                    merged.append(pending)
                    pending = dict(cue)

        if pending is not None:
            merged.append(pending)

        return merged
