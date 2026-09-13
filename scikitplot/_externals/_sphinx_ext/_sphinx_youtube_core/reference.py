"""
Total, future-proof parser for the YouTube URL ecosystem.

Turns *any* string a person can plausibly paste into one explicit, typed
:class:`YouTubeReference`: bare ids, watch URLs with a dozen tracking
parameters, ``youtu.be`` links, Shorts, Clips, live streams, playlist
embeds, channel handles, channel tabs, community posts, search result
pages, hashtag feeds, and the consent / redirect / attribution wrappers
YouTube itself puts around all of the above.

Why a typed reference rather than "extract the id"
--------------------------------------------------
The URLs a reader copies out of the address bar do not all denote the same
*kind* of thing, and several denote two or three things at once::

    https://www.youtube.com/watch?v=nfYOp3_SyqM&list=PLXOJEg4xbr50&index=2

That is simultaneously a video, a position, and a playlist. A function that
returns only a string has to pick one meaning and silently discard the rest,
which is how a channel URL ends up rendered as a dead ``<iframe>``.
Returning a structure instead lets each consumer decide without losing
context. The ``youtube`` directive, browser Add-video flow, and sync tool all
treat a ``watch?v=…&list=…`` reference as the exact ``video_id`` because the
URL names one concrete playable item; ``playlist_id`` remains available as
context. Collection ingestion is explicit through a playlist URL/id rather
than being inferred from incidental watch-page context.

Design for a moving target
--------------------------
YouTube adds URL shapes without notice -- ``/shorts/`` in 2021, the
``podcasts`` and ``courses`` channel tabs in 2023, the ``pp`` and ``si``
parameters after that. Three rules keep this parser from becoming a
liability when the next one lands:

1. **Recognition is layered, and "unknown" is a value.** An unrecognised
   channel tab yields ``tab_known=False``, not an exception. A playlist id
   with an unfamiliar prefix yields ``playlist_kind="unknown"``, not a
   rejection. The parser reports what it sees; *consumers* decide whether
   they can act on it. A new tab therefore degrades to a clear "I cannot
   enumerate that yet" from the sync tool, instead of a parse failure that
   breaks every page mentioning it.
2. **Structure, never regex-over-the-whole-URL.** Scanning a full URL with
   ``v=([A-Za-z0-9_-]{11})`` silently truncates a malformed 16-character
   ``v`` to a well-formed id *for a different video* -- a wrong answer no
   downstream check can catch. Splitting the URL and then validating each
   component with an anchored pattern makes that class of failure
   unrepresentable.
3. **One table, not a chain of conditionals.** Path handling is a registry
   (:data:`_PATH_HANDLERS`). Supporting a new URL shape is one entry.

Notes
-----
**User-focused.** Paste the URL. Scheme optional, ``www`` optional, any
YouTube host, wrapped or unwrapped.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import parse_qsl, unquote, urlsplit

__all__ = [
    "CHANNEL",
    "CLIP",
    "FEED",
    "HASHTAG",
    "PLAYLIST",
    "POST",
    "SEARCH",
    "VIDEO",
    "ReferenceError",
    "YouTubeReference",
    "is_reference_url",
    "parse_reference",
    "parse_video_reference",
    "validate_channel_id",
    "validate_handle",
    "validate_playlist_id",
]


class ReferenceError(ValueError):
    """
    Raised when a string cannot be parsed as a YouTube reference.

    Messages always name the input and the accepted forms, so they can be
    surfaced verbatim to a page author.
    """


# -- reference kinds ----------------------------------------------------------
#: Names exactly one video.
VIDEO = "video"
#: Names an ordered collection of videos.
PLAYLIST = "playlist"
#: Names a channel, optionally one of its tabs.
CHANNEL = "channel"
#: Names a user-created excerpt of a video. A Clip URL does **not** contain
#: the underlying video id; resolving one requires an API call.
CLIP = "clip"
#: Names a community post.
POST = "post"
#: Names a search results page.
SEARCH = "search"
#: Names a hashtag feed.
HASHTAG = "hashtag"
#: Names a personal or algorithmic feed (subscriptions, trending, history).
FEED = "feed"


# -- host recognition ---------------------------------------------------------
#: Registrable domains treated as YouTube. Matching is by *suffix*, so every
#: present and future subdomain (``m.``, ``music.``, ``studio.``, ``tv.``,
#: ``consent.``, and whatever comes next) is covered with no list to keep up
#: to date.
_HOST_SUFFIXES = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "youtubekids.com",
    "youtubeeducation.com",
)

#: YouTube operates country-code domains (``youtube.de``, ``youtube.co.uk``)
#: that redirect to ``.com``. Accepting them costs nothing and spares a
#: reader an inexplicable rejection.
_CCTLD_HOST_RE = re.compile(r"(?:^|\.)youtube\.(?:[a-z]{2}|co\.[a-z]{2})$")


# -- identifier shapes --------------------------------------------------------
#: Canonical video id: exactly 11 URL-safe base64 characters.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

#: Canonical channel id: ``UC`` followed by 22 URL-safe base64 characters.
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

#: Any URL-safe base64 run, used to validate opaque ids by alphabet.
_OPAQUE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

#: Known playlist id prefixes and what they mean. Used for *classification*,
#: not for admission: an unrecognised prefix is still accepted (see
#: :func:`_checked_playlist_id`) so a new YouTube playlist family does not
#: break the parser on the day it ships. Longer prefixes come first so the
#: more specific match wins.
_PLAYLIST_PREFIXES = (
    ("OLAK5uy_", "album"),
    ("RDCLAK", "mix"),
    ("RDMM", "mix"),
    ("PL", "user"),
    ("UU", "uploads"),
    ("LL", "liked"),
    ("FL", "favourites"),
    ("WL", "watch-later"),
    ("SP", "legacy"),
    ("RD", "mix"),
    ("UL", "mix"),
)

#: Playlist kinds YouTube generates per viewer and per session. They are not
#: stable content and must never be committed to a catalog.
_EPHEMERAL_PLAYLIST_KINDS = frozenset({"mix", "watch-later", "liked"})

#: Shortest string accepted as a playlist id. Every real family is far
#: longer; the floor exists to reject an obviously truncated paste such as
#: ``list=PL``, which would otherwise pass the prefix check and be committed
#: to a catalog as a playlist that does not exist.
_MIN_PLAYLIST_ID_LEN = 10

#: The two playlist ids that really are two characters long. Both are
#: per-account and therefore ephemeral, but they must parse rather than be
#: mistaken for a truncated id.
_SHORT_PLAYLIST_IDS = frozenset({"WL", "LL"})


# -- channel tabs -------------------------------------------------------------
#: Channel tabs known at the time of writing. Membership only sets
#: ``tab_known``; an unknown tab parses fine.
_KNOWN_TABS = frozenset(
    {
        "videos",
        "shorts",
        "streams",
        "live",
        "playlists",
        "podcasts",
        "courses",
        "releases",
        "community",
        "posts",
        "about",
        "featured",
        "channels",
        "store",
        "search",
        "membership",
    }
)

#: Path segments introducing a channel, mapped to the kind of name following.
_CHANNEL_PREFIXES = {"channel": "id", "c": "custom", "user": "legacy"}

#: Non-channel top-level segments, so a bare ``/NAME`` vanity URL is never
#: mistaken for one of YouTube's own pages.
_RESERVED_SEGMENTS = frozenset(
    {
        "watch",
        "watch_videos",
        "watch_popup",
        "playlist",
        "embed",
        "shorts",
        "live",
        "clip",
        "post",
        "results",
        "hashtag",
        "feed",
        "channel",
        "c",
        "user",
        "v",
        "e",
        "oembed",
        "redirect",
        "attribution_link",
        "premium",
        "about",
        "account",
        "upload",
        "gaming",
        "music",
        "movies",
        "sports",
        "reporthistory",
        "t",
        "howyoutubeworks",
        "creators",
        "new",
        "playables",
        "source",
        "profile",
        "signin",
        "logout",
        "shopping",
    }
)


# -- wrappers -----------------------------------------------------------------
#: Query parameters carrying a wrapped destination URL, in priority order.
#: These are how YouTube's consent gate, outbound redirector and attribution
#: links encapsulate a real URL.
_WRAPPER_PARAMS = ("continue", "next", "u", "q", "url", "target_url", "link")

#: Maximum wrapper layers to unwrap. A bound is required: without it a URL
#: that wraps itself is an infinite loop.
_MAX_UNWRAP = 5

#: Query parameters carrying a start offset, in the order they are honoured.
_START_PARAMS = ("t", "start", "time_continue")

#: Whether a cleaned string already begins with a URL or bare identifier,
#: in which case no extraction from surrounding prose is needed.
_URL_START_RE = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*:|//|[?\w@-]+[=/]|[\w@-]+$)", re.IGNORECASE
)

#: A URL-shaped run embedded in surrounding text. Stops at whitespace and at
#: the bracket characters that delimit Markdown and prose citations.
_EMBEDDED_URL_RE = re.compile(
    r"(?:https?://|//)?[\w.-]*(?:youtube\.[a-z.]+|youtu\.be|youtubekids\.com)"
    r"/[^\s<>()\[\]\"\'\u2018\u2019\u201c\u201d]*",
    re.IGNORECASE,
)


def _classify_playlist(value: str) -> str:
    """
    Classify a playlist id by its prefix.

    Parameters
    ----------
    value : str
        A playlist id.

    Returns
    -------
    str
        A kind from :data:`_PLAYLIST_PREFIXES`, or ``"unknown"`` for a
        prefix this parser has not seen. ``"unknown"`` is a normal outcome,
        not an error: it is what a newly introduced playlist family looks
        like before this table is updated.

    Examples
    --------
    >>> _classify_playlist("PLXOJEg4xbr50")
    'user'
    >>> _classify_playlist("RDdQw4w9WgXcQ")
    'mix'
    >>> _classify_playlist("ZZsomethingnew")
    'unknown'
    """
    for prefix, kind in _PLAYLIST_PREFIXES:
        if value.startswith(prefix):
            return kind
    return "unknown"


def _parse_start(value: str) -> int | None:
    """
    Parse a start-offset parameter into whole seconds.

    Parameters
    ----------
    value : str
        Bare seconds (``"90"``) or a compound offset (``"1h2m30s"``).

    Returns
    -------
    int or None
        Offset in seconds, or ``None`` when unparsable. An unparsable
        offset is dropped rather than raised on: it is decoration on a URL
        that is otherwise perfectly valid, and losing a timestamp is not
        worth failing a build over.
    """
    text = value.strip().lower()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(group or 0) for group in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _checked_video_id(value: str, raw: str) -> str:
    """
    Validate a candidate video id.

    Parameters
    ----------
    value : str
        Candidate id, already isolated from its surrounding URL.
    raw : str
        The original input, for the error message.

    Returns
    -------
    str
        The validated id.

    Raises
    ------
    ReferenceError
        If the candidate is not exactly 11 URL-safe base64 characters. The
        exact-length check is the whole point: accepting a prefix of a
        malformed 16-character value would yield a syntactically valid id
        for the wrong video, which no downstream check could catch.
    """
    if _VIDEO_ID_RE.match(value):
        return value
    raise ReferenceError(
        f"{raw!r}: {value!r} is not a valid YouTube video id (expected "
        f"exactly 11 characters from A-Z a-z 0-9 _ -, got {len(value)})"
    )


def validate_handle(value: Any) -> str:
    """
    Validate the structural form of a YouTube channel handle.

    The public YouTube policy supports letters/numbers from many scripts plus
    ``_``, ``-``, ``.`` and the Latin middle dot. Script-mixing, availability
    and language-specific minimum lengths are provider policies and therefore
    intentionally not duplicated here. This validator only rejects shapes that
    cannot be safely represented as a ``youtube.com/@handle`` path.
    """
    text = unicodedata.normalize("NFC", str(value or "").strip())
    if text.startswith("@"):
        text = text[1:]
    if not text:
        raise ReferenceError(f"{value!r} is not a valid YouTube handle (empty)")
    # YouTube's longest documented handle family is 30 characters. The
    # language-specific minimum varies, so do not impose an ASCII-centric min.
    if len(text) > 30:  # ruff: ignore[magic-value-comparison]
        raise ReferenceError(
            f"{value!r} is not a valid YouTube handle (more than 30 characters)"
        )
    separators = "_-.·"
    if text[0] in separators or text[-1] in separators:
        raise ReferenceError(
            f"{value!r} is not a valid YouTube handle "
            "(separator cannot be first or last)"
        )
    for char in text:
        category = unicodedata.category(char)
        if char in separators or category[:1] in {"L", "N", "M"}:
            continue
        raise ReferenceError(
            f"{value!r} is not a valid YouTube handle (unsupported character {char!r})"
        )
    return text


def validate_channel_id(value: Any) -> str:
    """Validate and return a canonical ``UC…`` channel id."""
    text = str(value or "").strip()
    if not _CHANNEL_ID_RE.fullmatch(text):
        raise ReferenceError(
            f"{value!r} is not a valid YouTube channel id "
            "(expected 'UC' followed by 22 URL-safe characters)"
        )
    return text


def validate_playlist_id(value: Any) -> str:
    """Validate a playlist id by shape without freezing known prefixes."""
    text = str(value or "").strip()
    return _checked_playlist_id(text, text)


def _checked_playlist_id(value: str, raw: str) -> str:
    """
    Validate a candidate playlist id by alphabet and length.

    Parameters
    ----------
    value : str
        Candidate playlist id.
    raw : str
        The original input, for the error message.

    Returns
    -------
    str
        The validated id.

    Raises
    ------
    ReferenceError
        If the candidate uses characters outside the URL-safe base64
        alphabet, or is too short to be any real playlist id.

    Notes
    -----
    Deliberately validated by *shape* rather than against the prefix table.
    Admission by prefix list would reject every playlist family YouTube
    introduces after this file was written -- the exact failure mode that
    made ``OLAK5uy_`` album playlists unusable in older tooling.
    """
    if not _OPAQUE_RE.match(value):
        raise ReferenceError(
            f"{raw!r}: {value!r} is not a valid playlist id (characters "
            f"outside A-Z a-z 0-9 _ -)"
        )
    if value in _SHORT_PLAYLIST_IDS:
        return value
    if len(value) < _MIN_PLAYLIST_ID_LEN:
        raise ReferenceError(
            f"{raw!r}: {value!r} is too short to be a playlist id (got "
            f"{len(value)} characters); the link may be truncated"
        )
    return value


@dataclass(frozen=True)
class YouTubeReference:
    """
    A parsed, typed YouTube reference.

    Attributes
    ----------
    kind : str
        What the URL primarily names: :data:`VIDEO`, :data:`PLAYLIST`,
        :data:`CHANNEL`, :data:`CLIP`, :data:`POST`, :data:`SEARCH`,
        :data:`HASHTAG` or :data:`FEED`. Secondary identifiers are still
        recorded, so nothing in the URL is discarded.
    video_id : str
        Canonical 11-character video id, or ``""``.
    playlist_id : str
        Canonical playlist id, or ``""``. Present on a ``watch?v=…&list=…``
        URL as well as on a bare playlist URL.
    playlist_kind : str
        Classification from :func:`_classify_playlist`, or ``""``.
    channel_id : str
        Canonical ``UC…`` channel id, or ``""``.
    handle : str
        Channel handle without its ``@``, or ``""``.
    channel_name : str
        A legacy ``/user/NAME``, custom ``/c/NAME`` or bare ``/NAME``
        channel name, or ``""``.
    tab : str
        Channel tab segment, or ``""`` for the channel root.
    tab_known : bool
        Whether ``tab`` was in :data:`_KNOWN_TABS` at parse time. ``False``
        means YouTube has a tab this build does not know about -- which is
        information a consumer can act on, not a parse failure.
    clip_id, post_id : str
        Opaque Clip / community-post identifiers, or ``""``.
    search_query : str
        Decoded search terms, or ``""``.
    hashtag : str
        Hashtag without its ``#``, or ``""``.
    feed : str
        Feed name such as ``subscriptions`` or ``trending``, or ``""``.
    index : int or None
        One-based position within ``playlist_id``.
    start : int or None
        Start offset in whole seconds.
    ephemeral : bool
        ``True`` when the reference names viewer-specific, session-scoped
        content (a mix, Watch Later, Liked videos, or a personal feed).
    host : str
        The normalised host the reference was parsed from.
    unwrapped_from : str
        The wrapper URL this reference was extracted from, or ``""``.
    raw : str
        The original input, verbatim, for error messages.
    params : dict
        All decoded query parameters, retained so a consumer can read a
        parameter this parser does not model yet.
    """

    kind: str
    video_id: str = ""
    playlist_id: str = ""
    playlist_kind: str = ""
    channel_id: str = ""
    handle: str = ""
    channel_name: str = ""
    tab: str = ""
    tab_known: bool = True
    clip_id: str = ""
    post_id: str = ""
    search_query: str = ""
    hashtag: str = ""
    feed: str = ""
    index: int | None = None
    start: int | None = None
    ephemeral: bool = False
    host: str = ""
    unwrapped_from: str = ""
    raw: str = ""
    params: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def is_collection(self) -> bool:
        """
        Whether this reference names many videos rather than one.

        Returns
        -------
        bool
            ``True`` for playlist, channel, hashtag, search and feed
            references.
        """
        return self.kind in (PLAYLIST, CHANNEL, HASHTAG, SEARCH, FEED)

    @property
    def channel_ref(self) -> str:
        """
        The most stable available channel identifier.

        Returns
        -------
        str
            The ``UC…`` id if known, else the handle, else the vanity name,
            else ``""``. Ordered by durability: ids never change, handles
            rarely do, vanity names are the least reliable.
        """
        return self.channel_id or self.handle or self.channel_name

    @property
    def watch_url(self) -> str:
        """
        Canonical, parameter-free watch URL for a video reference.

        Returns
        -------
        str
            ``https://www.youtube.com/watch?v=<id>``, or ``""`` when this
            reference names no single video. Playlist and start-time
            parameters are dropped: a catalog entry identifies a video, and
            the context it was watched in is not part of that identity.
        """
        if not self.video_id:
            return ""
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def describe(  # ruff: ignore[too-many-return-statements]
        self,
    ) -> str:
        """
        Render a short human description, for error and log messages.

        Returns
        -------
        str
            A phrase such as ``"the user playlist PLXOJEg4xbr50"`` or
            ``"the 'podcasts' tab of channel @cs50"``.
        """
        if self.kind == VIDEO:
            return f"the video {self.video_id}"
        if self.kind == PLAYLIST:
            return f"the {self.playlist_kind} playlist {self.playlist_id}"
        if self.kind == CLIP:
            return f"the clip {self.clip_id}"
        if self.kind == POST:
            return f"the community post {self.post_id}"
        if self.kind == SEARCH:
            return f"a search for {self.search_query!r}"
        if self.kind == HASHTAG:
            return f"the hashtag #{self.hashtag}"
        if self.kind == FEED:
            return f"the {self.feed!r} feed"
        target = (
            self.channel_id
            or (f"@{self.handle}" if self.handle else "")
            or self.channel_name
            or "?"
        )
        if self.tab:
            suffix = "" if self.tab_known else " (tab not recognised by this build)"
            return f"the {self.tab!r} tab of channel {target}{suffix}"
        return f"the channel {target}"


# -- shape classification -----------------------------------------------------


#: A value is treated as a URL when it carries an explicit scheme separator,
#: is protocol-relative, or is rooted at a dotted host followed by a path.
#: Everything else -- ``CS50``, ``@handle``, ``UC…``, a plain display name, or
#: a name that merely contains a slash ("Foo / Bar") -- is not URL-shaped.
_URL_SHAPED_RE = re.compile(
    r"""^(?:
          //                              # protocol-relative
        | [a-z][a-z0-9+.\-]*://           # explicit scheme with authority
        | [a-z][a-z0-9+.\-]*:[^\s]*/      # scheme URI carrying a path
        | [\w\-]+(?:\.[\w\-]+)+/         # schemeless dotted host, rooted
        )""",
    re.IGNORECASE | re.VERBOSE,
)


def is_reference_url(value: Any) -> bool:
    """
    Test whether a value is *shaped* like a URL, without judging its host.

    Parameters
    ----------
    value : Any
        Any candidate reference: a URL, a handle, a bare id, or a plain
        display name.

    Returns
    -------
    bool
        ``True`` when ``value`` is written as a URL and must therefore be
        validated by :func:`parse_reference` before being trusted or shown.

    See Also
    --------
    parse_reference : Validates the host and extracts the identifiers.

    Notes
    -----
    User: this answers "did the author paste a link here?" -- not "is this
    link a YouTube link?".  Only :func:`parse_reference` answers the second
    question, and it is the single place where a host is judged.

    Developer: exists so that no call site re-implements host detection with
    a substring test such as ``"youtube.com/" in value``.  A substring test
    is wrong in both directions.  It accepts
    ``https://evil.example/youtube.com/x`` -- the allowed host appears at an
    arbitrary position -- and it rejects ``https://youtu.be/ID``,
    ``https://music.youtube.com/watch?v=ID`` and every country domain, which
    then bypass validation entirely and are handled as if they were plain
    display text.  Splitting the decision in two -- *is it a URL* here, *is
    it a YouTube URL* in :func:`parse_reference` -- keeps the host allowlist
    in exactly one place.

    Examples
    --------
    >>> is_reference_url("https://www.youtube.com/@cs50")
    True
    >>> is_reference_url("www.youtube.com/@cs50")
    True
    >>> is_reference_url("//youtu.be/hKpAHgT9VxM")
    True
    >>> is_reference_url("https://evil.example/youtube.com/x")
    True
    >>> is_reference_url("javascript:alert(1)//x.y/")
    True
    >>> is_reference_url("@cs50")
    False
    >>> is_reference_url("CS50")
    False
    >>> is_reference_url("Foo / Bar")
    False
    >>> is_reference_url("CS50: Introduction to Computer Science")
    False
    """
    if not isinstance(value, str):
        return False
    return bool(_URL_SHAPED_RE.search(value.strip()))


# -- normalisation ------------------------------------------------------------


def _normalize(value: Any) -> str:
    """
    Clean a pasted reference before structural parsing.

    Strips the wrapping and encoding damage a URL picks up travelling
    through email clients, chat apps, HTML and Markdown: surrounding
    whitespace, angle brackets, matching quotes, a trailing Markdown link
    paren or sentence punctuation, HTML-escaped ampersands, invisible
    zero-width characters, and a missing scheme.

    Parameters
    ----------
    value : Any
        The raw input.

    Returns
    -------
    str
        A cleaned string, with a scheme guaranteed for anything URL-shaped.

    Raises
    ------
    ReferenceError
        If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str):
        raise ReferenceError(f"expected a string reference, got {type(value).__name__}")
    text = value.strip()
    # Zero-width and soft-hyphen characters survive copy-paste from rendered
    # pages and are invisible in any resulting error message.
    text = re.sub(r"[\u200b-\u200f\u00ad\ufeff]", "", text)
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    _len = len(text) >= 2  # ruff: ignore[magic-value-comparison]
    if _len and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    # Word processors and chat apps substitute curly quotes, whose opening
    # and closing forms differ, so the symmetric check above cannot see them.
    if _len and text[0] in "\u2018\u201c\u00ab" and text[-1] in "\u2019\u201d\u00bb":
        text = text[1:-1].strip()
    text = text.replace("&amp;", "&")
    # A link pasted from prose arrives wrapped in a sentence, a Markdown
    # link, or brackets: "Watch https://youtu.be/ID." or "(https://…)".
    # Lifting the first URL-shaped run out of the surrounding text is
    # strictly better than rejecting an input whose intent is unambiguous.
    if not _URL_START_RE.match(text):
        embedded = _EMBEDDED_URL_RE.search(text)
        if embedded:
            text = embedded.group(0)
    # Trailing sentence and Markdown punctuation is never part of a URL.
    text = text.rstrip(").,;:!?'\"\u2018\u2019\u201c\u201d\u00bb\u203a")
    if not text:
        raise ReferenceError("empty reference")
    if text.startswith("//"):
        return "https:" + text
    # A bare query fragment -- '?v=nfYOp3_SyqM&' or 'v=ID&list=PL…' -- is what
    # a partial copy out of an address bar or a log line looks like. It
    # carries every identifier a watch URL does, so completing it beats
    # rejecting a reference whose meaning is unambiguous.
    fragment = text[1:] if text.startswith("?") else text
    if "://" not in text and re.match(r"^(?:v|list|index|t|start)=", fragment):
        return "https://www.youtube.com/watch?" + fragment
    if "://" not in text and re.match(
        r"^[\w.-]*(?:youtube\.[a-z.]+|youtu\.be|youtubekids\.com)/", text, re.IGNORECASE
    ):
        return "https://" + text
    return text


def _host_of(url: str) -> str:
    """
    Extract a normalised host from a URL.

    Parameters
    ----------
    url : str
        Any URL.

    Returns
    -------
    str
        Lowercased host with userinfo and port removed.
    """
    return urlsplit(url).netloc.lower().split("@")[-1].split(":")[0]


def _is_youtube_host(host: str) -> bool:
    """
    Test whether a host belongs to the YouTube ecosystem.

    Parameters
    ----------
    host : str
        A lowercased host with any port and userinfo removed.

    Returns
    -------
    bool
        ``True`` for any subdomain of a known YouTube domain, or a YouTube
        country-code domain.

    Notes
    -----
    Matched by suffix on a *dot boundary*, never by substring. A substring
    test would accept ``youtube.com.evil.example``, turning this parser into
    an open redirect for anything that trusts its verdict.

    Examples
    --------
    >>> _is_youtube_host("music.youtube.com")
    True
    >>> _is_youtube_host("youtube.co.uk")
    True
    >>> _is_youtube_host("youtube.com.evil.example")
    False
    """
    for suffix in _HOST_SUFFIXES:
        if host == suffix or host.endswith("." + suffix):
            return True
    return bool(_CCTLD_HOST_RE.search(host))


def _unwrap(text: str) -> tuple[str, str]:
    """
    Follow YouTube's own URL wrappers to the destination they encapsulate.

    Handles the consent gate (``consent.youtube.com/?continue=…``), the
    outbound redirector (``/redirect?q=…``) and attribution links
    (``/attribution_link?u=%2Fwatch%3Fv%3D…``) -- what a reader gets when
    copying a link out of a cookie banner, a video description, or an
    embedded player.

    Parameters
    ----------
    text : str
        A normalised URL.

    Returns
    -------
    destination : str
        The innermost URL found, or ``text`` unchanged.
    wrapper : str
        The outermost wrapper that was unwrapped, or ``""``.

    Notes
    -----
    Bounded by :data:`_MAX_UNWRAP`. Only destinations that are themselves
    YouTube URLs or YouTube-relative paths are followed: an outbound
    ``/redirect?q=https://example.com`` is left alone rather than being
    reported as a YouTube reference it is not.
    """
    wrapper = ""
    current = text
    for _ in range(_MAX_UNWRAP):
        if not _is_youtube_host(_host_of(current)):
            break
        query = dict(parse_qsl(urlsplit(current).query, keep_blank_values=False))
        candidate = ""
        for param in _WRAPPER_PARAMS:
            if param in query:
                candidate = unquote(query[param]).strip()
                break
        if not candidate:
            break
        if candidate.startswith("/"):
            candidate = f"https://www.youtube.com{candidate}"
        elif "://" not in candidate or not _is_youtube_host(_host_of(candidate)):
            break
        wrapper = wrapper or current
        current = candidate
    return current, wrapper


# -- parse context ------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    """
    Everything a path handler needs, computed once per parse.

    Attributes
    ----------
    raw : str
        Original input, for error messages.
    host : str
        Normalised host.
    segments : list of str
        Decoded, non-empty path segments.
    query : dict
        Decoded query parameters.
    playlist_id, playlist_kind : str
        Playlist identifiers from a ``list`` parameter, if any.
    index, start : int or None
        Position and offset parameters.
    ephemeral : bool
        Whether the ``list`` parameter names viewer-specific content.
    wrapper : str
        Wrapper URL this was unwrapped from, if any.
    """

    raw: str
    host: str
    segments: list[str]
    query: dict[str, str]
    playlist_id: str
    playlist_kind: str
    index: int | None
    start: int | None
    ephemeral: bool
    wrapper: str
    playlist_error: str = ""

    def require_playlist(self) -> None:
        """
        Assert that a usable playlist id was present.

        Raises
        ------
        ReferenceError
            If the ``list`` parameter was missing or malformed. Called only
            by handlers whose URL *means* a playlist, so that a junk ``list``
            on an ordinary watch URL degrades to "no playlist context"
            instead of rejecting a video link that is otherwise valid.
        """
        if self.playlist_error:
            raise ReferenceError(self.playlist_error)
        if not self.playlist_id:
            raise ReferenceError(f"{self.raw!r}: URL has no 'list' parameter")

    @property
    def head(self) -> str:
        """
        First path segment, lowercased.

        Returns
        -------
        str
            The segment, or ``""`` for a root URL.
        """
        return self.segments[0].lower() if self.segments else ""

    def base(self, **overrides: Any) -> YouTubeReference:
        """
        Build a reference pre-filled with the context's shared fields.

        Parameters
        ----------
        **overrides
            Field values specific to the calling handler.

        Returns
        -------
        YouTubeReference
            The assembled reference.
        """
        fields: dict[str, Any] = {
            "playlist_id": self.playlist_id,
            "playlist_kind": self.playlist_kind,
            "index": self.index,
            "start": self.start,
            "ephemeral": self.ephemeral,
            "host": self.host,
            "unwrapped_from": self.wrapper,
            "raw": self.raw,
            "params": self.query,
        }
        fields.update(overrides)
        return YouTubeReference(**fields)

    def video(self, video_id: str) -> YouTubeReference:
        """
        Build a VIDEO reference, validating the id.

        Parameters
        ----------
        video_id : str
            Candidate video id.

        Returns
        -------
        YouTubeReference
            A validated video reference.
        """
        return self.base(kind=VIDEO, video_id=_checked_video_id(video_id, self.raw))


# -- path handlers ------------------------------------------------------------
#
# Each handler inspects the context and returns a reference, or ``None`` to
# decline. Adding a URL shape is one function plus one entry in the table.


def _handle_short_host(ctx: _Context) -> YouTubeReference | None:
    """Handle ``youtu.be/<id>``."""
    if ctx.host != "youtu.be" and not ctx.host.endswith(".youtu.be"):
        return None
    if not ctx.segments:
        raise ReferenceError(f"{ctx.raw!r}: youtu.be link carries no video id")
    return ctx.video(ctx.segments[0])


def _handle_watch(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/watch?v=``, ``/watch/<id>``, ``/watch?list=`` and ad-hoc lists."""
    if ctx.head not in ("watch", "watch_videos", "watch_popup"):
        return None
    if ctx.query.get("v"):
        return ctx.video(ctx.query["v"])
    if len(ctx.segments) > 1:
        return ctx.video(ctx.segments[1])
    if ctx.playlist_id or ctx.playlist_error:
        ctx.require_playlist()
        return ctx.base(kind=PLAYLIST)
    if ctx.query.get("video_ids"):
        # `/watch_videos?video_ids=a,b,c` is an ad-hoc playlist; its first
        # entry is the video the URL opens on.
        return ctx.video(ctx.query["video_ids"].split(",")[0])
    raise ReferenceError(
        f"{ctx.raw!r}: watch URL has neither a 'v' nor a 'list' parameter"
    )


def _handle_embed(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/embed/<id>``, ``/v/<id>``, ``/e/<id>`` and ``/embed/videoseries``."""
    if ctx.head not in ("embed", "v", "e"):
        return None
    if len(ctx.segments) < 2:  # ruff: ignore[magic-value-comparison]
        raise ReferenceError(f"{ctx.raw!r}: {ctx.head!r} URL carries no video id")
    if ctx.segments[1].lower() == "videoseries":
        ctx.require_playlist()
        return ctx.base(kind=PLAYLIST)
    return ctx.video(ctx.segments[1])


def _handle_short_form(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/shorts/<id>`` and ``/live/<id>``."""
    if ctx.head not in ("shorts", "live"):
        return None
    if len(ctx.segments) < 2:  # ruff: ignore[magic-value-comparison]
        raise ReferenceError(f"{ctx.raw!r}: {ctx.head!r} URL carries no video id")
    return ctx.video(ctx.segments[1])


def _handle_playlist(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/playlist?list=``."""
    if ctx.head != "playlist":
        return None
    ctx.require_playlist()
    return ctx.base(kind=PLAYLIST)


def _handle_clip(ctx: _Context) -> YouTubeReference | None:
    """
    Handle ``/clip/<id>``.

    A Clip URL carries no video id. It is returned as its own kind rather
    than rejected, so a consumer can say "clips must be resolved through the
    API" instead of "not a YouTube URL".
    """
    if ctx.head != "clip":
        return None
    if len(ctx.segments) < 2:  # ruff: ignore[magic-value-comparison]
        raise ReferenceError(f"{ctx.raw!r}: clip URL carries no clip id")
    return ctx.base(kind=CLIP, clip_id=ctx.segments[1])


def _handle_post(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/post/<id>``."""
    if ctx.head != "post":
        return None
    if len(ctx.segments) < 2:  # ruff: ignore[magic-value-comparison]
        raise ReferenceError(f"{ctx.raw!r}: post URL carries no post id")
    return ctx.base(kind=POST, post_id=ctx.segments[1])


def _handle_search(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/results?search_query=``."""
    if ctx.head != "results":
        return None
    return ctx.base(
        kind=SEARCH,
        search_query=ctx.query.get("search_query") or ctx.query.get("q", ""),
    )


def _handle_hashtag(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/hashtag/<name>``."""
    if ctx.head != "hashtag":
        return None
    if len(ctx.segments) < 2:  # ruff: ignore[magic-value-comparison]
        raise ReferenceError(f"{ctx.raw!r}: hashtag URL carries no tag")
    return ctx.base(kind=HASHTAG, hashtag=ctx.segments[1].lstrip("#"))


def _handle_feed(ctx: _Context) -> YouTubeReference | None:
    """
    Handle ``/feed/<name>``.

    Personal feeds are marked ephemeral: their contents depend on who is
    signed in, so they can never be a reproducible catalog source.
    """
    if ctx.head != "feed":
        return None
    name = ctx.segments[1].lower() if len(ctx.segments) > 1 else ""
    personal = name in {"subscriptions", "history", "library", "you", "downloads"}
    return ctx.base(kind=FEED, feed=name, ephemeral=personal)


def _tab_from(segments: list[str], offset: int) -> tuple[str, bool]:
    """
    Extract a channel tab from the segment after a channel identifier.

    Parameters
    ----------
    segments : list of str
        All path segments.
    offset : int
        Index of the segment following the channel identifier.

    Returns
    -------
    tab : str
        The tab name, or ``""``.
    known : bool
        Whether the tab is in :data:`_KNOWN_TABS`. An unknown tab is
        returned, not rejected: YouTube ships new tabs, and a parse failure
        on the day one lands would break every page that mentions it.
    """
    if len(segments) <= offset:
        return "", True
    tab = segments[offset].lower()
    return tab, tab in _KNOWN_TABS


def _handle_channel_prefix(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/channel/UC…``, ``/c/NAME`` and ``/user/NAME``, with tabs."""
    if ctx.head not in _CHANNEL_PREFIXES:
        return None
    if len(ctx.segments) < 2:  # ruff: ignore[magic-value-comparison]
        raise ReferenceError(
            f"{ctx.raw!r}: {ctx.head!r} URL carries no channel identifier"
        )
    tab, known = _tab_from(ctx.segments, 2)
    if _CHANNEL_PREFIXES[ctx.head] == "id":
        if not _CHANNEL_ID_RE.match(ctx.segments[1]):
            raise ReferenceError(
                f"{ctx.raw!r}: {ctx.segments[1]!r} is not a valid channel id "
                f"(expected 'UC' followed by 22 characters)"
            )
        return ctx.base(
            kind=CHANNEL, channel_id=ctx.segments[1], tab=tab, tab_known=known
        )
    return ctx.base(
        kind=CHANNEL, channel_name=ctx.segments[1], tab=tab, tab_known=known
    )


def _handle_handle(ctx: _Context) -> YouTubeReference | None:
    """Handle ``/@handle`` and ``/@handle/<tab>``."""
    if not ctx.segments or not ctx.segments[0].startswith("@"):
        return None
    handle = validate_handle(ctx.segments[0])
    tab, known = _tab_from(ctx.segments, 1)
    return ctx.base(kind=CHANNEL, handle=handle, tab=tab, tab_known=known)


def _handle_profile(ctx: _Context) -> YouTubeReference | None:
    """Handle the very old ``/profile?user=NAME`` form."""
    if ctx.head != "profile":
        return None
    name = ctx.query.get("user", "")
    if not name:
        raise ReferenceError(f"{ctx.raw!r}: profile URL carries no 'user' parameter")
    return ctx.base(kind=CHANNEL, channel_name=name)


def _handle_vanity(ctx: _Context) -> YouTubeReference | None:
    """
    Handle a bare ``/NAME`` legacy vanity URL, e.g. ``youtube.com/cs50``.

    Declines reserved top-level segments so YouTube's own pages are never
    misread as channels, and declines paths deeper than ``/NAME/tab``.
    """
    if not ctx.segments or ctx.head in _RESERVED_SEGMENTS:
        return None
    if (
        ctx.segments[0].startswith("@")  # lint
        or len(ctx.segments) > 2  # ruff: ignore[magic-value-comparison]
    ):
        return None
    tab, known = _tab_from(ctx.segments, 1)
    return ctx.base(
        kind=CHANNEL, channel_name=ctx.segments[0], tab=tab, tab_known=known
    )


def _handle_bare_list(ctx: _Context) -> YouTubeReference | None:
    """Handle any remaining URL that still carries a usable ``list``."""
    if not ctx.playlist_id:
        return None
    return ctx.base(kind=PLAYLIST)


#: Ordered path handlers. Order matters only where two could match; each
#: returns ``None`` to decline. Registering a new YouTube URL shape is one
#: entry here plus its handler function.
_PATH_HANDLERS: tuple[Callable[[_Context], YouTubeReference | None], ...] = (
    _handle_short_host,
    _handle_watch,
    _handle_embed,
    _handle_short_form,
    _handle_playlist,
    _handle_clip,
    _handle_post,
    _handle_search,
    _handle_hashtag,
    _handle_feed,
    _handle_profile,
    _handle_channel_prefix,
    _handle_handle,
    _handle_vanity,
    _handle_bare_list,
)


def _parse_bare(text: str, raw: str) -> YouTubeReference | None:
    """
    Parse a bare identifier that is not a URL.

    Parameters
    ----------
    text : str
        A string containing no URL punctuation.
    raw : str
        The original input.

    Returns
    -------
    YouTubeReference or None
        A reference, or ``None`` if the text is not a recognised identifier.

    Notes
    -----
    Bare-id recognition requires the absence of URL punctuation, which stops
    an 11-character path fragment or channel name from being mistaken for a
    video id. Unknown-prefix playlist ids are *not* accepted here: with no
    ``list=`` parameter to mark intent, an arbitrary opaque string is far
    more likely to be a typo than a playlist.
    """
    if text.startswith("@"):
        return YouTubeReference(kind=CHANNEL, handle=validate_handle(text), raw=raw)
    if _CHANNEL_ID_RE.match(text):
        return YouTubeReference(kind=CHANNEL, channel_id=text, raw=raw)
    if _VIDEO_ID_RE.match(text):
        return YouTubeReference(kind=VIDEO, video_id=text, raw=raw)
    if _OPAQUE_RE.match(text) and len(text) >= _MIN_PLAYLIST_ID_LEN:
        kind = _classify_playlist(text)  # noqa: F841 - read below
        if kind != "unknown":
            return YouTubeReference(
                kind=PLAYLIST,
                playlist_id=text,
                playlist_kind=kind,
                ephemeral=kind in _EPHEMERAL_PLAYLIST_KINDS,
                raw=raw,
            )
    return None


def parse_reference(value: Any) -> YouTubeReference:
    """
    Parse any YouTube reference into a typed :class:`YouTubeReference`.

    Parameters
    ----------
    value : Any
        A bare identifier, or a YouTube URL in any form the ecosystem
        produces -- including consent, redirect and attribution wrappers.

    Returns
    -------
    YouTubeReference
        The parsed reference.

    Raises
    ------
    ReferenceError
        If the value is not a YouTube reference, or is one whose identifiers
        are malformed. Every message names the input and the accepted forms.

    Notes
    -----
    When a watch URL carries both ``v`` and ``list``, the kind is
    :data:`VIDEO` -- that URL was produced by watching one video -- but
    ``playlist_id`` is retained so a collection consumer can prefer it.
    Resolving this ambiguity in the parser, rather than at each call site,
    is what keeps the directive and the sync tool consistent.

    Examples
    --------
    >>> parse_reference("nfYOp3_SyqM").kind
    'video'
    >>> ref = parse_reference(
    ...     "https://www.youtube.com/watch?v=nfYOp3_SyqM&list=PLXOJEg4xbr50&index=2"
    ... )
    >>> ref.kind, ref.video_id, ref.playlist_id, ref.index
    ('video', 'nfYOp3_SyqM', 'PLXOJEg4xbr50', 2)
    >>> parse_reference("https://youtu.be/hKpAHgT9VxM?t=1m30s").start
    90
    >>> parse_reference("https://www.youtube.com/shorts/hbT7vzCvEc8").video_id
    'hbT7vzCvEc8'
    >>> ref = parse_reference("https://www.youtube.com/@cs50/podcasts")
    >>> ref.kind, ref.handle, ref.tab
    ('channel', 'cs50', 'podcasts')
    >>> parse_reference("https://www.youtube.com/cs50").channel_name
    'cs50'
    >>> parse_reference(
    ...     "https://www.youtube.com/embed/videoseries?list=PLXSX209johrU"
    ... ).kind
    'playlist'
    >>> parse_reference("https://www.youtube.com/@cs50/newtabname").tab_known
    False
    """
    raw = value if isinstance(value, str) else repr(value)
    text = _normalize(value)

    if not re.search(r"[/:.?&]", text):
        reference = _parse_bare(text, raw)
        if reference is not None:
            return reference
        raise ReferenceError(
            f"{raw!r} is not a recognised YouTube reference: it is neither a "
            f"URL nor a bare video id (11 characters), playlist id (PL…), "
            f"channel id (UC…) or handle (@name)"
        )

    text, wrapper = _unwrap(text)
    parts = urlsplit(text)
    host = _host_of(text)
    if not _is_youtube_host(host):
        raise ReferenceError(
            f"{raw!r} is not a YouTube URL (host {parts.netloc!r}); recognised "
            f"domains are {list(_HOST_SUFFIXES)} and YouTube country domains"
        )

    query = dict(parse_qsl(parts.query, keep_blank_values=False))
    segments = [seg for seg in unquote(parts.path).split("/") if seg]

    playlist_id = ""
    playlist_kind = ""
    playlist_error = ""
    ephemeral = False
    if query.get("list"):
        # Validated leniently on purpose. On `/watch?v=…&list=junk` the video
        # is unambiguous and the playlist is incidental context, so the URL
        # still resolves; only handlers whose URL *means* a playlist call
        # `require_playlist()` and surface this error.
        try:
            playlist_id = _checked_playlist_id(query["list"], raw)
            playlist_kind = _classify_playlist(playlist_id)
            ephemeral = playlist_kind in _EPHEMERAL_PLAYLIST_KINDS
        except ReferenceError as exc:
            playlist_error = str(exc)

    index = int(query["index"]) if query.get("index", "").isdigit() else None

    start = None
    for param in _START_PARAMS:
        if param in query:
            start = _parse_start(query[param])
            if start is not None:
                break

    ctx = _Context(
        raw=raw,
        host=host,
        segments=segments,
        query=query,
        playlist_id=playlist_id,
        playlist_kind=playlist_kind,
        index=index,
        start=start,
        ephemeral=ephemeral,
        wrapper=wrapper,
        playlist_error=playlist_error,
    )

    for handler in _PATH_HANDLERS:
        reference = handler(ctx)
        if reference is not None:
            return reference

    raise ReferenceError(
        f"{raw!r} is a YouTube URL but names no video, playlist, channel, "
        f"clip, post, search or hashtag (path {parts.path!r})"
    )


def parse_video_reference(value: Any) -> YouTubeReference:
    """
    Parse a reference that must name exactly one video.

    Parameters
    ----------
    value : Any
        Any YouTube reference.

    Returns
    -------
    YouTubeReference
        A reference whose ``kind`` is :data:`VIDEO`.

    Raises
    ------
    ReferenceError
        If the reference names anything other than a single video. The
        message names what was found and points at the directive or step
        that *can* handle it -- the alternative, embedding the first video
        of a channel or nothing at all, would silently misread the author's
        intent.

    Examples
    --------
    >>> parse_video_reference("https://youtu.be/hbT7vzCvEc8").video_id
    'hbT7vzCvEc8'
    >>> try:
    ...     parse_video_reference("https://www.youtube.com/@cs50/playlists")
    ... except ReferenceError as exc:
    ...     print("refused")
    refused
    """
    reference = parse_reference(value)
    if reference.kind == VIDEO:
        return reference
    if reference.kind == CLIP:
        raise ReferenceError(
            f"{reference.raw!r} names {reference.describe()}. A Clip URL does "
            f"not contain the underlying video id, so it cannot be embedded "
            f"directly; open the clip and use the full video's URL."
        )
    if reference.kind in (POST, SEARCH, HASHTAG, FEED):
        raise ReferenceError(
            f"{reference.raw!r} names {reference.describe()}, which holds no "
            f"embeddable video."
        )
    raise ReferenceError(
        f"{reference.raw!r} names {reference.describe()}, not a single video. "
        f"Use the 'youtube-gallery' directive for collections (for example "
        f"':playlist: PL…' or ':channel: @name')."
    )
