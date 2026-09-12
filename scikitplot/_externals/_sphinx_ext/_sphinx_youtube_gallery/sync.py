r"""
Offline acquisition tool for the YouTube learning catalog.

This module is the *only* place that talks to YouTube, and it is never
imported by the Sphinx build. Run it deliberately -- by hand or on a
schedule -- to refresh a catalog file, review the diff, and commit it::

    python -m scikitplot._externals._sphinx_ext._sphinx_youtube_gallery.sync \
        --playlist PLxxxxxxxxxxxxxxxxxx \
        --output docs/_data/youtube.yaml

For the standalone package layout, start the command with
``python -m _sphinx_ext._sphinx_youtube_gallery.sync`` instead.

Notes
-----
**User-focused.** Without an API key, channel and playlist collection sources
fall back to YouTube's public RSS feeds, which need no credentials but expose
only roughly the 15 most recent items. An exact video source remains usable as
an ID-only record. With ``YOUTUBE_API_KEY`` set, provider metadata/full-history
collection acquisition uses the Data API v3.

**Developer-focused.** The output is deterministic and idempotent: records
are sorted by a stable key, timestamps are normalized to UTC, and re-running
against unchanged upstream data rewrites a byte-identical file. That is what
makes "did this sync actually change anything?" answerable from
``git diff`` rather than from trust, and what keeps the tool safe to run in
CI on a schedule.

The API key is read from the environment only. It is never accepted as a
command-line argument, because arguments leak into shell history, process
listings and CI logs.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

from .._sphinx_collection._yaml import (
    MAX_COLLECTION_ITEMS,
    BoundedYAMLError,
    load_bounded_yaml,
    read_bounded_utf8,
)
from .._sphinx_youtube_core.reference import (
    CHANNEL,
    CLIP,
    FEED,
    HASHTAG,
    PLAYLIST,
    POST,
    SEARCH,
    VIDEO,
    ReferenceError,
    YouTubeReference,
    parse_reference,
    validate_channel_id,
    validate_playlist_id,
)
from .model import CatalogError, VideoRecord, normalize_catalog

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_channel",
    "fetch_channel_playlists",
    "fetch_playlist",
    "fetch_video",
    "main",
    "merge_catalog_enrichment",
    "render_catalog",
    "resolve_source",
    "write_catalog",
]

#: YouTube Data API v3 base URL.
API_BASE = "https://www.googleapis.com/youtube/v3"

#: Public RSS feed used when no API key is available.
RSS_BASE = "https://www.youtube.com/feeds/videos.xml"

#: Connect and read timeout, in seconds, for every outbound request. Every
#: network call is bounded: an unbounded one turns a scheduled refresh into a
#: job that hangs until the CI runner is killed.
TIMEOUT = (5, 30)

#: Maximum result pages to walk before giving up. A bound is required
#: because a paging loop driven by a server-supplied token is otherwise
#: unbounded if the server ever returns a cycle.
MAX_PAGES = 200


def _require_requests():
    """
    Import ``requests`` lazily, with an actionable error if it is absent.

    Returns
    -------
    module
        The ``requests`` module.

    Raises
    ------
    CatalogError
        If ``requests`` is not installed. It is deliberately not a hard
        dependency of the documentation build, only of this optional tool.
    """
    try:
        import requests  # ruff: ignore[import-outside-top-level]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CatalogError(
            "the sync tool needs 'requests' (pip install 'requests>=2.28'); "
            "it is not required to build the documentation"
        ) from exc
    return requests


def _api_pages(endpoint: str, params: dict[str, Any], api_key: str) -> Iterator[dict]:
    """
    Yield successive pages from a YouTube Data API v3 endpoint.

    Parameters
    ----------
    endpoint : str
        Endpoint name, e.g. ``"playlistItems"``.
    params : dict
        Query parameters, excluding ``key`` and ``pageToken``.
    api_key : str
        Data API v3 key.

    Yields
    ------
    dict
        One decoded response page.

    Raises
    ------
    CatalogError
        On any transport or HTTP error, or if paging exceeds
        :data:`MAX_PAGES`.
    """
    requests = _require_requests()
    token = None
    seen_tokens: set[str] = set()
    for _ in range(MAX_PAGES):
        query = dict(params, key=api_key, maxResults=50)
        if token:
            query["pageToken"] = token
        try:
            response = requests.get(
                f"{API_BASE}/{endpoint}", params=query, timeout=TIMEOUT
            )
        except Exception as exc:  # requests.RequestException without hard dependency
            # Do not interpolate the exception text: requests commonly includes
            # the fully prepared URL there, which would leak the API key from
            # the query string into CI logs.
            raise CatalogError(
                f"YouTube API request to {endpoint} failed ({type(exc).__name__})"
            ) from exc

        try:
            response.raise_for_status()
        except Exception as exc:
            status = getattr(response, "status_code", None)
            detail = f"HTTP {status}" if isinstance(status, int) else type(exc).__name__
            raise CatalogError(
                f"YouTube API request to {endpoint} failed ({detail})"
            ) from exc

        try:
            page = response.json()
        except Exception as exc:
            raise CatalogError(
                f"YouTube API response from {endpoint} was not valid JSON"
            ) from exc
        if not isinstance(page, dict):
            raise CatalogError(
                f"YouTube API response from {endpoint} was a "
                f"{type(page).__name__}, expected an object"
            )
        if "items" in page and not isinstance(page["items"], list):
            raise CatalogError(
                f"YouTube API response from {endpoint} has non-list 'items'"
            )
        next_token = page.get("nextPageToken")
        if next_token is not None and not isinstance(next_token, str):
            raise CatalogError(
                f"YouTube API response from {endpoint} has a non-string nextPageToken"
            )

        yield page
        token = next_token or None
        if not token:
            return
        if token in seen_tokens:
            raise CatalogError(
                f"YouTube API paging for {endpoint} repeated page token; "
                "refusing to loop"
            )
        seen_tokens.add(token)
    raise CatalogError(
        f"YouTube API paging exceeded {MAX_PAGES} pages for {endpoint}; "
        f"refusing to continue"
    )


def _rss_items(feed_param: str, value: str) -> list[dict[str, Any]]:
    """
    Fetch and parse a public YouTube RSS feed.

    Parameters
    ----------
    feed_param : str
        Either ``"channel_id"`` or ``"playlist_id"``.
    value : str
        The identifier to request.

    Returns
    -------
    list of dict
        Raw catalog records. RSS carries no duration, so ``duration`` is
        absent rather than guessed.

    Raises
    ------
    CatalogError
        On any transport, HTTP or XML parsing error.
    """
    # import xml.etree.ElementTree as ET  # ruff: ignore[import-outside-top-level]
    from defusedxml import (  # ruff: ignore[camelcase-imported-as-acronym, import-outside-top-level]
        ElementTree as ET,
    )

    requests = _require_requests()
    try:
        response = requests.get(RSS_BASE, params={feed_param: value}, timeout=TIMEOUT)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        raise CatalogError(f"RSS fetch for {value} failed: {exc}") from exc

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
        "media": "http://search.yahoo.com/mrss/",
    }
    records = []
    feed_title = root.findtext("atom:title", default="", namespaces=ns)
    feed_author = root.findtext("atom:author/atom:name", default="", namespaces=ns)
    channel = feed_author or (feed_title if feed_param == "channel_id" else "")
    playlist = feed_title if feed_param == "playlist_id" else ""
    for entry in root.findall("atom:entry", ns):
        video_id = entry.findtext("yt:videoId", default="", namespaces=ns)
        if not video_id:
            continue
        group = entry.find("media:group", ns)
        description = ""
        if group is not None:
            description = group.findtext("media:description", default="", namespaces=ns)
        records.append(
            {
                "id": video_id,
                "title": entry.findtext("atom:title", default="", namespaces=ns),
                "description": description,
                "channel": channel,
                "channel_id": entry.findtext("yt:channelId", default="", namespaces=ns),
                "published": entry.findtext(
                    "atom:published", default="", namespaces=ns
                ),
                **({"playlist": playlist, "playlist_id": value} if playlist else {}),
            }
        )
    return records


def fetch_video(video_id: str, api_key: str = "") -> list[dict[str, Any]]:
    """
    Fetch metadata for one exact video when the Data API is available.

    Without a key there is no stable public per-video metadata endpoint used by
    this extension, so the exact id is returned as a minimal record.  During a
    partial/no-key refresh, merge logic preserves richer existing metadata for
    fields this minimal record cannot observe.
    """
    if not api_key:
        return [{"id": video_id}]

    for page in _api_pages(
        "videos", {"part": "snippet,contentDetails", "id": video_id}, api_key
    ):
        for item in page.get("items", []):
            snippet = item.get("snippet", {})
            return [
                {
                    "id": item.get("id", video_id),
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "published": snippet.get("publishedAt", ""),
                    "duration": item.get("contentDetails", {}).get("duration"),
                }
            ]
    raise CatalogError(f"video {video_id!r} was not found on YouTube")


def _playlist_title(playlist_id: str, api_key: str) -> str | None:
    """Return one playlist title, or ``None`` when the id is not accessible."""
    for page in _api_pages(
        "playlists", {"part": "snippet", "id": playlist_id}, api_key
    ):
        for item in page.get("items", []):
            if item.get("id") == playlist_id:
                return str(item.get("snippet", {}).get("title", "")).strip()
    return None


def _video_durations(video_ids: Sequence[str], api_key: str) -> dict[str, str]:
    """Fetch ISO-8601 durations in bounded batches of at most 50 video ids."""
    result: dict[str, str] = {}
    ids = [video_id for video_id in dict.fromkeys(video_ids) if video_id]
    for start in range(0, len(ids), 50):
        batch = ids[start : start + 50]
        for page in _api_pages(
            "videos", {"part": "contentDetails", "id": ",".join(batch)}, api_key
        ):
            for item in page.get("items", []):
                video_id = str(item.get("id", ""))
                duration = item.get("contentDetails", {}).get("duration")
                if video_id and duration:
                    result[video_id] = duration
    return result


def fetch_playlist(  # ruff: ignore[undocumented-param]
    playlist_id: str,
    api_key: str = "",
    *,
    playlist_title: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch every video in a playlist as raw catalog records.

    Parameters
    ----------
    playlist_id : str
        A ``PL...`` playlist identifier.
    api_key : str, optional
        Data API v3 key. When empty, the public RSS feed is used instead,
        which returns only the most recent items.

    Returns
    -------
    list of dict
        Raw records, ready for :func:`~.model.normalize_catalog`.
    """
    if not api_key:
        return _rss_items("playlist_id", playlist_id)

    if playlist_title is None:
        playlist_title = _playlist_title(playlist_id, api_key)
        if playlist_title is None:
            raise CatalogError(
                f"playlist {playlist_id!r} was not found or is not accessible"
            )
    records = []
    for page in _api_pages(
        "playlistItems",
        {"part": "snippet,contentDetails", "playlistId": playlist_id},
        api_key,
    ):
        for item in page.get("items", []):
            snippet = item.get("snippet", {})
            resource = snippet.get("resourceId", {})
            if resource.get("kind") != "youtube#video":
                continue
            records.append(
                {
                    "id": resource.get("videoId", ""),
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel": (
                        snippet.get("videoOwnerChannelTitle", "")
                        or snippet.get("channelTitle", "")
                    ),
                    "channel_id": snippet.get("videoOwnerChannelId", ""),
                    "playlist": playlist_title,
                    "playlist_id": playlist_id,
                    "position": snippet.get("position"),
                    "published": (
                        item.get("contentDetails", {}).get("videoPublishedAt")
                        or snippet.get("publishedAt")
                    ),
                }
            )
            if len(records) > MAX_COLLECTION_ITEMS:
                raise CatalogError(
                    f"playlist {playlist_id!r} exceeds the catalog limit of "
                    f"{MAX_COLLECTION_ITEMS:,} videos; split very large sources "
                    "across catalogs"
                )
    durations = _video_durations([record.get("id", "") for record in records], api_key)
    for record in records:
        duration = durations.get(record.get("id", ""))
        if duration:
            record["duration"] = duration
    return records


def fetch_channel(channel_id: str, api_key: str = "") -> list[dict[str, Any]]:
    """
    Fetch a channel's uploads as raw catalog records.

    Parameters
    ----------
    channel_id : str
        A ``UC...`` channel identifier.
    api_key : str, optional
        Data API v3 key. When empty, the public RSS feed is used.

    Returns
    -------
    list of dict
        Raw records.

    Notes
    -----
    With an API key this resolves the channel's ``uploads`` playlist and
    pages through it, which is the documented way to enumerate a channel's
    full upload history.
    """
    if not api_key:
        return _rss_items("channel_id", channel_id)

    for page in _api_pages(
        "channels", {"part": "contentDetails", "id": channel_id}, api_key
    ):
        for item in page.get("items", []):
            uploads = (
                item.get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads")
            )
            if uploads:
                records = fetch_playlist(uploads, api_key, playlist_title="")
                for record in records:
                    # The uploads pseudo-playlist is an implementation
                    # detail, not something a reader should ever see as a
                    # section heading.
                    record["playlist"] = ""
                    record["playlist_id"] = ""
                    record["position"] = None
                return records
    raise CatalogError(f"channel {channel_id!r} has no uploads playlist")


#: Channel tabs the Data API can enumerate, and how.
#:
#: The API has no endpoint for a channel's *tabs*. What it exposes is the
#: uploads playlist and the channel's playlist list, and every tab is a view
#: over one of those two. Stating the mapping explicitly here -- rather than
#: pretending the tabs are first-class -- is what keeps the tool honest about
#: what it can actually deliver.
#:
#: ``videos``, ``streams``, ``shorts``, ``live``, ``""``
#:     the uploads playlist. Shorts and live streams are *in* uploads but
#:     carry no flag distinguishing them, so these tabs cannot be isolated;
#:     the tool says so rather than returning a wrong subset.
#: ``playlists``, ``podcasts``, ``courses``, ``releases``
#:     the channel's playlists. Podcast/course/release groupings are
#:     presentation metadata the API does not expose, so all four resolve to
#:     every public playlist; narrow with an explicit playlist URL.
_TAB_STRATEGY = {
    "": "uploads",
    "videos": "uploads",
    "streams": "uploads",
    "shorts": "uploads",
    "live": "uploads",
    "playlists": "playlists",
    "podcasts": "playlists",
    "courses": "playlists",
    "releases": "playlists",
}

#: Tabs that resolve to a superset of what the reader asked for.
_INEXACT_TABS = {
    "shorts": (
        "Shorts are not distinguishable from other uploads in the "
        "Data API; all uploads are returned"
    ),
    "streams": (
        "live streams are not distinguishable from other uploads in "
        "the Data API; all uploads are returned"
    ),
    "live": (
        "live streams are not distinguishable from other uploads in the "
        "Data API; all uploads are returned"
    ),
    "podcasts": (
        "the Data API does not label podcast playlists; all public "
        "playlists are returned"
    ),
    "courses": (
        "the Data API does not label course playlists; all public "
        "playlists are returned"
    ),
    "releases": (
        "the Data API does not label release playlists; all public "
        "playlists are returned"
    ),
}


def resolve_channel_id(reference: YouTubeReference, api_key: str = "") -> str:
    """
    Resolve any channel reference to a canonical ``UC…`` id.

    Parameters
    ----------
    reference : YouTubeReference
        A reference whose kind is :data:`~.reference.CHANNEL`.
    api_key : str, optional
        Data API v3 key.

    Returns
    -------
    str
        The canonical channel id.

    Raises
    ------
    CatalogError
        If the channel cannot be resolved. Without an API key, only an
        explicit ``UC…`` id can be used: handles and vanity names are
        resolvable *only* through the API, and scraping the channel page for
        them would make a scheduled sync depend on YouTube's HTML, which
        changes without notice. Failing with an instruction beats a
        pipeline that breaks silently six months from now.
    """
    if reference.channel_id:
        return reference.channel_id

    target = reference.handle or reference.channel_name
    if not api_key:
        raise CatalogError(
            f"cannot resolve {reference.describe()} without an API key. "
            f"Either set YOUTUBE_API_KEY, or pass the canonical channel URL "
            f"(https://www.youtube.com/channel/UC…), which you can read off "
            f"the channel page."
        )

    # `forHandle` accepts the handle with or without '@'; `forUsername` is
    # the legacy path for pre-handle vanity names. Both are tried because a
    # bare `/NAME` URL is ambiguous between the two.
    attempts = []
    if reference.handle:
        attempts.append({"forHandle": f"@{reference.handle}"})
    if reference.channel_name:
        attempts.append({"forHandle": f"@{reference.channel_name}"})
        attempts.append({"forUsername": reference.channel_name})

    for params in attempts:
        for page in _api_pages("channels", dict(params, part="id"), api_key):
            for item in page.get("items", []):
                if item.get("id"):
                    return item["id"]
    raise CatalogError(f"channel {target!r} not found on YouTube")


def fetch_channel_playlists(channel_id: str, api_key: str) -> list[dict[str, Any]]:
    """
    Fetch a deterministic one-record-per-video view of public channel playlists.

    Parameters
    ----------
    channel_id : str
        A ``UC…`` channel id.
    api_key : str
        Data API v3 key. Required: there is no RSS feed listing a channel's
        playlists.

    Returns
    -------
    list of dict
        Raw records tagged with one stable playlist name/id. Playlist descriptors
        are sorted by playlist id before videos are fetched, and duplicate video
        ids are kept only once. This matches the catalog's canonical one-record-
        per-video model and makes the surviving playlist assignment independent
        of provider response order.

    Raises
    ------
    CatalogError
        If no API key is supplied.

    Notes
    -----
    A YouTube video may belong to several playlists. A canonical video catalog
    stores one record per video id, so this channel-wide convenience view cannot
    preserve every overlapping membership. When exact playlist membership is
    important, sync each explicit playlist into its own catalog rather than using
    a channel ``/playlists`` source.
    """
    if not api_key:
        raise CatalogError(
            "listing a channel's playlists requires YOUTUBE_API_KEY; "
            "the public RSS feeds expose videos but not playlist lists"
        )

    playlists: list[tuple[str, str]] = []
    for page in _api_pages(
        "playlists", {"part": "snippet", "channelId": channel_id}, api_key
    ):
        for item in page.get("items", []):
            playlist_id = str(item.get("id", "")).strip()
            if not playlist_id:
                continue
            title = str(item.get("snippet", {}).get("title", "")).strip()
            playlists.append((playlist_id, title))

    # Provider ordering is not an identity contract. Stable-id order keeps the
    # canonical membership deterministic if the API changes list ordering.
    playlists.sort(key=lambda pair: pair[0])
    records: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    for playlist_id, title in playlists:
        for record in fetch_playlist(playlist_id, api_key, playlist_title=title):
            video_id = str(record.get("id", "")).strip()
            if not video_id or video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)
            records.append(record)
            if len(records) > MAX_COLLECTION_ITEMS:
                raise CatalogError(
                    f"channel playlist view exceeds the catalog limit of "
                    f"{MAX_COLLECTION_ITEMS:,} unique videos; sync explicit "
                    "playlists into separate catalogs"
                )
    return records


def resolve_source(source: str, api_key: str = "") -> list[dict[str, Any]]:
    """
    Fetch raw records for any YouTube URL or identifier.

    This is the entry point that makes the tool link-driven: paste the URL
    from the address bar and the right acquisition strategy is chosen from
    its structure.

    Parameters
    ----------
    source : str
        Any YouTube reference: a video, a playlist, a channel, or a channel
        tab, in any URL form.
    api_key : str, optional
        Data API v3 key.

    Returns
    -------
    list of dict
        Raw catalog records.

    Raises
    ------
    CatalogError
        If the reference cannot be parsed or cannot be resolved.

    Notes
    -----
    A ``watch?v=…&list=…`` URL remains an **exact video** here, matching the
    standalone ``youtube`` directive and browser Add-video flow. The retained
    ``playlist_id`` is context, not an instruction to widen the acquisition.
    To ingest a collection, pass a playlist URL/ID or ``--playlist`` explicitly.
    This keeps pasted-source semantics consistent and makes destructive
    ``--prune`` intent easier to reason about.
    """
    try:
        reference = parse_reference(source)
    except ReferenceError as exc:
        raise CatalogError(str(exc)) from exc

    if reference.kind in (CLIP, POST, SEARCH, HASHTAG, FEED):
        raise CatalogError(
            f"{source!r} names {reference.describe()}, which is not an "
            f"enumerable collection of videos. Use a video, playlist or "
            f"channel URL."
        )

    if reference.kind == CHANNEL and reference.tab and not reference.tab_known:
        # A tab this build has never seen is a signal that YouTube shipped
        # something new, not a reason to guess. Guessing 'uploads' would
        # quietly ingest the wrong videos.
        raise CatalogError(
            f"{source!r} names {reference.describe()}. This build does not "
            f"know how to enumerate that tab; use the channel root, its "
            f"'videos' tab, or its 'playlists' tab, and please report the "
            f"new tab name."
        )

    if reference.ephemeral:
        raise CatalogError(
            f"{source!r} names {reference.describe()}, which YouTube "
            f"generates per viewer and per session. It is not stable content "
            f"and must not be committed to a catalog."
        )

    if reference.kind == VIDEO:
        return fetch_video(reference.video_id, api_key)

    if reference.kind == PLAYLIST:
        return fetch_playlist(reference.playlist_id, api_key)

    if reference.kind == CHANNEL:
        strategy = _TAB_STRATEGY.get(reference.tab)
        if strategy is None:
            raise CatalogError(
                f"{source!r}: the {reference.tab!r} tab holds no videos to "
                f"ingest; use the channel root, or its 'videos' or "
                f"'playlists' tab"
            )
        caveat = _INEXACT_TABS.get(reference.tab)
        if caveat:
            logger.warning("%s -- %s", source, caveat)
        channel_id = resolve_channel_id(reference, api_key)
        if strategy == "playlists":
            return fetch_channel_playlists(channel_id, api_key)
        return fetch_channel(channel_id, api_key)

    raise CatalogError(f"{source!r}: unsupported reference kind {reference.kind!r}")


def _sort_key(record: VideoRecord) -> tuple:
    """
    Stable, total ordering key for catalog serialization.

    Parameters
    ----------
    record : VideoRecord
        The record to key.

    Returns
    -------
    tuple
        A tuple ordering by playlist, then position, then publication date,
        then id. The trailing id makes the ordering total, so two runs over
        the same upstream data always serialize in the same order and an
        unchanged sync produces an empty diff.
    """
    return (
        record.playlist_id or record.playlist,
        record.position if record.position is not None else 1 << 30,
        record.published.isoformat() if record.published else "",
        record.id,
    )


def merge_catalog_enrichment(
    refreshed: Sequence[VideoRecord],
    existing: Sequence[VideoRecord],
    *,
    preserve_unmatched: bool = False,
    preserve_missing_provider: bool = False,
) -> list[VideoRecord]:
    """
    Preserve author-owned enrichment while refreshing provider metadata.

    YouTube owns volatile facts such as title, description, publication time,
    playlist position and channel id.  The documentation author owns the
    optional ``handle``, ``tags`` and ``fields`` enrichment used by
    ``youtube-gallery`` for offline channel projection and custom facets.

    Matching is by canonical video id.  For author-owned values, a non-empty
    value present in the refreshed record wins (useful to callers that enrich
    fetched data deliberately), otherwise the existing authored value is
    retained.  ``preserve_unmatched`` is used for partial RSS refreshes: those
    feeds expose only recent items, so records absent from the response are
    *unknown*, not confirmed deleted, and must not be pruned.
    ``preserve_missing_provider`` additionally retains previously known
    provider fields when the partial source cannot observe them (for example
    duration in RSS). It never overrides a non-empty refreshed value.
    """
    from dataclasses import replace  # ruff: ignore[import-outside-top-level]

    by_id = {record.id: record for record in existing}
    merged: list[VideoRecord] = []
    refreshed_ids: set[str] = set()
    for record in refreshed:
        refreshed_ids.add(record.id)
        previous = by_id.get(record.id)
        if previous is None:
            merged.append(record)
            continue
        provider = {}
        if preserve_missing_provider:
            provider = {
                "title": (
                    previous.title
                    if record.title == record.id and previous.title
                    else record.title
                ),
                "description": record.description or previous.description,
                "channel": record.channel or previous.channel,
                "channel_id": record.channel_id or previous.channel_id,
                "playlist": record.playlist or previous.playlist,
                "playlist_id": record.playlist_id or previous.playlist_id,
                "position": (
                    record.position
                    if record.position is not None
                    else previous.position
                ),
                "published": record.published or previous.published,
                "duration": (
                    record.duration
                    if record.duration is not None
                    else previous.duration
                ),
            }
        merged.append(
            replace(
                record,
                **provider,
                handle=record.handle or previous.handle,
                tags=list(record.tags) if record.tags else list(previous.tags),
                fields=dict(record.fields) if record.fields else dict(previous.fields),
            )
        )
    if preserve_unmatched:
        merged.extend(record for record in existing if record.id not in refreshed_ids)
    return merged


def _catalog_payload(records: Sequence[VideoRecord]) -> dict[str, Any]:
    """Return the stable, human-reviewable YAML payload for ``records``."""
    return {
        "videos": [
            {
                key: value
                for key, value in (
                    ("id", record.id),
                    ("title", record.title),
                    ("description", record.description),
                    ("channel", record.channel),
                    ("channel_id", record.channel_id),
                    ("handle", record.handle),
                    ("playlist", record.playlist),
                    ("playlist_id", record.playlist_id),
                    ("position", record.position),
                    (
                        "published",
                        record.published.isoformat() if record.published else None,
                    ),
                    ("duration", record.duration),
                    ("tags", record.tags or None),
                    ("fields", record.fields or None),
                )
                # Omitting empty fields keeps the committed file readable and
                # its diffs small without discarding authored enrichment.
                if value not in (None, "", [], {})
            }
            for record in sorted(records, key=_sort_key)
        ]
    }


def render_catalog(records: Sequence[VideoRecord]) -> str:
    """Serialize a normalized video catalog deterministically as YAML."""
    from yaml import safe_dump  # ruff: ignore[import-outside-top-level]

    return safe_dump(
        _catalog_payload(records),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def write_catalog(records: Sequence[VideoRecord], path: Path) -> bool:
    """
    Serialize records idempotently and replace the destination atomically.

    Returns ``True`` only when the exact normalized YAML changed.  The
    temporary file is created beside the destination so ``os.replace`` remains
    atomic on normal local filesystems and a failed write cannot leave a
    truncated catalog behind.
    """
    text = render_catalog(records)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_name = handle.name
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            try:  # ruff: ignore[suppressible-exception]
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
    return True


def main(  # ruff: ignore[too-many-branches, too-many-return-statements]
    argv: Sequence[str] | None = None,
) -> int:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a handled acquisition or validation
        failure.
    """
    parser = argparse.ArgumentParser(
        prog="youtube-catalog-sync",
        description=(
            "Fetch YouTube metadata into a catalog file for the docs build. "
            "Set YOUTUBE_API_KEY for provider metadata/full collection history; "
            "without it, channels/playlists use recent public RSS and exact videos "
            "remain ID-only."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="URL",
        help=(
            "any YouTube URL or id: a video, playlist, channel, handle "
            "(@name) or channel tab. Repeatable. This is the recommended "
            "form -- paste the link from the address bar."
        ),
    )
    parser.add_argument(
        "--playlist",
        action="append",
        default=[],
        metavar="PL...",
        help="playlist id to fetch; repeatable",
    )
    parser.add_argument(
        "--channel",
        action="append",
        default=[],
        metavar="UC...",
        help="channel id to fetch; repeatable",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="catalog file to write",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the catalog would change; for CI drift detection",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help=(
            "remove stored videos not returned by this refresh; requires "
            "YOUTUBE_API_KEY because public RSS is only a partial recent feed"
        ),
    )
    args = parser.parse_args(argv)

    if not (args.source or args.playlist or args.channel):
        parser.error("give at least one --source, --playlist or --channel")

    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if args.prune and not api_key:
        logger.error(
            "error: --prune requires YOUTUBE_API_KEY; public RSS is a partial "
            "recent feed and cannot prove that older videos were removed"
        )
        return 1
    if not api_key:
        logger.warning(
            "note: YOUTUBE_API_KEY is unset; channels/playlists use public RSS "
            "(recent items only), while exact videos remain ID-only"
        )

    raw: list[dict[str, Any]] = []
    try:
        for source in args.source:
            raw.extend(resolve_source(source, api_key))
        for playlist_id in args.playlist:
            try:
                playlist_id = validate_playlist_id(  # ruff: ignore[redefined-loop-name]
                    playlist_id,
                )
            except ReferenceError as exc:
                raise CatalogError(f"--playlist: {exc}") from exc
            raw.extend(fetch_playlist(playlist_id, api_key))
        for channel_id in args.channel:
            try:
                channel_id = validate_channel_id(  # ruff: ignore[redefined-loop-name]
                    channel_id,
                )
            except ReferenceError as exc:
                raise CatalogError(f"--channel: {exc}") from exc
            raw.extend(fetch_channel(channel_id, api_key))
        records = normalize_catalog(raw, "youtube sync")
    except CatalogError as exc:
        logger.error("error: %s", exc)
        return 1

    existing: list[VideoRecord] = []
    try:
        if args.output.exists():
            text = read_bounded_utf8(args.output, str(args.output))
            existing = normalize_catalog(
                load_bounded_yaml(text, str(args.output)), str(args.output)
            )
    except (BoundedYAMLError, CatalogError) as exc:
        logger.error("error: %s", exc)
        return 1

    # Public RSS feeds are intentionally partial (recent items only). Absence
    # from such a response is not evidence that an older catalog item should be
    # deleted. Even a keyed refresh preserves unmatched authored history unless
    # the caller explicitly opts into authoritative deletion with ``--prune``.
    records = merge_catalog_enrichment(
        records,
        existing,
        preserve_unmatched=not args.prune,
        preserve_missing_provider=not bool(api_key),
    )

    if args.check:
        expected = render_catalog(records)
        try:
            actual = (
                args.output.read_text(encoding="utf-8") if args.output.exists() else ""
            )
        except (OSError, UnicodeError) as exc:
            logger.error("error: could not read %s: %s", args.output, exc)
            return 1
        if actual != expected:
            logger.error(
                f"error: {args.output} is out of date "
                f"({len(existing)} stored, {len(records)} refreshed); "
                "normalized catalog content would change"
            )
            return 1
        logger.info("%s is up to date (%d videos)", args.output, len(records))
        return 0

    try:
        changed = write_catalog(records, args.output)
    except (OSError, UnicodeError) as exc:
        logger.error("error: could not write %s: %s", args.output, exc)
        return 1
    verb = "updated" if changed else "unchanged"
    logger.info("%s %s (%d videos)", args.output, verb, len(records))
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Behave like a normal Unix CLI when stdout is piped to a consumer that
    # exits early (for example ``... | head``): terminate on SIGPIPE instead
    # of printing a Python BrokenPipeError traceback. Windows simply lacks
    # SIGPIPE, so this is a no-op there.
    try:
        import signal

        if hasattr(signal, "SIGPIPE"):
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, OSError, ValueError):
        pass
    raise SystemExit(main())
