"""
Tests for :mod:`.sync`: the offline catalog acquisition tool.

``sync`` is the only module that talks to YouTube, so every test here runs
against a fake ``requests`` transport injected through
``sync._require_requests``; real sockets are disabled for the whole module.
Covered: API paging and its failure modes, RSS parsing, the fetchers, source
resolution, enrichment merging, deterministic serialisation, atomic catalog
writes in ``tmp_path`` and the command-line entry point.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import socket

import pytest
import yaml

from .. import sync
from ..model import CatalogError, VideoRecord, normalize_catalog, normalize_record
from ..reference import parse_reference

UTC = dt.timezone.utc
A, B, C, D = (letter * 11 for letter in "abcd")
UC_A = "UC" + "a" * 22
UC_B = "UC" + "b" * 22
PL_A = "PL" + "a" * 16
PL_B = "PL" + "b" * 16
UPLOADS = "UU" + "a" * 22
SECRET = "sk-SECRET-KEY"

#: The real lazy importer, captured before the autouse fixture replaces it.
REAL_REQUIRE_REQUESTS = sync._require_requests


# -- fake transport -----------------------------------------------------------


class FakeResponse:
    """A minimal stand-in for ``requests.Response``."""

    def __init__(self, payload=None, status=200, content=b"", json_error=None):
        self._payload = payload
        self.status_code = status
        self.content = content
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            # Real libraries put the full URL, key included, in this text.
            raise RuntimeError(f"{self.status_code} for url ...?key={SECRET}")

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class FakeRequests:
    """A fake ``requests`` module that records calls and never opens a socket."""

    def __init__(self, handler):
        self.calls = []
        self._handler = handler

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params, timeout))
        return self._handler(url.rsplit("/", 1)[-1], params)

    def endpoints(self):
        return [url.rsplit("/", 1)[-1] for url, _, _ in self.calls]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that reaches for a real socket or the real transport."""

    def refuse(*args, **kwargs):
        raise AssertionError("a sync test tried to open a network socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(sync, "_require_requests", refuse)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)


@pytest.fixture
def transport(monkeypatch):
    """Install a fake transport; call the fixture with a handler function."""

    def install(handler):
        fake = FakeRequests(handler)
        monkeypatch.setattr(sync, "_require_requests", lambda: fake)
        return fake

    return install


def pages(*payloads):
    """Return a handler serving the given payloads in order, then failing."""
    remaining = list(payloads)

    def handler(endpoint, params):
        return FakeResponse(remaining.pop(0))

    return handler


def video(video_id=A, **keys):
    """Return a normalized video record."""
    return normalize_record({"id": video_id, **keys})


def rss(entries, title="Feed title", author="Author name"):
    """Build a YouTube-shaped Atom feed."""
    body = "".join(
        "<entry>"
        + (f"<yt:videoId>{entry['id']}</yt:videoId>" if entry.get("id") else "")
        + f"<yt:channelId>{entry.get('channel_id', UC_A)}</yt:channelId>"
        + f"<title>{entry.get('title', 'T')}</title>"
        + f"<published>{entry.get('published', '2024-01-01T00:00:00+00:00')}</published>"
        + (
            "<media:group><media:description>"
            f"{entry['description']}</media:description></media:group>"
            if "description" in entry
            else ""
        )
        + "</entry>"
        for entry in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:yt="http://www.youtube.com/xml/schemas/2015" '
        'xmlns:media="http://search.yahoo.com/mrss/">'
        f"<title>{title}</title><author><name>{author}</name></author>{body}</feed>"
    ).encode("utf-8")


# -- _api_pages ---------------------------------------------------------------


class TestApiPages:
    def test_one_page_is_requested_with_key_page_size_and_timeout(self, transport):
        fake = transport(pages({"items": [1]}))
        assert list(sync._api_pages("videos", {"part": "snippet"}, "KEY")) == [
            {"items": [1]}
        ]
        assert fake.calls == [
            (
                "https://www.googleapis.com/youtube/v3/videos",
                {"part": "snippet", "key": "KEY", "maxResults": 50},
                sync.TIMEOUT,
            )
        ]

    def test_every_request_is_bounded_by_a_timeout(self):
        connect, read = sync.TIMEOUT
        assert connect > 0 and read > 0

    def test_paging_follows_the_server_token_until_it_stops(self, transport):
        fake = transport(
            pages(
                {"items": [1], "nextPageToken": "t1"},
                {"items": [2], "nextPageToken": "t2"},
                {"items": [3]},
            )
        )
        result = list(sync._api_pages("playlistItems", {"playlistId": PL_A}, "KEY"))
        assert [page["items"] for page in result] == [[1], [2], [3]]
        assert [params.get("pageToken") for _, params, _ in fake.calls] == [
            None,
            "t1",
            "t2",
        ]

    def test_an_empty_token_ends_paging(self, transport):
        fake = transport(pages({"items": [], "nextPageToken": ""}))
        assert len(list(sync._api_pages("videos", {}, "KEY"))) == 1
        assert len(fake.calls) == 1

    def test_the_caller_params_are_not_mutated(self, transport):
        transport(pages({"items": [], "nextPageToken": "t"}, {"items": []}))
        params = {"part": "snippet"}
        list(sync._api_pages("videos", params, "KEY"))
        assert params == {"part": "snippet"}

    def test_a_repeated_token_is_refused_rather_than_looped(self, transport):
        fake = transport(lambda endpoint, params: FakeResponse({"nextPageToken": "same"}))
        with pytest.raises(CatalogError, match="repeated page token"):
            list(sync._api_pages("videos", {}, "KEY"))
        assert len(fake.calls) == 2

    def test_paging_is_bounded(self, transport):
        counter = iter(range(10**6))
        fake = transport(
            lambda endpoint, params: FakeResponse({"nextPageToken": f"t{next(counter)}"})
        )
        with pytest.raises(CatalogError, match=f"exceeded {sync.MAX_PAGES} pages"):
            list(sync._api_pages("videos", {}, "KEY"))
        assert len(fake.calls) == sync.MAX_PAGES

    def test_a_transport_failure_never_leaks_the_api_key(self, transport):
        def explode(endpoint, params):
            raise OSError(f"connection refused for https://x/?key={params['key']}")

        transport(explode)
        with pytest.raises(CatalogError) as caught:
            list(sync._api_pages("videos", {}, SECRET))
        assert str(caught.value) == "YouTube API request to videos failed (OSError)"
        assert SECRET not in str(caught.value)

    @pytest.mark.parametrize("status", [400, 403, 404, 500])
    def test_an_http_error_reports_the_status_and_never_the_key(self, transport, status):
        transport(lambda endpoint, params: FakeResponse({}, status=status))
        with pytest.raises(CatalogError) as caught:
            list(sync._api_pages("videos", {}, SECRET))
        assert str(caught.value) == f"YouTube API request to videos failed (HTTP {status})"

    def test_invalid_json_is_a_catalog_error(self, transport):
        transport(lambda endpoint, params: FakeResponse(json_error=ValueError(SECRET)))
        with pytest.raises(CatalogError) as caught:
            list(sync._api_pages("videos", {}, SECRET))
        assert str(caught.value) == "YouTube API response from videos was not valid JSON"

    @pytest.mark.parametrize(
        "payload, fragment",
        [
            ([1], "was a list, expected an object"),
            ("text", "was a str, expected an object"),
            (None, "was a NoneType, expected an object"),
            ({"items": {}}, "has non-list 'items'"),
            ({"items": "x"}, "has non-list 'items'"),
            ({"items": [], "nextPageToken": 5}, "non-string nextPageToken"),
            ({"items": [], "nextPageToken": ["t"]}, "non-string nextPageToken"),
        ],
        ids=["list", "string", "null", "items-mapping", "items-string", "token-int",
             "token-list"],
    )
    def test_a_malformed_response_shape_is_a_catalog_error(
        self, transport, payload, fragment
    ):
        transport(pages(payload))
        with pytest.raises(CatalogError, match=fragment):
            list(sync._api_pages("videos", {}, "KEY"))


# -- RSS ----------------------------------------------------------------------


class TestRssItems:
    def test_a_playlist_feed_becomes_raw_records(self, transport):
        fake = transport(
            lambda endpoint, params: FakeResponse(
                content=rss(
                    [{"id": A, "title": "One", "description": "about"}, {"id": B}],
                    title="My playlist",
                )
            )
        )
        records = sync._rss_items("playlist_id", PL_A)
        assert fake.calls == [(sync.RSS_BASE, {"playlist_id": PL_A}, sync.TIMEOUT)]
        assert records == [
            {
                "id": A,
                "title": "One",
                "description": "about",
                "channel": "Author name",
                "channel_id": UC_A,
                "published": "2024-01-01T00:00:00+00:00",
                "playlist": "My playlist",
                "playlist_id": PL_A,
            },
            {
                "id": B,
                "title": "T",
                "description": "",
                "channel": "Author name",
                "channel_id": UC_A,
                "published": "2024-01-01T00:00:00+00:00",
                "playlist": "My playlist",
                "playlist_id": PL_A,
            },
        ]
        # RSS carries no duration; it must be absent rather than guessed.
        assert all("duration" not in record for record in records)

    def test_a_channel_feed_carries_no_playlist(self, transport):
        transport(lambda endpoint, params: FakeResponse(content=rss([{"id": A}])))
        (record,) = sync._rss_items("channel_id", UC_A)
        assert "playlist" not in record and "playlist_id" not in record
        assert record["channel"] == "Author name"

    def test_a_channel_feed_without_an_author_uses_the_feed_title(self, transport):
        transport(
            lambda endpoint, params: FakeResponse(
                content=rss([{"id": A}], title="Channel title", author="")
            )
        )
        assert sync._rss_items("channel_id", UC_A)[0]["channel"] == "Channel title"

    def test_entries_without_a_video_id_are_skipped(self, transport):
        transport(
            lambda endpoint, params: FakeResponse(content=rss([{"id": ""}, {"id": A}]))
        )
        assert [record["id"] for record in sync._rss_items("channel_id", UC_A)] == [A]

    def test_an_empty_feed_is_an_empty_list(self, transport):
        transport(lambda endpoint, params: FakeResponse(content=rss([])))
        assert sync._rss_items("channel_id", UC_A) == []

    def test_feed_records_normalize_into_a_catalog(self, transport):
        transport(
            lambda endpoint, params: FakeResponse(
                content=rss([{"id": A, "title": "One &amp; &lt;b&gt;"}])
            )
        )
        (record,) = normalize_catalog(sync._rss_items("playlist_id", PL_A))
        assert record.title == "One & <b>"
        assert record.published == dt.datetime(2024, 1, 1, tzinfo=UTC)
        assert record.playlist_id == PL_A

    @pytest.mark.parametrize(
        "content",
        [
            b"<notxml",
            b"",
            b'<?xml version="1.0"?><!DOCTYPE f [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
            b'<feed xmlns="http://www.w3.org/2005/Atom"><title>&x;</title></feed>',
            b'<?xml version="1.0"?><!DOCTYPE f [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;">]>'
            b'<feed xmlns="http://www.w3.org/2005/Atom"><title>&b;</title></feed>',
        ],
        ids=["malformed", "empty", "external-entity", "entity-expansion"],
    )
    def test_hostile_or_broken_xml_is_a_catalog_error(self, transport, content):
        transport(lambda endpoint, params: FakeResponse(content=content))
        with pytest.raises(CatalogError, match=f"RSS fetch for {PL_A} failed"):
            sync._rss_items("playlist_id", PL_A)

    def test_an_http_error_is_a_catalog_error(self, transport):
        transport(lambda endpoint, params: FakeResponse(status=404))
        with pytest.raises(CatalogError, match="RSS fetch"):
            sync._rss_items("channel_id", UC_A)

    def test_a_transport_failure_is_a_catalog_error(self, transport):
        def explode(endpoint, params):
            raise OSError("no route")

        transport(explode)
        with pytest.raises(CatalogError, match="RSS fetch"):
            sync._rss_items("channel_id", UC_A)


# -- fetchers -----------------------------------------------------------------


def api(routes):
    """Return a handler that dispatches on the API endpoint name."""

    def handler(endpoint, params):
        return FakeResponse(routes[endpoint](params))

    return handler


def playlist_item(video_id, position, **snippet):
    """Build one ``playlistItems`` API item."""
    return {
        "snippet": {
            "title": f"title {video_id[0]}",
            "description": "desc",
            "channelTitle": "Curator",
            "videoOwnerChannelTitle": "Owner",
            "videoOwnerChannelId": UC_A,
            "position": position,
            "publishedAt": "2020-01-01T00:00:00Z",
            "resourceId": {"kind": "youtube#video", "videoId": video_id},
            **snippet,
        },
        "contentDetails": {"videoPublishedAt": "2019-01-01T00:00:00Z"},
    }


class TestFetchVideo:
    def test_without_a_key_the_id_is_returned_without_any_request(self):
        # The autouse fixture makes any transport access fail.
        assert sync.fetch_video(A) == [{"id": A}]

    def test_with_a_key_the_metadata_is_fetched(self, transport):
        fake = transport(
            pages(
                {
                    "items": [
                        {
                            "id": A,
                            "snippet": {
                                "title": "T",
                                "description": "d",
                                "channelTitle": "C",
                                "channelId": UC_A,
                                "publishedAt": "2024-01-01T00:00:00Z",
                            },
                            "contentDetails": {"duration": "PT1M"},
                        }
                    ]
                }
            )
        )
        assert sync.fetch_video(A, "KEY") == [
            {
                "id": A,
                "title": "T",
                "description": "d",
                "channel": "C",
                "channel_id": UC_A,
                "published": "2024-01-01T00:00:00Z",
                "duration": "PT1M",
            }
        ]
        assert fake.calls[0][1]["id"] == A
        assert fake.calls[0][1]["part"] == "snippet,contentDetails"

    def test_a_sparse_item_still_yields_a_normalizable_record(self, transport):
        transport(pages({"items": [{}]}))
        (raw,) = sync.fetch_video(A, "KEY")
        assert normalize_record(raw).id == A

    def test_an_unknown_video_is_an_error(self, transport):
        transport(pages({"items": []}))
        with pytest.raises(CatalogError, match=f"video {A!r} was not found"):
            sync.fetch_video(A, "KEY")


class TestFetchPlaylist:
    def test_without_a_key_the_rss_feed_is_used(self, transport):
        fake = transport(lambda endpoint, params: FakeResponse(content=rss([{"id": A}])))
        records = sync.fetch_playlist(PL_A)
        assert [record["id"] for record in records] == [A]
        assert fake.calls[0][0] == sync.RSS_BASE

    def test_with_a_key_items_titles_and_durations_are_combined(self, transport):
        fake = transport(
            api(
                {
                    "playlists": lambda params: {
                        "items": [{"id": PL_A, "snippet": {"title": " Intro "}}]
                    },
                    "playlistItems": lambda params: {
                        "items": [
                            playlist_item(A, 0),
                            {"snippet": {"resourceId": {"kind": "youtube#channel"}}},
                            playlist_item(B, 1, videoOwnerChannelTitle=""),
                        ]
                    },
                    "videos": lambda params: {
                        "items": [{"id": A, "contentDetails": {"duration": "PT2M"}}]
                    },
                }
            )
        )
        records = sync.fetch_playlist(PL_A, "KEY")
        assert fake.endpoints() == ["playlists", "playlistItems", "videos"]
        assert records == [
            {
                "id": A,
                "title": "title a",
                "description": "desc",
                "channel": "Owner",
                "channel_id": UC_A,
                "playlist": "Intro",
                "playlist_id": PL_A,
                "position": 0,
                "published": "2019-01-01T00:00:00Z",
                "duration": "PT2M",
            },
            {
                "id": B,
                "title": "title b",
                "description": "desc",
                # Falls back to the playlist owner when the uploader is unknown.
                "channel": "Curator",
                "channel_id": UC_A,
                "playlist": "Intro",
                "playlist_id": PL_A,
                "position": 1,
                "published": "2019-01-01T00:00:00Z",
            },
        ]
        assert fake.calls[2][1]["id"] == f"{A},{B}"
        assert [r.id for r in normalize_catalog(records)] == [A, B]

    def test_the_upload_date_falls_back_to_the_playlist_insertion_date(self, transport):
        item = playlist_item(A, 0)
        item["contentDetails"] = {}
        transport(
            api(
                {
                    "playlistItems": lambda params: {"items": [item]},
                    "videos": lambda params: {"items": []},
                }
            )
        )
        (record,) = sync.fetch_playlist(PL_A, "KEY", playlist_title="Given")
        assert record["published"] == "2020-01-01T00:00:00Z"
        assert record["playlist"] == "Given"

    def test_a_supplied_title_skips_the_title_lookup(self, transport):
        fake = transport(
            api(
                {
                    "playlistItems": lambda params: {"items": []},
                    "videos": lambda params: {"items": []},
                }
            )
        )
        assert sync.fetch_playlist(PL_A, "KEY", playlist_title="") == []
        assert "playlists" not in fake.endpoints()

    def test_an_inaccessible_playlist_is_an_error(self, transport):
        transport(api({"playlists": lambda params: {"items": []}}))
        with pytest.raises(CatalogError, match="not found or is not accessible"):
            sync.fetch_playlist(PL_A, "KEY")

    def test_a_title_for_a_different_playlist_is_not_trusted(self, transport):
        transport(
            api({"playlists": lambda params: {"items": [{"id": PL_B, "snippet": {}}]}})
        )
        assert sync._playlist_title(PL_A, "KEY") is None

    def test_a_playlist_over_the_catalog_limit_is_refused(self, transport, monkeypatch):
        monkeypatch.setattr(sync, "MAX_COLLECTION_ITEMS", 2)
        transport(
            api(
                {
                    "playlistItems": lambda params: {
                        "items": [playlist_item(v, i) for i, v in enumerate((A, B, C))]
                    }
                }
            )
        )
        with pytest.raises(CatalogError, match="exceeds the catalog limit of 2"):
            sync.fetch_playlist(PL_A, "KEY", playlist_title="x")


class TestVideoDurations:
    def test_ids_are_deduplicated_and_batched_by_fifty(self, transport):
        ids = [f"{index:011d}" for index in range(120)]
        fake = transport(
            lambda endpoint, params: FakeResponse(
                {
                    "items": [
                        {"id": video_id, "contentDetails": {"duration": "PT1S"}}
                        for video_id in params["id"].split(",")
                    ]
                }
            )
        )
        result = sync._video_durations([*ids, "", ids[0], ids[0]], "KEY")
        assert result == dict.fromkeys(ids, "PT1S")
        assert [len(params["id"].split(",")) for _, params, _ in fake.calls] == [
            50,
            50,
            20,
        ]

    def test_no_ids_means_no_request(self):
        assert sync._video_durations([], "KEY") == {}
        assert sync._video_durations(["", ""], "KEY") == {}

    def test_items_without_an_id_or_duration_are_ignored(self, transport):
        transport(
            pages(
                {
                    "items": [
                        {"id": A},
                        {"contentDetails": {"duration": "PT1S"}},
                        {"id": B, "contentDetails": {"duration": ""}},
                        {"id": C, "contentDetails": {"duration": "PT3S"}},
                    ]
                }
            )
        )
        assert sync._video_durations([A, B, C], "KEY") == {C: "PT3S"}


class TestFetchChannel:
    def test_without_a_key_the_rss_feed_is_used(self, transport):
        fake = transport(lambda endpoint, params: FakeResponse(content=rss([{"id": A}])))
        assert [record["id"] for record in sync.fetch_channel(UC_A)] == [A]
        assert fake.calls == [(sync.RSS_BASE, {"channel_id": UC_A}, sync.TIMEOUT)]

    def test_with_a_key_the_uploads_playlist_is_hidden_from_the_records(self, transport):
        fake = transport(
            api(
                {
                    "channels": lambda params: {
                        "items": [
                            {"contentDetails": {"relatedPlaylists": {"uploads": UPLOADS}}}
                        ]
                    },
                    "playlistItems": lambda params: {"items": [playlist_item(A, 7)]},
                    "videos": lambda params: {"items": []},
                }
            )
        )
        (record,) = sync.fetch_channel(UC_A, "KEY")
        assert fake.endpoints() == ["channels", "playlistItems", "videos"]
        assert fake.calls[1][1]["playlistId"] == UPLOADS
        assert (record["playlist"], record["playlist_id"], record["position"]) == (
            "",
            "",
            None,
        )
        assert normalize_record(record).playlist_id == ""

    @pytest.mark.parametrize(
        "payload",
        [{"items": []}, {"items": [{}]}, {"items": [{"contentDetails": {}}]}, {}],
        ids=["no-items", "empty-item", "no-related", "no-items-key"],
    )
    def test_a_channel_without_uploads_is_an_error(self, transport, payload):
        transport(api({"channels": lambda params: payload}))
        with pytest.raises(CatalogError, match="has no uploads playlist"):
            sync.fetch_channel(UC_A, "KEY")


class TestResolveChannelId:
    def test_a_canonical_id_needs_no_lookup(self):
        assert sync.resolve_channel_id(parse_reference(UC_A)) == UC_A
        assert sync.resolve_channel_id(parse_reference(UC_A), "KEY") == UC_A

    @pytest.mark.parametrize(
        "reference", ["@foo", "https://www.youtube.com/c/Name"], ids=["handle", "vanity"]
    )
    def test_a_handle_or_name_needs_an_api_key(self, reference):
        with pytest.raises(CatalogError) as caught:
            sync.resolve_channel_id(parse_reference(reference))
        message = str(caught.value)
        assert "without an API key" in message
        assert "YOUTUBE_API_KEY" in message

    def test_a_handle_is_resolved_through_for_handle(self, transport):
        fake = transport(pages({"items": [{"id": UC_A}]}))
        assert sync.resolve_channel_id(parse_reference("@foo"), "KEY") == UC_A
        assert fake.calls[0][1]["forHandle"] == "@foo"
        assert fake.calls[0][1]["part"] == "id"

    def test_a_vanity_name_tries_handle_then_legacy_username(self, transport):
        fake = transport(pages({"items": []}, {"items": [{"id": UC_B}]}))
        reference = parse_reference("https://www.youtube.com/c/Name")
        assert sync.resolve_channel_id(reference, "KEY") == UC_B
        assert [
            {key: params[key] for key in ("forHandle", "forUsername") if key in params}
            for _, params, _ in fake.calls
        ] == [{"forHandle": "@Name"}, {"forUsername": "Name"}]

    def test_an_unknown_channel_is_an_error(self, transport):
        transport(lambda endpoint, params: FakeResponse({"items": [{}]}))
        with pytest.raises(CatalogError, match="channel 'foo' not found"):
            sync.resolve_channel_id(parse_reference("@foo"), "KEY")


class TestFetchChannelPlaylists:
    def test_an_api_key_is_required(self):
        with pytest.raises(CatalogError, match="requires YOUTUBE_API_KEY"):
            sync.fetch_channel_playlists(UC_A, "")

    def test_playlists_are_walked_in_id_order_with_one_record_per_video(self, transport):
        members = {PL_A: [A, B], PL_B: [B, C]}
        fake = transport(
            api(
                {
                    # Deliberately returned in reverse id order, with a blank id.
                    "playlists": lambda params: {
                        "items": [
                            {"id": PL_B, "snippet": {"title": "Second"}},
                            {"id": "", "snippet": {"title": "ghost"}},
                            {"id": PL_A, "snippet": {"title": " First "}},
                        ]
                    },
                    "playlistItems": lambda params: {
                        "items": [
                            playlist_item(video_id, index)
                            for index, video_id in enumerate(members[params["playlistId"]])
                        ]
                    },
                    "videos": lambda params: {"items": []},
                }
            )
        )
        records = sync.fetch_channel_playlists(UC_A, "KEY")
        assert fake.calls[0][1]["channelId"] == UC_A
        assert [(record["id"], record["playlist"]) for record in records] == [
            (A, "First"),
            (B, "First"),
            (C, "Second"),
        ]

    def test_the_result_does_not_depend_on_provider_ordering(self, transport):
        members = {PL_A: [A, B], PL_B: [B, C]}

        def run(order):
            transport(
                api(
                    {
                        "playlists": lambda params: {
                            "items": [{"id": pid, "snippet": {"title": pid}} for pid in order]
                        },
                        "playlistItems": lambda params: {
                            "items": [
                                playlist_item(video_id, index)
                                for index, video_id in enumerate(
                                    members[params["playlistId"]]
                                )
                            ]
                        },
                        "videos": lambda params: {"items": []},
                    }
                )
            )
            return sync.fetch_channel_playlists(UC_A, "KEY")

        assert run([PL_A, PL_B]) == run([PL_B, PL_A])

    def test_the_unique_video_limit_is_enforced(self, transport, monkeypatch):
        monkeypatch.setattr(sync, "MAX_COLLECTION_ITEMS", 1)
        transport(
            api(
                {
                    "playlists": lambda params: {
                        "items": [{"id": PL_A, "snippet": {"title": "x"}},
                                  {"id": PL_B, "snippet": {"title": "y"}}]
                    },
                    "playlistItems": lambda params: {
                        "items": [playlist_item(A if params["playlistId"] == PL_A else B, 0)]
                    },
                    "videos": lambda params: {"items": []},
                }
            )
        )
        with pytest.raises(CatalogError, match="exceeds the catalog limit"):
            sync.fetch_channel_playlists(UC_A, "KEY")


# -- resolve_source -----------------------------------------------------------


class TestResolveSource:
    @pytest.mark.parametrize(
        "source",
        [A, f"https://youtu.be/{A}", f"https://www.youtube.com/watch?v={A}&t=30",
         f"https://www.youtube.com/watch?v={A}&list={PL_A}&index=3",
         f"https://www.youtube.com/shorts/{A}"],
        ids=["id", "short-host", "watch", "watch-in-playlist", "shorts"],
    )
    def test_a_video_reference_is_an_exact_video(self, source):
        # No transport is installed: a keyless video needs no request, and a
        # playlist context must not widen the acquisition.
        assert sync.resolve_source(source) == [{"id": A}]

    @pytest.mark.xfail(
        strict=True,
        reason="a watch URL opened from a Mix is rejected as 'ephemeral' "
        "although it names one stable video",
    )
    def test_a_video_watched_from_a_mix_is_still_an_exact_video(self):
        # Docstring: "A watch?v=…&list=… URL remains an exact video here ...
        # The retained playlist_id is context, not an instruction".
        source = f"https://www.youtube.com/watch?v={A}&list=RD{A}"
        assert sync.resolve_source(source) == [{"id": A}]

    def test_a_playlist_reference_fetches_the_playlist(self, transport):
        fake = transport(lambda endpoint, params: FakeResponse(content=rss([{"id": A}])))
        records = sync.resolve_source(f"https://www.youtube.com/playlist?list={PL_A}")
        assert [record["id"] for record in records] == [A]
        assert fake.calls[0][1] == {"playlist_id": PL_A}

    @pytest.mark.parametrize(
        "source",
        [UC_A, f"https://www.youtube.com/channel/{UC_A}",
         f"https://www.youtube.com/channel/{UC_A}/videos"],
        ids=["id", "root", "videos-tab"],
    )
    def test_a_channel_reference_fetches_its_uploads(self, transport, source):
        fake = transport(lambda endpoint, params: FakeResponse(content=rss([{"id": A}])))
        assert [record["id"] for record in sync.resolve_source(source)] == [A]
        assert fake.calls[0][1] == {"channel_id": UC_A}

    @pytest.mark.parametrize("tab", ["shorts", "streams", "live"])
    def test_an_inexact_upload_tab_warns_and_returns_all_uploads(
        self, transport, caplog, tab
    ):
        transport(lambda endpoint, params: FakeResponse(content=rss([{"id": A}])))
        source = f"https://www.youtube.com/channel/{UC_A}/{tab}"
        with caplog.at_level(logging.WARNING, logger=sync.logger.name):
            records = sync.resolve_source(source)
        assert [record["id"] for record in records] == [A]
        (message,) = [r.getMessage() for r in caplog.records if r.name == sync.logger.name]
        assert message.startswith(source)
        assert "all uploads are returned" in message

    def test_an_exact_tab_does_not_warn(self, transport, caplog):
        transport(lambda endpoint, params: FakeResponse(content=rss([{"id": A}])))
        with caplog.at_level(logging.WARNING, logger=sync.logger.name):
            sync.resolve_source(f"https://www.youtube.com/channel/{UC_A}/videos")
        assert [r for r in caplog.records if r.name == sync.logger.name] == []

    def test_a_playlists_tab_needs_an_api_key(self):
        with pytest.raises(CatalogError, match="requires YOUTUBE_API_KEY"):
            sync.resolve_source(f"https://www.youtube.com/channel/{UC_A}/playlists")

    def test_a_handle_needs_an_api_key(self):
        with pytest.raises(CatalogError, match="without an API key"):
            sync.resolve_source("https://www.youtube.com/@foo")

    @pytest.mark.parametrize(
        "source, fragment",
        [
            ("https://www.youtube.com/results?search_query=x", "a search for 'x'"),
            ("https://www.youtube.com/hashtag/x", "the hashtag #x"),
            ("https://www.youtube.com/feed/subscriptions", "the 'subscriptions' feed"),
        ],
        ids=["search", "hashtag", "feed"],
    )
    def test_a_non_enumerable_reference_is_refused(self, source, fragment):
        with pytest.raises(CatalogError) as caught:
            sync.resolve_source(source)
        message = str(caught.value)
        assert fragment in message
        assert "not an enumerable collection of videos" in message

    @pytest.mark.parametrize(
        "source",
        [f"https://www.youtube.com/playlist?list=RD{A}",
         "https://www.youtube.com/playlist?list=WL",
         "https://www.youtube.com/playlist?list=LL"],
        ids=["mix", "watch-later", "liked"],
    )
    def test_a_per_viewer_playlist_is_refused(self, source):
        with pytest.raises(CatalogError, match="must not be committed to a catalog"):
            sync.resolve_source(source)

    @pytest.mark.parametrize("tab", ["community", "about", "featured"])
    def test_a_tab_without_videos_is_refused(self, tab):
        with pytest.raises(CatalogError, match=f"the {tab!r} tab holds no videos"):
            sync.resolve_source(f"https://www.youtube.com/channel/{UC_A}/{tab}")

    def test_an_unknown_tab_is_refused_rather_than_guessed(self):
        with pytest.raises(CatalogError, match="does not know how to enumerate"):
            sync.resolve_source(f"https://www.youtube.com/channel/{UC_A}/zzznew")

    @pytest.mark.parametrize(
        "source",
        ["https://evil.example/watch?v=" + A, "javascript:alert(1)", "", "not a ref",
         "../../etc/passwd"],
        ids=["foreign-host", "js-scheme", "empty", "garbage", "path"],
    )
    def test_an_unparsable_source_is_a_catalog_error(self, source):
        with pytest.raises(CatalogError):
            sync.resolve_source(source)

    def test_every_channel_tab_strategy_is_uploads_or_playlists(self):
        assert set(sync._TAB_STRATEGY.values()) == {"uploads", "playlists"}
        assert set(sync._INEXACT_TABS) <= set(sync._TAB_STRATEGY)


# -- merging ------------------------------------------------------------------


class TestMergeCatalogEnrichment:
    def test_new_records_pass_through(self):
        fresh = [video(A, title="New")]
        assert sync.merge_catalog_enrichment(fresh, []) == fresh

    def test_authored_enrichment_survives_a_provider_refresh(self):
        existing = [video(A, title="Old", handle="chan", tags=["mine"],
                          fields={"category": "ml"}, duration=10)]
        fresh = [video(A, title="New title", duration=99)]
        (merged,) = sync.merge_catalog_enrichment(fresh, existing)
        assert (merged.title, merged.duration) == ("New title", 99)
        assert merged.handle == "chan"
        assert merged.tags == ["mine"]
        assert merged.fields == {"category": "ml"}

    def test_non_empty_refreshed_enrichment_wins(self):
        existing = [video(A, handle="old", tags=["old"], fields={"a": 1})]
        fresh = [video(A, handle="new", tags=["new"], fields={"b": 2})]
        (merged,) = sync.merge_catalog_enrichment(fresh, existing)
        assert (merged.handle, merged.tags, merged.fields) == ("new", ["new"], {"b": 2})

    def test_provider_fields_are_replaced_by_default_even_when_now_empty(self):
        existing = [video(A, title="Old", description="old", duration=10, position=3,
                          playlist="P", published="2020-01-01")]
        (merged,) = sync.merge_catalog_enrichment([video(A)], existing)
        assert merged == video(A)

    def test_missing_provider_fields_can_be_preserved_for_partial_sources(self):
        existing = [
            video(A, title="Old", description="old", channel="Ch", channel_id=UC_A,
                  playlist="P", playlist_id=PL_A, position=0, published="2020-01-01",
                  duration=0)
        ]
        (merged,) = sync.merge_catalog_enrichment(
            [video(A)], existing, preserve_missing_provider=True
        )
        assert merged == existing[0]

    def test_preserving_never_overrides_a_refreshed_value(self):
        existing = [video(A, title="Old", description="old", duration=10, position=3)]
        fresh = [video(A, title="New", description="new", duration=0, position=0)]
        (merged,) = sync.merge_catalog_enrichment(
            fresh, existing, preserve_missing_provider=True
        )
        assert (merged.title, merged.description) == ("New", "new")
        assert (merged.duration, merged.position) == (0, 0)

    def test_unmatched_records_are_dropped_by_default(self):
        merged = sync.merge_catalog_enrichment([video(A)], [video(A), video(B)])
        assert [record.id for record in merged] == [A]

    def test_unmatched_records_can_be_kept_after_the_refreshed_ones(self):
        merged = sync.merge_catalog_enrichment(
            [video(C), video(A)], [video(B), video(A), video(D)], preserve_unmatched=True
        )
        assert [record.id for record in merged] == [C, A, B, D]

    def test_inputs_are_not_mutated_and_results_do_not_alias_them(self):
        existing = [video(A, tags=["mine"], fields={"k": "v"})]
        fresh = [video(A)]
        (merged,) = sync.merge_catalog_enrichment(fresh, existing)
        merged.tags.append("x")
        merged.fields["z"] = 1
        assert existing[0].tags == ["mine"] and existing[0].fields == {"k": "v"}
        assert fresh[0].tags == [] and fresh[0].fields == {}

    def test_nothing_refreshed_and_nothing_stored(self):
        assert sync.merge_catalog_enrichment([], []) == []
        assert sync.merge_catalog_enrichment([], [video(A)]) == []


# -- serialisation ------------------------------------------------------------


class TestRenderCatalog:
    def test_an_empty_catalog(self):
        assert yaml.safe_load(sync.render_catalog([])) == {"videos": []}

    def test_empty_fields_are_omitted_and_keys_keep_a_fixed_order(self):
        text = sync.render_catalog([video(A, title="T", position=0, duration=0)])
        (entry,) = yaml.safe_load(text)["videos"]
        assert list(entry) == ["id", "title", "position", "duration"]
        assert entry == {"id": A, "title": "T", "position": 0, "duration": 0}

    def test_every_field_is_written(self):
        record = video(
            A, title="T", description="d", channel="C", channel_id=UC_A, handle="h",
            playlist="P", playlist_id=PL_A, position=2, published="2024-03-01T10:00:00Z",
            duration=90, tags=["x"], fields={"category": "ml"},
        )
        (entry,) = yaml.safe_load(sync.render_catalog([record]))["videos"]
        assert entry == {
            "id": A, "title": "T", "description": "d", "channel": "C",
            "channel_id": UC_A, "handle": "h", "playlist": "P", "playlist_id": PL_A,
            "position": 2, "published": "2024-03-01T10:00:00+00:00", "duration": 90,
            "tags": ["x"], "fields": {"category": "ml"},
        }

    def test_the_output_normalizes_back_to_the_same_records(self):
        records = [
            video(A, title="Çok güzel: `x` <b> #1", description="multi\nline",
                  tags=["a", "b"], fields={"audience": {"level": "x"}, "n": 3},
                  published="2024-03-01T10:00:00+02:00", duration="1:02:30"),
            video(B, title="- leading dash", description="'quotes' \"both\""),
        ]
        reloaded = normalize_catalog(yaml.safe_load(sync.render_catalog(records)))
        assert sorted(reloaded, key=lambda r: r.id) == sorted(records, key=lambda r: r.id)

    def test_unicode_is_written_readably(self):
        assert "Çok güzel" in sync.render_catalog([video(A, title="Çok güzel")])

    def test_serialisation_is_independent_of_input_order(self):
        records = [
            video(A, playlist_id=PL_B, position=1),
            video(B, playlist_id=PL_A, position=5),
            video(C, playlist_id=PL_A, position=0),
            video(D, published="2020-01-01"),
        ]
        expected = sync.render_catalog(records)
        assert sync.render_catalog(records[::-1]) == expected
        assert sync.render_catalog([records[2], records[0], records[3], records[1]]) == (
            expected
        )

    def test_records_are_ordered_by_playlist_position_date_then_id(self):
        records = [
            video(D, playlist_id=PL_B, position=0),
            video(C, playlist_id=PL_A),
            video(B, playlist_id=PL_A, position=1),
            video(A, playlist_id=PL_A, position=0),
        ]
        ids = [entry["id"] for entry in yaml.safe_load(sync.render_catalog(records))["videos"]]
        assert ids == [A, B, C, D]

    def test_the_sort_key_is_total_through_the_trailing_id(self):
        first, second = video(A), video(B)
        assert sync._sort_key(first) < sync._sort_key(second)
        assert sync._sort_key(first)[-1] == A

    def test_rendering_does_not_mutate_the_records(self):
        record = video(A, tags=["x"], fields={"k": "v"})
        sync.render_catalog([record])
        assert record == video(A, tags=["x"], fields={"k": "v"})


class TestWriteCatalog:
    def test_a_new_file_is_written_and_reported_as_changed(self, tmp_path):
        target = tmp_path / "catalog.yaml"
        assert sync.write_catalog([video(A, title="T")], target) is True
        assert target.read_text(encoding="utf-8") == sync.render_catalog([video(A, title="T")])

    def test_missing_parent_directories_are_created(self, tmp_path):
        target = tmp_path / "deep" / "er" / "catalog.yaml"
        assert sync.write_catalog([video(A)], target) is True
        assert target.is_file()

    def test_an_unchanged_catalog_is_not_rewritten(self, tmp_path):
        target = tmp_path / "catalog.yaml"
        sync.write_catalog([video(A)], target)
        stamp = target.stat().st_mtime_ns
        os.utime(target, ns=(stamp - 10**9, stamp - 10**9))
        before = target.stat().st_mtime_ns
        assert sync.write_catalog([video(A)], target) is False
        assert target.stat().st_mtime_ns == before

    def test_changed_content_replaces_the_file(self, tmp_path):
        target = tmp_path / "catalog.yaml"
        sync.write_catalog([video(A)], target)
        assert sync.write_catalog([video(A), video(B)], target) is True
        assert [r.id for r in normalize_catalog(yaml.safe_load(target.read_text("utf-8")))] == [
            A,
            B,
        ]

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        target = tmp_path / "catalog.yaml"
        sync.write_catalog([video(A)], target)
        sync.write_catalog([video(B)], target)
        assert [path.name for path in tmp_path.iterdir()] == ["catalog.yaml"]

    def test_unicode_is_written_as_utf8_with_unix_newlines(self, tmp_path):
        target = tmp_path / "catalog.yaml"
        sync.write_catalog([video(A, title="Çok", description="a\nb")], target)
        raw = target.read_bytes()
        assert "Çok".encode("utf-8") in raw
        assert b"\r" not in raw

    def test_a_failed_replace_keeps_the_old_catalog_and_cleans_up(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "catalog.yaml"
        sync.write_catalog([video(A)], target)
        original = target.read_bytes()

        def fail(source, destination):
            raise OSError("disk full")

        monkeypatch.setattr(sync.os, "replace", fail)
        with pytest.raises(OSError, match="disk full"):
            sync.write_catalog([video(B)], target)
        assert target.read_bytes() == original
        assert [path.name for path in tmp_path.iterdir()] == ["catalog.yaml"]


# -- command line -------------------------------------------------------------


@pytest.fixture
def cli(caplog):
    """Run ``main`` and return ``(exit code, log messages from the tool)``."""

    def run(*argv):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=sync.logger.name):
            code = sync.main([str(arg) for arg in argv])
        messages = [r.getMessage() for r in caplog.records if r.name == sync.logger.name]
        return code, messages

    return run


class TestMain:
    def test_at_least_one_source_is_required(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as caught:
            sync.main(["--output", str(tmp_path / "c.yaml")])
        assert caught.value.code == 2
        assert "at least one --source" in capsys.readouterr().err

    def test_output_is_required(self, capsys):
        with pytest.raises(SystemExit) as caught:
            sync.main(["--source", A])
        assert caught.value.code == 2
        assert "--output" in capsys.readouterr().err

    def test_the_api_key_is_not_a_command_line_option(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            sync.main(["--source", A, "--output", str(tmp_path / "c.yaml"),
                       "--api-key", SECRET])
        assert "unrecognized arguments" in capsys.readouterr().err

    def test_a_keyless_video_source_writes_an_id_only_catalog(self, tmp_path, cli):
        target = tmp_path / "out" / "c.yaml"
        code, messages = cli("--source", f"https://youtu.be/{A}", "--source", B,
                             "--output", target)
        assert code == 0
        assert yaml.safe_load(target.read_text("utf-8")) == {
            "videos": [{"id": A, "title": A}, {"id": B, "title": B}]
        }
        assert any("YOUTUBE_API_KEY is unset" in message for message in messages)
        assert messages[-1] == f"{target} updated (2 videos)"

    def test_running_twice_is_idempotent(self, tmp_path, cli):
        target = tmp_path / "c.yaml"
        cli("--source", A, "--output", target)
        first = target.read_bytes()
        code, messages = cli("--source", A, "--output", target)
        assert code == 0
        assert target.read_bytes() == first
        assert messages[-1] == f"{target} unchanged (1 videos)"

    def test_a_keyless_playlist_uses_rss(self, tmp_path, cli, transport):
        fake = transport(
            lambda endpoint, params: FakeResponse(
                content=rss([{"id": A, "title": "One"}], title="Intro")
            )
        )
        target = tmp_path / "c.yaml"
        code, _ = cli("--playlist", PL_A, "--output", target)
        assert code == 0
        assert fake.calls == [(sync.RSS_BASE, {"playlist_id": PL_A}, sync.TIMEOUT)]
        (entry,) = yaml.safe_load(target.read_text("utf-8"))["videos"]
        assert (entry["title"], entry["playlist"], entry["playlist_id"]) == (
            "One",
            "Intro",
            PL_A,
        )

    def test_a_keyless_channel_uses_rss(self, tmp_path, cli, transport):
        fake = transport(lambda endpoint, params: FakeResponse(content=rss([{"id": A}])))
        code, _ = cli("--channel", UC_A, "--output", tmp_path / "c.yaml")
        assert code == 0
        assert fake.calls[0][1] == {"channel_id": UC_A}

    def test_the_api_key_comes_from_the_environment(
        self, tmp_path, cli, transport, monkeypatch
    ):
        monkeypatch.setenv("YOUTUBE_API_KEY", SECRET)
        fake = transport(
            pages({"items": [{"id": A, "snippet": {"title": "From API"}}]})
        )
        target = tmp_path / "c.yaml"
        code, messages = cli("--source", A, "--output", target)
        assert code == 0
        assert fake.calls[0][1]["key"] == SECRET
        assert yaml.safe_load(target.read_text("utf-8"))["videos"][0]["title"] == "From API"
        assert not any(SECRET in message for message in messages)
        assert not any("unset" in message for message in messages)

    @pytest.mark.parametrize(
        "flag, value, fragment",
        [
            ("--playlist", "??", "--playlist:"),
            ("--playlist", "../etc", "--playlist:"),
            ("--channel", "UCshort", "--channel:"),
            ("--channel", "@handle", "--channel:"),
            ("--source", "https://evil.example/x", "not a YouTube URL"),
            ("--source", "https://www.youtube.com/results?search_query=x", "search"),
        ],
        ids=["playlist-chars", "playlist-traversal", "channel-short", "channel-handle",
             "foreign-source", "search-source"],
    )
    def test_an_invalid_source_fails_cleanly_and_writes_nothing(
        self, tmp_path, cli, flag, value, fragment
    ):
        target = tmp_path / "c.yaml"
        code, messages = cli(flag, value, "--output", target)
        assert code == 1
        assert messages[-1].startswith("error: ")
        assert fragment in messages[-1]
        assert not target.exists()

    def test_prune_requires_an_api_key(self, tmp_path, cli):
        target = tmp_path / "c.yaml"
        code, messages = cli("--source", A, "--output", target, "--prune")
        assert code == 1
        assert "--prune requires YOUTUBE_API_KEY" in messages[-1]
        assert not target.exists()

    def test_a_refresh_keeps_stored_videos_and_authored_enrichment(self, tmp_path, cli):
        target = tmp_path / "c.yaml"
        sync.write_catalog(
            [
                video(A, title="Stored title", tags=["mine"], duration=42,
                      fields={"category": "ml"}),
                video(B, title="Older video"),
            ],
            target,
        )
        code, messages = cli("--source", A, "--output", target)
        assert code == 0
        entries = {e["id"]: e for e in yaml.safe_load(target.read_text("utf-8"))["videos"]}
        # A keyless video is ID-only, so every stored fact is preserved, and
        # the unmatched older video is unknown rather than deleted.
        assert entries[A] == {
            "id": A, "title": "Stored title", "duration": 42, "tags": ["mine"],
            "fields": {"category": "ml"},
        }
        assert entries[B] == {"id": B, "title": "Older video"}
        assert messages[-1] == f"{target} unchanged (2 videos)"

    def test_prune_with_a_key_removes_videos_absent_from_the_refresh(
        self, tmp_path, cli, transport, monkeypatch
    ):
        monkeypatch.setenv("YOUTUBE_API_KEY", SECRET)
        transport(lambda endpoint, params: FakeResponse(
            {"items": [{"id": A, "snippet": {"title": "Fresh"}}]}))
        target = tmp_path / "c.yaml"
        sync.write_catalog([video(A, title="Old", tags=["mine"]), video(B)], target)
        code, _ = cli("--source", A, "--output", target, "--prune")
        assert code == 0
        assert yaml.safe_load(target.read_text("utf-8")) == {
            "videos": [{"id": A, "title": "Fresh", "tags": ["mine"]}]
        }

    def test_without_prune_a_keyed_refresh_keeps_unmatched_videos(
        self, tmp_path, cli, transport, monkeypatch
    ):
        monkeypatch.setenv("YOUTUBE_API_KEY", SECRET)
        transport(lambda endpoint, params: FakeResponse(
            {"items": [{"id": A, "snippet": {"title": "Fresh"}}]}))
        target = tmp_path / "c.yaml"
        sync.write_catalog([video(A, title="Old"), video(B, title="Kept")], target)
        code, _ = cli("--source", A, "--output", target)
        assert code == 0
        titles = [e["title"] for e in yaml.safe_load(target.read_text("utf-8"))["videos"]]
        assert titles == ["Fresh", "Kept"]

    @pytest.mark.parametrize(
        "content, fragment",
        [
            (b"- id: [unclosed", "could not parse YAML"),
            (b"just text", "expected a list of records"),
            (b"- id: nope\n", "record 0"),
            ("- id: caf\xe9\n".encode("latin-1"), "UTF-8"),
        ],
        ids=["bad-yaml", "wrong-shape", "bad-record", "latin-1"],
    )
    def test_an_unreadable_existing_catalog_fails_and_is_left_untouched(
        self, tmp_path, cli, content, fragment
    ):
        target = tmp_path / "c.yaml"
        target.write_bytes(content)
        code, messages = cli("--source", A, "--output", target)
        assert code == 1
        assert fragment in messages[-1]
        assert target.read_bytes() == content

    def test_check_passes_on_an_up_to_date_catalog(self, tmp_path, cli):
        target = tmp_path / "c.yaml"
        cli("--source", A, "--output", target)
        before = target.read_bytes()
        code, messages = cli("--source", A, "--output", target, "--check")
        assert code == 0
        assert messages[-1] == f"{target} is up to date (1 videos)"
        assert target.read_bytes() == before

    def test_check_fails_on_drift_without_writing(self, tmp_path, cli):
        target = tmp_path / "c.yaml"
        cli("--source", A, "--output", target)
        before = target.read_bytes()
        code, messages = cli("--source", A, "--source", B, "--output", target, "--check")
        assert code == 1
        assert "is out of date (1 stored, 2 refreshed)" in messages[-1]
        assert target.read_bytes() == before

    def test_check_fails_when_the_catalog_does_not_exist(self, tmp_path, cli):
        target = tmp_path / "c.yaml"
        code, messages = cli("--source", A, "--output", target, "--check")
        assert code == 1
        assert "is out of date (0 stored, 1 refreshed)" in messages[-1]
        assert not target.exists()

    def test_check_detects_a_semantically_equal_but_unnormalized_file(self, tmp_path, cli):
        target = tmp_path / "c.yaml"
        target.write_text(f"# hand written\n- {A}\n", encoding="utf-8")
        code, messages = cli("--source", A, "--output", target, "--check")
        assert code == 1
        assert "normalized catalog content would change" in messages[-1]

    def test_a_write_failure_is_reported_not_raised(self, tmp_path, cli, monkeypatch):
        def fail(records, path):
            raise OSError("read-only file system")

        monkeypatch.setattr(sync, "write_catalog", fail)
        code, messages = cli("--source", A, "--output", tmp_path / "c.yaml")
        assert code == 1
        assert "could not write" in messages[-1]
        assert "read-only file system" in messages[-1]


class TestRequireRequests:
    def test_the_transport_is_imported_lazily_and_is_requests(self):
        requests = pytest.importorskip("requests")
        # Importing the library opens no socket (the autouse guard is active).
        assert REAL_REQUIRE_REQUESTS() is requests


class TestModuleSurface:
    def test_public_names_are_exported(self):
        for name in sync.__all__:
            assert callable(getattr(sync, name)), name

    def test_endpoints_are_https(self):
        assert sync.API_BASE.startswith("https://www.googleapis.com/")
        assert sync.RSS_BASE.startswith("https://www.youtube.com/")
