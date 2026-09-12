"""Dependency-light regression for youtube-gallery catalog refresh fidelity."""
from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path
import sys


_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)

from _sphinx_ext._sphinx_youtube_gallery import sync
from _sphinx_ext._sphinx_youtube_gallery.model import normalize_catalog

VIDEO_ID = "JXtISpdDPNY"
OLDER_ID = "abcdefghijk"

existing = normalize_catalog({
    "videos": [{
        "id": VIDEO_ID,
        "title": "Old provider title",
        "channel": "Statistics Globe",
        "channel_id": "UC" + "O" * 22,
        "description": "Rich description already known",
        "duration": 321,
        "playlist": "Curated playlist",
        "playlist_id": "PLexisting",
        "position": 7,
        "handle": "StatisticsGlobe",
        "tags": ["pca", "curated"],
        "fields": {"category": "Statistics", "audience": {"level": "intro"}},
    }, {
        "id": OLDER_ID,
        "title": "Older curated history",
        "tags": ["archive"],
    }]
}, "existing")
refreshed = normalize_catalog({
    "videos": [{
        "id": VIDEO_ID,
        "title": "Fresh provider title",
        "channel": "Statistics Globe Official",
        "channel_id": "UC" + "N" * 22,
        "published": "2026-09-10T08:00:00Z",
    }]
}, "refreshed")


# Source semantics stay consistent with the player/Add-video path: a watch URL
# carrying playlist context still means the exact ``v=`` video. Collection
# ingestion requires an explicit playlist URL/ID.
_seen = []
_orig_fetch_video, _orig_fetch_playlist = sync.fetch_video, sync.fetch_playlist
sync.fetch_video = lambda video_id, api_key="": (_seen.append(("video", video_id)) or [{"id": video_id}])
sync.fetch_playlist = lambda playlist_id, api_key="", **kwargs: (_seen.append(("playlist", playlist_id)) or [{"id": VIDEO_ID}])
try:
    sync.resolve_source(
        f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLcontext123456",
        "",
    )
    assert _seen == [("video", VIDEO_ID)], _seen
    _seen.clear()
    sync.resolve_source("https://www.youtube.com/playlist?list=PLcontext123456", "")
    assert _seen == [("playlist", "PLcontext123456")], _seen
finally:
    sync.fetch_video, sync.fetch_playlist = _orig_fetch_video, _orig_fetch_playlist

# Keyed exact-video acquisition returns real provider metadata instead of an ID-only record.
original_pages = sync._api_pages
sync._api_pages = lambda endpoint, params, api_key: iter([{
    "items": [{
        "id": VIDEO_ID,
        "snippet": {
            "title": "API exact video",
            "description": "Fetched description",
            "channelTitle": "Fetched channel",
            "channelId": "UC" + "A" * 22,
            "publishedAt": "2026-09-10T08:00:00Z",
            "tags": ["provider-tag-that-must-not-enter-author-tags"],
        },
        "contentDetails": {"duration": "PT2M3S"},
    }]
}])
try:
    exact = sync.fetch_video(VIDEO_ID, "test-key")
    assert exact[0]["title"] == "API exact video"
    assert exact[0]["duration"] == "PT2M3S"
    assert "tags" not in exact[0], "provider tags must not overwrite author-owned catalog tags"
finally:
    sync._api_pages = original_pages

# Keyed playlist acquisition must preserve playlist identity and obtain provider
# fields playlistItems cannot observe by itself (notably duration).
playlist_calls = []
def fake_playlist_pages(endpoint, params, api_key):
    playlist_calls.append((endpoint, dict(params)))
    if endpoint == "playlists":
        return iter([{"items": [{"id": "PLtest", "snippet": {"title": "Learning Path"}}]}])
    if endpoint == "playlistItems":
        return iter([{"items": [{
            "snippet": {
                "title": "Playlist video",
                "description": "Description",
                "channelTitle": "Owner Channel",
                "videoOwnerChannelTitle": "Owner Channel",
                "videoOwnerChannelId": "UC" + "B" * 22,
                "position": 2,
                "resourceId": {"kind": "youtube#video", "videoId": VIDEO_ID},
            },
            "contentDetails": {"videoPublishedAt": "2026-09-09T08:00:00Z"},
        }]}])
    if endpoint == "videos":
        return iter([{"items": [{"id": VIDEO_ID, "contentDetails": {"duration": "PT5M"}}]}])
    raise AssertionError(endpoint)

sync._api_pages = fake_playlist_pages
try:
    playlist_records = sync.fetch_playlist("PLtest", "test-key")
    assert playlist_records[0]["playlist"] == "Learning Path"
    assert playlist_records[0]["playlist_id"] == "PLtest"
    assert playlist_records[0]["duration"] == "PT5M"
    assert [call[0] for call in playlist_calls] == ["playlists", "playlistItems", "videos"]
finally:
    sync._api_pages = original_pages

# Acquisition stops at the same record ceiling enforced by catalog
# normalization, so an oversized provider response cannot burn quota/memory
# only to fail after all pages have been downloaded.
original_pages, original_limit = sync._api_pages, sync.MAX_COLLECTION_ITEMS
sync.MAX_COLLECTION_ITEMS = 1
def oversized_playlist_pages(endpoint, params, api_key):
    if endpoint == "playlists":
        return iter([{"items": [{"id": "PLbig", "snippet": {"title": "Big"}}]}])
    if endpoint == "playlistItems":
        return iter([{"items": [
            {"snippet": {"title": "One", "resourceId": {"kind": "youtube#video", "videoId": "11111111111"}}},
            {"snippet": {"title": "Two", "resourceId": {"kind": "youtube#video", "videoId": "22222222222"}}},
        ]}])
    raise AssertionError(f"unexpected API call after oversize detection: {endpoint}")
sync._api_pages = oversized_playlist_pages
try:
    try:
        sync.fetch_playlist("PLbig", "test-key")
    except sync.CatalogError as exc:
        assert "catalog limit" in str(exc)
    else:
        raise AssertionError("oversized playlist must fail during acquisition")
finally:
    sync._api_pages, sync.MAX_COLLECTION_ITEMS = original_pages, original_limit

# Keyed channel acquisition resolves the uploads playlist once and must not
# spend another API call looking up a generated playlist title that the channel
# wrapper immediately discards.
channel_calls = []
def fake_channel_pages(endpoint, params, api_key):
    channel_calls.append((endpoint, dict(params)))
    if endpoint == "channels":
        return iter([{"items": [{
            "id": "UC" + "C" * 22,
            "contentDetails": {"relatedPlaylists": {"uploads": "UU" + "C" * 22}},
        }]}])
    if endpoint == "playlistItems":
        return iter([{"items": [{
            "snippet": {
                "title": "Channel upload",
                "description": "Description",
                "videoOwnerChannelTitle": "Channel C",
                "videoOwnerChannelId": "UC" + "C" * 22,
                "position": 0,
                "resourceId": {"kind": "youtube#video", "videoId": VIDEO_ID},
            },
            "contentDetails": {"videoPublishedAt": "2026-09-09T09:00:00Z"},
        }]}])
    if endpoint == "videos":
        return iter([{"items": [{"id": VIDEO_ID, "contentDetails": {"duration": "PT7M"}}]}])
    raise AssertionError(endpoint)

sync._api_pages = fake_channel_pages
try:
    channel_records = sync.fetch_channel("UC" + "C" * 22, "test-key")
    assert channel_records[0]["id"] == VIDEO_ID
    assert channel_records[0]["duration"] == "PT7M"
    assert channel_records[0]["playlist"] == ""
    assert channel_records[0]["playlist_id"] == ""
    assert channel_records[0]["position"] is None
    assert [call[0] for call in channel_calls] == ["channels", "playlistItems", "videos"]
finally:
    sync._api_pages = original_pages

# Channel /playlists acquisition is deterministic even when the provider
# returns playlist descriptors in a different order. The canonical catalog is
# one-record-per-video, so overlapping membership keeps the lowest stable
# playlist id rather than whichever response happened to arrive first.
playlist_order_calls = []
def fake_channel_playlist_pages(endpoint, params, api_key):
    if endpoint == "playlists":
        return iter([{"items": [
            {"id": "PLz-last", "snippet": {"title": "Zed"}},
            {"id": "PLa-first", "snippet": {"title": "Alpha"}},
        ]}])
    raise AssertionError(endpoint)

def fake_fetch_playlist(playlist_id, api_key="", *, playlist_title=None):
    playlist_order_calls.append(playlist_id)
    return [{
        "id": VIDEO_ID,
        "title": "Overlapping video",
        "playlist": playlist_title or "",
        "playlist_id": playlist_id,
    }]

original_pages, original_fetch_playlist = sync._api_pages, sync.fetch_playlist
sync._api_pages, sync.fetch_playlist = fake_channel_playlist_pages, fake_fetch_playlist
try:
    overlap = sync.fetch_channel_playlists("UC" + "D" * 22, "test-key")
    assert playlist_order_calls == ["PLa-first", "PLz-last"]
    assert len(overlap) == 1
    assert overlap[0]["playlist_id"] == "PLa-first"
finally:
    sync._api_pages, sync.fetch_playlist = original_pages, original_fetch_playlist

# RSS playlist feeds distinguish the playlist title from the channel author and
# retain the playlist id so a later keyed refresh has the same semantic shape.
class _FakeResponse:
    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns:media="http://search.yahoo.com/mrss/">
      <title>Playlist Title</title><author><name>Channel Author</name></author>
      <entry><title>RSS video</title><yt:videoId>JXtISpdDPNY</yt:videoId><yt:channelId>UCBBBBBBBBBBBBBBBBBBBBBB</yt:channelId><published>2026-09-08T08:00:00+00:00</published><media:group><media:description>RSS description</media:description></media:group></entry>
    </feed>"""
    def raise_for_status(self): return None
class _FakeRequests:
    @staticmethod
    def get(*args, **kwargs): return _FakeResponse()
original_require = sync._require_requests
sync._require_requests = lambda: _FakeRequests
try:
    rss_records = sync._rss_items("playlist_id", "PLrss")
    assert rss_records[0]["channel"] == "Channel Author"
    assert rss_records[0]["playlist"] == "Playlist Title"
    assert rss_records[0]["playlist_id"] == "PLrss"
finally:
    sync._require_requests = original_require

merged = sync.merge_catalog_enrichment(refreshed, existing)
record = merged[0]
assert record.title == "Fresh provider title"
assert record.channel == "Statistics Globe Official"
assert record.channel_id == "UC" + "N" * 22
assert record.handle == "StatisticsGlobe"
assert record.tags == ["pca", "curated"]
assert record.fields == {"category": "Statistics", "audience": {"level": "intro"}}

text = sync.render_catalog(merged)
assert "handle: StatisticsGlobe" in text
assert "category: Statistics" in text
assert "level: intro" in text
assert "Fresh provider title" in text

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "youtube.yaml"
    assert sync.write_catalog(existing, path) is True
    before = path.read_text(encoding="utf-8")
    assert sync.write_catalog(existing, path) is False
    assert path.read_text(encoding="utf-8") == before

    # Patch only acquisition; exercise main's real normalize/merge/check/write path.
    original = sync.resolve_source
    sync.resolve_source = lambda source, api_key="": [{
        "id": VIDEO_ID,
        "title": "Fresh provider title",
        "channel": "Statistics Globe Official",
        "channel_id": "UC" + "N" * 22,
        "published": "2026-09-10T08:00:00Z",
    }]
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = sync.main(["--source", "https://youtu.be/example", "--output", str(path), "--check"])
        assert rc == 1, "metadata change with the same video id must fail --check"
        assert "normalized catalog content would change" in stderr.getvalue()

        with contextlib.redirect_stderr(io.StringIO()):
            assert sync.main(["--source", "https://youtu.be/example", "--output", str(path)]) == 0

        updated = normalize_catalog(__import__("yaml").safe_load(path.read_text(encoding="utf-8")), "updated")
        by_id = {record.id: record for record in updated}
        updated_record = by_id[VIDEO_ID]
        assert updated_record.title == "Fresh provider title"
        assert updated_record.handle == "StatisticsGlobe"
        assert updated_record.description == "Rich description already known"
        assert updated_record.duration == 321
        assert updated_record.playlist == "Curated playlist"
        assert updated_record.position == 7
        assert updated_record.tags == ["pca", "curated"]
        assert updated_record.fields["audience"]["level"] == "intro"
        assert OLDER_ID in by_id, "partial no-key RSS refresh must not prune unseen history"
        assert by_id[OLDER_ID].tags == ["archive"]

        with contextlib.redirect_stderr(io.StringIO()):
            assert sync.main(["--source", "https://youtu.be/example", "--output", str(path), "--check"]) == 0

        prune_err = io.StringIO()
        with contextlib.redirect_stderr(prune_err):
            assert sync.main(["--source", "https://youtu.be/example", "--output", str(path), "--prune"]) == 1
        assert "--prune requires YOUTUBE_API_KEY" in prune_err.getvalue()

        # A keyed refresh preserves unmatched history by default. Destructive
        # deletion is explicit via --prune, so credentials are not mistaken
        # for deletion intent.
        old_key = __import__("os").environ.get("YOUTUBE_API_KEY")
        __import__("os").environ["YOUTUBE_API_KEY"] = "test-key"
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                assert sync.main(["--source", "https://youtu.be/example", "--output", str(path)]) == 0
            keyed_safe = normalize_catalog(__import__("yaml").safe_load(path.read_text(encoding="utf-8")), "keyed-safe")
            assert {record.id for record in keyed_safe} == {VIDEO_ID, OLDER_ID}

            with contextlib.redirect_stderr(io.StringIO()):
                assert sync.main(["--source", "https://youtu.be/example", "--output", str(path), "--prune"]) == 0
            keyed = normalize_catalog(__import__("yaml").safe_load(path.read_text(encoding="utf-8")), "keyed")
            assert [record.id for record in keyed] == [VIDEO_ID]
            assert keyed[0].handle == "StatisticsGlobe"
        finally:
            if old_key is None:
                __import__("os").environ.pop("YOUTUBE_API_KEY", None)
            else:
                __import__("os").environ["YOUTUBE_API_KEY"] = old_key
    finally:
        sync.resolve_source = original


# API transport/shape failures must be bounded and secret-safe. requests'
# exception strings can contain the prepared URL, including ?key=..., so the
# sync layer must not echo those strings into CI logs.
class _ApiResponse:
    def __init__(self, payload=None, status=200, error=None):
        self._payload = payload
        self.status_code = status
        self._error = error
    def raise_for_status(self):
        if self._error is not None:
            raise RuntimeError(self._error)
    def json(self):
        return self._payload

class _ApiRequests:
    response = None
    @classmethod
    def get(cls, *args, **kwargs):
        return cls.response

original_require = sync._require_requests
sync._require_requests = lambda: _ApiRequests
try:
    _ApiRequests.response = _ApiResponse(
        {"items": []},
        status=403,
        error="403 for https://www.googleapis.com/youtube/v3/videos?key=SECRET_API_KEY",
    )
    try:
        list(sync._api_pages("videos", {"part": "snippet"}, "SECRET_API_KEY"))
    except sync.CatalogError as exc:
        message = str(exc)
        assert "HTTP 403" in message
        assert "SECRET_API_KEY" not in message
    else:
        raise AssertionError("HTTP failure must raise CatalogError")

    _ApiRequests.response = _ApiResponse([{"unexpected": "list"}])
    try:
        list(sync._api_pages("videos", {"part": "snippet"}, "SECRET_API_KEY"))
    except sync.CatalogError as exc:
        assert "expected an object" in str(exc)
    else:
        raise AssertionError("non-object API JSON must be rejected")
finally:
    sync._require_requests = original_require

# A keyed playlist id that is not visible to the API is an acquisition error,
# not an authoritative empty collection that --prune could use to erase data.
original_pages = sync._api_pages
sync._api_pages = lambda endpoint, params, api_key: iter([{"items": []}])
try:
    try:
        sync.fetch_playlist("PLmissing123", "test-key")
    except sync.CatalogError as exc:
        assert "not found or is not accessible" in str(exc)
    else:
        raise AssertionError("inaccessible keyed playlist must fail closed")
finally:
    sync._api_pages = original_pages

# Legacy explicit --playlist/--channel flags validate identifiers before any
# acquisition call, matching the typed --source path instead of sending malformed
# values to YouTube and reporting a remote error.
network_calls = []
orig_fp, orig_fc = sync.fetch_playlist, sync.fetch_channel
sync.fetch_playlist = lambda *a, **k: (network_calls.append(("playlist", a)) or [])
sync.fetch_channel = lambda *a, **k: (network_calls.append(("channel", a)) or [])
try:
    with tempfile.TemporaryDirectory() as tmp:
        badp = io.StringIO()
        with contextlib.redirect_stderr(badp):
            assert sync.main(["--playlist", "bad!playlist", "--output", str(Path(tmp)/"p.yaml")]) == 1
        assert "--playlist" in badp.getvalue()
        badc = io.StringIO()
        with contextlib.redirect_stderr(badc):
            assert sync.main(["--channel", "UC-short", "--output", str(Path(tmp)/"c.yaml")]) == 1
        assert "--channel" in badc.getvalue()
        assert network_calls == []
finally:
    sync.fetch_playlist, sync.fetch_channel = orig_fp, orig_fc

# Output-path I/O failures are handled as CLI errors instead of escaping with
# a traceback after a successful network refresh.
with tempfile.TemporaryDirectory() as tmp:
    blocker = Path(tmp) / "not-a-directory"
    blocker.write_text("file", encoding="utf-8")
    bad_output = blocker / "youtube.yaml"
    original = sync.resolve_source
    sync.resolve_source = lambda source, api_key="": [{"id": VIDEO_ID}]
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = sync.main(["--source", VIDEO_ID, "--output", str(bad_output)])
        assert rc == 1
        assert "could not write" in stderr.getvalue()
    finally:
        sync.resolve_source = original

print("SYNC_PIPELINE_PASS: enrichment + exact --check + partial-RSS retention + explicit keyed prune + idempotent atomic write + redacted API/I/O failures")
