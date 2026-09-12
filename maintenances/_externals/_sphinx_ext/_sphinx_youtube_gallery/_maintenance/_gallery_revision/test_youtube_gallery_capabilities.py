"""Dependency-free capability checks for the typed YouTube gallery model.

These tests intentionally avoid Sphinx/docutils.  The catalog/model/query layer is
usable by sync/export tooling without importing the render stack.
"""
from __future__ import annotations

from pathlib import Path
import sys


_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)

from _sphinx_ext._sphinx_youtube_gallery.model import (
    CatalogError,
    derive_channel_records,
    normalize_channel_record,
    normalize_gallery_catalog,
    normalize_catalog,
)
from _sphinx_ext._sphinx_youtube_gallery.query import Query, apply_query
from _sphinx_ext._sphinx_youtube_gallery._video_options import LEAF_VIDEO_SPEC, VIDEO_SPEC

# Standalone and gallery-generated players share one validated option contract.
assert VIDEO_SPEC["video-width"] is LEAF_VIDEO_SPEC["width"]
assert LEAF_VIDEO_SPEC["privacy_mode"]("false") == "false"
assert LEAF_VIDEO_SPEC["privacy_mode"](None) == ""
assert LEAF_VIDEO_SPEC["url_parameters"]("rel=0&start=90") == "?rel=0&start=90"
for key, bad in (("width", "0"), ("height", "0%"), ("aspect", "0:9"), ("align", "diagonal")):
    try:
        LEAF_VIDEO_SPEC[key](bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid leaf video option accepted: {key}={bad!r}")
try:
    LEAF_VIDEO_SPEC["url_parameters"]("&".join(f"k{i}=v" for i in range(129)))
except ValueError:
    pass
else:
    raise AssertionError("unbounded leaf URL parameters were accepted")

# Channel references are canonicalized to the root card identity.
channel = normalize_channel_record("https://www.youtube.com/@youtube/videos")
assert channel.id == "@youtube"
assert channel.title == "@youtube"
assert channel.url == "https://www.youtube.com/@youtube"

# YouTube identity stays strict while explicit custom fields remain available to
# the same dotted-path query vocabulary as gallery-grid.
kind, videos = normalize_gallery_catalog(
    {
        "videos": [
            {
                "id": "aaaaaaaaaaa",
                "title": "Beta",
                "channel": "Zeta Research",
                "handle": "zeta",
                "tags": ["agents", "shared"],
                "published": "2026-01-01T00:00:00Z",
                "fields": {"category": "Agents", "audience": {"level": "advanced"}},
            },
            {
                "id": "bbbbbbbbbbb",
                "title": "Alpha",
                "channel": "Alpha Learning",
                "handle": "alpha",
                "tags": ["python"],
                "published": "2026-02-01T00:00:00Z",
                "fields": {"category": "Python", "audience": {"level": "beginner"}},
            },
            {
                "id": "ccccccccccc",
                "title": "Gamma",
                "channel": "Zeta Research",
                "handle": "@zeta",
                "tags": ["shared", "rag"],
                "published": "2026-03-01T00:00:00Z",
                "fields": {"category": "Agents", "audience": {"level": "intermediate"}},
            },
        ]
    },
    origin="capability fixture",
)
assert kind == "video"
assert videos[0].fields["audience"]["level"] == "advanced"

selected, total = apply_query(videos, Query(sort_by="category"))
assert total == 3
assert [record.title for record in selected] == ["Beta", "Gamma", "Alpha"]

selected, total = apply_query(videos, Query(group_by="audience.level"))
assert total == 3

# Projection happens before channel-level ordering.  Aggregates with clear
# semantics are deterministic: tag union, contributing-video count and newest
# known catalog publication timestamp.
channels = derive_channel_records(videos)
assert len(channels) == 2
zeta = next(record for record in channels if record.title == "Zeta Research")
assert zeta.video_count == 2
assert zeta.tags == ["agents", "shared", "rag"]
assert zeta.published.isoformat() == "2026-03-01T00:00:00+00:00"
ordered, total = apply_query(channels, Query(sort_by="title"))
assert total == 2
assert [record.title for record in ordered] == ["Alpha Learning", "Zeta Research"]
ordered, _ = apply_query(channels, Query(sort_by="-video_count"))
assert ordered[0].title == "Zeta Research"

# Stable channel id owns deduplication, while an authored handle remains the
# friendlier canonical link when both identities are known.
kind2, both_identity = normalize_gallery_catalog({"videos": [{
    "id": "ddddddddddd",
    "channel": "Friendly Channel",
    "channel_id": "UC" + "A" * 22,
    "handle": "friendly",
}]}, origin="dual channel identity")
projected_both = derive_channel_records(both_identity)
assert len(projected_both) == 1
assert projected_both[0].id == "UC" + "A" * 22
assert projected_both[0].handle == "friendly"
assert projected_both[0].url == "https://www.youtube.com/@friendly"



# Native channel catalogs use the same identity rule as projected channels:
# UC id for stable deduplication, authored handle for the friendly URL.
for native_entries in (
    [
        {"handle": "native-friendly", "channel_id": "UC" + "N" * 22},
        {"channel_id": "UC" + "N" * 22, "title": "Duplicate canonical id"},
    ],
    [
        {"channel_id": "UC" + "N" * 22},
        {"handle": "native-friendly", "channel_id": "UC" + "N" * 22},
    ],
    [
        {"handle": "native-friendly"},
        {"handle": "native-friendly", "channel_id": "UC" + "N" * 22},
    ],
):
    native_kind, native_channels = normalize_gallery_catalog(
        {"channels": native_entries}, origin="native dual identity"
    )
    assert native_kind == "channel"
    assert len(native_channels) == 1
    assert native_channels[0].id == "UC" + "N" * 22
    assert native_channels[0].handle == "native-friendly"
    assert native_channels[0].url == "https://www.youtube.com/@native-friendly"

for bad_channel in (
    {"handle": "bad handle with spaces"},
    {"channel_id": "UC-short"},
    {"handle": "valid-handle", "channel_id": "UC-short"},
):
    try:
        normalize_gallery_catalog({"channels": [bad_channel]}, origin="channel identity fixture")
    except CatalogError:
        pass
    else:
        raise AssertionError(f"invalid native channel identity accepted: {bad_channel!r}")

# Identity attributes fail at normalization instead of surfacing later as
# missing projection/filter behavior. Playlist admission stays prefix-agnostic.
for bad in (
    {"id": "eeeeeeeeeee", "channel_id": "UC-short"},
    {"id": "eeeeeeeeeee", "playlist_id": "bad!playlist"},
):
    try:
        normalize_gallery_catalog({"videos": [bad]}, origin="identity fixture")
    except CatalogError:
        pass
    else:
        raise AssertionError(f"invalid identity accepted: {bad!r}")
kind3, future_playlist = normalize_gallery_catalog({"videos": [{
    "id": "fffffffffff", "playlist_id": "ZZfuturefamily123456"
}]}, origin="future playlist family")
assert future_playlist[0].playlist_id == "ZZfuturefamily123456"

# One directive invocation is intentionally homogeneous.
for payload in (
    {"videos": ["aaaaaaaaaaa"], "channels": ["@youtube"]},
    {"channels": ["https://www.youtube.com/watch?v=aaaaaaaaaaa"]},
):
    try:
        normalize_gallery_catalog(payload, origin="invalid fixture")
    except CatalogError:
        pass
    else:
        raise AssertionError(f"invalid catalog accepted: {payload!r}")

# Custom metadata cannot overwrite player/card identity or presentation.
for reserved in ("title", "link", "content", "shadow"):
    try:
        normalize_gallery_catalog(
            {"videos": [{"id": "aaaaaaaaaaa", "fields": {reserved: "bad"}}]},
            origin="reserved field fixture",
        )
    except CatalogError:
        pass
    else:
        raise AssertionError(f"reserved custom field accepted: {reserved}")

# Typoed custom sort paths fail rather than silently sorting missing values.
try:
    apply_query(videos, Query(sort_by="title"))
except CatalogError:
    pass
else:
    raise AssertionError("unknown custom sort field was accepted")


# Projection must collapse weak @handle identity into a canonical UC id regardless
# of record order, and preserve the handle as the friendlier URL.
alias_records = normalize_catalog({"videos": [
    {"id": "abcdefghijk", "channel": "@same-channel"},
    {"id": "lmnopqrstuv", "channel": "Same Channel", "channel_id": "UC" + "Z" * 22, "handle": "same-channel"},
]}, "alias-order")
alias_channels = derive_channel_records(alias_records)
assert len(alias_channels) == 1
assert alias_channels[0].id == "UC" + "Z" * 22
assert alias_channels[0].handle == "same-channel"
assert alias_channels[0].url == "https://www.youtube.com/@same-channel"
assert alias_channels[0].video_count == 2

# Explicit native channel catalogs fail closed on contradictory identity pairs
# rather than allowing dictionary/source order to decide which URL survives.
for conflicting in (
    [
        {"handle": "collision", "channel_id": "UC" + "A" * 22},
        {"handle": "collision", "channel_id": "UC" + "B" * 22},
    ],
    [
        {"handle": "old-handle", "channel_id": "UC" + "C" * 22},
        {"handle": "new-handle", "channel_id": "UC" + "C" * 22},
    ],
):
    try:
        normalize_gallery_catalog({"channels": conflicting}, origin="contradictory identity")
    except CatalogError:
        pass
    else:
        raise AssertionError(f"contradictory native channel identity accepted: {conflicting!r}")

# Historical video catalogs are less strict: a reused/ambiguous handle must not
# merge two stable channels.  A handle-only record stays separate rather than
# being guessed into either canonical UC identity.
ambiguous = normalize_catalog({"videos": [
    {"id": "11111111111", "channel": "@reused"},
    {"id": "22222222222", "channel": "Older A", "channel_id": "UC" + "D" * 22, "handle": "reused"},
    {"id": "33333333333", "channel": "Newer B", "channel_id": "UC" + "E" * 22, "handle": "reused"},
]}, "ambiguous historical handle")
ambiguous_channels = derive_channel_records(ambiguous)
assert {record.id for record in ambiguous_channels} == {
    "@reused", "UC" + "D" * 22, "UC" + "E" * 22
}

# A stable channel may legitimately have historical handle changes in a video
# catalog.  Keep one UC card and choose the handle attached to its newest known
# video for the friendly URL.
renamed = normalize_catalog({"videos": [
    {"id": "44444444444", "channel": "Old Display Name", "channel_id": "UC" + "F" * 22, "handle": "old-name", "published": "2025-01-01"},
    {"id": "55555555555", "channel": "New Display Name", "channel_id": "UC" + "F" * 22, "handle": "new-name", "published": "2026-01-01"},
]}, "historical handle rename")
renamed_channels = derive_channel_records(renamed)
assert len(renamed_channels) == 1
assert renamed_channels[0].id == "UC" + "F" * 22
assert renamed_channels[0].handle == "new-name"
assert renamed_channels[0].url == "https://www.youtube.com/@new-name"
assert renamed_channels[0].title == "New Display Name"
assert renamed_channels[0].channel == "New Display Name"

print("CAPABILITY_PASS: typed fields, strict native identity, collision-safe historical projection, separate channel display/handle identity, strict homogeneity, and channel-level sorting")
