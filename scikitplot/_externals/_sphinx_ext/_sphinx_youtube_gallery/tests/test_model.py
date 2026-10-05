"""
Tests for :mod:`.model`: the YouTube catalog data model.

Covers the value parsers (video reference, timestamp, duration), custom
``fields:`` validation, video and channel record normalisation, whole-catalog
normalisation (shape, limits, de-duplication) and the offline projection of a
video selection into channel cards. No Sphinx and no network are involved.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools

import pytest

from .. import model
from ..model import (
    CatalogError,
    ChannelRecord,
    VideoRecord,
    derive_channel_records,
    normalize_catalog,
    normalize_channel_record,
    normalize_gallery_catalog,
    normalize_record,
    parse_duration,
    parse_timestamp,
    parse_video_id,
)

UTC = dt.timezone.utc
VID = "dQw4w9WgXcQ"
VID2 = "JXtISpdDPNY"
UC_A = "UC" + "a" * 22
UC_B = "UC" + "b" * 22
PLAYLIST = "PL" + "x" * 16
IDS = [letter * 11 for letter in "abcdefgh"]


def video(index, **keys):
    """Return a normalized video record with a synthetic id."""
    return normalize_record({"id": IDS[index], **keys}, index)


# -- parse_video_id -----------------------------------------------------------


class TestParseVideoId:
    @pytest.mark.parametrize(
        "value",
        [
            VID,
            f"  {VID}  ",
            f"https://www.youtube.com/watch?v={VID}",
            f"http://www.youtube.com/watch?v={VID}",
            f"https://www.youtube.com/watch?v={VID}&list={PLAYLIST}&index=2",
            f"https://youtu.be/{VID}?t=30",
            f"https://www.youtube.com/shorts/{VID}",
            f"https://www.youtube.com/embed/{VID}",
            f"https://www.youtube-nocookie.com/embed/{VID}",
            f"https://m.youtube.com/watch?v={VID}",
        ],
        ids=[
            "bare-id",
            "padded-id",
            "watch",
            "http-watch",
            "watch-with-list",
            "short-host",
            "shorts",
            "embed",
            "nocookie",
            "mobile",
        ],
    )
    def test_every_single_video_form_reduces_to_the_id(self, value):
        assert parse_video_id(value) == VID

    @pytest.mark.parametrize(
        "value",
        [
            f"https://evil.example/watch?v={VID}",
            f"https://youtube.com.evil.example/watch?v={VID}",
            f"https://notyoutube.com/watch?v={VID}",
            "javascript:alert(1)",
            f"data:text/html,{VID}",
        ],
        ids=["foreign-host", "suffix-spoof", "prefix-spoof", "js-scheme", "data"],
    )
    def test_a_non_youtube_host_is_rejected(self, value):
        with pytest.raises(CatalogError):
            parse_video_id(value)

    @pytest.mark.parametrize(
        "value",
        [
            VID[:-1],
            VID + "Q",
            "dQw4w9WgXc!",
            "<script>aaa",
            "../../etc/pw",
            "",
            "   ",
            "https://www.youtube.com/watch",
            "https://youtu.be/",
        ],
        ids=[
            "ten-chars",
            "twelve-chars",
            "bad-char",
            "html",
            "path",
            "empty",
            "blank",
            "watch-without-v",
            "short-host-without-id",
        ],
    )
    def test_a_malformed_reference_is_rejected(self, value):
        with pytest.raises(CatalogError):
            parse_video_id(value)

    @pytest.mark.parametrize("value", [None, 5, 1.5, [VID], {"id": VID}, b"x"])
    def test_a_non_string_is_rejected_with_its_type(self, value):
        with pytest.raises(CatalogError, match=type(value).__name__):
            parse_video_id(value)

    @pytest.mark.parametrize(
        "value, noun",
        [
            (PLAYLIST, "playlist"),
            (f"https://www.youtube.com/playlist?list={PLAYLIST}", "playlist"),
            ("@handle", "channel"),
            (UC_A, "channel"),
            ("https://www.youtube.com/@handle/videos", "channel"),
        ],
        ids=["playlist-id", "playlist-url", "handle", "channel-id", "channel-tab"],
    )
    def test_a_collection_is_not_a_video(self, value, noun):
        with pytest.raises(CatalogError, match=noun):
            parse_video_id(value)

    def test_the_error_is_a_value_error(self):
        # Callers that predate CatalogError catch ValueError.
        assert issubclass(CatalogError, ValueError)


# -- parse_timestamp ----------------------------------------------------------


class TestParseTimestamp:
    def test_none_stays_none(self):
        assert parse_timestamp(None) is None

    @pytest.mark.parametrize("value", ["", "   ", "\n"], ids=["empty", "spaces", "nl"])
    def test_blank_text_is_treated_as_absent(self, value):
        assert parse_timestamp(value) is None

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("2024-03-01T10:00:00Z", dt.datetime(2024, 3, 1, 10, tzinfo=UTC)),
            ("2024-03-01t10:00:00z", dt.datetime(2024, 3, 1, 10, tzinfo=UTC)),
            ("2024-03-01", dt.datetime(2024, 3, 1, tzinfo=UTC)),
            ("  2024-03-01  ", dt.datetime(2024, 3, 1, tzinfo=UTC)),
            ("2024-03-01T10:00:00", dt.datetime(2024, 3, 1, 10, tzinfo=UTC)),
            ("2024-03-01T10:00:00+02:00", dt.datetime(2024, 3, 1, 8, tzinfo=UTC)),
            ("2024-03-01T00:30:00-01:00", dt.datetime(2024, 3, 1, 1, 30, tzinfo=UTC)),
        ],
        ids=[
            "rfc3339-Z",
            "lowercase-z",
            "bare-date",
            "padded",
            "naive-is-utc",
            "positive-offset",
            "negative-offset",
        ],
    )
    def test_strings_become_aware_utc(self, value, expected):
        parsed = parse_timestamp(value)
        assert parsed == expected
        assert parsed.utcoffset() == dt.timedelta(0)

    def test_a_date_becomes_utc_midnight(self):
        assert parse_timestamp(dt.date(2024, 1, 2)) == dt.datetime(
            2024, 1, 2, tzinfo=UTC
        )

    def test_a_naive_datetime_is_interpreted_as_utc(self):
        parsed = parse_timestamp(dt.datetime(2024, 1, 2, 3, 4, 5))
        assert parsed == dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)

    def test_an_aware_datetime_is_converted_to_utc(self):
        zone = dt.timezone(dt.timedelta(hours=2))
        parsed = parse_timestamp(dt.datetime(2024, 1, 2, 3, tzinfo=zone))
        assert parsed == dt.datetime(2024, 1, 2, 1, tzinfo=UTC)
        assert parsed.tzinfo is UTC

    @pytest.mark.parametrize(
        "value",
        ["nope", "2024-13-01", "2024-02-30", "01/03/2024", "<b>2024</b>"],
        ids=["word", "month-13", "feb-30", "slashes", "html"],
    )
    def test_unparsable_text_is_rejected_and_quoted(self, value):
        with pytest.raises(CatalogError) as caught:
            parse_timestamp(value)
        assert repr(value) in str(caught.value)

    @pytest.mark.parametrize("value", [5, 1.5, True, ["2024-01-01"], {"y": 2024}])
    def test_a_wrong_type_is_rejected_with_its_type(self, value):
        with pytest.raises(CatalogError, match=type(value).__name__):
            parse_timestamp(value)


# -- parse_duration -----------------------------------------------------------


class TestParseDuration:
    def test_none_stays_none(self):
        assert parse_duration(None) is None

    @pytest.mark.parametrize("value", [0, 1, 90, 10**9])
    def test_an_integer_is_already_seconds(self, value):
        assert parse_duration(value) == value

    @pytest.mark.parametrize(
        "value, seconds",
        [
            ("PT1H2M30S", 3750),
            ("PT5M", 300),
            ("PT45S", 45),
            ("PT0S", 0),
            ("P1D", 86400),
            ("P1DT1S", 86401),
            ("P2DT3H4M5S", 2 * 86400 + 3 * 3600 + 4 * 60 + 5),
            ("  PT5M  ", 300),
            ("PT90M", 5400),
        ],
        ids=[
            "h-m-s",
            "minutes",
            "seconds",
            "zero",
            "day",
            "day-and-second",
            "all-parts",
            "padded",
            "unnormalised-minutes",
        ],
    )
    def test_iso_8601_durations(self, value, seconds):
        assert parse_duration(value) == seconds

    @pytest.mark.parametrize(
        "value, seconds",
        [("2:30", 150), ("1:02:30", 3750), ("0:00", 0), ("10:00:00", 36000)],
        ids=["m-s", "h-m-s", "zero", "ten-hours"],
    )
    def test_clock_durations(self, value, seconds):
        assert parse_duration(value) == seconds

    def test_a_negative_integer_is_rejected(self):
        with pytest.raises(CatalogError, match="negative"):
            parse_duration(-1)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_not_a_number_of_seconds(self, value):
        with pytest.raises(CatalogError, match="boolean"):
            parse_duration(value)

    @pytest.mark.parametrize("value", [1.5, [90], {"s": 90}, b"90"])
    def test_a_wrong_type_is_rejected_with_its_type(self, value):
        with pytest.raises(CatalogError, match=type(value).__name__):
            parse_duration(value)

    @pytest.mark.parametrize(
        "value",
        ["", "P", "PT", "pt5m", "90", "ninety", "PT1H2M30", "PT-5M", "<b>"],
        ids=[
            "empty",
            "P-only",
            "PT-only",
            "lowercase-iso",
            "digit-string",
            "word",
            "missing-unit",
            "negative-iso",
            "html",
        ],
    )
    def test_unparsable_text_is_rejected_and_quoted(self, value):
        with pytest.raises(CatalogError) as caught:
            parse_duration(value)
        assert repr(value) in str(caught.value)

    @pytest.mark.parametrize(
        "value",
        ["1:2:3:4", "a:b", "1:", ":30", "-1:30", "1:-30", "1 : 30", "1:3.5"],
        ids=[
            "four-parts",
            "letters",
            "trailing-colon",
            "leading-colon",
            "negative-minutes",
            "negative-seconds",
            "inner-spaces",
            "fraction",
        ],
    )
    def test_a_malformed_clock_is_rejected(self, value):
        with pytest.raises(CatalogError, match="clock"):
            parse_duration(value)

    @pytest.mark.parametrize("value", ["²:30", "1:①"], ids=["sup-2", "circled"])
    def test_a_non_decimal_digit_clock_is_a_catalog_error(self, value):
        # Documented contract: "Raises CatalogError if the value is present
        # but cannot be parsed". A plain ValueError is not caught by
        # normalize_record or by the directive, so it aborts the build.
        with pytest.raises(CatalogError):
            parse_duration(value)


# -- custom ``fields:`` metadata ----------------------------------------------


class TestCustomFields:
    def test_absent_fields_are_an_empty_mapping(self):
        assert normalize_record(VID).fields == {}
        assert normalize_record({"id": VID, "fields": None}).fields == {}

    def test_scalars_lists_and_nested_mappings_are_kept_verbatim(self):
        fields = {
            "category": "tutorial",
            "level": 3,
            "ratio": 0.5,
            "featured": True,
            "missing": None,
            "when": dt.date(2024, 1, 2),
            "stamp": dt.datetime(2024, 1, 2, 3),
            "topics": ["pca", "t-sne", 3, None],
            "audience": {"level": "beginner", "langs": ["en", "tr"]},
            "with-dash_and_9": "ok",
        }
        assert normalize_record({"id": VID, "fields": fields}).fields == fields

    def test_the_result_does_not_alias_the_input_containers(self):
        raw = {"topics": ["a"], "audience": {"level": "x"}}
        record = normalize_record({"id": VID, "fields": raw})
        raw["topics"].append("b")
        raw["audience"]["level"] = "changed"
        assert record.fields == {"topics": ["a"], "audience": {"level": "x"}}

    @pytest.mark.parametrize(
        "value", [["a"], "text", 5, ("a",)], ids=["list", "str", "int", "tuple"]
    )
    def test_a_non_mapping_is_rejected(self, value):
        with pytest.raises(CatalogError, match=r"record 4: fields must be a mapping"):
            normalize_record({"id": VID, "fields": value}, 4)

    @pytest.mark.parametrize(
        "key",
        ["1a", "", "a b", "a.b", "_a", "-a", "a/b", "<script>", "café", 5, None],
        ids=[
            "leading-digit",
            "empty",
            "space",
            "dot",
            "leading-underscore",
            "leading-dash",
            "slash",
            "html",
            "non-ascii",
            "int-key",
            "none-key",
        ],
    )
    def test_an_unsafe_key_is_rejected(self, key):
        with pytest.raises(CatalogError, match=r"record 2: .*must start with"):
            normalize_record({"id": VID, "fields": {key: 1}}, 2)

    def test_an_unsafe_nested_key_names_its_parent_path(self):
        with pytest.raises(CatalogError, match=r"fields\.audience key 'a b'"):
            normalize_record({"id": VID, "fields": {"audience": {"a b": 1}}})

    @pytest.mark.parametrize(
        "key",
        sorted(model._CUSTOM_FIELD_RESERVED),
    )
    def test_every_reserved_top_level_name_is_rejected(self, key):
        with pytest.raises(CatalogError, match="reserved"):
            normalize_record({"id": VID, "fields": {key: "x"}})

    @pytest.mark.parametrize(
        "key", ["title", "id", "url", "link", "class-card", "img-top", "content"]
    )
    def test_reserved_names_cover_identity_and_card_options(self, key):
        assert key in model._CUSTOM_FIELD_RESERVED

    def test_a_reserved_name_is_allowed_below_the_top_level(self):
        # Only the flattened top-level keys can collide with card options.
        record = normalize_record({"id": VID, "fields": {"meta": {"title": "x"}}})
        assert record.fields == {"meta": {"title": "x"}}

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
    )
    def test_a_non_finite_float_is_rejected(self, value):
        with pytest.raises(CatalogError, match=r"fields\.score must be a finite"):
            normalize_record({"id": VID, "fields": {"score": value}})

    def test_a_non_finite_float_inside_a_list_is_located(self):
        with pytest.raises(CatalogError, match=r"fields\.scores\[1\] must be a finite"):
            normalize_record({"id": VID, "fields": {"scores": [1.0, float("nan")]}})

    @pytest.mark.parametrize(
        "item", [["nested"], {"k": "v"}], ids=["list-in-list", "mapping-in-list"]
    )
    def test_a_container_inside_a_list_is_rejected(self, item):
        with pytest.raises(CatalogError, match=r"fields\.topics\[1\] must be a scalar"):
            normalize_record({"id": VID, "fields": {"topics": ["ok", item]}})

    @pytest.mark.parametrize(
        "value",
        [{1, 2}, ("a", "b"), b"bytes", object(), 1 + 2j],
        ids=["set", "tuple", "bytes", "object", "complex"],
    )
    def test_an_unsupported_value_type_is_rejected(self, value):
        with pytest.raises(CatalogError, match="unsupported value type"):
            normalize_record({"id": VID, "fields": {"thing": value}})

    @staticmethod
    def _nested(depth):
        """Return a mapping whose single leaf sits ``depth`` keys deep."""
        node = 1
        for name in reversed("abcdefghij"[:depth]):
            node = {name: node}
        return node

    def test_nesting_up_to_the_limit_is_accepted(self):
        fields = self._nested(model._MAX_CUSTOM_FIELD_DEPTH)
        assert normalize_record({"id": VID, "fields": fields}).fields == fields

    def test_nesting_beyond_the_limit_is_rejected(self):
        fields = self._nested(model._MAX_CUSTOM_FIELD_DEPTH + 1)
        with pytest.raises(CatalogError, match="maximum metadata nesting depth"):
            normalize_record({"id": VID, "fields": fields})

    def test_hostile_text_values_are_kept_as_data_not_interpreted(self):
        payload = "<script>alert(1)</script>"
        record = normalize_record({"id": VID, "fields": {"category": payload}})
        assert record.fields["category"] == payload


# -- normalize_record ---------------------------------------------------------


class TestNormalizeRecord:
    def test_a_bare_string_is_a_video_reference(self):
        record = normalize_record(f"https://youtu.be/{VID}?t=10")
        assert record == VideoRecord(
            id=VID, title=VID, url=f"https://www.youtube.com/watch?v={VID}"
        )

    def test_every_field_is_normalised(self):
        record = normalize_record(
            {
                "id": VID,
                "title": "  PCA  ",
                "description": " explained ",
                "channel": " StatQuest ",
                "channel_id": UC_A,
                "handle": "@statquest",
                "playlist": " ML ",
                "playlist_id": PLAYLIST,
                "position": 0,
                "published": "2024-03-01T10:00:00Z",
                "duration": "PT1M30S",
                "tags": [" pca ", "", "stats"],
                "fields": {"category": "ml"},
            }
        )
        assert record == VideoRecord(
            id=VID,
            title="PCA",
            description="explained",
            channel="StatQuest",
            channel_id=UC_A,
            handle="statquest",
            playlist="ML",
            playlist_id=PLAYLIST,
            position=0,
            published=dt.datetime(2024, 3, 1, 10, tzinfo=UTC),
            duration=90,
            tags=["pca", "stats"],
            fields={"category": "ml"},
            url=f"https://www.youtube.com/watch?v={VID}",
        )

    def test_normalisation_is_deterministic(self):
        raw = {"id": VID, "title": "T", "tags": ["b", "a"], "published": "2024-01-01"}
        assert normalize_record(dict(raw)) == normalize_record(dict(raw))

    def test_the_input_mapping_is_not_mutated(self):
        raw = {"id": f"https://youtu.be/{VID}", "tags": [" a "], "title": " T "}
        snapshot = {"id": raw["id"], "tags": list(raw["tags"]), "title": raw["title"]}
        normalize_record(raw)
        assert raw == snapshot

    @pytest.mark.parametrize("title", [None, "", "   ", "\t\n"])
    def test_a_missing_title_falls_back_to_the_id(self, title):
        assert normalize_record({"id": VID, "title": title}).title == VID

    def test_url_alone_identifies_the_video(self):
        record = normalize_record({"url": f"https://www.youtube.com/watch?v={VID}"})
        assert record.id == VID

    def test_an_empty_id_falls_back_to_the_url(self):
        assert normalize_record({"id": "", "url": f"https://youtu.be/{VID}"}).id == VID

    def test_the_stored_url_is_canonical_and_drops_extra_parameters(self):
        record = normalize_record(
            {"url": f"https://www.youtube.com/watch?v={VID}&list={PLAYLIST}&t=30s&x=<y>"}
        )
        assert record.url == f"https://www.youtube.com/watch?v={VID}"

    def test_matching_id_and_url_are_accepted(self):
        record = normalize_record({"id": VID, "url": f"https://youtu.be/{VID}"})
        assert record.id == VID

    def test_conflicting_id_and_url_are_rejected(self):
        with pytest.raises(CatalogError, match="record 7: url and id name different"):
            normalize_record({"id": VID, "url": f"https://youtu.be/{VID2}"}, 7)

    @pytest.mark.parametrize(
        "url",
        ["https://evil.example/x", "javascript:alert(1)", PLAYLIST],
        ids=["foreign-host", "js-scheme", "playlist"],
    )
    def test_an_invalid_url_beside_an_id_names_the_record(self, url):
        # Docstring: "The message always names ``index``."
        with pytest.raises(CatalogError, match=r"^record 3:"):
            normalize_record({"id": VID, "url": url}, 3)

    @pytest.mark.parametrize(
        "url",
        ["https://evil.example/x", "javascript:alert(1)", PLAYLIST],
        ids=["foreign-host", "js-scheme", "playlist"],
    )
    def test_an_invalid_url_beside_an_id_is_rejected(self, url):
        with pytest.raises(CatalogError):
            normalize_record({"id": VID, "url": url}, 3)

    @pytest.mark.parametrize(
        "raw",
        [5, None, 1.5, ["x"], (VID,), True],
        ids=["int", "none", "float", "list", "tuple", "bool"],
    )
    def test_a_non_mapping_entry_is_rejected_with_its_type(self, raw):
        with pytest.raises(CatalogError) as caught:
            normalize_record(raw, 9)
        message = str(caught.value)
        assert message.startswith("record 9: expected a mapping")
        assert type(raw).__name__ in message

    def test_an_unknown_key_is_rejected_and_the_valid_keys_listed(self):
        with pytest.raises(CatalogError) as caught:
            normalize_record({"id": VID, "titles": "typo", "zzz": 1}, 1)
        message = str(caught.value)
        assert message.startswith("record 1: unknown key(s) ['titles', 'zzz']")
        assert "'title'" in message and "'fields'" in message

    @pytest.mark.parametrize("raw", [{}, {"title": "x"}, {"id": None}])
    def test_a_record_without_a_reference_is_rejected(self, raw):
        with pytest.raises(CatalogError, match=r"record 2: missing required key 'id'"):
            normalize_record(raw, 2)

    @pytest.mark.parametrize(
        "key",
        ["title", "description", "channel", "playlist", "handle", "channel_id",
         "playlist_id", "url"],
    )
    @pytest.mark.parametrize("value", [5, ["x"], {"a": 1}, True], ids=repr)
    def test_a_text_field_must_be_a_string(self, key, value):
        with pytest.raises(CatalogError) as caught:
            normalize_record({"id": VID, key: value}, 6)
        message = str(caught.value)
        assert message.startswith("record 6:")
        assert type(value).__name__ in message

    @pytest.mark.parametrize(
        "raw, fragment",
        [
            ({"id": [VID]}, "list"),
            ({"id": "not a video"}, "not a recognised YouTube reference"),
            ({"id": f"https://evil.example/watch?v={VID}"}, "not a YouTube URL"),
            ({"id": PLAYLIST}, "not a single video"),
            ({"id": VID, "published": "nope"}, "ISO-8601"),
            ({"id": VID, "published": 5}, "timestamp must be"),
            ({"id": VID, "duration": "x"}, "not a valid duration"),
            ({"id": VID, "duration": -3}, "negative"),
            ({"id": VID, "duration": True}, "boolean"),
            ({"id": VID, "tags": [1]}, "tags entries must be strings"),
            ({"id": VID, "tags": {"a": 1}}, "tags must be a string or list"),
            ({"id": VID, "tags": 5}, "tags must be a string or list"),
            ({"id": VID, "position": "1"}, "position must be an integer, got str"),
            ({"id": VID, "position": 1.0}, "position must be an integer, got float"),
            ({"id": VID, "position": True}, "position must be an integer, got bool"),
            ({"id": VID, "position": -1}, "position must not be negative"),
            ({"id": VID, "handle": "a b"}, "invalid channel handle"),
            ({"id": VID, "handle": "<script>"}, "invalid channel handle"),
            ({"id": VID, "channel_id": "UCshort"}, "invalid channel_id"),
            ({"id": VID, "channel_id": "@handle"}, "invalid channel_id"),
            ({"id": VID, "playlist_id": "??"}, "invalid playlist_id"),
            ({"id": VID, "playlist_id": "../x"}, "invalid playlist_id"),
        ],
        ids=[
            "id-list",
            "id-garbage",
            "id-foreign-host",
            "id-playlist",
            "published-word",
            "published-int",
            "duration-word",
            "duration-negative",
            "duration-bool",
            "tags-int-entry",
            "tags-mapping",
            "tags-int",
            "position-str",
            "position-float",
            "position-bool",
            "position-negative",
            "handle-space",
            "handle-html",
            "channel-id-short",
            "channel-id-handle",
            "playlist-id-chars",
            "playlist-id-traversal",
        ],
    )
    def test_every_field_failure_names_the_record_once(self, raw, fragment):
        with pytest.raises(CatalogError) as caught:
            normalize_record(raw, 3)
        message = str(caught.value)
        assert message.startswith("record 3: ")
        assert message.count("record 3:") == 1
        assert fragment in message

    def test_a_single_tag_string_becomes_a_one_item_list(self):
        assert normalize_record({"id": VID, "tags": " pca "}).tags == ["pca"]

    def test_tags_keep_order_and_drop_blank_entries(self):
        record = normalize_record({"id": VID, "tags": ["b", " ", "a", "", "B"]})
        assert record.tags == ["b", "a", "B"]

    def test_position_zero_is_kept_distinct_from_absent(self):
        assert normalize_record({"id": VID, "position": 0}).position == 0
        assert normalize_record({"id": VID}).position is None

    def test_a_handle_loses_its_leading_at_sign(self):
        assert normalize_record({"id": VID, "handle": "@Foo"}).handle == "Foo"

    def test_hostile_text_is_stored_verbatim_as_data(self):
        payload = "<script>alert(1)</script> `x` :ref:`y`"
        record = normalize_record(
            {"id": VID, "title": payload, "description": payload, "channel": payload}
        )
        assert record.title == record.description == record.channel == payload
        # The link target is rebuilt from the validated id, never from input.
        assert record.url == f"https://www.youtube.com/watch?v={VID}"

    def test_unicode_text_survives(self):
        record = normalize_record(
            {"id": VID, "title": "Çok güzel \U0001f600", "tags": ["ü"]}
        )
        assert record.title == "Çok güzel \U0001f600"
        assert record.tags == ["ü"]

    def test_the_record_is_immutable(self):
        record = normalize_record(VID)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.title = "changed"

    @pytest.mark.parametrize(
        "published, year",
        [("2024-03-01", "2024"), ("1999-12-31T23:59:59Z", "1999"), (None, "unknown")],
        ids=["date", "timestamp", "absent"],
    )
    def test_year_is_a_string_usable_as_a_heading(self, published, year):
        assert normalize_record({"id": VID, "published": published}).year == year

    def test_year_is_taken_in_utc(self):
        record = normalize_record({"id": VID, "published": "2024-01-01T00:30:00+02:00"})
        assert record.year == "2023"


# -- normalize_catalog --------------------------------------------------------


class TestNormalizeCatalog:
    @pytest.mark.parametrize(
        "payload", [None, [], {"videos": None}, {"videos": []}],
        ids=["none", "empty-list", "videos-none", "videos-empty"],
    )
    def test_an_empty_catalog_is_an_empty_list(self, payload):
        assert normalize_catalog(payload) == []

    def test_a_bare_list_and_a_videos_mapping_are_equivalent(self):
        entries = [VID, {"id": VID2, "title": "PCA"}]
        assert normalize_catalog(entries) == normalize_catalog({"videos": entries})

    def test_sibling_metadata_beside_videos_is_ignored(self):
        payload = {"synced": "2024-01-01", "source": "x", "videos": [VID]}
        assert [record.id for record in normalize_catalog(payload)] == [VID]

    def test_source_order_is_preserved(self):
        ids = [IDS[3], IDS[0], IDS[2], IDS[1]]
        assert [record.id for record in normalize_catalog(ids)] == ids

    def test_the_first_duplicate_wins_and_order_is_stable(self):
        records = normalize_catalog(
            [
                {"id": IDS[0], "title": "first"},
                IDS[1],
                {"url": f"https://youtu.be/{IDS[0]}", "title": "second"},
                {"id": IDS[0], "title": "third"},
            ]
        )
        assert [(record.id, record.title) for record in records] == [
            (IDS[0], "first"),
            (IDS[1], IDS[1]),
        ]

    @pytest.mark.parametrize(
        "payload, name",
        [("text", "str"), (5, "int"), ((VID,), "tuple"), ({"videos": "x"}, "str"),
         ({"videos": {"a": 1}}, "dict"), (True, "bool")],
        ids=["str", "int", "tuple", "videos-str", "videos-mapping", "bool"],
    )
    def test_a_wrong_shape_is_rejected_with_origin_and_type(self, payload, name):
        with pytest.raises(CatalogError) as caught:
            normalize_catalog(payload, "data.yaml")
        message = str(caught.value)
        assert message.startswith("data.yaml: expected a list of records")
        assert message.endswith(f"got {name}")

    def test_a_mapping_without_videos_lists_what_was_found(self):
        with pytest.raises(CatalogError) as caught:
            normalize_catalog({"video": [VID], "b": 1}, "data.yaml")
        assert str(caught.value) == (
            "data.yaml: mapping payload must contain a 'videos' key, "
            "found ['b', 'video']"
        )

    def test_a_bad_record_is_located_by_origin_and_index(self):
        with pytest.raises(CatalogError) as caught:
            normalize_catalog([VID, {"id": VID2, "titles": "x"}], "data.yaml")
        assert str(caught.value).startswith("data.yaml: record 1: unknown key(s)")

    def test_a_bad_record_fails_even_when_it_duplicates_an_earlier_id(self):
        with pytest.raises(CatalogError, match="record 1"):
            normalize_catalog([VID, {"id": VID, "position": -1}])

    def test_the_default_origin_is_named(self):
        with pytest.raises(CatalogError, match=r"^catalog: "):
            normalize_catalog("text")

    def test_the_item_limit_is_inclusive(self):
        limit = model.MAX_COLLECTION_ITEMS
        assert len(normalize_catalog([VID] * limit)) == 1

    def test_a_catalog_over_the_item_limit_is_rejected(self):
        limit = model.MAX_COLLECTION_ITEMS
        with pytest.raises(CatalogError) as caught:
            normalize_catalog([VID] * (limit + 1), "big.yaml")
        message = str(caught.value)
        assert message.startswith("big.yaml: catalog contains")
        assert f"{limit:,}" in message and f"{limit + 1:,}" in message


# -- normalize_channel_record -------------------------------------------------


class TestNormalizeChannelRecord:
    def test_a_handle_string(self):
        assert normalize_channel_record("@Foo") == ChannelRecord(
            id="@foo",
            title="@Foo",
            channel="@Foo",
            handle="Foo",
            url="https://www.youtube.com/@Foo",
        )

    def test_a_channel_id_string(self):
        assert normalize_channel_record(UC_A) == ChannelRecord(
            id=UC_A,
            title=UC_A,
            channel=UC_A,
            channel_id=UC_A,
            url=f"https://www.youtube.com/channel/{UC_A}",
        )

    @pytest.mark.parametrize(
        "reference",
        [
            "https://www.youtube.com/@Foo",
            "https://www.youtube.com/@Foo/videos",
            "https://www.youtube.com/@Foo/playlists?view=1",
            "https://m.youtube.com/@Foo/streams",
            "https://www.youtube.com/@Foo?x=<script>alert(1)</script>",
        ],
        ids=["root", "videos-tab", "tab-and-query", "mobile-host", "hostile-query"],
    )
    def test_a_handle_url_is_reduced_to_the_channel_root(self, reference):
        record = normalize_channel_record(reference)
        assert record.url == "https://www.youtube.com/@Foo"
        assert (record.id, record.handle, record.title) == ("@foo", "Foo", "@Foo")

    def test_a_channel_id_url_is_reduced_to_the_channel_root(self):
        record = normalize_channel_record(
            f"https://www.youtube.com/channel/{UC_A}/playlists"
        )
        assert record.url == f"https://www.youtube.com/channel/{UC_A}"
        assert record.id == record.channel_id == UC_A

    @pytest.mark.parametrize(
        "reference",
        [
            "https://www.youtube.com/c/Name",
            "https://www.youtube.com/user/Name",
            "https://www.youtube.com/Name",
        ],
        ids=["custom", "legacy-user", "vanity"],
    )
    def test_a_legacy_name_url_keeps_the_name_as_identity(self, reference):
        record = normalize_channel_record(reference)
        assert (record.id, record.title, record.handle, record.channel_id) == (
            "name",
            "Name",
            "",
            "",
        )
        assert record.url == "https://www.youtube.com/Name"

    @pytest.mark.parametrize(
        "raw",
        [{"handle": "Foo"}, {"handle": "@Foo"}, {"id": "@Foo"}, {"url": "@Foo"}],
        ids=["handle", "at-handle", "id", "url"],
    )
    def test_any_one_reference_key_is_enough(self, raw):
        record = normalize_channel_record(raw)
        assert (record.id, record.handle) == ("@foo", "Foo")

    def test_channel_id_alone_is_enough(self):
        assert normalize_channel_record({"channel_id": UC_A}).id == UC_A

    def test_a_handle_identity_is_case_insensitive_but_keeps_its_spelling(self):
        upper = normalize_channel_record("@FooBar")
        lower = normalize_channel_record("@foobar")
        assert upper.id == lower.id == "@foobar"
        assert upper.url == "https://www.youtube.com/@FooBar"

    def test_channel_id_owns_identity_and_the_handle_owns_the_link(self):
        record = normalize_channel_record({"handle": "@Foo", "channel_id": UC_A})
        assert record.id == UC_A
        assert record.url == "https://www.youtube.com/@Foo"
        assert record.title == record.channel == "@Foo"

    def test_metadata_is_stripped_and_kept(self):
        record = normalize_channel_record(
            {
                "id": "@Foo",
                "title": "  Foo Channel ",
                "channel": " Foo ",
                "description": " about ",
                "tags": ["a", " ", " b "],
                "fields": {"category": "x"},
            }
        )
        assert (record.title, record.channel, record.description) == (
            "Foo Channel",
            "Foo",
            "about",
        )
        assert record.tags == ["a", "b"]
        assert record.fields == {"category": "x"}

    def test_channel_defaults_to_the_title(self):
        assert normalize_channel_record({"id": "@Foo", "title": "T"}).channel == "T"

    def test_video_only_fields_are_empty_on_a_channel(self):
        record = normalize_channel_record("@Foo")
        assert (record.playlist, record.playlist_id) == ("", "")
        assert record.position is record.published is record.duration is None
        assert record.video_count is None
        assert record.year == "unknown"

    def test_agreeing_handle_and_reference_are_accepted_case_insensitively(self):
        record = normalize_channel_record({"id": "@Foo", "handle": "foo"})
        assert record.id == "@foo"

    def test_agreeing_channel_id_and_reference_are_accepted(self):
        record = normalize_channel_record({"id": UC_A, "channel_id": UC_A})
        assert record.channel_id == UC_A

    def test_a_handle_that_contradicts_the_reference_is_rejected(self):
        with pytest.raises(CatalogError, match="record 2: handle and channel URL"):
            normalize_channel_record({"id": "@Foo", "handle": "Bar"}, 2)

    def test_a_channel_id_that_contradicts_the_reference_is_rejected(self):
        with pytest.raises(CatalogError, match="record 2: channel_id and channel URL"):
            normalize_channel_record({"id": UC_A, "channel_id": UC_B}, 2)

    @pytest.mark.parametrize(
        "raw", [5, None, 1.5, ["@Foo"], ("@Foo",)], ids=["int", "none", "float", "list", "tuple"]
    )
    def test_a_non_mapping_entry_is_rejected_with_its_type(self, raw):
        with pytest.raises(CatalogError) as caught:
            normalize_channel_record(raw, 4)
        message = str(caught.value)
        assert message.startswith("record 4: expected a mapping or channel reference")
        assert type(raw).__name__ in message

    @pytest.mark.parametrize(
        "key", ["playlist", "playlist_id", "position", "published", "duration", "zzz"]
    )
    def test_a_video_only_or_unknown_key_is_rejected(self, key):
        with pytest.raises(CatalogError, match=r"record 1: unknown channel key"):
            normalize_channel_record({"id": "@Foo", key: "x"}, 1)

    @pytest.mark.parametrize("raw", ["", "   ", {}, {"title": "T"}, {"id": None}])
    def test_a_record_without_a_reference_is_rejected(self, raw):
        with pytest.raises(CatalogError, match="record 0: missing channel reference"):
            normalize_channel_record(raw)

    @pytest.mark.parametrize(
        "raw, fragment",
        [
            (VID, "expected a YouTube channel, got the video"),
            (f"https://youtu.be/{VID}", "expected a YouTube channel, got the video"),
            (PLAYLIST, "expected a YouTube channel, got the user playlist"),
            ("https://www.youtube.com/results?search_query=x", "got a search"),
            ("https://evil.example/@foo", "not a YouTube URL"),
            ("javascript:alert(1)", "not a YouTube URL"),
            ("Some Display Name", "not a recognised YouTube reference"),
            ("@<script>", "not a valid YouTube handle"),
            ({"handle": "a b"}, "invalid channel handle"),
            ({"handle": "<b>"}, "invalid channel handle"),
            ({"channel_id": "bad"}, "bad"),
            ({"id": "@Foo", "channel_id": "UCshort"}, "invalid channel_id"),
            ({"id": "@Foo", "title": 5}, "title must be a string, got int"),
            ({"id": "@Foo", "description": ["x"]}, "description must be a string"),
            ({"id": 5}, "id must be a string, got int"),
            ({"id": "@Foo", "tags": [1]}, "tags entries must be strings"),
            ({"id": "@Foo", "tags": 5}, "tags must be a string or list"),
            ({"id": "@Foo", "fields": {"title": 1}}, "reserved"),
            ({"id": "@Foo", "fields": [1]}, "fields must be a mapping"),
        ],
        ids=[
            "video-id",
            "video-url",
            "playlist",
            "search",
            "foreign-host",
            "js-scheme",
            "display-name",
            "html-handle",
            "handle-space",
            "handle-html",
            "channel-id-garbage",
            "channel-id-short",
            "title-int",
            "description-list",
            "id-int",
            "tags-int-entry",
            "tags-int",
            "fields-reserved",
            "fields-list",
        ],
    )
    def test_every_failure_names_the_record_once(self, raw, fragment):
        with pytest.raises(CatalogError) as caught:
            normalize_channel_record(raw, 5)
        message = str(caught.value)
        assert message.startswith("record 5: ")
        assert message.count("record 5:") == 1
        assert fragment in message

    def test_hostile_title_text_is_stored_as_data_and_the_url_is_rebuilt(self):
        record = normalize_channel_record({"id": "@Foo", "title": "<b>x</b>"})
        assert record.title == "<b>x</b>"
        assert record.url == "https://www.youtube.com/@Foo"


# -- normalize_gallery_catalog ------------------------------------------------


class TestNormalizeGalleryCatalog:
    @pytest.mark.parametrize(
        "payload", [None, [], {"videos": None}, [VID], {"videos": [VID]}],
        ids=["none", "empty", "videos-none", "list", "videos"],
    )
    def test_video_shapes_are_video_catalogs(self, payload):
        kind, records = normalize_gallery_catalog(payload)
        assert kind == "video"
        assert records == normalize_catalog(payload)
        assert all(isinstance(record, VideoRecord) for record in records)

    @pytest.mark.parametrize("entries", [None, []], ids=["none", "empty"])
    def test_an_empty_channels_key_is_an_empty_channel_catalog(self, entries):
        assert normalize_gallery_catalog({"channels": entries}) == ("channel", [])

    def test_a_channels_mapping_is_a_channel_catalog(self):
        kind, records = normalize_gallery_catalog({"channels": ["@Foo", UC_A]})
        assert kind == "channel"
        assert [record.id for record in records] == ["@foo", UC_A]
        assert all(isinstance(record, ChannelRecord) for record in records)

    @pytest.mark.parametrize(
        "payload",
        [{"channels": [], "videos": []}, {"channels": None, "videos": None},
         {"channels": ["@Foo"], "videos": [VID]}],
        ids=["both-empty", "both-none", "both-filled"],
    )
    def test_mixing_videos_and_channels_is_rejected(self, payload):
        with pytest.raises(CatalogError, match=r"^page\.rst: use either 'videos' or"):
            normalize_gallery_catalog(payload, "page.rst")

    @pytest.mark.parametrize(
        "entries, name", [("@Foo", "str"), ({"a": 1}, "dict"), (5, "int")]
    )
    def test_channels_must_be_a_list(self, entries, name):
        with pytest.raises(CatalogError) as caught:
            normalize_gallery_catalog({"channels": entries}, "page.rst")
        assert str(caught.value) == f"page.rst: 'channels' must be a list, got {name}"

    def test_a_bad_channel_is_located_by_origin_and_index(self):
        with pytest.raises(CatalogError, match=r"^page\.rst: record 1: expected"):
            normalize_gallery_catalog({"channels": ["@Foo", 5]}, "page.rst")

    def test_a_bad_video_is_located_by_origin_and_index(self):
        with pytest.raises(CatalogError, match=r"^page\.rst: record 0: expected"):
            normalize_gallery_catalog({"videos": [5]}, "page.rst")

    def test_a_channel_catalog_over_the_item_limit_is_rejected(self):
        limit = model.MAX_COLLECTION_ITEMS
        with pytest.raises(CatalogError, match=f"limit is {limit:,}"):
            normalize_gallery_catalog({"channels": ["@Foo"] * (limit + 1)})

    def test_duplicate_handles_collapse_to_the_first_case_insensitively(self):
        _, records = normalize_gallery_catalog(
            {"channels": ["@Foo", "@foo", {"id": "@FOO", "title": "later"}, "@Bar"]}
        )
        assert [(record.id, record.title) for record in records] == [
            ("@foo", "@Foo"),
            ("@bar", "@Bar"),
        ]

    def test_legacy_names_collapse_case_insensitively(self):
        _, records = normalize_gallery_catalog(
            {
                "channels": [
                    "https://www.youtube.com/Name",
                    "https://www.youtube.com/c/name",
                ]
            }
        )
        assert [record.id for record in records] == ["name"]

    def test_one_handle_paired_with_two_channel_ids_is_rejected(self):
        payload = {
            "channels": [
                {"handle": "Foo", "channel_id": UC_A},
                {"handle": "foo", "channel_id": UC_B},
            ]
        }
        with pytest.raises(CatalogError) as caught:
            normalize_gallery_catalog(payload, "page.rst")
        message = str(caught.value)
        assert message.startswith("page.rst: channel handle @foo is paired with")
        assert f"({UC_A}, {UC_B})" in message

    def test_one_channel_id_paired_with_two_handles_is_rejected(self):
        payload = {
            "channels": [
                {"handle": "Foo", "channel_id": UC_A},
                {"handle": "Bar", "channel_id": UC_A},
            ]
        }
        with pytest.raises(CatalogError) as caught:
            normalize_gallery_catalog(payload, "page.rst")
        message = str(caught.value)
        assert message.startswith(f"page.rst: channel_id {UC_A} is paired with")
        assert "(@Bar, @Foo)" in message

    def test_the_same_pair_in_two_spellings_is_one_channel(self):
        _, records = normalize_gallery_catalog(
            {
                "channels": [
                    {"handle": "Foo", "channel_id": UC_A},
                    {"handle": "foo", "channel_id": UC_A},
                ]
            }
        )
        assert [(record.id, record.handle) for record in records] == [(UC_A, "Foo")]

    @pytest.mark.parametrize(
        "entries",
        list(
            itertools.permutations(
                ["@Foo", UC_A, {"handle": "Foo", "channel_id": UC_A}]
            )
        ),
        ids=lambda entries: "-".join(
            "pair" if isinstance(entry, dict) else entry[:3] for entry in entries
        ),
    )
    def test_alias_resolution_does_not_depend_on_source_order(self, entries):
        # Whichever spelling comes first, the surviving card carries the
        # stable UC identity and the friendlier @handle link and label.
        _, records = normalize_gallery_catalog({"channels": list(entries)})
        assert len(records) == 1
        record = records[0]
        assert (record.id, record.channel_id, record.handle) == (UC_A, UC_A, "Foo")
        assert record.url == "https://www.youtube.com/@Foo"
        assert record.title == record.channel == "@Foo"

    def test_an_authored_title_survives_alias_resolution(self):
        _, records = normalize_gallery_catalog(
            {
                "channels": [
                    {"id": UC_A, "title": "Authored"},
                    {"handle": "Foo", "channel_id": UC_A},
                ]
            }
        )
        assert [(record.title, record.url) for record in records] == [
            ("Authored", "https://www.youtube.com/@Foo")
        ]

    def test_unrelated_channels_keep_source_order(self):
        _, records = normalize_gallery_catalog({"channels": ["@Zed", UC_B, "@Alpha"]})
        assert [record.id for record in records] == ["@zed", UC_B, "@alpha"]


# -- derive_channel_records ---------------------------------------------------


class TestDeriveChannelRecords:
    def test_no_videos_project_to_no_channels(self):
        assert derive_channel_records([]) == []

    @pytest.mark.parametrize(
        "channel",
        ["", "Some Display Name", "Foo / Bar", UC_A, "@", "@a b", "<script>"],
        ids=["empty", "display-name", "slash-name", "uc-as-name", "bare-at",
             "bad-handle", "html"],
    )
    def test_a_video_without_linkable_identity_is_skipped(self, channel):
        # A display name alone cannot produce a trustworthy link.
        assert derive_channel_records([video(0, channel=channel)]) == []

    @pytest.mark.parametrize(
        "channel",
        [
            "https://evil.example/@Foo",
            "https://youtube.com.evil.example/@Foo",
            f"https://youtu.be/{VID}",
            f"https://www.youtube.com/playlist?list={PLAYLIST}",
            "https://www.youtube.com/c/Name",
        ],
        ids=["foreign-host", "suffix-spoof", "video-url", "playlist-url", "vanity"],
    )
    def test_a_url_that_names_no_handle_or_id_is_skipped(self, channel):
        assert derive_channel_records([video(0, channel=channel)]) == []

    def test_an_explicit_handle_makes_a_channel(self):
        (channel,) = derive_channel_records(
            [video(0, handle="Foo", published="2024-05-01", tags=["a"])]
        )
        assert channel == ChannelRecord(
            id="@foo",
            title="@Foo",
            channel="@Foo",
            handle="Foo",
            url="https://www.youtube.com/@Foo",
            tags=["a"],
            published=dt.datetime(2024, 5, 1, tzinfo=UTC),
            video_count=1,
        )

    @pytest.mark.parametrize(
        "channel",
        ["@Foo", "https://www.youtube.com/@Foo", "https://www.youtube.com/@Foo/videos"],
        ids=["at-handle", "handle-url", "handle-tab-url"],
    )
    def test_a_handle_in_the_channel_field_is_a_fallback(self, channel):
        (derived,) = derive_channel_records([video(0, channel=channel)])
        assert (derived.id, derived.handle, derived.title) == ("@foo", "Foo", "@Foo")
        assert derived.url == "https://www.youtube.com/@Foo"

    def test_a_channel_id_alone_makes_a_channel(self):
        (derived,) = derive_channel_records([video(0, channel_id=UC_A)])
        assert (derived.id, derived.channel_id, derived.handle) == (UC_A, UC_A, "")
        assert derived.url == f"https://www.youtube.com/channel/{UC_A}"
        assert derived.title == UC_A

    def test_a_display_name_labels_a_channel_with_stable_identity(self):
        (derived,) = derive_channel_records(
            [video(0, channel_id=UC_A, channel="StatQuest")]
        )
        assert derived.title == derived.channel == "StatQuest"

    @pytest.mark.parametrize(
        "channel",
        ["https://evil.example/x", f"https://youtu.be/{VID}", UC_B,
         f"https://www.youtube.com/channel/{UC_B}"],
        ids=["foreign-url", "video-url", "other-uc-id", "channel-id-url"],
    )
    def test_a_raw_url_or_id_is_never_used_as_the_label(self, channel):
        (derived,) = derive_channel_records(
            [video(0, channel_id=UC_A, channel=channel)]
        )
        assert derived.title == UC_A
        assert channel not in (derived.title, derived.channel, derived.url)

    def test_the_link_is_built_from_validated_identity_only(self):
        (derived,) = derive_channel_records(
            [video(0, channel_id=UC_A, channel="<script>alert(1)</script>")]
        )
        assert derived.url == f"https://www.youtube.com/channel/{UC_A}"

    def test_videos_of_one_channel_are_counted_and_merged(self):
        (derived,) = derive_channel_records(
            [
                video(0, handle="Foo", tags=["a"], published="2022-01-01"),
                video(1, handle="foo", tags=["A", "b"], published="2024-01-01"),
                video(2, handle="FOO", tags=["b", "c"]),
            ]
        )
        assert derived.video_count == 3
        # Tags are unioned case-insensitively, first spelling and order kept.
        assert derived.tags == ["a", "b", "c"]
        assert derived.published == dt.datetime(2024, 1, 1, tzinfo=UTC)
        assert derived.year == "2024"

    def test_published_stays_none_when_no_video_is_dated(self):
        (derived,) = derive_channel_records([video(0, handle="Foo")])
        assert derived.published is None

    def test_channels_keep_first_appearance_order(self):
        derived = derive_channel_records(
            [
                video(0, handle="Zed"),
                video(1, channel_id=UC_A),
                video(2, handle="zed"),
                video(3, handle="Alpha"),
            ]
        )
        assert [channel.id for channel in derived] == ["@zed", UC_A, "@alpha"]
        assert [channel.video_count for channel in derived] == [2, 1, 1]

    def test_a_handle_only_video_joins_the_channel_id_learned_elsewhere(self):
        videos = [
            video(0, handle="Foo"),
            video(1, handle="foo", channel_id=UC_A, channel="Nice Name"),
        ]
        for ordering in (videos, videos[::-1]):
            (derived,) = derive_channel_records(ordering)
            assert derived.id == derived.channel_id == UC_A
            assert derived.video_count == 2
            assert derived.title == "Nice Name"
            assert derived.url == "https://www.youtube.com/@foo"

    def test_projection_identity_is_independent_of_input_order(self):
        videos = [
            video(0, handle="Foo", tags=["x"]),
            video(1, handle="foo", channel_id=UC_A),
            video(2, channel_id=UC_A, published="2023-06-01"),
            video(3, channel_id=UC_B, handle="Bar"),
            video(4, channel="Plain Name"),
        ]
        expected = {UC_A: 3, UC_B: 1}
        for ordering in itertools.permutations(videos):
            derived = derive_channel_records(list(ordering))
            assert {channel.id: channel.video_count for channel in derived} == expected

    def test_an_ambiguous_handle_never_merges_two_stable_ids(self):
        derived = derive_channel_records(
            [
                video(0, handle="Foo", channel_id=UC_A),
                video(1, handle="Foo", channel_id=UC_B),
                video(2, handle="Foo"),
            ]
        )
        assert [channel.id for channel in derived] == [UC_A, UC_B, "@foo"]
        assert [channel.video_count for channel in derived] == [1, 1, 1]

    def test_the_newest_handle_and_label_win_after_a_rename(self):
        (derived,) = derive_channel_records(
            [
                video(0, handle="New", channel_id=UC_A, channel="New Name",
                      published="2024-01-01"),
                video(1, handle="Old", channel_id=UC_A, channel="Old Name",
                      published="2020-01-01"),
                video(2, channel_id=UC_A),
            ]
        )
        assert derived.id == UC_A
        assert derived.handle == "New"
        assert derived.url == "https://www.youtube.com/@New"
        assert derived.title == "New Name"
        assert derived.video_count == 3

    def test_later_catalog_order_breaks_a_date_tie(self):
        (derived,) = derive_channel_records(
            [
                video(0, handle="First", channel_id=UC_A),
                video(1, handle="Second", channel_id=UC_A),
            ]
        )
        assert derived.handle == "Second"

    def test_a_later_display_name_upgrades_a_default_handle_label(self):
        (derived,) = derive_channel_records(
            [video(0, handle="foo"), video(1, handle="foo", channel="Nice Name")]
        )
        assert derived.title == derived.channel == "Nice Name"
        assert derived.url == "https://www.youtube.com/@foo"

    @pytest.mark.xfail(
        strict=True,
        reason="the upgrade compares the label with the case-folded identity, so "
        "a mixed-case handle keeps '@Handle' unless the named record comes first",
    )
    def test_the_label_of_a_handle_only_channel_ignores_record_order(self):
        videos = [video(0, handle="Foo"), video(1, handle="Foo", channel="Nice Name")]
        forward = derive_channel_records(videos)[0].title
        backward = derive_channel_records(videos[::-1])[0].title
        assert forward == backward == "Nice Name"

    def test_the_input_records_are_not_mutated(self):
        videos = [video(0, handle="Foo", tags=["a"]), video(1, handle="Foo", tags=["b"])]
        before = [dataclasses.replace(record, tags=list(record.tags)) for record in videos]
        derive_channel_records(videos)
        assert videos == before

    def test_derived_channels_carry_no_custom_fields(self):
        (derived,) = derive_channel_records(
            [video(0, handle="Foo", fields={"category": "x"})]
        )
        assert derived.fields == {}


class TestModuleSurface:
    def test_public_names_are_exported(self):
        for name in model.__all__:
            assert hasattr(model, name), name

    def test_the_model_needs_neither_sphinx_nor_docutils(self):
        # "deliberately free of Sphinx and network imports"
        imported = {
            getattr(value, "__name__", "").split(".")[0]
            for value in vars(model).values()
            if type(value).__name__ == "module"
        }
        assert not imported & {"sphinx", "docutils", "requests", "urllib", "socket"}
