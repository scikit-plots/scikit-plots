"""
Tests for the YouTube URL grammar in ``_sphinx_youtube_core.reference``.

Every expectation is taken from the module's docstrings: a reference is
parsed structurally, the host is judged on a dot boundary, identifiers are
validated by anchored shape, unknown tabs and playlist families are values
rather than errors, and anything that is not a YouTube reference raises
``ReferenceError``. No test touches the network.
"""

from __future__ import annotations

import dataclasses
from urllib.parse import quote

import pytest

from .. import reference as ref_module
from ..reference import (
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
    is_reference_url,
    parse_reference,
    parse_video_reference,
    validate_channel_id,
    validate_handle,
    validate_playlist_id,
)

VID = "nfYOp3_SyqM"
VID2 = "hbT7vzCvEc8"
PLID = "PLXOJEg4xbr50"
CHID = "UCcabW7890RKJzL968QWEykA"
WATCH = f"https://www.youtube.com/watch?v={VID}"


# -- module surface -----------------------------------------------------------


def test_all_names_are_importable():
    for name in ref_module.__all__:
        assert hasattr(ref_module, name), name


def test_all_is_unique_and_lists_the_public_api():
    names = list(ref_module.__all__)
    assert len(names) == len(set(names))
    assert {
        "parse_reference",
        "parse_video_reference",
        "is_reference_url",
        "validate_handle",
        "validate_channel_id",
        "validate_playlist_id",
        "YouTubeReference",
        "ReferenceError",
    } <= set(names)


def test_kind_constants_are_distinct_strings():
    kinds = [VIDEO, PLAYLIST, CHANNEL, CLIP, POST, SEARCH, HASHTAG, FEED]
    assert all(isinstance(kind, str) and kind for kind in kinds)
    assert len(set(kinds)) == len(kinds)


def test_reference_error_is_a_value_error():
    assert issubclass(ReferenceError, ValueError)
    assert ReferenceError is ref_module.ReferenceError


# -- validate_handle ----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("cs50", "cs50", id="plain"),
        pytest.param("@cs50", "cs50", id="leading-at"),
        pytest.param("  @cs50  ", "cs50", id="surrounding-space"),
        pytest.param("a", "a", id="one-char"),
        pytest.param("a" * 30, "a" * 30, id="max-30"),
        pytest.param("@" + "a" * 30, "a" * 30, id="at-not-counted"),
        pytest.param("a-b.c_d", "a-b.c_d", id="ascii-separators"),
        pytest.param("caf·e", "caf·e", id="middle-dot"),
        pytest.param("日本語", "日本語", id="cjk"),
        pytest.param("١٢٣", "١٢٣", id="non-ascii-digits"),
        pytest.param("é", "é", id="nfc-normalised"),
        pytest.param("CS50", "CS50", id="case-preserved"),
    ],
)
def test_validate_handle_accepts(value, expected):
    assert validate_handle(value) == expected


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        pytest.param("", "empty", id="empty"),
        pytest.param("   ", "empty", id="whitespace"),
        pytest.param("@", "empty", id="at-only"),
        pytest.param(None, "empty", id="none"),
        pytest.param(0, "empty", id="zero"),
        pytest.param("a" * 31, "more than 30", id="31-chars"),
        pytest.param("_a", "first or last", id="leading-underscore"),
        pytest.param("a_", "first or last", id="trailing-underscore"),
        pytest.param(".a", "first or last", id="leading-dot"),
        pytest.param("a-", "first or last", id="trailing-hyphen"),
        pytest.param("·a", "first or last", id="leading-middle-dot"),
        pytest.param("-", "first or last", id="separator-only"),
        pytest.param("a b", "unsupported character", id="inner-space"),
        pytest.param("a/b", "unsupported character", id="slash"),
        pytest.param("a?b", "unsupported character", id="question-mark"),
        pytest.param("a#b", "unsupported character", id="hash"),
        pytest.param("@@a", "unsupported character", id="double-at"),
        pytest.param("a\U0001f600", "unsupported character", id="emoji"),
        pytest.param("a\nb", "unsupported character", id="newline"),
        pytest.param("a<b", "unsupported character", id="angle-bracket"),
    ],
)
def test_validate_handle_rejects(value, fragment):
    with pytest.raises(ReferenceError, match=fragment):
        validate_handle(value)


def test_validate_handle_error_names_the_input():
    with pytest.raises(ReferenceError) as info:
        validate_handle("bad!name")
    assert "bad!name" in str(info.value)
    assert "'!'" in str(info.value)


# -- validate_channel_id ------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(CHID, id="canonical"),
        pytest.param(f"  {CHID}\n", id="surrounding-space"),
        pytest.param("UC" + "-_" * 11, id="url-safe-punctuation"),
    ],
)
def test_validate_channel_id_accepts(value):
    assert validate_channel_id(value) == value.strip()


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),
        pytest.param(None, id="none"),
        pytest.param(CHID[:-1], id="23-chars"),
        pytest.param(CHID + "A", id="25-chars"),
        pytest.param("uc" + CHID[2:], id="lowercase-prefix"),
        pytest.param("UU" + CHID[2:], id="uploads-prefix"),
        pytest.param(CHID[:-1] + "!", id="bad-character"),
        pytest.param(CHID[:10] + "\n" + CHID[11:], id="inner-newline"),
        pytest.param(CHID[:-1] + "é", id="non-ascii"),
        pytest.param(f"https://www.youtube.com/channel/{CHID}", id="url"),
        pytest.param(12345, id="int"),
    ],
)
def test_validate_channel_id_rejects(value):
    with pytest.raises(ReferenceError, match="not a valid YouTube channel id"):
        validate_channel_id(value)


# -- validate_playlist_id -----------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(PLID, id="user"),
        pytest.param("PLabcdefgh", id="minimum-10"),
        pytest.param("WL", id="watch-later"),
        pytest.param("LL", id="liked"),
        pytest.param("ZZsomethingnew", id="unknown-prefix"),
        pytest.param("OLAK5uy_abc-DEF_123", id="album"),
        pytest.param(f"  {PLID}  ", id="surrounding-space"),
    ],
)
def test_validate_playlist_id_accepts(value):
    assert validate_playlist_id(value) == value.strip()


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        pytest.param("", "not a valid playlist id", id="empty"),
        pytest.param(None, "not a valid playlist id", id="none"),
        pytest.param("PL", "too short", id="bare-prefix"),
        pytest.param("PLabcdefg", "too short", id="9-chars"),
        pytest.param("UU", "too short", id="two-char-not-special"),
        pytest.param("PL abcdefghij", "not a valid playlist id", id="space"),
        pytest.param("PLabc\ndefghij", "not a valid playlist id", id="inner-newline"),
        pytest.param("PLabcdefgh!", "not a valid playlist id", id="punctuation"),
        pytest.param("PLabcdefghé", "not a valid playlist id", id="non-ascii"),
        pytest.param("PLabc/defgh", "not a valid playlist id", id="slash"),
    ],
)
def test_validate_playlist_id_rejects(value, fragment):
    with pytest.raises(ReferenceError, match=fragment):
        validate_playlist_id(value)


# -- playlist classification --------------------------------------------------


@pytest.mark.parametrize(
    ("playlist_id", "kind", "ephemeral"),
    [
        pytest.param(PLID, "user", False, id="PL-user"),
        pytest.param("UUcabW7890RKJzL968QWEykA", "uploads", False, id="UU-uploads"),
        pytest.param("FLcabW7890RKJzL968QWEykA", "favourites", False, id="FL"),
        pytest.param("SPabcdefghij", "legacy", False, id="SP-legacy"),
        pytest.param("OLAK5uy_abcdefgh", "album", False, id="OLAK-album"),
        pytest.param("ZZsomethingnew", "unknown", False, id="unknown"),
        pytest.param("RD" + VID, "mix", True, id="RD-mix"),
        pytest.param("RDMM" + VID, "mix", True, id="RDMM-mix"),
        pytest.param("RDCLAK5uy_abcdef", "mix", True, id="RDCLAK-mix"),
        pytest.param("UL" + VID, "mix", True, id="UL-mix"),
        pytest.param("WL", "watch-later", True, id="WL"),
        pytest.param("LL", "liked", True, id="LL"),
    ],
)
def test_playlist_url_is_classified(playlist_id, kind, ephemeral):
    ref = parse_reference(f"https://www.youtube.com/playlist?list={playlist_id}")
    assert ref.kind == PLAYLIST
    assert ref.playlist_id == playlist_id
    assert ref.playlist_kind == kind
    assert ref.ephemeral is ephemeral
    assert ref.is_collection
    assert ref.video_id == ""
    assert ref.watch_url == ""


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        pytest.param("PLXOJEg4xbr50", "user", id="docstring-user"),
        pytest.param("RDdQw4w9WgXcQ", "mix", id="docstring-mix"),
        pytest.param("ZZsomethingnew", "unknown", id="docstring-unknown"),
        pytest.param("", "unknown", id="empty"),
    ],
)
def test_classify_playlist(value, kind):
    assert ref_module._classify_playlist(value) == kind


# -- start offsets ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("90", 90, id="bare-seconds"),
        pytest.param("0", 0, id="zero"),
        pytest.param(" 90 ", 90, id="padded"),
        pytest.param("90s", 90, id="seconds-suffix"),
        pytest.param("2m", 120, id="minutes"),
        pytest.param("1h", 3600, id="hours"),
        pytest.param("1m30s", 90, id="minutes-seconds"),
        pytest.param("1h2m30s", 3750, id="full-compound"),
        pytest.param("1h30s", 3630, id="hours-seconds"),
        pytest.param("1H2M3S", 3723, id="uppercase"),
        pytest.param("", None, id="empty"),
        pytest.param("   ", None, id="whitespace"),
        pytest.param("bogus", None, id="words"),
        pytest.param("-5", None, id="negative"),
        pytest.param("1.5", None, id="decimal"),
        pytest.param("30s1m", None, id="wrong-order"),
        pytest.param("1m 30s", None, id="inner-space"),
        pytest.param("hms", None, id="units-only"),
        pytest.param("1:30", None, id="clock-notation"),
    ],
)
def test_parse_start(value, expected):
    assert ref_module._parse_start(value) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("t=90", 90, id="t"),
        pytest.param("start=42", 42, id="start"),
        pytest.param("time_continue=7", 7, id="time_continue"),
        pytest.param("t=90&start=5", 90, id="t-beats-start"),
        pytest.param("start=5&t=90", 90, id="priority-not-url-order"),
        pytest.param("start=5&time_continue=7", 5, id="start-beats-continue"),
        pytest.param("t=bogus&start=5", 5, id="bad-t-falls-through"),
        pytest.param("t=bogus", None, id="bad-t-dropped"),
        pytest.param("si=abc", None, id="absent"),
    ],
)
def test_start_offset_from_query(query, expected):
    ref = parse_reference(f"{WATCH}&{query}")
    assert ref.video_id == VID
    assert ref.start == expected


@pytest.mark.parametrize("param", ["t", "start", "index"])
def test_non_decimal_digit_parameter_is_dropped_not_raised(param):
    # Documented: an unparsable offset is decoration and is dropped, never
    # raised on. A superscript two is a "digit" to str.isdigit() only.
    ref = parse_reference(f"{WATCH}&{param}=%C2%B2")
    assert ref.video_id == VID
    assert ref.start is None
    assert ref.index is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("index=2", 2, id="plain"),
        pytest.param("index=0", 0, id="zero"),
        pytest.param("index=x", None, id="letters"),
        pytest.param("index=-1", None, id="negative"),
        pytest.param("index=1.5", None, id="decimal"),
        pytest.param("si=abc", None, id="absent"),
    ],
)
def test_index_from_query(query, expected):
    ref = parse_reference(f"{WATCH}&list={PLID}&{query}")
    assert ref.index == expected


# -- host recognition ---------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "a.b.youtube.com",
        "youtu.be",
        "www.youtube-nocookie.com",
        "youtubekids.com",
        "www.youtubeeducation.com",
        "youtube.de",
        "www.youtube.de",
        "youtube.co.uk",
        "www.youtube.co.uk",
    ],
)
def test_is_youtube_host_accepts(host):
    assert ref_module._is_youtube_host(host) is True


@pytest.mark.parametrize(
    "host",
    [
        pytest.param("", id="empty"),
        pytest.param("youtube.com.evil.example", id="suffix-attack"),
        pytest.param("notyoutube.com", id="prefix-lookalike"),
        pytest.param("fakeyoutube.de", id="cctld-lookalike"),
        pytest.param("youtube.com-evil.example", id="hyphen-lookalike"),
        pytest.param("youtu.be.evil.example", id="short-suffix-attack"),
        pytest.param("xyoutu.be", id="short-prefix-lookalike"),
        pytest.param("youtube.evil", id="long-fake-tld"),
        pytest.param("youtube.co.uk.evil.example", id="cctld-suffix-attack"),
        pytest.param("youtube", id="no-tld"),
        pytest.param("example.com", id="unrelated"),
        pytest.param("yоutube.com", id="cyrillic-homoglyph"),
    ],
)
def test_is_youtube_host_rejects(host):
    assert ref_module._is_youtube_host(host) is False


@pytest.mark.parametrize(
    ("url", "host"),
    [
        pytest.param("https://WWW.YouTube.COM/x", "www.youtube.com", id="lowercased"),
        pytest.param("https://www.youtube.com:443/x", "www.youtube.com", id="port"),
        pytest.param("https://u:p@www.youtube.com/x", "www.youtube.com", id="userinfo"),
        pytest.param(
            "https://www.youtube.com@evil.example/x", "evil.example", id="at-trick"
        ),
        pytest.param("not a url", "", id="no-authority"),
    ],
)
def test_host_of(url, host):
    assert ref_module._host_of(url) == host


@pytest.mark.parametrize(
    ("url", "host"),
    [
        pytest.param(f"https://youtube.com/watch?v={VID}", "youtube.com", id="apex"),
        pytest.param(f"http://youtube.com/watch?v={VID}", "youtube.com", id="http"),
        pytest.param(f"https://m.youtube.com/watch?v={VID}", "m.youtube.com", id="m"),
        pytest.param(
            f"https://music.youtube.com/watch?v={VID}", "music.youtube.com", id="music"
        ),
        pytest.param(
            f"https://www.youtubekids.com/watch?v={VID}",
            "www.youtubekids.com",
            id="kids",
        ),
        pytest.param(f"https://youtube.de/watch?v={VID}", "youtube.de", id="cctld"),
        pytest.param(
            f"https://www.youtube.co.uk/watch?v={VID}",
            "www.youtube.co.uk",
            id="co-cctld",
        ),
        pytest.param(
            f"HTTPS://WWW.YOUTUBE.COM/watch?v={VID}", "www.youtube.com", id="uppercase"
        ),
        pytest.param(
            f"https://www.youtube.com:443/watch?v={VID}", "www.youtube.com", id="port"
        ),
        pytest.param(
            f"https://user:pw@www.youtube.com/watch?v={VID}",
            "www.youtube.com",
            id="credentials-stripped",
        ),
        pytest.param(f"www.youtube.com/watch?v={VID}", "www.youtube.com", id="no-scheme"),
        pytest.param(f"//www.youtube.com/watch?v={VID}", "www.youtube.com", id="relative"),
    ],
)
def test_watch_url_on_any_youtube_host(url, host):
    ref = parse_reference(url)
    assert ref.kind == VIDEO
    assert ref.video_id == VID
    assert ref.host == host
    assert ref.raw == url
    assert ref.watch_url == WATCH


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(f"https://youtube.com.evil.example/watch?v={VID}", id="suffix"),
        pytest.param(f"https://evil.example/youtube.com/watch?v={VID}", id="in-path"),
        pytest.param(f"https://notyoutube.com/watch?v={VID}", id="prefix-lookalike"),
        pytest.param(f"https://fakeyoutube.de/watch?v={VID}", id="cctld-lookalike"),
        pytest.param(f"https://www.youtube.com@evil.example/watch?v={VID}", id="at"),
        pytest.param(
            f"https://www.youtube.com:pw@evil.example/watch?v={VID}", id="userinfo"
        ),
        pytest.param(f"https://evil.example/?x=youtube.com/watch?v={VID}", id="query"),
        pytest.param(f"https://evil.example/#youtube.com/watch?v={VID}", id="fragment"),
        pytest.param(f"https://vimeo.com/{VID}", id="other-provider"),
        pytest.param(f"https://yоutube.com/watch?v={VID}", id="homoglyph"),
        pytest.param(f"https://youtube.com-evil.example/watch?v={VID}", id="hyphen"),
    ],
)
def test_foreign_host_is_rejected(url):
    with pytest.raises(ReferenceError, match="is not a YouTube URL"):
        parse_reference(url)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(f"javascript:alert(1)//youtube.com/watch?v={VID}", id="javascript"),
        pytest.param(f"data:text/html,youtube.com/watch?v={VID}", id="data"),
        pytest.param(f"mailto:a@youtube.com/watch?v={VID}", id="mailto"),
    ],
)
def test_opaque_scheme_is_rejected(url):
    with pytest.raises(ReferenceError):
        parse_reference(url)


def test_canonical_watch_url_never_reuses_the_input_scheme_or_host():
    ref = parse_reference(f"http://user:pw@m.youtube.com:8080/watch?v={VID}&t=5")
    assert ref.watch_url == WATCH


# -- is_reference_url ---------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("https://www.youtube.com/@cs50", id="https"),
        pytest.param("www.youtube.com/@cs50", id="schemeless-host"),
        pytest.param("//youtu.be/hKpAHgT9VxM", id="protocol-relative"),
        pytest.param("https://evil.example/youtube.com/x", id="foreign-host"),
        pytest.param("javascript:alert(1)//x.y/", id="javascript"),
        pytest.param("  https://youtu.be/x  ", id="padded"),
        pytest.param("HTTPS://YOUTU.BE/x", id="uppercase-scheme"),
        pytest.param("ftp://example.com", id="ftp-no-path"),
        pytest.param("a.b/", id="minimal-dotted-host"),
    ],
)
def test_is_reference_url_true(value):
    assert is_reference_url(value) is True


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("@cs50", id="handle"),
        pytest.param("CS50", id="display-name"),
        pytest.param("Foo / Bar", id="name-with-slash"),
        pytest.param("CS50: Introduction to Computer Science", id="name-with-colon"),
        pytest.param(CHID, id="channel-id"),
        pytest.param(VID, id="video-id"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("youtube.com", id="host-without-path"),
        pytest.param(None, id="none"),
        pytest.param(42, id="int"),
        pytest.param(b"https://youtu.be/x", id="bytes"),
        pytest.param(["https://youtu.be/x"], id="list"),
    ],
)
def test_is_reference_url_false(value):
    assert is_reference_url(value) is False


# -- bare identifiers ---------------------------------------------------------


def test_bare_video_id():
    ref = parse_reference(VID)
    assert (ref.kind, ref.video_id, ref.raw) == (VIDEO, VID, VID)
    assert ref.host == ""
    assert ref.params == {}
    assert not ref.is_collection


def test_bare_channel_id():
    ref = parse_reference(CHID)
    assert (ref.kind, ref.channel_id) == (CHANNEL, CHID)
    assert ref.channel_ref == CHID
    assert ref.is_collection


@pytest.mark.parametrize(
    ("value", "handle"),
    [
        pytest.param("@cs50", "cs50", id="ascii"),
        pytest.param("@caf·e", "caf·e", id="middle-dot"),
        pytest.param("@日本語", "日本語", id="cjk"),
        pytest.param("@a_b-c", "a_b-c", id="separators"),
        pytest.param("  @cs50 ", "cs50", id="padded"),
    ],
)
def test_bare_handle(value, handle):
    ref = parse_reference(value)
    assert (ref.kind, ref.handle) == (CHANNEL, handle)
    assert ref.channel_ref == handle
    assert ref.tab == ""
    assert ref.tab_known is True


def test_bare_handle_with_dot_agrees_with_validate_handle():
    assert validate_handle("@john.doe") == "john.doe"
    ref = parse_reference("@john.doe")
    assert (ref.kind, ref.handle) == (CHANNEL, "john.doe")


def test_handle_with_dot_parses_inside_a_url():
    ref = parse_reference("https://www.youtube.com/@john.doe")
    assert (ref.kind, ref.handle) == (CHANNEL, "john.doe")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("@", id="at-only"),
        pytest.param("@bad!name", id="bad-character"),
        pytest.param("@_lead", id="leading-separator"),
        pytest.param("@" + "a" * 31, id="too-long"),
    ],
)
def test_bare_handle_invalid(value):
    with pytest.raises(ReferenceError, match="not a valid YouTube handle"):
        parse_reference(value)


@pytest.mark.parametrize(
    ("value", "kind", "ephemeral"),
    [
        pytest.param(PLID, "user", False, id="user"),
        pytest.param("OLAK5uy_abcdefgh", "album", False, id="album"),
        pytest.param("RD" + VID, "mix", True, id="mix"),
        pytest.param("UUcabW7890RKJzL968QWEykA", "uploads", False, id="uploads"),
    ],
)
def test_bare_playlist_id_with_known_prefix(value, kind, ephemeral):
    ref = parse_reference(value)
    assert (ref.kind, ref.playlist_id, ref.playlist_kind) == (PLAYLIST, value, kind)
    assert ref.ephemeral is ephemeral


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("hello", id="short-word"),
        pytest.param(VID[:-1], id="10-char-id"),
        pytest.param(VID + "M", id="12-char-id"),
        pytest.param("ZZsomethingnew", id="unknown-prefix-playlist"),
        pytest.param("PLabc", id="truncated-playlist"),
        pytest.param(CHID[:-1], id="23-char-channel"),
        pytest.param("two words", id="space"),
        pytest.param("日本語", id="non-ascii"),
    ],
)
def test_bare_unrecognised_is_rejected(value):
    with pytest.raises(ReferenceError, match="not a recognised YouTube reference"):
        parse_reference(value)


def test_bare_error_names_input_and_accepted_forms():
    with pytest.raises(ReferenceError) as info:
        parse_reference("hello")
    message = str(info.value)
    assert "'hello'" in message
    for form in ("video id", "playlist id", "channel id", "handle"):
        assert form in message


# -- non-string and empty input -----------------------------------------------


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        pytest.param(None, "NoneType", id="none"),
        pytest.param(42, "int", id="int"),
        pytest.param(b"https://youtu.be/nfYOp3_SyqM", "bytes", id="bytes"),
        pytest.param(["x"], "list", id="list"),
        pytest.param(1.5, "float", id="float"),
    ],
)
def test_non_string_is_rejected(value, type_name):
    with pytest.raises(ReferenceError, match=f"expected a string.*{type_name}"):
        parse_reference(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces"),
        pytest.param("\n\t", id="newline-tab"),
        pytest.param("<>", id="angle-brackets"),
        pytest.param("“”", id="curly-quotes"),
        pytest.param("​﻿", id="zero-width-only"),
        pytest.param("...", id="punctuation-only"),
    ],
)
def test_empty_reference_is_rejected(value):
    with pytest.raises(ReferenceError, match="empty reference"):
        parse_reference(value)


@pytest.mark.parametrize("value", ["''", '""', "' '", "<''>"])
def test_empty_quoted_reference_is_rejected_cleanly(value):
    with pytest.raises(ReferenceError, match="empty reference"):
        parse_reference(value)


# -- normalisation of pasted text ---------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(f"  https://youtu.be/{VID} \n", id="whitespace"),
        pytest.param(f"<https://youtu.be/{VID}>", id="angle-brackets"),
        pytest.param(f"'https://youtu.be/{VID}'", id="single-quotes"),
        pytest.param(f'"https://youtu.be/{VID}"', id="double-quotes"),
        pytest.param(f"“https://youtu.be/{VID}”", id="curly-quotes"),
        pytest.param(f"«https://youtu.be/{VID}»", id="guillemets"),
        pytest.param(f"https://youtu.be/{VID}.", id="trailing-period"),
        pytest.param(f"https://youtu.be/{VID}),", id="trailing-paren-comma"),
        pytest.param(f"(https://youtu.be/{VID})", id="parenthesised"),
        pytest.param(f"Watch https://youtu.be/{VID}.", id="in-sentence"),
        pytest.param(f"[link](https://youtu.be/{VID})", id="markdown-link"),
        pytest.param(f"see youtu.be/{VID}, please", id="schemeless-in-prose"),
        pytest.param(f"https://youtu.be/{VID[:6]}​{VID[6:]}", id="zero-width"),
        pytest.param(f"https://youtu.be/{VID[:6]}­{VID[6:]}", id="soft-hyphen"),
        pytest.param(f"﻿https://youtu.be/{VID}", id="bom"),
        pytest.param(f"youtu.be/{VID}", id="no-scheme"),
        pytest.param(f"YOUTU.BE/{VID}", id="uppercase-no-scheme"),
        pytest.param(f"//youtu.be/{VID}", id="protocol-relative"),
        pytest.param(f"'{VID}'", id="quoted-bare-id"),
    ],
)
def test_pasted_damage_is_repaired(value):
    ref = parse_reference(value)
    assert (ref.kind, ref.video_id) == (VIDEO, VID)
    assert ref.raw == value


def test_html_escaped_ampersands_are_decoded():
    ref = parse_reference(f"{WATCH}&amp;list={PLID}&amp;index=3")
    assert (ref.video_id, ref.playlist_id, ref.index) == (VID, PLID, 3)
    assert ref.params == {"v": VID, "list": PLID, "index": "3"}


def test_prose_extraction_ignores_a_foreign_url_before_the_youtube_one():
    ref = parse_reference(f"see https://evil.example/x and youtu.be/{VID}")
    assert (ref.video_id, ref.host) == (VID, "youtu.be")


def test_prose_without_a_youtube_url_is_rejected():
    with pytest.raises(ReferenceError):
        parse_reference("see https://evil.example/watch?v=" + VID + " for more")


@pytest.mark.parametrize(
    ("value", "kind", "video_id", "playlist_id"),
    [
        pytest.param(f"?v={VID}&", VIDEO, VID, "", id="question-v"),
        pytest.param(f"v={VID}", VIDEO, VID, "", id="v"),
        pytest.param(f"v={VID}&list={PLID}", VIDEO, VID, PLID, id="v-and-list"),
        pytest.param(f"list={PLID}", PLAYLIST, "", PLID, id="list"),
        pytest.param(f"?list={PLID}&index=2", PLAYLIST, "", PLID, id="list-index"),
    ],
)
def test_bare_query_fragment_is_completed(value, kind, video_id, playlist_id):
    ref = parse_reference(value)
    assert (ref.kind, ref.video_id, ref.playlist_id) == (kind, video_id, playlist_id)
    assert ref.host == "www.youtube.com"


def test_bare_query_fragment_without_identifier_is_rejected():
    with pytest.raises(ReferenceError, match="neither a 'v' nor a 'list'"):
        parse_reference("t=30")


# -- video forms --------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(f"https://www.youtube.com/watch?v={VID}", id="watch"),
        pytest.param(f"https://www.youtube.com/watch/{VID}", id="watch-path"),
        pytest.param(f"https://www.youtube.com/WATCH?v={VID}", id="watch-uppercase"),
        pytest.param(f"https://www.youtube.com/watch_popup?v={VID}", id="watch_popup"),
        pytest.param(f"https://youtu.be/{VID}", id="short-host"),
        pytest.param(f"https://youtu.be/{VID}?si=AbCdEf", id="short-host-tracking"),
        pytest.param(f"https://www.youtu.be/{VID}", id="short-host-subdomain"),
        pytest.param(f"https://www.youtube.com/embed/{VID}", id="embed"),
        pytest.param(f"https://www.youtube-nocookie.com/embed/{VID}", id="nocookie"),
        pytest.param(f"https://www.youtube.com/v/{VID}", id="v"),
        pytest.param(f"https://www.youtube.com/e/{VID}", id="e"),
        pytest.param(f"https://www.youtube.com/shorts/{VID}", id="shorts"),
        pytest.param(f"https://www.youtube.com/Shorts/{VID}", id="shorts-mixed-case"),
        pytest.param(f"https://www.youtube.com/live/{VID}", id="live"),
        pytest.param(f"https://www.youtube.com/live/{VID}?feature=share", id="live-q"),
        pytest.param(f"https://www.youtube.com/watch?v={VID}#t=30", id="fragment"),
        pytest.param(f"https://www.youtube.com/watch?feature=x&v={VID}", id="v-not-1st"),
        pytest.param(
            f"https://www.youtube.com/watch?v={VID}&si=a&pp=b&feature=share",
            id="tracking-params",
        ),
        pytest.param(
            f"https://www.youtube.com/watch_videos?video_ids={VID},{VID2}",
            id="ad-hoc-list",
        ),
    ],
)
def test_video_url_forms(url):
    ref = parse_reference(url)
    assert ref.kind == VIDEO
    assert ref.video_id == VID
    assert ref.watch_url == WATCH
    assert not ref.is_collection
    assert ref.describe() == f"the video {VID}"
    assert parse_video_reference(url) == ref


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(f"https://www.youtube.com/watch?v={VID}XXXXX", id="watch-16"),
        pytest.param(f"https://www.youtube.com/watch?v={VID[:-1]}", id="watch-10"),
        pytest.param(f"https://www.youtube.com/watch?v={VID[:5]}%20{VID[6:]}", id="sp"),
        pytest.param(f"https://www.youtube.com/watch?v={VID[:-1]}%C3%A9", id="unicode"),
        pytest.param(f"https://www.youtube.com/watch?v={VID[:-1]}!x", id="punct"),
        pytest.param(f"https://www.youtube.com/watch/{VID}X", id="watch-path-12"),
        pytest.param(f"https://youtu.be/{VID}X", id="short-host-12"),
        pytest.param(f"https://youtu.be/{VID[:-1]}", id="short-host-10"),
        pytest.param(f"https://www.youtube.com/embed/{VID}X", id="embed-12"),
        pytest.param(f"https://www.youtube.com/v/{VID[:-2]}", id="v-9"),
        pytest.param(f"https://www.youtube.com/shorts/{VID}X", id="shorts-12"),
        pytest.param(f"https://www.youtube.com/live/{VID[:-1]}", id="live-10"),
        pytest.param(
            f"https://www.youtube.com/watch_videos?video_ids=bad,{VID}", id="ad-hoc-bad"
        ),
    ],
)
def test_malformed_video_id_is_never_truncated(url):
    with pytest.raises(ReferenceError, match="not a valid YouTube video id"):
        parse_reference(url)


def test_video_id_error_reports_length_and_input():
    url = f"https://www.youtube.com/watch?v={VID}XXXXX"
    with pytest.raises(ReferenceError) as info:
        parse_reference(url)
    message = str(info.value)
    assert url in message
    assert "got 16" in message


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(f"https://www.youtube.com/watch?v={VID}%0A", id="watch-v"),
        pytest.param(f"https://youtu.be/{VID}%0A", id="short-host"),
        pytest.param(f"https://www.youtube.com/embed/{VID}%0A", id="embed"),
        pytest.param(f"https://www.youtube.com/playlist?list={PLID}%0A", id="list"),
        pytest.param(f"https://www.youtube.com/channel/{CHID}%0A", id="channel"),
    ],
)
def test_encoded_trailing_newline_is_not_part_of_an_identifier(url):
    try:
        ref = parse_reference(url)
    except ReferenceError:
        return
    for value in (ref.video_id, ref.playlist_id, ref.channel_id):
        assert "\n" not in value


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        pytest.param("https://www.youtube.com/watch", "neither a 'v'", id="watch"),
        pytest.param("https://www.youtube.com/watch?vi=" + VID, "neither", id="vi"),
        pytest.param("https://www.youtube.com/watch?v=", "neither a 'v'", id="blank-v"),
        pytest.param("https://youtu.be/", "carries no video id", id="short-host"),
        pytest.param("https://youtu.be", "carries no video id", id="short-host-bare"),
        pytest.param("https://www.youtube.com/embed/", "no video id", id="embed"),
        pytest.param("https://www.youtube.com/v", "no video id", id="v"),
        pytest.param("https://www.youtube.com/shorts", "no video id", id="shorts"),
        pytest.param("https://www.youtube.com/live/", "no video id", id="live"),
    ],
)
def test_video_url_without_id(url, fragment):
    with pytest.raises(ReferenceError, match=fragment):
        parse_reference(url)


def test_watch_with_video_and_playlist_keeps_both():
    ref = parse_reference(f"{WATCH}&list={PLID}&index=2")
    assert ref.kind == VIDEO
    assert (ref.video_id, ref.playlist_id, ref.playlist_kind) == (VID, PLID, "user")
    assert ref.index == 2
    assert ref.ephemeral is False
    assert not ref.is_collection
    assert ref.watch_url == WATCH


def test_watch_with_junk_playlist_still_resolves_the_video():
    ref = parse_reference(f"{WATCH}&list=junk")
    assert (ref.kind, ref.video_id) == (VIDEO, VID)
    assert ref.playlist_id == ""
    assert ref.playlist_kind == ""
    assert ref.params["list"] == "junk"


def test_video_in_ephemeral_playlist_is_marked():
    ref = parse_reference(f"https://youtu.be/{VID}?t=1m30s&list=WL")
    assert (ref.kind, ref.video_id, ref.start) == (VIDEO, VID, 90)
    assert (ref.playlist_id, ref.playlist_kind, ref.ephemeral) == (
        "WL",
        "watch-later",
        True,
    )


def test_all_query_parameters_are_retained():
    ref = parse_reference(f"{WATCH}&si=abc&pp=xyz&newparam=1")
    assert ref.params == {"v": VID, "si": "abc", "pp": "xyz", "newparam": "1"}


# -- playlist forms -----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(f"https://www.youtube.com/playlist?list={PLID}", id="playlist"),
        pytest.param(f"https://www.youtube.com/watch?list={PLID}", id="watch-list"),
        pytest.param(f"https://www.youtube.com/watch?v=&list={PLID}", id="blank-v"),
        pytest.param(
            f"https://www.youtube.com/embed/videoseries?list={PLID}", id="videoseries"
        ),
        pytest.param(
            f"https://www.youtube.com/embed/VideoSeries?list={PLID}", id="mixed-case"
        ),
        pytest.param(f"https://music.youtube.com/playlist?list={PLID}", id="music"),
        pytest.param(f"https://www.youtube.com/premium?list={PLID}", id="other-page"),
    ],
)
def test_playlist_url_forms(url):
    ref = parse_reference(url)
    assert (ref.kind, ref.playlist_id, ref.playlist_kind) == (PLAYLIST, PLID, "user")
    assert ref.is_collection
    assert ref.describe() == f"the user playlist {PLID}"


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        pytest.param("https://www.youtube.com/playlist", "no 'list'", id="missing"),
        pytest.param("https://www.youtube.com/playlist?list=", "no 'list'", id="blank"),
        pytest.param("https://www.youtube.com/playlist?list=PL", "too short", id="PL"),
        pytest.param(
            "https://www.youtube.com/playlist?list=PLabcdefg", "too short", id="9-chars"
        ),
        pytest.param(
            "https://www.youtube.com/playlist?list=PL%20bad%20value",
            "not a valid playlist id",
            id="spaces",
        ),
        pytest.param(
            "https://www.youtube.com/playlist?list=PLabc%2Fdefghij",
            "not a valid playlist id",
            id="slash",
        ),
        pytest.param("https://www.youtube.com/watch?list=junk", "too short", id="watch"),
        pytest.param(
            "https://www.youtube.com/embed/videoseries", "no 'list'", id="series-none"
        ),
        pytest.param(
            "https://www.youtube.com/embed/videoseries?list=PL",
            "too short",
            id="series-short",
        ),
    ],
)
def test_playlist_url_with_unusable_list(url, fragment):
    with pytest.raises(ReferenceError, match=fragment):
        parse_reference(url)


def test_playlist_minimum_length_boundary():
    url = "https://www.youtube.com/playlist?list="
    assert parse_reference(url + "PLabcdefgh").playlist_id == "PLabcdefgh"
    with pytest.raises(ReferenceError, match="got 9 characters"):
        parse_reference(url + "PLabcdefg")


# -- channel forms ------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "attr", "value", "tab"),
    [
        pytest.param(
            f"https://www.youtube.com/channel/{CHID}", "channel_id", CHID, "", id="id"
        ),
        pytest.param(
            f"https://www.youtube.com/channel/{CHID}/videos",
            "channel_id",
            CHID,
            "videos",
            id="id-tab",
        ),
        pytest.param("https://www.youtube.com/@cs50", "handle", "cs50", "", id="handle"),
        pytest.param(
            "https://www.youtube.com/@cs50/", "handle", "cs50", "", id="handle-slash"
        ),
        pytest.param(
            "https://www.youtube.com/%40cs50", "handle", "cs50", "", id="handle-encoded"
        ),
        pytest.param(
            "https://www.youtube.com/@cs50/podcasts",
            "handle",
            "cs50",
            "podcasts",
            id="handle-tab",
        ),
        pytest.param(
            "https://www.youtube.com/@cs50/Podcasts",
            "handle",
            "cs50",
            "podcasts",
            id="tab-lowercased",
        ),
        pytest.param(
            "https://www.youtube.com/@%E6%97%A5%E6%9C%AC%E8%AA%9E",
            "handle",
            "日本語",
            "",
            id="handle-unicode",
        ),
        pytest.param(
            "https://www.youtube.com/c/Name", "channel_name", "Name", "", id="custom"
        ),
        pytest.param(
            "https://www.youtube.com/c/Name/playlists",
            "channel_name",
            "Name",
            "playlists",
            id="custom-tab",
        ),
        pytest.param(
            "https://www.youtube.com/user/Name", "channel_name", "Name", "", id="legacy"
        ),
        pytest.param(
            "https://www.youtube.com/cs50", "channel_name", "cs50", "", id="vanity"
        ),
        pytest.param(
            "https://www.youtube.com/cs50/videos",
            "channel_name",
            "cs50",
            "videos",
            id="vanity-tab",
        ),
        pytest.param(
            "https://www.youtube.com/profile?user=foo",
            "channel_name",
            "foo",
            "",
            id="profile",
        ),
    ],
)
def test_channel_url_forms(url, attr, value, tab):
    ref = parse_reference(url)
    assert ref.kind == CHANNEL
    assert getattr(ref, attr) == value
    assert ref.channel_ref == value
    assert ref.tab == tab
    assert ref.tab_known is True
    assert ref.is_collection
    assert ref.watch_url == ""
    others = {"channel_id", "handle", "channel_name"} - {attr}
    assert all(getattr(ref, other) == "" for other in others)


@pytest.mark.parametrize(
    "tab",
    sorted(ref_module._KNOWN_TABS),
)
def test_every_known_tab_is_known(tab):
    ref = parse_reference(f"https://www.youtube.com/@cs50/{tab}")
    assert (ref.tab, ref.tab_known) == (tab, True)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://www.youtube.com/@cs50/newtabname", id="handle"),
        pytest.param(f"https://www.youtube.com/channel/{CHID}/newtabname", id="id"),
        pytest.param("https://www.youtube.com/c/Name/NewTabName", id="custom"),
        pytest.param("https://www.youtube.com/cs50/newtabname", id="vanity"),
    ],
)
def test_unknown_tab_is_a_value_not_an_error(url):
    ref = parse_reference(url)
    assert ref.kind == CHANNEL
    assert (ref.tab, ref.tab_known) == ("newtabname", False)
    assert "tab not recognised by this build" in ref.describe()


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        pytest.param(
            "https://www.youtube.com/channel", "no channel identifier", id="channel"
        ),
        pytest.param("https://www.youtube.com/c/", "no channel identifier", id="c"),
        pytest.param("https://www.youtube.com/user", "no channel identifier", id="user"),
        pytest.param(
            "https://www.youtube.com/channel/UCshort", "not a valid channel id", id="short"
        ),
        pytest.param(
            f"https://www.youtube.com/channel/{CHID}A", "not a valid channel id", id="25"
        ),
        pytest.param(
            "https://www.youtube.com/channel/@cs50", "not a valid channel id", id="handle"
        ),
        pytest.param(
            f"https://www.youtube.com/channel/uc{CHID[2:]}",
            "not a valid channel id",
            id="lowercase-prefix",
        ),
        pytest.param("https://www.youtube.com/@", "handle \\(empty\\)", id="empty-handle"),
        pytest.param(
            "https://www.youtube.com/@bad!name", "unsupported character", id="bad-handle"
        ),
        pytest.param(
            "https://www.youtube.com/@" + "a" * 31, "more than 30", id="long-handle"
        ),
        pytest.param("https://www.youtube.com/profile", "no 'user'", id="profile"),
        pytest.param("https://www.youtube.com/profile?user=", "no 'user'", id="blank"),
    ],
)
def test_channel_url_errors(url, fragment):
    with pytest.raises(ReferenceError, match=fragment):
        parse_reference(url)


def test_channel_url_keeps_incidental_playlist_context():
    ref = parse_reference(f"https://www.youtube.com/@cs50?list={PLID}")
    assert (ref.kind, ref.handle, ref.playlist_id) == (CHANNEL, "cs50", PLID)


_HANDLED_HEADS = {
    "watch", "watch_videos", "watch_popup", "playlist", "embed", "shorts", "live",
    "clip", "post", "results", "hashtag", "feed", "channel", "c", "user", "v", "e",
    "profile",
}


@pytest.mark.parametrize(
    "segment", sorted(ref_module._RESERVED_SEGMENTS - _HANDLED_HEADS)
)
def test_reserved_page_is_never_read_as_a_vanity_channel(segment):
    with pytest.raises(ReferenceError, match="names no video"):
        parse_reference(f"https://www.youtube.com/{segment}")


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://www.youtube.com/", id="root-slash"),
        pytest.param("https://www.youtube.com", id="root"),
        pytest.param("https://www.youtube.com/?feature=share", id="root-query"),
        pytest.param("https://www.youtube.com/cs50/videos/more", id="too-deep"),
        pytest.param("https://www.youtube.com/Premium", id="reserved-mixed-case"),
    ],
)
def test_youtube_url_naming_nothing_is_rejected(url):
    with pytest.raises(ReferenceError, match="is a YouTube URL but names no"):
        parse_reference(url)


def test_deep_unknown_path_with_list_falls_back_to_the_playlist():
    ref = parse_reference(f"https://www.youtube.com/cs50/videos/more?list={PLID}")
    assert (ref.kind, ref.playlist_id) == (PLAYLIST, PLID)


# -- clip, post, search, hashtag, feed ----------------------------------------


def test_clip_url():
    ref = parse_reference("https://www.youtube.com/clip/UgkxAbC-123_x")
    assert (ref.kind, ref.clip_id) == (CLIP, "UgkxAbC-123_x")
    assert ref.video_id == ""
    assert not ref.is_collection
    assert ref.describe() == "the clip UgkxAbC-123_x"


def test_post_url():
    ref = parse_reference("https://www.youtube.com/post/UgkxAbC-123_x")
    assert (ref.kind, ref.post_id) == (POST, "UgkxAbC-123_x")
    assert not ref.is_collection
    assert ref.describe() == "the community post UgkxAbC-123_x"


@pytest.mark.parametrize(
    ("url", "query"),
    [
        pytest.param(
            "https://www.youtube.com/results?search_query=a+b%20c", "a b c", id="decoded"
        ),
        pytest.param("https://www.youtube.com/results?q=zz", "zz", id="q-fallback"),
        pytest.param(
            "https://www.youtube.com/results?search_query=sq&q=zz", "sq", id="priority"
        ),
        pytest.param(
            "https://www.youtube.com/results?search_query=%E6%97%A5%E6%9C%AC",
            "日本",
            id="unicode",
        ),
        pytest.param("https://www.youtube.com/results", "", id="no-terms"),
    ],
)
def test_search_url(url, query):
    ref = parse_reference(url)
    assert (ref.kind, ref.search_query) == (SEARCH, query)
    assert ref.is_collection
    assert ref.describe() == f"a search for {query!r}"


@pytest.mark.parametrize(
    ("url", "tag"),
    [
        pytest.param("https://www.youtube.com/hashtag/python", "python", id="plain"),
        pytest.param("https://www.youtube.com/hashtag/%23python", "python", id="hash"),
        pytest.param(
            "https://www.youtube.com/hashtag/%E6%97%A5%E6%9C%AC", "日本", id="cjk"
        ),
        pytest.param("https://www.youtube.com/hashtag/Python/shorts", "Python", id="sub"),
    ],
)
def test_hashtag_url(url, tag):
    ref = parse_reference(url)
    assert (ref.kind, ref.hashtag) == (HASHTAG, tag)
    assert ref.is_collection
    assert ref.describe() == f"the hashtag #{tag}"


@pytest.mark.parametrize(
    ("url", "feed", "ephemeral"),
    [
        pytest.param("https://www.youtube.com/feed/subscriptions", "subscriptions", True, id="subscriptions"),
        pytest.param("https://www.youtube.com/feed/history", "history", True, id="history"),
        pytest.param("https://www.youtube.com/feed/library", "library", True, id="library"),
        pytest.param("https://www.youtube.com/feed/you", "you", True, id="you"),
        pytest.param("https://www.youtube.com/feed/downloads", "downloads", True, id="downloads"),
        pytest.param("https://www.youtube.com/feed/Subscriptions", "subscriptions", True, id="mixed-case"),
        pytest.param("https://www.youtube.com/feed/trending", "trending", False, id="trending"),
        pytest.param("https://www.youtube.com/feed", "", False, id="no-name"),
    ],
)
def test_feed_url(url, feed, ephemeral):
    ref = parse_reference(url)
    assert (ref.kind, ref.feed) == (FEED, feed)
    assert ref.ephemeral is ephemeral
    assert ref.is_collection
    assert ref.describe() == f"the {feed!r} feed"


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        pytest.param("https://www.youtube.com/clip", "no clip id", id="clip"),
        pytest.param("https://www.youtube.com/clip/", "no clip id", id="clip-slash"),
        pytest.param("https://www.youtube.com/post", "no post id", id="post"),
        pytest.param("https://www.youtube.com/hashtag", "no tag", id="hashtag"),
    ],
)
def test_page_url_without_identifier(url, fragment):
    with pytest.raises(ReferenceError, match=fragment):
        parse_reference(url)


# -- wrappers -----------------------------------------------------------------


def _wrap(url, times=1, base="https://consent.youtube.com/?continue="):
    for _ in range(times):
        url = base + quote(url, safe="")
    return url


def test_consent_wrapper_is_unwrapped():
    wrapper = _wrap(f"{WATCH}&t=5")
    ref = parse_reference(wrapper)
    assert (ref.kind, ref.video_id, ref.start) == (VIDEO, VID, 5)
    assert ref.host == "www.youtube.com"
    assert ref.unwrapped_from == wrapper
    assert ref.raw == wrapper


def test_attribution_link_with_relative_destination():
    wrapper = f"https://www.youtube.com/attribution_link?u=%2Fwatch%3Fv%3D{VID}"
    ref = parse_reference(wrapper)
    assert (ref.kind, ref.video_id) == (VIDEO, VID)
    assert ref.unwrapped_from == wrapper


def test_redirect_to_another_youtube_host():
    wrapper = "https://www.youtube.com/redirect?q=" + quote(
        f"https://youtu.be/{VID}", safe=""
    )
    ref = parse_reference(wrapper)
    assert (ref.video_id, ref.host) == (VID, "youtu.be")


def test_unwrapped_reference_has_no_wrapper_recorded():
    assert parse_reference(WATCH).unwrapped_from == ""


def test_nested_wrappers_record_the_outermost():
    wrapper = _wrap(f"https://youtu.be/{VID}", times=3)
    ref = parse_reference(wrapper)
    assert ref.video_id == VID
    assert ref.unwrapped_from == wrapper


def test_unwrapping_is_bounded():
    limit = ref_module._MAX_UNWRAP
    assert parse_reference(_wrap(f"https://youtu.be/{VID}", times=limit)).video_id == VID
    with pytest.raises(ReferenceError):
        parse_reference(_wrap(f"https://youtu.be/{VID}", times=limit + 1))


@pytest.mark.parametrize(
    "destination",
    [
        pytest.param("https://example.com", id="foreign-root"),
        pytest.param(f"https://example.com/watch?v={VID}", id="foreign-watch"),
        pytest.param(f"https://youtube.com.evil.example/watch?v={VID}", id="suffix"),
        pytest.param(f"https://www.youtube.com@evil.example/watch?v={VID}", id="at"),
        pytest.param("javascript:alert(1)", id="javascript"),
        pytest.param(VID, id="bare-id"),
    ],
)
def test_outbound_redirect_is_not_reported_as_youtube(destination):
    url = "https://www.youtube.com/redirect?q=" + quote(destination, safe="")
    with pytest.raises(ReferenceError, match="names no video"):
        parse_reference(url)


def test_protocol_relative_outbound_redirect_is_not_reported_as_youtube():
    url = "https://www.youtube.com/redirect?q=" + quote(
        f"//evil.example/watch?v={VID}", safe=""
    )
    with pytest.raises(ReferenceError):
        parse_reference(url)


def test_unwrap_helper_leaves_plain_and_foreign_urls_alone():
    assert ref_module._unwrap(WATCH) == (WATCH, "")
    foreign = "https://example.com/?continue=" + quote(WATCH, safe="")
    assert ref_module._unwrap(foreign) == (foreign, "")


def test_wrapper_on_a_foreign_host_is_not_followed():
    foreign = "https://example.com/?continue=" + quote(WATCH, safe="")
    with pytest.raises(ReferenceError, match="is not a YouTube URL"):
        parse_reference(foreign)


# -- YouTubeReference ---------------------------------------------------------


def test_reference_defaults():
    ref = YouTubeReference(kind=VIDEO)
    for name in (
        "video_id", "playlist_id", "playlist_kind", "channel_id", "handle",
        "channel_name", "tab", "clip_id", "post_id", "search_query", "hashtag",
        "feed", "host", "unwrapped_from", "raw",
    ):
        assert getattr(ref, name) == "", name
    assert ref.index is None
    assert ref.start is None
    assert ref.ephemeral is False
    assert ref.tab_known is True
    assert ref.params == {}
    assert ref.watch_url == ""
    assert ref.channel_ref == ""


def test_reference_is_frozen():
    ref = parse_reference(VID)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.video_id = VID2


def test_reference_params_default_is_not_shared():
    first = YouTubeReference(kind=VIDEO)
    second = YouTubeReference(kind=VIDEO)
    assert first.params is not second.params


def test_reference_repr_omits_params():
    assert "params" not in repr(parse_reference(f"{WATCH}&si=secret"))


def test_equal_inputs_parse_to_equal_references():
    assert parse_reference(WATCH) == parse_reference(WATCH)
    assert parse_reference(WATCH) != parse_reference(WATCH.replace(VID, VID2))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (VIDEO, False),
        (CLIP, False),
        (POST, False),
        (PLAYLIST, True),
        (CHANNEL, True),
        (HASHTAG, True),
        (SEARCH, True),
        (FEED, True),
    ],
)
def test_is_collection(kind, expected):
    assert YouTubeReference(kind=kind).is_collection is expected


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param(
            {"channel_id": CHID, "handle": "h", "channel_name": "n"}, CHID, id="id-first"
        ),
        pytest.param({"handle": "h", "channel_name": "n"}, "h", id="then-handle"),
        pytest.param({"channel_name": "n"}, "n", id="then-name"),
        pytest.param({}, "", id="none"),
    ],
)
def test_channel_ref_orders_by_durability(fields, expected):
    assert YouTubeReference(kind=CHANNEL, **fields).channel_ref == expected


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param({"channel_id": CHID}, f"the channel {CHID}", id="id"),
        pytest.param({"handle": "cs50"}, "the channel @cs50", id="handle"),
        pytest.param({"channel_name": "cs50"}, "the channel cs50", id="name"),
        pytest.param({}, "the channel ?", id="nothing"),
        pytest.param(
            {"handle": "cs50", "tab": "podcasts"},
            "the 'podcasts' tab of channel @cs50",
            id="known-tab",
        ),
        pytest.param(
            {"channel_id": CHID, "handle": "cs50", "tab": "videos"},
            f"the 'videos' tab of channel {CHID}",
            id="id-beats-handle",
        ),
        pytest.param(
            {"handle": "cs50", "tab": "zzz", "tab_known": False},
            "the 'zzz' tab of channel @cs50 (tab not recognised by this build)",
            id="unknown-tab",
        ),
    ],
)
def test_describe_channel(fields, expected):
    assert YouTubeReference(kind=CHANNEL, **fields).describe() == expected


def test_describe_playlist_matches_docstring():
    assert parse_reference(PLID).describe() == "the user playlist PLXOJEg4xbr50"


# -- parse_video_reference ----------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(VID, id="bare"),
        pytest.param(f"https://youtu.be/{VID}", id="short-host"),
        pytest.param(f"{WATCH}&list={PLID}&index=2", id="watch-in-playlist"),
        pytest.param(f"https://www.youtube.com/shorts/{VID}", id="shorts"),
    ],
)
def test_parse_video_reference_accepts_single_videos(value):
    ref = parse_video_reference(value)
    assert (ref.kind, ref.video_id) == (VIDEO, VID)


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        pytest.param(
            "https://www.youtube.com/clip/UgkxABC", "does not contain the underlying", id="clip"
        ),
        pytest.param(
            "https://www.youtube.com/post/UgkxABC", "holds no embeddable video", id="post"
        ),
        pytest.param(
            "https://www.youtube.com/results?search_query=x", "holds no embeddable", id="search"
        ),
        pytest.param(
            "https://www.youtube.com/hashtag/python", "holds no embeddable", id="hashtag"
        ),
        pytest.param(
            "https://www.youtube.com/feed/trending", "holds no embeddable", id="feed"
        ),
        pytest.param(
            f"https://www.youtube.com/playlist?list={PLID}", "not a single video", id="playlist"
        ),
        pytest.param(PLID, "not a single video", id="bare-playlist"),
        pytest.param(
            "https://www.youtube.com/@cs50/playlists", "not a single video", id="channel"
        ),
        pytest.param("@cs50", "not a single video", id="bare-handle"),
        pytest.param(CHID, "not a single video", id="bare-channel-id"),
    ],
)
def test_parse_video_reference_refuses_non_videos(value, fragment):
    with pytest.raises(ReferenceError, match=fragment) as info:
        parse_video_reference(value)
    assert value in str(info.value)


def test_parse_video_reference_points_collections_at_the_gallery():
    with pytest.raises(ReferenceError, match="youtube-gallery"):
        parse_video_reference(f"https://www.youtube.com/playlist?list={PLID}")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),
        pytest.param(None, id="none"),
        pytest.param("https://example.com/watch?v=" + VID, id="foreign-host"),
        pytest.param(f"https://www.youtube.com/watch?v={VID}X", id="bad-id"),
    ],
)
def test_parse_video_reference_propagates_parse_errors(value):
    with pytest.raises(ReferenceError):
        parse_video_reference(value)


# -- private helpers with a documented contract -------------------------------


def test_checked_video_id_returns_or_raises():
    assert ref_module._checked_video_id(VID, "raw") == VID
    with pytest.raises(ReferenceError, match="got 0"):
        ref_module._checked_video_id("", "raw")


def test_tab_from():
    assert ref_module._tab_from(["@cs50"], 1) == ("", True)
    assert ref_module._tab_from(["@cs50", "Videos"], 1) == ("videos", True)
    assert ref_module._tab_from(["@cs50", "zzz"], 1) == ("zzz", False)


def test_parse_bare_returns_none_for_unrecognised_text():
    assert ref_module._parse_bare("hello", "hello") is None


def test_path_handlers_decline_urls_that_are_not_theirs():
    ctx = ref_module._Context(
        raw="raw", host="www.youtube.com", segments=[], query={}, playlist_id="",
        playlist_kind="", index=None, start=None, ephemeral=False, wrapper="",
    )
    assert ctx.head == ""
    for handler in ref_module._PATH_HANDLERS:
        assert handler(ctx) is None, handler.__name__


def test_normalize_requires_a_string_and_adds_a_scheme():
    assert ref_module._normalize(f"youtu.be/{VID}") == f"https://youtu.be/{VID}"
    assert ref_module._normalize(f"//youtu.be/{VID}") == f"https://youtu.be/{VID}"
    assert ref_module._normalize(f"http://youtu.be/{VID}") == f"http://youtu.be/{VID}"
    with pytest.raises(ReferenceError, match="expected a string"):
        ref_module._normalize(object())
