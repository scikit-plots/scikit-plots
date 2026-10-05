"""
Tests for the gallery-wide player options in ``video_options``.

The converters validate single-line directive option text; ``player_options``
maps validated ``video-*`` gallery options onto the leaf player's option
names. Expectations follow the module and function docstrings.
"""

from __future__ import annotations

import pytest

from .. import video_options
from ..video_options import LEAF_VIDEO_SPEC, VIDEO_SPEC, player_options

_single = video_options._single
_size = video_options._size
_aspect = video_options._aspect
_align = video_options._align
_privacy = video_options._privacy
_query = video_options._query

ALL_CONVERTERS = [_single, _size, _aspect, _align, _privacy, _query]


# -- spec tables --------------------------------------------------------------


def test_leaf_spec_names():
    assert set(LEAF_VIDEO_SPEC) == {
        "width", "height", "aspect", "align", "title", "privacy_mode",
        "url_parameters",
    }


@pytest.mark.parametrize(
    ("gallery_key", "leaf_key"),
    [
        ("video-width", "width"),
        ("video-height", "height"),
        ("video-aspect", "aspect"),
        ("video-align", "align"),
        ("video-title", "title"),
        ("video-privacy-mode", "privacy_mode"),
        ("video-url-parameters", "url_parameters"),
    ],
)
def test_gallery_spec_is_a_namespaced_view_of_the_leaf_spec(gallery_key, leaf_key):
    assert VIDEO_SPEC[gallery_key] is LEAF_VIDEO_SPEC[leaf_key]


def test_gallery_spec_has_no_extra_names():
    assert len(VIDEO_SPEC) == len(LEAF_VIDEO_SPEC)
    assert all(key.startswith("video-") and "_" not in key for key in VIDEO_SPEC)


# -- _single ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        pytest.param(None, "", id="none"),
        pytest.param("", "", id="empty"),
        pytest.param("   ", "", id="whitespace"),
        pytest.param("  My title \n", "My title", id="stripped"),
        pytest.param("café 日本", "café 日本", id="unicode"),
        pytest.param("a\tb", "a\tb", id="inner-tab-kept"),
    ],
)
def test_single_returns_stripped_text(argument, expected):
    assert _single(argument) == expected


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param("a\nb", id="newline"),
        pytest.param("a\rb", id="carriage-return"),
        pytest.param("a\r\nb", id="crlf"),
        pytest.param("a\x00b", id="nul"),
        pytest.param("title\n:width: 1", id="option-injection"),
    ],
)
@pytest.mark.parametrize("converter", ALL_CONVERTERS, ids=lambda f: f.__name__)
def test_every_converter_rejects_multi_line_source(converter, argument):
    with pytest.raises(ValueError, match="single line"):
        converter(argument)


# -- _size --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        pytest.param("640", "640", id="bare"),
        pytest.param("1", "1", id="one"),
        pytest.param("640px", "640px", id="px"),
        pytest.param("100%", "100%", id="percent"),
        pytest.param("  640px ", "640px", id="padded"),
        pytest.param("10", "10", id="inner-zero"),
    ],
)
def test_size_accepts(argument, expected):
    assert _size(argument) == expected


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty"),
        pytest.param("0", id="zero"),
        pytest.param("0px", id="zero-px"),
        pytest.param("064", id="leading-zero"),
        pytest.param("-5", id="negative"),
        pytest.param("+5", id="plus-sign"),
        pytest.param("1.5", id="decimal"),
        pytest.param("640 px", id="space-before-unit"),
        pytest.param("640PX", id="uppercase-unit"),
        pytest.param("640em", id="other-unit"),
        pytest.param("640px%", id="two-units"),
        pytest.param("px", id="unit-only"),
        pytest.param("auto", id="keyword"),
        pytest.param("١٢", id="non-ascii-digits"),
        pytest.param("640;height:1", id="css-injection"),
        pytest.param('640" onload="x', id="attribute-injection"),
    ],
)
def test_size_rejects(argument):
    with pytest.raises(ValueError, match="positive size"):
        _size(argument)


# -- _aspect ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        pytest.param("16:9", "16:9", id="widescreen"),
        pytest.param("4:3", "4:3", id="classic"),
        pytest.param("1:1", "1:1", id="square"),
        pytest.param(" 21:9 ", "21:9", id="padded"),
    ],
)
def test_aspect_accepts(argument, expected):
    assert _aspect(argument) == expected


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty"),
        pytest.param("16", id="no-colon"),
        pytest.param("16:", id="no-height"),
        pytest.param(":9", id="no-width"),
        pytest.param("0:9", id="zero-width"),
        pytest.param("16:0", id="zero-height"),
        pytest.param("016:9", id="leading-zero"),
        pytest.param("16 : 9", id="inner-space"),
        pytest.param("16/9", id="slash"),
        pytest.param("16x9", id="x"),
        pytest.param("16:9:1", id="three-parts"),
        pytest.param("1.5:1", id="decimal"),
        pytest.param("-16:9", id="negative"),
    ],
)
def test_aspect_rejects(argument):
    with pytest.raises(ValueError, match="positive ratio"):
        _aspect(argument)


# -- _align -------------------------------------------------------------------


@pytest.mark.parametrize("argument", ["left", "center", "right", "  center  "])
def test_align_accepts(argument):
    assert _align(argument) == argument.strip()


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty"),
        pytest.param("Left", id="capitalised"),
        pytest.param("centre", id="british"),
        pytest.param("middle", id="middle"),
        pytest.param("justify", id="justify"),
        pytest.param("left right", id="two-values"),
    ],
)
def test_align_rejects(argument):
    with pytest.raises(ValueError, match="left, center or right"):
        _align(argument)


# -- _privacy -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        pytest.param(None, "", id="flag-none"),
        pytest.param("", "", id="flag-empty"),
        pytest.param("  ", "", id="flag-blank"),
        pytest.param("true", "true", id="true"),
        pytest.param("TRUE", "true", id="uppercase"),
        pytest.param(" False ", "false", id="padded-mixed-case"),
        pytest.param("on", "on", id="on"),
        pytest.param("off", "off", id="off"),
        pytest.param("yes", "yes", id="yes"),
        pytest.param("no", "no", id="no"),
        pytest.param("1", "1", id="one"),
        pytest.param("0", "0", id="zero"),
    ],
)
def test_privacy_accepts(argument, expected):
    assert _privacy(argument) == expected


@pytest.mark.parametrize("argument", ["maybe", "2", "t", "enabled", "true false"])
def test_privacy_rejects(argument):
    with pytest.raises(ValueError, match="empty flag, true, or false"):
        _privacy(argument)


# -- _query -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        pytest.param(None, "", id="none"),
        pytest.param("", "", id="empty"),
        pytest.param("?", "", id="question-only"),
        pytest.param("&&", "", id="ampersands-only"),
        pytest.param("a=1", "?a=1", id="adds-question-mark"),
        pytest.param("?a=1&b=2", "?a=1&b=2", id="keeps-one-question-mark"),
        pytest.param("??&a=1", "?a=1", id="strips-leading-separators"),
        pytest.param("  ?rel=0  ", "?rel=0", id="padded"),
        pytest.param("a", "?a=", id="blank-value-kept"),
        pytest.param("a=&b=2", "?a=&b=2", id="explicit-blank"),
        pytest.param("a=1&a=2", "?a=1&a=2", id="repeats-kept-in-order"),
        pytest.param("a=b c", "?a=b+c", id="space-encoded"),
        pytest.param("a=%20b", "?a=+b", id="re-encoded"),
        pytest.param("a=<x>&b=\"y\"", "?a=%3Cx%3E&b=%22y%22", id="html-encoded"),
        pytest.param("q=日", "?q=%E6%97%A5", id="unicode-encoded"),
    ],
)
def test_query_normalises(argument, expected):
    assert _query(argument) == expected


def test_query_accepts_exactly_128_pairs():
    text = "&".join(f"k{i}={i}" for i in range(128))
    assert _query(text) == "?" + text


def test_query_rejects_more_than_128_pairs():
    with pytest.raises(ValueError):
        _query("&".join(f"k{i}={i}" for i in range(129)))


def test_query_is_idempotent():
    once = _query("a=b c&d=%3C")
    assert _query(once) == once


# -- player_options -----------------------------------------------------------


def test_player_options_uses_the_record_title_by_default():
    assert player_options({}, "My video") == {"title": "My video"}


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        pytest.param("  A \n B\t C ", "A B C", id="collapsed"),
        pytest.param("", "", id="empty"),
        pytest.param("   ", "", id="blank"),
        pytest.param("café 日本", "café 日本", id="unicode"),
    ],
)
def test_player_options_normalises_title_whitespace(title, expected):
    assert player_options({}, title) == {"title": expected}


def test_player_options_explicit_title_overrides_the_record_title():
    assert player_options({"video-title": "Override"}, "Record")["title"] == "Override"


def test_player_options_maps_names_to_leaf_names():
    options = {
        "video-width": "640",
        "video-height": "360px",
        "video-aspect": "16:9",
        "video-align": "center",
        "video-url-parameters": "?rel=0",
    }
    assert player_options(options, "T") == {
        "title": "T",
        "width": "640",
        "height": "360px",
        "aspect": "16:9",
        "align": "center",
        "url_parameters": "?rel=0",
    }


@pytest.mark.parametrize("value", ["", "true", "on", "yes", "1"])
def test_player_options_true_privacy_is_a_bare_flag(value):
    assert player_options({"video-privacy-mode": value}, "T") == {
        "title": "T",
        "privacy_mode": "",
    }


@pytest.mark.parametrize("value", ["false", "off", "no", "0"])
def test_player_options_false_privacy_is_absent(value):
    assert player_options({"video-privacy-mode": value}, "T") == {"title": "T"}


def test_player_options_ignores_unrelated_gallery_options():
    options = {"columns": "3", "width": "1", "video_width": "2", "class": "x"}
    assert player_options(options, "T") == {"title": "T"}


def test_player_options_does_not_mutate_its_input():
    options = {"video-privacy-mode": "true", "video-width": "640"}
    snapshot = dict(options)
    player_options(options, "T")
    assert options == snapshot


def test_player_options_only_emits_leaf_option_names():
    options = {key: "x" for key in VIDEO_SPEC}
    assert set(player_options(options, "T")) == set(LEAF_VIDEO_SPEC)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("FALSE", {"title": "T"}, id="uppercase-false"),
        pytest.param(" Off ", {"title": "T"}, id="padded-off"),
        pytest.param("YES", {"title": "T", "privacy_mode": ""}, id="uppercase-yes"),
        pytest.param(None, {"title": "T", "privacy_mode": ""}, id="bare-flag"),
    ],
)
def test_validated_privacy_spelling_round_trips(raw, expected):
    validated = {"video-privacy-mode": VIDEO_SPEC["video-privacy-mode"](raw)}
    assert player_options(validated, "T") == expected


def test_validated_options_round_trip_through_player_options():
    raw = {
        "video-width": " 100% ",
        "video-aspect": "4:3 ",
        "video-align": "right",
        "video-title": "  Talk  ",
        "video-url-parameters": "rel=0&start=5",
    }
    validated = {key: VIDEO_SPEC[key](value) for key, value in raw.items()}
    assert player_options(validated, "ignored") == {
        "title": "Talk",
        "width": "100%",
        "aspect": "4:3",
        "align": "right",
        "url_parameters": "?rel=0&start=5",
    }
