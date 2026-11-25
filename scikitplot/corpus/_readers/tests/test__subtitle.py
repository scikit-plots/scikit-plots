# scikitplot/corpus/_readers/tests/test__subtitle.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared subtitle parser and :class:`SubtitleReader`."""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import pytest

from ... import _readers  # noqa: F401  (registers the readers)
from ..._base import DocumentReader
from ..._schema import SourceType
from .. import _audio, _subtitle, _video
from .._subtitle import (
    SubtitleReader,
    find_companion,
    merge_rolling,
    parse_ass,
    parse_sbv,
    parse_srt,
    parse_sub,
    parse_vtt,
    lrc_parts,
)


def _texts(cues):
    return [c["text"] for c in cues]


def _times(cues):
    return [(c["timecode_start"], c["timecode_end"]) for c in cues]


def _write(tmp_path: Path, data, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


# ---------------------------------------------------------------------------
# SubRip and WebVTT
# ---------------------------------------------------------------------------

_SRT = """1
00:00:01,000 --> 00:00:02,500
Hello <i>there</i>,
general.

2
00:00:03,000 --> 00:00:04,000
{\\an8}Tom &amp; Jerry

3
00:01:05,25 --> 01:00:00,5
Last
"""


def test_srt_cues_times_markup_and_entities():
    cues = parse_srt(_SRT)
    assert _texts(cues) == ["Hello there, general.", "Tom & Jerry", "Last"]
    assert _times(cues) == [(1.0, 2.5), (3.0, 4.0), (65.25, 3600.5)]
    assert cues[0]["raw_text"] == "Hello <i>there</i>,\ngeneral."


def test_srt_crlf_and_missing_blank_lines():
    text = (
        "1\r\n00:00:01,000 --> 00:00:02,000\r\nOne\r\n"
        "2\r\n00:00:03,000 --> 00:00:04,000\r\nTwo\r\n"
        "00:00:05,000 --> 00:00:06,000\r\nThree"
    )
    assert _texts(parse_srt(text)) == ["One", "Two", "Three"]


def test_a_blank_line_ends_a_cue_as_the_specifications_say():
    stats: dict = {}
    text = "1\n00:00:01,000 --> 00:00:02,000\nOne\n\nstray line\n\n2\n00:00:03,000 --> 00:00:04,000\nTwo\n"
    assert _texts(parse_srt(text, stats)) == ["One", "Two"]
    assert stats == {"malformed": 1}


@pytest.mark.parametrize(
    "broken",
    [
        "2\n00:00:07,000 --> 00:00:1d,890\nBad\n",  # a letter in the time
        "00:00:07.000 --> 00:00:11:890\nBad\n",  # wrong separator
        "2\nBad without timing\n",
    ],
)
def test_a_malformed_cue_is_dropped_and_never_spills(broken):
    stats: dict = {}
    text = "1\n00:00:01,000 --> 00:00:02,000\nOne\n\n" + broken + "\n3\n00:00:09,000 --> 00:00:10,000\nThree\n"
    assert _texts(parse_srt(text, stats)) == ["One", "Three"]
    assert stats == {"malformed": 1}


def test_a_broken_timing_line_without_a_blank_line_ends_the_cue():
    stats: dict = {}
    text = "1\n00:00:01,000 --> 00:00:02,000\nOne\n2\n00:00:03,000 --> 00:00:0x,000\nLost\n"
    assert _texts(parse_srt(text, stats)) == ["One"] and stats == {"malformed": 1}


def test_sbv_malformed_blocks_are_counted():
    stats: dict = {}
    text = "0:00:01.000,0:00:02.000\nOk\n\n0:00:13.f89,0:00:15.489\nBad\n"
    assert _texts(parse_sbv(text, stats)) == ["Ok"] and stats == {"malformed": 1}


def test_the_reader_counts_and_warns_about_malformed_blocks(tmp_path):
    text = "1\n00:00:01,000 --> 00:00:02,000\nOne\n\njunk\n"
    reader = SubtitleReader(input_path=_write(tmp_path, text, "a.srt"))
    with mock.patch.object(_subtitle, "logger") as logger:
        assert [c["text"] for c in reader.get_raw_chunks()] == ["One"]
    assert reader.coverage.skipped == {"malformed": 1}
    assert "malformed" in logger.warning.call_args.args[0]


def test_srt_ignores_text_before_the_first_cue_and_empty_cues():
    text = "garbage\n\n1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:02,000 --> 00:00:03,000\nKept\n"
    assert _texts(parse_srt(text)) == ["Kept"]


def test_vtt_header_blocks_short_times_settings_voice_and_ids():
    vtt = (
        "\ufeffWEBVTT - Some title\nKind: captions\n\n"
        "STYLE\n::cue { color: red }\n\n"
        "NOTE a comment\nspanning lines\n\n"
        "intro\n00:01.000 --> 00:04.000 align:start position:10%\n"
        "<v.loud Alice>Hi <c.yellow>you</c></v>\n\n"
        "REGION\nid:r1\n\n"
        "00:00:05.000 --> 00:00:06.000\n<00:00:05.500>karaoke <b>word</b>\n"
    )
    cues = parse_vtt(vtt)
    assert _texts(cues) == ["Hi you", "karaoke word"]
    assert _times(cues) == [(1.0, 4.0), (5.0, 6.0)]
    assert cues[0]["speaker"] == "Alice" and "speaker" not in cues[1]


def test_a_large_file_parses_in_linear_time():
    block = "{}\n00:00:01,000 --> 00:00:02,000\nline of text\n\n"
    text = "".join(block.format(i) for i in range(100_000))
    start = time.perf_counter()
    cues = parse_srt(text)
    assert len(cues) == 100_000 and time.perf_counter() - start < 10


# ---------------------------------------------------------------------------
# SBV, MicroDVD and ASS
# ---------------------------------------------------------------------------


def test_sbv():
    text = "0:00:01.000,0:00:02.500\nHello\nworld\n\n0:00:03.000,0:00:04.000\n&lt;ok&gt;\n"
    cues = parse_sbv(text)
    assert _texts(cues) == ["Hello world", "<ok>"]
    assert _times(cues) == [(1.0, 2.5), (3.0, 4.0)]


@pytest.mark.parametrize(
    ("text", "rate", "expected"),
    [
        ("{25}{50}Hello|world\n", None, [(1.0, 2.0)]),  # default 25 fps
        ("{1}{1}50\n{25}{50}x\n", None, [(0.5, 1.0)]),  # header rate
        ("{1}{1}50\n{25}{50}x\n", 25.0, [(1.0, 2.0)]),  # explicit wins, header skipped
        ("{25}{}open\n{75}{100}next\n", None, [(1.0, 3.0), (3.0, 4.0)]),  # empty end
    ],
)
def test_microdvd_frame_rates(text, rate, expected):
    assert _times(parse_sub(text, rate)) == expected


def test_microdvd_style_codes_are_removed():
    assert _texts(parse_sub("{0}{25}{y:i}Italic|{c:$0000ff}red\n")) == ["Italic red"]


_ASS = """[Script Info]
Title: x

[V4+ Styles]
Format: Name, Fontname
Style: Default,Arial

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Comment: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,hidden note
Dialogue: 0,0:00:01.00,0:00:02.50,Default,Ann,0,0,0,,Hi, there\\Nfriend
Dialogue: 0,1:00:00.50,1:00:01.00,Default,,0,0,0,,{\\i1}Styled{\\i0}\\hword
Dialogue: broken line
"""


def test_ass_dialogue_fields_speaker_and_overrides():
    cues = parse_ass(_ASS)
    assert _texts(cues) == ["Hi, there friend", "Styled word"]
    assert _times(cues) == [(1.0, 2.5), (3600.5, 3601.0)]
    assert cues[0]["speaker"] == "Ann" and "speaker" not in cues[1]


def test_ass_custom_field_order():
    text = "[Events]\nFormat: Start, End, Text\nDialogue: 0:00:01.00,0:00:02.00,Reordered\n"
    assert _texts(parse_ass(text)) == ["Reordered"]


# ---------------------------------------------------------------------------
# Rolling captions
# ---------------------------------------------------------------------------


def test_merge_rolling_removes_repeated_lines_only():
    vtt = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nfirst line\n\n"
        "00:00:02.000 --> 00:00:03.000\nfirst line\nsecond line\n\n"
        "00:00:03.000 --> 00:00:04.000\nsecond line\n\n"
        "00:00:04.000 --> 00:00:05.000\nthird line\n"
    )
    cues = merge_rolling(parse_vtt(vtt))
    assert _texts(cues) == ["first line", "second line", "third line"]
    assert _times(cues)[1] == (2.0, 3.0)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.srt", "a.VTT", "a.sbv", "a.sub", "a.ass", "a.ssa", "a.lrc"])
def test_extensions_create_the_reader(tmp_path, name):
    assert type(DocumentReader.create(_write(tmp_path, "", name))) is SubtitleReader


def test_one_chunk_per_cue_with_metadata(tmp_path):
    reader = SubtitleReader(input_path=_write(tmp_path, _SRT, "a.srt"))
    chunks = list(reader.get_raw_chunks())
    assert [c["text"] for c in chunks] == ["Hello there, general.", "Tom & Jerry", "Last"]
    first = chunks[0]
    assert (first["cue_start"], first["cue_end"]) == (0, 0)
    assert (first["timecode_start"], first["timecode_end"]) == (1.0, 2.5)
    assert first["source_type"] == SourceType.SUBTITLE.value
    assert (first["subtitle_format"], first["encoding"]) == ("srt", "utf-8")
    assert reader.coverage.read == 3


def test_grouping_by_count_and_by_seconds(tmp_path):
    text = "".join(
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},900\nc{i}\n\n" for i in range(10)
    )
    path = _write(tmp_path, text, "a.srt")
    chunks = list(SubtitleReader(input_path=path, cues_per_chunk=4).get_raw_chunks())
    assert [(c["cue_start"], c["cue_end"]) for c in chunks] == [(0, 3), (4, 7), (8, 9)]
    assert chunks[0]["text"] == "c0 c1 c2 c3"
    assert (chunks[0]["timecode_start"], chunks[0]["timecode_end"]) == (0.0, 3.9)
    chunks = list(
        SubtitleReader(input_path=path, cues_per_chunk=100, max_chunk_seconds=2.5).get_raw_chunks()
    )
    assert [(c["cue_start"], c["cue_end"]) for c in chunks][:2] == [(0, 1), (2, 3)]


def test_grouped_speakers(tmp_path):
    vtt = (
        "WEBVTT\n\n00:01.000 --> 00:02.000\n<v A>one\n\n"
        "00:02.000 --> 00:03.000\n<v B>two\n\n00:03.000 --> 00:04.000\n<v A>three\n"
    )
    [chunk] = SubtitleReader(input_path=_write(tmp_path, vtt, "a.vtt"), cues_per_chunk=3).get_raw_chunks()
    assert chunk["speakers"] == ["A", "B"] and "speaker" not in chunk


def test_rolling_option_on_the_reader(tmp_path):
    vtt = "WEBVTT\n\n00:01.000 --> 00:02.000\na\n\n00:02.000 --> 00:03.000\na\nb\n"
    path = _write(tmp_path, vtt, "a.vtt")
    assert [c["text"] for c in SubtitleReader(input_path=path).get_raw_chunks()] == ["a", "a b"]
    on = SubtitleReader(input_path=path, dedupe_rolling=True)
    assert [c["text"] for c in on.get_raw_chunks()] == ["a", "b"]


@pytest.mark.parametrize(
    ("raw", "kwargs", "text", "codec"),
    [
        ("1\n00:00:01,000 --> 00:00:02,000\nHello\n".encode("utf-16"), {}, "Hello", "utf-16"),
        ("1\n00:00:01,000 --> 00:00:02,000\nПривет\n".encode("cp1251"),
         {"encoding": "cp1251"}, "Привет", "cp1251"),
        ("1\n00:00:01,000 --> 00:00:02,000\ncafé\n".encode("cp1252"), {}, "café", "cp1252"),
    ],
)
def test_encodings(tmp_path, raw, kwargs, text, codec):
    [chunk] = SubtitleReader(input_path=_write(tmp_path, raw, "a.srt"), **kwargs).get_raw_chunks()
    assert (chunk["text"], chunk["encoding"]) == (text, codec)


def test_a_binary_vobsub_is_refused(tmp_path):
    path = _write(tmp_path, b"\x00\x00\x01\xba" + b"\x00" * 64, "movie.sub")
    with pytest.raises(ValueError, match="not a text file"):
        list(SubtitleReader(input_path=path).get_raw_chunks())


def test_a_file_without_cues_warns(tmp_path):
    path = _write(tmp_path, "just some words\n", "a.srt")
    with mock.patch.object(_subtitle, "logger") as logger:
        assert list(SubtitleReader(input_path=path).get_raw_chunks()) == []
    assert "no timed cue" in logger.warning.call_args.args[0]


def test_an_explicit_format_overrides_the_extension(tmp_path):
    path = _write(tmp_path, "WEBVTT\n\n00:01.000 --> 00:02.000\nx\n", "captions.txt")
    [chunk] = SubtitleReader(input_path=path, subtitle_format="vtt").get_raw_chunks()
    assert chunk["subtitle_format"] == "vtt"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"subtitle_format": "smi"}, ValueError),
        ({"cues_per_chunk": 0}, ValueError),
        ({"cues_per_chunk": True}, ValueError),
        ({"max_chunk_seconds": 0}, ValueError),
        ({"frame_rate": -1}, ValueError),
        ({"dedupe_rolling": "yes"}, TypeError),
    ],
)
def test_bad_options(tmp_path, kwargs, error):
    with pytest.raises(error):
        SubtitleReader(input_path=_write(tmp_path, _SRT, "a.srt"), **kwargs)


def test_documents_are_built(tmp_path):
    docs = list(SubtitleReader(input_path=_write(tmp_path, _SRT, "a.srt")).get_documents())
    assert docs and docs[0].source_type == SourceType.SUBTITLE
    assert docs[0].timecode_start == 1.0


# ---------------------------------------------------------------------------
# Companions of audio and video
# ---------------------------------------------------------------------------


def test_find_companion_order_case_and_tags(tmp_path):
    media = _write(tmp_path, b"", "talk.mp4")
    assert find_companion(media, [".srt", ".vtt"]) is None
    _write(tmp_path, "x", "talk.fr.srt")
    _write(tmp_path, "x", "talk.en.srt")
    assert find_companion(media, [".srt"]).name == "talk.en.srt"  # sorted
    _write(tmp_path, "x", "TALK.SRT")
    assert find_companion(media, [".srt"]).name == "TALK.SRT"  # exact stem first
    _write(tmp_path, "x", "talk.vtt")
    assert find_companion(media, [".vtt", ".srt"]).name == "talk.vtt"  # extension order
    _write(tmp_path, "x", "talking.srt")
    assert find_companion(_write(tmp_path, b"", "talking.mp4"), [".srt"]).name == "talking.srt"


def test_video_and_audio_companions_use_the_shared_parser(tmp_path):
    vtt = "WEBVTT\n\n00:01.000 --> 00:02.000\n<v Ann>Hi &amp; bye\n"
    path = _write(tmp_path, vtt, "clip.en.vtt")
    [vcue] = _video._parse_subtitle(path, "vtt")
    [acue] = _audio._parse_vtt(path)
    assert vcue == acue
    assert (vcue["text"], vcue["speaker"], vcue["timecode_start"]) == ("Hi & bye", "Ann", 1.0)
    assert _video._find_subtitle(_write(tmp_path, b"", "clip.mp4")) == (path, "vtt")


def test_an_unreadable_companion_gives_no_cues_and_warns(tmp_path):
    path = _write(tmp_path, b"\x00\x01binary", "clip.srt")
    with mock.patch.object(_video, "logger") as logger:
        assert _video._parse_subtitle(path, "srt") == []
    assert "unreadable" in logger.warning.call_args.args[0]
    with mock.patch.object(_audio, "logger") as logger:
        assert _audio._parse_srt(path) == []
    assert "unreadable" in logger.warning.call_args.args[0]


# ---------------------------------------------------------------------------
# LRC lyrics
# ---------------------------------------------------------------------------

_LRC = """[ti:Song Title]
[ar:Some Artist]
[al:Album]
[length: 03:00]
[00:01.00][00:05.50]Chorus <00:01.50>line
[00:02.00]Verse
[00:03.00]
[01:02:03.25]Late
[00:00:04.50]Hours form
"""


def test_lrc_repeats_breaks_formats_and_metadata():
    cues, meta = lrc_parts(_LRC)
    assert [(c["text"], c["timecode_start"], c["timecode_end"]) for c in cues] == [
        ("Chorus line", 1.0, 2.0),
        ("Verse", 2.0, 3.0),  # the empty [00:03.00] ends it
        ("Hours form", 4.5, 5.5),
        ("Chorus line", 5.5, 3723.25),
        ("Late", 3723.25, 3728.25),
    ]
    assert [c["line_index"] for c in cues] == [0, 1, 2, 3, 4]
    assert meta["ti"] == "Song Title" and meta["ar"] == "Some Artist"


@pytest.mark.parametrize(("offset", "start"), [("+500", 0.5), ("-500", 1.5), ("bad", 1.0)])
def test_lrc_offset_makes_lyrics_sooner(offset, start):
    cues, _ = lrc_parts(f"[offset:{offset}]\n[00:01.00]x\n")
    assert cues[0]["timecode_start"] == start


def test_lrc_offset_never_goes_below_zero():
    cues, _ = lrc_parts("[offset:5000]\n[00:01.00]x\n")
    assert cues[0]["timecode_start"] == 0.0


def test_lrc_reader_carries_song_metadata(tmp_path):
    reader = SubtitleReader(input_path=_write(tmp_path, _LRC, "song.lrc"), cues_per_chunk=10)
    [chunk] = reader.get_raw_chunks()
    assert (chunk["source_title"], chunk["source_author"], chunk["album"]) == (
        "Song Title",
        "Some Artist",
        "Album",
    )
    assert chunk["subtitle_format"] == "lrc"
    [doc] = list(reader.get_documents())
    assert doc.source_title == "Song Title"


def test_audio_lrc_companion_uses_the_shared_parser(tmp_path):
    path = _write(tmp_path, "[ti:T]\n[00:01.00][00:03.00]La\n".encode("cp1252"), "song.lrc")
    cues, meta = _audio._parse_lrc(path)
    assert [c["timecode_start"] for c in cues] == [1.0, 3.0] and meta == {"ti": "T"}
    with mock.patch.object(_audio, "logger") as logger:
        assert _audio._parse_lrc(_write(tmp_path, b"\x00bin", "bad.lrc")) == ([], {})
    assert "unreadable" in logger.warning.call_args.args[0]


# ---------------------------------------------------------------------------
# Review: cost, language and companion options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spam", ["<", "<v ", "{\\", "{y:", "<i"])
def test_markup_that_never_closes_costs_linear_time(spam):
    # Each pattern stops at the next opener; this was 39 s for "<v " * 40000.
    start = time.perf_counter()
    from .._subtitle import _clean

    _clean([spam * 40_000])
    assert time.perf_counter() - start < 1.0


def test_find_companion_prefers_a_language():
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        media = folder / "talk.mp4"
        media.write_bytes(b"")
        for name in ("talk.srt", "talk.de.srt", "talk.en-US.forced.srt", "talk.fr.vtt"):
            (folder / name).write_text("x", encoding="utf-8")
        assert find_companion(media, [".srt", ".vtt"]).name == "talk.srt"
        assert find_companion(media, [".srt"], language="en").name == "talk.en-US.forced.srt"
        assert find_companion(media, [".srt"], language="DE").name == "talk.de.srt"
        # A language without a match keeps the usual order.
        assert find_companion(media, [".srt"], language="ja").name == "talk.srt"
        # The extension order still decides first.
        assert find_companion(media, [".vtt", ".srt"], language="en").name == "talk.fr.vtt"


def test_find_companion_lists_a_folder_once_per_change(tmp_path):
    import os

    from .. import _subtitle as sub

    media = [tmp_path / f"v{i}.mp4" for i in range(50)]
    for m in media:
        m.write_bytes(b"")
        m.with_suffix(".srt").write_text("x", encoding="utf-8")
    sub._LISTINGS.clear()
    with mock.patch.object(sub.os, "scandir", wraps=os.scandir) as scandir:
        assert all(find_companion(m, [".srt"]) for m in media)
        assert scandir.call_count == 1
        (tmp_path / "new.txt").write_text("x", encoding="utf-8")
        os.utime(tmp_path, ns=(1, 2))  # a change the folder's mtime shows
        find_companion(media[0], [".srt"])
        assert scandir.call_count == 2


def test_the_listing_cache_is_bounded(tmp_path):
    from .. import _subtitle as sub

    sub._LISTINGS.clear()
    for i in range(sub._LISTING_FOLDERS + 5):
        folder = tmp_path / f"d{i}"
        folder.mkdir()
        find_companion(folder / "a.mp4", [".srt"])
    assert len(sub._LISTINGS) == sub._LISTING_FOLDERS
    assert str(tmp_path / f"d{sub._LISTING_FOLDERS + 4}") in sub._LISTINGS


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ([("encoding", "x")], TypeError, "must be a dict"),
        ({"codec": "x"}, ValueError, "unknown key"),
        ({"language": ""}, TypeError, "non-empty str"),
        ({"encoding": "no-such-codec"}, ValueError, "not a codec"),
        ({"dedupe_rolling": "yes"}, TypeError, "dedupe_rolling"),
    ],
)
def test_companion_options_fail_when_the_reader_is_made(tmp_path, options, error, message):
    media = _write(tmp_path, b"", "clip.mp4")
    with pytest.raises(error, match=message):
        _video.VideoReader(input_path=media, companion_options=options)
    with pytest.raises(error, match=message):
        _audio.AudioReader(input_path=media.with_suffix(".mp3"), companion_options=options)


def test_video_companion_options_reach_the_parser(tmp_path):
    media = _write(tmp_path, b"\x00" * 16, "clip.mp4")
    _write(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nEnglish\n", "clip.en.srt")
    _write(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nПри\n".encode("cp1251"), "clip.ru.srt")
    reader = _video.VideoReader(
        input_path=media, companion_options={"language": "ru", "encoding": "cp1251"}
    )
    assert [c["text"] for c in reader.get_raw_chunks()] == ["При"]
    reader = _video.VideoReader(input_path=media)
    assert [c["text"] for c in reader.get_raw_chunks()] == ["English"]


def test_audio_txt_companion_follows_the_encoding_policy(tmp_path):
    path = _write(tmp_path, "café\nüber\n".encode("cp1252"), "song.txt")
    assert [c["text"] for c in _audio._parse_txt_companion(path)] == ["café", "über"]
    path = _write(tmp_path, "При\n".encode("cp1251"), "ru.txt")
    assert [c["text"] for c in _audio._parse_txt_companion(path, encoding="cp1251")] == ["При"]
