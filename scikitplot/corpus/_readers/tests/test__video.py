# scikitplot/corpus/_readers/tests/test__video.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Tests for scikitplot.corpus._readers._video
===========================================

Coverage
--------
CRITICAL-V1
    ``timecode_start`` and ``timecode_end`` are the yielded keys at the
    ``get_raw_chunks`` boundary for **both** subtitle and transcription paths.
    Old wrong keys ``timestamp_start`` / ``timestamp_end`` must be absent.
    Both promoted keys must be in ``_PROMOTED_RAW_KEYS``.

HIGH-V2
    Whisper transcription yields ``"source_type": SourceType.VIDEO.value``
    (not the invalid string ``"transcription"`` which is not a SourceType member
    and would silently fall back to ``SourceType.UNKNOWN`` in ``get_documents()``).
    The Whisper sub-type is preserved via ``"transcript_type": "whisper"`` which
    goes to ``metadata`` (non-promoted key).

HIGH-V3
    Subtitle cues yield ``"source_type": SourceType.SUBTITLE.value`` (explicit
    enum value, not the raw string ``"subtitle"`` which is fragile).

MEDIUM-V4
    All internal parser helpers (``_parse_srt``, ``_parse_vtt``, ``_parse_sbv``,
    ``_parse_sub``) and ``_transcribe_whisper`` use ``timecode_start`` /
    ``timecode_end`` as their internal dict keys — consistent with the promoted
    contract throughout.

NEW-V5 yield_frames field
    VideoReader has ``yield_frames: bool = False`` field.
    When set, documents carry ``raw_tensor`` of shape ``(T, H, W, C)`` uint8.

NEW-V6 VideoReader.__post_init__ validation
    ``__post_init__`` rejects unrecognised ``whisper_model`` values and
    is documented.

Additional coverage
    * ``_tc_to_seconds`` arithmetic correctness.
    * ``_parse_srt``: multi-block, HTML tag stripping, empty-text skip.
    * ``_parse_sbv``: timecode parsing, multi-line text.
    * ``_parse_sub``: frame-to-seconds conversion, pipe separator, custom fps.
    * ``_parse_vtt``: WEBVTT header stripped, delegates to SRT parser.
    * ``_find_subtitle``: priority order, absent file.
    * ``VideoReader.__post_init__``: invalid model, negative frame_rate.
    * ``get_raw_chunks``: file-too-large guard.
    * ``get_raw_chunks``: no subtitle + transcribe=False → zero chunks.
    * ``_transcribe_whisper``: faster-whisper path, openai-whisper fallback,
      both absent → ``ImportError``.
    * ``source_type`` not in ``_transcribe_whisper`` internal dicts.

All tests use ``unittest.mock`` — no video/subtitle files, no Whisper model
weights, and no OCR libraries are required.
"""

from __future__ import annotations

import builtins
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import types
import importlib.util
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from scikitplot.corpus import _base, _schema, _readers

# Imports after bootstrap
from scikitplot.corpus._readers._video import (  # noqa: E402
    VideoReader,
    _parse_srt,
    _parse_vtt,
    _parse_sbv,
    _parse_sub,
    _parse_subtitle,
    _find_subtitle,
    _transcribe_whisper,
    _tc_to_seconds,
)
from scikitplot.corpus._schema import (  # noqa: E402
    SectionType,
    SourceType,
    _PROMOTED_RAW_KEYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_MP4_HEADER = (
    b"\x00\x00\x00\x18ftypmp42"
    b"\x00\x00\x00\x00mp42mp41"
)

def _make_video_file(tmp_path: pathlib.Path, name: str = "clip.mp4") -> pathlib.Path:
    """Create a minimal placeholder video file (content irrelevant — stat only)."""
    f = tmp_path / name
    f.write_bytes(b"\x00" * 16)  # b"fake not a real video"
    return f


def _make_subtitle_file(
    tmp_path: pathlib.Path, name: str, content: str
) -> pathlib.Path:
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


def create_minimal_mp4(path: pathlib.Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-f", "lavfi",
            "-i", "color=c=black:s=320x240:d=1",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(path),
            "-y",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# ---------------------------------------------------------------------------
# Group 1: _PROMOTED_RAW_KEYS contract
# ---------------------------------------------------------------------------


class TestPromotedKeyContract:
    """Verify timecode key names match the _PROMOTED_RAW_KEYS authoritative set."""

    def test_timecode_start_in_promoted_raw_keys(self) -> None:
        assert "timecode_start" in _PROMOTED_RAW_KEYS

    def test_timecode_end_in_promoted_raw_keys(self) -> None:
        assert "timecode_end" in _PROMOTED_RAW_KEYS

    def test_old_timestamp_start_not_in_promoted_raw_keys(self) -> None:
        """Regression: old key name must not be in _PROMOTED_RAW_KEYS."""
        assert "timestamp_start" not in _PROMOTED_RAW_KEYS

    def test_old_timestamp_end_not_in_promoted_raw_keys(self) -> None:
        assert "timestamp_end" not in _PROMOTED_RAW_KEYS

    def test_source_type_in_promoted_raw_keys(self) -> None:
        assert "source_type" in _PROMOTED_RAW_KEYS


# ---------------------------------------------------------------------------
# Group 2: Internal helper key naming (MEDIUM-V4)
# ---------------------------------------------------------------------------


SRT_SAMPLE = textwrap.dedent("""\
    1
    00:00:01,000 --> 00:00:03,500
    Hello world

    2
    00:00:05,000 --> 00:00:07,200
    Second <b>cue</b>
""")

VTT_SAMPLE = textwrap.dedent("""\
    WEBVTT

    00:00:01.000 --> 00:00:03.500
    Hello world

    00:00:05.000 --> 00:00:07.200
    Second cue
""")

SBV_SAMPLE = textwrap.dedent("""\
    0:00:01.000,0:00:03.500
    Hello world

    0:00:05.000,0:00:07.200
    Second cue
""")

SUB_SAMPLE = textwrap.dedent("""\
    {25}{87}Hello world
    {125}{180}Second cue
""")


class TestInternalHelperKeyNames:
    """MEDIUM-V4: all parser helpers must use timecode_start / timecode_end."""

    def test_parse_srt_uses_timecode_keys(self) -> None:
        cues = _parse_srt(SRT_SAMPLE)
        assert len(cues) == 2
        for cue in cues:
            assert "timecode_start" in cue, "timecode_start missing from _parse_srt output"
            assert "timecode_end" in cue, "timecode_end missing from _parse_srt output"
            assert "timestamp_start" not in cue, "old timestamp_start present in _parse_srt"
            assert "timestamp_end" not in cue, "old timestamp_end present in _parse_srt"

    def test_parse_vtt_uses_timecode_keys(self) -> None:
        cues = _parse_vtt(VTT_SAMPLE)
        assert len(cues) == 2
        for cue in cues:
            assert "timecode_start" in cue
            assert "timecode_end" in cue
            assert "timestamp_start" not in cue
            assert "timestamp_end" not in cue

    def test_parse_sbv_uses_timecode_keys(self) -> None:
        cues = _parse_sbv(SBV_SAMPLE)
        assert len(cues) == 2
        for cue in cues:
            assert "timecode_start" in cue
            assert "timecode_end" in cue
            assert "timestamp_start" not in cue
            assert "timestamp_end" not in cue

    def test_parse_sub_uses_timecode_keys(self) -> None:
        cues = _parse_sub(SUB_SAMPLE, frame_rate=25.0)
        assert len(cues) == 2
        for cue in cues:
            assert "timecode_start" in cue
            assert "timecode_end" in cue
            assert "timestamp_start" not in cue
            assert "timestamp_end" not in cue

    def test_transcribe_whisper_internal_dicts_use_timecode_keys(self, tmp_path: pathlib.Path) -> None:
        """_transcribe_whisper must return dicts with timecode_start/end, not timestamp_*."""
        seg = MagicMock()
        seg.text = "hello"
        seg.start = 1.0
        seg.end = 3.5

        fw_mock = MagicMock()
        fw_mock.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        video = _make_video_file(tmp_path, "dummy.mp4")
        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            results = _transcribe_whisper(video, "base", None)

        assert len(results) == 1
        r = results[0]
        assert "timecode_start" in r, "timecode_start missing from _transcribe_whisper"
        assert "timecode_end" in r, "timecode_end missing from _transcribe_whisper"
        assert "timestamp_start" not in r
        assert "timestamp_end" not in r

    def test_transcribe_whisper_no_source_type_in_internal_dicts(self, tmp_path: pathlib.Path) -> None:
        """
        source_type must NOT be in _transcribe_whisper's internal dicts.
        It is set exclusively at the get_raw_chunks yield boundary.
        """
        seg = MagicMock()
        seg.text = "segment text"
        seg.start = 0.5
        seg.end = 2.0

        fw_mock = MagicMock()
        fw_mock.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        video = _make_video_file(tmp_path, "dummy.mp4")
        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            results = _transcribe_whisper(video, "base", "en")

        assert len(results) == 1
        assert "source_type" not in results[0], (
            "source_type found in _transcribe_whisper dict — HIGH-V2 regression: "
            "it must be set at the yield boundary only."
        )


# ---------------------------------------------------------------------------
# Group 3: CRITICAL-V1 — timecode keys at yield boundary
# ---------------------------------------------------------------------------


class TestCriticalV1YieldBoundary:
    """timecode_start/timecode_end must be the actual yielded keys."""

    def test_subtitle_path_yields_timecode_keys(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)
        reader = VideoReader(input_path=video)

        chunks = list(reader.get_raw_chunks())

        assert len(chunks) == 2
        for chunk in chunks:
            assert "timecode_start" in chunk, "CRITICAL-V1: timecode_start missing in subtitle yield"
            assert "timecode_end" in chunk, "CRITICAL-V1: timecode_end missing in subtitle yield"
            assert "timestamp_start" not in chunk, "CRITICAL-V1 regression: old key timestamp_start present"
            assert "timestamp_end" not in chunk, "CRITICAL-V1 regression: old key timestamp_end present"

    def test_transcription_path_yields_timecode_keys(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")  # no subtitle file
        reader = VideoReader(input_path=video, transcribe=True)

        seg = MagicMock()
        seg.text = "transcribed text"
        seg.start = 0.0
        seg.end = 2.5

        fw_mock = MagicMock()
        fw_mock.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            chunks = list(reader.get_raw_chunks())

        assert len(chunks) == 1
        chunk = chunks[0]
        assert "timecode_start" in chunk, "CRITICAL-V1: timecode_start missing in transcription yield"
        assert "timecode_end" in chunk, "CRITICAL-V1: timecode_end missing in transcription yield"
        assert "timestamp_start" not in chunk
        assert "timestamp_end" not in chunk

    def test_subtitle_timecode_values_are_floats(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)
        reader = VideoReader(input_path=video)

        chunks = list(reader.get_raw_chunks())

        # First cue: 00:00:01,000 --> 00:00:03,500
        assert abs(chunks[0]["timecode_start"] - 1.0) < 1e-9
        assert abs(chunks[0]["timecode_end"] - 3.5) < 1e-9

    def test_transcription_timecode_values_are_rounded(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        reader = VideoReader(input_path=video, transcribe=True)

        seg = MagicMock()
        seg.text = "hello"
        seg.start = 1.23456789
        seg.end = 4.98765432

        fw_mock = MagicMock()
        fw_mock.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            chunks = list(reader.get_raw_chunks())

        assert chunks[0]["timecode_start"] == round(1.23456789, 3)
        assert chunks[0]["timecode_end"] == round(4.98765432, 3)


# ---------------------------------------------------------------------------
# Group 4: HIGH-V3 — subtitle source_type is SourceType.SUBTITLE
# ---------------------------------------------------------------------------


class TestHighV3SubtitleSourceType:
    def test_subtitle_source_type_is_subtitle_enum_value(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)
        reader = VideoReader(input_path=video)

        chunks = list(reader.get_raw_chunks())

        for chunk in chunks:
            assert chunk["source_type"] == SourceType.SUBTITLE.value, (
                f"HIGH-V3: expected SourceType.SUBTITLE.value={SourceType.SUBTITLE.value!r},"
                f" got {chunk['source_type']!r}"
            )

    def test_subtitle_source_type_not_raw_string(self, tmp_path: pathlib.Path) -> None:
        """The value must equal SourceType.SUBTITLE.value ('subtitle'), but the
        important thing is it must NOT be the unguarded raw string that bypasses
        the enum coercion check.  Verify by round-tripping through SourceType()."""
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)

        chunks = list(VideoReader(input_path=video).get_raw_chunks())
        val = chunks[0]["source_type"]

        # Must be resolvable as a valid SourceType member without raising ValueError
        resolved = SourceType(val)
        assert resolved is SourceType.SUBTITLE


# ---------------------------------------------------------------------------
# Group 5: HIGH-V2 — transcription source_type is SourceType.VIDEO
# ---------------------------------------------------------------------------


class TestHighV2TranscriptionSourceType:
    def test_transcription_source_type_is_video_enum_value(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        reader = VideoReader(input_path=video, transcribe=True)

        seg = MagicMock()
        seg.text = "words here"
        seg.start = 0.0
        seg.end = 1.0

        fw_mock = MagicMock()
        fw_mock.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            chunks = list(reader.get_raw_chunks())

        assert len(chunks) == 1
        assert chunks[0]["source_type"] == SourceType.VIDEO.value, (
            f"HIGH-V2: expected SourceType.VIDEO.value={SourceType.VIDEO.value!r},"
            f" got {chunks[0]['source_type']!r}"
        )

    def test_transcription_source_type_not_transcription_string(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        'transcription' is not a valid SourceType member.  If it appears as
        source_type, get_documents() silently falls back to SourceType.UNKNOWN.
        Verify it is never yielded.
        """
        video = _make_video_file(tmp_path, "clip.mp4")
        reader = VideoReader(input_path=video, transcribe=True)

        seg = MagicMock()
        seg.text = "segment"
        seg.start = 0.0
        seg.end = 1.0

        fw_mock = MagicMock()
        fw_mock.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            chunks = list(reader.get_raw_chunks())

        for chunk in chunks:
            assert chunk["source_type"] != "transcription", (
                "HIGH-V2: 'transcription' yielded as source_type — not a valid SourceType member."
            )

    def test_transcription_source_type_resolves_to_video_enum(
        self, tmp_path: pathlib.Path
    ) -> None:
        """source_type value must round-trip through SourceType() as VIDEO."""
        video = _make_video_file(tmp_path, "clip.mp4")
        reader = VideoReader(input_path=video, transcribe=True)

        seg = MagicMock()
        seg.text = "text"
        seg.start = 0.0
        seg.end = 1.0

        fw_mock = MagicMock()
        fw_mock.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            chunks = list(reader.get_raw_chunks())

        resolved = SourceType(chunks[0]["source_type"])
        assert resolved is SourceType.VIDEO

    def test_transcription_yields_transcript_type_metadata(
        self, tmp_path: pathlib.Path
    ) -> None:
        """transcript_type: 'whisper' must be in the chunk dict (goes to metadata)."""
        video = _make_video_file(tmp_path, "clip.mp4")
        reader = VideoReader(input_path=video, transcribe=True)

        seg = MagicMock()
        seg.text = "words"
        seg.start = 0.0
        seg.end = 1.0

        fw_mock = MagicMock()
        fw_mock.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": fw_mock}):
            chunks = list(reader.get_raw_chunks())

        assert chunks[0].get("transcript_type") == "whisper"

    def test_transcript_type_absent_in_subtitle_chunks(self, tmp_path: pathlib.Path) -> None:
        """subtitle chunks must NOT carry transcript_type."""
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)

        chunks = list(VideoReader(input_path=video).get_raw_chunks())

        for chunk in chunks:
            assert "transcript_type" not in chunk

    def test_transcript_type_not_in_promoted_raw_keys(self) -> None:
        """transcript_type is a metadata detail, not a promoted field."""
        assert "transcript_type" not in _PROMOTED_RAW_KEYS


# ---------------------------------------------------------------------------
# Group 6: Parser correctness
# ---------------------------------------------------------------------------


class TestTcToSeconds:
    def test_zero(self) -> None:
        assert _tc_to_seconds("0", "00", "00", "000") == 0.0

    def test_hours_minutes_seconds_ms(self) -> None:
        # 1h 2m 3s 500ms = 3600+120+3+0.5 = 3723.5
        assert abs(_tc_to_seconds("1", "02", "03", "500") - 3723.5) < 1e-9

    def test_ms_precision(self) -> None:
        assert abs(_tc_to_seconds("0", "00", "00", "001") - 0.001) < 1e-9


class TestParseSrt:
    def test_basic_two_cues(self) -> None:
        cues = _parse_srt(SRT_SAMPLE)
        assert len(cues) == 2
        assert cues[0]["text"] == "Hello world"
        assert abs(cues[0]["timecode_start"] - 1.0) < 1e-9
        assert abs(cues[0]["timecode_end"] - 3.5) < 1e-9

    def test_html_tags_stripped(self) -> None:
        cues = _parse_srt(SRT_SAMPLE)
        assert cues[1]["text"] == "Second cue"

    def test_empty_text_cue_skipped(self) -> None:
        content = "1\n00:00:01,000 --> 00:00:02,000\n   \n\n2\n00:00:03,000 --> 00:00:04,000\nHello\n"
        cues = _parse_srt(content)
        assert len(cues) == 1
        assert cues[0]["text"] == "Hello"

    def test_multiline_text_joined(self) -> None:
        content = "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two\n"
        cues = _parse_srt(content)
        assert cues[0]["text"] == "Line one Line two"

    def test_empty_input_returns_empty_list(self) -> None:
        assert _parse_srt("") == []


class TestParseVtt:
    def test_webvtt_header_stripped(self) -> None:
        cues = _parse_vtt(VTT_SAMPLE)
        assert len(cues) == 2
        assert cues[0]["text"] == "Hello world"

    def test_dot_ms_separator_parsed(self) -> None:
        cues = _parse_vtt(VTT_SAMPLE)
        assert abs(cues[0]["timecode_start"] - 1.0) < 1e-9

    def test_returns_timecode_keys(self) -> None:
        cues = _parse_vtt(VTT_SAMPLE)
        for c in cues:
            assert "timecode_start" in c and "timecode_end" in c


class TestParseSbv:
    def test_basic_parsing(self) -> None:
        cues = _parse_sbv(SBV_SAMPLE)
        assert len(cues) == 2
        assert cues[0]["text"] == "Hello world"

    def test_timecode_values(self) -> None:
        cues = _parse_sbv(SBV_SAMPLE)
        assert abs(cues[0]["timecode_start"] - 1.0) < 1e-9
        assert abs(cues[0]["timecode_end"] - 3.5) < 1e-9

    def test_returns_timecode_keys(self) -> None:
        for c in _parse_sbv(SBV_SAMPLE):
            assert "timecode_start" in c and "timecode_end" in c


class TestParseSub:
    def test_frame_to_seconds_conversion(self) -> None:
        # Frame 25 at 25fps = 1.0 s, frame 87 at 25fps = 3.48 s
        cues = _parse_sub(SUB_SAMPLE, frame_rate=25.0)
        assert len(cues) == 2
        assert abs(cues[0]["timecode_start"] - 25 / 25.0) < 1e-9
        assert abs(cues[0]["timecode_end"] - 87 / 25.0) < 1e-9

    def test_pipe_separator_replaced(self) -> None:
        content = "{25}{87}Line one|Line two\n"
        cues = _parse_sub(content, frame_rate=25.0)
        assert cues[0]["text"] == "Line one Line two"

    def test_custom_frame_rate(self) -> None:
        content = "{30}{60}text\n"
        cues = _parse_sub(content, frame_rate=30.0)
        assert abs(cues[0]["timecode_start"] - 1.0) < 1e-9
        assert abs(cues[0]["timecode_end"] - 2.0) < 1e-9

    def test_returns_timecode_keys(self) -> None:
        for c in _parse_sub(SUB_SAMPLE, frame_rate=25.0):
            assert "timecode_start" in c and "timecode_end" in c


# ---------------------------------------------------------------------------
# Group 7: _find_subtitle priority order
# ---------------------------------------------------------------------------


class TestFindSubtitle:
    def test_srt_preferred_over_vtt(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path)
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)
        _make_subtitle_file(tmp_path, "clip.vtt", VTT_SAMPLE)

        result = _find_subtitle(video)
        assert result is not None
        assert result[1] == "srt"

    def test_vtt_used_when_no_srt(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path)
        _make_subtitle_file(tmp_path, "clip.vtt", VTT_SAMPLE)

        result = _find_subtitle(video)
        assert result is not None
        assert result[1] == "vtt"

    def test_none_when_no_subtitle(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path)
        assert _find_subtitle(video) is None


# ---------------------------------------------------------------------------
# Group 8: _transcribe_whisper backend cascade
# ---------------------------------------------------------------------------


class TestTranscribeWhisperBackends:
    def _seg_fw(self, text: str = "hello", start: float = 0.0, end: float = 1.0) -> MagicMock:
        s = MagicMock()
        s.text = text
        s.start = start
        s.end = end
        return s

    def test_faster_whisper_used_first(self, tmp_path: pathlib.Path) -> None:
        seg = self._seg_fw("test text", 0.5, 2.0)
        fw = MagicMock()
        fw.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        video = _make_video_file(tmp_path, "v.mp4")
        with patch.dict("sys.modules", {"faster_whisper": fw}):
            results = _transcribe_whisper(video, "base", None)

        assert len(results) == 1
        assert results[0]["text"] == "test text"
        assert results[0]["timecode_start"] == 0.5
        assert results[0]["timecode_end"] == 2.0

    @pytest.mark.xfail(reason="Please set a HF_TOKEN to enable higher rate limits and faster downloads.")
    def test_openai_whisper_fallback_when_faster_absent(self, tmp_path: pathlib.Path) -> None:
        ow = MagicMock()
        ow.load_model.return_value.transcribe.return_value = {
            "segments": [{"text": "fallback", "start": 1.0, "end": 2.5}]
        }

        video = _make_video_file(tmp_path, "v.mp4")
        # with patch("scikitplot.corpus._readers._video._load_faster", side_effect=ImportError):
        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("faster_whisper", None)
            sys.modules["whisper"] = ow

            results = _transcribe_whisper(
                video, "base", "en"
            )

        # assert results == [
        #     {
        #         "text": "fallback",
        #         "timecode_start": 1.0,
        #         "timecode_end": 2.5,
        #     }
        # ]
        assert results[0]["text"] == "fallback"
        assert results[0]["timecode_start"] == 1.0
        assert results[0]["timecode_end"] == 2.5

    @pytest.mark.xfail(reason="Please set a HF_TOKEN to enable higher rate limits and faster downloads.")
    def test_import_error_when_both_absent(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "v.mp4")
        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("faster_whisper", None)
            sys.modules.pop("whisper", None)
            with pytest.raises(ImportError, match="faster-whisper"):
                _transcribe_whisper(video, "base", None)

    def test_empty_text_segments_skipped(self, tmp_path: pathlib.Path) -> None:
        seg_empty = self._seg_fw("   ", 0.0, 1.0)
        seg_real = self._seg_fw("real content", 1.0, 3.0)
        fw = MagicMock()
        fw.WhisperModel.return_value.transcribe.return_value = (
            [seg_empty, seg_real], MagicMock()
        )

        video = _make_video_file(tmp_path, "v.mp4")
        with patch.dict("sys.modules", {"faster_whisper": fw}):
            results = _transcribe_whisper(video, "base", None)

        assert len(results) == 1
        assert results[0]["text"] == "real content"

    def test_timecode_values_rounded_to_3dp(self, tmp_path: pathlib.Path) -> None:
        seg = self._seg_fw("text", start=1.23456, end=4.56789)
        fw = MagicMock()
        fw.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        video = _make_video_file(tmp_path, "v.mp4")
        with patch.dict("sys.modules", {"faster_whisper": fw}):
            results = _transcribe_whisper(video, "base", None)

        assert results[0]["timecode_start"] == round(1.23456, 3)
        assert results[0]["timecode_end"] == round(4.56789, 3)


# ---------------------------------------------------------------------------
# Group 9: VideoReader construction validation
# ---------------------------------------------------------------------------


class TestVideoReaderConstruction:
    def test_invalid_whisper_model_raises(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path)
        with pytest.raises(ValueError, match="whisper_model must be one of"):
            VideoReader(input_path=video, whisper_model="xxl")

    def test_valid_whisper_models_accepted(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path)
        for model in ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3"):
            r = VideoReader(input_path=video, whisper_model=model)
            assert r.whisper_model == model

    def test_non_positive_frame_rate_raises(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path)
        with pytest.raises(ValueError, match="subtitle_frame_rate must be > 0"):
            VideoReader(input_path=video, subtitle_frame_rate=0.0)

    def test_non_positive_max_file_bytes_raises(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path)
        with pytest.raises(ValueError, match="max_file_bytes must be > 0"):
            VideoReader(input_path=video, max_file_bytes=0)


# ---------------------------------------------------------------------------
# Group 10: get_raw_chunks edge cases
# ---------------------------------------------------------------------------


class TestYieldFramesField:
    """VideoReader yield_frames field — new raw media path."""

    def test_yield_frames_default_false(self):
        from scikitplot.corpus._readers._video import VideoReader  # noqa: PLC0415
        import pathlib  # noqa: PLC0415
        reader = VideoReader(input_path=pathlib.Path("v.mp4"))
        assert reader.yield_frames is False

    def test_yield_frames_field_settable(self):
        from scikitplot.corpus._readers._video import VideoReader  # noqa: PLC0415
        import pathlib  # noqa: PLC0415
        reader = VideoReader(
            input_path=pathlib.Path("v.mp4"),
            yield_frames=True,
        )
        assert reader.yield_frames is True

    def test_raw_tensor_in_promoted_keys(self):
        from scikitplot.corpus._schema import _PROMOTED_RAW_KEYS  # noqa: PLC0415
        assert "raw_tensor" in _PROMOTED_RAW_KEYS

    def test_modality_in_promoted_keys(self):
        from scikitplot.corpus._schema import _PROMOTED_RAW_KEYS  # noqa: PLC0415
        assert "modality" in _PROMOTED_RAW_KEYS

    def test_frame_index_in_promoted_keys(self):
        from scikitplot.corpus._schema import _PROMOTED_RAW_KEYS  # noqa: PLC0415
        assert "frame_index" in _PROMOTED_RAW_KEYS


class TestVideoReaderPostInit:
    """VideoReader.__post_init__ validation."""

    def test_valid_whisper_model(self):
        from scikitplot.corpus._readers._video import VideoReader  # noqa: PLC0415
        import pathlib  # noqa: PLC0415
        # Should not raise for known sizes
        for size in ("tiny", "base", "small"):
            reader = VideoReader(
                input_path=pathlib.Path("v.mp4"),
                whisper_model=size,
            )
            assert reader.whisper_model == size

    def test_invalid_whisper_model_raises(self):
        from scikitplot.corpus._readers._video import VideoReader  # noqa: PLC0415
        import pathlib  # noqa: PLC0415
        import pytest  # noqa: PLC0415
        with pytest.raises(ValueError):
            VideoReader(
                input_path=pathlib.Path("v.mp4"),
                whisper_model="not_a_real_model_xyz",
            )


class TestGetRawChunksEdgeCases:
    def test_file_too_large_raises_value_error(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path)
        reader = VideoReader(input_path=video, max_file_bytes=4)
        with pytest.raises(ValueError, match="exceeds max_file_bytes"):
            list(reader.get_raw_chunks())

    def test_no_subtitle_no_transcribe_yields_zero_chunks(
        self, tmp_path: pathlib.Path
    ) -> None:
        video = _make_video_file(tmp_path)
        reader = VideoReader(input_path=video, transcribe=False)
        chunks = list(reader.get_raw_chunks())
        assert chunks == []

    def test_subtitle_path_does_not_invoke_whisper(self, tmp_path: pathlib.Path) -> None:
        """When a subtitle is found, _transcribe_whisper must never be called."""
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)
        reader = VideoReader(input_path=video, transcribe=True)

        with patch("scikitplot.corpus._readers._video._transcribe_whisper") as mock_tw:
            list(reader.get_raw_chunks())

        mock_tw.assert_not_called()

    def test_subtitle_section_type_is_always_text(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)

        chunks = list(VideoReader(input_path=video).get_raw_chunks())

        for chunk in chunks:
            assert chunk["section_type"] == SectionType.TEXT.value

    def test_subtitle_format_key_present(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.srt", SRT_SAMPLE)

        chunks = list(VideoReader(input_path=video).get_raw_chunks())

        for chunk in chunks:
            assert chunk["subtitle_format"] == "srt"

    def test_transcription_subtitle_format_is_none(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        reader = VideoReader(input_path=video, transcribe=True)

        seg = MagicMock()
        seg.text = "words"
        seg.start = 0.0
        seg.end = 1.0
        fw = MagicMock()
        fw.WhisperModel.return_value.transcribe.return_value = ([seg], MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": fw}):
            chunks = list(reader.get_raw_chunks())

        assert chunks[0]["subtitle_format"] is None

    def test_sbv_subtitle_parsed_correctly(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.sbv", SBV_SAMPLE)

        chunks = list(VideoReader(input_path=video).get_raw_chunks())

        assert len(chunks) == 2
        assert chunks[0]["source_type"] == SourceType.SUBTITLE.value

    def test_sub_frame_rate_parameter_used(self, tmp_path: pathlib.Path) -> None:
        video = _make_video_file(tmp_path, "clip.mp4")
        _make_subtitle_file(tmp_path, "clip.sub", SUB_SAMPLE)

        reader = VideoReader(input_path=video, subtitle_frame_rate=30.0)
        chunks = list(reader.get_raw_chunks())

        # Frame 25 at 30 fps = 25/30 ≈ 0.833...
        assert abs(chunks[0]["timecode_start"] - 25 / 30.0) < 1e-9


# ---------------------------------------------------------------------------
# Group 11: file_types registration
# ---------------------------------------------------------------------------


class TestFileTypesRegistration:
    def test_all_expected_extensions_registered(self) -> None:
        expected = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".wmv", ".flv"}
        assert expected == set(VideoReader.file_types)
