"""
Tests for scikitplot.corpus._readers._web
=========================================

Coverage
--------
CRITICAL-W1
    ``timecode_start`` and ``timecode_end`` are the yielded keys at the
    ``YouTubeReader.get_raw_chunks`` boundary.
    Old wrong keys ``timestamp_start`` / ``timestamp_end`` must be absent.
    Both promoted keys must be in ``_PROMOTED_RAW_KEYS``.

HIGH-W2
    ``YouTubeReader`` yields ``"source_type": SourceType.VIDEO.value``.
    The raw strings ``"manual"`` / ``"auto_generated"`` are not valid
    ``SourceType`` members and must NOT appear as ``source_type`` values.
    The caption-track sub-type is preserved via ``"transcript_type"``
    (``"manual"`` or ``"auto_generated"``) which goes to ``metadata``.

BUG-W3
    ``WebReader.get_raw_chunks`` now yields
    ``"source_type": SourceType.WEB.value`` so every fetched web-page
    document gets the correct type rather than ``SourceType.UNKNOWN``.

Additional coverage
    * ``_PROMOTED_RAW_KEYS`` contract verification.
    * ``WebReader`` yield dict keys: section_type, html_tag, url,
      element_index all present and correct.
    * ``WebReader.validate_input`` rejects non-HTTP URLs.
    * ``YouTubeReader`` timecode values (start, start+duration) round to 3 dp.
    * ``YouTubeReader`` prefers manual transcript; falls back to auto-generated.
    * ``YouTubeReader`` final fallback uses first available transcript.
    * ``YouTubeReader`` ``transcript_type`` reflects which track was used.
    * ``YouTubeReader`` raises ``RuntimeError`` when no transcripts available.
    * ``YouTubeReader`` raises ``ImportError`` when library absent.
    * ``YouTubeReader.validate_input`` rejects non-YouTube URLs.
    * ``transcript_type`` not in ``_PROMOTED_RAW_KEYS``.

All tests use ``unittest.mock`` — no network access, no YouTube API, no
``requests``, and no ``beautifulsoup4`` installation required.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scikitplot.corpus import _base, _schema, _readers
from scikitplot.corpus._readers._web import (   # noqa: E402
    WebReader,
    YouTubeReader,
    _extract_youtube_id,
    _section_type_for_tag,
)
from scikitplot.corpus._schema import (        # noqa: E402
    SectionType,
    SourceType,
    _PROMOTED_RAW_KEYS,
)

# Module reference for monkey-patching
from scikitplot.corpus._readers import _web
import scikitplot.corpus._readers._web as _web_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YT_URL = "https://www.youtube.com/watch?v=rwPISgZcYIk"
_YT_ID  = "rwPISgZcYIk"


def _yt_reader(**kwargs: Any) -> YouTubeReader:
    return YouTubeReader(
        input_file=pathlib.Path(_YT_URL),
        source_uri=_YT_URL,
        **kwargs,
    )


def _cue(text: str = "Hello world", start: float = 1.0, duration: float = 2.5) -> dict:
    return {"text": text, "start": start, "duration": duration}


def _make_transcript_list(
    manual_cues: list[dict] | None = None,
    auto_cues: list[dict] | None = None,
    *,
    first_is_generated: bool = False,
) -> MagicMock:
    """Build a minimal mock for YouTubeTranscriptApi.list_transcripts()."""
    tlist = MagicMock()

    manual_t = MagicMock()
    manual_t.is_generated = False
    manual_t.language_code = "en"
    manual_t.fetch.return_value = manual_cues or [_cue()]

    auto_t = MagicMock()
    auto_t.is_generated = True
    auto_t.language_code = "en"
    auto_t.fetch.return_value = auto_cues or [_cue("auto text")]

    if manual_cues is not None:
        tlist.find_manually_created_transcript.return_value = manual_t
    else:
        tlist.find_manually_created_transcript.side_effect = Exception("no manual")

    if auto_cues is not None:
        tlist.find_generated_transcript.return_value = auto_t
    else:
        tlist.find_generated_transcript.side_effect = Exception("no auto")

    # fallback list()
    fallback = auto_t if first_is_generated else manual_t
    tlist.__iter__ = MagicMock(return_value=iter([fallback]))

    return tlist


def _make_yt_api_mock(transcript_list: MagicMock) -> MagicMock:
    """Return a sys.modules stub for youtube_transcript_api."""
    api_m = MagicMock()
    # IMPORTANT: instance().list()
    api_m.YouTubeTranscriptApi.return_value.list.return_value = transcript_list
    api_m.CouldNotRetrieveTranscript = Exception
    api_m.NoTranscriptFound = Exception
    api_m.TranscriptsDisabled = Exception
    api_m.VideoUnavailable = Exception
    return api_m


def _make_web_reader(url: str = "https://example.com") -> WebReader:
    return WebReader(
        input_file=pathlib.Path(url),
        source_uri=url,
    )


def _make_requests_mock(html: str = "<html><body><p>Hello</p></body></html>") -> MagicMock:
    req_m = MagicMock()
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200
    resp.encoding = "utf-8"
    resp.iter_content.return_value = iter([html.encode("utf-8")])
    req_m.get.return_value = resp
    return req_m


def _make_bs4_mock(extracted_texts: list[tuple[str, str]]) -> MagicMock:
    """
    Return a bs4 stub.

    Parameters
    ----------
    extracted_texts : list of (tag_name, text)
        Elements that soup.find_all() should return, in order.
    """
    bs4_m = MagicMock()
    elements = []
    for tag_name, text in extracted_texts:
        el = MagicMock()
        el.name = tag_name
        el.get_text.return_value = text
        elements.append(el)

    def _find_all(tags, **kw):
        return [e for e in elements if e.name in tags]

    soup = MagicMock()
    soup.find_all.side_effect = _find_all
    soup.__call__ = lambda self, tags: []   # soup(["script", "style", "noscript"])
    soup.return_value = []
    bs4_m.BeautifulSoup.return_value = soup
    return bs4_m


# ---------------------------------------------------------------------------
# Group 1: _PROMOTED_RAW_KEYS contract
# ---------------------------------------------------------------------------


class TestPromotedKeyContract:
    def test_timecode_start_in_promoted_raw_keys(self) -> None:
        assert "timecode_start" in _PROMOTED_RAW_KEYS

    def test_timecode_end_in_promoted_raw_keys(self) -> None:
        assert "timecode_end" in _PROMOTED_RAW_KEYS

    def test_old_timestamp_start_absent(self) -> None:
        assert "timestamp_start" not in _PROMOTED_RAW_KEYS

    def test_old_timestamp_end_absent(self) -> None:
        assert "timestamp_end" not in _PROMOTED_RAW_KEYS

    def test_source_type_in_promoted_raw_keys(self) -> None:
        assert "source_type" in _PROMOTED_RAW_KEYS

    def test_url_in_promoted_raw_keys(self) -> None:
        assert "url" in _PROMOTED_RAW_KEYS

    def test_transcript_type_not_in_promoted_raw_keys(self) -> None:
        """transcript_type is a metadata detail, NOT a promoted field."""
        assert "transcript_type" not in _PROMOTED_RAW_KEYS


# ---------------------------------------------------------------------------
# Group 2: CRITICAL-W1 — YouTubeReader timecode keys at yield boundary
# ---------------------------------------------------------------------------


class TestCriticalW1YoutubeTimecodeKeys:
    def _get_chunks(self, cues: list[dict] | None = None) -> list[dict]:
        tlist = _make_transcript_list(manual_cues=cues or [_cue()])
        api_m = _make_yt_api_mock(tlist)
        with patch.dict("sys.modules", {"youtube_transcript_api": api_m}):
            return list(_yt_reader().get_raw_chunks())

    def test_timecode_start_present(self) -> None:
        chunks = self._get_chunks()
        assert len(chunks) == 1
        assert "timecode_start" in chunks[0], "CRITICAL-W1: timecode_start missing"

    def test_timecode_end_present(self) -> None:
        chunks = self._get_chunks()
        assert "timecode_end" in chunks[0], "CRITICAL-W1: timecode_end missing"

    def test_old_timestamp_start_absent(self) -> None:
        chunks = self._get_chunks()
        assert "timestamp_start" not in chunks[0], "CRITICAL-W1 regression: old key present"

    def test_old_timestamp_end_absent(self) -> None:
        chunks = self._get_chunks()
        assert "timestamp_end" not in chunks[0], "CRITICAL-W1 regression: old key present"

    def test_timecode_start_value_equals_cue_start(self) -> None:
        chunks = self._get_chunks([_cue(start=3.5, duration=1.25)])
        assert abs(chunks[0]["timecode_start"] - 3.5) < 1e-9

    def test_timecode_end_value_equals_start_plus_duration(self) -> None:
        chunks = self._get_chunks([_cue(start=3.5, duration=1.25)])
        assert abs(chunks[0]["timecode_end"] - (3.5 + 1.25)) < 1e-9

    def test_timecode_values_rounded_to_3dp(self) -> None:
        chunks = self._get_chunks([_cue(start=1.23456789, duration=2.98765432)])
        assert chunks[0]["timecode_start"] == round(1.23456789, 3)
        assert chunks[0]["timecode_end"] == round(1.23456789 + 2.98765432, 3)

    def test_timecode_start_in_promoted_raw_keys(self) -> None:
        """Ensure the key name matches the _PROMOTED_RAW_KEYS contract."""
        chunks = self._get_chunks()
        assert chunks[0]["timecode_start"] is not None   # key exists and is usable
        assert "timecode_start" in _PROMOTED_RAW_KEYS


# ---------------------------------------------------------------------------
# Group 3: HIGH-W2 — YouTubeReader source_type uses SourceType enum
# ---------------------------------------------------------------------------


class TestHighW2YoutubeSourceType:
    def _get_chunks(
        self,
        manual_cues: list[dict] | None = None,
        auto_cues: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        tlist = _make_transcript_list(manual_cues=manual_cues, auto_cues=auto_cues)
        api_m = _make_yt_api_mock(tlist)
        with patch.dict("sys.modules", {"youtube_transcript_api": api_m}):
            return list(_yt_reader(**kwargs).get_raw_chunks())

    def test_source_type_is_video_enum_value(self) -> None:
        chunks = self._get_chunks(manual_cues=[_cue()])
        assert chunks[0]["source_type"] == SourceType.VIDEO.value, (
            f"HIGH-W2: expected {SourceType.VIDEO.value!r}, got {chunks[0]['source_type']!r}"
        )

    def test_source_type_resolves_to_video_enum(self) -> None:
        chunks = self._get_chunks(manual_cues=[_cue()])
        resolved = SourceType(chunks[0]["source_type"])
        assert resolved is SourceType.VIDEO

    def test_source_type_never_manual_string(self) -> None:
        chunks = self._get_chunks(manual_cues=[_cue()])
        assert chunks[0]["source_type"] != "manual", (
            "HIGH-W2: 'manual' is not a SourceType member — would cause UNKNOWN"
        )

    def test_source_type_never_auto_generated_string(self) -> None:
        # Use auto-generated path
        chunks = self._get_chunks(auto_cues=[_cue("auto")])
        assert chunks[0]["source_type"] != "auto_generated", (
            "HIGH-W2: 'auto_generated' is not a SourceType member — would cause UNKNOWN"
        )

    def test_source_type_video_for_manual_track(self) -> None:
        chunks = self._get_chunks(manual_cues=[_cue()])
        assert SourceType(chunks[0]["source_type"]) is SourceType.VIDEO

    def test_source_type_video_for_auto_generated_track(self) -> None:
        """Even when using auto-generated captions, source_type must be VIDEO."""
        chunks = self._get_chunks(auto_cues=[_cue("auto text")])
        assert SourceType(chunks[0]["source_type"]) is SourceType.VIDEO

    def test_transcript_type_manual_when_manual_track_used(self) -> None:
        chunks = self._get_chunks(manual_cues=[_cue()])
        assert chunks[0]["transcript_type"] == "manual"

    def test_transcript_type_auto_generated_when_fallback_used(self) -> None:
        chunks = self._get_chunks(auto_cues=[_cue("auto text")])
        assert chunks[0]["transcript_type"] == "auto_generated"

    def test_transcript_type_not_in_promoted_raw_keys(self) -> None:
        assert "transcript_type" not in _PROMOTED_RAW_KEYS


# ---------------------------------------------------------------------------
# Group 4: BUG-W3 — WebReader yields source_type
# ---------------------------------------------------------------------------


class TestBugW3WebReaderSourceType:
    def _get_chunks(
        self,
        elements: list[tuple[str, str]] | None = None,
    ) -> list[dict]:
        els = elements or [("p", "Hello world"), ("h1", "Title text")]
        req_m = _make_requests_mock()
        bs4_m = _make_bs4_mock(els)
        reader = _make_web_reader()
        with patch.dict("sys.modules", {"requests": req_m, "bs4": bs4_m}):
            return list(reader.get_raw_chunks())

    def test_source_type_present_in_web_chunks(self) -> None:
        chunks = self._get_chunks([("p", "Hello")])
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "source_type" in chunk, "BUG-W3: source_type missing from WebReader yield"

    def test_source_type_is_web_enum_value(self) -> None:
        chunks = self._get_chunks([("p", "Hello")])
        for chunk in chunks:
            assert chunk["source_type"] == SourceType.WEB.value, (
                f"BUG-W3: expected {SourceType.WEB.value!r}, got {chunk['source_type']!r}"
            )

    def test_source_type_resolves_to_web_enum(self) -> None:
        chunks = self._get_chunks([("p", "Hello")])
        for chunk in chunks:
            resolved = SourceType(chunk["source_type"])
            assert resolved is SourceType.WEB

    def test_source_type_not_unknown(self) -> None:
        chunks = self._get_chunks([("p", "text")])
        for chunk in chunks:
            assert SourceType(chunk["source_type"]) is not SourceType.UNKNOWN


# ---------------------------------------------------------------------------
# Group 5: WebReader yield dict completeness
# ---------------------------------------------------------------------------


class TestWebReaderYieldKeys:
    def _get_chunks(self, elements: list[tuple[str, str]]) -> list[dict]:
        req_m = _make_requests_mock()
        bs4_m = _make_bs4_mock(elements)
        reader = _make_web_reader("https://example.com/page")
        with patch.dict("sys.modules", {"requests": req_m, "bs4": bs4_m}):
            return list(reader.get_raw_chunks())

    def test_required_keys_present(self) -> None:
        chunks = self._get_chunks([("p", "Hello"), ("h2", "Section")])
        for chunk in chunks:
            for key in ("text", "section_type", "source_type", "html_tag", "url", "element_index"):
                assert key in chunk, f"key {key!r} missing from WebReader yield"

    def test_element_index_increments(self) -> None:
        chunks = self._get_chunks([("p", "First"), ("p", "Second"), ("p", "Third")])
        indices = [c["element_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_url_value_is_correct(self) -> None:
        chunks = self._get_chunks([("p", "text")])
        assert all(c["url"] == "https://example.com/page" for c in chunks)

    def test_section_type_title_for_h1(self) -> None:
        chunks = self._get_chunks([("h1", "Main Title")])
        assert any(c["section_type"] == SectionType.TITLE.value for c in chunks)

    def test_section_type_header_for_h2(self) -> None:
        chunks = self._get_chunks([("h2", "Sub-heading")])
        assert any(c["section_type"] == SectionType.HEADER.value for c in chunks)

    def test_section_type_text_for_paragraph(self) -> None:
        chunks = self._get_chunks([("p", "Paragraph text")])
        assert any(c["section_type"] == SectionType.TEXT.value for c in chunks)

    def test_html_tag_value_matches_element(self) -> None:
        chunks = self._get_chunks([("p", "text"), ("h2", "heading")])
        tags = {c["html_tag"] for c in chunks}
        assert "p" in tags
        assert "h2" in tags


# ---------------------------------------------------------------------------
# Group 6: WebReader.validate_input
# ---------------------------------------------------------------------------


class TestWebReaderValidation:
    def test_http_url_accepted(self) -> None:
        r = _make_web_reader("http://example.com")
        r.validate_input()  # must not raise

    def test_https_url_accepted(self) -> None:
        r = _make_web_reader("https://example.com")
        r.validate_input()  # must not raise

    def test_non_http_url_raises(self) -> None:
        r = _make_web_reader("ftp://example.com")
        with pytest.raises(ValueError, match="http"):
            r.validate_input()

    def test_bare_path_raises(self) -> None:
        r = _make_web_reader("/local/path/file.txt")
        with pytest.raises(ValueError):
            r.validate_input()


# ---------------------------------------------------------------------------
# Group 7: YouTubeReader transcript selection logic
# ---------------------------------------------------------------------------


class TestYoutubeTranscriptSelection:
    def _chunks_for(
        self,
        manual_cues: list[dict] | None,
        auto_cues: list[dict] | None,
        **kwargs: Any,
    ) -> list[dict]:
        tlist = _make_transcript_list(manual_cues=manual_cues, auto_cues=auto_cues)
        api_m = _make_yt_api_mock(tlist)
        with patch.dict("sys.modules", {"youtube_transcript_api": api_m}):
            return list(_yt_reader(**kwargs).get_raw_chunks())

    def test_manual_transcript_preferred(self) -> None:
        manual_cue = _cue("manual text", start=1.0, duration=2.0)
        chunks = self._chunks_for(manual_cues=[manual_cue], auto_cues=None)
        assert chunks[0]["text"] == "manual text"
        assert chunks[0]["transcript_type"] == "manual"

    def test_auto_generated_fallback_when_no_manual(self) -> None:
        auto_cue = _cue("auto text", start=5.0, duration=3.0)
        chunks = self._chunks_for(manual_cues=None, auto_cues=[auto_cue])
        assert chunks[0]["text"] == "auto text"
        assert chunks[0]["transcript_type"] == "auto_generated"

    def test_no_fallback_when_include_auto_generated_false(self) -> None:
        tlist = _make_transcript_list(manual_cues=None, auto_cues=None)
        tlist.__iter__ = MagicMock(return_value=iter([]))
        tlist.__bool__ = lambda s: False
        api_m = _make_yt_api_mock(tlist)
        with patch.dict("sys.modules", {"youtube_transcript_api": api_m}):
            with pytest.raises(RuntimeError, match="no transcripts"):
                list(_yt_reader(include_auto_generated=False).get_raw_chunks())

    def test_empty_cue_text_skipped(self) -> None:
        cues = [
            {"text": "  ", "start": 0.0, "duration": 1.0},
            {"text": "real text", "start": 1.0, "duration": 2.0},
        ]
        chunks = self._chunks_for(manual_cues=cues, auto_cues=None)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "real text"

    def test_video_id_in_chunk(self) -> None:
        chunks = self._chunks_for(manual_cues=[_cue()], auto_cues=None)
        assert chunks[0]["video_id"] == _YT_ID

    def test_transcript_language_in_chunk(self) -> None:
        chunks = self._chunks_for(manual_cues=[_cue()], auto_cues=None)
        assert "transcript_language" in chunks[0]
        assert chunks[0]["transcript_language"] == "en"


# ---------------------------------------------------------------------------
# Group 8: YouTubeReader error handling
# ---------------------------------------------------------------------------


class TestYoutubeReaderErrors:
    def test_missing_library_raises_import_error(self) -> None:
        # this fail
        # with patch.dict(sys.modules, {}, clear=False):
        #     sys.modules.pop("youtube_transcript_api", None)
        with patch.dict("sys.modules", {"youtube_transcript_api": None}):
            with pytest.raises(ImportError, match="youtube-transcript-api"):
                list(_yt_reader().get_raw_chunks())

    def test_invalid_youtube_url_raises_value_error(self) -> None:
        reader = YouTubeReader(
            input_file=pathlib.Path("https://example.com/notYT"),
            source_uri="https://example.com/notYT",
        )
        with pytest.raises(ValueError, match="YouTube"):
            reader.validate_input()

    def test_runtime_error_when_no_transcripts_at_all(self) -> None:
        class EmptyTranscriptList:
            def find_manually_created_transcript(self, *_):
                raise Exception("no manual")

            def find_generated_transcript(self, *_):
                raise Exception("no auto")

            def __iter__(self):
                return iter([])

        # tlist = EmptyTranscriptList()
        tlist = _make_transcript_list(manual_cues=None, auto_cues=None)
        tlist.__iter__ = MagicMock(return_value=iter([]))
        api_m = _make_yt_api_mock(tlist)
        with patch.dict("sys.modules", {"youtube_transcript_api": api_m}):
            with pytest.raises(RuntimeError, match="no transcripts"):
                list(_yt_reader().get_raw_chunks())

    def test_transcript_disabled_raises_runtime_error(self) -> None:
        api_m = MagicMock()

        class FakeDisabled(Exception):
            pass

        api_m.TranscriptsDisabled = FakeDisabled
        api_m.VideoUnavailable = Exception
        api_m.CouldNotRetrieveTranscript = Exception
        api_m.NoTranscriptFound = Exception
        # IMPORTANT: instance().list()
        api_m.YouTubeTranscriptApi.return_value.list.side_effect = FakeDisabled("disabled")

        with patch.dict("sys.modules", {"youtube_transcript_api": api_m}):
            with pytest.raises(RuntimeError, match="could not retrieve"):
                list(_yt_reader().get_raw_chunks())


# ---------------------------------------------------------------------------
# Group 9: _extract_youtube_id
# ---------------------------------------------------------------------------


class TestExtractYoutubeId:
    def test_watch_url(self) -> None:
        assert _extract_youtube_id("https://www.youtube.com/watch?v=rwPISgZcYIk") == "rwPISgZcYIk"

    def test_short_url(self) -> None:
        assert _extract_youtube_id("https://youtu.be/rwPISgZcYIk") == "rwPISgZcYIk"

    def test_non_youtube_returns_none(self) -> None:
        assert _extract_youtube_id("https://example.com/page") is None

    def test_no_id_returns_none(self) -> None:
        assert _extract_youtube_id("https://www.youtube.com/watch") is None


# ---------------------------------------------------------------------------
# Group 10: _section_type_for_tag
# ---------------------------------------------------------------------------


class TestSectionTypeForTag:
    def test_title_tag(self) -> None:
        assert _section_type_for_tag("title") == SectionType.TITLE.value

    def test_h1_is_title(self) -> None:
        assert _section_type_for_tag("h1") == SectionType.TITLE.value

    def test_h2_is_header(self) -> None:
        assert _section_type_for_tag("h2") == SectionType.HEADER.value

    def test_h6_is_header(self) -> None:
        assert _section_type_for_tag("h6") == SectionType.HEADER.value

    def test_p_is_text(self) -> None:
        assert _section_type_for_tag("p") == SectionType.TEXT.value

    def test_li_is_text(self) -> None:
        assert _section_type_for_tag("li") == SectionType.TEXT.value
