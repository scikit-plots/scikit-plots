# scikitplot/corpus/_chunkers/tests/test__notices_and_scripts.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Log hygiene and script detection (Run 3, R3-7).

Constructing one ``SemanticChunker`` and chunking a text logged the same
missing-segmenter warnings again and again; the ``'_unknown'`` script
warning was the symptom of ``ScriptSegmenter`` calling ``regex.get_script``,
which does not exist, so every text was one span of unknown script. The
script tests need the optional ``regex`` package and are skipped without it.
"""

from __future__ import annotations

import logging
import random

import pytest

from ... import _capabilities
from ..._notices import notice_once, reset_notices
from .._custom_tokenizer import _TOKENIZER_REGISTRY, ScriptSegmenter, ScriptType
from .._semantic import SemanticChunker, SemanticChunkerConfig
from .._writing_system import DictionaryBoundaryStrategy, WritingSystemAdapterConfig

TEXT = "東京は大きい。大阪も大きい。\n\n今日は晴れです。Hello world. 北京很大。"


@pytest.fixture(autouse=True)
def _fresh_notices():
    reset_notices()
    yield
    reset_notices()


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def records():
    handler = _Records()
    logger = logging.getLogger("scikitplot")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    yield handler.records
    logger.removeHandler(handler)
    logger.setLevel(previous)


def _warnings(records):
    return [r.getMessage() for r in records if r.levelno >= logging.WARNING]


class TestNoticeOnce:
    def test_once_per_key(self, records):
        logger = logging.getLogger("scikitplot.test_notices")
        assert notice_once(logger, "k", "missing %s", "x")
        assert not notice_once(logger, "k", "missing %s", "x")
        assert notice_once(logger, "other", "missing %s", "y")
        assert _warnings(records) == ["missing x", "missing y"]

    def test_a_filtered_notice_is_not_used_up(self, records):
        """Review 5: a notice suppressed by the log level was spent."""
        logger = logging.getLogger("scikitplot.test_notices")
        logging.disable(logging.WARNING)
        try:
            assert not notice_once(logger, "k", "m")
        finally:
            logging.disable(logging.NOTSET)
        assert notice_once(logger, "k", "m")
        assert _warnings(records) == ["m"]

    def test_reset(self, records):
        logger = logging.getLogger("scikitplot.test_notices")
        notice_once(logger, "k", "m")
        reset_notices()
        notice_once(logger, "k", "m")
        assert _warnings(records) == ["m", "m"]


class TestSemanticChunkerNoise:
    def test_each_missing_segmenter_is_reported_once(self, records):
        """The review counted 7 warnings for 2 chunkers and 3 calls."""
        for _ in range(2):
            chunker = SemanticChunker(SemanticChunkerConfig(backend="morphological"))
            chunker.chunk(TEXT)
            chunker.chunk(TEXT)
        messages = _warnings(records)
        assert len(messages) == len(set(messages))

    def test_a_per_text_fallback_is_reported_once(self, records, monkeypatch):
        strategy = DictionaryBoundaryStrategy()
        monkeypatch.setattr(strategy, "_probe", lambda name: False)
        config = WritingSystemAdapterConfig()
        for _ in range(3):
            strategy._han_segment("北京很大", 0, config)
            strategy._thai_segment("สวัสดี", 0, config)
        messages = _warnings(records)
        assert len(messages) == 2
        assert all("component_capabilities" in m for m in messages)


class TestCapabilities:
    def test_every_optional_segmenter_is_reported(self):
        report = _capabilities.component_capabilities(role="segmentation")
        assert set(report) == {
            "segmenter:regex",
            "segmenter:jieba",
            "segmenter:mecab",
            "segmenter:fugashi",
            "segmenter:sudachipy",
            "segmenter:kss",
            "segmenter:pythainlp",
            "segmenter:khmer-nltk",
            "segmenter:camel-tools",
        }
        assert all(record["remedy"] for record in report.values())


class TestScriptMap:
    def test_every_mapped_script_is_a_script_type(self):
        """A value outside ScriptType raised and fell back with a warning."""
        values = {t.value for t in ScriptType}
        for name, value in ScriptSegmenter.SCRIPT_PROPERTY_MAP.items():
            assert value in values or value in {"_common", "_inherited"}, name


class TestScriptSegmenter:
    @pytest.fixture(autouse=True)
    def _regex(self):
        pytest.importorskip("regex")

    def test_scripts_are_detected(self):
        """Every text used to be one UNKNOWN span."""
        spans = ScriptSegmenter().segment("Hello مرحبا world 東京")
        assert [s.script.value for s in spans] == ["latin", "arabic", "latin", "han"]

    @pytest.mark.parametrize(
        "text",
        [
            "cafe\u0301 東京👩\u200d👩\u200d👧x",
            "ᚠᚢᚦ runes 𐌰",
            "  123 !! ",
            "Привет, мир! Hello.",
        ],
    )
    def test_offsets_are_characters_and_cover_the_text(self, text):
        """They were grapheme indices, which every consumer read as chars."""
        spans = ScriptSegmenter().segment(text)
        assert all(text[s.start : s.end] == s.text for s in spans)
        assert "".join(s.text for s in spans) == text
        assert spans[0].start == 0
        assert spans[-1].end == len(text)

    def test_short_spans_merge_by_grapheme_count_into_one_run(self):
        text = "Hello 東 world e\u0301e\u0301"
        spans = ScriptSegmenter(min_span_chars=3).segment(text)
        assert [(s.script, s.start, s.end) for s in spans] == [
            (ScriptType.LATIN, 0, len(text))
        ]

    def test_japanese_and_korean_stay_whole(self):
        """Review 1: 住んでいます was cut at the kanji/kana boundary."""
        spans = ScriptSegmenter().segment("東京都に住んでいます。Hello")
        assert [(s.script, s.text) for s in spans] == [
            (ScriptType.HIRAGANA, "東京都に住んでいます。"),
            (ScriptType.LATIN, "Hello"),
        ]
        korean = ScriptSegmenter().segment("漢字도 좋다")
        assert [(s.script, s.text) for s in korean] == [
            (ScriptType.HANGUL, "漢字도 좋다")
        ]
        chinese = ScriptSegmenter().segment("北京很大")
        assert [s.script for s in chinese] == [ScriptType.HAN]

    def test_a_sentence_boundary_separates_writing_systems(self):
        """Re-review 1: a Chinese sentence went to the Korean strategy."""
        spans = ScriptSegmenter().segment("한국어 문장. 中国人很好。キャベツ")
        assert [s.script for s in spans] == [
            ScriptType.HANGUL,
            ScriptType.HAN,
            ScriptType.KATAKANA,
        ]

    def test_every_run_is_one_span(self):
        """Re-review probe: two adjacent HANGUL spans around a full stop."""
        spans = ScriptSegmenter().segment("な글字。한カa글漢")
        scripts = [s.script for s in spans]
        assert all(a != b for a, b in zip(scripts, scripts[1:]))

    @pytest.mark.parametrize("min_span_chars", [1, 2, 4])
    def test_spans_cover_the_text_one_per_run(self, min_span_chars):
        """A fixed-seed fuzz over Han, kana, Hangul, Latin, digits, marks."""
        rng = random.Random(0)  # noqa: S311 (a reproducible fuzz, not security)
        alphabet = "漢字かなカナ한글ab1 。,\u0301"
        segmenter = ScriptSegmenter(min_span_chars=min_span_chars)
        for _ in range(500):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 14)))
            spans = segmenter.segment(text)
            assert "".join(s.text for s in spans) == text
            assert all(text[s.start : s.end] == s.text and s.text for s in spans)
            scripts = [s.script for s in spans]
            assert all(a != b for a, b in zip(scripts, scripts[1:])), (text, spans)

    def test_a_short_span_merge_keeps_the_writing_system(self):
        """Re-review 3: kana merged into Han was labelled Han."""
        spans = ScriptSegmenter(min_span_chars=3).segment("東京x に")
        assert [s.script for s in spans] == [ScriptType.HIRAGANA]

    def test_script_free_text_is_one_span(self):
        """Re-review 6: Inherited and Common made two UNKNOWN spans."""
        spans = ScriptSegmenter().segment("\u0301 。")
        assert [(s.script, s.text) for s in spans] == [
            (ScriptType.UNKNOWN, "\u0301 。")
        ]

    def test_unnamed_characters_share_one_notice(self, records):
        """Re-review 2: each private-use character had its own notice."""
        text = " a ".join(chr(0xE000 + k) for k in range(50))
        ScriptSegmenter().segment(text)
        assert len(_warnings(records)) == 1

    def test_scripts_sharing_a_first_word_are_told_apart(self, records):
        ScriptSegmenter().segment("\U00010300 a \U000103a0")  # Old Italic, Old Persian
        assert len(_warnings(records)) == 2

    def test_a_japanese_tokenizer_sees_whole_sentences(self, monkeypatch):
        seen = []

        class Recorder:
            def tokenize(self, text):
                seen.append(text)
                return [text]

        monkeypatch.setitem(_TOKENIZER_REGISTRY._store, "japanese", Recorder())
        source = "東京都に住んでいます。"
        SemanticChunker(SemanticChunkerConfig(backend="morphological")).chunk(source)
        assert seen
        assert all("住んでいます" in text for text in seen if "住" in text)

    @pytest.mark.parametrize(
        ("text", "script"),
        [
            ("தமிழ்", ScriptType.SOUTH_ASIAN),
            ("සිංහල", ScriptType.SOUTH_ASIAN),
            ("ພາສາລາວ", ScriptType.SOUTHEAST_ASIAN),
        ],
    )
    def test_documented_scripts_are_mapped(self, text, script):
        """Review 3: they came out UNKNOWN."""
        assert [s.script for s in ScriptSegmenter().segment(text)] == [script]

    def test_each_unmapped_script_is_noticed_once(self, records):
        """Review 6: one key for every unmapped script."""
        segmenter = ScriptSegmenter()
        for _ in range(2):
            segmenter.segment("ᚠᚢᚦ")
            segmenter.segment("𐌰𐌱")
        assert len(_warnings(records)) == 2

    def test_script_free_text_is_unknown_without_a_notice(self, records):
        spans = ScriptSegmenter().segment("  123 !! ")
        assert [s.script for s in spans] == [ScriptType.UNKNOWN]
        assert _warnings(records) == []

    def test_an_unmapped_script_is_its_own_span_and_noticed_once(self, records):
        segmenter = ScriptSegmenter()
        for _ in range(3):
            spans = segmenter.segment("ᚠᚢᚦ runes")
        assert [(s.script, s.text) for s in spans] == [
            (ScriptType.UNKNOWN, "ᚠᚢᚦ "),
            (ScriptType.LATIN, "runes"),
        ]
        assert len(_warnings(records)) == 1
        assert "U+16A0" in _warnings(records)[0]

    def test_the_semantic_span_invariant_holds_with_real_scripts(self):
        source = "cafe\u0301 東京👩\u200d👩\u200d👧 é. Hello مرحبا world."
        result = SemanticChunker(SemanticChunkerConfig(backend="morphological")).chunk(
            source
        )
        for chunk in result.chunks:
            start, end = chunk.context_span
            assert chunk.text == source[start:end]
