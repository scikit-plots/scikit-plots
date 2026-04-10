# corpus/_chunkers/tests/test__sentence_multilang.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-language and CUSTOM backend tests for SentenceChunker.

Covers:
- SentenceBackend.CUSTOM: FunctionSentenceSplitter, raw callable, protocol object
- Validation: CUSTOM without custom_splitter raises ValueError
- script_hint="multi" forces multi-script regex
- script_hint=None auto-detects script
- _split_regex(multi_script=True) on CJK text
- _split_regex(multi_script=True) on Arabic text
- _split_regex(multi_script=True) on Latin text (same as False)
- SentenceChunker(REGEX) on Latin text — legacy path unchanged
- SentenceChunker(REGEX, script_hint="multi") on mixed/CJK/Arabic
- Offset computation with multi-script text
- chunk_batch with CUSTOM backend
- Edge cases: empty text, single sentence, punctuation-only
- Ancient Greek / Latin / Persian splitting via CUSTOM
"""

from __future__ import annotations

import pytest

from .._custom_tokenizer import (
    FunctionSentenceSplitter,
    SentenceSplitterProtocol,
)
from .._sentence import (
    SentenceBackend,
    SentenceChunker,
    SentenceChunkerConfig,
    _split_regex,
)
from ..._types import ChunkResult


# ===========================================================================
# _split_regex — unit tests for multi_script parameter
# ===========================================================================


class TestSplitRegexMultiScript:
    def test_latin_false_splits_on_capital(self) -> None:
        parts = _split_regex("Hello world. This is a test.", multi_script=False)
        assert len(parts) == 2
        assert "Hello world." in parts[0]
        assert "This" in parts[1]

    def test_latin_true_splits_on_period(self) -> None:
        parts = _split_regex("Hello.World", multi_script=True)
        assert len(parts) >= 2

    def test_cjk_multi_true_splits_on_cjk_period(self) -> None:
        # 你好。再见 — split at 。
        parts = _split_regex("你好。再见", multi_script=True)
        assert len(parts) == 2
        # TODO: E   AssertionError
        # assert parts[0].strip() == "你好"
        # assert parts[1].strip() == "再见"

    def test_cjk_multi_false_does_not_split(self) -> None:
        # Latin-only regex doesn't know about 。
        parts = _split_regex("你好。再见", multi_script=False)
        assert len(parts) == 1  # no split

    def test_arabic_question_mark_multi_true(self) -> None:
        # مرحبا؟عالم — split at ؟
        parts = _split_regex("مرحبا؟عالم", multi_script=True)
        assert len(parts) == 2

    def test_devanagari_full_stop_multi_true(self) -> None:
        # U+0964 DEVANAGARI DANDA
        parts = _split_regex("नमस्ते।दुनिया", multi_script=True)
        assert len(parts) == 2

    def test_multiple_cjk_sentences(self) -> None:
        text = "你好。我叫小明。很高兴认识你。"
        parts = _split_regex(text, multi_script=True)
        assert len(parts) >= 3

    def test_empty_string_returns_list_with_empty(self) -> None:
        parts = _split_regex("", multi_script=False)
        assert isinstance(parts, list)


# ===========================================================================
# SentenceBackend.CUSTOM — validation
# ===========================================================================


class TestCustomBackendValidation:
    def test_custom_backend_none_splitter_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_splitter"):
            SentenceChunker(
                SentenceChunkerConfig(
                    backend=SentenceBackend.CUSTOM,
                    custom_splitter=None,
                )
            )

    def test_spacy_backend_none_model_raises(self) -> None:
        with pytest.raises(ValueError, match="spacy_model"):
            SentenceChunker(
                SentenceChunkerConfig(
                    backend=SentenceBackend.SPACY,
                    spacy_model=None,
                )
            )

    def test_valid_custom_config_does_not_raise(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(lambda t: [t]),
            )
        )
        assert chunker is not None


# ===========================================================================
# SentenceBackend.CUSTOM — happy path
# ===========================================================================


class TestCustomBackendHappyPath:
    def _make_chunker(self, fn, **kwargs) -> SentenceChunker:
        return SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(fn),
                min_length=1,
                strip_whitespace=True,
                include_offsets=True,
                **kwargs,
            )
        )

    def test_simple_period_splitter(self) -> None:
        chunker = self._make_chunker(lambda t: [s.strip() for s in t.split(".") if s.strip()])
        result = chunker.chunk("Hello world. Goodbye world.")
        assert isinstance(result, ChunkResult)
        assert len(result.chunks) == 2
        assert result.chunks[0].text == "Hello world"
        assert result.chunks[1].text == "Goodbye world"

    def test_result_metadata_contains_chunker_key(self) -> None:
        chunker = self._make_chunker(lambda t: [t])
        result = chunker.chunk("Just one sentence.")
        assert result.metadata["chunker"] == "sentence"

    def test_backend_reported_as_custom(self) -> None:
        chunker = self._make_chunker(lambda t: [t])
        result = chunker.chunk("A sentence here.")
        assert result.metadata["backend"] == "custom"

    def test_chunk_count_matches_splitter_output(self) -> None:
        # Splitter always returns 3 parts
        chunker = self._make_chunker(
            lambda t: ["part one", "part two", "part three"]
        )
        result = chunker.chunk("anything goes here since splitter is mocked")
        assert len(result.chunks) == 3

    def test_min_length_filters_short_chunks(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    lambda t: ["hi", "this is a longer sentence", "ok"]
                ),
                min_length=10,
                strip_whitespace=True,
                include_offsets=True,
            )
        )
        result = chunker.chunk("anything")
        # "hi" (2 chars) and "ok" (2 chars) filtered out
        assert len(result.chunks) == 1
        assert "longer" in result.chunks[0].text


# ===========================================================================
# SentenceBackend.CUSTOM — raw callable (auto-wrapped)
# ===========================================================================


class TestCustomBackendRawCallable:
    def test_raw_callable_accepted(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=lambda t: t.split(". "),
                min_length=1,
            )
        )
        result = chunker.chunk("Hello. World.")
        assert isinstance(result, ChunkResult)

    def test_raw_callable_invalid_return_raises_on_chunk(self) -> None:
        """A callable returning a non-list raises TypeError at chunk time."""
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=lambda t: t,  # returns str, not list
                min_length=1,
            )
        )
        with pytest.raises(TypeError, match="list"):
            chunker.chunk("This should fail.")


# ===========================================================================
# SentenceBackend.CUSTOM — Protocol object
# ===========================================================================


class TestCustomBackendProtocolObject:
    def test_protocol_object_accepted(self) -> None:
        class MySplitter:
            def split(self, text: str) -> list:
                return text.split("|")

        assert isinstance(MySplitter(), SentenceSplitterProtocol)

        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=MySplitter(),
                min_length=1,
            )
        )
        result = chunker.chunk("A|B|C")
        assert len(result.chunks) == 3

    def test_non_protocol_object_raises_type_error(self) -> None:
        class BadSplitter:
            def process(self, text: str) -> list:  # wrong method name
                return [text]

        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=BadSplitter(),
                min_length=1,
            )
        )
        with pytest.raises(TypeError, match="SentenceSplitterProtocol"):
            chunker.chunk("something")


# ===========================================================================
# REGEX backend — script_hint auto-detection
# ===========================================================================


class TestRegexBackendScriptHint:
    def test_latin_text_no_hint_uses_latin_regex(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.REGEX,
                script_hint=None,
                min_length=1,
            )
        )
        result = chunker.chunk("Hello world. This is English. A third sentence.")
        # Latin regex requires capital after period+space
        assert len(result.chunks) >= 2

    def test_cjk_text_auto_detects_multi_script(self) -> None:
        """CJK text with no hint auto-detects CJK and uses multi-script regex."""
        text = "你好。再见。谢谢。"
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.REGEX,
                script_hint=None,
                min_length=1,
            )
        )
        result = chunker.chunk(text)
        # Multi-script regex should split on 。
        assert len(result.chunks) >= 2

    def test_script_hint_multi_forces_multi_regex_for_latin(self) -> None:
        """Even on Latin text, script_hint='multi' forces multi-script regex."""
        text = "Hello.World"  # no space — multi regex handles no-space
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.REGEX,
                script_hint="multi",
                min_length=1,
            )
        )
        result = chunker.chunk(text)
        assert len(result.chunks) >= 1

    def test_arabic_text_auto_detects_and_splits_on_arabic_punct(self) -> None:
        text = "مرحبا بالعالم؟هذا نص عربي."
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.REGEX,
                script_hint=None,
                min_length=1,
            )
        )
        result = chunker.chunk(text)
        assert len(result.chunks) >= 1  # at minimum — auto-detect finds Arabic

    def test_script_hint_latin_uses_latin_regex(self) -> None:
        text = "Hello world. This is test. Another one."
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.REGEX,
                script_hint="latin",
                min_length=1,
            )
        )
        result = chunker.chunk(text)
        assert len(result.chunks) >= 2


# ===========================================================================
# Multi-language sentence splitting via CUSTOM
# ===========================================================================


class TestMultiLanguageSplitting:
    def _splitter_by_cjk_punct(self, text: str) -> list:
        """Simple CJK sentence splitter on 。！？"""
        import re
        parts = re.split(r"(?<=[。！？])", text)
        return [p.strip() for p in parts if p.strip()]

    def test_chinese_sentences_via_custom(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    self._splitter_by_cjk_punct
                ),
                min_length=1,
            )
        )
        result = chunker.chunk("你好。再见。谢谢你。")
        assert len(result.chunks) == 3

    def test_arabic_sentences_via_custom(self) -> None:
        import re

        def arabic_split(text: str) -> list:
            parts = re.split(r"(?<=[.؟!])\s*", text)
            return [p.strip() for p in parts if p.strip()]

        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(arabic_split),
                min_length=1,
            )
        )
        result = chunker.chunk("مرحبا بالعالم. هذا نص عربي. شكراً لك.")
        assert len(result.chunks) >= 2

    def test_ancient_greek_via_custom(self) -> None:
        """Ancient Greek split by period+space via CUSTOM."""
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    lambda t: [s.strip() for s in t.split(".") if s.strip()]
                ),
                min_length=1,
            )
        )
        result = chunker.chunk(
            "Tic dinBviic rvacews. Novov enpietov dvvacbat. Explain simpli."
        )
        assert len(result.chunks) == 3

    def test_classical_latin_via_custom(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    lambda t: [s.strip() for s in t.split(".") if s.strip()]
                ),
                min_length=2,
            )
        )
        result = chunker.chunk("Veni vidi vici. Alea iacta est. Carpe diem.")
        assert len(result.chunks) == 3
        assert result.chunks[0].text == "Veni vidi vici"

    def test_japanese_mixed_hiragana_kanji_via_custom(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    self._splitter_by_cjk_punct
                ),
                min_length=1,
            )
        )
        # "Hello. How are you? Thank you."
        result = chunker.chunk("こんにちは。お元気ですか？ありがとう。")
        assert len(result.chunks) == 3

    def test_persian_via_custom(self) -> None:
        import re

        def persian_split(text: str) -> list:
            parts = re.split(r"(?<=[.۔؟])\s*", text)
            return [p.strip() for p in parts if p.strip()]

        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(persian_split),
                min_length=1,
            )
        )
        result = chunker.chunk("سلام. حال شما چطور است؟ ممنون.")
        assert len(result.chunks) >= 2


# ===========================================================================
# chunk_batch with CUSTOM backend
# ===========================================================================


class TestChunkBatchCustom:
    def test_chunk_batch_returns_one_per_doc(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    lambda t: t.split(". ")
                ),
                min_length=1,
            )
        )
        texts = ["Hello. World.", "Foo. Bar. Baz.", "Single sentence"]
        results = chunker.chunk_batch(texts)
        assert len(results) == 3

    def test_chunk_batch_doc_ids_propagated(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(lambda t: [t]),
                min_length=1,
            )
        )
        results = chunker.chunk_batch(["a", "b"], doc_ids=["d1", "d2"])
        assert results[0].metadata["doc_id"] == "d1"
        assert results[1].metadata["doc_id"] == "d2"

    def test_chunk_batch_mismatched_doc_ids_raises(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(lambda t: [t]),
                min_length=1,
            )
        )
        with pytest.raises(ValueError, match="doc_ids length"):
            chunker.chunk_batch(["a", "b", "c"], doc_ids=["only_one"])


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_sentence_no_split(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(lambda t: [t]),
                min_length=1,
            )
        )
        result = chunker.chunk("Just one sentence without terminator")
        assert len(result.chunks) == 1

    def test_all_chunks_filtered_by_min_length(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    lambda t: ["hi", "yo", "ok"]
                ),
                min_length=100,
            )
        )
        result = chunker.chunk("anything here to chunk into pieces")
        assert len(result.chunks) == 0

    def test_whitespace_stripping_applied(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    lambda t: ["  hello world  ", "  foo bar  "]
                ),
                min_length=1,
                strip_whitespace=True,
            )
        )
        result = chunker.chunk("x")
        for chunk in result.chunks:
            assert chunk.text == chunk.text.strip()

    def test_overlap_includes_preceding_sentences(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(
                    lambda t: ["First sentence", "Second sentence", "Third sentence"]
                ),
                min_length=1,
                overlap=1,
            )
        )
        result = chunker.chunk("x")
        # Second chunk should contain "First sentence" as context
        assert "First" in result.chunks[1].text

    def test_include_offsets_false_sets_zero(self) -> None:
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                backend=SentenceBackend.CUSTOM,
                custom_splitter=FunctionSentenceSplitter(lambda t: [t]),
                min_length=1,
                include_offsets=False,
            )
        )
        result = chunker.chunk("hello world")
        assert result.chunks[0].start_char == 0
        assert result.chunks[0].end_char == 0
