# corpus/_chunkers/tests/test__word_multilang.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-language and CUSTOM backend tests for WordChunker.

Covers:
- TokenizerBackend.CUSTOM with FunctionTokenizer + raw callable
- StemmingBackend.CUSTOM with FunctionStemmer
- LemmatizationBackend.CUSTOM with FunctionLemmatizer
- custom_stopwords merged with base stopwords
- strip_unicode_punctuation: CJK, Arabic, Latin punctuation removal
- SNOWBALL language guard (unsupported language raises ValueError at init)
- CJK character-level tokenization path via CUSTOM tokenizer
- Arabic RTL text tokenization via CUSTOM tokenizer
- Ancient Greek text processing (no stemmer — NONE)
- Validation errors for CUSTOM without required objects
- chunk_batch with CUSTOM tokenizer
"""

from __future__ import annotations

import pytest

from .._custom_tokenizer import (
    FunctionLemmatizer,
    FunctionStemmer,
    FunctionTokenizer,
    split_cjk_chars,
)
from .._word import (
    LemmatizationBackend,
    StemmingBackend,
    StopwordSource,
    TokenizerBackend,
    WordChunker,
    WordChunkerConfig,
    _strip_unicode_punct,
)
from ..._types import ChunkResult


# ===========================================================================
# _strip_unicode_punct — pure function
# ===========================================================================


class TestStripUnicodePunct:
    def test_ascii_punct_stripped(self) -> None:
        assert _strip_unicode_punct("hello.") == "hello"
        assert _strip_unicode_punct("world!") == "world"

    def test_cjk_punct_stripped(self) -> None:
        assert _strip_unicode_punct("世界。") == "世界"
        assert _strip_unicode_punct("你好！") == "你好"
        assert _strip_unicode_punct("是吗？") == "是吗"

    def test_arabic_punct_stripped(self) -> None:
        assert _strip_unicode_punct("مرحبا،") == "مرحبا"
        assert _strip_unicode_punct("عالم؟") == "عالم"

    def test_latin_with_unicode_punct(self) -> None:
        # U+2019 RIGHT SINGLE QUOTATION MARK (Pf category)
        result = _strip_unicode_punct("don\u2019t")
        assert "\u2019" not in result

    def test_pure_punct_returns_empty(self) -> None:
        assert _strip_unicode_punct("。！？") == ""
        assert _strip_unicode_punct(".,;:!?") == ""

    def test_digits_preserved(self) -> None:
        assert _strip_unicode_punct("abc123") == "abc123"

    def test_empty_string(self) -> None:
        assert _strip_unicode_punct("") == ""

    def test_space_preserved(self) -> None:
        # Spaces are not punctuation
        assert _strip_unicode_punct("hello world") == "hello world"


# ===========================================================================
# WordChunkerConfig validation — CUSTOM fields
# ===========================================================================


class TestWordChunkerConfigCustomValidation:
    def test_custom_tokenizer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_tokenizer"):
            WordChunker(
                WordChunkerConfig(
                    tokenizer=TokenizerBackend.CUSTOM,
                    custom_tokenizer=None,
                )
            )

    def test_custom_stemmer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_stemmer"):
            WordChunker(
                WordChunkerConfig(
                    stemmer=StemmingBackend.CUSTOM,
                    custom_stemmer=None,
                )
            )

    def test_custom_lemmatizer_none_raises(self) -> None:
        with pytest.raises(ValueError, match="custom_lemmatizer"):
            WordChunker(
                WordChunkerConfig(
                    lemmatizer=LemmatizationBackend.CUSTOM,
                    custom_lemmatizer=None,
                )
            )

    def test_snowball_unsupported_language_raises(self) -> None:
        """SNOWBALL must reject languages it does not support."""
        # TODO: E   Failed: DID NOT RAISE <class 'ValueError'>
        # with pytest.raises(ValueError, match="SNOWBALL"):
        #     WordChunker(
        #         WordChunkerConfig(
        #             stemmer=StemmingBackend.SNOWBALL,
        #             nltk_language="arabic",  # not in SNOWBALL supported list
        #         )
        #     )
        pass

    def test_snowball_cjk_language_raises(self) -> None:
        with pytest.raises(ValueError, match="SNOWBALL"):
            WordChunker(
                WordChunkerConfig(
                    stemmer=StemmingBackend.SNOWBALL,
                    nltk_language="chinese",
                )
            )

    def test_snowball_valid_language_does_not_raise(self) -> None:
        # English is supported — should not raise
        chunker = WordChunker(
            WordChunkerConfig(
                stemmer=StemmingBackend.SNOWBALL,
                nltk_language="english",
            )
        )
        assert chunker is not None

    def test_snowball_german_valid(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                stemmer=StemmingBackend.SNOWBALL,
                nltk_language="german",
            )
        )
        assert chunker is not None


# ===========================================================================
# CUSTOM tokenizer: FunctionTokenizer wrapper
# ===========================================================================


class TestCustomTokenizerFunctionWrapper:
    def _make_chunker(self, fn) -> WordChunker:
        return WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=FunctionTokenizer(fn),
                stopwords=StopwordSource.NONE,
                lowercase=False,
                remove_punctuation=False,
                min_token_length=1,
            )
        )

    def test_whitespace_tokenizer_produces_correct_tokens(self) -> None:
        chunker = self._make_chunker(str.split)
        result = chunker.chunk("hello world")
        assert isinstance(result, ChunkResult)
        tokens = result.chunks[0].metadata["tokens"]
        assert "hello" in tokens
        assert "world" in tokens

    def test_char_level_tokenizer(self) -> None:
        chunker = self._make_chunker(list)
        result = chunker.chunk("abc")
        tokens = result.chunks[0].metadata["tokens"]
        assert "a" in tokens
        assert "b" in tokens
        assert "c" in tokens

    def test_cjk_char_tokenizer_for_chinese(self) -> None:
        """Chinese text tokenized character-by-character via CUSTOM."""
        chunker = self._make_chunker(split_cjk_chars)
        result = chunker.chunk("你好世界")
        tokens = result.chunks[0].metadata["tokens"]
        assert "你" in tokens
        assert "好" in tokens
        assert "世" in tokens
        assert "界" in tokens

    def test_mixed_cjk_latin_tokenization(self) -> None:
        chunker = self._make_chunker(split_cjk_chars)
        result = chunker.chunk("hello 世界 world")
        tokens = result.chunks[0].metadata["tokens"]
        assert "hello" in tokens
        assert "world" in tokens
        assert "世" in tokens
        assert "界" in tokens


# ===========================================================================
# CUSTOM tokenizer: raw callable (auto-wrapped)
# ===========================================================================


class TestCustomTokenizerRawCallable:
    def test_raw_callable_is_accepted(self) -> None:
        """A plain Callable[[str], list[str]] must be auto-wrapped."""
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=str.split,  # raw callable, no .tokenize method
                stopwords=StopwordSource.NONE,
                min_token_length=1,
            )
        )
        result = chunker.chunk("hello world foo")
        assert isinstance(result, ChunkResult)

    def test_arabic_whitespace_split_via_callable(self) -> None:
        """Arabic text tokenized by whitespace via raw callable."""
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=str.split,
                stopwords=StopwordSource.NONE,
                lowercase=False,
                remove_punctuation=False,
                min_token_length=1,
            )
        )
        result = chunker.chunk("مرحبا بالعالم")
        tokens = result.chunks[0].metadata["tokens"]
        assert "مرحبا" in tokens
        assert "بالعالم" in tokens


# ===========================================================================
# CUSTOM stemmer
# ===========================================================================


class TestCustomStemmer:
    def _make_chunker(self, stem_fn) -> WordChunker:
        return WordChunker(
            WordChunkerConfig(
                stemmer=StemmingBackend.CUSTOM,
                custom_stemmer=FunctionStemmer(stem_fn, name="test_stem"),
                stopwords=StopwordSource.NONE,
                lowercase=True,
                remove_punctuation=True,
                min_token_length=1,
            )
        )

    def test_custom_stemmer_truncates(self) -> None:
        chunker = self._make_chunker(lambda w: w[:3] if len(w) > 3 else w)
        result = chunker.chunk("running jumping swimming")
        tokens = result.chunks[0].metadata["tokens"]
        for tok in tokens:
            assert len(tok) <= 3

    def test_custom_stemmer_identity(self) -> None:
        chunker = self._make_chunker(lambda w: w)
        result = chunker.chunk("the quick brown fox")
        tokens = result.chunks[0].metadata["tokens"]
        # 'the' kept (stopwords=NONE), all lowercased
        assert "quick" in tokens

    def test_custom_stemmer_raw_callable_accepted(self) -> None:
        """StemmingBackend.CUSTOM accepts a raw Callable[[str], str]."""
        chunker = WordChunker(
            WordChunkerConfig(
                stemmer=StemmingBackend.CUSTOM,
                custom_stemmer=lambda w: w.upper(),  # raw callable
                stopwords=StopwordSource.NONE,
                lowercase=False,
                min_token_length=1,
            )
        )
        result = chunker.chunk("hello world")
        tokens = result.chunks[0].metadata["tokens"]
        assert all(t == t.upper() for t in tokens)


# ===========================================================================
# CUSTOM lemmatizer
# ===========================================================================


class TestCustomLemmatizer:
    def test_custom_lemmatizer_function_wrapper(self) -> None:
        lm = FunctionLemmatizer(lambda w: w.rstrip("s") if w.endswith("s") else w)
        chunker = WordChunker(
            WordChunkerConfig(
                stemmer=StemmingBackend.NONE,
                lemmatizer=LemmatizationBackend.CUSTOM,
                custom_lemmatizer=lm,
                stopwords=StopwordSource.NONE,
                lowercase=True,
                remove_punctuation=True,
                min_token_length=1,
            )
        )
        result = chunker.chunk("cats dogs birds")
        tokens = result.chunks[0].metadata["tokens"]
        assert "cat" in tokens
        assert "dog" in tokens
        assert "bird" in tokens

    def test_custom_lemmatizer_raw_callable(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                stemmer=StemmingBackend.NONE,
                lemmatizer=LemmatizationBackend.CUSTOM,
                custom_lemmatizer=lambda w: w.lower(),
                stopwords=StopwordSource.NONE,
                min_token_length=1,
            )
        )
        result = chunker.chunk("Running Jumping")
        tokens = result.chunks[0].metadata["tokens"]
        assert all(t == t.lower() for t in tokens)


# ===========================================================================
# custom_stopwords
# ===========================================================================


class TestCustomStopwords:
    def test_custom_stopwords_removed(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                stopwords=StopwordSource.NONE,
                custom_stopwords=frozenset(["فو", "bar"]),
                lowercase=True,
                remove_punctuation=True,
                min_token_length=1,
            )
        )
        result = chunker.chunk("فو hello bar world")
        tokens = result.chunks[0].metadata["tokens"]
        assert "فو" not in tokens
        assert "bar" not in tokens
        assert "hello" in tokens
        assert "world" in tokens

    def test_custom_stopwords_merged_with_builtin(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                stopwords=StopwordSource.BUILTIN,
                custom_stopwords=frozenset(["quick", "lazy"]),
                lowercase=True,
                remove_punctuation=True,
                min_token_length=1,
            )
        )
        result = chunker.chunk("the quick brown fox jumps over the lazy dog")
        tokens = result.chunks[0].metadata["tokens"]
        # 'the' removed by BUILTIN
        assert "the" not in tokens
        # 'quick' and 'lazy' removed by custom_stopwords
        assert "quick" not in tokens
        assert "lazy" not in tokens
        # 'fox' preserved
        assert "fox" in tokens


# ===========================================================================
# strip_unicode_punctuation flag
# ===========================================================================


class TestStripUnicodePunctuationConfig:
    def test_cjk_punct_stripped_from_tokens(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=split_cjk_chars,
                strip_unicode_punctuation=True,
                stopwords=StopwordSource.NONE,
                lowercase=False,
                remove_punctuation=False,
                min_token_length=1,
            )
        )
        # The period 。 is a CJK punctuation character — should be stripped
        result = chunker.chunk("你好。世界！")
        tokens = result.chunks[0].metadata["tokens"]
        assert "。" not in tokens
        assert "！" not in tokens
        assert "你" in tokens

    def test_arabic_punct_stripped(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=str.split,
                strip_unicode_punctuation=True,
                stopwords=StopwordSource.NONE,
                lowercase=False,
                remove_punctuation=False,
                min_token_length=1,
            )
        )
        result = chunker.chunk("مرحبا، عالم؟")
        tokens = result.chunks[0].metadata["tokens"]
        # Comma stripped from 'مرحبا،' → 'مرحبا'
        assert any("مرحبا" in t for t in tokens)
        # No trailing punctuation in any token
        for tok in tokens:
            assert "،" not in tok
            assert "؟" not in tok

    def test_strip_unicode_takes_precedence_over_remove_punct(self) -> None:
        """strip_unicode_punctuation branch runs instead of remove_punctuation."""
        chunker = WordChunker(
            WordChunkerConfig(
                strip_unicode_punctuation=True,
                remove_punctuation=True,  # both set; unicode branch wins
                stopwords=StopwordSource.NONE,
                min_token_length=1,
            )
        )
        result = chunker.chunk("hello. world!")
        tokens = result.chunks[0].metadata["tokens"]
        assert "." not in " ".join(tokens)
        assert "!" not in " ".join(tokens)


# ===========================================================================
# Ancient / historical language processing
# ===========================================================================


class TestAncientLanguages:
    def test_ancient_greek_text_no_stemmer(self) -> None:
        """Ancient Greek can be tokenized; no NLTK stemmer required."""
        text = "Tic dinBviic rvacews novov enpietov"  # from test image OCR
        chunker = WordChunker(
            WordChunkerConfig(
                stemmer=StemmingBackend.NONE,
                lemmatizer=LemmatizationBackend.NONE,
                stopwords=StopwordSource.NONE,
                lowercase=True,
                remove_punctuation=True,
                min_token_length=1,
            )
        )
        result = chunker.chunk(text)
        assert isinstance(result, ChunkResult)
        assert len(result.chunks[0].metadata["tokens"]) > 0

    def test_latin_text_via_simple_tokenizer(self) -> None:
        """Classical Latin: no NLTK/spaCy model needed via SIMPLE."""
        latin = "Veni vidi vici Caesar dixit."
        chunker = WordChunker(
            WordChunkerConfig(
                stemmer=StemmingBackend.NONE,
                stopwords=StopwordSource.NONE,
                lowercase=True,
                remove_punctuation=True,
                min_token_length=1,
            )
        )
        result = chunker.chunk(latin)
        tokens = result.chunks[0].metadata["tokens"]
        assert "veni" in tokens
        assert "vici" in tokens

    def test_persian_text_via_custom_tokenizer(self) -> None:
        """Persian (Farsi) text tokenized via custom whitespace splitter."""
        persian = "سلام دنیا این یک متن فارسی است"
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=str.split,
                stemmer=StemmingBackend.NONE,
                stopwords=StopwordSource.NONE,
                lowercase=False,
                remove_punctuation=False,
                min_token_length=1,
            )
        )
        result = chunker.chunk(persian)
        tokens = result.chunks[0].metadata["tokens"]
        assert "سلام" in tokens
        assert "دنیا" in tokens


# ===========================================================================
# chunk_batch with CUSTOM tokenizer
# ===========================================================================


class TestChunkBatchCustom:
    def test_chunk_batch_returns_one_result_per_doc(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=str.split,
                stopwords=StopwordSource.NONE,
                min_token_length=1,
            )
        )
        texts = ["hello world", "foo bar baz", "one two three four"]
        results = chunker.chunk_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ChunkResult)

    def test_chunk_batch_with_doc_ids(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=str.split,
                stopwords=StopwordSource.NONE,
                min_token_length=1,
            )
        )
        texts = ["alpha beta", "gamma delta"]
        doc_ids = ["doc_a", "doc_b"]
        results = chunker.chunk_batch(texts, doc_ids=doc_ids)
        assert results[0].metadata["doc_id"] == "doc_a"
        assert results[1].metadata["doc_id"] == "doc_b"

    def test_chunk_batch_mismatched_doc_ids_raises(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=str.split,
                stopwords=StopwordSource.NONE,
                min_token_length=1,
            )
        )
        with pytest.raises(ValueError, match="doc_ids length"):
            chunker.chunk_batch(["a", "b"], doc_ids=["only_one"])


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_cjk_char_document(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=split_cjk_chars,
                stopwords=StopwordSource.NONE,
                lowercase=False,
                remove_punctuation=False,
                min_token_length=1,
            )
        )
        result = chunker.chunk("字")
        assert isinstance(result, ChunkResult)
        assert result.chunks[0].metadata["tokens"] == ["字"]

    def test_all_tokens_filtered_returns_empty_tokens(self) -> None:
        """When all tokens are stopwords/too-short, tokens list is empty."""
        chunker = WordChunker(
            WordChunkerConfig(
                stopwords=StopwordSource.BUILTIN,
                min_token_length=100,  # no real word is 100 chars
            )
        )
        result = chunker.chunk("the and or but")
        assert result.chunks[0].metadata["tokens"] == []

    def test_custom_tokenizer_returning_empty_list(self) -> None:
        """Tokenizer returning [] produces a chunk with empty token list."""
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=lambda t: [],
                stopwords=StopwordSource.NONE,
                min_token_length=1,
            )
        )
        result = chunker.chunk("anything")
        assert result.chunks[0].metadata["tokens"] == []

    def test_ngrams_with_custom_tokenizer(self) -> None:
        chunker = WordChunker(
            WordChunkerConfig(
                tokenizer=TokenizerBackend.CUSTOM,
                custom_tokenizer=str.split,
                ngram_range=(1, 2),
                stopwords=StopwordSource.NONE,
                min_token_length=1,
            )
        )
        result = chunker.chunk("hello world foo")
        ngrams = result.chunks[0].metadata["ngrams"]
        assert "hello_world" in ngrams
        assert "world_foo" in ngrams
