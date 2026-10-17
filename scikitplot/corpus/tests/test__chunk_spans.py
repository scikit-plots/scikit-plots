# scikitplot/corpus/tests/test__chunk_spans.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The span model (Run 3, R3-3): a chunk's text is a slice of its source.

Every built-in chunker whose ``text_is_source_slice`` is ``True`` satisfies
``chunk.text == source[context_start:context_end]``, and the context covers
the anchor (``start_char``/``end_char``, the region the chunk stands for).
``WordChunker`` derives its text (tokens, case, stems); its context span is
the source region the text came from. ``DocumentReader`` maps a document to
the context span and records the anchor when it differs.
"""

from __future__ import annotations

import logging

import pytest

from .._base import _slice_raw_text
from .._chunkers._chunker_bridge import ChunkerBridge
from .._chunkers._multilang_mixin import MultilangConfig
from .._chunkers._semantic import SemanticChunker, SemanticChunkerConfig
from .._chunkers._sentence import SentenceChunker, SentenceChunkerConfig
from .._chunkers._spans import cover, locate, strip_span
from .._chunkers._word import WordChunker, WordChunkerConfig
from .._corpus_builder import BuilderConfig, CorpusBuilder
from .._registry._registry import ComponentRegistry
from .._types import Chunk

TEXTS = {
    "latin": "Alpha one. Beta two.\n\nGamma three, delta four. Epsilon five!",
    "spaces": "  Leading space.   Many   spaces here.\n\n\n  Trailing.  ",
    "cjk": "今日は良い天気です。明日も晴れるでしょう。\n\n空が青い。",
    "mixed": "Hello 世界. Привет мир! مرحبا بالعالم.\r\n\r\nNext paragraph here.",
    "long": " ".join(f"Sentence number {i} is here." for i in range(30)),
}
CONFIGS = [
    ("sentence", {}),
    ("sentence", {"overlap": 2}),
    ("sentence", {"overlap": 1, "strip_whitespace": False}),
    ("paragraph", {}),
    ("paragraph", {"overlap": 1}),
    ("paragraph", {"merge_short": True, "min_length": 15}),
    ("paragraph", {"max_length": 60, "min_length": 1}),
    ("fixed_window", {"window_size": 12, "step_size": 8}),
    ("fixed_window", {"window_size": 12, "step_size": 8, "strip_whitespace": False}),
    ("fixed_window", {"window_size": 3, "step_size": 2, "unit": "tokens"}),
    ("word", {"include_offsets": True}),
    ("word", {"chunk_by": "sentence", "include_offsets": True}),
    ("recursive", {"chunk_size": 30}),
    ("recursive", {"chunk_size": 30, "chunk_overlap": 12}),
    ("recursive", {"chunk_size": 4, "strip_whitespace": False}),
    ("parent_child", {"parent_size": 60, "child_size": 20}),
    ("parent_child", {"parent_size": 60, "child_size": 20, "child_overlap": 8}),
    ("structure", {"chunk_size": 30}),
    ("structure", {"markup": "rst", "chunk_size": 20, "include_heading": False}),
    ("semantic", {"backend": "morphological"}),
    ("semantic", {"backend": "morphological", "max_chunk_tokens": 3}),
    ("semantic", {"backend": "morphological", "min_chunk_tokens": 4}),
]


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def registry():
    registry = ComponentRegistry()
    registry.register_builtins()
    return registry


def _check(chunker, source):
    result = chunker.chunk(source)
    assert result.chunks
    for chunk in result.chunks:
        start, end = chunk.context_span
        assert start <= chunk.start_char <= chunk.end_char <= end
        if type(chunker).text_is_source_slice:
            assert chunk.text == source[start:end], (chunk.text, (start, end))
        else:
            assert source[start:end].strip()  # the region the text came from
    return result.chunks


@pytest.mark.parametrize("text_name", sorted(TEXTS))
@pytest.mark.parametrize(
    ("name", "options"), CONFIGS, ids=lambda v: v if isinstance(v, str) else repr(v)
)
def test_the_span_invariant(registry, name, options, text_name):
    """The plan's gate: every chunker, every script, overlap and merges."""
    _check(registry.build_chunker(name, **options), TEXTS[text_name])


class TestOverlap:
    def test_context_is_the_source_not_a_join(self):
        """The review's probe; CJK used to gain an inserted space."""
        source = TEXTS["cjk"]
        chunker = SentenceChunker(SentenceChunkerConfig(overlap=1, min_length=1))
        chunks = _check(chunker, source)
        second = chunks[1]
        assert second.text == "今日は良い天気です。明日も晴れるでしょう。"
        assert source[second.start_char : second.end_char] == "明日も晴れるでしょう。"
        assert second.context_span == (0, second.end_char)

    def test_a_chunk_without_context_has_one_span(self):
        chunk = _check(SentenceChunker(), TEXTS["latin"])[0]
        assert chunk.context_start is None
        assert chunk.anchor_span == chunk.context_span


class TestSemanticMerges:
    def test_a_model_merge_is_the_source_slice(self, monkeypatch):
        """The HYBRID merge joined parts with " "."""
        source = "東京は大きい。大阪も大きい。\n京都は古い。"
        chunker = SemanticChunker(
            SemanticChunkerConfig(similarity_threshold=0.0, adapter_unit="sentence")
        )
        monkeypatch.setattr(chunker, "_load_embed_model", lambda: True)
        monkeypatch.setattr(chunker, "_embed_texts", lambda texts: [[1.0]] * len(texts))
        chunks = _check(chunker, source)
        assert len(chunks) == 1
        assert chunks[0].text == source.strip()

    def test_an_oversized_split_gives_each_piece_its_own_span(self):
        """Pieces were " " joins, each carrying the whole chunk's span."""
        source = "Intro.  one two\nthree four five six seven"
        chunker = SemanticChunker(
            SemanticChunkerConfig(backend="morphological", max_chunk_tokens=3)
        )
        whole = Chunk(text=source[8:], start_char=8, end_char=len(source))
        pieces = chunker._split_oversized(whole, source)
        assert [p.text for p in pieces] == ["one two\nthree", "four five six", "seven"]
        assert all(p.text == source[p.start_char : p.end_char] for p in pieces)

    def test_an_undersized_merge_is_the_source_slice(self):
        source = "a  b\n c"
        parts = [
            Chunk(text="a", start_char=0, end_char=1),
            Chunk(text="c", start_char=6, end_char=7),
        ]
        merged = SemanticChunker._merge_buf(source, parts)
        assert (merged.text, merged.anchor_span) == ("a  b\n c", (0, 7))


class TestChunk:
    def test_the_context_must_cover_the_anchor(self):
        with pytest.raises(ValueError, match="must cover the anchor"):
            Chunk(text="x", start_char=2, end_char=4, context_start=3)
        with pytest.raises(ValueError, match="must cover the anchor"):
            Chunk(text="x", start_char=2, end_char=4, context_end=3)

    def test_a_disabled_offset_is_zero_zero(self):
        assert Chunk(text="x", start_char=0, end_char=0).context_span == (0, 0)


class TestHelpers:
    def test_locate_refuses_a_piece_that_is_not_in_the_source(self):
        assert locate("ab cd", ["ab", "cd"], where="t") == [(0, 2), (3, 5)]
        with pytest.raises(ValueError, match="not a substring"):
            locate("ab cd", ["cd", "ab"], where="t")

    def test_strip_span_and_cover(self):
        assert strip_span("  ab  ", 0, 6) == (2, 4)
        assert strip_span("    ", 0, 4) == (0, 0)
        assert cover([(1, 2), (5, 9)]) == (1, 9)


class TestDocuments:
    """``DocumentReader`` maps a document to the chunk's context span."""

    def _documents(self, tmp_path, **chunker_kwargs):
        source = tmp_path / "doc.txt"
        source.write_text(TEXTS["latin"], encoding="utf-8")
        builder = CorpusBuilder(
            BuilderConfig(
                chunker="sentence",
                chunker_kwargs=chunker_kwargs,
                normalize=False,
                filter_kwargs={"min_words": 0, "min_chars": 0},
            )
        )
        return builder.build(source).documents

    def test_an_overlap_document_spans_its_text(self, tmp_path):
        """It used to span anchor start + len(text): the wrong region."""
        documents = self._documents(tmp_path, overlap=1, min_length=1)
        overlapping = [d for d in documents if "anchor_span" in d.metadata]
        assert overlapping
        for doc in documents:
            assert doc.raw_text == doc.text
            assert doc.char_end - doc.char_start == len(doc.text)
        for doc in overlapping:
            start, end = doc.metadata["anchor_span"]
            assert doc.char_start <= start <= end == doc.char_end

    def test_without_offsets_the_span_is_unknown(self, tmp_path):
        """Review F1: raw_text was the whole parent for every document."""
        documents = self._documents(tmp_path, include_offsets=False)
        assert len(documents) > 1
        assert all(d.char_start is None for d in documents)
        assert all(d.char_end is None for d in documents)
        assert all(d.raw_text == d.text for d in documents)


class TestRawText:
    """``_slice_raw_text``: ``None`` when the chunk's region is unknown."""

    PARENT = "Alpha one. Beta two."

    def _call(self, parent, span, **kwargs):
        options = {
            "chunked_text": self.PARENT,
            "spans_index_parent": True,
            "only_chunk": False,
            "chunk_text": "Beta two.",
            "text_is_source_slice": True,
        }
        meta = kwargs.pop("meta", {})
        options.update(kwargs)
        return _slice_raw_text(parent, meta, span, **options)

    def test_spans_that_index_the_parent(self):
        parent = self.PARENT
        assert self._call(parent, (11, 20)) == "Beta two."
        assert self._call(parent, (11, 20), only_chunk=True) == "Beta two."
        assert self._call(parent, (11, 99)) is None
        assert self._call(parent, None) == "Beta two."
        assert self._call(parent, None, text_is_source_slice=False) is None
        segment = {"_raw_segment": "Beta two."}
        assert (
            self._call(parent, None, text_is_source_slice=False, meta=segment)
            == "Beta two."
        )
        assert self._call(None, (11, 20)) is None

    def test_spans_that_do_not_index_the_parent(self):
        markup = "<p>Alpha one. Beta two.</p>"
        foreign = {"spans_index_parent": False}
        assert self._call(markup, (11, 20), **foreign) is None
        assert self._call(markup, (11, 20), only_chunk=True, **foreign) is None
        whole = {"chunk_text": self.PARENT, "only_chunk": True, **foreign}
        assert self._call(markup, (0, 20), **whole) == markup
        word = {**whole, "text_is_source_slice": False, "chunk_text": "alpha beta"}
        segment = {"_raw_segment": self.PARENT}
        assert self._call(markup, None, meta=segment, **word) == markup
        assert self._call(markup, None, **word) is None

    def test_one_chunk_left_by_min_length_is_not_the_whole_parent(self, tmp_path):
        """Re-review 1: the only chunk got the whole parent."""
        source = tmp_path / "doc.txt"
        source.write_text("Hi. This is a much longer sentence here.", "utf-8")
        builder = CorpusBuilder(
            BuilderConfig(
                chunker="sentence",
                chunker_kwargs={"min_length": 10},
                normalize=False,
                filter_kwargs={"min_words": 0, "min_chars": 0},
            )
        )
        (doc,) = builder.build(source).documents
        assert doc.raw_text == doc.text == "This is a much longer sentence here."


class TestPreprocessedSpans:
    """Review F2: preprocessing makes spans index the preprocessed text."""

    SOURCE = "\ufeffAlpha one.\x07 Beta two."

    def _chunker(self):
        return SentenceChunker(
            SentenceChunkerConfig(
                min_length=1,
                multilang_config=MultilangConfig(include_raw_text=True),
            )
        )

    def test_the_basis_is_reported_and_no_raw_slice_is_taken(self):
        result = self._chunker().chunk(self.SOURCE)
        assert result.metadata["span_basis"] == "preprocessed"
        for chunk in result.chunks:
            assert "raw_text" not in chunk.metadata.get("multilang", {})

    def test_an_unchanged_text_keeps_the_input_basis(self):
        result = self._chunker().chunk("Alpha one. Beta two.")
        assert result.metadata["span_basis"] == "input"
        raws = [c.metadata["multilang"]["raw_text"] for c in result.chunks]
        assert raws == ["Alpha one.", "Beta two."]

    def test_an_overlap_raw_text_covers_the_context(self):
        """Re-review 3: it covered the anchor while the document spans the context."""
        source = "Alpha one. Beta two."
        chunker = SentenceChunker(
            SentenceChunkerConfig(
                overlap=1,
                min_length=1,
                multilang_config=MultilangConfig(include_raw_text=True),
            )
        )
        second = chunker.chunk(source).chunks[1]
        assert second.metadata["multilang"]["raw_text"] == second.text == source

    def test_word_records_no_raw_text_for_a_changed_text(self):
        """Re-review 2: WordChunker gave the NFC text as raw."""
        chunker = WordChunker(
            WordChunkerConfig(multilang_config=MultilangConfig(include_raw_text=True))
        )
        result = chunker.chunk(self.SOURCE)
        assert result.metadata["span_basis"] == "preprocessed"
        for chunk in result.chunks:
            assert "raw_text" not in chunk.metadata.get("multilang", {})

    def test_a_document_records_the_basis(self, tmp_path):
        source = tmp_path / "doc.txt"
        source.write_text(self.SOURCE, encoding="utf-8")
        builder = CorpusBuilder(
            BuilderConfig(
                chunker=self._chunker(),
                normalize=False,
                filter_kwargs={"min_words": 0, "min_chars": 0},
            )
        )
        documents = builder.build(source).documents
        assert len(documents) == 2
        for doc in documents:
            assert doc.metadata["span_basis"] == "preprocessed"
            assert doc.raw_text is None


class TestWord:
    def test_each_sentence_chunk_spans_its_sentence(self):
        """Review F4: every chunk used to report (0, len(sentence))."""
        source = TEXTS["latin"]
        sentences = SentenceChunker(SentenceChunkerConfig(min_length=1)).chunk(source)
        chunker = WordChunker(
            WordChunkerConfig(chunk_by="sentence", include_offsets=True)
        )
        spans = [c.context_span for c in chunker.chunk(source).chunks]
        assert spans == [c.anchor_span for c in sentences.chunks]
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))

    def test_offsets_off_reports_no_span(self):
        """Review F7: the flag was ignored."""
        chunks = (
            WordChunker(WordChunkerConfig(chunk_by="sentence"))
            .chunk(TEXTS["latin"])
            .chunks
        )
        assert {c.context_span for c in chunks} == {(0, 0)}


def test_legacy_pairs_start_where_the_text_starts():
    """Review F8: an overlap pair carried the anchor start."""
    source = TEXTS["latin"]
    result = SentenceChunker(SentenceChunkerConfig(overlap=1, min_length=1)).chunk(
        source
    )
    pairs = ChunkerBridge._to_tuples(source, result)
    assert all(source[start : start + len(text)] == text for start, text in pairs)
