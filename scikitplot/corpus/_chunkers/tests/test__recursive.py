# scikitplot/corpus/_chunkers/tests/test__recursive.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Recursive and parent-child chunking (Run 4, R4-2).

``RecursiveChunker`` cuts by a separator hierarchy until every piece fits;
``ParentChildChunker`` produces linked parents and children, and documents
built from them carry ``parent_doc_id`` (no producer set it before).
"""

from __future__ import annotations

import random

import pytest

from ... import _hierarchy
from ..._corpus_builder import BuilderConfig, CorpusBuilder
from .._recursive import (
    ParentChildChunker,
    ParentChildChunkerConfig,
    RecursiveChunker,
    RecursiveChunkerConfig,
    recursive_spans,
)

TEXT = (
    "Alpha one is here. Beta two is there.\n\n"
    "Gamma three goes on. Delta four ends it here now.\n"
    "A short line.\n\n" + "longwordwithoutanyspaces" * 3
)


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"chunk_size": 0}, "chunk_size must be a positive int"),
            ({"chunk_size": True}, "chunk_size must be a positive int"),
            ({"chunk_overlap": 10, "chunk_size": 10}, r"chunk_overlap must be"),
            ({"chunk_overlap": -1}, r"chunk_overlap must be"),
            ({"separators": ("\n", " ")}, "must end with ''"),
            ({"separators": ()}, "must not be empty"),
            ({"separators": (" ", " ", "")}, "must not repeat"),
            ({"min_length": 0}, "min_length must be a positive int"),
        ],
    )
    def test_recursive_refuses(self, kwargs, match):
        """Values out of range raise ``ValueError``."""
        with pytest.raises(ValueError, match=match):
            RecursiveChunkerConfig(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"child_size": 100, "parent_size": 100}, "smaller than parent_size"),
            ({"child_overlap": 400}, r"child_overlap must be"),
        ],
    )
    def test_parent_child_refuses(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ParentChildChunkerConfig(**kwargs)

    @pytest.mark.parametrize(
        ("factory", "match"),
        [
            (lambda: RecursiveChunkerConfig(separators={"\n", ""}), "tuple of str"),
            (lambda: RecursiveChunkerConfig(separators=("\n", 1, "")), "only str"),
            (lambda: ParentChildChunkerConfig(emit_parents=1), "must be a bool"),
        ],
    )
    def test_wrong_types_raise_type_error(self, factory, match):
        with pytest.raises(TypeError, match=match):
            factory()

    def test_a_config_of_another_type_is_refused(self):
        with pytest.raises(TypeError, match="RecursiveChunkerConfig"):
            RecursiveChunker(ParentChildChunkerConfig())
        with pytest.raises(TypeError, match="ParentChildChunkerConfig"):
            ParentChildChunker(RecursiveChunkerConfig())


class TestRecursiveSpans:
    @pytest.mark.parametrize("size", [1, 3, 8, 25, 60, 500])
    def test_the_spans_partition_and_fit(self, size):
        rng = random.Random(size)  # noqa: S311 (a reproducible fuzz)
        alphabet = "ab cd.\n東京"
        for _ in range(300):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
            spans = recursive_spans(
                text, (0, len(text)), size, ("\n\n", "\n", ". ", " ", "")
            )
            assert spans[0][0] == 0
            assert spans[-1][1] == len(text)
            assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
            assert all(len(text[s:e].strip()) <= size for s, e in spans)

    def test_the_coarsest_separator_that_occurs_is_used(self):
        chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=40))
        texts = [c.text for c in chunker.chunk(TEXT).chunks]
        assert texts[:2] == [
            "Alpha one is here. Beta two is there.",
            "Gamma three goes on.",
        ]

    def test_long_whitespace_and_separator_free_texts_are_linear(self):
        """
        Review 1-2: a whitespace run was quadratic (40k spaces: 38 s), and
        text without separators held one tuple per character (5 MB: +667 MB).
        """
        spaces = "a" + " " * 400_000 + "b"
        assert recursive_spans(spaces, (0, len(spaces)), 1000, ("\n", " ", "")) == [
            (0, len(spaces) - 1),
            (len(spaces) - 1, len(spaces)),
        ]
        solid = "x" * 2_000_000
        windows = recursive_spans(solid, (0, len(solid)), 1000, ("\n", " ", ""))
        assert len(windows) == 2000
        assert all(e - s == 1000 for s, e in windows)

    def test_many_tiny_pieces_are_not_held_at_once(self):
        """Re-review 1: a 5M-space run held one tuple per piece (+513 MB)."""
        import tracemalloc  # noqa: PLC0415

        text = "a" + " " * 1_000_000 + "b"
        tracemalloc.start()
        try:
            recursive_spans(text, (0, len(text)), 1000, ("\n", " ", ""))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 5_000_000  # a list of 1M tuples is about 100 MB

    def test_a_word_longer_than_the_budget_is_cut_into_characters(self):
        chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=20))
        long = [c.text for c in chunker.chunk(TEXT).chunks][-4:]
        assert all(len(t) <= 20 for t in long)
        assert "".join(long) == "longwordwithoutanyspaces" * 3


class TestRecursiveChunker:
    @pytest.mark.parametrize("overlap", [0, 5, 15])
    def test_every_chunk_is_its_source_slice(self, overlap):
        chunker = RecursiveChunker(
            RecursiveChunkerConfig(chunk_size=30, chunk_overlap=overlap)
        )
        chunks = chunker.chunk(TEXT).chunks
        for chunk in chunks:
            start, end = chunk.context_span
            assert chunk.text == TEXT[start:end]
            assert start <= chunk.start_char <= chunk.end_char == end
            assert len(TEXT[chunk.start_char : chunk.end_char]) <= 30

    def test_overlap_starts_at_a_word_and_stays_within_the_previous_chunk(self):
        chunker = RecursiveChunker(
            RecursiveChunkerConfig(chunk_size=20, chunk_overlap=8)
        )
        chunks = chunker.chunk(TEXT).chunks
        for previous, chunk in zip(chunks, chunks[1:]):
            start = chunk.context_span[0]
            assert previous.start_char <= start <= chunk.start_char
            assert start == chunk.start_char or TEXT[start - 1].isspace()

    def test_without_stripping_no_chunk_or_context_is_only_whitespace(self):
        r"""Review 6: '\n\n\n\n' was a chunk, and a context began in it."""
        text = "\n\n\n\n" + "word " * 10
        config = RecursiveChunkerConfig(
            chunk_size=12, chunk_overlap=5, strip_whitespace=False
        )
        chunks = RecursiveChunker(config).chunk(text).chunks
        assert all(c.text.strip() for c in chunks)
        assert all(not text[c.context_span[0]].isspace() for c in chunks[1:])

    def test_a_list_of_separators_is_accepted(self):
        """A JSON or YAML configuration gives a list."""
        config = RecursiveChunkerConfig(separators=["\n", " ", ""])
        assert config.separators == ("\n", " ", "")

    def test_min_length_drops_short_chunks(self):
        chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=20, min_length=14))
        assert all(len(c.text) >= 14 for c in chunker.chunk(TEXT).chunks)

    def test_input_is_validated(self):
        with pytest.raises(TypeError):
            RecursiveChunker().chunk(b"x")
        with pytest.raises(ValueError, match="empty"):
            RecursiveChunker().chunk("   ")


class TestParentChild:
    def _chunks(self, **kwargs):
        config = ParentChildChunkerConfig(parent_size=60, child_size=25, **kwargs)
        return ParentChildChunker(config).chunk(TEXT).chunks

    def test_children_lie_inside_their_parent(self):
        chunks = self._chunks()
        for index, chunk in enumerate(chunks):
            assert chunk.text == TEXT[slice(*chunk.context_span)]
            if chunk.metadata["chunk_role"] == "child":
                parent = chunks[chunk.metadata["parent_chunk"]]
                assert parent.metadata["chunk_role"] == "parent"
                assert (
                    parent.start_char
                    <= chunk.start_char
                    < chunk.end_char
                    <= parent.end_char
                )
                assert chunk.metadata["parent_span"] == [
                    parent.start_char,
                    parent.end_char,
                ]
            assert chunk.metadata["chunk_index"] == index

    def test_each_parent_precedes_its_children_in_source_order(self):
        chunks = self._chunks()
        positions = [c.metadata.get("parent_chunk") for c in chunks]
        for index, position in enumerate(positions):
            if position is not None:
                assert position < index

    def test_children_only(self):
        chunks = self._chunks(emit_parents=False)
        assert {c.metadata["chunk_role"] for c in chunks} == {"child"}
        assert all("parent_chunk" not in c.metadata for c in chunks)

    def test_child_overlap_stays_inside_the_parent(self):
        chunks = self._chunks(child_overlap=10)
        for chunk in chunks:
            if chunk.metadata["chunk_role"] == "child":
                p_start = chunk.metadata["parent_span"][0]
                assert p_start <= chunk.context_span[0]


class TestDocuments:
    def _build(self, tmp_path, chunker, **kwargs):
        source = tmp_path / "doc.txt"
        source.write_text(TEXT, encoding="utf-8")
        config = BuilderConfig(
            chunker=chunker,
            chunker_kwargs=kwargs,
            normalize=False,
            filter_kwargs={"min_words": 0, "min_chars": 0},
        )
        return CorpusBuilder(config).build(source).documents

    def test_children_carry_their_parents_doc_id(self, tmp_path):
        """The first producer of parent_doc_id (V2-HIER-001)."""
        docs = self._build(tmp_path, "parent_child", parent_size=60, child_size=25)
        parents = [d for d in docs if d.metadata["chunk_role"] == "parent"]
        kids = [d for d in docs if d.metadata["chunk_role"] == "child"]
        assert parents
        assert kids
        assert all(d.parent_doc_id is None for d in parents)
        ids = {d.doc_id for d in parents}
        assert all(d.parent_doc_id in ids for d in kids)
        assert _hierarchy.validate_hierarchy(docs).ok
        for parent in parents:
            for child in _hierarchy.children(docs, parent.doc_id):
                assert parent.char_start <= child.char_start
                assert child.char_end <= parent.char_end

    def test_a_chunk_naming_no_earlier_chunk_is_refused(self, tmp_path):
        from ..._types import Chunk, ChunkResult  # noqa: PLC0415

        class Broken:
            strategy = "custom"

            def chunk(self, text, extra_metadata=None):
                return ChunkResult(
                    chunks=[
                        Chunk(
                            text="a",
                            start_char=0,
                            end_char=1,
                            metadata={"parent_chunk": 1},
                        ),
                        Chunk(text="b", start_char=1, end_char=2),
                    ]
                )

        source = tmp_path / "doc.txt"
        source.write_text("ab", encoding="utf-8")
        with pytest.raises(ValueError, match="names parent chunk 1"):
            _read_all(source, Broken())

    def test_a_parent_the_filter_drops_leaves_no_dangling_link(self, tmp_path):
        """Review 3: children named a parent document that was not yielded."""
        from ..._base import DefaultFilter, DocumentReader  # noqa: PLC0415
        from .._chunker_bridge import bridge_chunker  # noqa: PLC0415

        class ShortOnly(DefaultFilter):
            def include(self, doc):
                return len(doc.text) <= 30

        source = tmp_path / "doc.txt"
        source.write_text(TEXT, encoding="utf-8")
        chunker = bridge_chunker(
            ParentChildChunker(ParentChildChunkerConfig(parent_size=60, child_size=25))
        )
        docs = list(
            DocumentReader.create(
                source, chunker=chunker, filter_=ShortOnly()
            ).get_documents()
        )
        kept = {d.doc_id for d in docs if d.metadata["chunk_role"] == "parent"}
        kids = [d for d in docs if d.metadata["chunk_role"] == "child"]
        orphans = [d for d in kids if d.metadata.get("parent_omitted")]
        assert orphans  # some parents were longer than 30 characters
        assert all(d.parent_doc_id is None for d in orphans)
        assert all(d.parent_doc_id in kept for d in kids if d not in orphans)
        assert _hierarchy.validate_hierarchy(docs).ok

    def test_an_omitted_parent_clears_the_source_parent_and_bool_is_refused(
        self, tmp_path
    ):
        """Re-review 2-3: the source-level parent survived; True was accepted."""
        from ..._base import DefaultFilter, DocumentReader  # noqa: PLC0415
        from ..._types import Chunk, ChunkResult  # noqa: PLC0415

        class DropFirst(DefaultFilter):
            def include(self, doc):
                return doc.text != "a"

        class Stub:
            strategy = "custom"

            def __init__(self, link):
                self.link = link

            def chunk(self, text, extra_metadata=None):
                return ChunkResult(
                    chunks=[
                        Chunk(text="a", start_char=0, end_char=1),
                        Chunk(
                            text="b",
                            start_char=1,
                            end_char=2,
                            metadata={"parent_chunk": self.link},
                        ),
                    ]
                )

        source = tmp_path / "doc.txt"
        source.write_text("ab", encoding="utf-8")
        reader = DocumentReader.create(source, chunker=Stub(0), filter_=DropFirst())
        reader.source_provenance = {"parent_doc_id": "f" * 64}
        (child,) = list(reader.get_documents())
        assert child.metadata["parent_omitted"] is True
        assert child.parent_doc_id is None
        with pytest.raises(ValueError, match="names parent chunk True"):
            list(
                DocumentReader.create(
                    source, chunker=Stub(True), filter_=DropFirst()
                ).get_documents()
            )

    def test_children_only_keep_their_parent_span(self, tmp_path):
        """Review 4: with emit_parents=False the document lost the link."""
        docs = self._build(
            tmp_path, "parent_child", parent_size=60, child_size=25, emit_parents=False
        )
        for doc in docs:
            start, end = doc.metadata["parent_span"]
            assert start <= doc.char_start < doc.char_end <= end

    def test_recursive_by_name(self, tmp_path):
        docs = self._build(tmp_path, "recursive", chunk_size=40)
        assert all(len(d.text) <= 40 for d in docs)
        assert all(d.raw_text == d.text for d in docs)


def _read_all(source, chunker):
    from ..._base import DocumentReader  # noqa: PLC0415

    reader = DocumentReader.create(source, chunker=chunker)
    return list(reader.get_documents())
