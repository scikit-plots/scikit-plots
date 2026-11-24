"""A plan carries a filter: ``FluentCorpus.filter`` reaches the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from .. import (
    HAMLET_TEXT,
    CorpusPipeline,
    FilterBase,
    FluentCorpus,
    HashEmbedder,
    ParagraphChunkerConfig,
    RetrievalConfig,
    RuntimePolicy,
)
from .._plan import CONFIG_DOMAINS, INGEST_DOMAINS


@dataclass(frozen=True)
class MinWords(FilterBase):
    """Keep documents with at least ``n`` words."""

    n: int = 40

    def include(self, doc) -> bool:
        return len(doc.text.split()) >= self.n


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "hamlet.txt"
    path.write_text(HAMLET_TEXT, encoding="utf-8")
    return path


def _plan(source, **extra):
    fluent = (
        FluentCorpus()
        .source(source)
        .chunker(ParagraphChunkerConfig(min_length=40, max_length=900))
        .embedder(HashEmbedder(dimension=64))
        .index(RetrievalConfig(backend="bruteforce"))
    )
    return fluent.filter(extra["filter"]) if "filter" in extra else fluent


def _run(fluent):
    with fluent.materialize(policy=RuntimePolicy(allow_network=False)) as rt:
        rt.run()
        return [d.text for d in rt.documents]


def test_filter_is_an_ingest_domain():
    assert "filter" in CONFIG_DOMAINS and "filter" in INGEST_DOMAINS


def test_the_filter_decides_which_documents_are_kept(source):
    everything = _run(_plan(source))
    kept = _run(_plan(source, filter=MinWords(40)))
    assert 0 < len(kept) < len(everything)
    assert all(len(t.split()) >= 40 for t in kept)


def test_add_uses_the_same_filter(source, tmp_path):
    short = tmp_path / "short.txt"
    short.write_text("HAMLET. Words, words, words.\n", encoding="utf-8")
    with _plan(source, filter=MinWords(40)).materialize(policy=RuntimePolicy()) as rt:
        rt.run()
        before = len(rt.documents)
        rt.add(short)
        assert len(rt.documents) == before  # the short speech is filtered out


def test_filter_moves_the_fingerprint_and_replace_filter_works(source):
    plain, filtered = _plan(source), _plan(source, filter=MinWords(40))
    assert plain.plan().fingerprint != filtered.plan().fingerprint
    other = filtered.replace_filter(MinWords(10))
    assert other.plan().get("filter") == MinWords(10)
    assert filtered.plan().fingerprint == _plan(source, filter=MinWords(40)).plan().fingerprint


def test_a_class_or_a_registered_name_is_accepted(source):
    assert _run(_plan(source, filter=MinWords)) == _run(_plan(source, filter=MinWords(40)))
    assert _run(_plan(source, filter="default"))  # the built-in DefaultFilter


def test_an_object_without_include_is_refused(source):
    with pytest.raises(TypeError, match="include"):
        _plan(source, filter=object()).materialize()


def test_a_reader_option_the_pipeline_sets_is_refused_early():
    with pytest.raises(ValueError, match=r"\['filter_'\].*FluentCorpus.filter"):
        CorpusPipeline(reader_kwargs={"filter_": MinWords()})
    with pytest.raises(ValueError, match="chunker"):
        CorpusPipeline(reader_kwargs={"chunker": None, "encoding": "detect"})
    CorpusPipeline(reader_kwargs={"encoding": "detect"})  # other options pass
