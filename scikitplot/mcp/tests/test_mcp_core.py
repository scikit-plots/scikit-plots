# tests/test_mcp_core.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for scikitplot.mcp core (retriever contract, result builder, safety)."""
from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

# Make the proposed package importable in isolation (scikitplot is an implicit
# namespace package here; no heavy scikitplot.__init__ is loaded).
_PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from scikitplot.mcp import (  # noqa: E402
    MAX_CHUNK_CHARS,
    MAX_RESULTS,
    CorpusAnnoyRetriever,
    DocsRetriever,
    RetrievedChunk,
    build_search_docs_result,
)


# ── Test doubles ─────────────────────────────────────────────────────────────
class MockRetriever:
    """Minimal DocsRetriever for tests (no corpus/annoy needed)."""

    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, k=5):
        return self._chunks[:k]


_CHUNKS = [
    RetrievedChunk(text="ROC curves plot TPR vs FPR.", source_uri="https://d.io/roc.html",
                   score=0.91, doc_id="roc-1", title="ROC", anchor="tpr-fpr"),
    RetrievedChunk(text="AUC summarises the ROC curve.", source_uri="https://d.io/auc.html",
                   score=0.83, doc_id="auc-1", title="AUC"),
]


# ── DocsRetriever protocol ───────────────────────────────────────────────────
def test_mock_satisfies_protocol():
    assert isinstance(MockRetriever(_CHUNKS), DocsRetriever)


def test_retriever_respects_k():
    r = MockRetriever(_CHUNKS)
    assert len(r.search("x", k=1)) == 1
    assert len(r.search("x", k=5)) == 2


# ── build_search_docs_result: shape + citations ──────────────────────────────
def test_result_shape_and_citations():
    res = build_search_docs_result("roc", _CHUNKS)
    assert res["isError"] is False
    assert len(res["content"]) == 2
    cites = res["structuredContent"]["citations"]
    assert cites[0]["source_uri"] == "https://d.io/roc.html#tpr-fpr", "anchor deep-linked"
    assert cites[1]["source_uri"] == "https://d.io/auc.html"
    assert res["structuredContent"]["query"] == "roc"
    # every content block carries its citation marker
    assert "[1]" in res["content"][0]["text"] and "d.io/roc.html" in res["content"][0]["text"]


def test_empty_results_graceful():
    res = build_search_docs_result("nothing", [])
    assert res["isError"] is False
    assert res["structuredContent"]["citations"] == []
    assert "No matching" in res["content"][0]["text"]


# ── Safety: injection, control chars, truncation ─────────────────────────────
def test_unsafe_uri_dropped():
    bad = [RetrievedChunk(text="x", source_uri="javascript:alert(1)", title="evil")]
    res = build_search_docs_result("q", bad)
    assert res["structuredContent"]["citations"][0]["source_uri"] == "", "js: scheme stripped"


def test_data_uri_dropped():
    bad = [RetrievedChunk(text="x", source_uri="data:text/html,<script>")]
    res = build_search_docs_result("q", bad)
    assert res["structuredContent"]["citations"][0]["source_uri"] == ""


def test_control_chars_stripped():
    bad = [RetrievedChunk(text="a\x00b\x07c", source_uri="https://d.io/x")]
    res = build_search_docs_result("q", bad)
    assert "\x00" not in res["content"][0]["text"]
    assert "abc" in res["content"][0]["text"]


def test_text_truncated():
    huge = [RetrievedChunk(text="x" * (MAX_CHUNK_CHARS + 500), source_uri="https://d.io/x")]
    res = build_search_docs_result("q", huge)
    body = res["content"][0]["text"]
    # header + capped text + cite line; the chunk text itself must be capped
    assert len(body) < MAX_CHUNK_CHARS + 300
    assert body.rstrip().endswith("\u2026") or "\u2026" in body


def test_max_results_capped():
    many = [
        RetrievedChunk(text=str(i), source_uri="https://d.io/%d" % i)
        for i in range(MAX_RESULTS + 10)
    ]
    res = build_search_docs_result("q", many, max_results=1000)
    assert len(res["content"]) == MAX_RESULTS


def test_relative_uri_allowed():
    rel = [RetrievedChunk(text="x", source_uri="/en/stable/api.html", anchor="sec")]
    res = build_search_docs_result("q", rel)
    assert res["structuredContent"]["citations"][0]["source_uri"] == "/en/stable/api.html#sec"


# ── CorpusAnnoyRetriever via dependency injection ────────────────────────────
class _FakeEmbedder:
    def embed(self, text):
        return [float(len(text)), 1.0, 2.0]


class _FakeIndex:
    def query(self, vector, k):
        return [("d1", 0.95), ("d2", 0.80)][:k]


def _fake_lookup(doc_id):
    return {
        "d1": {"text": "chunk one", "source_uri": "https://d.io/1", "title": "One", "anchor": "a1"},
        "d2": {"text": "chunk two", "source_uri": "https://d.io/2", "title": "Two"},
    }.get(doc_id, {})


def test_corpus_annoy_retriever_composition():
    r = CorpusAnnoyRetriever(_FakeEmbedder(), _FakeIndex(), _fake_lookup)
    hits = r.search("what is roc", k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "d1" and hits[0].source_uri == "https://d.io/1"
    assert hits[0].title == "One" and hits[0].anchor == "a1"
    # end-to-end through the result builder
    res = build_search_docs_result("what is roc", hits)
    assert res["structuredContent"]["citations"][0]["source_uri"] == "https://d.io/1#a1"


def test_corpus_annoy_empty_query():
    r = CorpusAnnoyRetriever(_FakeEmbedder(), _FakeIndex(), _fake_lookup)
    assert r.search("   ", k=3) == []


def test_from_corpus_annoy_raises_without_deps(monkeypatch):
    """The optional-dependency failure is simulated, not environment-dependent."""
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scikitplot.corpus" or name.startswith("scikitplot.corpus."):
            raise ImportError("injected missing scikitplot.corpus")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="scikitplot.corpus"):
        CorpusAnnoyRetriever.from_corpus_annoy("/tmp/docs")


def test_from_corpus_annoy_uses_current_corpus_api(monkeypatch, tmp_path):
    """Wire BuilderConfig, CorpusBuilder, and EmbeddingEngine as documented."""
    calls = {}

    class FakeBuilderConfig:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

    class FakeIndex:
        def query(self, vector, k):
            calls["index_query"] = (vector, k)
            return [("doc-1", 0.95)]

    class FakeCorpusBuilder:
        def __init__(self, config):
            calls["builder_config"] = config

        def build(self, docs_path):
            calls["docs_path"] = docs_path
            doc = types.SimpleNamespace(
                doc_id="doc-1",
                normalized_text="normalized passage",
                source_uri="https://docs.example/api.html",
                title="API reference",
                anchor="section-1",
            )
            return types.SimpleNamespace(documents=[doc], index=FakeIndex())

    class FakeEmbeddingEngine:
        def __init__(self, *, model_name):
            calls["embedding_model"] = model_name

        def embed(self, texts):
            calls["embed_texts"] = texts
            return [[1.0, 2.0, 3.0]]

    fake_corpus = types.ModuleType("scikitplot.corpus")
    fake_corpus.BuilderConfig = FakeBuilderConfig
    fake_corpus.CorpusBuilder = FakeCorpusBuilder
    fake_corpus.EmbeddingEngine = FakeEmbeddingEngine
    monkeypatch.setitem(sys.modules, "scikitplot.corpus", fake_corpus)

    retriever = CorpusAnnoyRetriever.from_corpus_annoy(
        str(tmp_path),
        metric="angular",
        n_trees=7,
        embedding_model="fake-model",
        backend="auto",
    )
    hits = retriever.search("where is the API?", k=1)

    assert calls["docs_path"] == str(tmp_path)
    assert calls["config"] == {
        "chunker": "paragraph",
        "normalize": True,
        "enrich": True,
        "embed": True,
        "embedding_model": "fake-model",
        "build_index": True,
        "index_kwargs": {
            "match_mode": "semantic",
            "backend": "auto",
            "annoy_metric": "angular",
            "annoy_n_trees": 7,
        },
    }
    assert calls["embedding_model"] == "fake-model"
    assert calls["embed_texts"] == ["where is the API?"]
    assert calls["index_query"] == ([1.0, 2.0, 3.0], 1)
    assert len(hits) == 1
    assert hits[0].doc_id == "doc-1"
    assert hits[0].text == "normalized passage"
    assert hits[0].source_uri == "https://docs.example/api.html"
    assert hits[0].title == "API reference"
    assert hits[0].anchor == "section-1"


def test_from_corpus_annoy_raises_when_build_has_no_index(monkeypatch, tmp_path):
    """A successful corpus import without a dense index remains actionable."""

    class FakeBuilderConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCorpusBuilder:
        def __init__(self, config):
            self.config = config

        def build(self, docs_path):
            return types.SimpleNamespace(documents=[], index=None)

    class FakeEmbeddingEngine:
        def __init__(self, *, model_name):
            self.model_name = model_name

    fake_corpus = types.ModuleType("scikitplot.corpus")
    fake_corpus.BuilderConfig = FakeBuilderConfig
    fake_corpus.CorpusBuilder = FakeCorpusBuilder
    fake_corpus.EmbeddingEngine = FakeEmbeddingEngine
    monkeypatch.setitem(sys.modules, "scikitplot.corpus", fake_corpus)

    with pytest.raises(RuntimeError, match="no queryable semantic index"):
        CorpusAnnoyRetriever.from_corpus_annoy(str(tmp_path))
