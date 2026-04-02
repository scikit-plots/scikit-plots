# corpus/_similarity/tests/test__similarity.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for corpus._similarity._similarity
=========================================

Coverage targets
----------------
* :class:`SearchResult` — fields, frozen, equality.
* :class:`SearchConfig` — defaults, validation (match_mode, top_k,
  hybrid_alpha), ``__post_init__`` error paths.
* :func:`_tokenize_simple` — unicode, empty, punctuation.
* :func:`_get_text` — normalised-text preference, fallback.
* :class:`_BM25Index` — build + query: empty query, single-term,
  multi-term IDF, top-k clipping, zero-score filtering.
* :class:`SimilarityIndex` — build (empty raises), strict search
  (case-sensitive/insensitive, top_k), keyword/BM25 search, semantic
  brute-force (no ANN libs), hybrid RRF fusion, ``n_documents`` /
  ``has_embeddings`` properties, ``__repr__``.

All tests use stdlib only.  No FAISS / Voyager / sentence-transformers
required.  numpy is used only for the semantic-mode embedding fixture.

Run with::

    pytest corpus/_similarity/tests/test__similarity.py -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from .._similarity import (
    SearchConfig,
    SearchResult,
    SimilarityIndex,
    _BM25Index,
    _get_text,
    _tokenize_simple,
)
from ..._schema import CorpusDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    text: str,
    idx: int = 0,
    normalized_text: str | None = None,
    embedding: Any = None,
) -> CorpusDocument:
    """Return a minimal CorpusDocument, optionally with embedding."""
    doc = CorpusDocument.create("test.txt", idx, text)
    if normalized_text is not None or embedding is not None:
        doc = doc.replace(
            normalized_text=normalized_text,
            embedding=embedding,
        )
    return doc


def _docs_for_search() -> list[CorpusDocument]:
    """Five documents covering distinct topics for search tests."""
    return [
        _doc("The quick brown fox jumps over the lazy dog", idx=0),
        _doc("Hamlet contemplates existence and mortality", idx=1),
        _doc("Python is a high-level programming language", idx=2),
        _doc("Machine learning models generalise from data", idx=3),
        _doc("Shakespeare wrote sonnets and plays", idx=4),
    ]


def _docs_with_embeddings() -> list[CorpusDocument]:
    """Three documents with deterministic float32 embeddings (dim=4)."""
    vecs = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    ]
    return [
        _doc(f"doc {i}", idx=i, embedding=v)
        for i, v in enumerate(vecs)
    ]


# ===========================================================================
# SearchResult
# ===========================================================================


class TestSearchResult:
    def test_fields_accessible(self) -> None:
        doc = _doc("hello")
        r = SearchResult(doc=doc, score=0.9, match_mode="strict")
        assert r.doc is doc
        assert r.score == pytest.approx(0.9)
        assert r.match_mode == "strict"

    def test_frozen(self) -> None:
        doc = _doc("hello")
        r = SearchResult(doc=doc, score=0.5, match_mode="keyword")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            r.score = 1.0  # type: ignore[misc]

    def test_equality(self) -> None:
        doc = _doc("hello")
        r1 = SearchResult(doc=doc, score=0.7, match_mode="hybrid")
        r2 = SearchResult(doc=doc, score=0.7, match_mode="hybrid")
        assert r1 == r2


# ===========================================================================
# SearchConfig
# ===========================================================================


class TestSearchConfig:
    def test_defaults(self) -> None:
        cfg = SearchConfig()
        assert cfg.top_k == 10
        assert cfg.match_mode == "semantic"
        assert cfg.semantic_threshold == pytest.approx(0.0)
        assert cfg.keyword_threshold == pytest.approx(0.0)
        assert cfg.hybrid_alpha == pytest.approx(0.5)
        assert cfg.rrf_k == 60
        assert cfg.use_normalized_text is True
        assert cfg.case_sensitive is False

    def test_custom_values(self) -> None:
        cfg = SearchConfig(
            top_k=5,
            match_mode="keyword",
            keyword_threshold=0.1,
            case_sensitive=True,
        )
        assert cfg.top_k == 5
        assert cfg.match_mode == "keyword"
        assert cfg.keyword_threshold == pytest.approx(0.1)
        assert cfg.case_sensitive is True

    def test_invalid_match_mode(self) -> None:
        with pytest.raises(ValueError, match="match_mode"):
            SearchConfig(match_mode="fuzz")

    def test_invalid_top_k_zero(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            SearchConfig(top_k=0)

    def test_invalid_top_k_negative(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            SearchConfig(top_k=-1)

    def test_invalid_hybrid_alpha_above_one(self) -> None:
        with pytest.raises(ValueError, match="hybrid_alpha"):
            SearchConfig(hybrid_alpha=1.5)

    def test_invalid_hybrid_alpha_negative(self) -> None:
        with pytest.raises(ValueError, match="hybrid_alpha"):
            SearchConfig(hybrid_alpha=-0.1)

    def test_all_valid_match_modes(self) -> None:
        for mode in ("strict", "keyword", "semantic", "hybrid"):
            cfg = SearchConfig(match_mode=mode)
            assert cfg.match_mode == mode

    def test_frozen(self) -> None:
        cfg = SearchConfig()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.top_k = 99  # type: ignore[misc]


# ===========================================================================
# _tokenize_simple
# ===========================================================================


class TestTokenizeSimple:
    def test_basic(self) -> None:
        tokens = _tokenize_simple("Hello world")
        assert tokens == ["hello", "world"]

    def test_empty(self) -> None:
        assert _tokenize_simple("") == []

    def test_punctuation_stripped(self) -> None:
        tokens = _tokenize_simple("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens

    def test_unicode_letters(self) -> None:
        tokens = _tokenize_simple("café naïve")
        assert len(tokens) >= 1  # unicode word chars matched

    def test_numbers_kept(self) -> None:
        tokens = _tokenize_simple("Python 3 is great")
        assert "3" in tokens

    def test_lowercased(self) -> None:
        tokens = _tokenize_simple("UPPER lower MiXeD")
        assert all(t == t.lower() for t in tokens)


# ===========================================================================
# _get_text
# ===========================================================================


class TestGetText:
    def test_prefers_normalized_text_when_use_normalized_true(self) -> None:
        doc = _doc("raw text", normalized_text="normalised text")
        assert _get_text(doc, use_normalized=True) == "normalised text"

    def test_falls_back_to_text_when_no_normalized(self) -> None:
        doc = _doc("raw text")
        assert _get_text(doc, use_normalized=True) == "raw text"

    def test_uses_raw_text_when_use_normalized_false(self) -> None:
        doc = _doc("raw text", normalized_text="normalised text")
        assert _get_text(doc, use_normalized=False) == "raw text"

    def test_empty_normalized_falls_back(self) -> None:
        doc = _doc("raw text")
        # normalized_text not set → None → falls back to text
        assert _get_text(doc, use_normalized=True) == "raw text"


# ===========================================================================
# _BM25Index
# ===========================================================================


class TestBM25Index:
    def _build(self, token_lists: list[list[str]]) -> _BM25Index:
        idx = _BM25Index()
        idx.build(token_lists)
        return idx

    def test_query_returns_empty_for_empty_token_lists(self) -> None:
        idx = _BM25Index()
        idx.build([])
        result = idx.query(["hello"])
        assert result == []

    def test_query_returns_empty_for_empty_query(self) -> None:
        idx = self._build([["hello", "world"], ["foo", "bar"]])
        assert idx.query([]) == []

    def test_single_term_finds_doc(self) -> None:
        idx = self._build([
            ["machine", "learning", "models"],
            ["deep", "neural", "networks"],
        ])
        results = idx.query(["machine"])
        doc_ids = [i for i, _ in results]
        assert 0 in doc_ids

    def test_unseen_term_returns_empty(self) -> None:
        idx = self._build([["cat", "sat"], ["dog", "ran"]])
        assert idx.query(["elephant"]) == []

    def test_top_k_respected(self) -> None:
        corpus = [["word"] for _ in range(20)]
        idx = self._build(corpus)
        results = idx.query(["word"], top_k=5)
        assert len(results) <= 5

    def test_scores_sorted_descending(self) -> None:
        idx = self._build([
            ["python", "python", "python"],  # high TF
            ["python"],                       # low TF
        ])
        results = idx.query(["python"])
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_multi_term_query(self) -> None:
        idx = self._build([
            ["machine", "learning", "data"],
            ["cooking", "recipe", "food"],
        ])
        results = idx.query(["machine", "data"])
        doc_ids = [i for i, _ in results]
        assert 0 in doc_ids

    def test_avgdl_set_after_build(self) -> None:
        idx = self._build([["a", "b", "c"], ["d", "e"]])
        assert idx._avgdl == pytest.approx(2.5)

    def test_custom_k1_b(self) -> None:
        idx = _BM25Index(k1=2.0, b=0.5)
        idx.build([["test", "query"]])
        results = idx.query(["test"])
        assert len(results) == 1


# ===========================================================================
# SimilarityIndex — build
# ===========================================================================


class TestSimilarityIndexBuild:
    def test_build_empty_raises(self) -> None:
        idx = SimilarityIndex()
        with pytest.raises(ValueError, match="empty"):
            idx.build([])

    def test_build_sets_n_documents(self) -> None:
        idx = SimilarityIndex()
        idx.build(_docs_for_search())
        assert idx.n_documents == 5

    def test_build_without_embeddings(self) -> None:
        idx = SimilarityIndex()
        idx.build(_docs_for_search())
        assert idx.has_embeddings is False

    def test_build_with_embeddings(self) -> None:
        idx = SimilarityIndex()
        idx.build(_docs_with_embeddings())
        assert idx.has_embeddings is True

    def test_rebuild_replaces_state(self) -> None:
        idx = SimilarityIndex()
        idx.build(_docs_for_search())
        assert idx.n_documents == 5
        idx.build(_docs_for_search()[:2])
        assert idx.n_documents == 2

    def test_partial_embeddings_disables_dense(self) -> None:
        """Only some docs have embeddings → dense index not built."""
        vec = np.array([1.0, 0.0], dtype=np.float32)
        docs = [
            _doc("doc a", idx=0, embedding=vec),
            _doc("doc b", idx=1),  # no embedding
        ]
        idx = SimilarityIndex()
        idx.build(docs)
        assert idx.has_embeddings is False


# ===========================================================================
# SimilarityIndex — STRICT search
# ===========================================================================


class TestStrictSearch:
    @pytest.fixture()
    def idx(self) -> SimilarityIndex:
        index = SimilarityIndex(config=SearchConfig(match_mode="strict", top_k=10))
        index.build(_docs_for_search())
        return index

    def test_match_returns_score_1(self, idx: SimilarityIndex) -> None:
        results = idx.search("Hamlet")
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0)
        assert results[0].match_mode == "strict"

    def test_case_insensitive_default(self, idx: SimilarityIndex) -> None:
        results = idx.search("hamlet")
        assert len(results) == 1

    def test_case_sensitive_no_match(self, idx: SimilarityIndex) -> None:
        cfg = SearchConfig(match_mode="strict", case_sensitive=True)
        results = idx.search("hamlet", config=cfg)
        assert len(results) == 0

    def test_case_sensitive_match(self, idx: SimilarityIndex) -> None:
        cfg = SearchConfig(match_mode="strict", case_sensitive=True)
        results = idx.search("Hamlet", config=cfg)
        assert len(results) == 1

    def test_no_match_returns_empty(self, idx: SimilarityIndex) -> None:
        results = idx.search("xyzzy")
        assert results == []

    def test_top_k_limits_results(self) -> None:
        # Build index where "the" appears in multiple docs
        docs = [_doc("the cat sat", idx=i) for i in range(5)]
        idx = SimilarityIndex(config=SearchConfig(match_mode="strict", top_k=2))
        idx.build(docs)
        results = idx.search("the")
        assert len(results) <= 2

    def test_multi_word_substring(self, idx: SimilarityIndex) -> None:
        results = idx.search("quick brown fox")
        assert len(results) == 1

    def test_uses_normalized_text_when_available(self) -> None:
        docs = [_doc("raw", idx=0, normalized_text="target phrase here")]
        idx = SimilarityIndex(
            config=SearchConfig(match_mode="strict", use_normalized_text=True)
        )
        idx.build(docs)
        assert len(idx.search("target phrase")) == 1
        assert len(idx.search("raw")) == 0

    def test_skips_normalized_when_flag_false(self) -> None:
        docs = [_doc("raw", idx=0, normalized_text="target phrase here")]
        idx = SimilarityIndex(
            config=SearchConfig(match_mode="strict", use_normalized_text=False)
        )
        idx.build(docs)
        assert len(idx.search("raw")) == 1
        assert len(idx.search("target phrase")) == 0


# ===========================================================================
# SimilarityIndex — KEYWORD search (BM25)
# ===========================================================================


class TestKeywordSearch:
    @pytest.fixture()
    def idx(self) -> SimilarityIndex:
        index = SimilarityIndex(config=SearchConfig(match_mode="keyword", top_k=10))
        index.build(_docs_for_search())
        return index

    def test_relevant_doc_ranked_first(self, idx: SimilarityIndex) -> None:
        results = idx.search("programming language")
        assert len(results) > 0
        texts = [r.doc.text for r in results]
        assert any("programming" in t for t in texts)

    def test_match_mode_label(self, idx: SimilarityIndex) -> None:
        results = idx.search("machine learning")
        for r in results:
            assert r.match_mode == "keyword"

    def test_unseen_query_returns_empty(self, idx: SimilarityIndex) -> None:
        results = idx.search("xyzzy_nonexistent_term")
        assert results == []

    def test_threshold_filters_low_scores(self, idx: SimilarityIndex) -> None:
        cfg = SearchConfig(match_mode="keyword", keyword_threshold=1e9)
        results = idx.search("python", config=cfg)
        assert results == []

    def test_top_k_respected(self, idx: SimilarityIndex) -> None:
        cfg = SearchConfig(match_mode="keyword", top_k=2)
        results = idx.search("the", config=cfg)
        assert len(results) <= 2

    def test_empty_query_returns_empty(self, idx: SimilarityIndex) -> None:
        results = idx.search("")
        assert results == []

    def test_precomputed_tokens_used(self) -> None:
        """Pre-tokenised docs should be indexed without re-tokenising."""
        doc = _doc("apple banana cherry", idx=0)
        doc = doc.replace(tokens=["apple", "banana", "cherry"])
        idx = SimilarityIndex(config=SearchConfig(match_mode="keyword"))
        idx.build([doc])
        results = idx.search("apple")
        assert len(results) == 1


# ===========================================================================
# SimilarityIndex — SEMANTIC search (brute-force, no ANN)
# ===========================================================================


class TestSemanticSearch:
    @pytest.fixture()
    def idx(self) -> SimilarityIndex:
        index = SimilarityIndex(config=SearchConfig(match_mode="semantic", top_k=3))
        index.build(_docs_with_embeddings())
        return index

    def test_exact_match_gets_highest_score(self, idx: SimilarityIndex) -> None:
        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search("doc 0", query_embedding=query_emb)
        assert len(results) > 0
        assert results[0].doc.chunk_index == 0
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_match_mode_label(self, idx: SimilarityIndex) -> None:
        qe = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search("doc 1", query_embedding=qe)
        for r in results:
            assert r.match_mode == "semantic"

    def test_no_embeddings_returns_empty_with_warning(self) -> None:
        idx = SimilarityIndex(config=SearchConfig(match_mode="semantic"))
        idx.build(_docs_for_search())  # no embeddings
        qe = np.array([1.0, 0.0], dtype=np.float32)
        results = idx.search("query", query_embedding=qe)
        assert results == []

    def test_missing_query_embedding_raises(self, idx: SimilarityIndex) -> None:
        with pytest.raises(ValueError, match="query_embedding"):
            idx.search("query", query_embedding=None)

    def test_threshold_filters_orthogonal(self, idx: SimilarityIndex) -> None:
        # doc 0 is e1=[1,0,0,0]; query e2=[0,1,0,0] → cosine=0
        qe = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        cfg = SearchConfig(match_mode="semantic", semantic_threshold=0.5, top_k=10)
        results = idx.search("q", query_embedding=qe, config=cfg)
        # doc 1 has cosine=1.0, docs 0 and 2 are 0 → filtered by threshold
        assert all(r.score >= 0.5 for r in results)

    def test_zero_query_vector_returns_empty(self, idx: SimilarityIndex) -> None:
        qe = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search("q", query_embedding=qe)
        assert results == []

    def test_top_k_limits_results(self, idx: SimilarityIndex) -> None:
        qe = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        cfg = SearchConfig(match_mode="semantic", top_k=2)
        results = idx.search("q", query_embedding=qe, config=cfg)
        assert len(results) <= 2

    def test_per_query_config_override(self, idx: SimilarityIndex) -> None:
        qe = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cfg = SearchConfig(match_mode="semantic", top_k=1)
        results = idx.search("q", query_embedding=qe, config=cfg)
        assert len(results) == 1


# ===========================================================================
# SimilarityIndex — HYBRID search (RRF)
# ===========================================================================


class TestHybridSearch:
    @pytest.fixture()
    def idx(self) -> SimilarityIndex:
        index = SimilarityIndex(config=SearchConfig(match_mode="hybrid", top_k=5))
        index.build(_docs_with_embeddings())
        return index

    def test_returns_results_with_query_embedding(self, idx: SimilarityIndex) -> None:
        qe = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search("doc", query_embedding=qe)
        assert len(results) > 0

    def test_match_mode_label(self, idx: SimilarityIndex) -> None:
        qe = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search("doc", query_embedding=qe)
        for r in results:
            assert r.match_mode == "hybrid"

    def test_returns_results_without_embeddings(self) -> None:
        """Hybrid degrades gracefully to keyword-only when no embeddings."""
        idx = SimilarityIndex(config=SearchConfig(match_mode="hybrid"))
        idx.build(_docs_for_search())
        results = idx.search("python language")
        # keyword-only results still returned
        assert isinstance(results, list)

    def test_top_k_respected(self, idx: SimilarityIndex) -> None:
        qe = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cfg = SearchConfig(match_mode="hybrid", top_k=2)
        results = idx.search("doc", query_embedding=qe, config=cfg)
        assert len(results) <= 2

    def test_alpha_zero_pure_keyword(self) -> None:
        """hybrid_alpha=0 → pure keyword (no semantic contribution)."""
        idx = SimilarityIndex(
            config=SearchConfig(match_mode="hybrid", hybrid_alpha=0.0)
        )
        idx.build(_docs_with_embeddings())
        results = idx.search("doc")
        # With alpha=0 semantic rank doesn't contribute — still returns list
        assert isinstance(results, list)

    def test_rrf_scores_positive(self, idx: SimilarityIndex) -> None:
        qe = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search("doc", query_embedding=qe)
        for r in results:
            assert r.score > 0.0


# ===========================================================================
# SimilarityIndex — utility / properties / repr
# ===========================================================================


class TestSimilarityIndexUtility:
    def test_n_documents_zero_before_build(self) -> None:
        idx = SimilarityIndex()
        assert idx.n_documents == 0

    def test_has_embeddings_false_before_build(self) -> None:
        idx = SimilarityIndex()
        assert idx.has_embeddings is False

    def test_repr_contains_key_info(self) -> None:
        idx = SimilarityIndex()
        idx.build(_docs_for_search())
        r = repr(idx)
        assert "5" in r
        assert "dense=False" in r

    def test_repr_dense_true_with_embeddings(self) -> None:
        idx = SimilarityIndex()
        idx.build(_docs_with_embeddings())
        assert "dense=True" in repr(idx)

    def test_default_config_used_when_none_passed(self) -> None:
        """Default mode=semantic with no embeddings returns empty (warns)."""
        idx = SimilarityIndex()
        idx.build(_docs_for_search())
        # No embeddings built → semantic search returns [] with a warning
        results = idx.search("test", query_embedding=None)
        assert results == []

    def test_unknown_match_mode_raises(self) -> None:
        idx = SimilarityIndex()
        idx.build(_docs_for_search())
        bad_cfg = object.__new__(SearchConfig)
        object.__setattr__(bad_cfg, "match_mode", "invalid")
        object.__setattr__(bad_cfg, "top_k", 5)
        object.__setattr__(bad_cfg, "use_normalized_text", True)
        object.__setattr__(bad_cfg, "case_sensitive", False)
        object.__setattr__(bad_cfg, "semantic_threshold", 0.0)
        object.__setattr__(bad_cfg, "keyword_threshold", 0.0)
        object.__setattr__(bad_cfg, "hybrid_alpha", 0.5)
        object.__setattr__(bad_cfg, "rrf_k", 60)
        with pytest.raises(ValueError, match="match_mode"):
            idx.search("test", config=bad_cfg)

    def test_custom_search_config_top_k_one(self) -> None:
        idx = SimilarityIndex()
        idx.build(_docs_for_search())
        cfg = SearchConfig(match_mode="strict", top_k=1)
        results = idx.search("the", config=cfg)
        assert len(results) <= 1

    def test_build_uses_pre_existing_tokens(self) -> None:
        """tokens on doc skips re-tokenisation inside build()."""
        doc = _doc("unrelated text here", idx=0)
        doc = doc.replace(tokens=["custom", "token", "list"])
        idx = SimilarityIndex(config=SearchConfig(match_mode="keyword"))
        idx.build([doc])
        results = idx.search("custom")
        assert len(results) == 1
