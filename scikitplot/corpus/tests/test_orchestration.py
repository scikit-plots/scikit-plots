"""
Tests for the new corpus orchestration modules.

Coverage:
- ``_chunker_bridge.py`` — adapter bridge for new chunkers
- ``_normalizers/_text_normalizer.py`` — text normalisation
- ``_enrichers/_nlp_enricher.py`` — NLP enrichment
- ``_similarity.py`` — multi-mode search
- ``_adapters.py`` — LangChain/MCP/RAG adapters
- ``_corpus_builder.py`` — unified orchestration API

All external NLP libraries are mocked.  Zero optional deps required.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =====================================================================
# Fixtures: minimal CorpusDocument stand-in for testing
# =====================================================================


@dataclass
class _MockDoc:
    """Minimal CorpusDocument-compatible mock."""

    doc_id: str = "abc123"
    text: str = "Hello world."
    normalized_text: str | None = None
    source_file: str = "test.txt"
    source_type: str = "book"
    source_title: str | None = None
    source_author: str | None = None
    source_date: str | None = None
    collection_id: str | None = None
    url: str | None = None
    doi: str | None = None
    isbn: str | None = None
    chunk_index: int = 0
    section_type: str = "text"
    chunking_strategy: str = "sentence"
    language: str | None = "en"
    page_number: int | None = None
    paragraph_index: int | None = None
    line_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    parent_doc_id: str | None = None
    act: int | None = None
    scene_number: int | None = None
    timecode_start: float | None = None
    timecode_end: float | None = None
    confidence: float | None = None
    ocr_engine: str | None = None
    bbox: tuple | None = None
    tokens: list[str] | None = None
    lemmas: list[str] | None = None
    stems: list[str] | None = None
    keywords: list[str] | None = None
    embedding: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def replace(self, **kwargs: Any) -> "_MockDoc":
        import copy
        new = copy.copy(self)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new


@pytest.fixture
def sample_doc() -> _MockDoc:
    return _MockDoc(
        doc_id="d001",
        text="The  ﬁrst  compu-\nter  was  huge.  It occupied a room.",
        source_file="history.txt",
        chunk_index=0,
    )


@pytest.fixture
def sample_docs() -> list[_MockDoc]:
    return [
        _MockDoc(
            doc_id=f"d{i:03d}",
            text=text,
            source_file="corpus.txt",
            chunk_index=i,
        )
        for i, text in enumerate([
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "The quick brown fox jumps over the lazy dog.",
            "A rose by any other name would smell as sweet.",
            "It was the best of times, it was the worst of times.",
        ])
    ]


# =====================================================================
# Tests: TextNormalizer
# =====================================================================


class TestNormalizerConfig:
    def test_default_config_valid(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            NormalizerConfig,
        )
        cfg = NormalizerConfig()
        assert cfg.unicode_form == "NFKC"
        assert cfg.expand_ligatures is True
        assert cfg.fix_hyphenation is True
        assert cfg.collapse_whitespace is True
        assert cfg.lowercase is False
        assert cfg.min_length == 1

    def test_invalid_unicode_form_raises(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            NormalizerConfig,
        )
        with pytest.raises(ValueError, match="unicode_form"):
            NormalizerConfig(unicode_form="INVALID")

    def test_negative_min_length_raises(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            NormalizerConfig,
        )
        with pytest.raises(ValueError, match="min_length"):
            NormalizerConfig(min_length=-1)


class TestNormalizeText:
    def test_ligature_expansion(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            normalize_text,
        )
        assert "fi" in normalize_text("ﬁnd")

    def test_hyphenation_fix(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            normalize_text,
        )
        result = normalize_text("compu-\nter")
        assert "computer" in result

    def test_whitespace_collapse(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            normalize_text,
        )
        result = normalize_text("hello    world")
        assert result == "hello world"

    def test_lowercase(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            NormalizerConfig,
            normalize_text,
        )
        cfg = NormalizerConfig(lowercase=True)
        assert normalize_text("HELLO", config=cfg) == "hello"

    def test_min_length_returns_none(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            NormalizerConfig,
            normalize_text,
        )
        cfg = NormalizerConfig(min_length=100)
        assert normalize_text("short", config=cfg) is None

    def test_custom_pipeline(self):
        from scikitplot.corpus._normalizers._text_normalizer import (
            NormalizerConfig,
            normalize_text,
        )
        cfg = NormalizerConfig(
            custom_pipeline=(lambda s: s.replace("x", "y"),)
        )
        assert normalize_text("fox", config=cfg) == "foy"


class TestTextNormalizerComponent:
    def test_normalize_documents(self, sample_doc):
        from scikitplot.corpus._normalizers._text_normalizer import (
            TextNormalizer,
        )
        norm = TextNormalizer()
        result = norm.normalize_documents([sample_doc])
        assert len(result) == 1
        assert result[0].normalized_text is not None
        assert "computer" in result[0].normalized_text

    def test_skip_already_normalised(self, sample_doc):
        from scikitplot.corpus._normalizers._text_normalizer import (
            TextNormalizer,
        )
        doc = sample_doc.replace(normalized_text="already done")
        norm = TextNormalizer()
        result = norm.normalize_documents([doc])
        assert result[0].normalized_text == "already done"

    def test_overwrite_normalised(self, sample_doc):
        from scikitplot.corpus._normalizers._text_normalizer import (
            TextNormalizer,
        )
        doc = sample_doc.replace(normalized_text="old")
        norm = TextNormalizer()
        result = norm.normalize_documents([doc], overwrite=True)
        assert result[0].normalized_text != "old"


# =====================================================================
# Tests: NLPEnricher
# =====================================================================


class TestEnricherConfig:
    def test_default_config_valid(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            EnricherConfig,
        )
        cfg = EnricherConfig()
        assert cfg.tokenizer == "simple"
        assert cfg.keyword_extractor == "frequency"

    def test_invalid_tokenizer_raises(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            EnricherConfig,
        )
        with pytest.raises(ValueError, match="tokenizer"):
            EnricherConfig(tokenizer="invalid")

    def test_invalid_stemmer_raises(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            EnricherConfig,
        )
        with pytest.raises(ValueError, match="stemmer"):
            EnricherConfig(stemmer="invalid")

    def test_invalid_max_keywords_raises(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            EnricherConfig,
        )
        with pytest.raises(ValueError, match="max_keywords"):
            EnricherConfig(max_keywords=0)


class TestNLPEnricher:
    def test_simple_tokenisation(self, sample_doc):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            NLPEnricher,
        )
        enricher = NLPEnricher()
        result = enricher.enrich_documents([sample_doc])
        assert result[0].tokens is not None
        assert len(result[0].tokens) > 0

    def test_stopword_removal(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            NLPEnricher,
        )
        doc = _MockDoc(text="The quick brown fox")
        enricher = NLPEnricher()
        result = enricher.enrich_documents([doc])
        tokens = result[0].tokens
        assert "the" not in tokens
        assert "quick" in tokens

    def test_frequency_keywords(self, sample_doc):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            NLPEnricher,
        )
        enricher = NLPEnricher()
        result = enricher.enrich_documents([sample_doc])
        assert result[0].keywords is not None

    def test_skip_already_enriched(self, sample_doc):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            NLPEnricher,
        )
        doc = sample_doc.replace(tokens=["pre", "existing"])
        enricher = NLPEnricher()
        result = enricher.enrich_documents([doc])
        assert result[0].tokens == ["pre", "existing"]

    def test_overwrite_enriched(self, sample_doc):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            NLPEnricher,
        )
        doc = sample_doc.replace(tokens=["old"])
        enricher = NLPEnricher()
        result = enricher.enrich_documents([doc], overwrite=True)
        assert result[0].tokens != ["old"]

    def test_no_keywords_when_disabled(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            EnricherConfig,
            NLPEnricher,
        )
        cfg = EnricherConfig(keyword_extractor=None)
        doc = _MockDoc(text="Hello world foo bar")
        enricher = NLPEnricher(config=cfg)
        result = enricher.enrich_documents([doc])
        assert result[0].keywords is None

    def test_uses_normalized_text(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (
            NLPEnricher,
        )
        doc = _MockDoc(
            text="RAW TEXT",
            normalized_text="clean text here",
        )
        enricher = NLPEnricher()
        result = enricher.enrich_documents([doc])
        # Should tokenise "clean text here" not "RAW TEXT"
        assert "clean" in result[0].tokens


# =====================================================================
# Tests: SimilarityIndex
# =====================================================================


class TestSearchConfig:
    def test_default_config(self):
        from scikitplot.corpus._similarity import SearchConfig
        cfg = SearchConfig()
        assert cfg.top_k == 10
        assert cfg.match_mode == "semantic"

    def test_invalid_match_mode(self):
        from scikitplot.corpus._similarity import SearchConfig
        with pytest.raises(ValueError, match="match_mode"):
            SearchConfig(match_mode="invalid")

    def test_invalid_top_k(self):
        from scikitplot.corpus._similarity import SearchConfig
        with pytest.raises(ValueError, match="top_k"):
            SearchConfig(top_k=0)


class TestSimilarityIndex:
    def test_build_empty_raises(self):
        from scikitplot.corpus._similarity import SimilarityIndex
        idx = SimilarityIndex()
        with pytest.raises(ValueError, match="empty"):
            idx.build([])

    def test_strict_search(self, sample_docs):
        from scikitplot.corpus._similarity import (
            SearchConfig,
            SimilarityIndex,
        )
        idx = SimilarityIndex(
            config=SearchConfig(match_mode="strict")
        )
        idx.build(sample_docs)
        results = idx.search("glitters")
        assert len(results) >= 1
        assert "glitters" in results[0].doc.text

    def test_strict_case_insensitive(self, sample_docs):
        from scikitplot.corpus._similarity import (
            SearchConfig,
            SimilarityIndex,
        )
        idx = SimilarityIndex(
            config=SearchConfig(match_mode="strict")
        )
        idx.build(sample_docs)
        results = idx.search("TO BE")
        assert len(results) >= 1

    def test_keyword_search(self, sample_docs):
        from scikitplot.corpus._similarity import (
            SearchConfig,
            SimilarityIndex,
        )
        idx = SimilarityIndex(
            config=SearchConfig(match_mode="keyword")
        )
        idx.build(sample_docs)
        results = idx.search("rose sweet name")
        assert len(results) >= 1
        # The rose doc should be highly ranked
        assert any("rose" in r.doc.text for r in results)

    def test_keyword_empty_query(self, sample_docs):
        from scikitplot.corpus._similarity import (
            SearchConfig,
            SimilarityIndex,
        )
        idx = SimilarityIndex(
            config=SearchConfig(match_mode="keyword")
        )
        idx.build(sample_docs)
        results = idx.search("")
        assert len(results) == 0

    def test_n_documents(self, sample_docs):
        from scikitplot.corpus._similarity import SimilarityIndex
        idx = SimilarityIndex()
        idx.build(sample_docs)
        assert idx.n_documents == 5

    def test_has_embeddings_false(self, sample_docs):
        from scikitplot.corpus._similarity import SimilarityIndex
        idx = SimilarityIndex()
        idx.build(sample_docs)
        assert idx.has_embeddings is False


class TestBM25Index:
    def test_basic_ranking(self):
        from scikitplot.corpus._similarity._similarity import _BM25Index
        idx = _BM25Index()
        idx.build([
            ["cat", "sat", "mat"],
            ["dog", "ran", "park"],
            ["cat", "dog", "friends"],
        ])
        results = idx.query(["cat"])
        assert len(results) > 0
        # First result should be doc 0 or 2 (both have "cat")
        top_indices = [r[0] for r in results]
        assert 0 in top_indices or 2 in top_indices


# =====================================================================
# Tests: Adapters
# =====================================================================


class TestLangChainAdapter:
    def test_to_langchain_documents(self, sample_docs):
        from scikitplot.corpus._adapters import to_langchain_documents
        result = to_langchain_documents(sample_docs)
        assert len(result) == 5
        # Without langchain installed, returns dicts
        first = result[0]
        if isinstance(first, dict):
            assert "page_content" in first
            assert "metadata" in first
        else:
            assert hasattr(first, "page_content")

    def test_metadata_includes_doc_id(self, sample_docs):
        from scikitplot.corpus._adapters import to_langchain_documents
        result = to_langchain_documents(sample_docs)
        first = result[0]
        meta = first["metadata"] if isinstance(first, dict) else first.metadata
        assert "doc_id" in meta

    def test_uses_normalized_text(self):
        from scikitplot.corpus._adapters import to_langchain_documents
        doc = _MockDoc(text="raw", normalized_text="clean")
        result = to_langchain_documents([doc])
        first = result[0]
        content = (
            first["page_content"]
            if isinstance(first, dict)
            else first.page_content
        )
        assert content == "clean"


class TestLangGraphAdapter:
    def test_state_structure(self, sample_docs):
        from scikitplot.corpus._adapters import to_langgraph_state
        state = to_langgraph_state(
            sample_docs, query="test", match_mode="hybrid"
        )
        assert "documents" in state
        assert "query" in state
        assert state["n_results"] == 5


class TestMCPAdapter:
    def test_to_mcp_resources(self, sample_docs):
        from scikitplot.corpus._adapters import to_mcp_resources
        resources = to_mcp_resources(sample_docs)
        assert len(resources) == 5
        assert resources[0]["uri"].startswith("corpus://")
        assert "text" in resources[0]
        assert "mimeType" in resources[0]

    def test_to_mcp_tool_result(self, sample_docs):
        from scikitplot.corpus._adapters import to_mcp_tool_result
        result = to_mcp_tool_result(sample_docs)
        assert "content" in result
        assert "isError" in result
        assert result["isError"] is False
        assert len(result["content"]) == 5

    def test_mcp_tool_result_content_structure(self, sample_docs):
        from scikitplot.corpus._adapters import to_mcp_tool_result
        result = to_mcp_tool_result(sample_docs)
        first = result["content"][0]
        assert first["type"] == "text"
        assert "annotations" in first


class TestHuggingFaceAdapter:
    def test_to_huggingface_returns_columns(self, sample_docs):
        from scikitplot.corpus._adapters import to_huggingface_dataset
        result = to_huggingface_dataset(sample_docs)
        # Without HF datasets, returns column dict
        if isinstance(result, dict):
            assert "text" in result
            assert "doc_id" in result
            assert len(result["text"]) == 5


class TestRAGAdapter:
    def test_to_rag_tuples(self, sample_docs):
        from scikitplot.corpus._adapters import to_rag_tuples
        tuples = to_rag_tuples(sample_docs)
        assert len(tuples) == 5
        text, meta, emb = tuples[0]
        assert isinstance(text, str)
        assert isinstance(meta, dict)
        assert emb is None  # No embedding set

    def test_rag_tuple_with_embedding(self):
        from scikitplot.corpus._adapters import to_rag_tuples
        doc = _MockDoc(embedding=[0.1, 0.2, 0.3])
        tuples = to_rag_tuples([doc])
        _, _, emb = tuples[0]
        assert emb == [0.1, 0.2, 0.3]


class TestJSONLAdapter:
    def test_to_jsonl(self, sample_docs):
        from scikitplot.corpus._adapters import to_jsonl
        lines = list(to_jsonl(sample_docs))
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj
            assert "doc_id" in obj


# =====================================================================
# Tests: ChunkerBridge
# =====================================================================


class TestChunkerBridge:
    def test_bridge_chunker_passes_through_legacy(self):
        from scikitplot.corpus._chunkers._chunker_bridge import bridge_chunker

        class LegacyChunker:
            strategy = "sentence"
            def chunk(self, text, metadata=None):
                return [(0, text)]

        c = LegacyChunker()
        assert bridge_chunker(c) is c

    def test_bridge_chunker_wraps_new_style(self):
        from scikitplot.corpus._chunkers._chunker_bridge import bridge_chunker

        @dataclass
        class FakeChunk:
            text: str
            char_start: int | None = None

        @dataclass
        class FakeResult:
            chunks: list

        class SentenceChunker:
            def chunk(self, text, doc_id=None, extra_metadata=None):
                return FakeResult(
                    chunks=[FakeChunk(text=text, char_start=0)]
                )

        c = SentenceChunker()
        bridged = bridge_chunker(c)
        assert hasattr(bridged, "strategy")
        result = bridged.chunk("Hello world.", metadata={})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0][1] == "Hello world."

    def test_bridge_repr(self):
        from scikitplot.corpus._chunkers._chunker_bridge import (
            SentenceChunkerBridge,
        )
        mock = MagicMock()
        bridge = SentenceChunkerBridge(mock)
        assert "SentenceChunkerBridge" in repr(bridge)

    def test_forward_cursor_fallback(self):
        from scikitplot.corpus._chunkers._chunker_bridge import ChunkerBridge

        @dataclass
        class FakeChunk:
            text: str
            char_start: int | None = None  # None triggers fallback

        @dataclass
        class FakeResult:
            chunks: list

        text = "Hello world. Goodbye world."
        result = FakeResult(chunks=[
            FakeChunk(text="Hello world."),
            FakeChunk(text="Goodbye world."),
        ])
        pairs = ChunkerBridge._to_tuples(text, result)
        assert pairs[0] == (0, "Hello world.")
        assert pairs[1][0] > 0  # Found via str.find


# =====================================================================
# Tests: CorpusBuilder
# =====================================================================


class TestBuilderConfig:
    def test_default_config(self):
        from scikitplot.corpus._corpus_builder import BuilderConfig
        cfg = BuilderConfig()
        assert cfg.chunker == "sentence"
        assert cfg.normalize is True
        assert cfg.embed is False

    def test_custom_config(self):
        from scikitplot.corpus._corpus_builder import BuilderConfig
        cfg = BuilderConfig(
            chunker="paragraph",
            embed=True,
            embedding_model="all-MiniLM-L6-v2",
        )
        assert cfg.chunker == "paragraph"
        assert cfg.embed is True


class TestBuildResult:
    def test_summary(self):
        from scikitplot.corpus._corpus_builder import BuildResult
        result = BuildResult(
            documents=[_MockDoc()],
            n_sources=1,
            n_raw=5,
            n_filtered=2,
        )
        summary = result.summary()
        assert "Sources:" in summary
        assert "Documents:  1" in summary

    def test_success_rate(self):
        from scikitplot.corpus._corpus_builder import BuildResult
        result = BuildResult(
            n_sources=4,
            errors=[("a.txt", ValueError("fail"))],
        )
        assert result.success_rate == 0.75

    def test_success_rate_no_sources(self):
        from scikitplot.corpus._corpus_builder import BuildResult
        result = BuildResult(n_sources=0)
        assert result.success_rate == 1.0


class TestCorpusBuilderInit:
    def test_default_init(self):
        from scikitplot.corpus._corpus_builder import CorpusBuilder
        builder = CorpusBuilder()
        assert builder.config.chunker == "sentence"

    def test_repr(self):
        from scikitplot.corpus._corpus_builder import CorpusBuilder
        builder = CorpusBuilder()
        assert "CorpusBuilder" in repr(builder)

    def test_no_result_raises(self):
        from scikitplot.corpus._corpus_builder import CorpusBuilder
        builder = CorpusBuilder()
        with pytest.raises(RuntimeError, match="No corpus built"):
            builder.to_langchain()


class TestCorpusBuilderExpandSources:
    def test_expand_url(self):
        from scikitplot.corpus._corpus_builder import CorpusBuilder
        expanded = CorpusBuilder._expand_sources(
            ["https://example.com/page"]
        )
        assert expanded == ["https://example.com/page"]

    def test_expand_missing_path(self, tmp_path):
        from scikitplot.corpus._corpus_builder import CorpusBuilder
        expanded = CorpusBuilder._expand_sources(
            [str(tmp_path / "nonexistent.txt")]
        )
        assert len(expanded) == 0

    def test_expand_file(self, tmp_path):
        from scikitplot.corpus._corpus_builder import CorpusBuilder
        f = tmp_path / "test.txt"
        f.write_text("hello")
        expanded = CorpusBuilder._expand_sources([str(f)])
        assert len(expanded) == 1

    def test_expand_directory(self, tmp_path):
        from scikitplot.corpus._corpus_builder import CorpusBuilder
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        (tmp_path / ".hidden").write_text("skip")
        expanded = CorpusBuilder._expand_sources([str(tmp_path)])
        assert len(expanded) == 2  # .hidden excluded


class TestCorpusBuilderInvalidChunker:
    def test_unknown_chunker_raises(self):
        from scikitplot.corpus._corpus_builder import (
            BuilderConfig,
            CorpusBuilder,
        )
        builder = CorpusBuilder(
            BuilderConfig(chunker="nonexistent")
        )
        with pytest.raises(ValueError, match="Unknown chunker"):
            builder._get_chunker()


# =====================================================================
# Tests: MCPCorpusServer
# =====================================================================


class TestMCPCorpusServer:
    def test_list_tools(self):
        from scikitplot.corpus._adapters import MCPCorpusServer
        mock_index = MagicMock()
        mock_index.n_documents = 10
        server = MCPCorpusServer(mock_index)
        tools = server.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "corpus_search"
        assert "inputSchema" in tools[0]

    def test_repr(self):
        from scikitplot.corpus._adapters import MCPCorpusServer
        mock_index = MagicMock()
        mock_index.n_documents = 5
        server = MCPCorpusServer(mock_index, server_name="test")
        assert "test" in repr(server)

    def test_handle_resource_not_found(self):
        from scikitplot.corpus._adapters import MCPCorpusServer
        mock_index = MagicMock()
        mock_index._documents = []
        server = MCPCorpusServer(mock_index)
        assert server.handle_resource("nonexistent") is None

    def test_handle_resource_found(self, sample_doc):
        from scikitplot.corpus._adapters import MCPCorpusServer
        mock_index = MagicMock()
        mock_index._documents = [sample_doc]
        server = MCPCorpusServer(mock_index)
        result = server.handle_resource("d001")
        assert result is not None
        assert result["uri"].startswith("corpus://")


class TestLangChainCorpusRetriever:
    # @pytest.mark.xfail(reason="Fatal error: LangChainCorpusRetriever.")
    def test_get_relevant_documents(self, sample_docs):
        from scikitplot.corpus._adapters import LangChainCorpusRetriever
        from scikitplot.corpus._similarity import (
            SearchConfig,
            SearchResult,
            SimilarityIndex,
        )

        idx = SimilarityIndex(
            config=SearchConfig(match_mode="keyword")
        )
        idx.build(sample_docs)
        retriever = LangChainCorpusRetriever(
            index=idx,
            config=SearchConfig(match_mode="keyword", top_k=3),
        )
        results = retriever.get_relevant_documents("fox jumps dog")
        assert len(results) > 0

    def test_repr(self):
        from scikitplot.corpus._adapters import LangChainCorpusRetriever
        mock_idx = MagicMock()
        retriever = LangChainCorpusRetriever(mock_idx)
        assert "LangChainCorpusRetriever" in repr(retriever)
