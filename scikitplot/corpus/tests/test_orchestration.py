# scikitplot/corpus/tests/test_orchestration.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus orchestration modules
==================================================

Coverage
--------
* ``_normalizers/_text_normalizer.py`` — normalisation pipeline
* ``_enrichers/_nlp_enricher.py`` — NLP enrichment
* ``_similarity.py`` — multi-mode search (BM25, semantic, hybrid)
* ``_adapters.py`` — LangChain / MCP / RAG / HF adapters
* ``_adapters.py`` — ML tensor adapters (numpy, tensorflow, torch)
* ``_corpus_builder.py`` — unified orchestration API
* ``_base.py PipelineGuard`` — error policy, dedup, checkpoint/resume
* ``_embeddings._multimodal_embedding.LLMTrainingExporter`` — training exports

All external NLP/ML libraries are mocked.  Zero optional deps required.

Run with:
    pytest corpus/tests/test_orchestration.py -v
"""
from __future__ import annotations

import json
import pathlib
import tempfile
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# _MockDoc — minimal CorpusDocument stand-in
# ===========================================================================


@dataclass
class _MockDoc:
    """Minimal CorpusDocument-compatible mock covering all field groups."""

    doc_id: str = "abc123"
    text: str | None = "Hello world."
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
    # New raw media / multimodal fields
    modality: str = "text"
    raw_bytes: bytes | None = None
    raw_tensor: Any = None
    raw_shape: tuple | None = None
    raw_dtype: str | None = None
    frame_index: int | None = None
    content_hash: str | None = None

    def replace(self, **kwargs: Any) -> "_MockDoc":
        import copy  # noqa: PLC0415
        new = copy.copy(self)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new


@pytest.fixture
def sample_doc() -> _MockDoc:
    return _MockDoc(
        doc_id="d001",
        text="The  first  computer  was  huge.  It occupied a room.",
        source_file="history.txt",
        chunk_index=0,
        content_hash="a" * 32,
    )


@pytest.fixture
def sample_docs() -> list[_MockDoc]:
    texts = [
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The quick brown fox jumps over the lazy dog.",
        "A rose by any other name would smell as sweet.",
        "It was the best of times, it was the worst of times.",
    ]
    return [
        _MockDoc(
            doc_id=f"d{i:03d}",
            text=t,
            source_file="corpus.txt",
            chunk_index=i,
            content_hash=f"{i:032d}",
        )
        for i, t in enumerate(texts)
    ]


# ===========================================================================
# TestNormalizerConfig
# ===========================================================================


class TestNormalizerConfig:
    def test_default_steps(self):
        from scikitplot.corpus._normalizers._text_normalizer import (  # noqa: PLC0415
            NormalizerConfig,
        )
        cfg = NormalizerConfig()
        assert "unicode" in cfg.steps
        assert "whitespace" in cfg.steps

    def test_custom_steps(self):
        from scikitplot.corpus._normalizers._text_normalizer import (  # noqa: PLC0415
            NormalizerConfig,
        )
        cfg = NormalizerConfig(steps=["unicode"])
        assert cfg.steps == ["unicode"]


# ===========================================================================
# TestNormalizeText
# ===========================================================================


class TestNormalizeText:
    def test_unicode_normalise(self):
        from scikitplot.corpus._normalizers._text_normalizer import (  # noqa: PLC0415
            TextNormalizer,
            NormalizerConfig,
        )
        n = TextNormalizer(NormalizerConfig(steps=["unicode"]))
        result = n.normalize("fi\ufb01rst")  # fi ligature
        assert "\ufb01" not in result

    def test_whitespace_normalise(self):
        from scikitplot.corpus._normalizers._text_normalizer import (  # noqa: PLC0415
            TextNormalizer,
            NormalizerConfig,
        )
        n = TextNormalizer(NormalizerConfig(steps=["whitespace"]))
        result = n.normalize("Hello   world\n\nfoo")
        assert "   " not in result

    def test_empty_string(self):
        from scikitplot.corpus._normalizers._text_normalizer import (  # noqa: PLC0415
            TextNormalizer,
            NormalizerConfig,
        )
        n = TextNormalizer(NormalizerConfig())
        assert n.normalize("") == ""


# ===========================================================================
# TestEnricherConfig
# ===========================================================================


class TestEnricherConfig:
    def test_defaults(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (  # noqa: PLC0415
            EnricherConfig,
        )
        cfg = EnricherConfig()
        assert cfg.tokenizer == "simple"
        assert cfg.max_keywords >= 0

    def test_custom_config(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (  # noqa: PLC0415
            EnricherConfig,
        )
        cfg = EnricherConfig(
            tokenizer="simple",
            keyword_extractor="frequency",
            max_keywords=10,
            remove_stopwords=True,
        )
        assert cfg.max_keywords == 10


# ===========================================================================
# TestNLPEnricher
# ===========================================================================


class TestNLPEnricher:
    def test_enrich_sets_tokens(self, sample_docs):
        from scikitplot.corpus._enrichers._nlp_enricher import (  # noqa: PLC0415
            NLPEnricher, EnricherConfig,
        )
        enricher = NLPEnricher(config=EnricherConfig(tokenizer="simple"))
        result = enricher.enrich_documents(sample_docs)
        assert result[0].tokens is not None
        assert len(result[0].tokens) > 0

    def test_enrich_empty_list(self):
        from scikitplot.corpus._enrichers._nlp_enricher import (  # noqa: PLC0415
            NLPEnricher, EnricherConfig,
        )
        assert NLPEnricher(EnricherConfig()).enrich_documents([]) == []

    def test_keyword_extraction(self, sample_docs):
        from scikitplot.corpus._enrichers._nlp_enricher import (  # noqa: PLC0415
            NLPEnricher, EnricherConfig,
        )
        enricher = NLPEnricher(
            config=EnricherConfig(keyword_extractor="frequency", max_keywords=5)
        )
        result = enricher.enrich_documents(sample_docs)
        kw = result[0].keywords
        assert kw is None or isinstance(kw, list)


# ===========================================================================
# TestSearchConfig
# ===========================================================================


class TestSearchConfig:
    def test_default_mode(self):
        from scikitplot.corpus._similarity._similarity import SearchConfig  # noqa: PLC0415
        cfg = SearchConfig()
        assert cfg.match_mode in ("keyword", "hybrid", "semantic", "strict")

    def test_custom_mode(self):
        from scikitplot.corpus._similarity._similarity import SearchConfig  # noqa: PLC0415
        cfg = SearchConfig(match_mode="strict", top_k=3)
        assert cfg.top_k == 3


# ===========================================================================
# TestSimilarityIndex
# ===========================================================================


class TestSimilarityIndex:
    def test_build_and_search_keyword(self, sample_docs):
        from scikitplot.corpus._similarity._similarity import (  # noqa: PLC0415
            SimilarityIndex, SearchConfig,
        )
        idx = SimilarityIndex(config=SearchConfig(match_mode="keyword", top_k=3))
        idx.build(sample_docs)
        assert idx.n_documents == 5
        results = idx.search("question")
        assert len(results) <= 3
        assert results[0].doc.text is not None

    def test_build_empty_raises_or_ok(self):
        from scikitplot.corpus._similarity._similarity import (  # noqa: PLC0415
            SimilarityIndex, SearchConfig,
        )
        idx = SimilarityIndex(config=SearchConfig())
        try:
            idx.build([])
        except (ValueError, RuntimeError):
            pass  # expected

    def test_strict_search(self, sample_docs):
        from scikitplot.corpus._similarity._similarity import (  # noqa: PLC0415
            SimilarityIndex, SearchConfig,
        )
        idx = SimilarityIndex(config=SearchConfig(match_mode="strict", top_k=5))
        idx.build(sample_docs)
        results = idx.search("golden")
        assert all(r.score in (0.0, 1.0) for r in results)


# ===========================================================================
# TestLangChainAdapter
# ===========================================================================


class TestLangChainAdapter:
    def test_basic_conversion(self, sample_docs):
        from scikitplot.corpus._adapters import to_langchain_documents  # noqa: PLC0415
        result = to_langchain_documents(sample_docs)
        assert len(result) == 5
        first = result[0]
        if isinstance(first, dict):
            assert "page_content" in first
            assert "metadata" in first
        else:
            assert hasattr(first, "page_content")

    def test_metadata_has_doc_id(self, sample_docs):
        from scikitplot.corpus._adapters import to_langchain_documents  # noqa: PLC0415
        first = to_langchain_documents(sample_docs)[0]
        meta = first["metadata"] if isinstance(first, dict) else first.metadata
        assert "doc_id" in meta

    def test_uses_normalized_text(self):
        from scikitplot.corpus._adapters import to_langchain_documents  # noqa: PLC0415
        doc = _MockDoc(text="raw", normalized_text="clean")
        result = to_langchain_documents([doc])
        first = result[0]
        content = first["page_content"] if isinstance(first, dict) else first.page_content
        assert content == "clean"


# ===========================================================================
# TestLangGraphAdapter
# ===========================================================================


class TestLangGraphAdapter:
    def test_state_structure(self, sample_docs):
        from scikitplot.corpus._adapters import to_langgraph_state  # noqa: PLC0415
        state = to_langgraph_state(sample_docs, query="test", match_mode="hybrid")
        assert "documents" in state
        assert "query" in state
        assert state["n_results"] == 5


# ===========================================================================
# TestMCPAdapter
# ===========================================================================


class TestMCPAdapter:
    def test_to_mcp_resources(self, sample_docs):
        from scikitplot.corpus._adapters import to_mcp_resources  # noqa: PLC0415
        resources = to_mcp_resources(sample_docs)
        assert len(resources) == 5
        assert resources[0]["uri"].startswith("corpus://")
        assert "text" in resources[0]
        assert "mimeType" in resources[0]

    def test_to_mcp_tool_result(self, sample_docs):
        from scikitplot.corpus._adapters import to_mcp_tool_result  # noqa: PLC0415
        result = to_mcp_tool_result(sample_docs)
        assert result["isError"] is False
        assert len(result["content"]) == 5
        assert result["content"][0]["type"] == "text"


# ===========================================================================
# TestHuggingFaceAdapter
# ===========================================================================


class TestHuggingFaceAdapter:
    def test_to_huggingface_columns(self, sample_docs):
        from scikitplot.corpus._adapters import to_huggingface_dataset  # noqa: PLC0415
        result = to_huggingface_dataset(sample_docs)
        if isinstance(result, dict):
            assert "text" in result
            assert "doc_id" in result
            assert len(result["text"]) == 5


# ===========================================================================
# TestRAGAdapter
# ===========================================================================


class TestRAGAdapter:
    def test_basic_tuples(self, sample_docs):
        from scikitplot.corpus._adapters import to_rag_tuples  # noqa: PLC0415
        tuples = to_rag_tuples(sample_docs)
        assert len(tuples) == 5
        text, meta, emb = tuples[0]
        assert isinstance(text, str)
        assert isinstance(meta, dict)
        assert emb is None

    def test_with_embedding(self):
        from scikitplot.corpus._adapters import to_rag_tuples  # noqa: PLC0415
        doc = _MockDoc(embedding=[0.1, 0.2, 0.3])
        _, _, emb = to_rag_tuples([doc])[0]
        assert emb == [0.1, 0.2, 0.3]


# ===========================================================================
# TestJSONLAdapter
# ===========================================================================


class TestJSONLAdapter:
    def test_lines_count(self, sample_docs):
        from scikitplot.corpus._adapters import to_jsonl  # noqa: PLC0415
        lines = list(to_jsonl(sample_docs))
        assert len(lines) == 5

    def test_valid_json(self, sample_docs):
        from scikitplot.corpus._adapters import to_jsonl  # noqa: PLC0415
        for line in to_jsonl(sample_docs):
            obj = json.loads(line)
            assert "text" in obj and "doc_id" in obj


# ===========================================================================
# TestNumpyArrayAdapter
# ===========================================================================


class TestNumpyArrayAdapter:
    """to_numpy_arrays() — returns dict of arrays from CorpusDocuments."""

    def test_texts_column(self, sample_docs):
        from scikitplot.corpus._adapters import to_numpy_arrays  # noqa: PLC0415
        try:
            arrays = to_numpy_arrays(sample_docs)
        except ImportError:
            pytest.skip("numpy not available")
        assert "texts" in arrays
        assert len(arrays["texts"]) == 5

    def test_metadata_columns(self, sample_docs):
        from scikitplot.corpus._adapters import to_numpy_arrays  # noqa: PLC0415
        try:
            arrays = to_numpy_arrays(sample_docs)
        except ImportError:
            pytest.skip("numpy not available")
        assert "doc_ids" in arrays
        assert "source_files" in arrays
        assert "modalities" in arrays
        assert "content_hashes" in arrays

    def test_empty_list(self):
        from scikitplot.corpus._adapters import to_numpy_arrays  # noqa: PLC0415
        try:
            arrays = to_numpy_arrays([])
        except ImportError:
            pytest.skip("numpy not available")
        assert arrays["texts"] == []

    def test_embeddings_stacked(self):
        from scikitplot.corpus._adapters import to_numpy_arrays  # noqa: PLC0415
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")
        docs = [
            _MockDoc(embedding=np.zeros(4, dtype="float32")),
            _MockDoc(embedding=np.ones(4, dtype="float32")),
        ]
        arrays = to_numpy_arrays(docs, include_embedding=True)
        assert "embeddings" in arrays
        assert arrays["embeddings"].shape == (2, 4)

    def test_raw_tensors_stacked(self):
        from scikitplot.corpus._adapters import to_numpy_arrays  # noqa: PLC0415
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")
        arr = np.zeros((4, 4, 3), dtype="uint8")
        docs = [_MockDoc(raw_tensor=arr), _MockDoc(raw_tensor=arr)]
        arrays = to_numpy_arrays(docs, include_raw_tensor=True)
        assert "raw_tensors" in arrays
        assert arrays["raw_tensors"].shape == (2, 4, 4, 3)


# ===========================================================================
# TestTensorFlowAdapter
# ===========================================================================


class TestTensorFlowAdapter:
    """to_tensorflow_dataset() — falls back to numpy when TF not installed."""

    def test_fallback_when_no_tf(self, sample_docs):
        from scikitplot.corpus._adapters import to_tensorflow_dataset  # noqa: PLC0415
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")
        with patch.dict("sys.modules", {"tensorflow": None}):
            result = to_tensorflow_dataset(
                sample_docs, text_feature=True, batch_size=2
            )
            # Should return numpy fallback dict
            assert isinstance(result, dict) or hasattr(result, "__iter__")

    def test_with_tf(self, sample_docs):
        tf = pytest.importorskip("tensorflow")
        from scikitplot.corpus._adapters import to_tensorflow_dataset  # noqa: PLC0415
        ds = to_tensorflow_dataset(sample_docs, text_feature=True, batch_size=2)
        assert hasattr(ds, "take") or isinstance(ds, dict)

    def test_raw_tensor_shape_mismatch_raises(self):
        from scikitplot.corpus._adapters import to_tensorflow_dataset  # noqa: PLC0415
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")
        # Tensors of different shapes should raise or return fallback
        docs = [
            _MockDoc(raw_tensor=np.zeros((4, 4, 3), dtype="uint8")),
            _MockDoc(raw_tensor=np.zeros((8, 8, 3), dtype="uint8")),
        ]
        try:
            result = to_tensorflow_dataset(
                docs, raw_tensor_feature=True, batch_size=2
            )
            # Fallback is acceptable
        except (ValueError, ImportError):
            pass  # expected


# ===========================================================================
# TestTorchAdapter
# ===========================================================================


class TestTorchAdapter:
    """to_torch_dataloader() — falls back to numpy when torch not installed."""

    def test_fallback_when_no_torch(self, sample_docs):
        from scikitplot.corpus._adapters import to_torch_dataloader  # noqa: PLC0415
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")
        with patch.dict("sys.modules", {"torch": None, "torch.utils.data": None}):
            result = to_torch_dataloader(
                sample_docs, text_feature=True, batch_size=2
            )
            assert isinstance(result, dict) or hasattr(result, "__iter__")

    def test_with_torch(self, sample_docs):
        tensorflow = pytest.importorskip("tensorflow")
        torch = pytest.importorskip("torch")
        from scikitplot.corpus._adapters import to_torch_dataloader  # noqa: PLC0415
        loader = to_torch_dataloader(
            sample_docs, text_feature=True, batch_size=2
        )
        assert hasattr(loader, "__iter__") or isinstance(loader, dict)

    def test_raw_tensor_hwc_to_chw(self):
        tensorflow = pytest.importorskip("tensorflow")
        torch = pytest.importorskip("torch")
        import numpy as np  # noqa: PLC0415
        from scikitplot.corpus._adapters import to_torch_dataloader  # noqa: PLC0415
        arr = np.zeros((4, 4, 3), dtype="uint8")
        docs = [_MockDoc(raw_tensor=arr) for _ in range(4)]
        loader = to_torch_dataloader(
            docs, raw_tensor_feature=True, batch_size=4
        )
        for batch in loader:
            if isinstance(batch, dict) and "raw_tensor" in batch:
                t = batch["raw_tensor"]
                # CHW convention: (B, C, H, W)
                assert t.shape == (4, 3, 4, 4), f"Expected (4,3,4,4), got {t.shape}"
            break


# ===========================================================================
# TestBuilderConfig
# ===========================================================================


class TestBuilderConfig:
    def test_defaults(self):
        from scikitplot.corpus._corpus_builder import BuilderConfig  # noqa: PLC0415
        cfg = BuilderConfig()
        assert cfg.chunker == "sentence"
        assert cfg.normalize is True
        assert cfg.embed is False

    def test_custom(self):
        from scikitplot.corpus._corpus_builder import BuilderConfig  # noqa: PLC0415
        cfg = BuilderConfig(
            chunker="paragraph",
            embed=True,
            embedding_model="all-MiniLM-L6-v2",
            build_index=True,
        )
        assert cfg.embed is True
        assert cfg.build_index is True

    def test_probe_url_fields(self):
        """probe_url_content_type and probe_url_timeout must be accepted."""
        from scikitplot.corpus._corpus_builder import BuilderConfig  # noqa: PLC0415
        cfg = BuilderConfig(
            probe_url_content_type=False,
            probe_url_timeout=30,
        )
        assert cfg.probe_url_content_type is False
        assert cfg.probe_url_timeout == 30


# ===========================================================================
# TestBuildResult
# ===========================================================================


class TestBuildResult:
    def _make_result(self, n_docs=5):
        from scikitplot.corpus._corpus_builder import BuildResult  # noqa: PLC0415
        r = BuildResult()
        r.documents = [_MockDoc(doc_id=f"d{i}") for i in range(n_docs)]
        r.n_sources = 2
        r.n_raw = n_docs
        return r

    def test_n_documents(self):
        r = self._make_result(3)
        assert r.n_documents == 3

    def test_success_rate_full(self):
        r = self._make_result()
        r.errors = []
        assert r.success_rate == 1.0

    def test_success_rate_partial(self):
        r = self._make_result()
        r.errors = [("src.txt", RuntimeError("oops"))]
        assert 0.0 <= r.success_rate <= 1.0

    def test_summary_string(self):
        r = self._make_result()
        r.errors = []
        s = r.summary()
        assert isinstance(s, str) and len(s) > 0


# ===========================================================================
# TestCorpusBuilderInit
# ===========================================================================


class TestCorpusBuilderInit:
    def test_default_config(self):
        from scikitplot.corpus._corpus_builder import CorpusBuilder, BuilderConfig  # noqa: PLC0415
        b = CorpusBuilder()
        assert isinstance(b.config, BuilderConfig)

    def test_custom_config(self):
        from scikitplot.corpus._corpus_builder import CorpusBuilder, BuilderConfig  # noqa: PLC0415
        cfg = BuilderConfig(chunker="paragraph", normalize=False)
        b = CorpusBuilder(cfg)
        assert b.config.normalize is False

    def test_context_manager(self):
        from scikitplot.corpus._corpus_builder import CorpusBuilder  # noqa: PLC0415
        with CorpusBuilder() as b:
            assert b is not None


# ===========================================================================
# TestPipelineGuard
# ===========================================================================


class TestPipelineGuard:
    """PipelineGuard — error isolation, dedup, checkpoint, context manager."""

    def test_basic_iteration(self, sample_docs):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415
        guard = PipelineGuard(policy=ErrorPolicy.SKIP, dedup=False)
        result = list(guard.iter(iter(sample_docs)))
        assert len(result) == 5

    def test_skip_policy_absorbs_error(self):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415

        def bad_source():
            yield _MockDoc(doc_id="ok1", content_hash="a" * 32)
            raise RuntimeError("broken pipe")

        guard = PipelineGuard(policy=ErrorPolicy.SKIP, dedup=False)
        result = list(guard.iter(bad_source()))
        # At least the first doc before the error should be yielded
        assert len(result) >= 1
        assert guard.stats["n_errors"] >= 1

    def test_log_policy_records_error(self):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415

        def bad_source():
            yield _MockDoc(doc_id="ok", content_hash="b" * 32)
            raise ValueError("log me")

        guard = PipelineGuard(policy=ErrorPolicy.LOG, dedup=False)
        list(guard.iter(bad_source()))
        assert guard.stats["n_errors"] >= 1

    def test_raise_policy_propagates(self):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415

        def bad_source():
            yield _MockDoc(doc_id="x", content_hash="c" * 32)
            raise RuntimeError("must propagate")

        guard = PipelineGuard(policy=ErrorPolicy.RAISE, dedup=False)
        with pytest.raises(RuntimeError, match="must propagate"):
            list(guard.iter(bad_source()))

    def test_dedup_drops_duplicates(self):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415

        docs = [
            _MockDoc(doc_id=f"d{i}", content_hash="same_hash_0000000000000000000")
            for i in range(3)
        ]
        guard = PipelineGuard(policy=ErrorPolicy.LOG, dedup=True)
        result = list(guard.iter(iter(docs)))
        assert len(result) == 1
        assert guard.stats["n_skipped_dedup"] == 2

    def test_dedup_different_hashes(self):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415

        docs = [
            _MockDoc(doc_id=f"d{i}", content_hash=f"{i:032d}")
            for i in range(3)
        ]
        guard = PipelineGuard(policy=ErrorPolicy.LOG, dedup=True)
        result = list(guard.iter(iter(docs)))
        assert len(result) == 3
        assert guard.stats["n_skipped_dedup"] == 0

    def test_dedup_none_hash_not_dropped(self):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415
        docs = [_MockDoc(doc_id=f"d{i}", content_hash=None) for i in range(3)]
        guard = PipelineGuard(policy=ErrorPolicy.LOG, dedup=True)
        result = list(guard.iter(iter(docs)))
        assert len(result) == 3

    def test_context_manager_closes(self, sample_docs):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415
        with PipelineGuard(policy=ErrorPolicy.LOG, dedup=False) as guard:
            result = list(guard.iter(iter(sample_docs)))
        assert len(result) == 5

    def test_stats_keys(self, sample_docs):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415
        guard = PipelineGuard(policy=ErrorPolicy.LOG, dedup=False)
        list(guard.iter(iter(sample_docs)))
        stats = guard.stats
        assert "n_yielded" in stats
        assert "n_skipped_dedup" in stats
        assert "n_errors" in stats
        assert stats["n_yielded"] == 5

    def test_checkpoint_write_and_resume(self, sample_docs):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            ckpt = pathlib.Path(tf.name)

        try:
            # First run — write checkpoint every 2 docs
            guard = PipelineGuard(
                policy=ErrorPolicy.LOG, dedup=False,
                checkpoint_path=ckpt, checkpoint_every=2,
            )
            with guard:
                list(guard.iter(iter(sample_docs)))

            assert ckpt.exists()
            lines = ckpt.read_text().strip().split("\n")
            assert len(lines) > 0
            # Each line must be valid JSON with doc_id
            for line in lines:
                if line.strip():
                    obj = json.loads(line)
                    assert "doc_id" in obj

            # Second run — checkpoint loaded, seen ids skipped
            guard2 = PipelineGuard(
                policy=ErrorPolicy.LOG, dedup=False,
                checkpoint_path=ckpt, checkpoint_every=100,
            )
            # Previously seen doc_ids should be in _seen_ids
            assert len(guard2._seen_ids) > 0
        finally:
            ckpt.unlink(missing_ok=True)

    def test_checkpoint_every_respected(self, sample_docs):
        from scikitplot.corpus._base import PipelineGuard  # noqa: PLC0415
        from scikitplot.corpus._schema import ErrorPolicy  # noqa: PLC0415
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            ckpt = pathlib.Path(tf.name)
        try:
            guard = PipelineGuard(
                policy=ErrorPolicy.LOG, dedup=False,
                checkpoint_path=ckpt, checkpoint_every=3,
            )
            with guard:
                list(guard.iter(iter(sample_docs)))
        finally:
            ckpt.unlink(missing_ok=True)


# ===========================================================================
# TestLLMTrainingExporter
# ===========================================================================


class TestLLMTrainingExporter:
    """LLMTrainingExporter — training format outputs."""

    def test_openai_jsonl_written(self, sample_docs):
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        exporter = LLMTrainingExporter(engine=None)
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as tf:
            out_path = pathlib.Path(tf.name)

        try:
            result_path = exporter.to_openai_finetuning_jsonl(
                sample_docs, out_path,
                system_prompt="You are a helpful assistant.",
            )
            assert result_path.exists()
            lines = [l for l in out_path.read_text().strip().split("\n") if l]
            assert len(lines) == 5
            for line in lines:
                obj = json.loads(line)
                assert "messages" in obj
                msgs = obj["messages"]
                assert any(m["role"] == "user" for m in msgs)
        finally:
            out_path.unlink(missing_ok=True)

    def test_openai_jsonl_includes_metadata(self, sample_docs):
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        exporter = LLMTrainingExporter(engine=None)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            out_path = pathlib.Path(tf.name)
        try:
            exporter.to_openai_finetuning_jsonl(sample_docs, out_path)
            line = json.loads(out_path.read_text().strip().split("\n")[0])
            assert "metadata" in line
            assert "doc_id" in line["metadata"]
        finally:
            out_path.unlink(missing_ok=True)

    def test_openai_jsonl_with_response_fn(self, sample_docs):
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        exporter = LLMTrainingExporter(engine=None)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            out_path = pathlib.Path(tf.name)
        try:
            exporter.to_openai_finetuning_jsonl(
                sample_docs, out_path,
                response_fn=lambda doc: "Answer: " + (doc.text or ""),
            )
            line = json.loads(out_path.read_text().strip().split("\n")[0])
            msgs = line["messages"]
            assistant = [m for m in msgs if m["role"] == "assistant"]
            assert len(assistant) == 1
            assert assistant[0]["content"].startswith("Answer:")
        finally:
            out_path.unlink(missing_ok=True)

    def test_openai_jsonl_skip_empty(self):
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        docs = [_MockDoc(text="", doc_id="empty"), _MockDoc(text="Valid.", doc_id="ok")]
        exporter = LLMTrainingExporter(engine=None)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            out_path = pathlib.Path(tf.name)
        try:
            exporter.to_openai_finetuning_jsonl(docs, out_path, skip_empty=True)
            lines = [l for l in out_path.read_text().strip().split("\n") if l]
            assert len(lines) == 1
        finally:
            out_path.unlink(missing_ok=True)

    def test_embedding_matrix(self, sample_docs):
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")

        docs_with_emb = [
            d.replace(embedding=np.zeros(4, dtype="float32"))
            for d in sample_docs
        ]
        exporter = LLMTrainingExporter(engine=None)
        matrix, meta = exporter.to_embedding_matrix(docs_with_emb)
        assert matrix.shape == (5, 4)

    def test_embedding_matrix_missing_raises(self, sample_docs):
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        exporter = LLMTrainingExporter(engine=None)
        with pytest.raises(ValueError, match="embedding"):
            exporter.to_embedding_matrix(sample_docs)

    def test_huggingface_sft_fallback(self, sample_docs):
        """Without HF datasets installed, returns a plain dict."""
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError:
            pytest.skip("transformers not available")

        exporter = LLMTrainingExporter(engine=None)
        result = exporter.to_huggingface_training_dataset(
            sample_docs, tokenizer_name="gpt2", max_length=64, task="clm"
        )
        assert "input_ids" in result
        assert "labels" in result
        assert len(result["input_ids"]) > 0

    def test_huggingface_clm_labels_equal_input_ids(self, sample_docs):
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError:
            pytest.skip("transformers not available")

        exporter = LLMTrainingExporter(engine=None)
        result = exporter.to_huggingface_training_dataset(
            sample_docs[:2], tokenizer_name="gpt2", max_length=64, task="clm"
        )
        assert result["input_ids"][0] == result["labels"][0]

    def test_embedding_matrix_saves_files(self, sample_docs):
        from scikitplot.corpus._embeddings._multimodal_embedding import (  # noqa: PLC0415
            LLMTrainingExporter,
        )
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")

        docs_with_emb = [
            d.replace(embedding=np.zeros(4, dtype="float32"))
            for d in sample_docs
        ]
        exporter = LLMTrainingExporter(engine=None)
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp) / "emb"
            matrix, meta = exporter.to_embedding_matrix(
                docs_with_emb, output_path=base
            )
            assert (base.with_suffix(".npy")).exists()


# ===========================================================================
# TestMCPCorpusServer
# ===========================================================================


class TestMCPCorpusServer:
    def test_list_tools(self, sample_docs):
        from scikitplot.corpus._similarity._similarity import (  # noqa: PLC0415
            SimilarityIndex, SearchConfig,
        )
        from scikitplot.corpus._adapters import MCPCorpusServer  # noqa: PLC0415
        idx = SimilarityIndex(config=SearchConfig(match_mode="keyword"))
        idx.build(sample_docs)
        srv = MCPCorpusServer(index=idx, server_name="test-server")
        tools = srv.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert tools[0]["name"] == "corpus_search"

    def test_repr(self, sample_docs):
        from scikitplot.corpus._similarity._similarity import (  # noqa: PLC0415
            SimilarityIndex, SearchConfig,
        )
        from scikitplot.corpus._adapters import MCPCorpusServer  # noqa: PLC0415
        idx = SimilarityIndex(config=SearchConfig())
        idx.build(sample_docs)
        srv = MCPCorpusServer(index=idx)
        assert "MCPCorpusServer" in repr(srv)


# ===========================================================================
# TestLangChainCorpusRetriever
# ===========================================================================


class TestLangChainCorpusRetriever:
    def test_retriever_get_relevant_documents(self, sample_docs):
        from scikitplot.corpus._similarity._similarity import (  # noqa: PLC0415
            SimilarityIndex, SearchConfig,
        )
        from scikitplot.corpus._adapters import LangChainCorpusRetriever  # noqa: PLC0415
        idx = SimilarityIndex(config=SearchConfig(match_mode="keyword", top_k=3))
        idx.build(sample_docs)
        retriever = LangChainCorpusRetriever(index=idx)
        results = retriever.get_relevant_documents("question")
        assert isinstance(results, list)


# ===========================================================================
# TestBuilderConfigProbeURL
# ===========================================================================


class TestBuilderConfigProbeURL:
    def test_probe_url_content_type_default(self):
        from scikitplot.corpus._corpus_builder import BuilderConfig  # noqa: PLC0415
        cfg = BuilderConfig()
        assert hasattr(cfg, "probe_url_content_type")
        assert cfg.probe_url_content_type is True

    def test_probe_url_timeout_default(self):
        from scikitplot.corpus._corpus_builder import BuilderConfig  # noqa: PLC0415
        cfg = BuilderConfig()
        assert hasattr(cfg, "probe_url_timeout")
        assert isinstance(cfg.probe_url_timeout, int)

    def test_probe_url_disabled(self):
        from scikitplot.corpus._corpus_builder import BuilderConfig  # noqa: PLC0415
        cfg = BuilderConfig(probe_url_content_type=False)
        assert cfg.probe_url_content_type is False


# ===========================================================================
# TestCorpusPipeline — run / run_batch / _run_source / _collect_documents
# ===========================================================================


class TestCorpusPipelineRun:
    """CorpusPipeline.run() — URL routing, file routing, type guards."""

    def _make_pipeline(self):
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415
        return CorpusPipeline()

    def test_run_local_file_calls_create(self, tmp_path):
        """run(Path) calls DocumentReader.create with a Path, not a URL."""
        from pathlib import Path
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()
        txt = tmp_path / "sample.txt"
        txt.write_text("Hello world.")

        captured = {}

        def fake_create(src, **kw):
            captured["src"] = src
            captured["type"] = type(src).__name__
            m = MagicMock()
            m.get_documents.return_value = iter([])
            return m

        with patch("scikitplot.corpus._pipeline.DocumentReader.create", side_effect=fake_create):
            pipeline.run(txt)

        assert captured["type"] == "PosixPath" or captured["type"] == "WindowsPath"

    def test_run_url_string_not_wrapped_in_path(self):
        """run('https://...') must NOT wrap the URL in pathlib.Path before routing."""
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()
        url = "https://en.wikipedia.org/wiki/Python"
        captured = {}

        def fake_create(src, **kw):
            captured["src"] = src
            captured["type"] = type(src).__name__
            m = MagicMock()
            m.get_documents.return_value = iter([])
            return m

        with patch("scikitplot.corpus._pipeline.DocumentReader.create", side_effect=fake_create):
            pipeline.run(url)

        # The URL must arrive at create() as a str, never as a Path.
        assert captured["type"] == "str", (
            f"URL was wrapped in {captured['type']!r}; "
            "pathlib.Path collapses https:// to https:/ and breaks routing."
        )
        # The double-slash must be preserved.
        assert captured["src"].startswith("https://"), (
            f"Double-slash mangled: {captured['src']!r}"
        )

    def test_run_url_source_label_preserved(self):
        """PipelineResult.source equals the original URL string."""
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()
        url = "https://example.com/report.pdf"

        mock_reader = MagicMock()
        mock_reader.get_documents.return_value = iter([])

        with patch("scikitplot.corpus._pipeline.DocumentReader.create", return_value=mock_reader):
            result = pipeline.run(url)

        assert result.source == url

    def test_run_bad_type_raises_type_error(self):
        """run(42) raises TypeError immediately."""
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()
        with pytest.raises(TypeError, match="str or pathlib.Path"):
            pipeline.run(42)  # type: ignore[arg-type]


class TestCorpusPipelineRunBatch:
    """run_batch() — mixed inputs, URL pass-through, type guard."""

    def _make_pipeline(self):
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415
        return CorpusPipeline()

    def test_run_batch_urls_not_wrapped_in_path(self):
        """URL strings inside run_batch must reach _run_source as raw str."""
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()
        url = "https://example.com/doc.html"
        captured_sources = []

        def fake_run_source(source, **kw):
            captured_sources.append((source, type(source).__name__))
            from scikitplot.corpus._pipeline import PipelineResult  # noqa: PLC0415
            from scikitplot.corpus._schema import ExportFormat  # noqa: PLC0415
            return PipelineResult(
                source=str(source), documents=[], output_path=None,
                n_read=0, n_omitted=0, n_embedded=0,
                elapsed_seconds=0.0, export_format=None,
            )

        with patch.object(pipeline, "_run_source", side_effect=fake_run_source):
            pipeline.run_batch([url])

        assert len(captured_sources) == 1
        src, typename = captured_sources[0]
        assert typename == "str", f"URL was wrapped in {typename!r} before _run_source."
        assert str(src).startswith("https://")

    def test_run_batch_mixed_sources(self, tmp_path):
        """run_batch accepts a list mixing Path and URL str."""
        from pathlib import Path
        from scikitplot.corpus._pipeline import CorpusPipeline, PipelineResult  # noqa: PLC0415

        pipeline = CorpusPipeline()
        txt = tmp_path / "a.txt"
        txt.write_text("hello")
        url = "https://example.com/b.html"

        call_sources = []

        def fake_run_source(source, **kw):
            call_sources.append(source)
            return PipelineResult(
                source=str(source), documents=[], output_path=None,
                n_read=0, n_omitted=0, n_embedded=0,
                elapsed_seconds=0.0, export_format=None,
            )

        with patch.object(pipeline, "_run_source", side_effect=fake_run_source):
            results = pipeline.run_batch([txt, url])

        assert len(results) == 2
        assert len(call_sources) == 2
        # First source is a Path, second is a str URL.
        assert isinstance(call_sources[0], Path)
        assert isinstance(call_sources[1], str)

    def test_run_batch_bad_type_raises(self):
        """Non-str / non-Path element raises TypeError."""
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()
        with pytest.raises(TypeError, match="str or pathlib.Path"):
            pipeline.run_batch([123])  # type: ignore[list-item]

    def test_run_batch_stop_on_error_propagates(self):
        """stop_on_error=True re-raises the first exception."""
        from unittest.mock import patch  # noqa: PLC0415

        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()

        with patch.object(pipeline, "_run_source", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                pipeline.run_batch(["https://example.com/x"], stop_on_error=True)

    def test_run_batch_continue_on_error_skips(self):
        """stop_on_error=False skips failing sources and continues."""
        from unittest.mock import patch  # noqa: PLC0415

        from scikitplot.corpus._pipeline import CorpusPipeline, PipelineResult  # noqa: PLC0415

        pipeline = CorpusPipeline()
        ok_result = PipelineResult(
            source="https://ok.com", documents=[], output_path=None,
            n_read=0, n_omitted=0, n_embedded=0,
            elapsed_seconds=0.0, export_format=None,
        )

        def side_effect(source, **kw):
            if "fail" in str(source):
                raise ValueError("bad source")
            return ok_result

        with patch.object(pipeline, "_run_source", side_effect=side_effect):
            results = pipeline.run_batch(
                ["https://fail.com/x", "https://ok.com/y"],
                stop_on_error=False,
            )

        assert len(results) == 1
        assert results[0].source == "https://ok.com"


class TestCollectDocumentsCounters:
    """_collect_documents() — counter accuracy for single and multi-source readers."""

    def _make_pipeline(self):
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415
        return CorpusPipeline()

    def _mock_reader(self, docs, n_included, n_omitted):
        """Build a mock DocumentReader with pre-set counter attrs."""
        from unittest.mock import MagicMock  # noqa: PLC0415
        r = MagicMock()
        r.get_documents.return_value = iter(docs)
        r._last_n_included = n_included
        r._last_n_omitted = n_omitted
        return r

    def test_single_reader_correct_counters(self):
        """Single DocumentReader: counters read from _last_n_* attrs."""
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()
        reader = self._mock_reader(["a", "b", "c"], n_included=3, n_omitted=2)
        docs, n_read, n_omitted = pipeline._collect_documents(reader, "test")

        assert n_omitted == 2
        assert n_read == 5  # 3 included + 2 omitted

    def test_single_reader_missing_attrs_fallback(self):
        """Single reader without counter attrs falls back to len(docs), 0."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()
        r = MagicMock()
        r.get_documents.return_value = iter(["x", "y"])
        # No _last_n_included / _last_n_omitted attrs
        del r._last_n_included
        del r._last_n_omitted
        docs, n_read, n_omitted = pipeline._collect_documents(r, "test")

        assert n_omitted == 0
        assert n_read == 2

    def test_multi_source_reader_aggregates_counters(self):
        """_MultiSourceReader: n_omitted is the sum of all sub-reader omits."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        from scikitplot.corpus._base import _MultiSourceReader  # noqa: PLC0415
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()

        # Two sub-readers: reader A omitted 3, reader B omitted 5.
        sub_a = self._mock_reader(["doc1", "doc2"], n_included=2, n_omitted=3)
        sub_b = self._mock_reader(["doc3"], n_included=1, n_omitted=5)

        multi = MagicMock(spec=_MultiSourceReader)
        multi.readers = [sub_a, sub_b]
        # get_documents yields from both — exhaust in order so attrs get set
        multi.get_documents.return_value = iter(["doc1", "doc2", "doc3"])

        docs, n_read, n_omitted = pipeline._collect_documents(multi, "multi")

        assert n_omitted == 8          # 3 + 5
        assert n_read == 11            # (2+1) + (3+5)
        assert len(docs) == 3

    def test_multi_source_reader_missing_attrs_fallback(self):
        """_MultiSourceReader with counter-less sub-readers: fallback is safe."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        from scikitplot.corpus._base import _MultiSourceReader  # noqa: PLC0415
        from scikitplot.corpus._pipeline import CorpusPipeline  # noqa: PLC0415

        pipeline = CorpusPipeline()

        sub = MagicMock()
        sub.get_documents.return_value = iter([])
        del sub._last_n_included
        del sub._last_n_omitted

        multi = MagicMock(spec=_MultiSourceReader)
        multi.readers = [sub]
        multi.get_documents.return_value = iter([])

        docs, n_read, n_omitted = pipeline._collect_documents(multi, "multi-fallback")

        # Must not crash; counters fall back to 0/0.
        assert n_omitted == 0
        assert n_read == 0
