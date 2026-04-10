"""Tests for corpus._types — the single source of truth for all type contracts."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from .. import _types as m
from .._types import (
    BowVector,
    Chunk,
    ChunkResult,
    ChunkStrategy,
    ChunkerConfig,
    ChunkerProtocol,
    ChunkerRegistration,
    ContentType,
    CorpusRecord,
    Document,
    DocumentStatus,
    EmbeddedChunk,
    MCPToolInput,
    MCPToolResult,
    MetadataDict,
    NormalizerConfig,
    NormalizerProtocol,
    NormalizerType,
    PipelineConfig,
    PipelineResult,
    PipelineStep,
    RetrievalQuery,
    RetrievalResult,
    SourceConfig,
    StorageBackend,
    StorageConfig,
    StorageProtocol,
    TrainingDataset,
    TrainingExample,
    ValidationError,
    ValidationResult,
)


# ===========================================================================
# Section 1 — Enumerations
# ===========================================================================


class TestDocumentStatus:
    def test_all_values_are_str(self) -> None:
        for member in DocumentStatus:
            assert isinstance(member.value, str)

    def test_pending_value(self) -> None:
        assert DocumentStatus.PENDING.value == "pending"

    def test_failed_value(self) -> None:
        assert DocumentStatus.FAILED.value == "failed"

    def test_ready_value(self) -> None:
        assert DocumentStatus.READY.value == "ready"

    def test_enum_str_comparison(self) -> None:
        assert DocumentStatus.PENDING == "pending"


class TestContentType:
    def test_plain_text_value(self) -> None:
        assert ContentType.PLAIN_TEXT.value == "text/plain"

    def test_json_value(self) -> None:
        assert ContentType.JSON.value == "application/json"

    def test_unknown_value(self) -> None:
        assert ContentType.UNKNOWN.value == "application/octet-stream"


class TestChunkStrategy:
    def test_all_strategies_present(self) -> None:
        expected = {
            "sentence", "paragraph", "fixed_window",
            "word", "semantic", "recursive", "custom",
        }
        actual = {s.value for s in ChunkStrategy}
        assert expected == actual

    def test_str_equality(self) -> None:
        assert ChunkStrategy.SENTENCE == "sentence"


class TestStorageBackend:
    def test_memory_value(self) -> None:
        assert StorageBackend.MEMORY.value == "memory"

    def test_chroma_value(self) -> None:
        assert StorageBackend.CHROMA.value == "chroma"


class TestNormalizerType:
    def test_unicode_value(self) -> None:
        assert NormalizerType.UNICODE.value == "unicode"

    def test_pii_redact_value(self) -> None:
        assert NormalizerType.PII_REDACT.value == "pii_redact"


# ===========================================================================
# Section 2 — Chunk
# ===========================================================================


class TestChunk:
    def test_basic_construction(self) -> None:
        c = Chunk(text="hello", start_char=0, end_char=5)
        assert c.text == "hello"
        assert c.start_char == 0
        assert c.end_char == 5

    def test_default_metadata_is_empty_dict(self) -> None:
        c = Chunk(text="hello", start_char=0, end_char=5)
        assert c.metadata == {}

    def test_custom_metadata(self) -> None:
        meta: MetadataDict = {"doc_id": "d1", "chunk_index": 0}
        c = Chunk(text="hello", start_char=0, end_char=5, metadata=meta)
        assert c.metadata["doc_id"] == "d1"

    def test_frozen_prevents_mutation(self) -> None:
        c = Chunk(text="hello", start_char=0, end_char=5)
        with pytest.raises((AttributeError, TypeError)):
            c.text = "world"  # type: ignore[misc]

    def test_char_length(self) -> None:
        c = Chunk(text="hello", start_char=10, end_char=15)
        assert c.char_length() == 5

    def test_char_length_offsets_disabled(self) -> None:
        c = Chunk(text="hello", start_char=0, end_char=0)
        assert c.char_length() == 0

    def test_content_hash_is_sha256(self) -> None:
        c = Chunk(text="hello", start_char=0, end_char=5)
        h = c.content_hash()
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(ch in "0123456789abcdef" for ch in h)

    def test_content_hash_deterministic(self) -> None:
        c1 = Chunk(text="hello", start_char=0, end_char=5)
        c2 = Chunk(text="hello", start_char=99, end_char=104)
        assert c1.content_hash() == c2.content_hash()

    def test_content_hash_different_texts(self) -> None:
        c1 = Chunk(text="hello", start_char=0, end_char=5)
        c2 = Chunk(text="world", start_char=0, end_char=5)
        assert c1.content_hash() != c2.content_hash()

    def test_with_metadata_returns_new_instance(self) -> None:
        c = Chunk(text="hello", start_char=0, end_char=5)
        c2 = c.with_metadata(doc_id="d1")
        assert c.metadata == {}
        assert c2.metadata["doc_id"] == "d1"

    def test_with_metadata_preserves_existing(self) -> None:
        c = Chunk(
            text="hello", start_char=0, end_char=5,
            metadata={"existing": "value"}
        )
        c2 = c.with_metadata(new_key="new_value")
        assert c2.metadata["existing"] == "value"
        assert c2.metadata["new_key"] == "new_value"

    def test_with_metadata_does_not_mutate_original(self) -> None:
        meta = {"a": 1}
        c = Chunk(text="hi", start_char=0, end_char=2, metadata=meta)
        _ = c.with_metadata(b=2)
        assert "b" not in c.metadata

    def test_equality_by_value(self) -> None:
        c1 = Chunk(text="hi", start_char=0, end_char=2)
        c2 = Chunk(text="hi", start_char=0, end_char=2)
        assert c1 == c2

    def test_hashable(self) -> None:
        c = Chunk(text="hi", start_char=0, end_char=2)
        s = {c}
        assert c in s


# ===========================================================================
# Section 3 — ChunkResult
# ===========================================================================


class TestChunkResult:
    def _make_chunks(self, n: int = 3) -> list[Chunk]:
        return [
            Chunk(text=f"chunk {i}", start_char=i * 10, end_char=i * 10 + 7)
            for i in range(n)
        ]

    def test_basic_construction(self) -> None:
        chunks = self._make_chunks(2)
        r = ChunkResult(chunks=chunks)
        assert len(r.chunks) == 2

    def test_default_metadata_is_empty_dict(self) -> None:
        r = ChunkResult(chunks=[])
        assert r.metadata == {}

    def test_is_empty_true(self) -> None:
        r = ChunkResult(chunks=[])
        assert r.is_empty() is True

    def test_is_empty_false(self) -> None:
        r = ChunkResult(chunks=self._make_chunks(1))
        assert r.is_empty() is False

    def test_total_chars(self) -> None:
        chunks = [
            Chunk(text="abc", start_char=0, end_char=3),
            Chunk(text="de", start_char=3, end_char=5),
        ]
        r = ChunkResult(chunks=chunks)
        assert r.total_chars() == 5

    def test_texts_returns_list_of_strings(self) -> None:
        chunks = self._make_chunks(3)
        r = ChunkResult(chunks=chunks)
        texts = r.texts()
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)

    def test_iter_chunks_yields_chunks(self) -> None:
        chunks = self._make_chunks(2)
        r = ChunkResult(chunks=chunks)
        collected = list(r.iter_chunks())
        assert collected == chunks

    def test_frozen_prevents_reassignment(self) -> None:
        r = ChunkResult(chunks=[])
        with pytest.raises((AttributeError, TypeError)):
            r.chunks = []  # type: ignore[misc]


# ===========================================================================
# Section 4 — Document
# ===========================================================================


class TestDocument:
    def test_basic_construction(self) -> None:
        doc = Document(doc_id="d1", text="Hello.")
        assert doc.doc_id == "d1"
        assert doc.text == "Hello."

    def test_default_status_is_pending(self) -> None:
        doc = Document(doc_id="d1", text="Hi.")
        assert doc.status == DocumentStatus.PENDING

    def test_default_content_type(self) -> None:
        doc = Document(doc_id="d1", text="Hi.")
        assert doc.content_type == ContentType.PLAIN_TEXT

    def test_default_checksum_is_none(self) -> None:
        doc = Document(doc_id="d1", text="Hi.")
        assert doc.checksum is None

    def test_new_generates_uuid(self) -> None:
        doc = Document.new("Hello world.")
        assert len(doc.doc_id) == 36
        assert uuid.UUID(doc.doc_id)  # does not raise

    def test_new_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Document.new("")

    def test_new_rejects_whitespace_text(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Document.new("   ")

    def test_with_checksum_sets_digest(self) -> None:
        doc = Document(doc_id="d1", text="hello")
        doc2 = doc.with_checksum()
        assert doc2.checksum is not None
        assert len(doc2.checksum) == 64

    def test_with_checksum_does_not_mutate(self) -> None:
        doc = Document(doc_id="d1", text="hello")
        _ = doc.with_checksum()
        assert doc.checksum is None

    def test_with_status_returns_new(self) -> None:
        doc = Document(doc_id="d1", text="hello")
        doc2 = doc.with_status(DocumentStatus.READY)
        assert doc2.status == DocumentStatus.READY
        assert doc.status == DocumentStatus.PENDING

    def test_char_count(self) -> None:
        doc = Document(doc_id="d1", text="hello")
        assert doc.char_count() == 5

    def test_frozen_prevents_mutation(self) -> None:
        doc = Document(doc_id="d1", text="hi")
        with pytest.raises((AttributeError, TypeError)):
            doc.text = "bye"  # type: ignore[misc]


# ===========================================================================
# Section 5 — Abstract config base classes
# ===========================================================================


class TestAbstractConfigs:
    def test_chunker_config_instantiates(self) -> None:
        cfg = ChunkerConfig()
        assert isinstance(cfg, ChunkerConfig)

    def test_source_config_instantiates(self) -> None:
        cfg = SourceConfig()
        assert isinstance(cfg, SourceConfig)

    def test_normalizer_config_defaults(self) -> None:
        cfg = NormalizerConfig()
        assert cfg.enabled is True
        assert cfg.normalizer_type == NormalizerType.CUSTOM

    def test_storage_config_defaults(self) -> None:
        cfg = StorageConfig()
        assert cfg.backend == StorageBackend.MEMORY
        assert cfg.collection_name == "corpus"

    def test_chunker_config_frozen(self) -> None:
        cfg = ChunkerConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.some_field = "x"  # type: ignore[attr-defined]


# ===========================================================================
# Section 6 — Pipeline types
# ===========================================================================


class TestPipelineStep:
    def test_basic_construction(self) -> None:
        step = PipelineStep(
            name="chunk_sentences",
            step_type="chunker",
            config=ChunkerConfig(),
        )
        assert step.name == "chunk_sentences"
        assert step.enabled is True

    def test_disabled_step(self) -> None:
        step = PipelineStep(
            name="embed", step_type="embedder",
            config=ChunkerConfig(), enabled=False
        )
        assert step.enabled is False


class TestPipelineConfig:
    def test_basic_construction(self) -> None:
        step = PipelineStep(
            name="s1", step_type="chunker", config=ChunkerConfig()
        )
        cfg = PipelineConfig(pipeline_id="p1", steps=[step])
        assert cfg.pipeline_id == "p1"
        assert len(cfg.steps) == 1

    def test_empty_steps_allowed(self) -> None:
        cfg = PipelineConfig(pipeline_id="p_empty", steps=[])
        assert cfg.steps == []


class TestPipelineResult:
    def _make_result(
        self, status: DocumentStatus = DocumentStatus.READY
    ) -> PipelineResult:
        chunk = Chunk(text="hi", start_char=0, end_char=2)
        cr = ChunkResult(chunks=[chunk])
        return PipelineResult(
            pipeline_id="p1",
            doc_id="d1",
            chunk_results=[cr],
            status=status,
        )

    def test_succeeded_true(self) -> None:
        r = self._make_result(DocumentStatus.READY)
        assert r.succeeded() is True

    def test_succeeded_false(self) -> None:
        r = self._make_result(DocumentStatus.FAILED)
        assert r.succeeded() is False

    def test_all_chunks_flattens(self) -> None:
        r = self._make_result()
        chunks = r.all_chunks()
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)

    def test_all_chunks_multiple_results(self) -> None:
        c1 = Chunk(text="a", start_char=0, end_char=1)
        c2 = Chunk(text="b", start_char=1, end_char=2)
        cr1 = ChunkResult(chunks=[c1])
        cr2 = ChunkResult(chunks=[c2])
        r = PipelineResult(
            pipeline_id="p1", doc_id="d1",
            chunk_results=[cr1, cr2], status=DocumentStatus.READY
        )
        assert len(r.all_chunks()) == 2


# ===========================================================================
# Section 7 — Embedding & retrieval types
# ===========================================================================


class TestEmbeddedChunk:
    def test_valid_construction(self) -> None:
        chunk = Chunk(text="hello", start_char=0, end_char=5)
        ec = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
            embedding_dim=3,
        )
        assert ec.embedding_dim == 3

    def test_validate_dimension_passes(self) -> None:
        chunk = Chunk(text="hi", start_char=0, end_char=2)
        ec = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2],
            model_name="m",
            embedding_dim=2,
        )
        ec.validate_dimension()  # should not raise

    def test_validate_dimension_fails(self) -> None:
        chunk = Chunk(text="hi", start_char=0, end_char=2)
        ec = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2],
            model_name="m",
            embedding_dim=5,
        )
        with pytest.raises(ValueError, match="dimension"):
            ec.validate_dimension()


class TestRetrievalQuery:
    def test_defaults(self) -> None:
        q = RetrievalQuery(query_id="q1", text="what is rl?")
        assert q.top_k == 10
        assert q.filters == {}
        assert q.embedding is None

    def test_custom_top_k(self) -> None:
        q = RetrievalQuery(query_id="q1", text="hello", top_k=5)
        assert q.top_k == 5


class TestRetrievalResult:
    def test_basic_construction(self) -> None:
        chunk = Chunk(text="hi", start_char=0, end_char=2)
        r = RetrievalResult(chunk=chunk, score=0.95, rank=0)
        assert r.score == pytest.approx(0.95)
        assert r.rank == 0


# ===========================================================================
# Section 8 — CorpusRecord
# ===========================================================================


class TestCorpusRecord:
    def test_basic_construction(self) -> None:
        chunk = Chunk(text="hello", start_char=0, end_char=5)
        record = CorpusRecord(
            record_id="r1",
            chunk=chunk,
            doc_id="d1",
            collection="main",
            created_at="2026-01-01T00:00:00Z",
        )
        assert record.record_id == "r1"
        assert record.embedding is None

    def test_with_embedding(self) -> None:
        chunk = Chunk(text="hi", start_char=0, end_char=2)
        record = CorpusRecord(
            record_id="r2",
            chunk=chunk,
            doc_id="d1",
            collection="main",
            created_at="2026-01-01T00:00:00Z",
            embedding=[0.1, 0.2],
        )
        assert record.embedding == [0.1, 0.2]


# ===========================================================================
# Section 9 — Protocol runtime checks
# ===========================================================================


class TestProtocols:
    """Verify runtime_checkable protocol isinstance behaviour."""

    class ValidChunker:
        def chunk(
            self,
            text: str,
            doc_id: str | None = None,
            extra_metadata: dict[str, Any] | None = None,
        ) -> ChunkResult:
            return ChunkResult(chunks=[])

        def chunk_batch(
            self,
            texts: list[str],
            doc_ids: list[str] | None = None,
            extra_metadata: dict[str, Any] | None = None,
        ) -> list[ChunkResult]:
            return []

    class InvalidChunker:
        pass

    class ValidNormalizer:
        def normalize(self, text: str) -> str:
            return text

        def normalize_batch(self, texts: list[str]) -> list[str]:
            return texts

    def test_valid_chunker_satisfies_protocol(self) -> None:
        assert isinstance(self.ValidChunker(), ChunkerProtocol)

    def test_invalid_chunker_fails_protocol(self) -> None:
        assert not isinstance(self.InvalidChunker(), ChunkerProtocol)

    def test_valid_normalizer_satisfies_protocol(self) -> None:
        assert isinstance(self.ValidNormalizer(), NormalizerProtocol)


# ===========================================================================
# Section 10 — ChunkerRegistration
# ===========================================================================


class TestChunkerRegistration:
    def test_basic_construction(self) -> None:
        reg = ChunkerRegistration(
            strategy=ChunkStrategy.SENTENCE,
            chunker_class=object,
            default_config=ChunkerConfig(),
            description="Sentence splitter",
        )
        assert reg.strategy == ChunkStrategy.SENTENCE

    def test_frozen(self) -> None:
        reg = ChunkerRegistration(
            strategy=ChunkStrategy.WORD,
            chunker_class=object,
            default_config=ChunkerConfig(),
        )
        with pytest.raises((AttributeError, TypeError)):
            reg.strategy = ChunkStrategy.PARAGRAPH  # type: ignore[misc]


# ===========================================================================
# Section 11 — Validation types
# ===========================================================================


class TestValidationError:
    def test_basic_construction(self) -> None:
        err = ValidationError(field="text", message="must not be empty", value="")
        assert err.field == "text"
        assert err.value == ""


class TestValidationResult:
    def test_valid_result(self) -> None:
        r = ValidationResult(valid=True, errors=[])
        assert r.valid is True

    def test_invalid_result(self) -> None:
        err = ValidationError(field="text", message="empty", value="")
        r = ValidationResult(valid=False, errors=[err])
        assert not r.valid

    def test_raise_if_invalid_does_not_raise_when_valid(self) -> None:
        r = ValidationResult(valid=True)
        r.raise_if_invalid()  # must not raise

    def test_raise_if_invalid_raises_when_invalid(self) -> None:
        err = ValidationError(field="text", message="too short", value="x")
        r = ValidationResult(valid=False, errors=[err])
        with pytest.raises(ValueError, match="Validation failed"):
            r.raise_if_invalid()

    def test_raise_if_invalid_context_prepended(self) -> None:
        err = ValidationError(field="f", message="bad", value=None)
        r = ValidationResult(valid=False, errors=[err])
        with pytest.raises(ValueError, match="MyContext"):
            r.raise_if_invalid(context="MyContext")

    def test_default_errors_is_empty_list(self) -> None:
        r = ValidationResult(valid=True)
        assert r.errors == []


# ===========================================================================
# Section 12 — LLM training types
# ===========================================================================


class TestTrainingExample:
    def test_basic_construction(self) -> None:
        ex = TrainingExample(
            example_id="e1",
            prompt="What is RL?",
            completion="RL is reinforcement learning.",
        )
        assert ex.chunk is None
        assert ex.metadata == {}

    def test_with_chunk(self) -> None:
        chunk = Chunk(text="source text", start_char=0, end_char=11)
        ex = TrainingExample(
            example_id="e2",
            prompt="summarise",
            completion="short",
            chunk=chunk,
        )
        assert ex.chunk is not None


class TestTrainingDataset:
    def _make_dataset(self, n: int = 3) -> TrainingDataset:
        examples = [
            TrainingExample(
                example_id=f"e{i}",
                prompt=f"prompt {i}",
                completion=f"completion {i}",
            )
            for i in range(n)
        ]
        return TrainingDataset(dataset_id="ds1", examples=examples)

    def test_len(self) -> None:
        ds = self._make_dataset(5)
        assert len(ds) == 5

    def test_iter(self) -> None:
        ds = self._make_dataset(3)
        collected = list(ds)
        assert len(collected) == 3
        assert all(isinstance(e, TrainingExample) for e in collected)

    def test_default_split(self) -> None:
        ds = self._make_dataset()
        assert ds.split == "train"


# ===========================================================================
# Section 13 — MCP types
# ===========================================================================


class TestMCPTypes:
    def test_tool_input_construction(self) -> None:
        inp = MCPToolInput(
            tool_name="search_corpus",
            arguments={"query": "transformer", "top_k": 5},
            call_id="call_001",
        )
        assert inp.tool_name == "search_corpus"

    def test_tool_result_success(self) -> None:
        r = MCPToolResult(call_id="call_001", content=["result1"])
        assert r.is_error is False
        assert r.error_message is None

    def test_tool_result_error(self) -> None:
        r = MCPToolResult(
            call_id="call_002",
            content=None,
            is_error=True,
            error_message="tool not found",
        )
        assert r.is_error is True
        assert "not found" in r.error_message

    def test_frozen_tool_input(self) -> None:
        inp = MCPToolInput(
            tool_name="t", arguments={}, call_id="c1"
        )
        with pytest.raises((AttributeError, TypeError)):
            inp.tool_name = "other"  # type: ignore[misc]


# ===========================================================================
# Section 14 — __all__ completeness
# ===========================================================================


class TestPublicAPI:
    def test_all_exports_importable(self) -> None:
        for name in m.__all__:
            assert hasattr(m, name), (
                f"{name!r} listed in __all__ but not defined in _types."
            )

    def test_no_private_names_in_all(self) -> None:
        for name in m.__all__:
            assert not name.startswith("_"), (
                f"Private name {name!r} should not be in __all__."
            )
