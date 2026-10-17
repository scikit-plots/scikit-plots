from __future__ import annotations

import sqlite3
from pathlib import Path

import scikitplot.corpus as corpus

HERE = Path(__file__).parent


def _text(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8")


def test_index_lists_every_guide_page() -> None:
    index = _text("index.rst")
    pages = {
        "getting_started",
        "architecture",
        "readers_and_backends",
        "fluent_and_policies",
        "customization",
        "downloads_and_network",
        "retrieval_and_similarity",
        "evaluation",
        "llm_evaluation",
        "formats_and_export",
        "security_and_limits",
        "troubleshooting",
    }
    for page in pages:
        assert page in index


def test_documented_new_public_contracts_exist() -> None:
    for name in (
        "ASRBackend",
        "ASRRequest",
        "BackendPlan",
        "BackendPolicy",
        "BuilderFactories",
        "CapabilityRegistry",
        "CapabilityReport",
        "CapabilitySpec",
        "CorpusBuilder",
        "CorpusPolicyBundle",
        "FactoryCorpusBuilder",
        "DownloadPlan",
        "DownloadPolicy",
        "FluentCorpus",
        "RuntimePolicy",
        "component_capabilities",
    ):
        assert hasattr(corpus, name), name


def test_readiness_vocabulary_is_kept_visible() -> None:
    text = _text("readers_and_backends.rst")
    for word in ("installed", "assets_ready", "ready", "selected", "active"):
        assert f"``{word}``" in text


def test_security_page_does_not_call_fail_soft_success() -> None:
    text = _text("security_and_limits.rst").lower()
    assert "fail-soft is not silent success" in text
    assert "backend_reports" in text


def test_customization_docs_cover_all_builder_factory_seams() -> None:
    text = _text("customization.rst")
    for field in (
        "reader_factory",
        "chunker_factory",
        "filter_factory",
        "downloader_factory",
        "normalizer_factory",
        "enricher_factory",
        "embedding_engine_factory",
    ):
        assert f"``{field}``" in text


def test_backend_outcome_vocabulary_does_not_collapse_unavailable_into_empty() -> None:
    text = _text("readers_and_backends.rst")
    for word in ("success", "empty", "degraded", "unavailable", "exhausted", "failed"):
        assert f"``{word}``" in text


def test_customization_docs_prefer_native_corpus_builder_factory_seam() -> None:
    text = _text("customization.rst")
    assert "CorpusBuilder" in text
    assert "compatibility" in text
    assert "FactoryCorpusBuilder" in text


def test_policy_and_generator_docs_cover_new_contracts() -> None:
    fluent = _text("fluent_and_policies.rst")
    downloads = _text("downloads_and_network.rst")
    readers = _text("readers_and_backends.rst")
    retrieval = _text("retrieval_and_similarity.rst")
    assert "iter_variants" in fluent
    assert "DownloadPolicy" in fluent
    assert "CorpusPolicyBundle" in fluent
    assert "safe_local" in fluent
    assert "strict_local" in fluent
    assert "DownloadPolicy" in downloads
    assert "DownloadPlan" in downloads
    assert "plan_all" in downloads
    assert "plan_asr_backends" in readers
    assert "BackendPlan" in readers
    assert "score_cutoff" in retrieval


def test_evaluation_page_names_resolve() -> None:
    """Every evaluation function the guide links is a public Corpus name."""
    text = _text("evaluation.rst")
    names = {
        "anls_score", "edit_similarity_score", "evaluate_retrieval", "exact_match_score",
        "get_corpus_scorer", "hit_rate_at_k", "make_corpus_scorer",
        "make_levenshtein_ranker", "make_pairwise_ranker", "mean_average_precision",
        "mean_reciprocal_rank", "ndcg_at_k", "precision_at_k", "r_precision",
        "ranking_metrics", "recall_at_k", "scorer_backend_info", "token_f1_score",
    }
    for name in names:
        assert f":func:`{name}`" in text, name
        assert hasattr(corpus, name), name


def test_llm_evaluation_page_names_resolve() -> None:
    """Every judge, harness and agreement name the LLM page links is public."""
    text = _text("llm_evaluation.rst")
    names = {
        "cohen_kappa", "context_texts", "evaluate_answers", "evaluate_preference",
        "fleiss_kappa", "krippendorff_alpha", "make_llm", "parse_verdict",
    }
    for name in names:
        assert f":func:`{name}`" in text, name
        assert hasattr(corpus, name), name
    for name in ("AnswerEvaluation", "CacheError", "IncompleteCompletionError", "JudgeResult", "LLMJudge"):
        assert hasattr(corpus, name), name


def test_export_page_lists_the_loadable_formats() -> None:
    """The load-back table names every text format and the functions exist."""
    text = _text("formats_and_export.rst")
    for name in ("export_documents", "load_documents"):
        assert f":func:`{name}`" in text, name
        assert hasattr(corpus, name), name
    for member in ("JSON", "JSONL", "YAML", "TOML"):
        assert f"``{member}``" in text, member
        assert hasattr(corpus.ExportFormat, member), member


def test_identity_section_names_resolve() -> None:
    """The document-identity section names public schemes and functions."""
    text = _text("formats_and_export.rst")
    for name in ("assign_doc_ids", "uuid7"):
        assert f":func:`{name}`" in text, name
        assert hasattr(corpus, name), name
    for name in (
        "DocIdScheme", "EmbeddingManifest", "IncompatibleSchemaError", "IndexGeneration",
    ):
        assert f":class:`{name}`" in text, name
        assert hasattr(corpus, name), name
    for member in corpus.DocIdScheme:
        assert f'``"{member.value}"``' in text, member


def test_fusion_section_names_resolve() -> None:
    """The hybrid-fusion section names the public contract and its users."""
    text = _text("retrieval_and_similarity.rst")
    assert ":class:`FusionPolicy`" in text
    assert hasattr(corpus, "FusionPolicy")
    assert ":func:`fuse_by_rank`" in text
    assert hasattr(corpus, "fuse_by_rank")


def test_filter_section_names_resolve() -> None:
    """The filtering section names public expressions and functions."""
    text = _text("retrieval_and_similarity.rst")
    for name in ("filter_from_dict", "filter_to_dict"):
        assert f":func:`{name}`" in text, name
        assert hasattr(corpus, name), name
    for name in ("Filter", "SecurityFilter"):
        assert f":class:`{name}`" in text, name
        assert hasattr(corpus, name), name


def test_embedding_cache_section_holds() -> None:
    """The embedding-cache section names public objects and true defaults."""
    text = _text("retrieval_and_similarity.rst")
    for name in ("CachePolicy", "CacheStatus", "EmbeddingEngine"):
        assert f":class:`{name}`" in text, name
        assert hasattr(corpus, name), name
    assert hasattr(corpus, "resolve_hub_revision")
    for mode in ("resolve", "allow"):
        assert f'unpinned="{mode}"' in text, mode
    assert corpus.CachePolicy().unpinned == "bypass"
    assert set(corpus.CachePolicy.__dataclass_fields__) == {
        "unpinned",
        "revision_resolver",
        "max_entries",
        "max_bytes",
    }


def test_segmenter_notice_section_holds() -> None:
    """The segmenter section's report call and its claims are true."""
    text = _text("troubleshooting.rst")
    assert 'component_capabilities(role="segmentation")' in text
    report = corpus.component_capabilities(role="segmentation")
    for name in ("jieba", "kss", "pythainlp", "khmer-nltk", "camel-tools", "regex"):
        assert f"segmenter:{name}" in report, name
    for record in report.values():
        assert {"status", "remedy"} <= set(record)


def test_unclosed_database_section_holds(tmp_path) -> None:
    """The resource section's example works and closes the store."""
    text = _text("troubleshooting.rst")
    assert "with SQLiteStorage(" in text
    with corpus.SQLiteStorage(tmp_path / "corpus.db") as store:
        store.save_batch([])
    try:
        store.count()
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("the store is still open after the with block")


def test_recursive_section_holds(tmp_path) -> None:
    """The recursive / parent-child section's names and example hold."""
    text = _text("customization.rst")
    for name in ("RecursiveChunker", "ParentChildChunker"):
        assert f":class:`{name}`" in text, name
        assert hasattr(corpus, name), name
    for name in ("children", "ancestors"):
        assert f":func:`{name}`" in text, name
        assert hasattr(corpus, name), name
    source = tmp_path / "guide.txt"
    source.write_text(("A sentence here. " * 20 + "\n\n") * 6, encoding="utf-8")
    config = corpus.BuilderConfig(
        chunker="parent_child",
        chunker_kwargs={"parent_size": 1500, "child_size": 200},
    )
    docs = corpus.CorpusBuilder(config).build(source).documents
    children = [d for d in docs if d.metadata["chunk_role"] == "child"]
    parents = {d.doc_id for d in docs if d.metadata["chunk_role"] == "parent"}
    assert children
    assert all(d.parent_doc_id in parents for d in children)


def test_structure_section_holds() -> None:
    """The structure-aware paragraph's names and metadata keys are true."""
    text = _text("customization.rst")
    assert ":class:`StructureChunker`" in text
    assert hasattr(corpus, "StructureChunker")
    source = "Title\n=====\n\nSome prose.\n\n.. code-block:: python\n\n   x = 1\n"
    chunker = corpus.StructureChunker(
        corpus.StructureChunkerConfig(markup="rst", chunk_size=500)
    )
    chunks = chunker.chunk(source).chunks
    assert [c.metadata["block"] for c in chunks] == ["prose", "code"]
    assert chunks[0].metadata["heading_path"] == ("Title",)
    assert chunks[0].text.startswith("Title")  # include_heading


def test_fallback_section_names_resolve() -> None:
    """The fallback-vocabulary section names public objects and the four names."""
    text = _text("readers_and_backends.rst")
    for name in ("requested_backend", "backend", "fallback_reason", "allow_fallback"):
        assert f"``{name}``" in text, name
    for name in ("capability_snapshot", "probe_backend"):
        assert f":func:`{name}`" in text, name
        assert hasattr(corpus, name), name
    for name in ("BackendOutcome", "FallbackRefusedError"):
        assert f":class:`{name}`" in text, name
        assert hasattr(corpus, name), name
    assert hasattr(corpus.RetrievalIndex, "backend_provenance")


def test_chunker_names_section_resolves() -> None:
    """The names-and-configs section names public objects and every built-in."""
    text = _text("customization.rst")
    for name in ("ComponentRegistry", "ProtocolChunkerBridge"):
        assert f":class:`{name}`" in text, name
        assert hasattr(corpus, name), name
    corpus.registry.register_builtins()
    for name in corpus.registry.list_chunkers():
        assert f"``{name}``" in text, name
    assert corpus.registry.build_chunker("sentence", overlap=1).config.overlap == 1


def test_chunk_spans_section_holds() -> None:
    """The chunk-spans section's example is true."""
    text = _text("customization.rst")
    for name in (
        "context_span",
        "anchor_span",
        "text_is_source_slice",
        "context_start",
        "span_basis",
    ):
        assert name in text, name
    source = "Alpha one. Beta two."
    chunk = (
        corpus.SentenceChunker(corpus.SentenceChunkerConfig(overlap=1, min_length=1))
        .chunk(source)
        .chunks[1]
    )
    assert chunk.text == source[slice(*chunk.context_span)] == source
    assert source[slice(*chunk.anchor_span)] == "Beta two."


def test_execution_plan_section_holds() -> None:
    """The execution-plan section names public objects, stages and statuses."""
    text = _text("retrieval_and_similarity.rst")
    for name in (
        "RetrievalExecutionPlan",
        "StageOutcome",
        "StageStatus",
        "StageBudget",
        "FusionPolicy",
    ):
        assert f":class:`{name}`" in text, name
        assert hasattr(corpus, name), name
    assert hasattr(corpus, "CharBudgetCompressor")
    order = [k.value for k in corpus.StageKind]
    assert order == ["transform", "retrieve", "fuse", "rerank", "compress"]
    positions = [text.index(f"``{stage}``") for stage in order]
    assert positions == sorted(positions)
    for status in corpus.StageStatus:
        assert f"``{status.value}``" in text, status
    assert corpus.RetrievalExecutionPlan().on_error == "degrade"


def test_corrective_retrieval_section_holds() -> None:
    """The corrective-retrieval section names real objects, grades and verdicts."""
    text = _text("retrieval_and_similarity.rst")
    section = text[text.index("Corrective retrieval\n") :]
    for name in ("CorrectiveRetrieval", "TermCoverageGrader", "FusionPolicy"):
        assert f":class:`{name}`" in section, name
        assert hasattr(corpus, name), name
    for value in [g.value for g in corpus.EvidenceGrade] + [
        v.value for v in corpus.CorrectionVerdict
    ]:
        assert f"``{value}``" in section, value
    defaults = corpus.CorrectiveRetrieval()
    assert defaults.grader is None  # TermCoverageGrader(), the table's first row
    assert defaults.rewrite is None
    assert defaults.max_rounds == 2  # noqa: PLR2004
    assert "bench_r5b.py" in section
