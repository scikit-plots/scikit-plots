from __future__ import annotations

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
