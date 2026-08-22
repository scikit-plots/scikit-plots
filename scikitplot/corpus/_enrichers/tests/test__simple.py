# scikitplot/corpus/_enrichers/tests/test__simple.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import pytest

from scikitplot.corpus import (
    CorpusDocument,
    FluentCorpus,
    ParagraphChunkerConfig,
    RuntimePolicy,
    SimpleEnricherSpec,
    SimpleFrequencyEnricher,
)


def _doc(
    text: str,
    *,
    normalized_text: str | None = None,
    tokens: list[str] | None = None,
    keywords: list[str] | None = None,
) -> CorpusDocument:
    return CorpusDocument.create(
        "example.txt",
        0,
        text,
        normalized_text=normalized_text,
        tokens=tokens,
        keywords=keywords,
    )


def test_simple_enricher_is_deterministic_and_uses_normalized_text() -> None:
    enricher = SimpleFrequencyEnricher(
        min_token_length=2,
        max_keywords=3,
    )
    doc = _doc(
        "RAW SHOULD NOT WIN",
        normalized_text="Ghost ghost father sleep sleep dream",
    )

    enriched = enricher.enrich_documents([doc])[0]

    assert enriched.tokens == ["ghost", "ghost", "father", "sleep", "sleep", "dream"]
    assert enriched.keywords == ["ghost", "sleep", "dream"]
    assert doc.tokens is None
    assert doc.keywords is None


def test_simple_enricher_supports_unicode_words() -> None:
    enricher = SimpleFrequencyEnricher(min_token_length=2, max_keywords=10)
    enriched = enricher.enrich_documents(
        [_doc("İstanbul dünya 東京 世界 مرحبا بالعالم")]
    )[0]

    assert enriched.tokens is not None
    assert enriched.tokens[0].startswith("i")
    assert enriched.tokens[0].endswith("stanbul")
    assert "dünya" in enriched.tokens
    assert "東京" in enriched.tokens
    assert "世界" in enriched.tokens
    assert "مرحبا" in enriched.tokens
    assert "بالعالم" in enriched.tokens


def test_simple_enricher_reuses_existing_tokens_without_overwrite() -> None:
    doc = _doc(
        "text is ignored because tokens already exist",
        tokens=["beta", "alpha", "beta"],
    )
    enriched = SimpleFrequencyEnricher(max_keywords=2).enrich_documents([doc])[0]

    assert enriched.tokens == ["beta", "alpha", "beta"]
    assert enriched.keywords == ["beta", "alpha"]


def test_simple_enricher_preserves_fully_enriched_document_identity() -> None:
    doc = _doc("text", tokens=["already"], keywords=["already"])
    enriched = SimpleFrequencyEnricher().enrich_documents([doc])[0]
    assert enriched is doc


def test_simple_enricher_overwrite_recomputes_tokens_and_keywords() -> None:
    doc = _doc("gamma gamma delta", tokens=["old"], keywords=["old"])
    enriched = SimpleFrequencyEnricher(max_keywords=2).enrich_documents(
        [doc], overwrite=True
    )[0]

    assert enriched.tokens == ["gamma", "gamma", "delta"]
    assert enriched.keywords == ["gamma", "delta"]


def test_simple_enricher_spec_validates_configuration() -> None:
    with pytest.raises(ValueError, match="min_token_length"):
        SimpleEnricherSpec(min_token_length=0)
    with pytest.raises(ValueError, match="max_keywords"):
        SimpleEnricherSpec(max_keywords=0)
    with pytest.raises(ValueError, match="invalid token_pattern"):
        SimpleEnricherSpec(token_pattern="[")


def test_direct_overrides_and_spec_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either spec or direct overrides"):
        SimpleFrequencyEnricher(SimpleEnricherSpec(), max_keywords=4)


def test_fluent_runtime_materializes_simple_enricher_spec(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Ghost ghost father.\n\nSleep dream death.", encoding="utf-8")

    fluent = (
        FluentCorpus()
        .source(source)
        .chunker(ParagraphChunkerConfig(min_length=1, max_length=500))
        .enricher(SimpleEnricherSpec(min_token_length=2, max_keywords=3))
        .storage("memory")
    )

    runtime = fluent.materialize(policy=RuntimePolicy(allow_network=False))
    try:
        result = runtime.run()
        assert result.n_documents >= 1
        assert all(doc.tokens for doc in runtime.documents)
        assert all(doc.keywords for doc in runtime.documents)
    finally:
        runtime.close()
