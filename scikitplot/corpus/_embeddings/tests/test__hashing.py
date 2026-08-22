# scikitplot/corpus/_embeddings/tests/test__hashing.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from scikitplot.corpus import (
    FluentCorpus,
    HashEmbedder,
    ParagraphChunkerConfig,
    RetrievalConfig,
    RuntimePolicy,
)


def test_hash_embedder_is_deterministic_normalized_float32() -> None:
    embedder = HashEmbedder(dimension=64)
    first = embedder(["ghost father", "sleep dream", ""])
    second = embedder(["ghost father", "sleep dream", ""])

    assert first.shape == (3, 64)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first[:2], axis=1), [1.0, 1.0])
    assert np.linalg.norm(first[2]) == 0.0


def test_hash_embedder_supports_unicode_tokens() -> None:
    embedder = HashEmbedder(dimension=128)
    vectors = embedder(["İstanbul dünya", "東京 世界", "مرحبا بالعالم"])

    assert vectors.shape == (3, 128)
    assert np.all(np.linalg.norm(vectors, axis=1) > 0.0)


def test_hash_embedder_accepts_one_shot_iterable() -> None:
    embedder = HashEmbedder(dimension=16)
    vectors = embedder(text for text in ["alpha", "beta"])
    assert vectors.shape == (2, 16)


@pytest.mark.parametrize("dimension", [0, -1])
def test_hash_embedder_rejects_nonpositive_dimension(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension must be > 0"):
        HashEmbedder(dimension=dimension)


def test_hash_embedder_rejects_non_string_text() -> None:
    embedder = HashEmbedder(dimension=16)
    with pytest.raises(TypeError, match=r"texts\[1\] must be str"):
        embedder(["alpha", 7])  # type: ignore[list-item]


def test_hash_embedder_materializes_through_fluent_runtime(tmp_path) -> None:
    source = tmp_path / "source.txt"
    source.write_text(
        "Ghost father spirit.\n\nSleep dream death.",
        encoding="utf-8",
    )

    fluent = (
        FluentCorpus()
        .source(source)
        .chunker(ParagraphChunkerConfig(min_length=1, max_length=500))
        .embedder(HashEmbedder(dimension=32))
        .storage("memory")
        .index(RetrievalConfig(backend="bruteforce"))
        .retrieval(RetrievalConfig(match_mode="semantic", top_k=2))
    )

    runtime = fluent.materialize(policy=RuntimePolicy(allow_network=False))
    try:
        result = runtime.run()
        assert result.n_documents >= 1
        assert runtime.index is not None
        assert runtime.index.has_embeddings
        assert runtime.documents[0].embedding is not None
        assert runtime.documents[0].embedding.shape == (32,)
        response = runtime.search("ghost spirit")
        assert response
    finally:
        runtime.close()
