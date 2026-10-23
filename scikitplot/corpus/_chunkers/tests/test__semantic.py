# scikitplot/corpus/_chunkers/tests/test__semantic.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Provenance tests for :mod:`scikitplot.corpus._chunkers._semantic`.

Notes
-----
**Developer.** ``SemanticChunker`` falls back from EMBEDDING/HYBRID to
MORPHOLOGICAL when ``sentence-transformers`` is missing or the model fails.
It used to report the *requested* backend afterwards -- ``backend="hybrid"``,
``chunking_strategy="semantic_hybrid"`` and a model name, with no model run
-- and warned on every ``chunk`` call. Every chunk's ``multilang`` metadata
named the model as well, MORPHOLOGICAL runs included, and a HYBRID run
encoded each chunk a second time, one call per chunk, and dropped the
vectors. These tests pin what a result and each chunk must say about how
they were produced, with and without a (fake, offline) model.
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time
import types
import weakref

import pytest

from .. import _semantic as S
from .._multilang_mixin import MultilangConfig
from .._semantic import SemanticBackend, SemanticChunker, SemanticChunkerConfig

logger = S.logger  # logging.getLogger(S.__name__)

TEXT = "First idea here. Second idea there. Third topic now."


def _install_model(monkeypatch, encode):
    """Install an offline ``sentence_transformers`` whose model calls *encode*."""

    class _Model:
        def __init__(self, name, device=None):
            self.name = name

        def encode(self, texts, **kwargs):
            return encode(list(texts))

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _chunk_models(result):
    """Return the set of (model_name, script_model_version) over all chunks.

    ``MultilangChunkMeta.to_dict`` omits ``None`` fields, so "no model" is an
    absent key; ``.get`` reads it as ``None``.
    """
    return {
        (
            c.metadata["multilang"].get("model_name"),
            c.metadata["multilang"].get("script_model_version"),
        )
        for c in result.chunks
    }


@pytest.fixture
def no_model(monkeypatch):
    """Make ``sentence_transformers`` unimportable, whatever is installed."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


@pytest.fixture
def fake_model(monkeypatch):
    """Provide a deterministic, offline model; return the list of its calls."""
    calls = []

    def encode(texts):
        calls.append(len(texts))
        return [[float(len(text)), 1.0] for text in texts]

    _install_model(monkeypatch, encode)
    return calls


@pytest.fixture
def broken_model(monkeypatch):
    """Provide a model that loads and then fails at encode time."""

    def encode(texts):
        raise RuntimeError("CUDA out of memory")

    _install_model(monkeypatch, encode)


@pytest.fixture
def unloadable_model(monkeypatch):
    """Provide a package whose model fails to load; return the attempt count."""
    attempts = []

    class _Model:
        def __init__(self, name, device=None):
            attempts.append(name)
            raise OSError("no network for the first download")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return attempts


@pytest.mark.parametrize(
    "requested", [SemanticBackend.HYBRID, SemanticBackend.EMBEDDING]
)
def test_a_fallback_reports_the_backend_that_ran(no_model, requested):
    result = SemanticChunker(SemanticChunkerConfig(backend=requested)).chunk(TEXT)
    assert result.metadata["backend"] == "morphological"
    assert result.metadata["requested_backend"] == requested.value
    assert result.metadata["fallback_reason"] == "sentence_transformers_unavailable"
    assert result.metadata["model_name"] is None  # no model produced these chunks
    assert _chunk_models(result) == {(None, None)}  # nor any one of them
    assert {c.metadata["chunking_strategy"] for c in result.chunks} == {
        "semantic_morphological"
    }


def test_a_forbidden_fallback_raises(no_model):
    chunker = SemanticChunker(
        SemanticChunkerConfig(backend=SemanticBackend.HYBRID, allow_fallback=False)
    )
    with pytest.raises(ImportError, match="sentence-transformers"):
        chunker.chunk(TEXT)


def test_the_fallback_is_reported_once_per_chunker(no_model, caplog):
    chunker = SemanticChunker(SemanticChunkerConfig(backend=SemanticBackend.HYBRID))

    with caplog.at_level("WARNING", logger=logger.name):
        logger.addHandler(caplog.handler)
        try:
            for _ in range(5):
                chunker.chunk(TEXT)
        finally:
            logger.removeHandler(caplog.handler)

    warnings = [r for r in caplog.records if r.name == logger.name]
    assert len(warnings) == 1
    assert "hybrid" in warnings[0].getMessage()


def test_morphological_by_request_claims_no_model(fake_model):
    """Even with a model installed, a MORPHOLOGICAL run names none."""
    result = SemanticChunker(
        SemanticChunkerConfig(backend=SemanticBackend.MORPHOLOGICAL)
    ).chunk(TEXT)
    meta = result.metadata
    assert (meta["backend"], meta["requested_backend"]) == ("morphological",) * 2
    assert meta["fallback_reason"] is None
    assert meta["model_name"] is None
    assert _chunk_models(result) == {(None, None)}
    assert fake_model == []  # and runs none


@pytest.mark.parametrize(
    ("version", "tag"), [(None, "fake-model"), ("1.2", "fake-model@1.2")]
)
def test_a_loaded_model_is_reported_as_used(fake_model, version, tag):
    config = SemanticChunkerConfig(
        backend=SemanticBackend.HYBRID, model_name="fake-model", model_version=version
    )
    result = SemanticChunker(config).chunk(TEXT)
    meta = result.metadata
    assert (meta["backend"], meta["requested_backend"]) == ("hybrid", "hybrid")
    assert meta["fallback_reason"] is None
    assert (meta["model_name"], meta["model_version"]) == ("fake-model", version)
    assert _chunk_models(result) == {("fake-model", tag)}
    assert {c.metadata["chunking_strategy"] for c in result.chunks} == {
        "semantic_hybrid"
    }


def test_one_chunk_call_encodes_once(fake_model):
    """The refinement batch is the only model call; nothing is re-encoded."""
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    result = SemanticChunker(SemanticChunkerConfig(backend="hybrid")).chunk(text)
    assert len(result.chunks) > 1
    assert len(fake_model) == 1


def test_a_failing_model_falls_back_and_says_so(broken_model, caplog):
    chunker = SemanticChunker(SemanticChunkerConfig(backend=SemanticBackend.HYBRID))

    with caplog.at_level("WARNING", logger=logger.name):
        logger.addHandler(caplog.handler)
        try:
            result = chunker.chunk(TEXT)
        finally:
            logger.removeHandler(caplog.handler)

    messages = [
        r.getMessage()
        for r in caplog.records
        if r.name == logger.name and "embedding model failed" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "RuntimeError: CUDA out of memory" in messages[0]

    meta = result.metadata
    assert (meta["backend"], meta["requested_backend"]) == ("morphological", "hybrid")
    assert meta["fallback_reason"] == "embedding_failed"
    assert meta["model_name"] is None
    assert _chunk_models(result) == {(None, None)}
    assert {c.metadata["chunking_strategy"] for c in result.chunks} == {
        "semantic_morphological"
    }


def test_a_failing_model_raises_when_fallback_is_forbidden(broken_model):
    chunker = SemanticChunker(
        SemanticChunkerConfig(backend=SemanticBackend.HYBRID, allow_fallback=False)
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        chunker.chunk(TEXT)


@pytest.mark.parametrize("value", ["morphological", "hybrid", "embedding"])
def test_the_backend_may_be_given_by_value(no_model, value):
    result = SemanticChunker(SemanticChunkerConfig(backend=value)).chunk(TEXT)
    assert result.metadata["requested_backend"] == value


def test_an_unknown_backend_is_refused_at_construction():
    with pytest.raises(ValueError, match="backend must be one of"):
        SemanticChunker(SemanticChunkerConfig(backend="fastest"))


def test_a_model_that_fails_to_load_falls_back_and_says_so(unloadable_model):
    """Every call names the real reason, and the load is not retried."""
    chunker = SemanticChunker(SemanticChunkerConfig(backend=SemanticBackend.HYBRID))
    for _ in range(3):
        meta = chunker.chunk(TEXT).metadata
        assert (meta["backend"], meta["fallback_reason"]) == (
            "morphological",
            "model_load_failed",
        )
        assert meta["model_name"] is None
    assert len(unloadable_model) == 1


def test_a_model_that_fails_to_load_raises_when_fallback_is_forbidden(unloadable_model):
    chunker = SemanticChunker(
        SemanticChunkerConfig(backend=SemanticBackend.HYBRID, allow_fallback=False)
    )
    for _ in range(2):  # the remembered failure raises again, chained
        with pytest.raises(RuntimeError, match="failed to load") as caught:
            chunker.chunk(TEXT)
        assert isinstance(caught.value.__cause__, OSError)


# -- the load state machine (independent review, round 33) --------------------


class _BrokenPackage(types.ModuleType):
    """A ``sentence_transformers`` whose import fails with a non-ImportError."""

    def __getattr__(self, name):
        raise OSError("libcudart.so.12: cannot open shared object file")


def test_a_broken_package_import_falls_back_and_says_so(monkeypatch):
    """It used to escape once, then every call raised ``TypeError``."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", _BrokenPackage("st"))
    chunker = SemanticChunker(SemanticChunkerConfig(backend=SemanticBackend.HYBRID))
    for _ in range(2):
        meta = chunker.chunk(TEXT).metadata
        assert (meta["backend"], meta["fallback_reason"]) == (
            "morphological",
            "model_load_failed",
        )


def test_a_broken_package_import_raises_when_fallback_is_forbidden(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", _BrokenPackage("st"))
    chunker = SemanticChunker(
        SemanticChunkerConfig(backend=SemanticBackend.HYBRID, allow_fallback=False)
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="failed to load") as caught:
            chunker.chunk(TEXT)
        assert isinstance(caught.value.__cause__, OSError)


def test_an_interrupted_first_load_is_tried_again(monkeypatch):
    """An interrupt is not an outcome; it used to leave the chunker broken."""
    attempts = []

    def encode(texts):
        return [[float(len(text)), 1.0] for text in texts]

    class _Model:
        def __init__(self, name, device=None):
            attempts.append(name)
            if len(attempts) == 1:
                raise KeyboardInterrupt

        def encode(self, texts, **kwargs):
            return encode(list(texts))

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    chunker = SemanticChunker(SemanticChunkerConfig(backend=SemanticBackend.HYBRID))
    with pytest.raises(KeyboardInterrupt):
        chunker.chunk(TEXT)
    meta = chunker.chunk(TEXT).metadata
    assert (meta["backend"], meta["fallback_reason"]) == ("hybrid", None)
    assert len(attempts) == 2  # noqa: PLR2004 - interrupted, then loaded


def test_concurrent_first_calls_share_one_load(monkeypatch):
    """A second thread used to read a half-set state and raise ``TypeError``."""
    attempts = []

    class _SlowModel:
        def __init__(self, name, device=None):
            attempts.append(name)
            time.sleep(0.01)

        def encode(self, texts, **kwargs):
            return [[float(len(text)), 1.0] for text in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _SlowModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    chunker = SemanticChunker(SemanticChunkerConfig(backend=SemanticBackend.HYBRID))
    outcomes = []

    def run():
        try:
            outcomes.append(chunker.chunk(TEXT).metadata["backend"])
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            outcomes.append(repr(exc))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == ["hybrid"] * 4
    assert len(attempts) == 1


def test_a_failed_load_does_not_keep_the_half_loaded_model(monkeypatch):
    """The remembered exception kept the failed frame, and its locals, alive."""
    alive = []

    class _Weights:
        """Stands in for tensors allocated before the load failed."""

    class _Model:
        def __init__(self, name, device=None):
            weights = _Weights()
            alive.append(weakref.ref(weights))
            raise OSError("CUDA out of memory")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    chunker = SemanticChunker(SemanticChunkerConfig(backend=SemanticBackend.HYBRID))
    assert chunker.chunk(TEXT).metadata["fallback_reason"] == "model_load_failed"
    gc.collect()
    assert alive[0]() is None


def test_a_single_unit_text_calls_and_names_no_model(fake_model):
    result = SemanticChunker(SemanticChunkerConfig(backend="hybrid")).chunk("Hello")
    meta = result.metadata
    assert (meta["backend"], meta["fallback_reason"]) == ("hybrid", None)
    assert (meta["model_invoked"], meta["model_name"]) == (False, None)
    assert _chunk_models(result) == {(None, None)}
    assert fake_model == []  # nothing to refine, so nothing encoded


def test_a_declared_model_is_recorded_as_declared(fake_model):
    declared = MultilangConfig(embedding_model_name="e5", embedding_model_version="2")
    morphological = SemanticChunker(
        SemanticChunkerConfig(backend="morphological", multilang_config=declared)
    ).chunk(TEXT)
    assert _chunk_models(morphological) == {("e5", "e5@2")}
    assert morphological.metadata["model_name"] is None  # no model was called
    hybrid = SemanticChunker(
        SemanticChunkerConfig(
            backend="hybrid", model_name="fake-model", multilang_config=declared
        )
    ).chunk(TEXT)
    assert _chunk_models(hybrid) == {("fake-model", "fake-model")}
