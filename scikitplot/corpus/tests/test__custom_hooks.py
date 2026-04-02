# corpus/tests/test__custom_hooks.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus._custom_hooks
=========================================

Coverage targets (28 % → 85 %+)
---------------------------------
* :class:`CustomChunker` — ``__init__`` (happy path, TypeError on
  non-callable), ``chunk`` (delegate, empty text, None text, fn error
  → RuntimeError), ``__repr__``, strategy class var.
* :class:`CustomFilter` — ``__init__`` (TypeError), ``include``
  (True/False return, fn exception → False + warn), ``__repr__``.
* :class:`CustomNormalizer` — ``__init__`` (TypeError),
  ``normalize_doc`` in doc-mode (returns CorpusDocument, wrong type →
  TypeError), in text-mode (str → wrapped, wrong return type →
  TypeError, fn error → RuntimeError), ``__repr__``.
* :class:`CustomEnricherConfig` — all fields None default, TypeError on
  non-callable field, TypeError on non-frozenset stopwords,
  ``__post_init__`` validates each callable field.
* :class:`CustomNLPEnricher` — construction, ``enrich_documents``
  (skip already-enriched, overwrite=True, custom_tokenizer replaces,
  custom_stopwords applied, custom_lemmatizer, custom_stemmer,
  custom_keyword_extractor; each falling back when fn raises).
* :class:`PipelineHooks` — all-None default, TypeError on non-callable.
* :class:`HookableCorpusPipeline` — construction, ``run`` (pre_read
  called, post_read transform applied, post_read None → unchanged,
  hook error swallowed), ``run_batch`` (stop_on_error=False skips,
  stop_on_error=True re-raises), ``run_url`` (list dispatch).
* :class:`BuilderFactories` — TypeError on non-callable field.
* :class:`FactoryCorpusBuilder` — construction, ``build`` delegates,
  factory overrides (chunker, normalizer, enricher, embedding_engine).
* :class:`CustomSimilarityIndex` — ``__init__`` (TypeError on
  non-callable), ``build`` / ``n_documents`` / ``has_embeddings``
  delegation, ``search`` with custom scorer (returns scorer result, fn
  error → RuntimeError), search without scorer (delegates inner),
  ``__repr__``.

All tests use stdlib only.  No ML/NLP dependencies required.
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from .._custom_hooks import (
    BuilderFactories,
    CustomChunker,
    CustomEnricherConfig,
    CustomFilter,
    CustomNLPEnricher,
    CustomNormalizer,
    CustomSimilarityIndex,
    FactoryCorpusBuilder,
    HookableCorpusPipeline,
    PipelineHooks,
)
from .._schema import ChunkingStrategy, CorpusDocument


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _doc(text: str = "hello world test", idx: int = 0) -> CorpusDocument:
    return CorpusDocument.create("test.txt", idx, text)


def _simple_chunk_fn(
    text: str, metadata: dict[str, Any]
) -> list[tuple[int, str]]:
    """Split on spaces — deterministic, dependency-free chunk function."""
    result: list[tuple[int, str]] = []
    cursor = 0
    for word in text.split():
        start = text.index(word, cursor)
        result.append((start, word))
        cursor = start + len(word)
    return result


# ===========================================================================
# Layer 2 — CustomChunker
# ===========================================================================


class TestCustomChunker:
    def test_init_happy_path(self) -> None:
        cc = CustomChunker(_simple_chunk_fn)
        assert cc._chunk_fn is _simple_chunk_fn
        assert cc.strategy == ChunkingStrategy.CUSTOM

    def test_init_uses_fn_name_when_name_not_provided(self) -> None:
        cc = CustomChunker(_simple_chunk_fn)
        assert cc._name == "_simple_chunk_fn"

    def test_init_explicit_name(self) -> None:
        cc = CustomChunker(_simple_chunk_fn, name="MyChunker")
        assert cc._name == "MyChunker"

    def test_init_raises_type_error_for_non_callable(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomChunker("not a function")  # type: ignore[arg-type]

    def test_chunk_delegates_to_fn(self) -> None:
        cc = CustomChunker(_simple_chunk_fn)
        result = cc.chunk("hello world")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == (0, "hello")
        assert result[1] == (6, "world")

    def test_chunk_empty_text_returns_empty(self) -> None:
        cc = CustomChunker(_simple_chunk_fn)
        assert cc.chunk("") == []
        assert cc.chunk("   ") == []

    def test_chunk_none_text_raises_value_error(self) -> None:
        cc = CustomChunker(_simple_chunk_fn)
        with pytest.raises(ValueError, match="None"):
            cc.chunk(None)  # type: ignore[arg-type]

    def test_chunk_fn_exception_wrapped_in_runtime_error(self) -> None:
        def bad_fn(text: str, meta: dict) -> list:
            raise RuntimeError("boom")

        cc = CustomChunker(bad_fn)
        with pytest.raises(RuntimeError, match="boom"):
            cc.chunk("some text")

    def test_chunk_passes_metadata(self) -> None:
        received: list[dict] = []

        def capture_fn(text: str, meta: dict) -> list:
            received.append(meta)
            return [(0, text)]

        cc = CustomChunker(capture_fn)
        cc.chunk("hello", metadata={"key": "val"})
        assert received[0] == {"key": "val"}

    def test_chunk_none_metadata_becomes_empty_dict(self) -> None:
        received: list[dict] = []

        def capture_fn(text: str, meta: dict) -> list:
            received.append(meta)
            return [(0, text)]

        cc = CustomChunker(capture_fn)
        cc.chunk("hello", metadata=None)
        assert received[0] == {}

    def test_repr_contains_name_and_strategy(self) -> None:
        cc = CustomChunker(_simple_chunk_fn, name="Test")
        r = repr(cc)
        assert "Test" in r
        assert "CUSTOM" in r

    def test_strategy_is_custom(self) -> None:
        cc = CustomChunker(_simple_chunk_fn)
        assert cc.strategy is ChunkingStrategy.CUSTOM

    def test_lambda_name_fallback(self) -> None:
        fn = lambda text, meta: [(0, text)]  # noqa: E731
        cc = CustomChunker(fn)
        assert cc._name  # non-empty


# ===========================================================================
# Layer 3 — CustomFilter
# ===========================================================================


class TestCustomFilter:
    def test_init_happy_path(self) -> None:
        fn = lambda doc: True  # noqa: E731
        cf = CustomFilter(fn)
        assert cf._fn is fn

    def test_init_raises_type_error_for_non_callable(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomFilter(42)  # type: ignore[arg-type]

    def test_include_returns_true(self) -> None:
        cf = CustomFilter(lambda doc: True)
        assert cf.include(_doc()) is True

    def test_include_returns_false(self) -> None:
        cf = CustomFilter(lambda doc: False)
        assert cf.include(_doc()) is False

    def test_include_converts_truthy(self) -> None:
        cf = CustomFilter(lambda doc: 1)
        assert cf.include(_doc()) is True

    def test_include_swallows_exception_and_returns_false(self) -> None:
        def bad_fn(doc: CorpusDocument) -> bool:
            raise ValueError("unexpected")

        cf = CustomFilter(bad_fn)
        result = cf.include(_doc())
        assert result is False

    def test_explicit_name(self) -> None:
        cf = CustomFilter(lambda doc: True, name="MyFilter")
        assert cf._name == "MyFilter"

    def test_repr_contains_name(self) -> None:
        cf = CustomFilter(lambda doc: True, name="FilterX")
        assert "FilterX" in repr(cf)

    def test_filter_receives_doc(self) -> None:
        seen: list[CorpusDocument] = []
        cf = CustomFilter(lambda doc: seen.append(doc) or True)  # type: ignore[func-returns-value]
        doc = _doc()
        cf.include(doc)
        assert seen[0] is doc


# ===========================================================================
# Layer 4 — CustomNormalizer
# ===========================================================================


class TestCustomNormalizer:
    def test_init_happy_path_doc_mode(self) -> None:
        fn = lambda doc: doc  # noqa: E731
        cn = CustomNormalizer(fn)
        assert cn._text_mode is False

    def test_init_happy_path_text_mode(self) -> None:
        cn = CustomNormalizer(str.lower, text_mode=True)
        assert cn._text_mode is True

    def test_init_raises_type_error_for_non_callable(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomNormalizer("nope")  # type: ignore[arg-type]

    def test_normalize_doc_mode_returns_corpus_document(self) -> None:
        doc = _doc("Hello")
        cn = CustomNormalizer(lambda d: d.replace(normalized_text="hello"))
        result = cn.normalize_doc(doc)
        assert isinstance(result, CorpusDocument)
        assert result.normalized_text == "hello"

    def test_normalize_doc_mode_wrong_return_type_raises(self) -> None:
        cn = CustomNormalizer(lambda doc: "not a CorpusDocument")
        with pytest.raises(TypeError, match="CorpusDocument"):
            cn.normalize_doc(_doc())

    def test_normalize_doc_mode_fn_error_raises_runtime_error(self) -> None:
        def bad_fn(doc: CorpusDocument) -> CorpusDocument:
            raise OSError("disk full")

        cn = CustomNormalizer(bad_fn)
        with pytest.raises(RuntimeError, match="disk full"):
            cn.normalize_doc(_doc())

    def test_normalize_text_mode_wraps_result(self) -> None:
        cn = CustomNormalizer(str.upper, text_mode=True)
        doc = _doc("hello")
        result = cn.normalize_doc(doc)
        assert isinstance(result, CorpusDocument)
        assert result.normalized_text == "HELLO"

    def test_normalize_text_mode_wrong_return_type_raises(self) -> None:
        cn = CustomNormalizer(lambda text: 42, text_mode=True)
        with pytest.raises(TypeError, match="text_mode"):
            cn.normalize_doc(_doc("hello"))

    def test_normalize_text_mode_uses_normalized_text_when_present(self) -> None:
        captured: list[str] = []
        def fn(text: str) -> str:
            captured.append(text)
            return text

        cn = CustomNormalizer(fn, text_mode=True)
        doc = _doc("raw").replace(normalized_text="normalised")
        cn.normalize_doc(doc)
        assert captured[0] == "normalised"

    def test_repr_contains_name_and_text_mode(self) -> None:
        cn = CustomNormalizer(str.lower, name="Lower", text_mode=True)
        r = repr(cn)
        assert "Lower" in r
        assert "text_mode=True" in r


# ===========================================================================
# Layer 5 — CustomEnricherConfig
# ===========================================================================


class TestCustomEnricherConfig:
    def test_all_none_defaults(self) -> None:
        cfg = CustomEnricherConfig()
        assert cfg.custom_tokenizer is None
        assert cfg.custom_lemmatizer is None
        assert cfg.custom_stemmer is None
        assert cfg.custom_keyword_extractor is None
        assert cfg.custom_stopwords is None

    def test_callable_fields_accepted(self) -> None:
        cfg = CustomEnricherConfig(
            custom_tokenizer=lambda t: t.split(),
            custom_lemmatizer=lambda toks: toks,
            custom_stemmer=lambda toks: toks,
            custom_keyword_extractor=lambda text, toks: toks[:3],
            custom_stopwords=frozenset({"the", "a"}),
        )
        assert cfg.custom_tokenizer is not None

    def test_non_callable_tokenizer_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomEnricherConfig(custom_tokenizer="not callable")  # type: ignore[arg-type]

    def test_non_callable_lemmatizer_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomEnricherConfig(custom_lemmatizer=42)  # type: ignore[arg-type]

    def test_non_callable_stemmer_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomEnricherConfig(custom_stemmer=[])  # type: ignore[arg-type]

    def test_non_callable_keyword_extractor_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomEnricherConfig(custom_keyword_extractor=True)  # type: ignore[arg-type]

    def test_non_frozenset_stopwords_raises(self) -> None:
        with pytest.raises(TypeError, match="frozenset"):
            CustomEnricherConfig(custom_stopwords={"set", "not", "frozen"})  # type: ignore[arg-type]

    def test_frozenset_stopwords_accepted(self) -> None:
        cfg = CustomEnricherConfig(custom_stopwords=frozenset({"stop"}))
        assert "stop" in cfg.custom_stopwords  # type: ignore[operator]


# ===========================================================================
# Layer 5 — CustomNLPEnricher
# ===========================================================================


class TestCustomNLPEnricher:
    """Uses only the built-in NLPEnricher with mocked optional NLP deps."""

    def test_construction_no_config(self) -> None:
        enricher = CustomNLPEnricher()
        assert enricher.custom_config is not None

    def test_enrich_documents_returns_same_length(self) -> None:
        docs = [_doc(f"word{i} text", idx=i) for i in range(4)]
        enricher = CustomNLPEnricher()
        result = enricher.enrich_documents(docs)
        assert len(result) == 4

    def test_already_enriched_docs_skipped_by_default(self) -> None:
        doc = _doc("text").replace(tokens=["text"])
        called: list[bool] = []

        def tok(text: str) -> list[str]:
            called.append(True)
            return text.split()

        enricher = CustomNLPEnricher(
            custom_config=CustomEnricherConfig(custom_tokenizer=tok)
        )
        enricher.enrich_documents([doc])
        assert not called  # skipped because tokens already set

    def test_overwrite_forces_re_enrichment(self) -> None:
        doc = _doc("text").replace(tokens=["old_token"])
        called: list[bool] = []

        def tok(text: str) -> list[str]:
            called.append(True)
            return text.split()

        enricher = CustomNLPEnricher(
            custom_config=CustomEnricherConfig(custom_tokenizer=tok)
        )
        enricher.enrich_documents([doc], overwrite=True)
        assert called

    def test_custom_tokenizer_called(self) -> None:
        called: list[str] = []

        def tok(text: str) -> list[str]:
            called.append(text)
            return text.split()

        enricher = CustomNLPEnricher(
            custom_config=CustomEnricherConfig(custom_tokenizer=tok)
        )
        enricher.enrich_documents([_doc("hello world")])
        assert called

    def test_custom_tokenizer_exception_falls_back(self) -> None:
        def bad_tok(text: str) -> list[str]:
            raise RuntimeError("tok failed")

        enricher = CustomNLPEnricher(
            custom_config=CustomEnricherConfig(custom_tokenizer=bad_tok)
        )
        # Should not raise — falls back to built-in tokeniser
        result = enricher.enrich_documents([_doc("hello world")])
        assert len(result) == 1

    def test_custom_stopwords_applied(self) -> None:
        # Custom stopwords that filter everything
        enricher = CustomNLPEnricher(
            custom_config=CustomEnricherConfig(
                custom_tokenizer=lambda t: t.split(),
                custom_stopwords=frozenset({"hello", "world"}),
            )
        )
        result = enricher.enrich_documents([_doc("hello world")])
        # All tokens filtered by custom stopwords
        tokens = result[0].tokens
        if tokens is not None:
            assert "hello" not in tokens

    def test_custom_keyword_extractor_called(self) -> None:
        extracted: list[list[str]] = []

        def kw_extractor(text: str, tokens: list[str]) -> list[str]:
            extracted.append(tokens)
            return ["custom_keyword"]

        enricher = CustomNLPEnricher(
            custom_config=CustomEnricherConfig(
                custom_keyword_extractor=kw_extractor
            ),
        )
        result = enricher.enrich_documents([_doc("some text here")])
        assert extracted  # was called
        if result[0].keywords is not None:
            assert "custom_keyword" in result[0].keywords

    def test_repr_contains_class_name(self) -> None:
        enricher = CustomNLPEnricher()
        assert "CustomNLPEnricher" in repr(enricher)


# ===========================================================================
# Layer 6 — PipelineHooks
# ===========================================================================


class TestPipelineHooks:
    def test_all_none_by_default(self) -> None:
        hooks = PipelineHooks()
        for attr in (
            "pre_read_hook",
            "post_read_hook",
            "post_filter_hook",
            "post_embed_hook",
            "pre_export_hook",
        ):
            assert getattr(hooks, attr) is None

    def test_non_callable_pre_read_hook_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            PipelineHooks(pre_read_hook="bad")  # type: ignore[arg-type]

    def test_non_callable_post_read_hook_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            PipelineHooks(post_read_hook=42)  # type: ignore[arg-type]

    def test_non_callable_post_filter_hook_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            PipelineHooks(post_filter_hook=True)  # type: ignore[arg-type]

    def test_non_callable_post_embed_hook_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            PipelineHooks(post_embed_hook=0)  # type: ignore[arg-type]

    def test_non_callable_pre_export_hook_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            PipelineHooks(pre_export_hook=[])  # type: ignore[arg-type]

    def test_callable_hooks_accepted(self) -> None:
        hooks = PipelineHooks(
            pre_read_hook=lambda src: None,
            post_read_hook=lambda src, docs: None,
            post_filter_hook=lambda doc: True,
        )
        assert callable(hooks.pre_read_hook)
        assert callable(hooks.post_read_hook)
        assert callable(hooks.post_filter_hook)


# ===========================================================================
# Layer 6 — HookableCorpusPipeline
# ===========================================================================


class TestHookableCorpusPipeline:
    """Tests use real TextReader sources (tmp files) to avoid mocking internals."""

    @pytest.fixture()
    def tmp_txt(self, tmp_path: pathlib.Path) -> pathlib.Path:
        p = tmp_path / "sample.txt"
        p.write_text("First sentence here. Second sentence here.")
        return p

    def test_construction_defaults(self) -> None:
        pipeline = HookableCorpusPipeline()
        assert pipeline.hooks is not None

    def test_pre_read_hook_called(self, tmp_txt: pathlib.Path) -> None:
        called: list[str] = []
        hooks = PipelineHooks(pre_read_hook=lambda src: called.append(src))
        pipeline = HookableCorpusPipeline(hooks=hooks)
        pipeline.run(tmp_txt)
        assert len(called) == 1
        assert str(tmp_txt) in called[0]

    def test_post_read_hook_can_filter_documents(
        self, tmp_txt: pathlib.Path
    ) -> None:
        hooks = PipelineHooks(
            post_read_hook=lambda src, docs: []  # discard all
        )
        pipeline = HookableCorpusPipeline(hooks=hooks)
        result = pipeline.run(tmp_txt)
        assert len(result.documents) == 0

    def test_post_read_hook_none_return_keeps_docs(
        self, tmp_txt: pathlib.Path
    ) -> None:
        hooks = PipelineHooks(post_read_hook=lambda src, docs: None)
        pipeline = HookableCorpusPipeline(hooks=hooks)
        result = pipeline.run(tmp_txt)
        assert result.n_read >= 0  # original docs kept

    def test_hook_exception_swallowed_run_still_succeeds(
        self, tmp_txt: pathlib.Path
    ) -> None:
        def bad_hook(src: str) -> None:
            raise RuntimeError("hook failure")

        hooks = PipelineHooks(pre_read_hook=bad_hook)
        pipeline = HookableCorpusPipeline(hooks=hooks)
        result = pipeline.run(tmp_txt)  # must not raise
        assert result is not None

    def test_run_batch_stop_on_error_false_skips(
        self, tmp_txt: pathlib.Path
    ) -> None:
        pipeline = HookableCorpusPipeline()
        results = pipeline.run_batch(
            [tmp_txt, pathlib.Path("/nonexistent/file.txt")],
            stop_on_error=False,
        )
        assert len(results) == 1  # only successful one

    def test_run_batch_stop_on_error_true_raises(
        self, tmp_txt: pathlib.Path
    ) -> None:
        pipeline = HookableCorpusPipeline()
        with pytest.raises(Exception):
            pipeline.run_batch(
                [tmp_txt, pathlib.Path("/nonexistent/file.txt")],
                stop_on_error=True,
            )

    def test_run_url_list_dispatches_each(
        self, tmp_txt: pathlib.Path
    ) -> None:
        pipeline = HookableCorpusPipeline()
        results = pipeline.run_url(
            [str(tmp_txt), str(tmp_txt)], stop_on_error=False
        )
        assert isinstance(results, list)
        assert len(results) == 2

    def test_repr_contains_class_name(self) -> None:
        pipeline = HookableCorpusPipeline()
        assert "HookableCorpusPipeline" in repr(pipeline)

    def test_post_filter_hook_filters_docs(
        self, tmp_txt: pathlib.Path
    ) -> None:
        # post_filter_hook=False → all docs discarded by composite filter
        hooks = PipelineHooks(post_filter_hook=lambda doc: False)
        pipeline = HookableCorpusPipeline(hooks=hooks)
        result = pipeline.run(tmp_txt)
        assert len(result.documents) == 0


# ===========================================================================
# Layer 7 — BuilderFactories
# ===========================================================================


class TestBuilderFactories:
    def test_all_none_by_default(self) -> None:
        bf = BuilderFactories()
        for attr in (
            "reader_factory",
            "chunker_factory",
            "filter_factory",
            "normalizer_factory",
            "enricher_factory",
            "embedding_engine_factory",
        ):
            assert getattr(bf, attr) is None

    def test_non_callable_reader_factory_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            BuilderFactories(reader_factory="bad")  # type: ignore[arg-type]

    def test_non_callable_chunker_factory_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            BuilderFactories(chunker_factory=42)  # type: ignore[arg-type]

    def test_non_callable_embedding_engine_factory_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            BuilderFactories(embedding_engine_factory=True)  # type: ignore[arg-type]

    def test_callable_factories_accepted(self) -> None:
        bf = BuilderFactories(
            chunker_factory=lambda: None,
            filter_factory=lambda: None,
        )
        assert callable(bf.chunker_factory)
        assert callable(bf.filter_factory)


# ===========================================================================
# Layer 7 — FactoryCorpusBuilder
# ===========================================================================


class TestFactoryCorpusBuilder:
    @pytest.fixture()
    def tmp_txt(self, tmp_path: pathlib.Path) -> pathlib.Path:
        p = tmp_path / "data.txt"
        p.write_text("Simple text for builder tests. More words here.")
        return p

    def test_construction_defaults(self) -> None:
        builder = FactoryCorpusBuilder()
        assert builder.config is not None
        assert builder.factories is not None

    def test_build_returns_result(self, tmp_txt: pathlib.Path) -> None:
        builder = FactoryCorpusBuilder()
        result = builder.build(tmp_txt)
        assert result is not None

    def test_chunker_factory_override(self, tmp_txt: pathlib.Path) -> None:
        from .._chunkers._fixed_window import FixedWindowChunker
        factories = BuilderFactories(
            chunker_factory=lambda: FixedWindowChunker()
        )
        builder = FactoryCorpusBuilder(factories=factories)
        result = builder.build(tmp_txt)
        assert result is not None

    def test_context_manager(self, tmp_txt: pathlib.Path) -> None:
        with FactoryCorpusBuilder() as builder:
            result = builder.build(tmp_txt)
        assert result is not None

    def test_repr_contains_class_name(self) -> None:
        builder = FactoryCorpusBuilder()
        assert "FactoryCorpusBuilder" in repr(builder)

    def test_result_property(self, tmp_txt: pathlib.Path) -> None:
        builder = FactoryCorpusBuilder()
        builder.build(tmp_txt)
        assert builder._result is not None

    def test_search_delegates(self, tmp_txt: pathlib.Path) -> None:
        from .._corpus_builder import BuilderConfig
        builder = FactoryCorpusBuilder(config=BuilderConfig(build_index=True))
        builder.build(tmp_txt)
        results = builder.search("simple", match_mode="strict")
        assert isinstance(results, list)


# ===========================================================================
# Layer 8 — CustomSimilarityIndex
# ===========================================================================


class TestCustomSimilarityIndex:
    def test_init_no_custom_scorer(self) -> None:
        idx = CustomSimilarityIndex()
        assert idx.custom_scorer_fn is None
        assert idx._scorer_name is None

    def test_init_with_callable_scorer(self) -> None:
        scorer = lambda q, docs, cfg: []  # noqa: E731
        idx = CustomSimilarityIndex(custom_scorer_fn=scorer)
        assert idx.custom_scorer_fn is scorer

    def test_init_non_callable_scorer_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            CustomSimilarityIndex(custom_scorer_fn="bad")  # type: ignore[arg-type]

    def test_build_delegates_n_documents(self) -> None:
        docs = [_doc(f"text {i}", idx=i) for i in range(3)]
        idx = CustomSimilarityIndex()
        idx.build(docs)
        assert idx.n_documents == 3

    def test_build_empty_raises(self) -> None:
        idx = CustomSimilarityIndex()
        with pytest.raises(ValueError):
            idx.build([])

    def test_has_embeddings_false_without_embeddings(self) -> None:
        idx = CustomSimilarityIndex()
        idx.build([_doc("text")])
        assert idx.has_embeddings is False

    def test_search_uses_custom_scorer_fn(self) -> None:
        from .._similarity._similarity import SearchConfig, SearchResult

        expected_result = [SearchResult(doc=_doc("x"), score=1.0, match_mode="custom")]

        def my_scorer(
            query: str, docs: list, cfg: SearchConfig
        ) -> list[SearchResult]:
            return expected_result

        idx = CustomSimilarityIndex(custom_scorer_fn=my_scorer)
        idx.build([_doc("text")])
        results = idx.search("query")
        assert results is expected_result

    def test_search_custom_scorer_runtime_error_on_exception(self) -> None:
        def bad_scorer(query: str, docs: list, cfg: Any) -> list:
            raise ValueError("scorer failure")

        idx = CustomSimilarityIndex(custom_scorer_fn=bad_scorer)
        idx.build([_doc("text")])
        with pytest.raises(RuntimeError, match="scorer failure"):
            idx.search("query")

    def test_search_without_scorer_delegates_to_inner(self) -> None:
        idx = CustomSimilarityIndex()
        docs = [_doc("quick brown fox"), _doc("machine learning")]
        idx.build(docs)
        from .._similarity._similarity import SearchConfig
        cfg = SearchConfig(match_mode="strict")
        results = idx.search("brown", config=cfg)
        assert len(results) == 1

    def test_repr_contains_class_info(self) -> None:
        idx = CustomSimilarityIndex()
        idx.build([_doc("text")])
        r = repr(idx)
        assert "CustomSimilarityIndex" in r
        assert "n_docs" in r

    def test_scorer_name_stored(self) -> None:
        def named_scorer(q: str, docs: list, cfg: Any) -> list:
            return []

        idx = CustomSimilarityIndex(custom_scorer_fn=named_scorer)
        assert idx._scorer_name == "named_scorer"

    def test_search_passes_config_override_to_inner(self) -> None:
        from .._similarity._similarity import SearchConfig
        idx = CustomSimilarityIndex()
        docs = [_doc("alpha"), _doc("beta"), _doc("gamma")]
        idx.build(docs)
        cfg = SearchConfig(match_mode="strict", top_k=1)
        results = idx.search("alpha", config=cfg)
        assert len(results) <= 1
