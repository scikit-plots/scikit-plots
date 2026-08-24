# scikitplot/corpus/_runtime.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime materialization for :class:`CorpusPlan` / :class:`FluentCorpus`.

``CorpusPlan`` is deliberately immutable and side-effect free.  This module is
its operational boundary: :func:`materialize_plan` resolves supported plan
fragments into the existing Corpus runtime components, and :class:`RuntimeCorpus`
coordinates those components without reimplementing ingestion, chunking,
embedding, storage, retrieval, or export logic.

The separation is intentional::

    FluentCorpus -> CorpusPlan -> materialize_plan() -> RuntimeCorpus -> run()

Materialization constructs runtime objects but does not read the configured
source.  Source processing starts only when :meth:`RuntimeCorpus.run` or
:meth:`RuntimeCorpus.add` is called.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ._plan import DEFAULT_STAGES, CorpusPlan

if TYPE_CHECKING:
    from typing_extensions import Self

    from ._registry import ComponentRegistry

__all__ = [
    "RuntimeCorpus",
    "RuntimePolicy",
    "materialize_plan",
]


@dataclasses.dataclass(frozen=True)
class RuntimePolicy:
    """Execution policy applied by :class:`RuntimeCorpus`.

    Parameters
    ----------
    allow_network : bool, optional
        Allow ``http://`` / ``https://`` sources at execution time.  The
        offline-safe default is ``False``.  This policy is checked before the
        existing reader/URL security layer; it does not replace SSRF, redirect,
        size, timeout, or archive protections.
    """

    allow_network: bool = False


def _is_url(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith(("http://", "https://"))


def _is_source_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path))


def _validate_runtime_stages(plan: CorpusPlan) -> None:
    """Require an executable subset of the canonical Corpus stage order."""
    if plan.stages is None:
        return
    positions = [DEFAULT_STAGES.index(stage) for stage in plan.effective_stages]
    if positions != sorted(positions):
        raise ValueError(
            "RuntimeCorpus currently supports explicit stage omission but not "
            "stage reordering; stages must remain in canonical order "
            f"{DEFAULT_STAGES}, got {plan.effective_stages}."
        )


def _component_registry(registry: ComponentRegistry | None) -> ComponentRegistry:
    if registry is not None:
        return registry
    from ._registry import ComponentRegistry  # noqa: PLC0415

    resolved = ComponentRegistry()
    resolved.register_builtins()
    return resolved


def _resolve_reader_kwargs(value: Any) -> dict[str, Any]:
    """Resolve the plan's reader fragment to ``CorpusPipeline.reader_kwargs``."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)

    from ._base import DocumentReader  # noqa: PLC0415

    if isinstance(value, str) and value.strip().lower() in {
        "auto",
        "default",
        "document",
    }:
        return {}
    if isinstance(value, type) and issubclass(value, DocumentReader):
        # CorpusPipeline dispatches through the canonical DocumentReader factory;
        # naming the base class in a plan therefore selects that default path.
        if value is DocumentReader:
            return {}
        raise TypeError(
            "RuntimeCorpus cannot inject a custom DocumentReader class into "
            "CorpusPipeline yet; register the reader with DocumentReader or "
            "configure reader kwargs instead."
        )
    if (
        isinstance(value, tuple)
        and len(value) == 2  # ruff: ignore[magic-value-comparison]
        and isinstance(value[1], Mapping)
        and isinstance(value[0], type)
        and issubclass(value[0], DocumentReader)
    ):
        if value[0] is not DocumentReader:
            raise TypeError(
                "RuntimeCorpus reader tuples currently support the canonical "
                "DocumentReader factory only."
            )
        return dict(value[1])
    raise TypeError(
        "reader must be None, 'auto', DocumentReader, a reader-kwargs mapping, "
        "or (DocumentReader, kwargs)."
    )


def _resolve_chunker(value: Any, registry: ComponentRegistry | None) -> Any:
    if value is None:
        return None

    from ._base import ChunkerBase  # noqa: PLC0415
    from ._chunkers import (  # noqa: PLC0415
        FixedWindowChunker,
        FixedWindowChunkerConfig,
        ParagraphChunker,
        ParagraphChunkerConfig,
        SemanticChunker,
        SemanticChunkerConfig,
        SentenceChunker,
        SentenceChunkerConfig,
        WordChunker,
        WordChunkerConfig,
    )

    config_pairs = (
        (SentenceChunkerConfig, SentenceChunker),
        (ParagraphChunkerConfig, ParagraphChunker),
        (FixedWindowChunkerConfig, FixedWindowChunker),
        (WordChunkerConfig, WordChunker),
        (SemanticChunkerConfig, SemanticChunker),
    )
    for config_type, chunker_type in config_pairs:
        if isinstance(value, config_type):
            return chunker_type(value)

    if isinstance(value, str):
        return _component_registry(registry).build_chunker(value)
    if isinstance(value, type):
        return value()
    if isinstance(value, ChunkerBase) or callable(getattr(value, "chunk", None)):
        return value
    raise TypeError(
        "chunker must be a registered name, chunker config, chunker class, "
        "or chunker instance."
    )


def _resolve_normalizer(value: Any) -> Any:
    if value is None:
        return None
    from ._normalizers import TextNormalizer, TextNormalizerConfig  # noqa: PLC0415

    if isinstance(value, TextNormalizer):
        return value
    if isinstance(value, TextNormalizerConfig):
        return TextNormalizer(value)
    if isinstance(value, Mapping):
        return TextNormalizer(TextNormalizerConfig(**dict(value)))
    if isinstance(value, type) and issubclass(value, TextNormalizer):
        return value()
    if callable(getattr(value, "normalize_documents", None)):
        return value
    raise TypeError(
        "normalizer must be TextNormalizerConfig, TextNormalizer, a config "
        "mapping, or an object implementing normalize_documents()."
    )


def _resolve_enricher(  # ruff: ignore[too-many-return-statements]
    value: Any,
) -> Any:
    if value is None:
        return None
    from ._enrichers import (  # noqa: PLC0415
        EnricherConfig,
        NLPEnricher,
        SimpleEnricherSpec,
        SimpleFrequencyEnricher,
    )

    if isinstance(value, (NLPEnricher, SimpleFrequencyEnricher)):
        return value
    if isinstance(value, SimpleEnricherSpec):
        return SimpleFrequencyEnricher(value)
    if isinstance(value, EnricherConfig):
        return NLPEnricher(value)
    if isinstance(value, Mapping):
        return NLPEnricher(EnricherConfig(**dict(value)))
    if isinstance(value, type) and issubclass(value, NLPEnricher):
        return value()
    if callable(getattr(value, "enrich_documents", None)):
        return value
    raise TypeError(
        "enricher must be EnricherConfig, SimpleEnricherSpec, "
        "NLPEnricher, SimpleFrequencyEnricher, a config mapping, "
        "or an object implementing enrich_documents()."
    )


def _resolve_embedder(value: Any) -> Any:
    if value is None:
        return None
    from ._embeddings import EmbeddingEngine  # noqa: PLC0415

    if isinstance(value, EmbeddingEngine):
        return value
    if isinstance(value, str):
        return EmbeddingEngine(model_name=value)
    if isinstance(value, Mapping):
        return EmbeddingEngine(**dict(value))
    if callable(value):
        return EmbeddingEngine(backend="custom", custom_fn=value)
    raise TypeError(
        "embedder must be an EmbeddingEngine, model-name string, constructor "
        "mapping, or custom embedding callable."
    )


def _resolve_storage(value: Any) -> tuple[Any, bool]:
    """Return ``(storage, owned_by_runtime)``."""
    if value is None:
        return None, False
    from ._storage import (  # noqa: PLC0415
        InMemoryStorage,
        SQLiteStorage,
        StorageBase,
    )

    if isinstance(value, StorageBase):
        return value, False
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
        if key in {"memory", "in_memory", "inmemory"}:
            return InMemoryStorage(), True
        if key in {"sqlite", "sqlite_memory", "sqlite_in_memory"}:
            return SQLiteStorage(), True
        raise ValueError(
            f"storage name {value!r} is not directly materializable; pass a "
            "StorageBase instance or (StorageClass, kwargs) for configured "
            "persistent storage."
        )
    if isinstance(value, type) and issubclass(value, StorageBase):
        return value(), True
    if (
        isinstance(value, tuple)
        and len(value) == 2  # ruff: ignore[magic-value-comparison]
        and isinstance(value[0], type)
        and issubclass(value[0], StorageBase)
        and isinstance(value[1], Mapping)
    ):
        return value[0](**dict(value[1])), True
    raise TypeError(
        "storage must be a StorageBase instance/subclass, a supported built-in "
        "name, or (StorageClass, kwargs)."
    )


def _resolve_index_config(value: Any) -> Any:
    if value is None:
        return None
    from ._similarity import RetrievalConfig  # noqa: PLC0415

    if isinstance(value, RetrievalConfig):
        return value
    if isinstance(value, Mapping):
        return RetrievalConfig(**dict(value))
    if isinstance(value, (str, type)):
        return RetrievalConfig(backend=value)
    raise TypeError(
        "index must be RetrievalConfig, a config mapping, backend name, or "
        "VectorIndexBackend subclass."
    )


def _resolve_retrieval_config(value: Any) -> Any:
    if value is None:
        return None
    from ._similarity import RetrievalConfig  # noqa: PLC0415

    if isinstance(value, RetrievalConfig):
        return value
    if isinstance(value, Mapping):
        return RetrievalConfig(**dict(value))
    if isinstance(value, str):
        return RetrievalConfig(match_mode=value)
    raise TypeError(
        "retrieval must be RetrievalConfig, a config mapping, or match-mode string."
    )


def _resolve_export_format(value: Any) -> Any:
    if value is None:
        return None
    from ._schema import ExportFormat  # noqa: PLC0415

    if isinstance(value, ExportFormat):
        return value
    if isinstance(value, str):
        return ExportFormat(value.strip().lower())
    raise TypeError("export must be ExportFormat or an ExportFormat value string.")


@dataclasses.dataclass
class RuntimeCorpus:
    """Materialized operational view of a validated :class:`CorpusPlan`.

    The runtime delegates content processing to :class:`CorpusPipeline`,
    persistence to :class:`StorageBase`, dense/sparse search to
    :class:`RetrievalIndex`, and serialization to :func:`export_documents`.
    It owns orchestration and lifecycle only.
    """

    plan: CorpusPlan
    pipeline: Any
    storage: Any = None
    index_config: Any = None
    retrieval_config: Any = None
    export_format: Any = None
    policy: RuntimePolicy = dataclasses.field(default_factory=RuntimePolicy)
    _owns_storage: bool = False
    _documents: tuple[Any, ...] = dataclasses.field(default=(), init=False, repr=False)
    _index: Any = dataclasses.field(default=None, init=False, repr=False)
    _closed: bool = dataclasses.field(default=False, init=False, repr=False)
    _has_run: bool = dataclasses.field(default=False, init=False, repr=False)

    @property
    def documents(self) -> tuple[Any, ...]:
        """Documents committed to this runtime generation."""
        return self._documents

    @property
    def index(self) -> Any:
        """Built :class:`RetrievalIndex`, or ``None`` before a successful run."""
        return self._index

    @property
    def plan_fingerprint(self) -> str:
        """Fingerprint of the immutable plan that produced this runtime."""
        return self.plan.fingerprint

    @property
    def index_generation(self) -> Any:
        """Current retrieval-index generation, or ``None`` when no index exists."""
        return None if self._index is None else self._index.index_generation

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RuntimeCorpus is closed.")

    def _check_source_policy(self, source: Any) -> None:
        sources = list(source) if _is_source_sequence(source) else [source]
        if not self.policy.allow_network:
            blocked = [value for value in sources if _is_url(value)]
            if blocked:
                raise PermissionError(
                    "RuntimePolicy(allow_network=False) rejects network source(s): "
                    f"{blocked}. Pass RuntimePolicy(allow_network=True) explicitly "
                    "to enable URL ingestion."
                )

    def _execute_source(self, source: Any) -> Any:
        self._check_source_policy(source)
        if _is_source_sequence(source):
            items = list(source)
            if not items:
                raise ValueError("RuntimeCorpus source sequence must not be empty.")
            # No silent partial success: run_batch must raise on the first failed
            # source because RuntimeCorpus does not yet return a batch-status envelope.
            return self.pipeline.run_batch(items, stop_on_error=True)
        return self.pipeline.run(source)

    @staticmethod
    def _documents_from_result(result: Any) -> list[Any]:
        if isinstance(result, list):
            return [doc for item in result for doc in item.documents]
        return list(result.documents)

    def _build_candidate_index(self, documents: list[Any]) -> Any:
        if self.index_config is None or not documents:
            return None
        from ._similarity import RetrievalIndex  # noqa: PLC0415

        candidate = RetrievalIndex(config=self.index_config)
        candidate.build(documents)
        return candidate

    def _commit(self, documents: list[Any], *, new_documents: list[Any]) -> None:
        # Build first so an explicit/unavailable backend fails before persistent
        # storage is touched. Storage backends promise atomic save_batch behavior.
        candidate_index = self._build_candidate_index(documents)
        if self.storage is not None and new_documents:
            self.storage.save_batch(new_documents)
        self._documents = tuple(documents)
        self._index = candidate_index

    def run(self, source: Any = None) -> Any:
        """Process the plan source and commit storage/index state once.

        Use :meth:`add` for subsequent sources.  Requiring an explicit ``add``
        avoids a second ``run`` silently mixing new data with storage/index state
        left by the first generation.
        """
        self._ensure_open()
        if self._has_run:
            raise RuntimeError(
                "RuntimeCorpus.run() already completed; use add() for new sources."
            )
        if "read" not in self.plan.effective_stages:
            raise RuntimeError(
                "RuntimeCorpus cannot run because the 'read' stage is disabled."
            )
        resolved_source = self.plan.get("source") if source is None else source
        if resolved_source is None:
            raise ValueError(
                "RuntimeCorpus.run() requires a source argument or plan.source."
            )

        result = self._execute_source(resolved_source)
        documents = self._documents_from_result(result)
        self._commit(documents, new_documents=documents)
        self._has_run = True
        return result

    def add(self, source: Any) -> Any:
        """Process additional source(s), then rebuild one coherent index generation."""
        self._ensure_open()
        if not self._has_run:
            raise RuntimeError("RuntimeCorpus.add() requires a successful run() first.")
        result = self._execute_source(source)
        new_documents = self._documents_from_result(result)
        merged = list(self._documents) + new_documents
        self._commit(merged, new_documents=new_documents)
        return result

    def search(self, query: str, *, config: Any = None) -> Any:
        """Search the current runtime index using the configured retrieval policy."""
        self._ensure_open()
        if self._index is None:
            raise RuntimeError(
                "RuntimeCorpus has no built retrieval index; run() a plan with index configured."
            )

        cfg = config or self.retrieval_config or self.index_config
        if cfg is None:
            from ._similarity import RetrievalConfig  # noqa: PLC0415

            cfg = RetrievalConfig()

        query_embedding = None
        if (
            cfg.match_mode in {"semantic", "hybrid"}
            and self._index.has_embeddings
            and self.pipeline.embedding_engine is not None
        ):
            query_embedding = self.pipeline.embedding_engine.embed([query])[0]
        return self._index.search(query, config=cfg, query_embedding=query_embedding)

    def query_storage(self, query: Any) -> Any:
        """Query the configured storage backend."""
        self._ensure_open()
        if self.storage is None:
            raise RuntimeError("RuntimeCorpus has no configured storage backend.")
        return self.storage.query(query)

    def export(
        self, output_path: str | Path, *, format: Any = None, **kwargs: Any
    ) -> Path:
        """Export the currently committed documents using the plan's export format."""
        self._ensure_open()
        selected = format or self.export_format
        if selected is None:
            raise ValueError(
                "export format is not configured; pass format=... explicitly."
            )
        selected = _resolve_export_format(selected)
        from ._export import export_documents  # noqa: PLC0415

        return export_documents(
            list(self._documents), Path(output_path), selected, **kwargs
        )

    def close(self) -> None:
        """Release runtime-owned resources. Safe to call more than once."""
        if self._closed:
            return
        if self._owns_storage and self.storage is not None:
            close = getattr(self.storage, "close", None)
            if callable(close):
                close()
        self._closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def materialize_plan(
    plan: CorpusPlan,
    *,
    policy: RuntimePolicy | None = None,
    registry: ComponentRegistry | None = None,
) -> RuntimeCorpus:
    """Resolve a validated :class:`CorpusPlan` into runtime components.

    Materialization does not read ``plan.source``.  It constructs only the
    runtime objects required by enabled stages, preserving the side-effect-free
    configuration contract of :class:`FluentCorpus` itself.
    """
    if not isinstance(plan, CorpusPlan):
        raise TypeError(f"plan must be CorpusPlan, got {type(plan).__name__!r}.")
    problems = plan.validate()
    if problems:
        joined = "; ".join(str(problem) for problem in problems)
        raise ValueError(f"invalid corpus plan: {joined}")
    _validate_runtime_stages(plan)

    stages = set(plan.effective_stages)
    reader_kwargs = (
        _resolve_reader_kwargs(plan.get("reader")) if "read" in stages else {}
    )
    chunker = (
        _resolve_chunker(plan.get("chunker"), registry) if "chunk" in stages else None
    )
    normalizer = (
        _resolve_normalizer(plan.get("normalizer")) if "normalize" in stages else None
    )
    enricher = _resolve_enricher(plan.get("enricher")) if "enrich" in stages else None
    embedder = _resolve_embedder(plan.get("embedder")) if "embed" in stages else None

    from ._pipeline import CorpusPipeline  # noqa: PLC0415

    pipeline = CorpusPipeline(
        chunker=chunker,
        normalizer=normalizer,
        enricher=enricher,
        embedding_engine=embedder,
        output_path=None,
        format=None,
        reader_kwargs=reader_kwargs,
    )

    storage, owns_storage = (
        _resolve_storage(plan.get("storage")) if "store" in stages else (None, False)
    )
    index_config = (
        _resolve_index_config(plan.get("index")) if "retrieve" in stages else None
    )
    retrieval_config = _resolve_retrieval_config(plan.get("retrieval"))
    if retrieval_config is None:
        retrieval_config = index_config
    export_format = _resolve_export_format(plan.get("export"))

    return RuntimeCorpus(
        plan=plan,
        pipeline=pipeline,
        storage=storage,
        index_config=index_config,
        retrieval_config=retrieval_config,
        export_format=export_format,
        policy=policy or RuntimePolicy(),
        _owns_storage=owns_storage,
    )
