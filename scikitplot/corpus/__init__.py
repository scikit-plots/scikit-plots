# scikitplot/corpus/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus
=================
Tools for turning files, URLs, media, and text sources into canonical
:class:`CorpusDocument` evidence that can be transformed, embedded, stored,
searched, adapted, and exported.

Choose the API that matches the job
-----------------------------------

``CorpusPipeline``
    Direct stage control for one source or an explicit batch.

``CorpusBuilder``
    High-level build/search/adapt convenience for end-to-end corpus workflows.

``FluentCorpus``
    Immutable, reusable configuration. Chained setter order describes *what*
    is configured; it does not define execution order.

``RuntimeCorpus``
    The operational form of a validated Fluent plan. Materialization constructs
    runtime components; source processing starts only with ``run()`` or
    ``add()``.

``RetrievalIndex`` / ``VectorIndexBackend``
    Lower-level retrieval and vector-backend extension APIs.

The common processing picture is::

    Source
      -> Read
      -> Chunk / Normalize / Enrich
      -> Embed
      -> Store
      -> Index
      -> Retrieve
      -> Adapt / Export

Examples
--------
Direct pipeline control:

>>> from pathlib import Path
>>> from scikitplot.corpus import CorpusPipeline, ParagraphChunker
>>> pipeline = CorpusPipeline(chunker=ParagraphChunker())
>>> result = pipeline.run(Path("article.txt"))  # doctest: +SKIP
>>> print(f"{result.n_documents} chunks from {result.source}")  # doctest: +SKIP

The dependency-free default sentence backend is ``REGEX``. Passing a spaCy
model name explicitly selects the spaCy shorthand instead:

>>> from scikitplot.corpus import SentenceChunker
>>> portable = SentenceChunker()
>>> spacy_chunker = SentenceChunker("en_core_web_sm")  # doctest: +SKIP

High-level builder:

>>> from scikitplot.corpus import CorpusBuilder, BuilderConfig
>>> builder = CorpusBuilder(
...     BuilderConfig(
...         chunker="paragraph",
...         normalize=True,
...         enrich=True,
...         build_index=True,
...     )
... )
>>> result = builder.build("./data/")  # doctest: +SKIP
>>> results = builder.search("quantum computing")  # doctest: +SKIP

Reusable declarative configuration:

>>> from scikitplot.corpus import FluentCorpus
>>> fluent = FluentCorpus().chunker("paragraph").storage("memory")
>>> fluent.validate()
[]

Materialization is explicit and performs no source read by itself:

>>> from scikitplot.corpus import RuntimePolicy
>>> with fluent.materialize(policy=RuntimePolicy(allow_network=False)) as runtime:
...     len(runtime.documents)
0

A configured source becomes operational only when ``run()`` is called:

>>> runtime_fluent = (
...     FluentCorpus().source("article.txt").chunker("paragraph").storage("memory")
... )
>>> with runtime_fluent.materialize() as runtime:
...     result = runtime.run()  # doctest: +SKIP

Network and optional-capability examples
----------------------------------------
Live URLs, OCR, ASR, spaCy, NLTK resource-backed NLP, model embeddings, and
native vector backends depend on the corresponding environment capability.
User-facing examples should not fabricate results when a capability is absent.
Documentation/gallery examples should either use a portable executed path or
report a clear skip while keeping the optional configuration visible.

URL ingestion:

>>> result = pipeline.run_url("https://en.wikipedia.org/wiki/Python")  # doctest: +SKIP

YouTube transcript:

>>> result = pipeline.run(
...     "https://www.youtube.com/watch?v=rwPISgZcYIk"
... )  # doctest: +SKIP

Image OCR:

>>> from scikitplot.corpus import DocumentReader
>>> reader = DocumentReader.create(Path("scan.png"))  # doctest: +SKIP
>>> docs = list(reader.get_documents())  # doctest: +SKIP

With model embeddings:

>>> from scikitplot.corpus import EmbeddingEngine
>>> engine = EmbeddingEngine(backend="sentence_transformers")  # doctest: +SKIP

Dependency-free local helpers:

>>> from scikitplot.corpus import HAMLET_TEXT, HashEmbedder, SimpleEnricherSpec
>>> HashEmbedder(dimension=32)([HAMLET_TEXT[:120]]).shape
(1, 32)
>>> FluentCorpus().enricher(SimpleEnricherSpec()).validate()
[]

``HashEmbedder`` is a deterministic lexical hashing baseline, not a learned
semantic model. ``HAMLET_TEXT`` is bundled convenience sample data rather than
an authoritative scholarly edition.

See ``scikitplot/corpus/README.md`` for the user-oriented API map, runtime
policy boundary, retrieval modes, and optional-capability guidance.
"""  # noqa: D205, D400

from __future__ import annotations  # ruff: ignore[unsorted-imports]

from . import (
    _adapters,
    _archive_handler,
    _base,
    _chunkers,
    _corpus_builder,
    _custom_hooks,
    _downloader,
    _embeddings,
    _enrichers,
    _export,
    _metadata,
    _normalizers,
    _pipeline,
    _readers,  # Readers -- import triggers registry population for all 24 extensions
    _registry,
    _runtime,
    _samples,
    _schema,
    _similarity,
    _sources,
    _storage,
    _types,  # CRITICAL-01: types layer was missing from exports entirely
    _url_handler,
)
from ._adapters import *  # noqa: F403  # --- Adapters (LangChain / LangGraph / MCP / HuggingFace / RAG) ---
from ._archive_handler import *  # noqa: F403  # --- Archive handling (zip / tar extraction) ---
from ._base import *  # noqa: F403  # Base classes
from ._chunkers import *  # noqa: F403
from ._corpus_builder import *  # noqa: F403  # --- Unified builder (the user-friendly orchestration API) ---
from ._custom_hooks import *  # noqa: F403
from ._downloader import *  # noqa: F403
from ._embeddings import *  # noqa: F403  # Embeddings by sentence-transformer
from ._enrichers import *  # noqa: F403  # --- NLP enricher ---
from ._export import *  # noqa: F403
from ._metadata import *  # noqa: F403
from ._normalizers import *  # noqa: F403  # --- Text normaliser ---
from ._pipeline import *  # noqa: F403
from ._readers import *  # noqa: F403
from ._registry import *  # noqa: F403  # Registry
from ._runtime import *  # noqa: F403  # Plan -> operational runtime materialization
from ._samples import *  # noqa: F403  # Small deterministic public-domain samples

# CORPUS-API-001: with ``from X import *`` the LAST import wins, so import
# order alone does NOT guarantee that a canonical symbol survives when two
# submodules export the same name. Canonical identities are therefore pinned
# explicitly at the end of this module (see the "canonical identities" block),
# independent of the order below. ``_types`` exports types-layer symbols, some
# of which are deprecated and carry a DeprecationWarning.
from ._schema import *  # noqa: F403  # Schema -- always first, zero optional dependencies
from ._similarity import *  # noqa: F403  # --- Similarity index ---
from ._sources import *  # noqa: F403  # Sources
from ._storage import *  # noqa: F403  # Storage
from . import _diagnostics  # noqa: F401

# O-4 / F-R02-05: capability_snapshot was reachable only through the private
# module path.  A capability probe that consumers cannot reach is not a
# capability contract.
from . import _capabilities  # noqa: F401
from . import _embedding_manifest  # noqa: F401
from . import _catalog  # noqa: F401
from . import _agentic  # noqa: F401
from . import _artifact  # noqa: F401
from . import _filters  # noqa: F401
from . import _generation  # noqa: F401
from . import _graph  # noqa: F401
from . import _plan  # noqa: F401
from . import _hierarchy  # noqa: F401
from . import _retrieval  # noqa: F401
from . import _retrievers  # noqa: F401
from ._diagnostics import *  # noqa: F403
from ._capabilities import *  # noqa: F403
from ._embedding_manifest import *  # noqa: F403
from ._catalog import *  # noqa: F403
from ._agentic import *  # noqa: F403
from ._artifact import *  # noqa: F403
from ._filters import *  # noqa: F403
from ._generation import *  # noqa: F403
from ._graph import *  # noqa: F403
from ._plan import *  # noqa: F403
from ._hierarchy import *  # noqa: F403
from ._retrieval import *  # noqa: F403
from ._retrievers import *  # noqa: F403
from ._types import *  # noqa: F403  # CRITICAL-01: Chunk, ChunkResult, ChunkerProtocol, etc.
from ._url_handler import *  # noqa: F403  # --- URL handling (classification, resolution, secure download) ---

__all__ = []
__all__ += _adapters.__all__
__all__ += _archive_handler.__all__
__all__ += _base.__all__
__all__ += _chunkers.__all__
__all__ += _corpus_builder.__all__
__all__ += _custom_hooks.__all__
__all__ += _downloader.__all__
__all__ += _embeddings.__all__
__all__ += _enrichers.__all__
__all__ += _export.__all__
__all__ += _metadata.__all__
__all__ += _normalizers.__all__
__all__ += _pipeline.__all__
__all__ += _readers.__all__
__all__ += _registry.__all__
__all__ += _runtime.__all__
__all__ += _samples.__all__
__all__ += _diagnostics.__all__
__all__ += _capabilities.__all__
__all__ += _embedding_manifest.__all__
__all__ += _catalog.__all__
__all__ += _agentic.__all__
__all__ += _artifact.__all__
__all__ += _filters.__all__
__all__ += _generation.__all__
__all__ += _graph.__all__
__all__ += _plan.__all__
__all__ += _hierarchy.__all__
__all__ += _retrieval.__all__
__all__ += _retrievers.__all__
__all__ += _schema.__all__
__all__ += _similarity.__all__
__all__ += _sources.__all__
__all__ += _storage.__all__
__all__ += _types.__all__  # CRITICAL-01: Chunk, ChunkResult, ChunkerProtocol, …
__all__ += _url_handler.__all__

# ===========================================================================
# CORPUS-API-001 — canonical identities and unique export manifest
# ===========================================================================
# The wildcard imports above are order-dependent (last import wins), which let
# the deprecated ``_types`` aliases shadow canonical symbols at the top level.
# Pin canonical identities explicitly here so they are independent of import
# order, then make ``__all__`` unique (order-preserving).

# ``PipelineResult`` is defined once, in ``_pipeline``. The former
# ``_types.PipelineResult`` alias to ``LegacyPipelineResult`` was deleted with the
# shim (ADR-C22 / DEC-157), so no de-shadowing import is required here.

# CORPUS-API-001 is CLOSED (ADR-R01-002 / ADR-C22).  The two distinct classes
# formerly both named ``NormalizerConfig`` now carry accurate names:
#   ``NormalizerConfigBase``   -- the abstract base in ``_types``
#   ``TextNormalizerConfig``   -- the concrete ``TextNormalizer`` config
# No import-order-dependent top-level binding remains.

# Order-preserving de-duplication of the aggregated manifest. Binding identity
# is set by the imports above; this only makes the advertised manifest unique.
__all__ = list(dict.fromkeys(__all__))
