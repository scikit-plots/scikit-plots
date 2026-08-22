"""
Build and Search a Real Hamlet Corpus with FluentCorpus
=======================================================

.. currentmodule:: scikitplot.corpus

This example uses a real public-domain *Hamlet* excerpt to demonstrate the
first-class :class:`FluentCorpus` runtime workflow.

The configuration remains declarative::

    FluentCorpus
      → source
      → reader
      → normalizer
      → chunker
      → enricher
      → embedder
      → storage
      → index
      → retrieval
      → export

The operational boundary is now explicit::

    FluentCorpus.materialize()
      → RuntimeCorpus
      → run()
      → add()
      → search() / query_storage() / export()

Unlike the earlier version of this example, no gallery-local ``RuntimeCorpus``
or ``materialize_plan`` helper is required.  The Corpus submodule owns that
translation and lifecycle directly.

The executed path is intentionally offline and deterministic:

* :data:`HAMLET_TEXT` provides a bundled public-domain sample,
* normalization and paragraph chunking use local Corpus components,
* :class:`SimpleEnricherSpec` configures :class:`SimpleFrequencyEnricher`,
* :class:`HashEmbedder` provides deterministic local lexical embeddings,
* dense retrieval uses the exact ``bruteforce`` backend,
* storage uses :class:`InMemoryStorage`,
* export uses JSONL.

The sample, simple enricher, enricher spec, and hashing embedder are public
:mod:`scikitplot.corpus` helpers rather than gallery-local implementations.

Because Sphinx-Gallery turns ``# %%`` blocks into independent notebook cells,
the runtime is kept explicitly open across the multi-cell workflow and closed
in the final cleanup cell.  For a single-cell script, a context manager remains
the preferred lifecycle form.  A bounded ``add()`` scenario extends the same
runtime generation with a second local source and rebuilds one coherent
retrieval index.

A final configuration-only branch shows the modern generic ``index_kwargs``
form for Annoy without requiring the native backend during the gallery build.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Imports
# -------
# Runtime construction, sample data, dependency-free enrichment, embeddings,
# lifecycle, storage, retrieval, and export all come from the public Corpus API.

from __future__ import annotations

import tempfile
from pathlib import Path

from scikitplot.corpus import (
    HAMLET_TEXT,
    DocumentReader,
    ExportFormat,
    FluentCorpus,
    HashEmbedder,
    InMemoryStorage,
    ParagraphChunkerConfig,
    RetrievalConfig,
    RuntimePolicy,
    SimpleEnricherSpec,
    SimpleFrequencyEnricher,
    StorageQuery,
    TextNormalizerConfig,
)


# %%
# 1. Use the public Hamlet sample
# -------------------------------
# ``HAMLET_TEXT`` is exported directly by :mod:`scikitplot.corpus`.  Keeping a
# small public-domain sample in the submodule makes examples, smoke tests, and
# offline experiments reproducible without copying the same text into every
# gallery file.
#
# The addendum stays local because it exists only to demonstrate
# :meth:`RuntimeCorpus.add` later in this example.

HAMLET_ADDENDUM = """\
POLONIUS. This business is well ended.
My liege, and madam, to expostulate
What majesty should be, what duty is,
Why day is day, night night, and time is time,
Were nothing but to waste night, day, and time.
Therefore, since brevity is the soul of wit,
And tediousness the limbs and outward flourishes,
I will be brief.

HAMLET. What a piece of work is a man! How noble in reason,
how infinite in faculty, in form and moving how express and admirable,
in action how like an angel, in apprehension how like a god:
the beauty of the world, the paragon of animals.
"""


# %%
# 2. Create temporary real source files
# --------------------------------------
# The runtime uses the normal file-oriented reader layer.  A temporary
# workspace gives us a genuine source path and export destination without
# modifying the gallery source tree.

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-hamlet-")
_WORK_DIR = Path(_WORKSPACE.name)
_HAMLET_PATH = _WORK_DIR / "hamlet_excerpt.txt"
_HAMLET_ADDENDUM_PATH = _WORK_DIR / "hamlet_addendum.txt"
_EXPORT_PATH = _WORK_DIR / "hamlet_corpus.jsonl"

_HAMLET_PATH.write_text(HAMLET_TEXT, encoding="utf-8")
_HAMLET_ADDENDUM_PATH.write_text(HAMLET_ADDENDUM, encoding="utf-8")

print("Initial source:", _HAMLET_PATH)
print("Initial characters:", len(HAMLET_TEXT))
print("Addendum source:", _HAMLET_ADDENDUM_PATH)


# %%
# 3. Configure the public dependency-free helpers
# ------------------------------------------------
# These helpers now belong to :mod:`scikitplot.corpus`; the gallery does not
# need to define its own hashing embedder or token/keyword enricher.
#
# ``SimpleEnricherSpec`` is the declarative form accepted directly by
# ``FluentCorpus.enricher(...)``.  ``SimpleFrequencyEnricher`` is the direct
# imperative equivalent when an application wants to enrich documents itself.

ENRICHER_SPEC = SimpleEnricherSpec(
    min_token_length=3,
    max_keywords=8,
)
DIRECT_ENRICHER = SimpleFrequencyEnricher(ENRICHER_SPEC)
HASH_EMBEDDER = HashEmbedder(dimension=1024)

print("Sample characters:", len(HAMLET_TEXT))
print("Declarative enricher spec:", ENRICHER_SPEC)
print("Direct enricher type:", type(DIRECT_ENRICHER).__name__)
print("Hash embedding dimension:", HASH_EMBEDDER.dimension)


# %%
# 4. Describe the whole corpus with FluentCorpus
# ----------------------------------------------
# ``index`` and ``retrieval`` are deliberately separate decisions:
#
# * ``index`` chooses how dense vectors are indexed.
# * ``retrieval`` chooses how the already-built corpus is queried/fused.
#
# ``RetrievalConfig`` is still used for both domains in the current transitional
# API, but backend-specific constructor settings now belong in generic
# ``index_kwargs`` rather than requiring a new top-level field per backend.

fluent = (
    FluentCorpus()
    .source(_HAMLET_PATH)
    .reader(DocumentReader)
    .normalizer(
        TextNormalizerConfig(
            steps=["unicode", "whitespace"],
            lowercase=False,
        )
    )
    .chunker(
        ParagraphChunkerConfig(
            min_length=40,
            max_length=900,
            overlap=0,
            merge_short=True,
        )
    )
    .enricher(ENRICHER_SPEC)
    .embedder(HASH_EMBEDDER)
    .storage(InMemoryStorage)
    .index(
        RetrievalConfig(
            backend="bruteforce",
        )
    )
    .retrieval(
        RetrievalConfig(
            match_mode="hybrid",
            top_k=4,
            hybrid_alpha=0.55,
        )
    )
    .export(ExportFormat.JSONL)
)

print(fluent)
print("Configured domains:", fluent.plan().configured)
print("Validation problems:", fluent.validate())


# %%
# 5. Materialize with an explicit runtime lifecycle
# -------------------------------------------------
# ``materialize()`` constructs operational components and returns
# :class:`RuntimeCorpus`, but it does **not** read ``plan.source``.  That keeps
# configuration/materialization separate from execution.
#
# ``RuntimePolicy`` is also explicit.  This gallery forbids network sources,
# even though both configured sources are local.
#
# A context manager is ideal when materialize/run/search/export all live in one
# Python scope.  This gallery intentionally spans multiple ``# %%`` cells,
# which become independent cells in the generated notebook.  Therefore we keep
# the runtime explicitly open across cells and close it in the final cleanup
# cell.  This makes both ``Run all`` and step-by-step notebook execution safe.
#
# Single-cell/script equivalent::
#
#     with fluent.materialize(policy=RuntimePolicy(allow_network=False)) as rt:
#         result = rt.run()

runtime = fluent.materialize(
    policy=RuntimePolicy(allow_network=False),
)

print("Runtime type:", type(runtime).__name__)
print("Plan fingerprint:", runtime.plan_fingerprint)
print("Documents before run:", len(runtime.documents))
print("Index before run:", runtime.index)

# %%
# 6. Run once: process, store, and build the index
# ------------------------------------------------
# ``RuntimeCorpus.run()`` owns:
#
# ``CorpusPipeline → storage.save_batch → RetrievalIndex.build``.
#
# ``run()`` is intentionally one-shot.  Additional sources belong to
# ``add()`` so the generation transition is explicit.

result = runtime.run()

print("Pipeline documents:", result.n_documents)
print("Runtime documents:", len(runtime.documents))
print("Stored documents:", runtime.storage.count())
print("Index documents:", runtime.index.n_documents)
print("Dense index available:", runtime.index.has_embeddings)
print("Dense backend:", runtime.index.backend_name)
print("Index generation:", runtime.index_generation)

for i, doc in enumerate(runtime.documents[:3]):
    print(f"\n[{i}] {doc.text[:180]!r}")
    print("    keywords:", doc.keywords)

# %%
# 7. Query storage through RuntimeCorpus
# --------------------------------------
# The runtime exposes the configured storage backend without making the
# user keep a parallel storage variable in sync with the plan.

ghost_docs = runtime.query_storage(
    StorageQuery(
        full_text="ghost",
        limit=3,
    )
)

print("Storage full-text matches:", ghost_docs.total)
print("Filter support:", ghost_docs.filter_support)

# %%
# 8. Run a real hybrid search
# ---------------------------
# ``RuntimeCorpus.search`` automatically embeds semantic/hybrid queries with
# the same embedding engine that was used for corpus documents.

query = "to die sleep dream death what comes after"
response = runtime.search(query)

print("Query:", query)
print("Retrieval status:", response.status)
print("Hits:", len(response))

for rank, hit in enumerate(response, start=1):
    preview = hit.doc.text.replace("\n", " ")[:170]
    print(
        f"\n#{rank} score={hit.score:.6f} "
        f"mode={hit.match_mode} backend={hit.backend}"
    )
    print(preview)

# %%
# 9. Add another source and advance the index generation
# -------------------------------------------------------
# ``add()`` processes new source data, appends the resulting documents to
# the runtime, commits them to storage, and rebuilds one coherent index over
# the complete document set.
#
# This is different from calling ``run()`` twice: ``run()`` establishes the
# first generation and ``add()`` makes later growth explicit.

documents_before_add = len(runtime.documents)
generation_before_add = runtime.index_generation

added = runtime.add(_HAMLET_ADDENDUM_PATH)

documents_after_add = len(runtime.documents)
generation_after_add = runtime.index_generation

print("Addendum pipeline documents:", added.n_documents)
print("Documents before add:", documents_before_add)
print("Documents after add:", documents_after_add)
print("Stored documents after add:", runtime.storage.count())
print("Generation before add:", generation_before_add)
print("Generation after add:", generation_after_add)
print("Generation changed:", generation_before_add != generation_after_add)

add_response = runtime.search("brevity soul wit tediousness")
print("Addendum search status:", add_response.status)
if add_response:
    print("Addendum top hit:", add_response[0].doc.text[:220])

# %%
# 10. Override retrieval policy without rebuilding
# -------------------------------------------------
# Search policy can change per request while the current index generation
# stays intact.  Keyword mode needs no query embedding; semantic mode is
# embedded automatically by ``RuntimeCorpus.search``.

keyword_response = runtime.search(
    "father spirit ghost",
    config=RetrievalConfig(
        match_mode="keyword",
        top_k=3,
    ),
)

semantic_response = runtime.search(
    "mortality dreams after death",
    config=RetrievalConfig(
        match_mode="semantic",
        top_k=3,
    ),
)

print("Keyword top hit:")
if keyword_response:
    print(keyword_response[0].doc.text[:220])

print("\nSemantic top hit:")
if semantic_response:
    print(semantic_response[0].doc.text[:220])

# %%
# 11. Export the current generation through RuntimeCorpus
# -------------------------------------------------------
# The plan already selected JSONL, so ``runtime.export`` only needs the
# output path.  Export happens after ``add()`` and therefore includes the
# complete current runtime generation.

exported = runtime.export(
    _EXPORT_PATH,
    include_embedding=False,
)

print("Exported:", exported)
print("Bytes:", exported.stat().st_size)
print("Exported documents:", len(runtime.documents))
print("First JSONL row:")
print(exported.read_text(encoding="utf-8").splitlines()[0][:300])

# The runtime intentionally remains open for the following independent gallery
# cells.  It is closed explicitly in the final cleanup section.
print("Runtime still open after export:", not runtime.closed)


# %%
# 12. Configure the same corpus for Annoy with generic index_kwargs
# -----------------------------------------------------------------
# Backend-specific constructor options now live in the generic ``index_kwargs``
# mapping.  The old ``annoy_*`` fields remain compatible, but new examples
# should prefer the generic form because it scales to other vector backends.
#
# This branch is configuration-only: it validates the immutable plan but does
# not materialize/build Annoy, so native Annoy is not required by the gallery.

annoy = fluent.replace_index(
    RetrievalConfig(
        backend="annoy",
        index_kwargs={
            "impl": "auto",
            "metric": "angular",
            "n_trees": 20,
            "search_k": -1,
        },
    )
)

print("Portable backend:", fluent.plan().get("index").backend)
print("Annoy backend:", annoy.plan().get("index").backend)
print("Annoy constructor kwargs:", annoy.plan().get("index").index_kwargs)
print("Source unchanged:", annoy.plan().get("source") == fluent.plan().get("source"))
print("Annoy plan validation:", annoy.validate())


# %%
# 13. Compare the old and new structure
# -------------------------------------
# The mature Corpus components perform the processing, and the reusable sample,
# enrichment, and hashing helpers now live in the submodule too.  The gallery
# therefore focuses on composition and lifecycle instead of local helper code.
#
# ::
#
#     BEFORE V2
#
#     FluentCorpus
#         ↓
#     CorpusPlan
#         ↓
#     gallery-local materialize_plan()
#         ↓
#     gallery-local RuntimeCorpus
#         ↓
#     pipeline.run()
#     storage.save_batch()
#     index.build()
#     index.search()
#     export_documents()
#
#
#     CURRENT
#
#     FluentCorpus
#         ↓
#     .materialize()
#         ↓
#     RuntimeCorpus
#         ↓
#     .run()
#     .add()
#     .query_storage()
#     .search()
#     .export()
#     .close()  # explicit across gallery/notebook cells


# %%
# 14. Cleanup
# -----------
# Close runtime-owned resources before deleting the temporary source/export
# workspace.  ``close()`` is idempotent, so rerunning this cleanup cell is safe.

runtime.close()
print("Runtime closed:", runtime.closed)

_WORKSPACE.cleanup()
print("Temporary workspace cleaned.")


# %%
#
# .. tags::
#
#    model-workflow: corpus
#    plot-type: text
#    level: intermediate
#    purpose: showcase
