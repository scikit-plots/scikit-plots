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
      → search() / query_storage() / export()

Unlike the earlier version of this example, no gallery-local ``RuntimeCorpus``
or ``materialize_plan`` helper is required.  The Corpus submodule owns that
translation and lifecycle directly.

The executed path is intentionally offline and deterministic:

* the Hamlet excerpt is embedded in the example,
* normalization and paragraph chunking use local Corpus components,
* token/keyword enrichment is a tiny deterministic gallery component,
* embeddings use a deterministic local hashing function,
* dense retrieval uses the exact ``bruteforce`` backend,
* storage uses :class:`InMemoryStorage`,
* export uses JSONL.

A final configuration-only branch shows the modern generic ``index_kwargs``
form for Annoy without requiring the native backend during the gallery build.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Imports
# -------
# Runtime construction, lifecycle, storage, retrieval, and export all come from
# the public Corpus API.  NumPy is used only by the deterministic teaching
# embedder defined below.

from __future__ import annotations

import hashlib
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scikitplot.corpus import (
    DocumentReader,
    ExportFormat,
    FluentCorpus,
    InMemoryStorage,
    ParagraphChunkerConfig,
    RetrievalConfig,
    RuntimePolicy,
    StorageQuery,
    TextNormalizerConfig,
)


# %%
# 1. Use real public-domain text
# ------------------------------
# The excerpt contains several well-known passages from *Hamlet*.  Bundling it
# directly keeps Sphinx-Gallery and offline CI independent of Project Gutenberg
# availability while still exercising a real literary text rather than toy
# placeholder sentences.

HAMLET_TEXT = """\
THE TRAGEDY OF HAMLET, PRINCE OF DENMARK

HAMLET. O that this too too solid flesh would melt,
Thaw, and resolve itself into a dew!
Or that the Everlasting had not fix'd
His canon 'gainst self-slaughter! O God! God!
How weary, stale, flat, and unprofitable
Seem to me all the uses of this world!
Fie on't! ah, fie! 'Tis an unweeded garden
That grows to seed. Things rank and gross in nature
Possess it merely. That it should come to this!

POLONIUS. Yet here, Laertes? Aboard, aboard, for shame!
The wind sits in the shoulder of your sail,
And you are stay'd for. There- my blessing with thee!
Give thy thoughts no tongue,
Nor any unproportion'd thought his act.
Be thou familiar, but by no means vulgar;
Those friends thou hast, and their adoption tried,
Grapple them unto thy soul with hoops of steel.
Give every man thy ear, but few thy voice;
Take each man's censure, but reserve thy judgment.
This above all- to thine own self be true,
And it must follow, as the night the day,
Thou canst not then be false to any man.

GHOST. I am thy father's spirit,
Doom'd for a certain term to walk the night,
And for the day confin'd to fast in fires,
Till the foul crimes done in my days of nature
Are burnt and purg'd away.
I could a tale unfold whose lightest word
Would harrow up thy soul, freeze thy young blood,
Make thy two eyes, like stars, start from their spheres.

HAMLET. O all you host of heaven! O earth! What else?
And shall I couple hell? O, fie! Hold, hold, my heart,
And you, my sinews, grow not instant old,
But bear me stiffly up. Remember thee!
Ay, thou poor ghost, while memory holds a seat
In this distracted globe. Remember thee!

HAMLET. To be, or not to be- that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles,
And by opposing end them. To die- to sleep-
No more; and by a sleep to say we end
The heartache, and the thousand natural shocks
That flesh is heir to. 'Tis a consummation
Devoutly to be wish'd. To die- to sleep.
To sleep- perchance to dream: ay, there's the rub!
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.

OPHELIA. O, what a noble mind is here o'erthrown!
The courtier's, soldier's, scholar's, eye, tongue, sword;
The expectancy and rose of the fair state,
The glass of fashion and the mould of form,
The observed of all observers, quite, quite down!

HAMLET. Speak the speech, I pray you, as I pronounced it to
you, trippingly on the tongue. But if you mouth it,
as many of your players do, I had as lief the
town crier spoke my lines. Nor do not saw the air
too much with your hand, thus, but use all gently;
for in the very torrent, tempest, and, as I may say,
the whirlwind of passion, you must acquire and beget
a temperance that may give it smoothness.

KING. O, my offence is rank, it smells to heaven;
It hath the primal eldest curse upon't,
A brother's murder. Pray can I not,
Though inclination be as sharp as will.
My stronger guilt defeats my strong intent,
And, like a man to double business bound,
I stand in pause where I shall first begin,
And both neglect.

HAMLET. Alas, poor Yorick! I knew him, Horatio. A fellow
of infinite jest, of most excellent fancy. He hath
borne me on his back a thousand times; and now, how
abhorred in my imagination it is!
Where be your gibes now? Your gambols? Your songs?
Your flashes of merriment that were wont to set the table on a roar?

HAMLET. There's a divinity that shapes our ends,
Rough-hew them how we will.

HORATIO. That is most certain.

HAMLET. Not a whit, we defy augury. There's a special
providence in the fall of a sparrow. If it be now,
'tis not to come; if it be not to come, it will be
now; if it be not now, yet it will come.
The readiness is all.
"""


# %%
# 2. Create a temporary real source file
# --------------------------------------
# The runtime uses the normal file-oriented reader layer.  A temporary
# workspace gives us a genuine source path and export destination without
# modifying the gallery source tree.

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-hamlet-")
_WORK_DIR = Path(_WORKSPACE.name)
_HAMLET_PATH = _WORK_DIR / "hamlet_excerpt.txt"
_EXPORT_PATH = _WORK_DIR / "hamlet_corpus.jsonl"

_HAMLET_PATH.write_text(HAMLET_TEXT, encoding="utf-8")

print("Source:", _HAMLET_PATH)
print("Characters:", len(HAMLET_TEXT))


# %%
# 3. Define a deterministic local embedder
# ----------------------------------------
# This small signed hashing bag-of-words embedder is intentionally simple.  It
# exists so the gallery can execute a real dense/hybrid retrieval path without
# downloading a transformer model.
#
# ``HashEmbedder`` is a frozen dataclass *and* callable.  That gives
# ``CorpusPlan`` a stable declarative description while allowing the runtime
# materializer to recognize it as a custom embedding function.

_TOKEN_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class HashEmbedder:
    """Deterministic teaching embedder with a stable plan identity."""

    dimension: int = 1024

    def __call__(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(
                    token.encode("utf-8"),
                    digest_size=8,
                ).digest()
                column = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, column] += sign

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms != 0)
        return matrix


# %%
# 4. Define deterministic token/keyword enrichment
# ------------------------------------------------
# The runtime accepts an object implementing ``enrich_documents``.  Keeping the
# example's enrichment local avoids making NLTK data packages a prerequisite
# for learning the Fluent/RuntimeCorpus workflow.

@dataclass(frozen=True)
class SimpleFrequencyEnricher:
    """Populate tokens and frequency-ranked keywords without optional data."""

    min_token_length: int = 3
    max_keywords: int = 8

    def enrich_documents(self, documents, *, overwrite: bool = False):
        enriched = []

        for doc in documents:
            if not overwrite and doc.tokens is not None:
                enriched.append(doc)
                continue

            source_text = doc.normalized_text or doc.text
            tokens = [
                token.lower()
                for token in _TOKEN_RE.findall(source_text)
                if len(token) >= self.min_token_length
            ]

            counts = Counter(tokens)
            keywords = [
                token
                for token, _count in sorted(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[: self.max_keywords]
            ]

            enriched.append(
                doc.replace(
                    tokens=tokens or None,
                    keywords=keywords or None,
                )
            )

        return enriched


# %%
# 5. Describe the whole corpus with FluentCorpus
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
    .enricher(
        SimpleFrequencyEnricher(
            min_token_length=3,
            max_keywords=8,
        )
    )
    .embedder(HashEmbedder(dimension=1024))
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
# 6. Materialize without processing the source
# --------------------------------------------
# ``materialize()`` constructs operational components and returns
# :class:`RuntimeCorpus`, but it does **not** read ``plan.source``.  That keeps
# configuration/materialization separate from execution.
#
# ``RuntimePolicy`` is also explicit.  This gallery forbids network sources,
# even though the configured source is already local.

runtime = fluent.materialize(
    policy=RuntimePolicy(allow_network=False),
)

print("Runtime type:", type(runtime).__name__)
print("Plan fingerprint:", runtime.plan_fingerprint)
print("Documents before run:", len(runtime.documents))
print("Index before run:", runtime.index)


# %%
# 7. Run once: process, store, and build the index
# ------------------------------------------------
# ``RuntimeCorpus.run()`` now owns the orchestration that the first showcase
# implemented manually:
#
# ``CorpusPipeline → storage.save_batch → RetrievalIndex.build``.
#
# The source comes directly from the Fluent plan, so no second source
# configuration is needed.

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
# 8. Query storage through RuntimeCorpus
# --------------------------------------
# The runtime exposes the configured storage backend without making the user
# keep a parallel storage variable in sync with the plan.

ghost_docs = runtime.query_storage(
    StorageQuery(
        full_text="ghost",
        limit=3,
    )
)

print("Storage full-text matches:", ghost_docs.total)
print("Filter support:", ghost_docs.filter_support)


# %%
# 9. Run a real hybrid search
# ---------------------------
# ``RuntimeCorpus.search`` automatically embeds semantic/hybrid queries with
# the same embedding engine that was used for corpus documents.  The gallery no
# longer has to create a query vector or call ``RetrievalIndex`` directly.

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
# 10. Override retrieval policy without rebuilding
# ------------------------------------------------
# Search policy can change per request while the current index generation stays
# intact.  Keyword mode needs no query embedding; semantic mode is embedded
# automatically by ``RuntimeCorpus.search``.

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
# 11. Export through RuntimeCorpus
# --------------------------------
# The plan already selected JSONL, so ``runtime.export`` only needs the output
# path.  The runtime exports the documents committed by ``run()``.

exported = runtime.export(
    _EXPORT_PATH,
    include_embedding=False,
)

print("Exported:", exported)
print("Bytes:", exported.stat().st_size)
print("First JSONL row:")
print(exported.read_text(encoding="utf-8").splitlines()[0][:300])


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
# 13. Close the runtime
# ---------------------
# Runtime state has a lifecycle.  ``close`` is idempotent and releases resources
# created/owned by the runtime.  A ``with fluent.materialize() as runtime:``
# block can be used when context-manager style is more convenient.

runtime.close()
runtime.close()

print("Runtime closed:", runtime.closed)


# %%
# 14. Compare the old and new structure
# -------------------------------------
# The mature Corpus components still perform the actual processing.  The new
# runtime layer removes orchestration code from user applications and gallery
# examples.
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
#     .query_storage()
#     .search()
#     .export()
#     .close()


# %%
# 15. Cleanup
# -----------
# Only the temporary Hamlet source and JSONL export were written by the example.

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
