"""
Write Your Own Corpus Components: a Hamlet Walkthrough
======================================================

.. currentmodule:: scikitplot.corpus

:doc:`plot_corpus_fluent_hamlet_retrieval_script` builds a searchable
*Hamlet* corpus from built-in parts. This example builds it by hand and
replaces each part with one written here, to show the contract every part
must meet and where a part of your own plugs in:

1. a **reader** for a new file type (``.play``: one record per speech, with
   the speaker kept as metadata);
2. a **filter**, the quality policy (which speeches enter the corpus);
3. an **enricher** (tokens and keywords from the text);
4. an **embedding generator** (any function from texts to vectors);
5. the **manual pipeline**: run, store, index, search, export, each step
   by hand;
6. the **same parts in FluentCorpus** (``.filter(...)`` included), so one
   plan reproduces the manual run, with a policy that forbids network
   access.

Every part is a few lines of standard Python and runs offline, so the
output is the same on every run. Replace any one of them with your own.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Imports and the source file
# ---------------------------
# The public sample :data:`HAMLET_TEXT` is written with a new extension,
# ``.play``, that no built-in reader handles: step 1 adds one.

from __future__ import annotations

import hashlib
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from scikitplot.corpus import (
    HAMLET_TEXT,
    CorpusDocument,
    CorpusPipeline,
    DocumentReader,
    EmbeddingEngine,
    ExportFormat,
    FilterBase,
    FluentCorpus,
    InMemoryStorage,
    RetrievalConfig,
    RetrievalIndex,
    RuntimePolicy,
    SectionType,
    StorageQuery,
    export_documents,
)

_WORKSPACE = tempfile.TemporaryDirectory(prefix="scikitplot-hamlet-custom-")
WORK_DIR = Path(_WORKSPACE.name)
PLAY_PATH = WORK_DIR / "hamlet.play"
PLAY_PATH.write_text(HAMLET_TEXT, encoding="utf-8")
print("Source:", PLAY_PATH.name)


# %%
# 1. A reader for a new file type
# -------------------------------
# A reader subclasses :class:`DocumentReader`, names its extension in
# ``file_type`` and implements one method, ``get_raw_chunks``, which yields
# dicts. ``"text"`` is required; any other key (here ``"speaker"``) is kept
# as metadata on the documents. Defining the class registers it: from then
# on ``DocumentReader.create("x.play")`` returns this reader.
#
# Read files with an explicit encoding, and keep the reader free of policy:
# it yields everything; step 2 decides what enters the corpus.

_SPEECH = re.compile(r"^([A-Z][A-Z ]+)\.\s+(.*)", re.DOTALL)


@dataclass
class PlayReader(DocumentReader):
    """Read a play: one record per speech, the speaker as metadata."""

    file_type: ClassVar[str] = ".play"
    file_types: ClassVar[list[str] | None] = [".play"]

    def get_raw_chunks(self):
        text = self.input_path.read_text(encoding="utf-8")
        for block in re.split(r"\n\s*\n", text):
            match = _SPEECH.match(block.strip())
            if match is None:  # a title or stage direction
                continue
            speaker, speech = match.groups()
            yield {
                "text": " ".join(speech.split()),
                "section_type": SectionType.TEXT.value,
                "speaker": speaker.title(),
            }


reader = DocumentReader.create(PLAY_PATH)
speeches = list(reader.get_documents())
print("Reader:", type(reader).__name__, "| speeches:", len(speeches))
print("Speakers:", sorted({d.metadata["speaker"] for d in speeches}))


# %%
# 2. A filter: the quality policy
# -------------------------------
# A filter implements ``include(doc) -> bool``. It is the place for rules
# about *which* documents enter the corpus: here, speeches of at least
# ``min_words`` words, optionally only from some speakers. Make the rule a
# parameter, not a constant, so one class serves several policies.


@dataclass(frozen=True)
class SpeechPolicy(FilterBase):
    """Keep speeches long enough to search, from the chosen speakers."""

    min_words: int = 8
    speakers: frozenset[str] | None = None  # None: every speaker

    def include(self, doc: CorpusDocument) -> bool:
        if len(doc.text.split()) < self.min_words:
            return False
        return self.speakers is None or doc.metadata.get("speaker") in self.speakers


POLICY = SpeechPolicy(min_words=8)
kept = [d for d in speeches if POLICY.include(d)]
print(f"Policy keeps {len(kept)} of {len(speeches)} speeches")


# %%
# 3. An enricher
# --------------
# An enricher implements ``enrich_documents(documents, *, overwrite=False)``
# and returns new documents: :class:`CorpusDocument` is immutable, so
# ``doc.replace(...)`` makes the changed copy. Sorting by count, then by
# word, keeps the keyword order the same on every run.

_WORD = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class KeywordEnricher:
    """Tokens and the most frequent words, without any NLP data download."""

    min_length: int = 4
    max_keywords: int = 6

    def enrich_documents(self, documents, *, overwrite: bool = False):
        out = []
        for doc in documents:
            if doc.keywords is not None and not overwrite:
                out.append(doc)
                continue
            tokens = [
                t.lower()
                for t in _WORD.findall(doc.normalized_text or doc.text)
                if len(t) >= self.min_length
            ]
            ranked = sorted(Counter(tokens).items(), key=lambda kv: (-kv[1], kv[0]))
            out.append(
                doc.replace(
                    tokens=tokens or None,
                    keywords=[w for w, _ in ranked[: self.max_keywords]] or None,
                )
            )
        return out


# %%
# 4. An embedding generator
# -------------------------
# Any callable ``texts -> 2-D array`` can embed a corpus: a hashing trick
# here, a model or a remote service in practice. A frozen dataclass gives
# it a stable identity, so a plan that uses it fingerprints the same way on
# every run. :class:`EmbeddingEngine` wraps the function (``backend="custom"``)
# and adds batching and normalization.


@dataclass(frozen=True)
class WordHashEmbedder:
    """Signed hashing of words into ``dimension`` buckets."""

    dimension: int = 512

    def __call__(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
                column = int.from_bytes(digest[:4], "little") % self.dimension
                matrix[row, column] += 1.0 if digest[4] & 1 else -1.0
        return matrix


EMBEDDER = WordHashEmbedder(dimension=512)
engine = EmbeddingEngine(
    backend="custom",
    custom_fn=EMBEDDER,
    custom_fn_id=repr(EMBEDDER),  # names the function in caches and fingerprints
    enable_cache=False,
)
print("Vector shape for two texts:", engine.embed(["to be", "or not to be"]).shape)


# %%
# 5. The manual pipeline
# ----------------------
# Each stage by hand: the pipeline reads (step 1), filters (step 2),
# enriches (step 3) and embeds (step 4); then storage, index and search are
# separate objects you call yourself. This is what ``FluentCorpus`` does for
# you in step 6.

pipeline = CorpusPipeline(
    filter_=POLICY,
    enricher=KeywordEnricher(),
    embedding_engine=engine,
)
result = pipeline.run(PLAY_PATH)
documents = result.documents
print("Documents:", len(documents), "| all embedded:", all(d.has_embedding for d in documents))

storage = InMemoryStorage()
storage.save_batch(documents)
print("Stored:", storage.count(), "| speeches with 'spirit':",
      storage.query(StorageQuery(full_text="spirit", limit=5)).total)

index = RetrievalIndex(config=RetrievalConfig(backend="bruteforce"))
index.build(documents)

QUERY = "death sleep dream"
hybrid = RetrievalConfig(match_mode="hybrid", top_k=3, hybrid_alpha=0.5)
# A hybrid query needs a vector from the *same* generator as the corpus.
response = index.search(QUERY, config=hybrid, query_embedding=engine.embed([QUERY])[0])
for rank, hit in enumerate(response, start=1):
    print(f"#{rank} {hit.doc.metadata['speaker']:9s} {hit.score:.4f}  {hit.doc.text[:60]!r}")

exported = export_documents(
    documents, WORK_DIR / "manual.jsonl", ExportFormat.JSONL, include_embedding=False
)
print("Exported:", exported.name)


# %%
# Which speakers does the policy keep? Changing ``SpeechPolicy`` changes this
# figure and nothing else in the pipeline.

counts = Counter(d.metadata["speaker"] for d in documents)
names = sorted(counts, key=lambda n: (-counts[n], n))
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(names[::-1], [counts[n] for n in names[::-1]], color="tab:blue")
ax.set_xlabel("speeches kept by the policy")
ax.set_title(f"SpeechPolicy(min_words={POLICY.min_words})", fontsize=10)
fig.tight_layout()
plt.show()


# %%
# 6. The same parts in FluentCorpus
# ---------------------------------
# The parts written above plug into the plan unchanged: the reader is found
# by the ``.play`` extension, and the filter, enricher and embedding
# function are passed as they are. The filter is part of what gets built,
# so changing it changes the plan's fingerprint. ``RuntimePolicy(
# allow_network=False)`` is the runtime policy: a plan that tries to read a
# URL fails instead of downloading.
#
# The plan gives the same documents as the manual run.

fluent = (
    FluentCorpus()
    .source(PLAY_PATH)
    .reader(DocumentReader)
    .filter(POLICY)
    .enricher(KeywordEnricher())
    .embedder(EMBEDDER)
    .storage(InMemoryStorage)
    .index(RetrievalConfig(backend="bruteforce"))
    .retrieval(hybrid)
)
print("Validation problems:", fluent.validate())

with fluent.materialize(policy=RuntimePolicy(allow_network=False)) as runtime:
    runtime.run()
    same = [d.text for d in runtime.documents] == [d.text for d in documents]
    print("Same documents as the manual run:", same)
    top = runtime.search(QUERY)
    print("Plan top hit:", top[0].doc.metadata["speaker"] if top else "(none)")


# %%
# Where to go next
# ----------------
# * Register a reader for an existing extension with
#   ``registry.register_reader``; a chunker with ``registry.register_chunker``.
# * Keep every part deterministic (sorted output, fixed seeds, explicit
#   encodings) so two runs of a plan give the same corpus.
# * Put rules in parameters (``SpeechPolicy(min_words=...)``), not in code,
#   so one class serves several policies.

_WORKSPACE.cleanup()


# %%
#
# .. tags::
#
#    model-workflow: corpus
#    plot-type: bar
#    level: advanced
#    purpose: showcase
