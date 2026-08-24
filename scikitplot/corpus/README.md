# `scikitplot.corpus`

`scikitplot.corpus` turns files, URLs, media, and text sources into canonical
`CorpusDocument` evidence that can be transformed, embedded, stored, searched,
adapted, and exported.

## Choose your API

| Goal | Start with |
| --- | --- |
| Process one source with explicit stages | `CorpusPipeline` |
| Build/search/adapt a corpus quickly | `CorpusBuilder` |
| Create reusable immutable configuration | `FluentCorpus` |
| Execute a Fluent plan and own runtime state | `RuntimeCorpus` |
| Extend retrieval/vector behavior | `RetrievalIndex` / `VectorIndexBackend` |

A useful mental model is:

```text
Source
  -> Read
  -> Chunk / Normalize / Enrich
  -> Embed
  -> Store
  -> Index
  -> Retrieve
  -> Adapt / Export
```

## 1. Direct processing with `CorpusPipeline`

Use `CorpusPipeline` when you want explicit control over the processing stages.

```python
from pathlib import Path
from scikitplot.corpus import CorpusPipeline, ParagraphChunker

pipeline = CorpusPipeline(chunker=ParagraphChunker())
result = pipeline.run(Path("article.txt"))

for doc in result.documents[:3]:
    print(doc.text)
```

`ParagraphChunker` is pure Python. For sentence chunking, `SentenceChunker()`
uses the portable `REGEX` backend by default. NLTK and spaCy are optional
choices and may require separately installed resource data/models.

## 2. End-to-end convenience with `CorpusBuilder`

Use `CorpusBuilder` when the goal is a compact build/search/adapt workflow.

```python
from scikitplot.corpus import BuilderConfig, CorpusBuilder

builder = CorpusBuilder(
    BuilderConfig(
        chunker="paragraph",
        normalize=True,
        enrich=True,
        build_index=True,
    )
)

result = builder.build("./data/")
hits = builder.search("financial protection")
```

`CorpusBuilder` is also the better fit when a scenario intentionally focuses on
broad multi-source orchestration and structured partial-source outcomes.

## 3. Reusable configuration with `FluentCorpus`

`FluentCorpus` is immutable and side-effect free while it is being configured.
Each setter returns a new builder.

```python
from scikitplot.corpus import FluentCorpus

base = (
    FluentCorpus()
    .chunker("paragraph")
    .storage("memory")
)

print(base.plan().configured)
print(base.plan().fingerprint)
print(base.validate())
```

Setter order does not define pipeline order:

```python
a = FluentCorpus().embedder("E").storage("S")
b = FluentCorpus().storage("S").embedder("E")
assert a.plan() == b.plan()
```

`build()` remains the validated-plan boundary:

```python
plan = base.build()
```

## 4. Operational execution with `RuntimeCorpus`

`materialize()` turns a validated Fluent plan into operational runtime objects.
It does **not** read the configured source. Source processing starts only when
`run()` or `add()` is called.

```python
from scikitplot.corpus import RuntimePolicy

fluent = (
    FluentCorpus()
    .source("article.txt")
    .chunker("paragraph")
    .storage("memory")
)

with fluent.materialize(
    policy=RuntimePolicy(allow_network=False),
) as runtime:
    result = runtime.run()
    print(len(runtime.documents))
```

After a successful `run()`, use `add()` for additional sources rather than
calling `run()` a second time.

When storage/index/retrieval are configured, the runtime can coordinate:

```text
run()
  -> CorpusPipeline
  -> storage commit
  -> retrieval-index build

add()
  -> process new source
  -> preserve previous documents
  -> commit one new coherent runtime generation

search()
query_storage()
export()
close()
```

`close()` is idempotent. A context manager is the preferred lifecycle form
when the whole runtime workflow lives in one Python scope or one notebook cell.
For multi-cell notebooks, materialize explicitly, keep the runtime open across
cells, and call `close()` in the final cleanup cell.

## 5. Dependency-free local helpers

Corpus includes small public helpers for deterministic local/offline workflows.
They require no model download and do not consult NLTK/spaCy resources.

```python
from scikitplot.corpus import (
    HAMLET_TEXT,
    HashEmbedder,
    SimpleEnricherSpec,
    SimpleFrequencyEnricher,
)
```

`HAMLET_TEXT` is a bundled public-domain excerpt intended for deterministic
examples, tests, and quick experiments. It is convenience sample data, not an
authoritative scholarly edition.

`HashEmbedder` is a deterministic signed feature-hashing baseline:

```python
embedder = HashEmbedder(dimension=256)
vectors = embedder(["ghost father", "sleep dream"])
print(vectors.shape)  # (2, 256)
```

It preserves lexical overlap and produces L2-normalized `float32` vectors. It
is useful for offline tests and local baselines, but it is **not** a learned
semantic embedding model.

`SimpleFrequencyEnricher` provides Unicode-aware tokenization and deterministic
frequency-ranked keywords:

```python
enricher = SimpleFrequencyEnricher(
    min_token_length=3,
    max_keywords=8,
)
```

For declarative Fluent configuration, use `SimpleEnricherSpec` directly:

```python
fluent = (
    FluentCorpus()
    .enricher(
        SimpleEnricherSpec(
            min_token_length=3,
            max_keywords=8,
        )
    )
)
```

`RuntimeCorpus.materialize()` resolves that spec to a
`SimpleFrequencyEnricher`. This is a zero-resource alternative when token and
frequency-keyword enrichment is enough and the richer `NLPEnricher` stack is
not needed.

## 6. Runtime network policy

The current runtime policy is intentionally narrow:

```python
RuntimePolicy(allow_network=False)
```

It rejects `http://` / `https://` **source ingestion** through `RuntimeCorpus`.
It does not claim to be a universal sandbox for model downloads, subprocesses,
native backends, or all filesystem activity. Those capabilities retain their
own contracts.

The existing URL-reader security layer still owns SSRF, redirect, size,
timeout, and archive protections. `RuntimePolicy` does not replace it.

## 7. Retrieval modes

| Mode | Use when | Dense embeddings required? |
| --- | --- | --- |
| `strict` | exact text conditions | no |
| `keyword` | lexical/BM25 retrieval | no |
| `semantic` | vector meaning similarity | yes |
| `hybrid` | lexical + vector fusion | for the dense leg |

Example:

```python
from scikitplot.corpus import RetrievalConfig

retrieval = RetrievalConfig(
    match_mode="hybrid",
    top_k=5,
    hybrid_alpha=0.5,
)
```

## 8. Generic vector-index configuration

Prefer backend-generic constructor options for new examples:

```python
RetrievalConfig(
    backend="annoy",
    index_kwargs={
        "metric": "angular",
        "n_trees": 20,
        "search_k": -1,
    },
)
```

The legacy `annoy_*` fields remain compatibility syntax. New documentation
should prefer `index_kwargs` because the same shape can configure other vector
backends without adding backend-specific top-level fields.

## 9. Optional capabilities

Corpus deliberately supports optional stacks. Not every environment needs or
can provide every capability.

| Capability | Typical requirement |
| --- | --- |
| NLTK sentence/token NLP | NLTK plus required resource data |
| spaCy NLP | spaCy plus a selected model |
| image OCR | OCR Python/backend/system capability |
| audio/video ASR | Whisper-compatible backend/model |
| model embeddings | configured embedding/model dependency |
| Annoy / FAISS / Voyager | corresponding native/vector backend |
| browser/WASM | portable subset; native/model-heavy paths may be unavailable |

### Gallery/example rule

If an **optional dependency or optional resource is absent**, a showcase should
report a clear skip and continue where that is safe. Missing optional capability
must not become a fabricated result.

A skip is appropriate for:

```text
optional package not installed
optional NLTK/spaCy/model resource unavailable
optional native vector backend unavailable
optional OCR/ASR capability unavailable
external network service intentionally disabled
```

A skip is **not** appropriate for hiding:

```text
wrong public API usage
unexpected TypeError/AttributeError
invalid data contract
security-policy violation
corrupt mandatory sidecar asset
regression in an installed backend
```

Those are real defects or explicit failures and should remain observable.

## 10. Which example should I read next?

Recommended learning order:

1. **FluentCorpus basics** — immutable plans, validation, branching.
2. **FluentCorpus + RuntimeCorpus Hamlet** — real local run/store/search/export.
3. **Chunking strategy comparison** — choose sentence/paragraph/window/semantic behavior.
4. **MP3** — optional ASR/media provenance.
5. **ZIP mixed media** — archive member routing and per-extension reader settings.
6. **YouTube** — offline proxy plus live-network configuration.
7. **WHO multi-source** — broad integration and partial-source outcomes.

The gallery review should keep portable executed paths separate from optional
network/native/model paths so documentation builds remain truthful and useful
across CPython, CI, and constrained browser environments.
