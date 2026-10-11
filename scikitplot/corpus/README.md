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
| Generate bounded configuration/tuning variants | `FluentCorpus.iter_variants()` / `variants()` |
| Execute a Fluent plan and own runtime state | `RuntimeCorpus` |
| Customize readers, filters, downloads, or stages | `CorpusBuilder(..., factories=BuilderFactories(...))` |
| Inspect optional backend readiness | `component_capabilities()` / `reader.plan_asr_backends()` |
| Control optional backend ordering/fallback | `BackendPolicy` |
| Reuse one typed policy family across runtime/reader/builder seams | `CorpusPolicyBundle` |
| Control transfer security/resource budgets | `DownloadPolicy` |
| Extend retrieval/vector behavior | `RetrievalIndex` / `VectorIndexBackend` |
| Add deterministic fuzzy lexical matching | `scikitplot.levenshtein` |

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
    FluentCorpus()  # initialize immutable
    .chunker("paragraph")  # setter
    .storage("memory")  # setter
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

For settings files, CLIs, or tuning jobs, the same model is available through
data-driven helpers:

```python
configured = FluentCorpus.from_config(
    {
        "reader": "auto",
        "storage": "memory",
    }
)

variants = configured.iter_variants(
    retrieval=("keyword", "hybrid"),
    max_variants=8,
)

for candidate in variants:
    print(candidate.plan().fingerprint, candidate.explain()["valid"])
```

`iter_variants()` is the lazy bounded deterministic generator; `variants()` is
its eager tuple convenience wrapper. Both prove the entire Cartesian size
before the first plan is yielded, refuse empty/explosive choice sets, and feed
every candidate through the same canonical plan/fingerprint model. Strings,
bytes, mappings and scalar fragments are one value; other iterables form
explicit axes. It is intended for explicit test/tuning matrices, not hidden
auto-optimization.

## 4. Operational execution with `RuntimeCorpus`

`materialize()` turns a validated Fluent plan into operational runtime objects.
It does **not** read the configured source. Source processing starts only when
`run()` or `add()` is called.

```python
from scikitplot.corpus import RuntimePolicy

fluent = (
    FluentCorpus()  # initialize immutable
    .source("article.txt")  # setter
    .chunker("paragraph")  # setter
    .storage("memory")  # setter
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
fluent = FluentCorpus().enricher(  # initialize immutable  # setter
    SimpleEnricherSpec(
        min_token_length=3,
        max_keywords=8,
    )
)
```

`RuntimeCorpus.materialize()` resolves that spec to a
`SimpleFrequencyEnricher`. This is a zero-resource alternative when token and
frequency-keyword enrichment is enough and the richer `NLPEnricher` stack is
not needed.

## 6. Reader, filter, and downloader customization

`CorpusBuilder` now has one construction seam for each user-replaceable
component and accepts `BuilderFactories` natively. `FactoryCorpusBuilder` remains
a compatibility facade over the same seam; new code can use the core builder
directly.

```python
from scikitplot.corpus import BuilderFactories, CorpusBuilder

factories = BuilderFactories(
    reader_factory=my_reader_factory,
    filter_factory=my_filter_factory,
    downloader_factory=my_downloader_factory,
)
builder = CorpusBuilder(config, factories=factories)
```

The downloader seam is deliberately parallel to the reader seam:

```text
source URL
   -> builder._make_downloader()
   -> AnyDownloader / custom downloader
   -> DownloadResult
   -> builder._make_reader()
   -> DocumentReader / custom reader
```

This keeps URL policy, retries, limits, and output-path provenance inside the
downloader contract rather than duplicating URL download logic in the builder.
`BuilderConfig.filter_kwargs` is also applied by the normal builder path, so
configuration and factory customization no longer diverge.

For advanced settings that still use the built-in downloader, use a
`DownloadPolicy` for transport/security budgets and reserve
`BuilderConfig(downloader_kwargs={...})` for per-source/router options such as
YouTube language, headers or a GitHub token:

```python
from scikitplot.corpus import BuilderConfig, DownloadPolicy

config = BuilderConfig(
    download_policy=DownloadPolicy.constrained(),
    downloader_kwargs={"youtube_language": "tr"},
)
```

`DownloadPolicy` rejects unknown fields and non-boolean TLS/SSRF flags.

Preview dispatch without touching DNS/network:

```python
from scikitplot.corpus import AnyDownloader

dl = AnyDownloader("https://github.com/org/repo/blob/main/data.csv")
plan = dl.plan()
print(plan.downloader, plan.verify_ssl, plan.block_private_ips)
```

`DownloadPlan` never serializes token/header contents and does not allocate a
temporary directory. `plan_all()` provides one plan per URL for batch UI/CI
inspection. This is dispatch preflight, not an SSRF/content security verdict;
DNS and remote probes still occur only at the real network boundary.
When `download_policy` is explicit, customized legacy scalar download fields
(`download_timeout`, `max_download_bytes`, retry fields) are rejected instead
of being silently ignored; use the policy itself or `downloader_kwargs` for a
deliberate per-build override. Its
`secure`, `constrained` and `large_files` presets all keep TLS verification and
private-IP blocking enabled. Explicit downloader kwargs still win when a caller
deliberately needs one transfer-specific override.

`CustomDownloader` is a trusted-code escape hatch, not a Python sandbox. The
wrapper still validates the original URL by default, forwards timeout/TLS/size
policy, requires the returned path to be a regular file inside its output
directory unless explicitly allowed otherwise, and enforces `max_bytes` as a
postcondition. A custom handler can still make arbitrary additional network
requests, so untrusted handler code must never be loaded merely because it was
downloaded or supplied by a remote source.

## 7. Runtime network policy

The current runtime policy is intentionally narrow and defaults offline:

```python
RuntimePolicy.offline()
RuntimePolicy.networked()  # explicit URL-source permission
RuntimePolicy.from_config({"allow_network": False})
```

Security booleans are type-checked: strings such as `"false"` are rejected
instead of being treated as truthy Python values.

It rejects `http://` / `https://` **source ingestion** through `RuntimeCorpus`.
It does not claim to be a universal sandbox for model downloads, subprocesses,
native backends, or all filesystem activity. Those capabilities retain their
own contracts.

The existing URL-reader security layer still owns SSRF, redirect, size,
timeout, and archive protections. `RuntimePolicy` does not replace it.

### Compose policies without a mega strict flag

`RuntimePolicy`, `BackendPolicy`, `DownloadPolicy`, and `ErrorPolicy` keep
different responsibilities. `CorpusPolicyBundle` is a convenience container
that creates, validates, serializes, and explains those policies together
without changing their ownership:

```python
from scikitplot.corpus import CorpusPolicyBundle

policies = CorpusPolicyBundle.safe_local()
reader_kwargs = policies.reader_kwargs(model_size="tiny")
builder_config = policies.builder_config(chunker="paragraph")
runtime_policy = policies.runtime
error_policy = policies.errors
```

Convenience presets are `default`, `safe-local`, `strict-local`, `networked`,
and `docs-ci`. Nested mappings reject unknown keys and truthy string booleans.
Use `with_overrides()` when one policy dimension needs deliberate tuning. The
bundle does not automatically push backend policy into PDF/OCR/text readers,
because those formats do not share ASR fallback semantics.

## 8. Retrieval modes

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


### Lexical search on a SQLite store

`SQLiteStorage.search_text(text, limit)` ranks documents against free text with
SQLite FTS5's BM25 and returns `(document, score)` pairs, higher meaning more
relevant. The text is never read as FTS5 syntax, so identifiers, API names and
pasted error messages work as typed — `roc_auc_score()`, `C++`, `random-state`,
`ValueError: Input contains NaN` — and a question need not contain every word
of a document to find it. A compound word is searched both as its exact phrase
and as its parts, so the document holding `sklearn.metrics` in that order ranks
above one that merely mentions `sklearn` and `metrics`.

`StorageQuery(full_text=...)` stays a *filter*: the whole string as one phrase,
the same meaning the in-memory and JSONL backends emulate by substring.

## 9. Generic vector-index configuration

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

## 10. Optional capabilities

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

Audio/video Whisper readers use a fail-soft optional-backend cascade by default:
``faster-whisper`` is attempted first, then ``openai-whisper``. Import or runtime
backend failures are logged. If neither backend succeeds, ``strict=False``
(the default) yields no ASR documents; pass ``strict=True`` to raise after the
fallback cascade is exhausted. A successful backend that legitimately returns no
speech segments is still a successful empty transcription and does not trigger
the next backend.

Optional backend orchestration is centralized internally, but **fallback policy
remains reader-specific**. Whisper may fall back on runtime failures; XML falls
back to the stdlib parser only when optional ``lxml`` is unavailable; explicit
OCR backend selection does not silently switch engines. This prevents a generic
backend manager from hiding format-specific correctness failures.

Fail-soft media extraction is observable through ``backend_reports``:

```python
reader = AudioReader(path, transcribe=True)
docs = list(reader.get_documents())

for report in reader.backend_reports:
    print(report["status"], report["backend"], report["errors"])
    for skipped in report.get("skip_details", []):
        print("skipped:", skipped["backend"], skipped["reason"])
```

Reports are JSON-compatible and store exception type/message strings rather than
live exception objects. ``get_documents()`` clears stale reports at the start of
each run. Typical statuses are ``success``, ``empty``, ``degraded`` (a fallback
succeeded after an earlier backend failed), ``unavailable`` (policy/readiness
left nothing runnable), ``exhausted`` (backends ran but produced no accepted
result), and ``failed`` (runtime exceptions exhausted the chain). Therefore an
empty ASR document set caused by backend failure or policy exclusion is not
indistinguishable from a successful empty transcription.

### Readiness is more precise than "installed"

Use `component_capabilities()` when the distinction matters:

```python
from scikitplot.corpus import component_capabilities

reports = component_capabilities(
    (
        "asr:faster-whisper",
        "asr:openai-whisper",
        "ocr:pytesseract",
    )
)

for name, report in reports.items():
    print(
        name,
        report["installed"],
        report["assets_ready"],
        report["ready"],
        report["selected"],
        report["active"],
    )
```

The vocabulary is intentional:

- `installed`: the package/module can be located;
- `assets_ready`: required model/corpus/executable is known ready, missing, or
  unknown without loading/downloading it;
- `ready`: combined side-effect-free preflight answer;
- `selected`: a backend policy chose the capability;
- `active`: runtime evidence says that capability actually completed work.

A preflight probe must not download a model or contact the network simply to
turn `unknown` into `True`.

For an actual reader configuration, preflight the exact ASR chain without
running any backend:

```python
reader = AudioReader(path, transcribe=True, backend_policy="offline")
plan = reader.plan_asr_backends()
print(plan.selected, plan.skipped)
print(plan.capability_view())
```

`BackendPlan` and runtime `BackendOutcome` share the same candidate factory, so
selection diagnostics cannot silently drift from execution order.

Discover families without knowing internal names:

```python
component_capabilities(role="asr")
component_capabilities(role="ocr")
component_capabilities(role="pdf")
component_capabilities(role="edit-distance")
```

Private/custom backends can use their own `CapabilityRegistry` and pass it to
`AudioReader`/`VideoReader`, so readiness, selection, and actual activity remain
connected without mutating the process-global registry.

### Backend policy and custom ASR

`BackendPolicy` controls orchestration while readers retain format semantics:

```python
from scikitplot.corpus import ASRBackend, BackendPolicy

policy = BackendPolicy.offline().with_order(
    "company-asr",
    include_unlisted=False,
)

backend = ASRBackend(
    name="company-asr",
    transcribe=my_transcriber,
    requires_network=False,
    may_download=False,
)
```

Custom ASR segments may include provider metadata. Reader-owned fields such as
`source_type` cannot be shadowed by those values; additional provider fields are
nested under `asr_metadata`, while `asr_backend` records the orchestrator-owned
backend provenance.

Policies can be serialized/tuned without multiplying named presets:

```python
policy = BackendPolicy.from_config(
    {
        "preset": "offline",
        "order": ["company-asr", "faster-whisper"],
        "include_unlisted": False,
        "on_exhausted": "raise",
    }
)
print(policy.to_dict())
```

Unknown configuration keys raise instead of being ignored.

`BackendPolicy.offline()` forbids network/download side effects, but does not
confuse “this backend may download on first use” with “it must download now.” A
backend marked `offline_capable=True` may make a local-only attempt when cache
readiness is unknown. The built-in faster-whisper adapter forces
`local_files_only=True` in this mode; backends that cannot guarantee local-only
execution are skipped.

## 11. Levenshtein lexical matching

`scikitplot.levenshtein` is a small independent facade that Corpus can consume:

```python
from scikitplot import levenshtein

assert levenshtein.distance("kitten", "sitting") == 3
scorer = levenshtein.make_corpus_scorer(score_cutoff=0.70)
```

Automatic selection is deliberately license-aware and remains importable even
without optional packages:

```text
bundled scikitplot.cexternals._editdistance
    -> RapidFuzz (optional, MIT)
    -> pure Python fallback
```

The separately distributed `Levenshtein`/`python-Levenshtein` implementation is
supported only by explicit backend selection; it is not silently inserted into
the automatic chain. Edit distance is lexical rather than semantic, so it is a
good fit for spelling/OCR/identifier variation rather than a replacement for
semantic embedding retrieval.

## 12. Gallery reliability rule


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

Those are real defects and should remain observable. For optional ASR, the
default observation is a warning plus backend fallback; ``strict=True`` converts
exhausted fallback into an exception. The gallery must never fabricate an ASR
result after a backend failure.

## 13. Which example should I read next?

Recommended learning order:

1. **FluentCorpus basics** — immutable plans, validation, branching, and bounded variants.
2. **Backend readiness/custom ASR** — explicit policy and user-provided backends.
3. **Levenshtein retrieval** — deterministic fuzzy lexical matching.
4. **FluentCorpus + RuntimeCorpus Hamlet** — real local run/store/search/export.
5. **Chunking strategy comparison** — choose sentence/paragraph/window/semantic behavior.
6. **MP3** — optional ASR/media provenance.
7. **ZIP mixed media** — archive member routing and per-extension reader settings.
8. **YouTube** — offline proxy plus live-network configuration.
9. **WHO multi-source** — broad integration and partial-source outcomes.

The gallery review should keep portable executed paths separate from optional
network/native/model paths so documentation builds remain truthful and useful
across CPython, CI, and constrained browser environments.
