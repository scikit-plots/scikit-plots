# Submodule Structure — review, rules, and where to put new things

Corpus is 78 source files across 14 subpackages. This file answers *where does a
new thing go*, and records the structural debt worth knowing about before you
add to it.

---

## 1. The shape today

```text
scikitplot/corpus/
│
├── contracts ─────────── _types  _schema  _diagnostics  _retrieval
│                         _embedding_manifest  _generation  _filters
│
├── orchestration ─────── _base  _pipeline  _corpus_builder  _custom_hooks
│                         _plan  _agentic
│
├── capability ────────── _capabilities  _catalog  _registry/
│
├── retrieval ─────────── _similarity/  _retrievers  _graph  _hierarchy
│                         _artifact
│
├── ingestion ─────────── _sources/  _downloader/  _readers/  _url_handler
│                         _archive_handler
│
├── processing ────────── _normalizers/  _chunkers/  _enrichers/
│                         _embeddings/
│
├── persistence ───────── _storage/  _export/  _atomic  _resources
│
└── _maintenance/ ─────── this folder (no runtime code)
```

Those seven groupings are **conceptual, not directories**. That is deliberate:
turning them into packages would be a large rename for a small gain, and the
campaign's standing rule is that no rename happens without a cited finding.

---

## 2. Where does a new thing go?

Follow the first row that matches.

| You are adding | Put it | Also do |
|---|---|---|
| A new reader format | `_readers/` | register in `_registry`; declare its optional dependency |
| A new chunking strategy | `_chunkers/` | add the `ChunkingStrategy` member |
| A new ANN backend | `_similarity/_backends.py` | declare all 11 capability members; add to `_BACKENDS` by **canonical** name, aliases separately |
| A new storage backend | `_storage/` | answer `FilterSupport` per filter — never ignore one |
| A new **value type** other code returns | its own root module | add a row to `TRACKER_LOGICAL.md` with an invariant + test |
| A new **status or outcome** enum | the module that produces it | check it is not a parallel vocabulary — see §4 |
| A cross-cutting concern (atomicity, resources) | its own root module | it must import nothing from the categories above |
| Anything touching MCP / CLI / ANNOY wire formats | **not here** | Corpus publishes a contract; it does not import the consumer |

**When in doubt:** a new root-level module is better than growing `_base.py`, and
a new subpackage is better than a sixth root-level module in the same area. Root
level is already 39% of the module.

---

## 3. Structural debt, ranked

Recorded so the next person adding a file knows whether they are joining a known
pile or starting a new one. **None is urgent.** Each has a "fix it when" trigger,
because refactoring for its own sake is how a maintenance folder becomes a
project.

### 3.1 `_base.py` — four component categories, 3 167 lines

Holds readers, filters, `PipelineGuard` and `DummyReader`. The
`ComponentCatalog` turned this from a line count into a visible boundary
violation: the catalog reports components from a module whose name claims none
of them.

**Fix it when:** you would add a fifth category, or it crosses 3 500 lines.

**Split, if you do:**

```text
_base.py  ->  _readers/_base.py       reader base + DummyReader
              _filters/_base.py       filter base
              _guard.py               PipelineGuard
```

`PipelineGuard` first — it is self-contained, already uses `ErrorRecord`, and
has the clearest test boundary.

### 3.2 `_writing_system.py` reaches into a private registry

Four call sites import `_custom_tokenizer._TOKENIZER_REGISTRY` — a cross-module
dependency on a private singleton, held together by deferred imports.

**Fix it when:** a third module needs the same registry. Then promote it to a
`ComponentCatalog` category rather than adding a second private reach.

### 3.3 Root level is 39% of source LOC

26 files, 21 650 lines. Not a defect on its own; the tripwire is 45%.

**Fix it when:** the tripwire fires. `check_trackers.py` will tell you.

### 3.4 Four registries in three shapes

`ComponentRegistry` methods, module-level singletons, a class-keyed bridge
registry, and a plain `_BACKENDS` dict. `ComponentCatalog` is the read-only view
over all four — deliberately **not** a rewrite, because merging registries with
different lookup keys and error behaviours is a large blast radius for no gain.

**Fix it when:** a *fifth* shape is proposed. That is the signal the catalog
stopped being sufficient.

---

## 4. Rules for expanding Corpus

These are the ones that get violated by well-intentioned additions.

### Do not create a parallel vocabulary

Before adding a status, trust tier or error category, check whether one exists:

| You want | It already exists as |
|---|---|
| "is this backend usable?" | `CapabilityStatus` (7 states) |
| "how did this operation go?" | `RetrievalStatus` / `LegStatus` |
| "how much do I trust this edge?" | `EdgeTrust` (DERIVED / EXTRACTED / CLAIMED) |
| "what went wrong?" | `ErrorRecord` + `ErrorCategory` |
| "did the backend apply my filter?" | `FilterSupport` |
| "why did the loop stop?" | `StopReason` |

Three separate review runs concluded "satisfied by reuse, not by a parallel
scheme". A second vocabulary for the same question is worse than none, because
consumers must then learn which one applies where.

### Declare capabilities; never assume them

A new component declares what it can do. If it cannot be probed, it reports
`UNKNOWN` — **not** `AVAILABLE`. `supports_persistence` was `False` for every
backend until an artifact format existed, and flipped only with a test asserting
both sides.

### Never let an operation succeed on partial evidence

The single rule this module exists to uphold. If your component can partially
fail, it returns a status and an `ErrorRecord`. If it cannot express that, it
raises. What it must not do is return a plausible answer silently.

### Optional dependencies import at call time

Module-scope `import torch` (or `nltk`, `lxml`, `annoy`…) breaks the import gate
for the whole package. Import inside the function and mark it
`# noqa: PLC0415`.

### Every rename cites a finding

Preference is not sufficient. This rule survived the withdrawal of backward
compatibility — what changed is what is *affordable*, not what is *justified*.

---

## 5. Submodule review checklist

Run before merging a new subpackage or a new component category.

```text
[ ] Does an existing subpackage own this concept?  (if yes, put it there)
[ ] Does it introduce a status/trust/error vocabulary that already exists?
[ ] Are optional dependencies imported at call time?
[ ] Does every new public type have a row in TRACKER_LOGICAL.md?
[ ] Does every invariant in that row have a test?
[ ] Can the component partially fail?  If so, can it say so?
[ ] Does it declare its capabilities, with UNKNOWN where unprobed?
[ ] Does it register itself so ComponentCatalog sees it?
[ ] Are there tests?  A subpackage with source and no tests is unowned.
[ ] python _maintenance/check_trackers.py  -> exit 0
[ ] Full suite green under the canonical config, no -W override
```

---

## 6. Innovative directions, with their prerequisites

Recorded so a future session proposes them *with* their cost rather than
discovering it halfway.

| Direction | Needs first | Value |
|---|---|---|
| **Wire Annoy's native save/load** into `ANNIndexArtifact` | nothing — the artifact and sidecar exist | unlocks 3 of Annoy's 4 stated roles: mmap serving, multi-process sharing, static read-heavy ANN |
| **Producer-side `parent_doc_id`** | a chunker that records its parent | G0's `contains`/`parent_of` edges stop being empty; hierarchy becomes real |
| **Adopt the filter AST inside `_storage`** | nothing — AST and `FilterSupport` both exist | disjunction and negation become expressible against real backends |
| **`VectorStore`** as a separate protocol | the artifact (done) + a conformance backend | remote vector stores without widening `StorageBase` into a capability lottery |
| **G1 entity extraction** | a deterministic extractor with provenance | `references` edges; entity nodes |
| **Split `[corpus]` into capability extras** | nothing — it is packaging | installing PDF support stops pulling PyTorch and two Whispers |
| **Typed-array embedding storage** | boundary conversion in `to_dict`/`from_dict` | ~8× memory reduction on the embedding path |
| **Caller-supplied token counter wiring** | nothing — `BudgetPolicy` accepts one | token budgets move from `UNENFORCEABLE` to enforced |

The first and last two are the cheapest with the clearest payoff. The
`VectorStore` and G1 items are genuinely new architecture and should get their
own review run rather than being slipped into a maintenance change.

---

## 7. What *not* to do

| Tempting | Why not |
|---|---|
| Widen `StorageBase` to answer vector queries | Produces one interface where some implementations can and others cannot — the capability lottery that made filters silently over-report |
| Merge the four registries | Large blast radius, no new capability; the catalog already answers the question |
| Add a generic `related` graph relation | An edge that does not say *how* two things relate cannot be filtered, explained or trusted |
| Materialise the G0 graph | A second copy of the same truth, needing invalidation on every document change |
| Persist the agentic session | Durable memory has seven prerequisites, all absent; ephemeral needs none |
| Infer tokens with a chunking tokenizer | Gives a budget wrong by an unknown factor — one that *appears* enforced |
| Rename `collection_id` to `corpus_id` | Preference, not a finding. "Collection" is accurate for a grouping within a corpus |
