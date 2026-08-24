# Logical Tracker — contracts, invariants, dependencies

What the code **promises**. This cannot be re-derived from the tree, so it is
maintained by hand and reviewed on every contract change.

Machine-readable mirror: `TRACKER.json` → `logical.contracts`.

---

## 1. How to use this file

When you change a contract:

1. Find its row below.
2. If the **invariant** still holds, update the row and add a test.
3. If the invariant no longer holds, **stop** — you are changing a promise, not
   an implementation. Record why in `HISTORY.md` and cite the finding that
   justifies it.

An invariant with no test is a comment. Every row below has one.

---

## 2. Contract dependency graph

Read top to bottom: nothing depends on something below it.

```text
                     EmbeddingManifest        CapabilityStatus
                            │                        │
                            ▼                        ▼
      ErrorRecord     IndexGeneration          ComponentCatalog
           │                │
   ┌───────┼────────┬───────┴──────────┐
   ▼       ▼        ▼                  ▼
ErrorPolicy  RetrievalResponse   ANNIndexArtifact
                │      │
                │      └── LegOutcome
                ▼
        Retriever legs ──► Agentic A1/A2
                │
                ▼
          RetrievalHit
```

Standalone: `VectorIndexBackend`, `FilterSupport` → `Filter AST`,
`DerivedGraph`, `Hierarchy`, `schema_version`, `CorpusPlan`/`FluentCorpus`.

**`ErrorRecord` is the root.** Eight findings across six review runs converged
on it, and every reporting contract depends on it. Change it last, and never
casually.

---

## 3. The contracts

| Contract | Module | Invariant that must not break |
|---|---|---|
| `ErrorRecord` | `_diagnostics.py` | **Never retains a live exception object.** Holding one keeps its traceback, every frame, and those frames' locals — measured at 139× the needed memory, worst exactly when a build is going worst. |
| `ErrorCategory` | `_diagnostics.py` | Few and stable. Fine-grained meaning belongs in `code`, which is free-form. |
| `ErrorPolicy` | `_schema.py` | **Behaviour only; logging is orthogonal.** No `LOG` member — it shared `SKIP`'s dispatch branch and encoded a logging decision as a behaviour value. `RETRY`'s terminal state is `COLLECT`. |
| `RetrievalResponse` | `_retrieval.py` | **Status is derived, never assigned.** `EMPTY` requires that *every requested leg actually ran*; a query returning nothing because a leg could not run is `DEGRADED`. |
| `LegOutcome` | `_retrieval.py` | A `DEGRADED`/`FAILED` leg **without** an `ErrorRecord` is unconstructible. `SKIPPED` exists only at leg level — an unrequested leg is not a failure. |
| `schema_version` | `_schema.py` | **Version-less payloads are refused, never assumed current.** Refusal is recoverable; a mis-read is not, because nothing downstream can detect it. Newer MINOR loads; different MAJOR does not. |
| `EmbeddingManifest` | `_embedding_manifest.py` | **Imports no embedding library.** Compatibility is fingerprint *equality* — "same dimension and model" is unsound, since differing normalization yields same-shaped vectors whose distances differ. |
| `IndexGeneration` | `_generation.py` | **Content-derived, not incremented.** The manifest fingerprint *is* `embedding_model_id` — one value, so they cannot disagree. Document digest is order-independent. |
| `ANNIndexArtifact` | `_artifact.py` | **The ordinal→doc_id sidecar is data, never positional coincidence.** Document validation compares *sets*, not order — order is what the sidecar records. Unknown sidecar schema is refused, not interpreted. |
| `VectorIndexBackend` | `_similarity/_backends.py` | Declares `metric` and `score_semantics`; the threshold guard reads the declaration rather than hardcoding which metrics are cosine-like. Registry keyed by **canonical identity**, aliases in a separate map. |
| `RetrievalHit` | `_similarity/_similarity.py` | **A fused hit has `native_score=None`** — it is on no leg's scale. Provenance fields use `compare=False`: they say how a hit was produced, not what it is. |
| `FilterSupport` | `_storage/_storage.py` | `SUPPORTED` / `EMULATED` / `REJECTED`. **Silently ignoring is not an option.** `EMULATED` is first-class: a scan genuinely answers the filter, just not natively. |
| Filter AST | `_filters.py` | Undeclared operators default to **`REJECTED`**. Absence is not inequality — `NotEq` on a missing field is `False`. A badly-typed field excludes the document, not the query. |
| `CapabilityStatus` | `_capabilities.py` | **`BROKEN` ≠ `ABSENT`** — a probe that *raises* means installed-and-failing; returning `False` means not installed. `available` is derived from `status`. |
| `ComponentCatalog` | `_catalog.py` | Read-only aggregating view; the four registries are not rewritten. **Unprobed is `UNKNOWN`, never `AVAILABLE`** — registered is not the same as usable. |
| `CorpusPlan` / `FluentCorpus` | `_plan.py` | Independent fragments **commute**; same-domain conflict is an **error**; **fluent call order never defines execution order** — `stages()` is the only way to set it. No I/O at configuration time. |
| `DerivedGraph` | `_graph.py` | **Derived, never stored.** Every edge carries relation, evidence, producer, trust, generation — including structural ones. No generic `related`. Budget exhaustion degrades with a naming record. |
| Hierarchy | `_hierarchy.py` | Traversal **terminates on cyclic data that validation would reject** — callers do not always validate first. Validation returns records, so every violation is seen in one pass. |
| Retriever legs | `_retrievers.py` | **Fusion keys on `doc_id`, never a row offset.** The fused score is recomputable from `contributions`. A leg returns an outcome, not a bare list. |
| Agentic A1/A2 | `_agentic.py` | **Budget policy is the sole authority** — a model may propose, the policy decides. A `StopReason` is always present. Token budgets are `UNENFORCEABLE` when no counter is supplied. Autonomy stops at A2. |

---

## 4. Cross-cutting invariants

These are not owned by one contract and are the easiest to lose.

| Invariant | Enforced by |
|---|---|
| Importing `scikitplot.corpus` loads **zero** optional heavyweights | `tests/test__import_hygiene.py` — a gate, not a convention |
| No import-time `DeprecationWarning` | `tests/test__deprecation_import_hygiene.py` |
| `_types` imports neither `pydantic` nor `mcp` | boundary test |
| Python 3.8 → 3.15+ grammar and API | `RULESET.md` audit checklist |
| Suite green under the **canonical** pytest config, no `-W` override | `VERIFICATION.md` |

The import gate deserves special care: it is held by roughly 288 deferred
imports across 41 modules, marked `# noqa: PLC0415`. **A single module-scope
`import torch` anywhere would undo all of them** while every other test kept
passing. That is why it is a test.

---

## 5. Contracts deliberately absent

Recorded so nobody "discovers" a gap that was a decision.

| Absent | Why |
|---|---|
| `VectorStore` | Designed (ADR-R07-001) but not built; `StorageBase` must not be widened into one |
| Durable agent memory | Its seven prerequisites are all absent; the ephemeral session needs none of them |
| `MetadataRetriever` | Would be the filter AST wearing a retriever hat |
| `symbol` / `topic` graph nodes | No producer — a node kind with no producer is a schema for data that cannot be created |
| `references` graph edges | Needs G1 body-link extraction; `url` is the document's *own* URL |
| Authorization / ACL | Speculative without a multi-tenant requirement; the traversal budgets are the enforcement point when it arrives |
| Concrete rerankers | Contract only; none exists |

---

## 6. Known logical debt

| Item | Consequence today |
|---|---|
| No producer sets `parent_doc_id` | G0's `contains`/`parent_of` edges are **empty**; `same_source` and `precedes` carry the graph |
| Four HIGH-04 chunking xfails | The only red marks in the suite; formally accepted, not forgotten |
| No validated normalization for non-cosine metrics | Score fusion is refused for them — correct, but limits fusion to cosine legs |
| `_base.py` holds four component categories | See `SUBMODULE_STRUCTURE.md` |
