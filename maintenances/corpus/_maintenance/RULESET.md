# `scikitplot.corpus` Maintenance Ruleset

This file contains the **non-negotiable working rules** for future Corpus
changes. It supersedes the old review-era methodology as the live maintenance
contract.

## 1. Source truth

Evidence priority:

```text
1. exact current source
2. executable tests/reproducers
3. package/build metadata
4. runtime/public API behavior
5. current maintenance registry
6. generated docs for the same snapshot
7. historical review/implementation artifacts
```

A historical finding is not current truth until reproduced against the current
source.

## 2. No chat-only state

Any material conclusion must land in one of:

```text
REGISTRY.md
STATE.json
VERIFICATION.md
HISTORY.md
```

A future maintainer must not need chat history to continue safely.

## 3. No duplicate architecture

Before adding a new abstraction, search the current source.

Current canonical contracts already include:

```text
ErrorRecord
RetrievalResponse / RetrievalStatus
EmbeddingManifest
ANNIndexArtifact
VectorIndexBackend
ComponentCatalog
CorpusPlan / FluentCorpus
Graph contracts
Retriever leg contracts
AgenticRetrievalSession / BudgetPolicy
```

Do not add a second equivalent type under a different name.

## 4. No silent partial success

This is the central Corpus invariant.

Operations must distinguish:

```text
SUCCESS
EMPTY
DEGRADED
FAILED
CANCELLED
```

Never convert:

```text
backend failed
unsupported filter
budget-truncated graph traversal
missing retrieval leg
```

into plausible complete success.

## 5. Diagnostics are data

Use `ErrorRecord`-style structured diagnostics.

Do not retain live exception objects in long-lived result collections unless
there is an explicit short-lived debugging reason.

Logging is observability, not failure semantics.

## 6. Unsupported capability is explicit

A requested operation must be:

```text
SUPPORTED
EMULATED with equivalent semantics
REJECTED as unsupported
```

Never silently ignore requested filters, namespaces, metrics, or retrieval legs.

## 7. Identity and generation rules

Never use backend row offsets as canonical Corpus identity.

Persisted/derived artifacts must be attributable to compatible generations.

Preserve:

```text
document identity
schema version
content-derived generation identity
embedding manifest identity
ordinal -> stable document ID mapping
artifact integrity
```

Reject incompatible combinations before serving results.

## 8. Embedding compatibility

Dimension equality alone does not prove embedding compatibility.

Preserve `EmbeddingManifest` validation across build/load/query paths.

A model/provider/revision/normalization change must invalidate incompatible
derived indexes.

## 9. Vector-index boundary

`VectorIndexBackend` owns nearest-neighbour index mechanics.

It does not own:

```text
canonical Corpus IDs
document persistence
metadata authorization
graph relationships
agent memory
MCP wire behavior
```

`ANNIndexArtifact` owns the persisted native index + versioned sidecar/manifest
boundary.

## 10. Graph rules

The implemented G0 graph is a **derived view**, not an independent source of
truth.

Preserve:

```text
typed nodes/edges
edge provenance
bounded traversal
explicit exhausted budgets
deterministic traversal
```

Do not silently materialize another durable graph truth without a new reviewed
generation/consistency contract.

## 11. Agentic rules

Agentic retrieval is bounded orchestration above retrieval.

Preserve:

```text
deterministic policy authority
hard step/time/tool/retrieval budgets
no-progress termination
explicit stop reason
degraded retrieval != success
no automatic durable memory write
```

The model may propose; deterministic policy enforces.

## 12. Fluent configuration rules

`CorpusPlan` / `FluentCorpus` are a configuration facade, not a second pipeline.

Preserve:

```text
independent fragments commute
same-domain conflict errors by default
explicit replacement only
fluent call order != execution order
configuration performs no network/model/backend initialization
plan fingerprint changes only for semantic config changes
```

Stage reordering must use an explicit stage-order API.

## 13. Reuse hardened primitives

Do not duplicate existing trust-boundary and durability helpers.

Preserve and reuse, where applicable:

```text
atomic_write_path / atomic_write_bytes
ensure_nltk_resource / downloads_allowed
_get_with_validated_redirects
_resolve_and_validate / _is_blocked_ip
hardened_lxml_parser / parse_stdlib_secure
stream_copy_bounded
_publish_extracted
_archive_ctx / ArchiveNestingError
capability_snapshot
_verify_artifact_integrity
_validate_loaded_documents
```

## 14. Security defaults

Default behavior must remain:

```text
offline-safe
bounded
fail-closed at trust boundaries
no surprise model/resource download
no surprise local -> remote fallback
no raw credential logging
no unsafe deserialize without explicit trust/integrity contract
```

Authorization/filtering must happen before evidence leaves its allowed scope.

## 15. Concurrency and resource rules

Do not rely on the GIL as a correctness contract.

Every new concurrent or long-running path must define:

```text
ownership
locking
cancellation
deadline
cleanup
partial failure
restart semantics
resource bounds
```

## 16. Python/platform rule

The project currently declares Python `>=3.8`.

Do not introduce unguarded runtime syntax/APIs incompatible with the project
floor. Optional backends may have narrower capability profiles; report those
honestly rather than weakening core behavior.

Platform claims require platform evidence.

## 17. Optional dependencies

Optional means optional in import behavior.

Base help/import/configuration/capability discovery must not initialize optional
heavy backends merely to inspect them.

Use capability discovery rather than accidental import failure as UX.

## 18. Testing rule

Every public guarantee must have an executable regression gate.

Do not delete a failing test just because an implementation changed. If the
contract legitimately changes:

```text
record decision
update contract
rewrite test
update registry/history
```

Tracked `xfail` tests stay strict enough that accidental fixes surface.

## 19. Performance rule

Do not claim optimization without measurement.

For relevant changes measure at least one of:

```text
latency
throughput
RSS
allocation/retention
disk/index size
build/rebuild time
recall/ranking quality
graph expansion
agent steps/tool calls
```

## 20. Maintenance-file rule

Live maintenance files are intentionally few.

Do not recreate a giant living review handbook.

Historical evidence belongs in external review/implementation bundles or
version-control history. The live source tree stores only:

```text
MAINTAINING.md
_maintenance/RULESET.md
_maintenance/REGISTRY.md
_maintenance/VERIFICATION.md
_maintenance/HISTORY.md
_maintenance/STATE.json
```


## 21. Cross-module ownership rule

Corpus owns protocol-neutral contracts.

Other submodules consume or implement those contracts without moving their
subsystem-specific logic into Corpus.

Canonical direction:

```text
Corpus RetrievalResponse
        ↓ consumed by
MCP
```

```text
Corpus VectorIndexBackend
        ↓ implemented by
Annoy adapter/backend
```

```text
Corpus CapabilityReport
        ↓ consumed by
CLI
```

Boundary rules:

```text
Corpus
  owns evidence identity, retrieval outcomes, capabilities, generation and
  protocol-neutral graph/retriever/agent contracts.

MCP
  owns MCP SDK, protocol models, resources/tools, transports and MCP lifecycle.

Annoy
  owns Annoy-specific native/Cython/index/mmap mechanics.

CLI
  owns command routing, presentation, exit/output semantics and lazy
  delegation.
```

Do not duplicate:

```text
MCP wire models in Corpus
Annoy native lifecycle in Corpus
Corpus retrieval semantics in MCP
Corpus capability probing in CLI
CLI presentation rules in Corpus
```

If a new requirement crosses a boundary, first decide whether it is:

```text
neutral contract
adapter
backend implementation
presentation/integration
```

and place it at the lowest correct ownership layer.

## 22. Cross-module compatibility rule

A change to one of these Corpus contracts requires downstream compatibility
review before release:

```text
RetrievalResponse / RetrievalStatus
VectorIndexBackend
CapabilityReport
EmbeddingManifest
ANNIndexArtifact
GraphResponse / retriever leg contracts
CorpusPlan / FluentCorpus
AgenticRetrievalSession outcome/budget contracts
```

Do not require immediate downstream code changes for internal-only Corpus
refactors that preserve these contracts.

## 23. Project sequence rule

The current review/closure sequence is:

```text
Corpus complete
→ MCP M00..closure
→ Annoy A00..closure
→ CLI C00..closure
→ cross-module verification
```

A later submodule campaign may reveal a real Corpus defect. If so:

1. reproduce it against current Corpus source;
2. register it in `REGISTRY.md`;
3. make the smallest boundary-preserving Corpus fix;
4. add a regression gate;
5. return to the active submodule campaign.

Do not restart the completed R00–R16 Corpus review.
