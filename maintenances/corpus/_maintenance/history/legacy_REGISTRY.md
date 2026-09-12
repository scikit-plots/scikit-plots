# Corpus Current Registry

**Role:** current human-readable state and open-item registry
**Source archive:** `scikit-plots(7).zip`
**Source SHA-256:** `099f9c414ffb6a80648d4a48b78670f243400a18b6362e1d735c44db89c15d13`

This file replaces the old review-era finding register as the **current
maintenance registry**.

## 1. Campaign status

```text
Review campaign          R00-R16       COMPLETE
Implementation campaign  IMPL-01-18    COMPLETE
Implemented waves        I0-I6         COMPLETE
Next project work        MCP -> Annoy -> CLI
```

Historical review totals at final R16 state:

```text
confirmed findings  55
P1                  4
P2                  39
P3                  12
disproofs           23
proposals           56
decisions           159
open unknowns       5
```

These counts describe the completed review campaign. They are not a current
bug count.

## 2. Implemented contract registry

### API and naming

Implemented normalization includes:

```text
RetrievalHit
ChunkHit
RetrievalConfig
RetrievalIndex
VectorIndexBackend
ToolCallInput
ToolCallResult
NormalizerConfigBase
TextNormalizerConfig
```

Do not reintroduce superseded public names without an explicit compatibility
decision.

### Diagnostics and outcomes

```text
ErrorRecord
ErrorCategory
RetrievalResponse
RetrievalStatus
LegOutcome
LegStatus
```

Key invariant:

```text
successful empty retrieval != degraded retrieval != failed retrieval
```

### Schema and hierarchy

Implemented:

```text
schema_version
hierarchy integrity checks
hierarchy query support
```

Known limitation: producer-side `parent_doc_id` population remains deferred
(see O-31).

### Capability and component discovery

Implemented:

```text
capability status/reporting
ComponentSpec
ComponentCatalog
registry overwrite protection
registration conformance checks
optional-import hygiene gate
```

### Embedding and generation identity

Implemented:

```text
EmbeddingManifest
manifest identity
build-time compatibility rejection
content-derived generation identity
```

### Local vector-index artifacts

Implemented:

```text
VectorIndexBackend capability declarations
threshold/metric scale guard
ANNIndexArtifact
versioned ordinal -> document-ID sidecar
generation/manifest validation
```

### Filtering and score provenance

Implemented:

```text
backend-neutral filter AST
explicit unsupported-filter behavior
native_score
native_metric
rank
rank-fusion default when score scales are not safely comparable
```

### Graph

Implemented G0 graph:

```text
GraphNode
GraphEdge
GraphQuery
GraphResponse
DerivedGraph
derive_graph
bounded BFS traversal
explicit budget exhaustion
edge provenance
```

Current design keeps G0 derived rather than persisted.

### Retriever legs and fusion provenance

Implemented:

```text
Retriever
LexicalRetriever
DenseRetriever
GraphRetriever
LegResult
LegContribution
fuse_by_rank
```

### Agentic retrieval

Implemented:

```text
BudgetPolicy
RoutePlan
rule-based routing
SufficiencySignals
AgenticRetrievalSession
InvestigationOutcome
bounded retrieve/evaluate/refine loop
explicit stop reasons
```

### Composable configuration

Implemented:

```text
CorpusPlan
FluentCorpus
order-independent independent fragments
explicit same-domain replacement
explicit stage-order API
side-effect-free configuration
stable plan fingerprint
```

## 3. Review/implementation lineage

Implementation increments:

| Increment | Main result |
| --- | --- |
| IMPL-01 | remove deprecated shims |
| IMPL-02 | naming normalization |
| IMPL-03 | registry/package/docs/import hygiene |
| IMPL-04 | `ErrorRecord` and extended error policy |
| IMPL-05 | `RetrievalResponse` + per-leg status |
| IMPL-06 | explicit storage/filter capability response |
| IMPL-07 | schema version + hierarchy integrity |
| IMPL-08 | capability status + `ComponentCatalog` |
| IMPL-09 | `EmbeddingManifest` chain |
| IMPL-10 | configuration composition + fluent facade |
| IMPL-11 | content-derived generation identity |
| IMPL-12 | vector-index capability + threshold-scale guard |
| IMPL-13 | `ANNIndexArtifact` + ordinal sidecar |
| IMPL-14 | backend-neutral filter AST |
| IMPL-15 | score provenance + fusion policy |
| IMPL-16 | G0 derived graph + bounded traversal |
| IMPL-17 | retriever legs + fusion provenance |
| IMPL-18 | bounded agentic retrieval |

## 4. Current tracked defects / accepted red marks

### HIGH-04 — four strict xfails

These are intentionally not marked fixed.

Current tests:

```text
_chunkers/tests/test__custom_tokenizer.py
  ScriptType closed/open enum decision
  punctuation-only detect_script behavior

_chunkers/tests/test__sentence_multilang.py
  CJK period splitting contract

_chunkers/tests/test__word_multilang.py
  invalid/unsupported language handling contract
```

Treat an unexpected XPASS as a signal to review the corresponding contract.

## 5. Deferred functional item

### O-31 — producer-side `parent_doc_id`

**Status:** DEFERRED

Current hierarchy integrity/traversal exists, but no normal producer stage
populates `parent_doc_id`.

Impact:

```text
parent_of / contains hierarchy edges can remain empty
same_source / precedes still make G0 useful
```

Required before closing O-31:

```text
define which producer owns parent assignment
define stable parent identity semantics
add construction tests
add graph edge tests
```

## 6. Environment-blocked verification items

### U-5 — HTTP adversarial cases

Need a faithful adversarial origin/network environment for:

```text
huge or absent Content-Length
slow response
partial response
proxy behavior
```

### U-6 — media-parser adversarial cases

Need optional parser stacks for:

```text
malformed PDF
OCR
audio/video
huge image
```

### U-7 — real filesystem/I/O fault injection

Call-site monkeypatching is insufficient to prove SQLite ordering under actual
I/O failure.

Need a real fault-injection environment.

### U-8 — multimodal embedding cache-key discipline

Needs a focused cross-modal cache identity verification.

### U-9 — native ANN quality/performance

Need actual optional native backends:

```text
annoy
faiss
voyager
```

Run recall/performance quality gates against the exact/brute-force oracle.

## 7. Current evidence artifacts

Historical audit bundle:

```text
CORPUS_R.zip
sha256: 6177c7d69b5eccf5089f2b00b87af731fa9599dc08f3585c34e702d7de9483bf
```

Implementation-log bundle:

```text
CORPUS_IMPL.zip
sha256: 42c0e721e8e1a6cfdd6678bb486857ffc8ea64ec34cc2a9c19a1e4ed74d75027
```

These are **historical evidence**, not required reading for normal fresh-chat
maintenance.

Use them only when:

```text
investigating why a decision was made
reconstructing a finding reproducer
auditing a rename/migration
checking the exact implementation increment
```

## 8. Next action

Do not create R17 or IMPL-19 merely to continue the old campaign.

Corpus is in normal maintenance mode.

Project sequence:

```text
MCP review/closure
→ Annoy review
→ CLI review
→ cross-module integration verification
```

New Corpus work begins only from a new reproduced issue or a new approved
feature requirement.


## 9. Cross-module boundary registry

This section is the current project boundary map for fresh-chat continuation.

### Corpus → MCP

```text
Corpus RetrievalResponse / RetrievalStatus
        ↓
MCP maps neutral outcomes into MCP tool/resource/protocol output
```

MCP may adapt:

```text
status
hits
citations/provenance
diagnostics
generation
```

MCP must not redefine Corpus retrieval truth.

In particular:

```text
Corpus FAILED
must not become
MCP "no matches"
```

### Corpus → Annoy

```text
Corpus VectorIndexBackend
        ↓
Annoy adapter/backend implementation
        ↓
Annoy native/Cython/index/mmap mechanics
```

Corpus owns:

```text
stable evidence/document identity
generation compatibility
EmbeddingManifest
ANNIndexArtifact contract
ordinal -> stable ID mapping
retrieval result semantics
```

Annoy owns:

```text
native item/index lifecycle
metric implementation
build/query mechanics
persistence/mmap internals
native type/specialization behavior
```

Annoy row/item identifiers must not become canonical Corpus IDs.

### Corpus → CLI

```text
Corpus CapabilityReport
        ↓
CLI renders/explains availability and configuration
```

CLI consumes, but does not reimplement:

```text
capability state
reason codes
effective backend selection
degraded operation status
generation/health information
```

CLI owns:

```text
human/machine output
exit codes
command routing
lazy delegation
```

### Corpus → framework integrations

Framework adapters consume protocol-neutral Corpus contracts.

They must not depend on MCP server internals merely to execute local retrieval.

### Corpus → future project-level verification

After MCP, Annoy and CLI reviews close, verify:

```text
Corpus retrieval outcome
↔ MCP mapping

Corpus VectorIndexBackend
↔ Annoy implementation

Corpus CapabilityReport
↔ CLI presentation

Corpus generation/identity
↔ all external adapters

Corpus optionality/import contract
↔ MCP/Annoy/CLI lazy paths
```

## 10. Recommended project sequence

```text
CORPUS
Review          COMPLETE
Implementation  COMPLETE
Maintenance     CURRENT
        |
        v
MCP
M00 deferred issue revalidation
-> final closure
        |
        v
ANNOY
A00 clean native build
-> native/type/lifecycle/mmap review
        |
        v
CLI
C00 bootstrap baseline
-> final UX/integration review
        |
        v
CROSS-MODULE VERIFICATION
```

### Why this order remains useful

MCP consumes neutral retrieval outcomes, so it should close against the
implemented Corpus contracts.

Annoy then verifies its native backend boundary against the implemented stable
identity/generation/index contracts.

CLI comes last so its capability/error/output UX reflects the final subsystem
contracts rather than temporary ones.

## 11. Remaining structural/logical big picture

Corpus is in normal maintenance mode, but the following architectural
boundaries must remain visible to future maintainers.

### Evidence layer

```text
source
→ reader
→ normalization
→ chunking/enrichment
→ CorpusDocument / chunk evidence
```

This layer owns canonical content/provenance, not vector or graph backend state.

### Embedding/generation layer

```text
evidence
→ EmbeddingManifest
→ compatible embedding generation
```

Dimension equality alone never proves compatibility.

### Local vector-index layer

```text
embedding generation
→ VectorIndexBackend
→ ANNIndexArtifact
→ stable ordinal mapping
```

The backend may be Annoy/FAISS/Voyager/exact, but Corpus identity remains
backend-independent.

### Retrieval layer

```text
lexical
dense
graph
   ↓
per-leg outcome
   ↓
fusion / rerank
   ↓
RetrievalResponse
```

This is the canonical truth boundary for success/empty/degraded/failure.

### Graph layer

Current G0 graph remains:

```text
derived
typed
provenance-bearing
bounded
deterministic
```

It is not yet a separately governed durable source of truth.

### Agentic layer

```text
route
→ retrieve
→ evaluate
→ refine if needed
→ stop / partial / abstain
```

It remains bounded orchestration over retrieval.

No autonomous policy escalation or automatic durable memory promotion.

### Configuration layer

```text
nested typed config
        \
         → CorpusPlan → validate/compile → runtime
        /
fluent facade
```

Independent fluent fragments commute. Call order does not define execution
order.

### Adapter layer

```text
Corpus neutral contracts
        ↓
MCP / CLI / framework adapters
```

Adapters translate; they do not redefine the core contract.

## 12. Remaining Corpus work classification

Only three categories should create new Corpus maintenance work:

### A. Reproduced current-source defect

Example:

```text
current implementation violates RULESET invariant
```

Action:

```text
register
reproduce
fix
test
update state/history
```

### B. Deferred item becomes executable

Current examples:

```text
O-31
U-5
U-6
U-7
U-8
U-9
```

Action:

```text
execute deferred verification
close or convert to a concrete finding
```

### C. Explicitly approved new capability

A new capability must identify:

```text
owner
neutral contract
backend/adapter boundary
security effect
resource effect
compatibility
verification
```

Do not use a new capability as a reason to reopen the completed review campaign.

## 13. Fresh-chat minimum registry

A fresh chat should normally read only:

```text
MAINTAINING.md
_maintenance/RULESET.md
_maintenance/REGISTRY.md
_maintenance/VERIFICATION.md
_maintenance/STATE.json
```

Read `HISTORY.md` only for rationale.

Read external Rxx/IMPL bundles only for historical forensic work.
