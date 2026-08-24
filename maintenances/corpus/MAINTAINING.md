# Maintaining `scikitplot.corpus`

This file is the **human entry point** for future maintenance of
`scikitplot.corpus`.

The R00–R16 deep-review campaign and IMPL-01–IMPL-18 implementation campaign are
complete. Do **not** use the old review handbook or old session logs as current
source truth.

## Current source anchor

```text
archive: scikit-plots.zip
sha256: c8c0a17538a4d3f90d75ef7182b79b7755f12b50bdbf4ecf338f922256ef921f
```

If the source hash changes, re-verify claims before carrying them forward.

## Read order for a fresh chat

Read only these files first, in this order:

1. `_maintenance/MAINTENANCE_MODEL.md` — why / when / where / which / how many / how much
2. `_maintenance/RULESET.md` — the durable rules
3. `_maintenance/TRACKER_LOGICAL.md` — what each contract promises
4. `_maintenance/TRACKER_PHYSICAL.md` — what is on disk, and the tripwires
5. `_maintenance/SUBMODULE_STRUCTURE.md` — where a new thing goes
6. `_maintenance/REGISTRY.md` — remaining work and cross-module boundaries
7. `_maintenance/VERIFICATION.md` — how to prove the tree is healthy

Read `_maintenance/HISTORY.md` only when historical rationale is needed.

**Run this before anything else:**

```console
$ python scikitplot/corpus/_maintenance/check_trackers.py
$ python -m pytest scikitplot/corpus -q -p no:cacheprovider
```

The first fails on structural drift, a crossed tripwire, or a contract naming a
module that no longer exists. The second must be green under the **canonical**
pytest config — no `-W` override, since `filterwarnings = ["error"]` is what
catches import-time regressions.

Machine-readable:

```text
_maintenance/STATE.json     campaign state
_maintenance/TRACKER.json   physical inventory + logical contracts
```

## The one rule behind all the others

Corpus's failure mode is not crashing. It is producing **plausible output that is
wrong or incomplete without saying so** — the review campaign found that shape at
seven independent sites.

> Never let an operation succeed on partial evidence. If it can partially fail,
> it returns a status and an `ErrorRecord`. If it cannot express that, it raises.
> An unverified claim is worse than a narrow one: prefer `UNKNOWN`, `REJECTED` or
> `DEGRADED` over a confident guess.

## Current campaign state

```text
Corpus deep review       COMPLETE  (R00–R16)
Corpus implementation    COMPLETE  (IMPL-01–IMPL-18, waves I0–I6)
P1 review findings       CLOSED by implementation
Known tracked defects    four HIGH-04 xfails
Known deferred evidence  O-31 and U-5..U-9
Next project campaign    MCP, then Annoy, then CLI
```

The implementation campaign reported its final canonical suite as:

```text
3206 passed
27 skipped
4 xfailed
```

That number is historical evidence for IMPL-18, not a substitute for rerunning
tests against a newer source snapshot.

## Current architectural vocabulary

Use the implemented names, not the superseded pre-review names:

```text
ChunkHit
RetrievalHit
RetrievalConfig
RetrievalIndex
VectorIndexBackend
ToolCallInput / ToolCallResult

ErrorRecord
RetrievalResponse
RetrievalStatus
LegOutcome / LegStatus

EmbeddingManifest
ANNIndexArtifact

CorpusPlan
FluentCorpus

GraphNode / GraphEdge / GraphQuery / GraphResponse
derive_graph

LexicalRetriever / DenseRetriever / GraphRetriever
LegContribution

BudgetPolicy
AgenticRetrievalSession
InvestigationOutcome
```

## Core architecture

```text
ingestion / transformation
        ↓
canonical CorpusDocument evidence
        ↓
EmbeddingManifest + generation identity
        ↓
local VectorIndexBackend / ANNIndexArtifact
        ↓
lexical + dense + graph retriever legs
        ↓
RetrievalResponse
        ↓
optional bounded agentic orchestration
        ↓
thin external adapters
```

`scikitplot.annoy` is a local vector-index backend. It is not the Corpus
identity layer, metadata database, graph store, or agent memory.

## Configuration UX

Both explicit and fluent configuration are supported through one canonical plan:

```python
Corpus(plan=CorpusPlan.of(reader=..., embedder=..., storage=...))
```

and:

```python
Corpus().reader(...).embedder(...).storage(...)
```

Independent configuration fragments are order-independent. Fluent call order
does not define pipeline execution order. Same-domain replacement must be
explicit.

## Fresh-chat continuation prompt

> Review the current `scikitplot.corpus` source from the supplied source
> snapshot. Verify the source hash first. Read `MAINTAINING.md`,
> `_maintenance/RULESET.md`, `_maintenance/REGISTRY.md`, and
> `_maintenance/VERIFICATION.md`. Treat `_maintenance/HISTORY.md` and any
> external Rxx/IMPL logs as historical evidence only. Do not reopen completed
> Corpus architecture or rename public contracts without a reproduced defect or
> an explicit new requirement. Preserve current security, retrieval-outcome,
> generation, graph-budget, agent-budget, and configuration invariants. Record
> any new issue in the registry with evidence, a regression gate, and an exact
> next action.

## Updating maintenance state

Whenever a material Corpus change lands:

1. update `_maintenance/REGISTRY.md`;
2. update `_maintenance/STATE.json`;
3. update `_maintenance/VERIFICATION.md` if gates changed;
4. update `_maintenance/HISTORY.md` only for a meaningful completed change set;
5. keep this file short.

Do not create new parallel files named `FINAL`, `REVISED`, `EXPANDED`,
date-suffixed, or chat-specific variants inside the source tree.


## Project big picture

Corpus is now the protocol-neutral evidence/retrieval foundation for the
remaining submodule campaigns.

```text
CORPUS
Review          COMPLETE
Implementation  COMPLETE
Maintenance     NORMAL / CURRENT
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

Do not reopen Corpus architecture merely because the next submodule needs an
adapter. First preserve the Corpus contract and adapt at the consuming boundary.

The live cross-module contracts are registered in `_maintenance/REGISTRY.md`.

The recommended live maintenance tree becomes only:

```text
scikitplot/corpus/
├── MAINTAINING.md
└── _maintenance/
    ├── RULESET.md
    ├── REGISTRY.md
    ├── VERIFICATION.md
    ├── HISTORY.md
    └── STATE.json
```

Why I recommend removing rather than updating the old giant guide:

```text
RULESET
    │
    ├── what must remain true
    │
REGISTRY
    │
    ├── what exists now
    ├── what remains open
    │
VERIFICATION
    │
    ├── how to prove changes
    │
STATE.json
    │
    └── where to continue

HISTORY
    └── why we got here
```

Recommended project sequence now:

```text
CORPUS
Review          ✅
Implementation  ✅
Maintenance     ✅ proposed cleanup
        │
        ▼
MCP
M00 deferred issue revalidation
→ final closure
        │
        ▼
ANNOY
A00 clean native build
→ native/type/lifecycle/mmap review
        │
        ▼
CLI
C00 bootstrap baseline
→ final UX/integration review
        │
        ▼
CROSS-MODULE VERIFICATION
```

After MCP + Annoy + CLI complete, create a project-level document—not another Corpus document:

```text
SCIKITPLOT_CROSS_SUBMODULE_CONTRACT_REGISTRY.md
```

containing only boundaries like:

```text
Corpus RetrievalResponse
        ↓ consumed by
MCP

Corpus VectorIndexBackend
        ↓ implemented by
Annoy adapter

Corpus CapabilityReport
        ↓ consumed by
CLI
```

Final recommended source structure:

```text
scikitplot/corpus/
│
├── MAINTAINING.md
│
└── _maintenance/
    │
    ├── RULESET.md
    │   └── durable architectural rules
    │
    ├── REGISTRY.md
    │   └── current contracts + remaining work
    │
    ├── VERIFICATION.md
    │   └── how future changes are proven
    │
    ├── HISTORY.md
    │   └── compact R00→R16→IMPL18 lineage
    │
    └── STATE.json
        └── machine-readable continuation
```
