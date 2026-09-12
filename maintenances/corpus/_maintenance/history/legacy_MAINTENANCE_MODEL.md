# Corpus Maintenance Model

The six questions this document answers, once, so no future session has to
reconstruct them from a chat log.

---

## WHY — what maintenance is for here

Corpus is a retrieval library. Its failure mode is not crashing; it is
**producing plausible output that is wrong or incomplete without saying so**.

A review campaign found that shape at **seven** independent sites:

| Site | It looked like |
|---|---|
| unsupported filter | success — it returned the *whole corpus* |
| hybrid missing a leg | success — fewer hits, exactly halved scores, same label |
| mixed embedding models | success — ranked hits across two incompatible spaces |
| inert hierarchy | present — recorded but unwalkable and unvalidatable |
| capability probe | absent — a *broken* backend reported as never installed |
| truncated traversal | complete — no signal the budget stopped it |
| deprecated shim | fine — until warnings-as-errors made 49 files uncollectable |

**Maintenance therefore has one governing purpose: keep every operation able to
say what it actually did.** Everything below serves that.

The corollary, and the rule that has decided the most arguments:

> An unverified claim is worse than a narrow one. Prefer `UNKNOWN`, `REJECTED`
> or `DEGRADED` over a confident guess.

---

## WHEN — what triggers maintenance work

| Trigger | Response |
|---|---|
| A new backend, reader, chunker or adapter is added | Run the **submodule review** (`SUBMODULE_STRUCTURE.md` §5) before merging |
| A contract in `TRACKER_LOGICAL.md` gains a field | Update the tracker in the same commit; a tracker that lags is worse than none |
| `check_trackers.py` fails | Physical drift — reconcile before anything else |
| A test is deleted or weakened | **Stop.** Justify in `HISTORY.md`, or revert |
| A dependency raises its floor | Re-check the 3.8→3.15+ audit (`RULESET.md`) |
| An optional heavyweight is imported at module scope | The import gate will fail — move it to call time |
| A finding in `REGISTRY.md` is claimed fixed | Attach the proof test; no claim without one |
| Before starting MCP / ANNOY / CLI | Read `REGISTRY.md` §cross-module boundaries first |

**Not a trigger:** elapsed time. There is no scheduled sweep. Maintenance is
event-driven, because a calendar-driven review reads everything and sees
nothing.

---

## WHERE — the two trackers, and why there are two

```text
scikitplot/corpus/
├── MAINTAINING.md              entry point for a human or a fresh AI session
└── _maintenance/
    ├── MAINTENANCE_MODEL.md    this file — the six questions
    ├── RULESET.md              durable rules that survive every campaign
    ├── TRACKER_PHYSICAL.md     what is on disk
    ├── TRACKER_LOGICAL.md      what the code promises
    ├── SUBMODULE_STRUCTURE.md  structure review + expansion rules
    ├── REGISTRY.md             contracts, remaining work, boundaries
    ├── VERIFICATION.md         how to prove the tree is healthy
    ├── HISTORY.md              what happened, compressed
    ├── TRACKER.json            both trackers, machine-readable
    ├── STATE.json              campaign state, machine-readable
    └── check_trackers.py       re-derives PHYSICAL and fails on drift
```

**Two trackers, because they rot differently.**

*Physical* drifts silently — a file grows, a subpackage doubles, a module
acquires a fifth responsibility. Nobody notices, because nothing breaks.
`check_trackers.py` re-derives it from the tree and exits non-zero on drift, so
it is a **gate**, not a document.

*Logical* drifts loudly but locally — someone adds a field to a contract and
only the tests near it fail. What is lost is the *invariant*: why the field
exists and what it must never do. That cannot be re-derived from the tree, so it
is written down and reviewed by a human.

A doc that describes what a script can check should be replaced by the script.
A doc that records *why* cannot be, and must be.

---

## WHICH — what is in scope

| In scope | Out of scope |
|---|---|
| `scikitplot/corpus/**` | `scikitplot/mcp`, `scikitplot/annoy`, `scikitplot/_cli` |
| Contracts Corpus publishes | How adapters consume them |
| Optional-dependency **declarations** | Vendoring or bundling those dependencies |
| The `[corpus]` extras split | Other packages' extras |
| Python 3.8 → 3.15+ compatibility | Dropping the floor without evidence |

**Boundary rule.** Corpus owns retrieval semantics — identity, outcome,
capability, provenance. It does **not** own wire formats, CLI presentation, or
native index internals. When those need something, Corpus *publishes a contract*;
it does not import the consumer.

The one live instance to keep honest: `ToolCallInput` / `ToolCallResult` are
protocol-**neutral** payload shapes. Adding a `pydantic` model or an `mcp`
import to them moves wire concerns into Corpus and breaks the boundary. Their
docstrings say so, and a test asserts `_types` imports neither.

---

## HOW MANY — the numbers that bound a change

Derived from the live tree, and re-derived by `check_trackers.py`:

```text
source files    78      source LOC   55 787
test files      78      test LOC     30 809
subpackages     14      contracts    19
```

**Ratios worth watching**, not as targets but as tripwires:

| Ratio | Now | Tripwire |
|---|---|---|
| test LOC : source LOC | 0.55 | falling below 0.40 |
| root-level source LOC share | 39% | rising above 45% |
| largest single module | 3 167 (`_base.py`) | any module above 3 500 |
| subpackages with no tests | 1 (`_maintenance`) | any *code* subpackage |

Root-level share is the one to watch. 26 files and 21 650 LOC sit directly in
`corpus/` rather than in a named subpackage, and `_base.py` alone holds readers,
filters, `PipelineGuard` and `DummyReader` — **four component categories in one
module**. That is the largest known structural debt; `SUBMODULE_STRUCTURE.md`
proposes the split.

---

## HOW MUCH — proportionality

The rule that keeps maintenance from becoming its own project:

> **Match the effort to the blast radius, and the evidence to the claim.**

| Change | Required evidence |
|---|---|
| Docstring, comment, typo | none beyond a green suite |
| New test | the test itself |
| New field with a safe default | a test pinning the default |
| Contract change | a test per invariant + `TRACKER_LOGICAL.md` update |
| New subpackage | submodule review + `check_trackers.py` update |
| Rename | **a cited finding** — preference is not sufficient |
| Removing a capability declaration | proof the capability is genuinely gone |
| Performance claim | before/after measurement, or no claim |

Three anti-patterns this codebase has actually committed and corrected:

1. **Fixing a symptom at the reporting layer** when the root cause was the
   registry. Fix the source; every consumer benefits.
2. **Weakening a test to make a change pass.** Test fakes were upgraded to
   satisfy a new conformance check, not the check relaxed to admit them.
3. **Claiming a capability that had no implementation.** `supports_persistence`
   was declared `False` — accurately — until an artifact existed, then flipped
   with a test asserting both.

**The single most useful habit:** when a property is verified, make it a test.
Seven separate defects in this module existed because a true property was
documented rather than enforced, and then quietly stopped being true.
