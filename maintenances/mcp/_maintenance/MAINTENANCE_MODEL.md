# MCP Maintenance Model

The six questions, answered once, so no future session reconstructs them from a
chat log.

---

## WHY — what maintenance is for here

`scikitplot.mcp` is an **adapter**. It exists to expose retrieval over the Model
Context Protocol without owning retrieval, and without making the rest of
`scikit-plots` depend on an SDK.

Its failure mode is therefore not the same as Corpus's. Corpus fails by
producing plausible-but-incomplete *results*. MCP fails by **leaking across a
boundary**:

| Leak | What it looks like |
|---|---|
| Wire types drift inward | `pydantic`/`mcp` models appear in Corpus, and Corpus now owns a protocol it does not control |
| Retrieval logic drifts outward | MCP reimplements scoring or fusion, and two implementations disagree |
| SDK import escapes `_server.py` | Importing `scikitplot.mcp` requires the SDK, and the optional dependency stops being optional |
| A capability is claimed, not probed | Clients call a tool this build cannot serve |

**Maintenance has one governing purpose: keep the boundary where it is.**

> MCP owns the **wire**. Corpus owns **retrieval semantics**. Neither imports the
> other's concerns. When one needs something from the other, a **contract** is
> published — not an import added.

The corollary, inherited from the Corpus campaign and equally true here:

> An unverified claim is worse than a narrow one. Prefer `UNKNOWN` or a declared
> degradation over a confident guess.

---

## WHEN — what triggers maintenance work

| Trigger | Response |
|---|---|
| A new tool or resource is added | Schema + strict validation + `server_capabilities()` entry, in the same commit |
| The MCP SDK releases a new major | `M11` compatibility run; the single-wire-protocol rule decides, not convenience |
| Corpus renames or reshapes a contract | **Check §"Corpus-derived drift" in `TRACKER_LOGICAL.md`** — most such changes hit docs here, not code |
| A test starts requiring an optional dependency | Guard it with `importorskip`, or the suite stops collecting without `[mcp]` |
| `check_trackers.py` fails | Physical drift or a contract naming a missing module |
| Anything imports `mcp` or `pydantic` outside `_server.py` | Revert; that is the boundary |
| A capability report gains an entry | It must be *probed*, not assumed |
| Before starting ANNOY or CLI | Read `REGISTRY.md` cross-module boundaries |

**Not a trigger:** elapsed time. Maintenance is event-driven.

---

## WHERE — the layout

```text
scikitplot/mcp/
├── MAINTAINING.md              entry point for a human or a fresh AI session
├── README.md                   user-facing
└── _maintenance/
    ├── README.md               read order + first command
    ├── MAINTENANCE_MODEL.md    this file
    ├── RULESET.md              durable rules
    ├── TRACKER_LOGICAL.md      contracts, invariants, Corpus-derived drift
    ├── TRACKER_PHYSICAL.md     on-disk inventory + tripwires
    ├── SUBMODULE_STRUCTURE.md  where a new thing goes; debt; directions
    ├── REGISTRY.md             open findings, boundaries, run sequence
    ├── VERIFICATION.md         how to prove the tree is healthy
    ├── HISTORY.md              compressed history
    ├── TRACKER.json            both trackers, machine-readable
    ├── STATE.json              campaign state, machine-readable
    ├── check_trackers.py       drift gate
    └── history/                superseded, point-in-time docs
```

**Two trackers, because they rot differently.** Physical drifts silently — a
module grows, nothing breaks — so it is a **gate**, not a document. Logical
records *why* a contract is shaped as it is, which cannot be re-derived from the
tree.

The rule: **a document that describes what a script can check should be replaced
by the script; one that records why cannot be.**

---

## WHICH — what is in scope

| In scope | Out of scope |
|---|---|
| `scikitplot/mcp/**` | `scikitplot/corpus`, `scikitplot/annoy`, `scikitplot/_cli` |
| The MCP wire protocol and its schemas | Retrieval scoring, fusion arithmetic, index formats |
| Tool/resource definitions and validation | What a tool *retrieves* — that is a `DocsRetriever` |
| The `[mcp]` extra and SDK tiering | Corpus's `[corpus]` extras |
| `server_capabilities()` truth | Corpus's `CapabilityStatus` model (consumed, not redefined) |

**The seam:** `DocsRetriever` is a Protocol. MCP declares the shape it needs;
implementations are **injected**. `CorpusAnnoyRetriever` performs the real wiring
and is the only place Corpus and Annoy meet MCP — behind that Protocol, at call
time.

---

## HOW MANY — the numbers that bound a change

Derived from the live tree, re-derived by `check_trackers.py`:

```text
source files    16      source LOC    3 182
test files      15      test LOC      2 096
markdown files  31      contracts         7
```

MCP is **17× smaller than Corpus** in source LOC. That matters for
proportionality: a change here that touches three files is a large change.

| Ratio | Now | Tripwire |
|---|---:|---|
| test : source LOC | 0.66 | falling below 0.50 |
| markdown : source files | **1.9** | rising above 2.5 |
| largest module | see `TRACKER_PHYSICAL.md` | above 1 200 LOC |
| modules importing the SDK | **1** (`_server.py`) | **any** second module |

The markdown ratio is the unusual one. **31 documentation files for 16 source
files** is the module's most visible debt — see `SUBMODULE_STRUCTURE.md` §3.

---

## HOW MUCH — proportionality

> **Match the effort to the blast radius, and the evidence to the claim.**

| Change | Required evidence |
|---|---|
| Docstring, comment | green suite |
| New test | the test itself |
| New tool/resource | schema test + strict-validation test + capability entry |
| Changing a wire schema | round-trip test + a note in `HISTORY.md` |
| Touching `_server.py`'s SDK import | proof the module still imports without the SDK |
| Adding a dependency | justification in `RULESET.md`; ranges, never `==` |
| Claiming SDK compatibility | tested against that SDK, or the claim is not made |

Three anti-patterns worth naming, because this module is *structurally* prone to
them:

1. **Reimplementing retrieval "just for MCP."** Fusion, scoring and ranking
   belong to Corpus. If MCP needs different behaviour, the contract changes —
   not a second implementation.
2. **Widening `DocsRetriever` to fit one backend.** That turns a Protocol into a
   capability lottery, where some implementations answer and others cannot.
3. **Documenting instead of testing.** With 31 markdown files against 16 source
   files, this module's habit is to write it down. Write the test.

**The most useful habit:** when a boundary property is verified, make it a test.
The single-SDK-import rule, the no-corpus-at-module-scope rule and the
capability-is-probed rule are all mechanically checkable — and a checked rule
survives a refactor that a documented one does not.
