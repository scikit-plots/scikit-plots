# `_maintenance/` — MCP maintenance set (LIVE v1)

Everything needed to continue work on `scikitplot.mcp` **from a fresh session
with no chat history**. Nothing here depends on a transcript.

Mirrors the structure of `scikitplot/corpus/_maintenance/`, deliberately: the two
modules are reviewed by the same discipline, so a maintainer moving between them
finds the same files answering the same questions.

## Read order for a fresh session

| # | File | Answers |
|---|---|---|
| 1 | `../MAINTAINING.md` | What is this module, what state is it in |
| 2 | `MAINTENANCE_MODEL.md` | **why / when / where / which / how many / how much** |
| 3 | `RULESET.md` | Durable rules — read before changing anything |
| 4 | `TRACKER_LOGICAL.md` | Contracts, invariants, **and Corpus-derived drift** |
| 5 | `TRACKER_PHYSICAL.md` | What is on disk; the tripwires |
| 6 | `SUBMODULE_STRUCTURE.md` | Where a new thing goes; doc debt; directions |
| 7 | `MCP_COMPATIBILITY_POLICY.md` | Python/SDK tiers; the single-wire-protocol rule |
| 8 | `DESIGN.md` | Intent — why the module is shaped this way |
| 9 | `VERIFICATION.md` | How to prove the tree is healthy |

Machine-readable: `TRACKER.json`, `STATE.json`.
Historical rationale only: `history/`.

## First command in any session

```console
$ python scikitplot/mcp/_maintenance/check_trackers.py
$ python -m pytest scikitplot/mcp -q -p no:cacheprovider
```

The gate checks physical drift, tripwires, **the two boundary import rules**, and
that every contract names a module that exists.

> **Known state:** the suite currently reports `2 skipped, 4 errors` without
> `[mcp]` installed — four test files import `pydantic` unguarded. That is a
> recorded finding, not a surprise. See `TRACKER_LOGICAL.md` §4.

## The one rule behind all the others

MCP is an **adapter**. Its failure mode is not producing bad results — it is
**leaking across a boundary**.

> MCP owns the **wire**. Corpus owns **retrieval semantics**. Neither imports the
> other's concerns. When one needs something from the other, a **contract** is
> published — not an import added.

Two mechanically-checked consequences:

- `_server.py` is the **only** module that may import `mcp` or `pydantic`;
- `corpus` and `annoy` are **never** imported at module scope — they arrive by
  injection through the `DocsRetriever` Protocol.

Both are enforced by `check_trackers.py`.

## Layout

```text
scikitplot/mcp/
├── MAINTAINING.md
├── README.md
└── _maintenance/
    ├── README.md                this file
    ├── MAINTENANCE_MODEL.md     the six questions              [NEW]
    ├── TRACKER_LOGICAL.md       contracts + Corpus drift       [NEW]
    ├── TRACKER_PHYSICAL.md      inventory + tripwires          [NEW]
    ├── SUBMODULE_STRUCTURE.md   where things go; doc debt      [NEW]
    ├── TRACKER.json             both trackers                  [NEW]
    ├── check_trackers.py        drift + boundary gate          [NEW]
    ├── RULESET.md               durable rules
    ├── MCP_COMPATIBILITY_POLICY.md
    ├── DESIGN.md
    ├── VERIFICATION.md          proof procedures
    ├── HISTORY.md               compressed history
    ├── STATE.json               campaign state
    ├── STALE_FILES.md + stale_lifecycle.py
    ├── CI_OUTPUT_ROUTING.md  DOCKER.md  IDEMPOTENT_TESTING.md
    ├── STRICT_WIRE_VALIDATION.md  UNKNOWN_ARGUMENTS_AND_MANIFESTS.md
    └── history/                 superseded, point-in-time docs
```

`SUBMODULE_STRUCTURE.md` §3 lists which of the pre-existing markdown files to
archive, fold or drop — 16 live files reduce to 9 without losing a decision.
