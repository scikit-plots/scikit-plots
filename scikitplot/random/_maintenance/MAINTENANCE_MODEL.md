# Maintenance Model — `scikitplot.random`

> **This submodule is one of four sharing a single C++ source tree.**
> `cexternals/_annoy` (the source) → `annoy`, `memmap`, `random` (the consumers).
> Read `_maintenance/FAMILY.md` before changing anything under
> `cexternals/_annoy/src/`.

The six questions, answered once, so no future session reconstructs them from a
chat log.

---

## WHY — what maintenance is for here

**Role: KISS RNG over the shared `kissrandom.h`.**

`scikitplot.random` exposes the KISS RNG from
`cexternals/_annoy/src/kissrandom.h`, with a NumPy-compatible
surface.

Its failure mode is **reproducibility**, which is unusual in this
family: a seeded RNG that changes its stream is not a crash and not
a wrong answer — it is a silently different experiment. Nothing
downstream can detect it.

It shares `kissrandom.h` with `annoy`, so a change there affects
index construction *and* user-facing randomness at once.

The rule inherited from the Corpus and MCP campaigns, and equally true here:

> An unverified claim is worse than a narrow one. Prefer a declared limitation
> over a confident guess.

---

## WHEN — what triggers maintenance work

| Trigger | Response |
|---|---|
| A change under `cexternals/_annoy/src/` | **Check all three consumers**, not just this one |
| A `cdef extern` block is added or edited | Record the ABI it mirrors in `TRACKER_LOGICAL.md` |
| `check_trackers.py` fails | Physical drift or a broken cross-submodule reference |
| A generated file is edited directly | Stop — edit the template |
| A new dtype or specialization is added | Update the support matrix; add a test |
| A test is deleted or weakened | Justify in `HISTORY.md`, or revert |
| Before starting CLI | Read `FAMILY.md` cross-module boundaries |

**Not a trigger:** elapsed time. Maintenance is event-driven.

---

## WHERE — the shared source, and the coupling

```text
scikitplot/cexternals/_annoy/src/        <- the single shared C++ source
    annoylib.h            ──► annoy/_annoy/annoylib.pxd.in      (4 extern sites)
    kissrandom.h          ──► annoy/_annoy/annoylib.pxd.in
                          ──► random/_kiss/kiss_random.pxd
    annoy_type_support.h  ──► annoy/_annoy/annoylib.pyx.in
    mman.h                ──► memmap/_memmap/mem_map.pxd
                          ──► memmap/_memmap/mem_map.pyx
```

**How the coupling is expressed, and why it matters.**  Consumers reach the
headers by a *relative path written into Cython source*:

```cython
cdef extern from "../../cexternals/_annoy/src/kissrandom.h" namespace "Annoy" nogil:
```

Not through `include_directories`.  In `memmap/_memmap/meson.build` and
`random/_kiss/meson.build` the `'src'` entry is present but **commented out**,
so the relative path is the only mechanism.

Two consequences worth stating plainly:

* moving or renaming anything under `cexternals/_annoy/src/` breaks three
  submodules at *Cython compile time*, with an error that names a path rather
  than a contract;
* the coupling is invisible to any tool that reads `meson.build` alone.

This is the family's defining structural fact.  Every maintenance decision below
follows from it.

### Layout

```text
scikitplot/random/
├── MAINTAINING.md              entry point for a human or a fresh AI session
└── _maintenance/
    ├── README.md               read order + first command
    ├── MAINTENANCE_MODEL.md    this file
    ├── FAMILY.md               the four-submodule contract  (identical copy)
    ├── TRACKER_LOGICAL.md      what this submodule promises
    ├── TRACKER_PHYSICAL.md     what is on disk + tripwires
    ├── SUBMODULE_STRUCTURE.md  where things go; debt; directions
    ├── VERIFICATION.md         how to prove the tree is healthy
    ├── HISTORY.md              compressed history
    ├── TRACKER.json            machine-readable
    ├── STATE.json              campaign state
    └── check_trackers.py       drift + cross-submodule gate
```

---

## WHICH — what this submodule owns

| Owns | Purpose |
|---|---|
| `_kiss/kiss_random.pyx` / `.pxd` / `.pxi` | the RNG binding |
| `_kiss/kiss_random.pyi` | the typed surface |
| NumPy-compatible generator surface | documented in `KISSRANDOM_NUMPY_COMPATIBLE_FINAL.md` |

**Out of scope:** the other three submodules in the family, `scikitplot.corpus`,
`scikitplot.mcp`, `scikitplot._cli`.

---

## HOW MANY

```text
source files      5      source LOC     4555
test files        4      test LOC       1435
markdown files    4
```

| Metric | Now | Tripwire |
|---|---|---|
| `cdef extern` sites against `cexternals` | 1 | any new one without an ABI note |
| documented stream-stability guarantee | absent | must exist before 1.0 |

---

## HOW MUCH — proportionality

> **Match the effort to the blast radius, and the evidence to the claim.**

| Change | Required evidence |
|---|---|
| Docstring or comment | green build |
| A test | the test itself |
| A Python-level change | test + green build |
| **A change under `cexternals/_annoy/src/`** | **all three consumers rebuilt and tested** |
| A `cdef extern` change | proof the declaration matches the header |
| A generated-file change | proof the *template* was the thing edited |
| A performance claim | before/after measurement, or no claim |

The one asymmetry in this family: a change in `cexternals` is never local. A
one-line header edit is a three-submodule change whether or not it is treated as
one.
