# Maintenance Model — `scikitplot.memmap`

> **This submodule is one of four sharing a single C++ source tree.**
> `cexternals/_annoy` (the source) → `annoy`, `memmap`, `random` (the consumers).
> Read `_maintenance/FAMILY.md` before changing anything under
> `cexternals/_annoy/src/`.

The six questions, answered once, so no future session reconstructs them from a
chat log.

---

## WHY — what maintenance is for here

**Role: memory-mapping over the shared `mman.h` shim.**

`scikitplot.memmap` wraps the portability shim `mman.h` from
`cexternals/_annoy/src/`, so that mmap behaves consistently across
POSIX and Windows.

It is small — 4 Python files, 2 Cython — and its failure mode is
**inherited**: it depends on a header owned by another submodule,
through a relative path, with `'src'` commented out of its own
include directories.

A resource-lifetime module is also the one place where a silent
failure is least acceptable: an unclosed mapping or a
double-unmap does not raise where it happened.

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
scikitplot/memmap/
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
| `_memmap/mem_map.pyx` / `.pxd` | the mmap binding |
| `_memmap/mem_map.pyi` | the typed surface |

**Out of scope:** the other three submodules in the family, `scikitplot.corpus`,
`scikitplot.mcp`, `scikitplot._cli`.

---

## HOW MANY

```text
source files      4      source LOC     2140
test files        2      test LOC        710
markdown files    2
```

| Metric | Now | Tripwire |
|---|---|---|
| `cdef extern` sites against `cexternals` | 2 | any new one without an ABI note |
| test : source LOC | see TRACKER.json | < 0.50 |

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
