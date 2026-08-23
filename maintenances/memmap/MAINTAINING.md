# Maintaining `scikitplot.memmap`

Entry point for future maintenance. **Self-contained**: a fresh session needs no
chat history to continue from here.

```text
archive: scikit-plots.zip
sha256:  a7941cb07e34fb8225061ad0fa1d7f08b66e75afffb1ba3707380d255e37bd9f
```

## This submodule is part of a family

`cexternals/_annoy` (C++ source) → `annoy`, `memmap`, `random` (consumers).
**This one is a consumer of `cexternals/_annoy/src/`.**

Read `_maintenance/FAMILY.md` before changing anything under
`cexternals/_annoy/src/`. A change there is never local.

## Read order for a fresh chat

1. `_maintenance/FAMILY.md` — the four-submodule contract
2. `_maintenance/MAINTENANCE_MODEL.md` — why / when / where / which / how many / how much
3. `_maintenance/TRACKER_LOGICAL.md` — what this submodule promises
4. `_maintenance/TRACKER_PHYSICAL.md` — what is on disk; tripwires
5. `_maintenance/SUBMODULE_STRUCTURE.md` — where things go; debt disposition
6. `_maintenance/VERIFICATION.md` — how to prove the tree is healthy

`_maintenance/HISTORY.md` only when historical rationale is needed.
Machine-readable: `_maintenance/STATE.json`, `_maintenance/TRACKER.json`.

Do not create new parallel files named `FINAL`, `REVISED`, `EXPANDED`,
date-suffixed, or chat-specific variants inside the source tree.

## Run this first

```console
$ python scikitplot/memmap/_maintenance/check_trackers.py
```

Then a **clean build**. Unlike Corpus, this family compiles — a green tracker
check on an unbuilt tree is necessary, not sufficient.

## Current state

```text
source files    4   source LOC    2140
test files      2   test LOC       710
markdown        2
open findings   3   (A00 input — revalidate, do not accept)

Corpus   review + implementation COMPLETE
MCP      maintenance set ready; M00 pending
ANNOY    maintenance set ready; run A00 NEXT
CLI      after ANNOY
```

**Do not begin implementation.** Establish the big picture across A00–A21 first,
exactly as Corpus and MCP did. The Corpus campaign's value came from 55 findings
and **23 disproofs** before any code — which is why 18 implementation increments
ran without a red suite.

## The one rule

> `cexternals/_annoy` is upstream of three submodules. The coupling is a
> **relative path written into Cython source**, not an `include_directories`
> entry — invisible to anything reading `meson.build` alone.

`check_trackers.py` verifies every such reference resolves, and that `cexternals`
never imports a Python layer built on top of it.
