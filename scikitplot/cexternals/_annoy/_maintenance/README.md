# `_maintenance/` — scikitplot.cexternals._annoy (LIVE v1)

Everything needed to continue work **from a fresh session with no chat history**.

Mirrors `scikitplot/corpus/_maintenance/` and `scikitplot/mcp/_maintenance/`
deliberately: same discipline, same file names, same questions.

> **This submodule is the shared C++ source three others depend on.** Read `FAMILY.md` before changing anything under
> `cexternals/_annoy/src/`.

## Read order

| # | File | Answers |
|---|---|---|
| 1 | `../MAINTAINING.md` | What is this, what state is it in |
| 2 | `FAMILY.md` | **The four-submodule contract** — read this early |
| 2b | `DEPENDENCY_MAP.md` | Antecessors, successors, review order (identical project-wide) |
| 3 | `MAINTENANCE_MODEL.md` | why / when / where / which / how many / how much |
| 4 | `TRACKER_LOGICAL.md` | What this submodule promises |
| 5 | `TRACKER_PHYSICAL.md` | What is on disk; the tripwires |
| 6 | `SUBMODULE_STRUCTURE.md` | Where things go; debt disposition; directions |
| 7 | `VERIFICATION.md` | How to prove the tree is healthy |

Machine-readable: `TRACKER.json`, `STATE.json`.

## First command

```console
$ python scikitplot/cexternals/_maintenance/check_trackers.py
```

Then a **clean build** — this family compiles, so a green tracker does not imply
a working module.

## The one rule

> `cexternals/_annoy` is upstream of three submodules. A change to
> `src/` is never local, whether or not it is treated as one.

The coupling is a **relative path written into Cython source**, not an
`include_directories` entry — invisible to anything reading `meson.build` alone.
`check_trackers.py` verifies every such reference resolves.
