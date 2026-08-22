# `_maintenance/` — `scikitplot.cython` (LIVE v1)

Self-contained: a fresh session needs no chat history.

Mirrors `corpus`, `mcp`, the annoy family, the sphinx family and `_cli`.

## Read order

| # | File | Answers |
|---|---|---|
| 1 | `../MAINTAINING.md` | What is this, what state is it in |
| 2 | `DEPENDENCY_MAP.md` | Why "no edges" still means an unusual dependency |
| 3 | `MAINTENANCE_MODEL.md` | why / when / where / which / how many / how much |
| 4 | `TRACKER_LOGICAL.md` | Contracts, and **what single-process testing cannot prove** |
| 5 | `TRACKER_PHYSICAL.md` | Inventory; tripwires; the template families |
| 6 | `SUBMODULE_STRUCTURE.md` | Where things go; debt; what not to do |
| 7 | `VERIFICATION.md` | Proof, and what a green suite does not establish |

Pre-existing and still authoritative: `../ADR-0001-runtime-lifecycle.md`,
`../OPERATIONS.md`, `../DEV_NOTES.md` — all three are **tested** against the code.

## First command

```console
$ python scikitplot/cython/_maintenance/check_trackers.py
$ python -m pytest scikitplot/cython -q -p no:cacheprovider
```

Read the **skip count**: this suite needs a working Cython and C toolchain, and
a skipped build test looks like a passing one in `-q`.

## The one rule

> This submodule compiles and loads code the caller supplies. Never let an
> operation succeed on **unvalidated input**, and never leave a resource in a
> state the next process cannot reason about.

The second half is not theoretical: a non-blocking lock probe once destroyed live
locks held by other processes.
