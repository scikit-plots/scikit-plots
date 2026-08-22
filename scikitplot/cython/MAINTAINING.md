# Maintaining `scikitplot.cython`

Entry point. **Self-contained**: a fresh session needs no chat history.

```text
archive: scikit-plots.zip
sha256:  59bfa61efc838a2e1daa17335a7f861f9d5232fd69930140455533a461385950
```

## Position: fully independent

**No sibling submodule imports it, and it imports none** — verified by AST. It
can be reviewed, changed and released without coordinating with another campaign.

But its dependency is unusual. The other five depend on *code*; this one depends
on a **toolchain** (Cython, a C compiler) and on **caller-supplied source**.

> Every other submodule processes data. This one processes *programs*.

## Read order

1. `_maintenance/DEPENDENCY_MAP.md` — why "no edges" still means an unusual dependency
2. `_maintenance/MAINTENANCE_MODEL.md`
3. `_maintenance/TRACKER_LOGICAL.md` — **what single-process testing cannot prove**
4. `_maintenance/TRACKER_PHYSICAL.md`
5. `_maintenance/SUBMODULE_STRUCTURE.md`
6. `_maintenance/VERIFICATION.md`

Still authoritative and **tested against the code**:
`_maintenance/ADR-0001-runtime-lifecycle.md`,
`_maintenance/OPERATIONS.md`, `_maintenance/DEV_NOTES.md`.

## Run this first

```console
$ python scikitplot/cython/_maintenance/check_trackers.py
$ python -m pytest scikitplot/cython -q -p no:cacheprovider
```

**Read the skip count.** The suite needs a working Cython and C toolchain, and a
skipped build test looks like a passing one under `-q`.

## Current state

```text
source  191 files /  15898 LOC     (includes 306 template files / 5 709 LOC)
tests    45 files /  11492 LOC     ratio 0.72
open findings: 5   (Y00 input — revalidate, do not accept)
```

Two readings of the same tree, both correct and worth knowing:

* the **gate** counts `_templates/` as source — it is compilable source;
* the **docs** count it separately — 306 files, 22 families, treated as *test
  inputs* by `test__templates_containment.py`.

Excluding templates the ratio is **0.95, the project's highest**; including them
it is 0.72. The gate uses the conservative reading.

**Do not begin implementation.** Establish the big picture first.

## The one rule

> Never let an operation succeed on **unvalidated input**, and never leave a
> resource in a state the next process cannot reason about.

The second half is not theoretical. A non-blocking lock probe with `timeout_s=0`
once **destroyed live locks held by other processes** — a probe that answered its
question by breaking the thing it was asking about, invisible to every
single-process test.

## Why this submodule matters beyond itself

Its 30-finding campaign **established the methodology every other submodule now
uses**: zero hallucination, root-cause fixes only, an always-green gate,
per-turn evidence, guard tests that make findings permanently verifiable.

It was reviewed first and documented last, which is why this set adds little to
its verification and much to the record of why its tests exist.
