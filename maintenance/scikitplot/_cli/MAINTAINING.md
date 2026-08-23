# Maintaining `scikitplot._cli`

Entry point. **Self-contained**: a fresh session needs no chat history.

```text
archive: scikit-plots.zip
sha256:  59bfa61efc838a2e1daa17335a7f861f9d5232fd69930140455533a461385950
```

## Position in the project

**`_cli` is the terminal consumer — nothing depends on it**, which is why it is
reviewed last. Every command it exposes is a surface over a submodule whose
contracts are still settling.

```text
corpus ✅  →  mcp M00  →  annoy A00  →  sphinx S00  →  _cli C00   ← last
```

## Read order

1. `_maintenance/DEPENDENCY_MAP.md` — **string delegation, not import**
2. `_maintenance/MAINTENANCE_MODEL.md`
3. `_maintenance/TRACKER_LOGICAL.md`
4. `_maintenance/TRACKER_PHYSICAL.md`
5. `_maintenance/SUBMODULE_STRUCTURE.md`
6. `_maintenance/VERIFICATION.md` — what is not verified

Do not create new parallel files named `FINAL`, `REVISED`, `EXPANDED`,
date-suffixed, or chat-specific variants inside the source tree.

Still authoritative from the pre-existing set: `CLI_SUBMODULE_DESIGN_GUIDE.md`,
`CONTRACT.md`, `DECISIONS.md`, `EXTENDING.md`, `RULESET.md`.

Also consider `_maintenance/history/` files that can be include
useful helper or logic.

## Run this first

```console
$ python scikitplot/_cli/_maintenance/check_trackers.py
$ python -m pytest scikitplot/_cli -q -p no:cacheprovider
$ python -m scikitplot._cli --help
```

## Current state

```text
source   33 files /   7383 LOC
tests    14 files /    636 LOC
backup   11 files /   2781 LOC   <- unreachable; archive it
open findings: 6   (C00 input — revalidate, do not accept)
```

**Do not begin implementation.** Establish the big picture first, as the four
campaigns before this one did.

## The one rule

> `_cli` delegates by **string** — `"scikitplot.mcp.__main__:main"` — resolved at
> runtime. It must never import a delegated submodule, or every optional
> dependency becomes mandatory for `--help`.

The cost of that design: a dangling target fails **in front of a user**, at the
moment they run the command. The gate resolves every target with
`importlib.util.find_spec`, which locates a module *without importing it*.

## What this submodule already does well

It is the **best-instrumented** of the five. It already has an import-contract
test and a frontend-parity test — the exact kind of check the other campaigns had
to add from scratch. The gate contributes only what was missing: verifying the
delegation strings point at something real.
