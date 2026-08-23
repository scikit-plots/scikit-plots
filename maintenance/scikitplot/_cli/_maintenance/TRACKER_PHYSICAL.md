# Physical Tracker — `scikitplot._cli`

Re-derived from the tree. Do not hand-edit:

```console
$ python scikitplot/_cli/_maintenance/check_trackers.py
```

## Totals

```text
source   33 files /   7383 LOC
tests    14 files /    636 LOC
backup   11 files /   2781 LOC
markdown 10
```

test : source LOC = **0.09**

## Areas

| Area | files | LOC |
|---|---:|---:|
| `(root)` | 14 | 1442 |
| `_backup` | 11 | 2781 |
| `_commands` | 7 | 304 |
| `_frontends` | 3 | 472 |
| `_maintenance` | 9 | 5165 |
| `tests` | 14 | 636 |

## Largest source files

| LOC | File |
|---:|---|
|  3530 | `_maintenance/CLI_SUBMODULE_DESIGN_GUIDE.md` |
|   495 | `_maintenance/CONTRACT.md` |
|   400 | `MAINTAINING.md` |
|   280 | `_maintenance/FINDINGS.md` |
|   244 | `_maintenance/DECISIONS.md` |
|   238 | `_frontends/_argparse.py` |
|   219 | `_frontends/_click.py` |
|   214 | `_maintenance/EXTENDING.md` |

## Delegation targets

```text
  scikitplot.mcp.__main__:main
```

Each names `module:attr`. The gate verifies the module is importable and the
attribute exists — because a dangling target fails at *use* time, in front of a
user.

## Known physical debt

### 1. `_backup/` — 11 files, 2781 LOC of the previous implementation

A complete older CLI: `optparse` and `click` option modules, a `_misc.py`, its
own `cli.py` and command set. **None of it is reachable.** The three largest
files in the whole submodule are all in `_backup/`.

| Verdict | What |
|---|---|
| **ARCHIVE** → `_maintenance/history/` or out of the tree | all of `_backup/` |

Git holds this. Shipping it inside an installed package doubles the apparent
source size and makes "where is the option parsing" ambiguous.

### 2. `__pycache__/` ships in the archive

Recorded as **O-6** during the Corpus campaign and routed here. Still present.
Byte-compiled files in a source archive are stale by definition.

### 3. test : source ratio is 0.09

The suite is well-*designed* — delegation, frontend parity, output contract,
import contract, verbosity and format coverage each have a file. It is thin in
volume, not in intent.
