# `_maintenance/` — `scikitplot._cli` (LIVE v1)

Self-contained: a fresh session needs no chat history.

Mirrors `corpus`, `mcp`, the annoy family and the sphinx family.

## Read order

| # | File | Answers |
|---|---|---|
| 1 | `../MAINTAINING.md` | What is this, what state is it in |
| 2 | `DEPENDENCY_MAP.md` | Where `_cli` sits; **string delegation vs import** |
| 3 | `MAINTENANCE_MODEL.md` | why / when / where / which / how many / how much |
| 4 | `TRACKER_LOGICAL.md` | What `_cli` promises |
| 5 | `TRACKER_PHYSICAL.md` | What is on disk; tripwires |
| 6 | `SUBMODULE_STRUCTURE.md` | Where things go; debt disposition |
| 7 | `VERIFICATION.md` | **What is not verified** |

Pre-existing and still authoritative: `CLI_SUBMODULE_DESIGN_GUIDE.md`,
`CONTRACT.md`, `DECISIONS.md`, `EXTENDING.md`, `RULESET.md`, `INTEGRATION.md`,
`FINDINGS.md`.

## First command

```console
$ python scikitplot/_cli/_maintenance/check_trackers.py
$ python -m pytest scikitplot/_cli -q -p no:cacheprovider
$ python -m scikitplot._cli --help
```

## The one rule

> `_cli` delegates by **string**, never by import. A command must appear in
> `--help` whether or not its submodule is installed — and must fail with a
> message that says what to install.

The gate resolves every delegation target with `importlib.util.find_spec`, which
locates a module **without importing it** — so running the gate does not violate
the import contract it protects.
