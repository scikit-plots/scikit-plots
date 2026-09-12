# Maintaining `scikitplot.corpus`

This is the live human entry point for Corpus maintenance. The supplied `corpus.zip` snapshot is **not a complete Corpus runtime snapshot**: `scikitplot/corpus/` is empty, while the inherited maintenance records describe a much larger historical implementation. Therefore historical “COMPLETE” claims are provenance, not current truth.

## First commands

```sh
python -B maintenances/corpus/_maintenance/check_trackers.py --json
python -B maintenances/corpus/_maintenance/review_subsystem.py --json
python -B maintenances/corpus/_maintenance/tests/test_contract.py
```

Interpret the three statuses separately:

```text
maintenance_status  -> whether the maintenance/skill plane is internally sound
runtime_status      -> whether the supplied runtime satisfies the declared Corpus contract
release_status      -> whether all declared runtime/test/integration evidence is current GREEN
```

For this supplied archive the expected truth is `maintenance=PASS`, `runtime=FAIL`, `release=BLOCKED`. Do not use `--update` to bless an incomplete runtime.

## Core Corpus rule

Never let an operation succeed on partial evidence. If an operation can partially fail, the public result must carry explicit status/error evidence; otherwise it raises. Prefer `UNKNOWN`, `REJECTED`, or `DEGRADED` to a confident guess.

## Ownership

Corpus owns protocol-neutral evidence and retrieval contracts. MCP owns wire adaptation, Annoy owns native index mechanics, and CLI owns presentation. A downstream adapter need is not permission to move those concerns into Corpus.

## Fresh-session read order

1. `_maintenance/FRESH_CHAT_HANDOFF.md`
2. `_maintenance/STATE.json`
3. `MAINTENANCE.json`
4. `_maintenance/FAMILY.md`
5. `_maintenance/VERIFICATION.md`
6. `_maintenance/REGISTRY.md`

Use `_maintenance/history/` only as historical evidence. Reproduce old claims against a complete current runtime before carrying them forward.
