# Maintaining `scikitplot.corpus`

> **Current state (2026-10-16).** The runtime is present and under the round 33
> review; its active plan, open decisions and results are in
> [`todo/todo.md`](todo/todo.md). The trackers below predate round 33 (stale
> fingerprints and inventories); refreshing them is open decision D6 there.
> Where every other record lives: [`maintenances/README.md`](../README.md).

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

## Reader and filter changes: snapshot before and after

Unit tests check the cases someone thought of. A change to a reader, the
default filter or encoding detection can alter files nobody wrote a test
for (line endings, an encoding, page order). Before such a change is
delivered, read a real folder of files with the old tree and the new one
and compare:

```sh
T=maintenances/corpus/_maintenance/tools/reader_snapshot.py
PYTHONPATH=<old tree> python -B $T snapshot FOLDER --out before.json
PYTHONPATH=<new tree> python -B $T snapshot FOLDER --out after.json
python -B $T compare before.json after.json             # everything
python -B $T compare before.json after.json --text-only # text and errors
```

Add `--documents` to both snapshots to include the default filter, and
`--reader-kwargs '{"ocr": "auto"}'` for options. `--include GLOB` and
`--exclude GLOB` (repeatable, case ignored, `*` crosses folders) choose the
files: `--include '*.pdf'` for a PDF change, `--exclude '*.jpg' --exclude
'*.gif'` to skip slow image OCR. Use the same patterns for both snapshots;
`compare` starts with a `selection:` line when they differ. Every file that changes is
either the intended change or a regression to fix; record the result in the
round's evidence log. Each file is read twice, and `unstable` means two
reads of the same bytes gave different data: a finding in itself. Running
the same tree under two interpreters (one with an optional library, one
without) shows results that depend on what is installed.

Tool tests: `python -B maintenances/corpus/_maintenance/tests/test_reader_snapshot.py`.

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
