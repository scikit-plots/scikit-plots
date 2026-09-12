# Fresh-chat handoff — `scikitplot.corpus`

## Authority

1. User-supplied current workspace and failure logs.
2. Current `scikitplot/corpus/` runtime files.
3. `STATE.json`, `MAINTENANCE.json`, `REVIEW.json`, `TRACKER.json`, and `EVIDENCE.json`.
4. Current tests/examples.
5. Historical notes/checkpoints.

The inherited R00–R16 / IMPL-01–18 narrative is historical until the runtime that produced it is present and reverified. This supplied snapshot has an empty runtime tree, so do not continue from “COMPLETE”.

## Planes

```text
scikitplot/corpus/          runtime + executable tests/examples
maintenances/corpus/        maintenance state + deterministic dev-only checks
skills/corpus/              fresh-chat maintainer skill
```

Runtime must never import `maintenances` or `skills`.

## First commands

```sh
python -B maintenances/corpus/_maintenance/check_trackers.py --json
python -B maintenances/corpus/_maintenance/review_subsystem.py --json
python -B maintenances/corpus/_maintenance/tests/test_contract.py
```

Expected on the supplied partial archive: maintenance `PASS`, runtime `FAIL`, release `BLOCKED`. The exact next action is to restore/provide the actual Corpus runtime snapshot, then rerun the gates before reviewing any historical defect or architecture claim.
