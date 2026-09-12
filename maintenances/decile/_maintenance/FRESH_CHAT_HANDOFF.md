# Fresh-chat handoff — `scikitplot.decile`

Read `MAINTAINING.md`, `STATE.json`, `FAMILY.md`, and `VERIFICATION.md`, then run:

```sh
python -B maintenances/decile/_maintenance/check_trackers.py --json
python -B maintenances/decile/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/decile/_maintenance/tests -q -p no:cacheprovider
```

Expected state after the 2026-09-12 review is **maintenance PASS / runtime structural FAIL / release BLOCKED**.

Open findings:

1. `DEC-KDS-001`: KDS plot functions do not forward `class_index`/`pos_label` into `decile_table`; `report()` additionally sends `digits` as `round_decimal`, which `decile_table(**kwargs)` silently ignores, and its child plot calls also drop the class-selection arguments.
2. `DEC-RNG-001`: legacy `modelplotpy.ModelPlotPy.prepare_scores_and_ntiles()` invokes `np.random.seed(self.seed)` and changes caller global RNG state. Do not remove the legacy namespace silently; repair or formally deprecate/disposition its contract.

Isolation evidence is strong (336 tests + 6 subtests), but the archive omits the real shared preprocess/validation/plotting support modules. Do not promote harness results to complete-package integration evidence.
