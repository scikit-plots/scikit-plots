# Fresh-chat handoff — `scikitplot.preprocessing`

Read `MAINTAINING.md`, `STATE.json`, `FAMILY.md`, and `VERIFICATION.md`, then run:

```sh
python -B maintenances/preprocessing/_maintenance/check_trackers.py --json
python -B maintenances/preprocessing/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/preprocessing/_maintenance/tests -q -p no:cacheprovider
```

Expected state after the 2026-09-12 review: **maintenance PASS / runtime structural FAIL / release BLOCKED**.

Open findings:

1. **PRE-DCE-001:** `_build_cache()` flattens all per-feature categories and creates `dict(zip(categories_flat_, ...))`. Duplicate labels across columns overwrite each other. A two-column `x/y` + `x/z` fit advertises four feature names but returns only three encoded columns; inverse transform rejects that matrix.
2. **PRE-DCE-002:** `min_frequency`/`max_categories` computes `_infrequent_indices`, mappings, `_n_features_outs`, and names, but the custom `_transform()` does not apply those mappings. Example: expected two columns (`a`, `infrequent_sklearn`) but transform returns three (`a`, `b`, `c`).
3. **PRE-GD-001:** `GetDummies(handle_unknown=<arbitrary>)` is not validated; unsupported values silently take the non-error path.
4. **PRE-TEST-001:** test docs claim direct standalone execution but bootstrap/path code is commented; direct script execution fails.

Do not repair PRE-DCE-001 by globally uniquifying raw label strings. Output identity must remain feature-local and compatible with names, drop offsets, infrequent grouping, and inverse decoding.
