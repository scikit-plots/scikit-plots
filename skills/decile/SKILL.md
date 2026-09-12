---
name: decile-maintainer
description: Maintain scikitplot.decile across the current ModelPlotPy API, legacy modelplotpy compatibility layer, and KDS adapter; protect ntile semantics, class-selection propagation, plotting/financial invariants, RNG hygiene, tests, and release evidence without taking ownership of shared validation/plotting utilities.
---

# `scikitplot.decile` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns the **decile subsystem contract**, not generic sklearn, pandas, matplotlib, or the shared scikitplot validation/plotting infrastructure.

## Read first

1. `maintenances/decile/MAINTAINING.md`
2. `maintenances/decile/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/decile/_maintenance/STATE.json`
4. `maintenances/decile/_maintenance/FAMILY.md`
5. `maintenances/decile/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/decile/_maintenance/check_trackers.py --json
python -B maintenances/decile/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/decile/_maintenance/tests -q -p no:cacheprovider
```

## Treat the three surfaces separately

`scikitplot.decile` has a current top-level ModelPlotPy implementation, a user-visible legacy `scikitplot.decile.modelplotpy` implementation, and a vendored KDS adapter. Never use tests from one surface to claim the other two are healthy. Preserve namespace/API compatibility intentionally; do not quietly redirect legacy imports to the current implementation without migration evidence.

## Protect ntile and ranking semantics

The current implementation intentionally assigns descending ntiles deterministically. Preserve stable tie handling, sample/index alignment, finite-value checks, `ntiles <= n_samples`, consistent classes across models, origin rows, cumulative metrics, and plotting-scope invariants. Changes to ranking can alter business decisions even when plots still render.

## KDS argument propagation is a hard contract

KDS plot/report APIs expose `class_index` and `pos_label`; those selections must reach `decile_table`. `report(digits=...)` must control `decile_table`'s `digits` parameter, not disappear through `**kwargs`. Child calls made by `report()` must preserve the same class selection. Current finding `DEC-KDS-001` tracks these silent drops. Add negative tests for non-default class indexes/labels and non-default digits before marking it resolved.

## Legacy compatibility must not poison caller global state

The legacy `modelplotpy.ModelPlotPy` remains exported in its namespace, so its behavior is still maintained behavior. Current finding `DEC-RNG-001` tracks `np.random.seed(self.seed)` inside score preparation. Prefer a local generator/deterministic tie-breaking strategy; never reseed the process-global NumPy RNG as an implementation detail. Test the caller RNG stream before and after legacy API calls.

## Keep financial metrics internally consistent

For cost/revenue/profit/ROI plots, validate required columns and numeric parameters, preserve cumulative totals, avoid mutating the caller's `plot_input`, and keep comparison scopes consistent across models/datasets/target classes. Rendering success alone is insufficient—assert metric relationships and input immutability.

## Shared decorators remain externally owned

`_preprocess`, validation decorators, `save_plot_decorator`, `_docstring`, `_testing`, and seaborn compatibility helpers are dependencies, not decile implementation. A harness may stub them only to isolate decile logic. Do not copy those utilities into decile to make a partial checkout green.

## Plot tests need semantic assertions

Matplotlib `Axes` return-type smoke tests are useful but weak. Maintain tests for line/reference values, selected groups, highlights, percent formatting, comparison scope, input immutability, and figure lifecycle. Use a non-interactive backend in CI and close figures deterministically.

## Evidence ladder

Keep these separate:

1. static contract + mutation tests;
2. current ModelPlotPy isolated tests;
3. legacy ModelPlotPy isolated tests + RNG negative probe;
4. KDS isolated tests + argument-propagation negative probes;
5. complete-package tests with real shared decorators/validation;
6. installed-package/API/docs/dependency/platform evidence.

Release remains blocked if level 5-6 is unavailable or either owned runtime finding remains open.

## Review discipline

When a change touches more than one surface, run each surface's focused tests before the combined suite so failures retain ownership. Record behavioral probes separately from static structure checks. Do not turn a compatibility namespace into an undocumented alias merely to reduce code duplication.

For bug fixes, add a regression that fails on the old behavior and passes for the intended contract. Prefer deterministic numeric assertions over image snapshots where possible; image-level evidence is supplemental because backend/font differences can create noise.
