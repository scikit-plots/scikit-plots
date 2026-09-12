---
name: seaborn-maintainer
description: Maintain scikitplot.seaborn high-level plotting wrappers; protect seaborn/matplotlib compatibility, estimator semantics, weighted deciles, hue mapping, plotting values/artists, focused tests, and release evidence.
---

# `scikitplot.seaborn` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns `scikitplot.seaborn`, not the external seaborn project and not `scikitplot.externals._seaborn`.

## Read first

1. `maintenances/seaborn/MAINTAINING.md`
2. `maintenances/seaborn/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/seaborn/_maintenance/STATE.json`
4. `maintenances/seaborn/_maintenance/FAMILY.md`
5. `maintenances/seaborn/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/seaborn/_maintenance/check_trackers.py --json
python -B maintenances/seaborn/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/seaborn/_maintenance/tests -q -p no:cacheprovider
```

## Keep the four plotting families separate

`aucplot`, `evalplot`, `decileplot`, and `modelplot` share VectorPlotter-style plumbing but have different semantic owners. A green ROC computation does not bless decile weighting; a green confusion-matrix renderer does not prove model-attribute visualization.

## Treat seaborn private APIs as volatile

The runtime imports `VectorPlotter`, `_docstrings`, `_statistics`, `_compat`, and `_default_color` from private seaborn namespaces. Private helpers are compatibility risks, not stable contracts. In particular, never call `_default_color` directly and assume its method-name introspection will survive Matplotlib decorators. Current finding **SBN-COMPAT-001** reproduces on seaborn 0.13.2 + Matplotlib 3.10.8 because `Axes.plot.__name__ == "wrapper"`, so `_default_color` returns `None` and plotting reaches `to_rgba(None)`.

Prefer a project-owned compatibility resolver with focused tests over scattered direct private-helper calls. Test installed-seaborn and vendored-fallback lanes separately; one must not silently bless the other.

## `modelplot` must be about the estimator

`modelplot(kind="feature_importances")` advertises model-instance attribute visualization. `x_estimator` must therefore be operationally consumed and feature names/importances must drive the artists. Current **SBN-MODEL-001** is critical: the parameter is unused and the implementation delegates to a confusion-matrix renderer over `x/y` labels.

A valid repair needs adversarial tests: estimator-only invocation, reordered feature importances, feature-name alignment, negative/zero importances if supported, estimators lacking the requested attribute, and stable axes labeling. Merely referencing `x_estimator` to satisfy a static checker is not a repair.

## Weighted decile semantics must be real

`decileplot(weights=...)` documents observation weights. Validating the vector and then discarding it is a contract violation. Current **SBN-DEC-001** shows weighted and unweighted tables are identical because `_sw` is not forwarded to `compute_decile_table`.

Before choosing a repair, define weighted semantics explicitly: weighted responders/nonresponders, population totals, random/wizard baselines, cumulative percentages, lift, and KS. Then test integer weights against row replication where mathematically equivalent. If weights are not supported, remove/reject the public parameter rather than silently ignoring it.

## Forward semantic mapping inputs

Seaborn-style APIs must honor `palette`, `hue_order`, and `hue_norm`. `aucplot`, `evalplot`, and `modelplot` already pass them to `map_hue`; `decileplot` currently calls bare `map_hue()` and ignores its public arguments (**SBN-DEC-002**). Test actual artist colors/order, not only number of lines.

## Test values as well as artists

A returned `Axes` is weak evidence. For ROC/PR verify curve coordinates, AP/AUC summaries, sample weights, degeneracy warnings, baselines, and hue subsets. For confusion matrices verify matrix values, normalization, classification-report content, labels, sample weights, probability-threshold behavior, annotations, and colorbar semantics. For deciles verify the table before verifying the picture.

## Preserve x/y/hue data semantics

All wrappers use VectorPlotter-style `data`, `x`, `y`, and `hue`. Maintain parity between string column references, NumPy arrays, and pandas Series. NaN dropping, finite-value validation, label domains, score ranges, and per-subset iteration must stay explicit. Do not let a plotting exception get disguised as a data-validation warning or vice versa.

## Dependency compatibility is a release dimension

Record exact seaborn, Matplotlib, pandas, NumPy, and sklearn versions for runtime evidence. The submodule relies on private seaborn internals and therefore needs a tested version matrix or a robust vendored compatibility path. One Linux environment is not a release matrix.

## Keep fallback ownership honest

Installed seaborn is preferred; `scikitplot.externals._seaborn` is the fallback in a complete project. An `ImportError` in any private installed-seaborn symbol can switch the whole import block to the vendored path. Test both paths intentionally. Do not classify a missing vendored subtree in a partial archive as a runtime algorithm defect.

## Do not confuse archive import shadowing with runtime behavior

This accumulated snapshot lacks `scikitplot/__init__.py`. Running pytest directly causes `scikitplot/seaborn` to become top-level package `seaborn`, shadowing the external dependency. Use a throwaway package-root harness for evidence and never modify runtime files merely to make the partial archive importable.

## Evidence ladder

Keep these levels distinct:

1. static contract + mutation tests;
2. package-root native focused tests against installed dependencies;
3. diagnostic compatibility harness used only to isolate dependency seams;
4. negative semantic probes for estimator, weights, hue mapping, metric values, and annotations;
5. complete-project installed-seaborn lane;
6. complete-project vendored-fallback lane;
7. supported dependency-version/platform matrix and docs/examples.

Release remains blocked while **SBN-COMPAT-001**, **SBN-MODEL-001**, **SBN-DEC-001**, or **SBN-DEC-002** is open, or while complete-project/fallback evidence is unavailable.

## Runtime edits require focused regressions

For every repair, add the smallest regression that fails on the old behavior and proves the semantic outcome, then run all four focused modules plus the maintenance mutation suite. Do not update runtime fingerprints before the new runtime and evidence agree.

## Plane separation

Runtime code under `scikitplot/seaborn` must never import `maintenances` or `skills`. Maintenance tools must inspect source structurally rather than importing `scikitplot.seaborn`, so they remain usable even when plotting dependencies are broken.
