# Verification

## Static lane

Run the maintenance checker/reviewer and mutation tests. The static gate owns inventory, public symbols, plane separation, current known contract failures, evidence hashes, and fresh-chat continuity.

## Runtime lanes

A minimal package-root harness (`scikitplot/__init__.py` only in a throwaway copy) is required for this archive to avoid importing `scikitplot/seaborn` as top-level `seaborn`.

Native dependency lane on this environment (seaborn 0.13.2, Matplotlib 3.10.8): **102 passed, 48 failed**. All 48 failures are consistent with direct `_default_color` returning `None` for decorated `Axes.plot` methods.

Harness-only compatibility lane: replacing only the imported `_default_color` aliases with a resolver that supplies Matplotlib's default `C0` when the private helper returns `None` yields **150 passed**. This is diagnostic evidence, not a source repair.

Negative probes independently verify:
- `modelplot(x_estimator=...)` alone draws nothing; different feature importances produce identical confusion-matrix images when x/y are held constant.
- weighted and unweighted `decileplot(kind="df")` results are identical.
- custom decile `palette` is ignored; default seaborn colors are used.

## Release lane

Release remains blocked until owned runtime findings are repaired with focused regressions and the complete package is tested with the real root package, vendored seaborn fallback, supported dependency versions, and project platform matrix.
