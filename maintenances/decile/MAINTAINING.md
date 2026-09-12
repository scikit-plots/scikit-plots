# Maintaining `scikitplot.decile`

`scikitplot.decile` contains three user-visible surfaces that must be reviewed separately: the current top-level ModelPlotPy implementation, the legacy `modelplotpy` compatibility namespace, and the KDS adapter. A green test count in one surface is not evidence for the others.

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`. Keep maintenance health, runtime structural health, isolated-harness behavior, complete-package integration, and release readiness as separate truths.

The 2026-09-12 onboarding review found two local runtime contract issues: KDS drops documented class/positive-label/precision arguments at internal calls, and the legacy ModelPlotPy implementation mutates NumPy's global RNG state.
